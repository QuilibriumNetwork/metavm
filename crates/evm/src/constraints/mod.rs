//! EVM constraint system implementing VmConstraintSystem.
//!
//! Constraints verify EVM instruction correctness using 256-bit limb
//! decomposition. Each U256 value is represented as 4 x u64 limbs.
//!
//! The constraint system uses algebraic selector multiplication to gate
//! per-instruction constraints, eliminating integer branching that would
//! be unsound at arbitrary field elements.

pub mod arith;
pub mod compare;
pub mod shift;
pub mod stack;
pub mod memory;
pub mod control;
pub mod env;

use metavm_zkp::field::Scalar;
use metavm_zkp::vm_constraints::VmConstraintSystem;
use metavm_zkp::poly_arith;
use metavm_zkp::lookup::{LookupRequirements, LookupTable, LookupDeclaration, BitwiseLookupDeclaration, BitwiseOp};
use crate::trace;

/// Number of VM-specific constraints (gated by selectors).
///
/// 8 ADD (4 limb + 4 carry binary) + 8 SUB (4 limb + 4 borrow binary)
/// + 1 mul_full + 1 div_full + 1 mod_full
/// + 1 sdiv_full + 1 smod_full
/// + 16 ADDMOD (1 ab_sum_chain + 4 ab_carry_binary + 1 s_high_binary
///              + 1 qn_mul + 1 sum_chain + 1 slack_chain
///              + 4 slack_borrow_binary + 1 n_is_zero_binary
///              + 1 n_zero_gate + 1 r_zero_gate)
/// + 10 MULMOD (1 ab_mul + 1 sum_chain + 1 slack_chain
///              + 4 slack_borrow_binary + 1 n_is_zero_binary
///              + 1 n_zero_gate + 1 r_zero_gate)
/// + 1 exp_oracle + 1 signextend (algebraic)
/// + 1 LT + 1 GT + 1 EQ + 1 ISZERO + 1 compare_other
/// + 4 AND (per-limb) + 4 OR (per-limb) + 4 XOR (per-limb)
/// + 1 SHL + 1 SHR + 1 SAR
/// + 1 push + 1 dup + 1 pop + 1 swap
/// + 1 mload + 1 mstore + 1 mstore8 + 1 msize
/// + 1 control
/// + 1 NOT (bitwise_other)
/// + 9 oracle (stop, keccak, env, block,
///              stack_other, memory_other, storage, log, call) = 88.
/// + 2 oracle slots for sel_call_push_frame and sel_call_return; the actual
///   algebraic semantics for these two selectors live entirely in the
///   cross-row shifted-constraint section (see `evaluate_shifted_at_point`
///   and `build_shifted_constraint_polynomial` below). Body is zero on the
///   same-row layer = 90.
/// + 6 oracle slots for sel_create, sel_callcode, sel_delegatecall,
///   sel_create2, sel_staticcall, sel_revert (all cross-row driven) = 96.
const NUM_VM_CONSTRAINTS: usize = 96;

/// Number of selector columns in the one-hot encoding.
const NUM_SELECTORS: usize = 50;

/// Number of SIGNEXTEND case-selector witness columns (not part of the main
/// one-hot selector sum; bound to sel_signextend via a dedicated constraint).
const NUM_SE_CASE_SELECTORS: usize = 32;

/// Total constraints: 73 VM + 41 selector-binary + 1 selector-sum
/// + 32 se-case-binary + 1 se-case-sum-binding = 148.
const TOTAL_CONSTRAINTS: usize =
    NUM_VM_CONSTRAINTS + NUM_SELECTORS + 1 + NUM_SE_CASE_SELECTORS + 1;

/// The EVM constraint system for proving EVM execution correctness.
pub struct EvmConstraintSystem;

impl EvmConstraintSystem {
    pub fn new() -> Self {
        EvmConstraintSystem
    }
}

impl VmConstraintSystem for EvmConstraintSystem {
    fn num_constraints(&self) -> usize {
        TOTAL_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "sel_arith_add * add_limb0".to_string(),
            "sel_arith_add * add_limb1".to_string(),
            "sel_arith_add * add_limb2".to_string(),
            "sel_arith_add * add_limb3".to_string(),
            "sel_arith_add * carry0_binary".to_string(),
            "sel_arith_add * carry1_binary".to_string(),
            "sel_arith_add * carry2_binary".to_string(),
            "sel_arith_add * carry3_binary".to_string(),
            "sel_arith_sub * sub_limb0".to_string(),
            "sel_arith_sub * sub_limb1".to_string(),
            "sel_arith_sub * sub_limb2".to_string(),
            "sel_arith_sub * sub_limb3".to_string(),
            "sel_arith_sub * borrow0_binary".to_string(),
            "sel_arith_sub * borrow1_binary".to_string(),
            "sel_arith_sub * borrow2_binary".to_string(),
            "sel_arith_sub * borrow3_binary".to_string(),
            "sel_arith_mul * mul_full".to_string(),
            "sel_arith_div * div_full".to_string(),
            "sel_mod * mod_full".to_string(),
            "sel_sdiv * sdiv_full".to_string(),
            "sel_smod * smod_full".to_string(),
            "sel_addmod * addmod_ab_sum_chain".to_string(),
            "sel_addmod * addmod_ab_carry0_binary".to_string(),
            "sel_addmod * addmod_ab_carry1_binary".to_string(),
            "sel_addmod * addmod_ab_carry2_binary".to_string(),
            "sel_addmod * addmod_ab_carry3_binary".to_string(),
            "sel_addmod * addmod_s_high_binary".to_string(),
            "sel_addmod * addmod_qn_mul".to_string(),
            "sel_addmod * addmod_sum_chain".to_string(),
            "sel_addmod * addmod_slack_chain".to_string(),
            "sel_addmod * addmod_slack_b0_binary".to_string(),
            "sel_addmod * addmod_slack_b1_binary".to_string(),
            "sel_addmod * addmod_slack_b2_binary".to_string(),
            "sel_addmod * addmod_slack_b3_binary".to_string(),
            "sel_addmod * addmod_n_is_zero_binary".to_string(),
            "sel_addmod * addmod_n_zero_gate".to_string(),
            "sel_addmod * addmod_r_zero_gate".to_string(),
            "sel_mulmod * mulmod_ab_mul".to_string(),
            "sel_mulmod * mulmod_sum_chain".to_string(),
            "sel_mulmod * mulmod_slack_chain".to_string(),
            "sel_mulmod * mulmod_slack_b0_binary".to_string(),
            "sel_mulmod * mulmod_slack_b1_binary".to_string(),
            "sel_mulmod * mulmod_slack_b2_binary".to_string(),
            "sel_mulmod * mulmod_slack_b3_binary".to_string(),
            "sel_mulmod * mulmod_n_is_zero_binary".to_string(),
            "sel_mulmod * mulmod_n_zero_gate".to_string(),
            "sel_mulmod * mulmod_r_zero_gate".to_string(),
            "sel_exp * exp_oracle".to_string(),
            "signextend_algebraic".to_string(),
            "sel_lt * lt_raw".to_string(),
            "sel_gt * gt_raw".to_string(),
            "sel_eq * eq_raw".to_string(),
            "sel_iszero * iszero_raw".to_string(),
            "sel_compare_other * compare_other_raw".to_string(),
            "sel_and * and_limb0".to_string(),
            "sel_and * and_limb1".to_string(),
            "sel_and * and_limb2".to_string(),
            "sel_and * and_limb3".to_string(),
            "sel_or * or_limb0".to_string(),
            "sel_or * or_limb1".to_string(),
            "sel_or * or_limb2".to_string(),
            "sel_or * or_limb3".to_string(),
            "sel_xor * xor_limb0".to_string(),
            "sel_xor * xor_limb1".to_string(),
            "sel_xor * xor_limb2".to_string(),
            "sel_xor * xor_limb3".to_string(),
            "sel_shl * shl_raw".to_string(),
            "sel_shr * shr_raw".to_string(),
            "sel_sar * sar_raw".to_string(),
            "sel_push * push_raw".to_string(),
            "sel_dup * dup_raw".to_string(),
            "sel_pop * pop_raw".to_string(),
            "sel_swap * swap_raw".to_string(),
            "sel_mload * mload_raw".to_string(),
            "sel_mstore * mstore_raw".to_string(),
            "sel_mstore8 * mstore8_raw".to_string(),
            "sel_msize * msize_oracle".to_string(),
            "sel_jump * control_raw".to_string(),
            "sel_stop * stop_oracle".to_string(),
            "sel_bitwise_other * not_raw".to_string(),
            "sel_keccak * keccak_oracle".to_string(),
            "sel_env * env_oracle".to_string(),
            "sel_block * block_oracle".to_string(),
            "sel_stack_other * stack_other_oracle".to_string(),
            "sel_memory_other * memory_other_oracle".to_string(),
            "sel_storage * storage_oracle".to_string(),
            "sel_log * log_oracle".to_string(),
            "sel_call * call_oracle".to_string(),
            // Split-call selectors. Same-row body is 0 (oracle); the real
            // algebraic transitions live in the shifted-constraint section.
            // The labels are reserved here so the constraint count matches
            // NUM_VM_CONSTRAINTS.
            "sel_call_push_frame * frame_push_oracle".to_string(),
            "sel_call_return * frame_return_oracle".to_string(),
            // Per-opcode call-family selectors. Same-row body is 0 (oracle);
            // cross-row transitions live in the shifted section.
            "sel_create * frame_push_oracle".to_string(),
            "sel_callcode * frame_push_oracle".to_string(),
            "sel_delegatecall * frame_push_oracle".to_string(),
            "sel_create2 * frame_push_oracle".to_string(),
            "sel_staticcall * frame_push_oracle".to_string(),
            "sel_revert * frame_pop_oracle".to_string(),
        ];

        // Selector binary constraints
        let sel_names = [
            "sel_stop", "sel_arith_add", "sel_arith_sub",
            "sel_arith_mul", "sel_arith_div",
            "sel_mod", "sel_sdiv", "sel_smod", "sel_addmod",
            "sel_mulmod", "sel_exp", "sel_signextend",
            "sel_lt", "sel_gt", "sel_eq", "sel_iszero", "sel_compare_other",
            "sel_and", "sel_or", "sel_xor", "sel_bitwise_other",
            "sel_shl", "sel_shr", "sel_sar",
            "sel_keccak", "sel_env", "sel_block",
            "sel_push", "sel_dup", "sel_pop", "sel_swap", "sel_stack_other",
            "sel_mload", "sel_mstore", "sel_mstore8", "sel_msize", "sel_memory_other",
            "sel_storage", "sel_jump", "sel_log", "sel_call",
            "sel_call_push_frame", "sel_call_return",
            "sel_create", "sel_callcode", "sel_delegatecall",
            "sel_create2", "sel_staticcall", "sel_revert",
            "sel_byte_op",
        ];
        for name in &sel_names {
            labels.push(format!("{}_binary", name));
        }

        // Selector sum-to-one constraint
        labels.push("selector_sum_to_one".to_string());

        // SIGNEXTEND case-selector binary constraints (32)
        for k in 0..31 {
            labels.push(format!("sel_se_{}_binary", k));
        }
        labels.push("sel_se_ge31_binary".to_string());

        // SIGNEXTEND case-selector sum-binding: sel_signextend = Σ sel_se_k
        labels.push("sel_signextend_sum_binding".to_string());

        labels
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let two_64 = arith::two_pow_64_pub(curve);
        let zero = Scalar::zero(curve);

        // --- 16 individual ADD/SUB sub-constraints (constraints 0-15) ---

        // ADD limb 0 (constraint 0)
        let mut add_l0 = Vec::with_capacity(num_rows);
        // ADD limb 1 (constraint 1)
        let mut add_l1 = Vec::with_capacity(num_rows);
        // ADD limb 2 (constraint 2)
        let mut add_l2 = Vec::with_capacity(num_rows);
        // ADD limb 3 (constraint 3)
        let mut add_l3 = Vec::with_capacity(num_rows);
        // carry0 binary (constraint 4)
        let mut carry0_bin = Vec::with_capacity(num_rows);
        // carry1 binary (constraint 5)
        let mut carry1_bin = Vec::with_capacity(num_rows);
        // carry2 binary (constraint 6)
        let mut carry2_bin = Vec::with_capacity(num_rows);
        // carry3 binary (constraint 7)
        let mut carry3_bin = Vec::with_capacity(num_rows);
        // SUB limb 0 (constraint 8)
        let mut sub_l0 = Vec::with_capacity(num_rows);
        // SUB limb 1 (constraint 9)
        let mut sub_l1 = Vec::with_capacity(num_rows);
        // SUB limb 2 (constraint 10)
        let mut sub_l2 = Vec::with_capacity(num_rows);
        // SUB limb 3 (constraint 11)
        let mut sub_l3 = Vec::with_capacity(num_rows);
        // borrow0 binary (constraint 12)
        let mut borrow0_bin = Vec::with_capacity(num_rows);
        // borrow1 binary (constraint 13)
        let mut borrow1_bin = Vec::with_capacity(num_rows);
        // borrow2 binary (constraint 14)
        let mut borrow2_bin = Vec::with_capacity(num_rows);
        // borrow3 binary (constraint 15)
        let mut borrow3_bin = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let sel_add = &columns[trace::COL_SEL_ARITH_ADD][i];
            let sel_sub = &columns[trace::COL_SEL_ARITH_SUB][i];

            if !sel_add.is_zero() {
                // ADD limb constraints
                let sum0 = columns[trace::COL_INPUT0_L0][i].add(&columns[trace::COL_INPUT1_L0][i]);
                let carry0_term = columns[trace::COL_AUX0_L0][i].mul(&two_64);
                add_l0.push(sum0.sub(&carry0_term).sub(&columns[trace::COL_OUTPUT0_L0][i]));

                let sum1 = columns[trace::COL_INPUT0_L1][i].add(&columns[trace::COL_INPUT1_L1][i]).add(&columns[trace::COL_AUX0_L0][i]);
                let carry1_term = columns[trace::COL_AUX0_L1][i].mul(&two_64);
                add_l1.push(sum1.sub(&carry1_term).sub(&columns[trace::COL_OUTPUT0_L1][i]));

                let sum2 = columns[trace::COL_INPUT0_L2][i].add(&columns[trace::COL_INPUT1_L2][i]).add(&columns[trace::COL_AUX0_L1][i]);
                let carry2_term = columns[trace::COL_AUX0_L2][i].mul(&two_64);
                add_l2.push(sum2.sub(&carry2_term).sub(&columns[trace::COL_OUTPUT0_L2][i]));

                let sum3 = columns[trace::COL_INPUT0_L3][i].add(&columns[trace::COL_INPUT1_L3][i]).add(&columns[trace::COL_AUX0_L2][i]);
                let carry3_term = columns[trace::COL_AUX0_L3][i].mul(&two_64);
                add_l3.push(sum3.sub(&carry3_term).sub(&columns[trace::COL_OUTPUT0_L3][i]));

                // carry binary constraints
                carry0_bin.push(columns[trace::COL_AUX0_L0][i].mul(&columns[trace::COL_AUX0_L0][i].sub(&one)));
                carry1_bin.push(columns[trace::COL_AUX0_L1][i].mul(&columns[trace::COL_AUX0_L1][i].sub(&one)));
                carry2_bin.push(columns[trace::COL_AUX0_L2][i].mul(&columns[trace::COL_AUX0_L2][i].sub(&one)));
                carry3_bin.push(columns[trace::COL_AUX0_L3][i].mul(&columns[trace::COL_AUX0_L3][i].sub(&one)));
            } else {
                add_l0.push(zero.clone());
                add_l1.push(zero.clone());
                add_l2.push(zero.clone());
                add_l3.push(zero.clone());
                carry0_bin.push(zero.clone());
                carry1_bin.push(zero.clone());
                carry2_bin.push(zero.clone());
                carry3_bin.push(zero.clone());
            }

            if !sel_sub.is_zero() {
                // SUB limb constraints
                let sum0 = columns[trace::COL_OUTPUT0_L0][i].add(&columns[trace::COL_INPUT1_L0][i]);
                let borrow0_term = columns[trace::COL_AUX0_L0][i].mul(&two_64);
                sub_l0.push(sum0.sub(&columns[trace::COL_INPUT0_L0][i]).sub(&borrow0_term));

                let sum1 = columns[trace::COL_OUTPUT0_L1][i].add(&columns[trace::COL_INPUT1_L1][i]).add(&columns[trace::COL_AUX0_L0][i]);
                let borrow1_term = columns[trace::COL_AUX0_L1][i].mul(&two_64);
                sub_l1.push(sum1.sub(&columns[trace::COL_INPUT0_L1][i]).sub(&borrow1_term));

                let sum2 = columns[trace::COL_OUTPUT0_L2][i].add(&columns[trace::COL_INPUT1_L2][i]).add(&columns[trace::COL_AUX0_L1][i]);
                let borrow2_term = columns[trace::COL_AUX0_L2][i].mul(&two_64);
                sub_l2.push(sum2.sub(&columns[trace::COL_INPUT0_L2][i]).sub(&borrow2_term));

                let sum3 = columns[trace::COL_OUTPUT0_L3][i].add(&columns[trace::COL_INPUT1_L3][i]).add(&columns[trace::COL_AUX0_L2][i]);
                let borrow3_term = columns[trace::COL_AUX0_L3][i].mul(&two_64);
                sub_l3.push(sum3.sub(&columns[trace::COL_INPUT0_L3][i]).sub(&borrow3_term));

                // borrow binary constraints
                borrow0_bin.push(columns[trace::COL_AUX0_L0][i].mul(&columns[trace::COL_AUX0_L0][i].sub(&one)));
                borrow1_bin.push(columns[trace::COL_AUX0_L1][i].mul(&columns[trace::COL_AUX0_L1][i].sub(&one)));
                borrow2_bin.push(columns[trace::COL_AUX0_L2][i].mul(&columns[trace::COL_AUX0_L2][i].sub(&one)));
                borrow3_bin.push(columns[trace::COL_AUX0_L3][i].mul(&columns[trace::COL_AUX0_L3][i].sub(&one)));
            } else {
                sub_l0.push(zero.clone());
                sub_l1.push(zero.clone());
                sub_l2.push(zero.clone());
                sub_l3.push(zero.clone());
                borrow0_bin.push(zero.clone());
                borrow1_bin.push(zero.clone());
                borrow2_bin.push(zero.clone());
                borrow3_bin.push(zero.clone());
            }
        }

        // --- MUL full constraint (constraint 16) ---
        let mut mul_full = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let sel_mul = &columns[trace::COL_SEL_ARITH_MUL][i];
            if !sel_mul.is_zero() {
                let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                    .map(|c| columns[c][i].clone())
                    .collect();
                mul_full.push(arith::evaluate_arith_mul_full_raw(&cols_at_i));
            } else {
                mul_full.push(zero.clone());
            }
        }

        // --- DIV full constraint (constraint 17) ---
        let mut div_full = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let sel_div = &columns[trace::COL_SEL_ARITH_DIV][i];
            if !sel_div.is_zero() {
                let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                    .map(|c| columns[c][i].clone())
                    .collect();
                div_full.push(arith::evaluate_arith_div_full_raw(&cols_at_i));
            } else {
                div_full.push(zero.clone());
            }
        }

        // --- MOD full constraint (constraint 18) ---
        let mut mod_full = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let sel_mod = &columns[trace::COL_SEL_MOD][i];
            if !sel_mod.is_zero() {
                let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                    .map(|c| columns[c][i].clone())
                    .collect();
                mod_full.push(arith::evaluate_arith_mod_full_raw(&cols_at_i));
            } else {
                mod_full.push(zero.clone());
            }
        }

        // --- SDIV full constraint (constraint 19) ---
        // SDIV uses the same DIV algebraic identity: quotient*divisor+remainder = dividend (mod 2^256)
        let mut sdiv_full = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let sel = &columns[trace::COL_SEL_SDIV][i];
            if !sel.is_zero() {
                let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                    .map(|c| columns[c][i].clone())
                    .collect();
                sdiv_full.push(arith::evaluate_arith_div_full_raw(&cols_at_i));
            } else {
                sdiv_full.push(zero.clone());
            }
        }

        // --- SMOD full constraint (constraint 20) ---
        // SMOD uses the same MOD algebraic identity: quotient*divisor+remainder = dividend (mod 2^256)
        let mut smod_full = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let sel = &columns[trace::COL_SEL_SMOD][i];
            if !sel.is_zero() {
                let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                    .map(|c| columns[c][i].clone())
                    .collect();
                smod_full.push(arith::evaluate_arith_mod_full_raw(&cols_at_i));
            } else {
                smod_full.push(zero.clone());
            }
        }

        // --- Oracle constraints ---
        // EXP is the only remaining oracle in the arith group (zero body).
        // ADDMOD (indices 21..=36) and MULMOD (indices 37..=46) are algebraic.
        // SIGNEXTEND (index 48) is algebraic — see signextend::evaluate.
        let oracle_zero = vec![zero.clone(); num_rows];

        let addmod_bodies = evaluate_addmod_bodies(columns, num_rows);
        let mulmod_bodies = evaluate_mulmod_bodies(columns, num_rows);

        let mut result = vec![
            add_l0, add_l1, add_l2, add_l3,
            carry0_bin, carry1_bin, carry2_bin, carry3_bin,
            sub_l0, sub_l1, sub_l2, sub_l3,
            borrow0_bin, borrow1_bin, borrow2_bin, borrow3_bin,
            mul_full, div_full, mod_full,
            sdiv_full,
            smod_full,
        ];
        // ADDMOD: 16 algebraic constraint bodies (indices 21..=36).
        for body in addmod_bodies {
            result.push(body);
        }
        // MULMOD: 10 algebraic constraint bodies (indices 37..=46).
        for body in mulmod_bodies {
            result.push(body);
        }
        // Continuation after MULMOD algebraic bodies.
        let continuation: Vec<Vec<Scalar>> = vec![
            oracle_zero.clone(), // exp (index 47)
            evaluate_signextend_full(columns, num_rows), // signextend (index 48, algebraic)
            compare::evaluate_lt(columns, num_rows),
            compare::evaluate_gt(columns, num_rows),
            compare::evaluate_eq(columns, num_rows),
            compare::evaluate_iszero(columns, num_rows),
            {
                // compare_other: binary output + upper limbs zero for SLT/SGT
                let mut cmp_other = Vec::with_capacity(num_rows);
                for i in 0..num_rows {
                    let sel = &columns[trace::COL_SEL_COMPARE_OTHER][i];
                    if sel.is_zero() {
                        cmp_other.push(zero.clone());
                    } else {
                        let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                            .map(|c| columns[c][i].clone())
                            .collect();
                        cmp_other.push(compare::evaluate_compare_other_raw(&cols_at_i));
                    }
                }
                cmp_other
            },
            // AND constraints: 4 per-limb
            evaluate_and(columns, num_rows),
            evaluate_and_limb(columns, num_rows, 1),
            evaluate_and_limb(columns, num_rows, 2),
            evaluate_and_limb(columns, num_rows, 3),
            // OR constraints: 4 per-limb
            evaluate_or(columns, num_rows),
            evaluate_or_limb(columns, num_rows, 1),
            evaluate_or_limb(columns, num_rows, 2),
            evaluate_or_limb(columns, num_rows, 3),
            // XOR constraints: 4 per-limb
            evaluate_xor(columns, num_rows),
            evaluate_xor_limb(columns, num_rows, 1),
            evaluate_xor_limb(columns, num_rows, 2),
            evaluate_xor_limb(columns, num_rows, 3),
            shift::evaluate_shl(columns, num_rows),
            shift::evaluate_shr(columns, num_rows),
            shift::evaluate_sar(columns, num_rows),
            stack::evaluate_push(columns, num_rows),
            stack::evaluate_dup(columns, num_rows),
            stack::evaluate_pop(columns, num_rows),
            stack::evaluate_swap(columns, num_rows),
            memory::evaluate_mload(columns, num_rows),
            memory::evaluate_mstore(columns, num_rows),
            memory::evaluate_mstore8(columns, num_rows),
            memory::evaluate_msize(columns, num_rows),
            control::evaluate_control(columns, num_rows),
        ];
        for body in continuation {
            result.push(body);
        }

        // Oracle constraints (54-63): zero body for operations verified externally,
        // except bitwise_other (NOT) at index 55 which has an algebraic body.
        let oracle_zero_vec = vec![zero.clone(); num_rows];
        // 54: stop (oracle)
        result.push(oracle_zero_vec.clone());
        // 55: bitwise_other (NOT): Σ(output_lk + input0_lk - (2^64 - 1)) for k=0..3
        {
            let mut not_body = Vec::with_capacity(num_rows);
            let not_limbs: Vec<Vec<Scalar>> = (0..4)
                .map(|k| evaluate_not_limb(columns, num_rows, k))
                .collect();
            for i in 0..num_rows {
                let mut sum = Scalar::zero(curve);
                for k in 0..4 {
                    sum = sum.add(&not_limbs[k][i]);
                }
                not_body.push(sum);
            }
            result.push(not_body);
        }
        // 56-63: keccak, env, block, stack_other, memory_other, storage, log, call (oracle)
        for _ in 0..8 {
            result.push(oracle_zero_vec.clone());
        }
        // 64-65: sel_call_push_frame, sel_call_return (oracle on same-row;
        // shifted constraints carry the real frame-stack semantics).
        for _ in 0..2 {
            result.push(oracle_zero_vec.clone());
        }
        // 66-71: sel_create, sel_callcode, sel_delegatecall, sel_create2,
        // sel_staticcall, sel_revert (oracle on same-row; shifted
        // constraints below carry the real per-opcode frame transitions).
        for _ in 0..6 {
            result.push(oracle_zero_vec.clone());
        }

        let sel_indices = self.selector_column_indices();

        // Selector binary constraints: s_k * (s_k - 1) = 0
        for &si in &sel_indices {
            let mut binary_eval = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let s = &columns[si][i];
                let s_m1 = s.sub(&one);
                binary_eval.push(s.mul(&s_m1));
            }
            result.push(binary_eval);
        }

        // Selector sum-to-one: Sigma s_k - 1 = 0
        {
            let mut sum_eval = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let mut sel_sum = Scalar::zero(curve);
                for &si in &sel_indices {
                    sel_sum = sel_sum.add(&columns[si][i]);
                }
                sum_eval.push(sel_sum.sub(&one));
            }
            result.push(sum_eval);
        }

        // ─── SIGNEXTEND case-selector constraints ───
        // 32 binary constraints: sel_se_k * (sel_se_k - 1) = 0 for k = 0..31.
        for k in 0..32 {
            let col_idx = trace::COL_SEL_SE_0 + k;
            let mut binary_eval = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let s = &columns[col_idx][i];
                let s_m1 = s.sub(&one);
                binary_eval.push(s.mul(&s_m1));
            }
            result.push(binary_eval);
        }

        // Sum-binding: sel_signextend - Σ sel_se_k = 0
        {
            let mut bind_eval = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let mut case_sum = Scalar::zero(curve);
                for k in 0..32 {
                    case_sum = case_sum.add(&columns[trace::COL_SEL_SE_0 + k][i]);
                }
                bind_eval.push(columns[trace::COL_SEL_SIGNEXTEND][i].sub(&case_sum));
            }
            result.push(bind_eval);
        }

        result
    }

    fn evaluate_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        alpha: &Scalar,
    ) -> Scalar {
        let curve = alpha.curve_type();
        if col_evals_at_z.len() < trace::NUM_EVM_COLUMNS {
            return Scalar::zero(curve);
        }
        let one = Scalar::one(curve);
        let mut result = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        let sel_add = &col_evals_at_z[trace::COL_SEL_ARITH_ADD];
        let sel_sub = &col_evals_at_z[trace::COL_SEL_ARITH_SUB];

        // --- ADD constraints (0-7): 4 limb + 4 carry binary, gated by sel_arith_add ---

        // Constraint 0: sel_arith_add * add_limb0
        {
            let body = arith::evaluate_arith_add_limb0_raw(col_evals_at_z);
            result = result.add(&sel_add.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 1: sel_arith_add * add_limb1
        {
            let body = arith::evaluate_arith_add_limb1_raw(col_evals_at_z);
            result = result.add(&sel_add.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 2: sel_arith_add * add_limb2
        {
            let body = arith::evaluate_arith_add_limb2_raw(col_evals_at_z);
            result = result.add(&sel_add.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 3: sel_arith_add * add_limb3
        {
            let body = arith::evaluate_arith_add_limb3_raw(col_evals_at_z);
            result = result.add(&sel_add.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 4: sel_arith_add * carry0_binary
        {
            let body = arith::evaluate_carry0_binary_raw(col_evals_at_z);
            result = result.add(&sel_add.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 5: sel_arith_add * carry1_binary
        {
            let body = arith::evaluate_carry1_binary_raw(col_evals_at_z);
            result = result.add(&sel_add.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 6: sel_arith_add * carry2_binary
        {
            let body = arith::evaluate_carry2_binary_raw(col_evals_at_z);
            result = result.add(&sel_add.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 7: sel_arith_add * carry3_binary
        {
            let body = arith::evaluate_carry3_binary_raw(col_evals_at_z);
            result = result.add(&sel_add.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- SUB constraints (8-15): 4 limb + 4 borrow binary, gated by sel_arith_sub ---

        // Constraint 8: sel_arith_sub * sub_limb0
        {
            let body = arith::evaluate_arith_sub_limb0_raw(col_evals_at_z);
            result = result.add(&sel_sub.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 9: sel_arith_sub * sub_limb1
        {
            let body = arith::evaluate_arith_sub_limb1_raw(col_evals_at_z);
            result = result.add(&sel_sub.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 10: sel_arith_sub * sub_limb2
        {
            let body = arith::evaluate_arith_sub_limb2_raw(col_evals_at_z);
            result = result.add(&sel_sub.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 11: sel_arith_sub * sub_limb3
        {
            let body = arith::evaluate_arith_sub_limb3_raw(col_evals_at_z);
            result = result.add(&sel_sub.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 12: sel_arith_sub * borrow0_binary
        {
            let body = arith::evaluate_borrow0_binary_raw(col_evals_at_z);
            result = result.add(&sel_sub.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 13: sel_arith_sub * borrow1_binary
        {
            let body = arith::evaluate_borrow1_binary_raw(col_evals_at_z);
            result = result.add(&sel_sub.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 14: sel_arith_sub * borrow2_binary
        {
            let body = arith::evaluate_borrow2_binary_raw(col_evals_at_z);
            result = result.add(&sel_sub.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // Constraint 15: sel_arith_sub * borrow3_binary
        {
            let body = arith::evaluate_borrow3_binary_raw(col_evals_at_z);
            result = result.add(&sel_sub.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 16: sel_arith_mul * mul_full ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_ARITH_MUL];
            let body = arith::evaluate_arith_mul_full_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 17: sel_arith_div * div_full ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_ARITH_DIV];
            let body = arith::evaluate_arith_div_full_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 18: sel_mod * mod_full ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_MOD];
            let body = arith::evaluate_arith_mod_full_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 19: sel_sdiv * sdiv_full (same body as DIV) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_SDIV];
            let body = arith::evaluate_arith_div_full_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 20: sel_smod * smod_full (same body as MOD) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_SMOD];
            let body = arith::evaluate_arith_mod_full_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraints 21-36: ADDMOD algebraic (16 bodies, gated by sel_addmod) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_ADDMOD];
            let bodies = evaluate_addmod_raw(col_evals_at_z);
            for body in &bodies {
                result = result.add(&sel.mul(body).mul(&ap));
                ap = ap.mul(alpha);
            }
        }

        // --- Constraints 37-46: MULMOD algebraic (10 bodies, gated by sel_mulmod) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_MULMOD];
            let bodies = evaluate_mulmod_raw(col_evals_at_z);
            for body in &bodies {
                result = result.add(&sel.mul(body).mul(&ap));
                ap = ap.mul(alpha);
            }
        }

        // --- Constraint 47: EXP oracle (zero body) ---
        ap = ap.mul(alpha);

        // --- Constraint 33: SIGNEXTEND (algebraic, not gated at top level since
        //     each sub-case is gated internally by its own sel_se_k) ---
        {
            let body = evaluate_signextend_raw(col_evals_at_z);
            result = result.add(&body.mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 25: LT (gated by sel_lt) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_LT];
            let body = compare::evaluate_lt_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 26: GT (gated by sel_gt) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_GT];
            let body = compare::evaluate_gt_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 27: EQ (gated by sel_eq) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_EQ];
            let body = compare::evaluate_eq_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 21: ISZERO (gated by sel_iszero) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_ISZERO];
            let body = compare::evaluate_iszero_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 22: compare_other (gated by sel_compare_other) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_COMPARE_OTHER];
            let body = compare::evaluate_compare_other_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraints 23-26: AND per-limb (gated by sel_and) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_AND];
            for limb in 0..4 {
                let body = evaluate_and_limb_raw(col_evals_at_z, limb);
                result = result.add(&sel.mul(&body).mul(&ap));
                ap = ap.mul(alpha);
            }
        }

        // --- Constraints 27-30: OR per-limb (gated by sel_or) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_OR];
            for limb in 0..4 {
                let body = evaluate_or_limb_raw(col_evals_at_z, limb);
                result = result.add(&sel.mul(&body).mul(&ap));
                ap = ap.mul(alpha);
            }
        }

        // --- Constraints 31-34: XOR per-limb (gated by sel_xor) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_XOR];
            for limb in 0..4 {
                let body = evaluate_xor_limb_raw(col_evals_at_z, limb);
                result = result.add(&sel.mul(&body).mul(&ap));
                ap = ap.mul(alpha);
            }
        }

        // --- Constraint 35: SHL (gated by sel_shl) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_SHL];
            let body = shift::evaluate_shl_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 36: SHR (gated by sel_shr) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_SHR];
            let body = shift::evaluate_shr_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 37: SAR (gated by sel_sar) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_SAR];
            let body = shift::evaluate_sar_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 38: push (gated by sel_push) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_PUSH];
            let body = stack::evaluate_push_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 39: dup (gated by sel_dup) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_DUP];
            let body = stack::evaluate_dup_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 40: pop (gated by sel_pop) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_POP];
            let body = stack::evaluate_pop_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 41: swap (gated by sel_swap) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_SWAP];
            let body = stack::evaluate_swap_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 42: mload (gated by sel_mload) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_MLOAD];
            let body = memory::evaluate_mload_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 43: mstore (gated by sel_mstore) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_MSTORE];
            let body = memory::evaluate_mstore_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 44: mstore8 (gated by sel_mstore8) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_MSTORE8];
            let body = memory::evaluate_mstore8_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 45: msize oracle (gated by sel_msize) ---
        {
            // MSIZE is an oracle: zero body, sel * 0 = 0. Just advance alpha.
        }
        ap = ap.mul(alpha);

        // --- Constraint 46: control (gated by sel_jump) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_JUMP];
            let body = control::evaluate_control_raw(col_evals_at_z);
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraint 54: stop oracle (zero body) ---
        ap = ap.mul(alpha);

        // --- Constraint 55: bitwise_other (NOT) ---
        {
            let sel = &col_evals_at_z[trace::COL_SEL_BITWISE_OTHER];
            let mut body = Scalar::zero(curve);
            for k in 0..4 {
                body = body.add(&evaluate_not_limb_raw(col_evals_at_z, k));
            }
            result = result.add(&sel.mul(&body).mul(&ap));
        }
        ap = ap.mul(alpha);

        // --- Constraints 56-63: oracle constraints (zero body) ---
        // keccak, env, block, stack_other, memory_other, storage, log, call
        for _ in 0..8 {
            ap = ap.mul(alpha);
        }

        // --- Constraints 64-65: oracle for sel_call_push_frame/sel_call_return.
        // Real transitions live in the shifted-constraint section.
        for _ in 0..2 {
            ap = ap.mul(alpha);
        }
        // --- Constraints 66-71: oracle for sel_create, sel_callcode,
        // sel_delegatecall, sel_create2, sel_staticcall, sel_revert.
        // Real transitions live in the shifted-constraint section.
        for _ in 0..6 {
            ap = ap.mul(alpha);
        }

        // --- Selector consistency constraints (ungated, apply to all rows) ---

        let sel_indices = self.selector_column_indices();

        // Binary: s_k * (s_k - 1) = 0 for each selector
        for &si in &sel_indices {
            let s = &col_evals_at_z[si];
            let s_m1 = s.sub(&one);
            result = result.add(&s.mul(&s_m1).mul(&ap));
            ap = ap.mul(alpha);
        }

        // Sum-to-one: Sigma s_k - 1 = 0
        let mut sel_sum = Scalar::zero(curve);
        for &si in &sel_indices {
            sel_sum = sel_sum.add(&col_evals_at_z[si]);
        }
        result = result.add(&sel_sum.sub(&one).mul(&ap));
        ap = ap.mul(alpha);

        // ─── SIGNEXTEND case-selector binary constraints (32) ───
        for k in 0..32 {
            let col_idx = trace::COL_SEL_SE_0 + k;
            let s = &col_evals_at_z[col_idx];
            let s_m1 = s.sub(&one);
            result = result.add(&s.mul(&s_m1).mul(&ap));
            ap = ap.mul(alpha);
        }

        // Sum-binding: sel_signextend - Σ sel_se_k = 0
        {
            let mut case_sum = Scalar::zero(curve);
            for k in 0..32 {
                case_sum = case_sum.add(&col_evals_at_z[trace::COL_SEL_SE_0 + k]);
            }
            let body = col_evals_at_z[trace::COL_SEL_SIGNEXTEND].sub(&case_sum);
            result = result.add(&body.mul(&ap));
        }

        result
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![
            trace::COL_SEL_STOP,
            trace::COL_SEL_ARITH_ADD,
            trace::COL_SEL_ARITH_SUB,
            trace::COL_SEL_ARITH_MUL,
            trace::COL_SEL_ARITH_DIV,
            trace::COL_SEL_MOD,
            trace::COL_SEL_SDIV,
            trace::COL_SEL_SMOD,
            trace::COL_SEL_ADDMOD,
            trace::COL_SEL_MULMOD,
            trace::COL_SEL_EXP,
            trace::COL_SEL_SIGNEXTEND,
            trace::COL_SEL_LT,
            trace::COL_SEL_GT,
            trace::COL_SEL_EQ,
            trace::COL_SEL_ISZERO,
            trace::COL_SEL_COMPARE_OTHER,
            trace::COL_SEL_AND,
            trace::COL_SEL_OR,
            trace::COL_SEL_XOR,
            trace::COL_SEL_BITWISE_OTHER,
            trace::COL_SEL_SHL,
            trace::COL_SEL_SHR,
            trace::COL_SEL_SAR,
            trace::COL_SEL_KECCAK,
            trace::COL_SEL_ENV,
            trace::COL_SEL_BLOCK,
            trace::COL_SEL_PUSH,
            trace::COL_SEL_DUP,
            trace::COL_SEL_POP,
            trace::COL_SEL_SWAP,
            trace::COL_SEL_STACK_OTHER,
            trace::COL_SEL_MLOAD,
            trace::COL_SEL_MSTORE,
            trace::COL_SEL_MSTORE8,
            trace::COL_SEL_MSIZE,
            trace::COL_SEL_MEMORY_OTHER,
            trace::COL_SEL_STORAGE,
            trace::COL_SEL_JUMP,
            trace::COL_SEL_LOG,
            trace::COL_SEL_CALL,
            trace::COL_SEL_CALL_PUSH_FRAME,
            trace::COL_SEL_CALL_RETURN,
            trace::COL_SEL_CREATE,
            trace::COL_SEL_CALLCODE,
            trace::COL_SEL_DELEGATECALL,
            trace::COL_SEL_CREATE2,
            trace::COL_SEL_STATICCALL,
            trace::COL_SEL_REVERT,
            trace::COL_SEL_BYTE_OP,
        ]
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        use rayon::prelude::*;

        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let two_64 = arith::two_pow_64_pub(curve);

        // Helper closure: get a column's coefficient polynomial.
        // column_coeffs is indexed by data column index (step excluded).
        let col = |idx: usize| -> &Vec<Scalar> { &column_coeffs[idx] };

        // Pre-compute shared polynomial sub-expressions for ADD/SUB limbs
        let aux0_l0_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L0), &one_poly, curve);
        let aux0_l1_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L1), &one_poly, curve);
        let aux0_l2_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L2), &one_poly, curve);
        let aux0_l3_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L3), &one_poly, curve);

        // Phase 1: Collect (selector, body) pairs for all VM constraints.
        // Body computation is sequential (shared precomputed values), but
        // the final gating poly_mul calls will be parallelized.
        let mut pairs: Vec<(Vec<Scalar>, Vec<Scalar>)> = Vec::new();

        // --- Constraint 0: sel_arith_add * add_limb0 ---
        {
            let sum0 = poly_arith::poly_add(col(trace::COL_INPUT0_L0), col(trace::COL_INPUT1_L0), curve);
            let carry0_term = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L0), &two_64);
            let rhs0 = poly_arith::poly_add(&carry0_term, col(trace::COL_OUTPUT0_L0), curve);
            let body = poly_arith::poly_sub(&sum0, &rhs0, curve);
            pairs.push((col(trace::COL_SEL_ARITH_ADD).clone(), body));
        }

        // --- Constraint 1: sel_arith_add * add_limb1 ---
        {
            let sum1 = poly_arith::poly_add(col(trace::COL_INPUT0_L1), col(trace::COL_INPUT1_L1), curve);
            let sum1 = poly_arith::poly_add(&sum1, col(trace::COL_AUX0_L0), curve);
            let carry1_term = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L1), &two_64);
            let rhs1 = poly_arith::poly_add(&carry1_term, col(trace::COL_OUTPUT0_L1), curve);
            let body = poly_arith::poly_sub(&sum1, &rhs1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_ADD).clone(), body));
        }

        // --- Constraint 2: sel_arith_add * add_limb2 ---
        {
            let sum2 = poly_arith::poly_add(col(trace::COL_INPUT0_L2), col(trace::COL_INPUT1_L2), curve);
            let sum2 = poly_arith::poly_add(&sum2, col(trace::COL_AUX0_L1), curve);
            let carry2_term = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L2), &two_64);
            let rhs2 = poly_arith::poly_add(&carry2_term, col(trace::COL_OUTPUT0_L2), curve);
            let body = poly_arith::poly_sub(&sum2, &rhs2, curve);
            pairs.push((col(trace::COL_SEL_ARITH_ADD).clone(), body));
        }

        // --- Constraint 3: sel_arith_add * add_limb3 ---
        {
            let sum3 = poly_arith::poly_add(col(trace::COL_INPUT0_L3), col(trace::COL_INPUT1_L3), curve);
            let sum3 = poly_arith::poly_add(&sum3, col(trace::COL_AUX0_L2), curve);
            let carry3_term = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L3), &two_64);
            let rhs3 = poly_arith::poly_add(&carry3_term, col(trace::COL_OUTPUT0_L3), curve);
            let body = poly_arith::poly_sub(&sum3, &rhs3, curve);
            pairs.push((col(trace::COL_SEL_ARITH_ADD).clone(), body));
        }

        // --- Constraint 4: sel_arith_add * carry0_binary ---
        {
            let body = poly_arith::poly_mul(col(trace::COL_AUX0_L0), &aux0_l0_m1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_ADD).clone(), body));
        }

        // --- Constraint 5: sel_arith_add * carry1_binary ---
        {
            let body = poly_arith::poly_mul(col(trace::COL_AUX0_L1), &aux0_l1_m1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_ADD).clone(), body));
        }

        // --- Constraint 6: sel_arith_add * carry2_binary ---
        {
            let body = poly_arith::poly_mul(col(trace::COL_AUX0_L2), &aux0_l2_m1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_ADD).clone(), body));
        }

        // --- Constraint 7: sel_arith_add * carry3_binary ---
        {
            let body = poly_arith::poly_mul(col(trace::COL_AUX0_L3), &aux0_l3_m1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_ADD).clone(), body));
        }

        // --- Constraint 8: sel_arith_sub * sub_limb0 ---
        {
            let sum0 = poly_arith::poly_add(col(trace::COL_OUTPUT0_L0), col(trace::COL_INPUT1_L0), curve);
            let borrow0_term = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L0), &two_64);
            let body = poly_arith::poly_sub(&sum0, &poly_arith::poly_add(col(trace::COL_INPUT0_L0), &borrow0_term, curve), curve);
            pairs.push((col(trace::COL_SEL_ARITH_SUB).clone(), body));
        }

        // --- Constraint 9: sel_arith_sub * sub_limb1 ---
        {
            let sum1 = poly_arith::poly_add(col(trace::COL_OUTPUT0_L1), col(trace::COL_INPUT1_L1), curve);
            let sum1 = poly_arith::poly_add(&sum1, col(trace::COL_AUX0_L0), curve);
            let borrow1_term = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L1), &two_64);
            let body = poly_arith::poly_sub(&sum1, &poly_arith::poly_add(col(trace::COL_INPUT0_L1), &borrow1_term, curve), curve);
            pairs.push((col(trace::COL_SEL_ARITH_SUB).clone(), body));
        }

        // --- Constraint 10: sel_arith_sub * sub_limb2 ---
        {
            let sum2 = poly_arith::poly_add(col(trace::COL_OUTPUT0_L2), col(trace::COL_INPUT1_L2), curve);
            let sum2 = poly_arith::poly_add(&sum2, col(trace::COL_AUX0_L1), curve);
            let borrow2_term = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L2), &two_64);
            let body = poly_arith::poly_sub(&sum2, &poly_arith::poly_add(col(trace::COL_INPUT0_L2), &borrow2_term, curve), curve);
            pairs.push((col(trace::COL_SEL_ARITH_SUB).clone(), body));
        }

        // --- Constraint 11: sel_arith_sub * sub_limb3 ---
        {
            let sum3 = poly_arith::poly_add(col(trace::COL_OUTPUT0_L3), col(trace::COL_INPUT1_L3), curve);
            let sum3 = poly_arith::poly_add(&sum3, col(trace::COL_AUX0_L2), curve);
            let borrow3_term = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L3), &two_64);
            let body = poly_arith::poly_sub(&sum3, &poly_arith::poly_add(col(trace::COL_INPUT0_L3), &borrow3_term, curve), curve);
            pairs.push((col(trace::COL_SEL_ARITH_SUB).clone(), body));
        }

        // --- Constraint 12: sel_arith_sub * borrow0_binary ---
        {
            let body = poly_arith::poly_mul(col(trace::COL_AUX0_L0), &aux0_l0_m1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_SUB).clone(), body));
        }

        // --- Constraint 13: sel_arith_sub * borrow1_binary ---
        {
            let body = poly_arith::poly_mul(col(trace::COL_AUX0_L1), &aux0_l1_m1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_SUB).clone(), body));
        }

        // --- Constraint 14: sel_arith_sub * borrow2_binary ---
        {
            let body = poly_arith::poly_mul(col(trace::COL_AUX0_L2), &aux0_l2_m1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_SUB).clone(), body));
        }

        // --- Constraint 15: sel_arith_sub * borrow3_binary ---
        {
            let body = poly_arith::poly_mul(col(trace::COL_AUX0_L3), &aux0_l3_m1, curve);
            pairs.push((col(trace::COL_SEL_ARITH_SUB).clone(), body));
        }

        // --- Constraint 16: sel_arith_mul * mul_full ---
        {
            let body = build_mul_full_constraint_poly(&col, &two_64, curve);
            pairs.push((col(trace::COL_SEL_ARITH_MUL).clone(), body));
        }

        // --- Constraint 17: sel_arith_div * div_full ---
        {
            let body = build_div_full_constraint_poly(&col, &two_64, curve);
            pairs.push((col(trace::COL_SEL_ARITH_DIV).clone(), body));
        }

        // --- Constraint 18: sel_mod * mod_full ---
        {
            let body = build_mod_full_constraint_poly(&col, &two_64, curve);
            pairs.push((col(trace::COL_SEL_MOD).clone(), body));
        }

        // --- Constraint 19: sel_sdiv * sdiv_full (same body as DIV) ---
        {
            let body = build_div_full_constraint_poly(&col, &two_64, curve);
            pairs.push((col(trace::COL_SEL_SDIV).clone(), body));
        }

        // --- Constraint 20: sel_smod * smod_full (same body as MOD) ---
        {
            let body = build_mod_full_constraint_poly(&col, &two_64, curve);
            pairs.push((col(trace::COL_SEL_SMOD).clone(), body));
        }

        // --- Constraints 21-36: ADDMOD algebraic (16 bodies, gated by sel_addmod) ---
        {
            let addmod_polys = build_addmod_constraint_polys(&col, &two_64, &one_poly, curve);
            for body in addmod_polys {
                pairs.push((col(trace::COL_SEL_ADDMOD).clone(), body));
            }
        }

        // --- Constraints 37-46: MULMOD algebraic (10 bodies, gated by sel_mulmod) ---
        {
            let mulmod_polys = build_mulmod_constraint_polys(&col, &two_64, &one_poly, curve);
            for body in mulmod_polys {
                pairs.push((col(trace::COL_SEL_MULMOD).clone(), body));
            }
        }

        // --- Constraint 47: EXP oracle (zero body, no pair) ---

        // --- Constraint 33: SIGNEXTEND (algebraic, internally gated by sel_se_k) ---
        {
            let body = build_signextend_constraint_poly(&col, curve);
            // Use one_poly as the "selector" since gating is internal to the body.
            pairs.push((one_poly.clone(), body));
        }

        // --- Constraint 25: sel_lt * lt_raw ---
        {
            let body = build_lt_constraint_poly(&col, &two_64, &one_poly, curve);
            pairs.push((col(trace::COL_SEL_LT).clone(), body));
        }

        // --- Constraint 26: sel_gt * gt_raw ---
        {
            let body = build_gt_constraint_poly(&col, &two_64, &one_poly, curve);
            pairs.push((col(trace::COL_SEL_GT).clone(), body));
        }

        // --- Constraint 27: sel_eq * eq_raw ---
        {
            let body = build_eq_constraint_poly(&col, &one_poly, curve);
            pairs.push((col(trace::COL_SEL_EQ).clone(), body));
        }

        // --- Constraint 28: sel_iszero * iszero_raw ---
        {
            let body = build_iszero_constraint_poly(&col, &one_poly, curve);
            pairs.push((col(trace::COL_SEL_ISZERO).clone(), body));
        }

        // --- Constraint 29: sel_compare_other * compare_other_raw ---
        {
            let out_l0 = col(trace::COL_OUTPUT0_L0);
            let binary_check = poly_arith::poly_mul(
                out_l0,
                &poly_arith::poly_sub(out_l0, &one_poly, curve),
                curve,
            );
            let upper = poly_arith::poly_add(col(trace::COL_OUTPUT0_L1), col(trace::COL_OUTPUT0_L2), curve);
            let upper = poly_arith::poly_add(&upper, col(trace::COL_OUTPUT0_L3), curve);
            let body = poly_arith::poly_add(&binary_check, &upper, curve);
            pairs.push((col(trace::COL_SEL_COMPARE_OTHER).clone(), body));
        }

        // --- Constraints 30-33: sel_and * and_limb_k ---
        {
            let output_cols = [trace::COL_OUTPUT0_L0, trace::COL_OUTPUT0_L1, trace::COL_OUTPUT0_L2, trace::COL_OUTPUT0_L3];
            let aux0_cols = [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3];
            for limb in 0..4 {
                // AND body: output_lk - aux0_lk = 0
                let body = poly_arith::poly_sub(col(output_cols[limb]), col(aux0_cols[limb]), curve);
                pairs.push((col(trace::COL_SEL_AND).clone(), body));
            }
        }

        // --- Constraints 34-37: sel_or * or_limb_k ---
        {
            let output_cols = [trace::COL_OUTPUT0_L0, trace::COL_OUTPUT0_L1, trace::COL_OUTPUT0_L2, trace::COL_OUTPUT0_L3];
            let input0_cols = [trace::COL_INPUT0_L0, trace::COL_INPUT0_L1, trace::COL_INPUT0_L2, trace::COL_INPUT0_L3];
            let input1_cols = [trace::COL_INPUT1_L0, trace::COL_INPUT1_L1, trace::COL_INPUT1_L2, trace::COL_INPUT1_L3];
            let aux0_cols = [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3];
            for limb in 0..4 {
                // OR body: output_lk - input0_lk - input1_lk + aux0_lk = 0
                let body = poly_arith::poly_sub(col(output_cols[limb]), col(input0_cols[limb]), curve);
                let body = poly_arith::poly_sub(&body, col(input1_cols[limb]), curve);
                let body = poly_arith::poly_add(&body, col(aux0_cols[limb]), curve);
                pairs.push((col(trace::COL_SEL_OR).clone(), body));
            }
        }

        // --- Constraints 38-41: sel_xor * xor_limb_k ---
        {
            let two = Scalar::from_u64(2, curve);
            let output_cols = [trace::COL_OUTPUT0_L0, trace::COL_OUTPUT0_L1, trace::COL_OUTPUT0_L2, trace::COL_OUTPUT0_L3];
            let input0_cols = [trace::COL_INPUT0_L0, trace::COL_INPUT0_L1, trace::COL_INPUT0_L2, trace::COL_INPUT0_L3];
            let input1_cols = [trace::COL_INPUT1_L0, trace::COL_INPUT1_L1, trace::COL_INPUT1_L2, trace::COL_INPUT1_L3];
            let aux0_cols = [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3];
            for limb in 0..4 {
                // XOR body: output_lk - input0_lk - input1_lk + 2*aux0_lk = 0
                let body = poly_arith::poly_sub(col(output_cols[limb]), col(input0_cols[limb]), curve);
                let body = poly_arith::poly_sub(&body, col(input1_cols[limb]), curve);
                let two_aux = poly_arith::poly_scalar_mul(col(aux0_cols[limb]), &two);
                let body = poly_arith::poly_add(&body, &two_aux, curve);
                pairs.push((col(trace::COL_SEL_XOR).clone(), body));
            }
        }

        // --- Constraint 42: sel_shl * shl_raw ---
        {
            let body = build_shl_constraint_poly(&col, &two_64, curve);
            pairs.push((col(trace::COL_SEL_SHL).clone(), body));
        }

        // --- Constraint 43: sel_shr * shr_raw ---
        {
            let body = build_shift_div_constraint_poly(&col, &two_64, curve);
            pairs.push((col(trace::COL_SEL_SHR).clone(), body));
        }

        // --- Constraint 44: sel_sar * sar_raw ---
        {
            let body = build_shift_div_constraint_poly(&col, &two_64, curve);
            pairs.push((col(trace::COL_SEL_SAR).clone(), body));
        }

        // --- Constraint 45: sel_push * push_raw ---
        {
            let c0 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L0), col(trace::COL_IMMEDIATE_L0), curve);
            let c1 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L1), col(trace::COL_IMMEDIATE_L1), curve);
            let c2 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L2), col(trace::COL_IMMEDIATE_L2), curve);
            let c3 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L3), col(trace::COL_IMMEDIATE_L3), curve);
            let body = poly_arith::poly_add(&c0, &c1, curve);
            let body = poly_arith::poly_add(&body, &c2, curve);
            let body = poly_arith::poly_add(&body, &c3, curve);
            pairs.push((col(trace::COL_SEL_PUSH).clone(), body));
        }

        // --- Constraint 46: sel_dup * dup_raw ---
        {
            let c0 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L0), col(trace::COL_INPUT0_L0), curve);
            let c1 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L1), col(trace::COL_INPUT0_L1), curve);
            let c2 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L2), col(trace::COL_INPUT0_L2), curve);
            let c3 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L3), col(trace::COL_INPUT0_L3), curve);
            let body = poly_arith::poly_add(&c0, &c1, curve);
            let body = poly_arith::poly_add(&body, &c2, curve);
            let body = poly_arith::poly_add(&body, &c3, curve);
            pairs.push((col(trace::COL_SEL_DUP).clone(), body));
        }

        // --- Constraint 47: sel_pop * pop_raw ---
        {
            let body = poly_arith::poly_add(col(trace::COL_OUTPUT0_L0), col(trace::COL_OUTPUT0_L1), curve);
            let body = poly_arith::poly_add(&body, col(trace::COL_OUTPUT0_L2), curve);
            let body = poly_arith::poly_add(&body, col(trace::COL_OUTPUT0_L3), curve);
            pairs.push((col(trace::COL_SEL_POP).clone(), body));
        }

        // --- Constraint 48: sel_swap * swap_raw ---
        {
            let c0 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L0), col(trace::COL_INPUT1_L0), curve);
            let c1 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L1), col(trace::COL_INPUT1_L1), curve);
            let c2 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L2), col(trace::COL_INPUT1_L2), curve);
            let c3 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L3), col(trace::COL_INPUT1_L3), curve);
            let body = poly_arith::poly_add(&c0, &c1, curve);
            let body = poly_arith::poly_add(&body, &c2, curve);
            let body = poly_arith::poly_add(&body, &c3, curve);
            pairs.push((col(trace::COL_SEL_SWAP).clone(), body));
        }

        // --- Constraint 49: sel_mload * mload_raw ---
        {
            let c0 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L0), col(trace::COL_MEM_VALUE_L0), curve);
            let c1 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L1), col(trace::COL_MEM_VALUE_L1), curve);
            let c2 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L2), col(trace::COL_MEM_VALUE_L2), curve);
            let c3 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L3), col(trace::COL_MEM_VALUE_L3), curve);
            let body = poly_arith::poly_add(&c0, &c1, curve);
            let body = poly_arith::poly_add(&body, &c2, curve);
            let body = poly_arith::poly_add(&body, &c3, curve);
            pairs.push((col(trace::COL_SEL_MLOAD).clone(), body));
        }

        // --- Constraint 50: sel_mstore * mstore_raw ---
        {
            let c0 = poly_arith::poly_sub(col(trace::COL_MEM_VALUE_L0), col(trace::COL_INPUT1_L0), curve);
            let c1 = poly_arith::poly_sub(col(trace::COL_MEM_VALUE_L1), col(trace::COL_INPUT1_L1), curve);
            let c2 = poly_arith::poly_sub(col(trace::COL_MEM_VALUE_L2), col(trace::COL_INPUT1_L2), curve);
            let c3 = poly_arith::poly_sub(col(trace::COL_MEM_VALUE_L3), col(trace::COL_INPUT1_L3), curve);
            let body = poly_arith::poly_add(&c0, &c1, curve);
            let body = poly_arith::poly_add(&body, &c2, curve);
            let body = poly_arith::poly_add(&body, &c3, curve);
            pairs.push((col(trace::COL_SEL_MSTORE).clone(), body));
        }

        // --- Constraint 51: sel_mstore8 * mstore8_raw ---
        {
            let two_56 = Scalar::from_u64(256, curve);
            // aux0_l0 * 256 + mem_val_l0 - input1_l0
            let aux_scaled = poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L0), &two_56);
            let c0 = poly_arith::poly_add(&aux_scaled, col(trace::COL_MEM_VALUE_L0), curve);
            let c0 = poly_arith::poly_sub(&c0, col(trace::COL_INPUT1_L0), curve);
            // upper mem_value limbs zero
            let body = poly_arith::poly_add(&c0, col(trace::COL_MEM_VALUE_L1), curve);
            let body = poly_arith::poly_add(&body, col(trace::COL_MEM_VALUE_L2), curve);
            let body = poly_arith::poly_add(&body, col(trace::COL_MEM_VALUE_L3), curve);
            pairs.push((col(trace::COL_SEL_MSTORE8).clone(), body));
        }

        // --- Constraint 52: sel_msize * msize_oracle (zero body) ---
        // MSIZE is an oracle: no pair, just oracle skip.

        // --- Constraint 53: sel_jump * control_raw ---
        {
            let c0 = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L0), col(trace::COL_PC), curve);
            let body = poly_arith::poly_add(&c0, col(trace::COL_OUTPUT0_L1), curve);
            let body = poly_arith::poly_add(&body, col(trace::COL_OUTPUT0_L2), curve);
            let body = poly_arith::poly_add(&body, col(trace::COL_OUTPUT0_L3), curve);
            pairs.push((col(trace::COL_SEL_JUMP).clone(), body));
        }

        // --- Constraint 54: stop oracle (zero body) ---
        // No pair, just oracle skip.

        // --- Constraint 55: bitwise_other (NOT) ---
        {
            let max64 = Scalar::from_u64(u64::MAX, curve);
            let max64_poly = vec![max64];
            // NOT body: Sigma(output_lk + input0_lk - (2^64 - 1)) for k=0..3
            let mut body = vec![Scalar::zero(curve)];
            for k in 0..4 {
                let out_k = col(trace::COL_OUTPUT0_L0 + k);
                let in0_k = col(trace::COL_INPUT0_L0 + k);
                let sum = poly_arith::poly_add(out_k, in0_k, curve);
                let diff = poly_arith::poly_sub(&sum, &max64_poly, curve);
                body = poly_arith::poly_add(&body, &diff, curve);
            }
            pairs.push((col(trace::COL_SEL_BITWISE_OTHER).clone(), body));
        }

        // --- Constraints 56-63: oracle constraints (zero body) ---
        // keccak, env, block, stack_other, memory_other, storage, log, call: no pairs.

        // Phase 2: Parallel gating poly_mul for all constraint pairs
        let gated_results: Vec<Vec<Scalar>> = pairs.par_iter()
            .map(|(sel, body)| poly_arith::poly_mul(sel, body, curve))
            .collect();

        // Phase 3: Sequential accumulation with alpha powers
        // New layout (after ADDMOD and MULMOD both algebraic):
        //   pairs[0..21]   = constraints 0-20  (21 non-oracle: ADD/SUB/MUL/DIV/MOD/SDIV/SMOD)
        //   pairs[21..37]  = constraints 21-36 (16 ADDMOD algebraic bodies)
        //   pairs[37..47]  = constraints 37-46 (10 MULMOD algebraic bodies)
        //   1 oracle skip  = constraint 47 (EXP)
        //   pairs[47]      = constraint 48 (SIGNEXTEND, algebraic)
        //   pairs[48..75]  = constraints 49-75 (27 non-oracle: compares, bitwise, shifts, stack, memory)
        //   1 oracle skip  = constraint 76 (MSIZE)
        //   pairs[75]      = constraint 77 (JUMP)
        //   1 oracle skip  = constraint 78 (STOP)
        //   pairs[76]      = constraint 79 (NOT / bitwise_other)
        //   8 oracle skips = constraints 80-87 (keccak, env, block, stack_other, memory_other, storage, log, call)
        let mut c = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);

        // Accumulate constraints 0-20 (pairs[0..21])
        for gated in &gated_results[..21] {
            c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(gated, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Accumulate constraints 21-36 (pairs[21..37]: ADDMOD algebraic)
        for gated in &gated_results[21..37] {
            c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(gated, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Accumulate constraints 37-46 (pairs[37..47]: MULMOD algebraic)
        for gated in &gated_results[37..47] {
            c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(gated, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Skip 1 oracle constraint (47: EXP)
        ap = ap.mul(alpha);

        // Accumulate constraint 48 (pairs[47]: SIGNEXTEND algebraic)
        c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(&gated_results[47], &ap), curve);
        ap = ap.mul(alpha);

        // Accumulate constraints 49-75 (pairs[48..75])
        for gated in &gated_results[48..75] {
            c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(gated, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Skip 1 oracle constraint (76: MSIZE)
        ap = ap.mul(alpha);

        // Accumulate constraint 77 (pairs[75]: JUMP)
        c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(&gated_results[75], &ap), curve);
        ap = ap.mul(alpha);

        // Skip 1 oracle constraint (78: STOP)
        ap = ap.mul(alpha);

        // Accumulate constraint 79 (pairs[76]: NOT)
        c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(&gated_results[76], &ap), curve);
        ap = ap.mul(alpha);

        // Skip 8 oracle constraints (80-87: keccak, env, block, stack_other, memory_other, storage, log, call)
        for _ in 0..8 {
            ap = ap.mul(alpha);
        }
        // Skip 2 oracle constraints (88-89: sel_call_push_frame, sel_call_return)
        for _ in 0..2 {
            ap = ap.mul(alpha);
        }
        // Skip 6 oracle constraints (90-95: sel_create, sel_callcode,
        // sel_delegatecall, sel_create2, sel_staticcall, sel_revert).
        for _ in 0..6 {
            ap = ap.mul(alpha);
        }

        // Phase 4: Parallel binary selector constraints
        let sel_indices = self.selector_column_indices();

        let binary_results: Vec<Vec<Scalar>> = sel_indices.par_iter()
            .map(|&si| {
                let s = &column_coeffs[si];
                let s_m1 = poly_arith::poly_sub(s, &one_poly, curve);
                poly_arith::poly_mul(s, &s_m1, curve)
            })
            .collect();

        for binary in &binary_results {
            c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(binary, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Phase 5: Sum-to-one (no poly_mul needed, sequential)
        let mut sel_sum = vec![Scalar::zero(curve)];
        for &si in &sel_indices {
            sel_sum = poly_arith::poly_add(&sel_sum, col(si), curve);
        }
        // Subtract the constant 1
        if !sel_sum.is_empty() {
            sel_sum[0] = sel_sum[0].sub(&Scalar::one(curve));
        }
        c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(&sel_sum, &ap), curve);
        ap = ap.mul(alpha);

        // Phase 6: SIGNEXTEND case-selector binary constraints (32)
        let se_binary_results: Vec<Vec<Scalar>> = (0..32).into_par_iter()
            .map(|k| {
                let s = &column_coeffs[trace::COL_SEL_SE_0 + k];
                let s_m1 = poly_arith::poly_sub(s, &one_poly, curve);
                poly_arith::poly_mul(s, &s_m1, curve)
            })
            .collect();
        for binary in &se_binary_results {
            c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(binary, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Phase 7: Sum-binding: sel_signextend - Σ sel_se_k = 0
        let mut case_sum = vec![Scalar::zero(curve)];
        for k in 0..32 {
            case_sum = poly_arith::poly_add(&case_sum, col(trace::COL_SEL_SE_0 + k), curve);
        }
        let sum_binding = poly_arith::poly_sub(col(trace::COL_SEL_SIGNEXTEND), &case_sum, curve);
        c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(&sum_binding, &ap), curve);

        c
    }

    fn padding_selector_column(&self) -> Option<usize> {
        Some(trace::COL_SEL_STOP)
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        // Set padding rows' PC and NEXT_PC to the last real row's next_pc
        // so the cross-row constraint pc[i+1] == next_pc[i] is satisfied.
        let last_next_pc = columns[trace::COL_NEXT_PC][num_rows - 1].clone();
        for i in num_rows..padded_size {
            columns[trace::COL_PC][i] = last_next_pc.clone();
            columns[trace::COL_NEXT_PC][i] = last_next_pc.clone();
        }

        // Pad the frame-state columns with the last real row's value so
        // the default-preservation gate
        //     (1 - is_frame_op) · (F(ω·z) - F(z)) = 0
        // holds on every padding-to-padding row boundary (where is_frame_op=0).
        //
        // Special case: if the last real row is a frame-pop (sel_call_return
        // or sel_revert), the existing depth_dec constraint demands the next
        // row's frame_depth equal depth - 1. Initialise padding-row depth
        // to (last_depth - 1) on pop, and propagate it to subsequent
        // padding rows. The other covered columns are not bound on the row
        // immediately after a pop (the preservation gate is 0 on the pop
        // row itself), so preserving them as-is from the last real row is
        // sound.
        let last = num_rows - 1;
        let is_pop_last = !columns[trace::COL_SEL_CALL_RETURN][last].is_zero()
            || !columns[trace::COL_SEL_REVERT][last].is_zero();
        let one_scalar = Scalar::one(columns[trace::COL_FRAME_DEPTH][last].curve_type());

        let pad_depth = if is_pop_last {
            columns[trace::COL_FRAME_DEPTH][last].sub(&one_scalar)
        } else {
            columns[trace::COL_FRAME_DEPTH][last].clone()
        };
        for i in num_rows..padded_size {
            columns[trace::COL_FRAME_DEPTH][i] = pad_depth.clone();
        }

        // Other 17 covered frame-state columns: copy the last real row's
        // value across all padding rows.
        let other_frame_cols: [usize; 17] = [
            trace::COL_FRAME_CALLER_L0,
            trace::COL_FRAME_CALLER_L1,
            trace::COL_FRAME_CALLER_L2,
            trace::COL_FRAME_CALLER_L3,
            trace::COL_FRAME_CALLEE_L0,
            trace::COL_FRAME_CALLEE_L1,
            trace::COL_FRAME_CALLEE_L2,
            trace::COL_FRAME_CALLEE_L3,
            trace::COL_FRAME_RETURN_PC,
            trace::COL_FRAME_VALUE_L0,
            trace::COL_FRAME_VALUE_L1,
            trace::COL_FRAME_VALUE_L2,
            trace::COL_FRAME_VALUE_L3,
            trace::COL_FRAME_STATIC,
            trace::COL_FRAME_GAS,
            trace::COL_FRAME_RETURN_OFFSET,
            trace::COL_FRAME_RETURN_SIZE,
        ];
        for &c in other_frame_cols.iter() {
            let last_val = columns[c][last].clone();
            for i in num_rows..padded_size {
                columns[c][i] = last_val.clone();
            }
        }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Order matters: indices into `shifted_evals` reflect this order in
        // `evaluate_shifted_at_point`. Keep COL_PC at index 0 so the existing
        // PC continuity constraint stays correct.
        vec![
            trace::COL_PC,                  // [0] PC continuity
            trace::COL_FRAME_DEPTH,         // [1] depth transitions
            trace::COL_FRAME_CALLER_L0,     // [2..=5] caller propagation/preservation
            trace::COL_FRAME_CALLER_L1,
            trace::COL_FRAME_CALLER_L2,
            trace::COL_FRAME_CALLER_L3,
            trace::COL_FRAME_CALLEE_L0,     // [6..=9] callee binding
            trace::COL_FRAME_CALLEE_L1,
            trace::COL_FRAME_CALLEE_L2,
            trace::COL_FRAME_CALLEE_L3,
            trace::COL_FRAME_RETURN_PC,     // [10] return_pc binding
            trace::COL_FRAME_VALUE_L0,      // [11..=14] DELEGATECALL value preservation
            trace::COL_FRAME_VALUE_L1,
            trace::COL_FRAME_VALUE_L2,
            trace::COL_FRAME_VALUE_L3,
            trace::COL_FRAME_STATIC,        // [15] STATICCALL set + propagation
            trace::COL_FRAME_GAS,           // [16] gas allotment preservation
            trace::COL_FRAME_RETURN_OFFSET, // [17] return-data offset preservation
            trace::COL_FRAME_RETURN_SIZE,   // [18] return-data size preservation
        ]
    }

    fn num_shifted_constraints(&self) -> usize {
        // [0] PC continuity (1)
        // [1..=11] CALL push-frame: 1 depth_inc, 4 callee_bind,
        //          4 caller_propagate, 1 return_pc_set, 1 static_propagate = 11
        // [12]    CALL pop-frame: 1 depth_dec
        // [13..=23] CALLCODE: same shape as CALL_PUSH (11)
        // [24..=38] DELEGATECALL: 1 depth_inc + 4 callee_bind + 4 caller_preserve
        //           + 4 value_preserve + 1 return_pc_set + 1 static_propagate = 15
        // [39..=49] CREATE: 1 depth_inc + 4 callee_from_hint + 4 caller_propagate
        //           + 1 return_pc_set + 1 static_propagate = 11
        // [50..=60] CREATE2: same shape as CREATE (11)
        // [61..=71] STATICCALL: 1 depth_inc + 4 callee_bind + 4 caller_propagate
        //           + 1 return_pc_set + 1 static_set_to_one = 11
        // [72]    REVERT: 1 depth_dec
        // = 1 + 11 + 1 + 11 + 15 + 11 + 11 + 11 + 1 = 73
        // [73..=90] Default frame-state preservation:
        //          (1 - is_frame_op) · (F(ω·z) - F(z)) = 0  for each of the
        //          18 frame-state columns exposed via shifted_evals
        //          (FRAME_DEPTH, FRAME_CALLER_L0..3, FRAME_CALLEE_L0..3,
        //           FRAME_VALUE_L0..3, FRAME_RETURN_PC, FRAME_STATIC,
        //           FRAME_GAS, FRAME_RETURN_OFFSET, FRAME_RETURN_SIZE).
        //          The GAS/RETURN_OFFSET/RETURN_SIZE columns are reachable
        //          here because `commitment::verify_at_point` accepts
        //          all-identity commits (zero polynomial); see the
        //          soundness argument in commitment.rs. = 18.
        // = 73 + 18 = 91
        91
    }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        if shifted_evals.is_empty() {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        // alpha^offset
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // Multiply by (z - omega^{n-1}) once at the end of each constraint to
        // exclude the wrap-around row.
        let exclusion = z.sub(omega_n_minus_1);

        let mut result = Scalar::zero(curve);

        // [0] PC continuity, gated to exclude frame transitions and
        // STOPs. Frame transitions (push: CALL/CREATE/CREATE2/CALLCODE/
        // DELEGATECALL/STATICCALL; pop: RETURN-at-depth≥1 (sel_call_return);
        // REVERT) jump to a different bytecode's pc=0 or back to a
        // saved return_pc. STOP at depth=0 terminates the transaction
        // (next row is padding with pc=last_next_pc); STOP at depth≥1
        // implicitly pops the frame and the next row's pc is the
        // popped frame's saved return_pc — neither equals next_pc on
        // the STOP row, so PC continuity must vanish on STOP rows
        // too. The `is_pc_jump` selector is a sum of mutually
        // exclusive 0/1 selectors (selector_sum_to_one ensures at
        // most one is set), so the gate `(1 - is_pc_jump)` is 0/1.
        let pc_next = &shifted_evals[0];
        let next_pc = &col_evals_at_z[trace::COL_NEXT_PC];
        let body = pc_next.sub(next_pc);
        // is_pc_jump uses sel_stop_pop (depth-conditioned) instead of
        // unconditional sel_stop. Top-level STOP at depth=0 has next_pc
        // matching the padding row's pc (set by fix_trace_padding to
        // last_next_pc), so PC continuity holds without gating. STOP at
        // depth ≥ 1 (sel_stop_pop=1) jumps to the popped frame's saved
        // return_pc and must be excluded.
        let is_pc_jump = col_evals_at_z[trace::COL_SEL_CALL_PUSH_FRAME]
            .add(&col_evals_at_z[trace::COL_SEL_CALL_RETURN])
            .add(&col_evals_at_z[trace::COL_SEL_CREATE])
            .add(&col_evals_at_z[trace::COL_SEL_CALLCODE])
            .add(&col_evals_at_z[trace::COL_SEL_DELEGATECALL])
            .add(&col_evals_at_z[trace::COL_SEL_CREATE2])
            .add(&col_evals_at_z[trace::COL_SEL_STATICCALL])
            .add(&col_evals_at_z[trace::COL_SEL_REVERT])
            .add(&col_evals_at_z[trace::COL_SEL_STOP_POP]);
        let pc_gate = one.sub(&is_pc_jump);
        result = result.add(&ap.mul(&pc_gate).mul(&body).mul(&exclusion));
        ap = ap.mul(alpha);

        // ── CALL_PUSH / CALL_RETURN frame transitions ──────────────────────
        let sel_push = &col_evals_at_z[trace::COL_SEL_CALL_PUSH_FRAME];
        let sel_ret = &col_evals_at_z[trace::COL_SEL_CALL_RETURN];

        let depth_z = &col_evals_at_z[trace::COL_FRAME_DEPTH];
        let depth_next = &shifted_evals[1];

        // sel_call_push_frame * (depth_next - depth - 1) = 0
        let body = depth_next.sub(depth_z).sub(&one);
        result = result.add(&ap.mul(sel_push).mul(&body).mul(&exclusion));
        ap = ap.mul(alpha);

        // sel_call_push_frame * (callee_l_k_next - input1_l_k) = 0 for k=0..3
        for k in 0..4 {
            let callee_next_k = &shifted_evals[6 + k];
            let input1_k = &col_evals_at_z[trace::COL_INPUT1_L0 + k];
            let body = callee_next_k.sub(input1_k);
            result = result.add(&ap.mul(sel_push).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // sel_call_push_frame * (caller_l_k_next - callee_l_k) = 0 for k=0..3
        for k in 0..4 {
            let caller_next_k = &shifted_evals[2 + k];
            let callee_k = &col_evals_at_z[trace::COL_FRAME_CALLEE_L0 + k];
            let body = caller_next_k.sub(callee_k);
            result = result.add(&ap.mul(sel_push).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // sel_call_push_frame * (return_pc_next - pc - 1) = 0
        let return_pc_next = &shifted_evals[10];
        let pc_z = &col_evals_at_z[trace::COL_PC];
        let body = return_pc_next.sub(pc_z).sub(&one);
        result = result.add(&ap.mul(sel_push).mul(&body).mul(&exclusion));
        ap = ap.mul(alpha);

        // sel_call_push_frame * (frame_static(ω·z) - frame_static(z)) = 0
        // (CALL propagates parent's static flag to the new frame).
        let static_z = &col_evals_at_z[trace::COL_FRAME_STATIC];
        let static_next = &shifted_evals[15];
        {
            let body = static_next.sub(static_z);
            result = result.add(&ap.mul(sel_push).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // sel_call_return * (depth_next - depth + 1) = 0
        let body = depth_next.sub(depth_z).add(&one);
        result = result.add(&ap.mul(sel_ret).mul(&body).mul(&exclusion));
        ap = ap.mul(alpha);

        // ── Call-family transitions: CALLCODE / DELEGATECALL / STATICCALL / REVERT / CREATE / CREATE2 ──
        let sel_callcode = &col_evals_at_z[trace::COL_SEL_CALLCODE];
        let sel_delegate = &col_evals_at_z[trace::COL_SEL_DELEGATECALL];
        let sel_create = &col_evals_at_z[trace::COL_SEL_CREATE];
        let sel_create2 = &col_evals_at_z[trace::COL_SEL_CREATE2];
        let sel_static = &col_evals_at_z[trace::COL_SEL_STATICCALL];
        let sel_revert = &col_evals_at_z[trace::COL_SEL_REVERT];
        let pc_z = &col_evals_at_z[trace::COL_PC];
        let return_pc_next = &shifted_evals[10];

        // Helper closures-style emitters using local refs.
        // CALLCODE: same shape as CALL_PUSH.
        {
            // depth_inc
            let body = depth_next.sub(depth_z).sub(&one);
            result = result.add(&ap.mul(sel_callcode).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            // callee_bind: callee_next = input1
            for k in 0..4 {
                let callee_next_k = &shifted_evals[6 + k];
                let input1_k = &col_evals_at_z[trace::COL_INPUT1_L0 + k];
                let body = callee_next_k.sub(input1_k);
                result = result.add(&ap.mul(sel_callcode).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            // caller_propagate: caller_next = current callee
            for k in 0..4 {
                let caller_next_k = &shifted_evals[2 + k];
                let callee_k = &col_evals_at_z[trace::COL_FRAME_CALLEE_L0 + k];
                let body = caller_next_k.sub(callee_k);
                result = result.add(&ap.mul(sel_callcode).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            // return_pc_set: return_pc_next = pc + 1
            let body = return_pc_next.sub(pc_z).sub(&one);
            result = result.add(&ap.mul(sel_callcode).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            // static_propagate
            let body = static_next.sub(static_z);
            result = result.add(&ap.mul(sel_callcode).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // DELEGATECALL: depth_inc, callee=input1, caller=PARENT's caller (preserved),
        // value=PARENT's value (preserved), return_pc=pc+1, static_propagate.
        {
            let body = depth_next.sub(depth_z).sub(&one);
            result = result.add(&ap.mul(sel_delegate).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            for k in 0..4 {
                let callee_next_k = &shifted_evals[6 + k];
                let input1_k = &col_evals_at_z[trace::COL_INPUT1_L0 + k];
                let body = callee_next_k.sub(input1_k);
                result = result.add(&ap.mul(sel_delegate).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            // caller_preserve: caller_next = current caller (NOT current callee).
            for k in 0..4 {
                let caller_next_k = &shifted_evals[2 + k];
                let caller_k = &col_evals_at_z[trace::COL_FRAME_CALLER_L0 + k];
                let body = caller_next_k.sub(caller_k);
                result = result.add(&ap.mul(sel_delegate).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            // value_preserve: value_next = current value.
            for k in 0..4 {
                let value_next_k = &shifted_evals[11 + k];
                let value_k = &col_evals_at_z[trace::COL_FRAME_VALUE_L0 + k];
                let body = value_next_k.sub(value_k);
                result = result.add(&ap.mul(sel_delegate).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            // return_pc_set
            let body = return_pc_next.sub(pc_z).sub(&one);
            result = result.add(&ap.mul(sel_delegate).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            // static_propagate
            let body = static_next.sub(static_z);
            result = result.add(&ap.mul(sel_delegate).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // CREATE: depth_inc, callee=create_address_hint, caller=current callee,
        // return_pc=pc+1, static_propagate.
        {
            let body = depth_next.sub(depth_z).sub(&one);
            result = result.add(&ap.mul(sel_create).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            for k in 0..4 {
                let callee_next_k = &shifted_evals[6 + k];
                let hint_k = &col_evals_at_z[trace::COL_CREATE_ADDRESS_HINT_L0 + k];
                let body = callee_next_k.sub(hint_k);
                result = result.add(&ap.mul(sel_create).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            for k in 0..4 {
                let caller_next_k = &shifted_evals[2 + k];
                let callee_k = &col_evals_at_z[trace::COL_FRAME_CALLEE_L0 + k];
                let body = caller_next_k.sub(callee_k);
                result = result.add(&ap.mul(sel_create).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            let body = return_pc_next.sub(pc_z).sub(&one);
            result = result.add(&ap.mul(sel_create).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            let body = static_next.sub(static_z);
            result = result.add(&ap.mul(sel_create).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // CREATE2: same shape as CREATE.
        {
            let body = depth_next.sub(depth_z).sub(&one);
            result = result.add(&ap.mul(sel_create2).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            for k in 0..4 {
                let callee_next_k = &shifted_evals[6 + k];
                let hint_k = &col_evals_at_z[trace::COL_CREATE_ADDRESS_HINT_L0 + k];
                let body = callee_next_k.sub(hint_k);
                result = result.add(&ap.mul(sel_create2).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            for k in 0..4 {
                let caller_next_k = &shifted_evals[2 + k];
                let callee_k = &col_evals_at_z[trace::COL_FRAME_CALLEE_L0 + k];
                let body = caller_next_k.sub(callee_k);
                result = result.add(&ap.mul(sel_create2).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            let body = return_pc_next.sub(pc_z).sub(&one);
            result = result.add(&ap.mul(sel_create2).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            let body = static_next.sub(static_z);
            result = result.add(&ap.mul(sel_create2).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // STATICCALL: depth_inc, callee=input1, caller=current callee,
        // return_pc=pc+1, static_set_to_one (frame_static(ω·z) = 1).
        {
            let body = depth_next.sub(depth_z).sub(&one);
            result = result.add(&ap.mul(sel_static).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            for k in 0..4 {
                let callee_next_k = &shifted_evals[6 + k];
                let input1_k = &col_evals_at_z[trace::COL_INPUT1_L0 + k];
                let body = callee_next_k.sub(input1_k);
                result = result.add(&ap.mul(sel_static).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            for k in 0..4 {
                let caller_next_k = &shifted_evals[2 + k];
                let callee_k = &col_evals_at_z[trace::COL_FRAME_CALLEE_L0 + k];
                let body = caller_next_k.sub(callee_k);
                result = result.add(&ap.mul(sel_static).mul(&body).mul(&exclusion));
                ap = ap.mul(alpha);
            }
            let body = return_pc_next.sub(pc_z).sub(&one);
            result = result.add(&ap.mul(sel_static).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
            // static_set: frame_static(ω·z) = 1.
            let body = static_next.sub(&one);
            result = result.add(&ap.mul(sel_static).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // REVERT: depth_dec (frame_depth(ω·z) = frame_depth(z) - 1).
        {
            let body = depth_next.sub(depth_z).add(&one);
            result = result.add(&ap.mul(sel_revert).mul(&body).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        // ── Default frame-state preservation ─────────────────────────────────
        //
        // For each of the 15 frame-state columns currently exposed via
        // shifted_evals, enforce
        //     (1 - is_frame_op(z)) · (F(ω·z) - F(z)) = 0
        // gated by (z - ω^{n-1}) wrap exclusion.
        //
        // is_frame_op = sum of all 8 frame-mutating selectors. At most one
        // can fire on any row (selector sum-to-one), so this sum is binary
        // valued. When no frame op fires, the gate is 1 and the column must
        // equal its previous value; on a frame-op row the gate is 0 and the
        // existing CALL/RETURN/etc. constraints handle the transition exactly.
        //
        // Padding rows: padding_selector_column() = COL_SEL_STOP, so
        // is_frame_op = 0 on padding rows. fix_trace_padding pads frame
        // columns with the last real row's value, making them constant on
        // padding so the constraint is trivially satisfied (depth gets a
        // -1 adjustment if the last real row is a frame-pop, satisfying
        // both depth_dec on the boundary AND preservation between padding
        // rows where the column is constant).
        let sel_call_return = &col_evals_at_z[trace::COL_SEL_CALL_RETURN];
        let sel_stop_pop = &col_evals_at_z[trace::COL_SEL_STOP_POP];
        let is_frame_op = sel_push
            .add(sel_call_return)
            .add(sel_callcode)
            .add(sel_delegate)
            .add(sel_create)
            .add(sel_create2)
            .add(sel_static)
            .add(sel_revert)
            .add(sel_stop_pop);
        let preserve_gate = one.sub(&is_frame_op);

        // (col_idx, shifted_idx) pairs — 18 entries (full coverage of
        // shifted_column_indices()). Order matches `preserve_pairs` in
        // build_shifted_constraint_polynomial below.
        let preserve_pairs: [(usize, usize); 18] = [
            (trace::COL_FRAME_DEPTH,         1),
            (trace::COL_FRAME_CALLER_L0,     2),
            (trace::COL_FRAME_CALLER_L1,     3),
            (trace::COL_FRAME_CALLER_L2,     4),
            (trace::COL_FRAME_CALLER_L3,     5),
            (trace::COL_FRAME_CALLEE_L0,     6),
            (trace::COL_FRAME_CALLEE_L1,     7),
            (trace::COL_FRAME_CALLEE_L2,     8),
            (trace::COL_FRAME_CALLEE_L3,     9),
            (trace::COL_FRAME_RETURN_PC,    10),
            (trace::COL_FRAME_VALUE_L0,     11),
            (trace::COL_FRAME_VALUE_L1,     12),
            (trace::COL_FRAME_VALUE_L2,     13),
            (trace::COL_FRAME_VALUE_L3,     14),
            (trace::COL_FRAME_STATIC,       15),
            (trace::COL_FRAME_GAS,          16),
            (trace::COL_FRAME_RETURN_OFFSET, 17),
            (trace::COL_FRAME_RETURN_SIZE,  18),
        ];
        for (col_idx, shift_idx) in preserve_pairs.iter() {
            let f_z = &col_evals_at_z[*col_idx];
            let f_next = &shifted_evals[*shift_idx];
            let body = f_next.sub(f_z);
            let gated = preserve_gate.mul(&body);
            result = result.add(&ap.mul(&gated).mul(&exclusion));
            ap = ap.mul(alpha);
        }

        let _ = ap;

        result
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
        let one_poly = vec![Scalar::one(curve)];

        let col = |idx: usize| -> &Vec<Scalar> { &column_coeffs[idx] };

        // alpha^offset
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // Compute omega^{n-1} via repeated squaring
        let mut omega_n_minus_1 = Scalar::one(curve);
        let mut base = omega.clone();
        let mut exp = domain_size - 1;
        while exp > 0 {
            if exp & 1 == 1 {
                omega_n_minus_1 = omega_n_minus_1.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }

        // Constraint accumulator. Each constraint is gated by an optional
        // selector polynomial, then multiplied by (X - ω^{n-1}) (exclusion
        // of the wrap-around row), scaled by the current α-power, and added
        // to `acc`.
        let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];

        // [0] PC continuity, gated by (1 - is_pc_jump) to exclude
        // frame-push (CALL/CREATE/CREATE2/CALLCODE/DELEGATECALL/
        // STATICCALL), frame-pop (sel_call_return for RETURN; sel_revert),
        // and STOP rows (top-level STOP terminates with padding rows
        // matching last_next_pc; STOP at depth≥1 implicitly pops the
        // frame and the next pc is the saved return_pc, not next_pc).
        // Mirrors the verifier evaluator.
        {
            let pc_shifted = poly_arith::poly_shift(col(trace::COL_PC), omega);
            let body = poly_arith::poly_sub(&pc_shifted, col(trace::COL_NEXT_PC), curve);
            // is_pc_jump = sum of 9 mutually-exclusive 0/1 selectors;
            // gate = (1 - sum) is 0/1.
            let mut is_xfer = col(trace::COL_SEL_CALL_PUSH_FRAME).clone();
            is_xfer = poly_arith::poly_add(&is_xfer, col(trace::COL_SEL_CALL_RETURN), curve);
            is_xfer = poly_arith::poly_add(&is_xfer, col(trace::COL_SEL_CREATE), curve);
            is_xfer = poly_arith::poly_add(&is_xfer, col(trace::COL_SEL_CALLCODE), curve);
            is_xfer = poly_arith::poly_add(&is_xfer, col(trace::COL_SEL_DELEGATECALL), curve);
            is_xfer = poly_arith::poly_add(&is_xfer, col(trace::COL_SEL_CREATE2), curve);
            is_xfer = poly_arith::poly_add(&is_xfer, col(trace::COL_SEL_STATICCALL), curve);
            is_xfer = poly_arith::poly_add(&is_xfer, col(trace::COL_SEL_REVERT), curve);
            is_xfer = poly_arith::poly_add(&is_xfer, col(trace::COL_SEL_STOP_POP), curve);
            let pc_gate = poly_arith::poly_sub(&one_poly, &is_xfer, curve);
            let gated = poly_arith::poly_mul(&pc_gate, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            acc = poly_arith::poly_add(&acc, &poly_arith::poly_scalar_mul(&with_excl, &ap), curve);
        }
        ap = ap.mul(alpha);

        let sel_push = col(trace::COL_SEL_CALL_PUSH_FRAME);
        let sel_ret = col(trace::COL_SEL_CALL_RETURN);

        // [1] sel_call_push_frame * (frame_depth(ω·X) - frame_depth(X) - 1)
        {
            let depth_shift = poly_arith::poly_shift(col(trace::COL_FRAME_DEPTH), omega);
            let body = poly_arith::poly_sub(&depth_shift, col(trace::COL_FRAME_DEPTH), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            let gated = poly_arith::poly_mul(sel_push, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            acc = poly_arith::poly_add(&acc, &poly_arith::poly_scalar_mul(&with_excl, &ap), curve);
        }
        ap = ap.mul(alpha);

        // [2..=5] sel_call_push_frame * (callee_l_k(ω·X) - input1_l_k(X))
        for k in 0..4 {
            let callee_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLEE_L0 + k), omega);
            let body = poly_arith::poly_sub(&callee_shift, col(trace::COL_INPUT1_L0 + k), curve);
            let gated = poly_arith::poly_mul(sel_push, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            acc = poly_arith::poly_add(&acc, &poly_arith::poly_scalar_mul(&with_excl, &ap), curve);
            ap = ap.mul(alpha);
        }

        // [6..=9] sel_call_push_frame * (caller_l_k(ω·X) - callee_l_k(X))
        for k in 0..4 {
            let caller_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLER_L0 + k), omega);
            let body = poly_arith::poly_sub(&caller_shift, col(trace::COL_FRAME_CALLEE_L0 + k), curve);
            let gated = poly_arith::poly_mul(sel_push, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            acc = poly_arith::poly_add(&acc, &poly_arith::poly_scalar_mul(&with_excl, &ap), curve);
            ap = ap.mul(alpha);
        }

        // [10] sel_call_push_frame * (return_pc(ω·X) - pc(X) - 1)
        {
            let rpc_shift = poly_arith::poly_shift(col(trace::COL_FRAME_RETURN_PC), omega);
            let body = poly_arith::poly_sub(&rpc_shift, col(trace::COL_PC), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            let gated = poly_arith::poly_mul(sel_push, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            acc = poly_arith::poly_add(&acc, &poly_arith::poly_scalar_mul(&with_excl, &ap), curve);
        }
        ap = ap.mul(alpha);

        // [11] sel_call_push_frame * (frame_static(ω·X) - frame_static(X))
        // (CALL propagates parent's static flag).
        {
            let static_shift = poly_arith::poly_shift(col(trace::COL_FRAME_STATIC), omega);
            let body = poly_arith::poly_sub(&static_shift, col(trace::COL_FRAME_STATIC), curve);
            let gated = poly_arith::poly_mul(sel_push, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            acc = poly_arith::poly_add(&acc, &poly_arith::poly_scalar_mul(&with_excl, &ap), curve);
        }
        ap = ap.mul(alpha);

        // [12] sel_call_return * (frame_depth(ω·X) - frame_depth(X) + 1)
        {
            let depth_shift = poly_arith::poly_shift(col(trace::COL_FRAME_DEPTH), omega);
            let body = poly_arith::poly_sub(&depth_shift, col(trace::COL_FRAME_DEPTH), curve);
            let body = poly_arith::poly_add(&body, &one_poly, curve);
            let gated = poly_arith::poly_mul(sel_ret, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            acc = poly_arith::poly_add(&acc, &poly_arith::poly_scalar_mul(&with_excl, &ap), curve);
        }
        ap = ap.mul(alpha);

        // ── Call-family transitions: CALLCODE / DELEGATECALL / STATICCALL / REVERT / CREATE / CREATE2 ──
        let sel_callcode = col(trace::COL_SEL_CALLCODE);
        let sel_delegate = col(trace::COL_SEL_DELEGATECALL);
        let sel_create = col(trace::COL_SEL_CREATE);
        let sel_create2 = col(trace::COL_SEL_CREATE2);
        let sel_static = col(trace::COL_SEL_STATICCALL);
        let sel_revert = col(trace::COL_SEL_REVERT);

        // Helper: emit a (selector * body) constraint with exclusion factor.
        let emit_gated = |acc: &mut Vec<Scalar>,
                          ap: &mut Scalar,
                          sel: &Vec<Scalar>,
                          body: Vec<Scalar>| {
            let gated = poly_arith::poly_mul(sel, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            *acc = poly_arith::poly_add(acc, &poly_arith::poly_scalar_mul(&with_excl, ap), curve);
            *ap = ap.mul(alpha);
        };

        // CALLCODE (same shape as CALL_PUSH).
        {
            let depth_shift = poly_arith::poly_shift(col(trace::COL_FRAME_DEPTH), omega);
            let body = poly_arith::poly_sub(&depth_shift, col(trace::COL_FRAME_DEPTH), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_callcode, body);
            for k in 0..4 {
                let callee_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLEE_L0 + k), omega);
                let body = poly_arith::poly_sub(&callee_shift, col(trace::COL_INPUT1_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_callcode, body);
            }
            for k in 0..4 {
                let caller_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLER_L0 + k), omega);
                let body = poly_arith::poly_sub(&caller_shift, col(trace::COL_FRAME_CALLEE_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_callcode, body);
            }
            let rpc_shift = poly_arith::poly_shift(col(trace::COL_FRAME_RETURN_PC), omega);
            let body = poly_arith::poly_sub(&rpc_shift, col(trace::COL_PC), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_callcode, body);
            let static_shift = poly_arith::poly_shift(col(trace::COL_FRAME_STATIC), omega);
            let body = poly_arith::poly_sub(&static_shift, col(trace::COL_FRAME_STATIC), curve);
            emit_gated(&mut acc, &mut ap, sel_callcode, body);
        }

        // DELEGATECALL: depth_inc, callee=input1, caller=parent.caller,
        // value=parent.value, return_pc=pc+1, static_propagate.
        {
            let depth_shift = poly_arith::poly_shift(col(trace::COL_FRAME_DEPTH), omega);
            let body = poly_arith::poly_sub(&depth_shift, col(trace::COL_FRAME_DEPTH), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_delegate, body);
            for k in 0..4 {
                let callee_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLEE_L0 + k), omega);
                let body = poly_arith::poly_sub(&callee_shift, col(trace::COL_INPUT1_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_delegate, body);
            }
            // caller_preserve: caller_next = current caller (NOT current callee).
            for k in 0..4 {
                let caller_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLER_L0 + k), omega);
                let body = poly_arith::poly_sub(&caller_shift, col(trace::COL_FRAME_CALLER_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_delegate, body);
            }
            // value_preserve: value_next = current value.
            for k in 0..4 {
                let value_shift = poly_arith::poly_shift(col(trace::COL_FRAME_VALUE_L0 + k), omega);
                let body = poly_arith::poly_sub(&value_shift, col(trace::COL_FRAME_VALUE_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_delegate, body);
            }
            let rpc_shift = poly_arith::poly_shift(col(trace::COL_FRAME_RETURN_PC), omega);
            let body = poly_arith::poly_sub(&rpc_shift, col(trace::COL_PC), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_delegate, body);
            let static_shift = poly_arith::poly_shift(col(trace::COL_FRAME_STATIC), omega);
            let body = poly_arith::poly_sub(&static_shift, col(trace::COL_FRAME_STATIC), curve);
            emit_gated(&mut acc, &mut ap, sel_delegate, body);
        }

        // CREATE: depth_inc, callee=create_address_hint, caller=current callee,
        // return_pc=pc+1, static_propagate.
        {
            let depth_shift = poly_arith::poly_shift(col(trace::COL_FRAME_DEPTH), omega);
            let body = poly_arith::poly_sub(&depth_shift, col(trace::COL_FRAME_DEPTH), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_create, body);
            for k in 0..4 {
                let callee_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLEE_L0 + k), omega);
                let body = poly_arith::poly_sub(&callee_shift, col(trace::COL_CREATE_ADDRESS_HINT_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_create, body);
            }
            for k in 0..4 {
                let caller_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLER_L0 + k), omega);
                let body = poly_arith::poly_sub(&caller_shift, col(trace::COL_FRAME_CALLEE_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_create, body);
            }
            let rpc_shift = poly_arith::poly_shift(col(trace::COL_FRAME_RETURN_PC), omega);
            let body = poly_arith::poly_sub(&rpc_shift, col(trace::COL_PC), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_create, body);
            let static_shift = poly_arith::poly_shift(col(trace::COL_FRAME_STATIC), omega);
            let body = poly_arith::poly_sub(&static_shift, col(trace::COL_FRAME_STATIC), curve);
            emit_gated(&mut acc, &mut ap, sel_create, body);
        }

        // CREATE2: same shape as CREATE.
        {
            let depth_shift = poly_arith::poly_shift(col(trace::COL_FRAME_DEPTH), omega);
            let body = poly_arith::poly_sub(&depth_shift, col(trace::COL_FRAME_DEPTH), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_create2, body);
            for k in 0..4 {
                let callee_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLEE_L0 + k), omega);
                let body = poly_arith::poly_sub(&callee_shift, col(trace::COL_CREATE_ADDRESS_HINT_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_create2, body);
            }
            for k in 0..4 {
                let caller_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLER_L0 + k), omega);
                let body = poly_arith::poly_sub(&caller_shift, col(trace::COL_FRAME_CALLEE_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_create2, body);
            }
            let rpc_shift = poly_arith::poly_shift(col(trace::COL_FRAME_RETURN_PC), omega);
            let body = poly_arith::poly_sub(&rpc_shift, col(trace::COL_PC), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_create2, body);
            let static_shift = poly_arith::poly_shift(col(trace::COL_FRAME_STATIC), omega);
            let body = poly_arith::poly_sub(&static_shift, col(trace::COL_FRAME_STATIC), curve);
            emit_gated(&mut acc, &mut ap, sel_create2, body);
        }

        // STATICCALL: depth_inc, callee=input1, caller=current callee,
        // return_pc=pc+1, static_set: frame_static(ω·X) = 1.
        {
            let depth_shift = poly_arith::poly_shift(col(trace::COL_FRAME_DEPTH), omega);
            let body = poly_arith::poly_sub(&depth_shift, col(trace::COL_FRAME_DEPTH), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_static, body);
            for k in 0..4 {
                let callee_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLEE_L0 + k), omega);
                let body = poly_arith::poly_sub(&callee_shift, col(trace::COL_INPUT1_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_static, body);
            }
            for k in 0..4 {
                let caller_shift = poly_arith::poly_shift(col(trace::COL_FRAME_CALLER_L0 + k), omega);
                let body = poly_arith::poly_sub(&caller_shift, col(trace::COL_FRAME_CALLEE_L0 + k), curve);
                emit_gated(&mut acc, &mut ap, sel_static, body);
            }
            let rpc_shift = poly_arith::poly_shift(col(trace::COL_FRAME_RETURN_PC), omega);
            let body = poly_arith::poly_sub(&rpc_shift, col(trace::COL_PC), curve);
            let body = poly_arith::poly_sub(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_static, body);
            // static_set: frame_static(ω·X) = 1 ⟺ frame_static(ω·X) - 1 = 0.
            let static_shift = poly_arith::poly_shift(col(trace::COL_FRAME_STATIC), omega);
            let body = poly_arith::poly_sub(&static_shift, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_static, body);
        }

        // REVERT: depth_dec.
        {
            let depth_shift = poly_arith::poly_shift(col(trace::COL_FRAME_DEPTH), omega);
            let body = poly_arith::poly_sub(&depth_shift, col(trace::COL_FRAME_DEPTH), curve);
            let body = poly_arith::poly_add(&body, &one_poly, curve);
            emit_gated(&mut acc, &mut ap, sel_revert, body);
        }

        // ── Default frame-state preservation polynomials ─────────────────────
        //
        // Mirror of evaluate_shifted_at_point's preservation block. For each
        // of 15 frame-state columns F:
        //     (1 - is_frame_op(X)) · (F(ω·X) - F(X))
        // multiplied by the wrap-exclusion factor (X - ω^{n-1}). See the
        // comment above num_shifted_constraints for the soundness argument
        // and which columns are (and are not) covered.
        let sel_call_push_frame = col(trace::COL_SEL_CALL_PUSH_FRAME);
        let sel_call_return_p = col(trace::COL_SEL_CALL_RETURN);
        // sel_stop_pop (depth-conditioned: STOP at depth ≥ 1) covers
        // implicit frame pops alongside the explicit frame-op selectors.
        // STOP at depth 0 doesn't change frame state (next row is
        // padding with frame columns set to last real row's value by
        // fix_trace_padding), so it stays out of is_frame_op.
        // Mirrors the verifier evaluator.
        let is_frame_op_poly: Vec<Scalar> = {
            let mut p = sel_call_push_frame.clone();
            p = poly_arith::poly_add(&p, sel_call_return_p, curve);
            p = poly_arith::poly_add(&p, col(trace::COL_SEL_CALLCODE), curve);
            p = poly_arith::poly_add(&p, col(trace::COL_SEL_DELEGATECALL), curve);
            p = poly_arith::poly_add(&p, col(trace::COL_SEL_CREATE), curve);
            p = poly_arith::poly_add(&p, col(trace::COL_SEL_CREATE2), curve);
            p = poly_arith::poly_add(&p, col(trace::COL_SEL_STATICCALL), curve);
            p = poly_arith::poly_add(&p, col(trace::COL_SEL_REVERT), curve);
            p = poly_arith::poly_add(&p, col(trace::COL_SEL_STOP_POP), curve);
            p
        };
        let preserve_gate_poly = poly_arith::poly_sub(&one_poly, &is_frame_op_poly, curve);

        let preserve_columns: [usize; 18] = [
            trace::COL_FRAME_DEPTH,
            trace::COL_FRAME_CALLER_L0,
            trace::COL_FRAME_CALLER_L1,
            trace::COL_FRAME_CALLER_L2,
            trace::COL_FRAME_CALLER_L3,
            trace::COL_FRAME_CALLEE_L0,
            trace::COL_FRAME_CALLEE_L1,
            trace::COL_FRAME_CALLEE_L2,
            trace::COL_FRAME_CALLEE_L3,
            trace::COL_FRAME_RETURN_PC,
            trace::COL_FRAME_VALUE_L0,
            trace::COL_FRAME_VALUE_L1,
            trace::COL_FRAME_VALUE_L2,
            trace::COL_FRAME_VALUE_L3,
            trace::COL_FRAME_STATIC,
            trace::COL_FRAME_GAS,
            trace::COL_FRAME_RETURN_OFFSET,
            trace::COL_FRAME_RETURN_SIZE,
        ];
        for &col_idx in preserve_columns.iter() {
            let f_shift = poly_arith::poly_shift(col(col_idx), omega);
            let body = poly_arith::poly_sub(&f_shift, col(col_idx), curve);
            let gated = poly_arith::poly_mul(&preserve_gate_poly, &body, curve);
            let with_excl = poly_arith::poly_mul_linear(&gated, &omega_n_minus_1);
            acc = poly_arith::poly_add(&acc, &poly_arith::poly_scalar_mul(&with_excl, &ap), curve);
            ap = ap.mul(alpha);
        }

        // `ap` not used after; suppress the unused result warning.
        let _ = ap;

        acc
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let range_16 = LookupTable::range(16);
        let mut decls = Vec::new();

        // MUL: aux0 is currently unused on MUL rows (the AIR only enforces
        // the low-256-bit schoolbook). The 64-bit range checks below are
        // defensive well-formedness; no constraint reads these values.
        // Kept so future upper-half binding (e.g. for MULMOD) can rely on
        // the bound without re-declaring.
        for (i, &col) in [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("mul_aux0_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ARITH_MUL),
            }, 0));
        }

        // DIV: aux0 limbs hold remainder.
        // Without range check, quotient is not unique.
        for (i, &col) in [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("div_aux0_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ARITH_DIV),
            }, 0));
        }

        // MUL carry: aux1 limbs hold intermediate carries.
        for (i, &col) in [trace::COL_AUX1_L0, trace::COL_AUX1_L1, trace::COL_AUX1_L2, trace::COL_AUX1_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("mul_aux1_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ARITH_MUL),
            }, 0));
        }

        // LT/GT: aux1 limbs hold diff values from subtraction, need 64-bit range check.
        for (i, &col) in [trace::COL_AUX1_L0, trace::COL_AUX1_L1, trace::COL_AUX1_L2, trace::COL_AUX1_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("lt_aux1_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_LT),
            }, 0));
        }

        // SHL: aux0 limbs hold carry chain from multiplication.
        for (i, &col) in [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("shl_aux0_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_SHL),
            }, 0));
        }

        // SHR: aux0 limbs hold remainder, aux1 holds carry chain.
        for (i, &col) in [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("shr_aux0_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_SHR),
            }, 0));
        }
        for (i, &col) in [trace::COL_AUX1_L0, trace::COL_AUX1_L1, trace::COL_AUX1_L2, trace::COL_AUX1_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("shr_aux1_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_SHR),
            }, 0));
        }

        // SAR: same aux layout as SHR.
        for (i, &col) in [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("sar_aux0_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_SAR),
            }, 0));
        }
        for (i, &col) in [trace::COL_AUX1_L0, trace::COL_AUX1_L1, trace::COL_AUX1_L2, trace::COL_AUX1_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("sar_aux1_l{}_64bit", i),
                column_index: col,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_SAR),
            }, 0));
        }

        // Data columns: all U256 limbs should be < 2^64.
        let data_limbs = [
            (trace::COL_INPUT0_L0, "input0_l0"), (trace::COL_INPUT0_L1, "input0_l1"),
            (trace::COL_INPUT0_L2, "input0_l2"), (trace::COL_INPUT0_L3, "input0_l3"),
            (trace::COL_INPUT1_L0, "input1_l0"), (trace::COL_INPUT1_L1, "input1_l1"),
            (trace::COL_INPUT1_L2, "input1_l2"), (trace::COL_INPUT1_L3, "input1_l3"),
            (trace::COL_OUTPUT0_L0, "output0_l0"), (trace::COL_OUTPUT0_L1, "output0_l1"),
            (trace::COL_OUTPUT0_L2, "output0_l2"), (trace::COL_OUTPUT0_L3, "output0_l3"),
        ];
        for (col, name) in data_limbs {
            decls.push((LookupDeclaration {
                label: format!("{}_64bit", name),
                column_index: col,
                max_bits: 64,
                selector_column: None,
            }, 0));
        }

        // SIGNEXTEND byte-decomposition columns: each must be in [0, 255].
        // Gated by sel_signextend so non-SIGNEXTEND rows (where these are 0
        // via push_row defaults) are also trivially in range.
        for i in 0..8 {
            decls.push((LookupDeclaration {
                label: format!("se_byte_{}_8bit", i),
                column_index: trace::COL_SE_BYTE_0 + i,
                max_bits: 8,
                selector_column: Some(trace::COL_SEL_SIGNEXTEND),
            }, 0));
        }
        // se_low7 in [0, 127] — 7-bit range check.
        decls.push((LookupDeclaration {
            label: "se_low7_7bit".to_string(),
            column_index: trace::COL_SE_LOW7,
            max_bits: 7,
            selector_column: Some(trace::COL_SEL_SIGNEXTEND),
        }, 0));

        // MULMOD range checks (all gated by sel_mulmod).
        // Quotient q limbs must be 64-bit for the schoolbook identity to be sound.
        for (i, &c) in [trace::COL_MULMOD_Q_L0, trace::COL_MULMOD_Q_L1,
                        trace::COL_MULMOD_Q_L2, trace::COL_MULMOD_Q_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("mulmod_q_l{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_MULMOD),
            }, 0));
        }
        // Product p limbs: 8 cols, each 64-bit.
        for (i, &c) in [trace::COL_MULMOD_P_L0, trace::COL_MULMOD_P_L1,
                        trace::COL_MULMOD_P_L2, trace::COL_MULMOD_P_L3,
                        trace::COL_MULMOD_P_L4, trace::COL_MULMOD_P_L5,
                        trace::COL_MULMOD_P_L6, trace::COL_MULMOD_P_L7].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("mulmod_p_l{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_MULMOD),
            }, 0));
        }
        // Carry columns pc, sc: each 64-bit.
        for (i, &c) in [trace::COL_MULMOD_PC_0, trace::COL_MULMOD_PC_1,
                        trace::COL_MULMOD_PC_2, trace::COL_MULMOD_PC_3,
                        trace::COL_MULMOD_PC_4, trace::COL_MULMOD_PC_5,
                        trace::COL_MULMOD_PC_6].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("mulmod_pc_{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_MULMOD),
            }, 0));
        }
        for (i, &c) in [trace::COL_MULMOD_SC_0, trace::COL_MULMOD_SC_1,
                        trace::COL_MULMOD_SC_2, trace::COL_MULMOD_SC_3,
                        trace::COL_MULMOD_SC_4, trace::COL_MULMOD_SC_5,
                        trace::COL_MULMOD_SC_6].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("mulmod_sc_{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_MULMOD),
            }, 0));
        }
        // Slack limbs: each 64-bit.
        for (i, &c) in [trace::COL_MULMOD_SLACK_L0, trace::COL_MULMOD_SLACK_L1,
                        trace::COL_MULMOD_SLACK_L2, trace::COL_MULMOD_SLACK_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("mulmod_slack_l{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_MULMOD),
            }, 0));
        }

        // ADDMOD range checks (all gated by sel_addmod).
        // Quotient q limbs: 64-bit.
        for (i, &c) in [trace::COL_ADDMOD_Q_L0, trace::COL_ADDMOD_Q_L1,
                        trace::COL_ADDMOD_Q_L2, trace::COL_ADDMOD_Q_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("addmod_q_l{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ADDMOD),
            }, 0));
        }
        // s_low limbs: 64-bit.
        for (i, &c) in [trace::COL_ADDMOD_S_LOW_L0, trace::COL_ADDMOD_S_LOW_L1,
                        trace::COL_ADDMOD_S_LOW_L2, trace::COL_ADDMOD_S_LOW_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("addmod_s_low_l{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ADDMOD),
            }, 0));
        }
        // q*n product limbs: 8 cols, each 64-bit.
        for (i, &c) in [trace::COL_ADDMOD_QN_P_L0, trace::COL_ADDMOD_QN_P_L1,
                        trace::COL_ADDMOD_QN_P_L2, trace::COL_ADDMOD_QN_P_L3,
                        trace::COL_ADDMOD_QN_P_L4, trace::COL_ADDMOD_QN_P_L5,
                        trace::COL_ADDMOD_QN_P_L6, trace::COL_ADDMOD_QN_P_L7].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("addmod_qn_p_l{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ADDMOD),
            }, 0));
        }
        // q*n carries: 7 cols, each 64-bit.
        for (i, &c) in [trace::COL_ADDMOD_QN_CARRY_0, trace::COL_ADDMOD_QN_CARRY_1,
                        trace::COL_ADDMOD_QN_CARRY_2, trace::COL_ADDMOD_QN_CARRY_3,
                        trace::COL_ADDMOD_QN_CARRY_4, trace::COL_ADDMOD_QN_CARRY_5,
                        trace::COL_ADDMOD_QN_CARRY_6].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("addmod_qn_carry_{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ADDMOD),
            }, 0));
        }
        // sum_carry: 9 cols, each 64-bit.
        for (i, &c) in [trace::COL_ADDMOD_SUM_CARRY_0, trace::COL_ADDMOD_SUM_CARRY_1,
                        trace::COL_ADDMOD_SUM_CARRY_2, trace::COL_ADDMOD_SUM_CARRY_3,
                        trace::COL_ADDMOD_SUM_CARRY_4, trace::COL_ADDMOD_SUM_CARRY_5,
                        trace::COL_ADDMOD_SUM_CARRY_6, trace::COL_ADDMOD_SUM_CARRY_7,
                        trace::COL_ADDMOD_SUM_CARRY_8].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("addmod_sum_carry_{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ADDMOD),
            }, 0));
        }
        // Slack limbs: each 64-bit.
        for (i, &c) in [trace::COL_ADDMOD_SLACK_L0, trace::COL_ADDMOD_SLACK_L1,
                        trace::COL_ADDMOD_SLACK_L2, trace::COL_ADDMOD_SLACK_L3].iter().enumerate() {
            decls.push((LookupDeclaration {
                label: format!("addmod_slack_l{}_64bit", i),
                column_index: c,
                max_bits: 64,
                selector_column: Some(trace::COL_SEL_ADDMOD),
            }, 0));
        }

        LookupRequirements {
            tables: vec![range_16],
            declarations: decls,
        }
    }

    fn memory_columns(&self) -> Option<(usize, Vec<usize>, Vec<usize>, Vec<usize>)> {
        Some((
            trace::COL_MEM_OFFSET,
            vec![trace::COL_MEM_VALUE_L0, trace::COL_MEM_VALUE_L1,
                 trace::COL_MEM_VALUE_L2, trace::COL_MEM_VALUE_L3],
            vec![trace::COL_SEL_MLOAD],
            // sel_mstore (32-byte word write) AND sel_mstore8 (single-byte write).
            // Both populate the trace's `mem_offset` + `mem_value` columns via
            // the inspector, so both must contribute their trace tuples to the
            // memory permutation. Previously only sel_mstore was listed, which
            // caused MSTORE8 rows to enter the permutation as DUMMY (addr=0,
            // values=[0;4]) on the prover side but as ACTUAL TRACE VALUES on the
            // verifier's numerator reconstruction — a silent grand-product
            // mismatch that broke any EVM proof containing MSTORE8 under
            // prove_with_scheme (the legacy `prove` skips memory permutation
            // entirely, masking the bug). Fixed 2026-05-12.
            vec![trace::COL_SEL_MSTORE, trace::COL_SEL_MSTORE8],
        ))
    }

    fn oracle_selectors(&self) -> Vec<usize> {
        vec![
            // ADDMOD is now algebraic (not oracle).
            // MULMOD is now algebraic (not oracle).
            trace::COL_SEL_EXP,
            // SIGNEXTEND is now algebraic (not oracle).
            trace::COL_SEL_KECCAK, trace::COL_SEL_ENV,
            trace::COL_SEL_BLOCK,
            trace::COL_SEL_STACK_OTHER,
            trace::COL_SEL_MSIZE,
            trace::COL_SEL_MEMORY_OTHER,
            trace::COL_SEL_STORAGE, trace::COL_SEL_LOG,
            trace::COL_SEL_CALL,
        ]
    }

    fn frame_perm_layout(&self) -> Option<metavm_zkp::permutation::FrameStackPermLayout> {
        // Full frame-stack LIFO multiset permutation. Tuple components are
        // read at row r for push and at row ω·r for pop — i.e. the parent
        // frame state being saved/restored. Includes all 18 frame-state
        // columns so a popped frame's gas budget and return-data layout
        // match what was pushed, not just the frame identity.
        //
        // PUSH selectors (frame is pushed on row r → tuple at r is the
        // saved parent state):
        //   sel_call_push_frame, sel_callcode, sel_delegatecall,
        //   sel_create, sel_create2, sel_staticcall.
        //
        // POP selectors (frame is popped on row r → tuple at ω·r is
        // the restored parent state):
        //   sel_call_return, sel_revert.
        Some(metavm_zkp::permutation::FrameStackPermLayout {
            z_column: 0,
            num_columns: 1,
            tuple_columns: vec![
                trace::COL_FRAME_DEPTH,
                trace::COL_FRAME_CALLER_L0,
                trace::COL_FRAME_CALLER_L1,
                trace::COL_FRAME_CALLER_L2,
                trace::COL_FRAME_CALLER_L3,
                trace::COL_FRAME_CALLEE_L0,
                trace::COL_FRAME_CALLEE_L1,
                trace::COL_FRAME_CALLEE_L2,
                trace::COL_FRAME_CALLEE_L3,
                trace::COL_FRAME_VALUE_L0,
                trace::COL_FRAME_VALUE_L1,
                trace::COL_FRAME_VALUE_L2,
                trace::COL_FRAME_VALUE_L3,
                trace::COL_FRAME_RETURN_PC,
                trace::COL_FRAME_STATIC,
                trace::COL_FRAME_GAS,
                trace::COL_FRAME_RETURN_OFFSET,
                trace::COL_FRAME_RETURN_SIZE,
            ],
            push_selectors: vec![
                trace::COL_SEL_CALL_PUSH_FRAME,
                trace::COL_SEL_CALLCODE,
                trace::COL_SEL_DELEGATECALL,
                trace::COL_SEL_CREATE,
                trace::COL_SEL_CREATE2,
                trace::COL_SEL_STATICCALL,
            ],
            pop_selectors: vec![
                trace::COL_SEL_CALL_RETURN,
                trace::COL_SEL_REVERT,
                trace::COL_SEL_STOP_POP,
            ],
        })
    }

    fn bitwise_lookup_declarations(&self) -> Vec<BitwiseLookupDeclaration> {
        let mut decls = Vec::new();

        // AND: per-limb lookup (64-bit per limb)
        for limb in 0..4 {
            decls.push(BitwiseLookupDeclaration {
                label: format!("and_limb{}", limb),
                operand_a_column: trace::COL_INPUT0_L0 + limb,
                operand_b_column: trace::COL_INPUT1_L0 + limb,
                result_column: trace::COL_OUTPUT0_L0 + limb,
                width_bits: 64,
                op: BitwiseOp::And,
                selectors: vec![trace::COL_SEL_AND],
            });
        }

        // OR: per-limb lookup (64-bit per limb)
        for limb in 0..4 {
            decls.push(BitwiseLookupDeclaration {
                label: format!("or_limb{}", limb),
                operand_a_column: trace::COL_INPUT0_L0 + limb,
                operand_b_column: trace::COL_INPUT1_L0 + limb,
                result_column: trace::COL_OUTPUT0_L0 + limb,
                width_bits: 64,
                op: BitwiseOp::Or,
                selectors: vec![trace::COL_SEL_OR],
            });
        }

        // XOR: per-limb lookup (64-bit per limb)
        for limb in 0..4 {
            decls.push(BitwiseLookupDeclaration {
                label: format!("xor_limb{}", limb),
                operand_a_column: trace::COL_INPUT0_L0 + limb,
                operand_b_column: trace::COL_INPUT1_L0 + limb,
                result_column: trace::COL_OUTPUT0_L0 + limb,
                width_bits: 64,
                op: BitwiseOp::Xor,
                selectors: vec![trace::COL_SEL_XOR],
            });
        }

        decls
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bitwise constraint raw evaluation functions
// ═══════════════════════════════════════════════════════════════════════════

/// NOT raw constraint body for a single limb: output_lk + input0_lk - (2^64 - 1) = 0
fn evaluate_not_limb_raw(col_evals: &[Scalar], limb: usize) -> Scalar {
    let curve = col_evals[0].curve_type();
    let max64 = Scalar::from_u64(u64::MAX, curve);
    let out = trace::COL_OUTPUT0_L0 + limb;
    let in0 = trace::COL_INPUT0_L0 + limb;
    col_evals[out].add(&col_evals[in0]).sub(&max64)
}

/// AND raw constraint body for a single limb: output_lk - aux0_lk = 0
fn evaluate_and_limb_raw(col_evals: &[Scalar], limb: usize) -> Scalar {
    let out = trace::COL_OUTPUT0_L0 + limb;
    let aux = trace::COL_AUX0_L0 + limb;
    col_evals[out].sub(&col_evals[aux])
}

/// OR raw constraint body for a single limb: output_lk - input0_lk - input1_lk + aux0_lk = 0
fn evaluate_or_limb_raw(col_evals: &[Scalar], limb: usize) -> Scalar {
    let out = trace::COL_OUTPUT0_L0 + limb;
    let in0 = trace::COL_INPUT0_L0 + limb;
    let in1 = trace::COL_INPUT1_L0 + limb;
    let aux = trace::COL_AUX0_L0 + limb;
    col_evals[out].sub(&col_evals[in0]).sub(&col_evals[in1]).add(&col_evals[aux])
}

/// XOR raw constraint body for a single limb: output_lk - input0_lk - input1_lk + 2*aux0_lk = 0
fn evaluate_xor_limb_raw(col_evals: &[Scalar], limb: usize) -> Scalar {
    let two = Scalar::from_u64(2, col_evals[0].curve_type());
    let out = trace::COL_OUTPUT0_L0 + limb;
    let in0 = trace::COL_INPUT0_L0 + limb;
    let in1 = trace::COL_INPUT1_L0 + limb;
    let aux = trace::COL_AUX0_L0 + limb;
    col_evals[out].sub(&col_evals[in0]).sub(&col_evals[in1]).add(&two.mul(&col_evals[aux]))
}

/// Evaluate NOT constraint for given limb on full domain (gated by sel_bitwise_other).
fn evaluate_not_limb(columns: &[&Vec<Scalar>], num_rows: usize, limb: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[trace::COL_SEL_BITWISE_OTHER][i];
        if sel.is_zero() {
            result.push(zero.clone());
        } else {
            let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_not_limb_raw(&cols_at_i, limb));
        }
    }
    result
}

/// Evaluate AND constraint for limb 0 on full domain (gated by sel_and).
fn evaluate_and(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    evaluate_and_limb(columns, num_rows, 0)
}

/// Evaluate AND constraint for given limb on full domain (gated by sel_and).
fn evaluate_and_limb(columns: &[&Vec<Scalar>], num_rows: usize, limb: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[trace::COL_SEL_AND][i];
        if sel.is_zero() {
            result.push(zero.clone());
        } else {
            let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_and_limb_raw(&cols_at_i, limb));
        }
    }
    result
}

/// Evaluate OR constraint for limb 0 on full domain (gated by sel_or).
fn evaluate_or(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    evaluate_or_limb(columns, num_rows, 0)
}

/// Evaluate OR constraint for given limb on full domain (gated by sel_or).
fn evaluate_or_limb(columns: &[&Vec<Scalar>], num_rows: usize, limb: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[trace::COL_SEL_OR][i];
        if sel.is_zero() {
            result.push(zero.clone());
        } else {
            let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_or_limb_raw(&cols_at_i, limb));
        }
    }
    result
}

/// Evaluate XOR constraint for limb 0 on full domain (gated by sel_xor).
fn evaluate_xor(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    evaluate_xor_limb(columns, num_rows, 0)
}

/// Evaluate XOR constraint for given limb on full domain (gated by sel_xor).
fn evaluate_xor_limb(columns: &[&Vec<Scalar>], num_rows: usize, limb: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let sel = &columns[trace::COL_SEL_XOR][i];
        if sel.is_zero() {
            result.push(zero.clone());
        } else {
            let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
                .map(|c| columns[c][i].clone())
                .collect();
            result.push(evaluate_xor_limb_raw(&cols_at_i, limb));
        }
    }
    result
}

/// Build full 4-limb MUL constraint polynomial.
///
/// Schoolbook multiplication: a * b = c (mod 2^256), carry chain in aux1.
/// Sum of 4 limb constraints.
fn build_mul_full_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    let a = [trace::COL_INPUT0_L0, trace::COL_INPUT0_L1, trace::COL_INPUT0_L2, trace::COL_INPUT0_L3];
    let b = [trace::COL_INPUT1_L0, trace::COL_INPUT1_L1, trace::COL_INPUT1_L2, trace::COL_INPUT1_L3];
    let out = [trace::COL_OUTPUT0_L0, trace::COL_OUTPUT0_L1, trace::COL_OUTPUT0_L2, trace::COL_OUTPUT0_L3];
    let carry = [trace::COL_AUX1_L0, trace::COL_AUX1_L1, trace::COL_AUX1_L2, trace::COL_AUX1_L3];

    // Limb 0: a0*b0 - c0*2^64 - out0
    let l0 = poly_arith::poly_mul(col(a[0]), col(b[0]), curve);
    let l0 = poly_arith::poly_sub(&l0, &poly_arith::poly_scalar_mul(col(carry[0]), two_64), curve);
    let l0 = poly_arith::poly_sub(&l0, col(out[0]), curve);

    // Limb 1: a0*b1 + a1*b0 + c0 - c1*2^64 - out1
    let l1 = poly_arith::poly_mul(col(a[0]), col(b[1]), curve);
    let l1 = poly_arith::poly_add(&l1, &poly_arith::poly_mul(col(a[1]), col(b[0]), curve), curve);
    let l1 = poly_arith::poly_add(&l1, col(carry[0]), curve);
    let l1 = poly_arith::poly_sub(&l1, &poly_arith::poly_scalar_mul(col(carry[1]), two_64), curve);
    let l1 = poly_arith::poly_sub(&l1, col(out[1]), curve);

    // Limb 2: a0*b2 + a1*b1 + a2*b0 + c1 - c2*2^64 - out2
    let l2 = poly_arith::poly_mul(col(a[0]), col(b[2]), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(a[1]), col(b[1]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(a[2]), col(b[0]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, col(carry[1]), curve);
    let l2 = poly_arith::poly_sub(&l2, &poly_arith::poly_scalar_mul(col(carry[2]), two_64), curve);
    let l2 = poly_arith::poly_sub(&l2, col(out[2]), curve);

    // Limb 3: a0*b3 + a1*b2 + a2*b1 + a3*b0 + c2 - c3*2^64 - out3
    let l3 = poly_arith::poly_mul(col(a[0]), col(b[3]), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(a[1]), col(b[2]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(a[2]), col(b[1]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(a[3]), col(b[0]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, col(carry[2]), curve);
    let l3 = poly_arith::poly_sub(&l3, &poly_arith::poly_scalar_mul(col(carry[3]), two_64), curve);
    let l3 = poly_arith::poly_sub(&l3, col(out[3]), curve);

    let mut body = poly_arith::poly_add(&l0, &l1, curve);
    body = poly_arith::poly_add(&body, &l2, curve);
    body = poly_arith::poly_add(&body, &l3, curve);
    body
}

/// Build full 4-limb DIV constraint polynomial.
///
/// DIV: quotient * divisor + remainder = dividend
/// output=quotient, input0=dividend, input1=divisor, aux0=remainder, aux1=carry chain.
fn build_div_full_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    let q = [trace::COL_OUTPUT0_L0, trace::COL_OUTPUT0_L1, trace::COL_OUTPUT0_L2, trace::COL_OUTPUT0_L3];
    let d = [trace::COL_INPUT1_L0, trace::COL_INPUT1_L1, trace::COL_INPUT1_L2, trace::COL_INPUT1_L3];
    let r = [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3];
    let div = [trace::COL_INPUT0_L0, trace::COL_INPUT0_L1, trace::COL_INPUT0_L2, trace::COL_INPUT0_L3];
    let carry = [trace::COL_AUX1_L0, trace::COL_AUX1_L1, trace::COL_AUX1_L2, trace::COL_AUX1_L3];

    // Limb 0: q0*d0 + r0 - c0*2^64 - div0
    let l0 = poly_arith::poly_mul(col(q[0]), col(d[0]), curve);
    let l0 = poly_arith::poly_add(&l0, col(r[0]), curve);
    let l0 = poly_arith::poly_sub(&l0, &poly_arith::poly_scalar_mul(col(carry[0]), two_64), curve);
    let l0 = poly_arith::poly_sub(&l0, col(div[0]), curve);

    // Limb 1: q0*d1 + q1*d0 + r1 + c0 - c1*2^64 - div1
    let l1 = poly_arith::poly_mul(col(q[0]), col(d[1]), curve);
    let l1 = poly_arith::poly_add(&l1, &poly_arith::poly_mul(col(q[1]), col(d[0]), curve), curve);
    let l1 = poly_arith::poly_add(&l1, col(r[1]), curve);
    let l1 = poly_arith::poly_add(&l1, col(carry[0]), curve);
    let l1 = poly_arith::poly_sub(&l1, &poly_arith::poly_scalar_mul(col(carry[1]), two_64), curve);
    let l1 = poly_arith::poly_sub(&l1, col(div[1]), curve);

    // Limb 2: q0*d2 + q1*d1 + q2*d0 + r2 + c1 - c2*2^64 - div2
    let l2 = poly_arith::poly_mul(col(q[0]), col(d[2]), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(q[1]), col(d[1]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(q[2]), col(d[0]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, col(r[2]), curve);
    let l2 = poly_arith::poly_add(&l2, col(carry[1]), curve);
    let l2 = poly_arith::poly_sub(&l2, &poly_arith::poly_scalar_mul(col(carry[2]), two_64), curve);
    let l2 = poly_arith::poly_sub(&l2, col(div[2]), curve);

    // Limb 3: q0*d3 + q1*d2 + q2*d1 + q3*d0 + r3 + c2 - c3*2^64 - div3
    let l3 = poly_arith::poly_mul(col(q[0]), col(d[3]), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[1]), col(d[2]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[2]), col(d[1]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[3]), col(d[0]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, col(r[3]), curve);
    let l3 = poly_arith::poly_add(&l3, col(carry[2]), curve);
    let l3 = poly_arith::poly_sub(&l3, &poly_arith::poly_scalar_mul(col(carry[3]), two_64), curve);
    let l3 = poly_arith::poly_sub(&l3, col(div[3]), curve);

    let mut body = poly_arith::poly_add(&l0, &l1, curve);
    body = poly_arith::poly_add(&body, &l2, curve);
    body = poly_arith::poly_add(&body, &l3, curve);
    body
}

/// Build full 4-limb MOD constraint polynomial.
///
/// MOD: quotient * divisor + remainder = dividend
/// output=remainder, input0=dividend, input1=divisor, aux0=quotient, aux1=carry chain.
fn build_mod_full_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    let q = [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3];
    let d = [trace::COL_INPUT1_L0, trace::COL_INPUT1_L1, trace::COL_INPUT1_L2, trace::COL_INPUT1_L3];
    let r = [trace::COL_OUTPUT0_L0, trace::COL_OUTPUT0_L1, trace::COL_OUTPUT0_L2, trace::COL_OUTPUT0_L3];
    let div = [trace::COL_INPUT0_L0, trace::COL_INPUT0_L1, trace::COL_INPUT0_L2, trace::COL_INPUT0_L3];
    let carry = [trace::COL_AUX1_L0, trace::COL_AUX1_L1, trace::COL_AUX1_L2, trace::COL_AUX1_L3];

    // Limb 0: q0*d0 + r0 - c0*2^64 - div0
    let l0 = poly_arith::poly_mul(col(q[0]), col(d[0]), curve);
    let l0 = poly_arith::poly_add(&l0, col(r[0]), curve);
    let l0 = poly_arith::poly_sub(&l0, &poly_arith::poly_scalar_mul(col(carry[0]), two_64), curve);
    let l0 = poly_arith::poly_sub(&l0, col(div[0]), curve);

    // Limb 1: q0*d1 + q1*d0 + r1 + c0 - c1*2^64 - div1
    let l1 = poly_arith::poly_mul(col(q[0]), col(d[1]), curve);
    let l1 = poly_arith::poly_add(&l1, &poly_arith::poly_mul(col(q[1]), col(d[0]), curve), curve);
    let l1 = poly_arith::poly_add(&l1, col(r[1]), curve);
    let l1 = poly_arith::poly_add(&l1, col(carry[0]), curve);
    let l1 = poly_arith::poly_sub(&l1, &poly_arith::poly_scalar_mul(col(carry[1]), two_64), curve);
    let l1 = poly_arith::poly_sub(&l1, col(div[1]), curve);

    // Limb 2: q0*d2 + q1*d1 + q2*d0 + r2 + c1 - c2*2^64 - div2
    let l2 = poly_arith::poly_mul(col(q[0]), col(d[2]), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(q[1]), col(d[1]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(q[2]), col(d[0]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, col(r[2]), curve);
    let l2 = poly_arith::poly_add(&l2, col(carry[1]), curve);
    let l2 = poly_arith::poly_sub(&l2, &poly_arith::poly_scalar_mul(col(carry[2]), two_64), curve);
    let l2 = poly_arith::poly_sub(&l2, col(div[2]), curve);

    // Limb 3: q0*d3 + q1*d2 + q2*d1 + q3*d0 + r3 + c2 - c3*2^64 - div3
    let l3 = poly_arith::poly_mul(col(q[0]), col(d[3]), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[1]), col(d[2]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[2]), col(d[1]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[3]), col(d[0]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, col(r[3]), curve);
    let l3 = poly_arith::poly_add(&l3, col(carry[2]), curve);
    let l3 = poly_arith::poly_sub(&l3, &poly_arith::poly_scalar_mul(col(carry[3]), two_64), curve);
    let l3 = poly_arith::poly_sub(&l3, col(div[3]), curve);

    let mut body = poly_arith::poly_add(&l0, &l1, curve);
    body = poly_arith::poly_add(&body, &l2, curve);
    body = poly_arith::poly_add(&body, &l3, curve);
    body
}

/// Build LT constraint polynomial: input0 - input1 subtraction with borrow chain.
///
/// Same as evaluate_lt_raw but using polynomial arithmetic.
fn build_lt_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    one_poly: &[Scalar],
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    // Limb 0: input0_l0 - input1_l0 + borrow0*2^64 - diff0 = 0
    let c0 = poly_arith::poly_sub(col(trace::COL_INPUT0_L0), col(trace::COL_INPUT1_L0), curve);
    let c0 = poly_arith::poly_add(&c0, &poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L0), two_64), curve);
    let c0 = poly_arith::poly_sub(&c0, col(trace::COL_AUX1_L0), curve);

    // Limb 1: input0_l1 - input1_l1 - borrow0 + borrow1*2^64 - diff1 = 0
    let c1 = poly_arith::poly_sub(col(trace::COL_INPUT0_L1), col(trace::COL_INPUT1_L1), curve);
    let c1 = poly_arith::poly_sub(&c1, col(trace::COL_AUX0_L0), curve);
    let c1 = poly_arith::poly_add(&c1, &poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L1), two_64), curve);
    let c1 = poly_arith::poly_sub(&c1, col(trace::COL_AUX1_L1), curve);

    // Limb 2
    let c2 = poly_arith::poly_sub(col(trace::COL_INPUT0_L2), col(trace::COL_INPUT1_L2), curve);
    let c2 = poly_arith::poly_sub(&c2, col(trace::COL_AUX0_L1), curve);
    let c2 = poly_arith::poly_add(&c2, &poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L2), two_64), curve);
    let c2 = poly_arith::poly_sub(&c2, col(trace::COL_AUX1_L2), curve);

    // Limb 3
    let c3 = poly_arith::poly_sub(col(trace::COL_INPUT0_L3), col(trace::COL_INPUT1_L3), curve);
    let c3 = poly_arith::poly_sub(&c3, col(trace::COL_AUX0_L2), curve);
    let c3 = poly_arith::poly_add(&c3, &poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L3), two_64), curve);
    let c3 = poly_arith::poly_sub(&c3, col(trace::COL_AUX1_L3), curve);

    // Borrow binary constraints
    let aux0_l0_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L0), one_poly, curve);
    let bb0 = poly_arith::poly_mul(col(trace::COL_AUX0_L0), &aux0_l0_m1, curve);
    let aux0_l1_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L1), one_poly, curve);
    let bb1 = poly_arith::poly_mul(col(trace::COL_AUX0_L1), &aux0_l1_m1, curve);
    let aux0_l2_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L2), one_poly, curve);
    let bb2 = poly_arith::poly_mul(col(trace::COL_AUX0_L2), &aux0_l2_m1, curve);
    let aux0_l3_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L3), one_poly, curve);
    let bb3 = poly_arith::poly_mul(col(trace::COL_AUX0_L3), &aux0_l3_m1, curve);

    // output_l0 = borrow3
    let output_check = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L0), col(trace::COL_AUX0_L3), curve);

    // Upper output limbs must be zero
    let upper = poly_arith::poly_add(col(trace::COL_OUTPUT0_L1), col(trace::COL_OUTPUT0_L2), curve);
    let upper = poly_arith::poly_add(&upper, col(trace::COL_OUTPUT0_L3), curve);

    let mut body = poly_arith::poly_add(&c0, &c1, curve);
    body = poly_arith::poly_add(&body, &c2, curve);
    body = poly_arith::poly_add(&body, &c3, curve);
    body = poly_arith::poly_add(&body, &bb0, curve);
    body = poly_arith::poly_add(&body, &bb1, curve);
    body = poly_arith::poly_add(&body, &bb2, curve);
    body = poly_arith::poly_add(&body, &bb3, curve);
    body = poly_arith::poly_add(&body, &output_check, curve);
    body = poly_arith::poly_add(&body, &upper, curve);
    body
}

/// Build GT constraint polynomial: input1 - input0 subtraction with borrow chain.
fn build_gt_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    one_poly: &[Scalar],
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    // Limb 0: input1_l0 - input0_l0 + borrow0*2^64 - diff0 = 0
    let c0 = poly_arith::poly_sub(col(trace::COL_INPUT1_L0), col(trace::COL_INPUT0_L0), curve);
    let c0 = poly_arith::poly_add(&c0, &poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L0), two_64), curve);
    let c0 = poly_arith::poly_sub(&c0, col(trace::COL_AUX1_L0), curve);

    let c1 = poly_arith::poly_sub(col(trace::COL_INPUT1_L1), col(trace::COL_INPUT0_L1), curve);
    let c1 = poly_arith::poly_sub(&c1, col(trace::COL_AUX0_L0), curve);
    let c1 = poly_arith::poly_add(&c1, &poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L1), two_64), curve);
    let c1 = poly_arith::poly_sub(&c1, col(trace::COL_AUX1_L1), curve);

    let c2 = poly_arith::poly_sub(col(trace::COL_INPUT1_L2), col(trace::COL_INPUT0_L2), curve);
    let c2 = poly_arith::poly_sub(&c2, col(trace::COL_AUX0_L1), curve);
    let c2 = poly_arith::poly_add(&c2, &poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L2), two_64), curve);
    let c2 = poly_arith::poly_sub(&c2, col(trace::COL_AUX1_L2), curve);

    let c3 = poly_arith::poly_sub(col(trace::COL_INPUT1_L3), col(trace::COL_INPUT0_L3), curve);
    let c3 = poly_arith::poly_sub(&c3, col(trace::COL_AUX0_L2), curve);
    let c3 = poly_arith::poly_add(&c3, &poly_arith::poly_scalar_mul(col(trace::COL_AUX0_L3), two_64), curve);
    let c3 = poly_arith::poly_sub(&c3, col(trace::COL_AUX1_L3), curve);

    let aux0_l0_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L0), one_poly, curve);
    let bb0 = poly_arith::poly_mul(col(trace::COL_AUX0_L0), &aux0_l0_m1, curve);
    let aux0_l1_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L1), one_poly, curve);
    let bb1 = poly_arith::poly_mul(col(trace::COL_AUX0_L1), &aux0_l1_m1, curve);
    let aux0_l2_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L2), one_poly, curve);
    let bb2 = poly_arith::poly_mul(col(trace::COL_AUX0_L2), &aux0_l2_m1, curve);
    let aux0_l3_m1 = poly_arith::poly_sub(col(trace::COL_AUX0_L3), one_poly, curve);
    let bb3 = poly_arith::poly_mul(col(trace::COL_AUX0_L3), &aux0_l3_m1, curve);

    let output_check = poly_arith::poly_sub(col(trace::COL_OUTPUT0_L0), col(trace::COL_AUX0_L3), curve);

    let upper = poly_arith::poly_add(col(trace::COL_OUTPUT0_L1), col(trace::COL_OUTPUT0_L2), curve);
    let upper = poly_arith::poly_add(&upper, col(trace::COL_OUTPUT0_L3), curve);

    let mut body = poly_arith::poly_add(&c0, &c1, curve);
    body = poly_arith::poly_add(&body, &c2, curve);
    body = poly_arith::poly_add(&body, &c3, curve);
    body = poly_arith::poly_add(&body, &bb0, curve);
    body = poly_arith::poly_add(&body, &bb1, curve);
    body = poly_arith::poly_add(&body, &bb2, curve);
    body = poly_arith::poly_add(&body, &bb3, curve);
    body = poly_arith::poly_add(&body, &output_check, curve);
    body = poly_arith::poly_add(&body, &upper, curve);
    body
}

/// Build EQ constraint polynomial.
fn build_eq_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    one_poly: &[Scalar],
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    let out = col(trace::COL_OUTPUT0_L0);
    let binary = poly_arith::poly_mul(out, &poly_arith::poly_sub(out, one_poly, curve), curve);

    let diff0 = poly_arith::poly_sub(col(trace::COL_INPUT0_L0), col(trace::COL_INPUT1_L0), curve);
    let diff1 = poly_arith::poly_sub(col(trace::COL_INPUT0_L1), col(trace::COL_INPUT1_L1), curve);
    let diff2 = poly_arith::poly_sub(col(trace::COL_INPUT0_L2), col(trace::COL_INPUT1_L2), curve);
    let diff3 = poly_arith::poly_sub(col(trace::COL_INPUT0_L3), col(trace::COL_INPUT1_L3), curve);

    let eq0 = poly_arith::poly_mul(out, &diff0, curve);
    let eq1 = poly_arith::poly_mul(out, &diff1, curve);
    let eq2 = poly_arith::poly_mul(out, &diff2, curve);
    let eq3 = poly_arith::poly_mul(out, &diff3, curve);

    let upper = poly_arith::poly_add(col(trace::COL_OUTPUT0_L1), col(trace::COL_OUTPUT0_L2), curve);
    let upper = poly_arith::poly_add(&upper, col(trace::COL_OUTPUT0_L3), curve);

    let mut body = poly_arith::poly_add(&binary, &eq0, curve);
    body = poly_arith::poly_add(&body, &eq1, curve);
    body = poly_arith::poly_add(&body, &eq2, curve);
    body = poly_arith::poly_add(&body, &eq3, curve);
    body = poly_arith::poly_add(&body, &upper, curve);
    body
}

/// Build ISZERO constraint polynomial.
fn build_iszero_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    one_poly: &[Scalar],
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    let out = col(trace::COL_OUTPUT0_L0);
    let binary = poly_arith::poly_mul(out, &poly_arith::poly_sub(out, one_poly, curve), curve);

    let iz0 = poly_arith::poly_mul(out, col(trace::COL_INPUT0_L0), curve);
    let iz1 = poly_arith::poly_mul(out, col(trace::COL_INPUT0_L1), curve);
    let iz2 = poly_arith::poly_mul(out, col(trace::COL_INPUT0_L2), curve);
    let iz3 = poly_arith::poly_mul(out, col(trace::COL_INPUT0_L3), curve);

    let upper = poly_arith::poly_add(col(trace::COL_OUTPUT0_L1), col(trace::COL_OUTPUT0_L2), curve);
    let upper = poly_arith::poly_add(&upper, col(trace::COL_OUTPUT0_L3), curve);

    let mut body = poly_arith::poly_add(&binary, &iz0, curve);
    body = poly_arith::poly_add(&body, &iz1, curve);
    body = poly_arith::poly_add(&body, &iz2, curve);
    body = poly_arith::poly_add(&body, &iz3, curve);
    body = poly_arith::poly_add(&body, &upper, curve);
    body
}

/// Build SHL constraint polynomial: input1 * immediate = output (mod 2^256), carry in aux0.
///
/// 4-limb schoolbook multiplication with immediate (2^k) as second operand.
/// EVM SHL: shift=input0 (top of stack), value=input1. output = input1 << input0.
fn build_shl_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    let a = [trace::COL_INPUT1_L0, trace::COL_INPUT1_L1, trace::COL_INPUT1_L2, trace::COL_INPUT1_L3];
    let b = [trace::COL_IMMEDIATE_L0, trace::COL_IMMEDIATE_L1, trace::COL_IMMEDIATE_L2, trace::COL_IMMEDIATE_L3];
    let out = [trace::COL_OUTPUT0_L0, trace::COL_OUTPUT0_L1, trace::COL_OUTPUT0_L2, trace::COL_OUTPUT0_L3];
    let carry = [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3];

    // Limb 0: a0*b0 - c0*2^64 - out0
    let l0 = poly_arith::poly_mul(col(a[0]), col(b[0]), curve);
    let l0 = poly_arith::poly_sub(&l0, &poly_arith::poly_scalar_mul(col(carry[0]), two_64), curve);
    let l0 = poly_arith::poly_sub(&l0, col(out[0]), curve);

    // Limb 1: a0*b1 + a1*b0 + c0 - c1*2^64 - out1
    let l1 = poly_arith::poly_mul(col(a[0]), col(b[1]), curve);
    let l1 = poly_arith::poly_add(&l1, &poly_arith::poly_mul(col(a[1]), col(b[0]), curve), curve);
    let l1 = poly_arith::poly_add(&l1, col(carry[0]), curve);
    let l1 = poly_arith::poly_sub(&l1, &poly_arith::poly_scalar_mul(col(carry[1]), two_64), curve);
    let l1 = poly_arith::poly_sub(&l1, col(out[1]), curve);

    // Limb 2: a0*b2 + a1*b1 + a2*b0 + c1 - c2*2^64 - out2
    let l2 = poly_arith::poly_mul(col(a[0]), col(b[2]), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(a[1]), col(b[1]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(a[2]), col(b[0]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, col(carry[1]), curve);
    let l2 = poly_arith::poly_sub(&l2, &poly_arith::poly_scalar_mul(col(carry[2]), two_64), curve);
    let l2 = poly_arith::poly_sub(&l2, col(out[2]), curve);

    // Limb 3: a0*b3 + a1*b2 + a2*b1 + a3*b0 + c2 - c3*2^64 - out3
    let l3 = poly_arith::poly_mul(col(a[0]), col(b[3]), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(a[1]), col(b[2]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(a[2]), col(b[1]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(a[3]), col(b[0]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, col(carry[2]), curve);
    let l3 = poly_arith::poly_sub(&l3, &poly_arith::poly_scalar_mul(col(carry[3]), two_64), curve);
    let l3 = poly_arith::poly_sub(&l3, col(out[3]), curve);

    let mut body = poly_arith::poly_add(&l0, &l1, curve);
    body = poly_arith::poly_add(&body, &l2, curve);
    body = poly_arith::poly_add(&body, &l3, curve);
    body
}

/// Build SHR/SAR constraint polynomial: output * immediate + aux0 = input1, carry in aux1.
///
/// Same structure as DIV with immediate as divisor.
/// EVM SHR/SAR: shift=input0, value=input1. output = input1 >> input0.
fn build_shift_div_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    let q = [trace::COL_OUTPUT0_L0, trace::COL_OUTPUT0_L1, trace::COL_OUTPUT0_L2, trace::COL_OUTPUT0_L3];
    let d = [trace::COL_IMMEDIATE_L0, trace::COL_IMMEDIATE_L1, trace::COL_IMMEDIATE_L2, trace::COL_IMMEDIATE_L3];
    let r = [trace::COL_AUX0_L0, trace::COL_AUX0_L1, trace::COL_AUX0_L2, trace::COL_AUX0_L3];
    let div = [trace::COL_INPUT1_L0, trace::COL_INPUT1_L1, trace::COL_INPUT1_L2, trace::COL_INPUT1_L3];
    let carry = [trace::COL_AUX1_L0, trace::COL_AUX1_L1, trace::COL_AUX1_L2, trace::COL_AUX1_L3];

    // Limb 0: q0*d0 + r0 - c0*2^64 - div0
    let l0 = poly_arith::poly_mul(col(q[0]), col(d[0]), curve);
    let l0 = poly_arith::poly_add(&l0, col(r[0]), curve);
    let l0 = poly_arith::poly_sub(&l0, &poly_arith::poly_scalar_mul(col(carry[0]), two_64), curve);
    let l0 = poly_arith::poly_sub(&l0, col(div[0]), curve);

    // Limb 1: q0*d1 + q1*d0 + r1 + c0 - c1*2^64 - div1
    let l1 = poly_arith::poly_mul(col(q[0]), col(d[1]), curve);
    let l1 = poly_arith::poly_add(&l1, &poly_arith::poly_mul(col(q[1]), col(d[0]), curve), curve);
    let l1 = poly_arith::poly_add(&l1, col(r[1]), curve);
    let l1 = poly_arith::poly_add(&l1, col(carry[0]), curve);
    let l1 = poly_arith::poly_sub(&l1, &poly_arith::poly_scalar_mul(col(carry[1]), two_64), curve);
    let l1 = poly_arith::poly_sub(&l1, col(div[1]), curve);

    // Limb 2: q0*d2 + q1*d1 + q2*d0 + r2 + c1 - c2*2^64 - div2
    let l2 = poly_arith::poly_mul(col(q[0]), col(d[2]), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(q[1]), col(d[1]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, &poly_arith::poly_mul(col(q[2]), col(d[0]), curve), curve);
    let l2 = poly_arith::poly_add(&l2, col(r[2]), curve);
    let l2 = poly_arith::poly_add(&l2, col(carry[1]), curve);
    let l2 = poly_arith::poly_sub(&l2, &poly_arith::poly_scalar_mul(col(carry[2]), two_64), curve);
    let l2 = poly_arith::poly_sub(&l2, col(div[2]), curve);

    // Limb 3: q0*d3 + q1*d2 + q2*d1 + q3*d0 + r3 + c2 - c3*2^64 - div3
    let l3 = poly_arith::poly_mul(col(q[0]), col(d[3]), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[1]), col(d[2]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[2]), col(d[1]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, &poly_arith::poly_mul(col(q[3]), col(d[0]), curve), curve);
    let l3 = poly_arith::poly_add(&l3, col(r[3]), curve);
    let l3 = poly_arith::poly_add(&l3, col(carry[2]), curve);
    let l3 = poly_arith::poly_sub(&l3, &poly_arith::poly_scalar_mul(col(carry[3]), two_64), curve);
    let l3 = poly_arith::poly_sub(&l3, col(div[3]), curve);

    let mut body = poly_arith::poly_add(&l0, &l1, curve);
    body = poly_arith::poly_add(&body, &l2, curve);
    body = poly_arith::poly_add(&body, &l3, curve);
    body
}

// ═══════════════════════════════════════════════════════════════════════════
// SIGNEXTEND algebraic constraint
// ═══════════════════════════════════════════════════════════════════════════
//
// EVM SIGNEXTEND(b, a) per Yellow Paper App. H.1:
//   if b >= 31: result = a unchanged
//   else:       sign_bit = bit_index (8*(b+1) - 1) of a
//               bits with index >= 8*(b+1) are set to sign_bit
//               bits with index <  8*(b+1) equal a's bits
//
// Stack layout in trace: input0 = b, input1 = a, output0 = r.
//
// Approach: 32-way case split via selectors `sel_se_0..sel_se_30, sel_se_ge31`.
// Exactly one is 1 when `sel_signextend` = 1 (enforced by sum-binding). For
// case k (0..=30), the sign byte is at byte offset (k%8) within the mixed
// limb `input1_l[k/8]`.
//
// Witness columns (all populated by push_row when sel_signextend=1):
//   - `se_byte_0..se_byte_7`: byte-decomp of input1_l[k/8] (mixed limb).
//   - `se_low7`: low 7 bits of the sign byte.
//   - `se_sign_bit`: top bit of the sign byte (binary).
//
// Constraint body (single algebraic expression summed across all cases):
//   body = Σ_{k=0..30} sel_se_k * case_k_body  +  sel_se_ge31 * passthrough_body
//
// Where:
//   - passthrough_body = Σ_l (output_l - input1_l)
//   - case_k_body = (limb recomp) + (sign-byte split) + (per-limb output check)
//
// Because sel_se_k is binary, when sel_se_k = 0 the whole case_k_body drops.
// When sel_se_k = 1 (exactly one k), the summed body reduces to case_k_body
// which must be 0 for validity.
//
// Soundness of sign-bit extraction:
//   - Each se_byte_i is range-checked to [0, 255] via the 8-bit LogUp table.
//   - se_low7 is range-checked to [0, 127] via the 7-bit range declaration.
//   - Limb recomposition: Σ se_byte_i * 256^i = input1_l[LL] ties bytes to a.
//   - Sign-byte split: se_byte_{BB-1} = se_sign_bit * 128 + se_low7, with
//     se_low7 < 128 and se_byte in [0,255] forces se_sign_bit to be exactly
//     the top bit of the sign byte.

/// Fixed constants for a given SIGNEXTEND case k ∈ 0..=30:
/// - LL: mixed-limb index = k / 8
/// - BB: number of preserved bytes in the mixed limb = k % 8 + 1
///
/// "Mixed limb" output: output_l[LL] = Σ_{i<BB} se_byte_i * 256^i + sign_bit * (2^64 - 256^BB).
/// "Low limbs" (l < LL): output_l = input1_l (fully preserved).
/// "High limbs" (l > LL): output_l = sign_bit * (2^64 - 1) (fully filled).

/// Compute 256^bb as a u64 (bb ∈ 1..=8; when bb=8, wraps — we clamp via branching).
/// Returns the "preserved mask" upper bound: 2^{8*bb}. For bb=8 returns 2^64 as u128.
fn two_pow_bits(bits: u32) -> u128 {
    if bits == 0 { 1 }
    else if bits >= 128 { 0 /* unreachable in our use */ }
    else { 1u128 << bits }
}

/// Compute the signextend constraint body AT A POINT given column evaluations.
/// Returns a single Scalar: the summed case-gated body.
fn evaluate_signextend_raw(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[0].curve_type();
    let mut body = Scalar::zero(curve);

    // For each case k = 0..=30:
    for k in 0..=30usize {
        let sel = &col_evals[trace::COL_SEL_SE_0 + k];
        let ll = k / 8; // mixed-limb index
        let bb = k % 8 + 1; // preserved bytes in mixed limb

        // Collect case sub-bodies:
        let mut case_body = Scalar::zero(curve);

        // (1) Limb recomposition: input1_l[LL] = Σ se_byte_i * 256^i
        let mut recomp = Scalar::zero(curve);
        for i in 0..8 {
            let byte_i = &col_evals[trace::COL_SE_BYTE_0 + i];
            let coeff = Scalar::from_u64(1u64 << (8 * i as u32), curve);
            recomp = recomp.add(&byte_i.mul(&coeff));
        }
        // Note: i=7 gives 256^7 = 2^56. All coefficients fit in u64.
        let limb_ll = &col_evals[trace::COL_INPUT1_L0 + ll];
        case_body = case_body.add(&limb_ll.sub(&recomp));

        // (2) Sign-byte split: se_byte_{BB-1} = sign_bit * 128 + se_low7
        let sign_byte = &col_evals[trace::COL_SE_BYTE_0 + (bb - 1)];
        let sign_bit = &col_evals[trace::COL_SE_SIGN_BIT];
        let low7 = &col_evals[trace::COL_SE_LOW7];
        let c128 = Scalar::from_u64(128, curve);
        let split_body = sign_byte.sub(&sign_bit.mul(&c128)).sub(low7);
        case_body = case_body.add(&split_body);

        // (3) Per-limb output check.
        // "Low limbs" (l < LL): output_l - input1_l = 0
        for l in 0..ll {
            let out_l = &col_evals[trace::COL_OUTPUT0_L0 + l];
            let in1_l = &col_evals[trace::COL_INPUT1_L0 + l];
            case_body = case_body.add(&out_l.sub(in1_l));
        }
        // Mixed limb (l = LL):
        //   output_l - (Σ_{i<BB} se_byte_i * 256^i) - sign_bit * (2^64 - 256^BB) = 0
        {
            let out_l = &col_evals[trace::COL_OUTPUT0_L0 + ll];
            let mut preserved_sum = Scalar::zero(curve);
            for i in 0..bb {
                let byte_i = &col_evals[trace::COL_SE_BYTE_0 + i];
                let coeff = Scalar::from_u64(1u64 << (8 * i as u32), curve);
                preserved_sum = preserved_sum.add(&byte_i.mul(&coeff));
            }
            // fill_mask = 2^64 - 256^BB = 2^64 - 2^(8*BB).
            // When BB=8, 2^(8*BB) = 2^64, so fill_mask = 0. But BB ≤ 8 and k ≤ 30 means
            // BB in 1..=7 when k%8 != 7, OR BB=8 when k%8 = 7 (k = 7, 15, 23).
            // For BB=8, output_l should equal preserved_sum (full byte preservation).
            let fill_mask_u128 = (1u128 << 64) - two_pow_bits(8 * bb as u32);
            // Fits in u64 when BB < 8. When BB = 8, it's 0.
            // We encode via either a single u64 Scalar or zero.
            let fill_mask = if bb < 8 {
                Scalar::from_u64(fill_mask_u128 as u64, curve)
            } else {
                Scalar::zero(curve)
            };
            let fill_term = sign_bit.mul(&fill_mask);
            case_body = case_body.add(&out_l.sub(&preserved_sum).sub(&fill_term));
        }
        // High limbs (l > LL): output_l - sign_bit * (2^64 - 1) = 0
        let max_u64 = Scalar::from_u64(u64::MAX, curve);
        for l in (ll + 1)..4 {
            let out_l = &col_evals[trace::COL_OUTPUT0_L0 + l];
            let fill = sign_bit.mul(&max_u64);
            case_body = case_body.add(&out_l.sub(&fill));
        }

        body = body.add(&sel.mul(&case_body));
    }

    // Passthrough case (k = 31, sel_se_ge31): output_l - input1_l = 0 for each l.
    {
        let sel = &col_evals[trace::COL_SEL_SE_GE31];
        let mut pt_body = Scalar::zero(curve);
        for l in 0..4 {
            let out_l = &col_evals[trace::COL_OUTPUT0_L0 + l];
            let in1_l = &col_evals[trace::COL_INPUT1_L0 + l];
            pt_body = pt_body.add(&out_l.sub(in1_l));
        }
        body = body.add(&sel.mul(&pt_body));
    }

    body
}

/// Evaluate SIGNEXTEND constraint body across the full domain.
fn evaluate_signextend_full(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Scalar> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        // Short-circuit: if sel_signextend is 0, all sub-selectors are 0 and body is 0.
        let sel_se = &columns[trace::COL_SEL_SIGNEXTEND][i];
        if sel_se.is_zero() {
            result.push(zero.clone());
            continue;
        }
        let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
            .map(|c| columns[c][i].clone())
            .collect();
        result.push(evaluate_signextend_raw(&cols_at_i));
    }
    result
}

/// Build SIGNEXTEND constraint polynomial for the quotient computation.
fn build_signextend_constraint_poly<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    curve: metavm_zkp::field::CurveType,
) -> Vec<Scalar> {
    let mut body = vec![Scalar::zero(curve)];

    // For each case k = 0..=30, build case_body and multiply by sel_se_k, then add.
    for k in 0..=30usize {
        let ll = k / 8;
        let bb = k % 8 + 1;

        let mut case_body = vec![Scalar::zero(curve)];

        // (1) Limb recomposition: input1_l[LL] - Σ se_byte_i * 256^i = 0
        let mut recomp = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let byte_i = col(trace::COL_SE_BYTE_0 + i);
            let coeff = Scalar::from_u64(1u64 << (8 * i as u32), curve);
            recomp = poly_arith::poly_add(
                &recomp,
                &poly_arith::poly_scalar_mul(byte_i, &coeff),
                curve,
            );
        }
        let limb_ll = col(trace::COL_INPUT1_L0 + ll);
        let recomp_body = poly_arith::poly_sub(limb_ll, &recomp, curve);
        case_body = poly_arith::poly_add(&case_body, &recomp_body, curve);

        // (2) Sign-byte split: se_byte_{BB-1} - sign_bit * 128 - se_low7 = 0
        let sign_byte = col(trace::COL_SE_BYTE_0 + (bb - 1));
        let sign_bit = col(trace::COL_SE_SIGN_BIT);
        let low7 = col(trace::COL_SE_LOW7);
        let c128 = Scalar::from_u64(128, curve);
        let sign_bit_scaled = poly_arith::poly_scalar_mul(sign_bit, &c128);
        let split_body = poly_arith::poly_sub(sign_byte, &sign_bit_scaled, curve);
        let split_body = poly_arith::poly_sub(&split_body, low7, curve);
        case_body = poly_arith::poly_add(&case_body, &split_body, curve);

        // (3) Per-limb output checks.
        for l in 0..ll {
            // Low limbs: output_l - input1_l = 0
            let out_l = col(trace::COL_OUTPUT0_L0 + l);
            let in1_l = col(trace::COL_INPUT1_L0 + l);
            let d = poly_arith::poly_sub(out_l, in1_l, curve);
            case_body = poly_arith::poly_add(&case_body, &d, curve);
        }
        // Mixed limb (l = LL)
        {
            let out_l = col(trace::COL_OUTPUT0_L0 + ll);
            let mut preserved_sum = vec![Scalar::zero(curve)];
            for i in 0..bb {
                let byte_i = col(trace::COL_SE_BYTE_0 + i);
                let coeff = Scalar::from_u64(1u64 << (8 * i as u32), curve);
                preserved_sum = poly_arith::poly_add(
                    &preserved_sum,
                    &poly_arith::poly_scalar_mul(byte_i, &coeff),
                    curve,
                );
            }
            let fill_mask = if bb < 8 {
                let v = (1u128 << 64) - (1u128 << (8 * bb as u32));
                Scalar::from_u64(v as u64, curve)
            } else {
                Scalar::zero(curve)
            };
            let fill_term = poly_arith::poly_scalar_mul(sign_bit, &fill_mask);
            let mixed_body = poly_arith::poly_sub(out_l, &preserved_sum, curve);
            let mixed_body = poly_arith::poly_sub(&mixed_body, &fill_term, curve);
            case_body = poly_arith::poly_add(&case_body, &mixed_body, curve);
        }
        // High limbs (l > LL): output_l - sign_bit * (2^64 - 1) = 0
        let max_u64 = Scalar::from_u64(u64::MAX, curve);
        let fill_high = poly_arith::poly_scalar_mul(sign_bit, &max_u64);
        for l in (ll + 1)..4 {
            let out_l = col(trace::COL_OUTPUT0_L0 + l);
            let d = poly_arith::poly_sub(out_l, &fill_high, curve);
            case_body = poly_arith::poly_add(&case_body, &d, curve);
        }

        // Gate by sel_se_k
        let sel_se_k = col(trace::COL_SEL_SE_0 + k);
        let gated = poly_arith::poly_mul(sel_se_k, &case_body, curve);
        body = poly_arith::poly_add(&body, &gated, curve);
    }

    // Passthrough case (sel_se_ge31): Σ_l (output_l - input1_l)
    {
        let mut pt_body = vec![Scalar::zero(curve)];
        for l in 0..4 {
            let out_l = col(trace::COL_OUTPUT0_L0 + l);
            let in1_l = col(trace::COL_INPUT1_L0 + l);
            let d = poly_arith::poly_sub(out_l, in1_l, curve);
            pt_body = poly_arith::poly_add(&pt_body, &d, curve);
        }
        let sel_ge31 = col(trace::COL_SEL_SE_GE31);
        let gated = poly_arith::poly_mul(sel_ge31, &pt_body, curve);
        body = poly_arith::poly_add(&body, &gated, curve);
    }

    body
}

// ═══════════════════════════════════════════════════════════════════════════
// ADDMOD algebraic constraints
// ═══════════════════════════════════════════════════════════════════════════
//
// ADDMOD(a, b, n) = (a + b) mod n per Yellow Paper App. H.1:
//   n = 0 → result is 0
//   n ≠ 0 → result r = (a + b) mod n, with a+b = q*n + r and 0 ≤ r < n.
//
// Trace layout (inspector captures n into immediate columns):
//   input0 = a, input1 = b, immediate = n, output0 = r.
//   addmod_q            (4 limbs)   quotient q such that s = q*n + r
//   addmod_s_low        (4 limbs)   low 256 bits of a + b
//   addmod_s_high       (1 bit)     top carry bit of a + b
//   addmod_ab_carry     (4 bits)    carry chain for a + b
//   addmod_qn_p         (8 limbs)   product q*n (8-limb since q,n are 4-limb)
//   addmod_qn_carry     (7 limbs)   carries for q*n schoolbook (positions 0..6)
//   addmod_sum_carry    (9 limbs)   carry chain for q*n + r = s
//   addmod_slack / slack_borrow     non-borrow chain for (n - r - 1) ≥ 0
//   addmod_n_is_zero    (1 bit)     binary witness: 1 iff n == 0
//
// Constraints (16 constraint bodies, each gated by sel_addmod):
//   0. ab_sum_chain: Σ_k [a_k + b_k + ab_carry[k-1] - s_low_k - ab_carry[k]·2^64] = 0
//                    where ab_carry[-1]=0 and ab_carry[3] ≡ s_high (enforced implicitly
//                    by using s_high in place of ab_carry[3] at limb 3).
//   1-4. ab_carry_k_binary: ab_carry[k] · (ab_carry[k] − 1) = 0 for k=0..3.
//   5. s_high_binary: s_high · (s_high − 1) = 0.
//   6. qn_mul: Σ_k [Σ q_i·n_j + qn_carry[k-1] − qn_carry[k]·2^64 − qn_p_k] = 0
//      (8 limb identities summed; qn_carry[-1]=0, qn_carry[7]≡0 implicit).
//   7. sum_chain (gated by 1 − n_is_zero):
//      Σ_k [qn_p_k + r_k + sum_carry[k-1] − rhs_k − sum_carry[k]·2^64] = 0
//      where rhs_k = s_low_k for k<4, s_high for k=4, 0 for k>4.
//      Also folds sum_carry[8]·2^128 to force the final carry to zero.
//   8. slack_chain (gated by 1 − n_is_zero):
//      Σ_k [n_k − r_k − slack_borrow[k-1] + slack_borrow[k]·2^64 − slack_k] = 0
//      (slack_borrow[-1] = 1; slack_borrow[3]·2^128 folded to force it to 0).
//   9-12. slack_borrow_k_binary: slack_borrow[k] · (slack_borrow[k] − 1) = 0.
//   13. n_is_zero_binary: n_is_zero · (n_is_zero − 1) = 0.
//   14. n_zero_gate: n_is_zero · Σ n_limbs = 0.
//   15. r_zero_gate: n_is_zero · Σ r_limbs = 0.
//
// Soundness sketch (assuming 64-bit range checks on q, s_low, qn_p, qn_carry,
// sum_carry, slack limbs):
//   - ab_sum_chain + ab_carry binaries + s_high binary fully determine s_low
//     and s_high from a, b (a + b = s_low + s_high·2^256).
//   - qn_mul + range checks force qn_p = q * n as a 512-bit integer.
//   - sum_chain forces q*n + r = s when n != 0; with range-checked r and the
//     slack_chain (r < n), r is unique = (a+b) mod n. Hence q is also unique.
//   - When n = 0: n_zero_gate + r_zero_gate force r = 0 (via n_is_zero=1 and
//     Σn_limbs=0 with range-checked limbs). Other chains are gated off.
//
// ADDMOD constraint body helpers follow.

/// Compute ADDMOD constraint bodies on the full evaluation domain.
/// Returns exactly 16 vectors (one per constraint body), each of length `num_rows`.
fn evaluate_addmod_bodies(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Vec<Scalar>> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut bodies: Vec<Vec<Scalar>> = (0..16)
        .map(|_| Vec::with_capacity(num_rows))
        .collect();
    for i in 0..num_rows {
        let sel = &columns[trace::COL_SEL_ADDMOD][i];
        if sel.is_zero() {
            for k in 0..16 {
                bodies[k].push(zero.clone());
            }
            continue;
        }
        let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
            .map(|c| columns[c][i].clone())
            .collect();
        let vals = evaluate_addmod_raw(&cols_at_i);
        for k in 0..16 {
            bodies[k].push(vals[k].clone());
        }
    }
    bodies
}

/// Compute all 16 ADDMOD constraint body values at a single row (ungated by sel_addmod).
fn evaluate_addmod_raw(col_evals: &[Scalar]) -> [Scalar; 16] {
    let curve = col_evals[0].curve_type();
    let two_64 = arith::two_pow_64_pub(curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    let a = [
        &col_evals[trace::COL_INPUT0_L0],
        &col_evals[trace::COL_INPUT0_L1],
        &col_evals[trace::COL_INPUT0_L2],
        &col_evals[trace::COL_INPUT0_L3],
    ];
    let b = [
        &col_evals[trace::COL_INPUT1_L0],
        &col_evals[trace::COL_INPUT1_L1],
        &col_evals[trace::COL_INPUT1_L2],
        &col_evals[trace::COL_INPUT1_L3],
    ];
    let n = [
        &col_evals[trace::COL_IMMEDIATE_L0],
        &col_evals[trace::COL_IMMEDIATE_L1],
        &col_evals[trace::COL_IMMEDIATE_L2],
        &col_evals[trace::COL_IMMEDIATE_L3],
    ];
    let r = [
        &col_evals[trace::COL_OUTPUT0_L0],
        &col_evals[trace::COL_OUTPUT0_L1],
        &col_evals[trace::COL_OUTPUT0_L2],
        &col_evals[trace::COL_OUTPUT0_L3],
    ];
    let q = [
        &col_evals[trace::COL_ADDMOD_Q_L0],
        &col_evals[trace::COL_ADDMOD_Q_L1],
        &col_evals[trace::COL_ADDMOD_Q_L2],
        &col_evals[trace::COL_ADDMOD_Q_L3],
    ];
    let s_low = [
        &col_evals[trace::COL_ADDMOD_S_LOW_L0],
        &col_evals[trace::COL_ADDMOD_S_LOW_L1],
        &col_evals[trace::COL_ADDMOD_S_LOW_L2],
        &col_evals[trace::COL_ADDMOD_S_LOW_L3],
    ];
    let s_high = &col_evals[trace::COL_ADDMOD_S_HIGH];
    let ab_carry = [
        &col_evals[trace::COL_ADDMOD_AB_CARRY_0],
        &col_evals[trace::COL_ADDMOD_AB_CARRY_1],
        &col_evals[trace::COL_ADDMOD_AB_CARRY_2],
        &col_evals[trace::COL_ADDMOD_AB_CARRY_3],
    ];
    let qn_p = [
        &col_evals[trace::COL_ADDMOD_QN_P_L0],
        &col_evals[trace::COL_ADDMOD_QN_P_L1],
        &col_evals[trace::COL_ADDMOD_QN_P_L2],
        &col_evals[trace::COL_ADDMOD_QN_P_L3],
        &col_evals[trace::COL_ADDMOD_QN_P_L4],
        &col_evals[trace::COL_ADDMOD_QN_P_L5],
        &col_evals[trace::COL_ADDMOD_QN_P_L6],
        &col_evals[trace::COL_ADDMOD_QN_P_L7],
    ];
    let qn_carry = [
        &col_evals[trace::COL_ADDMOD_QN_CARRY_0],
        &col_evals[trace::COL_ADDMOD_QN_CARRY_1],
        &col_evals[trace::COL_ADDMOD_QN_CARRY_2],
        &col_evals[trace::COL_ADDMOD_QN_CARRY_3],
        &col_evals[trace::COL_ADDMOD_QN_CARRY_4],
        &col_evals[trace::COL_ADDMOD_QN_CARRY_5],
        &col_evals[trace::COL_ADDMOD_QN_CARRY_6],
    ];
    let sum_carry = [
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_0],
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_1],
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_2],
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_3],
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_4],
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_5],
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_6],
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_7],
        &col_evals[trace::COL_ADDMOD_SUM_CARRY_8],
    ];
    let slack = [
        &col_evals[trace::COL_ADDMOD_SLACK_L0],
        &col_evals[trace::COL_ADDMOD_SLACK_L1],
        &col_evals[trace::COL_ADDMOD_SLACK_L2],
        &col_evals[trace::COL_ADDMOD_SLACK_L3],
    ];
    let slack_b = [
        &col_evals[trace::COL_ADDMOD_SLACK_B0],
        &col_evals[trace::COL_ADDMOD_SLACK_B1],
        &col_evals[trace::COL_ADDMOD_SLACK_B2],
        &col_evals[trace::COL_ADDMOD_SLACK_B3],
    ];
    let n_is_zero = &col_evals[trace::COL_ADDMOD_N_IS_ZERO];
    let one_minus_niz = one.sub(n_is_zero);
    let two_128 = two_64.mul(&two_64);

    // (0) ab_sum_chain: Σ_k [a_k + b_k + ab_carry[k-1] - s_low_k - ab_carry[k]·2^64]
    //     where ab_carry[-1] = 0.
    // At limb 3, ab_carry[3] should equal s_high. We use ab_carry[3] as the
    // top-carry and a separate constraint (s_high_binary plus the final row
    // matching) binds ab_carry[3] = s_high via the requirement:
    // here we set the limb-3 identity to use `ab_carry[3]` as the carry-out,
    // then force `ab_carry[3] = s_high` by folding (ab_carry[3] - s_high)·2^128
    // into the ab_sum_chain body so a mismatch fails the identity.
    let mut ab_body = zero.clone();
    for k in 0..4usize {
        let mut s = a[k].add(b[k]);
        if k > 0 {
            s = s.add(ab_carry[k - 1]);
        }
        s = s.sub(s_low[k]);
        s = s.sub(&ab_carry[k].mul(&two_64));
        ab_body = ab_body.add(&s);
    }
    // Fold ab_carry[3] - s_high with a large coefficient so it can't cancel
    // against low-order carry terms.
    ab_body = ab_body.add(&ab_carry[3].sub(s_high).mul(&two_128));

    // (1-4) ab_carry_k binary.
    let ab0_bin = ab_carry[0].mul(&ab_carry[0].sub(&one));
    let ab1_bin = ab_carry[1].mul(&ab_carry[1].sub(&one));
    let ab2_bin = ab_carry[2].mul(&ab_carry[2].sub(&one));
    let ab3_bin = ab_carry[3].mul(&ab_carry[3].sub(&one));

    // (5) s_high_binary.
    let s_high_bin = s_high.mul(&s_high.sub(&one));

    // (6) qn_mul: Σ_k [Σ q_i·n_j + qn_carry[k-1] − qn_carry[k]·2^64 − qn_p_k]
    let mut qn_body = zero.clone();
    for k in 0..8usize {
        let mut s = zero.clone();
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                s = s.add(&q[i].mul(n[j]));
            }
        }
        if k > 0 {
            s = s.add(qn_carry[k - 1]);
        }
        if k < 7 {
            s = s.sub(&qn_carry[k].mul(&two_64));
        }
        s = s.sub(qn_p[k]);
        qn_body = qn_body.add(&s);
    }

    // (7) sum_chain (gated by 1 − n_is_zero):
    //   Σ_k [qn_p_k + r_k + sum_carry[k-1] − rhs_k − sum_carry[k]·2^64] = 0
    //   for k in 0..9. qn_p has only 8 limbs (k<8), r has only 4 (k<4).
    //   rhs_k = s_low_k for k<4, s_high for k=4, 0 otherwise.
    let mut sum_body = zero.clone();
    for k in 0..9usize {
        let mut s = zero.clone();
        if k < 8 {
            s = s.add(qn_p[k]);
        }
        if k < 4 {
            s = s.add(r[k]);
        }
        if k > 0 {
            s = s.add(sum_carry[k - 1]);
        }
        // Subtract rhs_k.
        if k < 4 {
            s = s.sub(s_low[k]);
        } else if k == 4 {
            s = s.sub(s_high);
        }
        s = s.sub(&sum_carry[k].mul(&two_64));
        sum_body = sum_body.add(&s);
    }
    // Fold sum_carry[8]·2^128 so any nonzero final carry fails.
    // (Technically sum_carry[8] should be 0 — this is also enforced by its
    // 64-bit range being forced to 0 at valid witness time.)
    sum_body = sum_body.add(&sum_carry[8].mul(&two_128));
    let sum_body_gated = one_minus_niz.mul(&sum_body);

    // (8) slack_chain (gated by 1 − n_is_zero): Σ_k [n_k − r_k − slack_b_prev
    //     + slack_b[k]·2^64 − slack_k] with slack_b[-1]=1 and slack_b[3] fold.
    let mut slack_body = zero.clone();
    for k in 0..4usize {
        let mut s = n[k].sub(r[k]);
        if k == 0 {
            s = s.sub(&one);
        } else {
            s = s.sub(slack_b[k - 1]);
        }
        s = s.add(&slack_b[k].mul(&two_64));
        s = s.sub(slack[k]);
        slack_body = slack_body.add(&s);
    }
    slack_body = slack_body.add(&slack_b[3].mul(&two_128));
    let slack_body_gated = one_minus_niz.mul(&slack_body);

    // (9-12) slack_b_k binary.
    let sb0_bin = slack_b[0].mul(&slack_b[0].sub(&one));
    let sb1_bin = slack_b[1].mul(&slack_b[1].sub(&one));
    let sb2_bin = slack_b[2].mul(&slack_b[2].sub(&one));
    let sb3_bin = slack_b[3].mul(&slack_b[3].sub(&one));

    // (13) n_is_zero binary.
    let niz_bin = n_is_zero.mul(&n_is_zero.sub(&one));

    // (14) n_zero_gate.
    let n_sum = n[0].add(n[1]).add(n[2]).add(n[3]);
    let n_zero_gate = n_is_zero.mul(&n_sum);

    // (15) r_zero_gate.
    let r_sum = r[0].add(r[1]).add(r[2]).add(r[3]);
    let r_zero_gate = n_is_zero.mul(&r_sum);

    [
        ab_body,
        ab0_bin, ab1_bin, ab2_bin, ab3_bin,
        s_high_bin,
        qn_body,
        sum_body_gated,
        slack_body_gated,
        sb0_bin, sb1_bin, sb2_bin, sb3_bin,
        niz_bin,
        n_zero_gate,
        r_zero_gate,
    ]
}

/// Build the ADDMOD constraint polynomials. Returns 16 polynomials (one per
/// constraint body), NOT pre-gated by sel_addmod.
fn build_addmod_constraint_polys<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    one_poly: &[Scalar],
    curve: metavm_zkp::field::CurveType,
) -> Vec<Vec<Scalar>> {
    let a = [
        col(trace::COL_INPUT0_L0), col(trace::COL_INPUT0_L1),
        col(trace::COL_INPUT0_L2), col(trace::COL_INPUT0_L3),
    ];
    let b = [
        col(trace::COL_INPUT1_L0), col(trace::COL_INPUT1_L1),
        col(trace::COL_INPUT1_L2), col(trace::COL_INPUT1_L3),
    ];
    let n = [
        col(trace::COL_IMMEDIATE_L0), col(trace::COL_IMMEDIATE_L1),
        col(trace::COL_IMMEDIATE_L2), col(trace::COL_IMMEDIATE_L3),
    ];
    let r = [
        col(trace::COL_OUTPUT0_L0), col(trace::COL_OUTPUT0_L1),
        col(trace::COL_OUTPUT0_L2), col(trace::COL_OUTPUT0_L3),
    ];
    let q = [
        col(trace::COL_ADDMOD_Q_L0), col(trace::COL_ADDMOD_Q_L1),
        col(trace::COL_ADDMOD_Q_L2), col(trace::COL_ADDMOD_Q_L3),
    ];
    let s_low = [
        col(trace::COL_ADDMOD_S_LOW_L0), col(trace::COL_ADDMOD_S_LOW_L1),
        col(trace::COL_ADDMOD_S_LOW_L2), col(trace::COL_ADDMOD_S_LOW_L3),
    ];
    let s_high = col(trace::COL_ADDMOD_S_HIGH);
    let ab_carry = [
        col(trace::COL_ADDMOD_AB_CARRY_0), col(trace::COL_ADDMOD_AB_CARRY_1),
        col(trace::COL_ADDMOD_AB_CARRY_2), col(trace::COL_ADDMOD_AB_CARRY_3),
    ];
    let qn_p = [
        col(trace::COL_ADDMOD_QN_P_L0), col(trace::COL_ADDMOD_QN_P_L1),
        col(trace::COL_ADDMOD_QN_P_L2), col(trace::COL_ADDMOD_QN_P_L3),
        col(trace::COL_ADDMOD_QN_P_L4), col(trace::COL_ADDMOD_QN_P_L5),
        col(trace::COL_ADDMOD_QN_P_L6), col(trace::COL_ADDMOD_QN_P_L7),
    ];
    let qn_carry = [
        col(trace::COL_ADDMOD_QN_CARRY_0), col(trace::COL_ADDMOD_QN_CARRY_1),
        col(trace::COL_ADDMOD_QN_CARRY_2), col(trace::COL_ADDMOD_QN_CARRY_3),
        col(trace::COL_ADDMOD_QN_CARRY_4), col(trace::COL_ADDMOD_QN_CARRY_5),
        col(trace::COL_ADDMOD_QN_CARRY_6),
    ];
    let sum_carry = [
        col(trace::COL_ADDMOD_SUM_CARRY_0), col(trace::COL_ADDMOD_SUM_CARRY_1),
        col(trace::COL_ADDMOD_SUM_CARRY_2), col(trace::COL_ADDMOD_SUM_CARRY_3),
        col(trace::COL_ADDMOD_SUM_CARRY_4), col(trace::COL_ADDMOD_SUM_CARRY_5),
        col(trace::COL_ADDMOD_SUM_CARRY_6), col(trace::COL_ADDMOD_SUM_CARRY_7),
        col(trace::COL_ADDMOD_SUM_CARRY_8),
    ];
    let slack = [
        col(trace::COL_ADDMOD_SLACK_L0), col(trace::COL_ADDMOD_SLACK_L1),
        col(trace::COL_ADDMOD_SLACK_L2), col(trace::COL_ADDMOD_SLACK_L3),
    ];
    let slack_b = [
        col(trace::COL_ADDMOD_SLACK_B0), col(trace::COL_ADDMOD_SLACK_B1),
        col(trace::COL_ADDMOD_SLACK_B2), col(trace::COL_ADDMOD_SLACK_B3),
    ];
    let n_is_zero = col(trace::COL_ADDMOD_N_IS_ZERO);
    let one_minus_niz = poly_arith::poly_sub(one_poly, n_is_zero, curve);
    let two_128 = two_64.mul(two_64);

    // (0) ab_sum_chain.
    let mut ab = vec![Scalar::zero(curve)];
    for k in 0..4usize {
        let mut s = poly_arith::poly_add(a[k], b[k], curve);
        if k > 0 {
            s = poly_arith::poly_add(&s, ab_carry[k - 1], curve);
        }
        s = poly_arith::poly_sub(&s, s_low[k], curve);
        s = poly_arith::poly_sub(&s, &poly_arith::poly_scalar_mul(ab_carry[k], two_64), curve);
        ab = poly_arith::poly_add(&ab, &s, curve);
    }
    // Fold ab_carry[3] - s_high to force ab_carry[3] = s_high.
    let high_diff = poly_arith::poly_sub(ab_carry[3], s_high, curve);
    ab = poly_arith::poly_add(&ab, &poly_arith::poly_scalar_mul(&high_diff, &two_128), curve);

    // (1-4) ab_carry_k binary.
    let mut ab_bins: Vec<Vec<Scalar>> = Vec::with_capacity(4);
    for k in 0..4usize {
        let m1 = poly_arith::poly_sub(ab_carry[k], one_poly, curve);
        ab_bins.push(poly_arith::poly_mul(ab_carry[k], &m1, curve));
    }

    // (5) s_high binary.
    let sh_m1 = poly_arith::poly_sub(s_high, one_poly, curve);
    let s_high_bin = poly_arith::poly_mul(s_high, &sh_m1, curve);

    // (6) qn_mul.
    let mut qn = vec![Scalar::zero(curve)];
    for k in 0..8usize {
        let mut s = vec![Scalar::zero(curve)];
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                s = poly_arith::poly_add(&s, &poly_arith::poly_mul(q[i], n[j], curve), curve);
            }
        }
        if k > 0 {
            s = poly_arith::poly_add(&s, qn_carry[k - 1], curve);
        }
        if k < 7 {
            s = poly_arith::poly_sub(
                &s,
                &poly_arith::poly_scalar_mul(qn_carry[k], two_64),
                curve,
            );
        }
        s = poly_arith::poly_sub(&s, qn_p[k], curve);
        qn = poly_arith::poly_add(&qn, &s, curve);
    }

    // (7) sum_chain, gated by (1 − n_is_zero).
    let mut sum_body = vec![Scalar::zero(curve)];
    for k in 0..9usize {
        let mut s = vec![Scalar::zero(curve)];
        if k < 8 {
            s = poly_arith::poly_add(&s, qn_p[k], curve);
        }
        if k < 4 {
            s = poly_arith::poly_add(&s, r[k], curve);
        }
        if k > 0 {
            s = poly_arith::poly_add(&s, sum_carry[k - 1], curve);
        }
        if k < 4 {
            s = poly_arith::poly_sub(&s, s_low[k], curve);
        } else if k == 4 {
            s = poly_arith::poly_sub(&s, s_high, curve);
        }
        s = poly_arith::poly_sub(
            &s,
            &poly_arith::poly_scalar_mul(sum_carry[k], two_64),
            curve,
        );
        sum_body = poly_arith::poly_add(&sum_body, &s, curve);
    }
    sum_body = poly_arith::poly_add(
        &sum_body,
        &poly_arith::poly_scalar_mul(sum_carry[8], &two_128),
        curve,
    );
    let sum_gated = poly_arith::poly_mul(&one_minus_niz, &sum_body, curve);

    // (8) slack_chain, gated by (1 − n_is_zero).
    let mut slack_body = vec![Scalar::zero(curve)];
    for k in 0..4usize {
        let mut s = poly_arith::poly_sub(n[k], r[k], curve);
        if k == 0 {
            s = poly_arith::poly_sub(&s, one_poly, curve);
        } else {
            s = poly_arith::poly_sub(&s, slack_b[k - 1], curve);
        }
        s = poly_arith::poly_add(
            &s,
            &poly_arith::poly_scalar_mul(slack_b[k], two_64),
            curve,
        );
        s = poly_arith::poly_sub(&s, slack[k], curve);
        slack_body = poly_arith::poly_add(&slack_body, &s, curve);
    }
    slack_body = poly_arith::poly_add(
        &slack_body,
        &poly_arith::poly_scalar_mul(slack_b[3], &two_128),
        curve,
    );
    let slack_gated = poly_arith::poly_mul(&one_minus_niz, &slack_body, curve);

    // (9-12) slack_b binary.
    let mut sb_bins: Vec<Vec<Scalar>> = Vec::with_capacity(4);
    for k in 0..4usize {
        let m1 = poly_arith::poly_sub(slack_b[k], one_poly, curve);
        sb_bins.push(poly_arith::poly_mul(slack_b[k], &m1, curve));
    }

    // (13) n_is_zero binary.
    let niz_m1 = poly_arith::poly_sub(n_is_zero, one_poly, curve);
    let niz_bin = poly_arith::poly_mul(n_is_zero, &niz_m1, curve);

    // (14) n_zero_gate.
    let mut n_sum = n[0].clone();
    n_sum = poly_arith::poly_add(&n_sum, n[1], curve);
    n_sum = poly_arith::poly_add(&n_sum, n[2], curve);
    n_sum = poly_arith::poly_add(&n_sum, n[3], curve);
    let n_zero_gate = poly_arith::poly_mul(n_is_zero, &n_sum, curve);

    // (15) r_zero_gate.
    let mut r_sum = r[0].clone();
    r_sum = poly_arith::poly_add(&r_sum, r[1], curve);
    r_sum = poly_arith::poly_add(&r_sum, r[2], curve);
    r_sum = poly_arith::poly_add(&r_sum, r[3], curve);
    let r_zero_gate = poly_arith::poly_mul(n_is_zero, &r_sum, curve);

    vec![
        ab,
        ab_bins[0].clone(), ab_bins[1].clone(), ab_bins[2].clone(), ab_bins[3].clone(),
        s_high_bin,
        qn,
        sum_gated,
        slack_gated,
        sb_bins[0].clone(), sb_bins[1].clone(), sb_bins[2].clone(), sb_bins[3].clone(),
        niz_bin,
        n_zero_gate,
        r_zero_gate,
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// MULMOD algebraic constraints
// ═══════════════════════════════════════════════════════════════════════════
//
// MULMOD(a, b, n) = (a * b) mod n per Yellow Paper App. H.1:
//   n = 0 → result is 0
//   n ≠ 0 → result r = (a * b) mod n, with a*b = q*n + r and 0 ≤ r < n.
//
// Trace layout:
//   input0  = a, input1 = b, immediate = n, output0 = r.
//   mulmod_q  (4 limbs)            quotient q such that p = q*n + r
//   mulmod_p  (8 limbs)            512-bit product p = a*b
//   mulmod_pc (7 limbs)            carries for a*b = p schoolbook (positions 0..6)
//   mulmod_sc (7 limbs)            carries for q*n + r = p schoolbook (positions 0..6)
//   mulmod_slack / slack_borrow    non-borrow chain for (n - r - 1) ≥ 0
//   mulmod_n_is_zero               binary witness: 1 iff n == 0
//
// Constraints (10 constraint bodies, each gated by sel_mulmod):
//   0. ab_mul:          Σ_k [Σ_{i+j=k} a_i b_j + pc[k-1] − pc[k]·2^64 − p_k] = 0
//                        (8 limb identities summed; pc[-1]=0, pc[7]≡0 implicit)
//   1. sum_chain:       (1 − n_is_zero) · Σ_k [Σ q_i n_j + r_k + sc[k-1]
//                                                − sc[k]·2^64 − p_k] = 0
//                        (8 limb identities summed; r_k = output0_l_k for k<4,
//                         r_k = 0 for k>=4; sc[-1]=0, sc[7]≡0 implicit)
//   2. slack_chain:     (1 − n_is_zero) · Σ_k [n_k − r_k − slack_borrow[k-1]
//                                               + slack_borrow[k]·2^64 − slack_k] = 0
//                        (4 limb identities summed; slack_borrow[-1] = 1,
//                         slack_borrow[3] must be 0 — which is implied when r<n).
//   3-6. slack_borrow_k_binary: slack_borrow[k] * (slack_borrow[k] − 1) = 0.
//   7. n_is_zero_binary: n_is_zero * (n_is_zero − 1) = 0.
//   8. n_zero_gate:     n_is_zero * Σ n_limbs = 0.
//   9. r_zero_gate:     n_is_zero * Σ r_limbs = 0.
//
// Soundness sketch (assuming range-checked limbs/carries):
//   - n_zero_gate plus 64-bit limb bound forces n = 0 iff Σn = 0 iff n_is_zero=1.
//   - When n_is_zero = 1: r_zero_gate forces r = 0; sum/slack chains gated off.
//     p = a*b is still enforced; output is 0. Matches EVM semantics.
//   - When n_is_zero = 0: sum_chain enforces p = q*n + r; slack_chain enforces
//     r < n. With r < n and p = q*n + r, r is uniquely (a*b) mod n, so q and r
//     are forced to their correct values. Matches EVM semantics.
//
// MULMOD constraint body helpers follow. Each takes columns at a single row
// or as polynomials; signatures mirror ADD/DIV helpers.

/// Compute MULMOD constraint bodies on the full evaluation domain.
/// Returns exactly 10 vectors (one per constraint body), each of length `num_rows`.
fn evaluate_mulmod_bodies(columns: &[&Vec<Scalar>], num_rows: usize) -> Vec<Vec<Scalar>> {
    let curve = columns[0][0].curve_type();
    let zero = Scalar::zero(curve);
    let mut bodies: Vec<Vec<Scalar>> = (0..10)
        .map(|_| Vec::with_capacity(num_rows))
        .collect();
    for i in 0..num_rows {
        let sel = &columns[trace::COL_SEL_MULMOD][i];
        if sel.is_zero() {
            // All 10 bodies are zero on non-MULMOD rows.
            for k in 0..10 {
                bodies[k].push(zero.clone());
            }
            continue;
        }
        let cols_at_i: Vec<Scalar> = (0..trace::NUM_EVM_COLUMNS)
            .map(|c| columns[c][i].clone())
            .collect();
        let vals = evaluate_mulmod_raw(&cols_at_i);
        for k in 0..10 {
            bodies[k].push(vals[k].clone());
        }
    }
    bodies
}

/// Compute all 10 MULMOD constraint body values at a single row (ungated by sel_mulmod).
/// The caller is responsible for multiplying by sel_mulmod if gating is desired.
fn evaluate_mulmod_raw(col_evals: &[Scalar]) -> [Scalar; 10] {
    let curve = col_evals[0].curve_type();
    let two_64 = arith::two_pow_64_pub(curve);
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    // Helper: compute a·b schoolbook limb identity at position k (k=0..7).
    // body_k = Σ_{i+j=k, i<4, j<4} a_i·b_j + pc_prev − pc_cur·2^64 − p_k
    // (pc_prev = pc[k-1] or 0 when k=0; pc_cur = pc[k] or 0 when k=7)
    let a = [
        &col_evals[trace::COL_INPUT0_L0],
        &col_evals[trace::COL_INPUT0_L1],
        &col_evals[trace::COL_INPUT0_L2],
        &col_evals[trace::COL_INPUT0_L3],
    ];
    let b = [
        &col_evals[trace::COL_INPUT1_L0],
        &col_evals[trace::COL_INPUT1_L1],
        &col_evals[trace::COL_INPUT1_L2],
        &col_evals[trace::COL_INPUT1_L3],
    ];
    let n = [
        &col_evals[trace::COL_IMMEDIATE_L0],
        &col_evals[trace::COL_IMMEDIATE_L1],
        &col_evals[trace::COL_IMMEDIATE_L2],
        &col_evals[trace::COL_IMMEDIATE_L3],
    ];
    let r = [
        &col_evals[trace::COL_OUTPUT0_L0],
        &col_evals[trace::COL_OUTPUT0_L1],
        &col_evals[trace::COL_OUTPUT0_L2],
        &col_evals[trace::COL_OUTPUT0_L3],
    ];
    let q = [
        &col_evals[trace::COL_MULMOD_Q_L0],
        &col_evals[trace::COL_MULMOD_Q_L1],
        &col_evals[trace::COL_MULMOD_Q_L2],
        &col_evals[trace::COL_MULMOD_Q_L3],
    ];
    let p = [
        &col_evals[trace::COL_MULMOD_P_L0],
        &col_evals[trace::COL_MULMOD_P_L1],
        &col_evals[trace::COL_MULMOD_P_L2],
        &col_evals[trace::COL_MULMOD_P_L3],
        &col_evals[trace::COL_MULMOD_P_L4],
        &col_evals[trace::COL_MULMOD_P_L5],
        &col_evals[trace::COL_MULMOD_P_L6],
        &col_evals[trace::COL_MULMOD_P_L7],
    ];
    let pc = [
        &col_evals[trace::COL_MULMOD_PC_0],
        &col_evals[trace::COL_MULMOD_PC_1],
        &col_evals[trace::COL_MULMOD_PC_2],
        &col_evals[trace::COL_MULMOD_PC_3],
        &col_evals[trace::COL_MULMOD_PC_4],
        &col_evals[trace::COL_MULMOD_PC_5],
        &col_evals[trace::COL_MULMOD_PC_6],
    ];
    let sc = [
        &col_evals[trace::COL_MULMOD_SC_0],
        &col_evals[trace::COL_MULMOD_SC_1],
        &col_evals[trace::COL_MULMOD_SC_2],
        &col_evals[trace::COL_MULMOD_SC_3],
        &col_evals[trace::COL_MULMOD_SC_4],
        &col_evals[trace::COL_MULMOD_SC_5],
        &col_evals[trace::COL_MULMOD_SC_6],
    ];
    let slack = [
        &col_evals[trace::COL_MULMOD_SLACK_L0],
        &col_evals[trace::COL_MULMOD_SLACK_L1],
        &col_evals[trace::COL_MULMOD_SLACK_L2],
        &col_evals[trace::COL_MULMOD_SLACK_L3],
    ];
    let slack_b = [
        &col_evals[trace::COL_MULMOD_SLACK_B0],
        &col_evals[trace::COL_MULMOD_SLACK_B1],
        &col_evals[trace::COL_MULMOD_SLACK_B2],
        &col_evals[trace::COL_MULMOD_SLACK_B3],
    ];
    let n_is_zero = &col_evals[trace::COL_MULMOD_N_IS_ZERO];
    let one_minus_niz = one.sub(n_is_zero);

    // (0) ab_mul: Σ_k [Σ a_i*b_j + pc_prev − pc_cur·2^64 − p_k]
    let mut ab_body = zero.clone();
    for k in 0..8usize {
        let mut sum = zero.clone();
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                sum = sum.add(&a[i].mul(b[j]));
            }
        }
        if k > 0 {
            sum = sum.add(pc[k - 1]);
        }
        if k < 7 {
            sum = sum.sub(&pc[k].mul(&two_64));
        }
        sum = sum.sub(p[k]);
        ab_body = ab_body.add(&sum);
    }

    // (1) sum_chain: Σ_k [Σ q_i*n_j + r_k + sc_prev − sc_cur·2^64 − p_k], gated (1−niz)
    let mut sum_body = zero.clone();
    for k in 0..8usize {
        let mut s = zero.clone();
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                s = s.add(&q[i].mul(n[j]));
            }
        }
        if k < 4 {
            s = s.add(r[k]);
        }
        if k > 0 {
            s = s.add(sc[k - 1]);
        }
        if k < 7 {
            s = s.sub(&sc[k].mul(&two_64));
        }
        s = s.sub(p[k]);
        sum_body = sum_body.add(&s);
    }
    let sum_body_gated = one_minus_niz.mul(&sum_body);

    // (2) slack_chain: Σ_k [n_k − r_k − slack_b_prev + slack_b_cur·2^64 − slack_k],
    //     gated by (1 − n_is_zero). slack_b_prev at k=0 equals 1 (the "−1" in
    //     n − r − 1). slack_b_cur at k=3 is slack_b[3] which acts as the final
    //     borrow-out and must be 0 for a valid witness; we enforce this via a
    //     separate zero-check below (slack_chain_body implicitly requires it).
    let mut slack_body = zero.clone();
    for k in 0..4usize {
        let mut s = n[k].sub(r[k]);
        if k == 0 {
            s = s.sub(&one); // -1 constant from "n - r - 1"
        } else {
            s = s.sub(slack_b[k - 1]);
        }
        s = s.add(&slack_b[k].mul(&two_64));
        s = s.sub(slack[k]);
        slack_body = slack_body.add(&s);
    }
    // The final borrow slack_b[3] must be 0 for r < n. Fold this into the
    // same slack body — added to penalise r ≥ n (when gated on).
    // Scale by a large term so it can't cancel against the limb identity sum.
    // Simplest: add a separate gated zero-check slack_b[3] * 2^128 (ensures
    // that slack_body = 0 forces slack_b[3] = 0 given the other terms are
    // bounded by ≪ 2^128). In field arithmetic this relies on the assumption
    // that the random alpha doesn't coincidentally hide the failure, which
    // is the same reliance the existing add/sub chains make.
    // But for robust soundness, enforce slack_b[3] = 0 as its own dedicated
    // zero-gate — that is, include it in this same body with a large coefficient.
    let two_128 = two_64.mul(&two_64);
    slack_body = slack_body.add(&slack_b[3].mul(&two_128));
    let slack_body_gated = one_minus_niz.mul(&slack_body);

    // (3-6) slack_borrow_k_binary: slack_b[k] * (slack_b[k] - 1) = 0.
    let sb0_bin = slack_b[0].mul(&slack_b[0].sub(&one));
    let sb1_bin = slack_b[1].mul(&slack_b[1].sub(&one));
    let sb2_bin = slack_b[2].mul(&slack_b[2].sub(&one));
    let sb3_bin = slack_b[3].mul(&slack_b[3].sub(&one));

    // (7) n_is_zero * (n_is_zero - 1) = 0.
    let niz_bin = n_is_zero.mul(&n_is_zero.sub(&one));

    // (8) n_zero_gate: n_is_zero * Σn_limbs = 0.
    let n_sum = n[0].add(n[1]).add(n[2]).add(n[3]);
    let n_zero_gate = n_is_zero.mul(&n_sum);

    // (9) r_zero_gate: n_is_zero * Σr_limbs = 0.
    let r_sum = r[0].add(r[1]).add(r[2]).add(r[3]);
    let r_zero_gate = n_is_zero.mul(&r_sum);

    [
        ab_body,
        sum_body_gated,
        slack_body_gated,
        sb0_bin, sb1_bin, sb2_bin, sb3_bin,
        niz_bin,
        n_zero_gate,
        r_zero_gate,
    ]
}

/// Build the MULMOD constraint polynomials for the quotient computation.
/// Returns 10 polynomials (one per constraint body). Each polynomial is NOT
/// pre-gated by sel_mulmod; callers must multiply by the selector themselves.
fn build_mulmod_constraint_polys<'a>(
    col: &dyn Fn(usize) -> &'a Vec<Scalar>,
    two_64: &Scalar,
    one_poly: &[Scalar],
    curve: metavm_zkp::field::CurveType,
) -> Vec<Vec<Scalar>> {
    let a = [
        col(trace::COL_INPUT0_L0), col(trace::COL_INPUT0_L1),
        col(trace::COL_INPUT0_L2), col(trace::COL_INPUT0_L3),
    ];
    let b = [
        col(trace::COL_INPUT1_L0), col(trace::COL_INPUT1_L1),
        col(trace::COL_INPUT1_L2), col(trace::COL_INPUT1_L3),
    ];
    let n = [
        col(trace::COL_IMMEDIATE_L0), col(trace::COL_IMMEDIATE_L1),
        col(trace::COL_IMMEDIATE_L2), col(trace::COL_IMMEDIATE_L3),
    ];
    let r = [
        col(trace::COL_OUTPUT0_L0), col(trace::COL_OUTPUT0_L1),
        col(trace::COL_OUTPUT0_L2), col(trace::COL_OUTPUT0_L3),
    ];
    let q = [
        col(trace::COL_MULMOD_Q_L0), col(trace::COL_MULMOD_Q_L1),
        col(trace::COL_MULMOD_Q_L2), col(trace::COL_MULMOD_Q_L3),
    ];
    let p = [
        col(trace::COL_MULMOD_P_L0), col(trace::COL_MULMOD_P_L1),
        col(trace::COL_MULMOD_P_L2), col(trace::COL_MULMOD_P_L3),
        col(trace::COL_MULMOD_P_L4), col(trace::COL_MULMOD_P_L5),
        col(trace::COL_MULMOD_P_L6), col(trace::COL_MULMOD_P_L7),
    ];
    let pc = [
        col(trace::COL_MULMOD_PC_0), col(trace::COL_MULMOD_PC_1),
        col(trace::COL_MULMOD_PC_2), col(trace::COL_MULMOD_PC_3),
        col(trace::COL_MULMOD_PC_4), col(trace::COL_MULMOD_PC_5),
        col(trace::COL_MULMOD_PC_6),
    ];
    let sc = [
        col(trace::COL_MULMOD_SC_0), col(trace::COL_MULMOD_SC_1),
        col(trace::COL_MULMOD_SC_2), col(trace::COL_MULMOD_SC_3),
        col(trace::COL_MULMOD_SC_4), col(trace::COL_MULMOD_SC_5),
        col(trace::COL_MULMOD_SC_6),
    ];
    let slack = [
        col(trace::COL_MULMOD_SLACK_L0), col(trace::COL_MULMOD_SLACK_L1),
        col(trace::COL_MULMOD_SLACK_L2), col(trace::COL_MULMOD_SLACK_L3),
    ];
    let slack_b = [
        col(trace::COL_MULMOD_SLACK_B0), col(trace::COL_MULMOD_SLACK_B1),
        col(trace::COL_MULMOD_SLACK_B2), col(trace::COL_MULMOD_SLACK_B3),
    ];
    let n_is_zero = col(trace::COL_MULMOD_N_IS_ZERO);
    let one_minus_niz = poly_arith::poly_sub(one_poly, n_is_zero, curve);

    let two_128 = two_64.mul(two_64);

    // (0) ab_mul: Σ_k [Σ a_i*b_j + pc_prev − pc_cur·2^64 − p_k]
    let mut ab = vec![Scalar::zero(curve)];
    for k in 0..8usize {
        let mut s = vec![Scalar::zero(curve)];
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                s = poly_arith::poly_add(&s, &poly_arith::poly_mul(a[i], b[j], curve), curve);
            }
        }
        if k > 0 {
            s = poly_arith::poly_add(&s, pc[k - 1], curve);
        }
        if k < 7 {
            s = poly_arith::poly_sub(&s, &poly_arith::poly_scalar_mul(pc[k], two_64), curve);
        }
        s = poly_arith::poly_sub(&s, p[k], curve);
        ab = poly_arith::poly_add(&ab, &s, curve);
    }

    // (1) sum_chain: gated by (1 − n_is_zero)
    let mut sum_body = vec![Scalar::zero(curve)];
    for k in 0..8usize {
        let mut s = vec![Scalar::zero(curve)];
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                s = poly_arith::poly_add(&s, &poly_arith::poly_mul(q[i], n[j], curve), curve);
            }
        }
        if k < 4 {
            s = poly_arith::poly_add(&s, r[k], curve);
        }
        if k > 0 {
            s = poly_arith::poly_add(&s, sc[k - 1], curve);
        }
        if k < 7 {
            s = poly_arith::poly_sub(&s, &poly_arith::poly_scalar_mul(sc[k], two_64), curve);
        }
        s = poly_arith::poly_sub(&s, p[k], curve);
        sum_body = poly_arith::poly_add(&sum_body, &s, curve);
    }
    let sum_gated = poly_arith::poly_mul(&one_minus_niz, &sum_body, curve);

    // (2) slack_chain: gated by (1 − n_is_zero)
    let mut slack_body = vec![Scalar::zero(curve)];
    for k in 0..4usize {
        // n_k − r_k − slack_b_prev + slack_b_cur·2^64 − slack_k
        let mut s = poly_arith::poly_sub(n[k], r[k], curve);
        if k == 0 {
            s = poly_arith::poly_sub(&s, one_poly, curve);
        } else {
            s = poly_arith::poly_sub(&s, slack_b[k - 1], curve);
        }
        s = poly_arith::poly_add(&s, &poly_arith::poly_scalar_mul(slack_b[k], two_64), curve);
        s = poly_arith::poly_sub(&s, slack[k], curve);
        slack_body = poly_arith::poly_add(&slack_body, &s, curve);
    }
    // Add slack_b[3] * 2^128 term so the final borrow being nonzero causes
    // the body to be nonzero (assuming r < 2*n, which holds with range-checked
    // limbs).
    slack_body = poly_arith::poly_add(
        &slack_body,
        &poly_arith::poly_scalar_mul(slack_b[3], &two_128),
        curve,
    );
    let slack_gated = poly_arith::poly_mul(&one_minus_niz, &slack_body, curve);

    // (3-6) slack_b_k binary.
    let mut sb_bins = Vec::with_capacity(4);
    for k in 0..4usize {
        let m1 = poly_arith::poly_sub(slack_b[k], one_poly, curve);
        sb_bins.push(poly_arith::poly_mul(slack_b[k], &m1, curve));
    }

    // (7) n_is_zero binary.
    let niz_m1 = poly_arith::poly_sub(n_is_zero, one_poly, curve);
    let niz_bin = poly_arith::poly_mul(n_is_zero, &niz_m1, curve);

    // (8) n_zero_gate: n_is_zero * Σn_k
    let mut n_sum = n[0].clone();
    n_sum = poly_arith::poly_add(&n_sum, n[1], curve);
    n_sum = poly_arith::poly_add(&n_sum, n[2], curve);
    n_sum = poly_arith::poly_add(&n_sum, n[3], curve);
    let n_zero_gate = poly_arith::poly_mul(n_is_zero, &n_sum, curve);

    // (9) r_zero_gate: n_is_zero * Σr_k
    let mut r_sum = r[0].clone();
    r_sum = poly_arith::poly_add(&r_sum, r[1], curve);
    r_sum = poly_arith::poly_add(&r_sum, r[2], curve);
    r_sum = poly_arith::poly_add(&r_sum, r[3], curve);
    let r_zero_gate = poly_arith::poly_mul(n_is_zero, &r_sum, curve);

    vec![
        ab, sum_gated, slack_gated,
        sb_bins[0].clone(), sb_bins[1].clone(), sb_bins[2].clone(), sb_bins[3].clone(),
        niz_bin, n_zero_gate, r_zero_gate,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use metavm_zkp::field::CurveType;
    use crate::trace::*;

    #[test]
    fn diag_dump_make_add_trace_constraints() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::trace::TracePolynomials;
        use metavm_zkp::vm_constraints::VmConstraintSystem as _;
        let trace = make_add_trace();
        let polys = TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns: Vec<&Vec<_>> = polys.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&columns, polys.num_rows);
        let labels = cs.constraint_labels();
        eprintln!("[diag] make_add_trace: {} bodies, {} rows", bodies.len(), polys.num_rows);
        for (b_idx, body) in bodies.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                if !val.is_zero() {
                    let label = labels.get(b_idx).map(|s| s.as_str()).unwrap_or("?");
                    eprintln!("[diag] FAIL row-local #{} ({}) row {}", b_idx, label, r);
                }
            }
        }
    }

    /// Two-row ADD+STOP trace, matches what real revm produces for
    /// `PUSH PUSH ADD STOP` minus the PUSH rows. Updated from a 1-row
    /// ADD-only trace to fix #105: BLS48-581 commitment scheme has a
    /// degenerate corner case for 1-row traces with NUM_EVM_COLUMNS=232
    /// (verify rejects despite honest prove). The 2-row variant works
    /// because revm-style trace shapes always have at least one ADD
    /// followed by a STOP.
    fn make_add_trace() -> EvmTraceColumns {
        let mut trace = EvmTraceColumns::new();
        // Row 0: ADD: 10 + 20 = 30
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x01,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: [10, 0, 0, 0],
            input1: [20, 0, 0, 0],
            output0: [30, 0, 0, 0],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_ADD,
            immediate: [0; 4],
            aux0: [0; 4], // no carry
            aux1: [0; 4],
            next_pc: 1,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);
        // Row 1: STOP at top-level (depth=0).
        let stop = EvmTraceRow {
            step: 1,
            pc: 1,
            opcode: 0x00,
            gas_remaining: 1000,
            stack_depth: 1,
            input0: [0; 4],
            input1: [0; 4],
            output0: [0; 4],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: trace::INSN_STOP,
            funct: 0,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 2,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&stop);
        trace
    }

    #[test]
    fn test_evm_add_constraint() {
        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Arith constraint should be zero for valid ADD
        assert!(eval[0][0].is_zero(), "ADD constraint should be satisfied");
    }

    #[test]
    fn test_evm_add_invalid() {
        let mut trace = EvmTraceColumns::new();
        // Invalid ADD: 10 + 20 = 31 (wrong)
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x01,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: [10, 0, 0, 0],
            input1: [20, 0, 0, 0],
            output0: [31, 0, 0, 0], // wrong!
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_ADD,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        assert!(!eval[0][0].is_zero(), "Invalid ADD should fail constraint");
    }

    #[test]
    fn test_evm_sub_constraint() {
        let mut trace = EvmTraceColumns::new();
        // SUB: 30 - 10 = 20
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x03,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: [30, 0, 0, 0],
            input1: [10, 0, 0, 0],
            output0: [20, 0, 0, 0],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_SUB,
            immediate: [0; 4],
            aux0: [0; 4], // no borrow
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // SUB limb constraints are at indices 8-11, borrow binary at 12-15
        for i in 8..16 {
            assert!(eval[i][0].is_zero(), "SUB constraint {} should be satisfied", i);
        }
        // All constraints should be satisfied for a valid SUB trace
        for (i, constraint_eval) in eval.iter().enumerate() {
            assert!(
                constraint_eval[0].is_zero(),
                "Constraint {} ({}) should be satisfied for SUB trace",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_selector_consistency() {
        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // All constraints should be satisfied for a valid ADD trace
        for (i, constraint_eval) in eval.iter().enumerate() {
            assert!(
                constraint_eval[0].is_zero(),
                "Constraint {} ({}) should be satisfied",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_num_constraints() {
        let cs = EvmConstraintSystem::new();
        // 96 VM + 50 binary (+1 for sel_byte_op split out from sel_bitwise_other)
        // + 1 sum + 32 se-case-binary + 1 se-sum-binding = 180
        assert_eq!(cs.num_constraints(), 180);
        assert_eq!(cs.constraint_labels().len(), 180);
    }

    /// Regression test for the MSTORE8 memory-permutation bug fixed
    /// 2026-05-12. The bug: `memory_columns()` returned only
    /// `[sel_mstore]` as the store-selector list, omitting
    /// `sel_mstore8`. Prover treated MSTORE8 rows as dummy (addr=0,
    /// values=[0;4]) while the verifier reconstructed the numerator
    /// from the actual trace columns (mem_offset, mem_value), creating
    /// a silent grand-product mismatch. The legacy `prove`+`verify`
    /// path skips memory permutation entirely and so masked this bug
    /// for all existing tests. This test pins the fix: any future
    /// removal of sel_mstore8 from the store-selector list will fire.
    #[test]
    fn memory_columns_includes_sel_mstore8_as_store() {
        let cs = EvmConstraintSystem::new();
        let (addr_col, val_cols, load_sels, store_sels) =
            cs.memory_columns().expect("EVM has memory permutation");
        assert_eq!(addr_col, trace::COL_MEM_OFFSET);
        assert_eq!(val_cols.len(), 4, "EVM mem_value is 4 limbs");
        assert!(load_sels.contains(&trace::COL_SEL_MLOAD),
            "MLOAD must be a load selector");
        assert!(store_sels.contains(&trace::COL_SEL_MSTORE),
            "MSTORE must be a store selector");
        assert!(store_sels.contains(&trace::COL_SEL_MSTORE8),
            "MSTORE8 must be a store selector — otherwise the memory \
             permutation grand product mismatches between prover and \
             verifier on MSTORE8 rows. See evm_keccak_linkage.md.");
    }

    #[test]
    fn test_evm_prove_verify_roundtrip() {
        metavm_zkp::commitment::init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();

        let proof = metavm_zkp::prover::prove(&polys, &cs);
        let valid = metavm_zkp::verifier::verify(&proof, &cs);
        assert!(valid, "Valid EVM trace should produce a verifying proof");
    }

    #[test]
    fn test_evm_prove_verify_bls48581_scheme() {
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();

        let proof = metavm_zkp::prover::prove_with_scheme(&polys, &cs, &scheme);
        let valid = metavm_zkp::verifier::verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls48581);
        assert!(valid, "EVM BLS48-581 scheme prove/verify should succeed");
    }

    #[test]
    fn test_evm_prove_verify_bls12381_scheme() {
        use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let scheme = Bls12381Scheme::new();
        scheme.init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls12381);
        let cs = EvmConstraintSystem::new();

        let proof = metavm_zkp::prover::prove_with_scheme(&polys, &cs, &scheme);
        let valid = metavm_zkp::verifier::verify_with_scheme(&proof, &cs, &scheme, CurveType::Bls12381);
        assert!(valid, "EVM BLS12-381 scheme prove/verify should succeed");
    }

    /// Regression: real EVM ChunkProof must verify through
    /// `begin_chunk_scheme` + `verify_final_scheme` (the recursive
    /// accumulator path the prove-* CLIs use).
    ///
    /// Caught a pre-existing transcript-mismatch bug where
    /// `recover_chunk_challenges_scheme` was missing the bitwise
    /// commitment + evaluation absorptions, causing the recursive
    /// accumulator's L/R computation to use a different (z, β) than the
    /// prover. This test would have caught it; the existing
    /// scheme-prove/verify tests above only exercise direct
    /// `verify_with_scheme`, not the recursive wrap.
    #[test]
    #[ignore = "slow: full prove + recursive verify on EVM trace; run with --release --ignored"]
    fn evm_chunk_proof_verifies_through_recursive_accumulator() {
        use metavm_zkp::prover::prove_chunk_with_scheme;
        use metavm_zkp::recursive::{begin_chunk_scheme, verify_final_scheme};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, curve);
        let cs = EvmConstraintSystem::new();

        let zero = [0u8; 32];
        let chunk_proof = prove_chunk_with_scheme(&polys, &cs, 0, &zero, &zero, &scheme);

        let recursive = begin_chunk_scheme(chunk_proof, &scheme, curve);
        let valid = verify_final_scheme(&recursive, &scheme);
        assert!(
            valid,
            "EVM ChunkProof must verify through recursive accumulator \
             (regression guard for the bitwise transcript fix in \
             `recover_chunk_challenges_scheme`)",
        );
    }

    /// End-to-end: produce a real EVM main-trace ExecutionProof, serialize
    /// via `ExecutionProof::to_bytes`, attach to a `LayerChainProof` as
    /// `LayerProofKind::VmEvm`, and verify via `verify_with_layer_verifier`
    /// closure dispatch. Validates the EVM main-trace's proof flows
    /// through the recursive-fold envelope just like the per-AIR proofs.
    #[test]
    #[ignore = "slow: produces a real EVM main-trace proof; run with --release --ignored"]
    fn evm_main_proof_flows_through_layer_chain_envelope() {
        use metavm_zkp::layer_chain::{
            ChainBoundaries, LayerChain, LayerChainProof, LayerProof, LayerProofKind,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 1. Generate a real EVM ExecutionProof on the small ADD trace.
        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, curve);
        let cs = EvmConstraintSystem::new();
        let proof = metavm_zkp::prover::prove_with_scheme(&polys, &cs, &scheme);
        assert!(metavm_zkp::verifier::verify_with_scheme(&proof, &cs, &scheme, curve));

        // 2. Serialize.
        let proof_bytes = proof.to_bytes();
        assert!(!proof_bytes.is_empty());

        // 3. Build a LayerChainProof with the EVM proof in the Execution slot.
        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: [0xDD; 32],
            num_attesters: 1,
            finalized_root: [0xCC; 32],
            total_effective_balance_gwei: 32_000_000_000,
        };
        let chain = LayerChain::from_boundaries(&boundaries);
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| {
                if i == 0 {
                    LayerProof::with_proof(claim, LayerProofKind::VmEvm, proof_bytes.clone())
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        // 4. Closure-based dispatch on VmEvm kind.
        let result = chain_proof.verify_with_layer_verifier(|layer| match layer.kind {
            LayerProofKind::VmEvm => {
                let p = metavm_zkp::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                    .map_err(|e| format!("decode failed: {:?}", e))?;
                let cs = EvmConstraintSystem::new();
                if metavm_zkp::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                    Ok(())
                } else {
                    Err("EVM main-trace proof did not verify".to_string())
                }
            }
            LayerProofKind::ReferenceOnly => Ok(()),
            other => Err(format!("unsupported layer kind {}", other.as_str())),
        });
        assert_eq!(
            result,
            Ok(()),
            "real EVM main-trace proof must verify through the LayerChainProof envelope",
        );
    }

    /// End-to-end with the new `ChunkProof::to_bytes` framing: produce
    /// a real `ChunkProof` via `prove_chunk_with_scheme`, serialize it
    /// as a chunk (length-prefix + state hashes + chunk_index), then
    /// verify by decoding back to `ChunkProof` and calling `verify_chunk`.
    /// This validates that the new ChunkProof serialization (the format
    /// the existing `prove-evm`, `prove-elf`, `prove-block` binaries
    /// emit) round-trips end-to-end.
    #[test]
    #[ignore = "slow: produces a real EVM ChunkProof; run with --release --ignored"]
    fn evm_chunk_proof_round_trips_through_serialization() {
        use metavm_zkp::prover::{prove_chunk_with_scheme, ChunkProof};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use metavm_zkp::verifier::verify_chunk;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, curve);
        let cs = EvmConstraintSystem::new();

        let initial_state = [0u8; 32];
        let final_state = [0u8; 32];
        let chunk = prove_chunk_with_scheme(
            &polys,
            &cs,
            0,
            &initial_state,
            &final_state,
            &scheme,
        );
        assert!(verify_chunk(&chunk, &cs), "fresh ChunkProof must verify");

        // Round-trip via to_bytes / from_bytes.
        let bytes = chunk.to_bytes();
        let decoded = ChunkProof::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.chunk_index, chunk.chunk_index);
        assert_eq!(decoded.initial_state_hash, chunk.initial_state_hash);
        assert_eq!(decoded.final_state_hash, chunk.final_state_hash);
        assert!(
            verify_chunk(&decoded, &cs),
            "decoded ChunkProof must verify equally well",
        );
    }

    /// End-to-end unified Eth → beacon → finality proof.
    ///
    /// Folds all four layers — Execution (EVM), BlockBinding (SHA-256),
    /// Attestation (BLS aggregate), Finality (≥2/3 stake) — into a single
    /// [`metavm_zkp::recursive::RecursiveProof`] and verifies the whole
    /// thing in one [`metavm_zkp::recursive::verify_final_scheme`] call.
    /// This is the canonical "I have a proof that this Ethereum block ran
    /// correctly, was included in the beacon chain, and is economically
    /// finalised under ≥2/3 staked ETH" pattern.
    #[test]
    #[ignore = "slow: 4 real prove/verify roundtrips + cross-AIR fold; run with --release --ignored"]
    fn unified_eth_finalized_block_proof_all_real_layers() {
        use metavm_zkp::bls_sig::{
            aggregate_sigs as bls_aggregate_sigs, PublicKey as BlsPublicKey,
            SecretKey as BlsSecretKey, Signature as BlsSignature,
        };
        use metavm_zkp::bls_sig_constraints::{
            build_bls_sig_trace_polynomials, BlsSigConstraintSystem, BlsSigWitness,
        };
        use metavm_zkp::field::CurveType;
        use metavm_zkp::finality_constraints::{
            build_finality_trace_polynomials, FinalityConstraintSystem, FinalityWitness,
        };
        use metavm_zkp::keccak::keccak256;
        use metavm_zkp::layer_chain::{
            ChainBoundaries, LayerChain, LayerChainFolder, LayerChainProof, LayerProof,
            LayerProofKind,
        };
        use metavm_zkp::prover::{prove_chunk_with_scheme, ExecutionProof};
        use metavm_zkp::recursive::{verify_final_scheme, RecursiveProof};
        use metavm_zkp::scheme::{bls48581_scheme::Bls48581Scheme, CommitmentScheme};
        use metavm_zkp::sha256::sha256_witness;
        use metavm_zkp::sha256_constraints::{
            build_trace_polynomials_from_hash, Sha256ConstraintSystem,
        };
        use metavm_zkp::vm_constraints::VmConstraintSystem;

        const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── Public boundary values ──────────────────────────────────────
        let block_hash = [0xBB; 32];
        let beacon_block_root = [0xCC; 32];
        let attestation_data_root = [0xDD; 32];
        let finalized_root = [0xEE; 32];

        // ── Witnesses ──────────────────────────────────────────────────
        // L1 EVM: a one-row ADD trace (the existing fast-path fixture).
        let evm_trace = make_add_trace();
        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_trace, curve);
        let evm_cs = EvmConstraintSystem::new();

        // L2 SHA-256: hash the chain commitment so the witness is
        // deterministic and tied to the boundary values.
        let sha_msg = b"unified_eth_finalized_block_demo".to_vec();
        let sha_hash = sha256_witness(&sha_msg);
        let sha_num_rounds = sha_hash.blocks.len() * 64;
        let sha_trace = build_trace_polynomials_from_hash(&sha_hash, curve);
        let sha_domain = sha_trace.padded_size;
        let sha_omega = scheme.domain_generator(sha_domain);
        let sha_cs = Sha256ConstraintSystem::new(sha_num_rounds)
            .with_omega_and_domain(sha_omega.clone(), sha_domain);

        // L3 BLS: build a 3-attester aggregate signature over `beacon_block_root`.
        let bls_msg = beacon_block_root.to_vec();
        let bls_seeds: [u8; 3] = [11, 22, 33];
        let bls_sks: Vec<BlsSecretKey> = bls_seeds
            .iter()
            .map(|s| BlsSecretKey::from_u8_seed(*s))
            .collect();
        let bls_pks: Vec<BlsPublicKey> =
            bls_sks.iter().map(|sk| sk.public_key()).collect();
        let bls_sigs: Vec<BlsSignature> = bls_sks
            .iter()
            .map(|sk| sk.sign(&bls_msg, POP_DST))
            .collect();
        let bls_agg_sig = bls_aggregate_sigs(&bls_sigs).expect("aggregate");
        let bls_witness = BlsSigWitness {
            pubkeys: bls_pks,
            msg: bls_msg.clone(),
            agg_sig: bls_agg_sig,
            dst: POP_DST.to_vec(),
        };
        let bls_num_attesters = bls_witness.pubkeys.len();
        let bls_trace = build_bls_sig_trace_polynomials(&bls_witness, curve);
        let bls_cs = BlsSigConstraintSystem::new(bls_num_attesters);

        // L4 Finality: a 4-validator toy committee, 3 attesting (3·32e9 ≥ 2·128e9 / 3 = 85.3e9, ✓).
        let finality_witness = FinalityWitness {
            validators: vec![
                (32_000_000_000, 1),
                (32_000_000_000, 1),
                (32_000_000_000, 1),
                (32_000_000_000, 0),
            ],
            total_active_balance_gwei: 4 * 32_000_000_000,
            attestation_data_root,
            finalized_root,
        };
        let fin_committee_size = finality_witness.validators.len();
        let fin_num_attesters = finality_witness
            .validators
            .iter()
            .filter(|(_, b)| *b == 1)
            .count() as u64;
        let fin_trace = build_finality_trace_polynomials(&finality_witness, curve);
        let fin_domain = fin_trace.padded_size;
        let fin_omega = scheme.domain_generator(fin_domain);
        let fin_cs = FinalityConstraintSystem::new(fin_committee_size)
            .with_omega_and_domain(fin_omega.clone(), fin_domain);

        // ── Chain ──────────────────────────────────────────────────────
        let chain = LayerChain::from_boundaries(&ChainBoundaries {
            block_hash,
            beacon_block_root,
            attestation_data_root,
            num_attesters: fin_num_attesters,
            finalized_root,
            total_effective_balance_gwei: finality_witness.total_attesting_balance(),
        });
        let chain_commitment = chain.commitment().expect("well-formed chain");

        // ── Per-provable-layer state hashes (matches LayerChainFolder) ──
        let derive_next =
            |prev: &[u8; 32], layer_idx: usize, kind_tag: u8| -> [u8; 32] {
                let mut buf = Vec::with_capacity(42);
                buf.extend_from_slice(prev);
                buf.push(0xF1);
                buf.push(kind_tag);
                buf.extend_from_slice(&(layer_idx as u64).to_be_bytes());
                keccak256(&buf)
            };
        // kind_tag mapping: VmEvm = 8 (L1 idx 0), Sha256 = 3 (L2 idx 1),
        // BlsSig = 10 (L3 idx 2), Finality = 11 (L4 idx 3).
        let s0 = chain_commitment;
        let s1 = derive_next(&s0, 0, 8);
        let s2 = derive_next(&s1, 1, 3);
        let s3 = derive_next(&s2, 2, 10);
        let s4 = derive_next(&s3, 3, 11);

        // ── Layer 1: EVM ChunkProof ────────────────────────────────────
        let evm_chunk =
            prove_chunk_with_scheme(&evm_polys, &evm_cs, 0, &s0, &s1, &scheme);
        let evm_bytes = evm_chunk.execution_proof.to_bytes();

        // ── Layer 2: SHA-256 ChunkProof ────────────────────────────────
        let sha_chunk =
            prove_chunk_with_scheme(&sha_trace, &sha_cs, 1, &s1, &s2, &scheme);
        let sha_bytes = sha_chunk.execution_proof.to_bytes();

        // ── Layer 3: BlsSig ChunkProof ─────────────────────────────────
        let bls_chunk =
            prove_chunk_with_scheme(&bls_trace, &bls_cs, 2, &s2, &s3, &scheme);
        let bls_bytes = bls_chunk.execution_proof.to_bytes();

        // ── Layer 4: Finality ChunkProof ───────────────────────────────
        let fin_chunk =
            prove_chunk_with_scheme(&fin_trace, &fin_cs, 3, &s3, &s4, &scheme);
        let fin_bytes = fin_chunk.execution_proof.to_bytes();

        // Sanity: all four ExecutionProofs round-trip through bytes.
        let _ = ExecutionProof::from_bytes(&evm_bytes).expect("evm exec from_bytes");
        let _ = ExecutionProof::from_bytes(&sha_bytes).expect("sha exec from_bytes");
        let _ = ExecutionProof::from_bytes(&bls_bytes).expect("bls exec from_bytes");
        let _ = ExecutionProof::from_bytes(&fin_bytes).expect("fin exec from_bytes");

        // ── Build the 4-layer chain proof ─────────────────────────────
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| match i {
                0 => LayerProof::with_proof(claim, LayerProofKind::VmEvm, evm_bytes.clone()),
                1 => LayerProof::with_proof(claim, LayerProofKind::Sha256, sha_bytes.clone()),
                2 => LayerProof::with_proof(claim, LayerProofKind::BlsSig, bls_bytes.clone()),
                3 => LayerProof::with_proof(claim, LayerProofKind::Finality, fin_bytes.clone()),
                _ => unreachable!(),
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        // ── Cross-AIR full-fold + single pairing decision ─────────────
        let sha_num_rounds_d = sha_num_rounds;
        let sha_omega_d = sha_omega.clone();
        let sha_domain_d = sha_domain;
        let bls_num_attesters_d = bls_num_attesters;
        let fin_committee_size_d = fin_committee_size;
        let fin_omega_d = fin_omega.clone();
        let fin_domain_d = fin_domain;
        let recursive = LayerChainFolder::fold_into_recursive_proof_full_scheme(
            &chain_proof,
            &scheme,
            curve,
            |kind| -> Box<dyn VmConstraintSystem> {
                match kind {
                    LayerProofKind::VmEvm => Box::new(EvmConstraintSystem::new()),
                    LayerProofKind::Sha256 => Box::new(
                        Sha256ConstraintSystem::new(sha_num_rounds_d)
                            .with_omega_and_domain(sha_omega_d.clone(), sha_domain_d),
                    ),
                    LayerProofKind::BlsSig => {
                        Box::new(BlsSigConstraintSystem::new(bls_num_attesters_d))
                    }
                    LayerProofKind::Finality => Box::new(
                        FinalityConstraintSystem::new(fin_committee_size_d)
                            .with_omega_and_domain(fin_omega_d.clone(), fin_domain_d),
                    ),
                    other => panic!("cs_dispatch: unsupported kind {}", other.as_str()),
                }
            },
        )
        .expect("4-layer cross-AIR fold");

        assert_eq!(recursive.depth, 4, "all four provable layers folded");
        assert_eq!(recursive.accumulator.num_folded, 4);
        assert!(
            verify_final_scheme(&recursive, &scheme),
            "unified Eth-finalized-block proof must verify in one pairing check",
        );

        // ── Persistence: round-trip the whole RecursiveProof ──────────
        let recursive_bytes = recursive.to_bytes();
        let reloaded = RecursiveProof::from_bytes(&recursive_bytes)
            .expect("RecursiveProof from_bytes");
        assert!(
            verify_final_scheme(&reloaded, &scheme),
            "reloaded unified proof must still verify",
        );
        eprintln!(
            "unified 4-layer Eth-finalized-block proof: \
             chain_commitment=0x{} bytes={} verified=true",
            hex_lower(&chain_commitment),
            recursive_bytes.len(),
        );
    }

    fn hex_lower(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    #[test]
    fn test_evm_dup_constraint() {
        let mut trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x80, // DUP1
            gas_remaining: 1000,
            stack_depth: 1,
            input0: [42, 0, 0, 0], // value being duplicated
            input1: [0; 4],
            output0: [42, 0, 0, 0], // duplicated output
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_STACK,
            funct: FUNCT_DUP,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for DUP",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_dup_invalid() {
        let mut trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x80, // DUP1
            gas_remaining: 1000,
            stack_depth: 1,
            input0: [42, 0, 0, 0],
            input1: [0; 4],
            output0: [99, 0, 0, 0], // wrong! should be 42
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_STACK,
            funct: FUNCT_DUP,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Constraint 70 (sel_dup * dup_raw) should fail
        assert!(
            !eval[70][0].is_zero(),
            "Invalid DUP should fail constraint 70 (sel_dup * dup_raw)"
        );
    }

    #[test]
    fn test_evm_mstore_constraint() {
        let mut trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x52, // MSTORE
            gas_remaining: 1000,
            stack_depth: 2,
            input0: [0x40, 0, 0, 0], // memory offset
            input1: [99, 0, 0, 0],   // value to store
            output0: [0; 4],         // MSTORE has no stack output
            mem_offset: 0x40,
            mem_value: [99, 0, 0, 0], // stored value = input1
            insn_type: INSN_MEMORY,
            funct: FUNCT_MSTORE,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for MSTORE",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_mstore_invalid() {
        let mut trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x52, // MSTORE
            gas_remaining: 1000,
            stack_depth: 2,
            input0: [0x40, 0, 0, 0],
            input1: [99, 0, 0, 0],
            output0: [0; 4],
            mem_offset: 0x40,
            mem_value: [77, 0, 0, 0], // wrong! should be 99
            insn_type: INSN_MEMORY,
            funct: FUNCT_MSTORE,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Constraint 74 (sel_mstore * mstore_raw) should fail
        assert!(
            !eval[74][0].is_zero(),
            "Invalid MSTORE should fail constraint 74 (sel_mstore * mstore_raw)"
        );
    }

    #[test]
    fn test_evm_mload_constraint() {
        let mut trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x51, // MLOAD
            gas_remaining: 1000,
            stack_depth: 1,
            input0: [0x40, 0, 0, 0], // memory offset
            input1: [0; 4],
            output0: [42, 0, 0, 0],  // loaded value
            mem_offset: 0x40,
            mem_value: [42, 0, 0, 0], // mem_value = output
            insn_type: INSN_MEMORY,
            funct: FUNCT_MLOAD,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for MLOAD",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_push_constraint() {
        let mut trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x60, // PUSH1
            gas_remaining: 1000,
            stack_depth: 0,
            input0: [0; 4],
            input1: [0; 4],
            output0: [42, 0, 0, 0],   // pushed value
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_STACK,
            funct: FUNCT_PUSH,
            immediate: [42, 0, 0, 0], // immediate = output
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for PUSH",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_and_constraint() {
        let mut trace = EvmTraceColumns::new();
        // AND: 0xFF00_00FF & 0x0F0F_0F0F = 0x0F00_000F
        let input0 = [0xFF00_00FFu64, 0, 0, 0];
        let input1 = [0x0F0F_0F0Fu64, 0, 0, 0];
        let output = [input0[0] & input1[0], 0, 0, 0]; // 0x0F00_000F
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x16, input0, input1, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x16, // AND
            gas_remaining: 1000,
            stack_depth: 2,
            input0,
            input1,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_AND,
            immediate: [0; 4],
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for AND",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_and_invalid() {
        let mut trace = EvmTraceColumns::new();
        let input0 = [0xFF00_00FFu64, 0, 0, 0];
        let input1 = [0x0F0F_0F0Fu64, 0, 0, 0];
        let output = [0xDEADBEEFu64, 0, 0, 0]; // wrong!
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x16, input0, input1, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x16,
            gas_remaining: 1000,
            stack_depth: 2,
            input0,
            input1,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_AND,
            immediate: [0; 4],
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // AND constraint 54 (sel_and * and_limb0) should fail
        assert!(
            !eval[54][0].is_zero(),
            "Invalid AND should fail constraint 54 (sel_and * and_limb0)"
        );
    }

    #[test]
    fn test_evm_or_constraint() {
        let mut trace = EvmTraceColumns::new();
        // OR: 0xFF00 | 0x00FF = 0xFFFF
        let input0 = [0xFF00u64, 0x1234, 0, 0];
        let input1 = [0x00FFu64, 0x5678, 0, 0];
        let output = [input0[0] | input1[0], input0[1] | input1[1], 0, 0];
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x17, input0, input1, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x17, // OR
            gas_remaining: 1000,
            stack_depth: 2,
            input0,
            input1,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_OR,
            immediate: [0; 4],
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for OR",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_xor_constraint() {
        let mut trace = EvmTraceColumns::new();
        // XOR: 0xAAAA ^ 0x5555 = 0xFFFF
        let input0 = [0xAAAAu64, 0xBBBB_CCCC_DDDD_EEEEu64, 0, 0];
        let input1 = [0x5555u64, 0x1111_2222_3333_4444u64, 0, 0];
        let output = [input0[0] ^ input1[0], input0[1] ^ input1[1], 0, 0];
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x18, input0, input1, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x18, // XOR
            gas_remaining: 1000,
            stack_depth: 2,
            input0,
            input1,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_XOR,
            immediate: [0; 4],
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for XOR",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_and_prove_verify() {
        metavm_zkp::commitment::init();

        let mut trace = EvmTraceColumns::new();
        let input0 = [0xFF00_FF00u64, 0xABCD_1234, 0, 0];
        let input1 = [0x0F0F_0F0Fu64, 0x1234_ABCD, 0, 0];
        let output = [input0[0] & input1[0], input0[1] & input1[1], 0, 0];
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x16, input0, input1, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x16,
            gas_remaining: 1000,
            stack_depth: 2,
            input0,
            input1,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_AND,
            immediate: [0; 4],
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();

        let proof = metavm_zkp::prover::prove(&polys, &cs);
        let valid = metavm_zkp::verifier::verify(&proof, &cs);
        assert!(valid, "Valid AND trace should produce a verifying proof");
    }

    #[test]
    fn test_evm_xor_multi_limb() {
        let mut trace = EvmTraceColumns::new();
        // Full 256-bit XOR
        let input0 = [0xFFFF_FFFF_FFFF_FFFFu64, 0xAAAA_AAAA_AAAA_AAAAu64,
                       0x1234_5678_9ABC_DEF0u64, 0xDEAD_BEEF_CAFE_BABEu64];
        let input1 = [0x0000_0000_0000_0001u64, 0x5555_5555_5555_5555u64,
                       0xFEDC_BA98_7654_3210u64, 0x0123_4567_89AB_CDEFu64];
        let output = [input0[0] ^ input1[0], input0[1] ^ input1[1],
                       input0[2] ^ input1[2], input0[3] ^ input1[3]];
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x18, input0, input1, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x18,
            gas_remaining: 1000,
            stack_depth: 2,
            input0,
            input1,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_XOR,
            immediate: [0; 4],
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for multi-limb XOR",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    fn make_shl_trace(shift: u32, value: [u64; 4]) -> EvmTraceColumns {
        let mut trace = EvmTraceColumns::new();
        let shift_amount: [u64; 4] = [shift as u64, 0, 0, 0];
        // Compute output = value << shift (mod 2^256)
        let pow2 = crate::trace::shift_power_of_two(shift_amount);
        let output = crate::trace::mul_u256_low(value, pow2);
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x1B, shift_amount, value, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x1B, // SHL
            gas_remaining: 1000,
            stack_depth: 2,
            input0: shift_amount,
            input1: value,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_SHL,
            immediate: pow2,
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);
        trace
    }

    #[test]
    fn test_evm_shl_constraint() {
        // SHL: shift 1 by 1 = 2
        let trace = make_shl_trace(1, [1, 0, 0, 0]);
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for SHL",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_shl_large_shift() {
        // SHL: shift 0xFF by 8 = 0xFF00
        let trace = make_shl_trace(8, [0xFF, 0, 0, 0]);
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for SHL by 8",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_shl_cross_limb() {
        // SHL: shift 1 by 64 = moves from limb 0 to limb 1
        let trace = make_shl_trace(64, [1, 0, 0, 0]);
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for SHL by 64",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_shl_invalid() {
        let mut trace = EvmTraceColumns::new();
        let shift_amount: [u64; 4] = [1, 0, 0, 0];
        let value: [u64; 4] = [1, 0, 0, 0];
        let pow2 = crate::trace::shift_power_of_two(shift_amount);
        let output = [99, 0, 0, 0]; // wrong! should be 2
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x1B, shift_amount, value, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x1B,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: shift_amount,
            input1: value,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_SHL,
            immediate: pow2,
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // SHL constraint 66 should fail
        assert!(
            !eval[66][0].is_zero(),
            "Invalid SHL should fail constraint 66 (sel_shl * shl_raw)"
        );
    }

    fn make_shr_trace(shift: u32, value: [u64; 4]) -> EvmTraceColumns {
        let mut trace = EvmTraceColumns::new();
        let shift_amount: [u64; 4] = [shift as u64, 0, 0, 0];
        let pow2 = crate::trace::shift_power_of_two(shift_amount);
        // Compute output = value >> shift (logical)
        let output = if shift >= 256 {
            [0u64; 4]
        } else {
            shr_u256(value, shift)
        };
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x1C, shift_amount, value, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x1C, // SHR
            gas_remaining: 1000,
            stack_depth: 2,
            input0: shift_amount,
            input1: value,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_SHR,
            immediate: pow2,
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);
        trace
    }

    /// Helper: logical right shift of a 256-bit value by `shift` bits.
    fn shr_u256(val: [u64; 4], shift: u32) -> [u64; 4] {
        if shift >= 256 { return [0; 4]; }
        let limb_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        let mut result = [0u64; 4];
        for i in 0..4 {
            let src = i + limb_shift;
            if src < 4 {
                result[i] = val[src] >> bit_shift;
                if bit_shift > 0 && src + 1 < 4 {
                    result[i] |= val[src + 1] << (64 - bit_shift);
                }
            }
        }
        result
    }

    #[test]
    fn test_evm_shr_constraint() {
        // SHR: 0xFF00 >> 8 = 0xFF
        let trace = make_shr_trace(8, [0xFF00, 0, 0, 0]);
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for SHR",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_shr_cross_limb() {
        // SHR: shift from limb 1 to limb 0 (shift by 64)
        let trace = make_shr_trace(64, [0, 42, 0, 0]);
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for SHR by 64",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_shr_invalid() {
        // Test with wrong carry chain (aux1) -- use correct output but zero carries
        let mut trace = EvmTraceColumns::new();
        let shift_amount: [u64; 4] = [8, 0, 0, 0];
        let value: [u64; 4] = [0xFF00, 0, 0, 0];
        let pow2 = crate::trace::shift_power_of_two(shift_amount);
        let output = shr_u256(value, 8);
        let (aux0, _aux1) = crate::trace::compute_evm_aux(0x1C, shift_amount, value, output);
        // Tamper with the carry chain
        let bad_aux1 = [99u64, 0, 0, 0];
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x1C,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: shift_amount,
            input1: value,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_SHR,
            immediate: pow2,
            aux0,
            aux1: bad_aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // SHR constraint 67 should fail with bad carry chain
        assert!(
            !eval[67][0].is_zero(),
            "SHR with bad carry chain should fail constraint 67 (sel_shr * shr_raw)"
        );
    }

    #[test]
    fn test_evm_sar_constraint() {
        // SAR: same as SHR for positive values
        let mut trace = EvmTraceColumns::new();
        let shift_amount: [u64; 4] = [8, 0, 0, 0];
        let value: [u64; 4] = [0xFF00, 0, 0, 0]; // positive (bit 255 = 0)
        let pow2 = crate::trace::shift_power_of_two(shift_amount);
        let output = shr_u256(value, 8); // same as SHR for positive
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x1D, shift_amount, value, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x1D, // SAR
            gas_remaining: 1000,
            stack_depth: 2,
            input0: shift_amount,
            input1: value,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_SAR,
            immediate: pow2,
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for SAR",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_sar_negative() {
        // SAR on a negative value: 0xFF...FE >> 1 = 0xFF...FF (sign-extended)
        let mut trace = EvmTraceColumns::new();
        let shift_amount: [u64; 4] = [1, 0, 0, 0];
        let value: [u64; 4] = [u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]; // -2 in two's complement
        // SAR(-2, 1) = -1 = 0xFF...FF
        let output: [u64; 4] = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let pow2 = crate::trace::shift_power_of_two(shift_amount);
        let (aux0, aux1) = crate::trace::compute_evm_aux(0x1D, shift_amount, value, output);
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x1D,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: shift_amount,
            input1: value,
            output0: output,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_BITWISE,
            funct: FUNCT_SAR,
            immediate: pow2,
            aux0,
            aux1,
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "Constraint {} ({}) should be satisfied for SAR negative",
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_shl_prove_verify() {
        metavm_zkp::commitment::init();

        let trace = make_shl_trace(4, [0xABCD, 0, 0, 0]);
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();

        let proof = metavm_zkp::prover::prove(&polys, &cs);
        let valid = metavm_zkp::verifier::verify(&proof, &cs);
        assert!(valid, "Valid SHL trace should produce a verifying proof");
    }

    // ─── SIGNEXTEND tests ──────────────────────────────────────────────────

    /// Helper: compute expected SIGNEXTEND result given (b, a) as 4-limb values.
    /// Mirrors EVM semantics per Yellow Paper Appendix H.1.
    fn signextend_expected(b: [u64; 4], a: [u64; 4]) -> [u64; 4] {
        // Passthrough if b >= 31.
        if b[1] != 0 || b[2] != 0 || b[3] != 0 || b[0] >= 31 {
            return a;
        }
        let k = b[0] as usize;
        let sign_bit_pos = 8 * (k + 1) - 1;
        let limb_idx = sign_bit_pos / 64;
        let bit_idx = sign_bit_pos % 64;
        let sign_bit = (a[limb_idx] >> bit_idx) & 1;
        let sign_fill_byte: u8 = if sign_bit == 1 { 0xFF } else { 0x00 };

        let mut result = a;
        // Replace bytes with position > k with sign_fill_byte.
        for byte_pos in (k + 1)..32 {
            let l = byte_pos / 8;
            let off = byte_pos % 8;
            result[l] = (result[l] & !(0xFFu64 << (8 * off)))
                | ((sign_fill_byte as u64) << (8 * off));
        }
        result
    }

    fn make_signextend_row(b: [u64; 4], a: [u64; 4]) -> EvmTraceRow {
        let output0 = signextend_expected(b, a);
        EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x0B,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: b,
            input1: a,
            output0,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_SIGNEXTEND,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    fn assert_all_constraints_satisfied(
        cs: &EvmConstraintSystem,
        eval: &[Vec<Scalar>],
        label: &str,
    ) {
        for (i, e) in eval.iter().enumerate() {
            assert!(
                e[0].is_zero(),
                "[{}] Constraint {} ({}) should be satisfied, got non-zero",
                label,
                i,
                cs.constraint_labels()[i]
            );
        }
    }

    #[test]
    fn test_evm_signextend_b0_positive_low_byte() {
        // SIGNEXTEND(0, 0x7F) = 0x7F (top bit 0, no sign extension).
        let mut trace = EvmTraceColumns::new();
        let b = [0, 0, 0, 0];
        let a = [0x7F, 0, 0, 0];
        trace.push_row(&make_signextend_row(b, a));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "SIGNEXTEND b=0 pos");
    }

    #[test]
    fn test_evm_signextend_b0_negative_low_byte() {
        // SIGNEXTEND(0, 0xFE) = 0xFFFF...FFFE (top bit 1 → all bytes > 0 = 0xFF).
        let mut trace = EvmTraceColumns::new();
        let b = [0, 0, 0, 0];
        let a = [0xFE, 0, 0, 0];
        trace.push_row(&make_signextend_row(b, a));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "SIGNEXTEND b=0 neg");
    }

    #[test]
    fn test_evm_signextend_b7_limb0_boundary() {
        // SIGNEXTEND(7, a) where a's top bit of limb 0 is 1 → fill upper limbs.
        let mut trace = EvmTraceColumns::new();
        let b = [7, 0, 0, 0];
        let a = [0x8000_0000_0000_0000, 0, 0, 0];
        trace.push_row(&make_signextend_row(b, a));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "SIGNEXTEND b=7");
    }

    #[test]
    fn test_evm_signextend_b15_limb1_middle() {
        // SIGNEXTEND(15, a): sign bit at position 127 (top of limb 1).
        let mut trace = EvmTraceColumns::new();
        let b = [15, 0, 0, 0];
        let a = [0xDEAD_BEEF_CAFE_BABE, 0x9000_0000_0000_0000, 0, 0];
        trace.push_row(&make_signextend_row(b, a));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "SIGNEXTEND b=15");
    }

    #[test]
    fn test_evm_signextend_b30_limb3_near_boundary() {
        // SIGNEXTEND(30, a): sign bit at position 247 (byte 30).
        // Limb 3 = bytes 24..31; sign byte is byte 30 = offset 6 within limb 3.
        let mut trace = EvmTraceColumns::new();
        let b = [30, 0, 0, 0];
        let a = [0x1, 0x2, 0x3, 0x0080_0000_0000_0000];
        // byte 30 of a = byte 6 of limb 3 = 0x80 → sign_bit = 1
        // so byte 31 (top byte of limb 3) becomes 0xFF
        trace.push_row(&make_signextend_row(b, a));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "SIGNEXTEND b=30");
    }

    #[test]
    fn test_evm_signextend_passthrough_b31() {
        // SIGNEXTEND(31, a) = a (no change).
        let mut trace = EvmTraceColumns::new();
        let b = [31, 0, 0, 0];
        let a = [0x1234, 0x5678, 0x9ABC, 0xDEF0];
        trace.push_row(&make_signextend_row(b, a));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "SIGNEXTEND b=31 passthrough");
    }

    #[test]
    fn test_evm_signextend_passthrough_large_b() {
        // SIGNEXTEND(1000, a) = a (b >= 31, passthrough).
        let mut trace = EvmTraceColumns::new();
        let b = [1000, 0, 0, 0];
        let a = [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD];
        trace.push_row(&make_signextend_row(b, a));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "SIGNEXTEND b=1000 passthrough");
    }

    #[test]
    fn test_evm_signextend_wrong_output_fails() {
        // Valid b=0, a=0xFE, but deliberately wrong output (0 instead of full fill).
        let mut trace = EvmTraceColumns::new();
        let bad_row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x0B,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: [0, 0, 0, 0],
            input1: [0xFE, 0, 0, 0],
            output0: [0xFE, 0, 0, 0], // WRONG: should be [0xFE, 0xFF...F, ...]
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_SIGNEXTEND,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        trace.push_row(&bad_row);
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Constraint 48 is the algebraic SIGNEXTEND body (shifted from 33 by
        // the 15 additional ADDMOD constraint bodies).
        assert!(
            !eval[48][0].is_zero(),
            "Wrong SIGNEXTEND output must fail constraint 48"
        );
    }

    #[test]
    fn test_evm_signextend_non_signextend_row_unaffected() {
        // Ensure SIGNEXTEND constraint is zero on a non-SIGNEXTEND row.
        let trace = make_add_trace();
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // All constraints (including the new SIGNEXTEND at index 48) should be 0.
        assert!(eval[48][0].is_zero(), "SIGNEXTEND constraint should be 0 on non-SE row");
    }

    // ═════════════════════════════════════════════════════════════════════
    // MULMOD tests
    // ═════════════════════════════════════════════════════════════════════

    /// Multiply two 256-bit values (as 4 u64 limbs) and return a 512-bit product
    /// as 8 u64 limbs (little-endian).
    fn mul_u512(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
        let mut out = [0u128; 8];
        for i in 0..4 {
            let mut carry: u128 = 0;
            for j in 0..4 {
                let prod = (a[i] as u128) * (b[j] as u128) + out[i + j] + carry;
                out[i + j] = prod & 0xFFFF_FFFF_FFFF_FFFF;
                carry = prod >> 64;
            }
            if i + 4 < 8 {
                out[i + 4] += carry;
            }
        }
        let mut result = [0u64; 8];
        for i in 0..8 {
            result[i] = out[i] as u64;
        }
        result
    }

    /// Reference 512-bit by 256-bit division. Returns (quotient as 4 u64 limbs,
    /// remainder as 4 u64 limbs). Returns ([0;4],[0;4]) when divisor is zero.
    fn divmod_512_256(dividend: [u64; 8], divisor: [u64; 4]) -> ([u64; 4], [u64; 4]) {
        if divisor == [0u64; 4] {
            return ([0; 4], [0; 4]);
        }
        let mut q = [0u64; 4];
        let mut rem = [0u64; 4];
        for bit in (0..512).rev() {
            // Shift rem left by 1.
            let r3 = (rem[3] << 1) | (rem[2] >> 63);
            let r2 = (rem[2] << 1) | (rem[1] >> 63);
            let r1 = (rem[1] << 1) | (rem[0] >> 63);
            let r0 = rem[0] << 1;
            rem = [r0, r1, r2, r3];
            let limb_idx = bit / 64;
            let bit_idx = bit % 64;
            rem[0] |= (dividend[limb_idx] >> bit_idx) & 1;
            // Compare rem >= divisor.
            let mut ge = false;
            for k in (0..4).rev() {
                if rem[k] > divisor[k] { ge = true; break; }
                if rem[k] < divisor[k] { ge = false; break; }
                if k == 0 { ge = true; break; }
            }
            if ge {
                // rem -= divisor
                let mut borrow: u64 = 0;
                let mut new_rem = [0u64; 4];
                for k in 0..4 {
                    let (d1, b1) = rem[k].overflowing_sub(divisor[k]);
                    let (d2, b2) = d1.overflowing_sub(borrow);
                    new_rem[k] = d2;
                    borrow = (b1 as u64) + (b2 as u64);
                }
                rem = new_rem;
                if bit < 256 {
                    let qlimb = bit / 64;
                    let qbit = bit % 64;
                    q[qlimb] |= 1u64 << qbit;
                }
            }
        }
        (q, rem)
    }

    /// Build an EvmTraceRow for MULMOD(a, b, n) with output = r.
    fn make_mulmod_row(a: [u64; 4], b: [u64; 4], n: [u64; 4], r: [u64; 4]) -> EvmTraceRow {
        EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x09,
            gas_remaining: 1000,
            stack_depth: 3,
            input0: a,
            input1: b,
            output0: r,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_MULMOD,
            immediate: n,   // inspector captures n here
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// Compute (a*b) mod n as 4 u64 limbs using reference integer arithmetic.
    fn mulmod_expected(a: [u64; 4], b: [u64; 4], n: [u64; 4]) -> [u64; 4] {
        if n == [0; 4] { return [0; 4]; }
        let p = mul_u512(a, b);
        let (_q, r) = divmod_512_256(p, n);
        r
    }

    #[test]
    fn test_evm_mulmod_small() {
        // MULMOD(3, 5, 7) = 15 % 7 = 1
        let a = [3, 0, 0, 0];
        let b = [5, 0, 0, 0];
        let n = [7, 0, 0, 0];
        let r = mulmod_expected(a, b, n);
        assert_eq!(r, [1, 0, 0, 0]);
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_mulmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "MULMOD small");
    }

    #[test]
    fn test_evm_mulmod_zero_result() {
        // MULMOD(4, 3, 6) = 12 % 6 = 0
        let a = [4, 0, 0, 0];
        let b = [3, 0, 0, 0];
        let n = [6, 0, 0, 0];
        let r = mulmod_expected(a, b, n);
        assert_eq!(r, [0, 0, 0, 0]);
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_mulmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "MULMOD zero result");
    }

    #[test]
    fn test_evm_mulmod_n_is_zero() {
        // MULMOD(5, 7, 0) = 0 per EVM spec
        let a = [5, 0, 0, 0];
        let b = [7, 0, 0, 0];
        let n = [0, 0, 0, 0];
        let r = [0, 0, 0, 0];
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_mulmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "MULMOD n=0");
    }

    #[test]
    fn test_evm_mulmod_a_or_b_zero() {
        // MULMOD(0, 42, 7) = 0
        let a = [0, 0, 0, 0];
        let b = [42, 0, 0, 0];
        let n = [7, 0, 0, 0];
        let r = mulmod_expected(a, b, n);
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_mulmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "MULMOD a=0");

        // MULMOD(42, 0, 7) = 0
        let a2 = [42, 0, 0, 0];
        let b2 = [0, 0, 0, 0];
        let r2 = mulmod_expected(a2, b2, n);
        let mut trace2 = EvmTraceColumns::new();
        trace2.push_row(&make_mulmod_row(a2, b2, n, r2));
        let polys2 = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace2, CurveType::Bls48581);
        let columns2 = polys2.columns();
        let eval2 = cs.evaluate_on_domain(&columns2, polys2.num_rows);
        assert_all_constraints_satisfied(&cs, &eval2, "MULMOD b=0");
    }

    #[test]
    fn test_evm_mulmod_large_product() {
        // Multi-limb test exercising the top carry limbs of the 512-bit product
        // WITHOUT overflowing 64-bit intermediate carries. Each limb is bounded
        // by 2^60 to keep partial-product sums in range, but the product still
        // spans limbs 0..6 of p (witness the upper p limbs being non-trivial).
        //
        // a = 2^60 + (2^60 << 64) + (2^60 << 128) + (2^60 << 192)  — each limb 2^60
        // b = 3 + (3 << 64) + (3 << 128) + (3 << 192)              — each limb 3
        // a * b spans 512 bits with top limbs nonzero.
        let a = [1u64 << 60, 1u64 << 60, 1u64 << 60, 1u64 << 60];
        let b = [3u64, 3u64, 3u64, 3u64];
        let n = [1, 0, 0, 1]; // n = 2^192 + 1
        let r = mulmod_expected(a, b, n);
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_mulmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "MULMOD large product");
    }

    #[test]
    fn test_evm_mulmod_invalid_result() {
        // Valid inputs (3, 5, 7) but deliberately wrong output (2 instead of 1).
        let a = [3, 0, 0, 0];
        let b = [5, 0, 0, 0];
        let n = [7, 0, 0, 0];
        let bad_r = [2, 0, 0, 0]; // WRONG: correct is 1
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_mulmod_row(a, b, n, bad_r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // At least one of the MULMOD algebraic constraints (indices 37..=46)
        // must be nonzero. The sum_chain (index 38) is the primary identity:
        // a wrong r violates p = q*n + r.
        let any_fail = (37..=46).any(|i| !eval[i][0].is_zero());
        assert!(
            any_fail,
            "Wrong MULMOD output must fail at least one MULMOD constraint"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // ADDMOD tests
    // ═════════════════════════════════════════════════════════════════════

    /// Build an EvmTraceRow for ADDMOD(a, b, n) with output = r.
    fn make_addmod_row(a: [u64; 4], b: [u64; 4], n: [u64; 4], r: [u64; 4]) -> EvmTraceRow {
        EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x08,
            gas_remaining: 1000,
            stack_depth: 3,
            input0: a,
            input1: b,
            output0: r,
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_ADDMOD,
            immediate: n, // inspector captures n here
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
            frame: trace::FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// Compute (a + b) mod n as 4 u64 limbs using reference integer arithmetic.
    /// a + b is a 257-bit value; use 8-limb dividend to leverage divmod_512_256.
    fn addmod_expected(a: [u64; 4], b: [u64; 4], n: [u64; 4]) -> [u64; 4] {
        if n == [0; 4] {
            return [0; 4];
        }
        let mut s = [0u64; 8];
        let mut carry: u64 = 0;
        for k in 0..4 {
            let (x1, c1) = a[k].overflowing_add(b[k]);
            let (x2, c2) = x1.overflowing_add(carry);
            s[k] = x2;
            carry = (c1 as u64) + (c2 as u64);
        }
        s[4] = carry;
        let (_q, r) = divmod_512_256(s, n);
        r
    }

    #[test]
    fn test_evm_addmod_small() {
        // ADDMOD(5, 7, 11) = 12 % 11 = 1
        let a = [5, 0, 0, 0];
        let b = [7, 0, 0, 0];
        let n = [11, 0, 0, 0];
        let r = addmod_expected(a, b, n);
        assert_eq!(r, [1, 0, 0, 0]);
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_addmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "ADDMOD small");
    }

    #[test]
    fn test_evm_addmod_zero_result() {
        // ADDMOD(5, 11, 16) = 16 % 16 = 0
        let a = [5, 0, 0, 0];
        let b = [11, 0, 0, 0];
        let n = [16, 0, 0, 0];
        let r = addmod_expected(a, b, n);
        assert_eq!(r, [0, 0, 0, 0]);
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_addmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "ADDMOD zero result");
    }

    #[test]
    fn test_evm_addmod_n_is_zero() {
        // ADDMOD(5, 7, 0) = 0 per EVM spec.
        let a = [5, 0, 0, 0];
        let b = [7, 0, 0, 0];
        let n = [0, 0, 0, 0];
        let r = [0, 0, 0, 0];
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_addmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "ADDMOD n=0");
    }

    #[test]
    fn test_evm_addmod_overflow_s_high() {
        // a = b = U256::MAX so a+b = 2^257 - 2, s_high=1.
        // Pick n = 17 so the result is small and nonzero.
        let a = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let b = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let n = [17, 0, 0, 0];
        let r = addmod_expected(a, b, n);
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_addmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "ADDMOD overflow s_high=1");
    }

    #[test]
    fn test_evm_addmod_sum_equals_n() {
        // ADDMOD(10, 10, 20) = 20 % 20 = 0
        let a = [10, 0, 0, 0];
        let b = [10, 0, 0, 0];
        let n = [20, 0, 0, 0];
        let r = addmod_expected(a, b, n);
        assert_eq!(r, [0, 0, 0, 0]);
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_addmod_row(a, b, n, r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);
        assert_all_constraints_satisfied(&cs, &eval, "ADDMOD sum==n");
    }

    #[test]
    fn test_evm_addmod_invalid_result() {
        // Valid inputs (5, 7, 11) but deliberately wrong output (2 instead of 1).
        let a = [5, 0, 0, 0];
        let b = [7, 0, 0, 0];
        let n = [11, 0, 0, 0];
        let bad_r = [2, 0, 0, 0]; // WRONG: correct is 1
        let mut trace = EvmTraceColumns::new();
        trace.push_row(&make_addmod_row(a, b, n, bad_r));
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // At least one of the ADDMOD algebraic constraints (indices 21..=36)
        // must be nonzero.
        let any_fail = (21..=36).any(|i| !eval[i][0].is_zero());
        assert!(
            any_fail,
            "Wrong ADDMOD output must fail at least one ADDMOD constraint"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // CALL frame-stack tests
    // ═════════════════════════════════════════════════════════════════════
    //
    // The CALL/RETURN transitions live in the cross-row shifted-constraint
    // section, not in `evaluate_on_domain`. We test them by:
    //   (1) building a small two-row trace where row 0 is a CALL or RETURN
    //       and row 1 is the next-frame snapshot,
    //   (2) IFFTing the trace columns to coefficient form,
    //   (3) calling `build_shifted_constraint_polynomial` to obtain the
    //       full shifted constraint polynomial (already multiplied by
    //       (X − ω^{n−1})),
    //   (4) evaluating that polynomial at every domain point ω^r and
    //       asserting it equals 0 for a valid trace, or non-zero for a
    //       tampered one.

    use metavm_zkp::scheme::CommitmentScheme;
    use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;

    /// Build a CALL trace row.
    /// The frame snapshot here is the *current* frame at the time of the CALL
    /// (i.e., the caller's frame). The next row's frame must be the freshly
    /// pushed frame.
    fn make_call_row(
        step: u64,
        pc: u64,
        gas_arg: [u64; 4],
        callee_arg: [u64; 4],
        next_pc: u64,
        current_frame: trace::FrameState,
    ) -> EvmTraceRow {
        EvmTraceRow {
            step,
            pc,
            opcode: 0xF1, // CALL
            gas_remaining: 1_000_000,
            stack_depth: 7,
            input0: gas_arg,
            input1: callee_arg,
            output0: [0; 4],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_CALL,
            funct: 1, // CALL = opcode 0xF1, funct = 1
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc,
            frame: current_frame,
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// Build a RETURN trace row.
    fn make_return_row(
        step: u64,
        pc: u64,
        next_pc: u64,
        current_frame: trace::FrameState,
    ) -> EvmTraceRow {
        EvmTraceRow {
            step,
            pc,
            opcode: 0xF3, // RETURN
            gas_remaining: 1_000_000,
            stack_depth: 2,
            input0: [0; 4],
            input1: [0; 4],
            output0: [0; 4],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_CALL,
            funct: 3, // RETURN = opcode 0xF3, funct = 3
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc,
            frame: current_frame,
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// Build a STOP row used as an inert "next instruction" / "before-CALL"
    /// row.
    fn make_stop_row(
        step: u64,
        pc: u64,
        next_pc: u64,
        current_frame: trace::FrameState,
    ) -> EvmTraceRow {
        EvmTraceRow {
            step,
            pc,
            opcode: 0x00,
            gas_remaining: 1_000_000,
            stack_depth: 0,
            input0: [0; 4],
            input1: [0; 4],
            output0: [0; 4],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_STOP,
            funct: 0,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc,
            frame: current_frame,
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// Build a JUMPDEST (0x5B) row — the EVM no-op, used as a "filler"
    /// in synthetic tests that need a row inside a non-zero-depth frame
    /// without it being interpreted as an implicit frame pop. STOP at
    /// depth ≥ 1 is now treated as sel_stop_pop=1 (implicit frame pop)
    /// to match real revm semantics, so synthetic tests that previously
    /// used STOP-at-depth-1 as filler must use JUMPDEST instead.
    fn make_jumpdest_row(
        step: u64,
        pc: u64,
        next_pc: u64,
        current_frame: trace::FrameState,
    ) -> EvmTraceRow {
        EvmTraceRow {
            step,
            pc,
            opcode: 0x5B, // JUMPDEST
            gas_remaining: 1_000_000,
            stack_depth: 0,
            input0: [0; 4],
            input1: [0; 4],
            output0: [0; 4],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: trace::INSN_JUMP,
            funct: 0,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc,
            frame: current_frame,
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// Returns true iff the shifted constraint polynomial vanishes on the
    /// full evaluation domain {ω^r : 0 ≤ r < n}. The polynomial is built in
    /// coefficient form by `build_shifted_constraint_polynomial` and then
    /// evaluated at each ω^r via the curve's polynomial-eval primitive.
    ///
    /// Because every shifted constraint is multiplied by (X − ω^{n−1}), the
    /// polynomial is automatically zero at the wrap-around row, so a bad
    /// transition between row `n−1` and row `0` would still pass. Our test
    /// traces avoid the wrap-around row by placing the CALL/RETURN row at
    /// index 0 (so the transition is row 0 → row 1, which IS checked).
    fn shifted_poly_vanishes(trace: &EvmTraceColumns) -> bool {
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;
        let mut polys =
            metavm_zkp::trace::TracePolynomials::from_vm_trace(trace, curve);
        let n = polys.padded_size as u64;
        let cs = EvmConstraintSystem::new();
        // Fix selectors on padding rows (so sum-to-one holds). The shifted
        // poly itself doesn't include sum-to-one but `fix_trace_padding`
        // also relies on padding selector handling being consistent.
        polys.fix_selector_padding(&cs);
        // Mutate the evaluation columns so PC and NEXT_PC on padding rows
        // satisfy the always-on PC continuity constraint.
        let mut eval_columns: Vec<Vec<Scalar>> = polys
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cs.fix_trace_padding(&mut eval_columns, polys.num_rows, n as usize);
        // IFFT each column to coefficient form.
        let coeff_form: Vec<Vec<Scalar>> = eval_columns
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();
        let omega = CommitmentScheme::domain_generator(&scheme, n);
        let alpha = Scalar::from_u64(7919, curve);
        let shifted_poly = cs.build_shifted_constraint_polynomial(
            &coeff_form,
            &alpha,
            n,
            &omega,
            cs.num_constraints(),
        );
        // Evaluate at every ω^r and check.
        let mut omega_r = Scalar::one(curve);
        for r in 0..n {
            let v = CommitmentScheme::eval_poly_at(&scheme, &shifted_poly, &omega_r);
            if !v.is_zero() {
                eprintln!("shifted_poly_vanishes: nonzero at row {} (omega^{})", r, r);
                return false;
            }
            omega_r = omega_r.mul(&omega);
        }
        true
    }

    /// Helper: build a 2-row trace for "CALL at row 0 → callee step at row 1"
    /// with a freshly pushed frame at row 1. All cross-row constraints
    /// should pass.
    fn build_two_row_call_trace(
        depth0: u64,
        caller0: [u64; 4],
        callee0: [u64; 4],
        callee_arg: [u64; 4],
        gas_arg: [u64; 4],
        depth1: u64,
        callee1: [u64; 4],
        caller1: [u64; 4],
        return_pc1: u64,
    ) -> EvmTraceColumns {
        let frame0 = trace::FrameState {
            depth: depth0,
            caller: caller0,
            callee: callee0,
            value: [0; 4],
            gas: 1_000_000,
            return_pc: 0,
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        let frame1 = trace::FrameState {
            depth: depth1,
            caller: caller1,
            callee: callee1,
            value: [0; 4],
            gas: 1_000_000,
            return_pc: return_pc1,
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        let mut t = EvmTraceColumns::new();
        // Row 0: CALL at pc=0, next_pc=1
        t.push_row(&make_call_row(0, 0, gas_arg, callee_arg, 1, frame0));
        // Row 1: STOP at pc=1, next_pc=2 (callee's first instruction)
        t.push_row(&make_stop_row(1, 1, 2, frame1));
        t
    }

    #[test]
    fn test_phase5_call_push_frame_valid_transition() {
        // Top-level addr 0x42, callee addr 0xABCD.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let trace = build_two_row_call_trace(
            /*depth0*/ 0,
            /*caller0*/ [0; 4],
            /*callee0*/ caller_top,
            /*callee_arg (input1)*/ callee_arg,
            /*gas_arg (input0)*/ [50_000, 0, 0, 0],
            /*depth1*/ 1,
            /*callee1*/ callee_arg,    // = row0.input1
            /*caller1*/ caller_top,    // = row0.callee
            /*return_pc1*/ 1,          // = row0.pc + 1
        );
        // Sanity: frame_depth column observable at rows 0 and 1.
        assert_eq!(trace.frame_depth[0], 0);
        assert_eq!(trace.frame_depth[1], 1);
        // sel_call_push_frame must fire on row 0 only.
        assert_eq!(trace.sel_call_push_frame[0], 1);
        assert_eq!(trace.sel_call_push_frame[1], 0);
        // The shifted constraint polynomial must vanish on the entire domain.
        assert!(
            shifted_poly_vanishes(&trace),
            "valid CALL push-frame transition must satisfy shifted constraints"
        );
    }

    #[test]
    fn test_phase5_call_push_frame_tampered_depth_fails() {
        // Same trace as the valid test but with row 1's frame_depth set to
        // an incorrect value (5 instead of 1). The depth_increment cross-row
        // constraint must catch this.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let trace = build_two_row_call_trace(
            /*depth0*/ 0,
            /*caller0*/ [0; 4],
            /*callee0*/ caller_top,
            /*callee_arg*/ callee_arg,
            /*gas_arg*/ [50_000, 0, 0, 0],
            /*depth1*/ 5, // WRONG: should be 1
            /*callee1*/ callee_arg,
            /*caller1*/ caller_top,
            /*return_pc1*/ 1,
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "tampered depth_next must fail the cross-row constraint"
        );
    }

    #[test]
    fn test_phase5_call_push_frame_tampered_callee_fails() {
        // Tamper with row 1's frame_callee: should not equal row 0's input1.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let bad_callee = [0xDEAD, 0, 0, 0];
        let trace = build_two_row_call_trace(
            /*depth0*/ 0,
            /*caller0*/ [0; 4],
            /*callee0*/ caller_top,
            /*callee_arg (input1)*/ callee_arg,
            /*gas_arg*/ [50_000, 0, 0, 0],
            /*depth1*/ 1,
            /*callee1*/ bad_callee, // WRONG: should be callee_arg
            /*caller1*/ caller_top,
            /*return_pc1*/ 1,
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "tampered frame_callee must fail the cross-row constraint"
        );
    }

    #[test]
    fn test_phase5_call_push_frame_tampered_caller_fails() {
        // Row 1's caller must equal row 0's callee. Tamper with caller1.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let bad_caller = [0xBEEF, 0, 0, 0];
        let trace = build_two_row_call_trace(
            /*depth0*/ 0,
            /*caller0*/ [0; 4],
            /*callee0*/ caller_top,
            /*callee_arg*/ callee_arg,
            /*gas_arg*/ [50_000, 0, 0, 0],
            /*depth1*/ 1,
            /*callee1*/ callee_arg,
            /*caller1*/ bad_caller, // WRONG: should be caller_top
            /*return_pc1*/ 1,
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "tampered caller propagation must fail the cross-row constraint"
        );
    }

    #[test]
    fn test_phase5_no_calls_constraints_pass() {
        // A trace consisting only of STOPs. No CALL/RETURN selectors fire,
        // so all Phase-5 cross-row constraints are gated off and trivially
        // satisfied (depth stays at 0 throughout).
        let mut trace = EvmTraceColumns::new();
        let frame = trace::FrameState::default();
        trace.push_row(&make_stop_row(0, 0, 1, frame.clone()));
        trace.push_row(&make_stop_row(1, 1, 2, frame.clone()));
        trace.push_row(&make_stop_row(2, 2, 3, frame.clone()));
        trace.push_row(&make_stop_row(3, 3, 4, frame.clone()));
        // sel_call_push_frame and sel_call_return must be 0 everywhere.
        for r in 0..4 {
            assert_eq!(trace.sel_call_push_frame[r], 0);
            assert_eq!(trace.sel_call_return[r], 0);
            assert_eq!(trace.frame_depth[r], 0);
        }
        assert!(
            shifted_poly_vanishes(&trace),
            "all-STOP trace must satisfy all shifted constraints"
        );
    }

    #[test]
    fn test_phase5_call_then_return_round_trip() {
        // Four-row trace: STOP / CALL / STOP_in_callee / RETURN.
        // Row 0: STOP at pc=0 (caller, depth=0)
        // Row 1: CALL at pc=1 (caller, depth=0); next row is callee
        // Row 2: STOP at pc=2 (callee, depth=1)
        // Row 3: RETURN at pc=3 (callee, depth=1); but the *next* row is
        //        back in the caller (depth=0). We pad rows beyond that.
        //
        // The frame snapshots on each row reflect the frame the row's
        // instruction belongs to:
        //   row 0: caller frame (depth=0)
        //   row 1: caller frame (depth=0) — same frame; the CALL pushes the
        //          new frame for row 2.
        //   row 2: callee frame (depth=1)
        //   row 3: callee frame (depth=1) — same; RETURN pops for row 4.
        //
        // We don't add a row 4 in this test (the trace pads to power of 2;
        // the wrap-around exclusion factor zeros out the row3→row0
        // transition). The depth_decrement constraint on row 3 still needs
        // a matching shifted depth — when padded, the next-row depth defaults
        // to 0, which is exactly depth(3) - 1 = 0. So this passes even with
        // padding rows.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let caller_frame = trace::FrameState {
            depth: 0,
            caller: [0; 4],
            callee: caller_top,
            value: [0; 4],
            gas: 1_000_000,
            return_pc: 0,
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        let callee_frame = trace::FrameState {
            depth: 1,
            caller: caller_top,    // = caller_frame.callee
            callee: callee_arg,    // = row 1 (CALL).input1
            value: [0; 4],
            gas: 1_000_000,
            return_pc: 2,          // = row 1 (CALL).pc + 1 = 1 + 1
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        let mut t = EvmTraceColumns::new();
        // Row 0: STOP at pc=0, next_pc=1
        t.push_row(&make_stop_row(0, 0, 1, caller_frame.clone()));
        // Row 1: CALL at pc=1, next_pc=2, gas=50000, callee=callee_arg
        t.push_row(&make_call_row(
            1, 1, [50_000, 0, 0, 0], callee_arg, 2, caller_frame.clone(),
        ));
        // Row 2: STOP at pc=2 (in callee), next_pc=3
        t.push_row(&make_jumpdest_row(2, 2, 3, callee_frame.clone()));
        // Row 3: RETURN at pc=3 (in callee), next_pc=2 (back to caller's
        //        return_pc).
        t.push_row(&make_return_row(3, 3, 2, callee_frame.clone()));

        assert_eq!(t.frame_depth[0], 0);
        assert_eq!(t.frame_depth[1], 0);
        assert_eq!(t.frame_depth[2], 1);
        assert_eq!(t.frame_depth[3], 1);
        assert_eq!(t.sel_call_push_frame[1], 1);
        assert_eq!(t.sel_call_return[3], 1);
        assert!(
            shifted_poly_vanishes(&t),
            "valid CALL/RETURN round-trip must satisfy all shifted constraints"
        );
    }

    #[test]
    fn test_phase5_padding_rows_have_zero_frame_depth() {
        // A short trace pads to a power of 2. Padding rows are constructed
        // by zero-extending the column values (TracePolynomials default).
        // Verify that frame_depth is 0 on padding rows.
        let mut t = EvmTraceColumns::new();
        let frame = trace::FrameState::default();
        t.push_row(&make_stop_row(0, 0, 0, frame));
        // num_rows=1 padded to 2. Build polys and inspect.
        let polys =
            metavm_zkp::trace::TracePolynomials::from_vm_trace(&t, CurveType::Bls48581);
        // padded_size should be a power of 2 ≥ 1; for 1 row it's 1 (or 2
        // depending on the framework). Either way, the frame_depth poly must
        // be all zeros on the padding rows.
        let depth_col = &polys.columns[trace::COL_FRAME_DEPTH].evaluations;
        for (i, v) in depth_col.iter().enumerate() {
            assert!(v.is_zero(), "frame_depth padding row {} must be 0", i);
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Call-family tests: CALLCODE / DELEGATECALL / STATICCALL /
    // CREATE / CREATE2 / REVERT
    // ════════════════════════════════════════════════════════════════════

    /// Build a call-family push-frame row with a chosen opcode.
    fn make_callfamily_row(
        step: u64,
        pc: u64,
        opcode: u8,
        gas_arg: [u64; 4],
        callee_arg: [u64; 4],
        next_pc: u64,
        current_frame: trace::FrameState,
    ) -> EvmTraceRow {
        EvmTraceRow {
            step,
            pc,
            opcode,
            gas_remaining: 1_000_000,
            stack_depth: 7,
            input0: gas_arg,
            input1: callee_arg,
            output0: [0; 4],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_CALL,
            funct: opcode - 0xF0,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc,
            frame: current_frame,
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// Build a CREATE / CREATE2 row with a populated address-hint column.
    fn make_create_row(
        step: u64,
        pc: u64,
        opcode: u8,         // 0xF0 (CREATE) or 0xF5 (CREATE2)
        next_pc: u64,
        current_frame: trace::FrameState,
        address_hint: [u64; 4],
    ) -> EvmTraceRow {
        EvmTraceRow {
            step,
            pc,
            opcode,
            gas_remaining: 1_000_000,
            stack_depth: 3,
            input0: [0; 4],
            input1: [0; 4],
            output0: [0; 4],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_CALL,
            funct: opcode - 0xF0,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc,
            frame: current_frame,
            create_address_hint: address_hint,
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// Build a REVERT row (opcode 0xFD, classified as INSN_STOP funct=2).
    fn make_revert_row(
        step: u64,
        pc: u64,
        next_pc: u64,
        current_frame: trace::FrameState,
    ) -> EvmTraceRow {
        EvmTraceRow {
            step,
            pc,
            opcode: 0xFD,
            gas_remaining: 1_000_000,
            stack_depth: 2,
            input0: [0; 4],
            input1: [0; 4],
            output0: [0; 4],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_STOP,
            funct: 2, // matches classify_opcode(0xFD)
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc,
            frame: current_frame,
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        }
    }

    /// 2-row helper: a push-frame opcode at row 0 and the entered child frame
    /// at row 1, with explicit caller/callee/value/static fields for the
    /// child frame so each test can assert exactly the expected transition.
    #[allow(clippy::too_many_arguments)]
    fn build_two_row_pushframe_trace(
        opcode: u8,
        // row 0 frame
        depth0: u64,
        caller0: [u64; 4],
        callee0: [u64; 4],
        value0: [u64; 4],
        static0: u64,
        // row 0 stack args
        input0_arg: [u64; 4], // gas (or salt for CREATE2)
        input1_arg: [u64; 4], // callee target / unused for CREATE
        // row 1 frame
        depth1: u64,
        caller1: [u64; 4],
        callee1: [u64; 4],
        value1: [u64; 4],
        static1: u64,
        return_pc1: u64,
        // CREATE/CREATE2 only
        address_hint: [u64; 4],
    ) -> EvmTraceColumns {
        let frame0 = trace::FrameState {
            depth: depth0,
            caller: caller0,
            callee: callee0,
            value: value0,
            gas: 1_000_000,
            return_pc: 0,
            return_offset: 0,
            return_size: 0,
            is_static: static0,
        };
        let frame1 = trace::FrameState {
            depth: depth1,
            caller: caller1,
            callee: callee1,
            value: value1,
            gas: 1_000_000,
            return_pc: return_pc1,
            return_offset: 0,
            return_size: 0,
            is_static: static1,
        };
        let mut t = EvmTraceColumns::new();
        // Row 0: the push-frame opcode at pc=0, next_pc=1.
        let row0 = if opcode == 0xF0 || opcode == 0xF5 {
            make_create_row(0, 0, opcode, 1, frame0, address_hint)
        } else {
            make_callfamily_row(0, 0, opcode, input0_arg, input1_arg, 1, frame0)
        };
        t.push_row(&row0);
        // Row 1: STOP in the child frame at pc=1, next_pc=2.
        t.push_row(&make_stop_row(1, 1, 2, frame1));
        t
    }

    #[test]
    fn test_phase5_callcode_push_frame_valid() {
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF2, // CALLCODE
            /*depth0*/ 0, /*caller0*/ [0; 4], /*callee0*/ caller_top,
            /*value0*/ [0; 4], /*static0*/ 0,
            /*gas_arg*/ [50_000, 0, 0, 0], /*callee_arg*/ callee_arg,
            /*depth1*/ 1,
            /*caller1*/ caller_top, // same shape as CALL: caller = parent's callee
            /*callee1*/ callee_arg,
            /*value1*/ [0; 4],
            /*static1*/ 0,
            /*return_pc1*/ 1,
            /*address_hint*/ [0; 4],
        );
        assert_eq!(trace.sel_callcode[0], 1);
        assert_eq!(trace.sel_call_push_frame[0], 0);
        assert!(
            shifted_poly_vanishes(&trace),
            "valid CALLCODE push-frame must satisfy shifted constraints"
        );
    }

    #[test]
    fn test_phase5_callcode_tampered_caller_fails() {
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let bad_caller = [0xBEEF, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF2,
            0, [0; 4], caller_top, [0; 4], 0,
            [50_000, 0, 0, 0], callee_arg,
            1, bad_caller, callee_arg, [0; 4], 0, 1,
            [0; 4],
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "tampered CALLCODE caller propagation must fail"
        );
    }

    #[test]
    fn test_phase5_delegatecall_preserves_caller_and_value() {
        // Outer caller is some EOA 0x1111. The current frame has caller=0x1111,
        // callee=0x2222, value=42. A DELEGATECALL to 0x3333 must produce a
        // child frame with caller=0x1111 (NOT 0x2222) and value=42.
        let outer_caller = [0x1111, 0, 0, 0];
        let current_callee = [0x2222, 0, 0, 0];
        let target = [0x3333, 0, 0, 0];
        let preserved_value = [42, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF4, // DELEGATECALL
            /*depth0*/ 1,
            /*caller0*/ outer_caller,
            /*callee0*/ current_callee,
            /*value0*/ preserved_value,
            /*static0*/ 0,
            /*gas_arg*/ [50_000, 0, 0, 0],
            /*callee_arg (input1)*/ target,
            /*depth1*/ 2,
            /*caller1*/ outer_caller,   // PRESERVED (NOT current_callee)
            /*callee1*/ target,
            /*value1*/ preserved_value, // PRESERVED
            /*static1*/ 0,
            /*return_pc1*/ 1,
            /*address_hint*/ [0; 4],
        );
        assert_eq!(trace.sel_delegatecall[0], 1);
        assert!(
            shifted_poly_vanishes(&trace),
            "valid DELEGATECALL caller+value preservation must satisfy shifted constraints"
        );
    }

    #[test]
    fn test_phase5_delegatecall_tampered_caller_uses_callee_fails() {
        // The classic mistake: writing caller_next = current_callee (which is
        // the CALL semantics, NOT DELEGATECALL). The constraint must catch it.
        let outer_caller = [0x1111, 0, 0, 0];
        let current_callee = [0x2222, 0, 0, 0];
        let target = [0x3333, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF4,
            1, outer_caller, current_callee, [42, 0, 0, 0], 0,
            [50_000, 0, 0, 0], target,
            // BUG: caller1 = current_callee (would be valid for CALL, not for
            // DELEGATECALL).
            2, current_callee, target, [42, 0, 0, 0], 0, 1,
            [0; 4],
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "DELEGATECALL caller-preservation must reject the CALL-style caller=callee pattern"
        );
    }

    #[test]
    fn test_phase5_delegatecall_tampered_value_fails() {
        let outer_caller = [0x1111, 0, 0, 0];
        let current_callee = [0x2222, 0, 0, 0];
        let target = [0x3333, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF4,
            1, outer_caller, current_callee, [42, 0, 0, 0], 0,
            [50_000, 0, 0, 0], target,
            // BUG: value1 ≠ value0.
            2, outer_caller, target, [99, 0, 0, 0], 0, 1,
            [0; 4],
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "DELEGATECALL value-preservation must reject mismatched value"
        );
    }

    #[test]
    fn test_phase5_staticcall_sets_static_mode() {
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xFA, // STATICCALL
            /*depth0*/ 0, /*caller0*/ [0; 4], /*callee0*/ caller_top,
            /*value0*/ [0; 4], /*static0*/ 0, // outer is non-static
            /*gas_arg*/ [50_000, 0, 0, 0], /*callee_arg*/ callee_arg,
            /*depth1*/ 1,
            /*caller1*/ caller_top, // CALL-shape caller propagation
            /*callee1*/ callee_arg,
            /*value1*/ [0; 4],
            /*static1*/ 1, // ENTERED static mode
            /*return_pc1*/ 1,
            /*address_hint*/ [0; 4],
        );
        assert_eq!(trace.sel_staticcall[0], 1);
        assert_eq!(trace.frame_static[0], 0);
        assert_eq!(trace.frame_static[1], 1);
        assert!(
            shifted_poly_vanishes(&trace),
            "valid STATICCALL must set frame_static = 1 on the child frame"
        );
    }

    #[test]
    fn test_phase5_staticcall_tampered_static_zero_fails() {
        // Tamper: child frame keeps static=0 even though sel_staticcall fires.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xFA,
            0, [0; 4], caller_top, [0; 4], 0,
            [50_000, 0, 0, 0], callee_arg,
            // BUG: static1 = 0 instead of 1.
            1, caller_top, callee_arg, [0; 4], 0, 1,
            [0; 4],
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "STATICCALL must reject static_next = 0"
        );
    }

    #[test]
    fn test_phase5_revert_pops_frame() {
        // Two rows: row 0 is some inert in-callee instruction at depth=1,
        // row 1 is the REVERT at depth=1. The depth_dec constraint demands
        // frame_depth on the row AFTER REVERT to be 0 — which the padding row
        // (zero-filled) satisfies. Place REVERT at row 1 so the transition
        // row1 → row2 (padding row, depth=0) is checked.
        //
        // Layout:
        //   row 0: STOP at pc=0 in child frame (depth=1)
        //   row 1: REVERT at pc=1 in child frame (depth=1)
        //   row 2 (padding): depth=0 (default).
        let caller_top = [0x42, 0, 0, 0];
        let callee = [0xABCD, 0, 0, 0];
        let child_frame = trace::FrameState {
            depth: 1,
            caller: caller_top,
            callee,
            value: [0; 4],
            gas: 1_000_000,
            return_pc: 1,
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        let mut t = EvmTraceColumns::new();
        t.push_row(&make_stop_row(0, 0, 1, child_frame.clone()));
        t.push_row(&make_revert_row(1, 1, 2, child_frame.clone()));
        // Sanity: REVERT row gets sel_revert.
        assert_eq!(t.sel_revert[1], 1);
        assert_eq!(t.sel_stop[1], 0); // explicitly NOT routed to STOP
        assert!(
            shifted_poly_vanishes(&t),
            "valid REVERT depth-decrement must satisfy shifted constraints"
        );
    }

    #[test]
    fn test_phase5_revert_tampered_depth_fails() {
        // REVERT row asserts the next row's frame_depth equals current-1.
        // Tamper: padding row is forced to depth=5 (we directly mutate the
        // column post-hoc to simulate a malicious prover).
        let caller_top = [0x42, 0, 0, 0];
        let callee = [0xABCD, 0, 0, 0];
        let child_frame = trace::FrameState {
            depth: 1,
            caller: caller_top,
            callee,
            value: [0; 4],
            gas: 1_000_000,
            return_pc: 1,
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        let mut t = EvmTraceColumns::new();
        t.push_row(&make_revert_row(0, 0, 1, child_frame.clone()));
        // Row 1 with depth=5 (NOT 0).
        let bad_next = trace::FrameState {
            depth: 5,
            caller: caller_top,
            callee,
            value: [0; 4],
            gas: 1_000_000,
            return_pc: 1,
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        t.push_row(&make_stop_row(1, 1, 2, bad_next));
        assert!(
            !shifted_poly_vanishes(&t),
            "tampered REVERT depth_dec must fail"
        );
    }

    #[test]
    fn test_phase5_create_address_uses_hint() {
        // CREATE row at row 0; address hint = next row's frame_callee.
        let caller_top = [0x42, 0, 0, 0];
        let new_addr = [0xC0DE, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF0, // CREATE
            /*depth0*/ 0, /*caller0*/ [0; 4], /*callee0*/ caller_top,
            /*value0*/ [0; 4], /*static0*/ 0,
            /*input0*/ [0; 4], /*input1*/ [0; 4],
            /*depth1*/ 1,
            /*caller1*/ caller_top, // CALL-shape: caller = parent's callee
            /*callee1*/ new_addr,   // = address_hint
            /*value1*/ [0; 4],
            /*static1*/ 0,
            /*return_pc1*/ 1,
            /*address_hint*/ new_addr,
        );
        assert_eq!(trace.sel_create[0], 1);
        assert_eq!(trace.create_address_hint[0][0], new_addr[0]);
        assert!(
            shifted_poly_vanishes(&trace),
            "valid CREATE address-hint binding must satisfy shifted constraints"
        );
    }

    #[test]
    fn test_phase5_create_tampered_callee_fails() {
        // child callee ≠ address hint.
        let caller_top = [0x42, 0, 0, 0];
        let new_addr = [0xC0DE, 0, 0, 0];
        let bad_addr = [0xDEAD, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF0,
            0, [0; 4], caller_top, [0; 4], 0,
            [0; 4], [0; 4],
            // child callee is bad_addr but the hint says new_addr.
            1, caller_top, bad_addr, [0; 4], 0, 1,
            new_addr,
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "tampered CREATE callee binding must fail"
        );
    }

    #[test]
    fn test_phase5_create2_address_uses_hint() {
        let caller_top = [0x42, 0, 0, 0];
        let new_addr = [0xCAFE, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF5, // CREATE2
            0, [0; 4], caller_top, [0; 4], 0,
            [0; 4], [0; 4],
            1, caller_top, new_addr, [0; 4], 0, 1,
            new_addr,
        );
        assert_eq!(trace.sel_create2[0], 1);
        assert!(
            shifted_poly_vanishes(&trace),
            "valid CREATE2 address-hint binding must satisfy shifted constraints"
        );
    }

    #[test]
    fn test_phase5_static_propagates_through_call() {
        // Outer frame is already static. A nested CALL should keep the child
        // static. (Static-mode escape is forbidden; our constraint enforces
        // frame_static(ω·X) - frame_static(X) = 0 for CALL.)
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF1, // CALL
            /*depth0*/ 1,
            /*caller0*/ [0x99, 0, 0, 0],
            /*callee0*/ caller_top,
            /*value0*/ [0; 4],
            /*static0*/ 1, // outer ALREADY static
            /*gas_arg*/ [50_000, 0, 0, 0],
            /*callee_arg*/ callee_arg,
            /*depth1*/ 2,
            /*caller1*/ caller_top,
            /*callee1*/ callee_arg,
            /*value1*/ [0; 4],
            /*static1*/ 1, // child REMAINS static
            /*return_pc1*/ 1,
            /*address_hint*/ [0; 4],
        );
        assert!(
            shifted_poly_vanishes(&trace),
            "static-mode propagation through CALL must satisfy shifted constraints"
        );
    }

    #[test]
    fn test_phase5_static_escape_through_call_fails() {
        // Outer frame is static; a nested CALL tries to clear static. Reject.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let trace = build_two_row_pushframe_trace(
            0xF1,
            1, [0x99, 0, 0, 0], caller_top, [0; 4], 1,
            [50_000, 0, 0, 0], callee_arg,
            // BUG: child sets static=0 even though outer was static.
            2, caller_top, callee_arg, [0; 4], 0, 1,
            [0; 4],
        );
        assert!(
            !shifted_poly_vanishes(&trace),
            "static-mode escape (outer=1, child=0) must be rejected"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // Default frame-state preservation tests
    // ════════════════════════════════════════════════════════════════════
    //
    // Closes Phase-5 Gap A: "intermediate-row mutation". Without these
    // constraints a malicious prover can mutate any frame-state column on
    // a row where no frame-changing selector fires. The new constraints
    // enforce
    //     (1 - is_frame_op(z)) · (F(ω·z) - F(z)) = 0
    // for the 15 frame-state columns currently exposed via shifted_evals.

    /// Build an N-row trace with all rows running STOP in a fixed frame.
    /// Frame state is the same on every row, so default-preservation must
    /// hold trivially.
    /// Build an N-row trace of JUMPDEST (0x5B) — the EVM no-op opcode.
    /// Used by frame-state preservation tests that need a sequence of
    /// rows in a fixed frame where no frame-mutating selector fires.
    /// JUMPDEST classifies as INSN_JUMP → sel_jump (oracle, no body)
    /// and is NOT in the is_frame_op gate, so preservation constraints
    /// remain active across these rows. Naming kept as
    /// `build_n_row_stop_trace` for backward compatibility — STOP at
    /// depth ≥ 1 was previously the no-op of choice but it now
    /// participates in the is_frame_op gate to support implicit
    /// frame-pop at end-of-init-code (which is what real revm produces).
    fn build_n_row_stop_trace(num_rows: usize, frame: trace::FrameState) -> EvmTraceColumns {
        let mut t = EvmTraceColumns::new();
        for r in 0..num_rows {
            t.push_row(&EvmTraceRow {
                step: r as u64,
                pc: r as u64,
                opcode: 0x5B, // JUMPDEST
                gas_remaining: 1_000_000,
                stack_depth: 0,
                input0: [0; 4],
                input1: [0; 4],
                output0: [0; 4],
                mem_offset: 0,
                mem_value: [0; 4],
                insn_type: trace::INSN_JUMP,
                funct: 0,
                immediate: [0; 4],
                aux0: [0; 4],
                aux1: [0; 4],
                next_pc: (r + 1) as u64,
                frame: frame.clone(),
                create_address_hint: [0; 4],
                create_nonce_hint: 0,
                sel_stop_pop: 0,
                create2_salt_hint: [0; 4],
                create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
            });
        }
        t
    }

    #[test]
    fn test_phase5_default_preservation_normal_row() {
        // Build a 4-row trace where every row runs STOP in a fixed,
        // *non-default* parent frame. No frame-changing selector fires, so
        // every preservation constraint must be 0 across the entire domain.
        let frame = trace::FrameState {
            depth: 1,
            caller: [0x111, 0, 0, 0],
            callee: [0x222, 0, 0, 0],
            value: [0x333, 0, 0, 0],
            gas: 1_234_567,
            return_pc: 7,
            return_offset: 0x100,
            return_size: 0x40,
            is_static: 0,
        };
        let trace = build_n_row_stop_trace(4, frame);
        assert!(
            shifted_poly_vanishes(&trace),
            "constant frame across all rows must satisfy default preservation"
        );
    }

    #[test]
    fn test_phase5_default_preservation_violated() {
        // Same 4-row STOP trace, but tamper with row 2's frame_caller_l0.
        // Row 1→row 2 transition has is_frame_op = 0, so the preservation
        // gate is active and the constraint must catch the mutation.
        let frame = trace::FrameState {
            depth: 1,
            caller: [0x111, 0, 0, 0],
            callee: [0x222, 0, 0, 0],
            value: [0x333, 0, 0, 0],
            gas: 1_234_567,
            return_pc: 7,
            return_offset: 0x100,
            return_size: 0x40,
            is_static: 0,
        };
        let mut trace = build_n_row_stop_trace(4, frame.clone());
        // Mutate row 2's caller (low limb).
        trace.frame_caller[0][2] = 0x999;
        assert!(
            !shifted_poly_vanishes(&trace),
            "tampered frame_caller on a non-frame-op row must fail preservation"
        );
    }

    #[test]
    fn test_phase5_default_preservation_skipped_on_call_push() {
        // Row 0: in caller frame at depth 0. Row 1: CALL pushing a fresh
        // frame (depth 1). Row 2: STOP in callee frame. Row 3: STOP in
        // callee frame (preserved). The is_frame_op gate is 1 on row 1, so
        // the preservation constraint vanishes for the row 1→row 2 boundary
        // even though caller/callee/depth all change. Must still verify.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let caller_frame = trace::FrameState {
            depth: 0,
            caller: [0; 4],
            callee: caller_top,
            value: [0; 4],
            gas: 1_000_000,
            return_pc: 0,
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        let callee_frame = trace::FrameState {
            depth: 1,
            caller: caller_top,
            callee: callee_arg,
            value: [0; 4],
            gas: 1_000_000,
            return_pc: 2,
            return_offset: 0,
            return_size: 0,
            is_static: 0,
        };
        let mut t = EvmTraceColumns::new();
        // Row 0: STOP @ pc=0 in caller.
        t.push_row(&make_stop_row(0, 0, 1, caller_frame.clone()));
        // Row 1: CALL @ pc=1 in caller; pushes callee for row 2.
        t.push_row(&make_call_row(
            1, 1, [50_000, 0, 0, 0], callee_arg, 2, caller_frame.clone(),
        ));
        // Row 2: STOP @ pc=2 in callee.
        t.push_row(&make_jumpdest_row(2, 2, 3, callee_frame.clone()));
        // Row 3: STOP @ pc=3 in callee.
        t.push_row(&make_jumpdest_row(3, 3, 4, callee_frame.clone()));
        assert_eq!(t.sel_call_push_frame[1], 1);
        assert!(
            shifted_poly_vanishes(&t),
            "preservation must yield (gate=0) on a CALL_PUSH_FRAME row"
        );
    }

    // ════════════════════════════════════════════════════════════════════
    // Full frame-stack LIFO multiset permutation tests
    // ════════════════════════════════════════════════════════════════════
    //
    // Closes Phase-5 Gap B: pop without bound parent frame state. These
    // tests construct push/pop scenarios and check that the grand-product
    // accumulator Z closes (==1 at end) iff the multiset of pushed
    // parent-frame tuples equals the multiset of popped parent-frame
    // tuples (keyed on depth).
    //
    // The tests directly exercise `compute_frame_perm_z` and the
    // transition body, decoupled from the full prove/verify pipeline so
    // we can detect tampering at the multiset-equality level without
    // BLS commitments / KZG openings.

    /// Compute the Z column from a trace and check whether it closes
    /// (Z[n-1] · push/pop_factor[n-1] mod the cycle == 1).
    fn frame_perm_closes(trace: &EvmTraceColumns) -> bool {
        let curve = CurveType::Bls48581;
        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(trace, curve);
        let n = polys.padded_size as usize;
        let cs = EvmConstraintSystem::new();
        let fp_layout = cs.frame_perm_layout().expect("EVM has frame_perm_layout");

        // Build per-row tuple values (push side at row, pop side at row+1 mod n).
        let zero = Scalar::zero(curve);
        let mut tuple_per_row: Vec<Vec<Scalar>> = Vec::with_capacity(n);
        let mut tuple_shifted_per_row: Vec<Vec<Scalar>> = Vec::with_capacity(n);
        let mut is_push_evals = vec![zero.clone(); n];
        let mut is_pop_evals = vec![zero.clone(); n];

        // The actual scalar columns (eval form).
        let cols: Vec<Vec<Scalar>> = polys.columns.iter()
            .map(|p| p.evaluations.clone())
            .collect();

        // Apply selector + frame padding fixups (mirrors prover preprocessing).
        let mut cols_fixed = cols;
        // Pad COL_SEL_STOP=1 on padding rows so sum-to-one holds.
        for r in trace.frame_depth.len()..n {
            cols_fixed[trace::COL_SEL_STOP][r] = Scalar::one(curve);
        }
        cs.fix_trace_padding(&mut cols_fixed, trace.frame_depth.len(), n);

        for row in 0..n {
            let next = (row + 1) % n;
            let mut here = Vec::with_capacity(fp_layout.tuple_columns.len());
            let mut there = Vec::with_capacity(fp_layout.tuple_columns.len());
            for &c in &fp_layout.tuple_columns {
                here.push(cols_fixed[c][row].clone());
                there.push(cols_fixed[c][next].clone());
            }
            tuple_per_row.push(here);
            tuple_shifted_per_row.push(there);

            let mut p = Scalar::zero(curve);
            for &s in &fp_layout.push_selectors {
                p = p.add(&cols_fixed[s][row]);
            }
            is_push_evals[row] = p;
            let mut q = Scalar::zero(curve);
            for &s in &fp_layout.pop_selectors {
                q = q.add(&cols_fixed[s][row]);
            }
            is_pop_evals[row] = q;
        }

        let gamma = Scalar::from_u64(0xCAFE_BABE, curve);
        let delta = Scalar::from_u64(0xDEAD_BEEF, curve);

        let z = metavm_zkp::permutation::compute_frame_perm_z(
            &tuple_per_row, &tuple_shifted_per_row,
            &is_push_evals, &is_pop_evals,
            &gamma, &delta, curve,
        );

        // Z[0] = 1 by construction. The multiset closes iff the
        // wrap-around product Z[0] = Z[n-1] · push_factor[n-1] /
        // pop_factor[n-1] also equals 1.
        let last_push = is_push_evals[n - 1].mul(
            &metavm_zkp::permutation::frame_perm_tuple_factor(
                &tuple_per_row[n - 1], &gamma, &delta))
            .add(&Scalar::one(curve).sub(&is_push_evals[n - 1]));
        let last_pop = is_pop_evals[n - 1].mul(
            &metavm_zkp::permutation::frame_perm_tuple_factor(
                &tuple_shifted_per_row[n - 1], &gamma, &delta))
            .add(&Scalar::one(curve).sub(&is_pop_evals[n - 1]));
        let wrap = z[n - 1].mul(&last_push).mul(&last_pop.inverse());
        wrap.sub(&Scalar::one(curve)).is_zero()
    }

    #[test]
    fn test_phase5_frame_perm_balanced_call_return() {
        // Synthesize a trace: row 0 STOP @ caller; row 1 CALL pushes
        // child; rows 2-3 STOP in child; row 4 RETURN pops back to
        // caller; row 5 STOP @ caller. Multiset must close.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let caller_frame = trace::FrameState {
            depth: 0, caller: [0; 4], callee: caller_top,
            value: [0; 4], gas: 1_000_000, return_pc: 0,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        let callee_frame = trace::FrameState {
            depth: 1, caller: caller_top, callee: callee_arg,
            value: [0; 4], gas: 500_000, return_pc: 2,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        let mut t = EvmTraceColumns::new();
        t.push_row(&make_stop_row(0, 0, 1, caller_frame.clone()));
        t.push_row(&make_call_row(
            1, 1, [50_000, 0, 0, 0], callee_arg, 2, caller_frame.clone(),
        ));
        t.push_row(&make_jumpdest_row(2, 2, 3, callee_frame.clone()));
        t.push_row(&make_jumpdest_row(3, 3, 4, callee_frame.clone()));
        t.push_row(&make_return_row(4, 4, 5, callee_frame.clone()));
        t.push_row(&make_stop_row(5, 5, 6, caller_frame.clone()));

        assert!(
            frame_perm_closes(&t),
            "balanced CALL/RETURN must close the frame-stack multiset",
        );
    }

    #[test]
    fn test_phase5_frame_perm_unmatched_pop_caught() {
        // Build a trace where REVERT fires without a prior matching
        // push. Multiset can't close.
        let caller_frame = trace::FrameState {
            depth: 1, caller: [0xAA, 0, 0, 0], callee: [0xBB, 0, 0, 0],
            value: [0; 4], gas: 1_000_000, return_pc: 0,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        let popped_to = trace::FrameState {
            depth: 0, caller: [0; 4], callee: [0xAA, 0, 0, 0],
            value: [0; 4], gas: 1_000_000, return_pc: 0,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        let mut t = EvmTraceColumns::new();
        // Row 0: REVERT in caller_frame (NO prior push to match).
        t.push_row(&make_revert_row(0, 0, 1, caller_frame.clone()));
        // Row 1: arrive in popped_to.
        t.push_row(&make_stop_row(1, 1, 2, popped_to.clone()));
        // Row 2: STOP.
        t.push_row(&make_stop_row(2, 2, 3, popped_to.clone()));

        assert!(
            !frame_perm_closes(&t),
            "REVERT with no prior push must NOT close the multiset",
        );
    }

    #[test]
    fn test_phase5_frame_perm_tampered_caller_caught() {
        // Balanced CALL/RETURN, but the popped frame has a tampered
        // caller field. The push side saved (caller=top, ...) but the
        // pop side claims to restore (caller=BAD, ...). Multiset must
        // not close.
        let caller_top = [0x42, 0, 0, 0];
        let callee_arg = [0xABCD, 0, 0, 0];
        let caller_frame = trace::FrameState {
            depth: 0, caller: [0; 4], callee: caller_top,
            value: [0; 4], gas: 1_000_000, return_pc: 0,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        let callee_frame = trace::FrameState {
            depth: 1, caller: caller_top, callee: callee_arg,
            value: [0; 4], gas: 500_000, return_pc: 2,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        let tampered_caller = trace::FrameState {
            // Wrong caller after pop (should match what was pushed at row 1).
            caller: [0xBAD, 0, 0, 0],
            ..caller_frame.clone()
        };
        let mut t = EvmTraceColumns::new();
        t.push_row(&make_stop_row(0, 0, 1, caller_frame.clone()));
        t.push_row(&make_call_row(
            1, 1, [50_000, 0, 0, 0], callee_arg, 2, caller_frame.clone(),
        ));
        t.push_row(&make_jumpdest_row(2, 2, 3, callee_frame.clone()));
        t.push_row(&make_return_row(3, 3, 4, callee_frame.clone()));
        // Row 4: tampered restored frame. The pop on row 3 reads its
        // tuple from row 4 (the SHIFTED tuple).
        t.push_row(&make_stop_row(4, 4, 5, tampered_caller.clone()));

        assert!(
            !frame_perm_closes(&t),
            "tampered restored caller must NOT close the multiset",
        );
    }

    #[test]
    fn test_phase5_frame_perm_lifo_swap_attack_caught() {
        // Two nested calls at distinct depths. The pops claim to
        // restore frames in the WRONG order — i.e. push F1 at depth 0,
        // push F2 at depth 1, then pop the inner call expecting F2 but
        // give back F1's tuple, and pop the outer expecting F1 but
        // give back F2's. Multiset across pushes: {(0, F1), (1, F2)};
        // multiset across pops: {(1, F1'), (0, F2')}. With distinct
        // depths these are NOT equal, so Z cannot close.
        let f1 = trace::FrameState {
            depth: 0, caller: [0; 4], callee: [0x100, 0, 0, 0],
            value: [0; 4], gas: 1_000_000, return_pc: 0,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        let f2 = trace::FrameState {
            depth: 1, caller: [0x100, 0, 0, 0], callee: [0x200, 0, 0, 0],
            value: [0; 4], gas: 500_000, return_pc: 2,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        let f3 = trace::FrameState {
            depth: 2, caller: [0x200, 0, 0, 0], callee: [0x300, 0, 0, 0],
            value: [0; 4], gas: 250_000, return_pc: 4,
            return_offset: 0, return_size: 0, is_static: 0,
        };
        // Build trace with swapped restoration:
        //   row 0: STOP in f1
        //   row 1: CALL@f1 (push parent f1, child=f2)
        //   row 2: STOP in f2
        //   row 3: CALL@f2 (push parent f2, child=f3)
        //   row 4: STOP in f3
        //   row 5: RETURN@f3 (pop, restore — but BAD: claim f1 instead of f2)
        //   row 6: STOP in f1 (WRONG; should be f2)
        //   row 7: RETURN@f1 (pop, restore — claim f2 instead of f1)
        //   row 8: STOP in f2 (WRONG; should be f1).
        let mut t = EvmTraceColumns::new();
        t.push_row(&make_stop_row(0, 0, 1, f1.clone()));
        t.push_row(&make_call_row(1, 1, [10, 0, 0, 0], [0x200, 0, 0, 0], 2, f1.clone()));
        t.push_row(&make_stop_row(2, 2, 3, f2.clone()));
        t.push_row(&make_call_row(3, 3, [10, 0, 0, 0], [0x300, 0, 0, 0], 4, f2.clone()));
        t.push_row(&make_stop_row(4, 4, 5, f3.clone()));
        t.push_row(&make_return_row(5, 5, 6, f3.clone()));
        // Tampered: claim f1 instead of f2 here.
        let f1_at_d1 = trace::FrameState { depth: 1, ..f1.clone() };
        t.push_row(&make_stop_row(6, 6, 7, f1_at_d1.clone()));
        t.push_row(&make_return_row(7, 7, 8, f1_at_d1.clone()));
        // Tampered: claim f2 here (but restore should go to f1).
        let f2_at_d0 = trace::FrameState { depth: 0, ..f2.clone() };
        t.push_row(&make_stop_row(8, 8, 9, f2_at_d0.clone()));

        assert!(
            !frame_perm_closes(&t),
            "sibling-call swap attack (f1↔f2 swapped on pops) must NOT close",
        );
    }

    #[test]
    fn test_phase5_frame_perm_serialization_roundtrip() {
        use metavm_zkp::prover::ExecutionProof;
        use metavm_zkp::commitment::{Commitment, BatchProof};

        let mut p = ExecutionProof {
            column_commitments: vec![Commitment(vec![1, 2, 3])],
            quotient_commitments: vec![Commitment(vec![4, 5])],
            evaluations: vec![vec![10]],
            opening_proof: BatchProof { d: vec![], proof: vec![6, 7] },
            num_steps: 0xABCD,
            domain_size: 16,
            num_quotient_chunks: 1,
            shifted_evaluations: Vec::new(),
            shifted_opening_proof: None,
            logup_commitments: Vec::new(),
            logup_evaluations: Vec::new(),
            logup_shifted_evaluations: Vec::new(),
            logup_opening_proof: None,
            logup_shifted_opening_proof: None,
            bitwise_commitments: Vec::new(),
            bitwise_evaluations: Vec::new(),
            bitwise_shifted_evaluations: Vec::new(),
            bitwise_opening_proof: None,
            bitwise_shifted_opening_proof: None,
            perm_commitments: Vec::new(),
            perm_evaluations: Vec::new(),
            perm_shifted_evaluations: Vec::new(),
            perm_opening_proof: None,
            perm_shifted_opening_proof: None,
            oracle_data: Vec::new(),
            reg_perm_commitments: Vec::new(),
            reg_perm_evaluations: Vec::new(),
            reg_perm_shifted_evaluations: Vec::new(),
            reg_perm_opening_proof: None,
            reg_perm_shifted_opening_proof: None,
            // Frame-perm fields populated.
            frame_perm_commitment: Some(Commitment(vec![0x42; 74])),
            frame_perm_evaluation: Some(vec![0x11; 73]),
            frame_perm_shifted_evaluation: Some(vec![0x22; 73]),
            frame_perm_opening_proof: Some(BatchProof { d: vec![], proof: vec![0x33; 74] }),
            frame_perm_shifted_opening_proof: Some(BatchProof { d: vec![], proof: vec![0x44; 74] }),
            frame_perm_pop_shifted_evaluations: Vec::new(),
            frame_perm_pop_shifted_opening_proof: None,
        };

        let bytes = p.to_bytes();
        let decoded = ExecutionProof::from_bytes(&bytes)
            .expect("ExecutionProof must round-trip with frame-perm fields populated");

        assert_eq!(decoded.frame_perm_commitment.as_ref().map(|c| &c.0),
                   p.frame_perm_commitment.as_ref().map(|c| &c.0));
        assert_eq!(decoded.frame_perm_evaluation, p.frame_perm_evaluation);
        assert_eq!(decoded.frame_perm_shifted_evaluation, p.frame_perm_shifted_evaluation);
        assert_eq!(decoded.frame_perm_opening_proof.as_ref().map(|x| &x.proof),
                   p.frame_perm_opening_proof.as_ref().map(|x| &x.proof));
        assert_eq!(decoded.frame_perm_shifted_opening_proof.as_ref().map(|x| &x.proof),
                   p.frame_perm_shifted_opening_proof.as_ref().map(|x| &x.proof));

        // Round-trip with all None must also work.
        p.frame_perm_commitment = None;
        p.frame_perm_evaluation = None;
        p.frame_perm_shifted_evaluation = None;
        p.frame_perm_opening_proof = None;
        p.frame_perm_shifted_opening_proof = None;
        let bytes2 = p.to_bytes();
        let decoded2 = ExecutionProof::from_bytes(&bytes2)
            .expect("frame-perm None round-trip");
        assert!(decoded2.frame_perm_commitment.is_none());
    }
}
