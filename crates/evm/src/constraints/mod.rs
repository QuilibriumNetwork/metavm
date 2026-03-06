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

use metavm_zkp::field::Scalar;
use metavm_zkp::vm_constraints::VmConstraintSystem;
use metavm_zkp::poly_arith;
use metavm_zkp::lookup::{LookupRequirements, LookupTable, LookupDeclaration, BitwiseLookupDeclaration, BitwiseOp};
use crate::trace;

/// Number of VM-specific constraints (gated by selectors).
///
/// 8 ADD (4 limb + 4 carry binary) + 8 SUB (4 limb + 4 borrow binary)
/// + 1 mul_full + 1 div_full + 1 mod_full
/// + 1 sdiv_full + 1 smod_full + 1 addmod_oracle
/// + 1 mulmod_oracle + 1 exp_oracle + 1 signextend_oracle
/// + 1 LT + 1 GT + 1 EQ + 1 ISZERO + 1 compare_other
/// + 4 AND (per-limb) + 4 OR (per-limb) + 4 XOR (per-limb)
/// + 1 SHL + 1 SHR + 1 SAR
/// + 1 push + 1 dup + 1 pop + 1 swap
/// + 1 mload + 1 mstore + 1 mstore8 + 1 msize
/// + 1 control
/// + 1 NOT (bitwise_other)
/// + 9 oracle (stop, keccak, env, block,
///              stack_other, memory_other, storage, log, call) = 64.
const NUM_VM_CONSTRAINTS: usize = 64;

/// Number of selector columns in the one-hot encoding.
const NUM_SELECTORS: usize = 41;

/// Total constraints: 64 VM + 41 selector-binary + 1 selector-sum = 106.
const TOTAL_CONSTRAINTS: usize = NUM_VM_CONSTRAINTS + NUM_SELECTORS + 1;

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
            "sel_addmod * addmod_oracle".to_string(),
            "sel_mulmod * mulmod_oracle".to_string(),
            "sel_exp * exp_oracle".to_string(),
            "sel_signextend * signextend_oracle".to_string(),
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
        ];
        for name in &sel_names {
            labels.push(format!("{}_binary", name));
        }

        // Selector sum-to-one constraint
        labels.push("selector_sum_to_one".to_string());

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

        // --- Oracle constraints (constraints 21-24) ---
        // ADDMOD, MULMOD, EXP, SIGNEXTEND use zero-body oracle constraints.
        let oracle_zero = vec![zero.clone(); num_rows];

        let mut result = vec![
            add_l0, add_l1, add_l2, add_l3,
            carry0_bin, carry1_bin, carry2_bin, carry3_bin,
            sub_l0, sub_l1, sub_l2, sub_l3,
            borrow0_bin, borrow1_bin, borrow2_bin, borrow3_bin,
            mul_full, div_full, mod_full,
            sdiv_full,
            smod_full,
            oracle_zero.clone(), // addmod
            oracle_zero.clone(), // mulmod
            oracle_zero.clone(), // exp
            oracle_zero,         // signextend
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
            // AND constraints: 4 per-limb (constraints 23-26)
            evaluate_and(columns, num_rows),
            evaluate_and_limb(columns, num_rows, 1),
            evaluate_and_limb(columns, num_rows, 2),
            evaluate_and_limb(columns, num_rows, 3),
            // OR constraints: 4 per-limb (constraints 27-30)
            evaluate_or(columns, num_rows),
            evaluate_or_limb(columns, num_rows, 1),
            evaluate_or_limb(columns, num_rows, 2),
            evaluate_or_limb(columns, num_rows, 3),
            // XOR constraints: 4 per-limb (constraints 31-34)
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

        // --- Constraints 21-24: oracle constraints (zero body) for ADDMOD, MULMOD, EXP, SIGNEXTEND ---
        for _oracle_sel in &[
            trace::COL_SEL_ADDMOD,
            trace::COL_SEL_MULMOD, trace::COL_SEL_EXP, trace::COL_SEL_SIGNEXTEND,
        ] {
            // body = 0, so sel * 0 * alpha^k = 0. Just advance alpha.
            ap = ap.mul(alpha);
        }

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

        // --- Constraints 21-24: oracle constraints (zero body) ---
        // ADDMOD, MULMOD, EXP, SIGNEXTEND: no pairs, just oracle skips.

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
        // pairs[0..21]  = constraints 0-20  (21 non-oracle)
        // 4 oracle skips = constraints 21-24
        // pairs[21..48] = constraints 25-51 (27 non-oracle)
        // 1 oracle skip  = constraint 52 (msize)
        // pairs[48]      = constraint 53 (jump)
        // 1 oracle skip  = constraint 54 (stop)
        // pairs[49]      = constraint 55 (NOT)
        // 8 oracle skips = constraints 56-63
        let mut c = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);

        // Accumulate constraints 0-20 (pairs[0..21])
        for gated in &gated_results[..21] {
            c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(gated, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Skip 4 oracle constraints (21-24: ADDMOD, MULMOD, EXP, SIGNEXTEND)
        for _ in 0..4 {
            ap = ap.mul(alpha);
        }

        // Accumulate constraints 25-51 (pairs[21..48])
        for gated in &gated_results[21..48] {
            c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(gated, &ap), curve);
            ap = ap.mul(alpha);
        }

        // Skip 1 oracle constraint (52: MSIZE)
        ap = ap.mul(alpha);

        // Accumulate constraint 53 (pairs[48]: JUMP)
        c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(&gated_results[48], &ap), curve);
        ap = ap.mul(alpha);

        // Skip 1 oracle constraint (54: STOP)
        ap = ap.mul(alpha);

        // Accumulate constraint 55 (pairs[49]: NOT)
        c = poly_arith::poly_add(&c, &poly_arith::poly_scalar_mul(&gated_results[49], &ap), curve);
        ap = ap.mul(alpha);

        // Skip 8 oracle constraints (56-63: keccak, env, block, stack_other, memory_other, storage, log, call)
        for _ in 0..8 {
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
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        vec![trace::COL_PC]
    }

    fn num_shifted_constraints(&self) -> usize {
        1
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

        // alpha^offset
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // PC continuity: pc(omega*z) - next_pc(z) = 0
        let pc_next = &shifted_evals[0]; // col_pc evaluated at omega*z
        let next_pc = &col_evals_at_z[trace::COL_NEXT_PC];
        let body = pc_next.sub(next_pc);

        // Multiply by (z - omega^{n-1}) to exclude wrap-around row
        let exclusion = z.sub(omega_n_minus_1);

        ap.mul(&body).mul(&exclusion)
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

        let col = |idx: usize| -> &Vec<Scalar> { &column_coeffs[idx] };

        // alpha^offset
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }

        // PC continuity: poly_shift(col_pc, omega) - col_next_pc
        let pc_shifted = poly_arith::poly_shift(col(trace::COL_PC), omega);
        let body = poly_arith::poly_sub(&pc_shifted, col(trace::COL_NEXT_PC), curve);

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

        // Multiply by (X - omega^{n-1})
        let body_excluded = poly_arith::poly_mul_linear(&body, &omega_n_minus_1);

        poly_arith::poly_scalar_mul(&body_excluded, &ap)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let range_16 = LookupTable::range(16);
        let mut decls = Vec::new();

        // MUL: aux0 limbs hold high 256 bits of 512-bit product.
        // Each limb must be < 2^64 for the multiplication constraint to be sound.
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
            vec![trace::COL_SEL_MSTORE],
        ))
    }

    fn oracle_selectors(&self) -> Vec<usize> {
        vec![
            trace::COL_SEL_ADDMOD, trace::COL_SEL_MULMOD,
            trace::COL_SEL_EXP, trace::COL_SEL_SIGNEXTEND,
            trace::COL_SEL_KECCAK, trace::COL_SEL_ENV,
            trace::COL_SEL_BLOCK,
            trace::COL_SEL_STACK_OTHER,
            trace::COL_SEL_MSIZE,
            trace::COL_SEL_MEMORY_OTHER,
            trace::COL_SEL_STORAGE, trace::COL_SEL_LOG,
            trace::COL_SEL_CALL,
        ]
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

#[cfg(test)]
mod tests {
    use super::*;
    use metavm_zkp::field::CurveType;
    use crate::trace::*;

    fn make_add_trace() -> EvmTraceColumns {
        let mut trace = EvmTraceColumns::new();
        // ADD: 10 + 20 = 30
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
            next_pc: 0,
        };
        trace.push_row(&row);
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
        // 64 VM + 41 binary + 1 sum = 106
        assert_eq!(cs.num_constraints(), 106);
        assert_eq!(cs.constraint_labels().len(), 106);
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
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Constraint 46 (sel_dup * dup_raw) should fail
        assert!(
            !eval[46][0].is_zero(),
            "Invalid DUP should fail constraint 46 (sel_dup * dup_raw)"
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
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // Constraint 50 (sel_mstore * mstore_raw) should fail
        assert!(
            !eval[50][0].is_zero(),
            "Invalid MSTORE should fail constraint 50 (sel_mstore * mstore_raw)"
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
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // AND constraint 30 (sel_and * and_limb0) should fail
        assert!(
            !eval[30][0].is_zero(),
            "Invalid AND should fail constraint 30 (sel_and * and_limb0)"
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
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // SHL constraint 42 should fail
        assert!(
            !eval[42][0].is_zero(),
            "Invalid SHL should fail constraint 42 (sel_shl * shl_raw)"
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
        };
        trace.push_row(&row);

        let polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&trace, CurveType::Bls48581);
        let cs = EvmConstraintSystem::new();
        let columns = polys.columns();
        let eval = cs.evaluate_on_domain(&columns, polys.num_rows);

        // SHR constraint 43 should fail with bad carry chain
        assert!(
            !eval[43][0].is_zero(),
            "SHR with bad carry chain should fail constraint 43 (sel_shr * shr_raw)"
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
}
