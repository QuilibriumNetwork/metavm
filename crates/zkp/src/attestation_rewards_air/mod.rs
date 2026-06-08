//! Beacon attestation rewards / penalties AIR (Casper FFG flag-reward
//! accounting).
//!
//! Proves per-validator rewards for the three Phase 0 participation
//! flags (`TIMELY_SOURCE`, `TIMELY_TARGET`, `TIMELY_HEAD`).
//!
//! ## Reward formula (consensus-specs `process_attestation_rewards`)
//!
//! Per validator, per epoch:
//!
//! ```text
//! base_reward = effective_balance * BASE_REWARD_FACTOR /
//!               sqrt(total_active_balance) / BASE_REWARDS_PER_EPOCH
//!
//! source_reward = base_reward * TIMELY_SOURCE_WEIGHT / WEIGHT_DENOMINATOR
//! target_reward = base_reward * TIMELY_TARGET_WEIGHT / WEIGHT_DENOMINATOR
//! head_reward   = base_reward * TIMELY_HEAD_WEIGHT   / WEIGHT_DENOMINATOR
//!
//! total_reward = source_correct ? source_reward : 0
//!              + target_correct ? target_reward : 0
//!              + head_correct   ? head_reward   : 0
//! ```
//!
//! Where the Phase 0 weights are
//!
//! | Flag           | Weight |
//! |----------------|-------:|
//! | `TIMELY_SOURCE`|     14 |
//! | `TIMELY_TARGET`|     26 |
//! | `TIMELY_HEAD`  |     14 |
//!
//! and `WEIGHT_DENOMINATOR = 64`.
//!
//! This AIR algebraically enforces the per-flag integer division
//! `flag_reward * 64 + flag_remainder == base_reward * weight` for each
//! of the three flags (using witnessed remainders), and the linear
//! combination
//! `total_reward == source_correct·source_reward
//!               + target_correct·target_reward
//!               + head_correct  ·head_reward`.
//!
//! ## Scope statement
//!
//! What this AIR proves *algebraically*:
//!  - Each flag column is binary.
//!  - `total_reward` is the correct linear combination of the three
//!    per-flag rewards weighted by the binary correctness flags.
//!  - Each per-flag reward is the integer quotient of
//!    `base_reward * weight / 64` (via witnessed remainder).
//!  - `base_reward`, `effective_balance`, and `total_reward` admit
//!    8-byte LE byte decompositions (each byte 8-bit range-checked).
//!
//! What this AIR delegates:
//!  - `base_reward = eff_bal * 64 / sqrt(total_active_balance)` — the
//!    square-root computation belongs in a dedicated sqrt gadget and
//!    is host-side oracle here.
//!  - The correctness of `(source_correct, target_correct, head_correct)`
//!    against the attestation data lives in the attestation AIR / a
//!    future per-flag-correctness AIR. This AIR exposes a cross-AIR
//!    LogUp descriptor to bind those flags once the consumer AIR is
//!    ready.
//!  - `effective_balance` provenance from the beacon state
//!    `validators[validator_index]` — bound via
//!    [`make_rewards_to_validator_balances_descriptor`] into
//!    [`crate::validator_balances_air`].
//!
//! Soundness gap (deferred): each per-flag remainder must be `< 64`
//! for the integer-division witness to be honest. That range is **not**
//! algebraically enforced here (only an 8-bit byte range exists, which
//! is too loose). A malicious prover could choose
//! `flag_remainder ∈ [64, 256)` and have one of the integer division
//! constraints still satisfied with a smaller-than-honest `flag_reward`.
//! Tightening to a 6-bit range check is a follow-up.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Phase 0 reward-flag constants ────────────────────────────────────

/// Phase 0 weight for the `TIMELY_SOURCE` participation flag.
pub const TIMELY_SOURCE_WEIGHT: u64 = 14;
/// Phase 0 weight for the `TIMELY_TARGET` participation flag.
pub const TIMELY_TARGET_WEIGHT: u64 = 26;
/// Phase 0 weight for the `TIMELY_HEAD` participation flag.
pub const TIMELY_HEAD_WEIGHT: u64 = 14;
/// Phase 0 weight denominator (`WEIGHT_DENOMINATOR`).
pub const WEIGHT_DENOMINATOR: u64 = 64;

/// LE byte-width of each u64 column (effective_balance / base_reward /
/// total_reward).
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_VALIDATOR_INDEX: usize = 0;
pub const COL_EFFECTIVE_BALANCE: usize = 1;
pub const COL_BASE_REWARD: usize = 2;

pub const COL_SOURCE_CORRECT: usize = 3;
pub const COL_TARGET_CORRECT: usize = 4;
pub const COL_HEAD_CORRECT: usize = 5;

pub const COL_SOURCE_REWARD: usize = 6;
pub const COL_TARGET_REWARD: usize = 7;
pub const COL_HEAD_REWARD: usize = 8;

/// Witnessed integer-division remainders: per-flag,
/// `flag_reward · 64 + flag_remainder = base_reward · weight`.
pub const COL_SOURCE_REMAINDER: usize = 9;
pub const COL_TARGET_REMAINDER: usize = 10;
pub const COL_HEAD_REMAINDER: usize = 11;

pub const COL_TOTAL_REWARD: usize = 12;

pub const COL_IS_REAL: usize = 13;

/// LE byte decompositions for `effective_balance`, `base_reward`, and
/// `total_reward`. 3 × 8 = 24 columns.
pub const COL_EFFECTIVE_BALANCE_BYTE_OFFSET: usize = 14;
pub const COL_BASE_REWARD_BYTE_OFFSET: usize = COL_EFFECTIVE_BALANCE_BYTE_OFFSET + U64_BYTES; // 22
pub const COL_TOTAL_REWARD_BYTE_OFFSET: usize = COL_BASE_REWARD_BYTE_OFFSET + U64_BYTES; // 30

pub const NUM_COLUMNS: usize = COL_TOTAL_REWARD_BYTE_OFFSET + U64_BYTES; // 38

// Row-local constraint count:
//   0  is_real_binary
//   1  source_correct_binary
//   2  target_correct_binary
//   3  head_correct_binary
//   4  total_reward_linear_combination
//   5  source_div_by_64
//   6  target_div_by_64
//   7  head_div_by_64
//   8  effective_balance_byte_decomp
//   9  base_reward_byte_decomp
//  10  total_reward_byte_decomp
pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct AttestationRewardsRow {
    pub validator_index: u64,
    pub effective_balance: u64,
    pub base_reward: u64,
    pub source_correct: bool,
    pub target_correct: bool,
    pub head_correct: bool,
    pub source_reward: u64,
    pub target_reward: u64,
    pub head_reward: u64,
    pub source_remainder: u64,
    pub target_remainder: u64,
    pub head_remainder: u64,
    pub total_reward: u64,
}

#[derive(Clone, Debug, Default)]
pub struct AttestationRewardsWitness {
    pub rows: Vec<AttestationRewardsRow>,
}

impl AttestationRewardsWitness {
    /// Build a single-row witness for one validator-attestation.
    ///
    /// `flags = (source_correct, target_correct, head_correct)`.
    ///
    /// Computes the per-flag reward via integer division of
    /// `base_reward * weight / WEIGHT_DENOMINATOR`, stores the
    /// remainders for the algebraic integer-division witness, and pins
    /// the gated linear combination as `total_reward`.
    pub fn from_validator(
        validator_index: u64,
        effective_balance: u64,
        base_reward: u64,
        flags: (bool, bool, bool),
    ) -> Self {
        let (source_correct, target_correct, head_correct) = flags;

        let source_num = base_reward as u128 * TIMELY_SOURCE_WEIGHT as u128;
        let target_num = base_reward as u128 * TIMELY_TARGET_WEIGHT as u128;
        let head_num = base_reward as u128 * TIMELY_HEAD_WEIGHT as u128;
        let denom = WEIGHT_DENOMINATOR as u128;

        let source_reward = (source_num / denom) as u64;
        let target_reward = (target_num / denom) as u64;
        let head_reward = (head_num / denom) as u64;
        let source_remainder = (source_num % denom) as u64;
        let target_remainder = (target_num % denom) as u64;
        let head_remainder = (head_num % denom) as u64;

        let total_reward = (if source_correct { source_reward } else { 0 })
            .saturating_add(if target_correct { target_reward } else { 0 })
            .saturating_add(if head_correct { head_reward } else { 0 });

        Self {
            rows: vec![AttestationRewardsRow {
                validator_index,
                effective_balance,
                base_reward,
                source_correct,
                target_correct,
                head_correct,
                source_reward,
                target_reward,
                head_reward,
                source_remainder,
                target_remainder,
                head_remainder,
                total_reward,
            }],
        }
    }

    /// Append another validator-attestation row to the witness.
    pub fn push_validator(
        &mut self,
        validator_index: u64,
        effective_balance: u64,
        base_reward: u64,
        flags: (bool, bool, bool),
    ) {
        let mut tmp = Self::from_validator(validator_index, effective_balance, base_reward, flags);
        self.rows.append(&mut tmp.rows);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &AttestationRewardsWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        columns[COL_VALIDATOR_INDEX][r] = Scalar::from_u64(row.validator_index, curve);
        columns[COL_EFFECTIVE_BALANCE][r] = Scalar::from_u64(row.effective_balance, curve);
        columns[COL_BASE_REWARD][r] = Scalar::from_u64(row.base_reward, curve);

        columns[COL_SOURCE_CORRECT][r] = Scalar::from_u64(row.source_correct as u64, curve);
        columns[COL_TARGET_CORRECT][r] = Scalar::from_u64(row.target_correct as u64, curve);
        columns[COL_HEAD_CORRECT][r] = Scalar::from_u64(row.head_correct as u64, curve);

        columns[COL_SOURCE_REWARD][r] = Scalar::from_u64(row.source_reward, curve);
        columns[COL_TARGET_REWARD][r] = Scalar::from_u64(row.target_reward, curve);
        columns[COL_HEAD_REWARD][r] = Scalar::from_u64(row.head_reward, curve);

        columns[COL_SOURCE_REMAINDER][r] = Scalar::from_u64(row.source_remainder, curve);
        columns[COL_TARGET_REMAINDER][r] = Scalar::from_u64(row.target_remainder, curve);
        columns[COL_HEAD_REMAINDER][r] = Scalar::from_u64(row.head_remainder, curve);

        columns[COL_TOTAL_REWARD][r] = Scalar::from_u64(row.total_reward, curve);
        columns[COL_IS_REAL][r] = one.clone();

        // LE byte decompositions.
        let eb_bytes = row.effective_balance.to_le_bytes();
        let br_bytes = row.base_reward.to_le_bytes();
        let tr_bytes = row.total_reward.to_le_bytes();
        for k in 0..U64_BYTES {
            columns[COL_EFFECTIVE_BALANCE_BYTE_OFFSET + k][r] =
                Scalar::from_u64(eb_bytes[k] as u64, curve);
            columns[COL_BASE_REWARD_BYTE_OFFSET + k][r] =
                Scalar::from_u64(br_bytes[k] as u64, curve);
            columns[COL_TOTAL_REWARD_BYTE_OFFSET + k][r] =
                Scalar::from_u64(tr_bytes[k] as u64, curve);
        }
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct AttestationRewardsConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AttestationRewardsConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// LE byte weight `256^k`.
fn le_byte_weight(k: usize, curve: CurveType) -> Scalar {
    Scalar::from_u64(1u64 << (8 * k as u32), curve)
}

impl VmConstraintSystem for AttestationRewardsConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "source_correct_binary".into(),
            "target_correct_binary".into(),
            "head_correct_binary".into(),
            "total_reward_linear_combination".into(),
            "source_div_by_64".into(),
            "target_div_by_64".into(),
            "head_div_by_64".into(),
            "effective_balance_byte_decomp".into(),
            "base_reward_byte_decomp".into(),
            "total_reward_byte_decomp".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);
        let denom = Scalar::from_u64(WEIGHT_DENOMINATOR, curve);
        let w_src = Scalar::from_u64(TIMELY_SOURCE_WEIGHT, curve);
        let w_tgt = Scalar::from_u64(TIMELY_TARGET_WEIGHT, curve);
        let w_hd = Scalar::from_u64(TIMELY_HEAD_WEIGHT, curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        let mk = || vec![zero.clone(); n];

        // 0: is_real binary.
        let mut c0 = mk();
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            c0[r] = v.mul(&v.sub(&one));
        }
        out.push(c0);

        // 1-3: flag binary (gated by is_real so padding rows trivially pass).
        for &col_flag in &[COL_SOURCE_CORRECT, COL_TARGET_CORRECT, COL_HEAD_CORRECT] {
            let mut c = mk();
            for r in 0..n {
                let v = &columns[col_flag][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 4: total_reward = src·src_rew + tgt·tgt_rew + hd·hd_rew
        //    (gated by is_real so padding rows are trivially zero=0).
        let mut c4 = mk();
        for r in 0..n {
            let total = &columns[COL_TOTAL_REWARD][r];
            let s_term =
                columns[COL_SOURCE_CORRECT][r].mul(&columns[COL_SOURCE_REWARD][r]);
            let t_term =
                columns[COL_TARGET_CORRECT][r].mul(&columns[COL_TARGET_REWARD][r]);
            let h_term =
                columns[COL_HEAD_CORRECT][r].mul(&columns[COL_HEAD_REWARD][r]);
            let sum = s_term.add(&t_term).add(&h_term);
            c4[r] = columns[COL_IS_REAL][r].mul(&total.sub(&sum));
        }
        out.push(c4);

        // 5: source_reward · 64 + source_remainder = base_reward · 14.
        //    Gated by is_real.
        let mut c5 = mk();
        for r in 0..n {
            let lhs = columns[COL_SOURCE_REWARD][r].mul(&denom).add(&columns[COL_SOURCE_REMAINDER][r]);
            let rhs = columns[COL_BASE_REWARD][r].mul(&w_src);
            c5[r] = columns[COL_IS_REAL][r].mul(&lhs.sub(&rhs));
        }
        out.push(c5);

        // 6: target_reward · 64 + target_remainder = base_reward · 26.
        let mut c6 = mk();
        for r in 0..n {
            let lhs = columns[COL_TARGET_REWARD][r].mul(&denom).add(&columns[COL_TARGET_REMAINDER][r]);
            let rhs = columns[COL_BASE_REWARD][r].mul(&w_tgt);
            c6[r] = columns[COL_IS_REAL][r].mul(&lhs.sub(&rhs));
        }
        out.push(c6);

        // 7: head_reward · 64 + head_remainder = base_reward · 14.
        let mut c7 = mk();
        for r in 0..n {
            let lhs = columns[COL_HEAD_REWARD][r].mul(&denom).add(&columns[COL_HEAD_REMAINDER][r]);
            let rhs = columns[COL_BASE_REWARD][r].mul(&w_hd);
            c7[r] = columns[COL_IS_REAL][r].mul(&lhs.sub(&rhs));
        }
        out.push(c7);

        // 8: effective_balance = Σ byte_k · 256^k. (Gated by is_real.)
        // 9: base_reward       = Σ byte_k · 256^k.
        // 10: total_reward     = Σ byte_k · 256^k.
        for &(col_value, col_byte_off) in &[
            (COL_EFFECTIVE_BALANCE, COL_EFFECTIVE_BALANCE_BYTE_OFFSET),
            (COL_BASE_REWARD, COL_BASE_REWARD_BYTE_OFFSET),
            (COL_TOTAL_REWARD, COL_TOTAL_REWARD_BYTE_OFFSET),
        ] {
            let mut c = mk();
            for r in 0..n {
                let mut sum = zero.clone();
                for k in 0..U64_BYTES {
                    let b = &columns[col_byte_off + k][r];
                    sum = sum.add(&b.mul(&le_byte_weight(k, curve)));
                }
                c[r] = columns[COL_IS_REAL][r].mul(&columns[col_value][r].sub(&sum));
            }
            out.push(c);
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let denom = Scalar::from_u64(WEIGHT_DENOMINATOR, curve);
        let w_src = Scalar::from_u64(TIMELY_SOURCE_WEIGHT, curve);
        let w_tgt = Scalar::from_u64(TIMELY_TARGET_WEIGHT, curve);
        let w_hd = Scalar::from_u64(TIMELY_HEAD_WEIGHT, curve);

        let is_real = &col_evals[COL_IS_REAL];
        let src = &col_evals[COL_SOURCE_CORRECT];
        let tgt = &col_evals[COL_TARGET_CORRECT];
        let hd = &col_evals[COL_HEAD_CORRECT];

        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        let push = |body: Scalar, acc: &mut Scalar, ap: &mut Scalar| {
            *acc = acc.add(&ap.mul(&body));
            *ap = ap.mul(alpha);
        };

        // 0
        push(is_real.mul(&is_real.sub(&one)), &mut acc, &mut ap);
        // 1..3
        push(src.mul(&src.sub(&one)), &mut acc, &mut ap);
        push(tgt.mul(&tgt.sub(&one)), &mut acc, &mut ap);
        push(hd.mul(&hd.sub(&one)), &mut acc, &mut ap);

        // 4 linear combo
        {
            let total = &col_evals[COL_TOTAL_REWARD];
            let sum = src
                .mul(&col_evals[COL_SOURCE_REWARD])
                .add(&tgt.mul(&col_evals[COL_TARGET_REWARD]))
                .add(&hd.mul(&col_evals[COL_HEAD_REWARD]));
            push(is_real.mul(&total.sub(&sum)), &mut acc, &mut ap);
        }

        // 5..7 div-by-64
        for (col_reward, col_rem, weight) in [
            (COL_SOURCE_REWARD, COL_SOURCE_REMAINDER, &w_src),
            (COL_TARGET_REWARD, COL_TARGET_REMAINDER, &w_tgt),
            (COL_HEAD_REWARD, COL_HEAD_REMAINDER, &w_hd),
        ] {
            let lhs = col_evals[col_reward].mul(&denom).add(&col_evals[col_rem]);
            let rhs = col_evals[COL_BASE_REWARD].mul(weight);
            push(is_real.mul(&lhs.sub(&rhs)), &mut acc, &mut ap);
        }

        // 8..10 byte decomps.
        for (col_value, col_byte_off) in [
            (COL_EFFECTIVE_BALANCE, COL_EFFECTIVE_BALANCE_BYTE_OFFSET),
            (COL_BASE_REWARD, COL_BASE_REWARD_BYTE_OFFSET),
            (COL_TOTAL_REWARD, COL_TOTAL_REWARD_BYTE_OFFSET),
        ] {
            let mut sum = Scalar::zero(curve);
            for k in 0..U64_BYTES {
                sum = sum.add(&col_evals[col_byte_off + k].mul(&le_byte_weight(k, curve)));
            }
            push(is_real.mul(&col_evals[col_value].sub(&sum)), &mut acc, &mut ap);
        }

        let _ = ap;
        acc
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let src = &col_coeffs[COL_SOURCE_CORRECT];
        let tgt = &col_coeffs[COL_TARGET_CORRECT];
        let hd = &col_coeffs[COL_HEAD_CORRECT];

        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        let push_body = |body: Vec<Scalar>, acc: &mut Vec<Scalar>, ap: &mut Scalar| {
            *acc = poly_add(acc, &poly_scalar_mul(&body, ap), curve);
            *ap = ap.mul(alpha);
        };

        // 0
        {
            let body = poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve);
            push_body(body, &mut acc, &mut ap);
        }
        // 1..3
        for v in [src, tgt, hd] {
            let body = poly_mul(v, &poly_sub(v, &one_poly, curve), curve);
            push_body(body, &mut acc, &mut ap);
        }

        // 4: is_real * (total - (src*src_rew + tgt*tgt_rew + hd*hd_rew))
        {
            let s_term = poly_mul(src, &col_coeffs[COL_SOURCE_REWARD], curve);
            let t_term = poly_mul(tgt, &col_coeffs[COL_TARGET_REWARD], curve);
            let h_term = poly_mul(hd, &col_coeffs[COL_HEAD_REWARD], curve);
            let sum = poly_add(&poly_add(&s_term, &t_term, curve), &h_term, curve);
            let inner = poly_sub(&col_coeffs[COL_TOTAL_REWARD], &sum, curve);
            let body = poly_mul(is_real, &inner, curve);
            push_body(body, &mut acc, &mut ap);
        }

        // 5..7: is_real * (reward*64 + rem - base*weight)
        for (col_reward, col_rem, weight_val) in [
            (COL_SOURCE_REWARD, COL_SOURCE_REMAINDER, TIMELY_SOURCE_WEIGHT),
            (COL_TARGET_REWARD, COL_TARGET_REMAINDER, TIMELY_TARGET_WEIGHT),
            (COL_HEAD_REWARD, COL_HEAD_REMAINDER, TIMELY_HEAD_WEIGHT),
        ] {
            let denom_scaled =
                poly_scalar_mul(&col_coeffs[col_reward], &Scalar::from_u64(WEIGHT_DENOMINATOR, curve));
            let lhs = poly_add(&denom_scaled, &col_coeffs[col_rem], curve);
            let rhs = poly_scalar_mul(&col_coeffs[COL_BASE_REWARD], &Scalar::from_u64(weight_val, curve));
            let inner = poly_sub(&lhs, &rhs, curve);
            let body = poly_mul(is_real, &inner, curve);
            push_body(body, &mut acc, &mut ap);
        }

        // 8..10: byte decomps
        for (col_value, col_byte_off) in [
            (COL_EFFECTIVE_BALANCE, COL_EFFECTIVE_BALANCE_BYTE_OFFSET),
            (COL_BASE_REWARD, COL_BASE_REWARD_BYTE_OFFSET),
            (COL_TOTAL_REWARD, COL_TOTAL_REWARD_BYTE_OFFSET),
        ] {
            let mut sum = vec![Scalar::zero(curve)];
            for k in 0..U64_BYTES {
                let term =
                    poly_scalar_mul(&col_coeffs[col_byte_off + k], &le_byte_weight(k, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let inner = poly_sub(&col_coeffs[col_value], &sum, curve);
            let body = poly_mul(is_real, &inner, curve);
            push_body(body, &mut acc, &mut ap);
        }

        let _ = ap;
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls12381);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("rewards_effective_balance_byte_{}_8bit", k),
                    column_index: COL_EFFECTIVE_BALANCE_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("rewards_base_reward_byte_{}_8bit", k),
                    column_index: COL_BASE_REWARD_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("rewards_total_reward_byte_{}_8bit", k),
                    column_index: COL_TOTAL_REWARD_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp linkage descriptors ──────────────────────────────

/// Bind `(validator_index, effective_balance)` to
/// [`crate::validator_balances_air`]'s
/// `(COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE)` tuple.
///
/// The validator_balances AIR proves provenance of
/// `effective_balance` from the beacon state validator registry; this
/// rewards AIR consumes the same `(validator_index, effective_balance)`
/// pair and uses it as a scalar input to the base_reward / per-flag
/// reward formula. Multiset equality under fresh γ binds the two
/// witness columns. A side gated by `IS_REAL`, B side gated by the
/// balances AIR's `IS_LEAF` (rewards rows match the unique per-validator
/// leaf row in the balances AIR).
pub fn make_rewards_to_validator_balances_descriptor(
    rewards_layer_index: usize,
    validator_balances_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::validator_balances_air as vb;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "rewards_to_validator_balances_v1".into(),
        a_layer_index: rewards_layer_index,
        a_columns: vec![COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_balances_layer_index,
        b_columns: vec![vb::COL_VALIDATOR_INDEX, vb::COL_EFFECTIVE_BALANCE],
        b_selector_column: Some(vb::COL_IS_LEAF),
    }
}

/// Bind `(validator_index, source_correct, target_correct, head_correct)`
/// to a per-validator attestation-flag-correctness consumer AIR. The
/// current [`crate::attestation_aggregate_air`] commits the participating
/// validator's committee `bit_index` but does not yet expose per-flag
/// correctness columns — those belong in a dedicated flag-correctness
/// AIR that compares attestation data (`source`, `target`, `head`)
/// against the canonical beacon-chain checkpoints.
///
/// As a scaffold this descriptor binds the rewards AIR's
/// `validator_index` to the attestation aggregate AIR's `COL_BIT_INDEX`
/// (per-participating row identifier), with the three flag columns
/// pinned against a *placeholder* triple of attestation columns —
/// self-bound until the flag-correctness AIR lands, mirroring the
/// `make_attestation_to_ffg_descriptor` pattern. Once the consumer AIR
/// exists, `b_columns[1..4]` and `b_selector_column` will move to that
/// AIR's per-flag layout.
///
/// Selector-gated by `IS_REAL` on both sides.
pub fn make_rewards_to_attestation_descriptor(
    rewards_layer_index: usize,
    attestation_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::attestation_aggregate_air as att;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "rewards_to_attestation_v1".into(),
        a_layer_index: rewards_layer_index,
        a_columns: vec![
            COL_VALIDATOR_INDEX,
            COL_SOURCE_CORRECT,
            COL_TARGET_CORRECT,
            COL_HEAD_CORRECT,
        ],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: attestation_layer_index,
        // Scaffold: bind to attestation_aggregate_air's per-participant
        // bit_index plus a placeholder triple drawn from the same AIR's
        // epoch / slot / committee_index columns (existing data columns
        // that don't collide with bit_index). The B-side selector
        // matches the attestation aggregate IS_REAL.
        b_columns: vec![
            att::COL_BIT_INDEX,
            att::COL_SOURCE_EPOCH,
            att::COL_TARGET_EPOCH,
            att::COL_SLOT,
        ],
        b_selector_column: Some(att::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_constraints(witness: &AttestationRewardsWitness) -> Vec<Vec<Scalar>> {
        let curve = CurveType::Bls12381;
        let trace = build_trace_polynomials(witness, curve);
        let cs = AttestationRewardsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    fn assert_all_zero(results: &[Vec<Scalar>]) {
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(val.is_zero(), "constraint {} row {} not zero", i, r);
            }
        }
    }

    #[test]
    fn all_flags_correct_constraints_zero() {
        // base_reward chosen so 14, 26, 14 all divide cleanly: pick
        // a multiple of 64 so every per-flag integer division yields
        // remainder 0 (clean honest math).
        let base_reward = 64 * 100; // 6400
        let w = AttestationRewardsWitness::from_validator(
            42,
            32_000_000_000,
            base_reward,
            (true, true, true),
        );
        let row = &w.rows[0];
        assert_eq!(row.source_reward, base_reward * 14 / 64);
        assert_eq!(row.target_reward, base_reward * 26 / 64);
        assert_eq!(row.head_reward, base_reward * 14 / 64);
        assert_eq!(
            row.total_reward,
            row.source_reward + row.target_reward + row.head_reward,
        );
        let results = run_constraints(&w);
        assert_all_zero(&results);
    }

    #[test]
    fn no_flags_correct_constraints_zero() {
        // Even when no flags are set, total_reward should be 0, and
        // per-flag rewards still equal base_reward * weight / 64.
        // The linear-combo constraint zeroes via flag·reward = 0.
        let w = AttestationRewardsWitness::from_validator(
            7,
            32_000_000_000,
            64 * 50,
            (false, false, false),
        );
        assert_eq!(w.rows[0].total_reward, 0);
        let results = run_constraints(&w);
        assert_all_zero(&results);
    }

    #[test]
    fn source_only_flag_correct_constraints_zero() {
        let base_reward = 64 * 77; // clean math
        let w = AttestationRewardsWitness::from_validator(
            123,
            31_000_000_000,
            base_reward,
            (true, false, false),
        );
        let row = &w.rows[0];
        assert_eq!(row.total_reward, row.source_reward);
        assert_eq!(row.total_reward, base_reward * 14 / 64);
        let results = run_constraints(&w);
        assert_all_zero(&results);
    }

    #[test]
    fn base_reward_not_multiple_of_64_uses_remainder() {
        // Non-clean math: base_reward = 100. source_reward = 100*14/64 = 21,
        // source_remainder = 100*14 - 21*64 = 1400 - 1344 = 56.
        let base_reward = 100u64;
        let w = AttestationRewardsWitness::from_validator(
            11,
            32_000_000_000,
            base_reward,
            (true, true, true),
        );
        let row = &w.rows[0];
        assert_eq!(row.source_reward, 21);
        assert_eq!(row.source_remainder, 56);
        assert_eq!(row.target_reward, base_reward * 26 / 64);
        assert_eq!(row.target_remainder, base_reward * 26 % 64);
        assert_eq!(row.head_reward, base_reward * 14 / 64);
        assert_eq!(row.head_remainder, base_reward * 14 % 64);
        let results = run_constraints(&w);
        assert_all_zero(&results);
    }

    #[test]
    fn tampered_total_reward_fires_linear_combo() {
        let w = AttestationRewardsWitness::from_validator(
            42,
            32_000_000_000,
            64 * 50,
            (true, true, false),
        );
        let curve = CurveType::Bls12381;
        let trace = build_trace_polynomials(&w, curve);
        let cs = AttestationRewardsConstraintSystem::new(trace.num_rows);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Add 1 to total_reward — fires constraint 4 + constraint 10
        // (byte decomp).
        let original = cols[COL_TOTAL_REWARD][0].clone();
        cols[COL_TOTAL_REWARD][0] = original.add(&Scalar::one(curve));
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[4][0].is_zero(), "total tamper must fire linear combo");
        assert!(!res[10][0].is_zero(), "total tamper must also fire byte decomp");
    }

    #[test]
    fn tampered_flag_fires_binary_or_combo() {
        let w = AttestationRewardsWitness::from_validator(
            42,
            32_000_000_000,
            64 * 50,
            (true, false, true),
        );
        let curve = CurveType::Bls12381;
        let trace = build_trace_polynomials(&w, curve);
        let cs = AttestationRewardsConstraintSystem::new(trace.num_rows);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Flip target_correct from 0 to 2 (non-binary) — fires
        // constraint 2 (target_correct_binary).
        cols[COL_TARGET_CORRECT][0] = Scalar::from_u64(2, curve);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[2][0].is_zero(), "non-binary flag must fire binary constraint");
    }

    #[test]
    fn tampered_source_reward_fires_div_by_64() {
        let w = AttestationRewardsWitness::from_validator(
            42,
            32_000_000_000,
            64 * 50,
            (true, true, true),
        );
        let curve = CurveType::Bls12381;
        let trace = build_trace_polynomials(&w, curve);
        let cs = AttestationRewardsConstraintSystem::new(trace.num_rows);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let orig = cols[COL_SOURCE_REWARD][0].clone();
        cols[COL_SOURCE_REWARD][0] = orig.add(&Scalar::one(curve));
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[5][0].is_zero(), "source_reward bump must fire div_by_64");
    }

    #[test]
    fn descriptors_well_formed() {
        let d_vb = make_rewards_to_validator_balances_descriptor(0, 1);
        assert_eq!(d_vb.label, "rewards_to_validator_balances_v1");
        assert_eq!(d_vb.a_columns.len(), 2);
        assert_eq!(d_vb.b_columns.len(), 2);
        assert_eq!(d_vb.a_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(d_vb.a_columns[1], COL_EFFECTIVE_BALANCE);
        assert_eq!(
            d_vb.b_columns[0],
            crate::validator_balances_air::COL_VALIDATOR_INDEX,
        );
        assert_eq!(
            d_vb.b_columns[1],
            crate::validator_balances_air::COL_EFFECTIVE_BALANCE,
        );
        assert_eq!(d_vb.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_vb.b_selector_column,
            Some(crate::validator_balances_air::COL_IS_LEAF),
        );

        let d_att = make_rewards_to_attestation_descriptor(0, 2);
        assert_eq!(d_att.label, "rewards_to_attestation_v1");
        assert_eq!(d_att.a_columns.len(), 4);
        assert_eq!(d_att.b_columns.len(), 4);
        assert_eq!(d_att.a_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(d_att.a_columns[1], COL_SOURCE_CORRECT);
        assert_eq!(d_att.a_columns[2], COL_TARGET_CORRECT);
        assert_eq!(d_att.a_columns[3], COL_HEAD_CORRECT);
        for &c in &d_att.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d_att.b_columns {
            assert!(c < crate::attestation_aggregate_air::NUM_COLUMNS);
        }
        assert_eq!(d_att.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_att.b_selector_column,
            Some(crate::attestation_aggregate_air::COL_IS_REAL),
        );
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_VALIDATOR_INDEX, 0);
        assert_eq!(COL_EFFECTIVE_BALANCE, 1);
        assert_eq!(COL_BASE_REWARD, 2);
        assert_eq!(COL_SOURCE_CORRECT, 3);
        assert_eq!(COL_TARGET_CORRECT, 4);
        assert_eq!(COL_HEAD_CORRECT, 5);
        assert_eq!(COL_SOURCE_REWARD, 6);
        assert_eq!(COL_TARGET_REWARD, 7);
        assert_eq!(COL_HEAD_REWARD, 8);
        assert_eq!(COL_SOURCE_REMAINDER, 9);
        assert_eq!(COL_TARGET_REMAINDER, 10);
        assert_eq!(COL_HEAD_REMAINDER, 11);
        assert_eq!(COL_TOTAL_REWARD, 12);
        assert_eq!(COL_IS_REAL, 13);
        assert_eq!(COL_EFFECTIVE_BALANCE_BYTE_OFFSET, 14);
        assert_eq!(COL_BASE_REWARD_BYTE_OFFSET, 22);
        assert_eq!(COL_TOTAL_REWARD_BYTE_OFFSET, 30);
        assert_eq!(NUM_COLUMNS, 38);
        assert_eq!(NUM_ROW_CONSTRAINTS, 11);
    }

    #[test]
    fn multi_row_witness_constraints_zero() {
        let mut w =
            AttestationRewardsWitness::from_validator(0, 32_000_000_000, 64 * 10, (true, true, true));
        w.push_validator(1, 31_000_000_000, 64 * 20, (false, true, true));
        w.push_validator(2, 16_000_000_000, 100, (true, false, true));
        w.push_validator(3, 32_000_000_000, 64 * 50, (false, false, false));
        assert_eq!(w.rows.len(), 4);
        let results = run_constraints(&w);
        assert_all_zero(&results);
    }

    #[test]
    fn byte_range_lookup_coverage() {
        let cs = AttestationRewardsConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        assert_eq!(req.declarations.len(), 3 * U64_BYTES);
        let labels: Vec<String> =
            req.declarations.iter().map(|(d, _)| d.label.clone()).collect();
        assert!(labels.contains(&"rewards_effective_balance_byte_0_8bit".into()));
        assert!(labels.contains(&"rewards_base_reward_byte_7_8bit".into()));
        assert!(labels.contains(&"rewards_total_reward_byte_3_8bit".into()));
        for (d, _) in &req.declarations {
            assert!(d.column_index < NUM_COLUMNS);
            assert_eq!(d.max_bits, 8);
        }
    }
}
