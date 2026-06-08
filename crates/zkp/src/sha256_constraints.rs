//! [`VmConstraintSystem`] wiring for the SHA-256 bit-level AIR.
//!
//! This module adapts the constraint bodies defined in
//! [`crate::sha256_air`] into the shape required by the generic
//! [`prove_with_scheme`] / [`verify_with_scheme`] pipeline. It provides:
//!
//! - `Sha256ConstraintSystem`: the trait implementer. Construct with
//!   `Sha256ConstraintSystem::new(num_rows)` where `num_rows` is the
//!   real, pre-padding trace height (= `64 * num_blocks`).
//! - `build_trace_polynomials_from_rounds`: helper that lifts a
//!   `Vec<RoundTrace>` into the `TracePolynomials` the pipeline expects.
//!
//! # Constraint layout
//!
//! 10 row-local consolidated categories (label → index):
//!
//!   0.  `bit_validity`
//!   1.  `big_sigma0_a_definition`
//!   2.  `big_sigma1_e_definition`
//!   3.  `ch_definition`
//!   4.  `maj_definition`
//!   5.  `round_additions`
//!   6.  `passthrough`
//!   7.  `k_binding`
//!   8.  `sel_binary`
//!   9.  `sel_sum_01`
//!
//! 1 shifted (cross-row) constraint: `cross_row_transition`.
//!
//! Within each category, bit-level sub-constraints are aggregated via
//! powers of a β challenge; we fix β = α (single challenge) for
//! simplicity. The combined constraint polynomial remains a specific
//! polynomial in α and Schwartz-Zippel still applies.
//!
//! # Padding strategy
//!
//! We return `None` from [`VmConstraintSystem::padding_selector_column`] so
//! every bit column (including all 64 round selectors) is zero on padding
//! rows — this makes every row-local body vanish trivially. The selector
//! sum-to-one constraint is enforced as `sum · (sum − 1) = 0`, which
//! admits the sum = 0 case on padding.
//!
//! For the cross-row `after(X)[i] == before(ω·X)[i]` transition, we
//! exclude the wrap-around row (ω^{n-1}) and every block boundary row
//! (rows where `(r+1) mod 64 == 0`, i.e. r = 63, 127, …). Across a block
//! boundary the next row's `before` is the next block's public
//! `state_in`, not chained from this row's `after`.
//!
//! # Lookup declarations
//!
//! The SHA-256 AIR is purely bit-level — every data column already carries
//! a `b·(b−1) = 0` constraint via the `bit_validity` category — so
//! external range checks would be redundant. We forward an empty
//! [`LookupRequirements`] for now.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::sha256::{HashTrace, RoundTrace, INITIAL_HASH, NUM_ROUNDS, ROUND_CONSTANTS};
use crate::sha256_air::{
    self, ab_bit, ac_bit, after, alloc_trace, bc_bit, before, big_sigma0_a_bit, big_sigma1_e_bit,
    ch_efg_bit, ef_bit, k_bit, maj_abc_bit, not_e_g_bit, populate_round, populate_trace_from_hash,
    sel_round, xor01_s0_bit, xor01_s1_bit, xor_ab_ac_bit, BITS_PER_STATE, BITS_PER_WORD,
    COL_A_NEW_CARRY_OFFSET, COL_AFTER_OFFSET, COL_BEFORE_OFFSET, COL_BIG_SIGMA0_A_OFFSET,
    COL_BIG_SIGMA1_E_OFFSET, COL_CH_EFG_OFFSET, COL_E_NEW_CARRY_OFFSET, COL_K_OFFSET,
    COL_MAJ_ABC_OFFSET, COL_T1_CARRY_OFFSET, COL_T2_CARRY_OFFSET, COL_W_OFFSET,
    NUM_DATA_COLUMNS, NUM_SEL_ROUND, NUM_SHA256_COLUMNS, NUM_WORKING_VARS, T1_CARRY_BITS, VAR_A,
    VAR_B, VAR_C, VAR_D, VAR_E, VAR_F, VAR_G, VAR_H,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Number of consolidated row-local constraint categories. Layout:
///   - 10 (= [`sha256_air::NUM_CONSTRAINT_CATEGORIES`]) bit-level
///     SHA-256 round bodies.
///   - 11: `is_first_inv_row_binary` — anchor-row selector binarity
///     (LogUp linkage soundness).
///   - 12: `is_first_block_binary` — block-0 indicator binarity.
///   - 13: `is_first_block_pinned_at_anchor`.
///   - 14: `inv_input_byte_binding` — algebraic input binding (rounds
///     0..15 of block 0).
///   - 15: `block_0_after_word_pin` — pins `BLOCK_0_AFTER_WORD[v]` to
///     the AFTER reconstruction at row 63 of block 0.
///   - 16: `output_carry_bit_binary` — binarity of the 16
///     OUTPUT_CARRY bits.
///   - 15: `state_in_pin_at_anchor` — pins `STATE_IN_WORD[v] =
///     INITIAL_HASH[v]` at row 0.
///   - 16: `carry_bit_binary` — binarity of CHAIN_CARRY + BINDING_CARRY
///     (16 bit columns).
///   - 17: `output_byte_binding` — at the LAST round of the LAST
///     block of an aggregator-populated trace, pins the 32 INV_OUTPUT
///     bytes to the digest reconstruction `STATE_IN_WORD[v] +
///     AFTER_word[v]` mod 2^32.
///   - 18: `aggregator_active_binary` — `AGGREGATOR_ACTIVE` is 0 or 1.
///   - 19: `is_last_block_binary` — `IS_LAST_BLOCK` is 0 or 1.
///   - 20: `chain_carry_localized` — `CHAIN_CARRY_BIT[v]` can be non-zero
///     **only** at a block-end (`SEL_ROUND[NUM_ROUNDS-1] = 1`) of a
///     non-last block (`1 − IS_LAST_BLOCK`) in an aggregator-active
///     trace. Forces witness carries to their canonical positions; a
///     malicious prover that stashes a stray `1` in a CHAIN_CARRY_BIT
///     on the wrong row is now algebraically rejected. β-RLC over v.
///   - 21: `binding_carry_localized` — `BINDING_CARRY_BIT[v]` can be
///     non-zero **only** at the last block's last round
///     (`AGGREGATOR_ACTIVE · IS_LAST_BLOCK · SEL_ROUND[NUM_ROUNDS-1]`).
///     β-RLC over v. Mirror of #20 for the digest-reconstruction
///     carry.
///   - 22: `sel_exactly_one_when_aggregator_active` — sharpens
///     `sel_sum_01`: when `AGGREGATOR_ACTIVE = 1`, exactly one round
///     selector must fire (instead of the looser sum ∈ {0,1}). Pinned
///     to aggregator-active rows so legacy traces still pass.
///   - 23: `w_recurrence` — Task #179, generalized to all t in #193.
///     Algebraically pins the message-schedule recurrence
///     `W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16] (mod 2^32)`
///     for **every t ∈ 16..64**, on every block, via the four
///     word-valued aggregator columns
///     (`COL_W_RECURRENCE_W0_WORD`, `…_W9_WORD`, `…_SIGMA0_W1_WORD`,
///     `…_SIGMA1_W14_WORD` — names retained from the row-16 origin)
///     and the 2-bit `W_RECURRENCE_CARRY`. The columns are reused
///     per-t to hold the row-t-relative addends. The constraint is
///     gated by `Σ_{t∈16..64} SEL_ROUND[t]`, so by `sel_sum_01` at
///     most one t-row fires per row. Body shape:
///
///         (Σ_{t∈16..64} SEL_ROUND[t]) ·
///             (σ1_addend + W9_addend + σ0_addend + W0_addend
///              − W_word − carry · 2^32) = 0
///
///     where `W_word` is reconstructed LSB-first from the 32 W bit
///     columns and `carry` from the 2-bit `W_RECURRENCE_CARRY` column.
///     On honest witnesses the integer sum of the four addends equals
///     `W_word + carry·2^32`, so the body vanishes.
///
///     **Soundness scope** (#193): pins the *value* of the recurrence
///     sum to W[t] on every recurrence row. The four addends remain
///     unbound to actual W on rows t-16/t-7 and to σ-helper definitions
///     on rows t-15/t-2 — those bindings need multi-shift cross-row
///     constraints which exceed the current `NUM_SHIFTED = 6`
///     machinery. The 48-row aggregator-self-consistency LogUp gadget
///     that closes this gap is the deferred follow-up.
pub const NUM_ROW_CONSTRAINTS: usize = sha256_air::NUM_CONSTRAINT_CATEGORIES + 13; // 25
//
// (sha256_air::NUM_CONSTRAINT_CATEGORIES grew 10→12 when the message-
// schedule σ0(W)/σ1(W) definitional constraints landed — see task #170.
// The 12 extras (per-invocation byte aggregation + multi-block output
// binding + per-row σ-helper auxiliaries) remain unchanged in count.)

/// Number of consolidated cross-row (shifted) constraints.
///
///   0. `after → before` binding (existing): `after(X)[i] = before(ω·X)[i]`
///      for each of 256 working-variable bits, β-RLC'd. Excluded at every
///      block boundary + domain wrap (boundary set: rows 63, 127, …,
///      num_rows−1, domain_size−1).
///   1. **Aggregator invariance** (added by per-invocation byte aggregation):
///      `INV_INPUT_BYTE_b(ω·X) − INV_INPUT_BYTE_b(X) = 0` and
///      `INV_OUTPUT_BYTE_b(ω·X) − INV_OUTPUT_BYTE_b(X) = 0`, β-RLC'd over
///      96 bodies. Excluded only at the very last real-row transition
///      (row num_rows−1) + domain wrap — block boundaries are NOT
///      excluded, since the aggregator must remain constant across
///      block transitions within one invocation.
///   2. **IS_FIRST_BLOCK invariance within block**:
///      `(1 − SEL_ROUND[NUM_ROUNDS-1](X)) · (IS_FIRST_BLOCK(ω·X) −
///      IS_FIRST_BLOCK(X)) = 0`. Forces IS_FIRST_BLOCK to be constant
///      across all 63 internal-block transitions (where
///      `SEL_ROUND[NUM_ROUNDS-1] = 0`); allows IS_FIRST_BLOCK to
///      change only at the block-end transition. Combined with the
///      row-local pin `IS_FIRST_INV_ROW · (1 − IS_FIRST_BLOCK) = 0`,
///      forces IS_FIRST_BLOCK = 1 on all 64 rounds of block 0.
///      Excluded only at domain wrap (row domain_size − 1).
///   3. **STATE_IN_WORD within-block invariance**: same shape as
///      body 2 but for the 8 STATE_IN_WORD columns (β-RLC). Forces
///      STATE_IN_WORD constant within a block.
///   4. **STATE_IN_WORD cross-block chaining**:
///      `AGGREGATOR_ACTIVE(X) · SEL_ROUND[NUM_ROUNDS-1](X) ·
///      (STATE_IN_WORD(ω·X) + CHAIN_CARRY_BIT(X) · 2^32 −
///      STATE_IN_WORD(X) − AFTER_word(X)) = 0` for each v (β-RLC).
///      At each block-end transition (where SEL_ROUND[NUM_ROUNDS-1] =
///      1), forces STATE_IN_next = STATE_IN_curr + AFTER_word_curr
///      mod 2^32. Excluded only at last-real-row transition + domain
///      wrap; gated by AGGREGATOR_ACTIVE so legacy traces are
///      unaffected.
///   5. **IS_LAST_BLOCK within-block invariance**: same shape as body
///      2 but for the IS_LAST_BLOCK column. Forces IS_LAST_BLOCK
///      constant within a block.
pub const NUM_SHIFTED: usize = 6;

// ──── Constraint system ────────────────────────────────────────────────

/// [`VmConstraintSystem`] implementation for the SHA-256 bit-level AIR.
pub struct Sha256ConstraintSystem {
    /// Number of real trace rows (before padding). Should be a multiple of
    /// 64 (one row per SHA-256 round); a multi-block hash produces
    /// `64 * num_blocks` rows. The padded domain size is determined by
    /// [`TracePolynomials`] to the next power of two ≥ `num_rows`.
    pub num_rows: usize,
    /// The domain generator ω for the trace's padded domain. When `Some`,
    /// the verifier-side `evaluate_shifted_at_point` excludes every block
    /// boundary `(X − ω^{63}), (X − ω^{127}), …` in addition to the
    /// domain wrap `(X − ω^{n−1})`. When `None`, only the wrap-around
    /// factor is excluded — sound only when `num_rows == domain_size`.
    pub omega: Option<Scalar>,
    /// The padded domain size (power of two ≥ `num_rows`). Used together
    /// with `omega` to build the boundary-row exclusion product.
    pub domain_size: Option<u64>,
}

impl Sha256ConstraintSystem {
    /// Construct a SHA-256 constraint system for a trace of `num_rows`
    /// real rows (= `64 * num_blocks`).
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    /// Attach the domain generator `omega` and `domain_size` for the
    /// scheme/trace this constraint system is paired with. The verifier
    /// needs these to replicate the prover's boundary exclusion product
    /// `Π_{r ∈ boundary_rows} (z − ω^r)`.
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

// ──── Trace construction helpers ───────────────────────────────────────

/// Build a [`TracePolynomials`] wrapping the SHA-256 bit-level trace
/// produced by populating `rounds` in order. Each round populates exactly
/// one row; the trace pads up to the next power of two automatically
/// with zeros.
pub fn build_trace_polynomials_from_rounds(
    rounds: &[RoundTrace],
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = rounds.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let mut columns = alloc_trace(padded, curve);
    for (row, rt) in rounds.iter().enumerate() {
        populate_round(&mut columns, row, rt, curve);
    }
    // Task #193: populate the W[t] recurrence aggregator + carry on every
    // row t ∈ 16..NUM_ROUNDS so the generalized `w_recurrence` row-local
    // constraint vanishes. The per-row populator doesn't have cross-row
    // visibility, so we do it as a finalization step here.
    populate_w_recurrence_for_round_trace(&mut columns, rounds, curve);
    into_trace_polynomials(columns, num_rows, padded, curve)
}

/// Witness-populate the W[t] recurrence aggregator + carry on every row
/// t ∈ 16..NUM_ROUNDS for a single-block round-trace driven column set.
/// Mirrors the multi-block populator in
/// [`sha256_air::populate_w_recurrence_carry`] but for the row-trace
/// flavor used by [`build_trace_polynomials_from_rounds`]. Task #193.
fn populate_w_recurrence_for_round_trace(
    columns: &mut [Vec<Scalar>],
    rounds: &[RoundTrace],
    curve: CurveType,
) {
    if rounds.len() <= 16 || columns[0].len() <= 16 {
        return;
    }
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    for t in 16..NUM_ROUNDS.min(rounds.len()) {
        if t >= columns[0].len() {
            break;
        }
        let w_tm16 = rounds[t - 16].w as u64;
        let w_tm15 = rounds[t - 15].w;
        let w_tm7 = rounds[t - 7].w as u64;
        let w_tm2 = rounds[t - 2].w;
        let s0 = (w_tm15.rotate_right(7)
            ^ w_tm15.rotate_right(18)
            ^ (w_tm15 >> 3)) as u64;
        let s1 = (w_tm2.rotate_right(17)
            ^ w_tm2.rotate_right(19)
            ^ (w_tm2 >> 10)) as u64;
        let full_sum = s1 + w_tm7 + s0 + w_tm16;
        let carry = full_sum >> 32;
        for bit in 0..sha256_air::W_RECURRENCE_CARRY_BITS {
            let v = (carry >> bit) & 1;
            columns[sha256_air::w_recurrence_carry_bit(bit)][t] =
                if v == 1 { one.clone() } else { zero.clone() };
        }
        columns[sha256_air::w_recurrence_w0_word()][t]        = Scalar::from_u64(w_tm16, curve);
        columns[sha256_air::w_recurrence_w9_word()][t]        = Scalar::from_u64(w_tm7, curve);
        columns[sha256_air::w_recurrence_sigma0_w1_word()][t] = Scalar::from_u64(s0, curve);
        columns[sha256_air::w_recurrence_sigma1_w14_word()][t] = Scalar::from_u64(s1, curve);
    }
}

/// Variant for full hash witnesses (multi-block inputs).
pub fn build_trace_polynomials_from_hash(
    hash_trace: &HashTrace,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = hash_trace.blocks.len() * NUM_ROUNDS;
    let base_columns = populate_trace_from_hash(hash_trace, curve);
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let mut columns = base_columns;
    for col in columns.iter_mut() {
        if col.len() < padded {
            col.resize(padded, Scalar::zero(curve));
        }
    }
    into_trace_polynomials(columns, num_rows, padded, curve)
}

fn into_trace_polynomials(
    columns: Vec<Vec<Scalar>>,
    num_rows: usize,
    padded: usize,
    curve: CurveType,
) -> TracePolynomials {
    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial {
            evaluations: evals,
            degree: num_rows,
        })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ──── Helpers for scalar-point evaluation of constraint bodies ─────────

/// Compute `xor(a, b) = a + b − 2·a·b` on scalars.
#[inline]
fn xor_scalar(a: &Scalar, b: &Scalar, two: &Scalar) -> Scalar {
    let ab = a.mul(b);
    a.add(b).sub(&two.mul(&ab))
}

/// Fast-exponentiate a scalar by a `u64` exponent.
fn scalar_pow(base: &Scalar, exp: u64) -> Scalar {
    let mut result = Scalar::one(base.curve_type());
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&b);
        }
        b = b.mul(&b);
        e >>= 1;
    }
    result
}

/// Reconstruct a 32-bit (or `n`-bit) word value from the bit columns at
/// `offset..offset+n` for a given row of scalars.
fn word_value_at(cols: &[Scalar], offset: usize, n: usize, curve: CurveType) -> Scalar {
    let two = Scalar::from_u64(2, curve);
    let mut pow = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    for bit in 0..n {
        let c = &cols[offset + bit];
        acc = acc.add(&pow.mul(c));
        pow = pow.mul(&two);
    }
    acc
}

/// Reconstruct a 32-bit (or `n`-bit) word as a polynomial: Σ bit_poly_i · 2^i.
fn word_value_poly(cols: &[Vec<Scalar>], offset: usize, n: usize, curve: CurveType) -> Vec<Scalar> {
    let two = Scalar::from_u64(2, curve);
    let mut pow = Scalar::one(curve);
    let mut acc = vec![Scalar::zero(curve)];
    for bit in 0..n {
        let scaled = poly_scalar_mul(&cols[offset + bit], &pow);
        acc = poly_add(&acc, &scaled, curve);
        pow = pow.mul(&two);
    }
    acc
}

// ──── Scalar-point evaluation of each category body ────────────────────

/// 1. bit_validity: every data column satisfies b·(b−1) = 0.
fn eval_bit_validity_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for col_idx in 0..NUM_DATA_COLUMNS {
        let v = &cols[col_idx];
        let body = v.mul(&v.sub(&one));
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    acc
}

/// 2. big_sigma0_a definition (two-stage XOR over rotations 2/13/22).
fn eval_sigma0_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let a_rot2 = &cols[before(VAR_A, (bit + 2) % BITS_PER_WORD)];
        let a_rot13 = &cols[before(VAR_A, (bit + 13) % BITS_PER_WORD)];
        let a_rot22 = &cols[before(VAR_A, (bit + 22) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_s0_bit(bit)];
        let sig0 = &cols[big_sigma0_a_bit(bit)];

        // Stage 1: xor01 = a_rot2 XOR a_rot13.
        let x1 = xor_scalar(a_rot2, a_rot13, &two);
        let body1 = xor01.sub(&x1);
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        // Stage 2: sig0 = xor01 XOR a_rot22.
        let x2 = xor_scalar(xor01, a_rot22, &two);
        let body2 = sig0.sub(&x2);
        acc = acc.add(&bp.mul(&body2));
        bp = bp.mul(beta);
    }
    acc
}

/// 3. big_sigma1_e definition (two-stage XOR over rotations 6/11/25).
fn eval_sigma1_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let e_rot6 = &cols[before(VAR_E, (bit + 6) % BITS_PER_WORD)];
        let e_rot11 = &cols[before(VAR_E, (bit + 11) % BITS_PER_WORD)];
        let e_rot25 = &cols[before(VAR_E, (bit + 25) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_s1_bit(bit)];
        let sig1 = &cols[big_sigma1_e_bit(bit)];

        let x1 = xor_scalar(e_rot6, e_rot11, &two);
        let body1 = xor01.sub(&x1);
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        let x2 = xor_scalar(xor01, e_rot25, &two);
        let body2 = sig1.sub(&x2);
        acc = acc.add(&bp.mul(&body2));
        bp = bp.mul(beta);
    }
    acc
}

/// 4. ch_definition: ef = e·f, not_e_g = (1−e)·g, ch_efg = ef ⊕ not_e_g.
fn eval_ch_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let e = &cols[before(VAR_E, bit)];
        let f = &cols[before(VAR_F, bit)];
        let g = &cols[before(VAR_G, bit)];
        let ef = &cols[ef_bit(bit)];
        let ng = &cols[not_e_g_bit(bit)];
        let ch = &cols[ch_efg_bit(bit)];

        let body1 = ef.sub(&e.mul(f));
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        let not_e = one.sub(e);
        let body2 = ng.sub(&not_e.mul(g));
        acc = acc.add(&bp.mul(&body2));
        bp = bp.mul(beta);

        let x = xor_scalar(ef, ng, &two);
        let body3 = ch.sub(&x);
        acc = acc.add(&bp.mul(&body3));
        bp = bp.mul(beta);
    }
    acc
}

/// 5. maj_definition: ab,ac,bc products + double XOR.
fn eval_maj_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let a = &cols[before(VAR_A, bit)];
        let b = &cols[before(VAR_B, bit)];
        let c = &cols[before(VAR_C, bit)];
        let ab = &cols[ab_bit(bit)];
        let ac = &cols[ac_bit(bit)];
        let bc = &cols[bc_bit(bit)];
        let xab_ac = &cols[xor_ab_ac_bit(bit)];
        let mj = &cols[maj_abc_bit(bit)];

        let body1 = ab.sub(&a.mul(b));
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        let body2 = ac.sub(&a.mul(c));
        acc = acc.add(&bp.mul(&body2));
        bp = bp.mul(beta);

        let body3 = bc.sub(&b.mul(c));
        acc = acc.add(&bp.mul(&body3));
        bp = bp.mul(beta);

        let x = xor_scalar(ab, ac, &two);
        let body4 = xab_ac.sub(&x);
        acc = acc.add(&bp.mul(&body4));
        bp = bp.mul(beta);

        let x = xor_scalar(xab_ac, bc, &two);
        let body5 = mj.sub(&x);
        acc = acc.add(&bp.mul(&body5));
        bp = bp.mul(beta);
    }
    acc
}

/// 6. round_additions: two value-level relations.
///   (A) h + Σ₁(e) + Ch + K + W = T1_value + t1_carry · 2^32
///   (B) Σ₀(a) + Maj            = T2_value + t2_carry · 2^32
/// where T1 = (new_e + e_new_carry·2^32) − d
///       T2 = (new_a + a_new_carry·2^32) − T1.
fn eval_round_adds_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let two = Scalar::from_u64(2, curve);

    let h_val = word_value_at(cols, COL_BEFORE_OFFSET + VAR_H * BITS_PER_WORD, BITS_PER_WORD, curve);
    let d_val = word_value_at(cols, COL_BEFORE_OFFSET + VAR_D * BITS_PER_WORD, BITS_PER_WORD, curve);
    let s1_val = word_value_at(cols, COL_BIG_SIGMA1_E_OFFSET, BITS_PER_WORD, curve);
    let ch_val = word_value_at(cols, COL_CH_EFG_OFFSET, BITS_PER_WORD, curve);
    let k_val = word_value_at(cols, COL_K_OFFSET, BITS_PER_WORD, curve);
    let w_val = word_value_at(cols, COL_W_OFFSET, BITS_PER_WORD, curve);
    let new_a_val = word_value_at(cols, COL_AFTER_OFFSET + VAR_A * BITS_PER_WORD, BITS_PER_WORD, curve);
    let new_e_val = word_value_at(cols, COL_AFTER_OFFSET + VAR_E * BITS_PER_WORD, BITS_PER_WORD, curve);
    let s0_val = word_value_at(cols, COL_BIG_SIGMA0_A_OFFSET, BITS_PER_WORD, curve);
    let maj_val = word_value_at(cols, COL_MAJ_ABC_OFFSET, BITS_PER_WORD, curve);

    // T1 carry = Σ bit·2^i across T1_CARRY_BITS bits.
    let mut t1c = Scalar::zero(curve);
    let mut p = one.clone();
    for bit in 0..T1_CARRY_BITS {
        t1c = t1c.add(&p.mul(&cols[COL_T1_CARRY_OFFSET + bit]));
        p = p.mul(&two);
    }
    let t2c = &cols[COL_T2_CARRY_OFFSET];
    let anc = &cols[COL_A_NEW_CARRY_OFFSET];
    let enc = &cols[COL_E_NEW_CARRY_OFFSET];

    // T1 = new_e + e_new_carry·2^32 − d.
    let t1_val = new_e_val.add(&enc.mul(&two_pow_32)).sub(&d_val);

    // (A) lhs = h + Σ₁(e) + Ch + K + W
    //     rhs = T1_value + t1_carry · 2^32
    let lhs_a = h_val.add(&s1_val).add(&ch_val).add(&k_val).add(&w_val);
    let rhs_a = t1_val.add(&t1c.mul(&two_pow_32));
    let body_a = lhs_a.sub(&rhs_a);

    // T2 = new_a + a_new_carry·2^32 − T1.
    let t1_val_b = new_e_val.add(&enc.mul(&two_pow_32)).sub(&d_val);
    let t2_val = new_a_val.add(&anc.mul(&two_pow_32)).sub(&t1_val_b);

    // (B) lhs = Σ₀(a) + Maj
    //     rhs = T2_value + t2_carry · 2^32
    let lhs_b = s0_val.add(&maj_val);
    let rhs_b = t2_val.add(&t2c.mul(&two_pow_32));
    let body_b = lhs_b.sub(&rhs_b);

    let mut acc = body_a;
    acc = acc.add(&beta.mul(&body_b));
    acc
}

/// 7. passthrough: b'=a, c'=b, d'=c, f'=e, g'=f, h'=g — bit-by-bit.
fn eval_passthrough_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    let pairs: [(usize, usize); 6] = [
        (VAR_B, VAR_A),
        (VAR_C, VAR_B),
        (VAR_D, VAR_C),
        (VAR_F, VAR_E),
        (VAR_G, VAR_F),
        (VAR_H, VAR_G),
    ];
    for &(dst, src) in &pairs {
        for bit in 0..BITS_PER_WORD {
            let s = &cols[before(src, bit)];
            let d = &cols[after(dst, bit)];
            let body = d.sub(s);
            acc = acc.add(&bp.mul(&body));
            bp = bp.mul(beta);
        }
    }
    acc
}

/// 8. k_binding: Σ_k sel_round_k · (k_bit[i] − RC_k_bit[i]) = 0 for every i.
fn eval_k_binding_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let kb = &cols[k_bit(bit)];
        for k in 0..NUM_ROUNDS {
            let rc_bit = ((ROUND_CONSTANTS[k] >> bit) & 1) as u64;
            let rc_scalar = if rc_bit == 1 { one.clone() } else { zero.clone() };
            let sel = &cols[sel_round(k)];
            let diff = kb.sub(&rc_scalar);
            let gated = sel.mul(&diff);
            acc = acc.add(&bp.mul(&gated));
            bp = bp.mul(beta);
        }
    }
    acc
}

/// 9. sel_binary: each round selector satisfies s·(s−1) = 0.
fn eval_sel_binary_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for k in 0..NUM_SEL_ROUND {
        let s = &cols[sel_round(k)];
        let body = s.mul(&s.sub(&one));
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    acc
}

/// Small-σ0(W) definition: two-stage XOR over rotations 7/18 and shift 3.
///   xor01_ss0[i]   = W[(i+7)%32] XOR W[(i+18)%32]
///   sigma0_w[i]    = xor01_ss0[i] XOR (W[i+3] if i<29 else 0)
fn eval_small_sigma0_w_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let w_rot7 = &cols[sha256_air::w_bit((bit + 7) % BITS_PER_WORD)];
        let w_rot18 = &cols[sha256_air::w_bit((bit + 18) % BITS_PER_WORD)];
        let xor01 = &cols[sha256_air::xor01_ss0_bit(bit)];
        let ss0 = &cols[sha256_air::small_sigma0_w_bit(bit)];

        let x1 = xor_scalar(w_rot7, w_rot18, &two);
        let body1 = xor01.sub(&x1);
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        if bit + 3 < BITS_PER_WORD {
            let w_shr3 = &cols[sha256_air::w_bit(bit + 3)];
            let x2 = xor_scalar(xor01, w_shr3, &two);
            let body2 = ss0.sub(&x2);
            acc = acc.add(&bp.mul(&body2));
        } else {
            // SHR_3 bit is 0; ss0[i] = xor01[i].
            let body2 = ss0.sub(xor01);
            acc = acc.add(&bp.mul(&body2));
        }
        bp = bp.mul(beta);
    }
    acc
}

/// Small-σ1(W) definition: two-stage XOR over rotations 17/19 and shift 10.
fn eval_small_sigma1_w_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let w_rot17 = &cols[sha256_air::w_bit((bit + 17) % BITS_PER_WORD)];
        let w_rot19 = &cols[sha256_air::w_bit((bit + 19) % BITS_PER_WORD)];
        let xor01 = &cols[sha256_air::xor01_ss1_bit(bit)];
        let ss1 = &cols[sha256_air::small_sigma1_w_bit(bit)];

        let x1 = xor_scalar(w_rot17, w_rot19, &two);
        let body1 = xor01.sub(&x1);
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        if bit + 10 < BITS_PER_WORD {
            let w_shr10 = &cols[sha256_air::w_bit(bit + 10)];
            let x2 = xor_scalar(xor01, w_shr10, &two);
            let body2 = ss1.sub(&x2);
            acc = acc.add(&bp.mul(&body2));
        } else {
            let body2 = ss1.sub(xor01);
            acc = acc.add(&bp.mul(&body2));
        }
        bp = bp.mul(beta);
    }
    acc
}

/// 10. sel_sum_01: (Σ sel_round_k) · (Σ sel_round_k − 1) = 0.
fn eval_sel_sum_01_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let mut sum = Scalar::zero(curve);
    for k in 0..NUM_SEL_ROUND {
        sum = sum.add(&cols[sel_round(k)]);
    }
    sum.mul(&sum.sub(&one))
}

/// 11. is_first_inv_row_binary: `IS_FIRST_INV_ROW · (IS_FIRST_INV_ROW − 1) = 0`.
/// Required for cross-AIR LogUp linkage soundness: the linkage gates the
/// B-side tuple by IS_FIRST_INV_ROW, and a non-binary selector value
/// would let a malicious prover scale the running sum increment by
/// arbitrary field elements, breaking the multiset equivalence
/// guarantee.
fn eval_is_first_inv_row_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[sha256_air::COL_IS_FIRST_INV_ROW];
    v.mul(&v.sub(&one))
}

/// 12. is_first_block_binary: `IS_FIRST_BLOCK · (IS_FIRST_BLOCK − 1) = 0`.
fn eval_is_first_block_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[sha256_air::COL_IS_FIRST_BLOCK];
    v.mul(&v.sub(&one))
}

/// 13. is_first_block_pinned_at_anchor:
///     `IS_FIRST_INV_ROW · (1 − IS_FIRST_BLOCK) = 0`.
/// At row 0 (the only row where `IS_FIRST_INV_ROW = 1`),
/// `IS_FIRST_BLOCK` must equal 1. Combined with the cross-row
/// invariance constraint (within a block; see body 2 of the shifted
/// constraint), this forces IS_FIRST_BLOCK = 1 across all 64 rounds
/// of block 0.
fn eval_is_first_block_pinned_at_anchor_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let anchor = &cols[sha256_air::COL_IS_FIRST_INV_ROW];
    let fb = &cols[sha256_air::COL_IS_FIRST_BLOCK];
    anchor.mul(&one.sub(fb))
}

/// 14. inv_input_byte_binding:
///     `Σ_{r=0..16} SEL_ROUND[r] · IS_FIRST_BLOCK ·
///        (Σ_b INV_INPUT_BYTE[4r+b] · 2^(8(3−b)) − Σ_i W_bit[i] · 2^i) = 0`.
///
/// On rounds 0..15 of block 0 (gated by `IS_FIRST_BLOCK · SEL_ROUND[r]`),
/// pins the 4 input bytes at offsets 4r..4r+4 to the 32-bit
/// big-endian decomposition of W[r] (= `Σ_i W_bit[i] · 2^i` reading
/// W as LSB-first). Combined with the aggregator invariance, all 64
/// input bytes are bound to the actual block-0 input on rows 0..15.
/// 15. state_in_pin_at_anchor (β-RLC over v ∈ 0..NUM_OUTPUT_WORDS):
///     `IS_FIRST_INV_ROW · (STATE_IN_WORD[v] − INITIAL_HASH[v]) = 0`.
/// Pins the state_in chaining column to INITIAL_HASH at the row-0
/// anchor. Combined with the within-block invariance shifted body
/// (forces STATE_IN constant within a block) and the cross-block
/// chaining shifted body (forces STATE_IN_next = STATE_IN +
/// AFTER_word at block boundaries), this gives the canonical SHA-256
/// chaining-state value at every row.
fn eval_state_in_pin_at_anchor_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let anchor = &cols[sha256_air::COL_IS_FIRST_INV_ROW];
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let state_word = &cols[sha256_air::state_in_word(v)];
        let init = Scalar::from_u64(INITIAL_HASH[v] as u64, curve);
        let body = state_word.sub(&init);
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    anchor.mul(&acc)
}

/// 16. carry_bit_binary (β-RLC over 16 carry bit columns):
///     `Σ b · (b − 1) = 0` for each of CHAIN_CARRY_BIT[v] and
///     BINDING_CARRY_BIT[v], v ∈ 0..NUM_OUTPUT_WORDS.
fn eval_carry_bit_binary_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let cc = &cols[sha256_air::chain_carry_bit(v)];
        let bc = cc.mul(&cc.sub(&one));
        acc = acc.add(&bp.mul(&bc));
        bp = bp.mul(beta);
        let bb = &cols[sha256_air::binding_carry_bit(v)];
        let bb_body = bb.mul(&bb.sub(&one));
        acc = acc.add(&bp.mul(&bb_body));
        bp = bp.mul(beta);
    }
    acc
}

/// 16b. is_last_block_binary: `IS_LAST_BLOCK · (IS_LAST_BLOCK − 1) = 0`.
fn eval_is_last_block_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[sha256_air::COL_IS_LAST_BLOCK];
    v.mul(&v.sub(&one))
}

/// 17. output_byte_binding (β-RLC over v ∈ 0..NUM_OUTPUT_WORDS):
///     `AGGREGATOR_ACTIVE · IS_LAST_BLOCK · SEL_ROUND[NUM_ROUNDS-1] ·
///        (Σ_{k=0..4} INV_OUTPUT_BYTE[4v+k] · 2^(8(3-k))
///         + binding_carry_bit[v] · 2^32
///         − STATE_IN_WORD[v]
///         − Σ_i AFTER[v][i] · 2^i) = 0`.
///
/// At the LAST round of the LAST block of an aggregator-populated
/// trace — the only row where `AGGREGATOR_ACTIVE · IS_LAST_BLOCK ·
/// SEL_ROUND[NUM_ROUNDS-1] = 1` — pins the digest big-endian
/// decomposition to the SHA-256 state-out reconstruction:
///     digest[v] = state_in[last block][v] + AFTER_word[v] (mod 2^32)
/// The 1-bit carry handles the `mod 2^32`. `STATE_IN_WORD` is itself
/// algebraically chained from `INITIAL_HASH` through the cross-block
/// chaining shifted constraint, so the binding pins the digest to the
/// canonical SHA-256 of the input regardless of block count.
fn eval_output_byte_binding_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let two_to_32 = Scalar::from_u64(1u64 << 32, curve);
    let pow_be: [Scalar; 4] = [
        Scalar::from_u64(1u64 << 24, curve),
        Scalar::from_u64(1u64 << 16, curve),
        Scalar::from_u64(1u64 << 8, curve),
        Scalar::from_u64(1, curve),
    ];

    let last_block = &cols[sha256_air::COL_IS_LAST_BLOCK];
    let last_sel = &cols[sel_round(NUM_ROUNDS - 1)];
    let aggregator = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    // Gate: AGGREGATOR_ACTIVE · IS_LAST_BLOCK · SEL_ROUND[NUM_ROUNDS-1].
    // Fires at the LAST round of the LAST block of an aggregator-
    // populated trace.
    let gate = aggregator.mul(last_block).mul(last_sel);

    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        // BE byte decomposition of digest_word[v]
        let mut digest_word = Scalar::zero(curve);
        for k in 0..4 {
            let byte = &cols[sha256_air::inv_output_byte(4 * v + k)];
            digest_word = digest_word.add(&byte.mul(&pow_be[k]));
        }
        // 1-bit binding carry
        let bc = &cols[sha256_air::binding_carry_bit(v)];
        // RHS: STATE_IN_WORD[v] + Σ_i AFTER[v][i] · 2^i
        let state_in = &cols[sha256_air::state_in_word(v)];
        let mut after_now = Scalar::zero(curve);
        let mut pow = Scalar::one(curve);
        for bit in 0..BITS_PER_WORD {
            let bit_col = &cols[after(v, bit)];
            after_now = after_now.add(&pow.mul(bit_col));
            pow = pow.mul(&two);
        }
        let rhs = state_in.add(&after_now);
        let lhs = digest_word.add(&bc.mul(&two_to_32));
        let body = lhs.sub(&rhs);
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    gate.mul(&acc)
}

/// 18. aggregator_active_binary: `AGGREGATOR_ACTIVE · (AGGREGATOR_ACTIVE − 1) = 0`.
fn eval_aggregator_active_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    v.mul(&v.sub(&one))
}

/// 20. chain_carry_localized: `CHAIN_CARRY_BIT[v]` may only be non-zero
/// at a block-end row of a non-last block on aggregator-active traces.
///   body = (1 − AGGREGATOR_ACTIVE · SEL_ROUND[NUM_ROUNDS-1] · (1 − IS_LAST_BLOCK))
///          · CHAIN_CARRY_BIT[v]
/// β-RLC over v ∈ 0..NUM_OUTPUT_WORDS.
fn eval_chain_carry_localized_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let aggregator = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    let last_sel = &cols[sel_round(NUM_ROUNDS - 1)];
    let last_block = &cols[sha256_air::COL_IS_LAST_BLOCK];
    // active_gate = AGGREGATOR_ACTIVE · SEL_ROUND[NUM_ROUNDS-1] · (1 − IS_LAST_BLOCK)
    let active_gate = aggregator.mul(last_sel).mul(&one.sub(last_block));
    let neg_gate = one.sub(&active_gate);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let cc = &cols[sha256_air::chain_carry_bit(v)];
        let body = neg_gate.mul(cc);
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    acc
}

/// 21. binding_carry_localized: `BINDING_CARRY_BIT[v]` may only be
/// non-zero at the last block's last round on aggregator-active traces.
///   body = (1 − AGGREGATOR_ACTIVE · IS_LAST_BLOCK · SEL_ROUND[NUM_ROUNDS-1])
///          · BINDING_CARRY_BIT[v]
/// β-RLC over v.
fn eval_binding_carry_localized_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let aggregator = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    let last_sel = &cols[sel_round(NUM_ROUNDS - 1)];
    let last_block = &cols[sha256_air::COL_IS_LAST_BLOCK];
    let active_gate = aggregator.mul(last_block).mul(last_sel);
    let neg_gate = one.sub(&active_gate);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let bc = &cols[sha256_air::binding_carry_bit(v)];
        let body = neg_gate.mul(bc);
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    acc
}

/// 23. w_recurrence (Task #179, generalized in #193): pin the
/// message-schedule recurrence value at EVERY row t ∈ 16..64 of each
/// block. Body shape:
///   (Σ_{t∈16..64} SEL_ROUND[t]) ·
///       (σ1_addend + W9_addend + σ0_addend + W0_addend
///        − W_word − carry·2^32)
///
/// The aggregator columns hold the row-t-relative addends
/// `(W[t-16], W[t-7], σ0(W[t-15]), σ1(W[t-2]))`, each in [0, 2^32). Their
/// integer sum lies in [0, 2^34) and equals `W[t] + carry·2^32` where
/// `carry ∈ [0, 4)` is the committed 2-bit carry. By `sel_sum_01` at most
/// one SEL_ROUND is active per row, so the gate cleanly picks the
/// single active t and the body vanishes on non-recurrence rows.
///
/// Bug fix (Task #193): the original anchor-only body was
/// `W_word − (σ1 + W9 + σ0 + W0 + carry·2^32)`, which is `−2·carry·2^32`
/// on honest witnesses (only vanishes when carry = 0). The row-16
/// anchor test inputs happened to produce carry = 0, hiding the bug.
/// Extending to t ∈ 16..64 surfaces non-zero carries and forces the
/// correction.
fn eval_w_recurrence_at_anchor_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let two = Scalar::from_u64(2, curve);
    let two_to_32 = Scalar::from_u64(1u64 << 32, curve);
    // Recurrence-rows selector sum: Σ_{t=16..NUM_ROUNDS} SEL_ROUND[t].
    let mut sel_recurrence = Scalar::zero(curve);
    for t in 16..NUM_ROUNDS {
        sel_recurrence = sel_recurrence.add(&cols[sel_round(t)]);
    }
    // W_word = Σ_i W_bit[i] · 2^i.
    let mut w_word = Scalar::zero(curve);
    let mut pow = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        w_word = w_word.add(&pow.mul(&cols[sha256_air::w_bit(bit)]));
        pow = pow.mul(&two);
    }
    let mut carry = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..sha256_air::W_RECURRENCE_CARRY_BITS {
        carry = carry.add(&bp.mul(&cols[sha256_air::w_recurrence_carry_bit(bit)]));
        bp = bp.mul(&two);
    }
    let s1 = &cols[sha256_air::w_recurrence_sigma1_w14_word()];
    let w9 = &cols[sha256_air::w_recurrence_w9_word()];
    let s0 = &cols[sha256_air::w_recurrence_sigma0_w1_word()];
    let w0 = &cols[sha256_air::w_recurrence_w0_word()];
    // σ1 + W9 + σ0 + W0 − W_word − carry·2^32 (the addends' integer sum
    // equals W_word + carry·2^32 on honest witnesses).
    let addend_sum = s1.add(w9).add(s0).add(w0);
    let rhs = w_word.add(&carry.mul(&two_to_32));
    sel_recurrence.mul(&addend_sum.sub(&rhs))
}

/// 22. sel_exactly_one_when_aggregator_active:
///   `AGGREGATOR_ACTIVE · (Σ_k sel_round_k − 1) = 0`.
/// Tightens `sel_sum_01` to enforce exactly one active selector on every
/// aggregator-active row (the looser `sum · (sum − 1) = 0` also admits
/// sum = 0). Legacy traces (AGGREGATOR_ACTIVE = 0) are unaffected.
fn eval_sel_exactly_one_when_aggregator_active_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let aggregator = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    let mut sum = Scalar::zero(curve);
    for k in 0..NUM_SEL_ROUND {
        sum = sum.add(&cols[sel_round(k)]);
    }
    aggregator.mul(&sum.sub(&one))
}

fn eval_inv_input_byte_binding_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let fb = &cols[sha256_air::COL_IS_FIRST_BLOCK];
    // W word = Σ_i W_bit[i] · 2^i (LSB-first). Same value across all 16 r.
    let w_word = word_value_at(cols, sha256_air::COL_W_OFFSET, BITS_PER_WORD, curve);
    // 2^24, 2^16, 2^8, 2^0 (big-endian within a 4-byte word).
    let pow_be: [Scalar; 4] = [
        Scalar::from_u64(1u64 << 24, curve),
        Scalar::from_u64(1u64 << 16, curve),
        Scalar::from_u64(1u64 << 8, curve),
        Scalar::from_u64(1, curve),
    ];
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for r in 0..16 {
        let sel = &cols[sel_round(r)];
        // BLOCK_INPUT_WORD_r = Σ_b BYTE[4r+b] · 2^(8(3−b)).
        let mut block_word = Scalar::zero(curve);
        for b in 0..4 {
            let byte = &cols[sha256_air::inv_input_byte(4 * r + b)];
            block_word = block_word.add(&byte.mul(&pow_be[b]));
        }
        let body = sel.mul(&fb.mul(&block_word.sub(&w_word)));
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    acc
}

// ──── Polynomial-form builders for each category body ──────────────────

fn build_bit_validity_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for col_idx in 0..NUM_DATA_COLUMNS {
        let c = &cols[col_idx];
        let c_minus_1 = poly_sub(c, &one_poly, curve);
        let body = poly_mul(c, &c_minus_1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_sigma0_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let a_rot2 = &cols[before(VAR_A, (bit + 2) % BITS_PER_WORD)];
        let a_rot13 = &cols[before(VAR_A, (bit + 13) % BITS_PER_WORD)];
        let a_rot22 = &cols[before(VAR_A, (bit + 22) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_s0_bit(bit)];
        let sig0 = &cols[big_sigma0_a_bit(bit)];

        // Stage 1: xor01 - (a_rot2 + a_rot13 - 2·a_rot2·a_rot13).
        let ab = poly_mul(a_rot2, a_rot13, curve);
        let two_ab = poly_scalar_mul(&ab, &two);
        let xor1 = poly_sub(&poly_add(a_rot2, a_rot13, curve), &two_ab, curve);
        let body1 = poly_sub(xor01, &xor1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        // Stage 2: sig0 - (xor01 + a_rot22 - 2·xor01·a_rot22).
        let ab2 = poly_mul(xor01, a_rot22, curve);
        let two_ab2 = poly_scalar_mul(&ab2, &two);
        let xor2 = poly_sub(&poly_add(xor01, a_rot22, curve), &two_ab2, curve);
        let body2 = poly_sub(sig0, &xor2, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_sigma1_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let e_rot6 = &cols[before(VAR_E, (bit + 6) % BITS_PER_WORD)];
        let e_rot11 = &cols[before(VAR_E, (bit + 11) % BITS_PER_WORD)];
        let e_rot25 = &cols[before(VAR_E, (bit + 25) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_s1_bit(bit)];
        let sig1 = &cols[big_sigma1_e_bit(bit)];

        let ab = poly_mul(e_rot6, e_rot11, curve);
        let two_ab = poly_scalar_mul(&ab, &two);
        let xor1 = poly_sub(&poly_add(e_rot6, e_rot11, curve), &two_ab, curve);
        let body1 = poly_sub(xor01, &xor1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        let ab2 = poly_mul(xor01, e_rot25, curve);
        let two_ab2 = poly_scalar_mul(&ab2, &two);
        let xor2 = poly_sub(&poly_add(xor01, e_rot25, curve), &two_ab2, curve);
        let body2 = poly_sub(sig1, &xor2, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_ch_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let e = &cols[before(VAR_E, bit)];
        let f = &cols[before(VAR_F, bit)];
        let g = &cols[before(VAR_G, bit)];
        let ef = &cols[ef_bit(bit)];
        let ng = &cols[not_e_g_bit(bit)];
        let ch = &cols[ch_efg_bit(bit)];

        // ef - e·f
        let ef_prod = poly_mul(e, f, curve);
        let body1 = poly_sub(ef, &ef_prod, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        // ng - (1 - e)·g = ng - g + e·g
        let not_e = poly_sub(&one_poly, e, curve);
        let neg_prod = poly_mul(&not_e, g, curve);
        let body2 = poly_sub(ng, &neg_prod, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        bp = bp.mul(beta);

        // ch - (ef + ng - 2·ef·ng)
        let ab = poly_mul(ef, ng, curve);
        let two_ab = poly_scalar_mul(&ab, &two);
        let xor = poly_sub(&poly_add(ef, ng, curve), &two_ab, curve);
        let body3 = poly_sub(ch, &xor, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body3, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_maj_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let a = &cols[before(VAR_A, bit)];
        let b = &cols[before(VAR_B, bit)];
        let c = &cols[before(VAR_C, bit)];
        let ab = &cols[ab_bit(bit)];
        let ac = &cols[ac_bit(bit)];
        let bc = &cols[bc_bit(bit)];
        let xab_ac = &cols[xor_ab_ac_bit(bit)];
        let mj = &cols[maj_abc_bit(bit)];

        let ab_prod = poly_mul(a, b, curve);
        let body1 = poly_sub(ab, &ab_prod, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        let ac_prod = poly_mul(a, c, curve);
        let body2 = poly_sub(ac, &ac_prod, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        bp = bp.mul(beta);

        let bc_prod = poly_mul(b, c, curve);
        let body3 = poly_sub(bc, &bc_prod, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body3, &bp), curve);
        bp = bp.mul(beta);

        let abac = poly_mul(ab, ac, curve);
        let two_abac = poly_scalar_mul(&abac, &two);
        let xor_abac = poly_sub(&poly_add(ab, ac, curve), &two_abac, curve);
        let body4 = poly_sub(xab_ac, &xor_abac, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body4, &bp), curve);
        bp = bp.mul(beta);

        let xabbc = poly_mul(xab_ac, bc, curve);
        let two_xabbc = poly_scalar_mul(&xabbc, &two);
        let xor_final = poly_sub(&poly_add(xab_ac, bc, curve), &two_xabbc, curve);
        let body5 = poly_sub(mj, &xor_final, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body5, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_round_adds_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);
    let one = Scalar::one(curve);
    let two = Scalar::from_u64(2, curve);

    let h_val = word_value_poly(cols, COL_BEFORE_OFFSET + VAR_H * BITS_PER_WORD, BITS_PER_WORD, curve);
    let d_val = word_value_poly(cols, COL_BEFORE_OFFSET + VAR_D * BITS_PER_WORD, BITS_PER_WORD, curve);
    let s1_val = word_value_poly(cols, COL_BIG_SIGMA1_E_OFFSET, BITS_PER_WORD, curve);
    let ch_val = word_value_poly(cols, COL_CH_EFG_OFFSET, BITS_PER_WORD, curve);
    let k_val = word_value_poly(cols, COL_K_OFFSET, BITS_PER_WORD, curve);
    let w_val = word_value_poly(cols, COL_W_OFFSET, BITS_PER_WORD, curve);
    let new_a_val = word_value_poly(cols, COL_AFTER_OFFSET + VAR_A * BITS_PER_WORD, BITS_PER_WORD, curve);
    let new_e_val = word_value_poly(cols, COL_AFTER_OFFSET + VAR_E * BITS_PER_WORD, BITS_PER_WORD, curve);
    let s0_val = word_value_poly(cols, COL_BIG_SIGMA0_A_OFFSET, BITS_PER_WORD, curve);
    let maj_val = word_value_poly(cols, COL_MAJ_ABC_OFFSET, BITS_PER_WORD, curve);

    // T1 carry value polynomial.
    let mut t1c_poly = vec![Scalar::zero(curve)];
    let mut p = one.clone();
    for bit in 0..T1_CARRY_BITS {
        let scaled = poly_scalar_mul(&cols[COL_T1_CARRY_OFFSET + bit], &p);
        t1c_poly = poly_add(&t1c_poly, &scaled, curve);
        p = p.mul(&two);
    }
    let t2c_poly = &cols[COL_T2_CARRY_OFFSET];
    let anc_poly = &cols[COL_A_NEW_CARRY_OFFSET];
    let enc_poly = &cols[COL_E_NEW_CARRY_OFFSET];

    // T1_value(X) = new_e + enc·2^32 − d.
    let enc_scaled = poly_scalar_mul(enc_poly, &two_pow_32);
    let t1_val = poly_sub(&poly_add(&new_e_val, &enc_scaled, curve), &d_val, curve);

    // (A) body_a = (h + Σ₁ + Ch + K + W) − (T1_value + t1c·2^32)
    let mut lhs_a = poly_add(&h_val, &s1_val, curve);
    lhs_a = poly_add(&lhs_a, &ch_val, curve);
    lhs_a = poly_add(&lhs_a, &k_val, curve);
    lhs_a = poly_add(&lhs_a, &w_val, curve);
    let t1c_scaled = poly_scalar_mul(&t1c_poly, &two_pow_32);
    let rhs_a = poly_add(&t1_val, &t1c_scaled, curve);
    let body_a = poly_sub(&lhs_a, &rhs_a, curve);

    // T2_value(X) = new_a + anc·2^32 − T1_value.
    let anc_scaled = poly_scalar_mul(anc_poly, &two_pow_32);
    let t2_val = poly_sub(&poly_add(&new_a_val, &anc_scaled, curve), &t1_val, curve);

    // (B) body_b = (Σ₀ + Maj) − (T2_value + t2c·2^32)
    let lhs_b = poly_add(&s0_val, &maj_val, curve);
    let t2c_scaled = poly_scalar_mul(t2c_poly, &two_pow_32);
    let rhs_b = poly_add(&t2_val, &t2c_scaled, curve);
    let body_b = poly_sub(&lhs_b, &rhs_b, curve);

    // Combine: body_a + β · body_b
    let body_b_scaled = poly_scalar_mul(&body_b, beta);
    poly_add(&body_a, &body_b_scaled, curve)
}

fn build_passthrough_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    let pairs: [(usize, usize); 6] = [
        (VAR_B, VAR_A),
        (VAR_C, VAR_B),
        (VAR_D, VAR_C),
        (VAR_F, VAR_E),
        (VAR_G, VAR_F),
        (VAR_H, VAR_G),
    ];
    for &(dst, src) in &pairs {
        for bit in 0..BITS_PER_WORD {
            let s = &cols[before(src, bit)];
            let d = &cols[after(dst, bit)];
            let body = poly_sub(d, s, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
            bp = bp.mul(beta);
        }
    }
    acc
}

fn build_k_binding_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let zero_poly: Vec<Scalar> = vec![Scalar::zero(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let kb = &cols[k_bit(bit)];
        for k in 0..NUM_ROUNDS {
            let rc_bit = ((ROUND_CONSTANTS[k] >> bit) & 1) as u64;
            let rc_poly = if rc_bit == 1 { one_poly.clone() } else { zero_poly.clone() };
            let sel = &cols[sel_round(k)];
            let diff = poly_sub(kb, &rc_poly, curve);
            let gated = poly_mul(sel, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &bp), curve);
            bp = bp.mul(beta);
        }
    }
    acc
}

fn build_sel_binary_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for k in 0..NUM_SEL_ROUND {
        let s = &cols[sel_round(k)];
        let s_m1 = poly_sub(s, &one_poly, curve);
        let body = poly_mul(s, &s_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_small_sigma0_w_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let w_rot7 = &cols[sha256_air::w_bit((bit + 7) % BITS_PER_WORD)];
        let w_rot18 = &cols[sha256_air::w_bit((bit + 18) % BITS_PER_WORD)];
        let xor01 = &cols[sha256_air::xor01_ss0_bit(bit)];
        let ss0 = &cols[sha256_air::small_sigma0_w_bit(bit)];

        // Stage 1: xor01 - (w_rot7 + w_rot18 - 2·w_rot7·w_rot18).
        let ab = poly_mul(w_rot7, w_rot18, curve);
        let two_ab = poly_scalar_mul(&ab, &two);
        let xor1 = poly_sub(&poly_add(w_rot7, w_rot18, curve), &two_ab, curve);
        let body1 = poly_sub(xor01, &xor1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        // Stage 2: ss0 - (xor01 XOR SHR_3(W)[i]).
        if bit + 3 < BITS_PER_WORD {
            let w_shr3 = &cols[sha256_air::w_bit(bit + 3)];
            let ab2 = poly_mul(xor01, w_shr3, curve);
            let two_ab2 = poly_scalar_mul(&ab2, &two);
            let xor2 = poly_sub(&poly_add(xor01, w_shr3, curve), &two_ab2, curve);
            let body2 = poly_sub(ss0, &xor2, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        } else {
            // ss0[i] = xor01[i].
            let body2 = poly_sub(ss0, xor01, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        }
        bp = bp.mul(beta);
    }
    acc
}

fn build_small_sigma1_w_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let w_rot17 = &cols[sha256_air::w_bit((bit + 17) % BITS_PER_WORD)];
        let w_rot19 = &cols[sha256_air::w_bit((bit + 19) % BITS_PER_WORD)];
        let xor01 = &cols[sha256_air::xor01_ss1_bit(bit)];
        let ss1 = &cols[sha256_air::small_sigma1_w_bit(bit)];

        let ab = poly_mul(w_rot17, w_rot19, curve);
        let two_ab = poly_scalar_mul(&ab, &two);
        let xor1 = poly_sub(&poly_add(w_rot17, w_rot19, curve), &two_ab, curve);
        let body1 = poly_sub(xor01, &xor1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        if bit + 10 < BITS_PER_WORD {
            let w_shr10 = &cols[sha256_air::w_bit(bit + 10)];
            let ab2 = poly_mul(xor01, w_shr10, curve);
            let two_ab2 = poly_scalar_mul(&ab2, &two);
            let xor2 = poly_sub(&poly_add(xor01, w_shr10, curve), &two_ab2, curve);
            let body2 = poly_sub(ss1, &xor2, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        } else {
            let body2 = poly_sub(ss1, xor01, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        }
        bp = bp.mul(beta);
    }
    acc
}

fn build_sel_sum_01_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let mut sum = vec![Scalar::zero(curve)];
    for k in 0..NUM_SEL_ROUND {
        sum = poly_add(&sum, &cols[sel_round(k)], curve);
    }
    let mut sum_m1 = sum.clone();
    if sum_m1.is_empty() {
        sum_m1.push(Scalar::zero(curve));
    }
    sum_m1[0] = sum_m1[0].sub(&Scalar::one(curve));
    poly_mul(&sum, &sum_m1, curve)
}

fn build_is_first_inv_row_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[sha256_air::COL_IS_FIRST_INV_ROW];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_first_block_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[sha256_air::COL_IS_FIRST_BLOCK];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_first_block_pinned_at_anchor_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let anchor = &cols[sha256_air::COL_IS_FIRST_INV_ROW];
    let fb = &cols[sha256_air::COL_IS_FIRST_BLOCK];
    let one_minus_fb = poly_sub(&one_poly, fb, curve);
    poly_mul(anchor, &one_minus_fb, curve)
}

fn build_state_in_pin_at_anchor_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let anchor = &cols[sha256_air::COL_IS_FIRST_INV_ROW];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let state_word = &cols[sha256_air::state_in_word(v)];
        let init_poly = vec![Scalar::from_u64(INITIAL_HASH[v] as u64, curve)];
        let body = poly_sub(state_word, &init_poly, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    poly_mul(anchor, &acc, curve)
}

fn build_carry_bit_binary_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let cc = &cols[sha256_air::chain_carry_bit(v)];
        let cc_body = poly_mul(cc, &poly_sub(cc, &one_poly, curve), curve);
        acc = poly_add(&acc, &poly_scalar_mul(&cc_body, &bp), curve);
        bp = bp.mul(beta);
        let bc = &cols[sha256_air::binding_carry_bit(v)];
        let bc_body = poly_mul(bc, &poly_sub(bc, &one_poly, curve), curve);
        acc = poly_add(&acc, &poly_scalar_mul(&bc_body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_is_last_block_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[sha256_air::COL_IS_LAST_BLOCK];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_output_byte_binding_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let two_to_32 = Scalar::from_u64(1u64 << 32, curve);
    let pow_be: [Scalar; 4] = [
        Scalar::from_u64(1u64 << 24, curve),
        Scalar::from_u64(1u64 << 16, curve),
        Scalar::from_u64(1u64 << 8, curve),
        Scalar::from_u64(1, curve),
    ];

    let last_block = &cols[sha256_air::COL_IS_LAST_BLOCK];
    let last_sel = &cols[sel_round(NUM_ROUNDS - 1)];
    let aggregator = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    let gate_partial = poly_mul(last_block, last_sel, curve);
    let gate = poly_mul(aggregator, &gate_partial, curve);

    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let mut digest_word = vec![Scalar::zero(curve)];
        for k in 0..4 {
            let byte = &cols[sha256_air::inv_output_byte(4 * v + k)];
            digest_word =
                poly_add(&digest_word, &poly_scalar_mul(byte, &pow_be[k]), curve);
        }
        let bc = &cols[sha256_air::binding_carry_bit(v)];
        let carry_scaled = poly_scalar_mul(bc, &two_to_32);
        let lhs = poly_add(&digest_word, &carry_scaled, curve);
        let state_in = &cols[sha256_air::state_in_word(v)];
        let mut after_now = vec![Scalar::zero(curve)];
        let mut pow = Scalar::one(curve);
        for bit in 0..BITS_PER_WORD {
            let bit_col = &cols[after(v, bit)];
            after_now =
                poly_add(&after_now, &poly_scalar_mul(bit_col, &pow), curve);
            pow = pow.mul(&two);
        }
        let rhs = poly_add(state_in, &after_now, curve);
        let body = poly_sub(&lhs, &rhs, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    poly_mul(&gate, &acc, curve)
}

fn build_aggregator_active_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

/// 20. chain_carry_localized polynomial form.
fn build_chain_carry_localized_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let aggregator = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    let last_sel = &cols[sel_round(NUM_ROUNDS - 1)];
    let last_block = &cols[sha256_air::COL_IS_LAST_BLOCK];
    // active_gate = AGGREGATOR_ACTIVE · SEL_ROUND[NUM_ROUNDS-1] · (1 − IS_LAST_BLOCK)
    let one_minus_last_block = poly_sub(&one_poly, last_block, curve);
    let part = poly_mul(aggregator, last_sel, curve);
    let active_gate = poly_mul(&part, &one_minus_last_block, curve);
    let neg_gate = poly_sub(&one_poly, &active_gate, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let cc = &cols[sha256_air::chain_carry_bit(v)];
        let body = poly_mul(&neg_gate, cc, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

/// 21. binding_carry_localized polynomial form.
fn build_binding_carry_localized_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let aggregator = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    let last_sel = &cols[sel_round(NUM_ROUNDS - 1)];
    let last_block = &cols[sha256_air::COL_IS_LAST_BLOCK];
    let part = poly_mul(aggregator, last_block, curve);
    let active_gate = poly_mul(&part, last_sel, curve);
    let neg_gate = poly_sub(&one_poly, &active_gate, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for v in 0..sha256_air::NUM_OUTPUT_WORDS {
        let bc = &cols[sha256_air::binding_carry_bit(v)];
        let body = poly_mul(&neg_gate, bc, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

/// 22. sel_exactly_one_when_aggregator_active polynomial form.
fn build_sel_exactly_one_when_aggregator_active_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let aggregator = &cols[sha256_air::COL_AGGREGATOR_ACTIVE];
    let mut sum = vec![Scalar::zero(curve)];
    for k in 0..NUM_SEL_ROUND {
        sum = poly_add(&sum, &cols[sel_round(k)], curve);
    }
    let sum_minus_one = poly_sub(&sum, &one_poly, curve);
    poly_mul(aggregator, &sum_minus_one, curve)
}

/// 23. w_recurrence polynomial form (Task #179, generalized in #193):
///   (Σ_{t∈16..64} SEL_ROUND[t]) ·
///       (σ1_addend + W9_addend + σ0_addend + W0_addend
///        − W_word − carry·2^32)
fn build_w_recurrence_at_anchor_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let two = Scalar::from_u64(2, curve);
    let two_to_32 = Scalar::from_u64(1u64 << 32, curve);
    // Σ_{t=16..NUM_ROUNDS} SEL_ROUND[t] selector sum.
    let mut sel_recurrence = vec![Scalar::zero(curve)];
    for t in 16..NUM_ROUNDS {
        sel_recurrence = poly_add(&sel_recurrence, &cols[sel_round(t)], curve);
    }
    // W_word = Σ_i W_bit[i] · 2^i.
    let w_word = word_value_poly(cols, sha256_air::COL_W_OFFSET, BITS_PER_WORD, curve);
    // carry = Σ_i bit_i · 2^i (2 bits).
    let mut carry = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..sha256_air::W_RECURRENCE_CARRY_BITS {
        let scaled = poly_scalar_mul(
            &cols[sha256_air::w_recurrence_carry_bit(bit)],
            &bp,
        );
        carry = poly_add(&carry, &scaled, curve);
        bp = bp.mul(&two);
    }
    let carry_scaled = poly_scalar_mul(&carry, &two_to_32);
    let s1 = &cols[sha256_air::w_recurrence_sigma1_w14_word()];
    let w9 = &cols[sha256_air::w_recurrence_w9_word()];
    let s0 = &cols[sha256_air::w_recurrence_sigma0_w1_word()];
    let w0 = &cols[sha256_air::w_recurrence_w0_word()];
    // addend_sum = σ1 + W9 + σ0 + W0; rhs = W_word + carry·2^32; body =
    // addend_sum - rhs.
    let mut addend_sum = poly_add(s1, w9, curve);
    addend_sum = poly_add(&addend_sum, s0, curve);
    addend_sum = poly_add(&addend_sum, w0, curve);
    let rhs = poly_add(&w_word, &carry_scaled, curve);
    let body = poly_sub(&addend_sum, &rhs, curve);
    poly_mul(&sel_recurrence, &body, curve)
}

fn build_inv_input_byte_binding_poly(
    cols: &[Vec<Scalar>],
    beta: &Scalar,
) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let fb = &cols[sha256_air::COL_IS_FIRST_BLOCK];
    let w_word = word_value_poly(cols, sha256_air::COL_W_OFFSET, BITS_PER_WORD, curve);
    let pow_be: [Scalar; 4] = [
        Scalar::from_u64(1u64 << 24, curve),
        Scalar::from_u64(1u64 << 16, curve),
        Scalar::from_u64(1u64 << 8, curve),
        Scalar::from_u64(1, curve),
    ];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for r in 0..16 {
        let sel = &cols[sel_round(r)];
        let mut block_word = vec![Scalar::zero(curve)];
        for b in 0..4 {
            let byte = &cols[sha256_air::inv_input_byte(4 * r + b)];
            let scaled = poly_scalar_mul(byte, &pow_be[b]);
            block_word = poly_add(&block_word, &scaled, curve);
        }
        let diff = poly_sub(&block_word, &w_word, curve);
        let fb_diff = poly_mul(fb, &diff, curve);
        let body = poly_mul(sel, &fb_diff, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

// ──── Cross-row helpers ────────────────────────────────────────────────

/// Boundary rows whose cross-row transition must be excluded from
/// vanishing. For a SHA-256 trace with real rows `0..num_rows` padded to
/// `domain_size`, these are:
///   - Row index `63, 127, …, num_rows-1` (last round of each block —
///     next row begins a fresh block).
///   - Row index `domain_size-1` (domain wrap-around).
///
/// Returns a sorted, deduplicated list (modulo `domain_size`).
fn boundary_rows(num_rows: usize, domain_size: usize) -> Vec<usize> {
    let mut set: std::collections::BTreeSet<usize> = Default::default();
    if num_rows > 0 {
        let mut r = NUM_ROUNDS - 1;
        while r < num_rows {
            set.insert(r);
            r += NUM_ROUNDS;
        }
    }
    if domain_size > 0 {
        set.insert(domain_size - 1);
    }
    set.into_iter().collect()
}

/// Boundary rows for the aggregator-invariance shifted constraint
/// (body 1). Excludes ONLY the last real-row transition + domain wrap
/// — block boundaries are NOT excluded because the per-invocation
/// aggregator columns must remain constant across them within a single
/// invocation.
fn invariance_boundary_rows(num_rows: usize, domain_size: usize) -> Vec<usize> {
    let mut set: std::collections::BTreeSet<usize> = Default::default();
    if num_rows > 0 {
        set.insert(num_rows - 1);
    }
    if domain_size > 0 {
        set.insert(domain_size - 1);
    }
    set.into_iter().collect()
}

/// Derive the proof domain size from `omega_n_minus_1 = ω^(n−1)` by
/// repeated squaring. Domain sizes are always powers of two in this
/// codebase, so the order of `ω = omega_n_minus_1^(−1)` is the smallest
/// `2^k` such that `ω^(2^k) = 1`. Mirrors the helper in
/// [`crate::mpt_constraints`].
fn domain_size_from_omega_n_minus_1(omega_n_minus_1: &Scalar) -> usize {
    let curve = omega_n_minus_1.curve_type();
    let one = Scalar::one(curve);
    let omega = omega_n_minus_1.inverse();
    let mut p = omega;
    let mut size: usize = 1;
    while size <= (1 << 30) {
        if p.sub(&one).is_zero() {
            return size;
        }
        p = p.mul(&p);
        size *= 2;
    }
    size
}

// ──── VmConstraintSystem implementation ────────────────────────────────

impl VmConstraintSystem for Sha256ConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "bit_validity".into(),
            "big_sigma0_a_definition".into(),
            "big_sigma1_e_definition".into(),
            "ch_definition".into(),
            "maj_definition".into(),
            "round_additions".into(),
            "passthrough".into(),
            "k_binding".into(),
            "sel_binary".into(),
            "sel_sum_01".into(),
            "small_sigma0_w_definition".into(),
            "small_sigma1_w_definition".into(),
            "is_first_inv_row_binary".into(),
            "is_first_block_binary".into(),
            "is_first_block_pinned_at_anchor".into(),
            "inv_input_byte_binding".into(),
            "state_in_pin_at_anchor".into(),
            "carry_bit_binary".into(),
            "output_byte_binding".into(),
            "aggregator_active_binary".into(),
            "is_last_block_binary".into(),
            "chain_carry_localized".into(),
            "binding_carry_localized".into(),
            "sel_exactly_one_when_aggregator_active".into(),
            "w_recurrence".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(
            columns.len() == NUM_SHA256_COLUMNS,
            "sha256 AIR expects {} columns",
            NUM_SHA256_COLUMNS
        );
        let curve = columns[0][0].curve_type();
        let beta = Scalar::from_u64(2, curve);
        let mut evals: Vec<Vec<Scalar>> = sha256_air::evaluate_constraints(columns, &beta)
            .into_iter()
            .map(|c| c.values)
            .collect();
        let n = columns[0].len();
        let one = Scalar::one(curve);
        // 11th category: is_first_inv_row_binary.
        let mut anchor_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let v = &columns[sha256_air::COL_IS_FIRST_INV_ROW][row];
            anchor_bin[row] = v.mul(&v.sub(&one));
        }
        evals.push(anchor_bin);
        // 12th category: is_first_block_binary.
        let mut fb_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let v = &columns[sha256_air::COL_IS_FIRST_BLOCK][row];
            fb_bin[row] = v.mul(&v.sub(&one));
        }
        evals.push(fb_bin);
        // 13th category: is_first_block_pinned_at_anchor.
        let mut fb_pin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let anchor = &columns[sha256_air::COL_IS_FIRST_INV_ROW][row];
            let fb = &columns[sha256_air::COL_IS_FIRST_BLOCK][row];
            fb_pin[row] = anchor.mul(&one.sub(fb));
        }
        evals.push(fb_pin);
        // 14th category: inv_input_byte_binding (per-row).
        let pow_be: [Scalar; 4] = [
            Scalar::from_u64(1u64 << 24, curve),
            Scalar::from_u64(1u64 << 16, curve),
            Scalar::from_u64(1u64 << 8, curve),
            Scalar::from_u64(1, curve),
        ];
        let two = Scalar::from_u64(2, curve);
        let two_to_32 = Scalar::from_u64(1u64 << 32, curve);
        let mut binding = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let fb = &columns[sha256_air::COL_IS_FIRST_BLOCK][row];
            // W word at this row from W_bit columns (LSB-first).
            let mut w_word = Scalar::zero(curve);
            let mut pow = Scalar::one(curve);
            for bit in 0..BITS_PER_WORD {
                let bv = &columns[sha256_air::COL_W_OFFSET + bit][row];
                w_word = w_word.add(&pow.mul(bv));
                pow = pow.mul(&two);
            }
            // Σ_{r=0..16} β^r · sel_round[r] · fb · (block_word_r − w_word).
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for r in 0..16 {
                let sel = &columns[sel_round(r)][row];
                let mut block_word = Scalar::zero(curve);
                for b in 0..4 {
                    let byte = &columns[sha256_air::inv_input_byte(4 * r + b)][row];
                    block_word = block_word.add(&byte.mul(&pow_be[b]));
                }
                let body = sel.mul(&fb.mul(&block_word.sub(&w_word)));
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            binding[row] = acc;
        }
        evals.push(binding);
        // 15th category: state_in_pin_at_anchor (per-row).
        let mut state_pin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let anchor = &columns[sha256_air::COL_IS_FIRST_INV_ROW][row];
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for v in 0..sha256_air::NUM_OUTPUT_WORDS {
                let state_word = &columns[sha256_air::state_in_word(v)][row];
                let init = Scalar::from_u64(INITIAL_HASH[v] as u64, curve);
                let body = state_word.sub(&init);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            state_pin[row] = anchor.mul(&acc);
        }
        evals.push(state_pin);
        // 16th category: carry_bit_binary (per-row).
        let mut carry_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for v in 0..sha256_air::NUM_OUTPUT_WORDS {
                let cc = &columns[sha256_air::chain_carry_bit(v)][row];
                let cc_body = cc.mul(&cc.sub(&one));
                acc = acc.add(&bp.mul(&cc_body));
                bp = bp.mul(&beta);
                let bc = &columns[sha256_air::binding_carry_bit(v)][row];
                let bc_body = bc.mul(&bc.sub(&one));
                acc = acc.add(&bp.mul(&bc_body));
                bp = bp.mul(&beta);
            }
            carry_bin[row] = acc;
        }
        evals.push(carry_bin);
        // 17th category: output_byte_binding (per-row).
        let mut out_binding = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let last_block = &columns[sha256_air::COL_IS_LAST_BLOCK][row];
            let last_sel = &columns[sel_round(NUM_ROUNDS - 1)][row];
            let aggregator = &columns[sha256_air::COL_AGGREGATOR_ACTIVE][row];
            let gate = aggregator.mul(last_block).mul(last_sel);
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for v in 0..sha256_air::NUM_OUTPUT_WORDS {
                let mut digest_word = Scalar::zero(curve);
                for k in 0..4 {
                    let byte = &columns[sha256_air::inv_output_byte(4 * v + k)][row];
                    digest_word = digest_word.add(&byte.mul(&pow_be[k]));
                }
                let bc = &columns[sha256_air::binding_carry_bit(v)][row];
                let state_in = &columns[sha256_air::state_in_word(v)][row];
                let mut after_now = Scalar::zero(curve);
                let mut pow = Scalar::one(curve);
                for bit in 0..BITS_PER_WORD {
                    let bit_col = &columns[after(v, bit)][row];
                    after_now = after_now.add(&pow.mul(bit_col));
                    pow = pow.mul(&two);
                }
                let rhs = state_in.add(&after_now);
                let lhs = digest_word.add(&bc.mul(&two_to_32));
                let body = lhs.sub(&rhs);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            out_binding[row] = gate.mul(&acc);
        }
        evals.push(out_binding);
        // 18th category: aggregator_active_binary (per-row).
        let mut agg_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let v = &columns[sha256_air::COL_AGGREGATOR_ACTIVE][row];
            agg_bin[row] = v.mul(&v.sub(&one));
        }
        evals.push(agg_bin);
        // 19th category: is_last_block_binary (per-row).
        let mut last_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let v = &columns[sha256_air::COL_IS_LAST_BLOCK][row];
            last_bin[row] = v.mul(&v.sub(&one));
        }
        evals.push(last_bin);
        // 20th category: chain_carry_localized.
        let mut chain_loc = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let aggregator = &columns[sha256_air::COL_AGGREGATOR_ACTIVE][row];
            let last_sel = &columns[sel_round(NUM_ROUNDS - 1)][row];
            let last_block = &columns[sha256_air::COL_IS_LAST_BLOCK][row];
            let active_gate = aggregator.mul(last_sel).mul(&one.sub(last_block));
            let neg_gate = one.sub(&active_gate);
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for v in 0..sha256_air::NUM_OUTPUT_WORDS {
                let cc = &columns[sha256_air::chain_carry_bit(v)][row];
                let body = neg_gate.mul(cc);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            chain_loc[row] = acc;
        }
        evals.push(chain_loc);
        // 21st category: binding_carry_localized.
        let mut binding_loc = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let aggregator = &columns[sha256_air::COL_AGGREGATOR_ACTIVE][row];
            let last_sel = &columns[sel_round(NUM_ROUNDS - 1)][row];
            let last_block = &columns[sha256_air::COL_IS_LAST_BLOCK][row];
            let active_gate = aggregator.mul(last_block).mul(last_sel);
            let neg_gate = one.sub(&active_gate);
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for v in 0..sha256_air::NUM_OUTPUT_WORDS {
                let bc = &columns[sha256_air::binding_carry_bit(v)][row];
                let body = neg_gate.mul(bc);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            binding_loc[row] = acc;
        }
        evals.push(binding_loc);
        // 22nd category: sel_exactly_one_when_aggregator_active.
        let mut sel_exact = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let aggregator = &columns[sha256_air::COL_AGGREGATOR_ACTIVE][row];
            let mut sum = Scalar::zero(curve);
            for k in 0..NUM_SEL_ROUND {
                sum = sum.add(&columns[sel_round(k)][row]);
            }
            sel_exact[row] = aggregator.mul(&sum.sub(&one));
        }
        evals.push(sel_exact);
        // 23rd category: w_recurrence (Task #179, generalized in #193).
        // (Σ_{t=16..64} SEL_ROUND[t]) ·
        //     (σ1 + W9 + σ0 + W0 − W_word − carry·2^32) = 0
        let mut w_rec = vec![Scalar::zero(curve); n];
        for row in 0..n {
            // Σ_{t=16..NUM_ROUNDS} SEL_ROUND[t] at this row.
            let mut sel_recurrence = Scalar::zero(curve);
            for t in 16..NUM_ROUNDS {
                sel_recurrence = sel_recurrence.add(&columns[sel_round(t)][row]);
            }
            // Reconstruct W word LSB-first from the 32 W bit columns.
            let mut w_word = Scalar::zero(curve);
            let mut pow = Scalar::one(curve);
            for bit in 0..BITS_PER_WORD {
                let bv = &columns[sha256_air::w_bit(bit)][row];
                w_word = w_word.add(&pow.mul(bv));
                pow = pow.mul(&two);
            }
            // Reconstruct carry from the 2-bit carry column.
            let mut carry = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for bit in 0..sha256_air::W_RECURRENCE_CARRY_BITS {
                let cv = &columns[sha256_air::w_recurrence_carry_bit(bit)][row];
                carry = carry.add(&bp.mul(cv));
                bp = bp.mul(&two);
            }
            let s1 = &columns[sha256_air::w_recurrence_sigma1_w14_word()][row];
            let w9 = &columns[sha256_air::w_recurrence_w9_word()][row];
            let s0 = &columns[sha256_air::w_recurrence_sigma0_w1_word()][row];
            let w0 = &columns[sha256_air::w_recurrence_w0_word()][row];
            // addend_sum = σ1 + W9 + σ0 + W0; rhs = W_word + carry·2^32.
            let addend_sum = s1.add(w9).add(s0).add(w0);
            let rhs = w_word.add(&carry.mul(&two_to_32));
            let body = addend_sum.sub(&rhs);
            w_rec[row] = sel_recurrence.mul(&body);
        }
        evals.push(w_rec);
        evals
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals_at_z.len() < NUM_SHA256_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let beta = alpha; // β = α as in keccak / nonnative_fp wirings.
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            eval_bit_validity_at_point(col_evals_at_z, beta),
            eval_sigma0_at_point(col_evals_at_z, beta),
            eval_sigma1_at_point(col_evals_at_z, beta),
            eval_ch_at_point(col_evals_at_z, beta),
            eval_maj_at_point(col_evals_at_z, beta),
            eval_round_adds_at_point(col_evals_at_z, beta),
            eval_passthrough_at_point(col_evals_at_z, beta),
            eval_k_binding_at_point(col_evals_at_z, beta),
            eval_sel_binary_at_point(col_evals_at_z, beta),
            eval_sel_sum_01_at_point(col_evals_at_z),
            eval_small_sigma0_w_at_point(col_evals_at_z, beta),
            eval_small_sigma1_w_at_point(col_evals_at_z, beta),
            eval_is_first_inv_row_binary_at_point(col_evals_at_z),
            eval_is_first_block_binary_at_point(col_evals_at_z),
            eval_is_first_block_pinned_at_anchor_at_point(col_evals_at_z),
            eval_inv_input_byte_binding_at_point(col_evals_at_z, beta),
            eval_state_in_pin_at_anchor_at_point(col_evals_at_z, beta),
            eval_carry_bit_binary_at_point(col_evals_at_z, beta),
            eval_output_byte_binding_at_point(col_evals_at_z, beta),
            eval_aggregator_active_binary_at_point(col_evals_at_z),
            eval_is_last_block_binary_at_point(col_evals_at_z),
            eval_chain_carry_localized_at_point(col_evals_at_z, beta),
            eval_binding_carry_localized_at_point(col_evals_at_z, beta),
            eval_sel_exactly_one_when_aggregator_active_at_point(col_evals_at_z),
            eval_w_recurrence_at_anchor_at_point(col_evals_at_z),
        ];
        let curve = alpha.curve_type();
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        (0..NUM_SEL_ROUND).map(sel_round).collect()
    }

    /// Return `None`: padding rows carry all-zero data + selector columns,
    /// which makes every row-local body vanish. The `sel_sum_01`
    /// constraint `sum·(sum−1) = 0` is satisfied with sum = 0.
    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        // Defensive: zero every column on padding rows. The trace
        // constructor already zero-fills, but we make this explicit so any
        // upstream stage that mutated padding cells is reset.
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        if columns.len() < NUM_SHA256_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_SHA256_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn build_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        use rayon::prelude::*;
        let curve = alpha.curve_type();
        let beta = alpha.clone();
        type BuilderRet = Vec<Scalar>;
        let category_builders: Vec<Box<dyn Fn() -> BuilderRet + Sync + Send>> = vec![
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_bit_validity_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_sigma0_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_sigma1_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_ch_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_maj_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_round_adds_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_passthrough_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_k_binding_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_sel_binary_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                move || build_sel_sum_01_poly(&cols)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_small_sigma0_w_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_small_sigma1_w_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_is_first_inv_row_binary_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_is_first_block_binary_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_is_first_block_pinned_at_anchor_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_inv_input_byte_binding_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_state_in_pin_at_anchor_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_carry_bit_binary_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_output_byte_binding_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_aggregator_active_binary_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_is_last_block_binary_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_chain_carry_localized_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_binding_carry_localized_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_sel_exactly_one_when_aggregator_active_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_w_recurrence_at_anchor_poly(&cols, c)
            }),
        ];
        let bodies: Vec<Vec<Scalar>> = category_builders.par_iter().map(|f| f()).collect();

        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            let scaled = poly_scalar_mul(body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    // ── Cross-row support ──────────────────────────────────────────────

    fn shifted_column_indices(&self) -> Vec<usize> {
        // 256 BEFORE bits (body 0) + 64 INV_INPUT_BYTE + 32
        // INV_OUTPUT_BYTE (body 1) + 1 IS_FIRST_BLOCK (body 2) +
        // 8 STATE_IN_WORD (bodies 3 & 4) + 1 IS_LAST_BLOCK (body 5).
        // Order matters: the verifier reads `shifted_evals` in this
        // exact order.
        let mut idxs: Vec<usize> = (COL_BEFORE_OFFSET..COL_BEFORE_OFFSET + BITS_PER_STATE).collect();
        for b in 0..sha256_air::INV_INPUT_LEN {
            idxs.push(sha256_air::inv_input_byte(b));
        }
        for b in 0..sha256_air::INV_OUTPUT_LEN {
            idxs.push(sha256_air::inv_output_byte(b));
        }
        idxs.push(sha256_air::COL_IS_FIRST_BLOCK);
        for v in 0..sha256_air::NUM_OUTPUT_WORDS {
            idxs.push(sha256_air::state_in_word(v));
        }
        idxs.push(sha256_air::COL_IS_LAST_BLOCK);
        idxs
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
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
        let total_shifted = BITS_PER_STATE
            + sha256_air::INV_INPUT_LEN
            + sha256_air::INV_OUTPUT_LEN
            + 1
            + sha256_air::NUM_OUTPUT_WORDS
            + 1;
        if shifted_evals.len() != total_shifted {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let beta = alpha;

        // ── Body 0: after → before binding (existing) ──
        let mut body_0 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        let mut shift_idx = 0usize;
        for var in 0..NUM_WORKING_VARS {
            for bit in 0..BITS_PER_WORD {
                let af = &col_evals_at_z[after(var, bit)];
                let bf_shift = &shifted_evals[shift_idx];
                body_0 = body_0.add(&bp.mul(&af.sub(bf_shift)));
                bp = bp.mul(beta);
                shift_idx += 1;
            }
        }

        // ── Body 1: aggregator invariance (new) ──
        // body_1 = Σ β^i · (INV_X(ω·z) − INV_X(z)) over INPUT + OUTPUT
        // bytes. Constant across all rows of an invocation.
        let mut body_1 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for b in 0..sha256_air::INV_INPUT_LEN {
            let cur = &col_evals_at_z[sha256_air::inv_input_byte(b)];
            let next = &shifted_evals[shift_idx];
            body_1 = body_1.add(&bp.mul(&next.sub(cur)));
            bp = bp.mul(beta);
            shift_idx += 1;
        }
        for b in 0..sha256_air::INV_OUTPUT_LEN {
            let cur = &col_evals_at_z[sha256_air::inv_output_byte(b)];
            let next = &shifted_evals[shift_idx];
            body_1 = body_1.add(&bp.mul(&next.sub(cur)));
            bp = bp.mul(beta);
            shift_idx += 1;
        }

        // Boundary-row exclusion. Derive ω from `omega_n_minus_1` rather
        // than `self.omega`: the actual proof domain may be larger than
        // the trace's natural padded size when downstream LogUp
        // machinery inflates it, and `self.omega` would be stale.
        // ω · ω^(n-1) = ω^n = 1 ⟹ ω = inverse(ω^(n-1)).
        //
        // Body 0 exclusion: every block boundary (rows 63, 127, …) +
        // last real + wrap. MUST match `boundary_rows(num_rows,
        // domain_size)` used by the prover builder — a pre-existing
        // divergence that only excluded last+wrap was sound only for
        // single-block traces; this fix makes multi-block (e.g.
        // sha256_pair = 2 blocks) sound too.
        //
        // Body 1 exclusion: only last real + wrap. Block boundaries are
        // NOT excluded since aggregator invariance must hold across
        // them within an invocation.
        let omega = omega_n_minus_1.inverse();
        let domain_size = domain_size_from_omega_n_minus_1(omega_n_minus_1);
        let rows_0 = boundary_rows(self.num_rows, domain_size);
        let mut exclusion_0 = Scalar::one(curve);
        for &r in &rows_0 {
            let omega_r = scalar_pow(&omega, r as u64);
            exclusion_0 = exclusion_0.mul(&z.sub(&omega_r));
        }
        let rows_1 = invariance_boundary_rows(self.num_rows, domain_size);
        let mut exclusion_1 = Scalar::one(curve);
        for &r in &rows_1 {
            let omega_r = scalar_pow(&omega, r as u64);
            exclusion_1 = exclusion_1.mul(&z.sub(&omega_r));
        }

        // ── Body 2: IS_FIRST_BLOCK invariance within a block ──
        let one = Scalar::one(curve);
        let last_sel = &col_evals_at_z[sel_round(NUM_ROUNDS - 1)];
        let fb_curr = &col_evals_at_z[sha256_air::COL_IS_FIRST_BLOCK];
        let fb_next = &shifted_evals[shift_idx];
        let body_2 = one.sub(last_sel).mul(&fb_next.sub(fb_curr));
        shift_idx += 1;
        let exclusion_wrap = z.sub(omega_n_minus_1);

        // ── Body 3: STATE_IN_WORD within-block invariance (β-RLC over v) ──
        let mut body_3 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        let invariance_gate = one.sub(last_sel);
        for v in 0..sha256_air::NUM_OUTPUT_WORDS {
            let curr = &col_evals_at_z[sha256_air::state_in_word(v)];
            let next = &shifted_evals[shift_idx + v];
            let diff = next.sub(curr);
            body_3 = body_3.add(&bp.mul(&invariance_gate.mul(&diff)));
            bp = bp.mul(beta);
        }

        // ── Body 4: STATE_IN_WORD cross-block chaining (β-RLC over v) ──
        // body_4 = AGGREGATOR_ACTIVE · SEL_ROUND[NUM_ROUNDS-1] ·
        //          (1 − IS_LAST_BLOCK) · Σ β^v ·
        //   (STATE_IN_next + chain_carry · 2^32 − STATE_IN_curr − AFTER_word)
        // The (1 − IS_LAST_BLOCK) factor excludes the last-block-end
        // transition (where there's no "next block" to chain to and
        // STATE_IN_next is the zero padding row).
        let two = Scalar::from_u64(2, curve);
        let two_to_32 = Scalar::from_u64(1u64 << 32, curve);
        let aggregator = &col_evals_at_z[sha256_air::COL_AGGREGATOR_ACTIVE];
        let last_block = &col_evals_at_z[sha256_air::COL_IS_LAST_BLOCK];
        let chain_gate = aggregator.mul(last_sel).mul(&one.sub(last_block));
        let mut body_4 = Scalar::zero(curve);
        let mut bp4 = Scalar::one(curve);
        for v in 0..sha256_air::NUM_OUTPUT_WORDS {
            let curr = &col_evals_at_z[sha256_air::state_in_word(v)];
            let next = &shifted_evals[shift_idx + v];
            let cc = &col_evals_at_z[sha256_air::chain_carry_bit(v)];
            let mut after_word = Scalar::zero(curve);
            let mut pow = Scalar::one(curve);
            for bit in 0..BITS_PER_WORD {
                let bit_col = &col_evals_at_z[after(v, bit)];
                after_word = after_word.add(&pow.mul(bit_col));
                pow = pow.mul(&two);
            }
            let lhs = next.add(&cc.mul(&two_to_32));
            let rhs = curr.add(&after_word);
            let diff = lhs.sub(&rhs);
            body_4 = body_4.add(&bp4.mul(&chain_gate.mul(&diff)));
            bp4 = bp4.mul(beta);
        }
        shift_idx += sha256_air::NUM_OUTPUT_WORDS;

        // ── Body 5: IS_LAST_BLOCK invariance within block (mirror body 2) ──
        let lb_curr = &col_evals_at_z[sha256_air::COL_IS_LAST_BLOCK];
        let lb_next = &shifted_evals[shift_idx];
        let body_5 = invariance_gate.mul(&lb_next.sub(lb_curr));

        // α^alpha_offset for body_0; α^(alpha_offset+k) for body_k.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term_0 = ap.mul(&body_0).mul(&exclusion_0);
        ap = ap.mul(alpha);
        let term_1 = ap.mul(&body_1).mul(&exclusion_1);
        ap = ap.mul(alpha);
        let term_2 = ap.mul(&body_2).mul(&exclusion_wrap);
        ap = ap.mul(alpha);
        let term_3 = ap.mul(&body_3).mul(&exclusion_wrap);
        ap = ap.mul(alpha);
        let term_4 = ap.mul(&body_4).mul(&exclusion_wrap);
        ap = ap.mul(alpha);
        let term_5 = ap.mul(&body_5).mul(&exclusion_wrap);
        term_0
            .add(&term_1)
            .add(&term_2)
            .add(&term_3)
            .add(&term_4)
            .add(&term_5)
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
        let beta = alpha.clone();

        // ── Body 0: after(X) − before(ω·X) bit-by-bit, β-RLC ──
        let mut body_0 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for var in 0..NUM_WORKING_VARS {
            for bit in 0..BITS_PER_WORD {
                let af = &column_coeffs[after(var, bit)];
                let bf = &column_coeffs[before(var, bit)];
                let bf_shift = poly_shift(bf, omega);
                let diff = poly_sub(af, &bf_shift, curve);
                let scaled = poly_scalar_mul(&diff, &bp);
                body_0 = poly_add(&body_0, &scaled, curve);
                bp = bp.mul(&beta);
            }
        }
        // Body 0 exclusion: block boundaries + domain wrap (= the
        // existing boundary_rows function output).
        let rows_0 = boundary_rows(self.num_rows, domain_size as usize);
        let mut excluded_0 = body_0;
        for r in &rows_0 {
            let omega_r = scalar_pow(omega, *r as u64);
            excluded_0 = poly_mul_linear(&excluded_0, &omega_r);
        }

        // ── Body 1: aggregator invariance — INV_X(ω·X) − INV_X(X) ──
        // β-RLC over 64 INPUT bytes + 32 OUTPUT bytes.
        let mut body_1 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for b in 0..sha256_air::INV_INPUT_LEN {
            let cur = &column_coeffs[sha256_air::inv_input_byte(b)];
            let cur_shift = poly_shift(cur, omega);
            let diff = poly_sub(&cur_shift, cur, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            body_1 = poly_add(&body_1, &scaled, curve);
            bp = bp.mul(&beta);
        }
        for b in 0..sha256_air::INV_OUTPUT_LEN {
            let cur = &column_coeffs[sha256_air::inv_output_byte(b)];
            let cur_shift = poly_shift(cur, omega);
            let diff = poly_sub(&cur_shift, cur, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            body_1 = poly_add(&body_1, &scaled, curve);
            bp = bp.mul(&beta);
        }
        // Body 1 exclusion: only last-real + wrap. Block boundaries are
        // intentionally NOT excluded because aggregator invariance must
        // hold across them within an invocation.
        let rows_1 = invariance_boundary_rows(self.num_rows, domain_size as usize);
        let mut excluded_1 = body_1;
        for r in &rows_1 {
            let omega_r = scalar_pow(omega, *r as u64);
            excluded_1 = poly_mul_linear(&excluded_1, &omega_r);
        }

        // ── Body 2: IS_FIRST_BLOCK invariance within a block ──
        let one_poly = vec![Scalar::one(curve)];
        let last_sel = &column_coeffs[sel_round(NUM_ROUNDS - 1)];
        let one_minus_last_sel = poly_sub(&one_poly, last_sel, curve);
        let fb = &column_coeffs[sha256_air::COL_IS_FIRST_BLOCK];
        let fb_shift = poly_shift(fb, omega);
        let fb_diff = poly_sub(&fb_shift, fb, curve);
        let body_2 = poly_mul(&one_minus_last_sel, &fb_diff, curve);
        let omega_wrap = scalar_pow(omega, (domain_size - 1) as u64);
        let excluded_2 = poly_mul_linear(&body_2, &omega_wrap);

        // ── Body 3: STATE_IN_WORD within-block invariance (β-RLC over v) ──
        let mut body_3 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for v in 0..sha256_air::NUM_OUTPUT_WORDS {
            let state_curr = &column_coeffs[sha256_air::state_in_word(v)];
            let state_shift = poly_shift(state_curr, omega);
            let diff = poly_sub(&state_shift, state_curr, curve);
            let gated = poly_mul(&one_minus_last_sel, &diff, curve);
            body_3 = poly_add(&body_3, &poly_scalar_mul(&gated, &bp), curve);
            bp = bp.mul(&beta);
        }
        let excluded_3 = poly_mul_linear(&body_3, &omega_wrap);

        // ── Body 4: STATE_IN_WORD cross-block chaining (β-RLC over v) ──
        let two = Scalar::from_u64(2, curve);
        let two_to_32 = Scalar::from_u64(1u64 << 32, curve);
        let aggregator = &column_coeffs[sha256_air::COL_AGGREGATOR_ACTIVE];
        let last_block = &column_coeffs[sha256_air::COL_IS_LAST_BLOCK];
        let one_minus_last_block = poly_sub(&one_poly, last_block, curve);
        let chain_gate_partial = poly_mul(aggregator, last_sel, curve);
        let chain_gate = poly_mul(&chain_gate_partial, &one_minus_last_block, curve);
        let mut body_4 = vec![Scalar::zero(curve)];
        let mut bp4 = Scalar::one(curve);
        for v in 0..sha256_air::NUM_OUTPUT_WORDS {
            let state_curr = &column_coeffs[sha256_air::state_in_word(v)];
            let state_shift = poly_shift(state_curr, omega);
            let cc = &column_coeffs[sha256_air::chain_carry_bit(v)];
            let cc_scaled = poly_scalar_mul(cc, &two_to_32);
            let lhs = poly_add(&state_shift, &cc_scaled, curve);
            let mut after_word = vec![Scalar::zero(curve)];
            let mut pow = Scalar::one(curve);
            for bit in 0..BITS_PER_WORD {
                let bit_col = &column_coeffs[after(v, bit)];
                after_word =
                    poly_add(&after_word, &poly_scalar_mul(bit_col, &pow), curve);
                pow = pow.mul(&two);
            }
            let rhs = poly_add(state_curr, &after_word, curve);
            let diff = poly_sub(&lhs, &rhs, curve);
            let gated = poly_mul(&chain_gate, &diff, curve);
            body_4 = poly_add(&body_4, &poly_scalar_mul(&gated, &bp4), curve);
            bp4 = bp4.mul(&beta);
        }
        let excluded_4 = poly_mul_linear(&body_4, &omega_wrap);

        // ── Body 5: IS_LAST_BLOCK invariance within block ──
        let lb = &column_coeffs[sha256_air::COL_IS_LAST_BLOCK];
        let lb_shift = poly_shift(lb, omega);
        let lb_diff = poly_sub(&lb_shift, lb, curve);
        let body_5 = poly_mul(&one_minus_last_sel, &lb_diff, curve);
        let excluded_5 = poly_mul_linear(&body_5, &omega_wrap);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term_0 = poly_scalar_mul(&excluded_0, &ap);
        ap = ap.mul(alpha);
        let term_1 = poly_scalar_mul(&excluded_1, &ap);
        ap = ap.mul(alpha);
        let term_2 = poly_scalar_mul(&excluded_2, &ap);
        ap = ap.mul(alpha);
        let term_3 = poly_scalar_mul(&excluded_3, &ap);
        ap = ap.mul(alpha);
        let term_4 = poly_scalar_mul(&excluded_4, &ap);
        ap = ap.mul(alpha);
        let term_5 = poly_scalar_mul(&excluded_5, &ap);
        let mut sum = poly_add(&term_0, &term_1, curve);
        sum = poly_add(&sum, &term_2, curve);
        sum = poly_add(&sum, &term_3, curve);
        sum = poly_add(&sum, &term_4, curve);
        poly_add(&sum, &term_5, curve)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // SHA-256 is purely bit-level — `bit_validity` already pins every
        // data column to {0, 1} algebraically. Forward an empty
        // requirements set for now; future tightening could declare an
        // explicit binary range table for redundancy.
        LookupRequirements::none()
    }
}

// ──── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;
    use crate::sha256::{sha256_compress_witness, INITIAL_HASH};

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    /// One compression of INITIAL_HASH on the "abc" padded block; handy
    /// deterministic driver shared with the AIR tests.
    fn abc_rounds() -> Vec<RoundTrace> {
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[63] = 24;
        sha256_compress_witness(INITIAL_HASH, block)
    }

    /// Build a populated single-block column set without padding.
    fn single_block_columns() -> (Vec<Vec<Scalar>>, Vec<RoundTrace>) {
        let curve = CurveType::Bls48581;
        let rounds = abc_rounds();
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }
        // Task #179: populate the W[16] recurrence word aggregator
        // columns + carry at row 16 of the block, so the new
        // `w_recurrence_at_anchor` constraint vanishes on this honest
        // witness. (The `populate_round` path doesn't have cross-row
        // visibility; the populator is normally called by
        // `populate_trace_from_hash`.)
        populate_w_recurrence_for_single_block(&mut columns, &rounds, curve);
        (columns, rounds)
    }

    /// Populate the W[t] recurrence aggregator + carry on every row
    /// t ∈ 16..64 of a single-block column set, from the round-trace W
    /// values directly. Used by tests that build a column set via
    /// `populate_round` rather than the higher-level
    /// `populate_trace_from_hash`. Task #193 generalizes the earlier
    /// anchor-only (row 16) helper to all 48 recurrence rows.
    fn populate_w_recurrence_for_single_block(
        columns: &mut [Vec<Scalar>],
        rounds: &[RoundTrace],
        curve: CurveType,
    ) {
        if rounds.len() <= 16 || columns[0].len() <= 16 {
            return;
        }
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);
        for t in 16..NUM_ROUNDS.min(rounds.len()) {
            if t >= columns[0].len() {
                break;
            }
            let w_tm16 = rounds[t - 16].w as u64;
            let w_tm15 = rounds[t - 15].w;
            let w_tm7  = rounds[t - 7].w as u64;
            let w_tm2  = rounds[t - 2].w;
            let s0 = (w_tm15.rotate_right(7)
                ^ w_tm15.rotate_right(18)
                ^ (w_tm15 >> 3)) as u64;
            let s1 = (w_tm2.rotate_right(17)
                ^ w_tm2.rotate_right(19)
                ^ (w_tm2 >> 10)) as u64;
            let full_sum = s1 + w_tm7 + s0 + w_tm16;
            let carry = full_sum >> 32;
            for bit in 0..sha256_air::W_RECURRENCE_CARRY_BITS {
                let v = (carry >> bit) & 1;
                columns[sha256_air::w_recurrence_carry_bit(bit)][t] =
                    if v == 1 { one.clone() } else { zero.clone() };
            }
            columns[sha256_air::w_recurrence_w0_word()][t]        = Scalar::from_u64(w_tm16, curve);
            columns[sha256_air::w_recurrence_w9_word()][t]        = Scalar::from_u64(w_tm7, curve);
            columns[sha256_air::w_recurrence_sigma0_w1_word()][t] = Scalar::from_u64(s0, curve);
            columns[sha256_air::w_recurrence_sigma1_w14_word()][t] = Scalar::from_u64(s1, curve);
        }
    }

    #[test]
    fn sha256_cs_labels_and_counts() {
        let cs = Sha256ConstraintSystem::new(NUM_ROUNDS);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_constraints(), 25);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), 6);
        assert_eq!(cs.selector_column_indices().len(), NUM_SEL_ROUND);
        // 256 BEFORE bits + 64 INV_INPUT_BYTE + 32 INV_OUTPUT_BYTE
        // + 1 IS_FIRST_BLOCK + 8 STATE_IN_WORD + 1 IS_LAST_BLOCK = 362.
        assert_eq!(
            cs.shifted_column_indices().len(),
            BITS_PER_STATE
                + sha256_air::INV_INPUT_LEN
                + sha256_air::INV_OUTPUT_LEN
                + 1
                + sha256_air::NUM_OUTPUT_WORDS
                + 1
        );
        assert!(cs.padding_selector_column().is_none());
    }

    #[test]
    fn sha256_cs_evaluate_on_domain_matches_air_helper() {
        let (columns, _rounds) = single_block_columns();
        let cs = Sha256ConstraintSystem::new(NUM_ROUNDS);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, NUM_ROUNDS);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, vec) in evals.iter().enumerate() {
            for (row, v) in vec.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) fired at row {} on valid witness",
                    k,
                    cs.constraint_labels()[k],
                    row
                );
            }
        }
    }

    #[test]
    fn sha256_cs_evaluate_at_point_zero_on_real_rows() {
        let (columns, _rounds) = single_block_columns();
        let cs = Sha256ConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(17, CurveType::Bls48581);
        for row in 0..NUM_ROUNDS {
            let col_vals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let c_at_row = cs.evaluate_at_point(&col_vals, &alpha);
            assert!(
                c_at_row.is_zero(),
                "combined constraint C(row {}) nonzero on valid witness",
                row
            );
        }
    }

    #[test]
    fn sha256_cs_evaluate_at_point_zero_on_all_zero_row() {
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let col_vals = vec![zero.clone(); NUM_SHA256_COLUMNS];
        let cs = Sha256ConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(23, curve);
        let c_at = cs.evaluate_at_point(&col_vals, &alpha);
        assert!(
            c_at.is_zero(),
            "all-zero padding row must evaluate to zero (otherwise \
             padding_selector_column must be set)"
        );
    }

    #[test]
    fn sha256_cs_small_sigma_w_categories_present() {
        // The message-schedule σ helper definitions (categories 10 + 11)
        // are listed by label and the count is 24.
        let cs = Sha256ConstraintSystem::new(NUM_ROUNDS);
        let labels = cs.constraint_labels();
        assert!(labels.contains(&"small_sigma0_w_definition".to_string()));
        assert!(labels.contains(&"small_sigma1_w_definition".to_string()));
        // Ordering: small σ helpers slot in right after sel_sum_01.
        let i0 = labels.iter().position(|l| l == "small_sigma0_w_definition").unwrap();
        let i1 = labels.iter().position(|l| l == "small_sigma1_w_definition").unwrap();
        let i_sel = labels.iter().position(|l| l == "sel_sum_01").unwrap();
        assert_eq!(i0, i_sel + 1);
        assert_eq!(i1, i_sel + 2);
    }

    #[test]
    fn sha256_cs_small_sigma_w_tamper_fires_constraint() {
        // Tamper a small_sigma0_w bit and confirm the combined
        // `evaluate_at_point` registers it as non-zero.
        let (mut columns, _rounds) = single_block_columns();
        let curve = CurveType::Bls48581;
        let one = Scalar::one(curve);
        let col = sha256_air::small_sigma0_w_bit(5);
        let old = columns[col][3].clone();
        columns[col][3] = one.sub(&old);

        let cs = Sha256ConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(31, curve);
        // Build the row's column values.
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[3].clone()).collect();
        let c_at = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at.is_zero(),
            "tampered small_sigma0_w bit should fire combined constraint at row 3"
        );

        // Also: per-domain helper should mark only the small_sigma0_w
        // category as non-zero at row 3 (other categories that don't depend
        // on small_sigma0_w bits should remain zero there).
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, NUM_ROUNDS);
        let labels = cs.constraint_labels();
        let i_ss0 = labels.iter()
            .position(|l| l == "small_sigma0_w_definition").unwrap();
        assert!(
            !evals[i_ss0][3].is_zero(),
            "small_sigma0_w_definition must fire at row 3"
        );
    }

    #[test]
    fn sha256_cs_lookup_declarations_are_well_formed() {
        let cs = Sha256ConstraintSystem::new(NUM_ROUNDS);
        let reqs = cs.lookup_declarations();
        assert!(reqs.tables.is_empty());
        assert!(reqs.declarations.is_empty());
    }

    #[test]
    fn sha256_cs_evaluate_at_point_detects_mutation() {
        // Flip a pre-round `a` bit on row 5 — Σ₀, Maj, addition all use
        // it, so the combined constraint must fire there.
        let (mut columns, _rounds) = single_block_columns();
        let curve = CurveType::Bls48581;
        let one = Scalar::one(curve);
        let col = before(VAR_A, 0);
        columns[col][5] = one.sub(&columns[col][5].clone());

        let cs = Sha256ConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[5].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered BEFORE bit should make C(row 5) non-zero"
        );
    }

    /// Tampering test: starting from an aggregator-populated trace
    /// (`populate_trace_from_hash_with_invocation_bytes`), flip an
    /// INV_INPUT_BYTE at row 0 (the binding-fires row for round 0).
    /// The `inv_input_byte_binding` row-local constraint must catch
    /// it: at row 0 the gate `IS_FIRST_BLOCK · SEL_ROUND[0] = 1`, so
    /// the body becomes the (now-tampered) `block_word_0 − w_word_0`,
    /// which is non-zero.
    #[test]
    fn sha256_cs_inv_input_byte_binding_rejects_tampered_anchor_byte() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (0x10u8).wrapping_add(i as u8);
        }
        let ht = sha256_witness(&input);
        let mut columns = sha256_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, curve,
        );
        // Tamper INV_INPUT_BYTE[0] at row 0 (one byte of W[0]).
        let one = Scalar::one(curve);
        let col = sha256_air::inv_input_byte(0);
        columns[col][0] = columns[col][0].add(&one);

        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let alpha = Scalar::from_u64(11, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered INV_INPUT_BYTE at the binding-fires row must make \
             C(row 0) non-zero"
        );
    }

    /// Tampering test: output binding catches a flipped INV_OUTPUT_BYTE
    /// at the binding row (last round of the LAST block — row 127 for
    /// sha256_pair-shaped 2-block traces). The body
    /// `digest_word + binding_carry · 2^32 − STATE_IN_WORD − AFTER_word`
    /// becomes non-zero when INV_OUTPUT_BYTE differs from the actual
    /// digest.
    #[test]
    fn sha256_cs_inv_output_byte_binding_rejects_tampered_digest_byte() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (0x10u8).wrapping_add(i as u8);
        }
        let ht = sha256_witness(&input);
        assert_eq!(ht.blocks.len(), 2, "64-byte input must produce 2-block trace");
        let mut columns = sha256_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, curve,
        );
        // Tamper INV_OUTPUT_BYTE[0] at the binding row (row 127 = last
        // round of last block).
        let one = Scalar::one(curve);
        let last_row = 2 * NUM_ROUNDS - 1; // row 127
        let col = sha256_air::inv_output_byte(0);
        columns[col][last_row] = columns[col][last_row].add(&one);

        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let alpha = Scalar::from_u64(37, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[last_row].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered INV_OUTPUT_BYTE at row 127 must make the output binding fire"
        );
    }

    /// Tampering test: cross-block STATE_IN chaining catches a wrong
    /// state_in[block 1] value. STATE_IN_WORD should equal INITIAL_HASH
    /// + AFTER_word[63 of block 0] mod 2^32 at row 64. Tampering it
    /// breaks the cross-block chaining shifted constraint (body 4)
    /// at the row 63→64 transition.
    #[test]
    fn sha256_cs_state_in_chaining_rejects_wrong_state_at_block_1() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (0x20u8).wrapping_add(i as u8);
        }
        let ht = sha256_witness(&input);
        assert_eq!(ht.blocks.len(), 2);
        let mut columns = sha256_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, curve,
        );
        // Tamper STATE_IN_WORD[0] at row 64 (start of block 1).
        // Honest value = INITIAL_HASH[0] + AFTER_word[63 of block 0][0].
        // Replace with INITIAL_HASH[0] (drop the AFTER contribution).
        let init = Scalar::from_u64(INITIAL_HASH[0] as u64, curve);
        columns[sha256_air::state_in_word(0)][NUM_ROUNDS] = init.clone();
        // Also tamper subsequent rows of STATE_IN_WORD[0] to match
        // (otherwise within-block invariance fires first); witness
        // honesty is broken here so we just propagate.
        for r in NUM_ROUNDS + 1..2 * NUM_ROUNDS {
            columns[sha256_air::state_in_word(0)][r] = init.clone();
        }

        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let alpha = Scalar::from_u64(41, curve);
        // Body 4 is shifted: read at row 63 and shifted_evals at row
        // 64. Compute via the shifted evaluator.
        let row_63: usize = NUM_ROUNDS - 1;
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[row_63].clone()).collect();
        let mut shifted_evals = Vec::new();
        for &i in &cs.shifted_column_indices() {
            shifted_evals.push(columns[i][row_63 + 1].clone());
        }
        let z = Scalar::from_u64(149, curve);
        let padded =
            crate::trace::nearest_power_of_two((ht.blocks.len() * NUM_ROUNDS).max(1)) as u64;
        let scheme = Bls48581Scheme::new();
        crate::scheme::CommitmentScheme::init(&scheme);
        let omega =
            crate::scheme::CommitmentScheme::domain_generator(&scheme, padded);
        let omega_n_minus_1 = scalar_pow(&omega, padded - 1);
        let body_at_z = cs.evaluate_shifted_at_point(
            &row_vals,
            &shifted_evals,
            &z,
            &omega_n_minus_1,
            &alpha,
            0,
        );
        assert!(
            !body_at_z.is_zero(),
            "wrong STATE_IN_WORD at block 1 must make body 4 (cross-block chaining) non-zero"
        );
    }

    /// Tampering test: an aggregator-populated trace with
    /// `IS_FIRST_BLOCK = 1` only at row 0 (witness-suspicious — should
    /// be 1 on rows 0..NUM_ROUNDS-1). The within-block invariance
    /// shifted constraint at row 0→1 fires:
    /// `(1 − SEL_ROUND[NUM_ROUNDS-1](row 0)) · (IS_FIRST_BLOCK(row 1) −
    /// IS_FIRST_BLOCK(row 0)) = 1 · (0 − 1) = −1 ≠ 0`.
    #[test]
    fn sha256_cs_is_first_block_invariance_rejects_premature_drop() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let input = [0u8; 64];
        let ht = sha256_witness(&input);
        let mut columns = sha256_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, curve,
        );
        // Tamper: drop IS_FIRST_BLOCK to 0 at row 1 (but it's 1 at row
        // 0). Cross-row invariance constraint must fire at the row 0→1
        // transition.
        columns[sha256_air::COL_IS_FIRST_BLOCK][1] = Scalar::zero(curve);

        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let alpha = Scalar::from_u64(13, curve);
        // Body 2 of evaluate_shifted_at_point uses col_evals_at_z (row r)
        // and shifted_evals (row r+1). At r=0: SEL_ROUND[NUM_ROUNDS-1]
        // = 0, IS_FIRST_BLOCK_curr = 1, IS_FIRST_BLOCK_next = 0.
        // body_2 = (1 − 0) · (0 − 1) = −1.
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let mut shifted_evals = Vec::new();
        for &i in &cs.shifted_column_indices() {
            shifted_evals.push(columns[i][1].clone());
        }
        let z = Scalar::from_u64(101, curve); // arbitrary point
        // Build omega^(n-1) for a domain of size = padded power of 2.
        let padded =
            crate::trace::nearest_power_of_two((ht.blocks.len() * NUM_ROUNDS).max(1)) as u64;
        let scheme = crate::scheme::bls48581_scheme::Bls48581Scheme::new();
        crate::scheme::CommitmentScheme::init(&scheme);
        let omega = crate::scheme::CommitmentScheme::domain_generator(&scheme, padded);
        let omega_n_minus_1 = scalar_pow(&omega, padded - 1);

        let body_at_z =
            cs.evaluate_shifted_at_point(&row_vals, &shifted_evals, &z, &omega_n_minus_1, &alpha, 0);
        assert!(
            !body_at_z.is_zero(),
            "tampered IS_FIRST_BLOCK premature drop must make body 2 non-zero"
        );
    }

    /// Task #179: tampering test for the new `w_recurrence_at_anchor`
    /// constraint. Honest witness has the row-16 W word equal to the
    /// sum of the four committed aggregator values + carry · 2^32.
    /// Flip a bit of W at row 16 → the row-16 W word changes by ±2^bit
    /// while the committed aggregators stay put → the constraint
    /// (gated by SEL_ROUND[16] = 1) fires.
    #[test]
    fn sha256_cs_w_recurrence_rejects_tampered_w16() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (0x42u8).wrapping_add(i as u8);
        }
        let ht = sha256_witness(&input);
        let mut columns = sha256_air::populate_trace_from_hash(&ht, curve);
        // Tamper W[bit 0] at the per-block anchor row 16.
        let one = Scalar::one(curve);
        let col = sha256_air::w_bit(0);
        let old = columns[col][16].clone();
        columns[col][16] = one.sub(&old);

        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, ht.blocks.len() * NUM_ROUNDS);
        let labels = cs.constraint_labels();
        let i_wrec = labels.iter()
            .position(|l| l == "w_recurrence")
            .expect("w_recurrence category exists");
        assert!(
            !evals[i_wrec][16].is_zero(),
            "flipped W bit at anchor row 16 must fire w_recurrence"
        );
    }

    /// Task #179: tampering test for the aggregator addend. Flipping the
    /// committed σ1(W[14]) word at the anchor row breaks the recurrence
    /// sum without changing W[16], so the constraint must fire.
    #[test]
    fn sha256_cs_w_recurrence_rejects_tampered_sigma1_w14() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let input = [0xCDu8; 32];
        let ht = sha256_witness(&input);
        let mut columns = sha256_air::populate_trace_from_hash(&ht, curve);
        // Tamper σ1(W[14]) aggregator at the per-block anchor row 16.
        let col = sha256_air::w_recurrence_sigma1_w14_word();
        let one = Scalar::one(curve);
        columns[col][16] = columns[col][16].add(&one);

        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, ht.blocks.len() * NUM_ROUNDS);
        let labels = cs.constraint_labels();
        let i_wrec = labels.iter()
            .position(|l| l == "w_recurrence").unwrap();
        assert!(
            !evals[i_wrec][16].is_zero(),
            "tampered σ1(W[14]) aggregator must fire w_recurrence"
        );
    }

    /// Task #179 + #193: the constraint must NOT fire on non-recurrence
    /// rows (t ∈ 0..16 within each block) — the gate
    /// `Σ_{t=16..64} SEL_ROUND[t]` is zero there, so the aggregator
    /// columns being zero is consistent with vanishing.
    #[test]
    fn sha256_cs_w_recurrence_vacuous_on_non_anchor_rows() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let input = [0u8; 32];
        let ht = sha256_witness(&input);
        let columns = sha256_air::populate_trace_from_hash(&ht, curve);
        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, ht.blocks.len() * NUM_ROUNDS);
        let labels = cs.constraint_labels();
        let i_wrec = labels.iter()
            .position(|l| l == "w_recurrence").unwrap();
        let num_rows = columns[0].len();
        for row in 0..num_rows {
            assert!(
                evals[i_wrec][row].is_zero(),
                "w_recurrence fired at row {} on honest witness",
                row
            );
        }
    }

    /// Task #193: the honest recurrence vanishes for every t ∈ 16..64 of
    /// every block, demonstrating the constraint now fires on all 48
    /// recurrence rows per block (not just the row-16 anchor).
    #[test]
    fn sha256_cs_w_recurrence_honest_vanishes_at_all_t() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let input = [0x5Au8; 56]; // 2 blocks
        let ht = sha256_witness(&input);
        assert_eq!(ht.blocks.len(), 2);
        let columns = sha256_air::populate_trace_from_hash(&ht, curve);
        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, ht.blocks.len() * NUM_ROUNDS);
        let labels = cs.constraint_labels();
        let i_wrec = labels.iter()
            .position(|l| l == "w_recurrence").unwrap();
        let num_rows = columns[0].len();
        for row in 0..num_rows {
            assert!(
                evals[i_wrec][row].is_zero(),
                "w_recurrence fired at row {} on honest witness", row
            );
        }
        // Sanity-check that the gate is actually `1` on rows 16..63 of
        // each block (otherwise vanishing would be trivial / vacuous).
        for block_idx in 0..ht.blocks.len() {
            for t in 16..NUM_ROUNDS {
                let row = block_idx * NUM_ROUNDS + t;
                let sel = &columns[sel_round(t)][row];
                assert!(
                    !sel.is_zero(),
                    "SEL_ROUND[{}] should be 1 at row {}", t, row
                );
            }
        }
    }

    /// Task #193: tampering test for t=32 (a non-anchor recurrence row).
    /// Flip a bit of W at row 32 of block 0 → the row's W word changes
    /// while the committed aggregator addends stay put → the constraint
    /// (gated by SEL_ROUND[32] = 1) fires on row 32.
    #[test]
    fn sha256_cs_w_recurrence_rejects_tampered_w_at_t32() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (0x17u8).wrapping_add(i as u8);
        }
        let ht = sha256_witness(&input);
        let mut columns = sha256_air::populate_trace_from_hash(&ht, curve);
        let one = Scalar::one(curve);
        // Tamper W[bit 0] at row 32.
        let col = sha256_air::w_bit(0);
        let old = columns[col][32].clone();
        columns[col][32] = one.sub(&old);

        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, ht.blocks.len() * NUM_ROUNDS);
        let labels = cs.constraint_labels();
        let i_wrec = labels.iter()
            .position(|l| l == "w_recurrence").unwrap();
        assert!(
            !evals[i_wrec][32].is_zero(),
            "flipped W bit at recurrence row 32 must fire w_recurrence"
        );
    }

    /// Task #193: tampering test for t=63 (the final recurrence row of
    /// a block). Flip the σ1(W[t-2]) aggregator addend at row 63 → the
    /// recurrence sum no longer matches W[63] → the constraint fires.
    #[test]
    fn sha256_cs_w_recurrence_rejects_tampered_aggregator_at_t63() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let input = [0xA5u8; 32];
        let ht = sha256_witness(&input);
        let mut columns = sha256_air::populate_trace_from_hash(&ht, curve);
        let one = Scalar::one(curve);
        // Tamper the σ1(W[t-2]) aggregator at row 63.
        let col = sha256_air::w_recurrence_sigma1_w14_word();
        columns[col][63] = columns[col][63].add(&one);

        let cs = Sha256ConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, ht.blocks.len() * NUM_ROUNDS);
        let labels = cs.constraint_labels();
        let i_wrec = labels.iter()
            .position(|l| l == "w_recurrence").unwrap();
        assert!(
            !evals[i_wrec][63].is_zero(),
            "tampered σ1 aggregator at row 63 must fire w_recurrence"
        );
    }

    #[test]
    fn sha256_boundary_rows_single_block() {
        // num_rows=64, domain_size=64 → boundaries {63}.
        let br = boundary_rows(NUM_ROUNDS, NUM_ROUNDS);
        assert_eq!(br, vec![63]);
    }

    #[test]
    fn sha256_boundary_rows_two_blocks_padded() {
        // num_rows=128, domain_size=128 → boundaries {63, 127}.
        let br = boundary_rows(2 * NUM_ROUNDS, 2 * NUM_ROUNDS);
        assert_eq!(br, vec![63, 127]);
    }

    /// Marked `#[ignore]`: full prove/verify roundtrip on a SHA-256 trace
    /// is expensive (committing ~1,030 columns under BLS48-581 KZG + the
    /// row-local schoolbook polynomial builds). Run manually with:
    ///     cargo test --release -p metavm-zkp --lib \
    ///         sha256_cs_prove_verify_small -- --ignored --nocapture
    #[test]
    #[ignore = "slow: full prover roundtrip; run with --release --ignored"]
    fn sha256_cs_prove_verify_small() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let rounds = abc_rounds();
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = Sha256ConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "sha256 single-block proof must verify");
    }

    /// 2-block prove/verify with FULL aggregator + IS_FIRST_BLOCK
    /// population. Exercises:
    ///   - The `inv_input_byte_binding` row-local constraint (only fires
    ///     on rows where `IS_FIRST_BLOCK · SEL_ROUND[r] = 1`).
    ///   - The `is_first_block_binary` and `is_first_block_pinned_at_anchor`
    ///     row-local constraints.
    ///   - The IS_FIRST_BLOCK invariance shifted constraint (body 2).
    ///   - The aggregator invariance shifted constraint (body 1).
    /// Plus everything from the legacy 2-block test (body 0 multi-block).
    #[test]
    #[ignore = "slow: full prover roundtrip on 2-block trace with aggregator; run with --release --ignored"]
    fn sha256_cs_prove_verify_two_blocks_with_aggregator() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 64-byte input → 2-block trace (sha256_pair shape).
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (0x10u8).wrapping_add(i as u8);
        }
        let ht = sha256_witness(&input);
        assert_eq!(ht.blocks.len(), 2);

        // Use the aggregator-populating builder. IS_FIRST_BLOCK is set
        // = 1 on rows 0..63 and 0 on rows 64..127.
        let num_rows = ht.blocks.len() * NUM_ROUNDS;
        let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
        let mut columns = sha256_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, curve,
        );
        for col in columns.iter_mut() {
            if col.len() < padded {
                col.resize(padded, Scalar::zero(curve));
            }
        }
        let trace = into_trace_polynomials(columns, num_rows, padded, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = Sha256ConstraintSystem::new(num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(
            valid,
            "sha256 two-block aggregator-populated proof must verify (\
             exercises algebraic input binding + IS_FIRST_BLOCK + aggregator \
             invariance constraints)"
        );
    }

    /// 2-block prove/verify regression. Validates that body 0's
    /// boundary-row exclusion correctly excludes the block-internal
    /// boundary at row 63 in addition to the last-real (row 127) and
    /// domain wrap. Required for sha256_pair (always 2 blocks: input ||
    /// padding) — without the verifier-side fix to use the full
    /// `boundary_rows`, the constraint polynomial would not match the
    /// scalar evaluation at z and the proof would fail.
    #[test]
    #[ignore = "slow: full prover roundtrip on 2-block trace; run with --release --ignored"]
    fn sha256_cs_prove_verify_two_blocks() {
        use crate::sha256::sha256_witness;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 64-byte input forces SHA-256 padding into a separate (second)
        // block — exactly the sha256_pair shape.
        let mut input = [0u8; 64];
        for (i, b) in input.iter_mut().enumerate() {
            *b = (0x10u8).wrapping_add(i as u8);
        }
        let ht = sha256_witness(&input);
        assert_eq!(ht.blocks.len(), 2, "64-byte input must produce 2-block trace");

        let trace = build_trace_polynomials_from_hash(&ht, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let num_rows = ht.blocks.len() * NUM_ROUNDS;
        let cs = Sha256ConstraintSystem::new(num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "sha256 two-block proof must verify");
    }

    /// Cross-AIR LogUp joint prover/verifier end-to-end smoke test.
    ///
    /// Drives [`crate::cross_air_logup::joint_prove`] +
    /// [`crate::cross_air_logup::joint_verify`] on two identical SHA-256
    /// traces with a self-linkage (column 0 of A multiset-equal to
    /// column 0 of B). Confirms:
    /// - β/γ derivation is deterministic across prover and verifier
    /// - per-AIR proofs round-trip
    /// - linkage closure scalars match (`closure_a == closure_b`)
    /// - tampering with β, γ, or closure scalars is rejected
    ///
    /// Per-AIR phase-2 does not yet consume γ, so the per-AIR proofs
    /// here remain byte-equivalent to single-AIR proofs. This test
    /// exercises the orchestration scaffold + closure-equality check.
    #[test]
    #[ignore = "slow: two full SHA-256 prover roundtrips; run with --release --ignored"]
    fn joint_prove_two_sha256_traces() {
        use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let rounds = abc_rounds();
        let trace_a = build_trace_polynomials_from_rounds(&rounds, curve);
        let trace_b = build_trace_polynomials_from_rounds(&rounds, curve);
        let domain_size = trace_a.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs_a = Sha256ConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega.clone(), domain_size);
        let cs_b = Sha256ConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega, domain_size);

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn crate::vm_constraints::VmConstraintSystem)> =
            vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![CrossAirLogUpDescriptor {
            label: "sha256_self_smoke_v1".into(),
            a_layer_index: 0,
            a_columns: vec![0],
            a_selector_column: None,
            b_layer_index: 1,
            b_columns: vec![0],
            b_selector_column: None,
        }];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for identical traces");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.gamma_bytes.len(), 32);
        assert_eq!(extension.beta_bytes.len(), 32);
        assert_eq!(extension.linkage_proofs.len(), 1);
        assert_eq!(extension.linkage_proofs[0].label, "sha256_self_smoke_v1");
        assert_eq!(
            extension.linkage_proofs[0].closure_a,
            extension.linkage_proofs[0].closure_b,
            "honest closure scalars must match"
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> = vec![&cs_a, &cs_b];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(valid, "joint verifier must accept honest joint proof");

        // Tampering with γ must be rejected.
        let mut bad = extension.clone();
        bad.gamma_bytes[0] ^= 0x01;
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &bad, &scheme, curve),
            "joint verifier must reject tampered γ"
        );

        // Tampering with β must be rejected.
        let mut bad = extension.clone();
        bad.beta_bytes[0] ^= 0x01;
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &bad, &scheme, curve),
            "joint verifier must reject tampered β"
        );

        // Tampering with closure_a (breaking closure equality) must be rejected.
        let mut bad = extension.clone();
        bad.linkage_proofs[0].closure_a[0] ^= 0x01;
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &bad, &scheme, curve),
            "joint verifier must reject mismatched closure scalars"
        );
    }

    /// Cross-AIR LogUp `joint_prove`/`joint_verify` end-to-end test for
    /// Sha256Extract ↔ SHA-256 bit-level. Mirrors the
    /// `joint_prove_keccak_extract_keccak_linkage` test (which closes
    /// MPT ↔ Keccak end-to-end). Validates that the entire chain —
    /// from Sha256Extract's claimed `(left||right, sha256(left||right))`
    /// tuple all the way down to the bit-level SHA-256 algebraic
    /// input + output binding via STATE_IN chaining — is
    /// cryptographically sound.
    ///
    /// Note: relies on `joint_prove`'s auto-inflation of per-AIR
    /// traces to a common domain (Sha256Extract padded_size=16,
    /// SHA-256 padded_size=128 for sha256_pair).
    #[test]
    #[ignore = "slow: SHA-256 bit-level + Sha256Extract joint prove (~10 min); run with --release --ignored"]
    fn joint_prove_sha256_extract_sha256_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::sha256::sha256_witness;
        use crate::sha256_extract::{
            build_trace_polynomials as build_extract_trace,
            make_sha256_extract_sha256_linkage_descriptor, Sha256ExtractConstraintSystem,
            Sha256ExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let mut left = [0u8; 32];
        let mut right = [0u8; 32];
        for i in 0..32 {
            left[i] = (0x10u8).wrapping_add(i as u8);
            right[i] = (0xa0u8).wrapping_add(i as u8);
        }
        let mut input64 = [0u8; 64];
        input64[..32].copy_from_slice(&left);
        input64[32..].copy_from_slice(&right);
        let ht = sha256_witness(&input64);
        assert_eq!(ht.blocks.len(), 2, "64-byte sha256_pair input must produce 2-block trace");

        // ── Sha256Extract trace (A side) ──
        let extract_w = Sha256ExtractWitness::from_pair_inputs(&[(left, right)]);
        let extract_trace = build_extract_trace(&extract_w, curve);
        let extract_domain = extract_trace.padded_size;
        let extract_omega = scheme.domain_generator(extract_domain);
        let extract_cs = Sha256ExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_domain);

        // ── SHA-256 bit-level trace (B side, with full aggregator + binding) ──
        let num_rows = ht.blocks.len() * NUM_ROUNDS;
        let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
        let mut sha256_columns =
            sha256_air::populate_trace_from_hash_with_invocation_bytes(&ht, &input64, curve);
        for col in sha256_columns.iter_mut() {
            if col.len() < padded {
                col.resize(padded, Scalar::zero(curve));
            }
        }
        let sha256_trace = into_trace_polynomials(sha256_columns, num_rows, padded, curve);
        let sha256_domain = sha256_trace.padded_size;
        let sha256_omega = scheme.domain_generator(sha256_domain);
        let sha256_cs = Sha256ConstraintSystem::new(num_rows)
            .with_omega_and_domain(sha256_omega, sha256_domain);

        let linkage = make_sha256_extract_sha256_linkage_descriptor(0, 1);
        assert_eq!(linkage.label, "sha256_extract_sha256_v1");

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&extract_trace, &extract_cs), (&sha256_trace, &sha256_cs)];
        let linkages = vec![linkage];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for matched extract + bit-level traces");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        assert_eq!(
            extension.linkage_proofs[0].closure_a,
            extension.linkage_proofs[0].closure_b,
            "honest closure scalars must match"
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&extract_cs, &sha256_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the Sha256Extract↔SHA-256 honest joint proof"
        );
    }

    /// 3-AIR end-to-end SSZ tree-hashing chain: SSZ AIR + Sha256Extract
    /// + bit-level SHA-256. Mirrors `joint_prove_create2_input_chain_e2e`
    /// (CREATE2 chain with bit-level Keccak) for the SSZ side. Validates
    /// the full SSZ pair-hashing soundness chain end-to-end through
    /// algebraic SHA-256 constraints.
    ///
    /// Setup: 2-leaf merkleize → 1 hash row in SSZ AIR. The hash
    /// invocation `sha256(left || right)` is matched across all three
    /// AIRs via two cross-AIR LogUp linkages:
    ///   - L1: SSZ↔Sha256Extract (96-byte tuple = LEFT||RIGHT||PARENT
    ///     ↔ INPUT_BYTE[0..64]||OUTPUT_BYTE[0..32])
    ///   - L2: Sha256Extract↔bit-level SHA-256 (96-byte tuple via
    ///     INV_INPUT_BYTE / INV_OUTPUT_BYTE aggregator)
    ///
    /// Combined with the bit-level SHA-256's algebraic input/output
    /// bindings (closed via #87), this proves: for the 2-leaf SSZ
    /// trace, the parent column is the actual `sha256(left||right)`
    /// digest of the leaves — algebraically pinned by every layer of
    /// the chain.
    #[test]
    #[ignore = "slow: 3-AIR + 2-linkage joint_prove with bit-level SHA-256 \
                (~3 min, dominated by SHA-256 prover); run with --release --ignored"]
    fn joint_prove_ssz_chain_e2e() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::sha256::sha256_witness;
        use crate::sha256_extract::{
            build_trace_polynomials as build_extract_trace,
            make_sha256_extract_sha256_linkage_descriptor,
            make_ssz_sha256_extract_linkage_descriptor,
            Sha256ExtractConstraintSystem, Sha256ExtractWitness,
        };
        use crate::ssz::ZERO_CHUNK;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows, SszConstraintSystem,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 2-leaf merkleize: 1 hash row.
        let mut left = ZERO_CHUNK;
        let mut right = ZERO_CHUNK;
        for i in 0..32 {
            left[i] = (0x10u8).wrapping_add(i as u8);
            right[i] = (0xa0u8).wrapping_add(i as u8);
        }
        let rows = merkleize_witness(&[left, right], None);
        assert_eq!(rows.len(), 1, "2-leaf merkleize must produce 1 hash row");

        // ── SSZ trace ──
        let ssz_trace = build_trace_polynomials_from_rows(&rows, curve);
        let ssz_cs = SszConstraintSystem::new(rows.len());

        // ── Sha256Extract trace (one invocation) ──
        let extract_w = Sha256ExtractWitness::from_pair_inputs(&[(left, right)]);
        let extract_trace = build_extract_trace(&extract_w, curve);
        let extract_cs = Sha256ExtractConstraintSystem::new(extract_trace.num_rows);

        // ── Bit-level SHA-256 trace (one sha256_pair invocation = 2 blocks) ──
        let mut input64 = [0u8; 64];
        input64[..32].copy_from_slice(&left);
        input64[32..].copy_from_slice(&right);
        let ht = sha256_witness(&input64);
        let num_rows = ht.blocks.len() * NUM_ROUNDS;
        let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
        let mut sha256_columns =
            sha256_air::populate_trace_from_hash_with_invocation_bytes(&ht, &input64, curve);
        for col in sha256_columns.iter_mut() {
            if col.len() < padded {
                col.resize(padded, Scalar::zero(curve));
            }
        }
        let sha256_trace = into_trace_polynomials(sha256_columns, num_rows, padded, curve);
        let sha256_cs = Sha256ConstraintSystem::new(num_rows);

        // ── Linkages ──
        let l1 = make_ssz_sha256_extract_linkage_descriptor(/* ssz */ 0, /* extract */ 1);
        let l2 = make_sha256_extract_sha256_linkage_descriptor(/* extract */ 1, /* sha256 */ 2);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&ssz_trace, &ssz_cs),
            (&extract_trace, &extract_cs),
            (&sha256_trace, &sha256_cs),
        ];
        let linkages = vec![l1, l2];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the 3-AIR SSZ chain");
        assert_eq!(proofs.len(), 3);
        assert_eq!(extension.linkage_proofs.len(), 2);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&ssz_cs, &extract_cs, &sha256_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(valid, "joint verifier must accept the 3-AIR SSZ chain");
    }

    /// Tampering test for the SSZ chain L1 linkage (SSZ↔Sha256Extract):
    /// build the SSZ trace honestly but build Sha256Extract with a
    /// DIFFERENT (left, right) pair so the (LEFT||RIGHT, PARENT) tuples
    /// don't match. The L1 multiset MUST fail since SSZ's tuple no
    /// longer appears in Sha256Extract's table.
    ///
    /// Validates that a malicious SSZ prover cannot claim a wrong
    /// `parent` (= sha256(left, right)) without breaking the
    /// cross-AIR LogUp.
    #[test]
    fn joint_prove_ssz_chain_rejects_tampered_pair() {
        use crate::cross_air_logup::joint_prove;
        use crate::sha256_extract::{
            build_trace_polynomials as build_extract_trace,
            make_ssz_sha256_extract_linkage_descriptor,
            Sha256ExtractConstraintSystem, Sha256ExtractWitness,
        };
        use crate::ssz::ZERO_CHUNK;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows, SszConstraintSystem,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // SSZ trace: honest 2-leaf merkleize.
        let mut left = ZERO_CHUNK;
        let mut right = ZERO_CHUNK;
        for i in 0..32 {
            left[i] = (0x10u8).wrapping_add(i as u8);
            right[i] = (0xa0u8).wrapping_add(i as u8);
        }
        let rows = merkleize_witness(&[left, right], None);
        let ssz_trace = build_trace_polynomials_from_rows(&rows, curve);
        let ssz_cs = SszConstraintSystem::new(rows.len());

        // Sha256Extract: TAMPERED — swap left and right so the
        // (LEFT||RIGHT, PARENT) tuple differs from SSZ's row.
        let extract_w = Sha256ExtractWitness::from_pair_inputs(&[(right, left)]);
        let extract_trace = build_extract_trace(&extract_w, curve);
        let extract_cs = Sha256ExtractConstraintSystem::new(extract_trace.num_rows);

        let l1 = make_ssz_sha256_extract_linkage_descriptor(0, 1);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&ssz_trace, &ssz_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![l1];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(result.is_err(), "joint_prove must reject tampered Sha256Extract");
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}", err
        );
    }

    /// Multi-column tuple end-to-end: same shape as
    /// `joint_prove_two_sha256_traces` but with a 3-column tuple
    /// `(col_0, col_1, col_2)` instead of a single column. Validates
    /// that the β-RLC encoding flows correctly through the full
    /// joint_prove/joint_verify pipeline including the per-linkage
    /// SNARK, the multi-column cross-trace openings (one per source
    /// column), and the verifier's β-RLC reconstruction.
    #[test]
    #[ignore = "slow: two full SHA-256 prover roundtrips; run with --release --ignored"]
    fn joint_prove_two_sha256_traces_multi_column_tuple() {
        use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let rounds = abc_rounds();
        let trace_a = build_trace_polynomials_from_rounds(&rounds, curve);
        let trace_b = build_trace_polynomials_from_rounds(&rounds, curve);
        let domain_size = trace_a.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs_a = Sha256ConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega.clone(), domain_size);
        let cs_b = Sha256ConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega, domain_size);

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn crate::vm_constraints::VmConstraintSystem)> =
            vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![CrossAirLogUpDescriptor {
            label: "sha256_multi_col_v1".into(),
            a_layer_index: 0,
            a_columns: vec![0, 1, 2],
            a_selector_column: None,
            b_layer_index: 1,
            b_columns: vec![0, 1, 2],
            b_selector_column: None,
        }];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for identical traces with multi-column tuples");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        let lp = &extension.linkage_proofs[0];
        assert_eq!(lp.cross_trace_a.len(), 3, "one opening per source column on A side");
        assert_eq!(lp.cross_trace_b.len(), 3, "one opening per source column on B side");
        assert_eq!(lp.cross_trace_a[0].column_index, 0);
        assert_eq!(lp.cross_trace_a[1].column_index, 1);
        assert_eq!(lp.cross_trace_a[2].column_index, 2);

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> = vec![&cs_a, &cs_b];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept honest multi-column-tuple joint proof"
        );

        // Tampering with one of the cross-trace openings must be rejected.
        let mut bad = extension.clone();
        bad.linkage_proofs[0].cross_trace_a[1].eval_bytes[0] ^= 0x01;
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &bad, &scheme, curve),
            "joint verifier must reject tampered cross-trace opening"
        );
    }

    /// Cross-AIR LogUp negative test: a linkage whose A side has a tuple
    /// not present in B's table must fail at `joint_prove` time rather
    /// than producing a bogus proof.
    #[test]
    #[ignore = "slow: one full SHA-256 prover roundtrip; run with --release --ignored"]
    fn joint_prove_rejects_unmatched_linkage() {
        use crate::cross_air_logup::{joint_prove, CrossAirLogUpDescriptor};
        use crate::trace::Polynomial;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let rounds = abc_rounds();
        let trace_a = build_trace_polynomials_from_rounds(&rounds, curve);
        let mut trace_b = build_trace_polynomials_from_rounds(&rounds, curve);
        // Mutate B's column 0 row 0 so A's tuple no longer appears in B.
        // (This produces an invalid trace for SHA-256, so per-AIR proving
        // would fail too; we expect joint_prove to fail FIRST during
        // witness building — before phase-2 is even attempted.)
        trace_b.columns[0] = Polynomial::from_u64_vec_with_curve(
            &vec![0xdeadbeef; trace_a.columns[0].evaluations.len()],
            curve,
        );

        let domain_size = trace_a.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs_a = Sha256ConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega.clone(), domain_size);
        let cs_b = Sha256ConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega, domain_size);

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn crate::vm_constraints::VmConstraintSystem)> =
            vec![(&trace_a, &cs_a), (&trace_b, &cs_b)];
        let linkages = vec![CrossAirLogUpDescriptor {
            label: "sha256_unmatched_v1".into(),
            a_layer_index: 0,
            a_columns: vec![0],
            a_selector_column: None,
            b_layer_index: 1,
            b_columns: vec![0],
            b_selector_column: None,
        }];

        let r = joint_prove(&traces, &linkages, &scheme);
        assert!(
            r.is_err(),
            "joint_prove must fail when A's tuples are not a sub-multiset of B's"
        );
    }

    /// End-to-end: produce a real SHA-256 ExecutionProof, serialize it,
    /// stuff into a `LayerChainProof`, and verify via the chain's
    /// closure-based per-layer verifier. This is the first time a real
    /// cryptographic proof flows through the recursive-fold envelope.
    ///
    /// Demonstrates the full pipeline:
    ///   1. `prove_with_scheme` produces an `ExecutionProof`
    ///   2. `ExecutionProof::to_bytes` serializes it into proof bytes
    ///   3. `LayerProof::with_proof(claim, Sha256, bytes)` packages it
    ///   4. `LayerChainProof::verify_with_layer_verifier` dispatches via
    ///      a closure that calls `from_bytes` + `verify_with_scheme`
    #[test]
    #[ignore = "slow: produces a real SHA-256 proof; run with --release --ignored"]
    fn sha256_proof_flows_through_layer_chain_envelope() {
        use crate::layer_chain::{
            ChainBoundaries, LayerChainProof, LayerProof, LayerProofKind,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 1. Generate a real SHA-256 proof on the "abc" block.
        let rounds = abc_rounds();
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = Sha256ConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega.clone(), domain_size);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);

        // Sanity: the proof verifies in the direct path.
        assert!(crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve));

        // 2. Serialize.
        let proof_bytes = proof.to_bytes();
        assert!(!proof_bytes.is_empty(), "real proof must produce bytes");

        // 3. Build a small LayerChainProof. Use the BlockBinding layer
        //    (the natural slot for SHA-256 — beacon SSZ payload root).
        //    Boundary values are synthetic but consistent.
        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: [0xDD; 32],
            num_attesters: 1,
            finalized_root: [0xCC; 32],
            total_effective_balance_gwei: 32_000_000_000,
        };
        let chain = crate::layer_chain::LayerChain::from_boundaries(&boundaries);
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| {
                if i == 1 {
                    // BlockBinding layer carries the real SHA-256 proof.
                    LayerProof::with_proof(claim, LayerProofKind::Sha256, proof_bytes.clone())
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        // 4. Verify through the closure-based dispatch.
        let result = chain_proof.verify_with_layer_verifier(|layer| {
            match layer.kind {
                LayerProofKind::Sha256 => {
                    let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                        .map_err(|e| format!("decode failed: {:?}", e))?;
                    let cs = Sha256ConstraintSystem::new(rounds.len())
                        .with_omega_and_domain(omega.clone(), domain_size);
                    let valid = crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve);
                    if valid {
                        Ok(())
                    } else {
                        Err("Sha256 proof did not verify".to_string())
                    }
                }
                LayerProofKind::ReferenceOnly => Ok(()),
                other => Err(format!("unsupported layer kind {}", other.as_str())),
            }
        });
        assert_eq!(
            result,
            Ok(()),
            "real SHA-256 proof must verify through the LayerChainProof envelope",
        );
    }
}
