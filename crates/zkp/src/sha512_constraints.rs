//! [`VmConstraintSystem`] wiring for the SHA-512 bit-level AIR (task #258).
//!
//! Mirrors [`crate::sha256_constraints`] structurally, scaled to 64-bit
//! words. The 64-bit width is what differentiates SHA-512 from SHA-256:
//! every working-variable column block is 64 bits wide (instead of 32),
//! the round-addition carries are spread over `2^64` (instead of `2^32`),
//! and σ/Σ helper rotations use the SHA-512 amounts (28/34/39, 14/18/41,
//! 1/8/SHR7, 19/61/SHR6).
//!
//! # Scope (#258 time-box)
//!
//! This module wires a **row-local** constraint system for one SHA-512
//! round. It covers:
//!
//!   0.  `bit_validity`       — every data column satisfies `b·(b−1)=0`.
//!   1.  `big_sigma0_a`       — Σ₀(a) two-stage XOR over rot28/rot34/rot39.
//!   2.  `big_sigma1_e`       — Σ₁(e) two-stage XOR over rot14/rot18/rot41.
//!   3.  `ch_definition`      — ef, not_e_g, ch = ef ⊕ not_e_g.
//!   4.  `maj_definition`     — ab,ac,bc + double XOR.
//!   5.  `round_additions`    — algebraic T₁ / T₂ accounting modulo 2⁶⁴.
//!   6.  `passthrough`        — b'=a, c'=b, d'=c, f'=e, g'=f, h'=g.
//!   7.  `k_binding`          — Σ_k sel_round_k · (k_bit[i] − RC_k_bit[i]).
//!   8.  `sel_binary`         — each round selector is 0/1.
//!   9.  `sel_sum_01`         — `Σ sel · (Σ sel − 1) = 0`.
//!  10.  `small_sigma0_w`     — σ0(W) two-stage XOR over rot1/rot8/SHR7
//!                              (uses the **current** row's W).
//!  11.  `small_sigma1_w`     — σ1(W) two-stage XOR over rot19/rot61/SHR6.
//!
//! # Deferred (#258 follow-ups)
//!
//!   - **Cross-row W[t] recurrence binding**
//!     (`W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16]` for
//!     t ∈ 16..80). The host-side populator already pins the
//!     `W_RECURRENCE_CARRY` column, but a *row-local* recurrence
//!     constraint needs aggregator columns (mirror of SHA-256's
//!     `W_RECURRENCE_W0_WORD` / `W9_WORD` / `SIGMA0_W1_WORD` /
//!     `SIGMA1_W14_WORD`) holding the row-t addends. This requires
//!     extending the SHA-512 AIR layout (NUM_DATA_COLUMNS).
//!   - **Cross-row `after → before` transition**
//!     (`after(X)[i] == before(ω·X)[i]` excluding block boundaries) —
//!     the SHA-256 analogue. Requires a shifted-column descriptor and a
//!     block-boundary exclusion vector keyed to `NUM_ROUNDS = 80`.
//!   - **Final H_out[i] = H_in[i] + state[i] mod 2^64** (multi-block
//!     chaining + digest binding). Needs additional aggregator columns
//!     (`STATE_IN_WORD[v]`, chain/binding carries) plus an
//!     `aggregator_active` selector, identical in shape to the SHA-256
//!     layout.
//!   - **Per-invocation byte aggregator + cross-AIR LogUp linkage**
//!     into BLS12-381 hash_to_field / Ed25519 verification AIRs.
//!
//! These items are tracked as the natural successor follow-ups; they all
//! depend on growing the AIR layout, which is out of scope for #258.

use crate::field::{CurveType, Scalar};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::sha512_air::{
    self, after, alloc_trace, before, k_bit, populate_round, sel_round, w_bit, BlockTrace,
    HashTrace, RoundTrace, BITS_PER_WORD, COL_A_NEW_CARRY_OFFSET, COL_AFTER_OFFSET,
    COL_BEFORE_OFFSET, COL_BIG_SIGMA0_A_OFFSET, COL_BIG_SIGMA1_E_OFFSET, COL_CH_EFG_OFFSET,
    COL_E_NEW_CARRY_OFFSET, COL_K_OFFSET, COL_MAJ_ABC_OFFSET, COL_T1_CARRY_OFFSET,
    COL_T2_CARRY_OFFSET, COL_W_OFFSET, NUM_DATA_COLUMNS, NUM_ROUNDS, NUM_SEL_ROUND,
    NUM_SHA512_COLUMNS, ROUND_CONSTANTS, T1_CARRY_BITS, VAR_A, VAR_B, VAR_C, VAR_D, VAR_E, VAR_F,
    VAR_G, VAR_H,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Number of row-local constraint categories (see module docs).
pub const NUM_ROW_CONSTRAINTS: usize = 12;

// ──── Local helpers ────────────────────────────────────────────────────

/// Compute `xor(a, b) = a + b − 2·a·b` on scalars.
#[inline]
fn xor_scalar(a: &Scalar, b: &Scalar, two: &Scalar) -> Scalar {
    let ab = a.mul(b);
    a.add(b).sub(&two.mul(&ab))
}

/// `2^64` as a scalar. SHA-512's mod-2^64 accounting uses this in the
/// round-addition carry equations; `Scalar::from_u64` cannot represent
/// `1 << 64`, so we materialize it as `(1 << 32) · (1 << 32)`.
#[inline]
fn two_pow_64(curve: CurveType) -> Scalar {
    let p32 = Scalar::from_u64(1u64 << 32, curve);
    p32.mul(&p32)
}

/// Column offset helpers for the auxiliary bit blocks. These are not
/// exposed as free functions by `sha512_air`, so we compute them inline.
#[inline]
fn xor01_s0_bit(bit: usize) -> usize {
    sha512_air::COL_XOR01_S0_OFFSET + bit
}
#[inline]
fn big_sigma0_a_bit(bit: usize) -> usize {
    sha512_air::COL_BIG_SIGMA0_A_OFFSET + bit
}
#[inline]
fn xor01_s1_bit(bit: usize) -> usize {
    sha512_air::COL_XOR01_S1_OFFSET + bit
}
#[inline]
fn big_sigma1_e_bit(bit: usize) -> usize {
    sha512_air::COL_BIG_SIGMA1_E_OFFSET + bit
}
#[inline]
fn ef_bit(bit: usize) -> usize {
    sha512_air::COL_EF_OFFSET + bit
}
#[inline]
fn not_e_g_bit(bit: usize) -> usize {
    sha512_air::COL_NOT_E_G_OFFSET + bit
}
#[inline]
fn ch_efg_bit(bit: usize) -> usize {
    sha512_air::COL_CH_EFG_OFFSET + bit
}
#[inline]
fn ab_bit(bit: usize) -> usize {
    sha512_air::COL_AB_OFFSET + bit
}
#[inline]
fn ac_bit(bit: usize) -> usize {
    sha512_air::COL_AC_OFFSET + bit
}
#[inline]
fn bc_bit(bit: usize) -> usize {
    sha512_air::COL_BC_OFFSET + bit
}
#[inline]
fn xor_ab_ac_bit(bit: usize) -> usize {
    sha512_air::COL_XOR_AB_AC_OFFSET + bit
}
#[inline]
fn maj_abc_bit(bit: usize) -> usize {
    sha512_air::COL_MAJ_ABC_OFFSET + bit
}
#[inline]
fn xor01_ss0_bit(bit: usize) -> usize {
    sha512_air::COL_XOR01_SS0_OFFSET + bit
}
#[inline]
fn small_sigma0_w_bit(bit: usize) -> usize {
    sha512_air::COL_SMALL_SIGMA0_W_OFFSET + bit
}
#[inline]
fn xor01_ss1_bit(bit: usize) -> usize {
    sha512_air::COL_XOR01_SS1_OFFSET + bit
}
#[inline]
fn small_sigma1_w_bit(bit: usize) -> usize {
    sha512_air::COL_SMALL_SIGMA1_W_OFFSET + bit
}

/// Reconstruct a 64-bit word as a scalar from `n` bit columns at row `row`.
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

/// Reconstruct a 64-bit word as a polynomial: Σ bit_poly_i · 2^i.
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

// ──── Trace construction helpers ───────────────────────────────────────

/// Build a [`TracePolynomials`] from a `Vec<RoundTrace>`. Each round
/// populates one row; the trace pads to the next power of two with
/// zeros. The W[t] recurrence aggregator columns are populated as a
/// post-step.
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
    populate_w_recurrence_carry(&mut columns, rounds, curve);
    into_trace_polynomials(columns, num_rows, padded, curve)
}

/// Convenience: build a trace from a full `HashTrace` (concatenates the
/// per-block round traces).
pub fn build_trace_polynomials_from_hash(
    ht: &HashTrace,
    curve: CurveType,
) -> TracePolynomials {
    let mut rounds: Vec<RoundTrace> = Vec::with_capacity(ht.blocks.len() * NUM_ROUNDS);
    for BlockTrace { rounds: rs, .. } in &ht.blocks {
        rounds.extend(rs.iter().cloned());
    }
    build_trace_polynomials_from_rounds(&rounds, curve)
}

/// Populate the 2-bit W[t] recurrence carry column on rows t ∈ 16..80.
/// The carry value is the host-derived `floor((σ1+W7+σ0+W16)/2^64)`,
/// which fits in 2 bits (≤ 3). No row-local constraint pins this yet
/// (see "Deferred" in the module docs); the column is populated so the
/// follow-up wiring has the witness data ready.
fn populate_w_recurrence_carry(
    columns: &mut [Vec<Scalar>],
    rounds: &[RoundTrace],
    curve: CurveType,
) {
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    let n = columns[0].len();
    for t in 16..NUM_ROUNDS.min(rounds.len()) {
        if t >= n {
            break;
        }
        let w_tm16 = rounds[t - 16].w;
        let w_tm15 = rounds[t - 15].w;
        let w_tm7 = rounds[t - 7].w;
        let w_tm2 = rounds[t - 2].w;
        let s0 = sha512_air::small_sigma0(w_tm15);
        let s1 = sha512_air::small_sigma1(w_tm2);
        let full_sum: u128 =
            (s1 as u128) + (w_tm7 as u128) + (s0 as u128) + (w_tm16 as u128);
        let carry = (full_sum >> 64) as u64;
        for bit in 0..sha512_air::W_RECURRENCE_CARRY_BITS {
            let v = (carry >> bit) & 1;
            columns[sha512_air::COL_W_RECURRENCE_CARRY_OFFSET + bit][t] =
                if v == 1 { one.clone() } else { zero.clone() };
        }
    }
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

// ──── Constraint system struct ─────────────────────────────────────────

/// [`VmConstraintSystem`] implementation for the SHA-512 bit-level AIR.
pub struct Sha512ConstraintSystem {
    /// Number of real trace rows (= 80 · num_blocks). Padded domain size
    /// is the next power of two.
    pub num_rows: usize,
}

impl Sha512ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows }
    }
}

// ──── Scalar-point evaluation of each category body ────────────────────

/// 0. bit_validity: every data column satisfies `b·(b−1) = 0`.
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

/// 1. Σ₀(a) two-stage XOR over rotations 28/34/39.
fn eval_sigma0_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let a_r28 = &cols[before(VAR_A, (bit + 28) % BITS_PER_WORD)];
        let a_r34 = &cols[before(VAR_A, (bit + 34) % BITS_PER_WORD)];
        let a_r39 = &cols[before(VAR_A, (bit + 39) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_s0_bit(bit)];
        let sig0 = &cols[big_sigma0_a_bit(bit)];

        let x1 = xor_scalar(a_r28, a_r34, &two);
        let body1 = xor01.sub(&x1);
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        let x2 = xor_scalar(xor01, a_r39, &two);
        let body2 = sig0.sub(&x2);
        acc = acc.add(&bp.mul(&body2));
        bp = bp.mul(beta);
    }
    acc
}

/// 2. Σ₁(e) two-stage XOR over rotations 14/18/41.
fn eval_sigma1_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let e_r14 = &cols[before(VAR_E, (bit + 14) % BITS_PER_WORD)];
        let e_r18 = &cols[before(VAR_E, (bit + 18) % BITS_PER_WORD)];
        let e_r41 = &cols[before(VAR_E, (bit + 41) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_s1_bit(bit)];
        let sig1 = &cols[big_sigma1_e_bit(bit)];

        let x1 = xor_scalar(e_r14, e_r18, &two);
        let body1 = xor01.sub(&x1);
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        let x2 = xor_scalar(xor01, e_r41, &two);
        let body2 = sig1.sub(&x2);
        acc = acc.add(&bp.mul(&body2));
        bp = bp.mul(beta);
    }
    acc
}

/// 3. ch_definition: ef = e·f, not_e_g = (1−e)·g, ch = ef ⊕ not_e_g.
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

/// 4. maj_definition: ab, ac, bc + double XOR.
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

/// 5. round_additions: two value-level relations modulo `2^64`.
///   (A) h + Σ₁(e) + Ch + K + W = T1_value + t1_carry · 2^64
///   (B) Σ₀(a) + Maj            = T2_value + t2_carry · 2^64
/// with T1 = (new_e + e_new_carry · 2^64) − d
///      T2 = (new_a + a_new_carry · 2^64) − T1.
fn eval_round_adds_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two_64 = two_pow_64(curve);
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

    // T1 = new_e + e_new_carry·2^64 − d.
    let t1_val = new_e_val.add(&enc.mul(&two_64)).sub(&d_val);

    // (A) lhs = h + Σ₁(e) + Ch + K + W
    //     rhs = T1_value + t1_carry · 2^64
    let lhs_a = h_val.add(&s1_val).add(&ch_val).add(&k_val).add(&w_val);
    let rhs_a = t1_val.add(&t1c.mul(&two_64));
    let body_a = lhs_a.sub(&rhs_a);

    // T2 = new_a + a_new_carry·2^64 − T1.
    let t1_val_b = new_e_val.add(&enc.mul(&two_64)).sub(&d_val);
    let t2_val = new_a_val.add(&anc.mul(&two_64)).sub(&t1_val_b);

    let lhs_b = s0_val.add(&maj_val);
    let rhs_b = t2_val.add(&t2c.mul(&two_64));
    let body_b = lhs_b.sub(&rhs_b);

    body_a.add(&beta.mul(&body_b))
}

/// 6. passthrough: b'=a, c'=b, d'=c, f'=e, g'=f, h'=g — bit-by-bit.
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

/// 7. k_binding: Σ_k sel_round_k · (k_bit[i] − RC_k_bit[i]) = 0.
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

/// 8. sel_binary.
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

/// 9. sel_sum_01: `(Σ sel) · (Σ sel − 1) = 0`.
fn eval_sel_sum_01_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let mut sum = Scalar::zero(curve);
    for k in 0..NUM_SEL_ROUND {
        sum = sum.add(&cols[sel_round(k)]);
    }
    sum.mul(&sum.sub(&one))
}

/// 10. σ0(W) two-stage XOR over rotations 1/8 and SHR-7. Operates on the
/// **current** row's W. Cross-row binding of W[t] to the recurrence is
/// deferred (see module docs).
fn eval_small_sigma0_w_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let w_r1 = &cols[w_bit((bit + 1) % BITS_PER_WORD)];
        let w_r8 = &cols[w_bit((bit + 8) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_ss0_bit(bit)];
        let ss0 = &cols[small_sigma0_w_bit(bit)];

        let x1 = xor_scalar(w_r1, w_r8, &two);
        let body1 = xor01.sub(&x1);
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        if bit + 7 < BITS_PER_WORD {
            let w_shr7 = &cols[w_bit(bit + 7)];
            let x2 = xor_scalar(xor01, w_shr7, &two);
            let body2 = ss0.sub(&x2);
            acc = acc.add(&bp.mul(&body2));
        } else {
            let body2 = ss0.sub(xor01);
            acc = acc.add(&bp.mul(&body2));
        }
        bp = bp.mul(beta);
    }
    acc
}

/// 11. σ1(W) two-stage XOR over rotations 19/61 and SHR-6.
fn eval_small_sigma1_w_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let w_r19 = &cols[w_bit((bit + 19) % BITS_PER_WORD)];
        let w_r61 = &cols[w_bit((bit + 61) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_ss1_bit(bit)];
        let ss1 = &cols[small_sigma1_w_bit(bit)];

        let x1 = xor_scalar(w_r19, w_r61, &two);
        let body1 = xor01.sub(&x1);
        acc = acc.add(&bp.mul(&body1));
        bp = bp.mul(beta);

        if bit + 6 < BITS_PER_WORD {
            let w_shr6 = &cols[w_bit(bit + 6)];
            let x2 = xor_scalar(xor01, w_shr6, &two);
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

// ──── Polynomial-form builders ─────────────────────────────────────────

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
        let a_r28 = &cols[before(VAR_A, (bit + 28) % BITS_PER_WORD)];
        let a_r34 = &cols[before(VAR_A, (bit + 34) % BITS_PER_WORD)];
        let a_r39 = &cols[before(VAR_A, (bit + 39) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_s0_bit(bit)];
        let sig0 = &cols[big_sigma0_a_bit(bit)];

        let ab = poly_mul(a_r28, a_r34, curve);
        let two_ab = poly_scalar_mul(&ab, &two);
        let xor1 = poly_sub(&poly_add(a_r28, a_r34, curve), &two_ab, curve);
        let body1 = poly_sub(xor01, &xor1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        let ab2 = poly_mul(xor01, a_r39, curve);
        let two_ab2 = poly_scalar_mul(&ab2, &two);
        let xor2 = poly_sub(&poly_add(xor01, a_r39, curve), &two_ab2, curve);
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
        let e_r14 = &cols[before(VAR_E, (bit + 14) % BITS_PER_WORD)];
        let e_r18 = &cols[before(VAR_E, (bit + 18) % BITS_PER_WORD)];
        let e_r41 = &cols[before(VAR_E, (bit + 41) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_s1_bit(bit)];
        let sig1 = &cols[big_sigma1_e_bit(bit)];

        let ab = poly_mul(e_r14, e_r18, curve);
        let two_ab = poly_scalar_mul(&ab, &two);
        let xor1 = poly_sub(&poly_add(e_r14, e_r18, curve), &two_ab, curve);
        let body1 = poly_sub(xor01, &xor1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        let ab2 = poly_mul(xor01, e_r41, curve);
        let two_ab2 = poly_scalar_mul(&ab2, &two);
        let xor2 = poly_sub(&poly_add(xor01, e_r41, curve), &two_ab2, curve);
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

        let ef_prod = poly_mul(e, f, curve);
        let body1 = poly_sub(ef, &ef_prod, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        let not_e = poly_sub(&one_poly, e, curve);
        let neg_prod = poly_mul(&not_e, g, curve);
        let body2 = poly_sub(ng, &neg_prod, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        bp = bp.mul(beta);

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
    let two_64 = two_pow_64(curve);
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

    let enc_scaled = poly_scalar_mul(enc_poly, &two_64);
    let t1_val = poly_sub(&poly_add(&new_e_val, &enc_scaled, curve), &d_val, curve);

    let mut lhs_a = poly_add(&h_val, &s1_val, curve);
    lhs_a = poly_add(&lhs_a, &ch_val, curve);
    lhs_a = poly_add(&lhs_a, &k_val, curve);
    lhs_a = poly_add(&lhs_a, &w_val, curve);
    let t1c_scaled = poly_scalar_mul(&t1c_poly, &two_64);
    let rhs_a = poly_add(&t1_val, &t1c_scaled, curve);
    let body_a = poly_sub(&lhs_a, &rhs_a, curve);

    let anc_scaled = poly_scalar_mul(anc_poly, &two_64);
    let t2_val = poly_sub(&poly_add(&new_a_val, &anc_scaled, curve), &t1_val, curve);

    let lhs_b = poly_add(&s0_val, &maj_val, curve);
    let t2c_scaled = poly_scalar_mul(t2c_poly, &two_64);
    let rhs_b = poly_add(&t2_val, &t2c_scaled, curve);
    let body_b = poly_sub(&lhs_b, &rhs_b, curve);

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
        let s_minus_1 = poly_sub(s, &one_poly, curve);
        let body = poly_mul(s, &s_minus_1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

fn build_sel_sum_01_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let mut sum = vec![Scalar::zero(curve)];
    for k in 0..NUM_SEL_ROUND {
        sum = poly_add(&sum, &cols[sel_round(k)], curve);
    }
    let sum_minus_1 = poly_sub(&sum, &one_poly, curve);
    poly_mul(&sum, &sum_minus_1, curve)
}

fn build_small_sigma0_w_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for bit in 0..BITS_PER_WORD {
        let w_r1 = &cols[w_bit((bit + 1) % BITS_PER_WORD)];
        let w_r8 = &cols[w_bit((bit + 8) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_ss0_bit(bit)];
        let ss0 = &cols[small_sigma0_w_bit(bit)];

        let prod = poly_mul(w_r1, w_r8, curve);
        let two_prod = poly_scalar_mul(&prod, &two);
        let xor1 = poly_sub(&poly_add(w_r1, w_r8, curve), &two_prod, curve);
        let body1 = poly_sub(xor01, &xor1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        let body2 = if bit + 7 < BITS_PER_WORD {
            let w_shr7 = &cols[w_bit(bit + 7)];
            let prod2 = poly_mul(xor01, w_shr7, curve);
            let two_prod2 = poly_scalar_mul(&prod2, &two);
            let xor2 = poly_sub(&poly_add(xor01, w_shr7, curve), &two_prod2, curve);
            poly_sub(ss0, &xor2, curve)
        } else {
            poly_sub(ss0, xor01, curve)
        };
        acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
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
        let w_r19 = &cols[w_bit((bit + 19) % BITS_PER_WORD)];
        let w_r61 = &cols[w_bit((bit + 61) % BITS_PER_WORD)];
        let xor01 = &cols[xor01_ss1_bit(bit)];
        let ss1 = &cols[small_sigma1_w_bit(bit)];

        let prod = poly_mul(w_r19, w_r61, curve);
        let two_prod = poly_scalar_mul(&prod, &two);
        let xor1 = poly_sub(&poly_add(w_r19, w_r61, curve), &two_prod, curve);
        let body1 = poly_sub(xor01, &xor1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body1, &bp), curve);
        bp = bp.mul(beta);

        let body2 = if bit + 6 < BITS_PER_WORD {
            let w_shr6 = &cols[w_bit(bit + 6)];
            let prod2 = poly_mul(xor01, w_shr6, curve);
            let two_prod2 = poly_scalar_mul(&prod2, &two);
            let xor2 = poly_sub(&poly_add(xor01, w_shr6, curve), &two_prod2, curve);
            poly_sub(ss1, &xor2, curve)
        } else {
            poly_sub(ss1, xor01, curve)
        };
        acc = poly_add(&acc, &poly_scalar_mul(&body2, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

// ──── VmConstraintSystem impl ──────────────────────────────────────────

impl VmConstraintSystem for Sha512ConstraintSystem {
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
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(
            columns.len() == NUM_SHA512_COLUMNS,
            "sha512 AIR expects {} columns",
            NUM_SHA512_COLUMNS
        );
        let n = columns[0].len();
        let curve = columns[0][0].curve_type();
        let beta = Scalar::from_u64(2, curve);
        let mut evals: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        // Per-row scalar-point evaluation. The categories are pure
        // row-local polynomials in the trace columns, so we can drive
        // the same scalar-point evaluator on every row.
        for row in 0..n {
            let mut row_vals: Vec<Scalar> = Vec::with_capacity(NUM_SHA512_COLUMNS);
            for c in columns.iter().take(NUM_SHA512_COLUMNS) {
                row_vals.push(c[row].clone());
            }
            let v0 = eval_bit_validity_at_point(&row_vals, &beta);
            let v1 = eval_sigma0_at_point(&row_vals, &beta);
            let v2 = eval_sigma1_at_point(&row_vals, &beta);
            let v3 = eval_ch_at_point(&row_vals, &beta);
            let v4 = eval_maj_at_point(&row_vals, &beta);
            let v5 = eval_round_adds_at_point(&row_vals, &beta);
            let v6 = eval_passthrough_at_point(&row_vals, &beta);
            let v7 = eval_k_binding_at_point(&row_vals, &beta);
            let v8 = eval_sel_binary_at_point(&row_vals, &beta);
            let v9 = eval_sel_sum_01_at_point(&row_vals);
            let v10 = eval_small_sigma0_w_at_point(&row_vals, &beta);
            let v11 = eval_small_sigma1_w_at_point(&row_vals, &beta);
            evals[0][row] = v0;
            evals[1][row] = v1;
            evals[2][row] = v2;
            evals[3][row] = v3;
            evals[4][row] = v4;
            evals[5][row] = v5;
            evals[6][row] = v6;
            evals[7][row] = v7;
            evals[8][row] = v8;
            evals[9][row] = v9;
            evals[10][row] = v10;
            evals[11][row] = v11;
        }
        evals
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals_at_z.len() < NUM_SHA512_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let beta = alpha;
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

    /// Padding rows carry all-zero data + selector columns, which makes
    /// every row-local body vanish trivially. `sel_sum_01`'s
    /// `sum·(sum−1) = 0` form admits the sum = 0 case.
    fn padding_selector_column(&self) -> Option<usize> {
        None
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
        if columns.len() < NUM_SHA512_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_SHA512_COLUMNS) {
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
                let c = curve;
                move || build_sel_sum_01_poly(&cols, c)
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
}

// ──── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha512_air::sha512_witness;

    /// Honest "abc" witness: all 12 row-local constraint categories
    /// vanish at the scalar-point evaluator on every populated row.
    #[test]
    fn sha512_constraints_honest_witness_vanishes_scalar() {
        let curve = CurveType::Bls48581;
        let ht = sha512_witness(b"abc");
        let rounds: Vec<RoundTrace> = ht
            .blocks
            .iter()
            .flat_map(|b| b.rounds.iter().cloned())
            .collect();
        let mut cols = alloc_trace(rounds.len(), curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut cols, row, rt, curve);
        }
        populate_w_recurrence_carry(&mut cols, &rounds, curve);

        let beta = Scalar::from_u64(2, curve);
        let zero = Scalar::zero(curve);
        for row in 0..rounds.len() {
            let row_vals: Vec<Scalar> =
                cols.iter().take(NUM_SHA512_COLUMNS).map(|c| c[row].clone()).collect();
            let bodies: [(Scalar, &'static str); NUM_ROW_CONSTRAINTS] = [
                (eval_bit_validity_at_point(&row_vals, &beta), "bit_validity"),
                (eval_sigma0_at_point(&row_vals, &beta), "big_sigma0_a"),
                (eval_sigma1_at_point(&row_vals, &beta), "big_sigma1_e"),
                (eval_ch_at_point(&row_vals, &beta), "ch"),
                (eval_maj_at_point(&row_vals, &beta), "maj"),
                (eval_round_adds_at_point(&row_vals, &beta), "round_additions"),
                (eval_passthrough_at_point(&row_vals, &beta), "passthrough"),
                (eval_k_binding_at_point(&row_vals, &beta), "k_binding"),
                (eval_sel_binary_at_point(&row_vals, &beta), "sel_binary"),
                (eval_sel_sum_01_at_point(&row_vals), "sel_sum_01"),
                (eval_small_sigma0_w_at_point(&row_vals, &beta), "small_sigma0_w"),
                (eval_small_sigma1_w_at_point(&row_vals, &beta), "small_sigma1_w"),
            ];
            for (val, label) in &bodies {
                assert_eq!(
                    val.to_bytes(),
                    zero.to_bytes(),
                    "category `{}` did not vanish on honest row {}",
                    label,
                    row,
                );
            }
        }
    }

    /// `evaluate_on_domain` returns all-zero columns for an honest
    /// witness — sanity check that the per-row scalar evaluator matches
    /// the trait surface.
    #[test]
    fn sha512_constraints_evaluate_on_domain_vanishes() {
        let curve = CurveType::Bls48581;
        let ht = sha512_witness(b"abc");
        let trace = build_trace_polynomials_from_hash(&ht, curve);
        let cs = Sha512ConstraintSystem::new(trace.num_rows);
        let cols_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_refs, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        let zero = Scalar::zero(curve);
        for (i, col) in evals.iter().enumerate() {
            for (row, v) in col.iter().enumerate() {
                assert_eq!(
                    v.to_bytes(),
                    zero.to_bytes(),
                    "category {} non-zero at row {}",
                    i,
                    row,
                );
            }
        }
    }

    /// Tampering a single Ch bit on row 5 makes the `ch_definition`
    /// constraint fire — non-zero at the tampered row's scalar eval.
    #[test]
    fn sha512_constraints_tampered_ch_fires() {
        let curve = CurveType::Bls48581;
        let ht = sha512_witness(b"abc");
        let rounds: Vec<RoundTrace> = ht.blocks[0].rounds.clone();
        let mut cols = alloc_trace(rounds.len(), curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut cols, row, rt, curve);
        }
        // Flip ch bit 0 on row 5.
        let row = 5;
        let idx = ch_efg_bit(0);
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);
        cols[idx][row] = if cols[idx][row].to_bytes() == one.to_bytes() {
            zero.clone()
        } else {
            one.clone()
        };

        let beta = Scalar::from_u64(2, curve);
        let row_vals: Vec<Scalar> =
            cols.iter().take(NUM_SHA512_COLUMNS).map(|c| c[row].clone()).collect();
        let ch_body = eval_ch_at_point(&row_vals, &beta);
        assert_ne!(
            ch_body.to_bytes(),
            zero.to_bytes(),
            "ch_definition body must fire on a tampered ch bit",
        );
    }

    /// Shape sanity: constraint labels exposed match the count.
    #[test]
    fn sha512_constraints_labels_count() {
        let cs = Sha512ConstraintSystem::new(NUM_ROUNDS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        let selectors = cs.selector_column_indices();
        assert_eq!(selectors.len(), NUM_SEL_ROUND);
    }

    /// W[t] recurrence carry populator: on an honest `abc` witness the
    /// carry column is binary (each bit is 0 or 1) on rows t ∈ 16..80.
    /// Cross-row binding to actual W is deferred (see module docs).
    #[test]
    fn sha512_constraints_w_recurrence_carry_is_binary() {
        let curve = CurveType::Bls48581;
        let ht = sha512_witness(b"abc");
        let rounds: Vec<RoundTrace> = ht.blocks[0].rounds.clone();
        let mut cols = alloc_trace(rounds.len(), curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut cols, row, rt, curve);
        }
        populate_w_recurrence_carry(&mut cols, &rounds, curve);
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        for t in 16..NUM_ROUNDS {
            for bit in 0..sha512_air::W_RECURRENCE_CARRY_BITS {
                let v = &cols[sha512_air::COL_W_RECURRENCE_CARRY_OFFSET + bit][t];
                assert!(
                    v.to_bytes() == zero.to_bytes() || v.to_bytes() == one.to_bytes(),
                    "carry bit {} on row {} is not binary",
                    bit,
                    t,
                );
            }
        }
    }

    /// Marked `#[ignore]`: full prove/verify roundtrip on a SHA-512
    /// trace is expensive (≥ ~2,260 columns × 128-row padded domain
    /// under BLS12-381 KZG + the row-local polynomial builds). Run with:
    ///   cargo test --release -p metavm-zkp --lib \
    ///     sha512_cs_prove_verify_single_round -- --ignored --nocapture
    #[test]
    #[ignore = "slow: full prover roundtrip; run with --release --ignored"]
    fn sha512_cs_prove_verify_single_round() {
        let curve = CurveType::Bls12381;
        let scheme = crate::scheme::bls12381_scheme::Bls12381Scheme::new();
        let ht = sha512_witness(b"abc");
        // Single-block trace = 80 rows; padded to 128.
        let rounds: Vec<RoundTrace> = ht.blocks[0].rounds.clone();
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let cs = Sha512ConstraintSystem::new(rounds.len());
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "sha512 single-block proof must verify");
    }
}
