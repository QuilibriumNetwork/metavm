//! [`VmConstraintSystem`] wiring for the Keccak-f[1600] bit-level AIR.
//!
//! This module adapts the consolidated constraint bodies defined in
//! [`crate::keccak_air`] into the shape required by the generic
//! [`prove_with_scheme`] / [`verify_with_scheme`] pipeline. It provides:
//!
//! - `KeccakConstraintSystem`: the trait implementer. Construct with
//!   `KeccakConstraintSystem::new(num_rows)` where `num_rows` is
//!   `24 * number_of_permutations` (the real, pre-padding trace height).
//! - `build_trace_polynomials_from_rounds`: helper that lifts a
//!   `Vec<RoundTrace>` into the `TracePolynomials` the pipeline expects.
//!
//! # Constraint layout
//!
//! 12 row-local consolidated categories (label → index):
//!
//!   0.  `bit_validity`
//!   1.  `c_partial_recurrence`
//!   2.  `d_definition`
//!   3.  `theta_apply`
//!   4.  `rho_apply`
//!   5.  `pi_apply`
//!   6.  `chi_nand_definition`
//!   7.  `chi_apply`
//!   8.  `iota_passthrough`
//!   9.  `iota_xor_rc`
//!   10. `sel_binary`
//!   11. `sel_sum_01`
//!
//! 1 shifted (cross-row) constraint: `cross_row_transition`.
//!
//! Each category aggregates its many bit-level sub-constraints via powers
//! of a β challenge; we fix β = α for simplicity (the combined constraint
//! polynomial remains a specific polynomial in α, and Schwartz-Zippel still
//! applies).
//!
//! # Padding strategy
//!
//! We return `None` from [`VmConstraintSystem::padding_selector_column`] so
//! every bit column (including all 24 selectors) is zero on padding rows —
//! this makes every row-local body vanish. The selector sum-to-one
//! constraint is enforced as `sum · (sum − 1) = 0`, which admits the
//! sum = 0 case on padding.
//!
//! For the cross-row `after_iota(ω·X) == before(X)` transition, we exclude
//! both the domain wrap-around row (ω^{n-1}) and the last real row
//! (ω^{num_rows-1}) — because on that boundary a non-zero real `after_iota`
//! would otherwise be required to equal the zero `before` of the first
//! padding row. For traces containing multiple permutations we also
//! exclude each intermediate permutation boundary (rows `23, 47, …`).

use crate::field::{CurveType, Scalar};
use crate::keccak::{HashTrace, RoundTrace, NUM_ROUNDS, RHO_OFFSETS, ROUND_CONSTANTS};
use crate::keccak_air::{
    self, after_chi, after_iota, after_pi, after_rho, after_theta, alloc_trace, before, c_final,
    c_partial, d_col, nand_temp, populate_round, populate_trace_from_hash, sel_round,
    BITS_PER_LANE, BITS_PER_STATE, COL_BEFORE_OFFSET, NUM_DATA_COLUMNS,
    NUM_KECCAK_COLUMNS,
    NUM_SEL_ROUND,
};
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Number of consolidated row-local constraint categories. Layout:
///   - 12 bit-level Keccak round bodies.
///   - 13: `is_first_inv_row_binary` — anchor binarity (LogUp soundness).
///   - 14: `is_first_block_binary` — block-0 indicator binarity.
///   - 15: `is_first_block_pinned_at_anchor`.
///   - 16: `is_byte_active_binary` (per-byte at row 0, β-RLC).
///   - 17: `is_byte_active_monotone` (per-byte at row 0, β-RLC).
///   - 18: `is_byte_active_sum_pin` (Σ IS_BYTE_ACTIVE = INV_INPUT_LEN).
///   - 19: `inv_input_byte_binding` — gated by `IS_BYTE_ACTIVE[b]` per
///     byte, binds `INV_INPUT_BYTE[b]` to the rate-byte decomposition
///     of the BEFORE state at row 0. Closes the algebraic input
///     binding for single-absorption (≤ RATE_LEN-byte) inputs.
///   - 20: `is_last_block_binary` — `IS_LAST_BLOCK` is 0 or 1.
///   - 21: `inv_output_byte_binding` — at the LAST round of the LAST
///     block (gated by `IS_LAST_BLOCK · SEL_ROUND[NUM_ROUNDS-1]`),
///     pins each of the 32 INV_OUTPUT_BYTE bytes to the little-endian
///     decomposition of the first 4 lanes of `AFTER_IOTA` (= the
///     Keccak-256 digest at the squeeze step). Works for both
///     single-absorption (1 permutation) and multi-absorption
///     (N permutations).
///   - 22: `is_byte_active_off_anchor_zero` — `(1 − IS_FIRST_INV_ROW) ·
///     is_byte_active[b] = 0` for each `b ∈ 0..NUM_IS_BYTE_ACTIVE`.
///     β-RLC. The witness only meaningfully populates `is_byte_active`
///     on the anchor row, but the existing binarity / monotone / sum
///     constraints all gate on `IS_FIRST_INV_ROW`, so a malicious
///     prover could otherwise stash arbitrary garbage in those columns
///     on non-anchor rows. This pins them to zero, closing the
///     algebraic gap and making the column behaviour canonical.
///   - 23: `sel_one_when_first_block` — `IS_FIRST_BLOCK · (Σ_k
///     sel_round_k − 1) = 0`. Sharpens `sel_sum_01`: on every row of
///     block 0 (the only block whose rows are pinned to "active" by
///     the IS_FIRST_BLOCK chain), exactly one round selector must be
///     active. Forces well-formedness of selectors on block 0
///     specifically; legacy single-permutation traces have
///     IS_FIRST_BLOCK = 0 throughout (when not populated via the
///     aggregator path), and so are unaffected.
///   - 24: `rc_word_pin` — explicit algebraic pin of the round-constant
///     lane committed to by the prover. Defines the derived RC word
///     `derived_RC := Σ_bit 2^bit · XOR(after_iota[0][0][bit],
///     after_chi[0][0][bit])` (because ι sets
///     `after_iota[0][0] = after_chi[0][0] XOR RC[k]`, the XOR of the
///     two lane bits recovers the RC lane bit). The constraint is
///     `Σ_k SEL_ROUND[k] · (derived_RC − ROUND_CONSTANTS[k]) = 0`.
///     On every active row exactly one selector is high (enforced by
///     `sel_sum_01` + `sel_binary`), so this pins `derived_RC` to the
///     spec constant for that round. Provides word-level
///     defense-in-depth over the existing bit-level
///     [`iota_xor_rc`] constraint and explicitly commits the RC
///     value to the spec table, blocking any malicious prover from
///     splicing a tampered RC lane even if a bit-level body could
///     somehow be elided. Degree 3 in column polys (XOR is degree 2
///     in bit cols, gated by the linear selector).
pub const NUM_ROW_CONSTRAINTS: usize = 24;

/// Number of consolidated cross-row (shifted) constraints.
///
///   0. `after_iota → before` binding (existing): `after_iota(X)[i] =
///      before(ω·X)[i]` for each of 1600 lane bits, β-RLC'd. Excluded
///      at every permutation boundary + domain wrap.
///   1. **Aggregator invariance** (added by per-invocation byte
///      aggregation): `INV_INPUT_BYTE_b(ω·X) − INV_INPUT_BYTE_b(X) = 0`,
///      `INV_INPUT_LEN_COL(ω·X) − INV_INPUT_LEN_COL(X) = 0`, and
///      `INV_OUTPUT_BYTE_b(ω·X) − INV_OUTPUT_BYTE_b(X) = 0`, β-RLC'd
///      over 256 + 1 + 32 = 289 bodies. Excluded only at the very last
///      real-row transition + domain wrap.
///   2. **IS_FIRST_BLOCK invariance within permutation**.
///   3. **Block-1 input binding (multi-absorption)** — at the
///      block-0→block-1 transition (row 23, gated by
///      `IS_FIRST_BLOCK · SEL_ROUND[NUM_ROUNDS-1] · (1 − IS_LAST_BLOCK)`),
///      for each rate byte position `pos ∈ 0..136`, pins
///      `INV_INPUT_BYTE[136 + pos]` (gated by `IS_BYTE_ACTIVE[136 + pos]`)
///      to the bit-XOR of `BEFORE(ω·X)` (block 1's start state) and
///      `AFTER_IOTA(X)` (block 0's end state). Specifically per byte:
///        `byte = Σ_{k=0..8} (BEFORE_next_bit_k XOR AFTER_IOTA_curr_bit_k) · 2^k`
///      where the lane (x, y) and inner-bit-offset are derived from
///      `pos`, and XOR is encoded as `a + b − 2 · a · b`. β-RLC over
///      136 byte positions. Soundness scope: 2-block only (MPT
///      branch nodes ≤ 256 bytes); 3+ block traces would need the
///      same constraint at intermediate boundaries with a different
///      gate (deferred).
pub const NUM_SHIFTED: usize = 4;

// ──── Constraint system ────────────────────────────────────────────────

/// [`VmConstraintSystem`] implementation for the Keccak-f[1600] bit-level AIR.
pub struct KeccakConstraintSystem {
    /// Number of real trace rows (before padding). Must be a multiple of 24
    /// (one row per Keccak round). The padded domain size is determined by
    /// [`TracePolynomials`] to the next power of two ≥ `num_rows`.
    pub num_rows: usize,
    /// The domain generator ω for the trace's padded domain. When `Some`,
    /// the verifier-side `evaluate_shifted_at_point` excludes every
    /// permutation boundary `(X − ω^{23}), (X − ω^{47}), …` in addition to
    /// the domain wrap `(X − ω^{n−1})`. When `None`, only the wrap-around
    /// factor is excluded — which is only sound when `num_rows == domain_size`.
    pub omega: Option<Scalar>,
    /// The padded domain size (power of two ≥ `num_rows`). Used together
    /// with `omega` to build the boundary-row exclusion product.
    pub domain_size: Option<u64>,
}

impl KeccakConstraintSystem {
    /// Construct a Keccak constraint system for a trace of `num_rows` real
    /// rows (= `24 * num_permutations`). `omega` / `domain_size` default
    /// to `None` — callers that use traces where `num_rows < domain_size`
    /// should set them via [`Self::with_omega_and_domain`] so the
    /// verifier can replicate the prover's boundary-row exclusion product.
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

// ──── Trace construction helper ────────────────────────────────────────

/// Build a [`TracePolynomials`] wrapping the Keccak bit-level trace
/// produced by populating `rounds` in order. Each round populates exactly
/// one row; the trace pads up to the next power of two automatically
/// (with zeros — consistent with our "no selector on padding" strategy).
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
    into_trace_polynomials(columns, num_rows, padded, curve)
}

/// Variant for full hash witnesses (multi-block inputs).
pub fn build_trace_polynomials_from_hash(
    hash_trace: &HashTrace,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = hash_trace.blocks.len() * NUM_ROUNDS;
    let base_columns = populate_trace_from_hash(hash_trace, curve);
    // Already sized to num_rows, which may not be a power of two yet; re-pad.
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

// ──── Scalar-point evaluation of each category body ────────────────────

/// Evaluate `bit_validity` at a scalar point.
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

/// Evaluate `c_partial_recurrence` at a scalar point.
fn eval_c_partial_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for bit in 0..BITS_PER_LANE {
            // Initial: C_partial[x][0][bit] − before[x][0][bit] = 0
            let cp0 = &cols[c_partial(x, 0, bit)];
            let b0 = &cols[before(x, 0, bit)];
            let body = cp0.sub(b0);
            acc = acc.add(&bp.mul(&body));
            bp = bp.mul(beta);
            for y in 0..4 {
                let a = &cols[c_partial(x, y, bit)];
                let b = &cols[before(x, y + 1, bit)];
                let xor = xor_scalar(a, b, &two);
                let cp_next = &cols[c_partial(x, y + 1, bit)];
                let body = cp_next.sub(&xor);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

/// Evaluate `d_definition` at a scalar point.
fn eval_d_def_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for bit in 0..BITS_PER_LANE {
            let cl = &cols[c_final((x + 4) % 5, bit)];
            let cr = &cols[c_final((x + 1) % 5, (bit + 63) % BITS_PER_LANE)];
            let d = &cols[d_col(x, bit)];
            let xor = xor_scalar(cl, cr, &two);
            let body = d.sub(&xor);
            acc = acc.add(&bp.mul(&body));
            bp = bp.mul(beta);
        }
    }
    acc
}

/// Evaluate `theta_apply` at a scalar point.
fn eval_theta_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let b = &cols[before(x, y, bit)];
                let d = &cols[d_col(x, bit)];
                let at = &cols[after_theta(x, y, bit)];
                let xor = xor_scalar(b, d, &two);
                let body = at.sub(&xor);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

/// Evaluate `rho_apply` at a scalar point.
fn eval_rho_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            let off = RHO_OFFSETS[x][y] as usize;
            for bit in 0..BITS_PER_LANE {
                let src_bit = (bit + BITS_PER_LANE - off) % BITS_PER_LANE;
                let at = &cols[after_theta(x, y, src_bit)];
                let ar = &cols[after_rho(x, y, bit)];
                let body = ar.sub(at);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

/// Evaluate `pi_apply` at a scalar point.
fn eval_pi_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            let nx = y;
            let ny = (2 * x + 3 * y) % 5;
            for bit in 0..BITS_PER_LANE {
                let ar = &cols[after_rho(x, y, bit)];
                let ap = &cols[after_pi(nx, ny, bit)];
                let body = ap.sub(ar);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

/// Evaluate `chi_nand_definition` at a scalar point.
fn eval_chi_nand_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let ap1 = &cols[after_pi((x + 1) % 5, y, bit)];
                let ap2 = &cols[after_pi((x + 2) % 5, y, bit)];
                let nt = &cols[nand_temp(x, y, bit)];
                let prod = one.sub(ap1).mul(ap2);
                let body = nt.sub(&prod);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

/// Evaluate `chi_apply` at a scalar point.
fn eval_chi_apply_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let ap = &cols[after_pi(x, y, bit)];
                let nt = &cols[nand_temp(x, y, bit)];
                let ac = &cols[after_chi(x, y, bit)];
                let xor = xor_scalar(ap, nt, &two);
                let body = ac.sub(&xor);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

/// Evaluate `iota_passthrough` at a scalar point (excludes lane (0,0)).
fn eval_iota_passthrough_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            if x == 0 && y == 0 {
                continue;
            }
            for bit in 0..BITS_PER_LANE {
                let ac = &cols[after_chi(x, y, bit)];
                let ai = &cols[after_iota(x, y, bit)];
                let body = ai.sub(ac);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

/// Evaluate `iota_xor_rc` at a scalar point.
/// For each round k and bit b:
///   gated = sel_round_k · (after_iota[0][0][bit] − XOR(after_chi[0][0][bit], RC_k[bit]))
fn eval_iota_xor_rc_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for k in 0..NUM_ROUNDS {
        let sel = &cols[sel_round(k)];
        let rc = ROUND_CONSTANTS[k];
        for bit in 0..BITS_PER_LANE {
            let rc_bit = ((rc >> bit) & 1) as u64;
            let ac = &cols[after_chi(0, 0, bit)];
            let ai = &cols[after_iota(0, 0, bit)];
            // XOR(ac, rc_bit): if rc_bit=0, xor = ac; if rc_bit=1, xor = 1 − ac.
            let xor = if rc_bit == 1 { one.sub(ac) } else { ac.clone() };
            let diff = ai.sub(&xor);
            let gated = sel.mul(&diff);
            acc = acc.add(&bp.mul(&gated));
            bp = bp.mul(beta);
        }
    }
    acc
}

/// Evaluate `sel_binary` at a scalar point.
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

/// Evaluate `sel_sum_01` at a scalar point (no β-RLC: single sub-constraint).
fn eval_sel_sum_01_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let mut sum = Scalar::zero(curve);
    for k in 0..NUM_SEL_ROUND {
        sum = sum.add(&cols[sel_round(k)]);
    }
    sum.mul(&sum.sub(&one))
}

/// 13. is_first_inv_row_binary: `IS_FIRST_INV_ROW · (IS_FIRST_INV_ROW − 1) = 0`.
/// Required for cross-AIR LogUp linkage soundness; mirrors
/// [`crate::sha256_constraints`].
fn eval_is_first_inv_row_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    v.mul(&v.sub(&one))
}

/// 14. is_first_block_binary: `IS_FIRST_BLOCK · (IS_FIRST_BLOCK − 1) = 0`.
fn eval_is_first_block_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[crate::keccak_air::COL_IS_FIRST_BLOCK];
    v.mul(&v.sub(&one))
}

/// 15. is_first_block_pinned_at_anchor:
///     `IS_FIRST_INV_ROW · (1 − IS_FIRST_BLOCK) = 0`.
/// At row 0 (anchor), forces IS_FIRST_BLOCK = 1. Combined with the
/// within-permutation invariance shifted constraint, this gives
/// IS_FIRST_BLOCK = 1 on all 24 rounds of permutation 0.
fn eval_is_first_block_pinned_at_anchor_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let fb = &cols[crate::keccak_air::COL_IS_FIRST_BLOCK];
    anchor.mul(&one.sub(fb))
}

/// 16. is_byte_active_binary (β-RLC over b ∈ 0..NUM_IS_BYTE_ACTIVE):
fn eval_is_byte_active_binary_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
        let v = &cols[crate::keccak_air::is_byte_active(b)];
        let body = v.mul(&v.sub(&one));
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    anchor.mul(&acc)
}

/// 17. is_byte_active_monotone (β-RLC over b ∈ 1..NUM_IS_BYTE_ACTIVE):
fn eval_is_byte_active_monotone_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for b in 1..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
        let v_curr = &cols[crate::keccak_air::is_byte_active(b)];
        let v_prev = &cols[crate::keccak_air::is_byte_active(b - 1)];
        let body = v_curr.mul(&one.sub(v_prev));
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    anchor.mul(&acc)
}

/// 18. is_byte_active_sum_pin:
///     `IS_FIRST_INV_ROW · (Σ_b is_byte_active[b] − INV_INPUT_LEN_COL) = 0`.
/// Pins the count of active bytes to the witnessed input length —
/// combined with binarity and monotonicity, this forces
/// `is_byte_active[b] = 1` exactly for `b < INV_INPUT_LEN_COL`.
///
/// **`INV_INPUT_LEN_COL` is implicitly range-checked to `[0, 256]`** by
/// the existing constraints on `IS_FIRST_INV_ROW=1` rows:
///   1. `is_byte_active_binary` (constraint 16) forces every
///      `IS_BYTE_ACTIVE[b]` to be 0 or 1.
///   2. `is_byte_active_sum_pin` (this constraint, 18) forces
///      `INV_INPUT_LEN_COL = Σ_{b=0..256} IS_BYTE_ACTIVE[b]`.
///   3. Each `IS_BYTE_ACTIVE[b]` ∈ {0,1} so the sum is an integer in
///      `[0, 256]` (= 257 possible values, fits in 9 bits).
///   4. The aggregator-invariance shifted constraint (body 1, see
///      `evaluate_shifted_at_point`) pins `INV_INPUT_LEN_COL` constant
///      across all rows of an invocation, so the bound from row 0 (the
///      `IS_FIRST_INV_ROW=1` anchor) propagates to every row.
///
/// The cross-AIR LogUp multiplicity check ensures `IS_FIRST_INV_ROW=1`
/// fires on at least one row per invocation (any extract row's tuple
/// must have a matching anchor on the bit-level side). With the
/// linkage active, `INV_INPUT_LEN_COL ∈ [0, 256]` is fully algebraic.
///
/// For STANDALONE Keccak proofs (no linkage), a malicious prover could
/// set `IS_FIRST_INV_ROW = 0` on every row, gating off all four
/// constraints (16, 17, 18, and the aggregator-invariance shifted body)
/// — leaving `INV_INPUT_LEN_COL` unconstrained. This is a non-issue in
/// practice because Keccak is always linked via cross-AIR LogUp.
fn eval_is_byte_active_sum_pin_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let mut sum = Scalar::zero(curve);
    for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
        sum = sum.add(&cols[crate::keccak_air::is_byte_active(b)]);
    }
    let len = &cols[crate::keccak_air::COL_INV_INPUT_LEN_COL];
    anchor.mul(&sum.sub(len))
}

/// 19. inv_input_byte_binding (β-RLC over b ∈ 0..RATE_LEN):
///     `IS_FIRST_INV_ROW · Σ_b β^b · is_byte_active[b] ·
///        (INV_INPUT_BYTE[b] − Σ_{k=0..8} BEFORE[(x,y, 8(b%8)+k)] · 2^k) = 0`.
/// where `lane_idx = b/8`, `x = lane_idx%5`, `y = lane_idx/5`.
///
/// On the row-0 anchor, for each in-range byte position `b <
/// INV_INPUT_LEN_COL`, pins the witnessed input byte at position `b`
/// to the corresponding rate-byte decomposition of the BEFORE state.
/// For `b >= INV_INPUT_LEN_COL` (where the BEFORE state contains
/// keccak padding bytes), the gate `is_byte_active[b] = 0` makes the
/// body trivially vanish, so `INV_INPUT_BYTE[b]` (= 0 by witness
/// padding) does not need to match the rate's padding bytes.
/// 19b. is_last_block_binary: `IS_LAST_BLOCK · (IS_LAST_BLOCK − 1) = 0`.
fn eval_is_last_block_binary_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let v = &cols[crate::keccak_air::COL_IS_LAST_BLOCK];
    v.mul(&v.sub(&one))
}

/// 20. inv_output_byte_binding (β-RLC over b ∈ 0..INV_OUTPUT_LEN):
///     `IS_LAST_BLOCK · SEL_ROUND[NUM_ROUNDS-1] · Σ_b β^b ·
///        (INV_OUTPUT_BYTE[b] − Σ_{k=0..8} AFTER_IOTA[(x,y, 8(b%8)+k)] · 2^k) = 0`.
/// where `lane_idx = b/8`, `x = lane_idx%5`, `y = lane_idx/5`.
///
/// At the LAST round of the LAST absorption block (gated by
/// `IS_LAST_BLOCK · SEL_ROUND[NUM_ROUNDS-1]` — fires only on
/// aggregator-populated traces), pins the 32 output digest bytes to
/// the little-endian decomposition of the first 4 lanes (256 bits) of
/// the `AFTER_IOTA` state. For Keccak-256, the digest is the first
/// 256 bits of the post-permutation state at the squeeze step, which
/// is exactly `AFTER_IOTA` at the last round of the last absorption
/// permutation.
///
/// **Soundness for multi-absorption**: the gate uses `IS_LAST_BLOCK`
/// (not `IS_FIRST_BLOCK`), so the binding correctly fires at the
/// LAST permutation regardless of how many absorption blocks the
/// trace has. For single-absorption (1 permutation), `IS_LAST_BLOCK
/// = IS_FIRST_BLOCK` so the binding fires at row 23 (same as before).
/// For multi-absorption (N permutations), it fires at row 24N-1.
///
/// Legacy traces (no aggregator population) have `IS_LAST_BLOCK = 0`
/// throughout (alloc_trace zeros), so the gate vanishes.
fn eval_inv_output_byte_binding_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let last_block = &cols[crate::keccak_air::COL_IS_LAST_BLOCK];
    let last_sel = &cols[sel_round(NUM_ROUNDS - 1)];
    let gate = last_block.mul(last_sel);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for b in 0..crate::keccak_air::INV_OUTPUT_LEN {
        let lane_idx = b / 8;
        let inner_offset = 8 * (b % 8);
        let x = lane_idx % 5;
        let y = lane_idx / 5;
        let mut rate_byte = Scalar::zero(curve);
        let mut pow = Scalar::one(curve);
        for k in 0..8 {
            let bit_col = &cols[after_iota(x, y, inner_offset + k)];
            rate_byte = rate_byte.add(&pow.mul(bit_col));
            pow = pow.mul(&two);
        }
        let inv_byte = &cols[crate::keccak_air::inv_output_byte(b)];
        let body = inv_byte.sub(&rate_byte);
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    gate.mul(&acc)
}

/// 22. is_byte_active_off_anchor_zero (β-RLC over `b ∈ 0..NUM_IS_BYTE_ACTIVE`):
///     `(1 − IS_FIRST_INV_ROW) · is_byte_active[b] = 0`.
fn eval_is_byte_active_off_anchor_zero_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let one = Scalar::one(curve);
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let neg_anchor = one.sub(anchor);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
        let v = &cols[crate::keccak_air::is_byte_active(b)];
        let body = neg_anchor.mul(v);
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    acc
}

/// 23. sel_one_when_first_block:
///     `IS_FIRST_BLOCK · (Σ_k sel_round_k − 1) = 0`.
fn eval_sel_one_when_first_block_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let one = Scalar::one(curve);
    let fb = &cols[crate::keccak_air::COL_IS_FIRST_BLOCK];
    let mut sum = Scalar::zero(curve);
    for k in 0..NUM_SEL_ROUND {
        sum = sum.add(&cols[sel_round(k)]);
    }
    fb.mul(&sum.sub(&one))
}

/// 24. rc_word_pin: explicit algebraic pin of the round-constant lane.
///
/// Derive the committed RC word from the lane (0, 0) ι transition:
///   `derived_RC := Σ_bit 2^bit · XOR(after_iota[0][0][bit],
///                                    after_chi[0][0][bit])`
/// (because `after_iota[0][0] = after_chi[0][0] XOR RC[k]`, the bitwise
/// XOR of the two lanes recovers the RC lane bit-for-bit).
///
/// Then enforce, per row:
///   `Σ_k SEL_ROUND[k] · (derived_RC − ROUND_CONSTANTS[k]) = 0`.
///
/// Because `sel_binary` + `sel_sum_01` guarantee at most one selector is 1
/// per row (exactly one on active rows of block 0 via
/// `sel_one_when_first_block`), this pins the committed RC lane to the
/// FIPS 202 spec constant on every active round row.
fn eval_rc_word_pin_at_point(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let two = Scalar::from_u64(2, curve);
    // Compute derived_RC := Σ_bit 2^bit · XOR(after_iota[0][0][bit],
    //                                         after_chi[0][0][bit])
    let mut derived_rc = Scalar::zero(curve);
    let mut pow = Scalar::one(curve);
    for bit in 0..BITS_PER_LANE {
        let ai = &cols[after_iota(0, 0, bit)];
        let ac = &cols[after_chi(0, 0, bit)];
        let xor_bit = xor_scalar(ai, ac, &two);
        derived_rc = derived_rc.add(&pow.mul(&xor_bit));
        pow = pow.mul(&two);
    }
    // Σ_k SEL_ROUND[k] · (derived_RC − ROUND_CONSTANTS[k])
    let mut acc = Scalar::zero(curve);
    for k in 0..NUM_ROUNDS {
        let sel = &cols[sel_round(k)];
        let rc_k = Scalar::from_u64(ROUND_CONSTANTS[k], curve);
        let diff = derived_rc.sub(&rc_k);
        acc = acc.add(&sel.mul(&diff));
    }
    acc
}

fn eval_inv_input_byte_binding_at_point(cols: &[Scalar], beta: &Scalar) -> Scalar {
    let curve = beta.curve_type();
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let two = Scalar::from_u64(2, curve);
    let mut acc = Scalar::zero(curve);
    let mut bp = Scalar::one(curve);
    for b in 0..crate::keccak_air::RATE_LEN {
        let lane_idx = b / 8;
        let inner_offset = 8 * (b % 8);
        let x = lane_idx % 5;
        let y = lane_idx / 5;
        let mut rate_byte = Scalar::zero(curve);
        let mut pow = Scalar::one(curve);
        for k in 0..8 {
            let bit_col = &cols[before(x, y, inner_offset + k)];
            rate_byte = rate_byte.add(&pow.mul(bit_col));
            pow = pow.mul(&two);
        }
        let inv_byte = &cols[crate::keccak_air::inv_input_byte(b)];
        let active = &cols[crate::keccak_air::is_byte_active(b)];
        let body = active.mul(&inv_byte.sub(&rate_byte));
        acc = acc.add(&bp.mul(&body));
        bp = bp.mul(beta);
    }
    anchor.mul(&acc)
}

// ──── Polynomial-form builders for each category body ──────────────────
//
// Each returns the coefficient-form polynomial representing the category's
// β-aggregated body. The outer caller combines them with α-powers.

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

fn build_c_partial_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for bit in 0..BITS_PER_LANE {
            // Initial: C_partial[x][0][bit] − before[x][0][bit]
            let cp0 = &cols[c_partial(x, 0, bit)];
            let b0 = &cols[before(x, 0, bit)];
            let body = poly_sub(cp0, b0, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
            bp = bp.mul(beta);
            for y in 0..4 {
                let a = &cols[c_partial(x, y, bit)];
                let b = &cols[before(x, y + 1, bit)];
                let ab = poly_mul(a, b, curve);
                let two_ab = poly_scalar_mul(&ab, &two);
                let xor = poly_sub(&poly_add(a, b, curve), &two_ab, curve);
                let cp_next = &cols[c_partial(x, y + 1, bit)];
                let body = poly_sub(cp_next, &xor, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

fn build_d_def_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for bit in 0..BITS_PER_LANE {
            let cl = &cols[c_final((x + 4) % 5, bit)];
            let cr = &cols[c_final((x + 1) % 5, (bit + 63) % BITS_PER_LANE)];
            let d = &cols[d_col(x, bit)];
            let ab = poly_mul(cl, cr, curve);
            let xor = poly_sub(&poly_add(cl, cr, curve), &poly_scalar_mul(&ab, &two), curve);
            let body = poly_sub(d, &xor, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
            bp = bp.mul(beta);
        }
    }
    acc
}

fn build_theta_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let b = &cols[before(x, y, bit)];
                let d = &cols[d_col(x, bit)];
                let at = &cols[after_theta(x, y, bit)];
                let bd = poly_mul(b, d, curve);
                let xor = poly_sub(&poly_add(b, d, curve), &poly_scalar_mul(&bd, &two), curve);
                let body = poly_sub(at, &xor, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

fn build_rho_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            let off = RHO_OFFSETS[x][y] as usize;
            for bit in 0..BITS_PER_LANE {
                let src_bit = (bit + BITS_PER_LANE - off) % BITS_PER_LANE;
                let at = &cols[after_theta(x, y, src_bit)];
                let ar = &cols[after_rho(x, y, bit)];
                let body = poly_sub(ar, at, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

fn build_pi_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            let nx = y;
            let ny = (2 * x + 3 * y) % 5;
            for bit in 0..BITS_PER_LANE {
                let ar = &cols[after_rho(x, y, bit)];
                let ap = &cols[after_pi(nx, ny, bit)];
                let body = poly_sub(ap, ar, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

fn build_chi_nand_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let ap1 = &cols[after_pi((x + 1) % 5, y, bit)];
                let ap2 = &cols[after_pi((x + 2) % 5, y, bit)];
                let nt = &cols[nand_temp(x, y, bit)];
                let not_ap1 = poly_sub(&one_poly, ap1, curve);
                let prod = poly_mul(&not_ap1, ap2, curve);
                let body = poly_sub(nt, &prod, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

fn build_chi_apply_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            for bit in 0..BITS_PER_LANE {
                let ap = &cols[after_pi(x, y, bit)];
                let nt = &cols[nand_temp(x, y, bit)];
                let ac = &cols[after_chi(x, y, bit)];
                let prod = poly_mul(ap, nt, curve);
                let xor = poly_sub(
                    &poly_add(ap, nt, curve),
                    &poly_scalar_mul(&prod, &two),
                    curve,
                );
                let body = poly_sub(ac, &xor, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

fn build_iota_passthrough_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for x in 0..5 {
        for y in 0..5 {
            if x == 0 && y == 0 {
                continue;
            }
            for bit in 0..BITS_PER_LANE {
                let ac = &cols[after_chi(x, y, bit)];
                let ai = &cols[after_iota(x, y, bit)];
                let body = poly_sub(ai, ac, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
                bp = bp.mul(beta);
            }
        }
    }
    acc
}

fn build_iota_xor_rc_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for k in 0..NUM_ROUNDS {
        let sel = &cols[sel_round(k)];
        let rc = ROUND_CONSTANTS[k];
        for bit in 0..BITS_PER_LANE {
            let rc_bit = ((rc >> bit) & 1) as u64;
            let ac = &cols[after_chi(0, 0, bit)];
            let ai = &cols[after_iota(0, 0, bit)];
            // xor = ac if rc_bit=0, else 1 − ac
            let xor = if rc_bit == 1 {
                poly_sub(&one_poly, ac, curve)
            } else {
                ac.clone()
            };
            let diff = poly_sub(ai, &xor, curve);
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

fn build_sel_sum_01_poly(cols: &[Vec<Scalar>]) -> Vec<Scalar> {
    let curve = cols[0][0].curve_type();
    let mut sum = vec![Scalar::zero(curve)];
    for k in 0..NUM_SEL_ROUND {
        sum = poly_add(&sum, &cols[sel_round(k)], curve);
    }
    // sum − 1
    let mut sum_m1 = sum.clone();
    if sum_m1.is_empty() {
        sum_m1.push(Scalar::zero(curve));
    }
    sum_m1[0] = sum_m1[0].sub(&Scalar::one(curve));
    poly_mul(&sum, &sum_m1, curve)
}

fn build_is_first_inv_row_binary_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_first_block_binary_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[crate::keccak_air::COL_IS_FIRST_BLOCK];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_is_first_block_pinned_at_anchor_poly(
    cols: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let fb = &cols[crate::keccak_air::COL_IS_FIRST_BLOCK];
    let one_minus_fb = poly_sub(&one_poly, fb, curve);
    poly_mul(anchor, &one_minus_fb, curve)
}

fn build_is_byte_active_binary_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
        let v = &cols[crate::keccak_air::is_byte_active(b)];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    poly_mul(anchor, &acc, curve)
}

fn build_is_byte_active_monotone_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for b in 1..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
        let v_curr = &cols[crate::keccak_air::is_byte_active(b)];
        let v_prev = &cols[crate::keccak_air::is_byte_active(b - 1)];
        let one_minus_prev = poly_sub(&one_poly, v_prev, curve);
        let body = poly_mul(v_curr, &one_minus_prev, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    poly_mul(anchor, &acc, curve)
}

fn build_is_byte_active_sum_pin_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
        sum = poly_add(&sum, &cols[crate::keccak_air::is_byte_active(b)], curve);
    }
    let len = &cols[crate::keccak_air::COL_INV_INPUT_LEN_COL];
    let diff = poly_sub(&sum, len, curve);
    poly_mul(anchor, &diff, curve)
}

fn build_is_last_block_binary_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let v = &cols[crate::keccak_air::COL_IS_LAST_BLOCK];
    let v_m1 = poly_sub(v, &one_poly, curve);
    poly_mul(v, &v_m1, curve)
}

fn build_inv_output_byte_binding_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let last_block = &cols[crate::keccak_air::COL_IS_LAST_BLOCK];
    let last_sel = &cols[sel_round(NUM_ROUNDS - 1)];
    let gate = poly_mul(last_block, last_sel, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for b in 0..crate::keccak_air::INV_OUTPUT_LEN {
        let lane_idx = b / 8;
        let inner_offset = 8 * (b % 8);
        let x = lane_idx % 5;
        let y = lane_idx / 5;
        let mut rate_byte = vec![Scalar::zero(curve)];
        let mut pow = Scalar::one(curve);
        for k in 0..8 {
            let bit_col = &cols[after_iota(x, y, inner_offset + k)];
            rate_byte = poly_add(&rate_byte, &poly_scalar_mul(bit_col, &pow), curve);
            pow = pow.mul(&two);
        }
        let inv_byte = &cols[crate::keccak_air::inv_output_byte(b)];
        let body = poly_sub(inv_byte, &rate_byte, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    poly_mul(&gate, &acc, curve)
}

/// 22. is_byte_active_off_anchor_zero polynomial form.
fn build_is_byte_active_off_anchor_zero_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let one_poly = vec![Scalar::one(curve)];
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let neg_anchor = poly_sub(&one_poly, anchor, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
        let v = &cols[crate::keccak_air::is_byte_active(b)];
        let body = poly_mul(&neg_anchor, v, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    acc
}

/// 23. sel_one_when_first_block polynomial form.
fn build_sel_one_when_first_block_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let fb = &cols[crate::keccak_air::COL_IS_FIRST_BLOCK];
    let mut sum = vec![Scalar::zero(curve)];
    for k in 0..NUM_SEL_ROUND {
        sum = poly_add(&sum, &cols[sel_round(k)], curve);
    }
    let sum_minus_one = poly_sub(&sum, &one_poly, curve);
    poly_mul(fb, &sum_minus_one, curve)
}

/// 24. rc_word_pin polynomial. Mirrors [`eval_rc_word_pin_at_point`].
fn build_rc_word_pin_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let two = Scalar::from_u64(2, curve);
    // derived_RC := Σ_bit 2^bit · XOR(after_iota[0][0][bit],
    //                                  after_chi[0][0][bit])
    let mut derived_rc = vec![Scalar::zero(curve)];
    let mut pow = Scalar::one(curve);
    for bit in 0..BITS_PER_LANE {
        let ai = &cols[after_iota(0, 0, bit)];
        let ac = &cols[after_chi(0, 0, bit)];
        // XOR = ai + ac − 2·ai·ac
        let prod = poly_mul(ai, ac, curve);
        let two_prod = poly_scalar_mul(&prod, &two);
        let xor_bit = poly_sub(&poly_add(ai, ac, curve), &two_prod, curve);
        derived_rc = poly_add(&derived_rc, &poly_scalar_mul(&xor_bit, &pow), curve);
        pow = pow.mul(&two);
    }
    // Σ_k SEL_ROUND[k] · (derived_RC − ROUND_CONSTANTS[k])
    let mut acc = vec![Scalar::zero(curve)];
    for k in 0..NUM_ROUNDS {
        let sel = &cols[sel_round(k)];
        let rc_k = Scalar::from_u64(ROUND_CONSTANTS[k], curve);
        // diff = derived_rc − rc_k (subtract from constant term)
        let mut diff = derived_rc.clone();
        if diff.is_empty() {
            diff.push(Scalar::zero(curve));
        }
        diff[0] = diff[0].sub(&rc_k);
        let body = poly_mul(sel, &diff, curve);
        acc = poly_add(&acc, &body, curve);
    }
    acc
}

fn build_inv_input_byte_binding_poly(cols: &[Vec<Scalar>], beta: &Scalar) -> Vec<Scalar> {
    let curve = beta.curve_type();
    let two = Scalar::from_u64(2, curve);
    let anchor = &cols[crate::keccak_air::COL_IS_FIRST_INV_ROW];
    let mut acc = vec![Scalar::zero(curve)];
    let mut bp = Scalar::one(curve);
    for b in 0..crate::keccak_air::RATE_LEN {
        let lane_idx = b / 8;
        let inner_offset = 8 * (b % 8);
        let x = lane_idx % 5;
        let y = lane_idx / 5;
        let mut rate_byte = vec![Scalar::zero(curve)];
        let mut pow = Scalar::one(curve);
        for k in 0..8 {
            let bit_col = &cols[before(x, y, inner_offset + k)];
            rate_byte = poly_add(&rate_byte, &poly_scalar_mul(bit_col, &pow), curve);
            pow = pow.mul(&two);
        }
        let inv_byte = &cols[crate::keccak_air::inv_input_byte(b)];
        let diff = poly_sub(inv_byte, &rate_byte, curve);
        let active = &cols[crate::keccak_air::is_byte_active(b)];
        let body = poly_mul(active, &diff, curve);
        acc = poly_add(&acc, &poly_scalar_mul(&body, &bp), curve);
        bp = bp.mul(beta);
    }
    poly_mul(anchor, &acc, curve)
}

// ──── Cross-row helpers ────────────────────────────────────────────────

/// Boundary rows whose cross-row transition must be excluded from
/// vanishing. For a trace with real rows `0..num_rows` padded to
/// `domain_size`, these are:
///   - Row index `23, 47, 71, …, num_rows-1` (end of each permutation).
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
/// — permutation boundaries are NOT excluded because the per-invocation
/// aggregator columns must remain constant across them within one
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
/// repeated squaring. Mirrors the helper in [`crate::mpt_constraints`]
/// and [`crate::sha256_constraints`].
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

impl VmConstraintSystem for KeccakConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "bit_validity".into(),
            "c_partial_recurrence".into(),
            "d_definition".into(),
            "theta_apply".into(),
            "rho_apply".into(),
            "pi_apply".into(),
            "chi_nand_definition".into(),
            "chi_apply".into(),
            "iota_passthrough".into(),
            "iota_xor_rc".into(),
            "sel_binary".into(),
            "sel_sum_01".into(),
            "is_first_inv_row_binary".into(),
            "is_first_block_binary".into(),
            "is_first_block_pinned_at_anchor".into(),
            "is_byte_active_binary".into(),
            "is_byte_active_monotone".into(),
            "is_byte_active_sum_pin".into(),
            "inv_input_byte_binding".into(),
            "is_last_block_binary".into(),
            "inv_output_byte_binding".into(),
            "is_byte_active_off_anchor_zero".into(),
            "sel_one_when_first_block".into(),
            "rc_word_pin".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(
            columns.len() == NUM_KECCAK_COLUMNS,
            "keccak AIR expects {} columns", NUM_KECCAK_COLUMNS
        );
        // Use a fixed non-zero β so the returned evaluations are deterministic.
        // Because each sub-constraint is zero on every row of a valid witness,
        // the choice of β does not affect soundness of *this* helper; the
        // actual prove/verify pipeline does not call this method when
        // `selector_column_indices()` is non-empty.
        let curve = columns[0][0].curve_type();
        let beta = Scalar::from_u64(2, curve);
        let mut evals: Vec<Vec<Scalar>> = keccak_air::evaluate_constraints(columns, &beta)
            .into_iter()
            .map(|c| c.values)
            .collect();
        let n = columns[0].len();
        let one = Scalar::one(curve);
        // 13th: is_first_inv_row_binary.
        let mut anchor_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let v = &columns[crate::keccak_air::COL_IS_FIRST_INV_ROW][row];
            anchor_bin[row] = v.mul(&v.sub(&one));
        }
        evals.push(anchor_bin);
        // 14th: is_first_block_binary.
        let mut fb_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let v = &columns[crate::keccak_air::COL_IS_FIRST_BLOCK][row];
            fb_bin[row] = v.mul(&v.sub(&one));
        }
        evals.push(fb_bin);
        // 15th: is_first_block_pinned_at_anchor.
        let mut fb_pin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let anchor = &columns[crate::keccak_air::COL_IS_FIRST_INV_ROW][row];
            let fb = &columns[crate::keccak_air::COL_IS_FIRST_BLOCK][row];
            fb_pin[row] = anchor.mul(&one.sub(fb));
        }
        evals.push(fb_pin);
        // 16th: is_byte_active_binary (per-row, β = 2).
        let two = Scalar::from_u64(2, curve);
        let mut active_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let anchor = &columns[crate::keccak_air::COL_IS_FIRST_INV_ROW][row];
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
                let v = &columns[crate::keccak_air::is_byte_active(b)][row];
                let body = v.mul(&v.sub(&one));
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&two);
            }
            active_bin[row] = anchor.mul(&acc);
        }
        evals.push(active_bin);
        // 17th: is_byte_active_monotone.
        let mut active_mono = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let anchor = &columns[crate::keccak_air::COL_IS_FIRST_INV_ROW][row];
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for b in 1..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
                let v_curr = &columns[crate::keccak_air::is_byte_active(b)][row];
                let v_prev = &columns[crate::keccak_air::is_byte_active(b - 1)][row];
                let body = v_curr.mul(&one.sub(v_prev));
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&two);
            }
            active_mono[row] = anchor.mul(&acc);
        }
        evals.push(active_mono);
        // 18th: is_byte_active_sum_pin.
        let mut active_sum = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let anchor = &columns[crate::keccak_air::COL_IS_FIRST_INV_ROW][row];
            let mut sum = Scalar::zero(curve);
            for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
                sum = sum.add(&columns[crate::keccak_air::is_byte_active(b)][row]);
            }
            let len = &columns[crate::keccak_air::COL_INV_INPUT_LEN_COL][row];
            active_sum[row] = anchor.mul(&sum.sub(len));
        }
        evals.push(active_sum);
        // 19th: inv_input_byte_binding.
        let mut binding = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let anchor = &columns[crate::keccak_air::COL_IS_FIRST_INV_ROW][row];
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
                let lane_idx = b / 8;
                let inner_offset = 8 * (b % 8);
                let x = lane_idx % 5;
                let y = lane_idx / 5;
                let mut rate_byte = Scalar::zero(curve);
                let mut pow = Scalar::one(curve);
                for k in 0..8 {
                    let bit_col = &columns[before(x, y, inner_offset + k)][row];
                    rate_byte = rate_byte.add(&pow.mul(bit_col));
                    pow = pow.mul(&two);
                }
                let inv_byte = &columns[crate::keccak_air::inv_input_byte(b)][row];
                let active = &columns[crate::keccak_air::is_byte_active(b)][row];
                let body = active.mul(&inv_byte.sub(&rate_byte));
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&two);
            }
            binding[row] = anchor.mul(&acc);
        }
        evals.push(binding);
        // 20th: is_last_block_binary.
        let mut last_bin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let v = &columns[crate::keccak_air::COL_IS_LAST_BLOCK][row];
            last_bin[row] = v.mul(&v.sub(&one));
        }
        evals.push(last_bin);
        // 21st: inv_output_byte_binding.
        let mut out_binding = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let last_block = &columns[crate::keccak_air::COL_IS_LAST_BLOCK][row];
            let last_sel = &columns[sel_round(NUM_ROUNDS - 1)][row];
            let gate = last_block.mul(last_sel);
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for b in 0..crate::keccak_air::INV_OUTPUT_LEN {
                let lane_idx = b / 8;
                let inner_offset = 8 * (b % 8);
                let x = lane_idx % 5;
                let y = lane_idx / 5;
                let mut rate_byte = Scalar::zero(curve);
                let mut pow = Scalar::one(curve);
                for k in 0..8 {
                    let bit_col = &columns[after_iota(x, y, inner_offset + k)][row];
                    rate_byte = rate_byte.add(&pow.mul(bit_col));
                    pow = pow.mul(&two);
                }
                let inv_byte = &columns[crate::keccak_air::inv_output_byte(b)][row];
                let body = inv_byte.sub(&rate_byte);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&two);
            }
            out_binding[row] = gate.mul(&acc);
        }
        evals.push(out_binding);
        // 22nd: is_byte_active_off_anchor_zero.
        let mut byte_off = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let anchor = &columns[crate::keccak_air::COL_IS_FIRST_INV_ROW][row];
            let neg_anchor = one.sub(anchor);
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for b in 0..crate::keccak_air::NUM_IS_BYTE_ACTIVE {
                let v = &columns[crate::keccak_air::is_byte_active(b)][row];
                let body = neg_anchor.mul(v);
                acc = acc.add(&bp.mul(&body));
                bp = bp.mul(&beta);
            }
            byte_off[row] = acc;
        }
        evals.push(byte_off);
        // 23rd: sel_one_when_first_block.
        let mut sel_one_fb = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let fb = &columns[crate::keccak_air::COL_IS_FIRST_BLOCK][row];
            let mut sum = Scalar::zero(curve);
            for k in 0..NUM_SEL_ROUND {
                sum = sum.add(&columns[sel_round(k)][row]);
            }
            sel_one_fb[row] = fb.mul(&sum.sub(&one));
        }
        evals.push(sel_one_fb);
        // 24th: rc_word_pin — derived_RC = Σ 2^bit · XOR(after_iota[0][0][bit],
        // after_chi[0][0][bit]); pin = Σ_k SEL_ROUND[k] · (derived_RC −
        // ROUND_CONSTANTS[k]).
        let mut rc_pin = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let mut derived_rc = Scalar::zero(curve);
            let mut pow = Scalar::one(curve);
            for bit in 0..BITS_PER_LANE {
                let ai = &columns[after_iota(0, 0, bit)][row];
                let ac = &columns[after_chi(0, 0, bit)][row];
                let xor_bit = xor_scalar(ai, ac, &two);
                derived_rc = derived_rc.add(&pow.mul(&xor_bit));
                pow = pow.mul(&two);
            }
            let mut acc = Scalar::zero(curve);
            for k in 0..NUM_ROUNDS {
                let sel = &columns[sel_round(k)][row];
                let rc_k = Scalar::from_u64(ROUND_CONSTANTS[k], curve);
                let diff = derived_rc.sub(&rc_k);
                acc = acc.add(&sel.mul(&diff));
            }
            rc_pin[row] = acc;
        }
        evals.push(rc_pin);
        evals
    }

    fn evaluate_at_point(&self, col_evals_at_z: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals_at_z.len() < NUM_KECCAK_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let beta = alpha; // β = α as noted in the module docs.
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            eval_bit_validity_at_point(col_evals_at_z, beta),
            eval_c_partial_at_point(col_evals_at_z, beta),
            eval_d_def_at_point(col_evals_at_z, beta),
            eval_theta_at_point(col_evals_at_z, beta),
            eval_rho_at_point(col_evals_at_z, beta),
            eval_pi_at_point(col_evals_at_z, beta),
            eval_chi_nand_at_point(col_evals_at_z, beta),
            eval_chi_apply_at_point(col_evals_at_z, beta),
            eval_iota_passthrough_at_point(col_evals_at_z, beta),
            eval_iota_xor_rc_at_point(col_evals_at_z, beta),
            eval_sel_binary_at_point(col_evals_at_z, beta),
            eval_sel_sum_01_at_point(col_evals_at_z),
            eval_is_first_inv_row_binary_at_point(col_evals_at_z),
            eval_is_first_block_binary_at_point(col_evals_at_z),
            eval_is_first_block_pinned_at_anchor_at_point(col_evals_at_z),
            eval_is_byte_active_binary_at_point(col_evals_at_z, beta),
            eval_is_byte_active_monotone_at_point(col_evals_at_z, beta),
            eval_is_byte_active_sum_pin_at_point(col_evals_at_z),
            eval_inv_input_byte_binding_at_point(col_evals_at_z, beta),
            eval_is_last_block_binary_at_point(col_evals_at_z),
            eval_inv_output_byte_binding_at_point(col_evals_at_z, beta),
            eval_is_byte_active_off_anchor_zero_at_point(col_evals_at_z, beta),
            eval_sel_one_when_first_block_at_point(col_evals_at_z),
            eval_rc_word_pin_at_point(col_evals_at_z),
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

    /// Return `None`: padding rows carry all-zero bit columns (including
    /// all selectors), which makes every row-local body vanish. The
    /// `sel_sum_01` constraint `sum·(sum−1) = 0` is satisfied with sum=0.
    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        // Explicitly zero every Keccak column on padding rows. The trace
        // constructor should already do this, but we guard against any
        // prior stage having set a non-zero value there.
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        if columns.len() < NUM_KECCAK_COLUMNS {
            return;
        }
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_KECCAK_COLUMNS) {
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
        // Build the 12 category polynomials in parallel.
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
                move || build_c_partial_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_d_def_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_theta_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_rho_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_pi_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_chi_nand_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_chi_apply_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_iota_passthrough_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_iota_xor_rc_poly(&cols, &b)
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
                move || build_is_byte_active_binary_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_is_byte_active_monotone_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_is_byte_active_sum_pin_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_inv_input_byte_binding_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_is_last_block_binary_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_inv_output_byte_binding_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let b = beta.clone();
                move || build_is_byte_active_off_anchor_zero_poly(&cols, &b)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_sel_one_when_first_block_poly(&cols, c)
            }),
            Box::new({
                let cols = column_coeffs.to_vec();
                let c = curve;
                move || build_rc_word_pin_poly(&cols, c)
            }),
        ];
        let bodies: Vec<Vec<Scalar>> = category_builders
            .par_iter()
            .map(|f| f())
            .collect();

        // Combine with α-powers: Σ α^k · body_k
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
        // Body 0: 1600 BEFORE bits (after_iota → before binding).
        // Body 1: 289 aggregator (INV_INPUT_BYTE + INV_INPUT_LEN_COL + INV_OUTPUT_BYTE) invariance.
        // Body 2: 1 IS_FIRST_BLOCK invariance.
        // Body 3: BEFORE bits at the rate lanes (1088 bits =
        //         17 lanes × 64 bits) referenced shifted for the
        //         block-1 input binding XOR. (Already part of body 0's
        //         BEFORE bits — we DO NOT re-add them; we just
        //         reference them from `shifted_evals[0..1600]`.)
        // Total: 1890 columns. Order matters.
        let mut idxs: Vec<usize> = (COL_BEFORE_OFFSET..COL_BEFORE_OFFSET + BITS_PER_STATE).collect();
        for b in 0..crate::keccak_air::INV_INPUT_LEN {
            idxs.push(crate::keccak_air::inv_input_byte(b));
        }
        idxs.push(crate::keccak_air::COL_INV_INPUT_LEN_COL);
        for b in 0..crate::keccak_air::INV_OUTPUT_LEN {
            idxs.push(crate::keccak_air::inv_output_byte(b));
        }
        idxs.push(crate::keccak_air::COL_IS_FIRST_BLOCK);
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
            + crate::keccak_air::INV_INPUT_LEN
            + 1
            + crate::keccak_air::INV_OUTPUT_LEN
            + 1; // IS_FIRST_BLOCK shifted
        if shifted_evals.len() != total_shifted {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();

        let beta = alpha; // β = α (matches row-local bodies).

        // ── Body 0: after_iota → before binding (existing) ──
        // shifted_evals[0..1600] is indexed by column offset within the
        // BEFORE block (= grid_bit_index(x, y, bit)) — NOT by iteration
        // order. The β-RLC iteration must read shifted_evals[grid_idx]
        // to match the build-side (which iterates the same way and
        // reads column_coeffs[before(x, y, bit)] = column at
        // COL_BEFORE_OFFSET + grid_bit_index).
        let mut body_0 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for x in 0..5 {
            for y in 0..5 {
                for bit in 0..BITS_PER_LANE {
                    let grid_idx = (y * 5 + x) * BITS_PER_LANE + bit;
                    let bf_shift = &shifted_evals[grid_idx];
                    let ai = &col_evals_at_z[after_iota(x, y, bit)];
                    body_0 = body_0.add(&bp.mul(&ai.sub(bf_shift)));
                    bp = bp.mul(beta);
                }
            }
        }

        // ── Body 1: aggregator invariance (new) ──
        // body_1 = Σ β^i · (INV_X(ω·z) − INV_X(z)) over INPUT bytes,
        // INPUT_LEN_COL, and OUTPUT bytes. shifted_evals[1600..1889] is
        // appended in the same linear order the build-side iterates,
        // so a flat shift_idx walking from 1600 upward matches.
        let mut body_1 = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        let mut shift_idx = BITS_PER_STATE;
        for b in 0..crate::keccak_air::INV_INPUT_LEN {
            let cur = &col_evals_at_z[crate::keccak_air::inv_input_byte(b)];
            let next = &shifted_evals[shift_idx];
            body_1 = body_1.add(&bp.mul(&next.sub(cur)));
            bp = bp.mul(beta);
            shift_idx += 1;
        }
        let cur_len = &col_evals_at_z[crate::keccak_air::COL_INV_INPUT_LEN_COL];
        let next_len = &shifted_evals[shift_idx];
        body_1 = body_1.add(&bp.mul(&next_len.sub(cur_len)));
        bp = bp.mul(beta);
        shift_idx += 1;
        for b in 0..crate::keccak_air::INV_OUTPUT_LEN {
            let cur = &col_evals_at_z[crate::keccak_air::inv_output_byte(b)];
            let next = &shifted_evals[shift_idx];
            body_1 = body_1.add(&bp.mul(&next.sub(cur)));
            bp = bp.mul(beta);
            shift_idx += 1;
        }

        // Boundary-row exclusion. Body 0 must match the prover-side
        // `boundary_rows` (every permutation boundary + wrap) for
        // multi-permutation soundness; the legacy verifier only excluded
        // last+wrap, which was sound only for single-permutation
        // traces. Body 1 (invariance) excludes only last+wrap because
        // the per-invocation aggregator must remain constant across
        // permutation boundaries within one invocation.
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

        // ── Body 2: IS_FIRST_BLOCK invariance within a permutation ──
        let one = Scalar::one(curve);
        let last_sel = &col_evals_at_z[sel_round(NUM_ROUNDS - 1)];
        let fb_curr = &col_evals_at_z[crate::keccak_air::COL_IS_FIRST_BLOCK];
        let fb_next = &shifted_evals[shift_idx]; // last entry
        let body_2 = one.sub(last_sel).mul(&fb_next.sub(fb_curr));
        let exclusion_2 = z.sub(omega_n_minus_1);

        // ── Body 3: Block-1 input binding (multi-absorption) ──
        // At row 23 (block 0's last round, transitioning to block 1):
        // for each rate byte position pos ∈ 0..136, pin
        //   IS_BYTE_ACTIVE[136+pos] · (INV_INPUT_BYTE[136+pos] −
        //     bit-decompose(BEFORE_at_row_24[lane,bits] XOR
        //                   AFTER_IOTA_at_row_23[lane,bits])) = 0
        // where lane = pos/8, x = lane%5, y = lane/5, inner_offset = 8*(pos%8).
        // Gate: IS_FIRST_BLOCK · SEL_ROUND[NUM_ROUNDS-1] · (1 − IS_LAST_BLOCK).
        // For 2-block: at row 23, IS_FIRST_BLOCK=1, last_sel=1,
        // IS_LAST_BLOCK=0 → gate fires. For 1-block: IS_LAST_BLOCK=1,
        // gate=0. For legacy: IS_FIRST_BLOCK=0, gate=0.
        let two = Scalar::from_u64(2, curve);
        let last_block = &col_evals_at_z[crate::keccak_air::COL_IS_LAST_BLOCK];
        let block_1_gate = fb_curr
            .mul(last_sel)
            .mul(&one.sub(last_block));
        // Block-1 binding scope: positions 136..NUM_IS_BYTE_ACTIVE (=
        // 256) of INV_INPUT_BYTE. Positions 120..136 within block 1
        // (= INV_INPUT_BYTE[256..272], if they existed) are out of
        // bounds; for inputs with length < 256, the padding bytes
        // past input_len are not bound (`IS_BYTE_ACTIVE = 0`). For
        // inputs of exactly 256 bytes, the keccak padding bytes
        // 0x01...0x80 occupy block 1 positions 120..136 (`pos = 120..136`
        // here), which we cannot bind because INV_INPUT_BYTE only
        // commits the original input. Documented gap.
        let block_1_pos_end =
            crate::keccak_air::NUM_IS_BYTE_ACTIVE - crate::keccak_air::RATE_LEN;
        let mut body_3 = Scalar::zero(curve);
        let mut bp3 = Scalar::one(curve);
        for pos in 0..block_1_pos_end {
            let lane_idx = pos / 8;
            let inner_offset = 8 * (pos % 8);
            let x = lane_idx % 5;
            let y = lane_idx / 5;
            let mut rate_byte = Scalar::zero(curve);
            let mut pow = Scalar::one(curve);
            for k in 0..8 {
                let grid_idx = (y * 5 + x) * BITS_PER_LANE + (inner_offset + k);
                let bf_next = &shifted_evals[grid_idx];
                let ai_curr = &col_evals_at_z[after_iota(x, y, inner_offset + k)];
                let xor = bf_next
                    .add(ai_curr)
                    .sub(&two.mul(&bf_next.mul(ai_curr)));
                rate_byte = rate_byte.add(&pow.mul(&xor));
                pow = pow.mul(&two);
            }
            let inv_byte = &col_evals_at_z[crate::keccak_air::inv_input_byte(
                crate::keccak_air::RATE_LEN + pos,
            )];
            let active = &col_evals_at_z[crate::keccak_air::is_byte_active(
                crate::keccak_air::RATE_LEN + pos,
            )];
            let body = active.mul(&inv_byte.sub(&rate_byte));
            body_3 = body_3.add(&bp3.mul(&body));
            bp3 = bp3.mul(beta);
        }
        let body_3 = block_1_gate.mul(&body_3);
        let exclusion_3 = z.sub(omega_n_minus_1);

        // α^alpha_offset for body_0; α^(alpha_offset+k) for body_k.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term_0 = ap.mul(&body_0).mul(&exclusion_0);
        ap = ap.mul(alpha);
        let term_1 = ap.mul(&body_1).mul(&exclusion_1);
        ap = ap.mul(alpha);
        let term_2 = ap.mul(&body_2).mul(&exclusion_2);
        ap = ap.mul(alpha);
        let term_3 = ap.mul(&body_3).mul(&exclusion_3);
        term_0.add(&term_1).add(&term_2).add(&term_3)
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

        // ── Body 0: after_iota(X) − before(ω·X) bit-by-bit, β-RLC ──
        let mut body_0 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for x in 0..5 {
            for y in 0..5 {
                for bit in 0..BITS_PER_LANE {
                    let ai = &column_coeffs[after_iota(x, y, bit)];
                    let bf = &column_coeffs[before(x, y, bit)];
                    let bf_shift = poly_shift(bf, omega);
                    let diff = poly_sub(ai, &bf_shift, curve);
                    let scaled = poly_scalar_mul(&diff, &bp);
                    body_0 = poly_add(&body_0, &scaled, curve);
                    bp = bp.mul(&beta);
                }
            }
        }
        let rows_0 = boundary_rows(self.num_rows, domain_size as usize);
        let mut excluded_0 = body_0;
        for r in &rows_0 {
            let omega_r = scalar_pow(omega, *r as u64);
            excluded_0 = poly_mul_linear(&excluded_0, &omega_r);
        }

        // ── Body 1: aggregator invariance — INV_X(ω·X) − INV_X(X) ──
        // β-RLC over 256 INPUT bytes + 1 INPUT_LEN_COL + 32 OUTPUT bytes.
        let mut body_1 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for b in 0..crate::keccak_air::INV_INPUT_LEN {
            let cur = &column_coeffs[crate::keccak_air::inv_input_byte(b)];
            let cur_shift = poly_shift(cur, omega);
            let diff = poly_sub(&cur_shift, cur, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            body_1 = poly_add(&body_1, &scaled, curve);
            bp = bp.mul(&beta);
        }
        let cur_len = &column_coeffs[crate::keccak_air::COL_INV_INPUT_LEN_COL];
        let cur_len_shift = poly_shift(cur_len, omega);
        let diff_len = poly_sub(&cur_len_shift, cur_len, curve);
        body_1 = poly_add(&body_1, &poly_scalar_mul(&diff_len, &bp), curve);
        bp = bp.mul(&beta);
        for b in 0..crate::keccak_air::INV_OUTPUT_LEN {
            let cur = &column_coeffs[crate::keccak_air::inv_output_byte(b)];
            let cur_shift = poly_shift(cur, omega);
            let diff = poly_sub(&cur_shift, cur, curve);
            let scaled = poly_scalar_mul(&diff, &bp);
            body_1 = poly_add(&body_1, &scaled, curve);
            bp = bp.mul(&beta);
        }
        // Body 1 exclusion: only last-real + wrap. Permutation
        // boundaries are NOT excluded so invariance holds across them
        // within an invocation.
        let rows_1 = invariance_boundary_rows(self.num_rows, domain_size as usize);
        let mut excluded_1 = body_1;
        for r in &rows_1 {
            let omega_r = scalar_pow(omega, *r as u64);
            excluded_1 = poly_mul_linear(&excluded_1, &omega_r);
        }

        // ── Body 2: IS_FIRST_BLOCK invariance within a permutation ──
        let one_poly = vec![Scalar::one(curve)];
        let last_sel = &column_coeffs[sel_round(NUM_ROUNDS - 1)];
        let one_minus_last_sel = poly_sub(&one_poly, last_sel, curve);
        let fb = &column_coeffs[crate::keccak_air::COL_IS_FIRST_BLOCK];
        let fb_shift = poly_shift(fb, omega);
        let fb_diff = poly_sub(&fb_shift, fb, curve);
        let body_2 = poly_mul(&one_minus_last_sel, &fb_diff, curve);
        let omega_wrap = scalar_pow(omega, (domain_size - 1) as u64);
        let excluded_2 = poly_mul_linear(&body_2, &omega_wrap);

        // ── Body 3: Block-1 input binding (multi-absorption) ──
        let two = Scalar::from_u64(2, curve);
        let last_block = &column_coeffs[crate::keccak_air::COL_IS_LAST_BLOCK];
        let one_minus_last_block = poly_sub(&one_poly, last_block, curve);
        let fb_last_sel = poly_mul(fb, last_sel, curve);
        let block_1_gate = poly_mul(&fb_last_sel, &one_minus_last_block, curve);
        let block_1_pos_end =
            crate::keccak_air::NUM_IS_BYTE_ACTIVE - crate::keccak_air::RATE_LEN;
        let mut body_3 = vec![Scalar::zero(curve)];
        let mut bp3 = Scalar::one(curve);
        for pos in 0..block_1_pos_end {
            let lane_idx = pos / 8;
            let inner_offset = 8 * (pos % 8);
            let x = lane_idx % 5;
            let y = lane_idx / 5;
            let mut rate_byte = vec![Scalar::zero(curve)];
            let mut pow = Scalar::one(curve);
            for k in 0..8 {
                let bf = &column_coeffs[before(x, y, inner_offset + k)];
                let bf_shift = poly_shift(bf, omega);
                let ai = &column_coeffs[after_iota(x, y, inner_offset + k)];
                let ab = poly_mul(&bf_shift, ai, curve);
                let two_ab = poly_scalar_mul(&ab, &two);
                let sum = poly_add(&bf_shift, ai, curve);
                let xor = poly_sub(&sum, &two_ab, curve);
                rate_byte = poly_add(&rate_byte, &poly_scalar_mul(&xor, &pow), curve);
                pow = pow.mul(&two);
            }
            let inv_byte = &column_coeffs[crate::keccak_air::inv_input_byte(
                crate::keccak_air::RATE_LEN + pos,
            )];
            let active = &column_coeffs[crate::keccak_air::is_byte_active(
                crate::keccak_air::RATE_LEN + pos,
            )];
            let diff = poly_sub(inv_byte, &rate_byte, curve);
            let body = poly_mul(active, &diff, curve);
            body_3 = poly_add(&body_3, &poly_scalar_mul(&body, &bp3), curve);
            bp3 = bp3.mul(&beta);
        }
        let body_3 = poly_mul(&block_1_gate, &body_3, curve);
        let excluded_3 = poly_mul_linear(&body_3, &omega_wrap);

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
        let mut sum = poly_add(&term_0, &term_1, curve);
        sum = poly_add(&sum, &term_2, curve);
        poly_add(&sum, &term_3, curve)
    }
}

// ──── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;
    use crate::keccak::{keccak_f1600_witness, State};
    use crate::scheme::CommitmentScheme;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    /// Build a columns vector for a single Keccak permutation (no padding).
    fn single_permutation_columns(state: State) -> (Vec<Vec<Scalar>>, Vec<RoundTrace>) {
        let rounds = keccak_f1600_witness(state);
        let mut columns = alloc_trace(NUM_ROUNDS, CurveType::Bls48581);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, CurveType::Bls48581);
        }
        (columns, rounds)
    }

    #[test]
    fn keccak_cs_labels_and_counts() {
        let cs = KeccakConstraintSystem::new(NUM_ROUNDS);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert_eq!(cs.num_shifted_constraints(), 4);
        assert_eq!(cs.selector_column_indices().len(), NUM_SEL_ROUND);
        // 1600 BEFORE bits (body 0; body 3 reuses these for the XOR
        // formula) + 256 INV_INPUT_BYTE + 1 INV_INPUT_LEN_COL + 32
        // INV_OUTPUT_BYTE (body 1 invariance) + 1 IS_FIRST_BLOCK
        // (body 2) = 1890 columns. Body 3's BEFORE bits are NOT
        // re-added — they share with body 0's BEFORE block.
        assert_eq!(
            cs.shifted_column_indices().len(),
            BITS_PER_STATE
                + crate::keccak_air::INV_INPUT_LEN
                + 1
                + crate::keccak_air::INV_OUTPUT_LEN
                + 1
        );
        assert!(cs.padding_selector_column().is_none());
    }

    #[test]
    #[ignore = "slow in debug: 12 categories × 13k columns × 24 rows; run with --release --ignored"]
    fn keccak_cs_evaluate_on_domain_matches_keccak_air() {
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0x0123_4567_89ab_cdef;
        state[2][3] = 0xdead_beef_cafe_babe;
        let (columns, _rounds) = single_permutation_columns(state);

        let cs = KeccakConstraintSystem::new(NUM_ROUNDS);
        let refs = col_refs(&columns);
        let evals = cs.evaluate_on_domain(&refs, NUM_ROUNDS);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        // Every per-row value must be zero for a valid witness.
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

    /// Sanity check: `evaluate_at_point` on a valid witness with
    /// `z = ω^row` recovers exactly the α-RLC of the row's per-category
    /// body values. Because every category body is zero on every real row,
    /// the combined value must also be zero.
    #[test]
    #[ignore = "slow in debug: evaluate_at_point × 24 rows ≈ 20M scalar ops; run with --release --ignored"]
    fn keccak_cs_evaluate_at_point_zero_on_real_rows() {
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0x0123_4567_89ab_cdef;
        state[1][4] = 0xaaaa_5555_aaaa_5555;
        let (columns, _rounds) = single_permutation_columns(state);

        let cs = KeccakConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(17, CurveType::Bls48581);

        // For each real row, assemble the column-evaluation vector and
        // compute C(z) at z = that row (since columns are in evaluation
        // form on the trace, cols[i][row] is the value at ω^row).
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

    /// Mutating a BEFORE bit should cause at least one category body to
    /// be non-zero at that row, which `evaluate_at_point` detects.
    #[test]
    fn keccak_cs_evaluate_at_point_detects_mutation() {
        let mut state: State = [[0u64; 5]; 5];
        state[3][2] = 0xbeef_face_dead_cafe;
        let (mut columns, _rounds) = single_permutation_columns(state);

        // Flip before(1,1,0) on row 2.
        let curve = CurveType::Bls48581;
        let one = Scalar::one(curve);
        columns[before(1, 1, 0)][2] = one.sub(&columns[before(1, 1, 0)][2].clone());

        let cs = KeccakConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(17, curve);
        let col_vals: Vec<Scalar> = columns.iter().map(|c| c[2].clone()).collect();
        let c_at = cs.evaluate_at_point(&col_vals, &alpha);
        assert!(
            !c_at.is_zero(),
            "mutation should make C(row 2) non-zero"
        );
    }

    /// Tampering test: option-(a) algebraic input binding catches a
    /// flipped INV_INPUT_BYTE at the row-0 anchor. With
    /// `is_byte_active[b] = 1` for b < input_len, the binding constraint
    /// `is_byte_active[b] · (INV_INPUT_BYTE[b] − rate_byte_at_b) = 0`
    /// at b=0 fires when INV_INPUT_BYTE[0] is mutated.
    #[test]
    fn keccak_cs_inv_input_byte_binding_rejects_tampered_anchor_byte() {
        use crate::keccak::keccak_witness;
        let curve = CurveType::Bls48581;
        let input = b"hello world".to_vec();
        let ht = keccak_witness(&input);
        let digest = crate::keccak::keccak256(&input);
        let mut columns = crate::keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, &digest, curve,
        );
        // Tamper INV_INPUT_BYTE[0] at row 0.
        let one = Scalar::one(curve);
        let col = crate::keccak_air::inv_input_byte(0);
        columns[col][0] = columns[col][0].add(&one);

        let cs = KeccakConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let alpha = Scalar::from_u64(23, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered INV_INPUT_BYTE at the binding-fires row must make C(row 0) non-zero"
        );
    }

    /// Tampering test: output binding catches a flipped INV_OUTPUT_BYTE
    /// at the binding row (row 23, gated by `IS_FIRST_BLOCK ·
    /// SEL_ROUND[NUM_ROUNDS-1]`). The body
    /// `INV_OUTPUT_BYTE[b] − rate_byte_at_b` becomes non-zero.
    #[test]
    fn keccak_cs_inv_output_byte_binding_rejects_tampered_digest_byte() {
        use crate::keccak::keccak_witness;
        let curve = CurveType::Bls48581;
        let input = b"hello world".to_vec();
        let ht = keccak_witness(&input);
        let digest = crate::keccak::keccak256(&input);
        let mut columns = crate::keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, &digest, curve,
        );
        // Tamper INV_OUTPUT_BYTE[0] at the binding row (row 23).
        let one = Scalar::one(curve);
        let col = crate::keccak_air::inv_output_byte(0);
        columns[col][NUM_ROUNDS - 1] = columns[col][NUM_ROUNDS - 1].add(&one);

        let cs = KeccakConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let alpha = Scalar::from_u64(31, curve);
        let row_vals: Vec<Scalar> =
            columns.iter().map(|c| c[NUM_ROUNDS - 1].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered INV_OUTPUT_BYTE at row 23 must make C(row 23) non-zero"
        );
    }

    /// Tampering test: option-(a) algebraic input binding catches a
    /// flipped IS_BYTE_ACTIVE at the row-0 anchor — the sum_pin
    /// constraint fires (sum of IS_BYTE_ACTIVE no longer equals
    /// INV_INPUT_LEN_COL).
    #[test]
    fn keccak_cs_is_byte_active_sum_pin_rejects_tampered_active_count() {
        use crate::keccak::keccak_witness;
        let curve = CurveType::Bls48581;
        let input = b"hello world".to_vec();
        let ht = keccak_witness(&input);
        let digest = crate::keccak::keccak256(&input);
        let mut columns = crate::keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, &digest, curve,
        );
        // Honest: IS_BYTE_ACTIVE[10] = 1 (since input_len = 11). Force
        // it to 0 — sum drops from 11 to 10, mismatching INV_INPUT_LEN.
        columns[crate::keccak_air::is_byte_active(10)][0] = Scalar::zero(curve);

        let cs = KeccakConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let alpha = Scalar::from_u64(29, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "dropping IS_BYTE_ACTIVE within input range must make C(row 0) non-zero \
             (sum_pin or monotone fires)"
        );
    }

    /// Tampering test: an aggregator-populated trace with
    /// IS_FIRST_BLOCK = 0 at row 1 (should be 1 throughout the 24 rounds
    /// of permutation 0). The within-permutation invariance shifted
    /// constraint at row 0→1 fires:
    /// `(1 − SEL_ROUND[NUM_ROUNDS-1](row 0)) · (IS_FIRST_BLOCK(row 1) −
    /// IS_FIRST_BLOCK(row 0)) = 1 · (0 − 1) = −1 ≠ 0`.
    #[test]
    fn keccak_cs_is_first_block_invariance_rejects_premature_drop() {
        use crate::keccak::keccak_witness;
        let curve = CurveType::Bls48581;
        let input = b"hello world".to_vec();
        let ht = keccak_witness(&input);
        let digest = crate::keccak::keccak256(&input);
        let mut columns = crate::keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, &digest, curve,
        );
        // Tamper: drop IS_FIRST_BLOCK to 0 at row 1.
        columns[crate::keccak_air::COL_IS_FIRST_BLOCK][1] = Scalar::zero(curve);

        let cs = KeccakConstraintSystem::new(ht.blocks.len() * NUM_ROUNDS);
        let alpha = Scalar::from_u64(19, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let mut shifted_evals = Vec::new();
        for &i in &cs.shifted_column_indices() {
            shifted_evals.push(columns[i][1].clone());
        }
        let z = Scalar::from_u64(103, curve);
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
            "tampered IS_FIRST_BLOCK premature drop must make body 2 non-zero"
        );
    }

    /// Build a small trace and exercise `build_constraint_polynomial`
    /// indirectly: evaluating the polynomial at every real domain point
    /// should match the row-wise `evaluate_at_point` values (which are all
    /// zero on a valid witness). For padded rows the polynomial need not
    /// be zero; we check only real rows.
    ///
    /// Marked `#[ignore]` — exercises tens of thousands of 32-coefficient
    /// polynomial multiplications over BLS48-581 big integers in debug
    /// mode, which runs for several minutes per test. Re-enable manually
    /// with `--release --ignored`.
    #[test]
    #[ignore = "too slow in debug: ~30k poly_mul ops; run with --release --ignored"]
    fn keccak_cs_build_constraint_polynomial_vanishes_on_real_rows() {
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0x1;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let n = trace.padded_size;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Materialise coefficients via IFFT, then evaluate combined constraint.
        let eval_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let coeff_form: Vec<Vec<Scalar>> = eval_form
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();

        let cs = KeccakConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(23, curve);
        let c_coeffs = cs.build_constraint_polynomial(&coeff_form, &alpha, n);

        // Evaluate C at ω^0, ω^1, …, ω^{num_rows-1} and assert zero.
        let omega = CommitmentScheme::domain_generator(&scheme, n);
        for row in 0..NUM_ROUNDS {
            let z = scalar_pow(&omega, row as u64);
            let v = CommitmentScheme::eval_poly_at(&scheme, &c_coeffs, &z);
            assert!(
                v.is_zero(),
                "C(ω^{}) not zero (but all row-local bodies should vanish there)",
                row
            );
        }
    }

    /// Localizes the keccak prove/verify failure: at a non-domain challenge
    /// `z`, evaluating the combined constraint via the polynomial form
    /// (`build_constraint_polynomial(coeffs, α)(z)`) must equal evaluating
    /// it via the per-row body (`evaluate_at_point(col(z), α)`). If they
    /// disagree, a per-category builder doesn't match its evaluator.
    #[test]
    #[ignore = "slow: needs trace IFFT + 12 poly_mul builds; run with --release --ignored"]
    fn keccak_cs_eval_at_z_matches_poly_at_z() {
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0x1;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let n = trace.padded_size;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eval_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let coeff_form: Vec<Vec<Scalar>> = eval_form
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();

        let cs = KeccakConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(23, curve);

        // Pick a non-domain z (just an arbitrary scalar that's almost
        // certainly not in the FFT domain).
        let z = Scalar::from_u64(0xCAFE_BEEF_DEAD_BABE, curve);

        // Path 1: evaluate every column polynomial at z, then run
        // evaluate_at_point.
        let col_at_z: Vec<Scalar> = coeff_form
            .iter()
            .map(|c| CommitmentScheme::eval_poly_at(&scheme, c, &z))
            .collect();
        let via_eval = cs.evaluate_at_point(&col_at_z, &alpha);

        // Path 2: build the combined polynomial and evaluate at z.
        let c_coeffs = cs.build_constraint_polynomial(&coeff_form, &alpha, n);
        let via_poly = CommitmentScheme::eval_poly_at(&scheme, &c_coeffs, &z);

        let diff = via_eval.sub(&via_poly);
        assert!(
            diff.is_zero(),
            "evaluator and polynomial-form must agree at non-domain z; \
             a per-category build_*_poly disagrees with its eval_*_at_point"
        );
    }

    /// Check the full combined constraint polynomial vanishes at EVERY
    /// domain point — this is the divisibility precondition for the
    /// prover's `C(X) / Z_H(X)` division to be exact.
    #[test]
    #[ignore = "slow: needs trace IFFT + full poly build; run with --release --ignored"]
    fn keccak_cs_full_polynomial_vanishes_at_every_domain_point() {
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0x1;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let n = trace.padded_size;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eval_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let coeff_form: Vec<Vec<Scalar>> = eval_form
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();

        let omega = CommitmentScheme::domain_generator(&scheme, n);
        let cs = KeccakConstraintSystem::new(NUM_ROUNDS)
            .with_omega_and_domain(omega.clone(), n);
        let alpha = Scalar::from_u64(23, curve);

        // Build the FULL combined polynomial (row + shifted).
        let row_poly = cs.build_constraint_polynomial(&coeff_form, &alpha, n);
        let shifted_poly = cs.build_shifted_constraint_polynomial(
            &coeff_form,
            &alpha,
            n,
            &omega,
            cs.num_constraints(),
        );
        let mut full_poly = row_poly;
        // Sum row + shifted (length must match — pad if needed).
        if full_poly.len() < shifted_poly.len() {
            full_poly.resize(shifted_poly.len(), Scalar::zero(curve));
        }
        let mut sp_padded = shifted_poly;
        if sp_padded.len() < full_poly.len() {
            sp_padded.resize(full_poly.len(), Scalar::zero(curve));
        }
        for i in 0..full_poly.len() {
            full_poly[i] = full_poly[i].add(&sp_padded[i]);
        }

        // Evaluate at every ω^r and assert zero.
        for r in 0..n {
            let z_r = scalar_pow(&omega, r);
            let v = CommitmentScheme::eval_poly_at(&scheme, &full_poly, &z_r);
            assert!(
                v.is_zero(),
                "C(ω^{}) ≠ 0; combined polynomial is NOT divisible by Z_H = X^{} − 1",
                r, n,
            );
        }
    }

    /// Same as `keccak_cs_eval_at_z_matches_poly_at_z` but for the FULL
    /// combined constraint (row-local + shifted). If row-local agrees but
    /// the full check disagrees, the bug is in the shifted path.
    #[test]
    #[ignore = "slow: needs trace IFFT + row-local + shifted poly builds; run with --release --ignored"]
    fn keccak_cs_full_eval_matches_full_poly_at_z() {
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0x1;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let n = trace.padded_size;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let eval_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let coeff_form: Vec<Vec<Scalar>> = eval_form
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();

        let omega = CommitmentScheme::domain_generator(&scheme, n);
        let cs = KeccakConstraintSystem::new(NUM_ROUNDS)
            .with_omega_and_domain(omega.clone(), n);
        let alpha = Scalar::from_u64(23, curve);
        let z = Scalar::from_u64(0xCAFE_BEEF_DEAD_BABE, curve);

        // Verifier's path: row-local C(z) + shifted C(z).
        let col_at_z: Vec<Scalar> = coeff_form
            .iter()
            .map(|c| CommitmentScheme::eval_poly_at(&scheme, c, &z))
            .collect();
        let shifted_at_z: Vec<Scalar> = cs
            .shifted_column_indices()
            .iter()
            .map(|&i| {
                // Shifted column evaluation at z = column_poly evaluated at ω·z.
                let omega_z = omega.mul(&z);
                CommitmentScheme::eval_poly_at(&scheme, &coeff_form[i], &omega_z)
            })
            .collect();
        // Compute omega^(n-1) for the fallback path (we don't actually use it here
        // since self.omega is set, but the trait API wants it).
        let mut onm1 = Scalar::one(curve);
        for _ in 0..n - 1 {
            onm1 = onm1.mul(&omega);
        }
        let row_local_eval = cs.evaluate_at_point(&col_at_z, &alpha);
        let shifted_eval = cs.evaluate_shifted_at_point(
            &col_at_z,
            &shifted_at_z,
            &z,
            &onm1,
            &alpha,
            cs.num_constraints(),
        );
        let via_eval = row_local_eval.add(&shifted_eval);

        // Prover's path: row-local poly + shifted poly, both evaluated at z.
        let row_local_poly = cs.build_constraint_polynomial(&coeff_form, &alpha, n);
        let shifted_poly = cs.build_shifted_constraint_polynomial(
            &coeff_form,
            &alpha,
            n,
            &omega,
            cs.num_constraints(),
        );
        let row_local_at_z = CommitmentScheme::eval_poly_at(&scheme, &row_local_poly, &z);
        let shifted_at_z_poly = CommitmentScheme::eval_poly_at(&scheme, &shifted_poly, &z);
        let via_poly = row_local_at_z.add(&shifted_at_z_poly);

        let diff = via_eval.sub(&via_poly);
        assert!(
            diff.is_zero(),
            "FULL eval/poly must agree at non-domain z; the divergence is in shifted",
        );
    }

    /// Cross-row polynomial must vanish on every real, non-boundary row
    /// (rows 0..NUM_ROUNDS-1 except row NUM_ROUNDS-1 which is a
    /// permutation boundary). After multiplication by the boundary
    /// factors `(X − ω^r)`, the polynomial vanishes *everywhere* (because
    /// the body already vanishes on the non-excluded rows, and the
    /// boundary factors only matter for Z_H divisibility).
    ///
    /// `#[ignore]`: ~1600 poly operations on a 32-coeff polynomial; the
    /// full build + per-row eval sweep pushes debug past our test budget.
    #[test]
    #[ignore = "too slow in debug: 1600 poly_shift + eval sweep; run with --release --ignored"]
    fn keccak_cs_shifted_polynomial_vanishes_on_non_boundary_rows() {
        let mut state: State = [[0u64; 5]; 5];
        state[2][3] = 0xdead_beef_cafe_babe;
        let rounds = keccak_f1600_witness(state);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let n = trace.padded_size;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let eval_form: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let coeff_form: Vec<Vec<Scalar>> = eval_form
            .iter()
            .map(|v| CommitmentScheme::ifft(&scheme, v, n))
            .collect();

        let omega = CommitmentScheme::domain_generator(&scheme, n);
        let cs = KeccakConstraintSystem::new(NUM_ROUNDS)
            .with_omega_and_domain(omega.clone(), n);
        let alpha = Scalar::from_u64(31, curve);
        let c_shifted = cs.build_shifted_constraint_polynomial(
            &coeff_form, &alpha, n, &omega, 0,
        );

        // At every domain point ω^r, c_shifted must vanish:
        //   - On non-boundary real rows: body is zero.
        //   - On boundary rows: body may be nonzero but is multiplied by
        //     (X − ω^r) which is zero at X = ω^r.
        //   - On padding rows: body is zero (both sides are zero bits).
        for r in 0..(n as usize) {
            let z = scalar_pow(&omega, r as u64);
            let v = CommitmentScheme::eval_poly_at(&scheme, &c_shifted, &z);
            assert!(
                v.is_zero(),
                "cross-row polynomial did not vanish at row {} / domain {}",
                r,
                n
            );
        }
    }

    /// End-to-end prove/verify with FULL aggregator + IS_BYTE_ACTIVE
    /// + algebraic input binding. Builds a real `keccak256("hello world")`
    /// witness, populates the per-invocation aggregator + per-byte
    /// active gate via
    /// [`crate::keccak_air::populate_trace_from_hash_with_invocation_bytes`],
    /// then proves and verifies. Exercises:
    ///   - is_byte_active_binary, is_byte_active_monotone, is_byte_active_sum_pin
    ///   - inv_input_byte_binding (gated by is_byte_active per byte)
    ///   - aggregator invariance shifted body
    ///   - IS_FIRST_BLOCK invariance shifted body
    ///   - all the bit-level Keccak round bodies on the actual hash trace
    /// Cross-AIR LogUp `joint_prove` end-to-end test:
    /// KeccakExtract (per-row byte view) ↔ Keccak (bit-level), both
    /// with full aggregator + algebraic input + output binding active.
    ///
    /// This is the "everything together" test for #91 (MPT ↔ Keccak):
    ///   - Build a KeccakExtract trace with one invocation
    ///     (keccak256("hello world")).
    ///   - Build a Keccak bit-level trace for the same invocation
    ///     (single-absorption, full aggregator population).
    ///   - Wire up the linkage descriptor from
    ///     `keccak_extract::make_keccak_extract_keccak_linkage_descriptor`.
    ///   - Call joint_prove + joint_verify.
    ///
    /// On a successful run: the cross-AIR LogUp matches the
    /// KeccakExtract row's `(input_bytes, input_len, output_digest)`
    /// tuple against the Keccak bit-level AIR's anchor row's
    /// `(INV_INPUT_BYTE[..], INV_INPUT_LEN_COL, INV_OUTPUT_BYTE[..])`
    /// tuple. The bit-level AIR's algebraic bindings ensure the
    /// aggregator bytes match the actual input/digest computed by the
    /// permutation. Combined: KeccakExtract's claimed
    /// (rlp, hash) = bit-level's actual (rlp, keccak256(rlp)).
    /// **PASSING** end-to-end cross-AIR `joint_prove`/`joint_verify`
    /// test for KeccakExtract ↔ Keccak. Total time ~325s (~5 min)
    /// dominated by the bit-level Keccak prover (13692 cols × 32 rows
    /// × 5 quotient polynomials).
    ///
    /// Two cross-AIR LogUp infrastructure bugs were uncovered and
    /// fixed in `cross_air_logup.rs` to make this test pass cleanly:
    ///
    /// **Bug 1 (FIXED in `build_linkage_trace`)**: the `H_A`/`H_B`
    /// columns defaulted to 0 past the per-AIR's native size when the
    /// linkage trace was padded larger (asymmetric trace sizes). The
    /// running-sum cross-row constraint `H(ω·X) = H(X) + F(X)` then
    /// fired at the n_a−1→n_a transition. Fix: extend `H` past the
    /// native size with the constant closure value `h[n_a−1] + f[n_a−1]`.
    ///
    /// **Bug 2 (FIXED in `joint_prove`)**: when per-AIR and linkage
    /// trace domains differ, the cross-trace tuple binding check
    /// (`tuple_A(z) == β-RLC of per-AIR openings at z`) fails because
    /// the polynomials are interpolated over different domains. Fix:
    /// `joint_prove` auto-inflates all per-AIR traces' `padded_size`
    /// to `max(padded_size_i)` before phase-1 commitment, so all
    /// commitments are over a common domain.
    ///
    /// Net effect: the cross-AIR LogUp infrastructure now supports
    /// asymmetric trace sizes transparently. Callers don't need to
    /// pre-pad per-AIR traces.
    #[test]
    #[ignore = "slow: KeccakExtract↔Keccak full joint prove — ~16 min; run with --release --ignored"]
    fn joint_prove_keccak_extract_keccak_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak::keccak_witness;
        use crate::keccak_extract::{
            build_trace_polynomials as build_extract_trace,
            make_keccak_extract_keccak_linkage_descriptor, KeccakExtractConstraintSystem,
            KeccakExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let input = b"hello world".to_vec();
        let digest = crate::keccak::keccak256(&input);
        let ht = keccak_witness(&input);

        // ── KeccakExtract trace (A side of the linkage) ──
        // Note: `joint_prove` auto-inflates the extract trace's
        // padded_size to match Keccak's (larger) domain internally —
        // see `cross_air_logup.rs::joint_prove`. The constraint system
        // here uses the extract's natural domain; the inflation
        // happens transparently inside joint_prove.
        let extract_w = KeccakExtractWitness::from_inputs(&[input.clone()])
            .expect("input within MAX_INPUT_LEN");
        let extract_trace = build_extract_trace(&extract_w, curve);
        let extract_domain = extract_trace.padded_size;
        let extract_omega = scheme.domain_generator(extract_domain);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_domain);

        // ── Keccak bit-level trace (B side of the linkage) ──
        let num_rows = ht.blocks.len() * NUM_ROUNDS;
        let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
        let mut keccak_columns =
            crate::keccak_air::populate_trace_from_hash_with_invocation_bytes(
                &ht, &input, &digest, curve,
            );
        for col in keccak_columns.iter_mut() {
            if col.len() < padded {
                col.resize(padded, Scalar::zero(curve));
            }
        }
        let keccak_trace = into_trace_polynomials(keccak_columns, num_rows, padded, curve);
        let keccak_domain = keccak_trace.padded_size;
        let keccak_omega = scheme.domain_generator(keccak_domain);
        let keccak_cs = KeccakConstraintSystem::new(num_rows)
            .with_omega_and_domain(keccak_omega, keccak_domain);

        // ── Linkage descriptor: KeccakExtract layer 0 ↔ Keccak layer 1 ──
        let linkage = make_keccak_extract_keccak_linkage_descriptor(0, 1);
        assert_eq!(linkage.label, "keccak_extract_keccak_v1");

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&extract_trace, &extract_cs), (&keccak_trace, &keccak_cs)];
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
            vec![&extract_cs, &keccak_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the KeccakExtract↔Keccak honest joint proof"
        );
    }

    /// 3-AIR end-to-end MPT inclusion chain: MPT AIR + KeccakExtract +
    /// bit-level Keccak. Mirrors `joint_prove_ssz_chain_e2e` (SSZ side)
    /// for MPT. Validates the MPT node-hash-consistency soundness chain
    /// end-to-end through bit-level Keccak constraints.
    ///
    /// Setup: single-leaf MPT inclusion → 1 hash invocation
    /// `keccak256(leaf_rlp)`. The hash is matched across all three AIRs
    /// via two cross-AIR LogUp linkages:
    ///   - L1: MPT↔KeccakExtract (NODE_RLP[0..256] + NODE_RLP_LEN +
    ///     NODE_HASH[0..32] ↔ INPUT_BYTE[0..256] + INPUT_LEN +
    ///     OUTPUT_BYTE[0..32])
    ///   - L2: KeccakExtract↔bit-level Keccak (byte aggregator tuples)
    ///
    /// Combined with the bit-level Keccak's algebraic input/output
    /// bindings (#91), this proves: the MPT trace's NODE_HASH column is
    /// the actual `keccak256(NODE_RLP[..NODE_RLP_LEN])` digest of the
    /// committed RLP-encoded node — algebraically pinned through every
    /// layer.
    #[test]
    #[ignore = "slow: 3-AIR + 2-linkage joint_prove with bit-level Keccak \
                (~12 min); run with --release --ignored"]
    fn joint_prove_mpt_chain_e2e() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak::{keccak_witness, NUM_ROUNDS};
        use crate::keccak_extract::{
            build_trace_polynomials as build_extract_trace,
            make_keccak_extract_keccak_linkage_descriptor,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
            MAX_INPUT_LEN, OUTPUT_LEN,
        };
        use crate::mpt::{MptNode, Nibbles};
        use crate::mpt_air::inclusion_witness;
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows as build_mpt_trace,
            MptInclusionConstraintSystem,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Single-leaf MPT inclusion: 1 hash row.
        let key = vec![0x1a, 0xbc];
        let val = b"value-a".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_bytes(&key),
            value: val.clone(),
        };
        let leaf_rlp = crate::mpt::mpt_node_rlp(&leaf);
        let proof = vec![leaf_rlp.clone()];
        let mpt_rows = inclusion_witness(&key, &proof);
        assert_eq!(mpt_rows.len(), 1, "single-leaf MPT inclusion must be 1 row");

        // ── MPT trace ──
        let mpt_trace = build_mpt_trace(&mpt_rows, curve);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_rows.len());

        // ── KeccakExtract: same RLP input as MPT ──
        let inputs: Vec<Vec<u8>> = mpt_rows
            .iter()
            .map(|r| r.node_rlp[..r.node_rlp_len].to_vec())
            .collect();
        let extract_w = KeccakExtractWitness::from_inputs(&inputs)
            .expect("MPT RLP inputs are within MAX_INPUT_LEN");
        let extract_trace = build_extract_trace(&extract_w, curve);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows);

        // ── Bit-level Keccak: actual hash of leaf_rlp ──
        let canonical_input = inputs[0].clone();
        let digest = crate::keccak::keccak256(&canonical_input);
        let ht = keccak_witness(&canonical_input);
        let num_keccak_rows = ht.blocks.len() * NUM_ROUNDS;
        let keccak_padded = crate::trace::nearest_power_of_two(num_keccak_rows.max(1));
        let mut keccak_columns = crate::keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &canonical_input, &digest, curve,
        );
        for col in keccak_columns.iter_mut() {
            if col.len() < keccak_padded {
                col.resize(keccak_padded, Scalar::zero(curve));
            }
        }
        let keccak_trace = into_trace_polynomials(
            keccak_columns, num_keccak_rows, keccak_padded, curve,
        );
        let keccak_cs = KeccakConstraintSystem::new(num_keccak_rows);

        // ── Linkages ──
        // L1 uses NODE_RLP_LEN as the MPT-side selector — non-zero on
        // real rows, zero on padding. The default helper passes None
        // for the selector which works only if the MPT trace has no
        // padding (every row real). With auto-padding to power-of-two,
        // padding rows would emit zero tuples that don't appear in
        // KeccakExtract's multiset, and witness building would fail.
        let mpt_rlp_bytes: Vec<usize> = (0..MAX_INPUT_LEN)
            .map(|b| crate::mpt_air::col::NODE_RLP_OFFSET + b)
            .collect();
        let mpt_node_hash: Vec<usize> = (0..OUTPUT_LEN)
            .map(|b| crate::mpt_air::col::NODE_HASH_OFFSET + b)
            .collect();
        let l1 = make_mpt_keccak_extract_linkage_descriptor(
            /* mpt */ 0, /* extract */ 1,
            mpt_rlp_bytes,
            crate::mpt_air::col::NODE_RLP_LEN,
            mpt_node_hash,
            Some(crate::mpt_air::col::NODE_RLP_LEN),
        );
        let l2 = make_keccak_extract_keccak_linkage_descriptor(
            /* extract */ 1, /* keccak */ 2,
        );

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&mpt_trace, &mpt_cs),
            (&extract_trace, &extract_cs),
            (&keccak_trace, &keccak_cs),
        ];
        let linkages = vec![l1, l2];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the 3-AIR MPT chain");
        assert_eq!(proofs.len(), 3);
        assert_eq!(extension.linkage_proofs.len(), 2);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&mpt_cs, &extract_cs, &keccak_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(valid, "joint verifier must accept the 3-AIR MPT chain");
    }

    /// Tampering test for the MPT chain L1 linkage (MPT↔KeccakExtract):
    /// modify MPT's `node_hash` for the single inclusion row to a wrong
    /// value while leaving KeccakExtract honest. The L1 multiset
    /// (MPT (rlp_bytes, len, hash) ↔ KeccakExtract (input, len, output))
    /// MUST fail since MPT's tuple no longer matches any KeccakExtract
    /// row's tuple.
    ///
    /// Validates that a malicious MPT prover cannot claim a wrong hash
    /// for a node without breaking the cross-AIR LogUp. This is the
    /// soundness foundation of MPT inclusion proofs.
    #[test]
    fn joint_prove_mpt_chain_rejects_tampered_node_hash() {
        use crate::cross_air_logup::joint_prove;
        use crate::keccak_extract::{
            build_trace_polynomials as build_extract_trace,
            make_keccak_extract_keccak_linkage_descriptor,
            make_mpt_keccak_extract_linkage_descriptor,
            KeccakExtractConstraintSystem, KeccakExtractWitness, MAX_INPUT_LEN, OUTPUT_LEN,
        };
        use crate::mpt::{MptNode, Nibbles};
        use crate::mpt_air::inclusion_witness;
        use crate::mpt_constraints::{
            build_trace_polynomials_from_rows as build_mpt_trace,
            MptInclusionConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let key = vec![0x1a, 0xbc];
        let val = b"value-a".to_vec();
        let leaf = MptNode::Leaf {
            path: Nibbles::from_bytes(&key),
            value: val.clone(),
        };
        let leaf_rlp = crate::mpt::mpt_node_rlp(&leaf);
        let proof = vec![leaf_rlp.clone()];
        let mut mpt_rows = inclusion_witness(&key, &proof);

        // TAMPER: corrupt the MPT row's node_hash so the L1 tuple
        // (rlp_bytes, len, node_hash) no longer matches what
        // KeccakExtract sees. The witness builder will detect the
        // multiset mismatch.
        mpt_rows[0].node_hash[0] ^= 0xFF;

        let mpt_trace = build_mpt_trace(&mpt_rows, curve);
        let mpt_cs = MptInclusionConstraintSystem::new(mpt_rows.len());

        // KeccakExtract uses the HONEST input/output (real keccak256
        // of leaf_rlp).
        let extract_w = KeccakExtractWitness::from_inputs(&[leaf_rlp.clone()])
            .expect("input within MAX_INPUT_LEN");
        let extract_trace = build_extract_trace(&extract_w, curve);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows);

        // We don't need bit-level Keccak for tampering — L1 fails first.
        let mpt_rlp_bytes: Vec<usize> = (0..MAX_INPUT_LEN)
            .map(|b| crate::mpt_air::col::NODE_RLP_OFFSET + b)
            .collect();
        let mpt_node_hash: Vec<usize> = (0..OUTPUT_LEN)
            .map(|b| crate::mpt_air::col::NODE_HASH_OFFSET + b)
            .collect();
        let l1 = make_mpt_keccak_extract_linkage_descriptor(
            0, 1, mpt_rlp_bytes, crate::mpt_air::col::NODE_RLP_LEN, mpt_node_hash,
            Some(crate::mpt_air::col::NODE_RLP_LEN),
        );
        let _l2_unused = make_keccak_extract_keccak_linkage_descriptor(1, 2);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&mpt_trace, &mpt_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![l1];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(result.is_err(), "joint_prove must reject tampered MPT node_hash");
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}", err
        );
    }

    /// End-to-end prove/verify for a 2-absorption (multi-absorption)
    /// keccak256 invocation. Input length 200 bytes (> 136 = RATE_LEN).
    /// Exercises the new block-1 input binding shifted constraint
    /// (body 3) which fires at row 23 (block 0 → block 1 boundary)
    /// and pins INV_INPUT_BYTE[136..256] to the XOR of BEFORE_at_row_24
    /// and AFTER_IOTA_at_row_23.
    #[test]
    #[ignore = "slow: full prover roundtrip on 2-absorption Keccak; run in --release with --ignored"]
    fn keccak_cs_prove_verify_with_aggregator_two_absorptions() {
        use crate::keccak::keccak_witness;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // 200-byte input → 2 absorption blocks (block 0 absorbs first
        // 136 bytes; block 1 absorbs remaining 64 bytes + padding).
        let mut input = Vec::with_capacity(200);
        for i in 0..200 {
            input.push((0x10u8).wrapping_add(i as u8));
        }
        let ht = keccak_witness(&input);
        let digest = crate::keccak::keccak256(&input);
        assert_eq!(ht.blocks.len(), 2, "200-byte input must produce 2 absorption blocks");

        let num_rows = ht.blocks.len() * NUM_ROUNDS;
        let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
        let mut columns = crate::keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, &digest, curve,
        );
        for col in columns.iter_mut() {
            if col.len() < padded {
                col.resize(padded, Scalar::zero(curve));
            }
        }
        let trace = into_trace_polynomials(columns, num_rows, padded, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = KeccakConstraintSystem::new(num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(
            valid,
            "keccak 2-absorption proof must verify (exercises block-1 input binding)"
        );
    }

    #[test]
    #[ignore = "slow: full prover roundtrip with aggregator binding; run in --release with --ignored"]
    fn keccak_cs_prove_verify_with_aggregator() {
        use crate::keccak::keccak_witness;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let input = b"hello world".to_vec();
        let ht = keccak_witness(&input);
        let digest = crate::keccak::keccak256(&input);
        // Single absorption (input < RATE_LEN), single permutation.
        assert_eq!(ht.blocks.len(), 1);

        let num_rows = ht.blocks.len() * NUM_ROUNDS;
        let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
        let mut columns = crate::keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, &digest, curve,
        );
        for col in columns.iter_mut() {
            if col.len() < padded {
                col.resize(padded, Scalar::zero(curve));
            }
        }
        let trace = into_trace_polynomials(columns, num_rows, padded, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = KeccakConstraintSystem::new(num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(
            valid,
            "keccak aggregator-populated proof must verify with algebraic input binding"
        );
    }

    /// End-to-end prove/verify for a single Keccak-f[1600] permutation.
    ///
    /// Marked `#[ignore]` because committing all 13,144 trace columns with
    /// BLS48-581 KZG is prohibitively slow under the default test budget
    /// (hours in debug mode). Run manually with:
    ///     cargo test -p metavm-zkp --lib --release keccak_cs_prove_verify_single_permutation -- --ignored --nocapture
    #[test]
    #[ignore = "too slow for CI: ~13k KZG commitments; run in --release with --ignored"]
    fn keccak_cs_prove_verify_single_permutation() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let state: State = [[0u64; 5]; 5];
        let rounds = keccak_f1600_witness(state);
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = KeccakConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "keccak single-permutation proof must verify");
    }

    /// End-to-end: produce a real Keccak ExecutionProof, serialize, attach
    /// to the BlockBinding layer (block-header keccak256), and verify
    /// through `LayerChainProof::verify_with_layer_verifier`. Completes
    /// the wired-AIR set with SHA-256, SSZ, and NonnativeFp.
    #[test]
    #[ignore = "slow: produces a real Keccak proof; run with --release --ignored"]
    fn keccak_proof_flows_through_layer_chain_envelope() {
        use crate::layer_chain::{
            ChainBoundaries, LayerChainProof, LayerProof, LayerProofKind,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let state: State = [[0u64; 5]; 5];
        let rounds = keccak_f1600_witness(state);
        let trace = build_trace_polynomials_from_rounds(&rounds, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = KeccakConstraintSystem::new(rounds.len())
            .with_omega_and_domain(omega.clone(), domain_size);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        assert!(crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve));

        let proof_bytes = proof.to_bytes();
        assert!(!proof_bytes.is_empty());

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
                    LayerProof::with_proof(claim, LayerProofKind::Keccak, proof_bytes.clone())
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        let num_rounds = rounds.len();
        let result = chain_proof.verify_with_layer_verifier(|layer| match layer.kind {
            LayerProofKind::Keccak => {
                let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                    .map_err(|e| format!("decode failed: {:?}", e))?;
                let cs = KeccakConstraintSystem::new(num_rounds)
                    .with_omega_and_domain(omega.clone(), domain_size);
                if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                    Ok(())
                } else {
                    Err("Keccak proof did not verify".to_string())
                }
            }
            LayerProofKind::ReferenceOnly => Ok(()),
            other => Err(format!("unsupported layer kind {}", other.as_str())),
        });
        assert_eq!(
            result,
            Ok(()),
            "real Keccak proof must verify through the LayerChainProof envelope",
        );
    }

    /// Cheap smoke test: reject a tampered witness with high probability.
    /// Not routed through prove/verify (too slow); instead checks that
    /// `evaluate_at_point` on the tampered row returns non-zero. This
    /// exercises the same bodies the verifier would use to reject the
    /// proof, scoped to a single row for speed.
    #[test]
    fn keccak_cs_detects_tampered_after_iota() {
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0xbeef_face_dead_cafe;
        let (mut columns, _) = single_permutation_columns(state);
        let curve = CurveType::Bls48581;
        let one = Scalar::one(curve);
        let col = after_iota(2, 3, 17);
        columns[col][0] = one.sub(&columns[col][0].clone());

        let cs = KeccakConstraintSystem::new(NUM_ROUNDS);
        let alpha = Scalar::from_u64(7, curve);
        let row_vals: Vec<Scalar> = columns.iter().map(|c| c[0].clone()).collect();
        let c_at_row = cs.evaluate_at_point(&row_vals, &alpha);
        assert!(
            !c_at_row.is_zero(),
            "tampered AFTER_IOTA bit should make C(row 0) non-zero"
        );
    }

    /// The 24 Keccak-f[1600] round constants, copied here as an
    /// independent oracle so the test verifies our pin matches the FIPS
    /// 202 spec table (not just the value in `crate::keccak`).
    const RC_SPEC: [u64; 24] = [
        0x0000_0000_0000_0001, 0x0000_0000_0000_8082,
        0x8000_0000_0000_808a, 0x8000_0000_8000_8000,
        0x0000_0000_0000_808b, 0x0000_0000_8000_0001,
        0x8000_0000_8000_8081, 0x8000_0000_0000_8009,
        0x0000_0000_0000_008a, 0x0000_0000_0000_0088,
        0x0000_0000_8000_8009, 0x0000_0000_8000_000a,
        0x0000_0000_8000_808b, 0x8000_0000_0000_008b,
        0x8000_0000_0000_8089, 0x8000_0000_0000_8003,
        0x8000_0000_0000_8002, 0x8000_0000_0000_0080,
        0x0000_0000_0000_800a, 0x8000_0000_8000_000a,
        0x8000_0000_8000_8081, 0x8000_0000_0000_8080,
        0x0000_0000_8000_0001, 0x8000_0000_8000_8008,
    ];

    #[test]
    fn keccak_cs_rc_word_pin_matches_spec_table() {
        // The pin must use the same RC table as the Keccak reference.
        assert_eq!(ROUND_CONSTANTS, RC_SPEC);
    }

    /// On a valid witness, the `rc_word_pin` body must be zero on every
    /// row: derived_RC = XOR(after_iota[0][0], after_chi[0][0]) =
    /// ROUND_CONSTANTS[active round].
    #[test]
    fn keccak_cs_rc_word_pin_vanishes_on_valid_witness() {
        // Use a non-zero state so after_chi[0][0] is generally non-zero.
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0xfeed_face_cafe_d00d;
        state[2][3] = 0x0123_4567_89ab_cdef;
        let (columns, _) = single_permutation_columns(state);

        // Per-row scalar-point eval of just the rc_word_pin body; fast.
        for row in 0..NUM_ROUNDS {
            let col_vals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let pin = eval_rc_word_pin_at_point(&col_vals);
            assert!(
                pin.is_zero(),
                "eval_rc_word_pin_at_point non-zero at row {} on valid witness",
                row
            );
        }
    }

    /// Tampering with `after_iota[0][0][bit]` on a round row makes the
    /// derived RC word disagree with the spec constant, so the
    /// `rc_word_pin` body fires.
    #[test]
    fn keccak_cs_rc_word_pin_detects_tampered_iota_bit() {
        let curve = CurveType::Bls48581;
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0x0123_4567_89ab_cdef;
        let (mut columns, _) = single_permutation_columns(state);

        // Flip after_iota[0][0][0] on round 0. ROUND_CONSTANTS[0] = 0x1,
        // so bit 0 of derived_RC was 1; after flipping it becomes 0,
        // making derived_RC = ROUND_CONSTANTS[0] − 1, body non-zero.
        let one = Scalar::one(curve);
        let col = after_iota(0, 0, 0);
        let row = 0;
        columns[col][row] = one.sub(&columns[col][row].clone());

        let col_vals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
        let pin_at_row = eval_rc_word_pin_at_point(&col_vals);
        assert!(
            !pin_at_row.is_zero(),
            "eval_rc_word_pin_at_point must detect tampered after_iota bit"
        );
    }

    #[test]
    fn keccak_boundary_rows_single_permutation() {
        // num_rows=24, domain_size=32 → boundaries {23, 31}.
        let br = boundary_rows(NUM_ROUNDS, 32);
        assert_eq!(br, vec![23, 31]);
    }

    #[test]
    fn keccak_boundary_rows_two_permutations() {
        // num_rows=48, domain_size=64 → boundaries {23, 47, 63}.
        let br = boundary_rows(2 * NUM_ROUNDS, 64);
        assert_eq!(br, vec![23, 47, 63]);
    }

    #[test]
    fn keccak_boundary_rows_no_padding() {
        // num_rows=32 (not a multiple of 24 — unusual, but `boundary_rows`
        // should still behave: only rows 23 from the permutation scheme
        // plus wrap-around row 31.
        let br = boundary_rows(32, 32);
        assert_eq!(br, vec![23, 31]);
    }
}
