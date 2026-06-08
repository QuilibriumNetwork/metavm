//! Memory-expansion gas AIR (roadmap #60 — last major piece of gas
//! accounting).
//!
//! Algebraic version of [`crate::memory_expansion`]'s host-side oracle:
//!
//!     G_memory(words) = 3 * words + floor(words^2 / 512)
//!     expansion_cost  = max(0, G_memory(new_words) - G_memory(old_words))
//!
//! Each row of the AIR commits a single
//! `(old_size, new_size, old_words, new_words, old_cost, new_cost,
//! expansion_cost)` tuple plus helper remainders for the two divisions
//! involved (word-count from byte-count and the `w^2 / 512` term in
//! `G_memory`).
//!
//! ## Constraint sketch
//!
//! - `is_real ∈ {0, 1}`, `is_expansion ∈ {0, 1}`.
//! - Word count (ceil-div by 32):
//!     `new_words * 32 - new_size - new_words_rem = 0`,  with
//!     `new_words_rem ∈ [0..31]` (range check is a documented host-side
//!     responsibility for now — Phase A3 follow-up wires it through a
//!     5-bit table AIR; see "Deferred" below). Same for `old_*`.
//! - Cost formula. Let `q_new = new_cost - 3 * new_words`. Then
//!     `q_new * 512 + new_cost_rem = new_words * new_words`,  with
//!     `new_cost_rem ∈ [0..511]` (range check deferred, same as above).
//!     Same for `old_*`.
//! - Expansion = max(0, new_cost - old_cost):
//!     `is_expansion * (expansion_cost - (new_cost - old_cost)) = 0`  and
//!     `(1 - is_expansion) * expansion_cost = 0`.
//!     `is_expansion = 1` iff `new_cost > old_cost` (host-decided; the
//!     algebraic relation only enforces consistency given the flag).
//!
//! ## Cross-AIR LogUp
//!
//! - [`make_gas_tracking_to_memory_expansion_descriptor`] binds gas-tracking
//!   `COL_DYNAMIC_COST` rows (memory-op rows, gated host-side via
//!   `COL_IS_REAL` until a dedicated `is_memory_op` selector is added) to
//!   memory-expansion `COL_EXPANSION_COST`. The descriptor is therefore a
//!   single-column tuple, matching the soundness-equivalent "expansion cost
//!   equals dynamic cost" host invariant. (The opcode-side dispatching is
//!   the gas-tracking ↔ static-table responsibility; the
//!   memory-expansion side answers "is this number a valid expansion cost
//!   for some (old, new) pair".)
//!
//! ## Deferred (audit gaps, documented so future tightening is targeted)
//!
//! - The two range checks (`*_rem ∈ [0..31]`, `*_cost_rem ∈ [0..511]`)
//!   are currently host-side; without them the prover could pick
//!   `new_words` too large by 32 and absorb the difference into the
//!   `new_words_rem` column. Phase A3 follow-up: feed each remainder
//!   column through the existing byte-range table AIR (split the 9-bit
//!   `*_cost_rem` into two byte pieces).
//! - The `is_expansion` flag is *witnessed*, not constrained from a
//!   comparison; a future enhancement would add `(new_cost - old_cost) *
//!   (1 - is_expansion) = nonneg_witness` to force the host's claim.
//! - The descriptor's `a_selector_column` is currently `COL_IS_REAL`
//!   (i.e. every real gas-tracking row), so the closure only matches when
//!   `from_evm_trace` populates the gas-tracking witness with memory-op
//!   rows only. A dedicated `COL_IS_MEMORY_OP` column on
//!   [`crate::gas_tracking_air`] is the clean fix.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ───────────────────────────────────────────────────

pub const COL_OLD_SIZE: usize = 0;
pub const COL_NEW_SIZE: usize = 1;
pub const COL_OLD_WORDS: usize = 2;
pub const COL_NEW_WORDS: usize = 3;
pub const COL_OLD_COST: usize = 4;
pub const COL_NEW_COST: usize = 5;
pub const COL_EXPANSION_COST: usize = 6;
/// `new_words * 32 - new_size` ∈ [0..31] (range check deferred).
pub const COL_NEW_WORDS_REM: usize = 7;
/// `old_words * 32 - old_size` ∈ [0..31] (range check deferred).
pub const COL_OLD_WORDS_REM: usize = 8;
/// `new_words^2 - 512 * (new_cost - 3*new_words)` ∈ [0..511].
pub const COL_NEW_COST_REM: usize = 9;
/// `old_words^2 - 512 * (old_cost - 3*old_words)` ∈ [0..511].
pub const COL_OLD_COST_REM: usize = 10;
pub const COL_IS_EXPANSION: usize = 11;
pub const COL_IS_REAL: usize = 12;
pub const NUM_COLUMNS: usize = 13;

/// Per-row constraints:
/// 0. `is_real * (is_real - 1) = 0`
/// 1. `is_expansion * (is_expansion - 1) = 0`
/// 2. `is_real * (new_words*32 - new_size - new_words_rem) = 0`
/// 3. `is_real * (old_words*32 - old_size - old_words_rem) = 0`
/// 4. `is_real * (512 * (new_cost - 3*new_words) + new_cost_rem
///                - new_words*new_words) = 0`
/// 5. `is_real * (512 * (old_cost - 3*old_words) + old_cost_rem
///                - old_words*old_words) = 0`
/// 6. `is_expansion * (expansion_cost - (new_cost - old_cost)) = 0`
/// 7. `(is_real - is_expansion) * expansion_cost = 0`
///    (zero expansion cost when no expansion happened, gated by is_real
///    so padding rows don't fire)
pub const NUM_ROW_CONSTRAINTS: usize = 8;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct MemoryExpansionRow {
    pub old_size: u64,
    pub new_size: u64,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryExpansionWitness {
    pub rows: Vec<MemoryExpansionRow>,
}

impl MemoryExpansionWitness {
    pub fn from_rows(rows: Vec<MemoryExpansionRow>) -> Self { Self { rows } }
}

/// Build a witness from parallel `old_sizes` / `new_sizes` slices (one
/// row per memory-op invocation in the trace).
pub fn from_memory_trace(old_sizes: &[u64], new_sizes: &[u64]) -> MemoryExpansionWitness {
    assert_eq!(
        old_sizes.len(),
        new_sizes.len(),
        "old_sizes and new_sizes must align"
    );
    let rows = old_sizes
        .iter()
        .zip(new_sizes.iter())
        .map(|(&o, &n)| MemoryExpansionRow { old_size: o, new_size: n })
        .collect();
    MemoryExpansionWitness { rows }
}

pub fn build_trace_polynomials(
    w: &MemoryExpansionWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (r, row) in w.rows.iter().enumerate() {
        let old_size = row.old_size;
        let new_size = row.new_size;
        let old_words = crate::memory_expansion::memory_word_count(old_size);
        let new_words = crate::memory_expansion::memory_word_count(new_size);
        let old_cost = crate::memory_expansion::memory_gas_cost(old_words);
        let new_cost = crate::memory_expansion::memory_gas_cost(new_words);
        let expansion = if new_cost > old_cost { new_cost - old_cost } else { 0 };
        let is_expansion = if new_cost > old_cost { 1u64 } else { 0u64 };
        // Remainders (the algebraic constraints' right-hand sides).
        let new_words_rem = old_words_rem_helper(new_words, new_size);
        let old_words_rem = old_words_rem_helper(old_words, old_size);
        let new_cost_rem = cost_rem_helper(new_words, new_cost);
        let old_cost_rem = cost_rem_helper(old_words, old_cost);

        cols[COL_OLD_SIZE][r] = Scalar::from_u64(old_size, curve);
        cols[COL_NEW_SIZE][r] = Scalar::from_u64(new_size, curve);
        cols[COL_OLD_WORDS][r] = Scalar::from_u64(old_words, curve);
        cols[COL_NEW_WORDS][r] = Scalar::from_u64(new_words, curve);
        cols[COL_OLD_COST][r] = Scalar::from_u64(old_cost, curve);
        cols[COL_NEW_COST][r] = Scalar::from_u64(new_cost, curve);
        cols[COL_EXPANSION_COST][r] = Scalar::from_u64(expansion, curve);
        cols[COL_NEW_WORDS_REM][r] = Scalar::from_u64(new_words_rem, curve);
        cols[COL_OLD_WORDS_REM][r] = Scalar::from_u64(old_words_rem, curve);
        cols[COL_NEW_COST_REM][r] = Scalar::from_u64(new_cost_rem, curve);
        cols[COL_OLD_COST_REM][r] = Scalar::from_u64(old_cost_rem, curve);
        cols[COL_IS_EXPANSION][r] = Scalar::from_u64(is_expansion, curve);
        cols[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

#[inline]
fn old_words_rem_helper(words: u64, size: u64) -> u64 {
    // words*32 - size ∈ [0..31]
    words.saturating_mul(32).saturating_sub(size)
}

#[inline]
fn cost_rem_helper(words: u64, cost: u64) -> u64 {
    // words*words - 512 * (cost - 3*words) ∈ [0..511]
    let q = cost.saturating_sub(3u64.saturating_mul(words));
    words.saturating_mul(words).saturating_sub(512u64.saturating_mul(q))
}

// ─── Constraint system ───────────────────────────────────────────────

pub struct MemoryExpansionConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl MemoryExpansionConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for MemoryExpansionConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_expansion_binary".into(),
            "new_words_ceildiv".into(),
            "old_words_ceildiv".into(),
            "new_cost_formula".into(),
            "old_cost_formula".into(),
            "expansion_consistency".into(),
            "no_expansion_zero_cost".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let three = Scalar::from_u64(3, curve);
        let thirty_two = Scalar::from_u64(32, curve);
        let five_twelve = Scalar::from_u64(512, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for r in 0..n {
            let real = &columns[COL_IS_REAL][r];
            let is_exp = &columns[COL_IS_EXPANSION][r];
            let old_size = &columns[COL_OLD_SIZE][r];
            let new_size = &columns[COL_NEW_SIZE][r];
            let old_words = &columns[COL_OLD_WORDS][r];
            let new_words = &columns[COL_NEW_WORDS][r];
            let old_cost = &columns[COL_OLD_COST][r];
            let new_cost = &columns[COL_NEW_COST][r];
            let exp_cost = &columns[COL_EXPANSION_COST][r];
            let nwr = &columns[COL_NEW_WORDS_REM][r];
            let owr = &columns[COL_OLD_WORDS_REM][r];
            let ncr = &columns[COL_NEW_COST_REM][r];
            let ocr = &columns[COL_OLD_COST_REM][r];

            // 0. is_real binary
            bodies[0][r] = real.mul(&real.sub(&one));
            // 1. is_expansion binary
            bodies[1][r] = is_exp.mul(&is_exp.sub(&one));
            // 2. new_words * 32 - new_size - new_words_rem = 0   (gated)
            let nw32 = new_words.mul(&thirty_two);
            bodies[2][r] = real.mul(&nw32.sub(new_size).sub(nwr));
            // 3. old_words * 32 - old_size - old_words_rem = 0   (gated)
            let ow32 = old_words.mul(&thirty_two);
            bodies[3][r] = real.mul(&ow32.sub(old_size).sub(owr));
            // 4. 512 * (new_cost - 3*new_words) + new_cost_rem - new_words^2 = 0
            let n3w = three.mul(new_words);
            let nq = new_cost.sub(&n3w);
            let nws = new_words.mul(new_words);
            let lhs_n = five_twelve.mul(&nq).add(ncr).sub(&nws);
            bodies[4][r] = real.mul(&lhs_n);
            // 5. same for old
            let o3w = three.mul(old_words);
            let oq = old_cost.sub(&o3w);
            let ows = old_words.mul(old_words);
            let lhs_o = five_twelve.mul(&oq).add(ocr).sub(&ows);
            bodies[5][r] = real.mul(&lhs_o);
            // 6. is_expansion * (expansion_cost - (new_cost - old_cost)) = 0
            let diff = new_cost.sub(old_cost);
            bodies[6][r] = is_exp.mul(&exp_cost.sub(&diff));
            // 7. (is_real - is_expansion) * expansion_cost = 0
            //    On padding rows (real=0, is_exp=0) -> 0.
            //    On non-expansion real rows (real=1, is_exp=0) -> exp_cost must be 0.
            //    On expansion rows (real=1, is_exp=1) -> 0.
            //    Disallowed combo (real=0, is_exp=1) is blocked by 1+0 binary +
            //    the way the witness builder sets both; soft check.
            let gate = real.sub(is_exp);
            bodies[7][r] = gate.mul(exp_cost);
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let three = Scalar::from_u64(3, curve);
        let thirty_two = Scalar::from_u64(32, curve);
        let five_twelve = Scalar::from_u64(512, curve);

        let real = &ce[COL_IS_REAL];
        let is_exp = &ce[COL_IS_EXPANSION];

        let c0 = real.mul(&real.sub(&one));
        let c1 = is_exp.mul(&is_exp.sub(&one));
        let nw32 = ce[COL_NEW_WORDS].mul(&thirty_two);
        let c2 = real.mul(&nw32.sub(&ce[COL_NEW_SIZE]).sub(&ce[COL_NEW_WORDS_REM]));
        let ow32 = ce[COL_OLD_WORDS].mul(&thirty_two);
        let c3 = real.mul(&ow32.sub(&ce[COL_OLD_SIZE]).sub(&ce[COL_OLD_WORDS_REM]));
        let n3w = three.mul(&ce[COL_NEW_WORDS]);
        let nq = ce[COL_NEW_COST].sub(&n3w);
        let nws = ce[COL_NEW_WORDS].mul(&ce[COL_NEW_WORDS]);
        let c4 = real.mul(&five_twelve.mul(&nq).add(&ce[COL_NEW_COST_REM]).sub(&nws));
        let o3w = three.mul(&ce[COL_OLD_WORDS]);
        let oq = ce[COL_OLD_COST].sub(&o3w);
        let ows = ce[COL_OLD_WORDS].mul(&ce[COL_OLD_WORDS]);
        let c5 = real.mul(&five_twelve.mul(&oq).add(&ce[COL_OLD_COST_REM]).sub(&ows));
        let diff = ce[COL_NEW_COST].sub(&ce[COL_OLD_COST]);
        let c6 = is_exp.mul(&ce[COL_EXPANSION_COST].sub(&diff));
        let gate = real.sub(is_exp);
        let c7 = gate.mul(&ce[COL_EXPANSION_COST]);

        let bodies = [c0, c1, c2, c3, c4, c5, c6, c7];
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(
        &self,
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let three_p = vec![Scalar::from_u64(3, curve)];
        let thirty_two_p = vec![Scalar::from_u64(32, curve)];
        let five_twelve_p = vec![Scalar::from_u64(512, curve)];
        let real = &cc[COL_IS_REAL];
        let is_exp = &cc[COL_IS_EXPANSION];

        let real_m1 = poly_sub(real, &one_p, curve);
        let c0 = poly_mul(real, &real_m1, curve);

        let is_exp_m1 = poly_sub(is_exp, &one_p, curve);
        let c1 = poly_mul(is_exp, &is_exp_m1, curve);

        let nw32 = poly_mul(&cc[COL_NEW_WORDS], &thirty_two_p, curve);
        let nw32_minus_ns = poly_sub(&nw32, &cc[COL_NEW_SIZE], curve);
        let nw32_minus_ns_minus_rem = poly_sub(&nw32_minus_ns, &cc[COL_NEW_WORDS_REM], curve);
        let c2 = poly_mul(real, &nw32_minus_ns_minus_rem, curve);

        let ow32 = poly_mul(&cc[COL_OLD_WORDS], &thirty_two_p, curve);
        let ow32_minus_os = poly_sub(&ow32, &cc[COL_OLD_SIZE], curve);
        let ow32_minus_os_minus_rem = poly_sub(&ow32_minus_os, &cc[COL_OLD_WORDS_REM], curve);
        let c3 = poly_mul(real, &ow32_minus_os_minus_rem, curve);

        let n3w = poly_mul(&three_p, &cc[COL_NEW_WORDS], curve);
        let nq = poly_sub(&cc[COL_NEW_COST], &n3w, curve);
        let nq512 = poly_mul(&five_twelve_p, &nq, curve);
        let nq512_plus_rem = poly_add(&nq512, &cc[COL_NEW_COST_REM], curve);
        let nws = poly_mul(&cc[COL_NEW_WORDS], &cc[COL_NEW_WORDS], curve);
        let cost_lhs_n = poly_sub(&nq512_plus_rem, &nws, curve);
        let c4 = poly_mul(real, &cost_lhs_n, curve);

        let o3w = poly_mul(&three_p, &cc[COL_OLD_WORDS], curve);
        let oq = poly_sub(&cc[COL_OLD_COST], &o3w, curve);
        let oq512 = poly_mul(&five_twelve_p, &oq, curve);
        let oq512_plus_rem = poly_add(&oq512, &cc[COL_OLD_COST_REM], curve);
        let ows = poly_mul(&cc[COL_OLD_WORDS], &cc[COL_OLD_WORDS], curve);
        let cost_lhs_o = poly_sub(&oq512_plus_rem, &ows, curve);
        let c5 = poly_mul(real, &cost_lhs_o, curve);

        let diff = poly_sub(&cc[COL_NEW_COST], &cc[COL_OLD_COST], curve);
        let exp_minus_diff = poly_sub(&cc[COL_EXPANSION_COST], &diff, curve);
        let c6 = poly_mul(is_exp, &exp_minus_diff, curve);

        let gate = poly_sub(real, is_exp, curve);
        let c7 = poly_mul(&gate, &cc[COL_EXPANSION_COST], curve);

        let bodies = [c0, c1, c2, c3, c4, c5, c6, c7];
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL, COL_IS_EXPANSION] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
            return;
        }
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

// ─── Cross-AIR LogUp descriptor ──────────────────────────────────────

/// Gas-tracking `(dynamic_cost)` rows → memory-expansion
/// `(expansion_cost)` rows.
///
/// Single-column tuple, gated by `COL_IS_REAL` on both sides. This binds
/// "the dynamic cost the gas-tracking AIR charged for this row equals an
/// expansion cost produced by some `(old_size, new_size)` pair witnessed
/// in the memory-expansion AIR".
///
/// **Caveat**: the gating selector is `COL_IS_REAL` (every real
/// gas-tracking row), so the closure only matches when the gas-tracking
/// witness is populated with memory-op rows only. Adding a dedicated
/// `COL_IS_MEMORY_OP` selector to the gas-tracking AIR is the clean
/// follow-up (see module-level doc).
pub fn make_gas_tracking_to_memory_expansion_descriptor(
    gas_tracking_layer_index: usize,
    memory_expansion_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::gas_tracking_air::{COL_DYNAMIC_COST, COL_IS_REAL as GT_COL_IS_REAL};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "gas_tracking_to_memory_expansion_v1".into(),
        a_layer_index: gas_tracking_layer_index,
        a_columns: vec![COL_DYNAMIC_COST],
        a_selector_column: Some(GT_COL_IS_REAL),
        b_layer_index: memory_expansion_layer_index,
        b_columns: vec![COL_EXPANSION_COST],
        b_selector_column: Some(COL_IS_REAL),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_all_zero(cs: &MemoryExpansionConstraintSystem, t: &TracePolynomials) {
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn zero_expansion_no_growth() {
        // old == new -> expansion cost = 0
        let w = from_memory_trace(&[64, 32, 0], &[64, 32, 0]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = MemoryExpansionConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        // Spot-check the expansion cost column
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..w.rows.len() {
            assert!(cr[COL_EXPANSION_COST][r].is_zero());
            assert!(cr[COL_IS_EXPANSION][r].is_zero());
        }
    }

    #[test]
    fn small_expansion_zero_to_32_bytes() {
        let w = from_memory_trace(&[0], &[32]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = MemoryExpansionConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls48581;
        // new_words = 1, new_cost = 3*1 + 1/512 = 3, expansion = 3 - 0 = 3
        assert!(cr[COL_NEW_WORDS][0].sub(&Scalar::from_u64(1, curve)).is_zero());
        assert!(cr[COL_NEW_COST][0].sub(&Scalar::from_u64(3, curve)).is_zero());
        assert!(cr[COL_EXPANSION_COST][0].sub(&Scalar::from_u64(3, curve)).is_zero());
        assert!(cr[COL_IS_EXPANSION][0].sub(&Scalar::one(curve)).is_zero());
    }

    #[test]
    fn medium_expansion_32_to_1024() {
        // old: 32 bytes -> 1 word, cost = 3
        // new: 1024 bytes -> 32 words, cost = 96 + 1024/512 = 98
        // expansion = 98 - 3 = 95
        let w = from_memory_trace(&[32], &[1024]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = MemoryExpansionConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls48581;
        assert!(cr[COL_OLD_WORDS][0].sub(&Scalar::from_u64(1, curve)).is_zero());
        assert!(cr[COL_NEW_WORDS][0].sub(&Scalar::from_u64(32, curve)).is_zero());
        assert!(cr[COL_OLD_COST][0].sub(&Scalar::from_u64(3, curve)).is_zero());
        assert!(cr[COL_NEW_COST][0].sub(&Scalar::from_u64(98, curve)).is_zero());
        assert!(cr[COL_EXPANSION_COST][0].sub(&Scalar::from_u64(95, curve)).is_zero());
    }

    #[test]
    fn word_count_rounding_correct() {
        // 1 byte -> 1 word, 33 bytes -> 2 words, 65 bytes -> 3 words
        let w = from_memory_trace(&[0, 0, 0], &[1, 33, 65]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = MemoryExpansionConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls48581;
        assert!(cr[COL_NEW_WORDS][0].sub(&Scalar::from_u64(1, curve)).is_zero());
        assert!(cr[COL_NEW_WORDS][1].sub(&Scalar::from_u64(2, curve)).is_zero());
        assert!(cr[COL_NEW_WORDS][2].sub(&Scalar::from_u64(3, curve)).is_zero());
        // Remainders
        assert!(cr[COL_NEW_WORDS_REM][0].sub(&Scalar::from_u64(31, curve)).is_zero());
        assert!(cr[COL_NEW_WORDS_REM][1].sub(&Scalar::from_u64(31, curve)).is_zero());
        assert!(cr[COL_NEW_WORDS_REM][2].sub(&Scalar::from_u64(31, curve)).is_zero());
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_gas_tracking_to_memory_expansion_descriptor(0, 1);
        assert_eq!(d.label, "gas_tracking_to_memory_expansion_v1");
        assert_eq!(d.a_columns.len(), 1);
        assert_eq!(d.b_columns.len(), 1);
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns[0], crate::gas_tracking_air::COL_DYNAMIC_COST);
        assert_eq!(d.b_columns[0], COL_EXPANSION_COST);
        assert_eq!(d.a_selector_column, Some(crate::gas_tracking_air::COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn tampered_expansion_cost_detected() {
        // Build an honest witness, then poke a wrong expansion_cost cell.
        let w = from_memory_trace(&[0], &[32]);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        let curve = CurveType::Bls48581;
        // Corrupt expansion_cost from 3 -> 99
        t.columns[COL_EXPANSION_COST].evaluations[0] = Scalar::from_u64(99, curve);
        let cs = MemoryExpansionConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 6 (is_expansion=1 → expansion = new - old) should fire.
        assert!(!bodies[6][0].is_zero(), "tampered expansion cost not caught");
    }

    #[test]
    fn tampered_word_count_detected() {
        // Honest: new_size = 32 -> new_words = 1, rem = 0. Tamper rem to 31.
        let w = from_memory_trace(&[0], &[32]);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        let curve = CurveType::Bls48581;
        t.columns[COL_NEW_WORDS].evaluations[0] = Scalar::from_u64(2, curve);
        // rem must change to satisfy the equation as well — but we leave it
        // unchanged so the constraint should fire.
        let cs = MemoryExpansionConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[2][0].is_zero(), "tampered word count not caught");
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let w = from_memory_trace(&[0, 32, 64], &[32, 64, 1024]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = MemoryExpansionConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xabcd, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..w.rows.len() {
            let row_evals: Vec<Scalar> = cr.iter().map(|c| c[r].clone()).collect();
            let v = cs.evaluate_at_point(&row_evals, &alpha);
            assert!(v.is_zero(), "row {} nonzero", r);
        }
    }
}
