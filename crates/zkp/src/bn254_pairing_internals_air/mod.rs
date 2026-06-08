//! BN254 (alt_bn128) pairing internals — host-side scaffold AIR.
//!
//! # Purpose
//!
//! Mirrors [`crate::miller_step_air`] / [`crate::miller_loop_air`] /
//! [`crate::final_exp_air`]'s role for BLS12-381: provides a
//! per-iteration commitment surface for one **Miller-loop step** + one
//! **final-exponentiation step** of the BN254 (alt_bn128) optimal-ate
//! pairing.
//!
//! # BN254 pairing structure
//!
//! BN254 uses the optimal-ate pairing with loop parameter
//! `6·x + 2 = 0x44E992B44A6909F1` (binary weight ≈ 18 bits over a
//! 65-bit loop). The Miller loop unrolls into 64 doubling iterations
//! and (depending on signed-NAF representation) several
//! addition iterations. After the Miller loop, the final
//! exponentiation `f^((q^12 - 1) / r)` is split into:
//!
//!   1. **Easy part**: `f^(q^6 - 1) · f^(q^2 + 1)` — three Frobenius
//!      applications + two multiplications.
//!   2. **Hard part**: `f^((q^4 - q^2 + 1) / r)` — implemented via the
//!      standard Vercauteren-style 9-cyclotomic-square / 3-Frobenius
//!      chain.
//!
//! # Scope of this module
//!
//! This is a **scaffold**: it commits the IO of one Miller-step row
//! (or one final-exp step row) gated by `is_real`, but does **NOT**
//! algebraically enforce the per-row Fp12 arithmetic. The host-side
//! BN254 [`Fp12`] type provides the algebraic ground-truth; witness
//! rows commit the inputs and the host-computed output. The
//! decomposition of `acc_post = acc_pre² · line_value` (Miller step)
//! or `acc_post = cyclotomic_square_or_frobenius(acc_pre)`
//! (final-exp step) into Fp multiplications is **deferred** — the
//! same path [`crate::miller_fp_descriptors`] takes for BLS12-381.
//!
//! Once the deferred phase lands, this AIR will gain cross-AIR LogUp
//! descriptors binding each row's Fp12 step to a sequence of Fp
//! multiplications in a dedicated `bn254_fp_air` (mirroring
//! [`crate::nonnative_fp_air`]).
//!
//! # Why this module exists now
//!
//! Layer B / Layer C ambition for BN254 (EIP-197) needs an algebraic
//! pairing proof. Step 0 of the gadget is a stable column-layout
//! target for the deferred decomposition. That's what this module
//! provides:
//!
//!   1. Host-side BN254 [`Fp12`] re-exported here as the algebraic
//!      ground-truth surface.
//!   2. AIR commitment of one Miller-step / one final-exp-step IO so
//!      the deferred decomposition has a stable target column layout.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// Re-export the BN254 Fp / Fp2 from the sibling curve-ops scaffold so
// both modules share a single host-side reference.
pub use crate::bn254_curve_ops_air::{Fp, Fp2, LIMBS_PER_BN254_FP};

// ─── Host-side BN254 Fp12 scaffold ────────────────────────────────────

/// Element of BN254 Fp6 = Fp2[v]/(v³ - ξ), ξ = 9 + u.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp6 {
    pub c0: Fp2,
    pub c1: Fp2,
    pub c2: Fp2,
}

impl Fp6 {
    pub const fn zero() -> Self {
        Fp6 { c0: Fp2::zero(), c1: Fp2::zero(), c2: Fp2::zero() }
    }
    pub const fn one() -> Self {
        Fp6 { c0: Fp2::one(), c1: Fp2::zero(), c2: Fp2::zero() }
    }
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero() && self.c2.is_zero()
    }
}

/// Element of BN254 Fp12 = Fp6[w]/(w² - v).
///
/// Stored as `c0 + c1·w` where `c0, c1 ∈ Fp6`. The deferred algebraic
/// AIR enforces `mul`/`square`/`mul_by_014`/`cyclotomic_square` row
/// constraints over these 72 limb cells. This scaffold commits only
/// the host-computed values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp12 {
    pub c0: Fp6,
    pub c1: Fp6,
}

impl Fp12 {
    pub const fn zero() -> Self {
        Fp12 { c0: Fp6::zero(), c1: Fp6::zero() }
    }
    pub const fn one() -> Self {
        Fp12 { c0: Fp6::one(), c1: Fp6::zero() }
    }
    pub fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }
}

/// Number of distinct Fp elements packed into one Fp12.
pub const FP_PER_FP12: usize = 12;
/// Number of u64 limbs in one Fp12.
pub const LIMBS_PER_FP12: usize = FP_PER_FP12 * LIMBS_PER_BN254_FP;

// ─── Row enums ─────────────────────────────────────────────────────────

/// Which pairing-internals operation a row encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InternalsOp {
    /// One Miller-loop doubling step:
    /// `acc_post = acc_pre² · line_value` + `Q_next = 2·Q_curr` (the
    /// G2 step lives on the sibling [`crate::bn254_curve_ops_air`]).
    MillerDoubling,
    /// One Miller-loop addition step:
    /// `acc_post = acc_pre² · line_value` + `Q_next = Q_curr + Q_fixed`.
    MillerAddition,
    /// One final-exponentiation easy-part step:
    /// `acc_post = frobenius(acc_pre) · acc_pre_aux`.
    FinalExpEasy,
    /// One final-exponentiation hard-part step (Vercauteren chain
    /// element): `acc_post = cyclotomic_square(acc_pre)` or
    /// `acc_post = acc_pre · acc_pre_aux`.
    FinalExpHard,
}

/// One row of the BN254 pairing-internals AIR.
#[derive(Clone, Debug)]
pub struct Bn254PairingInternalsRow {
    /// Accumulator input.
    pub acc_pre: Fp12,
    /// Accumulator output (host-computed; commitment only).
    pub acc_post: Fp12,
    /// Auxiliary Fp12 operand: line value for Miller rows,
    /// previous-stage accumulator for final-exp rows.
    pub aux: Fp12,
    /// Which internals operation this row encodes.
    pub op: InternalsOp,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Bn254PairingInternalsWitness {
    pub rows: Vec<Bn254PairingInternalsRow>,
}

impl Bn254PairingInternalsWitness {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    pub fn push(&mut self, row: Bn254PairingInternalsRow) {
        self.rows.push(row);
    }
}

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_ACC_PRE_OFFSET: usize = 0;
pub const COL_ACC_POST_OFFSET: usize = COL_ACC_PRE_OFFSET + LIMBS_PER_FP12;
pub const COL_AUX_OFFSET: usize = COL_ACC_POST_OFFSET + LIMBS_PER_FP12;
pub const COL_IS_REAL: usize = COL_AUX_OFFSET + LIMBS_PER_FP12;
pub const COL_SEL_MILLER_DOUBLING: usize = COL_IS_REAL + 1;
pub const COL_SEL_MILLER_ADDITION: usize = COL_SEL_MILLER_DOUBLING + 1;
pub const COL_SEL_FINAL_EXP_EASY: usize = COL_SEL_MILLER_ADDITION + 1;
pub const COL_SEL_FINAL_EXP_HARD: usize = COL_SEL_FINAL_EXP_EASY + 1;
pub const NUM_COLUMNS: usize = COL_SEL_FINAL_EXP_HARD + 1;

/// Row-local constraints:
///
///   0. `is_real ∈ {0, 1}`
///   1. `sel_miller_doubling ∈ {0, 1}`
///   2. `sel_miller_addition ∈ {0, 1}`
///   3. `sel_final_exp_easy ∈ {0, 1}`
///   4. `sel_final_exp_hard ∈ {0, 1}`
///   5. `Σ sel_* = is_real`  (exactly one selector on real rows)
///   6. Pairwise mutex of all 4 selectors (one body summing the 6
///      pairs).
pub const NUM_ROW_CONSTRAINTS: usize = 7;

pub const NUM_SHIFTED: usize = 0;

// ─── Trace builder ────────────────────────────────────────────────────

fn write_fp_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    fp: &Fp,
    curve: CurveType,
) {
    for j in 0..LIMBS_PER_BN254_FP {
        columns[base + j][r] = Scalar::from_u64(fp.limbs[j], curve);
    }
}

fn write_fp12_limbs(
    columns: &mut [Vec<Scalar>],
    base: usize,
    r: usize,
    f: &Fp12,
    curve: CurveType,
) {
    // Layout matches miller_step_air's Fp12 flattening:
    //   [c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1,
    //    c1.c0.c0, c1.c0.c1, c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1]
    // each one LIMBS_PER_BN254_FP limbs big-endian.
    let fps: [&Fp; FP_PER_FP12] = [
        &f.c0.c0.c0, &f.c0.c0.c1,
        &f.c0.c1.c0, &f.c0.c1.c1,
        &f.c0.c2.c0, &f.c0.c2.c1,
        &f.c1.c0.c0, &f.c1.c0.c1,
        &f.c1.c1.c0, &f.c1.c1.c1,
        &f.c1.c2.c0, &f.c1.c2.c1,
    ];
    for (i, fp) in fps.iter().enumerate() {
        write_fp_limbs(columns, base + i * LIMBS_PER_BN254_FP, r, fp, curve);
    }
}

pub fn build_trace_polynomials(
    witness: &Bn254PairingInternalsWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        write_fp12_limbs(&mut columns, COL_ACC_PRE_OFFSET, r, &row.acc_pre, curve);
        write_fp12_limbs(&mut columns, COL_ACC_POST_OFFSET, r, &row.acc_post, curve);
        write_fp12_limbs(&mut columns, COL_AUX_OFFSET, r, &row.aux, curve);
        columns[COL_IS_REAL][r] = if row.is_real { one.clone() } else { zero.clone() };
        if row.is_real {
            let sel = match row.op {
                InternalsOp::MillerDoubling => COL_SEL_MILLER_DOUBLING,
                InternalsOp::MillerAddition => COL_SEL_MILLER_ADDITION,
                InternalsOp::FinalExpEasy => COL_SEL_FINAL_EXP_EASY,
                InternalsOp::FinalExpHard => COL_SEL_FINAL_EXP_HARD,
            };
            columns[sel][r] = one.clone();
        }
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct Bn254PairingInternalsConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bn254PairingInternalsConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn bin(v: &Scalar) -> Scalar {
    let curve = v.curve_type();
    let one = Scalar::one(curve);
    v.mul(&v.sub(&one))
}

const SEL_INDICES: [usize; 4] = [
    COL_SEL_MILLER_DOUBLING,
    COL_SEL_MILLER_ADDITION,
    COL_SEL_FINAL_EXP_EASY,
    COL_SEL_FINAL_EXP_HARD,
];

impl VmConstraintSystem for Bn254PairingInternalsConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        [
            "is_real_binary",
            "sel_miller_doubling_binary",
            "sel_miller_addition_binary",
            "sel_final_exp_easy_binary",
            "sel_final_exp_hard_binary",
            "selectors_sum_to_is_real",
            "selectors_pairwise_mutex_sum",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let n = columns[0].len();
        let curve = columns[0][0].curve_type();
        let mut out: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            out[0][r] = bin(is_real);
            let mut sels: [Scalar; 4] = [
                Scalar::zero(curve),
                Scalar::zero(curve),
                Scalar::zero(curve),
                Scalar::zero(curve),
            ];
            for k in 0..4 {
                sels[k] = columns[SEL_INDICES[k]][r].clone();
                out[1 + k][r] = bin(&sels[k]);
            }
            let mut sum = Scalar::zero(curve);
            for s in &sels {
                sum = sum.add(s);
            }
            out[5][r] = sum.sub(is_real);
            // Pairwise mutex: Σ_{i<j} sel_i · sel_j = 0.
            let mut mutex = Scalar::zero(curve);
            for i in 0..4 {
                for j in (i + 1)..4 {
                    mutex = mutex.add(&sels[i].mul(&sels[j]));
                }
            }
            out[6][r] = mutex;
        }
        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let is_real = &col_evals[COL_IS_REAL];
        let sels: [Scalar; 4] = [
            col_evals[SEL_INDICES[0]].clone(),
            col_evals[SEL_INDICES[1]].clone(),
            col_evals[SEL_INDICES[2]].clone(),
            col_evals[SEL_INDICES[3]].clone(),
        ];
        let mut sum = Scalar::zero(curve);
        for s in &sels { sum = sum.add(s); }
        let mut mutex = Scalar::zero(curve);
        for i in 0..4 {
            for j in (i + 1)..4 {
                mutex = mutex.add(&sels[i].mul(&sels[j]));
            }
        }
        let parts = [
            bin(is_real),
            bin(&sels[0]),
            bin(&sels[1]),
            bin(&sels[2]),
            bin(&sels[3]),
            sum.sub(is_real),
            mutex,
        ];
        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);
        for p in &parts {
            acc = acc.add(&alpha_pow.mul(p));
            alpha_pow = alpha_pow.mul(alpha);
        }
        acc
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let bin_poly = |v: &Vec<Scalar>| -> Vec<Scalar> {
            let v_m1 = poly_sub(v, &one_poly, curve);
            poly_mul(v, &v_m1, curve)
        };

        let sels: [&Vec<Scalar>; 4] = [
            &col_coeffs[SEL_INDICES[0]],
            &col_coeffs[SEL_INDICES[1]],
            &col_coeffs[SEL_INDICES[2]],
            &col_coeffs[SEL_INDICES[3]],
        ];
        let mut sum_poly: Vec<Scalar> = vec![Scalar::zero(curve)];
        for s in &sels {
            sum_poly = poly_add(&sum_poly, s, curve);
        }
        let mut mutex_poly: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..4 {
            for j in (i + 1)..4 {
                mutex_poly = poly_add(&mutex_poly, &poly_mul(sels[i], sels[j], curve), curve);
            }
        }
        let parts: [Vec<Scalar>; NUM_ROW_CONSTRAINTS] = [
            bin_poly(&col_coeffs[COL_IS_REAL]),
            bin_poly(sels[0]),
            bin_poly(sels[1]),
            bin_poly(sels[2]),
            bin_poly(sels[3]),
            poly_sub(&sum_poly, &col_coeffs[COL_IS_REAL], curve),
            mutex_poly,
        ];
        let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);
        for p in &parts {
            acc = poly_add(&acc, &poly_scalar_mul(p, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        let mut v = vec![COL_IS_REAL];
        v.extend_from_slice(&SEL_INDICES);
        v
    }

    fn padding_selector_column(&self) -> Option<usize> {
        // Returning `Some(COL_IS_REAL)` would cause the prover to set
        // `is_real = 1` on padding rows, which violates
        // `Σ selectors − is_real = 0` (all selectors are zero on padding).
        // All constraints already vanish on the all-zero padding row, so
        // no padding selector is needed.
        None
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        Vec::new()
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(op: InternalsOp) -> Bn254PairingInternalsRow {
        let one = Fp12::one();
        let pre = Fp12 {
            c0: Fp6 { c0: Fp2 { c0: Fp::from_u64(2), c1: Fp::from_u64(3) }, c1: Fp2::zero(), c2: Fp2::zero() },
            c1: Fp6::zero(),
        };
        Bn254PairingInternalsRow {
            acc_pre: pre,
            acc_post: one,
            aux: one,
            op,
            is_real: true,
        }
    }

    #[test]
    fn fp12_zero_one_identities() {
        let z = Fp12::zero();
        let o = Fp12::one();
        assert!(z.is_zero());
        assert!(!o.is_zero());
        assert_eq!(o.c1, Fp6::zero());
        assert_eq!(o.c0, Fp6::one());
    }

    #[test]
    fn column_layout_is_packed() {
        // 3 × Fp12 = 3 × 48 limbs = 144 limbs, plus 1 is_real + 4 selectors
        // = 149 cols.
        assert_eq!(LIMBS_PER_FP12, 48);
        assert_eq!(NUM_COLUMNS, 3 * LIMBS_PER_FP12 + 5);
        assert_eq!(NUM_COLUMNS, 149);
        assert_eq!(COL_ACC_PRE_OFFSET, 0);
        assert_eq!(COL_ACC_POST_OFFSET, 48);
        assert_eq!(COL_AUX_OFFSET, 96);
        assert_eq!(COL_IS_REAL, 144);
    }

    #[test]
    fn trace_builder_populates_shape() {
        let mut w = Bn254PairingInternalsWitness::new();
        w.push(sample_row(InternalsOp::MillerDoubling));
        w.push(sample_row(InternalsOp::MillerAddition));
        w.push(sample_row(InternalsOp::FinalExpEasy));
        w.push(sample_row(InternalsOp::FinalExpHard));
        let trace =
            build_trace_polynomials(&w, CurveType::Bls12381);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 4);
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);
        // Each row has one of the four selectors set.
        assert!(trace.columns[COL_SEL_MILLER_DOUBLING].evaluations[0]
            .sub(&one)
            .is_zero());
        assert!(trace.columns[COL_SEL_MILLER_ADDITION].evaluations[1]
            .sub(&one)
            .is_zero());
        assert!(trace.columns[COL_SEL_FINAL_EXP_EASY].evaluations[2]
            .sub(&one)
            .is_zero());
        assert!(trace.columns[COL_SEL_FINAL_EXP_HARD].evaluations[3]
            .sub(&one)
            .is_zero());
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let mut w = Bn254PairingInternalsWitness::new();
        w.push(sample_row(InternalsOp::MillerDoubling));
        w.push(sample_row(InternalsOp::MillerAddition));
        w.push(sample_row(InternalsOp::FinalExpEasy));
        w.push(sample_row(InternalsOp::FinalExpHard));
        let trace =
            build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = Bn254PairingInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} at row {} = {:?}",
                    i, r, v,
                );
            }
        }
    }

    #[test]
    fn tampered_two_selectors_high_fires_mutex() {
        let mut w = Bn254PairingInternalsWitness::new();
        w.push(sample_row(InternalsOp::MillerDoubling));
        let trace =
            build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        cols[COL_SEL_MILLER_ADDITION][0] = Scalar::one(curve);
        let cs = Bn254PairingInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Mutex constraint (index 6) must fire: doubling * addition = 1.
        assert!(!results[6][0].is_zero());
        // Sum-to-is_real (index 5) must also fire: 2 - 1 = 1.
        assert!(!results[5][0].is_zero());
    }

    #[test]
    fn tampered_is_real_one_no_selector_fires_sum_constraint() {
        let mut w = Bn254PairingInternalsWitness::new();
        w.push(sample_row(InternalsOp::MillerDoubling));
        let trace =
            build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Clear sel_miller_doubling on row 0 — is_real still 1, no
        // selectors high → sum constraint fires.
        cols[COL_SEL_MILLER_DOUBLING][0] = Scalar::zero(curve);
        let cs = Bn254PairingInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[5][0].is_zero());
    }

    #[test]
    fn padding_row_zeros_constraints() {
        let w = Bn254PairingInternalsWitness::new();
        let trace =
            build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = Bn254PairingInternalsConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} at row {} on padding-only trace must vanish",
                    i, r,
                );
            }
        }
    }

    #[test]
    #[ignore = "slow: standalone prove+verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let mut w = Bn254PairingInternalsWitness::new();
        w.push(sample_row(InternalsOp::MillerDoubling));
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = Bn254PairingInternalsConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone bn254_pairing_internals_air proof must verify",
        );
    }
}
