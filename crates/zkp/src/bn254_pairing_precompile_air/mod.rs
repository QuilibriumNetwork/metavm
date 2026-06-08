//! alt_bn128 ECPAIRING precompile AIR (`0x08`, EIP-197).
//!
//! # Purpose
//!
//! The Ethereum precompile `0x08` consumes `k` pairs of points
//! `(P_i ∈ G1, Q_i ∈ G2)` and outputs `0x...01` iff the optimal-Ate
//! pairing equation `Π_i e(P_i, Q_i) = 1_{G_T}` holds, otherwise it
//! outputs zero. The standard wire format is:
//!
//!   `k × 192` bytes input: per pair
//!       `Px[32] || Py[32] || Qx_c1[32] || Qx_c0[32] || Qy_c1[32] || Qy_c0[32]`
//!   `32` bytes output: zero-padded `0x00..00 || result_byte`
//!
//! Gas cost (Istanbul / EIP-1108): `45_000 + 34_000 · k`.
//!
//! # Algebraic surface (this AIR)
//!
//! This AIR commits one ECPAIRING invocation per row at the **shape**
//! level. Each row pins:
//!
//!   - the up-to `MAX_PAIRS × 192` input byte columns,
//!   - the 32-byte output (forced to `0..0 || pairing_result`),
//!   - the gas cost (forced to `45_000 + 34_000 · num_pairs`),
//!   - the `num_pairs` selector decomposition (one-hot over `0..=MAX_PAIRS`),
//!   - per-pair `pair_active` selectors (`pair_active[i] = 1` iff
//!     `num_pairs > i`).
//!
//! ## The pairing equation itself is **deferred**
//!
//! BN254's optimal-Ate pairing requires a full Miller loop + final
//! exponentiation gadget over `Fq^{12}` — a substantial sub-AIR family
//! (see `miller_loop_air`, `final_exp_air` for BLS12-381). Until a
//! BN254-flavoured version of those gadgets is wired, `pairing_result`
//! is **a witness commitment only** — the host fills it with the truth
//! value of `Π e(P_i, Q_i) == 1`. The shape constraints below
//! algebraically pin the precompile IO layout + gas accounting, and the
//! deferred cross-AIR descriptor `make_pairing_to_bls_pairing_air_descriptor`
//! documents the slot where the per-pair pairing gadget linkage will
//! plug in.
//!
//! # Cross-AIR LogUp descriptors
//!
//!   - [`make_pairing_to_precompile_dispatch_descriptor`] —
//!     binds `(sel_ecpairing, gas_cost, num_pairs)` on real rows to the
//!     matching `precompile_air` dispatch row (callee `0x08`).
//!   - [`make_pairing_to_precompile_io_descriptor`] —
//!     binds `(input_bytes[0..MAX_INPUT_LENGTH], output[0..32])` to
//!     `precompile_io_air` (memory glue).
//!   - [`make_pairing_to_bls_pairing_air_descriptor`] —
//!     **stub**, per-pair binding `(Px, Py, Qx_c1, Qx_c0, Qy_c1, Qy_c0)`
//!     to the future BN254 Miller-loop gadget, gated by
//!     `pair_active[i]`. NOTE: this currently labels through
//!     `bls_pairing_air` for parallel structure — the orchestrator will
//!     re-target it once a dedicated `bn254_pairing_air` ships.
//!
//! Same placeholder-sentinel convention as the rest of the precompile
//! family: cross-AIR column indices into other tables are stubbed with
//! `usize::MAX` and substituted by the joint-prover orchestrator when
//! the layers are wired.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum number of `(P, Q)` pairs supported in a single row (scaffold).
pub const MAX_PAIRS: usize = 4;
/// Bytes per pair on the wire (`Px||Py||Qx_c1||Qx_c0||Qy_c1||Qy_c0`).
pub const PAIR_BYTES: usize = 192;
/// Big-endian byte length of a base-field element.
pub const FE_BYTES: usize = 32;
/// Maximum input length (scaffold cap = `MAX_PAIRS × PAIR_BYTES`).
pub const MAX_INPUT_LENGTH: usize = MAX_PAIRS * PAIR_BYTES;
/// Output length for ECPAIRING (zero-padded `bool32`).
pub const OUTPUT_LENGTH: usize = 32;

/// EIP precompile id for ECPAIRING.
pub const PC_ECPAIRING: u64 = 0x08;
/// EIP-1108 gas base cost.
pub const GAS_PAIRING_BASE: u64 = 45_000;
/// EIP-1108 per-pair gas cost.
pub const GAS_PAIRING_PER_PAIR: u64 = 34_000;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_INPUT_OFFSET: usize = 0; // 0..MAX_INPUT_LENGTH
pub const COL_OUTPUT_OFFSET: usize = COL_INPUT_OFFSET + MAX_INPUT_LENGTH; // 768..800

/// One-hot decomposition selector for `num_pairs ∈ {0, 1, ..., MAX_PAIRS}`.
pub const COL_IS_NPAIRS_EQ_OFFSET: usize = COL_OUTPUT_OFFSET + OUTPUT_LENGTH;
pub const NUM_NPAIRS_SLOTS: usize = MAX_PAIRS + 1; // 0..=MAX_PAIRS inclusive

/// Per-pair activity selectors: `pair_active[i] = 1` iff `num_pairs > i`.
pub const COL_PAIR_ACTIVE_OFFSET: usize =
    COL_IS_NPAIRS_EQ_OFFSET + NUM_NPAIRS_SLOTS;

pub const COL_NUM_PAIRS: usize = COL_PAIR_ACTIVE_OFFSET + MAX_PAIRS;
pub const COL_GAS_COST: usize = COL_NUM_PAIRS + 1;
pub const COL_PAIRING_RESULT: usize = COL_GAS_COST + 1;
pub const COL_IS_REAL: usize = COL_PAIRING_RESULT + 1;
pub const COL_SEL_ECPAIRING: usize = COL_IS_REAL + 1;

pub const NUM_COLUMNS: usize = COL_SEL_ECPAIRING + 1;

/// Row-local constraint count.
///
/// Layout (see [`Bn254PairingPrecompileConstraintSystem::evaluate_on_domain`]):
///   0:   `is_real` binary
///   1:   `sel_ecpairing` binary
///   2:   `pairing_result` binary
///   3:   `sel_ecpairing = is_real`
///   4..36: `output[i] = 0` for `i in 0..OUTPUT_LENGTH-1`
///   36:  `output[31] = pairing_result`
///   37:  `gas_cost - is_real · (GAS_BASE + GAS_PER_PAIR · num_pairs) = 0`
///   38:  `Σ is_npairs_eq[k] = is_real`
///   39:  `Σ k · is_npairs_eq[k] = num_pairs`
///   40..40+NUM_NPAIRS_SLOTS: `is_npairs_eq[k]` binary
///   ... + MAX_PAIRS: `pair_active[i]` binary
///   ... + MAX_PAIRS: per-pair activity binding to `is_npairs_eq`
pub const NUM_ROW_CONSTRAINTS: usize =
      4                       // is_real bin + sel bin + result bin + sel=is_real
    + (OUTPUT_LENGTH - 1)     // output[0..31] = 0
    + 1                       // output[31] = pairing_result
    + 1                       // gas accounting
    + 1                       // sum of npairs slots = is_real
    + 1                       // weighted sum = num_pairs
    + NUM_NPAIRS_SLOTS        // is_npairs_eq binary
    + MAX_PAIRS               // pair_active binary
    + MAX_PAIRS;              // pair_active = Σ_{k>i} is_npairs_eq[k]

pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// Single ECPAIRING invocation witness. `MAX_PAIRS` is the scaffold
/// cap; rows with fewer pairs zero-pad the trailing inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bn254PairingPrecompileWitness {
    /// Number of `(P, Q)` pairs (`0..=MAX_PAIRS`).
    pub num_pairs: u8,
    /// Concatenated wire-format input, zero-padded to `MAX_INPUT_LENGTH`.
    pub input_bytes: [u8; MAX_INPUT_LENGTH],
    /// 32-byte precompile output (`0..0 || pairing_result`).
    pub output: [u8; OUTPUT_LENGTH],
    /// `45_000 + 34_000 · num_pairs`.
    pub gas_cost: u64,
    /// Truth value of `Π e(P_i, Q_i) == 1_{G_T}`. Witness-committed only.
    pub pairing_result: bool,
    /// Real (non-padding) row marker.
    pub is_real: bool,
}

impl Bn254PairingPrecompileWitness {
    /// Build an empty (padding) row.
    pub fn padding() -> Self {
        Self {
            num_pairs: 0,
            input_bytes: [0u8; MAX_INPUT_LENGTH],
            output: [0u8; OUTPUT_LENGTH],
            gas_cost: 0,
            pairing_result: false,
            is_real: false,
        }
    }

    /// Build an honest witness from a list of pairs and a host-provided
    /// truth value. The wire encoding follows EIP-197:
    /// `Px || Py || Qx_c1 || Qx_c0 || Qy_c1 || Qy_c0` per pair.
    ///
    /// Caller is responsible for the actual pairing arithmetic — see
    /// [`from_pairs`] for the convenience entry point used by the test
    /// suite.
    pub fn new(
        num_pairs: u8,
        input_bytes: [u8; MAX_INPUT_LENGTH],
        pairing_result: bool,
    ) -> Self {
        assert!(
            num_pairs as usize <= MAX_PAIRS,
            "num_pairs {} exceeds MAX_PAIRS = {}",
            num_pairs,
            MAX_PAIRS,
        );
        let mut output = [0u8; OUTPUT_LENGTH];
        output[OUTPUT_LENGTH - 1] = if pairing_result { 1 } else { 0 };
        let gas_cost =
            GAS_PAIRING_BASE + GAS_PAIRING_PER_PAIR * num_pairs as u64;
        Self {
            num_pairs,
            input_bytes,
            output,
            gas_cost,
            pairing_result,
            is_real: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Bn254PairingPrecompileTraceWitness {
    pub rows: Vec<Bn254PairingPrecompileWitness>,
}

impl Bn254PairingPrecompileTraceWitness {
    pub fn from_rows(rows: Vec<Bn254PairingPrecompileWitness>) -> Self {
        Self { rows }
    }
    pub fn push(&mut self, row: Bn254PairingPrecompileWitness) {
        self.rows.push(row);
    }
}

// ─── Host-side helpers ────────────────────────────────────────────────

/// Per-pair wire-format tuple:
///   `((Px, Py), (Qx_c1, Qx_c0, Qy_c1, Qy_c0))`
pub type PairBytes = (
    ([u8; FE_BYTES], [u8; FE_BYTES]),
    ([u8; FE_BYTES], [u8; FE_BYTES], [u8; FE_BYTES], [u8; FE_BYTES]),
);

/// Pack a list of pairs into a single honest ECPAIRING witness with the
/// pairing result hardcoded to `true` (intended for known-good test
/// vectors where the host has verified the pairing externally). The
/// `pairing_result` column is a witness commitment only — a malicious
/// host could lie here and the AIR would not catch it until the
/// deferred BN254 Miller-loop gadget lands.
pub fn from_pairs(pairs: &[PairBytes]) -> Bn254PairingPrecompileWitness {
    assert!(
        pairs.len() <= MAX_PAIRS,
        "ECPAIRING scaffold supports up to {} pairs (got {})",
        MAX_PAIRS,
        pairs.len(),
    );
    let mut input = [0u8; MAX_INPUT_LENGTH];
    for (i, pair) in pairs.iter().enumerate() {
        let base = i * PAIR_BYTES;
        let ((px, py), (qx_c1, qx_c0, qy_c1, qy_c0)) = pair;
        input[base..base + 32].copy_from_slice(px);
        input[base + 32..base + 64].copy_from_slice(py);
        input[base + 64..base + 96].copy_from_slice(qx_c1);
        input[base + 96..base + 128].copy_from_slice(qx_c0);
        input[base + 128..base + 160].copy_from_slice(qy_c1);
        input[base + 160..base + 192].copy_from_slice(qy_c0);
    }
    Bn254PairingPrecompileWitness::new(pairs.len() as u8, input, true)
}

/// Empty-input ECPAIRING witness: by EIP-197, an empty input returns
/// `0x00..01` (the empty product equals identity, which equals 1).
/// Gas cost is the base `45_000`.
pub fn zero_pair_identity_witness() -> Bn254PairingPrecompileWitness {
    Bn254PairingPrecompileWitness::new(0, [0u8; MAX_INPUT_LENGTH], true)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Bn254PairingPrecompileTraceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..MAX_INPUT_LENGTH {
            columns[COL_INPUT_OFFSET + k][r] =
                Scalar::from_u64(row.input_bytes[k] as u64, curve);
        }
        for k in 0..OUTPUT_LENGTH {
            columns[COL_OUTPUT_OFFSET + k][r] =
                Scalar::from_u64(row.output[k] as u64, curve);
        }
        // One-hot is_npairs_eq.
        for k in 0..NUM_NPAIRS_SLOTS {
            let v = if row.is_real && row.num_pairs as usize == k {
                one.clone()
            } else {
                zero.clone()
            };
            columns[COL_IS_NPAIRS_EQ_OFFSET + k][r] = v;
        }
        // pair_active[i] = 1 iff num_pairs > i (and row is real).
        for i in 0..MAX_PAIRS {
            let v = if row.is_real && (row.num_pairs as usize) > i {
                one.clone()
            } else {
                zero.clone()
            };
            columns[COL_PAIR_ACTIVE_OFFSET + i][r] = v;
        }
        columns[COL_NUM_PAIRS][r] = Scalar::from_u64(row.num_pairs as u64, curve);
        columns[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
        columns[COL_PAIRING_RESULT][r] =
            Scalar::from_u64(row.pairing_result as u64, curve);
        columns[COL_IS_REAL][r] = if row.is_real { one.clone() } else { zero.clone() };
        columns[COL_SEL_ECPAIRING][r] =
            if row.is_real { one.clone() } else { zero.clone() };
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct Bn254PairingPrecompileConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bn254PairingPrecompileConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Bn254PairingPrecompileConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        labels.push("sel_ecpairing_binary".into());
        labels.push("pairing_result_binary".into());
        labels.push("sel_ecpairing_eq_is_real".into());
        for k in 0..OUTPUT_LENGTH - 1 {
            labels.push(format!("output_zero_byte_{}", k));
        }
        labels.push("output_lsb_eq_pairing_result".into());
        labels.push("gas_cost_eq_base_plus_per_pair_npairs".into());
        labels.push("npairs_slots_sum_is_real".into());
        labels.push("weighted_npairs_eq_num_pairs".into());
        for k in 0..NUM_NPAIRS_SLOTS {
            labels.push(format!("is_npairs_eq_{}_binary", k));
        }
        for i in 0..MAX_PAIRS {
            labels.push(format!("pair_active_{}_binary", i));
        }
        for i in 0..MAX_PAIRS {
            labels.push(format!("pair_active_{}_eq_sum_npairs_gt_{}", i, i));
        }
        labels
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let gas_base = Scalar::from_u64(GAS_PAIRING_BASE, curve);
        let gas_per = Scalar::from_u64(GAS_PAIRING_PER_PAIR, curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 1: sel_ecpairing binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_SEL_ECPAIRING][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 2: pairing_result binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_PAIRING_RESULT][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 3: sel_ecpairing = is_real.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_SEL_ECPAIRING][r]
                    .sub(&columns[COL_IS_REAL][r]);
            }
            out.push(c);
        }
        // output[i] = 0 for i in 0..31.
        for k in 0..OUTPUT_LENGTH - 1 {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_OUTPUT_OFFSET + k][r].clone();
            }
            out.push(c);
        }
        // output[31] = pairing_result.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_OUTPUT_OFFSET + OUTPUT_LENGTH - 1][r]
                    .sub(&columns[COL_PAIRING_RESULT][r]);
            }
            out.push(c);
        }
        // gas_cost - is_real · (GAS_BASE + GAS_PER_PAIR · num_pairs) = 0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let expected = gas_base
                    .add(&gas_per.mul(&columns[COL_NUM_PAIRS][r]));
                let gated = columns[COL_IS_REAL][r].mul(&expected);
                c[r] = columns[COL_GAS_COST][r].sub(&gated);
            }
            out.push(c);
        }
        // Σ is_npairs_eq[k] = is_real.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut acc = Scalar::zero(curve);
                for k in 0..NUM_NPAIRS_SLOTS {
                    acc = acc.add(&columns[COL_IS_NPAIRS_EQ_OFFSET + k][r]);
                }
                c[r] = acc.sub(&columns[COL_IS_REAL][r]);
            }
            out.push(c);
        }
        // Σ k · is_npairs_eq[k] = num_pairs.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut acc = Scalar::zero(curve);
                for k in 0..NUM_NPAIRS_SLOTS {
                    let kf = Scalar::from_u64(k as u64, curve);
                    acc = acc.add(
                        &kf.mul(&columns[COL_IS_NPAIRS_EQ_OFFSET + k][r]),
                    );
                }
                c[r] = acc.sub(&columns[COL_NUM_PAIRS][r]);
            }
            out.push(c);
        }
        // is_npairs_eq[k] binary.
        for k in 0..NUM_NPAIRS_SLOTS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_NPAIRS_EQ_OFFSET + k][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // pair_active[i] binary.
        for i in 0..MAX_PAIRS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_PAIR_ACTIVE_OFFSET + i][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // pair_active[i] = Σ_{k > i} is_npairs_eq[k].
        for i in 0..MAX_PAIRS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut acc = Scalar::zero(curve);
                for k in (i + 1)..NUM_NPAIRS_SLOTS {
                    acc = acc.add(&columns[COL_IS_NPAIRS_EQ_OFFSET + k][r]);
                }
                c[r] = columns[COL_PAIR_ACTIVE_OFFSET + i][r].sub(&acc);
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
        let gas_base = Scalar::from_u64(GAS_PAIRING_BASE, curve);
        let gas_per = Scalar::from_u64(GAS_PAIRING_PER_PAIR, curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0..3
        let bodies_0_3 = [
            col_evals[COL_IS_REAL].mul(&col_evals[COL_IS_REAL].sub(&one)),
            col_evals[COL_SEL_ECPAIRING]
                .mul(&col_evals[COL_SEL_ECPAIRING].sub(&one)),
            col_evals[COL_PAIRING_RESULT]
                .mul(&col_evals[COL_PAIRING_RESULT].sub(&one)),
            col_evals[COL_SEL_ECPAIRING].sub(&col_evals[COL_IS_REAL]),
        ];
        for body in &bodies_0_3 {
            acc = acc.add(&alpha_pow.mul(body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // output[i] = 0 for i in 0..31.
        for k in 0..OUTPUT_LENGTH - 1 {
            acc = acc.add(&alpha_pow.mul(&col_evals[COL_OUTPUT_OFFSET + k]));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // output[31] = pairing_result.
        {
            let body = col_evals[COL_OUTPUT_OFFSET + OUTPUT_LENGTH - 1]
                .sub(&col_evals[COL_PAIRING_RESULT]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // gas accounting.
        {
            let expected = gas_base.add(&gas_per.mul(&col_evals[COL_NUM_PAIRS]));
            let gated = col_evals[COL_IS_REAL].mul(&expected);
            let body = col_evals[COL_GAS_COST].sub(&gated);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Σ npairs slots = is_real.
        {
            let mut sum = Scalar::zero(curve);
            for k in 0..NUM_NPAIRS_SLOTS {
                sum = sum.add(&col_evals[COL_IS_NPAIRS_EQ_OFFSET + k]);
            }
            let body = sum.sub(&col_evals[COL_IS_REAL]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Σ k · is_npairs_eq[k] = num_pairs.
        {
            let mut sum = Scalar::zero(curve);
            for k in 0..NUM_NPAIRS_SLOTS {
                let kf = Scalar::from_u64(k as u64, curve);
                sum = sum.add(&kf.mul(&col_evals[COL_IS_NPAIRS_EQ_OFFSET + k]));
            }
            let body = sum.sub(&col_evals[COL_NUM_PAIRS]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // is_npairs_eq binary.
        for k in 0..NUM_NPAIRS_SLOTS {
            let v = &col_evals[COL_IS_NPAIRS_EQ_OFFSET + k];
            let body = v.mul(&v.sub(&one));
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // pair_active binary.
        for i in 0..MAX_PAIRS {
            let v = &col_evals[COL_PAIR_ACTIVE_OFFSET + i];
            let body = v.mul(&v.sub(&one));
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // pair_active[i] = Σ_{k > i} is_npairs_eq[k].
        for i in 0..MAX_PAIRS {
            let mut sum = Scalar::zero(curve);
            for k in (i + 1)..NUM_NPAIRS_SLOTS {
                sum = sum.add(&col_evals[COL_IS_NPAIRS_EQ_OFFSET + k]);
            }
            let body = col_evals[COL_PAIR_ACTIVE_OFFSET + i].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
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
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let gas_base_poly = vec![Scalar::from_u64(GAS_PAIRING_BASE, curve)];
        let gas_per_poly = vec![Scalar::from_u64(GAS_PAIRING_PER_PAIR, curve)];

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        macro_rules! push {
            ($body:expr) => {{
                acc = poly_add(&acc, &poly_scalar_mul(&$body, &alpha_pow), curve);
                alpha_pow = alpha_pow.mul(alpha);
            }};
        }

        // 0: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let body = poly_mul(v, &poly_sub(v, &one_poly, curve), curve);
            push!(body);
        }
        // 1: sel_ecpairing binary.
        {
            let v = &col_coeffs[COL_SEL_ECPAIRING];
            let body = poly_mul(v, &poly_sub(v, &one_poly, curve), curve);
            push!(body);
        }
        // 2: pairing_result binary.
        {
            let v = &col_coeffs[COL_PAIRING_RESULT];
            let body = poly_mul(v, &poly_sub(v, &one_poly, curve), curve);
            push!(body);
        }
        // 3: sel_ecpairing = is_real.
        {
            let body = poly_sub(
                &col_coeffs[COL_SEL_ECPAIRING],
                &col_coeffs[COL_IS_REAL],
                curve,
            );
            push!(body);
        }
        // output[i] = 0 for i in 0..31.
        for k in 0..OUTPUT_LENGTH - 1 {
            let body = col_coeffs[COL_OUTPUT_OFFSET + k].clone();
            push!(body);
        }
        // output[31] = pairing_result.
        {
            let body = poly_sub(
                &col_coeffs[COL_OUTPUT_OFFSET + OUTPUT_LENGTH - 1],
                &col_coeffs[COL_PAIRING_RESULT],
                curve,
            );
            push!(body);
        }
        // gas accounting.
        {
            let gas_term = poly_mul(&gas_per_poly, &col_coeffs[COL_NUM_PAIRS], curve);
            let expected = poly_add(&gas_base_poly, &gas_term, curve);
            let gated = poly_mul(&col_coeffs[COL_IS_REAL], &expected, curve);
            let body = poly_sub(&col_coeffs[COL_GAS_COST], &gated, curve);
            push!(body);
        }
        // Σ npairs = is_real.
        {
            let mut sum = vec![Scalar::zero(curve)];
            for k in 0..NUM_NPAIRS_SLOTS {
                sum = poly_add(&sum, &col_coeffs[COL_IS_NPAIRS_EQ_OFFSET + k], curve);
            }
            let body = poly_sub(&sum, &col_coeffs[COL_IS_REAL], curve);
            push!(body);
        }
        // Σ k · is_npairs_eq[k] = num_pairs.
        {
            let mut sum = vec![Scalar::zero(curve)];
            for k in 0..NUM_NPAIRS_SLOTS {
                let kf = vec![Scalar::from_u64(k as u64, curve)];
                let term = poly_mul(
                    &kf,
                    &col_coeffs[COL_IS_NPAIRS_EQ_OFFSET + k],
                    curve,
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&sum, &col_coeffs[COL_NUM_PAIRS], curve);
            push!(body);
        }
        // is_npairs_eq[k] binary.
        for k in 0..NUM_NPAIRS_SLOTS {
            let v = &col_coeffs[COL_IS_NPAIRS_EQ_OFFSET + k];
            let body = poly_mul(v, &poly_sub(v, &one_poly, curve), curve);
            push!(body);
        }
        // pair_active[i] binary.
        for i in 0..MAX_PAIRS {
            let v = &col_coeffs[COL_PAIR_ACTIVE_OFFSET + i];
            let body = poly_mul(v, &poly_sub(v, &one_poly, curve), curve);
            push!(body);
        }
        // pair_active[i] = Σ_{k > i} is_npairs_eq[k].
        for i in 0..MAX_PAIRS {
            let mut sum = vec![Scalar::zero(curve)];
            for k in (i + 1)..NUM_NPAIRS_SLOTS {
                sum = poly_add(&sum, &col_coeffs[COL_IS_NPAIRS_EQ_OFFSET + k], curve);
            }
            let body = poly_sub(&col_coeffs[COL_PAIR_ACTIVE_OFFSET + i], &sum, curve);
            push!(body);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_SEL_ECPAIRING]
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
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        if columns.len() < NUM_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
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
        // 8-bit range check on every input + output byte.
        for k in 0..MAX_INPUT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("bn254_pairing_input_byte_{}_8bit", k),
                    column_index: COL_INPUT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..OUTPUT_LENGTH {
            declarations.push((
                LookupDeclaration {
                    label: format!("bn254_pairing_output_byte_{}_8bit", k),
                    column_index: COL_OUTPUT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // num_pairs is at most MAX_PAIRS ≤ 255: 8-bit range check binds it.
        declarations.push((
            LookupDeclaration {
                label: "bn254_pairing_num_pairs_8bit".into(),
                column_index: COL_NUM_PAIRS,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Placeholder sentinel for `precompile_air` / `precompile_io_air` /
/// future BN254 pairing gadget column indices.
pub const PRECOMPILE_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds `(sel_ecpairing, gas_cost,
/// num_pairs)` on this AIR's real rows to the matching `precompile_air`
/// dispatch row (callee `0x08`).
pub fn make_pairing_to_precompile_dispatch_descriptor(
    pairing_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> =
        vec![COL_SEL_ECPAIRING, COL_GAS_COST, COL_NUM_PAIRS];
    let b_columns: Vec<usize> = vec![
        PRECOMPILE_PLACEHOLDER, // precompile_air::sel_ecpairing
        PRECOMPILE_PLACEHOLDER, // precompile_air::gas_cost
        PRECOMPILE_PLACEHOLDER, // precompile_air::num_pairs
    ];
    CrossAirLogUpDescriptor {
        label: "bn254_pairing_to_precompile_dispatch_v1_stub".into(),
        a_layer_index: pairing_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: None,
    }
}

/// Cross-AIR LogUp descriptor (stub): binds
/// `(input_bytes[0..MAX_INPUT_LENGTH], output[0..32])` to the matching
/// `precompile_io_air` row (EVM memory glue).
pub fn make_pairing_to_precompile_io_descriptor(
    pairing_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> =
        Vec::with_capacity(MAX_INPUT_LENGTH + OUTPUT_LENGTH);
    for k in 0..MAX_INPUT_LENGTH {
        a_columns.push(COL_INPUT_OFFSET + k);
    }
    for k in 0..OUTPUT_LENGTH {
        a_columns.push(COL_OUTPUT_OFFSET + k);
    }
    let b_columns: Vec<usize> =
        vec![PRECOMPILE_PLACEHOLDER; MAX_INPUT_LENGTH + OUTPUT_LENGTH];
    CrossAirLogUpDescriptor {
        label: "bn254_pairing_to_precompile_io_v1_stub".into(),
        a_layer_index: pairing_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: None,
    }
}

/// Cross-AIR LogUp descriptor (stub): per-pair binding
/// `(Px[32], Py[32], Qx_c1[32], Qx_c0[32], Qy_c1[32], Qy_c0[32])`
/// gated by `pair_active[pair_index]` to a future BN254 pairing
/// gadget AIR.
///
/// **NOTE**: BN254 is *not* BLS12-381. The label points at
/// `bls_pairing_air` for parallel structural reference only — the
/// orchestrator MUST re-target this descriptor at a dedicated
/// `bn254_pairing_air` once one ships. This is a scaffold.
pub fn make_pairing_to_bls_pairing_air_descriptor(
    pairing_layer_index: usize,
    bls_pairing_layer_index: usize,
    pair_index: usize,
) -> CrossAirLogUpDescriptor {
    assert!(
        pair_index < MAX_PAIRS,
        "pair_index {} >= MAX_PAIRS {}",
        pair_index,
        MAX_PAIRS,
    );
    let base = COL_INPUT_OFFSET + pair_index * PAIR_BYTES;
    let mut a_columns: Vec<usize> = Vec::with_capacity(PAIR_BYTES);
    for k in 0..PAIR_BYTES {
        a_columns.push(base + k);
    }
    let b_columns: Vec<usize> = vec![PRECOMPILE_PLACEHOLDER; PAIR_BYTES];
    CrossAirLogUpDescriptor {
        label: format!(
            "bn254_pairing_pair_{}_to_bls_pairing_air_v1_stub",
            pair_index,
        ),
        a_layer_index: pairing_layer_index,
        a_columns,
        a_selector_column: Some(COL_PAIR_ACTIVE_OFFSET + pair_index),
        b_layer_index: bls_pairing_layer_index,
        b_columns,
        b_selector_column: None,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} row {} nonzero: {:?}",
                    i, r, v,
                );
            }
        }
    }

    #[test]
    fn bn254_pairing_zero_pairs_returns_one() {
        // EIP-197: empty input → output is 0x..01, gas = base only.
        let w = zero_pair_identity_witness();
        assert_eq!(w.num_pairs, 0);
        assert!(w.pairing_result);
        assert_eq!(w.gas_cost, GAS_PAIRING_BASE);
        assert_eq!(w.output[OUTPUT_LENGTH - 1], 1);
        for k in 0..OUTPUT_LENGTH - 1 {
            assert_eq!(w.output[k], 0);
        }

        let trace = build_trace_polynomials(
            &Bn254PairingPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Bn254PairingPrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn bn254_pairing_one_pair_host_reject() {
        // A single pair where the host says e(P, Q) != 1 (pairing_result = false).
        // Inputs are not algebraically constrained beyond byte-range, so any
        // bytes work; the shape constraints are what we're testing.
        let mut input = [0u8; MAX_INPUT_LENGTH];
        // Fill with arbitrary bytes for pair 0.
        for k in 0..PAIR_BYTES {
            input[k] = (k as u8).wrapping_add(7);
        }
        let w = Bn254PairingPrecompileWitness::new(1, input, false);
        assert_eq!(w.num_pairs, 1);
        assert!(!w.pairing_result);
        assert_eq!(w.gas_cost, GAS_PAIRING_BASE + GAS_PAIRING_PER_PAIR);
        assert_eq!(w.output[OUTPUT_LENGTH - 1], 0);

        let trace = build_trace_polynomials(
            &Bn254PairingPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Bn254PairingPrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn bn254_pairing_two_pairs_known_good() {
        // Two "good" pairs from a host-side oracle. We just commit
        // pairing_result = true and let the shape constraints fire.
        let pair0_p = ([1u8; FE_BYTES], [2u8; FE_BYTES]);
        let pair0_q = (
            [3u8; FE_BYTES],
            [4u8; FE_BYTES],
            [5u8; FE_BYTES],
            [6u8; FE_BYTES],
        );
        let pair1_p = ([7u8; FE_BYTES], [8u8; FE_BYTES]);
        let pair1_q = (
            [9u8; FE_BYTES],
            [10u8; FE_BYTES],
            [11u8; FE_BYTES],
            [12u8; FE_BYTES],
        );
        let w = from_pairs(&[(pair0_p, pair0_q), (pair1_p, pair1_q)]);
        assert_eq!(w.num_pairs, 2);
        assert_eq!(
            w.gas_cost,
            GAS_PAIRING_BASE + 2 * GAS_PAIRING_PER_PAIR,
        );
        assert!(w.pairing_result);
        // Check wire layout for pair 0.
        for k in 0..FE_BYTES {
            assert_eq!(w.input_bytes[k], 1);
            assert_eq!(w.input_bytes[FE_BYTES + k], 2);
            assert_eq!(w.input_bytes[2 * FE_BYTES + k], 3);
            assert_eq!(w.input_bytes[3 * FE_BYTES + k], 4);
            assert_eq!(w.input_bytes[4 * FE_BYTES + k], 5);
            assert_eq!(w.input_bytes[5 * FE_BYTES + k], 6);
        }
        // Trailing pairs 2, 3 should be zero.
        for k in 2 * PAIR_BYTES..MAX_INPUT_LENGTH {
            assert_eq!(w.input_bytes[k], 0);
        }

        let trace = build_trace_polynomials(
            &Bn254PairingPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Bn254PairingPrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn bn254_pairing_tampered_gas_detected() {
        let w = from_pairs(&[(
            ([1u8; FE_BYTES], [2u8; FE_BYTES]),
            (
                [3u8; FE_BYTES],
                [4u8; FE_BYTES],
                [5u8; FE_BYTES],
                [6u8; FE_BYTES],
            ),
        )]);
        let trace = build_trace_polynomials(
            &Bn254PairingPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper gas_cost: should fire the gas-accounting constraint.
        cols[COL_GAS_COST][0] = Scalar::from_u64(12_345, CurveType::Bls48581);

        let cs = Bn254PairingPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        // Gas constraint sits after:
        //   4 (selector/shape) + 31 (output zero bytes) + 1 (output lsb) = 36
        let gas_idx = 4 + (OUTPUT_LENGTH - 1) + 1;
        assert!(
            !bodies[gas_idx][0].is_zero(),
            "expected gas-cost constraint at index {} to fire",
            gas_idx,
        );

        // Also tamper the high output byte: should fire output-zero
        // constraint.
        let mut cols2: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols2[COL_OUTPUT_OFFSET + 3][0] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let col_refs2: Vec<&Vec<Scalar>> = cols2.iter().collect();
        let bodies2 = cs.evaluate_on_domain(&col_refs2, trace.num_rows);
        // output_zero_byte_3 sits at row-local index 4 + 3 = 7.
        let out_idx = 4 + 3;
        assert!(
            !bodies2[out_idx][0].is_zero(),
            "expected output-zero constraint at index {} to fire",
            out_idx,
        );

        // Tamper pairing_result: should fire output_lsb_eq_pairing_result.
        let mut cols3: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols3[COL_PAIRING_RESULT][0] = Scalar::zero(CurveType::Bls48581);
        let col_refs3: Vec<&Vec<Scalar>> = cols3.iter().collect();
        let bodies3 = cs.evaluate_on_domain(&col_refs3, trace.num_rows);
        let lsb_idx = 4 + (OUTPUT_LENGTH - 1);
        assert!(
            !bodies3[lsb_idx][0].is_zero(),
            "expected output_lsb_eq_pairing_result at index {} to fire",
            lsb_idx,
        );
    }

    #[test]
    fn bn254_pairing_descriptors_well_formed() {
        let d1 = make_pairing_to_precompile_dispatch_descriptor(0, 1);
        assert_eq!(d1.label, "bn254_pairing_to_precompile_dispatch_v1_stub");
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert_eq!(d1.a_columns.len(), 3);
        assert_eq!(d1.b_columns.len(), 3);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.a_columns[0], COL_SEL_ECPAIRING);
        assert_eq!(d1.a_columns[1], COL_GAS_COST);
        assert_eq!(d1.a_columns[2], COL_NUM_PAIRS);
        for b in &d1.b_columns {
            assert_eq!(*b, PRECOMPILE_PLACEHOLDER);
        }

        let d2 = make_pairing_to_precompile_io_descriptor(0, 2);
        assert_eq!(d2.label, "bn254_pairing_to_precompile_io_v1_stub");
        assert_eq!(d2.a_columns.len(), MAX_INPUT_LENGTH + OUTPUT_LENGTH);
        assert_eq!(d2.b_columns.len(), MAX_INPUT_LENGTH + OUTPUT_LENGTH);
        assert_eq!(d2.a_selector_column, Some(COL_IS_REAL));
        for k in 0..MAX_INPUT_LENGTH {
            assert_eq!(d2.a_columns[k], COL_INPUT_OFFSET + k);
        }
        for k in 0..OUTPUT_LENGTH {
            assert_eq!(
                d2.a_columns[MAX_INPUT_LENGTH + k],
                COL_OUTPUT_OFFSET + k,
            );
        }

        // Per-pair stub descriptor for each pair slot.
        for i in 0..MAX_PAIRS {
            let dp = make_pairing_to_bls_pairing_air_descriptor(0, 3, i);
            assert_eq!(
                dp.label,
                format!("bn254_pairing_pair_{}_to_bls_pairing_air_v1_stub", i),
            );
            assert_eq!(dp.a_columns.len(), PAIR_BYTES);
            assert_eq!(dp.b_columns.len(), PAIR_BYTES);
            assert_eq!(
                dp.a_selector_column,
                Some(COL_PAIR_ACTIVE_OFFSET + i),
            );
            assert_eq!(dp.a_columns[0], COL_INPUT_OFFSET + i * PAIR_BYTES);
            assert_eq!(
                dp.a_columns[PAIR_BYTES - 1],
                COL_INPUT_OFFSET + i * PAIR_BYTES + PAIR_BYTES - 1,
            );
        }
    }

    #[test]
    fn bn254_pairing_column_layout_pinned() {
        assert_eq!(MAX_PAIRS, 4);
        assert_eq!(PAIR_BYTES, 192);
        assert_eq!(FE_BYTES, 32);
        assert_eq!(MAX_INPUT_LENGTH, 4 * 192);
        assert_eq!(OUTPUT_LENGTH, 32);
        assert_eq!(PC_ECPAIRING, 0x08);
        assert_eq!(GAS_PAIRING_BASE, 45_000);
        assert_eq!(GAS_PAIRING_PER_PAIR, 34_000);

        assert_eq!(COL_INPUT_OFFSET, 0);
        assert_eq!(COL_OUTPUT_OFFSET, MAX_INPUT_LENGTH);
        assert_eq!(COL_IS_NPAIRS_EQ_OFFSET, MAX_INPUT_LENGTH + OUTPUT_LENGTH);
        assert_eq!(NUM_NPAIRS_SLOTS, MAX_PAIRS + 1);
        assert_eq!(
            COL_PAIR_ACTIVE_OFFSET,
            COL_IS_NPAIRS_EQ_OFFSET + NUM_NPAIRS_SLOTS,
        );
        assert_eq!(COL_NUM_PAIRS, COL_PAIR_ACTIVE_OFFSET + MAX_PAIRS);
        assert_eq!(COL_GAS_COST, COL_NUM_PAIRS + 1);
        assert_eq!(COL_PAIRING_RESULT, COL_GAS_COST + 1);
        assert_eq!(COL_IS_REAL, COL_PAIRING_RESULT + 1);
        assert_eq!(COL_SEL_ECPAIRING, COL_IS_REAL + 1);
        assert_eq!(NUM_COLUMNS, COL_SEL_ECPAIRING + 1);

        // 4 + 31 + 1 + 1 + 1 + 1 + 5 + 4 + 4 = 52 row-local constraints.
        let expected = 4
            + (OUTPUT_LENGTH - 1)
            + 1
            + 1
            + 1
            + 1
            + NUM_NPAIRS_SLOTS
            + MAX_PAIRS
            + MAX_PAIRS;
        assert_eq!(NUM_ROW_CONSTRAINTS, expected);
        assert_eq!(NUM_SHIFTED, 0);

        let cs = Bn254PairingPrecompileConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
    }

    #[test]
    fn bn254_pairing_npairs_selector_violations_detected() {
        // Force two is_npairs_eq slots high: the "sum = is_real" constraint
        // (and the weighted sum) must fire.
        let w = from_pairs(&[(
            ([1u8; FE_BYTES], [2u8; FE_BYTES]),
            (
                [3u8; FE_BYTES],
                [4u8; FE_BYTES],
                [5u8; FE_BYTES],
                [6u8; FE_BYTES],
            ),
        )]);
        let trace = build_trace_polynomials(
            &Bn254PairingPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Honest witness has is_npairs_eq[1] = 1. Force is_npairs_eq[3]
        // also high.
        cols[COL_IS_NPAIRS_EQ_OFFSET + 3][0] = Scalar::one(CurveType::Bls48581);

        let cs = Bn254PairingPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        // Sum-to-is_real sits at index 4 + 31 + 1 + 1 = 37.
        let sum_idx = 4 + (OUTPUT_LENGTH - 1) + 1 + 1;
        assert!(
            !bodies[sum_idx][0].is_zero(),
            "expected npairs sum constraint at index {} to fire",
            sum_idx,
        );
        // Weighted sum sits at sum_idx + 1.
        let weighted_idx = sum_idx + 1;
        assert!(
            !bodies[weighted_idx][0].is_zero(),
            "expected weighted-npairs constraint at index {} to fire",
            weighted_idx,
        );
        // pair_active[1] = 1 in the honest witness but with is_npairs_eq[3]
        // forced high, Σ_{k>1} is_npairs_eq[k] becomes 2, so the
        // pair_active[1] binding (index after pair_active binaries) fires.
        let pair_active_bind_base = sum_idx + 2 + NUM_NPAIRS_SLOTS + MAX_PAIRS;
        assert!(
            !bodies[pair_active_bind_base + 1][0].is_zero(),
            "expected pair_active[1] binding constraint to fire",
        );
    }

    #[test]
    fn bn254_pairing_evaluate_at_point_matches_domain() {
        let w = from_pairs(&[(
            ([1u8; FE_BYTES], [2u8; FE_BYTES]),
            (
                [3u8; FE_BYTES],
                [4u8; FE_BYTES],
                [5u8; FE_BYTES],
                [6u8; FE_BYTES],
            ),
        )]);
        let trace = build_trace_polynomials(
            &Bn254PairingPrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Bn254PairingPrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let alpha = Scalar::from_u64(0x9876_5432, CurveType::Bls48581);
        let mut expect = Scalar::zero(CurveType::Bls48581);
        let mut ap = Scalar::one(CurveType::Bls48581);
        for body in &bodies {
            expect = expect.add(&ap.mul(&body[0]));
            ap = ap.mul(&alpha);
        }
        let row0: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        let got = cs.evaluate_at_point(&row0, &alpha);
        assert!(expect.is_zero());
        assert!(
            got.is_zero(),
            "evaluate_at_point should be zero on honest witness",
        );
    }
}
