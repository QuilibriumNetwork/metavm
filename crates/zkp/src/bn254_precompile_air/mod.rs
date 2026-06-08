//! alt_bn128 (BN254) precompile AIRs — ECADD (`0x06`) + ECMUL (`0x07`).
//!
//! # Purpose
//!
//! Two Ethereum precompiles operate on the alt_bn128 curve introduced by
//! EIP-196 / EIP-1108:
//!
//!   - `0x06` **ECADD**: point addition. Input is
//!     `Px[32] || Py[32] || Qx[32] || Qy[32]` (128 bytes, big-endian).
//!     Output is `Rx[32] || Ry[32]` (64 bytes) where `R = P + Q` on the
//!     curve. Gas cost = 150.
//!   - `0x07` **ECMUL**: scalar multiplication. Input is
//!     `Px[32] || Py[32] || k[32]` (96 bytes). Output is
//!     `Rx[32] || Ry[32]` (64 bytes) where `R = k · P`. Gas cost = 6000.
//!
//! Identity (point at infinity) is encoded as `(0, 0)`; both precompiles
//! short-circuit identity inputs to identity outputs.
//!
//! # Algebraic surface (this AIR)
//!
//! This AIR commits one precompile invocation per row and proves the
//! **shape** of the IO buffer: that the 128-byte `input` field decomposes
//! into the `Px / Py / Qx / Qy` (and `k`) component byte arrays gated by
//! the corresponding selector, and that the 64-byte `output` field
//! decomposes into `Rx / Ry`. Per row:
//!
//!   0.        `is_real ∈ {0, 1}`
//!   1.        `sel_ecadd ∈ {0, 1}`
//!   2.        `sel_ecmul ∈ {0, 1}`
//!   3.        `sel_ecadd · sel_ecmul = 0`               (mutually exclusive)
//!   4.        `sel_ecadd + sel_ecmul - is_real = 0`     (selectors sum to is_real)
//!   5..37.    `output[i] - Rx[i] = 0` for i in 0..32
//!   37..69.   `output[32+i] - Ry[i] = 0` for i in 0..32
//!   69..101.  `is_real · (input[i] - Px[i]) = 0`    (Px shared by both)
//!   101..133. `is_real · (input[32+i] - Py[i]) = 0`  (Py shared by both)
//!   133..165. `sel_ecadd · (input[64+i] - Qx[i]) = 0`
//!   165..197. `sel_ecadd · (input[96+i] - Qy[i]) = 0`
//!   197..229. `sel_ecmul · (input[64+i] - k[i]) = 0`
//!
//! Total: **229** row-local row constraints + 8-bit range checks on every
//! byte column.
//!
//! # Algebraic curve operation is **deferred**
//!
//! BN254 point addition and scalar multiplication over a 254-bit prime
//! field cannot be expressed compactly on small-field STARK columns; the
//! algebraic gadget is a substantial effort and is **deferred**. Until
//! that lands the `Rx` / `Ry` columns are **witness commitments only**:
//! the host-side trace builder fills them from a BN254 host computation
//! (or from a hardcoded vector if no curve crate is available) and the
//! soundness of `R = P + Q` / `R = k · P` is taken on faith by this AIR.
//! The shape constraints above algebraically pin the precompile IO
//! layout.
//!
//! # Cross-AIR LogUp descriptors
//!
//!   - [`make_bn254_to_precompile_dispatch_descriptor`] —
//!     binds `(sel_ecadd, sel_ecmul, input_length, output_length=64, gas_cost)`
//!     on this AIR's real rows to the matching `precompile_air` dispatch
//!     row for callee `0x06` / `0x07`.
//!   - [`make_bn254_to_precompile_io_descriptor`] —
//!     binds `(input_bytes[0..128], output[0..64])` to the corresponding
//!     `precompile_io_air` row (memory glue).
//!
//! Same placeholder-sentinel convention as `ripemd160_precompile_air`:
//! zkp does not depend on the EVM crate, so EVM-side B-column indices
//! are stubbed with `usize::MAX` and substituted by the joint-prover
//! orchestrator when the layers are wired.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum input length supported by this AIR. ECADD = 128, ECMUL = 96;
/// we pick the maximum and zero-pad the ECMUL tail.
pub const MAX_INPUT_LENGTH: usize = 128;
/// Output length for both precompiles (Rx || Ry).
pub const OUTPUT_LENGTH: usize = 64;
/// Length of an alt_bn128 base-field element / scalar in big-endian bytes.
pub const FE_BYTES: usize = 32;

/// EIP precompile id for ECADD.
pub const PC_ECADD: u64 = 0x06;
/// EIP precompile id for ECMUL.
pub const PC_ECMUL: u64 = 0x07;

/// Gas cost for ECADD post-EIP-1108.
pub const GAS_ECADD: u64 = 150;
/// Gas cost for ECMUL post-EIP-1108.
pub const GAS_ECMUL: u64 = 6000;

/// ECADD input length (bytes).
pub const ECADD_INPUT_LENGTH: u64 = 128;
/// ECMUL input length (bytes).
pub const ECMUL_INPUT_LENGTH: u64 = 96;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_INPUT_OFFSET: usize = 0; // 0..128
pub const COL_OUTPUT_OFFSET: usize = COL_INPUT_OFFSET + MAX_INPUT_LENGTH; // 128..192
pub const COL_PX_OFFSET: usize = COL_OUTPUT_OFFSET + OUTPUT_LENGTH; // 192..224
pub const COL_PY_OFFSET: usize = COL_PX_OFFSET + FE_BYTES; // 224..256
pub const COL_QX_OFFSET: usize = COL_PY_OFFSET + FE_BYTES; // 256..288
pub const COL_QY_OFFSET: usize = COL_QX_OFFSET + FE_BYTES; // 288..320
pub const COL_K_OFFSET: usize = COL_QY_OFFSET + FE_BYTES; // 320..352
pub const COL_RX_OFFSET: usize = COL_K_OFFSET + FE_BYTES; // 352..384
pub const COL_RY_OFFSET: usize = COL_RX_OFFSET + FE_BYTES; // 384..416
pub const COL_IS_REAL: usize = COL_RY_OFFSET + FE_BYTES; // 416
pub const COL_SEL_ECADD: usize = COL_IS_REAL + 1; // 417
pub const COL_SEL_ECMUL: usize = COL_SEL_ECADD + 1; // 418
pub const COL_INPUT_LENGTH: usize = COL_SEL_ECMUL + 1; // 419
pub const COL_GAS_COST: usize = COL_INPUT_LENGTH + 1; // 420
pub const NUM_COLUMNS: usize = COL_GAS_COST + 1; // 421

/// Row-local constraint indices for documentation / labels:
///   0:   is_real binary
///   1:   sel_ecadd binary
///   2:   sel_ecmul binary
///   3:   sel_ecadd · sel_ecmul (mutually exclusive)
///   4:   sel_ecadd + sel_ecmul = is_real
///   5..37:   output[i]      = Rx[i]
///   37..69:  output[32+i]   = Ry[i]
///   69..101: is_real · (input[i]      - Px[i])
///   101..133:is_real · (input[32+i]   - Py[i])
///   133..165:sel_ecadd · (input[64+i] - Qx[i])
///   165..197:sel_ecadd · (input[96+i] - Qy[i])
///   197..229:sel_ecmul · (input[64+i] - k[i])
pub const NUM_ROW_CONSTRAINTS: usize =
    5 + 2 * FE_BYTES + 2 * FE_BYTES + 2 * FE_BYTES + FE_BYTES;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// One BN254 precompile invocation (ECADD or ECMUL).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bn254PrecompileWitness {
    /// Selector — true if this row encodes an ECADD invocation.
    pub is_ecadd: bool,
    /// Selector — true if this row encodes an ECMUL invocation.
    pub is_ecmul: bool,
    /// Input bytes (zero-padded to [`MAX_INPUT_LENGTH`]).
    pub input_bytes: [u8; MAX_INPUT_LENGTH],
    /// True input length (128 for ECADD, 96 for ECMUL).
    pub input_length: u64,
    /// `Px` big-endian.
    pub px: [u8; FE_BYTES],
    /// `Py` big-endian.
    pub py: [u8; FE_BYTES],
    /// `Qx` big-endian (ECADD only; zero for ECMUL).
    pub qx: [u8; FE_BYTES],
    /// `Qy` big-endian (ECADD only; zero for ECMUL).
    pub qy: [u8; FE_BYTES],
    /// `k` big-endian (ECMUL only; zero for ECADD).
    pub k: [u8; FE_BYTES],
    /// `Rx` big-endian.
    pub rx: [u8; FE_BYTES],
    /// `Ry` big-endian.
    pub ry: [u8; FE_BYTES],
    /// Full 64-byte precompile output `Rx || Ry`.
    pub output: [u8; OUTPUT_LENGTH],
    /// Gas cost (`GAS_ECADD` or `GAS_ECMUL`).
    pub gas_cost: u64,
}

impl Bn254PrecompileWitness {
    /// Construct an honest ECADD witness from two affine inputs and an
    /// already-computed result. Caller is responsible for the curve math
    /// — see [`bn254_host`] helpers below for known vectors. Use of a
    /// real BN254 crate is the production path; this constructor is the
    /// only one needed for AIR-shape soundness.
    pub fn from_ecadd_raw(
        p: ([u8; FE_BYTES], [u8; FE_BYTES]),
        q: ([u8; FE_BYTES], [u8; FE_BYTES]),
        r: ([u8; FE_BYTES], [u8; FE_BYTES]),
    ) -> Self {
        let mut input_bytes = [0u8; MAX_INPUT_LENGTH];
        input_bytes[0..32].copy_from_slice(&p.0);
        input_bytes[32..64].copy_from_slice(&p.1);
        input_bytes[64..96].copy_from_slice(&q.0);
        input_bytes[96..128].copy_from_slice(&q.1);

        let mut output = [0u8; OUTPUT_LENGTH];
        output[0..32].copy_from_slice(&r.0);
        output[32..64].copy_from_slice(&r.1);

        Self {
            is_ecadd: true,
            is_ecmul: false,
            input_bytes,
            input_length: ECADD_INPUT_LENGTH,
            px: p.0,
            py: p.1,
            qx: q.0,
            qy: q.1,
            k: [0u8; FE_BYTES],
            rx: r.0,
            ry: r.1,
            output,
            gas_cost: GAS_ECADD,
        }
    }

    /// Construct an honest ECMUL witness from `(P, k)` and an
    /// already-computed result. ECMUL input is 96 bytes; the trailing
    /// 32 bytes of `input_bytes` are zero.
    pub fn from_ecmul_raw(
        p: ([u8; FE_BYTES], [u8; FE_BYTES]),
        k: [u8; FE_BYTES],
        r: ([u8; FE_BYTES], [u8; FE_BYTES]),
    ) -> Self {
        let mut input_bytes = [0u8; MAX_INPUT_LENGTH];
        input_bytes[0..32].copy_from_slice(&p.0);
        input_bytes[32..64].copy_from_slice(&p.1);
        input_bytes[64..96].copy_from_slice(&k);

        let mut output = [0u8; OUTPUT_LENGTH];
        output[0..32].copy_from_slice(&r.0);
        output[32..64].copy_from_slice(&r.1);

        Self {
            is_ecadd: false,
            is_ecmul: true,
            input_bytes,
            input_length: ECMUL_INPUT_LENGTH,
            px: p.0,
            py: p.1,
            qx: [0u8; FE_BYTES],
            qy: [0u8; FE_BYTES],
            k,
            rx: r.0,
            ry: r.1,
            output,
            gas_cost: GAS_ECMUL,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Bn254PrecompileTraceWitness {
    pub rows: Vec<Bn254PrecompileWitness>,
}

impl Bn254PrecompileTraceWitness {
    pub fn from_rows(rows: Vec<Bn254PrecompileWitness>) -> Self {
        Self { rows }
    }
    pub fn push(&mut self, row: Bn254PrecompileWitness) {
        self.rows.push(row);
    }
}

// ─── Host-side helpers (known-vector / oracle path) ────────────────────

/// Standard BN254 test vectors used by the spec and by Ethereum
/// conformance tests. Convenience constants so the test suite does not
/// need a BN254 crate.
pub mod bn254_host {
    use super::FE_BYTES;
    use num_bigint::BigUint;

    /// Generator-like base point used in EIP-196 vectors: `G = (1, 2)`.
    pub fn point_one_two() -> ([u8; FE_BYTES], [u8; FE_BYTES]) {
        let mut x = [0u8; FE_BYTES];
        let mut y = [0u8; FE_BYTES];
        x[FE_BYTES - 1] = 1;
        y[FE_BYTES - 1] = 2;
        (x, y)
    }

    /// `(1, 2) + (1, 2) = 2·(1, 2)`. Standard EIP-196 vector:
    ///
    ///   Rx = 1368015179489954701390400359078579693043519447331113978918064868415326638035
    ///   Ry = 9918110051302171585080402603319702774565515993150576347155970296011453232463
    pub fn double_one_two() -> ([u8; FE_BYTES], [u8; FE_BYTES]) {
        let rx = BigUint::parse_bytes(
            b"1368015179489954701390400359078579693043519447331113978918064868415326638035",
            10,
        )
        .unwrap();
        let ry = BigUint::parse_bytes(
            b"9918110051302171585080402603319702774565515993150576347155970296011453232463",
            10,
        )
        .unwrap();
        (be32(&rx), be32(&ry))
    }

    /// Big-endian 32-byte encoding of a non-negative integer.
    pub fn be32(n: &BigUint) -> [u8; FE_BYTES] {
        let bytes = n.to_bytes_be();
        let mut out = [0u8; FE_BYTES];
        let off = FE_BYTES - bytes.len();
        out[off..].copy_from_slice(&bytes);
        out
    }

    /// Scalar `2` as 32-byte big-endian.
    pub fn scalar_two() -> [u8; FE_BYTES] {
        let mut s = [0u8; FE_BYTES];
        s[FE_BYTES - 1] = 2;
        s
    }
}

/// Honest ECADD witness for the EIP-196 vector `(1, 2) + (1, 2)`.
pub fn ecadd_one_two_doubled_witness() -> Bn254PrecompileWitness {
    let p = bn254_host::point_one_two();
    let r = bn254_host::double_one_two();
    Bn254PrecompileWitness::from_ecadd_raw(p, p, r)
}

/// Honest ECMUL witness for the EIP-196 vector `(1, 2) · 2`.
pub fn ecmul_one_two_times_two_witness() -> Bn254PrecompileWitness {
    let p = bn254_host::point_one_two();
    let r = bn254_host::double_one_two();
    Bn254PrecompileWitness::from_ecmul_raw(p, bn254_host::scalar_two(), r)
}

/// Identity-input row: `(0, 0) + (0, 0) = (0, 0)`. Useful as a
/// well-formed row that does not require any curve crate. The shape
/// constraints still apply.
pub fn ecadd_identity_witness() -> Bn254PrecompileWitness {
    let zero = ([0u8; FE_BYTES], [0u8; FE_BYTES]);
    Bn254PrecompileWitness::from_ecadd_raw(zero, zero, zero)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Bn254PrecompileTraceWitness,
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
        for k in 0..FE_BYTES {
            columns[COL_PX_OFFSET + k][r] = Scalar::from_u64(row.px[k] as u64, curve);
            columns[COL_PY_OFFSET + k][r] = Scalar::from_u64(row.py[k] as u64, curve);
            columns[COL_QX_OFFSET + k][r] = Scalar::from_u64(row.qx[k] as u64, curve);
            columns[COL_QY_OFFSET + k][r] = Scalar::from_u64(row.qy[k] as u64, curve);
            columns[COL_K_OFFSET + k][r] = Scalar::from_u64(row.k[k] as u64, curve);
            columns[COL_RX_OFFSET + k][r] = Scalar::from_u64(row.rx[k] as u64, curve);
            columns[COL_RY_OFFSET + k][r] = Scalar::from_u64(row.ry[k] as u64, curve);
        }

        columns[COL_IS_REAL][r] = one.clone();
        columns[COL_SEL_ECADD][r] = if row.is_ecadd { one.clone() } else { zero.clone() };
        columns[COL_SEL_ECMUL][r] = if row.is_ecmul { one.clone() } else { zero.clone() };
        columns[COL_INPUT_LENGTH][r] = Scalar::from_u64(row.input_length, curve);
        columns[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
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

pub struct Bn254PrecompileConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Bn254PrecompileConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Bn254PrecompileConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        labels.push("sel_ecadd_binary".into());
        labels.push("sel_ecmul_binary".into());
        labels.push("sel_ecadd_sel_ecmul_mutually_exclusive".into());
        labels.push("selectors_sum_is_real".into());
        for k in 0..FE_BYTES {
            labels.push(format!("output_eq_rx_byte_{}", k));
        }
        for k in 0..FE_BYTES {
            labels.push(format!("output_eq_ry_byte_{}", k));
        }
        for k in 0..FE_BYTES {
            labels.push(format!("input_eq_px_byte_{}", k));
        }
        for k in 0..FE_BYTES {
            labels.push(format!("input_eq_py_byte_{}", k));
        }
        for k in 0..FE_BYTES {
            labels.push(format!("ecadd_input_eq_qx_byte_{}", k));
        }
        for k in 0..FE_BYTES {
            labels.push(format!("ecadd_input_eq_qy_byte_{}", k));
        }
        for k in 0..FE_BYTES {
            labels.push(format!("ecmul_input_eq_k_byte_{}", k));
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
        // 1: sel_ecadd binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_SEL_ECADD][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 2: sel_ecmul binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_SEL_ECMUL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 3: sel_ecadd * sel_ecmul = 0 (mutually exclusive).
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_SEL_ECADD][r].mul(&columns[COL_SEL_ECMUL][r]);
            }
            out.push(c);
        }
        // 4: sel_ecadd + sel_ecmul - is_real = 0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let sum = columns[COL_SEL_ECADD][r].add(&columns[COL_SEL_ECMUL][r]);
                c[r] = sum.sub(&columns[COL_IS_REAL][r]);
            }
            out.push(c);
        }

        // 5..37: output[i] = Rx[i].
        for i in 0..FE_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_OUTPUT_OFFSET + i][r]
                    .sub(&columns[COL_RX_OFFSET + i][r]);
            }
            out.push(c);
        }
        // 37..69: output[32+i] = Ry[i].
        for i in 0..FE_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_OUTPUT_OFFSET + FE_BYTES + i][r]
                    .sub(&columns[COL_RY_OFFSET + i][r]);
            }
            out.push(c);
        }
        // 69..101: is_real * (input[i] - Px[i]).
        for i in 0..FE_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let diff = columns[COL_INPUT_OFFSET + i][r]
                    .sub(&columns[COL_PX_OFFSET + i][r]);
                c[r] = columns[COL_IS_REAL][r].mul(&diff);
            }
            out.push(c);
        }
        // 101..133: is_real * (input[32+i] - Py[i]).
        for i in 0..FE_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let diff = columns[COL_INPUT_OFFSET + FE_BYTES + i][r]
                    .sub(&columns[COL_PY_OFFSET + i][r]);
                c[r] = columns[COL_IS_REAL][r].mul(&diff);
            }
            out.push(c);
        }
        // 133..165: sel_ecadd * (input[64+i] - Qx[i]).
        for i in 0..FE_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let diff = columns[COL_INPUT_OFFSET + 2 * FE_BYTES + i][r]
                    .sub(&columns[COL_QX_OFFSET + i][r]);
                c[r] = columns[COL_SEL_ECADD][r].mul(&diff);
            }
            out.push(c);
        }
        // 165..197: sel_ecadd * (input[96+i] - Qy[i]).
        for i in 0..FE_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let diff = columns[COL_INPUT_OFFSET + 3 * FE_BYTES + i][r]
                    .sub(&columns[COL_QY_OFFSET + i][r]);
                c[r] = columns[COL_SEL_ECADD][r].mul(&diff);
            }
            out.push(c);
        }
        // 197..229: sel_ecmul * (input[64+i] - k[i]).
        for i in 0..FE_BYTES {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let diff = columns[COL_INPUT_OFFSET + 2 * FE_BYTES + i][r]
                    .sub(&columns[COL_K_OFFSET + i][r]);
                c[r] = columns[COL_SEL_ECMUL][r].mul(&diff);
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

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: sel_ecadd binary.
        {
            let v = &col_evals[COL_SEL_ECADD];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: sel_ecmul binary.
        {
            let v = &col_evals[COL_SEL_ECMUL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3: sel_ecadd * sel_ecmul.
        {
            let body = col_evals[COL_SEL_ECADD].mul(&col_evals[COL_SEL_ECMUL]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4: sel_ecadd + sel_ecmul - is_real.
        {
            let body = col_evals[COL_SEL_ECADD]
                .add(&col_evals[COL_SEL_ECMUL])
                .sub(&col_evals[COL_IS_REAL]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // output[i] = Rx[i].
        for i in 0..FE_BYTES {
            let body = col_evals[COL_OUTPUT_OFFSET + i]
                .sub(&col_evals[COL_RX_OFFSET + i]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // output[32+i] = Ry[i].
        for i in 0..FE_BYTES {
            let body = col_evals[COL_OUTPUT_OFFSET + FE_BYTES + i]
                .sub(&col_evals[COL_RY_OFFSET + i]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // is_real * (input[i] - Px[i]).
        for i in 0..FE_BYTES {
            let diff = col_evals[COL_INPUT_OFFSET + i]
                .sub(&col_evals[COL_PX_OFFSET + i]);
            let body = col_evals[COL_IS_REAL].mul(&diff);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // is_real * (input[32+i] - Py[i]).
        for i in 0..FE_BYTES {
            let diff = col_evals[COL_INPUT_OFFSET + FE_BYTES + i]
                .sub(&col_evals[COL_PY_OFFSET + i]);
            let body = col_evals[COL_IS_REAL].mul(&diff);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // sel_ecadd * (input[64+i] - Qx[i]).
        for i in 0..FE_BYTES {
            let diff = col_evals[COL_INPUT_OFFSET + 2 * FE_BYTES + i]
                .sub(&col_evals[COL_QX_OFFSET + i]);
            let body = col_evals[COL_SEL_ECADD].mul(&diff);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // sel_ecadd * (input[96+i] - Qy[i]).
        for i in 0..FE_BYTES {
            let diff = col_evals[COL_INPUT_OFFSET + 3 * FE_BYTES + i]
                .sub(&col_evals[COL_QY_OFFSET + i]);
            let body = col_evals[COL_SEL_ECADD].mul(&diff);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // sel_ecmul * (input[64+i] - k[i]).
        for i in 0..FE_BYTES {
            let diff = col_evals[COL_INPUT_OFFSET + 2 * FE_BYTES + i]
                .sub(&col_evals[COL_K_OFFSET + i]);
            let body = col_evals[COL_SEL_ECMUL].mul(&diff);
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

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // helper: accumulate `coeff * alpha_pow` and bump alpha_pow.
        macro_rules! push {
            ($body:expr) => {{
                acc = poly_add(&acc, &poly_scalar_mul(&$body, &alpha_pow), curve);
                alpha_pow = alpha_pow.mul(alpha);
            }};
        }

        // 0: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            push!(body);
        }
        // 1: sel_ecadd binary.
        {
            let v = &col_coeffs[COL_SEL_ECADD];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            push!(body);
        }
        // 2: sel_ecmul binary.
        {
            let v = &col_coeffs[COL_SEL_ECMUL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            push!(body);
        }
        // 3: sel_ecadd * sel_ecmul.
        {
            let body = poly_mul(
                &col_coeffs[COL_SEL_ECADD],
                &col_coeffs[COL_SEL_ECMUL],
                curve,
            );
            push!(body);
        }
        // 4: sel_ecadd + sel_ecmul - is_real.
        {
            let mut body = poly_add(
                &col_coeffs[COL_SEL_ECADD],
                &col_coeffs[COL_SEL_ECMUL],
                curve,
            );
            body = poly_sub(&body, &col_coeffs[COL_IS_REAL], curve);
            push!(body);
        }
        // output[i] = Rx[i].
        for i in 0..FE_BYTES {
            let body = poly_sub(
                &col_coeffs[COL_OUTPUT_OFFSET + i],
                &col_coeffs[COL_RX_OFFSET + i],
                curve,
            );
            push!(body);
        }
        // output[32+i] = Ry[i].
        for i in 0..FE_BYTES {
            let body = poly_sub(
                &col_coeffs[COL_OUTPUT_OFFSET + FE_BYTES + i],
                &col_coeffs[COL_RY_OFFSET + i],
                curve,
            );
            push!(body);
        }
        // is_real * (input[i] - Px[i]).
        for i in 0..FE_BYTES {
            let diff = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + i],
                &col_coeffs[COL_PX_OFFSET + i],
                curve,
            );
            let body = poly_mul(&col_coeffs[COL_IS_REAL], &diff, curve);
            push!(body);
        }
        // is_real * (input[32+i] - Py[i]).
        for i in 0..FE_BYTES {
            let diff = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + FE_BYTES + i],
                &col_coeffs[COL_PY_OFFSET + i],
                curve,
            );
            let body = poly_mul(&col_coeffs[COL_IS_REAL], &diff, curve);
            push!(body);
        }
        // sel_ecadd * (input[64+i] - Qx[i]).
        for i in 0..FE_BYTES {
            let diff = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + 2 * FE_BYTES + i],
                &col_coeffs[COL_QX_OFFSET + i],
                curve,
            );
            let body = poly_mul(&col_coeffs[COL_SEL_ECADD], &diff, curve);
            push!(body);
        }
        // sel_ecadd * (input[96+i] - Qy[i]).
        for i in 0..FE_BYTES {
            let diff = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + 3 * FE_BYTES + i],
                &col_coeffs[COL_QY_OFFSET + i],
                curve,
            );
            let body = poly_mul(&col_coeffs[COL_SEL_ECADD], &diff, curve);
            push!(body);
        }
        // sel_ecmul * (input[64+i] - k[i]).
        for i in 0..FE_BYTES {
            let diff = poly_sub(
                &col_coeffs[COL_INPUT_OFFSET + 2 * FE_BYTES + i],
                &col_coeffs[COL_K_OFFSET + i],
                curve,
            );
            let body = poly_mul(&col_coeffs[COL_SEL_ECMUL], &diff, curve);
            push!(body);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_SEL_ECADD, COL_SEL_ECMUL]
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
        // 8-bit range check every byte column.
        let byte_offsets_and_widths: &[(usize, usize, &str)] = &[
            (COL_INPUT_OFFSET, MAX_INPUT_LENGTH, "input"),
            (COL_OUTPUT_OFFSET, OUTPUT_LENGTH, "output"),
            (COL_PX_OFFSET, FE_BYTES, "px"),
            (COL_PY_OFFSET, FE_BYTES, "py"),
            (COL_QX_OFFSET, FE_BYTES, "qx"),
            (COL_QY_OFFSET, FE_BYTES, "qy"),
            (COL_K_OFFSET, FE_BYTES, "k"),
            (COL_RX_OFFSET, FE_BYTES, "rx"),
            (COL_RY_OFFSET, FE_BYTES, "ry"),
        ];
        for &(off, width, name) in byte_offsets_and_widths {
            for k in 0..width {
                declarations.push((
                    LookupDeclaration {
                        label: format!("bn254_precompile_{}_byte_{}_8bit", name, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Placeholder sentinel for `precompile_air` column indices. Same
/// convention as `ripemd160_precompile_air`.
pub const PRECOMPILE_DISPATCH_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds
/// `(sel_ecadd, sel_ecmul, input_length, output_length=64, gas_cost)` on
/// this AIR's `IS_REAL` rows to the matching `precompile_air` dispatch
/// row for callee `0x06` / `0x07`.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B-side
/// columns are filled with [`PRECOMPILE_DISPATCH_PLACEHOLDER`]. The
/// A-side `output_length` placeholder records the **shape** of the
/// binding — the joint-prover orchestrator substitutes the real
/// indices.
///
/// MUST NOT be passed to `joint_prove` until the placeholders are
/// resolved.
pub fn make_bn254_to_precompile_dispatch_descriptor(
    bn254_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = vec![
        COL_SEL_ECADD,
        COL_SEL_ECMUL,
        COL_INPUT_LENGTH,
        PRECOMPILE_DISPATCH_PLACEHOLDER, // synthetic output_length = 64
        COL_GAS_COST,
    ];
    let b_columns: Vec<usize> = vec![
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_SEL_ECADD
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_SEL_ECMUL
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_INPUT_LENGTH
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_OUTPUT_LENGTH
        PRECOMPILE_DISPATCH_PLACEHOLDER, // precompile_air::COL_GAS_COST
    ];
    CrossAirLogUpDescriptor {
        label: "bn254_to_precompile_dispatch_v1_stub".into(),
        a_layer_index: bn254_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns,
        b_selector_column: None,
    }
}

/// Placeholder sentinel for `precompile_io_air` column indices.
pub const PRECOMPILE_IO_PLACEHOLDER: usize = usize::MAX;

/// Cross-AIR LogUp descriptor (stub): binds
/// `(input_bytes[0..128], output[0..64])` to the corresponding
/// `precompile_io_air` row that records the EVM-side CALL memory I/O.
///
/// **Stub**: zkp does not depend on the EVM crate, so the B side is
/// filled with [`PRECOMPILE_IO_PLACEHOLDER`]. The orchestrator
/// substitutes the real B-side indices once `precompile_io_air` gains a
/// dedicated BN254 input/output channel.
pub fn make_bn254_to_precompile_io_descriptor(
    bn254_layer_index: usize,
    precompile_io_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_LENGTH + OUTPUT_LENGTH);
    for k in 0..MAX_INPUT_LENGTH {
        a_columns.push(COL_INPUT_OFFSET + k);
    }
    for k in 0..OUTPUT_LENGTH {
        a_columns.push(COL_OUTPUT_OFFSET + k);
    }
    let b_columns: Vec<usize> =
        vec![PRECOMPILE_IO_PLACEHOLDER; MAX_INPUT_LENGTH + OUTPUT_LENGTH];
    CrossAirLogUpDescriptor {
        label: "bn254_to_precompile_io_v1_stub".into(),
        a_layer_index: bn254_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
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
    fn bn254_precompile_ecadd_one_two_doubled_vector() {
        let w = ecadd_one_two_doubled_witness();
        assert!(w.is_ecadd);
        assert!(!w.is_ecmul);
        assert_eq!(w.input_length, ECADD_INPUT_LENGTH);
        assert_eq!(w.gas_cost, GAS_ECADD);
        // Spot check the canonical EIP-196 result for the x coord: the
        // big-endian byte representation of the decimal
        // 1368015179489954701390400359078579693043519447331113978918064868415326638035
        // is well-known.
        assert_ne!(w.rx, [0u8; FE_BYTES]);
        assert_ne!(w.ry, [0u8; FE_BYTES]);
        // Px = (1, 2).
        assert_eq!(w.px[31], 1);
        assert_eq!(w.py[31], 2);
        // Qx = Px (doubling).
        assert_eq!(w.qx, w.px);
        assert_eq!(w.qy, w.py);

        // Constraints zero on honest witness.
        let trace = build_trace_polynomials(
            &Bn254PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Bn254PrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn bn254_precompile_ecmul_one_two_times_two_vector() {
        let w = ecmul_one_two_times_two_witness();
        assert!(w.is_ecmul);
        assert!(!w.is_ecadd);
        assert_eq!(w.input_length, ECMUL_INPUT_LENGTH);
        assert_eq!(w.gas_cost, GAS_ECMUL);
        // Px = (1, 2).
        assert_eq!(w.px[31], 1);
        assert_eq!(w.py[31], 2);
        // Scalar = 2.
        assert_eq!(w.k[31], 2);
        // ECMUL(P, 2) = 2P which equals the ECADD double-result.
        let doubled = bn254_host::double_one_two();
        assert_eq!(w.rx, doubled.0);
        assert_eq!(w.ry, doubled.1);
        // Trailing 32 bytes of input_bytes are zero-padding.
        for k in 96..128 {
            assert_eq!(w.input_bytes[k], 0);
        }

        let trace = build_trace_polynomials(
            &Bn254PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Bn254PrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn bn254_precompile_identity_input_row() {
        // (0, 0) + (0, 0) = (0, 0). All byte columns zero, but
        // is_real / sel_ecadd are set: shape constraints still hold.
        let w = ecadd_identity_witness();
        let trace = build_trace_polynomials(
            &Bn254PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Bn254PrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn bn254_precompile_tampered_output_detected() {
        // Tamper output[5]: ECADD path should fire the
        // `output[i] = Rx[i]` constraint at i=5.
        let w = ecadd_one_two_doubled_witness();
        let trace = build_trace_polynomials(
            &Bn254PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Pick a byte position where Rx[5] is unlikely to equal 0xab.
        cols[COL_OUTPUT_OFFSET + 5][0] =
            Scalar::from_u64(0xab, CurveType::Bls48581);

        let cs = Bn254PrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        // Constraint index for output[i] = Rx[i] at i=5: 5 + 5 = 10.
        let idx = 5 + 5;
        assert!(
            !bodies[idx][0].is_zero(),
            "expected output==Rx constraint at index {} to fire",
            idx,
        );

        // Tamper a Py byte: `is_real * (input[32+i] - Py[i])` at i=7
        // (constraint index 5 + 64 + 32 + 7 = 108).
        let mut cols2: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols2[COL_PY_OFFSET + 7][0] =
            Scalar::from_u64(0xcd, CurveType::Bls48581);
        let col_refs2: Vec<&Vec<Scalar>> = cols2.iter().collect();
        let bodies2 = cs.evaluate_on_domain(&col_refs2, trace.num_rows);
        let idx2 = 5 + 2 * FE_BYTES + FE_BYTES + 7;
        assert!(
            !bodies2[idx2][0].is_zero(),
            "expected input==Py constraint at index {} to fire",
            idx2,
        );
    }

    #[test]
    fn bn254_precompile_selector_mutual_exclusion_violation_detected() {
        // Force both selectors to 1 on a single row: mutual exclusion
        // constraint (index 3) must fire, and the sum-to-is_real
        // constraint (index 4) must also fire (1 + 1 - 1 = 1).
        let w = ecadd_one_two_doubled_witness();
        let trace = build_trace_polynomials(
            &Bn254PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_SEL_ECMUL][0] = Scalar::one(CurveType::Bls48581);
        let cs = Bn254PrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "expected sel_ecadd*sel_ecmul=0 constraint to fire",
        );
        assert!(
            !bodies[4][0].is_zero(),
            "expected sel_ecadd+sel_ecmul=is_real constraint to fire",
        );
    }

    #[test]
    fn bn254_precompile_descriptors_well_formed() {
        let d1 = make_bn254_to_precompile_dispatch_descriptor(0, 1);
        assert_eq!(d1.label, "bn254_to_precompile_dispatch_v1_stub");
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert_eq!(d1.a_columns.len(), 5);
        assert_eq!(d1.b_columns.len(), 5);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.a_columns[0], COL_SEL_ECADD);
        assert_eq!(d1.a_columns[1], COL_SEL_ECMUL);
        assert_eq!(d1.a_columns[2], COL_INPUT_LENGTH);
        assert_eq!(d1.a_columns[3], PRECOMPILE_DISPATCH_PLACEHOLDER);
        assert_eq!(d1.a_columns[4], COL_GAS_COST);
        for b in &d1.b_columns {
            assert_eq!(*b, PRECOMPILE_DISPATCH_PLACEHOLDER);
        }

        let d2 = make_bn254_to_precompile_io_descriptor(0, 2);
        assert_eq!(d2.label, "bn254_to_precompile_io_v1_stub");
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
        for b in &d2.b_columns {
            assert_eq!(*b, PRECOMPILE_IO_PLACEHOLDER);
        }
    }

    #[test]
    fn bn254_precompile_column_layout_pinned() {
        assert_eq!(COL_INPUT_OFFSET, 0);
        assert_eq!(COL_OUTPUT_OFFSET, 128);
        assert_eq!(COL_PX_OFFSET, 192);
        assert_eq!(COL_PY_OFFSET, 224);
        assert_eq!(COL_QX_OFFSET, 256);
        assert_eq!(COL_QY_OFFSET, 288);
        assert_eq!(COL_K_OFFSET, 320);
        assert_eq!(COL_RX_OFFSET, 352);
        assert_eq!(COL_RY_OFFSET, 384);
        assert_eq!(COL_IS_REAL, 416);
        assert_eq!(COL_SEL_ECADD, 417);
        assert_eq!(COL_SEL_ECMUL, 418);
        assert_eq!(COL_INPUT_LENGTH, 419);
        assert_eq!(COL_GAS_COST, 420);
        assert_eq!(NUM_COLUMNS, 421);

        // 5 selector/shape + 2*32 output + 2*32 Px/Py + 2*32 Qx/Qy + 32 k
        // = 5 + 64 + 64 + 64 + 32 = 229.
        assert_eq!(NUM_ROW_CONSTRAINTS, 229);
        assert_eq!(NUM_SHIFTED, 0);

        assert_eq!(MAX_INPUT_LENGTH, 128);
        assert_eq!(OUTPUT_LENGTH, 64);
        assert_eq!(FE_BYTES, 32);
        assert_eq!(PC_ECADD, 0x06);
        assert_eq!(PC_ECMUL, 0x07);
        assert_eq!(GAS_ECADD, 150);
        assert_eq!(GAS_ECMUL, 6000);
        assert_eq!(ECADD_INPUT_LENGTH, 128);
        assert_eq!(ECMUL_INPUT_LENGTH, 96);

        let cs = Bn254PrecompileConstraintSystem::new(1);
        assert_eq!(cs.num_constraints(), NUM_ROW_CONSTRAINTS);
        assert_eq!(cs.constraint_labels().len(), NUM_ROW_CONSTRAINTS);
    }

    #[test]
    fn bn254_precompile_multi_row_ecadd_ecmul_mixed_passes() {
        let rows = vec![
            ecadd_one_two_doubled_witness(),
            ecmul_one_two_times_two_witness(),
            ecadd_identity_witness(),
        ];
        let trace = build_trace_polynomials(
            &Bn254PrecompileTraceWitness::from_rows(rows),
            CurveType::Bls48581,
        );
        let cs = Bn254PrecompileConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, trace.num_rows));
    }

    #[test]
    fn bn254_precompile_evaluate_at_point_matches_domain() {
        // Cross-check: `evaluate_at_point` on a row's column values
        // should match the running RLC of `evaluate_on_domain` for the
        // same row when reduced with the same alpha.
        let w = ecadd_one_two_doubled_witness();
        let trace = build_trace_polynomials(
            &Bn254PrecompileTraceWitness::from_rows(vec![w]),
            CurveType::Bls48581,
        );
        let cs = Bn254PrecompileConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Honest witness ⇒ all bodies zero ⇒ alpha-RLC of bodies at row
        // 0 is zero.
        let alpha = Scalar::from_u64(0x1234_5678, CurveType::Bls48581);
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
        assert!(got.is_zero(), "evaluate_at_point should be zero on honest witness");
    }
}
