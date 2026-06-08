//! Keccak-256 invocation extraction AIR.
//!
//! Per-invocation byte-level view of a `keccak256(input) → output`
//! invocation. Each row corresponds to one keccak invocation and
//! exposes:
//!
//!   - `INPUT_BYTE[0..MAX_INPUT_LEN]` — input bytes, zero-padded to the
//!     fixed maximum length.
//!   - `INPUT_LEN` — actual length of the input (so the keccak
//!     padding is well-defined; tuple matches must include this).
//!   - `OUTPUT_BYTE[0..32]` — the 32-byte keccak-256 digest.
//!   - `IS_REAL` — selector, `1` on real invocation rows, `0` on padding.
//!
//! Mirrors [`crate::sha256_extract`]: a focused byte-level view that
//! cross-AIR LogUp linkages match against. The heavy bit-level
//! `keccak_constraints` AIR stays unchanged.
//!
//! # Why a fixed max input length?
//!
//! Cross-AIR LogUp tuple matching requires fixed-shape rows. We bound
//! the input at `MAX_INPUT_LEN = 256` bytes — covers MPT leaf and
//! extension nodes (typically ~64 bytes max). MPT branch nodes
//! (potentially up to ~564 bytes) need a separate larger-bound AIR
//! variant or a different binding strategy. See
//! `cross_air_logup_dependent_tasks.md`.
//!
//! # Soundness scope
//!
//! This AIR proves: "there exists a sequence of `(input, len, output)`
//! invocation tuples; `IS_REAL` is binary."
//!
//! It does NOT prove `output = keccak256(input[0..len])`. Closing
//! that gap requires a cross-AIR LogUp linkage to the bit-level
//! `keccak_constraints` AIR — same shape as the
//! Sha256Extract↔SHA-256 future binding.
//!
//! # Constraint layout
//!
//! 5 row-local constraints (alpha-power weighted):
//!
//!   0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `address_limb_0_binding` — `ADDRESS_LIMB[0] − Σ_{i=0..8}
//!      2^(8 i) · OUTPUT_BYTE[12 + i] = 0`
//!   2. `address_limb_1_binding` — `ADDRESS_LIMB[1] − Σ_{i=0..8}
//!      2^(8 i) · OUTPUT_BYTE[20 + i] = 0`
//!   3. `address_limb_2_binding` — `ADDRESS_LIMB[2] − Σ_{i=0..4}
//!      2^(8 i) · OUTPUT_BYTE[28 + i] = 0` (limb 2 carries only 4
//!      address bytes — its high 4 bytes are pinned to zero by the
//!      truncated sum)
//!   4. `address_limb_3_zero` — `ADDRESS_LIMB[3] = 0` (a 20-byte
//!      address never spills into limb 3)
//!
//! The aggregated `ADDRESS_LIMB` columns mirror EVM's
//! `address_to_limbs` little-endian decomposition (see
//! `crates/evm/src/inspector.rs`), so the cross-AIR LogUp linkage
//! in `make_evm_create_address_keccak_extract_linkage_descriptor`
//! (in `crates/evm/src/cross_air_linkage.rs`) matches the existing
//! `COL_CREATE_ADDRESS_HINT_L0..L3` columns directly without a
//! byte-decomposition gadget on the EVM side.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column indices ────────────────────────────────────────────────────

/// Maximum input length supported by this extract AIR. Inputs shorter
/// than this are zero-padded; `INPUT_LEN` records the actual length.
pub const MAX_INPUT_LEN: usize = 256;

/// Output length is fixed for keccak-256.
pub const OUTPUT_LEN: usize = 32;

pub const COL_INPUT_BYTE_OFFSET: usize = 0;
pub const COL_INPUT_LEN: usize = COL_INPUT_BYTE_OFFSET + MAX_INPUT_LEN;
pub const COL_OUTPUT_BYTE_OFFSET: usize = COL_INPUT_LEN + 1;
pub const COL_ADDRESS_LIMB_OFFSET: usize = COL_OUTPUT_BYTE_OFFSET + OUTPUT_LEN;
/// 4 u64 aggregator columns packing the FULL 32-byte keccak output as
/// `U256::from_be_bytes(output).as_limbs()`. Used by the EVM SHA3 ↔
/// KeccakExtract cross-AIR LogUp (Phase A1a) which matches the EVM
/// trace's `output0` U256 limbs directly. Distinct from
/// `ADDRESS_LIMB_OFFSET` which uses LE-byte packing on `output[12..32]`
/// (the EVM `address_to_limbs` convention used by CREATE/CREATE2).
pub const COL_KECCAK_OUTPUT_LIMB_OFFSET: usize = COL_ADDRESS_LIMB_OFFSET + ADDRESS_LIMB_LEN;
pub const COL_IS_REAL: usize = COL_KECCAK_OUTPUT_LIMB_OFFSET + KECCAK_OUTPUT_LIMB_LEN;

/// Task #318 / #309 / #190 mirror column: per-invocation byte anchor for
/// cross-AIR LogUp descriptors that want to bind an arbitrary host-derived
/// byte to this AIR's row without requiring B-side overrides on
/// `OUTPUT_BYTE[0]`. Populated by the witness builder; defaults to
/// `output[0]` (back-compat). No row-local constraint binds this column —
/// soundness flows from the cross-AIR LogUp closure.
pub const COL_MIRROR_BYTE0: usize = COL_IS_REAL + 1;

/// Number of address-limb aggregator columns (4 u64 limbs covering the
/// 20-byte Ethereum address packed at `OUTPUT_BYTE[12..32]`).
pub const ADDRESS_LIMB_LEN: usize = 4;

/// Number of full-output u64 aggregator columns (4 u64 limbs covering
/// the entire 32-byte keccak output as `U256::from_be_bytes` limbs).
pub const KECCAK_OUTPUT_LIMB_LEN: usize = 4;

pub const NUM_COLUMNS: usize = COL_MIRROR_BYTE0 + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 9;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakExtractRow {
    /// Padded to [`MAX_INPUT_LEN`] with trailing zeros.
    pub input: [u8; MAX_INPUT_LEN],
    /// Actual input length (the keccak hash is over `input[0..len]`).
    pub input_len: usize,
    /// Keccak-256 digest of `input[0..input_len]`.
    pub output: [u8; OUTPUT_LEN],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakExtractWitness {
    pub invocations: Vec<KeccakExtractRow>,
    /// Task #318 / #309 mirror data: per-invocation byte value populated
    /// into [`COL_MIRROR_BYTE0`]. Empty `Vec` means "default to
    /// `invocations[i].output[0]`" (back-compat). Otherwise must match
    /// `invocations.len()` and is written verbatim per-row.
    #[doc(hidden)]
    pub mirror_byte0: Vec<u8>,
}

impl KeccakExtractWitness {
    /// Build the witness from `(input_bytes, ...)` tuples, computing
    /// `output = keccak256(input)` per row. Inputs longer than
    /// `MAX_INPUT_LEN` are rejected (returns `Err` on the first
    /// over-long input).
    pub fn from_inputs(inputs: &[Vec<u8>]) -> Result<Self, &'static str> {
        let mut invocations = Vec::with_capacity(inputs.len());
        for inp in inputs {
            if inp.len() > MAX_INPUT_LEN {
                return Err(
                    "keccak_extract: input exceeds MAX_INPUT_LEN \
                     (extend the AIR or use a separate larger-bound variant)",
                );
            }
            let mut padded = [0u8; MAX_INPUT_LEN];
            padded[..inp.len()].copy_from_slice(inp);
            let output = crate::keccak::keccak256(inp);
            invocations.push(KeccakExtractRow {
                input: padded,
                input_len: inp.len(),
                output,
            });
        }
        Ok(Self {
            invocations,
            mirror_byte0: Vec::new(),
        })
    }

    /// Builder: override `COL_MIRROR_BYTE0` values per-invocation. Used
    /// by integration tests that wire a cross-AIR LogUp descriptor
    /// against this AIR's mirror column without touching `OUTPUT_BYTE`.
    pub fn with_mirror_byte0(mut self, mirror: Vec<u8>) -> Self {
        assert_eq!(
            mirror.len(),
            self.invocations.len(),
            "mirror_byte0 length must equal invocations.len()",
        );
        self.mirror_byte0 = mirror;
        self
    }
}

/// Pack the last 20 bytes of a 32-byte keccak digest into 4 u64 limbs
/// using the same little-endian decomposition as EVM's
/// `address_to_limbs` (`crates/evm/src/inspector.rs`):
///
///   limb[0] = u64::from_le_bytes(output[12..20])
///   limb[1] = u64::from_le_bytes(output[20..28])
///   limb[2] = u64::from_le_bytes(output[28..32] || [0; 4])
///   limb[3] = 0
///
/// The cross-AIR LogUp linkage between EVM's `create_address_hint_l*`
/// columns and KeccakExtract's `address_limb_l*` aggregator columns
/// matches these 4-limb tuples directly, with the
/// `address_limb_binding` row-local constraint pinning the
/// aggregator to the keccak output bytes.
pub fn address_limbs_from_output(output: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    let mut tmp = [0u8; 8];
    tmp.copy_from_slice(&output[12..20]);
    limbs[0] = u64::from_le_bytes(tmp);
    tmp.copy_from_slice(&output[20..28]);
    limbs[1] = u64::from_le_bytes(tmp);
    let mut tmp4 = [0u8; 8];
    tmp4[..4].copy_from_slice(&output[28..32]);
    limbs[2] = u64::from_le_bytes(tmp4);
    limbs
}

/// Pack the full 32-byte keccak digest into 4 u64 limbs matching
/// `U256::from_be_bytes(output).as_limbs()`. This is the EVM stack
/// convention for SHA3: the keccak result is pushed as a U256 in
/// little-endian limbs, but the underlying bytes are big-endian
/// (high byte first) per `U256::from_be_bytes`. So:
///
///   limb[0] (low 64 bits)  = u64::from_be_bytes(output[24..32])
///   limb[1]                = u64::from_be_bytes(output[16..24])
///   limb[2]                = u64::from_be_bytes(output[8..16])
///   limb[3] (high 64 bits) = u64::from_be_bytes(output[0..8])
///
/// Used by the EVM main-trace `output0[k]` columns on SHA3 rows.
/// The cross-AIR LogUp `make_evm_keccak_keccak_extract_linkage_descriptor`
/// matches these limbs directly.
pub fn keccak_output_limbs_from_output(output: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for k in 0..KECCAK_OUTPUT_LIMB_LEN {
        let start = (3 - k) * 8;
        let mut tmp = [0u8; 8];
        tmp.copy_from_slice(&output[start..start + 8]);
        limbs[k] = u64::from_be_bytes(tmp);
    }
    limbs
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct KeccakExtractConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl KeccakExtractConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            omega: None,
            domain_size: None,
        }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

pub fn build_trace_polynomials(
    witness: &KeccakExtractWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));

    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (i, inv) in witness.invocations.iter().enumerate() {
        for (b, &v) in inv.input.iter().enumerate() {
            columns[COL_INPUT_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_INPUT_LEN][i] = Scalar::from_u64(inv.input_len as u64, curve);
        for (b, &v) in inv.output.iter().enumerate() {
            columns[COL_OUTPUT_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        // Address-limb aggregators: pack output[12..32] into 4 u64 limbs
        // matching EVM's `address_to_limbs` convention. Limb k = bytes
        // [12 + 8k .. 12 + 8k + min(8, 20 - 8k)] little-endian; limb 3
        // is identically zero (a 20-byte address is fully covered by
        // limbs 0..2 with limb 2 carrying only its low 4 bytes).
        let limbs = address_limbs_from_output(&inv.output);
        for (k, limb) in limbs.iter().enumerate() {
            columns[COL_ADDRESS_LIMB_OFFSET + k][i] = Scalar::from_u64(*limb, curve);
        }
        // Full-output U256 limbs (BE-packed). Distinct convention from
        // ADDRESS_LIMB; required for the EVM SHA3 cross-AIR LogUp.
        let out_limbs = keccak_output_limbs_from_output(&inv.output);
        for (k, limb) in out_limbs.iter().enumerate() {
            columns[COL_KECCAK_OUTPUT_LIMB_OFFSET + k][i] = Scalar::from_u64(*limb, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
        // Task #318 mirror: prefer explicit per-row override; fall back
        // to `output[0]` when the caller didn't populate `mirror_byte0`.
        let mirror_byte = if witness.mirror_byte0.is_empty() {
            inv.output[0]
        } else {
            witness.mirror_byte0[i]
        };
        columns[COL_MIRROR_BYTE0][i] = Scalar::from_u64(mirror_byte as u64, curve);
    }

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

/// Bytes that participate in `address_limb[k]` in the LE
/// decomposition of a 20-byte address packed at `output[12..32]`.
/// k=0 pulls bytes 12..20, k=1 pulls 20..28, k=2 pulls 28..32 (4
/// bytes — high 4 bytes of limb 2 are zero), k=3 pulls nothing
/// (limb 3 is identically zero).
fn address_limb_byte_count(k: usize) -> usize {
    match k {
        0 | 1 => 8,
        2 => 4,
        _ => 0,
    }
}

/// Build the byte-power scalar `Scalar::from_u64(1u64 << (8 * i))`
/// for a given byte position `i ∈ 0..8`.
fn byte_power(i: usize, curve: CurveType) -> Scalar {
    debug_assert!(i < 8, "byte_power overflows u64 for i ≥ 8");
    Scalar::from_u64(1u64 << (8 * i), curve)
}

/// Evaluate `address_limb[k] − Σ_{i=0..bytes} 2^(8 i) · OUTPUT_BYTE[12 + 8 k + i]`
/// at one row.
fn eval_address_limb_binding(col_evals: &[Scalar], k: usize) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let bytes = address_limb_byte_count(k);
    let mut sum = Scalar::zero(curve);
    for i in 0..bytes {
        let byte = &col_evals[COL_OUTPUT_BYTE_OFFSET + 12 + 8 * k + i];
        sum = sum.add(&byte.mul(&byte_power(i, curve)));
    }
    let limb = &col_evals[COL_ADDRESS_LIMB_OFFSET + k];
    limb.sub(&sum)
}

/// Evaluate body 1+k as a coefficient polynomial.
fn build_address_limb_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    k: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let bytes = address_limb_byte_count(k);
    let mut sum = vec![Scalar::zero(curve)];
    for i in 0..bytes {
        let byte_poly = &col_coeffs[COL_OUTPUT_BYTE_OFFSET + 12 + 8 * k + i];
        let scaled = poly_scalar_mul(byte_poly, &byte_power(i, curve));
        sum = poly_add(&sum, &scaled, curve);
    }
    let limb_poly = &col_coeffs[COL_ADDRESS_LIMB_OFFSET + k];
    poly_sub(limb_poly, &sum, curve)
}

/// Evaluate `KECCAK_OUTPUT_LIMB[k] − Σ_{j=0..8} 2^(8 j) · OUTPUT_BYTE[(3-k)·8 + (7-j)]`
/// at one row. The byte offsets implement big-endian aggregation
/// matching `u64::from_be_bytes(output[(3-k)*8 .. (3-k)*8+8])`, which
/// is in turn the `U256::from_be_bytes(output).as_limbs()[k]`
/// convention used by EVM stack pushes of keccak results.
fn eval_keccak_output_limb_binding(col_evals: &[Scalar], k: usize) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let base = COL_OUTPUT_BYTE_OFFSET + (3 - k) * 8;
    let mut sum = Scalar::zero(curve);
    for j in 0..8 {
        let byte = &col_evals[base + (7 - j)];
        sum = sum.add(&byte.mul(&byte_power(j, curve)));
    }
    let limb = &col_evals[COL_KECCAK_OUTPUT_LIMB_OFFSET + k];
    limb.sub(&sum)
}

fn build_keccak_output_limb_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    k: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let base = COL_OUTPUT_BYTE_OFFSET + (3 - k) * 8;
    let mut sum = vec![Scalar::zero(curve)];
    for j in 0..8 {
        let byte_poly = &col_coeffs[base + (7 - j)];
        let scaled = poly_scalar_mul(byte_poly, &byte_power(j, curve));
        sum = poly_add(&sum, &scaled, curve);
    }
    let limb_poly = &col_coeffs[COL_KECCAK_OUTPUT_LIMB_OFFSET + k];
    poly_sub(limb_poly, &sum, curve)
}

impl VmConstraintSystem for KeccakExtractConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "address_limb_0_binding".into(),
            "address_limb_1_binding".into(),
            "address_limb_2_binding".into(),
            "address_limb_3_zero".into(),
            "keccak_output_limb_0_binding".into(),
            "keccak_output_limb_1_binding".into(),
            "keccak_output_limb_2_binding".into(),
            "keccak_output_limb_3_binding".into(),
        ]
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
        let mut bin = vec![Scalar::zero(curve); n];
        let mut limb_bodies: [Vec<Scalar>; ADDRESS_LIMB_LEN] =
            std::array::from_fn(|_| vec![Scalar::zero(curve); n]);
        let mut keccak_limb_bodies: [Vec<Scalar>; KECCAK_OUTPUT_LIMB_LEN] =
            std::array::from_fn(|_| vec![Scalar::zero(curve); n]);
        for row in 0..n {
            let v = &columns[COL_IS_REAL][row];
            bin[row] = v.mul(&v.sub(&one));
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            for k in 0..ADDRESS_LIMB_LEN {
                limb_bodies[k][row] = eval_address_limb_binding(&row_evals, k);
            }
            for k in 0..KECCAK_OUTPUT_LIMB_LEN {
                keccak_limb_bodies[k][row] = eval_keccak_output_limb_binding(&row_evals, k);
            }
        }
        let mut out = vec![bin];
        for body in limb_bodies {
            out.push(body);
        }
        for body in keccak_limb_bodies {
            out.push(body);
        }
        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        let mut acc = v.mul(&v.sub(&one));
        let mut ap = alpha.clone();
        for k in 0..ADDRESS_LIMB_LEN {
            acc = acc.add(&eval_address_limb_binding(col_evals, k).mul(&ap));
            ap = ap.mul(alpha);
        }
        for k in 0..KECCAK_OUTPUT_LIMB_LEN {
            acc = acc.add(&eval_keccak_output_limb_binding(col_evals, k).mul(&ap));
            ap = ap.mul(alpha);
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
        let v = &col_coeffs[COL_IS_REAL];
        let v_minus_1 = poly_sub(v, &one_poly, curve);
        let mut acc = poly_mul(v, &v_minus_1, curve);
        let mut ap = alpha.clone();
        for k in 0..ADDRESS_LIMB_LEN {
            let body = build_address_limb_binding_poly(col_coeffs, k, curve);
            let scaled = poly_scalar_mul(&body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        for k in 0..KECCAK_OUTPUT_LIMB_LEN {
            let body = build_keccak_output_limb_binding_poly(col_coeffs, k, curve);
            let scaled = poly_scalar_mul(&body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
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
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp linkage helpers ──────────────────────────────────

/// Convenience wrapper over [`make_mpt_keccak_extract_linkage_descriptor`]
/// that uses the canonical MPT column indices from
/// [`crate::mpt_air::col`].
///
/// `mpt_layer_index` and `keccak_extract_layer_index` index the MPT and
/// KeccakExtract AIRs in the layer chain respectively. Returns a
/// ready-to-use descriptor binding MPT's `(NODE_RLP[0..256], NODE_RLP_LEN,
/// NODE_HASH[0..32])` to KeccakExtract's `(INPUT[0..256], INPUT_LEN,
/// OUTPUT[0..32])`.
pub fn make_mpt_keccak_extract_linkage_descriptor_default(
    mpt_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mpt_rlp_bytes: Vec<usize> = (0..MAX_INPUT_LEN)
        .map(|b| crate::mpt_air::col::NODE_RLP_OFFSET + b)
        .collect();
    let mpt_node_hash: Vec<usize> = (0..OUTPUT_LEN)
        .map(|b| crate::mpt_air::col::NODE_HASH_OFFSET + b)
        .collect();
    make_mpt_keccak_extract_linkage_descriptor(
        mpt_layer_index,
        keccak_extract_layer_index,
        mpt_rlp_bytes,
        crate::mpt_air::col::NODE_RLP_LEN,
        mpt_node_hash,
        // No selector — every MPT row is a real node, padding rows
        // are all-zero on both sides and tuples match in the multiset.
        None,
    )
}

/// Construct the descriptor binding MPT node-hash rows to this extract
/// AIR.
///
/// **A side (MPT)**: `(node_rlp_bytes, node_rlp_len, node_hash)` tuple
/// where `node_rlp_bytes` is the RLP-encoded node and `node_hash =
/// keccak256(node_rlp_bytes[..node_rlp_len])`.
///
/// **B side (this AIR)**: `(INPUT[0..MAX_INPUT_LEN], INPUT_LEN,
/// OUTPUT[0..32])` — same tuple shape on both sides.
///
/// **Soundness scope**: identical to [`crate::sha256_extract`] — this
/// linkage proves multiset equivalence of the byte tuples but doesn't
/// prove `OUTPUT = keccak256(INPUT)` algebraically. That requires a
/// separate cross-AIR LogUp from this AIR to the bit-level
/// `keccak_constraints` AIR.
///
/// The `mpt_rlp_byte_columns`/`mpt_rlp_len_column`/`mpt_node_hash_columns`
/// parameters allow callers to override the default MPT column indices
/// (e.g. for a future wide-RLP variant). Most callers should use
/// [`make_mpt_keccak_extract_linkage_descriptor_default`].
pub fn make_mpt_keccak_extract_linkage_descriptor(
    mpt_layer_index: usize,
    keccak_extract_layer_index: usize,
    mpt_rlp_byte_columns: Vec<usize>,
    mpt_rlp_len_column: usize,
    mpt_node_hash_columns: Vec<usize>,
    mpt_selector_column: Option<usize>,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    assert_eq!(
        mpt_rlp_byte_columns.len(),
        MAX_INPUT_LEN,
        "MPT RLP byte columns must match KeccakExtract MAX_INPUT_LEN"
    );
    assert_eq!(
        mpt_node_hash_columns.len(),
        OUTPUT_LEN,
        "MPT node-hash columns must be {} bytes",
        OUTPUT_LEN
    );

    // A side: MPT's RLP bytes + RLP length + node hash.
    let mut a_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_LEN + 1 + OUTPUT_LEN);
    a_columns.extend(mpt_rlp_byte_columns);
    a_columns.push(mpt_rlp_len_column);
    a_columns.extend(mpt_node_hash_columns);

    // B side: this AIR's INPUT bytes + INPUT_LEN + OUTPUT bytes.
    let mut b_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_LEN + 1 + OUTPUT_LEN);
    for b in 0..MAX_INPUT_LEN {
        b_columns.push(COL_INPUT_BYTE_OFFSET + b);
    }
    b_columns.push(COL_INPUT_LEN);
    for b in 0..OUTPUT_LEN {
        b_columns.push(COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "mpt_keccak_extract_v1".into(),
        a_layer_index: mpt_layer_index,
        a_columns,
        a_selector_column: mpt_selector_column,
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding this AIR to the bit-level Keccak
/// AIR ([`crate::keccak_constraints`]).
///
/// **A side (this AIR)**: per-row `(INPUT_BYTE[0..256], INPUT_LEN,
/// OUTPUT_BYTE[0..32])` gated by `IS_REAL`.
///
/// **B side (bit-level Keccak)**: per-invocation aggregator columns
/// `(INV_INPUT_BYTE[0..256], INV_INPUT_LEN_COL, INV_OUTPUT_BYTE[0..32])`
/// gated by `IS_FIRST_INV_ROW`. The bit-level Keccak AIR represents one
/// `keccak256(input)` invocation; on the anchor row (first round of the
/// first absorption), the aggregator columns hold the input bytes
/// (zero-padded to 256), the actual input length, and the 32-byte
/// digest.
///
/// **Soundness scope (current)**: structural plumbing only — the
/// aggregator bytes are witness-populated, NOT algebraically bound to
/// the bit-level state. Closing the binding requires:
///
///   1. **INV_INPUT_BYTE ↔ first absorption block's BEFORE bits at
///      row 0** — the Keccak rate (1088 bits = 136 bytes for SHA3-256)
///      decomposes to the input bytes XORed with the prior absorption
///      state; for the first absorption, prior state = 0.
///   2. **INV_OUTPUT_BYTE ↔ AFTER_IOTA at the squeeze row** — the
///      32-byte digest is the first 256 bits of the lane-encoded
///      AFTER_IOTA at the last round of the squeeze block.
///   3. **INV_*_BYTE invariance across rows** + **IS_FIRST_INV_ROW
///      binarity** — both deferred until the constraint extension lands.
///
/// Today the linkage proves "the bit-level Keccak AIR committed *some*
/// (input, len, output) tuple per invocation that matches the
/// KeccakExtract row" — equivalent to the witness builder's correctness
/// guarantee. The algebraic strength comes when items 1–3 land. Tracked
/// in `cross_air_logup_dependent_tasks.md` as the "shared SHA-256/Keccak
/// byte-aggregation" follow-up.
pub fn make_keccak_extract_keccak_linkage_descriptor(
    keccak_extract_layer_index: usize,
    keccak_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_air as kk;

    // A side (KeccakExtract): 256 input bytes + 1 input length + 32
    // output bytes.
    let mut a_columns: Vec<usize> = Vec::with_capacity(MAX_INPUT_LEN + 1 + OUTPUT_LEN);
    for b in 0..MAX_INPUT_LEN {
        a_columns.push(COL_INPUT_BYTE_OFFSET + b);
    }
    a_columns.push(COL_INPUT_LEN);
    for b in 0..OUTPUT_LEN {
        a_columns.push(COL_OUTPUT_BYTE_OFFSET + b);
    }

    // B side (bit-level Keccak): the per-invocation byte aggregator
    // columns gated by IS_FIRST_INV_ROW.
    let mut b_columns: Vec<usize> = Vec::with_capacity(kk::INV_INPUT_LEN + 1 + kk::INV_OUTPUT_LEN);
    for b in 0..kk::INV_INPUT_LEN {
        b_columns.push(kk::inv_input_byte(b));
    }
    b_columns.push(kk::COL_INV_INPUT_LEN_COL);
    for b in 0..kk::INV_OUTPUT_LEN {
        b_columns.push(kk::inv_output_byte(b));
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "keccak_extract_keccak_v1".into(),
        a_layer_index: keccak_extract_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_layer_index,
        b_columns,
        b_selector_column: Some(kk::COL_IS_FIRST_INV_ROW),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_witness() -> KeccakExtractWitness {
        KeccakExtractWitness::from_inputs(&[
            b"hello world".to_vec(),
            b"".to_vec(),
            b"foobar baz qux".to_vec(),
        ])
        .expect("test inputs are within MAX_INPUT_LEN")
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 3);
        // Row 0: input "hello world" = 11 bytes, first byte 'h' = 0x68.
        assert_eq!(
            trace.columns[COL_INPUT_BYTE_OFFSET].evaluations[0].to_bytes(),
            Scalar::from_u64(0x68, curve).to_bytes()
        );
        assert_eq!(
            trace.columns[COL_INPUT_LEN].evaluations[0].to_bytes(),
            Scalar::from_u64(11, curve).to_bytes()
        );
        assert_eq!(
            trace.columns[COL_IS_REAL].evaluations[0].to_bytes(),
            Scalar::one(curve).to_bytes()
        );
        // Row 1: empty input, INPUT_LEN = 0.
        assert!(trace.columns[COL_INPUT_LEN].evaluations[1].is_zero());
        // Padding row: IS_REAL = 0.
        assert!(trace.columns[COL_IS_REAL].evaluations.last().unwrap().is_zero());
    }

    #[test]
    fn from_inputs_computes_keccak256() {
        let w = small_witness();
        // First row: keccak256("hello world").
        let expected = crate::keccak::keccak256(b"hello world");
        assert_eq!(w.invocations[0].output, expected);
        // Second row: keccak256("") = empty digest.
        let expected_empty = crate::keccak::keccak256(b"");
        assert_eq!(w.invocations[1].output, expected_empty);
    }

    #[test]
    fn from_inputs_rejects_oversize() {
        let big = vec![0u8; MAX_INPUT_LEN + 1];
        let r = KeccakExtractWitness::from_inputs(&[big]);
        assert!(r.is_err(), "inputs longer than MAX_INPUT_LEN must be rejected");
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = KeccakExtractConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(7, curve);
        for row in 0..trace.padded_size as usize {
            let col_vals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[row].clone())
                .collect();
            let c_at = cs.evaluate_at_point(&col_vals, &alpha);
            assert!(c_at.is_zero(), "row {} body must vanish on honest witness", row);
        }
    }

    #[test]
    fn mpt_keccak_extract_descriptor_well_formed() {
        // Default constructor wires real MPT columns from
        // `crate::mpt_air::col`.
        let desc = make_mpt_keccak_extract_linkage_descriptor_default(0, 1);
        assert_eq!(desc.label, "mpt_keccak_extract_v1");
        // Tuple = MAX_INPUT_LEN + 1 (length) + OUTPUT_LEN.
        let tuple_len = MAX_INPUT_LEN + 1 + OUTPUT_LEN;
        assert_eq!(desc.a_columns.len(), tuple_len);
        assert_eq!(desc.b_columns.len(), tuple_len);
        // A side: real MPT NODE_RLP bytes, then NODE_RLP_LEN, then NODE_HASH.
        assert_eq!(desc.a_columns[0], crate::mpt_air::col::NODE_RLP_OFFSET);
        assert_eq!(
            desc.a_columns[MAX_INPUT_LEN - 1],
            crate::mpt_air::col::NODE_RLP_OFFSET + MAX_INPUT_LEN - 1
        );
        assert_eq!(
            desc.a_columns[MAX_INPUT_LEN],
            crate::mpt_air::col::NODE_RLP_LEN
        );
        assert_eq!(
            desc.a_columns[MAX_INPUT_LEN + 1],
            crate::mpt_air::col::NODE_HASH_OFFSET
        );
        // B side: INPUT bytes, INPUT_LEN, OUTPUT bytes.
        assert_eq!(desc.b_columns[0], COL_INPUT_BYTE_OFFSET);
        assert_eq!(desc.b_columns[MAX_INPUT_LEN - 1], COL_INPUT_BYTE_OFFSET + MAX_INPUT_LEN - 1);
        assert_eq!(desc.b_columns[MAX_INPUT_LEN], COL_INPUT_LEN);
        assert_eq!(desc.b_columns[MAX_INPUT_LEN + 1], COL_OUTPUT_BYTE_OFFSET);
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn keccak_extract_keccak_descriptor_well_formed() {
        use crate::keccak_air as kk;
        let desc = make_keccak_extract_keccak_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "keccak_extract_keccak_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        let tuple_len = MAX_INPUT_LEN + 1 + OUTPUT_LEN;
        assert_eq!(desc.a_columns.len(), tuple_len);
        assert_eq!(desc.b_columns.len(), tuple_len);
        // A side: KeccakExtract INPUT bytes, INPUT_LEN, OUTPUT bytes.
        assert_eq!(desc.a_columns[0], COL_INPUT_BYTE_OFFSET);
        assert_eq!(desc.a_columns[MAX_INPUT_LEN], COL_INPUT_LEN);
        assert_eq!(desc.a_columns[MAX_INPUT_LEN + 1], COL_OUTPUT_BYTE_OFFSET);
        // B side: bit-level Keccak INV_INPUT_BYTE, INV_INPUT_LEN_COL, INV_OUTPUT_BYTE.
        assert_eq!(desc.b_columns[0], kk::COL_INV_INPUT_BYTE_OFFSET);
        assert_eq!(desc.b_columns[kk::INV_INPUT_LEN], kk::COL_INV_INPUT_LEN_COL);
        assert_eq!(desc.b_columns[kk::INV_INPUT_LEN + 1], kk::COL_INV_OUTPUT_BYTE_OFFSET);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(kk::COL_IS_FIRST_INV_ROW));
    }

    #[test]
    fn keccak_extract_byte_tuples_match_bit_level_aggregator() {
        // Build a single keccak256 witness, populate the bit-level
        // Keccak trace WITH invocation aggregator columns, assert the
        // aggregator bytes match KeccakExtract's single row.
        use crate::keccak::keccak_witness;
        use crate::keccak_air as kk;
        let curve = CurveType::Bls48581;

        let input = b"hello world".to_vec();
        let ht = keccak_witness(&input);
        let digest = crate::keccak::keccak256(&input);
        let bit_cols = kk::populate_trace_from_hash_with_invocation_bytes(
            &ht, &input, &digest, curve,
        );

        let n = bit_cols[0].len();
        for r in 0..n {
            for (b, &v) in input.iter().enumerate() {
                let got = bit_cols[kk::inv_input_byte(b)][r].to_u64() as u8;
                assert_eq!(got, v, "row {} input byte {}", r, b);
            }
            // Bytes past input.len() are zero-padding.
            for b in input.len()..kk::INV_INPUT_LEN {
                assert!(
                    bit_cols[kk::inv_input_byte(b)][r].is_zero(),
                    "row {} byte {} must be zero-padded",
                    r,
                    b
                );
            }
            assert_eq!(
                bit_cols[kk::COL_INV_INPUT_LEN_COL][r].to_u64() as usize,
                input.len(),
                "row {} INV_INPUT_LEN_COL must equal input length",
                r
            );
            for b in 0..kk::INV_OUTPUT_LEN {
                let got = bit_cols[kk::inv_output_byte(b)][r].to_u64() as u8;
                assert_eq!(got, digest[b], "row {} output byte {}", r, b);
            }
        }
        assert_eq!(bit_cols[kk::COL_IS_FIRST_INV_ROW][0].to_u64(), 1);
        for r in 1..n {
            assert_eq!(
                bit_cols[kk::COL_IS_FIRST_INV_ROW][r].to_u64(),
                0,
                "IS_FIRST_INV_ROW must be 0 on row {}",
                r
            );
        }

        // KeccakExtract row matches aggregator at the anchor.
        let extract_w = KeccakExtractWitness::from_inputs(&[input.clone()])
            .expect("input within MAX_INPUT_LEN");
        let extract_trace = build_trace_polynomials(&extract_w, curve);
        for b in 0..MAX_INPUT_LEN {
            let extract_byte =
                extract_trace.columns[COL_INPUT_BYTE_OFFSET + b].evaluations[0].to_u64() as u8;
            let bit_anchor_byte = bit_cols[kk::inv_input_byte(b)][0].to_u64() as u8;
            assert_eq!(extract_byte, bit_anchor_byte, "input tuple mismatch at byte {}", b);
        }
        let extract_len = extract_trace.columns[COL_INPUT_LEN].evaluations[0].to_u64() as usize;
        let bit_len = bit_cols[kk::COL_INV_INPUT_LEN_COL][0].to_u64() as usize;
        assert_eq!(extract_len, bit_len, "input length tuple mismatch");
        for b in 0..OUTPUT_LEN {
            let extract_byte =
                extract_trace.columns[COL_OUTPUT_BYTE_OFFSET + b].evaluations[0].to_u64() as u8;
            let bit_anchor_byte = bit_cols[kk::inv_output_byte(b)][0].to_u64() as u8;
            assert_eq!(extract_byte, bit_anchor_byte, "output tuple mismatch at byte {}", b);
        }
    }

    #[test]
    fn mpt_keccak_extract_byte_tuples_match_real_mpt_witness() {
        // Build a real MPT inclusion witness, then verify the bytes
        // each AIR commits at row i match: the MPT row's
        // (node_rlp[..node_rlp_len], node_hash) tuple equals the
        // KeccakExtract row's (input, input_len, output) tuple.
        use crate::mpt::{single_leaf_trie, MptNode, Nibbles};
        use crate::mpt_air::inclusion_witness;

        let key_a = vec![0x1a, 0xbc];
        let val_a = b"value-a".to_vec();
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xa, 0xb, 0xc]),
            value: val_a.clone(),
        };
        let leaf_a_rlp = crate::mpt::mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = crate::mpt::mpt_node_rlp(&MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xd, 0xe, 0xf]),
            value: b"vb".to_vec(),
        });
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(crate::keccak::keccak256(&leaf_a_rlp));
        children[2] = Some(crate::keccak::keccak256(&leaf_b_rlp));
        let branch_rlp = crate::mpt::mpt_node_rlp(&MptNode::Branch {
            children,
            value: None,
        });

        let proof = vec![branch_rlp.clone(), leaf_a_rlp.clone()];
        let mpt_rows = inclusion_witness(&key_a, &proof);
        assert_eq!(mpt_rows.len(), 2);

        // Build matching KeccakExtract witness from the same RLP inputs.
        let inputs: Vec<Vec<u8>> = mpt_rows
            .iter()
            .map(|r| r.node_rlp[..r.node_rlp_len].to_vec())
            .collect();
        let ke_w = KeccakExtractWitness::from_inputs(&inputs)
            .expect("MPT RLP inputs are within MAX_INPUT_LEN");

        for (i, mpt_row) in mpt_rows.iter().enumerate() {
            let ke_row = &ke_w.invocations[i];
            assert_eq!(mpt_row.node_rlp_len, ke_row.input_len);
            assert_eq!(&mpt_row.node_rlp[..], &ke_row.input[..]);
            assert_eq!(mpt_row.node_hash, ke_row.output);
        }

        // Sanity: also confirm KeccakExtract's keccak256 computation
        // matches what MPT wrote.
        let _ = single_leaf_trie; // silence unused if not used elsewhere
    }

    #[test]
    fn address_limbs_match_evm_address_to_limbs_convention() {
        // Spot check: the 4 u64 limbs computed from output[12..32] must
        // equal LE u64 packing — same convention the EVM inspector uses
        // in `address_to_limbs` (`crates/evm/src/inspector.rs`).
        let mut output = [0u8; 32];
        for i in 12..32 {
            output[i] = (i - 12) as u8 + 1; // bytes 1..20 (skip 0)
        }
        let limbs = address_limbs_from_output(&output);
        // limb[0] = bytes 12..20 = [1,2,3,4,5,6,7,8] LE
        let expected_l0 = u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(limbs[0], expected_l0);
        // limb[1] = bytes 20..28 = [9..16] LE
        let expected_l1 = u64::from_le_bytes([9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(limbs[1], expected_l1);
        // limb[2] = bytes 28..32 || zeros = [17,18,19,20,0,0,0,0] LE
        let expected_l2 = u64::from_le_bytes([17, 18, 19, 20, 0, 0, 0, 0]);
        assert_eq!(limbs[2], expected_l2);
        // limb[3] = always 0 (a 20-byte address never spills into limb 3)
        assert_eq!(limbs[3], 0);
    }

    #[test]
    fn address_limb_binding_rejects_tampered_limb() {
        // Tampering with `address_limb_0` while leaving output bytes
        // untouched must make body 1 (`address_limb_0_binding`) fire.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = KeccakExtractConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(11, curve);
        // Honest baseline: row 0 body must vanish.
        let row_evals0: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(cs.evaluate_at_point(&row_evals0, &alpha).is_zero());
        // Flip limb 0 to a wrong value at row 0.
        trace.columns[COL_ADDRESS_LIMB_OFFSET].evaluations[0] =
            Scalar::from_u64(0xDEAD_BEEF_DEAD_BEEF, curve);
        let row_evals0_bad: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals0_bad, &alpha).is_zero(),
            "body must fire when address_limb_0 disagrees with OUTPUT_BYTE[12..20]"
        );
    }

    #[test]
    fn address_limb_binding_rejects_tampered_output_byte() {
        // Tampering with an output byte in [12..32] while leaving the
        // address limb untouched must also fire body 1+k.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = KeccakExtractConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(13, curve);
        // Tamper with byte 28 (start of limb 2).
        let cur = trace.columns[COL_OUTPUT_BYTE_OFFSET + 28].evaluations[0].clone();
        let one = Scalar::one(curve);
        trace.columns[COL_OUTPUT_BYTE_OFFSET + 28].evaluations[0] = cur.add(&one);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "body must fire when OUTPUT_BYTE[28] disagrees with address_limb[2]"
        );
    }

    #[test]
    fn address_limb_3_zero_rejects_nonzero_limb_3() {
        // Limb 3 must be identically zero — a 20-byte address never
        // spills into it. Setting it nonzero must fire body 4
        // (`address_limb_3_zero`).
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = KeccakExtractConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        trace.columns[COL_ADDRESS_LIMB_OFFSET + 3].evaluations[0] =
            Scalar::from_u64(1, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "body must fire when address_limb_3 is nonzero"
        );
    }

    #[test]
    fn build_trace_populates_address_limbs() {
        // The witness builder must compute address_limb[k] from
        // output[12..32] using the LE convention.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        for (i, inv) in w.invocations.iter().enumerate() {
            let expected = address_limbs_from_output(&inv.output);
            for (k, limb) in expected.iter().enumerate() {
                let got = trace.columns[COL_ADDRESS_LIMB_OFFSET + k].evaluations[i]
                    .to_u64();
                assert_eq!(
                    got, *limb,
                    "row {} limb {} mismatch", i, k
                );
            }
        }
    }

    #[test]
    fn keccak_output_limbs_match_u256_from_be_bytes_convention() {
        // The full-output limb helper must produce the same 4 u64 limbs
        // as `U256::from_be_bytes(output).as_limbs()` — the convention
        // EVM stack pushes use for keccak results (e.g. SHA3 opcode).
        // Hand-computed against the formula
        //   number = Σ_{i=0..32} output[i] · 2^(8·(31−i))
        //   limb[k] = (number / 2^(64k)) mod 2^64
        // which expands to limb[k] = u64::from_be_bytes(output[(3−k)·8 .. (3−k)·8+8]).
        let output: [u8; 32] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,   // bytes 0..8 → limb 3
            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,   // bytes 8..16 → limb 2
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,   // bytes 16..24 → limb 1
            0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,   // bytes 24..32 → limb 0
        ];
        let ours = keccak_output_limbs_from_output(&output);
        // limb[3] = u64::from_be_bytes([0x01..0x08]) = 0x01_02_03_04_05_06_07_08
        assert_eq!(ours[3], 0x0102030405060708u64);
        assert_eq!(ours[2], 0x090a0b0c0d0e0f10u64);
        assert_eq!(ours[1], 0x1112131415161718u64);
        // limb[0] (low 64 bits) = u64::from_be_bytes([0x19..0x20])
        assert_eq!(ours[0], 0x191a1b1c1d1e1f20u64);
    }

    #[test]
    fn build_trace_populates_keccak_output_limbs() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        for (i, inv) in w.invocations.iter().enumerate() {
            let expected = keccak_output_limbs_from_output(&inv.output);
            for (k, limb) in expected.iter().enumerate() {
                let got = trace.columns[COL_KECCAK_OUTPUT_LIMB_OFFSET + k]
                    .evaluations[i].to_u64();
                assert_eq!(got, *limb, "row {} keccak_output_limb {} mismatch", i, k);
            }
        }
    }

    #[test]
    fn keccak_output_limb_binding_rejects_tampered_limb() {
        // Tamper with KECCAK_OUTPUT_LIMB[2] while leaving OUTPUT_BYTE
        // bytes [8..16] honest → body 5+2 must fire.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = KeccakExtractConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(7, curve);
        let row_evals0: Vec<Scalar> = trace
            .columns.iter().map(|p| p.evaluations[0].clone()).collect();
        assert!(cs.evaluate_at_point(&row_evals0, &alpha).is_zero());
        trace.columns[COL_KECCAK_OUTPUT_LIMB_OFFSET + 2].evaluations[0] =
            Scalar::from_u64(0xCAFE_BABE_DEAD_BEEF, curve);
        let row_evals0_bad: Vec<Scalar> = trace
            .columns.iter().map(|p| p.evaluations[0].clone()).collect();
        assert!(
            !cs.evaluate_at_point(&row_evals0_bad, &alpha).is_zero(),
            "body must fire when KECCAK_OUTPUT_LIMB[2] disagrees with OUTPUT_BYTE[8..16]"
        );
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn keccak_extract_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = KeccakExtractConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "keccak extract proof must verify");
    }
}
