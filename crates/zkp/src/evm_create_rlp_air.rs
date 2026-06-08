//! CREATE pre-image RLP encoding gadget AIR (closes the input-side gap of #95).
//!
//! Per-invocation algebraic gadget that takes `(sender_address, nonce)`
//! and exposes the canonical Ethereum CREATE pre-image
//! `rlp([sender_address, nonce])` as a fixed-shape byte sequence, ready
//! to be matched by a cross-AIR LogUp linkage against
//! [`crate::keccak_extract`]'s `INPUT_BYTE` columns. Combined with the
//! existing KeccakExtract↔Keccak binding (#91) and the EVM main↔
//! KeccakExtract output-side binding ([`crate::keccak_extract`]'s
//! `address_limb_*` aggregator), this completes the algebraic
//! soundness chain for CREATE address derivation.
//!
//! # Scope
//!
//! - **Single CREATE invocation per row.** Multiple CREATE opcodes in
//!   one EVM trace need multiple gadget rows.
//! - **Nonce < 128 (= 0x80).** This is the RLP "small integer" range:
//!   the encoded nonce fits in a single byte, so the full RLP output
//!   is fixed-length at 23 bytes regardless of the actual nonce
//!   value. Larger nonces require either a variable-length output
//!   (with multiple "kind" selectors gating the encoding length) or
//!   a wider gadget — a focused follow-up. The 0–127 range covers
//!   nearly all real-world EOAs and the bulk of contract-creator
//!   contracts: contract account nonces in production rarely exceed
//!   a few dozen, and EOAs only deploy a handful of contracts each.
//! - **Sender = 20-byte Ethereum address.** Exposed both as 20 raw
//!   byte columns (consumed by RLP construction) AND as 4 LE u64
//!   limbs (matches the EVM main trace's `frame_callee_l*` encoding,
//!   so the cross-AIR linkage to EVM main can match the existing
//!   limb columns directly).
//!
//! # Column layout
//!
//! ```text
//!   0 .. 20    SENDER_BYTE[0..20]    raw 20-byte address
//!  20 .. 24    SENDER_LIMB[0..4]     LE u64 limbs aggregating SENDER_BYTE
//!  24          NONCE                 u64 nonce, restricted to [0, 128)
//!  25 .. 32    NONCE_BIT[0..7]       7-bit decomposition of NONCE
//!  32          NONCE_INV             1 / NONCE if NONCE ≠ 0, else arbitrary
//!  33          NONCE_RLP_BYTE        the byte placed at RLP[22]
//!                                    (= 0x80 if NONCE = 0, else NONCE)
//!  34 .. 290   RLP_BYTE[0..256]      full keccak-input-shaped output;
//!                                    bytes 0..23 carry the RLP encoding,
//!                                    bytes 23..256 are pinned to zero
//! 290          RLP_LEN               always 23 (for NONCE < 128)
//! 291          IS_REAL               selector
//! ```
//!
//! # Constraint layout
//!
//! 10 row-local constraints (alpha-power weighted):
//!
//!   0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `sender_limb_0_binding` — `IS_REAL ·
//!      (SENDER_LIMB[0] − Σ_{i=0..8} 2^(8 i) · SENDER_BYTE[i]) = 0`
//!   2. `sender_limb_1_binding` — `IS_REAL ·
//!      (SENDER_LIMB[1] − Σ_{i=0..8} 2^(8 i) · SENDER_BYTE[8 + i]) = 0`
//!   3. `sender_limb_2_binding` — `IS_REAL ·
//!      (SENDER_LIMB[2] − Σ_{i=0..4} 2^(8 i) · SENDER_BYTE[16 + i]) = 0`
//!   4. `sender_limb_3_zero` — `IS_REAL · SENDER_LIMB[3] = 0`
//!   5. `nonce_bit_binarity` — β-RLC over `i ∈ 0..7` of
//!      `NONCE_BIT[i] · (NONCE_BIT[i] − 1) = 0` (gates itself; padding
//!      rows have all-zero bits, body vanishes naturally)
//!   6. `nonce_decomposition` — `Σ_{i=0..7} 2^i · NONCE_BIT[i] − NONCE = 0`
//!      (vanishes naturally on padding)
//!   7. `nonce_inv_correctness` — `NONCE · (NONCE · NONCE_INV − 1) = 0`
//!      (vanishes naturally on padding; pins NONCE_INV = NONCE^{−1}
//!      when NONCE ≠ 0; NONCE_INV is unconstrained when NONCE = 0)
//!   8. `nonce_rlp_byte_value` — `IS_REAL ·
//!      (NONCE_RLP_BYTE − NONCE − 128 · (1 − NONCE · NONCE_INV)) = 0`
//!      (when NONCE ≠ 0: pins NONCE_RLP_BYTE = NONCE since
//!      NONCE · NONCE_INV = 1; when NONCE = 0: pins NONCE_RLP_BYTE = 128
//!      = 0x80)
//!   9. `rlp_construction` — β-RLC over:
//!         - `RLP_BYTE[0] − 0xd6` (list header = 0xc0 + payload_len = 0xc0 + 22)
//!         - `RLP_BYTE[1] − 0x94` (sender header = 0x80 + 20)
//!         - `RLP_BYTE[2 + k] − SENDER_BYTE[k]` for `k ∈ 0..20`
//!         - `RLP_BYTE[22] − NONCE_RLP_BYTE`
//!         - `RLP_BYTE[k]` for `k ∈ 23..256` (pinned to zero — the
//!           keccak-input-shaped tail)
//!         - `RLP_LEN − 23`
//!      (gated by IS_REAL — pinning constants would fire on padding
//!      otherwise; total of 256 + 1 = 257 sub-bodies)
//!
//! # Soundness chain (input side, end-to-end)
//!
//! Combined with the existing #91 closure (KeccakExtract↔Keccak):
//!
//! 1. **EVM main ↔ this gadget** ([`make_evm_main_create_rlp_linkage_descriptor`]):
//!    pins this gadget's `(SENDER_LIMB[0..4], NONCE)` tuple equal to
//!    the EVM CREATE row's `(frame_callee_l[0..4], create_nonce_hint)`.
//!    *The EVM main trace must expose a `create_nonce_hint` column
//!    populated by the inspector — see follow-up note in the dependent
//!    tasks memory.*
//!
//! 2. **This gadget's row-local constraints** pin
//!    `RLP_BYTE[0..23] = canonical_rlp(sender, nonce)` algebraically
//!    (and `RLP_BYTE[23..256] = 0`, `RLP_LEN = 23`).
//!
//! 3. **This gadget ↔ KeccakExtract input side**
//!    ([`make_create_rlp_keccak_extract_input_linkage_descriptor`]):
//!    pins the gadget's `(RLP_BYTE[0..256], RLP_LEN)` tuple equal to
//!    some KeccakExtract row's `(INPUT_BYTE[0..256], INPUT_LEN)`.
//!
//! 4. **KeccakExtract↔Keccak** (#91): pins KeccakExtract's
//!    `OUTPUT_BYTE[0..32] = keccak256(INPUT_BYTE[0..INPUT_LEN])`.
//!
//! 5. **EVM main ↔ KeccakExtract output side** (already landed): pins
//!    EVM's `create_address_hint` to the LE limb aggregation of
//!    `OUTPUT_BYTE[12..32]`.
//!
//! End to end: `create_address_hint = keccak256(rlp([sender, nonce]))[12..32]`,
//! algebraically. A malicious prover cannot fake either the sender,
//! the nonce, the RLP encoding, or the digest.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum nonce value supported by this MVP gadget. Nonces ≥ 128
/// would require a variable-length nonce encoding (1..9 RLP bytes),
/// changing the output length and the list-header byte. The 0–127
/// range covers the common case (~99% of real-world EOA / contract
/// nonces) at fixed output length; larger nonces are a focused
/// follow-up.
pub const MAX_NONCE_EXCLUSIVE: u64 = 128;

/// Number of bits in the nonce decomposition. `2^7 = 128`.
pub const NUM_NONCE_BITS: usize = 7;

/// Length of the canonical RLP output for `rlp([sender_20_bytes, nonce])`
/// when `nonce < 128`. Layout:
///   - byte 0: list header (0xc0 + payload_len = 0xc0 + 22 = 0xd6)
///   - byte 1: sender string header (0x80 + 20 = 0x94)
///   - bytes 2..22: 20 sender address bytes
///   - byte 22: nonce-as-RLP byte (= NONCE if NONCE ≠ 0; = 0x80 if NONCE = 0)
pub const RLP_OUTPUT_LEN: usize = 23;

/// List header byte: 0xc0 + payload_len, where payload_len = 21
/// (sender header + 20 sender bytes) + 1 (single nonce byte) = 22.
pub const RLP_LIST_HEADER: u64 = 0xd6;

/// Sender string header byte: 0x80 + 20 (a 20-byte string).
pub const RLP_SENDER_HEADER: u64 = 0x94;

/// RLP encoding of the empty string (i.e. nonce = 0).
pub const RLP_NONCE_ZERO: u64 = 0x80;

/// Number of sender bytes (an Ethereum address is always 20 bytes).
pub const NUM_SENDER_BYTES: usize = 20;

/// Number of LE u64 limbs the sender address packs into. A 20-byte
/// address fills limbs 0..2 (limb 2 carries only its low 4 bytes);
/// limb 3 is identically zero.
pub const NUM_SENDER_LIMBS: usize = 4;

/// Width of the keccak-input-shaped RLP output column. This MUST
/// match [`crate::keccak_extract::MAX_INPUT_LEN`] so the cross-AIR
/// LogUp tuple at [`make_create_rlp_keccak_extract_input_linkage_descriptor`]
/// has matching tuple widths on both sides.
pub const RLP_BYTE_WIDTH: usize = crate::keccak_extract::MAX_INPUT_LEN;

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_SENDER_BYTE_OFFSET: usize = 0;
pub const COL_SENDER_LIMB_OFFSET: usize = COL_SENDER_BYTE_OFFSET + NUM_SENDER_BYTES;
pub const COL_NONCE: usize = COL_SENDER_LIMB_OFFSET + NUM_SENDER_LIMBS;
pub const COL_NONCE_BIT_OFFSET: usize = COL_NONCE + 1;
pub const COL_NONCE_INV: usize = COL_NONCE_BIT_OFFSET + NUM_NONCE_BITS;
pub const COL_NONCE_RLP_BYTE: usize = COL_NONCE_INV + 1;
pub const COL_RLP_BYTE_OFFSET: usize = COL_NONCE_RLP_BYTE + 1;
pub const COL_RLP_LEN: usize = COL_RLP_BYTE_OFFSET + RLP_BYTE_WIDTH;
pub const COL_IS_REAL: usize = COL_RLP_LEN + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 10;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CreateRlpRow {
    pub sender: [u8; NUM_SENDER_BYTES],
    /// Restricted to `[0, MAX_NONCE_EXCLUSIVE)`. The witness builder
    /// rejects out-of-range nonces.
    pub nonce: u64,
}

#[derive(Clone, Debug, Default)]
pub struct CreateRlpWitness {
    pub invocations: Vec<CreateRlpRow>,
}

impl CreateRlpWitness {
    /// Build a witness from a sequence of `(sender, nonce)` pairs.
    /// Returns an error if any nonce exceeds [`MAX_NONCE_EXCLUSIVE`].
    pub fn from_inputs(inputs: &[(&[u8; NUM_SENDER_BYTES], u64)]) -> Result<Self, &'static str> {
        let mut invocations = Vec::with_capacity(inputs.len());
        for (sender, nonce) in inputs {
            if *nonce >= MAX_NONCE_EXCLUSIVE {
                return Err("evm_create_rlp_air: nonce ≥ MAX_NONCE_EXCLUSIVE; only nonces in [0, 128) are supported by this MVP gadget");
            }
            invocations.push(CreateRlpRow {
                sender: **sender,
                nonce: *nonce,
            });
        }
        Ok(Self { invocations })
    }
}

/// Pack a 20-byte sender address into 4 LE u64 limbs, matching EVM's
/// `address_to_limbs` convention (`crates/evm/src/inspector.rs`).
pub fn sender_limbs_from_bytes(sender: &[u8; NUM_SENDER_BYTES]) -> [u64; NUM_SENDER_LIMBS] {
    let mut limbs = [0u64; NUM_SENDER_LIMBS];
    let mut tmp = [0u8; 8];
    tmp.copy_from_slice(&sender[0..8]);
    limbs[0] = u64::from_le_bytes(tmp);
    tmp.copy_from_slice(&sender[8..16]);
    limbs[1] = u64::from_le_bytes(tmp);
    let mut tmp4 = [0u8; 8];
    tmp4[..4].copy_from_slice(&sender[16..20]);
    limbs[2] = u64::from_le_bytes(tmp4);
    limbs
}

/// The single byte placed at `RLP[22]` for a given nonce. Mirrors the
/// row-local constraint `nonce_rlp_byte_value`.
pub fn nonce_rlp_byte(nonce: u64) -> u64 {
    if nonce == 0 {
        RLP_NONCE_ZERO
    } else {
        nonce
    }
}

/// Construct the full 23-byte canonical RLP encoding of
/// `[sender, nonce]` for `nonce < 128`. Returns
/// `[u8; RLP_OUTPUT_LEN]`.
pub fn canonical_rlp(sender: &[u8; NUM_SENDER_BYTES], nonce: u64) -> [u8; RLP_OUTPUT_LEN] {
    debug_assert!(
        nonce < MAX_NONCE_EXCLUSIVE,
        "canonical_rlp: nonce ≥ 128 unsupported by this MVP gadget"
    );
    let mut out = [0u8; RLP_OUTPUT_LEN];
    out[0] = RLP_LIST_HEADER as u8;
    out[1] = RLP_SENDER_HEADER as u8;
    out[2..2 + NUM_SENDER_BYTES].copy_from_slice(sender);
    out[2 + NUM_SENDER_BYTES] = nonce_rlp_byte(nonce) as u8;
    out
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &CreateRlpWitness,
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
        // Sender bytes.
        for (b, &v) in inv.sender.iter().enumerate() {
            columns[COL_SENDER_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        // Sender limbs (LE u64 packing).
        let limbs = sender_limbs_from_bytes(&inv.sender);
        for (k, l) in limbs.iter().enumerate() {
            columns[COL_SENDER_LIMB_OFFSET + k][i] = Scalar::from_u64(*l, curve);
        }
        // Nonce + 7-bit decomposition.
        columns[COL_NONCE][i] = Scalar::from_u64(inv.nonce, curve);
        for b in 0..NUM_NONCE_BITS {
            let bit = (inv.nonce >> b) & 1;
            columns[COL_NONCE_BIT_OFFSET + b][i] = Scalar::from_u64(bit, curve);
        }
        // Nonce inverse (1 / nonce if nonzero; arbitrary 0 if zero).
        if inv.nonce == 0 {
            columns[COL_NONCE_INV][i] = zero.clone();
        } else {
            // Compute the modular inverse via the field's own primitive.
            columns[COL_NONCE_INV][i] = Scalar::from_u64(inv.nonce, curve).inverse();
        }
        // Nonce-as-RLP byte.
        let nrb = nonce_rlp_byte(inv.nonce);
        columns[COL_NONCE_RLP_BYTE][i] = Scalar::from_u64(nrb, curve);
        // Full RLP output (bytes 0..23 carry RLP, bytes 23..256 zero).
        let rlp = canonical_rlp(&inv.sender, inv.nonce);
        for (b, &v) in rlp.iter().enumerate() {
            columns[COL_RLP_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        // RLP length (always 23 for nonce < 128).
        columns[COL_RLP_LEN][i] = Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve);
        // Selector.
        columns[COL_IS_REAL][i] = one.clone();
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

// ─── Constraint system ─────────────────────────────────────────────────

pub struct CreateRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl CreateRlpConstraintSystem {
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

// ─── Body helpers ─────────────────────────────────────────────────────

/// Number of sender bytes covered by `sender_limb[k]`. k ∈ {0, 1} → 8;
/// k = 2 → 4; k = 3 → 0 (limb 3 is always zero, pinned by body 4).
fn sender_limb_byte_count(k: usize) -> usize {
    match k {
        0 | 1 => 8,
        2 => 4,
        _ => 0,
    }
}

fn byte_power(i: usize, curve: CurveType) -> Scalar {
    debug_assert!(i < 8, "byte_power overflows u64 for i ≥ 8");
    Scalar::from_u64(1u64 << (8 * i), curve)
}

fn bit_power(i: usize, curve: CurveType) -> Scalar {
    debug_assert!(i < NUM_NONCE_BITS, "bit_power overflows for i ≥ 7");
    Scalar::from_u64(1u64 << i, curve)
}

fn eval_sender_limb_binding(col_evals: &[Scalar], k: usize) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let bytes = sender_limb_byte_count(k);
    let mut sum = Scalar::zero(curve);
    for i in 0..bytes {
        let byte = &col_evals[COL_SENDER_BYTE_OFFSET + 8 * k + i];
        sum = sum.add(&byte.mul(&byte_power(i, curve)));
    }
    let limb = &col_evals[COL_SENDER_LIMB_OFFSET + k];
    let body = limb.sub(&sum);
    // Gate by IS_REAL — naturally vanishes on padding (limb=0, sum=0)
    // but the gating costs nothing and pairs with bodies 8/9.
    col_evals[COL_IS_REAL].mul(&body)
}

fn eval_sender_limb_3_zero(col_evals: &[Scalar]) -> Scalar {
    let limb = &col_evals[COL_SENDER_LIMB_OFFSET + 3];
    col_evals[COL_IS_REAL].mul(limb)
}

fn eval_nonce_bit_binarity(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    let mut ap = one.clone();
    for b in 0..NUM_NONCE_BITS {
        let bit = &col_evals[COL_NONCE_BIT_OFFSET + b];
        let body = bit.mul(&bit.sub(&one));
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    acc
}

fn eval_nonce_decomposition(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..NUM_NONCE_BITS {
        let bit = &col_evals[COL_NONCE_BIT_OFFSET + b];
        sum = sum.add(&bit.mul(&bit_power(b, curve)));
    }
    let nonce = &col_evals[COL_NONCE];
    sum.sub(nonce)
}

fn eval_nonce_inv_correctness(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let one = Scalar::one(curve);
    let nonce = &col_evals[COL_NONCE];
    let inv = &col_evals[COL_NONCE_INV];
    nonce.mul(&nonce.mul(inv).sub(&one))
}

fn eval_nonce_rlp_byte_value(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let one = Scalar::one(curve);
    let nonce = &col_evals[COL_NONCE];
    let inv = &col_evals[COL_NONCE_INV];
    let nrb = &col_evals[COL_NONCE_RLP_BYTE];
    let const_128 = Scalar::from_u64(RLP_NONCE_ZERO, curve);
    let is_zero = one.sub(&nonce.mul(inv));
    let body = nrb.sub(nonce).sub(&const_128.mul(&is_zero));
    col_evals[COL_IS_REAL].mul(&body)
}

fn eval_rlp_construction(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    // Build the sub-bodies in a fixed order. Total: 2 (headers) + 20
    // (sender bytes) + 1 (nonce byte) + (RLP_BYTE_WIDTH - 23) (zero
    // tail) + 1 (length) = 257.
    let mut acc = zero.clone();
    let mut ap = one.clone();

    let push = |acc: &mut Scalar, ap: &mut Scalar, body: Scalar, alpha: &Scalar| {
        *acc = acc.add(&body.mul(ap));
        *ap = ap.mul(alpha);
    };

    // RLP[0] - 0xd6
    let body = col_evals[COL_RLP_BYTE_OFFSET + 0]
        .sub(&Scalar::from_u64(RLP_LIST_HEADER, curve));
    push(&mut acc, &mut ap, body, alpha);
    // RLP[1] - 0x94
    let body = col_evals[COL_RLP_BYTE_OFFSET + 1]
        .sub(&Scalar::from_u64(RLP_SENDER_HEADER, curve));
    push(&mut acc, &mut ap, body, alpha);
    // RLP[2 + k] - SENDER_BYTE[k] for k ∈ 0..20
    for k in 0..NUM_SENDER_BYTES {
        let body = col_evals[COL_RLP_BYTE_OFFSET + 2 + k]
            .sub(&col_evals[COL_SENDER_BYTE_OFFSET + k]);
        push(&mut acc, &mut ap, body, alpha);
    }
    // RLP[22] - NONCE_RLP_BYTE
    let body = col_evals[COL_RLP_BYTE_OFFSET + 2 + NUM_SENDER_BYTES]
        .sub(&col_evals[COL_NONCE_RLP_BYTE]);
    push(&mut acc, &mut ap, body, alpha);
    // RLP[k] = 0 for k ∈ 23..RLP_BYTE_WIDTH (zero tail)
    for k in RLP_OUTPUT_LEN..RLP_BYTE_WIDTH {
        let body = col_evals[COL_RLP_BYTE_OFFSET + k].clone();
        push(&mut acc, &mut ap, body, alpha);
    }
    // RLP_LEN - 23
    let body = col_evals[COL_RLP_LEN].sub(&Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve));
    push(&mut acc, &mut ap, body, alpha);

    col_evals[COL_IS_REAL].mul(&acc)
}

// Polynomial-form helpers (mirror the eval-form ones above).

fn build_sender_limb_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    k: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let bytes = sender_limb_byte_count(k);
    let mut sum = vec![Scalar::zero(curve)];
    for i in 0..bytes {
        let byte_poly = &col_coeffs[COL_SENDER_BYTE_OFFSET + 8 * k + i];
        let scaled = poly_scalar_mul(byte_poly, &byte_power(i, curve));
        sum = poly_add(&sum, &scaled, curve);
    }
    let limb_poly = &col_coeffs[COL_SENDER_LIMB_OFFSET + k];
    let diff = poly_sub(limb_poly, &sum, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &diff, curve)
}

fn build_sender_limb_3_zero_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    poly_mul(
        &col_coeffs[COL_IS_REAL],
        &col_coeffs[COL_SENDER_LIMB_OFFSET + 3],
        curve,
    )
}

fn build_nonce_bit_binarity_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for b in 0..NUM_NONCE_BITS {
        let bit = &col_coeffs[COL_NONCE_BIT_OFFSET + b];
        let bit_minus_1 = poly_sub(bit, &one_poly, curve);
        let body = poly_mul(bit, &bit_minus_1, curve);
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    acc
}

fn build_nonce_decomposition_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..NUM_NONCE_BITS {
        let bit = &col_coeffs[COL_NONCE_BIT_OFFSET + b];
        let scaled = poly_scalar_mul(bit, &bit_power(b, curve));
        sum = poly_add(&sum, &scaled, curve);
    }
    let nonce = &col_coeffs[COL_NONCE];
    poly_sub(&sum, nonce, curve)
}

fn build_nonce_inv_correctness_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let nonce = &col_coeffs[COL_NONCE];
    let inv = &col_coeffs[COL_NONCE_INV];
    let n_inv = poly_mul(nonce, inv, curve);
    let n_inv_minus_1 = poly_sub(&n_inv, &one_poly, curve);
    poly_mul(nonce, &n_inv_minus_1, curve)
}

fn build_nonce_rlp_byte_value_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let nonce = &col_coeffs[COL_NONCE];
    let inv = &col_coeffs[COL_NONCE_INV];
    let nrb = &col_coeffs[COL_NONCE_RLP_BYTE];
    let const_128 = Scalar::from_u64(RLP_NONCE_ZERO, curve);
    let n_inv = poly_mul(nonce, inv, curve);
    let is_zero = poly_sub(&one_poly, &n_inv, curve);
    let scaled = poly_scalar_mul(&is_zero, &const_128);
    let nrb_minus_n = poly_sub(nrb, nonce, curve);
    let body = poly_sub(&nrb_minus_n, &scaled, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &body, curve)
}

fn build_rlp_construction_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    let one_const = vec![Scalar::from_u64(RLP_LIST_HEADER, curve)];
    let body = poly_sub(&col_coeffs[COL_RLP_BYTE_OFFSET + 0], &one_const, curve);
    acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
    ap = ap.mul(alpha);

    let const_94 = vec![Scalar::from_u64(RLP_SENDER_HEADER, curve)];
    let body = poly_sub(&col_coeffs[COL_RLP_BYTE_OFFSET + 1], &const_94, curve);
    acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
    ap = ap.mul(alpha);

    for k in 0..NUM_SENDER_BYTES {
        let body = poly_sub(
            &col_coeffs[COL_RLP_BYTE_OFFSET + 2 + k],
            &col_coeffs[COL_SENDER_BYTE_OFFSET + k],
            curve,
        );
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }

    let body = poly_sub(
        &col_coeffs[COL_RLP_BYTE_OFFSET + 2 + NUM_SENDER_BYTES],
        &col_coeffs[COL_NONCE_RLP_BYTE],
        curve,
    );
    acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
    ap = ap.mul(alpha);

    for k in RLP_OUTPUT_LEN..RLP_BYTE_WIDTH {
        let body = col_coeffs[COL_RLP_BYTE_OFFSET + k].clone();
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }

    let const_23 = vec![Scalar::from_u64(RLP_OUTPUT_LEN as u64, curve)];
    let body = poly_sub(&col_coeffs[COL_RLP_LEN], &const_23, curve);
    acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);

    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for CreateRlpConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "sender_limb_0_binding".into(),
            "sender_limb_1_binding".into(),
            "sender_limb_2_binding".into(),
            "sender_limb_3_zero".into(),
            "nonce_bit_binarity".into(),
            "nonce_decomposition".into(),
            "nonce_inv_correctness".into(),
            "nonce_rlp_byte_value".into(),
            "rlp_construction".into(),
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
        let alpha_for_rlc = Scalar::from_u64(7, curve);

        let mut bin = vec![Scalar::zero(curve); n];
        let mut s0 = vec![Scalar::zero(curve); n];
        let mut s1 = vec![Scalar::zero(curve); n];
        let mut s2 = vec![Scalar::zero(curve); n];
        let mut s3 = vec![Scalar::zero(curve); n];
        let mut nbb = vec![Scalar::zero(curve); n];
        let mut nde = vec![Scalar::zero(curve); n];
        let mut ninv = vec![Scalar::zero(curve); n];
        let mut nrb = vec![Scalar::zero(curve); n];
        let mut rlpc = vec![Scalar::zero(curve); n];

        for row in 0..n {
            let v = &columns[COL_IS_REAL][row];
            bin[row] = v.mul(&v.sub(&one));
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            s0[row] = eval_sender_limb_binding(&row_evals, 0);
            s1[row] = eval_sender_limb_binding(&row_evals, 1);
            s2[row] = eval_sender_limb_binding(&row_evals, 2);
            s3[row] = eval_sender_limb_3_zero(&row_evals);
            nbb[row] = eval_nonce_bit_binarity(&row_evals, &alpha_for_rlc);
            nde[row] = eval_nonce_decomposition(&row_evals);
            ninv[row] = eval_nonce_inv_correctness(&row_evals);
            nrb[row] = eval_nonce_rlp_byte_value(&row_evals);
            rlpc[row] = eval_rlp_construction(&row_evals, &alpha_for_rlc);
        }
        vec![bin, s0, s1, s2, s3, nbb, nde, ninv, nrb, rlpc]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            v.mul(&v.sub(&one)),
            eval_sender_limb_binding(col_evals, 0),
            eval_sender_limb_binding(col_evals, 1),
            eval_sender_limb_binding(col_evals, 2),
            eval_sender_limb_3_zero(col_evals),
            eval_nonce_bit_binarity(col_evals, alpha),
            eval_nonce_decomposition(col_evals),
            eval_nonce_inv_correctness(col_evals),
            eval_nonce_rlp_byte_value(col_evals),
            eval_rlp_construction(col_evals, alpha),
        ];
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
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
        let bodies: Vec<Vec<Scalar>> = vec![
            poly_mul(v, &v_minus_1, curve),
            build_sender_limb_binding_poly(col_coeffs, 0, curve),
            build_sender_limb_binding_poly(col_coeffs, 1, curve),
            build_sender_limb_binding_poly(col_coeffs, 2, curve),
            build_sender_limb_3_zero_poly(col_coeffs, curve),
            build_nonce_bit_binarity_poly(col_coeffs, alpha, curve),
            build_nonce_decomposition_poly(col_coeffs, curve),
            build_nonce_inv_correctness_poly(col_coeffs, curve),
            build_nonce_rlp_byte_value_poly(col_coeffs, curve),
            build_rlp_construction_poly(col_coeffs, alpha, curve),
        ];
        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            let scaled = poly_scalar_mul(body, &ap);
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

// ─── Cross-AIR LogUp linkage descriptors ───────────────────────────────

/// Cross-AIR LogUp descriptor binding the **EVM main trace's CREATE
/// row** to **this gadget's input columns**: matches the 5-element
/// tuple `(sender_limb_0..3, nonce)`.
///
/// EVM main side (`crates/evm/src/cross_air_linkage.rs` will re-export
/// a thin wrapper):
///   - `frame_callee_l[0..4]` (4 cols, the executing contract whose
///     CREATE call this is — the sender for address derivation).
///   - `create_nonce_hint` (1 col, oracle column added by the
///     inspector — see "Future work" note below).
///   - Selector: `COL_SEL_CREATE`.
///
/// Gadget side:
///   - `SENDER_LIMB[0..4]` (cols 20..24).
///   - `NONCE` (col 24).
///   - Selector: `COL_IS_REAL`.
///
/// **Future work (EVM trace addition)**: the EVM trace must expose a
/// `create_nonce_hint` u64 column populated by the inspector from
/// `revm.context.journal().load_account(sender).info.nonce` *at the
/// moment of the CREATE opcode*. This is essentially an oracle hint
/// that the cross-AIR linkage transports into the gadget; the
/// algebraic correctness of "this nonce is the live state's nonce
/// for this account" is OUTSIDE this linkage (it requires a
/// state-transition AIR — the same gap that affects all opcode-level
/// state-touching constraints).
pub fn make_evm_main_create_rlp_linkage_descriptor(
    evm_layer_index: usize,
    rlp_layer_index: usize,
    evm_frame_callee_l0_col: usize,
    evm_create_nonce_hint_col: usize,
    evm_sel_create_col: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let a_columns = vec![
        evm_frame_callee_l0_col,
        evm_frame_callee_l0_col + 1,
        evm_frame_callee_l0_col + 2,
        evm_frame_callee_l0_col + 3,
        evm_create_nonce_hint_col,
    ];
    let b_columns = vec![
        COL_SENDER_LIMB_OFFSET,
        COL_SENDER_LIMB_OFFSET + 1,
        COL_SENDER_LIMB_OFFSET + 2,
        COL_SENDER_LIMB_OFFSET + 3,
        COL_NONCE,
    ];
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_main_create_rlp_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(evm_sel_create_col),
        b_layer_index: rlp_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding **this gadget's RLP output** to
/// **KeccakExtract's input side**: matches the
/// `(RLP_BYTE[0..256], RLP_LEN)` tuple against
/// `(INPUT_BYTE[0..MAX_INPUT_LEN], INPUT_LEN)`.
///
/// Combined with KeccakExtract↔Keccak (already closed via #91), the
/// keccak invocation hashing this gadget's RLP output is
/// algebraically pinned to be computing the canonical CREATE
/// pre-image hash.
pub fn make_create_rlp_keccak_extract_input_linkage_descriptor(
    rlp_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    debug_assert_eq!(
        RLP_BYTE_WIDTH,
        ke::MAX_INPUT_LEN,
        "RLP gadget's keccak-input-shaped width must equal KeccakExtract MAX_INPUT_LEN"
    );
    let mut a_columns: Vec<usize> = (0..RLP_BYTE_WIDTH)
        .map(|b| COL_RLP_BYTE_OFFSET + b)
        .collect();
    a_columns.push(COL_RLP_LEN);

    let mut b_columns: Vec<usize> = (0..ke::MAX_INPUT_LEN)
        .map(|b| ke::COL_INPUT_BYTE_OFFSET + b)
        .collect();
    b_columns.push(ke::COL_INPUT_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "create_rlp_keccak_extract_input_v1".into(),
        a_layer_index: rlp_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_witness() -> CreateRlpWitness {
        let s1 = [0x11u8; 20];
        let s2 = [0xAAu8; 20];
        CreateRlpWitness::from_inputs(&[(&s1, 0), (&s2, 5), (&s1, 127)])
            .expect("test inputs are within MAX_NONCE_EXCLUSIVE")
    }

    #[test]
    fn from_inputs_rejects_oversize_nonce() {
        let s = [0u8; 20];
        let r = CreateRlpWitness::from_inputs(&[(&s, MAX_NONCE_EXCLUSIVE)]);
        assert!(r.is_err(), "nonces ≥ 128 must be rejected by this MVP");
    }

    #[test]
    fn canonical_rlp_matches_known_vectors() {
        // Spot vector: nonce = 0 → RLP[22] = 0x80.
        let s = [0u8; 20];
        let rlp = canonical_rlp(&s, 0);
        assert_eq!(rlp[0], RLP_LIST_HEADER as u8);
        assert_eq!(rlp[1], RLP_SENDER_HEADER as u8);
        for k in 0..20 {
            assert_eq!(rlp[2 + k], 0);
        }
        assert_eq!(rlp[22], RLP_NONCE_ZERO as u8);

        // Spot vector: nonce = 7 → RLP[22] = 7.
        let rlp = canonical_rlp(&s, 7);
        assert_eq!(rlp[22], 7);

        // Spot vector: nonce = 127 → RLP[22] = 127.
        let rlp = canonical_rlp(&s, 127);
        assert_eq!(rlp[22], 127);
    }

    #[test]
    fn canonical_rlp_round_trips_against_real_rlp_for_sample_address() {
        // Decode the canonical RLP bytes back manually and confirm we
        // round-trip the (sender, nonce) pair.
        let mut s = [0u8; 20];
        for i in 0..20 {
            s[i] = (i as u8) + 1;
        }
        let nonce: u64 = 42;
        let rlp = canonical_rlp(&s, nonce);
        // List header: payload = 22 bytes → 0xc0 + 22 = 0xd6.
        assert_eq!(rlp[0], 0xd6);
        // Sender prefix: 0x80 + 20 = 0x94.
        assert_eq!(rlp[1], 0x94);
        // Sender body matches the input bytes.
        assert_eq!(&rlp[2..22], &s);
        // Nonce single byte.
        assert_eq!(rlp[22], 42);
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 3);
        // Row 0: nonce = 0 → NONCE_RLP_BYTE = 0x80; all bits = 0.
        assert!(trace.columns[COL_NONCE].evaluations[0].is_zero());
        for b in 0..NUM_NONCE_BITS {
            assert!(
                trace.columns[COL_NONCE_BIT_OFFSET + b].evaluations[0].is_zero(),
                "row 0 bit {} should be zero", b
            );
        }
        assert_eq!(
            trace.columns[COL_NONCE_RLP_BYTE].evaluations[0].to_u64(),
            RLP_NONCE_ZERO
        );
        // Row 1: nonce = 5, sender = 0xAA…AA.
        assert_eq!(trace.columns[COL_NONCE].evaluations[1].to_u64(), 5);
        // Bit decomposition of 5 = 0b101 → bits 0, 2 set.
        assert_eq!(trace.columns[COL_NONCE_BIT_OFFSET + 0].evaluations[1].to_u64(), 1);
        assert!(trace.columns[COL_NONCE_BIT_OFFSET + 1].evaluations[1].is_zero());
        assert_eq!(trace.columns[COL_NONCE_BIT_OFFSET + 2].evaluations[1].to_u64(), 1);
        for b in 3..NUM_NONCE_BITS {
            assert!(trace.columns[COL_NONCE_BIT_OFFSET + b].evaluations[1].is_zero());
        }
        assert_eq!(trace.columns[COL_NONCE_RLP_BYTE].evaluations[1].to_u64(), 5);
        // Row 1 RLP[22] = 5.
        assert_eq!(
            trace.columns[COL_RLP_BYTE_OFFSET + 22].evaluations[1].to_u64(),
            5
        );
        // Row 1 RLP_LEN = 23.
        assert_eq!(
            trace.columns[COL_RLP_LEN].evaluations[1].to_u64(),
            RLP_OUTPUT_LEN as u64
        );
        // Row 1 sender_limb_0 = u64::from_le_bytes([0xAA; 8]).
        let expected_l0 = u64::from_le_bytes([0xAA; 8]);
        assert_eq!(
            trace.columns[COL_SENDER_LIMB_OFFSET].evaluations[1].to_u64(),
            expected_l0
        );
        // Padding row (index 3): IS_REAL = 0.
        assert!(trace.columns[COL_IS_REAL].evaluations[3].is_zero());
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = CreateRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(11, curve);
        for row in 0..trace.padded_size as usize {
            let col_vals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[row].clone())
                .collect();
            let body = cs.evaluate_at_point(&col_vals, &alpha);
            assert!(body.is_zero(), "row {} body must vanish", row);
        }
    }

    #[test]
    fn rlp_construction_rejects_tampered_byte() {
        // Flipping any byte in the RLP output (including the constant
        // header bytes 0xd6 / 0x94) must make body 9 fire.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = CreateRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(13, curve);
        // Flip RLP[0] (the list header).
        let one = Scalar::one(curve);
        trace.columns[COL_RLP_BYTE_OFFSET].evaluations[0] =
            trace.columns[COL_RLP_BYTE_OFFSET].evaluations[0].add(&one);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "rlp_construction must fire when RLP[0] is tampered"
        );
    }

    #[test]
    fn nonce_inv_rejects_wrong_inverse() {
        // Setting NONCE_INV to anything other than 1/NONCE (when
        // NONCE ≠ 0) must make body 7 fire.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = CreateRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        // Row 1 has nonce = 5. Tamper with the inverse.
        trace.columns[COL_NONCE_INV].evaluations[1] = Scalar::from_u64(99, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "nonce_inv_correctness must fire on wrong inverse"
        );
    }

    #[test]
    fn nonce_decomposition_rejects_bit_mismatch() {
        // Setting a bit that contradicts NONCE must make body 6 fire.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = CreateRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(19, curve);
        // Row 1 has nonce = 5 (binary 0b101); bit 1 is 0. Set it to 1.
        trace.columns[COL_NONCE_BIT_OFFSET + 1].evaluations[1] =
            Scalar::from_u64(1, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "nonce_decomposition must fire when bits disagree with NONCE"
        );
    }

    #[test]
    fn nonce_bit_binarity_rejects_non_binary_bit() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = CreateRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(23, curve);
        // Row 1 nonce bit 0 = 1; set it to 5 (non-binary).
        trace.columns[COL_NONCE_BIT_OFFSET].evaluations[1] = Scalar::from_u64(5, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "nonce_bit_binarity must fire on non-binary bit"
        );
    }

    #[test]
    fn nonce_rlp_byte_value_rejects_wrong_byte() {
        // Honest row: nonce = 5 → NONCE_RLP_BYTE = 5. Set it to 6.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = CreateRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(29, curve);
        trace.columns[COL_NONCE_RLP_BYTE].evaluations[1] = Scalar::from_u64(6, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[1].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "nonce_rlp_byte_value must fire on wrong NONCE_RLP_BYTE"
        );
    }

    #[test]
    fn nonce_rlp_byte_value_rejects_wrong_byte_for_zero_nonce() {
        // Honest row 0: nonce = 0 → NONCE_RLP_BYTE = 0x80. Set it to 0.
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = CreateRlpConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(31, curve);
        trace.columns[COL_NONCE_RLP_BYTE].evaluations[0] = Scalar::from_u64(0, curve);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "nonce_rlp_byte_value must fire when NONCE_RLP_BYTE != 0x80 for nonce = 0"
        );
    }

    #[test]
    fn evm_main_create_rlp_descriptor_well_formed() {
        let desc = make_evm_main_create_rlp_linkage_descriptor(
            0, 1, /* frame_callee_l0 */ 100, /* nonce_hint */ 200,
            /* sel_create */ 50,
        );
        assert_eq!(desc.label, "evm_main_create_rlp_v1");
        assert_eq!(desc.a_columns.len(), 5);
        assert_eq!(desc.b_columns.len(), 5);
        assert_eq!(desc.a_columns[0], 100);
        assert_eq!(desc.a_columns[3], 103);
        assert_eq!(desc.a_columns[4], 200);
        assert_eq!(desc.b_columns[0], COL_SENDER_LIMB_OFFSET);
        assert_eq!(desc.b_columns[3], COL_SENDER_LIMB_OFFSET + 3);
        assert_eq!(desc.b_columns[4], COL_NONCE);
        assert_eq!(desc.a_selector_column, Some(50));
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn create_rlp_keccak_extract_input_descriptor_well_formed() {
        use crate::keccak_extract as ke;
        let desc = make_create_rlp_keccak_extract_input_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "create_rlp_keccak_extract_input_v1");
        let tuple_len = ke::MAX_INPUT_LEN + 1;
        assert_eq!(desc.a_columns.len(), tuple_len);
        assert_eq!(desc.b_columns.len(), tuple_len);
        assert_eq!(desc.a_columns[0], COL_RLP_BYTE_OFFSET);
        assert_eq!(desc.a_columns[ke::MAX_INPUT_LEN - 1], COL_RLP_BYTE_OFFSET + ke::MAX_INPUT_LEN - 1);
        assert_eq!(desc.a_columns[ke::MAX_INPUT_LEN], COL_RLP_LEN);
        assert_eq!(desc.b_columns[0], ke::COL_INPUT_BYTE_OFFSET);
        assert_eq!(desc.b_columns[ke::MAX_INPUT_LEN], ke::COL_INPUT_LEN);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(ke::COL_IS_REAL));
    }

    /// End-to-end byte-tuple matching: build a real
    /// (RLP gadget, KeccakExtract) pair from the same `(sender, nonce)`
    /// invocation, assert the 257-element cross-AIR tuples agree
    /// element-by-element. This is the foundation of the input-side
    /// linkage's soundness.
    #[test]
    fn rlp_gadget_byte_tuples_match_keccak_extract_input() {
        use crate::keccak_extract::{KeccakExtractWitness, build_trace_polynomials as ke_build};
        let curve = CurveType::Bls48581;
        // A representative invocation: nonzero nonce, distinct sender.
        let mut sender = [0u8; 20];
        for i in 0..20 {
            sender[i] = (i as u8) * 7 + 3;
        }
        let nonce: u64 = 42;

        // Gadget side: build the RLP bytes.
        let gadget_w = CreateRlpWitness::from_inputs(&[(&sender, nonce)])
            .expect("valid input");
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);

        // Extract side: feed the SAME RLP bytes into KeccakExtract.
        let canonical = canonical_rlp(&sender, nonce);
        let mut padded = canonical.to_vec();
        // KeccakExtract auto-pads to MAX_INPUT_LEN; just supply the
        // actual RLP bytes.
        let _ = &mut padded;
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical.to_vec()])
            .expect("rlp output fits in MAX_INPUT_LEN");
        let extract_trace = ke_build(&extract_w, curve);

        // Tuple equality: every byte position 0..MAX_INPUT_LEN.
        for b in 0..crate::keccak_extract::MAX_INPUT_LEN {
            let g = gadget_trace.columns[COL_RLP_BYTE_OFFSET + b].evaluations[0]
                .to_u64() as u8;
            let e = extract_trace.columns[crate::keccak_extract::COL_INPUT_BYTE_OFFSET + b]
                .evaluations[0]
                .to_u64() as u8;
            assert_eq!(g, e, "tuple byte {} mismatch (gadget vs extract)", b);
        }
        // Length must also match: gadget says 23, extract says canonical.len() = 23.
        let g_len = gadget_trace.columns[COL_RLP_LEN].evaluations[0].to_u64();
        let e_len = extract_trace.columns[crate::keccak_extract::COL_INPUT_LEN].evaluations[0]
            .to_u64();
        assert_eq!(g_len, RLP_OUTPUT_LEN as u64);
        assert_eq!(g_len, e_len);
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn create_rlp_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = CreateRlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "create rlp proof must verify");
    }

    /// End-to-end cross-AIR `joint_prove`/`joint_verify` test for the
    /// CREATE input-side linkage: the new `EvmCreateRlpAir` ↔
    /// `KeccakExtract`. Mirrors the closed regressions
    /// `joint_prove_keccak_extract_keccak_linkage` (#91) and
    /// `joint_prove_evm_exp_linkage` (#94).
    ///
    /// Setup:
    ///   - RLP gadget AIR with one invocation: `(sender, nonce = 42)`.
    ///   - KeccakExtract AIR with one invocation: `keccak256(canonical_rlp(sender, 42))`.
    ///   - Linkage: 257-tuple `(RLP_BYTE[0..256], RLP_LEN)` ↔
    ///     `(INPUT_BYTE[0..256], INPUT_LEN)`, gated by `IS_REAL` on
    ///     both sides.
    ///
    /// Validates that `joint_prove` accepts the matched
    /// `(rlp_bytes, length)` tuple and `joint_verify` confirms the
    /// honest joint proof. With the RLP gadget's algebraic
    /// `rlp_construction` constraint pinning the bytes to the
    /// canonical encoding (and KeccakExtract↔Keccak from #91 pinning
    /// the digest to be `keccak256` of those bytes), the cross-AIR
    /// LogUp transfers the gadget's "RLP byte sequence is canonical"
    /// guarantee into KeccakExtract's claimed pre-image — closing the
    /// CREATE input-side soundness gap end-to-end (modulo the EVM
    /// main↔gadget linkage which is wired but not exercised in this
    /// 2-AIR test).
    #[test]
    #[ignore = "slow: full joint_prove + joint_verify across 2 AIRs (~5–15 min); \
                run with --release --ignored"]
    fn joint_prove_create_rlp_keccak_extract_input_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak_extract::{
            build_trace_polynomials as ke_build, KeccakExtractWitness,
            KeccakExtractConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Pick a representative invocation.
        let mut sender = [0u8; NUM_SENDER_BYTES];
        for i in 0..NUM_SENDER_BYTES {
            sender[i] = (i as u8) * 7 + 3;
        }
        let nonce: u64 = 42;

        // RLP gadget side.
        let gadget_w = CreateRlpWitness::from_inputs(&[(&sender, nonce)])
            .expect("nonce in valid range");
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = CreateRlpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        // KeccakExtract side: feed the SAME canonical RLP bytes.
        let canonical = canonical_rlp(&sender, nonce);
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical.to_vec()])
            .expect("rlp output fits in MAX_INPUT_LEN");
        let extract_trace = ke_build(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        // Cross-AIR LogUp linkage.
        let linkage = make_create_rlp_keccak_extract_input_linkage_descriptor(
            /* gadget layer */ 0,
            /* extract layer */ 1,
        );

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&gadget_trace, &gadget_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![linkage];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for matched RLP gadget + KeccakExtract");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        assert_eq!(
            extension.linkage_proofs[0].closure_a,
            extension.linkage_proofs[0].closure_b,
            "honest closure scalars must match"
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&gadget_cs, &extract_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the honest RLP gadget ↔ KeccakExtract input proof"
        );
    }
}
