//! CREATE2 pre-image concatenation gadget AIR.
//!
//! Per-invocation algebraic gadget that takes
//! `(sender_address, salt, initcode_hash)` and exposes the canonical
//! Ethereum CREATE2 pre-image
//! `0xff || sender_address || salt || keccak256(initcode)` (= 85
//! bytes) as a fixed-shape byte sequence ready for cross-AIR LogUp
//! matching against [`crate::keccak_extract`]'s `INPUT_BYTE` columns.
//!
//! Combined with the existing KeccakExtract↔Keccak binding (#91)
//! and an EVM main↔KeccakExtract output-side binding (the address
//! limbs of `keccak256(input)[12..32]`), this completes the
//! algebraic soundness chain for CREATE2 address derivation —
//! mirroring the now-closed CREATE chain
//! ([`crate::evm_create_rlp_air`]).
//!
//! # Scope
//!
//! - **Single CREATE2 invocation per row.** Multiple CREATE2 opcodes
//!   in one EVM trace need multiple gadget rows.
//! - **Fixed 85-byte output**: `0xff` (1) + sender (20) + salt (32) +
//!   initcode_hash (32). No variable-length encoding (unlike CREATE
//!   which has nonce-dependent length).
//! - **Sender = 20-byte Ethereum address**, exposed both as 20 raw
//!   byte columns (consumed by concatenation construction) and as 4
//!   LE u64 limbs (matches the EVM main trace's `frame_callee`
//!   encoding for the cross-AIR linkage).
//! - **Salt and initcode_hash** are 32-byte tuples, exposed as raw
//!   bytes only (no limb aggregation — they don't have a natural
//!   limb encoding in the EVM main trace; the cross-AIR linkage
//!   from EVM main passes them through dedicated oracle byte
//!   columns).
//!
//! # Column layout
//!
//! ```text
//!   0 .. 20    SENDER_BYTE[0..20]    raw 20-byte address
//!  20 .. 24    SENDER_LIMB[0..4]     LE u64 limbs aggregating SENDER_BYTE
//!  24 .. 56    SALT_BYTE[0..32]      raw 32-byte salt
//!  56 .. 88    INITCODE_HASH_BYTE[0..32] raw 32-byte keccak256(initcode)
//!  88 .. 344   INPUT_BYTE[0..256]    keccak-input-shaped output;
//!                                    bytes 0..85 carry the canonical
//!                                    pre-image, bytes 85..256 zero
//! 344          INPUT_LEN             always 85
//! 345          IS_REAL               selector
//! ```
//!
//! # Constraint layout
//!
//! 7 row-local constraints (alpha-power weighted):
//!
//!   0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `sender_limb_0_binding` — `IS_REAL ·
//!      (SENDER_LIMB[0] − Σ_{i=0..8} 2^(8 i) · SENDER_BYTE[i]) = 0`
//!   2. `sender_limb_1_binding` — `IS_REAL ·
//!      (SENDER_LIMB[1] − Σ_{i=0..8} 2^(8 i) · SENDER_BYTE[8 + i]) = 0`
//!   3. `sender_limb_2_binding` — `IS_REAL ·
//!      (SENDER_LIMB[2] − Σ_{i=0..4} 2^(8 i) · SENDER_BYTE[16 + i]) = 0`
//!   4. `sender_limb_3_zero` — `IS_REAL · SENDER_LIMB[3] = 0`
//!   5. `input_construction` — β-RLC over:
//!         - `INPUT_BYTE[0] − 0xff`
//!         - `INPUT_BYTE[1 + k] − SENDER_BYTE[k]` for `k ∈ 0..20`
//!         - `INPUT_BYTE[21 + k] − SALT_BYTE[k]` for `k ∈ 0..32`
//!         - `INPUT_BYTE[53 + k] − INITCODE_HASH_BYTE[k]` for `k ∈ 0..32`
//!         - `INPUT_BYTE[k]` for `k ∈ 85..256` (keccak-input zero tail)
//!         - `INPUT_LEN − 85`
//!      gated by IS_REAL — total of 86 + (256 - 85) + 1 = 258 sub-bodies.
//!
//! # Soundness chain (input side, CREATE2)
//!
//! Combined with the existing #91 closure (KeccakExtract↔Keccak):
//!
//! 1. **EVM main ↔ this gadget** (descriptor TBD): pins the gadget's
//!    `(SENDER_LIMB[0..4], SALT_BYTE[0..32], INITCODE_HASH_BYTE[0..32])`
//!    tuple equal to the EVM CREATE2 row's
//!    `(frame_callee, salt_hint, initcode_hash_hint)` — the latter
//!    two being NEW oracle columns the EVM trace must add (follow-up).
//!
//! 2. **This gadget's row-local constraints** pin
//!    `INPUT_BYTE[0..85] = 0xff || sender || salt || initcode_hash`
//!    algebraically (and `INPUT_BYTE[85..256] = 0`, `INPUT_LEN = 85`).
//!
//! 3. **This gadget ↔ KeccakExtract input side**
//!    ([`make_create2_input_keccak_extract_input_linkage_descriptor`]):
//!    pins the gadget's `(INPUT_BYTE[0..256], INPUT_LEN)` tuple to
//!    some KeccakExtract row's `(INPUT_BYTE[0..256], INPUT_LEN)`.
//!
//! 4. **KeccakExtract↔Keccak** (#91): pins
//!    `OUTPUT = keccak256(INPUT[0..len])`.
//!
//! 5. **EVM main ↔ KeccakExtract output side**: pins the EVM CREATE2
//!    row's `create_address_hint` to the LE limb-aggregation of
//!    `OUTPUT[12..32]` (already wired via
//!    `make_evm_create_address_keccak_extract_linkage_descriptor`
//!    with `selector = COL_SEL_CREATE2`).
//!
//! End to end: `create_address_hint = keccak256(0xff || sender || salt
//! || keccak256(initcode))[12..32]` algebraically.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// First byte of the CREATE2 pre-image — a fixed magic byte per EIP-1014.
pub const CREATE2_MAGIC_BYTE: u64 = 0xff;

/// Length of the canonical CREATE2 pre-image: 1 (magic) + 20 (sender) +
/// 32 (salt) + 32 (initcode_hash) = 85 bytes.
pub const INPUT_OUTPUT_LEN: usize = 85;

pub const NUM_SENDER_BYTES: usize = 20;
pub const NUM_SENDER_LIMBS: usize = 4;
pub const NUM_SALT_BYTES: usize = 32;
pub const NUM_INITCODE_HASH_BYTES: usize = 32;
/// Salt + initcode_hash are 32-byte values, packed into 4 LE u64 limbs
/// each. The limb columns are aggregators over the per-byte columns and
/// exist so the EVM-side cross-AIR LogUp linkage can match the EVM
/// trace's `create2_salt_hint` / `create2_initcode_hash_hint` limb
/// columns directly without flattening 32 byte columns.
pub const NUM_SALT_LIMBS: usize = 4;
pub const NUM_INITCODE_HASH_LIMBS: usize = 4;

/// Keccak-input-shaped output column width — must match
/// [`crate::keccak_extract::MAX_INPUT_LEN`] for the cross-AIR LogUp
/// tuple width to align.
pub const INPUT_BYTE_WIDTH: usize = crate::keccak_extract::MAX_INPUT_LEN;

// ─── Column indices ────────────────────────────────────────────────────

pub const COL_SENDER_BYTE_OFFSET: usize = 0;
pub const COL_SENDER_LIMB_OFFSET: usize = COL_SENDER_BYTE_OFFSET + NUM_SENDER_BYTES;
pub const COL_SALT_BYTE_OFFSET: usize = COL_SENDER_LIMB_OFFSET + NUM_SENDER_LIMBS;
pub const COL_SALT_LIMB_OFFSET: usize = COL_SALT_BYTE_OFFSET + NUM_SALT_BYTES;
pub const COL_INITCODE_HASH_BYTE_OFFSET: usize = COL_SALT_LIMB_OFFSET + NUM_SALT_LIMBS;
pub const COL_INITCODE_HASH_LIMB_OFFSET: usize =
    COL_INITCODE_HASH_BYTE_OFFSET + NUM_INITCODE_HASH_BYTES;
pub const COL_INPUT_BYTE_OFFSET: usize =
    COL_INITCODE_HASH_LIMB_OFFSET + NUM_INITCODE_HASH_LIMBS;
pub const COL_INPUT_LEN: usize = COL_INPUT_BYTE_OFFSET + INPUT_BYTE_WIDTH;
pub const COL_IS_REAL: usize = COL_INPUT_LEN + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// 1 (is_real binarity) + 3 (sender limb 0/1/2 binding) + 1 (sender limb 3 zero)
/// + 4 (salt limb bindings) + 4 (initcode_hash limb bindings) + 1 (input
/// construction RLC) = 14.
pub const NUM_ROW_CONSTRAINTS: usize = 14;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Create2InputRow {
    pub sender: [u8; NUM_SENDER_BYTES],
    pub salt: [u8; NUM_SALT_BYTES],
    pub initcode_hash: [u8; NUM_INITCODE_HASH_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct Create2InputWitness {
    pub invocations: Vec<Create2InputRow>,
}

impl Create2InputWitness {
    pub fn from_inputs(
        inputs: &[(
            &[u8; NUM_SENDER_BYTES],
            &[u8; NUM_SALT_BYTES],
            &[u8; NUM_INITCODE_HASH_BYTES],
        )],
    ) -> Self {
        let invocations = inputs
            .iter()
            .map(|(sender, salt, initcode_hash)| Create2InputRow {
                sender: **sender,
                salt: **salt,
                initcode_hash: **initcode_hash,
            })
            .collect();
        Self { invocations }
    }
}

/// LE u64 limb packing of a 20-byte sender address — matches EVM's
/// `address_to_limbs` (`crates/evm/src/inspector.rs`) and the
/// CREATE RLP gadget's `sender_limbs_from_bytes`.
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

/// LE u64 limb packing of a 32-byte value (salt or initcode_hash).
/// limb[i] = u64::from_le_bytes(bytes[8i..8i+8]). Matches the EVM
/// inspector's `b256_to_le_limbs` helper.
pub fn bytes32_to_le_limbs(bytes: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    let mut tmp = [0u8; 8];
    for i in 0..4 {
        tmp.copy_from_slice(&bytes[8 * i..8 * i + 8]);
        limbs[i] = u64::from_le_bytes(tmp);
    }
    limbs
}

/// Construct the canonical 85-byte CREATE2 pre-image
/// `0xff || sender || salt || initcode_hash`.
pub fn canonical_create2_input(
    sender: &[u8; NUM_SENDER_BYTES],
    salt: &[u8; NUM_SALT_BYTES],
    initcode_hash: &[u8; NUM_INITCODE_HASH_BYTES],
) -> [u8; INPUT_OUTPUT_LEN] {
    let mut out = [0u8; INPUT_OUTPUT_LEN];
    out[0] = CREATE2_MAGIC_BYTE as u8;
    out[1..1 + NUM_SENDER_BYTES].copy_from_slice(sender);
    out[1 + NUM_SENDER_BYTES..1 + NUM_SENDER_BYTES + NUM_SALT_BYTES].copy_from_slice(salt);
    out[1 + NUM_SENDER_BYTES + NUM_SALT_BYTES..]
        .copy_from_slice(initcode_hash);
    out
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Create2InputWitness,
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
        for (b, &v) in inv.sender.iter().enumerate() {
            columns[COL_SENDER_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        let limbs = sender_limbs_from_bytes(&inv.sender);
        for (k, l) in limbs.iter().enumerate() {
            columns[COL_SENDER_LIMB_OFFSET + k][i] = Scalar::from_u64(*l, curve);
        }
        for (b, &v) in inv.salt.iter().enumerate() {
            columns[COL_SALT_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        let salt_limbs = bytes32_to_le_limbs(&inv.salt);
        for (k, &l) in salt_limbs.iter().enumerate() {
            columns[COL_SALT_LIMB_OFFSET + k][i] = Scalar::from_u64(l, curve);
        }
        for (b, &v) in inv.initcode_hash.iter().enumerate() {
            columns[COL_INITCODE_HASH_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        let ih_limbs = bytes32_to_le_limbs(&inv.initcode_hash);
        for (k, &l) in ih_limbs.iter().enumerate() {
            columns[COL_INITCODE_HASH_LIMB_OFFSET + k][i] = Scalar::from_u64(l, curve);
        }
        let canonical = canonical_create2_input(&inv.sender, &inv.salt, &inv.initcode_hash);
        for (b, &v) in canonical.iter().enumerate() {
            columns[COL_INPUT_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_INPUT_LEN][i] = Scalar::from_u64(INPUT_OUTPUT_LEN as u64, curve);
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

pub struct Create2InputConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Create2InputConstraintSystem {
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
    col_evals[COL_IS_REAL].mul(&body)
}

fn eval_sender_limb_3_zero(col_evals: &[Scalar]) -> Scalar {
    let limb = &col_evals[COL_SENDER_LIMB_OFFSET + 3];
    col_evals[COL_IS_REAL].mul(limb)
}

/// Generic 8-byte LE limb binding: `IS_REAL · (LIMB[k] - Σ byte[8k+i]·256^i)`.
fn eval_byte_limb_binding(
    col_evals: &[Scalar],
    byte_offset: usize,
    limb_offset: usize,
    k: usize,
) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut sum = Scalar::zero(curve);
    for i in 0..8 {
        let byte = &col_evals[byte_offset + 8 * k + i];
        sum = sum.add(&byte.mul(&byte_power(i, curve)));
    }
    let limb = &col_evals[limb_offset + k];
    let body = limb.sub(&sum);
    col_evals[COL_IS_REAL].mul(&body)
}

fn eval_salt_limb_binding(col_evals: &[Scalar], k: usize) -> Scalar {
    eval_byte_limb_binding(col_evals, COL_SALT_BYTE_OFFSET, COL_SALT_LIMB_OFFSET, k)
}

fn eval_initcode_hash_limb_binding(col_evals: &[Scalar], k: usize) -> Scalar {
    eval_byte_limb_binding(
        col_evals,
        COL_INITCODE_HASH_BYTE_OFFSET,
        COL_INITCODE_HASH_LIMB_OFFSET,
        k,
    )
}

fn eval_input_construction(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    let mut acc = zero.clone();
    let mut ap = one.clone();

    let push = |acc: &mut Scalar, ap: &mut Scalar, body: Scalar, alpha: &Scalar| {
        *acc = acc.add(&body.mul(ap));
        *ap = ap.mul(alpha);
    };

    // INPUT_BYTE[0] - 0xff
    let body = col_evals[COL_INPUT_BYTE_OFFSET + 0]
        .sub(&Scalar::from_u64(CREATE2_MAGIC_BYTE, curve));
    push(&mut acc, &mut ap, body, alpha);
    // INPUT_BYTE[1 + k] - SENDER_BYTE[k] for k in 0..20
    for k in 0..NUM_SENDER_BYTES {
        let body = col_evals[COL_INPUT_BYTE_OFFSET + 1 + k]
            .sub(&col_evals[COL_SENDER_BYTE_OFFSET + k]);
        push(&mut acc, &mut ap, body, alpha);
    }
    // INPUT_BYTE[21 + k] - SALT_BYTE[k] for k in 0..32
    for k in 0..NUM_SALT_BYTES {
        let body = col_evals[COL_INPUT_BYTE_OFFSET + 1 + NUM_SENDER_BYTES + k]
            .sub(&col_evals[COL_SALT_BYTE_OFFSET + k]);
        push(&mut acc, &mut ap, body, alpha);
    }
    // INPUT_BYTE[53 + k] - INITCODE_HASH_BYTE[k] for k in 0..32
    for k in 0..NUM_INITCODE_HASH_BYTES {
        let body = col_evals[COL_INPUT_BYTE_OFFSET + 1 + NUM_SENDER_BYTES + NUM_SALT_BYTES + k]
            .sub(&col_evals[COL_INITCODE_HASH_BYTE_OFFSET + k]);
        push(&mut acc, &mut ap, body, alpha);
    }
    // INPUT_BYTE[k] = 0 for k in 85..256
    for k in INPUT_OUTPUT_LEN..INPUT_BYTE_WIDTH {
        let body = col_evals[COL_INPUT_BYTE_OFFSET + k].clone();
        push(&mut acc, &mut ap, body, alpha);
    }
    // INPUT_LEN - 85
    let body = col_evals[COL_INPUT_LEN].sub(&Scalar::from_u64(INPUT_OUTPUT_LEN as u64, curve));
    push(&mut acc, &mut ap, body, alpha);

    col_evals[COL_IS_REAL].mul(&acc)
}

// Polynomial-form helpers.

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

fn build_byte_limb_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    byte_offset: usize,
    limb_offset: usize,
    k: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for i in 0..8 {
        let byte_poly = &col_coeffs[byte_offset + 8 * k + i];
        let scaled = poly_scalar_mul(byte_poly, &byte_power(i, curve));
        sum = poly_add(&sum, &scaled, curve);
    }
    let limb_poly = &col_coeffs[limb_offset + k];
    let diff = poly_sub(limb_poly, &sum, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &diff, curve)
}

fn build_input_construction_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);

    let const_ff = vec![Scalar::from_u64(CREATE2_MAGIC_BYTE, curve)];
    let body = poly_sub(&col_coeffs[COL_INPUT_BYTE_OFFSET + 0], &const_ff, curve);
    acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
    ap = ap.mul(alpha);

    for k in 0..NUM_SENDER_BYTES {
        let body = poly_sub(
            &col_coeffs[COL_INPUT_BYTE_OFFSET + 1 + k],
            &col_coeffs[COL_SENDER_BYTE_OFFSET + k],
            curve,
        );
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }

    for k in 0..NUM_SALT_BYTES {
        let body = poly_sub(
            &col_coeffs[COL_INPUT_BYTE_OFFSET + 1 + NUM_SENDER_BYTES + k],
            &col_coeffs[COL_SALT_BYTE_OFFSET + k],
            curve,
        );
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }

    for k in 0..NUM_INITCODE_HASH_BYTES {
        let body = poly_sub(
            &col_coeffs[COL_INPUT_BYTE_OFFSET + 1 + NUM_SENDER_BYTES + NUM_SALT_BYTES + k],
            &col_coeffs[COL_INITCODE_HASH_BYTE_OFFSET + k],
            curve,
        );
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }

    for k in INPUT_OUTPUT_LEN..INPUT_BYTE_WIDTH {
        let body = col_coeffs[COL_INPUT_BYTE_OFFSET + k].clone();
        acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
        ap = ap.mul(alpha);
    }

    let const_85 = vec![Scalar::from_u64(INPUT_OUTPUT_LEN as u64, curve)];
    let body = poly_sub(&col_coeffs[COL_INPUT_LEN], &const_85, curve);
    acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);

    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for Create2InputConstraintSystem {
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
            "salt_limb_0_binding".into(),
            "salt_limb_1_binding".into(),
            "salt_limb_2_binding".into(),
            "salt_limb_3_binding".into(),
            "initcode_hash_limb_0_binding".into(),
            "initcode_hash_limb_1_binding".into(),
            "initcode_hash_limb_2_binding".into(),
            "initcode_hash_limb_3_binding".into(),
            "input_construction".into(),
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
        let mut salt0 = vec![Scalar::zero(curve); n];
        let mut salt1 = vec![Scalar::zero(curve); n];
        let mut salt2 = vec![Scalar::zero(curve); n];
        let mut salt3 = vec![Scalar::zero(curve); n];
        let mut ih0 = vec![Scalar::zero(curve); n];
        let mut ih1 = vec![Scalar::zero(curve); n];
        let mut ih2 = vec![Scalar::zero(curve); n];
        let mut ih3 = vec![Scalar::zero(curve); n];
        let mut ic = vec![Scalar::zero(curve); n];

        for row in 0..n {
            let v = &columns[COL_IS_REAL][row];
            bin[row] = v.mul(&v.sub(&one));
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            s0[row] = eval_sender_limb_binding(&row_evals, 0);
            s1[row] = eval_sender_limb_binding(&row_evals, 1);
            s2[row] = eval_sender_limb_binding(&row_evals, 2);
            s3[row] = eval_sender_limb_3_zero(&row_evals);
            salt0[row] = eval_salt_limb_binding(&row_evals, 0);
            salt1[row] = eval_salt_limb_binding(&row_evals, 1);
            salt2[row] = eval_salt_limb_binding(&row_evals, 2);
            salt3[row] = eval_salt_limb_binding(&row_evals, 3);
            ih0[row] = eval_initcode_hash_limb_binding(&row_evals, 0);
            ih1[row] = eval_initcode_hash_limb_binding(&row_evals, 1);
            ih2[row] = eval_initcode_hash_limb_binding(&row_evals, 2);
            ih3[row] = eval_initcode_hash_limb_binding(&row_evals, 3);
            ic[row] = eval_input_construction(&row_evals, &alpha_for_rlc);
        }
        vec![bin, s0, s1, s2, s3, salt0, salt1, salt2, salt3, ih0, ih1, ih2, ih3, ic]
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
            eval_salt_limb_binding(col_evals, 0),
            eval_salt_limb_binding(col_evals, 1),
            eval_salt_limb_binding(col_evals, 2),
            eval_salt_limb_binding(col_evals, 3),
            eval_initcode_hash_limb_binding(col_evals, 0),
            eval_initcode_hash_limb_binding(col_evals, 1),
            eval_initcode_hash_limb_binding(col_evals, 2),
            eval_initcode_hash_limb_binding(col_evals, 3),
            eval_input_construction(col_evals, alpha),
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
            build_byte_limb_binding_poly(
                col_coeffs, COL_SALT_BYTE_OFFSET, COL_SALT_LIMB_OFFSET, 0, curve,
            ),
            build_byte_limb_binding_poly(
                col_coeffs, COL_SALT_BYTE_OFFSET, COL_SALT_LIMB_OFFSET, 1, curve,
            ),
            build_byte_limb_binding_poly(
                col_coeffs, COL_SALT_BYTE_OFFSET, COL_SALT_LIMB_OFFSET, 2, curve,
            ),
            build_byte_limb_binding_poly(
                col_coeffs, COL_SALT_BYTE_OFFSET, COL_SALT_LIMB_OFFSET, 3, curve,
            ),
            build_byte_limb_binding_poly(
                col_coeffs,
                COL_INITCODE_HASH_BYTE_OFFSET,
                COL_INITCODE_HASH_LIMB_OFFSET,
                0,
                curve,
            ),
            build_byte_limb_binding_poly(
                col_coeffs,
                COL_INITCODE_HASH_BYTE_OFFSET,
                COL_INITCODE_HASH_LIMB_OFFSET,
                1,
                curve,
            ),
            build_byte_limb_binding_poly(
                col_coeffs,
                COL_INITCODE_HASH_BYTE_OFFSET,
                COL_INITCODE_HASH_LIMB_OFFSET,
                2,
                curve,
            ),
            build_byte_limb_binding_poly(
                col_coeffs,
                COL_INITCODE_HASH_BYTE_OFFSET,
                COL_INITCODE_HASH_LIMB_OFFSET,
                3,
                curve,
            ),
            build_input_construction_poly(col_coeffs, alpha, curve),
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

/// Cross-AIR LogUp descriptor binding the **EVM main trace's CREATE2
/// invocations** to **this gadget's invocations**.
///
/// Tuple matched: `(sender_limb[0..4], salt_limb[0..4],
/// initcode_hash_limb[0..4])` — 12 LE u64 limbs — gated on the EVM
/// side by `sel_create2` (only fires on opcode 0xF5) and on the gadget
/// side by `IS_REAL`.
///
/// On the EVM side the columns are:
/// * sender = `frame_callee_l0..l3` — the executing contract's
///   address, i.e. the CREATE2 sender (same convention as the CREATE
///   linkage; `frame_callee` snapshots the CALLEE-side of the
///   currently-executing frame, which IS the contract that issued the
///   CREATE2 opcode).
/// * salt = `create2_salt_hint_l0..l3` — backfilled by the inspector
///   from `inputs.scheme()` in the `create()` hook on CREATE2 rows.
/// * initcode_hash = `create2_initcode_hash_hint_l0..l3` — also
///   backfilled, as `keccak256(inputs.init_code())`.
///
/// On the gadget side the columns are `COL_SENDER_LIMB_OFFSET +
/// 0..4`, `COL_SALT_LIMB_OFFSET + 0..4`, `COL_INITCODE_HASH_LIMB_OFFSET
/// + 0..4` — each set populated from the per-byte columns by the new
/// limb-binding constraints.
pub fn make_evm_main_create2_input_linkage_descriptor(
    evm_layer_index: usize,
    gadget_layer_index: usize,
    evm_frame_callee_l0_col: usize,
    evm_create2_salt_hint_l0_col: usize,
    evm_create2_initcode_hash_hint_l0_col: usize,
    evm_sel_create2_col: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(12);
    for k in 0..4 { a_columns.push(evm_frame_callee_l0_col + k); }
    for k in 0..4 { a_columns.push(evm_create2_salt_hint_l0_col + k); }
    for k in 0..4 { a_columns.push(evm_create2_initcode_hash_hint_l0_col + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(12);
    for k in 0..4 { b_columns.push(COL_SENDER_LIMB_OFFSET + k); }
    for k in 0..4 { b_columns.push(COL_SALT_LIMB_OFFSET + k); }
    for k in 0..4 { b_columns.push(COL_INITCODE_HASH_LIMB_OFFSET + k); }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_main_create2_input_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(evm_sel_create2_col),
        b_layer_index: gadget_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding **this gadget's INPUT** to
/// **KeccakExtract's input side**: matches the
/// `(INPUT_BYTE[0..256], INPUT_LEN)` tuple against
/// `(INPUT_BYTE[0..256], INPUT_LEN)`.
///
/// Combined with KeccakExtract↔Keccak (#91), the keccak invocation
/// hashing this gadget's CREATE2 pre-image is algebraically pinned
/// to be computing the canonical CREATE2 input hash.
pub fn make_create2_input_keccak_extract_input_linkage_descriptor(
    gadget_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    debug_assert_eq!(
        INPUT_BYTE_WIDTH,
        ke::MAX_INPUT_LEN,
        "CREATE2 gadget's keccak-input width must equal KeccakExtract MAX_INPUT_LEN"
    );
    let mut a_columns: Vec<usize> = (0..INPUT_BYTE_WIDTH)
        .map(|b| COL_INPUT_BYTE_OFFSET + b)
        .collect();
    a_columns.push(COL_INPUT_LEN);

    let mut b_columns: Vec<usize> = (0..ke::MAX_INPUT_LEN)
        .map(|b| ke::COL_INPUT_BYTE_OFFSET + b)
        .collect();
    b_columns.push(ke::COL_INPUT_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "create2_input_keccak_extract_input_v1".into(),
        a_layer_index: gadget_layer_index,
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

    fn small_witness() -> Create2InputWitness {
        let s = [0x42u8; 20];
        let salt = [0x00u8; 32];
        let init = [0xAAu8; 32];
        Create2InputWitness::from_inputs(&[(&s, &salt, &init)])
    }

    #[test]
    fn canonical_input_layout() {
        let s = [0x11u8; 20];
        let mut salt = [0u8; 32];
        for i in 0..32 { salt[i] = (i as u8) + 1; }
        let mut init = [0u8; 32];
        for i in 0..32 { init[i] = 0xff - (i as u8); }
        let out = canonical_create2_input(&s, &salt, &init);
        assert_eq!(out[0], 0xff);
        assert_eq!(&out[1..21], &s);
        assert_eq!(&out[21..53], &salt);
        assert_eq!(&out[53..85], &init);
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 1);
        // INPUT_BYTE[0] = 0xff
        assert_eq!(
            trace.columns[COL_INPUT_BYTE_OFFSET].evaluations[0].to_u64(),
            CREATE2_MAGIC_BYTE
        );
        // INPUT_BYTE[1] = sender[0] = 0x42
        assert_eq!(
            trace.columns[COL_INPUT_BYTE_OFFSET + 1].evaluations[0].to_u64(),
            0x42
        );
        // INPUT_BYTE[21] = salt[0] = 0x00
        assert_eq!(
            trace.columns[COL_INPUT_BYTE_OFFSET + 21].evaluations[0].to_u64(),
            0x00
        );
        // INPUT_BYTE[53] = initcode_hash[0] = 0xAA
        assert_eq!(
            trace.columns[COL_INPUT_BYTE_OFFSET + 53].evaluations[0].to_u64(),
            0xAA
        );
        // INPUT_BYTE[85] = 0 (zero tail)
        assert_eq!(
            trace.columns[COL_INPUT_BYTE_OFFSET + 85].evaluations[0].to_u64(),
            0
        );
        // INPUT_LEN = 85
        assert_eq!(
            trace.columns[COL_INPUT_LEN].evaluations[0].to_u64(),
            INPUT_OUTPUT_LEN as u64
        );
        // SENDER_LIMB[0] = u64::from_le_bytes([0x42; 8])
        let expected_l0 = u64::from_le_bytes([0x42; 8]);
        assert_eq!(
            trace.columns[COL_SENDER_LIMB_OFFSET].evaluations[0].to_u64(),
            expected_l0
        );
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = Create2InputConstraintSystem::new(trace.num_rows);
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
    fn input_construction_rejects_tampered_magic_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = Create2InputConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(13, curve);
        let one = Scalar::one(curve);
        // Flip INPUT_BYTE[0] (the 0xff magic byte).
        trace.columns[COL_INPUT_BYTE_OFFSET].evaluations[0] =
            trace.columns[COL_INPUT_BYTE_OFFSET].evaluations[0].add(&one);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "input_construction must fire when INPUT_BYTE[0] != 0xff"
        );
    }

    #[test]
    fn input_construction_rejects_tampered_salt() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        let cs = Create2InputConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        let one = Scalar::one(curve);
        // Tamper SALT_BYTE[5] without updating INPUT_BYTE[26] (= 21+5).
        trace.columns[COL_SALT_BYTE_OFFSET + 5].evaluations[0] =
            trace.columns[COL_SALT_BYTE_OFFSET + 5].evaluations[0].add(&one);
        let row_evals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        assert!(
            !cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
            "input_construction must fire when SALT_BYTE[5] disagrees with INPUT_BYTE[26]"
        );
    }

    #[test]
    fn create2_input_keccak_extract_descriptor_well_formed() {
        use crate::keccak_extract as ke;
        let desc = make_create2_input_keccak_extract_input_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "create2_input_keccak_extract_input_v1");
        let tuple_len = ke::MAX_INPUT_LEN + 1;
        assert_eq!(desc.a_columns.len(), tuple_len);
        assert_eq!(desc.b_columns.len(), tuple_len);
        assert_eq!(desc.a_columns[0], COL_INPUT_BYTE_OFFSET);
        assert_eq!(desc.a_columns[ke::MAX_INPUT_LEN - 1], COL_INPUT_BYTE_OFFSET + ke::MAX_INPUT_LEN - 1);
        assert_eq!(desc.a_columns[ke::MAX_INPUT_LEN], COL_INPUT_LEN);
        assert_eq!(desc.b_columns[0], ke::COL_INPUT_BYTE_OFFSET);
        assert_eq!(desc.b_columns[ke::MAX_INPUT_LEN], ke::COL_INPUT_LEN);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(ke::COL_IS_REAL));
    }

    /// End-to-end byte-tuple matching: build a real
    /// (CREATE2 gadget, KeccakExtract) pair from the same
    /// `(sender, salt, initcode_hash)` invocation and assert the
    /// 257-element cross-AIR tuples agree byte-by-byte. This is the
    /// foundation of the input-side linkage's soundness.
    #[test]
    fn gadget_byte_tuples_match_keccak_extract_input() {
        use crate::keccak_extract::{KeccakExtractWitness, build_trace_polynomials as ke_build};
        let curve = CurveType::Bls48581;
        // Distinct, non-trivial inputs.
        let mut sender = [0u8; 20];
        for i in 0..20 { sender[i] = (i as u8) * 11 + 7; }
        let mut salt = [0u8; 32];
        for i in 0..32 { salt[i] = (i as u8).wrapping_mul(13).wrapping_add(1); }
        let mut init = [0u8; 32];
        for i in 0..32 { init[i] = (i as u8).wrapping_mul(17).wrapping_add(3); }

        let gadget_w = Create2InputWitness::from_inputs(&[(&sender, &salt, &init)]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);

        let canonical = canonical_create2_input(&sender, &salt, &init);
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical.to_vec()])
            .expect("CREATE2 input fits in MAX_INPUT_LEN (85 < 256)");
        let extract_trace = ke_build(&extract_w, curve);

        for b in 0..crate::keccak_extract::MAX_INPUT_LEN {
            let g = gadget_trace.columns[COL_INPUT_BYTE_OFFSET + b].evaluations[0]
                .to_u64() as u8;
            let e = extract_trace.columns[crate::keccak_extract::COL_INPUT_BYTE_OFFSET + b]
                .evaluations[0]
                .to_u64() as u8;
            assert_eq!(g, e, "tuple byte {} mismatch (gadget vs extract)", b);
        }
        let g_len = gadget_trace.columns[COL_INPUT_LEN].evaluations[0].to_u64();
        let e_len = extract_trace.columns[crate::keccak_extract::COL_INPUT_LEN].evaluations[0]
            .to_u64();
        assert_eq!(g_len, INPUT_OUTPUT_LEN as u64);
        assert_eq!(g_len, e_len);
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn create2_input_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = Create2InputConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "create2 input proof must verify");
    }

    /// 2-AIR joint_prove regression for the CREATE2 input-side
    /// linkage: the new gadget ↔ KeccakExtract input. Mirrors
    /// `joint_prove_create_rlp_keccak_extract_input_linkage` (CREATE
    /// case).
    #[test]
    #[ignore = "slow: 2-AIR joint_prove + joint_verify (~30s); run with --release --ignored"]
    fn joint_prove_create2_input_keccak_extract_input_linkage() {
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

        let mut sender = [0u8; 20];
        for i in 0..20 { sender[i] = (i as u8) * 11 + 7; }
        let mut salt = [0u8; 32];
        for i in 0..32 { salt[i] = (i as u8).wrapping_mul(13).wrapping_add(1); }
        let mut init = [0u8; 32];
        for i in 0..32 { init[i] = (i as u8).wrapping_mul(17).wrapping_add(3); }

        let gadget_w = Create2InputWitness::from_inputs(&[(&sender, &salt, &init)]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = Create2InputConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let canonical = canonical_create2_input(&sender, &salt, &init);
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical.to_vec()])
            .expect("CREATE2 input fits in MAX_INPUT_LEN");
        let extract_trace = ke_build(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        let linkage = make_create2_input_keccak_extract_input_linkage_descriptor(0, 1);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&gadget_trace, &gadget_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![linkage];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for matched CREATE2 gadget + KeccakExtract");
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
            "joint verifier must accept the honest CREATE2 gadget ↔ KeccakExtract input proof"
        );
    }

    /// 3-AIR end-to-end CREATE2 input chain: gadget → KeccakExtract →
    /// bit-level Keccak. Validates the full input-side soundness chain
    /// for CREATE2 algebraically (no EVM main side, since the
    /// EVM-side adapter for `salt` + `initcode_hash` oracle columns
    /// is a separate cascade).
    ///
    /// Linkages:
    ///   - L1: gadget↔KeccakExtract input (RLP byte tuples)
    ///   - L2: KeccakExtract↔bit-level Keccak (byte aggregator tuples)
    ///
    /// Combined with the bit-level Keccak's algebraic input/output
    /// bindings (closed via #91), this proves: for some
    /// `(sender, salt, initcode_hash)` invocation, the gadget's
    /// committed `INPUT_BYTE = 0xff || sender || salt || initcode_hash`
    /// is the actual input that bit-level Keccak hashes, and the
    /// resulting digest is computed correctly from that input.
    ///
    /// This is the CREATE2 analog of the 4-AIR CREATE chain
    /// (`joint_prove_evm_create_address_e2e_with_bit_level_keccak`)
    /// minus the EVM main layer. Once the EVM-side CREATE2 adapter
    /// (salt + initcode_hash columns) lands, the 4-AIR CREATE2
    /// version is a straightforward extension.
    #[test]
    #[ignore = "slow: 3-AIR + 2-linkage joint_prove with bit-level Keccak \
                (~10 min); run with --release --ignored"]
    fn joint_prove_create2_input_chain_e2e() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak::{keccak256, keccak_witness, NUM_ROUNDS};
        use crate::keccak_air;
        use crate::keccak_constraints::KeccakConstraintSystem;
        use crate::keccak_extract::{
            self, make_keccak_extract_keccak_linkage_descriptor,
            KeccakExtractWitness, KeccakExtractConstraintSystem,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Pick a representative invocation.
        let mut sender = [0u8; 20];
        for i in 0..20 { sender[i] = (i as u8) * 11 + 7; }
        let mut salt = [0u8; 32];
        for i in 0..32 { salt[i] = (i as u8).wrapping_mul(13).wrapping_add(1); }
        let mut init = [0u8; 32];
        for i in 0..32 { init[i] = (i as u8).wrapping_mul(17).wrapping_add(3); }

        // ── CREATE2 input gadget ──
        let gadget_w = Create2InputWitness::from_inputs(&[(&sender, &salt, &init)]);
        let gadget_trace = build_trace_polynomials(&gadget_w, curve);
        let gadget_cs = Create2InputConstraintSystem::new(gadget_trace.num_rows);

        // ── KeccakExtract: same input as gadget ──
        let canonical = canonical_create2_input(&sender, &salt, &init);
        let canonical_vec: Vec<u8> = canonical.to_vec();
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical_vec.clone()])
            .expect("CREATE2 input fits in MAX_INPUT_LEN");
        let extract_trace = keccak_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows);

        // ── Bit-level Keccak: actual hash computation ──
        let digest = keccak256(&canonical_vec);
        let ht = keccak_witness(&canonical_vec);
        let num_keccak_rows = ht.blocks.len() * NUM_ROUNDS;
        let keccak_padded = crate::trace::nearest_power_of_two(num_keccak_rows.max(1));
        let mut keccak_columns = keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &canonical_vec, &digest, curve,
        );
        for col in keccak_columns.iter_mut() {
            if col.len() < keccak_padded {
                col.resize(keccak_padded, Scalar::zero(curve));
            }
        }
        let keccak_polys: Vec<crate::trace::Polynomial> = keccak_columns
            .into_iter()
            .map(|evals| crate::trace::Polynomial {
                evaluations: evals,
                degree: num_keccak_rows,
            })
            .collect();
        let keccak_trace = crate::trace::TracePolynomials {
            columns: keccak_polys,
            num_rows: num_keccak_rows,
            padded_size: keccak_padded as u64,
            curve,
        };
        let keccak_cs = KeccakConstraintSystem::new(num_keccak_rows);

        // ── Linkages ──
        let l1 = make_create2_input_keccak_extract_input_linkage_descriptor(
            /* gadget */ 0, /* extract */ 1,
        );
        let l2 = make_keccak_extract_keccak_linkage_descriptor(
            /* extract */ 1, /* keccak */ 2,
        );

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
            (&keccak_trace, &keccak_cs),
        ];
        let linkages = vec![l1, l2];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the 3-AIR CREATE2 input chain");
        assert_eq!(proofs.len(), 3);
        assert_eq!(extension.linkage_proofs.len(), 2);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&gadget_cs, &extract_cs, &keccak_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(valid, "joint verifier must accept the 3-AIR CREATE2 input chain");
    }
}
