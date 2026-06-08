//! Per-validator `hash_tree_root` gadget AIR (Phase 1).
//!
//! Closes the residual binding gap exposed in #119/#120/#121: the
//! `validator_extract` AIR exposes a `(effective_balance, validator_root)`
//! pair as oracles, with no algebraic relation tying them together.
//! This gadget proves that for each row the prover commits to:
//!
//!   * raw beacon-`Validator` fields (pubkey, withdrawal_credentials,
//!     effective_balance, slashed, 4 epoch fields)
//!   * 8 derived leaf chunks corresponding to each field's
//!     `hash_tree_root` (matching `crate::beacon::Validator::hash_tree_root`)
//!   * a single `validator_root` chunk
//!
//! The cross-AIR LogUp linkages (Phases 2-4, future work) bind:
//!
//!   * `pubkey_root = sha256(pubkey_chunk_0 || pubkey_chunk_1)` via
//!     a SHA-256 extract row (the pubkey leaf is itself the hash of
//!     a 2-chunk tree; one external SHA-256 invocation per validator).
//!   * The 8-leaf merkleization producing `validator_root` via 7
//!     SHA-256 invocations exposed by the SSZ AIR.
//!   * `(effective_balance, validator_root)` ↔ the corresponding
//!     `validator_extract` row, gated by `IS_REAL`.
//!
//! End-to-end, this turns the residual #84 binding gap into algebraic
//! soundness: a malicious prover cannot decouple `effective_balance`
//! from `validator_root` because the gadget enforces both are
//! consistent with the same beacon `Validator` record, AND both are
//! linked to the rest of the chain through the cross-AIR LogUps.
//!
//! # Phase 1 scope (this file)
//!
//! All algebraic field-to-leaf encoding constraints. The pubkey leaf
//! and the validator_root remain **oracle columns** for now: the
//! witness builder populates them with the actual computed values
//! from `crate::beacon::Validator::hash_tree_root`, and Phase 2/3
//! linkages will pin those oracles cryptographically.
//!
//! # Constraint layout (Phase 1)
//!
//! 14 row-local constraints:
//!
//!   0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `slashed_binary` — `SLASHED · (SLASHED − 1) = 0`
//!   2. `eb_limb_binding` — `IS_REAL · (EFFECTIVE_BALANCE − Σ EB_BYTE[i]·256^i) = 0`
//!   3-6. `ae_epoch_limb_binding`, `act_epoch_limb_binding`,
//!        `exit_epoch_limb_binding`, `wd_epoch_limb_binding` — 4×
//!        identical to body 2 for each epoch field.
//!   7. `leaf_1_binding` (β-RLC over 32 sub-bodies) — pins
//!      `LEAF_1_BYTE[i] = WITHDRAWAL_CREDENTIALS_BYTE[i]` for i ∈ [0, 32).
//!   8. `leaf_2_binding` (β-RLC over 32 sub-bodies) — pins
//!      `LEAF_2_BYTE[i] = EFFECTIVE_BALANCE_BYTE[i]` for i ∈ [0, 8) and
//!      `LEAF_2_BYTE[i] = 0` for i ∈ [8, 32) (LE u64 right-padded with zeros).
//!   9. `leaf_3_binding` (β-RLC over 32 sub-bodies) — pins
//!      `LEAF_3_BYTE[0] = SLASHED` and `LEAF_3_BYTE[i] = 0` for i ∈ [1, 32).
//!   10-13. `leaf_4_binding`, `leaf_5_binding`, `leaf_6_binding`,
//!          `leaf_7_binding` — 4× identical to body 8 for each epoch field.
//!
//! All 14 bodies are gated by `IS_REAL` (so padding rows trivially
//! satisfy them with all columns zero).
//!
//! Pubkey leaf (LEAF_0) and validator_root remain unconstrained by
//! Phase 1 — they are oracles bound by future cross-AIR LogUps.
//!
//! # Sizing
//!
//! NUM_COLUMNS = 415 (per row):
//!   - 121 raw field columns (pubkey 48, wc 32, eb 8, slashed 1, 4×8 epochs)
//!   - 5 aggregator columns (eb + 4 epochs as u64 limbs)
//!   - 8 × 32 = 256 leaf chunk byte columns
//!   - 32 validator_root byte columns
//!   - 1 IS_REAL
//!
//! 14 row-local constraints, no shifted constraints.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const PUBKEY_BYTES: usize = 48;
pub const WC_BYTES: usize = 32;
pub const EB_BYTES: usize = 8;
pub const EPOCH_BYTES: usize = 8;
pub const NUM_EPOCH_FIELDS: usize = 4;
pub const CHUNK_BYTES: usize = 32;
pub const NUM_LEAVES: usize = 8;

// ─── Column indices ────────────────────────────────────────────────────
//
// Raw field byte columns (input region).

pub const COL_PUBKEY_BYTE_OFFSET: usize = 0;
pub const COL_WC_BYTE_OFFSET: usize = COL_PUBKEY_BYTE_OFFSET + PUBKEY_BYTES;
pub const COL_EB_BYTE_OFFSET: usize = COL_WC_BYTE_OFFSET + WC_BYTES;
pub const COL_SLASHED: usize = COL_EB_BYTE_OFFSET + EB_BYTES;
pub const COL_AE_EPOCH_BYTE_OFFSET: usize = COL_SLASHED + 1;
pub const COL_ACT_EPOCH_BYTE_OFFSET: usize =
    COL_AE_EPOCH_BYTE_OFFSET + EPOCH_BYTES;
pub const COL_EXIT_EPOCH_BYTE_OFFSET: usize =
    COL_ACT_EPOCH_BYTE_OFFSET + EPOCH_BYTES;
pub const COL_WD_EPOCH_BYTE_OFFSET: usize =
    COL_EXIT_EPOCH_BYTE_OFFSET + EPOCH_BYTES;

// Aggregator (u64-limb) columns. Each binds Σ byte[i]·256^i for the
// corresponding 8-byte field. These are the columns the cross-AIR
// LogUp linkages reference (e.g. validator_extract's
// `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)` tuple matches against
// `(invocation_index, EFFECTIVE_BALANCE)` here once the
// cross-AIR linkage descriptor is added in Phase 4).

pub const COL_EFFECTIVE_BALANCE: usize = COL_WD_EPOCH_BYTE_OFFSET + EPOCH_BYTES;
pub const COL_AE_EPOCH: usize = COL_EFFECTIVE_BALANCE + 1;
pub const COL_ACT_EPOCH: usize = COL_AE_EPOCH + 1;
pub const COL_EXIT_EPOCH: usize = COL_ACT_EPOCH + 1;
pub const COL_WD_EPOCH: usize = COL_EXIT_EPOCH + 1;

// Leaf chunk byte columns (8 leaves × 32 bytes).
pub const COL_LEAF_BYTE_OFFSET: usize = COL_WD_EPOCH + 1;

/// Helper: the column index of byte `b` of leaf `k`.
pub const fn col_leaf_byte(leaf: usize, byte: usize) -> usize {
    COL_LEAF_BYTE_OFFSET + leaf * CHUNK_BYTES + byte
}

// Validator root byte columns.
pub const COL_VALIDATOR_ROOT_BYTE_OFFSET: usize =
    COL_LEAF_BYTE_OFFSET + NUM_LEAVES * CHUNK_BYTES;

/// Right-chunk bytes for the pubkey leaf's SHA-256 pair invocation.
/// Layout: bytes 0..16 = `pubkey_byte[32..48]`; bytes 16..32 = 0
/// (the canonical zero-pad for `hash_tree_root_bytes_fixed(pubkey, 2)`).
/// The cross-AIR LogUp linkage to a SHA-256 extract row matches the
/// 96-byte tuple `(PUBKEY_BYTE[0..32], PUBKEY_CHUNK_1_BYTE[0..32],
/// LEAF_0_BYTE[0..32])` against the extract row's
/// `(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])` — pinning
/// `LEAF_0_BYTE = sha256(pubkey[0..32] || pubkey[32..48]||zeros)`
/// algebraically.
pub const COL_PUBKEY_CHUNK_1_BYTE_OFFSET: usize =
    COL_VALIDATOR_ROOT_BYTE_OFFSET + CHUNK_BYTES;

/// Layer-1 intermediate parent chunks (4 chunks × 32 bytes = 128 cols).
/// `LAYER1_PARENT[k]` for k ∈ [0, 4) is `sha256(LEAF[2k] || LEAF[2k+1])`.
/// Phase 3 oracle columns; each is bound by a cross-AIR LogUp to a
/// Sha256Extract row hashing the corresponding leaf pair.
pub const COL_LAYER1_PARENT_BYTE_OFFSET: usize =
    COL_PUBKEY_CHUNK_1_BYTE_OFFSET + CHUNK_BYTES;
pub const NUM_LAYER1_PARENTS: usize = 4;

pub const fn col_layer1_parent_byte(pair: usize, byte: usize) -> usize {
    COL_LAYER1_PARENT_BYTE_OFFSET + pair * CHUNK_BYTES + byte
}

/// Layer-2 intermediate parent chunks (2 chunks × 32 bytes = 64 cols).
/// `LAYER2_PARENT[k]` for k ∈ [0, 2) is
/// `sha256(LAYER1_PARENT[2k] || LAYER1_PARENT[2k+1])`.
pub const COL_LAYER2_PARENT_BYTE_OFFSET: usize =
    COL_LAYER1_PARENT_BYTE_OFFSET + NUM_LAYER1_PARENTS * CHUNK_BYTES;
pub const NUM_LAYER2_PARENTS: usize = 2;

pub const fn col_layer2_parent_byte(pair: usize, byte: usize) -> usize {
    COL_LAYER2_PARENT_BYTE_OFFSET + pair * CHUNK_BYTES + byte
}

/// Phase 4 chain-consistency anchor: validator index `i` on the i-th
/// real row, increasing by 1 per row. Pinned by the
/// `validator_index_chain` shifted constraint
/// (`IS_REAL(ω·X) · (VI(ω·X) − VI(X) − 1) = 0` with wrap exclusion),
/// matching ValidatorExtract's convention. Used by the
/// validator_extract ↔ validator_htr cross-AIR LogUp descriptor as
/// part of the 34-tuple
/// `(VALIDATOR_INDEX, EFFECTIVE_BALANCE, validator_root_byte[0..32])`.
pub const COL_VALIDATOR_INDEX: usize =
    COL_LAYER2_PARENT_BYTE_OFFSET + NUM_LAYER2_PARENTS * CHUNK_BYTES;

pub const COL_IS_REAL: usize = COL_VALIDATOR_INDEX + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// 1 (is_real_binary) + 1 (slashed_binary) + 1 (eb_limb) + 4 (epoch_limb)
/// + 7 (leaves 1..7 each via a single β-RLC body)
/// + 1 (pubkey_chunk_1 byte-by-byte binding via β-RLC) = 15.
pub const NUM_ROW_CONSTRAINTS: usize = 15;
/// 1 (validator_index_chain) — same shape as ValidatorExtract's.
pub const NUM_SHIFTED: usize = 1;

// ─── Witness type ──────────────────────────────────────────────────────

/// One validator's row of the gadget AIR. Mirrors a subset of
/// [`crate::beacon::Validator`]; the gadget only needs the fields that
/// participate in `hash_tree_root`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatorHtrRow {
    pub pubkey: [u8; PUBKEY_BYTES],
    pub withdrawal_credentials: [u8; WC_BYTES],
    pub effective_balance: u64,
    pub slashed: bool,
    pub activation_eligibility_epoch: u64,
    pub activation_epoch: u64,
    pub exit_epoch: u64,
    pub withdrawable_epoch: u64,
}

impl ValidatorHtrRow {
    pub fn from_beacon_validator(v: &crate::beacon::Validator) -> Self {
        Self {
            pubkey: v.pubkey,
            withdrawal_credentials: v.withdrawal_credentials,
            effective_balance: v.effective_balance,
            slashed: v.slashed,
            activation_eligibility_epoch: v.activation_eligibility_epoch,
            activation_epoch: v.activation_epoch,
            exit_epoch: v.exit_epoch,
            withdrawable_epoch: v.withdrawable_epoch,
        }
    }
}

/// Sequence of validator rows. The trace builder pads to a power-of-2
/// domain with zeros (so `IS_REAL = 0` on padding rows).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidatorHtrWitness {
    pub validators: Vec<ValidatorHtrRow>,
}

impl ValidatorHtrWitness {
    pub fn from_beacon_validators(vs: &[crate::beacon::Validator]) -> Self {
        Self {
            validators: vs.iter().map(ValidatorHtrRow::from_beacon_validator).collect(),
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

/// `hash_tree_root_uint(value)` as a 32-byte chunk: 8 LE bytes followed
/// by 24 zero bytes. Mirrors [`crate::ssz::hash_tree_root_uint`].
fn uint_leaf(value: u64) -> [u8; CHUNK_BYTES] {
    let mut out = [0u8; CHUNK_BYTES];
    out[..8].copy_from_slice(&value.to_le_bytes());
    out
}

/// `hash_tree_root_bool(value)` as a 32-byte chunk: 0x01 or 0x00 in
/// byte 0, all other bytes zero.
fn bool_leaf(value: bool) -> [u8; CHUNK_BYTES] {
    let mut out = [0u8; CHUNK_BYTES];
    out[0] = value as u8;
    out
}

/// Compute all 8 leaf chunks for a single validator. Matches the
/// per-field layout of [`crate::beacon::Validator::hash_tree_root`].
pub fn compute_leaves(row: &ValidatorHtrRow) -> [[u8; CHUNK_BYTES]; NUM_LEAVES] {
    use crate::ssz::hash_tree_root_bytes_fixed;
    let pubkey_root = hash_tree_root_bytes_fixed(&row.pubkey, 2);
    [
        pubkey_root,
        row.withdrawal_credentials,
        uint_leaf(row.effective_balance),
        bool_leaf(row.slashed),
        uint_leaf(row.activation_eligibility_epoch),
        uint_leaf(row.activation_epoch),
        uint_leaf(row.exit_epoch),
        uint_leaf(row.withdrawable_epoch),
    ]
}

/// Compute the validator hash_tree_root from a `ValidatorHtrRow`.
/// Equal to `Validator::hash_tree_root` for the corresponding beacon
/// validator (proven equal in the witness-builder unit test).
pub fn compute_validator_root(row: &ValidatorHtrRow) -> [u8; CHUNK_BYTES] {
    use crate::ssz::hash_tree_root_container;
    let leaves = compute_leaves(row);
    hash_tree_root_container(&leaves)
}

fn byte_power(i: usize, curve: CurveType) -> Scalar {
    debug_assert!(i < 8, "byte_power overflows u64 for i ≥ 8");
    Scalar::from_u64(1u64 << (8 * i), curve)
}

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

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ValidatorHtrWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.validators.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (i, v) in witness.validators.iter().enumerate() {
        // Phase 4 anchor.
        columns[COL_VALIDATOR_INDEX][i] = Scalar::from_u64(i as u64, curve);

        // Raw fields.
        for (b, &byte) in v.pubkey.iter().enumerate() {
            columns[COL_PUBKEY_BYTE_OFFSET + b][i] = Scalar::from_u64(byte as u64, curve);
        }
        for (b, &byte) in v.withdrawal_credentials.iter().enumerate() {
            columns[COL_WC_BYTE_OFFSET + b][i] = Scalar::from_u64(byte as u64, curve);
        }
        let eb_bytes = v.effective_balance.to_le_bytes();
        for (b, &byte) in eb_bytes.iter().enumerate() {
            columns[COL_EB_BYTE_OFFSET + b][i] = Scalar::from_u64(byte as u64, curve);
        }
        columns[COL_SLASHED][i] = Scalar::from_u64(v.slashed as u64, curve);
        let epoch_byte_offsets = [
            (v.activation_eligibility_epoch, COL_AE_EPOCH_BYTE_OFFSET),
            (v.activation_epoch, COL_ACT_EPOCH_BYTE_OFFSET),
            (v.exit_epoch, COL_EXIT_EPOCH_BYTE_OFFSET),
            (v.withdrawable_epoch, COL_WD_EPOCH_BYTE_OFFSET),
        ];
        for (val, off) in epoch_byte_offsets {
            for (b, &byte) in val.to_le_bytes().iter().enumerate() {
                columns[off + b][i] = Scalar::from_u64(byte as u64, curve);
            }
        }

        // Aggregators (u64 values).
        columns[COL_EFFECTIVE_BALANCE][i] =
            Scalar::from_u64(v.effective_balance, curve);
        columns[COL_AE_EPOCH][i] =
            Scalar::from_u64(v.activation_eligibility_epoch, curve);
        columns[COL_ACT_EPOCH][i] = Scalar::from_u64(v.activation_epoch, curve);
        columns[COL_EXIT_EPOCH][i] = Scalar::from_u64(v.exit_epoch, curve);
        columns[COL_WD_EPOCH][i] = Scalar::from_u64(v.withdrawable_epoch, curve);

        // Leaf chunks.
        let leaves = compute_leaves(v);
        for (k, leaf) in leaves.iter().enumerate() {
            for (b, &byte) in leaf.iter().enumerate() {
                columns[col_leaf_byte(k, b)][i] = Scalar::from_u64(byte as u64, curve);
            }
        }

        // Validator root (Phase 1 oracle).
        let root = compute_validator_root(v);
        for (b, &byte) in root.iter().enumerate() {
            columns[COL_VALIDATOR_ROOT_BYTE_OFFSET + b][i] =
                Scalar::from_u64(byte as u64, curve);
        }

        // Pubkey chunk 1 (right side of the SHA-256 pair invocation).
        // Bytes 0..16 = pubkey[32..48]; bytes 16..32 = 0.
        for b in 0..16 {
            columns[COL_PUBKEY_CHUNK_1_BYTE_OFFSET + b][i] =
                Scalar::from_u64(v.pubkey[32 + b] as u64, curve);
        }
        // Bytes 16..32 already zeroed at allocation.

        // Phase 3 intermediate parent chunks. Compute the 8-leaf
        // container merkleization layer by layer, populating the
        // 4 layer-1 + 2 layer-2 oracle columns.
        let layer1: [[u8; CHUNK_BYTES]; NUM_LAYER1_PARENTS] = [
            crate::sha256::sha256_pair(&leaves[0], &leaves[1]),
            crate::sha256::sha256_pair(&leaves[2], &leaves[3]),
            crate::sha256::sha256_pair(&leaves[4], &leaves[5]),
            crate::sha256::sha256_pair(&leaves[6], &leaves[7]),
        ];
        for (k, parent) in layer1.iter().enumerate() {
            for (b, &byte) in parent.iter().enumerate() {
                columns[col_layer1_parent_byte(k, b)][i] =
                    Scalar::from_u64(byte as u64, curve);
            }
        }
        let layer2: [[u8; CHUNK_BYTES]; NUM_LAYER2_PARENTS] = [
            crate::sha256::sha256_pair(&layer1[0], &layer1[1]),
            crate::sha256::sha256_pair(&layer1[2], &layer1[3]),
        ];
        for (k, parent) in layer2.iter().enumerate() {
            for (b, &byte) in parent.iter().enumerate() {
                columns[col_layer2_parent_byte(k, b)][i] =
                    Scalar::from_u64(byte as u64, curve);
            }
        }
        // Sanity: VALIDATOR_ROOT should equal sha256_pair(LAYER2[0], LAYER2[1]).
        debug_assert_eq!(
            crate::sha256::sha256_pair(&layer2[0], &layer2[1]),
            root,
            "validator_root must equal final-layer pair hash"
        );

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

pub struct ValidatorHtrConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ValidatorHtrConstraintSystem {
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

/// Generic 8-byte LE limb binding gated by IS_REAL:
/// `IS_REAL · (limb − Σ_i byte[i]·256^i) = 0`.
fn eval_limb_binding(
    col_evals: &[Scalar],
    byte_offset: usize,
    limb_col: usize,
) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut sum = Scalar::zero(curve);
    for i in 0..8 {
        let byte = &col_evals[byte_offset + i];
        sum = sum.add(&byte.mul(&byte_power(i, curve)));
    }
    let limb = &col_evals[limb_col];
    let body = limb.sub(&sum);
    col_evals[COL_IS_REAL].mul(&body)
}

fn build_limb_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    byte_offset: usize,
    limb_col: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for i in 0..8 {
        let byte_poly = &col_coeffs[byte_offset + i];
        let scaled = poly_scalar_mul(byte_poly, &byte_power(i, curve));
        sum = poly_add(&sum, &scaled, curve);
    }
    let limb_poly = &col_coeffs[limb_col];
    let diff = poly_sub(limb_poly, &sum, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &diff, curve)
}

/// β-RLC body for a leaf chunk's 32 byte sub-bodies. `byte_provider`
/// returns the column index that each leaf byte must equal (or
/// `None` to mean "must be zero"). The whole body is gated by IS_REAL.
fn eval_leaf_binding(
    col_evals: &[Scalar],
    leaf: usize,
    byte_provider: impl Fn(usize) -> Option<usize>,
    alpha: &Scalar,
) -> Scalar {
    let curve = alpha.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for byte_idx in 0..CHUNK_BYTES {
        let leaf_byte = &col_evals[col_leaf_byte(leaf, byte_idx)];
        let body = match byte_provider(byte_idx) {
            Some(field_col) => leaf_byte.sub(&col_evals[field_col]),
            None => leaf_byte.clone(),
        };
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    col_evals[COL_IS_REAL].mul(&acc)
}

fn build_leaf_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    leaf: usize,
    byte_provider: impl Fn(usize) -> Option<usize>,
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for byte_idx in 0..CHUNK_BYTES {
        let leaf_poly = &col_coeffs[col_leaf_byte(leaf, byte_idx)];
        let body = match byte_provider(byte_idx) {
            Some(field_col) => poly_sub(leaf_poly, &col_coeffs[field_col], curve),
            None => leaf_poly.clone(),
        };
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

/// Byte-provider helpers — each describes one leaf's byte-by-byte mapping.

/// Leaf 1: withdrawal_credentials 32 bytes (no zero-tail).
fn wc_leaf_byte(byte: usize) -> Option<usize> {
    Some(COL_WC_BYTE_OFFSET + byte)
}

/// Leaf 2: 8-byte LE u64 (effective_balance) + 24 zero-tail.
fn eb_leaf_byte(byte: usize) -> Option<usize> {
    if byte < EB_BYTES {
        Some(COL_EB_BYTE_OFFSET + byte)
    } else {
        None
    }
}

/// Leaf 3: 1-byte SLASHED + 31 zero-tail.
fn slashed_leaf_byte(byte: usize) -> Option<usize> {
    if byte == 0 {
        Some(COL_SLASHED)
    } else {
        None
    }
}

/// Leaves 4..7: u64 epoch fields (8-byte LE + 24 zero-tail).
fn epoch_leaf_byte(byte_offset: usize, byte: usize) -> Option<usize> {
    if byte < EPOCH_BYTES {
        Some(byte_offset + byte)
    } else {
        None
    }
}

/// β-RLC body for the PUBKEY_CHUNK_1 columns: each byte must equal
/// PUBKEY_BYTE[32+i] for i ∈ [0, 16) and 0 for i ∈ [16, 32). Same shape
/// as `eval_leaf_binding` but on a different output column range.
fn eval_pubkey_chunk_1_binding(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for byte_idx in 0..CHUNK_BYTES {
        let target = &col_evals[COL_PUBKEY_CHUNK_1_BYTE_OFFSET + byte_idx];
        let body = if byte_idx < 16 {
            target.sub(&col_evals[COL_PUBKEY_BYTE_OFFSET + 32 + byte_idx])
        } else {
            target.clone()
        };
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    col_evals[COL_IS_REAL].mul(&acc)
}

fn build_pubkey_chunk_1_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for byte_idx in 0..CHUNK_BYTES {
        let target_poly = &col_coeffs[COL_PUBKEY_CHUNK_1_BYTE_OFFSET + byte_idx];
        let body = if byte_idx < 16 {
            poly_sub(
                target_poly,
                &col_coeffs[COL_PUBKEY_BYTE_OFFSET + 32 + byte_idx],
                curve,
            )
        } else {
            target_poly.clone()
        };
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for ValidatorHtrConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "slashed_binary".into(),
            "eb_limb_binding".into(),
            "ae_epoch_limb_binding".into(),
            "act_epoch_limb_binding".into(),
            "exit_epoch_limb_binding".into(),
            "wd_epoch_limb_binding".into(),
            "leaf_1_withdrawal_credentials_binding".into(),
            "leaf_2_effective_balance_binding".into(),
            "leaf_3_slashed_binding".into(),
            "leaf_4_ae_epoch_binding".into(),
            "leaf_5_act_epoch_binding".into(),
            "leaf_6_exit_epoch_binding".into(),
            "leaf_7_wd_epoch_binding".into(),
            "pubkey_chunk_1_binding".into(),
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

        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let slashed = &row_evals[COL_SLASHED];

            // 0. is_real_binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            // 1. slashed_binary (gated by IS_REAL — padding rows have slashed=0 trivially).
            bodies[1][row] = is_real.mul(&slashed.mul(&slashed.sub(&one)));
            // 2. eb_limb_binding.
            bodies[2][row] =
                eval_limb_binding(&row_evals, COL_EB_BYTE_OFFSET, COL_EFFECTIVE_BALANCE);
            // 3-6. epoch limb bindings.
            bodies[3][row] =
                eval_limb_binding(&row_evals, COL_AE_EPOCH_BYTE_OFFSET, COL_AE_EPOCH);
            bodies[4][row] =
                eval_limb_binding(&row_evals, COL_ACT_EPOCH_BYTE_OFFSET, COL_ACT_EPOCH);
            bodies[5][row] =
                eval_limb_binding(&row_evals, COL_EXIT_EPOCH_BYTE_OFFSET, COL_EXIT_EPOCH);
            bodies[6][row] =
                eval_limb_binding(&row_evals, COL_WD_EPOCH_BYTE_OFFSET, COL_WD_EPOCH);
            // 7. leaf 1 (withdrawal_credentials).
            bodies[7][row] = eval_leaf_binding(&row_evals, 1, wc_leaf_byte, &alpha_for_rlc);
            // 8. leaf 2 (effective_balance).
            bodies[8][row] = eval_leaf_binding(&row_evals, 2, eb_leaf_byte, &alpha_for_rlc);
            // 9. leaf 3 (slashed).
            bodies[9][row] =
                eval_leaf_binding(&row_evals, 3, slashed_leaf_byte, &alpha_for_rlc);
            // 10-13. leaves 4-7 (epochs).
            bodies[10][row] = eval_leaf_binding(
                &row_evals,
                4,
                |b| epoch_leaf_byte(COL_AE_EPOCH_BYTE_OFFSET, b),
                &alpha_for_rlc,
            );
            bodies[11][row] = eval_leaf_binding(
                &row_evals,
                5,
                |b| epoch_leaf_byte(COL_ACT_EPOCH_BYTE_OFFSET, b),
                &alpha_for_rlc,
            );
            bodies[12][row] = eval_leaf_binding(
                &row_evals,
                6,
                |b| epoch_leaf_byte(COL_EXIT_EPOCH_BYTE_OFFSET, b),
                &alpha_for_rlc,
            );
            bodies[13][row] = eval_leaf_binding(
                &row_evals,
                7,
                |b| epoch_leaf_byte(COL_WD_EPOCH_BYTE_OFFSET, b),
                &alpha_for_rlc,
            );
            // 14. pubkey_chunk_1_binding.
            bodies[14][row] = eval_pubkey_chunk_1_binding(&row_evals, &alpha_for_rlc);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals[COL_IS_REAL];
        let slashed = &col_evals[COL_SLASHED];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_real.mul(&slashed.mul(&slashed.sub(&one))),
            eval_limb_binding(col_evals, COL_EB_BYTE_OFFSET, COL_EFFECTIVE_BALANCE),
            eval_limb_binding(col_evals, COL_AE_EPOCH_BYTE_OFFSET, COL_AE_EPOCH),
            eval_limb_binding(col_evals, COL_ACT_EPOCH_BYTE_OFFSET, COL_ACT_EPOCH),
            eval_limb_binding(col_evals, COL_EXIT_EPOCH_BYTE_OFFSET, COL_EXIT_EPOCH),
            eval_limb_binding(col_evals, COL_WD_EPOCH_BYTE_OFFSET, COL_WD_EPOCH),
            eval_leaf_binding(col_evals, 1, wc_leaf_byte, alpha),
            eval_leaf_binding(col_evals, 2, eb_leaf_byte, alpha),
            eval_leaf_binding(col_evals, 3, slashed_leaf_byte, alpha),
            eval_leaf_binding(
                col_evals,
                4,
                |b| epoch_leaf_byte(COL_AE_EPOCH_BYTE_OFFSET, b),
                alpha,
            ),
            eval_leaf_binding(
                col_evals,
                5,
                |b| epoch_leaf_byte(COL_ACT_EPOCH_BYTE_OFFSET, b),
                alpha,
            ),
            eval_leaf_binding(
                col_evals,
                6,
                |b| epoch_leaf_byte(COL_EXIT_EPOCH_BYTE_OFFSET, b),
                alpha,
            ),
            eval_leaf_binding(
                col_evals,
                7,
                |b| epoch_leaf_byte(COL_WD_EPOCH_BYTE_OFFSET, b),
                alpha,
            ),
            eval_pubkey_chunk_1_binding(col_evals, alpha),
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
        let is_real = &col_coeffs[COL_IS_REAL];
        let slashed = &col_coeffs[COL_SLASHED];

        let is_real_minus_1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_minus_1, curve);

        let slashed_minus_1 = poly_sub(slashed, &one_poly, curve);
        let slashed_squared_minus = poly_mul(slashed, &slashed_minus_1, curve);
        let slashed_binary = poly_mul(is_real, &slashed_squared_minus, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            slashed_binary,
            build_limb_binding_poly(col_coeffs, COL_EB_BYTE_OFFSET, COL_EFFECTIVE_BALANCE, curve),
            build_limb_binding_poly(col_coeffs, COL_AE_EPOCH_BYTE_OFFSET, COL_AE_EPOCH, curve),
            build_limb_binding_poly(col_coeffs, COL_ACT_EPOCH_BYTE_OFFSET, COL_ACT_EPOCH, curve),
            build_limb_binding_poly(col_coeffs, COL_EXIT_EPOCH_BYTE_OFFSET, COL_EXIT_EPOCH, curve),
            build_limb_binding_poly(col_coeffs, COL_WD_EPOCH_BYTE_OFFSET, COL_WD_EPOCH, curve),
            build_leaf_binding_poly(col_coeffs, 1, wc_leaf_byte, alpha, curve),
            build_leaf_binding_poly(col_coeffs, 2, eb_leaf_byte, alpha, curve),
            build_leaf_binding_poly(col_coeffs, 3, slashed_leaf_byte, alpha, curve),
            build_leaf_binding_poly(
                col_coeffs,
                4,
                |b| epoch_leaf_byte(COL_AE_EPOCH_BYTE_OFFSET, b),
                alpha,
                curve,
            ),
            build_leaf_binding_poly(
                col_coeffs,
                5,
                |b| epoch_leaf_byte(COL_ACT_EPOCH_BYTE_OFFSET, b),
                alpha,
                curve,
            ),
            build_leaf_binding_poly(
                col_coeffs,
                6,
                |b| epoch_leaf_byte(COL_EXIT_EPOCH_BYTE_OFFSET, b),
                alpha,
                curve,
            ),
            build_leaf_binding_poly(
                col_coeffs,
                7,
                |b| epoch_leaf_byte(COL_WD_EPOCH_BYTE_OFFSET, b),
                alpha,
                curve,
            ),
            build_pubkey_chunk_1_binding_poly(col_coeffs, alpha, curve),
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

    fn shifted_column_indices(&self) -> Vec<usize> {
        // VALIDATOR_INDEX(ω·z) and IS_REAL(ω·z) for the chain
        // consistency body — same shape as ValidatorExtract.
        vec![COL_VALIDATOR_INDEX, COL_IS_REAL]
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
        if shifted_evals.len() < 2 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        // body = IS_REAL(ω·z) · (VALIDATOR_INDEX(ω·z) − VALIDATOR_INDEX(z) − 1)
        let vi_curr = &col_evals_at_z[COL_VALIDATOR_INDEX];
        let vi_next = &shifted_evals[0];
        let is_real_next = &shifted_evals[1];
        let diff = vi_next.sub(vi_curr).sub(&one);
        let body = is_real_next.mul(&diff);
        // Wrap-around exclusion at row n-1.
        let exclusion = z.sub(omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        ap.mul(&body).mul(&exclusion)
    }

    fn build_shifted_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let vi = &col_coeffs[COL_VALIDATOR_INDEX];
        let is_real = &col_coeffs[COL_IS_REAL];
        let vi_shift = poly_shift(vi, omega);
        let is_real_shift = poly_shift(is_real, omega);
        let diff_pre = poly_sub(&vi_shift, vi, curve);
        let diff = poly_sub(&diff_pre, &one_poly, curve);
        let body = poly_mul(&is_real_shift, &diff, curve);
        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let excluded = poly_mul_linear(&body, &omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        poly_scalar_mul(&excluded, &ap)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp linkage descriptors (Phase 2+) ─────────────────────

/// Generic helper: build a 96-byte tuple cross-AIR LogUp descriptor
/// matching `(left_chunk_byte[0..32], right_chunk_byte[0..32],
/// parent_chunk_byte[0..32])` on the gadget side against a
/// Sha256Extract row's `(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])`.
/// Both gated by `IS_REAL`.
///
/// The Phase 2 / Phase 3 cross-AIR linkages all follow this shape;
/// each merkleization position is a separate Sha256Extract layer.
fn make_pair_hash_linkage(
    label: &str,
    htr_layer_index: usize,
    sha256_extract_layer_index: usize,
    left_col_start: usize,
    right_col_start: usize,
    parent_col_start: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;
    let mut a_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..CHUNK_BYTES {
        a_columns.push(left_col_start + b);
    }
    for b in 0..CHUNK_BYTES {
        a_columns.push(right_col_start + b);
    }
    for b in 0..CHUNK_BYTES {
        a_columns.push(parent_col_start + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..se::NUM_INPUT_BYTES {
        b_columns.push(se::COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..se::NUM_OUTPUT_BYTES {
        b_columns.push(se::COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: htr_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Phase 3 layer-1 pair-k linkage: `(LEAF[2k], LEAF[2k+1],
/// LAYER1_PARENT[k])`. `k ∈ [0, 4)`.
pub fn make_validator_htr_layer1_pair_linkage_descriptor(
    pair: usize,
    htr_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    assert!(pair < NUM_LAYER1_PARENTS, "pair index out of range");
    let label = match pair {
        0 => "validator_htr_layer1_pair_0_v1",
        1 => "validator_htr_layer1_pair_1_v1",
        2 => "validator_htr_layer1_pair_2_v1",
        3 => "validator_htr_layer1_pair_3_v1",
        _ => unreachable!(),
    };
    make_pair_hash_linkage(
        label,
        htr_layer_index,
        sha256_extract_layer_index,
        col_leaf_byte(2 * pair, 0),
        col_leaf_byte(2 * pair + 1, 0),
        col_layer1_parent_byte(pair, 0),
    )
}

/// Phase 3 layer-2 pair-k linkage: `(LAYER1_PARENT[2k],
/// LAYER1_PARENT[2k+1], LAYER2_PARENT[k])`. `k ∈ [0, 2)`.
pub fn make_validator_htr_layer2_pair_linkage_descriptor(
    pair: usize,
    htr_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    assert!(pair < NUM_LAYER2_PARENTS, "pair index out of range");
    let label = match pair {
        0 => "validator_htr_layer2_pair_0_v1",
        1 => "validator_htr_layer2_pair_1_v1",
        _ => unreachable!(),
    };
    make_pair_hash_linkage(
        label,
        htr_layer_index,
        sha256_extract_layer_index,
        col_layer1_parent_byte(2 * pair, 0),
        col_layer1_parent_byte(2 * pair + 1, 0),
        col_layer2_parent_byte(pair, 0),
    )
}

/// Phase 3 layer-3 (root) linkage: `(LAYER2_PARENT[0],
/// LAYER2_PARENT[1], VALIDATOR_ROOT)`. Top of the depth-3 tree.
pub fn make_validator_htr_layer3_root_linkage_descriptor(
    htr_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    make_pair_hash_linkage(
        "validator_htr_layer3_root_v1",
        htr_layer_index,
        sha256_extract_layer_index,
        col_layer2_parent_byte(0, 0),
        col_layer2_parent_byte(1, 0),
        COL_VALIDATOR_ROOT_BYTE_OFFSET,
    )
}

/// Phase 4 cross-AIR LogUp descriptor: ValidatorExtract ↔
/// validator_htr 34-tuple binding.
///
/// Tuple matched (34 columns):
/// `(VALIDATOR_INDEX, EFFECTIVE_BALANCE, validator_root_byte[0..32])`.
///
/// Both sides gated by `IS_REAL`. The descriptor pins each VE row's
/// `(idx, eb, root)` to the same triple in this gadget — closing the
/// algebraic binding between EFFECTIVE_BALANCE and the validator's
/// hash_tree_root that VE alone could not enforce. Combined with
/// Phases 1-3's algebraic field-to-leaf encodings + SHA-256 cross-AIR
/// LogUps, this closes the residual #84 binding gap end-to-end.
///
/// **Soundness chain (closing #84 inner)**:
/// 1. Phase 1 gadget row-locals pin
///    `LEAF_k = canonical_field_encoding(field_k)` for all 7
///    non-pubkey leaves.
/// 2. Phase 2 SHA-256 cross-AIR LogUp pins `LEAF_0 = pubkey_root`.
/// 3. Phase 3 SHA-256 cross-AIR LogUps (×7) pin all intermediate
///    parents and `VALIDATOR_ROOT = container_merkleize(8 leaves)`.
/// 4. **Phase 4** (this descriptor) pins
///    `(VE.idx, VE.eff_bal, VE.validator_root) = (gadget.idx,
///    gadget.eff_bal, gadget.validator_root)`. Combined with VE's own
///    `validator_index_chain` shifted constraint, the validator
///    indices on both sides are pinned in canonical order.
///
/// Net result: VE's `(eff_bal, validator_root)` pair is now
/// algebraically required to come from a real beacon `Validator` whose
/// fields produce that exact root via the canonical
/// `Validator::hash_tree_root` algorithm. A malicious prover cannot
/// supply a fake `effective_balance` paired with an honest
/// `validator_root` (or vice versa) — the gadget chain forces them
/// to be consistent with the same underlying `Validator` record.
pub fn make_validator_extract_validator_htr_linkage_descriptor(
    ve_layer_index: usize,
    htr_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::validator_extract as ve;
    let mut a_columns: Vec<usize> = Vec::with_capacity(34);
    a_columns.push(ve::COL_VALIDATOR_INDEX);
    a_columns.push(ve::COL_EFFECTIVE_BALANCE);
    for b in 0..ve::VALIDATOR_ROOT_BYTES {
        a_columns.push(ve::COL_VALIDATOR_ROOT_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(34);
    b_columns.push(COL_VALIDATOR_INDEX);
    b_columns.push(COL_EFFECTIVE_BALANCE);
    for b in 0..CHUNK_BYTES {
        b_columns.push(COL_VALIDATOR_ROOT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_extract_validator_htr_v1".into(),
        a_layer_index: ve_layer_index,
        a_columns,
        a_selector_column: Some(ve::COL_IS_REAL),
        b_layer_index: htr_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Phase 2 cross-AIR LogUp descriptor: pubkey leaf hash binding.
///
/// Tuple matched (96 bytes):
///   * A side (this gadget): `(PUBKEY_BYTE[0..32],
///     PUBKEY_CHUNK_1_BYTE[0..32], LEAF_0_BYTE[0..32])`.
///   * B side (Sha256Extract): `(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])`.
///
/// Both sides gated by `IS_REAL`. The gadget's
/// `pubkey_chunk_1_binding` row-local pins
/// `PUBKEY_CHUNK_1_BYTE = pubkey[32..48] || zeros[16..32]`, so the
/// 96-byte tuple is exactly `(pubkey[0..32], pubkey[32..48]||zeros,
/// pubkey_root)`.
///
/// Combined with the existing Sha256Extract↔SHA-256 binding (#87),
/// this pins `LEAF_0_BYTE = sha256(pubkey[0..32] ||
/// pubkey[32..48]||zeros)` algebraically — closing the pubkey-leaf
/// arm of the validator hash_tree_root chain.
pub fn make_validator_htr_pubkey_root_linkage_descriptor(
    htr_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    make_pair_hash_linkage(
        "validator_htr_pubkey_root_v1",
        htr_layer_index,
        sha256_extract_layer_index,
        COL_PUBKEY_BYTE_OFFSET,
        COL_PUBKEY_CHUNK_1_BYTE_OFFSET,
        col_leaf_byte(0, 0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_witness() -> ValidatorHtrWitness {
        let v = crate::beacon::Validator {
            pubkey: {
                let mut p = [0u8; 48];
                for i in 0..48 {
                    p[i] = (i as u8).wrapping_mul(3).wrapping_add(7);
                }
                p
            },
            withdrawal_credentials: {
                let mut w = [0u8; 32];
                for i in 0..32 {
                    w[i] = (i as u8).wrapping_mul(5).wrapping_add(1);
                }
                w
            },
            effective_balance: 32_000_000_000,
            slashed: false,
            activation_eligibility_epoch: 100,
            activation_epoch: 200,
            exit_epoch: u64::MAX,
            withdrawable_epoch: u64::MAX,
        };
        ValidatorHtrWitness::from_beacon_validators(&[v])
    }

    /// Sanity: the gadget's `compute_validator_root` matches the
    /// canonical `Validator::hash_tree_root` byte-for-byte. If this
    /// fails the gadget is computing a different leaf layout than the
    /// SSZ reference.
    #[test]
    fn compute_validator_root_matches_canonical() {
        let w = small_witness();
        let row = &w.validators[0];
        let canonical = crate::beacon::Validator {
            pubkey: row.pubkey,
            withdrawal_credentials: row.withdrawal_credentials,
            effective_balance: row.effective_balance,
            slashed: row.slashed,
            activation_eligibility_epoch: row.activation_eligibility_epoch,
            activation_epoch: row.activation_epoch,
            exit_epoch: row.exit_epoch,
            withdrawable_epoch: row.withdrawable_epoch,
        }
        .hash_tree_root();
        assert_eq!(compute_validator_root(row), canonical);
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 1);
        // IS_REAL = 1 on row 0.
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
        // EFFECTIVE_BALANCE column matches the input.
        assert_eq!(
            trace.columns[COL_EFFECTIVE_BALANCE].evaluations[0].to_u64(),
            32_000_000_000
        );
        // Leaf 2 byte 0 = LE byte 0 of effective_balance.
        let expected_byte = (32_000_000_000u64 & 0xff) as u64;
        assert_eq!(
            trace.columns[col_leaf_byte(2, 0)].evaluations[0].to_u64(),
            expected_byte
        );
        // Leaf 2 byte 8 = 0 (start of zero tail).
        assert_eq!(trace.columns[col_leaf_byte(2, 8)].evaluations[0].to_u64(), 0);
        // Leaf 3 byte 0 = SLASHED = 0.
        assert_eq!(trace.columns[col_leaf_byte(3, 0)].evaluations[0].to_u64(), 0);
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {}, got {:?}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    /// Tampering: corrupt LEAF_2_BYTE[3] (a byte of EB's LE encoding).
    /// `leaf_2_effective_balance_binding` body must fire (β-RLC over
    /// 32 sub-bodies non-zero).
    #[test]
    fn leaf_2_binding_rejects_tampered_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper.
        trace.columns[col_leaf_byte(2, 3)].evaluations[0] =
            Scalar::from_u64(0xFF, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // body 8 is leaf_2_effective_balance_binding (zero-indexed).
        assert!(
            !evals[8][0].is_zero(),
            "tampered LEAF_2_BYTE[3] must make leaf_2_binding non-zero"
        );
    }

    /// Tampering: corrupt EFFECTIVE_BALANCE aggregator. `eb_limb_binding`
    /// body must fire.
    #[test]
    fn eb_limb_binding_rejects_tampered_aggregator() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_EFFECTIVE_BALANCE].evaluations[0] =
            Scalar::from_u64(0xDEADBEEF, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // body 2 is eb_limb_binding.
        assert!(
            !evals[2][0].is_zero(),
            "tampered EFFECTIVE_BALANCE aggregator must make eb_limb_binding non-zero"
        );
    }

    /// Slow regression: full prove + verify via the BLS48-581 scheme.
    /// Validates that the 415-column trace + 14 row-local constraints
    /// produces an accepting proof end-to-end.
    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn validator_htr_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid =
            crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "verifier must accept honest validator htr proof");
    }

    /// Phase 2 sanity: descriptor wiring matches the gadget's column
    /// layout and the Sha256Extract layout. 96-byte tuple on each side.
    #[test]
    fn pubkey_root_linkage_descriptor_well_formed() {
        let desc = make_validator_htr_pubkey_root_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "validator_htr_pubkey_root_v1");
        assert_eq!(desc.a_columns.len(), 96);
        assert_eq!(desc.b_columns.len(), 96);
        // A-side first 32 = chunk 0 (pubkey[0..32]).
        for b in 0..32 {
            assert_eq!(desc.a_columns[b], COL_PUBKEY_BYTE_OFFSET + b);
        }
        // A-side next 32 = chunk 1 (PUBKEY_CHUNK_1).
        for b in 0..32 {
            assert_eq!(desc.a_columns[32 + b], COL_PUBKEY_CHUNK_1_BYTE_OFFSET + b);
        }
        // A-side last 32 = pubkey_root (leaf 0).
        for b in 0..32 {
            assert_eq!(desc.a_columns[64 + b], col_leaf_byte(0, b));
        }
        // Selectors gated by IS_REAL on both sides.
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::sha256_extract::COL_IS_REAL)
        );
    }

    /// Phase 2 unit-level sanity (no proof): assert that the gadget's
    /// (chunk_0, chunk_1, pubkey_root) tuple at row 0 equals the
    /// Sha256Extract row 0 tuple BYTE FOR BYTE for the same pubkey
    /// input. If this fails, the gadget's PUBKEY_CHUNK_1 layout or
    /// Sha256Extract's input/output layout has drifted.
    #[test]
    fn pubkey_root_tuple_matches_sha256_extract_row() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let pubkey = w.validators[0].pubkey;
        // Build matching Sha256Extract trace.
        let mut chunk_0 = [0u8; 32];
        chunk_0.copy_from_slice(&pubkey[0..32]);
        let mut chunk_1 = [0u8; 32];
        chunk_1[..16].copy_from_slice(&pubkey[32..48]);
        let extract_w = crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(
            &[(chunk_0, chunk_1)],
        );
        let extract_trace =
            crate::sha256_extract::build_trace_polynomials(&extract_w, curve);
        let htr_trace = build_trace_polynomials(&w, curve);
        // Compare 64 input bytes.
        for b in 0..32 {
            let g = htr_trace.columns[COL_PUBKEY_BYTE_OFFSET + b].evaluations[0]
                .to_u64();
            let e = extract_trace.columns
                [crate::sha256_extract::COL_INPUT_BYTE_OFFSET + b]
                .evaluations[0]
                .to_u64();
            assert_eq!(g, e, "chunk_0 byte {} mismatch", b);
        }
        for b in 0..32 {
            let g = htr_trace.columns[COL_PUBKEY_CHUNK_1_BYTE_OFFSET + b]
                .evaluations[0]
                .to_u64();
            let e = extract_trace.columns
                [crate::sha256_extract::COL_INPUT_BYTE_OFFSET + 32 + b]
                .evaluations[0]
                .to_u64();
            assert_eq!(g, e, "chunk_1 byte {} mismatch", b);
        }
        // Compare 32 output bytes.
        for b in 0..32 {
            let g = htr_trace.columns[col_leaf_byte(0, b)].evaluations[0].to_u64();
            let e = extract_trace.columns
                [crate::sha256_extract::COL_OUTPUT_BYTE_OFFSET + b]
                .evaluations[0]
                .to_u64();
            assert_eq!(g, e, "output byte {} mismatch", b);
        }
    }

    /// Phase 2 end-to-end: 2-AIR joint_prove validating the
    /// pubkey_root cross-AIR linkage. Honest gadget + matching
    /// Sha256Extract row → joint_prove must accept; tampering with
    /// the gadget's pubkey_root must trigger multiset-equality
    /// failure at witness-build time.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove; run with --release --ignored"]
    fn joint_prove_validator_htr_pubkey_root_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let pubkey = w.validators[0].pubkey;
        let mut chunk_0 = [0u8; 32];
        chunk_0.copy_from_slice(&pubkey[0..32]);
        let mut chunk_1 = [0u8; 32];
        chunk_1[..16].copy_from_slice(&pubkey[32..48]);
        let extract_w = crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(
            &[(chunk_0, chunk_1)],
        );

        let htr_trace = build_trace_polynomials(&w, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);
        let extract_trace =
            crate::sha256_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = crate::sha256_extract::Sha256ExtractConstraintSystem::new(
            extract_trace.num_rows,
        );

        let l1 = make_validator_htr_pubkey_root_linkage_descriptor(0, 1);

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&htr_trace, &htr_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![l1];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for honest pubkey_root chain");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        for lp in &extension.linkage_proofs {
            assert_eq!(lp.closure_a, lp.closure_b);
        }
        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&htr_cs, &extract_cs];
        let valid =
            joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(valid, "joint verifier must accept honest 2-AIR HTR chain");
    }

    /// Tampering test for Phase 2: gadget reports a fake pubkey_root.
    /// The 96-tuple's last 32 bytes (= pubkey_root) won't match any
    /// real Sha256Extract output → `compute_cross_air_logup_witness`
    /// rejects.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove tampering test; run with --release --ignored"]
    fn joint_prove_validator_htr_pubkey_root_rejects_tampered_root() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let pubkey = w.validators[0].pubkey;
        let mut chunk_0 = [0u8; 32];
        chunk_0.copy_from_slice(&pubkey[0..32]);
        let mut chunk_1 = [0u8; 32];
        chunk_1[..16].copy_from_slice(&pubkey[32..48]);
        let extract_w = crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(
            &[(chunk_0, chunk_1)],
        );
        let extract_trace =
            crate::sha256_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = crate::sha256_extract::Sha256ExtractConstraintSystem::new(
            extract_trace.num_rows,
        );

        // TAMPER: corrupt LEAF_0_BYTE[0] (the gadget's pubkey_root).
        let mut htr_trace = build_trace_polynomials(&w, curve);
        let prev = htr_trace.columns[col_leaf_byte(0, 0)].evaluations[0]
            .to_u64();
        htr_trace.columns[col_leaf_byte(0, 0)].evaluations[0] =
            Scalar::from_u64(prev ^ 0xFF, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);

        let l1 = make_validator_htr_pubkey_root_linkage_descriptor(0, 1);

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&htr_trace, &htr_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![l1];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject HTR chain with tampered pubkey_root"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}",
            err
        );
    }

    /// Phase 3 sanity: descriptor wiring is well-formed for all 7
    /// container merkleization linkages.
    #[test]
    fn container_merkleization_descriptors_well_formed() {
        // Layer-1 pair k binds (LEAF[2k], LEAF[2k+1], LAYER1_PARENT[k]).
        for pair in 0..NUM_LAYER1_PARENTS {
            let desc = make_validator_htr_layer1_pair_linkage_descriptor(pair, 0, 1);
            assert_eq!(desc.a_columns.len(), 96);
            assert_eq!(desc.b_columns.len(), 96);
            // Left chunk = LEAF[2*pair].
            assert_eq!(desc.a_columns[0], col_leaf_byte(2 * pair, 0));
            // Right chunk = LEAF[2*pair+1].
            assert_eq!(desc.a_columns[32], col_leaf_byte(2 * pair + 1, 0));
            // Parent chunk = LAYER1_PARENT[pair].
            assert_eq!(desc.a_columns[64], col_layer1_parent_byte(pair, 0));
        }
        // Layer-2 pair k binds (LAYER1_PARENT[2k], LAYER1_PARENT[2k+1], LAYER2_PARENT[k]).
        for pair in 0..NUM_LAYER2_PARENTS {
            let desc = make_validator_htr_layer2_pair_linkage_descriptor(pair, 0, 1);
            assert_eq!(desc.a_columns[0], col_layer1_parent_byte(2 * pair, 0));
            assert_eq!(desc.a_columns[32], col_layer1_parent_byte(2 * pair + 1, 0));
            assert_eq!(desc.a_columns[64], col_layer2_parent_byte(pair, 0));
        }
        // Layer-3 root binds (LAYER2_PARENT[0], LAYER2_PARENT[1], VALIDATOR_ROOT).
        let desc = make_validator_htr_layer3_root_linkage_descriptor(0, 1);
        assert_eq!(desc.a_columns[0], col_layer2_parent_byte(0, 0));
        assert_eq!(desc.a_columns[32], col_layer2_parent_byte(1, 0));
        assert_eq!(desc.a_columns[64], COL_VALIDATOR_ROOT_BYTE_OFFSET);
    }

    /// Phase 3 sanity: assert each merkleization-position tuple
    /// (left, right, parent) at gadget row 0 matches the
    /// corresponding `Sha256ExtractWitness::from_pair_inputs` row
    /// byte-for-byte.
    #[test]
    fn container_merkleization_tuples_match_sha256_extract() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        // Gather actual leaf bytes from the trace.
        let mut leaves = [[0u8; CHUNK_BYTES]; NUM_LEAVES];
        for k in 0..NUM_LEAVES {
            for b in 0..CHUNK_BYTES {
                leaves[k][b] = trace.columns[col_leaf_byte(k, b)].evaluations[0]
                    .to_u64() as u8;
            }
        }
        // Layer 1: build matching extracts.
        for pair in 0..NUM_LAYER1_PARENTS {
            let extract_w =
                crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(&[(
                    leaves[2 * pair],
                    leaves[2 * pair + 1],
                )]);
            let extract_trace =
                crate::sha256_extract::build_trace_polynomials(&extract_w, curve);
            // Compare parent (output) bytes.
            for b in 0..32 {
                let g = trace.columns[col_layer1_parent_byte(pair, b)].evaluations[0]
                    .to_u64();
                let e = extract_trace.columns
                    [crate::sha256_extract::COL_OUTPUT_BYTE_OFFSET + b]
                    .evaluations[0]
                    .to_u64();
                assert_eq!(
                    g, e,
                    "layer-1 pair {} parent byte {} mismatch", pair, b
                );
            }
        }
        // Layer 2 and layer 3 follow analogous matching; spot-check
        // layer 3 (the validator_root) since it's the final output.
        let mut l2_0 = [0u8; CHUNK_BYTES];
        let mut l2_1 = [0u8; CHUNK_BYTES];
        for b in 0..32 {
            l2_0[b] = trace.columns[col_layer2_parent_byte(0, b)].evaluations[0]
                .to_u64() as u8;
            l2_1[b] = trace.columns[col_layer2_parent_byte(1, b)].evaluations[0]
                .to_u64() as u8;
        }
        let extract_w = crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(
            &[(l2_0, l2_1)],
        );
        let extract_trace =
            crate::sha256_extract::build_trace_polynomials(&extract_w, curve);
        for b in 0..32 {
            let g = trace.columns[COL_VALIDATOR_ROOT_BYTE_OFFSET + b]
                .evaluations[0]
                .to_u64();
            let e = extract_trace.columns
                [crate::sha256_extract::COL_OUTPUT_BYTE_OFFSET + b]
                .evaluations[0]
                .to_u64();
            assert_eq!(g, e, "layer-3 root byte {} mismatch", b);
        }
    }

    /// Phase 3 end-to-end: 9-AIR joint_prove validating the full
    /// container merkleization chain.
    ///
    /// Layout:
    ///   - Layer 0: validator_htr gadget (1 row)
    ///   - Layers 1-8: 8 separate Sha256Extract instances, each
    ///     with 1 row, one for each merkleization position
    ///     (pubkey leaf + 4 layer-1 + 2 layer-2 + 1 layer-3).
    ///
    /// Eight cross-AIR LogUp linkages bind every gadget hash
    /// invocation to a real SHA-256 pair-hash. Combined with the
    /// existing Sha256Extract↔SHA-256 binding (#87), this pins
    /// `validator_root = container_merkleize(8 leaves)` algebraically
    /// end to end. A malicious prover cannot fabricate any
    /// intermediate parent without violating multiset equality
    /// against a real hash invocation.
    #[test]
    #[ignore = "slow: 9-AIR + 8-linkage joint_prove (~30s); run with --release --ignored"]
    fn joint_prove_validator_htr_full_container_merkleization() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let row = &w.validators[0];
        let leaves = compute_leaves(row);
        let layer1: [[u8; CHUNK_BYTES]; NUM_LAYER1_PARENTS] = [
            crate::sha256::sha256_pair(&leaves[0], &leaves[1]),
            crate::sha256::sha256_pair(&leaves[2], &leaves[3]),
            crate::sha256::sha256_pair(&leaves[4], &leaves[5]),
            crate::sha256::sha256_pair(&leaves[6], &leaves[7]),
        ];
        let layer2: [[u8; CHUNK_BYTES]; NUM_LAYER2_PARENTS] = [
            crate::sha256::sha256_pair(&layer1[0], &layer1[1]),
            crate::sha256::sha256_pair(&layer1[2], &layer1[3]),
        ];

        // Pubkey leaf inputs.
        let mut pubkey_chunk_0 = [0u8; 32];
        pubkey_chunk_0.copy_from_slice(&row.pubkey[0..32]);
        let mut pubkey_chunk_1 = [0u8; 32];
        pubkey_chunk_1[..16].copy_from_slice(&row.pubkey[32..48]);

        // Build 8 separate Sha256Extract instances.
        let extract_pairs: [(crate::ssz::Chunk, crate::ssz::Chunk); 8] = [
            (pubkey_chunk_0, pubkey_chunk_1),
            (leaves[0], leaves[1]),
            (leaves[2], leaves[3]),
            (leaves[4], leaves[5]),
            (leaves[6], leaves[7]),
            (layer1[0], layer1[1]),
            (layer1[2], layer1[3]),
            (layer2[0], layer2[1]),
        ];
        let extract_witnesses: Vec<crate::sha256_extract::Sha256ExtractWitness> =
            extract_pairs
                .iter()
                .map(|p| crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(&[*p]))
                .collect();
        let extract_traces: Vec<TracePolynomials> = extract_witnesses
            .iter()
            .map(|w| crate::sha256_extract::build_trace_polynomials(w, curve))
            .collect();
        let extract_constraints: Vec<crate::sha256_extract::Sha256ExtractConstraintSystem> =
            extract_traces
                .iter()
                .map(|t| {
                    crate::sha256_extract::Sha256ExtractConstraintSystem::new(t.num_rows)
                })
                .collect();

        // Gadget trace.
        let htr_trace = build_trace_polynomials(&w, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);

        // 8 cross-AIR linkages. Layers: 0 = htr; 1..=8 = extracts.
        let l_pubkey = make_validator_htr_pubkey_root_linkage_descriptor(0, 1);
        let l1_p0 = make_validator_htr_layer1_pair_linkage_descriptor(0, 0, 2);
        let l1_p1 = make_validator_htr_layer1_pair_linkage_descriptor(1, 0, 3);
        let l1_p2 = make_validator_htr_layer1_pair_linkage_descriptor(2, 0, 4);
        let l1_p3 = make_validator_htr_layer1_pair_linkage_descriptor(3, 0, 5);
        let l2_p0 = make_validator_htr_layer2_pair_linkage_descriptor(0, 0, 6);
        let l2_p1 = make_validator_htr_layer2_pair_linkage_descriptor(1, 0, 7);
        let l3 = make_validator_htr_layer3_root_linkage_descriptor(0, 8);

        let mut traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = Vec::with_capacity(9);
        traces.push((&htr_trace, &htr_cs));
        for (t, c) in extract_traces.iter().zip(extract_constraints.iter()) {
            traces.push((t, c));
        }
        let linkages =
            vec![l_pubkey, l1_p0, l1_p1, l1_p2, l1_p3, l2_p0, l2_p1, l3];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for the honest 9-AIR validator HTR chain",
        );
        assert_eq!(proofs.len(), 9);
        assert_eq!(extension.linkage_proofs.len(), 8);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }
        let mut cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            Vec::with_capacity(9);
        cs_refs.push(&htr_cs);
        for c in &extract_constraints {
            cs_refs.push(c);
        }
        let valid =
            joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the honest 9-AIR validator HTR chain"
        );
    }

    /// Build a 2-validator witness with distinct fields. Validators
    /// 0 and 1 differ in pubkey, withdrawal_credentials,
    /// effective_balance, slashed, and all 4 epochs — so any
    /// chain-consistency or cross-row bug would show up as a
    /// constraint failure on at least one body.
    fn two_validator_witness() -> ValidatorHtrWitness {
        let mut p0 = [0u8; 48];
        let mut p1 = [0u8; 48];
        for i in 0..48 {
            p0[i] = (i as u8).wrapping_mul(3).wrapping_add(7);
            p1[i] = (i as u8).wrapping_mul(13).wrapping_add(1);
        }
        let mut wc0 = [0u8; 32];
        let mut wc1 = [0u8; 32];
        for i in 0..32 {
            wc0[i] = (i as u8).wrapping_mul(5).wrapping_add(1);
            wc1[i] = (i as u8).wrapping_mul(11).wrapping_add(3);
        }
        let v0 = crate::beacon::Validator {
            pubkey: p0,
            withdrawal_credentials: wc0,
            effective_balance: 32_000_000_000,
            slashed: false,
            activation_eligibility_epoch: 100,
            activation_epoch: 200,
            exit_epoch: u64::MAX,
            withdrawable_epoch: u64::MAX,
        };
        let v1 = crate::beacon::Validator {
            pubkey: p1,
            withdrawal_credentials: wc1,
            effective_balance: 16_000_000_000,
            slashed: true,
            activation_eligibility_epoch: 50,
            activation_epoch: 75,
            exit_epoch: 1000,
            withdrawable_epoch: 1064,
        };
        ValidatorHtrWitness::from_beacon_validators(&[v0, v1])
    }

    /// Multi-validator (N=2) constraint vanishing: confirms the gadget's
    /// row-locals AND the new `validator_index_chain` shifted
    /// constraint vanish on a 2-row honest trace. If
    /// VALIDATOR_INDEX[0] != 0 or VALIDATOR_INDEX[1] != 1 the chain
    /// consistency body would fire.
    #[test]
    fn multi_validator_constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = two_validator_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.num_rows, 2);
        // VALIDATOR_INDEX is 0, 1 on the two real rows.
        assert_eq!(trace.columns[COL_VALIDATOR_INDEX].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_VALIDATOR_INDEX].evaluations[1].to_u64(), 1);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local {} ({}) must vanish at row {}, got {:?}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    /// Multi-validator round-trip prove/verify. Validates the full
    /// gadget AIR (640 cols, 15 row-locals, 1 shifted) on a 2-row
    /// trace.
    #[test]
    #[ignore = "slow: full prove + verify on 2 validators; run with --release --ignored"]
    fn validator_htr_proof_round_trips_two_validators() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = two_validator_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid =
            crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "verifier must accept honest 2-validator HTR proof");
    }

    /// Tampering: flip VALIDATOR_INDEX[1] to 0 (broken chain). The
    /// `validator_index_chain` shifted body must fire at the
    /// row-0 → row-1 transition.
    #[test]
    fn validator_index_chain_rejects_broken_chain() {
        let curve = CurveType::Bls48581;
        let w = two_validator_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        // TAMPER: VALIDATOR_INDEX[1] = 0 instead of 1.
        trace.columns[COL_VALIDATOR_INDEX].evaluations[1] =
            Scalar::from_u64(0, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);

        // For padded_size = 2, ω is the unique non-trivial 2nd root
        // of unity = −1.  z = ω^0 = 1 → shifted = ω·z = ω = −1, which
        // corresponds to row 1 in the evaluation domain.
        let one = Scalar::one(curve);
        let omega = Scalar::zero(curve).sub(&one);
        let z = one.clone();
        let omega_n_minus_1 = omega.clone();

        // Column evaluations at z (= row 0 in the natural domain
        // ordering).
        let col_evals_at_z: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        // Shifted evaluations at ω·z (= row 1).
        let shifted = vec![
            trace.columns[COL_VALIDATOR_INDEX].evaluations[1].clone(),
            trace.columns[COL_IS_REAL].evaluations[1].clone(),
        ];
        let alpha = Scalar::from_u64(7, curve);
        let body = cs.evaluate_shifted_at_point(
            &col_evals_at_z,
            &shifted,
            &z,
            &omega_n_minus_1,
            &alpha,
            0,
        );
        // body = IS_REAL(row 1) · (VI(row 1) − VI(row 0) − 1) · (z − ω^{n-1})
        //      = 1 · (0 − 0 − 1) · (1 − (−1))
        //      = 1 · −1 · 2 = −2 (non-zero).
        assert!(
            !body.is_zero(),
            "tampered VALIDATOR_INDEX chain must produce non-zero shifted body"
        );
    }

    /// Phase 5: 3-AIR Finality + VE + HTR end-to-end joint_prove.
    /// Validates the full eb-binding chain through Phase 4's
    /// VE↔HTR linkage:
    ///   - L1 (Finality↔VE on (idx, eff_bal)): pins finality's claim
    ///     to VE's row.
    ///   - L2 (VE↔HTR on (idx, eff_bal, validator_root)): pins VE's
    ///     `(eb, root)` pair to a real `Validator` record's
    ///     hash_tree_root output.
    ///
    /// Combined with HTR Phases 1-3 (which gate validator_root to be
    /// the canonical hash of the validator's fields), this proves
    /// algebraically that finality's claimed effective_balance
    /// belongs to a validator whose hash_tree_root matches the
    /// committed value.
    ///
    /// (Closing the link to the SSZ registry tree on top of this
    /// requires the existing 5-AIR chain in #114/#120; combining all
    /// of those into one e2e test is left as future work because of
    /// runtime: bit-level SHA-256 alone takes ~14 min per run.)
    #[test]
    #[ignore = "slow: 3-AIR Finality + VE + HTR joint_prove (~10s); run with --release --ignored"]
    fn joint_prove_finality_ve_htr_chain() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::finality_constraints::{
            build_finality_trace_polynomials, FinalityConstraintSystem, FinalityWitness,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            make_finality_validator_extract_linkage_descriptor,
            ValidatorExtractConstraintSystem, ValidatorExtractWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let v = &w.validators[0];
        let validator = crate::beacon::Validator {
            pubkey: v.pubkey,
            withdrawal_credentials: v.withdrawal_credentials,
            effective_balance: v.effective_balance,
            slashed: v.slashed,
            activation_eligibility_epoch: v.activation_eligibility_epoch,
            activation_epoch: v.activation_epoch,
            exit_epoch: v.exit_epoch,
            withdrawable_epoch: v.withdrawable_epoch,
        };

        // Finality claims this validator's effective balance.
        let finality_w = FinalityWitness {
            validators: vec![(v.effective_balance, 1)],
            total_active_balance_gwei: v.effective_balance,
            attestation_data_root: [0xDD; 32],
            finalized_root: [0xCC; 32],
        };
        let finality_trace = build_finality_trace_polynomials(&finality_w, curve);
        let finality_cs = FinalityConstraintSystem::new(finality_w.validators.len());

        let ve_w = ValidatorExtractWitness::from_beacon_validators(&[validator]);
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        let htr_trace = build_trace_polynomials(&w, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);

        let l1 = make_finality_validator_extract_linkage_descriptor(0, 1);
        let l2 = make_validator_extract_validator_htr_linkage_descriptor(1, 2);

        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&finality_trace, &finality_cs),
            (&ve_trace, &ve_cs),
            (&htr_trace, &htr_cs),
        ];
        let linkages = vec![l1, l2];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for the 3-AIR Finality+VE+HTR chain",
        );
        assert_eq!(proofs.len(), 3);
        assert_eq!(extension.linkage_proofs.len(), 2);
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }
        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&finality_cs, &ve_cs, &htr_cs];
        let valid =
            joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the 3-AIR Finality+VE+HTR chain"
        );
    }

    /// Phase 4 sanity: VE↔HTR descriptor wiring.
    #[test]
    fn validator_extract_validator_htr_descriptor_well_formed() {
        let desc = make_validator_extract_validator_htr_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "validator_extract_validator_htr_v1");
        assert_eq!(desc.a_columns.len(), 34);
        assert_eq!(desc.b_columns.len(), 34);
        // A side column 0 = VE's VALIDATOR_INDEX.
        assert_eq!(
            desc.a_columns[0],
            crate::validator_extract::COL_VALIDATOR_INDEX
        );
        // A side column 1 = VE's EFFECTIVE_BALANCE.
        assert_eq!(
            desc.a_columns[1],
            crate::validator_extract::COL_EFFECTIVE_BALANCE
        );
        // B side first two = HTR's VALIDATOR_INDEX, EFFECTIVE_BALANCE.
        assert_eq!(desc.b_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(desc.b_columns[1], COL_EFFECTIVE_BALANCE);
        // Both gated by IS_REAL.
        assert_eq!(
            desc.a_selector_column,
            Some(crate::validator_extract::COL_IS_REAL)
        );
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    /// Phase 4 end-to-end: 2-AIR joint_prove for VE ↔ HTR.
    /// Honest case: both AIRs hold the same single validator's
    /// (idx, eb, root) triple → multiset balances → joint_prove
    /// succeeds, joint_verify accepts.
    #[test]
    #[ignore = "slow: 2-AIR VE↔HTR joint_prove (~5s); run with --release --ignored"]
    fn joint_prove_validator_extract_validator_htr_linkage() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            ValidatorExtractConstraintSystem, ValidatorExtractWitness,
        };
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let validator = crate::beacon::Validator {
            pubkey: w.validators[0].pubkey,
            withdrawal_credentials: w.validators[0].withdrawal_credentials,
            effective_balance: w.validators[0].effective_balance,
            slashed: w.validators[0].slashed,
            activation_eligibility_epoch: w.validators[0]
                .activation_eligibility_epoch,
            activation_epoch: w.validators[0].activation_epoch,
            exit_epoch: w.validators[0].exit_epoch,
            withdrawable_epoch: w.validators[0].withdrawable_epoch,
        };

        // VE side mirrors the same validator.
        let ve_w = ValidatorExtractWitness::from_beacon_validators(&[validator]);
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        let htr_trace = build_trace_polynomials(&w, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);

        let l1 = make_validator_extract_validator_htr_linkage_descriptor(0, 1);
        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&ve_trace, &ve_cs), (&htr_trace, &htr_cs)];
        let linkages = vec![l1];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for honest VE↔HTR chain");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        for lp in &extension.linkage_proofs {
            assert_eq!(lp.closure_a, lp.closure_b);
        }
        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&ve_cs, &htr_cs];
        let valid =
            joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(valid, "joint verifier must accept honest VE↔HTR chain");
    }

    /// Tampering test for Phase 4: VE claims an honest validator_root
    /// but a tampered effective_balance. The 34-tuple
    /// `(idx, eb, root)` won't match the gadget's, so the witness
    /// builder rejects.
    #[test]
    #[ignore = "slow: 2-AIR VE↔HTR tampering test (~5s); run with --release --ignored"]
    fn joint_prove_validator_extract_validator_htr_rejects_tampered_eb() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::validator_extract::{
            build_trace_polynomials as build_ve_trace,
            ValidatorExtractConstraintSystem, ValidatorExtractRow,
            ValidatorExtractWitness,
        };
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let v = &w.validators[0];

        // VE TAMPERED: eb mismatches gadget's claim.
        let ve_row = ValidatorExtractRow {
            effective_balance: v.effective_balance + 1, // ← tampered
            validator_root: compute_validator_root(v),
        };
        let ve_w = ValidatorExtractWitness {
            validators: vec![ve_row],
        };
        let ve_trace = build_ve_trace(&ve_w, curve);
        let ve_cs = ValidatorExtractConstraintSystem::new(ve_trace.num_rows);

        let htr_trace = build_trace_polynomials(&w, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);

        let l1 = make_validator_extract_validator_htr_linkage_descriptor(0, 1);
        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&ve_trace, &ve_cs), (&htr_trace, &htr_cs)];
        let linkages = vec![l1];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject VE↔HTR chain when eb differs"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}",
            err
        );
    }

    /// Phase 6 tampering: corrupt LAYER1_PARENT[2] (the third
    /// intermediate parent in the depth-3 tree). The multiset
    /// equality for the layer-1 pair-2 cross-AIR LogUp must fail
    /// at witness-build time, AND so must the layer-2 pair-1
    /// linkage (since LAYER2_PARENT[1] depends on LAYER1_PARENT[2]
    /// and [3] — the gadget's witness builder recomputed it from
    /// the honest values, so layer 2 still uses honest inputs but
    /// the multiset for layer 1 pair 2 is broken).
    ///
    /// Most directly: with LAYER1_PARENT[2] tampered, the layer-1
    /// pair-2 linkage's gadget tuple is `(LEAF[4], LEAF[5],
    /// tampered_parent)` — this won't appear in the honest
    /// Sha256Extract trace, so multiset rejects.
    #[test]
    #[ignore = "slow: 9-AIR tampering test (~30s); run with --release --ignored"]
    fn joint_prove_validator_htr_rejects_tampered_layer1_parent() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let row = &w.validators[0];
        let leaves = compute_leaves(row);
        let layer1: [[u8; CHUNK_BYTES]; NUM_LAYER1_PARENTS] = [
            crate::sha256::sha256_pair(&leaves[0], &leaves[1]),
            crate::sha256::sha256_pair(&leaves[2], &leaves[3]),
            crate::sha256::sha256_pair(&leaves[4], &leaves[5]),
            crate::sha256::sha256_pair(&leaves[6], &leaves[7]),
        ];
        let layer2: [[u8; CHUNK_BYTES]; NUM_LAYER2_PARENTS] = [
            crate::sha256::sha256_pair(&layer1[0], &layer1[1]),
            crate::sha256::sha256_pair(&layer1[2], &layer1[3]),
        ];

        // Pubkey leaf inputs.
        let mut pubkey_chunk_0 = [0u8; 32];
        pubkey_chunk_0.copy_from_slice(&row.pubkey[0..32]);
        let mut pubkey_chunk_1 = [0u8; 32];
        pubkey_chunk_1[..16].copy_from_slice(&row.pubkey[32..48]);

        // Build 8 separate Sha256Extract instances (honest).
        let extract_pairs: [(crate::ssz::Chunk, crate::ssz::Chunk); 8] = [
            (pubkey_chunk_0, pubkey_chunk_1),
            (leaves[0], leaves[1]),
            (leaves[2], leaves[3]),
            (leaves[4], leaves[5]),
            (leaves[6], leaves[7]),
            (layer1[0], layer1[1]),
            (layer1[2], layer1[3]),
            (layer2[0], layer2[1]),
        ];
        let extract_witnesses: Vec<crate::sha256_extract::Sha256ExtractWitness> =
            extract_pairs
                .iter()
                .map(|p| crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(&[*p]))
                .collect();
        let extract_traces: Vec<TracePolynomials> = extract_witnesses
            .iter()
            .map(|w| crate::sha256_extract::build_trace_polynomials(w, curve))
            .collect();
        let extract_constraints: Vec<crate::sha256_extract::Sha256ExtractConstraintSystem> =
            extract_traces
                .iter()
                .map(|t| {
                    crate::sha256_extract::Sha256ExtractConstraintSystem::new(t.num_rows)
                })
                .collect();

        // TAMPERED gadget trace: corrupt LAYER1_PARENT[2] byte 0.
        let mut htr_trace = build_trace_polynomials(&w, curve);
        let prev = htr_trace.columns[col_layer1_parent_byte(2, 0)]
            .evaluations[0]
            .to_u64();
        htr_trace.columns[col_layer1_parent_byte(2, 0)].evaluations[0] =
            Scalar::from_u64(prev ^ 0xFF, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);

        // Use only the layer-1 pair-2 linkage to localise the failure.
        let l1_p2 = make_validator_htr_layer1_pair_linkage_descriptor(2, 0, 1);

        // Build a 2-AIR setup: gadget + just one Sha256Extract layer
        // (the one for layer-1 pair-2).
        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&htr_trace, &htr_cs), (&extract_traces[3], &extract_constraints[3])];
        let linkages = vec![l1_p2];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered LAYER1_PARENT[2]"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}",
            err
        );
    }

    /// Phase 6 tampering: corrupt LAYER2_PARENT[0]. The layer-2
    /// pair-0 linkage must reject.
    #[test]
    #[ignore = "slow: 2-AIR tampering test (~5s); run with --release --ignored"]
    fn joint_prove_validator_htr_rejects_tampered_layer2_parent() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let row = &w.validators[0];
        let leaves = compute_leaves(row);
        let layer1: [[u8; CHUNK_BYTES]; NUM_LAYER1_PARENTS] = [
            crate::sha256::sha256_pair(&leaves[0], &leaves[1]),
            crate::sha256::sha256_pair(&leaves[2], &leaves[3]),
            crate::sha256::sha256_pair(&leaves[4], &leaves[5]),
            crate::sha256::sha256_pair(&leaves[6], &leaves[7]),
        ];
        let extract_w = crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(
            &[(layer1[0], layer1[1])],
        );
        let extract_trace =
            crate::sha256_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = crate::sha256_extract::Sha256ExtractConstraintSystem::new(
            extract_trace.num_rows,
        );

        // TAMPER: LAYER2_PARENT[0] byte 5.
        let mut htr_trace = build_trace_polynomials(&w, curve);
        let prev = htr_trace.columns[col_layer2_parent_byte(0, 5)]
            .evaluations[0]
            .to_u64();
        htr_trace.columns[col_layer2_parent_byte(0, 5)].evaluations[0] =
            Scalar::from_u64(prev ^ 0xFF, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);

        let l2_p0 = make_validator_htr_layer2_pair_linkage_descriptor(0, 0, 1);
        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&htr_trace, &htr_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![l2_p0];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered LAYER2_PARENT[0]"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}",
            err
        );
    }

    /// Phase 6 tampering: corrupt VALIDATOR_ROOT (layer-3 / final).
    /// The layer-3 root linkage must reject.
    #[test]
    #[ignore = "slow: 2-AIR tampering test (~5s); run with --release --ignored"]
    fn joint_prove_validator_htr_rejects_tampered_validator_root() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let row = &w.validators[0];
        let leaves = compute_leaves(row);
        let layer1: [[u8; CHUNK_BYTES]; NUM_LAYER1_PARENTS] = [
            crate::sha256::sha256_pair(&leaves[0], &leaves[1]),
            crate::sha256::sha256_pair(&leaves[2], &leaves[3]),
            crate::sha256::sha256_pair(&leaves[4], &leaves[5]),
            crate::sha256::sha256_pair(&leaves[6], &leaves[7]),
        ];
        let layer2: [[u8; CHUNK_BYTES]; NUM_LAYER2_PARENTS] = [
            crate::sha256::sha256_pair(&layer1[0], &layer1[1]),
            crate::sha256::sha256_pair(&layer1[2], &layer1[3]),
        ];
        let extract_w = crate::sha256_extract::Sha256ExtractWitness::from_pair_inputs(
            &[(layer2[0], layer2[1])],
        );
        let extract_trace =
            crate::sha256_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = crate::sha256_extract::Sha256ExtractConstraintSystem::new(
            extract_trace.num_rows,
        );

        // TAMPER: corrupt VALIDATOR_ROOT byte 17.
        let mut htr_trace = build_trace_polynomials(&w, curve);
        let prev = htr_trace.columns[COL_VALIDATOR_ROOT_BYTE_OFFSET + 17]
            .evaluations[0]
            .to_u64();
        htr_trace.columns[COL_VALIDATOR_ROOT_BYTE_OFFSET + 17].evaluations[0] =
            Scalar::from_u64(prev ^ 0xAA, curve);
        let htr_cs = ValidatorHtrConstraintSystem::new(htr_trace.num_rows);

        let l3 = make_validator_htr_layer3_root_linkage_descriptor(0, 1);
        let traces: Vec<(
            &TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&htr_trace, &htr_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![l3];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject tampered VALIDATOR_ROOT"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}",
            err
        );
    }

    /// Phase 7: tampering coverage for the remaining row-locals.
    /// Each test corrupts ONE column and verifies the corresponding
    /// row-local constraint body becomes non-zero.
    ///
    /// Helper: run `evaluate_on_domain` and assert the named body
    /// fires at row 0 while all OTHER bodies remain zero — this
    /// localises the failure to the expected constraint.
    fn assert_only_body_fires(
        cs: &ValidatorHtrConstraintSystem,
        cols_owned: &[&Vec<Scalar>],
        num_rows: usize,
        firing_body_idx: usize,
        firing_label: &str,
    ) {
        let evals = cs.evaluate_on_domain(cols_owned, num_rows);
        for (k, body) in evals.iter().enumerate() {
            if k == firing_body_idx {
                assert!(
                    !body[0].is_zero(),
                    "expected body {} ({}) to fire at row 0 but it vanished",
                    k,
                    firing_label
                );
            } else {
                assert!(
                    body[0].is_zero(),
                    "body {} ({}) must remain zero (only body {} should fire), got {:?}",
                    k,
                    cs.constraint_labels()[k],
                    firing_body_idx,
                    body[0].to_u64(),
                );
            }
        }
    }

    /// Phase 7: ae_epoch limb binding (body 3).
    #[test]
    fn ae_epoch_limb_binding_rejects_tampered_aggregator() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_AE_EPOCH].evaluations[0] =
            Scalar::from_u64(0xFEEDFACE, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 3, "ae_epoch_limb_binding",
        );
    }

    /// Phase 7: act_epoch limb binding (body 4).
    #[test]
    fn act_epoch_limb_binding_rejects_tampered_aggregator() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_ACT_EPOCH].evaluations[0] =
            Scalar::from_u64(0xCAFEBABE, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 4, "act_epoch_limb_binding",
        );
    }

    /// Phase 7: exit_epoch limb binding (body 5).
    #[test]
    fn exit_epoch_limb_binding_rejects_tampered_aggregator() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_EXIT_EPOCH].evaluations[0] =
            Scalar::from_u64(0xC001D00D, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 5, "exit_epoch_limb_binding",
        );
    }

    /// Phase 7: wd_epoch limb binding (body 6).
    #[test]
    fn wd_epoch_limb_binding_rejects_tampered_aggregator() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_WD_EPOCH].evaluations[0] =
            Scalar::from_u64(0xDEADC0DE, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 6, "wd_epoch_limb_binding",
        );
    }

    /// Phase 7: leaf_1 (withdrawal_credentials) binding (body 7).
    #[test]
    fn leaf_1_binding_rejects_tampered_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        // Corrupt LEAF_1_BYTE[12].
        trace.columns[col_leaf_byte(1, 12)].evaluations[0] =
            Scalar::from_u64(0xAA, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 7,
            "leaf_1_withdrawal_credentials_binding",
        );
    }

    /// Phase 7: leaf_3 (slashed) binding (body 9). Tamper a non-zero
    /// byte into LEAF_3_BYTE[10] (the canonical encoding has zeros
    /// at byte indices 1..32).
    #[test]
    fn leaf_3_binding_rejects_tampered_zero_tail() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[col_leaf_byte(3, 10)].evaluations[0] =
            Scalar::from_u64(0x55, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 9, "leaf_3_slashed_binding",
        );
    }

    /// Phase 7: leaf_4 (ae_epoch) binding (body 10). Corrupt
    /// LEAF_4_BYTE[3] (one of the 8 LE bytes of the value).
    #[test]
    fn leaf_4_binding_rejects_tampered_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[col_leaf_byte(4, 3)].evaluations[0] =
            Scalar::from_u64(0x99, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 10, "leaf_4_ae_epoch_binding",
        );
    }

    /// Phase 7: leaf_5 (act_epoch) binding (body 11).
    #[test]
    fn leaf_5_binding_rejects_tampered_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[col_leaf_byte(5, 5)].evaluations[0] =
            Scalar::from_u64(0x77, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 11, "leaf_5_act_epoch_binding",
        );
    }

    /// Phase 7: leaf_6 (exit_epoch) binding (body 12).
    #[test]
    fn leaf_6_binding_rejects_tampered_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[col_leaf_byte(6, 7)].evaluations[0] =
            Scalar::from_u64(0xBB, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 12, "leaf_6_exit_epoch_binding",
        );
    }

    /// Phase 7: leaf_7 (wd_epoch) binding (body 13).
    #[test]
    fn leaf_7_binding_rejects_tampered_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[col_leaf_byte(7, 6)].evaluations[0] =
            Scalar::from_u64(0xDD, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 13, "leaf_7_wd_epoch_binding",
        );
    }

    /// Phase 7: pubkey_chunk_1 binding (body 14). Corrupt
    /// PUBKEY_CHUNK_1_BYTE[5] (a byte that should equal pubkey[37]).
    #[test]
    fn pubkey_chunk_1_binding_rejects_tampered_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_PUBKEY_CHUNK_1_BYTE_OFFSET + 5].evaluations[0] =
            Scalar::from_u64(0x11, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 14, "pubkey_chunk_1_binding",
        );
    }

    /// Phase 7: pubkey_chunk_1 binding rejects a non-zero byte in
    /// the canonical zero-padding region (bytes 16..32).
    #[test]
    fn pubkey_chunk_1_binding_rejects_nonzero_pad_byte() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_PUBKEY_CHUNK_1_BYTE_OFFSET + 20].evaluations[0] =
            Scalar::from_u64(0xFF, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        assert_only_body_fires(
            &cs, &cols_owned, trace.num_rows, 14, "pubkey_chunk_1_binding",
        );
    }

    /// Tampering: corrupt SLASHED to 2 (non-binary). `slashed_binary`
    /// body must fire.
    #[test]
    fn slashed_binary_rejects_non_binary_value() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_SLASHED].evaluations[0] = Scalar::from_u64(2, curve);
        let cs = ValidatorHtrConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // body 1 is slashed_binary.
        assert!(
            !evals[1][0].is_zero(),
            "non-binary SLASHED must make slashed_binary non-zero"
        );
    }
}
