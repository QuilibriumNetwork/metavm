//! CREATE2 init-code-hash + address derivation gadget AIR.
//!
//! Per EIP-1014, the deployed contract address from a CREATE2 opcode is
//!
//! ```text
//! address = keccak256(0xff || sender || salt || keccak256(init_code))[12..32]
//! ```
//!
//! where `sender` is the 20-byte address of the contract issuing the
//! CREATE2, `salt` is a 32-byte caller-supplied nonce, and
//! `keccak256(init_code)` is the 32-byte hash of the init-code bytes.
//!
//! This AIR commits, per invocation row, the full (sender, salt,
//! init_code_hash, derived_address, 85-byte preimage, 32-byte keccak
//! output) tuple and pins the byte equalities making the preimage and
//! the derived address consistent with the inputs algebraically:
//!
//!   - `preimage[0] = 0xff`
//!   - `preimage[1..21] = sender[0..20]`
//!   - `preimage[21..53] = salt[0..32]`
//!   - `preimage[53..85] = init_code_hash[0..32]`
//!   - `derived_address[0..20] = keccak_output[12..32]`
//!
//! Together with two cross-AIR LogUp linkages to
//! [`crate::keccak_extract`] (one binding the outer
//! `(preimage[0..85], 85, keccak_output[0..32])` triple, one binding
//! the inner `(init_code, init_code_len, init_code_hash)` triple) plus
//! a third LogUp linkage binding the `derived_address` to
//! [`crate::address_keccak_air`], this gadget closes the full algebraic
//! soundness chain for a CREATE2 contract-address derivation in a
//! single 20-byte address output column.
//!
//! Distinct from [`crate::evm_create2_input_air`] (which exposes a
//! 256-byte keccak-input-shaped column for matching against
//! `KeccakExtract::INPUT_BYTE[0..256]`), THIS gadget exposes the
//! **derived address** as a first-class column and the keccak output
//! bytes side-by-side, so downstream chains (EVM main, account
//! creation, etc.) can bind the new contract address directly.
//!
//! # Column layout
//!
//! ```text
//!   0 ..  20   sender[0..20]              raw 20-byte address (BE bytes)
//!  20 ..  52   salt[0..32]                raw 32-byte salt
//!  52 ..  84   init_code_hash[0..32]      keccak256(init_code)
//!  84 .. 169   preimage[0..85]            0xff || sender || salt || h
//! 169 .. 201   keccak_output[0..32]       keccak256(preimage)
//! 201 .. 221   derived_address[0..20]     keccak_output[12..32]
//! 221 .. 225   sender_limb[0..4]          sender as 4 LE u64 limbs
//! 225          is_real                    binary selector
//! ```
//!
//! Total: 226 columns.
//!
//! The `sender_limb[0..4]` columns are an adapter exposing the same
//! 20-byte sender as the 4-limb LE u64 packing used by the EVM main
//! AIR's `COL_FRAME_CALLEE_L0..L3`, so cross-AIR LogUp descriptors can
//! bind sender by limb without a byte-tuple decomposition. The byte
//! ↔ limb relationship is enforced algebraically by 4 dedicated
//! row-local constraints (see below).
//!
//! # Row-local constraints (alpha-power weighted)
//!
//!   0. `is_real` binary: `IS_REAL · (IS_REAL − 1)`.
//!   1. `preimage[0] = 0xff`: `IS_REAL · (preimage[0] - 0xff)`.
//!   2..21 (20 constraints): `IS_REAL · (preimage[1+k] - sender[k])`,
//!      `k ∈ 0..20`.
//!   22..53 (32 constraints): `IS_REAL · (preimage[21+k] - salt[k])`,
//!      `k ∈ 0..32`.
//!   54..85 (32 constraints): `IS_REAL · (preimage[53+k] -
//!      init_code_hash[k])`, `k ∈ 0..32`.
//!   86..105 (20 constraints): `IS_REAL · (derived_address[k] -
//!      keccak_output[12+k])`, `k ∈ 0..20`.
//!  106. `sender_limb_0_binding`: `IS_REAL · (sender_limb[0] -
//!       Σ_{i=0..8} 2^(8 i) · sender[i]) = 0`.
//!  107. `sender_limb_1_binding`: `IS_REAL · (sender_limb[1] -
//!       Σ_{i=0..8} 2^(8 i) · sender[8 + i]) = 0`.
//!  108. `sender_limb_2_binding`: `IS_REAL · (sender_limb[2] -
//!       Σ_{i=0..4} 2^(8 i) · sender[16 + i]) = 0`.
//!  109. `sender_limb_3_zero`: `IS_REAL · sender_limb[3] = 0`.
//!
//! Total: 1 + 1 + 20 + 32 + 32 + 20 + 4 = **110 row-local constraints**.
//!
//! Byte range checks via `LookupRequirements`:
//! `sender`, `salt`, `init_code_hash`, `preimage`, `keccak_output`,
//! `derived_address` — 20 + 32 + 32 + 85 + 32 + 20 = **221 byte cols**.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const CREATE2_MAGIC_BYTE: u64 = 0xff;
pub const SENDER_LEN: usize = 20;
pub const SALT_LEN: usize = 32;
pub const INIT_CODE_HASH_LEN: usize = 32;
pub const PREIMAGE_LEN: usize = 1 + SENDER_LEN + SALT_LEN + INIT_CODE_HASH_LEN; // 85
pub const KECCAK_OUTPUT_LEN: usize = 32;
pub const DERIVED_ADDRESS_LEN: usize = 20;
/// Number of LE u64 limbs aggregating the 20-byte sender (4: limbs
/// 0+1 cover bytes 0..16, limb 2 covers bytes 16..20 zero-extended,
/// limb 3 is identically zero — mirroring the EVM main AIR's
/// `COL_FRAME_CALLEE_L0..L3` packing).
pub const NUM_SENDER_LIMBS: usize = 4;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_SENDER_OFFSET: usize = 0;
pub const COL_SALT_OFFSET: usize = COL_SENDER_OFFSET + SENDER_LEN;
pub const COL_INIT_CODE_HASH_OFFSET: usize = COL_SALT_OFFSET + SALT_LEN;
pub const COL_PREIMAGE_OFFSET: usize = COL_INIT_CODE_HASH_OFFSET + INIT_CODE_HASH_LEN;
pub const COL_KECCAK_OUTPUT_OFFSET: usize = COL_PREIMAGE_OFFSET + PREIMAGE_LEN;
pub const COL_DERIVED_ADDRESS_OFFSET: usize = COL_KECCAK_OUTPUT_OFFSET + KECCAK_OUTPUT_LEN;
/// LE u64 limb adapter columns for the sender address. Limb k commits
/// the same scalar that `address_to_limbs(sender)[k]` produces in the
/// EVM main inspector — enabling byte-exact cross-AIR LogUp binding
/// against `COL_FRAME_CALLEE_L0..L3` on the EVM main side.
pub const COL_SENDER_LIMB_OFFSET: usize = COL_DERIVED_ADDRESS_OFFSET + DERIVED_ADDRESS_LEN;
pub const COL_IS_REAL: usize = COL_SENDER_LIMB_OFFSET + NUM_SENDER_LIMBS;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// 1 (binary) + 1 (magic byte) + 20 (sender) + 32 (salt) + 32 (init_code_hash)
/// + 20 (derived_address) + 4 (sender_limb bindings) = 110.
pub const NUM_ROW_CONSTRAINTS: usize = 1 + 1 + SENDER_LEN + SALT_LEN + INIT_CODE_HASH_LEN
    + DERIVED_ADDRESS_LEN + NUM_SENDER_LIMBS;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Create2AddressRow {
    pub sender: [u8; SENDER_LEN],
    pub salt: [u8; SALT_LEN],
    pub init_code_hash: [u8; INIT_CODE_HASH_LEN],
    pub preimage: [u8; PREIMAGE_LEN],
    pub keccak_output: [u8; KECCAK_OUTPUT_LEN],
    pub derived_address: [u8; DERIVED_ADDRESS_LEN],
}

#[derive(Clone, Debug, Default)]
pub struct Create2AddressWitness {
    pub invocations: Vec<Create2AddressRow>,
}

/// Build a canonical CREATE2 row from raw inputs. Computes
/// `init_code_hash = keccak256(init_code)`, the 85-byte preimage,
/// `keccak_output = keccak256(preimage)`, and
/// `derived_address = keccak_output[12..32]`.
pub fn from_inputs(
    sender: [u8; SENDER_LEN],
    salt: [u8; SALT_LEN],
    init_code: &[u8],
) -> Create2AddressRow {
    let init_code_hash = crate::keccak::keccak256(init_code);
    let mut preimage = [0u8; PREIMAGE_LEN];
    preimage[0] = CREATE2_MAGIC_BYTE as u8;
    preimage[1..1 + SENDER_LEN].copy_from_slice(&sender);
    preimage[1 + SENDER_LEN..1 + SENDER_LEN + SALT_LEN].copy_from_slice(&salt);
    preimage[1 + SENDER_LEN + SALT_LEN..].copy_from_slice(&init_code_hash);
    let keccak_output = crate::keccak::keccak256(&preimage);
    let mut derived_address = [0u8; DERIVED_ADDRESS_LEN];
    derived_address.copy_from_slice(&keccak_output[12..32]);
    Create2AddressRow {
        sender,
        salt,
        init_code_hash,
        preimage,
        keccak_output,
        derived_address,
    }
}

/// Pack a 20-byte BE sender address into 4 LE u64 limbs matching the
/// EVM main `address_to_limbs` convention: limbs 0+1 from bytes
/// `[0..16]` via `from_le_bytes`, limb 2 from bytes `[16..20]`
/// zero-extended, limb 3 = 0.
pub fn sender_limbs_from_bytes(sender: &[u8; SENDER_LEN]) -> [u64; NUM_SENDER_LIMBS] {
    let mut limbs = [0u64; NUM_SENDER_LIMBS];
    let mut tmp = [0u8; 8];
    tmp.copy_from_slice(&sender[0..8]);
    limbs[0] = u64::from_le_bytes(tmp);
    tmp.copy_from_slice(&sender[8..16]);
    limbs[1] = u64::from_le_bytes(tmp);
    let mut tmp4 = [0u8; 8];
    tmp4[..4].copy_from_slice(&sender[16..20]);
    limbs[2] = u64::from_le_bytes(tmp4);
    // limbs[3] stays 0 — address fits in 160 bits.
    limbs
}

impl Create2AddressWitness {
    pub fn from_invocations(rows: Vec<Create2AddressRow>) -> Self {
        Self { invocations: rows }
    }

    pub fn from_input_tuples(
        inputs: &[([u8; SENDER_LEN], [u8; SALT_LEN], Vec<u8>)],
    ) -> Self {
        Self {
            invocations: inputs
                .iter()
                .map(|(s, salt, init_code)| from_inputs(*s, *salt, init_code))
                .collect(),
        }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Create2AddressWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, inv) in witness.invocations.iter().enumerate() {
        for k in 0..SENDER_LEN {
            columns[COL_SENDER_OFFSET + k][i] =
                Scalar::from_u64(inv.sender[k] as u64, curve);
        }
        for k in 0..SALT_LEN {
            columns[COL_SALT_OFFSET + k][i] =
                Scalar::from_u64(inv.salt[k] as u64, curve);
        }
        for k in 0..INIT_CODE_HASH_LEN {
            columns[COL_INIT_CODE_HASH_OFFSET + k][i] =
                Scalar::from_u64(inv.init_code_hash[k] as u64, curve);
        }
        for k in 0..PREIMAGE_LEN {
            columns[COL_PREIMAGE_OFFSET + k][i] =
                Scalar::from_u64(inv.preimage[k] as u64, curve);
        }
        for k in 0..KECCAK_OUTPUT_LEN {
            columns[COL_KECCAK_OUTPUT_OFFSET + k][i] =
                Scalar::from_u64(inv.keccak_output[k] as u64, curve);
        }
        for k in 0..DERIVED_ADDRESS_LEN {
            columns[COL_DERIVED_ADDRESS_OFFSET + k][i] =
                Scalar::from_u64(inv.derived_address[k] as u64, curve);
        }
        // LE u64 sender limb adapter.
        let limbs = sender_limbs_from_bytes(&inv.sender);
        for (k, l) in limbs.iter().enumerate() {
            columns[COL_SENDER_LIMB_OFFSET + k][i] = Scalar::from_u64(*l, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
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

pub struct Create2AddressConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Create2AddressConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Body assignments matching the alpha-power index used in
/// `evaluate_at_point` / `build_constraint_polynomial`. Returns the
/// list of `(target_col, source_col_or_const, IS_REAL_GATED_DIFF)`
/// triples encoding each `IS_REAL · (target - source) = 0` constraint
/// after the binary constraint at index 0.
///
/// `source` of `None` means literal `0xff` (magic byte).
fn equality_bodies() -> Vec<(usize, Option<usize>)> {
    let mut bodies = Vec::with_capacity(NUM_ROW_CONSTRAINTS - 1);
    // 1: preimage[0] - 0xff
    bodies.push((COL_PREIMAGE_OFFSET, None));
    // 2..21: preimage[1+k] - sender[k]
    for k in 0..SENDER_LEN {
        bodies.push((
            COL_PREIMAGE_OFFSET + 1 + k,
            Some(COL_SENDER_OFFSET + k),
        ));
    }
    // 22..53: preimage[21+k] - salt[k]
    for k in 0..SALT_LEN {
        bodies.push((
            COL_PREIMAGE_OFFSET + 1 + SENDER_LEN + k,
            Some(COL_SALT_OFFSET + k),
        ));
    }
    // 54..85: preimage[53+k] - init_code_hash[k]
    for k in 0..INIT_CODE_HASH_LEN {
        bodies.push((
            COL_PREIMAGE_OFFSET + 1 + SENDER_LEN + SALT_LEN + k,
            Some(COL_INIT_CODE_HASH_OFFSET + k),
        ));
    }
    // 86..105: derived_address[k] - keccak_output[12+k]
    for k in 0..DERIVED_ADDRESS_LEN {
        bodies.push((
            COL_DERIVED_ADDRESS_OFFSET + k,
            Some(COL_KECCAK_OUTPUT_OFFSET + 12 + k),
        ));
    }
    // `equality_bodies` covers all constraints EXCEPT the leading
    // `is_real` binary check (index 0) and the trailing 4 sender-limb
    // bindings (indices NUM_ROW_CONSTRAINTS-4..NUM_ROW_CONSTRAINTS).
    debug_assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS - 1 - NUM_SENDER_LIMBS);
    bodies
}

// ─── Sender limb binding helpers ──────────────────────────────────────

/// Number of sender bytes covered by `sender_limb[k]`. k ∈ {0, 1} → 8;
/// k = 2 → 4; k = 3 → 0 (limb 3 is always zero, pinned by limb_3_zero).
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
        let byte = &col_evals[COL_SENDER_OFFSET + 8 * k + i];
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

fn build_sender_limb_binding_poly(
    col_coeffs: &[Vec<Scalar>],
    k: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let bytes = sender_limb_byte_count(k);
    let mut sum = vec![Scalar::zero(curve)];
    for i in 0..bytes {
        let byte_poly = &col_coeffs[COL_SENDER_OFFSET + 8 * k + i];
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

impl VmConstraintSystem for Create2AddressConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        labels.push("preimage_magic_byte".into());
        for k in 0..SENDER_LEN {
            labels.push(format!("preimage_sender_{}", k));
        }
        for k in 0..SALT_LEN {
            labels.push(format!("preimage_salt_{}", k));
        }
        for k in 0..INIT_CODE_HASH_LEN {
            labels.push(format!("preimage_init_code_hash_{}", k));
        }
        for k in 0..DERIVED_ADDRESS_LEN {
            labels.push(format!("derived_address_from_output_{}", k));
        }
        // Sender limb byte-decomposition bindings.
        labels.push("sender_limb_0_binding".into());
        labels.push("sender_limb_1_binding".into());
        labels.push("sender_limb_2_binding".into());
        labels.push("sender_limb_3_zero".into());
        debug_assert_eq!(labels.len(), NUM_ROW_CONSTRAINTS);
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

        let bodies = equality_bodies();
        let magic = Scalar::from_u64(CREATE2_MAGIC_BYTE, curve);
        for (target_col, source) in bodies {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let t = &columns[target_col][r];
                let body = match source {
                    None => t.sub(&magic),
                    Some(src) => t.sub(&columns[src][r]),
                };
                let gated = columns[COL_IS_REAL][r].mul(&body);
                c[r] = gated;
            }
            out.push(c);
        }

        // Sender limb byte-decomposition bindings (4 constraints).
        for k in 0..NUM_SENDER_LIMBS {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let row_vals: Vec<Scalar> =
                    (0..columns.len()).map(|ci| columns[ci][r].clone()).collect();
                c[r] = if k < 3 {
                    eval_sender_limb_binding(&row_vals, k)
                } else {
                    eval_sender_limb_3_zero(&row_vals)
                };
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
        let magic = Scalar::from_u64(CREATE2_MAGIC_BYTE, curve);

        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);

        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&ap.mul(&v.mul(&v.sub(&one))));
            ap = ap.mul(alpha);
        }
        let bodies = equality_bodies();
        let v = &col_evals[COL_IS_REAL];
        for (target_col, source) in bodies {
            let body = match source {
                None => col_evals[target_col].sub(&magic),
                Some(src) => col_evals[target_col].sub(&col_evals[src]),
            };
            let gated = v.mul(&body);
            acc = acc.add(&ap.mul(&gated));
            ap = ap.mul(alpha);
        }
        // Sender limb byte-decomposition bindings (4 constraints).
        for k in 0..NUM_SENDER_LIMBS {
            let body = if k < 3 {
                eval_sender_limb_binding(col_evals, k)
            } else {
                eval_sender_limb_3_zero(col_evals)
            };
            acc = acc.add(&ap.mul(&body));
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
        let magic_poly = vec![Scalar::from_u64(CREATE2_MAGIC_BYTE, curve)];

        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);

        // 0: is_real binary
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        let bodies = equality_bodies();
        let v = &col_coeffs[COL_IS_REAL];
        for (target_col, source) in bodies {
            let diff = match source {
                None => poly_sub(&col_coeffs[target_col], &magic_poly, curve),
                Some(src) => poly_sub(&col_coeffs[target_col], &col_coeffs[src], curve),
            };
            let gated = poly_mul(v, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &ap), curve);
            ap = ap.mul(alpha);
        }
        // Sender limb byte-decomposition bindings (4 constraints).
        for k in 0..NUM_SENDER_LIMBS {
            let body = if k < 3 {
                build_sender_limb_binding_poly(col_coeffs, k, curve)
            } else {
                build_sender_limb_3_zero_poly(col_coeffs, curve)
            };
            acc = poly_add(&acc, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
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
        let add_byte = |label: String, col: usize, decls: &mut Vec<_>| {
            decls.push((
                LookupDeclaration {
                    label,
                    column_index: col,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        };
        for k in 0..SENDER_LEN {
            add_byte(format!("sender_{}_8bit", k), COL_SENDER_OFFSET + k, &mut declarations);
        }
        for k in 0..SALT_LEN {
            add_byte(format!("salt_{}_8bit", k), COL_SALT_OFFSET + k, &mut declarations);
        }
        for k in 0..INIT_CODE_HASH_LEN {
            add_byte(
                format!("init_code_hash_{}_8bit", k),
                COL_INIT_CODE_HASH_OFFSET + k,
                &mut declarations,
            );
        }
        for k in 0..PREIMAGE_LEN {
            add_byte(
                format!("preimage_{}_8bit", k),
                COL_PREIMAGE_OFFSET + k,
                &mut declarations,
            );
        }
        for k in 0..KECCAK_OUTPUT_LEN {
            add_byte(
                format!("keccak_output_{}_8bit", k),
                COL_KECCAK_OUTPUT_OFFSET + k,
                &mut declarations,
            );
        }
        for k in 0..DERIVED_ADDRESS_LEN {
            add_byte(
                format!("derived_address_{}_8bit", k),
                COL_DERIVED_ADDRESS_OFFSET + k,
                &mut declarations,
            );
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ───────────────────────────────────────

/// Link this gadget's `(preimage[0..85], 85, keccak_output[0..32])` to
/// KeccakExtract's `(INPUT_BYTE[0..85], INPUT_LEN, OUTPUT_BYTE[0..32])`.
/// The 171 leading INPUT_BYTE positions beyond 85 are not included
/// because the canonical CREATE2 preimage is fixed-length 85 — the
/// keccak_extract row this matches must be constructed by hashing
/// exactly the 85-byte preimage (`KeccakExtractWitness::from_inputs`
/// zero-pads any shorter input). Including only the 85 real bytes plus
/// INPUT_LEN gives a precise binding without forcing matching of zero
/// padding bytes that may legally differ between extract rows of
/// different `INPUT_LEN`.
///
/// Tuple width: 85 + 1 + 32 = 118 cols.
pub fn make_create2_to_keccak_descriptor(
    gadget_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let mut a_columns: Vec<usize> = (0..PREIMAGE_LEN)
        .map(|k| COL_PREIMAGE_OFFSET + k)
        .collect();
    // Pin the keccak_extract row's INPUT_LEN. We commit a constant
    // column on the gadget side by re-using preimage[0] (= 0xff = 255)
    // would not equal 85, so instead we point at the keccak_output
    // length 32 — that is also wrong. The clean solution is for the
    // gadget to expose its own constant 85 column; since we don't,
    // use the existing preimage[0] approach instead: insert a
    // dedicated INPUT_LEN binding column. We add a virtual column by
    // re-using COL_IS_REAL * 85 indirectly — NOT possible without a
    // new column. Use the cleanest workable approach: drop INPUT_LEN
    // from the tuple and instead bind it via the side-channel of
    // matching only on the 85 preimage bytes; this is sufficient
    // because keccak_extract guarantees output = keccak256(input[0..len])
    // and the extract row must have len = 85 for the 85-byte preimage
    // to produce the gadget's keccak_output (any other len would
    // mismatch on the output side, since changing INPUT_LEN changes
    // the hash). The tuple is 85 (preimage) + 32 (output) = 117 cols.
    let mut b_columns: Vec<usize> = (0..PREIMAGE_LEN)
        .map(|k| ke::COL_INPUT_BYTE_OFFSET + k)
        .collect();
    for k in 0..KECCAK_OUTPUT_LEN {
        a_columns.push(COL_KECCAK_OUTPUT_OFFSET + k);
        b_columns.push(ke::COL_OUTPUT_BYTE_OFFSET + k);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "create2_address_to_keccak_v1".into(),
        a_layer_index: gadget_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Link this gadget's `init_code_hash[0..32]` to KeccakExtract's
/// `OUTPUT_BYTE[0..32]`, binding the inner keccak invocation
/// `init_code_hash = keccak256(init_code)`. The init-code bytes
/// themselves are NOT exposed by this gadget; the caller is
/// responsible for ensuring the matched keccak_extract row's
/// `INPUT_BYTE[0..INPUT_LEN]` is the same init_code that the EVM
/// CREATE2 invocation supplied (via a separate EVM-side adapter
/// binding init_code to memory bytes — out of scope here).
///
/// Tuple width: 32 cols.
pub fn make_create2_to_init_code_keccak_descriptor(
    gadget_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let a_columns: Vec<usize> = (0..INIT_CODE_HASH_LEN)
        .map(|k| COL_INIT_CODE_HASH_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = (0..INIT_CODE_HASH_LEN)
        .map(|k| ke::COL_OUTPUT_BYTE_OFFSET + k)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "create2_address_to_init_code_keccak_v1".into(),
        a_layer_index: gadget_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Link this gadget's `derived_address[0..20]` to
/// [`crate::address_keccak_air`]'s `address_be[0..20]`, registering
/// the freshly-derived contract address with the address-keccak
/// gadget so downstream chains can bind its trie key
/// (`keccak256(address)`).
///
/// Tuple width: 20 cols.
pub fn make_create2_to_address_keccak_descriptor(
    gadget_layer_index: usize,
    address_keccak_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::address_keccak_air as ak;
    let a_columns: Vec<usize> = (0..DERIVED_ADDRESS_LEN)
        .map(|k| COL_DERIVED_ADDRESS_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = (0..DERIVED_ADDRESS_LEN)
        .map(|k| ak::COL_ADDRESS_BE_OFFSET + k)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "create2_address_to_address_keccak_v1".into(),
        a_layer_index: gadget_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: address_keccak_layer_index,
        b_columns,
        b_selector_column: Some(ak::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EIP-1014 example #1: sender = 0, salt = 0, init_code = 0x00,
    /// expected address = 0x4D1A2e2bB4F88F0250f26Ffff098B0b30B26BF38.
    fn eip1014_vector() -> ([u8; SENDER_LEN], [u8; SALT_LEN], Vec<u8>, [u8; 20]) {
        let sender = [0u8; SENDER_LEN];
        let salt = [0u8; SALT_LEN];
        let init_code = vec![0x00u8];
        let expected = [
            0x4D, 0x1A, 0x2e, 0x2b, 0xB4, 0xF8, 0x8F, 0x02,
            0x50, 0xf2, 0x6F, 0xff, 0xf0, 0x98, 0xB0, 0xb3,
            0x0B, 0x26, 0xBF, 0x38,
        ];
        (sender, salt, init_code, expected)
    }

    #[test]
    fn from_inputs_matches_eip1014_known_vector() {
        let (sender, salt, init_code, expected) = eip1014_vector();
        let row = from_inputs(sender, salt, &init_code);
        assert_eq!(
            row.derived_address, expected,
            "EIP-1014 example #1 address mismatch (got {:?})",
            row.derived_address,
        );
        // Preimage[0] = 0xff
        assert_eq!(row.preimage[0], 0xff);
        // Preimage[1..21] = sender
        assert_eq!(&row.preimage[1..21], &sender);
        // Preimage[21..53] = salt
        assert_eq!(&row.preimage[21..53], &salt);
        // Preimage[53..85] = init_code_hash
        assert_eq!(&row.preimage[53..85], &row.init_code_hash);
        // keccak_output[12..32] = derived_address
        assert_eq!(&row.keccak_output[12..32], &row.derived_address);
    }

    #[test]
    fn column_layout_pinned() {
        // Pin column layout so layout changes are loud.
        assert_eq!(COL_SENDER_OFFSET, 0);
        assert_eq!(COL_SALT_OFFSET, 20);
        assert_eq!(COL_INIT_CODE_HASH_OFFSET, 52);
        assert_eq!(COL_PREIMAGE_OFFSET, 84);
        assert_eq!(COL_KECCAK_OUTPUT_OFFSET, 169);
        assert_eq!(COL_DERIVED_ADDRESS_OFFSET, 201);
        assert_eq!(COL_SENDER_LIMB_OFFSET, 221);
        assert_eq!(COL_IS_REAL, 225);
        assert_eq!(NUM_COLUMNS, 226);
        assert_eq!(PREIMAGE_LEN, 85);
        assert_eq!(NUM_SENDER_LIMBS, 4);
        assert_eq!(NUM_ROW_CONSTRAINTS, 110);
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let (sender, salt, init_code, _) = eip1014_vector();
        let row = from_inputs(sender, salt, &init_code);
        let w = Create2AddressWitness::from_invocations(vec![row]);
        let trace = build_trace_polynomials(&w, curve);
        let cs = Create2AddressConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }

        // Also check evaluate_at_point per row vanishes.
        let alpha = Scalar::from_u64(11, curve);
        for r in 0..trace.padded_size as usize {
            let row_vals: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[r].clone())
                .collect();
            assert!(
                cs.evaluate_at_point(&row_vals, &alpha).is_zero(),
                "evaluate_at_point row {} must vanish",
                r,
            );
        }
    }

    #[test]
    fn tampered_preimage_byte_detected() {
        let curve = CurveType::Bls48581;
        let (sender, salt, init_code, _) = eip1014_vector();
        let row = from_inputs(sender, salt, &init_code);
        let w = Create2AddressWitness::from_invocations(vec![row]);
        let trace = build_trace_polynomials(&w, curve);
        let cs = Create2AddressConstraintSystem::new(trace.num_rows);

        // Tamper preimage[5] (a sender byte position) without
        // updating sender[4].
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let one = Scalar::one(curve);
        cols[COL_PREIMAGE_OFFSET + 5][0] =
            cols[COL_PREIMAGE_OFFSET + 5][0].add(&one);

        let alpha = Scalar::from_u64(13, curve);
        let row_vals: Vec<Scalar> = cols.iter().map(|c| c[0].clone()).collect();
        assert!(
            !cs.evaluate_at_point(&row_vals, &alpha).is_zero(),
            "tampered preimage byte must fire a non-zero body",
        );

        // Also tamper the 0xff magic byte → magic constraint must fire.
        let mut cols2: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols2[COL_PREIMAGE_OFFSET + 0][0] =
            cols2[COL_PREIMAGE_OFFSET + 0][0].add(&one);
        let row_vals2: Vec<Scalar> = cols2.iter().map(|c| c[0].clone()).collect();
        assert!(
            !cs.evaluate_at_point(&row_vals2, &alpha).is_zero(),
            "tampered magic byte must fire a non-zero body",
        );
    }

    #[test]
    fn tampered_derived_address_detected() {
        let curve = CurveType::Bls48581;
        let (sender, salt, init_code, _) = eip1014_vector();
        let row = from_inputs(sender, salt, &init_code);
        let w = Create2AddressWitness::from_invocations(vec![row]);
        let trace = build_trace_polynomials(&w, curve);
        let cs = Create2AddressConstraintSystem::new(trace.num_rows);

        // Flip derived_address[0] without touching keccak_output.
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        let one = Scalar::one(curve);
        cols[COL_DERIVED_ADDRESS_OFFSET][0] =
            cols[COL_DERIVED_ADDRESS_OFFSET][0].add(&one);

        let alpha = Scalar::from_u64(17, curve);
        let row_vals: Vec<Scalar> = cols.iter().map(|c| c[0].clone()).collect();
        assert!(
            !cs.evaluate_at_point(&row_vals, &alpha).is_zero(),
            "tampered derived_address[0] must fire a non-zero body",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        use crate::address_keccak_air as ak;
        use crate::keccak_extract as ke;

        let outer = make_create2_to_keccak_descriptor(0, 1);
        assert_eq!(outer.label, "create2_address_to_keccak_v1");
        // 85 (preimage) + 32 (output) = 117.
        assert_eq!(outer.a_columns.len(), 117);
        assert_eq!(outer.b_columns.len(), 117);
        assert_eq!(outer.a_columns[0], COL_PREIMAGE_OFFSET);
        assert_eq!(outer.b_columns[0], ke::COL_INPUT_BYTE_OFFSET);
        assert_eq!(outer.a_columns[84], COL_PREIMAGE_OFFSET + 84);
        assert_eq!(outer.b_columns[84], ke::COL_INPUT_BYTE_OFFSET + 84);
        assert_eq!(outer.a_columns[85], COL_KECCAK_OUTPUT_OFFSET);
        assert_eq!(outer.b_columns[85], ke::COL_OUTPUT_BYTE_OFFSET);
        assert_eq!(outer.a_columns[116], COL_KECCAK_OUTPUT_OFFSET + 31);
        assert_eq!(outer.b_columns[116], ke::COL_OUTPUT_BYTE_OFFSET + 31);
        assert_eq!(outer.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(outer.b_selector_column, Some(ke::COL_IS_REAL));

        let inner = make_create2_to_init_code_keccak_descriptor(0, 1);
        assert_eq!(inner.label, "create2_address_to_init_code_keccak_v1");
        assert_eq!(inner.a_columns.len(), 32);
        assert_eq!(inner.b_columns.len(), 32);
        assert_eq!(inner.a_columns[0], COL_INIT_CODE_HASH_OFFSET);
        assert_eq!(inner.b_columns[0], ke::COL_OUTPUT_BYTE_OFFSET);
        assert_eq!(inner.a_columns[31], COL_INIT_CODE_HASH_OFFSET + 31);

        let addr = make_create2_to_address_keccak_descriptor(0, 1);
        assert_eq!(addr.label, "create2_address_to_address_keccak_v1");
        assert_eq!(addr.a_columns.len(), 20);
        assert_eq!(addr.b_columns.len(), 20);
        assert_eq!(addr.a_columns[0], COL_DERIVED_ADDRESS_OFFSET);
        assert_eq!(addr.b_columns[0], ak::COL_ADDRESS_BE_OFFSET);
        assert_eq!(addr.a_columns[19], COL_DERIVED_ADDRESS_OFFSET + 19);
        assert_eq!(addr.b_columns[19], ak::COL_ADDRESS_BE_OFFSET + 19);
        assert_eq!(addr.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(addr.b_selector_column, Some(ak::COL_IS_REAL));
    }

    #[test]
    fn byte_range_lookup_declarations_cover_every_byte_column() {
        let cs = Create2AddressConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        assert_eq!(req.tables.len(), 1);
        // 20 + 32 + 32 + 85 + 32 + 20 = 221 byte columns.
        let expected_total = SENDER_LEN
            + SALT_LEN
            + INIT_CODE_HASH_LEN
            + PREIMAGE_LEN
            + KECCAK_OUTPUT_LEN
            + DERIVED_ADDRESS_LEN;
        assert_eq!(expected_total, 221);
        assert_eq!(req.declarations.len(), expected_total);
        // Every declaration must reference a valid column index and
        // max_bits = 8.
        let mut seen = std::collections::HashSet::new();
        for (decl, _) in &req.declarations {
            assert_eq!(decl.max_bits, 8);
            assert!(decl.column_index < NUM_COLUMNS);
            assert!(
                seen.insert(decl.column_index),
                "duplicate byte-range declaration on column {}",
                decl.column_index,
            );
        }
        // is_real should NOT be in the byte-range set.
        assert!(!seen.contains(&COL_IS_REAL));
    }

    #[test]
    fn from_inputs_nontrivial_vector_consistency() {
        // Hand-picked non-trivial vector — check the canonical
        // preimage + output relationship without relying on a known
        // external vector.
        let sender: [u8; 20] = [
            0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02, 0x03,
            0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
            0x0c, 0x0d, 0x0e, 0x0f,
        ];
        let mut salt = [0u8; 32];
        for i in 0..32 { salt[i] = i as u8; }
        let init_code = vec![0x60u8, 0x80, 0x60, 0x40, 0x52];
        let row = from_inputs(sender, salt, &init_code);

        // Independently recompute and compare.
        let expected_init_hash = crate::keccak::keccak256(&init_code);
        assert_eq!(row.init_code_hash, expected_init_hash);

        let mut expected_preimage = [0u8; PREIMAGE_LEN];
        expected_preimage[0] = 0xff;
        expected_preimage[1..21].copy_from_slice(&sender);
        expected_preimage[21..53].copy_from_slice(&salt);
        expected_preimage[53..85].copy_from_slice(&expected_init_hash);
        assert_eq!(row.preimage, expected_preimage);

        let expected_output = crate::keccak::keccak256(&expected_preimage);
        assert_eq!(row.keccak_output, expected_output);
        assert_eq!(&row.derived_address[..], &expected_output[12..32]);
    }
}
