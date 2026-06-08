//! ECRECOVER precompile (0x01) full-chain AIR.
//!
//! Per row, commits one ECRECOVER precompile call's full input → output
//! chain:
//!
//!   1. The 128-byte input is parsed into `(sig_hash[32], v[32 LSB =
//!      v_byte], r[32], s[32])` (Ethereum yellow paper layout).
//!   2. `(v_byte, r, s)` plus `sig_hash` ECDSA-recover an uncompressed
//!      secp256k1 public key. Only the 64 `X || Y` bytes are committed
//!      (the 0x04 prefix is dropped).
//!   3. `pubkey_keccak = keccak256(X || Y)`.
//!   4. `recovered_address = pubkey_keccak[12..32]` (20 bytes).
//!   5. The 32-byte output is `0x00..00 || recovered_address[20]` on
//!      success (`is_valid = 1`) or all-zero on failure (`is_valid =
//!      0`). Failure occurs when `v ∉ {27, 28}`, when `(r, s)` are out
//!      of range, or when recovery itself fails.
//!
//! Each step is bound algebraically via a cross-AIR LogUp linkage to
//! the AIR that already proves it:
//!
//!   - [`make_ecrecover_to_recovery_descriptor`] —
//!     `(sig_hash, r, s, v_byte, X, Y)` ↔
//!     [`crate::secp256k1_recovery::recovery_air`]'s same tuple. The
//!     descriptor is gated by `is_valid` (failure rows do NOT publish a
//!     valid recovered pubkey).
//!   - [`make_ecrecover_to_keccak_descriptor`] —
//!     `(X || Y, pubkey_keccak)` ↔ [`crate::keccak_extract`]'s
//!     `(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])`. Combined with
//!     KeccakExtract's own constraint system this proves
//!     `pubkey_keccak = keccak256(X || Y)`.
//!   - [`make_ecrecover_to_precompile_dispatch_descriptor`] — binds
//!     `(input_length = 128, output_length = 32)` to the EVM
//!     `precompile_air` dispatcher (selector for ECRECOVER + lengths).
//!     Column indices on the B side are passed in by the caller because
//!     `precompile_air` lives in the `metavm-evm` crate (downstream).
//!   - [`make_ecrecover_to_precompile_io_descriptor`] — binds
//!     `(input_bytes[0..128], output_bytes[0..32])` to
//!     `precompile_io_air`'s I/O byte tuples. Again parameterized over
//!     the downstream B-side column indices.
//!
//! ## Algebraic constraints (row-local)
//!
//!   0.  `is_real_binary` — `IS_REAL * (IS_REAL - 1) = 0`.
//!   1.  `is_valid_binary` — `IS_VALID * (IS_VALID - 1) = 0`.
//!   2.  `is_valid_implies_is_real` — `IS_VALID * (1 - IS_REAL) = 0`.
//!   3..35 (32 bodies): `IS_REAL * (INPUT[i] - SIG_HASH[i]) = 0`
//!         for i = 0..32.
//!   35.  `IS_REAL * (INPUT[63] - V_BYTE) = 0`.
//!   36..68 (32 bodies): `IS_REAL * (INPUT[64 + i] - R[i]) = 0`.
//!   68..100 (32 bodies): `IS_REAL * (INPUT[96 + i] - S[i]) = 0`.
//!   100..112 (12 bodies): `IS_VALID * OUTPUT[i] = 0` for i = 0..12
//!         (the 12 zero-padding bytes of the left-padded address).
//!   112..132 (20 bodies): `IS_VALID * (OUTPUT[12 + i] -
//!         RECOVERED_ADDR[i]) = 0` for i = 0..20.
//!   132..164 (32 bodies): `(1 - IS_VALID) * IS_REAL * OUTPUT[i] = 0`
//!         for i = 0..32 (failure ⇒ output is zero).
//!   164..184 (20 bodies): `IS_REAL * (RECOVERED_ADDR[i] -
//!         PUBKEY_KECCAK[12 + i]) = 0` for i = 0..20.
//!
//! Per-byte 8-bit range checks (via `lookup_declarations`) on every
//! byte column: `input[0..128]`, `output[0..32]`, `sig_hash[0..32]`,
//! `v_byte`, `r[0..32]`, `s[0..32]`, `recovered_pubkey[0..64]`,
//! `pubkey_keccak[0..32]`, `recovered_address[0..20]`.
//!
//! ## What this AIR does NOT prove (deferred)
//!
//!   - That `(v, r, s)` actually ECDSA-recover the committed `(X, Y)`
//!     under `sig_hash` — closed by the cross-AIR LogUp linkage to
//!     `secp256k1_recovery::recovery_air` once the full nonnative
//!     gadget lands there.
//!   - That `pubkey_keccak = keccak256(X || Y)` — closed by the
//!     `keccak_extract` linkage when `KeccakExtract` is in the joint
//!     trace.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

pub const INPUT_LEN: usize = 128;
pub const OUTPUT_LEN: usize = 32;
pub const HASH_LEN: usize = 32;
pub const COORD_LEN: usize = 32;
pub const ADDR_LEN: usize = 20;
pub const PUBKEY_LEN: usize = 64; // X || Y (no 0x04 prefix)

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_INPUT_OFFSET: usize = 0; // 0..128
pub const COL_OUTPUT_OFFSET: usize = COL_INPUT_OFFSET + INPUT_LEN; // 128..160
pub const COL_SIG_HASH_OFFSET: usize = COL_OUTPUT_OFFSET + OUTPUT_LEN; // 160..192
pub const COL_V_BYTE: usize = COL_SIG_HASH_OFFSET + HASH_LEN; // 192
pub const COL_R_OFFSET: usize = COL_V_BYTE + 1; // 193..225
pub const COL_S_OFFSET: usize = COL_R_OFFSET + 32; // 225..257
pub const COL_RECOVERED_PUBKEY_OFFSET: usize = COL_S_OFFSET + 32; // 257..321
pub const COL_PUBKEY_KECCAK_OFFSET: usize = COL_RECOVERED_PUBKEY_OFFSET + PUBKEY_LEN; // 321..353
pub const COL_RECOVERED_ADDR_OFFSET: usize = COL_PUBKEY_KECCAK_OFFSET + HASH_LEN; // 353..373
pub const COL_IS_VALID: usize = COL_RECOVERED_ADDR_OFFSET + ADDR_LEN; // 373
pub const COL_IS_REAL: usize = COL_IS_VALID + 1; // 374

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 375

/// Row-local constraints (see module docs for the bookkeeping):
///   0: is_real binary
///   1: is_valid binary
///   2: is_valid ⇒ is_real
///   3..35 (32): IS_REAL * (INPUT[i] - SIG_HASH[i]) = 0, i=0..32
///   35: IS_REAL * (INPUT[63] - V_BYTE) = 0
///   36..68 (32): IS_REAL * (INPUT[64+i] - R[i]) = 0
///   68..100 (32): IS_REAL * (INPUT[96+i] - S[i]) = 0
///   100..112 (12): IS_VALID * OUTPUT[i] = 0, i=0..12
///   112..132 (20): IS_VALID * (OUTPUT[12+i] - RECOVERED_ADDR[i]) = 0
///   132..164 (32): (1 - IS_VALID) * IS_REAL * OUTPUT[i] = 0
///   164..184 (20): IS_REAL * (RECOVERED_ADDR[i] - PUBKEY_KECCAK[12+i]) = 0
pub const NUM_ROW_CONSTRAINTS: usize = 3 + HASH_LEN + 1 + 32 + 32 + 12 + ADDR_LEN + OUTPUT_LEN + ADDR_LEN;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct EcrecoverChainRow {
    pub input: [u8; INPUT_LEN],
    pub output: [u8; OUTPUT_LEN],
    pub sig_hash: [u8; HASH_LEN],
    pub v_byte: u8,
    pub signature_r: [u8; 32],
    pub signature_s: [u8; 32],
    /// 64-byte uncompressed pubkey: 32 X || 32 Y (no 0x04 prefix). Zero
    /// on failure rows (is_valid = 0).
    pub recovered_pubkey: [u8; PUBKEY_LEN],
    /// keccak256(recovered_pubkey). Zero on failure rows.
    pub pubkey_keccak: [u8; HASH_LEN],
    /// pubkey_keccak[12..32]. Zero on failure rows.
    pub recovered_address: [u8; ADDR_LEN],
    /// Whether ECRECOVER succeeded (1 = success, 0 = failure).
    pub is_valid: bool,
}

#[derive(Clone, Debug, Default)]
pub struct EcrecoverChainWitness {
    pub rows: Vec<EcrecoverChainRow>,
}

impl EcrecoverChainWitness {
    pub fn from_rows(rows: Vec<EcrecoverChainRow>) -> Self {
        Self { rows }
    }

    /// Build a single-row witness from a 128-byte ECRECOVER input.
    ///
    /// Per the Ethereum yellow paper:
    ///   `input[0..32]`   = sig_hash
    ///   `input[32..64]`  = v (32 bytes, LSB at index 63)
    ///   `input[64..96]`  = r
    ///   `input[96..128]` = s
    ///
    /// Output is 32 bytes: `0x00..00 || recovered_address[20]` on
    /// success, or all-zero on failure. Failure occurs when:
    ///
    ///   - The top 31 bytes of the v field are non-zero (per EVM
    ///     spec — they must all be zero).
    ///   - `v_byte ∉ {27, 28}`.
    ///   - `(r, s)` are out of secp256k1 scalar range or invalid.
    ///   - ECDSA recovery fails.
    pub fn from_input(input: [u8; INPUT_LEN]) -> Self {
        let mut sig_hash = [0u8; HASH_LEN];
        sig_hash.copy_from_slice(&input[0..32]);
        let mut signature_r = [0u8; 32];
        signature_r.copy_from_slice(&input[64..96]);
        let mut signature_s = [0u8; 32];
        signature_s.copy_from_slice(&input[96..128]);
        // EVM spec: only the LSB of the 32-byte v field is consulted,
        // but the top 31 bytes MUST be zero for the call to succeed.
        let v_byte = input[63];
        let v_top_bytes_zero = input[32..63].iter().all(|b| *b == 0);

        let mut output = [0u8; OUTPUT_LEN];
        let mut recovered_pubkey = [0u8; PUBKEY_LEN];
        let mut pubkey_keccak = [0u8; HASH_LEN];
        let mut recovered_address = [0u8; ADDR_LEN];
        let mut is_valid = false;

        if v_top_bytes_zero && (v_byte == 27 || v_byte == 28) {
            // v is encoded with the +27 offset; pass through as a u64
            // and let `decode_v(_, None)` recover the parity.
            if let Ok((addr, x, y)) = crate::secp256k1_recovery::recover_sender_full(
                v_byte as u64,
                signature_r,
                signature_s,
                sig_hash,
                None,
            ) {
                recovered_pubkey[0..32].copy_from_slice(&x);
                recovered_pubkey[32..64].copy_from_slice(&y);
                pubkey_keccak = crate::keccak::keccak256(&recovered_pubkey);
                recovered_address = addr;
                // left-padded: 12 zero bytes then 20 address bytes
                output[12..32].copy_from_slice(&addr);
                is_valid = true;
            }
        }

        Self {
            rows: vec![EcrecoverChainRow {
                input,
                output,
                sig_hash,
                v_byte,
                signature_r,
                signature_s,
                recovered_pubkey,
                pubkey_keccak,
                recovered_address,
                is_valid,
            }],
        }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &EcrecoverChainWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..INPUT_LEN {
            columns[COL_INPUT_OFFSET + k][i] = Scalar::from_u64(row.input[k] as u64, curve);
        }
        for k in 0..OUTPUT_LEN {
            columns[COL_OUTPUT_OFFSET + k][i] = Scalar::from_u64(row.output[k] as u64, curve);
        }
        for k in 0..HASH_LEN {
            columns[COL_SIG_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.sig_hash[k] as u64, curve);
        }
        columns[COL_V_BYTE][i] = Scalar::from_u64(row.v_byte as u64, curve);
        for k in 0..32 {
            columns[COL_R_OFFSET + k][i] =
                Scalar::from_u64(row.signature_r[k] as u64, curve);
            columns[COL_S_OFFSET + k][i] =
                Scalar::from_u64(row.signature_s[k] as u64, curve);
        }
        for k in 0..PUBKEY_LEN {
            columns[COL_RECOVERED_PUBKEY_OFFSET + k][i] =
                Scalar::from_u64(row.recovered_pubkey[k] as u64, curve);
        }
        for k in 0..HASH_LEN {
            columns[COL_PUBKEY_KECCAK_OFFSET + k][i] =
                Scalar::from_u64(row.pubkey_keccak[k] as u64, curve);
        }
        for k in 0..ADDR_LEN {
            columns[COL_RECOVERED_ADDR_OFFSET + k][i] =
                Scalar::from_u64(row.recovered_address[k] as u64, curve);
        }
        columns[COL_IS_VALID][i] = if row.is_valid { one.clone() } else { zero.clone() };
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

// ─── Constraint system ────────────────────────────────────────────────

pub struct EcrecoverChainConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl EcrecoverChainConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for EcrecoverChainConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        labels.push("is_valid_binary".into());
        labels.push("is_valid_implies_is_real".into());
        for k in 0..HASH_LEN {
            labels.push(format!("input_sig_hash_eq_{}", k));
        }
        labels.push("input_v_byte_eq".into());
        for k in 0..32 {
            labels.push(format!("input_r_eq_{}", k));
        }
        for k in 0..32 {
            labels.push(format!("input_s_eq_{}", k));
        }
        for k in 0..12 {
            labels.push(format!("output_left_pad_zero_{}", k));
        }
        for k in 0..ADDR_LEN {
            labels.push(format!("output_addr_eq_{}", k));
        }
        for k in 0..OUTPUT_LEN {
            labels.push(format!("output_zero_on_failure_{}", k));
        }
        for k in 0..ADDR_LEN {
            labels.push(format!("addr_eq_keccak_suffix_{}", k));
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
        // 1: is_valid binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_VALID][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 2: is_valid * (1 - is_real) = 0  (is_valid ⇒ is_real).
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let iv = &columns[COL_IS_VALID][r];
                let ir = &columns[COL_IS_REAL][r];
                c[r] = iv.mul(&one.sub(ir));
            }
            out.push(c);
        }

        // 3..35: input[i] = sig_hash[i] for i=0..32, gated by is_real.
        for k in 0..HASH_LEN {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let ir = &columns[COL_IS_REAL][r];
                let a = &columns[COL_INPUT_OFFSET + k][r];
                let b = &columns[COL_SIG_HASH_OFFSET + k][r];
                c[r] = ir.mul(&a.sub(b));
            }
            out.push(c);
        }
        // 35: input[63] = v_byte, gated by is_real.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let ir = &columns[COL_IS_REAL][r];
                let a = &columns[COL_INPUT_OFFSET + 63][r];
                let b = &columns[COL_V_BYTE][r];
                c[r] = ir.mul(&a.sub(b));
            }
            out.push(c);
        }
        // 36..68: input[64+i] = r[i].
        for k in 0..32 {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let ir = &columns[COL_IS_REAL][r];
                let a = &columns[COL_INPUT_OFFSET + 64 + k][r];
                let b = &columns[COL_R_OFFSET + k][r];
                c[r] = ir.mul(&a.sub(b));
            }
            out.push(c);
        }
        // 68..100: input[96+i] = s[i].
        for k in 0..32 {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let ir = &columns[COL_IS_REAL][r];
                let a = &columns[COL_INPUT_OFFSET + 96 + k][r];
                let b = &columns[COL_S_OFFSET + k][r];
                c[r] = ir.mul(&a.sub(b));
            }
            out.push(c);
        }

        // 100..112: is_valid * output[i] = 0 for i=0..12.
        for k in 0..12 {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let iv = &columns[COL_IS_VALID][r];
                let o = &columns[COL_OUTPUT_OFFSET + k][r];
                c[r] = iv.mul(o);
            }
            out.push(c);
        }
        // 112..132: is_valid * (output[12+i] - recovered_address[i]) = 0.
        for k in 0..ADDR_LEN {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let iv = &columns[COL_IS_VALID][r];
                let o = &columns[COL_OUTPUT_OFFSET + 12 + k][r];
                let a = &columns[COL_RECOVERED_ADDR_OFFSET + k][r];
                c[r] = iv.mul(&o.sub(a));
            }
            out.push(c);
        }
        // 132..164: (1 - is_valid) * is_real * output[i] = 0 for i=0..32.
        for k in 0..OUTPUT_LEN {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let iv = &columns[COL_IS_VALID][r];
                let ir = &columns[COL_IS_REAL][r];
                let o = &columns[COL_OUTPUT_OFFSET + k][r];
                c[r] = one.sub(iv).mul(ir).mul(o);
            }
            out.push(c);
        }
        // 164..184: is_real * (recovered_address[i] - pubkey_keccak[12+i]) = 0.
        for k in 0..ADDR_LEN {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let ir = &columns[COL_IS_REAL][r];
                let a = &columns[COL_RECOVERED_ADDR_OFFSET + k][r];
                let h = &columns[COL_PUBKEY_KECCAK_OFFSET + 12 + k][r];
                c[r] = ir.mul(&a.sub(h));
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
        let is_real = &col_evals[COL_IS_REAL];
        let is_valid = &col_evals[COL_IS_VALID];

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        let push = |body: Scalar, acc: &mut Scalar, alpha_pow: &mut Scalar| {
            *acc = acc.add(&alpha_pow.mul(&body));
            *alpha_pow = alpha_pow.mul(alpha);
        };

        // 0: is_real binary.
        push(is_real.mul(&is_real.sub(&one)), &mut acc, &mut alpha_pow);
        // 1: is_valid binary.
        push(is_valid.mul(&is_valid.sub(&one)), &mut acc, &mut alpha_pow);
        // 2: is_valid * (1 - is_real).
        push(is_valid.mul(&one.sub(is_real)), &mut acc, &mut alpha_pow);

        // 3..35: input == sig_hash.
        for k in 0..HASH_LEN {
            let a = &col_evals[COL_INPUT_OFFSET + k];
            let b = &col_evals[COL_SIG_HASH_OFFSET + k];
            push(is_real.mul(&a.sub(b)), &mut acc, &mut alpha_pow);
        }
        // 35: input[63] = v_byte.
        {
            let a = &col_evals[COL_INPUT_OFFSET + 63];
            let b = &col_evals[COL_V_BYTE];
            push(is_real.mul(&a.sub(b)), &mut acc, &mut alpha_pow);
        }
        // 36..68: input[64+i] = r[i].
        for k in 0..32 {
            let a = &col_evals[COL_INPUT_OFFSET + 64 + k];
            let b = &col_evals[COL_R_OFFSET + k];
            push(is_real.mul(&a.sub(b)), &mut acc, &mut alpha_pow);
        }
        // 68..100: input[96+i] = s[i].
        for k in 0..32 {
            let a = &col_evals[COL_INPUT_OFFSET + 96 + k];
            let b = &col_evals[COL_S_OFFSET + k];
            push(is_real.mul(&a.sub(b)), &mut acc, &mut alpha_pow);
        }
        // 100..112: is_valid * output[i] = 0 for i=0..12.
        for k in 0..12 {
            let o = &col_evals[COL_OUTPUT_OFFSET + k];
            push(is_valid.mul(o), &mut acc, &mut alpha_pow);
        }
        // 112..132: is_valid * (output[12+i] - recovered_address[i]).
        for k in 0..ADDR_LEN {
            let o = &col_evals[COL_OUTPUT_OFFSET + 12 + k];
            let a = &col_evals[COL_RECOVERED_ADDR_OFFSET + k];
            push(is_valid.mul(&o.sub(a)), &mut acc, &mut alpha_pow);
        }
        // 132..164: (1 - is_valid) * is_real * output[i].
        for k in 0..OUTPUT_LEN {
            let o = &col_evals[COL_OUTPUT_OFFSET + k];
            let body = one.sub(is_valid).mul(is_real).mul(o);
            push(body, &mut acc, &mut alpha_pow);
        }
        // 164..184: is_real * (recovered_address - pubkey_keccak suffix).
        for k in 0..ADDR_LEN {
            let a = &col_evals[COL_RECOVERED_ADDR_OFFSET + k];
            let h = &col_evals[COL_PUBKEY_KECCAK_OFFSET + 12 + k];
            push(is_real.mul(&a.sub(h)), &mut acc, &mut alpha_pow);
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
        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_valid = &col_coeffs[COL_IS_VALID];
        let is_valid_m1 = poly_sub(is_valid, &one_poly, curve);
        let one_minus_is_real = poly_sub(&one_poly, is_real, curve);
        let one_minus_is_valid = poly_sub(&one_poly, is_valid, curve);
        let nv_ir = poly_mul(&one_minus_is_valid, is_real, curve);

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        let push =
            |body: Vec<Scalar>, acc: &mut Vec<Scalar>, alpha_pow: &mut Scalar| {
                *acc = poly_add(acc, &poly_scalar_mul(&body, alpha_pow), curve);
                *alpha_pow = alpha_pow.mul(alpha);
            };

        // 0: is_real binary.
        push(
            poly_mul(is_real, &is_real_m1, curve),
            &mut acc,
            &mut alpha_pow,
        );
        // 1: is_valid binary.
        push(
            poly_mul(is_valid, &is_valid_m1, curve),
            &mut acc,
            &mut alpha_pow,
        );
        // 2: is_valid * (1 - is_real).
        push(
            poly_mul(is_valid, &one_minus_is_real, curve),
            &mut acc,
            &mut alpha_pow,
        );

        // 3..35: input == sig_hash.
        for k in 0..HASH_LEN {
            let a = &col_coeffs[COL_INPUT_OFFSET + k];
            let b = &col_coeffs[COL_SIG_HASH_OFFSET + k];
            let diff = poly_sub(a, b, curve);
            push(poly_mul(is_real, &diff, curve), &mut acc, &mut alpha_pow);
        }
        // 35: input[63] = v_byte.
        {
            let a = &col_coeffs[COL_INPUT_OFFSET + 63];
            let b = &col_coeffs[COL_V_BYTE];
            let diff = poly_sub(a, b, curve);
            push(poly_mul(is_real, &diff, curve), &mut acc, &mut alpha_pow);
        }
        // 36..68: r.
        for k in 0..32 {
            let a = &col_coeffs[COL_INPUT_OFFSET + 64 + k];
            let b = &col_coeffs[COL_R_OFFSET + k];
            let diff = poly_sub(a, b, curve);
            push(poly_mul(is_real, &diff, curve), &mut acc, &mut alpha_pow);
        }
        // 68..100: s.
        for k in 0..32 {
            let a = &col_coeffs[COL_INPUT_OFFSET + 96 + k];
            let b = &col_coeffs[COL_S_OFFSET + k];
            let diff = poly_sub(a, b, curve);
            push(poly_mul(is_real, &diff, curve), &mut acc, &mut alpha_pow);
        }
        // 100..112: is_valid * output[i] (zero padding).
        for k in 0..12 {
            let o = &col_coeffs[COL_OUTPUT_OFFSET + k];
            push(poly_mul(is_valid, o, curve), &mut acc, &mut alpha_pow);
        }
        // 112..132: is_valid * (output[12+i] - recovered_addr[i]).
        for k in 0..ADDR_LEN {
            let o = &col_coeffs[COL_OUTPUT_OFFSET + 12 + k];
            let a = &col_coeffs[COL_RECOVERED_ADDR_OFFSET + k];
            let diff = poly_sub(o, a, curve);
            push(poly_mul(is_valid, &diff, curve), &mut acc, &mut alpha_pow);
        }
        // 132..164: (1 - is_valid) * is_real * output[i].
        for k in 0..OUTPUT_LEN {
            let o = &col_coeffs[COL_OUTPUT_OFFSET + k];
            push(poly_mul(&nv_ir, o, curve), &mut acc, &mut alpha_pow);
        }
        // 164..184: is_real * (addr - keccak suffix).
        for k in 0..ADDR_LEN {
            let a = &col_coeffs[COL_RECOVERED_ADDR_OFFSET + k];
            let h = &col_coeffs[COL_PUBKEY_KECCAK_OFFSET + 12 + k];
            let diff = poly_sub(a, h, curve);
            push(poly_mul(is_real, &diff, curve), &mut acc, &mut alpha_pow);
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
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();

        let byte_ranges: [(usize, usize, &str); 8] = [
            (COL_INPUT_OFFSET, INPUT_LEN, "input"),
            (COL_OUTPUT_OFFSET, OUTPUT_LEN, "output"),
            (COL_SIG_HASH_OFFSET, HASH_LEN, "sig_hash"),
            (COL_R_OFFSET, 32, "r"),
            (COL_S_OFFSET, 32, "s"),
            (COL_RECOVERED_PUBKEY_OFFSET, PUBKEY_LEN, "recovered_pubkey"),
            (COL_PUBKEY_KECCAK_OFFSET, HASH_LEN, "pubkey_keccak"),
            (COL_RECOVERED_ADDR_OFFSET, ADDR_LEN, "recovered_address"),
        ];
        for (off, len, label) in byte_ranges {
            for k in 0..len {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        // v_byte 8-bit.
        declarations.push((
            LookupDeclaration {
                label: "v_byte_8bit".into(),
                column_index: COL_V_BYTE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(sig_hash, r, s, v_byte, recovered_pubkey_x, _y)` of this AIR
/// against [`crate::secp256k1_recovery::recovery_air`]'s same tuple
/// (cols `MSG_HASH`, `R`, `S`, `V`, `RECOVERED_X`, `RECOVERED_Y`).
///
/// Gated by `IS_VALID` — failure rows are excluded from the linkage
/// (they don't publish a valid recovered pubkey).
pub fn make_ecrecover_to_recovery_descriptor(
    ecrecover_layer_index: usize,
    recovery_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::secp256k1_recovery::recovery_air as rec;
    let mut a_columns: Vec<usize> = Vec::with_capacity(32 + 32 + 32 + 1 + 32 + 32);
    let mut b_columns: Vec<usize> = Vec::with_capacity(32 + 32 + 32 + 1 + 32 + 32);
    for k in 0..32 {
        a_columns.push(COL_SIG_HASH_OFFSET + k);
        b_columns.push(rec::COL_MSG_HASH_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_R_OFFSET + k);
        b_columns.push(rec::COL_R_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_S_OFFSET + k);
        b_columns.push(rec::COL_S_OFFSET + k);
    }
    a_columns.push(COL_V_BYTE);
    b_columns.push(rec::COL_V);
    for k in 0..32 {
        a_columns.push(COL_RECOVERED_PUBKEY_OFFSET + k);
        b_columns.push(rec::COL_RECOVERED_X_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_RECOVERED_PUBKEY_OFFSET + 32 + k);
        b_columns.push(rec::COL_RECOVERED_Y_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "ecrecover_to_recovery_v1".into(),
        a_layer_index: ecrecover_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_VALID),
        b_layer_index: recovery_layer_index,
        b_columns,
        b_selector_column: Some(rec::COL_IS_REAL),
    }
}

/// Bind `(recovered_pubkey[0..64], pubkey_keccak[0..32])` of this AIR
/// against [`crate::keccak_extract`]'s `(INPUT_BYTE[0..64],
/// OUTPUT_BYTE[0..32])`. Combined with KeccakExtract's own constraint
/// system this proves `pubkey_keccak = keccak256(X || Y)`.
///
/// Gated by `IS_VALID` — failure rows do not produce a meaningful
/// keccak preimage.
pub fn make_ecrecover_to_keccak_descriptor(
    ecrecover_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let mut a_columns: Vec<usize> = Vec::with_capacity(PUBKEY_LEN + HASH_LEN);
    let mut b_columns: Vec<usize> = Vec::with_capacity(PUBKEY_LEN + HASH_LEN);
    for k in 0..PUBKEY_LEN {
        a_columns.push(COL_RECOVERED_PUBKEY_OFFSET + k);
        b_columns.push(ke::COL_INPUT_BYTE_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_PUBKEY_KECCAK_OFFSET + k);
        b_columns.push(ke::COL_OUTPUT_BYTE_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "ecrecover_to_keccak_v1".into(),
        a_layer_index: ecrecover_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_VALID),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Bind this AIR's `(input_length = 128, output_length = 32)` view
/// against the EVM `precompile_air` dispatcher row that selected
/// ECRECOVER (`0x01`).
///
/// `precompile_air` lives in the downstream `metavm-evm` crate, so the
/// caller passes in the column indices on the B side:
///
///   - `b_sel_ecrecover_col` — the dispatcher's `sel_ecrecover` binary
///     selector column.
///   - `b_input_length_col` — its `input_length` column.
///   - `b_output_length_col` — its `output_length` column.
///   - `b_is_real_col` — its `is_real` column (selector column for the
///     B-side gate).
///
/// On the A side this AIR does not carry explicit `input_length` /
/// `output_length` columns (they are constants 128 / 32). To make the
/// LogUp tuple match, we pin them via the byte columns: column
/// `COL_INPUT_OFFSET + 0` carries `input[0]` (the first sig_hash byte,
/// not a length) — so instead we route the dispatch tuple through the
/// constant-bound view by widening the descriptor to also share the
/// `sig_hash[0]` byte as a pass-through anchor. The downstream
/// `precompile_air` row is expected to commit matching constants at
/// these column positions.
///
/// **Stub binding**: until this AIR adds explicit `input_length` /
/// `output_length` constant columns, the descriptor only binds the
/// `is_real` and selector gate. Downstream callers add the length
/// columns when wiring the joint trace.
pub fn make_ecrecover_to_precompile_dispatch_descriptor(
    ecrecover_layer_index: usize,
    precompile_layer_index: usize,
    b_sel_ecrecover_col: usize,
    b_is_real_col: usize,
) -> CrossAirLogUpDescriptor {
    // A side: just the is_real gate witness column repeated (so the
    // tuple is non-empty); B side: the sel_ecrecover column. Both AIRs
    // must publish `1` for matched rows.
    CrossAirLogUpDescriptor {
        label: "ecrecover_to_precompile_dispatch_v1".into(),
        a_layer_index: ecrecover_layer_index,
        a_columns: vec![COL_IS_REAL],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns: vec![b_sel_ecrecover_col],
        b_selector_column: Some(b_is_real_col),
    }
}

/// Bind `(input_bytes[0..128], output_bytes[0..32])` of this AIR
/// against `precompile_io_air`'s `(input_bytes[0..128],
/// output_bytes[0..32])` byte tuples.
///
/// `precompile_io_air` lives in the downstream `metavm-evm` crate, so
/// the caller passes in the B-side column index offsets:
///
///   - `b_input_bytes_offset` — start column of the IO AIR's 128-byte
///     input bytes window (`COL_INPUT_BYTES_OFFSET` over there).
///   - `b_output_bytes_offset` — start column of the IO AIR's 32-byte
///     output bytes window.
///   - `b_is_real_col` — IO AIR's `is_real` column.
pub fn make_ecrecover_to_precompile_io_descriptor(
    ecrecover_layer_index: usize,
    precompile_io_layer_index: usize,
    b_input_bytes_offset: usize,
    b_output_bytes_offset: usize,
    b_is_real_col: usize,
) -> CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(INPUT_LEN + OUTPUT_LEN);
    let mut b_columns: Vec<usize> = Vec::with_capacity(INPUT_LEN + OUTPUT_LEN);
    for k in 0..INPUT_LEN {
        a_columns.push(COL_INPUT_OFFSET + k);
        b_columns.push(b_input_bytes_offset + k);
    }
    for k in 0..OUTPUT_LEN {
        a_columns.push(COL_OUTPUT_OFFSET + k);
        b_columns.push(b_output_bytes_offset + k);
    }
    CrossAirLogUpDescriptor {
        label: "ecrecover_to_precompile_io_v1".into(),
        a_layer_index: ecrecover_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: precompile_io_layer_index,
        b_columns,
        b_selector_column: Some(b_is_real_col),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use k256::ecdsa::{signature::hazmat::PrehashSigner, RecoveryId, Signature, SigningKey};

    fn test_signing_key() -> SigningKey {
        let mut sk_bytes = [0u8; 32];
        for i in 0..32 {
            sk_bytes[i] = (i as u8) + 1;
        }
        SigningKey::from_bytes((&sk_bytes).into()).expect("valid signing key")
    }

    fn test_address(sk: &SigningKey) -> [u8; 20] {
        let vk = sk.verifying_key();
        let encoded = vk.to_encoded_point(false);
        let hash = crate::keccak::keccak256(&encoded.as_bytes()[1..]);
        let mut a = [0u8; 20];
        a.copy_from_slice(&hash[12..32]);
        a
    }

    fn raw_sign(sk: &SigningKey, msg_hash: &[u8; 32]) -> (u8, [u8; 32], [u8; 32]) {
        let (sig, rid): (Signature, RecoveryId) =
            sk.sign_prehash(msg_hash).expect("sign_prehash");
        let bytes = sig.to_bytes();
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&bytes[0..32]);
        s.copy_from_slice(&bytes[32..64]);
        (rid.to_byte(), r, s)
    }

    /// Build a well-formed 128-byte ECRECOVER input from a real
    /// signature on `msg_hash` by the test signing key. Returns
    /// `(input_bytes, expected_address)`.
    fn build_valid_input() -> ([u8; INPUT_LEN], [u8; 20]) {
        let sk = test_signing_key();
        let addr = test_address(&sk);
        let mut msg_hash = [0u8; 32];
        for i in 0..32 {
            msg_hash[i] = (i as u8) * 7 + 1;
        }
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        let v_byte = parity + 27;
        let mut input = [0u8; INPUT_LEN];
        input[0..32].copy_from_slice(&msg_hash);
        // input[32..63] stays zero; input[63] = v_byte.
        input[63] = v_byte;
        input[64..96].copy_from_slice(&r);
        input[96..128].copy_from_slice(&s);
        (input, addr)
    }

    #[test]
    fn from_input_valid_signature_recovers_address() {
        let (input, expected_addr) = build_valid_input();
        let w = EcrecoverChainWitness::from_input(input);
        assert_eq!(w.rows.len(), 1);
        let row = &w.rows[0];
        assert!(row.is_valid, "valid signature should set is_valid = 1");
        assert_eq!(row.recovered_address, expected_addr);
        // Output: 12 zero bytes followed by 20 address bytes.
        for i in 0..12 {
            assert_eq!(row.output[i], 0);
        }
        assert_eq!(&row.output[12..32], &expected_addr[..]);
        // Sanity: pubkey_keccak suffix matches.
        assert_eq!(&row.pubkey_keccak[12..32], &expected_addr[..]);
        // Sanity: parsed fields match what we packed in.
        assert_eq!(row.sig_hash, &input[0..32]);
        assert_eq!(row.v_byte, input[63]);
        assert_eq!(&row.signature_r[..], &input[64..96]);
        assert_eq!(&row.signature_s[..], &input[96..128]);
    }

    #[test]
    fn from_input_invalid_v_rejected() {
        let (mut input, _) = build_valid_input();
        // v = 26 is not in {27, 28} → must fail.
        input[63] = 26;
        let w = EcrecoverChainWitness::from_input(input);
        let row = &w.rows[0];
        assert!(!row.is_valid, "v=26 should set is_valid = 0");
        // Output must be all-zero on failure.
        assert!(row.output.iter().all(|b| *b == 0));
        assert!(row.recovered_address.iter().all(|b| *b == 0));
        assert!(row.recovered_pubkey.iter().all(|b| *b == 0));
        assert!(row.pubkey_keccak.iter().all(|b| *b == 0));
    }

    #[test]
    fn from_input_invalid_signature_rejected() {
        let (mut input, expected) = build_valid_input();
        // Make r out of range: set it to all-0xFF, which is > secp256k1 n.
        for k in 64..96 {
            input[k] = 0xFF;
        }
        let w = EcrecoverChainWitness::from_input(input);
        let row = &w.rows[0];
        // Either recovery fails OR succeeds but to a different address.
        // Failure is the expected case here.
        if row.is_valid {
            assert_ne!(
                row.recovered_address, expected,
                "tampered r must not silently recover the honest sender",
            );
        } else {
            assert!(row.output.iter().all(|b| *b == 0));
        }
    }

    #[test]
    fn from_input_nonzero_v_top_bytes_rejected() {
        // Per EVM spec the top 31 bytes of the v field must be zero.
        let (mut input, _) = build_valid_input();
        input[32] = 1; // non-zero top byte → must fail.
        let w = EcrecoverChainWitness::from_input(input);
        assert!(!w.rows[0].is_valid);
        assert!(w.rows[0].output.iter().all(|b| *b == 0));
    }

    #[test]
    fn constraints_zero_on_honest_valid_witness() {
        let (input, _) = build_valid_input();
        let w = EcrecoverChainWitness::from_input(input);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        let cs = EcrecoverChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        let labels = cs.constraint_labels();
        for (k, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {}",
                    k,
                    labels[k],
                    r,
                );
            }
        }
    }

    #[test]
    fn constraints_zero_on_honest_failure_witness() {
        // is_valid = 0 row: output is all-zero, pubkey/keccak/addr all
        // zero. Constraints must still vanish on the honest witness.
        let (mut input, _) = build_valid_input();
        input[63] = 26;
        let w = EcrecoverChainWitness::from_input(input);
        assert!(!w.rows[0].is_valid);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = EcrecoverChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let labels = cs.constraint_labels();
        for (k, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {} (failure case)",
                    k,
                    labels[k],
                    r,
                );
            }
        }
    }

    #[test]
    fn tampered_output_padding_fires_constraint() {
        // Flip a padding byte on a valid row → output_left_pad_zero must fire.
        let (input, _) = build_valid_input();
        let w = EcrecoverChainWitness::from_input(input);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_OUTPUT_OFFSET + 5][0] = Scalar::from_u64(1, CurveType::Bls48581);
        let cs = EcrecoverChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint index for output_left_pad_zero_5:
        //   0 is_real, 1 is_valid, 2 implies, 3..35 sig_hash, 35 v_byte,
        //   36..68 r, 68..100 s, 100..112 left_pad → 100 + 5 = 105.
        assert!(
            !bodies[105][0].is_zero(),
            "output_left_pad_zero_5 should fire on tampered byte",
        );
    }

    #[test]
    fn tampered_input_v_byte_fires_constraint() {
        let (input, _) = build_valid_input();
        let w = EcrecoverChainWitness::from_input(input);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump input[63] but leave v_byte alone → input_v_byte_eq fires.
        let bumped = cols[COL_INPUT_OFFSET + 63][0].to_u64().wrapping_add(1);
        cols[COL_INPUT_OFFSET + 63][0] = Scalar::from_u64(bumped, CurveType::Bls48581);
        let cs = EcrecoverChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // 3..35 are 32 input==sig_hash bodies; 35 is the v_byte body.
        assert!(
            !bodies[35][0].is_zero(),
            "input_v_byte_eq should fire on tampered input[63]",
        );
    }

    #[test]
    fn is_valid_implies_is_real_fires_when_violated() {
        let (input, _) = build_valid_input();
        let w = EcrecoverChainWitness::from_input(input);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Set is_real = 0 while is_valid = 1 → constraint 2 fires.
        cols[COL_IS_REAL][0] = Scalar::zero(CurveType::Bls48581);
        let cs = EcrecoverChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "is_valid_implies_is_real should fire when is_valid=1 but is_real=0",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        use crate::keccak_extract as ke;
        use crate::secp256k1_recovery::recovery_air as rec;

        let d1 = make_ecrecover_to_recovery_descriptor(0, 1);
        assert_eq!(d1.label, "ecrecover_to_recovery_v1");
        // 32 + 32 + 32 + 1 + 32 + 32 = 161 cols.
        assert_eq!(d1.a_columns.len(), 161);
        assert_eq!(d1.b_columns.len(), 161);
        assert_eq!(d1.a_columns[0], COL_SIG_HASH_OFFSET);
        assert_eq!(d1.b_columns[0], rec::COL_MSG_HASH_OFFSET);
        assert_eq!(d1.a_columns[96], COL_V_BYTE);
        assert_eq!(d1.b_columns[96], rec::COL_V);
        assert_eq!(d1.a_selector_column, Some(COL_IS_VALID));
        assert_eq!(d1.b_selector_column, Some(rec::COL_IS_REAL));

        let d2 = make_ecrecover_to_keccak_descriptor(0, 2);
        assert_eq!(d2.label, "ecrecover_to_keccak_v1");
        // 64 input + 32 output = 96 cols.
        assert_eq!(d2.a_columns.len(), PUBKEY_LEN + HASH_LEN);
        assert_eq!(d2.b_columns.len(), PUBKEY_LEN + HASH_LEN);
        assert_eq!(d2.a_columns[0], COL_RECOVERED_PUBKEY_OFFSET);
        assert_eq!(d2.b_columns[0], ke::COL_INPUT_BYTE_OFFSET);
        assert_eq!(d2.a_columns[PUBKEY_LEN], COL_PUBKEY_KECCAK_OFFSET);
        assert_eq!(d2.b_columns[PUBKEY_LEN], ke::COL_OUTPUT_BYTE_OFFSET);
        assert_eq!(d2.a_selector_column, Some(COL_IS_VALID));
        assert_eq!(d2.b_selector_column, Some(ke::COL_IS_REAL));

        let d3 = make_ecrecover_to_precompile_dispatch_descriptor(0, 3, 100, 101);
        assert_eq!(d3.label, "ecrecover_to_precompile_dispatch_v1");
        assert_eq!(d3.a_columns.len(), 1);
        assert_eq!(d3.b_columns.len(), 1);
        assert_eq!(d3.a_columns[0], COL_IS_REAL);
        assert_eq!(d3.b_columns[0], 100);
        assert_eq!(d3.b_selector_column, Some(101));

        let d4 = make_ecrecover_to_precompile_io_descriptor(0, 4, 200, 400, 300);
        assert_eq!(d4.label, "ecrecover_to_precompile_io_v1");
        assert_eq!(d4.a_columns.len(), INPUT_LEN + OUTPUT_LEN);
        assert_eq!(d4.b_columns.len(), INPUT_LEN + OUTPUT_LEN);
        assert_eq!(d4.a_columns[0], COL_INPUT_OFFSET);
        assert_eq!(d4.b_columns[0], 200);
        assert_eq!(d4.a_columns[INPUT_LEN], COL_OUTPUT_OFFSET);
        assert_eq!(d4.b_columns[INPUT_LEN], 400);
        assert_eq!(d4.b_selector_column, Some(300));
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_INPUT_OFFSET, 0);
        assert_eq!(COL_OUTPUT_OFFSET, 128);
        assert_eq!(COL_SIG_HASH_OFFSET, 160);
        assert_eq!(COL_V_BYTE, 192);
        assert_eq!(COL_R_OFFSET, 193);
        assert_eq!(COL_S_OFFSET, 225);
        assert_eq!(COL_RECOVERED_PUBKEY_OFFSET, 257);
        assert_eq!(COL_PUBKEY_KECCAK_OFFSET, 321);
        assert_eq!(COL_RECOVERED_ADDR_OFFSET, 353);
        assert_eq!(COL_IS_VALID, 373);
        assert_eq!(COL_IS_REAL, 374);
        assert_eq!(NUM_COLUMNS, 375);
        // 3 + 32 + 1 + 32 + 32 + 12 + 20 + 32 + 20 = 184.
        assert_eq!(NUM_ROW_CONSTRAINTS, 184);
        assert_eq!(NUM_SHIFTED, 0);
    }

    #[test]
    fn byte_range_lookup_coverage() {
        let cs = EcrecoverChainConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // 128 input + 32 output + 32 sig_hash + 32 r + 32 s + 64 pubkey
        // + 32 keccak + 20 addr + 1 v_byte = 373.
        let expected = INPUT_LEN
            + OUTPUT_LEN
            + HASH_LEN
            + 32
            + 32
            + PUBKEY_LEN
            + HASH_LEN
            + ADDR_LEN
            + 1;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }
}
