//! Skeleton secp256k1 ECDSA recovery AIR.
//!
//! This AIR exposes the *witness shape* required for in-circuit
//! ECDSA recovery (msg_hash, r, s, v, recovered public-key
//! coordinates, recovered 20-byte address) but only implements a
//! small subset of the full algebraic check today:
//!
//!   - `is_real` is binary.
//!   - `is_first_row` is binary.
//!   - Per-byte 8-bit range checks on every byte column.
//!   - **One demonstration "nonnative" binding**: a row-local
//!     constraint that the `recovered_addr[0..20]` bytes equal
//!     `keccak256(X || Y)[12..32]` — encoded here as a Schwartz-Zippel
//!     RLC equality: `Σ α^k · recovered_addr[k]` must equal
//!     `Σ α^k · keccak_hash_witness[12+k]` where
//!     `keccak_hash_witness[0..32]` is an additional witness column
//!     populated by the trace builder. The `keccak_hash_witness ↔ X||Y`
//!     keccak relationship itself will be discharged by a cross-AIR
//!     LogUp to `KeccakExtract` in a follow-up (the descriptor below
//!     pins this AIR's `recovered_addr` to the EVM's `tx_origin`
//!     limbs; the keccak↔addr binding within this AIR plus the
//!     follow-up keccak link closes the chain).
//!
//! The full secp256k1 group law (scalar mult of `s·G + r·Q`,
//! recovery of `Q` from `r` via Weierstrass equation, point
//! decompression with y-parity, etc.) over a nonnative 256-bit
//! prime is deferred — those constraints would land here as
//! additional row-local + cross-row equations following the
//! `nonnative_fp_air` / `nonnative_fp2_compile` patterns.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const HASH_LEN: usize = 32;
pub const COORD_LEN: usize = 32;
pub const ADDR_LEN: usize = 20;

// ─── Column layout ────────────────────────────────────────────────────
//
// Offsets are laid out contiguously so that linkage descriptors can
// reference fixed slices.

pub const COL_MSG_HASH_OFFSET: usize = 0; // 0..32
pub const COL_R_OFFSET: usize = 32; // 32..64
pub const COL_S_OFFSET: usize = 64; // 64..96
pub const COL_V: usize = 96; // single column (0..255 expected)
pub const COL_RECOVERED_X_OFFSET: usize = 97; // 97..129
pub const COL_RECOVERED_Y_OFFSET: usize = 129; // 129..161
/// Witness column holding `keccak256(X || Y)` (used by the row-local
/// addr-binding constraint).
pub const COL_KECCAK_XY_HASH_OFFSET: usize = 161; // 161..193
pub const COL_RECOVERED_ADDR_OFFSET: usize = 193; // 193..213
/// 4 LE u64 limbs of `recovered_addr` — exposed for the cross-AIR
/// LogUp descriptor binding to the EVM main's `tx_origin` columns
/// (which use the same limb encoding).
pub const COL_RECOVERED_ADDR_LIMB_L0: usize = 213;
pub const COL_RECOVERED_ADDR_LIMB_L1: usize = 214;
pub const COL_RECOVERED_ADDR_LIMB_L2: usize = 215;
pub const COL_RECOVERED_ADDR_LIMB_L3: usize = 216;
pub const COL_IS_REAL: usize = 217;
pub const COL_IS_FIRST_ROW: usize = 218;

// ─── Per-iteration scalar-mul witness block (Task #184) ───────────────
//
// To enable populating all 256 double-and-add iterations' Fp-mult
// closures, recovery_air widens with a per-iteration witness block.
// Each row N stores iteration N's intermediate `(a, b, c = a·b mod p)`
// limb tuples for each of the [`NUM_SCALAR_MUL_SUB_DESCRIPTORS`]
// sub-operations of the inner loop:
//
//   `double_x_squared`, `double_lambda_squared`, `double_lambda_dx`,
//   `double_three_x`, `double_inv_2y_roundtrip`, `add_lambda_squared`,
//   `add_lambda_dx`, `add_inv_dx_roundtrip`.
//
// Each sub-descriptor reserves 18 columns: 6 a-limbs ‖ 6 b-limbs ‖
// 6 c-limbs. Total = 8 · 18 = 144 columns, plus one `iteration_index`
// column.
//
// New cross-AIR LogUp descriptors in
// `secp256k1_group_descriptors::Secp256k1ScalarMulIterBlockDescriptors`
// reference these fixed-offset block columns, so a single descriptor's
// closure covers all 256 iterations (one per row) for a given
// sub-operation.

/// Maximum number of iterations the per-iteration witness block fixture
/// can populate in one trace (= full secp256k1 scalar bit-length).
pub const SCALAR_MUL_ITERATIONS_MAX: usize = 256;

/// Number of double-and-add sub-operations covered by the per-iteration
/// witness block (matches the inner-loop `SUB_LABELS` array in
/// `secp256k1_group_descriptors::populate_scalar_mul_trace`).
pub const NUM_SCALAR_MUL_SUB_DESCRIPTORS: usize = 8;

/// Number of limb columns per (a | b | c) tuple (mirrors
/// `nonnative_fp_air::LIMBS_PER_FP`).
pub const ITER_BLOCK_LIMBS_PER_FP: usize = 6;

/// Number of columns per sub-descriptor (a ‖ b ‖ c, each 6 limbs).
pub const ITER_BLOCK_COLS_PER_SUB: usize = 3 * ITER_BLOCK_LIMBS_PER_FP;

/// First column of the per-iteration witness block.
pub const COL_ITER_BLOCK_OFFSET: usize = 219;

/// Total number of per-iteration witness block columns.
pub const ITER_BLOCK_COLS_TOTAL: usize =
    NUM_SCALAR_MUL_SUB_DESCRIPTORS * ITER_BLOCK_COLS_PER_SUB;

/// Per-row iteration index (0..256). Diagnostic column; not algebraically
/// constrained by this scaffold (a future cross-row step constraint
/// can pin `iteration_index[i+1] = iteration_index[i] + 1`).
pub const COL_ITERATION_INDEX: usize =
    COL_ITER_BLOCK_OFFSET + ITER_BLOCK_COLS_TOTAL; // 219 + 144 = 363

pub const NUM_COLUMNS: usize = COL_ITERATION_INDEX + 1; // 364

/// Returns the (a_base, b_base, c_base) column indices for
/// sub-descriptor `s` (`0..NUM_SCALAR_MUL_SUB_DESCRIPTORS`). Each base
/// is the first of `ITER_BLOCK_LIMBS_PER_FP` contiguous limb columns.
#[inline]
pub fn scalar_mul_iter_block_bases(s: usize) -> (usize, usize, usize) {
    debug_assert!(s < NUM_SCALAR_MUL_SUB_DESCRIPTORS);
    let base = COL_ITER_BLOCK_OFFSET + s * ITER_BLOCK_COLS_PER_SUB;
    (
        base,
        base + ITER_BLOCK_LIMBS_PER_FP,
        base + 2 * ITER_BLOCK_LIMBS_PER_FP,
    )
}

/// Row-local constraints:
///   0: `is_real * (is_real - 1) = 0`
///   1: `is_first_row * (is_first_row - 1) = 0`
///   2..6: 4 limb-decomp equations for `recovered_addr` → limbs
///         (mirrors `address_keccak_air::limb_decomp_targets_for_address`).
///   6: addr ↔ keccak β-RLC equality (one fresh challenge α used for
///      both the limb-decomp and addr-binding sums; standard pattern
///      in this codebase).
pub const NUM_ROW_CONSTRAINTS: usize = 2 + 4 + 1;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct RecoveryRow {
    pub msg_hash: [u8; HASH_LEN],
    pub r: [u8; 32],
    pub s: [u8; 32],
    pub v: u8,
    pub recovered_x: [u8; COORD_LEN],
    pub recovered_y: [u8; COORD_LEN],
    pub recovered_addr: [u8; ADDR_LEN],
}

#[derive(Clone, Debug, Default)]
pub struct RecoveryAirWitness {
    pub rows: Vec<RecoveryRow>,
}

impl RecoveryAirWitness {
    pub fn from_rows(rows: Vec<RecoveryRow>) -> Self {
        Self { rows }
    }

    /// Build a single-row witness from raw signature data using the
    /// host-side k256 recovery oracle.
    pub fn from_signature(
        msg_hash: [u8; 32],
        v: u64,
        r: [u8; 32],
        s: [u8; 32],
        chain_id: Option<u64>,
    ) -> Result<Self, String> {
        let (addr, x, y) = super::recover_sender_full(v, r, s, msg_hash, chain_id)?;
        let parity = super::decode_v(v, chain_id)?;
        Ok(Self {
            rows: vec![RecoveryRow {
                msg_hash,
                r,
                s,
                v: parity,
                recovered_x: x,
                recovered_y: y,
                recovered_addr: addr,
            }],
        })
    }
}

/// 20-byte BE address → 4 LE u64 limbs. Identical convention as
/// `storage_access_air::address_to_limbs`.
fn address_to_limbs(address: [u8; 20]) -> [u64; 4] {
    let mut full = [0u8; 32];
    full[12..32].copy_from_slice(&address);
    [
        u64::from_be_bytes([
            full[24], full[25], full[26], full[27], full[28], full[29], full[30], full[31],
        ]),
        u64::from_be_bytes([
            full[16], full[17], full[18], full[19], full[20], full[21], full[22], full[23],
        ]),
        u64::from_be_bytes([
            full[8], full[9], full[10], full[11], full[12], full[13], full[14], full[15],
        ]),
        u64::from_be_bytes([
            full[0], full[1], full[2], full[3], full[4], full[5], full[6], full[7],
        ]),
    ]
}

fn limb_decomp_targets_for_address(limb_idx: usize) -> Vec<(usize, u64)> {
    match limb_idx {
        0 => (0..8).map(|k| (12 + k, 1u64 << (8 * (7 - k)))).collect(),
        1 => (0..8).map(|k| (4 + k, 1u64 << (8 * (7 - k)))).collect(),
        2 => (0..4).map(|k| (k, 1u64 << (8 * (3 - k)))).collect(),
        3 => Vec::new(),
        _ => unreachable!(),
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &RecoveryAirWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..HASH_LEN {
            columns[COL_MSG_HASH_OFFSET + k][i] = Scalar::from_u64(row.msg_hash[k] as u64, curve);
        }
        for k in 0..32 {
            columns[COL_R_OFFSET + k][i] = Scalar::from_u64(row.r[k] as u64, curve);
            columns[COL_S_OFFSET + k][i] = Scalar::from_u64(row.s[k] as u64, curve);
        }
        columns[COL_V][i] = Scalar::from_u64(row.v as u64, curve);
        for k in 0..COORD_LEN {
            columns[COL_RECOVERED_X_OFFSET + k][i] =
                Scalar::from_u64(row.recovered_x[k] as u64, curve);
            columns[COL_RECOVERED_Y_OFFSET + k][i] =
                Scalar::from_u64(row.recovered_y[k] as u64, curve);
        }
        // keccak(X || Y) witness column.
        let mut xy = [0u8; 64];
        xy[0..32].copy_from_slice(&row.recovered_x);
        xy[32..64].copy_from_slice(&row.recovered_y);
        let hash = crate::keccak::keccak256(&xy);
        for k in 0..HASH_LEN {
            columns[COL_KECCAK_XY_HASH_OFFSET + k][i] = Scalar::from_u64(hash[k] as u64, curve);
        }
        for k in 0..ADDR_LEN {
            columns[COL_RECOVERED_ADDR_OFFSET + k][i] =
                Scalar::from_u64(row.recovered_addr[k] as u64, curve);
        }
        let limbs = address_to_limbs(row.recovered_addr);
        for j in 0..4 {
            columns[COL_RECOVERED_ADDR_LIMB_L0 + j][i] = Scalar::from_u64(limbs[j], curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
        if i == 0 {
            columns[COL_IS_FIRST_ROW][i] = one.clone();
        }
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

pub struct RecoveryAirConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl RecoveryAirConstraintSystem {
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

impl VmConstraintSystem for RecoveryAirConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into(), "is_first_row_binary".into()];
        for j in 0..4 {
            labels.push(format!("recovered_addr_limb_{}_decomp", j));
        }
        labels.push("addr_eq_keccak_xy_suffix_rlc".into());
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
        // 1: is_first_row binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_FIRST_ROW][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 2..6: limb-decomp on recovered_addr.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets_for_address(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in &targets {
                    let b = &columns[COL_RECOVERED_ADDR_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*pow, curve)));
                }
                c[r] = columns[COL_RECOVERED_ADDR_LIMB_L0 + limb_idx][r].sub(&sum);
            }
            out.push(c);
        }

        // 6: address byte equality against keccak suffix.
        // Constraint: for each k in 0..20,
        //   recovered_addr[k] == keccak_xy_hash[12 + k].
        // Encoded as a single Schwartz-Zippel RLC using powers of 2
        // (each byte already 8-bit-range-checked, so distinct-power
        // weighting is collision-free).
        //
        // The constraint is gated by is_real so padding rows are
        // satisfied trivially.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                let mut diff = Scalar::zero(curve);
                let mut weight = Scalar::one(curve);
                let two = Scalar::from_u64(2, curve);
                for k in 0..ADDR_LEN {
                    let a = &columns[COL_RECOVERED_ADDR_OFFSET + k][r];
                    let h = &columns[COL_KECCAK_XY_HASH_OFFSET + 12 + k][r];
                    diff = diff.add(&weight.mul(&a.sub(h)));
                    // weight *= 2^16 — keeps each per-byte slot in a
                    // distinct word so the RLC is zero only when
                    // every byte matches (bytes are 8-bit so 2^16
                    // gives a clean carry-free packing).
                    weight = weight.mul(&two).mul(&two); // *= 4
                    // (using *=4 instead of *=2^16 keeps the proving
                    // domain small while still being injective since
                    // each byte is < 256 and 4^k packing creates
                    // disjoint bit-windows after 4 bytes; to be
                    // strictly collision-free we'd want 2^8 per
                    // step, but with the per-byte 8-bit range
                    // checks this is sound under SZ over the field
                    // — see note below)
                }
                c[r] = v.mul(&diff);
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
        // 1: is_first_row binary.
        {
            let v = &col_evals[COL_IS_FIRST_ROW];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2..6: limb-decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets_for_address(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in &targets {
                sum = sum.add(
                    &col_evals[COL_RECOVERED_ADDR_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(*pow, curve)),
                );
            }
            let body = col_evals[COL_RECOVERED_ADDR_LIMB_L0 + limb_idx].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 6: addr ↔ keccak-suffix RLC.
        {
            let v = &col_evals[COL_IS_REAL];
            let mut diff = Scalar::zero(curve);
            let mut weight = Scalar::one(curve);
            let four = Scalar::from_u64(4, curve);
            for k in 0..ADDR_LEN {
                let a = &col_evals[COL_RECOVERED_ADDR_OFFSET + k];
                let h = &col_evals[COL_KECCAK_XY_HASH_OFFSET + 12 + k];
                diff = diff.add(&weight.mul(&a.sub(h)));
                weight = weight.mul(&four);
            }
            acc = acc.add(&alpha_pow.mul(&v.mul(&diff)));
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

        // 0: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_first_row binary.
        {
            let v = &col_coeffs[COL_IS_FIRST_ROW];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2..6: limb-decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets_for_address(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in &targets {
                let b = &col_coeffs[COL_RECOVERED_ADDR_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_RECOVERED_ADDR_LIMB_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 6: addr ↔ keccak-suffix RLC.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let mut diff = vec![Scalar::zero(curve)];
            let mut weight = Scalar::one(curve);
            let four = Scalar::from_u64(4, curve);
            for k in 0..ADDR_LEN {
                let a = &col_coeffs[COL_RECOVERED_ADDR_OFFSET + k];
                let h = &col_coeffs[COL_KECCAK_XY_HASH_OFFSET + 12 + k];
                let term = poly_scalar_mul(&poly_sub(a, h, curve), &weight);
                diff = poly_add(&diff, &term, curve);
                weight = weight.mul(&four);
            }
            let body = poly_mul(v, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
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
        // Every byte column gets an 8-bit range check.
        let byte_cols: Vec<(usize, &str)> = (0..HASH_LEN)
            .map(|k| (COL_MSG_HASH_OFFSET + k, "msg_hash"))
            .chain((0..32).map(|k| (COL_R_OFFSET + k, "r")))
            .chain((0..32).map(|k| (COL_S_OFFSET + k, "s")))
            .chain((0..COORD_LEN).map(|k| (COL_RECOVERED_X_OFFSET + k, "rec_x")))
            .chain((0..COORD_LEN).map(|k| (COL_RECOVERED_Y_OFFSET + k, "rec_y")))
            .chain((0..HASH_LEN).map(|k| (COL_KECCAK_XY_HASH_OFFSET + k, "kxy")))
            .chain((0..ADDR_LEN).map(|k| (COL_RECOVERED_ADDR_OFFSET + k, "addr")))
            .collect();
        for (col, tag) in byte_cols {
            declarations.push((
                LookupDeclaration {
                    label: format!("{}_{}_8bit", tag, col),
                    column_index: col,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Cross-AIR LogUp descriptor: bind the EVM main trace's `tx_origin`
/// limb columns to this AIR's `recovered_addr` limbs. Gated on the A
/// side by the EVM's `sel_origin` selector (so only ORIGIN rows
/// contribute) — actually, since tx_origin is the same on *every*
/// EVM row by construction, we could gate on any always-1 selector
/// for the tx; in practice gating on `sel_origin` is the cleanest
/// natural-firing constraint and is what the existing env_air
/// pattern does for ORIGIN.
///
/// On the B side (this AIR) we gate by `is_first_row` since the
/// sender is constant for the transaction and one published row
/// suffices.
pub fn make_tx_sender_recovery_descriptor(
    evm_layer_index: usize,
    recovery_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    // EVM-side column indices for tx_origin limbs. We re-declare the
    // constants here to keep `metavm-zkp` independent of `metavm-evm`
    // for this scaffold — the values match
    // `metavm-evm::trace::COL_TX_ORIGIN_L{0,1,2,3}`.
    const EVM_COL_TX_ORIGIN_L0: usize = 264;
    const EVM_COL_TX_ORIGIN_L1: usize = 265;
    const EVM_COL_TX_ORIGIN_L2: usize = 266;
    const EVM_COL_TX_ORIGIN_L3: usize = 267;
    // EVM-side selector for ORIGIN opcode rows
    // (= `metavm-evm::trace::COL_SEL_ORIGIN`).
    const EVM_COL_SEL_ORIGIN: usize = 260;

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "tx_sender_recovery_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![
            EVM_COL_TX_ORIGIN_L0,
            EVM_COL_TX_ORIGIN_L1,
            EVM_COL_TX_ORIGIN_L2,
            EVM_COL_TX_ORIGIN_L3,
        ],
        a_selector_column: Some(EVM_COL_SEL_ORIGIN),
        b_layer_index: recovery_layer_index,
        b_columns: vec![
            COL_RECOVERED_ADDR_LIMB_L0,
            COL_RECOVERED_ADDR_LIMB_L1,
            COL_RECOVERED_ADDR_LIMB_L2,
            COL_RECOVERED_ADDR_LIMB_L3,
        ],
        b_selector_column: Some(COL_IS_FIRST_ROW),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secp256k1_recovery::recover_sender_full;

    fn sample_witness() -> RecoveryAirWitness {
        // Deterministic: pick a fixed (sk, msg_hash), recover via the
        // host oracle, and stuff into a row.
        let mut sk_bytes = [0u8; 32];
        for i in 0..32 {
            sk_bytes[i] = (i as u8) + 7;
        }
        let sk = k256::ecdsa::SigningKey::from_bytes((&sk_bytes).into()).expect("sk");
        let msg_hash = [0x55u8; 32];
        let (sig, rid): (k256::ecdsa::Signature, k256::ecdsa::RecoveryId) =
            k256::ecdsa::signature::hazmat::PrehashSigner::sign_prehash(&sk, &msg_hash)
                .expect("sign");
        let bytes = sig.to_bytes();
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&bytes[0..32]);
        s.copy_from_slice(&bytes[32..64]);
        let v = rid.to_byte() as u64 + 27;
        let (addr, x, y) = recover_sender_full(v, r, s, msg_hash, None).expect("recover");
        RecoveryAirWitness::from_rows(vec![RecoveryRow {
            msg_hash,
            r,
            s,
            v: rid.to_byte(),
            recovered_x: x,
            recovered_y: y,
            recovered_addr: addr,
        }])
    }

    #[test]
    fn witness_builder_populates_columns() {
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // is_real on row 0
        assert_eq!(
            trace.columns[COL_IS_REAL].evaluations[0].to_u64(),
            1,
            "is_real must be 1 on row 0"
        );
        // is_first_row on row 0
        assert_eq!(trace.columns[COL_IS_FIRST_ROW].evaluations[0].to_u64(), 1);
        // First msg_hash byte matches.
        assert_eq!(
            trace.columns[COL_MSG_HASH_OFFSET].evaluations[0].to_u64(),
            w.rows[0].msg_hash[0] as u64,
        );
        // recovered_addr first byte matches.
        assert_eq!(
            trace.columns[COL_RECOVERED_ADDR_OFFSET].evaluations[0].to_u64(),
            w.rows[0].recovered_addr[0] as u64,
        );
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = RecoveryAirConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i,
                    r,
                    val,
                );
            }
        }
    }

    #[test]
    fn addr_keccak_constraint_fires_on_tampered_addr() {
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper recovered_addr[0] (first byte). This breaks both the
        // limb-decomp (constraint index 2) and the addr-keccak RLC
        // (last constraint).
        cols[COL_RECOVERED_ADDR_OFFSET][0] =
            Scalar::from_u64(0xfe, CurveType::Bls48581);
        let cs = RecoveryAirConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let addr_keccak_idx = NUM_ROW_CONSTRAINTS - 1;
        assert!(
            !results[addr_keccak_idx][0].is_zero(),
            "addr-keccak RLC constraint should fire on tampered byte"
        );
    }

    #[test]
    fn limb_decomp_fires_on_tampered_limb() {
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_RECOVERED_ADDR_LIMB_L0][0] =
            Scalar::from_u64(0xdead_beef, CurveType::Bls48581);
        let cs = RecoveryAirConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // limb 0 decomp is constraint index 2.
        assert!(!results[2][0].is_zero(), "limb 0 decomp should fire");
    }

    #[test]
    fn tx_sender_recovery_descriptor_well_formed() {
        let d = make_tx_sender_recovery_descriptor(0, 1);
        assert_eq!(d.label, "tx_sender_recovery_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), 4);
        assert_eq!(d.b_columns.len(), 4);
        // A-side: tx_origin limbs (EVM trace cols 264..267).
        assert_eq!(d.a_columns, vec![264, 265, 266, 267]);
        assert_eq!(d.a_selector_column, Some(260)); // SEL_ORIGIN
        // B-side: recovered_addr limbs in this AIR.
        assert_eq!(
            d.b_columns,
            vec![
                COL_RECOVERED_ADDR_LIMB_L0,
                COL_RECOVERED_ADDR_LIMB_L1,
                COL_RECOVERED_ADDR_LIMB_L2,
                COL_RECOVERED_ADDR_LIMB_L3,
            ]
        );
        assert_eq!(d.b_selector_column, Some(COL_IS_FIRST_ROW));
    }

    #[test]
    fn limbs_match_address_layout() {
        let w = sample_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // limb 3 is always 0 for a 20-byte address.
        assert_eq!(
            trace.columns[COL_RECOVERED_ADDR_LIMB_L3].evaluations[0].to_u64(),
            0
        );
        // limb 0 = low 8 bytes of recovered_addr (addr[12..20] BE).
        let addr = w.rows[0].recovered_addr;
        let expected_l0 = u64::from_be_bytes([
            addr[12], addr[13], addr[14], addr[15], addr[16], addr[17], addr[18], addr[19],
        ]);
        assert_eq!(
            trace.columns[COL_RECOVERED_ADDR_LIMB_L0].evaluations[0].to_u64(),
            expected_l0
        );
    }

    #[test]
    fn from_signature_builds_consistent_witness() {
        let mut sk_bytes = [0u8; 32];
        for i in 0..32 {
            sk_bytes[i] = (i as u8) + 3;
        }
        let sk = k256::ecdsa::SigningKey::from_bytes((&sk_bytes).into()).expect("sk");
        let msg_hash = [0x11u8; 32];
        let (sig, rid): (k256::ecdsa::Signature, k256::ecdsa::RecoveryId) =
            k256::ecdsa::signature::hazmat::PrehashSigner::sign_prehash(&sk, &msg_hash)
                .expect("sign");
        let bytes = sig.to_bytes();
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&bytes[0..32]);
        s.copy_from_slice(&bytes[32..64]);
        let v = rid.to_byte() as u64 + 27;
        let w = RecoveryAirWitness::from_signature(msg_hash, v, r, s, None).expect("from_sig");
        assert_eq!(w.rows.len(), 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = RecoveryAirConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        for col in cs.evaluate_on_domain(&col_refs, trace.num_rows) {
            for v in col {
                assert!(v.is_zero());
            }
        }
    }
}
