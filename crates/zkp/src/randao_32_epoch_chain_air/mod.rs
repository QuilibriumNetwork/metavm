//! 32-epoch RANDAO chain AIR.
//!
//! Composes 32 iterations of the single-mix RANDAO update pattern
//! (`randao_chain_air`) into one AIR whose trace represents a full
//! historical rotation window of 32 consecutive epochs (one row per
//! epoch).
//!
//! Beacon-chain spec:
//!
//!   `state.randao_mixes[epoch % 65536] = current_mix XOR sha256(reveal)`
//!
//! At each epoch the RANDAO mix evolves by XOR-ing with the SHA-256 hash
//! of the validator's randao reveal signature. This AIR proves 32 such
//! evolutions and chains them via a β-RLC cross-row equality body:
//!
//!   `prev_mix[i+1] = new_mix[i]` (β-RLC over 32 bytes, gated by
//!   `is_real(X) · is_real(ω·X)`, with boundary exclusion).
//!
//! ## Row columns (per epoch)
//!
//!   - `prev_mix[0..32]`    — 32 BE bytes of the prior RANDAO mix.
//!   - `reveal_sig[0..96]`  — 96 BE bytes of the BLS RANDAO reveal sig.
//!   - `reveal_hash[0..32]` — `sha256(reveal_sig)` output bytes.
//!   - `new_mix[0..32]`     — 32 BE bytes of the updated RANDAO mix.
//!   - `and_vals[0..32]`    — per-byte witness for `prev_mix[k] AND
//!     reveal_hash[k]`, encoding XOR as `new = prev + hash - 2 * and`.
//!   - `new_mix_limb[0..4]` — 4 LE u64 limbs of `new_mix`, matching
//!     `block_header_air::prev_randao` BE→LE convention.
//!   - `epoch`              — u64 epoch index.
//!   - `epoch_byte[0..8]`   — 8 LE bytes of `epoch`, bound via LE decomp.
//!   - `is_real`            — 1 on real rows, 0 on padding.
//!   - `is_first`           — 1 on the first row, 0 elsewhere.
//!   - `is_last`            — 1 on the final (32nd) row, 0 elsewhere.
//!
//! All non-selector / non-epoch byte columns are byte-range checked.
//!
//! ## Row-local constraints (40 bodies)
//!
//! Same shape as `randao_chain_air` plus one extra binary selector:
//!
//! 0. `is_real_binary`     — `is_real * (is_real - 1) = 0`.
//! 1. `is_first_binary`    — `is_first * (is_first - 1) = 0`.
//! 2. `is_last_binary`     — `is_last * (is_last - 1) = 0`.
//! 3..35. Per-byte XOR identity.
//! 35..39. New-mix LE u64 limb decomp (4 limbs, BE→LE convention).
//! 39. `epoch_le_decomp`   — `epoch - Σ_b epoch_byte[b] * 2^(8b) = 0`.
//!
//! ## Shifted constraint (1 body, cross-row)
//!
//! 0. `chain_prev_eq_new` —
//!    `is_real(X) * is_real(ω·X) * Σ_k α^k * (PREV_MIX_k(ω·X) -
//!    NEW_MIX_k(X)) = 0`, multiplied by `(z - ω^{n-1})` to exclude the
//!    wrap-around row. Same β-RLC trick as `randao_chain_air`.
//!
//! ## Cross-AIR LogUp descriptors
//!
//!   - `make_randao_32_to_randao_chain_descriptor` — per-row link of
//!     `(prev_mix, reveal_sig, reveal_hash, new_mix)` (192 byte cols)
//!     against the single-mix `randao_chain_air`. Algebraically asserts
//!     each epoch's update is exactly a row in the canonical single-mix
//!     chain — soundness is delegated to that AIR.
//!   - `make_randao_32_root_mix_to_block_header_descriptor` — binds the
//!     **last** row's `new_mix_limb[0..4]` to `block_header_air::
//!     prev_randao[0..4]`, gated by this AIR's `is_last` selector. This
//!     pins the rotation-window root mix to the corresponding block
//!     header's RANDAO field.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const MIX_LEN: usize = 32;
pub const HASH_LEN: usize = 32;
pub const REVEAL_SIG_LEN: usize = 96;
pub const U64_BYTES: usize = 8;

/// Number of epochs (rows) in one rotation window.
pub const EPOCH_COUNT: usize = 32;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_PREV_MIX_OFFSET: usize = 0;                                       // 0..32
pub const COL_REVEAL_SIG_OFFSET: usize = COL_PREV_MIX_OFFSET + MIX_LEN;         // 32..128
pub const COL_REVEAL_HASH_OFFSET: usize = COL_REVEAL_SIG_OFFSET + REVEAL_SIG_LEN; // 128..160
pub const COL_NEW_MIX_OFFSET: usize = COL_REVEAL_HASH_OFFSET + HASH_LEN;        // 160..192
pub const COL_AND_VALS_OFFSET: usize = COL_NEW_MIX_OFFSET + MIX_LEN;            // 192..224
pub const COL_NEW_MIX_LIMB_L0: usize = COL_AND_VALS_OFFSET + MIX_LEN;           // 224
pub const COL_NEW_MIX_LIMB_L1: usize = COL_NEW_MIX_LIMB_L0 + 1;                 // 225
pub const COL_NEW_MIX_LIMB_L2: usize = COL_NEW_MIX_LIMB_L0 + 2;                 // 226
pub const COL_NEW_MIX_LIMB_L3: usize = COL_NEW_MIX_LIMB_L0 + 3;                 // 227
pub const COL_EPOCH: usize = COL_NEW_MIX_LIMB_L3 + 1;                           // 228
pub const COL_EPOCH_BYTE_OFFSET: usize = COL_EPOCH + 1;                         // 229..237
pub const COL_IS_REAL: usize = COL_EPOCH_BYTE_OFFSET + U64_BYTES;               // 237
pub const COL_IS_FIRST: usize = COL_IS_REAL + 1;                                // 238
pub const COL_IS_LAST: usize = COL_IS_FIRST + 1;                                // 239
pub const NUM_COLUMNS: usize = COL_IS_LAST + 1;                                 // 240

/// Row-local constraints:
///   0: is_real binary
///   1: is_first binary
///   2: is_last binary
///   3..35: per-byte XOR identity (32 bodies)
///   35..39: new_mix limb decomp (4 bodies)
///   39: epoch LE decomp
pub const NUM_ROW_CONSTRAINTS: usize = 3 + MIX_LEN + 4 + 1;
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Randao32EpochRow {
    pub prev_mix: [u8; MIX_LEN],
    pub reveal_sig: [u8; REVEAL_SIG_LEN],
    pub reveal_hash: [u8; HASH_LEN],
    pub new_mix: [u8; MIX_LEN],
    pub epoch: u64,
    pub is_first: bool,
    pub is_last: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Randao32EpochWitness {
    pub rows: Vec<Randao32EpochRow>,
}

impl Randao32EpochWitness {
    /// Build a 32-epoch rotation-window witness from an `initial_mix`,
    /// a `start_epoch`, and a list of 32 reveal signatures (one per
    /// epoch). Each row computes `reveal_hash = sha256(reveal_sig)` and
    /// chains `new_mix = prev_mix XOR reveal_hash`. The first row has
    /// `is_first = true` and the last row has `is_last = true`.
    pub fn from_window(
        initial_mix: [u8; MIX_LEN],
        start_epoch: u64,
        reveals: &[[u8; REVEAL_SIG_LEN]; EPOCH_COUNT],
    ) -> Self {
        let mut rows = Vec::with_capacity(EPOCH_COUNT);
        let mut prev = initial_mix;
        for i in 0..EPOCH_COUNT {
            let sig = reveals[i];
            let reveal_hash = crate::sha256::sha256(&sig);
            let mut new_mix = [0u8; MIX_LEN];
            for k in 0..MIX_LEN {
                new_mix[k] = prev[k] ^ reveal_hash[k];
            }
            rows.push(Randao32EpochRow {
                prev_mix: prev,
                reveal_sig: sig,
                reveal_hash,
                new_mix,
                epoch: start_epoch + i as u64,
                is_first: i == 0,
                is_last: i == EPOCH_COUNT - 1,
            });
            prev = new_mix;
        }
        Self { rows }
    }

    pub fn from_rows(rows: Vec<Randao32EpochRow>) -> Self {
        Self { rows }
    }
}

/// 32-byte BE → 4 LE u64 limbs (matches `block_header_air::prev_randao`).
fn mix_to_limbs(mix: &[u8; MIX_LEN]) -> [u64; 4] {
    [
        u64::from_be_bytes([
            mix[24], mix[25], mix[26], mix[27], mix[28], mix[29], mix[30], mix[31],
        ]),
        u64::from_be_bytes([
            mix[16], mix[17], mix[18], mix[19], mix[20], mix[21], mix[22], mix[23],
        ]),
        u64::from_be_bytes([
            mix[8], mix[9], mix[10], mix[11], mix[12], mix[13], mix[14], mix[15],
        ]),
        u64::from_be_bytes([
            mix[0], mix[1], mix[2], mix[3], mix[4], mix[5], mix[6], mix[7],
        ]),
    ]
}

fn new_mix_limb_decomp_targets(limb_idx: usize) -> Vec<(usize, u64)> {
    match limb_idx {
        0 => (0..8).map(|k| (24 + k, 1u64 << (8 * (7 - k)))).collect(),
        1 => (0..8).map(|k| (16 + k, 1u64 << (8 * (7 - k)))).collect(),
        2 => (0..8).map(|k| (8 + k, 1u64 << (8 * (7 - k)))).collect(),
        3 => (0..8).map(|k| (k, 1u64 << (8 * (7 - k)))).collect(),
        _ => unreachable!(),
    }
}

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Randao32EpochWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..MIX_LEN {
            columns[COL_PREV_MIX_OFFSET + k][i] =
                Scalar::from_u64(row.prev_mix[k] as u64, curve);
        }
        for k in 0..REVEAL_SIG_LEN {
            columns[COL_REVEAL_SIG_OFFSET + k][i] =
                Scalar::from_u64(row.reveal_sig[k] as u64, curve);
        }
        for k in 0..HASH_LEN {
            columns[COL_REVEAL_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.reveal_hash[k] as u64, curve);
        }
        for k in 0..MIX_LEN {
            columns[COL_NEW_MIX_OFFSET + k][i] =
                Scalar::from_u64(row.new_mix[k] as u64, curve);
        }
        for k in 0..MIX_LEN {
            let and_byte = row.prev_mix[k] & row.reveal_hash[k];
            columns[COL_AND_VALS_OFFSET + k][i] =
                Scalar::from_u64(and_byte as u64, curve);
        }
        let limbs = mix_to_limbs(&row.new_mix);
        for j in 0..4 {
            columns[COL_NEW_MIX_LIMB_L0 + j][i] = Scalar::from_u64(limbs[j], curve);
        }
        columns[COL_EPOCH][i] = Scalar::from_u64(row.epoch, curve);
        let epoch_bytes = row.epoch.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_EPOCH_BYTE_OFFSET + b][i] =
                Scalar::from_u64(epoch_bytes[b] as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_FIRST][i] = if row.is_first { one.clone() } else { zero.clone() };
        columns[COL_IS_LAST][i] = if row.is_last { one.clone() } else { zero.clone() };
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

pub struct Randao32EpochChainConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Randao32EpochChainConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Randao32EpochChainConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = vec![
            "is_real_binary".into(),
            "is_first_binary".into(),
            "is_last_binary".into(),
        ];
        for k in 0..MIX_LEN {
            labels.push(format!("xor_byte_{}", k));
        }
        for j in 0..4 {
            labels.push(format!("new_mix_limb_{}_decomp", j));
        }
        labels.push("epoch_le_decomp".into());
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
        let two = Scalar::from_u64(2, curve);
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

        // 1: is_first binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_FIRST][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 2: is_last binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_LAST][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 3..35: per-byte XOR identity.
        for k in 0..MIX_LEN {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let prev_b = &columns[COL_PREV_MIX_OFFSET + k][r];
                let hash_b = &columns[COL_REVEAL_HASH_OFFSET + k][r];
                let new_b = &columns[COL_NEW_MIX_OFFSET + k][r];
                let and_b = &columns[COL_AND_VALS_OFFSET + k][r];
                let two_and = two.mul(and_b);
                let body = new_b.sub(prev_b).sub(hash_b).add(&two_and);
                c[r] = body;
            }
            out.push(c);
        }

        // 35..39: new_mix limb decomp.
        for limb_idx in 0..4 {
            let targets = new_mix_limb_decomp_targets(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in &targets {
                    let b = &columns[COL_NEW_MIX_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*pow, curve)));
                }
                c[r] = columns[COL_NEW_MIX_LIMB_L0 + limb_idx][r].sub(&sum);
            }
            out.push(c);
        }

        // 39: epoch LE decomp.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    let byte = &columns[COL_EPOCH_BYTE_OFFSET + b][r];
                    sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
                }
                c[r] = columns[COL_EPOCH][r].sub(&sum);
            }
            out.push(c);
        }

        out
    }

    #[allow(unused_assignments)]
    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(2, curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_first binary.
        {
            let v = &col_evals[COL_IS_FIRST];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: is_last binary.
        {
            let v = &col_evals[COL_IS_LAST];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3..35: per-byte XOR identity.
        for k in 0..MIX_LEN {
            let prev_b = &col_evals[COL_PREV_MIX_OFFSET + k];
            let hash_b = &col_evals[COL_REVEAL_HASH_OFFSET + k];
            let new_b = &col_evals[COL_NEW_MIX_OFFSET + k];
            let and_b = &col_evals[COL_AND_VALS_OFFSET + k];
            let two_and = two.mul(and_b);
            let body = new_b.sub(prev_b).sub(hash_b).add(&two_and);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 35..39: limb decomp.
        for limb_idx in 0..4 {
            let targets = new_mix_limb_decomp_targets(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in &targets {
                sum = sum.add(
                    &col_evals[COL_NEW_MIX_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(*pow, curve)),
                );
            }
            let body = col_evals[COL_NEW_MIX_LIMB_L0 + limb_idx].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 39: epoch LE decomp.
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                let byte = &col_evals[COL_EPOCH_BYTE_OFFSET + b];
                sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
            }
            let body = col_evals[COL_EPOCH].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        acc
    }

    #[allow(unused_assignments)]
    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let two_scalar = Scalar::from_u64(2, curve);

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
        // 1: is_first binary.
        {
            let v = &col_coeffs[COL_IS_FIRST];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: is_last binary.
        {
            let v = &col_coeffs[COL_IS_LAST];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3..35: per-byte XOR identity.
        for k in 0..MIX_LEN {
            let prev_b = &col_coeffs[COL_PREV_MIX_OFFSET + k];
            let hash_b = &col_coeffs[COL_REVEAL_HASH_OFFSET + k];
            let new_b = &col_coeffs[COL_NEW_MIX_OFFSET + k];
            let and_b = &col_coeffs[COL_AND_VALS_OFFSET + k];
            let two_and = poly_scalar_mul(and_b, &two_scalar);
            let body = poly_add(
                &poly_sub(&poly_sub(new_b, prev_b, curve), hash_b, curve),
                &two_and,
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 35..39: limb decomp.
        for limb_idx in 0..4 {
            let targets = new_mix_limb_decomp_targets(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in &targets {
                let b = &col_coeffs[COL_NEW_MIX_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_NEW_MIX_LIMB_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 39: epoch LE decomp.
        {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..U64_BYTES {
                let byte_poly = &col_coeffs[COL_EPOCH_BYTE_OFFSET + b];
                let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_EPOCH], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
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
        for k in 0..MIX_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("prev_mix_{}_8bit", k),
                    column_index: COL_PREV_MIX_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..REVEAL_SIG_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("reveal_sig_{}_8bit", k),
                    column_index: COL_REVEAL_SIG_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..HASH_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("reveal_hash_{}_8bit", k),
                    column_index: COL_REVEAL_HASH_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MIX_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("new_mix_{}_8bit", k),
                    column_index: COL_NEW_MIX_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MIX_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("and_vals_{}_8bit", k),
                    column_index: COL_AND_VALS_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("epoch_byte_{}_8bit", b),
                    column_index: COL_EPOCH_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // ω·z evaluations layout (total = 1 + MIX_LEN = 33):
        //   [0] IS_REAL_NEXT
        //   [1..33] PREV_MIX_NEXT[0..32]
        let mut cols = vec![COL_IS_REAL];
        for k in 0..MIX_LEN {
            cols.push(COL_PREV_MIX_OFFSET + k);
        }
        cols
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
        if shifted_evals.len() != 1 + MIX_LEN || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let is_real = &col_evals_at_z[COL_IS_REAL];
        let is_real_next = &shifted_evals[0];
        let gating = is_real.mul(is_real_next);

        let mut rlc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MIX_LEN {
            let prev_next_k = &shifted_evals[1 + k];
            let new_k = &col_evals_at_z[COL_NEW_MIX_OFFSET + k];
            rlc = rlc.add(&bp.mul(&prev_next_k.sub(new_k)));
            bp = bp.mul(alpha);
        }
        let body = gating.mul(&rlc);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let total = ap.mul(&body);

        // Boundary exclusion: multiply by (z - ω^{n-1}).
        total.mul(&z.sub(omega_n_minus_1))
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();

        let is_real = &column_coeffs[COL_IS_REAL];
        let is_real_next = poly_shift(is_real, omega);
        let gating = poly_mul(is_real, &is_real_next, curve);

        let mut rlc_poly = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..MIX_LEN {
            let prev_k = &column_coeffs[COL_PREV_MIX_OFFSET + k];
            let prev_k_next = poly_shift(prev_k, omega);
            let new_k = &column_coeffs[COL_NEW_MIX_OFFSET + k];
            let diff = poly_sub(&prev_k_next, new_k, curve);
            rlc_poly = poly_add(&rlc_poly, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body = poly_mul(&gating, &rlc_poly, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut total = poly_scalar_mul(&body, &ap);

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let factor = vec![neg, Scalar::one(curve)];
        total = poly_mul(&total, &factor, curve);
        total
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Per-row link of `(prev_mix, reveal_sig, reveal_hash, new_mix)` (192
/// columns total) to the single-mix `randao_chain_air`. Delegates the
/// per-row XOR + sha256 + chain soundness to that AIR; this AIR adds the
/// 32-row cross-row composition.
pub fn make_randao_32_to_randao_chain_descriptor(
    randao_32_layer_index: usize,
    randao_chain_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::randao_chain_air as rc;
    let mut a_columns: Vec<usize> = Vec::with_capacity(192);
    for k in 0..MIX_LEN {
        a_columns.push(COL_PREV_MIX_OFFSET + k);
    }
    for k in 0..REVEAL_SIG_LEN {
        a_columns.push(COL_REVEAL_SIG_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_REVEAL_HASH_OFFSET + k);
    }
    for k in 0..MIX_LEN {
        a_columns.push(COL_NEW_MIX_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(192);
    for k in 0..MIX_LEN {
        b_columns.push(rc::COL_PREV_MIX_OFFSET + k);
    }
    for k in 0..REVEAL_SIG_LEN {
        b_columns.push(rc::COL_REVEAL_SIG_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        b_columns.push(rc::COL_REVEAL_HASH_OFFSET + k);
    }
    for k in 0..MIX_LEN {
        b_columns.push(rc::COL_NEW_MIX_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "randao_32_epoch_chain_to_randao_chain_v1".into(),
        a_layer_index: randao_32_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: randao_chain_layer_index,
        b_columns,
        b_selector_column: Some(rc::COL_IS_REAL),
    }
}

/// Bind the **last** row's `new_mix_limb[0..4]` (i.e. the
/// rotation-window root mix) to `block_header_air::prev_randao[0..4]`.
/// Gated by `is_last` on the A side so only the final epoch's output
/// contributes; gated by `is_real` on the B side.
pub fn make_randao_32_root_mix_to_block_header_descriptor(
    randao_32_layer_index: usize,
    block_header_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    CrossAirLogUpDescriptor {
        label: "randao_32_epoch_chain_root_mix_to_block_header_v1".into(),
        a_layer_index: randao_32_layer_index,
        a_columns: vec![
            COL_NEW_MIX_LIMB_L0,
            COL_NEW_MIX_LIMB_L1,
            COL_NEW_MIX_LIMB_L2,
            COL_NEW_MIX_LIMB_L3,
        ],
        a_selector_column: Some(COL_IS_LAST),
        b_layer_index: block_header_layer_index,
        b_columns: vec![
            bh::COL_PREV_RANDAO_L0,
            bh::COL_PREV_RANDAO_L1,
            bh::COL_PREV_RANDAO_L2,
            bh::COL_PREV_RANDAO_L3,
        ],
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reveals(byte_seed: u8) -> [[u8; REVEAL_SIG_LEN]; EPOCH_COUNT] {
        let mut out = [[0u8; REVEAL_SIG_LEN]; EPOCH_COUNT];
        for i in 0..EPOCH_COUNT {
            for k in 0..REVEAL_SIG_LEN {
                out[i][k] = byte_seed
                    .wrapping_add(i as u8)
                    .wrapping_mul(7)
                    .wrapping_add(k as u8);
            }
        }
        out
    }

    fn check_all_row_bodies_vanish(witness: &Randao32EpochWitness) {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(witness, curve);
        let cs = Randao32EpochChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_PREV_MIX_OFFSET, 0);
        assert_eq!(COL_REVEAL_SIG_OFFSET, 32);
        assert_eq!(COL_REVEAL_HASH_OFFSET, 128);
        assert_eq!(COL_NEW_MIX_OFFSET, 160);
        assert_eq!(COL_AND_VALS_OFFSET, 192);
        assert_eq!(COL_NEW_MIX_LIMB_L0, 224);
        assert_eq!(COL_EPOCH, 228);
        assert_eq!(COL_EPOCH_BYTE_OFFSET, 229);
        assert_eq!(COL_IS_REAL, 237);
        assert_eq!(COL_IS_FIRST, 238);
        assert_eq!(COL_IS_LAST, 239);
        assert_eq!(NUM_COLUMNS, 240);
        assert_eq!(NUM_ROW_CONSTRAINTS, 3 + 32 + 4 + 1);
        assert_eq!(NUM_SHIFTED, 1);
        assert_eq!(EPOCH_COUNT, 32);
    }

    #[test]
    fn full_32_epoch_window_constraints_vanish() {
        let initial_mix = [0xa5u8; MIX_LEN];
        let reveals = make_reveals(0x11);
        let w = Randao32EpochWitness::from_window(initial_mix, 1000, &reveals);
        assert_eq!(w.rows.len(), EPOCH_COUNT);
        // Chain wiring host-side: row[i+1].prev_mix == row[i].new_mix.
        for i in 0..EPOCH_COUNT - 1 {
            assert_eq!(w.rows[i + 1].prev_mix, w.rows[i].new_mix);
        }
        // is_first / is_last selectors.
        assert!(w.rows[0].is_first);
        assert!(!w.rows[0].is_last);
        assert!(!w.rows[EPOCH_COUNT - 1].is_first);
        assert!(w.rows[EPOCH_COUNT - 1].is_last);
        // Epoch counter increments.
        for i in 0..EPOCH_COUNT {
            assert_eq!(w.rows[i].epoch, 1000 + i as u64);
        }
        check_all_row_bodies_vanish(&w);
    }

    #[test]
    fn cross_row_chain_body_vanishes_for_honest_window() {
        let initial_mix = [0u8; MIX_LEN];
        let reveals = make_reveals(0x33);
        let w = Randao32EpochWitness::from_window(initial_mix, 0, &reveals);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let beta = Scalar::from_u64(0xC0FFEE, curve);
        // Verify the β-RLC body on every real→real transition.
        for transition in 0..EPOCH_COUNT - 1 {
            let is_real = &col_refs[COL_IS_REAL][transition];
            let is_real_next = &col_refs[COL_IS_REAL][transition + 1];
            let gating = is_real.mul(is_real_next);
            let mut rlc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..MIX_LEN {
                let prev_next_k = &col_refs[COL_PREV_MIX_OFFSET + k][transition + 1];
                let new_k = &col_refs[COL_NEW_MIX_OFFSET + k][transition];
                rlc = rlc.add(&bp.mul(&prev_next_k.sub(new_k)));
                bp = bp.mul(&beta);
            }
            let body = gating.mul(&rlc);
            assert!(
                body.is_zero(),
                "honest β-RLC chain body must vanish at transition {}",
                transition,
            );
        }
    }

    #[test]
    fn tampered_chain_link_fires_shifted_body() {
        // Break the prev_mix=new_mix wiring between rows 7 and 8.
        let initial_mix = [0u8; MIX_LEN];
        let reveals = make_reveals(0x55);
        let w = Randao32EpochWitness::from_window(initial_mix, 0, &reveals);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper prev_mix[12] on row 8.
        cols[COL_PREV_MIX_OFFSET + 12][8] = Scalar::from_u64(0xDE, curve);

        let beta = Scalar::from_u64(0x42, curve);
        let is_real = &cols[COL_IS_REAL][7];
        let is_real_next = &cols[COL_IS_REAL][8];
        let gating = is_real.mul(is_real_next);
        let mut rlc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MIX_LEN {
            let prev_next_k = &cols[COL_PREV_MIX_OFFSET + k][8];
            let new_k = &cols[COL_NEW_MIX_OFFSET + k][7];
            rlc = rlc.add(&bp.mul(&prev_next_k.sub(new_k)));
            bp = bp.mul(&beta);
        }
        let body = gating.mul(&rlc);
        assert!(
            !body.is_zero(),
            "tampered chain link must fire β-RLC shifted body",
        );

        // Also: the row-local XOR identity on row 8 byte 12 fires
        // because new_mix[8] was computed against the original prev_mix.
        let cs = Randao32EpochChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // XOR byte index k is constraint 3 + k.
        assert!(
            !bodies[3 + 12][8].is_zero(),
            "XOR byte_12 constraint should fire on tampered prev_mix[12]",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_randao_32_to_randao_chain_descriptor(0, 1);
        assert_eq!(d1.label, "randao_32_epoch_chain_to_randao_chain_v1");
        // 32 prev + 96 sig + 32 hash + 32 new = 192.
        assert_eq!(d1.a_columns.len(), 192);
        assert_eq!(d1.b_columns.len(), 192);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        // First 32 cols are prev_mix on both sides.
        for k in 0..MIX_LEN {
            assert_eq!(d1.a_columns[k], COL_PREV_MIX_OFFSET + k);
        }

        let d2 = make_randao_32_root_mix_to_block_header_descriptor(0, 2);
        assert_eq!(
            d2.label,
            "randao_32_epoch_chain_root_mix_to_block_header_v1"
        );
        assert_eq!(d2.a_columns.len(), 4);
        assert_eq!(d2.b_columns.len(), 4);
        assert_eq!(d2.a_columns[0], COL_NEW_MIX_LIMB_L0);
        assert_eq!(d2.a_columns[3], COL_NEW_MIX_LIMB_L3);
        // Critical: root-mix linkage is gated by IS_LAST not IS_REAL.
        assert_eq!(d2.a_selector_column, Some(COL_IS_LAST));
    }

    #[test]
    fn epoch_le_decomp_fires_on_tampered_byte() {
        let initial_mix = [0u8; MIX_LEN];
        let reveals = make_reveals(0x77);
        let w = Randao32EpochWitness::from_window(initial_mix, 0x1234_5678_9abc_def0, &reveals);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Zero out epoch byte 2 on row 5 without updating the u64 column.
        cols[COL_EPOCH_BYTE_OFFSET + 2][5] = Scalar::from_u64(0, curve);
        let cs = Randao32EpochChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // epoch_le_decomp is the last constraint.
        assert!(
            !bodies[NUM_ROW_CONSTRAINTS - 1][5].is_zero(),
            "epoch_le_decomp must fire on tampered epoch byte",
        );
    }

    #[test]
    fn is_last_selector_isolates_root_mix_row() {
        let initial_mix = [0u8; MIX_LEN];
        let reveals = make_reveals(0x99);
        let w = Randao32EpochWitness::from_window(initial_mix, 100, &reveals);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        // Exactly one row has IS_LAST = 1, and it's row 31.
        let mut last_rows = 0usize;
        for r in 0..EPOCH_COUNT {
            let v = &trace.columns[COL_IS_LAST].evaluations[r];
            if !v.is_zero() {
                last_rows += 1;
                assert_eq!(r, EPOCH_COUNT - 1);
                assert_eq!(v.to_u64(), 1);
            }
        }
        assert_eq!(last_rows, 1);

        // The is_last row's new_mix matches the final XOR-folded mix
        // host-side.
        let mut expected = initial_mix;
        for i in 0..EPOCH_COUNT {
            let h = crate::sha256::sha256(&reveals[i]);
            for k in 0..MIX_LEN {
                expected[k] ^= h[k];
            }
        }
        assert_eq!(w.rows[EPOCH_COUNT - 1].new_mix, expected);
    }

    #[test]
    fn shifted_column_indices_match_layout() {
        let cs = Randao32EpochChainConstraintSystem::new(EPOCH_COUNT);
        let cols = cs.shifted_column_indices();
        assert_eq!(cols.len(), 1 + MIX_LEN);
        assert_eq!(cols[0], COL_IS_REAL);
        for k in 0..MIX_LEN {
            assert_eq!(cols[1 + k], COL_PREV_MIX_OFFSET + k);
        }
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
    }
}
