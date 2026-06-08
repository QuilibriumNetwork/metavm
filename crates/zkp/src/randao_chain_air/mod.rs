//! RANDAO mix multi-block chain AIR.
//!
//! Proves the RANDAO mix evolves correctly across a sequence of
//! consecutive beacon blocks:
//!
//!   `mix[i+1] = mix[i] XOR sha256(reveal_sig[i+1])`
//!
//! Each row represents one block's RANDAO update. Compared to
//! `randao_proposer_air` (single-block), this AIR adds a **cross-row
//! chain constraint** that binds row `i`'s `new_mix` to row `i+1`'s
//! `prev_mix` via a β-RLC compressed equality over all 32 bytes, gated
//! by `is_real(X) · is_real(ω·X)`. The first row's `prev_mix` is left
//! as a free input column — the chain's initial mix is pinned
//! externally (e.g., via a boundary descriptor or by binding it into
//! the parent block's `prev_randao`).
//!
//! ## Row columns (per block)
//!
//!   - `prev_mix[0..32]`  — 32 BE bytes of the prior RANDAO mix (BE).
//!   - `reveal_sig[0..96]` — 96 BE bytes of the BLS RANDAO reveal sig.
//!   - `reveal_hash[0..32]` — `sha256(reveal_sig)` output bytes.
//!   - `new_mix[0..32]`   — 32 BE bytes of the updated RANDAO mix.
//!   - `and_vals[0..32]`  — per-byte witness for `prev_mix[i] AND
//!     reveal_hash[i]`, used to encode XOR as
//!     `new = prev + hash - 2 * and` (byte-range checked).
//!   - `new_mix_limb[0..4]` — 4 LE u64 limbs of `new_mix`, bound to
//!     `new_mix` bytes via decomp constraints. Mirrors the form
//!     `block_header_air::prev_randao` exposes for cross-AIR linkage.
//!   - `block_slot`       — u64 committed slot for the block.
//!   - `slot_byte[0..8]`  — 8 LE bytes of `block_slot`, bound to it
//!     via the LE decomp constraint.
//!   - `is_real`          — `1` on real rows, `0` on padding.
//!   - `is_first`         — `1` on the first row, `0` elsewhere.
//!
//! ## Row-local constraints (39 bodies)
//!
//! 0. `is_real_binary` — `is_real * (is_real - 1) = 0`.
//! 1. `is_first_binary` — `is_first * (is_first - 1) = 0`.
//! 2..34. Per-byte XOR identity:
//!    `new_mix[k] - prev_mix[k] - reveal_hash[k] + 2 * and_vals[k] = 0`.
//! 34..38. New-mix LE u64 limb decomp (4 limbs, BE→LE convention).
//! 38. `slot_le_decomp` — `block_slot - Σ_b slot_byte[b] * 2^(8b) = 0`.
//!
//! ## Shifted constraint (1 body, cross-row)
//!
//! 0. `chain_prev_eq_new` —
//!    `is_real(X) * is_real(ω·X) * Σ_k α^k * (PREV_MIX_k(ω·X) -
//!    NEW_MIX_k(X)) = 0`
//!    multiplied by `(z - ω^{n-1})` to exclude the wrap-around row.
//!    Under fresh α this β-RLC compresses all 32 byte equalities via
//!    Schwartz–Zippel; failing any single byte fires the body w.h.p.
//!
//! ## Cross-AIR LogUp descriptors
//!
//!   - `make_randao_chain_to_sha256_descriptor` — per-row link of the
//!     first 64 bytes of `reveal_sig` + 32 bytes of `reveal_hash` to a
//!     single `sha256_extract` block. Stepping-stone (full 96-byte
//!     chain needs a multi-block sha256 gadget).
//!   - `make_randao_chain_to_block_header_descriptor` — per-row link
//!     of `new_mix_limb[0..4]` to `block_header_air::prev_randao[0..4]`.
//!
//! ## Soundness scope
//!
//! Closed algebraically here:
//!   - Per-row XOR identity (the same shape as `randao_proposer_air`),
//!     byte-range checks on every byte column.
//!   - LE limb decomp of `new_mix` and LE decomp of `block_slot`.
//!   - Cross-row chain: `prev_mix(ω·X) = new_mix(X)` via β-RLC.
//!
//! Deferred (same caveats as `randao_proposer_air`):
//!   - Bit-decomp gadget on `and_vals` for full XOR pinning.
//!   - Multi-block sha256 for the full 96-byte signature hash.
//!   - First-row boundary descriptor pinning `prev_mix[0]` externally.

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

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_PREV_MIX_OFFSET: usize = 0;                                   // 0..32
pub const COL_REVEAL_SIG_OFFSET: usize = COL_PREV_MIX_OFFSET + MIX_LEN;     // 32..128
pub const COL_REVEAL_HASH_OFFSET: usize = COL_REVEAL_SIG_OFFSET + REVEAL_SIG_LEN; // 128..160
pub const COL_NEW_MIX_OFFSET: usize = COL_REVEAL_HASH_OFFSET + HASH_LEN;    // 160..192
pub const COL_AND_VALS_OFFSET: usize = COL_NEW_MIX_OFFSET + MIX_LEN;        // 192..224
pub const COL_NEW_MIX_LIMB_L0: usize = COL_AND_VALS_OFFSET + MIX_LEN;       // 224
pub const COL_NEW_MIX_LIMB_L1: usize = COL_NEW_MIX_LIMB_L0 + 1;             // 225
pub const COL_NEW_MIX_LIMB_L2: usize = COL_NEW_MIX_LIMB_L0 + 2;             // 226
pub const COL_NEW_MIX_LIMB_L3: usize = COL_NEW_MIX_LIMB_L0 + 3;             // 227
pub const COL_BLOCK_SLOT: usize = COL_NEW_MIX_LIMB_L3 + 1;                  // 228
pub const COL_SLOT_BYTE_OFFSET: usize = COL_BLOCK_SLOT + 1;                 // 229..237
pub const COL_IS_REAL: usize = COL_SLOT_BYTE_OFFSET + U64_BYTES;            // 237
pub const COL_IS_FIRST: usize = COL_IS_REAL + 1;                            // 238
pub const NUM_COLUMNS: usize = COL_IS_FIRST + 1;                            // 239

/// Row-local constraints:
///   0: is_real binary
///   1: is_first binary
///   2..34: per-byte XOR identity (32 bodies)
///   34..38: new_mix limb decomp (4 bodies)
///   38: slot LE decomp
pub const NUM_ROW_CONSTRAINTS: usize = 2 + MIX_LEN + 4 + 1;
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct RandaoChainRow {
    pub prev_mix: [u8; MIX_LEN],
    pub reveal_sig: [u8; REVEAL_SIG_LEN],
    pub reveal_hash: [u8; HASH_LEN],
    pub new_mix: [u8; MIX_LEN],
    pub block_slot: u64,
    pub is_first: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RandaoChainWitness {
    pub rows: Vec<RandaoChainRow>,
}

impl RandaoChainWitness {
    /// Build a chain witness from an `initial_mix` and a list of
    /// `(block_slot, reveal_sig)` pairs. Each row computes
    /// `reveal_hash = sha256(reveal_sig)` and chains
    /// `new_mix = prev_mix XOR reveal_hash` with the next row's
    /// `prev_mix` taken from this row's `new_mix`.
    pub fn from_block_chain(
        initial_mix: [u8; MIX_LEN],
        blocks: &[(u64, [u8; REVEAL_SIG_LEN])],
    ) -> Self {
        let mut rows = Vec::with_capacity(blocks.len());
        let mut prev = initial_mix;
        for (i, (slot, sig)) in blocks.iter().enumerate() {
            let reveal_hash = crate::sha256::sha256(sig);
            let mut new_mix = [0u8; MIX_LEN];
            for k in 0..MIX_LEN {
                new_mix[k] = prev[k] ^ reveal_hash[k];
            }
            rows.push(RandaoChainRow {
                prev_mix: prev,
                reveal_sig: *sig,
                reveal_hash,
                new_mix,
                block_slot: *slot,
                is_first: i == 0,
            });
            prev = new_mix;
        }
        Self { rows }
    }

    pub fn from_rows(rows: Vec<RandaoChainRow>) -> Self {
        Self { rows }
    }
}

/// 32-byte BE → 4 LE u64 limbs (matches `block_header_air::prev_randao`).
fn mix_to_limbs(mix: &[u8; MIX_LEN]) -> [u64; 4] {
    [
        u64::from_be_bytes([
            mix[24], mix[25], mix[26], mix[27],
            mix[28], mix[29], mix[30], mix[31],
        ]),
        u64::from_be_bytes([
            mix[16], mix[17], mix[18], mix[19],
            mix[20], mix[21], mix[22], mix[23],
        ]),
        u64::from_be_bytes([
            mix[8],  mix[9],  mix[10], mix[11],
            mix[12], mix[13], mix[14], mix[15],
        ]),
        u64::from_be_bytes([
            mix[0], mix[1], mix[2], mix[3],
            mix[4], mix[5], mix[6], mix[7],
        ]),
    ]
}

/// `(byte_index_within_new_mix, weight)` for the `limb_idx`-th LE u64
/// limb (matches `block_header_air::prev_randao` BE→LE convention).
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
    witness: &RandaoChainWitness,
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
        // and_vals[k] = prev_mix[k] AND reveal_hash[k].
        for k in 0..MIX_LEN {
            let and_byte = row.prev_mix[k] & row.reveal_hash[k];
            columns[COL_AND_VALS_OFFSET + k][i] =
                Scalar::from_u64(and_byte as u64, curve);
        }
        let limbs = mix_to_limbs(&row.new_mix);
        for j in 0..4 {
            columns[COL_NEW_MIX_LIMB_L0 + j][i] = Scalar::from_u64(limbs[j], curve);
        }
        columns[COL_BLOCK_SLOT][i] = Scalar::from_u64(row.block_slot, curve);
        let slot_bytes = row.block_slot.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_SLOT_BYTE_OFFSET + b][i] =
                Scalar::from_u64(slot_bytes[b] as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_FIRST][i] = if row.is_first { one.clone() } else { zero.clone() };
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

pub struct RandaoChainConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl RandaoChainConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for RandaoChainConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = vec![
            "is_real_binary".into(),
            "is_first_binary".into(),
        ];
        for k in 0..MIX_LEN {
            labels.push(format!("xor_byte_{}", k));
        }
        for j in 0..4 {
            labels.push(format!("new_mix_limb_{}_decomp", j));
        }
        labels.push("slot_le_decomp".into());
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

        // 2..34: per-byte XOR identity.
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

        // 34..38: new_mix limb decomp.
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

        // 38: slot LE decomp.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    let byte = &columns[COL_SLOT_BYTE_OFFSET + b][r];
                    sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
                }
                c[r] = columns[COL_BLOCK_SLOT][r].sub(&sum);
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

        // 2..34: per-byte XOR identity.
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

        // 34..38: limb decomp.
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

        // 38: slot LE decomp.
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                let byte = &col_evals[COL_SLOT_BYTE_OFFSET + b];
                sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
            }
            let body = col_evals[COL_BLOCK_SLOT].sub(&sum);
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

        // 2..34: per-byte XOR identity.
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

        // 34..38: limb decomp.
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

        // 38: slot LE decomp.
        {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..U64_BYTES {
                let byte_poly = &col_coeffs[COL_SLOT_BYTE_OFFSET + b];
                let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_BLOCK_SLOT], &sum, curve);
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
                    label: format!("slot_byte_{}_8bit", b),
                    column_index: COL_SLOT_BYTE_OFFSET + b,
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

        // β-RLC over 32 bodies: Σ_k α^k * (PREV_NEXT[k] - NEW[k]).
        // Note: we reuse the α-RLC challenge as β (both are derived from
        // the same Fiat-Shamir transcript; the verifier and prover agree
        // on the same α). Each byte contribution is multiplied into the
        // shifted body's α^alpha_offset slot.
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

        // β-RLC body in polynomial form.
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

        // Multiply by (X - ω^{n-1}) to exclude wrap-around row.
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

/// Link the first 64 bytes of `reveal_sig` and the full 32-byte
/// `reveal_hash` against `sha256_extract`'s `(INPUT_BYTE[0..64],
/// OUTPUT_BYTE[0..32])`. Stepping-stone descriptor — the full 96-byte
/// sha256 chain needs a multi-block sha256 gadget.
pub fn make_randao_chain_to_sha256_descriptor(
    randao_layer_index: usize,
    sha256_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;
    let a_columns: Vec<usize> = (0..se::NUM_INPUT_BYTES)
        .map(|k| COL_REVEAL_SIG_OFFSET + k)
        .chain((0..HASH_LEN).map(|k| COL_REVEAL_HASH_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = (0..se::NUM_INPUT_BYTES)
        .map(|k| se::COL_INPUT_BYTE_OFFSET + k)
        .chain((0..HASH_LEN).map(|k| se::COL_OUTPUT_BYTE_OFFSET + k))
        .collect();
    CrossAirLogUpDescriptor {
        label: "randao_chain_to_sha256_v1".into(),
        a_layer_index: randao_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Link `new_mix_limb[0..4]` to `block_header_air::prev_randao[0..4]`.
/// Binds the per-block chain output of this AIR to the corresponding
/// block-header's `mix_hash` (in LE u64 limb form).
pub fn make_randao_chain_to_block_header_descriptor(
    randao_layer_index: usize,
    block_header_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    CrossAirLogUpDescriptor {
        label: "randao_chain_to_block_header_v1".into(),
        a_layer_index: randao_layer_index,
        a_columns: vec![
            COL_NEW_MIX_LIMB_L0,
            COL_NEW_MIX_LIMB_L1,
            COL_NEW_MIX_LIMB_L2,
            COL_NEW_MIX_LIMB_L3,
        ],
        a_selector_column: Some(COL_IS_REAL),
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

    fn sigs(byte_seed: u8, slot: u64) -> (u64, [u8; REVEAL_SIG_LEN]) {
        let mut sig = [0u8; REVEAL_SIG_LEN];
        for k in 0..REVEAL_SIG_LEN {
            sig[k] = byte_seed.wrapping_add(k as u8);
        }
        (slot, sig)
    }

    fn check_all_row_bodies_vanish(witness: &RandaoChainWitness) {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(witness, curve);
        let cs = RandaoChainConstraintSystem::new(trace.num_rows);
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
    fn single_block_chain_constraints_vanish() {
        let initial_mix = [0u8; MIX_LEN];
        let blocks = vec![sigs(0x33, 100)];
        let w = RandaoChainWitness::from_block_chain(initial_mix, &blocks);
        assert_eq!(w.rows.len(), 1);
        // Honest: new_mix = 0 XOR hash = hash.
        assert_eq!(w.rows[0].new_mix, w.rows[0].reveal_hash);
        assert!(w.rows[0].is_first);
        check_all_row_bodies_vanish(&w);
    }

    #[test]
    fn three_block_chain_constraints_vanish() {
        let initial_mix = [0xa5u8; MIX_LEN];
        let blocks = vec![sigs(0x11, 200), sigs(0x22, 201), sigs(0x33, 202)];
        let w = RandaoChainWitness::from_block_chain(initial_mix, &blocks);
        assert_eq!(w.rows.len(), 3);
        // Chain wiring host-side: row[i+1].prev_mix == row[i].new_mix.
        assert_eq!(w.rows[0].prev_mix, initial_mix);
        assert_eq!(w.rows[1].prev_mix, w.rows[0].new_mix);
        assert_eq!(w.rows[2].prev_mix, w.rows[1].new_mix);
        assert!(w.rows[0].is_first);
        assert!(!w.rows[1].is_first);
        assert!(!w.rows[2].is_first);
        check_all_row_bodies_vanish(&w);

        // Direct shifted-body algebraic check on rows 0→1 and 1→2.
        // Without the (z - ω^{n-1}) boundary factor the per-transition
        // β-RLC body must vanish under honest data.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let beta = Scalar::from_u64(13, curve);
        for transition in [0usize, 1] {
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
    fn tampered_chain_break_fires_shifted() {
        // 2-block chain with row 1's prev_mix tampered (no longer matches
        // row 0's new_mix). The β-RLC shifted body must fire.
        let initial_mix = [0u8; MIX_LEN];
        let blocks = vec![sigs(0x77, 1), sigs(0x88, 2)];
        let w = RandaoChainWitness::from_block_chain(initial_mix, &blocks);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper prev_mix[5] on row 1.
        cols[COL_PREV_MIX_OFFSET + 5][1] = Scalar::from_u64(0xAA, curve);
        // Verify the β-RLC body on transition 0→1 is non-zero.
        let beta = Scalar::from_u64(17, curve);
        let is_real = &cols[COL_IS_REAL][0];
        let is_real_next = &cols[COL_IS_REAL][1];
        let gating = is_real.mul(is_real_next);
        let mut rlc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MIX_LEN {
            let prev_next_k = &cols[COL_PREV_MIX_OFFSET + k][1];
            let new_k = &cols[COL_NEW_MIX_OFFSET + k][0];
            rlc = rlc.add(&bp.mul(&prev_next_k.sub(new_k)));
            bp = bp.mul(&beta);
        }
        let body = gating.mul(&rlc);
        assert!(
            !body.is_zero(),
            "tampered chain must fire β-RLC shifted body",
        );

        // Sanity: row-local XOR bodies on the tampered row also fire,
        // because new_mix on row 1 was computed against the original
        // prev_mix but prev_mix is now different.
        let cs = RandaoChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // XOR byte 5 is constraint index 2 + 5 = 7.
        assert!(
            !bodies[2 + 5][1].is_zero(),
            "XOR byte_5 constraint should fire on tampered prev_mix[5]",
        );
    }

    #[test]
    fn first_row_prev_mix_set_externally() {
        // The chain's first row's `prev_mix` is taken from `initial_mix`
        // and is a free committed column — no constraint pins it
        // internally. Verify two different `initial_mix` values produce
        // distinct row-0 prev_mix columns while still yielding a valid
        // honest witness (constraints vanish).
        let mix_a = [0u8; MIX_LEN];
        let mut mix_b = [0u8; MIX_LEN];
        mix_b[31] = 0x01;
        let (slot, sig) = sigs(0x42, 7);
        let blocks = vec![(slot, sig)];
        let wa = RandaoChainWitness::from_block_chain(mix_a, &blocks);
        let wb = RandaoChainWitness::from_block_chain(mix_b, &blocks);
        // Different initial mixes ⇒ different first-row prev_mix and
        // (consequently) different new_mix.
        assert_ne!(wa.rows[0].prev_mix, wb.rows[0].prev_mix);
        assert_ne!(wa.rows[0].new_mix, wb.rows[0].new_mix);
        // Both honest witnesses satisfy all row-local constraints.
        check_all_row_bodies_vanish(&wa);
        check_all_row_bodies_vanish(&wb);
        // is_first is set on the row regardless of initial mix.
        assert!(wa.rows[0].is_first);
        assert!(wb.rows[0].is_first);
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_randao_chain_to_sha256_descriptor(0, 1);
        assert_eq!(d1.label, "randao_chain_to_sha256_v1");
        // 64 reveal_sig bytes + 32 reveal_hash bytes = 96.
        assert_eq!(d1.a_columns.len(), 96);
        assert_eq!(d1.b_columns.len(), 96);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        for k in 0..64 {
            assert_eq!(d1.a_columns[k], COL_REVEAL_SIG_OFFSET + k);
        }
        for k in 0..HASH_LEN {
            assert_eq!(d1.a_columns[64 + k], COL_REVEAL_HASH_OFFSET + k);
        }

        let d2 = make_randao_chain_to_block_header_descriptor(0, 1);
        assert_eq!(d2.label, "randao_chain_to_block_header_v1");
        assert_eq!(d2.a_columns.len(), 4);
        assert_eq!(d2.b_columns.len(), 4);
        assert_eq!(d2.a_columns[0], COL_NEW_MIX_LIMB_L0);
        assert_eq!(d2.a_columns[3], COL_NEW_MIX_LIMB_L3);
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_PREV_MIX_OFFSET, 0);
        assert_eq!(COL_REVEAL_SIG_OFFSET, 32);
        assert_eq!(COL_REVEAL_HASH_OFFSET, 128);
        assert_eq!(COL_NEW_MIX_OFFSET, 160);
        assert_eq!(COL_AND_VALS_OFFSET, 192);
        assert_eq!(COL_NEW_MIX_LIMB_L0, 224);
        assert_eq!(COL_BLOCK_SLOT, 228);
        assert_eq!(COL_SLOT_BYTE_OFFSET, 229);
        assert_eq!(COL_IS_REAL, 237);
        assert_eq!(COL_IS_FIRST, 238);
        assert_eq!(NUM_COLUMNS, 239);
        assert_eq!(NUM_ROW_CONSTRAINTS, 2 + 32 + 4 + 1);
        assert_eq!(NUM_SHIFTED, 1);
    }

    #[test]
    fn slot_le_decomp_fires_on_tampered_byte() {
        let initial_mix = [0u8; MIX_LEN];
        let blocks = vec![sigs(0x44, 0x1234_5678_9abc_def0)];
        let w = RandaoChainWitness::from_block_chain(initial_mix, &blocks);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper one slot byte without updating the u64 slot column.
        cols[COL_SLOT_BYTE_OFFSET + 3][0] = Scalar::from_u64(0, curve);
        let cs = RandaoChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // slot_le_decomp is the last constraint (index NUM_ROW_CONSTRAINTS-1).
        assert!(
            !bodies[NUM_ROW_CONSTRAINTS - 1][0].is_zero(),
            "slot_le_decomp must fire on tampered slot byte",
        );
    }

    #[test]
    fn shifted_column_indices_match_layout() {
        let cs = RandaoChainConstraintSystem::new(2);
        let cols = cs.shifted_column_indices();
        // 1 IS_REAL + 32 PREV_MIX bytes.
        assert_eq!(cols.len(), 1 + MIX_LEN);
        assert_eq!(cols[0], COL_IS_REAL);
        for k in 0..MIX_LEN {
            assert_eq!(cols[1 + k], COL_PREV_MIX_OFFSET + k);
        }
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
    }
}
