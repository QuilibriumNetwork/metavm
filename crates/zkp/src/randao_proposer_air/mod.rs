//! RANDAO mix + proposer index verification AIR.
//!
//! Per-beacon-block view of the RANDAO state transition:
//!
//!   `new_randao_mix = old_randao_mix XOR sha256(randao_reveal_bls_sig)`
//!
//! plus a committed witness for the slot's proposer index. The
//! proposer-shuffle algebraic gadget is deferred to a future module
//! (it requires the full shuffle / seed-derivation chain); here we
//! simply commit `proposer_index` and `slot` as witness columns so
//! downstream chain proofs can carry them.
//!
//! Each row represents one beacon block / one RANDAO update:
//!
//!   - `old_mix[0..32]` — 32 BE bytes of the previous RANDAO mix.
//!   - `reveal_sig[0..96]` — 96 BE bytes of the BLS signature
//!     (`RandaoReveal`). Committed as a witness column so the
//!     downstream BLS verification AIR can pin it; the sha256 → hash
//!     link below binds the first 64 bytes against a single
//!     `sha256_extract` block (the multi-block sha256 over the full
//!     96-byte signature is deferred to a wider sha256 gadget).
//!   - `reveal_hash[0..32]` — `sha256(reveal_sig)` output bytes (the
//!     "hash to XOR in").
//!   - `new_mix[0..32]` — 32 BE bytes of the updated RANDAO mix.
//!   - `and_vals[0..32]` — per-byte witness for `old_mix[i] AND
//!     reveal_hash[i]`, used to encode the XOR constraint as
//!     `new = old + hash - 2 * and` (each byte is in `[0, 255]` via
//!     the byte-range LogUp lookup).
//!   - `new_mix_limb[0..4]` — 4 LE u64 limbs of `new_mix`, bound to
//!     `new_mix` via byte-decomp constraints; this is the form the
//!     downstream `block_header_air` exposes for `prev_randao`, so
//!     the cross-AIR LogUp linkage can match on shared limbs.
//!   - `proposer_index` — committed u64 (no algebraic shuffle yet).
//!   - `slot` — committed u64 (no algebraic slot↔epoch yet).
//!   - `is_real` — `1` on real rows, `0` on padding.
//!
//! Row-local constraints (1 + 32 + 4 = 37):
//!   - 0: `is_real` binary.
//!   - 1..33: per-byte XOR `new_mix[i] - old_mix[i] - reveal_hash[i]
//!     + 2 * and_vals[i] = 0`. With every byte byte-range-checked,
//!     the only valid algebraic solution is `and_vals[i] = old_mix[i]
//!     AND reveal_hash[i]` and `new_mix[i] = old_mix[i] XOR
//!     reveal_hash[i]` (proof sketch: lift each byte to its 8 bits;
//!     under bit-level constraints the formula `a + b - 2(a AND b)`
//!     uniquely defines XOR; the byte-range check forces each value
//!     into `[0, 255]`, and the linear equation pins `and_vals` once
//!     `new_mix` is the XOR — for soundness against a fully-tampered
//!     `and_vals` column the per-byte decomposition gadget is the
//!     follow-up: see "Soundness scope" below).
//!   - 33..37: 4 limb-decomp equations binding `new_mix_limb` to
//!     `new_mix[0..32]` in the same BE→LE convention as
//!     `block_header_air::prev_randao`.
//!
//! Lookup declarations: every byte column (`old_mix`, `reveal_sig`,
//! `reveal_hash`, `new_mix`, `and_vals`) gets an 8-bit range check.
//!
//! # Soundness scope (current step)
//!
//! Closed algebraically:
//!   - Byte-range on `old_mix`, `reveal_sig`, `reveal_hash`,
//!     `new_mix`, `and_vals`.
//!   - Linear identity `new = old + hash - 2 * and` per byte.
//!   - Limb form of `new_mix` matches its byte form.
//!
//! Not yet closed (deferred):
//!   - Per-bit binary decomposition of `and_vals`, which would
//!     uniquely pin `and_vals[i] = old_mix[i] AND reveal_hash[i]`
//!     algebraically. Currently the byte-range check + linear
//!     identity admit "any pair `(new, and)` such that `new + 2 *
//!     and = old + hash`": the honest prover always supplies the
//!     correct AND, but a malicious prover could potentially supply
//!     a non-AND pair if the resulting `new` and `and` both stay in
//!     `[0, 255]`. The bit-decomp gadget closes this in a follow-up.
//!   - `reveal_hash = sha256(reveal_sig)` — the descriptor
//!     `make_randao_to_sha256_descriptor` binds the FIRST 64 bytes
//!     of `reveal_sig` against `sha256_extract`'s 64-byte input. The
//!     full 96-byte sha256 chain needs a wider sha256 gadget
//!     (multi-block padding + length encoding). The current
//!     descriptor is a stepping-stone — honest provers must still
//!     compute the full sha256, but the descriptor only catches
//!     tampering in the first 64 bytes.
//!   - Proposer-shuffle derivation. `proposer_index` is committed
//!     but unconstrained beyond being a value in the trace; future
//!     proposer-shuffle module will add a cross-AIR LogUp linking
//!     `(slot, seed, proposer_index)` to the shuffle output.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const MIX_LEN: usize = 32;
pub const HASH_LEN: usize = 32;
pub const REVEAL_SIG_LEN: usize = 96;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_OLD_MIX_OFFSET: usize = 0;                            // 0..32
pub const COL_REVEAL_SIG_OFFSET: usize = COL_OLD_MIX_OFFSET + MIX_LEN; // 32..128
pub const COL_REVEAL_HASH_OFFSET: usize = COL_REVEAL_SIG_OFFSET + REVEAL_SIG_LEN; // 128..160
pub const COL_NEW_MIX_OFFSET: usize = COL_REVEAL_HASH_OFFSET + HASH_LEN; // 160..192
pub const COL_AND_VALS_OFFSET: usize = COL_NEW_MIX_OFFSET + MIX_LEN; // 192..224
pub const COL_NEW_MIX_LIMB_L0: usize = COL_AND_VALS_OFFSET + MIX_LEN; // 224
pub const COL_NEW_MIX_LIMB_L1: usize = COL_NEW_MIX_LIMB_L0 + 1;      // 225
pub const COL_NEW_MIX_LIMB_L2: usize = COL_NEW_MIX_LIMB_L0 + 2;      // 226
pub const COL_NEW_MIX_LIMB_L3: usize = COL_NEW_MIX_LIMB_L0 + 3;      // 227
pub const COL_PROPOSER_INDEX: usize = COL_NEW_MIX_LIMB_L3 + 1;       // 228
pub const COL_SLOT: usize = COL_PROPOSER_INDEX + 1;                  // 229
pub const COL_IS_REAL: usize = COL_SLOT + 1;                         // 230
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;                      // 231

/// Row-local constraints:
///   0: is_real binary
///   1..33: per-byte XOR identity (`new = old + hash - 2 * and`)
///   33..37: 4 new_mix limb-decomp equations
pub const NUM_ROW_CONSTRAINTS: usize = 1 + MIX_LEN + 4;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct RandaoProposerRow {
    pub old_mix: [u8; MIX_LEN],
    pub reveal_sig: [u8; REVEAL_SIG_LEN],
    pub reveal_hash: [u8; HASH_LEN],
    pub new_mix: [u8; MIX_LEN],
    pub proposer_index: u64,
    pub slot: u64,
}

#[derive(Clone, Debug, Default)]
pub struct RandaoProposerWitness {
    pub rows: Vec<RandaoProposerRow>,
}

impl RandaoProposerWitness {
    /// Build a witness row from the canonical inputs. Computes
    /// `reveal_hash = sha256(reveal_sig)` and `new_mix = old_mix XOR
    /// reveal_hash` host-side; algebraic constraints + cross-AIR
    /// LogUp descriptors are what bind those derivations on-chain.
    pub fn from_inputs(
        old_mix: [u8; MIX_LEN],
        reveal_sig: [u8; REVEAL_SIG_LEN],
        proposer: u64,
        slot: u64,
    ) -> Self {
        let reveal_hash = crate::sha256::sha256(&reveal_sig);
        let mut new_mix = [0u8; MIX_LEN];
        for i in 0..MIX_LEN {
            new_mix[i] = old_mix[i] ^ reveal_hash[i];
        }
        Self {
            rows: vec![RandaoProposerRow {
                old_mix,
                reveal_sig,
                reveal_hash,
                new_mix,
                proposer_index: proposer,
                slot,
            }],
        }
    }

    pub fn from_rows(rows: Vec<RandaoProposerRow>) -> Self {
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

/// `(byte_index_within_new_mix, weight)` for the `limb_idx`-th LE
/// u64 limb. Mirrors the BE→LE convention used by
/// `block_header_air::prev_randao`.
fn new_mix_limb_decomp_targets(limb_idx: usize) -> Vec<(usize, u64)> {
    match limb_idx {
        // limb 0 (low 64 bits) = BE bytes 24..32, weights 2^56..2^0.
        0 => (0..8).map(|k| (24 + k, 1u64 << (8 * (7 - k)))).collect(),
        1 => (0..8).map(|k| (16 + k, 1u64 << (8 * (7 - k)))).collect(),
        2 => (0..8).map(|k| (8 + k, 1u64 << (8 * (7 - k)))).collect(),
        3 => (0..8).map(|k| (k, 1u64 << (8 * (7 - k)))).collect(),
        _ => unreachable!(),
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &RandaoProposerWitness,
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
            columns[COL_OLD_MIX_OFFSET + k][i] =
                Scalar::from_u64(row.old_mix[k] as u64, curve);
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
        // and_vals[i] = old_mix[i] AND reveal_hash[i].
        for k in 0..MIX_LEN {
            let and_byte = row.old_mix[k] & row.reveal_hash[k];
            columns[COL_AND_VALS_OFFSET + k][i] =
                Scalar::from_u64(and_byte as u64, curve);
        }
        let limbs = mix_to_limbs(&row.new_mix);
        for j in 0..4 {
            columns[COL_NEW_MIX_LIMB_L0 + j][i] = Scalar::from_u64(limbs[j], curve);
        }
        columns[COL_PROPOSER_INDEX][i] = Scalar::from_u64(row.proposer_index, curve);
        columns[COL_SLOT][i] = Scalar::from_u64(row.slot, curve);
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

pub struct RandaoProposerConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl RandaoProposerConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for RandaoProposerConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for k in 0..MIX_LEN {
            labels.push(format!("xor_byte_{}", k));
        }
        for j in 0..4 {
            labels.push(format!("new_mix_limb_{}_decomp", j));
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

        // 1..33: per-byte XOR identity.
        for k in 0..MIX_LEN {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let old_b = &columns[COL_OLD_MIX_OFFSET + k][r];
                let hash_b = &columns[COL_REVEAL_HASH_OFFSET + k][r];
                let new_b = &columns[COL_NEW_MIX_OFFSET + k][r];
                let and_b = &columns[COL_AND_VALS_OFFSET + k][r];
                // new - old - hash + 2 * and
                let two_and = two.mul(and_b);
                let body = new_b.sub(old_b).sub(hash_b).add(&two_and);
                c[r] = body;
            }
            out.push(c);
        }

        // 33..37: 4 new_mix limb decomp equations.
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

        out
    }

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

        // 1..33: per-byte XOR identity.
        for k in 0..MIX_LEN {
            let old_b = &col_evals[COL_OLD_MIX_OFFSET + k];
            let hash_b = &col_evals[COL_REVEAL_HASH_OFFSET + k];
            let new_b = &col_evals[COL_NEW_MIX_OFFSET + k];
            let and_b = &col_evals[COL_AND_VALS_OFFSET + k];
            let two_and = two.mul(and_b);
            let body = new_b.sub(old_b).sub(hash_b).add(&two_and);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 33..37: 4 new_mix limb decomp.
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

        // 1..33: per-byte XOR identity (polynomial form is linear, so
        // we treat it as: new - old - hash + 2 * and).
        for k in 0..MIX_LEN {
            let old_b = &col_coeffs[COL_OLD_MIX_OFFSET + k];
            let hash_b = &col_coeffs[COL_REVEAL_HASH_OFFSET + k];
            let new_b = &col_coeffs[COL_NEW_MIX_OFFSET + k];
            let and_b = &col_coeffs[COL_AND_VALS_OFFSET + k];
            let two_and = poly_scalar_mul(and_b, &two_scalar);
            let body = poly_add(
                &poly_sub(&poly_sub(new_b, old_b, curve), hash_b, curve),
                &two_and,
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 33..37: 4 new_mix limb decomp.
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
                    label: format!("old_mix_{}_8bit", k),
                    column_index: COL_OLD_MIX_OFFSET + k,
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
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Link the first 64 bytes of `reveal_sig` and the full 32-byte
/// `reveal_hash` against `sha256_extract`'s `(INPUT_BYTE[0..64],
/// OUTPUT_BYTE[0..32])`. This is a stepping-stone descriptor — the
/// full 96-byte signature hash needs a multi-block sha256 gadget
/// (padding + length encoding), which is deferred.
pub fn make_randao_to_sha256_descriptor(
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
        label: "randao_to_sha256_v1".into(),
        a_layer_index: randao_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Link `new_mix_limb[0..4]` to `block_header_air::prev_randao[0..4]`.
/// Binds the RANDAO update output of this AIR to the `mix_hash`
/// field that the block-header AIR carries in its 4 LE u64 limb
/// form. The byte-form binding (via block_header's RLP) is bridged
/// elsewhere through the keccak chain.
pub fn make_randao_to_block_header_descriptor(
    randao_layer_index: usize,
    block_header_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    CrossAirLogUpDescriptor {
        label: "randao_to_block_header_v1".into(),
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

    #[test]
    fn zero_old_mix_xor_hash_equals_hash() {
        let old_mix = [0u8; MIX_LEN];
        let reveal_sig = [0x33u8; REVEAL_SIG_LEN];
        let w = RandaoProposerWitness::from_inputs(old_mix, reveal_sig, 7, 42);
        let row = &w.rows[0];
        assert_eq!(row.new_mix, row.reveal_hash,
            "0 XOR hash should equal hash");
        // Honest constraints evaluate to zero.
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = RandaoProposerConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(val.is_zero(),
                    "constraint {} row {} = {:?}", i, r, val);
            }
        }
    }

    #[test]
    fn deterministic_witness_from_same_inputs() {
        let old_mix = [0xab; MIX_LEN];
        let reveal_sig = [0xcd; REVEAL_SIG_LEN];
        let w1 = RandaoProposerWitness::from_inputs(old_mix, reveal_sig, 9, 100);
        let w2 = RandaoProposerWitness::from_inputs(old_mix, reveal_sig, 9, 100);
        assert_eq!(w1.rows.len(), w2.rows.len());
        assert_eq!(w1.rows[0].new_mix, w2.rows[0].new_mix);
        assert_eq!(w1.rows[0].reveal_hash, w2.rows[0].reveal_hash);
        assert_eq!(w1.rows[0].proposer_index, w2.rows[0].proposer_index);
        assert_eq!(w1.rows[0].slot, w2.rows[0].slot);

        // Sanity: hash actually equals sha256(reveal_sig).
        let expected_hash = crate::sha256::sha256(&reveal_sig);
        assert_eq!(w1.rows[0].reveal_hash, expected_hash);
        // new_mix matches host XOR.
        for k in 0..MIX_LEN {
            assert_eq!(w1.rows[0].new_mix[k], old_mix[k] ^ expected_hash[k]);
        }
    }

    #[test]
    fn xor_constraint_fires_on_tampered_new_mix() {
        let old_mix = [0x11; MIX_LEN];
        let reveal_sig = [0x22; REVEAL_SIG_LEN];
        let w = RandaoProposerWitness::from_inputs(old_mix, reveal_sig, 3, 50);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper new_mix byte 0.
        cols[COL_NEW_MIX_OFFSET][0] =
            Scalar::from_u64((cols[COL_NEW_MIX_OFFSET][0].to_u64() ^ 0x80) as u64,
                CurveType::Bls48581);
        let cs = RandaoProposerConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 1 corresponds to XOR byte 0.
        assert!(!results[1][0].is_zero(),
            "xor_byte_0 constraint should fire on tampered new_mix");
    }

    #[test]
    fn xor_constraint_fires_on_tampered_and_vals() {
        let old_mix = [0xff; MIX_LEN];
        let reveal_sig = [0xaa; REVEAL_SIG_LEN];
        let w = RandaoProposerWitness::from_inputs(old_mix, reveal_sig, 1, 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper and_vals byte 5 (without matching new_mix change).
        cols[COL_AND_VALS_OFFSET + 5][0] = Scalar::from_u64(0, CurveType::Bls48581);
        let cs = RandaoProposerConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[1 + 5][0].is_zero(),
            "xor_byte_5 constraint should fire on tampered and_vals");
    }

    #[test]
    fn limb_decomp_fires_on_tampered_limb() {
        let old_mix = [0u8; MIX_LEN];
        let reveal_sig = [0u8; REVEAL_SIG_LEN];
        let w = RandaoProposerWitness::from_inputs(old_mix, reveal_sig, 0, 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper limb 0.
        cols[COL_NEW_MIX_LIMB_L0][0] = Scalar::from_u64(0xdeadbeef, CurveType::Bls48581);
        let cs = RandaoProposerConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Limb-decomp constraints start at 1 + MIX_LEN = 33.
        assert!(!results[1 + MIX_LEN][0].is_zero(),
            "new_mix_limb_0_decomp should fire");
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_randao_to_sha256_descriptor(0, 1);
        assert_eq!(d1.label, "randao_to_sha256_v1");
        // 64 reveal_sig bytes + 32 reveal_hash bytes = 96.
        assert_eq!(d1.a_columns.len(), 96);
        assert_eq!(d1.b_columns.len(), 96);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        // First 64 entries reference reveal_sig bytes.
        for k in 0..64 {
            assert_eq!(d1.a_columns[k], COL_REVEAL_SIG_OFFSET + k);
        }
        // Last 32 entries reference reveal_hash.
        for k in 0..HASH_LEN {
            assert_eq!(d1.a_columns[64 + k], COL_REVEAL_HASH_OFFSET + k);
        }

        let d2 = make_randao_to_block_header_descriptor(0, 1);
        assert_eq!(d2.label, "randao_to_block_header_v1");
        assert_eq!(d2.a_columns.len(), 4);
        assert_eq!(d2.b_columns.len(), 4);
        assert_eq!(d2.a_columns[0], COL_NEW_MIX_LIMB_L0);
        assert_eq!(d2.a_columns[3], COL_NEW_MIX_LIMB_L3);
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_OLD_MIX_OFFSET, 0);
        assert_eq!(COL_REVEAL_SIG_OFFSET, 32);
        assert_eq!(COL_REVEAL_HASH_OFFSET, 128);
        assert_eq!(COL_NEW_MIX_OFFSET, 160);
        assert_eq!(COL_AND_VALS_OFFSET, 192);
        assert_eq!(COL_NEW_MIX_LIMB_L0, 224);
        assert_eq!(COL_PROPOSER_INDEX, 228);
        assert_eq!(COL_SLOT, 229);
        assert_eq!(COL_IS_REAL, 230);
        assert_eq!(NUM_COLUMNS, 231);
        assert_eq!(NUM_ROW_CONSTRAINTS, 1 + 32 + 4);
    }

    #[test]
    fn proposer_and_slot_committed() {
        let w = RandaoProposerWitness::from_inputs(
            [0u8; MIX_LEN],
            [0u8; REVEAL_SIG_LEN],
            12345,
            67890,
        );
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(
            trace.columns[COL_PROPOSER_INDEX].evaluations[0].to_u64(),
            12345,
        );
        assert_eq!(trace.columns[COL_SLOT].evaluations[0].to_u64(), 67890);
        // is_real = 1 on real row, 0 on padding (if padded).
        assert_eq!(
            trace.columns[COL_IS_REAL].evaluations[0].to_u64(),
            1,
        );
    }

    #[test]
    fn limb_decomp_matches_block_header_convention() {
        // Build a witness with a deterministic new_mix that we can
        // compute the LE-limb form of by hand, and assert the columns
        // line up.
        let old_mix = [0u8; MIX_LEN];
        let mut reveal_sig = [0u8; REVEAL_SIG_LEN];
        // Pick reveal_sig such that sha256(reveal_sig) XOR 0 = sha256(reveal_sig).
        reveal_sig[0] = 0x42;
        let w = RandaoProposerWitness::from_inputs(old_mix, reveal_sig, 0, 0);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let new_mix = w.rows[0].new_mix;
        let expected = mix_to_limbs(&new_mix);
        for j in 0..4 {
            assert_eq!(
                trace.columns[COL_NEW_MIX_LIMB_L0 + j].evaluations[0].to_u64(),
                expected[j],
            );
        }
    }
}
