//! SHA-256 invocation extraction AIR.
//!
//! Per-invocation byte-level view of a `sha256_pair(left, right) →
//! parent` invocation. Each row corresponds to one SHA-256 invocation
//! and exposes:
//!
//!   - `INPUT_BYTE[0..64]` — the 64 input bytes (`left || right`)
//!   - `OUTPUT_BYTE[0..32]` — the 32 output bytes (`parent`)
//!   - `IS_REAL` — selector, `1` on real invocation rows, `0` on padding
//!
//! This lightweight AIR mirrors the [`crate::validator_extract`]
//! pattern: a focused, byte-level view that downstream cross-AIR LogUp
//! linkages match against. The heavy bit-level SHA-256 AIR stays
//! unchanged.
//!
//! # Soundness scope
//!
//! This AIR proves: "there exists a sequence of `(input, output)`
//! invocation tuples with the witnessed bytes; `IS_REAL` is binary."
//!
//! It does NOT prove: that each row's `output` actually equals
//! `sha256(input)`. Closing that gap requires a cross-AIR LogUp
//! linkage between this AIR and the bit-level
//! `crate::sha256_constraints::Sha256ConstraintSystem` AIR proving
//! the byte-level rows correspond to actual SHA-256 invocations.
//! That's the same shape of follow-up as the
//! `ValidatorExtract ↔ SHA-256` binding documented for #84.
//!
//! # Constraint layout
//!
//! 1 row-local constraint:
//!
//!   0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//!
//! No shifted constraints. Byte values are unconstrained at the
//! per-AIR level beyond `IS_REAL`'s binary check; the cross-AIR LogUp
//! linkages are what tie these to actual SSZ chunks and SHA-256
//! computations.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column indices ────────────────────────────────────────────────────

/// 64 input bytes (`left || right`).
pub const COL_INPUT_BYTE_OFFSET: usize = 0;
pub const NUM_INPUT_BYTES: usize = 64;

/// 32 output bytes (`parent`).
pub const COL_OUTPUT_BYTE_OFFSET: usize = COL_INPUT_BYTE_OFFSET + NUM_INPUT_BYTES;
pub const NUM_OUTPUT_BYTES: usize = 32;

/// `1` on real invocation rows, `0` on padding.
pub const COL_IS_REAL: usize = COL_OUTPUT_BYTE_OFFSET + NUM_OUTPUT_BYTES;

/// Task #309 / #190 mirror column: per-invocation byte anchor for
/// cross-AIR LogUp descriptors that want to bind an arbitrary host-
/// derived byte (e.g. a deposit-root byte) to this AIR's row without
/// requiring B-side overrides on `OUTPUT_BYTE[0]`. Populated by the
/// witness builder; defaults to `output[0]` to preserve back-compat
/// when callers don't set it. No row-local constraint binds this
/// column — soundness flows from the cross-AIR LogUp closure.
pub const COL_MIRROR_BYTE0: usize = COL_IS_REAL + 1;

pub const NUM_COLUMNS: usize = COL_MIRROR_BYTE0 + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 1;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// One SHA-256 pair-hashing invocation as a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sha256ExtractRow {
    pub input: [u8; 64],
    pub output: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sha256ExtractWitness {
    pub invocations: Vec<Sha256ExtractRow>,
    /// Task #309 / #190 mirror data: per-invocation byte value populated
    /// into [`COL_MIRROR_BYTE0`]. Empty `Vec` means "default to
    /// `invocations[i].output[0]`" (back-compat). Otherwise must match
    /// `invocations.len()` and is written verbatim per-row.
    #[doc(hidden)]
    pub mirror_byte0: Vec<u8>,
}

impl Sha256ExtractWitness {
    /// Convenience: build from a slice of `(left, right)` chunk pairs,
    /// computing `parent = sha256_pair(left, right)` per pair.
    pub fn from_pair_inputs(pairs: &[(crate::ssz::Chunk, crate::ssz::Chunk)]) -> Self {
        let invocations = pairs
            .iter()
            .map(|(left, right)| {
                let mut input = [0u8; 64];
                input[..32].copy_from_slice(left);
                input[32..].copy_from_slice(right);
                let output = crate::sha256::sha256_pair(left, right);
                Sha256ExtractRow { input, output }
            })
            .collect();
        Self {
            invocations,
            mirror_byte0: Vec::new(),
        }
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

// ─── Constraint system ─────────────────────────────────────────────────

pub struct Sha256ExtractConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Sha256ExtractConstraintSystem {
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

// ─── Trace builder ─────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Sha256ExtractWitness,
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
        for (b, &v) in inv.output.iter().enumerate() {
            columns[COL_OUTPUT_BYTE_OFFSET + b][i] = Scalar::from_u64(v as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
        // Task #309 mirror: prefer explicit per-row override; fall back
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

// ─── VmConstraintSystem implementation ─────────────────────────────────

impl VmConstraintSystem for Sha256ExtractConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec!["is_real_binary".into()]
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
        for row in 0..n {
            let v = &columns[COL_IS_REAL][row];
            bin[row] = v.mul(&v.sub(&one));
        }
        vec![bin]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        v.mul(&v.sub(&one))
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
        poly_mul(v, &v_minus_1, curve)
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

/// Construct the descriptor binding SSZ pair-hashing rows to this
/// extract AIR. Each SSZ row at any layer holds `(LEFT, RIGHT, PARENT)`
/// chunk byte columns and per-pair selectors. The cross-AIR LogUp tuple
/// is the 96-byte concatenation `(LEFT || RIGHT || PARENT)` matched
/// against this AIR's `(INPUT[0..64], OUTPUT[0..32])`.
///
/// **Selector**: SSZ side gated by `IS_LEFT_REAL` (the left side must
/// be real for the pair-hashing to be meaningful); this AIR side gated
/// by `IS_REAL`.
///
/// **Sub-multiset semantics.** The cross-AIR LogUp asserts every
/// SSZ-side `(LEFT, RIGHT, PARENT)` triple appears in this AIR's
/// `(INPUT, OUTPUT)` multiset (where `INPUT = LEFT || RIGHT` and
/// `OUTPUT = PARENT`). Since this AIR's witness builder
/// `from_pair_inputs` computes `OUTPUT = sha256_pair(LEFT, RIGHT)`,
/// honest provers populate it correctly. A malicious prover would
/// need to commit a row where `OUTPUT ≠ sha256_pair(INPUT)` and have
/// it match an SSZ pair — but SSZ's chain (which proves
/// `PARENT = sha256(LEFT, RIGHT)` algebraically across the bit-level
/// SHA-256 AIR) is what actually anchors `PARENT`. So this linkage
/// transitively binds SSZ's pair-hashing claim to the bit-level
/// computation **once the Sha256Extract↔SHA-256 binding lands**.
///
/// **Soundness gap (deferred)**: this descriptor alone proves "every
/// SSZ pair's (LEFT, RIGHT, PARENT) appears as some Sha256Extract
/// row's (INPUT, OUTPUT)." Without binding Sha256Extract.OUTPUT to
/// the actual SHA-256 computation, a prover could commit any
/// (INPUT, OUTPUT) pairs in this AIR and have them coincidentally
/// match SSZ's pairs. The host-side witness builder
/// `from_pair_inputs` ensures correctness for honest provers; the
/// cryptographic version requires the per-bit SHA-256 binding
/// (cross-AIR LogUp from Sha256Extract to the bit-level SHA-256 AIR
/// computing `output_bits = compress(input_bits, IV)`).
pub fn make_ssz_sha256_extract_linkage_descriptor(
    ssz_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::ssz_air::col as ssz_col;

    // SSZ A side: 96 bytes — LEFT (32) || RIGHT (32) || PARENT (32).
    let mut a_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..32 {
        a_columns.push(ssz_col::LEFT_OFFSET + b);
    }
    for b in 0..32 {
        a_columns.push(ssz_col::RIGHT_OFFSET + b);
    }
    for b in 0..32 {
        a_columns.push(ssz_col::PARENT_OFFSET + b);
    }

    // Sha256Extract B side: 64 input bytes + 32 output bytes.
    let mut b_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..NUM_INPUT_BYTES {
        b_columns.push(COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..NUM_OUTPUT_BYTES {
        b_columns.push(COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "ssz_sha256_extract_v1".into(),
        a_layer_index: ssz_layer_index,
        a_columns,
        a_selector_column: Some(ssz_col::IS_LEFT_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding this AIR to the bit-level
/// SHA-256 AIR ([`crate::sha256_constraints`]).
///
/// **A side (this AIR)**: per-row `(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])`
/// gated by `IS_REAL`.
///
/// **B side (bit-level SHA-256)**: per-invocation aggregator columns
/// `(INV_INPUT_BYTE[0..64], INV_OUTPUT_BYTE[0..32])` gated by
/// `IS_FIRST_INV_ROW`. The bit-level AIR represents one full
/// `sha256(input)` invocation; on the anchor row (round 0 of block 0),
/// the aggregator columns hold the 64 input bytes (for `sha256_pair`,
/// `left || right`) and the 32-byte digest. Invariance constraints
/// (deferred — see soundness scope below) pin those byte values across
/// all rows of the trace.
///
/// **Soundness scope (current)**: this descriptor wires the multiset
/// equivalence between Sha256Extract's per-invocation tuples and the
/// bit-level AIR's per-invocation aggregator. It does NOT yet prove
/// that the aggregator bytes are algebraically derived from the
/// bit-level state — specifically:
///
///   1. **INV_INPUT_BYTE ↔ W bits at rounds 0..15** — the first 16 W
///      words of the first block decompose to the 64 input bytes.
///      Constraint shape: at round r ∈ 0..16 (gated by `SEL_ROUND[r]`),
///      `Σ_b INV_INPUT_BYTE[4r + b] · 2^(8(3 - b)) = Σ_i W_bit[i] · 2^i`.
///      Plus a "first-block" indicator to gate this only on block 0.
///   2. **INV_OUTPUT_BYTE ↔ post-compression state at round 63** — the
///      32-byte digest is the big-endian serialization of `state_in[0..8]
///      + AFTER[round 63][0..8]` (componentwise mod 2^32). Requires
///      folding the IV (or per-block `state_in`) into a column and
///      summing with carry.
///   3. **INV_*_BYTE invariance across rows** — `INV_*(ω·X) − INV_*(X) = 0`
///      with last-row exclusion (and invocation-boundary exclusion in a
///      multi-invocation extension).
///   4. **IS_FIRST_INV_ROW = 1 exactly once per invocation** — implicit
///      via the LogUp multiplicity check on Sha256Extract's per-row
///      tuples (multiplicity 1 each), but enforcing `IS_FIRST_INV_ROW
///      ∈ {0, 1}` algebraically prevents trivial multiplicity-0 attacks.
///
/// Today the linkage proves "the bit-level SHA-256 AIR committed *some*
/// (input, output) tuple per invocation that matches the Sha256Extract
/// row" — equivalent to the witness builder's correctness guarantee. The
/// algebraic strength comes when items 1–4 land. Tracked in
/// `cross_air_logup_dependent_tasks.md` as the "shared SHA-256/Keccak
/// byte-aggregation" follow-up.
pub fn make_sha256_extract_sha256_linkage_descriptor(
    sha256_extract_layer_index: usize,
    sha256_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sha256_air as sha;

    // A side (Sha256Extract): 64 input bytes + 32 output bytes.
    let mut a_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..NUM_INPUT_BYTES {
        a_columns.push(COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..NUM_OUTPUT_BYTES {
        a_columns.push(COL_OUTPUT_BYTE_OFFSET + b);
    }

    // B side (bit-level SHA-256): the per-invocation byte aggregator
    // columns gated by IS_FIRST_INV_ROW (1 only on round 0 of block 0).
    let mut b_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..sha::INV_INPUT_LEN {
        b_columns.push(sha::inv_input_byte(b));
    }
    for b in 0..sha::INV_OUTPUT_LEN {
        b_columns.push(sha::inv_output_byte(b));
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sha256_extract_sha256_v1".into(),
        a_layer_index: sha256_extract_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_layer_index,
        b_columns,
        b_selector_column: Some(sha::COL_IS_FIRST_INV_ROW),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssz::Chunk;

    fn small_witness() -> Sha256ExtractWitness {
        let mk_chunk = |seed: u8| {
            let mut c: Chunk = [0u8; 32];
            for (i, b) in c.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            c
        };
        Sha256ExtractWitness::from_pair_inputs(&[
            (mk_chunk(0x10), mk_chunk(0x20)),
            (mk_chunk(0x30), mk_chunk(0x40)),
            (mk_chunk(0x50), mk_chunk(0x60)),
        ])
    }

    #[test]
    fn build_poly_matches_evaluate_at_point() {
        use crate::commitment;
        use bls48581::bls48581::big;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Sha256ExtractConstraintSystem::new(trace.num_rows);
        let domain_size = trace.padded_size;

        let col_coeffs: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| {
            let big_evals: Vec<big::BIG> = p.evaluations.iter()
                .map(|s| big::BIG::new_copy(s.as_bls48581()))
                .collect();
            let coeffs = commitment::eval_to_coeff(&big_evals, domain_size);
            coeffs.into_iter().map(Scalar::Bls48581).collect()
        }).collect();

        let z_big = big::BIG::new_int(123456789);
        let alpha_big = big::BIG::new_int(987654321);
        let alpha = Scalar::Bls48581(big::BIG::new_copy(&alpha_big));

        let col_evals_at_z: Vec<Scalar> = col_coeffs.iter().map(|coeffs| {
            let big_coeffs: Vec<big::BIG> = coeffs.iter()
                .map(|s| big::BIG::new_copy(s.as_bls48581()))
                .collect();
            Scalar::Bls48581(commitment::eval_poly_at(&big_coeffs, &z_big))
        }).collect();

        let c_at_z_eval = cs.evaluate_at_point(&col_evals_at_z, &alpha);

        let c_poly = cs.build_constraint_polynomial(&col_coeffs, &alpha, domain_size);
        let c_poly_big: Vec<big::BIG> = c_poly.iter()
            .map(|s| big::BIG::new_copy(s.as_bls48581()))
            .collect();
        let c_at_z_poly = Scalar::Bls48581(commitment::eval_poly_at(&c_poly_big, &z_big));

        assert_eq!(
            c_at_z_eval.as_bls48581().tostring(),
            c_at_z_poly.as_bls48581().tostring(),
            "sha256_extract: build_constraint_polynomial(z) must equal evaluate_at_point",
        );
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 3);
        // Row 0 input byte 0 = 0x10.
        assert_eq!(
            trace.columns[COL_INPUT_BYTE_OFFSET].evaluations[0].to_bytes(),
            Scalar::from_u64(0x10, curve).to_bytes()
        );
        // Row 0 IS_REAL = 1.
        assert_eq!(
            trace.columns[COL_IS_REAL].evaluations[0].to_bytes(),
            Scalar::one(curve).to_bytes()
        );
        // Padding row IS_REAL = 0.
        assert!(trace.columns[COL_IS_REAL].evaluations.last().unwrap().is_zero());
    }

    #[test]
    fn from_pair_inputs_computes_sha256_pair() {
        let mut left: Chunk = [0u8; 32];
        let mut right: Chunk = [0u8; 32];
        left[0] = 1;
        right[0] = 2;
        let w = Sha256ExtractWitness::from_pair_inputs(&[(left, right)]);
        assert_eq!(w.invocations.len(), 1);
        let expected = crate::sha256::sha256_pair(&left, &right);
        assert_eq!(w.invocations[0].output, expected);
        assert_eq!(&w.invocations[0].input[..32], &left);
        assert_eq!(&w.invocations[0].input[32..], &right);
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = Sha256ExtractConstraintSystem::new(trace.num_rows);
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
    fn ssz_sha256_extract_descriptor_well_formed() {
        let desc = make_ssz_sha256_extract_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "ssz_sha256_extract_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        // 96-byte tuple on each side.
        assert_eq!(desc.a_columns.len(), 96);
        assert_eq!(desc.b_columns.len(), 96);
        // SSZ A-side: LEFT[0..32], RIGHT[0..32], PARENT[0..32] in order.
        assert_eq!(desc.a_columns[0], crate::ssz_air::col::LEFT_OFFSET);
        assert_eq!(desc.a_columns[31], crate::ssz_air::col::LEFT_OFFSET + 31);
        assert_eq!(desc.a_columns[32], crate::ssz_air::col::RIGHT_OFFSET);
        assert_eq!(desc.a_columns[63], crate::ssz_air::col::RIGHT_OFFSET + 31);
        assert_eq!(desc.a_columns[64], crate::ssz_air::col::PARENT_OFFSET);
        assert_eq!(desc.a_columns[95], crate::ssz_air::col::PARENT_OFFSET + 31);
        // Sha256Extract B-side: INPUT[0..64], OUTPUT[0..32] in order.
        assert_eq!(desc.b_columns[0], COL_INPUT_BYTE_OFFSET);
        assert_eq!(desc.b_columns[63], COL_INPUT_BYTE_OFFSET + 63);
        assert_eq!(desc.b_columns[64], COL_OUTPUT_BYTE_OFFSET);
        assert_eq!(desc.b_columns[95], COL_OUTPUT_BYTE_OFFSET + 31);
        // Selectors.
        assert_eq!(
            desc.a_selector_column,
            Some(crate::ssz_air::col::IS_LEFT_REAL)
        );
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn sha256_extract_sha256_descriptor_well_formed() {
        use crate::sha256_air as sha;
        let desc = make_sha256_extract_sha256_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "sha256_extract_sha256_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        // 96-byte tuple on each side.
        assert_eq!(desc.a_columns.len(), 96);
        assert_eq!(desc.b_columns.len(), 96);
        // A side: Sha256Extract INPUT[0..64], OUTPUT[0..32].
        assert_eq!(desc.a_columns[0], COL_INPUT_BYTE_OFFSET);
        assert_eq!(desc.a_columns[63], COL_INPUT_BYTE_OFFSET + 63);
        assert_eq!(desc.a_columns[64], COL_OUTPUT_BYTE_OFFSET);
        assert_eq!(desc.a_columns[95], COL_OUTPUT_BYTE_OFFSET + 31);
        // B side: bit-level SHA-256 INV_INPUT_BYTE[0..64], INV_OUTPUT_BYTE[0..32].
        assert_eq!(desc.b_columns[0], sha::COL_INV_INPUT_BYTE_OFFSET);
        assert_eq!(desc.b_columns[63], sha::COL_INV_INPUT_BYTE_OFFSET + 63);
        assert_eq!(desc.b_columns[64], sha::COL_INV_OUTPUT_BYTE_OFFSET);
        assert_eq!(desc.b_columns[95], sha::COL_INV_OUTPUT_BYTE_OFFSET + 31);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.b_selector_column, Some(sha::COL_IS_FIRST_INV_ROW));
    }

    #[test]
    fn sha256_extract_byte_tuples_match_bit_level_aggregator() {
        // Build a single sha256_pair witness, populate the bit-level
        // SHA-256 trace WITH invocation aggregator columns, then assert
        // the aggregator bytes match Sha256Extract's single row.
        use crate::sha256::sha256_witness;
        use crate::sha256_air as sha;
        let curve = CurveType::Bls48581;

        let mut left = [0u8; 32];
        let mut right = [0u8; 32];
        for i in 0..32 {
            left[i] = (0x10u8).wrapping_add(i as u8);
            right[i] = (0xa0u8).wrapping_add(i as u8);
        }
        let mut input64 = [0u8; 64];
        input64[..32].copy_from_slice(&left);
        input64[32..].copy_from_slice(&right);

        let ht = sha256_witness(&input64);
        let bit_cols = sha::populate_trace_from_hash_with_invocation_bytes(&ht, &input64, curve);

        // Bit-level: aggregator columns must equal input64 / digest on
        // every row, and IS_FIRST_INV_ROW must be 1 on row 0 only.
        let n = bit_cols[0].len();
        for r in 0..n {
            for b in 0..sha::INV_INPUT_LEN {
                let got = bit_cols[sha::inv_input_byte(b)][r].to_u64() as u8;
                assert_eq!(got, input64[b], "row {} input byte {}", r, b);
            }
            for b in 0..sha::INV_OUTPUT_LEN {
                let got = bit_cols[sha::inv_output_byte(b)][r].to_u64() as u8;
                assert_eq!(got, ht.digest[b], "row {} output byte {}", r, b);
            }
        }
        assert_eq!(bit_cols[sha::COL_IS_FIRST_INV_ROW][0].to_u64(), 1);
        for r in 1..n {
            assert_eq!(
                bit_cols[sha::COL_IS_FIRST_INV_ROW][r].to_u64(),
                0,
                "IS_FIRST_INV_ROW must be 0 on row {}",
                r
            );
        }

        // Sha256Extract: per-row tuple matches aggregator at the anchor.
        let extract_w = Sha256ExtractWitness::from_pair_inputs(&[(left, right)]);
        let extract_trace = build_trace_polynomials(&extract_w, curve);
        for b in 0..NUM_INPUT_BYTES {
            let extract_byte =
                extract_trace.columns[COL_INPUT_BYTE_OFFSET + b].evaluations[0].to_u64() as u8;
            let bit_anchor_byte = bit_cols[sha::inv_input_byte(b)][0].to_u64() as u8;
            assert_eq!(
                extract_byte, bit_anchor_byte,
                "input tuple mismatch at byte {}",
                b
            );
        }
        for b in 0..NUM_OUTPUT_BYTES {
            let extract_byte =
                extract_trace.columns[COL_OUTPUT_BYTE_OFFSET + b].evaluations[0].to_u64() as u8;
            let bit_anchor_byte = bit_cols[sha::inv_output_byte(b)][0].to_u64() as u8;
            assert_eq!(
                extract_byte, bit_anchor_byte,
                "output tuple mismatch at byte {}",
                b
            );
        }
    }

    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn sha256_extract_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = Sha256ExtractConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "sha256 extract proof must verify");
    }
}
