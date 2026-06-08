//! Attester slashing AIR — proves a Casper-FFG slashing condition is met.
//!
//! Ethereum 2.0 attester slashing condition:
//!   A validator is slashed if they sign two attestations `att_1`, `att_2`
//!   such that `att_1.data != att_2.data` AND one of:
//!     (i)  same-target double vote: `att_1.target.epoch == att_2.target.epoch`
//!     (ii) surround vote:           `source_1 < source_2 AND target_2 < target_1`
//!                                   (or the symmetric case).
//!   AND the two attesting-indices sets intersect.
//!
//! This scaffold AIR commits one row per slashing operation. It binds the
//! arithmetic of the slashing condition (epoch ordering, distinctness) and
//! exposes signature columns for cross-AIR LogUp linkage into two BLS
//! pairing AIRs. Deferred:
//!   - Actual BLS verification of `signature_1` / `signature_2` (descriptor
//!     only; the closure shape is documented and pinned by tests).
//!   - Full intersection enumeration of `attesting_indices_*` (we commit a
//!     host-computed `intersection_count` and the first 16 indices of each
//!     side as a truncated scaffold; an inclusion-AIR follow-up will prove
//!     non-emptiness algebraically).
//!
//! # Column layout (NUM_COLUMNS = 134)
//!
//!   0..16   `ATTESTING_INDICES_1`              — first 16 indices of att_1
//!   16..32  `ATTESTING_INDICES_2`              — first 16 indices of att_2
//!   32      `SOURCE_EPOCH_1` (u64)
//!   33      `TARGET_EPOCH_1`
//!   34      `SOURCE_EPOCH_2`
//!   35      `TARGET_EPOCH_2`
//!   36..68  `BLOCK_ROOT_1[0..32]` (BE bytes of att_1.data.beacon_block_root)
//!   68..100 `BLOCK_ROOT_2[0..32]`
//!   100     `IS_SAME_TARGET`           — binary; 1 iff target_1 == target_2
//!   101     `IS_SURROUND`              — binary; 1 iff surround vote
//!   102     `IS_REAL`                  — binary; 1 on real rows, 0 padding
//!   103     `INTERSECTION_COUNT`       — host-computed, > 0 on real rows
//!   104     `DELTA_SOURCE`             — witnessed `|source_1 − source_2|`
//!                                        when surround, else 0
//!   105     `DELTA_TARGET`             — witnessed `|target_1 − target_2|`
//!                                        when surround, else 0
//!   106     `BLOCK_ROOT_DIFF_INV`      — witnessed inverse of the β-RLC
//!                                        difference of the two block roots;
//!                                        binds `block_root_1 ≠ block_root_2`
//!   107     `SIGNATURE_1_FIRST_LIMB`   — first 8 bytes of signature_1 packed
//!                                        as a u64 (handle for descriptor)
//!   108     `SIGNATURE_2_FIRST_LIMB`
//!   109..134 (padding placeholder columns reserved for future expansion)
//!
//! NUM_COLUMNS chosen to leave room without bloating; only 0..109 are
//! actively constrained.
//!
//! # Row-local constraints (≥ 10)
//!
//!   0. `is_real_binary`            — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `is_same_target_binary`     — `IS_SAME_TARGET · (IS_SAME_TARGET − 1) = 0`
//!   2. `is_surround_binary`        — `IS_SURROUND · (IS_SURROUND − 1) = 0`
//!   3. `at_least_one_condition`    — `IS_REAL · (1 − IS_SAME_TARGET − IS_SURROUND
//!                                     + IS_SAME_TARGET · IS_SURROUND) = 0`
//!      (the bracket is `1 − (a ∨ b)` when `a, b` are binary, so equals 0 iff
//!      at least one of `IS_SAME_TARGET, IS_SURROUND` is 1.)
//!   4. `same_target_consistency`   — `IS_SAME_TARGET · (TARGET_EPOCH_1 − TARGET_EPOCH_2) = 0`
//!   5. `surround_source_consistent` — `IS_SURROUND · (SOURCE_EPOCH_1 − SOURCE_EPOCH_2 − DELTA_SOURCE) · ...`
//!      degree-2 expression encoding `(s1 − s2) ∈ {+δ_s, −δ_s}` (we use the
//!      simpler "witnessed signed delta" form below).
//!   6. `surround_target_consistent` — analogous for target.
//!   7. `intersection_count_nonzero_on_real`
//!      — `IS_REAL · (INTERSECTION_COUNT_INV · INTERSECTION_COUNT − 1) = 0`
//!      Witnessed via an internal inverse handle. We fold this into a
//!      single constraint with a co-witnessed value (see note below).
//!   8. `block_root_distinct`       — β-RLC of the byte-diff of the two
//!      block roots times `BLOCK_ROOT_DIFF_INV` equals 1 on real rows
//!      (forces `block_root_1 ≠ block_root_2` via Schwartz-Zippel on β).
//!   9. `delta_zero_when_not_surround` — `(1 − IS_SURROUND) · DELTA_SOURCE = 0`
//!   10. `delta_target_zero_when_not_surround` — `(1 − IS_SURROUND) · DELTA_TARGET = 0`
//!
//! Plus 32 + 32 = 64 byte range checks (8-bit) on `BLOCK_ROOT_1` / `_2`.
//!
//! # Cross-AIR LogUp descriptors
//!
//!   * [`make_attester_slashing_to_sig_1_descriptor`] —
//!     A side = this AIR's `SIGNATURE_1_FIRST_LIMB` (one row, gated by
//!     `IS_REAL`) ↔ B side = `bls_pairing_air`'s `COL_SIG_BYTES_OFFSET`
//!     limb. Once the signature byte-decomposition is wired the tuple
//!     widens to 96 bytes; for the scaffold a single representative limb
//!     pins the shape.
//!   * [`make_attester_slashing_to_sig_2_descriptor`] — same for signature 2.

use crate::beacon::{Epoch, IndexedAttestation};
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ─────────────────────────────────────────────────────────

pub const MAX_INDICES_SCAFFOLD: usize = 16;
pub const BLOCK_ROOT_LEN: usize = 32;

// ─── Column layout ─────────────────────────────────────────────────────

pub const COL_ATTESTING_INDICES_1_OFFSET: usize = 0;
pub const COL_ATTESTING_INDICES_2_OFFSET: usize = COL_ATTESTING_INDICES_1_OFFSET + MAX_INDICES_SCAFFOLD;
pub const COL_SOURCE_EPOCH_1: usize = COL_ATTESTING_INDICES_2_OFFSET + MAX_INDICES_SCAFFOLD;
pub const COL_TARGET_EPOCH_1: usize = COL_SOURCE_EPOCH_1 + 1;
pub const COL_SOURCE_EPOCH_2: usize = COL_TARGET_EPOCH_1 + 1;
pub const COL_TARGET_EPOCH_2: usize = COL_SOURCE_EPOCH_2 + 1;
pub const COL_BLOCK_ROOT_1_OFFSET: usize = COL_TARGET_EPOCH_2 + 1;
pub const COL_BLOCK_ROOT_2_OFFSET: usize = COL_BLOCK_ROOT_1_OFFSET + BLOCK_ROOT_LEN;
pub const COL_IS_SAME_TARGET: usize = COL_BLOCK_ROOT_2_OFFSET + BLOCK_ROOT_LEN;
pub const COL_IS_SURROUND: usize = COL_IS_SAME_TARGET + 1;
pub const COL_IS_REAL: usize = COL_IS_SURROUND + 1;
pub const COL_INTERSECTION_COUNT: usize = COL_IS_REAL + 1;
pub const COL_INTERSECTION_COUNT_INV: usize = COL_INTERSECTION_COUNT + 1;
pub const COL_DELTA_SOURCE: usize = COL_INTERSECTION_COUNT_INV + 1;
pub const COL_DELTA_TARGET: usize = COL_DELTA_SOURCE + 1;
pub const COL_BLOCK_ROOT_DIFF_INV: usize = COL_DELTA_TARGET + 1;
pub const COL_SIGNATURE_1_FIRST_LIMB: usize = COL_BLOCK_ROOT_DIFF_INV + 1;
pub const COL_SIGNATURE_2_FIRST_LIMB: usize = COL_SIGNATURE_1_FIRST_LIMB + 1;
/// Surround-sign selector: `SURROUND_SIGN = 1` iff `source_1 > source_2`
/// (and correspondingly `target_1 < target_2` is enforced); else 0.
/// Used to encode the signed delta: `(source_1 − source_2) =
/// (2·SURROUND_SIGN − 1) · DELTA_SOURCE`.
pub const COL_SURROUND_SIGN: usize = COL_SIGNATURE_2_FIRST_LIMB + 1;

pub const NUM_COLUMNS: usize = COL_SURROUND_SIGN + 1;

/// Row-local constraint count.
///   0:  is_real_binary
///   1:  is_same_target_binary
///   2:  is_surround_binary
///   3:  surround_sign_binary
///   4:  at_least_one_condition_on_real
///   5:  same_target_consistency
///   6:  surround_source_signed_consistency
///   7:  surround_target_signed_consistency
///   8:  intersection_nonzero_on_real
///   9:  block_root_distinct_on_real
///   10: delta_source_zero_when_not_surround
///   11: delta_target_zero_when_not_surround
pub const NUM_ROW_CONSTRAINTS: usize = 12;

pub const NUM_SHIFTED: usize = 0;

// ─── Witness ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AttesterSlashingWitness {
    /// First 16 attesting indices of `attestation_1` (truncated).
    pub attesting_indices_1: Vec<u64>,
    /// First 16 attesting indices of `attestation_2` (truncated).
    pub attesting_indices_2: Vec<u64>,
    pub source_epoch_1: Epoch,
    pub target_epoch_1: Epoch,
    pub source_epoch_2: Epoch,
    pub target_epoch_2: Epoch,
    pub block_root_1: [u8; 32],
    pub block_root_2: [u8; 32],
    pub is_same_target: bool,
    pub is_surround: bool,
    /// Host-computed: number of validator indices appearing in both
    /// `attesting_indices_1` and `attesting_indices_2`. Must be > 0 for
    /// a valid slashing operation.
    pub intersection_count: u64,
    /// 96-byte BLS signatures (exposed via first-limb columns for the
    /// pairing-air linkage; full byte decomposition deferred).
    pub signature_1: [u8; 96],
    pub signature_2: [u8; 96],
}

impl AttesterSlashingWitness {
    /// Build a witness from two `IndexedAttestation`s. Asserts the host
    /// preconditions:
    ///   - the two `AttestationData`s differ,
    ///   - at least one slashing condition (same-target OR surround) holds,
    ///   - the attesting-indices sets intersect.
    pub fn from_attestations(att_1: &IndexedAttestation, att_2: &IndexedAttestation) -> Self {
        assert!(
            att_1.data != att_2.data,
            "attester slashing requires distinct AttestationData",
        );

        let target_1 = att_1.data.target.epoch;
        let target_2 = att_2.data.target.epoch;
        let source_1 = att_1.data.source.epoch;
        let source_2 = att_2.data.source.epoch;

        let is_same_target = target_1 == target_2;
        // Surround vote: one attestation's source/target strictly
        // surrounds the other's. Casper FFG: `s_a < s_b AND t_b < t_a`.
        let is_surround = (source_1 < source_2 && target_2 < target_1)
            || (source_2 < source_1 && target_1 < target_2);

        assert!(
            is_same_target || is_surround,
            "attester slashing requires same-target OR surround vote",
        );

        // Compute intersection.
        let set_1: std::collections::HashSet<u64> = att_1.attesting_indices.iter().copied().collect();
        let intersection_count = att_2
            .attesting_indices
            .iter()
            .filter(|i| set_1.contains(*i))
            .count() as u64;
        assert!(
            intersection_count > 0,
            "attester slashing requires non-empty index intersection",
        );

        let mut indices_1 = att_1.attesting_indices.clone();
        indices_1.resize(MAX_INDICES_SCAFFOLD, 0);
        indices_1.truncate(MAX_INDICES_SCAFFOLD);
        let mut indices_2 = att_2.attesting_indices.clone();
        indices_2.resize(MAX_INDICES_SCAFFOLD, 0);
        indices_2.truncate(MAX_INDICES_SCAFFOLD);

        Self {
            attesting_indices_1: indices_1,
            attesting_indices_2: indices_2,
            source_epoch_1: source_1,
            target_epoch_1: target_1,
            source_epoch_2: source_2,
            target_epoch_2: target_2,
            block_root_1: att_1.data.beacon_block_root,
            block_root_2: att_2.data.beacon_block_root,
            is_same_target,
            is_surround,
            intersection_count,
            signature_1: att_1.signature,
            signature_2: att_2.signature,
        }
    }

    /// β-RLC of `(block_root_1 − block_root_2)` byte-by-byte under base
    /// `BLOCK_ROOT_BETA` (a constant chosen to be field-safe; for the
    /// scaffold we use a small power-of-two-style fixed scalar).
    /// Returns the field difference and its inverse, or `None` if the
    /// roots are identical (witness construction error).
    pub fn block_root_rlc_diff_and_inv(&self, curve: CurveType) -> Option<(Scalar, Scalar)> {
        let beta = Scalar::from_u64(BLOCK_ROOT_RLC_BETA, curve);
        let mut diff = Scalar::zero(curve);
        for k in 0..BLOCK_ROOT_LEN {
            let b1 = Scalar::from_u64(self.block_root_1[k] as u64, curve);
            let b2 = Scalar::from_u64(self.block_root_2[k] as u64, curve);
            diff = diff.mul(&beta).add(&b1.sub(&b2));
        }
        if diff.is_zero() {
            None
        } else {
            let inv = diff.inverse();
            Some((diff, inv))
        }
    }
}

/// Fixed scalar β used to RLC-fold the 32-byte block-root diff into a
/// single field-element for the distinctness check. The actual binding
/// soundness is improved when β is sampled from the Fiat-Shamir
/// transcript; this constant scaffold gives a clean witness builder for
/// the standalone AIR. A tampered (b1 == b2) witness has no valid
/// inverse, so the constraint cannot be satisfied.
pub const BLOCK_ROOT_RLC_BETA: u64 = 257;

// ─── Trace builder ─────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &AttesterSlashingWitness,
    curve: CurveType,
) -> TracePolynomials {
    // One row per slashing operation in this scaffold.
    let num_rows = 1usize;
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    // Row 0 — the slashing op.
    for k in 0..MAX_INDICES_SCAFFOLD {
        let v1 = witness
            .attesting_indices_1
            .get(k)
            .copied()
            .unwrap_or(0);
        let v2 = witness
            .attesting_indices_2
            .get(k)
            .copied()
            .unwrap_or(0);
        columns[COL_ATTESTING_INDICES_1_OFFSET + k][0] = Scalar::from_u64(v1, curve);
        columns[COL_ATTESTING_INDICES_2_OFFSET + k][0] = Scalar::from_u64(v2, curve);
    }
    columns[COL_SOURCE_EPOCH_1][0] = Scalar::from_u64(witness.source_epoch_1, curve);
    columns[COL_TARGET_EPOCH_1][0] = Scalar::from_u64(witness.target_epoch_1, curve);
    columns[COL_SOURCE_EPOCH_2][0] = Scalar::from_u64(witness.source_epoch_2, curve);
    columns[COL_TARGET_EPOCH_2][0] = Scalar::from_u64(witness.target_epoch_2, curve);

    for k in 0..BLOCK_ROOT_LEN {
        columns[COL_BLOCK_ROOT_1_OFFSET + k][0] =
            Scalar::from_u64(witness.block_root_1[k] as u64, curve);
        columns[COL_BLOCK_ROOT_2_OFFSET + k][0] =
            Scalar::from_u64(witness.block_root_2[k] as u64, curve);
    }

    columns[COL_IS_SAME_TARGET][0] = if witness.is_same_target {
        one.clone()
    } else {
        zero.clone()
    };
    columns[COL_IS_SURROUND][0] = if witness.is_surround {
        one.clone()
    } else {
        zero.clone()
    };
    columns[COL_IS_REAL][0] = one.clone();
    columns[COL_INTERSECTION_COUNT][0] =
        Scalar::from_u64(witness.intersection_count, curve);
    columns[COL_INTERSECTION_COUNT_INV][0] = if witness.intersection_count == 0 {
        zero.clone()
    } else {
        Scalar::from_u64(witness.intersection_count, curve).inverse()
    };

    // Signed delta encoding: source_1 − source_2 = (2·sign − 1) · |source_1 − source_2|.
    // SURROUND_SIGN = 1 iff source_1 > source_2.
    let (delta_source, delta_target, surround_sign) = if witness.is_surround {
        if witness.source_epoch_1 > witness.source_epoch_2 {
            // Symmetric case: s_2 < s_1, t_2 > t_1 (att_2 surrounds att_1)?
            // Actually FFG surround is "one surrounds the other"; with
            // SURROUND_SIGN = 1 → s_1 > s_2 and the validity constraint
            // also requires t_1 < t_2.
            (
                witness.source_epoch_1 - witness.source_epoch_2,
                witness.target_epoch_2 - witness.target_epoch_1,
                1u64,
            )
        } else {
            (
                witness.source_epoch_2 - witness.source_epoch_1,
                witness.target_epoch_1 - witness.target_epoch_2,
                0u64,
            )
        }
    } else {
        (0u64, 0u64, 0u64)
    };
    columns[COL_DELTA_SOURCE][0] = Scalar::from_u64(delta_source, curve);
    columns[COL_DELTA_TARGET][0] = Scalar::from_u64(delta_target, curve);
    columns[COL_SURROUND_SIGN][0] = Scalar::from_u64(surround_sign, curve);

    // Block-root distinctness inverse.
    let (_diff, inv) = witness
        .block_root_rlc_diff_and_inv(curve)
        .expect("attester_slashing witness must commit distinct block_roots");
    columns[COL_BLOCK_ROOT_DIFF_INV][0] = inv;

    // Signature first-limb packing: first 8 bytes BE → u64.
    let sig1_first = u64::from_be_bytes([
        witness.signature_1[0], witness.signature_1[1], witness.signature_1[2],
        witness.signature_1[3], witness.signature_1[4], witness.signature_1[5],
        witness.signature_1[6], witness.signature_1[7],
    ]);
    let sig2_first = u64::from_be_bytes([
        witness.signature_2[0], witness.signature_2[1], witness.signature_2[2],
        witness.signature_2[3], witness.signature_2[4], witness.signature_2[5],
        witness.signature_2[6], witness.signature_2[7],
    ]);
    columns[COL_SIGNATURE_1_FIRST_LIMB][0] = Scalar::from_u64(sig1_first, curve);
    columns[COL_SIGNATURE_2_FIRST_LIMB][0] = Scalar::from_u64(sig2_first, curve);

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

pub struct AttesterSlashingConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AttesterSlashingConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// β-RLC fold of `block_root_1 − block_root_2` byte-by-byte as a scalar
/// polynomial. Returns a degree-0 (constant per-row) representation
/// suitable for `evaluate_*`.
fn block_root_rlc_diff_scalar(cols: &[Scalar]) -> Scalar {
    let curve = cols[0].curve_type();
    let beta = Scalar::from_u64(BLOCK_ROOT_RLC_BETA, curve);
    let mut diff = Scalar::zero(curve);
    for k in 0..BLOCK_ROOT_LEN {
        let b1 = &cols[COL_BLOCK_ROOT_1_OFFSET + k];
        let b2 = &cols[COL_BLOCK_ROOT_2_OFFSET + k];
        diff = diff.mul(&beta).add(&b1.sub(b2));
    }
    diff
}

fn block_root_rlc_diff_poly(cols: &[Vec<Scalar>], curve: CurveType) -> Vec<Scalar> {
    let beta = Scalar::from_u64(BLOCK_ROOT_RLC_BETA, curve);
    let mut diff = vec![Scalar::zero(curve)];
    for k in 0..BLOCK_ROOT_LEN {
        diff = poly_scalar_mul(&diff, &beta);
        let b1 = &cols[COL_BLOCK_ROOT_1_OFFSET + k];
        let b2 = &cols[COL_BLOCK_ROOT_2_OFFSET + k];
        let bdiff = poly_sub(b1, b2, curve);
        diff = poly_add(&diff, &bdiff, curve);
    }
    diff
}

impl VmConstraintSystem for AttesterSlashingConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_same_target_binary".into(),
            "is_surround_binary".into(),
            "surround_sign_binary".into(),
            "at_least_one_condition_on_real".into(),
            "same_target_consistency".into(),
            "surround_source_signed_consistency".into(),
            "surround_target_signed_consistency".into(),
            "intersection_nonzero_on_real".into(),
            "block_root_distinct_on_real".into(),
            "delta_source_zero_when_not_surround".into(),
            "delta_target_zero_when_not_surround".into(),
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
        let two = Scalar::from_u64(2, curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        for r in 0..n {
            let row: Vec<Scalar> = columns.iter().map(|c| c[r].clone()).collect();
            let is_real = &row[COL_IS_REAL];
            let is_st = &row[COL_IS_SAME_TARGET];
            let is_sr = &row[COL_IS_SURROUND];
            let surround_sign = &row[COL_SURROUND_SIGN];

            // 0: is_real binary.
            out[0][r] = is_real.mul(&is_real.sub(&one));
            // 1: is_same_target binary.
            out[1][r] = is_st.mul(&is_st.sub(&one));
            // 2: is_surround binary.
            out[2][r] = is_sr.mul(&is_sr.sub(&one));
            // 3: surround_sign binary.
            out[3][r] = surround_sign.mul(&surround_sign.sub(&one));
            // 4: at-least-one on real rows.
            //   bracket = 1 - st - sr + st*sr  (= 1 - (st OR sr) for binary)
            let bracket = one
                .sub(is_st)
                .sub(is_sr)
                .add(&is_st.mul(is_sr));
            out[4][r] = is_real.mul(&bracket);
            // 5: same target consistency.
            let t_diff = row[COL_TARGET_EPOCH_1].sub(&row[COL_TARGET_EPOCH_2]);
            out[5][r] = is_st.mul(&t_diff);
            // 6: surround source signed: IS_SURROUND · (source_1 − source_2 − (2·sign − 1)·delta_source) = 0
            let sign_factor = two.mul(surround_sign).sub(&one);
            let s_diff = row[COL_SOURCE_EPOCH_1].sub(&row[COL_SOURCE_EPOCH_2]);
            let s_body = s_diff.sub(&sign_factor.mul(&row[COL_DELTA_SOURCE]));
            out[6][r] = is_sr.mul(&s_body);
            // 7: surround target signed: IS_SURROUND · (target_2 − target_1 − (2·sign − 1)·delta_target) = 0
            //    On sign=1 (s1>s2) the surround validity requires t1<t2, i.e. t2−t1 = +delta_t.
            //    On sign=0 (s1<s2) requires t1>t2, i.e. t1−t2 = +delta_t, equivalently t2−t1 = −delta_t.
            let t_diff_2_1 = row[COL_TARGET_EPOCH_2].sub(&row[COL_TARGET_EPOCH_1]);
            let t_body = t_diff_2_1.sub(&sign_factor.mul(&row[COL_DELTA_TARGET]));
            out[7][r] = is_sr.mul(&t_body);
            // 8: IS_REAL · (INTERSECTION_COUNT · INTERSECTION_COUNT_INV − 1) = 0
            let ic_prod = row[COL_INTERSECTION_COUNT].mul(&row[COL_INTERSECTION_COUNT_INV]);
            out[8][r] = is_real.mul(&ic_prod.sub(&one));
            // 9: block-root distinct on real: IS_REAL · (rlc_diff · inv − 1) = 0
            let rlc_diff = block_root_rlc_diff_scalar(&row);
            let prod = rlc_diff.mul(&row[COL_BLOCK_ROOT_DIFF_INV]);
            out[9][r] = is_real.mul(&prod.sub(&one));
            // 10: (1 − IS_SURROUND) · DELTA_SOURCE = 0
            out[10][r] = one.sub(is_sr).mul(&row[COL_DELTA_SOURCE]);
            // 11: (1 − IS_SURROUND) · DELTA_TARGET = 0
            out[11][r] = one.sub(is_sr).mul(&row[COL_DELTA_TARGET]);
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

        let is_real = &col_evals[COL_IS_REAL];
        let is_st = &col_evals[COL_IS_SAME_TARGET];
        let is_sr = &col_evals[COL_IS_SURROUND];
        let surround_sign = &col_evals[COL_SURROUND_SIGN];

        let bodies: Vec<Scalar> = vec![
            is_real.mul(&is_real.sub(&one)),
            is_st.mul(&is_st.sub(&one)),
            is_sr.mul(&is_sr.sub(&one)),
            surround_sign.mul(&surround_sign.sub(&one)),
            is_real.mul(&one.sub(is_st).sub(is_sr).add(&is_st.mul(is_sr))),
            is_st.mul(&col_evals[COL_TARGET_EPOCH_1].sub(&col_evals[COL_TARGET_EPOCH_2])),
            {
                let sign_factor = two.mul(surround_sign).sub(&one);
                let s_diff = col_evals[COL_SOURCE_EPOCH_1].sub(&col_evals[COL_SOURCE_EPOCH_2]);
                is_sr.mul(&s_diff.sub(&sign_factor.mul(&col_evals[COL_DELTA_SOURCE])))
            },
            {
                let sign_factor = two.mul(surround_sign).sub(&one);
                let t_diff = col_evals[COL_TARGET_EPOCH_2].sub(&col_evals[COL_TARGET_EPOCH_1]);
                is_sr.mul(&t_diff.sub(&sign_factor.mul(&col_evals[COL_DELTA_TARGET])))
            },
            is_real.mul(
                &col_evals[COL_INTERSECTION_COUNT]
                    .mul(&col_evals[COL_INTERSECTION_COUNT_INV])
                    .sub(&one),
            ),
            {
                let rlc_diff = block_root_rlc_diff_scalar(col_evals);
                is_real.mul(
                    &rlc_diff
                        .mul(&col_evals[COL_BLOCK_ROOT_DIFF_INV])
                        .sub(&one),
                )
            },
            one.sub(is_sr).mul(&col_evals[COL_DELTA_SOURCE]),
            one.sub(is_sr).mul(&col_evals[COL_DELTA_TARGET]),
        ];

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&alpha_pow.mul(body));
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
        let two_poly = vec![Scalar::from_u64(2, curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_st = &col_coeffs[COL_IS_SAME_TARGET];
        let is_sr = &col_coeffs[COL_IS_SURROUND];
        let surround_sign = &col_coeffs[COL_SURROUND_SIGN];

        let bodies: Vec<Vec<Scalar>> = vec![
            // 0
            poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve),
            // 1
            poly_mul(is_st, &poly_sub(is_st, &one_poly, curve), curve),
            // 2
            poly_mul(is_sr, &poly_sub(is_sr, &one_poly, curve), curve),
            // 3
            poly_mul(surround_sign, &poly_sub(surround_sign, &one_poly, curve), curve),
            // 4: is_real · (1 - st - sr + st*sr)
            {
                let st_sr = poly_mul(is_st, is_sr, curve);
                let mut bracket = poly_sub(&one_poly, is_st, curve);
                bracket = poly_sub(&bracket, is_sr, curve);
                bracket = poly_add(&bracket, &st_sr, curve);
                poly_mul(is_real, &bracket, curve)
            },
            // 5: is_st · (target_1 − target_2)
            {
                let t_diff = poly_sub(
                    &col_coeffs[COL_TARGET_EPOCH_1],
                    &col_coeffs[COL_TARGET_EPOCH_2],
                    curve,
                );
                poly_mul(is_st, &t_diff, curve)
            },
            // 6: is_sr · (s1 − s2 − (2·sign − 1)·delta_source)
            {
                let two_sign = poly_mul(&two_poly, surround_sign, curve);
                let sign_factor = poly_sub(&two_sign, &one_poly, curve);
                let s_diff = poly_sub(
                    &col_coeffs[COL_SOURCE_EPOCH_1],
                    &col_coeffs[COL_SOURCE_EPOCH_2],
                    curve,
                );
                let sf_delta = poly_mul(&sign_factor, &col_coeffs[COL_DELTA_SOURCE], curve);
                let body = poly_sub(&s_diff, &sf_delta, curve);
                poly_mul(is_sr, &body, curve)
            },
            // 7: is_sr · ((t2 − t1) − (2·sign − 1)·delta_target)
            {
                let two_sign = poly_mul(&two_poly, surround_sign, curve);
                let sign_factor = poly_sub(&two_sign, &one_poly, curve);
                let t_diff = poly_sub(
                    &col_coeffs[COL_TARGET_EPOCH_2],
                    &col_coeffs[COL_TARGET_EPOCH_1],
                    curve,
                );
                let sf_delta = poly_mul(&sign_factor, &col_coeffs[COL_DELTA_TARGET], curve);
                let body = poly_sub(&t_diff, &sf_delta, curve);
                poly_mul(is_sr, &body, curve)
            },
            // 8: is_real · (ic · ic_inv − 1)
            {
                let prod = poly_mul(
                    &col_coeffs[COL_INTERSECTION_COUNT],
                    &col_coeffs[COL_INTERSECTION_COUNT_INV],
                    curve,
                );
                let body = poly_sub(&prod, &one_poly, curve);
                poly_mul(is_real, &body, curve)
            },
            // 9: is_real · (rlc_diff · inv − 1)
            {
                let rlc_diff = block_root_rlc_diff_poly(col_coeffs, curve);
                let prod = poly_mul(&rlc_diff, &col_coeffs[COL_BLOCK_ROOT_DIFF_INV], curve);
                let body = poly_sub(&prod, &one_poly, curve);
                poly_mul(is_real, &body, curve)
            },
            // 10: (1 − is_sr) · delta_source
            {
                let one_minus = poly_sub(&one_poly, is_sr, curve);
                poly_mul(&one_minus, &col_coeffs[COL_DELTA_SOURCE], curve)
            },
            // 11: (1 − is_sr) · delta_target
            {
                let one_minus = poly_sub(&one_poly, is_sr, curve);
                poly_mul(&one_minus, &col_coeffs[COL_DELTA_TARGET], curve)
            },
        ];

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);
        for body in &bodies {
            let scaled = poly_scalar_mul(body, &alpha_pow);
            acc = poly_add(&acc, &scaled, curve);
            alpha_pow = alpha_pow.mul(alpha);
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
        for k in 0..BLOCK_ROOT_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("block_root_1_{}_8bit", k),
                    column_index: COL_BLOCK_ROOT_1_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("block_root_2_{}_8bit", k),
                    column_index: COL_BLOCK_ROOT_2_OFFSET + k,
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

/// Bind `attestation_1.signature` to the `bls_pairing_air`'s
/// `(pk, sig, msg)` tuple — scaffold: aligns only the first signature
/// limb (one column on each side). Once the full per-byte signature
/// decomposition is wired into both AIRs the tuple widens to the
/// canonical 48 + 96 + 32 bytes used by [`crate::sync_committee_sig_air`].
pub fn make_attester_slashing_to_sig_1_descriptor(
    attester_slashing_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "attester_slashing_to_sig_1_v1".into(),
        a_layer_index: attester_slashing_layer_index,
        a_columns: vec![COL_SIGNATURE_1_FIRST_LIMB],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bls_pairing_layer_index,
        b_columns: vec![bp::COL_SIG_BYTES_OFFSET],
        b_selector_column: Some(bp::COL_IS_REAL),
    }
}

/// Bind `attestation_2.signature` to a (second) `bls_pairing_air`'s
/// signature column — scaffold for the second pairing check.
pub fn make_attester_slashing_to_sig_2_descriptor(
    attester_slashing_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "attester_slashing_to_sig_2_v1".into(),
        a_layer_index: attester_slashing_layer_index,
        a_columns: vec![COL_SIGNATURE_2_FIRST_LIMB],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bls_pairing_layer_index,
        b_columns: vec![bp::COL_SIG_BYTES_OFFSET],
        b_selector_column: Some(bp::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon::{AttestationData, Checkpoint, IndexedAttestation};

    fn attestation(
        source_epoch: u64,
        target_epoch: u64,
        block_root: [u8; 32],
        indices: Vec<u64>,
    ) -> IndexedAttestation {
        IndexedAttestation {
            attesting_indices: indices,
            data: AttestationData {
                slot: 0,
                index: 0,
                beacon_block_root: block_root,
                source: Checkpoint { epoch: source_epoch, root: [0u8; 32] },
                target: Checkpoint { epoch: target_epoch, root: [0u8; 32] },
            },
            signature: [0x11u8; 96],
        }
    }

    #[test]
    fn column_layout_pinned() {
        // Pin the column layout so refactors don't silently move things.
        assert_eq!(COL_ATTESTING_INDICES_1_OFFSET, 0);
        assert_eq!(COL_ATTESTING_INDICES_2_OFFSET, 16);
        assert_eq!(COL_SOURCE_EPOCH_1, 32);
        assert_eq!(COL_TARGET_EPOCH_1, 33);
        assert_eq!(COL_SOURCE_EPOCH_2, 34);
        assert_eq!(COL_TARGET_EPOCH_2, 35);
        assert_eq!(COL_BLOCK_ROOT_1_OFFSET, 36);
        assert_eq!(COL_BLOCK_ROOT_2_OFFSET, 68);
        assert_eq!(COL_IS_SAME_TARGET, 100);
        assert_eq!(COL_IS_SURROUND, 101);
        assert_eq!(COL_IS_REAL, 102);
        assert_eq!(NUM_ROW_CONSTRAINTS, 12);
    }

    #[test]
    fn same_target_double_vote_satisfies_constraints() {
        // Two attestations at the SAME target epoch, distinct block roots
        // → same-target double vote.
        let mut br1 = [0u8; 32];
        br1[0] = 0xAA;
        let mut br2 = [0u8; 32];
        br2[0] = 0xBB;
        let att_1 = attestation(5, 10, br1, vec![1, 2, 3, 4]);
        let att_2 = attestation(6, 10, br2, vec![3, 4, 5, 6]);
        let w = AttesterSlashingWitness::from_attestations(&att_1, &att_2);
        assert!(w.is_same_target);
        assert!(!w.is_surround);
        assert!(w.intersection_count >= 2); // 3, 4 shared

        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AttesterSlashingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) fired at row {}",
                    i,
                    cs.constraint_labels()[i],
                    r,
                );
            }
        }
    }

    #[test]
    fn surround_vote_satisfies_constraints() {
        // Surround: source_1 < source_2 AND target_2 < target_1.
        let mut br1 = [0u8; 32];
        br1[5] = 0x11;
        let mut br2 = [0u8; 32];
        br2[5] = 0x22;
        let att_1 = attestation(2, 20, br1, vec![10, 11, 12]);
        let att_2 = attestation(5, 15, br2, vec![11, 12, 13]);
        let w = AttesterSlashingWitness::from_attestations(&att_1, &att_2);
        assert!(!w.is_same_target);
        assert!(w.is_surround);

        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AttesterSlashingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) fired at row {} (surround)",
                    i,
                    cs.constraint_labels()[i],
                    r,
                );
            }
        }
    }

    #[test]
    fn surround_symmetric_satisfies_constraints() {
        // The other symmetric surround: source_1 > source_2 AND target_1 < target_2.
        let mut br1 = [0u8; 32];
        br1[31] = 0x99;
        let mut br2 = [0u8; 32];
        br2[31] = 0xAA;
        let att_1 = attestation(7, 12, br1, vec![20, 21]);
        let att_2 = attestation(3, 20, br2, vec![21, 22]);
        let w = AttesterSlashingWitness::from_attestations(&att_1, &att_2);
        assert!(!w.is_same_target);
        assert!(w.is_surround);

        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AttesterSlashingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) fired at row {} (symmetric surround)",
                    i,
                    cs.constraint_labels()[i],
                    r,
                );
            }
        }
    }

    #[test]
    fn tampered_same_target_with_different_epochs_fails() {
        // Build a valid same-target witness, then maliciously tamper
        // TARGET_EPOCH_2. Constraint 5 (same_target_consistency) must fire.
        let mut br1 = [0u8; 32];
        br1[0] = 0xAA;
        let mut br2 = [0u8; 32];
        br2[0] = 0xBB;
        let att_1 = attestation(5, 10, br1, vec![1, 2]);
        let att_2 = attestation(6, 10, br2, vec![2, 3]);
        let w = AttesterSlashingWitness::from_attestations(&att_1, &att_2);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: pretend target_2 is a different epoch but leave
        // IS_SAME_TARGET = 1 — constraint 5 must catch this.
        cols[COL_TARGET_EPOCH_2][0] = Scalar::from_u64(99, CurveType::Bls48581);
        let cs = AttesterSlashingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[5][0].is_zero(),
            "same_target_consistency must fire on tampered target_2",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_attester_slashing_to_sig_1_descriptor(0, 2);
        assert_eq!(d1.label, "attester_slashing_to_sig_1_v1");
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 2);
        assert_eq!(d1.a_columns, vec![COL_SIGNATURE_1_FIRST_LIMB]);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));

        let d2 = make_attester_slashing_to_sig_2_descriptor(0, 3);
        assert_eq!(d2.label, "attester_slashing_to_sig_2_v1");
        assert_eq!(d2.a_columns, vec![COL_SIGNATURE_2_FIRST_LIMB]);
        assert_eq!(d2.b_layer_index, 3);
    }

    #[test]
    fn both_signatures_distinct_in_trace() {
        let mut br1 = [0u8; 32];
        br1[1] = 0xDE;
        let mut br2 = [0u8; 32];
        br2[1] = 0xAD;
        let mut att_1 = attestation(1, 7, br1, vec![1]);
        let mut att_2 = attestation(1, 7, br2, vec![1]);
        // Make signatures distinct so the first-limb columns differ.
        att_1.signature[0] = 0xAA;
        att_2.signature[0] = 0xBB;
        let w = AttesterSlashingWitness::from_attestations(&att_1, &att_2);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);

        let sig1_col = trace.columns[COL_SIGNATURE_1_FIRST_LIMB].evaluations[0].to_u64();
        let sig2_col = trace.columns[COL_SIGNATURE_2_FIRST_LIMB].evaluations[0].to_u64();
        assert_ne!(sig1_col, sig2_col, "signature first limbs must differ");
        // First byte 0xAA in MSB position of 8-byte BE → 0xAA00_..._00.
        assert_eq!(sig1_col >> 56, 0xAA);
        assert_eq!(sig2_col >> 56, 0xBB);
    }

    #[test]
    fn block_root_distinct_fails_when_roots_equal_under_real_flag() {
        // Build a valid witness, then maliciously set block_root_2 ==
        // block_root_1 and the inverse to anything — constraint 9 must fire.
        let mut br1 = [0u8; 32];
        br1[0] = 0xAA;
        let mut br2 = [0u8; 32];
        br2[0] = 0xBB;
        let att_1 = attestation(5, 10, br1, vec![1]);
        let att_2 = attestation(6, 10, br2, vec![1]);
        let w = AttesterSlashingWitness::from_attestations(&att_1, &att_2);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: set block_root_2 byte 0 to 0xAA (matching root 1).
        cols[COL_BLOCK_ROOT_2_OFFSET][0] = Scalar::from_u64(0xAA, CurveType::Bls48581);
        let cs = AttesterSlashingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[9][0].is_zero(),
            "block_root_distinct_on_real must fire when roots tampered to be equal",
        );
    }

    #[test]
    fn intersection_zero_fails_constraint() {
        // Manually construct a witness with intersection_count = 0 and
        // verify constraint 8 fires.
        let mut br1 = [0u8; 32];
        br1[0] = 0xAA;
        let mut br2 = [0u8; 32];
        br2[0] = 0xBB;
        let att_1 = attestation(5, 10, br1, vec![1, 2]);
        let att_2 = attestation(6, 10, br2, vec![1, 2]);
        let mut w = AttesterSlashingWitness::from_attestations(&att_1, &att_2);
        // Tamper the witness: claim no intersection.
        w.intersection_count = 0;
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AttesterSlashingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[8][0].is_zero(),
            "intersection_nonzero_on_real must fire when intersection_count = 0",
        );
    }
}
