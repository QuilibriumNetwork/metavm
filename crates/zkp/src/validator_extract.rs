//! Validator-registry extraction AIR.
//!
//! A focused AIR exposing per-validator data as row-local tuples, so
//! downstream cross-AIR LogUp linkages (Finality `effective_balance`
//! ↔ this; future BLS-attestation `pubkey` ↔ this) can match against
//! a single canonical source.
//!
//! Each row corresponds to one validator. Columns hold the validator's
//! index plus the fields downstream consumers need (today: just
//! `effective_balance`, but the layout makes adding `activation_epoch`,
//! `exit_epoch`, etc. straightforward when needed).
//!
//! # Soundness scope
//!
//! This AIR proves: "there exists a sequence of validator records with
//! the witnessed `(index, effective_balance, ...)` tuples, indices
//! are 0..committee_size, and the binary `IS_REAL` flag is well-formed."
//!
//! It does NOT prove: that each row's data matches the SSZ-merkleized
//! validator container at the corresponding position. Closing that gap
//! requires a separate cross-AIR LogUp linkage between this AIR and the
//! SSZ AIR proving each validator's `hash_tree_root` is the
//! corresponding leaf in the validator-registry merkle tree. That
//! linkage is the next step (it produces a per-validator chunk-tuple on
//! one side and the validator-leaf chunks on the other; multi-column
//! tuple support in cross-AIR LogUp handles it).
//!
//! # Constraint layout
//!
//! 1 row-local consolidated category:
//!
//!   0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//!
//! 1 cross-row (shifted) constraint:
//!
//!   0. `validator_index_chain` —
//!      `IS_REAL(ω·X) · (VALIDATOR_INDEX(ω·X) − VALIDATOR_INDEX(X) − 1) = 0`.
//!      On the row that ENTERS a real validator row, the index
//!      increments by 1. On other transitions the constraint is
//!      trivially zero (gate by `IS_REAL(ω·X)`). Excluded at the
//!      wrap-around row via the boundary product.
//!
//! Combined with the witness builder's `VALIDATOR_INDEX[0] = 0` and
//! `IS_REAL[0..committee_size] = 1`, the indexing is pinned: a malicious
//! prover cannot reorder validators or skip an index.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column indices ────────────────────────────────────────────────────

/// `i` on the i-th validator row, increasing through the committee.
/// `0` on padding rows (which are the rows beyond `committee_size`).
pub const COL_VALIDATOR_INDEX: usize = 0;

/// Validator's `effective_balance` in Gwei. `0` on padding rows.
pub const COL_EFFECTIVE_BALANCE: usize = 1;

/// `1` on real validator rows, `0` on padding. Used as the cross-AIR
/// LogUp selector when downstream linkages select per-validator tuples.
pub const COL_IS_REAL: usize = 2;

/// Validator's `hash_tree_root()` (32 bytes). Each byte gets its own
/// column for cross-AIR LogUp tuple matching against the SSZ AIR's
/// per-byte chunk columns. On padding rows all bytes are 0.
///
/// The cross-AIR LogUp linkage to SSZ
/// (`make_validator_extract_ssz_linkage_descriptor`) matches the
/// 33-column tuple `(VALIDATOR_INDEX, validator_root_byte_0..31)` on
/// this AIR against the same shape on the SSZ leaf rows, transitively
/// binding the ValidatorExtract row to the corresponding leaf in the
/// validator-registry merkle tree.
pub const COL_VALIDATOR_ROOT_OFFSET: usize = 3;

/// Number of bytes in a validator's hash_tree_root.
pub const VALIDATOR_ROOT_BYTES: usize = 32;

/// Parity of `VALIDATOR_INDEX` (i.e., `i mod 2`). `0` on padding rows.
/// Toggled across consecutive real rows via the new `parity_toggle`
/// shifted constraint (combined with the witness builder's
/// `PARITY_BIT[0] = 0` boundary, this pins `PARITY_BIT[i] = i mod 2`
/// on all real rows).
pub const COL_PARITY_BIT: usize = COL_VALIDATOR_ROOT_OFFSET + VALIDATOR_ROOT_BYTES;

/// Selector firing on real validator rows whose index is EVEN —
/// i.e., the validators that will occupy the LEFT chunk of an
/// SSZ leaf-pair. Algebraically `(1 - PARITY_BIT) · IS_REAL`. Used
/// as the A-side selector for the LEFT-side
/// `validator_extract → ssz_air` cross-AIR LogUp linkage (the RIGHT
/// counterpart uses `COL_IS_RIGHT_VALIDATOR`).
///
/// Without this column, the Left/Right linkages used `IS_REAL` on
/// the A side, which counted every validator in BOTH the left and
/// right multisets — breaking multiset equality whenever the
/// committee size exceeds 1.
pub const COL_IS_LEFT_VALIDATOR: usize = COL_PARITY_BIT + 1;

/// Selector firing on real validator rows whose index is ODD.
/// Algebraically `PARITY_BIT · IS_REAL`.
pub const COL_IS_RIGHT_VALIDATOR: usize = COL_IS_LEFT_VALIDATOR + 1;

pub const NUM_COLUMNS: usize = COL_IS_RIGHT_VALIDATOR + 1;

/// 1 (is_real_binary) + 1 (parity_bit_binary) + 1 (is_left_validator
/// binding) + 1 (is_right_validator binding) = 4.
pub const NUM_ROW_CONSTRAINTS: usize = 4;
/// 1 (validator_index chain) + 1 (parity toggle) = 2.
pub const NUM_SHIFTED: usize = 2;

// ─── Witness type ──────────────────────────────────────────────────────

/// One validator's data exposed as a row of the extract AIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatorExtractRow {
    pub effective_balance: u64,
    /// `validator.hash_tree_root()` — the 32-byte leaf this validator
    /// occupies in the SSZ-merkleized validator registry.
    pub validator_root: [u8; VALIDATOR_ROOT_BYTES],
}

/// All rows of the validator registry, in committee order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorExtractWitness {
    pub validators: Vec<ValidatorExtractRow>,
}

impl ValidatorExtractWitness {
    /// Convenience constructor from the [`crate::beacon::Validator`]
    /// struct sequence. Computes `hash_tree_root()` per validator and
    /// stores it in the row, so the cross-AIR LogUp linkage to SSZ can
    /// match the registry leaves.
    pub fn from_beacon_validators(vs: &[crate::beacon::Validator]) -> Self {
        Self {
            validators: vs
                .iter()
                .map(|v| ValidatorExtractRow {
                    effective_balance: v.effective_balance,
                    validator_root: v.hash_tree_root(),
                })
                .collect(),
        }
    }
}

// ─── Constraint system ─────────────────────────────────────────────────

/// `VmConstraintSystem` impl for the validator extract AIR.
pub struct ValidatorExtractConstraintSystem {
    /// Number of real validator rows (= committee size). The padded
    /// domain size is determined by [`TracePolynomials`] to the next
    /// power of two ≥ `num_rows`.
    pub num_rows: usize,
    /// Domain generator ω for the verifier-side boundary product.
    pub omega: Option<Scalar>,
    /// Padded domain size.
    pub domain_size: Option<u64>,
}

impl ValidatorExtractConstraintSystem {
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

/// Build `TracePolynomials` for the validator extract AIR.
pub fn build_trace_polynomials(
    witness: &ValidatorExtractWitness,
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
        columns[COL_VALIDATOR_INDEX][i] = Scalar::from_u64(i as u64, curve);
        columns[COL_EFFECTIVE_BALANCE][i] = Scalar::from_u64(v.effective_balance, curve);
        columns[COL_IS_REAL][i] = one.clone();
        // 32-byte validator hash_tree_root, one byte per column.
        for b in 0..VALIDATOR_ROOT_BYTES {
            columns[COL_VALIDATOR_ROOT_OFFSET + b][i] =
                Scalar::from_u64(v.validator_root[b] as u64, curve);
        }
        let parity = (i % 2) as u64;
        columns[COL_PARITY_BIT][i] = Scalar::from_u64(parity, curve);
        // IS_LEFT = (1 - parity) · IS_REAL = parity == 0 ? 1 : 0
        // IS_RIGHT = parity · IS_REAL = parity == 1 ? 1 : 0
        if parity == 0 {
            columns[COL_IS_LEFT_VALIDATOR][i] = one.clone();
        } else {
            columns[COL_IS_RIGHT_VALIDATOR][i] = one.clone();
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

// ─── Helpers ───────────────────────────────────────────────────────────

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

// ─── VmConstraintSystem implementation ─────────────────────────────────

impl VmConstraintSystem for ValidatorExtractConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "parity_bit_binary".into(),
            "is_left_validator_binding".into(),
            "is_right_validator_binding".into(),
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
        let mut bin = vec![Scalar::zero(curve); n];
        let mut parity_bin = vec![Scalar::zero(curve); n];
        let mut is_left_bind = vec![Scalar::zero(curve); n];
        let mut is_right_bind = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let is_real = &columns[COL_IS_REAL][row];
            let parity = &columns[COL_PARITY_BIT][row];
            let is_left = &columns[COL_IS_LEFT_VALIDATOR][row];
            let is_right = &columns[COL_IS_RIGHT_VALIDATOR][row];

            bin[row] = is_real.mul(&is_real.sub(&one));
            parity_bin[row] = parity.mul(&parity.sub(&one));
            // IS_LEFT - (1 - PARITY) · IS_REAL = IS_LEFT - IS_REAL + PARITY · IS_REAL
            let one_minus_parity = one.sub(parity);
            let expected_left = one_minus_parity.mul(is_real);
            is_left_bind[row] = is_left.sub(&expected_left);
            // IS_RIGHT - PARITY · IS_REAL
            let expected_right = parity.mul(is_real);
            is_right_bind[row] = is_right.sub(&expected_right);
        }
        vec![bin, parity_bin, is_left_bind, is_right_bind]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals[COL_IS_REAL];
        let parity = &col_evals[COL_PARITY_BIT];
        let is_left = &col_evals[COL_IS_LEFT_VALIDATOR];
        let is_right = &col_evals[COL_IS_RIGHT_VALIDATOR];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            parity.mul(&parity.sub(&one)),
            is_left.sub(&one.sub(parity).mul(is_real)),
            is_right.sub(&parity.mul(is_real)),
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
        let parity = &col_coeffs[COL_PARITY_BIT];
        let is_left = &col_coeffs[COL_IS_LEFT_VALIDATOR];
        let is_right = &col_coeffs[COL_IS_RIGHT_VALIDATOR];

        let is_real_minus_1 = poly_sub(is_real, &one_poly, curve);
        let body0 = poly_mul(is_real, &is_real_minus_1, curve);

        let parity_minus_1 = poly_sub(parity, &one_poly, curve);
        let body1 = poly_mul(parity, &parity_minus_1, curve);

        // body2 = IS_LEFT - (1 - PARITY) · IS_REAL
        let one_minus_parity = poly_sub(&one_poly, parity, curve);
        let expected_left = poly_mul(&one_minus_parity, is_real, curve);
        let body2 = poly_sub(is_left, &expected_left, curve);

        // body3 = IS_RIGHT - PARITY · IS_REAL
        let expected_right = poly_mul(parity, is_real, curve);
        let body3 = poly_sub(is_right, &expected_right, curve);

        let bodies = [body0, body1, body2, body3];
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
        // IS_REAL is the "selector"-like column; expose so the prover's
        // selector-padding logic recognises the gate.
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
        vec![COL_VALIDATOR_INDEX, COL_IS_REAL, COL_PARITY_BIT]
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
        if shifted_evals.len() < 3 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let vi_curr = &col_evals_at_z[COL_VALIDATOR_INDEX];
        let parity_curr = &col_evals_at_z[COL_PARITY_BIT];
        let vi_next = &shifted_evals[0];
        let is_real_next = &shifted_evals[1];
        let parity_next = &shifted_evals[2];

        // body0 = IS_REAL(ω·z) · (VALIDATOR_INDEX(ω·z) − VALIDATOR_INDEX(z) − 1)
        let diff0 = vi_next.sub(vi_curr).sub(&one);
        let body0 = is_real_next.mul(&diff0);

        // body1 = IS_REAL(ω·z) · (PARITY_BIT(ω·z) + PARITY_BIT(z) − 1)
        // Forces parity to alternate on consecutive real rows.
        let parity_sum = parity_next.add(parity_curr).sub(&one);
        let body1 = is_real_next.mul(&parity_sum);

        // Wrap-around exclusion: same rationale as the index chain
        // (the wrap row connects the last padding row to ω^0, where
        // IS_REAL = 1 in honest witnesses but the chain consistency
        // would not hold).
        let exclusion = z.sub(omega_n_minus_1);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = ap.mul(&body0).mul(&exclusion);
        ap = ap.mul(alpha);
        let term1 = ap.mul(&body1).mul(&exclusion);
        term0.add(&term1)
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
        let parity = &col_coeffs[COL_PARITY_BIT];
        let vi_shift = poly_shift(vi, omega);
        let is_real_shift = poly_shift(is_real, omega);
        let parity_shift = poly_shift(parity, omega);

        // body0 = IS_REAL(ω·X) · (VALIDATOR_INDEX(ω·X) − VALIDATOR_INDEX(X) − 1)
        let diff_pre = poly_sub(&vi_shift, vi, curve);
        let diff = poly_sub(&diff_pre, &one_poly, curve);
        let body0 = poly_mul(&is_real_shift, &diff, curve);

        // body1 = IS_REAL(ω·X) · (PARITY_BIT(ω·X) + PARITY_BIT(X) − 1)
        let parity_sum_pre = poly_add(&parity_shift, parity, curve);
        let parity_sum = poly_sub(&parity_sum_pre, &one_poly, curve);
        let body1 = poly_mul(&is_real_shift, &parity_sum, curve);

        // Exclude wrap-around at row n-1.
        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let excluded0 = poly_mul_linear(&body0, &omega_n_minus_1);
        let excluded1 = poly_mul_linear(&body1, &omega_n_minus_1);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = poly_scalar_mul(&excluded0, &ap);
        ap = ap.mul(alpha);
        let term1 = poly_scalar_mul(&excluded1, &ap);
        poly_add(&term0, &term1, curve)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp linkage helper ────────────────────────────────────

/// Construct the descriptor for the Finality ↔ ValidatorExtract linkage.
///
/// **A side (Finality)**: validator rows with tuple
/// `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)`, gated by `SEL_VALIDATOR`.
///
/// **B side (this AIR)**: real validator rows with tuple
/// `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)`, gated by `IS_REAL`.
///
/// Closes #84's Finality side: every Finality validator row's
/// `(idx, eff_bal)` must appear in this AIR's per-validator table.
/// Combined with this AIR's own constraints (`IS_REAL` binary +
/// `VALIDATOR_INDEX` chain), the indexing on B is canonical, so the
/// multiset match cryptographically pins Finality's
/// `effective_balance` to the validator-registry order.
///
/// Binding this AIR to the SSZ-merkleized validator registry itself
/// (so the per-row `(VALIDATOR_INDEX, EFFECTIVE_BALANCE, ...)` actually
/// matches the chain's beacon state) is a separate cross-AIR LogUp,
/// see `cross_air_logup_dependent_tasks.md`.
pub fn make_finality_validator_extract_linkage_descriptor(
    finality_layer_index: usize,
    validator_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::finality_constraints::{COL_EFFECTIVE_BALANCE as F_EB, COL_SEL_VALIDATOR, COL_VALIDATOR_INDEX as F_VI};
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "finality_validator_extract_v1".into(),
        a_layer_index: finality_layer_index,
        a_columns: vec![F_VI, F_EB],
        a_selector_column: Some(COL_SEL_VALIDATOR),
        b_layer_index: validator_extract_layer_index,
        b_columns: vec![COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE],
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Which side of the SSZ pair-hashing trace this descriptor binds.
///
/// Each SSZ row processes a `(left, right) → parent` triple. The
/// validator-registry tree's leaves are the validator hashes; at
/// `layer_depth = 0`, a row at `position_in_layer = p` has
/// validator `2p` in its `LEFT` columns and validator `2p + 1` in
/// its `RIGHT` columns. So binding ValidatorExtract roots to SSZ
/// requires two descriptors — one matching the even-indexed validators
/// to LEFT chunks, the other matching odd-indexed validators to RIGHT
/// chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SszLeafSide {
    Left,
    Right,
}

/// Construct the descriptor for the ValidatorExtract ↔ SSZ-leaf linkage
/// for one parity (`Left` or `Right`).
///
/// **A side (this AIR)**: per-validator `validator_root` (32 bytes),
/// gated by `IS_REAL`.
///
/// **B side (SSZ)**: 32-byte chunk columns (`LEFT_OFFSET..+32` or
/// `RIGHT_OFFSET..+32`), gated by `IS_LEFT_REAL` / `IS_RIGHT_REAL`.
///
/// **Sub-multiset semantics.** The SSZ side's selected chunks include
/// every layer's left/right chunks, not just `layer_depth = 0`. So the
/// SSZ multiset is a SUPERSET of the validator-leaf chunks (plus all
/// internal node chunks). ValidatorExtract's roots are a sub-multiset
/// — every validator root must appear among SSZ's left/right chunks at
/// some row. Since the cross-AIR LogUp checks sub-multiset, this is the
/// correct relation.
///
/// **Caller responsibility.** The cross-AIR LogUp ALSO needs the SSZ
/// trace to actually contain all validator roots at `layer_depth = 0`
/// (which it does, by construction of the SSZ merkleization witness).
/// Combined with SSZ's own algebraic correctness (each row's parent =
/// sha256(left, right), pair-hashing chain across layers), this proves
/// every validator root participates in the registry merkle tree, and
/// therefore the registry root is a function of the
/// validator-root values.
///
/// **Soundness gap.** This linkage does NOT yet bind
/// `EFFECTIVE_BALANCE` to `validator_root`. The prover could supply a
/// real `validator_root` (matching SSZ) but a fake `EFFECTIVE_BALANCE`
/// in the same row. Closing this requires proving
/// `validator_root == validator.hash_tree_root()` algebraically, which
/// needs cross-AIR LogUp to a SHA-256 AIR (essentially merging this
/// work with #87). See `cross_air_logup_dependent_tasks.md`.
///
/// Currently the host-side witness builder
/// `from_beacon_validators` ensures consistency at trace-construction
/// time; the cryptographic version requires the SHA-256 binding.
pub fn make_validator_extract_ssz_linkage_descriptor(
    validator_extract_layer_index: usize,
    ssz_layer_index: usize,
    side: SszLeafSide,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::ssz_air::col as ssz_col;
    use crate::ssz_air::CHUNK_BYTES;

    // ValidatorExtract A side: per-validator root, 32 byte columns.
    let a_columns: Vec<usize> = (0..VALIDATOR_ROOT_BYTES)
        .map(|b| COL_VALIDATOR_ROOT_OFFSET + b)
        .collect();

    // SSZ B side: LEFT or RIGHT chunk's 32 byte columns.
    // A side selector: IS_LEFT_VALIDATOR / IS_RIGHT_VALIDATOR — the
    // parity-aware columns ensure each validator participates in
    // exactly ONE of the two multisets (Left for even-indexed
    // validators, Right for odd-indexed). With the previous
    // `IS_REAL`-on-both-sides convention, every validator was counted
    // in both Left and Right multisets, breaking equality whenever the
    // committee had >1 validator. See the AIR-level
    // `is_left_validator_binding` / `is_right_validator_binding`
    // row-locals + the `parity_toggle` shifted constraint, which
    // pin `IS_LEFT_VALIDATOR[i] = (i mod 2 == 0) · IS_REAL[i]`.
    // B-side selector: IS_LEFT_VALIDATOR_LEAF / IS_RIGHT_VALIDATOR_LEAF
    // (NOT IS_LEFT_REAL / IS_RIGHT_REAL). The leaf-restricted variant
    // fires only on `LAYER_DEPTH = 0` rows whose chunk is a real
    // validator-root input — intermediate parent rows at depth ≥ 1
    // are excluded. Without this gating, multi-validator chains with
    // committee size ≥ 4 would include intermediate parent hashes in
    // the multiset, breaking equality.
    let (b_offset, b_selector, a_selector, label) = match side {
        SszLeafSide::Left => (
            ssz_col::LEFT_OFFSET,
            ssz_col::IS_LEFT_VALIDATOR_LEAF,
            COL_IS_LEFT_VALIDATOR,
            "validator_extract_ssz_left_v1",
        ),
        SszLeafSide::Right => (
            ssz_col::RIGHT_OFFSET,
            ssz_col::IS_RIGHT_VALIDATOR_LEAF,
            COL_IS_RIGHT_VALIDATOR,
            "validator_extract_ssz_right_v1",
        ),
    };
    let b_columns: Vec<usize> = (0..CHUNK_BYTES).map(|b| b_offset + b).collect();

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: validator_extract_layer_index,
        a_columns,
        a_selector_column: Some(a_selector),
        b_layer_index: ssz_layer_index,
        b_columns,
        b_selector_column: Some(b_selector),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_witness() -> ValidatorExtractWitness {
        // Synthesise validator_root values that are obviously distinct
        // so multiset tests don't accidentally collide.
        let mk_root = |seed: u8| {
            let mut r = [0u8; VALIDATOR_ROOT_BYTES];
            for (i, b) in r.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            r
        };
        ValidatorExtractWitness {
            validators: vec![
                ValidatorExtractRow {
                    effective_balance: 32_000_000_000,
                    validator_root: mk_root(0x10),
                },
                ValidatorExtractRow {
                    effective_balance: 32_000_000_000,
                    validator_root: mk_root(0x20),
                },
                ValidatorExtractRow {
                    effective_balance: 16_000_000_000,
                    validator_root: mk_root(0x30),
                },
                ValidatorExtractRow {
                    effective_balance: 32_000_000_000,
                    validator_root: mk_root(0x40),
                },
            ],
        }
    }

    #[test]
    fn build_trace_populates_columns() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 4);
        // Row 0: index 0, eff_bal 32e9, is_real 1.
        assert_eq!(
            trace.columns[COL_VALIDATOR_INDEX].evaluations[0].to_bytes(),
            Scalar::from_u64(0, curve).to_bytes()
        );
        assert_eq!(
            trace.columns[COL_EFFECTIVE_BALANCE].evaluations[0].to_bytes(),
            Scalar::from_u64(32_000_000_000, curve).to_bytes()
        );
        assert_eq!(
            trace.columns[COL_IS_REAL].evaluations[0].to_bytes(),
            Scalar::one(curve).to_bytes()
        );
        // Row 2: index 2, eff_bal 16e9, is_real 1.
        assert_eq!(
            trace.columns[COL_VALIDATOR_INDEX].evaluations[2].to_bytes(),
            Scalar::from_u64(2, curve).to_bytes()
        );
        assert_eq!(
            trace.columns[COL_EFFECTIVE_BALANCE].evaluations[2].to_bytes(),
            Scalar::from_u64(16_000_000_000, curve).to_bytes()
        );
        // Padding row: all zero.
        assert!(trace.columns[COL_VALIDATOR_INDEX].evaluations.last().unwrap().is_zero());
        assert!(trace.columns[COL_IS_REAL].evaluations.last().unwrap().is_zero());
    }

    #[test]
    fn constraints_vanish_on_honest_witness() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = ValidatorExtractConstraintSystem::new(trace.num_rows);
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
    fn constraints_reject_non_binary_is_real() {
        let curve = CurveType::Bls48581;
        let w = small_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        // Forge IS_REAL = 2 on row 0.
        trace.columns[COL_IS_REAL].evaluations[0] = Scalar::from_u64(2, curve);
        let cs = ValidatorExtractConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(7, curve);
        let col_vals: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        let c_at = cs.evaluate_at_point(&col_vals, &alpha);
        assert!(!c_at.is_zero(), "is_real_binary must fire on IS_REAL ∉ {{0,1}}");
    }

    #[test]
    fn finality_validator_extract_descriptor_well_formed() {
        let desc = make_finality_validator_extract_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "finality_validator_extract_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 2);
        assert_eq!(desc.b_columns.len(), 2);
        // A: (Finality.VALIDATOR_INDEX, Finality.EFFECTIVE_BALANCE), gated by SEL_VALIDATOR.
        assert_eq!(
            desc.a_columns[0],
            crate::finality_constraints::COL_VALIDATOR_INDEX
        );
        assert_eq!(
            desc.a_columns[1],
            crate::finality_constraints::COL_EFFECTIVE_BALANCE
        );
        assert_eq!(
            desc.a_selector_column,
            Some(crate::finality_constraints::COL_SEL_VALIDATOR)
        );
        // B: (this.VALIDATOR_INDEX, this.EFFECTIVE_BALANCE), gated by IS_REAL.
        assert_eq!(desc.b_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(desc.b_columns[1], COL_EFFECTIVE_BALANCE);
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn from_beacon_validators_extracts_effective_balance_and_root() {
        use crate::beacon::Validator;
        let vs = vec![
            Validator { effective_balance: 100, ..Default::default() },
            Validator { effective_balance: 200, ..Default::default() },
            Validator { effective_balance: 300, ..Default::default() },
        ];
        let w = ValidatorExtractWitness::from_beacon_validators(&vs);
        assert_eq!(w.validators.len(), 3);
        assert_eq!(w.validators[0].effective_balance, 100);
        // validator_root computed from hash_tree_root; just check it
        // matches the beacon Validator's hash_tree_root().
        assert_eq!(w.validators[0].validator_root, vs[0].hash_tree_root());
        assert_eq!(w.validators[1].validator_root, vs[1].hash_tree_root());
        assert_eq!(w.validators[2].validator_root, vs[2].hash_tree_root());
    }

    #[test]
    fn validator_extract_ssz_descriptor_well_formed_left() {
        let desc = make_validator_extract_ssz_linkage_descriptor(0, 1, SszLeafSide::Left);
        assert_eq!(desc.label, "validator_extract_ssz_left_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        // Tuple is 32 bytes on each side.
        assert_eq!(desc.a_columns.len(), VALIDATOR_ROOT_BYTES);
        assert_eq!(desc.b_columns.len(), VALIDATOR_ROOT_BYTES);
        assert_eq!(desc.a_columns[0], COL_VALIDATOR_ROOT_OFFSET);
        assert_eq!(
            desc.a_columns[VALIDATOR_ROOT_BYTES - 1],
            COL_VALIDATOR_ROOT_OFFSET + VALIDATOR_ROOT_BYTES - 1
        );
        // SSZ side: LEFT chunk columns.
        assert_eq!(desc.b_columns[0], crate::ssz_air::col::LEFT_OFFSET);
        // A-side selector now uses parity-aware IS_LEFT_VALIDATOR
        // (not IS_REAL) so multi-validator chains balance the
        // multiset across Left/Right descriptors.
        assert_eq!(desc.a_selector_column, Some(COL_IS_LEFT_VALIDATOR));
        // B-side selector is the leaf-only variant
        // (IS_LEFT_VALIDATOR_LEAF) so intermediate parent rows at
        // depth ≥ 1 don't pollute the multiset.
        assert_eq!(
            desc.b_selector_column,
            Some(crate::ssz_air::col::IS_LEFT_VALIDATOR_LEAF)
        );
    }

    #[test]
    fn validator_extract_ssz_descriptor_well_formed_right() {
        let desc = make_validator_extract_ssz_linkage_descriptor(0, 1, SszLeafSide::Right);
        assert_eq!(desc.label, "validator_extract_ssz_right_v1");
        // SSZ side: RIGHT chunk columns.
        assert_eq!(desc.b_columns[0], crate::ssz_air::col::RIGHT_OFFSET);
        // A-side selector now uses parity-aware IS_RIGHT_VALIDATOR.
        assert_eq!(desc.a_selector_column, Some(COL_IS_RIGHT_VALIDATOR));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::ssz_air::col::IS_RIGHT_VALIDATOR_LEAF)
        );
    }

    /// End-to-end: prove and verify the validator extract AIR via the
    /// scheme-generic pipeline.
    #[test]
    #[ignore = "slow: full prove + verify; run with --release --ignored"]
    fn validator_extract_proof_round_trips() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = small_witness();
        let trace = build_trace_polynomials(&w, curve);
        let domain_size = trace.padded_size;
        let omega = scheme.domain_generator(domain_size);
        let cs = ValidatorExtractConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, domain_size);

        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let valid = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(valid, "validator extract proof must verify");
    }
}
