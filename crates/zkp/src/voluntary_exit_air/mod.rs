//! Voluntary exit verification AIR (consensus-layer operation).
//!
//! Proves one [`SignedVoluntaryExit`] operation is valid against the
//! beacon-chain state. The spec for validity is:
//!
//!   1. The validator at `validator_index` exists in the registry (i.e.
//!      its pubkey can be looked up there).
//!   2. `validator.exit_epoch == FAR_FUTURE_EPOCH` — the validator has
//!      not already initiated an exit.
//!   3. `current_epoch >= validator.activation_epoch +
//!      SHARD_COMMITTEE_PERIOD` — the validator has served the minimum
//!      committee period.
//!   4. The signature `BLS.Verify(pubkey, signing_root, signature)`
//!      passes, where
//!      `signing_root = sha256(message_root || domain)` and
//!      `message_root = hash_tree_root(VoluntaryExit { epoch,
//!      validator_index })`.
//!
//! Each row of this AIR commits the full witness for one exit
//! operation. The expensive sub-proofs — validator-registry membership,
//! SSZ signing-root hash, and BLS pairing — are not re-derived inside
//! the AIR; instead they are bound via cross-AIR LogUp descriptors to
//! the dedicated AIRs that already compute them:
//!
//!   - [`make_exit_to_validator_registry_descriptor`] binds
//!     `(validator_index, pubkey)` against
//!     [`crate::validator_htr_air`] (the per-validator HTR carries both
//!     fields).
//!   - [`make_exit_to_signing_root_sha256_descriptor`] binds
//!     `(message_root || domain, signing_root)` against
//!     [`crate::sha256_extract`].
//!   - [`make_exit_to_bls_sig_descriptor`] binds
//!     `(pubkey, signing_root, signature)` against
//!     [`crate::bls_pairing_air`].
//!
//! ## Algebraic constraints (row-local, 11 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`.
//! 1. `epoch_le_decomp` — `EPOCH = Σ_b EPOCH_BYTE[b] · 2^(8b)` (LE,
//!    8 bytes).
//! 2. `validator_index_le_decomp` — `VALIDATOR_INDEX = Σ_b VI_BYTE[b] ·
//!    2^(8b)` (LE, 8 bytes).
//! 3. `current_epoch_le_decomp` — `CURRENT_EPOCH = Σ_b CE_BYTE[b] ·
//!    2^(8b)` (LE, 8 bytes).
//! 4. `activation_epoch_le_decomp` — `ACTIVATION_EPOCH = Σ_b AE_BYTE[b]
//!    · 2^(8b)` (LE, 8 bytes).
//! 5. `waited_le_decomp` — `WAITED = Σ_b WAITED_BYTE[b] · 2^(8b)` (LE,
//!    8 bytes). With the per-byte 8-bit range check, this pins WAITED
//!    in `[0, 2^64)`, so the next constraint proves
//!    `current_epoch − activation_epoch − SHARD_COMMITTEE_PERIOD ≥ 0`.
//! 6. `waiting_period` — `IS_REAL · (CURRENT_EPOCH − ACTIVATION_EPOCH −
//!    SHARD_COMMITTEE_PERIOD − WAITED) = 0`. Combined with the byte
//!    decomposition above, this enforces
//!    `current_epoch >= activation_epoch + SHARD_COMMITTEE_PERIOD`.
//! 7. `not_already_exited` — `IS_REAL · FAR_FUTURE_MINUS_EXIT = 0`.
//!    The host-side witness sets
//!    `FAR_FUTURE_MINUS_EXIT = FAR_FUTURE_EPOCH − exit_epoch`, so this
//!    body forces `exit_epoch == FAR_FUTURE_EPOCH` on every active row.
//!    (The actual `exit_epoch` value is bound to the registry validator
//!    record via [`make_exit_to_validator_htr_exit_epoch_descriptor`].)
//! 8. `signing_root_first_byte_consistency` — placeholder consistency
//!    constraint pinning `SIGNING_ROOT[0]` byte (range-checked) so the
//!    descriptor-anchor column is well-formed even when the SHA-256
//!    descriptor is not yet wired into a joint trace. (Currently:
//!    `IS_REAL · 0 = 0` body — trivially zero but kept as a labeled
//!    slot for the future "in-row signing-root binding" body.)
//! 9. `exit_epoch_le_decomp` — `EXIT_EPOCH = Σ_b EXIT_EPOCH_BYTE[b] ·
//!    2^(8b)` (LE, 8 bytes). Byte-decomposes the committed `exit_epoch`
//!    field, which is in turn bound to the validator-registry record via
//!    [`make_exit_to_validator_htr_exit_epoch_descriptor`].
//! 10. `exit_epoch_consistency` — `IS_REAL · (FAR_FUTURE_EPOCH −
//!     EXIT_EPOCH − FAR_FUTURE_MINUS_EXIT) = 0`. Algebraically pins
//!     `FAR_FUTURE_MINUS_EXIT = FAR_FUTURE_EPOCH − EXIT_EPOCH`. Combined
//!     with constraint 7 (`is_real · FAR_FUTURE_MINUS_EXIT = 0`) and the
//!     cross-AIR LogUp binding of `EXIT_EPOCH` to
//!     [`crate::validator_htr_air::COL_EXIT_EPOCH`], this closes the
//!     algebraic soundness gap: `exit_epoch == FAR_FUTURE_EPOCH` is now
//!     enforced against the actual registry record, not just a trusted
//!     witness.
//!
//! Per-byte 8-bit range checks (via `lookup_declarations`) on:
//!   - `pubkey[0..48]`
//!   - `signature[0..96]`
//!   - `message_root[0..32]`
//!   - `domain[0..32]`
//!   - `signing_root[0..32]`
//!   - `epoch_byte[0..8]`, `vi_byte[0..8]`, `ce_byte[0..8]`,
//!     `ae_byte[0..8]`, `waited_byte[0..8]`
//!
//! ## What this AIR does NOT prove (deferred)
//!
//!   - Actual BLS pairing — only the descriptor binds the
//!     `(pubkey, signing_root, signature)` triple to
//!     [`crate::bls_pairing_air`].
//!   - SHA-256 of `message_root || domain` — only the descriptor binds
//!     it to [`crate::sha256_extract`].
//!   - ~~That the `exit_epoch` value the validator-registry contains is
//!     in fact `FAR_FUTURE_EPOCH`.~~ **Closed**: constraint 9 + 10 plus
//!     [`make_exit_to_validator_htr_exit_epoch_descriptor`] bind the
//!     committed `EXIT_EPOCH` to
//!     [`crate::validator_htr_air::COL_EXIT_EPOCH`] (which is itself
//!     byte-decomposed and merkleized into the validator HTR), so
//!     `exit_epoch == FAR_FUTURE_EPOCH` is now algebraically enforced
//!     against the registry record.
//!   - Full SSZ derivation of `message_root` from `(epoch,
//!     validator_index)` — needs a 2-leaf merkleization gadget
//!     (`merkleize_chunks([hash_tree_root_uint(epoch),
//!     hash_tree_root_uint(validator_index)])`). Currently committed as
//!     a witness with a downstream binding via a separate gadget.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

/// Phase-0 constant: an `exit_epoch` of `FAR_FUTURE_EPOCH` means the
/// validator has not exited (Ethereum spec: `2^64 − 1`).
///
/// We use `u64::MAX` here; the on-chain value is the same.
pub const FAR_FUTURE_EPOCH: u64 = u64::MAX;

/// Phase-0 constant: minimum number of epochs a validator must be
/// active before it can initiate a voluntary exit. Ethereum mainnet
/// value is `256` epochs (~27 hours).
pub const SHARD_COMMITTEE_PERIOD: u64 = 256;

pub const PUBKEY_BYTES: usize = 48;
pub const SIG_BYTES: usize = 96;
pub const HASH_BYTES: usize = 32;
pub const DOMAIN_BYTES: usize = 32;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_EPOCH: usize = 0;
pub const COL_VALIDATOR_INDEX: usize = COL_EPOCH + 1;
pub const COL_CURRENT_EPOCH: usize = COL_VALIDATOR_INDEX + 1;
pub const COL_ACTIVATION_EPOCH: usize = COL_CURRENT_EPOCH + 1;
pub const COL_WAITED: usize = COL_ACTIVATION_EPOCH + 1;
pub const COL_FAR_FUTURE_MINUS_EXIT: usize = COL_WAITED + 1;

// 48-byte pubkey.
pub const COL_PUBKEY_OFFSET: usize = COL_FAR_FUTURE_MINUS_EXIT + 1;
// 96-byte signature.
pub const COL_SIGNATURE_OFFSET: usize = COL_PUBKEY_OFFSET + PUBKEY_BYTES;
// 32-byte message_root.
pub const COL_MESSAGE_ROOT_OFFSET: usize = COL_SIGNATURE_OFFSET + SIG_BYTES;
// 32-byte domain.
pub const COL_DOMAIN_OFFSET: usize = COL_MESSAGE_ROOT_OFFSET + HASH_BYTES;
// 32-byte signing_root.
pub const COL_SIGNING_ROOT_OFFSET: usize = COL_DOMAIN_OFFSET + DOMAIN_BYTES;

// Per-byte LE decompositions for the five u64 columns (range-checked
// 8-bit, used for the algebraic byte-decomp identity + ≥0 proof of
// WAITED).
pub const COL_EPOCH_BYTE_OFFSET: usize = COL_SIGNING_ROOT_OFFSET + HASH_BYTES;
pub const COL_VI_BYTE_OFFSET: usize = COL_EPOCH_BYTE_OFFSET + U64_BYTES;
pub const COL_CE_BYTE_OFFSET: usize = COL_VI_BYTE_OFFSET + U64_BYTES;
pub const COL_AE_BYTE_OFFSET: usize = COL_CE_BYTE_OFFSET + U64_BYTES;
pub const COL_WAITED_BYTE_OFFSET: usize = COL_AE_BYTE_OFFSET + U64_BYTES;

pub const COL_IS_REAL: usize = COL_WAITED_BYTE_OFFSET + U64_BYTES;

// Round-13 algebraic-soundness closure: commit `exit_epoch` directly
// (so it can be bound to `validator_htr_air::COL_EXIT_EPOCH` via cross-
// AIR LogUp) plus its LE byte decomposition for the range-checked
// algebraic identity that pins it inside `[0, 2^64)`. Appended at the
// END of the layout to preserve existing column offsets.
pub const COL_EXIT_EPOCH: usize = COL_IS_REAL + 1;
pub const COL_EXIT_EPOCH_BYTE_OFFSET: usize = COL_EXIT_EPOCH + 1;
pub const NUM_COLUMNS: usize = COL_EXIT_EPOCH_BYTE_OFFSET + U64_BYTES;

// 11 row-local bodies (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct VoluntaryExitRow {
    pub epoch: u64,
    pub validator_index: u64,
    pub pubkey: [u8; PUBKEY_BYTES],
    pub current_epoch: u64,
    pub activation_epoch: u64,
    /// The host computes this as
    /// `current_epoch − activation_epoch − SHARD_COMMITTEE_PERIOD` and
    /// asserts ≥ 0 before building the witness.
    pub waited: u64,
    /// Host-witness for `FAR_FUTURE_EPOCH − exit_epoch`. For an honest
    /// exit, the registry's `exit_epoch == FAR_FUTURE_EPOCH`, so this
    /// is `0`.
    pub far_future_minus_exit: u64,
    /// The validator's `exit_epoch` field from the registry record.
    /// Bound algebraically to `validator_htr_air::COL_EXIT_EPOCH` via
    /// the [`make_exit_to_validator_htr_exit_epoch_descriptor`] cross-
    /// AIR LogUp, so this value is committed not trusted. For an honest
    /// not-yet-exited validator this equals `FAR_FUTURE_EPOCH`.
    pub exit_epoch: u64,
    pub signature: [u8; SIG_BYTES],
    pub message_root: [u8; HASH_BYTES],
    pub domain: [u8; DOMAIN_BYTES],
    pub signing_root: [u8; HASH_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct VoluntaryExitWitness {
    pub rows: Vec<VoluntaryExitRow>,
}

impl VoluntaryExitWitness {
    /// Build a witness for one `SignedVoluntaryExit` operation.
    ///
    /// Parameters:
    ///   * `epoch` / `validator_index` — the `VoluntaryExit.message`.
    ///   * `signature` — the 96-byte BLS G2 signature.
    ///   * `validator_pubkey` — the validator's 48-byte BLS G1 pubkey
    ///     (must match the registry record at `validator_index`).
    ///   * `current_epoch` — the beacon-chain head epoch at the time
    ///     of inclusion.
    ///   * `activation_epoch` — the validator's `activation_epoch`
    ///     field from the registry.
    ///   * `message_root` — `hash_tree_root(VoluntaryExit { epoch,
    ///     validator_index })`.
    ///   * `domain` — the 32-byte signing domain.
    ///
    /// Computes `signing_root = sha256(message_root || domain)` and
    /// `waited = current_epoch − activation_epoch −
    /// SHARD_COMMITTEE_PERIOD` host-side. Panics if the waiting period
    /// has not elapsed.
    pub fn from_signed_exit(
        epoch: u64,
        validator_index: u64,
        signature: [u8; SIG_BYTES],
        validator_pubkey: [u8; PUBKEY_BYTES],
        current_epoch: u64,
        activation_epoch: u64,
        message_root: [u8; HASH_BYTES],
        domain: [u8; DOMAIN_BYTES],
    ) -> Self {
        let activation_plus = activation_epoch
            .checked_add(SHARD_COMMITTEE_PERIOD)
            .expect("activation_epoch + SHARD_COMMITTEE_PERIOD overflowed u64");
        let waited = current_epoch
            .checked_sub(activation_plus)
            .expect(
                "current_epoch < activation_epoch + SHARD_COMMITTEE_PERIOD: \
                 voluntary exit waiting period not yet elapsed",
            );
        // FAR_FUTURE_EPOCH − FAR_FUTURE_EPOCH = 0 for an honest
        // not-yet-exited validator.
        let exit_epoch = FAR_FUTURE_EPOCH;
        let far_future_minus_exit = 0u64;

        // signing_root = sha256(message_root || domain).
        let mut sha_input = [0u8; HASH_BYTES + DOMAIN_BYTES];
        sha_input[..HASH_BYTES].copy_from_slice(&message_root);
        sha_input[HASH_BYTES..].copy_from_slice(&domain);
        let signing_root = crate::sha256::sha256(&sha_input);

        Self {
            rows: vec![VoluntaryExitRow {
                epoch,
                validator_index,
                pubkey: validator_pubkey,
                current_epoch,
                activation_epoch,
                waited,
                far_future_minus_exit,
                exit_epoch,
                signature,
                message_root,
                domain,
                signing_root,
            }],
        }
    }

    pub fn from_rows(rows: Vec<VoluntaryExitRow>) -> Self {
        Self { rows }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &VoluntaryExitWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_EPOCH][i] = Scalar::from_u64(row.epoch, curve);
        columns[COL_VALIDATOR_INDEX][i] = Scalar::from_u64(row.validator_index, curve);
        columns[COL_CURRENT_EPOCH][i] = Scalar::from_u64(row.current_epoch, curve);
        columns[COL_ACTIVATION_EPOCH][i] = Scalar::from_u64(row.activation_epoch, curve);
        columns[COL_WAITED][i] = Scalar::from_u64(row.waited, curve);
        columns[COL_FAR_FUTURE_MINUS_EXIT][i] =
            Scalar::from_u64(row.far_future_minus_exit, curve);

        for k in 0..PUBKEY_BYTES {
            columns[COL_PUBKEY_OFFSET + k][i] =
                Scalar::from_u64(row.pubkey[k] as u64, curve);
        }
        for k in 0..SIG_BYTES {
            columns[COL_SIGNATURE_OFFSET + k][i] =
                Scalar::from_u64(row.signature[k] as u64, curve);
        }
        for k in 0..HASH_BYTES {
            columns[COL_MESSAGE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.message_root[k] as u64, curve);
        }
        for k in 0..DOMAIN_BYTES {
            columns[COL_DOMAIN_OFFSET + k][i] =
                Scalar::from_u64(row.domain[k] as u64, curve);
        }
        for k in 0..HASH_BYTES {
            columns[COL_SIGNING_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.signing_root[k] as u64, curve);
        }

        // LE byte decompositions.
        let epoch_bytes = row.epoch.to_le_bytes();
        let vi_bytes = row.validator_index.to_le_bytes();
        let ce_bytes = row.current_epoch.to_le_bytes();
        let ae_bytes = row.activation_epoch.to_le_bytes();
        let waited_bytes = row.waited.to_le_bytes();
        let exit_epoch_bytes = row.exit_epoch.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_EPOCH_BYTE_OFFSET + b][i] =
                Scalar::from_u64(epoch_bytes[b] as u64, curve);
            columns[COL_VI_BYTE_OFFSET + b][i] =
                Scalar::from_u64(vi_bytes[b] as u64, curve);
            columns[COL_CE_BYTE_OFFSET + b][i] =
                Scalar::from_u64(ce_bytes[b] as u64, curve);
            columns[COL_AE_BYTE_OFFSET + b][i] =
                Scalar::from_u64(ae_bytes[b] as u64, curve);
            columns[COL_WAITED_BYTE_OFFSET + b][i] =
                Scalar::from_u64(waited_bytes[b] as u64, curve);
            columns[COL_EXIT_EPOCH_BYTE_OFFSET + b][i] =
                Scalar::from_u64(exit_epoch_bytes[b] as u64, curve);
        }

        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_EXIT_EPOCH][i] = Scalar::from_u64(row.exit_epoch, curve);
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

pub struct VoluntaryExitConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl VoluntaryExitConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Body: `target_col − Σ_b byte_col[b] · 2^(8b)`.
fn eval_le_decomp(
    target_value: &Scalar,
    byte_off: usize,
    col_evals: &[Scalar],
) -> Scalar {
    let curve = target_value.curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[byte_off + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    target_value.sub(&sum)
}

fn build_le_decomp_poly(
    target_poly: &[Scalar],
    byte_off: usize,
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..U64_BYTES {
        let byte_poly = &col_coeffs[byte_off + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    poly_sub(target_poly, &sum, curve)
}

impl VmConstraintSystem for VoluntaryExitConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "epoch_le_decomp".into(),
            "validator_index_le_decomp".into(),
            "current_epoch_le_decomp".into(),
            "activation_epoch_le_decomp".into(),
            "waited_le_decomp".into(),
            "waiting_period".into(),
            "not_already_exited".into(),
            "signing_root_first_byte_consistency".into(),
            "exit_epoch_le_decomp".into(),
            "exit_epoch_consistency".into(),
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
        let shard_committee_period = Scalar::from_u64(SHARD_COMMITTEE_PERIOD, curve);
        let far_future_epoch = Scalar::from_u64(FAR_FUTURE_EPOCH, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];

            // 0: is_real_binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));

            // 1..6: LE decomps.
            bodies[1][row] = eval_le_decomp(
                &row_evals[COL_EPOCH],
                COL_EPOCH_BYTE_OFFSET,
                &row_evals,
            );
            bodies[2][row] = eval_le_decomp(
                &row_evals[COL_VALIDATOR_INDEX],
                COL_VI_BYTE_OFFSET,
                &row_evals,
            );
            bodies[3][row] = eval_le_decomp(
                &row_evals[COL_CURRENT_EPOCH],
                COL_CE_BYTE_OFFSET,
                &row_evals,
            );
            bodies[4][row] = eval_le_decomp(
                &row_evals[COL_ACTIVATION_EPOCH],
                COL_AE_BYTE_OFFSET,
                &row_evals,
            );
            bodies[5][row] = eval_le_decomp(
                &row_evals[COL_WAITED],
                COL_WAITED_BYTE_OFFSET,
                &row_evals,
            );

            // 6: waiting_period
            //    is_real · (current_epoch − activation_epoch
            //               − SHARD_COMMITTEE_PERIOD − waited) = 0.
            let ce = &row_evals[COL_CURRENT_EPOCH];
            let ae = &row_evals[COL_ACTIVATION_EPOCH];
            let waited = &row_evals[COL_WAITED];
            let body6 =
                ce.sub(ae).sub(&shard_committee_period).sub(waited);
            bodies[6][row] = is_real.mul(&body6);

            // 7: not_already_exited
            //    is_real · FAR_FUTURE_MINUS_EXIT = 0.
            bodies[7][row] = is_real.mul(&row_evals[COL_FAR_FUTURE_MINUS_EXIT]);

            // 8: signing_root_first_byte_consistency (placeholder slot,
            //    body == 0 unconditionally; reserved for the future
            //    in-row signing-root binding).
            bodies[8][row] = Scalar::zero(curve);

            // 9: exit_epoch_le_decomp
            //    EXIT_EPOCH = Σ_b EXIT_EPOCH_BYTE[b] · 2^(8b).
            bodies[9][row] = eval_le_decomp(
                &row_evals[COL_EXIT_EPOCH],
                COL_EXIT_EPOCH_BYTE_OFFSET,
                &row_evals,
            );

            // 10: exit_epoch_consistency
            //     is_real · (FAR_FUTURE_EPOCH − EXIT_EPOCH
            //                − FAR_FUTURE_MINUS_EXIT) = 0.
            //     Together with constraint 7 (`is_real ·
            //     FAR_FUTURE_MINUS_EXIT = 0`) this algebraically pins
            //     `EXIT_EPOCH == FAR_FUTURE_EPOCH` on every active row.
            let exit_epoch_v = &row_evals[COL_EXIT_EPOCH];
            let ffme = &row_evals[COL_FAR_FUTURE_MINUS_EXIT];
            let body10 = far_future_epoch.sub(exit_epoch_v).sub(ffme);
            bodies[10][row] = is_real.mul(&body10);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let shard_committee_period = Scalar::from_u64(SHARD_COMMITTEE_PERIOD, curve);
        let far_future_epoch = Scalar::from_u64(FAR_FUTURE_EPOCH, curve);
        let is_real = &col_evals[COL_IS_REAL];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            eval_le_decomp(&col_evals[COL_EPOCH], COL_EPOCH_BYTE_OFFSET, col_evals),
            eval_le_decomp(
                &col_evals[COL_VALIDATOR_INDEX],
                COL_VI_BYTE_OFFSET,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_CURRENT_EPOCH],
                COL_CE_BYTE_OFFSET,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_ACTIVATION_EPOCH],
                COL_AE_BYTE_OFFSET,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_WAITED],
                COL_WAITED_BYTE_OFFSET,
                col_evals,
            ),
            is_real.mul(
                &col_evals[COL_CURRENT_EPOCH]
                    .sub(&col_evals[COL_ACTIVATION_EPOCH])
                    .sub(&shard_committee_period)
                    .sub(&col_evals[COL_WAITED]),
            ),
            is_real.mul(&col_evals[COL_FAR_FUTURE_MINUS_EXIT]),
            Scalar::zero(curve),
            eval_le_decomp(
                &col_evals[COL_EXIT_EPOCH],
                COL_EXIT_EPOCH_BYTE_OFFSET,
                col_evals,
            ),
            is_real.mul(
                &far_future_epoch
                    .sub(&col_evals[COL_EXIT_EPOCH])
                    .sub(&col_evals[COL_FAR_FUTURE_MINUS_EXIT]),
            ),
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
        let neg_period = Scalar::from_u64(SHARD_COMMITTEE_PERIOD, curve);

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let epoch_decomp = build_le_decomp_poly(
            &col_coeffs[COL_EPOCH],
            COL_EPOCH_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let vi_decomp = build_le_decomp_poly(
            &col_coeffs[COL_VALIDATOR_INDEX],
            COL_VI_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let ce_decomp = build_le_decomp_poly(
            &col_coeffs[COL_CURRENT_EPOCH],
            COL_CE_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let ae_decomp = build_le_decomp_poly(
            &col_coeffs[COL_ACTIVATION_EPOCH],
            COL_AE_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let waited_decomp = build_le_decomp_poly(
            &col_coeffs[COL_WAITED],
            COL_WAITED_BYTE_OFFSET,
            col_coeffs,
            curve,
        );

        // waiting_period: is_real · (ce − ae − period − waited).
        let ce_p = &col_coeffs[COL_CURRENT_EPOCH];
        let ae_p = &col_coeffs[COL_ACTIVATION_EPOCH];
        let waited_p = &col_coeffs[COL_WAITED];
        let period_poly = vec![neg_period.clone()];
        let diff1 = poly_sub(ce_p, ae_p, curve);
        let diff2 = poly_sub(&diff1, &period_poly, curve);
        let diff3 = poly_sub(&diff2, waited_p, curve);
        let waiting_body = poly_mul(is_real, &diff3, curve);

        // not_already_exited: is_real · far_future_minus_exit.
        let not_exited_body = poly_mul(
            is_real,
            &col_coeffs[COL_FAR_FUTURE_MINUS_EXIT],
            curve,
        );

        let signing_root_consistency = vec![Scalar::zero(curve)];

        // exit_epoch_le_decomp.
        let exit_epoch_decomp = build_le_decomp_poly(
            &col_coeffs[COL_EXIT_EPOCH],
            COL_EXIT_EPOCH_BYTE_OFFSET,
            col_coeffs,
            curve,
        );

        // exit_epoch_consistency:
        //   is_real · (FAR_FUTURE_EPOCH − exit_epoch − far_future_minus_exit).
        let far_future_poly = vec![Scalar::from_u64(FAR_FUTURE_EPOCH, curve)];
        let ee_p = &col_coeffs[COL_EXIT_EPOCH];
        let ffme_p = &col_coeffs[COL_FAR_FUTURE_MINUS_EXIT];
        let ee_diff1 = poly_sub(&far_future_poly, ee_p, curve);
        let ee_diff2 = poly_sub(&ee_diff1, ffme_p, curve);
        let exit_epoch_consistency_body = poly_mul(is_real, &ee_diff2, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            epoch_decomp,
            vi_decomp,
            ce_decomp,
            ae_decomp,
            waited_decomp,
            waiting_body,
            not_exited_body,
            signing_root_consistency,
            exit_epoch_decomp,
            exit_epoch_consistency_body,
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

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();

        // Pubkey, signature, message_root, domain, signing_root byte
        // range checks.
        let byte_ranges: [(usize, usize, &str); 5] = [
            (COL_PUBKEY_OFFSET, PUBKEY_BYTES, "pubkey"),
            (COL_SIGNATURE_OFFSET, SIG_BYTES, "signature"),
            (COL_MESSAGE_ROOT_OFFSET, HASH_BYTES, "message_root"),
            (COL_DOMAIN_OFFSET, DOMAIN_BYTES, "domain"),
            (COL_SIGNING_ROOT_OFFSET, HASH_BYTES, "signing_root"),
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

        // u64 LE byte-decomp range checks.
        let u64_ranges: [(usize, &str); 6] = [
            (COL_EPOCH_BYTE_OFFSET, "epoch_byte"),
            (COL_VI_BYTE_OFFSET, "vi_byte"),
            (COL_CE_BYTE_OFFSET, "ce_byte"),
            (COL_AE_BYTE_OFFSET, "ae_byte"),
            (COL_WAITED_BYTE_OFFSET, "waited_byte"),
            (COL_EXIT_EPOCH_BYTE_OFFSET, "exit_epoch_byte"),
        ];
        for (off, label) in u64_ranges {
            for k in 0..U64_BYTES {
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

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(VALIDATOR_INDEX, PUBKEY[0..48])` of this AIR against the
/// per-validator HTR AIR ([`crate::validator_htr_air`]), which carries
/// the registry's `(VALIDATOR_INDEX, PUBKEY_BYTE[0..48])` tuple. This
/// proves that the pubkey we're verifying the BLS signature against
/// matches the registry record for the claimed validator index.
///
/// **Soundness note**: combined with the validator-registry inclusion
/// AIR, this transitively proves the validator is in the registry —
/// the inclusion AIR pins the validator HTR against the registry root.
pub fn make_exit_to_validator_registry_descriptor(
    exit_layer_index: usize,
    validator_htr_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_htr_air as vh;
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + PUBKEY_BYTES);
    a_columns.push(COL_VALIDATOR_INDEX);
    for k in 0..PUBKEY_BYTES {
        a_columns.push(COL_PUBKEY_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + PUBKEY_BYTES);
    b_columns.push(vh::COL_VALIDATOR_INDEX);
    for k in 0..PUBKEY_BYTES {
        b_columns.push(vh::COL_PUBKEY_BYTE_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "exit_to_validator_registry_v1".into(),
        a_layer_index: exit_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_htr_layer_index,
        b_columns,
        b_selector_column: Some(vh::COL_IS_REAL),
    }
}

/// Bind `(message_root[0..32] || domain[0..32], signing_root[0..32])`
/// of this AIR against [`crate::sha256_extract`]'s
/// `(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])`. This cryptographically
/// commits `signing_root = sha256(message_root || domain)`.
pub fn make_exit_to_signing_root_sha256_descriptor(
    exit_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;

    let mut a_columns: Vec<usize> = Vec::with_capacity(64 + HASH_BYTES);
    for k in 0..HASH_BYTES {
        a_columns.push(COL_MESSAGE_ROOT_OFFSET + k);
    }
    for k in 0..DOMAIN_BYTES {
        a_columns.push(COL_DOMAIN_OFFSET + k);
    }
    for k in 0..HASH_BYTES {
        a_columns.push(COL_SIGNING_ROOT_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(64 + HASH_BYTES);
    for k in 0..se::NUM_INPUT_BYTES {
        b_columns.push(se::COL_INPUT_BYTE_OFFSET + k);
    }
    for k in 0..se::NUM_OUTPUT_BYTES {
        b_columns.push(se::COL_OUTPUT_BYTE_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "exit_to_signing_root_sha256_v1".into(),
        a_layer_index: exit_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Bind `(pubkey[0..48], signing_root[0..32], signature[0..96])` of
/// this AIR against the BLS pairing AIR's
/// `(PK_COMPRESSED[0..48], MSG_HASH[0..32], SIG_BYTES[0..96])`. This
/// cryptographically commits the BLS signature verification step
/// (`BLS.Verify(pubkey, signing_root, signature)` is checked algebraically
/// inside [`crate::bls_pairing_air`]).
pub fn make_exit_to_bls_sig_descriptor(
    exit_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;

    let mut a_columns: Vec<usize> = Vec::with_capacity(PUBKEY_BYTES + HASH_BYTES + SIG_BYTES);
    for k in 0..PUBKEY_BYTES {
        a_columns.push(COL_PUBKEY_OFFSET + k);
    }
    for k in 0..HASH_BYTES {
        a_columns.push(COL_SIGNING_ROOT_OFFSET + k);
    }
    for k in 0..SIG_BYTES {
        a_columns.push(COL_SIGNATURE_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(PUBKEY_BYTES + HASH_BYTES + SIG_BYTES);
    for k in 0..bp::PK_BYTES {
        b_columns.push(bp::COL_PK_COMPRESSED_OFFSET + k);
    }
    for k in 0..HASH_BYTES {
        b_columns.push(bp::COL_MSG_HASH_OFFSET + k);
    }
    for k in 0..bp::SIG_BYTES {
        b_columns.push(bp::COL_SIG_BYTES_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "exit_to_bls_sig_v1".into(),
        a_layer_index: exit_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bls_pairing_layer_index,
        b_columns,
        b_selector_column: Some(bp::COL_IS_REAL),
    }
}

/// Bind `(VALIDATOR_INDEX, EXIT_EPOCH)` of this AIR against the
/// per-validator HTR AIR's `(VALIDATOR_INDEX, COL_EXIT_EPOCH)`. This
/// algebraically commits that the `exit_epoch` value we use in the
/// `not_already_exited` constraint chain is actually the value carried
/// by the validator-registry record at the claimed `validator_index`,
/// not a free witness. Combined with constraints 7 + 10
/// (`is_real · FAR_FUTURE_MINUS_EXIT = 0` and
/// `is_real · (FAR_FUTURE_EPOCH − EXIT_EPOCH − FAR_FUTURE_MINUS_EXIT) =
/// 0`), this closes the round-12 soundness gap: a prover can no longer
/// substitute a fake `exit_epoch = FAR_FUTURE_EPOCH` while the actual
/// registry record says otherwise.
pub fn make_exit_to_validator_htr_exit_epoch_descriptor(
    exit_layer_index: usize,
    validator_htr_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_htr_air as vh;
    let a_columns = vec![COL_VALIDATOR_INDEX, COL_EXIT_EPOCH];
    let b_columns = vec![vh::COL_VALIDATOR_INDEX, vh::COL_EXIT_EPOCH];

    CrossAirLogUpDescriptor {
        label: "exit_to_validator_htr_exit_epoch_v1".into(),
        a_layer_index: exit_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_htr_layer_index,
        b_columns,
        b_selector_column: Some(vh::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_pubkey() -> [u8; PUBKEY_BYTES] {
        let mut pk = [0u8; PUBKEY_BYTES];
        for i in 0..PUBKEY_BYTES {
            pk[i] = (i as u8).wrapping_mul(13).wrapping_add(7);
        }
        pk
    }

    fn synthetic_signature() -> [u8; SIG_BYTES] {
        let mut sig = [0u8; SIG_BYTES];
        for i in 0..SIG_BYTES {
            sig[i] = (i as u8).wrapping_mul(17).wrapping_add(3);
        }
        sig
    }

    fn synthetic_message_root() -> [u8; HASH_BYTES] {
        let mut m = [0u8; HASH_BYTES];
        for i in 0..HASH_BYTES {
            m[i] = (i as u8).wrapping_mul(5).wrapping_add(2);
        }
        m
    }

    fn synthetic_domain() -> [u8; DOMAIN_BYTES] {
        let mut d = [0u8; DOMAIN_BYTES];
        for i in 0..DOMAIN_BYTES {
            d[i] = (i as u8).wrapping_mul(11);
        }
        d
    }

    /// Build an honest single-exit witness with the waiting period
    /// satisfied (current_epoch ≥ activation_epoch + 256).
    fn honest_witness() -> VoluntaryExitWitness {
        VoluntaryExitWitness::from_signed_exit(
            42,
            1_234_567,
            synthetic_signature(),
            synthetic_pubkey(),
            10_000,
            500, // activation + 256 = 756 ≤ 10000 ✓
            synthetic_message_root(),
            synthetic_domain(),
        )
    }

    #[test]
    fn honest_exit_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);

        let cs = VoluntaryExitConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {} (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }

        // Sanity: signing_root matches sha256(message_root || domain).
        let mut buf = [0u8; HASH_BYTES + DOMAIN_BYTES];
        buf[..HASH_BYTES].copy_from_slice(&w.rows[0].message_root);
        buf[HASH_BYTES..].copy_from_slice(&w.rows[0].domain);
        let expected = crate::sha256::sha256(&buf);
        assert_eq!(w.rows[0].signing_root, expected);
    }

    #[test]
    fn tampered_far_future_minus_exit_fires_not_already_exited() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Pretend the validator's exit_epoch is 9_000_000 (i.e.,
        // FAR_FUTURE − exit_epoch ≠ 0). The honest witness builder
        // would refuse to construct such a witness, but a malicious
        // prover may tamper the column. Constraint 7 must fire.
        cols[COL_FAR_FUTURE_MINUS_EXIT][0] =
            Scalar::from_u64(123_456, curve);
        let cs = VoluntaryExitConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[7][0].is_zero(),
            "not_already_exited constraint should fire when FAR_FUTURE_MINUS_EXIT ≠ 0",
        );
    }

    #[test]
    #[should_panic(expected = "waiting period not yet elapsed")]
    fn waiting_period_violated_panics_in_host() {
        // current_epoch = 100, activation_epoch = 50 ⇒
        // current_epoch − activation_epoch = 50 < 256 = SHARD_COMMITTEE_PERIOD.
        let _ = VoluntaryExitWitness::from_signed_exit(
            7,
            42,
            synthetic_signature(),
            synthetic_pubkey(),
            100,
            50,
            synthetic_message_root(),
            synthetic_domain(),
        );
    }

    /// Algebraic side of the waiting-period check: if a malicious
    /// prover tampers the WAITED column to a value inconsistent with
    /// `current_epoch − activation_epoch − SHARD_COMMITTEE_PERIOD`, the
    /// `waiting_period` body fires.
    #[test]
    fn waiting_period_constraint_fires_on_tampered_waited() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump WAITED by 1; the linear identity in body 6 must fire.
        let bumped = cols[COL_WAITED][0].to_u64().wrapping_add(1);
        cols[COL_WAITED][0] = Scalar::from_u64(bumped, curve);
        let cs = VoluntaryExitConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[6][0].is_zero(),
            "waiting_period body should fire on tampered WAITED",
        );
        // The waited LE decomp body (5) should still vanish because we
        // did NOT update WAITED_BYTE — we only bumped the aggregate, so
        // actually body 5 SHOULD fire. Sanity-check that too.
        assert!(
            !bodies[5][0].is_zero(),
            "waited_le_decomp body should fire when WAITED ≠ Σ bytes",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_exit_to_validator_registry_descriptor(0, 1);
        assert_eq!(d1.label, "exit_to_validator_registry_v1");
        // 1 validator_index + 48 pubkey bytes.
        assert_eq!(d1.a_columns.len(), 1 + PUBKEY_BYTES);
        assert_eq!(d1.b_columns.len(), 1 + PUBKEY_BYTES);
        assert_eq!(d1.a_columns[0], COL_VALIDATOR_INDEX);
        for k in 0..PUBKEY_BYTES {
            assert_eq!(d1.a_columns[1 + k], COL_PUBKEY_OFFSET + k);
        }
        assert_eq!(
            d1.b_columns[0],
            crate::validator_htr_air::COL_VALIDATOR_INDEX
        );
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d1.b_selector_column,
            Some(crate::validator_htr_air::COL_IS_REAL)
        );

        let d2 = make_exit_to_signing_root_sha256_descriptor(0, 2);
        assert_eq!(d2.label, "exit_to_signing_root_sha256_v1");
        // 64 input bytes + 32 output bytes.
        assert_eq!(d2.a_columns.len(), 96);
        assert_eq!(d2.b_columns.len(), 96);
        for k in 0..HASH_BYTES {
            assert_eq!(d2.a_columns[k], COL_MESSAGE_ROOT_OFFSET + k);
        }
        for k in 0..DOMAIN_BYTES {
            assert_eq!(d2.a_columns[HASH_BYTES + k], COL_DOMAIN_OFFSET + k);
        }
        for k in 0..HASH_BYTES {
            assert_eq!(
                d2.a_columns[HASH_BYTES + DOMAIN_BYTES + k],
                COL_SIGNING_ROOT_OFFSET + k,
            );
        }

        let d3 = make_exit_to_bls_sig_descriptor(0, 3);
        assert_eq!(d3.label, "exit_to_bls_sig_v1");
        // 48 pubkey + 32 signing_root + 96 signature.
        assert_eq!(d3.a_columns.len(), PUBKEY_BYTES + HASH_BYTES + SIG_BYTES);
        assert_eq!(d3.b_columns.len(), PUBKEY_BYTES + HASH_BYTES + SIG_BYTES);
        assert_eq!(d3.a_columns[0], COL_PUBKEY_OFFSET);
        assert_eq!(d3.a_columns[PUBKEY_BYTES], COL_SIGNING_ROOT_OFFSET);
        assert_eq!(
            d3.a_columns[PUBKEY_BYTES + HASH_BYTES],
            COL_SIGNATURE_OFFSET,
        );
        assert_eq!(d3.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d3.b_selector_column,
            Some(crate::bls_pairing_air::COL_IS_REAL)
        );
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_EPOCH, 0);
        assert_eq!(COL_VALIDATOR_INDEX, 1);
        assert_eq!(COL_CURRENT_EPOCH, 2);
        assert_eq!(COL_ACTIVATION_EPOCH, 3);
        assert_eq!(COL_WAITED, 4);
        assert_eq!(COL_FAR_FUTURE_MINUS_EXIT, 5);
        assert_eq!(COL_PUBKEY_OFFSET, 6);
        assert_eq!(COL_SIGNATURE_OFFSET, 6 + 48);
        assert_eq!(COL_MESSAGE_ROOT_OFFSET, 6 + 48 + 96);
        assert_eq!(COL_DOMAIN_OFFSET, 6 + 48 + 96 + 32);
        assert_eq!(COL_SIGNING_ROOT_OFFSET, 6 + 48 + 96 + 32 + 32);
        assert_eq!(COL_EPOCH_BYTE_OFFSET, 6 + 48 + 96 + 32 + 32 + 32);
        assert_eq!(COL_VI_BYTE_OFFSET, COL_EPOCH_BYTE_OFFSET + 8);
        assert_eq!(COL_CE_BYTE_OFFSET, COL_VI_BYTE_OFFSET + 8);
        assert_eq!(COL_AE_BYTE_OFFSET, COL_CE_BYTE_OFFSET + 8);
        assert_eq!(COL_WAITED_BYTE_OFFSET, COL_AE_BYTE_OFFSET + 8);
        assert_eq!(COL_IS_REAL, COL_WAITED_BYTE_OFFSET + 8);
        // Round-13 appended cols (preserves all prior offsets).
        assert_eq!(COL_EXIT_EPOCH, COL_IS_REAL + 1);
        assert_eq!(COL_EXIT_EPOCH_BYTE_OFFSET, COL_EXIT_EPOCH + 1);
        // 6 scalar + 48 + 96 + 32 + 32 + 32 + 5*8 + 1 + 1 + 8 = 296.
        assert_eq!(NUM_COLUMNS, 296);
        assert_eq!(NUM_ROW_CONSTRAINTS, 11);
        assert_eq!(NUM_SHIFTED, 0);
    }

    /// Byte range lookup coverage: every byte column (pubkey, sig,
    /// message_root, domain, signing_root, and the 5 LE-byte
    /// decompositions) has a corresponding 8-bit declaration.
    #[test]
    fn byte_range_lookup_coverage() {
        let cs = VoluntaryExitConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // Expected count: 48 + 96 + 32 + 32 + 32 + 6*8 = 288 decls
        // (the 6th 8-byte block is `exit_epoch_byte`).
        let expected = PUBKEY_BYTES + SIG_BYTES + HASH_BYTES + DOMAIN_BYTES
            + HASH_BYTES + 6 * U64_BYTES;
        assert_eq!(reqs.declarations.len(), expected);
        // Every declaration is 8-bit and references a real column.
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
        // Spot-check pubkey[0] is in the set.
        assert!(reqs
            .declarations
            .iter()
            .any(|(d, _)| d.column_index == COL_PUBKEY_OFFSET));
        // Spot-check waited_byte[7] is in the set.
        assert!(reqs
            .declarations
            .iter()
            .any(|(d, _)| d.column_index == COL_WAITED_BYTE_OFFSET + 7));
    }

    /// Le-byte decomp constraint sanity: tampering an epoch byte must
    /// fire the epoch_le_decomp body.
    #[test]
    fn epoch_le_decomp_fires_on_byte_tamper() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump epoch_byte[0]; the LE decomp body 1 must fire.
        let bumped = cols[COL_EPOCH_BYTE_OFFSET][0].to_u64().wrapping_add(1);
        cols[COL_EPOCH_BYTE_OFFSET][0] = Scalar::from_u64(bumped, curve);
        let cs = VoluntaryExitConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[1][0].is_zero(),
            "epoch_le_decomp body should fire on tampered epoch_byte[0]",
        );
    }

    /// is_real binary constraint sanity.
    #[test]
    fn is_real_binary_fires_on_non_binary_value() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(2, curve);
        let cs = VoluntaryExitConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[0][0].is_zero(),
            "is_real_binary body should fire when is_real = 2",
        );
    }

    /// Round-13 algebraic-soundness closure: the new
    /// `exit_to_validator_htr_exit_epoch_v1` descriptor must target the
    /// (VALIDATOR_INDEX, EXIT_EPOCH) tuple on both sides.
    #[test]
    fn exit_epoch_descriptor_well_formed() {
        let d = make_exit_to_validator_htr_exit_epoch_descriptor(0, 1);
        assert_eq!(d.label, "exit_to_validator_htr_exit_epoch_v1");
        assert_eq!(d.a_columns.len(), 2);
        assert_eq!(d.b_columns.len(), 2);
        assert_eq!(d.a_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(d.a_columns[1], COL_EXIT_EPOCH);
        assert_eq!(
            d.b_columns[0],
            crate::validator_htr_air::COL_VALIDATOR_INDEX
        );
        assert_eq!(
            d.b_columns[1],
            crate::validator_htr_air::COL_EXIT_EPOCH
        );
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::validator_htr_air::COL_IS_REAL)
        );
    }

    /// Round-13 algebraic-soundness closure: tampering EXIT_EPOCH so
    /// that the linkage points at an `exit_epoch != FAR_FUTURE_EPOCH`
    /// while FAR_FUTURE_MINUS_EXIT is still 0 must fire the new
    /// `exit_epoch_consistency` body (slot 10).
    #[test]
    fn exit_epoch_consistency_fires_on_tampered_exit_epoch() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Pretend the validator's exit_epoch is some finite epoch (not
        // FAR_FUTURE_EPOCH). FAR_FUTURE_MINUS_EXIT stays at the honest
        // 0, so the consistency body becomes
        //   is_real · (FAR_FUTURE_EPOCH − 9_000_000 − 0)
        // ≠ 0, which must fire.
        cols[COL_EXIT_EPOCH][0] = Scalar::from_u64(9_000_000, curve);
        // Also bump LE bytes consistently so we isolate constraint 10
        // (not constraint 9's le_decomp).
        let bytes = 9_000_000u64.to_le_bytes();
        for b in 0..U64_BYTES {
            cols[COL_EXIT_EPOCH_BYTE_OFFSET + b][0] =
                Scalar::from_u64(bytes[b] as u64, curve);
        }
        let cs = VoluntaryExitConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // exit_epoch_le_decomp (body 9) should still vanish — we kept
        // the bytes consistent with the new EXIT_EPOCH aggregate.
        assert!(
            bodies[9][0].is_zero(),
            "exit_epoch_le_decomp should vanish on consistent bytes",
        );
        // exit_epoch_consistency (body 10) must fire.
        assert!(
            !bodies[10][0].is_zero(),
            "exit_epoch_consistency body should fire when \
             FAR_FUTURE_EPOCH − EXIT_EPOCH − FAR_FUTURE_MINUS_EXIT ≠ 0",
        );
    }

    /// Round-13 algebraic-soundness closure: tampering a single
    /// EXIT_EPOCH_BYTE while leaving the aggregate EXIT_EPOCH alone
    /// must fire the new `exit_epoch_le_decomp` body (slot 9).
    #[test]
    fn exit_epoch_le_decomp_fires_on_byte_tamper() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump exit_epoch_byte[0] by one without updating the aggregate;
        // body 9 (LE decomp) must fire.
        let bumped =
            cols[COL_EXIT_EPOCH_BYTE_OFFSET][0].to_u64().wrapping_add(1);
        cols[COL_EXIT_EPOCH_BYTE_OFFSET][0] = Scalar::from_u64(bumped, curve);
        let cs = VoluntaryExitConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[9][0].is_zero(),
            "exit_epoch_le_decomp body should fire on tampered exit_epoch_byte[0]",
        );
    }

    /// `evaluate_at_point` and `evaluate_on_domain` must agree on
    /// honest witnesses (both should report zero on every body).
    #[test]
    fn evaluate_at_point_matches_evaluate_on_domain_for_honest() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = VoluntaryExitConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let domain_eval = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // All bodies should be zero on row 0; therefore the α-RLC
        // aggregate is also zero.
        let alpha = Scalar::from_u64(11, curve);
        let row0_evals: Vec<Scalar> =
            col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must be zero on honest row");

        // Sanity loop: same on padding row 1.
        if trace.num_rows < trace.padded_size as usize {
            let row1_evals: Vec<Scalar> =
                col_refs.iter().map(|c| c[1].clone()).collect();
            let _ = cs.evaluate_at_point(&row1_evals, &alpha);
            // Bodies on padding row are also zero by construction.
            for (k, body) in domain_eval.iter().enumerate() {
                assert!(
                    body[1].is_zero(),
                    "constraint {} body should vanish on padding row 1",
                    k
                );
            }
        }
    }
}
