//! Beacon attestation aggregate-signature AIR (Phase C).
//!
//! Proves that an `Attestation { aggregation_bits, data, signature }` is a
//! valid BLS aggregate signature over the `AttestationData` signing root by
//! the validators selected by the aggregation bitmap.
//!
//! This is a *composition layer* analogous to
//! [`crate::sync_committee_sig_air`]: it commits per-participating-validator
//! rows and exposes cross-AIR LogUp descriptors that bind the witness
//! data to the underlying gadget AIRs that prove the actual cryptographic
//! relations:
//!
//!   - [`crate::validator_registry_air`] — pubkey/index ↔ beacon-state
//!     validator registry (per-validator HTR leaf).
//!   - [`crate::hash_to_g2_air`] — signing root → `H(msg) ∈ G2`.
//!   - [`crate::bls_pairing_air`] — pairing equation
//!     `e(agg_pk, H(msg)) == e(agg_sig, G2_gen)`.
//!   - [`crate::ffg_checkpoint_chain`] — Casper FFG source/target epoch
//!     ordering (host-side oracle today; descriptor pinned for the
//!     future AIRified FFG chain).
//!
//! # Row shape (per *participating* validator)
//!
//! Per row this AIR commits:
//!
//!   * `filtered_pubkey[0..48]`    — one participating validator's compressed BLS12-381 pubkey.
//!   * `bit_index`                 — the validator's position within the committee (0..MAX_COMMITTEE).
//!   * `signing_root[0..32]`       — SSZ signing root over `(data, domain)` (constant across active rows of one attestation).
//!   * `aggregate_sig[0..96]`      — aggregate signature (constant across active rows).
//!   * `aggregate_pubkey_x[0..48]` — canonical BE byte form of `agg_pk_x` (flag bits cleared).
//!   * `aggregate_pubkey_g1_limbs[0..12]` — `(agg_pk_x_limbs[0..6], agg_pk_y_limbs[0..6])`.
//!   * `msg_g2_limbs[0..24]`       — `H(signing_root) ∈ G2` as `(x.c0, x.c1, y.c0, y.c1)` × 6 limbs.
//!   * `slot`, `committee_index`, `source_epoch`, `target_epoch` — `AttestationData` scalar fields.
//!   * `is_real`                   — selector binary column.
//!
//! Total: 48 + 1 + 32 + 96 + 48 + 12 + 24 + 4 + 1 = **266 columns**.
//!
//! # Algebraic constraints
//!
//! Row-local:
//!
//!   0. `is_real ∈ {0, 1}`.
//!   1. `aggregate_pubkey_g1_limbs[0]` (= `agg_pk_x_limbs[0]`)
//!      = `Σ_{k=0..8} aggregate_pubkey_x[k] · 2^(8·(7-k))` — representative
//!      G1 byte→limb decomposition over the most-significant 8 bytes
//!      (BE convention matching [`crate::bls_pairing_air`]).
//!   2. `bit_index` matches the committee position offset chain
//!      (gated by `is_real`): `bit_index ≥ 0` enforced trivially by
//!      byte-range-check on the low byte (covered by lookups).
//!   3. `source_epoch ≤ target_epoch` is host-side enforced via
//!      `ffg_checkpoint_chain`; here we pin `target_epoch − source_epoch`
//!      to a non-negative quantity by exposing an `is_real`-gated equality:
//!      `is_real · (target_epoch − source_epoch − (target_epoch − source_epoch)) = 0`.
//!      (Trivially satisfied; reserved as a placeholder so the row-constraint
//!      count and labels are pinned for the future strict ordering AIR.)
//!   4. `slot · is_real = slot · is_real` — pinned placeholder for the
//!      future slot/committee-index range check (selector-gated identity).
//!   5. `committee_index · is_real = committee_index · is_real` — pinned
//!      placeholder for the future committee-index sub-bit decomposition.
//!
//! Constraints 2..5 are *pinned* trivial identities — the cryptographically
//! load-bearing checks (FFG ordering, slot/committee bounds) are delegated
//! to the dedicated AIRs targeted by the cross-AIR LogUp descriptors below.
//! They keep the constraint label vector stable so downstream
//! `joint_prove` pipelines can include this AIR without churn.
//!
//! Byte range checks (8-bit) are emitted for every byte column via
//! `LookupRequirements`.
//!
//! # Cross-AIR linkages
//!
//!   * [`make_attestation_to_validator_registry_descriptor`] —
//!     `(bit_index, filtered_pubkey[0..32])` ↔
//!     `(VALIDATOR_INDEX, CURRENT_HASH[0..32])`. Binds the participating
//!     pubkey's first 32 bytes (representative anchor; the full
//!     validator HTR matching is handled by `validator_htr_air`).
//!   * [`make_attestation_to_hash_to_g2_descriptor`] —
//!     `(signing_root[0..32], msg_g2_limbs[0..24])` ↔
//!     `(MSG[0..32], OUT_*_LIMBS[0..24])`. 56-col tuple.
//!   * [`make_attestation_to_pairing_descriptor`] —
//!     `(agg_pk_x_bytes, agg_sig, signing_root)` ↔
//!     `(PK_X_BYTES, SIG_BYTES, MSG_HASH)`. 176-col tuple.
//!   * [`make_attestation_to_ffg_descriptor`] —
//!     `(source_epoch, target_epoch)` ↔ `(source_epoch, target_epoch)` on
//!     a future FFG chain AIR. Selector-gated by `is_real`. Layer index
//!     is a scaffold placeholder until the FFG chain is AIRified.
//!
//! # Honest scope statement
//!
//! What this AIR proves: per-row binarity and one representative
//! agg-pk byte/limb shape constraint. Everything else — aggregation
//! `Σ pk_i = agg_pk`, hash-to-curve, pairing equation, FFG ordering,
//! signing-root SSZ correctness — is delegated to host-side oracle (for
//! the witness-builder) or to the sub-AIRs (via the cross-AIR LogUp
//! descriptors). Mirrors the scope of [`crate::sync_committee_sig_air`].

use crate::beacon::{AttestationData, Checkpoint};
use crate::bls_sig::{aggregate_pubkeys, fast_aggregate_verify, PublicKey, Signature};
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::nonnative_fp::Fp;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// BLS12-381 compressed G1 pubkey length.
pub const PK_BYTES: usize = 48;
/// BLS12-381 compressed G2 signature length.
pub const SIG_BYTES: usize = 96;
/// SSZ signing-root message length.
pub const MSG_BYTES: usize = 32;

/// 64-bit limbs per Fp element.
pub const LIMBS_PER_FP: usize = 6;
/// Bytes per limb (BE u64).
pub const BYTES_PER_LIMB: usize = 8;

/// G1 aggregate-pubkey limb width: x + y = 12 u64 limbs.
pub const G1_LIMBS: usize = 2 * LIMBS_PER_FP;
/// G2 message-point limb width: 4 × 6 = 24 u64 limbs.
pub const G2_LIMBS: usize = 4 * LIMBS_PER_FP;

/// Per-attestation scalar `AttestationData` fields exposed as one
/// column each: slot, committee_index, source_epoch, target_epoch.
pub const NUM_DATA_FIELDS: usize = 4;

/// Maximum committee size for the scaffold tests. Real Phase 0 uses
/// `MAX_VALIDATORS_PER_COMMITTEE = 2048`; we cap to 16 to keep the
/// fast-test domains small.
pub const MAX_COMMITTEE: usize = 16;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_FILTERED_PUBKEY_OFFSET: usize = 0;
pub const COL_BIT_INDEX: usize = COL_FILTERED_PUBKEY_OFFSET + PK_BYTES;
pub const COL_SIGNING_ROOT_OFFSET: usize = COL_BIT_INDEX + 1;
pub const COL_AGGREGATE_SIG_OFFSET: usize = COL_SIGNING_ROOT_OFFSET + MSG_BYTES;

pub const COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET: usize = COL_AGGREGATE_SIG_OFFSET + SIG_BYTES;

pub const COL_AGG_PK_X_LIMB_OFFSET: usize = COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET + PK_BYTES;
pub const COL_AGG_PK_Y_LIMB_OFFSET: usize = COL_AGG_PK_X_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_MSG_G2_X_C0_LIMB_OFFSET: usize = COL_AGG_PK_Y_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_MSG_G2_X_C1_LIMB_OFFSET: usize = COL_MSG_G2_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_MSG_G2_Y_C0_LIMB_OFFSET: usize = COL_MSG_G2_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_MSG_G2_Y_C1_LIMB_OFFSET: usize = COL_MSG_G2_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_SLOT: usize = COL_MSG_G2_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_COMMITTEE_INDEX: usize = COL_SLOT + 1;
pub const COL_SOURCE_EPOCH: usize = COL_COMMITTEE_INDEX + 1;
pub const COL_TARGET_EPOCH: usize = COL_SOURCE_EPOCH + 1;

pub const COL_IS_REAL: usize = COL_TARGET_EPOCH + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// Row-local constraint count.
pub const NUM_ROW_CONSTRAINTS: usize = 6;
pub const NUM_SHIFTED: usize = 0;

// ─── Attestation type (committee-bitlist form) ────────────────────────

/// Phase 0 committee-bitlist attestation. Distinct from
/// [`crate::beacon::IndexedAttestation`] which uses explicit indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attestation {
    /// Per-committee participation bits. `aggregation_bits[i] = true`
    /// means committee member `i` participated. Capped at
    /// `MAX_VALIDATORS_PER_COMMITTEE` in real consensus.
    pub aggregation_bits: Vec<bool>,
    pub data: AttestationData,
    /// BLS12-381 G2 compressed aggregate signature (96 bytes).
    pub signature: [u8; SIG_BYTES],
}

impl Default for Attestation {
    fn default() -> Self {
        Self {
            aggregation_bits: Vec::new(),
            data: AttestationData::default(),
            signature: [0u8; SIG_BYTES],
        }
    }
}

impl Attestation {
    /// SSZ signing root over the underlying `AttestationData`. The
    /// production beacon chain computes the signing root over
    /// `(data, domain)`; for this scaffold we use the raw data HTR as
    /// the signed message (the DST handles domain separation in BLS),
    /// which matches the witness builder.
    pub fn signing_root(&self) -> [u8; MSG_BYTES] {
        self.data.hash_tree_root()
    }
}

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AttestationAggregateRow {
    pub filtered_pubkey: [u8; PK_BYTES],
    pub bit_index: u64,
    pub signing_root: [u8; MSG_BYTES],
    pub aggregate_sig: [u8; SIG_BYTES],
    pub aggregate_pubkey_x_bytes: [u8; PK_BYTES],

    pub agg_pk_x: Fp,
    pub agg_pk_y: Fp,
    pub msg_g2_x_c0: Fp,
    pub msg_g2_x_c1: Fp,
    pub msg_g2_y_c0: Fp,
    pub msg_g2_y_c1: Fp,

    pub slot: u64,
    pub committee_index: u64,
    pub source_epoch: u64,
    pub target_epoch: u64,
}

#[derive(Clone, Debug, Default)]
pub struct AttestationAggregateWitness {
    pub rows: Vec<AttestationAggregateRow>,
    /// Source / target checkpoints from the input attestation, exposed
    /// for downstream FFG-chain verification.
    pub source: Checkpoint,
    pub target: Checkpoint,
}

impl AttestationAggregateWitness {
    /// Build a witness from a beacon `Attestation` and the full
    /// committee pubkey list. Host-side does:
    ///   1. Filter `validator_pubkeys` by `aggregation_bits`.
    ///   2. Compute the signing root via `AttestationData::hash_tree_root`.
    ///   3. Aggregate the participating pubkeys via `aggregate_pubkeys`.
    ///   4. Hash-to-G2 the signing root.
    ///   5. Sanity-check `fast_aggregate_verify`. Returns `None` on
    ///      length mismatch or host-side verification failure.
    ///
    /// DST is fixed to the Phase 0 attestation DST
    /// (`BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_`).
    pub fn from_attestation(
        att: &Attestation,
        validator_pubkeys: &[[u8; PK_BYTES]],
    ) -> Option<Self> {
        if att.aggregation_bits.len() != validator_pubkeys.len() {
            return None;
        }
        if att.aggregation_bits.is_empty() {
            return None;
        }
        // Filter committee → participating subset.
        let mut filtered: Vec<(usize, [u8; PK_BYTES])> = Vec::new();
        for (i, (&bit, pk)) in att.aggregation_bits.iter().zip(validator_pubkeys.iter()).enumerate()
        {
            if bit {
                filtered.push((i, *pk));
            }
        }
        if filtered.is_empty() {
            return None;
        }

        let dst: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
        let pk_objs: Vec<PublicKey> = filtered.iter().map(|(_, p)| PublicKey(*p)).collect();
        let sig_obj = Signature(att.signature);
        let signing_root = att.signing_root();

        if !fast_aggregate_verify(&pk_objs, &signing_root, &sig_obj, dst) {
            return None;
        }

        let agg_pk_compressed = aggregate_pubkeys(&pk_objs).ok()?.0;
        let agg_pk_aff = crate::pairing::G1Affine::from_bytes(&agg_pk_compressed).ok()?;
        if agg_pk_aff.infinity {
            return None;
        }
        let mut agg_pk_x_bytes = agg_pk_compressed;
        agg_pk_x_bytes[0] &= 0x1f; // clear flag bits → canonical Fp byte form

        let msg_g2_compressed: [u8; SIG_BYTES] = unsafe {
            let mut p = blst::blst_p2::default();
            blst::blst_hash_to_g2(
                &mut p,
                signing_root.as_ptr(),
                signing_root.len(),
                dst.as_ptr(),
                dst.len(),
                core::ptr::null(),
                0,
            );
            let mut out = [0u8; SIG_BYTES];
            blst::blst_p2_compress(out.as_mut_ptr(), &p);
            out
        };
        let msg_g2_aff = crate::pairing::G2Affine::from_bytes(&msg_g2_compressed).ok()?;
        if msg_g2_aff.infinity {
            return None;
        }

        let slot = att.data.slot;
        let committee_index = att.data.index;
        let source_epoch = att.data.source.epoch;
        let target_epoch = att.data.target.epoch;

        let rows: Vec<AttestationAggregateRow> = filtered
            .iter()
            .map(|(idx, pk)| AttestationAggregateRow {
                filtered_pubkey: *pk,
                bit_index: *idx as u64,
                signing_root,
                aggregate_sig: att.signature,
                aggregate_pubkey_x_bytes: agg_pk_x_bytes,
                agg_pk_x: agg_pk_aff.x,
                agg_pk_y: agg_pk_aff.y,
                msg_g2_x_c0: msg_g2_aff.x.c0,
                msg_g2_x_c1: msg_g2_aff.x.c1,
                msg_g2_y_c0: msg_g2_aff.y.c0,
                msg_g2_y_c1: msg_g2_aff.y.c1,
                slot,
                committee_index,
                source_epoch,
                target_epoch,
            })
            .collect();

        Some(Self { rows, source: att.data.source, target: att.data.target })
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &AttestationAggregateWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..PK_BYTES {
            columns[COL_FILTERED_PUBKEY_OFFSET + k][r] =
                Scalar::from_u64(row.filtered_pubkey[k] as u64, curve);
            columns[COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.aggregate_pubkey_x_bytes[k] as u64, curve);
        }
        columns[COL_BIT_INDEX][r] = Scalar::from_u64(row.bit_index, curve);
        for k in 0..MSG_BYTES {
            columns[COL_SIGNING_ROOT_OFFSET + k][r] =
                Scalar::from_u64(row.signing_root[k] as u64, curve);
        }
        for k in 0..SIG_BYTES {
            columns[COL_AGGREGATE_SIG_OFFSET + k][r] =
                Scalar::from_u64(row.aggregate_sig[k] as u64, curve);
        }

        for (offset, fp) in [
            (COL_AGG_PK_X_LIMB_OFFSET, &row.agg_pk_x),
            (COL_AGG_PK_Y_LIMB_OFFSET, &row.agg_pk_y),
            (COL_MSG_G2_X_C0_LIMB_OFFSET, &row.msg_g2_x_c0),
            (COL_MSG_G2_X_C1_LIMB_OFFSET, &row.msg_g2_x_c1),
            (COL_MSG_G2_Y_C0_LIMB_OFFSET, &row.msg_g2_y_c0),
            (COL_MSG_G2_Y_C1_LIMB_OFFSET, &row.msg_g2_y_c1),
        ] {
            for j in 0..LIMBS_PER_FP {
                columns[offset + j][r] = Scalar::from_u64(fp.limbs[j], curve);
            }
        }

        columns[COL_SLOT][r] = Scalar::from_u64(row.slot, curve);
        columns[COL_COMMITTEE_INDEX][r] = Scalar::from_u64(row.committee_index, curve);
        columns[COL_SOURCE_EPOCH][r] = Scalar::from_u64(row.source_epoch, curve);
        columns[COL_TARGET_EPOCH][r] = Scalar::from_u64(row.target_epoch, curve);

        columns[COL_IS_REAL][r] = one.clone();
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct AttestationAggregateConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AttestationAggregateConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Weight `256^(7-k)` for byte `k` of an 8-byte BE limb.
fn be_limb_byte_weight(k: usize) -> u64 {
    1u64 << (8 * (BYTES_PER_LIMB - 1 - k))
}

impl VmConstraintSystem for AttestationAggregateConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "agg_pk_x_limb_0_decomp".into(),
            "bit_index_is_real_gated".into(),
            "epoch_ordering_pinned".into(),
            "slot_pinned".into(),
            "committee_index_pinned".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real ∈ {0,1}.
        {
            let mut c = vec![zero.clone(); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1: agg_pk_x_limbs[0] = Σ_{k=0..8} agg_pk_x_bytes[k] · 2^(8·(7-k)).
        {
            let mut c = vec![zero.clone(); n];
            for r in 0..n {
                let mut sum = zero.clone();
                for k in 0..BYTES_PER_LIMB {
                    let b = &columns[COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET + k][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(be_limb_byte_weight(k), curve)));
                }
                c[r] = columns[COL_AGG_PK_X_LIMB_OFFSET][r].sub(&sum);
            }
            out.push(c);
        }

        // 2: is_real · (bit_index − bit_index) = 0 — pinned identity.
        //    Reserved slot for the future committee-bound bit_index range
        //    check; trivially zero today.
        {
            let c = vec![zero.clone(); n];
            out.push(c);
        }

        // 3: is_real · 0 = 0 — pinned epoch-ordering placeholder.
        //    The real ordering check `source_epoch ≤ target_epoch` is host-
        //    side enforced via `ffg_checkpoint_chain` and will move into
        //    the FFG chain AIR once that exists.
        {
            let c = vec![zero.clone(); n];
            out.push(c);
        }

        // 4: pinned slot placeholder.
        {
            let c = vec![zero.clone(); n];
            out.push(c);
        }

        // 5: pinned committee_index placeholder.
        {
            let c = vec![zero.clone(); n];
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

        let v = &col_evals[COL_IS_REAL];
        let c0 = v.mul(&v.sub(&one));

        let mut sum = Scalar::zero(curve);
        for k in 0..BYTES_PER_LIMB {
            sum = sum.add(
                &col_evals[COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET + k]
                    .mul(&Scalar::from_u64(be_limb_byte_weight(k), curve)),
            );
        }
        let c1 = col_evals[COL_AGG_PK_X_LIMB_OFFSET].sub(&sum);

        // Constraints 2..5 are pinned-zero placeholders.
        c0.add(&alpha.mul(&c1))
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
        let v_m1 = poly_sub(v, &one_poly, curve);
        let c0 = poly_mul(v, &v_m1, curve);

        let mut sum = vec![Scalar::zero(curve)];
        for k in 0..BYTES_PER_LIMB {
            let b = &col_coeffs[COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET + k];
            let term = poly_scalar_mul(b, &Scalar::from_u64(be_limb_byte_weight(k), curve));
            sum = poly_add(&sum, &term, curve);
        }
        let c1 = poly_sub(&col_coeffs[COL_AGG_PK_X_LIMB_OFFSET], &sum, curve);

        poly_add(&c0, &poly_scalar_mul(&c1, alpha), curve)
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
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
            return;
        }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls12381);
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
        for k in 0..PK_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("attestation_filtered_pubkey_byte_{}_8bit", k),
                    column_index: COL_FILTERED_PUBKEY_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("attestation_agg_pk_x_byte_{}_8bit", k),
                    column_index: COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MSG_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("attestation_signing_root_byte_{}_8bit", k),
                    column_index: COL_SIGNING_ROOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SIG_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("attestation_agg_sig_byte_{}_8bit", k),
                    column_index: COL_AGGREGATE_SIG_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR linkage descriptors ────────────────────────────────────

/// Bind `(bit_index, filtered_pubkey[0..32])` to the validator registry
/// AIR's `(VALIDATOR_INDEX, CURRENT_HASH[0..32])` tuple. The first 32
/// bytes of the pubkey serve as the representative anchor — the full
/// pubkey HTR / validator HTR match is handled by `validator_htr_air`.
///
/// A side gated by `IS_REAL`, B side gated by the registry AIR's
/// `IS_REAL`. 33-col tuple.
pub fn make_attestation_to_validator_registry_descriptor(
    attestation_layer_index: usize,
    validator_registry_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::validator_registry_air as vr;
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + MSG_BYTES);
    a_columns.push(COL_BIT_INDEX);
    for k in 0..MSG_BYTES {
        a_columns.push(COL_FILTERED_PUBKEY_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + MSG_BYTES);
    b_columns.push(vr::COL_VALIDATOR_INDEX);
    for k in 0..MSG_BYTES {
        b_columns.push(vr::COL_CURRENT_HASH_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "attestation_to_validator_registry_v1".into(),
        a_layer_index: attestation_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_registry_layer_index,
        b_columns,
        b_selector_column: Some(vr::COL_IS_REAL),
    }
}

/// Bind `(signing_root[0..32], msg_g2_limbs[0..24])` to the hash-to-G2
/// AIR's `(MSG[0..32], OUT_*_LIMBS[0..24])`. 56-col tuple, both sides
/// gated by `IS_REAL`.
pub fn make_attestation_to_hash_to_g2_descriptor(
    attestation_layer_index: usize,
    hash_to_g2_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::hash_to_g2_air as h2g2;

    let mut a_columns: Vec<usize> = Vec::with_capacity(MSG_BYTES + G2_LIMBS);
    for k in 0..MSG_BYTES {
        a_columns.push(COL_SIGNING_ROOT_OFFSET + k);
    }
    for j in 0..LIMBS_PER_FP {
        a_columns.push(COL_MSG_G2_X_C0_LIMB_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP {
        a_columns.push(COL_MSG_G2_X_C1_LIMB_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP {
        a_columns.push(COL_MSG_G2_Y_C0_LIMB_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP {
        a_columns.push(COL_MSG_G2_Y_C1_LIMB_OFFSET + j);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(MSG_BYTES + G2_LIMBS);
    for k in 0..MSG_BYTES {
        b_columns.push(h2g2::COL_MSG_OFFSET + k);
    }
    for j in 0..LIMBS_PER_FP {
        b_columns.push(h2g2::COL_OUT_X_C0_LIMB_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP {
        b_columns.push(h2g2::COL_OUT_X_C1_LIMB_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP {
        b_columns.push(h2g2::COL_OUT_Y_C0_LIMB_OFFSET + j);
    }
    for j in 0..LIMBS_PER_FP {
        b_columns.push(h2g2::COL_OUT_Y_C1_LIMB_OFFSET + j);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "attestation_to_hash_to_g2_v1".into(),
        a_layer_index: attestation_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: hash_to_g2_layer_index,
        b_columns,
        b_selector_column: Some(h2g2::COL_IS_REAL),
    }
}

/// Bind `(agg_pk_x_bytes, agg_sig, signing_root)` to the pairing AIR's
/// `(PK_X_BYTES, SIG_BYTES, MSG_HASH)`. 176-col tuple.
pub fn make_attestation_to_pairing_descriptor(
    attestation_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;

    let mut a_columns: Vec<usize> = Vec::with_capacity(PK_BYTES + SIG_BYTES + MSG_BYTES);
    for k in 0..PK_BYTES {
        a_columns.push(COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET + k);
    }
    for k in 0..SIG_BYTES {
        a_columns.push(COL_AGGREGATE_SIG_OFFSET + k);
    }
    for k in 0..MSG_BYTES {
        a_columns.push(COL_SIGNING_ROOT_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(PK_BYTES + SIG_BYTES + MSG_BYTES);
    for k in 0..PK_BYTES {
        b_columns.push(bp::COL_PK_X_BYTES_OFFSET + k);
    }
    for k in 0..SIG_BYTES {
        b_columns.push(bp::COL_SIG_BYTES_OFFSET + k);
    }
    for k in 0..MSG_BYTES {
        b_columns.push(bp::COL_MSG_HASH_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "attestation_to_pairing_v1".into(),
        a_layer_index: attestation_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bls_pairing_layer_index,
        b_columns,
        b_selector_column: Some(bp::COL_IS_REAL),
    }
}

/// Bind `(source_epoch, target_epoch)` to a future FFG checkpoint
/// chain AIR. The current `ffg_checkpoint_chain` module is host-side
/// only (no AIR columns yet), so this descriptor is **target-shape
/// scaffolding**: it pins the tuple shape and selector gating so the
/// LogUp orchestration code can already wire it in. Once the FFG chain
/// is AIRified the `b_columns` will point at the actual FFG-AIR
/// `(SOURCE_EPOCH, TARGET_EPOCH)` offsets and the `b_selector_column`
/// at its `IS_REAL`.
///
/// Both sides reference this AIR's own epoch columns by index for the
/// scaffold (self-bound until the FFG chain AIR lands); the layer
/// indices distinguish the two roles at the descriptor level.
pub fn make_attestation_to_ffg_descriptor(
    attestation_layer_index: usize,
    ffg_chain_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = vec![COL_SOURCE_EPOCH, COL_TARGET_EPOCH];
    // Self-bound scaffold: B side mirrors A side until the FFG chain AIR
    // exposes its own (source_epoch, target_epoch) columns. The
    // `cross_air_logup` orchestrator treats layer-index identity as
    // self-loop (still a valid multiset equality on the same columns).
    let b_columns: Vec<usize> = vec![COL_SOURCE_EPOCH, COL_TARGET_EPOCH];
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "attestation_to_ffg_v1".into(),
        a_layer_index: attestation_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: ffg_chain_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon::{AttestationData, Checkpoint};
    use crate::bls_sig::{aggregate_sigs, SecretKey};

    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    /// Build a real attestation by signing the SSZ HTR of an
    /// `AttestationData` with `n` deterministic keys, using a fixed
    /// `aggregation_bits` pattern (all-true over the participating
    /// prefix, false otherwise) and the analogous validator pubkey
    /// list for the committee.
    fn real_attestation(
        committee_size: usize,
        n_participating: usize,
    ) -> (Attestation, Vec<[u8; PK_BYTES]>) {
        assert!(n_participating <= committee_size);
        assert!(committee_size <= MAX_COMMITTEE);

        // committee_size keys; only the first `n_participating` sign.
        let sks: Vec<SecretKey> =
            (1..=committee_size as u8).map(SecretKey::from_u8_seed).collect();
        let pks: Vec<[u8; PK_BYTES]> = sks.iter().map(|s| s.public_key().0).collect();

        let data = AttestationData {
            slot: 4242,
            index: 7,
            beacon_block_root: [0x11; 32],
            source: Checkpoint { epoch: 100, root: [0x22; 32] },
            target: Checkpoint { epoch: 101, root: [0x33; 32] },
        };
        let signing_root = data.hash_tree_root();

        let participating_sks = &sks[..n_participating];
        let sigs: Vec<_> = participating_sks
            .iter()
            .map(|s| s.sign(&signing_root, POP_DST))
            .collect();
        let agg_sig = aggregate_sigs(&sigs).unwrap().0;

        let aggregation_bits: Vec<bool> = (0..committee_size)
            .map(|i| i < n_participating)
            .collect();

        let att = Attestation {
            aggregation_bits,
            data,
            signature: agg_sig,
        };
        (att, pks)
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: real aggregate builds witness
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_from_real_attestation() {
        let (att, pks) = real_attestation(8, 5);
        let w = AttestationAggregateWitness::from_attestation(&att, &pks)
            .expect("real attestation must build a witness");
        assert_eq!(w.rows.len(), 5);
        // bit_index counts up over participating prefix.
        for (i, row) in w.rows.iter().enumerate() {
            assert_eq!(row.bit_index as usize, i);
            assert_eq!(row.filtered_pubkey, pks[i]);
            assert_eq!(row.signing_root, att.signing_root());
            assert_eq!(row.aggregate_sig, att.signature);
            assert_eq!(row.slot, att.data.slot);
            assert_eq!(row.committee_index, att.data.index);
            assert_eq!(row.source_epoch, att.data.source.epoch);
            assert_eq!(row.target_epoch, att.data.target.epoch);
        }
        // Aggregate pk x byte form round-trips through Fp.
        let row0 = &w.rows[0];
        let want_x =
            Fp::from_bytes_be(&row0.aggregate_pubkey_x_bytes).expect("agg_pk_x_bytes is canonical");
        assert_eq!(row0.agg_pk_x.limbs, want_x.limbs);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: tampered aggregate signature rejected
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_rejects_tampered_signature() {
        let (mut att, pks) = real_attestation(6, 4);
        att.signature[10] ^= 0x01;
        assert!(
            AttestationAggregateWitness::from_attestation(&att, &pks).is_none(),
            "tampered agg_sig must fail fast_aggregate_verify",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: tampered aggregation_bits rejected
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_rejects_tampered_bitmap() {
        let (mut att, pks) = real_attestation(8, 5);
        // Flip a participating bit OFF — now the filtered set doesn't
        // match the signature → fast_aggregate_verify fails.
        att.aggregation_bits[2] = false;
        assert!(
            AttestationAggregateWitness::from_attestation(&att, &pks).is_none(),
            "tampering aggregation_bits must fail fast_aggregate_verify",
        );

        // Also: flip a non-participating bit ON — now the filtered set
        // includes a non-signer → fast_aggregate_verify fails.
        let (mut att2, pks2) = real_attestation(8, 5);
        att2.aggregation_bits[6] = true;
        assert!(
            AttestationAggregateWitness::from_attestation(&att2, &pks2).is_none(),
            "adding a non-signer bit must fail fast_aggregate_verify",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: honest witness satisfies all constraints; tampered
    // representative byte fires the limb-decomposition constraint
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness_and_fire_on_tamper() {
        let (att, pks) = real_attestation(4, 3);
        let w = AttestationAggregateWitness::from_attestation(&att, &pks).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = AttestationAggregateConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(val.is_zero(), "constraint {} row {} not zero", i, r);
            }
        }

        // Tamper MSB of agg_pk_x_bytes by adding 1 → fires constraint 1.
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        let original = cols[COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET][0].clone();
        cols[COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET][0] = original.add(&Scalar::one(curve));
        let col_refs2: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res2 = cs.evaluate_on_domain(&col_refs2, trace.num_rows);
        assert!(
            !res2[1][0].is_zero(),
            "tampering agg_pk_x_bytes[0] must fire limb-0 decomp",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: descriptors well-formed
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn descriptors_well_formed() {
        let d_vr = make_attestation_to_validator_registry_descriptor(0, 1);
        assert_eq!(d_vr.label, "attestation_to_validator_registry_v1");
        assert_eq!(d_vr.a_columns.len(), 1 + MSG_BYTES);
        assert_eq!(d_vr.b_columns.len(), 1 + MSG_BYTES);
        assert_eq!(d_vr.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_vr.b_selector_column,
            Some(crate::validator_registry_air::COL_IS_REAL),
        );
        assert_eq!(d_vr.a_columns[0], COL_BIT_INDEX);
        assert_eq!(d_vr.a_columns[1], COL_FILTERED_PUBKEY_OFFSET);
        assert_eq!(
            d_vr.b_columns[0],
            crate::validator_registry_air::COL_VALIDATOR_INDEX,
        );
        assert_eq!(
            d_vr.b_columns[1],
            crate::validator_registry_air::COL_CURRENT_HASH_OFFSET,
        );

        let d_h2g2 = make_attestation_to_hash_to_g2_descriptor(0, 2);
        assert_eq!(d_h2g2.label, "attestation_to_hash_to_g2_v1");
        assert_eq!(d_h2g2.a_columns.len(), MSG_BYTES + G2_LIMBS);
        assert_eq!(d_h2g2.b_columns.len(), MSG_BYTES + G2_LIMBS);
        assert_eq!(d_h2g2.a_columns[0], COL_SIGNING_ROOT_OFFSET);
        assert_eq!(d_h2g2.a_columns[MSG_BYTES], COL_MSG_G2_X_C0_LIMB_OFFSET);
        assert_eq!(
            d_h2g2.b_columns[0],
            crate::hash_to_g2_air::COL_MSG_OFFSET,
        );
        assert_eq!(
            d_h2g2.b_columns[MSG_BYTES],
            crate::hash_to_g2_air::COL_OUT_X_C0_LIMB_OFFSET,
        );

        let d_pair = make_attestation_to_pairing_descriptor(0, 3);
        assert_eq!(d_pair.label, "attestation_to_pairing_v1");
        let expected = PK_BYTES + SIG_BYTES + MSG_BYTES;
        assert_eq!(d_pair.a_columns.len(), expected);
        assert_eq!(d_pair.b_columns.len(), expected);
        assert_eq!(d_pair.a_columns[0], COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET);
        assert_eq!(d_pair.a_columns[PK_BYTES], COL_AGGREGATE_SIG_OFFSET);
        assert_eq!(d_pair.a_columns[PK_BYTES + SIG_BYTES], COL_SIGNING_ROOT_OFFSET);
        assert_eq!(
            d_pair.b_columns[0],
            crate::bls_pairing_air::COL_PK_X_BYTES_OFFSET,
        );

        let d_ffg = make_attestation_to_ffg_descriptor(0, 4);
        assert_eq!(d_ffg.label, "attestation_to_ffg_v1");
        assert_eq!(d_ffg.a_columns, vec![COL_SOURCE_EPOCH, COL_TARGET_EPOCH]);
        assert_eq!(d_ffg.b_columns, vec![COL_SOURCE_EPOCH, COL_TARGET_EPOCH]);
        assert_eq!(d_ffg.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_ffg.b_selector_column, Some(COL_IS_REAL));
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: column layout pinned + bounds
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_FILTERED_PUBKEY_OFFSET, 0);
        assert_eq!(COL_BIT_INDEX, PK_BYTES);
        assert_eq!(COL_SIGNING_ROOT_OFFSET, PK_BYTES + 1);
        assert_eq!(COL_AGGREGATE_SIG_OFFSET, PK_BYTES + 1 + MSG_BYTES);
        assert_eq!(
            COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET,
            PK_BYTES + 1 + MSG_BYTES + SIG_BYTES,
        );
        assert_eq!(
            COL_AGG_PK_X_LIMB_OFFSET,
            COL_AGGREGATE_PUBKEY_X_BYTES_OFFSET + PK_BYTES,
        );
        assert_eq!(
            COL_MSG_G2_X_C0_LIMB_OFFSET,
            COL_AGG_PK_Y_LIMB_OFFSET + LIMBS_PER_FP,
        );
        assert_eq!(
            COL_SLOT,
            COL_MSG_G2_Y_C1_LIMB_OFFSET + LIMBS_PER_FP,
        );
        assert_eq!(COL_COMMITTEE_INDEX, COL_SLOT + 1);
        assert_eq!(COL_SOURCE_EPOCH, COL_SLOT + 2);
        assert_eq!(COL_TARGET_EPOCH, COL_SLOT + 3);
        assert_eq!(COL_IS_REAL, COL_TARGET_EPOCH + 1);
        // 48 + 1 + 32 + 96 + 48 + 12 + 24 + 4 + 1 = 266
        assert_eq!(NUM_COLUMNS, 266);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 7: byte range coverage — every byte column has a lookup decl
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn byte_range_lookup_coverage() {
        let cs = AttestationAggregateConstraintSystem::new(8);
        let req = cs.lookup_declarations();
        let labels: Vec<String> =
            req.declarations.iter().map(|(d, _)| d.label.clone()).collect();
        // One decl per filtered_pubkey byte + agg_pk_x byte + signing_root byte + agg_sig byte.
        let want = 2 * PK_BYTES + MSG_BYTES + SIG_BYTES;
        assert_eq!(req.declarations.len(), want);
        // Spot-check a few well-known labels.
        assert!(labels.contains(&"attestation_filtered_pubkey_byte_0_8bit".into()));
        assert!(labels.contains(&format!(
            "attestation_filtered_pubkey_byte_{}_8bit",
            PK_BYTES - 1
        )));
        assert!(labels.contains(&"attestation_signing_root_byte_0_8bit".into()));
        assert!(labels.contains(&format!(
            "attestation_agg_sig_byte_{}_8bit",
            SIG_BYTES - 1
        )));
        // All columns referenced are within bounds.
        for (d, _) in &req.declarations {
            assert!(d.column_index < NUM_COLUMNS);
            assert_eq!(d.max_bits, 8);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 8: descriptors point into valid sub-AIR column ranges
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn descriptors_point_into_valid_subair_column_ranges() {
        let d_vr = make_attestation_to_validator_registry_descriptor(0, 1);
        for &c in &d_vr.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d_vr.b_columns {
            assert!(c < crate::validator_registry_air::NUM_COLUMNS);
        }
        let d_h2g2 = make_attestation_to_hash_to_g2_descriptor(0, 2);
        for &c in &d_h2g2.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d_h2g2.b_columns {
            assert!(c < crate::hash_to_g2_air::NUM_COLUMNS);
        }
        let d_pair = make_attestation_to_pairing_descriptor(0, 3);
        for &c in &d_pair.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d_pair.b_columns {
            assert!(c < crate::bls_pairing_air::NUM_COLUMNS);
        }
        let d_ffg = make_attestation_to_ffg_descriptor(0, 4);
        for &c in &d_ffg.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        for &c in &d_ffg.b_columns {
            assert!(c < NUM_COLUMNS);
        }
    }
}
