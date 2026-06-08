//! Sync committee aggregate-signature verification AIR (Phase C1 cap).
//!
//! Composition layer that wires the three sub-AIRs needed for end-to-end
//! sync-committee BLS aggregate-signature verification into one
//! coherent column-/witness-shape:
//!
//!   - [`crate::sync_committee_filter_air`] — filters the 512-pubkey
//!     committee to the participating subset via the participation
//!     bitmap.
//!   - [`crate::hash_to_g2_air`] — turns the 32-byte signing-root
//!     message into the `H(msg) ∈ G2` point.
//!   - [`crate::bls_pairing_air`] — checks the pairing equation
//!     `e(agg_pk, H(msg)) == e(agg_sig, G2_gen)`.
//!
//! This AIR is the *composition* of those three. It does NOT itself
//! prove the pairing equation algebraically — that is delegated to
//! `bls_pairing_air`. What it does:
//!
//!   1. Commits one row per participating pubkey, the message, the
//!      aggregate signature, the aggregate pubkey (`Σ pk_i`), and the
//!      hash-to-G2 output point (`H(msg)`).
//!   2. Exposes the row layout via cross-AIR LogUp descriptors so that
//!      the filter AIR's selected pubkeys, the hash-to-G2 AIR's output
//!      G2 limbs, and the pairing AIR's `(pk, sig, msg_hash)` columns
//!      are all multiset-bound to *this* AIR's columns.
//!
//! Combined with the three sub-AIRs and their internal closures, this
//! gives end-to-end algebraic binding of:
//!
//!     filter(committee, bitmap) → participating subset
//!         → aggregate(participating subset) = agg_pk      (oracle today)
//!     hash_to_g2(msg, dst) = H(msg) ∈ G2                  (sub-AIR)
//!     e(agg_pk, H(msg)) == e(agg_sig, G2_gen)             (sub-AIR)
//!
//! # Row shape
//!
//! Per row this AIR commits:
//!
//!   * `pubkey_bytes[0..48]` — one participating pubkey (filter A side).
//!   * `message[0..32]`      — signing root (constant across the active
//!     rows of one verification; we commit it per-row so the LogUp
//!     descriptor to `hash_to_g2_air` has a uniform 32-byte tuple).
//!   * `agg_sig[0..96]`      — aggregate signature (constant across
//!     active rows of one verification).
//!   * `agg_pk_x_limbs[0..6]`, `agg_pk_y_limbs[0..6]` — aggregate G1
//!     pubkey x/y as 6 BE u64 limbs each (12 limbs total).
//!   * `msg_g2_x_c0_limbs[0..6]`, `msg_g2_x_c1_limbs`, `msg_g2_y_c0_limbs`,
//!     `msg_g2_y_c1_limbs` — `H(msg)` as 4 × 6 = 24 BE u64 limbs.
//!   * `agg_pk_x_bytes[0..48]` — canonical BE Fp byte form of the
//!     aggregate pubkey x-coordinate (matches the
//!     [`crate::bls_pairing_air::COL_PK_X_BYTES_OFFSET`] convention).
//!     This is the representative byte/limb decomp anchor.
//!   * `member_index`        — 0..N position within the participating
//!     subset.
//!   * `is_real`             — selector binary column.
//!
//! Total: 48 + 32 + 96 + 12 + 24 + 48 + 1 + 1 = **262 columns**.
//!
//! # Algebraic constraints (scaffold)
//!
//!   1. `is_real ∈ {0, 1}` — selector binarity.
//!   2. One representative G1 limb decomposition: enforce
//!      `agg_pk_x_limbs[0] = Σ_{k=0..8} agg_pk_x_bytes[k] · 2^(8·(7-k))`.
//!      That is, the most-significant 8 bytes of `agg_pk_x_bytes` form
//!      limb 0 of `agg_pk_x_limbs` (BE convention, matching
//!      [`crate::bls_pairing_air`]). One representative limb gets a
//!      shape constraint; the remaining 5 are deferred to the pairing
//!      AIR's full 6-limb decomposition.
//!
//! Byte range checks (8-bit) are emitted for every byte column via
//! `LookupRequirements`.
//!
//! Total row-local algebraic constraints: **2**.
//!
//! # Cross-AIR linkages
//!
//!   * [`make_sync_sig_to_filter_descriptor`] — A side = this AIR's
//!     `pubkey_bytes` (gated by `IS_REAL`), B side = filter AIR's
//!     `PUBKEY_BYTES` (gated by `BITMAP_BIT`). 48-byte tuple.
//!   * [`make_sync_sig_to_hash_to_g2_descriptor`] — A side = this AIR's
//!     `message` + 24 G2 output limbs (gated by `IS_REAL`), B side =
//!     hash-to-G2 AIR's `MSG` + 24 G2 output limbs (gated by `IS_REAL`).
//!     56-column tuple binding `(msg, H(msg))`.
//!   * [`make_sync_sig_to_pairing_descriptor`] — A side = this AIR's
//!     `agg_pk_bytes` + `agg_sig` + 32 zero-bytes (placeholder for the
//!     `H(msg)` byte form on the pairing side which is still oracle),
//!     gated by `IS_REAL`. B side = pairing AIR's
//!     `(pk_compressed, sig_bytes, msg_hash)` row tuple gated by
//!     `IS_REAL`. 176-byte tuple.
//!
//! All three descriptors are *target-shape* scaffolding: they pin the
//! tuple alignment and selector gating so the LogUp orchestration code
//! can already include them. The descriptors are well-formed in the
//! sense that their column lists are within the bounds of each AIR's
//! NUM_COLUMNS.
//!
//! # Honest scope statement
//!
//! What this AIR proves: per-row binarity and one representative
//! agg-pk byte/limb shape constraint. Everything else — the
//! aggregation `Σ pk_i = agg_pk`, the hash-to-curve, and the pairing
//! equation — is delegated to host-side oracle (for aggregation) or
//! the sub-AIRs (for hash-to-G2 and pairing). The cross-AIR LogUp
//! descriptors are the contract that, once all sub-AIRs are joint-
//! proved together with the closures matching, the entire verification
//! is bound end-to-end.

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

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_PUBKEY_OFFSET: usize = 0;
pub const COL_MESSAGE_OFFSET: usize = COL_PUBKEY_OFFSET + PK_BYTES;
pub const COL_AGG_SIG_OFFSET: usize = COL_MESSAGE_OFFSET + MSG_BYTES;

pub const COL_AGG_PK_X_LIMB_OFFSET: usize = COL_AGG_SIG_OFFSET + SIG_BYTES;
pub const COL_AGG_PK_Y_LIMB_OFFSET: usize = COL_AGG_PK_X_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_MSG_G2_X_C0_LIMB_OFFSET: usize = COL_AGG_PK_Y_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_MSG_G2_X_C1_LIMB_OFFSET: usize = COL_MSG_G2_X_C0_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_MSG_G2_Y_C0_LIMB_OFFSET: usize = COL_MSG_G2_X_C1_LIMB_OFFSET + LIMBS_PER_FP;
pub const COL_MSG_G2_Y_C1_LIMB_OFFSET: usize = COL_MSG_G2_Y_C0_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_AGG_PK_X_BYTES_OFFSET: usize = COL_MSG_G2_Y_C1_LIMB_OFFSET + LIMBS_PER_FP;

pub const COL_MEMBER_INDEX: usize = COL_AGG_PK_X_BYTES_OFFSET + PK_BYTES;
pub const COL_IS_REAL: usize = COL_MEMBER_INDEX + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// Row-local constraints:
///   0: is_real ∈ {0, 1}
///   1: agg_pk_x_limbs[0] = byte-recomposition of agg_pk_x_bytes[0..8]
pub const NUM_ROW_CONSTRAINTS: usize = 2;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SyncCommitteeSigRow {
    pub pubkey: [u8; PK_BYTES],
    pub message: [u8; MSG_BYTES],
    pub agg_sig: [u8; SIG_BYTES],

    pub agg_pk_x: Fp,
    pub agg_pk_y: Fp,

    pub msg_g2_x_c0: Fp,
    pub msg_g2_x_c1: Fp,
    pub msg_g2_y_c0: Fp,
    pub msg_g2_y_c1: Fp,

    /// Canonical BE byte form of `agg_pk_x` (flag bits cleared).
    pub agg_pk_x_bytes: [u8; PK_BYTES],

    pub member_index: u64,
}

#[derive(Clone, Debug, Default)]
pub struct SyncCommitteeSigWitness {
    pub rows: Vec<SyncCommitteeSigRow>,
}

impl SyncCommitteeSigWitness {
    /// Build a witness from a filtered participating-pubkey list, the
    /// signing-root message, and the aggregate signature.
    ///
    /// Host-side does:
    ///   1. Aggregate pubkeys via `aggregate_pubkeys` (sum of G1 points).
    ///   2. Hash `message` (with the POP DST) to G2 via `blst_hash_to_g2`.
    ///   3. Sanity-check `fast_aggregate_verify(filtered, msg, sig, dst)`.
    ///      Returns `None` on host-side verification failure or on any
    ///      decoding error.
    ///
    /// The DST is fixed to the beacon-chain POP variant
    /// (`BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_`).
    pub fn from_aggregate(
        filtered_pubkeys: &[[u8; PK_BYTES]],
        message: [u8; MSG_BYTES],
        aggregate_sig: [u8; SIG_BYTES],
    ) -> Option<Self> {
        if filtered_pubkeys.is_empty() {
            return None;
        }
        let dst: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
        let pk_objs: Vec<PublicKey> = filtered_pubkeys.iter().map(|b| PublicKey(*b)).collect();
        let sig_obj = Signature(aggregate_sig);

        // Host-side verify; refuse to build a witness for a bad triple.
        if !fast_aggregate_verify(&pk_objs, &message, &sig_obj, dst) {
            return None;
        }

        // Aggregate G1: Σ pk_i (compressed).
        let agg_pk_compressed = aggregate_pubkeys(&pk_objs).ok()?.0;
        let agg_pk_aff = crate::pairing::G1Affine::from_bytes(&agg_pk_compressed).ok()?;
        if agg_pk_aff.infinity {
            return None;
        }
        let mut agg_pk_x_bytes = agg_pk_compressed;
        agg_pk_x_bytes[0] &= 0x1f; // clear flag bits → canonical Fp byte form

        // Hash-to-G2 oracle for H(msg). Use blst directly mirroring
        // [`crate::bls_sig::hash_to_g2_affine`] + `compress_g2`.
        let msg_g2_compressed: [u8; SIG_BYTES] = unsafe {
            let mut p = blst::blst_p2::default();
            blst::blst_hash_to_g2(
                &mut p,
                message.as_ptr(),
                message.len(),
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

        let row = SyncCommitteeSigRow {
            // Row 0 carries the first participating pubkey + the shared
            // message / signature / agg_pk / msg_g2 fields. Later rows
            // (one per additional participating pubkey) carry the per-
            // pubkey value plus the same shared fields.
            pubkey: filtered_pubkeys[0],
            message,
            agg_sig: aggregate_sig,
            agg_pk_x: agg_pk_aff.x,
            agg_pk_y: agg_pk_aff.y,
            msg_g2_x_c0: msg_g2_aff.x.c0,
            msg_g2_x_c1: msg_g2_aff.x.c1,
            msg_g2_y_c0: msg_g2_aff.y.c0,
            msg_g2_y_c1: msg_g2_aff.y.c1,
            agg_pk_x_bytes,
            member_index: 0,
        };
        let mut rows = vec![row];
        for (i, pk) in filtered_pubkeys.iter().enumerate().skip(1) {
            let mut next = rows[0].clone();
            next.pubkey = *pk;
            next.member_index = i as u64;
            rows.push(next);
        }
        Some(Self { rows })
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &SyncCommitteeSigWitness,
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
            columns[COL_PUBKEY_OFFSET + k][r] =
                Scalar::from_u64(row.pubkey[k] as u64, curve);
            columns[COL_AGG_PK_X_BYTES_OFFSET + k][r] =
                Scalar::from_u64(row.agg_pk_x_bytes[k] as u64, curve);
        }
        for k in 0..MSG_BYTES {
            columns[COL_MESSAGE_OFFSET + k][r] =
                Scalar::from_u64(row.message[k] as u64, curve);
        }
        for k in 0..SIG_BYTES {
            columns[COL_AGG_SIG_OFFSET + k][r] =
                Scalar::from_u64(row.agg_sig[k] as u64, curve);
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

        columns[COL_MEMBER_INDEX][r] = Scalar::from_u64(row.member_index, curve);
        columns[COL_IS_REAL][r] = one.clone();
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct SyncCommitteeSigConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl SyncCommitteeSigConstraintSystem {
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

impl VmConstraintSystem for SyncCommitteeSigConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec!["is_real_binary".into(), "agg_pk_x_limb_0_decomp".into()]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _num_rows: usize) -> Vec<Vec<Scalar>> {
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

        // 1: agg_pk_x_limbs[0] = Σ_{k=0..8} agg_pk_x_bytes[k] · 2^(8·(7-k)).
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for k in 0..BYTES_PER_LIMB {
                    let b = &columns[COL_AGG_PK_X_BYTES_OFFSET + k][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(be_limb_byte_weight(k), curve)));
                }
                c[r] = columns[COL_AGG_PK_X_LIMB_OFFSET][r].sub(&sum);
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

        let v = &col_evals[COL_IS_REAL];
        let c0 = v.mul(&v.sub(&one));

        let mut sum = Scalar::zero(curve);
        for k in 0..BYTES_PER_LIMB {
            sum = sum.add(
                &col_evals[COL_AGG_PK_X_BYTES_OFFSET + k]
                    .mul(&Scalar::from_u64(be_limb_byte_weight(k), curve)),
            );
        }
        let c1 = col_evals[COL_AGG_PK_X_LIMB_OFFSET].sub(&sum);

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
            let b = &col_coeffs[COL_AGG_PK_X_BYTES_OFFSET + k];
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
                    label: format!("sync_sig_pubkey_byte_{}_8bit", k),
                    column_index: COL_PUBKEY_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_sig_agg_pk_x_byte_{}_8bit", k),
                    column_index: COL_AGG_PK_X_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MSG_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_sig_msg_byte_{}_8bit", k),
                    column_index: COL_MESSAGE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SIG_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_sig_agg_sig_byte_{}_8bit", k),
                    column_index: COL_AGG_SIG_OFFSET + k,
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

/// Bind this AIR's per-row participating pubkey to the filter AIR's
/// selected pubkey rows.
///
/// A side = this AIR (gated by `IS_REAL`).
/// B side = filter AIR (gated by `BITMAP_BIT`).
///
/// 48-byte tuple shape.
pub fn make_sync_sig_to_filter_descriptor(
    sync_sig_layer_index: usize,
    filter_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sync_committee_filter_air as filter;
    let a_columns: Vec<usize> = (0..PK_BYTES).map(|k| COL_PUBKEY_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..PK_BYTES).map(|k| filter::COL_PUBKEY_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_sig_to_filter_v1".into(),
        a_layer_index: sync_sig_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: filter_layer_index,
        b_columns,
        b_selector_column: Some(filter::COL_BITMAP_BIT),
    }
}

/// Bind this AIR's `(message, msg_g2_limbs[0..24])` tuple to the
/// hash-to-G2 AIR's `(MSG, OUT_*_LIMBS)` tuple.
///
/// 32 + 24 = **56-column tuple**.
pub fn make_sync_sig_to_hash_to_g2_descriptor(
    sync_sig_layer_index: usize,
    hash_to_g2_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::hash_to_g2_air as h2g2;

    let mut a_columns: Vec<usize> = Vec::with_capacity(MSG_BYTES + 4 * LIMBS_PER_FP);
    for k in 0..MSG_BYTES {
        a_columns.push(COL_MESSAGE_OFFSET + k);
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

    let mut b_columns: Vec<usize> = Vec::with_capacity(MSG_BYTES + 4 * LIMBS_PER_FP);
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
        label: "sync_sig_to_hash_to_g2_v1".into(),
        a_layer_index: sync_sig_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: hash_to_g2_layer_index,
        b_columns,
        b_selector_column: Some(h2g2::COL_IS_REAL),
    }
}

/// Bind this AIR's `(agg_pk_x_bytes, agg_sig, message)` to the pairing
/// AIR's `(pk_compressed, sig_bytes, msg_hash)`.
///
/// Note: in the current scaffold the pairing AIR commits the row's
/// `pk_compressed` as one *individual* pubkey rather than the aggregate.
/// This descriptor is named after the *target-shape* alignment for the
/// future extension where the pairing AIR exposes a dedicated
/// `agg_pk_compressed` column. For now, the A-side reuses
/// `agg_pk_x_bytes` (48 bytes, canonical form) so the closure is
/// well-defined; once the pairing AIR adds an `agg_pk_compressed`
/// column the A-side will switch to that.
///
/// Tuple shape: 48 (agg_pk byte form) + 96 (agg_sig) + 32 (msg) = **176 bytes**.
pub fn make_sync_sig_to_pairing_descriptor(
    sync_sig_layer_index: usize,
    bls_pairing_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as bp;

    let mut a_columns: Vec<usize> = Vec::with_capacity(PK_BYTES + SIG_BYTES + MSG_BYTES);
    for k in 0..PK_BYTES {
        a_columns.push(COL_AGG_PK_X_BYTES_OFFSET + k);
    }
    for k in 0..SIG_BYTES {
        a_columns.push(COL_AGG_SIG_OFFSET + k);
    }
    for k in 0..MSG_BYTES {
        a_columns.push(COL_MESSAGE_OFFSET + k);
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
        label: "sync_sig_to_pairing_v1".into(),
        a_layer_index: sync_sig_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bls_pairing_layer_index,
        b_columns,
        b_selector_column: Some(bp::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bls_sig::{aggregate_sigs, SecretKey};

    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    /// Produce a real (pubkeys, msg, agg_sig) triple by signing one
    /// message with several deterministic keys.
    fn real_aggregate(n: u8) -> (Vec<[u8; PK_BYTES]>, [u8; MSG_BYTES], [u8; SIG_BYTES]) {
        let sks: Vec<SecretKey> = (1..=n).map(SecretKey::from_u8_seed).collect();
        let pks: Vec<[u8; 48]> = sks.iter().map(|s| s.public_key().0).collect();
        let msg_str = b"sync-committee-sig-air-test";
        let msg_hash = crate::keccak::keccak256(msg_str);
        let sigs: Vec<_> = sks.iter().map(|s| s.sign(&msg_hash, POP_DST)).collect();
        let agg_sig = aggregate_sigs(&sigs).unwrap().0;
        (pks, msg_hash, agg_sig)
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: witness builds from a real BLS aggregate
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_builds_from_real_aggregate() {
        let (pks, msg, agg_sig) = real_aggregate(4);
        let w = SyncCommitteeSigWitness::from_aggregate(&pks, msg, agg_sig)
            .expect("real aggregate must build a witness");
        assert_eq!(w.rows.len(), 4, "one row per participating pubkey");
        // First row carries pubkey 0; later rows carry pubkeys 1..3.
        assert_eq!(w.rows[0].pubkey, pks[0]);
        assert_eq!(w.rows[3].pubkey, pks[3]);
        // Aggregate-pk canonical bytes round-trip through Fp.
        let row0 = &w.rows[0];
        let want_x = Fp::from_bytes_be(&row0.agg_pk_x_bytes).expect("agg_pk_x_bytes is canonical");
        assert_eq!(row0.agg_pk_x.limbs, want_x.limbs);
        // Member indices count up.
        for (i, r) in w.rows.iter().enumerate() {
            assert_eq!(r.member_index as usize, i);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: tampered aggregate signature is rejected
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn witness_rejects_tampered_aggregate() {
        let (pks, msg, mut agg_sig) = real_aggregate(3);
        agg_sig[20] ^= 0x01;
        assert!(
            SyncCommitteeSigWitness::from_aggregate(&pks, msg, agg_sig).is_none(),
            "tampered agg_sig must fail fast_aggregate_verify",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: honest witness satisfies all constraints
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let (pks, msg, agg_sig) = real_aggregate(2);
        let w = SyncCommitteeSigWitness::from_aggregate(&pks, msg, agg_sig).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = SyncCommitteeSigConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(val.is_zero(), "constraint {} row {} not zero", i, r);
            }
        }
    }

    #[test]
    fn limb_decomp_fires_on_tampered_byte() {
        let (pks, msg, agg_sig) = real_aggregate(2);
        let w = SyncCommitteeSigWitness::from_aggregate(&pks, msg, agg_sig).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;
        // Tamper byte 0 (MSB of limb 0) of agg_pk_x_bytes by adding 1.
        let original = cols[COL_AGG_PK_X_BYTES_OFFSET][0].clone();
        cols[COL_AGG_PK_X_BYTES_OFFSET][0] = original.add(&Scalar::one(curve));
        let cs = SyncCommitteeSigConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[1][0].is_zero(),
            "tampering agg_pk_x_bytes[0] must fire limb-0 decomp constraint",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: descriptor well-formedness
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn filter_descriptor_well_formed() {
        let d = make_sync_sig_to_filter_descriptor(0, 1);
        assert_eq!(d.label, "sync_sig_to_filter_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), PK_BYTES);
        assert_eq!(d.b_columns.len(), PK_BYTES);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::sync_committee_filter_air::COL_BITMAP_BIT),
        );
        assert_eq!(d.a_columns[0], COL_PUBKEY_OFFSET);
        assert_eq!(
            d.b_columns[0],
            crate::sync_committee_filter_air::COL_PUBKEY_OFFSET,
        );
    }

    #[test]
    fn hash_to_g2_descriptor_well_formed() {
        let d = make_sync_sig_to_hash_to_g2_descriptor(0, 2);
        assert_eq!(d.label, "sync_sig_to_hash_to_g2_v1");
        assert_eq!(d.a_columns.len(), MSG_BYTES + 4 * LIMBS_PER_FP);
        assert_eq!(d.b_columns.len(), MSG_BYTES + 4 * LIMBS_PER_FP);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::hash_to_g2_air::COL_IS_REAL),
        );
        // A-side message prefix, then 24 G2 limb columns.
        assert_eq!(d.a_columns[0], COL_MESSAGE_OFFSET);
        assert_eq!(d.a_columns[MSG_BYTES], COL_MSG_G2_X_C0_LIMB_OFFSET);
        // B-side message prefix lands on hash_to_g2's MSG offset.
        assert_eq!(d.b_columns[0], crate::hash_to_g2_air::COL_MSG_OFFSET);
        assert_eq!(
            d.b_columns[MSG_BYTES],
            crate::hash_to_g2_air::COL_OUT_X_C0_LIMB_OFFSET,
        );
    }

    #[test]
    fn pairing_descriptor_well_formed() {
        let d = make_sync_sig_to_pairing_descriptor(0, 3);
        assert_eq!(d.label, "sync_sig_to_pairing_v1");
        let expected_len = PK_BYTES + SIG_BYTES + MSG_BYTES;
        assert_eq!(d.a_columns.len(), expected_len);
        assert_eq!(d.b_columns.len(), expected_len);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::bls_pairing_air::COL_IS_REAL),
        );
        // A-side blocks: agg_pk_bytes | agg_sig | msg.
        assert_eq!(d.a_columns[0], COL_AGG_PK_X_BYTES_OFFSET);
        assert_eq!(d.a_columns[PK_BYTES], COL_AGG_SIG_OFFSET);
        assert_eq!(d.a_columns[PK_BYTES + SIG_BYTES], COL_MESSAGE_OFFSET);
        // B-side blocks land on the matching pairing-air columns.
        assert_eq!(d.b_columns[0], crate::bls_pairing_air::COL_PK_X_BYTES_OFFSET);
        assert_eq!(d.b_columns[PK_BYTES], crate::bls_pairing_air::COL_SIG_BYTES_OFFSET);
        assert_eq!(
            d.b_columns[PK_BYTES + SIG_BYTES],
            crate::bls_pairing_air::COL_MSG_HASH_OFFSET,
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: column layout pinned + bounds within sub-AIR widths
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_PUBKEY_OFFSET, 0);
        assert_eq!(COL_MESSAGE_OFFSET, 48);
        assert_eq!(COL_AGG_SIG_OFFSET, 48 + 32);
        assert_eq!(COL_AGG_PK_X_LIMB_OFFSET, 48 + 32 + 96);
        assert_eq!(COL_MSG_G2_X_C0_LIMB_OFFSET, COL_AGG_PK_Y_LIMB_OFFSET + LIMBS_PER_FP);
        assert_eq!(
            COL_AGG_PK_X_BYTES_OFFSET,
            COL_MSG_G2_Y_C1_LIMB_OFFSET + LIMBS_PER_FP,
        );
        assert_eq!(COL_MEMBER_INDEX, COL_AGG_PK_X_BYTES_OFFSET + PK_BYTES);
        assert_eq!(COL_IS_REAL, COL_MEMBER_INDEX + 1);
        assert_eq!(NUM_COLUMNS, COL_IS_REAL + 1);
        // 48 + 32 + 96 + 12 + 24 + 48 + 1 + 1 = 262
        assert_eq!(NUM_COLUMNS, 262);
    }

    /// Integration smoke test: the three descriptors produced above
    /// reference column indices that are within the bounds of each
    /// sub-AIR's `NUM_COLUMNS`. This proves the composition of
    /// filter + hash_to_g2 + pairing AIRs through this AIR's tuples is
    /// **structurally valid** at the layer-chain wiring level.
    #[test]
    fn descriptors_point_into_valid_subair_column_ranges() {
        // Filter side.
        let d_filter = make_sync_sig_to_filter_descriptor(0, 1);
        for &c in &d_filter.b_columns {
            assert!(c < crate::sync_committee_filter_air::NUM_COLUMNS);
        }
        for &c in &d_filter.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        // Hash-to-G2 side.
        let d_h2g2 = make_sync_sig_to_hash_to_g2_descriptor(0, 2);
        for &c in &d_h2g2.b_columns {
            assert!(c < crate::hash_to_g2_air::NUM_COLUMNS);
        }
        for &c in &d_h2g2.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        // Pairing side.
        let d_pair = make_sync_sig_to_pairing_descriptor(0, 3);
        for &c in &d_pair.b_columns {
            assert!(c < crate::bls_pairing_air::NUM_COLUMNS);
        }
        for &c in &d_pair.a_columns {
            assert!(c < NUM_COLUMNS);
        }
        // Sanity: filter-side selector indices are also in range.
        assert!(
            d_filter.b_selector_column.unwrap()
                < crate::sync_committee_filter_air::NUM_COLUMNS,
        );
    }
}
