//! Sync-committee aggregate signature **composition AIR** (round 1).
//!
//! # Purpose
//!
//! Composes the four sync-committee aggregate-signature sub-AIRs into
//! a single 4-row multi-phase gadget proving the beacon-chain pipeline
//!
//! ```text
//!     committee + bitmap ── filter ──▶ filtered_pubkeys
//!     msg              ── hash_to_g2 ──▶ H(msg) ∈ G2
//!     filtered_pubkeys ── aggregate  ──▶ aggregate_pubkey ∈ G1
//!     (aggregate_pubkey, H(msg), aggregate_sig) ── pairing ──▶ pairing_result ∈ {0,1}
//! ```
//!
//! end-to-end with a **single per-instance witness**.
//!
//! Each [`SyncCommitteeAggregateComposerWitness::from_aggregate`] call
//! emits **[`ROWS_PER_INSTANCE`] = 4 consecutive rows**, one per phase:
//!
//! | row | `phase_index` | selector              | meaning                                   |
//! |----:|--------------:|-----------------------|-------------------------------------------|
//! |  0  |             0 | `IS_FILTER_PHASE`     | bitmap × committee → filtered pubkey set  |
//! |  1  |             1 | `IS_HASH_TO_G2_PHASE` | msg → H(msg) ∈ G2                         |
//! |  2  |             2 | `IS_AGG_PUBKEY_PHASE` | filtered_pubkeys → aggregate_pubkey ∈ G1  |
//! |  3  |             3 | `IS_PAIRING_PHASE`    | (agg_pk, H(msg), agg_sig) → pairing check |
//!
//! All four rows commit the same `msg`, `aggregate_pubkey`,
//! `aggregate_sig`, `hash_to_g2_output`, `pairing_result`,
//! `participation_count`, and `threshold`; only the phase selector and
//! `phase_index` change between rows. The cross-row β-RLC shifted
//! constraint binds the `msg` slice between adjacent rows so the
//! same message is processed across all four phase rows of an instance.
//!
//! # What is committed
//!
//! Per row:
//!
//!   * `msg[0..MSG_LEN]`               — signing root, 32 bytes.
//!   * `aggregate_pubkey[0..PK_LEN]`   — aggregated G1 pubkey, 48 bytes.
//!   * `aggregate_sig[0..SIG_LEN]`     — aggregated G2 signature, 96 bytes.
//!   * `hash_to_g2_output[0..G2_LEN]`  — compressed H(msg) ∈ G2, 96 bytes.
//!   * `pairing_result`                — binary {0, 1}.
//!   * `participation_count` (integer) + LE bytes (4 cols).
//!   * `threshold`           (integer) + LE bytes (4 cols).
//!   * `phase_index` ∈ {0, 1, 2, 3} + `phase_index_byte` (range-checked copy).
//!   * `is_filter_phase, is_hash_to_g2_phase, is_agg_pubkey_phase,
//!      is_pairing_phase` — 4 binary phase selectors.
//!   * `is_real` — global "real instance" gate.
//!
//! Heavy algebraic intermediates are owned by the four sub-AIRs; this
//! composition AIR only commits the per-phase outputs and threads them
//! into the four [`make_sync_composer_to_*_descriptor`] cross-AIR LogUp
//! links emitted below.
//!
//! # Algebraic constraints
//!
//!   Row-local ([`NUM_ROW_CONSTRAINTS`] = 11):
//!     0:   `is_real ∈ {0, 1}`.
//!     1..4: 4 phase-selector binarities.
//!     5:   `Σ is_phase_k = is_real`  (phase sum).
//!     6:   `phase_index = Σ k · is_phase_k`  (LE-byte decomp from selectors).
//!     7:   `phase_index = phase_index_byte`.
//!     8:   `pairing_result ∈ {0, 1}`.
//!     9:   `participation_count = Σ_k 256^k · participation_count_byte[k]`
//!          (LE byte decomposition).
//!    10:   `threshold           = Σ_k 256^k · threshold_byte[k]`
//!          (LE byte decomposition).
//!
//!   Cross-row ([`NUM_SHIFTED`] = 1):
//!     0:   β-RLC byte-bundle equality binding the `msg` slice on row
//!          `r` to the `msg` slice on row `r+1`, gated by
//!          `is_real(r) · is_real(r+1)`.
//!
//! Range checks: every byte column (msg, aggregate_pubkey,
//! aggregate_sig, hash_to_g2_output, participation_count_bytes,
//! threshold_bytes, phase_index_byte) is 8-bit-bounded via
//! [`LookupRequirements`].
//!
//! # Cross-AIR linkages
//!
//! Four descriptors are emitted, one per phase:
//!
//!   * [`make_sync_composer_to_filter_descriptor`] — phase 0 →
//!     [`crate::sync_committee_filter_air`] (binds `aggregate_pubkey`
//!     anchor bytes to filter-AIR pubkey columns).
//!   * [`make_sync_composer_to_hash_to_g2_descriptor`] — phase 1 →
//!     [`crate::hash_to_g2_air`] (binds the 32-byte msg to the h2g2 msg
//!     prefix).
//!   * [`make_sync_composer_to_sync_sig_descriptor`] — phase 2 →
//!     [`crate::sync_committee_sig_air`] (binds aggregate_pubkey bytes
//!     to sync-sig AIR's `agg_pk_x_bytes` columns).
//!   * [`make_sync_composer_to_pairing_descriptor`] — phase 3 →
//!     [`crate::bls_pairing_air`] (binds aggregate_pubkey + sig + msg
//!     hash to pairing AIR's pk/sig/msg byte columns).

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const MSG_LEN: usize = 32;
pub const PK_LEN: usize = 48;
pub const SIG_LEN: usize = 96;
pub const G2_LEN: usize = 96;

/// Number of LE bytes used to commit the `u32` participation/threshold
/// counters.
pub const COUNTER_BYTES: usize = 4;

/// Rows per instance — one row per phase.
pub const ROWS_PER_INSTANCE: usize = 4;
pub const NUM_PHASE_SELECTORS: usize = 4;

// Phase index labels (the `phase_index` integer value per row).
pub const PHASE_FILTER: u8 = 0;
pub const PHASE_HASH_TO_G2: u8 = 1;
pub const PHASE_AGG_PUBKEY: u8 = 2;
pub const PHASE_PAIRING: u8 = 3;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_MSG_OFFSET: usize = 0;
pub const COL_AGG_PUBKEY_OFFSET: usize = COL_MSG_OFFSET + MSG_LEN;
pub const COL_AGG_SIG_OFFSET: usize = COL_AGG_PUBKEY_OFFSET + PK_LEN;
pub const COL_HASH_TO_G2_OFFSET: usize = COL_AGG_SIG_OFFSET + SIG_LEN;
pub const COL_PAIRING_RESULT: usize = COL_HASH_TO_G2_OFFSET + G2_LEN;
pub const COL_PARTICIPATION_COUNT: usize = COL_PAIRING_RESULT + 1;
pub const COL_PARTICIPATION_COUNT_BYTES_OFFSET: usize = COL_PARTICIPATION_COUNT + 1;
pub const COL_THRESHOLD: usize =
    COL_PARTICIPATION_COUNT_BYTES_OFFSET + COUNTER_BYTES;
pub const COL_THRESHOLD_BYTES_OFFSET: usize = COL_THRESHOLD + 1;
pub const COL_PHASE_INDEX: usize = COL_THRESHOLD_BYTES_OFFSET + COUNTER_BYTES;
pub const COL_PHASE_INDEX_BYTE: usize = COL_PHASE_INDEX + 1;
pub const COL_IS_FILTER_PHASE: usize = COL_PHASE_INDEX_BYTE + 1;
pub const COL_IS_HASH_TO_G2_PHASE: usize = COL_IS_FILTER_PHASE + 1;
pub const COL_IS_AGG_PUBKEY_PHASE: usize = COL_IS_HASH_TO_G2_PHASE + 1;
pub const COL_IS_PAIRING_PHASE: usize = COL_IS_AGG_PUBKEY_PHASE + 1;
pub const COL_IS_REAL: usize = COL_IS_PAIRING_PHASE + 1;

// ─── Sub-AIR mirror columns (task #190) ───────────────────────────────
//
// The cross-AIR LogUp anchors used by
// `integration_sync_committee_aggregate_joint_prove` bind a single
// composer column to a single sub-AIR column. The straightforward
// "raw aggregate pubkey byte 0" candidate is NOT a value the
// `sync_committee_filter_air` or `sync_committee_sig_air` AIRs commit
// on their B-side:
//   * filter AIR commits per-member pubkey bytes (each committee
//     member, not the aggregate);
//   * sync_sig AIR commits `agg_pk_x_bytes[0]`, which is the
//     aggregate compressed byte 0 **with the 3 IETF flag bits
//     masked off** (`& 0x1f`).
// The cross-AIR LogUp multiset-equality check would reject every
// composer tuple that isn't present in the corresponding sub-AIR
// table, which is exactly the multiset mismatch the slow integration
// test was hitting under BLS48-581.
//
// We add two dedicated mirror columns here that hold the **same
// scalar values** the corresponding sub-AIRs commit, derived
// host-side from the same oracle inputs:
//   * `COL_FIRST_MEMBER_PK_BYTE0_MIRROR`: byte 0 of the first
//     committee member's compressed pubkey (matches filter AIR's
//     `pubkey[0]` on its row 0).
//   * `COL_AGG_PK_X_BYTE0_MIRROR`: `aggregate_pubkey[0] & 0x1f`
//     (matches sync_sig AIR's `agg_pk_x_bytes[0]`).
//
// These mirror columns are pure witness data — no row-local
// algebraic constraint is attached. Their soundness flows from the
// cross-AIR LogUp closure: any prover that tampers with them breaks
// the corresponding closure-equality. Same pattern as the
// hash_to_curve composer (task #189).
pub const COL_FIRST_MEMBER_PK_BYTE0_MIRROR: usize = COL_IS_REAL + 1;
pub const COL_AGG_PK_X_BYTE0_MIRROR: usize = COL_FIRST_MEMBER_PK_BYTE0_MIRROR + 1;

pub const NUM_COLUMNS: usize = COL_AGG_PK_X_BYTE0_MIRROR + 1;

/// Row-local constraint count.
///   0: is_real binary
///   1..4: 4 phase-selector binarities
///   5: phase-sum = is_real
///   6: phase_index = Σ k·is_phase_k
///   7: phase_index = phase_index_byte
///   8: pairing_result binary
///   9: participation_count LE byte decomp
///  10: threshold           LE byte decomp
pub const NUM_ROW_CONSTRAINTS: usize = 11;

/// Cross-row (shifted) constraint count.
///   0: msg β-RLC equality between adjacent rows of the same instance,
///      gated by `is_real(r) * is_real(r+1)`.
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SyncCommitteeAggregateComposerRow {
    pub msg: [u8; MSG_LEN],
    pub aggregate_pubkey: [u8; PK_LEN],
    pub aggregate_sig: [u8; SIG_LEN],
    pub hash_to_g2_output: [u8; G2_LEN],
    pub pairing_result: u8,
    pub participation_count: u32,
    pub threshold: u32,
    pub phase_index: u8,
    /// Task #190 sub-AIR mirror: byte 0 of the first committee
    /// member's compressed pubkey (matches
    /// [`crate::sync_committee_filter_air::COL_PUBKEY_OFFSET`] on its
    /// row 0). Zero when the filter input is empty.
    pub first_member_pk_byte0: u8,
    /// Task #190 sub-AIR mirror: `aggregate_pubkey[0] & 0x1f` —
    /// the IETF-flag-bits-masked aggregate compressed byte 0,
    /// matching [`crate::sync_committee_sig_air::COL_AGG_PK_X_BYTES_OFFSET`].
    pub agg_pk_x_byte0: u8,
}

#[derive(Clone, Debug, Default)]
pub struct SyncCommitteeAggregateComposerWitness {
    pub rows: Vec<SyncCommitteeAggregateComposerRow>,
}

impl SyncCommitteeAggregateComposerWitness {
    /// Build a [`ROWS_PER_INSTANCE`]-row witness for the sync-committee
    /// aggregate pipeline.
    ///
    /// Host-side oracle: aggregates the supplied `filtered_pubkeys` via
    /// blst, computes `H(msg) ∈ G2` via blst's `hash_to_g2`, and runs
    /// `fast_aggregate_verify` (with the POP DST) to obtain the
    /// `pairing_result` boolean.
    ///
    /// The threshold defaults to ⌈2/3 · 512⌉ (the IETF sync-committee
    /// supermajority cutoff). The witness commits the integer
    /// `participation_count` plus its 4-byte LE decomposition for
    /// algebraic range / equality checks.
    ///
    /// All four rows share the same `msg`, `aggregate_pubkey`,
    /// `aggregate_sig`, `hash_to_g2_output`, `pairing_result`,
    /// `participation_count`, and `threshold`; only the phase selector
    /// and `phase_index` differ.
    pub fn from_aggregate(
        msg: [u8; MSG_LEN],
        filtered_pubkeys: &[[u8; PK_LEN]],
        aggregate_sig: [u8; SIG_LEN],
        participation_count: u32,
    ) -> Self {
        use crate::bls_sig as bs;

        // Beacon-chain sync-committee BLS DST (POP variant).
        const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
        // Sync committee size / supermajority threshold (host-side
        // default; the algebraic constraint only checks LE decomp).
        const SYNC_COMMITTEE_SIZE: u32 = 512;
        let threshold: u32 =
            ((SYNC_COMMITTEE_SIZE as u64 * 2 + 2) / 3) as u32;

        // Host-side aggregate pubkey via blst (returns identity-pubkey
        // bytes for empty input — this is acceptable as long as the
        // sub-AIR enforces non-empty / membership).
        let aggregate_pubkey: [u8; PK_LEN] = if filtered_pubkeys.is_empty() {
            [0u8; PK_LEN]
        } else {
            let pks: Vec<bs::PublicKey> =
                filtered_pubkeys.iter().map(|b| bs::PublicKey(*b)).collect();
            match bs::aggregate_pubkeys(&pks) {
                Ok(pk) => pk.0,
                Err(_) => [0u8; PK_LEN],
            }
        };

        // Host-side hash-to-G2 via blst (compressed 96 bytes).
        let hash_to_g2_output: [u8; G2_LEN] = {
            let aff = bs::hash_to_g2_affine(&msg, POP_DST);
            let mut out = [0u8; G2_LEN];
            unsafe { blst::blst_p2_affine_compress(out.as_mut_ptr(), &aff); }
            out
        };

        // Host-side pairing result.
        let pairing_result: u8 = if filtered_pubkeys.is_empty() {
            0
        } else {
            let pks: Vec<bs::PublicKey> =
                filtered_pubkeys.iter().map(|b| bs::PublicKey(*b)).collect();
            let sig = bs::Signature(aggregate_sig);
            if bs::fast_aggregate_verify(&pks, &msg, &sig, POP_DST) {
                1
            } else {
                0
            }
        };

        // Task #190 mirror values: derive host-side from the same
        // oracle inputs the sub-AIRs use.
        let first_member_pk_byte0: u8 = if filtered_pubkeys.is_empty() {
            0
        } else {
            filtered_pubkeys[0][0]
        };
        let agg_pk_x_byte0: u8 = aggregate_pubkey[0] & 0x1f;

        let mut rows = Vec::with_capacity(ROWS_PER_INSTANCE);
        for phase in 0..ROWS_PER_INSTANCE {
            rows.push(SyncCommitteeAggregateComposerRow {
                msg,
                aggregate_pubkey,
                aggregate_sig,
                hash_to_g2_output,
                pairing_result,
                participation_count,
                threshold,
                phase_index: phase as u8,
                first_member_pk_byte0,
                agg_pk_x_byte0,
            });
        }
        Self { rows }
    }

    /// Append a raw row (used by tampering / fixture tests).
    pub fn push_raw(&mut self, row: SyncCommitteeAggregateComposerRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &SyncCommitteeAggregateComposerWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..MSG_LEN {
            columns[COL_MSG_OFFSET + k][r] =
                Scalar::from_u64(row.msg[k] as u64, curve);
        }
        for k in 0..PK_LEN {
            columns[COL_AGG_PUBKEY_OFFSET + k][r] =
                Scalar::from_u64(row.aggregate_pubkey[k] as u64, curve);
        }
        for k in 0..SIG_LEN {
            columns[COL_AGG_SIG_OFFSET + k][r] =
                Scalar::from_u64(row.aggregate_sig[k] as u64, curve);
        }
        for k in 0..G2_LEN {
            columns[COL_HASH_TO_G2_OFFSET + k][r] =
                Scalar::from_u64(row.hash_to_g2_output[k] as u64, curve);
        }
        columns[COL_PAIRING_RESULT][r] =
            Scalar::from_u64(row.pairing_result as u64, curve);
        columns[COL_PARTICIPATION_COUNT][r] =
            Scalar::from_u64(row.participation_count as u64, curve);
        let pc_bytes = row.participation_count.to_le_bytes();
        for k in 0..COUNTER_BYTES {
            columns[COL_PARTICIPATION_COUNT_BYTES_OFFSET + k][r] =
                Scalar::from_u64(pc_bytes[k] as u64, curve);
        }
        columns[COL_THRESHOLD][r] =
            Scalar::from_u64(row.threshold as u64, curve);
        let th_bytes = row.threshold.to_le_bytes();
        for k in 0..COUNTER_BYTES {
            columns[COL_THRESHOLD_BYTES_OFFSET + k][r] =
                Scalar::from_u64(th_bytes[k] as u64, curve);
        }
        columns[COL_PHASE_INDEX][r] =
            Scalar::from_u64(row.phase_index as u64, curve);
        columns[COL_PHASE_INDEX_BYTE][r] =
            Scalar::from_u64(row.phase_index as u64, curve);
        columns[COL_IS_FILTER_PHASE][r] = if row.phase_index == PHASE_FILTER {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_HASH_TO_G2_PHASE][r] = if row.phase_index == PHASE_HASH_TO_G2 {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_AGG_PUBKEY_PHASE][r] = if row.phase_index == PHASE_AGG_PUBKEY {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_PAIRING_PHASE][r] = if row.phase_index == PHASE_PAIRING {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_REAL][r] = one.clone();
        // Task #190 mirror columns: same value committed on every row
        // (only the phase selector differs across the 4 phase rows).
        columns[COL_FIRST_MEMBER_PK_BYTE0_MIRROR][r] =
            Scalar::from_u64(row.first_member_pk_byte0 as u64, curve);
        columns[COL_AGG_PK_X_BYTE0_MIRROR][r] =
            Scalar::from_u64(row.agg_pk_x_byte0 as u64, curve);
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

pub struct SyncCommitteeAggregateComposerConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl SyncCommitteeAggregateComposerConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Fixed β used in the cross-row `msg` β-RLC bundle (same convention
/// as the other composition AIRs).
fn beta_rlc(curve: CurveType) -> Scalar {
    Scalar::from_u64(7, curve)
}

/// Powers of 256 (the LE-byte recomposition base) as field constants.
fn byte_powers(curve: CurveType) -> [Scalar; COUNTER_BYTES] {
    let mut out = [Scalar::zero(curve), Scalar::zero(curve), Scalar::zero(curve), Scalar::zero(curve)];
    let mut acc = Scalar::one(curve);
    let base = Scalar::from_u64(256, curve);
    for k in 0..COUNTER_BYTES {
        out[k] = acc.clone();
        acc = acc.mul(&base);
    }
    out
}

impl VmConstraintSystem for SyncCommitteeAggregateComposerConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_filter_phase_binary".into(),
            "is_hash_to_g2_phase_binary".into(),
            "is_agg_pubkey_phase_binary".into(),
            "is_pairing_phase_binary".into(),
            "phase_sum_eq_is_real".into(),
            "phase_index_eq_weighted_selector_sum".into(),
            "phase_index_eq_phase_index_byte".into(),
            "pairing_result_binary".into(),
            "participation_count_le_byte_decomp".into(),
            "threshold_le_byte_decomp".into(),
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
        let three = Scalar::from_u64(3, curve);
        let pow = byte_powers(curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0..4: binarities.
        for col in [
            COL_IS_REAL,
            COL_IS_FILTER_PHASE,
            COL_IS_HASH_TO_G2_PHASE,
            COL_IS_AGG_PUBKEY_PHASE,
            COL_IS_PAIRING_PHASE,
        ] {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[col][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 5: phase sum = is_real.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let sum = columns[COL_IS_FILTER_PHASE][r]
                    .add(&columns[COL_IS_HASH_TO_G2_PHASE][r])
                    .add(&columns[COL_IS_AGG_PUBKEY_PHASE][r])
                    .add(&columns[COL_IS_PAIRING_PHASE][r]);
                c[r] = sum.sub(&columns[COL_IS_REAL][r]);
            }
            out.push(c);
        }

        // 6: phase_index = 0·filter + 1·h2g2 + 2·agg + 3·pairing.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let weighted = columns[COL_IS_HASH_TO_G2_PHASE][r]
                    .add(&two.mul(&columns[COL_IS_AGG_PUBKEY_PHASE][r]))
                    .add(&three.mul(&columns[COL_IS_PAIRING_PHASE][r]));
                c[r] = columns[COL_PHASE_INDEX][r].sub(&weighted);
            }
            out.push(c);
        }

        // 7: phase_index = phase_index_byte.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_PHASE_INDEX][r]
                    .sub(&columns[COL_PHASE_INDEX_BYTE][r]);
            }
            out.push(c);
        }

        // 8: pairing_result binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_PAIRING_RESULT][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 9: participation_count = Σ 256^k * pc_byte[k].
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut acc = Scalar::zero(curve);
                for k in 0..COUNTER_BYTES {
                    acc = acc.add(&pow[k].mul(
                        &columns[COL_PARTICIPATION_COUNT_BYTES_OFFSET + k][r],
                    ));
                }
                c[r] = columns[COL_PARTICIPATION_COUNT][r].sub(&acc);
            }
            out.push(c);
        }

        // 10: threshold = Σ 256^k * th_byte[k].
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut acc = Scalar::zero(curve);
                for k in 0..COUNTER_BYTES {
                    acc = acc.add(&pow[k].mul(
                        &columns[COL_THRESHOLD_BYTES_OFFSET + k][r],
                    ));
                }
                c[r] = columns[COL_THRESHOLD][r].sub(&acc);
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
        let three = Scalar::from_u64(3, curve);
        let pow = byte_powers(curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        let push = |acc: &mut Scalar, alpha_pow: &mut Scalar, body: Scalar| {
            *acc = acc.add(&alpha_pow.mul(&body));
            *alpha_pow = alpha_pow.mul(alpha);
        };

        // 0..4: binarities.
        for col in [
            COL_IS_REAL,
            COL_IS_FILTER_PHASE,
            COL_IS_HASH_TO_G2_PHASE,
            COL_IS_AGG_PUBKEY_PHASE,
            COL_IS_PAIRING_PHASE,
        ] {
            let v = &col_evals[col];
            push(&mut acc, &mut alpha_pow, v.mul(&v.sub(&one)));
        }

        // 5: phase sum.
        let sum = col_evals[COL_IS_FILTER_PHASE]
            .add(&col_evals[COL_IS_HASH_TO_G2_PHASE])
            .add(&col_evals[COL_IS_AGG_PUBKEY_PHASE])
            .add(&col_evals[COL_IS_PAIRING_PHASE]);
        push(&mut acc, &mut alpha_pow, sum.sub(&col_evals[COL_IS_REAL]));

        // 6: phase_index = weighted selector sum.
        let weighted = col_evals[COL_IS_HASH_TO_G2_PHASE]
            .add(&two.mul(&col_evals[COL_IS_AGG_PUBKEY_PHASE]))
            .add(&three.mul(&col_evals[COL_IS_PAIRING_PHASE]));
        push(
            &mut acc,
            &mut alpha_pow,
            col_evals[COL_PHASE_INDEX].sub(&weighted),
        );

        // 7: phase_index = phase_index_byte.
        push(
            &mut acc,
            &mut alpha_pow,
            col_evals[COL_PHASE_INDEX].sub(&col_evals[COL_PHASE_INDEX_BYTE]),
        );

        // 8: pairing_result binary.
        {
            let v = &col_evals[COL_PAIRING_RESULT];
            push(&mut acc, &mut alpha_pow, v.mul(&v.sub(&one)));
        }

        // 9: participation_count LE byte decomp.
        {
            let mut s = Scalar::zero(curve);
            for k in 0..COUNTER_BYTES {
                s = s.add(&pow[k].mul(
                    &col_evals[COL_PARTICIPATION_COUNT_BYTES_OFFSET + k],
                ));
            }
            push(
                &mut acc,
                &mut alpha_pow,
                col_evals[COL_PARTICIPATION_COUNT].sub(&s),
            );
        }

        // 10: threshold LE byte decomp.
        {
            let mut s = Scalar::zero(curve);
            for k in 0..COUNTER_BYTES {
                s = s.add(&pow[k].mul(
                    &col_evals[COL_THRESHOLD_BYTES_OFFSET + k],
                ));
            }
            push(&mut acc, &mut alpha_pow, col_evals[COL_THRESHOLD].sub(&s));
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
        let three_poly = vec![Scalar::from_u64(3, curve)];
        let pow = byte_powers(curve);

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        let push =
            |acc: &mut Vec<Scalar>, alpha_pow: &mut Scalar, body: Vec<Scalar>| {
                let term = poly_scalar_mul(&body, alpha_pow);
                *acc = poly_add(acc, &term, curve);
                *alpha_pow = alpha_pow.mul(alpha);
            };

        // 0..4: binarities.
        for col in [
            COL_IS_REAL,
            COL_IS_FILTER_PHASE,
            COL_IS_HASH_TO_G2_PHASE,
            COL_IS_AGG_PUBKEY_PHASE,
            COL_IS_PAIRING_PHASE,
        ] {
            let v = &col_coeffs[col];
            let v_m1 = poly_sub(v, &one_poly, curve);
            push(&mut acc, &mut alpha_pow, poly_mul(v, &v_m1, curve));
        }

        // 5: phase sum.
        let sum = poly_add(
            &poly_add(
                &col_coeffs[COL_IS_FILTER_PHASE],
                &col_coeffs[COL_IS_HASH_TO_G2_PHASE],
                curve,
            ),
            &poly_add(
                &col_coeffs[COL_IS_AGG_PUBKEY_PHASE],
                &col_coeffs[COL_IS_PAIRING_PHASE],
                curve,
            ),
            curve,
        );
        push(
            &mut acc,
            &mut alpha_pow,
            poly_sub(&sum, &col_coeffs[COL_IS_REAL], curve),
        );

        // 6: phase_index = weighted selector sum.
        let weighted = poly_add(
            &col_coeffs[COL_IS_HASH_TO_G2_PHASE],
            &poly_add(
                &poly_mul(&two_poly, &col_coeffs[COL_IS_AGG_PUBKEY_PHASE], curve),
                &poly_mul(&three_poly, &col_coeffs[COL_IS_PAIRING_PHASE], curve),
                curve,
            ),
            curve,
        );
        push(
            &mut acc,
            &mut alpha_pow,
            poly_sub(&col_coeffs[COL_PHASE_INDEX], &weighted, curve),
        );

        // 7: phase_index = phase_index_byte.
        push(
            &mut acc,
            &mut alpha_pow,
            poly_sub(
                &col_coeffs[COL_PHASE_INDEX],
                &col_coeffs[COL_PHASE_INDEX_BYTE],
                curve,
            ),
        );

        // 8: pairing_result binary.
        {
            let v = &col_coeffs[COL_PAIRING_RESULT];
            let v_m1 = poly_sub(v, &one_poly, curve);
            push(&mut acc, &mut alpha_pow, poly_mul(v, &v_m1, curve));
        }

        // 9: participation_count LE byte decomp.
        {
            let mut s = vec![Scalar::zero(curve)];
            for k in 0..COUNTER_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_PARTICIPATION_COUNT_BYTES_OFFSET + k],
                    &pow[k],
                );
                s = poly_add(&s, &term, curve);
            }
            push(
                &mut acc,
                &mut alpha_pow,
                poly_sub(&col_coeffs[COL_PARTICIPATION_COUNT], &s, curve),
            );
        }

        // 10: threshold LE byte decomp.
        {
            let mut s = vec![Scalar::zero(curve)];
            for k in 0..COUNTER_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_THRESHOLD_BYTES_OFFSET + k],
                    &pow[k],
                );
                s = poly_add(&s, &term, curve);
            }
            push(
                &mut acc,
                &mut alpha_pow,
                poly_sub(&col_coeffs[COL_THRESHOLD], &s, curve),
            );
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
            .unwrap_or(CurveType::Bls12381);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    // ── Shifted (cross-row) constraint: msg β-RLC continuity across
    //    adjacent rows of the same instance, gated by
    //    is_real(r) * is_real(r+1).
    //
    // Shifted column layout:
    //   [0]                  IS_REAL_NEXT
    //   [1..1+MSG_LEN]       MSG_NEXT bytes
    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        let mut cols = Vec::with_capacity(1 + MSG_LEN);
        cols.push(COL_IS_REAL);
        for k in 0..MSG_LEN {
            cols.push(COL_MSG_OFFSET + k);
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
        let curve = alpha.curve_type();
        let zero = Scalar::zero(curve);
        let expected_shifted = 1 + MSG_LEN;
        if shifted_evals.len() != expected_shifted
            || col_evals_at_z.len() < NUM_COLUMNS
        {
            return zero;
        }
        let beta = beta_rlc(curve);
        let is_real_now = &col_evals_at_z[COL_IS_REAL];
        let is_real_next = &shifted_evals[0];

        // β-RLC over MSG_LEN bytes: Σ β^k * (msg_now[k] - msg_next[k]).
        let mut rlc = zero.clone();
        let mut bp = Scalar::one(curve);
        for k in 0..MSG_LEN {
            let cur = &col_evals_at_z[COL_MSG_OFFSET + k];
            let nxt = &shifted_evals[1 + k];
            rlc = rlc.add(&bp.mul(&cur.sub(nxt)));
            bp = bp.mul(&beta);
        }
        let body = is_real_now.mul(is_real_next).mul(&rlc);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        ap.mul(&body).mul(&z.sub(omega_n_minus_1))
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
        let beta = beta_rlc(curve);

        let mut rlc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..MSG_LEN {
            let cur = &column_coeffs[COL_MSG_OFFSET + k];
            let nxt = poly_shift(cur, omega);
            let diff = poly_sub(cur, &nxt, curve);
            rlc = poly_add(&rlc, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(&beta);
        }

        let is_real_now = &column_coeffs[COL_IS_REAL];
        let is_real_next = poly_shift(is_real_now, omega);
        let gate = poly_mul(is_real_now, &is_real_next, curve);
        let body = poly_mul(&gate, &rlc, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let total = poly_scalar_mul(&body, &ap);

        // Multiply by (X - ω^{n-1}) so the wrap-around row (row n-1)
        // is excluded from the cross-row constraint — matches the
        // `(z - ω^{n-1})` factor in `evaluate_shifted_at_point`.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();

        for k in 0..MSG_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_composer_msg_byte_{}_8bit", k),
                    column_index: COL_MSG_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..PK_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_composer_agg_pubkey_byte_{}_8bit", k),
                    column_index: COL_AGG_PUBKEY_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SIG_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_composer_agg_sig_byte_{}_8bit", k),
                    column_index: COL_AGG_SIG_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..G2_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_composer_h2g2_byte_{}_8bit", k),
                    column_index: COL_HASH_TO_G2_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..COUNTER_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_composer_participation_byte_{}_8bit", k),
                    column_index: COL_PARTICIPATION_COUNT_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..COUNTER_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sync_composer_threshold_byte_{}_8bit", k),
                    column_index: COL_THRESHOLD_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        declarations.push((
            LookupDeclaration {
                label: "sync_composer_phase_index_byte_8bit".into(),
                column_index: COL_PHASE_INDEX_BYTE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));

        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Phase-0 binding: filter row of this AIR ↔
/// [`crate::sync_committee_filter_air`].
///
/// The A-side tuple is the first [`crate::sync_committee_filter_air::PUBKEY_LEN`]
/// bytes of `aggregate_pubkey` (column-shape anchor; the per-member
/// pubkey accumulation is owned by the filter AIR via its bitmap-gated
/// columns). A side selector = [`COL_IS_FILTER_PHASE`]. B side selector
/// = the filter AIR's [`crate::sync_committee_filter_air::COL_IS_REAL`].
pub fn make_sync_composer_to_filter_descriptor(
    composition_layer_index: usize,
    filter_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sync_committee_filter_air as filter;
    let a_columns: Vec<usize> =
        (0..filter::PUBKEY_LEN).map(|k| COL_AGG_PUBKEY_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..filter::PUBKEY_LEN).map(|k| filter::COL_PUBKEY_OFFSET + k).collect();
    debug_assert_eq!(a_columns.len(), b_columns.len());
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_composer_to_filter_v1".into(),
        a_layer_index: composition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_FILTER_PHASE),
        b_layer_index: filter_layer_index,
        b_columns,
        b_selector_column: Some(filter::COL_IS_REAL),
    }
}

/// Phase-1 binding: hash-to-G2 row of this AIR ↔
/// [`crate::hash_to_g2_air`].
///
/// The A-side tuple is the 32-byte `msg` slice. The B-side tuple is the
/// first 32 bytes of the hash_to_g2 AIR's `msg` column slab. A side
/// selector = [`COL_IS_HASH_TO_G2_PHASE`]. B side selector = the
/// hash_to_g2 AIR's [`crate::hash_to_g2_air::COL_IS_REAL`].
pub fn make_sync_composer_to_hash_to_g2_descriptor(
    composition_layer_index: usize,
    hash_to_g2_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::hash_to_g2_air as h2g2;
    let a_columns: Vec<usize> = (0..MSG_LEN).map(|k| COL_MSG_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..MSG_LEN).map(|k| h2g2::COL_MSG_OFFSET + k).collect();
    debug_assert_eq!(a_columns.len(), b_columns.len());
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_composer_to_hash_to_g2_v1".into(),
        a_layer_index: composition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_HASH_TO_G2_PHASE),
        b_layer_index: hash_to_g2_layer_index,
        b_columns,
        b_selector_column: Some(h2g2::COL_IS_REAL),
    }
}

/// Phase-2 binding: aggregate-pubkey row of this AIR ↔
/// [`crate::sync_committee_sig_air`] (binds the 48-byte aggregate
/// pubkey to the sync-sig AIR's `agg_pk_x_bytes` columns). A side
/// selector = [`COL_IS_AGG_PUBKEY_PHASE`]. B side selector = the
/// sync-sig AIR's `COL_IS_REAL`.
pub fn make_sync_composer_to_sync_sig_descriptor(
    composition_layer_index: usize,
    sync_sig_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sync_committee_sig_air as ssig;
    let a_columns: Vec<usize> =
        (0..ssig::PK_BYTES).map(|k| COL_AGG_PUBKEY_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..ssig::PK_BYTES).map(|k| ssig::COL_AGG_PK_X_BYTES_OFFSET + k).collect();
    debug_assert_eq!(a_columns.len(), b_columns.len());
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_composer_to_sync_sig_v1".into(),
        a_layer_index: composition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_AGG_PUBKEY_PHASE),
        b_layer_index: sync_sig_layer_index,
        b_columns,
        b_selector_column: Some(ssig::COL_IS_REAL),
    }
}

/// Phase-3 binding: pairing row of this AIR ↔
/// [`crate::bls_pairing_air`].
///
/// The A-side tuple is the concatenation `(aggregate_pubkey[0..48] ||
/// aggregate_sig[0..96] || msg[0..32])`. The B-side tuple is the
/// pairing AIR's `(COL_PK_COMPRESSED || COL_SIG_BYTES || COL_MSG_HASH)`
/// column triple. A side selector = [`COL_IS_PAIRING_PHASE`]. B side
/// selector = the pairing AIR's [`crate::bls_pairing_air::COL_IS_REAL`].
pub fn make_sync_composer_to_pairing_descriptor(
    composition_layer_index: usize,
    pairing_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls_pairing_air as pair;
    let mut a_columns: Vec<usize> =
        Vec::with_capacity(pair::PK_BYTES + pair::SIG_BYTES + pair::MSG_HASH_BYTES);
    for k in 0..pair::PK_BYTES {
        a_columns.push(COL_AGG_PUBKEY_OFFSET + k);
    }
    for k in 0..pair::SIG_BYTES {
        a_columns.push(COL_AGG_SIG_OFFSET + k);
    }
    for k in 0..pair::MSG_HASH_BYTES {
        a_columns.push(COL_MSG_OFFSET + k);
    }
    let mut b_columns: Vec<usize> =
        Vec::with_capacity(pair::PK_BYTES + pair::SIG_BYTES + pair::MSG_HASH_BYTES);
    for k in 0..pair::PK_BYTES {
        b_columns.push(pair::COL_PK_COMPRESSED_OFFSET + k);
    }
    for k in 0..pair::SIG_BYTES {
        b_columns.push(pair::COL_SIG_BYTES_OFFSET + k);
    }
    for k in 0..pair::MSG_HASH_BYTES {
        b_columns.push(pair::COL_MSG_HASH_OFFSET + k);
    }
    debug_assert_eq!(a_columns.len(), b_columns.len());
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_composer_to_pairing_v1".into(),
        a_layer_index: composition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_PAIRING_PHASE),
        b_layer_index: pairing_layer_index,
        b_columns,
        b_selector_column: Some(pair::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(witness: &SyncCommitteeAggregateComposerWitness) -> Vec<Vec<Scalar>> {
        let trace = build_trace_polynomials(witness, CurveType::Bls12381);
        let cs = SyncCommitteeAggregateComposerConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    fn assert_all_zero(results: &[Vec<Scalar>]) {
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    /// Build a tiny well-formed witness using a real aggregate signature
    /// across two random signers, hash-to-G2 via blst, and the IETF POP
    /// DST.
    fn build_honest_witness() -> SyncCommitteeAggregateComposerWitness {
        use crate::bls_sig as bs;

        let msg: [u8; MSG_LEN] = *b"sync-composer-test-msg-32-bytes!";
        let sk0 = bs::SecretKey::from_u8_seed(11);
        let sk1 = bs::SecretKey::from_u8_seed(22);
        let pk0 = sk0.public_key();
        let pk1 = sk1.public_key();
        const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
        let s0 = sk0.sign(&msg, POP_DST);
        let s1 = sk1.sign(&msg, POP_DST);
        let agg_sig = bs::aggregate_sigs(&[s0, s1]).unwrap();
        let participation_count: u32 = 2;
        SyncCommitteeAggregateComposerWitness::from_aggregate(
            msg,
            &[pk0.0, pk1.0],
            agg_sig.0,
            participation_count,
        )
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: well-formed witness — all row-local constraints vanish.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn well_formed_witness_passes_row_local_constraints() {
        let w = build_honest_witness();
        assert_eq!(w.rows.len(), ROWS_PER_INSTANCE);
        // Phase indices are 0..3 in order.
        for (r, row) in w.rows.iter().enumerate() {
            assert_eq!(row.phase_index, r as u8);
        }
        // All four rows share the same per-instance state.
        let row0 = &w.rows[0];
        for row in &w.rows[1..] {
            assert_eq!(row.msg, row0.msg);
            assert_eq!(row.aggregate_pubkey, row0.aggregate_pubkey);
            assert_eq!(row.aggregate_sig, row0.aggregate_sig);
            assert_eq!(row.hash_to_g2_output, row0.hash_to_g2_output);
            assert_eq!(row.pairing_result, row0.pairing_result);
            assert_eq!(row.participation_count, row0.participation_count);
            assert_eq!(row.threshold, row0.threshold);
        }
        // Honest pairing of two valid signers on the same message
        // should verify.
        assert_eq!(row0.pairing_result, 1);
        // Hash-to-G2 output is non-zero.
        assert!(row0.hash_to_g2_output.iter().any(|&b| b != 0));
        // Threshold matches the supermajority default for committee=512.
        assert_eq!(row0.threshold, 342);

        let results = evaluate(&w);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        assert_all_zero(&results);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: tampered cross-phase msg detected by the shifted msg-
    // continuity constraint.
    // ───────────────────────────────────────────────────────────────────

    fn shifted_msg_continuity_per_row(
        cols: &[Vec<Scalar>],
        curve: CurveType,
    ) -> Vec<Scalar> {
        let n = cols[0].len();
        let beta = beta_rlc(curve);
        let mut out = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let r_next = (r + 1) % n;
            let is_real_now = &cols[COL_IS_REAL][r];
            let is_real_next = &cols[COL_IS_REAL][r_next];
            let mut rlc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..MSG_LEN {
                let cur = &cols[COL_MSG_OFFSET + k][r];
                let nxt = &cols[COL_MSG_OFFSET + k][r_next];
                rlc = rlc.add(&bp.mul(&cur.sub(nxt)));
                bp = bp.mul(&beta);
            }
            out[r] = is_real_now.mul(is_real_next).mul(&rlc);
        }
        out
    }

    #[test]
    fn tampered_cross_phase_msg_detected_by_shifted_constraint() {
        let w = build_honest_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let curve = CurveType::Bls12381;

        // Honest closure: msg continuity vanishes everywhere.
        let cols_honest: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let honest = shifted_msg_continuity_per_row(&cols_honest, curve);
        for (r, val) in honest.iter().enumerate() {
            assert!(
                val.is_zero(),
                "honest msg-continuity must vanish at row {} (got {:?})",
                r, val,
            );
        }

        // Tamper msg[0] on the hash_to_g2 row (row 1): change byte.
        let mut cols: Vec<Vec<Scalar>> = cols_honest.clone();
        let original = cols[COL_MSG_OFFSET][1].clone();
        cols[COL_MSG_OFFSET][1] = original.add(&Scalar::one(curve));
        let tampered = shifted_msg_continuity_per_row(&cols, curve);
        assert!(
            !tampered[0].is_zero() || !tampered[1].is_zero(),
            "tampering msg on phase row 1 must fire the shifted msg-continuity constraint",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: tampered participation_count (without retagging LE bytes)
    // is detected by the byte-decomp constraint.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_participation_count_detected() {
        let w = build_honest_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let curve = CurveType::Bls12381;
        let cs = SyncCommitteeAggregateComposerConstraintSystem::new(trace.num_rows);

        // Sanity: honest trace vanishes on constraint 9.
        {
            let col_refs: Vec<&Vec<Scalar>> =
                trace.columns.iter().map(|p| &p.evaluations).collect();
            let r = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            for v in &r[9] {
                assert!(v.is_zero(), "honest participation byte-decomp must vanish");
            }
        }

        // Tamper participation_count integer on row 0 — keep LE bytes
        // untouched. The LE byte-decomp constraint should fire.
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_PARTICIPATION_COUNT][0] = Scalar::from_u64(999_999, curve);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let r = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !r[9][0].is_zero(),
            "tampered participation_count must fire LE byte-decomp constraint 9",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: descriptors are well-formed.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn descriptors_well_formed() {
        let d0 = make_sync_composer_to_filter_descriptor(0, 1);
        assert_eq!(d0.label, "sync_composer_to_filter_v1");
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d0.a_columns.len(), crate::sync_committee_filter_air::PUBKEY_LEN);
        assert_eq!(d0.b_columns.len(), crate::sync_committee_filter_air::PUBKEY_LEN);
        assert_eq!(d0.a_selector_column, Some(COL_IS_FILTER_PHASE));
        assert_eq!(
            d0.b_selector_column,
            Some(crate::sync_committee_filter_air::COL_IS_REAL),
        );
        assert_eq!(d0.a_columns[0], COL_AGG_PUBKEY_OFFSET);
        assert_eq!(
            d0.b_columns[0],
            crate::sync_committee_filter_air::COL_PUBKEY_OFFSET,
        );

        let d1 = make_sync_composer_to_hash_to_g2_descriptor(0, 2);
        assert_eq!(d1.label, "sync_composer_to_hash_to_g2_v1");
        assert_eq!(d1.a_columns.len(), MSG_LEN);
        assert_eq!(d1.b_columns.len(), MSG_LEN);
        assert_eq!(d1.a_selector_column, Some(COL_IS_HASH_TO_G2_PHASE));
        assert_eq!(
            d1.b_selector_column,
            Some(crate::hash_to_g2_air::COL_IS_REAL),
        );
        assert_eq!(d1.a_columns[0], COL_MSG_OFFSET);
        assert_eq!(d1.b_columns[0], crate::hash_to_g2_air::COL_MSG_OFFSET);

        let d2 = make_sync_composer_to_sync_sig_descriptor(0, 3);
        assert_eq!(d2.label, "sync_composer_to_sync_sig_v1");
        assert_eq!(d2.a_columns.len(), crate::sync_committee_sig_air::PK_BYTES);
        assert_eq!(d2.b_columns.len(), crate::sync_committee_sig_air::PK_BYTES);
        assert_eq!(d2.a_selector_column, Some(COL_IS_AGG_PUBKEY_PHASE));
        assert_eq!(
            d2.b_selector_column,
            Some(crate::sync_committee_sig_air::COL_IS_REAL),
        );
        assert_eq!(d2.a_columns[0], COL_AGG_PUBKEY_OFFSET);
        assert_eq!(
            d2.b_columns[0],
            crate::sync_committee_sig_air::COL_AGG_PK_X_BYTES_OFFSET,
        );

        let d3 = make_sync_composer_to_pairing_descriptor(0, 4);
        assert_eq!(d3.label, "sync_composer_to_pairing_v1");
        let expected_pair_tuple = crate::bls_pairing_air::PK_BYTES
            + crate::bls_pairing_air::SIG_BYTES
            + crate::bls_pairing_air::MSG_HASH_BYTES;
        assert_eq!(d3.a_columns.len(), expected_pair_tuple);
        assert_eq!(d3.b_columns.len(), expected_pair_tuple);
        assert_eq!(d3.a_selector_column, Some(COL_IS_PAIRING_PHASE));
        assert_eq!(
            d3.b_selector_column,
            Some(crate::bls_pairing_air::COL_IS_REAL),
        );
        assert_eq!(d3.a_columns[0], COL_AGG_PUBKEY_OFFSET);
        assert_eq!(
            d3.b_columns[0],
            crate::bls_pairing_air::COL_PK_COMPRESSED_OFFSET,
        );
        // Descriptor labels are unique.
        let labels = [
            d0.label.as_str(),
            d1.label.as_str(),
            d2.label.as_str(),
            d3.label.as_str(),
        ];
        let mut sorted = labels.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            labels.len(),
            "descriptor labels must be unique",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: column layout is pinned.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_MSG_OFFSET, 0);
        assert_eq!(COL_AGG_PUBKEY_OFFSET, MSG_LEN);
        assert_eq!(COL_AGG_SIG_OFFSET, MSG_LEN + PK_LEN);
        assert_eq!(COL_HASH_TO_G2_OFFSET, MSG_LEN + PK_LEN + SIG_LEN);
        assert_eq!(COL_PAIRING_RESULT, MSG_LEN + PK_LEN + SIG_LEN + G2_LEN);
        // Trailing scalars: pairing_result (1) + participation_count (1)
        // + 4 LE bytes + threshold (1) + 4 LE bytes + phase_index (1) +
        // phase_index_byte (1) + 4 phase selectors + is_real (1) = 18,
        // plus 2 sub-AIR mirror columns (task #190) = 20.
        let bytes = MSG_LEN + PK_LEN + SIG_LEN + G2_LEN;
        assert_eq!(NUM_COLUMNS, bytes + 20);
        // 32 + 48 + 96 + 96 + 20 = 292.
        assert_eq!(NUM_COLUMNS, 292);
        assert_eq!(NUM_ROW_CONSTRAINTS, 11);
        assert_eq!(NUM_SHIFTED, 1);
        // Phase order matches the pipeline order.
        assert_eq!(PHASE_FILTER, 0);
        assert_eq!(PHASE_HASH_TO_G2, 1);
        assert_eq!(PHASE_AGG_PUBKEY, 2);
        assert_eq!(PHASE_PAIRING, 3);
    }

    // ───────────────────────────────────────────────────────────────────
    // Bonus: tampering pairing_result to non-binary fires constraint 8.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_pairing_result_fires_binary_constraint() {
        let w = build_honest_witness();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let curve = CurveType::Bls12381;
        let cs = SyncCommitteeAggregateComposerConstraintSystem::new(trace.num_rows);

        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_PAIRING_RESULT][0] = Scalar::from_u64(7, curve);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let r = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !r[8][0].is_zero(),
            "non-binary pairing_result must fire constraint 8",
        );
    }

    /// Task #203: standalone prove+verify under BLS48-581. Companion to
    /// `hash_to_curve_composition_air::diagnostic_composer_standalone_prove_verify`
    /// — both composer AIRs need this end-to-end sanity check after the
    /// task #202 `(X - ω^{n-1})` wrap-row exclusion fix in
    /// `build_shifted_constraint_polynomial`.
    #[test]
    #[ignore = "slow: standalone composer prove+verify under BLS48-581"]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let w = build_honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = SyncCommitteeAggregateComposerConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone sync_committee_aggregate_composer_air proof must verify",
        );
    }
}
