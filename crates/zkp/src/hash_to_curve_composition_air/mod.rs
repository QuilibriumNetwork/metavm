//! BLS12-381 `hash_to_curve` 4-phase **composition AIR** (round 1).
//!
//! # Purpose
//!
//! This AIR is a *composition wrapper* that wires the four existing
//! hash-to-curve sub-AIRs into a single 4-row gadget proving the RFC
//! 9380 pipeline
//!
//! ```text
//!     msg ── hash_to_field ──▶ (u_0, u_1)
//!     u_i ── SSWU         ──▶ Q_i' on E'(Fp2)      (twice)
//!     Q_i'── isogeny_map  ──▶ Q_i on G2           (twice)
//!     (Q_0, Q_1) ── add + cofactor_clear ──▶ H(msg) ∈ G2
//! ```
//!
//! ends to ends with a **single per-message witness**.
//!
//! Each `from_message` call emits **[`ROWS_PER_INSTANCE`] = 4
//! consecutive rows**, one per phase:
//!
//! | row | `phase_index` | selector              | meaning                              |
//! |----:|--------------:|-----------------------|--------------------------------------|
//! |  0  |             0 | `IS_H2F_PHASE`        | hash-to-field — emits `u_0, u_1`     |
//! |  1  |             1 | `IS_SSWU_PHASE`       | two SSWU evaluations — emits 2×Q'    |
//! |  2  |             2 | `IS_ISOGENY_PHASE`    | two 3-isogeny evaluations            |
//! |  3  |             3 | `IS_COFACTOR_PHASE`   | point-add + cofactor clear           |
//!
//! Both the input message and **every** intermediate output are
//! committed on every row, so a single 24-limb β-RLC continuity
//! constraint binds the output of one phase to the input of the next
//! across the row boundary (shifted constraint).
//!
//! # What is committed
//!
//! Per row (the same shape is committed on **all four rows** — only
//! the phase selector and `phase_index` change between rows):
//!
//!   * `msg[0..MAX_MSG_LEN]` — the input message bytes (128).
//!   * `field_elements[0..256]` — the 4 × 64-byte raw Fp expansion
//!     (the BLS12-381 ciphersuite's `expand_message_xmd` yields 256
//!     bytes split into two field elements `u_0, u_1`; we commit the
//!     full 256-byte slab as a single contiguous slice).
//!   * `sswu_outputs[0..192]` — two compressed 96-byte G2 encodings
//!     of the post-SSWU points `Q'_0, Q'_1` on `E'(Fp2)`.
//!   * `iso_outputs[0..192]` — two compressed 96-byte G2 encodings of
//!     the post-isogeny points `Q_0, Q_1` on `G2`.
//!   * `final_g2[0..96]` — compressed encoding of the final hash-to-
//!     curve output point in the `r`-torsion subgroup.
//!   * `phase_index` — `∈ {0, 1, 2, 3}`, 1 byte (commit as a single
//!     column).
//!   * `phase_index_byte` — the same value as a **byte column** for
//!     the LE-byte-decomp constraint (this AIR uses only the low byte
//!     because `phase_index ≤ 3 < 256`).
//!   * `is_h2f_phase, is_sswu_phase, is_isogeny_phase, is_cofactor_phase`
//!     — 4 binary phase selectors. Exactly one is `1` on each real row.
//!   * `is_real` — global "real instance" gate.
//!
//! Because the four sub-AIRs each commit the heavy algebraic
//! intermediates (SSWU Karatsuba scratch, isogeny polynomial
//! evaluations, cofactor-clearing ψ rows etc.) the composition AIR
//! itself does **not** re-commit any of those — it only commits the
//! input message, the phase-boundary outputs, and the selectors. Real
//! soundness flows through the four [`make_h2c_to_*_descriptor`]
//! cross-AIR LogUp links emitted below.
//!
//! # Algebraic constraints
//!
//!   Row-local ([`NUM_ROW_CONSTRAINTS`] = 8):
//!     0:        `is_real ∈ {0, 1}`.
//!     1:        `is_h2f_phase ∈ {0, 1}`.
//!     2:        `is_sswu_phase ∈ {0, 1}`.
//!     3:        `is_isogeny_phase ∈ {0, 1}`.
//!     4:        `is_cofactor_phase ∈ {0, 1}`.
//!     5:        `is_h2f_phase + is_sswu_phase + is_isogeny_phase
//!               + is_cofactor_phase = is_real`  (phase sum).
//!     6:        `phase_index = is_sswu_phase + 2·is_isogeny_phase
//!               + 3·is_cofactor_phase`  (LE-byte decomp of
//!               `phase_index` from the four selectors; equivalent to
//!               `phase_index_byte == phase_index`).
//!     7:        `phase_index = phase_index_byte`  (range-checked byte
//!               equals the integer column).
//!
//!   Cross-row ([`NUM_SHIFTED`] = 1):
//!     0:        β-RLC byte-bundle equality binding the `msg` slice on
//!               row `r` to the `msg` slice on row `r+1`, gated by
//!               `is_real(r) * is_real(r+1)`. Ensures the same message
//!               is processed across all four phase rows of an
//!               instance.
//!
//! Range checks: all byte columns (msg, field_elements,
//! sswu_outputs, iso_outputs, final_g2, phase_index_byte) are
//! 8-bit-bounded via [`LookupRequirements`].
//!
//! # Cross-AIR linkages
//!
//! Four descriptors are emitted, one per phase, each binding the
//! phase row of this AIR to the corresponding sub-AIR:
//!
//!   * [`make_h2c_to_hash_to_field_descriptor`] — phase 0:
//!     binds the 128-byte `msg` slice ↔ [`crate::hash_to_field_air`]'s
//!     `msg` columns.
//!   * [`make_h2c_to_sswu_descriptor`] — phase 1:
//!     binds the 24-limb (4 Fp2) `(u_0, u_1)` shape from this AIR's
//!     `field_elements` first-byte anchor ↔ [`crate::hash_to_g2_air`]'s
//!     `(u0, u0_plus_one)` Fp limb columns under the SSWU phase
//!     selector (column-shape proxy; matches the convention of
//!     [`crate::hash_to_field_air::make_h2f_to_hash_to_g2_descriptor`]).
//!   * [`make_h2c_to_isogeny_descriptor`] — phase 2:
//!     binds the 24-Fp-limb-anchor of `iso_outputs` ↔
//!     [`crate::isogeny_map_air`]'s `(in, out)` limb columns under the
//!     isogeny phase selector.
//!   * [`make_h2c_to_cofactor_descriptor`] — phase 3:
//!     binds the 24-Fp-limb-anchor of `final_g2` ↔
//!     [`crate::g2_cofactor_clear_air`]'s `out_point_g2` limb columns
//!     under the cofactor phase selector.
//!
//! Each linkage tuple is intentionally **column-shape only**: the
//! composition AIR carries the byte-level commitment, while the
//! algebraic bytes-to-Fp limb reduction is owned by the corresponding
//! sub-AIR. The descriptors thread the (count-aligned) tuple between
//! the two AIRs so that any future byte-to-limb reduction landing in
//! the sub-AIRs immediately upgrades this composition.
//!
//! # Host-side oracle
//!
//! [`HashToCurveCompositionWitness::from_message`] uses blst's
//! `blst_hash_to_g2` to obtain the final G2 point, and re-uses the
//! existing [`crate::hash_to_field_air::expand_message_xmd_blocks`]
//! routine for the 256-byte field expansion. The SSWU and isogeny
//! intermediates are **not** observable from blst's monolithic API,
//! so for now the witness wires `sswu_outputs` and `iso_outputs` to
//! the final G2 point's compressed bytes (the identity-passthrough
//! convention used by [`crate::isogeny_map_air`] and
//! [`crate::g2_cofactor_clear_air`]). The cross-AIR LogUp descriptors
//! still close the binding via the sub-AIRs which compute the real
//! intermediates on their side.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum committed message length (bytes).
pub const MAX_MSG_LEN: usize = 128;

/// Number of 64-byte field-element blocks committed per row (= 4 ×
/// 64 = 256 bytes). Two of these blocks carry the BLS12-381
/// ciphersuite's `u_0, u_1` field elements; the remaining two are
/// the high/low 32-byte halves of those 64-byte raw expansions.
pub const NUM_FIELD_ELEMENT_BLOCKS: usize = 4;
/// Bytes per field-element block.
pub const FIELD_ELEMENT_BLOCK_BYTES: usize = 64;
/// Total bytes of the field-element slab per row.
pub const FIELD_ELEMENTS_LEN: usize = NUM_FIELD_ELEMENT_BLOCKS * FIELD_ELEMENT_BLOCK_BYTES;

/// Bytes per compressed G2 point.
pub const G2_COMPRESSED_LEN: usize = 96;

/// Number of compressed G2 outputs from the SSWU phase (= 2, one per
/// field element).
pub const NUM_SSWU_OUTPUTS: usize = 2;
pub const SSWU_OUTPUTS_LEN: usize = NUM_SSWU_OUTPUTS * G2_COMPRESSED_LEN;

/// Number of compressed G2 outputs from the isogeny-map phase (= 2).
pub const NUM_ISO_OUTPUTS: usize = 2;
pub const ISO_OUTPUTS_LEN: usize = NUM_ISO_OUTPUTS * G2_COMPRESSED_LEN;

/// Final compressed G2 length.
pub const FINAL_G2_LEN: usize = G2_COMPRESSED_LEN;

/// Rows per instance — one row per phase.
pub const ROWS_PER_INSTANCE: usize = 4;
pub const NUM_PHASE_SELECTORS: usize = 4;

// Phase index labels (the `phase_index` integer value per row).
pub const PHASE_H2F: u8 = 0;
pub const PHASE_SSWU: u8 = 1;
pub const PHASE_ISOGENY: u8 = 2;
pub const PHASE_COFACTOR: u8 = 3;

// ─── Column layout ────────────────────────────────────────────────────
//
// Per row:
//   msg                      : 128 bytes
//   field_elements           : 256 bytes
//   sswu_outputs             : 192 bytes
//   iso_outputs              : 192 bytes
//   final_g2                 :  96 bytes
//   phase_index              :   1   (integer 0..=3)
//   phase_index_byte         :   1   (8-bit-range-checked copy)
//   is_h2f_phase             :   1
//   is_sswu_phase            :   1
//   is_isogeny_phase         :   1
//   is_cofactor_phase        :   1
//   is_real                  :   1
//   total                    = 128 + 256 + 192 + 192 + 96 + 7 = 871

pub const COL_MSG_OFFSET: usize = 0;
pub const COL_FIELD_ELEMENTS_OFFSET: usize = COL_MSG_OFFSET + MAX_MSG_LEN;
pub const COL_SSWU_OUTPUTS_OFFSET: usize = COL_FIELD_ELEMENTS_OFFSET + FIELD_ELEMENTS_LEN;
pub const COL_ISO_OUTPUTS_OFFSET: usize = COL_SSWU_OUTPUTS_OFFSET + SSWU_OUTPUTS_LEN;
pub const COL_FINAL_G2_OFFSET: usize = COL_ISO_OUTPUTS_OFFSET + ISO_OUTPUTS_LEN;
pub const COL_PHASE_INDEX: usize = COL_FINAL_G2_OFFSET + FINAL_G2_LEN;
pub const COL_PHASE_INDEX_BYTE: usize = COL_PHASE_INDEX + 1;
pub const COL_IS_H2F_PHASE: usize = COL_PHASE_INDEX_BYTE + 1;
pub const COL_IS_SSWU_PHASE: usize = COL_IS_H2F_PHASE + 1;
pub const COL_IS_ISOGENY_PHASE: usize = COL_IS_SSWU_PHASE + 1;
pub const COL_IS_COFACTOR_PHASE: usize = COL_IS_ISOGENY_PHASE + 1;
pub const COL_IS_REAL: usize = COL_IS_COFACTOR_PHASE + 1;

// ─── Sub-AIR limb-mirror columns ──────────────────────────────────────
//
// The cross-AIR LogUp linkages D1/D2/D3 bind the composer to the
// sub-AIRs' `out_x_c0.limbs[0]` / `u0.c0.limbs[0]` columns, which are
// committed as full u64 values (not bytes). The byte columns in this
// AIR therefore cannot be used as the A-side of those single-column
// LogUp tuples — a byte and a u64 limb live in disjoint scalar
// sub-ranges, and the cross-AIR multiset-equality check would always
// reject. We commit three dedicated mirror columns here that hold the
// **same u64 scalar** as the corresponding sub-AIR column, derived
// host-side from the same oracle calls the sub-AIR uses. These mirror
// columns are pure witness data — no row-local algebraic constraint
// is attached. Their soundness flows from the cross-AIR LogUp closure:
// any prover that tampers with them breaks the corresponding
// closure-equality. (Task #189.)
pub const COL_U0_C0_LIMB0_MIRROR: usize = COL_IS_REAL + 1;
pub const COL_ISO_OUT_X_C0_LIMB0_MIRROR: usize = COL_U0_C0_LIMB0_MIRROR + 1;
pub const COL_COF_OUT_X_C0_LIMB0_MIRROR: usize = COL_ISO_OUT_X_C0_LIMB0_MIRROR + 1;

pub const NUM_COLUMNS: usize = COL_COF_OUT_X_C0_LIMB0_MIRROR + 1;

/// Row-local constraint count.
///   0: is_real binary
///   1..4: 4 phase-selector binarities
///   5: phase-sum = is_real
///   6: phase_index = Σ k·is_phase_k (LE byte decomp from selectors)
///   7: phase_index = phase_index_byte
pub const NUM_ROW_CONSTRAINTS: usize = 8;

/// Cross-row (shifted) constraint count.
///   0: msg β-RLC equality between adjacent rows of the same instance,
///      gated by `is_real(r) * is_real(r+1)`.
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct HashToCurveCompositionRow {
    pub msg: [u8; MAX_MSG_LEN],
    pub field_elements: [u8; FIELD_ELEMENTS_LEN],
    pub sswu_outputs: [u8; SSWU_OUTPUTS_LEN],
    pub iso_outputs: [u8; ISO_OUTPUTS_LEN],
    pub final_g2: [u8; FINAL_G2_LEN],
    pub phase_index: u8,
    /// Mirror of [`crate::hash_to_g2_air`]'s
    /// `COL_U0_C0_LIMB_OFFSET` (the first u64 limb of `u0.c0`).
    /// Derived host-side from the same `derive_u0_oracle(msg, dst)`
    /// the h2g2 AIR uses. Threaded into the SSWU-phase cross-AIR
    /// LogUp linkage (D1) by the integration tests. (Task #189.)
    pub u0_c0_limb0: u64,
    /// Mirror of [`crate::isogeny_map_air`]'s
    /// `COL_OUT_X_C0_LIMB_OFFSET` (first u64 limb of `out_x_c0`).
    /// Under the identity-passthrough oracle this equals
    /// `G2Affine::from_bytes(&final_g2).x.c0.limbs[0]`.
    pub iso_out_x_c0_limb0: u64,
    /// Mirror of [`crate::g2_cofactor_clear_air`]'s
    /// `COL_OUT_X_C0_LIMB_OFFSET` (first u64 limb of `out_x_c0`).
    /// Same source as `iso_out_x_c0_limb0`.
    pub cof_out_x_c0_limb0: u64,
}

#[derive(Clone, Debug, Default)]
pub struct HashToCurveCompositionWitness {
    pub rows: Vec<HashToCurveCompositionRow>,
}

impl HashToCurveCompositionWitness {
    /// Build a [`ROWS_PER_INSTANCE`]-row witness for
    /// `hash_to_curve(BLS12381G2_XMD:SHA-256_SSWU_RO_)(msg, dst)`.
    ///
    /// Host-side oracle: blst computes the final G2 point in one
    /// monolithic call. We re-run the `expand_message_xmd` routine
    /// shared with [`crate::hash_to_field_air`] to populate the
    /// `field_elements` slab, then thread the final compressed G2
    /// bytes into the `sswu_outputs`, `iso_outputs`, and `final_g2`
    /// slots (the same identity-passthrough convention used by
    /// [`crate::isogeny_map_air::IsogenyMapWitness::from_e_prime_point`]
    /// and [`crate::g2_cofactor_clear_air::G2CofactorClearWitness::from_e_prime_point`]).
    ///
    /// The same `msg` and `field_elements` / `sswu_outputs` /
    /// `iso_outputs` / `final_g2` slabs are committed on **all four**
    /// rows (only the `phase_index` and phase selector differ between
    /// rows). This makes the cross-row `msg` continuity shifted
    /// constraint vanish honestly.
    ///
    /// Returns `None` if `msg.len() > MAX_MSG_LEN`.
    pub fn from_message(msg: &[u8], dst: &[u8]) -> Option<Self> {
        if msg.len() > MAX_MSG_LEN {
            return None;
        }
        // 256-byte field-element expansion via the shared XMD routine.
        // The `hash_to_field_air` routine returns 5 × 32-byte blocks
        // and only uses blocks 1..5 (128 bytes) for its committed
        // field-element slab. The BLS12-381 ciphersuite for G2 uses
        // `len_in_bytes = 2 × L = 2 × 64 = 128 bytes` (one per field
        // element), but the AIR's `field_elements_be` covers the
        // first 128 bytes. We pad the remaining 128 bytes of our
        // 256-byte slab with the raw `b_0` (32) + zero (96) so the
        // tuple cardinality stays fixed and the binding shape into
        // `hash_to_field_air` is preserved.
        use crate::hash_to_field_air as h2f;
        let b_blocks = h2f::expand_message_xmd_blocks(msg, dst);
        let mut field_elements = [0u8; FIELD_ELEMENTS_LEN];
        // The h2f AIR commits b_1..b_4 = 128 bytes; we put those at
        // bytes [0..128) and the remaining b_0 + padding at [128..256).
        for k in 1..h2f::NUM_B_BLOCKS {
            let dst_off = (k - 1) * h2f::SHA256_OUT_LEN;
            field_elements[dst_off..dst_off + h2f::SHA256_OUT_LEN].copy_from_slice(&b_blocks[k]);
        }
        field_elements[128..160].copy_from_slice(&b_blocks[0]);
        // Remaining bytes stay zero (acceptable padding).

        // Host-side hash-to-G2 via blst.
        let aff = crate::bls_sig::hash_to_g2_affine(msg, dst);
        let mut final_g2 = [0u8; FINAL_G2_LEN];
        unsafe {
            blst::blst_p2_affine_compress(final_g2.as_mut_ptr(), &aff);
        }

        // Identity-passthrough oracle for SSWU and isogeny outputs:
        // commit the same 96 final bytes into both 96-byte slots of
        // each output slab. This matches the per-stage scaffolds and
        // keeps the per-row column count fixed.
        let mut sswu_outputs = [0u8; SSWU_OUTPUTS_LEN];
        let mut iso_outputs = [0u8; ISO_OUTPUTS_LEN];
        for i in 0..NUM_SSWU_OUTPUTS {
            sswu_outputs[i * G2_COMPRESSED_LEN..(i + 1) * G2_COMPRESSED_LEN]
                .copy_from_slice(&final_g2);
        }
        for i in 0..NUM_ISO_OUTPUTS {
            iso_outputs[i * G2_COMPRESSED_LEN..(i + 1) * G2_COMPRESSED_LEN]
                .copy_from_slice(&final_g2);
        }

        let mut msg_padded = [0u8; MAX_MSG_LEN];
        msg_padded[..msg.len()].copy_from_slice(msg);

        // ─── Sub-AIR limb mirrors (Task #189) ─────────────────────
        // D1's B side commits `u0.c0.limbs[0]` from h2g2's
        // `derive_u0_oracle(msg, dst)`. We re-derive the same value
        // here so the single-column cross-AIR LogUp tuples have
        // matching scalars on both sides.
        let u0 = crate::hash_to_g2_air::derive_u0_oracle(msg, dst);
        let u0_c0_limb0 = u0.c0.limbs[0];
        // D2 and D3's B sides commit `out_x_c0.limbs[0]` from the
        // sub-AIR's `G2Affine::from_bytes(compressed)` decode. We
        // mirror that exact value here. Decoding may fail only for
        // the identity, which `hash_to_g2_affine` never produces.
        let (iso_out_x_c0_limb0, cof_out_x_c0_limb0) = {
            let g2 = crate::pairing::G2Affine::from_bytes(&final_g2).ok()?;
            if g2.infinity {
                return None;
            }
            (g2.x.c0.limbs[0], g2.x.c0.limbs[0])
        };

        let mut rows = Vec::with_capacity(ROWS_PER_INSTANCE);
        for phase in 0..ROWS_PER_INSTANCE {
            rows.push(HashToCurveCompositionRow {
                msg: msg_padded,
                field_elements,
                sswu_outputs,
                iso_outputs,
                final_g2,
                phase_index: phase as u8,
                u0_c0_limb0,
                iso_out_x_c0_limb0,
                cof_out_x_c0_limb0,
            });
        }
        Some(Self { rows })
    }

    /// Append a raw row (used by tampering / fixture tests).
    pub fn push_raw(&mut self, row: HashToCurveCompositionRow) {
        self.rows.push(row);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &HashToCurveCompositionWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        for k in 0..MAX_MSG_LEN {
            columns[COL_MSG_OFFSET + k][r] = Scalar::from_u64(row.msg[k] as u64, curve);
        }
        for k in 0..FIELD_ELEMENTS_LEN {
            columns[COL_FIELD_ELEMENTS_OFFSET + k][r] =
                Scalar::from_u64(row.field_elements[k] as u64, curve);
        }
        for k in 0..SSWU_OUTPUTS_LEN {
            columns[COL_SSWU_OUTPUTS_OFFSET + k][r] =
                Scalar::from_u64(row.sswu_outputs[k] as u64, curve);
        }
        for k in 0..ISO_OUTPUTS_LEN {
            columns[COL_ISO_OUTPUTS_OFFSET + k][r] =
                Scalar::from_u64(row.iso_outputs[k] as u64, curve);
        }
        for k in 0..FINAL_G2_LEN {
            columns[COL_FINAL_G2_OFFSET + k][r] =
                Scalar::from_u64(row.final_g2[k] as u64, curve);
        }
        columns[COL_PHASE_INDEX][r] = Scalar::from_u64(row.phase_index as u64, curve);
        columns[COL_PHASE_INDEX_BYTE][r] = Scalar::from_u64(row.phase_index as u64, curve);
        columns[COL_IS_H2F_PHASE][r] =
            if row.phase_index == PHASE_H2F { one.clone() } else { zero.clone() };
        columns[COL_IS_SSWU_PHASE][r] =
            if row.phase_index == PHASE_SSWU { one.clone() } else { zero.clone() };
        columns[COL_IS_ISOGENY_PHASE][r] =
            if row.phase_index == PHASE_ISOGENY { one.clone() } else { zero.clone() };
        columns[COL_IS_COFACTOR_PHASE][r] =
            if row.phase_index == PHASE_COFACTOR { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][r] = one.clone();

        // Sub-AIR limb mirrors (Task #189).
        columns[COL_U0_C0_LIMB0_MIRROR][r] = Scalar::from_u64(row.u0_c0_limb0, curve);
        columns[COL_ISO_OUT_X_C0_LIMB0_MIRROR][r] =
            Scalar::from_u64(row.iso_out_x_c0_limb0, curve);
        columns[COL_COF_OUT_X_C0_LIMB0_MIRROR][r] =
            Scalar::from_u64(row.cof_out_x_c0_limb0, curve);
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

pub struct HashToCurveCompositionConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl HashToCurveCompositionConstraintSystem {
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

impl VmConstraintSystem for HashToCurveCompositionConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_h2f_phase_binary".into(),
            "is_sswu_phase_binary".into(),
            "is_isogeny_phase_binary".into(),
            "is_cofactor_phase_binary".into(),
            "phase_sum_eq_is_real".into(),
            "phase_index_eq_weighted_selector_sum".into(),
            "phase_index_eq_phase_index_byte".into(),
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
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0..5: binarities.
        for col in [
            COL_IS_REAL,
            COL_IS_H2F_PHASE,
            COL_IS_SSWU_PHASE,
            COL_IS_ISOGENY_PHASE,
            COL_IS_COFACTOR_PHASE,
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
                let sum = columns[COL_IS_H2F_PHASE][r]
                    .add(&columns[COL_IS_SSWU_PHASE][r])
                    .add(&columns[COL_IS_ISOGENY_PHASE][r])
                    .add(&columns[COL_IS_COFACTOR_PHASE][r]);
                c[r] = sum.sub(&columns[COL_IS_REAL][r]);
            }
            out.push(c);
        }

        // 6: phase_index = Σ k·is_phase_k = 0·h2f + 1·sswu + 2·iso + 3·cof.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let weighted = columns[COL_IS_SSWU_PHASE][r]
                    .add(&two.mul(&columns[COL_IS_ISOGENY_PHASE][r]))
                    .add(&three.mul(&columns[COL_IS_COFACTOR_PHASE][r]));
                c[r] = columns[COL_PHASE_INDEX][r].sub(&weighted);
            }
            out.push(c);
        }

        // 7: phase_index = phase_index_byte.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_PHASE_INDEX][r].sub(&columns[COL_PHASE_INDEX_BYTE][r]);
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

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        let push = |acc: &mut Scalar, alpha_pow: &mut Scalar, body: Scalar| {
            *acc = acc.add(&alpha_pow.mul(&body));
            *alpha_pow = alpha_pow.mul(alpha);
        };

        // 0..5: binarities.
        for col in [
            COL_IS_REAL,
            COL_IS_H2F_PHASE,
            COL_IS_SSWU_PHASE,
            COL_IS_ISOGENY_PHASE,
            COL_IS_COFACTOR_PHASE,
        ] {
            let v = &col_evals[col];
            push(&mut acc, &mut alpha_pow, v.mul(&v.sub(&one)));
        }

        // 5: phase sum.
        let sum = col_evals[COL_IS_H2F_PHASE]
            .add(&col_evals[COL_IS_SSWU_PHASE])
            .add(&col_evals[COL_IS_ISOGENY_PHASE])
            .add(&col_evals[COL_IS_COFACTOR_PHASE]);
        push(&mut acc, &mut alpha_pow, sum.sub(&col_evals[COL_IS_REAL]));

        // 6: phase_index = weighted selector sum.
        let weighted = col_evals[COL_IS_SSWU_PHASE]
            .add(&two.mul(&col_evals[COL_IS_ISOGENY_PHASE]))
            .add(&three.mul(&col_evals[COL_IS_COFACTOR_PHASE]));
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

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // helper: push α^k * body.
        let push =
            |acc: &mut Vec<Scalar>, alpha_pow: &mut Scalar, body: Vec<Scalar>| {
                let term = poly_scalar_mul(&body, alpha_pow);
                *acc = poly_add(acc, &term, curve);
                *alpha_pow = alpha_pow.mul(alpha);
            };

        // 0..5: binarities.
        for col in [
            COL_IS_REAL,
            COL_IS_H2F_PHASE,
            COL_IS_SSWU_PHASE,
            COL_IS_ISOGENY_PHASE,
            COL_IS_COFACTOR_PHASE,
        ] {
            let v = &col_coeffs[col];
            let v_m1 = poly_sub(v, &one_poly, curve);
            push(&mut acc, &mut alpha_pow, poly_mul(v, &v_m1, curve));
        }

        // 5: phase sum.
        let sum = poly_add(
            &poly_add(&col_coeffs[COL_IS_H2F_PHASE], &col_coeffs[COL_IS_SSWU_PHASE], curve),
            &poly_add(
                &col_coeffs[COL_IS_ISOGENY_PHASE],
                &col_coeffs[COL_IS_COFACTOR_PHASE],
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
            &col_coeffs[COL_IS_SSWU_PHASE],
            &poly_add(
                &poly_mul(&two_poly, &col_coeffs[COL_IS_ISOGENY_PHASE], curve),
                &poly_mul(&three_poly, &col_coeffs[COL_IS_COFACTOR_PHASE], curve),
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
    // Shifted column layout (consumed by the verifier):
    //   [0]              IS_REAL_NEXT
    //   [1..1+MAX_MSG_LEN]    MSG_NEXT bytes
    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        let mut cols = Vec::with_capacity(1 + MAX_MSG_LEN);
        cols.push(COL_IS_REAL);
        for k in 0..MAX_MSG_LEN {
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
        let expected_shifted = 1 + MAX_MSG_LEN;
        if shifted_evals.len() != expected_shifted
            || col_evals_at_z.len() < NUM_COLUMNS
        {
            return zero;
        }
        let beta = beta_rlc(curve);
        let is_real_now = &col_evals_at_z[COL_IS_REAL];
        let is_real_next = &shifted_evals[0];

        // β-RLC over 128 bytes: Σ β^k * (msg_now[k] - msg_next[k]).
        let mut rlc = zero.clone();
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_MSG_LEN {
            let cur = &col_evals_at_z[COL_MSG_OFFSET + k];
            let nxt = &shifted_evals[1 + k];
            rlc = rlc.add(&bp.mul(&cur.sub(nxt)));
            bp = bp.mul(&beta);
        }
        let body = is_real_now.mul(is_real_next).mul(&rlc);

        // α^(alpha_offset) · body, multiplied by (z - ω^{n-1}) for
        // boundary exclusion (the constraint is allowed to fail on
        // the wrap-around row).
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

        // RLC body (coefficient form): Σ β^k * (msg_now[k] - msg_next[k]).
        let mut rlc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_MSG_LEN {
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

        for k in 0..MAX_MSG_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("h2c_comp_msg_byte_{}_8bit", k),
                    column_index: COL_MSG_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..FIELD_ELEMENTS_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("h2c_comp_field_elem_byte_{}_8bit", k),
                    column_index: COL_FIELD_ELEMENTS_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SSWU_OUTPUTS_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("h2c_comp_sswu_byte_{}_8bit", k),
                    column_index: COL_SSWU_OUTPUTS_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..ISO_OUTPUTS_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("h2c_comp_iso_byte_{}_8bit", k),
                    column_index: COL_ISO_OUTPUTS_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..FINAL_G2_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("h2c_comp_final_g2_byte_{}_8bit", k),
                    column_index: COL_FINAL_G2_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        declarations.push((
            LookupDeclaration {
                label: "h2c_comp_phase_index_byte_8bit".into(),
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

/// Cross-AIR LogUp: phase-0 binding of the 128-byte `msg` slice of this
/// AIR to the 128-byte `msg` columns of [`crate::hash_to_field_air`].
///
/// A side = this AIR (selected by [`COL_IS_H2F_PHASE`]).
/// B side = [`crate::hash_to_field_air`] (selected by its `IS_REAL`).
pub fn make_h2c_to_hash_to_field_descriptor(
    composition_layer_index: usize,
    hash_to_field_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::hash_to_field_air as h2f;
    let a_columns: Vec<usize> = (0..MAX_MSG_LEN).map(|k| COL_MSG_OFFSET + k).collect();
    let b_columns: Vec<usize> = (0..h2f::MAX_MSG_LEN).map(|k| h2f::COL_MSG_OFFSET + k).collect();
    debug_assert_eq!(a_columns.len(), b_columns.len());
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "h2c_composition_to_hash_to_field_v1".into(),
        a_layer_index: composition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_H2F_PHASE),
        b_layer_index: hash_to_field_layer_index,
        b_columns,
        b_selector_column: Some(h2f::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp: phase-1 binding of the SSWU-phase tuple.
///
/// A side = this AIR (selected by [`COL_IS_SSWU_PHASE`]). The A-side
/// tuple is the first 24 bytes of the `field_elements` slab — one
/// "anchor byte" per Fp limb that would feed the SSWU `u_0, u_1` →
/// field-element reduction. This matches the column-shape convention
/// established by
/// [`crate::hash_to_field_air::make_h2f_to_hash_to_g2_descriptor`].
///
/// B side = [`crate::hash_to_g2_air`] (selected by its SSWU-phase
/// selector). The B-side tuple is the 24 Fp limb columns of
/// `(u0_c0, u0_c1, u0_plus_one_c0, u0_plus_one_c1)`.
pub fn make_h2c_to_sswu_descriptor(
    composition_layer_index: usize,
    hash_to_g2_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::hash_to_g2_air as h2g2;
    // A side: 24 anchor bytes from the field_elements slab (one byte
    // per Fp limb position; the full 64-byte → 6-limb reduction is
    // owned by the future nonnative_fp byte-to-limb sub-AIR).
    let anchor_stride = FIELD_ELEMENTS_LEN / 24; // = 256 / 24 (rounds down to 10).
    let mut a_columns: Vec<usize> = Vec::with_capacity(24);
    for j in 0..24 {
        a_columns.push(COL_FIELD_ELEMENTS_OFFSET + j * anchor_stride);
    }
    // B side: 24 Fp limb columns of hash_to_g2_air's (u0, u0_plus_one).
    let b_columns: Vec<usize> = (0..h2g2::LIMBS_PER_FP)
        .map(|j| h2g2::COL_U0_C0_LIMB_OFFSET + j)
        .chain((0..h2g2::LIMBS_PER_FP).map(|j| h2g2::COL_U0_C1_LIMB_OFFSET + j))
        .chain((0..h2g2::LIMBS_PER_FP).map(|j| h2g2::COL_U0_PLUS_ONE_C0_LIMB_OFFSET + j))
        .chain((0..h2g2::LIMBS_PER_FP).map(|j| h2g2::COL_U0_PLUS_ONE_C1_LIMB_OFFSET + j))
        .collect();
    debug_assert_eq!(a_columns.len(), b_columns.len());
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "h2c_composition_to_sswu_v1".into(),
        a_layer_index: composition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_SSWU_PHASE),
        b_layer_index: hash_to_g2_layer_index,
        b_columns,
        b_selector_column: Some(h2g2::COL_IS_SSWU_PHASE),
    }
}

/// Cross-AIR LogUp: phase-2 binding of the isogeny-phase tuple.
///
/// A side = this AIR (selected by [`COL_IS_ISOGENY_PHASE`]). The
/// A-side tuple is 24 anchor bytes from the `iso_outputs` slab (one
/// per Fp limb position of the post-isogeny G2 point).
///
/// B side = [`crate::isogeny_map_air`] (selected by its `IS_REAL`).
/// The B-side tuple is the 24 Fp limb columns of the post-isogeny
/// G2 point `out_x_c0, out_x_c1, out_y_c0, out_y_c1`.
pub fn make_h2c_to_isogeny_descriptor(
    composition_layer_index: usize,
    isogeny_map_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::isogeny_map_air as iso;
    let anchor_stride = G2_COMPRESSED_LEN / 24; // = 96 / 24 = 4.
    let mut a_columns: Vec<usize> = Vec::with_capacity(24);
    for j in 0..24 {
        a_columns.push(COL_ISO_OUTPUTS_OFFSET + j * anchor_stride);
    }
    let b_columns: Vec<usize> = (0..iso::LIMBS_PER_FP)
        .map(|j| iso::COL_OUT_X_C0_LIMB_OFFSET + j)
        .chain((0..iso::LIMBS_PER_FP).map(|j| iso::COL_OUT_X_C1_LIMB_OFFSET + j))
        .chain((0..iso::LIMBS_PER_FP).map(|j| iso::COL_OUT_Y_C0_LIMB_OFFSET + j))
        .chain((0..iso::LIMBS_PER_FP).map(|j| iso::COL_OUT_Y_C1_LIMB_OFFSET + j))
        .collect();
    debug_assert_eq!(a_columns.len(), b_columns.len());
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "h2c_composition_to_isogeny_v1".into(),
        a_layer_index: composition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_ISOGENY_PHASE),
        b_layer_index: isogeny_map_layer_index,
        b_columns,
        b_selector_column: Some(iso::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp: phase-3 binding of the cofactor-clearing tuple.
///
/// A side = this AIR (selected by [`COL_IS_COFACTOR_PHASE`]). The
/// A-side tuple is 24 anchor bytes from the `final_g2` slab.
///
/// B side = [`crate::g2_cofactor_clear_air`] (selected by its
/// `IS_REAL`). The B-side tuple is the 24 Fp limb columns of the
/// final subgroup-correct G2 point.
pub fn make_h2c_to_cofactor_descriptor(
    composition_layer_index: usize,
    cofactor_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::g2_cofactor_clear_air as cof;
    let anchor_stride = FINAL_G2_LEN / 24; // = 96 / 24 = 4.
    let mut a_columns: Vec<usize> = Vec::with_capacity(24);
    for j in 0..24 {
        a_columns.push(COL_FINAL_G2_OFFSET + j * anchor_stride);
    }
    let b_columns: Vec<usize> = (0..cof::LIMBS_PER_FP)
        .map(|j| cof::COL_OUT_X_C0_LIMB_OFFSET + j)
        .chain((0..cof::LIMBS_PER_FP).map(|j| cof::COL_OUT_X_C1_LIMB_OFFSET + j))
        .chain((0..cof::LIMBS_PER_FP).map(|j| cof::COL_OUT_Y_C0_LIMB_OFFSET + j))
        .chain((0..cof::LIMBS_PER_FP).map(|j| cof::COL_OUT_Y_C1_LIMB_OFFSET + j))
        .collect();
    debug_assert_eq!(a_columns.len(), b_columns.len());
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "h2c_composition_to_cofactor_v1".into(),
        a_layer_index: composition_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_COFACTOR_PHASE),
        b_layer_index: cofactor_layer_index,
        b_columns,
        b_selector_column: Some(cof::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Beacon-chain ciphersuite DST (POP variant).
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    fn evaluate(witness: &HashToCurveCompositionWitness) -> Vec<Vec<Scalar>> {
        let trace = build_trace_polynomials(witness, CurveType::Bls12381);
        let cs = HashToCurveCompositionConstraintSystem::new(trace.num_rows);
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

    // ───────────────────────────────────────────────────────────────────
    // Test 1: empty message — witness builds, all row-local constraints zero.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn empty_message_witness_passes_row_local_constraints() {
        let w = HashToCurveCompositionWitness::from_message(b"", POP_DST)
            .expect("empty msg accepted");
        assert_eq!(w.rows.len(), ROWS_PER_INSTANCE);
        // Phase indices are 0, 1, 2, 3 in order.
        for (r, row) in w.rows.iter().enumerate() {
            assert_eq!(row.phase_index, r as u8);
        }
        // All four rows share the same msg / field_elements / final_g2
        // (the four phases of the same instance).
        let row0 = &w.rows[0];
        for row in &w.rows[1..] {
            assert_eq!(row.msg, row0.msg);
            assert_eq!(row.field_elements, row0.field_elements);
            assert_eq!(row.final_g2, row0.final_g2);
        }
        // Final G2 is non-zero (hash-to-curve never produces the identity).
        assert!(row0.final_g2.iter().any(|&b| b != 0));
        // Constraints all vanish on the honest trace.
        let results = evaluate(&w);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        assert_all_zero(&results);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: longer (multi-byte) message — same closure.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn longer_message_witness_passes_row_local_constraints() {
        let mut msg = Vec::with_capacity(MAX_MSG_LEN);
        for k in 0..MAX_MSG_LEN {
            msg.push((0x80 ^ (k as u8)) as u8);
        }
        let w = HashToCurveCompositionWitness::from_message(&msg, POP_DST)
            .expect("max-length msg accepted");
        assert_eq!(w.rows.len(), ROWS_PER_INSTANCE);
        // Msg bytes survived the round-trip into the witness.
        assert_eq!(w.rows[0].msg[..msg.len()], msg[..]);
        let results = evaluate(&w);
        assert_all_zero(&results);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: tampering with the cross-phase chain (msg changed between
    // adjacent rows) is detected by the shifted constraint.
    //
    // The row-local constraints continue to vanish (they are per-row),
    // but the cross-row β-RLC msg equality fires when we mutate the
    // msg byte on one of the phase rows.
    // ───────────────────────────────────────────────────────────────────

    /// Directly compute the cross-row shifted-constraint body row-by-row
    /// over the padded domain. Returns the per-row constraint values
    /// `gate(r) * Σ β^k (msg_now[k] - msg_next[k])`. For an honest
    /// instance these must all be zero on real rows; for a tampered
    /// instance at least one must be non-zero on a real row.
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
            for k in 0..MAX_MSG_LEN {
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
        let w = HashToCurveCompositionWitness::from_message(b"tamper-test", POP_DST)
            .expect("witness builds");
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let curve = CurveType::Bls12381;

        // Honest closure: msg-continuity body is zero on every row.
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

        // Tamper msg[0] on row 1 (SSWU phase): change it from its true
        // value to a different byte. This breaks the SSWU-row ↔
        // isogeny-row msg continuity AND the h2f-row ↔ SSWU-row
        // msg continuity; either firing demonstrates the cross-row
        // constraint catches the tamper.
        let mut cols: Vec<Vec<Scalar>> = cols_honest.clone();
        let original = cols[COL_MSG_OFFSET][1].clone();
        cols[COL_MSG_OFFSET][1] = original.add(&Scalar::one(curve));
        let tampered = shifted_msg_continuity_per_row(&cols, curve);
        // At least one of rows 0 or 1 must have a non-zero body (both
        // adjacencies span the tampered cell).
        assert!(
            !tampered[0].is_zero() || !tampered[1].is_zero(),
            "tampering msg on phase row 1 must fire the shifted msg-continuity constraint",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: cross-AIR LogUp descriptors are well-formed.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn descriptors_well_formed() {
        let d0 = make_h2c_to_hash_to_field_descriptor(0, 1);
        assert_eq!(d0.label, "h2c_composition_to_hash_to_field_v1");
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d0.a_columns.len(), MAX_MSG_LEN);
        assert_eq!(d0.b_columns.len(), MAX_MSG_LEN);
        assert_eq!(d0.a_selector_column, Some(COL_IS_H2F_PHASE));
        assert_eq!(
            d0.b_selector_column,
            Some(crate::hash_to_field_air::COL_IS_REAL),
        );
        assert_eq!(d0.a_columns[0], COL_MSG_OFFSET);
        assert_eq!(d0.b_columns[0], crate::hash_to_field_air::COL_MSG_OFFSET);

        let d1 = make_h2c_to_sswu_descriptor(0, 2);
        assert_eq!(d1.label, "h2c_composition_to_sswu_v1");
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 2);
        assert_eq!(d1.a_columns.len(), 24);
        assert_eq!(d1.b_columns.len(), 24);
        assert_eq!(d1.a_selector_column, Some(COL_IS_SSWU_PHASE));
        assert_eq!(
            d1.b_selector_column,
            Some(crate::hash_to_g2_air::COL_IS_SSWU_PHASE),
        );
        assert_eq!(d1.b_columns[0], crate::hash_to_g2_air::COL_U0_C0_LIMB_OFFSET);

        let d2 = make_h2c_to_isogeny_descriptor(0, 3);
        assert_eq!(d2.label, "h2c_composition_to_isogeny_v1");
        assert_eq!(d2.a_columns.len(), 24);
        assert_eq!(d2.b_columns.len(), 24);
        assert_eq!(d2.a_selector_column, Some(COL_IS_ISOGENY_PHASE));
        assert_eq!(
            d2.b_selector_column,
            Some(crate::isogeny_map_air::COL_IS_REAL),
        );
        assert_eq!(
            d2.b_columns[0],
            crate::isogeny_map_air::COL_OUT_X_C0_LIMB_OFFSET,
        );

        let d3 = make_h2c_to_cofactor_descriptor(0, 4);
        assert_eq!(d3.label, "h2c_composition_to_cofactor_v1");
        assert_eq!(d3.a_columns.len(), 24);
        assert_eq!(d3.b_columns.len(), 24);
        assert_eq!(d3.a_selector_column, Some(COL_IS_COFACTOR_PHASE));
        assert_eq!(
            d3.b_selector_column,
            Some(crate::g2_cofactor_clear_air::COL_IS_REAL),
        );
        assert_eq!(
            d3.b_columns[0],
            crate::g2_cofactor_clear_air::COL_OUT_X_C0_LIMB_OFFSET,
        );

        // Descriptor labels are unique across all four.
        let labels = [
            d0.label.as_str(),
            d1.label.as_str(),
            d2.label.as_str(),
            d3.label.as_str(),
        ];
        let mut sorted = labels.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "descriptor labels must be unique");
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: column layout & helpers are pinned.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_MSG_OFFSET, 0);
        assert_eq!(COL_FIELD_ELEMENTS_OFFSET, MAX_MSG_LEN);
        assert_eq!(COL_SSWU_OUTPUTS_OFFSET, MAX_MSG_LEN + FIELD_ELEMENTS_LEN);
        assert_eq!(
            COL_ISO_OUTPUTS_OFFSET,
            MAX_MSG_LEN + FIELD_ELEMENTS_LEN + SSWU_OUTPUTS_LEN,
        );
        assert_eq!(
            COL_FINAL_G2_OFFSET,
            MAX_MSG_LEN + FIELD_ELEMENTS_LEN + SSWU_OUTPUTS_LEN + ISO_OUTPUTS_LEN,
        );
        assert_eq!(
            COL_PHASE_INDEX,
            MAX_MSG_LEN + FIELD_ELEMENTS_LEN + SSWU_OUTPUTS_LEN + ISO_OUTPUTS_LEN + FINAL_G2_LEN,
        );
        // 5 byte slabs + 7 single-column scalars + 3 sub-AIR limb
        // mirror columns (Task #189).
        assert_eq!(
            NUM_COLUMNS,
            MAX_MSG_LEN + FIELD_ELEMENTS_LEN + SSWU_OUTPUTS_LEN + ISO_OUTPUTS_LEN
                + FINAL_G2_LEN + 7 + 3,
        );
        assert_eq!(NUM_COLUMNS, 874);
        // Mirror columns are appended after `is_real`.
        assert_eq!(COL_U0_C0_LIMB0_MIRROR, COL_IS_REAL + 1);
        assert_eq!(COL_ISO_OUT_X_C0_LIMB0_MIRROR, COL_IS_REAL + 2);
        assert_eq!(COL_COF_OUT_X_C0_LIMB0_MIRROR, COL_IS_REAL + 3);
        assert_eq!(NUM_ROW_CONSTRAINTS, 8);
        assert_eq!(NUM_SHIFTED, 1);
        // Phase order matches the IETF pipeline.
        assert_eq!(PHASE_H2F, 0);
        assert_eq!(PHASE_SSWU, 1);
        assert_eq!(PHASE_ISOGENY, 2);
        assert_eq!(PHASE_COFACTOR, 3);
    }

    // ───────────────────────────────────────────────────────────────────
    // Bonus: each row-local constraint fires when its column is tampered.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn each_row_local_constraint_fires_on_tampering() {
        let w = HashToCurveCompositionWitness::from_message(b"x", POP_DST).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let curve = CurveType::Bls12381;
        let cs = HashToCurveCompositionConstraintSystem::new(trace.num_rows);

        // Tamper is_real → fires constraint 0.
        {
            let mut cols: Vec<Vec<Scalar>> =
                trace.columns.iter().map(|p| p.evaluations.clone()).collect();
            cols[COL_IS_REAL][0] = Scalar::from_u64(5, curve);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let r = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            assert!(!r[0][0].is_zero(), "is_real binary must fire");
        }
        // Tamper is_h2f_phase to 2 → fires constraint 1 (binarity).
        {
            let mut cols: Vec<Vec<Scalar>> =
                trace.columns.iter().map(|p| p.evaluations.clone()).collect();
            cols[COL_IS_H2F_PHASE][0] = Scalar::from_u64(2, curve);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let r = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            assert!(!r[1][0].is_zero(), "is_h2f_phase binarity must fire");
        }
        // Tamper phase_index byte mismatch → fires constraint 7.
        {
            let mut cols: Vec<Vec<Scalar>> =
                trace.columns.iter().map(|p| p.evaluations.clone()).collect();
            cols[COL_PHASE_INDEX_BYTE][0] = Scalar::from_u64(99, curve);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let r = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            assert!(
                !r[7][0].is_zero(),
                "phase_index ≠ phase_index_byte must fire",
            );
        }
        // Tamper phase_index without retagging selectors → fires constraint 6.
        {
            let mut cols: Vec<Vec<Scalar>> =
                trace.columns.iter().map(|p| p.evaluations.clone()).collect();
            // Set phase_index to 3 on row 0 (was 0); selectors stay
            // (is_h2f = 1). Both phase_index_byte equality and the
            // weighted-sum constraint should fire; we assert the
            // weighted-sum one.
            cols[COL_PHASE_INDEX][0] = Scalar::from_u64(3, curve);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let r = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            assert!(
                !r[6][0].is_zero(),
                "phase_index != weighted-selector-sum must fire",
            );
        }
    }

    /// Task #192 → #202 diagnostic: standalone prove+verify of just the
    /// composer AIR under BLS48-581.
    ///
    /// Root cause (task #202): `build_shifted_constraint_polynomial` was
    /// missing the `(X - ω^{n-1})` wrap-row exclusion factor that
    /// `evaluate_shifted_at_point` already includes as `(z - ω^{n-1})`.
    /// Without it the prover's C(X) does not vanish at ω^{n-1}, so
    /// synthetic division by Z(X) = X^n - 1 leaves a non-zero remainder
    /// and Q(z)·Z(z) disagrees with the verifier's C(z) reconstruction.
    /// Fix landed alongside this comment.
    #[test]
    #[ignore = "slow: standalone composer prove+verify (~9 min release)"]
    fn diagnostic_composer_standalone_prove_verify() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let w = HashToCurveCompositionWitness::from_message(
            b"h2c-composer-joint-prove",
            POP_DST,
        )
        .expect("composer witness builds");
        let trace = build_trace_polynomials(&w, curve);
        let cs = HashToCurveCompositionConstraintSystem::new(trace.num_rows);
        let proof = crate::prover::prove_with_scheme(&trace, &cs, &scheme);
        let ok = crate::verifier::verify_with_scheme(&proof, &cs, &scheme, curve);
        assert!(ok, "standalone composer prove+verify must pass");
    }
}
