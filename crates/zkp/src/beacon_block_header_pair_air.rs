//! Beacon block header pair-invocation AIR — Phase C3 step 1.
//!
//! Per-row exposes one `sha256_pair(left, right) -> hash` invocation
//! drawn from a [`BeaconBlockHeaderHtrWitness`]. Companion to step 0's
//! [`crate::beacon_block_header_air`] witness builder.
//!
//! Each row carries 96 byte columns (`LEFT[32] || RIGHT[32] || HASH[32]`)
//! plus `IS_REAL`. The cross-AIR LogUp descriptor
//! `make_beacon_block_header_pair_to_sha256_extract_linkage_descriptor`
//! binds each row's `(left||right, hash)` tuple to a real
//! [`crate::sha256_extract`] invocation, transitively pulling in the
//! bit-level SHA-256 binding.
//!
//! # Soundness scope (step 1)
//!
//! - Proves: each row's `(input_64, output_32)` matches a row in
//!   Sha256Extract (which itself binds to bit-level SHA-256).
//! - Does NOT prove (deferred to step 2):
//!   - Layer chaining: `layer2[i].left = layer1[2i].hash` and
//!     `root.left = layer2[0].hash` (cross-row equalities by index).
//!   - Leaf binding: `leaf[2..5]` equal the BeaconBlockHeader's
//!     `parent_root`, `state_root`, `body_root` columns; `leaf[0..2]`
//!     are `htr_uint(slot)` / `htr_uint(proposer_index)` shapes.
//!   - Zero-padding pin: `leaf[5..8] = ZERO_CHUNK` and the resulting
//!     pair-shape pins on `layer1[2..4]`.
//!
//! Step 2 will add cross-row equality constraints (using `NUM_SHIFTED`
//! bodies analogous to MPT's chain), and a host-built order convention
//! that places the 7 invocations in a canonical row order so the
//! algebraic chain checks at fixed row offsets.

use crate::beacon_block_header_air::{BeaconBlockHeaderHtrWitness, Sha256PairInvocation};
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_LEFT_OFFSET: usize = 0;          // 0..32
pub const COL_RIGHT_OFFSET: usize = 32;        // 32..64
pub const COL_HASH_OFFSET: usize = 64;         // 64..96
pub const COL_IS_REAL: usize = 96;

// Steps 2a+2c+2d: row-positional one-hot selectors for leaf binding.
//   IS_PAIR0 = 1 only on row index 0 (= sha256(htr_uint(slot), htr_uint(proposer_index)))
//   IS_PAIR1 = 1 only on row index 1 (= sha256(parent_root, state_root))
//   IS_PAIR2 = 1 only on row index 2 (= sha256(body_root, ZERO_CHUNK))
//   IS_PAIR3 = 1 only on row index 3 (= sha256(ZERO_CHUNK, ZERO_CHUNK))
pub const COL_IS_PAIR1: usize = 97;
pub const COL_IS_PAIR2: usize = 98;

// Step 2a: claimed BeaconBlockHeader Bytes32 fields, replicated identically
// across all rows (cross-row constancy enforced via NUM_SHIFTED bodies).
pub const COL_PARENT_ROOT_OFFSET: usize = 99;   // 99..131
pub const COL_STATE_ROOT_OFFSET: usize = 131;   // 131..163
pub const COL_BODY_ROOT_OFFSET: usize = 163;    // 163..195

// Step 2c+2d additions.
pub const COL_IS_PAIR0: usize = 195;
pub const COL_IS_PAIR3: usize = 196;

// uint64 leaves as 8 LE bytes each (htr_uint = 8-byte LE || 24-byte zero).
pub const COL_SLOT_BYTE_OFFSET: usize = 197;          // 197..205 (8 bytes)
pub const COL_PROPOSER_INDEX_BYTE_OFFSET: usize = 205; // 205..213 (8 bytes)

// Step 2b-min: row-positional one-hot selectors for the 3 consumer rows.
// IS_PAIR4 = 1 only on row 4 (layer2 pair 0, consumes HASH[0] || HASH[1]).
// IS_PAIR5 = 1 only on row 5 (layer2 pair 1, consumes HASH[2] || HASH[3]).
// IS_ROOT  = 1 only on row 6 (root, consumes HASH[4] || HASH[5]).
pub const COL_IS_PAIR4: usize = 213;
pub const COL_IS_PAIR5: usize = 214;
pub const COL_IS_ROOT: usize = 215;

// Step 2b-offset2: relay column for row 4 HASH → row 6 LEFT binding.
// At row 4 (pickup): REL4_HASH[r] = HASH[r] (β-RLC equality).
// At rows 4 and 5 (propagate): REL4_HASH[r+1] = REL4_HASH[r] (shifted
// constancy gated by IS_PAIR4 + IS_PAIR5).
// At row 6 (consumption): LEFT[r] = REL4_HASH[r] (β-RLC equality).
pub const COL_REL4_HASH_OFFSET: usize = 216;  // 216..248 (32 bytes)

// Step 2b-full: relay columns for the remaining 4 chain bindings.
// Each follows the same pickup/constancy/consumption pattern.
pub const COL_REL0_HASH_OFFSET: usize = 248;  // 248..280: Row 0 HASH → Row 4 LEFT  (offset −4)
pub const COL_REL1_HASH_OFFSET: usize = 280;  // 280..312: Row 1 HASH → Row 4 RIGHT (offset −3)
pub const COL_REL2_HASH_OFFSET: usize = 312;  // 312..344: Row 2 HASH → Row 5 LEFT  (offset −3)
pub const COL_REL3_HASH_OFFSET: usize = 344;  // 344..376: Row 3 HASH → Row 5 RIGHT (offset −2)

// Step 2e: stable output column exposing the BBH HTR root, replicated
// across all real rows. Downstream AIRs (finality, validator chain,
// etc.) link against this column instead of having to access row 6's
// HASH directly.
pub const COL_CLAIMED_ROOT_OFFSET: usize = 376;  // 376..408 (32 bytes)

pub const NUM_COLUMNS: usize = 408;

// Row-locals:
//   0:  is_real binary
//   1:  is_pair1 binary
//   2:  is_pair2 binary
//   3:  is_pair1 * (LEFT[r] - PARENT_ROOT[r]) — β-RLC over 32 bytes
//   4:  is_pair1 * (RIGHT[r] - STATE_ROOT[r]) — β-RLC over 32 bytes
//   5:  is_pair2 * (LEFT[r] - BODY_ROOT[r]) — β-RLC over 32 bytes
//   6:  is_pair2 * RIGHT[r] (RIGHT must equal ZERO_CHUNK at row 2) — β-RLC
//   7:  is_pair0 binary
//   8:  is_pair3 binary
//   9:  is_pair0 * (LEFT[0..8] - SLOT_BYTES) — β-RLC over 8 bytes
//   10: is_pair0 * LEFT[8..32] — β-RLC over 24 bytes (htr_uint zero-tail)
//   11: is_pair0 * (RIGHT[0..8] - PROPOSER_INDEX_BYTES) — β-RLC over 8 bytes
//   12: is_pair0 * RIGHT[8..32] — β-RLC over 24 bytes
//   13: is_pair3 * LEFT[r] — β-RLC over 32 bytes (ZERO_CHUNK)
//   14: is_pair3 * RIGHT[r] — β-RLC over 32 bytes (ZERO_CHUNK)
//   15: is_pair4 binary
//   16: is_pair5 binary
//   17: is_root binary
//   18: is_pair4 * Σ β^k * (REL4_HASH[k] - HASH[k]) — pickup at row 4
//   19: is_root  * Σ β^k * (LEFT[k] - REL4_HASH[k]) — consumption at row 6
//   20: is_pair0 * Σ β^k * (REL0_HASH[k] - HASH[k]) — pickup at row 0
//   21: is_pair4 * Σ β^k * (LEFT[k]  - REL0_HASH[k]) — consumption at row 4 LEFT
//   22: is_pair1 * Σ β^k * (REL1_HASH[k] - HASH[k]) — pickup at row 1
//   23: is_pair4 * Σ β^k * (RIGHT[k] - REL1_HASH[k]) — consumption at row 4 RIGHT
//   24: is_pair2 * Σ β^k * (REL2_HASH[k] - HASH[k]) — pickup at row 2
//   25: is_pair5 * Σ β^k * (LEFT[k]  - REL2_HASH[k]) — consumption at row 5 LEFT
//   26: is_pair3 * Σ β^k * (REL3_HASH[k] - HASH[k]) — pickup at row 3
//   27: is_pair5 * Σ β^k * (RIGHT[k] - REL3_HASH[k]) — consumption at row 5 RIGHT
//   28: is_root  * Σ β^k * (HASH[k] - CLAIMED_ROOT[k]) — output binding at row 6
pub const NUM_ROW_CONSTRAINTS: usize = 29;

// Shifted (cross-row constancy of the 5 claimed-field column groups,
// gated by IS_REAL[r] * IS_REAL[r+1]):
//   0: parent_root_byte_constancy    — β-RLC over 32 bytes
//   1: state_root_byte_constancy     — β-RLC over 32 bytes
//   2: body_root_byte_constancy      — β-RLC over 32 bytes
//   3: slot_byte_constancy           — β-RLC over 8 bytes
//   4: proposer_index_byte_constancy — β-RLC over 8 bytes
//   5: root_right_eq_pair5_hash      — IS_PAIR5(r) * IS_ROOT(r+1) *
//                                      Σ β^k * (RIGHT_NEXT[k] - HASH[k])
//   6: rel4_hash_constancy           — (IS_PAIR4 + IS_PAIR5) * β-RLC
//   7: rel0_hash_constancy           — (IS_PAIR0 + IS_PAIR1 + IS_PAIR2 + IS_PAIR3) * β-RLC
//   8: rel1_hash_constancy           — (IS_PAIR1 + IS_PAIR2 + IS_PAIR3) * β-RLC
//   9: rel2_hash_constancy           — (IS_PAIR2 + IS_PAIR3 + IS_PAIR4) * β-RLC
//  10: rel3_hash_constancy           — (IS_PAIR3 + IS_PAIR4) * β-RLC
//  11: claimed_root_constancy        — IS_REAL[r] * IS_REAL[r+1] * β-RLC over 32
pub const NUM_SHIFTED: usize = 12;

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BeaconBlockHeaderHtrWitness,
    curve: CurveType,
) -> TracePolynomials {
    build_trace_polynomials_with_header_fields(
        &witness.invocations,
        witness.header.slot,
        witness.header.proposer_index,
        witness.header.parent_root,
        witness.header.state_root,
        witness.header.body_root,
        curve,
    )
}

/// Lower-level builder that takes invocations + all 5 claimed
/// BeaconBlockHeader fields directly. Used by tests that need to
/// construct adversarial witnesses.
pub fn build_trace_polynomials_with_header_fields(
    invocations: &[Sha256PairInvocation],
    slot: u64,
    proposer_index: u64,
    parent_root: [u8; 32],
    state_root: [u8; 32],
    body_root: [u8; 32],
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, inv) in invocations.iter().enumerate() {
        for k in 0..32 {
            columns[COL_LEFT_OFFSET + k][i] = Scalar::from_u64(inv.left[k] as u64, curve);
            columns[COL_RIGHT_OFFSET + k][i] = Scalar::from_u64(inv.right[k] as u64, curve);
            columns[COL_HASH_OFFSET + k][i] = Scalar::from_u64(inv.hash[k] as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
    }

    // Steps 2a+2c+2d+2b-min: row-positional one-hot selectors, set on real rows.
    if num_rows > 0 { columns[COL_IS_PAIR0][0] = one.clone(); }
    if num_rows > 1 { columns[COL_IS_PAIR1][1] = one.clone(); }
    if num_rows > 2 { columns[COL_IS_PAIR2][2] = one.clone(); }
    if num_rows > 3 { columns[COL_IS_PAIR3][3] = one.clone(); }
    if num_rows > 4 { columns[COL_IS_PAIR4][4] = one.clone(); }
    if num_rows > 5 { columns[COL_IS_PAIR5][5] = one.clone(); }
    if num_rows > 6 { columns[COL_IS_ROOT][6] = one.clone(); }

    // Step 2b-offset2: populate REL4_HASH at rows 4..6 (pickup at row 4).
    if num_rows > 4 {
        let h = invocations[4].hash;
        for k in 0..32 {
            let v = Scalar::from_u64(h[k] as u64, curve);
            for r in 4..num_rows.min(7) {
                columns[COL_REL4_HASH_OFFSET + k][r] = v.clone();
            }
        }
    }

    // Step 2b-full: populate the remaining 4 relays.
    // REL0_HASH: rows 0..4 = HASH from invocation 0.
    if num_rows > 0 {
        let h = invocations[0].hash;
        for k in 0..32 {
            let v = Scalar::from_u64(h[k] as u64, curve);
            for r in 0..num_rows.min(5) {
                columns[COL_REL0_HASH_OFFSET + k][r] = v.clone();
            }
        }
    }
    // REL1_HASH: rows 1..4 = HASH from invocation 1.
    if num_rows > 1 {
        let h = invocations[1].hash;
        for k in 0..32 {
            let v = Scalar::from_u64(h[k] as u64, curve);
            for r in 1..num_rows.min(5) {
                columns[COL_REL1_HASH_OFFSET + k][r] = v.clone();
            }
        }
    }
    // REL2_HASH: rows 2..5 = HASH from invocation 2.
    if num_rows > 2 {
        let h = invocations[2].hash;
        for k in 0..32 {
            let v = Scalar::from_u64(h[k] as u64, curve);
            for r in 2..num_rows.min(6) {
                columns[COL_REL2_HASH_OFFSET + k][r] = v.clone();
            }
        }
    }
    // REL3_HASH: rows 3..5 = HASH from invocation 3.
    if num_rows > 3 {
        let h = invocations[3].hash;
        for k in 0..32 {
            let v = Scalar::from_u64(h[k] as u64, curve);
            for r in 3..num_rows.min(6) {
                columns[COL_REL3_HASH_OFFSET + k][r] = v.clone();
            }
        }
    }

    // Step 2e: claimed root = HASH from the root invocation (row 6),
    // replicated across all real rows for cross-row constancy.
    if num_rows > 6 {
        let root_hash = invocations[6].hash;
        for k in 0..32 {
            let v = Scalar::from_u64(root_hash[k] as u64, curve);
            for r in 0..num_rows {
                columns[COL_CLAIMED_ROOT_OFFSET + k][r] = v.clone();
            }
        }
    }

    // Step 2a: replicate the 3 claimed Bytes32 leaves identically across
    // every real row (and zero on padding so cross-row constancy gated
    // by IS_REAL[r]*IS_REAL[r+1] vanishes at the boundary).
    for k in 0..32 {
        let p = Scalar::from_u64(parent_root[k] as u64, curve);
        let s = Scalar::from_u64(state_root[k] as u64, curve);
        let b = Scalar::from_u64(body_root[k] as u64, curve);
        for r in 0..num_rows {
            columns[COL_PARENT_ROOT_OFFSET + k][r] = p.clone();
            columns[COL_STATE_ROOT_OFFSET + k][r] = s.clone();
            columns[COL_BODY_ROOT_OFFSET + k][r] = b.clone();
        }
    }

    // Step 2c: replicate the 2 uint64 leaves as 8 LE bytes each.
    let slot_bytes = slot.to_le_bytes();
    let pi_bytes = proposer_index.to_le_bytes();
    for k in 0..8 {
        let s = Scalar::from_u64(slot_bytes[k] as u64, curve);
        let p = Scalar::from_u64(pi_bytes[k] as u64, curve);
        for r in 0..num_rows {
            columns[COL_SLOT_BYTE_OFFSET + k][r] = s.clone();
            columns[COL_PROPOSER_INDEX_BYTE_OFFSET + k][r] = p.clone();
        }
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

/// Backwards-compatible witness-only builder; uses zero-leaves for the
/// claimed fields (used only by old tests where header-field
/// constraints aren't relevant — see step 2a tests for the proper API).
pub fn build_trace_polynomials_from_invocations(
    invocations: &[Sha256PairInvocation],
    curve: CurveType,
) -> TracePolynomials {
    build_trace_polynomials_with_header_fields(
        invocations, 0, 0, [0u8; 32], [0u8; 32], [0u8; 32], curve,
    )
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct BeaconBlockHeaderPairConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BeaconBlockHeaderPairConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for BeaconBlockHeaderPairConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_pair1_binary".into(),
            "is_pair2_binary".into(),
            "pair1_left_eq_parent_root".into(),
            "pair1_right_eq_state_root".into(),
            "pair2_left_eq_body_root".into(),
            "pair2_right_eq_zero_chunk".into(),
            "is_pair0_binary".into(),
            "is_pair3_binary".into(),
            "pair0_left_low_eq_slot_le".into(),
            "pair0_left_high_eq_zero".into(),
            "pair0_right_low_eq_proposer_index_le".into(),
            "pair0_right_high_eq_zero".into(),
            "pair3_left_eq_zero_chunk".into(),
            "pair3_right_eq_zero_chunk".into(),
            "is_pair4_binary".into(),
            "is_pair5_binary".into(),
            "is_root_binary".into(),
            "rel4_hash_pickup_at_pair4".into(),
            "root_left_eq_rel4_hash".into(),
            "rel0_hash_pickup_at_pair0".into(),
            "pair4_left_eq_rel0_hash".into(),
            "rel1_hash_pickup_at_pair1".into(),
            "pair4_right_eq_rel1_hash".into(),
            "rel2_hash_pickup_at_pair2".into(),
            "pair5_left_eq_rel2_hash".into(),
            "rel3_hash_pickup_at_pair3".into(),
            "pair5_right_eq_rel3_hash".into(),
            "root_hash_eq_claimed_root".into(),
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
        let beta_test = Scalar::from_u64(7, curve);

        let mut is_real_bin = vec![Scalar::zero(curve); n];
        let mut is_pair1_bin = vec![Scalar::zero(curve); n];
        let mut is_pair2_bin = vec![Scalar::zero(curve); n];
        let mut pair1_left_eq = vec![Scalar::zero(curve); n];
        let mut pair1_right_eq = vec![Scalar::zero(curve); n];
        let mut pair2_left_eq = vec![Scalar::zero(curve); n];
        let mut pair2_right_zero = vec![Scalar::zero(curve); n];
        let mut is_pair0_bin = vec![Scalar::zero(curve); n];
        let mut is_pair3_bin = vec![Scalar::zero(curve); n];
        let mut pair0_l_lo_eq_slot = vec![Scalar::zero(curve); n];
        let mut pair0_l_hi_eq_zero = vec![Scalar::zero(curve); n];
        let mut pair0_r_lo_eq_pi = vec![Scalar::zero(curve); n];
        let mut pair0_r_hi_eq_zero = vec![Scalar::zero(curve); n];
        let mut pair3_l_eq_zero = vec![Scalar::zero(curve); n];
        let mut pair3_r_eq_zero = vec![Scalar::zero(curve); n];
        let mut is_pair4_bin = vec![Scalar::zero(curve); n];
        let mut is_pair5_bin = vec![Scalar::zero(curve); n];
        let mut is_root_bin = vec![Scalar::zero(curve); n];
        let mut rel4_pickup = vec![Scalar::zero(curve); n];
        let mut root_left_eq_rel4 = vec![Scalar::zero(curve); n];
        let mut rel0_pickup = vec![Scalar::zero(curve); n];
        let mut pair4_left_eq_rel0 = vec![Scalar::zero(curve); n];
        let mut rel1_pickup = vec![Scalar::zero(curve); n];
        let mut pair4_right_eq_rel1 = vec![Scalar::zero(curve); n];
        let mut rel2_pickup = vec![Scalar::zero(curve); n];
        let mut pair5_left_eq_rel2 = vec![Scalar::zero(curve); n];
        let mut rel3_pickup = vec![Scalar::zero(curve); n];
        let mut pair5_right_eq_rel3 = vec![Scalar::zero(curve); n];
        let mut root_eq_claimed = vec![Scalar::zero(curve); n];

        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            is_real_bin[r] = v.mul(&v.sub(&one));
            let p1 = &columns[COL_IS_PAIR1][r];
            is_pair1_bin[r] = p1.mul(&p1.sub(&one));
            let p2 = &columns[COL_IS_PAIR2][r];
            is_pair2_bin[r] = p2.mul(&p2.sub(&one));
            let p0 = &columns[COL_IS_PAIR0][r];
            is_pair0_bin[r] = p0.mul(&p0.sub(&one));
            let p3 = &columns[COL_IS_PAIR3][r];
            is_pair3_bin[r] = p3.mul(&p3.sub(&one));
            let p4 = &columns[COL_IS_PAIR4][r];
            is_pair4_bin[r] = p4.mul(&p4.sub(&one));
            let p5 = &columns[COL_IS_PAIR5][r];
            is_pair5_bin[r] = p5.mul(&p5.sub(&one));
            let pr = &columns[COL_IS_ROOT][r];
            is_root_bin[r] = pr.mul(&pr.sub(&one));

            // Step 2b-offset2: row-local β-RLC bodies for relay pickup
            // (at row 4) and consumption (at row 6).
            let mut acc_pickup = Scalar::zero(curve);
            let mut acc_consume = Scalar::zero(curve);
            let mut bp_r = Scalar::one(curve);
            for k in 0..32 {
                let hash_k = &columns[COL_HASH_OFFSET + k][r];
                let rel_k = &columns[COL_REL4_HASH_OFFSET + k][r];
                let left_k = &columns[COL_LEFT_OFFSET + k][r];
                acc_pickup = acc_pickup.add(&bp_r.mul(&rel_k.sub(hash_k)));
                acc_consume = acc_consume.add(&bp_r.mul(&left_k.sub(rel_k)));
                bp_r = bp_r.mul(&beta_test);
            }
            rel4_pickup[r] = p4.mul(&acc_pickup);
            root_left_eq_rel4[r] = pr.mul(&acc_consume);

            // Step 2b-full: 4 more pickup+consumption pairs.
            let mut acc_pk0 = Scalar::zero(curve);
            let mut acc_pk1 = Scalar::zero(curve);
            let mut acc_pk2 = Scalar::zero(curve);
            let mut acc_pk3 = Scalar::zero(curve);
            let mut acc_cm_p4l = Scalar::zero(curve);
            let mut acc_cm_p4r = Scalar::zero(curve);
            let mut acc_cm_p5l = Scalar::zero(curve);
            let mut acc_cm_p5r = Scalar::zero(curve);
            let mut bp_x = Scalar::one(curve);
            for k in 0..32 {
                let hash_k = &columns[COL_HASH_OFFSET + k][r];
                let left_k = &columns[COL_LEFT_OFFSET + k][r];
                let right_k = &columns[COL_RIGHT_OFFSET + k][r];
                let rel0_k = &columns[COL_REL0_HASH_OFFSET + k][r];
                let rel1_k = &columns[COL_REL1_HASH_OFFSET + k][r];
                let rel2_k = &columns[COL_REL2_HASH_OFFSET + k][r];
                let rel3_k = &columns[COL_REL3_HASH_OFFSET + k][r];
                acc_pk0 = acc_pk0.add(&bp_x.mul(&rel0_k.sub(hash_k)));
                acc_pk1 = acc_pk1.add(&bp_x.mul(&rel1_k.sub(hash_k)));
                acc_pk2 = acc_pk2.add(&bp_x.mul(&rel2_k.sub(hash_k)));
                acc_pk3 = acc_pk3.add(&bp_x.mul(&rel3_k.sub(hash_k)));
                acc_cm_p4l = acc_cm_p4l.add(&bp_x.mul(&left_k.sub(rel0_k)));
                acc_cm_p4r = acc_cm_p4r.add(&bp_x.mul(&right_k.sub(rel1_k)));
                acc_cm_p5l = acc_cm_p5l.add(&bp_x.mul(&left_k.sub(rel2_k)));
                acc_cm_p5r = acc_cm_p5r.add(&bp_x.mul(&right_k.sub(rel3_k)));
                bp_x = bp_x.mul(&beta_test);
            }
            rel0_pickup[r] = p0.mul(&acc_pk0);
            pair4_left_eq_rel0[r] = p4.mul(&acc_cm_p4l);
            rel1_pickup[r] = p1.mul(&acc_pk1);
            pair4_right_eq_rel1[r] = p4.mul(&acc_cm_p4r);
            rel2_pickup[r] = p2.mul(&acc_pk2);
            pair5_left_eq_rel2[r] = p5.mul(&acc_cm_p5l);
            rel3_pickup[r] = p3.mul(&acc_pk3);
            pair5_right_eq_rel3[r] = p5.mul(&acc_cm_p5r);

            // Step 2e: at row 6 (is_root), HASH == CLAIMED_ROOT (β-RLC).
            let mut acc_root = Scalar::zero(curve);
            let mut bp_root = Scalar::one(curve);
            for k in 0..32 {
                let hash_k = &columns[COL_HASH_OFFSET + k][r];
                let claimed_k = &columns[COL_CLAIMED_ROOT_OFFSET + k][r];
                acc_root = acc_root.add(&bp_root.mul(&hash_k.sub(claimed_k)));
                bp_root = bp_root.mul(&beta_test);
            }
            root_eq_claimed[r] = pr.mul(&acc_root);

            // β-RLC sub-bodies for the equality constraints.
            let mut acc_pl = Scalar::zero(curve);    // pair1 left == parent_root
            let mut acc_pr = Scalar::zero(curve);    // pair1 right == state_root
            let mut acc_bl = Scalar::zero(curve);    // pair2 left == body_root
            let mut acc_brz = Scalar::zero(curve);   // pair2 right == 0
            let mut acc_p3l = Scalar::zero(curve);   // pair3 left == 0
            let mut acc_p3r = Scalar::zero(curve);   // pair3 right == 0
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let l = &columns[COL_LEFT_OFFSET + k][r];
                let rr = &columns[COL_RIGHT_OFFSET + k][r];
                let pr = &columns[COL_PARENT_ROOT_OFFSET + k][r];
                let sr = &columns[COL_STATE_ROOT_OFFSET + k][r];
                let br = &columns[COL_BODY_ROOT_OFFSET + k][r];
                acc_pl = acc_pl.add(&bp.mul(&l.sub(pr)));
                acc_pr = acc_pr.add(&bp.mul(&rr.sub(sr)));
                acc_bl = acc_bl.add(&bp.mul(&l.sub(br)));
                acc_brz = acc_brz.add(&bp.mul(rr));
                acc_p3l = acc_p3l.add(&bp.mul(l));
                acc_p3r = acc_p3r.add(&bp.mul(rr));
                bp = bp.mul(&beta_test);
            }
            pair1_left_eq[r] = p1.mul(&acc_pl);
            pair1_right_eq[r] = p1.mul(&acc_pr);
            pair2_left_eq[r] = p2.mul(&acc_bl);
            pair2_right_zero[r] = p2.mul(&acc_brz);
            pair3_l_eq_zero[r] = p3.mul(&acc_p3l);
            pair3_r_eq_zero[r] = p3.mul(&acc_p3r);

            // Pair0: htr_uint shape. LEFT[0..8] = slot_le, LEFT[8..32] = 0.
            // RIGHT[0..8] = proposer_index_le, RIGHT[8..32] = 0.
            let mut acc_l_lo = Scalar::zero(curve);
            let mut acc_l_hi = Scalar::zero(curve);
            let mut acc_r_lo = Scalar::zero(curve);
            let mut acc_r_hi = Scalar::zero(curve);
            let mut bp_lo = Scalar::one(curve);
            for k in 0..8 {
                let l = &columns[COL_LEFT_OFFSET + k][r];
                let rr = &columns[COL_RIGHT_OFFSET + k][r];
                let s = &columns[COL_SLOT_BYTE_OFFSET + k][r];
                let pi = &columns[COL_PROPOSER_INDEX_BYTE_OFFSET + k][r];
                acc_l_lo = acc_l_lo.add(&bp_lo.mul(&l.sub(s)));
                acc_r_lo = acc_r_lo.add(&bp_lo.mul(&rr.sub(pi)));
                bp_lo = bp_lo.mul(&beta_test);
            }
            let mut bp_hi = Scalar::one(curve);
            for k in 8..32 {
                let l = &columns[COL_LEFT_OFFSET + k][r];
                let rr = &columns[COL_RIGHT_OFFSET + k][r];
                acc_l_hi = acc_l_hi.add(&bp_hi.mul(l));
                acc_r_hi = acc_r_hi.add(&bp_hi.mul(rr));
                bp_hi = bp_hi.mul(&beta_test);
            }
            pair0_l_lo_eq_slot[r] = p0.mul(&acc_l_lo);
            pair0_l_hi_eq_zero[r] = p0.mul(&acc_l_hi);
            pair0_r_lo_eq_pi[r] = p0.mul(&acc_r_lo);
            pair0_r_hi_eq_zero[r] = p0.mul(&acc_r_hi);
        }
        vec![
            is_real_bin,
            is_pair1_bin,
            is_pair2_bin,
            pair1_left_eq,
            pair1_right_eq,
            pair2_left_eq,
            pair2_right_zero,
            is_pair0_bin,
            is_pair3_bin,
            pair0_l_lo_eq_slot,
            pair0_l_hi_eq_zero,
            pair0_r_lo_eq_pi,
            pair0_r_hi_eq_zero,
            pair3_l_eq_zero,
            pair3_r_eq_zero,
            is_pair4_bin,
            is_pair5_bin,
            is_root_bin,
            rel4_pickup,
            root_left_eq_rel4,
            rel0_pickup,
            pair4_left_eq_rel0,
            rel1_pickup,
            pair4_right_eq_rel1,
            rel2_pickup,
            pair5_left_eq_rel2,
            rel3_pickup,
            pair5_right_eq_rel3,
            root_eq_claimed,
        ]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        let p0 = &col_evals[COL_IS_PAIR0];
        let p1 = &col_evals[COL_IS_PAIR1];
        let p2 = &col_evals[COL_IS_PAIR2];
        let p3 = &col_evals[COL_IS_PAIR3];
        let p4 = &col_evals[COL_IS_PAIR4];
        let p5 = &col_evals[COL_IS_PAIR5];
        let proot = &col_evals[COL_IS_ROOT];
        let is_real_bin = v.mul(&v.sub(&one));
        let is_pair1_bin = p1.mul(&p1.sub(&one));
        let is_pair2_bin = p2.mul(&p2.sub(&one));
        let is_pair0_bin = p0.mul(&p0.sub(&one));
        let is_pair3_bin = p3.mul(&p3.sub(&one));
        let is_pair4_bin = p4.mul(&p4.sub(&one));
        let is_pair5_bin = p5.mul(&p5.sub(&one));
        let is_root_bin = proot.mul(&proot.sub(&one));

        let mut acc_pl = Scalar::zero(curve);
        let mut acc_pr = Scalar::zero(curve);
        let mut acc_bl = Scalar::zero(curve);
        let mut acc_brz = Scalar::zero(curve);
        let mut acc_p3l = Scalar::zero(curve);
        let mut acc_p3r = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let l = &col_evals[COL_LEFT_OFFSET + k];
            let rr = &col_evals[COL_RIGHT_OFFSET + k];
            let pr = &col_evals[COL_PARENT_ROOT_OFFSET + k];
            let sr = &col_evals[COL_STATE_ROOT_OFFSET + k];
            let br = &col_evals[COL_BODY_ROOT_OFFSET + k];
            acc_pl = acc_pl.add(&bp.mul(&l.sub(pr)));
            acc_pr = acc_pr.add(&bp.mul(&rr.sub(sr)));
            acc_bl = acc_bl.add(&bp.mul(&l.sub(br)));
            acc_brz = acc_brz.add(&bp.mul(rr));
            acc_p3l = acc_p3l.add(&bp.mul(l));
            acc_p3r = acc_p3r.add(&bp.mul(rr));
            bp = bp.mul(alpha);
        }
        let pair1_left_eq = p1.mul(&acc_pl);
        let pair1_right_eq = p1.mul(&acc_pr);
        let pair2_left_eq = p2.mul(&acc_bl);
        let pair2_right_zero = p2.mul(&acc_brz);
        let pair3_l_eq = p3.mul(&acc_p3l);
        let pair3_r_eq = p3.mul(&acc_p3r);

        // Pair0 htr_uint shape constraints.
        let mut acc_l_lo = Scalar::zero(curve);
        let mut acc_l_hi = Scalar::zero(curve);
        let mut acc_r_lo = Scalar::zero(curve);
        let mut acc_r_hi = Scalar::zero(curve);
        let mut bp_lo = Scalar::one(curve);
        for k in 0..8 {
            let l = &col_evals[COL_LEFT_OFFSET + k];
            let rr = &col_evals[COL_RIGHT_OFFSET + k];
            let s = &col_evals[COL_SLOT_BYTE_OFFSET + k];
            let pi = &col_evals[COL_PROPOSER_INDEX_BYTE_OFFSET + k];
            acc_l_lo = acc_l_lo.add(&bp_lo.mul(&l.sub(s)));
            acc_r_lo = acc_r_lo.add(&bp_lo.mul(&rr.sub(pi)));
            bp_lo = bp_lo.mul(alpha);
        }
        let mut bp_hi = Scalar::one(curve);
        for k in 8..32 {
            let l = &col_evals[COL_LEFT_OFFSET + k];
            let rr = &col_evals[COL_RIGHT_OFFSET + k];
            acc_l_hi = acc_l_hi.add(&bp_hi.mul(l));
            acc_r_hi = acc_r_hi.add(&bp_hi.mul(rr));
            bp_hi = bp_hi.mul(alpha);
        }
        let pair0_l_lo = p0.mul(&acc_l_lo);
        let pair0_l_hi = p0.mul(&acc_l_hi);
        let pair0_r_lo = p0.mul(&acc_r_lo);
        let pair0_r_hi = p0.mul(&acc_r_hi);

        let mut total = is_real_bin;
        let mut ap = alpha.clone();
        total = total.add(&ap.mul(&is_pair1_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_pair2_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair1_left_eq));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair1_right_eq));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair2_left_eq));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair2_right_zero));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_pair0_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_pair3_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair0_l_lo));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair0_l_hi));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair0_r_lo));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair0_r_hi));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair3_l_eq));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&pair3_r_eq));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_pair4_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_pair5_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_root_bin));

        // Step 2b-offset2 row-locals.
        let mut acc_pickup = Scalar::zero(curve);
        let mut acc_consume = Scalar::zero(curve);
        let mut bp_r = Scalar::one(curve);
        for k in 0..32 {
            let hash_k = &col_evals[COL_HASH_OFFSET + k];
            let rel_k = &col_evals[COL_REL4_HASH_OFFSET + k];
            let left_k = &col_evals[COL_LEFT_OFFSET + k];
            acc_pickup = acc_pickup.add(&bp_r.mul(&rel_k.sub(hash_k)));
            acc_consume = acc_consume.add(&bp_r.mul(&left_k.sub(rel_k)));
            bp_r = bp_r.mul(alpha);
        }
        let rel4_pickup = p4.mul(&acc_pickup);
        let root_left_eq_rel4 = proot.mul(&acc_consume);

        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&rel4_pickup));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&root_left_eq_rel4));

        // Step 2b-full row-locals.
        let mut acc_pk0 = Scalar::zero(curve);
        let mut acc_pk1 = Scalar::zero(curve);
        let mut acc_pk2 = Scalar::zero(curve);
        let mut acc_pk3 = Scalar::zero(curve);
        let mut acc_cm_p4l = Scalar::zero(curve);
        let mut acc_cm_p4r = Scalar::zero(curve);
        let mut acc_cm_p5l = Scalar::zero(curve);
        let mut acc_cm_p5r = Scalar::zero(curve);
        let mut bp_x = Scalar::one(curve);
        for k in 0..32 {
            let hash_k = &col_evals[COL_HASH_OFFSET + k];
            let left_k = &col_evals[COL_LEFT_OFFSET + k];
            let right_k = &col_evals[COL_RIGHT_OFFSET + k];
            let rel0_k = &col_evals[COL_REL0_HASH_OFFSET + k];
            let rel1_k = &col_evals[COL_REL1_HASH_OFFSET + k];
            let rel2_k = &col_evals[COL_REL2_HASH_OFFSET + k];
            let rel3_k = &col_evals[COL_REL3_HASH_OFFSET + k];
            acc_pk0 = acc_pk0.add(&bp_x.mul(&rel0_k.sub(hash_k)));
            acc_pk1 = acc_pk1.add(&bp_x.mul(&rel1_k.sub(hash_k)));
            acc_pk2 = acc_pk2.add(&bp_x.mul(&rel2_k.sub(hash_k)));
            acc_pk3 = acc_pk3.add(&bp_x.mul(&rel3_k.sub(hash_k)));
            acc_cm_p4l = acc_cm_p4l.add(&bp_x.mul(&left_k.sub(rel0_k)));
            acc_cm_p4r = acc_cm_p4r.add(&bp_x.mul(&right_k.sub(rel1_k)));
            acc_cm_p5l = acc_cm_p5l.add(&bp_x.mul(&left_k.sub(rel2_k)));
            acc_cm_p5r = acc_cm_p5r.add(&bp_x.mul(&right_k.sub(rel3_k)));
            bp_x = bp_x.mul(alpha);
        }
        let rel0_pickup = p0.mul(&acc_pk0);
        let p4_left_rel0 = p4.mul(&acc_cm_p4l);
        let rel1_pickup = p1.mul(&acc_pk1);
        let p4_right_rel1 = p4.mul(&acc_cm_p4r);
        let rel2_pickup = p2.mul(&acc_pk2);
        let p5_left_rel2 = p5.mul(&acc_cm_p5l);
        let rel3_pickup = p3.mul(&acc_pk3);
        let p5_right_rel3 = p5.mul(&acc_cm_p5r);

        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&rel0_pickup));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&p4_left_rel0));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&rel1_pickup));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&p4_right_rel1));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&rel2_pickup));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&p5_left_rel2));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&rel3_pickup));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&p5_right_rel3));

        // Step 2e: at row 6, HASH == CLAIMED_ROOT (β-RLC).
        let mut acc_root = Scalar::zero(curve);
        let mut bp_root = Scalar::one(curve);
        for k in 0..32 {
            let hash_k = &col_evals[COL_HASH_OFFSET + k];
            let claimed_k = &col_evals[COL_CLAIMED_ROOT_OFFSET + k];
            acc_root = acc_root.add(&bp_root.mul(&hash_k.sub(claimed_k)));
            bp_root = bp_root.mul(alpha);
        }
        let root_eq_claimed = proot.mul(&acc_root);
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&root_eq_claimed));
        total
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
        let is_real_bin = poly_mul(v, &v_m1, curve);

        let p1 = &col_coeffs[COL_IS_PAIR1];
        let p1_m1 = poly_sub(p1, &one_poly, curve);
        let is_pair1_bin = poly_mul(p1, &p1_m1, curve);

        let p2 = &col_coeffs[COL_IS_PAIR2];
        let p2_m1 = poly_sub(p2, &one_poly, curve);
        let is_pair2_bin = poly_mul(p2, &p2_m1, curve);

        let p0 = &col_coeffs[COL_IS_PAIR0];
        let p0_m1 = poly_sub(p0, &one_poly, curve);
        let is_pair0_bin = poly_mul(p0, &p0_m1, curve);

        let p3 = &col_coeffs[COL_IS_PAIR3];
        let p3_m1 = poly_sub(p3, &one_poly, curve);
        let is_pair3_bin = poly_mul(p3, &p3_m1, curve);

        let p4 = &col_coeffs[COL_IS_PAIR4];
        let p4_m1 = poly_sub(p4, &one_poly, curve);
        let is_pair4_bin = poly_mul(p4, &p4_m1, curve);

        let p5 = &col_coeffs[COL_IS_PAIR5];
        let p5_m1 = poly_sub(p5, &one_poly, curve);
        let is_pair5_bin = poly_mul(p5, &p5_m1, curve);

        let proot = &col_coeffs[COL_IS_ROOT];
        let proot_m1 = poly_sub(proot, &one_poly, curve);
        let is_root_bin = poly_mul(proot, &proot_m1, curve);

        let mut acc_pl = vec![Scalar::zero(curve)];
        let mut acc_pr = vec![Scalar::zero(curve)];
        let mut acc_bl = vec![Scalar::zero(curve)];
        let mut acc_brz = vec![Scalar::zero(curve)];
        let mut acc_p3l = vec![Scalar::zero(curve)];
        let mut acc_p3r = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let l = &col_coeffs[COL_LEFT_OFFSET + k];
            let rr = &col_coeffs[COL_RIGHT_OFFSET + k];
            let pr = &col_coeffs[COL_PARENT_ROOT_OFFSET + k];
            let sr = &col_coeffs[COL_STATE_ROOT_OFFSET + k];
            let br = &col_coeffs[COL_BODY_ROOT_OFFSET + k];
            let dpl = poly_sub(l, pr, curve);
            let dpr = poly_sub(rr, sr, curve);
            let dbl = poly_sub(l, br, curve);
            acc_pl = poly_add(&acc_pl, &poly_scalar_mul(&dpl, &bp), curve);
            acc_pr = poly_add(&acc_pr, &poly_scalar_mul(&dpr, &bp), curve);
            acc_bl = poly_add(&acc_bl, &poly_scalar_mul(&dbl, &bp), curve);
            acc_brz = poly_add(&acc_brz, &poly_scalar_mul(rr, &bp), curve);
            acc_p3l = poly_add(&acc_p3l, &poly_scalar_mul(l, &bp), curve);
            acc_p3r = poly_add(&acc_p3r, &poly_scalar_mul(rr, &bp), curve);
            bp = bp.mul(alpha);
        }
        let pair1_left_eq = poly_mul(p1, &acc_pl, curve);
        let pair1_right_eq = poly_mul(p1, &acc_pr, curve);
        let pair2_left_eq = poly_mul(p2, &acc_bl, curve);
        let pair2_right_zero = poly_mul(p2, &acc_brz, curve);
        let pair3_l_eq = poly_mul(p3, &acc_p3l, curve);
        let pair3_r_eq = poly_mul(p3, &acc_p3r, curve);

        // Pair0 htr_uint shape.
        let mut acc_l_lo = vec![Scalar::zero(curve)];
        let mut acc_l_hi = vec![Scalar::zero(curve)];
        let mut acc_r_lo = vec![Scalar::zero(curve)];
        let mut acc_r_hi = vec![Scalar::zero(curve)];
        let mut bp_lo = Scalar::one(curve);
        for k in 0..8 {
            let l = &col_coeffs[COL_LEFT_OFFSET + k];
            let rr = &col_coeffs[COL_RIGHT_OFFSET + k];
            let s = &col_coeffs[COL_SLOT_BYTE_OFFSET + k];
            let pi = &col_coeffs[COL_PROPOSER_INDEX_BYTE_OFFSET + k];
            let dl = poly_sub(l, s, curve);
            let dr = poly_sub(rr, pi, curve);
            acc_l_lo = poly_add(&acc_l_lo, &poly_scalar_mul(&dl, &bp_lo), curve);
            acc_r_lo = poly_add(&acc_r_lo, &poly_scalar_mul(&dr, &bp_lo), curve);
            bp_lo = bp_lo.mul(alpha);
        }
        let mut bp_hi = Scalar::one(curve);
        for k in 8..32 {
            let l = &col_coeffs[COL_LEFT_OFFSET + k];
            let rr = &col_coeffs[COL_RIGHT_OFFSET + k];
            acc_l_hi = poly_add(&acc_l_hi, &poly_scalar_mul(l, &bp_hi), curve);
            acc_r_hi = poly_add(&acc_r_hi, &poly_scalar_mul(rr, &bp_hi), curve);
            bp_hi = bp_hi.mul(alpha);
        }
        let pair0_l_lo = poly_mul(p0, &acc_l_lo, curve);
        let pair0_l_hi = poly_mul(p0, &acc_l_hi, curve);
        let pair0_r_lo = poly_mul(p0, &acc_r_lo, curve);
        let pair0_r_hi = poly_mul(p0, &acc_r_hi, curve);

        let mut total = is_real_bin;
        let mut ap = alpha.clone();
        total = poly_add(&total, &poly_scalar_mul(&is_pair1_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_pair2_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair1_left_eq, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair1_right_eq, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair2_left_eq, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair2_right_zero, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_pair0_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_pair3_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair0_l_lo, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair0_l_hi, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair0_r_lo, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair0_r_hi, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair3_l_eq, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&pair3_r_eq, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_pair4_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_pair5_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_root_bin, &ap), curve);

        // Step 2b-offset2 row-locals in coefficient form.
        let mut acc_pickup = vec![Scalar::zero(curve)];
        let mut acc_consume = vec![Scalar::zero(curve)];
        let mut bp_r = Scalar::one(curve);
        for k in 0..32 {
            let hash_k = &col_coeffs[COL_HASH_OFFSET + k];
            let rel_k = &col_coeffs[COL_REL4_HASH_OFFSET + k];
            let left_k = &col_coeffs[COL_LEFT_OFFSET + k];
            let d_pickup = poly_sub(rel_k, hash_k, curve);
            let d_consume = poly_sub(left_k, rel_k, curve);
            acc_pickup = poly_add(&acc_pickup, &poly_scalar_mul(&d_pickup, &bp_r), curve);
            acc_consume = poly_add(&acc_consume, &poly_scalar_mul(&d_consume, &bp_r), curve);
            bp_r = bp_r.mul(alpha);
        }
        let rel4_pickup = poly_mul(p4, &acc_pickup, curve);
        let root_left_eq_rel4 = poly_mul(proot, &acc_consume, curve);

        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&rel4_pickup, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&root_left_eq_rel4, &ap), curve);

        // Step 2b-full row-locals in coefficient form.
        let mut acc_pk0 = vec![Scalar::zero(curve)];
        let mut acc_pk1 = vec![Scalar::zero(curve)];
        let mut acc_pk2 = vec![Scalar::zero(curve)];
        let mut acc_pk3 = vec![Scalar::zero(curve)];
        let mut acc_cm_p4l = vec![Scalar::zero(curve)];
        let mut acc_cm_p4r = vec![Scalar::zero(curve)];
        let mut acc_cm_p5l = vec![Scalar::zero(curve)];
        let mut acc_cm_p5r = vec![Scalar::zero(curve)];
        let mut bp_x = Scalar::one(curve);
        for k in 0..32 {
            let hash_k = &col_coeffs[COL_HASH_OFFSET + k];
            let left_k = &col_coeffs[COL_LEFT_OFFSET + k];
            let right_k = &col_coeffs[COL_RIGHT_OFFSET + k];
            let rel0_k = &col_coeffs[COL_REL0_HASH_OFFSET + k];
            let rel1_k = &col_coeffs[COL_REL1_HASH_OFFSET + k];
            let rel2_k = &col_coeffs[COL_REL2_HASH_OFFSET + k];
            let rel3_k = &col_coeffs[COL_REL3_HASH_OFFSET + k];
            let d_pk0 = poly_sub(rel0_k, hash_k, curve);
            let d_pk1 = poly_sub(rel1_k, hash_k, curve);
            let d_pk2 = poly_sub(rel2_k, hash_k, curve);
            let d_pk3 = poly_sub(rel3_k, hash_k, curve);
            let d_p4l = poly_sub(left_k, rel0_k, curve);
            let d_p4r = poly_sub(right_k, rel1_k, curve);
            let d_p5l = poly_sub(left_k, rel2_k, curve);
            let d_p5r = poly_sub(right_k, rel3_k, curve);
            acc_pk0 = poly_add(&acc_pk0, &poly_scalar_mul(&d_pk0, &bp_x), curve);
            acc_pk1 = poly_add(&acc_pk1, &poly_scalar_mul(&d_pk1, &bp_x), curve);
            acc_pk2 = poly_add(&acc_pk2, &poly_scalar_mul(&d_pk2, &bp_x), curve);
            acc_pk3 = poly_add(&acc_pk3, &poly_scalar_mul(&d_pk3, &bp_x), curve);
            acc_cm_p4l = poly_add(&acc_cm_p4l, &poly_scalar_mul(&d_p4l, &bp_x), curve);
            acc_cm_p4r = poly_add(&acc_cm_p4r, &poly_scalar_mul(&d_p4r, &bp_x), curve);
            acc_cm_p5l = poly_add(&acc_cm_p5l, &poly_scalar_mul(&d_p5l, &bp_x), curve);
            acc_cm_p5r = poly_add(&acc_cm_p5r, &poly_scalar_mul(&d_p5r, &bp_x), curve);
            bp_x = bp_x.mul(alpha);
        }
        let rel0_pickup = poly_mul(p0, &acc_pk0, curve);
        let p4_left_rel0 = poly_mul(p4, &acc_cm_p4l, curve);
        let rel1_pickup = poly_mul(p1, &acc_pk1, curve);
        let p4_right_rel1 = poly_mul(p4, &acc_cm_p4r, curve);
        let rel2_pickup = poly_mul(p2, &acc_pk2, curve);
        let p5_left_rel2 = poly_mul(p5, &acc_cm_p5l, curve);
        let rel3_pickup = poly_mul(p3, &acc_pk3, curve);
        let p5_right_rel3 = poly_mul(p5, &acc_cm_p5r, curve);

        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&rel0_pickup, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&p4_left_rel0, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&rel1_pickup, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&p4_right_rel1, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&rel2_pickup, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&p5_left_rel2, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&rel3_pickup, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&p5_right_rel3, &ap), curve);

        // Step 2e: root binding in coefficient form.
        let mut acc_root = vec![Scalar::zero(curve)];
        let mut bp_root = Scalar::one(curve);
        for k in 0..32 {
            let hash_k = &col_coeffs[COL_HASH_OFFSET + k];
            let claimed_k = &col_coeffs[COL_CLAIMED_ROOT_OFFSET + k];
            let diff = poly_sub(hash_k, claimed_k, curve);
            acc_root = poly_add(&acc_root, &poly_scalar_mul(&diff, &bp_root), curve);
            bp_root = bp_root.mul(alpha);
        }
        let root_eq_claimed = poly_mul(proot, &acc_root, curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&root_eq_claimed, &ap), curve);
        total
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // ω·z evaluations layout (total 306):
        //   [0]         IS_REAL_NEXT (gating)
        //   [1..33]     PARENT_ROOT
        //   [33..65]    STATE_ROOT
        //   [65..97]    BODY_ROOT
        //   [97..105]   SLOT
        //   [105..113]  PROPOSER_INDEX
        //   [113]       IS_ROOT_NEXT (chain gating)
        //   [114..146]  RIGHT_NEXT (chain target)
        //   [146..178]  REL4_HASH_NEXT (relay constancy target)
        //   [178..210]  REL0_HASH_NEXT
        //   [210..242]  REL1_HASH_NEXT
        //   [242..274]  REL2_HASH_NEXT
        //   [274..306]  REL3_HASH_NEXT
        //   [306..338]  CLAIMED_ROOT_NEXT
        let mut cols = Vec::with_capacity(338);
        cols.push(COL_IS_REAL);
        for k in 0..32 { cols.push(COL_PARENT_ROOT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_STATE_ROOT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_BODY_ROOT_OFFSET + k); }
        for k in 0..8 { cols.push(COL_SLOT_BYTE_OFFSET + k); }
        for k in 0..8 { cols.push(COL_PROPOSER_INDEX_BYTE_OFFSET + k); }
        cols.push(COL_IS_ROOT);
        for k in 0..32 { cols.push(COL_RIGHT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL4_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL0_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL1_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL2_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL3_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_CLAIMED_ROOT_OFFSET + k); }
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
        if shifted_evals.len() != 338 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let gating = v.mul(v_next);

        // 5 cross-row constancy bodies, one per claimed-field column block.
        let make_constancy_body = |off_z: usize, shifted_base: usize, n: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..n {
                let cur = &col_evals_at_z[off_z + k];
                let nxt = &shifted_evals[shifted_base + k];
                acc = acc.add(&bp.mul(&nxt.sub(cur)));
                bp = bp.mul(alpha);
            }
            acc
        };
        let body_pr = gating.mul(&make_constancy_body(COL_PARENT_ROOT_OFFSET, 1, 32));
        let body_sr = gating.mul(&make_constancy_body(COL_STATE_ROOT_OFFSET, 33, 32));
        let body_br = gating.mul(&make_constancy_body(COL_BODY_ROOT_OFFSET, 65, 32));
        let body_slot = gating.mul(&make_constancy_body(COL_SLOT_BYTE_OFFSET, 97, 8));
        let body_pi = gating.mul(&make_constancy_body(COL_PROPOSER_INDEX_BYTE_OFFSET, 105, 8));

        // 6th body: layer chain at row 5 → row 6.
        // IS_PAIR5(r) * IS_ROOT(r+1) * Σ β^k * (RIGHT_NEXT[k] - HASH[k]) = 0
        // shifted_evals layout: [IS_REAL_NEXT (1), PARENT_ROOT (32),
        //   STATE_ROOT (32), BODY_ROOT (32), SLOT (8), PROPOSER_INDEX (8),
        //   IS_ROOT_NEXT (1), RIGHT_NEXT (32)] = total 146.
        let p5 = &col_evals_at_z[COL_IS_PAIR5];
        let is_root_next = &shifted_evals[113];
        let chain_gating = p5.mul(is_root_next);
        let mut chain_body = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let hash_curr = &col_evals_at_z[COL_HASH_OFFSET + k];
            let right_next = &shifted_evals[114 + k];
            chain_body = chain_body.add(&bp.mul(&right_next.sub(hash_curr)));
            bp = bp.mul(alpha);
        }
        let body_chain = chain_gating.mul(&chain_body);

        // Relay constancy bodies. Each: gating * Σ β^k * (REL_NEXT - REL).
        let p0 = &col_evals_at_z[COL_IS_PAIR0];
        let p1 = &col_evals_at_z[COL_IS_PAIR1];
        let p2 = &col_evals_at_z[COL_IS_PAIR2];
        let p3 = &col_evals_at_z[COL_IS_PAIR3];
        let p4 = &col_evals_at_z[COL_IS_PAIR4];
        let p5_gate = &col_evals_at_z[COL_IS_PAIR5];

        let make_relay = |rel_off: usize, shifted_base: usize, gating: Scalar| -> Scalar {
            let mut body = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let rel_curr = &col_evals_at_z[rel_off + k];
                let rel_next = &shifted_evals[shifted_base + k];
                body = body.add(&bp.mul(&rel_next.sub(rel_curr)));
                bp = bp.mul(alpha);
            }
            gating.mul(&body)
        };

        // Body 6: REL4_HASH constancy. Gating: IS_PAIR4 + IS_PAIR5.
        let body_relay = make_relay(COL_REL4_HASH_OFFSET, 146, p4.add(p5_gate));
        // Body 7: REL0_HASH constancy. Gating: IS_PAIR0 + IS_PAIR1 + IS_PAIR2 + IS_PAIR3.
        let body_rel0 = make_relay(COL_REL0_HASH_OFFSET, 178, p0.add(p1).add(p2).add(p3));
        // Body 8: REL1_HASH constancy. Gating: IS_PAIR1 + IS_PAIR2 + IS_PAIR3.
        let body_rel1 = make_relay(COL_REL1_HASH_OFFSET, 210, p1.add(p2).add(p3));
        // Body 9: REL2_HASH constancy. Gating: IS_PAIR2 + IS_PAIR3 + IS_PAIR4.
        let body_rel2 = make_relay(COL_REL2_HASH_OFFSET, 242, p2.add(p3).add(p4));
        // Body 10: REL3_HASH constancy. Gating: IS_PAIR3 + IS_PAIR4.
        let body_rel3 = make_relay(COL_REL3_HASH_OFFSET, 274, p3.add(p4));
        // Body 11: CLAIMED_ROOT cross-row constancy. Gating: IS_REAL[r]*IS_REAL[r+1].
        let body_cr = make_constancy_body(COL_CLAIMED_ROOT_OFFSET, 306, 32);
        let body_claimed_root = gating.mul(&body_cr);

        // α^(alpha_offset+i) · body_i
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = ap.mul(&body_pr);
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_sr));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_br));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_slot));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_pi));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_chain));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_relay));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_rel0));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_rel1));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_rel2));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_rel3));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_claimed_root));

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
        let v = &column_coeffs[COL_IS_REAL];
        let v_next = poly_shift(v, omega);
        let gating = poly_mul(v, &v_next, curve);

        let make_body = |off: usize, n: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            let mut bp = Scalar::one(curve);
            for k in 0..n {
                let cur = &column_coeffs[off + k];
                let nxt = poly_shift(cur, omega);
                let diff = poly_sub(&nxt, cur, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
                bp = bp.mul(alpha);
            }
            acc
        };
        let body_pr = poly_mul(&gating, &make_body(COL_PARENT_ROOT_OFFSET, 32), curve);
        let body_sr = poly_mul(&gating, &make_body(COL_STATE_ROOT_OFFSET, 32), curve);
        let body_br = poly_mul(&gating, &make_body(COL_BODY_ROOT_OFFSET, 32), curve);
        let body_slot = poly_mul(&gating, &make_body(COL_SLOT_BYTE_OFFSET, 8), curve);
        let body_pi = poly_mul(&gating, &make_body(COL_PROPOSER_INDEX_BYTE_OFFSET, 8), curve);

        // 6th body: layer chain. IS_PAIR5(X) * IS_ROOT(ω·X) * Σ β^k * (RIGHT(ω·X) - HASH(X))
        let p5 = &column_coeffs[COL_IS_PAIR5];
        let is_root = &column_coeffs[COL_IS_ROOT];
        let is_root_next = poly_shift(is_root, omega);
        let chain_gating = poly_mul(p5, &is_root_next, curve);
        let mut chain_body = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let hash_curr = &column_coeffs[COL_HASH_OFFSET + k];
            let right_curr = &column_coeffs[COL_RIGHT_OFFSET + k];
            let right_next = poly_shift(right_curr, omega);
            let diff = poly_sub(&right_next, hash_curr, curve);
            chain_body = poly_add(&chain_body, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body_chain = poly_mul(&chain_gating, &chain_body, curve);

        // Relay constancy bodies in coefficient form.
        let p0_poly = &column_coeffs[COL_IS_PAIR0];
        let p1_poly = &column_coeffs[COL_IS_PAIR1];
        let p2_poly = &column_coeffs[COL_IS_PAIR2];
        let p3_poly = &column_coeffs[COL_IS_PAIR3];
        let p4_poly = &column_coeffs[COL_IS_PAIR4];
        let p5_poly = &column_coeffs[COL_IS_PAIR5];

        let make_relay_poly = |rel_off: usize, gating: Vec<Scalar>| -> Vec<Scalar> {
            let mut body = vec![Scalar::zero(curve)];
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let rel_curr = &column_coeffs[rel_off + k];
                let rel_next = poly_shift(rel_curr, omega);
                let diff = poly_sub(&rel_next, rel_curr, curve);
                body = poly_add(&body, &poly_scalar_mul(&diff, &bp), curve);
                bp = bp.mul(alpha);
            }
            poly_mul(&gating, &body, curve)
        };

        let body_relay = make_relay_poly(COL_REL4_HASH_OFFSET, poly_add(p4_poly, p5_poly, curve));
        let body_rel0 = make_relay_poly(
            COL_REL0_HASH_OFFSET,
            poly_add(&poly_add(p0_poly, p1_poly, curve), &poly_add(p2_poly, p3_poly, curve), curve),
        );
        let body_rel1 = make_relay_poly(
            COL_REL1_HASH_OFFSET,
            poly_add(&poly_add(p1_poly, p2_poly, curve), p3_poly, curve),
        );
        let body_rel2 = make_relay_poly(
            COL_REL2_HASH_OFFSET,
            poly_add(&poly_add(p2_poly, p3_poly, curve), p4_poly, curve),
        );
        let body_rel3 = make_relay_poly(COL_REL3_HASH_OFFSET, poly_add(p3_poly, p4_poly, curve));

        // CLAIMED_ROOT cross-row constancy in coefficient form.
        let body_cr = make_body(COL_CLAIMED_ROOT_OFFSET, 32);
        let body_claimed_root = poly_mul(&gating, &body_cr, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = poly_scalar_mul(&body_pr, &ap);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_sr, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_br, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_slot, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_pi, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_chain, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_relay, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_rel0, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_rel1, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_rel2, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_rel3, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_claimed_root, &ap), curve);

        // Multiply by (X - ω^{n-1}) to exclude wrap-around row.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![
            COL_IS_REAL,
            COL_IS_PAIR0, COL_IS_PAIR1, COL_IS_PAIR2, COL_IS_PAIR3,
            COL_IS_PAIR4, COL_IS_PAIR5, COL_IS_ROOT,
        ]
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
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
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
        for k in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("bbh_pair_left_{}_8bit", k),
                    column_index: COL_LEFT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("bbh_pair_right_{}_8bit", k),
                    column_index: COL_RIGHT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("bbh_pair_hash_{}_8bit", k),
                    column_index: COL_HASH_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("bbh_pair_parent_root_{}_8bit", k),
                    column_index: COL_PARENT_ROOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("bbh_pair_state_root_{}_8bit", k),
                    column_index: COL_STATE_ROOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("bbh_pair_body_root_{}_8bit", k),
                    column_index: COL_BODY_ROOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..8 {
            declarations.push((
                LookupDeclaration {
                    label: format!("bbh_pair_slot_{}_8bit", k),
                    column_index: COL_SLOT_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("bbh_pair_proposer_index_{}_8bit", k),
                    column_index: COL_PROPOSER_INDEX_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..32 {
            for (name, off) in [
                ("rel4_hash", COL_REL4_HASH_OFFSET),
                ("rel0_hash", COL_REL0_HASH_OFFSET),
                ("rel1_hash", COL_REL1_HASH_OFFSET),
                ("rel2_hash", COL_REL2_HASH_OFFSET),
                ("rel3_hash", COL_REL3_HASH_OFFSET),
                ("claimed_root", COL_CLAIMED_ROOT_OFFSET),
            ] {
                declarations.push((
                    LookupDeclaration {
                        label: format!("bbh_pair_{}_{}_8bit", name, k),
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

// ─── Cross-AIR LogUp linkage ──────────────────────────────────────────

/// Bind each `(left || right, hash)` row of this AIR to a real
/// [`crate::sha256_extract`] row.
///
/// **A side (this AIR)**: 96 bytes — `LEFT[32] || RIGHT[32] || HASH[32]`,
/// gated by `COL_IS_REAL`.
///
/// **B side (Sha256Extract)**: 64 input bytes + 32 output bytes, gated
/// by `Sha256Extract::COL_IS_REAL`.
///
/// Soundness: closes only the structural binding to a real SHA-256
/// invocation row. Layer chaining and leaf binding are deferred to step
/// 2 (cross-row equality + leaf-column constraints).
pub fn make_beacon_block_header_pair_to_sha256_extract_linkage_descriptor(
    pair_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..32 { a_columns.push(COL_LEFT_OFFSET + b); }
    for b in 0..32 { a_columns.push(COL_RIGHT_OFFSET + b); }
    for b in 0..32 { a_columns.push(COL_HASH_OFFSET + b); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..crate::sha256_extract::NUM_INPUT_BYTES {
        b_columns.push(crate::sha256_extract::COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..crate::sha256_extract::NUM_OUTPUT_BYTES {
        b_columns.push(crate::sha256_extract::COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "beacon_block_header_pair_sha256_extract_v1".into(),
        a_layer_index: pair_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(crate::sha256_extract::COL_IS_REAL),
    }
}

/// Build a tuple selector for downstream AIRs that want to consume the
/// computed BBH root + the 5 input fields. The descriptor exposes a
/// 112-byte tuple `(CLAIMED_ROOT[32] || PARENT_ROOT[32] || STATE_ROOT[32]
/// || BODY_ROOT[32] || SLOT[8] || PROPOSER_INDEX[8])` gated by
/// `IS_REAL`. The B-side AIR (downstream consumer) must align its 112
/// columns to the same layout.
///
/// Soundness: under cross-row constancy of all 6 column groups (which
/// this AIR enforces), the LogUp multiset reduces to one canonical
/// `(root, fields)` tuple per real row, all equal. Downstream consumers
/// see a stable, algebraically-bound `(root, fields)` pair.
///
/// Use case: a finality AIR consuming BBH HTR roots; or a sync
/// committee binding that wants the `(slot, root)` checkpoint.
pub fn make_bbh_pair_root_output_column_indices() -> Vec<usize> {
    let mut cols = Vec::with_capacity(32 + 32 + 32 + 32 + 8 + 8);
    for k in 0..32 { cols.push(COL_CLAIMED_ROOT_OFFSET + k); }
    for k in 0..32 { cols.push(COL_PARENT_ROOT_OFFSET + k); }
    for k in 0..32 { cols.push(COL_STATE_ROOT_OFFSET + k); }
    for k in 0..32 { cols.push(COL_BODY_ROOT_OFFSET + k); }
    for k in 0..8 { cols.push(COL_SLOT_BYTE_OFFSET + k); }
    for k in 0..8 { cols.push(COL_PROPOSER_INDEX_BYTE_OFFSET + k); }
    cols
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon::BeaconBlockHeader;

    fn sample_header() -> BeaconBlockHeader {
        BeaconBlockHeader {
            slot: 7777,
            proposer_index: 42,
            parent_root: [0xaau8; 32],
            state_root: [0xbbu8; 32],
            body_root: [0xccu8; 32],
        }
    }

    #[test]
    fn trace_builder_populates_seven_real_rows() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // 7 real rows, padded to nearest_power_of_two minimum 16.
        assert_eq!(trace.num_rows, 7);
        assert_eq!(trace.padded_size, 16);
        for r in 0..7 {
            assert_eq!(trace.columns[COL_IS_REAL].evaluations[r].to_u64(), 1);
        }
        for r in 7..16 {
            assert!(trace.columns[COL_IS_REAL].evaluations[r].is_zero());
        }
    }

    #[test]
    fn pair_rows_match_witness_invocations() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for (r, inv) in w.invocations.iter().enumerate() {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_LEFT_OFFSET + k].evaluations[r].to_u64(),
                    inv.left[k] as u64,
                );
                assert_eq!(
                    trace.columns[COL_RIGHT_OFFSET + k].evaluations[r].to_u64(),
                    inv.right[k] as u64,
                );
                assert_eq!(
                    trace.columns[COL_HASH_OFFSET + k].evaluations[r].to_u64(),
                    inv.hash[k] as u64,
                );
            }
        }
    }

    #[test]
    fn is_real_binary_constraint_zero_on_honest_witness() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for v in &res[0] {
            assert!(v.is_zero(), "is_real_binary should be zero");
        }
    }

    #[test]
    fn leaf_binding_constraints_zero_on_honest_witness() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} should be zero on honest witness",
                    i, r,
                );
            }
        }
    }

    #[test]
    fn pair1_left_eq_fires_on_tampered_parent_root() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: corrupt PARENT_ROOT byte 0 — must break pair1_left_eq at row 1.
        cols[COL_PARENT_ROOT_OFFSET][1] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 3 = pair1_left_eq_parent_root.
        assert!(!res[3][1].is_zero(), "pair1_left_eq must fire at row 1");
    }

    #[test]
    fn pair2_right_zero_fires_on_nonzero_right_at_row2() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Honest row 2 has RIGHT = ZERO_CHUNK; tamper one byte to break the
        // ZERO_CHUNK pin.
        cols[COL_RIGHT_OFFSET + 5][2] = Scalar::from_u64(0x42, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 6 = pair2_right_eq_zero_chunk.
        assert!(!res[6][2].is_zero(), "pair2_right_zero must fire on tamper");
    }

    #[test]
    fn consumer_row_selectors_populated_in_witness() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // is_pair4 = 1 only on row 4
        for r in 0..16 {
            let expected = if r == 4 { 1 } else { 0 };
            assert_eq!(
                trace.columns[COL_IS_PAIR4].evaluations[r].to_u64(), expected,
                "is_pair4 at row {}", r,
            );
        }
        // is_pair5 = 1 only on row 5
        for r in 0..16 {
            let expected = if r == 5 { 1 } else { 0 };
            assert_eq!(
                trace.columns[COL_IS_PAIR5].evaluations[r].to_u64(), expected,
                "is_pair5 at row {}", r,
            );
        }
        // is_root = 1 only on row 6
        for r in 0..16 {
            let expected = if r == 6 { 1 } else { 0 };
            assert_eq!(
                trace.columns[COL_IS_ROOT].evaluations[r].to_u64(), expected,
                "is_root at row {}", r,
            );
        }
    }

    #[test]
    fn claimed_root_populated_and_constant() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for r in 0..trace.num_rows {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_CLAIMED_ROOT_OFFSET + k].evaluations[r].to_u64(),
                    w.root[k] as u64,
                    "CLAIMED_ROOT[r={},k={}]", r, k,
                );
            }
        }
        // Also: HASH[6] (the actual computed root invocation output)
        // must equal CLAIMED_ROOT[6].
        for k in 0..32 {
            assert_eq!(
                trace.columns[COL_HASH_OFFSET + k].evaluations[6].to_u64(),
                trace.columns[COL_CLAIMED_ROOT_OFFSET + k].evaluations[6].to_u64(),
                "HASH[6] vs CLAIMED_ROOT[6] at byte {}", k,
            );
        }
    }

    #[test]
    fn root_eq_claimed_fires_on_tampered_claimed_root() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper CLAIMED_ROOT byte 0 at row 6 — must break root binding.
        cols[COL_CLAIMED_ROOT_OFFSET][6] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 28 = root_hash_eq_claimed_root.
        assert!(!res[28][6].is_zero(), "root_eq_claimed must fire");
    }

    #[test]
    fn root_output_column_indices_shape() {
        let cols = make_bbh_pair_root_output_column_indices();
        assert_eq!(cols.len(), 32 + 32 + 32 + 32 + 8 + 8); // 144
        // First 32 = CLAIMED_ROOT
        assert_eq!(cols[0], COL_CLAIMED_ROOT_OFFSET);
        assert_eq!(cols[31], COL_CLAIMED_ROOT_OFFSET + 31);
        // Next 32 = PARENT_ROOT
        assert_eq!(cols[32], COL_PARENT_ROOT_OFFSET);
        // Last 8 = PROPOSER_INDEX
        assert_eq!(*cols.last().unwrap(), COL_PROPOSER_INDEX_BYTE_OFFSET + 7);
    }

    #[test]
    fn all_relay_columns_propagate_correctly() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // REL0: rows 0..4 = HASH[0]
        let h0 = w.invocations[0].hash;
        for r in 0..=4 {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_REL0_HASH_OFFSET + k].evaluations[r].to_u64(),
                    h0[k] as u64,
                    "REL0[r={},k={}]", r, k,
                );
            }
        }
        // REL1: rows 1..4 = HASH[1]
        let h1 = w.invocations[1].hash;
        for r in 1..=4 {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_REL1_HASH_OFFSET + k].evaluations[r].to_u64(),
                    h1[k] as u64,
                    "REL1[r={},k={}]", r, k,
                );
            }
        }
        // REL2: rows 2..5 = HASH[2]
        let h2 = w.invocations[2].hash;
        for r in 2..=5 {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_REL2_HASH_OFFSET + k].evaluations[r].to_u64(),
                    h2[k] as u64,
                    "REL2[r={},k={}]", r, k,
                );
            }
        }
        // REL3: rows 3..5 = HASH[3]
        let h3 = w.invocations[3].hash;
        for r in 3..=5 {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_REL3_HASH_OFFSET + k].evaluations[r].to_u64(),
                    h3[k] as u64,
                    "REL3[r={},k={}]", r, k,
                );
            }
        }
        // Sanity: at consumer rows, LEFT/RIGHT match the relayed hashes.
        // Row 4 LEFT = HASH[0], Row 4 RIGHT = HASH[1].
        for k in 0..32 {
            assert_eq!(
                trace.columns[COL_LEFT_OFFSET + k].evaluations[4].to_u64(),
                h0[k] as u64, "LEFT[4]={}", k,
            );
            assert_eq!(
                trace.columns[COL_RIGHT_OFFSET + k].evaluations[4].to_u64(),
                h1[k] as u64, "RIGHT[4]={}", k,
            );
            // Row 5 LEFT = HASH[2], Row 5 RIGHT = HASH[3].
            assert_eq!(
                trace.columns[COL_LEFT_OFFSET + k].evaluations[5].to_u64(),
                h2[k] as u64, "LEFT[5]={}", k,
            );
            assert_eq!(
                trace.columns[COL_RIGHT_OFFSET + k].evaluations[5].to_u64(),
                h3[k] as u64, "RIGHT[5]={}", k,
            );
        }
    }

    #[test]
    fn rel4_hash_relay_populated_in_witness() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // REL4_HASH = HASH[row 4] at rows 4, 5, 6.
        let row4_hash = w.invocations[4].hash;
        for r in 4..=6 {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_REL4_HASH_OFFSET + k].evaluations[r].to_u64(),
                    row4_hash[k] as u64,
                    "REL4_HASH[r={},k={}] mismatch", r, k,
                );
            }
        }
        // Earlier rows (0..3): REL4_HASH = 0.
        for r in 0..4 {
            for k in 0..32 {
                assert!(
                    trace.columns[COL_REL4_HASH_OFFSET + k].evaluations[r].is_zero(),
                    "REL4_HASH[r={},k={}] should be zero on pre-pickup row",
                    r, k,
                );
            }
        }
    }

    #[test]
    fn root_left_eq_rel4_hash_holds_on_honest_witness() {
        // At row 6, LEFT (root's first child) must equal HASH[4] (layer-2 pair-0).
        // The relay column makes this binding indirect: REL4_HASH[6] = HASH[4]
        // via the constancy chain, and row-local enforces LEFT[6] = REL4_HASH[6].
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for k in 0..32 {
            assert_eq!(
                trace.columns[COL_LEFT_OFFSET + k].evaluations[6].to_u64(),
                trace.columns[COL_HASH_OFFSET + k].evaluations[4].to_u64(),
                "byte {} of LEFT[6] vs HASH[4] mismatch", k,
            );
        }
    }

    #[test]
    fn root_chain_binding_holds_on_honest_witness() {
        // At row 5: HASH[5] is the layer-2 pair1 hash; at row 6 the root pair
        // consumes it as RIGHT input. Constraint:
        //   IS_PAIR5(5) * IS_ROOT(6) * (RIGHT[6] - HASH[5]) = 0 (β-RLC)
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // Confirm structural: HASH[5] equals RIGHT[6] byte-by-byte.
        for k in 0..32 {
            assert_eq!(
                trace.columns[COL_HASH_OFFSET + k].evaluations[5].to_u64(),
                trace.columns[COL_RIGHT_OFFSET + k].evaluations[6].to_u64(),
                "byte {} of HASH[5] vs RIGHT[6] mismatch", k,
            );
        }
    }

    #[test]
    fn pair0_slot_le_byte_decomp_populated_in_witness() {
        let mut h = sample_header();
        h.slot = 0x1122334455667788u64;
        h.proposer_index = 0x99aabbccddeeff00u64;
        let w = BeaconBlockHeaderHtrWitness::from_header(h);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let slot_bytes = h.slot.to_le_bytes();
        let pi_bytes = h.proposer_index.to_le_bytes();
        for k in 0..8 {
            assert_eq!(
                trace.columns[COL_SLOT_BYTE_OFFSET + k].evaluations[0].to_u64(),
                slot_bytes[k] as u64,
            );
            assert_eq!(
                trace.columns[COL_PROPOSER_INDEX_BYTE_OFFSET + k].evaluations[0].to_u64(),
                pi_bytes[k] as u64,
            );
        }
        // pair0 selector set on row 0, pair3 selector set on row 3.
        assert_eq!(trace.columns[COL_IS_PAIR0].evaluations[0].to_u64(), 1);
        assert!(trace.columns[COL_IS_PAIR0].evaluations[1].is_zero());
        assert_eq!(trace.columns[COL_IS_PAIR3].evaluations[3].to_u64(), 1);
    }

    #[test]
    fn pair0_left_low_eq_slot_fires_on_tampered_slot_byte() {
        let mut h = sample_header();
        h.slot = 0xabcdu64;
        let w = BeaconBlockHeaderHtrWitness::from_header(h);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: change slot byte 0 — must break pair0_left_low_eq_slot at row 0.
        cols[COL_SLOT_BYTE_OFFSET][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 9 = pair0_left_low_eq_slot_le.
        assert!(!res[9][0].is_zero(), "pair0_left_low_eq_slot must fire");
    }

    #[test]
    fn pair0_left_high_eq_zero_fires_on_nonzero_high_byte() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: place a non-zero byte in LEFT[15] at row 0 — htr_uint
        // requires LEFT[8..32] = 0.
        cols[COL_LEFT_OFFSET + 15][0] = Scalar::from_u64(0x55, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 10 = pair0_left_high_eq_zero.
        assert!(!res[10][0].is_zero(), "pair0_left_high_eq_zero must fire");
    }

    #[test]
    fn pair3_left_zero_fires_on_nonzero_left_at_row3() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Honest row 3 has LEFT = ZERO_CHUNK; tamper one byte.
        cols[COL_LEFT_OFFSET + 7][3] = Scalar::from_u64(0x77, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 13 = pair3_left_eq_zero_chunk.
        assert!(!res[13][3].is_zero(), "pair3_left_zero must fire on tamper");
    }

    #[test]
    fn header_field_columns_constant_across_real_rows() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let pr0 = &trace.columns[COL_PARENT_ROOT_OFFSET].evaluations[0];
        let sr0 = &trace.columns[COL_STATE_ROOT_OFFSET].evaluations[0];
        let br0 = &trace.columns[COL_BODY_ROOT_OFFSET].evaluations[0];
        for r in 1..trace.num_rows {
            assert_eq!(
                trace.columns[COL_PARENT_ROOT_OFFSET].evaluations[r].to_u64(),
                pr0.to_u64(), "parent_root col not constant at row {}", r,
            );
            assert_eq!(
                trace.columns[COL_STATE_ROOT_OFFSET].evaluations[r].to_u64(),
                sr0.to_u64(), "state_root col not constant at row {}", r,
            );
            assert_eq!(
                trace.columns[COL_BODY_ROOT_OFFSET].evaluations[r].to_u64(),
                br0.to_u64(), "body_root col not constant at row {}", r,
            );
        }
    }

    #[test]
    fn is_real_binary_constraint_fires_on_tampered_selector() {
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(7, CurveType::Bls48581);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[0][0].is_zero(), "is_real_binary should fire on non-binary");
    }

    #[test]
    fn linkage_descriptor_shape_matches_sha256_extract() {
        let d = make_beacon_block_header_pair_to_sha256_extract_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 96);
        assert_eq!(d.b_columns.len(), 96);
        assert_eq!(d.label, "beacon_block_header_pair_sha256_extract_v1");
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(crate::sha256_extract::COL_IS_REAL));
    }

    /// Phase C3 step 1 cap-stone: 2-AIR `joint_prove` + `joint_verify`
    /// linking BBH-pair AIR ↔ Sha256Extract via the 96-byte tuple
    /// linkage. Validates that each pair invocation row's
    /// `(left || right, hash)` matches a real Sha256Extract row.
    ///
    /// Marked `#[ignore]` — slow: ~5-10 min release expected.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with 1 96-col linkage \
                (~5-10 min); run with --release --ignored"]
    fn joint_prove_bbh_pair_to_sha256_extract() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::sha256_extract::{
            self as se, build_trace_polynomials as build_se_trace,
            Sha256ExtractConstraintSystem,
        };
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());

        // Pair-AIR trace.
        let pair_trace = build_trace_polynomials(&w, curve);
        let pair_omega = scheme.domain_generator(pair_trace.padded_size);
        let pair_cs = BeaconBlockHeaderPairConstraintSystem::new(pair_trace.num_rows)
            .with_omega_and_domain(pair_omega, pair_trace.padded_size);

        // Sha256Extract trace from the same 7 invocations (chunk pairs).
        let pairs: Vec<(crate::ssz::Chunk, crate::ssz::Chunk)> = w
            .invocations
            .iter()
            .map(|inv| (inv.left, inv.right))
            .collect();
        let se_w = se::Sha256ExtractWitness::from_pair_inputs(&pairs);
        let se_trace = build_se_trace(&se_w, curve);
        let se_omega = scheme.domain_generator(se_trace.padded_size);
        let se_cs = Sha256ExtractConstraintSystem::new(se_trace.num_rows)
            .with_omega_and_domain(se_omega, se_trace.padded_size);

        let linkage =
            make_beacon_block_header_pair_to_sha256_extract_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&pair_trace, &pair_cs), (&se_trace, &se_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("BBH-pair ↔ Sha256Extract joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&pair_cs, &se_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
            "joint_verify must accept honest witness",
        );
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;
        let w = BeaconBlockHeaderHtrWitness::from_header(sample_header());
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = BeaconBlockHeaderPairConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }
}
