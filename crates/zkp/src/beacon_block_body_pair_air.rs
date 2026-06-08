//! BeaconBlockBody pair-invocation AIR — Phase C↔B bridge step 4 step 1.
//!
//! Per-row exposes one `sha256_pair(left, right) -> hash` invocation
//! drawn from a [`BeaconBlockBodyHtrWitness`]. Companion to step 0's
//! [`crate::beacon_block_body_air`] witness builder; mirrors the
//! architecture of [`crate::beacon_block_header_pair_air`].
//!
//! Each row carries 96 byte columns (`LEFT[32] || RIGHT[32] || HASH[32]`)
//! plus `IS_REAL`. The cross-AIR LogUp descriptor
//! [`make_body_pair_to_sha256_extract_linkage_descriptor`] binds each
//! row's `(left||right, hash)` tuple to a real [`crate::sha256_extract`]
//! invocation, transitively pulling in the bit-level SHA-256 binding.
//!
//! # Soundness scope (step 1)
//!
//! - Proves: each row's `(input_64, output_32)` matches a row in
//!   Sha256Extract (which itself binds to bit-level SHA-256).
//! - Does NOT yet prove (deferred to step 2+):
//!   - Layer chaining across the 4 layers (12→6→3→2→1).
//!   - Leaf binding to the 12 BeaconBlockBody field roots (in
//!     particular field index 9 = execution_payload_header_root).
//!   - ZERO_CHUNK pin at layer 2's odd-tail right (uses ZH(2)).
//!   - CLAIMED_ROOT output column with cross-row constancy.
//!
//! All of these follow the same patterns landed in BBH-pair AIR's
//! steps 2a/2b/2c/2d/2e.

use crate::beacon_block_body_air::BeaconBlockBodyHtrWitness;
use crate::beacon_block_header_air::Sha256PairInvocation;
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

// Step 2e additions:
//   COL_CLAIMED_ROOT[0..32] — body's computed root, replicated across
//     every real row for cross-row constancy.
//   COL_IS_ROOT_BOUND_AT — one-hot row selector =1 only at the row
//     index of the root invocation (invocation index 11, the final
//     layer-3 output). Used to:
//       (a) gate the row-local "HASH == CLAIMED_ROOT" binding so it
//           only fires where HASH carries the actual computed root;
//       (b) gate the body→BBH cross-AIR LogUp descriptor so the
//           multiset has exactly 1 entry (matching BBH's IS_PAIR2
//           single-entry gating on the body_root leaf).
pub const COL_CLAIMED_ROOT_OFFSET: usize = 97;  // 97..129 (32 bytes)
pub const COL_IS_ROOT_BOUND_AT: usize = 129;
/// Step 2f: one-hot row selector =1 at the invocation where body
/// field 9 (= execution_payload_header_root) is the RIGHT operand.
/// Layer 0 pairs are (0,1)..(10,11); field 9 sits in pair (8,9) which
/// is layer-0 pair index 4 — invocation index 4 in evaluation order.
pub const COL_IS_BODY_FIELD_9_BOUND_AT: usize = 130;

// Step 2a: leaf binding for all 12 body fields.
pub const COL_CLAIMED_LEFT_OFFSET: usize = 131;   // 131..163
pub const COL_CLAIMED_RIGHT_OFFSET: usize = 163;  // 163..195
pub const COL_IS_LAYER_0: usize = 195;

// Step 2b-min: offset -1 chain binding. Only ONE offset-1 binding
// exists in the body tree: Row 11 RIGHT = Row 10 HASH. (Row 10's
// RIGHT is ZH(2) because layer 1 had 3 outputs → odd-tail padding.)
pub const COL_IS_PAIR10: usize = 196;

// ZH(2) pin: row 10's RIGHT must equal zero_hash(2).
pub const COL_ZH2_OFFSET: usize = 197;  // 197..229

// Offset-2 relay columns + per-row selectors for layer-1 and layer-2.
pub const COL_IS_PAIR7: usize = 229;
pub const COL_IS_PAIR8: usize = 230;
pub const COL_IS_PAIR9: usize = 231;
pub const COL_REL7_HASH_OFFSET: usize = 232;  // 232..264: Row 7 HASH → Row 9 RIGHT
pub const COL_REL8_HASH_OFFSET: usize = 264;  // 264..296: Row 8 HASH → Row 10 LEFT
pub const COL_REL9_HASH_OFFSET: usize = 296;  // 296..328: Row 9 HASH → Row 11 LEFT

// Per-layer-0-row selectors (IS_PAIR0..IS_PAIR5) + IS_PAIR6 for row 6.
pub const COL_IS_PAIR0: usize = 328;
pub const COL_IS_PAIR1: usize = 329;
pub const COL_IS_PAIR2: usize = 330;
pub const COL_IS_PAIR3: usize = 331;
pub const COL_IS_PAIR4: usize = 332;
pub const COL_IS_PAIR5: usize = 333;
pub const COL_IS_PAIR6: usize = 334;

// Remaining relay columns (offsets -3 through -6).
pub const COL_REL6_HASH_OFFSET: usize = 335;  // 335..367: Row 6 → Row 9 LEFT (offset -3)
pub const COL_REL5_HASH_OFFSET: usize = 367;  // 367..399: Row 5 → Row 8 RIGHT (offset -3)
pub const COL_REL4_HASH_OFFSET: usize = 399;  // 399..431: Row 4 → Row 8 LEFT (offset -4)
pub const COL_REL3_HASH_OFFSET: usize = 431;  // 431..463: Row 3 → Row 7 RIGHT (offset -4)
pub const COL_REL2_HASH_OFFSET: usize = 463;  // 463..495: Row 2 → Row 7 LEFT (offset -5)
pub const COL_REL1_HASH_OFFSET: usize = 495;  // 495..527: Row 1 → Row 6 RIGHT (offset -5)
pub const COL_REL0_HASH_OFFSET: usize = 527;  // 527..559: Row 0 → Row 6 LEFT (offset -6)
pub const NUM_COLUMNS: usize = COL_REL0_HASH_OFFSET + 32; // 559

/// Invocation index of the body's root (the final layer-3 pair).
/// Body has 12 invocations: layer 0 (6) + layer 1 (3) + layer 2 (2) +
/// layer 3 (1). Indices 0..11. Index 11 is the root.
pub const ROOT_INVOCATION_INDEX: usize = 11;

/// Invocation index where body field 9 (payload_root) is the RIGHT
/// operand of a pair invocation. Layer 0 pair (8, 9) is index 4.
pub const BODY_FIELD_9_RIGHT_INVOCATION: usize = 4;

// Row-locals:
//   0: is_real binary
//   1: is_root_bound_at binary
//   2: is_root_bound_at * Σ β^k * (HASH[k] - CLAIMED_ROOT[k])
//   3: is_body_field_9_bound_at binary
//   4: is_layer_0 binary
//   5: is_layer_0 * Σ β^k * (LEFT[k] - CLAIMED_LEFT[k])  (leaf left binding)
//   6: is_layer_0 * Σ β^k * (RIGHT[k] - CLAIMED_RIGHT[k]) (leaf right binding)
//   7: is_pair10 binary
//   8: is_pair10 * Σ β^k * (RIGHT[k] - ZH2[k]) — ZH(2) pin at row 10
//   9: is_pair7 binary
//  10: is_pair8 binary
//  11: is_pair9 binary
//  12: is_pair7 * Σ β^k * (REL7[k] - HASH[k]) — REL7 pickup at row 7
//  13: is_pair9 * Σ β^k * (RIGHT[k] - REL7[k]) — REL7 consumption at row 9
//  14: is_pair8 * Σ β^k * (REL8[k] - HASH[k]) — REL8 pickup at row 8
//  15: is_pair10 * Σ β^k * (LEFT[k] - REL8[k]) — REL8 consumption at row 10
//  16: is_pair9 * Σ β^k * (REL9[k] - HASH[k]) — REL9 pickup at row 9
//  17: is_root * Σ β^k * (LEFT[k] - REL9[k]) — REL9 consumption at row 11
pub const NUM_ROW_CONSTRAINTS: usize = 39; // 18 + 7 binary + 14 pickup/consumption

pub const NUM_SHIFTED: usize = 12; // 5 existing + 7 relay constancy bodies

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BeaconBlockBodyHtrWitness,
    curve: CurveType,
) -> TracePolynomials {
    build_trace_polynomials_from_invocations(&witness.invocations, curve)
}

pub fn build_trace_polynomials_from_invocations(
    invocations: &[Sha256PairInvocation],
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

    // Step 2e: replicate CLAIMED_ROOT (= body's computed root, taken
    // from the root invocation's hash output) across every real row.
    // Set IS_ROOT_BOUND_AT = 1 only at the root invocation row.
    if num_rows > ROOT_INVOCATION_INDEX {
        let root_hash = invocations[ROOT_INVOCATION_INDEX].hash;
        for k in 0..32 {
            let v = Scalar::from_u64(root_hash[k] as u64, curve);
            for r in 0..num_rows {
                columns[COL_CLAIMED_ROOT_OFFSET + k][r] = v.clone();
            }
        }
        columns[COL_IS_ROOT_BOUND_AT][ROOT_INVOCATION_INDEX] = one.clone();
    }

    // Step 2f: one-hot selector for the payload_root leaf (field 9).
    if num_rows > BODY_FIELD_9_RIGHT_INVOCATION {
        columns[COL_IS_BODY_FIELD_9_BOUND_AT][BODY_FIELD_9_RIGHT_INVOCATION] = one.clone();
    }

    // Per-row selectors for all non-layer-0 rows.
    for (row_idx, col) in [
        (0, COL_IS_PAIR0), (1, COL_IS_PAIR1), (2, COL_IS_PAIR2),
        (3, COL_IS_PAIR3), (4, COL_IS_PAIR4), (5, COL_IS_PAIR5),
        (6, COL_IS_PAIR6), (7, COL_IS_PAIR7), (8, COL_IS_PAIR8),
        (9, COL_IS_PAIR9), (10, COL_IS_PAIR10),
    ] {
        if num_rows > row_idx { columns[col][row_idx] = one.clone(); }
    }

    // Offset-2 relay columns.
    // REL7: Row 7 HASH relayed to rows 7..9.
    if num_rows > 7 {
        let h = invocations[7].hash;
        for k in 0..32 { let v = Scalar::from_u64(h[k] as u64, curve);
            for r in 7..num_rows.min(10) { columns[COL_REL7_HASH_OFFSET + k][r] = v.clone(); } }
    }
    // REL8: Row 8 HASH relayed to rows 8..10.
    if num_rows > 8 {
        let h = invocations[8].hash;
        for k in 0..32 { let v = Scalar::from_u64(h[k] as u64, curve);
            for r in 8..num_rows.min(11) { columns[COL_REL8_HASH_OFFSET + k][r] = v.clone(); } }
    }
    // REL9: Row 9 HASH relayed to rows 9..11.
    if num_rows > 9 {
        let h = invocations[9].hash;
        for k in 0..32 { let v = Scalar::from_u64(h[k] as u64, curve);
            for r in 9..num_rows.min(12) { columns[COL_REL9_HASH_OFFSET + k][r] = v.clone(); } }
    }

    // Remaining relay columns (offsets -3 through -6).
    let relay_specs: &[(usize, usize, usize, usize)] = &[
        // (producer_row, relay_col_offset, relay_start, relay_end_exclusive)
        (6, COL_REL6_HASH_OFFSET, 6, 10),  // Row 6 → Row 9 LEFT
        (5, COL_REL5_HASH_OFFSET, 5, 9),   // Row 5 → Row 8 RIGHT
        (4, COL_REL4_HASH_OFFSET, 4, 9),   // Row 4 → Row 8 LEFT
        (3, COL_REL3_HASH_OFFSET, 3, 8),   // Row 3 → Row 7 RIGHT
        (2, COL_REL2_HASH_OFFSET, 2, 8),   // Row 2 → Row 7 LEFT
        (1, COL_REL1_HASH_OFFSET, 1, 7),   // Row 1 → Row 6 RIGHT
        (0, COL_REL0_HASH_OFFSET, 0, 7),   // Row 0 → Row 6 LEFT
    ];
    for &(prod, col_off, start, end) in relay_specs {
        if num_rows > prod {
            let h = invocations[prod].hash;
            for k in 0..32 {
                let v = Scalar::from_u64(h[k] as u64, curve);
                for r in start..num_rows.min(end) {
                    columns[col_off + k][r] = v.clone();
                }
            }
        }
    }

    // ZH(2) pin: populate the zero_hash(2) constant on all real rows.
    // zero_hash(d) = sha256_pair chain d times from ZERO_CHUNK.
    let zh2 = {
        let z0 = crate::ssz::ZERO_CHUNK;
        let z1 = crate::sha256::sha256_pair(&z0, &z0);
        crate::sha256::sha256_pair(&z1, &z1)
    };
    for k in 0..32 {
        let v = Scalar::from_u64(zh2[k] as u64, curve);
        for r in 0..num_rows {
            columns[COL_ZH2_OFFSET + k][r] = v.clone();
        }
    }

    // Step 2a: leaf binding for layer 0 (rows 0..5).
    let num_layer0_pairs = 6usize.min(num_rows);
    for r in 0..num_layer0_pairs {
        columns[COL_IS_LAYER_0][r] = one.clone();
        for k in 0..32 {
            columns[COL_CLAIMED_LEFT_OFFSET + k][r] =
                Scalar::from_u64(invocations[r].left[k] as u64, curve);
            columns[COL_CLAIMED_RIGHT_OFFSET + k][r] =
                Scalar::from_u64(invocations[r].right[k] as u64, curve);
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

// ─── Constraint system ─────────────────────────────────────────────────

pub struct BeaconBlockBodyPairConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BeaconBlockBodyPairConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for BeaconBlockBodyPairConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_root_bound_at_binary".into(),
            "hash_eq_claimed_root_at_root_invocation".into(),
            "is_body_field_9_bound_at_binary".into(),
            "is_layer_0_binary".into(),
            "leaf_left_binding_rlc".into(),
            "leaf_right_binding_rlc".into(),
            "is_pair10_binary".into(),
            "zh2_pin_right_at_pair10".into(),
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
        let mut is_root_bound_at_bin = vec![Scalar::zero(curve); n];
        let mut hash_eq_claimed = vec![Scalar::zero(curve); n];
        let mut is_field9_bin = vec![Scalar::zero(curve); n];
        let mut is_layer0_bin = vec![Scalar::zero(curve); n];
        let mut leaf_left = vec![Scalar::zero(curve); n];
        let mut leaf_right = vec![Scalar::zero(curve); n];

        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            is_real_bin[r] = v.mul(&v.sub(&one));
            let rb = &columns[COL_IS_ROOT_BOUND_AT][r];
            is_root_bound_at_bin[r] = rb.mul(&rb.sub(&one));
            let f9 = &columns[COL_IS_BODY_FIELD_9_BOUND_AT][r];
            is_field9_bin[r] = f9.mul(&f9.sub(&one));
            let l0 = &columns[COL_IS_LAYER_0][r];
            is_layer0_bin[r] = l0.mul(&l0.sub(&one));

            let mut acc = Scalar::zero(curve);
            let mut acc_ll = Scalar::zero(curve);
            let mut acc_lr = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let h = &columns[COL_HASH_OFFSET + k][r];
                let c = &columns[COL_CLAIMED_ROOT_OFFSET + k][r];
                let l = &columns[COL_LEFT_OFFSET + k][r];
                let rr = &columns[COL_RIGHT_OFFSET + k][r];
                let cl = &columns[COL_CLAIMED_LEFT_OFFSET + k][r];
                let cr = &columns[COL_CLAIMED_RIGHT_OFFSET + k][r];
                acc = acc.add(&bp.mul(&h.sub(c)));
                acc_ll = acc_ll.add(&bp.mul(&l.sub(cl)));
                acc_lr = acc_lr.add(&bp.mul(&rr.sub(cr)));
                bp = bp.mul(&beta_test);
            }
            hash_eq_claimed[r] = rb.mul(&acc);
            leaf_left[r] = l0.mul(&acc_ll);
            leaf_right[r] = l0.mul(&acc_lr);
        }
        let mut is_pair10_bin = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let p10 = &columns[COL_IS_PAIR10][r];
            is_pair10_bin[r] = p10.mul(&p10.sub(&one));
        }
        // ZH2 pin: IS_PAIR10 * Σ β^k * (RIGHT[k] - ZH2[k])
        let mut zh2_pin = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let p10 = &columns[COL_IS_PAIR10][r];
            let mut acc_zh = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let rr = &columns[COL_RIGHT_OFFSET + k][r];
                let zh = &columns[COL_ZH2_OFFSET + k][r];
                acc_zh = acc_zh.add(&bp.mul(&rr.sub(zh)));
                bp = bp.mul(&beta_test);
            }
            zh2_pin[r] = p10.mul(&acc_zh);
        }
        // Offset-2 relay constraints.
        let mk2 = || vec![Scalar::zero(curve); n];
        let (mut p7b,mut p8b,mut p9b) = (mk2(),mk2(),mk2());
        let (mut r7pk,mut r7cm,mut r8pk,mut r8cm,mut r9pk,mut r9cm) = (mk2(),mk2(),mk2(),mk2(),mk2(),mk2());
        for r in 0..n {
            let s7 = &columns[COL_IS_PAIR7][r]; p7b[r] = s7.mul(&s7.sub(&one));
            let s8 = &columns[COL_IS_PAIR8][r]; p8b[r] = s8.mul(&s8.sub(&one));
            let s9 = &columns[COL_IS_PAIR9][r]; p9b[r] = s9.mul(&s9.sub(&one));
            let rb_root = &columns[COL_IS_ROOT_BOUND_AT][r];
            let s10 = &columns[COL_IS_PAIR10][r];
            let mut a7pk=Scalar::zero(curve); let mut a7cm=Scalar::zero(curve);
            let mut a8pk=Scalar::zero(curve); let mut a8cm=Scalar::zero(curve);
            let mut a9pk=Scalar::zero(curve); let mut a9cm=Scalar::zero(curve);
            let mut bp=Scalar::one(curve);
            for k in 0..32 {
                let h=&columns[COL_HASH_OFFSET+k][r]; let l=&columns[COL_LEFT_OFFSET+k][r]; let rr=&columns[COL_RIGHT_OFFSET+k][r];
                let rl7=&columns[COL_REL7_HASH_OFFSET+k][r]; let rl8=&columns[COL_REL8_HASH_OFFSET+k][r]; let rl9=&columns[COL_REL9_HASH_OFFSET+k][r];
                a7pk=a7pk.add(&bp.mul(&rl7.sub(h))); a7cm=a7cm.add(&bp.mul(&rr.sub(rl7)));
                a8pk=a8pk.add(&bp.mul(&rl8.sub(h))); a8cm=a8cm.add(&bp.mul(&l.sub(rl8)));
                a9pk=a9pk.add(&bp.mul(&rl9.sub(h))); a9cm=a9cm.add(&bp.mul(&l.sub(rl9)));
                bp=bp.mul(&beta_test);
            }
            r7pk[r]=s7.mul(&a7pk); r7cm[r]=s9.mul(&a7cm);
            r8pk[r]=s8.mul(&a8pk); r8cm[r]=s10.mul(&a8cm);
            r9pk[r]=s9.mul(&a9pk); r9cm[r]=rb_root.mul(&a9cm);
        }
        let mut result = vec![is_real_bin, is_root_bound_at_bin, hash_eq_claimed, is_field9_bin,
             is_layer0_bin, leaf_left, leaf_right, is_pair10_bin, zh2_pin,
             p7b, p8b, p9b, r7pk, r7cm, r8pk, r8cm, r9pk, r9cm];
        // Bulk: 7 binary + 14 pickup/consumption for REL0..6.
        let relay_table: &[(usize, usize, usize, bool)] = &[
            (COL_IS_PAIR6, COL_IS_PAIR9, COL_REL6_HASH_OFFSET, true),
            (COL_IS_PAIR5, COL_IS_PAIR8, COL_REL5_HASH_OFFSET, false),
            (COL_IS_PAIR4, COL_IS_PAIR8, COL_REL4_HASH_OFFSET, true),
            (COL_IS_PAIR3, COL_IS_PAIR7, COL_REL3_HASH_OFFSET, false),
            (COL_IS_PAIR2, COL_IS_PAIR7, COL_REL2_HASH_OFFSET, true),
            (COL_IS_PAIR1, COL_IS_PAIR6, COL_REL1_HASH_OFFSET, false),
            (COL_IS_PAIR0, COL_IS_PAIR6, COL_REL0_HASH_OFFSET, true),
        ];
        for &(pk_sel, cm_sel, rel_off, is_left) in relay_table {
            let mut bin_v = vec![Scalar::zero(curve); n];
            let mut pk_v = vec![Scalar::zero(curve); n];
            let mut cm_v = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let ps = &columns[pk_sel][r];
                bin_v[r] = ps.mul(&ps.sub(&one));
                let cs_sel = &columns[cm_sel][r];
                let mut apk = Scalar::zero(curve); let mut acm = Scalar::zero(curve);
                let mut bp = Scalar::one(curve);
                for k in 0..32 {
                    let h = &columns[COL_HASH_OFFSET+k][r]; let rl = &columns[rel_off+k][r];
                    let tgt = if is_left { &columns[COL_LEFT_OFFSET+k][r] } else { &columns[COL_RIGHT_OFFSET+k][r] };
                    apk = apk.add(&bp.mul(&rl.sub(h)));
                    acm = acm.add(&bp.mul(&tgt.sub(rl)));
                    bp = bp.mul(&beta_test);
                }
                pk_v[r] = ps.mul(&apk);
                cm_v[r] = cs_sel.mul(&acm);
            }
            result.push(bin_v); result.push(pk_v); result.push(cm_v);
        }
        result
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        let rb = &col_evals[COL_IS_ROOT_BOUND_AT];
        let f9 = &col_evals[COL_IS_BODY_FIELD_9_BOUND_AT];
        let l0 = &col_evals[COL_IS_LAYER_0];
        let is_real_bin = v.mul(&v.sub(&one));
        let is_root_bound_at_bin = rb.mul(&rb.sub(&one));
        let is_field9_bin = f9.mul(&f9.sub(&one));
        let is_layer0_bin = l0.mul(&l0.sub(&one));

        let mut acc = Scalar::zero(curve);
        let mut acc_ll = Scalar::zero(curve);
        let mut acc_lr = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let h = &col_evals[COL_HASH_OFFSET + k];
            let c = &col_evals[COL_CLAIMED_ROOT_OFFSET + k];
            let l = &col_evals[COL_LEFT_OFFSET + k];
            let rr = &col_evals[COL_RIGHT_OFFSET + k];
            let cl = &col_evals[COL_CLAIMED_LEFT_OFFSET + k];
            let cr = &col_evals[COL_CLAIMED_RIGHT_OFFSET + k];
            acc = acc.add(&bp.mul(&h.sub(c)));
            acc_ll = acc_ll.add(&bp.mul(&l.sub(cl)));
            acc_lr = acc_lr.add(&bp.mul(&rr.sub(cr)));
            bp = bp.mul(alpha);
        }
        let hash_eq_claimed = rb.mul(&acc);
        let leaf_left = l0.mul(&acc_ll);
        let leaf_right = l0.mul(&acc_lr);

        let mut total = is_real_bin;
        let mut ap = alpha.clone();
        total = total.add(&ap.mul(&is_root_bound_at_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&hash_eq_claimed));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_field9_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_layer0_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&leaf_left));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&leaf_right));
        let p10 = &col_evals[COL_IS_PAIR10];
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&p10.mul(&p10.sub(&one))));
        // ZH2 pin.
        let mut acc_zh = Scalar::zero(curve);
        let mut bp_zh = Scalar::one(curve);
        for k in 0..32 {
            let rr = &col_evals[COL_RIGHT_OFFSET + k];
            let zh = &col_evals[COL_ZH2_OFFSET + k];
            acc_zh = acc_zh.add(&bp_zh.mul(&rr.sub(zh)));
            bp_zh = bp_zh.mul(alpha);
        }
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&p10.mul(&acc_zh)));
        // Offset-2 relay constraints.
        let s7=&col_evals[COL_IS_PAIR7]; let s8=&col_evals[COL_IS_PAIR8]; let s9=&col_evals[COL_IS_PAIR9];
        let rb_root=&col_evals[COL_IS_ROOT_BOUND_AT];
        for (sel, bin_body) in [(s7,s7),(s8,s8),(s9,s9)] {
            ap=ap.mul(alpha); total=total.add(&ap.mul(&sel.mul(&bin_body.sub(&one))));
        }
        let mut a7pk=Scalar::zero(curve); let mut a7cm=Scalar::zero(curve);
        let mut a8pk=Scalar::zero(curve); let mut a8cm=Scalar::zero(curve);
        let mut a9pk=Scalar::zero(curve); let mut a9cm=Scalar::zero(curve);
        let mut bp2=Scalar::one(curve);
        for k in 0..32 {
            let h=&col_evals[COL_HASH_OFFSET+k]; let l=&col_evals[COL_LEFT_OFFSET+k]; let rr=&col_evals[COL_RIGHT_OFFSET+k];
            let rl7=&col_evals[COL_REL7_HASH_OFFSET+k]; let rl8=&col_evals[COL_REL8_HASH_OFFSET+k]; let rl9=&col_evals[COL_REL9_HASH_OFFSET+k];
            a7pk=a7pk.add(&bp2.mul(&rl7.sub(h))); a7cm=a7cm.add(&bp2.mul(&rr.sub(rl7)));
            a8pk=a8pk.add(&bp2.mul(&rl8.sub(h))); a8cm=a8cm.add(&bp2.mul(&l.sub(rl8)));
            a9pk=a9pk.add(&bp2.mul(&rl9.sub(h))); a9cm=a9cm.add(&bp2.mul(&l.sub(rl9)));
            bp2=bp2.mul(alpha);
        }
        ap=ap.mul(alpha); total=total.add(&ap.mul(&s7.mul(&a7pk)));
        ap=ap.mul(alpha); total=total.add(&ap.mul(&s9.mul(&a7cm)));
        ap=ap.mul(alpha); total=total.add(&ap.mul(&s8.mul(&a8pk)));
        ap=ap.mul(alpha); total=total.add(&ap.mul(&p10.mul(&a8cm)));
        ap=ap.mul(alpha); total=total.add(&ap.mul(&s9.mul(&a9pk)));
        ap=ap.mul(alpha); total=total.add(&ap.mul(&rb_root.mul(&a9cm)));
        // Bulk REL0..6 constraints.
        let relay_table: &[(usize, usize, usize, bool)] = &[
            (COL_IS_PAIR6, COL_IS_PAIR9, COL_REL6_HASH_OFFSET, true),
            (COL_IS_PAIR5, COL_IS_PAIR8, COL_REL5_HASH_OFFSET, false),
            (COL_IS_PAIR4, COL_IS_PAIR8, COL_REL4_HASH_OFFSET, true),
            (COL_IS_PAIR3, COL_IS_PAIR7, COL_REL3_HASH_OFFSET, false),
            (COL_IS_PAIR2, COL_IS_PAIR7, COL_REL2_HASH_OFFSET, true),
            (COL_IS_PAIR1, COL_IS_PAIR6, COL_REL1_HASH_OFFSET, false),
            (COL_IS_PAIR0, COL_IS_PAIR6, COL_REL0_HASH_OFFSET, true),
        ];
        for &(pk_sel, cm_sel, rel_off, is_left) in relay_table {
            let ps = &col_evals[pk_sel]; let cs2 = &col_evals[cm_sel];
            ap=ap.mul(alpha); total=total.add(&ap.mul(&ps.mul(&ps.sub(&one))));
            let mut apk=Scalar::zero(curve); let mut acm=Scalar::zero(curve);
            let mut bp3=Scalar::one(curve);
            for k in 0..32 {
                let h=&col_evals[COL_HASH_OFFSET+k]; let rl=&col_evals[rel_off+k];
                let tgt = if is_left { &col_evals[COL_LEFT_OFFSET+k] } else { &col_evals[COL_RIGHT_OFFSET+k] };
                apk=apk.add(&bp3.mul(&rl.sub(h))); acm=acm.add(&bp3.mul(&tgt.sub(rl)));
                bp3=bp3.mul(alpha);
            }
            ap=ap.mul(alpha); total=total.add(&ap.mul(&ps.mul(&apk)));
            ap=ap.mul(alpha); total=total.add(&ap.mul(&cs2.mul(&acm)));
        }
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

        let rb = &col_coeffs[COL_IS_ROOT_BOUND_AT];
        let rb_m1 = poly_sub(rb, &one_poly, curve);
        let is_root_bound_at_bin = poly_mul(rb, &rb_m1, curve);

        let f9 = &col_coeffs[COL_IS_BODY_FIELD_9_BOUND_AT];
        let f9_m1 = poly_sub(f9, &one_poly, curve);
        let is_field9_bin = poly_mul(f9, &f9_m1, curve);

        let l0 = &col_coeffs[COL_IS_LAYER_0];
        let l0_m1 = poly_sub(l0, &one_poly, curve);
        let is_layer0_bin = poly_mul(l0, &l0_m1, curve);

        let mut acc = vec![Scalar::zero(curve)];
        let mut acc_ll = vec![Scalar::zero(curve)];
        let mut acc_lr = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let h = &col_coeffs[COL_HASH_OFFSET + k];
            let c = &col_coeffs[COL_CLAIMED_ROOT_OFFSET + k];
            let l = &col_coeffs[COL_LEFT_OFFSET + k];
            let rr = &col_coeffs[COL_RIGHT_OFFSET + k];
            let cl = &col_coeffs[COL_CLAIMED_LEFT_OFFSET + k];
            let cr = &col_coeffs[COL_CLAIMED_RIGHT_OFFSET + k];
            let dh = poly_sub(h, c, curve);
            let dl = poly_sub(l, cl, curve);
            let dr = poly_sub(rr, cr, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&dh, &bp), curve);
            acc_ll = poly_add(&acc_ll, &poly_scalar_mul(&dl, &bp), curve);
            acc_lr = poly_add(&acc_lr, &poly_scalar_mul(&dr, &bp), curve);
            bp = bp.mul(alpha);
        }
        let hash_eq_claimed = poly_mul(rb, &acc, curve);
        let leaf_left = poly_mul(l0, &acc_ll, curve);
        let leaf_right = poly_mul(l0, &acc_lr, curve);

        let mut total = is_real_bin;
        let mut ap = alpha.clone();
        total = poly_add(&total, &poly_scalar_mul(&is_root_bound_at_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&hash_eq_claimed, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_field9_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_layer0_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&leaf_left, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&leaf_right, &ap), curve);
        let p10 = &col_coeffs[COL_IS_PAIR10];
        let p10_m1 = poly_sub(p10, &one_poly, curve);
        let is_p10_bin = poly_mul(p10, &p10_m1, curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_p10_bin, &ap), curve);
        // ZH2 pin.
        let mut acc_zh = vec![Scalar::zero(curve)];
        let mut bp_zh = Scalar::one(curve);
        for k in 0..32 {
            let rr = &col_coeffs[COL_RIGHT_OFFSET + k];
            let zh = &col_coeffs[COL_ZH2_OFFSET + k];
            let diff = poly_sub(rr, zh, curve);
            acc_zh = poly_add(&acc_zh, &poly_scalar_mul(&diff, &bp_zh), curve);
            bp_zh = bp_zh.mul(alpha);
        }
        let zh2_pin = poly_mul(p10, &acc_zh, curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&zh2_pin, &ap), curve);
        // Offset-2 relay constraints in coeff form.
        let s7p=&col_coeffs[COL_IS_PAIR7]; let s8p=&col_coeffs[COL_IS_PAIR8]; let s9p=&col_coeffs[COL_IS_PAIR9];
        let rbp=&col_coeffs[COL_IS_ROOT_BOUND_AT];
        for sp in [s7p, s8p, s9p] {
            let b = poly_mul(sp, &poly_sub(sp, &one_poly, curve), curve);
            ap=ap.mul(alpha); total=poly_add(&total, &poly_scalar_mul(&b, &ap), curve);
        }
        let mut a7pk=vec![Scalar::zero(curve)]; let mut a7cm=vec![Scalar::zero(curve)];
        let mut a8pk=vec![Scalar::zero(curve)]; let mut a8cm=vec![Scalar::zero(curve)];
        let mut a9pk=vec![Scalar::zero(curve)]; let mut a9cm=vec![Scalar::zero(curve)];
        let mut bp2=Scalar::one(curve);
        for k in 0..32 {
            let h=&col_coeffs[COL_HASH_OFFSET+k]; let l=&col_coeffs[COL_LEFT_OFFSET+k]; let rr=&col_coeffs[COL_RIGHT_OFFSET+k];
            let rl7=&col_coeffs[COL_REL7_HASH_OFFSET+k]; let rl8=&col_coeffs[COL_REL8_HASH_OFFSET+k]; let rl9=&col_coeffs[COL_REL9_HASH_OFFSET+k];
            a7pk=poly_add(&a7pk,&poly_scalar_mul(&poly_sub(rl7,h,curve),&bp2),curve);
            a7cm=poly_add(&a7cm,&poly_scalar_mul(&poly_sub(rr,rl7,curve),&bp2),curve);
            a8pk=poly_add(&a8pk,&poly_scalar_mul(&poly_sub(rl8,h,curve),&bp2),curve);
            a8cm=poly_add(&a8cm,&poly_scalar_mul(&poly_sub(l,rl8,curve),&bp2),curve);
            a9pk=poly_add(&a9pk,&poly_scalar_mul(&poly_sub(rl9,h,curve),&bp2),curve);
            a9cm=poly_add(&a9cm,&poly_scalar_mul(&poly_sub(l,rl9,curve),&bp2),curve);
            bp2=bp2.mul(alpha);
        }
        ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&poly_mul(s7p,&a7pk,curve),&ap),curve);
        ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&poly_mul(s9p,&a7cm,curve),&ap),curve);
        ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&poly_mul(s8p,&a8pk,curve),&ap),curve);
        ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&poly_mul(p10,&a8cm,curve),&ap),curve);
        ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&poly_mul(s9p,&a9pk,curve),&ap),curve);
        ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&poly_mul(rbp,&a9cm,curve),&ap),curve);
        // Bulk REL0..6 in coeff form.
        let relay_table: &[(usize, usize, usize, bool)] = &[
            (COL_IS_PAIR6, COL_IS_PAIR9, COL_REL6_HASH_OFFSET, true),
            (COL_IS_PAIR5, COL_IS_PAIR8, COL_REL5_HASH_OFFSET, false),
            (COL_IS_PAIR4, COL_IS_PAIR8, COL_REL4_HASH_OFFSET, true),
            (COL_IS_PAIR3, COL_IS_PAIR7, COL_REL3_HASH_OFFSET, false),
            (COL_IS_PAIR2, COL_IS_PAIR7, COL_REL2_HASH_OFFSET, true),
            (COL_IS_PAIR1, COL_IS_PAIR6, COL_REL1_HASH_OFFSET, false),
            (COL_IS_PAIR0, COL_IS_PAIR6, COL_REL0_HASH_OFFSET, true),
        ];
        for &(pk_sel, cm_sel, rel_off, is_left) in relay_table {
            let psp = &col_coeffs[pk_sel]; let csp = &col_coeffs[cm_sel];
            let b = poly_mul(psp, &poly_sub(psp, &one_poly, curve), curve);
            ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&b,&ap),curve);
            let mut apk=vec![Scalar::zero(curve)]; let mut acm=vec![Scalar::zero(curve)];
            let mut bp3=Scalar::one(curve);
            for k in 0..32 {
                let h=&col_coeffs[COL_HASH_OFFSET+k]; let rl=&col_coeffs[rel_off+k];
                let tgt = if is_left { &col_coeffs[COL_LEFT_OFFSET+k] } else { &col_coeffs[COL_RIGHT_OFFSET+k] };
                apk=poly_add(&apk,&poly_scalar_mul(&poly_sub(rl,h,curve),&bp3),curve);
                acm=poly_add(&acm,&poly_scalar_mul(&poly_sub(tgt,rl,curve),&bp3),curve);
                bp3=bp3.mul(alpha);
            }
            ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&poly_mul(psp,&apk,curve),&ap),curve);
            ap=ap.mul(alpha); total=poly_add(&total,&poly_scalar_mul(&poly_mul(csp,&acm,curve),&ap),curve);
        }
        total
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Layout: IS_REAL(1) + CLAIMED_ROOT(32) + IS_ROOT(1) + RIGHT(32)
        //       + REL7(32) + REL8(32) + REL9(32) = 162
        let mut cols = Vec::with_capacity(162);
        cols.push(COL_IS_REAL);
        for k in 0..32 { cols.push(COL_CLAIMED_ROOT_OFFSET + k); }
        cols.push(COL_IS_ROOT_BOUND_AT);
        for k in 0..32 { cols.push(COL_RIGHT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL7_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL8_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL9_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL6_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL5_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL4_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL3_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL2_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL1_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_REL0_HASH_OFFSET + k); }
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
        if shifted_evals.len() != 386 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let gating = v.mul(v_next);

        let mut acc = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let cur = &col_evals_at_z[COL_CLAIMED_ROOT_OFFSET + k];
            let nxt = &shifted_evals[1 + k];
            acc = acc.add(&bp.mul(&nxt.sub(cur)));
            bp = bp.mul(alpha);
        }
        let body = gating.mul(&acc);

        // Body 1: offset -1 chain. IS_PAIR10(r) * (RIGHT_NEXT - HASH)
        // Row 10→11: root's RIGHT = layer-2 pair-1's HASH.
        let p10 = &col_evals_at_z[COL_IS_PAIR10];
        let chain_gating = p10.clone();
        let mut chain_body = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let hash_curr = &col_evals_at_z[COL_HASH_OFFSET + k];
            let right_next = &shifted_evals[34 + k];
            chain_body = chain_body.add(&bp.mul(&right_next.sub(hash_curr)));
            bp = bp.mul(alpha);
        }
        let body_chain = chain_gating.mul(&chain_body);

        // Relay constancy bodies (offset-2).
        let make_relay_body = |rel_off: usize, shifted_base: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let cur = &col_evals_at_z[rel_off + k];
                let nxt = &shifted_evals[shifted_base + k];
                acc = acc.add(&bp.mul(&nxt.sub(cur)));
                bp = bp.mul(alpha);
            }
            acc
        };
        let s7 = &col_evals_at_z[COL_IS_PAIR7];
        let s8 = &col_evals_at_z[COL_IS_PAIR8];
        let s9 = &col_evals_at_z[COL_IS_PAIR9];
        // REL7 constancy gated by IS_PAIR7 + IS_PAIR8.
        let rel7_body = s7.add(s8).mul(&make_relay_body(COL_REL7_HASH_OFFSET, 66));
        // REL8 constancy gated by IS_PAIR8 + IS_PAIR9.
        let rel8_body = s8.add(s9).mul(&make_relay_body(COL_REL8_HASH_OFFSET, 98));
        // REL9 constancy gated by IS_PAIR9 + IS_PAIR10.
        let rel9_body = s9.add(p10).mul(&make_relay_body(COL_REL9_HASH_OFFSET, 130));

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = ap.mul(&body);
        ap = ap.mul(alpha); total = total.add(&ap.mul(&body_chain));
        ap = ap.mul(alpha); total = total.add(&ap.mul(&rel7_body));
        ap = ap.mul(alpha); total = total.add(&ap.mul(&rel8_body));
        ap = ap.mul(alpha); total = total.add(&ap.mul(&rel9_body));
        // Bulk REL0..6 constancy bodies.
        // (relay_col_offset, shifted_base, gating_selectors)
        let constancy_table: &[(usize, usize, &[usize])] = &[
            (COL_REL6_HASH_OFFSET, 162, &[COL_IS_PAIR6, COL_IS_PAIR7, COL_IS_PAIR8]),
            (COL_REL5_HASH_OFFSET, 194, &[COL_IS_PAIR5, COL_IS_PAIR6, COL_IS_PAIR7]),
            (COL_REL4_HASH_OFFSET, 226, &[COL_IS_PAIR4, COL_IS_PAIR5, COL_IS_PAIR6, COL_IS_PAIR7]),
            (COL_REL3_HASH_OFFSET, 258, &[COL_IS_PAIR3, COL_IS_PAIR4, COL_IS_PAIR5, COL_IS_PAIR6]),
            (COL_REL2_HASH_OFFSET, 290, &[COL_IS_PAIR2, COL_IS_PAIR3, COL_IS_PAIR4, COL_IS_PAIR5, COL_IS_PAIR6]),
            (COL_REL1_HASH_OFFSET, 322, &[COL_IS_PAIR1, COL_IS_PAIR2, COL_IS_PAIR3, COL_IS_PAIR4, COL_IS_PAIR5]),
            (COL_REL0_HASH_OFFSET, 354, &[COL_IS_PAIR0, COL_IS_PAIR1, COL_IS_PAIR2, COL_IS_PAIR3, COL_IS_PAIR4, COL_IS_PAIR5]),
        ];
        for &(rel_off, sh_base, gate_sels) in constancy_table {
            let mut gate = Scalar::zero(curve);
            for &gs in gate_sels { gate = gate.add(&col_evals_at_z[gs]); }
            let body = gate.mul(&make_relay_body(rel_off, sh_base));
            ap = ap.mul(alpha); total = total.add(&ap.mul(&body));
        }
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

        let mut acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let cur = &column_coeffs[COL_CLAIMED_ROOT_OFFSET + k];
            let nxt = poly_shift(cur, omega);
            let diff = poly_sub(&nxt, cur, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body = poly_mul(&gating, &acc, curve);

        // Body 1: offset -1 chain. IS_PAIR10(r) * (RIGHT_NEXT - HASH).
        let p10 = &column_coeffs[COL_IS_PAIR10];
        let mut chain_body_poly = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let hash_curr = &column_coeffs[COL_HASH_OFFSET + k];
            let right_curr = &column_coeffs[COL_RIGHT_OFFSET + k];
            let right_next = poly_shift(right_curr, omega);
            let diff = poly_sub(&right_next, hash_curr, curve);
            chain_body_poly = poly_add(&chain_body_poly, &poly_scalar_mul(&diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let body_chain = poly_mul(p10, &chain_body_poly, curve);

        // Relay constancy in coeff form.
        let make_relay_poly = |rel_off: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let cur = &column_coeffs[rel_off + k];
                let nxt = poly_shift(cur, omega);
                let diff = poly_sub(&nxt, cur, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
                bp = bp.mul(alpha);
            }
            acc
        };
        let s7p = &column_coeffs[COL_IS_PAIR7];
        let s8p = &column_coeffs[COL_IS_PAIR8];
        let s9p = &column_coeffs[COL_IS_PAIR9];
        let rel7b = poly_mul(&poly_add(s7p, s8p, curve), &make_relay_poly(COL_REL7_HASH_OFFSET), curve);
        let rel8b = poly_mul(&poly_add(s8p, s9p, curve), &make_relay_poly(COL_REL8_HASH_OFFSET), curve);
        let rel9b = poly_mul(&poly_add(s9p, p10, curve), &make_relay_poly(COL_REL9_HASH_OFFSET), curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = poly_scalar_mul(&body, &ap);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&body_chain, &ap), curve);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&rel7b, &ap), curve);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&rel8b, &ap), curve);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&rel9b, &ap), curve);
        // Bulk REL0..6 constancy in coeff form.
        let constancy_table: &[(usize, &[usize])] = &[
            (COL_REL6_HASH_OFFSET, &[COL_IS_PAIR6, COL_IS_PAIR7, COL_IS_PAIR8]),
            (COL_REL5_HASH_OFFSET, &[COL_IS_PAIR5, COL_IS_PAIR6, COL_IS_PAIR7]),
            (COL_REL4_HASH_OFFSET, &[COL_IS_PAIR4, COL_IS_PAIR5, COL_IS_PAIR6, COL_IS_PAIR7]),
            (COL_REL3_HASH_OFFSET, &[COL_IS_PAIR3, COL_IS_PAIR4, COL_IS_PAIR5, COL_IS_PAIR6]),
            (COL_REL2_HASH_OFFSET, &[COL_IS_PAIR2, COL_IS_PAIR3, COL_IS_PAIR4, COL_IS_PAIR5, COL_IS_PAIR6]),
            (COL_REL1_HASH_OFFSET, &[COL_IS_PAIR1, COL_IS_PAIR2, COL_IS_PAIR3, COL_IS_PAIR4, COL_IS_PAIR5]),
            (COL_REL0_HASH_OFFSET, &[COL_IS_PAIR0, COL_IS_PAIR1, COL_IS_PAIR2, COL_IS_PAIR3, COL_IS_PAIR4, COL_IS_PAIR5]),
        ];
        for &(rel_off, gate_sels) in constancy_table {
            let mut gate = vec![Scalar::zero(curve)];
            for &gs in gate_sels { gate = poly_add(&gate, &column_coeffs[gs], curve); }
            let rb = make_relay_poly(rel_off);
            let b = poly_mul(&gate, &rb, curve);
            ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&b, &ap), curve);
        }

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_IS_ROOT_BOUND_AT, COL_IS_BODY_FIELD_9_BOUND_AT, COL_IS_LAYER_0,
             COL_IS_PAIR0, COL_IS_PAIR1, COL_IS_PAIR2, COL_IS_PAIR3,
             COL_IS_PAIR4, COL_IS_PAIR5, COL_IS_PAIR6,
             COL_IS_PAIR7, COL_IS_PAIR8, COL_IS_PAIR9, COL_IS_PAIR10]
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
            for (name, off) in [
                ("left", COL_LEFT_OFFSET),
                ("right", COL_RIGHT_OFFSET),
                ("hash", COL_HASH_OFFSET),
                ("claimed_root", COL_CLAIMED_ROOT_OFFSET),
                ("claimed_left", COL_CLAIMED_LEFT_OFFSET),
                ("claimed_right", COL_CLAIMED_RIGHT_OFFSET),
                ("zh2", COL_ZH2_OFFSET),
                ("rel7", COL_REL7_HASH_OFFSET),
                ("rel8", COL_REL8_HASH_OFFSET),
                ("rel9", COL_REL9_HASH_OFFSET),
                ("rel6", COL_REL6_HASH_OFFSET),
                ("rel5", COL_REL5_HASH_OFFSET),
                ("rel4", COL_REL4_HASH_OFFSET),
                ("rel3", COL_REL3_HASH_OFFSET),
                ("rel2", COL_REL2_HASH_OFFSET),
                ("rel1", COL_REL1_HASH_OFFSET),
                ("rel0", COL_REL0_HASH_OFFSET),
            ] {
                declarations.push((
                    LookupDeclaration {
                        label: format!("body_pair_{}_{}_8bit", name, k),
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
/// invocation row. Layer chaining and leaf binding are deferred to
/// step 2+.
pub fn make_body_pair_to_sha256_extract_linkage_descriptor(
    body_pair_layer_index: usize,
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
        label: "beacon_block_body_pair_sha256_extract_v1".into(),
        a_layer_index: body_pair_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(crate::sha256_extract::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp linkage: body-pair's CLAIMED_ROOT (the computed
/// body root) ↔ BBH-pair's LEFT@row2 (the body_root leaf at BBH
/// invocation index 2, which is leaf 4 in the BBH container).
///
/// **A side (body-pair)**: 32-byte CLAIMED_ROOT tuple gated by
/// `COL_IS_ROOT_BOUND_AT` (=1 only at body's root invocation row 11
/// → exactly 1 entry in the multiset).
///
/// **B side (BBH-pair)**: 32-byte LEFT tuple gated by BBH-pair's
/// `COL_IS_PAIR2` (=1 only at BBH invocation row 2 → exactly 1 entry).
///
/// Multiset equality on 1-vs-1 tuples enforces `body.CLAIMED_ROOT ==
/// BBH.LEFT@row2`, i.e., the body witness's computed root equals the
/// `body_root` leaf inside the BBH merkleization.
pub fn make_body_pair_to_bbh_pair_linkage_descriptor(
    body_pair_layer_index: usize,
    bbh_pair_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::beacon_block_header_pair_air as bbh;

    let mut a_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { a_columns.push(COL_CLAIMED_ROOT_OFFSET + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { b_columns.push(bbh::COL_LEFT_OFFSET + k); }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "body_pair_root_to_bbh_pair_leaf_v1".into(),
        a_layer_index: body_pair_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_ROOT_BOUND_AT),
        b_layer_index: bbh_pair_layer_index,
        b_columns,
        b_selector_column: Some(bbh::COL_IS_PAIR2),
    }
}

/// Cross-AIR LogUp linkage: payload-pair's CLAIMED_ROOT (the
/// computed payload root) ↔ body-pair's RIGHT@invocation4 (= body
/// field 9 = `execution_payload_header_root`).
///
/// **A side (payload-pair)**: 32-byte CLAIMED_ROOT tuple gated by
/// payload-pair's `IS_ROOT_BOUND_AT` (=1 only at payload's root
/// invocation row 19 → exactly 1 entry).
///
/// **B side (body-pair)**: 32-byte RIGHT tuple gated by body-pair's
/// `COL_IS_BODY_FIELD_9_BOUND_AT` (=1 only at body invocation row 4
/// → exactly 1 entry).
///
/// Multiset equality on 1-vs-1 tuples enforces `payload.computed_root
/// == body.field_9_leaf`, i.e., the payload witness's HTR root
/// matches the body merkleization's `execution_payload_header_root`
/// field.
pub fn make_payload_pair_to_body_pair_linkage_descriptor(
    payload_pair_layer_index: usize,
    body_pair_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::execution_payload_pair_air as payload;

    let mut a_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { a_columns.push(payload::COL_CLAIMED_ROOT_OFFSET + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { b_columns.push(COL_RIGHT_OFFSET + k); }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "payload_pair_root_to_body_pair_field_9_v1".into(),
        a_layer_index: payload_pair_layer_index,
        a_columns,
        a_selector_column: Some(payload::COL_IS_ROOT_BOUND_AT),
        b_layer_index: body_pair_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_BODY_FIELD_9_BOUND_AT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon_block_body::BeaconBlockBody;
    use crate::execution_payload::ExecutionPayloadHeader;

    fn sample_body() -> BeaconBlockBody {
        let mut p = ExecutionPayloadHeader::default();
        p.block_hash = [0xab; 32];
        BeaconBlockBody {
            graffiti: [0x47; 32],
            attestations_root: [0xee; 32],
            execution_payload_header: p,
            ..Default::default()
        }
    }

    #[test]
    fn trace_builder_populates_12_real_rows() {
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // 12 invocations padded to nearest_power_of_two minimum 16.
        assert_eq!(trace.num_rows, 12);
        assert_eq!(trace.padded_size, 16);
        for r in 0..12 {
            assert_eq!(trace.columns[COL_IS_REAL].evaluations[r].to_u64(), 1);
        }
        for r in 12..16 {
            assert!(trace.columns[COL_IS_REAL].evaluations[r].is_zero());
        }
    }

    #[test]
    fn pair_rows_match_witness_invocations() {
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
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
    fn is_real_binary_zero_on_honest_witness() {
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BeaconBlockBodyPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for v in &res[0] {
            assert!(v.is_zero());
        }
    }

    #[test]
    fn is_real_binary_fires_on_tampered_selector() {
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(7, CurveType::Bls48581);
        let cs = BeaconBlockBodyPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[0][0].is_zero());
    }

    #[test]
    fn claimed_root_populated_and_constant() {
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let root = w.root;
        for r in 0..trace.num_rows {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_CLAIMED_ROOT_OFFSET + k].evaluations[r].to_u64(),
                    root[k] as u64,
                    "CLAIMED_ROOT[r={},k={}]", r, k,
                );
            }
        }
        // HASH at the root invocation row equals CLAIMED_ROOT there.
        for k in 0..32 {
            assert_eq!(
                trace.columns[COL_HASH_OFFSET + k].evaluations[ROOT_INVOCATION_INDEX].to_u64(),
                trace.columns[COL_CLAIMED_ROOT_OFFSET + k].evaluations[ROOT_INVOCATION_INDEX].to_u64(),
            );
        }
    }

    #[test]
    fn is_root_bound_at_one_hot_at_root_invocation() {
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for r in 0..16 {
            let expected = if r == ROOT_INVOCATION_INDEX { 1 } else { 0 };
            assert_eq!(
                trace.columns[COL_IS_ROOT_BOUND_AT].evaluations[r].to_u64(), expected,
                "IS_ROOT_BOUND_AT[{}]", r,
            );
        }
    }

    #[test]
    fn hash_eq_claimed_root_fires_on_tampered_claimed_root() {
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_CLAIMED_ROOT_OFFSET][ROOT_INVOCATION_INDEX] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = BeaconBlockBodyPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 2 = hash_eq_claimed_root_at_root_invocation.
        assert!(!res[2][ROOT_INVOCATION_INDEX].is_zero());
    }

    #[test]
    fn is_body_field_9_bound_at_one_hot_at_invocation_4() {
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for r in 0..16 {
            let expected = if r == BODY_FIELD_9_RIGHT_INVOCATION { 1 } else { 0 };
            assert_eq!(
                trace.columns[COL_IS_BODY_FIELD_9_BOUND_AT].evaluations[r].to_u64(),
                expected,
                "IS_BODY_FIELD_9_BOUND_AT[{}]", r,
            );
        }
        // RIGHT at invocation 4 must equal body's field_root_9 (= payload root).
        for k in 0..32 {
            assert_eq!(
                trace.columns[COL_RIGHT_OFFSET + k].evaluations[BODY_FIELD_9_RIGHT_INVOCATION].to_u64(),
                w.field_roots[9][k] as u64,
                "RIGHT@invocation4 byte {} should equal body field_root_9", k,
            );
        }
    }

    #[test]
    fn payload_to_body_linkage_descriptor_shape() {
        let d = make_payload_pair_to_body_pair_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 32);
        assert_eq!(d.b_columns.len(), 32);
        assert_eq!(d.label, "payload_pair_root_to_body_pair_field_9_v1");
        use crate::execution_payload_pair_air as payload;
        assert_eq!(d.a_selector_column, Some(payload::COL_IS_ROOT_BOUND_AT));
        assert_eq!(d.b_selector_column, Some(COL_IS_BODY_FIELD_9_BOUND_AT));
        assert_eq!(d.a_columns[0], payload::COL_CLAIMED_ROOT_OFFSET);
        assert_eq!(d.b_columns[0], COL_RIGHT_OFFSET);
    }

    /// Composition pin: body field_root_9 equals payload's HTR root,
    /// validated by independently building both witnesses.
    #[test]
    fn body_field_9_equals_payload_witness_root() {
        use crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness;
        let body = sample_body();
        let body_w = BeaconBlockBodyHtrWitness::from_body(body.clone());
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(
            body.execution_payload_header.clone(),
        );
        assert_eq!(body_w.field_roots[9], payload_w.root);
    }

    #[test]
    fn bbh_linkage_descriptor_shape() {
        let d = make_body_pair_to_bbh_pair_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 32);
        assert_eq!(d.b_columns.len(), 32);
        assert_eq!(d.label, "body_pair_root_to_bbh_pair_leaf_v1");
        assert_eq!(d.a_selector_column, Some(COL_IS_ROOT_BOUND_AT));
        // Body's CLAIMED_ROOT must match the body_root leaf at BBH's
        // pair invocation row 2 (= BBH leaf 4).
        use crate::beacon_block_header_pair_air as bbh;
        assert_eq!(d.b_selector_column, Some(bbh::COL_IS_PAIR2));
        assert_eq!(d.a_columns[0], COL_CLAIMED_ROOT_OFFSET);
        assert_eq!(d.b_columns[0], bbh::COL_LEFT_OFFSET);
    }

    #[test]
    fn linkage_descriptor_shape() {
        let d = make_body_pair_to_sha256_extract_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 96);
        assert_eq!(d.b_columns.len(), 96);
        assert_eq!(d.label, "beacon_block_body_pair_sha256_extract_v1");
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(crate::sha256_extract::COL_IS_REAL));
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
        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = BeaconBlockBodyPairConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }

    /// 2-AIR joint_prove cap-stone for step 2f: payload-pair AIR ↔
    /// body-pair AIR via the 32-byte CLAIMED_ROOT/RIGHT@4 linkage.
    /// Validates `make_payload_pair_to_body_pair_linkage_descriptor`
    /// end-to-end: payload's computed root must equal body's field-9
    /// leaf, enforced by multiset equality on 1-vs-1 tuples.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with 1 32-col linkage \
                (~3-5 min release)"]
    fn joint_prove_payload_pair_to_body_pair() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness;
        use crate::execution_payload_pair_air as payload;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let body = sample_body();
        let body_w = BeaconBlockBodyHtrWitness::from_body(body.clone());
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(
            body.execution_payload_header.clone(),
        );

        // Payload-pair (layer 0).
        let p_trace = payload::build_trace_polynomials(&payload_w, curve);
        let p_omega = scheme.domain_generator(p_trace.padded_size);
        let p_cs = payload::ExecutionPayloadHeaderPairConstraintSystem::new(p_trace.num_rows)
            .with_omega_and_domain(p_omega, p_trace.padded_size);

        // Body-pair (layer 1).
        let b_trace = build_trace_polynomials(&body_w, curve);
        let b_omega = scheme.domain_generator(b_trace.padded_size);
        let b_cs = BeaconBlockBodyPairConstraintSystem::new(b_trace.num_rows)
            .with_omega_and_domain(b_omega, b_trace.padded_size);

        let linkage = make_payload_pair_to_body_pair_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&p_trace, &p_cs), (&b_trace, &b_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("payload-pair ↔ body-pair joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&p_cs, &b_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
            "joint_verify must accept honest payload→body composition",
        );
    }

    /// 2-AIR joint_prove cap-stone for step 2e: body-pair AIR ↔
    /// BBH-pair AIR via the 32-byte CLAIMED_ROOT/LEFT@row2 linkage.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with 1 32-col linkage \
                (~3-5 min release)"]
    fn joint_prove_body_pair_to_bbh_pair() {
        use crate::beacon::BeaconBlockHeader;
        use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;
        use crate::beacon_block_header_pair_air as bbh;
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let body = sample_body();
        let body_w = BeaconBlockBodyHtrWitness::from_body(body.clone());
        let bbh = BeaconBlockHeader {
            slot: 12345,
            proposer_index: 42,
            parent_root: [0xaa; 32],
            state_root: [0xbb; 32],
            body_root: body.hash_tree_root(),
        };
        let bbh_w = BeaconBlockHeaderHtrWitness::from_header(bbh);

        // Body-pair (layer 0).
        let b_trace = build_trace_polynomials(&body_w, curve);
        let b_omega = scheme.domain_generator(b_trace.padded_size);
        let b_cs = BeaconBlockBodyPairConstraintSystem::new(b_trace.num_rows)
            .with_omega_and_domain(b_omega, b_trace.padded_size);

        // BBH-pair (layer 1).
        let h_trace = bbh::build_trace_polynomials(&bbh_w, curve);
        let h_omega = scheme.domain_generator(h_trace.padded_size);
        let h_cs = bbh::BeaconBlockHeaderPairConstraintSystem::new(h_trace.num_rows)
            .with_omega_and_domain(h_omega, h_trace.padded_size);

        let linkage = make_body_pair_to_bbh_pair_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&b_trace, &b_cs), (&h_trace, &h_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("body-pair ↔ BBH-pair joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&b_cs, &h_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
        );
    }

    /// 2-AIR joint_prove cap-stone for step 1: body-pair AIR ↔
    /// Sha256Extract via the 96-byte tuple linkage.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with 1 96-col linkage \
                (~5-10 min release)"]
    fn joint_prove_body_pair_to_sha256_extract() {
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

        let w = BeaconBlockBodyHtrWitness::from_body(sample_body());

        let body_trace = build_trace_polynomials(&w, curve);
        let body_omega = scheme.domain_generator(body_trace.padded_size);
        let body_cs = BeaconBlockBodyPairConstraintSystem::new(body_trace.num_rows)
            .with_omega_and_domain(body_omega, body_trace.padded_size);

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

        let linkage = make_body_pair_to_sha256_extract_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&body_trace, &body_cs), (&se_trace, &se_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("body-pair ↔ Sha256Extract joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&body_cs, &se_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
        );
    }
}
