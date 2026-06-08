//! ExecutionPayloadHeader pair-invocation AIR — Phase C↔B bridge step 3 step 1.
//!
//! Per-row exposes one `sha256_pair(left, right) -> hash` invocation
//! drawn from a [`ExecutionPayloadHeaderHtrWitness`]. Companion to
//! step 0's [`crate::execution_payload_air`] witness builder; mirrors
//! the architecture of [`crate::beacon_block_header_pair_air`] and
//! [`crate::beacon_block_body_pair_air`].
//!
//! Each row carries 96 byte columns (`LEFT[32] || RIGHT[32] || HASH[32]`)
//! plus `IS_REAL`. The cross-AIR LogUp descriptor
//! [`make_payload_pair_to_sha256_extract_linkage_descriptor`] binds
//! each row's `(left||right, hash)` tuple to a real
//! [`crate::sha256_extract`] invocation, transitively pulling in the
//! bit-level SHA-256 binding.
//!
//! # Soundness scope (step 1)
//!
//! - Proves: each row's `(input_64, output_32)` matches a row in
//!   Sha256Extract.
//! - Does NOT yet prove (deferred to step 2+):
//!   - Layer chaining across the 5 layers (17→9→5→3→2→1).
//!   - Leaf binding to the 17 ExecutionPayloadHeader field roots (in
//!     particular field index 12 = block_hash — the Layer B bridge).
//!   - ZERO_CHUNK pin at each layer's odd-tail right (uses ZH(d) for
//!     layer d).
//!   - CLAIMED_ROOT + CLAIMED_BLOCK_HASH output columns for downstream
//!     consumers (BeaconBlockBody pair AIR + Layer B BlockHeader AIR).

use crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness;
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
//   CLAIMED_ROOT[0..32] — payload's computed root, replicated across
//     every real row. Bridge target for the body-pair AIR's field
//     index 9 (= payload root).
//   IS_ROOT_BOUND_AT — one-hot row selector =1 only at the root
//     invocation (index 19, the final layer-4 output).
//   CLAIMED_BLOCK_HASH[0..32] — payload's block_hash (field index 12),
//     replicated across every real row. Bridge target for Layer B's
//     BlockHeader.block_hash.
//   IS_BLOCK_HASH_BOUND_AT — one-hot row selector =1 only at the row
//     where block_hash is the LEFT operand of a pair invocation
//     (invocation index 6, the layer-0 pair (12, 13)).
pub const COL_CLAIMED_ROOT_OFFSET: usize = 97;             // 97..129
pub const COL_IS_ROOT_BOUND_AT: usize = 129;
pub const COL_CLAIMED_BLOCK_HASH_OFFSET: usize = 130;      // 130..162
pub const COL_IS_BLOCK_HASH_BOUND_AT: usize = 162;
// Sub-tree binding outputs (logs_bloom + extra_data) — see
// crates/zkp/src/logs_bloom_air.rs and crates/zkp/src/extra_data_air.rs.
pub const COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET: usize = 163;  // 163..195
pub const COL_IS_LOGS_BLOOM_LEAF_BOUND_AT: usize = 195;
pub const COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET: usize = 196;  // 196..228
pub const COL_IS_EXTRA_DATA_LEAF_BOUND_AT: usize = 228;

// Step 2a: leaf binding for all 17 payload fields. Per layer-0 row
// (rows 0..8, 9 pairs), claimed_left/right hold the field roots.
pub const COL_CLAIMED_LEFT_OFFSET: usize = 229;   // 229..261
pub const COL_CLAIMED_RIGHT_OFFSET: usize = 261;  // 261..293
pub const COL_IS_LAYER_0: usize = 293;

// Chain binding selectors + ZH pins for the 5-layer payload tree.
pub const COL_IS_PAIR8: usize = 294;   // layer 0 last pair (ZH(0) pin)
pub const COL_IS_PAIR13: usize = 295;  // layer 1 last pair (ZH(1) pin)
pub const COL_IS_PAIR16: usize = 296;  // layer 2 last pair (ZH(2) pin)
pub const COL_IS_PAIR18: usize = 297;  // layer 3 last pair (ZH(3) pin + offset-1 chain)

// ZH constant columns (4 depths × 32 bytes = 128 cols).
pub const COL_ZH0_OFFSET: usize = 298;   // 298..330
pub const COL_ZH1_OFFSET: usize = 330;   // 330..362
pub const COL_ZH2_OFFSET: usize = 362;   // 362..394
pub const COL_ZH3_OFFSET: usize = 394;   // 394..426

// Per-row selectors for chain binding (12 new, for rows without existing selectors).
pub const COL_IS_ROW0: usize = 426;
pub const COL_IS_ROW1: usize = 427;
pub const COL_IS_ROW3: usize = 428;
pub const COL_IS_ROW4: usize = 429;
pub const COL_IS_ROW7: usize = 430;
pub const COL_IS_ROW9: usize = 431;
pub const COL_IS_ROW10: usize = 432;
pub const COL_IS_ROW11: usize = 433;
pub const COL_IS_ROW12: usize = 434;
pub const COL_IS_ROW14: usize = 435;
pub const COL_IS_ROW15: usize = 436;
pub const COL_IS_ROW17: usize = 437;

// 15 relay column groups (15 × 32 = 480 cols).
pub const COL_RELAY_OFFSET: usize = 438;
pub const NUM_RELAYS: usize = 15;
pub const NUM_COLUMNS: usize = COL_RELAY_OFFSET + NUM_RELAYS * 32; // 918

/// Invocation index of the payload's root. Payload has 20 invocations
/// across 5 layers (9+5+3+2+1). Index 19 is the root.
pub const ROOT_INVOCATION_INDEX: usize = 19;

/// Invocation index where `block_hash` (payload field 12) is the LEFT
/// operand. In layer 0, fields are paired (0,1), (2,3), ..., (12,13),
/// (14,15), (16, ZH(0)). The pair (12,13) is the 7th — invocation
/// index 6.
pub const BLOCK_HASH_LEFT_INVOCATION: usize = 6;

/// Invocation index where `logs_bloom` (payload field 4) is the LEFT
/// operand. Layer-0 pair (4, 5) is invocation index 2.
pub const LOGS_BLOOM_LEFT_INVOCATION: usize = 2;

/// Invocation index where `extra_data` (payload field 10) is the LEFT
/// operand. Layer-0 pair (10, 11) is invocation index 5.
pub const EXTRA_DATA_LEFT_INVOCATION: usize = 5;

// Row-locals:
//   0: is_real binary
//   1: is_root_bound_at binary
//   2: is_root_bound_at * Σ β^k * (HASH[k] - CLAIMED_ROOT[k])
//   3: is_block_hash_bound_at binary
//   4: is_block_hash_bound_at * Σ β^k * (LEFT[k] - CLAIMED_BLOCK_HASH[k])
//   5: is_logs_bloom_leaf_bound_at binary
//   6: is_logs_bloom_leaf_bound_at * Σ β^k * (LEFT[k] - CLAIMED_LOGS_BLOOM_ROOT[k])
//   7: is_extra_data_leaf_bound_at binary
//   8: is_extra_data_leaf_bound_at * Σ β^k * (LEFT[k] - CLAIMED_EXTRA_DATA_ROOT[k])
//   9: is_layer_0 binary
//  10: is_layer_0 * Σ β^k * (LEFT[k] - CLAIMED_LEFT[k]) (leaf left)
//  11: is_layer_0 * Σ β^k * (RIGHT[k] - CLAIMED_RIGHT[k]) (leaf right)
//  12..15: is_pair8/13/16/18 binary (4)
//  16..19: ZH pins at rows 8/13/16/18
//  20..31: 12 binary checks for new per-row selectors
//  32..61: 15 pickup + 15 consumption for relays
pub const NUM_ROW_CONSTRAINTS: usize = 62;

// Shifted (cross-row constancy of the 4 claimed-output columns,
// gated by IS_REAL[r] * IS_REAL[r+1]):
//   0: claimed_root_constancy             — β-RLC over 32 bytes
//   1: claimed_block_hash_constancy       — β-RLC over 32 bytes
//   2: claimed_logs_bloom_root_constancy  — β-RLC over 32 bytes
//   3: claimed_extra_data_root_constancy  — β-RLC over 32 bytes
// + offset-1 chain + 15 relay constancy bodies
pub const NUM_SHIFTED: usize = 5 + NUM_RELAYS; // 20

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ExecutionPayloadHeaderHtrWitness,
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

    // Step 2e: CLAIMED_ROOT (= payload's computed root from the root
    // invocation's hash) replicated across every real row.
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

    // Step 2e: CLAIMED_BLOCK_HASH (= payload's block_hash, the LEFT of
    // pair invocation at BLOCK_HASH_LEFT_INVOCATION).
    if num_rows > BLOCK_HASH_LEFT_INVOCATION {
        let block_hash_bytes = invocations[BLOCK_HASH_LEFT_INVOCATION].left;
        for k in 0..32 {
            let v = Scalar::from_u64(block_hash_bytes[k] as u64, curve);
            for r in 0..num_rows {
                columns[COL_CLAIMED_BLOCK_HASH_OFFSET + k][r] = v.clone();
            }
        }
        columns[COL_IS_BLOCK_HASH_BOUND_AT][BLOCK_HASH_LEFT_INVOCATION] = one.clone();
    }

    // Sub-tree binding: CLAIMED_LOGS_BLOOM_ROOT (= LEFT of pair
    // invocation at LOGS_BLOOM_LEFT_INVOCATION).
    if num_rows > LOGS_BLOOM_LEFT_INVOCATION {
        let bytes = invocations[LOGS_BLOOM_LEFT_INVOCATION].left;
        for k in 0..32 {
            let v = Scalar::from_u64(bytes[k] as u64, curve);
            for r in 0..num_rows {
                columns[COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET + k][r] = v.clone();
            }
        }
        columns[COL_IS_LOGS_BLOOM_LEAF_BOUND_AT][LOGS_BLOOM_LEFT_INVOCATION] = one.clone();
    }

    // Step 2a: leaf binding for layer 0 (rows 0..8). 9 layer-0 pairs.
    let num_layer0_pairs = 9usize.min(num_rows);
    for r in 0..num_layer0_pairs {
        columns[COL_IS_LAYER_0][r] = one.clone();
        for k in 0..32 {
            columns[COL_CLAIMED_LEFT_OFFSET + k][r] =
                Scalar::from_u64(invocations[r].left[k] as u64, curve);
            columns[COL_CLAIMED_RIGHT_OFFSET + k][r] =
                Scalar::from_u64(invocations[r].right[k] as u64, curve);
        }
    }

    // Chain binding selectors (ZH pin + per-row).
    for (row, col) in [
        (0, COL_IS_ROW0), (1, COL_IS_ROW1), (3, COL_IS_ROW3), (4, COL_IS_ROW4),
        (7, COL_IS_ROW7), (8, COL_IS_PAIR8), (9, COL_IS_ROW9), (10, COL_IS_ROW10),
        (11, COL_IS_ROW11), (12, COL_IS_ROW12), (13, COL_IS_PAIR13),
        (14, COL_IS_ROW14), (15, COL_IS_ROW15), (16, COL_IS_PAIR16),
        (17, COL_IS_ROW17), (18, COL_IS_PAIR18),
    ] {
        if num_rows > row { columns[col][row] = one.clone(); }
    }

    // Populate ZH constant columns across all real rows.
    let zh = |depth: u32| -> [u8; 32] {
        let mut z = crate::ssz::ZERO_CHUNK;
        for _ in 0..depth { z = crate::sha256::sha256_pair(&z, &z); }
        z
    };
    for (col_off, d) in [(COL_ZH0_OFFSET, 0u32), (COL_ZH1_OFFSET, 1), (COL_ZH2_OFFSET, 2), (COL_ZH3_OFFSET, 3)] {
        let z = zh(d);
        for k in 0..32 {
            let v = Scalar::from_u64(z[k] as u64, curve);
            for r in 0..num_rows { columns[col_off + k][r] = v.clone(); }
        }
    }

    // 15 relay columns for chain bindings.
    // (producer_row, relay_index, start_row, end_row_exclusive)
    let relay_specs: &[(usize, usize, usize, usize)] = &[
        // Layer 0→1 (8 relays):
        (0, 0, 0, 10),   // Row 0→Row 9 LEFT (offset -9)
        (1, 1, 1, 10),   // Row 1→Row 9 RIGHT (offset -8)
        (2, 2, 2, 11),   // Row 2→Row 10 LEFT (offset -8)
        (3, 3, 3, 11),   // Row 3→Row 10 RIGHT (offset -7)
        (4, 4, 4, 12),   // Row 4→Row 11 LEFT (offset -7)
        (5, 5, 5, 12),   // Row 5→Row 11 RIGHT (offset -6)
        (6, 6, 6, 13),   // Row 6→Row 12 LEFT (offset -6)
        (7, 7, 7, 13),   // Row 7→Row 12 RIGHT (offset -5)
        // Layer 1→2 (4 relays):
        (9, 8, 9, 15),   // Row 9→Row 14 LEFT (offset -5)
        (10, 9, 10, 15),  // Row 10→Row 14 RIGHT (offset -4)
        (11, 10, 11, 16), // Row 11→Row 15 LEFT (offset -4)
        (12, 11, 12, 16), // Row 12→Row 15 RIGHT (offset -3)
        // Layer 2→3 (2 relays):
        (14, 12, 14, 18), // Row 14→Row 17 LEFT (offset -3)
        (15, 13, 15, 18), // Row 15→Row 17 RIGHT (offset -2)
        // Layer 3→Root (1 relay):
        (17, 14, 17, 20), // Row 17→Row 19 LEFT (offset -2)
    ];
    for &(prod, idx, start, end) in relay_specs {
        if num_rows > prod {
            let h = invocations[prod].hash;
            let col_off = COL_RELAY_OFFSET + idx * 32;
            for k in 0..32 {
                let v = Scalar::from_u64(h[k] as u64, curve);
                for r in start..num_rows.min(end) {
                    columns[col_off + k][r] = v.clone();
                }
            }
        }
    }

    // Sub-tree binding: CLAIMED_EXTRA_DATA_ROOT (= LEFT of pair
    // invocation at EXTRA_DATA_LEFT_INVOCATION).
    if num_rows > EXTRA_DATA_LEFT_INVOCATION {
        let bytes = invocations[EXTRA_DATA_LEFT_INVOCATION].left;
        for k in 0..32 {
            let v = Scalar::from_u64(bytes[k] as u64, curve);
            for r in 0..num_rows {
                columns[COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET + k][r] = v.clone();
            }
        }
        columns[COL_IS_EXTRA_DATA_LEAF_BOUND_AT][EXTRA_DATA_LEFT_INVOCATION] = one.clone();
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

pub struct ExecutionPayloadHeaderPairConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ExecutionPayloadHeaderPairConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for ExecutionPayloadHeaderPairConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_root_bound_at_binary".into(),
            "hash_eq_claimed_root_at_root_invocation".into(),
            "is_block_hash_bound_at_binary".into(),
            "left_eq_claimed_block_hash_at_pair6".into(),
            "is_logs_bloom_leaf_bound_at_binary".into(),
            "left_eq_claimed_logs_bloom_root_at_pair2".into(),
            "is_extra_data_leaf_bound_at_binary".into(),
            "left_eq_claimed_extra_data_root_at_pair5".into(),
            "is_layer_0_binary".into(),
            "leaf_left_binding_rlc".into(),
            "leaf_right_binding_rlc".into(),
            "is_pair8_binary".into(), "is_pair13_binary".into(),
            "is_pair16_binary".into(), "is_pair18_binary".into(),
            "zh0_pin_right_at_pair8".into(), "zh1_pin_right_at_pair13".into(),
            "zh2_pin_right_at_pair16".into(), "zh3_pin_right_at_pair18".into(),
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
        let mut is_root_bin = vec![Scalar::zero(curve); n];
        let mut hash_eq_root = vec![Scalar::zero(curve); n];
        let mut is_bh_bin = vec![Scalar::zero(curve); n];
        let mut left_eq_bh = vec![Scalar::zero(curve); n];
        let mut is_lb_bin = vec![Scalar::zero(curve); n];
        let mut left_eq_lb = vec![Scalar::zero(curve); n];
        let mut is_ed_bin = vec![Scalar::zero(curve); n];
        let mut left_eq_ed = vec![Scalar::zero(curve); n];
        let mut is_l0_bin = vec![Scalar::zero(curve); n];
        let mut leaf_left = vec![Scalar::zero(curve); n];
        let mut leaf_right = vec![Scalar::zero(curve); n];

        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            is_real_bin[r] = v.mul(&v.sub(&one));
            let rb = &columns[COL_IS_ROOT_BOUND_AT][r];
            is_root_bin[r] = rb.mul(&rb.sub(&one));
            let bb = &columns[COL_IS_BLOCK_HASH_BOUND_AT][r];
            is_bh_bin[r] = bb.mul(&bb.sub(&one));
            let lb = &columns[COL_IS_LOGS_BLOOM_LEAF_BOUND_AT][r];
            is_lb_bin[r] = lb.mul(&lb.sub(&one));
            let ed = &columns[COL_IS_EXTRA_DATA_LEAF_BOUND_AT][r];
            is_ed_bin[r] = ed.mul(&ed.sub(&one));
            let l0 = &columns[COL_IS_LAYER_0][r];
            is_l0_bin[r] = l0.mul(&l0.sub(&one));

            let mut acc_root = Scalar::zero(curve);
            let mut acc_bh = Scalar::zero(curve);
            let mut acc_lb = Scalar::zero(curve);
            let mut acc_ed = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let h = &columns[COL_HASH_OFFSET + k][r];
                let cr = &columns[COL_CLAIMED_ROOT_OFFSET + k][r];
                let l = &columns[COL_LEFT_OFFSET + k][r];
                let cbh = &columns[COL_CLAIMED_BLOCK_HASH_OFFSET + k][r];
                let clb = &columns[COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET + k][r];
                let ced = &columns[COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET + k][r];
                acc_root = acc_root.add(&bp.mul(&h.sub(cr)));
                acc_bh = acc_bh.add(&bp.mul(&l.sub(cbh)));
                acc_lb = acc_lb.add(&bp.mul(&l.sub(clb)));
                acc_ed = acc_ed.add(&bp.mul(&l.sub(ced)));
                bp = bp.mul(&beta_test);
            }
            hash_eq_root[r] = rb.mul(&acc_root);
            left_eq_bh[r] = bb.mul(&acc_bh);
            left_eq_lb[r] = lb.mul(&acc_lb);
            left_eq_ed[r] = ed.mul(&acc_ed);

            // Step 2a: leaf binding.
            let mut acc_ll = Scalar::zero(curve);
            let mut acc_lr = Scalar::zero(curve);
            let mut bp2 = Scalar::one(curve);
            for k in 0..32 {
                let ll = &columns[COL_LEFT_OFFSET + k][r];
                let cl = &columns[COL_CLAIMED_LEFT_OFFSET + k][r];
                let rr2 = &columns[COL_RIGHT_OFFSET + k][r];
                let cr = &columns[COL_CLAIMED_RIGHT_OFFSET + k][r];
                acc_ll = acc_ll.add(&bp2.mul(&ll.sub(cl)));
                acc_lr = acc_lr.add(&bp2.mul(&rr2.sub(cr)));
                bp2 = bp2.mul(&beta_test);
            }
            leaf_left[r] = l0.mul(&acc_ll);
            leaf_right[r] = l0.mul(&acc_lr);
        }
        let mut result = vec![
            is_real_bin, is_root_bin, hash_eq_root, is_bh_bin, left_eq_bh,
            is_lb_bin, left_eq_lb, is_ed_bin, left_eq_ed,
            is_l0_bin, leaf_left, leaf_right,
        ];
        // ZH pin constraints: 4 binary + 4 pin bodies.
        let zh_pins: &[(usize, usize)] = &[
            (COL_IS_PAIR8, COL_ZH0_OFFSET), (COL_IS_PAIR13, COL_ZH1_OFFSET),
            (COL_IS_PAIR16, COL_ZH2_OFFSET), (COL_IS_PAIR18, COL_ZH3_OFFSET),
        ];
        for &(sel_col, zh_col) in zh_pins {
            let mut bin_v = vec![Scalar::zero(curve); n];
            let mut pin_v = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let s = &columns[sel_col][r];
                bin_v[r] = s.mul(&s.sub(&one));
                let mut acc = Scalar::zero(curve); let mut bp = Scalar::one(curve);
                for k in 0..32 {
                    let rr = &columns[COL_RIGHT_OFFSET+k][r];
                    let zh = &columns[zh_col+k][r];
                    acc = acc.add(&bp.mul(&rr.sub(zh)));
                    bp = bp.mul(&beta_test);
                }
                pin_v[r] = s.mul(&acc);
            }
            result.push(bin_v);
            result.push(pin_v);
        }
        // Bulk: 12 binary + 30 pickup/consumption for 15 relays.
        let new_sels = [COL_IS_ROW0,COL_IS_ROW1,COL_IS_ROW3,COL_IS_ROW4,COL_IS_ROW7,
                        COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11,COL_IS_ROW12,
                        COL_IS_ROW14,COL_IS_ROW15,COL_IS_ROW17];
        for &sc in &new_sels {
            let mut bv = vec![Scalar::zero(curve); n];
            for r in 0..n { let s = &columns[sc][r]; bv[r] = s.mul(&s.sub(&one)); }
            result.push(bv);
        }
        // Relay pickup/consumption table.
        // (pickup_sel_col, consumption_sel_col, relay_index, is_left_consumption)
        // Reuse existing selectors: row2=IS_LOGS_BLOOM, row5=IS_EXTRA_DATA, row6=IS_BLOCK_HASH, row19=IS_ROOT_BOUND_AT
        let relay_ct: &[(usize, usize, usize, bool)] = &[
            (COL_IS_ROW0, COL_IS_ROW9, 0, true),    // Row 0→9 LEFT
            (COL_IS_ROW1, COL_IS_ROW9, 1, false),   // Row 1→9 RIGHT
            (COL_IS_LOGS_BLOOM_LEAF_BOUND_AT, COL_IS_ROW10, 2, true), // Row 2→10 LEFT
            (COL_IS_ROW3, COL_IS_ROW10, 3, false),  // Row 3→10 RIGHT
            (COL_IS_ROW4, COL_IS_ROW11, 4, true),   // Row 4→11 LEFT
            (COL_IS_EXTRA_DATA_LEAF_BOUND_AT, COL_IS_ROW11, 5, false), // Row 5→11 RIGHT
            (COL_IS_BLOCK_HASH_BOUND_AT, COL_IS_ROW12, 6, true), // Row 6→12 LEFT
            (COL_IS_ROW7, COL_IS_ROW12, 7, false),  // Row 7→12 RIGHT
            (COL_IS_ROW9, COL_IS_ROW14, 8, true),   // Row 9→14 LEFT
            (COL_IS_ROW10, COL_IS_ROW14, 9, false),  // Row 10→14 RIGHT
            (COL_IS_ROW11, COL_IS_ROW15, 10, true),  // Row 11→15 LEFT
            (COL_IS_ROW12, COL_IS_ROW15, 11, false),  // Row 12→15 RIGHT
            (COL_IS_ROW14, COL_IS_ROW17, 12, true),  // Row 14→17 LEFT
            (COL_IS_ROW15, COL_IS_ROW17, 13, false),  // Row 15→17 RIGHT
            (COL_IS_ROW17, COL_IS_ROOT_BOUND_AT, 14, true), // Row 17→19 LEFT
        ];
        for &(pk_sel, cm_sel, idx, is_left) in relay_ct {
            let rel_off = COL_RELAY_OFFSET + idx * 32;
            let mut pk_v = vec![Scalar::zero(curve); n];
            let mut cm_v = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let ps = &columns[pk_sel][r]; let cs2 = &columns[cm_sel][r];
                let mut apk = Scalar::zero(curve); let mut acm = Scalar::zero(curve);
                let mut bp = Scalar::one(curve);
                for k in 0..32 {
                    let h = &columns[COL_HASH_OFFSET+k][r]; let rl = &columns[rel_off+k][r];
                    let tgt = if is_left { &columns[COL_LEFT_OFFSET+k][r] } else { &columns[COL_RIGHT_OFFSET+k][r] };
                    apk = apk.add(&bp.mul(&rl.sub(h)));
                    acm = acm.add(&bp.mul(&tgt.sub(rl)));
                    bp = bp.mul(&beta_test);
                }
                pk_v[r] = ps.mul(&apk); cm_v[r] = cs2.mul(&acm);
            }
            result.push(pk_v); result.push(cm_v);
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
        let bb = &col_evals[COL_IS_BLOCK_HASH_BOUND_AT];
        let lb = &col_evals[COL_IS_LOGS_BLOOM_LEAF_BOUND_AT];
        let ed = &col_evals[COL_IS_EXTRA_DATA_LEAF_BOUND_AT];
        let is_real_bin = v.mul(&v.sub(&one));
        let is_root_bin = rb.mul(&rb.sub(&one));
        let is_bh_bin = bb.mul(&bb.sub(&one));
        let is_lb_bin = lb.mul(&lb.sub(&one));
        let is_ed_bin = ed.mul(&ed.sub(&one));

        let mut acc_root = Scalar::zero(curve);
        let mut acc_bh = Scalar::zero(curve);
        let mut acc_lb = Scalar::zero(curve);
        let mut acc_ed = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let h = &col_evals[COL_HASH_OFFSET + k];
            let cr = &col_evals[COL_CLAIMED_ROOT_OFFSET + k];
            let l = &col_evals[COL_LEFT_OFFSET + k];
            let cbh = &col_evals[COL_CLAIMED_BLOCK_HASH_OFFSET + k];
            let clb = &col_evals[COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET + k];
            let ced = &col_evals[COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET + k];
            acc_root = acc_root.add(&bp.mul(&h.sub(cr)));
            acc_bh = acc_bh.add(&bp.mul(&l.sub(cbh)));
            acc_lb = acc_lb.add(&bp.mul(&l.sub(clb)));
            acc_ed = acc_ed.add(&bp.mul(&l.sub(ced)));
            bp = bp.mul(alpha);
        }
        let hash_eq_root = rb.mul(&acc_root);
        let left_eq_bh = bb.mul(&acc_bh);
        let left_eq_lb = lb.mul(&acc_lb);
        let left_eq_ed = ed.mul(&acc_ed);

        let mut total = is_real_bin;
        let mut ap = alpha.clone();
        total = total.add(&ap.mul(&is_root_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&hash_eq_root));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_bh_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&left_eq_bh));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_lb_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&left_eq_lb));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&is_ed_bin));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&left_eq_ed));

        let l0 = &col_evals[COL_IS_LAYER_0];
        let is_l0_bin = l0.mul(&l0.sub(&one));
        let mut acc_ll = Scalar::zero(curve); let mut acc_lr = Scalar::zero(curve);
        let mut bp2 = Scalar::one(curve);
        for k in 0..32 {
            let ll = &col_evals[COL_LEFT_OFFSET + k]; let cl = &col_evals[COL_CLAIMED_LEFT_OFFSET + k];
            let rr2 = &col_evals[COL_RIGHT_OFFSET + k]; let cr = &col_evals[COL_CLAIMED_RIGHT_OFFSET + k];
            acc_ll = acc_ll.add(&bp2.mul(&ll.sub(cl))); acc_lr = acc_lr.add(&bp2.mul(&rr2.sub(cr)));
            bp2 = bp2.mul(alpha);
        }
        ap = ap.mul(alpha); total = total.add(&ap.mul(&is_l0_bin));
        ap = ap.mul(alpha); total = total.add(&ap.mul(&l0.mul(&acc_ll)));
        ap = ap.mul(alpha); total = total.add(&ap.mul(&l0.mul(&acc_lr)));
        // ZH pin constraints.
        let zh_pins: &[(usize, usize)] = &[
            (COL_IS_PAIR8, COL_ZH0_OFFSET), (COL_IS_PAIR13, COL_ZH1_OFFSET),
            (COL_IS_PAIR16, COL_ZH2_OFFSET), (COL_IS_PAIR18, COL_ZH3_OFFSET),
        ];
        for &(sel_col, zh_col) in zh_pins {
            let s = &col_evals[sel_col];
            ap = ap.mul(alpha); total = total.add(&ap.mul(&s.mul(&s.sub(&one))));
            let mut acc_zh = Scalar::zero(curve); let mut bp_zh = Scalar::one(curve);
            for k in 0..32 {
                let rr = &col_evals[COL_RIGHT_OFFSET+k]; let zh = &col_evals[zh_col+k];
                acc_zh = acc_zh.add(&bp_zh.mul(&rr.sub(zh))); bp_zh = bp_zh.mul(alpha);
            }
            ap = ap.mul(alpha); total = total.add(&ap.mul(&s.mul(&acc_zh)));
        }
        // Bulk relay constraints (evaluate_at_point).
        let new_sels = [COL_IS_ROW0,COL_IS_ROW1,COL_IS_ROW3,COL_IS_ROW4,COL_IS_ROW7,
                        COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11,COL_IS_ROW12,
                        COL_IS_ROW14,COL_IS_ROW15,COL_IS_ROW17];
        for &sc in &new_sels {
            let s = &col_evals[sc];
            ap = ap.mul(alpha); total = total.add(&ap.mul(&s.mul(&s.sub(&one))));
        }
        let relay_ct: &[(usize, usize, usize, bool)] = &[
            (COL_IS_ROW0, COL_IS_ROW9, 0, true), (COL_IS_ROW1, COL_IS_ROW9, 1, false),
            (COL_IS_LOGS_BLOOM_LEAF_BOUND_AT, COL_IS_ROW10, 2, true), (COL_IS_ROW3, COL_IS_ROW10, 3, false),
            (COL_IS_ROW4, COL_IS_ROW11, 4, true), (COL_IS_EXTRA_DATA_LEAF_BOUND_AT, COL_IS_ROW11, 5, false),
            (COL_IS_BLOCK_HASH_BOUND_AT, COL_IS_ROW12, 6, true), (COL_IS_ROW7, COL_IS_ROW12, 7, false),
            (COL_IS_ROW9, COL_IS_ROW14, 8, true), (COL_IS_ROW10, COL_IS_ROW14, 9, false),
            (COL_IS_ROW11, COL_IS_ROW15, 10, true), (COL_IS_ROW12, COL_IS_ROW15, 11, false),
            (COL_IS_ROW14, COL_IS_ROW17, 12, true), (COL_IS_ROW15, COL_IS_ROW17, 13, false),
            (COL_IS_ROW17, COL_IS_ROOT_BOUND_AT, 14, true),
        ];
        for &(pk_sel, cm_sel, idx, is_left) in relay_ct {
            let ps = &col_evals[pk_sel]; let cs2 = &col_evals[cm_sel];
            let rel_off = COL_RELAY_OFFSET + idx * 32;
            let mut apk = Scalar::zero(curve); let mut acm = Scalar::zero(curve); let mut bp3 = Scalar::one(curve);
            for k in 0..32 {
                let h = &col_evals[COL_HASH_OFFSET+k]; let rl = &col_evals[rel_off+k];
                let tgt = if is_left { &col_evals[COL_LEFT_OFFSET+k] } else { &col_evals[COL_RIGHT_OFFSET+k] };
                apk = apk.add(&bp3.mul(&rl.sub(h))); acm = acm.add(&bp3.mul(&tgt.sub(rl)));
                bp3 = bp3.mul(alpha);
            }
            ap = ap.mul(alpha); total = total.add(&ap.mul(&ps.mul(&apk)));
            ap = ap.mul(alpha); total = total.add(&ap.mul(&cs2.mul(&acm)));
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
        let is_root_bin = poly_mul(rb, &rb_m1, curve);

        let bb = &col_coeffs[COL_IS_BLOCK_HASH_BOUND_AT];
        let bb_m1 = poly_sub(bb, &one_poly, curve);
        let is_bh_bin = poly_mul(bb, &bb_m1, curve);

        let lb = &col_coeffs[COL_IS_LOGS_BLOOM_LEAF_BOUND_AT];
        let lb_m1 = poly_sub(lb, &one_poly, curve);
        let is_lb_bin = poly_mul(lb, &lb_m1, curve);

        let ed = &col_coeffs[COL_IS_EXTRA_DATA_LEAF_BOUND_AT];
        let ed_m1 = poly_sub(ed, &one_poly, curve);
        let is_ed_bin = poly_mul(ed, &ed_m1, curve);

        let mut acc_root = vec![Scalar::zero(curve)];
        let mut acc_bh = vec![Scalar::zero(curve)];
        let mut acc_lb = vec![Scalar::zero(curve)];
        let mut acc_ed = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..32 {
            let h = &col_coeffs[COL_HASH_OFFSET + k];
            let cr = &col_coeffs[COL_CLAIMED_ROOT_OFFSET + k];
            let l = &col_coeffs[COL_LEFT_OFFSET + k];
            let cbh = &col_coeffs[COL_CLAIMED_BLOCK_HASH_OFFSET + k];
            let clb = &col_coeffs[COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET + k];
            let ced = &col_coeffs[COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET + k];
            let d_root = poly_sub(h, cr, curve);
            let d_bh = poly_sub(l, cbh, curve);
            let d_lb = poly_sub(l, clb, curve);
            let d_ed = poly_sub(l, ced, curve);
            acc_root = poly_add(&acc_root, &poly_scalar_mul(&d_root, &bp), curve);
            acc_bh = poly_add(&acc_bh, &poly_scalar_mul(&d_bh, &bp), curve);
            acc_lb = poly_add(&acc_lb, &poly_scalar_mul(&d_lb, &bp), curve);
            acc_ed = poly_add(&acc_ed, &poly_scalar_mul(&d_ed, &bp), curve);
            bp = bp.mul(alpha);
        }
        let hash_eq_root = poly_mul(rb, &acc_root, curve);
        let left_eq_bh = poly_mul(bb, &acc_bh, curve);
        let left_eq_lb = poly_mul(lb, &acc_lb, curve);
        let left_eq_ed = poly_mul(ed, &acc_ed, curve);

        let mut total = is_real_bin;
        let mut ap = alpha.clone();
        total = poly_add(&total, &poly_scalar_mul(&is_root_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&hash_eq_root, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_bh_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&left_eq_bh, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_lb_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&left_eq_lb, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&is_ed_bin, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&left_eq_ed, &ap), curve);

        let l0p = &col_coeffs[COL_IS_LAYER_0];
        let l0p_m1 = poly_sub(l0p, &one_poly, curve);
        let is_l0_bin = poly_mul(l0p, &l0p_m1, curve);
        let mut acc_ll = vec![Scalar::zero(curve)]; let mut acc_lr = vec![Scalar::zero(curve)];
        let mut bp2 = Scalar::one(curve);
        for k in 0..32 {
            let ll = &col_coeffs[COL_LEFT_OFFSET + k]; let cl = &col_coeffs[COL_CLAIMED_LEFT_OFFSET + k];
            let rr2 = &col_coeffs[COL_RIGHT_OFFSET + k]; let cr = &col_coeffs[COL_CLAIMED_RIGHT_OFFSET + k];
            acc_ll = poly_add(&acc_ll, &poly_scalar_mul(&poly_sub(ll, cl, curve), &bp2), curve);
            acc_lr = poly_add(&acc_lr, &poly_scalar_mul(&poly_sub(rr2, cr, curve), &bp2), curve);
            bp2 = bp2.mul(alpha);
        }
        let leaf_left = poly_mul(l0p, &acc_ll, curve);
        let leaf_right = poly_mul(l0p, &acc_lr, curve);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&is_l0_bin, &ap), curve);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&leaf_left, &ap), curve);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&leaf_right, &ap), curve);
        // ZH pin constraints in coeff form.
        let zh_pins: &[(usize, usize)] = &[
            (COL_IS_PAIR8, COL_ZH0_OFFSET), (COL_IS_PAIR13, COL_ZH1_OFFSET),
            (COL_IS_PAIR16, COL_ZH2_OFFSET), (COL_IS_PAIR18, COL_ZH3_OFFSET),
        ];
        for &(sel_col, zh_col) in zh_pins {
            let sp = &col_coeffs[sel_col];
            let b = poly_mul(sp, &poly_sub(sp, &one_poly, curve), curve);
            ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&b, &ap), curve);
            let mut acc_zh = vec![Scalar::zero(curve)]; let mut bp_zh = Scalar::one(curve);
            for k in 0..32 {
                let rr = &col_coeffs[COL_RIGHT_OFFSET+k]; let zh = &col_coeffs[zh_col+k];
                acc_zh = poly_add(&acc_zh, &poly_scalar_mul(&poly_sub(rr, zh, curve), &bp_zh), curve);
                bp_zh = bp_zh.mul(alpha);
            }
            let pin = poly_mul(sp, &acc_zh, curve);
            ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&pin, &ap), curve);
        }
        // Bulk relay constraints (build_constraint_polynomial).
        let new_sels = [COL_IS_ROW0,COL_IS_ROW1,COL_IS_ROW3,COL_IS_ROW4,COL_IS_ROW7,
                        COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11,COL_IS_ROW12,
                        COL_IS_ROW14,COL_IS_ROW15,COL_IS_ROW17];
        for &sc in &new_sels {
            let sp = &col_coeffs[sc]; let b = poly_mul(sp, &poly_sub(sp, &one_poly, curve), curve);
            ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&b, &ap), curve);
        }
        let relay_ct: &[(usize, usize, usize, bool)] = &[
            (COL_IS_ROW0, COL_IS_ROW9, 0, true), (COL_IS_ROW1, COL_IS_ROW9, 1, false),
            (COL_IS_LOGS_BLOOM_LEAF_BOUND_AT, COL_IS_ROW10, 2, true), (COL_IS_ROW3, COL_IS_ROW10, 3, false),
            (COL_IS_ROW4, COL_IS_ROW11, 4, true), (COL_IS_EXTRA_DATA_LEAF_BOUND_AT, COL_IS_ROW11, 5, false),
            (COL_IS_BLOCK_HASH_BOUND_AT, COL_IS_ROW12, 6, true), (COL_IS_ROW7, COL_IS_ROW12, 7, false),
            (COL_IS_ROW9, COL_IS_ROW14, 8, true), (COL_IS_ROW10, COL_IS_ROW14, 9, false),
            (COL_IS_ROW11, COL_IS_ROW15, 10, true), (COL_IS_ROW12, COL_IS_ROW15, 11, false),
            (COL_IS_ROW14, COL_IS_ROW17, 12, true), (COL_IS_ROW15, COL_IS_ROW17, 13, false),
            (COL_IS_ROW17, COL_IS_ROOT_BOUND_AT, 14, true),
        ];
        for &(pk_sel, cm_sel, idx, is_left) in relay_ct {
            let psp = &col_coeffs[pk_sel]; let csp = &col_coeffs[cm_sel];
            let rel_off = COL_RELAY_OFFSET + idx * 32;
            let mut apk = vec![Scalar::zero(curve)]; let mut acm = vec![Scalar::zero(curve)]; let mut bp3 = Scalar::one(curve);
            for k in 0..32 {
                let h = &col_coeffs[COL_HASH_OFFSET+k]; let rl = &col_coeffs[rel_off+k];
                let tgt = if is_left { &col_coeffs[COL_LEFT_OFFSET+k] } else { &col_coeffs[COL_RIGHT_OFFSET+k] };
                apk = poly_add(&apk, &poly_scalar_mul(&poly_sub(rl, h, curve), &bp3), curve);
                acm = poly_add(&acm, &poly_scalar_mul(&poly_sub(tgt, rl, curve), &bp3), curve);
                bp3 = bp3.mul(alpha);
            }
            ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&poly_mul(psp, &apk, curve), &ap), curve);
            ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&poly_mul(csp, &acm, curve), &ap), curve);
        }
        total
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        let mut cols = Vec::with_capacity(1 + 32 * 4 + 1 + 32);
        cols.push(COL_IS_REAL);
        for k in 0..32 { cols.push(COL_CLAIMED_ROOT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_CLAIMED_BLOCK_HASH_OFFSET + k); }
        for k in 0..32 { cols.push(COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET + k); }
        // For offset-1 chain: IS_ROOT_BOUND_AT_NEXT + RIGHT_NEXT.
        cols.push(COL_IS_ROOT_BOUND_AT);
        for k in 0..32 { cols.push(COL_RIGHT_OFFSET + k); }
        // For 15 relay constancy bodies.
        for idx in 0..NUM_RELAYS {
            for k in 0..32 { cols.push(COL_RELAY_OFFSET + idx * 32 + k); }
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
        // 1 + 4*32 + 1 + 32 + 15*32 = 642
        if shifted_evals.len() != 1 + 32*4 + 1 + 32 + NUM_RELAYS*32 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let gating = v.mul(v_next);

        let make_body = |off_z: usize, shifted_base: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let cur = &col_evals_at_z[off_z + k];
                let nxt = &shifted_evals[shifted_base + k];
                acc = acc.add(&bp.mul(&nxt.sub(cur)));
                bp = bp.mul(alpha);
            }
            acc
        };
        let b_root = gating.mul(&make_body(COL_CLAIMED_ROOT_OFFSET, 1));
        let b_bh = gating.mul(&make_body(COL_CLAIMED_BLOCK_HASH_OFFSET, 33));
        let b_lb = gating.mul(&make_body(COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET, 65));
        let b_ed = gating.mul(&make_body(COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET, 97));

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = ap.mul(&b_root);
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&b_bh));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&b_lb));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&b_ed));
        // Shifted body 4: offset-1 chain. IS_PAIR18(r) * (RIGHT_NEXT - HASH).
        let p18 = &col_evals_at_z[COL_IS_PAIR18];
        let mut chain_body = Scalar::zero(curve);
        let mut bp_ch = Scalar::one(curve);
        for k in 0..32 {
            let hash_curr = &col_evals_at_z[COL_HASH_OFFSET + k];
            let right_next = &shifted_evals[129 + 1 + k]; // after 4×32 constancy + IS_ROOT_NEXT
            chain_body = chain_body.add(&bp_ch.mul(&right_next.sub(hash_curr)));
            bp_ch = bp_ch.mul(alpha);
        }
        let body_chain = p18.mul(&chain_body);
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_chain));
        // 15 relay constancy bodies.
        // Constancy gating: sum of per-row selectors for the propagation window.
        // Relay shifted_evals start at offset 162 (after 1+128+1+32).
        let relay_sh_base = 162;
        // (relay_index, gating_sel_cols) — sels covering rows from pickup through consumption-1.
        let constancy_specs: &[(usize, &[usize])] = &[
            (0, &[COL_IS_ROW0,COL_IS_ROW1,COL_IS_LOGS_BLOOM_LEAF_BOUND_AT,COL_IS_ROW3,COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8]),
            (1, &[COL_IS_ROW1,COL_IS_LOGS_BLOOM_LEAF_BOUND_AT,COL_IS_ROW3,COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8]),
            (2, &[COL_IS_LOGS_BLOOM_LEAF_BOUND_AT,COL_IS_ROW3,COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9]),
            (3, &[COL_IS_ROW3,COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9]),
            (4, &[COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9,COL_IS_ROW10]),
            (5, &[COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9,COL_IS_ROW10]),
            (6, &[COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11]),
            (7, &[COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11]),
            (8, &[COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11,COL_IS_ROW12,COL_IS_PAIR13]),
            (9, &[COL_IS_ROW10,COL_IS_ROW11,COL_IS_ROW12,COL_IS_PAIR13]),
            (10, &[COL_IS_ROW11,COL_IS_ROW12,COL_IS_PAIR13,COL_IS_ROW14]),
            (11, &[COL_IS_ROW12,COL_IS_PAIR13,COL_IS_ROW14]),
            (12, &[COL_IS_ROW14,COL_IS_ROW15,COL_IS_PAIR16]),
            (13, &[COL_IS_ROW15,COL_IS_PAIR16]),
            (14, &[COL_IS_ROW17,COL_IS_PAIR18]),
        ];
        for &(idx, gate_sels) in constancy_specs {
            let mut gate = Scalar::zero(curve);
            for &gs in gate_sels { gate = gate.add(&col_evals_at_z[gs]); }
            let sh_off = relay_sh_base + idx * 32;
            let rel_col = COL_RELAY_OFFSET + idx * 32;
            let mut body = Scalar::zero(curve); let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let cur = &col_evals_at_z[rel_col + k];
                let nxt = &shifted_evals[sh_off + k];
                body = body.add(&bp.mul(&nxt.sub(cur)));
                bp = bp.mul(alpha);
            }
            let rb = gate.mul(&body);
            ap = ap.mul(alpha); total = total.add(&ap.mul(&rb));
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

        let make_body = |off: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let cur = &column_coeffs[off + k];
                let nxt = poly_shift(cur, omega);
                let diff = poly_sub(&nxt, cur, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
                bp = bp.mul(alpha);
            }
            acc
        };
        let b_root = poly_mul(&gating, &make_body(COL_CLAIMED_ROOT_OFFSET), curve);
        let b_bh = poly_mul(&gating, &make_body(COL_CLAIMED_BLOCK_HASH_OFFSET), curve);
        let b_lb = poly_mul(&gating, &make_body(COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET), curve);
        let b_ed = poly_mul(&gating, &make_body(COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET), curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = poly_scalar_mul(&b_root, &ap);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&b_bh, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&b_lb, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&b_ed, &ap), curve);

        // Shifted body 4: offset-1 chain.
        let p18p = &column_coeffs[COL_IS_PAIR18];
        let mut chain_body_poly = vec![Scalar::zero(curve)];
        let mut bp_ch = Scalar::one(curve);
        for k in 0..32 {
            let hash_curr = &column_coeffs[COL_HASH_OFFSET + k];
            let right_curr = &column_coeffs[COL_RIGHT_OFFSET + k];
            let right_next = poly_shift(right_curr, omega);
            let diff = poly_sub(&right_next, hash_curr, curve);
            chain_body_poly = poly_add(&chain_body_poly, &poly_scalar_mul(&diff, &bp_ch), curve);
            bp_ch = bp_ch.mul(alpha);
        }
        let body_chain = poly_mul(p18p, &chain_body_poly, curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_chain, &ap), curve);

        // 15 relay constancy bodies in coeff form.
        let constancy_specs: &[(usize, &[usize])] = &[
            (0, &[COL_IS_ROW0,COL_IS_ROW1,COL_IS_LOGS_BLOOM_LEAF_BOUND_AT,COL_IS_ROW3,COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8]),
            (1, &[COL_IS_ROW1,COL_IS_LOGS_BLOOM_LEAF_BOUND_AT,COL_IS_ROW3,COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8]),
            (2, &[COL_IS_LOGS_BLOOM_LEAF_BOUND_AT,COL_IS_ROW3,COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9]),
            (3, &[COL_IS_ROW3,COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9]),
            (4, &[COL_IS_ROW4,COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9,COL_IS_ROW10]),
            (5, &[COL_IS_EXTRA_DATA_LEAF_BOUND_AT,COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9,COL_IS_ROW10]),
            (6, &[COL_IS_BLOCK_HASH_BOUND_AT,COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11]),
            (7, &[COL_IS_ROW7,COL_IS_PAIR8,COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11]),
            (8, &[COL_IS_ROW9,COL_IS_ROW10,COL_IS_ROW11,COL_IS_ROW12,COL_IS_PAIR13]),
            (9, &[COL_IS_ROW10,COL_IS_ROW11,COL_IS_ROW12,COL_IS_PAIR13]),
            (10, &[COL_IS_ROW11,COL_IS_ROW12,COL_IS_PAIR13,COL_IS_ROW14]),
            (11, &[COL_IS_ROW12,COL_IS_PAIR13,COL_IS_ROW14]),
            (12, &[COL_IS_ROW14,COL_IS_ROW15,COL_IS_PAIR16]),
            (13, &[COL_IS_ROW15,COL_IS_PAIR16]),
            (14, &[COL_IS_ROW17,COL_IS_PAIR18]),
        ];
        for &(idx, gate_sels) in constancy_specs {
            let mut gate = vec![Scalar::zero(curve)];
            for &gs in gate_sels { gate = poly_add(&gate, &column_coeffs[gs], curve); }
            let rel_col = COL_RELAY_OFFSET + idx * 32;
            let mut body_p = vec![Scalar::zero(curve)]; let mut bp = Scalar::one(curve);
            for k in 0..32 {
                let cur = &column_coeffs[rel_col + k];
                let nxt = poly_shift(cur, omega);
                let diff = poly_sub(&nxt, cur, curve);
                body_p = poly_add(&body_p, &poly_scalar_mul(&diff, &bp), curve);
                bp = bp.mul(alpha);
            }
            let rb = poly_mul(&gate, &body_p, curve);
            ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&rb, &ap), curve);
        }

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![
            COL_IS_REAL,
            COL_IS_ROOT_BOUND_AT,
            COL_IS_BLOCK_HASH_BOUND_AT,
            COL_IS_LOGS_BLOOM_LEAF_BOUND_AT,
            COL_IS_EXTRA_DATA_LEAF_BOUND_AT,
            COL_IS_LAYER_0,
            COL_IS_PAIR8, COL_IS_PAIR13, COL_IS_PAIR16, COL_IS_PAIR18,
            COL_IS_ROW0, COL_IS_ROW1, COL_IS_ROW3, COL_IS_ROW4, COL_IS_ROW7,
            COL_IS_ROW9, COL_IS_ROW10, COL_IS_ROW11, COL_IS_ROW12,
            COL_IS_ROW14, COL_IS_ROW15, COL_IS_ROW17,
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
            for (name, off) in [
                ("left", COL_LEFT_OFFSET),
                ("right", COL_RIGHT_OFFSET),
                ("hash", COL_HASH_OFFSET),
                ("claimed_root", COL_CLAIMED_ROOT_OFFSET),
                ("claimed_block_hash", COL_CLAIMED_BLOCK_HASH_OFFSET),
                ("claimed_logs_bloom_root", COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET),
                ("claimed_extra_data_root", COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET),
                ("claimed_left", COL_CLAIMED_LEFT_OFFSET),
                ("claimed_right", COL_CLAIMED_RIGHT_OFFSET),
                ("zh0", COL_ZH0_OFFSET), ("zh1", COL_ZH1_OFFSET),
                ("zh2", COL_ZH2_OFFSET), ("zh3", COL_ZH3_OFFSET),
            ] {
                declarations.push((
                    LookupDeclaration {
                        label: format!("payload_pair_{}_{}_8bit", name, k),
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
pub fn make_payload_pair_to_sha256_extract_linkage_descriptor(
    payload_pair_layer_index: usize,
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
        label: "execution_payload_header_pair_sha256_extract_v1".into(),
        a_layer_index: payload_pair_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(crate::sha256_extract::COL_IS_REAL),
    }
}

/// Column indices for the payload-pair AIR's `CLAIMED_BLOCK_HASH`
/// output. Returned for use by downstream cross-AIR LogUp descriptors
/// (Layer B's `BlockHeader.block_hash` binding).
pub fn make_payload_pair_block_hash_output_column_indices() -> Vec<usize> {
    (0..32).map(|k| COL_CLAIMED_BLOCK_HASH_OFFSET + k).collect()
}

/// Column indices for the payload-pair AIR's `CLAIMED_ROOT` output.
/// Returned for use by downstream cross-AIR LogUp descriptors
/// (body-pair AIR's field-9 column once it exists).
pub fn make_payload_pair_root_output_column_indices() -> Vec<usize> {
    (0..32).map(|k| COL_CLAIMED_ROOT_OFFSET + k).collect()
}

/// Column indices for the payload-pair AIR's `CLAIMED_LOGS_BLOOM_ROOT`
/// output. Returned for the future LogsBloomPairAIR ↔ payload bridge.
pub fn make_payload_pair_logs_bloom_root_output_column_indices() -> Vec<usize> {
    (0..32).map(|k| COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET + k).collect()
}

/// Column indices for the payload-pair AIR's `CLAIMED_EXTRA_DATA_ROOT`
/// output. Returned for the future ExtraDataPairAIR ↔ payload bridge.
pub fn make_payload_pair_extra_data_root_output_column_indices() -> Vec<usize> {
    (0..32).map(|k| COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET + k).collect()
}

/// Cross-AIR LogUp linkage: payload-pair's CLAIMED_BLOCK_HASH (the
/// execution block hash committed inside the payload at field 12) ↔
/// Layer B's `BlockHeader.block_hash` (the keccak256 of the canonical
/// RLP-encoded execution block header).
///
/// **A side (payload-pair)**: 32-byte CLAIMED_BLOCK_HASH tuple gated
/// by `COL_IS_BLOCK_HASH_BOUND_AT` (=1 only at payload invocation 6
/// → exactly 1 entry = block_hash field).
///
/// **B side (block_header_air)**: 32-byte BLOCK_HASH tuple gated by
/// `block_header_air::COL_IS_REAL`. For single-header proofs
/// (`witness.headers.len() == 1`) → exactly 1 entry. Multi-header
/// proofs are supported only when the same `block_hash` is repeated
/// across all rows (the per-row constancy isn't enforced by this
/// descriptor alone; for multi-block scenarios use a per-row binding
/// pattern instead).
///
/// Multiset equality on 1-vs-1 tuples algebraically pins
/// `payload.block_hash == block_header.block_hash`. This is the
/// **consensus↔execution bridge** — the final cross-AIR linkage
/// completing the algebraic chain BBH → body → payload → block_header
/// → tx/receipt/storage.
pub fn make_payload_pair_to_block_header_linkage_descriptor(
    payload_pair_layer_index: usize,
    block_header_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;

    let mut a_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { a_columns.push(COL_CLAIMED_BLOCK_HASH_OFFSET + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(32);
    for k in 0..32 { b_columns.push(bh::COL_BLOCK_HASH_OFFSET + k); }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "payload_pair_block_hash_to_block_header_v1".into(),
        a_layer_index: payload_pair_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_BLOCK_HASH_BOUND_AT),
        b_layer_index: block_header_layer_index,
        b_columns,
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_payload::ExecutionPayloadHeader;

    fn sample_payload() -> ExecutionPayloadHeader {
        let mut p = ExecutionPayloadHeader::default();
        p.parent_hash = [0x11u8; 32];
        p.state_root = [0x22u8; 32];
        p.receipts_root = [0x33u8; 32];
        p.block_hash = [0xabu8; 32];
        p.block_number = 19_000_000;
        p.gas_limit = 30_000_000;
        p.timestamp = 1_700_000_000;
        p
    }

    #[test]
    fn trace_builder_populates_20_real_rows() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // 20 invocations padded to nearest_power_of_two(20) = 32.
        assert_eq!(trace.num_rows, 20);
        assert_eq!(trace.padded_size, 32);
        for r in 0..20 {
            assert_eq!(trace.columns[COL_IS_REAL].evaluations[r].to_u64(), 1);
        }
        for r in 20..32 {
            assert!(trace.columns[COL_IS_REAL].evaluations[r].is_zero());
        }
    }

    #[test]
    fn pair_rows_match_witness_invocations() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
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
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ExecutionPayloadHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for v in &res[0] {
            assert!(v.is_zero());
        }
    }

    #[test]
    fn is_real_binary_fires_on_tampered_selector() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(7, CurveType::Bls48581);
        let cs = ExecutionPayloadHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!res[0][0].is_zero());
    }

    #[test]
    fn claimed_root_populated_and_constant() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let root = w.root;
        for r in 0..trace.num_rows {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_CLAIMED_ROOT_OFFSET + k].evaluations[r].to_u64(),
                    root[k] as u64,
                );
            }
        }
        // is_root_bound_at one-hot at row 19.
        assert_eq!(
            trace.columns[COL_IS_ROOT_BOUND_AT].evaluations[ROOT_INVOCATION_INDEX].to_u64(),
            1,
        );
        for r in 0..32 {
            if r != ROOT_INVOCATION_INDEX {
                assert!(trace.columns[COL_IS_ROOT_BOUND_AT].evaluations[r].is_zero());
            }
        }
    }

    #[test]
    fn claimed_block_hash_populated_and_constant() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // payload.block_hash = [0xab; 32]
        for r in 0..trace.num_rows {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_CLAIMED_BLOCK_HASH_OFFSET + k].evaluations[r].to_u64(),
                    0xab,
                );
            }
        }
        // is_block_hash_bound_at one-hot at row 6.
        assert_eq!(
            trace.columns[COL_IS_BLOCK_HASH_BOUND_AT].evaluations[BLOCK_HASH_LEFT_INVOCATION].to_u64(),
            1,
        );
        // The LEFT bytes at row 6 equal block_hash (field 12 in container).
        for k in 0..32 {
            assert_eq!(
                trace.columns[COL_LEFT_OFFSET + k].evaluations[BLOCK_HASH_LEFT_INVOCATION].to_u64(),
                0xab,
            );
        }
    }

    #[test]
    fn hash_eq_claimed_root_fires_on_tamper() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_CLAIMED_ROOT_OFFSET][ROOT_INVOCATION_INDEX] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = ExecutionPayloadHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 2 = hash_eq_claimed_root_at_root_invocation.
        assert!(!res[2][ROOT_INVOCATION_INDEX].is_zero());
    }

    #[test]
    fn left_eq_claimed_block_hash_fires_on_tamper() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_CLAIMED_BLOCK_HASH_OFFSET][BLOCK_HASH_LEFT_INVOCATION] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = ExecutionPayloadHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 4 = left_eq_claimed_block_hash_at_pair6.
        assert!(!res[4][BLOCK_HASH_LEFT_INVOCATION].is_zero());
    }

    #[test]
    fn claimed_logs_bloom_root_populated_and_constant() {
        let mut p = sample_payload();
        p.logs_bloom = [0xCD; 256];
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(p);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // Logs_bloom is field 4. LEFT of layer-0 pair at invocation 2
        // equals the field root. Cross-row constancy makes
        // CLAIMED_LOGS_BLOOM_ROOT[k] equal to this at every row.
        let expected = w.field_roots[4];
        for r in 0..trace.num_rows {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET + k]
                        .evaluations[r].to_u64(),
                    expected[k] as u64,
                );
            }
        }
        assert_eq!(
            trace.columns[COL_IS_LOGS_BLOOM_LEAF_BOUND_AT]
                .evaluations[LOGS_BLOOM_LEFT_INVOCATION].to_u64(),
            1,
        );
        for r in 0..32 {
            if r != LOGS_BLOOM_LEFT_INVOCATION {
                assert!(trace.columns[COL_IS_LOGS_BLOOM_LEAF_BOUND_AT]
                    .evaluations[r].is_zero());
            }
        }
    }

    #[test]
    fn claimed_extra_data_root_populated_and_constant() {
        let mut p = sample_payload();
        p.extra_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(p);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let expected = w.field_roots[10];
        for r in 0..trace.num_rows {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET + k]
                        .evaluations[r].to_u64(),
                    expected[k] as u64,
                );
            }
        }
        assert_eq!(
            trace.columns[COL_IS_EXTRA_DATA_LEAF_BOUND_AT]
                .evaluations[EXTRA_DATA_LEFT_INVOCATION].to_u64(),
            1,
        );
    }

    #[test]
    fn left_eq_claimed_logs_bloom_root_fires_on_tamper() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_CLAIMED_LOGS_BLOOM_ROOT_OFFSET][LOGS_BLOOM_LEFT_INVOCATION] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = ExecutionPayloadHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 6 = left_eq_claimed_logs_bloom_root_at_pair2.
        assert!(!res[6][LOGS_BLOOM_LEFT_INVOCATION].is_zero());
    }

    #[test]
    fn left_eq_claimed_extra_data_root_fires_on_tamper() {
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_CLAIMED_EXTRA_DATA_ROOT_OFFSET][EXTRA_DATA_LEFT_INVOCATION] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = ExecutionPayloadHeaderPairConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 8 = left_eq_claimed_extra_data_root_at_pair5.
        assert!(!res[8][EXTRA_DATA_LEFT_INVOCATION].is_zero());
    }

    #[test]
    fn block_header_linkage_descriptor_shape() {
        let d = make_payload_pair_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 32);
        assert_eq!(d.b_columns.len(), 32);
        assert_eq!(d.label, "payload_pair_block_hash_to_block_header_v1");
        assert_eq!(d.a_selector_column, Some(COL_IS_BLOCK_HASH_BOUND_AT));
        use crate::block_header_air as bh;
        assert_eq!(d.b_selector_column, Some(bh::COL_IS_REAL));
        assert_eq!(d.a_columns[0], COL_CLAIMED_BLOCK_HASH_OFFSET);
        assert_eq!(d.b_columns[0], bh::COL_BLOCK_HASH_OFFSET);
    }

    #[test]
    fn output_column_helpers_match_layout() {
        assert_eq!(make_payload_pair_root_output_column_indices().len(), 32);
        assert_eq!(make_payload_pair_block_hash_output_column_indices().len(), 32);
        assert_eq!(make_payload_pair_root_output_column_indices()[0], COL_CLAIMED_ROOT_OFFSET);
        assert_eq!(
            make_payload_pair_block_hash_output_column_indices()[0],
            COL_CLAIMED_BLOCK_HASH_OFFSET,
        );
    }

    #[test]
    fn linkage_descriptor_shape() {
        let d = make_payload_pair_to_sha256_extract_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 96);
        assert_eq!(d.b_columns.len(), 96);
        assert_eq!(d.label, "execution_payload_header_pair_sha256_extract_v1");
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
        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = ExecutionPayloadHeaderPairConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }

    /// 2-AIR joint_prove cap-stone for step 2g: payload-pair AIR ↔
    /// Layer B's `block_header_air` via the 32-byte
    /// CLAIMED_BLOCK_HASH ↔ BLOCK_HASH linkage. **This is the
    /// consensus↔execution bridge** — proves the block_hash carried
    /// inside the beacon block's execution payload equals the
    /// block_hash that Layer B's BlockHeader AIR works with.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with 1 32-col linkage \
                (~5-15 min release)"]
    fn joint_prove_payload_pair_to_block_header_air() {
        use crate::block_header::BlockHeader;
        use crate::block_header_air as bh;
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Pick a block header and compute its canonical block hash.
        let block_header = BlockHeader {
            state_root: [0x55; 32],
            transactions_root: [0x66; 32],
            receipts_root: [0x77; 32],
            ..Default::default()
        };
        let block_hash = crate::block_header::block_header_hash(&block_header);

        // Build a payload whose block_hash matches.
        let mut payload = sample_payload();
        payload.block_hash = block_hash;
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(payload);

        // Build the block_header AIR witness for that single header.
        let bh_row = bh::from_block_header(&block_header);
        let bh_w = bh::BlockHeaderWitness::from_headers(vec![bh_row]);

        // Payload-pair (layer 0).
        let p_trace = build_trace_polynomials(&payload_w, curve);
        let p_omega = scheme.domain_generator(p_trace.padded_size);
        let p_cs = ExecutionPayloadHeaderPairConstraintSystem::new(p_trace.num_rows)
            .with_omega_and_domain(p_omega, p_trace.padded_size);

        // Block header (layer 1).
        let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
        let bh_omega = scheme.domain_generator(bh_trace.padded_size);
        let bh_cs = bh::BlockHeaderConstraintSystem::new(bh_trace.num_rows)
            .with_omega_and_domain(bh_omega, bh_trace.padded_size);

        let linkage = make_payload_pair_to_block_header_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&p_trace, &p_cs), (&bh_trace, &bh_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("payload-pair ↔ block_header_air joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&p_cs, &bh_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
            "joint_verify must accept honest payload ↔ Layer B composition",
        );
    }

    /// **Cap-stone: 4-AIR joint_prove for the full C↔B algebraic
    /// chain.** Combines BBH-pair AIR + body-pair AIR + payload-pair
    /// AIR + block_header_air with all 3 cross-AIR LogUp bridges in
    /// a single composition:
    ///
    ///   - `make_body_pair_to_bbh_pair_linkage_descriptor`
    ///   - `make_payload_pair_to_body_pair_linkage_descriptor`
    ///   - `make_payload_pair_to_block_header_linkage_descriptor`
    ///
    /// Successful prove+verify with all 3 `closure_match=true`
    /// demonstrates that a single algebraic proof binds:
    ///   BeaconBlockHeader.body_root
    ///     == body.computed_root
    ///     == body.field_9
    ///     == payload.computed_root  AND  payload.block_hash
    ///     == BlockHeader.block_hash
    ///
    /// I.e., the chain from beacon header → execution block hash is
    /// fully algebraic in one proof.
    #[test]
    #[ignore = "slow: 4-AIR joint_prove with 3 linkages (~20-40 min release)"]
    fn joint_prove_4_air_full_c_to_b_chain() {
        use crate::beacon::BeaconBlockHeader;
        use crate::beacon_block_body::BeaconBlockBody;
        use crate::beacon_block_body_air::BeaconBlockBodyHtrWitness;
        use crate::beacon_block_body_pair_air as body;
        use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;
        use crate::beacon_block_header_pair_air as bbh;
        use crate::block_header::BlockHeader;
        use crate::block_header_air as bh;
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::execution_payload::ExecutionPayloadHeader;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── Build the consistent chain bottom-up ──────────────────────
        // 1. Layer B BlockHeader and its block_hash.
        let block_header = BlockHeader {
            state_root: [0x55; 32],
            transactions_root: [0x66; 32],
            receipts_root: [0x77; 32],
            ..Default::default()
        };
        let block_hash = crate::block_header::block_header_hash(&block_header);

        // 2. ExecutionPayloadHeader with matching block_hash.
        let mut payload = ExecutionPayloadHeader::default();
        payload.block_hash = block_hash;
        payload.block_number = 18_500_000;
        payload.gas_limit = 30_000_000;
        payload.timestamp = 1_700_000_000;

        // 3. BeaconBlockBody with that payload.
        let body_struct = BeaconBlockBody {
            graffiti: [0x47; 32],
            execution_payload_header: payload.clone(),
            ..Default::default()
        };

        // 4. BeaconBlockHeader committing to body's root.
        let beacon_header = BeaconBlockHeader {
            slot: 7_777_777,
            proposer_index: 13,
            parent_root: [0xaa; 32],
            state_root: [0xbb; 32],
            body_root: body_struct.hash_tree_root(),
        };

        // ── Build all 4 witnesses + traces + CSes ────────────────────
        // Layer 0: BBH-pair.
        let bbh_w = BeaconBlockHeaderHtrWitness::from_header(beacon_header);
        let bbh_trace = bbh::build_trace_polynomials(&bbh_w, curve);
        let bbh_omega = scheme.domain_generator(bbh_trace.padded_size);
        let bbh_cs = bbh::BeaconBlockHeaderPairConstraintSystem::new(bbh_trace.num_rows)
            .with_omega_and_domain(bbh_omega, bbh_trace.padded_size);

        // Layer 1: body-pair.
        let body_w = BeaconBlockBodyHtrWitness::from_body(body_struct);
        let body_trace = body::build_trace_polynomials(&body_w, curve);
        let body_omega = scheme.domain_generator(body_trace.padded_size);
        let body_cs = body::BeaconBlockBodyPairConstraintSystem::new(body_trace.num_rows)
            .with_omega_and_domain(body_omega, body_trace.padded_size);

        // Layer 2: payload-pair.
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(payload);
        let payload_trace = build_trace_polynomials(&payload_w, curve);
        let payload_omega = scheme.domain_generator(payload_trace.padded_size);
        let payload_cs = ExecutionPayloadHeaderPairConstraintSystem::new(payload_trace.num_rows)
            .with_omega_and_domain(payload_omega, payload_trace.padded_size);

        // Layer 3: block_header_air.
        let bh_row = bh::from_block_header(&block_header);
        let bh_w = bh::BlockHeaderWitness::from_headers(vec![bh_row]);
        let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
        let bh_omega = scheme.domain_generator(bh_trace.padded_size);
        let bh_cs = bh::BlockHeaderConstraintSystem::new(bh_trace.num_rows)
            .with_omega_and_domain(bh_omega, bh_trace.padded_size);

        // ── Wire all 3 linkage descriptors ──────────────────────────
        // body (layer 1) → BBH (layer 0).
        let link_body_bbh = body::make_body_pair_to_bbh_pair_linkage_descriptor(1, 0);
        // payload (layer 2) → body (layer 1).
        let link_payload_body = body::make_payload_pair_to_body_pair_linkage_descriptor(2, 1);
        // payload (layer 2) → block_header (layer 3).
        let link_payload_bh = make_payload_pair_to_block_header_linkage_descriptor(2, 3);
        let linkages = vec![
            link_body_bbh.clone(),
            link_payload_body.clone(),
            link_payload_bh.clone(),
        ];

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> = vec![
            (&bbh_trace, &bbh_cs),
            (&body_trace, &body_cs),
            (&payload_trace, &payload_cs),
            (&bh_trace, &bh_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("4-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 4);
        assert_eq!(ext.linkage_proofs.len(), 3);
        for lp in &ext.linkage_proofs {
            eprintln!(
                "[diag] linkage label={} closure_match={}",
                lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b, "linkage {} must close", lp.label);
        }

        let cs_refs: Vec<&dyn VmConstraintSystem> =
            vec![&bbh_cs, &body_cs, &payload_cs, &bh_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must accept honest 4-AIR composition with 3 bridges",
        );
    }

    /// **Cap-stone: 5-AIR joint_prove for the full Finality + C↔B
    /// algebraic chain.** Adds the Finality AIR on top of the 4-AIR
    /// cap-stone, with the new Finality↔BBH bridge.
    ///
    /// 4 cross-AIR LogUp bridges:
    ///   - `make_body_pair_to_bbh_pair_linkage_descriptor`
    ///   - `make_payload_pair_to_body_pair_linkage_descriptor`
    ///   - `make_payload_pair_to_block_header_linkage_descriptor`
    ///   - `make_finality_to_bbh_pair_linkage_descriptor`
    ///
    /// Successful prove+verify with all 4 `closure_match=true`
    /// algebraically demonstrates:
    ///
    ///   FFG-finalized BeaconBlockHeader
    ///     → BeaconBlockBody
    ///     → ExecutionPayloadHeader
    ///     → BlockHeader (Layer B's block_hash)
    ///
    /// **all bound in a single algebraic proof.** This is the
    /// algebraic version of `verify_finalized_beacon_world_proof_oracle`
    /// from `beacon_world_proof.rs`.
    #[test]
    #[ignore = "very slow: 5-AIR joint_prove with 4 linkages (~25-45 min release)"]
    fn joint_prove_5_air_finality_to_block_hash() {
        use crate::beacon::BeaconBlockHeader;
        use crate::beacon_block_body::BeaconBlockBody;
        use crate::beacon_block_body_air::BeaconBlockBodyHtrWitness;
        use crate::beacon_block_body_pair_air as body;
        use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;
        use crate::beacon_block_header_pair_air as bbh;
        use crate::block_header::BlockHeader;
        use crate::block_header_air as bh;
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::execution_payload::ExecutionPayloadHeader;
        use crate::finality_constraints as fin;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── Build the consistent chain bottom-up (Layer B → C → Finality) ─
        let block_header = BlockHeader {
            state_root: [0x55; 32],
            transactions_root: [0x66; 32],
            receipts_root: [0x77; 32],
            ..Default::default()
        };
        let block_hash = crate::block_header::block_header_hash(&block_header);

        let mut payload = ExecutionPayloadHeader::default();
        payload.block_hash = block_hash;
        payload.block_number = 18_500_000;
        payload.gas_limit = 30_000_000;
        payload.timestamp = 1_700_000_000;

        let body_struct = BeaconBlockBody {
            graffiti: [0x47; 32],
            execution_payload_header: payload.clone(),
            ..Default::default()
        };

        let beacon_header = BeaconBlockHeader {
            slot: 7_777_777,
            proposer_index: 13,
            parent_root: [0xaa; 32],
            state_root: [0xbb; 32],
            body_root: body_struct.hash_tree_root(),
        };
        let bbh_root = beacon_header.hash_tree_root();

        // FFG-supermajority: 3 of 4 validators attest (75% stake > 2/3).
        let eb = 32_000_000_000u64;
        let finality_witness = fin::FinalityWitness {
            validators: vec![(eb, 1), (eb, 1), (eb, 1), (eb, 0)],
            total_active_balance_gwei: 4 * eb,
            attestation_data_root: [0xDD; 32],
            finalized_root: bbh_root,
        };

        // ── Build all 5 witnesses + traces + CSes ───────────────────
        // Layer 0: BBH-pair.
        let bbh_w = BeaconBlockHeaderHtrWitness::from_header(beacon_header);
        let bbh_trace = bbh::build_trace_polynomials(&bbh_w, curve);
        let bbh_omega = scheme.domain_generator(bbh_trace.padded_size);
        let bbh_cs = bbh::BeaconBlockHeaderPairConstraintSystem::new(bbh_trace.num_rows)
            .with_omega_and_domain(bbh_omega, bbh_trace.padded_size);

        // Layer 1: body-pair.
        let body_w = BeaconBlockBodyHtrWitness::from_body(body_struct);
        let body_trace = body::build_trace_polynomials(&body_w, curve);
        let body_omega = scheme.domain_generator(body_trace.padded_size);
        let body_cs = body::BeaconBlockBodyPairConstraintSystem::new(body_trace.num_rows)
            .with_omega_and_domain(body_omega, body_trace.padded_size);

        // Layer 2: payload-pair.
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(payload);
        let payload_trace = build_trace_polynomials(&payload_w, curve);
        let payload_omega = scheme.domain_generator(payload_trace.padded_size);
        let payload_cs = ExecutionPayloadHeaderPairConstraintSystem::new(payload_trace.num_rows)
            .with_omega_and_domain(payload_omega, payload_trace.padded_size);

        // Layer 3: block_header_air.
        let bh_row = bh::from_block_header(&block_header);
        let bh_w = bh::BlockHeaderWitness::from_headers(vec![bh_row]);
        let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
        let bh_omega = scheme.domain_generator(bh_trace.padded_size);
        let bh_cs = bh::BlockHeaderConstraintSystem::new(bh_trace.num_rows)
            .with_omega_and_domain(bh_omega, bh_trace.padded_size);

        // Layer 4: Finality.
        let fin_trace = fin::build_finality_trace_polynomials(&finality_witness, curve);
        let fin_omega = scheme.domain_generator(fin_trace.padded_size);
        let fin_cs = fin::FinalityConstraintSystem::new(finality_witness.validators.len())
            .with_omega_and_domain(fin_omega, fin_trace.padded_size);

        // ── Wire all 4 linkage descriptors ──────────────────────────
        let link_body_bbh = body::make_body_pair_to_bbh_pair_linkage_descriptor(1, 0);
        let link_payload_body = body::make_payload_pair_to_body_pair_linkage_descriptor(2, 1);
        let link_payload_bh = make_payload_pair_to_block_header_linkage_descriptor(2, 3);
        let link_finality_bbh = fin::make_finality_to_bbh_pair_linkage_descriptor(4, 0);
        let linkages = vec![
            link_body_bbh.clone(),
            link_payload_body.clone(),
            link_payload_bh.clone(),
            link_finality_bbh.clone(),
        ];

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> = vec![
            (&bbh_trace, &bbh_cs),
            (&body_trace, &body_cs),
            (&payload_trace, &payload_cs),
            (&bh_trace, &bh_cs),
            (&fin_trace, &fin_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("5-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 5);
        assert_eq!(ext.linkage_proofs.len(), 4);
        for lp in &ext.linkage_proofs {
            eprintln!(
                "[diag] linkage label={} closure_match={}",
                lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b, "linkage {} must close", lp.label);
        }

        let cs_refs: Vec<&dyn VmConstraintSystem> =
            vec![&bbh_cs, &body_cs, &payload_cs, &bh_cs, &fin_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must accept honest 5-AIR composition with 4 bridges",
        );
    }

    /// **Ultimate cap-stone: 7-AIR joint_prove fully closing the
    /// C↔B algebraic chain at the SSZ leaf level.** No opaque field
    /// roots anywhere in the chain — every non-trivial payload field
    /// root (logs_bloom, extra_data) is algebraically reachable from
    /// a real merkleization.
    ///
    /// 6 cross-AIR LogUp bridges:
    ///   1. body→BBH
    ///   2. payload→body
    ///   3. payload→Layer B (consensus↔execution)
    ///   4. Finality→BBH
    ///   5. logs_bloom→payload (sub-tree #1)
    ///   6. extra_data→payload (sub-tree #2)
    ///
    /// Successful prove+verify with all 6 closures matching
    /// demonstrates:
    ///   FFG-finalized BeaconBlockHeader
    ///     → committed BeaconBlockBody
    ///     → committed ExecutionPayloadHeader
    ///       → algebraically-merkleized logs_bloom
    ///       → algebraically-merkleized extra_data
    ///     → committed Layer B BlockHeader.block_hash
    ///   …all bound in a single algebraic proof, no host-supplied
    ///   field roots.
    #[test]
    #[ignore = "very slow: 7-AIR joint_prove with 6 linkages (~30-50 min release)"]
    fn joint_prove_7_air_full_subtree_chain() {
        use crate::beacon::BeaconBlockHeader;
        use crate::beacon_block_body::BeaconBlockBody;
        use crate::beacon_block_body_air::BeaconBlockBodyHtrWitness;
        use crate::beacon_block_body_pair_air as body;
        use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;
        use crate::beacon_block_header_pair_air as bbh;
        use crate::block_header::BlockHeader;
        use crate::block_header_air as bh;
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::execution_payload::ExecutionPayloadHeader;
        use crate::extra_data_air::ExtraDataHtrWitness;
        use crate::extra_data_pair_air as extra;
        use crate::finality_constraints as fin;
        use crate::logs_bloom_air::LogsBloomHtrWitness;
        use crate::logs_bloom_pair_air as logs;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── Build the consistent chain bottom-up ──────────────────────
        let block_header = BlockHeader {
            state_root: [0x55; 32],
            transactions_root: [0x66; 32],
            receipts_root: [0x77; 32],
            ..Default::default()
        };
        let block_hash = crate::block_header::block_header_hash(&block_header);

        let logs_bloom = [0xAB; 256];
        let extra_data = vec![0xDE, 0xAD, 0xBE, 0xEF];

        let mut payload = ExecutionPayloadHeader::default();
        payload.block_hash = block_hash;
        payload.logs_bloom = logs_bloom;
        payload.extra_data = extra_data.clone();
        payload.block_number = 18_500_000;
        payload.gas_limit = 30_000_000;
        payload.timestamp = 1_700_000_000;

        let body_struct = BeaconBlockBody {
            graffiti: [0x47; 32],
            execution_payload_header: payload.clone(),
            ..Default::default()
        };

        let beacon_header = BeaconBlockHeader {
            slot: 7_777_777,
            proposer_index: 13,
            parent_root: [0xaa; 32],
            state_root: [0xbb; 32],
            body_root: body_struct.hash_tree_root(),
        };
        let bbh_root = beacon_header.hash_tree_root();

        let eb = 32_000_000_000u64;
        let finality_witness = fin::FinalityWitness {
            validators: vec![(eb, 1), (eb, 1), (eb, 1), (eb, 0)],
            total_active_balance_gwei: 4 * eb,
            attestation_data_root: [0xDD; 32],
            finalized_root: bbh_root,
        };

        // ── Build all 7 witnesses + traces + CSes ────────────────────
        // Layer 0: BBH-pair.
        let bbh_w = BeaconBlockHeaderHtrWitness::from_header(beacon_header);
        let bbh_trace = bbh::build_trace_polynomials(&bbh_w, curve);
        let bbh_omega = scheme.domain_generator(bbh_trace.padded_size);
        let bbh_cs = bbh::BeaconBlockHeaderPairConstraintSystem::new(bbh_trace.num_rows)
            .with_omega_and_domain(bbh_omega, bbh_trace.padded_size);

        // Layer 1: body-pair.
        let body_w = BeaconBlockBodyHtrWitness::from_body(body_struct);
        let body_trace = body::build_trace_polynomials(&body_w, curve);
        let body_omega = scheme.domain_generator(body_trace.padded_size);
        let body_cs = body::BeaconBlockBodyPairConstraintSystem::new(body_trace.num_rows)
            .with_omega_and_domain(body_omega, body_trace.padded_size);

        // Layer 2: payload-pair.
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(payload);
        let payload_trace = build_trace_polynomials(&payload_w, curve);
        let payload_omega = scheme.domain_generator(payload_trace.padded_size);
        let payload_cs = ExecutionPayloadHeaderPairConstraintSystem::new(payload_trace.num_rows)
            .with_omega_and_domain(payload_omega, payload_trace.padded_size);

        // Layer 3: block_header_air.
        let bh_row = bh::from_block_header(&block_header);
        let bh_w = bh::BlockHeaderWitness::from_headers(vec![bh_row]);
        let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
        let bh_omega = scheme.domain_generator(bh_trace.padded_size);
        let bh_cs = bh::BlockHeaderConstraintSystem::new(bh_trace.num_rows)
            .with_omega_and_domain(bh_omega, bh_trace.padded_size);

        // Layer 4: Finality.
        let fin_trace = fin::build_finality_trace_polynomials(&finality_witness, curve);
        let fin_omega = scheme.domain_generator(fin_trace.padded_size);
        let fin_cs = fin::FinalityConstraintSystem::new(finality_witness.validators.len())
            .with_omega_and_domain(fin_omega, fin_trace.padded_size);

        // Layer 5: logs_bloom-pair.
        let logs_w = LogsBloomHtrWitness::from_logs_bloom(logs_bloom);
        let logs_trace = logs::build_trace_polynomials(&logs_w, curve);
        let logs_omega = scheme.domain_generator(logs_trace.padded_size);
        let logs_cs = logs::LogsBloomPairConstraintSystem::new(logs_trace.num_rows)
            .with_omega_and_domain(logs_omega, logs_trace.padded_size);

        // Layer 6: extra_data-pair.
        let extra_w = ExtraDataHtrWitness::from_extra_data(extra_data);
        let extra_trace = extra::build_trace_polynomials(&extra_w, curve);
        let extra_omega = scheme.domain_generator(extra_trace.padded_size);
        let extra_cs = extra::ExtraDataPairConstraintSystem::new(extra_trace.num_rows)
            .with_omega_and_domain(extra_omega, extra_trace.padded_size);

        // ── Wire all 6 linkage descriptors ──────────────────────────
        let link_body_bbh = body::make_body_pair_to_bbh_pair_linkage_descriptor(1, 0);
        let link_payload_body = body::make_payload_pair_to_body_pair_linkage_descriptor(2, 1);
        let link_payload_bh = make_payload_pair_to_block_header_linkage_descriptor(2, 3);
        let link_finality_bbh = fin::make_finality_to_bbh_pair_linkage_descriptor(4, 0);
        let link_logs_payload = logs::make_logs_bloom_pair_to_payload_pair_linkage_descriptor(5, 2);
        let link_extra_payload = extra::make_extra_data_pair_to_payload_pair_linkage_descriptor(6, 2);
        let linkages = vec![
            link_body_bbh.clone(),
            link_payload_body.clone(),
            link_payload_bh.clone(),
            link_finality_bbh.clone(),
            link_logs_payload.clone(),
            link_extra_payload.clone(),
        ];

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> = vec![
            (&bbh_trace, &bbh_cs),
            (&body_trace, &body_cs),
            (&payload_trace, &payload_cs),
            (&bh_trace, &bh_cs),
            (&fin_trace, &fin_cs),
            (&logs_trace, &logs_cs),
            (&extra_trace, &extra_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("7-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 7);
        assert_eq!(ext.linkage_proofs.len(), 6);
        for lp in &ext.linkage_proofs {
            eprintln!(
                "[diag] linkage label={} closure_match={}",
                lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b, "linkage {} must close", lp.label);
        }

        let cs_refs: Vec<&dyn VmConstraintSystem> =
            vec![&bbh_cs, &body_cs, &payload_cs, &bh_cs, &fin_cs, &logs_cs, &extra_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must accept honest 7-AIR composition with 6 bridges",
        );
    }

    /// 2-AIR joint_prove cap-stone for step 1: payload-pair AIR ↔
    /// Sha256Extract via the 96-byte tuple linkage.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with 1 96-col linkage \
                (~5-15 min release)"]
    fn joint_prove_payload_pair_to_sha256_extract() {
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

        let w = ExecutionPayloadHeaderHtrWitness::from_payload(sample_payload());

        let payload_trace = build_trace_polynomials(&w, curve);
        let payload_omega = scheme.domain_generator(payload_trace.padded_size);
        let payload_cs = ExecutionPayloadHeaderPairConstraintSystem::new(payload_trace.num_rows)
            .with_omega_and_domain(payload_omega, payload_trace.padded_size);

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

        let linkage = make_payload_pair_to_sha256_extract_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&payload_trace, &payload_cs), (&se_trace, &se_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("payload-pair ↔ Sha256Extract joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&payload_cs, &se_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
        );
    }
}
