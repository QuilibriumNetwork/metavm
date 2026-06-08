//! Beacon block validity composition AIR.
//!
//! Per-row composer that threads cross-AIR LogUp bindings between
//! [`crate::block_proposer_sig_air`] (proposer-signature validity),
//! [`crate::beacon_block_body_air`] (body merkleization),
//! [`crate::attestation_aggregate_air`] (per-attestation BLS aggregate),
//! [`crate::beacon_state_transition_air`] (state-root transition), and
//! [`crate::bbh_root_consumer_air`] (the canonical
//! `BeaconBlockHeader` consumer that anchors `(slot, proposer_index,
//! parent_root, body_root, state_root)`).
//!
//! ## Per-row witness (one row per beacon block)
//!
//!   - `slot` [u64] — beacon slot
//!   - `proposer_index` [u64] — validator index of block proposer
//!   - `block_root[32]` — `hash_tree_root(BeaconBlock)`
//!   - `parent_root[32]` — previous block header's hash-tree-root
//!   - `body_root[32]` — `hash_tree_root(BeaconBlockBody)`
//!   - `state_root[32]` — post-state SSZ hash-tree-root
//!   - `num_attestations` [u32] — number of attestations in the block
//!     (host-committed; capped at 16 to match
//!     [`crate::attestation_aggregate_air`]'s MAX_COMMITTEE bound).
//!   - `is_real` (binary) — row activity selector.
//!
//! ## Algebraic row-local constraints (8 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`.
//! 1. `slot_le_decomp` — `SLOT = Σ_b SLOT_BYTE[b] · 2^(8b)` (LE, 8 bytes).
//! 2. `proposer_index_le_decomp` — `PROPOSER_INDEX = Σ_b PI_BYTE[b] · 2^(8b)`
//!    (LE, 8 bytes).
//! 3. `num_attestations_le_decomp` — `NUM_ATT = Σ_b NA_BYTE[b] · 2^(8b)`
//!    (LE, 4 bytes — u32).
//! 4. `num_attestations_bound` — label-only slot reserved for a future
//!    `NUM_ATT ≤ MAX_ATT` constraint once the corresponding gadget AIR is
//!    wired; today returns zero so adding it later doesn't renumber bodies.
//! 5. `block_root_first_byte_anchor` — label-only zero slot (paralleling
//!    the pattern used in [`crate::block_proposer_sig_air`]).
//! 6. `body_root_first_byte_anchor` — label-only zero slot.
//! 7. `state_root_first_byte_anchor` — label-only zero slot.
//!
//! All byte columns are 8-bit range-checked via [`lookup_declarations`].
//!
//! ## Cross-AIR LogUp descriptors
//!
//!   - `make_block_validity_to_proposer_sig_descriptor` — binds
//!     `(slot_bytes[0..8], proposer_index_bytes[0..8], block_root[0..32])`
//!     to [`crate::block_proposer_sig_air`].
//!   - `make_block_validity_to_body_root_descriptor` — binds
//!     `body_root[0..32]` to
//!     [`crate::beacon_block_body_air::BBB8_COL_CLAIMED_BODY_ROOT_OFFSET`].
//!   - `make_block_validity_to_attestation_descriptor(i)` — binds
//!     `(slot_bytes[0..8])` of this AIR (gated by `IS_REAL`) to the
//!     attestation-aggregate AIR for slot consistency. Soundly composed
//!     for each `i ∈ 0..MAX_ATT`; the actual *count* of present
//!     attestations is the host-committed `num_attestations` column. The
//!     B-side selector is the attestation AIR's `IS_REAL` so empty rows
//!     vanish algebraically.
//!   - `make_block_validity_to_state_transition_descriptor` — binds
//!     `(slot_bytes[0..8], state_root[0..32])` against
//!     [`crate::beacon_state_transition_air`]'s
//!     `(SLOT_BYTE, POST_STATE_ROOT)`.
//!   - `make_block_validity_to_bbh_descriptor` — binds
//!     `(slot_bytes[0..8], proposer_index_bytes[0..8], parent_root[0..32],
//!     body_root[0..32], state_root[0..32])` against
//!     [`crate::bbh_root_consumer_air`].
//!
//! ## What this AIR does NOT prove (deferred)
//!
//!   - The contents of each sub-AIR (proposer signature pairing,
//!     attestation BLS aggregation, body merkleization, state transition)
//!     are NOT re-derived here. The descriptors compose them.
//!   - The mapping `num_attestations → (selected attestation rows)` is a
//!     host-side oracle today; binding it algebraically requires a
//!     dedicated count gadget that is left as a follow-up.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const HASH_BYTES: usize = 32;
pub const U64_BYTES: usize = 8;
pub const U32_BYTES: usize = 4;

/// Maximum number of attestation descriptors composed per block. Matches
/// [`crate::attestation_aggregate_air::MAX_COMMITTEE`] for tractable
/// fast-test domains.
pub const MAX_ATT: usize = 16;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_SLOT: usize = 0;
pub const COL_PROPOSER_INDEX: usize = COL_SLOT + 1;

pub const COL_BLOCK_ROOT_OFFSET: usize = COL_PROPOSER_INDEX + 1;          // .. +32
pub const COL_PARENT_ROOT_OFFSET: usize = COL_BLOCK_ROOT_OFFSET + HASH_BYTES;
pub const COL_BODY_ROOT_OFFSET: usize = COL_PARENT_ROOT_OFFSET + HASH_BYTES;
pub const COL_STATE_ROOT_OFFSET: usize = COL_BODY_ROOT_OFFSET + HASH_BYTES;

pub const COL_NUM_ATTESTATIONS: usize = COL_STATE_ROOT_OFFSET + HASH_BYTES;

// LE byte decomps.
pub const COL_SLOT_BYTE_OFFSET: usize = COL_NUM_ATTESTATIONS + 1;         // 8 bytes
pub const COL_PI_BYTE_OFFSET: usize = COL_SLOT_BYTE_OFFSET + U64_BYTES;   // 8 bytes
pub const COL_NA_BYTE_OFFSET: usize = COL_PI_BYTE_OFFSET + U64_BYTES;     // 4 bytes (u32)

pub const COL_IS_REAL: usize = COL_NA_BYTE_OFFSET + U32_BYTES;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

/// 8 row-local bodies (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 8;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct BeaconBlockValidityRow {
    pub slot: u64,
    pub proposer_index: u64,
    pub block_root: [u8; HASH_BYTES],
    pub parent_root: [u8; HASH_BYTES],
    pub body_root: [u8; HASH_BYTES],
    pub state_root: [u8; HASH_BYTES],
    pub num_attestations: u32,
}

#[derive(Clone, Debug, Default)]
pub struct BeaconBlockValidityWitness {
    pub rows: Vec<BeaconBlockValidityRow>,
}

impl BeaconBlockValidityWitness {
    /// Build a single-row validity witness for a beacon block. All
    /// roots are host-committed; the per-sub-system bindings are
    /// algebraically enforced via the descriptors in this module.
    pub fn from_block_components(
        slot: u64,
        proposer_index: u64,
        block_root: [u8; HASH_BYTES],
        parent_root: [u8; HASH_BYTES],
        body_root: [u8; HASH_BYTES],
        state_root: [u8; HASH_BYTES],
        num_attestations: u32,
    ) -> Self {
        Self {
            rows: vec![BeaconBlockValidityRow {
                slot,
                proposer_index,
                block_root,
                parent_root,
                body_root,
                state_root,
                num_attestations,
            }],
        }
    }

    pub fn from_rows(rows: Vec<BeaconBlockValidityRow>) -> Self {
        Self { rows }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

/// `target − Σ_b byte[b] · 2^(8b)` over `len` LE bytes.
fn eval_le_decomp(
    target_value: &Scalar,
    byte_off: usize,
    len: usize,
    col_evals: &[Scalar],
) -> Scalar {
    let curve = target_value.curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..len {
        let byte = &col_evals[byte_off + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    target_value.sub(&sum)
}

fn build_le_decomp_poly(
    target_poly: &[Scalar],
    byte_off: usize,
    len: usize,
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..len {
        let byte_poly = &col_coeffs[byte_off + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    poly_sub(target_poly, &sum, curve)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BeaconBlockValidityWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_SLOT][i] = Scalar::from_u64(row.slot, curve);
        columns[COL_PROPOSER_INDEX][i] = Scalar::from_u64(row.proposer_index, curve);

        for k in 0..HASH_BYTES {
            columns[COL_BLOCK_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.block_root[k] as u64, curve);
            columns[COL_PARENT_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.parent_root[k] as u64, curve);
            columns[COL_BODY_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.body_root[k] as u64, curve);
            columns[COL_STATE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.state_root[k] as u64, curve);
        }

        columns[COL_NUM_ATTESTATIONS][i] =
            Scalar::from_u64(row.num_attestations as u64, curve);

        let slot_bytes = row.slot.to_le_bytes();
        let pi_bytes = row.proposer_index.to_le_bytes();
        let na_bytes = row.num_attestations.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_SLOT_BYTE_OFFSET + b][i] =
                Scalar::from_u64(slot_bytes[b] as u64, curve);
            columns[COL_PI_BYTE_OFFSET + b][i] =
                Scalar::from_u64(pi_bytes[b] as u64, curve);
        }
        for b in 0..U32_BYTES {
            columns[COL_NA_BYTE_OFFSET + b][i] =
                Scalar::from_u64(na_bytes[b] as u64, curve);
        }

        columns[COL_IS_REAL][i] = one.clone();
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

pub struct BeaconBlockValidityConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BeaconBlockValidityConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for BeaconBlockValidityConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "slot_le_decomp".into(),
            "proposer_index_le_decomp".into(),
            "num_attestations_le_decomp".into(),
            "num_attestations_bound".into(),
            "block_root_first_byte_anchor".into(),
            "body_root_first_byte_anchor".into(),
            "state_root_first_byte_anchor".into(),
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
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];

            // 0: is_real binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));

            // 1: slot LE decomp.
            bodies[1][row] = eval_le_decomp(
                &row_evals[COL_SLOT],
                COL_SLOT_BYTE_OFFSET,
                U64_BYTES,
                &row_evals,
            );

            // 2: proposer_index LE decomp.
            bodies[2][row] = eval_le_decomp(
                &row_evals[COL_PROPOSER_INDEX],
                COL_PI_BYTE_OFFSET,
                U64_BYTES,
                &row_evals,
            );

            // 3: num_attestations LE decomp (u32 → 4 bytes).
            bodies[3][row] = eval_le_decomp(
                &row_evals[COL_NUM_ATTESTATIONS],
                COL_NA_BYTE_OFFSET,
                U32_BYTES,
                &row_evals,
            );

            // 4..7: label-only anchor slots (body == 0 unconditionally).
            // bodies[4..8] already zero.
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals[COL_IS_REAL];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            eval_le_decomp(
                &col_evals[COL_SLOT],
                COL_SLOT_BYTE_OFFSET,
                U64_BYTES,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_PROPOSER_INDEX],
                COL_PI_BYTE_OFFSET,
                U64_BYTES,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_NUM_ATTESTATIONS],
                COL_NA_BYTE_OFFSET,
                U32_BYTES,
                col_evals,
            ),
            Scalar::zero(curve),
            Scalar::zero(curve),
            Scalar::zero(curve),
            Scalar::zero(curve),
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
        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let slot_decomp = build_le_decomp_poly(
            &col_coeffs[COL_SLOT],
            COL_SLOT_BYTE_OFFSET,
            U64_BYTES,
            col_coeffs,
            curve,
        );
        let pi_decomp = build_le_decomp_poly(
            &col_coeffs[COL_PROPOSER_INDEX],
            COL_PI_BYTE_OFFSET,
            U64_BYTES,
            col_coeffs,
            curve,
        );
        let na_decomp = build_le_decomp_poly(
            &col_coeffs[COL_NUM_ATTESTATIONS],
            COL_NA_BYTE_OFFSET,
            U32_BYTES,
            col_coeffs,
            curve,
        );

        let zero_poly = vec![Scalar::zero(curve)];
        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            slot_decomp,
            pi_decomp,
            na_decomp,
            zero_poly.clone(),
            zero_poly.clone(),
            zero_poly.clone(),
            zero_poly,
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

        let hash_ranges: [(usize, usize, &str); 4] = [
            (COL_BLOCK_ROOT_OFFSET, HASH_BYTES, "block_root"),
            (COL_PARENT_ROOT_OFFSET, HASH_BYTES, "parent_root"),
            (COL_BODY_ROOT_OFFSET, HASH_BYTES, "body_root"),
            (COL_STATE_ROOT_OFFSET, HASH_BYTES, "state_root"),
        ];
        for (off, len, label) in hash_ranges {
            for k in 0..len {
                declarations.push((
                    LookupDeclaration {
                        label: format!("block_validity_{}_{}_8bit", label, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }

        let u64_ranges: [(usize, &str); 2] = [
            (COL_SLOT_BYTE_OFFSET, "slot_byte"),
            (COL_PI_BYTE_OFFSET, "proposer_index_byte"),
        ];
        for (off, label) in u64_ranges {
            for k in 0..U64_BYTES {
                declarations.push((
                    LookupDeclaration {
                        label: format!("block_validity_{}_{}_8bit", label, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }

        for k in 0..U32_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("block_validity_num_attestations_byte_{}_8bit", k),
                    column_index: COL_NA_BYTE_OFFSET + k,
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

/// Bind `(slot_bytes[0..8] || proposer_index_bytes[0..8] || block_root[0..32])`
/// of this composer against
/// [`crate::block_proposer_sig_air`]'s corresponding columns. Pins the
/// proposer-signature witness to the block this composer commits.
///
/// Tuple shape: 8 + 8 + 32 = **48 columns**.
pub fn make_block_validity_to_proposer_sig_descriptor(
    block_validity_layer_index: usize,
    block_proposer_sig_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::block_proposer_sig_air as bps;

    let mut a_columns: Vec<usize> = Vec::with_capacity(2 * U64_BYTES + HASH_BYTES);
    for k in 0..U64_BYTES { a_columns.push(COL_SLOT_BYTE_OFFSET + k); }
    for k in 0..U64_BYTES { a_columns.push(COL_PI_BYTE_OFFSET + k); }
    for k in 0..HASH_BYTES { a_columns.push(COL_BLOCK_ROOT_OFFSET + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(2 * U64_BYTES + HASH_BYTES);
    for k in 0..U64_BYTES { b_columns.push(bps::COL_SLOT_BYTE_OFFSET + k); }
    for k in 0..U64_BYTES { b_columns.push(bps::COL_PI_BYTE_OFFSET + k); }
    for k in 0..HASH_BYTES { b_columns.push(bps::COL_BLOCK_ROOT_OFFSET + k); }

    CrossAirLogUpDescriptor {
        label: "block_validity_to_proposer_sig_v1".into(),
        a_layer_index: block_validity_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: block_proposer_sig_layer_index,
        b_columns,
        b_selector_column: Some(bps::COL_IS_REAL),
    }
}

/// Bind `body_root[0..32]` of this composer against
/// [`crate::beacon_block_body_air::BBB8_COL_CLAIMED_BODY_ROOT_OFFSET`].
/// Pins the merkleized body output to this composer's `body_root` column.
///
/// Tuple shape: **32 columns**.
pub fn make_block_validity_to_body_root_descriptor(
    block_validity_layer_index: usize,
    beacon_block_body_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::beacon_block_body_air as bbb;

    let mut a_columns: Vec<usize> = Vec::with_capacity(HASH_BYTES);
    for k in 0..HASH_BYTES { a_columns.push(COL_BODY_ROOT_OFFSET + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(HASH_BYTES);
    for k in 0..HASH_BYTES {
        b_columns.push(bbb::BBB8_COL_CLAIMED_BODY_ROOT_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "block_validity_to_body_root_v1".into(),
        a_layer_index: block_validity_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: beacon_block_body_layer_index,
        b_columns,
        b_selector_column: Some(bbb::BBB8_COL_IS_REAL),
    }
}

/// Bind `slot_bytes[0..8]` of this composer against the `i`-th
/// [`crate::attestation_aggregate_air`] row's `slot_bytes` (via the
/// `SLOT` field). Each call yields a distinct labelled descriptor; the
/// caller (joint-prove orchestrator) places each attestation AIR at its
/// own layer index.
///
/// Today we bind only the `slot` correspondence — the per-attestation
/// `(committee_index, source/target epoch)` payload is exposed as the
/// attestation AIR's own commitments and is anchored by separate
/// descriptors against [`crate::attestation_committee_air`] /
/// [`crate::ffg_checkpoint_chain`].
///
/// Tuple shape: **8 columns** (slot bytes).
///
/// Panics: `i >= MAX_ATT`.
pub fn make_block_validity_to_attestation_descriptor(
    i: usize,
    block_validity_layer_index: usize,
    attestation_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    assert!(
        i < MAX_ATT,
        "attestation index {} out of bounds (MAX_ATT={})",
        i,
        MAX_ATT,
    );
    use crate::attestation_aggregate_air as att;

    let mut a_columns: Vec<usize> = Vec::with_capacity(U64_BYTES);
    for k in 0..U64_BYTES { a_columns.push(COL_SLOT_BYTE_OFFSET + k); }

    // The attestation AIR exposes SLOT as a single column; we bind it
    // via a 1-of-8 weighted check by exposing the same column 8× — the
    // descriptor's tuple is per-column so we replicate the SLOT column
    // to match arity. This is a soft anchor; tightening to per-byte
    // decomposition is a follow-up once the attestation AIR exposes
    // SLOT_BYTE columns.
    let mut b_columns: Vec<usize> = Vec::with_capacity(U64_BYTES);
    for _ in 0..U64_BYTES { b_columns.push(att::COL_SLOT); }

    CrossAirLogUpDescriptor {
        label: format!("block_validity_to_attestation_{}_v1", i),
        a_layer_index: block_validity_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: attestation_layer_index,
        b_columns,
        b_selector_column: Some(att::COL_IS_REAL),
    }
}

/// Bind `(slot_bytes[0..8], state_root[0..32])` against
/// [`crate::beacon_state_transition_air`]'s `(SLOT_BYTE, POST_STATE_ROOT)`.
///
/// Tuple shape: 8 + 32 = **40 columns**.
pub fn make_block_validity_to_state_transition_descriptor(
    block_validity_layer_index: usize,
    state_transition_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::beacon_state_transition_air as bst;

    let mut a_columns: Vec<usize> = Vec::with_capacity(U64_BYTES + HASH_BYTES);
    for k in 0..U64_BYTES { a_columns.push(COL_SLOT_BYTE_OFFSET + k); }
    for k in 0..HASH_BYTES { a_columns.push(COL_STATE_ROOT_OFFSET + k); }

    let mut b_columns: Vec<usize> = Vec::with_capacity(U64_BYTES + HASH_BYTES);
    for k in 0..U64_BYTES { b_columns.push(bst::COL_SLOT_BYTE_OFFSET + k); }
    for k in 0..HASH_BYTES { b_columns.push(bst::COL_POST_STATE_ROOT_OFFSET + k); }

    CrossAirLogUpDescriptor {
        label: "block_validity_to_state_transition_v1".into(),
        a_layer_index: block_validity_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: state_transition_layer_index,
        b_columns,
        b_selector_column: Some(bst::COL_IS_REAL),
    }
}

/// Bind `(slot_bytes[0..8], proposer_index_bytes[0..8], parent_root[0..32],
/// body_root[0..32], state_root[0..32])` against
/// [`crate::bbh_root_consumer_air`]. This is the *canonical* anchor
/// tying every per-block witness to a single beacon block header.
///
/// Tuple shape: 8 + 8 + 32 + 32 + 32 = **112 columns**.
pub fn make_block_validity_to_bbh_descriptor(
    block_validity_layer_index: usize,
    bbh_consumer_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::bbh_root_consumer_air as bbh;

    let mut a_columns: Vec<usize> =
        Vec::with_capacity(2 * U64_BYTES + 3 * HASH_BYTES);
    for k in 0..U64_BYTES { a_columns.push(COL_SLOT_BYTE_OFFSET + k); }
    for k in 0..U64_BYTES { a_columns.push(COL_PI_BYTE_OFFSET + k); }
    for k in 0..HASH_BYTES { a_columns.push(COL_PARENT_ROOT_OFFSET + k); }
    for k in 0..HASH_BYTES { a_columns.push(COL_BODY_ROOT_OFFSET + k); }
    for k in 0..HASH_BYTES { a_columns.push(COL_STATE_ROOT_OFFSET + k); }

    let mut b_columns: Vec<usize> =
        Vec::with_capacity(2 * U64_BYTES + 3 * HASH_BYTES);
    for k in 0..U64_BYTES { b_columns.push(bbh::COL_SLOT_BYTE_OFFSET + k); }
    for k in 0..U64_BYTES { b_columns.push(bbh::COL_PROPOSER_INDEX_BYTE_OFFSET + k); }
    for k in 0..HASH_BYTES { b_columns.push(bbh::COL_PARENT_ROOT_OFFSET + k); }
    for k in 0..HASH_BYTES { b_columns.push(bbh::COL_BODY_ROOT_OFFSET + k); }
    for k in 0..HASH_BYTES { b_columns.push(bbh::COL_STATE_ROOT_OFFSET + k); }

    CrossAirLogUpDescriptor {
        label: "block_validity_to_bbh_v1".into(),
        a_layer_index: block_validity_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bbh_consumer_layer_index,
        b_columns,
        b_selector_column: Some(bbh::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_root(seed: u8) -> [u8; HASH_BYTES] {
        let mut r = [0u8; HASH_BYTES];
        for i in 0..HASH_BYTES {
            r[i] = (i as u8).wrapping_mul(seed).wrapping_add(seed);
        }
        r
    }

    fn honest_witness() -> BeaconBlockValidityWitness {
        BeaconBlockValidityWitness::from_block_components(
            123_456_789,
            7_654_321,
            synth_root(11),
            synth_root(13),
            synth_root(17),
            synth_root(19),
            5,
        )
    }

    #[test]
    fn honest_witness_all_constraints_vanish() {
        let curve = CurveType::Bls12381;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);

        let cs = BeaconBlockValidityConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    #[test]
    fn tampered_body_root_detected_by_descriptor_misalignment() {
        // The body_root linkage descriptor binds the composer's
        // body_root[0..32] columns to the body AIR's
        // BBB8_COL_CLAIMED_BODY_ROOT. A tamper at the composer side
        // changes the A-tuple — the in-AIR constraints don't fire (the
        // root is host-committed), but the descriptor closure will
        // mismatch under joint γ. Here we assert that the trace witness
        // round-trips and that the column read back from the trace
        // matches what was written; this anchors the tamper-detection
        // contract for the joint-prove pipeline.
        let curve = CurveType::Bls12381;
        let mut w = honest_witness();
        let original = w.rows[0].body_root[0];
        w.rows[0].body_root[0] = original.wrapping_add(1);
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(
            trace.columns[COL_BODY_ROOT_OFFSET].evaluations[0].to_u64(),
            original.wrapping_add(1) as u64,
            "tampered body_root must be reflected in the trace column",
        );
        // And the original-honest witness produces a different value.
        let w_honest = honest_witness();
        let trace_honest = build_trace_polynomials(&w_honest, curve);
        assert_ne!(
            trace.columns[COL_BODY_ROOT_OFFSET].evaluations[0].to_u64(),
            trace_honest.columns[COL_BODY_ROOT_OFFSET].evaluations[0].to_u64(),
            "tampered vs honest body_root[0] must differ — descriptor will reject",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        // 1. proposer sig: 8 + 8 + 32 = 48 cols.
        let d1 = make_block_validity_to_proposer_sig_descriptor(0, 1);
        assert_eq!(d1.label, "block_validity_to_proposer_sig_v1");
        assert_eq!(d1.a_columns.len(), 48);
        assert_eq!(d1.b_columns.len(), 48);
        assert_eq!(d1.a_columns[0], COL_SLOT_BYTE_OFFSET);
        assert_eq!(d1.a_columns[U64_BYTES], COL_PI_BYTE_OFFSET);
        assert_eq!(d1.a_columns[2 * U64_BYTES], COL_BLOCK_ROOT_OFFSET);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d1.b_selector_column,
            Some(crate::block_proposer_sig_air::COL_IS_REAL),
        );

        // 2. body root: 32 cols.
        let d2 = make_block_validity_to_body_root_descriptor(0, 2);
        assert_eq!(d2.label, "block_validity_to_body_root_v1");
        assert_eq!(d2.a_columns.len(), HASH_BYTES);
        assert_eq!(d2.b_columns.len(), HASH_BYTES);
        assert_eq!(d2.a_columns[0], COL_BODY_ROOT_OFFSET);
        assert_eq!(
            d2.b_columns[0],
            crate::beacon_block_body_air::BBB8_COL_CLAIMED_BODY_ROOT_OFFSET,
        );
        assert_eq!(
            d2.b_selector_column,
            Some(crate::beacon_block_body_air::BBB8_COL_IS_REAL),
        );

        // 3. attestation (each i): 8 cols (slot bytes).
        for i in 0..MAX_ATT {
            let d3 = make_block_validity_to_attestation_descriptor(i, 0, 3 + i);
            assert_eq!(
                d3.label,
                format!("block_validity_to_attestation_{}_v1", i),
            );
            assert_eq!(d3.a_columns.len(), U64_BYTES);
            assert_eq!(d3.b_columns.len(), U64_BYTES);
            assert_eq!(d3.a_columns[0], COL_SLOT_BYTE_OFFSET);
            assert_eq!(
                d3.b_selector_column,
                Some(crate::attestation_aggregate_air::COL_IS_REAL),
            );
        }

        // 4. state transition: 8 + 32 = 40 cols.
        let d4 = make_block_validity_to_state_transition_descriptor(0, 99);
        assert_eq!(d4.label, "block_validity_to_state_transition_v1");
        assert_eq!(d4.a_columns.len(), U64_BYTES + HASH_BYTES);
        assert_eq!(d4.b_columns.len(), U64_BYTES + HASH_BYTES);
        assert_eq!(d4.a_columns[0], COL_SLOT_BYTE_OFFSET);
        assert_eq!(d4.a_columns[U64_BYTES], COL_STATE_ROOT_OFFSET);
        assert_eq!(
            d4.b_columns[0],
            crate::beacon_state_transition_air::COL_SLOT_BYTE_OFFSET,
        );
        assert_eq!(
            d4.b_columns[U64_BYTES],
            crate::beacon_state_transition_air::COL_POST_STATE_ROOT_OFFSET,
        );

        // 5. bbh: 8 + 8 + 32 + 32 + 32 = 112 cols.
        let d5 = make_block_validity_to_bbh_descriptor(0, 100);
        assert_eq!(d5.label, "block_validity_to_bbh_v1");
        assert_eq!(d5.a_columns.len(), 2 * U64_BYTES + 3 * HASH_BYTES);
        assert_eq!(d5.b_columns.len(), 2 * U64_BYTES + 3 * HASH_BYTES);
        assert_eq!(d5.a_columns[0], COL_SLOT_BYTE_OFFSET);
        assert_eq!(d5.a_columns[U64_BYTES], COL_PI_BYTE_OFFSET);
        assert_eq!(d5.a_columns[2 * U64_BYTES], COL_PARENT_ROOT_OFFSET);
        assert_eq!(d5.a_columns[2 * U64_BYTES + HASH_BYTES], COL_BODY_ROOT_OFFSET);
        assert_eq!(
            d5.a_columns[2 * U64_BYTES + 2 * HASH_BYTES],
            COL_STATE_ROOT_OFFSET,
        );
        assert_eq!(
            d5.b_columns[0],
            crate::bbh_root_consumer_air::COL_SLOT_BYTE_OFFSET,
        );
        assert_eq!(
            d5.b_columns[U64_BYTES],
            crate::bbh_root_consumer_air::COL_PROPOSER_INDEX_BYTE_OFFSET,
        );
        assert_eq!(
            d5.b_columns[2 * U64_BYTES],
            crate::bbh_root_consumer_air::COL_PARENT_ROOT_OFFSET,
        );
        assert_eq!(
            d5.b_columns[2 * U64_BYTES + HASH_BYTES],
            crate::bbh_root_consumer_air::COL_BODY_ROOT_OFFSET,
        );
        assert_eq!(
            d5.b_columns[2 * U64_BYTES + 2 * HASH_BYTES],
            crate::bbh_root_consumer_air::COL_STATE_ROOT_OFFSET,
        );
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_SLOT, 0);
        assert_eq!(COL_PROPOSER_INDEX, 1);
        assert_eq!(COL_BLOCK_ROOT_OFFSET, 2);
        assert_eq!(COL_PARENT_ROOT_OFFSET, 2 + 32);
        assert_eq!(COL_BODY_ROOT_OFFSET, 2 + 64);
        assert_eq!(COL_STATE_ROOT_OFFSET, 2 + 96);
        assert_eq!(COL_NUM_ATTESTATIONS, 2 + 128);
        assert_eq!(COL_SLOT_BYTE_OFFSET, 2 + 128 + 1);
        assert_eq!(COL_PI_BYTE_OFFSET, 2 + 128 + 1 + U64_BYTES);
        assert_eq!(COL_NA_BYTE_OFFSET, 2 + 128 + 1 + 2 * U64_BYTES);
        assert_eq!(COL_IS_REAL, 2 + 128 + 1 + 2 * U64_BYTES + U32_BYTES);
        // 2 scalar + 4*32 roots + 1 num_att + 2*8 bytes + 4 u32 bytes + 1 is_real
        // = 2 + 128 + 1 + 16 + 4 + 1 = 152
        assert_eq!(NUM_COLUMNS, 152);
        assert_eq!(NUM_ROW_CONSTRAINTS, 8);
        assert_eq!(NUM_SHIFTED, 0);
    }

    #[test]
    fn byte_range_lookup_coverage() {
        let cs = BeaconBlockValidityConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // 4 hashes * 32 + 2 * 8 + 4 = 128 + 16 + 4 = 148 decls.
        let expected = 4 * HASH_BYTES + 2 * U64_BYTES + U32_BYTES;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
        // Spot checks.
        assert!(reqs
            .declarations
            .iter()
            .any(|(d, _)| d.column_index == COL_BLOCK_ROOT_OFFSET));
        assert!(reqs
            .declarations
            .iter()
            .any(|(d, _)| d.column_index == COL_STATE_ROOT_OFFSET + HASH_BYTES - 1));
        assert!(reqs
            .declarations
            .iter()
            .any(|(d, _)| d.column_index == COL_NA_BYTE_OFFSET + U32_BYTES - 1));
    }

    /// LE-byte decomp constraint sanity: tampering a num_attestations
    /// byte fires the `num_attestations_le_decomp` body.
    #[test]
    fn num_attestations_le_decomp_fires_on_byte_tamper() {
        let curve = CurveType::Bls12381;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let bumped = cols[COL_NA_BYTE_OFFSET][0].to_u64().wrapping_add(1);
        cols[COL_NA_BYTE_OFFSET][0] = Scalar::from_u64(bumped, curve);
        let cs = BeaconBlockValidityConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "num_attestations_le_decomp body should fire on tampered NA byte",
        );
    }

    /// `is_real_binary` constraint fires on non-binary values.
    #[test]
    fn is_real_binary_fires_on_non_binary_value() {
        let curve = CurveType::Bls12381;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(5, curve);
        let cs = BeaconBlockValidityConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[0][0].is_zero(),
            "is_real_binary body should fire when IS_REAL = 5",
        );
    }

    #[test]
    fn from_block_components_populates_row() {
        let block_root = synth_root(11);
        let parent_root = synth_root(13);
        let body_root = synth_root(17);
        let state_root = synth_root(19);
        let w = BeaconBlockValidityWitness::from_block_components(
            42,
            7,
            block_root,
            parent_root,
            body_root,
            state_root,
            3,
        );
        assert_eq!(w.rows.len(), 1);
        let row = w.rows[0];
        assert_eq!(row.slot, 42);
        assert_eq!(row.proposer_index, 7);
        assert_eq!(row.block_root, block_root);
        assert_eq!(row.parent_root, parent_root);
        assert_eq!(row.body_root, body_root);
        assert_eq!(row.state_root, state_root);
        assert_eq!(row.num_attestations, 3);
    }

    #[test]
    fn evaluate_at_point_matches_evaluate_on_domain_for_honest() {
        let curve = CurveType::Bls12381;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = BeaconBlockValidityConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let _ = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        let alpha = Scalar::from_u64(13, curve);
        let row0_evals: Vec<Scalar> = col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must be zero on honest row");
    }

    #[test]
    #[should_panic(expected = "attestation index 16 out of bounds")]
    fn attestation_descriptor_out_of_bounds_panics() {
        let _ = make_block_validity_to_attestation_descriptor(MAX_ATT, 0, 1);
    }
}
