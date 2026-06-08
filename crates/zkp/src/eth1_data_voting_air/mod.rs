//! Eth1Data voting AIR — beacon-chain Phase C step.
//!
//! Proves that a beacon block's `body.eth1_data` vote is consistent
//! with the validator-signed beacon block (via cross-AIR LogUp to
//! [`crate::beacon_block_body_air`]'s eth1_data field commitment) and
//! that the voted `deposit_root` matches the deposit contract's merkle
//! root (via cross-AIR LogUp to
//! [`crate::deposit_tree_air`]).
//!
//! # Eth1Data shape
//!
//! ```text
//! Eth1Data {
//!   deposit_root:  Bytes32,   // root of the deposit contract trie
//!   deposit_count: u64,       // total deposits ever (LE-serialized)
//!   block_hash:    Bytes32,   // execution block hash of the Eth1 vote
//! }
//! ```
//!
//! Each beacon block includes a `body.eth1_data` vote. Votes are
//! tallied across an `EPOCHS_PER_ETH1_VOTING_PERIOD * SLOTS_PER_EPOCH`
//! window. A vote becomes the new canonical `state.eth1_data` once it
//! crosses a `slots_per_period / 2` threshold (witnessed as
//! `vote_count_threshold`).
//!
//! # Column layout (single row per Eth1Data observation)
//!
//! - `deposit_root[0..32]`            cols 0..32
//! - `deposit_count_bytes[0..8]`      cols 32..40  (LE bytes)
//! - `deposit_count`                  col 40       (= Σ bytes * 2^(8k))
//! - `block_hash[0..32]`              cols 41..73
//! - `vote_count`                     col 73
//! - `vote_count_threshold`           col 74
//! - `vote_margin`                    col 75       (= vote_count - threshold)
//! - `slot_bytes[0..8]`               cols 76..84  (LE bytes)
//! - `slot`                           col 84       (= Σ bytes * 2^(8k))
//! - `eth1_data_htr[0..32]`           cols 85..117
//! - `is_winning_vote`                col 117
//! - `is_real`                        col 118
//!
//! # Row-local constraints
//!
//! 0. `is_real` binary
//! 1. `is_winning_vote` binary
//! 2. `deposit_count = Σ deposit_count_bytes[k] · 2^(8k)`
//! 3. `slot           = Σ slot_bytes[k]         · 2^(8k)`
//! 4. `is_winning_vote · (vote_count − vote_count_threshold − vote_margin) = 0`
//! 5. `is_winning_vote · (1 − is_real) = 0` (winning implies real)
//!
//! 8-bit range checks on all byte columns
//! (deposit_root 32 + deposit_count_bytes 8 + block_hash 32 +
//!  slot_bytes 8 + eth1_data_htr 32 = 112 lookups).
//!
//! # Linkage descriptors
//!
//! - [`make_eth1_data_to_deposit_tree_descriptor`] — 32-col tuple binding
//!   this row's `deposit_root` ↔ deposit_tree_air's `DEPOSIT_ROOT`
//!   column (gated by `IS_REAL` here, `IS_TOP` there).
//! - [`make_eth1_data_to_bbb_descriptor`] — 72-col tuple binding
//!   `(deposit_root || deposit_count_bytes || block_hash)` to a
//!   block-body-side column range parameterized by byte offset and
//!   selector column (mirrors `make_deposit_root_to_block_descriptor`).
//! - [`make_eth1_data_to_htr_descriptor`] — 32-col tuple binding the
//!   `eth1_data_htr` bytes to a SHA-256 invocation's `OUTPUT_BYTE`
//!   range, witnessing that the host-side HTR closure (the final
//!   merkle pair of the 4-leaf Eth1Data tree) is consistent with the
//!   exposed root.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const DEPOSIT_ROOT_LEN: usize = 32;
pub const BLOCK_HASH_LEN: usize = 32;
pub const HTR_LEN: usize = 32;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_DEPOSIT_ROOT_OFFSET: usize = 0;
pub const COL_DEPOSIT_COUNT_BYTES_OFFSET: usize =
    COL_DEPOSIT_ROOT_OFFSET + DEPOSIT_ROOT_LEN; // 32
pub const COL_DEPOSIT_COUNT: usize =
    COL_DEPOSIT_COUNT_BYTES_OFFSET + U64_BYTES; // 40
pub const COL_BLOCK_HASH_OFFSET: usize = COL_DEPOSIT_COUNT + 1; // 41
pub const COL_VOTE_COUNT: usize = COL_BLOCK_HASH_OFFSET + BLOCK_HASH_LEN; // 73
pub const COL_VOTE_COUNT_THRESHOLD: usize = COL_VOTE_COUNT + 1; // 74
pub const COL_VOTE_MARGIN: usize = COL_VOTE_COUNT_THRESHOLD + 1; // 75
pub const COL_SLOT_BYTES_OFFSET: usize = COL_VOTE_MARGIN + 1; // 76
pub const COL_SLOT: usize = COL_SLOT_BYTES_OFFSET + U64_BYTES; // 84
pub const COL_ETH1_DATA_HTR_OFFSET: usize = COL_SLOT + 1; // 85
pub const COL_IS_WINNING_VOTE: usize = COL_ETH1_DATA_HTR_OFFSET + HTR_LEN; // 117
pub const COL_IS_REAL: usize = COL_IS_WINNING_VOTE + 1; // 118
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 119

pub const NUM_ROW_CONSTRAINTS: usize = 6;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

/// Lite local mirror of the consensus-spec `Eth1Data` container. We
/// keep this local to avoid coupling to a full beacon-chain `Eth1Data`
/// import; the field layout matches the spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Eth1DataLite {
    pub deposit_root: [u8; DEPOSIT_ROOT_LEN],
    pub deposit_count: u64,
    pub block_hash: [u8; BLOCK_HASH_LEN],
}

impl Eth1DataLite {
    /// SSZ `hash_tree_root` of an `Eth1Data` container: a 3-field
    /// container padded to 4 leaves (depth 2). Leaves are
    /// `deposit_root`, `deposit_count` LE-serialised + zero-padded to
    /// 32 bytes, `block_hash`, and `ZERO_CHUNK`.
    pub fn hash_tree_root(&self) -> [u8; HTR_LEN] {
        let mut count_leaf = [0u8; 32];
        count_leaf[..U64_BYTES].copy_from_slice(&self.deposit_count.to_le_bytes());
        let zero = [0u8; 32];
        let left = crate::sha256::sha256_pair(&self.deposit_root, &count_leaf);
        let right = crate::sha256::sha256_pair(&self.block_hash, &zero);
        crate::sha256::sha256_pair(&left, &right)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Eth1DataVotingRow {
    pub deposit_root: [u8; DEPOSIT_ROOT_LEN],
    pub deposit_count: u64,
    pub block_hash: [u8; BLOCK_HASH_LEN],
    pub vote_count: u32,
    pub vote_count_threshold: u32,
    pub slot: u64,
    pub eth1_data_htr: [u8; HTR_LEN],
    pub is_winning_vote: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Eth1DataVotingWitness {
    pub observations: Vec<Eth1DataVotingRow>,
}

impl Eth1DataVotingWitness {
    pub fn from_rows(observations: Vec<Eth1DataVotingRow>) -> Self {
        Self { observations }
    }
}

/// Host-side builder: turn an `Eth1DataLite` + scalar context into a
/// single-row voting witness. `vote_count_threshold` is derived as
/// `slots_per_period / 2`. `is_winning_vote = (vote_count >
/// threshold)` and (necessarily) `vote_count >= threshold + 1`.
///
/// `vote_margin` (carried in the row) is `vote_count.saturating_sub(
/// threshold)`. On a winning vote `vote_margin >= 1`; the algebraic
/// row constraint pins the linear relation. For losing rows the margin
/// is left at zero and `is_winning_vote = 0` makes the constraint
/// vacuous.
pub fn from_eth1_data(
    data: Eth1DataLite,
    slot: u64,
    vote_count: u32,
    slots_per_period: u32,
) -> Eth1DataVotingWitness {
    let threshold = slots_per_period / 2;
    let is_winning_vote = vote_count > threshold;
    let row = Eth1DataVotingRow {
        deposit_root: data.deposit_root,
        deposit_count: data.deposit_count,
        block_hash: data.block_hash,
        vote_count,
        vote_count_threshold: threshold,
        slot,
        eth1_data_htr: data.hash_tree_root(),
        is_winning_vote,
    };
    Eth1DataVotingWitness::from_rows(vec![row])
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Eth1DataVotingWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.observations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.observations.iter().enumerate() {
        for k in 0..DEPOSIT_ROOT_LEN {
            columns[COL_DEPOSIT_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.deposit_root[k] as u64, curve);
        }
        let count_bytes = row.deposit_count.to_le_bytes();
        for k in 0..U64_BYTES {
            columns[COL_DEPOSIT_COUNT_BYTES_OFFSET + k][i] =
                Scalar::from_u64(count_bytes[k] as u64, curve);
        }
        columns[COL_DEPOSIT_COUNT][i] = Scalar::from_u64(row.deposit_count, curve);

        for k in 0..BLOCK_HASH_LEN {
            columns[COL_BLOCK_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.block_hash[k] as u64, curve);
        }
        columns[COL_VOTE_COUNT][i] = Scalar::from_u64(row.vote_count as u64, curve);
        columns[COL_VOTE_COUNT_THRESHOLD][i] =
            Scalar::from_u64(row.vote_count_threshold as u64, curve);
        let margin = (row.vote_count as i64) - (row.vote_count_threshold as i64);
        let margin_u64 = if margin < 0 { 0u64 } else { margin as u64 };
        columns[COL_VOTE_MARGIN][i] = Scalar::from_u64(margin_u64, curve);

        let slot_bytes = row.slot.to_le_bytes();
        for k in 0..U64_BYTES {
            columns[COL_SLOT_BYTES_OFFSET + k][i] =
                Scalar::from_u64(slot_bytes[k] as u64, curve);
        }
        columns[COL_SLOT][i] = Scalar::from_u64(row.slot, curve);

        for k in 0..HTR_LEN {
            columns[COL_ETH1_DATA_HTR_OFFSET + k][i] =
                Scalar::from_u64(row.eth1_data_htr[k] as u64, curve);
        }
        columns[COL_IS_WINNING_VOTE][i] =
            if row.is_winning_vote { one.clone() } else { zero.clone() };
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

// ─── Constraint system ─────────────────────────────────────────────────

pub struct Eth1DataVotingConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Eth1DataVotingConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// `(byte_idx, weight)` weights for LE u64 decomp.
fn le_u64_decomp_weights() -> Vec<(usize, u64)> {
    (0..U64_BYTES).map(|k| (k, 1u64 << (8 * k))).collect()
}

impl VmConstraintSystem for Eth1DataVotingConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_winning_vote_binary".into(),
            "deposit_count_le_decomp".into(),
            "slot_le_decomp".into(),
            "winning_vote_margin_eq".into(),
            "winning_implies_real".into(),
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
        // 1: is_winning_vote binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_WINNING_VOTE][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 2: deposit_count LE decomp.
        {
            let weights = le_u64_decomp_weights();
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (k, w) in &weights {
                    let b = &columns[COL_DEPOSIT_COUNT_BYTES_OFFSET + k][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*w, curve)));
                }
                c[r] = columns[COL_DEPOSIT_COUNT][r].sub(&sum);
            }
            out.push(c);
        }
        // 3: slot LE decomp.
        {
            let weights = le_u64_decomp_weights();
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (k, w) in &weights {
                    let b = &columns[COL_SLOT_BYTES_OFFSET + k][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*w, curve)));
                }
                c[r] = columns[COL_SLOT][r].sub(&sum);
            }
            out.push(c);
        }
        // 4: is_winning_vote · (vote_count - threshold - margin) = 0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let w = &columns[COL_IS_WINNING_VOTE][r];
                let body = columns[COL_VOTE_COUNT][r]
                    .sub(&columns[COL_VOTE_COUNT_THRESHOLD][r])
                    .sub(&columns[COL_VOTE_MARGIN][r]);
                c[r] = w.mul(&body);
            }
            out.push(c);
        }
        // 5: is_winning_vote · (1 - is_real) = 0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let w = &columns[COL_IS_WINNING_VOTE][r];
                let body = one.sub(&columns[COL_IS_REAL][r]);
                c[r] = w.mul(&body);
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

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1
        {
            let v = &col_evals[COL_IS_WINNING_VOTE];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2
        {
            let weights = le_u64_decomp_weights();
            let mut sum = Scalar::zero(curve);
            for (k, w) in &weights {
                sum = sum.add(
                    &col_evals[COL_DEPOSIT_COUNT_BYTES_OFFSET + k]
                        .mul(&Scalar::from_u64(*w, curve)),
                );
            }
            let body = col_evals[COL_DEPOSIT_COUNT].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3
        {
            let weights = le_u64_decomp_weights();
            let mut sum = Scalar::zero(curve);
            for (k, w) in &weights {
                sum = sum.add(
                    &col_evals[COL_SLOT_BYTES_OFFSET + k]
                        .mul(&Scalar::from_u64(*w, curve)),
                );
            }
            let body = col_evals[COL_SLOT].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4
        {
            let w = &col_evals[COL_IS_WINNING_VOTE];
            let body = col_evals[COL_VOTE_COUNT]
                .sub(&col_evals[COL_VOTE_COUNT_THRESHOLD])
                .sub(&col_evals[COL_VOTE_MARGIN]);
            acc = acc.add(&alpha_pow.mul(&w.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 5
        {
            let w = &col_evals[COL_IS_WINNING_VOTE];
            let body = one.sub(&col_evals[COL_IS_REAL]);
            acc = acc.add(&alpha_pow.mul(&w.mul(&body)));
            // alpha_pow no longer used.
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

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // 0
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1
        {
            let v = &col_coeffs[COL_IS_WINNING_VOTE];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2
        {
            let weights = le_u64_decomp_weights();
            let mut sum = vec![Scalar::zero(curve)];
            for (k, w) in &weights {
                let b = &col_coeffs[COL_DEPOSIT_COUNT_BYTES_OFFSET + k];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*w, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_DEPOSIT_COUNT], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3
        {
            let weights = le_u64_decomp_weights();
            let mut sum = vec![Scalar::zero(curve)];
            for (k, w) in &weights {
                let b = &col_coeffs[COL_SLOT_BYTES_OFFSET + k];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*w, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_SLOT], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4
        {
            let w = &col_coeffs[COL_IS_WINNING_VOTE];
            let vc = &col_coeffs[COL_VOTE_COUNT];
            let th = &col_coeffs[COL_VOTE_COUNT_THRESHOLD];
            let mg = &col_coeffs[COL_VOTE_MARGIN];
            let body = poly_sub(&poly_sub(vc, th, curve), mg, curve);
            let term = poly_mul(w, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&term, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 5
        {
            let w = &col_coeffs[COL_IS_WINNING_VOTE];
            let is_real = &col_coeffs[COL_IS_REAL];
            let body = poly_sub(&one_poly, is_real, curve);
            let term = poly_mul(w, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&term, &alpha_pow), curve);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_IS_WINNING_VOTE]
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
        for k in 0..DEPOSIT_ROOT_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("deposit_root_{}_8bit", k),
                    column_index: COL_DEPOSIT_ROOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("deposit_count_byte_{}_8bit", k),
                    column_index: COL_DEPOSIT_COUNT_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..BLOCK_HASH_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("block_hash_{}_8bit", k),
                    column_index: COL_BLOCK_HASH_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("slot_byte_{}_8bit", k),
                    column_index: COL_SLOT_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..HTR_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("eth1_data_htr_{}_8bit", k),
                    column_index: COL_ETH1_DATA_HTR_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Bind this gadget's `deposit_root[0..32]` ↔
/// `deposit_tree_air::COL_DEPOSIT_ROOT_OFFSET[0..32]`. 32-col tuple.
/// A-selector = `IS_REAL` (this AIR), B-selector = `IS_TOP` (deposit_tree).
pub fn make_eth1_data_to_deposit_tree_descriptor(
    eth1_layer_index: usize,
    deposit_tree_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::deposit_tree_air as dt;
    let a_columns: Vec<usize> =
        (0..DEPOSIT_ROOT_LEN).map(|k| COL_DEPOSIT_ROOT_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..DEPOSIT_ROOT_LEN).map(|k| dt::COL_DEPOSIT_ROOT_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "eth1_data_to_deposit_tree_v1".into(),
        a_layer_index: eth1_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: deposit_tree_layer_index,
        b_columns,
        b_selector_column: Some(dt::COL_IS_TOP),
    }
}

/// Bind `(deposit_root[0..32] || deposit_count_bytes[0..8] ||
/// block_hash[0..32])` (72-col tuple) to a beacon-block-body-side
/// column range. The B-side `bbb_eth1_root_byte_offset` parameter is
/// the byte offset on the BBB AIR where the (currently host-side)
/// flattened `eth1_data` tuple will be exposed; once `beacon_block_body_air`
/// publishes the eth1_data fields explicitly this becomes a direct
/// link. Mirrors the parameterized
/// `make_deposit_root_to_block_descriptor` pattern.
pub fn make_eth1_data_to_bbb_descriptor(
    eth1_layer_index: usize,
    bbb_layer_index: usize,
    bbb_eth1_data_byte_offset: usize,
    bbb_selector_column: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> =
        Vec::with_capacity(DEPOSIT_ROOT_LEN + U64_BYTES + BLOCK_HASH_LEN);
    for k in 0..DEPOSIT_ROOT_LEN {
        a_columns.push(COL_DEPOSIT_ROOT_OFFSET + k);
    }
    for k in 0..U64_BYTES {
        a_columns.push(COL_DEPOSIT_COUNT_BYTES_OFFSET + k);
    }
    for k in 0..BLOCK_HASH_LEN {
        a_columns.push(COL_BLOCK_HASH_OFFSET + k);
    }

    let mut b_columns: Vec<usize> =
        Vec::with_capacity(DEPOSIT_ROOT_LEN + U64_BYTES + BLOCK_HASH_LEN);
    for k in 0..(DEPOSIT_ROOT_LEN + U64_BYTES + BLOCK_HASH_LEN) {
        b_columns.push(bbb_eth1_data_byte_offset + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "eth1_data_to_bbb_v1".into(),
        a_layer_index: eth1_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bbb_layer_index,
        b_columns,
        b_selector_column: Some(bbb_selector_column),
    }
}

/// Bind `eth1_data_htr[0..32]` ↔ `sha256_extract::COL_OUTPUT_BYTE`. The
/// witnessed B-side row holds the final pair invocation of the 4-leaf
/// Eth1Data merkleization (`sha256_pair(left_child, right_child) =
/// eth1_data_htr`); the host-side trace builder for sha256_extract is
/// expected to include that pair as one of its rows. 32-col tuple.
pub fn make_eth1_data_to_htr_descriptor(
    eth1_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;
    let a_columns: Vec<usize> =
        (0..HTR_LEN).map(|k| COL_ETH1_DATA_HTR_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..HTR_LEN).map(|k| se::COL_OUTPUT_BYTE_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "eth1_data_to_htr_v1".into(),
        a_layer_index: eth1_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Eth1DataLite {
        Eth1DataLite {
            deposit_root: [0xab; 32],
            deposit_count: 0x0102_0304_0506_0708,
            block_hash: [0xcd; 32],
        }
    }

    #[test]
    fn column_layout_pinned() {
        // If any of these offsets shift, downstream descriptors must
        // be re-audited. This test pins them.
        assert_eq!(COL_DEPOSIT_ROOT_OFFSET, 0);
        assert_eq!(COL_DEPOSIT_COUNT_BYTES_OFFSET, 32);
        assert_eq!(COL_DEPOSIT_COUNT, 40);
        assert_eq!(COL_BLOCK_HASH_OFFSET, 41);
        assert_eq!(COL_VOTE_COUNT, 73);
        assert_eq!(COL_VOTE_COUNT_THRESHOLD, 74);
        assert_eq!(COL_VOTE_MARGIN, 75);
        assert_eq!(COL_SLOT_BYTES_OFFSET, 76);
        assert_eq!(COL_SLOT, 84);
        assert_eq!(COL_ETH1_DATA_HTR_OFFSET, 85);
        assert_eq!(COL_IS_WINNING_VOTE, 117);
        assert_eq!(COL_IS_REAL, 118);
        assert_eq!(NUM_COLUMNS, 119);
        assert_eq!(NUM_ROW_CONSTRAINTS, 6);
    }

    #[test]
    fn constraints_zero_on_honest_winning_vote() {
        // slots_per_period = 64 → threshold = 32. vote_count = 40 (>32) wins.
        let w = from_eth1_data(sample_data(), 12_345, 40, 64);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Eth1DataVotingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} row {} = {:?}, expected zero",
                    i, r, val,
                );
            }
        }
        // Sanity: is_winning_vote = 1.
        assert_eq!(
            trace.columns[COL_IS_WINNING_VOTE].evaluations[0].to_u64(),
            1,
        );
    }

    #[test]
    fn losing_vote_records_zero_winning_selector() {
        // vote_count = 10, threshold = 32 → losing.
        let w = from_eth1_data(sample_data(), 12_345, 10, 64);
        assert!(!w.observations[0].is_winning_vote);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Eth1DataVotingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // All constraints satisfied (vote_margin saturates to 0 on
        // losing rows; the winning-vote-margin constraint is gated by
        // is_winning_vote=0 so it vacuously holds).
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} row {} = {:?} (losing-vote witness)",
                    i, r, val,
                );
            }
        }
        assert_eq!(
            trace.columns[COL_IS_WINNING_VOTE].evaluations[0].to_u64(),
            0,
        );
    }

    #[test]
    fn tampered_winning_margin_fires_constraint() {
        let w = from_eth1_data(sample_data(), 12_345, 40, 64);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper vote_margin: should equal vote_count - threshold = 8;
        // overwrite with 7 (matches a lower true vote_count).
        cols[COL_VOTE_MARGIN][0] = Scalar::from_u64(7, CurveType::Bls48581);
        let cs = Eth1DataVotingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint index 4 = winning_vote_margin_eq.
        assert!(
            !results[4][0].is_zero(),
            "winning_vote_margin_eq must fire on tampered margin",
        );
    }

    #[test]
    fn tampered_deposit_count_byte_fires_le_decomp() {
        let w = from_eth1_data(sample_data(), 7, 40, 64);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_DEPOSIT_COUNT_BYTES_OFFSET][0] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = Eth1DataVotingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint index 2 = deposit_count_le_decomp.
        assert!(
            !results[2][0].is_zero(),
            "deposit_count LE decomp must fire on tampered byte",
        );
    }

    #[test]
    fn tampered_is_real_winning_fires_implication() {
        let w = from_eth1_data(sample_data(), 7, 40, 64);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Set is_real = 0 while is_winning_vote = 1: constraint 5 must fire.
        cols[COL_IS_REAL][0] = Scalar::zero(CurveType::Bls48581);
        let cs = Eth1DataVotingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[5][0].is_zero(),
            "winning_implies_real must fire",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_eth1_data_to_deposit_tree_descriptor(0, 1);
        assert_eq!(d1.label, "eth1_data_to_deposit_tree_v1");
        assert_eq!(d1.a_columns.len(), 32);
        assert_eq!(d1.b_columns.len(), 32);
        for k in 0..DEPOSIT_ROOT_LEN {
            assert_eq!(d1.a_columns[k], COL_DEPOSIT_ROOT_OFFSET + k);
            assert_eq!(
                d1.b_columns[k],
                crate::deposit_tree_air::COL_DEPOSIT_ROOT_OFFSET + k,
            );
        }

        let d2 = make_eth1_data_to_bbb_descriptor(0, 2, 500, 999);
        assert_eq!(d2.label, "eth1_data_to_bbb_v1");
        assert_eq!(d2.a_columns.len(), 32 + 8 + 32);
        assert_eq!(d2.b_columns.len(), 32 + 8 + 32);
        assert_eq!(d2.b_selector_column, Some(999));
        // First 32 = deposit_root range.
        assert_eq!(d2.a_columns[0], COL_DEPOSIT_ROOT_OFFSET);
        assert_eq!(d2.a_columns[31], COL_DEPOSIT_ROOT_OFFSET + 31);
        // Next 8 = deposit_count bytes.
        assert_eq!(d2.a_columns[32], COL_DEPOSIT_COUNT_BYTES_OFFSET);
        assert_eq!(d2.a_columns[39], COL_DEPOSIT_COUNT_BYTES_OFFSET + 7);
        // Next 32 = block_hash.
        assert_eq!(d2.a_columns[40], COL_BLOCK_HASH_OFFSET);
        assert_eq!(d2.a_columns[71], COL_BLOCK_HASH_OFFSET + 31);
        // B side is contiguous from byte_offset.
        for k in 0..72 {
            assert_eq!(d2.b_columns[k], 500 + k);
        }

        let d3 = make_eth1_data_to_htr_descriptor(0, 3);
        assert_eq!(d3.label, "eth1_data_to_htr_v1");
        assert_eq!(d3.a_columns.len(), 32);
        assert_eq!(d3.b_columns.len(), 32);
        for k in 0..HTR_LEN {
            assert_eq!(d3.a_columns[k], COL_ETH1_DATA_HTR_OFFSET + k);
            assert_eq!(
                d3.b_columns[k],
                crate::sha256_extract::COL_OUTPUT_BYTE_OFFSET + k,
            );
        }
    }

    #[test]
    fn htr_matches_manual_4_leaf_merkleization() {
        let data = sample_data();
        let computed = data.hash_tree_root();

        let mut count_leaf = [0u8; 32];
        count_leaf[..8].copy_from_slice(&data.deposit_count.to_le_bytes());
        let zero = [0u8; 32];
        let left = crate::sha256::sha256_pair(&data.deposit_root, &count_leaf);
        let right = crate::sha256::sha256_pair(&data.block_hash, &zero);
        let expected = crate::sha256::sha256_pair(&left, &right);
        assert_eq!(computed, expected);
    }

    #[test]
    fn from_eth1_data_threshold_boundary_loses() {
        // vote_count == threshold (not strictly greater) → losing.
        let w = from_eth1_data(sample_data(), 1, 32, 64);
        assert!(!w.observations[0].is_winning_vote);
        assert_eq!(w.observations[0].vote_count_threshold, 32);
    }

    #[test]
    fn deposit_count_byte_order_is_little_endian() {
        let data = Eth1DataLite {
            deposit_root: [0; 32],
            deposit_count: 0x0102_0304_0506_0708,
            block_hash: [0; 32],
        };
        let w = from_eth1_data(data, 0, 0, 64);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // LE → byte 0 = 0x08 (LSB), byte 7 = 0x01 (MSB).
        assert_eq!(
            trace.columns[COL_DEPOSIT_COUNT_BYTES_OFFSET].evaluations[0].to_u64(),
            0x08,
        );
        assert_eq!(
            trace.columns[COL_DEPOSIT_COUNT_BYTES_OFFSET + 7].evaluations[0].to_u64(),
            0x01,
        );
    }
}
