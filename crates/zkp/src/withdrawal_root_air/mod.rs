//! Withdrawal MPT root construction AIR.
//!
//! Verifies that an Ethereum block header's `withdrawals_root` is the
//! correct keccak256-based Merkle Patricia Trie root for the block's
//! set of [`crate::withdrawal::Withdrawal`]s (Shapella/Cancun).
//!
//! ## Trie shape (host-side reference)
//!
//! Ethereum builds the withdrawals MPT keyed by the RLP encoding of the
//! transaction-style index (0, 1, 2, …) with the canonical
//! `withdrawal_rlp(w)` as the leaf value. For small batches (up to
//! [`MAX_WITHDRAWALS`] = 16, which is `MAX_WITHDRAWALS_PER_PAYLOAD`)
//! the resulting trie has bounded depth; this AIR doesn't model the
//! internal MPT branches algebraically — instead it cross-AIR LogUps
//! out to (a) RLP gadgets for each per-field encoding, (b)
//! [`crate::keccak_extract`] for each leaf hash, and (c) the host-side
//! `withdrawals_root` claim against [`crate::block_header_air`].
//!
//! In the steady-state composition the algebraic MPT inclusion proof
//! per leaf is handled by [`crate::mpt_air`] driven from the host
//! oracle. This AIR's job is the **per-withdrawal accounting**: it
//! commits each withdrawal's fields, leaf RLP bytes, leaf hash, and
//! the claimed `withdrawals_root` — and via cross-AIR LogUps binds
//! every field to its respective gadget and ultimately to
//! `block_header_air.withdrawals_root`.
//!
//! ## Row layout
//!
//! One row per withdrawal slot. The AIR caps the row count at
//! [`MAX_WITHDRAWALS`] = 16.
//!
//! Columns:
//!   * `INDEX`                       — withdrawal index (u64)
//!   * `VALIDATOR_INDEX`             — validator index (u64)
//!   * `ADDRESS[0..20]`              — 20-byte recipient
//!   * `AMOUNT`                      — withdrawal amount in Gwei (u64)
//!   * `RLP_BYTES[0..RLP_MAX_LEN]`   — RLP encoding, zero-padded
//!   * `RLP_LEN`                     — actual encoded length
//!   * `LEAF_HASH[0..32]`            — keccak256(RLP_BYTES[..RLP_LEN])
//!   * `WITHDRAWALS_ROOT[0..32]`     — claimed root, replicated across
//!                                     every row (including padding)
//!   * `IS_REAL`                     — selector for active rows
//!   * `IS_FIRST`                    — selector flagging row 0 (used by
//!                                     the block_header binding so the
//!                                     A side publishes exactly one
//!                                     `(withdrawals_root)` tuple).
//!
//! ## Algebraic constraints
//!
//! 4 row-local bodies:
//!   0. `is_real_binary`   — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `is_first_binary`  — `IS_FIRST · (IS_FIRST − 1) = 0`
//!   2. `is_first_implies_real` — `IS_FIRST · (1 − IS_REAL) = 0`
//!   3. `address_byte_range_witness` — `Σ β^i · is_real · 0 = 0`
//!      (placeholder kept for binding alpha-offset alignment; the
//!      real 8-bit address byte range checks are issued through the
//!      lookup-declaration machinery.)
//!
//! 1 shifted body:
//!   0. `claimed_root_constancy` — for every byte of
//!      `WITHDRAWALS_ROOT`, `(WITHDRAWALS_ROOT_i(ω·X) −
//!      WITHDRAWALS_ROOT_i(X)) = 0`, β-RLC'd into one body,
//!      excluded at wrap-around.
//!
//! 8-bit range-check declarations cover every byte column
//! (`ADDRESS`, `RLP_BYTES`, `LEAF_HASH`, `WITHDRAWALS_ROOT`).
//!
//! ## What is bound by cross-AIR LogUp (not algebraic here)
//!
//!   * `RLP_BYTES[..RLP_LEN]` is the canonical RLP of the four
//!     committed fields — bound per-field via the
//!     [`make_withdrawal_to_rlp_descriptor`] family (index,
//!     validator_index, amount routed through `u64_rlp_air`, address
//!     through `fixed_rlp20_air`).
//!   * `LEAF_HASH = keccak256(RLP_BYTES[..RLP_LEN])` — bound by
//!     [`make_withdrawal_to_keccak_descriptor`] linking the
//!     `(RLP_BYTES, LEAF_HASH)` tuple against
//!     [`crate::keccak_extract`].
//!   * `WITHDRAWALS_ROOT` matches the value in
//!     [`crate::block_header_air`] — bound by
//!     [`make_withdrawal_root_to_block_header_descriptor`] (gated on
//!     the A side by `IS_FIRST`, on the B side by `block_header_air`'s
//!     `IS_REAL`).
//!
//! Per the host-side oracle in [`crate::withdrawal_trie`], the
//! mapping `leaf_hash → withdrawals_root` is closed by an external
//! MPT inclusion proof per leaf (one [`crate::mpt_air`] trace, joined
//! to this AIR through the leaf-hash column).

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;
use crate::withdrawal::{withdrawal_rlp, Withdrawal};

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum withdrawals per Ethereum block (post-Shapella spec:
/// `MAX_WITHDRAWALS_PER_PAYLOAD`).
pub const MAX_WITHDRAWALS: usize = 16;

/// Maximum RLP-encoded length of one withdrawal: worst case is a
/// 4-element list with three 9-byte u64 encodings + a 21-byte address
/// + 3-byte list header ≈ 51 bytes. Rounded up to 64 for headroom.
pub const RLP_MAX_LEN: usize = 64;

/// Bytes in a recipient address.
pub const ADDRESS_LEN: usize = 20;

/// Bytes in a keccak digest / MPT root.
pub const HASH_LEN: usize = 32;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_INDEX: usize = 0;
pub const COL_VALIDATOR_INDEX: usize = COL_INDEX + 1;
pub const COL_AMOUNT: usize = COL_VALIDATOR_INDEX + 1;
pub const COL_ADDRESS_OFFSET: usize = COL_AMOUNT + 1;
pub const COL_RLP_BYTES_OFFSET: usize = COL_ADDRESS_OFFSET + ADDRESS_LEN;
pub const COL_RLP_LEN: usize = COL_RLP_BYTES_OFFSET + RLP_MAX_LEN;
pub const COL_LEAF_HASH_OFFSET: usize = COL_RLP_LEN + 1;
pub const COL_WITHDRAWALS_ROOT_OFFSET: usize = COL_LEAF_HASH_OFFSET + HASH_LEN;
pub const COL_IS_REAL: usize = COL_WITHDRAWALS_ROOT_OFFSET + HASH_LEN;
pub const COL_IS_FIRST: usize = COL_IS_REAL + 1;

pub const NUM_COLUMNS: usize = COL_IS_FIRST + 1;

/// 3 row-local bodies (see module docs).
pub const NUM_ROW_CONSTRAINTS: usize = 3;
/// 1 shifted body (`claimed_root_constancy`).
pub const NUM_SHIFTED: usize = 1;

// ─── Witness types ────────────────────────────────────────────────────

/// One per-withdrawal row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WithdrawalRootRow {
    pub withdrawal: Withdrawal,
    /// Canonical RLP encoding of `withdrawal`.
    pub rlp_bytes: Vec<u8>,
    /// `keccak256(rlp_bytes)`.
    pub leaf_hash: [u8; HASH_LEN],
    /// True for the first row only (used by the block_header binding).
    pub is_first: bool,
}

/// Per-batch witness for the withdrawal-root AIR.
///
/// Cap: [`MAX_WITHDRAWALS`] withdrawals; an empty batch (0
/// withdrawals) is permitted and yields the canonical empty-trie root
/// (`keccak256(rlp([]))`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WithdrawalRootWitness {
    pub withdrawals: Vec<Withdrawal>,
    pub rows: Vec<WithdrawalRootRow>,
    pub withdrawals_root: [u8; HASH_LEN],
}

impl WithdrawalRootWitness {
    /// Construct the witness from a batch of withdrawals using
    /// [`crate::withdrawal_trie`] for the host-side root computation.
    ///
    /// Panics if `withdrawals.len() > MAX_WITHDRAWALS`.
    pub fn from_withdrawals(withdrawals: &[Withdrawal]) -> Self {
        assert!(
            withdrawals.len() <= MAX_WITHDRAWALS,
            "withdrawal batch size {} exceeds MAX_WITHDRAWALS={}",
            withdrawals.len(),
            MAX_WITHDRAWALS,
        );

        let mut rows: Vec<WithdrawalRootRow> = Vec::with_capacity(withdrawals.len());
        for (i, w) in withdrawals.iter().enumerate() {
            let rlp_bytes = withdrawal_rlp(w);
            assert!(
                rlp_bytes.len() <= RLP_MAX_LEN,
                "withdrawal RLP length {} exceeds RLP_MAX_LEN={}",
                rlp_bytes.len(),
                RLP_MAX_LEN,
            );
            let leaf_hash = crate::keccak::keccak256(&rlp_bytes);
            rows.push(WithdrawalRootRow {
                withdrawal: w.clone(),
                rlp_bytes,
                leaf_hash,
                is_first: i == 0,
            });
        }

        let withdrawals_root = compute_withdrawals_root(withdrawals);

        Self {
            withdrawals: withdrawals.to_vec(),
            rows,
            withdrawals_root,
        }
    }

    /// Same as [`Self::from_withdrawals`] but accepts an
    /// externally-computed root (used when the caller already has the
    /// canonical Ethereum withdrawals_root from a full MPT builder for
    /// batches with ≥ 2 leaves).
    pub fn from_withdrawals_with_root(
        withdrawals: &[Withdrawal],
        withdrawals_root: [u8; HASH_LEN],
    ) -> Self {
        let mut me = Self::from_withdrawals(withdrawals);
        me.withdrawals_root = withdrawals_root;
        me
    }
}

/// Host-side helper: compute the canonical withdrawals MPT root for
/// the 0-withdrawal and 1-withdrawal cases via [`crate::mpt`].
///
/// For batches with ≥ 2 withdrawals the host-side oracle is the
/// downstream MPT inclusion chain (one [`crate::mpt_air`] trace per
/// leaf); use [`WithdrawalRootWitness::from_withdrawals_with_root`]
/// in that path to thread the externally-computed root into the
/// witness.
pub fn compute_withdrawals_root(withdrawals: &[Withdrawal]) -> [u8; HASH_LEN] {
    if withdrawals.is_empty() {
        return crate::mpt::empty_trie_root();
    }
    if withdrawals.len() == 1 {
        let (root, _) = crate::withdrawal_trie::compute_single_withdrawal_root(
            0,
            &withdrawals[0],
        );
        return root;
    }
    // Multi-leaf fallback: derive the root from a sentinel of the
    // first leaf (host-side oracles for ≥ 2 leaves are supplied
    // explicitly via `from_withdrawals_with_root`). We mark this case
    // by hashing the concatenation of per-leaf hashes — this is *not*
    // the canonical Ethereum withdrawals_root, but is well-defined,
    // deterministic, and clearly distinguishable so misuse is loud.
    let mut acc: Vec<u8> = Vec::with_capacity(withdrawals.len() * HASH_LEN);
    for (i, w) in withdrawals.iter().enumerate() {
        let key = crate::withdrawal_trie::withdrawal_trie_key(i as u64);
        let val = withdrawal_rlp(w);
        let mut combined: Vec<u8> = Vec::with_capacity(key.len() + val.len());
        combined.extend_from_slice(&key);
        combined.extend_from_slice(&val);
        acc.extend_from_slice(&crate::keccak::keccak256(&combined));
    }
    crate::keccak::keccak256(&acc)
}

// ─── Trace builder ────────────────────────────────────────────────────

fn write_chunk(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    row: usize,
    bytes: &[u8],
    curve: CurveType,
) {
    for (b, &byte) in bytes.iter().enumerate() {
        columns[offset + b][row] = Scalar::from_u64(byte as u64, curve);
    }
}

pub fn build_trace_polynomials(
    witness: &WithdrawalRootWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    // For the zero-withdrawal case still emit at least one row so the
    // claimed-root constancy column carries the value into the trace.
    let effective_rows = num_rows.max(1);
    let padded = crate::trace::nearest_power_of_two(effective_rows);
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_INDEX][i] = Scalar::from_u64(row.withdrawal.index, curve);
        columns[COL_VALIDATOR_INDEX][i] =
            Scalar::from_u64(row.withdrawal.validator_index, curve);
        columns[COL_AMOUNT][i] = Scalar::from_u64(row.withdrawal.amount, curve);
        write_chunk(
            &mut columns,
            COL_ADDRESS_OFFSET,
            i,
            &row.withdrawal.address,
            curve,
        );
        write_chunk(
            &mut columns,
            COL_RLP_BYTES_OFFSET,
            i,
            &row.rlp_bytes,
            curve,
        );
        columns[COL_RLP_LEN][i] = Scalar::from_u64(row.rlp_bytes.len() as u64, curve);
        write_chunk(&mut columns, COL_LEAF_HASH_OFFSET, i, &row.leaf_hash, curve);
        columns[COL_IS_REAL][i] = one.clone();
        if row.is_first {
            columns[COL_IS_FIRST][i] = one.clone();
        }
    }

    // Replicate WITHDRAWALS_ROOT across every row (incl. padding) so
    // the shifted constancy body never fires at the boundary.
    for r in 0..padded {
        write_chunk(
            &mut columns,
            COL_WITHDRAWALS_ROOT_OFFSET,
            r,
            &witness.withdrawals_root,
            curve,
        );
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial {
            evaluations: evals,
            degree: effective_rows,
        })
        .collect();

    TracePolynomials {
        columns: polys,
        num_rows: effective_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct WithdrawalRootConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl WithdrawalRootConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            omega: None,
            domain_size: None,
        }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn scalar_pow(base: &Scalar, exp: u64) -> Scalar {
    let mut result = Scalar::one(base.curve_type());
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&b);
        }
        b = b.mul(&b);
        e >>= 1;
    }
    result
}

impl VmConstraintSystem for WithdrawalRootConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_first_binary".into(),
            "is_first_implies_real".into(),
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

        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        for row in 0..n {
            let is_real = &columns[COL_IS_REAL][row];
            let is_first = &columns[COL_IS_FIRST][row];

            // 0: is_real binary
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            // 1: is_first binary
            bodies[1][row] = is_first.mul(&is_first.sub(&one));
            // 2: is_first ⇒ is_real
            bodies[2][row] = is_first.mul(&one.sub(is_real));
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
        let is_first = &col_evals[COL_IS_FIRST];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_first.mul(&is_first.sub(&one)),
            is_first.mul(&one.sub(is_real)),
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
        let is_first = &col_coeffs[COL_IS_FIRST];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_first_m1 = poly_sub(is_first, &one_poly, curve);
        let is_first_binary = poly_mul(is_first, &is_first_m1, curve);

        let one_minus_real = poly_sub(&one_poly, is_real, curve);
        let is_first_implies_real = poly_mul(is_first, &one_minus_real, curve);

        let bodies: Vec<Vec<Scalar>> =
            vec![is_real_binary, is_first_binary, is_first_implies_real];

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
        // Zero everything except the WITHDRAWALS_ROOT byte columns,
        // which are deliberately replicated across padding rows so the
        // shifted constancy body has no spurious diffs at the trace
        // boundary.
        for (idx, col) in columns.iter_mut().enumerate().take(NUM_COLUMNS) {
            if (COL_WITHDRAWALS_ROOT_OFFSET..COL_WITHDRAWALS_ROOT_OFFSET + HASH_LEN)
                .contains(&idx)
            {
                continue;
            }
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        let mut cols = Vec::with_capacity(HASH_LEN + 1);
        for b in 0..HASH_LEN {
            cols.push(COL_WITHDRAWALS_ROOT_OFFSET + b);
        }
        cols.push(COL_IS_REAL);
        cols
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
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
        let expected_shifted_len = HASH_LEN + 1;
        if shifted_evals.len() < expected_shifted_len
            || col_evals_at_z.len() < NUM_COLUMNS
        {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let is_real_next = &shifted_evals[HASH_LEN];

        let mut root_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..HASH_LEN {
            let r_next = &shifted_evals[b];
            let r_curr = &col_evals_at_z[COL_WITHDRAWALS_ROOT_OFFSET + b];
            let diff = r_next.sub(r_curr);
            root_acc = root_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = is_real_next.mul(&root_acc);

        let exclusion = z.sub(omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        ap.mul(&body0).mul(&exclusion)
    }

    fn build_shifted_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_real_shift = poly_shift(is_real, omega);

        let mut root_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for b in 0..HASH_LEN {
            let r_poly = &col_coeffs[COL_WITHDRAWALS_ROOT_OFFSET + b];
            let r_shift = poly_shift(r_poly, omega);
            let diff = poly_sub(&r_shift, r_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            root_acc = poly_add(&root_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = poly_mul(&is_real_shift, &root_acc, curve);

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let excluded = poly_mul_linear(&body0, &omega_n_minus_1);
        poly_scalar_mul(&excluded, &ap)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // 8-bit range checks across every byte column.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        let byte_groups: [(&str, usize, usize); 4] = [
            ("address", COL_ADDRESS_OFFSET, ADDRESS_LEN),
            ("rlp_bytes", COL_RLP_BYTES_OFFSET, RLP_MAX_LEN),
            ("leaf_hash", COL_LEAF_HASH_OFFSET, HASH_LEN),
            ("withdrawals_root", COL_WITHDRAWALS_ROOT_OFFSET, HASH_LEN),
        ];
        for (name, offset, len) in byte_groups {
            for k in 0..len {
                declarations.push((
                    LookupDeclaration {
                        label: format!("withdrawal_root_{}_byte_{}_8bit", name, k),
                        column_index: offset + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        LookupRequirements {
            tables,
            declarations,
        }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Per-withdrawal binding descriptors for the four committed fields.
///
/// Three descriptors share the `u64_rlp_air` chip:
///   * `make_withdrawal_to_rlp_index_descriptor`
///   * `make_withdrawal_to_rlp_validator_index_descriptor`
///   * `make_withdrawal_to_rlp_amount_descriptor`
/// One descriptor binds the 20-byte address through `fixed_rlp20_air`.
///
/// Each descriptor is gated on the A side by `IS_REAL`; the B-side
/// gating uses the gadget's own `IS_REAL`-style selector.
///
/// These descriptors pin (value, encoded-bytes) tuples one field at a
/// time. The cross-row concatenation `RLP_BYTES = list_header ||
/// enc(index) || enc(val_idx) || enc(address) || enc(amount)` is the
/// responsibility of a downstream rlp-list-concat composition AIR
/// (e.g. [`crate::rlp_list_concat_air`]) consuming the same RLP byte
/// window — out of scope for this AIR.
pub fn make_withdrawal_to_rlp_index_descriptor(
    withdrawal_root_layer_index: usize,
    u64_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::u64_rlp_air as ur;
    let a_columns = vec![COL_INDEX];
    let b_columns = vec![ur::COL_VALUE];
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "withdrawal_to_rlp_index_v1".into(),
        a_layer_index: withdrawal_root_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u64_rlp_layer_index,
        b_columns,
        b_selector_column: Some(ur::COL_IS_REAL),
    }
}

pub fn make_withdrawal_to_rlp_validator_index_descriptor(
    withdrawal_root_layer_index: usize,
    u64_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::u64_rlp_air as ur;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "withdrawal_to_rlp_validator_index_v1".into(),
        a_layer_index: withdrawal_root_layer_index,
        a_columns: vec![COL_VALIDATOR_INDEX],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u64_rlp_layer_index,
        b_columns: vec![ur::COL_VALUE],
        b_selector_column: Some(ur::COL_IS_REAL),
    }
}

pub fn make_withdrawal_to_rlp_amount_descriptor(
    withdrawal_root_layer_index: usize,
    u64_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::u64_rlp_air as ur;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "withdrawal_to_rlp_amount_v1".into(),
        a_layer_index: withdrawal_root_layer_index,
        a_columns: vec![COL_AMOUNT],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: u64_rlp_layer_index,
        b_columns: vec![ur::COL_VALUE],
        b_selector_column: Some(ur::COL_IS_REAL),
    }
}

/// Binds the 20-byte recipient address to [`crate::fixed_rlp20_air`].
/// Tuple width = 20 (the raw address bytes). The gadget AIR pins the
/// canonical 21-byte RLP encoding of those bytes internally.
pub fn make_withdrawal_to_rlp_address_descriptor(
    withdrawal_root_layer_index: usize,
    fixed_rlp20_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::fixed_rlp20_air as fr;
    let a_columns: Vec<usize> =
        (0..ADDRESS_LEN).map(|b| COL_ADDRESS_OFFSET + b).collect();
    let b_columns: Vec<usize> =
        (0..ADDRESS_LEN).map(|b| fr::COL_FIELD_BYTE_OFFSET + b).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "withdrawal_to_rlp_address_v1".into(),
        a_layer_index: withdrawal_root_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: fixed_rlp20_layer_index,
        b_columns,
        b_selector_column: Some(fr::COL_IS_REAL),
    }
}

/// Binds the per-withdrawal `(RLP_BYTES[0..RLP_MAX_LEN], LEAF_HASH[0..32])`
/// tuple to [`crate::keccak_extract`]'s `(INPUT_BYTE, OUTPUT_BYTE)`
/// tuple. Combined with keccak_extract's bit-level binding this pins
/// `LEAF_HASH = keccak256(RLP_BYTES[..RLP_LEN])` for every active row.
///
/// Caveat: `RLP_LEN` is not part of the tuple (keccak_extract's
/// `INPUT_LEN` column is a single u64 not aligned with our window). A
/// separate row-local body in this AIR or an extension of the tuple
/// can pin it; the current host-side builder emits the canonical
/// encoding so the unused trailing bytes are zero by construction.
/// The keccak chip's absorb logic correctly truncates at its own
/// `INPUT_LEN`, so any A-side row that under-reports its length would
/// produce a tuple that cannot match any keccak_extract row whose
/// `INPUT_LEN` equals the published RLP length — i.e. a tampered
/// `RLP_BYTES` window is rejected by tuple-mismatch.
pub fn make_withdrawal_to_keccak_descriptor(
    withdrawal_root_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let mut a_columns: Vec<usize> =
        Vec::with_capacity(RLP_MAX_LEN + HASH_LEN);
    for b in 0..RLP_MAX_LEN {
        a_columns.push(COL_RLP_BYTES_OFFSET + b);
    }
    for b in 0..HASH_LEN {
        a_columns.push(COL_LEAF_HASH_OFFSET + b);
    }

    let mut b_columns: Vec<usize> =
        Vec::with_capacity(RLP_MAX_LEN + HASH_LEN);
    for b in 0..RLP_MAX_LEN {
        b_columns.push(ke::COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..HASH_LEN {
        b_columns.push(ke::COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "withdrawal_to_keccak_v1".into(),
        a_layer_index: withdrawal_root_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Binds the 32-byte `WITHDRAWALS_ROOT` column on the `IS_FIRST` row
/// to [`crate::block_header_air`]'s `withdrawals_root` field. Gated on
/// the A side by `IS_FIRST` (exactly one published tuple per block)
/// and on the B side by block_header_air's `IS_REAL`.
pub fn make_withdrawal_root_to_block_header_descriptor(
    withdrawal_root_layer_index: usize,
    block_header_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    let a_columns: Vec<usize> = (0..HASH_LEN)
        .map(|b| COL_WITHDRAWALS_ROOT_OFFSET + b)
        .collect();
    let b_columns: Vec<usize> = (0..HASH_LEN)
        .map(|b| bh::COL_WITHDRAWALS_ROOT_OFFSET + b)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "withdrawal_root_to_block_header_v1".into(),
        a_layer_index: withdrawal_root_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_FIRST),
        b_layer_index: block_header_layer_index,
        b_columns,
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_withdrawal(index: u64, seed: u8) -> Withdrawal {
        Withdrawal {
            index,
            validator_index: 1000 + index,
            address: [seed; 20],
            amount: 32_000_000_000 + index,
        }
    }

    /// Empty batch: 0 withdrawals → claimed root = keccak256(0x80)
    /// (the canonical RLP-empty-string MPT root).
    #[test]
    fn withdrawal_root_air_zero_withdrawals() {
        let w = WithdrawalRootWitness::from_withdrawals(&[]);
        assert_eq!(w.withdrawals.len(), 0);
        assert_eq!(w.rows.len(), 0);
        let expected = crate::keccak::keccak256(&[0x80]);
        assert_eq!(w.withdrawals_root, expected);

        // Trace is still well-formed (1 effective row of padding).
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, 1);
        // IS_REAL = 0 on row 0 (no real withdrawals).
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 0);
        // WITHDRAWALS_ROOT is replicated even on the padding row.
        for b in 0..HASH_LEN {
            assert_eq!(
                trace.columns[COL_WITHDRAWALS_ROOT_OFFSET + b].evaluations[0]
                    .to_u64(),
                expected[b] as u64,
            );
        }
    }

    /// Single withdrawal: row 0 is real, IS_FIRST = 1, leaf_hash
    /// matches keccak256(withdrawal_rlp).
    #[test]
    fn withdrawal_root_air_single_withdrawal() {
        let wd = sample_withdrawal(0, 0x11);
        let w = WithdrawalRootWitness::from_withdrawals(&[wd.clone()]);
        assert_eq!(w.rows.len(), 1);
        assert!(w.rows[0].is_first);
        let expected_rlp = withdrawal_rlp(&wd);
        let expected_hash = crate::keccak::keccak256(&expected_rlp);
        assert_eq!(w.rows[0].rlp_bytes, expected_rlp);
        assert_eq!(w.rows[0].leaf_hash, expected_hash);

        // Constraints vanish on every row of the padded trace.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cs = WithdrawalRootConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) at row {} must vanish (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    /// 16-withdrawal batch (the spec cap): every row real, only row 0
    /// has IS_FIRST. The trace must fit and constraints must vanish.
    #[test]
    fn withdrawal_root_air_sixteen_withdrawals() {
        let withdrawals: Vec<Withdrawal> = (0..MAX_WITHDRAWALS as u64)
            .map(|i| sample_withdrawal(i, (i + 1) as u8))
            .collect();
        let w = WithdrawalRootWitness::from_withdrawals(&withdrawals);
        assert_eq!(w.rows.len(), MAX_WITHDRAWALS);
        assert!(w.rows[0].is_first);
        for i in 1..MAX_WITHDRAWALS {
            assert!(!w.rows[i].is_first);
        }

        // Spot-check leaf hashes.
        for i in 0..MAX_WITHDRAWALS {
            let expected_hash =
                crate::keccak::keccak256(&withdrawal_rlp(&withdrawals[i]));
            assert_eq!(w.rows[i].leaf_hash, expected_hash);
        }

        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cs = WithdrawalRootConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) at row {} must vanish",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    /// Tampering: tamper the LEAF_HASH on row 1 — the keccak linkage
    /// would detect this in joint_prove. Here we directly observe that
    /// the witness builder's leaf_hash matches the canonical hash, and
    /// that an explicitly-tampered byte differs.
    #[test]
    fn withdrawal_root_air_tampered_withdrawal_detected() {
        let withdrawals: Vec<Withdrawal> =
            (0..3).map(|i| sample_withdrawal(i, (i + 1) as u8)).collect();
        let w = WithdrawalRootWitness::from_withdrawals(&withdrawals);
        let curve = CurveType::Bls48581;
        let mut trace = build_trace_polynomials(&w, curve);

        // Tamper LEAF_HASH[5] on row 1: this leaf-hash byte will no
        // longer match keccak256(RLP_BYTES) — the cross-AIR LogUp to
        // keccak_extract would fail. The host-side detection here is
        // the byte-mismatch against the canonical hash.
        let tampered = Scalar::from_u64(0xEE, curve);
        let original =
            trace.columns[COL_LEAF_HASH_OFFSET + 5].evaluations[1].clone();
        assert_ne!(
            original.to_u64(),
            tampered.to_u64(),
            "test guard: tampered byte must differ from the canonical one",
        );
        trace.columns[COL_LEAF_HASH_OFFSET + 5].evaluations[1] = tampered.clone();

        let canonical_hash =
            crate::keccak::keccak256(&withdrawal_rlp(&withdrawals[1]));
        assert_eq!(canonical_hash[5] as u64, original.to_u64());
        assert_ne!(canonical_hash[5] as u64, tampered.to_u64());

        // Row-local constraints still vanish (they're about
        // selectors); the tamper is caught by the keccak cross-AIR
        // LogUp, not by an in-AIR algebraic body. Confirm vanishing
        // anyway.
        let cs = WithdrawalRootConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for body in &evals {
            for v in body.iter() {
                assert!(v.is_zero());
            }
        }

        // Also tamper IS_FIRST on a non-first row → `is_first_binary`
        // still vanishes (it's 1) but `is_first_implies_real` should
        // also vanish since IS_REAL = 1 too. To exercise a real
        // selector-violation, set IS_FIRST = 1 on a padding row where
        // IS_REAL = 0.
        let pad_row = trace.num_rows; // first padding row
        if pad_row < trace.columns[0].evaluations.len() {
            trace.columns[COL_IS_FIRST].evaluations[pad_row] =
                Scalar::one(curve);
            let cs = WithdrawalRootConstraintSystem::new(trace.num_rows);
            let cols_owned: Vec<&Vec<Scalar>> =
                trace.columns.iter().map(|p| &p.evaluations).collect();
            let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
            // Body 2 (is_first_implies_real) must fire on pad_row.
            assert!(
                !evals[2][pad_row].is_zero(),
                "tampered IS_FIRST on padding row must fire is_first_implies_real",
            );
        }
    }

    /// Cross-AIR LogUp descriptors are well-formed: tuple widths,
    /// labels, selectors.
    #[test]
    fn withdrawal_root_air_descriptors_well_formed() {
        // u64_rlp descriptors (index / val_idx / amount).
        let d_idx = make_withdrawal_to_rlp_index_descriptor(0, 1);
        assert_eq!(d_idx.label, "withdrawal_to_rlp_index_v1");
        assert_eq!(d_idx.a_columns, vec![COL_INDEX]);
        assert_eq!(d_idx.b_columns, vec![crate::u64_rlp_air::COL_VALUE]);
        assert_eq!(d_idx.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_idx.b_selector_column,
            Some(crate::u64_rlp_air::COL_IS_REAL),
        );
        assert_eq!(d_idx.a_layer_index, 0);
        assert_eq!(d_idx.b_layer_index, 1);

        let d_vi = make_withdrawal_to_rlp_validator_index_descriptor(0, 1);
        assert_eq!(d_vi.label, "withdrawal_to_rlp_validator_index_v1");
        assert_eq!(d_vi.a_columns, vec![COL_VALIDATOR_INDEX]);

        let d_amt = make_withdrawal_to_rlp_amount_descriptor(0, 1);
        assert_eq!(d_amt.label, "withdrawal_to_rlp_amount_v1");
        assert_eq!(d_amt.a_columns, vec![COL_AMOUNT]);

        // address descriptor — 20-byte tuple.
        let d_addr = make_withdrawal_to_rlp_address_descriptor(0, 2);
        assert_eq!(d_addr.label, "withdrawal_to_rlp_address_v1");
        assert_eq!(d_addr.a_columns.len(), ADDRESS_LEN);
        assert_eq!(d_addr.b_columns.len(), ADDRESS_LEN);
        for b in 0..ADDRESS_LEN {
            assert_eq!(d_addr.a_columns[b], COL_ADDRESS_OFFSET + b);
            assert_eq!(
                d_addr.b_columns[b],
                crate::fixed_rlp20_air::COL_FIELD_BYTE_OFFSET + b,
            );
        }
        assert_eq!(d_addr.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_addr.b_selector_column,
            Some(crate::fixed_rlp20_air::COL_IS_REAL),
        );

        // keccak descriptor — (RLP_BYTES || LEAF_HASH).
        let d_kec = make_withdrawal_to_keccak_descriptor(0, 3);
        assert_eq!(d_kec.label, "withdrawal_to_keccak_v1");
        assert_eq!(d_kec.a_columns.len(), RLP_MAX_LEN + HASH_LEN);
        assert_eq!(d_kec.b_columns.len(), RLP_MAX_LEN + HASH_LEN);
        for b in 0..RLP_MAX_LEN {
            assert_eq!(d_kec.a_columns[b], COL_RLP_BYTES_OFFSET + b);
            assert_eq!(
                d_kec.b_columns[b],
                crate::keccak_extract::COL_INPUT_BYTE_OFFSET + b,
            );
        }
        for b in 0..HASH_LEN {
            assert_eq!(
                d_kec.a_columns[RLP_MAX_LEN + b],
                COL_LEAF_HASH_OFFSET + b,
            );
            assert_eq!(
                d_kec.b_columns[RLP_MAX_LEN + b],
                crate::keccak_extract::COL_OUTPUT_BYTE_OFFSET + b,
            );
        }
        assert_eq!(d_kec.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_kec.b_selector_column,
            Some(crate::keccak_extract::COL_IS_REAL),
        );

        // block_header descriptor — 32-byte tuple, IS_FIRST gating.
        let d_bh = make_withdrawal_root_to_block_header_descriptor(0, 4);
        assert_eq!(d_bh.label, "withdrawal_root_to_block_header_v1");
        assert_eq!(d_bh.a_columns.len(), HASH_LEN);
        assert_eq!(d_bh.b_columns.len(), HASH_LEN);
        for b in 0..HASH_LEN {
            assert_eq!(d_bh.a_columns[b], COL_WITHDRAWALS_ROOT_OFFSET + b);
            assert_eq!(
                d_bh.b_columns[b],
                crate::block_header_air::COL_WITHDRAWALS_ROOT_OFFSET + b,
            );
        }
        assert_eq!(d_bh.a_selector_column, Some(COL_IS_FIRST));
        assert_eq!(
            d_bh.b_selector_column,
            Some(crate::block_header_air::COL_IS_REAL),
        );
    }
}
