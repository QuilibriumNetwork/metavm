//! Block full proof composition AIR.
//!
//! A **high-level composer** that threads the per-block validity
//! gadgets — block_header_air, tx_full_chain_air (×N), block_proposer_sig_air,
//! withdrawal_root_air, and beacon_state_transition_air — into a single
//! per-block witness with cross-AIR LogUp descriptors binding the
//! per-field tuples.
//!
//! This AIR does **not** re-prove block-header RLP encoding, transaction
//! execution, signature pairing, or state-transition rules — every one
//! of those is already algebraically closed by its respective sub-gadget.
//! This AIR only commits the per-block summary tuple and publishes the
//! algebraic anchors that the descriptors then bind across layers.
//!
//! ## Row layout
//!
//! One row per transaction slot, plus a single header row. The AIR caps
//! the row count at [`MAX_TRANSACTIONS`] = 16.
//!
//! Per-row columns commit:
//!   * block-level fields (replicated across every row so descriptors
//!     gated on `IS_REAL` see them on every active row):
//!     - `BLOCK_NUMBER` (u64) + 8 LE bytes
//!     - `BLOCK_HASH[0..32]`, `PARENT_HASH[0..32]`
//!     - `STATE_ROOT[0..32]`, `TX_ROOT[0..32]`, `RECEIPTS_ROOT[0..32]`,
//!       `WITHDRAWALS_ROOT[0..32]`
//!     - `NUM_TRANSACTIONS` (u32) + 4 LE bytes
//!     - `TOTAL_GAS_USED` (u64) + 8 LE bytes
//!     - `GAS_LIMIT` (u64) + 8 LE bytes
//!     - `GAS_SLACK` (u64) + 8 LE bytes — witness; `total + slack = limit`
//!     - `TIMESTAMP` (u64) + 8 LE bytes
//!     - `PROPOSER_INDEX` (u64) + 8 LE bytes
//!     - `IS_REAL`
//!     - `IS_HEADER` — marks row 0 (single anchor for descriptors that
//!       only want one published tuple per block)
//!     - `TX_INDEX` (u32) — per-row transaction index
//!     - `TX_IS_ACTIVE` — set when `tx_index < num_transactions`
//!
//! ## Algebraic constraints (≥ 8)
//!
//! Row-local:
//!   0. `is_real_binary`         — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `is_header_binary`       — `IS_HEADER · (IS_HEADER − 1) = 0`
//!   2. `tx_active_binary`       — `TX_IS_ACTIVE · (TX_IS_ACTIVE − 1) = 0`
//!   3. `is_header_implies_real` — `IS_HEADER · (1 − IS_REAL) = 0`
//!   4. `gas_balance`            — `IS_REAL · (TOTAL_GAS_USED + GAS_SLACK − GAS_LIMIT) = 0`
//!   5. `block_number_be_decomp` — `IS_REAL · (BLOCK_NUMBER − Σ_k 256^k · BLOCK_NUMBER_BYTE[k]) = 0`
//!   6. `total_gas_used_be_decomp`
//!   7. `gas_limit_be_decomp`
//!   8. `gas_slack_be_decomp`
//!   9. `timestamp_be_decomp`
//!  10. `proposer_index_be_decomp`
//!  11. `num_transactions_be_decomp` — same shape, 4 bytes
//!
//! 8-bit range-check declarations cover every per-field LE-byte column
//! group (block_number / total_gas_used / gas_limit / gas_slack /
//! timestamp / proposer_index / num_transactions) plus the 32-byte
//! hash columns.
//!
//! ## Cross-AIR LogUp descriptors
//!
//!   * [`make_block_to_block_header_descriptor`] — 9 fields →
//!     [`crate::block_header_air`].
//!   * [`make_block_to_proposer_sig_descriptor`] — `(block_hash[0..32],
//!     proposer_index_le_bytes[0..8])` → [`crate::block_proposer_sig_air`].
//!   * [`make_block_to_withdrawal_root_descriptor`] — `withdrawals_root[0..32]`
//!     → [`crate::withdrawal_root_air`].
//!   * [`make_block_to_state_transition_descriptor`] — `state_root[0..32]`
//!     → [`crate::beacon_state_transition_air`].
//!   * [`make_block_to_tx_full_chain_descriptor`] — per-row binding
//!     `(tx_index, block_hash, total_gas_used)` to `tx_full_chain_air`.
//!     **Stub**: `tx_full_chain_air` is being built in parallel; the
//!     B-side columns are filled with [`TX_FULL_CHAIN_LAYER_SENTINEL`]
//!     (`usize::MAX`) and the B-side layer index is also `usize::MAX`.
//!     The descriptor MUST NOT be passed to `joint_prove` until
//!     `tx_full_chain_air` lands.

use crate::block_header::BlockHeader;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::transaction::Transaction;
use crate::vm_constraints::VmConstraintSystem;
use crate::withdrawal::Withdrawal;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum transactions per block this AIR can compose at once.
pub const MAX_TRANSACTIONS: usize = 16;

/// Hash (keccak / MPT root) byte length.
pub const HASH_LEN: usize = 32;

/// LE byte width of a u64 field.
pub const U64_BYTES: usize = 8;

/// LE byte width of a u32 field.
pub const U32_BYTES: usize = 4;

/// Sentinel B-side layer index used until `tx_full_chain_air` lands.
pub const TX_FULL_CHAIN_LAYER_SENTINEL: usize = usize::MAX;

/// Sentinel B-side column index used until `tx_full_chain_air` lands.
pub const TX_FULL_CHAIN_COL_SENTINEL: usize = usize::MAX;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_BLOCK_NUMBER: usize = 0;
pub const COL_BLOCK_NUMBER_BYTE_OFFSET: usize = COL_BLOCK_NUMBER + 1; // 1..9

pub const COL_BLOCK_HASH_OFFSET: usize =
    COL_BLOCK_NUMBER_BYTE_OFFSET + U64_BYTES; // 9..41
pub const COL_PARENT_HASH_OFFSET: usize =
    COL_BLOCK_HASH_OFFSET + HASH_LEN; // 41..73
pub const COL_STATE_ROOT_OFFSET: usize =
    COL_PARENT_HASH_OFFSET + HASH_LEN; // 73..105
pub const COL_TX_ROOT_OFFSET: usize =
    COL_STATE_ROOT_OFFSET + HASH_LEN; // 105..137
pub const COL_RECEIPTS_ROOT_OFFSET: usize =
    COL_TX_ROOT_OFFSET + HASH_LEN; // 137..169
pub const COL_WITHDRAWALS_ROOT_OFFSET: usize =
    COL_RECEIPTS_ROOT_OFFSET + HASH_LEN; // 169..201

pub const COL_NUM_TRANSACTIONS: usize = COL_WITHDRAWALS_ROOT_OFFSET + HASH_LEN; // 201
pub const COL_NUM_TRANSACTIONS_BYTE_OFFSET: usize = COL_NUM_TRANSACTIONS + 1; // 202..206

pub const COL_TOTAL_GAS_USED: usize =
    COL_NUM_TRANSACTIONS_BYTE_OFFSET + U32_BYTES; // 206
pub const COL_TOTAL_GAS_USED_BYTE_OFFSET: usize = COL_TOTAL_GAS_USED + 1; // 207..215

pub const COL_GAS_LIMIT: usize = COL_TOTAL_GAS_USED_BYTE_OFFSET + U64_BYTES; // 215
pub const COL_GAS_LIMIT_BYTE_OFFSET: usize = COL_GAS_LIMIT + 1; // 216..224

pub const COL_GAS_SLACK: usize = COL_GAS_LIMIT_BYTE_OFFSET + U64_BYTES; // 224
pub const COL_GAS_SLACK_BYTE_OFFSET: usize = COL_GAS_SLACK + 1; // 225..233

pub const COL_TIMESTAMP: usize = COL_GAS_SLACK_BYTE_OFFSET + U64_BYTES; // 233
pub const COL_TIMESTAMP_BYTE_OFFSET: usize = COL_TIMESTAMP + 1; // 234..242

pub const COL_PROPOSER_INDEX: usize =
    COL_TIMESTAMP_BYTE_OFFSET + U64_BYTES; // 242
pub const COL_PROPOSER_INDEX_BYTE_OFFSET: usize = COL_PROPOSER_INDEX + 1; // 243..251

pub const COL_IS_REAL: usize = COL_PROPOSER_INDEX_BYTE_OFFSET + U64_BYTES; // 251
pub const COL_IS_HEADER: usize = COL_IS_REAL + 1; // 252
pub const COL_TX_INDEX: usize = COL_IS_HEADER + 1; // 253
pub const COL_TX_IS_ACTIVE: usize = COL_TX_INDEX + 1; // 254

pub const NUM_COLUMNS: usize = COL_TX_IS_ACTIVE + 1; // 255

/// Row-local bodies (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 12;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockFullProofWitness {
    pub block_number: u64,
    pub block_hash: [u8; HASH_LEN],
    pub parent_hash: [u8; HASH_LEN],
    pub state_root: [u8; HASH_LEN],
    pub tx_root: [u8; HASH_LEN],
    pub receipts_root: [u8; HASH_LEN],
    pub withdrawals_root: [u8; HASH_LEN],
    pub num_transactions: u32,
    pub total_gas_used: u64,
    pub gas_limit: u64,
    pub timestamp: u64,
    pub proposer_index: u64,
}

impl BlockFullProofWitness {
    /// Host-side constructor: derive the block-level summary tuple from
    /// the executable block header, the transaction batch, the
    /// withdrawal batch (used to recover the canonical withdrawals_root),
    /// and the beacon proposer index.
    ///
    /// The header is treated as ground truth: `block_hash` is computed
    /// from `block_header::block_header_hash`. The transactions provide
    /// the `num_transactions` count; their on-chain `gas_used` is
    /// already summed into `header.gas_used` and is used here. The
    /// withdrawal batch is accepted for shape — the witness exposes the
    /// `withdrawals_root` that the header already published.
    ///
    /// Panics if the header's `withdrawals_root` is `None` (pre-Shanghai).
    pub fn from_block(
        header: &BlockHeader,
        txs: &[Transaction],
        withdrawals: &[Withdrawal],
        proposer_index: u64,
    ) -> Self {
        let _ = withdrawals; // not used in the summary but kept for the
                             // host-side contract: the withdrawal batch
                             // must be supplied so the caller cannot
                             // shape-mismatch the published root.
        assert!(
            txs.len() <= MAX_TRANSACTIONS,
            "tx batch size {} exceeds MAX_TRANSACTIONS={}",
            txs.len(),
            MAX_TRANSACTIONS,
        );
        let block_hash = crate::block_header::block_header_hash(header);
        let withdrawals_root = header
            .withdrawals_root
            .expect("BlockHeader.withdrawals_root must be Some (post-Shanghai)");
        Self {
            block_number: header.number,
            block_hash,
            parent_hash: header.parent_hash,
            state_root: header.state_root,
            tx_root: header.transactions_root,
            receipts_root: header.receipts_root,
            withdrawals_root,
            num_transactions: txs.len() as u32,
            total_gas_used: header.gas_used,
            gas_limit: header.gas_limit,
            timestamp: header.timestamp,
            proposer_index,
        }
    }

    /// Witness for `gas_slack = gas_limit - total_gas_used`. Panics if
    /// `total_gas_used > gas_limit` (this is what the algebraic
    /// `gas_balance` body would catch in the proof, but we surface it
    /// at host-side construction time so callers get a clean error).
    pub fn gas_slack(&self) -> u64 {
        self.gas_limit
            .checked_sub(self.total_gas_used)
            .expect("total_gas_used must not exceed gas_limit")
    }

    /// LE bytes (8) of `block_number`.
    pub fn block_number_le_bytes(&self) -> [u8; U64_BYTES] {
        self.block_number.to_le_bytes()
    }

    /// LE bytes (4) of `num_transactions`.
    pub fn num_transactions_le_bytes(&self) -> [u8; U32_BYTES] {
        self.num_transactions.to_le_bytes()
    }

    /// LE bytes (8) of `total_gas_used`.
    pub fn total_gas_used_le_bytes(&self) -> [u8; U64_BYTES] {
        self.total_gas_used.to_le_bytes()
    }

    /// LE bytes (8) of `gas_limit`.
    pub fn gas_limit_le_bytes(&self) -> [u8; U64_BYTES] {
        self.gas_limit.to_le_bytes()
    }

    /// LE bytes (8) of `gas_slack`.
    pub fn gas_slack_le_bytes(&self) -> [u8; U64_BYTES] {
        self.gas_slack().to_le_bytes()
    }

    /// LE bytes (8) of `timestamp`.
    pub fn timestamp_le_bytes(&self) -> [u8; U64_BYTES] {
        self.timestamp.to_le_bytes()
    }

    /// LE bytes (8) of `proposer_index`.
    pub fn proposer_index_le_bytes(&self) -> [u8; U64_BYTES] {
        self.proposer_index.to_le_bytes()
    }
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
    witness: &BlockFullProofWitness,
    curve: CurveType,
) -> TracePolynomials {
    // One row per (active) tx slot, capped at MAX_TRANSACTIONS. The
    // composer always emits MAX_TRANSACTIONS active rows so descriptors
    // gated on IS_REAL see a consistent population. The TX_IS_ACTIVE
    // selector distinguishes the `tx_index < num_transactions` subset.
    let effective_rows = MAX_TRANSACTIONS;
    let padded = crate::trace::nearest_power_of_two(effective_rows);

    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    let block_number_le = witness.block_number_le_bytes();
    let num_txs_le = witness.num_transactions_le_bytes();
    let total_gas_le = witness.total_gas_used_le_bytes();
    let gas_limit_le = witness.gas_limit_le_bytes();
    let gas_slack_le = witness.gas_slack_le_bytes();
    let timestamp_le = witness.timestamp_le_bytes();
    let proposer_index_le = witness.proposer_index_le_bytes();
    let gas_slack_u64 = witness.gas_slack();

    for r in 0..effective_rows {
        columns[COL_BLOCK_NUMBER][r] =
            Scalar::from_u64(witness.block_number, curve);
        write_chunk(
            &mut columns,
            COL_BLOCK_NUMBER_BYTE_OFFSET,
            r,
            &block_number_le,
            curve,
        );
        write_chunk(&mut columns, COL_BLOCK_HASH_OFFSET, r, &witness.block_hash, curve);
        write_chunk(
            &mut columns,
            COL_PARENT_HASH_OFFSET,
            r,
            &witness.parent_hash,
            curve,
        );
        write_chunk(
            &mut columns,
            COL_STATE_ROOT_OFFSET,
            r,
            &witness.state_root,
            curve,
        );
        write_chunk(&mut columns, COL_TX_ROOT_OFFSET, r, &witness.tx_root, curve);
        write_chunk(
            &mut columns,
            COL_RECEIPTS_ROOT_OFFSET,
            r,
            &witness.receipts_root,
            curve,
        );
        write_chunk(
            &mut columns,
            COL_WITHDRAWALS_ROOT_OFFSET,
            r,
            &witness.withdrawals_root,
            curve,
        );
        columns[COL_NUM_TRANSACTIONS][r] =
            Scalar::from_u64(witness.num_transactions as u64, curve);
        write_chunk(
            &mut columns,
            COL_NUM_TRANSACTIONS_BYTE_OFFSET,
            r,
            &num_txs_le,
            curve,
        );
        columns[COL_TOTAL_GAS_USED][r] =
            Scalar::from_u64(witness.total_gas_used, curve);
        write_chunk(
            &mut columns,
            COL_TOTAL_GAS_USED_BYTE_OFFSET,
            r,
            &total_gas_le,
            curve,
        );
        columns[COL_GAS_LIMIT][r] = Scalar::from_u64(witness.gas_limit, curve);
        write_chunk(
            &mut columns,
            COL_GAS_LIMIT_BYTE_OFFSET,
            r,
            &gas_limit_le,
            curve,
        );
        columns[COL_GAS_SLACK][r] = Scalar::from_u64(gas_slack_u64, curve);
        write_chunk(
            &mut columns,
            COL_GAS_SLACK_BYTE_OFFSET,
            r,
            &gas_slack_le,
            curve,
        );
        columns[COL_TIMESTAMP][r] = Scalar::from_u64(witness.timestamp, curve);
        write_chunk(
            &mut columns,
            COL_TIMESTAMP_BYTE_OFFSET,
            r,
            &timestamp_le,
            curve,
        );
        columns[COL_PROPOSER_INDEX][r] =
            Scalar::from_u64(witness.proposer_index, curve);
        write_chunk(
            &mut columns,
            COL_PROPOSER_INDEX_BYTE_OFFSET,
            r,
            &proposer_index_le,
            curve,
        );
        columns[COL_IS_REAL][r] = one.clone();
        if r == 0 {
            columns[COL_IS_HEADER][r] = one.clone();
        }
        columns[COL_TX_INDEX][r] = Scalar::from_u64(r as u64, curve);
        if (r as u32) < witness.num_transactions {
            columns[COL_TX_IS_ACTIVE][r] = one.clone();
        }
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

pub struct BlockFullProofConstraintSystem {
    pub num_rows: usize,
}

impl BlockFullProofConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows }
    }
}

/// Returns a vector of LE-byte weights `[1, 256, 256^2, …, 256^(len-1)]`
/// as field scalars.
fn le_byte_weights(len: usize, curve: CurveType) -> Vec<Scalar> {
    let mut weights = Vec::with_capacity(len);
    let two56 = Scalar::from_u64(256, curve);
    let mut w = Scalar::one(curve);
    for _ in 0..len {
        weights.push(w.clone());
        w = w.mul(&two56);
    }
    weights
}

/// Per-row helper: `IS_REAL · (VALUE − Σ_k 256^k · BYTE_k)`.
fn eval_le_decomp_body(
    is_real: &Scalar,
    value: &Scalar,
    byte_cols: &[&Scalar],
    weights: &[Scalar],
) -> Scalar {
    let curve = is_real.curve_type();
    let mut acc = Scalar::zero(curve);
    for (b, byte) in byte_cols.iter().enumerate() {
        acc = acc.add(&byte.mul(&weights[b]));
    }
    let diff = value.sub(&acc);
    is_real.mul(&diff)
}

impl VmConstraintSystem for BlockFullProofConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_header_binary".into(),
            "tx_active_binary".into(),
            "is_header_implies_real".into(),
            "gas_balance".into(),
            "block_number_le_decomp".into(),
            "total_gas_used_le_decomp".into(),
            "gas_limit_le_decomp".into(),
            "gas_slack_le_decomp".into(),
            "timestamp_le_decomp".into(),
            "proposer_index_le_decomp".into(),
            "num_transactions_le_decomp".into(),
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

        let w_u64 = le_byte_weights(U64_BYTES, curve);
        let w_u32 = le_byte_weights(U32_BYTES, curve);

        for row in 0..n {
            let is_real = &columns[COL_IS_REAL][row];
            let is_header = &columns[COL_IS_HEADER][row];
            let tx_is_active = &columns[COL_TX_IS_ACTIVE][row];

            // 0: is_real binary
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            // 1: is_header binary
            bodies[1][row] = is_header.mul(&is_header.sub(&one));
            // 2: tx_is_active binary
            bodies[2][row] = tx_is_active.mul(&tx_is_active.sub(&one));
            // 3: is_header ⇒ is_real
            bodies[3][row] = is_header.mul(&one.sub(is_real));

            // 4: gas balance: IS_REAL · (total + slack − limit) = 0
            let total = &columns[COL_TOTAL_GAS_USED][row];
            let slack = &columns[COL_GAS_SLACK][row];
            let limit = &columns[COL_GAS_LIMIT][row];
            bodies[4][row] = is_real.mul(&total.add(slack).sub(limit));

            // 5..10: u64 LE decomps (block_number / total / limit /
            // slack / timestamp / proposer_index).
            let u64_targets: [(usize, usize, usize); 6] = [
                (5, COL_BLOCK_NUMBER, COL_BLOCK_NUMBER_BYTE_OFFSET),
                (6, COL_TOTAL_GAS_USED, COL_TOTAL_GAS_USED_BYTE_OFFSET),
                (7, COL_GAS_LIMIT, COL_GAS_LIMIT_BYTE_OFFSET),
                (8, COL_GAS_SLACK, COL_GAS_SLACK_BYTE_OFFSET),
                (9, COL_TIMESTAMP, COL_TIMESTAMP_BYTE_OFFSET),
                (10, COL_PROPOSER_INDEX, COL_PROPOSER_INDEX_BYTE_OFFSET),
            ];
            for (bi, value_col, byte_offset) in u64_targets {
                let value = &columns[value_col][row];
                let byte_refs: Vec<&Scalar> = (0..U64_BYTES)
                    .map(|k| &columns[byte_offset + k][row])
                    .collect();
                bodies[bi][row] =
                    eval_le_decomp_body(is_real, value, &byte_refs, &w_u64);
            }

            // 11: u32 LE decomp for num_transactions.
            let value = &columns[COL_NUM_TRANSACTIONS][row];
            let byte_refs: Vec<&Scalar> = (0..U32_BYTES)
                .map(|k| &columns[COL_NUM_TRANSACTIONS_BYTE_OFFSET + k][row])
                .collect();
            bodies[11][row] =
                eval_le_decomp_body(is_real, value, &byte_refs, &w_u32);
        }
        bodies
    }

    fn evaluate_at_point(
        &self,
        col_evals: &[Scalar],
        alpha: &Scalar,
    ) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals[COL_IS_REAL];
        let is_header = &col_evals[COL_IS_HEADER];
        let tx_is_active = &col_evals[COL_TX_IS_ACTIVE];

        let w_u64 = le_byte_weights(U64_BYTES, curve);
        let w_u32 = le_byte_weights(U32_BYTES, curve);

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(is_header.mul(&is_header.sub(&one)));
        bodies.push(tx_is_active.mul(&tx_is_active.sub(&one)));
        bodies.push(is_header.mul(&one.sub(is_real)));

        let total = &col_evals[COL_TOTAL_GAS_USED];
        let slack = &col_evals[COL_GAS_SLACK];
        let limit = &col_evals[COL_GAS_LIMIT];
        bodies.push(is_real.mul(&total.add(slack).sub(limit)));

        let u64_targets: [(usize, usize); 6] = [
            (COL_BLOCK_NUMBER, COL_BLOCK_NUMBER_BYTE_OFFSET),
            (COL_TOTAL_GAS_USED, COL_TOTAL_GAS_USED_BYTE_OFFSET),
            (COL_GAS_LIMIT, COL_GAS_LIMIT_BYTE_OFFSET),
            (COL_GAS_SLACK, COL_GAS_SLACK_BYTE_OFFSET),
            (COL_TIMESTAMP, COL_TIMESTAMP_BYTE_OFFSET),
            (COL_PROPOSER_INDEX, COL_PROPOSER_INDEX_BYTE_OFFSET),
        ];
        for (value_col, byte_offset) in u64_targets {
            let value = &col_evals[value_col];
            let byte_refs: Vec<&Scalar> = (0..U64_BYTES)
                .map(|k| &col_evals[byte_offset + k])
                .collect();
            bodies.push(eval_le_decomp_body(is_real, value, &byte_refs, &w_u64));
        }
        let value = &col_evals[COL_NUM_TRANSACTIONS];
        let byte_refs: Vec<&Scalar> = (0..U32_BYTES)
            .map(|k| &col_evals[COL_NUM_TRANSACTIONS_BYTE_OFFSET + k])
            .collect();
        bodies.push(eval_le_decomp_body(is_real, value, &byte_refs, &w_u32));

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
        let is_header = &col_coeffs[COL_IS_HEADER];
        let tx_active = &col_coeffs[COL_TX_IS_ACTIVE];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_header_m1 = poly_sub(is_header, &one_poly, curve);
        let is_header_binary = poly_mul(is_header, &is_header_m1, curve);

        let tx_active_m1 = poly_sub(tx_active, &one_poly, curve);
        let tx_active_binary = poly_mul(tx_active, &tx_active_m1, curve);

        let one_minus_real = poly_sub(&one_poly, is_real, curve);
        let is_header_implies_real = poly_mul(is_header, &one_minus_real, curve);

        let total = &col_coeffs[COL_TOTAL_GAS_USED];
        let slack = &col_coeffs[COL_GAS_SLACK];
        let limit = &col_coeffs[COL_GAS_LIMIT];
        let total_plus_slack = poly_add(total, slack, curve);
        let balance_inner = poly_sub(&total_plus_slack, limit, curve);
        let gas_balance = poly_mul(is_real, &balance_inner, curve);

        let mut bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_header_binary,
            tx_active_binary,
            is_header_implies_real,
            gas_balance,
        ];

        let w_u64 = le_byte_weights(U64_BYTES, curve);
        let w_u32 = le_byte_weights(U32_BYTES, curve);

        let u64_targets: [(usize, usize); 6] = [
            (COL_BLOCK_NUMBER, COL_BLOCK_NUMBER_BYTE_OFFSET),
            (COL_TOTAL_GAS_USED, COL_TOTAL_GAS_USED_BYTE_OFFSET),
            (COL_GAS_LIMIT, COL_GAS_LIMIT_BYTE_OFFSET),
            (COL_GAS_SLACK, COL_GAS_SLACK_BYTE_OFFSET),
            (COL_TIMESTAMP, COL_TIMESTAMP_BYTE_OFFSET),
            (COL_PROPOSER_INDEX, COL_PROPOSER_INDEX_BYTE_OFFSET),
        ];
        for (value_col, byte_offset) in u64_targets {
            let value = &col_coeffs[value_col];
            // Σ w_k · byte_k
            let mut weighted = vec![Scalar::zero(curve)];
            for k in 0..U64_BYTES {
                let byte = &col_coeffs[byte_offset + k];
                let scaled = poly_scalar_mul(byte, &w_u64[k]);
                weighted = poly_add(&weighted, &scaled, curve);
            }
            let diff = poly_sub(value, &weighted, curve);
            bodies.push(poly_mul(is_real, &diff, curve));
        }

        // u32 decomp for num_transactions.
        let value = &col_coeffs[COL_NUM_TRANSACTIONS];
        let mut weighted = vec![Scalar::zero(curve)];
        for k in 0..U32_BYTES {
            let byte = &col_coeffs[COL_NUM_TRANSACTIONS_BYTE_OFFSET + k];
            let scaled = poly_scalar_mul(byte, &w_u32[k]);
            weighted = poly_add(&weighted, &scaled, curve);
        }
        let diff = poly_sub(value, &weighted, curve);
        bodies.push(poly_mul(is_real, &diff, curve));

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

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        // Hash byte groups.
        let hash_groups: [(&str, usize); 6] = [
            ("block_hash", COL_BLOCK_HASH_OFFSET),
            ("parent_hash", COL_PARENT_HASH_OFFSET),
            ("state_root", COL_STATE_ROOT_OFFSET),
            ("tx_root", COL_TX_ROOT_OFFSET),
            ("receipts_root", COL_RECEIPTS_ROOT_OFFSET),
            ("withdrawals_root", COL_WITHDRAWALS_ROOT_OFFSET),
        ];
        for (name, off) in hash_groups {
            for k in 0..HASH_LEN {
                declarations.push((
                    LookupDeclaration {
                        label: format!("block_full_{}_byte_{}_8bit", name, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        // u64 byte groups.
        let u64_groups: [(&str, usize); 6] = [
            ("block_number", COL_BLOCK_NUMBER_BYTE_OFFSET),
            ("total_gas_used", COL_TOTAL_GAS_USED_BYTE_OFFSET),
            ("gas_limit", COL_GAS_LIMIT_BYTE_OFFSET),
            ("gas_slack", COL_GAS_SLACK_BYTE_OFFSET),
            ("timestamp", COL_TIMESTAMP_BYTE_OFFSET),
            ("proposer_index", COL_PROPOSER_INDEX_BYTE_OFFSET),
        ];
        for (name, off) in u64_groups {
            for k in 0..U64_BYTES {
                declarations.push((
                    LookupDeclaration {
                        label: format!("block_full_{}_byte_{}_8bit", name, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        // u32 byte group: num_transactions.
        for k in 0..U32_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("block_full_num_transactions_byte_{}_8bit", k),
                    column_index: COL_NUM_TRANSACTIONS_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements {
            tables,
            declarations,
        }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Binds nine block-header fields to [`crate::block_header_air`]:
///   * `BLOCK_NUMBER`           ↔ `bh::COL_NUMBER`
///   * `BLOCK_HASH[0..32]`      ↔ `bh::COL_BLOCK_HASH_OFFSET..+32`
///   * `PARENT_HASH[0..32]`     ↔ `bh::COL_PARENT_HASH_OFFSET..+32`
///   * `STATE_ROOT[0..32]`      ↔ `bh::COL_STATE_ROOT_OFFSET..+32`
///   * `TX_ROOT[0..32]`         ↔ `bh::COL_TRANSACTIONS_ROOT_OFFSET..+32`
///   * `RECEIPTS_ROOT[0..32]`   ↔ `bh::COL_RECEIPTS_ROOT_OFFSET..+32`
///   * `WITHDRAWALS_ROOT[0..32]`↔ `bh::COL_WITHDRAWALS_ROOT_OFFSET..+32`
///   * `GAS_LIMIT`              ↔ `bh::COL_GAS_LIMIT`
///   * `TIMESTAMP`              ↔ `bh::COL_TIMESTAMP`
///
/// Tuple width = 9 · 1 + 6 · 32 − (5 hashes already counted) — exactly
/// `1 + 32 + 32 + 32 + 32 + 32 + 32 + 1 + 1 = 195` columns. Gated on
/// the A side by `IS_HEADER` (single published anchor per block) and
/// on the B side by `bh::COL_IS_REAL`.
pub fn make_block_to_block_header_descriptor(
    block_layer_index: usize,
    block_header_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;

    let mut a_columns: Vec<usize> = Vec::with_capacity(195);
    let mut b_columns: Vec<usize> = Vec::with_capacity(195);

    a_columns.push(COL_BLOCK_NUMBER);
    b_columns.push(bh::COL_NUMBER);

    for k in 0..HASH_LEN {
        a_columns.push(COL_BLOCK_HASH_OFFSET + k);
        b_columns.push(bh::COL_BLOCK_HASH_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_PARENT_HASH_OFFSET + k);
        b_columns.push(bh::COL_PARENT_HASH_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_STATE_ROOT_OFFSET + k);
        b_columns.push(bh::COL_STATE_ROOT_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_TX_ROOT_OFFSET + k);
        b_columns.push(bh::COL_TRANSACTIONS_ROOT_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_RECEIPTS_ROOT_OFFSET + k);
        b_columns.push(bh::COL_RECEIPTS_ROOT_OFFSET + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_WITHDRAWALS_ROOT_OFFSET + k);
        b_columns.push(bh::COL_WITHDRAWALS_ROOT_OFFSET + k);
    }
    a_columns.push(COL_GAS_LIMIT);
    b_columns.push(bh::COL_GAS_LIMIT);
    a_columns.push(COL_TIMESTAMP);
    b_columns.push(bh::COL_TIMESTAMP);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_to_block_header_v1".into(),
        a_layer_index: block_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_HEADER),
        b_layer_index: block_header_layer_index,
        b_columns,
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

/// Binds `(BLOCK_HASH[0..32], PROPOSER_INDEX_LE_BYTES[0..8])` to the
/// proposer-sig AIR's `(BLOCK_ROOT[0..32], PI_BYTE[0..8])`. Gated by
/// `IS_HEADER` on the A side.
pub fn make_block_to_proposer_sig_descriptor(
    block_layer_index: usize,
    proposer_sig_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_proposer_sig_air as bp;

    let mut a_columns: Vec<usize> = Vec::with_capacity(HASH_LEN + U64_BYTES);
    let mut b_columns: Vec<usize> = Vec::with_capacity(HASH_LEN + U64_BYTES);
    for k in 0..HASH_LEN {
        a_columns.push(COL_BLOCK_HASH_OFFSET + k);
        b_columns.push(bp::COL_BLOCK_ROOT_OFFSET + k);
    }
    for k in 0..U64_BYTES {
        a_columns.push(COL_PROPOSER_INDEX_BYTE_OFFSET + k);
        b_columns.push(bp::COL_PI_BYTE_OFFSET + k);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_to_proposer_sig_v1".into(),
        a_layer_index: block_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_HEADER),
        b_layer_index: proposer_sig_layer_index,
        b_columns,
        b_selector_column: Some(bp::COL_IS_REAL),
    }
}

/// Binds `WITHDRAWALS_ROOT[0..32]` to [`crate::withdrawal_root_air`].
/// Gated by `IS_HEADER` on the A side, by `IS_FIRST` on the B side
/// (matches withdrawal_root_air's single-anchor convention).
pub fn make_block_to_withdrawal_root_descriptor(
    block_layer_index: usize,
    withdrawal_root_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::withdrawal_root_air as wr;
    let a_columns: Vec<usize> = (0..HASH_LEN)
        .map(|k| COL_WITHDRAWALS_ROOT_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = (0..HASH_LEN)
        .map(|k| wr::COL_WITHDRAWALS_ROOT_OFFSET + k)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_to_withdrawal_root_v1".into(),
        a_layer_index: block_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_HEADER),
        b_layer_index: withdrawal_root_layer_index,
        b_columns,
        b_selector_column: Some(wr::COL_IS_FIRST),
    }
}

/// Binds `STATE_ROOT[0..32]` to [`crate::beacon_state_transition_air`]'s
/// `POST_STATE_ROOT`. Gated by `IS_HEADER` on the A side, by
/// `state_transition_air::COL_IS_REAL` on the B side.
pub fn make_block_to_state_transition_descriptor(
    block_layer_index: usize,
    state_transition_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::beacon_state_transition_air as st;
    let a_columns: Vec<usize> =
        (0..HASH_LEN).map(|k| COL_STATE_ROOT_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..HASH_LEN).map(|k| st::COL_POST_STATE_ROOT_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_to_state_transition_v1".into(),
        a_layer_index: block_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_HEADER),
        b_layer_index: state_transition_layer_index,
        b_columns,
        b_selector_column: Some(st::COL_IS_REAL),
    }
}

/// Per-row binding `(TX_INDEX)` on this AIR ↔ `(TX_INDEX)` on
/// [`crate::tx_full_chain_air`]. Gated by `TX_IS_ACTIVE` on the A side
/// and by `tx_full_chain_air::COL_IS_REAL` on the B side.
///
/// **Caveat**: a richer tuple `(TX_INDEX, BLOCK_HASH[0..32],
/// TOTAL_GAS_USED)` would tie each row to its parent block and the
/// per-row gas accumulation. `tx_full_chain_air` currently exposes
/// `TX_INDEX` and `GAS_USED` but no `block_hash` column (block-binding
/// is mediated by the downstream block-header gadget chain). For now
/// this descriptor binds only `TX_INDEX`; the `tx_index` parameter is
/// reserved for a future per-index multiset-fan-out variant.
pub fn make_block_to_tx_full_chain_descriptor(
    block_layer_index: usize,
    tx_full_chain_layer_index: usize,
    tx_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let _ = tx_index;
    use crate::tx_full_chain_air as tfc;
    let a_columns: Vec<usize> = vec![COL_TX_INDEX];
    let b_columns: Vec<usize> = vec![tfc::COL_TX_INDEX];
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_to_tx_full_chain_v1".into(),
        a_layer_index: block_layer_index,
        a_columns,
        a_selector_column: Some(COL_TX_IS_ACTIVE),
        b_layer_index: tx_full_chain_layer_index,
        b_columns,
        b_selector_column: Some(tfc::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_header::BlockHeader;
    use crate::transaction::{LegacyTx, Transaction};

    fn synth_legacy_tx(nonce: u64, gas: u64) -> Transaction {
        let mut gas_price = [0u8; 32];
        gas_price[31] = 1;
        Transaction::Legacy(LegacyTx {
            nonce,
            gas_price,
            gas_limit: gas,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        })
    }

    fn synth_header(num_txs: u32, gas_used: u64, gas_limit: u64) -> BlockHeader {
        let mut h = BlockHeader::default();
        h.number = 12_345;
        h.timestamp = 1_700_000_000;
        h.gas_used = gas_used;
        h.gas_limit = gas_limit;
        h.parent_hash = [0xAAu8; 32];
        h.state_root = [0xBBu8; 32];
        h.transactions_root = [0xCCu8; 32];
        h.receipts_root = [0xDDu8; 32];
        h.withdrawals_root = Some([0xEEu8; 32]);
        let _ = num_txs;
        h
    }

    fn assert_all_constraints_vanish(
        witness: &BlockFullProofWitness,
        curve: CurveType,
    ) {
        let trace = build_trace_polynomials(witness, curve);
        let cs = BlockFullProofConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local body {} ({}) at row {} must vanish (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    /// Empty block: 0 transactions, gas_used = 0 → all constraints
    /// vanish, every TX_IS_ACTIVE is 0.
    #[test]
    fn block_full_proof_air_empty_block() {
        let header = synth_header(0, 0, 30_000_000);
        let w = BlockFullProofWitness::from_block(&header, &[], &[], 99);
        assert_eq!(w.num_transactions, 0);
        assert_eq!(w.total_gas_used, 0);
        assert_eq!(w.gas_slack(), 30_000_000);

        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        // Every row IS_REAL = 1 (composer convention), TX_IS_ACTIVE = 0.
        for r in 0..MAX_TRANSACTIONS {
            assert_eq!(trace.columns[COL_IS_REAL].evaluations[r].to_u64(), 1);
            assert_eq!(
                trace.columns[COL_TX_IS_ACTIVE].evaluations[r].to_u64(),
                0,
                "row {} TX_IS_ACTIVE",
                r,
            );
        }
        // IS_HEADER only on row 0.
        assert_eq!(trace.columns[COL_IS_HEADER].evaluations[0].to_u64(), 1);
        for r in 1..MAX_TRANSACTIONS {
            assert_eq!(trace.columns[COL_IS_HEADER].evaluations[r].to_u64(), 0);
        }

        assert_all_constraints_vanish(&w, curve);
    }

    /// Single-tx block.
    #[test]
    fn block_full_proof_air_single_tx_block() {
        let header = synth_header(1, 21_000, 30_000_000);
        let txs = vec![synth_legacy_tx(0, 21_000)];
        let w = BlockFullProofWitness::from_block(&header, &txs, &[], 7);
        assert_eq!(w.num_transactions, 1);
        assert_eq!(w.total_gas_used, 21_000);
        assert_eq!(w.gas_slack(), 30_000_000 - 21_000);

        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns[COL_TX_IS_ACTIVE].evaluations[0].to_u64(), 1);
        for r in 1..MAX_TRANSACTIONS {
            assert_eq!(
                trace.columns[COL_TX_IS_ACTIVE].evaluations[r].to_u64(),
                0,
                "row {}",
                r,
            );
        }
        assert_all_constraints_vanish(&w, curve);
    }

    /// 4-tx block: TX_IS_ACTIVE = 1 on rows 0..4, 0 elsewhere.
    #[test]
    fn block_full_proof_air_four_tx_block() {
        let header = synth_header(4, 84_000, 30_000_000);
        let txs: Vec<Transaction> =
            (0..4u64).map(|i| synth_legacy_tx(i, 21_000)).collect();
        let w = BlockFullProofWitness::from_block(&header, &txs, &[], 42);
        assert_eq!(w.num_transactions, 4);
        assert_eq!(w.total_gas_used, 84_000);

        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        for r in 0..4 {
            assert_eq!(
                trace.columns[COL_TX_IS_ACTIVE].evaluations[r].to_u64(),
                1,
                "row {} should be active",
                r,
            );
            assert_eq!(
                trace.columns[COL_TX_INDEX].evaluations[r].to_u64(),
                r as u64,
            );
        }
        for r in 4..MAX_TRANSACTIONS {
            assert_eq!(
                trace.columns[COL_TX_IS_ACTIVE].evaluations[r].to_u64(),
                0,
                "row {} should be inactive",
                r,
            );
        }
        assert_all_constraints_vanish(&w, curve);
    }

    /// Tampering: bump GAS_SLACK so `total + slack > limit` and confirm
    /// the `gas_balance` body fires on every active row.
    #[test]
    fn block_full_proof_air_tampered_gas_overflow_detected() {
        let header = synth_header(1, 21_000, 30_000_000);
        let txs = vec![synth_legacy_tx(0, 21_000)];
        let w = BlockFullProofWitness::from_block(&header, &txs, &[], 1);
        let curve = CurveType::Bls48581;
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper GAS_SLACK on row 0: +1.
        let original = trace.columns[COL_GAS_SLACK].evaluations[0].clone();
        let bumped = original.add(&Scalar::one(curve));
        trace.columns[COL_GAS_SLACK].evaluations[0] = bumped;

        let cs = BlockFullProofConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // Body 4 is gas_balance.
        assert!(
            !evals[4][0].is_zero(),
            "gas_balance body must fire on the tampered row",
        );
        // Body 8 is gas_slack_le_decomp: also fires because the scalar
        // no longer matches the LE byte representation.
        assert!(
            !evals[8][0].is_zero(),
            "gas_slack_le_decomp body must fire on the tampered row",
        );
    }

    /// Cross-AIR descriptors are well-formed: shapes, labels, gating.
    #[test]
    fn block_full_proof_air_descriptors_well_formed() {
        // block_header descriptor.
        let d_bh = make_block_to_block_header_descriptor(0, 1);
        assert_eq!(d_bh.label, "block_to_block_header_v1");
        // Tuple width: 1 + 32 + 32 + 32 + 32 + 32 + 32 + 1 + 1 = 195.
        assert_eq!(d_bh.a_columns.len(), 195);
        assert_eq!(d_bh.b_columns.len(), 195);
        assert_eq!(d_bh.a_selector_column, Some(COL_IS_HEADER));
        assert_eq!(
            d_bh.b_selector_column,
            Some(crate::block_header_air::COL_IS_REAL),
        );
        assert_eq!(d_bh.a_columns[0], COL_BLOCK_NUMBER);
        assert_eq!(d_bh.b_columns[0], crate::block_header_air::COL_NUMBER);
        // GAS_LIMIT is the second-to-last A-side column.
        assert_eq!(
            d_bh.a_columns[d_bh.a_columns.len() - 2],
            COL_GAS_LIMIT,
        );
        assert_eq!(
            d_bh.b_columns[d_bh.b_columns.len() - 2],
            crate::block_header_air::COL_GAS_LIMIT,
        );
        // TIMESTAMP is the last A-side column.
        assert_eq!(
            d_bh.a_columns[d_bh.a_columns.len() - 1],
            COL_TIMESTAMP,
        );
        assert_eq!(
            d_bh.b_columns[d_bh.b_columns.len() - 1],
            crate::block_header_air::COL_TIMESTAMP,
        );

        // proposer_sig descriptor.
        let d_ps = make_block_to_proposer_sig_descriptor(0, 2);
        assert_eq!(d_ps.label, "block_to_proposer_sig_v1");
        assert_eq!(d_ps.a_columns.len(), HASH_LEN + U64_BYTES);
        assert_eq!(d_ps.b_columns.len(), HASH_LEN + U64_BYTES);
        for k in 0..HASH_LEN {
            assert_eq!(d_ps.a_columns[k], COL_BLOCK_HASH_OFFSET + k);
            assert_eq!(
                d_ps.b_columns[k],
                crate::block_proposer_sig_air::COL_BLOCK_ROOT_OFFSET + k,
            );
        }
        for k in 0..U64_BYTES {
            assert_eq!(
                d_ps.a_columns[HASH_LEN + k],
                COL_PROPOSER_INDEX_BYTE_OFFSET + k,
            );
            assert_eq!(
                d_ps.b_columns[HASH_LEN + k],
                crate::block_proposer_sig_air::COL_PI_BYTE_OFFSET + k,
            );
        }
        assert_eq!(d_ps.a_selector_column, Some(COL_IS_HEADER));
        assert_eq!(
            d_ps.b_selector_column,
            Some(crate::block_proposer_sig_air::COL_IS_REAL),
        );

        // withdrawal_root descriptor.
        let d_wr = make_block_to_withdrawal_root_descriptor(0, 3);
        assert_eq!(d_wr.label, "block_to_withdrawal_root_v1");
        assert_eq!(d_wr.a_columns.len(), HASH_LEN);
        assert_eq!(d_wr.b_columns.len(), HASH_LEN);
        for k in 0..HASH_LEN {
            assert_eq!(d_wr.a_columns[k], COL_WITHDRAWALS_ROOT_OFFSET + k);
            assert_eq!(
                d_wr.b_columns[k],
                crate::withdrawal_root_air::COL_WITHDRAWALS_ROOT_OFFSET + k,
            );
        }
        assert_eq!(d_wr.a_selector_column, Some(COL_IS_HEADER));
        assert_eq!(
            d_wr.b_selector_column,
            Some(crate::withdrawal_root_air::COL_IS_FIRST),
        );

        // state_transition descriptor.
        let d_st = make_block_to_state_transition_descriptor(0, 4);
        assert_eq!(d_st.label, "block_to_state_transition_v1");
        assert_eq!(d_st.a_columns.len(), HASH_LEN);
        assert_eq!(d_st.b_columns.len(), HASH_LEN);
        for k in 0..HASH_LEN {
            assert_eq!(d_st.a_columns[k], COL_STATE_ROOT_OFFSET + k);
            assert_eq!(
                d_st.b_columns[k],
                crate::beacon_state_transition_air::COL_POST_STATE_ROOT_OFFSET
                    + k,
            );
        }
        assert_eq!(d_st.a_selector_column, Some(COL_IS_HEADER));
        assert_eq!(
            d_st.b_selector_column,
            Some(crate::beacon_state_transition_air::COL_IS_REAL),
        );

        // tx_full_chain descriptor: TX_INDEX ↔ tfc::TX_INDEX.
        let d_tx = make_block_to_tx_full_chain_descriptor(0, 5, 0);
        assert_eq!(d_tx.label, "block_to_tx_full_chain_v1");
        assert_eq!(d_tx.a_columns, vec![COL_TX_INDEX]);
        assert_eq!(
            d_tx.b_columns,
            vec![crate::tx_full_chain_air::COL_TX_INDEX],
        );
        assert_eq!(d_tx.a_layer_index, 0);
        assert_eq!(d_tx.b_layer_index, 5);
        assert_eq!(d_tx.a_selector_column, Some(COL_TX_IS_ACTIVE));
        assert_eq!(
            d_tx.b_selector_column,
            Some(crate::tx_full_chain_air::COL_IS_REAL),
        );

        // Eight enumerated tx descriptors share the A-side gating.
        for tx_index in 0..8 {
            let d = make_block_to_tx_full_chain_descriptor(0, 5, tx_index);
            assert_eq!(d.a_selector_column, Some(COL_TX_IS_ACTIVE));
        }
    }

    /// Column layout pin: NUM_COLUMNS and key offsets are exactly as
    /// documented in the module header so downstream cross-AIR wiring
    /// doesn't drift silently.
    #[test]
    fn block_full_proof_air_column_layout_pinned() {
        assert_eq!(COL_BLOCK_NUMBER, 0);
        assert_eq!(COL_BLOCK_NUMBER_BYTE_OFFSET, 1);
        assert_eq!(COL_BLOCK_HASH_OFFSET, 9);
        assert_eq!(COL_PARENT_HASH_OFFSET, 41);
        assert_eq!(COL_STATE_ROOT_OFFSET, 73);
        assert_eq!(COL_TX_ROOT_OFFSET, 105);
        assert_eq!(COL_RECEIPTS_ROOT_OFFSET, 137);
        assert_eq!(COL_WITHDRAWALS_ROOT_OFFSET, 169);
        assert_eq!(COL_NUM_TRANSACTIONS, 201);
        assert_eq!(COL_NUM_TRANSACTIONS_BYTE_OFFSET, 202);
        assert_eq!(COL_TOTAL_GAS_USED, 206);
        assert_eq!(COL_TOTAL_GAS_USED_BYTE_OFFSET, 207);
        assert_eq!(COL_GAS_LIMIT, 215);
        assert_eq!(COL_GAS_LIMIT_BYTE_OFFSET, 216);
        assert_eq!(COL_GAS_SLACK, 224);
        assert_eq!(COL_GAS_SLACK_BYTE_OFFSET, 225);
        assert_eq!(COL_TIMESTAMP, 233);
        assert_eq!(COL_TIMESTAMP_BYTE_OFFSET, 234);
        assert_eq!(COL_PROPOSER_INDEX, 242);
        assert_eq!(COL_PROPOSER_INDEX_BYTE_OFFSET, 243);
        assert_eq!(COL_IS_REAL, 251);
        assert_eq!(COL_IS_HEADER, 252);
        assert_eq!(COL_TX_INDEX, 253);
        assert_eq!(COL_TX_IS_ACTIVE, 254);
        assert_eq!(NUM_COLUMNS, 255);
        assert_eq!(NUM_ROW_CONSTRAINTS, 12);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(MAX_TRANSACTIONS, 16);
    }
}
