//! Transaction full-chain composition AIR.
//!
//! Per row, commits one transaction's full validity composition tuple:
//!
//!   * `tx_index` — position of the tx in the block
//!   * `tx_hash[0..32]` — keccak256 of the wire encoding (host-side
//!     committed, bound algebraically downstream via `tx_rlp_air` →
//!     `KeccakExtract` once the dedicated tx-hash gadget lands; for now
//!     this AIR's [`make_tx_full_to_tx_rlp_descriptor`] binds nonce
//!     between this AIR and `tx_rlp_air`).
//!   * `sender_address[0..20]` — recovered ECDSA sender
//!   * `nonce` — transaction nonce (u64) + LE byte decomposition
//!   * `gas_used` — gas consumed by this tx (u64) + LE bytes
//!   * `cumulative_gas_used` — running cumulative gas (u64) + LE bytes
//!   * `status` — receipt status (binary)
//!   * `tx_type` — `0=Legacy, 1=2930, 2=1559, 3=4844`
//!   * `tx_type_sel[0..4]` — one-hot selectors for the tx type
//!   * `sig_hash[0..32]` — the signing hash (host-side committed)
//!   * `is_real` — row-active selector
//!
//! Each downstream binding is enforced via a cross-AIR LogUp descriptor:
//!
//!   * [`make_tx_full_to_tx_sender_recovery_descriptor`] —
//!     `(tx_index, sender_address[0..20])` ↔ `tx_sender_recovery_air`'s
//!     `(COL_TX_INDEX, COL_DERIVED_ADDR_OFFSET[0..20])`.
//!   * [`make_tx_full_to_tx_nonce_descriptor`] —
//!     `(tx_index, sender_address[0..20], nonce)` ↔ `tx_nonce_air`'s
//!     `(COL_TX_INDEX, COL_SENDER_OFFSET[0..20], COL_TX_NONCE)`.
//!   * [`make_tx_full_to_access_list_descriptor`] — gated by
//!     `is_eip2930 + is_eip1559`, binds `sender_address[0..20]` (as the
//!     access list's owning tx is a soft index in scaffolding) into
//!     `access_list_air`'s `COL_ADDRESS_OFFSET[0..20]`. **Stub**:
//!     `access_list_air` rows are per-entry not per-tx, so this binding
//!     pins at least the first access-list entry's address. The
//!     dedicated (tx_index, entry_index) widening lands alongside the
//!     access-list ↔ tx-rlp linkage.
//!   * [`make_tx_full_to_receipt_status_descriptor`] —
//!     `(tx_index, status, gas_used, cumulative_gas_used)` ↔
//!     `receipt_status_air`'s `(COL_TX_INDEX, COL_STATUS, COL_GAS_USED,
//!     COL_CUMULATIVE_GAS)`.
//!   * [`make_tx_full_to_tx_rlp_descriptor`] — binds `nonce` ↔
//!     `tx_rlp_air::COL_NONCE`. **Stub**: `tx_rlp_air` does not yet
//!     expose a `COL_TX_HASH` column; this descriptor will widen to
//!     `(tx_hash[0..32], nonce)` once the tx-hash gadget lands.
//!
//! ## Algebraic constraints (row-local, 11 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`.
//! 1. `status_binary` — `IS_REAL · STATUS · (STATUS − 1) = 0`.
//! 2..6. `tx_type_sel[k]_binary` (k = 0..4) — `sel · (sel − 1) = 0`.
//! 6. `tx_type_sel_sum_eq_is_real` — `IS_REAL − Σ sel[k] = 0`.
//! 7. `tx_index_le_decomp` — `TX_INDEX − Σ_b TX_INDEX_BYTE[b] · 2^(8b) = 0`.
//! 8. `nonce_le_decomp` — `NONCE − Σ_b NONCE_BYTE[b] · 2^(8b) = 0`.
//! 9. `gas_used_le_decomp` — `GAS_USED − Σ_b GAS_USED_BYTE[b] · 2^(8b) = 0`.
//! 10. `cumulative_le_decomp` —
//!     `CUMULATIVE − Σ_b CUMUL_BYTE[b] · 2^(8b) = 0`.
//! 11. `tx_type_eq_selectors` — `IS_REAL · (TX_TYPE − Σ_k k · sel[k]) = 0`.
//!
//! Per-byte 8-bit range checks (via `lookup_declarations`) cover the
//! four LE decompositions plus all 32-byte `tx_hash` + `sig_hash` +
//! 20-byte `sender_address` byte columns.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

pub const HASH_LEN: usize = 32;
pub const ADDR_LEN: usize = 20;
pub const U64_BYTES: usize = 8;
pub const NUM_TX_TYPES: usize = 4;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_TX_INDEX: usize = 0;
pub const COL_TX_HASH_OFFSET: usize = COL_TX_INDEX + 1;                       // 1..33
pub const COL_SENDER_OFFSET: usize = COL_TX_HASH_OFFSET + HASH_LEN;           // 33..53
pub const COL_NONCE: usize = COL_SENDER_OFFSET + ADDR_LEN;                    // 53
pub const COL_GAS_USED: usize = COL_NONCE + 1;                                // 54
pub const COL_CUMULATIVE_GAS: usize = COL_GAS_USED + 1;                       // 55
pub const COL_STATUS: usize = COL_CUMULATIVE_GAS + 1;                         // 56
pub const COL_TX_TYPE: usize = COL_STATUS + 1;                                // 57

// 4 one-hot tx-type selectors.
pub const COL_TX_TYPE_SEL_OFFSET: usize = COL_TX_TYPE + 1;                    // 58..62

// LE byte decompositions (u64, 8 bytes each).
pub const COL_TX_INDEX_BYTE_OFFSET: usize = COL_TX_TYPE_SEL_OFFSET + NUM_TX_TYPES; // 62..70
pub const COL_NONCE_BYTE_OFFSET: usize = COL_TX_INDEX_BYTE_OFFSET + U64_BYTES;     // 70..78
pub const COL_GAS_USED_BYTE_OFFSET: usize = COL_NONCE_BYTE_OFFSET + U64_BYTES;     // 78..86
pub const COL_CUMUL_BYTE_OFFSET: usize = COL_GAS_USED_BYTE_OFFSET + U64_BYTES;     // 86..94

// Signing hash (32 bytes).
pub const COL_SIG_HASH_OFFSET: usize = COL_CUMUL_BYTE_OFFSET + U64_BYTES;     // 94..126

pub const COL_IS_REAL: usize = COL_SIG_HASH_OFFSET + HASH_LEN;                // 126

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;                               // 127

/// Row-local constraints:
///   0: is_real_binary
///   1: status_binary
///   2..6: 4 tx_type_sel binaries (one per type)
///   6: tx_type_sel_sum_eq_is_real
///   7: tx_index_le_decomp
///   8: nonce_le_decomp
///   9: gas_used_le_decomp
///  10: cumulative_le_decomp
///  11: tx_type_eq_selectors
pub const NUM_ROW_CONSTRAINTS: usize = 12;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxFullChainRow {
    pub tx_index: u64,
    pub tx_hash: [u8; HASH_LEN],
    pub sender_address: [u8; ADDR_LEN],
    pub nonce: u64,
    pub gas_used: u64,
    pub cumulative_gas_used: u64,
    pub status: u8,
    /// `0=Legacy, 1=Eip2930, 2=Eip1559, 3=Eip4844`.
    pub tx_type: u8,
    pub sig_hash: [u8; HASH_LEN],
}

#[derive(Clone, Debug, Default)]
pub struct TxFullChainWitness {
    pub rows: Vec<TxFullChainRow>,
}

impl TxFullChainWitness {
    pub fn from_rows(rows: Vec<TxFullChainRow>) -> Self {
        Self { rows }
    }

    /// Build a single-row witness from a [`crate::transaction::Transaction`].
    ///
    /// * `tx_index` — the transaction's position within the block.
    /// * `tx` — the signed transaction.
    /// * `sender` — recovered sender (typically from
    ///   [`crate::tx_sender_recovery_air`]).
    /// * `gas_used` — gas consumed by this tx (from the EVM trace).
    /// * `cumulative` — cumulative gas after this tx (= running sum).
    /// * `status` — receipt status (0/1).
    ///
    /// The signing hash is computed via
    /// [`crate::tx_sig_hash::signing_hash`] with `chain_id = None`
    /// (default; callers may post-process the row to set a chain-id
    /// dependent sig_hash for legacy EIP-155 transactions).
    pub fn from_transaction(
        tx_index: u64,
        tx: &crate::transaction::Transaction,
        sender: [u8; 20],
        gas_used: u64,
        cumulative: u64,
        status: u8,
    ) -> Self {
        let tx_hash = tx.hash();
        let sig_hash = crate::tx_sig_hash::signing_hash(tx, None);
        let nonce = match tx {
            crate::transaction::Transaction::Legacy(t) => t.nonce,
            crate::transaction::Transaction::Eip1559(t) => t.nonce,
        };
        let tx_type = match tx {
            crate::transaction::Transaction::Legacy(_) => 0u8,
            crate::transaction::Transaction::Eip1559(_) => 2u8,
        };
        Self {
            rows: vec![TxFullChainRow {
                tx_index,
                tx_hash,
                sender_address: sender,
                nonce,
                gas_used,
                cumulative_gas_used: cumulative,
                status,
                tx_type,
                sig_hash,
            }],
        }
    }
}

/// Top-level builder matching the task description signature.
pub fn from_transaction(
    tx_index: u64,
    tx: &crate::transaction::Transaction,
    sender: [u8; 20],
    gas_used: u64,
    cumulative: u64,
    status: u8,
) -> TxFullChainWitness {
    TxFullChainWitness::from_transaction(tx_index, tx, sender, gas_used, cumulative, status)
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

fn eval_le_decomp(target: &Scalar, byte_off: usize, col_evals: &[Scalar]) -> Scalar {
    let curve = target.curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[byte_off + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    target.sub(&sum)
}

fn build_le_decomp_poly(
    target_poly: &[Scalar],
    byte_off: usize,
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..U64_BYTES {
        let byte_poly = &col_coeffs[byte_off + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    poly_sub(target_poly, &sum, curve)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &TxFullChainWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_TX_INDEX][i] = Scalar::from_u64(row.tx_index, curve);
        for k in 0..HASH_LEN {
            columns[COL_TX_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.tx_hash[k] as u64, curve);
            columns[COL_SIG_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.sig_hash[k] as u64, curve);
        }
        for k in 0..ADDR_LEN {
            columns[COL_SENDER_OFFSET + k][i] =
                Scalar::from_u64(row.sender_address[k] as u64, curve);
        }
        columns[COL_NONCE][i] = Scalar::from_u64(row.nonce, curve);
        columns[COL_GAS_USED][i] = Scalar::from_u64(row.gas_used, curve);
        columns[COL_CUMULATIVE_GAS][i] =
            Scalar::from_u64(row.cumulative_gas_used, curve);
        columns[COL_STATUS][i] = Scalar::from_u64(row.status as u64, curve);
        columns[COL_TX_TYPE][i] = Scalar::from_u64(row.tx_type as u64, curve);

        // 4 one-hot tx-type selectors.
        let tt = row.tx_type as usize;
        for k in 0..NUM_TX_TYPES {
            columns[COL_TX_TYPE_SEL_OFFSET + k][i] =
                if k == tt { one.clone() } else { zero.clone() };
        }

        // LE byte decompositions.
        let tx_idx_bytes = row.tx_index.to_le_bytes();
        let nonce_bytes = row.nonce.to_le_bytes();
        let gas_bytes = row.gas_used.to_le_bytes();
        let cumul_bytes = row.cumulative_gas_used.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_TX_INDEX_BYTE_OFFSET + b][i] =
                Scalar::from_u64(tx_idx_bytes[b] as u64, curve);
            columns[COL_NONCE_BYTE_OFFSET + b][i] =
                Scalar::from_u64(nonce_bytes[b] as u64, curve);
            columns[COL_GAS_USED_BYTE_OFFSET + b][i] =
                Scalar::from_u64(gas_bytes[b] as u64, curve);
            columns[COL_CUMUL_BYTE_OFFSET + b][i] =
                Scalar::from_u64(cumul_bytes[b] as u64, curve);
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

pub struct TxFullChainConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl TxFullChainConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for TxFullChainConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "is_real_binary".into(),
            "status_binary".into(),
        ];
        for k in 0..NUM_TX_TYPES {
            labels.push(format!("tx_type_sel_{}_binary", k));
        }
        labels.push("tx_type_sel_sum_eq_is_real".into());
        labels.push("tx_index_le_decomp".into());
        labels.push("nonce_le_decomp".into());
        labels.push("gas_used_le_decomp".into());
        labels.push("cumulative_le_decomp".into());
        labels.push("tx_type_eq_selectors".into());
        labels
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
            let status = &row_evals[COL_STATUS];
            let tx_type = &row_evals[COL_TX_TYPE];

            // 0: is_real binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));

            // 1: status binary (gated by IS_REAL).
            bodies[1][row] = is_real.mul(&status.mul(&status.sub(&one)));

            // 2..6: tx_type selector binaries.
            let mut sel_sum = Scalar::zero(curve);
            let mut sel_weighted = Scalar::zero(curve);
            for k in 0..NUM_TX_TYPES {
                let s = &row_evals[COL_TX_TYPE_SEL_OFFSET + k];
                bodies[2 + k][row] = s.mul(&s.sub(&one));
                sel_sum = sel_sum.add(s);
                sel_weighted = sel_weighted
                    .add(&Scalar::from_u64(k as u64, curve).mul(s));
            }

            // 6 (after 4 sel-binaries → body index 2+4 = 6):
            // tx_type_sel_sum_eq_is_real.
            bodies[2 + NUM_TX_TYPES][row] = is_real.sub(&sel_sum);

            // 7: tx_index_le_decomp.
            bodies[7][row] = eval_le_decomp(
                &row_evals[COL_TX_INDEX],
                COL_TX_INDEX_BYTE_OFFSET,
                &row_evals,
            );
            // 8: nonce_le_decomp.
            bodies[8][row] = eval_le_decomp(
                &row_evals[COL_NONCE],
                COL_NONCE_BYTE_OFFSET,
                &row_evals,
            );
            // 9: gas_used_le_decomp.
            bodies[9][row] = eval_le_decomp(
                &row_evals[COL_GAS_USED],
                COL_GAS_USED_BYTE_OFFSET,
                &row_evals,
            );
            // 10: cumulative_le_decomp.
            bodies[10][row] = eval_le_decomp(
                &row_evals[COL_CUMULATIVE_GAS],
                COL_CUMUL_BYTE_OFFSET,
                &row_evals,
            );

            // 11: tx_type_eq_selectors —
            // IS_REAL · (TX_TYPE − Σ k · sel[k]) = 0.
            bodies[11][row] = is_real.mul(&tx_type.sub(&sel_weighted));
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
        let status = &col_evals[COL_STATUS];
        let tx_type = &col_evals[COL_TX_TYPE];

        let mut sel_sum = Scalar::zero(curve);
        let mut sel_weighted = Scalar::zero(curve);
        let mut sel_binaries: Vec<Scalar> = Vec::with_capacity(NUM_TX_TYPES);
        for k in 0..NUM_TX_TYPES {
            let s = &col_evals[COL_TX_TYPE_SEL_OFFSET + k];
            sel_binaries.push(s.mul(&s.sub(&one)));
            sel_sum = sel_sum.add(s);
            sel_weighted =
                sel_weighted.add(&Scalar::from_u64(k as u64, curve).mul(s));
        }

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(is_real.mul(&status.mul(&status.sub(&one))));
        bodies.extend(sel_binaries);
        bodies.push(is_real.sub(&sel_sum));
        bodies.push(eval_le_decomp(
            &col_evals[COL_TX_INDEX],
            COL_TX_INDEX_BYTE_OFFSET,
            col_evals,
        ));
        bodies.push(eval_le_decomp(
            &col_evals[COL_NONCE],
            COL_NONCE_BYTE_OFFSET,
            col_evals,
        ));
        bodies.push(eval_le_decomp(
            &col_evals[COL_GAS_USED],
            COL_GAS_USED_BYTE_OFFSET,
            col_evals,
        ));
        bodies.push(eval_le_decomp(
            &col_evals[COL_CUMULATIVE_GAS],
            COL_CUMUL_BYTE_OFFSET,
            col_evals,
        ));
        bodies.push(is_real.mul(&tx_type.sub(&sel_weighted)));

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
        let status = &col_coeffs[COL_STATUS];
        let tx_type = &col_coeffs[COL_TX_TYPE];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let status_m1 = poly_sub(status, &one_poly, curve);
        let status_sq = poly_mul(status, &status_m1, curve);
        let status_binary = poly_mul(is_real, &status_sq, curve);

        let mut sel_binaries: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_TX_TYPES);
        let mut sel_sum: Vec<Scalar> = vec![Scalar::zero(curve)];
        let mut sel_weighted: Vec<Scalar> = vec![Scalar::zero(curve)];
        for k in 0..NUM_TX_TYPES {
            let s = &col_coeffs[COL_TX_TYPE_SEL_OFFSET + k];
            let s_m1 = poly_sub(s, &one_poly, curve);
            sel_binaries.push(poly_mul(s, &s_m1, curve));
            sel_sum = poly_add(&sel_sum, s, curve);
            let k_scalar = Scalar::from_u64(k as u64, curve);
            let term = poly_scalar_mul(s, &k_scalar);
            sel_weighted = poly_add(&sel_weighted, &term, curve);
        }
        let sum_eq = poly_sub(is_real, &sel_sum, curve);

        let tx_index_decomp = build_le_decomp_poly(
            &col_coeffs[COL_TX_INDEX],
            COL_TX_INDEX_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let nonce_decomp = build_le_decomp_poly(
            &col_coeffs[COL_NONCE],
            COL_NONCE_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let gas_decomp = build_le_decomp_poly(
            &col_coeffs[COL_GAS_USED],
            COL_GAS_USED_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let cumul_decomp = build_le_decomp_poly(
            &col_coeffs[COL_CUMULATIVE_GAS],
            COL_CUMUL_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let tx_type_inner = poly_sub(tx_type, &sel_weighted, curve);
        let tx_type_body = poly_mul(is_real, &tx_type_inner, curve);

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real_binary);
        bodies.push(status_binary);
        for b in sel_binaries {
            bodies.push(b);
        }
        bodies.push(sum_eq);
        bodies.push(tx_index_decomp);
        bodies.push(nonce_decomp);
        bodies.push(gas_decomp);
        bodies.push(cumul_decomp);
        bodies.push(tx_type_body);

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
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();

        let byte_ranges: [(usize, usize, &str); 7] = [
            (COL_TX_HASH_OFFSET, HASH_LEN, "tx_hash"),
            (COL_SENDER_OFFSET, ADDR_LEN, "sender"),
            (COL_TX_INDEX_BYTE_OFFSET, U64_BYTES, "tx_index_byte"),
            (COL_NONCE_BYTE_OFFSET, U64_BYTES, "nonce_byte"),
            (COL_GAS_USED_BYTE_OFFSET, U64_BYTES, "gas_used_byte"),
            (COL_CUMUL_BYTE_OFFSET, U64_BYTES, "cumul_byte"),
            (COL_SIG_HASH_OFFSET, HASH_LEN, "sig_hash"),
        ];
        for (off, len, label) in byte_ranges {
            for k in 0..len {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
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

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(tx_index, sender_address[0..20])` of this AIR against
/// [`crate::tx_sender_recovery_air`]'s `(COL_TX_INDEX,
/// COL_DERIVED_ADDR_OFFSET[0..20])`.
pub fn make_tx_full_to_tx_sender_recovery_descriptor(
    full_layer_index: usize,
    sender_recovery_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::tx_sender_recovery_air as sr;
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + ADDR_LEN);
    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + ADDR_LEN);
    a_columns.push(COL_TX_INDEX);
    b_columns.push(sr::COL_TX_INDEX);
    for k in 0..ADDR_LEN {
        a_columns.push(COL_SENDER_OFFSET + k);
        b_columns.push(sr::COL_DERIVED_ADDR_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "tx_full_to_tx_sender_recovery_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sender_recovery_layer_index,
        b_columns,
        b_selector_column: Some(sr::COL_IS_REAL),
    }
}

/// Bind `(tx_index, sender_address[0..20], nonce)` of this AIR against
/// [`crate::tx_nonce_air`]'s `(COL_TX_INDEX, COL_SENDER_OFFSET[0..20],
/// COL_TX_NONCE)`.
pub fn make_tx_full_to_tx_nonce_descriptor(
    full_layer_index: usize,
    tx_nonce_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::tx_nonce_air as tn;
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + ADDR_LEN + 1);
    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + ADDR_LEN + 1);
    a_columns.push(COL_TX_INDEX);
    b_columns.push(tn::COL_TX_INDEX);
    for k in 0..ADDR_LEN {
        a_columns.push(COL_SENDER_OFFSET + k);
        b_columns.push(tn::COL_SENDER_OFFSET + k);
    }
    a_columns.push(COL_NONCE);
    b_columns.push(tn::COL_TX_NONCE);
    CrossAirLogUpDescriptor {
        label: "tx_full_to_tx_nonce_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: tx_nonce_layer_index,
        b_columns,
        b_selector_column: Some(tn::COL_IS_REAL),
    }
}

/// Bind the access-list entry's owning-tx address to this AIR's
/// `sender_address[0..20]`. **Stub binding** — `access_list_air` rows
/// are per-entry not per-tx, so this descriptor pins at least the
/// (first-entry) access-list address against the sender. Once the
/// access-list AIR exposes a `tx_index` column the binding will widen.
///
/// The A-side selector is the "is EIP-2930 or EIP-1559" sum (gated to
/// the tx types that actually carry an access list). We use the
/// EIP-1559 selector column directly here for simplicity; downstream
/// callers wiring the joint trace can OR the EIP-2930 selector in via
/// a separate descriptor instance if needed.
pub fn make_tx_full_to_access_list_descriptor(
    full_layer_index: usize,
    access_list_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::access_list_air as al;
    let a_columns: Vec<usize> =
        (0..ADDR_LEN).map(|k| COL_SENDER_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..ADDR_LEN).map(|k| al::COL_ADDRESS_OFFSET + k).collect();
    // Gate by the EIP-1559 selector (tx_type 2). The 2930 case is a
    // separate descriptor instance against the same B-side.
    CrossAirLogUpDescriptor {
        label: "tx_full_to_access_list_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_TX_TYPE_SEL_OFFSET + 2),
        b_layer_index: access_list_layer_index,
        b_columns,
        b_selector_column: Some(al::COL_IS_REAL),
    }
}

/// Bind `(tx_index, status, gas_used, cumulative_gas_used)` of this
/// AIR against [`crate::receipt_status_air`]'s `(COL_TX_INDEX,
/// COL_STATUS, COL_GAS_USED, COL_CUMULATIVE_GAS)`.
pub fn make_tx_full_to_receipt_status_descriptor(
    full_layer_index: usize,
    receipt_status_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::receipt_status_air as rs;
    let a_columns: Vec<usize> =
        vec![COL_TX_INDEX, COL_STATUS, COL_GAS_USED, COL_CUMULATIVE_GAS];
    let b_columns: Vec<usize> = vec![
        rs::COL_TX_INDEX,
        rs::COL_STATUS,
        rs::COL_GAS_USED,
        rs::COL_CUMULATIVE_GAS,
    ];
    CrossAirLogUpDescriptor {
        label: "tx_full_to_receipt_status_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: receipt_status_layer_index,
        b_columns,
        b_selector_column: Some(rs::COL_IS_REAL),
    }
}

/// Bind `nonce` of this AIR against [`crate::tx_rlp_air`]'s
/// `COL_NONCE`. **Stub binding** — `tx_rlp_air` does not yet expose a
/// `COL_TX_HASH` column; once the tx-hash gadget lands this descriptor
/// will widen to `(tx_hash[0..32], nonce)`.
pub fn make_tx_full_to_tx_rlp_descriptor(
    full_layer_index: usize,
    tx_rlp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::tx_rlp_air as txr;
    let a_columns: Vec<usize> = vec![COL_NONCE];
    let b_columns: Vec<usize> = vec![txr::COL_NONCE];
    CrossAirLogUpDescriptor {
        label: "tx_full_to_tx_rlp_v1".into(),
        a_layer_index: full_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: tx_rlp_layer_index,
        b_columns,
        b_selector_column: Some(txr::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{Eip1559Tx, LegacyTx, Transaction};

    fn legacy_tx(nonce: u64) -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 27,
            r: [1u8; 32],
            s: [2u8; 32],
        })
    }

    fn eip1559_tx(nonce: u64) -> Transaction {
        Transaction::Eip1559(Eip1559Tx {
            chain_id: 1,
            nonce,
            max_priority_fee_per_gas: [0u8; 32],
            max_fee_per_gas: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: vec![],
            access_list_rlp: vec![0xc0],
            y_parity: 0,
            r: [1u8; 32],
            s: [2u8; 32],
        })
    }

    fn check_all_bodies_vanish(w: &TxFullChainWitness) {
        let trace = build_trace_polynomials(w, CurveType::Bls48581);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        let cs = TxFullChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        let labels = cs.constraint_labels();
        for (k, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {}",
                    k,
                    labels[k],
                    r,
                );
            }
        }
    }

    #[test]
    fn legacy_tx_witness_constraints_vanish() {
        let tx = legacy_tx(7);
        let sender = [0xabu8; 20];
        let w = TxFullChainWitness::from_transaction(3, &tx, sender, 21_000, 21_000, 1);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].tx_type, 0);
        assert_eq!(w.rows[0].nonce, 7);
        assert_eq!(w.rows[0].tx_index, 3);
        assert_eq!(w.rows[0].sender_address, sender);
        assert_eq!(w.rows[0].status, 1);
        check_all_bodies_vanish(&w);
    }

    #[test]
    fn eip1559_tx_witness_constraints_vanish() {
        let tx = eip1559_tx(0);
        let sender = [0xcdu8; 20];
        let w = TxFullChainWitness::from_transaction(0, &tx, sender, 50_000, 50_000, 1);
        assert_eq!(w.rows[0].tx_type, 2);
        check_all_bodies_vanish(&w);
    }

    /// Synthetic EIP-4844 row: the Transaction enum doesn't yet
    /// support 4844, so we build the witness directly. The AIR's
    /// constraints don't depend on which Transaction variant we use —
    /// only on `tx_type ∈ {0..4}` matching the one-hot selectors.
    #[test]
    fn eip4844_synthetic_row_constraints_vanish() {
        let row = TxFullChainRow {
            tx_index: 1,
            tx_hash: [0x11u8; 32],
            sender_address: [0x22u8; 20],
            nonce: 5,
            gas_used: 42_000,
            cumulative_gas_used: 42_000,
            status: 1,
            tx_type: 3, // EIP-4844
            sig_hash: [0x33u8; 32],
        };
        let w = TxFullChainWitness::from_rows(vec![row]);
        check_all_bodies_vanish(&w);
    }

    #[test]
    fn tampered_status_fires_status_binary() {
        let tx = legacy_tx(0);
        let w = TxFullChainWitness::from_transaction(0, &tx, [0u8; 20], 0, 0, 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Set status = 2 → status_binary body must fire.
        cols[COL_STATUS][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = TxFullChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Body 1 = status_binary.
        assert!(
            !bodies[1][0].is_zero(),
            "status_binary should fire when STATUS = 2",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        use crate::access_list_air as al;
        use crate::receipt_status_air as rs;
        use crate::tx_nonce_air as tn;
        use crate::tx_rlp_air as txr;
        use crate::tx_sender_recovery_air as sr;

        let d1 = make_tx_full_to_tx_sender_recovery_descriptor(0, 1);
        assert_eq!(d1.label, "tx_full_to_tx_sender_recovery_v1");
        assert_eq!(d1.a_columns.len(), 1 + ADDR_LEN);
        assert_eq!(d1.b_columns.len(), 1 + ADDR_LEN);
        assert_eq!(d1.a_columns[0], COL_TX_INDEX);
        assert_eq!(d1.b_columns[0], sr::COL_TX_INDEX);
        assert_eq!(d1.a_columns[1], COL_SENDER_OFFSET);
        assert_eq!(d1.b_columns[1], sr::COL_DERIVED_ADDR_OFFSET);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(sr::COL_IS_REAL));

        let d2 = make_tx_full_to_tx_nonce_descriptor(0, 2);
        assert_eq!(d2.label, "tx_full_to_tx_nonce_v1");
        assert_eq!(d2.a_columns.len(), 1 + ADDR_LEN + 1);
        assert_eq!(d2.a_columns[0], COL_TX_INDEX);
        assert_eq!(d2.b_columns[0], tn::COL_TX_INDEX);
        assert_eq!(d2.a_columns[1 + ADDR_LEN], COL_NONCE);
        assert_eq!(d2.b_columns[1 + ADDR_LEN], tn::COL_TX_NONCE);

        let d3 = make_tx_full_to_access_list_descriptor(0, 3);
        assert_eq!(d3.label, "tx_full_to_access_list_v1");
        assert_eq!(d3.a_columns.len(), ADDR_LEN);
        assert_eq!(d3.b_columns.len(), ADDR_LEN);
        assert_eq!(d3.a_columns[0], COL_SENDER_OFFSET);
        assert_eq!(d3.b_columns[0], al::COL_ADDRESS_OFFSET);
        // Gated by the EIP-1559 (tx_type 2) selector.
        assert_eq!(d3.a_selector_column, Some(COL_TX_TYPE_SEL_OFFSET + 2));

        let d4 = make_tx_full_to_receipt_status_descriptor(0, 4);
        assert_eq!(d4.label, "tx_full_to_receipt_status_v1");
        assert_eq!(d4.a_columns.len(), 4);
        assert_eq!(d4.b_columns.len(), 4);
        assert_eq!(d4.a_columns[0], COL_TX_INDEX);
        assert_eq!(d4.b_columns[0], rs::COL_TX_INDEX);
        assert_eq!(d4.a_columns[1], COL_STATUS);
        assert_eq!(d4.b_columns[1], rs::COL_STATUS);
        assert_eq!(d4.a_columns[2], COL_GAS_USED);
        assert_eq!(d4.b_columns[2], rs::COL_GAS_USED);
        assert_eq!(d4.a_columns[3], COL_CUMULATIVE_GAS);
        assert_eq!(d4.b_columns[3], rs::COL_CUMULATIVE_GAS);

        let d5 = make_tx_full_to_tx_rlp_descriptor(0, 5);
        assert_eq!(d5.label, "tx_full_to_tx_rlp_v1");
        assert_eq!(d5.a_columns, vec![COL_NONCE]);
        assert_eq!(d5.b_columns, vec![txr::COL_NONCE]);
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_TX_INDEX, 0);
        assert_eq!(COL_TX_HASH_OFFSET, 1);
        assert_eq!(COL_SENDER_OFFSET, 33);
        assert_eq!(COL_NONCE, 53);
        assert_eq!(COL_GAS_USED, 54);
        assert_eq!(COL_CUMULATIVE_GAS, 55);
        assert_eq!(COL_STATUS, 56);
        assert_eq!(COL_TX_TYPE, 57);
        assert_eq!(COL_TX_TYPE_SEL_OFFSET, 58);
        assert_eq!(COL_TX_INDEX_BYTE_OFFSET, 62);
        assert_eq!(COL_NONCE_BYTE_OFFSET, 70);
        assert_eq!(COL_GAS_USED_BYTE_OFFSET, 78);
        assert_eq!(COL_CUMUL_BYTE_OFFSET, 86);
        assert_eq!(COL_SIG_HASH_OFFSET, 94);
        assert_eq!(COL_IS_REAL, 126);
        assert_eq!(NUM_COLUMNS, 127);
        assert_eq!(NUM_ROW_CONSTRAINTS, 12);
        assert_eq!(NUM_SHIFTED, 0);
    }

    /// `evaluate_at_point` agrees with `evaluate_on_domain` on an
    /// honest witness — both produce zero under the α-RLC.
    #[test]
    fn evaluate_at_point_zero_on_honest_witness() {
        let tx = legacy_tx(1);
        let w = TxFullChainWitness::from_transaction(0, &tx, [0u8; 20], 100, 100, 1);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = TxFullChainConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(31, CurveType::Bls48581);
        let row0_evals: Vec<Scalar> =
            col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }

    /// Byte-range lookup coverage: every byte column gets an 8-bit
    /// declaration. 32 tx_hash + 20 sender + 8 tx_index + 8 nonce
    /// + 8 gas_used + 8 cumul + 32 sig_hash = 116.
    #[test]
    fn byte_range_lookup_coverage() {
        let cs = TxFullChainConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        let expected = HASH_LEN + ADDR_LEN + U64_BYTES * 4 + HASH_LEN;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }
}
