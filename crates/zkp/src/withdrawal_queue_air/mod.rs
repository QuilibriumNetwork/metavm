//! EIP-4895 withdrawal queue AIR.
//!
//! Proves that each `Withdrawal { index, validator_index, address,
//! amount }` in a block's execution payload corresponds to a beacon-side
//! withdrawal entry, and that the amount transferred to the execution
//! address has been correctly debited from the validator's balance.
//!
//! ## Row layout
//!
//! One row per withdrawal slot. The AIR caps the row count at
//! [`MAX_WITHDRAWALS`] = `MAX_WITHDRAWALS_PER_PAYLOAD` = 16.
//!
//! Committed columns:
//!
//!   * `INDEX` — withdrawal index (u64)
//!   * `VALIDATOR_INDEX` — validator index (u64)
//!   * `ADDRESS[0..20]` — execution-layer recipient address
//!   * `AMOUNT` — withdrawal amount in Gwei (u64)
//!   * `VALIDATOR_BALANCE_PRE` — beacon-side validator balance prior to
//!     the withdrawal (u64, Gwei)
//!   * `VALIDATOR_BALANCE_POST` — beacon-side validator balance after
//!     the withdrawal (u64, Gwei, = pre − amount)
//!   * 4×8-byte LE byte decompositions for the four u64 columns
//!     (`INDEX`, `VALIDATOR_INDEX`, `AMOUNT`, `VALIDATOR_BALANCE_PRE`)
//!   * `IS_REAL` (binary) — selector for active rows
//!   * `IS_FIRST` (binary) — selector flagging row 0; used to gate the
//!     cross-row monotonicity exclusion on the head of the queue.
//!
//! ## Algebraic constraints (row-local)
//!
//!   0. `is_real_binary`              — `IS_REAL · (IS_REAL − 1) = 0`
//!   1. `is_first_binary`             — `IS_FIRST · (IS_FIRST − 1) = 0`
//!   2. `is_first_implies_real`       — `IS_FIRST · (1 − IS_REAL) = 0`
//!   3. `balance_update`              — `IS_REAL · (PRE − AMOUNT − POST) = 0`
//!   4. `index_le_decomp`             — `IS_REAL · (INDEX − Σ b_i · 256^i) = 0`
//!   5. `validator_index_le_decomp`   — `IS_REAL · (VI − Σ b_i · 256^i) = 0`
//!   6. `amount_le_decomp`            — `IS_REAL · (AMOUNT − Σ b_i · 256^i) = 0`
//!   7. `balance_pre_le_decomp`       — `IS_REAL · (PRE − Σ b_i · 256^i) = 0`
//!
//! 8-bit range checks cover every byte column (`ADDRESS`, all four LE
//! byte decompositions).
//!
//! ## Shifted constraint
//!
//!   * `index_monotonic` —
//!     `IS_REAL(X) · IS_REAL(ω·X) · (1 − IS_FIRST(ω·X)) ·
//!      (INDEX(ω·X) − INDEX(X) − 1) = 0`.
//!     Withdrawals in a payload share a contiguous index range starting
//!     at the beacon-state `next_withdrawal_index`; this constraint pins
//!     the in-payload row-to-row increment without forcing a specific
//!     starting value.
//!
//! ## Cross-AIR LogUp descriptors
//!
//!   * [`make_withdrawal_queue_to_root_descriptor`] — binds
//!     `(INDEX, VALIDATOR_INDEX, ADDRESS, AMOUNT)` to the same fields on
//!     [`crate::withdrawal_root_air`], anchoring each queue row to a
//!     row in the per-payload withdrawals-root AIR (and transitively to
//!     the block header's `withdrawals_root`).
//!   * [`make_withdrawal_queue_to_credential_descriptor`] — binds
//!     `(VALIDATOR_INDEX, ADDRESS)` to
//!     [`crate::withdrawal_credential_air`]'s
//!     `(VALIDATOR_INDEX, EXEC_ADDR)`, anchoring the recipient address
//!     to the beacon-side `withdrawal_credentials[12..32]` for the
//!     claimed validator.
//!   * [`make_withdrawal_queue_to_balance_pre_descriptor`] — binds
//!     `(VALIDATOR_INDEX, VALIDATOR_BALANCE_PRE)` to
//!     [`crate::validator_balances_air`]'s `(VALIDATOR_INDEX,
//!     CURRENT_BALANCE)`, anchoring the pre-state balance.
//!   * [`make_withdrawal_queue_to_balance_post_descriptor`] — same shape
//!     but for the post-state balance.
//!
//! Together with the in-AIR `balance_update` constraint and the two
//! balance-side cross-AIR LogUps, the AIR algebraically witnesses that
//! the queue's `amount` debits the validator's beacon balance.
//!
//! ## What this AIR does NOT prove (deferred)
//!
//!   * Two distinct `validator_balances_air` traces (one for pre, one
//!     for post). Composing them into a single root-update gadget is
//!     a follow-up.
//!   * The starting `INDEX` value (= beacon state's
//!     `next_withdrawal_index`). The host supplies it.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Maximum withdrawals per Ethereum block (Shapella spec
/// `MAX_WITHDRAWALS_PER_PAYLOAD`).
pub const MAX_WITHDRAWALS: usize = 16;

/// Bytes in an execution address.
pub const ADDRESS_LEN: usize = 20;

/// Bytes in a u64 LE decomposition.
pub const U64_BYTES: usize = 8;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_INDEX: usize = 0;
pub const COL_VALIDATOR_INDEX: usize = COL_INDEX + 1;
pub const COL_AMOUNT: usize = COL_VALIDATOR_INDEX + 1;
pub const COL_VALIDATOR_BALANCE_PRE: usize = COL_AMOUNT + 1;
pub const COL_VALIDATOR_BALANCE_POST: usize = COL_VALIDATOR_BALANCE_PRE + 1;
pub const COL_ADDRESS_OFFSET: usize = COL_VALIDATOR_BALANCE_POST + 1;
pub const COL_INDEX_BYTE_OFFSET: usize = COL_ADDRESS_OFFSET + ADDRESS_LEN;
pub const COL_VI_BYTE_OFFSET: usize = COL_INDEX_BYTE_OFFSET + U64_BYTES;
pub const COL_AMOUNT_BYTE_OFFSET: usize = COL_VI_BYTE_OFFSET + U64_BYTES;
pub const COL_BALANCE_PRE_BYTE_OFFSET: usize = COL_AMOUNT_BYTE_OFFSET + U64_BYTES;
pub const COL_IS_REAL: usize = COL_BALANCE_PRE_BYTE_OFFSET + U64_BYTES;
pub const COL_IS_FIRST: usize = COL_IS_REAL + 1;

pub const NUM_COLUMNS: usize = COL_IS_FIRST + 1;

/// 8 row-local bodies (see module docs).
pub const NUM_ROW_CONSTRAINTS: usize = 8;

/// 1 shifted body (`index_monotonic`).
pub const NUM_SHIFTED: usize = 1;

// ─── Witness types ────────────────────────────────────────────────────

/// One per-withdrawal row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WithdrawalQueueRow {
    pub index: u64,
    pub validator_index: u64,
    pub address: [u8; ADDRESS_LEN],
    pub amount: u64,
    pub validator_balance_pre: u64,
    pub validator_balance_post: u64,
    pub is_first: bool,
}

#[derive(Clone, Debug, Default)]
pub struct WithdrawalQueueWitness {
    pub rows: Vec<WithdrawalQueueRow>,
}

impl WithdrawalQueueWitness {
    pub fn from_rows(rows: Vec<WithdrawalQueueRow>) -> Self {
        Self { rows }
    }
}

/// Host-side helper.
///
/// Builds a witness from a list of `(index, validator_index, address,
/// amount)` tuples and the per-row pre-withdrawal validator balances.
///
/// Panics if `withdrawals.len()` exceeds [`MAX_WITHDRAWALS`], the two
/// slice lengths disagree, or any `pre_balances[i] < withdrawals[i].3`
/// (would underflow the post balance).
pub fn from_withdrawals(
    withdrawals: &[(u64, u64, [u8; ADDRESS_LEN], u64)],
    pre_balances: &[u64],
) -> WithdrawalQueueWitness {
    assert!(
        withdrawals.len() <= MAX_WITHDRAWALS,
        "withdrawal_queue_air: batch size {} exceeds MAX_WITHDRAWALS={}",
        withdrawals.len(),
        MAX_WITHDRAWALS,
    );
    assert_eq!(
        withdrawals.len(),
        pre_balances.len(),
        "withdrawal_queue_air: withdrawals.len() ({}) != pre_balances.len() ({})",
        withdrawals.len(),
        pre_balances.len(),
    );

    let mut rows = Vec::with_capacity(withdrawals.len());
    for (i, ((idx, vi, addr, amount), pre)) in
        withdrawals.iter().zip(pre_balances.iter()).enumerate()
    {
        assert!(
            *pre >= *amount,
            "withdrawal_queue_air: pre balance {} < amount {} at row {}",
            pre,
            amount,
            i,
        );
        rows.push(WithdrawalQueueRow {
            index: *idx,
            validator_index: *vi,
            address: *addr,
            amount: *amount,
            validator_balance_pre: *pre,
            validator_balance_post: pre - amount,
            is_first: i == 0,
        });
    }
    WithdrawalQueueWitness { rows }
}

// ─── Trace builder ────────────────────────────────────────────────────

fn write_address(
    columns: &mut [Vec<Scalar>],
    row: usize,
    addr: &[u8; ADDRESS_LEN],
    curve: CurveType,
) {
    for (b, &byte) in addr.iter().enumerate() {
        columns[COL_ADDRESS_OFFSET + b][row] = Scalar::from_u64(byte as u64, curve);
    }
}

fn write_u64_le(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    row: usize,
    value: u64,
    curve: CurveType,
) {
    let bytes = value.to_le_bytes();
    for (b, &byte) in bytes.iter().enumerate() {
        columns[offset + b][row] = Scalar::from_u64(byte as u64, curve);
    }
}

pub fn build_trace_polynomials(
    witness: &WithdrawalQueueWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let effective_rows = num_rows.max(1);
    let padded = crate::trace::nearest_power_of_two(effective_rows);
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_INDEX][i] = Scalar::from_u64(row.index, curve);
        columns[COL_VALIDATOR_INDEX][i] = Scalar::from_u64(row.validator_index, curve);
        columns[COL_AMOUNT][i] = Scalar::from_u64(row.amount, curve);
        columns[COL_VALIDATOR_BALANCE_PRE][i] =
            Scalar::from_u64(row.validator_balance_pre, curve);
        columns[COL_VALIDATOR_BALANCE_POST][i] =
            Scalar::from_u64(row.validator_balance_post, curve);
        write_address(&mut columns, i, &row.address, curve);
        write_u64_le(&mut columns, COL_INDEX_BYTE_OFFSET, i, row.index, curve);
        write_u64_le(&mut columns, COL_VI_BYTE_OFFSET, i, row.validator_index, curve);
        write_u64_le(&mut columns, COL_AMOUNT_BYTE_OFFSET, i, row.amount, curve);
        write_u64_le(
            &mut columns,
            COL_BALANCE_PRE_BYTE_OFFSET,
            i,
            row.validator_balance_pre,
            curve,
        );
        columns[COL_IS_REAL][i] = one.clone();
        if row.is_first {
            columns[COL_IS_FIRST][i] = one.clone();
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

pub struct WithdrawalQueueConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl WithdrawalQueueConstraintSystem {
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

/// Reconstruct a u64-shaped value from 8 LE byte cells `Σ b_i · 256^i`.
fn le_byte_reconstruct(byte_cells: &[Scalar], curve: CurveType) -> Scalar {
    let mut acc = Scalar::zero(curve);
    let mut p = Scalar::one(curve);
    let base = Scalar::from_u64(256, curve);
    for c in byte_cells {
        acc = acc.add(&c.mul(&p));
        p = p.mul(&base);
    }
    acc
}

fn le_byte_reconstruct_poly(
    col_coeffs: &[Vec<Scalar>],
    offset: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    let mut p = Scalar::one(curve);
    let base = Scalar::from_u64(256, curve);
    for k in 0..U64_BYTES {
        let scaled = poly_scalar_mul(&col_coeffs[offset + k], &p);
        acc = poly_add(&acc, &scaled, curve);
        p = p.mul(&base);
    }
    acc
}

impl VmConstraintSystem for WithdrawalQueueConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_first_binary".into(),
            "is_first_implies_real".into(),
            "balance_update".into(),
            "index_le_decomp".into(),
            "validator_index_le_decomp".into(),
            "amount_le_decomp".into(),
            "balance_pre_le_decomp".into(),
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
            // 3: balance update: is_real · (pre − amount − post)
            let pre = &columns[COL_VALIDATOR_BALANCE_PRE][row];
            let amount = &columns[COL_AMOUNT][row];
            let post = &columns[COL_VALIDATOR_BALANCE_POST][row];
            let diff = pre.sub(amount).sub(post);
            bodies[3][row] = is_real.mul(&diff);

            // 4..7: LE byte decomposition for index / val_idx / amount /
            // balance_pre — all gated by is_real.
            let decomps: [(usize, usize); 4] = [
                (COL_INDEX, COL_INDEX_BYTE_OFFSET),
                (COL_VALIDATOR_INDEX, COL_VI_BYTE_OFFSET),
                (COL_AMOUNT, COL_AMOUNT_BYTE_OFFSET),
                (COL_VALIDATOR_BALANCE_PRE, COL_BALANCE_PRE_BYTE_OFFSET),
            ];
            for (k, (val_col, byte_off)) in decomps.iter().enumerate() {
                let byte_cells: Vec<Scalar> = (0..U64_BYTES)
                    .map(|b| columns[*byte_off + b][row].clone())
                    .collect();
                let recon = le_byte_reconstruct(&byte_cells, curve);
                let val = &columns[*val_col][row];
                bodies[4 + k][row] = is_real.mul(&val.sub(&recon));
            }
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

        let pre = &col_evals[COL_VALIDATOR_BALANCE_PRE];
        let amount = &col_evals[COL_AMOUNT];
        let post = &col_evals[COL_VALIDATOR_BALANCE_POST];
        let bal_diff = pre.sub(amount).sub(post);

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(is_first.mul(&is_first.sub(&one)));
        bodies.push(is_first.mul(&one.sub(is_real)));
        bodies.push(is_real.mul(&bal_diff));

        let decomps: [(usize, usize); 4] = [
            (COL_INDEX, COL_INDEX_BYTE_OFFSET),
            (COL_VALIDATOR_INDEX, COL_VI_BYTE_OFFSET),
            (COL_AMOUNT, COL_AMOUNT_BYTE_OFFSET),
            (COL_VALIDATOR_BALANCE_PRE, COL_BALANCE_PRE_BYTE_OFFSET),
        ];
        for (val_col, byte_off) in decomps {
            let byte_cells: Vec<Scalar> = (0..U64_BYTES)
                .map(|b| col_evals[byte_off + b].clone())
                .collect();
            let recon = le_byte_reconstruct(&byte_cells, curve);
            bodies.push(is_real.mul(&col_evals[val_col].sub(&recon)));
        }

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

        let pre = &col_coeffs[COL_VALIDATOR_BALANCE_PRE];
        let amount = &col_coeffs[COL_AMOUNT];
        let post = &col_coeffs[COL_VALIDATOR_BALANCE_POST];
        let pre_minus_amount = poly_sub(pre, amount, curve);
        let bal_diff = poly_sub(&pre_minus_amount, post, curve);
        let balance_update = poly_mul(is_real, &bal_diff, curve);

        let mut bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_first_binary,
            is_first_implies_real,
            balance_update,
        ];

        let decomps: [(usize, usize); 4] = [
            (COL_INDEX, COL_INDEX_BYTE_OFFSET),
            (COL_VALIDATOR_INDEX, COL_VI_BYTE_OFFSET),
            (COL_AMOUNT, COL_AMOUNT_BYTE_OFFSET),
            (COL_VALIDATOR_BALANCE_PRE, COL_BALANCE_PRE_BYTE_OFFSET),
        ];
        for (val_col, byte_off) in decomps {
            let recon = le_byte_reconstruct_poly(col_coeffs, byte_off, curve);
            let diff = poly_sub(&col_coeffs[val_col], &recon, curve);
            let body = poly_mul(is_real, &diff, curve);
            bodies.push(body);
        }

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

    fn shifted_column_indices(&self) -> Vec<usize> {
        vec![COL_INDEX, COL_IS_REAL, COL_IS_FIRST]
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
        if shifted_evals.len() < 3 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let index_next = &shifted_evals[0];
        let is_real_next = &shifted_evals[1];
        let is_first_next = &shifted_evals[2];

        let is_real_curr = &col_evals_at_z[COL_IS_REAL];
        let index_curr = &col_evals_at_z[COL_INDEX];

        // body: is_real · is_real_next · (1 − is_first_next) ·
        //       (index_next − index_curr − 1)
        let gate = is_real_curr
            .mul(is_real_next)
            .mul(&one.sub(is_first_next));
        let inc = index_next.sub(index_curr).sub(&one);
        let body = gate.mul(&inc);

        let exclusion = z.sub(omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        ap.mul(&body).mul(&exclusion)
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
        let one_poly = vec![Scalar::one(curve)];

        let index = &col_coeffs[COL_INDEX];
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_first = &col_coeffs[COL_IS_FIRST];

        let index_shift = poly_shift(index, omega);
        let is_real_shift = poly_shift(is_real, omega);
        let is_first_shift = poly_shift(is_first, omega);

        let one_minus_first_next = poly_sub(&one_poly, &is_first_shift, curve);
        let gate1 = poly_mul(is_real, &is_real_shift, curve);
        let gate = poly_mul(&gate1, &one_minus_first_next, curve);

        let inc_pre = poly_sub(&index_shift, index, curve);
        let inc = poly_sub(&inc_pre, &one_poly, curve);

        let body = poly_mul(&gate, &inc, curve);

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let excluded = poly_mul_linear(&body, &omega_n_minus_1);
        poly_scalar_mul(&excluded, &ap)
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        let byte_groups: [(&str, usize, usize); 5] = [
            ("address", COL_ADDRESS_OFFSET, ADDRESS_LEN),
            ("index_byte", COL_INDEX_BYTE_OFFSET, U64_BYTES),
            ("validator_index_byte", COL_VI_BYTE_OFFSET, U64_BYTES),
            ("amount_byte", COL_AMOUNT_BYTE_OFFSET, U64_BYTES),
            ("balance_pre_byte", COL_BALANCE_PRE_BYTE_OFFSET, U64_BYTES),
        ];
        for (name, offset, len) in byte_groups {
            for k in 0..len {
                declarations.push((
                    LookupDeclaration {
                        label: format!("withdrawal_queue_{}_{}_8bit", name, k),
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

/// Bind `(INDEX, VALIDATOR_INDEX, ADDRESS[0..20], AMOUNT)` on this AIR
/// to the same fields on [`crate::withdrawal_root_air`]. Anchors each
/// queue row to a row of the per-payload withdrawals-root AIR, which
/// in turn binds to the block header's `withdrawals_root`.
///
/// Tuple width: 1 + 1 + 20 + 1 = 23 columns.
pub fn make_withdrawal_queue_to_root_descriptor(
    queue_layer_index: usize,
    withdrawal_root_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::withdrawal_root_air as wr;
    let mut a_columns: Vec<usize> = Vec::with_capacity(3 + ADDRESS_LEN);
    a_columns.push(COL_INDEX);
    a_columns.push(COL_VALIDATOR_INDEX);
    for b in 0..ADDRESS_LEN {
        a_columns.push(COL_ADDRESS_OFFSET + b);
    }
    a_columns.push(COL_AMOUNT);

    let mut b_columns: Vec<usize> = Vec::with_capacity(3 + ADDRESS_LEN);
    b_columns.push(wr::COL_INDEX);
    b_columns.push(wr::COL_VALIDATOR_INDEX);
    for b in 0..ADDRESS_LEN {
        b_columns.push(wr::COL_ADDRESS_OFFSET + b);
    }
    b_columns.push(wr::COL_AMOUNT);

    CrossAirLogUpDescriptor {
        label: "withdrawal_queue_to_root_v1".into(),
        a_layer_index: queue_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: withdrawal_root_layer_index,
        b_columns,
        b_selector_column: Some(wr::COL_IS_REAL),
    }
}

/// Bind `(VALIDATOR_INDEX, ADDRESS[0..20])` to
/// [`crate::withdrawal_credential_air`]'s `(VALIDATOR_INDEX,
/// EXEC_ADDR[0..20])`. Pins the recipient address to the beacon-side
/// `withdrawal_credentials[12..32]` of the claimed validator.
///
/// Tuple width: 1 + 20 = 21 columns.
pub fn make_withdrawal_queue_to_credential_descriptor(
    queue_layer_index: usize,
    credential_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::withdrawal_credential_air as wc;
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + ADDRESS_LEN);
    a_columns.push(COL_VALIDATOR_INDEX);
    for b in 0..ADDRESS_LEN {
        a_columns.push(COL_ADDRESS_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + ADDRESS_LEN);
    b_columns.push(wc::COL_VALIDATOR_INDEX);
    for b in 0..ADDRESS_LEN {
        b_columns.push(wc::COL_EXEC_ADDR_BYTE_OFFSET + b);
    }

    CrossAirLogUpDescriptor {
        label: "withdrawal_queue_to_credential_v1".into(),
        a_layer_index: queue_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: credential_layer_index,
        b_columns,
        b_selector_column: Some(wc::COL_IS_EXEC),
    }
}

/// Bind `(VALIDATOR_INDEX, VALIDATOR_BALANCE_PRE)` to
/// [`crate::validator_balances_air`]'s `(VALIDATOR_INDEX,
/// CURRENT_BALANCE)`. Pins the pre-state balance read by this
/// withdrawal.
///
/// Tuple width: 2 columns.
pub fn make_withdrawal_queue_to_balance_pre_descriptor(
    queue_layer_index: usize,
    validator_balances_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_balances_air as vb;
    let a_columns = vec![COL_VALIDATOR_INDEX, COL_VALIDATOR_BALANCE_PRE];
    let b_columns = vec![vb::COL_VALIDATOR_INDEX, vb::COL_CURRENT_BALANCE];
    CrossAirLogUpDescriptor {
        label: "withdrawal_queue_to_balance_pre_v1".into(),
        a_layer_index: queue_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_balances_layer_index,
        b_columns,
        b_selector_column: Some(vb::COL_IS_LEAF),
    }
}

/// Bind `(VALIDATOR_INDEX, VALIDATOR_BALANCE_POST)` to
/// [`crate::validator_balances_air`]'s `(VALIDATOR_INDEX,
/// CURRENT_BALANCE)`. Pins the post-state balance produced by this
/// withdrawal — typically threaded against a *second*
/// validator_balances trace built from the post-state.
///
/// Tuple width: 2 columns.
pub fn make_withdrawal_queue_to_balance_post_descriptor(
    queue_layer_index: usize,
    validator_balances_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_balances_air as vb;
    let a_columns = vec![COL_VALIDATOR_INDEX, COL_VALIDATOR_BALANCE_POST];
    let b_columns = vec![vb::COL_VALIDATOR_INDEX, vb::COL_CURRENT_BALANCE];
    CrossAirLogUpDescriptor {
        label: "withdrawal_queue_to_balance_post_v1".into(),
        a_layer_index: queue_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_balances_layer_index,
        b_columns,
        b_selector_column: Some(vb::COL_IS_LEAF),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_addr(seed: u8) -> [u8; ADDRESS_LEN] {
        [seed; ADDRESS_LEN]
    }

    /// Column layout pinned: spot-check named offsets and the total
    /// column count.
    #[test]
    fn withdrawal_queue_air_column_layout_pinned() {
        assert_eq!(COL_INDEX, 0);
        assert_eq!(COL_VALIDATOR_INDEX, 1);
        assert_eq!(COL_AMOUNT, 2);
        assert_eq!(COL_VALIDATOR_BALANCE_PRE, 3);
        assert_eq!(COL_VALIDATOR_BALANCE_POST, 4);
        assert_eq!(COL_ADDRESS_OFFSET, 5);
        assert_eq!(COL_INDEX_BYTE_OFFSET, 5 + ADDRESS_LEN);
        assert_eq!(COL_VI_BYTE_OFFSET, 5 + ADDRESS_LEN + U64_BYTES);
        assert_eq!(
            COL_AMOUNT_BYTE_OFFSET,
            5 + ADDRESS_LEN + 2 * U64_BYTES,
        );
        assert_eq!(
            COL_BALANCE_PRE_BYTE_OFFSET,
            5 + ADDRESS_LEN + 3 * U64_BYTES,
        );
        assert_eq!(
            COL_IS_REAL,
            5 + ADDRESS_LEN + 4 * U64_BYTES,
        );
        assert_eq!(COL_IS_FIRST, COL_IS_REAL + 1);
        assert_eq!(NUM_COLUMNS, COL_IS_FIRST + 1);
        // Sanity: 5 + 20 + 32 + 2 = 59.
        assert_eq!(NUM_COLUMNS, 59);
        assert_eq!(NUM_ROW_CONSTRAINTS, 8);
        assert_eq!(NUM_SHIFTED, 1);
    }

    /// Single withdrawal: row 0 is real, IS_FIRST = 1, balance update
    /// vanishes algebraically, and the byte-decomp constraints vanish.
    #[test]
    fn withdrawal_queue_air_single_withdrawal() {
        let w = from_withdrawals(
            &[(7, 100, sample_addr(0x11), 32_000_000_000)],
            &[64_000_000_000],
        );
        assert_eq!(w.rows.len(), 1);
        assert!(w.rows[0].is_first);
        assert_eq!(w.rows[0].validator_balance_pre, 64_000_000_000);
        assert_eq!(w.rows[0].validator_balance_post, 32_000_000_000);

        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);

        let cs = WithdrawalQueueConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) at row {} must vanish",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }
    }

    /// 4 withdrawals: indices 100..104; every row real, only row 0 has
    /// IS_FIRST; balance update vanishes for each row.
    #[test]
    fn withdrawal_queue_air_four_withdrawals() {
        let inputs: Vec<(u64, u64, [u8; ADDRESS_LEN], u64)> = (0..4)
            .map(|i| {
                (
                    100 + i as u64,
                    1000 + i as u64,
                    sample_addr((i as u8).wrapping_mul(17) + 3),
                    10_000_000 + i as u64,
                )
            })
            .collect();
        let pre_balances: Vec<u64> =
            (0..4).map(|i| 50_000_000 + i as u64).collect();
        let w = from_withdrawals(&inputs, &pre_balances);
        assert_eq!(w.rows.len(), 4);
        assert!(w.rows[0].is_first);
        for i in 1..4 {
            assert!(!w.rows[i].is_first);
            assert_eq!(
                w.rows[i].validator_balance_post,
                pre_balances[i] - inputs[i].3,
            );
        }

        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cs = WithdrawalQueueConstraintSystem::new(trace.num_rows);
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

    /// Tampering: directly mutate AMOUNT on row 1 without updating
    /// VALIDATOR_BALANCE_POST → balance_update body must fire on that
    /// row. Also confirm an AMOUNT tamper breaks the
    /// `amount_le_decomp` body too.
    #[test]
    fn withdrawal_queue_air_tampered_amount_detected() {
        let inputs: Vec<(u64, u64, [u8; ADDRESS_LEN], u64)> = (0..3)
            .map(|i| (i as u64, 200 + i as u64, sample_addr(i as u8 + 1), 1_000))
            .collect();
        let pre = vec![10_000u64; 3];
        let w = from_withdrawals(&inputs, &pre);
        let curve = CurveType::Bls48581;
        let mut trace = build_trace_polynomials(&w, curve);

        // Tamper AMOUNT on row 1.
        trace.columns[COL_AMOUNT].evaluations[1] = Scalar::from_u64(2_000, curve);

        let cs = WithdrawalQueueConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);

        // body 3 (balance_update) must fire on row 1.
        assert!(
            !evals[3][1].is_zero(),
            "tampered AMOUNT must fire balance_update on row 1",
        );
        // body 6 (amount_le_decomp) must also fire on row 1.
        assert!(
            !evals[6][1].is_zero(),
            "tampered AMOUNT must fire amount_le_decomp on row 1",
        );
        // Untampered rows still vanish for both bodies.
        assert!(evals[3][0].is_zero());
        assert!(evals[3][2].is_zero());
        assert!(evals[6][0].is_zero());
        assert!(evals[6][2].is_zero());
    }

    /// Byte range: every byte cell in the address and the four LE
    /// decompositions must fit in `[0, 256)` on real rows; verified
    /// numerically.
    #[test]
    fn withdrawal_queue_air_byte_range_pinned() {
        let w = from_withdrawals(
            &[(0xdeadbeefcafebabeu64, 0x1234567890abcdefu64, sample_addr(0x42), 0xffffffffu64)],
            &[0xffffffff_ffffffffu64],
        );
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);

        let byte_offsets: [(usize, usize); 5] = [
            (COL_ADDRESS_OFFSET, ADDRESS_LEN),
            (COL_INDEX_BYTE_OFFSET, U64_BYTES),
            (COL_VI_BYTE_OFFSET, U64_BYTES),
            (COL_AMOUNT_BYTE_OFFSET, U64_BYTES),
            (COL_BALANCE_PRE_BYTE_OFFSET, U64_BYTES),
        ];
        for (offset, len) in byte_offsets {
            for k in 0..len {
                let v = trace.columns[offset + k].evaluations[0].to_u64();
                assert!(v < 256, "byte cell at col {} row 0 must fit in 8 bits (got {})", offset + k, v);
            }
        }

        // Spot-check the LE byte decomp for INDEX: reconstructing
        // Σ b_i · 256^i must equal the canonical u64 value.
        let mut recon: u128 = 0;
        for k in 0..U64_BYTES {
            recon |=
                (trace.columns[COL_INDEX_BYTE_OFFSET + k].evaluations[0].to_u64() as u128)
                    << (8 * k);
        }
        assert_eq!(recon as u64, 0xdeadbeefcafebabeu64);

        // Confirm the AIR exposes the 8-bit declarations.
        let cs = WithdrawalQueueConstraintSystem::new(trace.num_rows);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        assert_eq!(
            reqs.declarations.len(),
            ADDRESS_LEN + 4 * U64_BYTES,
        );
        for (decl, _) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
        }
    }

    /// Cross-AIR LogUp descriptors are well-formed: tuple widths, A/B
    /// layer indices distinct, labels, selectors.
    #[test]
    fn withdrawal_queue_air_descriptors_well_formed() {
        // queue → withdrawal_root (23-col tuple).
        let d_root = make_withdrawal_queue_to_root_descriptor(0, 1);
        assert_eq!(d_root.label, "withdrawal_queue_to_root_v1");
        assert_eq!(d_root.a_columns.len(), 3 + ADDRESS_LEN);
        assert_eq!(d_root.b_columns.len(), 3 + ADDRESS_LEN);
        assert_eq!(d_root.a_columns[0], COL_INDEX);
        assert_eq!(d_root.a_columns[1], COL_VALIDATOR_INDEX);
        for b in 0..ADDRESS_LEN {
            assert_eq!(d_root.a_columns[2 + b], COL_ADDRESS_OFFSET + b);
            assert_eq!(
                d_root.b_columns[2 + b],
                crate::withdrawal_root_air::COL_ADDRESS_OFFSET + b,
            );
        }
        assert_eq!(d_root.a_columns[2 + ADDRESS_LEN], COL_AMOUNT);
        assert_eq!(d_root.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_root.b_selector_column,
            Some(crate::withdrawal_root_air::COL_IS_REAL),
        );
        assert_eq!(d_root.a_layer_index, 0);
        assert_eq!(d_root.b_layer_index, 1);

        // queue → withdrawal_credential (21-col tuple).
        let d_cred = make_withdrawal_queue_to_credential_descriptor(0, 2);
        assert_eq!(d_cred.label, "withdrawal_queue_to_credential_v1");
        assert_eq!(d_cred.a_columns.len(), 1 + ADDRESS_LEN);
        assert_eq!(d_cred.b_columns.len(), 1 + ADDRESS_LEN);
        assert_eq!(d_cred.a_columns[0], COL_VALIDATOR_INDEX);
        assert_eq!(
            d_cred.b_columns[0],
            crate::withdrawal_credential_air::COL_VALIDATOR_INDEX,
        );
        for b in 0..ADDRESS_LEN {
            assert_eq!(d_cred.a_columns[1 + b], COL_ADDRESS_OFFSET + b);
            assert_eq!(
                d_cred.b_columns[1 + b],
                crate::withdrawal_credential_air::COL_EXEC_ADDR_BYTE_OFFSET + b,
            );
        }
        assert_eq!(d_cred.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_cred.b_selector_column,
            Some(crate::withdrawal_credential_air::COL_IS_EXEC),
        );

        // queue → validator_balances pre (2-col tuple).
        let d_pre = make_withdrawal_queue_to_balance_pre_descriptor(0, 3);
        assert_eq!(d_pre.label, "withdrawal_queue_to_balance_pre_v1");
        assert_eq!(
            d_pre.a_columns,
            vec![COL_VALIDATOR_INDEX, COL_VALIDATOR_BALANCE_PRE],
        );
        assert_eq!(
            d_pre.b_columns,
            vec![
                crate::validator_balances_air::COL_VALIDATOR_INDEX,
                crate::validator_balances_air::COL_CURRENT_BALANCE,
            ],
        );
        assert_eq!(d_pre.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_pre.b_selector_column,
            Some(crate::validator_balances_air::COL_IS_LEAF),
        );

        // queue → validator_balances post (2-col tuple).
        let d_post = make_withdrawal_queue_to_balance_post_descriptor(0, 4);
        assert_eq!(d_post.label, "withdrawal_queue_to_balance_post_v1");
        assert_eq!(
            d_post.a_columns,
            vec![COL_VALIDATOR_INDEX, COL_VALIDATOR_BALANCE_POST],
        );
        assert_eq!(
            d_post.b_columns,
            vec![
                crate::validator_balances_air::COL_VALIDATOR_INDEX,
                crate::validator_balances_air::COL_CURRENT_BALANCE,
            ],
        );
    }
}
