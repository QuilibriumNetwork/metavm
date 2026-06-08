//! Receipt status + cumulative-gas AIR.
//!
//! Proves, for a block's transaction-receipt list, that
//!
//!   * each receipt's `status` is a single bit (0 = failure, 1 = success),
//!   * the `cumulative_gas_used` field is monotonically non-decreasing
//!     across receipts:
//!         `cumulative_gas_used[i] = cumulative_gas_used[i-1] + gas_used[i]`
//!     with `cumulative_gas_used[-1] = 0`.
//!
//! Each row commits one receipt's `(status, gas_used, prev_cumulative,
//! cumulative)` tuple (plus per-byte LE decompositions for the two u64
//! fields). The first-row flag `IS_FIRST` pins
//! `prev_cumulative_gas_used = 0` on row 0; the cross-row shifted body
//! threads `cumulative_gas_used[i] → prev_cumulative_gas_used[i+1]` for
//! every subsequent receipt.
//!
//! The actual `gas_used` correctness against EVM execution is bound via
//! a future cross-AIR LogUp against the EVM gas-tracking AIR (in the
//! EVM crate); this AIR only proves the algebra of cumulative_gas_used
//! and that the receipt's encoded `(status, cumulative_gas_used)` match
//! the receipt-RLP AIR.
//!
//! ## Algebraic constraints (row-local, 9 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`.
//! 1. `is_first_binary` — `IS_FIRST · (IS_FIRST − 1) = 0`.
//! 2. `status_binary` — `IS_REAL · STATUS · (STATUS − 1) = 0`.
//! 3. `is_first_implies_prev_zero` — `IS_FIRST · PREV_CUMULATIVE = 0`.
//! 4. `cumulative_chain` —
//!    `IS_REAL · (CUMULATIVE − PREV_CUMULATIVE − GAS_USED) = 0`.
//! 5. `gas_used_le_decomp` —
//!    `GAS_USED − Σ_b GAS_USED_BYTE[b] · 2^(8b) = 0`.
//! 6. `cumulative_le_decomp` —
//!    `CUMULATIVE − Σ_b CUMUL_BYTE[b] · 2^(8b) = 0`.
//! 7. `is_first_implies_real` — `IS_FIRST · (1 − IS_REAL) = 0`.
//! 8. `is_first_zero_when_padding` — `(1 − IS_REAL) · IS_FIRST = 0`
//!    (mirrors 7; redundant body kept for symmetry / clearer audit).
//!
//! ## Shifted constraint (1 body, cross-row)
//!
//! 0. `prev_chains_from_cumulative` —
//!    `IS_REAL(X) · IS_REAL(ω·X) · (PREV_CUMULATIVE(ω·X) − CUMULATIVE(X)) = 0`.
//!
//! ## What this AIR does NOT prove (deferred)
//!
//!   * `gas_used` matches the EVM execution trace — bound via a future
//!     cross-AIR LogUp descriptor against the EVM gas-tracking AIR.
//!   * `status` matches the EVM execution-success signal — also future
//!     EVM cross-AIR LogUp.
//!   * `cumulative_gas_used` ≤ block `gas_limit` — bound via a future
//!     `block_header_air` descriptor.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::receipt::Receipt;
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_TX_INDEX: usize = 0;
pub const COL_STATUS: usize = 1;
pub const COL_GAS_USED: usize = 2;
pub const COL_CUMULATIVE_GAS: usize = 3;
pub const COL_PREV_CUMULATIVE_GAS: usize = 4;
pub const COL_GAS_USED_BYTE_OFFSET: usize = 5;                            // 5..13
pub const COL_CUMUL_BYTE_OFFSET: usize = COL_GAS_USED_BYTE_OFFSET + U64_BYTES; // 13..21
pub const COL_IS_REAL: usize = COL_CUMUL_BYTE_OFFSET + U64_BYTES;         // 21
pub const COL_IS_FIRST: usize = COL_IS_REAL + 1;                          // 22
pub const NUM_COLUMNS: usize = COL_IS_FIRST + 1;                          // 23

pub const NUM_ROW_CONSTRAINTS: usize = 9;
pub const NUM_SHIFTED: usize = 1;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReceiptStatusRow {
    pub tx_index: u64,
    pub status: u8,
    pub gas_used: u64,
    pub cumulative_gas_used: u64,
    pub prev_cumulative_gas_used: u64,
    pub is_first: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ReceiptStatusWitness {
    pub rows: Vec<ReceiptStatusRow>,
}

impl ReceiptStatusWitness {
    /// Build a witness from a block's `[Receipt]` list. The first
    /// receipt's `prev_cumulative_gas_used` is `0`; every subsequent
    /// row reads `prev_cumulative_gas_used` from the previous row's
    /// `cumulative_gas_used`.
    pub fn from_receipts(receipts: &[Receipt]) -> Self {
        let mut rows = Vec::with_capacity(receipts.len());
        let mut prev: u64 = 0;
        for (i, r) in receipts.iter().enumerate() {
            let cumul = r.cumulative_gas_used;
            // Saturating_sub guards against malformed input; honest
            // receipt lists have cumul >= prev.
            let gas_used = cumul.saturating_sub(prev);
            rows.push(ReceiptStatusRow {
                tx_index: i as u64,
                status: r.status,
                gas_used,
                cumulative_gas_used: cumul,
                prev_cumulative_gas_used: prev,
                is_first: i == 0,
            });
            prev = cumul;
        }
        Self { rows }
    }

    pub fn from_rows(rows: Vec<ReceiptStatusRow>) -> Self {
        Self { rows }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

fn eval_le_decomp(
    target: &Scalar,
    byte_off: usize,
    col_evals: &[Scalar],
) -> Scalar {
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
    witness: &ReceiptStatusWitness,
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
        columns[COL_STATUS][i] = Scalar::from_u64(row.status as u64, curve);
        columns[COL_GAS_USED][i] = Scalar::from_u64(row.gas_used, curve);
        columns[COL_CUMULATIVE_GAS][i] =
            Scalar::from_u64(row.cumulative_gas_used, curve);
        columns[COL_PREV_CUMULATIVE_GAS][i] =
            Scalar::from_u64(row.prev_cumulative_gas_used, curve);

        let gas_bytes = row.gas_used.to_le_bytes();
        let cumul_bytes = row.cumulative_gas_used.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_GAS_USED_BYTE_OFFSET + b][i] =
                Scalar::from_u64(gas_bytes[b] as u64, curve);
            columns[COL_CUMUL_BYTE_OFFSET + b][i] =
                Scalar::from_u64(cumul_bytes[b] as u64, curve);
        }

        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_FIRST][i] = if row.is_first { one.clone() } else { zero.clone() };
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

pub struct ReceiptStatusConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ReceiptStatusConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for ReceiptStatusConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_first_binary".into(),
            "status_binary".into(),
            "is_first_implies_prev_zero".into(),
            "cumulative_chain".into(),
            "gas_used_le_decomp".into(),
            "cumulative_le_decomp".into(),
            "is_first_implies_real".into(),
            "is_first_zero_when_padding".into(),
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
            let is_first = &row_evals[COL_IS_FIRST];
            let status = &row_evals[COL_STATUS];
            let gas_used = &row_evals[COL_GAS_USED];
            let cumul = &row_evals[COL_CUMULATIVE_GAS];
            let prev = &row_evals[COL_PREV_CUMULATIVE_GAS];

            // 0: is_real_binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));

            // 1: is_first_binary.
            bodies[1][row] = is_first.mul(&is_first.sub(&one));

            // 2: status_binary (gated by IS_REAL so padding rows don't fire).
            bodies[2][row] = is_real.mul(&status.mul(&status.sub(&one)));

            // 3: is_first_implies_prev_zero.
            bodies[3][row] = is_first.mul(prev);

            // 4: cumulative_chain — cumul = prev + gas_used.
            let chain = cumul.sub(prev).sub(gas_used);
            bodies[4][row] = is_real.mul(&chain);

            // 5: gas_used_le_decomp.
            bodies[5][row] = eval_le_decomp(
                gas_used,
                COL_GAS_USED_BYTE_OFFSET,
                &row_evals,
            );

            // 6: cumulative_le_decomp.
            bodies[6][row] = eval_le_decomp(
                cumul,
                COL_CUMUL_BYTE_OFFSET,
                &row_evals,
            );

            // 7: is_first_implies_real (IS_FIRST=1 ⇒ IS_REAL=1).
            bodies[7][row] = is_first.mul(&one.sub(is_real));

            // 8: symmetry / redundancy of 7 (kept for audit clarity).
            bodies[8][row] = one.sub(is_real).mul(is_first);
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
        let status = &col_evals[COL_STATUS];
        let gas_used = &col_evals[COL_GAS_USED];
        let cumul = &col_evals[COL_CUMULATIVE_GAS];
        let prev = &col_evals[COL_PREV_CUMULATIVE_GAS];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_first.mul(&is_first.sub(&one)),
            is_real.mul(&status.mul(&status.sub(&one))),
            is_first.mul(prev),
            is_real.mul(&cumul.sub(prev).sub(gas_used)),
            eval_le_decomp(gas_used, COL_GAS_USED_BYTE_OFFSET, col_evals),
            eval_le_decomp(cumul, COL_CUMUL_BYTE_OFFSET, col_evals),
            is_first.mul(&one.sub(is_real)),
            one.sub(is_real).mul(is_first),
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
        let status = &col_coeffs[COL_STATUS];
        let gas_used = &col_coeffs[COL_GAS_USED];
        let cumul = &col_coeffs[COL_CUMULATIVE_GAS];
        let prev = &col_coeffs[COL_PREV_CUMULATIVE_GAS];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_first_m1 = poly_sub(is_first, &one_poly, curve);
        let is_first_binary = poly_mul(is_first, &is_first_m1, curve);

        let status_m1 = poly_sub(status, &one_poly, curve);
        let status_sq = poly_mul(status, &status_m1, curve);
        let status_binary = poly_mul(is_real, &status_sq, curve);

        let is_first_prev = poly_mul(is_first, prev, curve);

        let chain_inner = poly_sub(&poly_sub(cumul, prev, curve), gas_used, curve);
        let chain_body = poly_mul(is_real, &chain_inner, curve);

        let gas_decomp = build_le_decomp_poly(
            gas_used,
            COL_GAS_USED_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let cumul_decomp = build_le_decomp_poly(
            cumul,
            COL_CUMUL_BYTE_OFFSET,
            col_coeffs,
            curve,
        );

        let one_minus_is_real = poly_sub(&one_poly, is_real, curve);
        let is_first_implies_real = poly_mul(is_first, &one_minus_is_real, curve);
        let redundant_pad = poly_mul(&one_minus_is_real, is_first, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_first_binary,
            status_binary,
            is_first_prev,
            chain_body,
            gas_decomp,
            cumul_decomp,
            is_first_implies_real,
            redundant_pad,
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

        let ranges: [(usize, &str); 2] = [
            (COL_GAS_USED_BYTE_OFFSET, "gas_used_byte"),
            (COL_CUMUL_BYTE_OFFSET, "cumul_byte"),
        ];
        for (off, label) in ranges {
            for k in 0..U64_BYTES {
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

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // ω·z evaluations layout (total 2):
        //   [0] IS_REAL_NEXT (gating)
        //   [1] PREV_CUMULATIVE_NEXT (chain target)
        vec![COL_IS_REAL, COL_PREV_CUMULATIVE_GAS]
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
        if shifted_evals.len() != 2 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let is_real = &col_evals_at_z[COL_IS_REAL];
        let is_real_next = &shifted_evals[0];
        let prev_next = &shifted_evals[1];
        let cumul = &col_evals_at_z[COL_CUMULATIVE_GAS];

        // IS_REAL(X) · IS_REAL(ω·X) · (PREV_NEXT - CUMUL).
        let gating = is_real.mul(is_real_next);
        let body = gating.mul(&prev_next.sub(cumul));

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let total = ap.mul(&body);

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

        let is_real = &column_coeffs[COL_IS_REAL];
        let is_real_next = poly_shift(is_real, omega);
        let gating = poly_mul(is_real, &is_real_next, curve);

        let prev = &column_coeffs[COL_PREV_CUMULATIVE_GAS];
        let prev_next = poly_shift(prev, omega);
        let cumul = &column_coeffs[COL_CUMULATIVE_GAS];
        let diff = poly_sub(&prev_next, cumul, curve);
        let body = poly_mul(&gating, &diff, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut total = poly_scalar_mul(&body, &ap);

        // Multiply by (X - ω^{n-1}) to exclude wrap-around row.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) {
            omega_n_minus_1 = omega_n_minus_1.mul(omega);
        }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        // (X - ω^{n-1}) → coefficients [-ω^{n-1}, 1].
        let factor = vec![neg, Scalar::one(curve)];
        total = poly_mul(&total, &factor, curve);
        total
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(STATUS, CUMULATIVE_GAS)` of this AIR against the
/// receipt-RLP AIR's `(COL_STATUS, COL_CUMULATIVE_GAS)`. This pins the
/// status + cumulative_gas_used pair that gets RLP-encoded into the
/// receipt to the algebraically-checked values in this AIR.
pub fn make_receipt_status_to_receipt_rlp_descriptor(
    status_layer_index: usize,
    receipt_rlp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::receipt_rlp_air as rr;

    let a_columns: Vec<usize> = vec![COL_STATUS, COL_CUMULATIVE_GAS];
    let b_columns: Vec<usize> = vec![rr::COL_STATUS, rr::COL_CUMULATIVE_GAS];

    CrossAirLogUpDescriptor {
        label: "receipt_status_to_receipt_rlp_v1".into(),
        a_layer_index: status_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: receipt_rlp_layer_index,
        b_columns,
        b_selector_column: Some(rr::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipt::{Receipt, ReceiptType};

    fn make_receipt(status: u8, cumulative_gas_used: u64) -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status,
            cumulative_gas_used,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }
    }

    fn check_all_bodies_vanish(witness: &ReceiptStatusWitness) {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(witness, curve);
        let cs = ReceiptStatusConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {} (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    #[test]
    fn empty_witness_has_no_rows() {
        let receipts: Vec<Receipt> = Vec::new();
        let w = ReceiptStatusWitness::from_receipts(&receipts);
        assert_eq!(w.rows.len(), 0);
        // build_trace_polynomials must still succeed (padded to 1 row).
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.num_rows, 0);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        // Padding row constraints must vanish.
        let cs = ReceiptStatusConstraintSystem::new(0);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, 0);
        for body in &bodies {
            for v in body {
                assert!(v.is_zero(), "padding-only row must vanish all bodies");
            }
        }
    }

    #[test]
    fn single_receipt_constraints_vanish() {
        let receipts = vec![make_receipt(1, 21_000)];
        let w = ReceiptStatusWitness::from_receipts(&receipts);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].tx_index, 0);
        assert_eq!(w.rows[0].status, 1);
        assert_eq!(w.rows[0].gas_used, 21_000);
        assert_eq!(w.rows[0].cumulative_gas_used, 21_000);
        assert_eq!(w.rows[0].prev_cumulative_gas_used, 0);
        assert!(w.rows[0].is_first);
        check_all_bodies_vanish(&w);
    }

    #[test]
    fn multi_receipt_chain_constraints_vanish() {
        // 3 receipts: gas_used = 21000, 50000, 30000.
        // cumulative: 21000, 71000, 101000.
        let receipts = vec![
            make_receipt(1, 21_000),
            make_receipt(0, 71_000), // failed tx
            make_receipt(1, 101_000),
        ];
        let w = ReceiptStatusWitness::from_receipts(&receipts);
        assert_eq!(w.rows.len(), 3);

        assert_eq!(w.rows[0].gas_used, 21_000);
        assert_eq!(w.rows[1].gas_used, 50_000);
        assert_eq!(w.rows[2].gas_used, 30_000);

        assert_eq!(w.rows[0].prev_cumulative_gas_used, 0);
        assert_eq!(w.rows[1].prev_cumulative_gas_used, 21_000);
        assert_eq!(w.rows[2].prev_cumulative_gas_used, 71_000);

        assert!(w.rows[0].is_first);
        assert!(!w.rows[1].is_first);
        assert!(!w.rows[2].is_first);

        check_all_bodies_vanish(&w);

        // The cross-row shifted constraint also must vanish in honest
        // shifted polynomial form. We check the evaluate_shifted_at_point
        // contract on row 0→1.
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let cs = ReceiptStatusConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        // Row 0 evaluations:
        let row0: Vec<Scalar> = col_refs.iter().map(|c| c[0].clone()).collect();
        // ω·z corresponds to row 1.
        let shifted = vec![
            col_refs[COL_IS_REAL][1].clone(),
            col_refs[COL_PREV_CUMULATIVE_GAS][1].clone(),
        ];
        // PREV[1] = CUMUL[0] = 21000, so body must vanish.
        assert_eq!(shifted[1].to_u64(), 21_000u64);

        // Direct algebraic check without the boundary factor.
        let is_real = &row0[COL_IS_REAL];
        let is_real_next = &shifted[0];
        let prev_next = &shifted[1];
        let cumul = &row0[COL_CUMULATIVE_GAS];
        let gating = is_real.mul(is_real_next);
        let body = gating.mul(&prev_next.sub(cumul));
        assert!(body.is_zero(), "shifted body must vanish on honest chain");

        // Sanity: constraint labels report the shifted count via the
        // VmConstraintSystem trait.
        assert_eq!(cs.num_shifted_constraints(), NUM_SHIFTED);
        assert_eq!(cs.shifted_column_indices().len(), 2);
    }

    #[test]
    fn tampered_cumulative_fires_chain() {
        let receipts = vec![
            make_receipt(1, 21_000),
            make_receipt(1, 71_000),
        ];
        let w = ReceiptStatusWitness::from_receipts(&receipts);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper row 1's cumulative_gas_used (drop a chunk). The
        // cumulative_chain body must fire on row 1.
        let tampered = 60_000u64;
        cols[COL_CUMULATIVE_GAS][1] = Scalar::from_u64(tampered, curve);
        // Update the LE byte decomp so body 6 still vanishes (we want
        // body 4 specifically to fire).
        let bytes = tampered.to_le_bytes();
        for b in 0..U64_BYTES {
            cols[COL_CUMUL_BYTE_OFFSET + b][1] =
                Scalar::from_u64(bytes[b] as u64, curve);
        }
        let cs = ReceiptStatusConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[4][1].is_zero(),
            "cumulative_chain body should fire on tampered cumulative",
        );
        // Body 6 (cumul_le_decomp) should still vanish.
        assert!(
            bodies[6][1].is_zero(),
            "cumul_le_decomp body should still vanish (we updated the bytes)",
        );
    }

    #[test]
    fn tampered_first_row_prev_fires() {
        // Build an honest witness, then set PREV_CUMULATIVE on row 0
        // to a non-zero value. The `is_first_implies_prev_zero` body
        // must fire on row 0.
        let receipts = vec![make_receipt(1, 21_000)];
        let w = ReceiptStatusWitness::from_receipts(&receipts);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_PREV_CUMULATIVE_GAS][0] = Scalar::from_u64(5, curve);
        let cs = ReceiptStatusConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[3][0].is_zero(),
            "is_first_implies_prev_zero body should fire on row 0",
        );
        // The cumulative_chain body also fires now (cumul - prev - gas
        // = 21000 - 5 - 21000 ≠ 0).
        assert!(
            !bodies[4][0].is_zero(),
            "cumulative_chain should also fire on the tampered row",
        );
    }

    #[test]
    fn tampered_non_binary_status_fires() {
        let receipts = vec![make_receipt(1, 21_000)];
        let w = ReceiptStatusWitness::from_receipts(&receipts);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_STATUS][0] = Scalar::from_u64(2, curve);
        let cs = ReceiptStatusConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "status_binary body should fire when status = 2",
        );
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_receipt_status_to_receipt_rlp_descriptor(0, 1);
        assert_eq!(d.label, "receipt_status_to_receipt_rlp_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns, vec![COL_STATUS, COL_CUMULATIVE_GAS]);
        assert_eq!(
            d.b_columns,
            vec![
                crate::receipt_rlp_air::COL_STATUS,
                crate::receipt_rlp_air::COL_CUMULATIVE_GAS,
            ],
        );
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(crate::receipt_rlp_air::COL_IS_REAL),
        );
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_TX_INDEX, 0);
        assert_eq!(COL_STATUS, 1);
        assert_eq!(COL_GAS_USED, 2);
        assert_eq!(COL_CUMULATIVE_GAS, 3);
        assert_eq!(COL_PREV_CUMULATIVE_GAS, 4);
        assert_eq!(COL_GAS_USED_BYTE_OFFSET, 5);
        assert_eq!(COL_CUMUL_BYTE_OFFSET, 13);
        assert_eq!(COL_IS_REAL, 21);
        assert_eq!(COL_IS_FIRST, 22);
        assert_eq!(NUM_COLUMNS, 23);
        assert_eq!(NUM_ROW_CONSTRAINTS, 9);
        assert_eq!(NUM_SHIFTED, 1);
    }

    #[test]
    fn byte_range_lookup_coverage() {
        let cs = ReceiptStatusConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // 8 gas_used_byte + 8 cumul_byte = 16 declarations.
        assert_eq!(reqs.declarations.len(), 2 * U64_BYTES);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }

    #[test]
    fn evaluate_at_point_matches_domain_for_honest() {
        let curve = CurveType::Bls48581;
        let receipts = vec![
            make_receipt(1, 21_000),
            make_receipt(1, 50_000),
        ];
        let w = ReceiptStatusWitness::from_receipts(&receipts);
        let trace = build_trace_polynomials(&w, curve);
        let cs = ReceiptStatusConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(7, curve);
        for r in 0..trace.padded_size as usize {
            let row_evals: Vec<Scalar> =
                col_refs.iter().map(|c| c[r].clone()).collect();
            let agg = cs.evaluate_at_point(&row_evals, &alpha);
            assert!(
                agg.is_zero(),
                "α-RLC aggregate must be zero on honest row {}",
                r,
            );
        }
    }
}
