//! Byte-granularity multi-log concatenation AIR — wires multiple
//! per-log RLP encodings into a single logs list, completing the
//! receipts-with-logs algebraic close.
//!
//! # Motivation
//!
//! [`crate::logs_rlp_air`] encodes a single log per row (~261 cols, 17
//! constraints) and emits per-log RLP byte sequences. A canonical
//! Ethereum receipt's logs field is `RLP([log_0, log_1, ...])` — a
//! variable-length list whose payload is the concatenation of each
//! log's RLP encoding. To bind a receipt's logs region algebraically
//! (a missing piece called out by the comment in
//! [`crate::receipt_rlp_air`]'s no-logs-only `EMPTY_LOGS_BYTE=0xc0`
//! constraint), we need an AIR that:
//!
//! 1. Concatenates per-log encoded byte streams into one ordered
//!    `(log_index, byte_in_log, byte_value)` sequence with running
//!    offset tracking, and
//! 2. Provides cross-AIR LogUp descriptors binding (a) per-log AIR's
//!    encoded bytes ↔ this AIR's per-log byte rows, and (b) this AIR's
//!    `(absolute_offset_in_logs_list, byte_value)` rows ↔ the
//!    receipt's logs region.
//!
//! This module mirrors [`crate::rlp_byte_concat_air`]'s byte-granular
//! pattern (one row per byte) but groups by `log_index` (instead of
//! `field_index`). The cross-AIR linkage shape composes naturally with
//! the receipt's payload byte stream once the receipt AIR is widened to
//! a variable-logs-list case.
//!
//! # Algebraic constraints
//!
//! Row-local (3 total + 8-bit byte_value range check via lookup):
//!   0. `is_real * (is_real - 1) = 0`                              (binary)
//!   1. `is_same_log * (is_same_log - 1) = 0`                      (binary)
//!   2. `is_real * (absolute_offset_in_logs_list
//!                  - running_offset_at_log
//!                  - byte_in_log) = 0`                            (offset eq)
//!
//! Cross-row (shifted, 3 total):
//!   3. `is_same_log(ω·z) * (log_index(ω·z) - log_index(z)) = 0`
//!      — when same_log=1 next row, the log_index actually matches.
//!   4. `is_same_log(ω·z) * is_real(ω·z) *
//!       (running_offset_at_log(ω·z) - running_offset_at_log(z)) = 0`
//!      — running offset constant within a single log.
//!   5. `is_same_log(ω·z) * is_real(ω·z) *
//!       (byte_in_log(ω·z) - byte_in_log(z) - 1) = 0`
//!      — byte_in_log strictly monotonic +1 inside a log.
//!
//! # Soundness sketch
//!
//! The byte_in_log monotonicity + running_offset constancy together
//! force the rows of a given log to form a contiguous ordered byte
//! enumeration starting at `byte_in_log = 0` and at
//! `absolute_offset_in_logs_list = running_offset_at_log`. The
//! `absolute_offset` row-local equation then pins each byte's position
//! in the assembled logs-list stream. The cross-AIR LogUp descriptors
//! complete the binding: (a) ties each per-log AIR row's emitted
//! `encoded_bytes` to this AIR's per-log rows (multiset of byte values
//! at known `byte_in_log` positions), and (b) ties this AIR's
//! `(absolute_offset, byte_value)` rows to whichever byte-row view of
//! the receipt's payload region the caller wires in (e.g. an adapter
//! AIR that unfolds the receipt RLP into per-byte rows).
//!
//! # Deferred
//!
//! - List-prefix-bytes for the logs list (currently the running-offset
//!   start is supplied by the caller; the caller is expected to place
//!   the first log at offset `prefix_len`, where `prefix_len` is the
//!   RLP list-prefix size — 1 byte for short lists `<56`, otherwise
//!   `1 + ceil(log_256(N))`). A small companion AIR or witness column
//!   for the prefix can close this remaining piece.
//! - Adapter AIR unfolding `logs_rlp_air`'s 1-row-per-log × 190-byte
//!   layout into 190 rows-per-log; for joint-prove the descriptor
//!   below presumes such an adapter sits between `logs_rlp_air` and
//!   this AIR.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub};
use crate::receipt::Log;
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

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

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_LOG_INDEX: usize = 0;
pub const COL_BYTE_IN_LOG: usize = 1;
pub const COL_LOG_ENCODED_LEN: usize = 2;
pub const COL_ABSOLUTE_OFFSET: usize = 3;
pub const COL_BYTE_VALUE: usize = 4;
pub const COL_RUNNING_OFFSET_AT_LOG: usize = 5;
pub const COL_IS_REAL: usize = 6;

/// `is_same_log[r]` = 1 iff `log_index[r] == log_index[r-1]`; 0 at r=0.
/// `inv_diff_log[r]` = `(log_index[r] - log_index[r-1])^{-1}` when
/// they differ; 0 otherwise. Reserved for future strict sort
/// constraint pinning a non-zero log_index diff on the same_log=0
/// transition (the shifted constraint 3 below already enforces the
/// other direction: same_log=1 forces equal indices).
pub const COL_IS_SAME_LOG: usize = 7;
pub const COL_INV_DIFF_LOG: usize = 8;

pub const NUM_COLUMNS: usize = 9;

/// Row-local constraints (3, lookup separate):
///   0. `is_real * (is_real - 1) = 0`
///   1. `is_same_log * (is_same_log - 1) = 0`
///   2. `is_real * (absolute_offset - running_offset_at_log - byte_in_log) = 0`
pub const NUM_ROW_CONSTRAINTS: usize = 3;

/// Cross-row (shifted) constraints (3):
///   3. `is_same_log(ω·z) * (log_index(ω·z) - log_index(z)) = 0`
///   4. `is_same_log(ω·z) * is_real(ω·z) *
///       (running_offset_at_log(ω·z) - running_offset_at_log(z)) = 0`
///   5. `is_same_log(ω·z) * is_real(ω·z) *
///       (byte_in_log(ω·z) - byte_in_log(z) - 1) = 0`
pub const NUM_SHIFTED: usize = 3;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LogsListConcatRow {
    pub log_index: u64,
    pub byte_in_log: u64,
    pub log_encoded_len: u64,
    pub absolute_offset_in_logs_list: u64,
    pub byte_value: u8,
    pub running_offset_at_log: u64,
}

#[derive(Clone, Debug, Default)]
pub struct LogsListConcatWitness {
    pub rows: Vec<LogsListConcatRow>,
    /// Encoded length of each log in canonical order (cached for
    /// host-side cross-checks and per-log descriptor adapter wiring).
    pub log_encoded_lens: Vec<u64>,
    /// Total bytes across all logs (i.e. the logs-list payload length,
    /// excluding the list prefix).
    pub total_payload_len: u64,
    /// Starting offset of the first log within the logs-list stream
    /// (= the list prefix length). Defaults to 0 when built via
    /// [`Self::from_logs`], leaving the caller to add the prefix.
    pub list_prefix_len: u64,
}

impl LogsListConcatWitness {
    /// Build the byte-granular witness from a slice of logs.
    ///
    /// One row per byte of each log, in the order
    /// `log0_byte0, log0_byte1, ..., log0_byteN0-1, log1_byte0, ...`.
    /// Each log's bytes are placed at `running_offset_at_log`
    /// starting at `list_prefix_len` (default 0).
    pub fn from_logs(logs: &[Log]) -> Self {
        Self::from_logs_with_prefix(logs, 0)
    }

    /// Same as [`Self::from_logs`] but threads a non-zero
    /// `list_prefix_len` (the RLP list-prefix size for the logs list).
    pub fn from_logs_with_prefix(logs: &[Log], list_prefix_len: u64) -> Self {
        let mut rows = Vec::new();
        let mut log_encoded_lens = Vec::with_capacity(logs.len());
        let mut running = list_prefix_len;
        for (i, log) in logs.iter().enumerate() {
            let encoded = log.rlp_encode();
            let enc_len = encoded.len() as u64;
            log_encoded_lens.push(enc_len);
            for (k, &b) in encoded.iter().enumerate() {
                rows.push(LogsListConcatRow {
                    log_index: i as u64,
                    byte_in_log: k as u64,
                    log_encoded_len: enc_len,
                    absolute_offset_in_logs_list: running + k as u64,
                    byte_value: b,
                    running_offset_at_log: running,
                });
            }
            running += enc_len;
        }
        let total_payload_len = running - list_prefix_len;
        Self {
            rows,
            log_encoded_lens,
            total_payload_len,
            list_prefix_len,
        }
    }

    /// Host-side: verify each row's
    /// `absolute_offset_in_logs_list == running_offset_at_log + byte_in_log`.
    pub fn verify_offset_equation(&self) -> Result<(), String> {
        for (r, row) in self.rows.iter().enumerate() {
            if row.absolute_offset_in_logs_list != row.running_offset_at_log + row.byte_in_log {
                return Err(format!(
                    "row {} offset equation broken: absolute={} \
                     running_offset_at_log={} byte_in_log={}",
                    r,
                    row.absolute_offset_in_logs_list,
                    row.running_offset_at_log,
                    row.byte_in_log,
                ));
            }
        }
        Ok(())
    }

    /// Host-side: verify byte_in_log monotonicity within a log,
    /// running_offset constancy, and that each row's log_encoded_len
    /// equals the cached per-log length.
    pub fn verify_log_grouping(&self) -> Result<(), String> {
        for i in 1..self.rows.len() {
            let prev = &self.rows[i - 1];
            let cur = &self.rows[i];
            if cur.log_index == prev.log_index {
                if cur.byte_in_log != prev.byte_in_log + 1 {
                    return Err(format!(
                        "row {} byte_in_log {} not consecutive with prev {}",
                        i, cur.byte_in_log, prev.byte_in_log,
                    ));
                }
                if cur.running_offset_at_log != prev.running_offset_at_log {
                    return Err(format!(
                        "row {} running_offset_at_log={} differs from prev={} in log {}",
                        i, cur.running_offset_at_log,
                        prev.running_offset_at_log, cur.log_index,
                    ));
                }
                if cur.log_encoded_len != prev.log_encoded_len {
                    return Err(format!(
                        "row {} log_encoded_len={} differs from prev={} in log {}",
                        i, cur.log_encoded_len,
                        prev.log_encoded_len, cur.log_index,
                    ));
                }
            }
        }
        // Cross-check final running offset accounts for total_payload_len.
        if let Some(last) = self.rows.last() {
            let final_offset = last.absolute_offset_in_logs_list + 1;
            let expected = self.list_prefix_len + self.total_payload_len;
            if final_offset != expected {
                return Err(format!(
                    "final absolute_offset+1 = {} != list_prefix_len + total_payload_len = {}",
                    final_offset, expected,
                ));
            }
        }
        Ok(())
    }

    /// Host-side: verify the assembled byte stream matches
    /// `rlp_encode_list([log_0_encoded, log_1_encoded, ...])` byte-by-byte
    /// after stripping the list prefix.
    pub fn verify_against_canonical_payload(&self, canonical_payload: &[u8]) -> Result<(), String> {
        let start = self.list_prefix_len as usize;
        for (r, row) in self.rows.iter().enumerate() {
            let pos = row.absolute_offset_in_logs_list as usize;
            if pos < start {
                return Err(format!(
                    "row {} absolute_offset {} below list_prefix_len {}",
                    r, pos, start,
                ));
            }
            let canon_pos = pos - start;
            if canon_pos >= canonical_payload.len() {
                return Err(format!(
                    "row {} payload index {} exceeds canonical_payload len {}",
                    r, canon_pos, canonical_payload.len(),
                ));
            }
            if canonical_payload[canon_pos] != row.byte_value {
                return Err(format!(
                    "row {} byte at payload index {} = {} but canonical = {}",
                    r, canon_pos, row.byte_value, canonical_payload[canon_pos],
                ));
            }
        }
        Ok(())
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &LogsListConcatWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_LOG_INDEX][i] = Scalar::from_u64(row.log_index, curve);
        columns[COL_BYTE_IN_LOG][i] = Scalar::from_u64(row.byte_in_log, curve);
        columns[COL_LOG_ENCODED_LEN][i] = Scalar::from_u64(row.log_encoded_len, curve);
        columns[COL_ABSOLUTE_OFFSET][i] =
            Scalar::from_u64(row.absolute_offset_in_logs_list, curve);
        columns[COL_BYTE_VALUE][i] = Scalar::from_u64(row.byte_value as u64, curve);
        columns[COL_RUNNING_OFFSET_AT_LOG][i] =
            Scalar::from_u64(row.running_offset_at_log, curve);
        columns[COL_IS_REAL][i] = one.clone();

        if i >= 1 {
            let prev_li = witness.rows[i - 1].log_index;
            if row.log_index == prev_li {
                columns[COL_IS_SAME_LOG][i] = one.clone();
                // inv_diff_log stays zero.
            } else {
                let diff_scalar = if row.log_index > prev_li {
                    Scalar::from_u64(row.log_index - prev_li, curve)
                } else {
                    let pos = Scalar::from_u64(prev_li - row.log_index, curve);
                    Scalar::zero(curve).sub(&pos)
                };
                columns[COL_INV_DIFF_LOG][i] = diff_scalar.inverse();
            }
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

pub struct LogsListConcatConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl LogsListConcatConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for LogsListConcatConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_same_log_binary".into(),
            "offset_equation".into(),
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

        let mut bin_real = vec![Scalar::zero(curve); n];
        let mut bin_sl = vec![Scalar::zero(curve); n];
        let mut off_eq = vec![Scalar::zero(curve); n];

        for r in 0..n {
            let v_real = &columns[COL_IS_REAL][r];
            bin_real[r] = v_real.mul(&v_real.sub(&one));

            let v_sl = &columns[COL_IS_SAME_LOG][r];
            bin_sl[r] = v_sl.mul(&v_sl.sub(&one));

            let abs = &columns[COL_ABSOLUTE_OFFSET][r];
            let rol = &columns[COL_RUNNING_OFFSET_AT_LOG][r];
            let bil = &columns[COL_BYTE_IN_LOG][r];
            let diff = abs.sub(rol).sub(bil);
            off_eq[r] = v_real.mul(&diff);
        }

        vec![bin_real, bin_sl, off_eq]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let v_real = &col_evals[COL_IS_REAL];
        let bin_real = v_real.mul(&v_real.sub(&one));

        let v_sl = &col_evals[COL_IS_SAME_LOG];
        let bin_sl = v_sl.mul(&v_sl.sub(&one));

        let abs = &col_evals[COL_ABSOLUTE_OFFSET];
        let rol = &col_evals[COL_RUNNING_OFFSET_AT_LOG];
        let bil = &col_evals[COL_BYTE_IN_LOG];
        let off_eq = v_real.mul(&abs.sub(rol).sub(bil));

        let mut acc = bin_real;
        let mut ap = alpha.clone();
        acc = acc.add(&ap.mul(&bin_sl));
        ap = ap.mul(alpha);
        acc = acc.add(&ap.mul(&off_eq));
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

        let v_real = &col_coeffs[COL_IS_REAL];
        let v_real_m1 = poly_sub(v_real, &one_poly, curve);
        let bin_real = poly_mul(v_real, &v_real_m1, curve);

        let v_sl = &col_coeffs[COL_IS_SAME_LOG];
        let v_sl_m1 = poly_sub(v_sl, &one_poly, curve);
        let bin_sl = poly_mul(v_sl, &v_sl_m1, curve);

        let abs = &col_coeffs[COL_ABSOLUTE_OFFSET];
        let rol = &col_coeffs[COL_RUNNING_OFFSET_AT_LOG];
        let bil = &col_coeffs[COL_BYTE_IN_LOG];
        let diff = poly_sub(&poly_sub(abs, rol, curve), bil, curve);
        let off_eq = poly_mul(v_real, &diff, curve);

        let mut acc = bin_real;
        let mut ap = alpha.clone();
        acc = poly_add(&acc, &poly_scalar_mul(&bin_sl, &ap), curve);
        ap = ap.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&off_eq, &ap), curve);
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
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
        // 8-bit range check on byte_value column.
        let tables = vec![LookupTable::range(256)];
        let declarations = vec![(
            LookupDeclaration {
                label: "logs_list_concat_byte_value_8bit".into(),
                column_index: COL_BYTE_VALUE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        )];
        LookupRequirements { tables, declarations }
    }

    // ── Cross-row (shifted) constraints ──────────────────────────────

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Shift order (must match `evaluate_shifted_at_point` indexing):
        //   0: LOG_INDEX
        //   1: IS_REAL
        //   2: IS_SAME_LOG
        //   3: INV_DIFF_LOG
        //   4: RUNNING_OFFSET_AT_LOG
        //   5: BYTE_IN_LOG
        vec![
            COL_LOG_INDEX,
            COL_IS_REAL,
            COL_IS_SAME_LOG,
            COL_INV_DIFF_LOG,
            COL_RUNNING_OFFSET_AT_LOG,
            COL_BYTE_IN_LOG,
        ]
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        if shifted_evals.len() < 6 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let li_wz = &shifted_evals[0];
        let is_real_wz = &shifted_evals[1];
        let is_same_log_wz = &shifted_evals[2];
        let inv_diff_log_wz = &shifted_evals[3];
        let rol_wz = &shifted_evals[4];
        let bil_wz = &shifted_evals[5];

        let li_z = &col_evals_at_z[COL_LOG_INDEX];
        let rol_z = &col_evals_at_z[COL_RUNNING_OFFSET_AT_LOG];
        let bil_z = &col_evals_at_z[COL_BYTE_IN_LOG];

        let exclusion = z.sub(omega_n_minus_1);
        let log_diff = li_wz.sub(li_z);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut result = Scalar::zero(curve);

        // Constraint 3: is_same_log(ω·z) * (li(ω·z) - li(z)) = 0
        let body0 = is_same_log_wz.mul(&log_diff);
        result = result.add(&ap.mul(&body0).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 4: is_same_log(ω·z) * is_real(ω·z) * (rol(ω·z) - rol(z)) = 0
        let rol_diff = rol_wz.sub(rol_z);
        let body1 = is_same_log_wz.mul(is_real_wz).mul(&rol_diff);
        result = result.add(&ap.mul(&body1).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 5: is_same_log(ω·z) * is_real(ω·z) *
        //   (bil(ω·z) - bil(z) - 1) = 0
        let bil_step = bil_wz.sub(bil_z).sub(&one);
        let body2 = is_same_log_wz.mul(is_real_wz).mul(&bil_step);
        result = result.add(&ap.mul(&body2).mul(&exclusion));

        // Touch reserved witness column to avoid dead-code warnings.
        let _ = inv_diff_log_wz;

        result
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

        let li = &col_coeffs[COL_LOG_INDEX];
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_same_log = &col_coeffs[COL_IS_SAME_LOG];
        let rol = &col_coeffs[COL_RUNNING_OFFSET_AT_LOG];
        let bil = &col_coeffs[COL_BYTE_IN_LOG];

        let li_shift = poly_shift(li, omega);
        let is_real_shift = poly_shift(is_real, omega);
        let is_same_log_shift = poly_shift(is_same_log, omega);
        let rol_shift = poly_shift(rol, omega);
        let bil_shift = poly_shift(bil, omega);

        let log_diff = poly_sub(&li_shift, li, curve);
        let rol_diff = poly_sub(&rol_shift, rol, curve);
        let bil_step = poly_sub(&poly_sub(&bil_shift, bil, curve), &one_poly, curve);

        // Body 0
        let body0 = poly_mul(&is_same_log_shift, &log_diff, curve);
        // Body 1
        let body1 = poly_mul(
            &poly_mul(&is_same_log_shift, &is_real_shift, curve),
            &rol_diff,
            curve,
        );
        // Body 2
        let body2 = poly_mul(
            &poly_mul(&is_same_log_shift, &is_real_shift, curve),
            &bil_step,
            curve,
        );

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let ex0 = poly_mul_linear(&body0, &omega_n_minus_1);
        let ex1 = poly_mul_linear(&body1, &omega_n_minus_1);
        let ex2 = poly_mul_linear(&body2, &omega_n_minus_1);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = poly_scalar_mul(&ex0, &ap);
        ap = ap.mul(alpha);
        let term1 = poly_scalar_mul(&ex1, &ap);
        ap = ap.mul(alpha);
        let term2 = poly_scalar_mul(&ex2, &ap);

        let s01 = poly_add(&term0, &term1, curve);
        poly_add(&s01, &term2, curve)
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind a per-log AIR's encoded byte sequence (for log `log_index`) to
/// this AIR's rows for that log, as a multiset of `(byte_value,)`
/// tuples. The caller is expected to provide a per-log selector column
/// on the concat side so only that log's rows participate.
///
/// **Shape note**: identical to
/// [`crate::rlp_byte_concat_air::make_field_byte_to_concat_descriptor`].
/// `logs_rlp_air` has 1 row × ~190 byte columns per log; full
/// per-position binding requires an adapter AIR unfolding that row
/// into rows of `(byte_in_log, byte_value)`. The descriptor here is
/// the SHAPE CONTRACT; the caller passes the gadget's per-byte value
/// columns and a per-log gate.
pub fn make_per_log_to_concat_descriptor(
    log_index: u64,
    per_log_layer_index: usize,
    per_log_byte_value_cols: Vec<usize>,
    per_log_selector_col: Option<usize>,
    concat_layer_index: usize,
    concat_per_log_selector_col: Option<usize>,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: format!("logs_list_concat_log_{}_byte_v1", log_index),
        a_layer_index: per_log_layer_index,
        a_columns: per_log_byte_value_cols,
        a_selector_column: per_log_selector_col,
        b_layer_index: concat_layer_index,
        b_columns: vec![COL_BYTE_VALUE],
        b_selector_column: concat_per_log_selector_col,
    }
}

/// Bind this AIR's `(absolute_offset_in_logs_list, byte_value)` rows
/// to a receipt-side byte-row view of the logs region. The receipt
/// AIR currently exposes its no-logs `EMPTY_LOGS_BYTE=0xc0` column;
/// for the multi-log case the receipt AIR (or a companion adapter)
/// must expose `(position, byte_value)` rows for the logs payload at
/// known column indices, which the caller passes in here.
///
/// Tuple: `(absolute_offset_in_logs_list, byte_value)` (2 cols on each
/// side). Multiset equality forces every byte the concat AIR emits to
/// be present at the same absolute position in the receipt's logs
/// region.
pub fn make_concat_to_receipt_logs_descriptor(
    concat_layer_index: usize,
    receipt_logs_layer_index: usize,
    receipt_logs_position_col: usize,
    receipt_logs_byte_value_col: usize,
    receipt_logs_selector_col: Option<usize>,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "logs_list_concat_to_receipt_logs_v1".into(),
        a_layer_index: concat_layer_index,
        a_columns: vec![COL_ABSOLUTE_OFFSET, COL_BYTE_VALUE],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: receipt_logs_layer_index,
        b_columns: vec![receipt_logs_position_col, receipt_logs_byte_value_col],
        b_selector_column: receipt_logs_selector_col,
    }
}

/// Bind this AIR's per-log `(log_index, log_encoded_len)` value to the
/// corresponding per-log AIR row's `encoded_len` column. Tuple:
/// `(log_index_or_synthetic_row_id, log_encoded_len)` (2 cols on each
/// side). Rows of the same log publish identical tuples (length
/// constancy follows from running_offset constancy in shifted
/// constraint 4 combined with the offset equation — though
/// `log_encoded_len` itself is not currently subject to a constancy
/// constraint; pair this with the host-side check in
/// [`LogsListConcatWitness::verify_log_grouping`] until that
/// constraint is added).
pub fn make_concat_to_per_log_len_descriptor(
    concat_layer_index: usize,
    per_log_layer_index: usize,
    per_log_log_index_col: usize,
    per_log_encoded_len_col: usize,
    per_log_selector_col: Option<usize>,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "logs_list_concat_to_per_log_len_v1".into(),
        a_layer_index: concat_layer_index,
        a_columns: vec![COL_LOG_INDEX, COL_LOG_ENCODED_LEN],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: per_log_layer_index,
        b_columns: vec![per_log_log_index_col, per_log_encoded_len_col],
        b_selector_column: per_log_selector_col,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_no_topics_empty_data() -> Log {
        Log {
            address: [0x11u8; 20],
            topics: vec![],
            data: vec![],
        }
    }

    fn log_one_topic_short_data() -> Log {
        Log {
            address: [0x22u8; 20],
            topics: vec![[0xaa; 32]],
            data: vec![0xde, 0xad, 0xbe, 0xef],
        }
    }

    fn log_two_topics_thirty_two_data() -> Log {
        Log {
            address: [0x33u8; 20],
            topics: vec![[0x55; 32], [0x66; 32]],
            data: vec![0xff; 32],
        }
    }

    // ─── Witness-level sanity ─────────────────────────────────────────

    #[test]
    fn witness_builds_for_single_log() {
        let l = log_no_topics_empty_data();
        let expected_len = l.rlp_encode().len();
        let w = LogsListConcatWitness::from_logs(&[l.clone()]);
        assert_eq!(w.rows.len(), expected_len);
        assert_eq!(w.log_encoded_lens, vec![expected_len as u64]);
        assert_eq!(w.total_payload_len, expected_len as u64);
        assert_eq!(w.list_prefix_len, 0);
        assert_eq!(w.rows[0].log_index, 0);
        assert_eq!(w.rows[0].byte_in_log, 0);
        assert_eq!(w.rows[0].absolute_offset_in_logs_list, 0);
        assert_eq!(w.rows[0].running_offset_at_log, 0);
        // Last row
        let last = w.rows.last().unwrap();
        assert_eq!(last.byte_in_log as usize, expected_len - 1);
        assert_eq!(last.absolute_offset_in_logs_list as usize, expected_len - 1);
    }

    #[test]
    fn witness_builds_for_multiple_logs() {
        let logs = vec![
            log_no_topics_empty_data(),
            log_one_topic_short_data(),
            log_two_topics_thirty_two_data(),
        ];
        let lens: Vec<usize> = logs.iter().map(|l| l.rlp_encode().len()).collect();
        let total: usize = lens.iter().sum();
        let w = LogsListConcatWitness::from_logs(&logs);
        assert_eq!(w.rows.len(), total);
        assert_eq!(w.log_encoded_lens.len(), 3);
        assert_eq!(
            w.log_encoded_lens,
            lens.iter().map(|&n| n as u64).collect::<Vec<_>>(),
        );
        // Running offset of log 1 should equal len of log 0.
        let row_for_log1 = w.rows.iter().find(|r| r.log_index == 1).unwrap();
        assert_eq!(row_for_log1.running_offset_at_log, lens[0] as u64);
        // Running offset of log 2 should equal lens[0] + lens[1].
        let row_for_log2 = w.rows.iter().find(|r| r.log_index == 2).unwrap();
        assert_eq!(
            row_for_log2.running_offset_at_log,
            (lens[0] + lens[1]) as u64,
        );
    }

    #[test]
    fn host_side_offset_equation_and_grouping_hold_for_honest_witness() {
        let logs = vec![
            log_no_topics_empty_data(),
            log_one_topic_short_data(),
            log_two_topics_thirty_two_data(),
        ];
        let w = LogsListConcatWitness::from_logs(&logs);
        w.verify_offset_equation().unwrap();
        w.verify_log_grouping().unwrap();
    }

    #[test]
    fn host_side_against_canonical_logs_list_payload_passes() {
        let logs = vec![log_one_topic_short_data(), log_two_topics_thirty_two_data()];
        let w = LogsListConcatWitness::from_logs(&logs);
        // Canonical payload (no list prefix) = concat of each log's RLP.
        let mut payload = Vec::new();
        for l in &logs {
            payload.extend_from_slice(&l.rlp_encode());
        }
        w.verify_against_canonical_payload(&payload).unwrap();
    }

    #[test]
    fn host_side_detects_tampered_byte() {
        let logs = vec![log_one_topic_short_data(), log_two_topics_thirty_two_data()];
        let mut w = LogsListConcatWitness::from_logs(&logs);
        let mut payload = Vec::new();
        for l in &logs {
            payload.extend_from_slice(&l.rlp_encode());
        }
        // Tamper a byte mid-stream.
        w.rows[10].byte_value ^= 0xff;
        let err = w.verify_against_canonical_payload(&payload).unwrap_err();
        assert!(err.contains("row 10"), "got: {}", err);
    }

    #[test]
    fn host_side_detects_tampered_offset() {
        let logs = vec![log_one_topic_short_data()];
        let mut w = LogsListConcatWitness::from_logs(&logs);
        w.rows[5].absolute_offset_in_logs_list += 1;
        let err = w.verify_offset_equation().unwrap_err();
        assert!(err.contains("offset equation broken"), "got: {}", err);
    }

    #[test]
    fn witness_with_list_prefix_shifts_offsets() {
        let logs = vec![log_one_topic_short_data()];
        let w = LogsListConcatWitness::from_logs_with_prefix(&logs, 3);
        assert_eq!(w.list_prefix_len, 3);
        assert_eq!(w.rows[0].running_offset_at_log, 3);
        assert_eq!(w.rows[0].absolute_offset_in_logs_list, 3);
        // Last row absolute = 3 + len - 1.
        let last = w.rows.last().unwrap();
        let len = logs[0].rlp_encode().len() as u64;
        assert_eq!(last.absolute_offset_in_logs_list, 3 + len - 1);
        // Total payload still equals concat length.
        assert_eq!(w.total_payload_len, len);
    }

    // ─── Algebraic constraint tests ───────────────────────────────────

    fn run_evaluate_on_domain(witness: &LogsListConcatWitness) -> Vec<Vec<Scalar>> {
        let trace = build_trace_polynomials(witness, CurveType::Bls48581);
        let cs = LogsListConcatConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    #[test]
    fn row_local_constraints_zero_on_honest_multi_log_witness() {
        let logs = vec![
            log_no_topics_empty_data(),
            log_one_topic_short_data(),
            log_two_topics_thirty_two_data(),
        ];
        let w = LogsListConcatWitness::from_logs(&logs);
        let results = run_evaluate_on_domain(&w);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "row-local constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn row_local_constraints_zero_on_single_log() {
        let w = LogsListConcatWitness::from_logs(&[log_one_topic_short_data()]);
        let results = run_evaluate_on_domain(&w);
        for col in results.iter() {
            for val in col.iter() {
                assert!(val.is_zero(), "constraint nonzero on single log");
            }
        }
    }

    #[test]
    fn offset_equation_constraint_fires_on_tampered_absolute_offset() {
        let logs = vec![log_one_topic_short_data(), log_two_topics_thirty_two_data()];
        let w = LogsListConcatWitness::from_logs(&logs);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: increment row 7's absolute_offset by 1.
        let orig = cols[COL_ABSOLUTE_OFFSET][7].clone();
        cols[COL_ABSOLUTE_OFFSET][7] = orig.add(&Scalar::one(CurveType::Bls48581));
        let cs = LogsListConcatConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 2 = offset_equation should fire at row 7.
        assert!(
            !results[2][7].is_zero(),
            "offset_equation should detect tampered absolute_offset",
        );
        assert!(results[2][0].is_zero());
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary_value() {
        let w = LogsListConcatWitness::from_logs(&[log_one_topic_short_data()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(3, CurveType::Bls48581);
        let cs = LogsListConcatConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "is_real_binary should fire");
    }

    // ─── Shifted constraint tests ─────────────────────────────────────

    fn check_shifted_at_transition(
        trace: &TracePolynomials,
        r: usize,
        alpha: &Scalar,
    ) -> Scalar {
        let curve = trace.curve;
        let z = Scalar::from_u64(7, curve);
        let omega_n_minus_1 = Scalar::zero(curve);
        let col_evals_at_z: Vec<Scalar> =
            trace.columns.iter().map(|p| p.evaluations[r].clone()).collect();
        let next = r + 1;
        let shifted_evals = vec![
            trace.columns[COL_LOG_INDEX].evaluations[next].clone(),
            trace.columns[COL_IS_REAL].evaluations[next].clone(),
            trace.columns[COL_IS_SAME_LOG].evaluations[next].clone(),
            trace.columns[COL_INV_DIFF_LOG].evaluations[next].clone(),
            trace.columns[COL_RUNNING_OFFSET_AT_LOG].evaluations[next].clone(),
            trace.columns[COL_BYTE_IN_LOG].evaluations[next].clone(),
        ];
        let cs = LogsListConcatConstraintSystem::new(trace.num_rows);
        cs.evaluate_shifted_at_point(
            &col_evals_at_z,
            &shifted_evals,
            &z,
            &omega_n_minus_1,
            alpha,
            NUM_ROW_CONSTRAINTS,
        )
    }

    #[test]
    fn shifted_constraints_zero_on_honest_multi_log_witness() {
        let logs = vec![
            log_no_topics_empty_data(),
            log_one_topic_short_data(),
            log_two_topics_thirty_two_data(),
        ];
        let w = LogsListConcatWitness::from_logs(&logs);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let alpha = Scalar::from_u64(31337, CurveType::Bls48581);
        for r in 0..(trace.num_rows - 1) {
            let v = check_shifted_at_transition(&trace, r, &alpha);
            assert!(
                v.is_zero(),
                "shifted constraint at row {} = {:?} (expected zero)",
                r, v,
            );
        }
    }

    #[test]
    fn shifted_constraint_detects_broken_byte_in_log_monotonicity() {
        let logs = vec![log_one_topic_short_data(), log_two_topics_thirty_two_data()];
        let w = LogsListConcatWitness::from_logs(&logs);
        let mut trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // Find intra-log transition.
        let curve = CurveType::Bls48581;
        let mut victim = None;
        for r in 0..(trace.num_rows - 1) {
            let li_r = trace.columns[COL_LOG_INDEX].evaluations[r].to_u64();
            let li_next = trace.columns[COL_LOG_INDEX].evaluations[r + 1].to_u64();
            if li_r == li_next {
                victim = Some(r + 1);
                break;
            }
        }
        let v = victim.expect("expected intra-log transition");
        let orig = trace.columns[COL_BYTE_IN_LOG].evaluations[v].clone();
        trace.columns[COL_BYTE_IN_LOG].evaluations[v] =
            orig.add(&Scalar::from_u64(4, curve));
        let alpha = Scalar::from_u64(31337, curve);
        let val = check_shifted_at_transition(&trace, v - 1, &alpha);
        assert!(
            !val.is_zero(),
            "byte_in_log monotonicity should fire at tampered transition",
        );
    }

    #[test]
    fn shifted_constraint_detects_broken_running_offset_constancy() {
        let logs = vec![log_one_topic_short_data(), log_two_topics_thirty_two_data()];
        let w = LogsListConcatWitness::from_logs(&logs);
        let mut trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let curve = CurveType::Bls48581;
        let mut victim = None;
        for r in 0..(trace.num_rows - 1) {
            let li_r = trace.columns[COL_LOG_INDEX].evaluations[r].to_u64();
            let li_next = trace.columns[COL_LOG_INDEX].evaluations[r + 1].to_u64();
            if li_r == li_next {
                victim = Some(r + 1);
                break;
            }
        }
        let v = victim.expect("expected intra-log transition");
        let orig = trace.columns[COL_RUNNING_OFFSET_AT_LOG].evaluations[v].clone();
        trace.columns[COL_RUNNING_OFFSET_AT_LOG].evaluations[v] =
            orig.add(&Scalar::from_u64(7, curve));
        let alpha = Scalar::from_u64(31337, curve);
        let val = check_shifted_at_transition(&trace, v - 1, &alpha);
        assert!(
            !val.is_zero(),
            "running_offset constancy should fire on tampered RoF",
        );
    }

    // ─── Descriptor well-formedness ───────────────────────────────────

    #[test]
    fn per_log_to_concat_descriptor_well_formed() {
        let desc = make_per_log_to_concat_descriptor(
            2,
            1,
            vec![100, 101, 102],
            Some(99),
            0,
            Some(COL_IS_REAL),
        );
        assert!(desc.label.contains("log_2"));
        assert_eq!(desc.a_layer_index, 1);
        assert_eq!(desc.a_columns, vec![100, 101, 102]);
        assert_eq!(desc.a_selector_column, Some(99));
        assert_eq!(desc.b_layer_index, 0);
        assert_eq!(desc.b_columns, vec![COL_BYTE_VALUE]);
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn concat_to_receipt_logs_descriptor_well_formed() {
        let desc = make_concat_to_receipt_logs_descriptor(0, 2, 4321, 8765, Some(13));
        assert_eq!(desc.label, "logs_list_concat_to_receipt_logs_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.a_columns, vec![COL_ABSOLUTE_OFFSET, COL_BYTE_VALUE]);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(desc.b_layer_index, 2);
        assert_eq!(desc.b_columns, vec![4321, 8765]);
        assert_eq!(desc.b_selector_column, Some(13));
        assert_eq!(desc.a_columns.len(), desc.b_columns.len());
    }

    #[test]
    fn concat_to_per_log_len_descriptor_well_formed() {
        let desc = make_concat_to_per_log_len_descriptor(0, 1, 42, 43, Some(44));
        assert_eq!(desc.label, "logs_list_concat_to_per_log_len_v1");
        assert_eq!(desc.a_columns, vec![COL_LOG_INDEX, COL_LOG_ENCODED_LEN]);
        assert_eq!(desc.b_columns, vec![42, 43]);
        assert_eq!(desc.a_columns.len(), desc.b_columns.len());
    }

    // ─── Coverage / structural pinning ───────────────────────────────

    #[test]
    fn three_log_witness_has_consecutive_byte_indices_per_log() {
        let logs = vec![
            log_no_topics_empty_data(),
            log_one_topic_short_data(),
            log_two_topics_thirty_two_data(),
        ];
        let w = LogsListConcatWitness::from_logs(&logs);

        // Sanity: bytes-per-log counts add up against canonical lens.
        let mut bytes_per_log = std::collections::HashMap::new();
        for r in &w.rows {
            *bytes_per_log.entry(r.log_index).or_insert(0u64) += 1;
        }
        for (i, &expected_len) in w.log_encoded_lens.iter().enumerate() {
            assert_eq!(
                bytes_per_log.get(&(i as u64)).copied().unwrap_or(0),
                expected_len,
            );
        }

        // Sanity: each log's bytes are 0..N contiguous.
        let mut prev_log: Option<u64> = None;
        let mut counter = 0u64;
        for row in &w.rows {
            if Some(row.log_index) != prev_log {
                counter = 0;
            }
            assert_eq!(row.byte_in_log, counter);
            counter += 1;
            prev_log = Some(row.log_index);
        }
    }

    #[test]
    fn num_columns_and_constraint_counts_pinned() {
        assert_eq!(NUM_COLUMNS, 9);
        assert_eq!(NUM_ROW_CONSTRAINTS, 3);
        assert_eq!(NUM_SHIFTED, 3);
    }
}
