//! EIP-7251 compounding-credentials AIR.
//!
//! Proves the per-validator effective-balance cap matches the
//! withdrawal-credentials discriminator under EIP-7251 (Electra):
//!
//!   * Legacy (`withdrawal_credentials[0] ∈ {0x00, 0x01}`) — the
//!     validator's `effective_balance` MUST be ≤
//!     `MAX_EFFECTIVE_BALANCE` (32 ETH = 32 · 10⁹ Gwei).
//!   * Compounding (`withdrawal_credentials[0] = 0x02`) — the
//!     validator's `effective_balance` MUST be ≤
//!     `MAX_EFFECTIVE_BALANCE_ELECTRA` (2048 ETH = 2048 · 10⁹ Gwei).
//!
//! The AIR commits a slack column per cap so the cap check is encoded
//! algebraically as
//!
//!     CAP − effective_balance − slack = 0
//!
//! gated by the matching prefix selector, with byte-range checks
//! constraining `slack` and `effective_balance` to 8 bytes (sufficient
//! to bound both caps, ≤ 2^41 Gwei).
//!
//! ## Constraints (10 row-local bodies)
//!
//! 0. `is_real_binary`           — `is_real · (is_real − 1) = 0`
//! 1. `is_compounding_binary`    — `is_compounding · (is_compounding − 1) = 0`
//! 2. `compounding_prefix_pin`   — `is_compounding · (prefix − 2) = 0`
//! 3. `is_real_gates_compound`   — `is_compounding · (1 − is_real) = 0`
//!    (i.e. compounding only fires on real rows).
//! 4. `legacy_cap_bind`          —
//!    `(1 − is_compounding) · is_real ·
//!     (MAX_EFFECTIVE_BALANCE − effective_balance − slack_legacy) = 0`.
//! 5. `compounding_cap_bind`     —
//!    `is_compounding ·
//!     (MAX_EFFECTIVE_BALANCE_ELECTRA − effective_balance − slack_electra) = 0`.
//! 6. `effective_balance_le_decomp` —
//!    `is_real · (effective_balance − Σ_b EB_BYTE[b]·2^(8b)) = 0`.
//! 7. `slack_legacy_le_decomp`   —
//!    `is_real · (slack_legacy − Σ_b SLACK_LEGACY_BYTE[b]·2^(8b)) = 0`.
//! 8. `slack_electra_le_decomp`  —
//!    `is_real · (slack_electra − Σ_b SLACK_ELECTRA_BYTE[b]·2^(8b)) = 0`.
//! 9. `validator_index_le_decomp` —
//!    `validator_index − Σ_b VI_BYTE[b]·2^(8b) = 0`.
//!
//! Byte-range lookups (8-bit) cover every committed byte column,
//! which together with the LE-decomp bodies upper-bound each scalar to
//! `< 2^64` — strictly enough to bind both caps (the legacy cap is
//! ~2^35 and the Electra cap is ~2^41).
//!
//! ## Cross-AIR linkages
//!
//! * [`make_compounding_to_withdrawal_credential_descriptor`] binds
//!   `(validator_index, prefix)` to
//!   [`crate::withdrawal_credential_air`]'s
//!   `(COL_VALIDATOR_INDEX, COL_WC_BYTE_OFFSET[0])`. This forces the
//!   `prefix` column on this AIR to match the on-chain
//!   withdrawal-credentials first byte for the claimed validator.
//! * [`make_compounding_to_validator_balances_descriptor`] binds
//!   `(validator_index, effective_balance)` to
//!   [`crate::validator_balances_air`]'s
//!   `(COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE)`. This forces the
//!   `effective_balance` column on this AIR to match the registry's
//!   exposed effective balance for the claimed validator.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

/// Gwei per ETH.
pub const GWEI_PER_ETH: u64 = 1_000_000_000;

/// `MAX_EFFECTIVE_BALANCE` = 32 ETH (legacy / pre-Electra cap on
/// `effective_balance` for `0x00`/`0x01` prefixes).
pub const MAX_EFFECTIVE_BALANCE: u64 = 32 * GWEI_PER_ETH;

/// `MAX_EFFECTIVE_BALANCE_ELECTRA` = 2048 ETH (EIP-7251 compounding
/// cap on `effective_balance` for the `0x02` prefix).
pub const MAX_EFFECTIVE_BALANCE_ELECTRA: u64 = 2048 * GWEI_PER_ETH;

/// Bytes per u64 LE decomposition (covers both caps; ~2^41 max).
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

/// `u64` validator_index (bound by the WC linkage).
pub const COL_VALIDATOR_INDEX: usize = 0;
/// 8-byte LE decomposition of `validator_index`.
pub const COL_VI_BYTE_OFFSET: usize = COL_VALIDATOR_INDEX + 1;

/// `withdrawal_credentials[0]` discriminator byte ∈ {0, 1, 2}.
pub const COL_PREFIX: usize = COL_VI_BYTE_OFFSET + U64_BYTES;

/// `effective_balance` (Gwei) as a single u64 limb.
pub const COL_EFFECTIVE_BALANCE: usize = COL_PREFIX + 1;
/// 8-byte LE decomposition of `effective_balance`.
pub const COL_EB_BYTE_OFFSET: usize = COL_EFFECTIVE_BALANCE + 1;

/// Slack column for the legacy cap (`MAX_EFFECTIVE_BALANCE`).
pub const COL_SLACK_LEGACY: usize = COL_EB_BYTE_OFFSET + U64_BYTES;
/// 8-byte LE decomposition of `slack_legacy`.
pub const COL_SLACK_LEGACY_BYTE_OFFSET: usize = COL_SLACK_LEGACY + 1;

/// Slack column for the Electra cap (`MAX_EFFECTIVE_BALANCE_ELECTRA`).
pub const COL_SLACK_ELECTRA: usize = COL_SLACK_LEGACY_BYTE_OFFSET + U64_BYTES;
/// 8-byte LE decomposition of `slack_electra`.
pub const COL_SLACK_ELECTRA_BYTE_OFFSET: usize = COL_SLACK_ELECTRA + 1;

/// `is_compounding` flag (binary). Equal to 1 iff `prefix == 2`.
pub const COL_IS_COMPOUNDING: usize = COL_SLACK_ELECTRA_BYTE_OFFSET + U64_BYTES;
/// `is_real` selector (binary).
pub const COL_IS_REAL: usize = COL_IS_COMPOUNDING + 1;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 10;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Eip7251CompoundingRow {
    pub validator_index: u64,
    pub prefix: u8,
    pub effective_balance: u64,
    pub is_compounding: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Eip7251CompoundingWitness {
    pub rows: Vec<Eip7251CompoundingRow>,
}

impl Eip7251CompoundingWitness {
    pub fn from_rows(rows: Vec<Eip7251CompoundingRow>) -> Self {
        Self { rows }
    }
}

/// Host-side single-row builder.
///
/// Panics if:
/// * `prefix` is not in `{0, 1, 2}`.
/// * `prefix ∈ {0, 1}` and `effective_balance > MAX_EFFECTIVE_BALANCE`.
/// * `prefix == 2` and `effective_balance > MAX_EFFECTIVE_BALANCE_ELECTRA`.
pub fn from_validator(
    validator_index: u64,
    prefix: u8,
    effective_balance: u64,
) -> Eip7251CompoundingWitness {
    let is_compounding = match prefix {
        0 | 1 => false,
        2 => true,
        _ => panic!(
            "eip7251_compounding_air: invalid prefix byte 0x{:02x} \
             (expected 0x00, 0x01, or 0x02)",
            prefix,
        ),
    };
    let cap = if is_compounding {
        MAX_EFFECTIVE_BALANCE_ELECTRA
    } else {
        MAX_EFFECTIVE_BALANCE
    };
    assert!(
        effective_balance <= cap,
        "eip7251_compounding_air: effective_balance {} exceeds cap {} \
         for prefix 0x{:02x}",
        effective_balance,
        cap,
        prefix,
    );
    Eip7251CompoundingWitness {
        rows: vec![Eip7251CompoundingRow {
            validator_index,
            prefix,
            effective_balance,
            is_compounding,
            is_real: true,
        }],
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < 8);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

pub fn build_trace_polynomials(
    witness: &Eip7251CompoundingWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_VALIDATOR_INDEX][i] =
            Scalar::from_u64(row.validator_index, curve);
        let vi_bytes = row.validator_index.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_VI_BYTE_OFFSET + b][i] =
                Scalar::from_u64(vi_bytes[b] as u64, curve);
        }

        columns[COL_PREFIX][i] = Scalar::from_u64(row.prefix as u64, curve);
        columns[COL_EFFECTIVE_BALANCE][i] =
            Scalar::from_u64(row.effective_balance, curve);
        let eb_bytes = row.effective_balance.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_EB_BYTE_OFFSET + b][i] =
                Scalar::from_u64(eb_bytes[b] as u64, curve);
        }

        // Compute slack values. Both are valid (≤ 2^41) regardless of
        // which cap the prefix selects; the cap-bind constraints only
        // fire under the correct gate, but we populate both for the
        // LE-decomp bodies which are gated by IS_REAL.
        //
        // For non-compounding rows: slack_legacy = MAX - eb;
        //                            slack_electra = MAX_ELECTRA - eb.
        // For compounding rows the legacy slack body is unconstrained
        // (gated off by (1 - is_compounding)), so we still set
        // slack_legacy = MAX_ELECTRA - eb so its byte decomposition
        // stays in range — but we MUST also ensure the legacy body
        // vanishes under its own gate. Since (1 - is_compounding) = 0
        // on compounding rows the legacy body trivially vanishes
        // regardless of slack_legacy's actual value.
        let slack_legacy_val = if row.is_compounding {
            // The (1 - is_compounding) gate kills the legacy body; we
            // still need a byte-range-valid value. Use 0.
            0u64
        } else {
            MAX_EFFECTIVE_BALANCE
                .checked_sub(row.effective_balance)
                .expect("legacy slack underflow — honest host check failed")
        };
        let slack_electra_val = if row.is_compounding {
            MAX_EFFECTIVE_BALANCE_ELECTRA
                .checked_sub(row.effective_balance)
                .expect("electra slack underflow — honest host check failed")
        } else {
            // Gate (is_compounding) = 0 kills the body. Use 0.
            0u64
        };

        columns[COL_SLACK_LEGACY][i] = Scalar::from_u64(slack_legacy_val, curve);
        let sl_bytes = slack_legacy_val.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_SLACK_LEGACY_BYTE_OFFSET + b][i] =
                Scalar::from_u64(sl_bytes[b] as u64, curve);
        }
        columns[COL_SLACK_ELECTRA][i] = Scalar::from_u64(slack_electra_val, curve);
        let se_bytes = slack_electra_val.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_SLACK_ELECTRA_BYTE_OFFSET + b][i] =
                Scalar::from_u64(se_bytes[b] as u64, curve);
        }

        columns[COL_IS_COMPOUNDING][i] =
            if row.is_compounding { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] =
            if row.is_real { one.clone() } else { zero.clone() };
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

pub struct Eip7251CompoundingConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Eip7251CompoundingConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn sum_le_bytes(col_evals: &[Scalar], offset: usize, curve: CurveType) -> Scalar {
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[offset + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    sum
}

fn sum_le_bytes_poly(
    col_coeffs: &[Vec<Scalar>],
    offset: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for b in 0..U64_BYTES {
        let byte_poly = &col_coeffs[offset + b];
        let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
        sum = poly_add(&sum, &term, curve);
    }
    sum
}

impl VmConstraintSystem for Eip7251CompoundingConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_compounding_binary".into(),
            "compounding_prefix_pin".into(),
            "is_real_gates_compound".into(),
            "legacy_cap_bind".into(),
            "compounding_cap_bind".into(),
            "effective_balance_le_decomp".into(),
            "slack_legacy_le_decomp".into(),
            "slack_electra_le_decomp".into(),
            "validator_index_le_decomp".into(),
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
        let two = Scalar::from_u64(2, curve);
        let cap_legacy = Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve);
        let cap_electra = Scalar::from_u64(MAX_EFFECTIVE_BALANCE_ELECTRA, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_compounding = &row_evals[COL_IS_COMPOUNDING];
            let prefix = &row_evals[COL_PREFIX];
            let eb = &row_evals[COL_EFFECTIVE_BALANCE];
            let slack_l = &row_evals[COL_SLACK_LEGACY];
            let slack_e = &row_evals[COL_SLACK_ELECTRA];

            // 0: is_real_binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            // 1: is_compounding_binary.
            bodies[1][row] = is_compounding.mul(&is_compounding.sub(&one));
            // 2: compounding_prefix_pin: is_compounding · (prefix − 2) = 0.
            bodies[2][row] = is_compounding.mul(&prefix.sub(&two));
            // 3: is_real_gates_compound: is_compounding · (1 − is_real) = 0.
            bodies[3][row] = is_compounding.mul(&one.sub(is_real));
            // 4: legacy_cap_bind:
            //    (1 − is_compounding) · is_real · (CAP_LEGACY − eb − slack_l) = 0.
            let gate_legacy = one.sub(is_compounding).mul(is_real);
            let legacy_residual = cap_legacy.sub(eb).sub(slack_l);
            bodies[4][row] = gate_legacy.mul(&legacy_residual);
            // 5: compounding_cap_bind:
            //    is_compounding · (CAP_ELECTRA − eb − slack_e) = 0.
            let electra_residual = cap_electra.sub(eb).sub(slack_e);
            bodies[5][row] = is_compounding.mul(&electra_residual);
            // 6: effective_balance LE decomp (gated by is_real).
            let eb_sum = sum_le_bytes(&row_evals, COL_EB_BYTE_OFFSET, curve);
            bodies[6][row] = is_real.mul(&eb.sub(&eb_sum));
            // 7: slack_legacy LE decomp (gated by is_real).
            let sl_sum = sum_le_bytes(&row_evals, COL_SLACK_LEGACY_BYTE_OFFSET, curve);
            bodies[7][row] = is_real.mul(&slack_l.sub(&sl_sum));
            // 8: slack_electra LE decomp (gated by is_real).
            let se_sum = sum_le_bytes(&row_evals, COL_SLACK_ELECTRA_BYTE_OFFSET, curve);
            bodies[8][row] = is_real.mul(&slack_e.sub(&se_sum));
            // 9: validator_index LE decomp (unconditional, mirrors WC AIR).
            let vi_sum = sum_le_bytes(&row_evals, COL_VI_BYTE_OFFSET, curve);
            bodies[9][row] = row_evals[COL_VALIDATOR_INDEX].sub(&vi_sum);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(2, curve);
        let cap_legacy = Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve);
        let cap_electra = Scalar::from_u64(MAX_EFFECTIVE_BALANCE_ELECTRA, curve);

        let is_real = &col_evals[COL_IS_REAL];
        let is_compounding = &col_evals[COL_IS_COMPOUNDING];
        let prefix = &col_evals[COL_PREFIX];
        let eb = &col_evals[COL_EFFECTIVE_BALANCE];
        let slack_l = &col_evals[COL_SLACK_LEGACY];
        let slack_e = &col_evals[COL_SLACK_ELECTRA];

        let gate_legacy = one.sub(is_compounding).mul(is_real);
        let legacy_residual = cap_legacy.sub(eb).sub(slack_l);
        let electra_residual = cap_electra.sub(eb).sub(slack_e);

        let eb_sum = sum_le_bytes(col_evals, COL_EB_BYTE_OFFSET, curve);
        let sl_sum = sum_le_bytes(col_evals, COL_SLACK_LEGACY_BYTE_OFFSET, curve);
        let se_sum = sum_le_bytes(col_evals, COL_SLACK_ELECTRA_BYTE_OFFSET, curve);
        let vi_sum = sum_le_bytes(col_evals, COL_VI_BYTE_OFFSET, curve);

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_compounding.mul(&is_compounding.sub(&one)),
            is_compounding.mul(&prefix.sub(&two)),
            is_compounding.mul(&one.sub(is_real)),
            gate_legacy.mul(&legacy_residual),
            is_compounding.mul(&electra_residual),
            is_real.mul(&eb.sub(&eb_sum)),
            is_real.mul(&slack_l.sub(&sl_sum)),
            is_real.mul(&slack_e.sub(&se_sum)),
            col_evals[COL_VALIDATOR_INDEX].sub(&vi_sum),
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
        let two_poly = vec![Scalar::from_u64(2, curve)];
        let cap_legacy_poly = vec![Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve)];
        let cap_electra_poly =
            vec![Scalar::from_u64(MAX_EFFECTIVE_BALANCE_ELECTRA, curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_compounding = &col_coeffs[COL_IS_COMPOUNDING];
        let prefix = &col_coeffs[COL_PREFIX];
        let eb = &col_coeffs[COL_EFFECTIVE_BALANCE];
        let slack_l = &col_coeffs[COL_SLACK_LEGACY];
        let slack_e = &col_coeffs[COL_SLACK_ELECTRA];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real_binary.
        bodies.push(poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve));
        // 1: is_compounding_binary.
        bodies.push(poly_mul(
            is_compounding,
            &poly_sub(is_compounding, &one_poly, curve),
            curve,
        ));
        // 2: compounding_prefix_pin.
        bodies.push(poly_mul(
            is_compounding,
            &poly_sub(prefix, &two_poly, curve),
            curve,
        ));
        // 3: is_real_gates_compound.
        bodies.push(poly_mul(
            is_compounding,
            &poly_sub(&one_poly, is_real, curve),
            curve,
        ));
        // 4: legacy_cap_bind:
        //    (1 − is_compounding) · is_real · (CAP_LEGACY − eb − slack_l).
        let one_minus_comp = poly_sub(&one_poly, is_compounding, curve);
        let gate_legacy = poly_mul(&one_minus_comp, is_real, curve);
        let cap_minus_eb_l = poly_sub(&cap_legacy_poly, eb, curve);
        let legacy_residual = poly_sub(&cap_minus_eb_l, slack_l, curve);
        bodies.push(poly_mul(&gate_legacy, &legacy_residual, curve));
        // 5: compounding_cap_bind.
        let cap_minus_eb_e = poly_sub(&cap_electra_poly, eb, curve);
        let electra_residual = poly_sub(&cap_minus_eb_e, slack_e, curve);
        bodies.push(poly_mul(is_compounding, &electra_residual, curve));
        // 6: effective_balance LE decomp.
        let eb_sum = sum_le_bytes_poly(col_coeffs, COL_EB_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(eb, &eb_sum, curve), curve));
        // 7: slack_legacy LE decomp.
        let sl_sum = sum_le_bytes_poly(col_coeffs, COL_SLACK_LEGACY_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(slack_l, &sl_sum, curve), curve));
        // 8: slack_electra LE decomp.
        let se_sum =
            sum_le_bytes_poly(col_coeffs, COL_SLACK_ELECTRA_BYTE_OFFSET, curve);
        bodies.push(poly_mul(is_real, &poly_sub(slack_e, &se_sum, curve), curve));
        // 9: validator_index LE decomp.
        let vi_sum = sum_le_bytes_poly(col_coeffs, COL_VI_BYTE_OFFSET, curve);
        bodies.push(poly_sub(&col_coeffs[COL_VALIDATOR_INDEX], &vi_sum, curve));

        debug_assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);

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
        // All u64-byte columns are 8-bit range checked.
        let byte_ranges: [(usize, usize, &str); 4] = [
            (COL_VI_BYTE_OFFSET, U64_BYTES, "vi_byte"),
            (COL_EB_BYTE_OFFSET, U64_BYTES, "eb_byte"),
            (COL_SLACK_LEGACY_BYTE_OFFSET, U64_BYTES, "slack_legacy_byte"),
            (COL_SLACK_ELECTRA_BYTE_OFFSET, U64_BYTES, "slack_electra_byte"),
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
        // Prefix byte ∈ [0, 256).
        declarations.push((
            LookupDeclaration {
                label: "prefix_8bit".into(),
                column_index: COL_PREFIX,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(VALIDATOR_INDEX, PREFIX)` of this AIR against
/// [`crate::withdrawal_credential_air`]'s `(COL_VALIDATOR_INDEX,
/// COL_WC_BYTE_OFFSET[0])`. Gated by `IS_REAL` on both sides.
///
/// This pins the prefix discriminator on this AIR to the on-chain
/// `withdrawal_credentials[0]` for the same validator, so the
/// cap-selection logic cannot be honeypotted by faking the prefix.
pub fn make_compounding_to_withdrawal_credential_descriptor(
    compounding_layer_index: usize,
    wc_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::withdrawal_credential_air as wc;
    let a_columns = vec![COL_VALIDATOR_INDEX, COL_PREFIX];
    let b_columns = vec![wc::COL_VALIDATOR_INDEX, wc::COL_WC_BYTE_OFFSET];
    CrossAirLogUpDescriptor {
        label: "eip7251_compounding_to_wc_v1".into(),
        a_layer_index: compounding_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: wc_layer_index,
        b_columns,
        b_selector_column: Some(wc::COL_IS_REAL),
    }
}

/// Bind `(VALIDATOR_INDEX, EFFECTIVE_BALANCE)` of this AIR against
/// [`crate::validator_balances_air`]'s `(COL_VALIDATOR_INDEX,
/// COL_EFFECTIVE_BALANCE)`. Gated by `IS_REAL` on both sides.
///
/// This pins the effective-balance column on this AIR to the registry's
/// authoritative value, so the cap check is anchored to the real
/// `state.validators[i].effective_balance`.
pub fn make_compounding_to_validator_balances_descriptor(
    compounding_layer_index: usize,
    balances_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_balances_air as vb;
    let a_columns = vec![COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE];
    let b_columns = vec![vb::COL_VALIDATOR_INDEX, vb::COL_EFFECTIVE_BALANCE];
    CrossAirLogUpDescriptor {
        label: "eip7251_compounding_to_validator_balances_v1".into(),
        a_layer_index: compounding_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: balances_layer_index,
        b_columns,
        b_selector_column: Some(vb::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate_bodies(
        witness: &Eip7251CompoundingWitness,
        curve: CurveType,
    ) -> (TracePolynomials, Vec<Vec<Scalar>>) {
        let trace = build_trace_polynomials(witness, curve);
        let cs = Eip7251CompoundingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        (trace, bodies)
    }

    fn assert_all_vanish(bodies: &[Vec<Scalar>]) {
        for (k, body) in bodies.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} should vanish at row {} (got {:?})",
                    k,
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    #[test]
    fn legacy_32eth_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        // prefix 0x01, exactly at the legacy cap.
        let w = from_validator(7, 1, MAX_EFFECTIVE_BALANCE);
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
        assert_eq!(trace.columns[COL_IS_COMPOUNDING].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_PREFIX].evaluations[0].to_u64(), 1);
        // slack_legacy must be 0 at the cap.
        assert_eq!(trace.columns[COL_SLACK_LEGACY].evaluations[0].to_u64(), 0);
    }

    #[test]
    fn legacy_bls_prefix_under_cap_vanishes() {
        let curve = CurveType::Bls48581;
        // prefix 0x00, well below cap.
        let w = from_validator(42, 0, 16 * GWEI_PER_ETH);
        let (_trace, bodies) = evaluate_bodies(&w, curve);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn compounding_100eth_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        let w = from_validator(1_234, 2, 100 * GWEI_PER_ETH);
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_all_vanish(&bodies);
        assert_eq!(trace.columns[COL_IS_COMPOUNDING].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_PREFIX].evaluations[0].to_u64(), 2);
        let slack_e = trace.columns[COL_SLACK_ELECTRA].evaluations[0].to_u64();
        assert_eq!(slack_e, MAX_EFFECTIVE_BALANCE_ELECTRA - 100 * GWEI_PER_ETH);
    }

    #[test]
    fn compounding_boundary_2048eth_vanishes() {
        let curve = CurveType::Bls48581;
        let w = from_validator(
            999_999,
            2,
            MAX_EFFECTIVE_BALANCE_ELECTRA,
        );
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_all_vanish(&bodies);
        // slack_electra must be 0 at the boundary.
        assert_eq!(trace.columns[COL_SLACK_ELECTRA].evaluations[0].to_u64(), 0);
        // is_compounding = 1, prefix = 2.
        assert_eq!(trace.columns[COL_IS_COMPOUNDING].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn legacy_over_cap_tampered_balance_detected() {
        // Honest host would panic, so build the row manually.
        let curve = CurveType::Bls48581;
        // Prefix 0x01 (legacy), balance = 33 ETH (over the legacy cap).
        let row = Eip7251CompoundingRow {
            validator_index: 1,
            prefix: 1,
            effective_balance: 33 * GWEI_PER_ETH,
            is_compounding: false,
            is_real: true,
        };
        let mut w = Eip7251CompoundingWitness { rows: vec![row] };
        // Trace builder would panic via checked_sub; bypass by replacing
        // the row after the fact. Use a balance just at the cap, then
        // tamper the column.
        w.rows[0].effective_balance = MAX_EFFECTIVE_BALANCE;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Now tamper: bump effective_balance scalar above the cap. The
        // legacy_cap_bind body (constraint 4) becomes:
        //   1 · 1 · (CAP − (CAP + 1) − slack_l) = −1 − slack_l, which is
        // non-zero (and slack_l = 0 in the honest cap trace).
        cols[COL_EFFECTIVE_BALANCE][0] =
            Scalar::from_u64(MAX_EFFECTIVE_BALANCE + 1, curve);
        let cs = Eip7251CompoundingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[4][0].is_zero(),
            "legacy_cap_bind body should fire when eb > MAX_EFFECTIVE_BALANCE",
        );
    }

    #[test]
    fn compounding_over_cap_tampered_balance_detected() {
        let curve = CurveType::Bls48581;
        // Honest at the cap, then tamper the eb column.
        let w = from_validator(7, 2, MAX_EFFECTIVE_BALANCE_ELECTRA);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_EFFECTIVE_BALANCE][0] =
            Scalar::from_u64(MAX_EFFECTIVE_BALANCE_ELECTRA + 1, curve);
        let cs = Eip7251CompoundingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[5][0].is_zero(),
            "compounding_cap_bind body should fire when eb > MAX_EFFECTIVE_BALANCE_ELECTRA",
        );
    }

    #[test]
    fn compounding_with_wrong_prefix_detected() {
        let curve = CurveType::Bls48581;
        // Honest compounding row.
        let w = from_validator(7, 2, 100 * GWEI_PER_ETH);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper prefix from 2 → 1 while keeping is_compounding = 1.
        // The compounding_prefix_pin body (2) fires: 1 · (1 − 2) = −1.
        cols[COL_PREFIX][0] = Scalar::from_u64(1, curve);
        let cs = Eip7251CompoundingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "compounding_prefix_pin body should fire when prefix != 2",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_compounding_to_withdrawal_credential_descriptor(0, 1);
        assert_eq!(d1.label, "eip7251_compounding_to_wc_v1");
        assert_eq!(d1.a_columns, vec![COL_VALIDATOR_INDEX, COL_PREFIX]);
        assert_eq!(
            d1.b_columns,
            vec![
                crate::withdrawal_credential_air::COL_VALIDATOR_INDEX,
                crate::withdrawal_credential_air::COL_WC_BYTE_OFFSET,
            ],
        );
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d1.b_selector_column,
            Some(crate::withdrawal_credential_air::COL_IS_REAL),
        );
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);

        let d2 = make_compounding_to_validator_balances_descriptor(0, 2);
        assert_eq!(d2.label, "eip7251_compounding_to_validator_balances_v1");
        assert_eq!(
            d2.a_columns,
            vec![COL_VALIDATOR_INDEX, COL_EFFECTIVE_BALANCE],
        );
        assert_eq!(
            d2.b_columns,
            vec![
                crate::validator_balances_air::COL_VALIDATOR_INDEX,
                crate::validator_balances_air::COL_EFFECTIVE_BALANCE,
            ],
        );
        assert_eq!(d2.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d2.b_selector_column,
            Some(crate::validator_balances_air::COL_IS_REAL),
        );
    }

    #[test]
    fn column_layout_pinned() {
        // 1 (vi) + 8 (vi_bytes) + 1 (prefix) + 1 (eb) + 8 (eb_bytes)
        // + 1 (slack_l) + 8 (sl_bytes) + 1 (slack_e) + 8 (se_bytes)
        // + 1 (is_comp) + 1 (is_real) = 39.
        assert_eq!(NUM_COLUMNS, 39);
        assert_eq!(NUM_ROW_CONSTRAINTS, 10);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(MAX_EFFECTIVE_BALANCE, 32_000_000_000);
        assert_eq!(MAX_EFFECTIVE_BALANCE_ELECTRA, 2048_000_000_000);
    }

    #[test]
    fn evaluate_at_point_matches_for_honest() {
        let curve = CurveType::Bls48581;
        let w = from_validator(555, 2, 100 * GWEI_PER_ETH);
        let trace = build_trace_polynomials(&w, curve);
        let cs = Eip7251CompoundingConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(17, curve);
        let row0_evals: Vec<Scalar> =
            col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }

    #[test]
    #[should_panic(expected = "exceeds cap")]
    fn from_validator_panics_on_legacy_over_cap() {
        let _ = from_validator(0, 1, MAX_EFFECTIVE_BALANCE + 1);
    }

    #[test]
    #[should_panic(expected = "invalid prefix byte")]
    fn from_validator_panics_on_invalid_prefix() {
        let _ = from_validator(0, 5, 0);
    }
}
