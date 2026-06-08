//! Withdrawal-credential discriminator AIR.
//!
//! Proves a validator's 32-byte `withdrawal_credentials` field is one
//! of the three accepted Ethereum consensus-layer prefixes and, for the
//! execution-address shapes, that bytes `[12..32]` form the claimed
//! execution address.
//!
//! Accepted shapes (see [`crate::deposit::validate_withdrawal_credentials`]):
//!
//!   * `0x00` — BLS_WITHDRAWAL_PREFIX (legacy; credentials are
//!     `sha256(pubkey)` with the first byte forced to `0x00`).
//!   * `0x01` — ETH1_ADDRESS_WITHDRAWAL_PREFIX (`0x01 || 11 × 0x00 ||
//!     execution_address[20]`).
//!   * `0x02` — COMPOUNDING_WITHDRAWAL_PREFIX (EIP-7251; identical
//!     layout to `0x01` for byte purposes).
//!
//! ## What this AIR proves algebraically
//!
//! For each `is_real = 1` row, the constraints together force:
//!
//!   1. Exactly one of `is_bls`, `is_eth1`, `is_compounding` is `1`,
//!      and the chosen flag matches `withdrawal_credentials[0]`.
//!   2. For `is_eth1 + is_compounding = 1` rows, bytes `[1..12]` of the
//!      credentials are zero, and bytes `[12..32]` equal
//!      `execution_address[0..20]`.
//!   3. Every committed byte is in `[0, 256)`.
//!
//! ## Cross-AIR linkages
//!
//! * [`make_withdrawal_cred_to_validator_registry_descriptor`] binds
//!   `(validator_index, withdrawal_credentials[0..32])` to the
//!   per-validator HTR AIR (`crate::validator_htr_air`), which carries
//!   the registry's authoritative `(VALIDATOR_INDEX, WC_BYTE[0..32])`
//!   tuple. This transitively pins the prefix discriminator to the
//!   on-chain `Validator.withdrawal_credentials`.
//! * [`make_withdrawal_cred_to_address_descriptor`] binds the 20-byte
//!   `execution_address` to [`crate::address_keccak_air`]'s
//!   `COL_ADDRESS_BE_OFFSET[0..20]`, providing the bridge to the
//!   `address_trie_key = keccak256(address)` derivation. The selector
//!   `IS_EXEC` (= `is_eth1 + is_compounding`) gates the linkage so
//!   BLS-prefix rows do not require an address-keccak row.
//!
//! ## What this AIR does NOT prove (deferred)
//!
//!   * For BLS prefix: that the credentials equal `sha256(pubkey)` with
//!     the high byte forced to `0x00`. Closing this requires linking
//!     against [`crate::sha256_extract`] on `(pubkey, wc_tail)` and is
//!     a follow-up gadget.
//!   * That `IS_EXEC = 1` rows must produce a matching row in
//!     `address_keccak_air` (descriptor is one-direction). Reverse
//!     direction is unnecessary because the consumer (e.g. withdrawal
//!     payout chain) is the one that consumes both sides.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

/// Length of the `withdrawal_credentials` SSZ field (32 bytes).
pub const WC_BYTES: usize = 32;

/// Length of an Ethereum execution address (20 bytes).
pub const ADDRESS_BYTES: usize = 20;

/// Number of zero bytes at positions `[1..12]` for execution-address
/// shapes.
pub const ZERO_PAD_BYTES: usize = 11;

/// `BLS_WITHDRAWAL_PREFIX_BYTE`.
pub const BLS_PREFIX_BYTE: u8 = 0x00;

/// `ETH1_ADDRESS_WITHDRAWAL_PREFIX_BYTE`.
pub const ETH1_PREFIX_BYTE: u8 = 0x01;

/// `COMPOUNDING_WITHDRAWAL_PREFIX_BYTE` (EIP-7251).
pub const COMPOUNDING_PREFIX_BYTE: u8 = 0x02;

// ─── Column layout ────────────────────────────────────────────────────

/// 32-byte `withdrawal_credentials[0..32]`. Position 0 is the prefix
/// discriminator; positions `[1..12]` are zero on execution-address
/// shapes; positions `[12..32]` carry the execution address on
/// execution-address shapes.
pub const COL_WC_BYTE_OFFSET: usize = 0;

/// 20-byte `execution_address[0..20]`. Zero on BLS-prefix rows.
pub const COL_EXEC_ADDR_BYTE_OFFSET: usize = COL_WC_BYTE_OFFSET + WC_BYTES;

/// 8-byte LE decomposition of `validator_index` (range-checked 8-bit).
pub const COL_VI_BYTE_OFFSET: usize = COL_EXEC_ADDR_BYTE_OFFSET + ADDRESS_BYTES;

/// `u64` validator_index (this is the column bound by the
/// validator-registry descriptor).
pub const COL_VALIDATOR_INDEX: usize = COL_VI_BYTE_OFFSET + 8;

/// `is_bls` flag (binary).
pub const COL_IS_BLS: usize = COL_VALIDATOR_INDEX + 1;

/// `is_eth1` flag (binary).
pub const COL_IS_ETH1: usize = COL_IS_BLS + 1;

/// `is_compounding` flag (binary).
pub const COL_IS_COMPOUNDING: usize = COL_IS_ETH1 + 1;

/// `is_real` selector (binary). Equals `is_bls + is_eth1 + is_compounding`.
pub const COL_IS_REAL: usize = COL_IS_COMPOUNDING + 1;

/// `is_exec = is_eth1 + is_compounding` aux flag (binary). Used as
/// the linkage selector for the address descriptor.
pub const COL_IS_EXEC: usize = COL_IS_REAL + 1;

pub const NUM_COLUMNS: usize = COL_IS_EXEC + 1;

/// Total row-local algebraic constraints (see body labels).
///
/// 0: is_real_binary
/// 1: is_bls_binary
/// 2: is_eth1_binary
/// 3: is_compounding_binary
/// 4: sum_to_is_real             (is_bls + is_eth1 + is_compounding − is_real)
/// 5: is_exec_definition         (is_exec − is_eth1 − is_compounding)
/// 6: bls_prefix_byte            is_bls · WC[0]
/// 7: eth1_prefix_byte           is_eth1 · (WC[0] − 1)
/// 8: compounding_prefix_byte    is_compounding · (WC[0] − 2)
/// 9..(9+ZERO_PAD_BYTES):        is_exec · WC[1+i] for i in 0..11      (11 bodies)
/// (9+11)..(9+11+ADDRESS_BYTES): is_exec · (WC[12+i] − exec_addr[i])    (20 bodies)
/// (9+11+20): validator_index_le_decomp
///
/// Total: 9 + 11 + 20 + 1 = 41.
pub const NUM_ZERO_PAD_BODIES: usize = ZERO_PAD_BYTES;
pub const NUM_ADDR_BODIES: usize = ADDRESS_BYTES;
pub const NUM_ROW_CONSTRAINTS: usize =
    9 + NUM_ZERO_PAD_BODIES + NUM_ADDR_BODIES + 1;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct WithdrawalCredentialRow {
    pub withdrawal_credentials: [u8; WC_BYTES],
    pub execution_address: [u8; ADDRESS_BYTES],
    pub validator_index: u64,
    pub is_bls: bool,
    pub is_eth1: bool,
    pub is_compounding: bool,
}

#[derive(Clone, Debug, Default)]
pub struct WithdrawalCredentialWitness {
    pub rows: Vec<WithdrawalCredentialRow>,
}

impl WithdrawalCredentialWitness {
    pub fn from_rows(rows: Vec<WithdrawalCredentialRow>) -> Self {
        Self { rows }
    }
}

/// Build a single-row witness for one `(withdrawal_credentials,
/// validator_index)` pair.
///
/// The execution address is extracted as `creds[12..32]` for execution-
/// address shapes (`0x01`, `0x02`) and is left as `0u8; 20` for the BLS
/// shape (`0x00`).
///
/// Panics if `creds[0]` is not in the accepted set `{0x00, 0x01, 0x02}`.
pub fn from_credentials(
    creds: [u8; WC_BYTES],
    validator_index: u64,
) -> WithdrawalCredentialWitness {
    let prefix = creds[0];
    let (is_bls, is_eth1, is_compounding) = match prefix {
        BLS_PREFIX_BYTE => (true, false, false),
        ETH1_PREFIX_BYTE => (false, true, false),
        COMPOUNDING_PREFIX_BYTE => (false, false, true),
        _ => panic!(
            "withdrawal_credential_air: invalid prefix byte 0x{:02x} \
             (expected 0x00, 0x01, or 0x02)",
            prefix,
        ),
    };

    let mut execution_address = [0u8; ADDRESS_BYTES];
    if is_eth1 || is_compounding {
        // Honest-host check: bytes [1..12] must be zero.
        for (i, &b) in creds[1..1 + ZERO_PAD_BYTES].iter().enumerate() {
            assert!(
                b == 0,
                "withdrawal_credential_air: execution-address shape \
                 requires creds[{}] == 0 (got 0x{:02x})",
                1 + i,
                b,
            );
        }
        execution_address.copy_from_slice(&creds[12..32]);
    }

    WithdrawalCredentialWitness {
        rows: vec![WithdrawalCredentialRow {
            withdrawal_credentials: creds,
            execution_address,
            validator_index,
            is_bls,
            is_eth1,
            is_compounding,
        }],
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < 8);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

pub fn build_trace_polynomials(
    witness: &WithdrawalCredentialWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..WC_BYTES {
            columns[COL_WC_BYTE_OFFSET + k][i] =
                Scalar::from_u64(row.withdrawal_credentials[k] as u64, curve);
        }
        for k in 0..ADDRESS_BYTES {
            columns[COL_EXEC_ADDR_BYTE_OFFSET + k][i] =
                Scalar::from_u64(row.execution_address[k] as u64, curve);
        }
        let vi_bytes = row.validator_index.to_le_bytes();
        for b in 0..8 {
            columns[COL_VI_BYTE_OFFSET + b][i] =
                Scalar::from_u64(vi_bytes[b] as u64, curve);
        }
        columns[COL_VALIDATOR_INDEX][i] =
            Scalar::from_u64(row.validator_index, curve);
        columns[COL_IS_BLS][i] =
            if row.is_bls { one.clone() } else { zero.clone() };
        columns[COL_IS_ETH1][i] =
            if row.is_eth1 { one.clone() } else { zero.clone() };
        columns[COL_IS_COMPOUNDING][i] =
            if row.is_compounding { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = if row.is_bls || row.is_eth1 || row.is_compounding {
            one.clone()
        } else {
            zero.clone()
        };
        columns[COL_IS_EXEC][i] = if row.is_eth1 || row.is_compounding {
            one.clone()
        } else {
            zero.clone()
        };
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

pub struct WithdrawalCredentialConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl WithdrawalCredentialConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for WithdrawalCredentialConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "is_real_binary".to_string(),
            "is_bls_binary".to_string(),
            "is_eth1_binary".to_string(),
            "is_compounding_binary".to_string(),
            "sum_to_is_real".to_string(),
            "is_exec_definition".to_string(),
            "bls_prefix_byte".to_string(),
            "eth1_prefix_byte".to_string(),
            "compounding_prefix_byte".to_string(),
        ];
        for i in 0..NUM_ZERO_PAD_BODIES {
            labels.push(format!("exec_zero_pad_{}", i + 1));
        }
        for i in 0..NUM_ADDR_BODIES {
            labels.push(format!("exec_address_match_{}", i));
        }
        labels.push("validator_index_le_decomp".to_string());
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
        let two = Scalar::from_u64(2, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_bls = &row_evals[COL_IS_BLS];
            let is_eth1 = &row_evals[COL_IS_ETH1];
            let is_compounding = &row_evals[COL_IS_COMPOUNDING];
            let is_exec = &row_evals[COL_IS_EXEC];
            let wc0 = &row_evals[COL_WC_BYTE_OFFSET];

            // 0: is_real_binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            // 1: is_bls_binary.
            bodies[1][row] = is_bls.mul(&is_bls.sub(&one));
            // 2: is_eth1_binary.
            bodies[2][row] = is_eth1.mul(&is_eth1.sub(&one));
            // 3: is_compounding_binary.
            bodies[3][row] = is_compounding.mul(&is_compounding.sub(&one));
            // 4: sum_to_is_real:
            //    (is_bls + is_eth1 + is_compounding) − is_real = 0.
            bodies[4][row] =
                is_bls.add(is_eth1).add(is_compounding).sub(is_real);
            // 5: is_exec_definition: is_exec − (is_eth1 + is_compounding) = 0.
            bodies[5][row] = is_exec.sub(&is_eth1.add(is_compounding));
            // 6: bls_prefix_byte: is_bls · WC[0] = 0.
            bodies[6][row] = is_bls.mul(wc0);
            // 7: eth1_prefix_byte: is_eth1 · (WC[0] − 1) = 0.
            bodies[7][row] = is_eth1.mul(&wc0.sub(&one));
            // 8: compounding_prefix_byte: is_compounding · (WC[0] − 2) = 0.
            bodies[8][row] = is_compounding.mul(&wc0.sub(&two));

            // Zero-pad: is_exec · WC[1 + i] = 0 for i in 0..11.
            for i in 0..NUM_ZERO_PAD_BODIES {
                let wc_byte = &row_evals[COL_WC_BYTE_OFFSET + 1 + i];
                bodies[9 + i][row] = is_exec.mul(wc_byte);
            }

            // Address match: is_exec · (WC[12 + i] − exec_addr[i]) = 0.
            for i in 0..NUM_ADDR_BODIES {
                let wc_byte = &row_evals[COL_WC_BYTE_OFFSET + 12 + i];
                let addr_byte = &row_evals[COL_EXEC_ADDR_BYTE_OFFSET + i];
                bodies[9 + NUM_ZERO_PAD_BODIES + i][row] =
                    is_exec.mul(&wc_byte.sub(addr_byte));
            }

            // Validator-index LE decomp:
            //   VALIDATOR_INDEX − Σ_b VI_BYTE[b] · 2^(8b) = 0.
            let mut sum = Scalar::zero(curve);
            for b in 0..8 {
                let byte = &row_evals[COL_VI_BYTE_OFFSET + b];
                sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
            }
            let vi_idx = NUM_ROW_CONSTRAINTS - 1;
            bodies[vi_idx][row] = row_evals[COL_VALIDATOR_INDEX].sub(&sum);
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
        let is_real = &col_evals[COL_IS_REAL];
        let is_bls = &col_evals[COL_IS_BLS];
        let is_eth1 = &col_evals[COL_IS_ETH1];
        let is_compounding = &col_evals[COL_IS_COMPOUNDING];
        let is_exec = &col_evals[COL_IS_EXEC];
        let wc0 = &col_evals[COL_WC_BYTE_OFFSET];

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(is_bls.mul(&is_bls.sub(&one)));
        bodies.push(is_eth1.mul(&is_eth1.sub(&one)));
        bodies.push(is_compounding.mul(&is_compounding.sub(&one)));
        bodies.push(is_bls.add(is_eth1).add(is_compounding).sub(is_real));
        bodies.push(is_exec.sub(&is_eth1.add(is_compounding)));
        bodies.push(is_bls.mul(wc0));
        bodies.push(is_eth1.mul(&wc0.sub(&one)));
        bodies.push(is_compounding.mul(&wc0.sub(&two)));
        for i in 0..NUM_ZERO_PAD_BODIES {
            bodies.push(is_exec.mul(&col_evals[COL_WC_BYTE_OFFSET + 1 + i]));
        }
        for i in 0..NUM_ADDR_BODIES {
            let wc_byte = &col_evals[COL_WC_BYTE_OFFSET + 12 + i];
            let addr_byte = &col_evals[COL_EXEC_ADDR_BYTE_OFFSET + i];
            bodies.push(is_exec.mul(&wc_byte.sub(addr_byte)));
        }
        let mut sum = Scalar::zero(curve);
        for b in 0..8 {
            let byte = &col_evals[COL_VI_BYTE_OFFSET + b];
            sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
        }
        bodies.push(col_evals[COL_VALIDATOR_INDEX].sub(&sum));

        debug_assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);

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

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_bls = &col_coeffs[COL_IS_BLS];
        let is_eth1 = &col_coeffs[COL_IS_ETH1];
        let is_compounding = &col_coeffs[COL_IS_COMPOUNDING];
        let is_exec = &col_coeffs[COL_IS_EXEC];
        let wc0 = &col_coeffs[COL_WC_BYTE_OFFSET];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0..3: binary checks.
        bodies.push(poly_mul(is_real, &poly_sub(is_real, &one_poly, curve), curve));
        bodies.push(poly_mul(is_bls, &poly_sub(is_bls, &one_poly, curve), curve));
        bodies.push(poly_mul(is_eth1, &poly_sub(is_eth1, &one_poly, curve), curve));
        bodies.push(poly_mul(
            is_compounding,
            &poly_sub(is_compounding, &one_poly, curve),
            curve,
        ));

        // 4: sum_to_is_real.
        let sum_flags = poly_add(
            &poly_add(is_bls, is_eth1, curve),
            is_compounding,
            curve,
        );
        bodies.push(poly_sub(&sum_flags, is_real, curve));

        // 5: is_exec_definition.
        let sum_exec = poly_add(is_eth1, is_compounding, curve);
        bodies.push(poly_sub(is_exec, &sum_exec, curve));

        // 6: is_bls · WC[0].
        bodies.push(poly_mul(is_bls, wc0, curve));
        // 7: is_eth1 · (WC[0] − 1).
        bodies.push(poly_mul(is_eth1, &poly_sub(wc0, &one_poly, curve), curve));
        // 8: is_compounding · (WC[0] − 2).
        bodies.push(poly_mul(
            is_compounding,
            &poly_sub(wc0, &two_poly, curve),
            curve,
        ));

        // 9..(9+11): is_exec · WC[1 + i].
        for i in 0..NUM_ZERO_PAD_BODIES {
            let wc_byte = &col_coeffs[COL_WC_BYTE_OFFSET + 1 + i];
            bodies.push(poly_mul(is_exec, wc_byte, curve));
        }

        // (9+11)..(9+11+20): is_exec · (WC[12 + i] − exec_addr[i]).
        for i in 0..NUM_ADDR_BODIES {
            let wc_byte = &col_coeffs[COL_WC_BYTE_OFFSET + 12 + i];
            let addr_byte = &col_coeffs[COL_EXEC_ADDR_BYTE_OFFSET + i];
            let diff = poly_sub(wc_byte, addr_byte, curve);
            bodies.push(poly_mul(is_exec, &diff, curve));
        }

        // last: validator_index LE decomp.
        let mut sum = vec![Scalar::zero(curve)];
        for b in 0..8 {
            let byte_poly = &col_coeffs[COL_VI_BYTE_OFFSET + b];
            let term = poly_scalar_mul(byte_poly, &le_byte_pow(b, curve));
            sum = poly_add(&sum, &term, curve);
        }
        bodies.push(poly_sub(&col_coeffs[COL_VALIDATOR_INDEX], &sum, curve));

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

        // WC bytes (32) + execution_address bytes (20) + VI bytes (8).
        let byte_ranges: [(usize, usize, &str); 3] = [
            (COL_WC_BYTE_OFFSET, WC_BYTES, "wc"),
            (COL_EXEC_ADDR_BYTE_OFFSET, ADDRESS_BYTES, "exec_addr"),
            (COL_VI_BYTE_OFFSET, 8, "vi_byte"),
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

/// Bind `(VALIDATOR_INDEX, WC_BYTE[0..32])` of this AIR against
/// [`crate::validator_htr_air`]'s
/// `(COL_VALIDATOR_INDEX, COL_WC_BYTE_OFFSET[0..32])`. Composed with the
/// validator-registry inclusion AIR, this pins the discriminator bits
/// of this AIR to the on-chain `Validator.withdrawal_credentials` for
/// the claimed `validator_index`.
pub fn make_withdrawal_cred_to_validator_registry_descriptor(
    wc_layer_index: usize,
    validator_htr_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_htr_air as vh;
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + WC_BYTES);
    a_columns.push(COL_VALIDATOR_INDEX);
    for k in 0..WC_BYTES {
        a_columns.push(COL_WC_BYTE_OFFSET + k);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + WC_BYTES);
    b_columns.push(vh::COL_VALIDATOR_INDEX);
    for k in 0..WC_BYTES {
        b_columns.push(vh::COL_WC_BYTE_OFFSET + k);
    }

    CrossAirLogUpDescriptor {
        label: "withdrawal_cred_to_validator_registry_v1".into(),
        a_layer_index: wc_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_htr_layer_index,
        b_columns,
        b_selector_column: Some(vh::COL_IS_REAL),
    }
}

/// Bind `execution_address[0..20]` of this AIR against
/// [`crate::address_keccak_air`]'s `COL_ADDRESS_BE_OFFSET[0..20]`,
/// gated by `IS_EXEC = is_eth1 + is_compounding`. For BLS-prefix rows
/// the selector is zero so no corresponding row is required in the
/// address-keccak AIR.
pub fn make_withdrawal_cred_to_address_descriptor(
    wc_layer_index: usize,
    address_keccak_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::address_keccak_air as ak;
    let a_columns: Vec<usize> = (0..ADDRESS_BYTES)
        .map(|k| COL_EXEC_ADDR_BYTE_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = (0..ADDRESS_BYTES)
        .map(|k| ak::COL_ADDRESS_BE_OFFSET + k)
        .collect();

    CrossAirLogUpDescriptor {
        label: "withdrawal_cred_to_address_v1".into(),
        a_layer_index: wc_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_EXEC),
        b_layer_index: address_keccak_layer_index,
        b_columns,
        b_selector_column: Some(ak::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bls_creds() -> [u8; WC_BYTES] {
        // sha256(pubkey) with high byte forced to 0x00; we just stub
        // some non-zero tail since the BLS preimage check is deferred.
        let mut w = [0u8; WC_BYTES];
        for i in 1..WC_BYTES {
            w[i] = (i as u8).wrapping_mul(7).wrapping_add(1);
        }
        // w[0] already 0x00.
        w
    }

    fn eth1_creds() -> ([u8; WC_BYTES], [u8; ADDRESS_BYTES]) {
        let mut addr = [0u8; ADDRESS_BYTES];
        for i in 0..ADDRESS_BYTES {
            addr[i] = (i as u8).wrapping_mul(13).wrapping_add(0x40);
        }
        let mut w = [0u8; WC_BYTES];
        w[0] = ETH1_PREFIX_BYTE;
        // bytes [1..12] left zero.
        w[12..32].copy_from_slice(&addr);
        (w, addr)
    }

    fn compounding_creds() -> ([u8; WC_BYTES], [u8; ADDRESS_BYTES]) {
        let mut addr = [0u8; ADDRESS_BYTES];
        for i in 0..ADDRESS_BYTES {
            addr[i] = (i as u8).wrapping_mul(11).wrapping_add(0x10);
        }
        let mut w = [0u8; WC_BYTES];
        w[0] = COMPOUNDING_PREFIX_BYTE;
        w[12..32].copy_from_slice(&addr);
        (w, addr)
    }

    fn evaluate_bodies(
        witness: &WithdrawalCredentialWitness,
        curve: CurveType,
    ) -> (TracePolynomials, Vec<Vec<Scalar>>) {
        let trace = build_trace_polynomials(witness, curve);
        let cs = WithdrawalCredentialConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        (trace, bodies)
    }

    #[test]
    fn bls_prefix_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        let w = from_credentials(bls_creds(), 7);
        let (trace, bodies) = evaluate_bodies(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
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
        // Sanity: IS_BLS = 1, IS_EXEC = 0.
        assert_eq!(trace.columns[COL_IS_BLS].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_IS_EXEC].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_VALIDATOR_INDEX].evaluations[0].to_u64(), 7);
    }

    #[test]
    fn eth1_prefix_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        let (creds, addr) = eth1_creds();
        let w = from_credentials(creds, 999);
        let (trace, bodies) = evaluate_bodies(&w, curve);
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
        assert_eq!(trace.columns[COL_IS_ETH1].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_IS_EXEC].evaluations[0].to_u64(), 1);
        for i in 0..ADDRESS_BYTES {
            assert_eq!(
                trace.columns[COL_EXEC_ADDR_BYTE_OFFSET + i].evaluations[0].to_u64(),
                addr[i] as u64,
            );
        }
    }

    #[test]
    fn compounding_prefix_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        let (creds, addr) = compounding_creds();
        let w = from_credentials(creds, 1_234_567);
        let (trace, bodies) = evaluate_bodies(&w, curve);
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
        assert_eq!(trace.columns[COL_IS_COMPOUNDING].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_IS_EXEC].evaluations[0].to_u64(), 1);
        for i in 0..ADDRESS_BYTES {
            assert_eq!(
                trace.columns[COL_EXEC_ADDR_BYTE_OFFSET + i].evaluations[0].to_u64(),
                addr[i] as u64,
            );
        }
    }

    #[test]
    fn tampered_prefix_byte_fires_constraint() {
        let curve = CurveType::Bls48581;
        let (creds, _addr) = eth1_creds();
        let w = from_credentials(creds, 42);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Flip prefix byte to 0x05 (invalid). The eth1_prefix_byte body
        // (constraint 7) must fire because is_eth1 still = 1.
        cols[COL_WC_BYTE_OFFSET][0] = Scalar::from_u64(5, curve);
        let cs = WithdrawalCredentialConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[7][0].is_zero(),
            "eth1_prefix_byte body should fire when WC[0] ≠ 1",
        );
    }

    #[test]
    fn tampered_zero_bytes_fires_constraint() {
        let curve = CurveType::Bls48581;
        let (creds, _addr) = eth1_creds();
        let w = from_credentials(creds, 42);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Set WC[5] to non-zero. is_exec = 1 so zero-pad body at index
        // 9 + (5 - 1) = 13 must fire.
        cols[COL_WC_BYTE_OFFSET + 5][0] = Scalar::from_u64(0xAB, curve);
        let cs = WithdrawalCredentialConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let body_idx = 9 + (5 - 1); // WC[5] ↔ zero-pad index 4 (i.e. byte 1+4=5).
        assert!(
            !bodies[body_idx][0].is_zero(),
            "exec_zero_pad body should fire on non-zero WC[5]",
        );
    }

    #[test]
    fn tampered_execution_address_fires_constraint() {
        let curve = CurveType::Bls48581;
        let (creds, _addr) = compounding_creds();
        let w = from_credentials(creds, 42);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper the committed execution_address[3] so it no longer
        // equals WC[15]. The exec_address_match_3 body fires.
        let orig = cols[COL_EXEC_ADDR_BYTE_OFFSET + 3][0].to_u64();
        cols[COL_EXEC_ADDR_BYTE_OFFSET + 3][0] =
            Scalar::from_u64(orig.wrapping_add(1) & 0xFF, curve);
        let cs = WithdrawalCredentialConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let body_idx = 9 + NUM_ZERO_PAD_BODIES + 3;
        assert!(
            !bodies[body_idx][0].is_zero(),
            "exec_address_match body 3 should fire on tampered address byte",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_withdrawal_cred_to_validator_registry_descriptor(0, 1);
        assert_eq!(d1.label, "withdrawal_cred_to_validator_registry_v1");
        assert_eq!(d1.a_columns.len(), 1 + WC_BYTES);
        assert_eq!(d1.b_columns.len(), 1 + WC_BYTES);
        assert_eq!(d1.a_columns[0], COL_VALIDATOR_INDEX);
        for k in 0..WC_BYTES {
            assert_eq!(d1.a_columns[1 + k], COL_WC_BYTE_OFFSET + k);
        }
        assert_eq!(
            d1.b_columns[0],
            crate::validator_htr_air::COL_VALIDATOR_INDEX,
        );
        for k in 0..WC_BYTES {
            assert_eq!(
                d1.b_columns[1 + k],
                crate::validator_htr_air::COL_WC_BYTE_OFFSET + k,
            );
        }
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d1.b_selector_column,
            Some(crate::validator_htr_air::COL_IS_REAL),
        );

        let d2 = make_withdrawal_cred_to_address_descriptor(0, 2);
        assert_eq!(d2.label, "withdrawal_cred_to_address_v1");
        assert_eq!(d2.a_columns.len(), ADDRESS_BYTES);
        assert_eq!(d2.b_columns.len(), ADDRESS_BYTES);
        for k in 0..ADDRESS_BYTES {
            assert_eq!(d2.a_columns[k], COL_EXEC_ADDR_BYTE_OFFSET + k);
            assert_eq!(
                d2.b_columns[k],
                crate::address_keccak_air::COL_ADDRESS_BE_OFFSET + k,
            );
        }
        // Gated by IS_EXEC, not IS_REAL.
        assert_eq!(d2.a_selector_column, Some(COL_IS_EXEC));
        assert_eq!(
            d2.b_selector_column,
            Some(crate::address_keccak_air::COL_IS_REAL),
        );
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_WC_BYTE_OFFSET, 0);
        assert_eq!(COL_EXEC_ADDR_BYTE_OFFSET, 32);
        assert_eq!(COL_VI_BYTE_OFFSET, 32 + 20);
        assert_eq!(COL_VALIDATOR_INDEX, 32 + 20 + 8);
        assert_eq!(COL_IS_BLS, COL_VALIDATOR_INDEX + 1);
        assert_eq!(COL_IS_ETH1, COL_IS_BLS + 1);
        assert_eq!(COL_IS_COMPOUNDING, COL_IS_ETH1 + 1);
        assert_eq!(COL_IS_REAL, COL_IS_COMPOUNDING + 1);
        assert_eq!(COL_IS_EXEC, COL_IS_REAL + 1);
        assert_eq!(NUM_COLUMNS, COL_IS_EXEC + 1);
        // 32 (wc) + 20 (exec_addr) + 8 (vi_bytes) + 1 (validator_index)
        // + 3 (is_bls/eth1/compounding) + 1 (is_real) + 1 (is_exec) = 66.
        assert_eq!(NUM_COLUMNS, 66);
        // 9 + 11 + 20 + 1 = 41.
        assert_eq!(NUM_ROW_CONSTRAINTS, 41);
        assert_eq!(NUM_SHIFTED, 0);
    }

    #[test]
    fn byte_range_lookup_coverage() {
        let cs = WithdrawalCredentialConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        let expected = WC_BYTES + ADDRESS_BYTES + 8;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }

    #[test]
    fn evaluate_at_point_matches_for_honest() {
        let curve = CurveType::Bls48581;
        let (creds, _addr) = eth1_creds();
        let w = from_credentials(creds, 555);
        let trace = build_trace_polynomials(&w, curve);
        let cs = WithdrawalCredentialConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(17, curve);
        let row0_evals: Vec<Scalar> =
            col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }

    #[test]
    #[should_panic(expected = "invalid prefix byte")]
    fn invalid_prefix_panics_in_host() {
        let mut creds = [0u8; WC_BYTES];
        creds[0] = 0xFF;
        let _ = from_credentials(creds, 0);
    }

    #[test]
    fn validator_index_le_decomp_fires_on_tamper() {
        let curve = CurveType::Bls48581;
        let w = from_credentials(bls_creds(), 0xDEAD_BEEF);
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let bumped = cols[COL_VI_BYTE_OFFSET][0].to_u64().wrapping_add(1);
        cols[COL_VI_BYTE_OFFSET][0] = Scalar::from_u64(bumped, curve);
        let cs = WithdrawalCredentialConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let vi_idx = NUM_ROW_CONSTRAINTS - 1;
        assert!(
            !bodies[vi_idx][0].is_zero(),
            "validator_index_le_decomp body should fire on tampered VI byte",
        );
    }
}
