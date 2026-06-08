//! Transaction nonce binding AIR.
//!
//! Per row, commits one (sender, transaction) nonce binding:
//!
//!   1. The transaction's nonce field equals the sender's account
//!      nonce in the **pre-execution** world state:
//!      `tx_nonce == account[sender].nonce`.
//!   2. After successful execution, the sender's nonce in the
//!      **post-execution** world state has incremented by one:
//!      `post_account_nonce == pre_account_nonce + 1`.
//!
//! This AIR is the algebraic glue between the per-transaction RLP
//! gadget ([`crate::tx_rlp_air`]) and the world-state account gadget
//! ([`crate::account_state_air`]). It does **not** re-derive the
//! sender (signature recovery lives in [`crate::secp256k1_recovery`])
//! and it does **not** re-derive the post-state account record (that
//! lives downstream in the world-state transition chain).
//!
//! Instead, three cross-AIR LogUp descriptors bind:
//!
//!   - [`make_tx_nonce_to_tx_rlp_descriptor`] —
//!     `(tx_nonce)` ↔ `(tx_rlp_air::COL_NONCE)`.
//!   - [`make_tx_nonce_to_account_pre_descriptor`] —
//!     `(sender_address_limbs, pre_account_nonce)` ↔
//!     `(account_state_air::COL_ADDR_L*, account_state_air::COL_NONCE)`.
//!   - [`make_tx_nonce_to_account_post_descriptor`] —
//!     `(sender_address_limbs, post_account_nonce)` ↔
//!     `(account_state_air::COL_ADDR_L*, account_state_air::COL_NONCE_POST)`.
//!     Targets the dedicated post-state column on `account_state_air`,
//!     which is pinned to `pre + 1` by an in-AIR increment constraint.
//!     A single account_state row per sender now provides both the
//!     pre and post bindings.
//!
//! ## Algebraic constraints (row-local, 8 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`.
//! 1. `nonce_matches_pre` — `IS_REAL · (TX_NONCE − PRE_NONCE) = 0`.
//! 2. `post_increments_pre` — `IS_REAL · (POST_NONCE − PRE_NONCE − 1) = 0`.
//! 3. `tx_nonce_le_decomp` — `TX_NONCE = Σ_b TX_NONCE_BYTE[b] · 2^(8b)`.
//! 4. `pre_nonce_le_decomp` — `PRE_NONCE = Σ_b PRE_NONCE_BYTE[b] · 2^(8b)`.
//! 5. `post_nonce_le_decomp` — `POST_NONCE = Σ_b POST_NONCE_BYTE[b] · 2^(8b)`.
//! 6. `sender_address_high_bytes_zero` — `IS_REAL · (sender_addr_byte[20] + ...) = 0`
//!    soft-binding for the unused upper 12 bytes of the 32-byte address
//!    field (today no upper bytes are committed — kept as a placeholder
//!    body, evaluates to zero unconditionally).
//! 7. `is_real_tx_index_consistency` — placeholder body that pins
//!    `TX_INDEX · 0 = 0` (trivially zero, slot reserved for the future
//!    in-row tx-index binding once the tx_rlp_air gains a tx_index
//!    column).
//!
//! Per-byte 8-bit range checks (via `lookup_declarations`) on:
//!   - `sender_address[0..20]`
//!   - `tx_nonce_le_bytes[0..8]`, `pre_nonce_le_bytes[0..8]`,
//!     `post_nonce_le_bytes[0..8]`

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

pub const ADDR_BYTES: usize = 20;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

// 20-byte sender address (one byte per column).
pub const COL_SENDER_OFFSET: usize = 0;

// u64 scalar columns.
pub const COL_TX_NONCE: usize = COL_SENDER_OFFSET + ADDR_BYTES;          // 20
pub const COL_PRE_ACCOUNT_NONCE: usize = COL_TX_NONCE + 1;               // 21
pub const COL_POST_ACCOUNT_NONCE: usize = COL_PRE_ACCOUNT_NONCE + 1;     // 22
pub const COL_TX_INDEX: usize = COL_POST_ACCOUNT_NONCE + 1;              // 23

// LE byte decompositions for the three nonce columns.
pub const COL_TX_NONCE_BYTE_OFFSET: usize = COL_TX_INDEX + 1;            // 24
pub const COL_PRE_NONCE_BYTE_OFFSET: usize = COL_TX_NONCE_BYTE_OFFSET + U64_BYTES;   // 32
pub const COL_POST_NONCE_BYTE_OFFSET: usize = COL_PRE_NONCE_BYTE_OFFSET + U64_BYTES; // 40

// Address as 4 LE u64 limbs (for the account_state cross-AIR linkage,
// which uses the same limb shape as `account_state_air::COL_ADDR_L*`).
pub const COL_ADDR_L0: usize = COL_POST_NONCE_BYTE_OFFSET + U64_BYTES;   // 48
pub const COL_ADDR_L1: usize = COL_ADDR_L0 + 1;                          // 49
pub const COL_ADDR_L2: usize = COL_ADDR_L1 + 1;                          // 50
pub const COL_ADDR_L3: usize = COL_ADDR_L2 + 1;                          // 51

pub const COL_IS_REAL: usize = COL_ADDR_L3 + 1;                          // 52
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;                          // 53

// 8 row-local bodies (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 8;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct TxNonceRow {
    pub sender_address: [u8; ADDR_BYTES],
    pub tx_nonce: u64,
    pub pre_account_nonce: u64,
    pub post_account_nonce: u64,
    pub tx_index: u64,
}

#[derive(Clone, Debug, Default)]
pub struct TxNonceWitness {
    pub rows: Vec<TxNonceRow>,
}

impl TxNonceWitness {
    pub fn from_rows(rows: Vec<TxNonceRow>) -> Self {
        Self { rows }
    }

    /// Build an honest single-binding witness.
    ///
    /// Asserts `tx_nonce == pre`; sets `post = pre + 1` host-side.
    /// Panics on `u64` overflow of `pre + 1`.
    pub fn from_inputs(
        sender: [u8; ADDR_BYTES],
        tx_nonce: u64,
        pre: u64,
        tx_index: u64,
    ) -> Self {
        assert_eq!(
            tx_nonce, pre,
            "tx_nonce ({}) must equal pre-account nonce ({})",
            tx_nonce, pre,
        );
        let post = pre
            .checked_add(1)
            .expect("post nonce overflowed u64 (pre = u64::MAX)");
        Self {
            rows: vec![TxNonceRow {
                sender_address: sender,
                tx_nonce,
                pre_account_nonce: pre,
                post_account_nonce: post,
                tx_index,
            }],
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn le_byte_pow(b: usize, curve: CurveType) -> Scalar {
    debug_assert!(b < U64_BYTES);
    Scalar::from_u64(1u64 << (8 * b), curve)
}

/// Pack a 20-byte BE address into 4 LE u64 limbs the same way
/// `account_state_air` does (top 12 bytes zero-padded).
fn address_to_limbs(address: [u8; ADDR_BYTES]) -> [u64; 4] {
    let mut full = [0u8; 32];
    full[12..32].copy_from_slice(&address);
    [
        u64::from_be_bytes([
            full[24], full[25], full[26], full[27],
            full[28], full[29], full[30], full[31],
        ]),
        u64::from_be_bytes([
            full[16], full[17], full[18], full[19],
            full[20], full[21], full[22], full[23],
        ]),
        u64::from_be_bytes([
            full[8], full[9], full[10], full[11],
            full[12], full[13], full[14], full[15],
        ]),
        u64::from_be_bytes([
            full[0], full[1], full[2], full[3],
            full[4], full[5], full[6], full[7],
        ]),
    ]
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &TxNonceWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..ADDR_BYTES {
            columns[COL_SENDER_OFFSET + k][i] =
                Scalar::from_u64(row.sender_address[k] as u64, curve);
        }
        columns[COL_TX_NONCE][i] = Scalar::from_u64(row.tx_nonce, curve);
        columns[COL_PRE_ACCOUNT_NONCE][i] =
            Scalar::from_u64(row.pre_account_nonce, curve);
        columns[COL_POST_ACCOUNT_NONCE][i] =
            Scalar::from_u64(row.post_account_nonce, curve);
        columns[COL_TX_INDEX][i] = Scalar::from_u64(row.tx_index, curve);

        let tx_bytes = row.tx_nonce.to_le_bytes();
        let pre_bytes = row.pre_account_nonce.to_le_bytes();
        let post_bytes = row.post_account_nonce.to_le_bytes();
        for b in 0..U64_BYTES {
            columns[COL_TX_NONCE_BYTE_OFFSET + b][i] =
                Scalar::from_u64(tx_bytes[b] as u64, curve);
            columns[COL_PRE_NONCE_BYTE_OFFSET + b][i] =
                Scalar::from_u64(pre_bytes[b] as u64, curve);
            columns[COL_POST_NONCE_BYTE_OFFSET + b][i] =
                Scalar::from_u64(post_bytes[b] as u64, curve);
        }

        let limbs = address_to_limbs(row.sender_address);
        columns[COL_ADDR_L0][i] = Scalar::from_u64(limbs[0], curve);
        columns[COL_ADDR_L1][i] = Scalar::from_u64(limbs[1], curve);
        columns[COL_ADDR_L2][i] = Scalar::from_u64(limbs[2], curve);
        columns[COL_ADDR_L3][i] = Scalar::from_u64(limbs[3], curve);

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

pub struct TxNonceConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl TxNonceConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Body: `target_value − Σ_b byte_col[b] · 2^(8b)`.
fn eval_le_decomp(
    target_value: &Scalar,
    byte_off: usize,
    col_evals: &[Scalar],
) -> Scalar {
    let curve = target_value.curve_type();
    let mut sum = Scalar::zero(curve);
    for b in 0..U64_BYTES {
        let byte = &col_evals[byte_off + b];
        sum = sum.add(&byte.mul(&le_byte_pow(b, curve)));
    }
    target_value.sub(&sum)
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

impl VmConstraintSystem for TxNonceConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "nonce_matches_pre".into(),
            "post_increments_pre".into(),
            "tx_nonce_le_decomp".into(),
            "pre_nonce_le_decomp".into(),
            "post_nonce_le_decomp".into(),
            "sender_address_high_bytes_zero".into(),
            "is_real_tx_index_consistency".into(),
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

            // 0: is_real_binary.
            bodies[0][row] = is_real.mul(&is_real.sub(&one));

            // 1: is_real · (TX_NONCE − PRE_NONCE).
            let diff_pre = row_evals[COL_TX_NONCE]
                .sub(&row_evals[COL_PRE_ACCOUNT_NONCE]);
            bodies[1][row] = is_real.mul(&diff_pre);

            // 2: is_real · (POST_NONCE − PRE_NONCE − 1).
            let diff_post = row_evals[COL_POST_ACCOUNT_NONCE]
                .sub(&row_evals[COL_PRE_ACCOUNT_NONCE])
                .sub(&one);
            bodies[2][row] = is_real.mul(&diff_post);

            // 3..5: LE decomps.
            bodies[3][row] = eval_le_decomp(
                &row_evals[COL_TX_NONCE],
                COL_TX_NONCE_BYTE_OFFSET,
                &row_evals,
            );
            bodies[4][row] = eval_le_decomp(
                &row_evals[COL_PRE_ACCOUNT_NONCE],
                COL_PRE_NONCE_BYTE_OFFSET,
                &row_evals,
            );
            bodies[5][row] = eval_le_decomp(
                &row_evals[COL_POST_ACCOUNT_NONCE],
                COL_POST_NONCE_BYTE_OFFSET,
                &row_evals,
            );

            // 6: placeholder (no upper-byte columns committed).
            bodies[6][row] = Scalar::zero(curve);

            // 7: placeholder (tx_index reserved for future binding).
            bodies[7][row] = Scalar::zero(curve);
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
            is_real.mul(
                &col_evals[COL_TX_NONCE].sub(&col_evals[COL_PRE_ACCOUNT_NONCE]),
            ),
            is_real.mul(
                &col_evals[COL_POST_ACCOUNT_NONCE]
                    .sub(&col_evals[COL_PRE_ACCOUNT_NONCE])
                    .sub(&one),
            ),
            eval_le_decomp(&col_evals[COL_TX_NONCE], COL_TX_NONCE_BYTE_OFFSET, col_evals),
            eval_le_decomp(
                &col_evals[COL_PRE_ACCOUNT_NONCE],
                COL_PRE_NONCE_BYTE_OFFSET,
                col_evals,
            ),
            eval_le_decomp(
                &col_evals[COL_POST_ACCOUNT_NONCE],
                COL_POST_NONCE_BYTE_OFFSET,
                col_evals,
            ),
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

        // body 1: is_real · (TX_NONCE − PRE_NONCE).
        let diff_pre = poly_sub(
            &col_coeffs[COL_TX_NONCE],
            &col_coeffs[COL_PRE_ACCOUNT_NONCE],
            curve,
        );
        let body1 = poly_mul(is_real, &diff_pre, curve);

        // body 2: is_real · (POST_NONCE − PRE_NONCE − 1).
        let diff_post1 = poly_sub(
            &col_coeffs[COL_POST_ACCOUNT_NONCE],
            &col_coeffs[COL_PRE_ACCOUNT_NONCE],
            curve,
        );
        let diff_post = poly_sub(&diff_post1, &one_poly, curve);
        let body2 = poly_mul(is_real, &diff_post, curve);

        // bodies 3..5: LE decomps.
        let tx_decomp = build_le_decomp_poly(
            &col_coeffs[COL_TX_NONCE],
            COL_TX_NONCE_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let pre_decomp = build_le_decomp_poly(
            &col_coeffs[COL_PRE_ACCOUNT_NONCE],
            COL_PRE_NONCE_BYTE_OFFSET,
            col_coeffs,
            curve,
        );
        let post_decomp = build_le_decomp_poly(
            &col_coeffs[COL_POST_ACCOUNT_NONCE],
            COL_POST_NONCE_BYTE_OFFSET,
            col_coeffs,
            curve,
        );

        // bodies 6, 7: zero placeholders.
        let zero_poly_a = vec![Scalar::zero(curve)];
        let zero_poly_b = vec![Scalar::zero(curve)];

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            body1,
            body2,
            tx_decomp,
            pre_decomp,
            post_decomp,
            zero_poly_a,
            zero_poly_b,
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

        // Sender address byte range checks.
        for k in 0..ADDR_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sender_byte_{}_8bit", k),
                    column_index: COL_SENDER_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }

        // u64 LE byte-decomp range checks.
        let u64_ranges: [(usize, &str); 3] = [
            (COL_TX_NONCE_BYTE_OFFSET, "tx_nonce_byte"),
            (COL_PRE_NONCE_BYTE_OFFSET, "pre_nonce_byte"),
            (COL_POST_NONCE_BYTE_OFFSET, "post_nonce_byte"),
        ];
        for (off, label) in u64_ranges {
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
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(TX_NONCE)` of this AIR against
/// [`crate::tx_rlp_air::COL_NONCE`]. Forces the committed transaction
/// nonce to match the per-transaction RLP gadget's nonce field.
pub fn make_tx_nonce_to_tx_rlp_descriptor(
    nonce_layer_index: usize,
    tx_rlp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::tx_rlp_air as tr;
    CrossAirLogUpDescriptor {
        label: "tx_nonce_to_tx_rlp_v1".into(),
        a_layer_index: nonce_layer_index,
        a_columns: vec![COL_TX_NONCE],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: tx_rlp_layer_index,
        b_columns: vec![tr::COL_NONCE],
        b_selector_column: Some(tr::COL_IS_REAL),
    }
}

/// Bind `(sender_address_limbs, PRE_ACCOUNT_NONCE)` of this AIR
/// against [`crate::account_state_air`] on
/// `(COL_ADDR_L0..L3, COL_NONCE)`. Forces the committed pre-state
/// nonce to match a row in the account_state gadget for the same
/// sender. The host-side convention is: this descriptor matches the
/// **pre-execution** account_state row for the sender.
pub fn make_tx_nonce_to_account_pre_descriptor(
    nonce_layer_index: usize,
    account_state_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::account_state_air as as_air;
    CrossAirLogUpDescriptor {
        label: "tx_nonce_to_account_pre_v1".into(),
        a_layer_index: nonce_layer_index,
        a_columns: vec![COL_ADDR_L0, COL_ADDR_L1, COL_ADDR_L2, COL_ADDR_L3, COL_PRE_ACCOUNT_NONCE],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: account_state_layer_index,
        b_columns: vec![
            as_air::COL_ADDR_L0,
            as_air::COL_ADDR_L1,
            as_air::COL_ADDR_L2,
            as_air::COL_ADDR_L3,
            as_air::COL_NONCE,
        ],
        b_selector_column: Some(as_air::COL_IS_REAL),
    }
}

/// Bind `(sender_address_limbs, POST_ACCOUNT_NONCE)` of this AIR
/// against [`crate::account_state_air`] on
/// `(COL_ADDR_L0..L3, COL_NONCE_POST)`.
///
/// Closes the prior soundness gap: previously this descriptor targeted
/// the same `account_state_air::COL_NONCE` column as the pre
/// descriptor, so the cross-AIR LogUp could not algebraically
/// distinguish pre- from post-state. With the new
/// `account_state_air::COL_NONCE_POST` column (and the in-AIR
/// `is_real · (nonce_post − nonce − 1) = 0` increment constraint),
/// this descriptor now binds the post-state nonce to a column that is
/// algebraically pinned as `pre + 1`.
pub fn make_tx_nonce_to_account_post_descriptor(
    nonce_layer_index: usize,
    account_state_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::account_state_air as as_air;
    CrossAirLogUpDescriptor {
        label: "tx_nonce_to_account_post_v1".into(),
        a_layer_index: nonce_layer_index,
        a_columns: vec![COL_ADDR_L0, COL_ADDR_L1, COL_ADDR_L2, COL_ADDR_L3, COL_POST_ACCOUNT_NONCE],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: account_state_layer_index,
        b_columns: vec![
            as_air::COL_ADDR_L0,
            as_air::COL_ADDR_L1,
            as_air::COL_ADDR_L2,
            as_air::COL_ADDR_L3,
            as_air::COL_NONCE_POST,
        ],
        b_selector_column: Some(as_air::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sender() -> [u8; ADDR_BYTES] {
        let mut a = [0u8; ADDR_BYTES];
        for i in 0..ADDR_BYTES {
            a[i] = (i as u8).wrapping_mul(7).wrapping_add(11);
        }
        a
    }

    fn honest_witness() -> TxNonceWitness {
        TxNonceWitness::from_inputs(sample_sender(), 42, 42, 0)
    }

    #[test]
    fn honest_witness_all_constraints_vanish() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);

        let cs = TxNonceConstraintSystem::new(trace.num_rows);
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

        // Sanity: post = pre + 1.
        assert_eq!(w.rows[0].post_account_nonce, w.rows[0].pre_account_nonce + 1);
    }

    #[test]
    fn tampered_pre_nonce_fires_nonce_matches_pre() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump PRE_NONCE; constraint 1 must fire (TX_NONCE != PRE_NONCE).
        cols[COL_PRE_ACCOUNT_NONCE][0] = Scalar::from_u64(99, curve);
        let cs = TxNonceConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[1][0].is_zero(),
            "nonce_matches_pre should fire on tampered PRE_NONCE",
        );
    }

    #[test]
    fn tampered_post_nonce_fires_post_increments_pre() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump POST_NONCE; constraint 2 must fire (POST != PRE + 1).
        let bumped = cols[COL_POST_ACCOUNT_NONCE][0].to_u64().wrapping_add(1);
        cols[COL_POST_ACCOUNT_NONCE][0] = Scalar::from_u64(bumped, curve);
        let cs = TxNonceConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[2][0].is_zero(),
            "post_increments_pre should fire on tampered POST_NONCE",
        );
    }

    #[test]
    fn tampered_tx_nonce_fires_match_and_decomp() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump TX_NONCE; both constraint 1 (TX != PRE) and constraint 3
        // (TX != Σ bytes) must fire.
        let bumped = cols[COL_TX_NONCE][0].to_u64().wrapping_add(7);
        cols[COL_TX_NONCE][0] = Scalar::from_u64(bumped, curve);
        let cs = TxNonceConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[1][0].is_zero(),
            "nonce_matches_pre should fire on tampered TX_NONCE",
        );
        assert!(
            !bodies[3][0].is_zero(),
            "tx_nonce_le_decomp should fire on tampered TX_NONCE",
        );
    }

    #[test]
    #[should_panic(expected = "tx_nonce")]
    fn from_inputs_panics_on_mismatch() {
        let _ = TxNonceWitness::from_inputs(sample_sender(), 10, 9, 0);
    }

    #[test]
    fn is_real_binary_fires_on_non_binary() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(2, curve);
        let cs = TxNonceConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!bodies[0][0].is_zero(), "is_real_binary should fire");
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_tx_nonce_to_tx_rlp_descriptor(0, 1);
        assert_eq!(d1.label, "tx_nonce_to_tx_rlp_v1");
        assert_eq!(d1.a_columns, vec![COL_TX_NONCE]);
        assert_eq!(d1.b_columns, vec![crate::tx_rlp_air::COL_NONCE]);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(crate::tx_rlp_air::COL_IS_REAL));

        let d2 = make_tx_nonce_to_account_pre_descriptor(0, 2);
        assert_eq!(d2.label, "tx_nonce_to_account_pre_v1");
        // 4 address limbs + 1 nonce = 5 cols.
        assert_eq!(d2.a_columns.len(), 5);
        assert_eq!(d2.b_columns.len(), 5);
        assert_eq!(d2.a_columns[4], COL_PRE_ACCOUNT_NONCE);
        assert_eq!(d2.b_columns[4], crate::account_state_air::COL_NONCE);
        assert_eq!(d2.b_columns[0], crate::account_state_air::COL_ADDR_L0);

        let d3 = make_tx_nonce_to_account_post_descriptor(0, 2);
        assert_eq!(d3.label, "tx_nonce_to_account_post_v1");
        assert_eq!(d3.a_columns.len(), 5);
        assert_eq!(d3.b_columns.len(), 5);
        assert_eq!(d3.a_columns[4], COL_POST_ACCOUNT_NONCE);
        // Post descriptor now targets the dedicated COL_NONCE_POST
        // column on account_state_air, which is pinned to `pre + 1`
        // by an in-AIR increment constraint.
        assert_eq!(d3.b_columns[4], crate::account_state_air::COL_NONCE_POST);
        // Pre & post target distinct b-side columns now.
        assert_ne!(d2.b_columns[4], d3.b_columns[4]);
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_SENDER_OFFSET, 0);
        assert_eq!(COL_TX_NONCE, 20);
        assert_eq!(COL_PRE_ACCOUNT_NONCE, 21);
        assert_eq!(COL_POST_ACCOUNT_NONCE, 22);
        assert_eq!(COL_TX_INDEX, 23);
        assert_eq!(COL_TX_NONCE_BYTE_OFFSET, 24);
        assert_eq!(COL_PRE_NONCE_BYTE_OFFSET, 32);
        assert_eq!(COL_POST_NONCE_BYTE_OFFSET, 40);
        assert_eq!(COL_ADDR_L0, 48);
        assert_eq!(COL_ADDR_L1, 49);
        assert_eq!(COL_ADDR_L2, 50);
        assert_eq!(COL_ADDR_L3, 51);
        assert_eq!(COL_IS_REAL, 52);
        assert_eq!(NUM_COLUMNS, 53);
        assert_eq!(NUM_ROW_CONSTRAINTS, 8);
        assert_eq!(NUM_SHIFTED, 0);
    }

    #[test]
    fn byte_range_lookup_coverage() {
        let cs = TxNonceConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // 20 sender bytes + 3 * 8 nonce bytes = 44 decls.
        let expected = ADDR_BYTES + 3 * U64_BYTES;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }

    #[test]
    fn evaluate_at_point_matches_domain_for_honest() {
        let curve = CurveType::Bls48581;
        let w = honest_witness();
        let trace = build_trace_polynomials(&w, curve);
        let cs = TxNonceConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(7, curve);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let row0_evals: Vec<Scalar> =
            col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must be zero on honest row");
    }

    #[test]
    fn address_limbs_match_account_state_convention() {
        // Pin: the limb layout matches account_state_air's
        // address_to_limbs convention (BE 20-byte zero-padded → 4
        // LE u64 limbs from low to high). Sanity-check on [0xab; 20].
        let curve = CurveType::Bls48581;
        let sender = [0xab; ADDR_BYTES];
        let w = TxNonceWitness::from_inputs(sender, 0, 0, 0);
        let trace = build_trace_polynomials(&w, curve);
        let l0 = trace.columns[COL_ADDR_L0].evaluations[0].to_u64();
        // Same as account_state_air's documented value for [0xab; 20].
        assert_eq!(l0, u64::from_be_bytes([0xab; 8]));
    }
}
