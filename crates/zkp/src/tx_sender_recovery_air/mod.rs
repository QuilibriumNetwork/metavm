//! Transaction sender recovery chain AIR.
//!
//! Per row, commits one transaction's full sender-derivation chain:
//!
//!   1. The transaction has a signing hash `sig_hash` computed by
//!      [`crate::tx_sig_hash`] from the transaction's RLP-canonical
//!      fields (EIP-155 / EIP-1559 / EIP-2930 dispatch host-side).
//!   2. `(v, r, s)` plus `sig_hash` ECDSA-recover a 65-byte
//!      uncompressed public key `0x04 || X || Y` (the prefix byte is
//!      dropped from this AIR's columns; only the 64 X||Y bytes are
//!      committed).
//!   3. `pubkey_keccak = keccak256(X || Y)`.
//!   4. `derived_address = pubkey_keccak[12..32]` (20 bytes).
//!
//! Each step is bound algebraically via a cross-AIR LogUp linkage to
//! the AIR that already proves it:
//!
//!   - [`make_tx_sender_to_sig_hash_descriptor`] —
//!     `(tx_index, sig_hash)` ↔ `tx_rlp_air` row at the same tx_index.
//!     **Stub binding**: today `tx_rlp_air` does not expose a
//!     `COL_SIG_HASH` or `COL_TX_INDEX` column directly, so the
//!     descriptor wires this AIR's `(R_BYTE..32, S_BYTE..32)` to
//!     `tx_rlp_air`'s `(COL_R_BYTE_OFFSET..32, COL_S_BYTE_OFFSET..32)`
//!     — pinning at least that the (r, s) we recover from match the
//!     transaction's published (r, s). The dedicated `sig_hash`
//!     column will land alongside the tx_sig_hash AIR build-out.
//!   - [`make_tx_sender_to_recovery_descriptor`] —
//!     `(sig_hash, r, s, v_byte, X, Y)` ↔
//!     [`crate::secp256k1_recovery::recovery_air`]'s same tuple.
//!     Today the recovery_air carries a skeleton ECDSA gadget; once
//!     the full secp256k1 nonnative gadget lands, this descriptor is
//!     a drop-in upgrade.
//!   - [`make_tx_sender_to_keccak_descriptor`] —
//!     `(X || Y, pubkey_keccak)` ↔ [`crate::keccak_extract`]'s
//!     `(INPUT_BYTE[0..64], OUTPUT_BYTE[0..32])` tuple with
//!     `INPUT_LEN = 64`.
//!   - [`make_tx_sender_to_address_keccak_descriptor`] —
//!     `derived_address[0..20]` ↔
//!     [`crate::address_keccak_air`]'s `COL_ADDRESS_BE_OFFSET[0..20]`
//!     tuple (pins the derived address as a published row in the
//!     world-state addressing chain).
//!   - [`make_tx_sender_to_tx_nonce_descriptor`] —
//!     `derived_address[0..20]` ↔
//!     [`crate::tx_nonce_air`]'s `COL_SENDER_OFFSET[0..20]` plus
//!     `tx_index` ↔ `COL_TX_INDEX`. Forces the nonce-binding AIR
//!     to read the *recovered* sender.
//!
//! ## Algebraic constraints (row-local)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`.
//! 1..20. `address_byte_equals_keccak_suffix[k]` for k = 0..20 —
//!    `IS_REAL · (DERIVED_ADDR[k] − PUBKEY_KECCAK[12 + k]) = 0`.
//!
//! Per-byte 8-bit range checks (via `lookup_declarations`) on every
//! byte column: `sig_hash[0..32]`, `r[0..32]`, `s[0..32]`, `v_byte`,
//! `pubkey_x[0..32]`, `pubkey_y[0..32]`, `pubkey_keccak[0..32]`,
//! `derived_address[0..20]`.
//!
//! ## What this AIR does NOT prove (deferred)
//!
//!   - That `(v, r, s)` actually ECDSA-recover the committed `(X, Y)`
//!     under `sig_hash` — closed by the upgrade in
//!     [`crate::secp256k1_recovery::recovery_air`] once the full
//!     nonnative gadget lands.
//!   - That `pubkey_keccak = keccak256(X || Y)` — closed by the
//!     [`crate::keccak_extract`] linkage when `KeccakExtract` is in
//!     the joint trace.
//!   - That `sig_hash` is the canonical signing hash of the tx —
//!     closed by [`crate::tx_sig_hash`] / dedicated tx-sig-hash AIR
//!     when wired.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Domain constants ─────────────────────────────────────────────────

pub const HASH_LEN: usize = 32;
pub const COORD_LEN: usize = 32;
pub const ADDR_LEN: usize = 20;
pub const PUBKEY_LEN: usize = 64; // X || Y (no 0x04 prefix)

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_TX_INDEX: usize = 0;
pub const COL_SIG_HASH_OFFSET: usize = COL_TX_INDEX + 1; // 1..33
pub const COL_R_OFFSET: usize = COL_SIG_HASH_OFFSET + HASH_LEN; // 33..65
pub const COL_S_OFFSET: usize = COL_R_OFFSET + 32; // 65..97
pub const COL_V_BYTE: usize = COL_S_OFFSET + 32; // 97
pub const COL_PUBKEY_X_OFFSET: usize = COL_V_BYTE + 1; // 98..130
pub const COL_PUBKEY_Y_OFFSET: usize = COL_PUBKEY_X_OFFSET + COORD_LEN; // 130..162
pub const COL_PUBKEY_KECCAK_OFFSET: usize = COL_PUBKEY_Y_OFFSET + COORD_LEN; // 162..194
pub const COL_DERIVED_ADDR_OFFSET: usize = COL_PUBKEY_KECCAK_OFFSET + HASH_LEN; // 194..214
pub const COL_IS_REAL: usize = COL_DERIVED_ADDR_OFFSET + ADDR_LEN; // 214

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 215

/// Row-local constraints:
///   0: is_real binary
///   1..21 (20 bodies): IS_REAL · (DERIVED_ADDR[k] − PUBKEY_KECCAK[12 + k]) = 0
pub const NUM_ROW_CONSTRAINTS: usize = 1 + ADDR_LEN;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct TxSenderRecoveryRow {
    pub tx_index: u64,
    pub sig_hash: [u8; HASH_LEN],
    pub signature_r: [u8; 32],
    pub signature_s: [u8; 32],
    pub v_byte: u8,
    /// 64-byte uncompressed pubkey: 32 X || 32 Y (no 0x04 prefix).
    pub recovered_pubkey: [u8; PUBKEY_LEN],
    /// keccak256(recovered_pubkey).
    pub pubkey_keccak: [u8; HASH_LEN],
    /// pubkey_keccak[12..32].
    pub derived_address: [u8; ADDR_LEN],
}

#[derive(Clone, Debug, Default)]
pub struct TxSenderRecoveryWitness {
    pub rows: Vec<TxSenderRecoveryRow>,
}

impl TxSenderRecoveryWitness {
    pub fn from_rows(rows: Vec<TxSenderRecoveryRow>) -> Self {
        Self { rows }
    }

    /// Build a single-row witness from a [`crate::transaction::Transaction`].
    ///
    /// * `tx` — the signed transaction.
    /// * `tx_index` — the transaction's position within the block.
    /// * `chain_id` — passed through to the sig-hash + v-decoding
    ///   helpers; semantics match
    ///   [`crate::secp256k1_recovery::verify_tx_sender`]: pass
    ///   `Some(cid)` for EIP-155 legacy, `None` for pre-EIP-155 legacy
    ///   or EIP-1559.
    pub fn from_transaction(
        tx: &crate::transaction::Transaction,
        tx_index: u64,
        chain_id: Option<u64>,
    ) -> Result<Self, String> {
        let sig_hash = crate::tx_sig_hash::signing_hash(tx, chain_id);
        let (v, r, s) = crate::secp256k1_recovery::extract_vrs(tx);
        // Per existing convention in
        // `secp256k1_recovery::verify_tx_sender`: only legacy
        // transactions' v carries chain_id encoding; EIP-1559's
        // y_parity is already 0/1.
        let v_chain = match tx {
            crate::transaction::Transaction::Legacy(_) => chain_id,
            crate::transaction::Transaction::Eip1559(_) => None,
        };
        let (derived_address, x, y) =
            crate::secp256k1_recovery::recover_sender_full(v, r, s, sig_hash, v_chain)?;
        let v_byte = crate::secp256k1_recovery::decode_v(v, v_chain)?;

        let mut recovered_pubkey = [0u8; PUBKEY_LEN];
        recovered_pubkey[0..32].copy_from_slice(&x);
        recovered_pubkey[32..64].copy_from_slice(&y);
        let pubkey_keccak = crate::keccak::keccak256(&recovered_pubkey);

        // Sanity: keccak suffix matches derived address (k256 also
        // computes this internally — this is a defense in depth so
        // the algebraic constraint can never fire on an
        // honest-builder witness).
        debug_assert_eq!(&pubkey_keccak[12..32], &derived_address[..]);

        Ok(Self {
            rows: vec![TxSenderRecoveryRow {
                tx_index,
                sig_hash,
                signature_r: r,
                signature_s: s,
                v_byte,
                recovered_pubkey,
                pubkey_keccak,
                derived_address,
            }],
        })
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &TxSenderRecoveryWitness,
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
            columns[COL_SIG_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.sig_hash[k] as u64, curve);
        }
        for k in 0..32 {
            columns[COL_R_OFFSET + k][i] =
                Scalar::from_u64(row.signature_r[k] as u64, curve);
            columns[COL_S_OFFSET + k][i] =
                Scalar::from_u64(row.signature_s[k] as u64, curve);
        }
        columns[COL_V_BYTE][i] = Scalar::from_u64(row.v_byte as u64, curve);
        for k in 0..COORD_LEN {
            columns[COL_PUBKEY_X_OFFSET + k][i] =
                Scalar::from_u64(row.recovered_pubkey[k] as u64, curve);
            columns[COL_PUBKEY_Y_OFFSET + k][i] =
                Scalar::from_u64(row.recovered_pubkey[32 + k] as u64, curve);
        }
        for k in 0..HASH_LEN {
            columns[COL_PUBKEY_KECCAK_OFFSET + k][i] =
                Scalar::from_u64(row.pubkey_keccak[k] as u64, curve);
        }
        for k in 0..ADDR_LEN {
            columns[COL_DERIVED_ADDR_OFFSET + k][i] =
                Scalar::from_u64(row.derived_address[k] as u64, curve);
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

pub struct TxSenderRecoveryConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl TxSenderRecoveryConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for TxSenderRecoveryConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for k in 0..ADDR_LEN {
            labels.push(format!("derived_addr_byte_{}_eq_keccak_suffix", k));
        }
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

        // 1..21: per-byte address equality with keccak suffix.
        for k in 0..ADDR_LEN {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let is_real = &columns[COL_IS_REAL][r];
                let a = &columns[COL_DERIVED_ADDR_OFFSET + k][r];
                let h = &columns[COL_PUBKEY_KECCAK_OFFSET + 12 + k][r];
                c[r] = is_real.mul(&a.sub(h));
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
        let is_real = &col_evals[COL_IS_REAL];

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        acc = acc.add(&alpha_pow.mul(&is_real.mul(&is_real.sub(&one))));
        alpha_pow = alpha_pow.mul(alpha);

        for k in 0..ADDR_LEN {
            let a = &col_evals[COL_DERIVED_ADDR_OFFSET + k];
            let h = &col_evals[COL_PUBKEY_KECCAK_OFFSET + 12 + k];
            let body = is_real.mul(&a.sub(h));
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
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

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary.
        {
            let body = poly_mul(is_real, &is_real_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        for k in 0..ADDR_LEN {
            let a = &col_coeffs[COL_DERIVED_ADDR_OFFSET + k];
            let h = &col_coeffs[COL_PUBKEY_KECCAK_OFFSET + 12 + k];
            let diff = poly_sub(a, h, curve);
            let body = poly_mul(is_real, &diff, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
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

        let byte_ranges: [(usize, usize, &str); 6] = [
            (COL_SIG_HASH_OFFSET, HASH_LEN, "sig_hash"),
            (COL_R_OFFSET, 32, "r"),
            (COL_S_OFFSET, 32, "s"),
            (COL_PUBKEY_X_OFFSET, COORD_LEN, "pubkey_x"),
            (COL_PUBKEY_Y_OFFSET, COORD_LEN, "pubkey_y"),
            (COL_PUBKEY_KECCAK_OFFSET, HASH_LEN, "pubkey_keccak"),
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
        for k in 0..ADDR_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("derived_addr_{}_8bit", k),
                    column_index: COL_DERIVED_ADDR_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // v_byte 8-bit (typically 0/1 but the range check pins it
        // <256).
        declarations.push((
            LookupDeclaration {
                label: "v_byte_8bit".into(),
                column_index: COL_V_BYTE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind this AIR's `(R, S)` tuple against
/// [`crate::tx_rlp_air`]'s `(R_BYTE, S_BYTE)` columns.
///
/// **Stub binding**: `tx_rlp_air` does not yet expose a `sig_hash`
/// column. Once the dedicated tx-sig-hash AIR or a `COL_SIG_HASH` on
/// `tx_rlp_air` exists, this descriptor will be widened to include
/// `(tx_index, sig_hash)`. For now binding `(R, S)` already pins the
/// signature bytes between the per-tx RLP carrier and this recovery
/// gadget.
pub fn make_tx_sender_to_sig_hash_descriptor(
    sender_layer_index: usize,
    tx_rlp_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::tx_rlp_air as txr;
    let mut a_columns: Vec<usize> = Vec::with_capacity(64);
    let mut b_columns: Vec<usize> = Vec::with_capacity(64);
    for k in 0..32 {
        a_columns.push(COL_R_OFFSET + k);
        b_columns.push(txr::COL_R_BYTE_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_S_OFFSET + k);
        b_columns.push(txr::COL_S_BYTE_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "tx_sender_to_sig_hash_v1".into(),
        a_layer_index: sender_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: tx_rlp_layer_index,
        b_columns,
        b_selector_column: Some(txr::COL_IS_REAL),
    }
}

/// Bind `(sig_hash, r, s, v_byte, recovered_pubkey_x, _y)` of this
/// AIR against [`crate::secp256k1_recovery::recovery_air`]'s same
/// tuple (cols `MSG_HASH`, `R`, `S`, `V`, `RECOVERED_X`, `RECOVERED_Y`).
pub fn make_tx_sender_to_recovery_descriptor(
    sender_layer_index: usize,
    recovery_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::secp256k1_recovery::recovery_air as rec;
    let mut a_columns: Vec<usize> = Vec::with_capacity(32 + 32 + 32 + 1 + 32 + 32);
    let mut b_columns: Vec<usize> = Vec::with_capacity(32 + 32 + 32 + 1 + 32 + 32);
    for k in 0..32 {
        a_columns.push(COL_SIG_HASH_OFFSET + k);
        b_columns.push(rec::COL_MSG_HASH_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_R_OFFSET + k);
        b_columns.push(rec::COL_R_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_S_OFFSET + k);
        b_columns.push(rec::COL_S_OFFSET + k);
    }
    a_columns.push(COL_V_BYTE);
    b_columns.push(rec::COL_V);
    for k in 0..32 {
        a_columns.push(COL_PUBKEY_X_OFFSET + k);
        b_columns.push(rec::COL_RECOVERED_X_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_PUBKEY_Y_OFFSET + k);
        b_columns.push(rec::COL_RECOVERED_Y_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "tx_sender_to_recovery_v1".into(),
        a_layer_index: sender_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: recovery_layer_index,
        b_columns,
        b_selector_column: Some(rec::COL_IS_REAL),
    }
}

/// Bind `(recovered_pubkey[0..64], pubkey_keccak[0..32])` of this AIR
/// against [`crate::keccak_extract`]'s `(INPUT_BYTE[0..64],
/// OUTPUT_BYTE[0..32])`. Combined with KeccakExtract's own constraint
/// system this proves `pubkey_keccak = keccak256(X || Y)`.
pub fn make_tx_sender_to_keccak_descriptor(
    sender_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let mut a_columns: Vec<usize> = Vec::with_capacity(PUBKEY_LEN + HASH_LEN);
    let mut b_columns: Vec<usize> = Vec::with_capacity(PUBKEY_LEN + HASH_LEN);
    for k in 0..32 {
        a_columns.push(COL_PUBKEY_X_OFFSET + k);
        b_columns.push(ke::COL_INPUT_BYTE_OFFSET + k);
    }
    for k in 0..32 {
        a_columns.push(COL_PUBKEY_Y_OFFSET + k);
        b_columns.push(ke::COL_INPUT_BYTE_OFFSET + 32 + k);
    }
    for k in 0..HASH_LEN {
        a_columns.push(COL_PUBKEY_KECCAK_OFFSET + k);
        b_columns.push(ke::COL_OUTPUT_BYTE_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "tx_sender_to_keccak_v1".into(),
        a_layer_index: sender_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Bind `derived_address[0..20]` of this AIR against
/// [`crate::address_keccak_air`]'s `COL_ADDRESS_BE_OFFSET[0..20]`.
/// Hooks the recovered sender into the address-keccak world-state
/// addressing chain so downstream account-state lookups for the
/// sender can fire off the recovered address.
pub fn make_tx_sender_to_address_keccak_descriptor(
    sender_layer_index: usize,
    address_keccak_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::address_keccak_air as ak;
    let a_columns: Vec<usize> = (0..ADDR_LEN)
        .map(|k| COL_DERIVED_ADDR_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = (0..ADDR_LEN)
        .map(|k| ak::COL_ADDRESS_BE_OFFSET + k)
        .collect();
    CrossAirLogUpDescriptor {
        label: "tx_sender_to_address_keccak_v1".into(),
        a_layer_index: sender_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: address_keccak_layer_index,
        b_columns,
        b_selector_column: Some(ak::COL_IS_REAL),
    }
}

/// Bind `(tx_index, derived_address[0..20])` of this AIR against
/// [`crate::tx_nonce_air`]'s `(COL_TX_INDEX, COL_SENDER_OFFSET[0..20])`.
/// Forces the nonce-binding AIR to use the *recovered* sender so the
/// `tx_nonce == account[sender].nonce` check is sound against the
/// signed-transaction author.
pub fn make_tx_sender_to_tx_nonce_descriptor(
    sender_layer_index: usize,
    tx_nonce_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::tx_nonce_air as tn;
    let mut a_columns: Vec<usize> = Vec::with_capacity(1 + ADDR_LEN);
    let mut b_columns: Vec<usize> = Vec::with_capacity(1 + ADDR_LEN);
    a_columns.push(COL_TX_INDEX);
    b_columns.push(tn::COL_TX_INDEX);
    for k in 0..ADDR_LEN {
        a_columns.push(COL_DERIVED_ADDR_OFFSET + k);
        b_columns.push(tn::COL_SENDER_OFFSET + k);
    }
    CrossAirLogUpDescriptor {
        label: "tx_sender_to_tx_nonce_v1".into(),
        a_layer_index: sender_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: tx_nonce_layer_index,
        b_columns,
        b_selector_column: Some(tn::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{Eip1559Tx, LegacyTx, Transaction};
    use k256::ecdsa::{signature::hazmat::PrehashSigner, RecoveryId, Signature, SigningKey};

    fn test_signing_key() -> SigningKey {
        let mut sk_bytes = [0u8; 32];
        for i in 0..32 {
            sk_bytes[i] = (i as u8) + 1;
        }
        SigningKey::from_bytes((&sk_bytes).into()).expect("valid signing key")
    }

    fn test_address(sk: &SigningKey) -> [u8; 20] {
        let vk = sk.verifying_key();
        let encoded = vk.to_encoded_point(false);
        let hash = crate::keccak::keccak256(&encoded.as_bytes()[1..]);
        let mut a = [0u8; 20];
        a.copy_from_slice(&hash[12..32]);
        a
    }

    /// Sign a pre-hash returning (parity, r, s).
    fn raw_sign(sk: &SigningKey, msg_hash: &[u8; 32]) -> (u8, [u8; 32], [u8; 32]) {
        let (sig, rid): (Signature, RecoveryId) =
            sk.sign_prehash(msg_hash).expect("sign_prehash");
        let bytes = sig.to_bytes();
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&bytes[0..32]);
        s.copy_from_slice(&bytes[32..64]);
        (rid.to_byte(), r, s)
    }

    /// Build a fully-signed legacy (pre-EIP-155) transaction for the
    /// test signing key.
    fn signed_legacy_tx() -> (Transaction, [u8; 20]) {
        let sk = test_signing_key();
        let addr = test_address(&sk);
        let mut tx = LegacyTx {
            nonce: 9,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 0,
            r: [0u8; 32],
            s: [0u8; 32],
        };
        let msg_hash = crate::tx_sig_hash::signing_hash(&Transaction::Legacy(tx.clone()), None);
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        tx.v = parity as u64 + 27;
        tx.r = r;
        tx.s = s;
        (Transaction::Legacy(tx), addr)
    }

    fn signed_eip1559_tx() -> (Transaction, [u8; 20]) {
        let sk = test_signing_key();
        let addr = test_address(&sk);
        let mut tx = Eip1559Tx {
            chain_id: 1,
            nonce: 0,
            max_priority_fee_per_gas: [0u8; 32],
            max_fee_per_gas: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: vec![],
            access_list_rlp: vec![0xc0],
            y_parity: 0,
            r: [0u8; 32],
            s: [0u8; 32],
        };
        let msg_hash =
            crate::tx_sig_hash::signing_hash(&Transaction::Eip1559(tx.clone()), None);
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        tx.y_parity = parity as u64;
        tx.r = r;
        tx.s = s;
        (Transaction::Eip1559(tx), addr)
    }

    #[test]
    fn from_transaction_legacy_matches_k256_ground_truth() {
        let (tx, expected_addr) = signed_legacy_tx();
        let w = TxSenderRecoveryWitness::from_transaction(&tx, 7, None)
            .expect("witness builds for legacy");
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].derived_address, expected_addr);
        // sig_hash matches the signing_hash oracle.
        let sig_hash = crate::tx_sig_hash::signing_hash(&tx, None);
        assert_eq!(w.rows[0].sig_hash, sig_hash);
        // pubkey_keccak suffix matches the derived address.
        assert_eq!(&w.rows[0].pubkey_keccak[12..32], &expected_addr[..]);
        // tx_index plumbed through.
        assert_eq!(w.rows[0].tx_index, 7);
    }

    #[test]
    fn from_transaction_eip1559_matches_k256_ground_truth() {
        let (tx, expected_addr) = signed_eip1559_tx();
        let w = TxSenderRecoveryWitness::from_transaction(&tx, 0, None)
            .expect("witness builds for eip1559");
        assert_eq!(w.rows[0].derived_address, expected_addr);
        let sig_hash = crate::tx_sig_hash::signing_hash(&tx, None);
        assert_eq!(w.rows[0].sig_hash, sig_hash);
    }

    #[test]
    fn constraints_zero_on_honest_legacy_witness() {
        let (tx, _) = signed_legacy_tx();
        let w = TxSenderRecoveryWitness::from_transaction(&tx, 0, None).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        let cs = TxSenderRecoveryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) should vanish at row {}",
                    k,
                    cs.constraint_labels()[k],
                    r,
                );
            }
        }
    }

    #[test]
    fn tampered_derived_address_fires_address_equality_body() {
        let (tx, _) = signed_legacy_tx();
        let w = TxSenderRecoveryWitness::from_transaction(&tx, 0, None).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Flip the first byte of derived_address. Body 1 (k=0) must fire.
        let bumped = cols[COL_DERIVED_ADDR_OFFSET][0].to_u64().wrapping_add(1);
        cols[COL_DERIVED_ADDR_OFFSET][0] =
            Scalar::from_u64(bumped, CurveType::Bls48581);
        let cs = TxSenderRecoveryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[1][0].is_zero(),
            "derived_addr_byte_0 equality body should fire on tampered byte",
        );
    }

    #[test]
    fn tampered_signature_detected_by_host_recovery() {
        // Tampering r before recovery should either (a) cause recovery
        // to fail, or (b) recover a *different* address. Either way the
        // from_transaction helper must NOT silently return the original
        // expected address.
        let (tx, expected) = signed_legacy_tx();
        let mut tampered = match tx {
            Transaction::Legacy(mut l) => {
                l.r[0] ^= 0x01;
                Transaction::Legacy(l)
            }
            _ => unreachable!(),
        };
        // Re-derive the matching v if needed — we just want to ensure the
        // recovered address (when recovery succeeds) is not the original.
        match TxSenderRecoveryWitness::from_transaction(&mut tampered, 0, None) {
            Ok(w) => {
                assert_ne!(
                    w.rows[0].derived_address, expected,
                    "tampered signature must not silently recover the honest sender",
                );
            }
            Err(_) => {
                // Recovery failure is also an acceptable detection.
            }
        }
    }

    #[test]
    fn is_real_binary_fires_on_non_binary() {
        let (tx, _) = signed_legacy_tx();
        let w = TxSenderRecoveryWitness::from_transaction(&tx, 0, None).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = TxSenderRecoveryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[0][0].is_zero(),
            "is_real_binary should fire when IS_REAL = 2",
        );
    }

    #[test]
    fn descriptors_well_formed() {
        use crate::keccak_extract as ke;
        use crate::secp256k1_recovery::recovery_air as rec;
        use crate::tx_rlp_air as txr;

        let d1 = make_tx_sender_to_sig_hash_descriptor(0, 1);
        assert_eq!(d1.label, "tx_sender_to_sig_hash_v1");
        // 32 r + 32 s = 64 cols.
        assert_eq!(d1.a_columns.len(), 64);
        assert_eq!(d1.b_columns.len(), 64);
        assert_eq!(d1.a_columns[0], COL_R_OFFSET);
        assert_eq!(d1.b_columns[0], txr::COL_R_BYTE_OFFSET);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(txr::COL_IS_REAL));

        let d2 = make_tx_sender_to_recovery_descriptor(0, 2);
        assert_eq!(d2.label, "tx_sender_to_recovery_v1");
        // 32 + 32 + 32 + 1 + 32 + 32 = 161 cols.
        assert_eq!(d2.a_columns.len(), 161);
        assert_eq!(d2.b_columns.len(), 161);
        assert_eq!(d2.a_columns[0], COL_SIG_HASH_OFFSET);
        assert_eq!(d2.b_columns[0], rec::COL_MSG_HASH_OFFSET);
        assert_eq!(d2.a_columns[96], COL_V_BYTE);
        assert_eq!(d2.b_columns[96], rec::COL_V);

        let d3 = make_tx_sender_to_keccak_descriptor(0, 3);
        assert_eq!(d3.label, "tx_sender_to_keccak_v1");
        // 64 input + 32 output = 96 cols.
        assert_eq!(d3.a_columns.len(), PUBKEY_LEN + HASH_LEN);
        assert_eq!(d3.b_columns.len(), PUBKEY_LEN + HASH_LEN);
        assert_eq!(d3.a_columns[0], COL_PUBKEY_X_OFFSET);
        assert_eq!(d3.b_columns[0], ke::COL_INPUT_BYTE_OFFSET);
        assert_eq!(d3.a_columns[PUBKEY_LEN], COL_PUBKEY_KECCAK_OFFSET);
        assert_eq!(d3.b_columns[PUBKEY_LEN], ke::COL_OUTPUT_BYTE_OFFSET);

        let d4 = make_tx_sender_to_address_keccak_descriptor(0, 4);
        assert_eq!(d4.label, "tx_sender_to_address_keccak_v1");
        assert_eq!(d4.a_columns.len(), ADDR_LEN);
        assert_eq!(d4.b_columns.len(), ADDR_LEN);
        assert_eq!(d4.a_columns[0], COL_DERIVED_ADDR_OFFSET);
        assert_eq!(
            d4.b_columns[0],
            crate::address_keccak_air::COL_ADDRESS_BE_OFFSET,
        );

        let d5 = make_tx_sender_to_tx_nonce_descriptor(0, 5);
        assert_eq!(d5.label, "tx_sender_to_tx_nonce_v1");
        assert_eq!(d5.a_columns.len(), 1 + ADDR_LEN);
        assert_eq!(d5.b_columns.len(), 1 + ADDR_LEN);
        assert_eq!(d5.a_columns[0], COL_TX_INDEX);
        assert_eq!(d5.b_columns[0], crate::tx_nonce_air::COL_TX_INDEX);
        for k in 0..ADDR_LEN {
            assert_eq!(d5.a_columns[1 + k], COL_DERIVED_ADDR_OFFSET + k);
            assert_eq!(
                d5.b_columns[1 + k],
                crate::tx_nonce_air::COL_SENDER_OFFSET + k,
            );
        }
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_TX_INDEX, 0);
        assert_eq!(COL_SIG_HASH_OFFSET, 1);
        assert_eq!(COL_R_OFFSET, 33);
        assert_eq!(COL_S_OFFSET, 65);
        assert_eq!(COL_V_BYTE, 97);
        assert_eq!(COL_PUBKEY_X_OFFSET, 98);
        assert_eq!(COL_PUBKEY_Y_OFFSET, 130);
        assert_eq!(COL_PUBKEY_KECCAK_OFFSET, 162);
        assert_eq!(COL_DERIVED_ADDR_OFFSET, 194);
        assert_eq!(COL_IS_REAL, 214);
        assert_eq!(NUM_COLUMNS, 215);
        assert_eq!(NUM_ROW_CONSTRAINTS, 21);
        assert_eq!(NUM_SHIFTED, 0);
    }

    /// Byte range lookup coverage: every byte column gets an 8-bit
    /// declaration, plus v_byte.
    #[test]
    fn byte_range_lookup_coverage() {
        let cs = TxSenderRecoveryConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        // 32 + 32 + 32 + 32 + 32 + 32 + 20 + 1 (v_byte) = 213.
        let expected = HASH_LEN + 32 + 32 + COORD_LEN + COORD_LEN + HASH_LEN + ADDR_LEN + 1;
        assert_eq!(reqs.declarations.len(), expected);
        for (decl, table_idx) in &reqs.declarations {
            assert_eq!(decl.max_bits, 8);
            assert_eq!(*table_idx, 0);
            assert!(decl.column_index < NUM_COLUMNS);
        }
    }

    /// `evaluate_at_point` agrees with `evaluate_on_domain` on an
    /// honest witness (both produce zero).
    #[test]
    fn evaluate_at_point_matches_domain_for_honest() {
        let (tx, _) = signed_legacy_tx();
        let w = TxSenderRecoveryWitness::from_transaction(&tx, 0, None).unwrap();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = TxSenderRecoveryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let alpha = Scalar::from_u64(17, CurveType::Bls48581);
        let row0_evals: Vec<Scalar> = col_refs.iter().map(|c| c[0].clone()).collect();
        let agg = cs.evaluate_at_point(&row0_evals, &alpha);
        assert!(agg.is_zero(), "α-RLC aggregate must vanish on honest row");
    }
}
