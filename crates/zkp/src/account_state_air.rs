//! Account state gadget AIR — Phase A2 step 3 (#69) commitment skeleton.
//!
//! Per row, commits one Ethereum account state binding:
//!   `(address, account, state_root, address_trie_key)`
//! where `address_trie_key = keccak256(address)` and the account RLP
//! at that key in the world MPT (rooted at `state_root`) decodes to
//! the committed `(nonce, balance, storage_root, code_hash)`.
//!
//! **Step 0 (this commit)**: witness skeleton with row-local
//! constraints. The MPT inclusion linkage (binding `state_root` →
//! `account_rlp` at `address_trie_key`) and the RLP encoding
//! correctness (binding committed fields → `account_rlp_bytes`) are
//! deferred — see `account.rs` for the host-side oracle.
//!
//! Cross-AIR linkages (descriptors below):
//!   - L_storage_to_account: storage_access_air ↔ this gadget on
//!     `(address_limbs, storage_root_bytes)` — binds the storage
//!     gadget's contract address+storage_root to a known account.
//!   - L_account_to_address_keccak: this gadget ↔ address_keccak_air
//!     on `(address_limbs, address_trie_key_bytes)` — chains through
//!     the address-keccak gadget for trie-key derivation.
//!   - L_account_to_block_header: this gadget ↔ block_header_air on
//!     `state_root_bytes` — binds state_root to the canonical block.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

// 20-byte address as 4 LE u64 limbs (top 96 bits = 0).
pub const COL_ADDR_L0: usize = 0;
pub const COL_ADDR_L1: usize = 1;
pub const COL_ADDR_L2: usize = 2;
pub const COL_ADDR_L3: usize = 3;

// Account state fields.
pub const COL_NONCE: usize = 4;
pub const COL_BALANCE_OFFSET: usize = 5;        // 5..37 (32 BE bytes)
pub const COL_STORAGE_ROOT_OFFSET: usize = 37;  // 37..69 (32 bytes)
pub const COL_CODE_HASH_OFFSET: usize = 69;     // 69..101 (32 bytes)

// World state context.
pub const COL_STATE_ROOT_OFFSET: usize = 101;       // 101..133 (32 bytes)
pub const COL_ADDRESS_TRIE_KEY_OFFSET: usize = 133; // 133..165 (= keccak256(address))

// Balance as 4 LE u64 limbs (for SELFBALANCE cross-AIR LogUp).
pub const COL_BALANCE_L0: usize = 166;
pub const COL_BALANCE_L1: usize = 167;
pub const COL_BALANCE_L2: usize = 168;
pub const COL_BALANCE_L3: usize = 169;

pub const COL_IS_REAL: usize = 170;

// Post-state nonce snapshot. Committed alongside the pre-state
// `COL_NONCE` so that downstream descriptors (e.g.
// `tx_nonce_air::make_tx_nonce_to_account_post_descriptor`) can
// algebraically distinguish the pre- and post-execution nonce of the
// same account row. A row-local constraint enforces
// `is_real · (nonce_post − nonce − 1) = 0`, i.e. the canonical
// +1 increment for a successful transaction.
pub const COL_NONCE_POST: usize = 171;

pub const NUM_COLUMNS: usize = COL_NONCE_POST + 1; // 172

/// Step 0 row-local constraints: just `is_real` binary. Step 1+
/// adds RLP encoding of (nonce, balance, storage_root, code_hash) →
/// `account_rlp_bytes`, plus MPT inclusion verification at
/// `state_root`.
/// Constraints:
/// 0: is_real binary
/// 1-4: balance limb consistency (each limb = Horner reconstruction of 8 BE bytes)
/// 5: nonce_post = nonce + 1 increment (gated by is_real)
pub const NUM_ROW_CONSTRAINTS: usize = 6;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AccountStateRow {
    pub address: [u8; 20],
    pub account: crate::account::Account,
    pub state_root: [u8; 32],
    /// Post-execution nonce snapshot. Canonical convention for a
    /// successful transaction is `nonce_post = account.nonce + 1`.
    /// Use [`AccountStateRow::new`] to populate this automatically.
    pub nonce_post: u64,
}

impl AccountStateRow {
    /// Convenience constructor: populates `nonce_post = account.nonce + 1`.
    pub fn new(
        address: [u8; 20],
        account: crate::account::Account,
        state_root: [u8; 32],
    ) -> Self {
        let nonce_post = account.nonce.saturating_add(1);
        Self { address, account, state_root, nonce_post }
    }

    /// Explicit constructor allowing the caller to pin `nonce_post`
    /// (e.g. for negative tests, or for non-default post-state
    /// scenarios once world-state transitions land).
    pub fn with_nonce_post(
        address: [u8; 20],
        account: crate::account::Account,
        state_root: [u8; 32],
        nonce_post: u64,
    ) -> Self {
        Self { address, account, state_root, nonce_post }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AccountStateWitness {
    pub accesses: Vec<AccountStateRow>,
}

impl AccountStateWitness {
    pub fn from_rows(rows: Vec<AccountStateRow>) -> Self {
        Self { accesses: rows }
    }
}

fn address_to_limbs(address: [u8; 20]) -> [u64; 4] {
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
    witness: &AccountStateWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.accesses.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.accesses.iter().enumerate() {
        // Address limbs.
        let addr_limbs = address_to_limbs(row.address);
        for j in 0..4 {
            columns[COL_ADDR_L0 + j][i] = Scalar::from_u64(addr_limbs[j], curve);
        }
        // Account fields.
        columns[COL_NONCE][i] = Scalar::from_u64(row.account.nonce, curve);
        columns[COL_NONCE_POST][i] = Scalar::from_u64(row.nonce_post, curve);
        for k in 0..32 {
            columns[COL_BALANCE_OFFSET + k][i] =
                Scalar::from_u64(row.account.balance[k] as u64, curve);
            columns[COL_STORAGE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.account.storage_root[k] as u64, curve);
            columns[COL_CODE_HASH_OFFSET + k][i] =
                Scalar::from_u64(row.account.code_hash[k] as u64, curve);
            columns[COL_STATE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.state_root[k] as u64, curve);
        }
        // address_trie_key = keccak256(address).
        let addr_hash = crate::keccak::keccak256(&row.address);
        for k in 0..32 {
            columns[COL_ADDRESS_TRIE_KEY_OFFSET + k][i] =
                Scalar::from_u64(addr_hash[k] as u64, curve);
        }
        // Balance as 4 LE u64 limbs (BE bytes → LE limbs).
        let b = &row.account.balance;
        columns[COL_BALANCE_L0][i] = Scalar::from_u64(
            u64::from_be_bytes([b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]]), curve);
        columns[COL_BALANCE_L1][i] = Scalar::from_u64(
            u64::from_be_bytes([b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23]]), curve);
        columns[COL_BALANCE_L2][i] = Scalar::from_u64(
            u64::from_be_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]), curve);
        columns[COL_BALANCE_L3][i] = Scalar::from_u64(
            u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]), curve);
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

// ─── Constraint system ─────────────────────────────────────────────────

pub struct AccountStateConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AccountStateConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for AccountStateConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "balance_limb0_consistency".into(),
            "balance_limb1_consistency".into(),
            "balance_limb2_consistency".into(),
            "balance_limb3_consistency".into(),
            "nonce_post_increments_nonce".into(),
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
        let two56 = Scalar::from_u64(256, curve);
        let n = columns[0].len();
        let mut c0 = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            c0[r] = v.mul(&v.sub(&one));
        }
        let mut limb_bodies = Vec::with_capacity(4);
        for j in 0..4usize {
            let mut body = vec![Scalar::zero(curve); n];
            let byte_start = (3 - j) * 8;
            for r in 0..n {
                let mut horner = Scalar::zero(curve);
                let mut weight = Scalar::one(curve);
                for k in (0..8).rev() {
                    let b = &columns[COL_BALANCE_OFFSET + byte_start + k][r];
                    horner = horner.add(&weight.mul(b));
                    weight = weight.mul(&two56);
                }
                body[r] = columns[COL_BALANCE_L0 + j][r].sub(&horner);
            }
            limb_bodies.push(body);
        }
        // Body 5: is_real · (nonce_post − nonce − 1).
        let mut nonce_post_body = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let diff = columns[COL_NONCE_POST][r]
                .sub(&columns[COL_NONCE][r])
                .sub(&one);
            nonce_post_body[r] = is_real.mul(&diff);
        }
        let mut result = vec![c0];
        result.extend(limb_bodies);
        result.push(nonce_post_body);
        result
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two56 = Scalar::from_u64(256, curve);
        let v = &col_evals[COL_IS_REAL];
        let c0 = v.mul(&v.sub(&one));
        let mut total = c0;
        let mut ap = alpha.clone();
        for j in 0..4usize {
            let byte_start = (3 - j) * 8;
            let mut horner = Scalar::zero(curve);
            let mut weight = Scalar::one(curve);
            for k in (0..8).rev() {
                horner = horner.add(&weight.mul(&col_evals[COL_BALANCE_OFFSET + byte_start + k]));
                weight = weight.mul(&two56);
            }
            let body = col_evals[COL_BALANCE_L0 + j].sub(&horner);
            total = total.add(&ap.mul(&body));
            ap = ap.mul(alpha);
        }
        // Body 5: is_real · (nonce_post − nonce − 1).
        let nonce_post_diff = col_evals[COL_NONCE_POST]
            .sub(&col_evals[COL_NONCE])
            .sub(&one);
        let nonce_post_body = col_evals[COL_IS_REAL].mul(&nonce_post_diff);
        total = total.add(&ap.mul(&nonce_post_body));
        total
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let two56 = Scalar::from_u64(256, curve);
        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let c0 = poly_mul(v, &v_m1, curve);
        let mut total = c0;
        let mut ap = alpha.clone();
        for j in 0..4usize {
            let byte_start = (3 - j) * 8;
            let mut horner = vec![Scalar::zero(curve)];
            let mut weight = Scalar::one(curve);
            for k in (0..8).rev() {
                horner = poly_add(&horner, &poly_scalar_mul(&col_coeffs[COL_BALANCE_OFFSET + byte_start + k], &weight), curve);
                weight = weight.mul(&two56);
            }
            let body = poly_sub(&col_coeffs[COL_BALANCE_L0 + j], &horner, curve);
            total = poly_add(&total, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        // Body 5: is_real · (nonce_post − nonce − 1).
        let diff_np_n = poly_sub(
            &col_coeffs[COL_NONCE_POST],
            &col_coeffs[COL_NONCE],
            curve,
        );
        let diff_np = poly_sub(&diff_np_n, &one_poly, curve);
        let nonce_post_body = poly_mul(&col_coeffs[COL_IS_REAL], &diff_np, curve);
        total = poly_add(&total, &poly_scalar_mul(&nonce_post_body, &ap), curve);
        total
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
        // 8-bit range checks on byte columns: balance(32) + storage_root(32)
        // + code_hash(32) + state_root(32) + address_trie_key(32) = 160.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        let byte_starts = [
            ("balance", COL_BALANCE_OFFSET),
            ("storage_root", COL_STORAGE_ROOT_OFFSET),
            ("code_hash", COL_CODE_HASH_OFFSET),
            ("state_root", COL_STATE_ROOT_OFFSET),
            ("address_trie_key", COL_ADDRESS_TRIE_KEY_OFFSET),
        ];
        for (name, offset) in byte_starts {
            for k in 0..32 {
                declarations.push((
                    LookupDeclaration {
                        label: format!("account_state_{}_byte_{}_8bit", name, k),
                        column_index: offset + k,
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

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Link from storage gadget to this account-state gadget on
/// `(address_limbs[0..4], storage_root_bytes[0..32])` — 36-col tuple.
/// Forces every storage access's (address, storage_root) pair to
/// match a row in this gadget (which itself binds storage_root via
/// MPT inclusion at state_root in step 1+).
pub fn make_storage_to_account_state_linkage_descriptor(
    storage_layer_index: usize,
    account_state_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::storage_access_air as sa;
    let a_columns: Vec<usize> = vec![sa::COL_ADDR_L0, sa::COL_ADDR_L1, sa::COL_ADDR_L2, sa::COL_ADDR_L3]
        .into_iter()
        .chain((0..32).map(|k| sa::COL_STORAGE_ROOT_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = vec![COL_ADDR_L0, COL_ADDR_L1, COL_ADDR_L2, COL_ADDR_L3]
        .into_iter()
        .chain((0..32).map(|k| COL_STORAGE_ROOT_OFFSET + k))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "storage_to_account_state_v1".into(),
        a_layer_index: storage_layer_index,
        a_columns,
        a_selector_column: Some(sa::COL_IS_REAL),
        b_layer_index: account_state_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Link from account_state_air to address_keccak_air on `(address_limbs[0..4],
/// address_trie_key_bytes[0..32])` — 36-col tuple. Forces the
/// account_state's claimed `(address, address_trie_key)` to match a
/// row in the address_keccak gadget (which itself has the keccak
/// link to KeccakExtract).
pub fn make_account_state_to_address_keccak_linkage_descriptor(
    account_state_layer_index: usize,
    address_keccak_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::address_keccak_air as ak;
    let a_columns: Vec<usize> = vec![COL_ADDR_L0, COL_ADDR_L1, COL_ADDR_L2, COL_ADDR_L3]
        .into_iter()
        .chain((0..32).map(|k| COL_ADDRESS_TRIE_KEY_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = vec![
        ak::COL_ADDRESS_LIMB_L0,
        ak::COL_ADDRESS_LIMB_L1,
        ak::COL_ADDRESS_LIMB_L2,
        ak::COL_ADDRESS_LIMB_L3,
    ]
    .into_iter()
    .chain((0..32).map(|k| ak::COL_ADDRESS_TRIE_KEY_OFFSET + k))
    .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "account_state_to_address_keccak_v1".into(),
        a_layer_index: account_state_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: address_keccak_layer_index,
        b_columns,
        b_selector_column: Some(ak::COL_IS_REAL),
    }
}

/// Link from this gadget to block_header_air on `state_root_bytes`
/// — 32-col tuple. Forces every account-state binding's state_root
/// to match a canonical block header's state_root.
pub fn make_account_state_to_block_header_linkage_descriptor(
    account_state_layer_index: usize,
    block_header_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    let a_columns: Vec<usize> = (0..32).map(|k| COL_STATE_ROOT_OFFSET + k).collect();
    let b_columns: Vec<usize> = (0..32).map(|k| bh::COL_STATE_ROOT_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "account_state_to_block_header_v1".into(),
        a_layer_index: account_state_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: block_header_layer_index,
        b_columns,
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::Account;

    fn sample_row() -> AccountStateRow {
        AccountStateRow::new(
            [0xab; 20],
            Account {
                nonce: 7,
                balance: [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x00,
                ],
                storage_root: [0x11; 32],
                code_hash: [0x22; 32],
            },
            [0x33; 32],
        )
    }

    #[test]
    fn trace_builder_populates_all_columns() {
        let row = sample_row();
        let w = AccountStateWitness::from_rows(vec![row.clone()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // Address limb 0 (low 64 bits of zero-padded address).
        assert_eq!(trace.columns[COL_ADDR_L0].evaluations[0].to_u64(),
                   u64::from_be_bytes([0xab; 8]));
        // Nonce.
        assert_eq!(trace.columns[COL_NONCE].evaluations[0].to_u64(), 7);
        // Balance bytes.
        for k in 0..30 {
            assert_eq!(trace.columns[COL_BALANCE_OFFSET + k].evaluations[0].to_u64(), 0);
        }
        assert_eq!(trace.columns[COL_BALANCE_OFFSET + 30].evaluations[0].to_u64(), 0x10);
        assert_eq!(trace.columns[COL_BALANCE_OFFSET + 31].evaluations[0].to_u64(), 0x00);
        // Storage root.
        assert_eq!(trace.columns[COL_STORAGE_ROOT_OFFSET].evaluations[0].to_u64(), 0x11);
        // Code hash.
        assert_eq!(trace.columns[COL_CODE_HASH_OFFSET].evaluations[0].to_u64(), 0x22);
        // State root.
        assert_eq!(trace.columns[COL_STATE_ROOT_OFFSET].evaluations[0].to_u64(), 0x33);
        // Address trie key = keccak256([0xab; 20]) — first byte should match.
        let expected = crate::keccak::keccak256(&[0xab; 20]);
        assert_eq!(trace.columns[COL_ADDRESS_TRIE_KEY_OFFSET].evaluations[0].to_u64(),
                   expected[0] as u64);
        // Balance limbs (LE). Balance = 0x1000 = 4096.
        // BE bytes: [0..30]=0, [30]=0x10, [31]=0x00
        // L0 = u64::from_be_bytes([0,0,0,0,0,0,0x10,0x00]) = 0x1000
        assert_eq!(trace.columns[COL_BALANCE_L0].evaluations[0].to_u64(), 0x1000);
        assert_eq!(trace.columns[COL_BALANCE_L1].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_BALANCE_L2].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_BALANCE_L3].evaluations[0].to_u64(), 0);
        // is_real = 1.
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = AccountStateWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AccountStateConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for col in results {
            for v in col {
                assert!(v.is_zero(), "constraint must vanish");
            }
        }
    }

    #[test]
    fn storage_to_account_state_descriptor_well_formed() {
        let desc = make_storage_to_account_state_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "storage_to_account_state_v1");
        // 4 address limbs + 32 storage_root bytes = 36 cols.
        assert_eq!(desc.a_columns.len(), 36);
        assert_eq!(desc.b_columns.len(), 36);
    }

    #[test]
    fn account_state_to_address_keccak_descriptor_well_formed() {
        let desc = make_account_state_to_address_keccak_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "account_state_to_address_keccak_v1");
        // 4 address limbs + 32 trie key bytes = 36 cols.
        assert_eq!(desc.a_columns.len(), 36);
        assert_eq!(desc.b_columns.len(), 36);
        assert_eq!(desc.a_columns[0], COL_ADDR_L0);
        assert_eq!(desc.a_columns[4], COL_ADDRESS_TRIE_KEY_OFFSET);
        assert_eq!(
            desc.b_columns[0],
            crate::address_keccak_air::COL_ADDRESS_LIMB_L0,
        );
        assert_eq!(
            desc.b_columns[4],
            crate::address_keccak_air::COL_ADDRESS_TRIE_KEY_OFFSET,
        );
    }

    #[test]
    fn account_state_to_block_header_descriptor_well_formed() {
        let desc = make_account_state_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "account_state_to_block_header_v1");
        assert_eq!(desc.a_columns.len(), 32);
        assert_eq!(desc.b_columns.len(), 32);
    }

    /// Phase A2 step 3 / #53 step 0 multi-AIR joint test: 3 gadgets
    /// (storage_access_air + account_state_air + block_header_air)
    /// chained via 2 linkages:
    ///   - L_storage_to_account_state (36-col tuple binding storage
    ///     gadget's address+storage_root → account_state gadget)
    ///   - L_account_state_to_block_header (32-col state_root tuple
    ///     binding account_state gadget → block_header_air)
    ///
    /// Validates that the chain "storage access → account → block
    /// header" is algebraically composable. Excludes EVM main so the
    /// test is fast (~2-4 min release).
    #[test]
    #[ignore = "slow: 3-gadget joint_prove (~3-5 min); run with --release --ignored"]
    fn joint_prove_storage_account_block_chain() {
        use crate::block_header::{block_header_hash, BlockHeader};
        use crate::block_header_air::{
            self as bh, build_trace_polynomials as build_bh_trace,
            BlockHeaderConstraintSystem, BlockHeaderWitness, from_block_header,
        };
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::storage_access_air::{
            self as sa, build_trace_polynomials as build_storage_trace,
            StorageAccessConstraintSystem, StorageAccessRow, StorageAccessWitness,
        };
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // One contract at address [0xab; 20], one storage access at
        // slot=5, value=0x42. Account has storage_root = [0x11; 32].
        // Block header's state_root = [0x99; 32].
        let address = [0xab; 20];
        let storage_root = [0x11; 32];
        let state_root_bytes = [0x99; 32];

        // Storage gadget witness: 1 access (SLOAD).
        let storage_w = StorageAccessWitness::from_rows(vec![StorageAccessRow {
            address,
            slot: [0x05, 0, 0, 0],
            value: [0x42, 0, 0, 0],
            storage_root,
            is_write: false,
        }]);
        let storage_trace = build_storage_trace(&storage_w, curve);
        let storage_omega = scheme.domain_generator(storage_trace.padded_size);
        let storage_cs = StorageAccessConstraintSystem::new(storage_trace.num_rows)
            .with_omega_and_domain(storage_omega, storage_trace.padded_size);

        // Account state gadget witness: 1 row matching the storage
        // gadget's (address, storage_root) and our state_root.
        let account_state_w = AccountStateWitness::from_rows(vec![AccountStateRow::new(
            address,
            crate::account::Account {
                nonce: 1,
                balance: [0u8; 32],
                storage_root,
                code_hash: crate::account::empty_code_hash(),
            },
            state_root_bytes,
        )]);
        let account_state_trace = build_trace_polynomials(&account_state_w, curve);
        let account_omega = scheme.domain_generator(account_state_trace.padded_size);
        let account_cs = AccountStateConstraintSystem::new(account_state_trace.num_rows)
            .with_omega_and_domain(account_omega, account_state_trace.padded_size);

        // Block header gadget witness: 1 row with our state_root.
        let mut h = BlockHeader::default();
        h.state_root = state_root_bytes;
        h.number = 17_000_000;
        let bh_row = from_block_header(&h);
        // Sanity: the row's block_hash is computed.
        assert_eq!(bh_row.block_hash, block_header_hash(&h));
        let bh_w = BlockHeaderWitness::from_headers(vec![bh_row]);
        let bh_trace = build_bh_trace(&bh_w, curve);
        let bh_omega = scheme.domain_generator(bh_trace.padded_size);
        let bh_cs = BlockHeaderConstraintSystem::new(bh_trace.num_rows)
            .with_omega_and_domain(bh_omega, bh_trace.padded_size);

        // Linkages: storage→account_state, account_state→block_header.
        let linkages = vec![
            make_storage_to_account_state_linkage_descriptor(0, 1),
            make_account_state_to_block_header_linkage_descriptor(1, 2),
        ];

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> = vec![
            (&storage_trace, &storage_cs),
            (&account_state_trace, &account_cs),
            (&bh_trace, &bh_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("3-gadget joint_prove must succeed");
        assert_eq!(proofs.len(), 3);
        assert_eq!(ext.linkage_proofs.len(), 2);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            eprintln!(
                "[diag] linkage {} label={} closure_match={}",
                i, lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b);
        }

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&storage_cs, &account_cs, &bh_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "3-gadget joint_verify must accept honest witness",
        );
        let _ = sa::COL_ADDR_L0; // silence unused-import warning
        let _ = bh::COL_STATE_ROOT_OFFSET;
    }

    /// **Phase A2 step 3 5-AIR chain joint_prove**: extends the 3-
    /// gadget chain with `address_keccak_air` + `KeccakExtract`,
    /// algebraically binding the address_trie_key derivation.
    ///
    /// AIRs (in order):
    ///   0. storage_access_air
    ///   1. account_state_air
    ///   2. address_keccak_air
    ///   3. KeccakExtract
    ///   4. block_header_air
    ///
    /// Linkages:
    ///   - L1 storage→account_state (binding address+storage_root)
    ///   - L2 account_state→block_header (binding state_root)
    ///   - L3 storage→address_keccak (binding address limbs)
    ///   - L4 account_state→address_keccak (binding address +
    ///     address_trie_key)
    ///   - L5 address_keccak→KeccakExtract (binding
    ///     address_trie_key = keccak256(address))
    ///
    /// All 5 closures must match. Estimated ~6-10 min release.
    #[test]
    #[ignore = "slow: 5-AIR joint_prove (~6-10 min); run with --release --ignored"]
    fn joint_prove_storage_account_address_keccak_block_chain() {
        use crate::address_keccak_air::{
            self as ak,
            build_trace_polynomials as build_ak_trace,
            AddressKeccakConstraintSystem, AddressKeccakWitness,
        };
        use crate::block_header::{block_header_hash, BlockHeader};
        use crate::block_header_air::{
            build_trace_polynomials as build_bh_trace,
            from_block_header, BlockHeaderConstraintSystem, BlockHeaderWitness,
        };
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak_extract::{
            build_trace_polynomials as build_keccak_trace,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::storage_access_air::{
            build_trace_polynomials as build_storage_trace,
            StorageAccessConstraintSystem, StorageAccessRow, StorageAccessWitness,
        };
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let address = [0xab; 20];
        let storage_root = [0x11; 32];
        let state_root_bytes = [0x99; 32];

        // Storage gadget witness: 1 SLOAD.
        let storage_w = StorageAccessWitness::from_rows(vec![StorageAccessRow {
            address,
            slot: [0x05, 0, 0, 0],
            value: [0x42, 0, 0, 0],
            storage_root,
            is_write: false,
        }]);
        let storage_trace = build_storage_trace(&storage_w, curve);
        let storage_omega = scheme.domain_generator(storage_trace.padded_size);
        let storage_cs = StorageAccessConstraintSystem::new(storage_trace.num_rows)
            .with_omega_and_domain(storage_omega, storage_trace.padded_size);

        // Account state gadget witness.
        let account_state_w = AccountStateWitness::from_rows(vec![AccountStateRow::new(
            address,
            crate::account::Account {
                nonce: 1,
                balance: [0u8; 32],
                storage_root,
                code_hash: crate::account::empty_code_hash(),
            },
            state_root_bytes,
        )]);
        let account_state_trace = build_trace_polynomials(&account_state_w, curve);
        let account_omega = scheme.domain_generator(account_state_trace.padded_size);
        let account_cs = AccountStateConstraintSystem::new(account_state_trace.num_rows)
            .with_omega_and_domain(account_omega, account_state_trace.padded_size);

        // Address keccak gadget witness — needs 2 rows because both
        // storage_access_air and account_state_air emit address tuples
        // and the LogUp matches multiset, so we need 2 matching rows
        // on the address_keccak side.
        let ak_w = AddressKeccakWitness::from_addresses(vec![address, address]);
        let ak_trace = build_ak_trace(&ak_w, curve);
        let ak_omega = scheme.domain_generator(ak_trace.padded_size);
        let ak_cs = AddressKeccakConstraintSystem::new(ak_trace.num_rows)
            .with_omega_and_domain(ak_omega, ak_trace.padded_size);

        // KeccakExtract witness: 2 invocations (one per address_keccak row).
        let keccak_w = KeccakExtractWitness::from_inputs(&[address.to_vec(), address.to_vec()]).unwrap();
        let keccak_trace = build_keccak_trace(&keccak_w, curve);
        let keccak_omega = scheme.domain_generator(keccak_trace.padded_size);
        let keccak_cs = KeccakExtractConstraintSystem::new(keccak_trace.num_rows)
            .with_omega_and_domain(keccak_omega, keccak_trace.padded_size);

        // Block header gadget witness.
        let mut h = BlockHeader::default();
        h.state_root = state_root_bytes;
        h.number = 17_000_000;
        let bh_row = from_block_header(&h);
        assert_eq!(bh_row.block_hash, block_header_hash(&h));
        let bh_w = BlockHeaderWitness::from_headers(vec![bh_row]);
        let bh_trace = build_bh_trace(&bh_w, curve);
        let bh_omega = scheme.domain_generator(bh_trace.padded_size);
        let bh_cs = BlockHeaderConstraintSystem::new(bh_trace.num_rows)
            .with_omega_and_domain(bh_omega, bh_trace.padded_size);

        // Linkages: 5 cross-AIR LogUps.
        let linkages = vec![
            make_storage_to_account_state_linkage_descriptor(0, 1),
            make_account_state_to_block_header_linkage_descriptor(1, 4),
            ak::make_storage_to_address_keccak_linkage_descriptor(0, 2),
            make_account_state_to_address_keccak_linkage_descriptor(1, 2),
            ak::make_address_to_keccak_extract_linkage_descriptor(2, 3),
        ];

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> = vec![
            (&storage_trace, &storage_cs),
            (&account_state_trace, &account_cs),
            (&ak_trace, &ak_cs),
            (&keccak_trace, &keccak_cs),
            (&bh_trace, &bh_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("5-AIR storage chain joint_prove must succeed");
        assert_eq!(proofs.len(), 5);
        assert_eq!(ext.linkage_proofs.len(), 5);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            eprintln!(
                "[diag] linkage {} label={} closure_match={}",
                i, lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b);
        }

        let cs_refs: Vec<&dyn VmConstraintSystem> =
            vec![&storage_cs, &account_cs, &ak_cs, &keccak_cs, &bh_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "5-AIR joint_verify must accept honest witness",
        );
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary() {
        let w = AccountStateWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = AccountStateConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "is_real_binary should fire");
    }

    #[test]
    fn tampered_balance_limb_detected() {
        let w = AccountStateWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper balance_l0 to a wrong value
        cols[COL_BALANCE_L0][0] = Scalar::from_u64(0xDEAD, CurveType::Bls48581);
        let cs = AccountStateConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 1 (balance_limb0_consistency) should be nonzero
        assert!(!results[1][0].is_zero(), "balance limb tamper should be detected");
    }

    #[test]
    fn nonce_post_populated_and_constraint_vanishes_on_honest() {
        let row = sample_row();
        // sample_row uses AccountStateRow::new which sets nonce_post = nonce + 1.
        assert_eq!(row.nonce_post, row.account.nonce + 1);
        let w = AccountStateWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        // COL_NONCE_POST populated to nonce + 1 = 8.
        assert_eq!(trace.columns[COL_NONCE_POST].evaluations[0].to_u64(), 8);
        // The increment body must vanish on honest witness.
        let cs = AccountStateConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Body 5 = nonce_post_increments_nonce.
        for v in &bodies[5] {
            assert!(v.is_zero(), "nonce_post increment must vanish on honest witness");
        }
    }

    #[test]
    fn nonce_post_tamper_fires_increment_constraint() {
        let w = AccountStateWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: set nonce_post to nonce + 2 (i.e. honest pre=7, dishonest post=9).
        cols[COL_NONCE_POST][0] = Scalar::from_u64(9, CurveType::Bls48581);
        let cs = AccountStateConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Body 5 must fire.
        assert!(
            !bodies[5][0].is_zero(),
            "nonce_post increment body should fire on +2 tamper",
        );
        // And on a -1 tamper.
        cols[COL_NONCE_POST][0] = Scalar::from_u64(7, CurveType::Bls48581);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[5][0].is_zero(),
            "nonce_post increment body should fire on equal-to-pre tamper",
        );
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let w = AccountStateWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AccountStateConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(0x4321, CurveType::Bls48581);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = col_refs.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "evaluate_at_point should be zero on honest row");
    }
}
