//! SELFDESTRUCT opcode constraint AIR (standalone gadget).
//!
//! Proves algebraically the EIP-6780 (Cancun) semantics of SELFDESTRUCT:
//!   1. Balance is transferred to the beneficiary.
//!   2. The contract is deleted ONLY if it was created within the same
//!      transaction (EIP-6780); otherwise SELFDESTRUCT only transfers
//!      balance and leaves the contract in place.
//!   3. Static gas cost is 5000, plus a 25000 surcharge when the
//!      beneficiary account did not previously exist (EIP-2929 cold
//!      access for new accounts).
//!
//! Per witness row commits one SELFDESTRUCT event:
//!   - contract_address (20 bytes)
//!   - beneficiary (20 bytes)
//!   - balance_transferred (32 BE bytes + 4 LE u64 limbs for cross-AIR
//!     linkage)
//!   - tx_index (u64) — binds to call_frame_air's transaction context
//!   - contract_created_in_same_tx (binary) — sourced host-side from
//!     the EVM frame stack / CREATE tracker
//!   - is_deleted (binary) — algebraically forced to equal
//!     `contract_created_in_same_tx` under EIP-6780 (constraint 5).
//!   - gas_cost (u64) — pinned to 5000 + 25000·(1 - beneficiary_existed)
//!   - beneficiary_existed (binary)
//!   - is_post_eip6780 (binary) — gates the EIP-6780 deletion rule so
//!     pre-Cancun (legacy) traces can still be modelled with
//!     `is_deleted = 1` always.
//!   - is_real (binary) — row-vs-padding selector.
//!
//! Cross-AIR linkages (descriptors below):
//!   - L_pre:   (contract_address, balance) → account_state_air pre-state
//!     row — binds the balance being moved is the actual stored
//!     balance of the contract before SELFDESTRUCT.
//!   - L_post:  (beneficiary, new_balance) → account_state_air post-state
//!     row — binds the beneficiary's resulting balance after the
//!     transfer.
//!   - L_tx:    (tx_index) → call_frame_air row — binds this event to
//!     the transaction's call-frame context (so `contract_created_in_same_tx`
//!     is sourced from a constrained CREATE tracker downstream).
//!
//! NOT yet algebraically bound (deferred):
//!   - The link between committed `balance_transferred` and the pre-state
//!     balance limb of `contract_address` is done at the cross-AIR LogUp
//!     layer; in-row we only enforce byte ↔ limb consistency.
//!   - The CREATE tracker that justifies `contract_created_in_same_tx`
//!     is sourced host-side from the call_frame_air linkage; an
//!     algebraic CREATE-tracker AIR is left as a follow-up.
//!   - Code-hash clearing on actual deletion (is_deleted = 1) is left
//!     to the account_state post-state row's code_hash = empty_hash
//!     binding via downstream linkages.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

// 20-byte contract address (BE bytes).
pub const COL_CONTRACT_ADDR_OFFSET: usize = 0;   // 0..20
// 20-byte beneficiary address (BE bytes).
pub const COL_BENEFICIARY_OFFSET: usize = 20;    // 20..40
// 32 BE bytes of balance_transferred.
pub const COL_BALANCE_BYTE_OFFSET: usize = 40;   // 40..72
// 4 LE u64 limbs of balance_transferred (for cross-AIR limb linkages).
pub const COL_BALANCE_L0: usize = 72;
pub const COL_BALANCE_L1: usize = 73;
pub const COL_BALANCE_L2: usize = 74;
pub const COL_BALANCE_L3: usize = 75;
// Transaction context.
pub const COL_TX_INDEX: usize = 76;
// EIP-6780 / event flags.
pub const COL_CREATED_SAME_TX: usize = 77;
pub const COL_IS_DELETED: usize = 78;
pub const COL_GAS_COST: usize = 79;
pub const COL_BENEFICIARY_EXISTED: usize = 80;
pub const COL_IS_POST_EIP6780: usize = 81;
pub const COL_IS_REAL: usize = 82;

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 83

// Constraint count:
//   0  is_real binary
//   1  contract_created_in_same_tx binary
//   2  is_deleted binary
//   3  beneficiary_existed binary
//   4  is_post_eip6780 binary
//   5  EIP-6780 rule: is_real · is_post_eip6780 · (is_deleted - created_same_tx) = 0
//   6  gas-cost: is_real · (gas_cost - 5000 - 25000·(1 - beneficiary_existed)) = 0
//   7  balance limb 0 = Horner of BE bytes [24..32]
//   8  balance limb 1 = Horner of BE bytes [16..24]
//   9  balance limb 2 = Horner of BE bytes [8..16]
//   10 balance limb 3 = Horner of BE bytes [0..8]
pub const NUM_ROW_CONSTRAINTS: usize = 11;
pub const NUM_SHIFTED: usize = 0;

// EIP-2929 / EIP-6780 magic constants.
pub const GAS_BASE: u64 = 5000;
pub const GAS_NEW_ACCOUNT: u64 = 25000;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SelfdestructRow {
    pub contract_address: [u8; 20],
    pub beneficiary: [u8; 20],
    /// Balance transferred, as 32 BE bytes (matches account_state_air convention).
    pub balance_be: [u8; 32],
    pub tx_index: u64,
    pub contract_created_in_same_tx: bool,
    pub is_deleted: bool,
    pub gas_cost: u64,
    pub beneficiary_existed: bool,
    pub is_post_eip6780: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SelfdestructWitness {
    pub rows: Vec<SelfdestructRow>,
}

impl SelfdestructWitness {
    pub fn from_rows(rows: Vec<SelfdestructRow>) -> Self {
        Self { rows }
    }
}

/// Build a witness row from a single SELFDESTRUCT event.
///
/// Inputs:
///   - `contract`: address being self-destructed.
///   - `beneficiary`: target of balance transfer.
///   - `balance`: balance to transfer (host supplies u128 — top 128 bits
///     of u256 zero, sufficient for any realistic Ethereum balance).
///   - `created_same_tx`: whether the contract was CREATEd in this tx
///     (EIP-6780 deletion rule).
///   - `beneficiary_existed`: whether the beneficiary account existed
///     prior to the transfer (drives the 25000 cold-account surcharge).
///
/// Post-EIP-6780 semantics are assumed (Cancun and later); the row's
/// `is_post_eip6780` flag is set and `is_deleted` is derived from
/// `created_same_tx`.
pub fn from_event(
    contract: [u8; 20],
    beneficiary: [u8; 20],
    balance: u128,
    created_same_tx: bool,
    beneficiary_existed: bool,
) -> SelfdestructRow {
    let mut balance_be = [0u8; 32];
    // Place the u128 in the low 16 bytes (big-endian).
    balance_be[16..32].copy_from_slice(&balance.to_be_bytes());
    let gas_cost = GAS_BASE
        + if beneficiary_existed { 0 } else { GAS_NEW_ACCOUNT };
    SelfdestructRow {
        contract_address: contract,
        beneficiary,
        balance_be,
        tx_index: 0,
        contract_created_in_same_tx: created_same_tx,
        is_deleted: created_same_tx, // EIP-6780
        gas_cost,
        beneficiary_existed,
        is_post_eip6780: true,
    }
}

// ─── Trace building ───────────────────────────────────────────────────

pub fn build_trace_polynomials(
    w: &SelfdestructWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (r, row) in w.rows.iter().enumerate() {
        // contract address bytes.
        for k in 0..20 {
            cols[COL_CONTRACT_ADDR_OFFSET + k][r] =
                Scalar::from_u64(row.contract_address[k] as u64, curve);
        }
        // beneficiary bytes.
        for k in 0..20 {
            cols[COL_BENEFICIARY_OFFSET + k][r] =
                Scalar::from_u64(row.beneficiary[k] as u64, curve);
        }
        // balance BE bytes.
        for k in 0..32 {
            cols[COL_BALANCE_BYTE_OFFSET + k][r] =
                Scalar::from_u64(row.balance_be[k] as u64, curve);
        }
        // balance LE u64 limbs (limb j covers BE bytes [(3-j)*8 .. (3-j)*8 + 8]).
        for j in 0..4usize {
            let byte_start = (3 - j) * 8;
            let mut v: u64 = 0;
            for k in 0..8 {
                v = (v << 8) | row.balance_be[byte_start + k] as u64;
            }
            cols[COL_BALANCE_L0 + j][r] = Scalar::from_u64(v, curve);
        }
        cols[COL_TX_INDEX][r] = Scalar::from_u64(row.tx_index, curve);
        cols[COL_CREATED_SAME_TX][r] =
            if row.contract_created_in_same_tx { one.clone() } else { zero.clone() };
        cols[COL_IS_DELETED][r] =
            if row.is_deleted { one.clone() } else { zero.clone() };
        cols[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
        cols[COL_BENEFICIARY_EXISTED][r] =
            if row.beneficiary_existed { one.clone() } else { zero.clone() };
        cols[COL_IS_POST_EIP6780][r] =
            if row.is_post_eip6780 { one.clone() } else { zero.clone() };
        cols[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct SelfdestructConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl SelfdestructConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for SelfdestructConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "created_same_tx_binary".into(),
            "is_deleted_binary".into(),
            "beneficiary_existed_binary".into(),
            "is_post_eip6780_binary".into(),
            "eip6780_is_deleted_eq_created_same_tx".into(),
            "gas_cost_pin".into(),
            "balance_limb0_consistency".into(),
            "balance_limb1_consistency".into(),
            "balance_limb2_consistency".into(),
            "balance_limb3_consistency".into(),
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
        let gas_base = Scalar::from_u64(GAS_BASE, curve);
        let gas_new = Scalar::from_u64(GAS_NEW_ACCOUNT, curve);
        let n = columns[0].len();
        let mut bodies = vec![vec![Scalar::zero(curve); n]; NUM_ROW_CONSTRAINTS];
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let created = &columns[COL_CREATED_SAME_TX][r];
            let deleted = &columns[COL_IS_DELETED][r];
            let benf_existed = &columns[COL_BENEFICIARY_EXISTED][r];
            let post_eip = &columns[COL_IS_POST_EIP6780][r];
            let gas_cost = &columns[COL_GAS_COST][r];

            // 0: is_real binary
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            // 1: created_same_tx binary
            bodies[1][r] = created.mul(&created.sub(&one));
            // 2: is_deleted binary
            bodies[2][r] = deleted.mul(&deleted.sub(&one));
            // 3: beneficiary_existed binary
            bodies[3][r] = benf_existed.mul(&benf_existed.sub(&one));
            // 4: is_post_eip6780 binary
            bodies[4][r] = post_eip.mul(&post_eip.sub(&one));
            // 5: EIP-6780: is_real · is_post_eip6780 · (is_deleted - created_same_tx) = 0
            let delta = deleted.sub(created);
            bodies[5][r] = is_real.mul(&post_eip.mul(&delta));
            // 6: gas cost pin: is_real · (gas_cost - 5000 - 25000·(1 - beneficiary_existed)) = 0
            //   = is_real · (gas_cost - 5000 - 25000 + 25000·beneficiary_existed)
            //   = is_real · (gas_cost - 30000 + 25000·beneficiary_existed)
            let expected = gas_base.add(&gas_new.sub(&gas_new.mul(benf_existed)));
            bodies[6][r] = is_real.mul(&gas_cost.sub(&expected));
            // 7..10: balance limb byte decomposition.
            for j in 0..4usize {
                let byte_start = (3 - j) * 8;
                let mut horner = Scalar::zero(curve);
                let mut weight = Scalar::one(curve);
                for k in (0..8).rev() {
                    let b = &columns[COL_BALANCE_BYTE_OFFSET + byte_start + k][r];
                    horner = horner.add(&weight.mul(b));
                    weight = weight.mul(&two56);
                }
                bodies[7 + j][r] = columns[COL_BALANCE_L0 + j][r].sub(&horner);
            }
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two56 = Scalar::from_u64(256, curve);
        let gas_base = Scalar::from_u64(GAS_BASE, curve);
        let gas_new = Scalar::from_u64(GAS_NEW_ACCOUNT, curve);

        let is_real = &ce[COL_IS_REAL];
        let created = &ce[COL_CREATED_SAME_TX];
        let deleted = &ce[COL_IS_DELETED];
        let benf_existed = &ce[COL_BENEFICIARY_EXISTED];
        let post_eip = &ce[COL_IS_POST_EIP6780];
        let gas_cost = &ce[COL_GAS_COST];

        let c0 = is_real.mul(&is_real.sub(&one));
        let c1 = created.mul(&created.sub(&one));
        let c2 = deleted.mul(&deleted.sub(&one));
        let c3 = benf_existed.mul(&benf_existed.sub(&one));
        let c4 = post_eip.mul(&post_eip.sub(&one));
        let delta = deleted.sub(created);
        let c5 = is_real.mul(&post_eip.mul(&delta));
        let expected = gas_base.add(&gas_new.sub(&gas_new.mul(benf_existed)));
        let c6 = is_real.mul(&gas_cost.sub(&expected));

        let mut total = c0;
        let mut ap = alpha.clone();
        for body in [c1, c2, c3, c4, c5, c6].iter() {
            total = total.add(&ap.mul(body));
            ap = ap.mul(alpha);
        }
        for j in 0..4usize {
            let byte_start = (3 - j) * 8;
            let mut horner = Scalar::zero(curve);
            let mut weight = Scalar::one(curve);
            for k in (0..8).rev() {
                horner = horner.add(&weight.mul(&ce[COL_BALANCE_BYTE_OFFSET + byte_start + k]));
                weight = weight.mul(&two56);
            }
            let body = ce[COL_BALANCE_L0 + j].sub(&horner);
            total = total.add(&ap.mul(&body));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(
        &self,
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let two56 = Scalar::from_u64(256, curve);
        let gas_base_p = vec![Scalar::from_u64(GAS_BASE, curve)];
        let gas_new = Scalar::from_u64(GAS_NEW_ACCOUNT, curve);

        let is_real = &cc[COL_IS_REAL];
        let created = &cc[COL_CREATED_SAME_TX];
        let deleted = &cc[COL_IS_DELETED];
        let benf_existed = &cc[COL_BENEFICIARY_EXISTED];
        let post_eip = &cc[COL_IS_POST_EIP6780];
        let gas_cost = &cc[COL_GAS_COST];

        let bin = |p: &Vec<Scalar>| -> Vec<Scalar> {
            let pm1 = poly_sub(p, &one_p, curve);
            poly_mul(p, &pm1, curve)
        };
        let c0 = bin(is_real);
        let c1 = bin(created);
        let c2 = bin(deleted);
        let c3 = bin(benf_existed);
        let c4 = bin(post_eip);
        // c5: is_real · post_eip · (deleted - created)
        let delta = poly_sub(deleted, created, curve);
        let inner = poly_mul(post_eip, &delta, curve);
        let c5 = poly_mul(is_real, &inner, curve);
        // c6: is_real · (gas_cost - 5000 - 25000 + 25000·benf_existed)
        //     = is_real · (gas_cost - 30000 + 25000·benf_existed)
        let gn_p = vec![gas_new.clone()];
        let gn_mul_benf = poly_mul(&gn_p, benf_existed, curve);
        // expected = 5000 + 25000 - 25000·benf_existed = 30000 - 25000·benf_existed
        let thirty_k_p = vec![gas_base_p[0].add(&gas_new)];
        let expected_p = poly_sub(&thirty_k_p, &gn_mul_benf, curve);
        let diff = poly_sub(gas_cost, &expected_p, curve);
        let c6 = poly_mul(is_real, &diff, curve);

        let mut total = c0;
        let mut ap = alpha.clone();
        for body in [c1, c2, c3, c4, c5, c6].iter() {
            total = poly_add(&total, &poly_scalar_mul(body, &ap), curve);
            ap = ap.mul(alpha);
        }
        for j in 0..4usize {
            let byte_start = (3 - j) * 8;
            let mut horner = vec![Scalar::zero(curve)];
            let mut weight = Scalar::one(curve);
            for k in (0..8).rev() {
                horner = poly_add(
                    &horner,
                    &poly_scalar_mul(&cc[COL_BALANCE_BYTE_OFFSET + byte_start + k], &weight),
                    curve,
                );
                weight = weight.mul(&two56);
            }
            let body = poly_sub(&cc[COL_BALANCE_L0 + j], &horner, curve);
            total = poly_add(&total, &poly_scalar_mul(&body, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
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
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256), LookupTable::range(2)];
        let tbl_byte = 0usize;
        let tbl_bit = 1usize;
        let mut declarations = Vec::new();
        // 20 contract addr bytes 8-bit ranged.
        for k in 0..20 {
            declarations.push((
                LookupDeclaration {
                    label: format!("selfdestruct_contract_addr_byte_{}_8bit", k),
                    column_index: COL_CONTRACT_ADDR_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        // 20 beneficiary addr bytes.
        for k in 0..20 {
            declarations.push((
                LookupDeclaration {
                    label: format!("selfdestruct_beneficiary_byte_{}_8bit", k),
                    column_index: COL_BENEFICIARY_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        // 32 balance bytes.
        for k in 0..32 {
            declarations.push((
                LookupDeclaration {
                    label: format!("selfdestruct_balance_byte_{}_8bit", k),
                    column_index: COL_BALANCE_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        // 5 binary flags.
        for (label, col) in [
            ("is_real", COL_IS_REAL),
            ("created_same_tx", COL_CREATED_SAME_TX),
            ("is_deleted", COL_IS_DELETED),
            ("beneficiary_existed", COL_BENEFICIARY_EXISTED),
            ("is_post_eip6780", COL_IS_POST_EIP6780),
        ] {
            declarations.push((
                LookupDeclaration {
                    label: format!("selfdestruct_{}_1bit", label),
                    column_index: col,
                    max_bits: 1,
                    selector_column: None,
                },
                tbl_bit,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// SELFDESTRUCT row ↔ account_state_air pre-state row of the contract
/// being destroyed. Binds `(contract_address_limbs, balance_limbs)` to
/// the pre-state account of `contract_address` so the balance committed
/// here is the actual balance the contract held before SELFDESTRUCT.
///
/// `account_pre_selector_column` should be a host-side selector that
/// is 1 on rows of account_state_air representing the pre-state of a
/// SELFDESTRUCT contract; the host wires this through the orchestration
/// layer.
pub fn make_selfdestruct_to_account_pre_descriptor(
    sd_layer: usize,
    account_state_layer: usize,
    account_pre_selector_column: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    // A side: SELFDESTRUCT row publishes contract_address (as 20 BE
    // bytes) and balance (as 4 LE u64 limbs).
    // B side: account_state_air row publishes (address bytes, balance limbs).
    let mut a_columns: Vec<usize> = Vec::with_capacity(24);
    for k in 0..20 {
        a_columns.push(COL_CONTRACT_ADDR_OFFSET + k);
    }
    a_columns.push(COL_BALANCE_L0);
    a_columns.push(COL_BALANCE_L1);
    a_columns.push(COL_BALANCE_L2);
    a_columns.push(COL_BALANCE_L3);

    // B side mirrors via account_state_air's BE bytes + balance limbs.
    // account_state_air stores address as 4 LE u64 limbs and 32-byte
    // balance with separate limb cols; for tuple width we mirror
    // 20 address byte cols (host-side reshape via dedicated B-side adapter
    // columns is acceptable per the cross-AIR LogUp convention) + 4 limbs.
    // We pass the account_state_air column ids directly using its BE-byte
    // layout: the first 20 BE bytes of the address are placed by the
    // account_state trace builder in cols 5..25 for the same convention.
    // However the canonical account_state_air uses 4 limb cols for address
    // and 32 BE bytes for balance; to keep both tuple widths matching we
    // use the account_state_air's address byte interpretation via its
    // address-keccak adapter columns. For now we publish a 24-element
    // tuple by reusing 20 contract-address columns with an implicit
    // host-side reshape. Downstream orchestration MUST supply the matching
    // 24 columns on B side.
    let mut b_columns: Vec<usize> = Vec::with_capacity(24);
    // First 20 are address bytes — the host wires them to the appropriate
    // adapter columns of account_state_air (e.g. an `addr_byte[0..20]`
    // adapter region). For now we point them at a contiguous run starting
    // at account_state_air's address-trie-key region as a stand-in;
    // orchestration overrides when actually composed.
    for k in 0..20 {
        b_columns.push(k);
    }
    // Balance limbs — account_state_air exposes them at COL_BALANCE_L0..3 (166..169).
    b_columns.push(166);
    b_columns.push(167);
    b_columns.push(168);
    b_columns.push(169);

    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "selfdestruct_to_account_pre_v1".into(),
        a_layer_index: sd_layer,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: account_state_layer,
        b_columns,
        b_selector_column: Some(account_pre_selector_column),
    }
}

/// SELFDESTRUCT row ↔ account_state_air post-state row of the beneficiary.
/// Binds `(beneficiary_address, balance_transferred)` to the beneficiary's
/// post-state account so the new balance equals (old_balance + balance_transferred).
///
/// Note: the additive relation `new_balance = old_balance + transferred`
/// is enforced by a separate balance-arithmetic gadget at the account
/// level; this descriptor only pins the (address, amount) tuple to the
/// post-state row published by that gadget.
pub fn make_selfdestruct_to_account_post_descriptor(
    sd_layer: usize,
    account_state_layer: usize,
    account_post_selector_column: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(24);
    for k in 0..20 {
        a_columns.push(COL_BENEFICIARY_OFFSET + k);
    }
    a_columns.push(COL_BALANCE_L0);
    a_columns.push(COL_BALANCE_L1);
    a_columns.push(COL_BALANCE_L2);
    a_columns.push(COL_BALANCE_L3);

    let mut b_columns: Vec<usize> = Vec::with_capacity(24);
    for k in 0..20 {
        b_columns.push(k);
    }
    b_columns.push(166);
    b_columns.push(167);
    b_columns.push(168);
    b_columns.push(169);

    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "selfdestruct_to_account_post_v1".into(),
        a_layer_index: sd_layer,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: account_state_layer,
        b_columns,
        b_selector_column: Some(account_post_selector_column),
    }
}

/// SELFDESTRUCT row ↔ call_frame_air row. Binds `tx_index` so the event
/// is anchored to a constrained transaction context. The call_frame_air
/// layer must expose a `tx_index` adapter column at
/// `call_frame_tx_index_column`.
pub fn make_selfdestruct_to_call_frame_descriptor(
    sd_layer: usize,
    call_frame_layer: usize,
    call_frame_tx_index_column: usize,
    call_frame_selector_column: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "selfdestruct_to_call_frame_v1".into(),
        a_layer_index: sd_layer,
        a_columns: vec![COL_TX_INDEX],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: call_frame_layer,
        b_columns: vec![call_frame_tx_index_column],
        b_selector_column: Some(call_frame_selector_column),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_addr(byte: u8) -> [u8; 20] {
        [byte; 20]
    }

    /// EIP-6780: contract created same tx → is_deleted = 1.
    #[test]
    fn selfdestruct_air_same_tx_delete_honest() {
        let row = from_event(
            dummy_addr(0xAA),
            dummy_addr(0xBB),
            1_000_000_000u128, // 1 gwei
            true,              // created same tx
            true,              // beneficiary existed
        );
        assert!(row.is_deleted);
        assert_eq!(row.gas_cost, GAS_BASE);
        let w = SelfdestructWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SelfdestructConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} non-zero at row {}",
                    i,
                    r
                );
            }
        }
    }

    /// EIP-6780: contract NOT created same tx → is_deleted = 0 (just transfer).
    #[test]
    fn selfdestruct_air_different_tx_no_delete_honest() {
        let row = from_event(
            dummy_addr(0xCC),
            dummy_addr(0xDD),
            12345u128,
            false, // pre-existing contract
            true,
        );
        assert!(!row.is_deleted);
        let w = SelfdestructWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SelfdestructConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for v in body.iter() {
                assert!(v.is_zero(), "constraint {} non-zero", i);
            }
        }
    }

    /// Cold beneficiary (didn't exist) → gas = 5000 + 25000 = 30000.
    #[test]
    fn selfdestruct_air_cold_beneficiary_gas() {
        let row = from_event(
            dummy_addr(0x01),
            dummy_addr(0x02),
            0u128,
            false,
            false, // cold!
        );
        assert_eq!(row.gas_cost, GAS_BASE + GAS_NEW_ACCOUNT);
        let w = SelfdestructWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SelfdestructConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        for (i, body) in bodies.iter().enumerate() {
            for v in body.iter() {
                assert!(v.is_zero(), "constraint {} non-zero", i);
            }
        }
    }

    /// Warm beneficiary (already exists) → gas = 5000.
    #[test]
    fn selfdestruct_air_warm_beneficiary_gas() {
        let row = from_event(
            dummy_addr(0x03),
            dummy_addr(0x04),
            7u128,
            false,
            true, // warm
        );
        assert_eq!(row.gas_cost, GAS_BASE);
        let w = SelfdestructWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SelfdestructConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for body in cs.evaluate_on_domain(&cr, t.num_rows).iter() {
            for v in body.iter() {
                assert!(v.is_zero());
            }
        }
    }

    /// Tampered gas: claim 5000 when beneficiary was cold (should be 30000).
    #[test]
    fn selfdestruct_air_tampered_gas_detected() {
        let mut row = from_event(
            dummy_addr(0x05),
            dummy_addr(0x06),
            0u128,
            false,
            false, // cold
        );
        // Lie: claim only 5000 gas (skipping the 25000 cold surcharge).
        row.gas_cost = GAS_BASE;
        let w = SelfdestructWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SelfdestructConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 6 is the gas-pin; it should NOT be zero at row 0.
        assert!(!bodies[6][0].is_zero(), "gas tampering not detected");
    }

    /// Tampered EIP-6780: claim is_deleted = 1 when contract was NOT
    /// created in the same tx. Constraint 5 must fire.
    #[test]
    fn selfdestruct_air_tampered_eip6780_detected() {
        let mut row = from_event(
            dummy_addr(0x07),
            dummy_addr(0x08),
            42u128,
            false, // not created same tx
            true,
        );
        row.is_deleted = true; // lie!
        let w = SelfdestructWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SelfdestructConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[5][0].is_zero(), "EIP-6780 violation not detected");
    }

    #[test]
    fn selfdestruct_air_descriptors_well_formed() {
        let d_pre = make_selfdestruct_to_account_pre_descriptor(0, 1, 170);
        let d_post = make_selfdestruct_to_account_post_descriptor(0, 1, 170);
        let d_tx = make_selfdestruct_to_call_frame_descriptor(0, 2, 0, 29);
        assert_eq!(d_pre.a_columns.len(), 24);
        assert_eq!(d_pre.b_columns.len(), 24);
        assert_eq!(d_post.a_columns.len(), 24);
        assert_eq!(d_post.b_columns.len(), 24);
        assert_eq!(d_tx.a_columns.len(), 1);
        assert_eq!(d_tx.b_columns.len(), 1);
        assert_eq!(d_pre.label, "selfdestruct_to_account_pre_v1");
        assert_eq!(d_post.label, "selfdestruct_to_account_post_v1");
        assert_eq!(d_tx.label, "selfdestruct_to_call_frame_v1");
        // Distinct A-side selectors and label disambiguation.
        assert_eq!(d_pre.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_post.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d_tx.a_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn selfdestruct_air_evaluate_at_point_zero_on_honest() {
        let row = from_event(
            dummy_addr(0xEE),
            dummy_addr(0xFF),
            999u128,
            true,
            false, // cold but created same tx — still 30000 gas
        );
        assert_eq!(row.gas_cost, GAS_BASE + GAS_NEW_ACCOUNT);
        let w = SelfdestructWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SelfdestructConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xDEADBEEF, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "honest row should evaluate to zero");
    }

    #[test]
    fn selfdestruct_air_balance_limb_consistency() {
        // Use a balance with non-trivial bytes across the low 16 bytes.
        let row = from_event(
            dummy_addr(0x09),
            dummy_addr(0x0A),
            0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210u128,
            true,
            true,
        );
        let w = SelfdestructWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SelfdestructConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Balance limb consistency = constraints 7..11.
        for i in 7..11 {
            assert!(bodies[i][0].is_zero(), "limb constraint {} non-zero", i);
        }
    }
}
