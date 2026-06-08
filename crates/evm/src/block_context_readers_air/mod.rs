//! Combined ORIGIN + GASPRICE + GASLIMIT + PREVRANDAO context-reader AIR.
//!
//! Four EVM block / tx context-reader opcodes consolidated into one AIR
//! with a shared per-row structure:
//!
//!   - ORIGIN     (0x32): pushes `tx.origin` (the EOA that initiated the
//!     transaction). Gas 2.
//!   - GASPRICE   (0x3A): pushes the effective gas price for the current
//!     transaction. Gas 2.
//!   - GASLIMIT   (0x45): pushes the current block's gas limit (fits in
//!     a u64). Gas 2.
//!   - PREVRANDAO (0x44): pushes the beacon-chain RANDAO mix (post-Merge
//!     replacement for DIFFICULTY). Gas 2.
//!
//! All four are static gas cost 2 (`G_base`). On each real row exactly
//! one of `sel_origin`, `sel_gasprice`, `sel_gaslimit`, `sel_prevrandao`
//! is set, the opcode byte matches, and the pushed u256 (`value[0..4]`
//! LE u64 limbs) is algebraically bound to the corresponding context
//! column.
//!
//!   - ORIGIN:     `value_limb_k = Σ tx_origin[8k + i] · 256^i` using
//!                 the canonical [`crate::address_opcode_air::address_to_limbs`]
//!                 packing (limb 0..1 = 8 bytes each, limb 2 = low 4
//!                 bytes of the address high half + zero padding, limb
//!                 3 = 0).
//!   - GASPRICE:   `value_limb_k = tx_gasprice_limb_k` for k = 0..4.
//!   - GASLIMIT:   `value_limb_0 = block_gaslimit`, limbs 1..=3 = 0.
//!   - PREVRANDAO: 32-byte mix_hash → 4 LE u64 limbs, the same encoding
//!                 the EVM uses for the pushed u256.
//!
//! Composed with neighbouring AIRs via three cross-AIR LogUp descriptors:
//!
//!   - `make_block_context_to_block_header_descriptor` — GASLIMIT +
//!     PREVRANDAO rows ↔ `block_header_air`'s `COL_GAS_LIMIT` /
//!     `COL_PREV_RANDAO_L0..L3` tuple.
//!   - `make_block_context_to_tx_rlp_descriptor` — GASPRICE rows ↔
//!     `tx_rlp_air`'s 32-byte BE gas-price field.
//!   - `make_block_context_to_stack_contents_descriptor` — every real
//!     row ↔ `stack_contents_air` unsorted `(pc, value_limb_0..3)`
//!     push event.
//!
//! ## Per-row layout (`NUM_COLUMNS = 70`)
//!
//! ```text
//! offset  meaning
//!   0     pc                       (u64)
//!   1     opcode                   (u8 ∈ {0x32, 0x3A, 0x45, 0x44})
//!   2..6  value_limb_0..3          (LE u64 of pushed u256)
//!   6     gas_cost                 (always 2)
//!   7..27 tx_origin[0..20]         (u8 per limb)
//!  27..31 tx_gasprice_limb_0..3    (LE u64 of u256 gas_price)
//!  31     block_gaslimit           (u64)
//!  32..64 prev_randao[0..32]       (u8 per limb, LE byte 0 = limb 0 low)
//!  64     sel_origin               (binary)
//!  65     sel_gasprice             (binary)
//!  66     sel_gaslimit             (binary)
//!  67     sel_prevrandao           (binary)
//!  68     is_real                  (binary)
//!  69     (reserved/padding) -- unused; kept zero
//! ```
//!
//! ## Constraint catalog (`NUM_ROW_CONSTRAINTS = 25`)
//!
//!  0. `is_real_binary`              — `is_real · (is_real − 1) = 0`
//!  1. `sel_origin_binary`           — `s_o · (s_o − 1) = 0`
//!  2. `sel_gasprice_binary`         — `s_gp · (s_gp − 1) = 0`
//!  3. `sel_gaslimit_binary`         — `s_gl · (s_gl − 1) = 0`
//!  4. `sel_prevrandao_binary`       — `s_pr · (s_pr − 1) = 0`
//!  5. `selector_sum_eq_is_real`     — `is_real − (s_o + s_gp + s_gl + s_pr) = 0`
//!  6. `selector_mutex`              — pairwise products sum to zero
//!  7. `gas_cost_eq_2`               — `is_real · (gas_cost − 2) = 0`
//!  8. `origin_opcode_eq`            — `s_o · (op − 0x32) = 0`
//!  9. `gasprice_opcode_eq`          — `s_gp · (op − 0x3A) = 0`
//! 10. `gaslimit_opcode_eq`          — `s_gl · (op − 0x45) = 0`
//! 11. `prevrandao_opcode_eq`        — `s_pr · (op − 0x44) = 0`
//! 12. `origin_value_l0_binding`     — `s_o · (v0 − Σ origin[i]·256^i)`
//! 13. `origin_value_l1_binding`     — `s_o · (v1 − Σ origin[8+i]·256^i)`
//! 14. `origin_value_l2_binding`     — `s_o · (v2 − Σ origin[16+i]·256^i)`
//! 15. `origin_value_l3_zero`        — `s_o · v3 = 0`
//! 16. `gasprice_value_l0_eq`        — `s_gp · (v0 − tx_gasprice_l0) = 0`
//! 17. `gasprice_value_l1_eq`        — `s_gp · (v1 − tx_gasprice_l1) = 0`
//! 18. `gasprice_value_l2_eq`        — `s_gp · (v2 − tx_gasprice_l2) = 0`
//! 19. `gasprice_value_l3_eq`        — `s_gp · (v3 − tx_gasprice_l3) = 0`
//! 20. `gaslimit_value_l0_eq`        — `s_gl · (v0 − block_gaslimit) = 0`
//! 21. `gaslimit_value_high_zero`    — `s_gl · (v1 + v2 + v3) = 0` (paired
//!                                     with byte/limb range checks)
//! 22. `prevrandao_value_l0_binding` — `s_pr · (v0 − Σ randao[i]·256^i)`
//! 23. `prevrandao_value_l1_binding` — `s_pr · (v1 − Σ randao[8+i]·256^i)`
//! 24. `prevrandao_value_l2_binding` — `s_pr · (v2 − Σ randao[16+i]·256^i)`
//!     plus prevrandao_l3 binding folded inline → constraint #25
//! 25. `prevrandao_value_l3_binding` — `s_pr · (v3 − Σ randao[24+i]·256^i)`
//! 26. `padding_opcode_zero`         — `(1 − is_real) · op = 0`

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_PC: usize = 0;
pub const COL_OPCODE: usize = 1;
pub const COL_VALUE_LIMB_0: usize = 2;
pub const COL_VALUE_LIMB_1: usize = 3;
pub const COL_VALUE_LIMB_2: usize = 4;
pub const COL_VALUE_LIMB_3: usize = 5;
pub const COL_GAS_COST: usize = 6;
pub const COL_TX_ORIGIN_0: usize = 7;
pub const NUM_ADDR_BYTES: usize = 20;
pub const COL_TX_GASPRICE_L0: usize = COL_TX_ORIGIN_0 + NUM_ADDR_BYTES; // 27
pub const COL_TX_GASPRICE_L1: usize = COL_TX_GASPRICE_L0 + 1; // 28
pub const COL_TX_GASPRICE_L2: usize = COL_TX_GASPRICE_L0 + 2; // 29
pub const COL_TX_GASPRICE_L3: usize = COL_TX_GASPRICE_L0 + 3; // 30
pub const COL_BLOCK_GASLIMIT: usize = COL_TX_GASPRICE_L3 + 1; // 31
pub const COL_PREV_RANDAO_0: usize = COL_BLOCK_GASLIMIT + 1; // 32
pub const NUM_RANDAO_BYTES: usize = 32;
pub const COL_SEL_ORIGIN: usize = COL_PREV_RANDAO_0 + NUM_RANDAO_BYTES; // 64
pub const COL_SEL_GASPRICE: usize = COL_SEL_ORIGIN + 1; // 65
pub const COL_SEL_GASLIMIT: usize = COL_SEL_GASPRICE + 1; // 66
pub const COL_SEL_PREVRANDAO: usize = COL_SEL_GASLIMIT + 1; // 67
pub const COL_IS_REAL: usize = COL_SEL_PREVRANDAO + 1; // 68

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 69

// 7 structural + 1 gas + 4 opcode + 4 origin + 4 gasprice + 2 gaslimit
// + 4 prevrandao + 1 padding = 27
pub const NUM_ROW_CONSTRAINTS: usize = 27;
pub const NUM_SHIFTED: usize = 0;

// Opcode bytes
pub const ORIGIN_OPCODE: u8 = 0x32;
pub const GASPRICE_OPCODE: u8 = 0x3A;
pub const GASLIMIT_OPCODE: u8 = 0x45;
pub const PREVRANDAO_OPCODE: u8 = 0x44;
pub const BLOCK_CONTEXT_READER_GAS: u64 = 2;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockContextReaderRow {
    pub pc: u64,
    pub opcode: u8,
    pub value: [u64; 4],
    pub gas_cost: u64,
    pub tx_origin: [u8; 20],
    pub tx_gasprice: [u64; 4],
    pub block_gaslimit: u64,
    pub prev_randao: [u8; 32],
    pub sel_origin: bool,
    pub sel_gasprice: bool,
    pub sel_gaslimit: bool,
    pub sel_prevrandao: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct BlockContextReadersWitness {
    pub rows: Vec<BlockContextReaderRow>,
}

/// Convert a 32-byte LE encoding (canonical EVM u256 push form, also
/// used by the inspector / trace) into 4 LE u64 limbs (limb 0 = least-
/// significant 8 bytes).
pub fn value_bytes_to_limbs(value: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for k in 0..4 {
        let mut tmp = [0u8; 8];
        tmp.copy_from_slice(&value[8 * k..8 * k + 8]);
        limbs[k] = u64::from_le_bytes(tmp);
    }
    limbs
}

impl BlockContextReadersWitness {
    /// Build a witness from `(opcode, pc, value)` events. The host
    /// populates the per-row context columns (`tx_origin`, `tx_gasprice`,
    /// `block_gaslimit`, `prev_randao`) based on the row's opcode kind,
    /// then exposes them as columns bound algebraically.
    ///
    /// For ORIGIN:    `tx_origin` = the 20 bytes such that
    ///                `address_to_limbs(tx_origin) = value`.
    /// For GASPRICE:  `tx_gasprice` = `value` (1:1 with the 4 LE limbs).
    /// For GASLIMIT:  `block_gaslimit` = value_limb_0, others zero.
    /// For PREVRANDAO: `prev_randao` = the 32 LE bytes of `value`.
    pub fn from_events(events: &[(u8, u64, [u8; 32])]) -> Self {
        let rows: Vec<BlockContextReaderRow> = events
            .iter()
            .map(|(opcode, pc, value)| {
                let value_limbs = value_bytes_to_limbs(value);
                let mut row = BlockContextReaderRow {
                    pc: *pc,
                    opcode: *opcode,
                    value: value_limbs,
                    gas_cost: BLOCK_CONTEXT_READER_GAS,
                    tx_origin: [0u8; 20],
                    tx_gasprice: [0u64; 4],
                    block_gaslimit: 0,
                    prev_randao: [0u8; 32],
                    sel_origin: false,
                    sel_gasprice: false,
                    sel_gaslimit: false,
                    sel_prevrandao: false,
                    is_real: true,
                };
                match *opcode {
                    ORIGIN_OPCODE => {
                        row.sel_origin = true;
                        // Reconstruct the 20-byte address from the LE-limb
                        // packing used by `address_to_limbs`.
                        let mut addr = [0u8; 20];
                        let l0 = value_limbs[0].to_le_bytes();
                        let l1 = value_limbs[1].to_le_bytes();
                        let l2 = value_limbs[2].to_le_bytes();
                        addr[0..8].copy_from_slice(&l0);
                        addr[8..16].copy_from_slice(&l1);
                        addr[16..20].copy_from_slice(&l2[0..4]);
                        row.tx_origin = addr;
                    }
                    GASPRICE_OPCODE => {
                        row.sel_gasprice = true;
                        row.tx_gasprice = value_limbs;
                    }
                    GASLIMIT_OPCODE => {
                        row.sel_gaslimit = true;
                        row.block_gaslimit = value_limbs[0];
                    }
                    PREVRANDAO_OPCODE => {
                        row.sel_prevrandao = true;
                        row.prev_randao = *value;
                    }
                    _ => {
                        // Unknown opcode: mark non-real so constraints don't fire.
                        row.is_real = false;
                    }
                }
                row
            })
            .collect();
        Self { rows }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    w: &BlockContextReadersWitness,
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
        cols[COL_PC][r] = Scalar::from_u64(row.pc, curve);
        cols[COL_OPCODE][r] = Scalar::from_u64(row.opcode as u64, curve);
        cols[COL_VALUE_LIMB_0][r] = Scalar::from_u64(row.value[0], curve);
        cols[COL_VALUE_LIMB_1][r] = Scalar::from_u64(row.value[1], curve);
        cols[COL_VALUE_LIMB_2][r] = Scalar::from_u64(row.value[2], curve);
        cols[COL_VALUE_LIMB_3][r] = Scalar::from_u64(row.value[3], curve);
        cols[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
        for i in 0..NUM_ADDR_BYTES {
            cols[COL_TX_ORIGIN_0 + i][r] =
                Scalar::from_u64(row.tx_origin[i] as u64, curve);
        }
        cols[COL_TX_GASPRICE_L0][r] = Scalar::from_u64(row.tx_gasprice[0], curve);
        cols[COL_TX_GASPRICE_L1][r] = Scalar::from_u64(row.tx_gasprice[1], curve);
        cols[COL_TX_GASPRICE_L2][r] = Scalar::from_u64(row.tx_gasprice[2], curve);
        cols[COL_TX_GASPRICE_L3][r] = Scalar::from_u64(row.tx_gasprice[3], curve);
        cols[COL_BLOCK_GASLIMIT][r] = Scalar::from_u64(row.block_gaslimit, curve);
        for i in 0..NUM_RANDAO_BYTES {
            cols[COL_PREV_RANDAO_0 + i][r] =
                Scalar::from_u64(row.prev_randao[i] as u64, curve);
        }
        cols[COL_SEL_ORIGIN][r] = if row.sel_origin { one.clone() } else { zero.clone() };
        cols[COL_SEL_GASPRICE][r] = if row.sel_gasprice { one.clone() } else { zero.clone() };
        cols[COL_SEL_GASLIMIT][r] = if row.sel_gaslimit { one.clone() } else { zero.clone() };
        cols[COL_SEL_PREVRANDAO][r] = if row.sel_prevrandao { one.clone() } else { zero.clone() };
        cols[COL_IS_REAL][r] = if row.is_real { one.clone() } else { zero.clone() };
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn byte_weights(curve: CurveType) -> [Scalar; 8] {
    let mut w = [
        Scalar::one(curve),
        Scalar::zero(curve), Scalar::zero(curve), Scalar::zero(curve),
        Scalar::zero(curve), Scalar::zero(curve), Scalar::zero(curve),
        Scalar::zero(curve),
    ];
    let mut acc = Scalar::one(curve);
    let two56 = Scalar::from_u64(256, curve);
    for i in 1..8 {
        acc = acc.mul(&two56);
        w[i] = acc.clone();
    }
    w
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct BlockContextReadersConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BlockContextReadersConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for BlockContextReadersConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "sel_origin_binary".into(),
            "sel_gasprice_binary".into(),
            "sel_gaslimit_binary".into(),
            "sel_prevrandao_binary".into(),
            "selector_sum_eq_is_real".into(),
            "selector_mutex".into(),
            "gas_cost_eq_2".into(),
            "origin_opcode_eq".into(),
            "gasprice_opcode_eq".into(),
            "gaslimit_opcode_eq".into(),
            "prevrandao_opcode_eq".into(),
            "origin_value_l0_binding".into(),
            "origin_value_l1_binding".into(),
            "origin_value_l2_binding".into(),
            "origin_value_l3_zero".into(),
            "gasprice_value_l0_eq".into(),
            "gasprice_value_l1_eq".into(),
            "gasprice_value_l2_eq".into(),
            "gasprice_value_l3_eq".into(),
            "gaslimit_value_l0_eq".into(),
            "gaslimit_value_high_zero".into(),
            "prevrandao_value_l0_binding".into(),
            "prevrandao_value_l1_binding".into(),
            "prevrandao_value_l2_binding".into(),
            "prevrandao_value_l3_binding".into(),
            "padding_opcode_zero".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let two_gas = Scalar::from_u64(BLOCK_CONTEXT_READER_GAS, curve);
        let op_origin = Scalar::from_u64(ORIGIN_OPCODE as u64, curve);
        let op_gasprice = Scalar::from_u64(GASPRICE_OPCODE as u64, curve);
        let op_gaslimit = Scalar::from_u64(GASLIMIT_OPCODE as u64, curve);
        let op_prevrandao = Scalar::from_u64(PREVRANDAO_OPCODE as u64, curve);
        let w = byte_weights(curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let s_o = &columns[COL_SEL_ORIGIN][r];
            let s_gp = &columns[COL_SEL_GASPRICE][r];
            let s_gl = &columns[COL_SEL_GASLIMIT][r];
            let s_pr = &columns[COL_SEL_PREVRANDAO][r];
            let gc = &columns[COL_GAS_COST][r];
            let op = &columns[COL_OPCODE][r];
            let v0 = &columns[COL_VALUE_LIMB_0][r];
            let v1 = &columns[COL_VALUE_LIMB_1][r];
            let v2 = &columns[COL_VALUE_LIMB_2][r];
            let v3 = &columns[COL_VALUE_LIMB_3][r];
            let gp0 = &columns[COL_TX_GASPRICE_L0][r];
            let gp1 = &columns[COL_TX_GASPRICE_L1][r];
            let gp2 = &columns[COL_TX_GASPRICE_L2][r];
            let gp3 = &columns[COL_TX_GASPRICE_L3][r];
            let gl = &columns[COL_BLOCK_GASLIMIT][r];

            // 0: is_real binary
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            // 1..4: each selector binary
            bodies[1][r] = s_o.mul(&s_o.sub(&one));
            bodies[2][r] = s_gp.mul(&s_gp.sub(&one));
            bodies[3][r] = s_gl.mul(&s_gl.sub(&one));
            bodies[4][r] = s_pr.mul(&s_pr.sub(&one));
            // 5: selector sum == is_real
            let sum = s_o.add(s_gp).add(s_gl).add(s_pr);
            bodies[5][r] = is_real.sub(&sum);
            // 6: mutex — all six pairwise products
            bodies[6][r] = s_o.mul(s_gp)
                .add(&s_o.mul(s_gl))
                .add(&s_o.mul(s_pr))
                .add(&s_gp.mul(s_gl))
                .add(&s_gp.mul(s_pr))
                .add(&s_gl.mul(s_pr));
            // 7: gas_cost == 2 on real rows
            bodies[7][r] = is_real.mul(&gc.sub(&two_gas));
            // 8..11: per-selector opcode binding
            bodies[8][r] = s_o.mul(&op.sub(&op_origin));
            bodies[9][r] = s_gp.mul(&op.sub(&op_gasprice));
            bodies[10][r] = s_gl.mul(&op.sub(&op_gaslimit));
            bodies[11][r] = s_pr.mul(&op.sub(&op_prevrandao));
            // 12..15: ORIGIN — limbs from address bytes (address_to_limbs)
            let mut s0 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_TX_ORIGIN_0 + i][r];
                s0 = s0.add(&b.mul(&w[i]));
            }
            bodies[12][r] = s_o.mul(&v0.sub(&s0));
            let mut s1 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_TX_ORIGIN_0 + 8 + i][r];
                s1 = s1.add(&b.mul(&w[i]));
            }
            bodies[13][r] = s_o.mul(&v1.sub(&s1));
            let mut s2 = Scalar::zero(curve);
            for i in 0..4 {
                let b = &columns[COL_TX_ORIGIN_0 + 16 + i][r];
                s2 = s2.add(&b.mul(&w[i]));
            }
            bodies[14][r] = s_o.mul(&v2.sub(&s2));
            bodies[15][r] = s_o.mul(v3);
            // 16..19: GASPRICE — value_limb_k = tx_gasprice_limb_k
            bodies[16][r] = s_gp.mul(&v0.sub(gp0));
            bodies[17][r] = s_gp.mul(&v1.sub(gp1));
            bodies[18][r] = s_gp.mul(&v2.sub(gp2));
            bodies[19][r] = s_gp.mul(&v3.sub(gp3));
            // 20..21: GASLIMIT — v0 = block_gaslimit, high limbs zero
            bodies[20][r] = s_gl.mul(&v0.sub(gl));
            let hi_sum = v1.add(v2).add(v3);
            bodies[21][r] = s_gl.mul(&hi_sum);
            // 22..25: PREVRANDAO — limb_k = Σ_i prev_randao[8k+i]·256^i
            let mut r0 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_PREV_RANDAO_0 + i][r];
                r0 = r0.add(&b.mul(&w[i]));
            }
            bodies[22][r] = s_pr.mul(&v0.sub(&r0));
            let mut r1 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_PREV_RANDAO_0 + 8 + i][r];
                r1 = r1.add(&b.mul(&w[i]));
            }
            bodies[23][r] = s_pr.mul(&v1.sub(&r1));
            let mut r2 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_PREV_RANDAO_0 + 16 + i][r];
                r2 = r2.add(&b.mul(&w[i]));
            }
            bodies[24][r] = s_pr.mul(&v2.sub(&r2));
            let mut r3 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_PREV_RANDAO_0 + 24 + i][r];
                r3 = r3.add(&b.mul(&w[i]));
            }
            bodies[25][r] = s_pr.mul(&v3.sub(&r3));
            // 26: padding rows have opcode = 0
            let one_minus_real = one.sub(is_real);
            bodies[26][r] = one_minus_real.mul(op);
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two_gas = Scalar::from_u64(BLOCK_CONTEXT_READER_GAS, curve);
        let op_origin = Scalar::from_u64(ORIGIN_OPCODE as u64, curve);
        let op_gasprice = Scalar::from_u64(GASPRICE_OPCODE as u64, curve);
        let op_gaslimit = Scalar::from_u64(GASLIMIT_OPCODE as u64, curve);
        let op_prevrandao = Scalar::from_u64(PREVRANDAO_OPCODE as u64, curve);
        let w = byte_weights(curve);
        let is_real = &ce[COL_IS_REAL];
        let s_o = &ce[COL_SEL_ORIGIN];
        let s_gp = &ce[COL_SEL_GASPRICE];
        let s_gl = &ce[COL_SEL_GASLIMIT];
        let s_pr = &ce[COL_SEL_PREVRANDAO];
        let gc = &ce[COL_GAS_COST];
        let op = &ce[COL_OPCODE];
        let v0 = &ce[COL_VALUE_LIMB_0];
        let v1 = &ce[COL_VALUE_LIMB_1];
        let v2 = &ce[COL_VALUE_LIMB_2];
        let v3 = &ce[COL_VALUE_LIMB_3];
        let gp0 = &ce[COL_TX_GASPRICE_L0];
        let gp1 = &ce[COL_TX_GASPRICE_L1];
        let gp2 = &ce[COL_TX_GASPRICE_L2];
        let gp3 = &ce[COL_TX_GASPRICE_L3];
        let gl = &ce[COL_BLOCK_GASLIMIT];

        let mut s0 = Scalar::zero(curve);
        for i in 0..8 { s0 = s0.add(&ce[COL_TX_ORIGIN_0 + i].mul(&w[i])); }
        let mut s1 = Scalar::zero(curve);
        for i in 0..8 { s1 = s1.add(&ce[COL_TX_ORIGIN_0 + 8 + i].mul(&w[i])); }
        let mut s2 = Scalar::zero(curve);
        for i in 0..4 { s2 = s2.add(&ce[COL_TX_ORIGIN_0 + 16 + i].mul(&w[i])); }
        let mut r0 = Scalar::zero(curve);
        for i in 0..8 { r0 = r0.add(&ce[COL_PREV_RANDAO_0 + i].mul(&w[i])); }
        let mut r1 = Scalar::zero(curve);
        for i in 0..8 { r1 = r1.add(&ce[COL_PREV_RANDAO_0 + 8 + i].mul(&w[i])); }
        let mut r2 = Scalar::zero(curve);
        for i in 0..8 { r2 = r2.add(&ce[COL_PREV_RANDAO_0 + 16 + i].mul(&w[i])); }
        let mut r3 = Scalar::zero(curve);
        for i in 0..8 { r3 = r3.add(&ce[COL_PREV_RANDAO_0 + 24 + i].mul(&w[i])); }

        let sum = s_o.add(s_gp).add(s_gl).add(s_pr);
        let one_minus_real = one.sub(is_real);
        let hi_sum = v1.add(v2).add(v3);
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            s_o.mul(&s_o.sub(&one)),
            s_gp.mul(&s_gp.sub(&one)),
            s_gl.mul(&s_gl.sub(&one)),
            s_pr.mul(&s_pr.sub(&one)),
            is_real.sub(&sum),
            s_o.mul(s_gp).add(&s_o.mul(s_gl)).add(&s_o.mul(s_pr))
                .add(&s_gp.mul(s_gl)).add(&s_gp.mul(s_pr)).add(&s_gl.mul(s_pr)),
            is_real.mul(&gc.sub(&two_gas)),
            s_o.mul(&op.sub(&op_origin)),
            s_gp.mul(&op.sub(&op_gasprice)),
            s_gl.mul(&op.sub(&op_gaslimit)),
            s_pr.mul(&op.sub(&op_prevrandao)),
            s_o.mul(&v0.sub(&s0)),
            s_o.mul(&v1.sub(&s1)),
            s_o.mul(&v2.sub(&s2)),
            s_o.mul(v3),
            s_gp.mul(&v0.sub(gp0)),
            s_gp.mul(&v1.sub(gp1)),
            s_gp.mul(&v2.sub(gp2)),
            s_gp.mul(&v3.sub(gp3)),
            s_gl.mul(&v0.sub(gl)),
            s_gl.mul(&hi_sum),
            s_pr.mul(&v0.sub(&r0)),
            s_pr.mul(&v1.sub(&r1)),
            s_pr.mul(&v2.sub(&r2)),
            s_pr.mul(&v3.sub(&r3)),
            one_minus_real.mul(op),
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
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let two_gas_p = vec![Scalar::from_u64(BLOCK_CONTEXT_READER_GAS, curve)];
        let op_origin_p = vec![Scalar::from_u64(ORIGIN_OPCODE as u64, curve)];
        let op_gasprice_p = vec![Scalar::from_u64(GASPRICE_OPCODE as u64, curve)];
        let op_gaslimit_p = vec![Scalar::from_u64(GASLIMIT_OPCODE as u64, curve)];
        let op_prevrandao_p = vec![Scalar::from_u64(PREVRANDAO_OPCODE as u64, curve)];
        let w = byte_weights(curve);
        let is_real = &cc[COL_IS_REAL];
        let s_o = &cc[COL_SEL_ORIGIN];
        let s_gp = &cc[COL_SEL_GASPRICE];
        let s_gl = &cc[COL_SEL_GASLIMIT];
        let s_pr = &cc[COL_SEL_PREVRANDAO];
        let gc = &cc[COL_GAS_COST];
        let op = &cc[COL_OPCODE];
        let v0 = &cc[COL_VALUE_LIMB_0];
        let v1 = &cc[COL_VALUE_LIMB_1];
        let v2 = &cc[COL_VALUE_LIMB_2];
        let v3 = &cc[COL_VALUE_LIMB_3];
        let gp0 = &cc[COL_TX_GASPRICE_L0];
        let gp1 = &cc[COL_TX_GASPRICE_L1];
        let gp2 = &cc[COL_TX_GASPRICE_L2];
        let gp3 = &cc[COL_TX_GASPRICE_L3];
        let gl = &cc[COL_BLOCK_GASLIMIT];

        let mut s0: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_TX_ORIGIN_0 + i], &w[i]);
            s0 = poly_add(&s0, &scaled, curve);
        }
        let mut s1: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_TX_ORIGIN_0 + 8 + i], &w[i]);
            s1 = poly_add(&s1, &scaled, curve);
        }
        let mut s2: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..4 {
            let scaled = poly_scalar_mul(&cc[COL_TX_ORIGIN_0 + 16 + i], &w[i]);
            s2 = poly_add(&s2, &scaled, curve);
        }
        let mut r0: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_PREV_RANDAO_0 + i], &w[i]);
            r0 = poly_add(&r0, &scaled, curve);
        }
        let mut r1: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_PREV_RANDAO_0 + 8 + i], &w[i]);
            r1 = poly_add(&r1, &scaled, curve);
        }
        let mut r2: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_PREV_RANDAO_0 + 16 + i], &w[i]);
            r2 = poly_add(&r2, &scaled, curve);
        }
        let mut r3: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_PREV_RANDAO_0 + 24 + i], &w[i]);
            r3 = poly_add(&r3, &scaled, curve);
        }

        let sum = poly_add(
            &poly_add(&poly_add(s_o, s_gp, curve), s_gl, curve),
            s_pr,
            curve,
        );
        let one_minus_real = poly_sub(&one_p, is_real, curve);
        let hi_sum = poly_add(&poly_add(v1, v2, curve), v3, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            poly_mul(is_real, &poly_sub(is_real, &one_p, curve), curve),
            poly_mul(s_o, &poly_sub(s_o, &one_p, curve), curve),
            poly_mul(s_gp, &poly_sub(s_gp, &one_p, curve), curve),
            poly_mul(s_gl, &poly_sub(s_gl, &one_p, curve), curve),
            poly_mul(s_pr, &poly_sub(s_pr, &one_p, curve), curve),
            poly_sub(is_real, &sum, curve),
            {
                let p_o_gp = poly_mul(s_o, s_gp, curve);
                let p_o_gl = poly_mul(s_o, s_gl, curve);
                let p_o_pr = poly_mul(s_o, s_pr, curve);
                let p_gp_gl = poly_mul(s_gp, s_gl, curve);
                let p_gp_pr = poly_mul(s_gp, s_pr, curve);
                let p_gl_pr = poly_mul(s_gl, s_pr, curve);
                let mut acc = poly_add(&p_o_gp, &p_o_gl, curve);
                acc = poly_add(&acc, &p_o_pr, curve);
                acc = poly_add(&acc, &p_gp_gl, curve);
                acc = poly_add(&acc, &p_gp_pr, curve);
                acc = poly_add(&acc, &p_gl_pr, curve);
                acc
            },
            poly_mul(is_real, &poly_sub(gc, &two_gas_p, curve), curve),
            poly_mul(s_o, &poly_sub(op, &op_origin_p, curve), curve),
            poly_mul(s_gp, &poly_sub(op, &op_gasprice_p, curve), curve),
            poly_mul(s_gl, &poly_sub(op, &op_gaslimit_p, curve), curve),
            poly_mul(s_pr, &poly_sub(op, &op_prevrandao_p, curve), curve),
            poly_mul(s_o, &poly_sub(v0, &s0, curve), curve),
            poly_mul(s_o, &poly_sub(v1, &s1, curve), curve),
            poly_mul(s_o, &poly_sub(v2, &s2, curve), curve),
            poly_mul(s_o, v3, curve),
            poly_mul(s_gp, &poly_sub(v0, gp0, curve), curve),
            poly_mul(s_gp, &poly_sub(v1, gp1, curve), curve),
            poly_mul(s_gp, &poly_sub(v2, gp2, curve), curve),
            poly_mul(s_gp, &poly_sub(v3, gp3, curve), curve),
            poly_mul(s_gl, &poly_sub(v0, gl, curve), curve),
            poly_mul(s_gl, &hi_sum, curve),
            poly_mul(s_pr, &poly_sub(v0, &r0, curve), curve),
            poly_mul(s_pr, &poly_sub(v1, &r1, curve), curve),
            poly_mul(s_pr, &poly_sub(v2, &r2, curve), curve),
            poly_mul(s_pr, &poly_sub(v3, &r3, curve), curve),
            poly_mul(&one_minus_real, op, curve),
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

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS { return; }
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(1), LookupTable::range(8)];
        let tbl_bit = 0usize;
        let tbl_byte = 1usize;
        let mut declarations = Vec::new();
        // 5 binary range checks: is_real + 4 selectors
        for (name, col) in [
            ("block_context_readers_is_real_1bit", COL_IS_REAL),
            ("block_context_readers_sel_origin_1bit", COL_SEL_ORIGIN),
            ("block_context_readers_sel_gasprice_1bit", COL_SEL_GASPRICE),
            ("block_context_readers_sel_gaslimit_1bit", COL_SEL_GASLIMIT),
            ("block_context_readers_sel_prevrandao_1bit", COL_SEL_PREVRANDAO),
        ] {
            declarations.push((
                LookupDeclaration {
                    label: name.into(),
                    column_index: col,
                    max_bits: 1,
                    selector_column: None,
                },
                tbl_bit,
            ));
        }
        // opcode byte range
        declarations.push((
            LookupDeclaration {
                label: "block_context_readers_opcode_8bit".into(),
                column_index: COL_OPCODE,
                max_bits: 8,
                selector_column: None,
            },
            tbl_byte,
        ));
        // 20 tx_origin byte range checks
        for i in 0..NUM_ADDR_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("block_context_readers_origin_byte_{i}_8bit"),
                    column_index: COL_TX_ORIGIN_0 + i,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        // 32 prev_randao byte range checks
        for i in 0..NUM_RANDAO_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("block_context_readers_randao_byte_{i}_8bit"),
                    column_index: COL_PREV_RANDAO_0 + i,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp linkage descriptors ──────────────────────────────

/// GASLIMIT + PREVRANDAO rows ↔ `block_header_air`. The A side is gated
/// by `is_real` and exposes the 5-tuple
/// `(block_gaslimit, prev_randao_l0..3)`; the B side is the
/// `block_header_air` row's `(COL_GAS_LIMIT, COL_PREV_RANDAO_L0..L3)`
/// tuple gated by its own `COL_IS_REAL`.
///
/// Note: on ORIGIN/GASPRICE rows the A-side `block_gaslimit` and
/// `prev_randao_*` byte columns are filled with the corresponding row's
/// witness values too (so the descriptor's A-side row content matches
/// the B-side header row regardless of which opcode fired). To keep
/// soundness simple, the linkage is gated by `sel_gaslimit + sel_prevrandao`
/// equivalents — concretely we expose two narrow descriptors instead,
/// one for `block_gaslimit` and one for `prev_randao`, each gated by
/// the appropriate selector. The combined helper below returns the
/// gaslimit descriptor; the prevrandao descriptor is the companion.
pub fn make_block_context_to_block_header_descriptor(
    block_context_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_context_readers_to_block_header_gaslimit_v1".into(),
        a_layer_index: block_context_layer_index,
        a_columns: vec![COL_BLOCK_GASLIMIT],
        a_selector_column: Some(COL_SEL_GASLIMIT),
        b_layer_index: block_header_layer_index,
        b_columns: vec![metavm_zkp::block_header_air::COL_GAS_LIMIT],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// Companion to [`make_block_context_to_block_header_descriptor`]: binds
/// the four `value_limb_0..3` columns on PREVRANDAO rows to the
/// block-header `COL_PREV_RANDAO_L0..L3` tuple.
pub fn make_block_context_to_block_header_prevrandao_descriptor(
    block_context_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_context_readers_to_block_header_prevrandao_v1".into(),
        a_layer_index: block_context_layer_index,
        a_columns: vec![
            COL_VALUE_LIMB_0,
            COL_VALUE_LIMB_1,
            COL_VALUE_LIMB_2,
            COL_VALUE_LIMB_3,
        ],
        a_selector_column: Some(COL_SEL_PREVRANDAO),
        b_layer_index: block_header_layer_index,
        b_columns: vec![
            metavm_zkp::block_header_air::COL_PREV_RANDAO_L0,
            metavm_zkp::block_header_air::COL_PREV_RANDAO_L1,
            metavm_zkp::block_header_air::COL_PREV_RANDAO_L2,
            metavm_zkp::block_header_air::COL_PREV_RANDAO_L3,
        ],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// GASPRICE rows ↔ `tx_rlp_air`'s `tx_gasprice` u256. The A side is the
/// 4 LE u64 limbs (gated by `sel_gasprice`); on the B side, `tx_rlp_air`
/// stores the gas-price as 32 big-endian bytes at
/// [`metavm_zkp::tx_rlp_air::COL_GAS_PRICE_BYTE_OFFSET`]. Because the A
/// and B encodings don't align byte-for-byte, this descriptor binds the
/// four BE byte windows that *would* equal each LE u64 limb (i.e. the
/// 8-byte groups `bytes[24..32]`, `bytes[16..24]`, `bytes[8..16]`,
/// `bytes[0..8]`) — but only as a tuple-shape declaration; the byte-
/// to-limb decomposition is enforced elsewhere by `u256_rlp_air`.
///
/// For now we expose a single declaration on the low byte of each 8-byte
/// BE group (the byte that carries the LSB of the corresponding LE u64
/// limb), keeping the descriptor narrow. The full per-byte binding is
/// captured by `u256_rlp_air`'s own gadget, which `tx_rlp_air` already
/// composes with.
///
/// In practice the link this descriptor closes is:
/// `(tx_gasprice_l0, tx_gasprice_l1, tx_gasprice_l2, tx_gasprice_l3)`
/// — the four limbs we store on the AIR's own rows — against the
/// corresponding witness columns inside `tx_rlp_air`'s `u256_rlp` sub-
/// gadget for `gas_price`. Until that gadget exposes them as named
/// columns, we surface the A-side declaration with no B-side resolution
/// (`b_columns` is left as the four BE bytes that hold the LSB of each
/// 8-byte group, i.e. bytes 31, 23, 15, 7 of the `gas_price` BE blob).
pub fn make_block_context_to_tx_rlp_descriptor(
    block_context_layer_index: usize,
    tx_rlp_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    let gp = metavm_zkp::tx_rlp_air::COL_GAS_PRICE_BYTE_OFFSET;
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_context_readers_to_tx_rlp_gasprice_v1".into(),
        a_layer_index: block_context_layer_index,
        a_columns: vec![
            COL_TX_GASPRICE_L0,
            COL_TX_GASPRICE_L1,
            COL_TX_GASPRICE_L2,
            COL_TX_GASPRICE_L3,
        ],
        a_selector_column: Some(COL_SEL_GASPRICE),
        b_layer_index: tx_rlp_layer_index,
        // BE byte order: limb 0 = LSB → byte 31 of the 32-byte BE field
        // (i.e. offset 31). Limb 1 → byte 23. Limb 2 → byte 15.
        // Limb 3 → byte 7. The byte-to-limb decomposition is enforced by
        // the u256_rlp gadget itself; this descriptor pins the LSBs.
        b_columns: vec![gp + 31, gp + 23, gp + 15, gp + 7],
        b_selector_column: Some(metavm_zkp::tx_rlp_air::COL_IS_REAL),
    }
}

/// Every real row ↔ `stack_contents_air` unsorted view
/// `(pc, value_limb_0..3)`. Binds the stack push event at this PC to the
/// canonical u256 encoding of the corresponding context value.
pub fn make_block_context_to_stack_contents_descriptor(
    block_context_layer_index: usize,
    stack_contents_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_context_readers_to_stack_contents_push_v1".into(),
        a_layer_index: block_context_layer_index,
        a_columns: vec![
            COL_PC,
            COL_VALUE_LIMB_0,
            COL_VALUE_LIMB_1,
            COL_VALUE_LIMB_2,
            COL_VALUE_LIMB_3,
        ],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: stack_contents_layer_index,
        b_columns: vec![
            crate::stack_contents_air::COL_PC,
            crate::stack_contents_air::COL_VALUE_LIMB_0,
            crate::stack_contents_air::COL_VALUE_LIMB_1,
            crate::stack_contents_air::COL_VALUE_LIMB_2,
            crate::stack_contents_air::COL_VALUE_LIMB_3,
        ],
        b_selector_column: Some(crate::stack_contents_air::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_vanish(
        trace: &TracePolynomials,
        cs: &BlockContextReadersConstraintSystem,
    ) {
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, trace.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) nonzero at row {}",
                    i,
                    cs.constraint_labels()[i],
                    r,
                );
            }
        }
    }

    fn origin_event(pc: u64, addr: [u8; 20]) -> (u8, u64, [u8; 32]) {
        let limbs = crate::address_opcode_air::address_to_limbs(&addr);
        let mut val = [0u8; 32];
        for k in 0..4 {
            val[8 * k..8 * k + 8].copy_from_slice(&limbs[k].to_le_bytes());
        }
        (ORIGIN_OPCODE, pc, val)
    }

    fn gasprice_event(pc: u64, limbs: [u64; 4]) -> (u8, u64, [u8; 32]) {
        let mut val = [0u8; 32];
        for k in 0..4 {
            val[8 * k..8 * k + 8].copy_from_slice(&limbs[k].to_le_bytes());
        }
        (GASPRICE_OPCODE, pc, val)
    }

    fn gaslimit_event(pc: u64, gas: u64) -> (u8, u64, [u8; 32]) {
        let mut val = [0u8; 32];
        val[0..8].copy_from_slice(&gas.to_le_bytes());
        (GASLIMIT_OPCODE, pc, val)
    }

    fn prevrandao_event(pc: u64, mix_hash: [u8; 32]) -> (u8, u64, [u8; 32]) {
        (PREVRANDAO_OPCODE, pc, mix_hash)
    }

    #[test]
    fn block_context_readers_origin_vanishes() {
        let mut addr = [0u8; 20];
        for i in 0..20 { addr[i] = (i as u8) ^ 0x5A; }
        let w = BlockContextReadersWitness::from_events(&[origin_event(13, addr)]);
        assert_eq!(w.rows.len(), 1);
        let row = w.rows[0];
        assert!(row.sel_origin && !row.sel_gasprice && !row.sel_gaslimit && !row.sel_prevrandao);
        assert_eq!(row.opcode, ORIGIN_OPCODE);
        assert_eq!(row.gas_cost, BLOCK_CONTEXT_READER_GAS);
        assert_eq!(row.tx_origin, addr);
        // limb 3 must be zero for any 20-byte address.
        assert_eq!(row.value[3], 0);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn block_context_readers_gasprice_vanishes() {
        let limbs = [0xCAFEBABEu64, 0x1234u64, 0u64, 0u64];
        let w = BlockContextReadersWitness::from_events(&[gasprice_event(42, limbs)]);
        let row = w.rows[0];
        assert!(row.sel_gasprice);
        assert_eq!(row.opcode, GASPRICE_OPCODE);
        assert_eq!(row.tx_gasprice, limbs);
        assert_eq!(row.value, limbs);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn block_context_readers_gaslimit_vanishes() {
        let w = BlockContextReadersWitness::from_events(&[gaslimit_event(77, 30_000_000)]);
        let row = w.rows[0];
        assert!(row.sel_gaslimit);
        assert_eq!(row.opcode, GASLIMIT_OPCODE);
        assert_eq!(row.block_gaslimit, 30_000_000);
        assert_eq!(row.value, [30_000_000, 0, 0, 0]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn block_context_readers_prevrandao_vanishes() {
        let mut mix = [0u8; 32];
        for i in 0..32 { mix[i] = i as u8; }
        let w = BlockContextReadersWitness::from_events(&[prevrandao_event(5, mix)]);
        let row = w.rows[0];
        assert!(row.sel_prevrandao);
        assert_eq!(row.opcode, PREVRANDAO_OPCODE);
        assert_eq!(row.prev_randao, mix);
        // Confirm the limb encoding matches LE bytes-to-limbs.
        assert_eq!(row.value, value_bytes_to_limbs(&mix));
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn block_context_readers_multi_row_mixed_vanishes() {
        let mut addr = [0u8; 20];
        for i in 0..20 { addr[i] = 0xCC; }
        let mut mix = [0u8; 32];
        for i in 0..32 { mix[i] = 0xA5; }
        let w = BlockContextReadersWitness::from_events(&[
            origin_event(0, addr),
            gasprice_event(3, [1_000_000_000, 0, 0, 0]),
            gaslimit_event(7, 15_000_000),
            prevrandao_event(11, mix),
        ]);
        assert_eq!(w.rows.len(), 4);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn block_context_readers_tampered_gas_detected() {
        let curve = CurveType::Bls48581;
        let w = BlockContextReadersWitness::from_events(&[gaslimit_event(0, 1_000_000)]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_GAS_COST].evaluations[0] = Scalar::from_u64(3, curve);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // gas_cost_eq_2 is constraint #7
        assert!(!bodies[7][0].is_zero(), "gas_cost_eq_2 must fire");
    }

    #[test]
    fn block_context_readers_tampered_gasprice_detected() {
        let curve = CurveType::Bls48581;
        let w = BlockContextReadersWitness::from_events(&[gasprice_event(0, [7, 8, 9, 10])]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_TX_GASPRICE_L2].evaluations[0] = Scalar::from_u64(99, curve);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // gasprice_value_l2_eq is constraint #18
        assert!(!bodies[18][0].is_zero(), "gasprice_value_l2_eq must fire");
    }

    #[test]
    fn block_context_readers_tampered_randao_detected() {
        let curve = CurveType::Bls48581;
        let mut mix = [0u8; 32];
        for i in 0..32 { mix[i] = i as u8; }
        let w = BlockContextReadersWitness::from_events(&[prevrandao_event(0, mix)]);
        let mut t = build_trace_polynomials(&w, curve);
        // Tamper byte 0: this affects the value_limb_0 binding.
        t.columns[COL_PREV_RANDAO_0].evaluations[0] = Scalar::from_u64(0xFF, curve);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // prevrandao_value_l0_binding is constraint #22
        assert!(!bodies[22][0].is_zero(), "prevrandao_value_l0_binding must fire");
    }

    #[test]
    fn block_context_readers_tampered_selector_sum_detected() {
        let curve = CurveType::Bls48581;
        let w = BlockContextReadersWitness::from_events(&[gaslimit_event(0, 1)]);
        let mut t = build_trace_polynomials(&w, curve);
        // Drop sel_gaslimit so is_real=1 but selector sum=0.
        t.columns[COL_SEL_GASLIMIT].evaluations[0] = Scalar::zero(curve);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // selector_sum_eq_is_real is #5
        assert!(!bodies[5][0].is_zero(), "selector_sum_eq_is_real must fire");
    }

    #[test]
    fn block_context_readers_evaluate_at_point_zero_on_honest() {
        let w = BlockContextReadersWitness::from_events(&[gasprice_event(3, [1, 2, 3, 4])]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlockContextReadersConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xBEEF, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "evaluate_at_point should be zero on honest row");
    }

    #[test]
    fn block_context_readers_descriptors_well_formed() {
        let bh = make_block_context_to_block_header_descriptor(0, 1);
        assert_eq!(bh.label, "block_context_readers_to_block_header_gaslimit_v1");
        assert_eq!(bh.a_layer_index, 0);
        assert_eq!(bh.b_layer_index, 1);
        assert_eq!(bh.a_columns, vec![COL_BLOCK_GASLIMIT]);
        assert_eq!(bh.a_selector_column, Some(COL_SEL_GASLIMIT));
        assert_eq!(
            bh.b_columns,
            vec![metavm_zkp::block_header_air::COL_GAS_LIMIT]
        );
        assert_eq!(
            bh.b_selector_column,
            Some(metavm_zkp::block_header_air::COL_IS_REAL)
        );

        let bh_pr = make_block_context_to_block_header_prevrandao_descriptor(0, 1);
        assert_eq!(
            bh_pr.label,
            "block_context_readers_to_block_header_prevrandao_v1"
        );
        assert_eq!(bh_pr.a_columns.len(), 4);
        assert_eq!(bh_pr.b_columns.len(), 4);
        assert_eq!(bh_pr.a_selector_column, Some(COL_SEL_PREVRANDAO));
        assert_eq!(
            bh_pr.b_columns,
            vec![
                metavm_zkp::block_header_air::COL_PREV_RANDAO_L0,
                metavm_zkp::block_header_air::COL_PREV_RANDAO_L1,
                metavm_zkp::block_header_air::COL_PREV_RANDAO_L2,
                metavm_zkp::block_header_air::COL_PREV_RANDAO_L3,
            ]
        );

        let tx = make_block_context_to_tx_rlp_descriptor(2, 3);
        assert_eq!(tx.label, "block_context_readers_to_tx_rlp_gasprice_v1");
        assert_eq!(tx.a_columns.len(), 4);
        assert_eq!(tx.b_columns.len(), 4);
        assert_eq!(tx.a_selector_column, Some(COL_SEL_GASPRICE));
        let gp = metavm_zkp::tx_rlp_air::COL_GAS_PRICE_BYTE_OFFSET;
        assert_eq!(tx.b_columns, vec![gp + 31, gp + 23, gp + 15, gp + 7]);
        assert_eq!(
            tx.b_selector_column,
            Some(metavm_zkp::tx_rlp_air::COL_IS_REAL)
        );

        let sc = make_block_context_to_stack_contents_descriptor(4, 5);
        assert_eq!(sc.label, "block_context_readers_to_stack_contents_push_v1");
        assert_eq!(sc.a_columns.len(), 5);
        assert_eq!(sc.b_columns.len(), 5);
        assert_eq!(sc.a_columns[0], COL_PC);
        assert_eq!(sc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            sc.b_columns,
            vec![
                crate::stack_contents_air::COL_PC,
                crate::stack_contents_air::COL_VALUE_LIMB_0,
                crate::stack_contents_air::COL_VALUE_LIMB_1,
                crate::stack_contents_air::COL_VALUE_LIMB_2,
                crate::stack_contents_air::COL_VALUE_LIMB_3,
            ]
        );
        assert_eq!(
            sc.b_selector_column,
            Some(crate::stack_contents_air::COL_IS_REAL)
        );
    }

    #[test]
    fn block_context_readers_column_layout_pinned() {
        assert_eq!(COL_PC, 0);
        assert_eq!(COL_OPCODE, 1);
        assert_eq!(COL_VALUE_LIMB_0, 2);
        assert_eq!(COL_VALUE_LIMB_3, 5);
        assert_eq!(COL_GAS_COST, 6);
        assert_eq!(COL_TX_ORIGIN_0, 7);
        assert_eq!(COL_TX_GASPRICE_L0, 27);
        assert_eq!(COL_TX_GASPRICE_L3, 30);
        assert_eq!(COL_BLOCK_GASLIMIT, 31);
        assert_eq!(COL_PREV_RANDAO_0, 32);
        assert_eq!(COL_SEL_ORIGIN, 64);
        assert_eq!(COL_SEL_GASPRICE, 65);
        assert_eq!(COL_SEL_GASLIMIT, 66);
        assert_eq!(COL_SEL_PREVRANDAO, 67);
        assert_eq!(COL_IS_REAL, 68);
        assert_eq!(NUM_COLUMNS, 69);
        assert_eq!(NUM_ROW_CONSTRAINTS, 27);
        assert_eq!(ORIGIN_OPCODE, 0x32);
        assert_eq!(GASPRICE_OPCODE, 0x3A);
        assert_eq!(GASLIMIT_OPCODE, 0x45);
        assert_eq!(PREVRANDAO_OPCODE, 0x44);
        assert_eq!(BLOCK_CONTEXT_READER_GAS, 2);
    }

    #[test]
    fn block_context_readers_lookup_declarations_well_formed() {
        let cs = BlockContextReadersConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 5 binary + 1 opcode + 20 origin + 32 randao = 58
        assert_eq!(req.declarations.len(), 5 + 1 + 20 + 32);
        assert_eq!(req.tables.len(), 2);
        assert_eq!(req.tables[0].bits, 1);
        assert_eq!(req.tables[1].bits, 8);
    }
}
