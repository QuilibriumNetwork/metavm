//! BASEFEE opcode constraint AIR (EIP-3198).
//!
//! Proves algebraically that on a BASEFEE row, the value pushed onto the
//! stack equals the current block's `base_fee_per_gas`, and that the gas
//! cost charged equals 2 (`G_base`). The BASEFEE opcode (0x48) takes no
//! stack inputs and pushes a single U256.
//!
//! Composed with the EVM main trace + block_header_air + eip1559 fee
//! market AIR via cross-AIR LogUp:
//!
//!   EVM main row gated by SEL_BASEFEE  →
//!   BASEFEE AIR row binds (base_fee_value, gas_cost == 2)
//!   BASEFEE AIR ↔ block_header_air      binds block_base_fee == header base_fee
//!   BASEFEE AIR ↔ eip1559_fee_market_air binds block_base_fee == new_base_fee

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────
//
// Per BASEFEE event row:
//   pc, gas_cost,
//   base_fee_value[0..4]   (4 LE u64 limbs, value pushed onto the stack)
//   block_base_fee[0..4]   (4 LE u64 limbs, the block's authoritative
//                          base_fee_per_gas — the binding target)
//   base_fee_le_bytes[0..32] (LE byte decomposition of base_fee_value)
//   is_real (binary)
//
// The 4 limb-equality constraints enforce that base_fee_value matches
// block_base_fee. The byte-decomposition constraints (4 of them) enforce
// that each limb equals Σ byte_k · 2^(8k) for k in that limb's 8-byte
// window; combined with the byte-range LogUp, this proves each limb is
// 64-bit canonical.

pub const COL_PC: usize = 0;
pub const COL_GAS_COST: usize = 1;
pub const COL_BASE_FEE_VAL_L0: usize = 2;
pub const COL_BASE_FEE_VAL_L1: usize = 3;
pub const COL_BASE_FEE_VAL_L2: usize = 4;
pub const COL_BASE_FEE_VAL_L3: usize = 5;
pub const COL_BLOCK_BASE_FEE_L0: usize = 6;
pub const COL_BLOCK_BASE_FEE_L1: usize = 7;
pub const COL_BLOCK_BASE_FEE_L2: usize = 8;
pub const COL_BLOCK_BASE_FEE_L3: usize = 9;
pub const COL_BASE_FEE_BYTE_0: usize = 10; // bytes 0..32 (LE)
pub const COL_BASE_FEE_BYTE_31: usize = 41;
pub const COL_IS_REAL: usize = 42;
pub const NUM_COLUMNS: usize = 43;

// Row-local constraints (10 total, well above the 6-minimum):
//   0 is_real binary
//   1..4 limb equality base_fee_value[i] == block_base_fee[i] (gated)
//   5 gas_cost == 2 (gated)
//   6..9 LE byte decomposition for each of the 4 limbs (gated):
//        base_fee_value_lk = Σ_{j=0..8} base_fee_byte[8k+j] * 2^(8j)
pub const NUM_ROW_CONSTRAINTS: usize = 10;
pub const NUM_SHIFTED: usize = 0;

/// EIP-3198: BASEFEE static gas cost (G_base).
pub const BASEFEE_GAS_COST: u64 = 2;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BaseFeeRow {
    pub pc: u64,
    /// LE u64 limbs of the value pushed onto the stack.
    pub base_fee_value: [u64; 4],
    /// LE u64 limbs of the block's authoritative base_fee_per_gas.
    pub block_base_fee: [u64; 4],
    /// Static gas cost (always 2 for BASEFEE).
    pub gas_cost: u64,
    /// LE byte decomposition of `base_fee_value`.
    pub base_fee_le_bytes: [u8; 32],
}

#[derive(Clone, Debug, Default)]
pub struct BaseFeeWitness {
    pub rows: Vec<BaseFeeRow>,
}

impl BaseFeeWitness {
    pub fn from_rows(rows: Vec<BaseFeeRow>) -> Self {
        Self { rows }
    }

    /// Build a witness from a slice of `(pc, block_base_fee_be)` events.
    /// `block_base_fee_be` is the 32-byte big-endian U256 representation
    /// of the block's base_fee_per_gas. On an honest trace, the pushed
    /// value equals the block base fee, so both `base_fee_value` and
    /// `block_base_fee` are set to the same LE limbs.
    pub fn from_events(events: &[(u64, [u8; 32])]) -> Self {
        let rows = events
            .iter()
            .map(|(pc, be)| {
                let limbs = be_bytes_to_le_limbs(be);
                let le_bytes = be_to_le_bytes(be);
                BaseFeeRow {
                    pc: *pc,
                    base_fee_value: limbs,
                    block_base_fee: limbs,
                    gas_cost: BASEFEE_GAS_COST,
                    base_fee_le_bytes: le_bytes,
                }
            })
            .collect();
        Self { rows }
    }
}

fn be_to_le_bytes(be: &[u8; 32]) -> [u8; 32] {
    let mut le = [0u8; 32];
    for i in 0..32 {
        le[i] = be[31 - i];
    }
    le
}

fn be_bytes_to_le_limbs(be: &[u8; 32]) -> [u64; 4] {
    let le = be_to_le_bytes(be);
    let mut limbs = [0u64; 4];
    for k in 0..4 {
        let mut acc = 0u64;
        for j in (0..8).rev() {
            acc = (acc << 8) | (le[8 * k + j] as u64);
        }
        limbs[k] = acc;
    }
    limbs
}

pub fn build_trace_polynomials(w: &BaseFeeWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_PC][r] = Scalar::from_u64(row.pc, curve);
        cols[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
        for j in 0..4 {
            cols[COL_BASE_FEE_VAL_L0 + j][r] = Scalar::from_u64(row.base_fee_value[j], curve);
            cols[COL_BLOCK_BASE_FEE_L0 + j][r] = Scalar::from_u64(row.block_base_fee[j], curve);
        }
        for k in 0..32 {
            cols[COL_BASE_FEE_BYTE_0 + k][r] = Scalar::from_u64(row.base_fee_le_bytes[k] as u64, curve);
        }
        cols[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct BaseFeeConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BaseFeeConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Compute `Σ_{j=0..8} bytes[8k+j] * 2^(8j)` at a single row of `columns`.
fn limb_from_bytes_row(columns: &[&Vec<Scalar>], r: usize, k: usize, curve: CurveType) -> Scalar {
    let mut acc = Scalar::zero(curve);
    let mut shift = Scalar::one(curve);
    let two_pow_8 = Scalar::from_u64(256, curve);
    for j in 0..8 {
        let b = &columns[COL_BASE_FEE_BYTE_0 + 8 * k + j][r];
        acc = acc.add(&shift.mul(b));
        shift = shift.mul(&two_pow_8);
    }
    acc
}

fn limb_from_bytes_scalar(ce: &[Scalar], k: usize, curve: CurveType) -> Scalar {
    let mut acc = Scalar::zero(curve);
    let mut shift = Scalar::one(curve);
    let two_pow_8 = Scalar::from_u64(256, curve);
    for j in 0..8 {
        let b = &ce[COL_BASE_FEE_BYTE_0 + 8 * k + j];
        acc = acc.add(&shift.mul(b));
        shift = shift.mul(&two_pow_8);
    }
    acc
}

fn limb_from_bytes_poly(cc: &[Vec<Scalar>], k: usize, curve: CurveType) -> Vec<Scalar> {
    let mut acc: Vec<Scalar> = vec![Scalar::zero(curve)];
    let mut shift = Scalar::one(curve);
    let two_pow_8 = Scalar::from_u64(256, curve);
    for j in 0..8 {
        let b = &cc[COL_BASE_FEE_BYTE_0 + 8 * k + j];
        let term = poly_scalar_mul(b, &shift);
        acc = poly_add(&acc, &term, curve);
        shift = shift.mul(&two_pow_8);
    }
    acc
}

impl VmConstraintSystem for BaseFeeConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "base_fee_eq_block_l0".into(),
            "base_fee_eq_block_l1".into(),
            "base_fee_eq_block_l2".into(),
            "base_fee_eq_block_l3".into(),
            "gas_cost_eq_2".into(),
            "le_byte_decomp_l0".into(),
            "le_byte_decomp_l1".into(),
            "le_byte_decomp_l2".into(),
            "le_byte_decomp_l3".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(BASEFEE_GAS_COST, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            // 0: is_real binary
            bodies[0][r] = v.mul(&v.sub(&one));
            // 1..4: limb equality
            for k in 0..4 {
                let diff = columns[COL_BASE_FEE_VAL_L0 + k][r]
                    .sub(&columns[COL_BLOCK_BASE_FEE_L0 + k][r]);
                bodies[1 + k][r] = v.mul(&diff);
            }
            // 5: gas_cost == 2
            bodies[5][r] = v.mul(&columns[COL_GAS_COST][r].sub(&two));
            // 6..9: LE byte decomposition for each limb
            for k in 0..4 {
                let sum = limb_from_bytes_row(columns, r, k, curve);
                let diff = columns[COL_BASE_FEE_VAL_L0 + k][r].sub(&sum);
                bodies[6 + k][r] = v.mul(&diff);
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
        let two = Scalar::from_u64(BASEFEE_GAS_COST, curve);
        let v = &ce[COL_IS_REAL];
        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(v.mul(&v.sub(&one)));
        for k in 0..4 {
            let diff = ce[COL_BASE_FEE_VAL_L0 + k].sub(&ce[COL_BLOCK_BASE_FEE_L0 + k]);
            bodies.push(v.mul(&diff));
        }
        bodies.push(v.mul(&ce[COL_GAS_COST].sub(&two)));
        for k in 0..4 {
            let sum = limb_from_bytes_scalar(ce, k, curve);
            let diff = ce[COL_BASE_FEE_VAL_L0 + k].sub(&sum);
            bodies.push(v.mul(&diff));
        }
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let two_p = vec![Scalar::from_u64(BASEFEE_GAS_COST, curve)];
        let v = &cc[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_p, curve);
        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        // 0
        bodies.push(poly_mul(v, &v_m1, curve));
        // 1..4
        for k in 0..4 {
            let diff = poly_sub(&cc[COL_BASE_FEE_VAL_L0 + k], &cc[COL_BLOCK_BASE_FEE_L0 + k], curve);
            bodies.push(poly_mul(v, &diff, curve));
        }
        // 5
        let g_diff = poly_sub(&cc[COL_GAS_COST], &two_p, curve);
        bodies.push(poly_mul(v, &g_diff, curve));
        // 6..9
        for k in 0..4 {
            let sum = limb_from_bytes_poly(cc, k, curve);
            let diff = poly_sub(&cc[COL_BASE_FEE_VAL_L0 + k], &sum, curve);
            bodies.push(poly_mul(v, &diff, curve));
        }
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
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
        // Range-check each of the 32 base_fee_le_bytes columns to [0, 255].
        let tables = vec![LookupTable::range(8)];
        let mut decls = Vec::with_capacity(32);
        for k in 0..32 {
            decls.push((
                LookupDeclaration {
                    label: format!("basefee_byte_{}_8bit", k),
                    column_index: COL_BASE_FEE_BYTE_0 + k,
                    max_bits: 8,
                    selector_column: Some(COL_IS_REAL),
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations: decls }
    }
}

// ─── Cross-AIR LogUp descriptor builders ─────────────────────────────

/// EVM main row gated by `COL_SEL_BASEFEE` → BASEFEE AIR row.
///
/// Tuple: (output0_l0, output0_l1, output0_l2, output0_l3) on EVM side,
/// (base_fee_value_l0..l3) on the gadget side. Binds the EVM-pushed
/// value to the BASEFEE AIR's `base_fee_value`.
pub fn make_evm_to_basefee_descriptor(
    evm_layer: usize,
    basefee_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_BASEFEE,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_basefee_to_basefee_air_v1".into(),
        a_layer_index: evm_layer,
        a_columns: vec![COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3],
        a_selector_column: Some(COL_SEL_BASEFEE),
        b_layer_index: basefee_layer,
        b_columns: vec![
            COL_BASE_FEE_VAL_L0, COL_BASE_FEE_VAL_L1, COL_BASE_FEE_VAL_L2, COL_BASE_FEE_VAL_L3,
        ],
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// BASEFEE AIR → block_header_air. Binds the gadget's `block_base_fee`
/// to the block header's `base_fee_per_gas` (4 LE u64 limbs).
pub fn make_basefee_to_block_header_descriptor(
    basefee_layer: usize,
    block_header_layer: usize,
    block_header_is_real_col: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use metavm_zkp::block_header_air::{
        COL_BASE_FEE_L0, COL_BASE_FEE_L1, COL_BASE_FEE_L2, COL_BASE_FEE_L3,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "basefee_air_to_block_header_v1".into(),
        a_layer_index: basefee_layer,
        a_columns: vec![
            COL_BLOCK_BASE_FEE_L0, COL_BLOCK_BASE_FEE_L1, COL_BLOCK_BASE_FEE_L2, COL_BLOCK_BASE_FEE_L3,
        ],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: block_header_layer,
        b_columns: vec![COL_BASE_FEE_L0, COL_BASE_FEE_L1, COL_BASE_FEE_L2, COL_BASE_FEE_L3],
        b_selector_column: Some(block_header_is_real_col),
    }
}

/// BASEFEE AIR → eip1559_fee_market_air. Binds the gadget's
/// `block_base_fee` (low limb) to the eip1559 AIR's `new_base_fee` for
/// the block-class row. The eip1559 AIR stores base_fee values as single
/// u64 cells (sufficient for realistic ETH base fees ≪ 2^64 wei), so we
/// bind only limb 0 here and rely on the BASEFEE AIR's byte decomposition
/// to pin the higher limbs to zero when the value fits in 64 bits.
pub fn make_basefee_to_eip1559_descriptor(
    basefee_layer: usize,
    eip1559_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use metavm_zkp::eip1559_fee_market_air::{COL_IS_BLOCK_ROW, COL_NEW_BASE_FEE};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "basefee_air_to_eip1559_v1".into(),
        a_layer_index: basefee_layer,
        a_columns: vec![COL_BLOCK_BASE_FEE_L0],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: eip1559_layer,
        b_columns: vec![COL_NEW_BASE_FEE],
        b_selector_column: Some(COL_IS_BLOCK_ROW),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn be_from_u64(v: u64) -> [u8; 32] {
        let mut be = [0u8; 32];
        be[24..32].copy_from_slice(&v.to_be_bytes());
        be
    }

    #[test]
    fn simple_basefee_constraints_zero() {
        // Single BASEFEE event with base fee = 1_000_000_000 wei.
        let be = be_from_u64(1_000_000_000);
        let w = BaseFeeWitness::from_events(&[(7, be)]);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].pc, 7);
        assert_eq!(w.rows[0].base_fee_value[0], 1_000_000_000);
        assert_eq!(w.rows[0].gas_cost, BASEFEE_GAS_COST);
        // First LE byte should be the low byte of 1e9.
        assert_eq!(w.rows[0].base_fee_le_bytes[0], (1_000_000_000u64 & 0xff) as u8);

        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BaseFeeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn tampered_value_detected() {
        // Build an honest row, then corrupt base_fee_value limb 0 so it
        // disagrees with block_base_fee.
        let be = be_from_u64(2_000_000_000);
        let mut w = BaseFeeWitness::from_events(&[(0, be)]);
        w.rows[0].base_fee_value[0] = 999; // disagrees with block_base_fee
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BaseFeeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 1 = limb 0 equality should fire.
        assert!(!bodies[1][0].is_zero(), "limb 0 equality should fire");
        // Constraint 6 = LE byte decomp for limb 0 should also fire
        // (bytes are still those of 2e9 but limb 0 is now 999).
        assert!(!bodies[6][0].is_zero(), "byte decomp for limb 0 should fire");
    }

    #[test]
    fn tampered_gas_cost_detected() {
        let be = be_from_u64(100);
        let mut w = BaseFeeWitness::from_events(&[(0, be)]);
        w.rows[0].gas_cost = 3; // BASEFEE costs 2, not 3
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BaseFeeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 5 = gas_cost == 2 should fire.
        assert!(!bodies[5][0].is_zero(), "gas_cost constraint should fire");
    }

    #[test]
    fn multi_row_chain_honest() {
        // Three BASEFEE events at different PCs across the same block —
        // all should bind to the same block_base_fee.
        let be = be_from_u64(500_000_000);
        let w = BaseFeeWitness::from_events(&[(0, be), (10, be), (42, be)]);
        assert_eq!(w.rows.len(), 3);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BaseFeeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "c{} r{} nonzero on multi-row honest", i, r);
            }
        }
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let be = be_from_u64(7_777);
        let w = BaseFeeWitness::from_events(&[(3, be)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BaseFeeConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xdeadbeef, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "evaluate_at_point should be zero on honest row");
    }

    #[test]
    fn descriptors_well_formed() {
        let evm = make_evm_to_basefee_descriptor(0, 1);
        assert_eq!(evm.label, "evm_basefee_to_basefee_air_v1");
        assert_eq!(evm.a_columns.len(), 4);
        assert_eq!(evm.b_columns.len(), 4);
        assert!(evm.a_selector_column.is_some());
        assert!(evm.b_selector_column.is_some());
        assert_eq!(evm.a_layer_index, 0);
        assert_eq!(evm.b_layer_index, 1);

        // Block header: 4-limb base_fee binding.
        let bh = make_basefee_to_block_header_descriptor(1, 2, 0);
        assert_eq!(bh.label, "basefee_air_to_block_header_v1");
        assert_eq!(bh.a_columns.len(), 4);
        assert_eq!(bh.b_columns.len(), 4);
        assert_eq!(bh.a_columns[0], COL_BLOCK_BASE_FEE_L0);
        // The block_header b_columns must be the 4 BASE_FEE columns.
        assert_eq!(
            bh.b_columns,
            vec![
                metavm_zkp::block_header_air::COL_BASE_FEE_L0,
                metavm_zkp::block_header_air::COL_BASE_FEE_L1,
                metavm_zkp::block_header_air::COL_BASE_FEE_L2,
                metavm_zkp::block_header_air::COL_BASE_FEE_L3,
            ]
        );

        // EIP1559: low-limb binding to NEW_BASE_FEE on block rows.
        let fee = make_basefee_to_eip1559_descriptor(1, 3);
        assert_eq!(fee.label, "basefee_air_to_eip1559_v1");
        assert_eq!(fee.a_columns, vec![COL_BLOCK_BASE_FEE_L0]);
        assert_eq!(
            fee.b_columns,
            vec![metavm_zkp::eip1559_fee_market_air::COL_NEW_BASE_FEE]
        );
        assert_eq!(
            fee.b_selector_column,
            Some(metavm_zkp::eip1559_fee_market_air::COL_IS_BLOCK_ROW)
        );
    }

    #[test]
    fn column_layout_pinned() {
        // Pin the column layout so refactors fail loudly.
        assert_eq!(COL_PC, 0);
        assert_eq!(COL_GAS_COST, 1);
        assert_eq!(COL_BASE_FEE_VAL_L0, 2);
        assert_eq!(COL_BASE_FEE_VAL_L3, 5);
        assert_eq!(COL_BLOCK_BASE_FEE_L0, 6);
        assert_eq!(COL_BLOCK_BASE_FEE_L3, 9);
        assert_eq!(COL_BASE_FEE_BYTE_0, 10);
        assert_eq!(COL_BASE_FEE_BYTE_31, 41);
        assert_eq!(COL_IS_REAL, 42);
        assert_eq!(NUM_COLUMNS, 43);
        assert_eq!(NUM_ROW_CONSTRAINTS, 10);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(BASEFEE_GAS_COST, 2);
    }

    #[test]
    fn from_events_decomposes_le_bytes() {
        // Use a value with a known pattern: 0x0102030405060708 in limb 0,
        // all other limbs zero. Verify that base_fee_le_bytes[0..8] are
        // the little-endian bytes 08, 07, 06, 05, 04, 03, 02, 01.
        let mut be = [0u8; 32];
        be[24..32].copy_from_slice(&0x0102030405060708u64.to_be_bytes());
        let w = BaseFeeWitness::from_events(&[(0, be)]);
        let row = &w.rows[0];
        let expected: [u8; 8] = [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01];
        assert_eq!(&row.base_fee_le_bytes[0..8], &expected[..]);
        // High limbs and high bytes should be zero.
        assert_eq!(row.base_fee_value[1], 0);
        assert_eq!(row.base_fee_value[2], 0);
        assert_eq!(row.base_fee_value[3], 0);
        for k in 8..32 {
            assert_eq!(row.base_fee_le_bytes[k], 0);
        }
    }

    #[test]
    fn tampered_byte_decomp_detected() {
        // Honest setup, then corrupt a single byte cell so it no longer
        // matches base_fee_value's limb.
        let be = be_from_u64(0xabcd_ef12_3456_7890);
        let mut w = BaseFeeWitness::from_events(&[(0, be)]);
        w.rows[0].base_fee_le_bytes[0] = w.rows[0].base_fee_le_bytes[0].wrapping_add(1);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BaseFeeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 6 (LE byte decomp for limb 0) should fire.
        assert!(!bodies[6][0].is_zero(), "byte decomp for limb 0 should fire");
    }

    #[test]
    fn lookup_declarations_cover_all_bytes() {
        let cs = BaseFeeConstraintSystem::new(1);
        let reqs = cs.lookup_declarations();
        assert_eq!(reqs.tables.len(), 1);
        assert_eq!(reqs.tables[0].bits, 8);
        assert_eq!(reqs.declarations.len(), 32);
        // Each declaration targets a distinct byte column gated by IS_REAL.
        for (k, (decl, table_idx)) in reqs.declarations.iter().enumerate() {
            assert_eq!(*table_idx, 0);
            assert_eq!(decl.column_index, COL_BASE_FEE_BYTE_0 + k);
            assert_eq!(decl.max_bits, 8);
            assert_eq!(decl.selector_column, Some(COL_IS_REAL));
        }
    }
}
