//! ADDRESS opcode AIR (0x30).
//!
//! Proves algebraically that on an ADDRESS row:
//!
//!   - The opcode equals 0x30.
//!   - The static gas cost equals 2 (`G_BASE`).
//!   - The pushed u256 (4 LE u64 limbs) equals the current contract's
//!     20-byte address packed using the codebase's canonical little-endian
//!     limb convention:
//!         limb[0] = u64::from_le_bytes(bytes[0..8])
//!         limb[1] = u64::from_le_bytes(bytes[8..16])
//!         limb[2] = u64::from_le_bytes(bytes[16..20] || 0x00..00)
//!         limb[3] = 0
//!     (Matches [`crate::inspector::address_to_limbs`] used to populate
//!     `frame_callee` in the EVM trace.)
//!
//! Composed with neighboring AIRs via cross-AIR LogUp:
//!
//!   - `make_address_to_call_frame_descriptor` joins the 20-byte address
//!     (as 4 limbs) on real rows to `call_frame_air`'s callee_pre column,
//!     pinning the current contract address to the call frame's notion of
//!     the callee.
//!   - `make_address_to_stack_contents_descriptor` joins `(pc, value_limbs)`
//!     on real rows to `stack_contents_air`'s unsorted view, pinning the
//!     stack push event at this PC to the same u256 value.
//!
//! ## Per-row layout (`NUM_COLUMNS = 28`)
//!
//! ```text
//! offset  meaning
//!  0      pc                       (u64)
//!  1..21  address_byte[0..20]      (u8 per limb; bytes of the 20-byte address)
//! 21      value_limb_0             (LE u64 of pushed u256)
//! 22      value_limb_1
//! 23      value_limb_2
//! 24      value_limb_3             (must be 0)
//! 25      gas_cost                 (must be 2)
//! 26      opcode                   (must be 0x30)
//! 27      is_real                  (binary)
//! ```
//!
//! ## Constraint catalog (`NUM_ROW_CONSTRAINTS = 9`)
//!
//!  0. `is_real_binary`        — `is_real · (is_real − 1) = 0`
//!  1. `gas_cost_eq_2`         — `is_real · (gas_cost − 2) = 0`
//!  2. `opcode_eq_address`     — `is_real · (opcode − 0x30) = 0`
//!  3. `value_limb_0_binding`  — `is_real · (limb0 − Σ bytes[i] · 256^i)`
//!  4. `value_limb_1_binding`  — `is_real · (limb1 − Σ bytes[8+i] · 256^i)`
//!  5. `value_limb_2_binding`  — `is_real · (limb2 − Σ bytes[16+i] · 256^i)` (i=0..4)
//!  6. `value_limb_3_zero`     — `is_real · limb3 = 0`
//!  7. `padding_opcode_zero`   — `(1 − is_real) · opcode = 0`
//!  8. `padding_gas_cost_zero` — `(1 − is_real) · gas_cost = 0`
//!
//! Byte range checks on each of the 20 address bytes plus the 1-bit
//! `is_real` and 8-bit `opcode` columns.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_PC: usize = 0;
// 20 address bytes [LE-indexed: byte 0 = least significant]
pub const COL_ADDR_BYTE_0: usize = 1;
pub const NUM_ADDR_BYTES: usize = 20;
pub const COL_VALUE_LIMB_0: usize = COL_ADDR_BYTE_0 + NUM_ADDR_BYTES; // 21
pub const COL_VALUE_LIMB_1: usize = COL_VALUE_LIMB_0 + 1; // 22
pub const COL_VALUE_LIMB_2: usize = COL_VALUE_LIMB_0 + 2; // 23
pub const COL_VALUE_LIMB_3: usize = COL_VALUE_LIMB_0 + 3; // 24
pub const COL_GAS_COST: usize = 25;
pub const COL_OPCODE: usize = 26;
pub const COL_IS_REAL: usize = 27;

pub const NUM_COLUMNS: usize = 28;
pub const NUM_ROW_CONSTRAINTS: usize = 9;
pub const NUM_SHIFTED: usize = 0;

const _: () = assert!(NUM_COLUMNS == 28);

/// ADDRESS opcode byte.
pub const ADDRESS_OPCODE: u8 = 0x30;
/// ADDRESS static gas cost (`G_base`).
pub const ADDRESS_GAS_COST: u64 = 2;

// ─── Address ↔ limb conversion ────────────────────────────────────────

/// Pack a 20-byte address into the codebase's canonical 4-limb LE form.
/// This must match [`crate::inspector::address_to_limbs`] so that the
/// cross-AIR LogUp link to `frame_callee` aligns tuple-wise.
pub fn address_to_limbs(addr: &[u8; 20]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for i in 0..2 {
        let mut tmp = [0u8; 8];
        tmp.copy_from_slice(&addr[8 * i..8 * i + 8]);
        limbs[i] = u64::from_le_bytes(tmp);
    }
    let mut tmp = [0u8; 8];
    tmp[..4].copy_from_slice(&addr[16..20]);
    limbs[2] = u64::from_le_bytes(tmp);
    limbs
}

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddressOpcodeRow {
    pub pc: u64,
    pub address: [u8; 20],
    pub value: [u64; 4],
    pub gas_cost: u64,
    pub opcode: u8,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AddressOpcodeWitness {
    pub rows: Vec<AddressOpcodeRow>,
}

impl AddressOpcodeWitness {
    /// Build a witness from a slice of `(pc, address)` events.
    /// Each event populates a row with `value = address_to_limbs(address)`,
    /// `gas_cost = 2`, `opcode = 0x30`, `is_real = true`.
    pub fn from_events(events: &[(u64, [u8; 20])]) -> Self {
        let rows = events
            .iter()
            .map(|&(pc, address)| AddressOpcodeRow {
                pc,
                address,
                value: address_to_limbs(&address),
                gas_cost: ADDRESS_GAS_COST,
                opcode: ADDRESS_OPCODE,
                is_real: true,
            })
            .collect();
        Self { rows }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(w: &AddressOpcodeWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_PC][r] = Scalar::from_u64(row.pc, curve);
        for i in 0..NUM_ADDR_BYTES {
            cols[COL_ADDR_BYTE_0 + i][r] = Scalar::from_u64(row.address[i] as u64, curve);
        }
        cols[COL_VALUE_LIMB_0][r] = Scalar::from_u64(row.value[0], curve);
        cols[COL_VALUE_LIMB_1][r] = Scalar::from_u64(row.value[1], curve);
        cols[COL_VALUE_LIMB_2][r] = Scalar::from_u64(row.value[2], curve);
        cols[COL_VALUE_LIMB_3][r] = Scalar::from_u64(row.value[3], curve);
        cols[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
        cols[COL_OPCODE][r] = Scalar::from_u64(row.opcode as u64, curve);
        cols[COL_IS_REAL][r] = if row.is_real { one.clone() } else { zero.clone() };
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Helpers ──────────────────────────────────────────────────────────

/// Build the 8 byte-weights `[1, 256, 256^2, ..., 256^7]` in the field.
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

pub struct AddressOpcodeConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AddressOpcodeConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for AddressOpcodeConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "gas_cost_eq_2".into(),
            "opcode_eq_address".into(),
            "value_limb_0_binding".into(),
            "value_limb_1_binding".into(),
            "value_limb_2_binding".into(),
            "value_limb_3_zero".into(),
            "padding_opcode_zero".into(),
            "padding_gas_cost_zero".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(ADDRESS_GAS_COST, curve);
        let op_address = Scalar::from_u64(ADDRESS_OPCODE as u64, curve);
        let w = byte_weights(curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let gc = &columns[COL_GAS_COST][r];
            let op = &columns[COL_OPCODE][r];
            let v0 = &columns[COL_VALUE_LIMB_0][r];
            let v1 = &columns[COL_VALUE_LIMB_1][r];
            let v2 = &columns[COL_VALUE_LIMB_2][r];
            let v3 = &columns[COL_VALUE_LIMB_3][r];
            let one_minus_real = one.sub(is_real);

            // 0: is_real binary
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            // 1: gas_cost == 2
            bodies[1][r] = is_real.mul(&gc.sub(&two));
            // 2: opcode == 0x30
            bodies[2][r] = is_real.mul(&op.sub(&op_address));

            // 3..5: limb_k == Σ byte_{8k+i} · 256^i for k = 0,1; k=2 uses 4 bytes.
            // Limb 0
            let mut s0 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_ADDR_BYTE_0 + i][r];
                s0 = s0.add(&b.mul(&w[i]));
            }
            bodies[3][r] = is_real.mul(&v0.sub(&s0));
            // Limb 1
            let mut s1 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_ADDR_BYTE_0 + 8 + i][r];
                s1 = s1.add(&b.mul(&w[i]));
            }
            bodies[4][r] = is_real.mul(&v1.sub(&s1));
            // Limb 2 (only 4 bytes — high 4 bytes are zero)
            let mut s2 = Scalar::zero(curve);
            for i in 0..4 {
                let b = &columns[COL_ADDR_BYTE_0 + 16 + i][r];
                s2 = s2.add(&b.mul(&w[i]));
            }
            bodies[5][r] = is_real.mul(&v2.sub(&s2));
            // 6: limb 3 == 0
            bodies[6][r] = is_real.mul(v3);
            // 7: padding rows have opcode = 0
            bodies[7][r] = one_minus_real.mul(op);
            // 8: padding rows have gas_cost = 0
            bodies[8][r] = one_minus_real.mul(gc);
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(ADDRESS_GAS_COST, curve);
        let op_address = Scalar::from_u64(ADDRESS_OPCODE as u64, curve);
        let w = byte_weights(curve);
        let is_real = &ce[COL_IS_REAL];
        let gc = &ce[COL_GAS_COST];
        let op = &ce[COL_OPCODE];
        let v0 = &ce[COL_VALUE_LIMB_0];
        let v1 = &ce[COL_VALUE_LIMB_1];
        let v2 = &ce[COL_VALUE_LIMB_2];
        let v3 = &ce[COL_VALUE_LIMB_3];
        let one_minus_real = one.sub(is_real);

        let mut s0 = Scalar::zero(curve);
        for i in 0..8 { s0 = s0.add(&ce[COL_ADDR_BYTE_0 + i].mul(&w[i])); }
        let mut s1 = Scalar::zero(curve);
        for i in 0..8 { s1 = s1.add(&ce[COL_ADDR_BYTE_0 + 8 + i].mul(&w[i])); }
        let mut s2 = Scalar::zero(curve);
        for i in 0..4 { s2 = s2.add(&ce[COL_ADDR_BYTE_0 + 16 + i].mul(&w[i])); }

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_real.mul(&gc.sub(&two)),
            is_real.mul(&op.sub(&op_address)),
            is_real.mul(&v0.sub(&s0)),
            is_real.mul(&v1.sub(&s1)),
            is_real.mul(&v2.sub(&s2)),
            is_real.mul(v3),
            one_minus_real.mul(op),
            one_minus_real.mul(gc),
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
        let two_p = vec![Scalar::from_u64(ADDRESS_GAS_COST, curve)];
        let op_addr_p = vec![Scalar::from_u64(ADDRESS_OPCODE as u64, curve)];
        let w = byte_weights(curve);
        let is_real = &cc[COL_IS_REAL];
        let gc = &cc[COL_GAS_COST];
        let op = &cc[COL_OPCODE];
        let v0 = &cc[COL_VALUE_LIMB_0];
        let v1 = &cc[COL_VALUE_LIMB_1];
        let v2 = &cc[COL_VALUE_LIMB_2];
        let v3 = &cc[COL_VALUE_LIMB_3];
        let one_minus_real = poly_sub(&one_p, is_real, curve);

        let mut s0: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_ADDR_BYTE_0 + i], &w[i]);
            s0 = poly_add(&s0, &scaled, curve);
        }
        let mut s1: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_ADDR_BYTE_0 + 8 + i], &w[i]);
            s1 = poly_add(&s1, &scaled, curve);
        }
        let mut s2: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..4 {
            let scaled = poly_scalar_mul(&cc[COL_ADDR_BYTE_0 + 16 + i], &w[i]);
            s2 = poly_add(&s2, &scaled, curve);
        }

        let bodies: Vec<Vec<Scalar>> = vec![
            poly_mul(is_real, &poly_sub(is_real, &one_p, curve), curve),
            poly_mul(is_real, &poly_sub(gc, &two_p, curve), curve),
            poly_mul(is_real, &poly_sub(op, &op_addr_p, curve), curve),
            poly_mul(is_real, &poly_sub(v0, &s0, curve), curve),
            poly_mul(is_real, &poly_sub(v1, &s1, curve), curve),
            poly_mul(is_real, &poly_sub(v2, &s2, curve), curve),
            poly_mul(is_real, v3, curve),
            poly_mul(&one_minus_real, op, curve),
            poly_mul(&one_minus_real, gc, curve),
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
        // is_real binary range check
        declarations.push((
            LookupDeclaration {
                label: "address_opcode_is_real_1bit".into(),
                column_index: COL_IS_REAL,
                max_bits: 1,
                selector_column: None,
            },
            tbl_bit,
        ));
        // opcode 8-bit range
        declarations.push((
            LookupDeclaration {
                label: "address_opcode_opcode_8bit".into(),
                column_index: COL_OPCODE,
                max_bits: 8,
                selector_column: None,
            },
            tbl_byte,
        ));
        // 20 address-byte range checks
        for i in 0..NUM_ADDR_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("address_opcode_addr_byte_{i}_8bit"),
                    column_index: COL_ADDR_BYTE_0 + i,
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

/// ADDRESS AIR → `call_frame_air`. Binds the gadget's 4-limb address
/// representation `(value_limb_0..3)` on real rows to the call-frame
/// gadget's `callee_pre` column tuple. Combined with constraints 3..6
/// (limb-from-bytes binding + limb3 = 0), this proves that the value
/// pushed on the stack equals the current frame's callee address.
pub fn make_address_to_call_frame_descriptor(
    address_layer_index: usize,
    call_frame_layer_index: usize,
    call_frame_selector_col: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "address_to_call_frame_callee_v1".into(),
        a_layer_index: address_layer_index,
        a_columns: vec![
            COL_VALUE_LIMB_0,
            COL_VALUE_LIMB_1,
            COL_VALUE_LIMB_2,
            COL_VALUE_LIMB_3,
        ],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: call_frame_layer_index,
        b_columns: vec![
            crate::call_frame_air::COL_CALLEE_PRE_L0,
            crate::call_frame_air::COL_CALLEE_PRE_L1,
            crate::call_frame_air::COL_CALLEE_PRE_L2,
            crate::call_frame_air::COL_CALLEE_PRE_L3,
        ],
        b_selector_column: Some(call_frame_selector_col),
    }
}

/// ADDRESS AIR → `stack_contents_air` (unsorted view). Binds the gadget's
/// `(pc, value_limb_0..3)` tuple on real rows to the stack-contents AIR's
/// unsorted view `(pc, value_limb_0..3)`. Combined with the limb-binding
/// constraints, this pins the stack write at this PC to the canonical
/// u256 encoding of the current contract address.
pub fn make_address_to_stack_contents_descriptor(
    address_layer_index: usize,
    stack_contents_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "address_to_stack_contents_push_v1".into(),
        a_layer_index: address_layer_index,
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

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_vanish(trace: &TracePolynomials, cs: &AddressOpcodeConstraintSystem) {
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

    fn sample_addr() -> [u8; 20] {
        // bytes 0..20 = 0x01, 0x02, ..., 0x14
        let mut a = [0u8; 20];
        for i in 0..20 { a[i] = (i as u8) + 1; }
        a
    }

    #[test]
    fn address_opcode_air_simple_address_vanishes() {
        let addr = sample_addr();
        let w = AddressOpcodeWitness::from_events(&[(42, addr)]);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].pc, 42);
        assert_eq!(w.rows[0].opcode, ADDRESS_OPCODE);
        assert_eq!(w.rows[0].gas_cost, ADDRESS_GAS_COST);
        assert_eq!(w.rows[0].value, address_to_limbs(&addr));
        // limb 3 must be zero (20 bytes < 32)
        assert_eq!(w.rows[0].value[3], 0);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AddressOpcodeConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn address_opcode_air_tampered_address_detected() {
        let curve = CurveType::Bls48581;
        let addr = sample_addr();
        let w = AddressOpcodeWitness::from_events(&[(0, addr)]);
        let mut t = build_trace_polynomials(&w, curve);
        // Tamper: change address byte 0 (but leave the value_limb_0 unchanged)
        // → byte-to-limb binding for limb 0 must fire.
        t.columns[COL_ADDR_BYTE_0].evaluations[0] = Scalar::from_u64(0xFF, curve);
        let cs = AddressOpcodeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[3][0].is_zero(), "value_limb_0_binding must fire");
    }

    #[test]
    fn address_opcode_air_tampered_gas_detected() {
        let curve = CurveType::Bls48581;
        let addr = sample_addr();
        let w = AddressOpcodeWitness::from_events(&[(0, addr)]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_GAS_COST].evaluations[0] = Scalar::from_u64(3, curve);
        let cs = AddressOpcodeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[1][0].is_zero(), "gas_cost_eq_2 must fire");
    }

    #[test]
    fn address_opcode_air_tampered_opcode_detected() {
        let curve = CurveType::Bls48581;
        let addr = sample_addr();
        let w = AddressOpcodeWitness::from_events(&[(0, addr)]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_OPCODE].evaluations[0] = Scalar::from_u64(0x33, curve); // CALLER
        let cs = AddressOpcodeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[2][0].is_zero(), "opcode_eq_address must fire");
    }

    #[test]
    fn address_opcode_air_tampered_limb_3_detected() {
        let curve = CurveType::Bls48581;
        let addr = sample_addr();
        let w = AddressOpcodeWitness::from_events(&[(0, addr)]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_VALUE_LIMB_3].evaluations[0] = Scalar::from_u64(1, curve);
        let cs = AddressOpcodeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[6][0].is_zero(), "value_limb_3_zero must fire");
    }

    #[test]
    fn address_opcode_air_multi_row_chain_honest() {
        let addr_a = sample_addr();
        let mut addr_b = [0u8; 20];
        for i in 0..20 { addr_b[i] = 0xAA; }
        let mut addr_c = [0u8; 20];
        for i in 0..20 { addr_c[i] = (0x80 + i as u8) & 0xFF; }
        let w = AddressOpcodeWitness::from_events(&[
            (0, addr_a),
            (10, addr_b),
            (42, addr_c),
        ]);
        assert_eq!(w.rows.len(), 3);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AddressOpcodeConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn address_opcode_air_descriptors_well_formed() {
        let d_cf = make_address_to_call_frame_descriptor(0, 1, crate::call_frame_air::COL_IS_REAL);
        assert_eq!(d_cf.label, "address_to_call_frame_callee_v1");
        assert_eq!(d_cf.a_layer_index, 0);
        assert_eq!(d_cf.b_layer_index, 1);
        assert_eq!(d_cf.a_columns.len(), 4);
        assert_eq!(d_cf.b_columns.len(), 4);
        assert_eq!(d_cf.a_columns[0], COL_VALUE_LIMB_0);
        assert_eq!(d_cf.a_columns[3], COL_VALUE_LIMB_3);
        assert_eq!(d_cf.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_cf.b_columns,
            vec![
                crate::call_frame_air::COL_CALLEE_PRE_L0,
                crate::call_frame_air::COL_CALLEE_PRE_L1,
                crate::call_frame_air::COL_CALLEE_PRE_L2,
                crate::call_frame_air::COL_CALLEE_PRE_L3,
            ]
        );

        let d_sc = make_address_to_stack_contents_descriptor(2, 3);
        assert_eq!(d_sc.label, "address_to_stack_contents_push_v1");
        assert_eq!(d_sc.a_layer_index, 2);
        assert_eq!(d_sc.b_layer_index, 3);
        assert_eq!(d_sc.a_columns.len(), 5);
        assert_eq!(d_sc.b_columns.len(), 5);
        assert_eq!(d_sc.a_columns[0], COL_PC);
        assert_eq!(d_sc.a_columns[1], COL_VALUE_LIMB_0);
        assert_eq!(d_sc.a_columns[4], COL_VALUE_LIMB_3);
        assert_eq!(d_sc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_sc.b_columns,
            vec![
                crate::stack_contents_air::COL_PC,
                crate::stack_contents_air::COL_VALUE_LIMB_0,
                crate::stack_contents_air::COL_VALUE_LIMB_1,
                crate::stack_contents_air::COL_VALUE_LIMB_2,
                crate::stack_contents_air::COL_VALUE_LIMB_3,
            ]
        );
        assert_eq!(
            d_sc.b_selector_column,
            Some(crate::stack_contents_air::COL_IS_REAL)
        );
    }

    #[test]
    fn address_opcode_air_column_layout_pinned() {
        assert_eq!(COL_PC, 0);
        assert_eq!(COL_ADDR_BYTE_0, 1);
        assert_eq!(NUM_ADDR_BYTES, 20);
        assert_eq!(COL_VALUE_LIMB_0, 21);
        assert_eq!(COL_VALUE_LIMB_1, 22);
        assert_eq!(COL_VALUE_LIMB_2, 23);
        assert_eq!(COL_VALUE_LIMB_3, 24);
        assert_eq!(COL_GAS_COST, 25);
        assert_eq!(COL_OPCODE, 26);
        assert_eq!(COL_IS_REAL, 27);
        assert_eq!(NUM_COLUMNS, 28);
        assert_eq!(NUM_ROW_CONSTRAINTS, 9);
        assert_eq!(NUM_SHIFTED, 0);
        assert_eq!(ADDRESS_OPCODE, 0x30);
        assert_eq!(ADDRESS_GAS_COST, 2);
    }

    #[test]
    fn address_opcode_air_evaluate_at_point_zero_on_honest() {
        let w = AddressOpcodeWitness::from_events(&[(11, sample_addr())]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AddressOpcodeConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xC0FFEE, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "evaluate_at_point should be zero on honest row");
    }

    #[test]
    fn address_opcode_air_lookup_declarations_well_formed() {
        let cs = AddressOpcodeConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 1 binary + 1 byte (opcode) + 20 address bytes = 22 declarations.
        assert_eq!(req.declarations.len(), 22);
        assert_eq!(req.tables.len(), 2);
        assert_eq!(req.tables[0].bits, 1);
        assert_eq!(req.tables[1].bits, 8);
    }

    #[test]
    fn address_to_limbs_matches_inspector_convention() {
        // Spot-check against the canonical convention:
        // address 0x0102030405060708 _ 0x090A0B0C0D0E0F10 _ 0x11121314
        let addr = sample_addr();
        let limbs = address_to_limbs(&addr);
        // limb 0: LE u64 of bytes 0..8 = bytes [0x01, 0x02, ..., 0x08] LE
        let expected_l0 = u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
        let expected_l1 = u64::from_le_bytes([9, 10, 11, 12, 13, 14, 15, 16]);
        let expected_l2 = u64::from_le_bytes([17, 18, 19, 20, 0, 0, 0, 0]);
        assert_eq!(limbs[0], expected_l0);
        assert_eq!(limbs[1], expected_l1);
        assert_eq!(limbs[2], expected_l2);
        assert_eq!(limbs[3], 0);
    }
}
