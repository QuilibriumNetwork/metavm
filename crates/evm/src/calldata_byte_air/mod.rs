//! Calldata byte-memory AIR.
//!
//! Per row: (offset, byte_value, is_real). Represents the
//! transaction's calldata as a sorted byte sequence. CALLDATALOAD
//! rows in the EVM trace can cross-AIR LogUp to 32 entries in this
//! AIR to algebraically bind the loaded U256 to calldata[offset..+32].

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

pub const COL_OFFSET: usize = 0;
pub const COL_BYTE_VAL: usize = 1;
pub const COL_IS_REAL: usize = 2;
pub const NUM_COLUMNS: usize = 3;

pub const NUM_ROW_CONSTRAINTS: usize = 1; // is_real binary

#[derive(Clone, Debug)]
pub struct CalldataRow {
    pub offset: u64,
    pub byte_val: u8,
}

#[derive(Clone, Debug, Default)]
pub struct CalldataWitness { pub rows: Vec<CalldataRow> }

impl CalldataWitness {
    pub fn from_calldata(calldata: &[u8]) -> Self {
        let rows = calldata.iter().enumerate()
            .map(|(i, &b)| CalldataRow { offset: i as u64, byte_val: b })
            .collect();
        Self { rows }
    }
}

pub fn build_trace_polynomials(w: &CalldataWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_OFFSET][r] = Scalar::from_u64(row.offset, curve);
        cols[COL_BYTE_VAL][r] = Scalar::from_u64(row.byte_val as u64, curve);
        cols[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols.into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows }).collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

pub struct CalldataConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl CalldataConstraintSystem {
    pub fn new(num_rows: usize) -> Self { Self { num_rows, omega: None, domain_size: None } }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega); self.domain_size = Some(domain_size); self
    }
}

impl VmConstraintSystem for CalldataConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }
    fn constraint_labels(&self) -> Vec<String> { vec!["is_real_binary".into()] }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let mut bin = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            bin[r] = v.mul(&v.sub(&one));
        }
        vec![bin]
    }
    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let one = Scalar::one(alpha.curve_type());
        let v = &ce[COL_IS_REAL];
        v.mul(&v.sub(&one))
    }
    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let v = &cc[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_p, curve);
        poly_mul(v, &v_m1, curve)
    }
    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS { return; }
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) { *cell = zero.clone(); }
        }
    }
    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(8)];
        let decls = vec![(LookupDeclaration {
            label: "calldata_byte_8bit".into(),
            column_index: COL_BYTE_VAL,
            max_bits: 8,
            selector_column: None,
        }, 0)];
        LookupRequirements { tables, declarations: decls }
    }
}

// ─── Cross-AIR LogUp descriptor builders ─────────────────────────────

/// Per-byte CALLDATALOAD descriptor: at byte position `k` (0..32),
/// the EVM main row's output byte (extracted from output limbs) should
/// match calldata_byte_air's (offset = input0_l0 + k, byte_val).
pub fn make_calldataload_byte_descriptor(
    evm_layer: usize,
    cd_layer: usize,
    _byte_index: usize, // documentation only — for now a single descriptor at offset 0
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_INPUT0_L0, COL_OUTPUT0_L0};
    // Simplified: bind offset → first output byte. Full per-byte binding
    // requires 32 separate descriptors with byte-extraction columns.
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_calldataload_byte0_v1".into(),
        a_layer_index: evm_layer,
        a_columns: vec![COL_INPUT0_L0, COL_OUTPUT0_L0],
        a_selector_column: None, // would be COL_SEL_CALLDATALOAD if it existed
        b_layer_index: cd_layer,
        b_columns: vec![COL_OFFSET, COL_BYTE_VAL],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honest_witness_constraints_zero() {
        let calldata = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34];
        let w = CalldataWitness::from_calldata(&calldata);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CalldataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for body in cs.evaluate_on_domain(&cr, t.num_rows).iter() {
            for v in body.iter() {
                assert!(v.is_zero());
            }
        }
    }

    #[test]
    fn witness_from_empty_calldata() {
        let w = CalldataWitness::from_calldata(&[]);
        assert_eq!(w.rows.len(), 0);
    }

    #[test]
    fn calldataload_descriptor_well_formed() {
        let d = make_calldataload_byte_descriptor(0, 1, 0);
        assert_eq!(d.label, "evm_calldataload_byte0_v1");
        assert_eq!(d.a_columns.len(), 2);
        assert_eq!(d.b_columns.len(), 2);
    }

    #[test]
    fn witness_byte_order_preserved() {
        let calldata = vec![0x42, 0x11];
        let w = CalldataWitness::from_calldata(&calldata);
        assert_eq!(w.rows[0].offset, 0);
        assert_eq!(w.rows[0].byte_val, 0x42);
        assert_eq!(w.rows[1].offset, 1);
        assert_eq!(w.rows[1].byte_val, 0x11);
    }
}
