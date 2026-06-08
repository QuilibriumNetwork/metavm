//! BLOCKHASH history AIR.
//!
//! Stores the (block_number, block_hash) pairs accessible to BLOCKHASH
//! (the previous 256 blocks per EVM rules). Cross-AIR LogUp from EVM
//! BLOCKHASH rows binds the queried block number → returned hash.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

pub const COL_BLOCK_NUMBER: usize = 0;
pub const COL_HASH_L0: usize = 1;
pub const COL_HASH_L1: usize = 2;
pub const COL_HASH_L2: usize = 3;
pub const COL_HASH_L3: usize = 4;
pub const COL_IS_REAL: usize = 5;
pub const NUM_COLUMNS: usize = 6;

pub const NUM_ROW_CONSTRAINTS: usize = 1; // is_real binary

#[derive(Clone, Debug)]
pub struct HistoryRow {
    pub block_number: u64,
    pub hash: [u64; 4], // truncated to 256 bits low limbs
}

#[derive(Clone, Debug, Default)]
pub struct HistoryWitness { pub rows: Vec<HistoryRow> }

impl HistoryWitness {
    pub fn from_rows(rows: Vec<HistoryRow>) -> Self { Self { rows } }
}

pub fn build_trace_polynomials(w: &HistoryWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_BLOCK_NUMBER][r] = Scalar::from_u64(row.block_number, curve);
        for j in 0..4 { cols[COL_HASH_L0 + j][r] = Scalar::from_u64(row.hash[j], curve); }
        cols[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols.into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows }).collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

pub struct HistoryConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl HistoryConstraintSystem {
    pub fn new(num_rows: usize) -> Self { Self { num_rows, omega: None, domain_size: None } }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega); self.domain_size = Some(domain_size); self
    }
}

impl VmConstraintSystem for HistoryConstraintSystem {
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
        let body = poly_mul(v, &v_m1, curve);
        let _ = (poly_add::<>, poly_scalar_mul::<>);
        body
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
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

pub fn make_blockhash_descriptor(
    evm_layer: usize,
    history_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_INPUT0_L0, COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3,
        COL_SEL_BLOCKHASH,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_blockhash_to_history_v1".into(),
        a_layer_index: evm_layer,
        a_columns: vec![
            COL_INPUT0_L0, // queried block number (low limb)
            COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3,
        ],
        a_selector_column: Some(COL_SEL_BLOCKHASH),
        b_layer_index: history_layer,
        b_columns: vec![
            COL_BLOCK_NUMBER,
            COL_HASH_L0, COL_HASH_L1, COL_HASH_L2, COL_HASH_L3,
        ],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honest_history_constraints_zero() {
        let rows = vec![
            HistoryRow { block_number: 100, hash: [1, 2, 3, 4] },
            HistoryRow { block_number: 101, hash: [5, 6, 7, 8] },
        ];
        let w = HistoryWitness::from_rows(rows);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = HistoryConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for body in cs.evaluate_on_domain(&cr, t.num_rows).iter() {
            for v in body.iter() {
                assert!(v.is_zero());
            }
        }
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_blockhash_descriptor(0, 1);
        assert_eq!(d.label, "evm_blockhash_to_history_v1");
        assert_eq!(d.a_columns.len(), 5);
        assert_eq!(d.b_columns.len(), 5);
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let rows = vec![HistoryRow { block_number: 1, hash: [0; 4] }];
        let w = HistoryWitness::from_rows(rows);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = HistoryConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(7, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        assert!(cs.evaluate_at_point(&row_evals, &alpha).is_zero());
    }
}
