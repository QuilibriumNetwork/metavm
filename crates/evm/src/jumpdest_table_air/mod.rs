//! JUMPDEST validity table AIR (roadmap #57).
//!
//! Per row: `(pc, is_jumpdest, is_real)` — one row per byte position in
//! the executing bytecode. `is_jumpdest = 1` iff the byte at `pc` is an
//! executable JUMPDEST (0x5B) in the instruction stream (NOT a byte
//! inside a PUSHx immediate). Built host-side from
//! [`crate::jumpdest::valid_jumpdest_bitmap`].
//!
//! # Cross-AIR LogUp binding
//!
//! [`make_evm_jump_to_jumpdest_table_descriptor`] proves that every
//! EVM main row firing `JUMP` (opcode 0x56) has a `(target, 1)` tuple
//! that appears in this table's `(pc, is_jumpdest)` rows where
//! `is_jumpdest = 1`. Multiset equality under a single random γ
//! algebraically forces every JUMP target to land on a valid JUMPDEST.
//!
//! # JUMPI gating limitation
//!
//! JUMPI (0x57) only branches when its second stack input (the
//! condition) is non-zero. A pure LogUp lookup cannot natively gate on
//! "condition != 0" because LogUp selectors are 0/1-valued, not
//! arbitrary field-element-zero tests. The lookup machinery would
//! incorrectly include not-taken JUMPI rows in the A-side multiset, and
//! these rows have garbage targets (whatever happened to be on stack
//! top) that won't be in the JUMPDEST set — a sound but
//! over-restrictive constraint that would falsely reject honest traces.
//!
//! The clean fix: introduce a synthetic EVM main column
//! `COL_SEL_JUMPI_TAKEN` populated by the inspector that is set to 1
//! iff `opcode == 0x57 AND condition != 0`. We can then gate the JUMPI
//! descriptor on that column. Until that column lands, the JUMPI
//! descriptor here is *shape-only*: callers should treat it as a
//! planned constraint, not an active one.
//!
//! Algebraic constraints:
//!   (0) is_real binary
//!   (1) is_jumpdest binary
//!   (2) is_jumpdest * (1 - is_real) = 0   [is_jumpdest only on real rows]

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

pub const COL_PC: usize = 0;
pub const COL_IS_JUMPDEST: usize = 1;
pub const COL_IS_REAL: usize = 2;
pub const NUM_COLUMNS: usize = 3;

pub const NUM_ROW_CONSTRAINTS: usize = 3;

#[derive(Clone, Debug)]
pub struct JumpdestTableRow {
    pub pc: u64,
    pub is_jumpdest: bool,
}

#[derive(Clone, Debug, Default)]
pub struct JumpdestTableWitness {
    pub rows: Vec<JumpdestTableRow>,
}

impl JumpdestTableWitness {
    pub fn from_rows(rows: Vec<JumpdestTableRow>) -> Self {
        Self { rows }
    }

    /// Build a witness from raw bytecode. Emits one row per byte
    /// position, with `is_jumpdest = true` iff that byte is a valid
    /// JUMPDEST per [`crate::jumpdest::valid_jumpdest_bitmap`].
    pub fn from_bytecode(bytecode: &[u8]) -> Self {
        let bitmap = crate::jumpdest::valid_jumpdest_bitmap(bytecode);
        let rows = bitmap
            .into_iter()
            .enumerate()
            .map(|(pc, is_jd)| JumpdestTableRow {
                pc: pc as u64,
                is_jumpdest: is_jd,
            })
            .collect();
        Self { rows }
    }
}

pub fn build_trace_polynomials(
    w: &JumpdestTableWitness,
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
        cols[COL_IS_JUMPDEST][r] = if row.is_jumpdest {
            one.clone()
        } else {
            zero.clone()
        };
        cols[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial {
            evaluations: e,
            degree: num_rows,
        })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

pub struct JumpdestTableConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl JumpdestTableConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            omega: None,
            domain_size: None,
        }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for JumpdestTableConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }
    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_jumpdest_binary".into(),
            "is_jumpdest_only_on_real".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let mut bin_real = vec![Scalar::zero(curve); n];
        let mut bin_jd = vec![Scalar::zero(curve); n];
        let mut jd_only_real = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let real = &columns[COL_IS_REAL][r];
            let jd = &columns[COL_IS_JUMPDEST][r];
            bin_real[r] = real.mul(&real.sub(&one));
            bin_jd[r] = jd.mul(&jd.sub(&one));
            jd_only_real[r] = jd.mul(&one.sub(real));
        }
        vec![bin_real, bin_jd, jd_only_real]
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let real = &ce[COL_IS_REAL];
        let jd = &ce[COL_IS_JUMPDEST];
        let bin_real = real.mul(&real.sub(&one));
        let bin_jd = jd.mul(&jd.sub(&one));
        let jd_only_real = jd.mul(&one.sub(real));
        // RLC by powers of alpha.
        let alpha2 = alpha.mul(alpha);
        bin_real.add(&alpha.mul(&bin_jd)).add(&alpha2.mul(&jd_only_real))
    }

    fn build_constraint_polynomial(
        &self,
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let real = &cc[COL_IS_REAL];
        let jd = &cc[COL_IS_JUMPDEST];

        // bin_real = real * (real - 1)
        let real_m1 = poly_sub(real, &one_p, curve);
        let bin_real = poly_mul(real, &real_m1, curve);

        // bin_jd = jd * (jd - 1)
        let jd_m1 = poly_sub(jd, &one_p, curve);
        let bin_jd = poly_mul(jd, &jd_m1, curve);

        // jd_only_real = jd * (1 - real)
        let one_minus_real = poly_sub(&one_p, real, curve);
        let jd_only_real = poly_mul(jd, &one_minus_real, curve);

        // RLC by α, α^2.
        let alpha2 = alpha.mul(alpha);
        let scaled_jd = {
            let mut v = bin_jd.clone();
            for c in v.iter_mut() {
                *c = c.mul(alpha);
            }
            v
        };
        let scaled_jd_only = {
            let mut v = jd_only_real.clone();
            for c in v.iter_mut() {
                *c = c.mul(&alpha2);
            }
            v
        };
        // Sum the three polynomials.
        let max_len = bin_real
            .len()
            .max(scaled_jd.len())
            .max(scaled_jd_only.len());
        let zero = Scalar::zero(curve);
        let mut out = vec![zero; max_len];
        for (i, c) in bin_real.iter().enumerate() {
            out[i] = out[i].add(c);
        }
        for (i, c) in scaled_jd.iter().enumerate() {
            out[i] = out[i].add(c);
        }
        for (i, c) in scaled_jd_only.iter().enumerate() {
            out[i] = out[i].add(c);
        }
        out
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
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
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
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp descriptors ────────────────────────────────────

/// EVM JUMP rows → JUMPDEST table.
///
/// A side: EVM main rows gated by `COL_SEL_JUMP`. Tuple = (input0_l0,
/// constant 1). The "constant 1" entry pins the JUMPDEST flag.
///
/// B side: jumpdest_table rows gated by `COL_IS_JUMPDEST`. Tuple =
/// (pc, is_jumpdest). Multiset equality forces each EVM JUMP's target
/// (the stack-top input0) to equal some `pc` whose `is_jumpdest = 1`.
///
/// NOTE: `COL_SEL_JUMP` is an umbrella selector that fires for *both*
/// JUMP and JUMPI rows. This descriptor over-includes JUMPI rows
/// (including not-taken ones), so it is intended to be used in
/// conjunction with a future `COL_SEL_JUMP_ONLY` (set only when
/// opcode == 0x56). Until that column lands, callers who only want to
/// constrain `JUMP` should add a synthetic selector via the inspector.
pub fn make_evm_jump_to_jumpdest_table_descriptor(
    evm_layer: usize,
    table_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_INPUT0_L0, COL_SEL_JUMP};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_jump_to_jumpdest_table_v1".into(),
        a_layer_index: evm_layer,
        a_columns: vec![COL_INPUT0_L0, COL_IS_JUMPDEST_CONST_ONE_PLACEHOLDER],
        a_selector_column: Some(COL_SEL_JUMP),
        b_layer_index: table_layer,
        b_columns: vec![COL_PC, COL_IS_JUMPDEST],
        b_selector_column: Some(COL_IS_JUMPDEST),
    }
}

/// EVM JUMPI taken-branch rows → JUMPDEST table.
///
/// **Gating limitation**: a pure LogUp descriptor cannot express
/// "select only if input1 (condition) is non-zero". The proper closure
/// requires a synthetic EVM column `COL_SEL_JUMPI_TAKEN` populated by
/// the inspector to be 1 iff `opcode == 0x57 AND input1 != 0`. This
/// descriptor names that selector by index `COL_SEL_JUMPI_TAKEN_PLACEHOLDER`
/// — when the column is actually added to the EVM trace it should
/// replace this placeholder. Until then, callers should NOT activate
/// this descriptor (the placeholder index is out-of-range for the
/// current EVM trace and will fail trace lookup).
pub fn make_evm_jumpi_to_jumpdest_table_descriptor(
    evm_layer: usize,
    table_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::COL_INPUT0_L0;
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_jumpi_taken_to_jumpdest_table_v1".into(),
        a_layer_index: evm_layer,
        a_columns: vec![COL_INPUT0_L0, COL_IS_JUMPDEST_CONST_ONE_PLACEHOLDER],
        a_selector_column: Some(COL_SEL_JUMPI_TAKEN_PLACEHOLDER),
        b_layer_index: table_layer,
        b_columns: vec![COL_PC, COL_IS_JUMPDEST],
        b_selector_column: Some(COL_IS_JUMPDEST),
    }
}

/// Placeholder index for "a column that is constantly 1 on every EVM
/// row" — the EVM trace doesn't currently expose such a column, so the
/// descriptor's A-side tuple second element is a planning placeholder.
/// A future EVM trace revision should expose either a literal `COL_ONE`
/// or use a per-opcode synthetic column. Setting this to a sentinel
/// out-of-range value documents the unfinished wiring.
pub const COL_IS_JUMPDEST_CONST_ONE_PLACEHOLDER: usize = usize::MAX - 1;

/// Placeholder index for the synthetic `COL_SEL_JUMPI_TAKEN` column
/// described above. Out-of-range to make accidental use loud.
pub const COL_SEL_JUMPI_TAKEN_PLACEHOLDER: usize = usize::MAX - 2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jumpdest::{OPCODE_JUMPDEST, OPCODE_PUSH1};

    #[test]
    fn simple_bytecode_table_builds() {
        // JUMPDEST, STOP — one valid JUMPDEST at PC 0.
        let bc = vec![OPCODE_JUMPDEST, 0x00];
        let w = JumpdestTableWitness::from_bytecode(&bc);
        assert_eq!(w.rows.len(), 2);
        assert_eq!(w.rows[0].pc, 0);
        assert!(w.rows[0].is_jumpdest);
        assert_eq!(w.rows[1].pc, 1);
        assert!(!w.rows[1].is_jumpdest);
    }

    #[test]
    fn jumpdest_at_offset_4_detected() {
        // PUSH1 1, PUSH1 2, ADD, JUMPDEST(=PC 5), STOP — JUMPDEST is
        // actually at PC=5 in this 7-byte snippet. We pick offset 4
        // semantics: build a snippet so a JUMPDEST sits at PC=4.
        // ADD at PC 0, JUMPDEST at PC 1 — no, let's craft: 4 NOPs then JUMPDEST.
        // Easiest: PUSH1 1, PUSH1 2, JUMPDEST(=PC 4), STOP.
        let bc = vec![0x60, 0x01, 0x60, 0x02, OPCODE_JUMPDEST, 0x00];
        let w = JumpdestTableWitness::from_bytecode(&bc);
        assert_eq!(w.rows.len(), 6);
        for (i, row) in w.rows.iter().enumerate() {
            assert_eq!(row.pc, i as u64);
            assert_eq!(row.is_jumpdest, i == 4, "row {} jumpdest mismatch", i);
        }
    }

    #[test]
    fn push_immediate_does_not_count_as_jumpdest() {
        // PUSH1 0x5B (the operand looks like JUMPDEST but is data).
        // Bytecode: [PUSH1, 0x5B, STOP] — neither offset should be
        // is_jumpdest = true.
        let bc = vec![OPCODE_PUSH1, OPCODE_JUMPDEST, 0x00];
        let w = JumpdestTableWitness::from_bytecode(&bc);
        assert_eq!(w.rows.len(), 3);
        for row in &w.rows {
            assert!(!row.is_jumpdest, "PC {} should not be jumpdest", row.pc);
        }
    }

    #[test]
    fn honest_witness_constraints_zero() {
        let bc = vec![
            OPCODE_JUMPDEST,
            OPCODE_PUSH1, OPCODE_JUMPDEST,
            OPCODE_JUMPDEST,
            0x00,
        ];
        let w = JumpdestTableWitness::from_bytecode(&bc);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = JumpdestTableConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} non-zero",
                    i, r
                );
            }
        }
    }

    #[test]
    fn jump_descriptor_well_formed() {
        let d = make_evm_jump_to_jumpdest_table_descriptor(0, 1);
        assert_eq!(d.label, "evm_jump_to_jumpdest_table_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), 2);
        assert_eq!(d.b_columns.len(), 2);
        assert_eq!(d.b_columns, vec![COL_PC, COL_IS_JUMPDEST]);
        assert_eq!(d.b_selector_column, Some(COL_IS_JUMPDEST));
        assert_eq!(d.a_selector_column, Some(crate::trace::COL_SEL_JUMP));
    }

    #[test]
    fn jumpi_descriptor_documents_placeholder_gating() {
        let d = make_evm_jumpi_to_jumpdest_table_descriptor(2, 3);
        assert_eq!(d.label, "evm_jumpi_taken_to_jumpdest_table_v1");
        // The A-side selector should be the placeholder — make it loud.
        assert_eq!(d.a_selector_column, Some(COL_SEL_JUMPI_TAKEN_PLACEHOLDER));
        assert_eq!(d.a_columns[1], COL_IS_JUMPDEST_CONST_ONE_PLACEHOLDER);
    }

    #[test]
    fn empty_bytecode_yields_empty_witness() {
        let w = JumpdestTableWitness::from_bytecode(&[]);
        assert_eq!(w.rows.len(), 0);
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let bc = vec![OPCODE_JUMPDEST, 0x00, OPCODE_JUMPDEST];
        let w = JumpdestTableWitness::from_bytecode(&bc);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = JumpdestTableConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(11, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..t.num_rows {
            let row_evals: Vec<Scalar> = cr.iter().map(|c| c[r].clone()).collect();
            assert!(
                cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
                "row {} evaluate_at_point non-zero",
                r
            );
        }
    }
}
