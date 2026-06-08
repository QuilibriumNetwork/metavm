//! EVM JUMP destination validation AIR.
//!
//! Round-4 control-flow gadget that proves every `JUMP` / `JUMPI` event
//! observed in the EVM trace lands on a valid `JUMPDEST` position of the
//! contract bytecode (i.e. not on a byte that is part of a `PUSHx`
//! immediate). One witness row per `JUMP` / `JUMPI` event.
//!
//! # Witness row schema
//!
//! - `pc` — program counter of the JUMP / JUMPI opcode itself.
//! - `dest` — branch destination (stack top at the opcode).
//! - `sel_jump` — binary; 1 iff opcode == 0x56 (JUMP).
//! - `sel_jumpi` — binary; 1 iff opcode == 0x57 (JUMPI).
//! - `condition` — JUMPI condition (stack[1]); irrelevant on JUMP rows.
//! - `condition_inv` — witnessed inverse helper: 0 iff condition == 0,
//!     `condition^-1` otherwise. Constraint `condition * condition_inv =
//!     is_taken` (when `sel_jumpi = 1`) algebraically forces
//!     `is_taken = 1 iff condition != 0`.
//! - `is_taken` — binary; 1 iff the branch was actually taken.
//! - `dest_le_bytes[0..4]` — little-endian byte decomposition of `dest`
//!     (≤ 2^32 PC range is fine for any honest contract).
//! - `is_real` — binary row activity flag.
//!
//! # Algebraic constraints (row-local, 13 total)
//!
//!   0. `is_real * (is_real - 1) = 0` — binary
//!   1. `sel_jump * (sel_jump - 1) = 0` — binary
//!   2. `sel_jumpi * (sel_jumpi - 1) = 0` — binary
//!   3. `is_taken * (is_taken - 1) = 0` — binary
//!   4. `sel_jump * sel_jumpi = 0` — disjoint
//!   5. `is_real - sel_jump - sel_jumpi = 0` — every real row is exactly
//!      one of the two
//!   6. `sel_jump * (1 - is_taken) = 0` — JUMP always taken
//!   7. `sel_jumpi * (is_taken - condition * condition_inv) = 0` —
//!      JUMPI: `is_taken = condition * condition_inv`. Combined with
//!      constraint 8 below, `is_taken = 1 iff condition != 0`.
//!   8. `sel_jumpi * condition * (1 - is_taken) = 0` — JUMPI:
//!      `condition != 0 ⇒ is_taken = 1`. Together with the binary
//!      constraint and (7), this rules out the malicious witness
//!      `condition != 0, condition_inv = 0, is_taken = 0`.
//!   9. `dest - Σ_k dest_le_bytes[k] * 256^k = 0` — byte decomp of dest
//!  10..=12. (placeholder slot; range checks are handled by the
//!      `LookupRequirements` byte range table.)
//!
//! # Byte range
//!
//! The 4 `dest_le_bytes` columns are each declared as 8-bit lookups via
//! [`lookup_declarations`].
//!
//! # Cross-AIR LogUp descriptor
//!
//! [`make_jump_validity_to_jumpdest_table_descriptor`] binds taken-branch
//! rows (`is_taken = 1`) to `jumpdest_table_air`'s `(pc, is_jumpdest)`
//! published table. A side: `(dest, const_1_placeholder)` gated by
//! `is_taken`. B side: `(pc, is_jumpdest)` gated by `is_jumpdest`. Under
//! a single shared γ multiset equality, every taken `dest` is forced to
//! equal some `pc` whose `is_jumpdest = 1` in the bytecode table — i.e.
//! a valid JUMPDEST.
//!
//! See [`crate::jumpdest_table_air::COL_IS_JUMPDEST_CONST_ONE_PLACEHOLDER`]
//! for the rationale behind the `const-1` placeholder column index.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ───────────────────────────────────────────────────
pub const COL_PC: usize = 0;
pub const COL_DEST: usize = 1;
pub const COL_SEL_JUMP: usize = 2;
pub const COL_SEL_JUMPI: usize = 3;
pub const COL_CONDITION: usize = 4;
pub const COL_CONDITION_INV: usize = 5;
pub const COL_IS_TAKEN: usize = 6;
pub const COL_DEST_BYTE_0: usize = 7;
pub const COL_DEST_BYTE_1: usize = 8;
pub const COL_DEST_BYTE_2: usize = 9;
pub const COL_DEST_BYTE_3: usize = 10;
pub const COL_IS_REAL: usize = 11;
pub const NUM_COLUMNS: usize = 12;

pub const NUM_DEST_BYTES: usize = 4;

// Row-local constraint count: see module docs.
pub const NUM_ROW_CONSTRAINTS: usize = 10;

// ─── Witness types ───────────────────────────────────────────────────

/// A single JUMP / JUMPI event observed in an EVM trace.
#[derive(Clone, Debug)]
pub struct JumpEvent {
    pub pc: u64,
    pub dest: u64,
    /// JUMPI condition (stack[1]); ignored when `sel_jump = 1`.
    pub condition: u64,
    pub sel_jump: bool,
    pub sel_jumpi: bool,
    /// True iff the branch was actually taken in the trace. The host
    /// computes this from `(sel_jump || (sel_jumpi && condition != 0))`.
    pub is_taken: bool,
}

#[derive(Clone, Debug, Default)]
pub struct JumpValidityWitness {
    pub rows: Vec<JumpEvent>,
}

impl JumpValidityWitness {
    pub fn from_events(rows: Vec<JumpEvent>) -> Self {
        Self { rows }
    }
}

/// Build a [`JumpValidityWitness`] from a slice of host-side JUMP events.
/// Normalizes `is_taken` so JUMP rows always have `is_taken = true`
/// and JUMPI rows have `is_taken = (condition != 0)`.
pub fn from_jump_events(events: &[JumpEvent]) -> JumpValidityWitness {
    let rows: Vec<JumpEvent> = events
        .iter()
        .map(|e| {
            let is_taken = if e.sel_jump {
                true
            } else if e.sel_jumpi {
                e.condition != 0
            } else {
                false
            };
            JumpEvent {
                pc: e.pc,
                dest: e.dest,
                condition: e.condition,
                sel_jump: e.sel_jump,
                sel_jumpi: e.sel_jumpi,
                is_taken,
            }
        })
        .collect();
    JumpValidityWitness { rows }
}

// ─── Trace builder ───────────────────────────────────────────────────

/// Compute the multiplicative inverse of a u64 value when interpreted
/// as a field scalar. Returns the zero scalar when `v == 0` (the
/// `condition * condition_inv` algebraic check tolerates that, as long
/// as the prover does not lie by setting `is_taken = 1`).
fn condition_inv_scalar(v: u64, curve: CurveType) -> Scalar {
    if v == 0 {
        Scalar::zero(curve)
    } else {
        Scalar::from_u64(v, curve).inverse()
    }
}

pub fn build_trace_polynomials(
    w: &JumpValidityWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_PC][r] = Scalar::from_u64(row.pc, curve);
        cols[COL_DEST][r] = Scalar::from_u64(row.dest, curve);
        cols[COL_SEL_JUMP][r] = if row.sel_jump { one.clone() } else { zero.clone() };
        cols[COL_SEL_JUMPI][r] = if row.sel_jumpi { one.clone() } else { zero.clone() };
        cols[COL_CONDITION][r] = Scalar::from_u64(row.condition, curve);
        cols[COL_CONDITION_INV][r] = condition_inv_scalar(row.condition, curve);
        cols[COL_IS_TAKEN][r] = if row.is_taken { one.clone() } else { zero.clone() };
        // dest as u32 little-endian byte decomposition (PC ≤ 2^24
        // realistically; we allow 4 bytes for headroom).
        let dest_u32 = row.dest as u32;
        let bytes = dest_u32.to_le_bytes();
        for k in 0..NUM_DEST_BYTES {
            cols[COL_DEST_BYTE_0 + k][r] = Scalar::from_u64(bytes[k] as u64, curve);
        }
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

// ─── Constraint system ───────────────────────────────────────────────

pub struct JumpValidityConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl JumpValidityConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for JumpValidityConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "sel_jump_binary".into(),
            "sel_jumpi_binary".into(),
            "is_taken_binary".into(),
            "sel_jump_xor_jumpi".into(),
            "is_real_eq_sum_selectors".into(),
            "jump_implies_taken".into(),
            "jumpi_taken_eq_cond_times_inv".into(),
            "jumpi_cond_nonzero_implies_taken".into(),
            "dest_byte_decomp".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let two56_pows: Vec<Scalar> = (0..NUM_DEST_BYTES)
            .map(|k| Scalar::from_u64(1u64 << (8 * k), curve))
            .collect();
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let sel_j = &columns[COL_SEL_JUMP][r];
            let sel_ji = &columns[COL_SEL_JUMPI][r];
            let is_taken = &columns[COL_IS_TAKEN][r];
            let cond = &columns[COL_CONDITION][r];
            let cond_inv = &columns[COL_CONDITION_INV][r];

            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            bodies[1][r] = sel_j.mul(&sel_j.sub(&one));
            bodies[2][r] = sel_ji.mul(&sel_ji.sub(&one));
            bodies[3][r] = is_taken.mul(&is_taken.sub(&one));
            bodies[4][r] = sel_j.mul(sel_ji);
            bodies[5][r] = is_real.sub(sel_j).sub(sel_ji);
            // 6: sel_jump * (1 - is_taken)
            bodies[6][r] = sel_j.mul(&one.sub(is_taken));
            // 7: sel_jumpi * (is_taken - condition * condition_inv)
            let prod = cond.mul(cond_inv);
            bodies[7][r] = sel_ji.mul(&is_taken.sub(&prod));
            // 8: sel_jumpi * condition * (1 - is_taken)
            bodies[8][r] = sel_ji.mul(&cond.mul(&one.sub(is_taken)));
            // 9: dest - Σ dest_bytes[k] * 256^k
            let mut acc = Scalar::zero(curve);
            for k in 0..NUM_DEST_BYTES {
                acc = acc.add(&columns[COL_DEST_BYTE_0 + k][r].mul(&two56_pows[k]));
            }
            bodies[9][r] = columns[COL_DEST][r].sub(&acc);
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two56_pows: Vec<Scalar> = (0..NUM_DEST_BYTES)
            .map(|k| Scalar::from_u64(1u64 << (8 * k), curve))
            .collect();
        let is_real = &ce[COL_IS_REAL];
        let sel_j = &ce[COL_SEL_JUMP];
        let sel_ji = &ce[COL_SEL_JUMPI];
        let is_taken = &ce[COL_IS_TAKEN];
        let cond = &ce[COL_CONDITION];
        let cond_inv = &ce[COL_CONDITION_INV];

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(sel_j.mul(&sel_j.sub(&one)));
        bodies.push(sel_ji.mul(&sel_ji.sub(&one)));
        bodies.push(is_taken.mul(&is_taken.sub(&one)));
        bodies.push(sel_j.mul(sel_ji));
        bodies.push(is_real.sub(sel_j).sub(sel_ji));
        bodies.push(sel_j.mul(&one.sub(is_taken)));
        let prod = cond.mul(cond_inv);
        bodies.push(sel_ji.mul(&is_taken.sub(&prod)));
        bodies.push(sel_ji.mul(&cond.mul(&one.sub(is_taken))));
        let mut acc = Scalar::zero(curve);
        for k in 0..NUM_DEST_BYTES {
            acc = acc.add(&ce[COL_DEST_BYTE_0 + k].mul(&two56_pows[k]));
        }
        bodies.push(ce[COL_DEST].sub(&acc));

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(
        &self,
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let is_real = &cc[COL_IS_REAL];
        let sel_j = &cc[COL_SEL_JUMP];
        let sel_ji = &cc[COL_SEL_JUMPI];
        let is_taken = &cc[COL_IS_TAKEN];
        let cond = &cc[COL_CONDITION];
        let cond_inv = &cc[COL_CONDITION_INV];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        // 0
        let real_m1 = poly_sub(is_real, &one_p, curve);
        bodies.push(poly_mul(is_real, &real_m1, curve));
        // 1
        let j_m1 = poly_sub(sel_j, &one_p, curve);
        bodies.push(poly_mul(sel_j, &j_m1, curve));
        // 2
        let ji_m1 = poly_sub(sel_ji, &one_p, curve);
        bodies.push(poly_mul(sel_ji, &ji_m1, curve));
        // 3
        let t_m1 = poly_sub(is_taken, &one_p, curve);
        bodies.push(poly_mul(is_taken, &t_m1, curve));
        // 4
        bodies.push(poly_mul(sel_j, sel_ji, curve));
        // 5
        bodies.push(poly_sub(&poly_sub(is_real, sel_j, curve), sel_ji, curve));
        // 6: sel_j * (1 - is_taken)
        let one_minus_taken = poly_sub(&one_p, is_taken, curve);
        bodies.push(poly_mul(sel_j, &one_minus_taken, curve));
        // 7: sel_ji * (is_taken - cond * cond_inv)
        let prod = poly_mul(cond, cond_inv, curve);
        let diff = poly_sub(is_taken, &prod, curve);
        bodies.push(poly_mul(sel_ji, &diff, curve));
        // 8: sel_ji * cond * (1 - is_taken)
        let cond_times_one_minus_taken = poly_mul(cond, &one_minus_taken, curve);
        bodies.push(poly_mul(sel_ji, &cond_times_one_minus_taken, curve));
        // 9: dest - Σ dest_byte_k * 256^k
        let mut acc: Vec<Scalar> = Vec::new();
        for k in 0..NUM_DEST_BYTES {
            let coeff = Scalar::from_u64(1u64 << (8 * k), curve);
            let term = poly_scalar_mul(&cc[COL_DEST_BYTE_0 + k], &coeff);
            if acc.is_empty() {
                acc = term;
            } else {
                acc = poly_add(&acc, &term, curve);
            }
        }
        bodies.push(poly_sub(&cc[COL_DEST], &acc, curve));

        // RLC.
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
    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS { return; }
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(8)];
        let mut decls = Vec::with_capacity(NUM_DEST_BYTES);
        for k in 0..NUM_DEST_BYTES {
            decls.push((
                LookupDeclaration {
                    label: format!("jump_validity_dest_byte_{}_8bit", k),
                    column_index: COL_DEST_BYTE_0 + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations: decls }
    }
}

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// Bind taken-branch `dest` values published by this AIR to the
/// `jumpdest_table_air`'s `(pc, is_jumpdest = 1)` rows. Multiset
/// equality forces every taken JUMP / JUMPI destination to equal some
/// valid JUMPDEST byte position in the bytecode.
///
/// **A side**: `(dest, const-1 placeholder)` gated by `is_taken`.
/// **B side**: `(pc, is_jumpdest)` gated by `is_jumpdest`.
///
/// The "const-1 placeholder" mirrors `jumpdest_table_air`'s convention
/// — there is no literal `COL_ONE` column on either AIR yet, so the
/// second tuple entry is a documented placeholder (sentinel) that must
/// be wired to a real `1`-valued column before the descriptor can be
/// activated in a real `joint_prove`. See
/// [`crate::jumpdest_table_air::COL_IS_JUMPDEST_CONST_ONE_PLACEHOLDER`].
pub fn make_jump_validity_to_jumpdest_table_descriptor(
    jump_layer: usize,
    table_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "jump_validity_to_jumpdest_table_v1".into(),
        a_layer_index: jump_layer,
        a_columns: vec![
            COL_DEST,
            crate::jumpdest_table_air::COL_IS_JUMPDEST_CONST_ONE_PLACEHOLDER,
        ],
        a_selector_column: Some(COL_IS_TAKEN),
        b_layer_index: table_layer,
        b_columns: vec![
            crate::jumpdest_table_air::COL_PC,
            crate::jumpdest_table_air::COL_IS_JUMPDEST,
        ],
        b_selector_column: Some(crate::jumpdest_table_air::COL_IS_JUMPDEST),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn jump_evt(pc: u64, dest: u64) -> JumpEvent {
        JumpEvent {
            pc,
            dest,
            condition: 0,
            sel_jump: true,
            sel_jumpi: false,
            is_taken: true,
        }
    }

    fn jumpi_evt(pc: u64, dest: u64, condition: u64) -> JumpEvent {
        JumpEvent {
            pc,
            dest,
            condition,
            sel_jump: false,
            sel_jumpi: true,
            is_taken: condition != 0,
        }
    }

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} non-zero", i, r);
            }
        }
    }

    #[test]
    fn simple_jump_witness_satisfies_constraints() {
        // Single JUMP from PC 2 to PC 5.
        let events = vec![jump_evt(2, 5)];
        let w = from_jump_events(&events);
        assert_eq!(w.rows.len(), 1);
        assert!(w.rows[0].is_taken);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = JumpValidityConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn jumpi_taken_and_not_taken_both_satisfy() {
        // Row 0: JUMPI taken (condition = 42). Row 1: JUMPI not taken (condition = 0).
        let events = vec![
            jumpi_evt(2, 5, 42),
            jumpi_evt(7, 9, 0),
        ];
        let w = from_jump_events(&events);
        assert!(w.rows[0].is_taken);
        assert!(!w.rows[1].is_taken);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = JumpValidityConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn tampered_jump_not_taken_breaks_constraint_6() {
        // Honest event: JUMP with is_taken=true. Tamper so is_taken=0.
        let events = vec![jump_evt(2, 5)];
        let w = from_jump_events(&events);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        let zero = Scalar::zero(CurveType::Bls48581);
        // Tamper.
        t.columns[COL_IS_TAKEN].evaluations[0] = zero;
        let cs = JumpValidityConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 6 (`sel_jump * (1 - is_taken)`) is now non-zero on row 0.
        assert!(!bodies[6][0].is_zero());
    }

    #[test]
    fn tampered_dest_byte_breaks_decomp() {
        // Honest event: dest = 0x1234. Tamper byte 0 to 0xFF (so the
        // decomp identity breaks).
        let events = vec![jump_evt(0, 0x1234)];
        let w = from_jump_events(&events);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        t.columns[COL_DEST_BYTE_0].evaluations[0] =
            Scalar::from_u64(0xFF, CurveType::Bls48581);
        let cs = JumpValidityConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 9 (byte decomp) fires.
        assert!(!bodies[9][0].is_zero());
    }

    #[test]
    fn jumpi_lying_condition_inv_breaks_constraint_8() {
        // Malicious row: sel_jumpi=1, condition=5, condition_inv=0,
        // is_taken=0. Constraint 7 is satisfied (`is_taken=0=5*0`), but
        // constraint 8 (`sel_jumpi * cond * (1 - is_taken)`) is not.
        let events = vec![JumpEvent {
            pc: 0,
            dest: 5,
            condition: 5,
            sel_jump: false,
            sel_jumpi: true,
            is_taken: false, // malicious
        }];
        let w = JumpValidityWitness::from_events(events);
        // Build the trace manually with `condition_inv = 0`.
        let curve = CurveType::Bls48581;
        let mut t = build_trace_polynomials(&w, curve);
        // The builder normally writes `condition_inv = condition^-1`
        // since condition != 0 — overwrite it to 0 to simulate cheating.
        t.columns[COL_CONDITION_INV].evaluations[0] = Scalar::zero(curve);
        let cs = JumpValidityConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 7: sel_ji * (is_taken - cond * cond_inv) = 1 * (0 - 0) = 0 — OK.
        assert!(bodies[7][0].is_zero());
        // Constraint 8: sel_ji * cond * (1 - is_taken) = 1 * 5 * 1 = 5 — fires.
        assert!(!bodies[8][0].is_zero());
    }

    #[test]
    fn descriptor_well_formed() {
        let d = make_jump_validity_to_jumpdest_table_descriptor(0, 1);
        assert_eq!(d.label, "jump_validity_to_jumpdest_table_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), 2);
        assert_eq!(d.a_columns[0], COL_DEST);
        assert_eq!(
            d.a_columns[1],
            crate::jumpdest_table_air::COL_IS_JUMPDEST_CONST_ONE_PLACEHOLDER,
        );
        assert_eq!(d.a_selector_column, Some(COL_IS_TAKEN));
        assert_eq!(
            d.b_columns,
            vec![
                crate::jumpdest_table_air::COL_PC,
                crate::jumpdest_table_air::COL_IS_JUMPDEST,
            ],
        );
        assert_eq!(
            d.b_selector_column,
            Some(crate::jumpdest_table_air::COL_IS_JUMPDEST),
        );
    }

    #[test]
    fn evaluate_at_point_zero_on_honest_witness() {
        let events = vec![
            jump_evt(0, 4),
            jumpi_evt(7, 12, 99),
            jumpi_evt(14, 0, 0),
        ];
        let w = from_jump_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = JumpValidityConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(7, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..t.num_rows {
            let row_evals: Vec<Scalar> = cr.iter().map(|c| c[r].clone()).collect();
            assert!(
                cs.evaluate_at_point(&row_evals, &alpha).is_zero(),
                "row {} evaluate_at_point non-zero",
                r,
            );
        }
    }

    #[test]
    fn empty_events_yields_empty_witness() {
        let w = from_jump_events(&[]);
        assert_eq!(w.rows.len(), 0);
    }

    #[test]
    fn lookup_declarations_include_all_dest_bytes() {
        let cs = JumpValidityConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        assert_eq!(req.declarations.len(), NUM_DEST_BYTES);
        for k in 0..NUM_DEST_BYTES {
            assert_eq!(req.declarations[k].0.column_index, COL_DEST_BYTE_0 + k);
            assert_eq!(req.declarations[k].0.max_bits, 8);
        }
    }
}
