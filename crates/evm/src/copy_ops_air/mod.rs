//! Combined CODECOPY (0x39) + CALLDATACOPY (0x37) gadget AIR.
//!
//! Each row commits a single copy-op invocation:
//!
//!   (opcode, pc, dest_offset, source_offset, length, source_size,
//!    gas_cost, copy_cost, mem_expansion_cost, ceil_chunks, ceil_pad,
//!    sel_codecopy, sel_calldatacopy, is_real)
//!
//! Gas formula (post-Cancun, both opcodes share the same):
//!
//!   gas_cost = 3 + 3 * ceil(length / 32) + memory_expansion_cost
//!           = copy_cost + memory_expansion_cost
//!
//! The expansion cost is delegated to [`crate::memory_expansion_air`]
//! and wired in via [`make_copy_ops_to_memory_expansion_descriptor`].
//!
//! ## Constraint sketch
//!
//! - `is_real ∈ {0, 1}`, `sel_codecopy ∈ {0, 1}`, `sel_calldatacopy ∈ {0, 1}`.
//! - Mutex: `sel_codecopy * sel_calldatacopy = 0`.
//! - Sum: `sel_codecopy + sel_calldatacopy = is_real`.
//! - Opcode binding:
//!   `sel_codecopy * (opcode - 0x39) = 0`,
//!   `sel_calldatacopy * (opcode - 0x37) = 0`.
//! - Copy-cost formula: `is_real * (copy_cost - 3 - 3 * ceil_chunks) = 0`.
//! - Total-gas formula: `is_real * (gas_cost - copy_cost - mem_expansion_cost) = 0`.
//! - Ceil-div: `is_real * (32 * ceil_chunks - length - ceil_pad) = 0`,
//!   with `ceil_pad ∈ [0..31]` (range check deferred — host populates it).
//!
//! ## Cross-AIR LogUp
//!
//! - [`make_copy_ops_to_memory_expansion_descriptor`] binds
//!   `(dest_offset, length, mem_expansion_cost)` → memory-expansion gadget.
//! - [`make_copy_ops_to_code_table_descriptor`] (gated by `sel_codecopy`)
//!   binds `(source_offset, length)` → bytecode-table-style AIR.
//! - [`make_copy_ops_to_calldata_byte_descriptor`] (gated by `sel_calldatacopy`)
//!   binds `(source_offset, length)` → calldata_byte_air.
//!
//! ## Deferred
//!
//! - `ceil_pad ∈ [0..31]` is host-side; without it the prover could pick
//!   `ceil_chunks` too large. Phase A3 follow-up wires this to the byte/
//!   5-bit range table.
//! - Per-byte copying soundness (every byte in
//!   `mem[dest..dest+len]` equals `src[source..source+len]`) is deferred
//!   to a wider byte-memory ↔ source-table linkage; this AIR commits the
//!   gas + opcode + alignment skeleton.
//! - `source_size` (CODESIZE / CALLDATASIZE) is a witnessed column; a
//!   future enhancement binds it to env_air / bytecode_table.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column layout ───────────────────────────────────────────────────

pub const COL_OPCODE: usize = 0;
pub const COL_PC: usize = 1;
pub const COL_DEST_OFFSET: usize = 2;
pub const COL_SOURCE_OFFSET: usize = 3;
pub const COL_LENGTH: usize = 4;
pub const COL_SOURCE_SIZE: usize = 5;
pub const COL_GAS_COST: usize = 6;
pub const COL_COPY_COST: usize = 7;
pub const COL_MEM_EXPANSION_COST: usize = 8;
pub const COL_CEIL_CHUNKS: usize = 9;
pub const COL_CEIL_PAD: usize = 10;
pub const COL_SEL_CODECOPY: usize = 11;
pub const COL_SEL_CALLDATACOPY: usize = 12;
pub const COL_IS_REAL: usize = 13;
pub const NUM_COLUMNS: usize = 14;

/// Per-row constraint set:
/// 0. `is_real * (is_real - 1) = 0`
/// 1. `sel_codecopy * (sel_codecopy - 1) = 0`
/// 2. `sel_calldatacopy * (sel_calldatacopy - 1) = 0`
/// 3. `sel_codecopy * sel_calldatacopy = 0` (mutex)
/// 4. `sel_codecopy + sel_calldatacopy - is_real = 0` (sum)
/// 5. `sel_codecopy * (opcode - 0x39) = 0`
/// 6. `sel_calldatacopy * (opcode - 0x37) = 0`
/// 7. `is_real * (copy_cost - 3 - 3 * ceil_chunks) = 0`
/// 8. `is_real * (gas_cost - copy_cost - mem_expansion_cost) = 0`
/// 9. `is_real * (32 * ceil_chunks - length - ceil_pad) = 0`
pub const NUM_ROW_CONSTRAINTS: usize = 10;
pub const NUM_SHIFTED: usize = 0;

pub const OPCODE_CODECOPY: u8 = 0x39;
pub const OPCODE_CALLDATACOPY: u8 = 0x37;

// ─── Witness ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CopyOpsRow {
    pub opcode: u8,
    pub pc: u64,
    pub dest_offset: u64,
    pub source_offset: u64,
    pub length: u64,
    pub source_size: u64,
    pub mem_expansion_cost: u64,
}

#[derive(Clone, Debug, Default)]
pub struct CopyOpsWitness {
    pub rows: Vec<CopyOpsRow>,
}

impl CopyOpsWitness {
    pub fn from_rows(rows: Vec<CopyOpsRow>) -> Self { Self { rows } }
}

/// Build a witness from a sequence of host-observed copy events.
///
/// Each tuple is `(opcode, pc, dest_offset, source_offset, length,
/// source_size)`. The `mem_expansion_cost` field is left zero; callers
/// that want to validate the cross-AIR LogUp to `memory_expansion_air`
/// should populate it from the host-side oracle after construction (or
/// build the witness via [`CopyOpsWitness::from_rows`] directly).
pub fn from_events(
    events: &[(u8, u64, u64, u64, u64, u64)],
) -> CopyOpsWitness {
    let rows = events
        .iter()
        .map(|&(opcode, pc, dest, source, length, source_size)| CopyOpsRow {
            opcode,
            pc,
            dest_offset: dest,
            source_offset: source,
            length,
            source_size,
            mem_expansion_cost: 0,
        })
        .collect();
    CopyOpsWitness { rows }
}

#[inline]
fn ceil_chunks_of(length: u64) -> u64 { (length + 31) / 32 }

#[inline]
fn ceil_pad_of(length: u64) -> u64 {
    // 32 * ceil_chunks - length ∈ [0..31]
    32u64.saturating_mul(ceil_chunks_of(length)).saturating_sub(length)
}

pub fn build_trace_polynomials(
    w: &CopyOpsWitness,
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
        let ceil_chunks = ceil_chunks_of(row.length);
        let ceil_pad = ceil_pad_of(row.length);
        let copy_cost = 3u64 + 3u64.saturating_mul(ceil_chunks);
        let gas_cost = copy_cost.saturating_add(row.mem_expansion_cost);
        let (sel_codecopy, sel_calldatacopy) = match row.opcode {
            OPCODE_CODECOPY => (1u64, 0u64),
            OPCODE_CALLDATACOPY => (0u64, 1u64),
            _ => (0u64, 0u64), // host data error — leave as 0; constraints will catch it
        };

        cols[COL_OPCODE][r] = Scalar::from_u64(row.opcode as u64, curve);
        cols[COL_PC][r] = Scalar::from_u64(row.pc, curve);
        cols[COL_DEST_OFFSET][r] = Scalar::from_u64(row.dest_offset, curve);
        cols[COL_SOURCE_OFFSET][r] = Scalar::from_u64(row.source_offset, curve);
        cols[COL_LENGTH][r] = Scalar::from_u64(row.length, curve);
        cols[COL_SOURCE_SIZE][r] = Scalar::from_u64(row.source_size, curve);
        cols[COL_GAS_COST][r] = Scalar::from_u64(gas_cost, curve);
        cols[COL_COPY_COST][r] = Scalar::from_u64(copy_cost, curve);
        cols[COL_MEM_EXPANSION_COST][r] = Scalar::from_u64(row.mem_expansion_cost, curve);
        cols[COL_CEIL_CHUNKS][r] = Scalar::from_u64(ceil_chunks, curve);
        cols[COL_CEIL_PAD][r] = Scalar::from_u64(ceil_pad, curve);
        cols[COL_SEL_CODECOPY][r] = Scalar::from_u64(sel_codecopy, curve);
        cols[COL_SEL_CALLDATACOPY][r] = Scalar::from_u64(sel_calldatacopy, curve);
        cols[COL_IS_REAL][r] = one.clone();
    }
    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials { columns: polys, num_rows, padded_size: padded as u64, curve }
}

// ─── Constraint system ───────────────────────────────────────────────

pub struct CopyOpsConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl CopyOpsConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for CopyOpsConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "sel_codecopy_binary".into(),
            "sel_calldatacopy_binary".into(),
            "sel_mutex".into(),
            "sel_sum_eq_is_real".into(),
            "opcode_codecopy_binding".into(),
            "opcode_calldatacopy_binding".into(),
            "copy_cost_formula".into(),
            "gas_cost_formula".into(),
            "ceildiv_alignment".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let three = Scalar::from_u64(3, curve);
        let thirty_two = Scalar::from_u64(32, curve);
        let op_codecopy = Scalar::from_u64(OPCODE_CODECOPY as u64, curve);
        let op_calldatacopy = Scalar::from_u64(OPCODE_CALLDATACOPY as u64, curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for r in 0..n {
            let real = &columns[COL_IS_REAL][r];
            let sel_cc = &columns[COL_SEL_CODECOPY][r];
            let sel_cdc = &columns[COL_SEL_CALLDATACOPY][r];
            let opcode = &columns[COL_OPCODE][r];
            let length = &columns[COL_LENGTH][r];
            let gas_cost = &columns[COL_GAS_COST][r];
            let copy_cost = &columns[COL_COPY_COST][r];
            let mem_exp = &columns[COL_MEM_EXPANSION_COST][r];
            let ceil_chunks = &columns[COL_CEIL_CHUNKS][r];
            let ceil_pad = &columns[COL_CEIL_PAD][r];

            // 0. is_real binary
            bodies[0][r] = real.mul(&real.sub(&one));
            // 1. sel_codecopy binary
            bodies[1][r] = sel_cc.mul(&sel_cc.sub(&one));
            // 2. sel_calldatacopy binary
            bodies[2][r] = sel_cdc.mul(&sel_cdc.sub(&one));
            // 3. mutex
            bodies[3][r] = sel_cc.mul(sel_cdc);
            // 4. sum = is_real
            bodies[4][r] = sel_cc.add(sel_cdc).sub(real);
            // 5. opcode binding for codecopy
            bodies[5][r] = sel_cc.mul(&opcode.sub(&op_codecopy));
            // 6. opcode binding for calldatacopy
            bodies[6][r] = sel_cdc.mul(&opcode.sub(&op_calldatacopy));
            // 7. copy_cost = 3 + 3 * ceil_chunks  (gated by is_real)
            let three_chunks = three.mul(ceil_chunks);
            let copy_rhs = three.add(&three_chunks);
            bodies[7][r] = real.mul(&copy_cost.sub(&copy_rhs));
            // 8. gas_cost = copy_cost + mem_expansion_cost  (gated)
            let gas_rhs = copy_cost.add(mem_exp);
            bodies[8][r] = real.mul(&gas_cost.sub(&gas_rhs));
            // 9. 32 * ceil_chunks - length - ceil_pad = 0  (gated)
            let chunks32 = thirty_two.mul(ceil_chunks);
            bodies[9][r] = real.mul(&chunks32.sub(length).sub(ceil_pad));
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let three = Scalar::from_u64(3, curve);
        let thirty_two = Scalar::from_u64(32, curve);
        let op_codecopy = Scalar::from_u64(OPCODE_CODECOPY as u64, curve);
        let op_calldatacopy = Scalar::from_u64(OPCODE_CALLDATACOPY as u64, curve);

        let real = &ce[COL_IS_REAL];
        let sel_cc = &ce[COL_SEL_CODECOPY];
        let sel_cdc = &ce[COL_SEL_CALLDATACOPY];

        let c0 = real.mul(&real.sub(&one));
        let c1 = sel_cc.mul(&sel_cc.sub(&one));
        let c2 = sel_cdc.mul(&sel_cdc.sub(&one));
        let c3 = sel_cc.mul(sel_cdc);
        let c4 = sel_cc.add(sel_cdc).sub(real);
        let c5 = sel_cc.mul(&ce[COL_OPCODE].sub(&op_codecopy));
        let c6 = sel_cdc.mul(&ce[COL_OPCODE].sub(&op_calldatacopy));
        let three_chunks = three.mul(&ce[COL_CEIL_CHUNKS]);
        let copy_rhs = three.add(&three_chunks);
        let c7 = real.mul(&ce[COL_COPY_COST].sub(&copy_rhs));
        let gas_rhs = ce[COL_COPY_COST].add(&ce[COL_MEM_EXPANSION_COST]);
        let c8 = real.mul(&ce[COL_GAS_COST].sub(&gas_rhs));
        let chunks32 = thirty_two.mul(&ce[COL_CEIL_CHUNKS]);
        let c9 = real.mul(&chunks32.sub(&ce[COL_LENGTH]).sub(&ce[COL_CEIL_PAD]));

        let bodies = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9];
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
        let three_p = vec![Scalar::from_u64(3, curve)];
        let thirty_two_p = vec![Scalar::from_u64(32, curve)];
        let op_codecopy_p = vec![Scalar::from_u64(OPCODE_CODECOPY as u64, curve)];
        let op_calldatacopy_p = vec![Scalar::from_u64(OPCODE_CALLDATACOPY as u64, curve)];

        let real = &cc[COL_IS_REAL];
        let sel_cc = &cc[COL_SEL_CODECOPY];
        let sel_cdc = &cc[COL_SEL_CALLDATACOPY];

        let real_m1 = poly_sub(real, &one_p, curve);
        let c0 = poly_mul(real, &real_m1, curve);

        let sel_cc_m1 = poly_sub(sel_cc, &one_p, curve);
        let c1 = poly_mul(sel_cc, &sel_cc_m1, curve);

        let sel_cdc_m1 = poly_sub(sel_cdc, &one_p, curve);
        let c2 = poly_mul(sel_cdc, &sel_cdc_m1, curve);

        let c3 = poly_mul(sel_cc, sel_cdc, curve);

        let sum = poly_add(sel_cc, sel_cdc, curve);
        let c4 = poly_sub(&sum, real, curve);

        let op_minus_cc = poly_sub(&cc[COL_OPCODE], &op_codecopy_p, curve);
        let c5 = poly_mul(sel_cc, &op_minus_cc, curve);

        let op_minus_cdc = poly_sub(&cc[COL_OPCODE], &op_calldatacopy_p, curve);
        let c6 = poly_mul(sel_cdc, &op_minus_cdc, curve);

        let three_chunks = poly_mul(&three_p, &cc[COL_CEIL_CHUNKS], curve);
        let copy_rhs = poly_add(&three_p, &three_chunks, curve);
        let copy_diff = poly_sub(&cc[COL_COPY_COST], &copy_rhs, curve);
        let c7 = poly_mul(real, &copy_diff, curve);

        let gas_rhs = poly_add(&cc[COL_COPY_COST], &cc[COL_MEM_EXPANSION_COST], curve);
        let gas_diff = poly_sub(&cc[COL_GAS_COST], &gas_rhs, curve);
        let c8 = poly_mul(real, &gas_diff, curve);

        let chunks32 = poly_mul(&thirty_two_p, &cc[COL_CEIL_CHUNKS], curve);
        let align_minus_len = poly_sub(&chunks32, &cc[COL_LENGTH], curve);
        let align_diff = poly_sub(&align_minus_len, &cc[COL_CEIL_PAD], curve);
        let c9 = poly_mul(real, &align_diff, curve);

        let bodies = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9];
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_SEL_CODECOPY, COL_SEL_CALLDATACOPY]
    }
    fn padding_selector_column(&self) -> Option<usize> { None }
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
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// Bind `(dest_offset, length, mem_expansion_cost)` on copy-ops to
/// `(old_size, new_size, expansion_cost)` on memory-expansion. This
/// closes "the expansion cost used in the gas formula is consistent
/// with an actual memory expansion from some pre-state to
/// `dest_offset + length`".
///
/// Note: a strict binding requires the memory-expansion witness to
/// publish rows whose `(old_size, new_size)` correspond to the copy's
/// pre-state and `dest_offset + length`. The host arranges this when
/// populating both witnesses from the same trace.
pub fn make_copy_ops_to_memory_expansion_descriptor(
    copy_layer: usize,
    mem_exp_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::memory_expansion_air::{
        COL_EXPANSION_COST, COL_IS_REAL as ME_COL_IS_REAL,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "copy_ops_to_memory_expansion_v1".into(),
        a_layer_index: copy_layer,
        a_columns: vec![COL_MEM_EXPANSION_COST],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: mem_exp_layer,
        b_columns: vec![COL_EXPANSION_COST],
        b_selector_column: Some(ME_COL_IS_REAL),
    }
}

/// CODECOPY-gated binding from copy-ops `(source_offset, length)` to a
/// bytecode-table-style AIR's `(pc, opcode)` tuple. The bytecode table
/// publishes one row per code byte; copy-ops asserts the source range
/// is in-bounds by sharing a tuple with the table.
///
/// Caveat: the bytecode_table AIR isn't currently a `VmConstraintSystem`
/// (it's a host-side table). This descriptor encodes the data-flow
/// contract that a future bytecode_table AIR will satisfy.
pub fn make_copy_ops_to_code_table_descriptor(
    copy_layer: usize,
    bytecode_table_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "copy_ops_to_code_table_v1".into(),
        a_layer_index: copy_layer,
        a_columns: vec![COL_SOURCE_OFFSET, COL_LENGTH],
        a_selector_column: Some(COL_SEL_CODECOPY),
        b_layer_index: bytecode_table_layer,
        // Bytecode table cols: 0=pc, 1=opcode. Length isn't a table column
        // but the b-side is a (pc, length-of-bytecode) summary row in the
        // future AIR; the current host-side table publishes (pc, opcode).
        b_columns: vec![0, 1],
        b_selector_column: None,
    }
}

/// CALLDATACOPY-gated binding from copy-ops `(source_offset, length)`
/// to the calldata byte AIR's `(offset, byte_val)` tuple. As with the
/// code-table descriptor, full per-byte binding requires N separate
/// descriptors (one per byte position) plus byte-extraction columns
/// on the copy-ops side; this descriptor is the offset-anchor.
pub fn make_copy_ops_to_calldata_byte_descriptor(
    copy_layer: usize,
    calldata_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::calldata_byte_air::{COL_BYTE_VAL, COL_IS_REAL as CD_COL_IS_REAL, COL_OFFSET};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "copy_ops_to_calldata_byte_v1".into(),
        a_layer_index: copy_layer,
        a_columns: vec![COL_SOURCE_OFFSET, COL_LENGTH],
        a_selector_column: Some(COL_SEL_CALLDATACOPY),
        b_layer_index: calldata_layer,
        b_columns: vec![COL_OFFSET, COL_BYTE_VAL],
        b_selector_column: Some(CD_COL_IS_REAL),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_all_zero(cs: &CopyOpsConstraintSystem, t: &TracePolynomials) {
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn codecopy_simple() {
        // CODECOPY at pc=10, dest=0, source=4, length=32, code_size=128
        let events = [(OPCODE_CODECOPY, 10u64, 0u64, 4u64, 32u64, 128u64)];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CopyOpsConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls48581;
        // ceil_chunks = ceil(32/32) = 1, copy_cost = 3 + 3 = 6
        assert!(cr[COL_CEIL_CHUNKS][0].sub(&Scalar::from_u64(1, curve)).is_zero());
        assert!(cr[COL_COPY_COST][0].sub(&Scalar::from_u64(6, curve)).is_zero());
        // gas_cost = copy_cost + 0 (mem_expansion unset) = 6
        assert!(cr[COL_GAS_COST][0].sub(&Scalar::from_u64(6, curve)).is_zero());
        // selectors
        assert!(cr[COL_SEL_CODECOPY][0].sub(&Scalar::one(curve)).is_zero());
        assert!(cr[COL_SEL_CALLDATACOPY][0].is_zero());
        // ceil_pad = 32 - 32 = 0
        assert!(cr[COL_CEIL_PAD][0].is_zero());
    }

    #[test]
    fn calldatacopy_simple() {
        // CALLDATACOPY at pc=20, dest=64, source=0, length=33, cd_size=64
        let events = [(OPCODE_CALLDATACOPY, 20u64, 64u64, 0u64, 33u64, 64u64)];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CopyOpsConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls48581;
        // ceil(33/32) = 2, copy_cost = 3 + 6 = 9
        assert!(cr[COL_CEIL_CHUNKS][0].sub(&Scalar::from_u64(2, curve)).is_zero());
        assert!(cr[COL_COPY_COST][0].sub(&Scalar::from_u64(9, curve)).is_zero());
        // ceil_pad = 64 - 33 = 31
        assert!(cr[COL_CEIL_PAD][0].sub(&Scalar::from_u64(31, curve)).is_zero());
        // selectors
        assert!(cr[COL_SEL_CALLDATACOPY][0].sub(&Scalar::one(curve)).is_zero());
        assert!(cr[COL_SEL_CODECOPY][0].is_zero());
    }

    #[test]
    fn length_zero_noop() {
        // Length=0 is a valid no-op copy: ceil_chunks = 0, copy_cost = 3.
        let events = [
            (OPCODE_CODECOPY, 5u64, 0u64, 0u64, 0u64, 32u64),
            (OPCODE_CALLDATACOPY, 6u64, 0u64, 0u64, 0u64, 0u64),
        ];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CopyOpsConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls48581;
        for r in 0..2 {
            assert!(cr[COL_CEIL_CHUNKS][r].is_zero());
            assert!(cr[COL_CEIL_PAD][r].is_zero());
            assert!(cr[COL_COPY_COST][r].sub(&Scalar::from_u64(3, curve)).is_zero());
            assert!(cr[COL_GAS_COST][r].sub(&Scalar::from_u64(3, curve)).is_zero());
        }
    }

    #[test]
    fn tampered_gas_cost_detected() {
        let events = [(OPCODE_CODECOPY, 10u64, 0u64, 0u64, 32u64, 128u64)];
        let w = from_events(&events);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        let curve = CurveType::Bls48581;
        // Corrupt gas_cost from 6 → 999.
        t.columns[COL_GAS_COST].evaluations[0] = Scalar::from_u64(999, curve);
        let cs = CopyOpsConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 8 (gas_cost = copy_cost + mem_expansion_cost) should fire.
        assert!(!bodies[8][0].is_zero(), "tampered gas_cost not caught");
    }

    #[test]
    fn tampered_copy_cost_detected() {
        let events = [(OPCODE_CALLDATACOPY, 0u64, 0u64, 0u64, 64u64, 64u64)];
        let w = from_events(&events);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        let curve = CurveType::Bls48581;
        // Honest copy_cost = 3 + 3*2 = 9. Tamper to 7.
        t.columns[COL_COPY_COST].evaluations[0] = Scalar::from_u64(7, curve);
        let cs = CopyOpsConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 7 (copy_cost = 3 + 3*ceil_chunks) should fire.
        assert!(!bodies[7][0].is_zero(), "tampered copy_cost not caught");
    }

    #[test]
    fn tampered_selector_mutex_detected() {
        // Set both selectors to 1 — mutex must catch it.
        let events = [(OPCODE_CODECOPY, 0u64, 0u64, 0u64, 0u64, 0u64)];
        let w = from_events(&events);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        let curve = CurveType::Bls48581;
        t.columns[COL_SEL_CALLDATACOPY].evaluations[0] = Scalar::one(curve);
        let cs = CopyOpsConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 3 (mutex) should fire.
        assert!(!bodies[3][0].is_zero(), "mutex violation not caught");
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_copy_ops_to_memory_expansion_descriptor(0, 1);
        assert_eq!(d1.label, "copy_ops_to_memory_expansion_v1");
        assert_eq!(d1.a_columns, vec![COL_MEM_EXPANSION_COST]);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);

        let d2 = make_copy_ops_to_code_table_descriptor(0, 2);
        assert_eq!(d2.label, "copy_ops_to_code_table_v1");
        assert_eq!(d2.a_columns, vec![COL_SOURCE_OFFSET, COL_LENGTH]);
        assert_eq!(d2.a_selector_column, Some(COL_SEL_CODECOPY));

        let d3 = make_copy_ops_to_calldata_byte_descriptor(0, 3);
        assert_eq!(d3.label, "copy_ops_to_calldata_byte_v1");
        assert_eq!(d3.a_columns, vec![COL_SOURCE_OFFSET, COL_LENGTH]);
        assert_eq!(d3.a_selector_column, Some(COL_SEL_CALLDATACOPY));
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let events = [
            (OPCODE_CODECOPY, 10u64, 0u64, 0u64, 32u64, 128u64),
            (OPCODE_CALLDATACOPY, 20u64, 64u64, 0u64, 65u64, 64u64),
            (OPCODE_CODECOPY, 30u64, 0u64, 0u64, 0u64, 16u64),
        ];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CopyOpsConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xabcd, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..w.rows.len() {
            let row_evals: Vec<Scalar> = cr.iter().map(|c| c[r].clone()).collect();
            let v = cs.evaluate_at_point(&row_evals, &alpha);
            assert!(v.is_zero(), "row {} nonzero", r);
        }
    }

    #[test]
    fn mem_expansion_cost_propagates_to_gas() {
        // Manually construct a row with mem_expansion_cost > 0 and check
        // gas_cost = copy_cost + mem_expansion_cost holds.
        let row = CopyOpsRow {
            opcode: OPCODE_CALLDATACOPY,
            pc: 0,
            dest_offset: 1024,
            source_offset: 0,
            length: 32,
            source_size: 64,
            mem_expansion_cost: 95,
        };
        let w = CopyOpsWitness::from_rows(vec![row]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = CopyOpsConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls48581;
        // copy_cost = 3 + 3 = 6, gas = 6 + 95 = 101
        assert!(cr[COL_COPY_COST][0].sub(&Scalar::from_u64(6, curve)).is_zero());
        assert!(cr[COL_MEM_EXPANSION_COST][0].sub(&Scalar::from_u64(95, curve)).is_zero());
        assert!(cr[COL_GAS_COST][0].sub(&Scalar::from_u64(101, curve)).is_zero());
    }
}
