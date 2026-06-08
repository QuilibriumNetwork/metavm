//! EVM opcode dispatch AIR.
//!
//! Proves, per EVM step:
//!
//!   - The tuple `(opcode, push_count, pop_count, base_gas, is_invalid)`
//!     committed on each row matches the canonical EVM opcode-dispatch
//!     table built inline as [`OPCODE_TABLE`].
//!   - `is_real` and `is_invalid` are binary.
//!   - `opcode` is in `0..256` (byte range).
//!
//! The full per-step tuple ↔ table binding is delegated to a cross-AIR
//! LogUp against a 256-row constant table AIR; see
//! [`make_opcode_dispatch_table_descriptor`].
//!
//! ## Per-row layout (witness)
//!
//! ```text
//! offset  meaning
//!  0      opcode       (u8)
//!  1      push_count   (u8)
//!  2      pop_count    (u8)
//!  3      base_gas     (u16; per-opcode static base cost, 0 if dynamic)
//!  4      is_invalid   (binary; 1 iff opcode is unallocated)
//!  5      is_real      (binary; 1 on real steps, 0 on padding)
//! ```
//!
//! Total: **6 data columns**.
//!
//! ## Row-local constraints (4)
//!
//!  1. `is_real_binary`   — `is_real · (is_real - 1) = 0`
//!  2. `is_invalid_binary` — `is_invalid · (is_invalid - 1) = 0`
//!  3. `opcode_byte_range_placeholder` — algebraic zero, range delegated
//!     to a LogUp into the 256-row byte table.
//!  4. `is_invalid_implies_zero_counts` —
//!     `is_invalid · (push_count + pop_count + base_gas) = 0`. Invalid
//!     opcodes have zero push/pop/gas in the canonical table; this
//!     binds the per-row witness directly.
//!
//! Range checking the count and gas columns is delegated to LogUp
//! declarations against the 256-row byte table for the 8-bit columns;
//! `base_gas` is bound by the cross-AIR LogUp to the 256-row constant
//! table.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_OPCODE: usize = 0;
pub const COL_PUSH_COUNT: usize = 1;
pub const COL_POP_COUNT: usize = 2;
pub const COL_BASE_GAS: usize = 3;
pub const COL_IS_INVALID: usize = 4;
pub const COL_IS_REAL: usize = 5;

pub const NUM_COLUMNS: usize = 6;
pub const NUM_ROW_CONSTRAINTS: usize = 4;
pub const NUM_SHIFTED: usize = 0;

// ─── Canonical opcode dispatch table ──────────────────────────────────
//
// Tuple: `(opcode, push_count, pop_count, base_gas, is_invalid)`.
//
// `base_gas` is the static gas cost where the opcode is constant-cost,
// or `0` for dynamic-cost opcodes (SHA3, SLOAD/SSTORE, CALL family,
// LOG*, EXP, copy ops, memory ops, etc.). The base-gas column is bound
// to this same `0` placeholder for those entries; downstream gadgets
// (gas_tracking_air, memory_expansion_air, …) carry the dynamic
// component.
//
// `is_invalid = true` iff the opcode byte is unallocated in the
// post-Cancun EVM spec. Invalid opcodes have `(0, 0, 0)` for
// `(push, pop, base_gas)`.

/// `OPCODE_TABLE[op] = (opcode, push_count, pop_count, base_gas, is_invalid)`.
///
/// Indexed by `opcode` (the first field is redundant but kept so the
/// table can be iterated directly into the constant-table AIR's row
/// layout without re-indexing).
pub const OPCODE_TABLE: [(u8, u8, u8, u16, bool); 256] = build_opcode_table();

const fn build_opcode_table() -> [(u8, u8, u8, u16, bool); 256] {
    // Default: invalid opcode with all-zero counts/gas.
    let mut t: [(u8, u8, u8, u16, bool); 256] = [(0u8, 0, 0, 0, true); 256];
    // Fill in opcode bytes (first field).
    let mut i = 0usize;
    while i < 256 {
        t[i].0 = i as u8;
        i += 1;
    }

    // ── 0x00..0x0B: arithmetic + STOP ───────────────────────────────
    t[0x00] = (0x00, 0, 0, 0, false); // STOP
    t[0x01] = (0x01, 1, 2, 3, false); // ADD
    t[0x02] = (0x02, 1, 2, 5, false); // MUL
    t[0x03] = (0x03, 1, 2, 3, false); // SUB
    t[0x04] = (0x04, 1, 2, 5, false); // DIV
    t[0x05] = (0x05, 1, 2, 5, false); // SDIV
    t[0x06] = (0x06, 1, 2, 5, false); // MOD
    t[0x07] = (0x07, 1, 2, 5, false); // SMOD
    t[0x08] = (0x08, 1, 3, 8, false); // ADDMOD
    t[0x09] = (0x09, 1, 3, 8, false); // MULMOD
    t[0x0A] = (0x0A, 1, 2, 0, false); // EXP (dynamic)
    t[0x0B] = (0x0B, 1, 2, 5, false); // SIGNEXTEND

    // ── 0x10..0x1D: comparison + bitwise ────────────────────────────
    t[0x10] = (0x10, 1, 2, 3, false); // LT
    t[0x11] = (0x11, 1, 2, 3, false); // GT
    t[0x12] = (0x12, 1, 2, 3, false); // SLT
    t[0x13] = (0x13, 1, 2, 3, false); // SGT
    t[0x14] = (0x14, 1, 2, 3, false); // EQ
    t[0x15] = (0x15, 1, 1, 3, false); // ISZERO
    t[0x16] = (0x16, 1, 2, 3, false); // AND
    t[0x17] = (0x17, 1, 2, 3, false); // OR
    t[0x18] = (0x18, 1, 2, 3, false); // XOR
    t[0x19] = (0x19, 1, 1, 3, false); // NOT
    t[0x1A] = (0x1A, 1, 2, 3, false); // BYTE
    t[0x1B] = (0x1B, 1, 2, 3, false); // SHL
    t[0x1C] = (0x1C, 1, 2, 3, false); // SHR
    t[0x1D] = (0x1D, 1, 2, 3, false); // SAR

    // ── 0x20: SHA3 ──────────────────────────────────────────────────
    t[0x20] = (0x20, 1, 2, 0, false); // SHA3 (base 30 + dynamic; treated as dynamic)

    // ── 0x30..0x4A: environment / block context ─────────────────────
    t[0x30] = (0x30, 1, 0, 2, false); // ADDRESS
    t[0x31] = (0x31, 1, 1, 0, false); // BALANCE (cold/warm, dynamic)
    t[0x32] = (0x32, 1, 0, 2, false); // ORIGIN
    t[0x33] = (0x33, 1, 0, 2, false); // CALLER
    t[0x34] = (0x34, 1, 0, 2, false); // CALLVALUE
    t[0x35] = (0x35, 1, 1, 3, false); // CALLDATALOAD
    t[0x36] = (0x36, 1, 0, 2, false); // CALLDATASIZE
    t[0x37] = (0x37, 0, 3, 0, false); // CALLDATACOPY (dynamic)
    t[0x38] = (0x38, 1, 0, 2, false); // CODESIZE
    t[0x39] = (0x39, 0, 3, 0, false); // CODECOPY (dynamic)
    t[0x3A] = (0x3A, 1, 0, 2, false); // GASPRICE
    t[0x3B] = (0x3B, 1, 1, 0, false); // EXTCODESIZE
    t[0x3C] = (0x3C, 0, 4, 0, false); // EXTCODECOPY (dynamic)
    t[0x3D] = (0x3D, 1, 0, 2, false); // RETURNDATASIZE
    t[0x3E] = (0x3E, 0, 3, 0, false); // RETURNDATACOPY (dynamic)
    t[0x3F] = (0x3F, 1, 1, 0, false); // EXTCODEHASH
    t[0x40] = (0x40, 1, 1, 0, false); // BLOCKHASH (20, treated dynamic-ish)
    t[0x41] = (0x41, 1, 0, 2, false); // COINBASE
    t[0x42] = (0x42, 1, 0, 2, false); // TIMESTAMP
    t[0x43] = (0x43, 1, 0, 2, false); // NUMBER
    t[0x44] = (0x44, 1, 0, 2, false); // PREVRANDAO
    t[0x45] = (0x45, 1, 0, 2, false); // GASLIMIT
    t[0x46] = (0x46, 1, 0, 2, false); // CHAINID
    t[0x47] = (0x47, 1, 0, 5, false); // SELFBALANCE
    t[0x48] = (0x48, 1, 0, 2, false); // BASEFEE
    t[0x49] = (0x49, 1, 0, 2, false); // BLOBHASH
    t[0x4A] = (0x4A, 1, 0, 2, false); // BLOBBASEFEE

    // ── 0x50..0x5F: stack/memory/control ────────────────────────────
    t[0x50] = (0x50, 0, 1, 2, false); // POP
    t[0x51] = (0x51, 1, 1, 0, false); // MLOAD  (3 + mem; dynamic)
    t[0x52] = (0x52, 0, 2, 0, false); // MSTORE (3 + mem; dynamic)
    t[0x53] = (0x53, 0, 2, 0, false); // MSTORE8 (3 + mem; dynamic)
    t[0x54] = (0x54, 1, 1, 0, false); // SLOAD  (dynamic, EIP-2929)
    t[0x55] = (0x55, 0, 2, 0, false); // SSTORE (dynamic)
    t[0x56] = (0x56, 0, 1, 8, false); // JUMP
    t[0x57] = (0x57, 0, 2, 10, false); // JUMPI
    t[0x58] = (0x58, 1, 0, 2, false); // PC
    t[0x59] = (0x59, 1, 0, 2, false); // MSIZE
    t[0x5A] = (0x5A, 1, 0, 2, false); // GAS
    t[0x5B] = (0x5B, 0, 0, 1, false); // JUMPDEST
    t[0x5C] = (0x5C, 1, 1, 0, false); // TLOAD  (EIP-1153; 100, treated dynamic)
    t[0x5D] = (0x5D, 0, 2, 0, false); // TSTORE
    t[0x5E] = (0x5E, 0, 3, 0, false); // MCOPY  (EIP-5656; dynamic)
    t[0x5F] = (0x5F, 1, 0, 2, false); // PUSH0

    // ── 0x60..0x7F: PUSH1..PUSH32 ───────────────────────────────────
    let mut op = 0x60usize;
    while op <= 0x7F {
        t[op] = (op as u8, 1, 0, 3, false);
        op += 1;
    }

    // ── 0x80..0x8F: DUP1..DUP16 (push=1, pop=0 net commitment) ──────
    //
    // NOTE: this AIR uses the convention `push_count=1, pop_count=0`
    // for DUPx (matches the user-facing roadmap spec). The
    // stack_depth_air uses a different convention (`push=n+1, pop=n`)
    // because it needs the absolute counts to reconstruct depth
    // deltas. Both are correct; this AIR commits dispatch metadata.
    let mut op = 0x80usize;
    while op <= 0x8F {
        t[op] = (op as u8, 1, 0, 3, false);
        op += 1;
    }

    // ── 0x90..0x9F: SWAP1..SWAP16 (push=0, pop=0 net) ───────────────
    let mut op = 0x90usize;
    while op <= 0x9F {
        t[op] = (op as u8, 0, 0, 3, false);
        op += 1;
    }

    // ── 0xA0..0xA4: LOG0..LOG4 (dynamic) ────────────────────────────
    t[0xA0] = (0xA0, 0, 2, 0, false);
    t[0xA1] = (0xA1, 0, 3, 0, false);
    t[0xA2] = (0xA2, 0, 4, 0, false);
    t[0xA3] = (0xA3, 0, 5, 0, false);
    t[0xA4] = (0xA4, 0, 6, 0, false);

    // ── 0xF0..0xFF: CREATE / CALL family / system ───────────────────
    t[0xF0] = (0xF0, 1, 3, 0, false); // CREATE
    t[0xF1] = (0xF1, 1, 7, 0, false); // CALL
    t[0xF2] = (0xF2, 1, 7, 0, false); // CALLCODE
    t[0xF3] = (0xF3, 0, 2, 0, false); // RETURN
    t[0xF4] = (0xF4, 1, 6, 0, false); // DELEGATECALL
    t[0xF5] = (0xF5, 1, 4, 0, false); // CREATE2
    t[0xFA] = (0xFA, 1, 6, 0, false); // STATICCALL
    t[0xFD] = (0xFD, 0, 2, 0, false); // REVERT
    t[0xFE] = (0xFE, 0, 0, 0, false); // INVALID (designated invalid opcode, but allocated)
    t[0xFF] = (0xFF, 0, 1, 0, false); // SELFDESTRUCT

    t
}

/// Look up `(push, pop, base_gas, is_invalid)` for an opcode byte.
pub const fn opcode_dispatch(opcode: u8) -> (u8, u8, u16, bool) {
    let e = OPCODE_TABLE[opcode as usize];
    (e.1, e.2, e.3, e.4)
}

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpcodeDispatchRow {
    pub opcode: u8,
    pub push_count: u8,
    pub pop_count: u8,
    pub base_gas: u16,
    pub is_invalid: bool,
}

#[derive(Clone, Debug, Default)]
pub struct OpcodeDispatchWitness {
    pub rows: Vec<OpcodeDispatchRow>,
}

impl OpcodeDispatchWitness {
    /// Build a witness from a stream of opcode bytes (e.g., as collected
    /// by the EVM inspector). Each opcode is dispatched through
    /// [`OPCODE_TABLE`] to populate the row.
    pub fn from_inspector_trace(opcodes: &[u8]) -> Self {
        let mut rows = Vec::with_capacity(opcodes.len());
        for &op in opcodes {
            let (push, pop, base_gas, is_invalid) = opcode_dispatch(op);
            rows.push(OpcodeDispatchRow {
                opcode: op,
                push_count: push,
                pop_count: pop,
                base_gas,
                is_invalid,
            });
        }
        Self { rows }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &OpcodeDispatchWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (i, r) in witness.rows.iter().enumerate() {
        columns[COL_OPCODE][i] = Scalar::from_u64(r.opcode as u64, curve);
        columns[COL_PUSH_COUNT][i] = Scalar::from_u64(r.push_count as u64, curve);
        columns[COL_POP_COUNT][i] = Scalar::from_u64(r.pop_count as u64, curve);
        columns[COL_BASE_GAS][i] = Scalar::from_u64(r.base_gas as u64, curve);
        if r.is_invalid {
            columns[COL_IS_INVALID][i] = one.clone();
        }
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

// ─── Constant 256-row dispatch table AIR trace builder ───────────────
//
// Lays out the full 256-row opcode table in the same column order as
// the per-step AIR (`opcode, push, pop, base_gas, is_invalid, is_real`)
// with `is_real = 1` on every row. Useful for wiring the cross-AIR
// LogUp B-side.

/// Builds the 256-row constant dispatch table as a [`TracePolynomials`]
/// in the same column layout as this AIR. Every row has `is_real = 1`.
pub fn build_table_trace_polynomials(curve: CurveType) -> TracePolynomials {
    let num_rows = OPCODE_TABLE.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows);
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (i, &(op, push, pop, gas, invalid)) in OPCODE_TABLE.iter().enumerate() {
        columns[COL_OPCODE][i] = Scalar::from_u64(op as u64, curve);
        columns[COL_PUSH_COUNT][i] = Scalar::from_u64(push as u64, curve);
        columns[COL_POP_COUNT][i] = Scalar::from_u64(pop as u64, curve);
        columns[COL_BASE_GAS][i] = Scalar::from_u64(gas as u64, curve);
        if invalid {
            columns[COL_IS_INVALID][i] = one.clone();
        }
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

// ─── Constraint system ────────────────────────────────────────────────

pub struct OpcodeDispatchConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl OpcodeDispatchConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for OpcodeDispatchConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_invalid_binary".into(),
            "opcode_byte_range_placeholder".into(),
            "is_invalid_implies_zero_counts".into(),
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
        let zero = Scalar::zero(curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![zero.clone(); n])
            .collect();
        for row in 0..n {
            let v = &columns[COL_IS_REAL][row];
            let inv = &columns[COL_IS_INVALID][row];
            let push = &columns[COL_PUSH_COUNT][row];
            let pop = &columns[COL_POP_COUNT][row];
            let gas = &columns[COL_BASE_GAS][row];

            // 1. is_real binary
            bodies[0][row] = v.mul(&v.sub(&one));
            // 2. is_invalid binary
            bodies[1][row] = inv.mul(&inv.sub(&one));
            // 3. opcode byte range — placeholder (LogUp).
            bodies[2][row] = zero.clone();
            // 4. is_invalid · (push + pop + base_gas) = 0
            let sum = push.add(pop).add(gas);
            bodies[3][row] = inv.mul(&sum);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);
        let v = &col_evals[COL_IS_REAL];
        let inv = &col_evals[COL_IS_INVALID];
        let push = &col_evals[COL_PUSH_COUNT];
        let pop = &col_evals[COL_POP_COUNT];
        let gas = &col_evals[COL_BASE_GAS];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            v.mul(&v.sub(&one)),
            inv.mul(&inv.sub(&one)),
            zero.clone(),
            inv.mul(&push.add(pop).add(gas)),
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
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let zero_p = vec![Scalar::zero(curve)];

        let v = &col_coeffs[COL_IS_REAL];
        let inv = &col_coeffs[COL_IS_INVALID];
        let push = &col_coeffs[COL_PUSH_COUNT];
        let pop = &col_coeffs[COL_POP_COUNT];
        let gas = &col_coeffs[COL_BASE_GAS];

        // 1. v · (v - 1)
        let v_m1 = poly_sub(v, &one_p, curve);
        let b1 = poly_mul(v, &v_m1, curve);
        // 2. inv · (inv - 1)
        let inv_m1 = poly_sub(inv, &one_p, curve);
        let b2 = poly_mul(inv, &inv_m1, curve);
        // 3. placeholder
        let b3 = zero_p.clone();
        // 4. inv · (push + pop + gas)
        let sum_pp = poly_add(push, pop, curve);
        let sum = poly_add(&sum_pp, gas, curve);
        let b4 = poly_mul(inv, &sum, curve);

        let bodies: Vec<Vec<Scalar>> = vec![b1, b2, b3, b4];

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

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
        let curve = columns[0][0].curve_type();
        let zero = Scalar::zero(curve);
        // Padding rows: zero everything. is_invalid=0 on padding means
        // constraint 4 vanishes trivially.
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let tbl = 0usize;
        let decls = vec![
            (LookupDeclaration {
                label: "opcode_dispatch_opcode_8bit".into(),
                column_index: COL_OPCODE,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "opcode_dispatch_push_8bit".into(),
                column_index: COL_PUSH_COUNT,
                max_bits: 8,
                selector_column: None,
            }, tbl),
            (LookupDeclaration {
                label: "opcode_dispatch_pop_8bit".into(),
                column_index: COL_POP_COUNT,
                max_bits: 8,
                selector_column: None,
            }, tbl),
        ];
        LookupRequirements { tables, declarations: decls }
    }
}

// ─── Cross-AIR LogUp linkage descriptors ──────────────────────────────

/// Binds the trace AIR's `(opcode, push_count, pop_count, base_gas)`
/// tuple on `is_real=1` rows to the 256-row constant dispatch table
/// AIR's matching columns. Both sides use the same column layout via
/// [`build_table_trace_polynomials`].
///
/// `dispatch_layer_index` — layer index of the per-step dispatch AIR
/// (A-side).
///
/// `table_layer_index` — layer index of the 256-row constant table AIR
/// (B-side). The B-side columns are the table's
/// `(opcode, push, pop, base_gas)` columns indexed identically.
pub fn make_opcode_dispatch_table_descriptor(
    dispatch_layer_index: usize,
    table_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "opcode_dispatch_to_table_v1".into(),
        a_layer_index: dispatch_layer_index,
        a_columns: vec![COL_OPCODE, COL_PUSH_COUNT, COL_POP_COUNT, COL_BASE_GAS],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: table_layer_index,
        b_columns: vec![COL_OPCODE, COL_PUSH_COUNT, COL_POP_COUNT, COL_BASE_GAS],
        b_selector_column: Some(COL_IS_REAL),
    }
}

/// Variant that also binds the `is_invalid` flag, ensuring the
/// per-step witness's invalid-marker matches the canonical table.
pub fn make_opcode_dispatch_table_descriptor_with_invalid(
    dispatch_layer_index: usize,
    table_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "opcode_dispatch_to_table_with_invalid_v1".into(),
        a_layer_index: dispatch_layer_index,
        a_columns: vec![
            COL_OPCODE,
            COL_PUSH_COUNT,
            COL_POP_COUNT,
            COL_BASE_GAS,
            COL_IS_INVALID,
        ],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: table_layer_index,
        b_columns: vec![
            COL_OPCODE,
            COL_PUSH_COUNT,
            COL_POP_COUNT,
            COL_BASE_GAS,
            COL_IS_INVALID,
        ],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcode_dispatch_air_table_spot_checks() {
        // Spec-driven spot checks for OPCODE_TABLE entries.

        // PUSH1
        let (p, q, g, inv) = opcode_dispatch(0x60);
        assert_eq!((p, q, g, inv), (1, 0, 3, false));
        // PUSH32
        let (p, q, g, inv) = opcode_dispatch(0x7F);
        assert_eq!((p, q, g, inv), (1, 0, 3, false));
        // DUP1
        let (p, q, g, inv) = opcode_dispatch(0x80);
        assert_eq!((p, q, g, inv), (1, 0, 3, false));
        // DUP16
        let (p, q, g, inv) = opcode_dispatch(0x8F);
        assert_eq!((p, q, g, inv), (1, 0, 3, false));
        // SWAP1
        let (p, q, g, inv) = opcode_dispatch(0x90);
        assert_eq!((p, q, g, inv), (0, 0, 3, false));
        // SWAP16
        let (p, q, g, inv) = opcode_dispatch(0x9F);
        assert_eq!((p, q, g, inv), (0, 0, 3, false));
        // POP
        let (p, q, g, inv) = opcode_dispatch(0x50);
        assert_eq!((p, q, g, inv), (0, 1, 2, false));
        // JUMPDEST
        let (p, q, g, inv) = opcode_dispatch(0x5B);
        assert_eq!((p, q, g, inv), (0, 0, 1, false));
        // ADD
        let (p, q, g, inv) = opcode_dispatch(0x01);
        assert_eq!((p, q, g, inv), (1, 2, 3, false));
        // MUL
        let (p, q, g, inv) = opcode_dispatch(0x02);
        assert_eq!((p, q, g, inv), (1, 2, 5, false));
        // SHA3 (dynamic-gas marker)
        let (p, q, g, inv) = opcode_dispatch(0x20);
        assert_eq!((p, q, g, inv), (1, 2, 0, false));
        // SLOAD (dynamic)
        let (p, q, g, inv) = opcode_dispatch(0x54);
        assert_eq!((p, q, g, inv), (1, 1, 0, false));
        // SSTORE (dynamic)
        let (p, q, g, inv) = opcode_dispatch(0x55);
        assert_eq!((p, q, g, inv), (0, 2, 0, false));
        // CALL
        let (p, q, g, inv) = opcode_dispatch(0xF1);
        assert_eq!((p, q, g, inv), (1, 7, 0, false));
    }

    #[test]
    fn opcode_dispatch_air_invalid_opcode() {
        // 0x0E is in the unallocated gap between SIGNEXTEND (0x0B) and
        // the comparison block (0x10..).
        let (push, pop, gas, invalid) = opcode_dispatch(0x0E);
        assert_eq!(push, 0);
        assert_eq!(pop, 0);
        assert_eq!(gas, 0);
        assert!(invalid, "0x0E must be marked invalid");

        // 0x21..0x2F are between SHA3 (0x20) and ADDRESS (0x30).
        for op in 0x21u8..=0x2F {
            let (_, _, _, invalid) = opcode_dispatch(op);
            assert!(invalid, "{:#x} must be marked invalid", op);
        }

        // 0xFE INVALID — designated invalid in the EVM spec but the byte
        // *is* allocated. Our table treats it as a valid (allocated)
        // dispatch row with zero counts.
        let (_, _, _, invalid) = opcode_dispatch(0xFE);
        assert!(!invalid, "0xFE INVALID is allocated, must not be table-invalid");
    }

    #[test]
    fn opcode_dispatch_air_push1_witness_and_constraints_vanish() {
        // Single PUSH1 step.
        let opcodes = vec![0x60u8];
        let w = OpcodeDispatchWitness::from_inspector_trace(&opcodes);
        assert_eq!(w.rows.len(), 1);
        assert_eq!(w.rows[0].opcode, 0x60);
        assert_eq!(w.rows[0].push_count, 1);
        assert_eq!(w.rows[0].pop_count, 0);
        assert_eq!(w.rows[0].base_gas, 3);
        assert!(!w.rows[0].is_invalid);

        let curve = CurveType::Bls48581;
        let t = build_trace_polynomials(&w, curve);
        let cs = OpcodeDispatchConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(7, curve);
        for row in 0..t.padded_size as usize {
            let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            let v = cs.evaluate_at_point(&cols, &alpha);
            assert!(v.is_zero(), "PUSH1 row {} body must vanish", row);
        }
    }

    #[test]
    fn opcode_dispatch_air_add_pop_witness_and_constraints_vanish() {
        // ADD; POP — two real rows.
        let opcodes = vec![0x01u8, 0x50u8];
        let w = OpcodeDispatchWitness::from_inspector_trace(&opcodes);
        assert_eq!(w.rows[0].push_count, 1);
        assert_eq!(w.rows[0].pop_count, 2);
        assert_eq!(w.rows[0].base_gas, 3);
        assert_eq!(w.rows[1].push_count, 0);
        assert_eq!(w.rows[1].pop_count, 1);
        assert_eq!(w.rows[1].base_gas, 2);

        let curve = CurveType::Bls48581;
        let t = build_trace_polynomials(&w, curve);
        let cs = OpcodeDispatchConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(11, curve);
        for row in 0..t.padded_size as usize {
            let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            let v = cs.evaluate_at_point(&cols, &alpha);
            assert!(v.is_zero(), "ADD/POP row {} body must vanish", row);
        }
    }

    #[test]
    fn opcode_dispatch_air_invalid_opcode_witness_and_constraints_vanish() {
        // 0x0E is unallocated — should round-trip through the witness
        // with is_invalid=1 and zero counts/gas, and the constraint #4
        // (`is_invalid · (push + pop + gas) = 0`) must vanish.
        let opcodes = vec![0x0Eu8];
        let w = OpcodeDispatchWitness::from_inspector_trace(&opcodes);
        assert!(w.rows[0].is_invalid);
        assert_eq!(w.rows[0].push_count, 0);
        assert_eq!(w.rows[0].pop_count, 0);
        assert_eq!(w.rows[0].base_gas, 0);

        let curve = CurveType::Bls48581;
        let t = build_trace_polynomials(&w, curve);
        let cs = OpcodeDispatchConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(3, curve);
        for row in 0..t.padded_size as usize {
            let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            let v = cs.evaluate_at_point(&cols, &alpha);
            assert!(v.is_zero(), "invalid-opcode row {} body must vanish", row);
        }
    }

    #[test]
    fn opcode_dispatch_air_tampered_invalid_row_detected() {
        // Honest invalid opcode 0x0E: counts/gas all zero. Tamper
        // push_count to 1 while keeping is_invalid=1 — constraint #4
        // (`is_invalid · (push + pop + gas) = 0`) must catch it.
        let opcodes = vec![0x0Eu8];
        let w = OpcodeDispatchWitness::from_inspector_trace(&opcodes);
        let curve = CurveType::Bls48581;
        let mut t = build_trace_polynomials(&w, curve);
        let cs = OpcodeDispatchConstraintSystem::new(t.num_rows);
        // Tamper the witness.
        t.columns[COL_PUSH_COUNT].evaluations[0] = Scalar::from_u64(1, curve);
        let alpha = Scalar::from_u64(13, curve);
        let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[0].clone()).collect();
        let v = cs.evaluate_at_point(&cols, &alpha);
        assert!(!v.is_zero(), "tampered invalid-opcode row must violate constraint 4");
    }

    #[test]
    fn opcode_dispatch_air_tampered_is_real_detected() {
        // Real PUSH1 row, tamper is_real to 2 — constraint #1 (binary)
        // must catch it.
        let opcodes = vec![0x60u8];
        let w = OpcodeDispatchWitness::from_inspector_trace(&opcodes);
        let curve = CurveType::Bls48581;
        let mut t = build_trace_polynomials(&w, curve);
        let cs = OpcodeDispatchConstraintSystem::new(t.num_rows);
        t.columns[COL_IS_REAL].evaluations[0] = Scalar::from_u64(2, curve);
        let alpha = Scalar::from_u64(5, curve);
        let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[0].clone()).collect();
        let v = cs.evaluate_at_point(&cols, &alpha);
        assert!(!v.is_zero(), "tampered is_real must violate binary constraint");
    }

    #[test]
    fn opcode_dispatch_air_descriptor_well_formed() {
        let d = make_opcode_dispatch_table_descriptor(0, 1);
        assert_eq!(d.label, "opcode_dispatch_to_table_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(
            d.a_columns,
            vec![COL_OPCODE, COL_PUSH_COUNT, COL_POP_COUNT, COL_BASE_GAS]
        );
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(
            d.b_columns,
            vec![COL_OPCODE, COL_PUSH_COUNT, COL_POP_COUNT, COL_BASE_GAS]
        );
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));

        let d = make_opcode_dispatch_table_descriptor_with_invalid(2, 3);
        assert_eq!(d.label, "opcode_dispatch_to_table_with_invalid_v1");
        assert_eq!(d.a_layer_index, 2);
        assert_eq!(d.b_layer_index, 3);
        assert_eq!(d.a_columns.len(), 5);
        assert!(d.a_columns.contains(&COL_IS_INVALID));
        assert!(d.b_columns.contains(&COL_IS_INVALID));
    }

    #[test]
    fn opcode_dispatch_air_table_trace_has_256_rows_and_all_opcodes() {
        let curve = CurveType::Bls48581;
        let t = build_table_trace_polynomials(curve);
        assert_eq!(t.num_rows, 256);
        // Spot-check a few rows: opcode byte at index i should equal i,
        // and is_real should be 1.
        for i in [0u8, 0x01, 0x50, 0x5B, 0x60, 0x7F, 0x80, 0x9F, 0xF1, 0xFF] {
            let scalar_op = &t.columns[COL_OPCODE].evaluations[i as usize];
            let expected = Scalar::from_u64(i as u64, curve);
            assert!(
                scalar_op.sub(&expected).is_zero(),
                "table opcode column at {} must equal {}",
                i, i,
            );
            let is_real = &t.columns[COL_IS_REAL].evaluations[i as usize];
            let one = Scalar::one(curve);
            assert!(
                is_real.sub(&one).is_zero(),
                "table is_real column at {} must equal 1",
                i,
            );
        }
        // The 256-row table is exactly a power of two — padded == num_rows.
        assert_eq!(t.padded_size, 256);
    }

    #[test]
    fn opcode_dispatch_air_table_constraints_vanish() {
        // The full constant table must satisfy all row-local
        // constraints (is_real binary, is_invalid binary,
        // is_invalid → zero counts).
        let curve = CurveType::Bls48581;
        let t = build_table_trace_polynomials(curve);
        let cs = OpcodeDispatchConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(17, curve);
        for row in 0..t.padded_size as usize {
            let cols: Vec<Scalar> = t.columns.iter().map(|p| p.evaluations[row].clone()).collect();
            let v = cs.evaluate_at_point(&cols, &alpha);
            assert!(v.is_zero(), "table row {} body must vanish", row);
        }
    }
}
