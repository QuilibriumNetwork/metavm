//! Combined CHAINID + CALLER + CALLVALUE context-reader opcode AIR.
//!
//! Three EVM context-reader opcodes consolidated into one AIR with a
//! shared per-row structure:
//!
//!   - CHAINID    (0x46): pushes the EIP-155 chain ID (fits in a u64).
//!   - CALLER     (0x33): pushes msg.sender (the immediate caller address).
//!   - CALLVALUE  (0x34): pushes msg.value (the current call's value).
//!
//! All three are static gas cost 2 (`G_base`). On each real row exactly one
//! of `sel_chainid`, `sel_caller`, `sel_callvalue` is set, the opcode byte
//! matches, and the pushed u256 (`value[0..4]` LE u64 limbs) is
//! algebraically bound to the corresponding context column:
//!
//!   - CHAINID:   `value_limb_0 = chain_id`, `value_limb_{1,2,3} = 0`
//!   - CALLER:    `value_limb_k = Σ caller_address[8k + i] · 256^i` using
//!                the canonical [`crate::address_opcode_air::address_to_limbs`]
//!                packing (limb 0..1 = 8 bytes each, limb 2 = low 4 bytes
//!                of the address high half + zero padding, limb 3 = 0).
//!   - CALLVALUE: `value_limb_k = call_value_limb_k` for k = 0..4.
//!
//! Composed with neighbouring AIRs via three cross-AIR LogUp descriptors:
//!
//!   - `make_context_to_chain_id_descriptor`     — CHAINID rows ↔
//!     `block_header_air::COL_CHAIN_ID`.
//!   - `make_context_to_call_frame_descriptor`   — CALLER + CALLVALUE rows
//!     ↔ `call_frame_air` caller / value-pre tuples.
//!   - `make_context_to_stack_contents_descriptor` — every real row ↔
//!     `stack_contents_air` unsorted `(pc, value_limb_0..3)` push event.
//!
//! ## Per-row layout (`NUM_COLUMNS = 65`)
//!
//! ```text
//! offset  meaning
//!   0     pc                    (u64)
//!   1     opcode                (u8 ∈ {0x46, 0x33, 0x34})
//!   2..6  value_limb_0..3       (LE u64 of pushed u256)
//!   6     gas_cost              (always 2)
//!   7     chain_id              (u64)
//!   8..28 caller_address[0..20] (u8 per limb)
//!  28..32 call_value_limb_0..3  (LE u64 of msg.value)
//!  32     sel_chainid           (binary)
//!  33     sel_caller            (binary)
//!  34     sel_callvalue         (binary)
//!  35     is_real               (binary)
//! ```
//!
//! ## Constraint catalog (`NUM_ROW_CONSTRAINTS = 17`)
//!
//!  0. `is_real_binary`            — `is_real · (is_real − 1) = 0`
//!  1. `sel_chainid_binary`        — `sel_chainid · (sel_chainid − 1) = 0`
//!  2. `sel_caller_binary`         — `sel_caller   · (sel_caller   − 1) = 0`
//!  3. `sel_callvalue_binary`      — `sel_callvalue · (sel_callvalue − 1) = 0`
//!  4. `selector_sum_eq_is_real`   — `is_real − (sel_c+sel_cl+sel_cv) = 0`
//!  5. `selector_mutex`            — `sel_c·sel_cl + sel_c·sel_cv + sel_cl·sel_cv = 0`
//!  6. `gas_cost_eq_2`             — `is_real · (gas_cost − 2) = 0`
//!  7. `chainid_opcode_eq`         — `sel_chainid · (opcode − 0x46) = 0`
//!  8. `caller_opcode_eq`          — `sel_caller  · (opcode − 0x33) = 0`
//!  9. `callvalue_opcode_eq`       — `sel_callvalue · (opcode − 0x34) = 0`
//! 10. `chainid_value_l0_eq`       — `sel_chainid · (value_l0 − chain_id) = 0`
//! 11. `chainid_value_high_zero`   — `sel_chainid · (value_l1+value_l2+value_l3) = 0`
//!                                    (combined with binary range checks on the
//!                                    sum components — see note)
//! 12. `caller_value_l0_binding`   — `sel_caller · (value_l0 − Σ addr[i]·256^i)`
//! 13. `caller_value_l1_binding`   — `sel_caller · (value_l1 − Σ addr[8+i]·256^i)`
//! 14. `caller_value_l2_binding`   — `sel_caller · (value_l2 − Σ addr[16+i]·256^i)`
//! 15. `caller_value_l3_zero`      — `sel_caller · value_l3 = 0`
//! 16. `callvalue_limbs_bind`      — `sel_callvalue · Σ_k α_k · (value_k − cv_k) = 0`
//!                                    (folded with constraint-system α — but to keep
//!                                    each constraint independent we instead emit one
//!                                    per limb; see the implementation: this expands
//!                                    `NUM_ROW_CONSTRAINTS` by 4 to 17 + 3 = 20.)

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
pub const COL_CHAIN_ID: usize = 7;
pub const COL_CALLER_ADDR_0: usize = 8;
pub const NUM_ADDR_BYTES: usize = 20;
pub const COL_CALL_VALUE_LIMB_0: usize = COL_CALLER_ADDR_0 + NUM_ADDR_BYTES; // 28
pub const COL_CALL_VALUE_LIMB_1: usize = COL_CALL_VALUE_LIMB_0 + 1; // 29
pub const COL_CALL_VALUE_LIMB_2: usize = COL_CALL_VALUE_LIMB_0 + 2; // 30
pub const COL_CALL_VALUE_LIMB_3: usize = COL_CALL_VALUE_LIMB_0 + 3; // 31
pub const COL_SEL_CHAINID: usize = 32;
pub const COL_SEL_CALLER: usize = 33;
pub const COL_SEL_CALLVALUE: usize = 34;
pub const COL_IS_REAL: usize = 35;

pub const NUM_COLUMNS: usize = 36;

// 6 structural + 1 gas + 3 opcode + 5 chainid + 4 caller + 4 callvalue = 23
// (Each callvalue limb gets its own constraint to keep degrees low.)
pub const NUM_ROW_CONSTRAINTS: usize = 23;
pub const NUM_SHIFTED: usize = 0;

// Opcode bytes
pub const CHAINID_OPCODE: u8 = 0x46;
pub const CALLER_OPCODE: u8 = 0x33;
pub const CALLVALUE_OPCODE: u8 = 0x34;
pub const CONTEXT_READER_GAS: u64 = 2;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextReaderRow {
    pub pc: u64,
    pub opcode: u8,
    pub value: [u64; 4],
    pub gas_cost: u64,
    pub chain_id: u64,
    pub caller_address: [u8; 20],
    pub call_value: [u64; 4],
    pub sel_chainid: bool,
    pub sel_caller: bool,
    pub sel_callvalue: bool,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ContextReadersWitness {
    pub rows: Vec<ContextReaderRow>,
}

/// Convert a 32-byte big-endian-or-LE u256 value (stored as `[u8; 32]`) into
/// 4 LE u64 limbs (limb 0 = least-significant 8 bytes). The events feed
/// `value` as the canonical 32-byte LE encoding already used by the EVM
/// trace's `output0` limbs (i.e. bytes 0..8 → limb 0, etc.).
pub fn value_bytes_to_limbs(value: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for k in 0..4 {
        let mut tmp = [0u8; 8];
        tmp.copy_from_slice(&value[8 * k..8 * k + 8]);
        limbs[k] = u64::from_le_bytes(tmp);
    }
    limbs
}

impl ContextReadersWitness {
    /// Build a witness from `(opcode, pc, value)` events. The host populates
    /// `chain_id` / `caller_address` / `call_value` per row based on the
    /// row's opcode kind, then exposes them as columns bound algebraically.
    ///
    /// For CHAINID:  `chain_id` = value_limb_0, others zero.
    /// For CALLER:   `caller_address` = the 20 bytes such that
    ///               `address_to_limbs(caller_address) = value`.
    /// For CALLVALUE: `call_value` = `value` (1:1 with the 4 LE limbs).
    pub fn from_events(events: &[(u8, u64, [u8; 32])]) -> Self {
        let rows: Vec<ContextReaderRow> = events
            .iter()
            .map(|(opcode, pc, value)| {
                let value_limbs = value_bytes_to_limbs(value);
                let mut row = ContextReaderRow {
                    pc: *pc,
                    opcode: *opcode,
                    value: value_limbs,
                    gas_cost: CONTEXT_READER_GAS,
                    chain_id: 0,
                    caller_address: [0u8; 20],
                    call_value: [0u64; 4],
                    sel_chainid: false,
                    sel_caller: false,
                    sel_callvalue: false,
                    is_real: true,
                };
                match *opcode {
                    CHAINID_OPCODE => {
                        row.sel_chainid = true;
                        row.chain_id = value_limbs[0];
                    }
                    CALLER_OPCODE => {
                        row.sel_caller = true;
                        // Reconstruct the 20-byte address from the LE-limb
                        // packing used by `address_to_limbs`.
                        let mut addr = [0u8; 20];
                        let l0 = value_limbs[0].to_le_bytes();
                        let l1 = value_limbs[1].to_le_bytes();
                        let l2 = value_limbs[2].to_le_bytes();
                        addr[0..8].copy_from_slice(&l0);
                        addr[8..16].copy_from_slice(&l1);
                        addr[16..20].copy_from_slice(&l2[0..4]);
                        row.caller_address = addr;
                    }
                    CALLVALUE_OPCODE => {
                        row.sel_callvalue = true;
                        row.call_value = value_limbs;
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
    w: &ContextReadersWitness,
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
        cols[COL_CHAIN_ID][r] = Scalar::from_u64(row.chain_id, curve);
        for i in 0..NUM_ADDR_BYTES {
            cols[COL_CALLER_ADDR_0 + i][r] =
                Scalar::from_u64(row.caller_address[i] as u64, curve);
        }
        cols[COL_CALL_VALUE_LIMB_0][r] = Scalar::from_u64(row.call_value[0], curve);
        cols[COL_CALL_VALUE_LIMB_1][r] = Scalar::from_u64(row.call_value[1], curve);
        cols[COL_CALL_VALUE_LIMB_2][r] = Scalar::from_u64(row.call_value[2], curve);
        cols[COL_CALL_VALUE_LIMB_3][r] = Scalar::from_u64(row.call_value[3], curve);
        cols[COL_SEL_CHAINID][r] = if row.sel_chainid { one.clone() } else { zero.clone() };
        cols[COL_SEL_CALLER][r] = if row.sel_caller { one.clone() } else { zero.clone() };
        cols[COL_SEL_CALLVALUE][r] = if row.sel_callvalue { one.clone() } else { zero.clone() };
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

pub struct ContextReadersConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ContextReadersConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for ContextReadersConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "sel_chainid_binary".into(),
            "sel_caller_binary".into(),
            "sel_callvalue_binary".into(),
            "selector_sum_eq_is_real".into(),
            "selector_mutex".into(),
            "gas_cost_eq_2".into(),
            "chainid_opcode_eq".into(),
            "caller_opcode_eq".into(),
            "callvalue_opcode_eq".into(),
            "chainid_value_l0_eq".into(),
            "chainid_value_l1_zero".into(),
            "chainid_value_l2_zero".into(),
            "chainid_value_l3_zero".into(),
            "caller_value_l0_binding".into(),
            "caller_value_l1_binding".into(),
            "caller_value_l2_binding".into(),
            "caller_value_l3_zero".into(),
            "callvalue_l0_eq".into(),
            "callvalue_l1_eq".into(),
            "callvalue_l2_eq".into(),
            "callvalue_l3_eq".into(),
            "padding_opcode_zero".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let two_gas = Scalar::from_u64(CONTEXT_READER_GAS, curve);
        let op_chainid = Scalar::from_u64(CHAINID_OPCODE as u64, curve);
        let op_caller = Scalar::from_u64(CALLER_OPCODE as u64, curve);
        let op_callvalue = Scalar::from_u64(CALLVALUE_OPCODE as u64, curve);
        let w = byte_weights(curve);
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let s_ci = &columns[COL_SEL_CHAINID][r];
            let s_cl = &columns[COL_SEL_CALLER][r];
            let s_cv = &columns[COL_SEL_CALLVALUE][r];
            let gc = &columns[COL_GAS_COST][r];
            let op = &columns[COL_OPCODE][r];
            let v0 = &columns[COL_VALUE_LIMB_0][r];
            let v1 = &columns[COL_VALUE_LIMB_1][r];
            let v2 = &columns[COL_VALUE_LIMB_2][r];
            let v3 = &columns[COL_VALUE_LIMB_3][r];
            let chain_id = &columns[COL_CHAIN_ID][r];
            let cv0 = &columns[COL_CALL_VALUE_LIMB_0][r];
            let cv1 = &columns[COL_CALL_VALUE_LIMB_1][r];
            let cv2 = &columns[COL_CALL_VALUE_LIMB_2][r];
            let cv3 = &columns[COL_CALL_VALUE_LIMB_3][r];

            // 0: is_real binary
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            // 1..3: each selector binary
            bodies[1][r] = s_ci.mul(&s_ci.sub(&one));
            bodies[2][r] = s_cl.mul(&s_cl.sub(&one));
            bodies[3][r] = s_cv.mul(&s_cv.sub(&one));
            // 4: selector sum == is_real
            let sum = s_ci.add(s_cl).add(s_cv);
            bodies[4][r] = is_real.sub(&sum);
            // 5: mutex
            bodies[5][r] = s_ci.mul(s_cl).add(&s_ci.mul(s_cv)).add(&s_cl.mul(s_cv));
            // 6: gas_cost == 2 on real rows
            bodies[6][r] = is_real.mul(&gc.sub(&two_gas));
            // 7..9: per-selector opcode binding
            bodies[7][r] = s_ci.mul(&op.sub(&op_chainid));
            bodies[8][r] = s_cl.mul(&op.sub(&op_caller));
            bodies[9][r] = s_cv.mul(&op.sub(&op_callvalue));
            // 10..13: CHAINID — value_limb_0 = chain_id, l1..l3 = 0
            bodies[10][r] = s_ci.mul(&v0.sub(chain_id));
            bodies[11][r] = s_ci.mul(v1);
            bodies[12][r] = s_ci.mul(v2);
            bodies[13][r] = s_ci.mul(v3);
            // 14..17: CALLER — limbs from address bytes (same as address_opcode_air)
            let mut s0 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_CALLER_ADDR_0 + i][r];
                s0 = s0.add(&b.mul(&w[i]));
            }
            bodies[14][r] = s_cl.mul(&v0.sub(&s0));
            let mut s1 = Scalar::zero(curve);
            for i in 0..8 {
                let b = &columns[COL_CALLER_ADDR_0 + 8 + i][r];
                s1 = s1.add(&b.mul(&w[i]));
            }
            bodies[15][r] = s_cl.mul(&v1.sub(&s1));
            let mut s2 = Scalar::zero(curve);
            for i in 0..4 {
                let b = &columns[COL_CALLER_ADDR_0 + 16 + i][r];
                s2 = s2.add(&b.mul(&w[i]));
            }
            bodies[16][r] = s_cl.mul(&v2.sub(&s2));
            bodies[17][r] = s_cl.mul(v3);
            // 18..21: CALLVALUE — value_limb_k = call_value_limb_k
            bodies[18][r] = s_cv.mul(&v0.sub(cv0));
            bodies[19][r] = s_cv.mul(&v1.sub(cv1));
            bodies[20][r] = s_cv.mul(&v2.sub(cv2));
            bodies[21][r] = s_cv.mul(&v3.sub(cv3));
            // 22: padding rows have opcode = 0
            let one_minus_real = one.sub(is_real);
            bodies[22][r] = one_minus_real.mul(op);
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two_gas = Scalar::from_u64(CONTEXT_READER_GAS, curve);
        let op_chainid = Scalar::from_u64(CHAINID_OPCODE as u64, curve);
        let op_caller = Scalar::from_u64(CALLER_OPCODE as u64, curve);
        let op_callvalue = Scalar::from_u64(CALLVALUE_OPCODE as u64, curve);
        let w = byte_weights(curve);
        let is_real = &ce[COL_IS_REAL];
        let s_ci = &ce[COL_SEL_CHAINID];
        let s_cl = &ce[COL_SEL_CALLER];
        let s_cv = &ce[COL_SEL_CALLVALUE];
        let gc = &ce[COL_GAS_COST];
        let op = &ce[COL_OPCODE];
        let v0 = &ce[COL_VALUE_LIMB_0];
        let v1 = &ce[COL_VALUE_LIMB_1];
        let v2 = &ce[COL_VALUE_LIMB_2];
        let v3 = &ce[COL_VALUE_LIMB_3];
        let chain_id = &ce[COL_CHAIN_ID];
        let cv0 = &ce[COL_CALL_VALUE_LIMB_0];
        let cv1 = &ce[COL_CALL_VALUE_LIMB_1];
        let cv2 = &ce[COL_CALL_VALUE_LIMB_2];
        let cv3 = &ce[COL_CALL_VALUE_LIMB_3];

        let mut s0 = Scalar::zero(curve);
        for i in 0..8 { s0 = s0.add(&ce[COL_CALLER_ADDR_0 + i].mul(&w[i])); }
        let mut s1 = Scalar::zero(curve);
        for i in 0..8 { s1 = s1.add(&ce[COL_CALLER_ADDR_0 + 8 + i].mul(&w[i])); }
        let mut s2 = Scalar::zero(curve);
        for i in 0..4 { s2 = s2.add(&ce[COL_CALLER_ADDR_0 + 16 + i].mul(&w[i])); }

        let sum = s_ci.add(s_cl).add(s_cv);
        let one_minus_real = one.sub(is_real);
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            s_ci.mul(&s_ci.sub(&one)),
            s_cl.mul(&s_cl.sub(&one)),
            s_cv.mul(&s_cv.sub(&one)),
            is_real.sub(&sum),
            s_ci.mul(s_cl).add(&s_ci.mul(s_cv)).add(&s_cl.mul(s_cv)),
            is_real.mul(&gc.sub(&two_gas)),
            s_ci.mul(&op.sub(&op_chainid)),
            s_cl.mul(&op.sub(&op_caller)),
            s_cv.mul(&op.sub(&op_callvalue)),
            s_ci.mul(&v0.sub(chain_id)),
            s_ci.mul(v1),
            s_ci.mul(v2),
            s_ci.mul(v3),
            s_cl.mul(&v0.sub(&s0)),
            s_cl.mul(&v1.sub(&s1)),
            s_cl.mul(&v2.sub(&s2)),
            s_cl.mul(v3),
            s_cv.mul(&v0.sub(cv0)),
            s_cv.mul(&v1.sub(cv1)),
            s_cv.mul(&v2.sub(cv2)),
            s_cv.mul(&v3.sub(cv3)),
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
        let two_gas_p = vec![Scalar::from_u64(CONTEXT_READER_GAS, curve)];
        let op_chainid_p = vec![Scalar::from_u64(CHAINID_OPCODE as u64, curve)];
        let op_caller_p = vec![Scalar::from_u64(CALLER_OPCODE as u64, curve)];
        let op_callvalue_p = vec![Scalar::from_u64(CALLVALUE_OPCODE as u64, curve)];
        let w = byte_weights(curve);
        let is_real = &cc[COL_IS_REAL];
        let s_ci = &cc[COL_SEL_CHAINID];
        let s_cl = &cc[COL_SEL_CALLER];
        let s_cv = &cc[COL_SEL_CALLVALUE];
        let gc = &cc[COL_GAS_COST];
        let op = &cc[COL_OPCODE];
        let v0 = &cc[COL_VALUE_LIMB_0];
        let v1 = &cc[COL_VALUE_LIMB_1];
        let v2 = &cc[COL_VALUE_LIMB_2];
        let v3 = &cc[COL_VALUE_LIMB_3];
        let chain_id = &cc[COL_CHAIN_ID];
        let cv0 = &cc[COL_CALL_VALUE_LIMB_0];
        let cv1 = &cc[COL_CALL_VALUE_LIMB_1];
        let cv2 = &cc[COL_CALL_VALUE_LIMB_2];
        let cv3 = &cc[COL_CALL_VALUE_LIMB_3];

        let mut s0: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_CALLER_ADDR_0 + i], &w[i]);
            s0 = poly_add(&s0, &scaled, curve);
        }
        let mut s1: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..8 {
            let scaled = poly_scalar_mul(&cc[COL_CALLER_ADDR_0 + 8 + i], &w[i]);
            s1 = poly_add(&s1, &scaled, curve);
        }
        let mut s2: Vec<Scalar> = vec![Scalar::zero(curve)];
        for i in 0..4 {
            let scaled = poly_scalar_mul(&cc[COL_CALLER_ADDR_0 + 16 + i], &w[i]);
            s2 = poly_add(&s2, &scaled, curve);
        }

        let sum = poly_add(&poly_add(s_ci, s_cl, curve), s_cv, curve);
        let one_minus_real = poly_sub(&one_p, is_real, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            poly_mul(is_real, &poly_sub(is_real, &one_p, curve), curve),
            poly_mul(s_ci, &poly_sub(s_ci, &one_p, curve), curve),
            poly_mul(s_cl, &poly_sub(s_cl, &one_p, curve), curve),
            poly_mul(s_cv, &poly_sub(s_cv, &one_p, curve), curve),
            poly_sub(is_real, &sum, curve),
            poly_add(
                &poly_add(
                    &poly_mul(s_ci, s_cl, curve),
                    &poly_mul(s_ci, s_cv, curve),
                    curve,
                ),
                &poly_mul(s_cl, s_cv, curve),
                curve,
            ),
            poly_mul(is_real, &poly_sub(gc, &two_gas_p, curve), curve),
            poly_mul(s_ci, &poly_sub(op, &op_chainid_p, curve), curve),
            poly_mul(s_cl, &poly_sub(op, &op_caller_p, curve), curve),
            poly_mul(s_cv, &poly_sub(op, &op_callvalue_p, curve), curve),
            poly_mul(s_ci, &poly_sub(v0, chain_id, curve), curve),
            poly_mul(s_ci, v1, curve),
            poly_mul(s_ci, v2, curve),
            poly_mul(s_ci, v3, curve),
            poly_mul(s_cl, &poly_sub(v0, &s0, curve), curve),
            poly_mul(s_cl, &poly_sub(v1, &s1, curve), curve),
            poly_mul(s_cl, &poly_sub(v2, &s2, curve), curve),
            poly_mul(s_cl, v3, curve),
            poly_mul(s_cv, &poly_sub(v0, cv0, curve), curve),
            poly_mul(s_cv, &poly_sub(v1, cv1, curve), curve),
            poly_mul(s_cv, &poly_sub(v2, cv2, curve), curve),
            poly_mul(s_cv, &poly_sub(v3, cv3, curve), curve),
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
        // binary range checks on is_real + 3 selectors
        for (name, col) in [
            ("context_readers_is_real_1bit", COL_IS_REAL),
            ("context_readers_sel_chainid_1bit", COL_SEL_CHAINID),
            ("context_readers_sel_caller_1bit", COL_SEL_CALLER),
            ("context_readers_sel_callvalue_1bit", COL_SEL_CALLVALUE),
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
                label: "context_readers_opcode_8bit".into(),
                column_index: COL_OPCODE,
                max_bits: 8,
                selector_column: None,
            },
            tbl_byte,
        ));
        // 20 caller-address byte range checks
        for i in 0..NUM_ADDR_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("context_readers_caller_byte_{i}_8bit"),
                    column_index: COL_CALLER_ADDR_0 + i,
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

/// CHAINID rows ↔ `block_header_air::COL_CHAIN_ID`. The A side is gated by
/// `sel_chainid` and exposes a single column (`chain_id`); the B side is
/// `block_header_air`'s `COL_CHAIN_ID` (single u64 column) gated by its
/// own `COL_IS_REAL`. This binds the chain id pushed by the EVM CHAINID
/// opcode to the canonical block-header chain id.
pub fn make_context_to_chain_id_descriptor(
    context_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "context_readers_to_chain_id_v1".into(),
        a_layer_index: context_layer_index,
        a_columns: vec![COL_CHAIN_ID],
        a_selector_column: Some(COL_SEL_CHAINID),
        b_layer_index: block_header_layer_index,
        b_columns: vec![metavm_zkp::block_header_air::COL_CHAIN_ID],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// CALLER ↔ `call_frame_air::COL_CALLER_PRE_L0..L3`. Single descriptor
/// binding the 4 caller-address limbs (gated by `sel_caller`) to the
/// call frame's pre-call caller tuple. Combined with the constraint-system
/// limb binding (constraints 14..17), this proves the pushed address
/// equals the active frame's caller.
pub fn make_context_to_call_frame_descriptor(
    context_layer_index: usize,
    call_frame_layer_index: usize,
    call_frame_selector_col: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "context_readers_caller_to_call_frame_v1".into(),
        a_layer_index: context_layer_index,
        a_columns: vec![
            COL_VALUE_LIMB_0,
            COL_VALUE_LIMB_1,
            COL_VALUE_LIMB_2,
            COL_VALUE_LIMB_3,
        ],
        a_selector_column: Some(COL_SEL_CALLER),
        b_layer_index: call_frame_layer_index,
        b_columns: vec![
            crate::call_frame_air::COL_CALLER_PRE_L0,
            crate::call_frame_air::COL_CALLER_PRE_L1,
            crate::call_frame_air::COL_CALLER_PRE_L2,
            crate::call_frame_air::COL_CALLER_PRE_L3,
        ],
        b_selector_column: Some(call_frame_selector_col),
    }
}

/// CALLVALUE ↔ `call_frame_air::COL_VALUE_PRE_L0..L3`. Companion descriptor
/// to the CALLER linkage above: binds the 4 call_value limbs (gated by
/// `sel_callvalue`) to the call frame's pre-call value tuple.
pub fn make_context_to_call_frame_value_descriptor(
    context_layer_index: usize,
    call_frame_layer_index: usize,
    call_frame_selector_col: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "context_readers_callvalue_to_call_frame_v1".into(),
        a_layer_index: context_layer_index,
        a_columns: vec![
            COL_CALL_VALUE_LIMB_0,
            COL_CALL_VALUE_LIMB_1,
            COL_CALL_VALUE_LIMB_2,
            COL_CALL_VALUE_LIMB_3,
        ],
        a_selector_column: Some(COL_SEL_CALLVALUE),
        b_layer_index: call_frame_layer_index,
        b_columns: vec![
            crate::call_frame_air::COL_VALUE_PRE_L0,
            crate::call_frame_air::COL_VALUE_PRE_L1,
            crate::call_frame_air::COL_VALUE_PRE_L2,
            crate::call_frame_air::COL_VALUE_PRE_L3,
        ],
        b_selector_column: Some(call_frame_selector_col),
    }
}

/// Every real context-reader row ↔ `stack_contents_air` unsorted view
/// `(pc, value_limb_0..3)`. Binds the stack push event at this PC to the
/// canonical u256 encoding of the corresponding context value.
pub fn make_context_to_stack_contents_descriptor(
    context_layer_index: usize,
    stack_contents_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "context_readers_to_stack_contents_push_v1".into(),
        a_layer_index: context_layer_index,
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

    fn assert_all_vanish(trace: &TracePolynomials, cs: &ContextReadersConstraintSystem) {
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

    fn chainid_event(pc: u64, chain_id: u64) -> (u8, u64, [u8; 32]) {
        let mut val = [0u8; 32];
        val[0..8].copy_from_slice(&chain_id.to_le_bytes());
        (CHAINID_OPCODE, pc, val)
    }

    fn caller_event(pc: u64, addr: [u8; 20]) -> (u8, u64, [u8; 32]) {
        // Pack the address into the canonical LE-limb encoding used by
        // address_to_limbs, then re-serialise as 32 LE bytes.
        let limbs = crate::address_opcode_air::address_to_limbs(&addr);
        let mut val = [0u8; 32];
        for k in 0..4 {
            val[8 * k..8 * k + 8].copy_from_slice(&limbs[k].to_le_bytes());
        }
        (CALLER_OPCODE, pc, val)
    }

    fn callvalue_event(pc: u64, limbs: [u64; 4]) -> (u8, u64, [u8; 32]) {
        let mut val = [0u8; 32];
        for k in 0..4 {
            val[8 * k..8 * k + 8].copy_from_slice(&limbs[k].to_le_bytes());
        }
        (CALLVALUE_OPCODE, pc, val)
    }

    #[test]
    fn context_readers_chainid_mainnet_vanishes() {
        let w = ContextReadersWitness::from_events(&[chainid_event(7, 1)]);
        assert_eq!(w.rows.len(), 1);
        let row = w.rows[0];
        assert!(row.sel_chainid && !row.sel_caller && !row.sel_callvalue);
        assert_eq!(row.opcode, CHAINID_OPCODE);
        assert_eq!(row.gas_cost, CONTEXT_READER_GAS);
        assert_eq!(row.chain_id, 1);
        assert_eq!(row.value, [1, 0, 0, 0]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn context_readers_caller_vanishes() {
        let mut addr = [0u8; 20];
        for i in 0..20 { addr[i] = (i as u8) + 1; }
        let w = ContextReadersWitness::from_events(&[caller_event(13, addr)]);
        assert_eq!(w.rows.len(), 1);
        let row = w.rows[0];
        assert!(row.sel_caller && !row.sel_chainid && !row.sel_callvalue);
        assert_eq!(row.opcode, CALLER_OPCODE);
        assert_eq!(row.caller_address, addr);
        // limb 3 must be zero (20-byte address fits in 160 bits)
        assert_eq!(row.value[3], 0);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn context_readers_callvalue_vanishes() {
        let limbs = [0xDEADBEEFu64, 0xC0FFEEu64, 0xABCDu64, 0x42u64];
        let w = ContextReadersWitness::from_events(&[callvalue_event(21, limbs)]);
        assert_eq!(w.rows.len(), 1);
        let row = w.rows[0];
        assert!(row.sel_callvalue && !row.sel_chainid && !row.sel_caller);
        assert_eq!(row.opcode, CALLVALUE_OPCODE);
        assert_eq!(row.call_value, limbs);
        assert_eq!(row.value, limbs);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn context_readers_tampered_value_detected() {
        let curve = CurveType::Bls48581;
        let w = ContextReadersWitness::from_events(&[
            chainid_event(0, 1),
        ]);
        let mut t = build_trace_polynomials(&w, curve);
        // Tamper: set value_limb_0 = 99 while keeping chain_id = 1.
        t.columns[COL_VALUE_LIMB_0].evaluations[0] = Scalar::from_u64(99, curve);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // chainid_value_l0_eq is constraint #10
        assert!(!bodies[10][0].is_zero(), "chainid_value_l0_eq must fire");
    }

    #[test]
    fn context_readers_tampered_callvalue_limb_detected() {
        let curve = CurveType::Bls48581;
        let w = ContextReadersWitness::from_events(&[callvalue_event(0, [7, 8, 9, 10])]);
        let mut t = build_trace_polynomials(&w, curve);
        // Tamper: change call_value_limb_2 to break the binding.
        t.columns[COL_CALL_VALUE_LIMB_2].evaluations[0] = Scalar::from_u64(99, curve);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // callvalue_l2_eq is constraint #20
        assert!(!bodies[20][0].is_zero(), "callvalue_l2_eq must fire");
    }

    #[test]
    fn context_readers_tampered_gas_detected() {
        let curve = CurveType::Bls48581;
        let w = ContextReadersWitness::from_events(&[chainid_event(0, 1)]);
        let mut t = build_trace_polynomials(&w, curve);
        t.columns[COL_GAS_COST].evaluations[0] = Scalar::from_u64(3, curve);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[6][0].is_zero(), "gas_cost_eq_2 must fire");
    }

    #[test]
    fn context_readers_multi_row_mixed_vanishes() {
        let mut addr = [0u8; 20];
        for i in 0..20 { addr[i] = 0xAA; }
        let w = ContextReadersWitness::from_events(&[
            chainid_event(0, 1),
            caller_event(5, addr),
            callvalue_event(11, [100, 0, 0, 0]),
            chainid_event(20, 11155111), // sepolia
        ]);
        assert_eq!(w.rows.len(), 4);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        assert_all_vanish(&t, &cs);
    }

    #[test]
    fn context_readers_descriptors_well_formed() {
        let chain = make_context_to_chain_id_descriptor(0, 1);
        assert_eq!(chain.label, "context_readers_to_chain_id_v1");
        assert_eq!(chain.a_layer_index, 0);
        assert_eq!(chain.b_layer_index, 1);
        assert_eq!(chain.a_columns, vec![COL_CHAIN_ID]);
        assert_eq!(chain.a_selector_column, Some(COL_SEL_CHAINID));
        assert_eq!(
            chain.b_columns,
            vec![metavm_zkp::block_header_air::COL_CHAIN_ID]
        );
        assert_eq!(
            chain.b_selector_column,
            Some(metavm_zkp::block_header_air::COL_IS_REAL)
        );

        let cf = make_context_to_call_frame_descriptor(
            2, 3, crate::call_frame_air::COL_IS_REAL,
        );
        assert_eq!(cf.label, "context_readers_caller_to_call_frame_v1");
        assert_eq!(cf.a_columns.len(), 4);
        assert_eq!(cf.b_columns.len(), 4);
        assert_eq!(cf.a_selector_column, Some(COL_SEL_CALLER));
        assert_eq!(
            cf.b_columns,
            vec![
                crate::call_frame_air::COL_CALLER_PRE_L0,
                crate::call_frame_air::COL_CALLER_PRE_L1,
                crate::call_frame_air::COL_CALLER_PRE_L2,
                crate::call_frame_air::COL_CALLER_PRE_L3,
            ]
        );

        let cfv = make_context_to_call_frame_value_descriptor(
            2, 3, crate::call_frame_air::COL_IS_REAL,
        );
        assert_eq!(cfv.label, "context_readers_callvalue_to_call_frame_v1");
        assert_eq!(cfv.a_selector_column, Some(COL_SEL_CALLVALUE));
        assert_eq!(
            cfv.b_columns,
            vec![
                crate::call_frame_air::COL_VALUE_PRE_L0,
                crate::call_frame_air::COL_VALUE_PRE_L1,
                crate::call_frame_air::COL_VALUE_PRE_L2,
                crate::call_frame_air::COL_VALUE_PRE_L3,
            ]
        );

        let sc = make_context_to_stack_contents_descriptor(4, 5);
        assert_eq!(sc.label, "context_readers_to_stack_contents_push_v1");
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
    fn context_readers_column_layout_pinned() {
        assert_eq!(COL_PC, 0);
        assert_eq!(COL_OPCODE, 1);
        assert_eq!(COL_VALUE_LIMB_0, 2);
        assert_eq!(COL_VALUE_LIMB_3, 5);
        assert_eq!(COL_GAS_COST, 6);
        assert_eq!(COL_CHAIN_ID, 7);
        assert_eq!(COL_CALLER_ADDR_0, 8);
        assert_eq!(COL_CALL_VALUE_LIMB_0, 28);
        assert_eq!(COL_CALL_VALUE_LIMB_3, 31);
        assert_eq!(COL_SEL_CHAINID, 32);
        assert_eq!(COL_SEL_CALLER, 33);
        assert_eq!(COL_SEL_CALLVALUE, 34);
        assert_eq!(COL_IS_REAL, 35);
        assert_eq!(NUM_COLUMNS, 36);
        assert_eq!(NUM_ROW_CONSTRAINTS, 23);
        assert_eq!(CHAINID_OPCODE, 0x46);
        assert_eq!(CALLER_OPCODE, 0x33);
        assert_eq!(CALLVALUE_OPCODE, 0x34);
        assert_eq!(CONTEXT_READER_GAS, 2);
    }

    #[test]
    fn context_readers_evaluate_at_point_zero_on_honest() {
        let w = ContextReadersWitness::from_events(&[callvalue_event(3, [1, 2, 3, 4])]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xBEEF, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let row_evals: Vec<Scalar> = cr.iter().map(|c| c[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "evaluate_at_point should be zero on honest row");
    }

    #[test]
    fn context_readers_lookup_declarations_well_formed() {
        let cs = ContextReadersConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 4 binary + 1 opcode + 20 caller bytes = 25
        assert_eq!(req.declarations.len(), 25);
        assert_eq!(req.tables.len(), 2);
        assert_eq!(req.tables[0].bits, 1);
        assert_eq!(req.tables[1].bits, 8);
    }

    #[test]
    fn context_readers_tampered_selector_sum_detected() {
        let curve = CurveType::Bls48581;
        let w = ContextReadersWitness::from_events(&[chainid_event(0, 1)]);
        let mut t = build_trace_polynomials(&w, curve);
        // Tamper: drop sel_chainid so is_real=1 but selector sum=0.
        t.columns[COL_SEL_CHAINID].evaluations[0] = Scalar::zero(curve);
        let cs = ContextReadersConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // selector_sum_eq_is_real is #4
        assert!(!bodies[4][0].is_zero(), "selector_sum_eq_is_real must fire");
    }
}
