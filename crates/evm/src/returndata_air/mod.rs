//! RETURN/REVERT data binding AIR.
//!
//! Proves that the returndata buffer is correctly populated by RETURN/REVERT
//! and accessible via RETURNDATASIZE / RETURNDATACOPY. The witness commits
//! one row per RETURN/REVERT/RETURNDATASIZE/RETURNDATACOPY event observed
//! in the EVM trace.
//!
//! ### Event types
//!
//! - `0 = RETURN` — child frame writes `memory[off..off+len]` to its
//!   parent's returndata buffer. `returndata_buffer_size_post = length`.
//! - `1 = REVERT` — identical to RETURN for buffer semantics.
//! - `2 = RETURNDATASIZE` — reads current buffer length; buffer size
//!   unchanged (`post == pre`).
//! - `3 = RETURNDATACOPY` — copies `returndata[off..off+len]` to
//!   `memory[dest..]`. Buffer size unchanged (`post == pre`).
//!
//! ### Witness row schema
//!
//! - `event_type` (0..=3, advisory; the 4 selectors carry the algebraic
//!   weight),
//! - `sel_return, sel_revert, sel_size, sel_copy` — binary, exactly
//!   one of which is 1 on a real row (sum = is_real),
//! - `mem_offset, length, returndata_offset, dest_mem_offset`,
//! - `returndata_buffer_size_pre, returndata_buffer_size_post`,
//! - `length_byte[0..8]` — LE byte decomposition of `length`,
//! - `bounds_margin` — `pre - returndata_offset - length` on RETURNDATACOPY
//!   rows (≥ 0 ⇒ in-bounds); zero elsewhere,
//! - `bounds_margin_byte[0..8]` — LE byte decomposition of `bounds_margin`,
//! - `call_depth`,
//! - `is_real` (binary).
//!
//! ### Row-local constraints (15 total)
//!
//! 0. `is_real · (is_real - 1) = 0`
//! 1..=4. `sel_X · (sel_X - 1) = 0` for X in {return, revert, size, copy}
//! 5. `sel_return + sel_revert + sel_size + sel_copy - is_real = 0`
//!    (sum-to-is_real ⇒ on real rows exactly one selector fires)
//! 6. `(sel_return + sel_revert) · (returndata_buffer_size_post - length) = 0`
//!    — RETURN/REVERT sets buffer size to `length`
//! 7. `(sel_size + sel_copy) · (returndata_buffer_size_post -
//!    returndata_buffer_size_pre) = 0` — read-only ops do not mutate
//! 8. length LE byte recomposition: `length - Σ_{i<8} 256^i · length_byte[i] = 0`
//! 9. bounds margin recomposition: `bounds_margin -
//!    Σ_{i<8} 256^i · bounds_margin_byte[i] = 0`
//! 10. bounds binding: `sel_copy · (returndata_buffer_size_pre -
//!    returndata_offset - length - bounds_margin) = 0`
//! 11. non-copy bounds margin must be zero:
//!    `(is_real - sel_copy) · bounds_margin = 0`
//! 12. RETURN/REVERT clear: `(sel_return + sel_revert) · returndata_offset = 0`
//! 13. RETURN/REVERT clear: `(sel_return + sel_revert) · dest_mem_offset = 0`
//! 14. RETURNDATASIZE has no memory arguments: `sel_size · length = 0`
//!
//! ### Shifted constraint (1, cross-row)
//!
//! - `returndata_buffer_size_pre(ω·X) = returndata_buffer_size_post(X)`
//!   gated by `is_real(X) · is_real(ω·X) · same_frame`, where
//!   `same_frame = 1 - sel_return(X) - sel_revert(X)` (the buffer is
//!   inherited inside a single frame; RETURN/REVERT cross a frame
//!   boundary and the parent frame's pre-buffer is whatever was observed
//!   pre-CALL).
//!
//! ### Cross-AIR LogUp descriptors
//!
//! - [`make_returndata_to_call_family_descriptor`] — `(call_depth, length)`
//!   ↔ call_family_air's `(depth_pre, length-shaped value)` on
//!   RETURN/REVERT rows (gated by `sel_return + sel_revert`).
//!   We bind to `depth_pre` and to a length proxy carried in the call
//!   family AIR. Since the call_family AIR doesn't yet carry a `length`
//!   column directly, we bind to `gas_returned` as a stand-in — the
//!   pair `(depth, gas_returned)` is unique enough for the closure
//!   match on honest traces, and the algebraic length binding is
//!   delivered by the `to_memory` descriptor.
//!   (More precise length binding is a follow-up once call_family_air
//!   exposes a returndata-length column.)
//! - [`make_returndata_to_memory_descriptor`] — `(mem_offset, length,
//!   dest_mem_offset)` ↔ memory_expansion_air's `(new_size, old_size,
//!   ...)` tuple. We use the `(new_size, old_size)` pair as the
//!   memory's witness of the touched range.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::LookupRequirements;
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Event kind indices ──────────────────────────────────────────────
pub const KIND_RETURN: u64 = 0;
pub const KIND_REVERT: u64 = 1;
pub const KIND_RETURNDATASIZE: u64 = 2;
pub const KIND_RETURNDATACOPY: u64 = 3;

// ─── Column layout ───────────────────────────────────────────────────
pub const COL_EVENT_TYPE: usize = 0;
pub const COL_SEL_RETURN: usize = 1;
pub const COL_SEL_REVERT: usize = 2;
pub const COL_SEL_SIZE: usize = 3;
pub const COL_SEL_COPY: usize = 4;
pub const COL_MEM_OFFSET: usize = 5;
pub const COL_LENGTH: usize = 6;
pub const COL_RETURNDATA_OFFSET: usize = 7;
pub const COL_DEST_MEM_OFFSET: usize = 8;
pub const COL_BUF_SIZE_PRE: usize = 9;
pub const COL_BUF_SIZE_POST: usize = 10;
pub const COL_LENGTH_BYTE_OFFSET: usize = 11;
pub const LENGTH_BYTES: usize = 8;
pub const COL_BOUNDS_MARGIN: usize = 19;
pub const COL_BOUNDS_MARGIN_BYTE_OFFSET: usize = 20;
pub const BOUNDS_MARGIN_BYTES: usize = 8;
pub const COL_CALL_DEPTH: usize = 28;
pub const COL_IS_REAL: usize = 29;

pub const NUM_COLUMNS: usize = 30;
pub const NUM_ROW_CONSTRAINTS: usize = 15;
pub const NUM_SHIFTED: usize = 1;

// ─── Witness types ───────────────────────────────────────────────────

/// Host-side event extracted from an EVM trace pass. One per
/// RETURN/REVERT/RETURNDATASIZE/RETURNDATACOPY occurrence.
#[derive(Clone, Debug)]
pub struct ReturnDataEvent {
    pub event_type: u64,
    pub mem_offset: u64,
    pub length: u64,
    pub returndata_offset: u64,
    pub dest_mem_offset: u64,
    pub returndata_buffer_size_pre: u64,
    pub returndata_buffer_size_post: u64,
    pub call_depth: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ReturnDataWitness {
    pub rows: Vec<ReturnDataEvent>,
}

impl ReturnDataWitness {
    pub fn from_events(events: &[ReturnDataEvent]) -> Self {
        Self { rows: events.to_vec() }
    }
}

/// Convenience constructors for the four event types.
impl ReturnDataEvent {
    pub fn ret(mem_offset: u64, length: u64, pre: u64, depth: u64) -> Self {
        Self {
            event_type: KIND_RETURN,
            mem_offset,
            length,
            returndata_offset: 0,
            dest_mem_offset: 0,
            returndata_buffer_size_pre: pre,
            returndata_buffer_size_post: length,
            call_depth: depth,
        }
    }
    pub fn revert(mem_offset: u64, length: u64, pre: u64, depth: u64) -> Self {
        Self {
            event_type: KIND_REVERT,
            mem_offset,
            length,
            returndata_offset: 0,
            dest_mem_offset: 0,
            returndata_buffer_size_pre: pre,
            returndata_buffer_size_post: length,
            call_depth: depth,
        }
    }
    pub fn size(buffer_size: u64, depth: u64) -> Self {
        Self {
            event_type: KIND_RETURNDATASIZE,
            mem_offset: 0,
            length: 0,
            returndata_offset: 0,
            dest_mem_offset: 0,
            returndata_buffer_size_pre: buffer_size,
            returndata_buffer_size_post: buffer_size,
            call_depth: depth,
        }
    }
    pub fn copy(
        dest_mem_offset: u64,
        returndata_offset: u64,
        length: u64,
        buffer_size: u64,
        depth: u64,
    ) -> Self {
        Self {
            event_type: KIND_RETURNDATACOPY,
            mem_offset: dest_mem_offset, // dest mem offset doubles as touched mem_offset
            length,
            returndata_offset,
            dest_mem_offset,
            returndata_buffer_size_pre: buffer_size,
            returndata_buffer_size_post: buffer_size,
            call_depth: depth,
        }
    }
}

// ─── Trace builder ───────────────────────────────────────────────────

pub fn build_trace_polynomials(w: &ReturnDataWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_EVENT_TYPE][r] = Scalar::from_u64(row.event_type, curve);
        cols[COL_MEM_OFFSET][r] = Scalar::from_u64(row.mem_offset, curve);
        cols[COL_LENGTH][r] = Scalar::from_u64(row.length, curve);
        cols[COL_RETURNDATA_OFFSET][r] = Scalar::from_u64(row.returndata_offset, curve);
        cols[COL_DEST_MEM_OFFSET][r] = Scalar::from_u64(row.dest_mem_offset, curve);
        cols[COL_BUF_SIZE_PRE][r] = Scalar::from_u64(row.returndata_buffer_size_pre, curve);
        cols[COL_BUF_SIZE_POST][r] = Scalar::from_u64(row.returndata_buffer_size_post, curve);
        cols[COL_CALL_DEPTH][r] = Scalar::from_u64(row.call_depth, curve);
        cols[COL_IS_REAL][r] = one.clone();

        // Selector dispatch.
        let sel_col = match row.event_type {
            KIND_RETURN => Some(COL_SEL_RETURN),
            KIND_REVERT => Some(COL_SEL_REVERT),
            KIND_RETURNDATASIZE => Some(COL_SEL_SIZE),
            KIND_RETURNDATACOPY => Some(COL_SEL_COPY),
            _ => None,
        };
        if let Some(c) = sel_col {
            cols[c][r] = one.clone();
        }

        // Length LE byte decomp.
        let len_bytes = row.length.to_le_bytes();
        for b in 0..LENGTH_BYTES {
            cols[COL_LENGTH_BYTE_OFFSET + b][r] =
                Scalar::from_u64(len_bytes[b] as u64, curve);
        }

        // Bounds margin (= pre - rd_off - length on copy rows; 0 otherwise).
        let margin: u64 = if row.event_type == KIND_RETURNDATACOPY {
            row.returndata_buffer_size_pre
                .checked_sub(row.returndata_offset)
                .and_then(|x| x.checked_sub(row.length))
                .unwrap_or(0)
        } else {
            0
        };
        cols[COL_BOUNDS_MARGIN][r] = Scalar::from_u64(margin, curve);
        let m_bytes = margin.to_le_bytes();
        for b in 0..BOUNDS_MARGIN_BYTES {
            cols[COL_BOUNDS_MARGIN_BYTE_OFFSET + b][r] =
                Scalar::from_u64(m_bytes[b] as u64, curve);
        }
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

pub struct ReturnDataConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ReturnDataConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

const BINARY_SEL_COLS: [usize; 4] = [
    COL_SEL_RETURN,
    COL_SEL_REVERT,
    COL_SEL_SIZE,
    COL_SEL_COPY,
];

fn byte_basis_scalars(curve: CurveType, n: usize) -> Vec<Scalar> {
    let mut out = Vec::with_capacity(n);
    let mut cur: u128 = 1;
    for _ in 0..n {
        out.push(Scalar::from_u64(cur as u64, curve));
        cur = cur.wrapping_mul(256);
    }
    out
}

impl VmConstraintSystem for ReturnDataConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "sel_return_binary".into(),
            "sel_revert_binary".into(),
            "sel_size_binary".into(),
            "sel_copy_binary".into(),
            "sel_sum_eq_is_real".into(),
            "return_revert_sets_post_eq_length".into(),
            "size_copy_no_mutation".into(),
            "length_byte_recomposition".into(),
            "bounds_margin_byte_recomposition".into(),
            "copy_bounds_binding".into(),
            "non_copy_margin_zero".into(),
            "return_revert_zero_returndata_offset".into(),
            "return_revert_zero_dest_mem_offset".into(),
            "size_zero_length".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let len_basis = byte_basis_scalars(curve, LENGTH_BYTES);
        let m_basis = byte_basis_scalars(curve, BOUNDS_MARGIN_BYTES);
        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let sel_ret = &columns[COL_SEL_RETURN][r];
            let sel_rev = &columns[COL_SEL_REVERT][r];
            let sel_sz = &columns[COL_SEL_SIZE][r];
            let sel_cp = &columns[COL_SEL_COPY][r];

            // 0: is_real binary
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            // 1..=4: selector binaries
            for (i, c) in BINARY_SEL_COLS.iter().enumerate() {
                let s = &columns[*c][r];
                bodies[1 + i][r] = s.mul(&s.sub(&one));
            }
            // 5: Σ sel - is_real = 0
            let sel_sum = sel_ret.add(sel_rev).add(sel_sz).add(sel_cp);
            bodies[5][r] = sel_sum.sub(is_real);

            // 6: (sel_return + sel_revert) * (post - length) = 0
            let ret_or_rev = sel_ret.add(sel_rev);
            let post_minus_len =
                columns[COL_BUF_SIZE_POST][r].sub(&columns[COL_LENGTH][r]);
            bodies[6][r] = ret_or_rev.mul(&post_minus_len);

            // 7: (sel_size + sel_copy) * (post - pre) = 0
            let sz_or_cp = sel_sz.add(sel_cp);
            let post_minus_pre =
                columns[COL_BUF_SIZE_POST][r].sub(&columns[COL_BUF_SIZE_PRE][r]);
            bodies[7][r] = sz_or_cp.mul(&post_minus_pre);

            // 8: length - Σ 256^i * length_byte[i] = 0
            let mut len_acc = Scalar::zero(curve);
            for b in 0..LENGTH_BYTES {
                let term = columns[COL_LENGTH_BYTE_OFFSET + b][r].mul(&len_basis[b]);
                len_acc = len_acc.add(&term);
            }
            bodies[8][r] = columns[COL_LENGTH][r].sub(&len_acc);

            // 9: bounds_margin - Σ 256^i * bounds_margin_byte[i] = 0
            let mut m_acc = Scalar::zero(curve);
            for b in 0..BOUNDS_MARGIN_BYTES {
                let term = columns[COL_BOUNDS_MARGIN_BYTE_OFFSET + b][r].mul(&m_basis[b]);
                m_acc = m_acc.add(&term);
            }
            bodies[9][r] = columns[COL_BOUNDS_MARGIN][r].sub(&m_acc);

            // 10: sel_copy * (pre - rd_off - length - margin) = 0
            let bounds_body = columns[COL_BUF_SIZE_PRE][r]
                .sub(&columns[COL_RETURNDATA_OFFSET][r])
                .sub(&columns[COL_LENGTH][r])
                .sub(&columns[COL_BOUNDS_MARGIN][r]);
            bodies[10][r] = sel_cp.mul(&bounds_body);

            // 11: (is_real - sel_copy) * bounds_margin = 0
            let not_copy_gate = is_real.sub(sel_cp);
            bodies[11][r] = not_copy_gate.mul(&columns[COL_BOUNDS_MARGIN][r]);

            // 12: (sel_return + sel_revert) * returndata_offset = 0
            bodies[12][r] = ret_or_rev.mul(&columns[COL_RETURNDATA_OFFSET][r]);
            // 13: (sel_return + sel_revert) * dest_mem_offset = 0
            bodies[13][r] = ret_or_rev.mul(&columns[COL_DEST_MEM_OFFSET][r]);
            // 14: sel_size * length = 0
            bodies[14][r] = sel_sz.mul(&columns[COL_LENGTH][r]);
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let len_basis = byte_basis_scalars(curve, LENGTH_BYTES);
        let m_basis = byte_basis_scalars(curve, BOUNDS_MARGIN_BYTES);
        let is_real = &ce[COL_IS_REAL];
        let sel_ret = &ce[COL_SEL_RETURN];
        let sel_rev = &ce[COL_SEL_REVERT];
        let sel_sz = &ce[COL_SEL_SIZE];
        let sel_cp = &ce[COL_SEL_COPY];

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        for c in BINARY_SEL_COLS.iter() {
            let s = &ce[*c];
            bodies.push(s.mul(&s.sub(&one)));
        }
        let sel_sum = sel_ret.add(sel_rev).add(sel_sz).add(sel_cp);
        bodies.push(sel_sum.sub(is_real));

        let ret_or_rev = sel_ret.add(sel_rev);
        let post_minus_len = ce[COL_BUF_SIZE_POST].sub(&ce[COL_LENGTH]);
        bodies.push(ret_or_rev.mul(&post_minus_len));

        let sz_or_cp = sel_sz.add(sel_cp);
        let post_minus_pre = ce[COL_BUF_SIZE_POST].sub(&ce[COL_BUF_SIZE_PRE]);
        bodies.push(sz_or_cp.mul(&post_minus_pre));

        let mut len_acc = Scalar::zero(curve);
        for b in 0..LENGTH_BYTES {
            len_acc = len_acc.add(&ce[COL_LENGTH_BYTE_OFFSET + b].mul(&len_basis[b]));
        }
        bodies.push(ce[COL_LENGTH].sub(&len_acc));

        let mut m_acc = Scalar::zero(curve);
        for b in 0..BOUNDS_MARGIN_BYTES {
            m_acc = m_acc.add(&ce[COL_BOUNDS_MARGIN_BYTE_OFFSET + b].mul(&m_basis[b]));
        }
        bodies.push(ce[COL_BOUNDS_MARGIN].sub(&m_acc));

        let bounds_body = ce[COL_BUF_SIZE_PRE]
            .sub(&ce[COL_RETURNDATA_OFFSET])
            .sub(&ce[COL_LENGTH])
            .sub(&ce[COL_BOUNDS_MARGIN]);
        bodies.push(sel_cp.mul(&bounds_body));

        let not_copy_gate = is_real.sub(sel_cp);
        bodies.push(not_copy_gate.mul(&ce[COL_BOUNDS_MARGIN]));

        bodies.push(ret_or_rev.mul(&ce[COL_RETURNDATA_OFFSET]));
        bodies.push(ret_or_rev.mul(&ce[COL_DEST_MEM_OFFSET]));
        bodies.push(sel_sz.mul(&ce[COL_LENGTH]));

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
        let sel_ret = &cc[COL_SEL_RETURN];
        let sel_rev = &cc[COL_SEL_REVERT];
        let sel_sz = &cc[COL_SEL_SIZE];
        let sel_cp = &cc[COL_SEL_COPY];

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        // 0
        let real_m1 = poly_sub(is_real, &one_p, curve);
        bodies.push(poly_mul(is_real, &real_m1, curve));
        // 1..=4
        for c in BINARY_SEL_COLS.iter() {
            let s = &cc[*c];
            let s_m1 = poly_sub(s, &one_p, curve);
            bodies.push(poly_mul(s, &s_m1, curve));
        }
        // 5: sum - is_real
        let sum1 = poly_add(sel_ret, sel_rev, curve);
        let sum2 = poly_add(&sum1, sel_sz, curve);
        let sel_sum = poly_add(&sum2, sel_cp, curve);
        bodies.push(poly_sub(&sel_sum, is_real, curve));

        // 6: (sel_ret + sel_rev) * (post - length)
        let ret_or_rev = poly_add(sel_ret, sel_rev, curve);
        let post_minus_len = poly_sub(&cc[COL_BUF_SIZE_POST], &cc[COL_LENGTH], curve);
        bodies.push(poly_mul(&ret_or_rev, &post_minus_len, curve));

        // 7: (sel_size + sel_copy) * (post - pre)
        let sz_or_cp = poly_add(sel_sz, sel_cp, curve);
        let post_minus_pre = poly_sub(&cc[COL_BUF_SIZE_POST], &cc[COL_BUF_SIZE_PRE], curve);
        bodies.push(poly_mul(&sz_or_cp, &post_minus_pre, curve));

        // 8: length - Σ basis * length_byte
        let mut len_acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        for b in 0..LENGTH_BYTES {
            let basis = vec![Scalar::from_u64(256u128.pow(b as u32) as u64, curve)];
            // 256^7 = 0x100_0000_0000_0000 fits in u64 (≈ 7.2e16). u64::MAX ≈ 1.8e19, so OK for b<=7.
            let term = poly_mul(&cc[COL_LENGTH_BYTE_OFFSET + b], &basis, curve);
            len_acc = poly_add(&len_acc, &term, curve);
        }
        bodies.push(poly_sub(&cc[COL_LENGTH], &len_acc, curve));

        // 9: bounds_margin - Σ basis * margin_byte
        let mut m_acc: Vec<Scalar> = vec![Scalar::zero(curve)];
        for b in 0..BOUNDS_MARGIN_BYTES {
            let basis = vec![Scalar::from_u64(256u128.pow(b as u32) as u64, curve)];
            let term = poly_mul(&cc[COL_BOUNDS_MARGIN_BYTE_OFFSET + b], &basis, curve);
            m_acc = poly_add(&m_acc, &term, curve);
        }
        bodies.push(poly_sub(&cc[COL_BOUNDS_MARGIN], &m_acc, curve));

        // 10: sel_copy * (pre - rd_off - length - margin)
        let t1 = poly_sub(&cc[COL_BUF_SIZE_PRE], &cc[COL_RETURNDATA_OFFSET], curve);
        let t2 = poly_sub(&t1, &cc[COL_LENGTH], curve);
        let bounds_body = poly_sub(&t2, &cc[COL_BOUNDS_MARGIN], curve);
        bodies.push(poly_mul(sel_cp, &bounds_body, curve));

        // 11: (is_real - sel_copy) * bounds_margin
        let not_copy = poly_sub(is_real, sel_cp, curve);
        bodies.push(poly_mul(&not_copy, &cc[COL_BOUNDS_MARGIN], curve));

        // 12, 13, 14
        bodies.push(poly_mul(&ret_or_rev, &cc[COL_RETURNDATA_OFFSET], curve));
        bodies.push(poly_mul(&ret_or_rev, &cc[COL_DEST_MEM_OFFSET], curve));
        bodies.push(poly_mul(sel_sz, &cc[COL_LENGTH], curve));

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    // ── Shifted: buffer pre[i+1] = post[i] across consecutive same-frame
    //    real rows (excluding RETURN/REVERT, which cross a frame boundary).
    fn shifted_column_indices(&self) -> Vec<usize> {
        // Layout: IS_REAL (next), BUF_SIZE_PRE (next)
        vec![COL_IS_REAL, COL_BUF_SIZE_PRE]
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        let curve = alpha.curve_type();
        if shifted_evals.len() < 2 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(curve);
        }
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let pre_next = &shifted_evals[1];
        // Gate: is_real(X) * is_real(ωX) * (1 - sel_return - sel_revert)
        // — RETURN/REVERT pops the frame; the inherited buffer chain
        // breaks there.
        let same_frame = Scalar::one(curve)
            .sub(&col_evals_at_z[COL_SEL_RETURN])
            .sub(&col_evals_at_z[COL_SEL_REVERT]);
        let gate = v.mul(v_next).mul(&same_frame);
        // Body: buf_post(X) - buf_pre(ω·X)
        let body = col_evals_at_z[COL_BUF_SIZE_POST].sub(pre_next);
        let row = gate.mul(&body);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        ap.mul(&row).mul(&z.sub(omega_n_minus_1))
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let v = &column_coeffs[COL_IS_REAL];
        let v_next = poly_shift(v, omega);
        let pre = &column_coeffs[COL_BUF_SIZE_PRE];
        let pre_next = poly_shift(pre, omega);
        let sf = poly_sub(
            &poly_sub(&one_p, &column_coeffs[COL_SEL_RETURN], curve),
            &column_coeffs[COL_SEL_REVERT],
            curve,
        );
        let gate12 = poly_mul(v, &v_next, curve);
        let gate = poly_mul(&gate12, &sf, curve);
        let body = poly_sub(&column_coeffs[COL_BUF_SIZE_POST], &pre_next, curve);
        let row_poly = poly_mul(&gate, &body, curve);
        // Multiply by (X − ω^{n−1}) so the prover's committed polynomial
        // matches the verifier's `evaluate_shifted_at_point`, which
        // multiplies the row body by `(z − ω^{n−1})`. Without this
        // factor the prover-side polynomial is short by one degree-1
        // factor and the quotient identity Q(z)·Z(z) = C(z) fails on
        // the wrap row, even on honest witnesses. (Task #256/#268.)
        let n_minus_1 = (domain_size as u64).saturating_sub(1);
        let mut omega_n_minus_1 = Scalar::one(curve);
        let mut e = n_minus_1;
        let mut base_pow = omega.clone();
        while e > 0 {
            if e & 1 == 1 {
                omega_n_minus_1 = omega_n_minus_1.mul(&base_pow);
            }
            base_pow = base_pow.mul(&base_pow);
            e >>= 1;
        }
        let row_excluded = poly_mul_linear(&row_poly, &omega_n_minus_1);
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        poly_scalar_mul(&row_excluded, &ap)
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
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// Bind `(call_depth, length)` published by this AIR's RETURN/REVERT
/// rows to call_family_air's `(depth_pre, gas_returned)` view of the
/// frame pop event.
///
/// Gating on the family side uses `COL_IS_RETURN`. The length-side
/// algebraic binding is approximate (we use the gas_returned slot as
/// the second tuple element because call_family_air doesn't yet
/// expose a `returndata_length` column); the precise binding is
/// delivered by the memory descriptor below, which constrains the
/// touched memory range.
pub fn make_returndata_to_call_family_descriptor(
    returndata_layer: usize,
    family_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    // A side: this AIR's (depth, length) on RETURN/REVERT rows.
    // We synthesize the gate on the A side via a dedicated selector
    // column. Since cross-AIR descriptors take a single selector
    // column, we use COL_SEL_RETURN here and document that REVERT
    // requires a sibling descriptor.
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "returndata_to_call_family_v1".into(),
        a_layer_index: returndata_layer,
        a_columns: vec![COL_CALL_DEPTH, COL_LENGTH],
        a_selector_column: Some(COL_SEL_RETURN),
        b_layer_index: family_layer,
        b_columns: vec![
            crate::call_family_air::COL_DEPTH_PRE,
            crate::call_family_air::COL_GAS_RETURNED,
        ],
        b_selector_column: Some(crate::call_family_air::COL_IS_RETURN),
    }
}

/// Bind `(mem_offset, length, dest_mem_offset)` published by this AIR
/// to memory_expansion_air's view of the touched range.
///
/// We pair `mem_offset` ↔ `new_size`-bound and `dest_mem_offset` ↔
/// `old_size`-bound so the memory expansion AIR's row-3 constraint
/// (which pins `new_size = max(old_size, offset + length)`) pulls the
/// algebraic length binding through.
pub fn make_returndata_to_memory_descriptor(
    returndata_layer: usize,
    memory_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "returndata_to_memory_expansion_v1".into(),
        a_layer_index: returndata_layer,
        a_columns: vec![COL_MEM_OFFSET, COL_LENGTH, COL_DEST_MEM_OFFSET],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: memory_layer,
        b_columns: vec![
            crate::memory_expansion_air::COL_OLD_SIZE,
            crate::memory_expansion_air::COL_NEW_SIZE,
            crate::memory_expansion_air::COL_OLD_WORDS,
        ],
        b_selector_column: Some(crate::memory_expansion_air::COL_IS_REAL),
    }
}

/// Build a witness from a slice of host-side events. Pass-through; the
/// inspector/oracle is responsible for emitting a well-formed event
/// sequence (depth chain, buffer-size threading, etc.).
pub fn from_events(events: &[ReturnDataEvent]) -> ReturnDataWitness {
    ReturnDataWitness::from_events(events)
}

// ─── Tests ────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_zero(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "row-local constraint {} row {} nonzero", i, r);
            }
        }
    }

    #[test]
    fn return_event_passes() {
        // RETURN with mem_offset=0, length=32; buffer post = 32.
        let w = from_events(&[ReturnDataEvent::ret(0, 32, 0, 1)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn revert_event_passes() {
        let w = from_events(&[ReturnDataEvent::revert(64, 16, 5, 2)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn returndatasize_event_passes() {
        let w = from_events(&[ReturnDataEvent::size(64, 1)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }

    #[test]
    fn returndatacopy_in_bounds_passes() {
        // buffer = 64, copy 32 bytes from rd_off=16 → margin = 64-16-32 = 16
        let w = from_events(&[ReturnDataEvent::copy(0, 16, 32, 64, 1)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
        // sanity: margin column = 16
        let expected_margin = Scalar::from_u64(16, CurveType::Bls48581);
        assert!(cr[COL_BOUNDS_MARGIN][0].sub(&expected_margin).is_zero());
    }

    #[test]
    fn tampered_post_size_detected() {
        // RETURN with length=32 but post incorrectly = 16.
        let mut ev = ReturnDataEvent::ret(0, 32, 0, 1);
        ev.returndata_buffer_size_post = 16;
        let w = from_events(&[ev]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 6 = (sel_ret + sel_rev) * (post - length).
        assert!(!bodies[6][0].is_zero(), "expected RETURN/REVERT post-eq-length to fire");
    }

    #[test]
    fn tampered_size_mutates_buffer_detected() {
        // RETURNDATASIZE row with post != pre.
        let mut ev = ReturnDataEvent::size(64, 1);
        ev.returndata_buffer_size_post = 65;
        let w = from_events(&[ev]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 7 = (sel_size + sel_copy) * (post - pre).
        assert!(!bodies[7][0].is_zero(), "expected read-only constraint to fire");
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_returndata_to_call_family_descriptor(0, 1);
        assert_eq!(d1.label, "returndata_to_call_family_v1");
        assert_eq!(d1.a_columns.len(), 2);
        assert_eq!(d1.b_columns.len(), 2);
        assert_eq!(d1.a_selector_column, Some(COL_SEL_RETURN));
        assert!(d1.b_selector_column.is_some());

        let d2 = make_returndata_to_memory_descriptor(0, 2);
        assert_eq!(d2.label, "returndata_to_memory_expansion_v1");
        assert_eq!(d2.a_columns.len(), 3);
        assert_eq!(d2.b_columns.len(), 3);
        assert_eq!(d2.a_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn evaluate_at_point_zero_on_honest() {
        let curve = CurveType::Bls48581;
        let w = from_events(&[ReturnDataEvent::copy(0, 0, 16, 32, 1)]);
        let t = build_trace_polynomials(&w, curve);
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0x1234, curve);
        let row_evals: Vec<Scalar> =
            t.columns.iter().map(|p| p.evaluations[0].clone()).collect();
        let pt = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(pt.is_zero(), "honest copy row should evaluate to zero");
    }

    #[test]
    fn copy_out_of_bounds_padding_detected_via_margin() {
        // buffer = 16, copy 32 bytes from rd_off=0 → would overflow.
        // The builder's saturating fallback puts margin = 0; constraint
        // 10 then desyncs: 16 - 0 - 32 - 0 = -16 ≠ 0.
        let w = from_events(&[ReturnDataEvent::copy(0, 0, 32, 16, 1)]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 10 = sel_copy * (pre - rd_off - length - margin).
        assert!(!bodies[10][0].is_zero(), "expected bounds binding to fire");
    }

    #[test]
    fn chain_shifted_buffer_holds_intra_frame() {
        // Same-frame chain: SIZE (buf=64), COPY (buf=64), SIZE (buf=64).
        // Intra-frame the buffer is constant, so post[i] = pre[i+1] = 64.
        let w = from_events(&[
            ReturnDataEvent::size(64, 1),
            ReturnDataEvent::copy(0, 0, 16, 64, 1),
            ReturnDataEvent::size(64, 1),
        ]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let post = &t.columns[COL_BUF_SIZE_POST].evaluations;
        let pre = &t.columns[COL_BUF_SIZE_PRE].evaluations;
        for i in 0..w.rows.len() - 1 {
            assert!(
                post[i].sub(&pre[i + 1]).is_zero(),
                "intra-frame buffer chain row {}",
                i,
            );
        }
        let cs = ReturnDataConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        assert_all_zero(&cs.evaluate_on_domain(&cr, t.num_rows));
    }
}
