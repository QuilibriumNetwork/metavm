//! Byte-memory AIR — Phase A1b-mem step 1.
//!
//! Provides a dedicated AIR whose rows are byte-granularity memory
//! accesses (`(addr, val, ts, rw, source_row)`). Two parallel views per
//! row are committed:
//!   - **Unsorted** view: accesses in trace-emission order (matches the
//!     order produced by `metavm_evm::sha3_mem_binding::build_byte_access_trace`).
//!     Cross-AIR LogUp pulls into this view from EVM main MSTORE/MSTORE8
//!     rows (writes) and from SHA3 input gadget INPUT_BYTE columns
//!     (reads) — see step 1c.
//!   - **Sorted** view: same multiset, re-sorted by `(addr, ts)`. The
//!     read-consistency constraint (step 1b) operates here:
//!       * `addr_s[r+1] == addr_s[r]` AND `rw_s[r+1] == 0` (Read) ⇒
//!         `val_s[r+1] == val_s[r]`
//!       * `addr_s[r+1] != addr_s[r]` (or first row) AND `rw_s == 0`
//!         (Read) ⇒ `val_s == 0` (default zero memory)
//!
//! A grand-product permutation argument (step 1b) ensures the sorted
//! view is a multiset permutation of the unsorted view, so the
//! constraints over the sorted view also hold over the original
//! accesses.
//!
//! **Step 1a (this commit)**: column layout, witness type, trace
//! builder that populates both views and asserts host-side
//! read-consistency. No algebraic constraints yet — `num_constraints
//! = 0`. The constraint system is wired into the existing
//! `VmConstraintSystem` trait so the AIR can already be plugged into
//! `joint_prove` and validated to produce a (constraint-trivial) proof
//! with the right column shape; future steps add the per-row checks
//! and the cross-AIR linkages.

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

fn scalar_pow(base: &Scalar, exp: u64) -> Scalar {
    let mut result = Scalar::one(base.curve_type());
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&b);
        }
        b = b.mul(&b);
        e >>= 1;
    }
    result
}

// ─── Column layout ────────────────────────────────────────────────────
//
// Unsorted view (the trace-emission order):
pub const COL_ADDR: usize = 0;
pub const COL_VAL: usize = 1;
pub const COL_TS: usize = 2;
pub const COL_RW: usize = 3;
pub const COL_SOURCE_ROW: usize = 4;
pub const COL_IS_REAL: usize = 5;

// Sorted view (re-sorted by (addr, ts)):
pub const COL_ADDR_S: usize = 6;
pub const COL_VAL_S: usize = 7;
pub const COL_TS_S: usize = 8;
pub const COL_RW_S: usize = 9;
pub const COL_SOURCE_ROW_S: usize = 10;
pub const COL_IS_REAL_S: usize = 11;

// Read-consistency auxiliary columns (populated by the trace builder
// from the sorted view; consumed by phase-2 cross-row constraints).
//
// Convention (matches `crate::permutation` after the 2026-05-12 fix):
//   - `is_same_addr_s[i]` = 1 iff `addr_s[i] == addr_s[i-1]` for i ≥ 1
//                          and 0 at i = 0 (no previous row).
//   - `inv_diff_addr_s[i]` = (addr_s[i] - addr_s[i-1])^{-1} when the
//                          addrs differ; 0 otherwise (including i = 0).
//
// The cross-row constraints reference *shifted* (ω·z) values of these
// columns so that at constraint point z, `is_same_addr_s(ω·z)` and
// `inv_diff_addr_s(ω·z)` describe the (z → ω·z) transition — i.e. the
// pair (addr_s[r], addr_s[r+1]) where r corresponds to z.
pub const COL_IS_SAME_ADDR_S: usize = 12;
pub const COL_INV_DIFF_ADDR_S: usize = 13;

/// Boundary selector for row 0 (task #147). Set to 1 on row 0 by the
/// honest witness builder; zero elsewhere. Pins the sorted-view row-0
/// boundary that the shifted constraints (constraint 8, read-first-
/// zero) cannot reach — because the shifted body fires on the
/// transition row r → r+1 and references the *next* row's data, row 0
/// is never the "next row" of any transition (the wrap r=n-1 → 0 is
/// excluded). Soft binary + implies-real-s constraints keep the
/// selector well-formed; a malicious prover could still set
/// `IS_FIRST = 0` on row 0, so a downstream consumer (or `is_first`
/// hard pin via cross-AIR linkage) is required for full soundness.
pub const COL_IS_FIRST: usize = 14;

pub const NUM_COLUMNS: usize = 15;

/// Row-local constraints (phase 1 + phase 2 binary + row-0 boundary):
///   0. `is_real * (is_real - 1) = 0`
///   1. `is_real_s * (is_real_s - 1) = 0`
///   2. `rw * (rw - 1) = 0`           (binary read/write flag, unsorted view)
///   3. `rw_s * (rw_s - 1) = 0`       (binary read/write flag, sorted view)
///   4. `is_same_addr_s * (is_same_addr_s - 1) = 0` (binary)
///   5. `is_first * (is_first - 1) = 0`               (binary)
///   6. `is_first * (1 - is_real_s) = 0`              (is_first ⇒ is_real_s)
///   7. `is_first * is_same_addr_s = 0`               — row 0 of the
///      sorted view has no prior row, so `is_same_addr_s[0] = 0`.
///   8. `is_first * is_real_s * (1 - rw_s) * val_s = 0` (task #147) —
///      **row-0 read-first-zero boundary**: if the sorted view's row 0
///      is a real Read, its `val_s` must be 0 (default zero memory).
///      The shifted constraint 8 (`read-first-zero`) only fires on
///      transitions r → r+1 where the *next* row is the first of a
///      new addr group — it never reaches row 0 itself (the wrap
///      `r = n-1 → 0` is excluded by the `(z - ω^{n-1})` factor).
pub const NUM_ROW_CONSTRAINTS: usize = 9;

/// Cross-row (shifted) constraints (phase 2):
///   5. `is_real_s(ω·z) * is_same_addr_s(ω·z) * (addr_s(ω·z) - addr_s(z)) = 0`
///        — when next row is real AND same-addr-as-prev, addrs match.
///   6. `is_real_s(ω·z) * (1 - is_same_addr_s(ω·z)) *
///         (1 - (addr_s(ω·z) - addr_s(z)) * inv_diff_addr_s(ω·z)) = 0`
///        — when next row is real AND new addr, the inverse witness pins
///        the diff to be non-zero (locally enforces sort group boundary).
///   7. `is_real_s(ω·z) * is_same_addr_s(ω·z) * (1 - rw_s(ω·z)) *
///         (val_s(ω·z) - val_s(z)) = 0`
///        — within a same-addr group, a Read sees the prior row's val.
///   8. `is_real_s(ω·z) * (1 - is_same_addr_s(ω·z)) * (1 - rw_s(ω·z)) *
///         val_s(ω·z) = 0`
///        — first access at a new addr that is a Read must return 0
///        (default zero memory).
///
/// All shifted constraints are multiplied by `(z - ω^{n-1})` for wrap
/// exclusion (same convention as `validator_extract` / EVM main perm).
///
/// Phase 3 (grand-product multiset permutation between sorted ↔
/// unsorted views) needs transcript challenges and is deferred.
pub const NUM_SHIFTED: usize = 4;

// ─── Witness ──────────────────────────────────────────────────────────

/// One byte access row, mirroring `metavm_evm::sha3_mem_binding::ByteAccess`
/// but expressed in the integer types this crate uses (avoids a
/// dependency on the EVM crate from inside zkp).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteMemoryAccess {
    pub addr: u64,
    pub val: u8,
    pub ts: u64,
    /// 0 = Read, 1 = Write.
    pub rw: u64,
    pub source_row: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ByteMemoryWitness {
    pub accesses: Vec<ByteMemoryAccess>,
}

impl ByteMemoryWitness {
    pub fn from_accesses(accesses: Vec<ByteMemoryAccess>) -> Self {
        Self { accesses }
    }

    /// Sort a copy of the accesses by `(addr, ts)`.
    pub fn sorted(&self) -> Vec<ByteMemoryAccess> {
        let mut s = self.accesses.clone();
        s.sort_by_key(|a| (a.addr, a.ts));
        s
    }

    /// Host-side read-consistency check, equivalent to the algebraic
    /// constraint step 1b will encode over the sorted view. Returns
    /// `Ok(())` on success or `Err(diagnostic)` on first violation.
    pub fn verify_read_consistency(&self) -> Result<(), String> {
        let sorted = self.sorted();
        let mut prev: Option<ByteMemoryAccess> = None;
        for cur in &sorted {
            let same_addr = prev.map_or(false, |p| p.addr == cur.addr);
            if cur.rw == 0 {
                let expected = if same_addr {
                    prev.unwrap().val
                } else {
                    0u8
                };
                if cur.val != expected {
                    return Err(format!(
                        "read at addr={} ts={} returned val={} but \
                         expected {} (same_addr={}, prev_val={:?})",
                        cur.addr,
                        cur.ts,
                        cur.val,
                        expected,
                        same_addr,
                        prev.map(|p| p.val),
                    ));
                }
            }
            prev = Some(*cur);
        }
        Ok(())
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ByteMemoryWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.accesses.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    let sorted = witness.sorted();

    for i in 0..num_rows {
        let u = &witness.accesses[i];
        columns[COL_ADDR][i] = Scalar::from_u64(u.addr, curve);
        columns[COL_VAL][i] = Scalar::from_u64(u.val as u64, curve);
        columns[COL_TS][i] = Scalar::from_u64(u.ts, curve);
        columns[COL_RW][i] = Scalar::from_u64(u.rw, curve);
        columns[COL_SOURCE_ROW][i] = Scalar::from_u64(u.source_row, curve);
        columns[COL_IS_REAL][i] = one.clone();

        let s = &sorted[i];
        columns[COL_ADDR_S][i] = Scalar::from_u64(s.addr, curve);
        columns[COL_VAL_S][i] = Scalar::from_u64(s.val as u64, curve);
        columns[COL_TS_S][i] = Scalar::from_u64(s.ts, curve);
        columns[COL_RW_S][i] = Scalar::from_u64(s.rw, curve);
        columns[COL_SOURCE_ROW_S][i] = Scalar::from_u64(s.source_row, curve);
        columns[COL_IS_REAL_S][i] = one.clone();

        // Task #147: pin the row-0 boundary selector.
        if i == 0 {
            columns[COL_IS_FIRST][i] = one.clone();
        }

        // is_same_addr_s[i] and inv_diff_addr_s[i] describe the
        // (i-1 → i) transition (backward-looking by index, matching
        // permutation.rs convention). At i=0 both are zero — the gate
        // `(1 - is_same_addr_s) * (1 - diff*inv)` would fail there,
        // but the corresponding constraint applies at z = row i-1 and
        // references is_same_addr_s(ω·z) = is_same_addr_s[i], so the
        // first row's columns are not actually consumed by any cross-
        // row constraint at row -1 (which doesn't exist). The wrap
        // exclusion `(z - ω^{n-1})` zeroes out the row n-1 → row 0
        // wrap that would otherwise consume them.
        if i >= 1 {
            let prev_addr = sorted[i - 1].addr;
            if s.addr == prev_addr {
                columns[COL_IS_SAME_ADDR_S][i] = one.clone();
                // inv_diff stays zero.
            } else {
                // addr is strictly different. Compute the modular
                // inverse of (s.addr - prev_addr) in the field. The
                // raw u64 subtraction wraps if s.addr < prev_addr,
                // but the field reduction makes that consistent: the
                // inverse just lives in F_r.
                let diff_u64 = s.addr.wrapping_sub(prev_addr);
                let diff_scalar = if s.addr >= prev_addr {
                    Scalar::from_u64(diff_u64, curve)
                } else {
                    // s.addr < prev_addr — diff is negative. Build it
                    // as -(prev_addr - s.addr) in the field.
                    let pos = Scalar::from_u64(prev_addr - s.addr, curve);
                    Scalar::zero(curve).sub(&pos)
                };
                columns[COL_INV_DIFF_ADDR_S][i] = diff_scalar.inverse();
                // is_same_addr_s stays zero.
            }
        }
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

// ─── Constraint system (step 1a stub) ─────────────────────────────────

pub struct ByteMemoryConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ByteMemoryConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for ByteMemoryConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_real_s_binary".into(),
            "rw_binary".into(),
            "rw_s_binary".into(),
            "is_same_addr_s_binary".into(),
            "is_first_binary".into(),
            "is_first_implies_real_s".into(),
            "is_first_blocks_same_addr".into(),
            "is_first_read_first_zero_boundary".into(),
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
        let n = columns[0].len();

        // Helper: x * (x - 1).
        let bin = |col_idx: usize| -> Vec<Scalar> {
            let mut out = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[col_idx][r];
                out[r] = v.mul(&v.sub(&one));
            }
            out
        };

        // Row-0 boundary bodies (task #147).
        let mut c6 = vec![Scalar::zero(curve); n]; // is_first ⇒ is_real_s
        let mut c7 = vec![Scalar::zero(curve); n]; // is_first · is_same_addr_s = 0
        let mut c8 = vec![Scalar::zero(curve); n]; // is_first · is_real_s · (1 - rw_s) · val_s = 0
        for r in 0..n {
            let f = &columns[COL_IS_FIRST][r];
            let is_real_s = &columns[COL_IS_REAL_S][r];
            let isa = &columns[COL_IS_SAME_ADDR_S][r];
            let rw_s = &columns[COL_RW_S][r];
            let val_s = &columns[COL_VAL_S][r];
            c6[r] = f.mul(&one.sub(is_real_s));
            c7[r] = f.mul(isa);
            c8[r] = f.mul(is_real_s).mul(&one.sub(rw_s)).mul(val_s);
        }

        vec![
            bin(COL_IS_REAL),
            bin(COL_IS_REAL_S),
            bin(COL_RW),
            bin(COL_RW_S),
            bin(COL_IS_SAME_ADDR_S),
            bin(COL_IS_FIRST),
            c6,
            c7,
            c8,
        ]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let bin = |col_idx: usize| -> Scalar {
            let v = &col_evals[col_idx];
            v.mul(&v.sub(&one))
        };

        let f = &col_evals[COL_IS_FIRST];
        let is_real_s = &col_evals[COL_IS_REAL_S];
        let isa = &col_evals[COL_IS_SAME_ADDR_S];
        let rw_s = &col_evals[COL_RW_S];
        let val_s = &col_evals[COL_VAL_S];
        let c6 = f.mul(&one.sub(is_real_s));
        let c7 = f.mul(isa);
        let c8 = f.mul(is_real_s).mul(&one.sub(rw_s)).mul(val_s);

        // Combined: Σ α^i · bin_i(z).
        let mut acc = bin(COL_IS_REAL);
        let mut alpha_pow = alpha.clone();
        acc = acc.add(&alpha_pow.mul(&bin(COL_IS_REAL_S)));
        alpha_pow = alpha_pow.mul(alpha);
        acc = acc.add(&alpha_pow.mul(&bin(COL_RW)));
        alpha_pow = alpha_pow.mul(alpha);
        acc = acc.add(&alpha_pow.mul(&bin(COL_RW_S)));
        alpha_pow = alpha_pow.mul(alpha);
        acc = acc.add(&alpha_pow.mul(&bin(COL_IS_SAME_ADDR_S)));
        alpha_pow = alpha_pow.mul(alpha);
        acc = acc.add(&alpha_pow.mul(&bin(COL_IS_FIRST)));
        alpha_pow = alpha_pow.mul(alpha);
        acc = acc.add(&alpha_pow.mul(&c6));
        alpha_pow = alpha_pow.mul(alpha);
        acc = acc.add(&alpha_pow.mul(&c7));
        alpha_pow = alpha_pow.mul(alpha);
        acc = acc.add(&alpha_pow.mul(&c8));
        acc
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let bin = |col_idx: usize| -> Vec<Scalar> {
            let v = &col_coeffs[col_idx];
            let v_m1 = poly_sub(v, &one_poly, curve);
            poly_mul(v, &v_m1, curve)
        };

        let f = &col_coeffs[COL_IS_FIRST];
        let is_real_s = &col_coeffs[COL_IS_REAL_S];
        let isa = &col_coeffs[COL_IS_SAME_ADDR_S];
        let rw_s = &col_coeffs[COL_RW_S];
        let val_s = &col_coeffs[COL_VAL_S];
        let one_minus_real_s = poly_sub(&one_poly, is_real_s, curve);
        let one_minus_rw_s = poly_sub(&one_poly, rw_s, curve);
        let c6 = poly_mul(f, &one_minus_real_s, curve);
        let c7 = poly_mul(f, isa, curve);
        let f_real = poly_mul(f, is_real_s, curve);
        let f_real_read = poly_mul(&f_real, &one_minus_rw_s, curve);
        let c8 = poly_mul(&f_real_read, val_s, curve);

        let mut acc = bin(COL_IS_REAL);
        let mut alpha_pow = alpha.clone();
        acc = poly_add(&acc, &poly_scalar_mul(&bin(COL_IS_REAL_S), &alpha_pow), curve);
        alpha_pow = alpha_pow.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&bin(COL_RW), &alpha_pow), curve);
        alpha_pow = alpha_pow.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&bin(COL_RW_S), &alpha_pow), curve);
        alpha_pow = alpha_pow.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&bin(COL_IS_SAME_ADDR_S), &alpha_pow), curve);
        alpha_pow = alpha_pow.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&bin(COL_IS_FIRST), &alpha_pow), curve);
        alpha_pow = alpha_pow.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&c6, &alpha_pow), curve);
        alpha_pow = alpha_pow.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&c7, &alpha_pow), curve);
        alpha_pow = alpha_pow.mul(alpha);
        acc = poly_add(&acc, &poly_scalar_mul(&c8, &alpha_pow), curve);
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_IS_REAL_S]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // Step 1a: no lookup declarations. Step 1b will add an 8-bit
        // range check on val/val_s and binary range checks on rw/rw_s.
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    // ── Cross-row (shifted) constraints (phase 2) ────────────────────

    fn shifted_column_indices(&self) -> Vec<usize> {
        // We reference these columns at ω·z:
        vec![
            COL_ADDR_S,
            COL_VAL_S,
            COL_RW_S,
            COL_IS_REAL_S,
            COL_IS_SAME_ADDR_S,
            COL_INV_DIFF_ADDR_S,
        ]
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
        if shifted_evals.len() < 6 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        // Map shifted_evals indices: same order as shifted_column_indices().
        let addr_s_wz = &shifted_evals[0];
        let val_s_wz = &shifted_evals[1];
        let rw_s_wz = &shifted_evals[2];
        let is_real_s_wz = &shifted_evals[3];
        let is_same_addr_s_wz = &shifted_evals[4];
        let inv_diff_addr_s_wz = &shifted_evals[5];

        let addr_s_z = &col_evals_at_z[COL_ADDR_S];
        let val_s_z = &col_evals_at_z[COL_VAL_S];

        let exclusion = z.sub(omega_n_minus_1);
        let addr_diff = addr_s_wz.sub(addr_s_z);
        let one_minus_rw_wz = one.sub(rw_s_wz);
        let one_minus_isa_wz = one.sub(is_same_addr_s_wz);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut result = Scalar::zero(curve);

        // Constraint 5 (sort-same):
        //   is_real_s(ω·z) * is_same_addr_s(ω·z) * (addr_s(ω·z) - addr_s(z)) = 0
        let body0 = is_real_s_wz.mul(is_same_addr_s_wz).mul(&addr_diff);
        result = result.add(&ap.mul(&body0).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 6 (sort-diff):
        //   is_real_s(ω·z) * (1 - is_same_addr_s(ω·z)) *
        //     (1 - addr_diff * inv_diff_addr_s(ω·z)) = 0
        let inv_check = one.sub(&addr_diff.mul(inv_diff_addr_s_wz));
        let body1 = is_real_s_wz.mul(&one_minus_isa_wz).mul(&inv_check);
        result = result.add(&ap.mul(&body1).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 7 (read-same):
        //   is_real_s(ω·z) * is_same_addr_s(ω·z) * (1 - rw_s(ω·z)) *
        //     (val_s(ω·z) - val_s(z)) = 0
        let val_diff = val_s_wz.sub(val_s_z);
        let body2 = is_real_s_wz.mul(is_same_addr_s_wz).mul(&one_minus_rw_wz).mul(&val_diff);
        result = result.add(&ap.mul(&body2).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 8 (read-first-zero):
        //   is_real_s(ω·z) * (1 - is_same_addr_s(ω·z)) * (1 - rw_s(ω·z)) *
        //     val_s(ω·z) = 0
        let body3 = is_real_s_wz.mul(&one_minus_isa_wz).mul(&one_minus_rw_wz).mul(val_s_wz);
        result = result.add(&ap.mul(&body3).mul(&exclusion));

        result
    }

    fn build_shifted_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let addr_s = &col_coeffs[COL_ADDR_S];
        let val_s = &col_coeffs[COL_VAL_S];
        let rw_s = &col_coeffs[COL_RW_S];
        let is_real_s = &col_coeffs[COL_IS_REAL_S];
        let is_same_addr_s = &col_coeffs[COL_IS_SAME_ADDR_S];
        let inv_diff_addr_s = &col_coeffs[COL_INV_DIFF_ADDR_S];

        let addr_s_shift = poly_shift(addr_s, omega);
        let val_s_shift = poly_shift(val_s, omega);
        let rw_s_shift = poly_shift(rw_s, omega);
        let is_real_s_shift = poly_shift(is_real_s, omega);
        let is_same_addr_s_shift = poly_shift(is_same_addr_s, omega);
        let inv_diff_addr_s_shift = poly_shift(inv_diff_addr_s, omega);

        let addr_diff = poly_sub(&addr_s_shift, addr_s, curve);
        let one_minus_rw_shift = poly_sub(&one_poly, &rw_s_shift, curve);
        let one_minus_isa_shift = poly_sub(&one_poly, &is_same_addr_s_shift, curve);

        // Constraint 5 (sort-same):
        let body0 = poly_mul(
            &poly_mul(&is_real_s_shift, &is_same_addr_s_shift, curve),
            &addr_diff,
            curve,
        );
        // Constraint 6 (sort-diff):
        let diff_times_inv = poly_mul(&addr_diff, &inv_diff_addr_s_shift, curve);
        let inv_check = poly_sub(&one_poly, &diff_times_inv, curve);
        let body1 = poly_mul(
            &poly_mul(&is_real_s_shift, &one_minus_isa_shift, curve),
            &inv_check,
            curve,
        );
        // Constraint 7 (read-same):
        let val_diff = poly_sub(&val_s_shift, val_s, curve);
        let read_same_left = poly_mul(
            &poly_mul(&is_real_s_shift, &is_same_addr_s_shift, curve),
            &one_minus_rw_shift,
            curve,
        );
        let body2 = poly_mul(&read_same_left, &val_diff, curve);
        // Constraint 8 (read-first-zero):
        let first_left = poly_mul(
            &poly_mul(&is_real_s_shift, &one_minus_isa_shift, curve),
            &one_minus_rw_shift,
            curve,
        );
        let body3 = poly_mul(&first_left, &val_s_shift, curve);

        // Wrap exclusion (X - ω^{n-1}) applied to each body.
        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let ex0 = poly_mul_linear(&body0, &omega_n_minus_1);
        let ex1 = poly_mul_linear(&body1, &omega_n_minus_1);
        let ex2 = poly_mul_linear(&body2, &omega_n_minus_1);
        let ex3 = poly_mul_linear(&body3, &omega_n_minus_1);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let term0 = poly_scalar_mul(&ex0, &ap);
        ap = ap.mul(alpha);
        let term1 = poly_scalar_mul(&ex1, &ap);
        ap = ap.mul(alpha);
        let term2 = poly_scalar_mul(&ex2, &ap);
        ap = ap.mul(alpha);
        let term3 = poly_scalar_mul(&ex3, &ap);

        let s01 = poly_add(&term0, &term1, curve);
        let s23 = poly_add(&term2, &term3, curve);
        poly_add(&s01, &s23, curve)
    }
}

/// Phase 3 — multiset permutation between unsorted ↔ sorted views via
/// cross-AIR LogUp self-linkage.
///
/// The byte-memory AIR's sorted-view constraints (phase 2) prove the
/// sorted view satisfies sort grouping + read-consistency + zero-default
/// memory locally. Phase 3 ties the sorted view back to the unsorted
/// view as a multiset, closing the soundness gap where a malicious
/// prover could otherwise commit a "fake sorted view" with no
/// relationship to the actual access trace.
///
/// We use the existing [`crate::cross_air_logup`] machinery as a
/// **self-linkage**: both A-side and B-side reference the same byte-
/// memory AIR, but A projects the unsorted columns and B projects the
/// sorted columns. The joint γ challenge is derived from the joint
/// transcript across all AIRs in the proof, and the closure check
/// `Σ active_a / (γ - tuple_a) == Σ active_b / (γ - tuple_b)` enforces
/// multiset equality of the two views' real (`is_real == 1`) rows.
///
/// `layer_index` is the index of the byte-memory AIR in the
/// [`crate::layer_chain::LayerChainProof`]; both endpoints share it.
pub fn make_byte_memory_self_linkage_descriptor(
    layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "byte_memory_unsorted_eq_sorted_v1".into(),
        a_layer_index: layer_index,
        a_columns: vec![COL_ADDR, COL_VAL, COL_TS, COL_RW, COL_SOURCE_ROW],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: layer_index,
        b_columns: vec![COL_ADDR_S, COL_VAL_S, COL_TS_S, COL_RW_S, COL_SOURCE_ROW_S],
        b_selector_column: Some(COL_IS_REAL_S),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(addr: u64, val: u8, ts: u64, source_row: u64) -> ByteMemoryAccess {
        ByteMemoryAccess { addr, val, ts, rw: 1, source_row }
    }
    fn read(addr: u64, val: u8, ts: u64, source_row: u64) -> ByteMemoryAccess {
        ByteMemoryAccess { addr, val, ts, rw: 0, source_row }
    }

    #[test]
    fn empty_witness_pads_to_minimum_fft_size() {
        // nearest_power_of_two has a minimum of 16 (the smallest BLS48-581
        // cached FFT width).
        let w = ByteMemoryWitness::default();
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.num_rows, 0);
        assert_eq!(trace.padded_size, 16);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
    }

    #[test]
    fn sorted_view_orders_by_addr_then_ts() {
        // Trace order: write @ 5, write @ 1, read @ 5, read @ 1.
        let w = ByteMemoryWitness::from_accesses(vec![
            write(5, 0xab, 0, 0),
            write(1, 0xcd, 1, 1),
            read(5, 0xab, 2, 2),
            read(1, 0xcd, 3, 3),
        ]);
        let s = w.sorted();
        // Sorted by (addr, ts): (1,1), (1,3), (5,0), (5,2).
        assert_eq!((s[0].addr, s[0].ts, s[0].val, s[0].rw), (1, 1, 0xcd, 1));
        assert_eq!((s[1].addr, s[1].ts, s[1].val, s[1].rw), (1, 3, 0xcd, 0));
        assert_eq!((s[2].addr, s[2].ts, s[2].val, s[2].rw), (5, 0, 0xab, 1));
        assert_eq!((s[3].addr, s[3].ts, s[3].val, s[3].rw), (5, 2, 0xab, 0));
    }

    #[test]
    fn read_consistency_holds_for_simple_write_then_read() {
        let w = ByteMemoryWitness::from_accesses(vec![
            write(0, 0xab, 0, 0),
            read(0, 0xab, 1, 1),
        ]);
        w.verify_read_consistency().unwrap();
    }

    #[test]
    fn read_consistency_holds_for_default_zero_read() {
        let w = ByteMemoryWitness::from_accesses(vec![
            read(42, 0, 0, 0),
        ]);
        w.verify_read_consistency().unwrap();
    }

    #[test]
    fn read_consistency_detects_value_mismatch() {
        let w = ByteMemoryWitness::from_accesses(vec![
            write(0, 0xab, 0, 0),
            read(0, 0xff, 1, 1),
        ]);
        let err = w.verify_read_consistency().unwrap_err();
        assert!(err.contains("expected 171"), "got: {}", err);
    }

    #[test]
    fn read_consistency_detects_uninit_nonzero_read() {
        let w = ByteMemoryWitness::from_accesses(vec![
            read(99, 0x77, 0, 0),
        ]);
        let err = w.verify_read_consistency().unwrap_err();
        assert!(err.contains("expected 0"), "got: {}", err);
    }

    #[test]
    fn trace_columns_are_populated_in_both_views() {
        let w = ByteMemoryWitness::from_accesses(vec![
            write(5, 0xab, 0, 0),
            write(1, 0xcd, 1, 1),
        ]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // Unsorted: row 0 = (addr=5, val=0xab); row 1 = (addr=1, val=0xcd).
        assert_eq!(trace.columns[COL_ADDR].evaluations[0].to_u64(), 5);
        assert_eq!(trace.columns[COL_VAL].evaluations[0].to_u64(), 0xab);
        assert_eq!(trace.columns[COL_ADDR].evaluations[1].to_u64(), 1);
        // Sorted: row 0 = (addr=1, ts=1, val=0xcd); row 1 = (addr=5, ts=0, val=0xab).
        assert_eq!(trace.columns[COL_ADDR_S].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_TS_S].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_VAL_S].evaluations[0].to_u64(), 0xcd);
        assert_eq!(trace.columns[COL_ADDR_S].evaluations[1].to_u64(), 5);
        assert_eq!(trace.columns[COL_TS_S].evaluations[1].to_u64(), 0);
        assert_eq!(trace.columns[COL_VAL_S].evaluations[1].to_u64(), 0xab);
    }

    #[test]
    fn is_real_flags_one_on_real_rows_and_zero_on_padding() {
        let w = ByteMemoryWitness::from_accesses(vec![
            write(0, 0x11, 0, 0),
            write(1, 0x22, 1, 0),
            write(2, 0x33, 2, 0),
        ]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.padded_size, 16);
        for r in 0..3 {
            assert_eq!(trace.columns[COL_IS_REAL].evaluations[r].to_u64(), 1);
            assert_eq!(trace.columns[COL_IS_REAL_S].evaluations[r].to_u64(), 1);
        }
        // Padding rows 3..16.
        for r in 3..16 {
            assert_eq!(trace.columns[COL_IS_REAL].evaluations[r].to_u64(), 0);
            assert_eq!(trace.columns[COL_IS_REAL_S].evaluations[r].to_u64(), 0);
        }
    }

    fn run_evaluate_on_domain(witness: &ByteMemoryWitness) -> Vec<Vec<Scalar>> {
        let trace = build_trace_polynomials(witness, CurveType::Bls48581);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    #[test]
    fn binary_constraints_zero_on_honest_witness() {
        let w = ByteMemoryWitness::from_accesses(vec![
            write(0, 0xab, 0, 0),
            write(1, 0xcd, 1, 1),
            read(0, 0xab, 2, 2),
        ]);
        let results = run_evaluate_on_domain(&w);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn is_real_binary_constraint_fires_on_nonbinary_value() {
        let w = ByteMemoryWitness::from_accesses(vec![write(0, 0xab, 0, 0)]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // Tamper: set is_real on row 0 to 2 (not binary).
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 0 = is_real_binary should fail at row 0 (2 * 1 = 2, ≠ 0).
        assert!(!results[0][0].is_zero(), "is_real_binary should detect 2");
        // Other constraints unaffected at row 0.
        assert!(results[1][0].is_zero());
        assert!(results[2][0].is_zero());
        assert!(results[3][0].is_zero());
    }

    #[test]
    fn is_first_set_on_row_zero_only() {
        // Task #147: honest witness pins IS_FIRST = 1 on row 0 only.
        let w = ByteMemoryWitness::from_accesses(vec![
            write(0, 0xab, 0, 0),
            write(1, 0xcd, 1, 1),
        ]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let f = &trace.columns[COL_IS_FIRST].evaluations;
        assert_eq!(f[0].to_u64(), 1, "IS_FIRST[0] must be 1");
        for r in 1..trace.padded_size as usize {
            assert_eq!(f[r].to_u64(), 0, "IS_FIRST[{}] must be 0", r);
        }
    }

    #[test]
    fn row_zero_read_first_zero_boundary_fires_on_tampered_val_s() {
        // Task #147: row-0 of the sorted view is a real Read; if its
        // val_s is non-zero, constraint 8 (is_first_read_first_zero_
        // boundary) must fire. Construct a single-row witness with a
        // tampered val_s.
        let w = ByteMemoryWitness::from_accesses(vec![
            read(42, 0, 0, 0),  // honest: read of default-zero memory.
        ]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper: change sorted val_s on row 0 to 0xff (nonzero) — a
        // read of uninitialised memory must return 0.
        cols[COL_VAL_S][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        // Constraint 8 = is_first_read_first_zero_boundary must fire
        // at row 0.
        assert!(
            !results[8][0].is_zero(),
            "row-0 read-first-zero boundary must fire on tampered val_s[0]",
        );
        // Constraint 8 zero on rows >=1 (IS_FIRST=0 there).
        for r in 1..trace.padded_size as usize {
            assert!(
                results[8][r].is_zero(),
                "c8 must be zero on row {} (IS_FIRST=0)",
                r,
            );
        }
    }

    #[test]
    fn row_zero_boundary_zero_on_honest_witness() {
        // Task #147: row-0 boundary constraints (5..8) all zero on
        // honest witnesses across several patterns.
        let cases = vec![
            ByteMemoryWitness::from_accesses(vec![
                write(0, 0xab, 0, 0),
                read(0, 0xab, 1, 1),
            ]),
            ByteMemoryWitness::from_accesses(vec![
                read(42, 0, 0, 0),
            ]),
            ByteMemoryWitness::from_accesses(vec![
                write(5, 0xab, 0, 0),
                write(1, 0xcd, 1, 1),
                read(5, 0xab, 2, 2),
                read(1, 0xcd, 3, 3),
            ]),
        ];
        for (i, w) in cases.iter().enumerate() {
            let trace = build_trace_polynomials(w, CurveType::Bls48581);
            let cs = ByteMemoryConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = trace
                .columns
                .iter()
                .map(|p| &p.evaluations)
                .collect();
            let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            for c in 5..NUM_ROW_CONSTRAINTS {
                for (r, v) in results[c].iter().enumerate() {
                    assert!(
                        v.is_zero(),
                        "case {}: c{} at row {} = {:?} (expected zero)",
                        i, c, r, v,
                    );
                }
            }
        }
    }

    #[test]
    fn is_first_binary_constraint_fires_on_nonbinary_value() {
        let w = ByteMemoryWitness::from_accesses(vec![write(0, 0xab, 0, 0)]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_FIRST][0] = Scalar::from_u64(3, CurveType::Bls48581);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 5 = is_first_binary should fire (3 * 2 = 6).
        assert!(!results[5][0].is_zero(), "is_first_binary should detect 3");
    }

    #[test]
    fn rw_binary_constraint_fires_on_nonbinary_value() {
        let w = ByteMemoryWitness::from_accesses(vec![write(0, 0xab, 0, 0)]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper: rw_s = 5 (not binary).
        cols[COL_RW_S][0] = Scalar::from_u64(5, CurveType::Bls48581);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 3 = rw_s_binary should fail at row 0 (5 * 4 = 20, ≠ 0).
        assert!(!results[3][0].is_zero(), "rw_s_binary should detect 5");
    }

    /// Helper: evaluate `evaluate_shifted_at_point` at z = ω^row_idx,
    /// using the honest witness's column-evaluation form. Returns the
    /// combined shifted-constraint value (before α-aggregation, so we
    /// just use α=1 and inspect each constraint independently by zero
    /// padding alpha higher up).
    ///
    /// For phase-2 unit tests we evaluate the column polynomials at a
    /// random z (not on the domain) by using a fixed scalar and
    /// computing the shifted columns by hand: shifted_evals[k] =
    /// column[k+1] at the next domain index. This is simpler than
    /// running the full FFT/IFFT machinery.
    ///
    /// We don't have a generator ω easily at this scope; instead we
    /// directly invoke `evaluate_shifted_at_point` with hand-crafted
    /// `col_evals_at_z` and `shifted_evals` for one transition.
    fn check_shifted_at_transition(
        trace: &TracePolynomials,
        r: usize,
        alpha: &Scalar,
    ) -> Scalar {
        let curve = trace.curve;
        // Use z = scalar "row index" — irrelevant since the constraint
        // is row-local in its evaluation; we just need col_evals_at_z
        // and shifted_evals = col_evals at row r+1. We also need an
        // omega_n_minus_1 that ISN'T equal to z so the exclusion factor
        // (z - ω^{n-1}) is nonzero. Pick z = 7 (arbitrary), ω^{n-1} = 0.
        let z = Scalar::from_u64(7, curve);
        let omega_n_minus_1 = Scalar::zero(curve);
        let col_evals_at_z: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[r].clone())
            .collect();
        // Shifted: same order as shifted_column_indices().
        let next = r + 1;
        let shifted_evals = vec![
            trace.columns[COL_ADDR_S].evaluations[next].clone(),
            trace.columns[COL_VAL_S].evaluations[next].clone(),
            trace.columns[COL_RW_S].evaluations[next].clone(),
            trace.columns[COL_IS_REAL_S].evaluations[next].clone(),
            trace.columns[COL_IS_SAME_ADDR_S].evaluations[next].clone(),
            trace.columns[COL_INV_DIFF_ADDR_S].evaluations[next].clone(),
        ];
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows);
        cs.evaluate_shifted_at_point(
            &col_evals_at_z,
            &shifted_evals,
            &z,
            &omega_n_minus_1,
            alpha,
            NUM_ROW_CONSTRAINTS,
        )
    }

    #[test]
    fn shifted_constraints_zero_on_honest_a1b_trace() {
        // Honest 4 writes + 4 reads pattern.
        let mut accs = Vec::new();
        for k in 0..4u64 {
            accs.push(write(k, 0xab + k as u8, k, k));
        }
        for k in 0..4u64 {
            accs.push(read(k, 0xab + k as u8, 4 + k, 4));
        }
        let w = ByteMemoryWitness::from_accesses(accs);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let alpha = Scalar::from_u64(31337, CurveType::Bls48581);
        // Check every real → real transition: rows 0..(num_rows - 1).
        for r in 0..(trace.num_rows - 1) {
            let v = check_shifted_at_transition(&trace, r, &alpha);
            assert!(
                v.is_zero(),
                "shifted constraints at row {} = {:?} (expected zero)",
                r, v,
            );
        }
    }

    #[test]
    fn shifted_constraint_detects_broken_sort_within_addr_group() {
        // Manually construct a sorted view that VIOLATES read-same:
        // two reads at addr=0 with different vals.
        // Unsorted view: write(0,0xab,0), read(0,0xab,1), read(0,0xff,2).
        // Honest sorted: write(0,0xab,0), read(0,0xab,1), read(0,0xff,2)
        // → second read at addr=0 with same_addr=1 would require
        //   val=0xab but witness says 0xff → fires constraint 7.
        let w = ByteMemoryWitness::from_accesses(vec![
            write(0, 0xab, 0, 0),
            read(0, 0xab, 1, 1),
            read(0, 0xff, 2, 2),
        ]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let alpha = Scalar::from_u64(31337, CurveType::Bls48581);
        let mut found_nonzero = false;
        for r in 0..(trace.num_rows - 1) {
            let v = check_shifted_at_transition(&trace, r, &alpha);
            if !v.is_zero() {
                found_nonzero = true;
            }
        }
        assert!(found_nonzero,
                "broken read-same should fire at least one shifted constraint");
    }

    #[test]
    fn shifted_constraint_detects_uninit_nonzero_read_on_sorted_view() {
        // Single Read at addr=5, val=0xff. The sorted view has
        // is_same_addr_s = 0 at row 0 (no previous), and constraint 8
        // (read-first-zero) requires val_s(ω·z) = 0 when (1 -
        // is_same_addr_s(ω·z)) * (1 - rw_s(ω·z)) = 1. The "next row"
        // is the padding row; is_real_s(ω·z) = 0 there, so constraint
        // doesn't fire at the row-0 → padding transition.
        //
        // Instead, force the issue: build a witness with TWO accesses
        // at DIFFERENT addrs, the second being a Read of a nonzero
        // value. Sorted view: addr=5 row 0 with rw=Write val=0xab,
        // addr=99 row 1 with rw=Read val=0xff. Transition 0 → 1:
        // is_real_s(ω·z) = 1, is_same_addr_s(ω·z) = 0 (different addr),
        // rw_s(ω·z) = 0 (read), val_s(ω·z) = 0xff. Constraint 8 fires.
        let w = ByteMemoryWitness::from_accesses(vec![
            write(5, 0xab, 0, 0),
            read(99, 0xff, 1, 1),
        ]);
        // Don't call verify_read_consistency since this witness is by
        // design inconsistent. Just check the constraint catches it.
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let alpha = Scalar::from_u64(31337, CurveType::Bls48581);
        let v = check_shifted_at_transition(&trace, 0, &alpha);
        assert!(
            !v.is_zero(),
            "read-first-zero violation should fire at transition 0",
        );
    }

    #[test]
    fn evaluate_at_point_zero_on_honest_row() {
        let w = ByteMemoryWitness::from_accesses(vec![
            write(0, 0xab, 0, 0),
            write(1, 0xcd, 1, 1),
            read(0, 0xab, 2, 2),
        ]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(31337, CurveType::Bls48581);
        for r in 0..trace.padded_size as usize {
            let evals_at_r: Vec<Scalar> = trace
                .columns
                .iter()
                .map(|p| p.evaluations[r].clone())
                .collect();
            let combined = cs.evaluate_at_point(&evals_at_r, &alpha);
            assert!(
                combined.is_zero(),
                "combined constraint at row {} = {:?} (expected zero)",
                r,
                combined,
            );
        }
    }

    #[test]
    fn standalone_prove_with_scheme_honest_witness() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        // A1b-pattern witness: 4 byte writes + 4 byte reads.
        let mut accs = Vec::new();
        for k in 0..4u64 {
            accs.push(write(k, 0xab + k as u8, k, k));
        }
        for k in 0..4u64 {
            accs.push(read(k, 0xab + k as u8, 4 + k, 4));
        }
        let w = ByteMemoryWitness::from_accesses(accs);
        let trace = build_trace_polynomials(&w, curve);

        let omega = scheme.domain_generator(trace.padded_size);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);

        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone byte_memory_air proof must verify",
        );
    }

    #[test]
    fn standalone_prove_with_scheme_tampered_read_value_panics() {
        // Tamper the sorted-view val_s on a read row so it doesn't
        // match the prior write's value. The prover should still
        // produce a proof (it doesn't reject the witness up front),
        // but verify_with_scheme should reject — or the prover panics
        // when its internal sanity check on quotient division fails,
        // which is the more common outcome in this codebase.
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let w = ByteMemoryWitness::from_accesses(vec![
            write(0, 0xab, 0, 0),
            read(0, 0xab, 1, 1),
        ]);
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper: change val_s on the read row (row 1 in sorted view) to 0xff.
        trace.columns[COL_VAL_S].evaluations[1] = Scalar::from_u64(0xff, curve);

        let omega = scheme.domain_generator(trace.padded_size);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);

        // Either the prover panics (quotient division non-clean) or
        // verify rejects. Both are acceptable soundness outcomes.
        let prove_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prove_with_scheme(&trace, &cs, &scheme)
        }));
        match prove_result {
            Ok(proof) => {
                let ok = verify_with_scheme(&proof, &cs, &scheme, curve);
                assert!(!ok, "tampered read-same val should be rejected at verify");
            }
            Err(_) => {
                // Prover panicked — the quotient division detected
                // that the constraint polynomial doesn't vanish on
                // the domain. This is the codebase's normal soundness
                // signal.
            }
        }
    }

    #[test]
    fn phase3_self_linkage_closures_match_on_honest_witness() {
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let mut accs = Vec::new();
        for k in 0..4u64 {
            accs.push(write(k, 0xab + k as u8, k, k));
        }
        for k in 0..4u64 {
            accs.push(read(k, 0xab + k as u8, 4 + k, 4));
        }
        let w = ByteMemoryWitness::from_accesses(accs);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);

        let traces: Vec<(&TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&trace, &cs)];
        let linkage = make_byte_memory_self_linkage_descriptor(0);

        let (proofs, ext) = joint_prove(&traces, &[linkage], &scheme)
            .expect("honest self-linkage joint_prove must succeed");
        assert_eq!(proofs.len(), 1);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] self-linkage label={} closure_match={}",
            lp.label,
            lp.closure_a == lp.closure_b,
        );
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest sort = honest unsorted multiset → closures must match",
        );
    }

    #[test]
    fn phase3_joint_verify_accepts_honest_self_linkage_e2e() {
        // Full end-to-end: joint_prove → joint_verify on the byte-
        // memory AIR with self-linkage. Validates the complete phase
        // 1 + 2 + 3 chain through the proof pipeline.
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let mut accs = Vec::new();
        for k in 0..4u64 {
            accs.push(write(k, 0xab + k as u8, k, k));
        }
        for k in 0..4u64 {
            accs.push(read(k, 0xab + k as u8, 4 + k, 4));
        }
        let w = ByteMemoryWitness::from_accesses(accs);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);

        let traces: Vec<(&TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&trace, &cs)];
        let linkages = vec![make_byte_memory_self_linkage_descriptor(0)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest self-linkage joint_prove must succeed");

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&cs];
        let ok = joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve);
        assert!(ok, "honest joint_verify on byte-memory self-linkage must pass");
    }

    #[test]
    fn phase3_self_linkage_closures_mismatch_on_tampered_sort() {
        // Tamper: change one byte in the sorted view so the multiset
        // no longer equals the unsorted view. Closures should mismatch.
        use crate::cross_air_logup::joint_prove;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let mut accs = Vec::new();
        for k in 0..4u64 {
            accs.push(write(k, 0xab + k as u8, k, k));
        }
        for k in 0..4u64 {
            accs.push(read(k, 0xab + k as u8, 4 + k, 4));
        }
        let w = ByteMemoryWitness::from_accesses(accs);
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper: change val_s at sorted row 0 (a write at addr=0)
        // from 0xab to 0xee. The sorted-view multiset now contains
        // 0xee instead of 0xab.
        trace.columns[COL_VAL_S].evaluations[0] = Scalar::from_u64(0xee, curve);

        let omega = scheme.domain_generator(trace.padded_size);
        let cs = ByteMemoryConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);

        let traces: Vec<(&TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&trace, &cs)];
        let linkage = make_byte_memory_self_linkage_descriptor(0);

        // joint_prove may panic on the per-AIR proof (because the
        // tampered val_s violates the read-same constraint). Catch
        // either outcome.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            joint_prove(&traces, &[linkage], &scheme)
        }));
        match result {
            Ok(Ok((_, ext))) => {
                let lp = &ext.linkage_proofs[0];
                assert_ne!(
                    lp.closure_a, lp.closure_b,
                    "tampered sort multiset → closures must mismatch",
                );
            }
            Ok(Err(_)) | Err(_) => {
                // joint_prove returned an error or panicked. Both are
                // acceptable soundness signals: the tampered witness
                // failed somewhere in the proof pipeline.
            }
        }
    }

    #[test]
    fn read_consistency_holds_for_a1b_replay_pattern() {
        // 4 byte writes then 4 reads (mirrors A1b joint_prove pattern).
        let mut accs = Vec::new();
        for k in 0..4u64 {
            accs.push(write(k, 0xab + k as u8, k, k));
        }
        for k in 0..4u64 {
            accs.push(read(k, 0xab + k as u8, 4 + k, 4));
        }
        let w = ByteMemoryWitness::from_accesses(accs);
        w.verify_read_consistency().unwrap();
    }
}
