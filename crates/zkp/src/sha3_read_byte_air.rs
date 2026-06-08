//! SHA3 byte-read forwarding gadget AIR — Phase A1b-mem step 1c-c.
//!
//! Bridges the SHA3 input gadget AIR (1 row per SHA3 invocation, 256
//! INPUT_BYTE columns, INPUT_LEN) to the byte-memory AIR's Read
//! entries (one Read per byte that SHA3 actually consumed, gated by
//! `k < INPUT_LEN`).
//!
//! Like [`crate::mstore_byte_air`] but for the READ side: instead of
//! decomposing a U256 into 32 bytes, this gadget MIRRORS the per-row
//! 256-byte input + length from `crate::sha3_input_air` and exposes:
//!   - 256 ADDR_K = offset + k columns (so the byte-memory linkage can
//!     project per-byte tuples).
//!   - 256 IS_ACTIVE_K columns (1 iff k < INPUT_LEN), constrained to
//!     be a binary monotone-non-increasing prefix of length INPUT_LEN.
//!
//! Constraints:
//!   - `is_real` binary.
//!   - 256 ADDR_K bindings: `is_real * (addr_k - offset - k) = 0`.
//!   - 256 IS_ACTIVE_K binary checks.
//!   - 255 monotonicity constraints: `is_active_k >= is_active_{k+1}`
//!     (encoded as `(is_active_k - is_active_{k+1}) ∈ {0, 1}` via
//!     `(is_active_k - is_active_{k+1}) * (is_active_k -
//!     is_active_{k+1} - 1) = 0`).
//!   - 1 length-binding: `Σ_{k=0..256} is_active_k - input_len = 0`,
//!     gated by `is_real`.
//!   - 256 byte-range checks (8-bit) on INPUT_BYTE via LogUp.
//!
//! Linkages (descriptors below + companion EVM-side linkage):
//!   - L_sha3_input_to_read_byte (deferred): SHA3 input gadget row's
//!     `(INPUT_BYTE[0..256], INPUT_LEN)` ↔ this gadget's
//!     `(INPUT_BYTE[0..256], INPUT_LEN)` (257-col, 1:1 multiset).
//!     Plus EVM SHA3 row's `(input0=offset, input1=len)` ↔ this
//!     gadget's `(offset, input_len)` (2-col, 1:1).
//!   - L_read_byte_to_bytemem_k for k ∈ 0..256: this gadget's
//!     `(addr_k, byte_k)` ↔ byte-memory's `(addr, val)` gated by
//!     `rw == 0` and `is_active_k == 1`.
//!
//! For k ∈ 0..256, the per-position byte-memory linkage emits at most
//! one tuple per SHA3 invocation (the byte at position k iff k <
//! input_len). The byte-memory side has at most `Σ_invocations
//! input_len` Read entries, so the multisets match up to multiplicity.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const NUM_BYTES: usize = 256;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_OFFSET: usize = 0;
pub const COL_INPUT_LEN: usize = 1;
pub const COL_INPUT_BYTE_OFFSET: usize = 2; // byte[0..256] at cols 2..258
pub const COL_ADDR_OFFSET: usize = COL_INPUT_BYTE_OFFSET + NUM_BYTES; // 258..514
pub const COL_IS_ACTIVE_OFFSET: usize = COL_ADDR_OFFSET + NUM_BYTES; // 514..770
pub const COL_IS_REAL: usize = COL_IS_ACTIVE_OFFSET + NUM_BYTES; // 770
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 771

/// Row-local constraints:
///   0:          is_real binary
///   1..257:     256 addr_k bindings
///   257..513:   256 is_active_k binary
///   513..768:   255 monotonicity constraints
///   768:        length binding
pub const NUM_ROW_CONSTRAINTS: usize = 1 + NUM_BYTES + NUM_BYTES + (NUM_BYTES - 1) + 1;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness type ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Sha3ReadByteRow {
    pub offset: u64,
    /// Number of bytes actually read by SHA3 (0..=NUM_BYTES).
    pub input_len: u32,
    /// 256 input bytes, padded with trailing zeros beyond input_len.
    pub input_byte: [u8; NUM_BYTES],
}

#[derive(Clone, Debug, Default)]
pub struct Sha3ReadByteWitness {
    pub invocations: Vec<Sha3ReadByteRow>,
}

impl Sha3ReadByteWitness {
    pub fn from_invocations(invocations: Vec<Sha3ReadByteRow>) -> Result<Self, &'static str> {
        for inv in &invocations {
            if inv.input_len as usize > NUM_BYTES {
                return Err("sha3_read_byte_air: input_len exceeds NUM_BYTES");
            }
        }
        Ok(Self { invocations })
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Sha3ReadByteWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, inv) in witness.invocations.iter().enumerate() {
        columns[COL_OFFSET][i] = Scalar::from_u64(inv.offset, curve);
        columns[COL_INPUT_LEN][i] = Scalar::from_u64(inv.input_len as u64, curve);
        for k in 0..NUM_BYTES {
            columns[COL_INPUT_BYTE_OFFSET + k][i] =
                Scalar::from_u64(inv.input_byte[k] as u64, curve);
            columns[COL_ADDR_OFFSET + k][i] =
                Scalar::from_u64(inv.offset.wrapping_add(k as u64), curve);
            columns[COL_IS_ACTIVE_OFFSET + k][i] = if (k as u32) < inv.input_len {
                one.clone()
            } else {
                zero.clone()
            };
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

// ─── Constraint system ─────────────────────────────────────────────────

pub struct Sha3ReadByteConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Sha3ReadByteConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Sha3ReadByteConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for k in 0..NUM_BYTES { labels.push(format!("addr_{}_binding", k)); }
        for k in 0..NUM_BYTES { labels.push(format!("is_active_{}_binary", k)); }
        for k in 0..(NUM_BYTES - 1) { labels.push(format!("monotone_{}_to_{}", k, k+1)); }
        labels.push("length_binding".into());
        labels
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
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real binary
        let mut bin_real = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            bin_real[r] = v.mul(&v.sub(&one));
        }
        out.push(bin_real);

        // 1..257: is_real * (addr_k - offset - k) = 0
        for k in 0..NUM_BYTES {
            let mut col = vec![Scalar::zero(curve); n];
            let k_scalar = Scalar::from_u64(k as u64, curve);
            for r in 0..n {
                let addr_k = &columns[COL_ADDR_OFFSET + k][r];
                let offset = &columns[COL_OFFSET][r];
                let is_real = &columns[COL_IS_REAL][r];
                col[r] = is_real.mul(&addr_k.sub(offset).sub(&k_scalar));
            }
            out.push(col);
        }

        // 257..513: is_active_k binary
        for k in 0..NUM_BYTES {
            let mut col = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_ACTIVE_OFFSET + k][r];
                col[r] = v.mul(&v.sub(&one));
            }
            out.push(col);
        }

        // 513..768: monotonicity (is_active_k - is_active_{k+1}) ∈ {0, 1}
        // i.e. diff * (diff - 1) = 0
        for k in 0..(NUM_BYTES - 1) {
            let mut col = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let cur = &columns[COL_IS_ACTIVE_OFFSET + k][r];
                let nxt = &columns[COL_IS_ACTIVE_OFFSET + k + 1][r];
                let diff = cur.sub(nxt);
                col[r] = diff.mul(&diff.sub(&one));
            }
            out.push(col);
        }

        // 768: is_real * (Σ is_active_k - input_len) = 0
        {
            let mut col = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for k in 0..NUM_BYTES {
                    sum = sum.add(&columns[COL_IS_ACTIVE_OFFSET + k][r]);
                }
                let input_len = &columns[COL_INPUT_LEN][r];
                let is_real = &columns[COL_IS_REAL][r];
                col[r] = is_real.mul(&sum.sub(input_len));
            }
            out.push(col);
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 1..257: addr_k binding
        for k in 0..NUM_BYTES {
            let k_scalar = Scalar::from_u64(k as u64, curve);
            let body = col_evals[COL_IS_REAL].mul(
                &col_evals[COL_ADDR_OFFSET + k]
                    .sub(&col_evals[COL_OFFSET])
                    .sub(&k_scalar),
            );
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 257..513: is_active_k binary
        for k in 0..NUM_BYTES {
            let v = &col_evals[COL_IS_ACTIVE_OFFSET + k];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 513..768: monotonicity
        for k in 0..(NUM_BYTES - 1) {
            let cur = &col_evals[COL_IS_ACTIVE_OFFSET + k];
            let nxt = &col_evals[COL_IS_ACTIVE_OFFSET + k + 1];
            let diff = cur.sub(nxt);
            acc = acc.add(&alpha_pow.mul(&diff.mul(&diff.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 768: length binding
        {
            let mut sum = Scalar::zero(curve);
            for k in 0..NUM_BYTES {
                sum = sum.add(&col_evals[COL_IS_ACTIVE_OFFSET + k]);
            }
            let body = col_evals[COL_IS_REAL].mul(&sum.sub(&col_evals[COL_INPUT_LEN]));
            acc = acc.add(&alpha_pow.mul(&body));
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
        let one_poly = vec![Scalar::one(curve)];

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        // 0: is_real binary
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 1..257: is_real * (addr_k - offset - k)
        for k in 0..NUM_BYTES {
            let k_scalar = Scalar::from_u64(k as u64, curve);
            let k_poly = vec![k_scalar];
            let body0 = poly_sub(
                &col_coeffs[COL_ADDR_OFFSET + k],
                &col_coeffs[COL_OFFSET],
                curve,
            );
            let body1 = poly_sub(&body0, &k_poly, curve);
            let body = poly_mul(&col_coeffs[COL_IS_REAL], &body1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 257..513: is_active_k binary
        for k in 0..NUM_BYTES {
            let v = &col_coeffs[COL_IS_ACTIVE_OFFSET + k];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 513..768: monotonicity
        for k in 0..(NUM_BYTES - 1) {
            let cur = &col_coeffs[COL_IS_ACTIVE_OFFSET + k];
            let nxt = &col_coeffs[COL_IS_ACTIVE_OFFSET + k + 1];
            let diff = poly_sub(cur, nxt, curve);
            let diff_m1 = poly_sub(&diff, &one_poly, curve);
            let body = poly_mul(&diff, &diff_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 768: length binding
        {
            let mut sum = vec![Scalar::zero(curve)];
            for k in 0..NUM_BYTES {
                sum = poly_add(&sum, &col_coeffs[COL_IS_ACTIVE_OFFSET + k], curve);
            }
            let body0 = poly_sub(&sum, &col_coeffs[COL_INPUT_LEN], curve);
            let body = poly_mul(&col_coeffs[COL_IS_REAL], &body0, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
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
        // 8-bit range check on every byte column.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..NUM_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sha3_read_byte_{}_8bit", k),
                    column_index: COL_INPUT_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Build the cross-AIR LogUp descriptor for the k-th byte position
/// linkage from this SHA3 byte-read gadget to the byte-memory AIR.
/// 256 such linkages total (one per `k ∈ 0..256`).
///
/// A side (this gadget): tuple `(addr_k, byte_k)` gated by
/// `is_active_k` (which is 1 iff k < input_len, 0 on padding rows
/// and on rows where k ≥ input_len).
/// B side (byte-memory): tuple `(addr, val)` gated by inverse of `rw`
/// — i.e. Read entries only. But the byte-memory AIR's selector
/// column for Reads is `(1 - rw) * is_real`, which isn't a single
/// column. **The byte-memory AIR's `rw=0` rows include both Reads AND
/// padding** (both have rw=0). To distinguish, we'd need a `sel_read`
/// column on byte-memory. Until that column is added, this descriptor
/// is built but should be paired with an extension that filters
/// padding out via the joint multi-AIR proof's column-projection.
///
/// As-is, the b_selector is `IS_REAL` (which is 1 on real rows
/// regardless of rw). The resulting linkage is: "every (addr+k,
/// byte_k) tuple from active SHA3 reads appears in some real byte-
/// memory row". This is a *subset* check, not just-Read. The full
/// soundness binding (Read-only) requires adding a sel_read column
/// to byte-memory.
pub fn make_sha3_read_byte_to_byte_memory_linkage_descriptor(
    sha3_read_byte_layer_index: usize,
    byte_memory_layer_index: usize,
    byte_position: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    assert!(byte_position < NUM_BYTES);
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: format!("sha3_read_byte_{}_to_byte_memory_v1", byte_position),
        a_layer_index: sha3_read_byte_layer_index,
        a_columns: vec![
            COL_ADDR_OFFSET + byte_position,
            COL_INPUT_BYTE_OFFSET + byte_position,
        ],
        a_selector_column: Some(COL_IS_ACTIVE_OFFSET + byte_position),
        b_layer_index: byte_memory_layer_index,
        b_columns: vec![
            crate::byte_memory_air::COL_ADDR,
            crate::byte_memory_air::COL_VAL,
        ],
        b_selector_column: Some(crate::byte_memory_air::COL_IS_REAL),
    }
}

/// Build the descriptor for the SHA3 input gadget ↔ this read-byte
/// gadget linkage, binding the full (INPUT_BYTE[0..256], INPUT_LEN)
/// tuple. 257-col tuple, 1:1 multiset between SHA3 invocations on the
/// input gadget and rows on this gadget.
pub fn make_sha3_input_to_read_byte_linkage_descriptor(
    sha3_input_layer_index: usize,
    sha3_read_byte_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sha3_input_air;
    let a_columns: Vec<usize> = (0..NUM_BYTES)
        .map(|k| sha3_input_air::COL_INPUT_BYTE_OFFSET + k)
        .chain(std::iter::once(sha3_input_air::COL_INPUT_LEN))
        .collect();
    let b_columns: Vec<usize> = (0..NUM_BYTES)
        .map(|k| COL_INPUT_BYTE_OFFSET + k)
        .chain(std::iter::once(COL_INPUT_LEN))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sha3_input_to_read_byte_v1".into(),
        a_layer_index: sha3_input_layer_index,
        a_columns,
        a_selector_column: Some(sha3_input_air::COL_IS_REAL),
        b_layer_index: sha3_read_byte_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_witness(offset: u64, input: &[u8]) -> Sha3ReadByteWitness {
        let mut bytes = [0u8; NUM_BYTES];
        bytes[..input.len()].copy_from_slice(input);
        Sha3ReadByteWitness::from_invocations(vec![Sha3ReadByteRow {
            offset,
            input_len: input.len() as u32,
            input_byte: bytes,
        }])
        .unwrap()
    }

    #[test]
    fn trace_builder_populates_all_columns() {
        let w = make_witness(100, &[0xab, 0xcd, 0xef, 0x12]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.columns[COL_OFFSET].evaluations[0].to_u64(), 100);
        assert_eq!(trace.columns[COL_INPUT_LEN].evaluations[0].to_u64(), 4);
        assert_eq!(trace.columns[COL_INPUT_BYTE_OFFSET + 0].evaluations[0].to_u64(), 0xab);
        assert_eq!(trace.columns[COL_INPUT_BYTE_OFFSET + 3].evaluations[0].to_u64(), 0x12);
        assert_eq!(trace.columns[COL_INPUT_BYTE_OFFSET + 4].evaluations[0].to_u64(), 0);
        // addr_k.
        for k in 0..NUM_BYTES {
            assert_eq!(
                trace.columns[COL_ADDR_OFFSET + k].evaluations[0].to_u64(),
                100 + k as u64,
            );
        }
        // is_active_k: 1 for k=0..3, 0 for k=4..256.
        for k in 0..4 {
            assert_eq!(trace.columns[COL_IS_ACTIVE_OFFSET + k].evaluations[0].to_u64(), 1);
        }
        for k in 4..NUM_BYTES {
            assert_eq!(trace.columns[COL_IS_ACTIVE_OFFSET + k].evaluations[0].to_u64(), 0);
        }
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = make_witness(50, &[0x01, 0x02, 0x03]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Sha3ReadByteConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
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
    fn length_binding_fires_on_tampered_input_len() {
        let w = make_witness(0, &[0xab, 0xcd, 0xef, 0x12]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper: input_len = 5 (but is_active has only 4 ones).
        cols[COL_INPUT_LEN][0] = Scalar::from_u64(5, CurveType::Bls48581);
        let cs = Sha3ReadByteConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // length_binding is the last constraint, index NUM_ROW_CONSTRAINTS - 1.
        let length_idx = NUM_ROW_CONSTRAINTS - 1;
        assert!(
            !results[length_idx][0].is_zero(),
            "length_binding should fire on tampered input_len",
        );
    }

    #[test]
    fn monotonicity_fires_on_non_prefix_active() {
        let w = make_witness(0, &[0xab, 0xcd]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper: set is_active[5] = 1, breaking the prefix shape
        // (is_active[2] = 0 then is_active[5] = 1 — not monotone).
        cols[COL_IS_ACTIVE_OFFSET + 5][0] = Scalar::from_u64(1, CurveType::Bls48581);
        let cs = Sha3ReadByteConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Monotonicity constraints start at 1 + NUM_BYTES + NUM_BYTES.
        // The k=4 → k=5 monotonicity is at offset (k=4 within mono block).
        let mono_4_5_idx = 1 + NUM_BYTES + NUM_BYTES + 4;
        // is_active[4] - is_active[5] = 0 - 1 = -1, then diff*(diff-1)
        // = -1 * -2 = 2 ≠ 0.
        assert!(
            !results[mono_4_5_idx][0].is_zero(),
            "monotonicity 4→5 should fire",
        );
    }

    #[test]
    fn read_byte_to_byte_memory_descriptor_well_formed() {
        let desc = make_sha3_read_byte_to_byte_memory_linkage_descriptor(0, 1, 5);
        assert_eq!(desc.label, "sha3_read_byte_5_to_byte_memory_v1");
        assert_eq!(desc.a_columns, vec![COL_ADDR_OFFSET + 5, COL_INPUT_BYTE_OFFSET + 5]);
        assert_eq!(
            desc.b_columns,
            vec![
                crate::byte_memory_air::COL_ADDR,
                crate::byte_memory_air::COL_VAL,
            ],
        );
        assert_eq!(desc.a_selector_column, Some(COL_IS_ACTIVE_OFFSET + 5));
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let w = make_witness(0, &[0xab, 0xcd, 0xef, 0x12]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = Sha3ReadByteConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone SHA3 read-byte gadget proof must verify",
        );
    }

    /// Phase A1b-mem step 1c-c minimal multi-AIR proof: 2-AIR
    /// joint_prove + joint_verify linking `sha3_input_air` ↔ this
    /// `sha3_read_byte_air` via the 257-col tuple linkage. Validates
    /// that the SHA3 input gadget's `(INPUT_BYTE[0..256], INPUT_LEN)`
    /// row matches this read-byte gadget's row, so the read-byte
    /// gadget's bytes are pinned to the (already-A1b-bound) SHA3
    /// input gadget's witnesses.
    ///
    /// Marked #[ignore] — moderately slow: ~6-8 min release expected
    /// (dominated by sha3_read_byte_air's 771 cols).
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with 1 257-col linkage \
                (~5-10 min); run with --release --ignored"]
    fn joint_prove_sha3_input_to_read_byte() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::sha3_input_air::{
            build_trace_polynomials as build_input_trace,
            Sha3InputConstraintSystem, Sha3InputWitness,
        };
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let input_bytes = vec![0xab, 0xcd, 0xef, 0x12];

        // SHA3 input gadget witness.
        let input_w = Sha3InputWitness::from_inputs(&[input_bytes.clone()]).unwrap();
        let input_trace = build_input_trace(&input_w, curve);
        let input_omega = scheme.domain_generator(input_trace.padded_size);
        let input_cs = Sha3InputConstraintSystem::new(input_trace.num_rows)
            .with_omega_and_domain(input_omega, input_trace.padded_size);

        // SHA3 read-byte gadget witness (mirrors input).
        let read_w = make_witness(/* offset */ 0, &input_bytes);
        let read_trace = build_trace_polynomials(&read_w, curve);
        let read_omega = scheme.domain_generator(read_trace.padded_size);
        let read_cs = Sha3ReadByteConstraintSystem::new(read_trace.num_rows)
            .with_omega_and_domain(read_omega, read_trace.padded_size);

        let linkage = make_sha3_input_to_read_byte_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&input_trace, &input_cs), (&read_trace, &read_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("sha3_input ↔ sha3_read_byte joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&input_cs, &read_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
            "joint_verify must accept honest witness",
        );
    }

    #[test]
    fn sha3_input_to_read_byte_descriptor_well_formed() {
        let desc = make_sha3_input_to_read_byte_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "sha3_input_to_read_byte_v1");
        // 256 byte cols + 1 length col = 257.
        assert_eq!(desc.a_columns.len(), 257);
        assert_eq!(desc.b_columns.len(), 257);
        assert_eq!(desc.a_columns[256], crate::sha3_input_air::COL_INPUT_LEN);
        assert_eq!(desc.b_columns[256], COL_INPUT_LEN);
    }
}
