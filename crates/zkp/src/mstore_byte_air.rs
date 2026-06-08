//! MSTORE byte-decomposition gadget AIR — Phase A1b-mem step 1c-b.
//!
//! Bridges EVM main MSTORE rows (1 row per MSTORE invocation, U256
//! value as 4 LE u64 limbs) to the byte-memory AIR's Write entries (32
//! byte-granular writes per MSTORE invocation). This AIR has **1 row
//! per MSTORE invocation**, exposing:
//!   - The source offset and U256 (for linking back to EVM main).
//!   - 32 BE byte witness columns (for linking forward to byte-memory).
//!   - 32 ADDR_K = offset + k columns (so the byte-memory linkage can
//!     project per-byte tuples).
//!
//! Constraints:
//!   - `is_real` binary.
//!   - 4 limb-decomposition equations:
//!     `limb_3 = Σ_{k=0..8} byte_k * 2^(8*(7-k))`           — bytes 0..7  (MSB end)
//!     `limb_2 = Σ_{k=8..16} byte_k * 2^(8*(15-k))`         — bytes 8..15
//!     `limb_1 = Σ_{k=16..24} byte_k * 2^(8*(23-k))`        — bytes 16..23
//!     `limb_0 = Σ_{k=24..32} byte_k * 2^(8*(31-k))`        — bytes 24..31 (LSB end)
//!   - 32 ADDR_K binding equations: `addr_k - offset - k = 0`.
//!   - 32 byte range checks (8-bit) via LogUp.
//!
//! Linkages (descriptors live in `crates/evm/src/cross_air_linkage.rs`):
//!   - L_evm_to_gadget: EVM main MSTORE row tuple
//!     `(mem_offset, mem_value[L0..L3])` ↔ this gadget's tuple
//!     `(source_offset, source_limb[L0..L3])` (5-col, 1:1 multiset).
//!   - L_gadget_to_bytemem_k for k ∈ 0..32: this gadget's tuple
//!     `(addr_k, byte_k)` ↔ byte-memory AIR's tuple `(addr, val)`
//!     gated by `rw==1` (2-col, 1:1 multiset).
//!
//! Combined: every EVM MSTORE event commits to a 32-byte BE
//! decomposition that the byte-memory AIR's Write multiset must
//! contain (via the 32 per-position linkages).

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_OFFSET: usize = 0;
pub const COL_LIMB_L0: usize = 1;
pub const COL_LIMB_L1: usize = 2;
pub const COL_LIMB_L2: usize = 3;
pub const COL_LIMB_L3: usize = 4;
pub const COL_BYTE_OFFSET: usize = 5; // byte[0..32] at cols 5..37
pub const NUM_BYTES: usize = 32;
pub const COL_ADDR_OFFSET: usize = COL_BYTE_OFFSET + NUM_BYTES; // addr[0..32] at cols 37..69
pub const COL_IS_REAL: usize = COL_ADDR_OFFSET + NUM_BYTES; // col 69
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 70

/// Row-local constraints:
///   0:        is_real binary
///   1..5:     4 limb-decomposition equations
///   5..37:    32 addr_k = offset + k bindings
pub const NUM_ROW_CONSTRAINTS: usize = 1 + 4 + NUM_BYTES;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness type ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct MstoreByteRow {
    pub offset: u64,
    /// 4 LE u64 limbs of the U256 value being stored.
    pub limb: [u64; 4],
}

#[derive(Clone, Debug, Default)]
pub struct MstoreByteWitness {
    pub invocations: Vec<MstoreByteRow>,
}

impl MstoreByteWitness {
    pub fn from_invocations(invocations: Vec<MstoreByteRow>) -> Self {
        Self { invocations }
    }
}

/// Extract the 32 BE bytes of a U256 held as 4 LE u64 limbs.
/// byte[0] is the MSB of the value, byte[31] is the LSB.
pub fn be_bytes_of_u256(limb: [u64; 4]) -> [u8; NUM_BYTES] {
    let mut bytes = [0u8; NUM_BYTES];
    for i in 0..NUM_BYTES {
        let limb_idx = 3 - (i / 8); // BE: byte 0 is MSB → limb 3
        let byte_in_limb = 7 - (i % 8); // BE: byte 0 of limb is MSB of limb
        bytes[i] = ((limb[limb_idx] >> (byte_in_limb * 8)) & 0xff) as u8;
    }
    bytes
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &MstoreByteWitness,
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
        for j in 0..4 {
            columns[COL_LIMB_L0 + j][i] = Scalar::from_u64(inv.limb[j], curve);
        }
        let bytes = be_bytes_of_u256(inv.limb);
        for k in 0..NUM_BYTES {
            columns[COL_BYTE_OFFSET + k][i] = Scalar::from_u64(bytes[k] as u64, curve);
            columns[COL_ADDR_OFFSET + k][i] =
                Scalar::from_u64(inv.offset.wrapping_add(k as u64), curve);
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

pub struct MstoreByteConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl MstoreByteConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// Build the coefficient vector for `Σ_{k=0..8} byte_{lo+k} * 2^(8*(7-k))`,
/// where `lo` is the starting byte index for this limb (0, 8, 16, 24
/// for limbs 3, 2, 1, 0 respectively). Each byte contributes a single
/// monomial in the resulting expression.
fn limb_decomp_targets(limb_idx: usize) -> [(usize, u64); 8] {
    // limb 0 (LSB) ↔ bytes 24..32, byte 24 → 2^56 (high), byte 31 → 2^0 (low)
    // limb 3 (MSB) ↔ bytes 0..7,    byte 0  → 2^56,        byte 7  → 2^0
    let lo = match limb_idx {
        3 => 0,
        2 => 8,
        1 => 16,
        0 => 24,
        _ => unreachable!(),
    };
    let mut out = [(0usize, 0u64); 8];
    for k in 0..8 {
        let byte_idx = lo + k;
        let pow = 1u64 << (8 * (7 - k));
        out[k] = (byte_idx, pow);
    }
    out
}

impl VmConstraintSystem for MstoreByteConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for j in 0..4 {
            labels.push(format!("limb_{}_decomp", j));
        }
        for k in 0..NUM_BYTES {
            labels.push(format!("addr_{}_binding", k));
        }
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
        let mut bin = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            bin[r] = v.mul(&v.sub(&one));
        }
        out.push(bin);

        // 1..5: limb_j_decomp
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut col = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in targets {
                    let b = &columns[COL_BYTE_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(pow, curve)));
                }
                col[r] = columns[COL_LIMB_L0 + limb_idx][r].sub(&sum);
            }
            out.push(col);
        }

        // 5..37: is_real * (addr_k - offset - k) = 0
        // Gated so the constraint vanishes on padding rows (where
        // addr_k = offset = 0 but k is a nonzero constant).
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
        let v = &col_evals[COL_IS_REAL];
        acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
        alpha_pow = alpha_pow.mul(alpha);

        // 1..5: limb decomp
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in targets {
                sum = sum.add(
                    &col_evals[COL_BYTE_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(pow, curve)),
                );
            }
            let body = col_evals[COL_LIMB_L0 + limb_idx].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 5..37: is_real * (addr_k - offset - k)
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

        // 1..5: limb decomp
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in targets {
                let b = &col_coeffs[COL_BYTE_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_LIMB_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 5..37: is_real * (addr_k - offset - k)
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
                    label: format!("mstore_byte_{}_8bit", k),
                    column_index: COL_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptor: gadget → byte-memory (per byte position) ────

/// Build the cross-AIR LogUp descriptor for the k-th byte position
/// linkage from this MSTORE byte-decomposition gadget to the byte-
/// memory AIR. There are 32 such linkages total (one per `k ∈ 0..32`).
///
/// A side (this gadget): tuple `(addr_k, byte_k)` gated by `is_real`.
/// B side (byte-memory): tuple `(addr, val)` gated by `rw` (1 on Writes).
///
/// Combined with `make_evm_mstore_to_byte_decomp_linkage_descriptor`
/// (binding EVM main MSTORE rows to this gadget), the chain forces
/// every MSTORE event's k-th BE byte to appear in byte-memory at
/// address `offset + k`.
pub fn make_mstore_byte_to_byte_memory_linkage_descriptor(
    mstore_byte_layer_index: usize,
    byte_memory_layer_index: usize,
    byte_position: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    assert!(byte_position < NUM_BYTES);
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: format!("mstore_byte_{}_to_byte_memory_v1", byte_position),
        a_layer_index: mstore_byte_layer_index,
        a_columns: vec![
            COL_ADDR_OFFSET + byte_position,
            COL_BYTE_OFFSET + byte_position,
        ],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: byte_memory_layer_index,
        b_columns: vec![
            crate::byte_memory_air::COL_ADDR,
            crate::byte_memory_air::COL_VAL,
        ],
        b_selector_column: Some(crate::byte_memory_air::COL_RW),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn be_bytes_decomposition_known_value() {
        // 0x0102030405060708_090a0b0c0d0e0f10_1112131415161718_191a1b1c1d1e1f20
        // BE bytes = 0x01..0x20
        let limb = [
            0x191a1b1c1d1e1f20u64, // L0 = low 64 bits = LSB
            0x1112131415161718u64,
            0x090a0b0c0d0e0f10u64,
            0x0102030405060708u64, // L3 = high 64 bits = MSB
        ];
        let bytes = be_bytes_of_u256(limb);
        let want: Vec<u8> = (1..=32u8).collect();
        assert_eq!(&bytes[..], &want[..]);
    }

    #[test]
    fn trace_builder_populates_all_columns() {
        let w = MstoreByteWitness::from_invocations(vec![MstoreByteRow {
            offset: 100,
            limb: [
                0x191a1b1c1d1e1f20u64,
                0x1112131415161718u64,
                0x090a0b0c0d0e0f10u64,
                0x0102030405060708u64,
            ],
        }]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // Offset.
        assert_eq!(trace.columns[COL_OFFSET].evaluations[0].to_u64(), 100);
        // Limbs.
        assert_eq!(trace.columns[COL_LIMB_L0].evaluations[0].to_u64(), 0x191a1b1c1d1e1f20);
        assert_eq!(trace.columns[COL_LIMB_L3].evaluations[0].to_u64(), 0x0102030405060708);
        // Bytes — byte 0 = 0x01 (MSB), byte 31 = 0x20 (LSB).
        assert_eq!(trace.columns[COL_BYTE_OFFSET + 0].evaluations[0].to_u64(), 0x01);
        assert_eq!(trace.columns[COL_BYTE_OFFSET + 31].evaluations[0].to_u64(), 0x20);
        // Addr_k.
        assert_eq!(trace.columns[COL_ADDR_OFFSET + 0].evaluations[0].to_u64(), 100);
        assert_eq!(trace.columns[COL_ADDR_OFFSET + 31].evaluations[0].to_u64(), 131);
        // is_real.
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = MstoreByteWitness::from_invocations(vec![MstoreByteRow {
            offset: 100,
            limb: [
                0x191a1b1c1d1e1f20u64,
                0x1112131415161718u64,
                0x090a0b0c0d0e0f10u64,
                0x0102030405060708u64,
            ],
        }]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = MstoreByteConstraintSystem::new(trace.num_rows);
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
    fn limb_decomp_constraint_fires_on_tampered_byte() {
        let w = MstoreByteWitness::from_invocations(vec![MstoreByteRow {
            offset: 0,
            limb: [0xab, 0, 0, 0],
        }]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper byte 31 (LSB of limb 0): 0xab → 0xff.
        cols[COL_BYTE_OFFSET + 31][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = MstoreByteConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint index 1 (limb_0 decomp comes right after is_real_binary;
        // limbs are pushed in order 0, 1, 2, 3 by `evaluate_on_domain`).
        let limb0_idx = 1;
        assert!(
            !results[limb0_idx][0].is_zero(),
            "limb_0 decomp should fire on tampered byte",
        );
    }

    #[test]
    fn addr_binding_constraint_fires_on_tampered_addr() {
        let w = MstoreByteWitness::from_invocations(vec![MstoreByteRow {
            offset: 100,
            limb: [0; 4],
        }]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper addr_5 to 200 (should be 105).
        cols[COL_ADDR_OFFSET + 5][0] = Scalar::from_u64(200, CurveType::Bls48581);
        let cs = MstoreByteConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // addr_k bindings start at constraint index 1 + 4 = 5.
        let addr_5_idx = 1 + 4 + 5;
        assert!(
            !results[addr_5_idx][0].is_zero(),
            "addr_5 binding should fire on tampered addr",
        );
    }

    #[test]
    fn linkage_descriptor_well_formed() {
        let desc = make_mstore_byte_to_byte_memory_linkage_descriptor(0, 1, 7);
        assert_eq!(desc.label, "mstore_byte_7_to_byte_memory_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(
            desc.a_columns,
            vec![COL_ADDR_OFFSET + 7, COL_BYTE_OFFSET + 7],
        );
        assert_eq!(
            desc.b_columns,
            vec![
                crate::byte_memory_air::COL_ADDR,
                crate::byte_memory_air::COL_VAL,
            ],
        );
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::byte_memory_air::COL_RW),
        );
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

        let w = MstoreByteWitness::from_invocations(vec![MstoreByteRow {
            offset: 0,
            limb: [
                0x191a1b1c1d1e1f20u64,
                0x1112131415161718u64,
                0x090a0b0c0d0e0f10u64,
                0x0102030405060708u64,
            ],
        }]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = MstoreByteConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone MSTORE byte gadget proof must verify",
        );
    }
}
