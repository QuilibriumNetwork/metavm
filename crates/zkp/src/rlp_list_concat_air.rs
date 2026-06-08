//! RLP list concatenation witness (step 0).
//!
//! Running-offset accumulator for assembling 20 per-field RLP
//! encodings into the full block header RLP byte stream. Each of the
//! 20 rows carries a field's encoding + its byte offset within the
//! assembled output. The cross-row chain
//! `running_offset[r+1] = running_offset[r] + field_encoded_len[r]`
//! links them into a contiguous stream.
//!
//! # How this closes `block_hash = keccak256(rlp(header))`
//!
//! 1. Each per-field RLP gadget (u64_rlp_air, fixed_rlp_air, etc.)
//!    produces `field_encoded[0..len]`.
//! 2. This accumulator places each field at its correct byte offset
//!    and verifies the running-offset chain.
//! 3. A byte-memory-style cross-AIR LogUp (step 1+) binds each
//!    `(running_offset + k, field_encoded[k])` tuple to the
//!    `header_rlp[offset + k]` column in `block_header_air`.
//! 4. `block_header_air` → `keccak_extract_wide` closes the keccak.
//!
//! # Soundness scope (step 0 + 1a)
//!
//! Step 0: host-side witness + offset-chain verification.
//! Step 1a: algebraic running-offset chain constraint (shifted body):
//!   `IS_REAL[r] · IS_REAL[r+1] · (OFFSET[r+1] − OFFSET[r] − LEN[r]) = 0`
//! Step 1b (future work): per-byte binding via cross-AIR LogUp to header_rlp.

use crate::block_header::BlockHeader;
use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Maximum per-field encoded length (3-byte prefix + 256 data for
/// logs_bloom long-string = 259).
pub const MAX_FIELD_ENCODED_LEN: usize = 259;

// ─── Column layout for the algebraic AIR ──────────────────────────────

pub const COL_FIELD_INDEX: usize = 0;
pub const COL_RUNNING_OFFSET: usize = 1;
pub const COL_FIELD_ENCODED_LEN: usize = 2;
pub const COL_IS_REAL: usize = 3;
pub const NUM_AIR_COLUMNS: usize = COL_IS_REAL + 1; // 4

pub const NUM_ROW_CONSTRAINTS: usize = 1; // is_real binary
pub const NUM_SHIFTED: usize = 1; // running-offset chain

/// Maximum number of fields in a Cancun block header.
pub const MAX_FIELDS: usize = 20;

#[derive(Clone, Debug)]
pub struct RlpListConcatRow {
    pub field_index: usize,
    pub running_offset: usize,
    pub field_encoded: Vec<u8>,
    pub field_encoded_len: usize,
}

#[derive(Clone, Debug)]
pub struct RlpListConcatWitness {
    pub rows: Vec<RlpListConcatRow>,
    pub list_header: Vec<u8>,
    pub total_payload_len: usize,
    pub total_rlp_len: usize,
}

impl RlpListConcatWitness {
    /// Build the witness from a `BlockHeader` by decomposing its
    /// canonical RLP into per-field encodings with running offsets.
    pub fn from_block_header(h: &BlockHeader) -> Self {
        let canonical = crate::block_header::block_header_rlp(h);

        // Build per-field encodings using the canonical RLP functions.
        let mut field_encodings: Vec<Vec<u8>> = Vec::with_capacity(MAX_FIELDS);
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.parent_hash));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.ommers_hash));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.beneficiary));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.state_root));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.transactions_root));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.receipts_root));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.logs_bloom));
        field_encodings.push(crate::rlp::rlp_encode_u256(&h.difficulty));
        field_encodings.push(crate::rlp::rlp_encode_uint(h.number));
        field_encodings.push(crate::rlp::rlp_encode_uint(h.gas_limit));
        field_encodings.push(crate::rlp::rlp_encode_uint(h.gas_used));
        field_encodings.push(crate::rlp::rlp_encode_uint(h.timestamp));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.extra_data));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.mix_hash));
        field_encodings.push(crate::rlp::rlp_encode_bytes(&h.nonce));

        if let Some(ref bf) = h.base_fee_per_gas {
            field_encodings.push(crate::rlp::rlp_encode_u256(bf));
            if let Some(ref wr) = h.withdrawals_root {
                field_encodings.push(crate::rlp::rlp_encode_bytes(wr));
                if let Some(bgu) = h.blob_gas_used {
                    field_encodings.push(crate::rlp::rlp_encode_uint(bgu));
                    if let Some(ebg) = h.excess_blob_gas {
                        field_encodings.push(crate::rlp::rlp_encode_uint(ebg));
                        if let Some(ref pbbr) = h.parent_beacon_block_root {
                            field_encodings.push(crate::rlp::rlp_encode_bytes(pbbr));
                        }
                    }
                }
            }
        }

        let total_payload_len: usize = field_encodings.iter().map(|f| f.len()).sum();

        // Compute the list header.
        let list_header = if total_payload_len < 56 {
            vec![0xc0 + total_payload_len as u8]
        } else {
            let mut len_be = Vec::new();
            let mut n = total_payload_len;
            while n > 0 { len_be.push((n & 0xff) as u8); n >>= 8; }
            len_be.reverse();
            let mut hdr = Vec::with_capacity(1 + len_be.len());
            hdr.push(0xf7 + len_be.len() as u8);
            hdr.extend_from_slice(&len_be);
            hdr
        };

        let header_len = list_header.len();
        let total_rlp_len = header_len + total_payload_len;

        // Build rows with running offsets.
        let mut rows = Vec::with_capacity(field_encodings.len());
        let mut offset = header_len;
        for (i, enc) in field_encodings.iter().enumerate() {
            rows.push(RlpListConcatRow {
                field_index: i,
                running_offset: offset,
                field_encoded: enc.clone(),
                field_encoded_len: enc.len(),
            });
            offset += enc.len();
        }

        // Verify: assembled stream matches canonical.
        let mut assembled = list_header.clone();
        for enc in &field_encodings {
            assembled.extend_from_slice(enc);
        }
        assert_eq!(
            assembled, canonical,
            "assembled RLP must match canonical block_header_rlp",
        );

        Self {
            rows,
            list_header,
            total_payload_len,
            total_rlp_len,
        }
    }

    /// Verify the running-offset chain: each row's offset equals the
    /// previous row's offset + previous row's encoded length.
    pub fn verify_offset_chain(&self) -> Result<(), String> {
        if self.rows.is_empty() { return Ok(()); }
        let expected_start = self.list_header.len();
        if self.rows[0].running_offset != expected_start {
            return Err(format!(
                "row 0 offset {} != list_header_len {}",
                self.rows[0].running_offset, expected_start,
            ));
        }
        for i in 1..self.rows.len() {
            let expected = self.rows[i - 1].running_offset + self.rows[i - 1].field_encoded_len;
            if self.rows[i].running_offset != expected {
                return Err(format!(
                    "row {} offset {} != expected {}",
                    i, self.rows[i].running_offset, expected,
                ));
            }
        }
        let last = self.rows.last().unwrap();
        let final_offset = last.running_offset + last.field_encoded_len;
        if final_offset != self.total_rlp_len {
            return Err(format!(
                "final offset {} != total_rlp_len {}",
                final_offset, self.total_rlp_len,
            ));
        }
        Ok(())
    }

    /// Verify each field's bytes match the canonical header_rlp at
    /// the right offset.
    pub fn verify_byte_alignment(&self, canonical_rlp: &[u8]) -> Result<(), String> {
        for row in &self.rows {
            for k in 0..row.field_encoded_len {
                let pos = row.running_offset + k;
                if pos >= canonical_rlp.len() {
                    return Err(format!(
                        "field {} byte {} at pos {} exceeds canonical len {}",
                        row.field_index, k, pos, canonical_rlp.len(),
                    ));
                }
                if canonical_rlp[pos] != row.field_encoded[k] {
                    return Err(format!(
                        "field {} byte {} mismatch at pos {}: got {} expected {}",
                        row.field_index, k, pos,
                        row.field_encoded[k], canonical_rlp[pos],
                    ));
                }
            }
        }
        Ok(())
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_concat_trace_polynomials(
    witness: &RlpListConcatWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_AIR_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        columns[COL_FIELD_INDEX][r] = Scalar::from_u64(row.field_index as u64, curve);
        columns[COL_RUNNING_OFFSET][r] = Scalar::from_u64(row.running_offset as u64, curve);
        columns[COL_FIELD_ENCODED_LEN][r] = Scalar::from_u64(row.field_encoded_len as u64, curve);
        columns[COL_IS_REAL][r] = one.clone();
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

pub struct RlpListConcatConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl RlpListConcatConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for RlpListConcatConstraintSystem {
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
        if ce.len() < NUM_AIR_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let one = Scalar::one(alpha.curve_type());
        let v = &ce[COL_IS_REAL];
        v.mul(&v.sub(&one))
    }

    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
        let _ = alpha;
        let curve = cc[0][0].curve_type();
        let one_p = vec![Scalar::one(curve)];
        let v = &cc[COL_IS_REAL];
        poly_mul(v, &poly_sub(v, &one_p, curve), curve)
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL, COL_RUNNING_OFFSET]
    }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        if shifted_evals.len() != 2 || col_evals_at_z.len() < NUM_AIR_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let offset_next = &shifted_evals[1];
        let offset_curr = &col_evals_at_z[COL_RUNNING_OFFSET];
        let len_curr = &col_evals_at_z[COL_FIELD_ENCODED_LEN];

        // IS_REAL[r] · IS_REAL[r+1] · (OFFSET[r+1] − OFFSET[r] − LEN[r])
        let gating = v.mul(v_next);
        let chain = offset_next.sub(offset_curr).sub(len_curr);
        let body = gating.mul(&chain);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        ap.mul(&body).mul(&z.sub(omega_n_minus_1))
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
        let v = &column_coeffs[COL_IS_REAL];
        let v_next = poly_shift(v, omega);
        let gating = poly_mul(v, &v_next, curve);

        let offset_curr = &column_coeffs[COL_RUNNING_OFFSET];
        let offset_next = poly_shift(offset_curr, omega);
        let len_curr = &column_coeffs[COL_FIELD_ENCODED_LEN];
        let chain = poly_sub(&poly_sub(&offset_next, offset_curr, curve), len_curr, curve);
        let body = poly_mul(&gating, &chain, curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let total = poly_scalar_mul(&body, &ap);

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, cols: &mut [Vec<Scalar>], nr: usize, ps: usize) {
        if nr == 0 || nr >= ps || cols.len() < NUM_AIR_COLUMNS { return; }
        let z = Scalar::zero(cols[0][0].curve_type());
        for c in cols.iter_mut().take(NUM_AIR_COLUMNS) {
            for cell in c.iter_mut().skip(nr).take(ps - nr) { *cell = z.clone(); }
        }
    }
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cancun_header() -> BlockHeader {
        BlockHeader {
            parent_hash: [0x11; 32],
            beneficiary: [0x33; 20],
            state_root: [0x44; 32],
            transactions_root: [0x55; 32],
            receipts_root: [0x66; 32],
            logs_bloom: [0x77; 256],
            number: 18_500_000,
            gas_limit: 30_000_000,
            gas_used: 15_000_000,
            timestamp: 1_700_000_000,
            extra_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            base_fee_per_gas: Some({
                let mut b = [0u8; 32]; b[24..32].copy_from_slice(&15_000_000_000u64.to_be_bytes()); b
            }),
            withdrawals_root: Some([0x99; 32]),
            blob_gas_used: Some(393_216),
            excess_blob_gas: Some(786_432),
            parent_beacon_block_root: Some([0xaa; 32]),
            ..Default::default()
        }
    }

    #[test]
    fn witness_builds_for_cancun_header() {
        let w = RlpListConcatWitness::from_block_header(&cancun_header());
        assert_eq!(w.rows.len(), 20);
    }

    #[test]
    fn offset_chain_is_valid() {
        let w = RlpListConcatWitness::from_block_header(&cancun_header());
        w.verify_offset_chain().unwrap();
    }

    #[test]
    fn byte_alignment_matches_canonical() {
        let h = cancun_header();
        let w = RlpListConcatWitness::from_block_header(&h);
        let canonical = crate::block_header::block_header_rlp(&h);
        w.verify_byte_alignment(&canonical).unwrap();
    }

    #[test]
    fn pre_london_header_works() {
        let h = BlockHeader {
            number: 12_000_000,
            gas_limit: 15_000_000,
            ..Default::default()
        };
        let w = RlpListConcatWitness::from_block_header(&h);
        assert_eq!(w.rows.len(), 15);
        w.verify_offset_chain().unwrap();
        let canonical = crate::block_header::block_header_rlp(&h);
        w.verify_byte_alignment(&canonical).unwrap();
    }

    #[test]
    fn total_rlp_len_matches_canonical() {
        let h = cancun_header();
        let w = RlpListConcatWitness::from_block_header(&h);
        let canonical = crate::block_header::block_header_rlp(&h);
        assert_eq!(w.total_rlp_len, canonical.len());
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = RlpListConcatWitness::from_block_header(&cancun_header());
        let trace = build_concat_trace_polynomials(&w, CurveType::Bls48581);
        let cs = RlpListConcatConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&cr, trace.num_rows);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(val.is_zero(), "constraint {} row {} nonzero", i, r);
            }
        }
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
        let w = RlpListConcatWitness::from_block_header(&cancun_header());
        let trace = build_concat_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = RlpListConcatConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }

    #[test]
    fn list_header_correct() {
        let h = cancun_header();
        let w = RlpListConcatWitness::from_block_header(&h);
        let canonical = crate::block_header::block_header_rlp(&h);
        // List header bytes must equal the first bytes of canonical.
        assert_eq!(&canonical[..w.list_header.len()], &w.list_header[..]);
        // For a ~580-byte payload, the header should be [0xf9, hi, lo].
        assert_eq!(w.list_header[0], 0xf9);
    }
}
