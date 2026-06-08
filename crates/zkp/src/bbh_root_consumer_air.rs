//! BeaconBlockHeader root consumer AIR — Phase C3 step 3.
//!
//! Minimal pass-through AIR that consumes a 144-byte composite tuple
//! from [`crate::beacon_block_header_pair_air`]'s output descriptor:
//!
//! ```text
//! (claimed_root[32] || parent_root[32] || state_root[32] || body_root[32]
//!  || slot[8] || proposer_index[8])
//! ```
//!
//! This AIR serves as the template for downstream Phase C consumers —
//! finality AIR, validator HTR chain, sync committee binding, etc.
//! Each can replace this skeleton with its real constraint logic while
//! reusing the cross-AIR LogUp descriptor pattern this AIR establishes.
//!
//! # Soundness scope
//!
//! Per-row constraints:
//!   - `is_real` binary
//!
//! 144 byte range checks (LookupRequirements) on every byte column.
//!
//! Cross-AIR LogUp linkage to BBH-pair AIR: the multiset of this AIR's
//! `(claimed_root, fields)` tuples gated by `IS_REAL` equals the
//! multiset of BBH-pair AIR's same tuples gated by `IS_REAL`. Because
//! both sides enforce cross-row constancy of the 144-col tuple across
//! all real rows, multiset equality reduces to a single canonical
//! `(root, fields)` value shared between the two AIRs.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_CLAIMED_ROOT_OFFSET: usize = 0;          // 0..32
pub const COL_PARENT_ROOT_OFFSET: usize = 32;          // 32..64
pub const COL_STATE_ROOT_OFFSET: usize = 64;           // 64..96
pub const COL_BODY_ROOT_OFFSET: usize = 96;            // 96..128
pub const COL_SLOT_BYTE_OFFSET: usize = 128;           // 128..136
pub const COL_PROPOSER_INDEX_BYTE_OFFSET: usize = 136; // 136..144
pub const COL_IS_REAL: usize = 144;
pub const NUM_COLUMNS: usize = COL_IS_REAL + 1; // 145

pub const NUM_ROW_CONSTRAINTS: usize = 1; // is_real binary
pub const NUM_SHIFTED: usize = 6;          // cross-row constancy of 6 field groups

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct BbhRootConsumerRow {
    pub claimed_root: [u8; 32],
    pub parent_root: [u8; 32],
    pub state_root: [u8; 32],
    pub body_root: [u8; 32],
    pub slot: u64,
    pub proposer_index: u64,
}

#[derive(Clone, Debug, Default)]
pub struct BbhRootConsumerWitness {
    pub rows: Vec<BbhRootConsumerRow>,
}

impl BbhRootConsumerWitness {
    /// Build a witness with `num_rows` identical rows holding the given
    /// BBH composite tuple. Matches the BBH-pair AIR's cross-row
    /// constancy so the LogUp closure equality holds.
    pub fn from_bbh_tuple(
        num_rows: usize,
        claimed_root: [u8; 32],
        parent_root: [u8; 32],
        state_root: [u8; 32],
        body_root: [u8; 32],
        slot: u64,
        proposer_index: u64,
    ) -> Self {
        let row = BbhRootConsumerRow {
            claimed_root,
            parent_root,
            state_root,
            body_root,
            slot,
            proposer_index,
        };
        Self { rows: vec![row; num_rows] }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BbhRootConsumerWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..32 {
            columns[COL_CLAIMED_ROOT_OFFSET + k][i] = Scalar::from_u64(row.claimed_root[k] as u64, curve);
            columns[COL_PARENT_ROOT_OFFSET + k][i] = Scalar::from_u64(row.parent_root[k] as u64, curve);
            columns[COL_STATE_ROOT_OFFSET + k][i] = Scalar::from_u64(row.state_root[k] as u64, curve);
            columns[COL_BODY_ROOT_OFFSET + k][i] = Scalar::from_u64(row.body_root[k] as u64, curve);
        }
        let slot_bytes = row.slot.to_le_bytes();
        let pi_bytes = row.proposer_index.to_le_bytes();
        for k in 0..8 {
            columns[COL_SLOT_BYTE_OFFSET + k][i] = Scalar::from_u64(slot_bytes[k] as u64, curve);
            columns[COL_PROPOSER_INDEX_BYTE_OFFSET + k][i] = Scalar::from_u64(pi_bytes[k] as u64, curve);
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

pub struct BbhRootConsumerConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BbhRootConsumerConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for BbhRootConsumerConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec!["is_real_binary".into()]
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
        let mut bin = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            bin[r] = v.mul(&v.sub(&one));
        }
        vec![bin]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let one = Scalar::one(alpha.curve_type());
        let v = &col_evals[COL_IS_REAL];
        v.mul(&v.sub(&one))
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let _ = alpha;
        let curve = col_coeffs[0][0].curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        poly_mul(v, &v_m1, curve)
    }

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // ω·z evaluations: IS_REAL_NEXT (gating) + 144 field bytes.
        let mut cols = Vec::with_capacity(1 + 144);
        cols.push(COL_IS_REAL);
        for k in 0..32 { cols.push(COL_CLAIMED_ROOT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_PARENT_ROOT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_STATE_ROOT_OFFSET + k); }
        for k in 0..32 { cols.push(COL_BODY_ROOT_OFFSET + k); }
        for k in 0..8 { cols.push(COL_SLOT_BYTE_OFFSET + k); }
        for k in 0..8 { cols.push(COL_PROPOSER_INDEX_BYTE_OFFSET + k); }
        cols
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
        if shifted_evals.len() != 1 + 144 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let v = &col_evals_at_z[COL_IS_REAL];
        let v_next = &shifted_evals[0];
        let gating = v.mul(v_next);

        let make_body = |off_z: usize, shifted_base: usize, n: usize| -> Scalar {
            let mut acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..n {
                let cur = &col_evals_at_z[off_z + k];
                let nxt = &shifted_evals[shifted_base + k];
                acc = acc.add(&bp.mul(&nxt.sub(cur)));
                bp = bp.mul(alpha);
            }
            acc
        };
        let b_root = gating.mul(&make_body(COL_CLAIMED_ROOT_OFFSET, 1, 32));
        let b_parent = gating.mul(&make_body(COL_PARENT_ROOT_OFFSET, 33, 32));
        let b_state = gating.mul(&make_body(COL_STATE_ROOT_OFFSET, 65, 32));
        let b_body = gating.mul(&make_body(COL_BODY_ROOT_OFFSET, 97, 32));
        let b_slot = gating.mul(&make_body(COL_SLOT_BYTE_OFFSET, 129, 8));
        let b_pi = gating.mul(&make_body(COL_PROPOSER_INDEX_BYTE_OFFSET, 137, 8));

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = ap.mul(&b_root);
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&b_parent));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&b_state));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&b_body));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&b_slot));
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&b_pi));

        total.mul(&z.sub(omega_n_minus_1))
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

        let make_body = |off: usize, n: usize| -> Vec<Scalar> {
            let mut acc = vec![Scalar::zero(curve)];
            let mut bp = Scalar::one(curve);
            for k in 0..n {
                let cur = &column_coeffs[off + k];
                let nxt = poly_shift(cur, omega);
                let diff = poly_sub(&nxt, cur, curve);
                acc = poly_add(&acc, &poly_scalar_mul(&diff, &bp), curve);
                bp = bp.mul(alpha);
            }
            acc
        };
        let b_root = poly_mul(&gating, &make_body(COL_CLAIMED_ROOT_OFFSET, 32), curve);
        let b_parent = poly_mul(&gating, &make_body(COL_PARENT_ROOT_OFFSET, 32), curve);
        let b_state = poly_mul(&gating, &make_body(COL_STATE_ROOT_OFFSET, 32), curve);
        let b_body = poly_mul(&gating, &make_body(COL_BODY_ROOT_OFFSET, 32), curve);
        let b_slot = poly_mul(&gating, &make_body(COL_SLOT_BYTE_OFFSET, 8), curve);
        let b_pi = poly_mul(&gating, &make_body(COL_PROPOSER_INDEX_BYTE_OFFSET, 8), curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = poly_scalar_mul(&b_root, &ap);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&b_parent, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&b_state, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&b_body, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&b_slot, &ap), curve);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&b_pi, &ap), curve);

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
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
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        let groups: [(&str, usize, usize); 6] = [
            ("claimed_root", COL_CLAIMED_ROOT_OFFSET, 32),
            ("parent_root", COL_PARENT_ROOT_OFFSET, 32),
            ("state_root", COL_STATE_ROOT_OFFSET, 32),
            ("body_root", COL_BODY_ROOT_OFFSET, 32),
            ("slot", COL_SLOT_BYTE_OFFSET, 8),
            ("proposer_index", COL_PROPOSER_INDEX_BYTE_OFFSET, 8),
        ];
        for (name, off, n) in groups {
            for k in 0..n {
                declarations.push((
                    LookupDeclaration {
                        label: format!("bbh_consumer_{}_{}_8bit", name, k),
                        column_index: off + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp linkage ──────────────────────────────────────────

/// Cross-AIR LogUp descriptor binding BBH-pair AIR's 144-col output
/// tuple to this consumer AIR's matching 144-col input. The descriptor
/// uses both AIRs' `IS_REAL` selector for gating.
///
/// Soundness: under cross-row constancy on both sides, the multiset of
/// 144-byte tuples reduces to one canonical value, enforcing that the
/// BBH-pair AIR's computed root + the 5 input fields equal this
/// consumer's claimed values.
pub fn make_bbh_pair_to_root_consumer_linkage_descriptor(
    bbh_pair_layer_index: usize,
    consumer_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::beacon_block_header_pair_air as bbh;

    let a_columns = bbh::make_bbh_pair_root_output_column_indices();
    let mut b_columns = Vec::with_capacity(144);
    for k in 0..32 { b_columns.push(COL_CLAIMED_ROOT_OFFSET + k); }
    for k in 0..32 { b_columns.push(COL_PARENT_ROOT_OFFSET + k); }
    for k in 0..32 { b_columns.push(COL_STATE_ROOT_OFFSET + k); }
    for k in 0..32 { b_columns.push(COL_BODY_ROOT_OFFSET + k); }
    for k in 0..8 { b_columns.push(COL_SLOT_BYTE_OFFSET + k); }
    for k in 0..8 { b_columns.push(COL_PROPOSER_INDEX_BYTE_OFFSET + k); }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "bbh_pair_root_to_consumer_v1".into(),
        a_layer_index: bbh_pair_layer_index,
        a_columns,
        a_selector_column: Some(bbh::COL_IS_REAL),
        b_layer_index: consumer_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon::BeaconBlockHeader;
    use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;

    fn sample_header() -> BeaconBlockHeader {
        BeaconBlockHeader {
            slot: 99999,
            proposer_index: 13,
            parent_root: [0x55u8; 32],
            state_root: [0x66u8; 32],
            body_root: [0x77u8; 32],
        }
    }

    fn make_consumer_witness_from_header(num_rows: usize) -> BbhRootConsumerWitness {
        let h = sample_header();
        let w = BeaconBlockHeaderHtrWitness::from_header(h);
        BbhRootConsumerWitness::from_bbh_tuple(
            num_rows,
            w.root,
            h.parent_root,
            h.state_root,
            h.body_root,
            h.slot,
            h.proposer_index,
        )
    }

    #[test]
    fn trace_builder_populates_all_field_groups() {
        let w = make_consumer_witness_from_header(7);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.num_rows, 7);
        let h = sample_header();
        let bbh = BeaconBlockHeaderHtrWitness::from_header(h);
        for r in 0..7 {
            for k in 0..32 {
                assert_eq!(
                    trace.columns[COL_CLAIMED_ROOT_OFFSET + k].evaluations[r].to_u64(),
                    bbh.root[k] as u64,
                );
                assert_eq!(
                    trace.columns[COL_PARENT_ROOT_OFFSET + k].evaluations[r].to_u64(),
                    h.parent_root[k] as u64,
                );
            }
            let slot_bytes = h.slot.to_le_bytes();
            for k in 0..8 {
                assert_eq!(
                    trace.columns[COL_SLOT_BYTE_OFFSET + k].evaluations[r].to_u64(),
                    slot_bytes[k] as u64,
                );
            }
        }
    }

    #[test]
    fn is_real_binary_zero_on_honest_witness() {
        let w = make_consumer_witness_from_header(7);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BbhRootConsumerConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for v in &res[0] {
            assert!(v.is_zero());
        }
    }

    #[test]
    fn linkage_descriptor_shape() {
        let d = make_bbh_pair_to_root_consumer_linkage_descriptor(0, 1);
        assert_eq!(d.a_columns.len(), 144);
        assert_eq!(d.b_columns.len(), 144);
        assert_eq!(d.label, "bbh_pair_root_to_consumer_v1");
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
        let w = make_consumer_witness_from_header(7);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = BbhRootConsumerConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }

    /// 3-AIR `joint_prove` composition: BBH-pair ↔ Sha256Extract (96-byte
    /// per-invocation linkage) AND BBH-pair ↔ Consumer (144-byte composite
    /// linkage). Demonstrates that BBH HTR's output is now algebraically
    /// available for downstream consumers.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove with 2 linkages (~5-10 min release)"]
    fn joint_prove_bbh_sha256_consumer_3_air() {
        use crate::beacon_block_header_pair_air as bbh;
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::sha256_extract::{
            self as se, build_trace_polynomials as build_se_trace,
            Sha256ExtractConstraintSystem,
        };
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let h = sample_header();
        let bbh_w = BeaconBlockHeaderHtrWitness::from_header(h);

        // BBH-pair AIR (layer 0).
        let bbh_trace = bbh::build_trace_polynomials(&bbh_w, curve);
        let bbh_omega = scheme.domain_generator(bbh_trace.padded_size);
        let bbh_cs = bbh::BeaconBlockHeaderPairConstraintSystem::new(bbh_trace.num_rows)
            .with_omega_and_domain(bbh_omega, bbh_trace.padded_size);

        // Sha256Extract (layer 1).
        let pairs: Vec<(crate::ssz::Chunk, crate::ssz::Chunk)> = bbh_w
            .invocations
            .iter()
            .map(|inv| (inv.left, inv.right))
            .collect();
        let se_w = se::Sha256ExtractWitness::from_pair_inputs(&pairs);
        let se_trace = build_se_trace(&se_w, curve);
        let se_omega = scheme.domain_generator(se_trace.padded_size);
        let se_cs = Sha256ExtractConstraintSystem::new(se_trace.num_rows)
            .with_omega_and_domain(se_omega, se_trace.padded_size);

        // Consumer (layer 2). Tuple must equal BBH's CLAIMED_ROOT etc.
        let consumer_w = BbhRootConsumerWitness::from_bbh_tuple(
            bbh_trace.num_rows,
            bbh_w.root,
            h.parent_root,
            h.state_root,
            h.body_root,
            h.slot,
            h.proposer_index,
        );
        let consumer_trace = build_trace_polynomials(&consumer_w, curve);
        let consumer_omega = scheme.domain_generator(consumer_trace.padded_size);
        let consumer_cs = BbhRootConsumerConstraintSystem::new(consumer_trace.num_rows)
            .with_omega_and_domain(consumer_omega, consumer_trace.padded_size);

        let l1 = bbh::make_beacon_block_header_pair_to_sha256_extract_linkage_descriptor(0, 1);
        let l2 = make_bbh_pair_to_root_consumer_linkage_descriptor(0, 2);
        let linkages = vec![l1.clone(), l2.clone()];

        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> = vec![
            (&bbh_trace, &bbh_cs),
            (&se_trace, &se_cs),
            (&consumer_trace, &consumer_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("3-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 3);
        assert_eq!(ext.linkage_proofs.len(), 2);
        for lp in &ext.linkage_proofs {
            eprintln!("[diag] linkage label={} closure_match={}",
                lp.label, lp.closure_a == lp.closure_b);
            assert_eq!(lp.closure_a, lp.closure_b);
        }

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&bbh_cs, &se_cs, &consumer_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must accept honest 3-AIR composition",
        );
    }
}
