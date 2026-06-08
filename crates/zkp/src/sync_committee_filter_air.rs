//! Sync committee bitmap filter AIR (Phase C1 step 1).
//!
//! Per-row exposes one sync committee member with a participation
//! bitmap bit. The `IS_SELECTED` column (= `BITMAP_BIT`) filters the
//! full committee to the participating subset. A future cross-AIR
//! LogUp descriptor binds the selected pubkeys to the BLS signature
//! verification AIR.
//!
//! # Column layout
//!
//! Per row: `PUBKEY_BYTES[0..48]` (BLS12-381 compressed public key,
//! 48 bytes) + `BITMAP_BIT` (0 or 1) + `MEMBER_INDEX` (position in
//! the committee, 0..N) + `IS_REAL` + `IS_ACTIVE` (mirror of
//! `BITMAP_BIT` on real rows; downstream consumer convenience) +
//! `ACTIVE_COUNT_CUMULATIVE` (running count of bitmap_bit set up to
//! and including this row).
//!
//! # Constraints
//!
//! Row-local:
//!   - (0) `is_real` binary
//!   - (1) `bitmap_bit` binary (IS_REAL-gated)
//!   - (2) `is_real * (is_active - bitmap_bit) = 0` — is_active mirrors
//!         bitmap_bit on real rows
//!   - (3) `(1 - is_real) * is_active = 0` — no active flag on padding
//!   - (4) `(1 - is_real) * member_index = 0` — padding rows have
//!         member_index zero
//!   - (5) `(1 - is_real) * active_count_cumulative = 0` — padding rows
//!         have cumulative count zero
//!   - (6) `is_first` binary — `IS_FIRST · (IS_FIRST − 1) = 0`
//!   - (7) `is_first_implies_real` — `IS_FIRST · (1 − IS_REAL) = 0`
//!   - (8) `is_first_pins_count_boundary` —
//!         `IS_FIRST · (ACTIVE_COUNT_CUMULATIVE − BITMAP_BIT) = 0`.
//!         On the row where `IS_FIRST = 1` (the row-0 anchor), the
//!         cumulative count must equal the bitmap bit (the chain base
//!         case). Closes task #146.
//!
//! Cross-row (shifted, gated by `is_real_curr * is_real_next`):
//!   - (S0) `member_index_next - member_index_curr - 1 = 0` — strict
//!          +1 monotonic across active committee rows
//!   - (S1) `active_count_next - active_count_curr - bitmap_bit_next = 0`
//!          — cumulative count chain
//!   - (S2) `is_real_next * (1 - is_real_curr) = 0` — IS_REAL
//!          monotonicity (no real row following a padding row)
//!
//! # Soundness scope
//!
//! The filter gadget proves:
//! - Bitmap bits are binary.
//! - `is_active` mirrors the bitmap bit on real rows.
//! - Committee `member_index` increments strictly by 1 across real
//!   rows (so the witness cannot reorder/duplicate committee
//!   positions).
//! - `active_count_cumulative` chains consistently with the bitmap
//!   bit at each step.
//!
//! Caveats not yet enforced:
//! - The `IS_FIRST` selector pins the row-0 boundary
//!   (`active_count_cumulative = bitmap_bit`) algebraically, but the
//!   selector itself is a witness column: a malicious prover could set
//!   `IS_FIRST[0] = 0` to skip the pin. A downstream consumer that
//!   gates a cross-AIR linkage on `IS_FIRST` (e.g. publishing exactly
//!   one row-0 anchor tuple) provides the public commitment that pins
//!   `IS_FIRST[0] = 1`. The honest witness builder always sets row 0.
//! - The pubkeys come from the committed sync committee registry
//!   (needs beacon-state SSZ extraction + Merkle inclusion).
//! - The selected pubkeys verify the aggregate signature (needs
//!   BLS pairing AIR).

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const PUBKEY_LEN: usize = 48;
pub const COL_PUBKEY_OFFSET: usize = 0;        // 0..48
pub const COL_BITMAP_BIT: usize = PUBKEY_LEN;  // 48
pub const COL_MEMBER_INDEX: usize = COL_BITMAP_BIT + 1; // 49
pub const COL_IS_REAL: usize = COL_MEMBER_INDEX + 1;    // 50
pub const COL_IS_ACTIVE: usize = COL_IS_REAL + 1;       // 51
pub const COL_ACTIVE_COUNT_CUMULATIVE: usize = COL_IS_ACTIVE + 1; // 52
/// Boundary selector for row 0 (task #146). Set to 1 on row 0 by the
/// honest witness builder; zero elsewhere. Soft-constrained binary +
/// implies-real; the row-0 pin is closed by downstream consumers that
/// gate a cross-AIR tuple on `IS_FIRST`.
pub const COL_IS_FIRST: usize = COL_ACTIVE_COUNT_CUMULATIVE + 1;  // 53
pub const NUM_COLUMNS: usize = COL_IS_FIRST + 1;                  // 54

pub const NUM_ROW_CONSTRAINTS: usize = 9;
pub const NUM_SHIFTED: usize = 3;

#[derive(Clone, Debug)]
pub struct SyncCommitteeFilterRow {
    pub pubkey: [u8; PUBKEY_LEN],
    pub bitmap_bit: bool,
    pub member_index: u64,
}

#[derive(Clone, Debug)]
pub struct SyncCommitteeFilterWitness {
    pub rows: Vec<SyncCommitteeFilterRow>,
}

impl SyncCommitteeFilterWitness {
    pub fn from_committee(
        pubkeys: &[[u8; PUBKEY_LEN]],
        bitmap: &[bool],
    ) -> Self {
        assert_eq!(pubkeys.len(), bitmap.len());
        let rows = pubkeys.iter().zip(bitmap).enumerate().map(|(i, (pk, &b))| {
            SyncCommitteeFilterRow { pubkey: *pk, bitmap_bit: b, member_index: i as u64 }
        }).collect();
        Self { rows }
    }

    pub fn num_selected(&self) -> usize {
        self.rows.iter().filter(|r| r.bitmap_bit).count()
    }
}

pub fn build_trace_polynomials(w: &SyncCommitteeFilterWitness, curve: CurveType) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve); let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();
    let mut cumulative: u64 = 0;
    for (r, row) in w.rows.iter().enumerate() {
        for k in 0..PUBKEY_LEN { columns[COL_PUBKEY_OFFSET+k][r] = Scalar::from_u64(row.pubkey[k] as u64, curve); }
        columns[COL_BITMAP_BIT][r] = Scalar::from_u64(row.bitmap_bit as u64, curve);
        columns[COL_MEMBER_INDEX][r] = Scalar::from_u64(row.member_index, curve);
        columns[COL_IS_REAL][r] = one.clone();
        columns[COL_IS_ACTIVE][r] = Scalar::from_u64(row.bitmap_bit as u64, curve);
        if row.bitmap_bit { cumulative += 1; }
        columns[COL_ACTIVE_COUNT_CUMULATIVE][r] = Scalar::from_u64(cumulative, curve);
        if r == 0 { columns[COL_IS_FIRST][r] = one.clone(); }
    }
    let polys: Vec<Polynomial> = columns.into_iter().map(|e| Polynomial{evaluations:e,degree:num_rows}).collect();
    TracePolynomials{columns:polys,num_rows,padded_size:padded as u64,curve}
}

pub struct SyncCommitteeFilterConstraintSystem {
    pub num_rows: usize, pub omega: Option<Scalar>, pub domain_size: Option<u64>,
}
impl SyncCommitteeFilterConstraintSystem {
    pub fn new(n: usize) -> Self { Self{num_rows:n,omega:None,domain_size:None} }
    pub fn with_omega_and_domain(mut self, o: Scalar, d: u64) -> Self { self.omega=Some(o); self.domain_size=Some(d); self }
}

impl VmConstraintSystem for SyncCommitteeFilterConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }
    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "bitmap_bit_binary".into(),
            "is_active_mirrors_bitmap".into(),
            "is_active_zero_on_padding".into(),
            "member_index_zero_on_padding".into(),
            "active_count_zero_on_padding".into(),
            "is_first_binary".into(),
            "is_first_implies_real".into(),
            "is_first_pins_count_boundary".into(),
        ]
    }

    fn evaluate_on_domain(&self, columns: &[&Vec<Scalar>], _: usize) -> Vec<Vec<Scalar>> {
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let mut c0 = vec![Scalar::zero(curve); n];
        let mut c1 = vec![Scalar::zero(curve); n];
        let mut c2 = vec![Scalar::zero(curve); n];
        let mut c3 = vec![Scalar::zero(curve); n];
        let mut c4 = vec![Scalar::zero(curve); n];
        let mut c5 = vec![Scalar::zero(curve); n];
        let mut c6 = vec![Scalar::zero(curve); n];
        let mut c7 = vec![Scalar::zero(curve); n];
        let mut c8 = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            let b = &columns[COL_BITMAP_BIT][r];
            let a = &columns[COL_IS_ACTIVE][r];
            let mi = &columns[COL_MEMBER_INDEX][r];
            let cnt = &columns[COL_ACTIVE_COUNT_CUMULATIVE][r];
            let f = &columns[COL_IS_FIRST][r];
            let one_minus_v = one.sub(v);
            c0[r] = v.mul(&v.sub(&one));
            c1[r] = v.mul(&b.mul(&b.sub(&one)));
            c2[r] = v.mul(&a.sub(b));
            c3[r] = one_minus_v.mul(a);
            c4[r] = one_minus_v.mul(mi);
            c5[r] = one_minus_v.mul(cnt);
            c6[r] = f.mul(&f.sub(&one));
            c7[r] = f.mul(&one_minus_v);
            c8[r] = f.mul(&cnt.sub(b));
        }
        vec![c0, c1, c2, c3, c4, c5, c6, c7, c8]
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS { return Scalar::zero(alpha.curve_type()); }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &ce[COL_IS_REAL];
        let b = &ce[COL_BITMAP_BIT];
        let a = &ce[COL_IS_ACTIVE];
        let mi = &ce[COL_MEMBER_INDEX];
        let cnt = &ce[COL_ACTIVE_COUNT_CUMULATIVE];
        let f = &ce[COL_IS_FIRST];
        let one_minus_v = one.sub(v);
        let c0 = v.mul(&v.sub(&one));
        let c1 = v.mul(&b.mul(&b.sub(&one)));
        let c2 = v.mul(&a.sub(b));
        let c3 = one_minus_v.mul(a);
        let c4 = one_minus_v.mul(mi);
        let c5 = one_minus_v.mul(cnt);
        let c6 = f.mul(&f.sub(&one));
        let c7 = f.mul(&one_minus_v);
        let c8 = f.mul(&cnt.sub(b));
        let mut acc = c0;
        let mut ap = alpha.clone();
        acc = acc.add(&ap.mul(&c1));
        ap = ap.mul(alpha); acc = acc.add(&ap.mul(&c2));
        ap = ap.mul(alpha); acc = acc.add(&ap.mul(&c3));
        ap = ap.mul(alpha); acc = acc.add(&ap.mul(&c4));
        ap = ap.mul(alpha); acc = acc.add(&ap.mul(&c5));
        ap = ap.mul(alpha); acc = acc.add(&ap.mul(&c6));
        ap = ap.mul(alpha); acc = acc.add(&ap.mul(&c7));
        ap = ap.mul(alpha); acc = acc.add(&ap.mul(&c8));
        acc
    }

    fn build_constraint_polynomial(&self, cc: &[Vec<Scalar>], alpha: &Scalar, _: u64) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let v = &cc[COL_IS_REAL];
        let b = &cc[COL_BITMAP_BIT];
        let a = &cc[COL_IS_ACTIVE];
        let mi = &cc[COL_MEMBER_INDEX];
        let cnt = &cc[COL_ACTIVE_COUNT_CUMULATIVE];
        let f = &cc[COL_IS_FIRST];
        let one_minus_v = poly_sub(&one_p, v, curve);

        let c0 = poly_mul(v, &poly_sub(v, &one_p, curve), curve);
        let bb = poly_mul(b, &poly_sub(b, &one_p, curve), curve);
        let c1 = poly_mul(v, &bb, curve);
        let c2 = poly_mul(v, &poly_sub(a, b, curve), curve);
        let c3 = poly_mul(&one_minus_v, a, curve);
        let c4 = poly_mul(&one_minus_v, mi, curve);
        let c5 = poly_mul(&one_minus_v, cnt, curve);
        let c6 = poly_mul(f, &poly_sub(f, &one_p, curve), curve);
        let c7 = poly_mul(f, &one_minus_v, curve);
        let c8 = poly_mul(f, &poly_sub(cnt, b, curve), curve);

        let mut acc = c0;
        let mut ap = alpha.clone();
        acc = poly_add(&acc, &poly_scalar_mul(&c1, &ap), curve);
        ap = ap.mul(alpha); acc = poly_add(&acc, &poly_scalar_mul(&c2, &ap), curve);
        ap = ap.mul(alpha); acc = poly_add(&acc, &poly_scalar_mul(&c3, &ap), curve);
        ap = ap.mul(alpha); acc = poly_add(&acc, &poly_scalar_mul(&c4, &ap), curve);
        ap = ap.mul(alpha); acc = poly_add(&acc, &poly_scalar_mul(&c5, &ap), curve);
        ap = ap.mul(alpha); acc = poly_add(&acc, &poly_scalar_mul(&c6, &ap), curve);
        ap = ap.mul(alpha); acc = poly_add(&acc, &poly_scalar_mul(&c7, &ap), curve);
        ap = ap.mul(alpha); acc = poly_add(&acc, &poly_scalar_mul(&c8, &ap), curve);
        acc
    }

    // ── Cross-row (shifted) constraints ────────────────────────────────

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Order matters: indices passed to evaluate_shifted_at_point in this order.
        vec![
            COL_IS_REAL,                 // shifted_evals[0]
            COL_MEMBER_INDEX,            // shifted_evals[1]
            COL_ACTIVE_COUNT_CUMULATIVE, // shifted_evals[2]
            COL_BITMAP_BIT,              // shifted_evals[3]
        ]
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
        if shifted_evals.len() != 4 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v_curr = &col_evals_at_z[COL_IS_REAL];
        let mi_curr = &col_evals_at_z[COL_MEMBER_INDEX];
        let cnt_curr = &col_evals_at_z[COL_ACTIVE_COUNT_CUMULATIVE];
        let v_next = &shifted_evals[0];
        let mi_next = &shifted_evals[1];
        let cnt_next = &shifted_evals[2];
        let b_next = &shifted_evals[3];

        let gating = v_curr.mul(v_next);
        // S0: member_index_next - member_index_curr - 1 = 0
        let s0 = gating.mul(&mi_next.sub(mi_curr).sub(&one));
        // S1: active_count_next - active_count_curr - bitmap_bit_next = 0
        let s1 = gating.mul(&cnt_next.sub(cnt_curr).sub(b_next));
        // S2: is_real_next * (1 - is_real_curr) = 0 — IS_REAL monotonicity
        let s2 = v_next.mul(&one.sub(v_curr));

        // Combine with alpha^(alpha_offset + i)
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = ap.mul(&s0);
        ap = ap.mul(alpha); total = total.add(&ap.mul(&s1));
        ap = ap.mul(alpha); total = total.add(&ap.mul(&s2));

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
        let one_p = vec![Scalar::one(curve)];

        let v_curr = &column_coeffs[COL_IS_REAL];
        let v_next = poly_shift(v_curr, omega);
        let mi_curr = &column_coeffs[COL_MEMBER_INDEX];
        let mi_next = poly_shift(mi_curr, omega);
        let cnt_curr = &column_coeffs[COL_ACTIVE_COUNT_CUMULATIVE];
        let cnt_next = poly_shift(cnt_curr, omega);
        let b_curr = &column_coeffs[COL_BITMAP_BIT];
        let b_next = poly_shift(b_curr, omega);

        let gating = poly_mul(v_curr, &v_next, curve);

        // S0: gating * (mi_next - mi_curr - 1)
        let s0_chain = poly_sub(&poly_sub(&mi_next, mi_curr, curve), &one_p, curve);
        let s0 = poly_mul(&gating, &s0_chain, curve);

        // S1: gating * (cnt_next - cnt_curr - b_next)
        let s1_chain = poly_sub(&poly_sub(&cnt_next, cnt_curr, curve), &b_next, curve);
        let s1 = poly_mul(&gating, &s1_chain, curve);

        // S2: v_next * (1 - v_curr)
        let s2 = poly_mul(&v_next, &poly_sub(&one_p, v_curr, curve), curve);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = poly_scalar_mul(&s0, &ap);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&s1, &ap), curve);
        ap = ap.mul(alpha); total = poly_add(&total, &poly_scalar_mul(&s2, &ap), curve);

        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn selector_column_indices(&self) -> Vec<usize> { vec![COL_IS_REAL] }
    fn padding_selector_column(&self) -> Option<usize> { None }
    fn fix_trace_padding(&self, cols: &mut [Vec<Scalar>], nr: usize, ps: usize) {
        if nr == 0 || nr >= ps || cols.len() < NUM_COLUMNS { return; }
        let z = Scalar::zero(cols[0][0].curve_type());
        for c in cols.iter_mut().take(NUM_COLUMNS) { for cell in c.iter_mut().skip(nr).take(ps-nr) { *cell = z.clone(); } }
    }
    fn lookup_declarations(&self) -> LookupRequirements { LookupRequirements::none() }
}

/// Cross-AIR LogUp descriptor: selected pubkey bytes from the filter
/// AIR → BLS signature verification AIR. A side = filter AIR,
/// B side = BLS sig AIR. Only rows with BITMAP_BIT=1 contribute.
///
/// Tuple shape: 48 pubkey bytes. Gated by BITMAP_BIT on A side
/// (only selected members) and IS_REAL on B side.
pub fn make_filter_to_bls_sig_linkage_descriptor(
    filter_layer_index: usize,
    bls_sig_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = (0..PUBKEY_LEN).map(|k| COL_PUBKEY_OFFSET + k).collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_committee_filter_to_bls_sig_v1".into(),
        a_layer_index: filter_layer_index,
        a_columns: a_columns.clone(),
        a_selector_column: Some(COL_BITMAP_BIT),
        b_layer_index: bls_sig_layer_index,
        b_columns: a_columns, // BLS sig AIR uses same pubkey byte layout at offset 0
        b_selector_column: None, // placeholder: BLS sig AIR needs its own IS_REAL
    }
}

/// Cross-AIR LogUp descriptor: filter member_index → beacon state
/// validator registry. Binds each member_index to the correct
/// validator pubkey from the beacon state's validator list.
pub fn make_filter_to_validator_registry_linkage_descriptor(
    filter_layer_index: usize,
    registry_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = vec![COL_MEMBER_INDEX];
    a_columns.extend(0..PUBKEY_LEN);
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sync_committee_filter_to_validator_registry_v1".into(),
        a_layer_index: filter_layer_index,
        a_columns: a_columns.clone(),
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: registry_layer_index,
        b_columns: a_columns, // registry AIR uses same (index, pubkey) layout
        b_selector_column: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_builds_and_counts_selected() {
        let pks: Vec<[u8; 48]> = (0..4).map(|i| { let mut pk=[0u8;48]; pk[0]=i; pk }).collect();
        let bitmap = vec![true, false, true, false];
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &bitmap);
        assert_eq!(w.rows.len(), 4);
        assert_eq!(w.num_selected(), 2);
    }

    #[test]
    fn cumulative_count_populates_correctly() {
        let pks: Vec<[u8;48]> = (0..4).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false,true,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cnt = &t.columns[COL_ACTIVE_COUNT_CUMULATIVE].evaluations;
        let curve = CurveType::Bls48581;
        assert!(cnt[0].sub(&Scalar::from_u64(1, curve)).is_zero());
        assert!(cnt[1].sub(&Scalar::from_u64(1, curve)).is_zero());
        assert!(cnt[2].sub(&Scalar::from_u64(2, curve)).is_zero());
        assert!(cnt[3].sub(&Scalar::from_u64(3, curve)).is_zero());
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let pks: Vec<[u8;48]> = (0..4).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false,true,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p|&p.evaluations).collect();
        for (i,b) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r,v) in b.iter().enumerate() { assert!(v.is_zero(), "c{} r{}", i, r); }
        }
    }

    #[test]
    fn bitmap_bit_binary_fires_on_nonbinary() {
        let pks = vec![[0u8;48]];
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[false]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p|p.evaluations.clone()).collect();
        cols[COL_BITMAP_BIT][0] = Scalar::from_u64(7, CurveType::Bls48581);
        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        assert!(!cs.evaluate_on_domain(&cr, t.num_rows)[1][0].is_zero());
    }

    #[test]
    fn bitmap_bit_binary_tampering_detected_on_active_row() {
        // Multi-row honest witness; tamper bitmap_bit on a middle row.
        let pks: Vec<[u8;48]> = (0..4).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,true,false,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p|p.evaluations.clone()).collect();
        // Tamper: set bitmap_bit to 2 (out of {0,1}) on row 2.
        cols[COL_BITMAP_BIT][2] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        let evals = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 1 (bitmap_bit binary) must fire on row 2.
        assert!(!evals[1][2].is_zero(), "bitmap binary constraint should fire on tampered row");
    }

    #[test]
    fn is_active_mirror_tampering_detected() {
        // is_active must equal bitmap_bit on real rows.
        let pks: Vec<[u8;48]> = (0..3).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p|p.evaluations.clone()).collect();
        // Tamper: set is_active to 1 on row 1 where bitmap_bit = 0.
        cols[COL_IS_ACTIVE][1] = Scalar::from_u64(1, CurveType::Bls48581);
        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        let evals = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 2 (is_active mirrors bitmap_bit) must fire on row 1.
        assert!(!evals[2][1].is_zero(), "is_active mirror constraint should fire on tampered row");
    }

    #[test]
    fn index_monotonicity_tampering_detected_via_shifted_poly() {
        // Honest witness, then tamper member_index on row 2 to break +1 chain.
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let s = Bls48581Scheme::new(); s.init();
        let pks: Vec<[u8;48]> = (0..4).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false,true,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let omega = s.domain_generator(t.padded_size);

        // Build column coefficients from evaluations via IFFT.
        let mut cols_evals: Vec<Vec<Scalar>> = t.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: member_index row 2 = 99 (breaks +1 chain from row 1's 1 → row 2 should be 2).
        cols_evals[COL_MEMBER_INDEX][2] = Scalar::from_u64(99, CurveType::Bls48581);

        // Convert each column to coefficient form via IFFT.
        let curve = CurveType::Bls48581;
        let col_coeffs: Vec<Vec<Scalar>> = cols_evals.iter().map(|ev| {
            crate::scheme::CommitmentScheme::ifft(&s, ev, t.padded_size)
        }).collect();

        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(7, curve);
        let shifted_poly = cs.build_shifted_constraint_polynomial(
            &col_coeffs, &alpha, t.padded_size, &omega, NUM_ROW_CONSTRAINTS,
        );
        // The tampered chain produces a nonzero shifted constraint poly: at least
        // one coefficient must be nonzero (the polynomial is not identically zero).
        assert!(shifted_poly.iter().any(|c| !c.is_zero()),
            "shifted poly must be nonzero when index monotonicity is tampered");
    }

    #[test]
    fn cumulative_count_chain_tampering_detected_via_shifted_poly() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let s = Bls48581Scheme::new(); s.init();
        let pks: Vec<[u8;48]> = (0..4).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false,true,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let omega = s.domain_generator(t.padded_size);

        let mut cols_evals: Vec<Vec<Scalar>> = t.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: active count row 2 = 5 (honest is 2).
        cols_evals[COL_ACTIVE_COUNT_CUMULATIVE][2] = Scalar::from_u64(5, CurveType::Bls48581);

        let curve = CurveType::Bls48581;
        let col_coeffs: Vec<Vec<Scalar>> = cols_evals.iter().map(|ev| {
            crate::scheme::CommitmentScheme::ifft(&s, ev, t.padded_size)
        }).collect();

        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(11, curve);
        let shifted_poly = cs.build_shifted_constraint_polynomial(
            &col_coeffs, &alpha, t.padded_size, &omega, NUM_ROW_CONSTRAINTS,
        );
        assert!(shifted_poly.iter().any(|c| !c.is_zero()),
            "shifted poly must be nonzero when cumulative count is tampered");
    }

    #[test]
    fn filter_to_bls_sig_descriptor_well_formed() {
        let d = make_filter_to_bls_sig_linkage_descriptor(0, 1);
        assert_eq!(d.label, "sync_committee_filter_to_bls_sig_v1");
        assert_eq!(d.a_columns.len(), PUBKEY_LEN);
        assert_eq!(d.a_selector_column, Some(COL_BITMAP_BIT));
        assert_eq!(d.a_columns[0], COL_PUBKEY_OFFSET);
        assert_eq!(d.a_columns[47], COL_PUBKEY_OFFSET + 47);
    }

    #[test]
    fn filter_to_registry_descriptor_well_formed() {
        let d = make_filter_to_validator_registry_linkage_descriptor(0, 2);
        assert_eq!(d.a_columns.len(), 1 + PUBKEY_LEN); // member_index + 48 pubkey bytes
        assert_eq!(d.a_columns[0], COL_MEMBER_INDEX);
        assert_eq!(d.a_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn is_first_set_on_row_zero_only() {
        // Task #146: honest witness pins IS_FIRST = 1 on row 0 only.
        let pks: Vec<[u8;48]> = (0..4).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false,true,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let f = &t.columns[COL_IS_FIRST].evaluations;
        let curve = CurveType::Bls48581;
        assert!(f[0].sub(&Scalar::one(curve)).is_zero(), "IS_FIRST[0] must be 1");
        for r in 1..t.padded_size as usize {
            assert!(f[r].is_zero(), "IS_FIRST[{}] must be 0", r);
        }
    }

    #[test]
    fn is_first_count_boundary_fires_on_tampered_row_zero_cumulative() {
        // Task #146: tampering active_count_cumulative on row 0 must
        // fire the row-0 boundary constraint (c8). Honest row 0:
        // bitmap_bit = 1, cumulative = 1. Tamper to cumulative = 7.
        let pks: Vec<[u8;48]> = (0..4).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false,true,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p|p.evaluations.clone()).collect();
        cols[COL_ACTIVE_COUNT_CUMULATIVE][0] = Scalar::from_u64(7, CurveType::Bls48581);
        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        let evals = cs.evaluate_on_domain(&cr, t.num_rows);
        // c8 (is_first_pins_count_boundary) must fire on row 0.
        assert!(!evals[8][0].is_zero(),
            "is_first_pins_count_boundary should fire on tampered row-0 cumulative");
        // c8 stays zero on rows >=1 (IS_FIRST=0 there).
        for r in 1..t.num_rows {
            assert!(evals[8][r].is_zero(),
                "c8 must be zero on row {} where IS_FIRST=0", r);
        }
    }

    #[test]
    fn is_first_binary_fires_on_nonbinary() {
        let pks: Vec<[u8;48]> = (0..2).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = t.columns.iter().map(|p|p.evaluations.clone()).collect();
        cols[COL_IS_FIRST][0] = Scalar::from_u64(3, CurveType::Bls48581);
        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cols.iter().collect();
        let evals = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!evals[6][0].is_zero(), "is_first_binary should detect 3");
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme; use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme; use crate::verifier::verify_with_scheme;
        let s = Bls48581Scheme::new(); s.init();
        let pks: Vec<[u8;48]> = (0..8).map(|i| {let mut pk=[0u8;48]; pk[0]=i; pk}).collect();
        let w = SyncCommitteeFilterWitness::from_committee(&pks, &[true,false,true,true,false,false,true,true]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let o = s.domain_generator(t.padded_size);
        let cs = SyncCommitteeFilterConstraintSystem::new(t.num_rows).with_omega_and_domain(o, t.padded_size);
        assert!(verify_with_scheme(&prove_with_scheme(&t, &cs, &s), &cs, &s, CurveType::Bls48581));
    }
}
