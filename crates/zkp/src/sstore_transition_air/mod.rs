//! SSTORE pre/post state root transition AIR — roadmap #68 step 1.
//!
//! Bridges EVM SSTORE rows to the storage gadget AIR and (eventually)
//! the MPT inclusion AIR by committing a per-SSTORE state transition:
//!
//!   `(address, slot, old_value, new_value, pre_storage_root,
//!     post_storage_root)`
//!
//! Per row, the gadget commits:
//!   - `address[L0..L3]`           — 20-byte contract address as 4 LE u64 limbs
//!   - `slot[L0..L3]`              — U256 slot (4 LE u64 limbs)
//!   - `slot_be[0..32]`            — 32 BE bytes of slot
//!   - `old_value[L0..L3]`         — U256 pre-state value (4 LE u64 limbs)
//!   - `old_value_be[0..32]`       — 32 BE bytes
//!   - `new_value[L0..L3]`         — U256 post-state value (4 LE u64 limbs)
//!   - `new_value_be[0..32]`       — 32 BE bytes
//!   - `pre_storage_root[0..32]`   — 32-byte storage trie root BEFORE the SSTORE
//!   - `post_storage_root[0..32]`  — 32-byte storage trie root AFTER the SSTORE
//!   - `is_real`                   — 1 on real rows, 0 on padding
//!   - `is_clear`                  — 1 iff new_value == 0 (slot cleared)
//!   - `is_create`                 — 1 iff old_value == 0 && new_value != 0
//!   - `original_value_be[0..32]`  — 32 BE bytes of the slot value at the
//!                                    START of the transaction (EIP-2200);
//!                                    bound algebraically to the
//!                                    `sstore_prepost_air` row's
//!                                    `original_value_be` view via the
//!                                    `make_sstore_prepost_to_sstore_transition_descriptor`
//!                                    cross-AIR LogUp.
//!   - `original_value[L0..L3]`    — 4 LE u64 limbs of the same value
//!
//! Row-local constraints (this commit):
//!   0:   is_real binary
//!   1:   is_clear binary
//!   2:   is_create binary
//!   3..7: slot limb decomp (4)
//!   7..11: old_value limb decomp (4)
//!   11..15: new_value limb decomp (4)
//!   15..19: is_clear * new_value_limb_j = 0 (4 equations binding
//!           is_clear=1 ⇒ new_value=0)
//!   19..23: is_create * old_value_limb_j = 0 (4 equations binding
//!           is_create=1 ⇒ old_value=0)
//!   23..27: original_value limb decomp (4) — binds the 4 LE limbs to
//!           the 32 BE bytes of original_value.
//!
//! 8-bit range checks: slot_be, old_value_be, new_value_be,
//! pre_storage_root, post_storage_root, original_value_be → 32 * 6 =
//! 192 byte cols.
//!
//! ## Soundness gaps (deferred follow-ups)
//!
//! 1. **`pre_storage_root` and `post_storage_root` are oracles** in this
//!    AIR. Closing them requires cross-AIR LogUp linkages to two MPT
//!    inclusion AIRs (one per root) plus the storage gadget AIR
//!    enforcing that `(slot, old_value)` is included under
//!    `pre_storage_root` and `(slot, new_value)` is included under
//!    `post_storage_root`. The descriptors for the latter two are
//!    provided here. The MPT root binding follows the
//!    `make_storage_root_to_mpt_root_linkage_descriptor` pattern.
//!
//! 2. **`is_clear` and `is_create` are one-sided bindings**. We only
//!    enforce `is_flag = 1 ⇒ value matches` (the forward direction).
//!    The reverse (`value matches ⇒ flag = 1`) is NOT enforced, so a
//!    malicious prover could under-claim clears/creates. Downstream
//!    consumers that depend on flag truth (e.g. gas accounting for
//!    SSTORE refunds) must add the reverse direction via an
//!    inverse-witness pattern when it lands.
//!
//! 3. **The pre/post root chain across multiple SSTOREs is not bound
//!    here**. Sequential SSTOREs require an additional cross-row
//!    constraint `post_storage_root[r] == pre_storage_root[r+1]` when
//!    both rows are real and refer to the same contract address.
//!    Single-SSTORE soundness is unaffected.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_ADDR_L0: usize = 0;
pub const COL_ADDR_L1: usize = 1;
pub const COL_ADDR_L2: usize = 2;
pub const COL_ADDR_L3: usize = 3;

pub const COL_SLOT_L0: usize = 4;
pub const COL_SLOT_L1: usize = 5;
pub const COL_SLOT_L2: usize = 6;
pub const COL_SLOT_L3: usize = 7;

pub const COL_SLOT_BE_OFFSET: usize = 8; // 8..40

pub const COL_OLD_VALUE_L0: usize = 40;
pub const COL_OLD_VALUE_L1: usize = 41;
pub const COL_OLD_VALUE_L2: usize = 42;
pub const COL_OLD_VALUE_L3: usize = 43;

pub const COL_OLD_VALUE_BE_OFFSET: usize = 44; // 44..76

pub const COL_NEW_VALUE_L0: usize = 76;
pub const COL_NEW_VALUE_L1: usize = 77;
pub const COL_NEW_VALUE_L2: usize = 78;
pub const COL_NEW_VALUE_L3: usize = 79;

pub const COL_NEW_VALUE_BE_OFFSET: usize = 80; // 80..112

pub const COL_PRE_STORAGE_ROOT_OFFSET: usize = 112; // 112..144
pub const COL_POST_STORAGE_ROOT_OFFSET: usize = 144; // 144..176

pub const COL_IS_REAL: usize = 176;
pub const COL_IS_CLEAR: usize = 177;
pub const COL_IS_CREATE: usize = 178;

/// 32 BE bytes of `original_value` (slot value at the start of the tx).
/// Appended at the END of the layout to avoid shifting existing offsets.
pub const COL_ORIGINAL_VALUE_BE_OFFSET: usize = 179; // 179..211

pub const COL_ORIGINAL_VALUE_L0: usize = 211;
pub const COL_ORIGINAL_VALUE_L1: usize = 212;
pub const COL_ORIGINAL_VALUE_L2: usize = 213;
pub const COL_ORIGINAL_VALUE_L3: usize = 214;

pub const NUM_COLUMNS: usize = COL_ORIGINAL_VALUE_L3 + 1; // 215

/// Row-local constraint count.
///   3 binary flags + 4 slot decomp + 4 old_value decomp + 4 new_value
///   decomp + 4 is_clear bindings + 4 is_create bindings
///   + 4 original_value decomp = 27
pub const NUM_ROW_CONSTRAINTS: usize = 3 + 4 + 4 + 4 + 4 + 4 + 4;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness type ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct SstoreTransitionRow {
    /// 20-byte contract address.
    pub address: [u8; 20],
    /// U256 slot, BE.
    pub slot: [u8; 32],
    /// U256 pre-SSTORE value at this slot, BE.
    pub old_value: [u8; 32],
    /// U256 post-SSTORE value, BE.
    pub new_value: [u8; 32],
    /// U256 value at this slot at the START of the transaction, BE
    /// (EIP-2200 `original_value`). For the first SSTORE that touches
    /// this slot in the transaction this equals `old_value`; for
    /// subsequent SSTOREs it stays pinned to the tx-start value while
    /// `old_value` tracks the running pre-state.
    pub original_value: [u8; 32],
    /// 32-byte pre-state storage root.
    pub pre_storage_root: [u8; 32],
    /// 32-byte post-state storage root.
    pub post_storage_root: [u8; 32],
}

#[derive(Clone, Debug, Default)]
pub struct SstoreTransitionWitness {
    pub rows: Vec<SstoreTransitionRow>,
}

impl SstoreTransitionWitness {
    pub fn from_rows(rows: Vec<SstoreTransitionRow>) -> Self {
        Self { rows }
    }
}

/// Convert a 32-byte BE U256 into 4 LE u64 limbs.
fn be_bytes_to_le_limbs(bytes: [u8; 32]) -> [u64; 4] {
    [
        u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]),
        u64::from_be_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19],
            bytes[20], bytes[21], bytes[22], bytes[23],
        ]),
        u64::from_be_bytes([
            bytes[8],  bytes[9],  bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15],
        ]),
        u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
    ]
}

/// Convert a 20-byte BE address into 4 LE u64 limbs (top 96 bits = 0).
fn address_to_limbs(address: [u8; 20]) -> [u64; 4] {
    let mut full = [0u8; 32];
    full[12..32].copy_from_slice(&address);
    be_bytes_to_le_limbs(full)
}

fn is_all_zero(bytes: &[u8; 32]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &SstoreTransitionWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        // Address.
        let addr_limbs = address_to_limbs(row.address);
        for j in 0..4 {
            columns[COL_ADDR_L0 + j][i] = Scalar::from_u64(addr_limbs[j], curve);
        }

        // Slot (BE bytes + limbs).
        let slot_limbs = be_bytes_to_le_limbs(row.slot);
        for j in 0..4 {
            columns[COL_SLOT_L0 + j][i] = Scalar::from_u64(slot_limbs[j], curve);
        }
        for k in 0..32 {
            columns[COL_SLOT_BE_OFFSET + k][i] = Scalar::from_u64(row.slot[k] as u64, curve);
        }

        // Old value.
        let old_limbs = be_bytes_to_le_limbs(row.old_value);
        for j in 0..4 {
            columns[COL_OLD_VALUE_L0 + j][i] = Scalar::from_u64(old_limbs[j], curve);
        }
        for k in 0..32 {
            columns[COL_OLD_VALUE_BE_OFFSET + k][i] =
                Scalar::from_u64(row.old_value[k] as u64, curve);
        }

        // New value.
        let new_limbs = be_bytes_to_le_limbs(row.new_value);
        for j in 0..4 {
            columns[COL_NEW_VALUE_L0 + j][i] = Scalar::from_u64(new_limbs[j], curve);
        }
        for k in 0..32 {
            columns[COL_NEW_VALUE_BE_OFFSET + k][i] =
                Scalar::from_u64(row.new_value[k] as u64, curve);
        }

        // Original value (BE bytes + limbs).
        let orig_limbs = be_bytes_to_le_limbs(row.original_value);
        for j in 0..4 {
            columns[COL_ORIGINAL_VALUE_L0 + j][i] =
                Scalar::from_u64(orig_limbs[j], curve);
        }
        for k in 0..32 {
            columns[COL_ORIGINAL_VALUE_BE_OFFSET + k][i] =
                Scalar::from_u64(row.original_value[k] as u64, curve);
        }

        // Pre/post storage roots.
        for k in 0..32 {
            columns[COL_PRE_STORAGE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.pre_storage_root[k] as u64, curve);
            columns[COL_POST_STORAGE_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.post_storage_root[k] as u64, curve);
        }

        // Flags.
        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_CLEAR][i] =
            if is_all_zero(&row.new_value) { one.clone() } else { zero.clone() };
        columns[COL_IS_CREATE][i] =
            if is_all_zero(&row.old_value) && !is_all_zero(&row.new_value) {
                one.clone()
            } else {
                zero.clone()
            };
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

pub struct SstoreTransitionConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl SstoreTransitionConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// `(byte_index_in_be, weight)` for the `limb_idx`-th LE u64 limb of a
/// U256 stored as 32 BE bytes (byte 0 = MSB).
fn limb_decomp_targets(limb_idx: usize) -> [(usize, u64); 8] {
    let lo = match limb_idx {
        3 => 0,
        2 => 8,
        1 => 16,
        0 => 24,
        _ => unreachable!(),
    };
    let mut out = [(0usize, 0u64); 8];
    for k in 0..8 {
        out[k] = (lo + k, 1u64 << (8 * (7 - k)));
    }
    out
}

impl VmConstraintSystem for SstoreTransitionConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec![
            "is_real_binary".into(),
            "is_clear_binary".into(),
            "is_create_binary".into(),
        ];
        for j in 0..4 { labels.push(format!("slot_limb_{}_decomp", j)); }
        for j in 0..4 { labels.push(format!("old_value_limb_{}_decomp", j)); }
        for j in 0..4 { labels.push(format!("new_value_limb_{}_decomp", j)); }
        for j in 0..4 { labels.push(format!("is_clear_implies_new_value_{}_zero", j)); }
        for j in 0..4 { labels.push(format!("is_create_implies_old_value_{}_zero", j)); }
        for j in 0..4 { labels.push(format!("original_value_limb_{}_decomp", j)); }
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

        // Binary flags.
        for &flag_col in &[COL_IS_REAL, COL_IS_CLEAR, COL_IS_CREATE] {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[flag_col][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // Slot limb decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in targets {
                    let b = &columns[COL_SLOT_BE_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(pow, curve)));
                }
                c[r] = columns[COL_SLOT_L0 + limb_idx][r].sub(&sum);
            }
            out.push(c);
        }
        // Old value limb decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in targets {
                    let b = &columns[COL_OLD_VALUE_BE_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(pow, curve)));
                }
                c[r] = columns[COL_OLD_VALUE_L0 + limb_idx][r].sub(&sum);
            }
            out.push(c);
        }
        // New value limb decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in targets {
                    let b = &columns[COL_NEW_VALUE_BE_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(pow, curve)));
                }
                c[r] = columns[COL_NEW_VALUE_L0 + limb_idx][r].sub(&sum);
            }
            out.push(c);
        }

        // is_clear * new_value_limb_j = 0 (forward implication only).
        for limb_idx in 0..4 {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_IS_CLEAR][r]
                    .mul(&columns[COL_NEW_VALUE_L0 + limb_idx][r]);
            }
            out.push(c);
        }
        // is_create * old_value_limb_j = 0 (forward implication only).
        for limb_idx in 0..4 {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                c[r] = columns[COL_IS_CREATE][r]
                    .mul(&columns[COL_OLD_VALUE_L0 + limb_idx][r]);
            }
            out.push(c);
        }

        // Original value limb decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in targets {
                    let b = &columns[COL_ORIGINAL_VALUE_BE_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(pow, curve)));
                }
                c[r] = columns[COL_ORIGINAL_VALUE_L0 + limb_idx][r].sub(&sum);
            }
            out.push(c);
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

        // Binary flags.
        for &flag_col in &[COL_IS_REAL, COL_IS_CLEAR, COL_IS_CREATE] {
            let v = &col_evals[flag_col];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // Slot decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in targets {
                sum = sum.add(
                    &col_evals[COL_SLOT_BE_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(pow, curve)),
                );
            }
            let body = col_evals[COL_SLOT_L0 + limb_idx].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Old value decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in targets {
                sum = sum.add(
                    &col_evals[COL_OLD_VALUE_BE_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(pow, curve)),
                );
            }
            let body = col_evals[COL_OLD_VALUE_L0 + limb_idx].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // New value decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in targets {
                sum = sum.add(
                    &col_evals[COL_NEW_VALUE_BE_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(pow, curve)),
                );
            }
            let body = col_evals[COL_NEW_VALUE_L0 + limb_idx].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // is_clear ⇒ new_value zero.
        for limb_idx in 0..4 {
            let body = col_evals[COL_IS_CLEAR]
                .mul(&col_evals[COL_NEW_VALUE_L0 + limb_idx]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // is_create ⇒ old_value zero.
        for limb_idx in 0..4 {
            let body = col_evals[COL_IS_CREATE]
                .mul(&col_evals[COL_OLD_VALUE_L0 + limb_idx]);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
        }

        // Original value decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in targets {
                sum = sum.add(
                    &col_evals[COL_ORIGINAL_VALUE_BE_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(pow, curve)),
                );
            }
            let body = col_evals[COL_ORIGINAL_VALUE_L0 + limb_idx].sub(&sum);
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

        // Binary flags.
        for &flag_col in &[COL_IS_REAL, COL_IS_CLEAR, COL_IS_CREATE] {
            let v = &col_coeffs[flag_col];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // Slot decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in targets {
                let b = &col_coeffs[COL_SLOT_BE_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_SLOT_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // Old value decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in targets {
                let b = &col_coeffs[COL_OLD_VALUE_BE_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_OLD_VALUE_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // New value decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in targets {
                let b = &col_coeffs[COL_NEW_VALUE_BE_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_NEW_VALUE_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // is_clear * new_value_limb_j.
        for limb_idx in 0..4 {
            let body = poly_mul(
                &col_coeffs[COL_IS_CLEAR],
                &col_coeffs[COL_NEW_VALUE_L0 + limb_idx],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // is_create * old_value_limb_j.
        for limb_idx in 0..4 {
            let body = poly_mul(
                &col_coeffs[COL_IS_CREATE],
                &col_coeffs[COL_OLD_VALUE_L0 + limb_idx],
                curve,
            );
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }

        // Original value decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in targets {
                let b = &col_coeffs[COL_ORIGINAL_VALUE_BE_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(
                &col_coeffs[COL_ORIGINAL_VALUE_L0 + limb_idx],
                &sum,
                curve,
            );
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
        // 8-bit range checks on the 5 byte regions (32 bytes each):
        //   slot_be, old_value_be, new_value_be, pre_storage_root,
        //   post_storage_root → 160 byte cols total.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        let regions = [
            ("sstore_slot_be", COL_SLOT_BE_OFFSET),
            ("sstore_old_value_be", COL_OLD_VALUE_BE_OFFSET),
            ("sstore_new_value_be", COL_NEW_VALUE_BE_OFFSET),
            ("sstore_pre_storage_root", COL_PRE_STORAGE_ROOT_OFFSET),
            ("sstore_post_storage_root", COL_POST_STORAGE_ROOT_OFFSET),
            ("sstore_original_value_be", COL_ORIGINAL_VALUE_BE_OFFSET),
        ];
        for (label, base) in regions {
            for k in 0..32 {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
                        column_index: base + k,
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

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// **Pre-state binding** — sstore_transition_air ↔ storage_access_air on
/// the PRE-state side of the transition. The storage gadget AIR row
/// claims `(slot, old_value)` and (transitively, via its own
/// `make_storage_full_to_mpt_root_linkage_descriptor`) binds that to
/// inclusion under `pre_storage_root`.
///
/// 8-col tuple `(slot[L0..L3], old_value[L0..L3])`, gated by `is_real`
/// on both sides.
///
/// **Caveat (B-side over-gating)**: storage_access_air's `IS_REAL`
/// covers BOTH SLOAD and SSTORE accesses. The closure multiset on the
/// B side may include extra SLOAD tuples that aren't on the A side.
/// A strict pre-state binding requires the storage gadget to expose
/// an `is_pre_state_lookup` selector (or for the prover to schedule
/// one storage gadget row per pre-state slot read, separate from the
/// post-state write row).
pub fn make_sstore_to_storage_gadget_pre_descriptor(
    sstore_layer_index: usize,
    storage_gadget_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::storage_access_air as sg;
    let a_columns = vec![
        COL_SLOT_L0, COL_SLOT_L1, COL_SLOT_L2, COL_SLOT_L3,
        COL_OLD_VALUE_L0, COL_OLD_VALUE_L1, COL_OLD_VALUE_L2, COL_OLD_VALUE_L3,
    ];
    let b_columns = vec![
        sg::COL_SLOT_L0, sg::COL_SLOT_L1, sg::COL_SLOT_L2, sg::COL_SLOT_L3,
        sg::COL_VALUE_L0, sg::COL_VALUE_L1, sg::COL_VALUE_L2, sg::COL_VALUE_L3,
    ];
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sstore_pre_to_storage_gadget_v1".into(),
        a_layer_index: sstore_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: storage_gadget_layer_index,
        b_columns,
        b_selector_column: Some(sg::COL_IS_REAL),
    }
}

/// **Post-state binding** — sstore_transition_air ↔ storage_access_air
/// on the POST-state side of the transition. Same shape as the
/// pre-state descriptor but binds `new_value` instead of `old_value`,
/// matched against a storage gadget row whose `storage_root` is the
/// `post_storage_root` (must be wired separately via the gadget's
/// MPT root binding to a DIFFERENT MPT chain than the pre-state row).
///
/// 8-col tuple `(slot, new_value)`. Same B-side over-gating caveat.
pub fn make_sstore_to_storage_gadget_post_descriptor(
    sstore_layer_index: usize,
    storage_gadget_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::storage_access_air as sg;
    let a_columns = vec![
        COL_SLOT_L0, COL_SLOT_L1, COL_SLOT_L2, COL_SLOT_L3,
        COL_NEW_VALUE_L0, COL_NEW_VALUE_L1, COL_NEW_VALUE_L2, COL_NEW_VALUE_L3,
    ];
    let b_columns = vec![
        sg::COL_SLOT_L0, sg::COL_SLOT_L1, sg::COL_SLOT_L2, sg::COL_SLOT_L3,
        sg::COL_VALUE_L0, sg::COL_VALUE_L1, sg::COL_VALUE_L2, sg::COL_VALUE_L3,
    ];
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sstore_post_to_storage_gadget_v1".into(),
        a_layer_index: sstore_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: storage_gadget_layer_index,
        b_columns,
        b_selector_column: Some(sg::COL_IS_REAL),
    }
}

/// **EVM main ↔ sstore_transition_air** — binds EVM SSTORE rows
/// (gated by the caller-supplied `evm_sstore_selector_col`, typically
/// `metavm_evm::trace::COL_SEL_SSTORE = 242`) to this gadget's
/// `(slot, new_value)` tuple, gated by `is_real`.
///
/// The caller passes EVM-side column indices to keep `metavm-zkp`
/// independent of `metavm-evm` (the zkp crate doesn't depend on the
/// evm crate, so trace column constants must be supplied here). The
/// canonical wiring is:
///
/// ```ignore
/// use metavm_evm::trace::{
///     COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
///     COL_INPUT1_L0, COL_INPUT1_L1, COL_INPUT1_L2, COL_INPUT1_L3,
///     COL_SEL_SSTORE,
/// };
/// let desc = make_sstore_to_evm_main_descriptor(
///     evm_layer, sstore_layer,
///     [COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3],
///     [COL_INPUT1_L0, COL_INPUT1_L1, COL_INPUT1_L2, COL_INPUT1_L3],
///     COL_SEL_SSTORE,
/// );
/// ```
///
/// **Caveat**: this binds only the EVM-side `(slot, new_value)` —
/// the `old_value`, `pre_storage_root`, and `post_storage_root`
/// columns are oracles in this AIR (see module-level soundness gap
/// notes). The post_storage_root specifically comes from the inspector
/// in the EVM main trace as a side-channel oracle and is NOT (yet)
/// exposed as a dedicated EVM main column.
pub fn make_sstore_to_evm_main_descriptor(
    evm_layer_index: usize,
    sstore_layer_index: usize,
    evm_slot_input_cols: [usize; 4],
    evm_new_value_input_cols: [usize; 4],
    evm_sstore_selector_col: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = evm_slot_input_cols
        .iter()
        .chain(evm_new_value_input_cols.iter())
        .copied()
        .collect();
    let b_columns = vec![
        COL_SLOT_L0, COL_SLOT_L1, COL_SLOT_L2, COL_SLOT_L3,
        COL_NEW_VALUE_L0, COL_NEW_VALUE_L1, COL_NEW_VALUE_L2, COL_NEW_VALUE_L3,
    ];
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_sstore_to_sstore_transition_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(evm_sstore_selector_col),
        b_layer_index: sstore_layer_index,
        b_columns,
        b_selector_column: Some(COL_IS_REAL),
    }
}

// ─── Witness extractor ────────────────────────────────────────────────

/// Build a witness row from the host-side SSTORE transition oracle
/// (`crate::sstore_transition::SstoreTransition`) plus the address /
/// old value / pre-root the caller has on hand (these don't appear in
/// the bare `SstoreTransition` struct; they come from the EVM
/// inspector side-channel).
///
/// Symmetric to `crates/evm/src/storage_access.rs ::
/// extract_storage_accesses_from_trace` — the EVM crate is the natural
/// home for a trace-walking extractor (it has `EvmTraceColumns`), but
/// this helper assembles individual rows from already-extracted oracle
/// data. The matching trace-walking extractor in the EVM crate will be
/// `extract_sstore_transitions_from_trace` (deferred).
pub fn from_evm_trace_sstore(
    address: [u8; 20],
    old_value: [u8; 32],
    original_value: [u8; 32],
    pre_storage_root: [u8; 32],
    transition: &crate::sstore_transition::SstoreTransition,
) -> SstoreTransitionRow {
    SstoreTransitionRow {
        address,
        slot: transition.slot,
        old_value,
        new_value: transition.new_value,
        original_value,
        pre_storage_root,
        post_storage_root: transition.post_storage_root,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sstore_transition::build_single_slot_sstore_transition;

    fn sample_row() -> SstoreTransitionRow {
        let mut slot = [0u8; 32]; slot[31] = 1;
        let mut new_value = [0u8; 32]; new_value[31] = 0x42;
        let old_value = [0u8; 32]; // empty pre-state
        let original_value = [0u8; 32]; // first SSTORE → original = pre = 0
        let post = build_single_slot_sstore_transition(slot, new_value);
        SstoreTransitionRow {
            address: [0xab; 20],
            slot,
            old_value,
            new_value,
            original_value,
            pre_storage_root: [0u8; 32], // empty pre-state (oracle)
            post_storage_root: post.post_storage_root,
        }
    }

    #[test]
    fn trace_builder_populates_columns_and_flags() {
        let row = sample_row();
        let w = SstoreTransitionWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);

        // slot LSB = 1.
        assert_eq!(trace.columns[COL_SLOT_L0].evaluations[0].to_u64(), 1);
        // new_value LSB = 0x42.
        assert_eq!(trace.columns[COL_NEW_VALUE_L0].evaluations[0].to_u64(), 0x42);
        // old_value all-zero.
        assert_eq!(trace.columns[COL_OLD_VALUE_L0].evaluations[0].to_u64(), 0);
        // is_real = 1, is_clear = 0 (new_value != 0), is_create = 1
        // (old_value == 0 && new_value != 0).
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_IS_CLEAR].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_IS_CREATE].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = SstoreTransitionWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SstoreTransitionConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
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
    fn is_clear_detects_zero_new_value() {
        // Set new_value to all-zero (slot cleared); is_clear should be 1.
        let mut slot = [0u8; 32]; slot[31] = 1;
        let old_value = {
            let mut o = [0u8; 32];
            o[31] = 0x99; // had non-zero before
            o
        };
        let new_value = [0u8; 32];
        let post = build_single_slot_sstore_transition(slot, new_value);
        let w = SstoreTransitionWitness::from_rows(vec![SstoreTransitionRow {
            address: [0xab; 20],
            slot,
            old_value,
            new_value,
            original_value: old_value, // first SSTORE: original == pre
            pre_storage_root: [0xaa; 32], // oracle
            post_storage_root: post.post_storage_root,
        }]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.columns[COL_IS_CLEAR].evaluations[0].to_u64(), 1);
        // Not a create (old_value != 0).
        assert_eq!(trace.columns[COL_IS_CREATE].evaluations[0].to_u64(), 0);

        // Constraints should still be zero (is_clear * new_value = 0
        // holds because new_value is zero).
        let cs = SstoreTransitionConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for col in &results {
            for v in col { assert!(v.is_zero()); }
        }
    }

    #[test]
    fn tampered_new_value_breaks_limb_decomp() {
        let w = SstoreTransitionWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper the LSB byte (BE byte 31) of new_value WITHOUT
        // updating the limb column.
        cols[COL_NEW_VALUE_BE_OFFSET + 31][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = SstoreTransitionConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // new_value_limb_0 is at index 3 (binary flags) + 4 (slot)
        // + 4 (old_value) + 0 = 11.
        let new_value_limb_0_idx = 3 + 4 + 4 + 0;
        assert!(
            !results[new_value_limb_0_idx][0].is_zero(),
            "new_value limb decomp must fire on tampered byte",
        );
    }

    #[test]
    fn is_clear_lies_break_constraint_when_new_value_nonzero() {
        // Honest is_clear = 0 (new_value = 0x42); flip to 1 → the
        // "is_clear * new_value_limb_0 = 0" constraint must fire.
        let w = SstoreTransitionWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_CLEAR][0] = Scalar::one(CurveType::Bls48581);
        let cs = SstoreTransitionConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // is_clear * new_value_limb_0 is at index
        // 3 (binary) + 4+4+4 (decomp) + 0 = 15.
        let idx = 3 + 4 + 4 + 4 + 0;
        assert!(
            !results[idx][0].is_zero(),
            "is_clear lie should fire forward-implication constraint",
        );
    }

    #[test]
    fn is_create_binary_fires_on_nonbinary() {
        let w = SstoreTransitionWitness::from_rows(vec![sample_row()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_CREATE][0] = Scalar::from_u64(5, CurveType::Bls48581);
        let cs = SstoreTransitionConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // is_create_binary is at index 2.
        assert!(!results[2][0].is_zero(), "is_create binary must fire");
    }

    #[test]
    fn sstore_to_storage_gadget_pre_descriptor_well_formed() {
        let desc = make_sstore_to_storage_gadget_pre_descriptor(0, 1);
        assert_eq!(desc.label, "sstore_pre_to_storage_gadget_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 8);
        assert_eq!(desc.b_columns.len(), 8);
        // A side: slot limbs + old_value limbs.
        assert_eq!(desc.a_columns[0], COL_SLOT_L0);
        assert_eq!(desc.a_columns[4], COL_OLD_VALUE_L0);
        // B side: storage_access_air slot + value.
        assert_eq!(desc.b_columns[0], crate::storage_access_air::COL_SLOT_L0);
        assert_eq!(desc.b_columns[4], crate::storage_access_air::COL_VALUE_L0);
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::storage_access_air::COL_IS_REAL),
        );
    }

    #[test]
    fn sstore_to_storage_gadget_post_descriptor_well_formed() {
        let desc = make_sstore_to_storage_gadget_post_descriptor(0, 1);
        assert_eq!(desc.label, "sstore_post_to_storage_gadget_v1");
        assert_eq!(desc.a_columns.len(), 8);
        assert_eq!(desc.b_columns.len(), 8);
        // A side: slot limbs + new_value limbs.
        assert_eq!(desc.a_columns[0], COL_SLOT_L0);
        assert_eq!(desc.a_columns[4], COL_NEW_VALUE_L0);
    }

    #[test]
    fn sstore_to_evm_main_descriptor_well_formed() {
        // Mimic metavm-evm's column constants (COL_INPUT0_L0..L3 = 4..7,
        // COL_INPUT1_L0..L3 = 8..11, COL_SEL_SSTORE = 242).
        let desc = make_sstore_to_evm_main_descriptor(
            0, 1,
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            242,
        );
        assert_eq!(desc.label, "evm_sstore_to_sstore_transition_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns, vec![4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(desc.a_selector_column, Some(242));
        assert_eq!(desc.b_columns.len(), 8);
        assert_eq!(desc.b_columns[0], COL_SLOT_L0);
        assert_eq!(desc.b_columns[4], COL_NEW_VALUE_L0);
        assert_eq!(desc.b_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn from_evm_trace_sstore_assembles_row() {
        let mut slot = [0u8; 32]; slot[31] = 7;
        let mut new_value = [0u8; 32]; new_value[31] = 0xAB;
        let transition = build_single_slot_sstore_transition(slot, new_value);
        let mut old_value = [0u8; 32]; old_value[31] = 0x33;
        let mut original_value = [0u8; 32]; original_value[31] = 0x11;
        let pre_storage_root = [0x77u8; 32];
        let row = from_evm_trace_sstore(
            [0xde; 20],
            old_value,
            original_value,
            pre_storage_root,
            &transition,
        );
        assert_eq!(row.address, [0xde; 20]);
        assert_eq!(row.slot, slot);
        assert_eq!(row.old_value, old_value);
        assert_eq!(row.new_value, new_value);
        assert_eq!(row.original_value, original_value);
        assert_eq!(row.pre_storage_root, pre_storage_root);
        assert_eq!(row.post_storage_root, transition.post_storage_root);
    }

    #[test]
    fn original_value_limb_decomp_populated_and_binds() {
        // Distinct original_value (≠ old_value, ≠ new_value) — tests
        // that the new column is fully wired: trace populated, limb
        // decomp constraint zero on honest, fires on tamper.
        let mut slot = [0u8; 32]; slot[31] = 0x05;
        let mut new_value = [0u8; 32]; new_value[31] = 0xAA;
        let mut old_value = [0u8; 32]; old_value[31] = 0x55;
        // original_value populates multiple BE byte positions to
        // exercise all 4 limb decomp constraints non-trivially.
        let mut original_value = [0u8; 32];
        original_value[0] = 0x11;   // MSB → limb 3
        original_value[8] = 0x22;   // → limb 2
        original_value[16] = 0x33;  // → limb 1
        original_value[31] = 0x44;  // LSB → limb 0
        let post = build_single_slot_sstore_transition(slot, new_value);
        let w = SstoreTransitionWitness::from_rows(vec![SstoreTransitionRow {
            address: [0xcd; 20],
            slot,
            old_value,
            new_value,
            original_value,
            pre_storage_root: [0xaa; 32],
            post_storage_root: post.post_storage_root,
        }]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);

        // BE bytes populated.
        assert_eq!(
            trace.columns[COL_ORIGINAL_VALUE_BE_OFFSET + 0].evaluations[0].to_u64(),
            0x11,
        );
        assert_eq!(
            trace.columns[COL_ORIGINAL_VALUE_BE_OFFSET + 31].evaluations[0].to_u64(),
            0x44,
        );
        // LE limbs populated: limb 0 = LSB-bearing limb.
        assert_eq!(
            trace.columns[COL_ORIGINAL_VALUE_L0].evaluations[0].to_u64(),
            0x44,
        );

        // Honest constraints all zero.
        let cs = SstoreTransitionConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&cr, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} row {} expected zero", i, r);
            }
        }

        // Tamper the LSB byte of original_value WITHOUT updating limb 0;
        // original_value_limb_0_decomp (constraint 23) must fire.
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_ORIGINAL_VALUE_BE_OFFSET + 31][0] =
            Scalar::from_u64(0xff, CurveType::Bls48581);
        let cr2: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results2 = cs.evaluate_on_domain(&cr2, trace.num_rows);
        // original_value_limb_0_decomp index =
        //   3 (binary) + 4 (slot) + 4 (old) + 4 (new)
        //   + 4 (is_clear bindings) + 4 (is_create bindings) + 0 = 23.
        let original_limb_0_idx = 3 + 4 + 4 + 4 + 4 + 4 + 0;
        assert!(
            !results2[original_limb_0_idx][0].is_zero(),
            "tampered original_value byte must fire limb decomp",
        );
    }
}
