//! Variable-length u64 RLP encoding witness scaffolding (step 0).
//!
//! Host-side witness builder for the algebraic u64 RLP encoding gadget.
//! The witness lays out every column the future constraint system
//! ([step 1+] will validate, so an honest prover already has a
//! consistent trace ready and downstream consumers can wire to the
//! column constants today.
//!
//! # RLP encoding of u64 (Ethereum scalar)
//!
//! The Ethereum RLP encoding of a non-negative scalar `v` strips
//! leading zero bytes and uses a one-byte length prefix for non-trivial
//! cases:
//!
//! ```text
//! v == 0          → encoded = [0x80]                                  len 1
//! 1 ≤ v ≤ 0x7f    → encoded = [v as u8]                                len 1
//! v ≥ 0x80        → encoded = [0x80 + n, b_{n-1}, …, b_0]              len 1+n
//!                   where n = number of significant BE bytes (1..=8)
//! ```
//!
//! # Witness layout
//!
//! Per row (per encoded value) the witness holds:
//!
//!   - `value` — the u64 value being encoded.
//!   - `byte[0..8]` — big-endian byte decomposition; `byte[0]` is the
//!     most significant byte, `byte[7]` is the least significant.
//!   - `lz_mask[0..8]` — leading-zero mask; `lz_mask[i] = 1` iff
//!     `byte[i]` is one of the leading zero bytes (i.e., all bytes at
//!     positions `j ≤ i` are zero). Monotonically descending from 1s
//!     to 0s in BE order.
//!   - `byte7_high_bit`, `byte7_low7` — decomposition of `byte[7]`
//!     into its top bit + bottom 7 bits, used to distinguish the
//!     short (`v ≤ 0x7f`) case from the n=1 long (`v ≥ 0x80`) case.
//!   - `n_significant` — number of significant BE bytes after
//!     stripping leading zeros (`= 8 - Σ lz_mask[i]`), in `0..=8`.
//!   - `n_eq[0..9]` — one-hot indicator for `n_significant`. Exactly
//!     one of these is 1 (the index matching `n_significant`).
//!   - `is_zero`, `is_short`, `is_long` — case selectors:
//!     - `is_zero`: `v == 0` → encoded `[0x80]`, len 1.
//!     - `is_short`: `1 ≤ v ≤ 0x7f` → encoded `[v]`, len 1.
//!     - `is_long`: `v ≥ 0x80` → encoded `[0x80 + n, b_{n-1}, …]`, len 1+n.
//!     Exactly one fires per real row.
//!   - `encoded_byte[0..9]` — the encoded RLP bytes, zero-padded out
//!     to the max length of 9.
//!   - `encoded_len` — actual encoded length (`1..=9`).
//!
//! # Soundness scope (step 0)
//!
//! Host-side only. The witness builder constructs a consistent
//! row for any `u64`. Tests confirm the encoded bytes match the
//! canonical RLP encoding (`crate::rlp::rlp_encode_uint`) byte-for-byte,
//! and that the case selectors / n_eq / lz_mask invariants hold.
//!
//! Step 1+ will add the constraint system enforcing these invariants
//! algebraically. The column constants exported here are the contract
//! that the constraint system + downstream cross-AIR LogUp linkages
//! will use.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_VALUE: usize = 0;
pub const COL_BYTE_OFFSET: usize = 1;          // 1..9 (8 BE bytes)
pub const COL_LZ_MASK_OFFSET: usize = 9;       // 9..17 (8 mask bits)
pub const COL_BYTE7_HIGH_BIT: usize = 17;
pub const COL_BYTE7_LOW7: usize = 18;
pub const COL_N_SIGNIFICANT: usize = 19;
pub const COL_N_EQ_OFFSET: usize = 20;         // 20..29 (one-hot, indices 0..=8)
pub const COL_IS_ZERO: usize = 29;
pub const COL_IS_SHORT: usize = 30;
pub const COL_IS_LONG: usize = 31;
pub const COL_ENCODED_OFFSET: usize = 32;      // 32..41 (9 bytes max)
pub const COL_ENCODED_LEN: usize = 41;
pub const COL_IS_REAL: usize = 42;

// Step 1b: auxiliary cols for the dynamic-shift encoded[1..=8]
// alignment. shifted_byte[k-1] = Σ_{n=k}^{8} n_eq[n] * byte[7-n+k]
// for k ∈ 1..=8. Then encoded[k] = is_long * shifted_byte[k-1].
pub const COL_SHIFTED_BYTE_OFFSET: usize = 43;  // 43..51 (8 cols)

pub const NUM_COLUMNS: usize = COL_SHIFTED_BYTE_OFFSET + 8; // 51

/// Maximum encoded length (for a full 8-byte u64 ≥ 2^56, encoded is
/// `[0x88, b_7, b_6, …, b_0]` = 9 bytes).
pub const MAX_ENCODED_LEN: usize = 9;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U64RlpRow {
    pub value: u64,
}

#[derive(Clone, Debug, Default)]
pub struct U64RlpWitness {
    pub rows: Vec<U64RlpRow>,
}

impl U64RlpWitness {
    pub fn from_values(values: &[u64]) -> Self {
        Self {
            rows: values.iter().copied().map(|value| U64RlpRow { value }).collect(),
        }
    }
}

/// Canonical RLP encoding of a u64 (matches `crate::rlp::rlp_encode_uint`).
pub fn rlp_encode_u64(value: u64) -> Vec<u8> {
    if value == 0 {
        return vec![0x80];
    }
    if value < 0x80 {
        return vec![value as u8];
    }
    // Multi-byte: strip leading zeros from BE representation.
    let be = value.to_be_bytes();
    let first_nz = be.iter().position(|&b| b != 0).unwrap();
    let significant = &be[first_nz..];
    let mut out = Vec::with_capacity(1 + significant.len());
    out.push(0x80 + significant.len() as u8);
    out.extend_from_slice(significant);
    out
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &U64RlpWitness,
    curve: CurveType,
) -> crate::trace::TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        populate_row(&mut columns, r, row.value, curve);
        columns[COL_IS_REAL][r] = one.clone();
    }

    let polys: Vec<crate::trace::Polynomial> = columns
        .into_iter()
        .map(|evals| crate::trace::Polynomial { evaluations: evals, degree: num_rows })
        .collect();
    crate::trace::TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

/// Populate a single row's columns from a u64 value. Pure function for
/// host-side witness building + step 1 constraint testing.
fn populate_row(columns: &mut [Vec<Scalar>], r: usize, value: u64, curve: CurveType) {
    columns[COL_VALUE][r] = Scalar::from_u64(value, curve);

    // BE byte decomposition.
    let be = value.to_be_bytes();
    for i in 0..8 {
        columns[COL_BYTE_OFFSET + i][r] = Scalar::from_u64(be[i] as u64, curve);
    }

    // Leading-zero mask: lz_mask[i] = 1 iff be[0..=i] are all zero.
    let mut prefix_all_zero = true;
    for i in 0..8 {
        if be[i] != 0 {
            prefix_all_zero = false;
        }
        columns[COL_LZ_MASK_OFFSET + i][r] = Scalar::from_u64(prefix_all_zero as u64, curve);
    }

    // byte[7] = byte7_high_bit * 128 + byte7_low7.
    let b7 = be[7];
    let high = (b7 >> 7) & 1;
    let low7 = b7 & 0x7f;
    columns[COL_BYTE7_HIGH_BIT][r] = Scalar::from_u64(high as u64, curve);
    columns[COL_BYTE7_LOW7][r] = Scalar::from_u64(low7 as u64, curve);

    // n_significant = number of bytes after stripping leading zeros.
    let n: u64 = if value == 0 {
        0
    } else {
        let first_nz = be.iter().position(|&b| b != 0).unwrap();
        (8 - first_nz) as u64
    };
    columns[COL_N_SIGNIFICANT][r] = Scalar::from_u64(n, curve);

    // n_eq one-hot.
    for k in 0..=8 {
        let eq = if k as u64 == n { 1 } else { 0 };
        columns[COL_N_EQ_OFFSET + k][r] = Scalar::from_u64(eq, curve);
    }

    // Case selectors.
    let is_zero = value == 0;
    let is_short = (1..=0x7f).contains(&value);
    let is_long = value >= 0x80;
    columns[COL_IS_ZERO][r] = Scalar::from_u64(is_zero as u64, curve);
    columns[COL_IS_SHORT][r] = Scalar::from_u64(is_short as u64, curve);
    columns[COL_IS_LONG][r] = Scalar::from_u64(is_long as u64, curve);

    // Encoded bytes + length.
    let encoded = rlp_encode_u64(value);
    for (i, &eb) in encoded.iter().enumerate() {
        columns[COL_ENCODED_OFFSET + i][r] = Scalar::from_u64(eb as u64, curve);
    }
    columns[COL_ENCODED_LEN][r] = Scalar::from_u64(encoded.len() as u64, curve);

    // Step 1b: shifted_byte[k-1] for k=1..=8.
    // shifted_byte[k-1] = Σ_{t=k..=8} n_eq[t] * byte[7+k-t]
    // On an honest witness with n_eq one-hot at n_significant:
    //   shifted_byte[k-1] = byte[7+k-n] if k ≤ n, else 0.
    // Populated unconditionally (the formula applies to all cases;
    // the constraint `encoded[k] = is_long · shifted_byte[k-1]`
    // handles the case gating for short/zero).
    for k in 1..=8usize {
        let mut sb = 0u64;
        if n > 0 && k as u64 <= n {
            let byte_idx = 7 + k - n as usize;
            sb = be[byte_idx] as u64;
        }
        columns[COL_SHIFTED_BYTE_OFFSET + (k - 1)][r] =
            Scalar::from_u64(sb, curve);
    }
}

// ─── Constraint system (step 1a) ──────────────────────────────────────
//
// Enforces the simpler invariants of the u64 RLP witness:
//   - All binary cols are 0/1.
//   - Case selectors are mutually exclusive and sum to is_real.
//   - lz_mask is monotonically descending (mask[i+1] ≤ mask[i]).
//   - Leading zeros: byte[i] * lz_mask[i] = 0.
//   - value = Σ byte[i] · 256^(7-i) (BE decomp).
//   - byte[7] = byte7_high_bit · 128 + byte7_low7 (split).
//   - is_zero ↔ lz_mask[7] (value is zero iff all bytes zero iff mask[7]=1).
//   - is_short = (lz_mask[6] − lz_mask[7]) · (1 − byte7_high_bit).
//   - is_long = is_real − is_zero − is_short (case dichotomy).
//   - n_eq[k] one-hot (sum to is_real), Σ k·n_eq[k] = n_significant.
//   - n_significant = 8 − Σ lz_mask[i].
//   - encoded_len = is_real + is_long · n_significant.
//   - encoded[0] = 0x80·(is_zero + is_long) + is_short·byte[7] + is_long·n_significant.
//
// Not enforced (deferred to step 1b): encoded[1..=8] dynamic-shift
// alignment for the long case (`encoded[k] = byte[7-n+k]` for k ≤ n
// and 0 otherwise). Without it, the encoded byte cols beyond
// encoded[0] are unconstrained for the long case — the gadget proves
// "value has these case properties + length" but not "encoded body
// matches the value's significant bytes."

// Step 1b adds:
//   18: shifted_byte_from_neq_byte — β-RLC over k=1..=8:
//       shifted_byte[k-1] − Σ_{n=k}^{8} n_eq[n] · byte[7−n+k] = 0
//   19: encoded_k_eq_long_times_shifted — β-RLC over k=1..=8:
//       encoded[k] − is_long · shifted_byte[k-1] = 0
pub const NUM_ROW_CONSTRAINTS: usize = 20;
pub const NUM_SHIFTED: usize = 0;

pub struct U64RlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl U64RlpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for U64RlpConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_zero_binary".into(),
            "is_short_binary".into(),
            "is_long_binary".into(),
            "case_exclusivity_sums_to_is_real".into(),
            "lz_mask_binary_rlc".into(),
            "lz_mask_monotonic_descending_rlc".into(),
            "leading_zeros_byte_times_mask_rlc".into(),
            "value_equals_be_byte_decomp".into(),
            "byte7_decomp".into(),
            "byte7_high_bit_binary".into(),
            "is_zero_eq_lz_mask_7".into(),
            "is_short_formula".into(),
            "n_eq_binary_rlc".into(),
            "n_eq_one_hot_sum".into(),
            "n_significant_linear_combo".into(),
            "encoded_len_formula".into(),
            "encoded_0_formula".into(),
            "shifted_byte_from_neq_byte_rlc".into(),
            "encoded_k_eq_long_times_shifted_rlc".into(),
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
        let beta_test = Scalar::from_u64(7, curve);
        let n = columns[0].len();

        let mk = || vec![Scalar::zero(curve); n];
        let mut c_shifted_byte_rlc = mk();
        let mut c_enc_k_rlc = mk();
        let mut c_is_real_bin = mk();
        let mut c_is_zero_bin = mk();
        let mut c_is_short_bin = mk();
        let mut c_is_long_bin = mk();
        let mut c_case_excl = mk();
        let mut c_lz_bin_rlc = mk();
        let mut c_lz_mono_rlc = mk();
        let mut c_lz_mul_byte = mk();
        let mut c_value_decomp = mk();
        let mut c_byte7_decomp = mk();
        let mut c_byte7_hb_bin = mk();
        let mut c_is_zero_eq_mask7 = mk();
        let mut c_is_short_formula = mk();
        let mut c_n_eq_bin_rlc = mk();
        let mut c_n_eq_sum = mk();
        let mut c_n_lin = mk();
        let mut c_enc_len = mk();
        let mut c_enc_0 = mk();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let is_zero = &columns[COL_IS_ZERO][r];
            let is_short = &columns[COL_IS_SHORT][r];
            let is_long = &columns[COL_IS_LONG][r];
            let value = &columns[COL_VALUE][r];
            let byte7 = &columns[COL_BYTE_OFFSET + 7][r];
            let byte7_hb = &columns[COL_BYTE7_HIGH_BIT][r];
            let byte7_lo7 = &columns[COL_BYTE7_LOW7][r];
            let lz6 = &columns[COL_LZ_MASK_OFFSET + 6][r];
            let lz7 = &columns[COL_LZ_MASK_OFFSET + 7][r];
            let n_sig = &columns[COL_N_SIGNIFICANT][r];
            let enc_len = &columns[COL_ENCODED_LEN][r];
            let enc_0 = &columns[COL_ENCODED_OFFSET][r];

            c_is_real_bin[r] = is_real.mul(&is_real.sub(&one));
            c_is_zero_bin[r] = is_zero.mul(&is_zero.sub(&one));
            c_is_short_bin[r] = is_short.mul(&is_short.sub(&one));
            c_is_long_bin[r] = is_long.mul(&is_long.sub(&one));
            // is_real * (is_zero + is_short + is_long - is_real) = 0
            let sum_cases = is_zero.add(is_short).add(is_long);
            c_case_excl[r] = is_real.mul(&sum_cases.sub(is_real));

            // lz_mask binary β-RLC.
            let mut bin_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for i in 0..8 {
                let m = &columns[COL_LZ_MASK_OFFSET + i][r];
                let b = m.mul(&m.sub(&one));
                bin_acc = bin_acc.add(&bp.mul(&b));
                bp = bp.mul(&beta_test);
            }
            c_lz_bin_rlc[r] = bin_acc;

            // monotonic descending: lz_mask[i+1] * (1 - lz_mask[i]) = 0 for i=0..7.
            let mut mono_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for i in 0..7 {
                let m_i = &columns[COL_LZ_MASK_OFFSET + i][r];
                let m_i1 = &columns[COL_LZ_MASK_OFFSET + i + 1][r];
                let term = m_i1.mul(&one.sub(m_i));
                mono_acc = mono_acc.add(&bp.mul(&term));
                bp = bp.mul(&beta_test);
            }
            c_lz_mono_rlc[r] = mono_acc;

            // byte[i] * lz_mask[i] = 0 for i=0..8.
            let mut lzm_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for i in 0..8 {
                let bi = &columns[COL_BYTE_OFFSET + i][r];
                let mi = &columns[COL_LZ_MASK_OFFSET + i][r];
                lzm_acc = lzm_acc.add(&bp.mul(&bi.mul(mi)));
                bp = bp.mul(&beta_test);
            }
            c_lz_mul_byte[r] = lzm_acc;

            // value = Σ byte[i] * 256^(7-i)
            let two56 = Scalar::from_u64(256, curve);
            let mut acc = Scalar::zero(curve);
            let mut weight = Scalar::one(curve);
            for i in (0..8).rev() {
                let bi = &columns[COL_BYTE_OFFSET + i][r];
                acc = acc.add(&weight.mul(bi));
                weight = weight.mul(&two56);
            }
            c_value_decomp[r] = value.sub(&acc);

            // byte[7] = byte7_high_bit * 128 + byte7_low7
            let one_28 = Scalar::from_u64(128, curve);
            let recombined = byte7_hb.mul(&one_28).add(byte7_lo7);
            c_byte7_decomp[r] = byte7.sub(&recombined);
            c_byte7_hb_bin[r] = byte7_hb.mul(&byte7_hb.sub(&one));

            // is_zero - lz_mask[7] = 0
            c_is_zero_eq_mask7[r] = is_zero.sub(lz7);

            // is_short = (lz_mask[6] - lz_mask[7]) * (1 - byte7_high_bit)
            let single = lz6.sub(lz7);
            let lo_bit = one.sub(byte7_hb);
            c_is_short_formula[r] = is_short.sub(&single.mul(&lo_bit));

            // n_eq[k] binary β-RLC.
            let mut neq_bin_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..=8 {
                let neq = &columns[COL_N_EQ_OFFSET + k][r];
                let bn = neq.mul(&neq.sub(&one));
                neq_bin_acc = neq_bin_acc.add(&bp.mul(&bn));
                bp = bp.mul(&beta_test);
            }
            c_n_eq_bin_rlc[r] = neq_bin_acc;

            // Σ n_eq[k] = is_real (one-hot on real rows, all-zero on padding).
            let mut neq_sum = Scalar::zero(curve);
            for k in 0..=8 {
                neq_sum = neq_sum.add(&columns[COL_N_EQ_OFFSET + k][r]);
            }
            c_n_eq_sum[r] = neq_sum.sub(is_real);

            // Σ k * n_eq[k] = n_significant
            let mut n_lin = Scalar::zero(curve);
            for k in 0..=8 {
                let coef = Scalar::from_u64(k as u64, curve);
                n_lin = n_lin.add(&coef.mul(&columns[COL_N_EQ_OFFSET + k][r]));
            }
            c_n_lin[r] = n_lin.sub(n_sig);

            // encoded_len = is_real + is_long * n_significant
            let expected_len = is_real.add(&is_long.mul(n_sig));
            c_enc_len[r] = enc_len.sub(&expected_len);

            // encoded[0] = 0x80*(is_zero+is_long) + is_short*byte[7] + is_long*n_significant
            let prefix = Scalar::from_u64(0x80, curve);
            let zero_or_long = is_zero.add(is_long);
            let expected_enc0 =
                prefix.mul(&zero_or_long).add(&is_short.mul(byte7)).add(&is_long.mul(n_sig));
            c_enc_0[r] = enc_0.sub(&expected_enc0);

            // Step 1b: shifted_byte + encoded[k] constraints.
            let mut sb_acc = Scalar::zero(curve);
            let mut ek_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 1..=8usize {
                let sb = &columns[COL_SHIFTED_BYTE_OFFSET + (k - 1)][r];
                // shifted_byte[k-1] = Σ_{t=k..=8} n_eq[t] * byte[7-t+k]
                let mut sum_neq_byte = Scalar::zero(curve);
                for t in k..=8usize {
                    let neq_t = &columns[COL_N_EQ_OFFSET + t][r];
                    let byte_idx = 7 + k - t;
                    let byte_val = &columns[COL_BYTE_OFFSET + byte_idx][r];
                    sum_neq_byte = sum_neq_byte.add(&neq_t.mul(byte_val));
                }
                let sb_diff = sb.sub(&sum_neq_byte);
                sb_acc = sb_acc.add(&bp.mul(&sb_diff));

                // encoded[k] = is_long * shifted_byte[k-1]
                let enc_k = &columns[COL_ENCODED_OFFSET + k][r];
                let expected_enc_k = is_long.mul(sb);
                let ek_diff = enc_k.sub(&expected_enc_k);
                ek_acc = ek_acc.add(&bp.mul(&ek_diff));

                bp = bp.mul(&beta_test);
            }
            c_shifted_byte_rlc[r] = sb_acc;
            c_enc_k_rlc[r] = ek_acc;
        }

        vec![
            c_is_real_bin,
            c_is_zero_bin,
            c_is_short_bin,
            c_is_long_bin,
            c_case_excl,
            c_lz_bin_rlc,
            c_lz_mono_rlc,
            c_lz_mul_byte,
            c_value_decomp,
            c_byte7_decomp,
            c_byte7_hb_bin,
            c_is_zero_eq_mask7,
            c_is_short_formula,
            c_n_eq_bin_rlc,
            c_n_eq_sum,
            c_n_lin,
            c_enc_len,
            c_enc_0,
            c_shifted_byte_rlc,
            c_enc_k_rlc,
        ]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);

        let is_real = &col_evals[COL_IS_REAL];
        let is_zero = &col_evals[COL_IS_ZERO];
        let is_short = &col_evals[COL_IS_SHORT];
        let is_long = &col_evals[COL_IS_LONG];
        let value = &col_evals[COL_VALUE];
        let byte7 = &col_evals[COL_BYTE_OFFSET + 7];
        let byte7_hb = &col_evals[COL_BYTE7_HIGH_BIT];
        let byte7_lo7 = &col_evals[COL_BYTE7_LOW7];
        let lz6 = &col_evals[COL_LZ_MASK_OFFSET + 6];
        let lz7 = &col_evals[COL_LZ_MASK_OFFSET + 7];
        let n_sig = &col_evals[COL_N_SIGNIFICANT];
        let enc_len = &col_evals[COL_ENCODED_LEN];
        let enc_0 = &col_evals[COL_ENCODED_OFFSET];

        let bodies: Vec<Scalar> = {
            let c0 = is_real.mul(&is_real.sub(&one));
            let c1 = is_zero.mul(&is_zero.sub(&one));
            let c2 = is_short.mul(&is_short.sub(&one));
            let c3 = is_long.mul(&is_long.sub(&one));
            let sum_cases = is_zero.add(is_short).add(is_long);
            let c4 = is_real.mul(&sum_cases.sub(is_real));

            let mut bin_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for i in 0..8 {
                let m = &col_evals[COL_LZ_MASK_OFFSET + i];
                let b = m.mul(&m.sub(&one));
                bin_acc = bin_acc.add(&bp.mul(&b));
                bp = bp.mul(alpha);
            }
            let c5 = bin_acc;

            let mut mono_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for i in 0..7 {
                let m_i = &col_evals[COL_LZ_MASK_OFFSET + i];
                let m_i1 = &col_evals[COL_LZ_MASK_OFFSET + i + 1];
                let term = m_i1.mul(&one.sub(m_i));
                mono_acc = mono_acc.add(&bp.mul(&term));
                bp = bp.mul(alpha);
            }
            let c6 = mono_acc;

            let mut lzm_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for i in 0..8 {
                let bi = &col_evals[COL_BYTE_OFFSET + i];
                let mi = &col_evals[COL_LZ_MASK_OFFSET + i];
                lzm_acc = lzm_acc.add(&bp.mul(&bi.mul(mi)));
                bp = bp.mul(alpha);
            }
            let c7 = lzm_acc;

            let two56 = Scalar::from_u64(256, curve);
            let mut acc = Scalar::zero(curve);
            let mut weight = Scalar::one(curve);
            for i in (0..8).rev() {
                let bi = &col_evals[COL_BYTE_OFFSET + i];
                acc = acc.add(&weight.mul(bi));
                weight = weight.mul(&two56);
            }
            let c8 = value.sub(&acc);

            let one_28 = Scalar::from_u64(128, curve);
            let recombined = byte7_hb.mul(&one_28).add(byte7_lo7);
            let c9 = byte7.sub(&recombined);
            let c10 = byte7_hb.mul(&byte7_hb.sub(&one));

            let c11 = is_zero.sub(lz7);

            let single = lz6.sub(lz7);
            let lo_bit = one.sub(byte7_hb);
            let c12 = is_short.sub(&single.mul(&lo_bit));

            let mut neq_bin_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..=8 {
                let neq = &col_evals[COL_N_EQ_OFFSET + k];
                let bn = neq.mul(&neq.sub(&one));
                neq_bin_acc = neq_bin_acc.add(&bp.mul(&bn));
                bp = bp.mul(alpha);
            }
            let c13 = neq_bin_acc;

            let mut neq_sum = Scalar::zero(curve);
            for k in 0..=8 {
                neq_sum = neq_sum.add(&col_evals[COL_N_EQ_OFFSET + k]);
            }
            let c14 = neq_sum.sub(is_real);

            let mut n_lin = Scalar::zero(curve);
            for k in 0..=8 {
                let coef = Scalar::from_u64(k as u64, curve);
                n_lin = n_lin.add(&coef.mul(&col_evals[COL_N_EQ_OFFSET + k]));
            }
            let c15 = n_lin.sub(n_sig);

            let expected_len = is_real.add(&is_long.mul(n_sig));
            let c16 = enc_len.sub(&expected_len);

            let prefix = Scalar::from_u64(0x80, curve);
            let zero_or_long = is_zero.add(is_long);
            let expected_enc0 =
                prefix.mul(&zero_or_long).add(&is_short.mul(byte7)).add(&is_long.mul(n_sig));
            let c17 = enc_0.sub(&expected_enc0);

            // Step 1b: shifted_byte + encoded_k constraints.
            let mut sb_acc = Scalar::zero(curve);
            let mut ek_acc = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 1..=8usize {
                let sb = &col_evals[COL_SHIFTED_BYTE_OFFSET + (k - 1)];
                let mut sum_nb = Scalar::zero(curve);
                for t in k..=8usize {
                    let neq_t = &col_evals[COL_N_EQ_OFFSET + t];
                    let byte_val = &col_evals[COL_BYTE_OFFSET + 7 + k - t];
                    sum_nb = sum_nb.add(&neq_t.mul(byte_val));
                }
                sb_acc = sb_acc.add(&bp.mul(&sb.sub(&sum_nb)));

                let enc_k = &col_evals[COL_ENCODED_OFFSET + k];
                ek_acc = ek_acc.add(&bp.mul(&enc_k.sub(&is_long.mul(sb))));
                bp = bp.mul(alpha);
            }
            let c18 = sb_acc;
            let c19 = ek_acc;

            vec![
                c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
                c12, c13, c14, c15, c16, c17, c18, c19,
            ]
        };

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
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_zero = &col_coeffs[COL_IS_ZERO];
        let is_short = &col_coeffs[COL_IS_SHORT];
        let is_long = &col_coeffs[COL_IS_LONG];
        let value = &col_coeffs[COL_VALUE];
        let byte7 = &col_coeffs[COL_BYTE_OFFSET + 7];
        let byte7_hb = &col_coeffs[COL_BYTE7_HIGH_BIT];
        let byte7_lo7 = &col_coeffs[COL_BYTE7_LOW7];
        let lz6 = &col_coeffs[COL_LZ_MASK_OFFSET + 6];
        let lz7 = &col_coeffs[COL_LZ_MASK_OFFSET + 7];
        let n_sig = &col_coeffs[COL_N_SIGNIFICANT];
        let enc_len = &col_coeffs[COL_ENCODED_LEN];
        let enc_0 = &col_coeffs[COL_ENCODED_OFFSET];

        let bin = |x: &Vec<Scalar>| {
            let x_m1 = poly_sub(x, &one_poly, curve);
            poly_mul(x, &x_m1, curve)
        };

        let c0 = bin(is_real);
        let c1 = bin(is_zero);
        let c2 = bin(is_short);
        let c3 = bin(is_long);
        let sum_cases = poly_add(&poly_add(is_zero, is_short, curve), is_long, curve);
        let diff_real = poly_sub(&sum_cases, is_real, curve);
        let c4 = poly_mul(is_real, &diff_real, curve);

        let mut c5 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..8 {
            let m = &col_coeffs[COL_LZ_MASK_OFFSET + i];
            let b = poly_mul(m, &poly_sub(m, &one_poly, curve), curve);
            c5 = poly_add(&c5, &poly_scalar_mul(&b, &bp), curve);
            bp = bp.mul(alpha);
        }

        let mut c6 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..7 {
            let m_i = &col_coeffs[COL_LZ_MASK_OFFSET + i];
            let m_i1 = &col_coeffs[COL_LZ_MASK_OFFSET + i + 1];
            let one_minus = poly_sub(&one_poly, m_i, curve);
            let term = poly_mul(m_i1, &one_minus, curve);
            c6 = poly_add(&c6, &poly_scalar_mul(&term, &bp), curve);
            bp = bp.mul(alpha);
        }

        let mut c7 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for i in 0..8 {
            let bi = &col_coeffs[COL_BYTE_OFFSET + i];
            let mi = &col_coeffs[COL_LZ_MASK_OFFSET + i];
            let p = poly_mul(bi, mi, curve);
            c7 = poly_add(&c7, &poly_scalar_mul(&p, &bp), curve);
            bp = bp.mul(alpha);
        }

        let two56 = Scalar::from_u64(256, curve);
        let mut acc = vec![Scalar::zero(curve)];
        let mut weight = Scalar::one(curve);
        for i in (0..8).rev() {
            let bi = &col_coeffs[COL_BYTE_OFFSET + i];
            acc = poly_add(&acc, &poly_scalar_mul(bi, &weight), curve);
            weight = weight.mul(&two56);
        }
        let c8 = poly_sub(value, &acc, curve);

        let one_28 = Scalar::from_u64(128, curve);
        let hi_term = poly_scalar_mul(byte7_hb, &one_28);
        let recombined = poly_add(&hi_term, byte7_lo7, curve);
        let c9 = poly_sub(byte7, &recombined, curve);
        let c10 = bin(byte7_hb);

        let c11 = poly_sub(is_zero, lz7, curve);

        let single = poly_sub(lz6, lz7, curve);
        let lo_bit = poly_sub(&one_poly, byte7_hb, curve);
        let prod = poly_mul(&single, &lo_bit, curve);
        let c12 = poly_sub(is_short, &prod, curve);

        let mut c13 = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..=8 {
            let neq = &col_coeffs[COL_N_EQ_OFFSET + k];
            let b = poly_mul(neq, &poly_sub(neq, &one_poly, curve), curve);
            c13 = poly_add(&c13, &poly_scalar_mul(&b, &bp), curve);
            bp = bp.mul(alpha);
        }

        let mut neq_sum = vec![Scalar::zero(curve)];
        for k in 0..=8 {
            neq_sum = poly_add(&neq_sum, &col_coeffs[COL_N_EQ_OFFSET + k], curve);
        }
        let c14 = poly_sub(&neq_sum, is_real, curve);

        let mut n_lin = vec![Scalar::zero(curve)];
        for k in 0..=8 {
            let coef = Scalar::from_u64(k as u64, curve);
            n_lin = poly_add(&n_lin, &poly_scalar_mul(&col_coeffs[COL_N_EQ_OFFSET + k], &coef), curve);
        }
        let c15 = poly_sub(&n_lin, n_sig, curve);

        let long_ns = poly_mul(is_long, n_sig, curve);
        let expected_len = poly_add(is_real, &long_ns, curve);
        let c16 = poly_sub(enc_len, &expected_len, curve);

        let prefix = Scalar::from_u64(0x80, curve);
        let zero_or_long = poly_add(is_zero, is_long, curve);
        let prefix_term = poly_scalar_mul(&zero_or_long, &prefix);
        let short_byte = poly_mul(is_short, byte7, curve);
        let long_n = poly_mul(is_long, n_sig, curve);
        let expected_enc0 =
            poly_add(&poly_add(&prefix_term, &short_byte, curve), &long_n, curve);
        let c17 = poly_sub(enc_0, &expected_enc0, curve);

        // Step 1b: shifted_byte + encoded_k in coeff form.
        let mut sb_acc = vec![Scalar::zero(curve)];
        let mut ek_acc = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 1..=8usize {
            let sb = &col_coeffs[COL_SHIFTED_BYTE_OFFSET + (k - 1)];
            let mut sum_nb = vec![Scalar::zero(curve)];
            for t in k..=8usize {
                let neq_t = &col_coeffs[COL_N_EQ_OFFSET + t];
                let byte_val = &col_coeffs[COL_BYTE_OFFSET + 7 + k - t];
                let prod = poly_mul(neq_t, byte_val, curve);
                sum_nb = poly_add(&sum_nb, &prod, curve);
            }
            let sb_diff = poly_sub(sb, &sum_nb, curve);
            sb_acc = poly_add(&sb_acc, &poly_scalar_mul(&sb_diff, &bp), curve);

            let enc_k = &col_coeffs[COL_ENCODED_OFFSET + k];
            let long_sb = poly_mul(is_long, sb, curve);
            let ek_diff = poly_sub(enc_k, &long_sb, curve);
            ek_acc = poly_add(&ek_acc, &poly_scalar_mul(&ek_diff, &bp), curve);
            bp = bp.mul(alpha);
        }
        let c18 = sb_acc;
        let c19 = ek_acc;

        let bodies = [
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
            c12, c13, c14, c15, c16, c17, c18, c19,
        ];
        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
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
        // 8-bit range checks for all byte cols (BE bytes + encoded bytes + byte7_low7).
        let tables = vec![LookupTable::range(8)];
        let mut declarations = Vec::new();
        for i in 0..8 {
            declarations.push((
                LookupDeclaration {
                    label: format!("u64_rlp_byte_{}_8bit", i),
                    column_index: COL_BYTE_OFFSET + i,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for i in 0..MAX_ENCODED_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("u64_rlp_encoded_{}_8bit", i),
                    column_index: COL_ENCODED_OFFSET + i,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        // byte7_low7 ∈ [0, 128) — range as 8-bit (subsumes 7-bit).
        declarations.push((
            LookupDeclaration {
                label: "u64_rlp_byte7_low7_8bit".into(),
                column_index: COL_BYTE7_LOW7,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        for i in 0..8 {
            declarations.push((
                LookupDeclaration {
                    label: format!("u64_rlp_shifted_byte_{}_8bit", i),
                    column_index: COL_SHIFTED_BYTE_OFFSET + i,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check against the existing canonical RLP encoder for
    /// representative values across all 3 cases.
    #[test]
    fn rlp_encode_u64_matches_canonical_encoder() {
        let cases: &[(u64, &[u8])] = &[
            (0, &[0x80]),
            (0x01, &[0x01]),
            (0x7f, &[0x7f]),
            (0x80, &[0x81, 0x80]),
            (0xff, &[0x81, 0xff]),
            (0x100, &[0x82, 0x01, 0x00]),
            (0xffff, &[0x82, 0xff, 0xff]),
            (0x010000, &[0x83, 0x01, 0x00, 0x00]),
            (0x123456, &[0x83, 0x12, 0x34, 0x56]),
            (u64::MAX, &[0x88, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
        ];
        for &(v, expected) in cases {
            assert_eq!(rlp_encode_u64(v), expected, "value {:#x}", v);
            // Also matches the project's canonical encoder.
            let canonical = crate::rlp::rlp_encode_uint(v);
            assert_eq!(rlp_encode_u64(v), canonical, "value {:#x} vs rlp::rlp_encode_uint", v);
        }
    }

    fn build_for(values: &[u64]) -> crate::trace::TracePolynomials {
        let w = U64RlpWitness::from_values(values);
        build_trace_polynomials(&w, CurveType::Bls48581)
    }

    #[test]
    fn witness_value_column_matches() {
        let values = [0, 0x42, 0x80, 0x12345678abcdef99];
        let trace = build_for(&values);
        for (r, &v) in values.iter().enumerate() {
            assert_eq!(trace.columns[COL_VALUE].evaluations[r].to_u64(), v);
        }
    }

    #[test]
    fn witness_byte_decomp_be_correct() {
        let values = [0u64, 0xff, 0x010000, u64::MAX];
        let trace = build_for(&values);
        for (r, &v) in values.iter().enumerate() {
            let be = v.to_be_bytes();
            for i in 0..8 {
                assert_eq!(
                    trace.columns[COL_BYTE_OFFSET + i].evaluations[r].to_u64(),
                    be[i] as u64,
                    "value={:#x} byte[{}]", v, i,
                );
            }
        }
    }

    #[test]
    fn witness_lz_mask_monotonic_descending() {
        let trace = build_for(&[0x00, 0x42, 0x010000, u64::MAX]);
        for r in 0..4 {
            let mut prev = 1u64;
            for i in 0..8 {
                let m = trace.columns[COL_LZ_MASK_OFFSET + i].evaluations[r].to_u64();
                assert!(m == 0 || m == 1, "mask must be binary");
                assert!(m <= prev, "monotonic descending");
                prev = m;
            }
        }
    }

    #[test]
    fn witness_n_significant_and_n_eq_consistent() {
        let cases: &[(u64, u64)] = &[
            (0, 0),
            (0x01, 1),
            (0x80, 1),
            (0xff, 1),
            (0x100, 2),
            (0x010000, 3),
            (u64::MAX, 8),
        ];
        let values: Vec<u64> = cases.iter().map(|&(v, _)| v).collect();
        let trace = build_for(&values);
        for (r, &(v, expected_n)) in cases.iter().enumerate() {
            let n = trace.columns[COL_N_SIGNIFICANT].evaluations[r].to_u64();
            assert_eq!(n, expected_n, "n_significant for {:#x}", v);
            // n_eq is one-hot with the 1 at index n.
            for k in 0..=8 {
                let eq = trace.columns[COL_N_EQ_OFFSET + k].evaluations[r].to_u64();
                let want = if k as u64 == expected_n { 1 } else { 0 };
                assert_eq!(eq, want, "n_eq[{}] for {:#x}", k, v);
            }
        }
    }

    #[test]
    fn witness_case_selectors_are_one_hot() {
        let trace = build_for(&[0, 0x01, 0x7f, 0x80, 0xffff, u64::MAX]);
        for r in 0..6 {
            let z = trace.columns[COL_IS_ZERO].evaluations[r].to_u64();
            let s = trace.columns[COL_IS_SHORT].evaluations[r].to_u64();
            let l = trace.columns[COL_IS_LONG].evaluations[r].to_u64();
            assert!(z == 0 || z == 1);
            assert!(s == 0 || s == 1);
            assert!(l == 0 || l == 1);
            assert_eq!(z + s + l, 1, "exactly one case selector at row {}", r);
        }
    }

    #[test]
    fn witness_byte7_high_low_decomp() {
        let trace = build_for(&[0x7f, 0x80, 0xff]);
        // value=0x7f: byte[7]=0x7f, high=0, low=0x7f.
        assert_eq!(trace.columns[COL_BYTE7_HIGH_BIT].evaluations[0].to_u64(), 0);
        assert_eq!(trace.columns[COL_BYTE7_LOW7].evaluations[0].to_u64(), 0x7f);
        // value=0x80: byte[7]=0x80, high=1, low=0.
        assert_eq!(trace.columns[COL_BYTE7_HIGH_BIT].evaluations[1].to_u64(), 1);
        assert_eq!(trace.columns[COL_BYTE7_LOW7].evaluations[1].to_u64(), 0);
        // value=0xff: byte[7]=0xff, high=1, low=0x7f.
        assert_eq!(trace.columns[COL_BYTE7_HIGH_BIT].evaluations[2].to_u64(), 1);
        assert_eq!(trace.columns[COL_BYTE7_LOW7].evaluations[2].to_u64(), 0x7f);
    }

    #[test]
    fn witness_encoded_bytes_match_canonical() {
        let values = [0u64, 1, 0x7f, 0x80, 0x100, 0x123456, u64::MAX];
        let trace = build_for(&values);
        for (r, &v) in values.iter().enumerate() {
            let expected = rlp_encode_u64(v);
            let n_encoded = trace.columns[COL_ENCODED_LEN].evaluations[r].to_u64() as usize;
            assert_eq!(n_encoded, expected.len(), "encoded_len for {:#x}", v);
            for i in 0..expected.len() {
                assert_eq!(
                    trace.columns[COL_ENCODED_OFFSET + i].evaluations[r].to_u64(),
                    expected[i] as u64,
                    "encoded[{}] for {:#x}", i, v,
                );
            }
            // Padding past encoded_len is zero.
            for i in expected.len()..MAX_ENCODED_LEN {
                assert!(
                    trace.columns[COL_ENCODED_OFFSET + i].evaluations[r].is_zero(),
                    "padding at encoded[{}] for {:#x}", i, v,
                );
            }
        }
    }

    /// Pin: column count constant. Step 1 constraint system will use
    /// `NUM_COLUMNS` to size its column array; if this changes, the
    /// constraint impl must be updated in lockstep.
    #[test]
    fn num_columns_pinned_to_51() {
        assert_eq!(NUM_COLUMNS, 51);
    }

    #[test]
    fn constraints_evaluate_zero_on_honest_witness() {
        let values = [0u64, 1, 0x7f, 0x80, 0xff, 0x100, 0x123456, u64::MAX];
        let trace = build_for(&values);
        let cs = U64RlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn case_exclusivity_fires_on_tamper() {
        let trace = build_for(&[0x42]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper: clear is_short so the case sum becomes 0 (≠ is_real=1).
        cols[COL_IS_SHORT][0] = Scalar::zero(CurveType::Bls48581);
        let cs = U64RlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 4 = case_exclusivity_sums_to_is_real.
        assert!(!res[4][0].is_zero());
    }

    #[test]
    fn value_decomp_fires_on_tampered_byte() {
        let trace = build_for(&[0x123456]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_BYTE_OFFSET + 7][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = U64RlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 8 = value_equals_be_byte_decomp.
        assert!(!res[8][0].is_zero());
    }

    #[test]
    fn encoded_len_formula_fires_on_tamper() {
        let trace = build_for(&[0x100]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_ENCODED_LEN][0] = Scalar::from_u64(99, CurveType::Bls48581);
        let cs = U64RlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 16 = encoded_len_formula.
        assert!(!res[16][0].is_zero());
    }

    #[test]
    fn encoded_0_formula_fires_on_tamper() {
        let trace = build_for(&[0x80]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_ENCODED_OFFSET][0] = Scalar::from_u64(0x99, CurveType::Bls48581);
        let cs = U64RlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 17 = encoded_0_formula.
        assert!(!res[17][0].is_zero());
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
        let w = U64RlpWitness::from_values(&[0, 1, 0x7f, 0x80, 0xff, 0x100, 0x123456, u64::MAX]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = U64RlpConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(verify_with_scheme(&proof, &cs, &scheme, curve));
    }

    #[test]
    fn lz_mask_relates_to_n_significant() {
        let values = [0u64, 0x42, 0x123456, u64::MAX];
        let trace = build_for(&values);
        for r in 0..values.len() {
            let n = trace.columns[COL_N_SIGNIFICANT].evaluations[r].to_u64();
            // n_significant = 8 - Σ lz_mask[i]
            let mut mask_sum = 0u64;
            for i in 0..8 {
                mask_sum += trace.columns[COL_LZ_MASK_OFFSET + i].evaluations[r].to_u64();
            }
            assert_eq!(n + mask_sum, 8, "row {} n_significant + mask_sum = 8", r);
        }
    }
}
