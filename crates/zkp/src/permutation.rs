//! Memory permutation argument for load/store consistency.
//!
//! Implements a grand-product permutation argument that proves the multiset
//! of memory accesses `{(addr, val, timestamp, rw)}` from the execution trace
//! matches a sorted version ordered by `(addr, timestamp)`. This ensures:
//!
//! 1. Every load returns the value from the most recent store to that address.
//! 2. No memory values are fabricated (every read must correspond to a prior write).
//!
//! # Protocol
//!
//! The prover:
//! 1. Sorts the memory access tuples by `(addr, timestamp)`.
//! 2. Computes auxiliary columns: `is_same_addr`, `inv_addr_diff`.
//! 3. Draws random challenges `γ` (gamma) and `δ` (delta) from the transcript.
//! 4. Builds a grand-product accumulator column `Z` such that:
//!    ```text
//!    Z[0] = 1
//!    Z[i+1] = Z[i] * (γ + addr[i] + δ·val[i] + δ²·ts[i] + δ³·rw[i])
//!                    / (γ + sorted_addr[i] + δ·sorted_val[i] + δ²·sorted_ts[i] + δ³·sorted_rw[i])
//!    ```
//! 5. Commits Z and the sorted/auxiliary columns.
//!
//! The verifier checks:
//! - Grand product transition (cross-row): `Z(ω·X) * denom(X) = Z(X) * numer(X)`.
//! - Boundary: `Z(1) = 1`.
//! - Sorted trace consistency (cross-row):
//!   - `is_same_addr · (sorted_addr(ω·X) - sorted_addr(X)) = 0`
//!   - `(1 - is_same_addr) · (1 - (sorted_addr_diff) · inv_addr_diff) = 0`
//!   - Read consistency: `is_same_addr · (1 - sorted_rw(ω·X)) · (sorted_val(ω·X) - sorted_val(X)) = 0`

use crate::field::{Scalar, CurveType};

/// Layout of permutation-related columns in the extended trace.
///
/// These columns are appended after the VM's normal trace columns.
/// The indices are offsets from the start of the permutation columns,
/// not absolute trace column indices.
#[derive(Debug, Clone)]
pub struct MemoryPermutationLayout {
    /// Offset of the sorted address column.
    pub sorted_addr: usize,
    /// Offset of the sorted value column (or first limb for EVM).
    pub sorted_val_start: usize,
    /// Number of value columns (1 for RISC-V/SBF, 4 for EVM U256 limbs).
    pub num_val_columns: usize,
    /// Offset of the sorted timestamp column.
    pub sorted_ts: usize,
    /// Offset of the sorted read/write flag column.
    pub sorted_rw: usize,
    /// Offset of the grand product accumulator Z column.
    pub z_column: usize,
    /// Offset of the `is_same_addr` binary flag column.
    pub is_same_addr: usize,
    /// Offset of the `inv_addr_diff` inverse witness column.
    pub inv_addr_diff: usize,
    /// Offset of the original timestamp column (row index).
    pub original_ts: usize,
    /// Total number of permutation columns.
    pub num_columns: usize,
}

impl MemoryPermutationLayout {
    /// Create a layout for a VM with `num_val_cols` value columns.
    ///
    /// RISC-V/SBF: `num_val_cols = 1` → 8 total columns.
    /// EVM: `num_val_cols = 4` → 11 total columns.
    pub fn new(num_val_cols: usize) -> Self {
        let sorted_addr = 0;
        let sorted_val_start = 1;
        let sorted_ts = 1 + num_val_cols;
        let sorted_rw = sorted_ts + 1;
        let z_column = sorted_rw + 1;
        let is_same_addr = z_column + 1;
        let inv_addr_diff = is_same_addr + 1;
        let original_ts = inv_addr_diff + 1;
        let num_columns = original_ts + 1;

        MemoryPermutationLayout {
            sorted_addr,
            sorted_val_start,
            num_val_columns: num_val_cols,
            sorted_ts,
            sorted_rw,
            z_column,
            is_same_addr,
            inv_addr_diff,
            original_ts,
            num_columns,
        }
    }
}

/// A single memory access tuple for sorting.
#[derive(Clone, Debug)]
pub struct MemoryAccess {
    /// Memory address accessed.
    pub addr: u64,
    /// Value(s) — for RISC-V/SBF this is `[val]`, for EVM `[l0, l1, l2, l3]`.
    pub values: Vec<u64>,
    /// Timestamp (execution step index).
    pub timestamp: u64,
    /// 0 = read, 1 = write.
    pub rw: u64,
}

/// Sort memory accesses by (address, timestamp) and compute auxiliary columns.
///
/// Returns `(sorted_accesses, is_same_addr, inv_addr_diff)`.
/// - `is_same_addr[i]` = 1 if `sorted[i].addr == sorted[i-1].addr`, 0 at i=0.
/// - `inv_addr_diff[i]` = `(sorted[i].addr - sorted[i-1].addr)^{-1}` when different, 0 when same.
pub fn sort_and_compute_aux(
    accesses: &[MemoryAccess],
    curve: CurveType,
) -> (Vec<MemoryAccess>, Vec<Scalar>, Vec<Scalar>) {
    let n = accesses.len();
    if n == 0 {
        return (vec![], vec![], vec![]);
    }

    // Sort by (addr, timestamp)
    let mut sorted = accesses.to_vec();
    sorted.sort_by_key(|a| (a.addr, a.timestamp));

    let mut is_same_addr = vec![Scalar::zero(curve); n];
    let mut inv_addr_diff = vec![Scalar::zero(curve); n];

    let one = Scalar::one(curve);

    for i in 1..n {
        if sorted[i].addr == sorted[i - 1].addr {
            is_same_addr[i] = one.clone();
            // inv_addr_diff stays zero
        } else {
            // is_same_addr stays zero
            let diff = Scalar::from_u64(sorted[i].addr - sorted[i - 1].addr, curve);
            inv_addr_diff[i] = diff.inverse();
        }
    }

    (sorted, is_same_addr, inv_addr_diff)
}

/// Compute the grand product accumulator column Z.
///
/// Z[0] = 1
/// Z[i+1] = Z[i] * numer[i] / denom[i]
///
/// where:
///   numer[i] = γ + addr[i] + δ·val[i] + δ²·ts[i] + δ³·rw[i]    (original trace)
///   denom[i] = γ + s_addr[i] + δ·s_val[i] + δ²·s_ts[i] + δ³·s_rw[i]  (sorted trace)
///
/// For multivalue (EVM), val is combined as: val_l0 + δ_v·val_l1 + δ_v²·val_l2 + δ_v³·val_l3,
/// where δ_v = δ⁴ to avoid collision with the ts/rw powers.
pub fn compute_grand_product(
    original: &[MemoryAccess],
    sorted: &[MemoryAccess],
    gamma: &Scalar,
    delta: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let n = original.len();
    // Z has n+1 entries conceptually; we store n entries aligned with rows
    // Z[0] = 1, Z[i] = Z[i-1] * numer[i-1] / denom[i-1]
    let mut z = vec![Scalar::one(curve); n];

    let delta2 = delta.mul(delta);
    let delta3 = delta2.mul(delta);

    for i in 0..n.saturating_sub(1) {
        let numer = compute_tuple_hash(
            &original[i], gamma, delta, &delta2, &delta3, curve,
        );
        let denom = compute_tuple_hash(
            &sorted[i], gamma, delta, &delta2, &delta3, curve,
        );
        let denom_inv = denom.inverse();
        z[i + 1] = z[i].mul(&numer).mul(&denom_inv);
    }

    z
}

/// Compute γ + addr + δ·val + δ²·ts + δ³·rw for a memory access tuple.
fn compute_tuple_hash(
    access: &MemoryAccess,
    gamma: &Scalar,
    delta: &Scalar,
    delta2: &Scalar,
    delta3: &Scalar,
    curve: CurveType,
) -> Scalar {
    let addr = Scalar::from_u64(access.addr, curve);
    let ts = Scalar::from_u64(access.timestamp, curve);
    let rw = Scalar::from_u64(access.rw, curve);

    // For multi-value, combine with powers of delta starting from delta
    // Single value: delta * val
    // Multi-value: delta * (val0 + delta^4 * val1 + delta^8 * val2 + ...)
    let val_combined = if access.values.len() == 1 {
        Scalar::from_u64(access.values[0], curve)
    } else {
        // Multi-value EVM: combine limbs with higher delta powers
        let mut combined = Scalar::zero(curve);
        let mut delta_pow = Scalar::one(curve);
        let delta4 = delta2.mul(delta2);
        for &v in &access.values {
            let val_s = Scalar::from_u64(v, curve);
            combined = combined.add(&delta_pow.mul(&val_s));
            delta_pow = delta_pow.mul(&delta4);
        }
        combined
    };

    gamma
        .add(&addr)
        .add(&delta.mul(&val_combined))
        .add(&delta2.mul(&ts))
        .add(&delta3.mul(&rw))
}

/// Evaluate the permutation constraint contributions at a single point.
///
/// Returns the combined constraint value at z for the sorted trace consistency checks:
/// 1. `is_same_addr · (is_same_addr - 1) = 0` (binary check)
/// 2. `is_same_addr(ω·z) · (sorted_addr(ω·z) - sorted_addr(z)) = 0`
/// 3. `(1 - is_same_addr(ω·z)) · (1 - addr_diff · inv_addr_diff(z)) = 0`
/// 4. Read consistency: `is_same_addr(ω·z) · (1 - sorted_rw(ω·z)) · (sorted_val(ω·z) - sorted_val(z)) = 0`
///
/// All constraints are multiplied by `(z - ω^{n-1})` for wrap-around exclusion
/// and accumulated with alpha powers starting from `alpha_offset`.
pub fn evaluate_permutation_constraints_at_point(
    perm_col_evals_at_z: &[Scalar],
    perm_shifted_evals: &[Scalar],
    layout: &MemoryPermutationLayout,
    z: &Scalar,
    omega_n_minus_1: &Scalar,
    alpha: &Scalar,
    alpha_offset: usize,
) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let exclusion = z.sub(omega_n_minus_1);

    let mut ap = Scalar::one(curve);
    for _ in 0..alpha_offset {
        ap = ap.mul(alpha);
    }
    let mut result = Scalar::zero(curve);

    // Constraint 1: is_same_addr binary
    let isa = &perm_col_evals_at_z[layout.is_same_addr];
    let body1 = isa.mul(&isa.sub(&one));
    result = result.add(&ap.mul(&body1));
    ap = ap.mul(alpha);

    // Constraints 2-4 use shifted evaluations (cross-row)
    // shifted_evals layout matches perm columns in order
    if !perm_shifted_evals.is_empty() {
        let sa_z = &perm_col_evals_at_z[layout.sorted_addr];
        let sa_wz = &perm_shifted_evals[layout.sorted_addr];
        let isa_wz = &perm_shifted_evals[layout.is_same_addr];
        let inv_diff = &perm_col_evals_at_z[layout.inv_addr_diff];

        // Constraint 2: is_same_addr(ω·z) * (sorted_addr(ω·z) - sorted_addr(z)) = 0
        let addr_diff = sa_wz.sub(sa_z);
        let body2 = isa_wz.mul(&addr_diff);
        result = result.add(&ap.mul(&body2).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 3: (1 - is_same_addr(ω·z)) * (1 - addr_diff * inv_addr_diff) = 0
        let body3 = one.sub(isa_wz).mul(&one.sub(&addr_diff.mul(inv_diff)));
        result = result.add(&ap.mul(&body3).mul(&exclusion));
        ap = ap.mul(alpha);

        // Constraint 4: Read consistency per value column
        for v in 0..layout.num_val_columns {
            let sv_z = &perm_col_evals_at_z[layout.sorted_val_start + v];
            let sv_wz = &perm_shifted_evals[layout.sorted_val_start + v];
            let srw_wz = &perm_shifted_evals[layout.sorted_rw];

            let body4 = isa_wz.mul(&one.sub(srw_wz)).mul(&sv_wz.sub(sv_z));
            result = result.add(&ap.mul(&body4).mul(&exclusion));
            ap = ap.mul(alpha);
        }
    }

    result
}

/// Evaluate the grand product transition and boundary constraints at point z.
///
/// Grand product transition:
///   `Z(ω·z) · denom(z) - Z(z) · numer(z) = 0`
/// where:
///   numer(z) = γ + addr(z) + δ·val(z) + δ²·ts(z) + δ³·rw(z)    (original trace)
///   denom(z) = γ + s_addr(z) + δ·s_val(z) + δ²·s_ts(z) + δ³·s_rw(z)  (sorted trace)
///
/// Boundary:
///   `L_0(z) · (Z(z) - 1) = 0`
/// where L_0(z) = (z^n - 1) / (n · (z - 1)) is the first Lagrange basis polynomial.
///
/// Returns the combined constraint value, accumulated with alpha powers.
pub fn evaluate_grand_product_at_point(
    perm_col_evals_at_z: &[Scalar],
    perm_shifted_evals: &[Scalar],
    col_evals_at_z: &[Scalar],
    layout: &MemoryPermutationLayout,
    addr_col: usize,
    val_cols: &[usize],
    store_sels: &[usize],
    gamma: &Scalar,
    delta: &Scalar,
    z: &Scalar,
    domain_size: u64,
    alpha: &Scalar,
    alpha_offset: usize,
) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);

    let mut ap = Scalar::one(curve);
    for _ in 0..alpha_offset {
        ap = ap.mul(alpha);
    }
    let mut result = Scalar::zero(curve);

    // Compute numer(z) = γ + addr(z) + δ·val_combined(z) + δ²·ts(z) + δ³·rw(z)
    let delta2 = delta.mul(delta);
    let delta3 = delta2.mul(delta);

    let addr_z = &col_evals_at_z[addr_col];
    // rw = sum of all store selector evaluations at z
    let mut rw_z = Scalar::zero(curve);
    for &ss in store_sels {
        rw_z = rw_z.add(&col_evals_at_z[ss]);
    }
    let ts_z = &perm_col_evals_at_z[layout.original_ts];

    let val_combined_z = if val_cols.len() == 1 {
        col_evals_at_z[val_cols[0]].clone()
    } else {
        let delta4 = delta2.mul(&delta2);
        let mut combined = Scalar::zero(curve);
        let mut delta_pow = Scalar::one(curve);
        for &vc in val_cols {
            combined = combined.add(&delta_pow.mul(&col_evals_at_z[vc]));
            delta_pow = delta_pow.mul(&delta4);
        }
        combined
    };

    let numer_z = gamma.add(addr_z)
        .add(&delta.mul(&val_combined_z))
        .add(&delta2.mul(ts_z))
        .add(&delta3.mul(&rw_z));

    // Compute denom(z) from sorted columns
    let sorted_addr_z = &perm_col_evals_at_z[layout.sorted_addr];
    let sorted_ts_z = &perm_col_evals_at_z[layout.sorted_ts];
    let sorted_rw_z = &perm_col_evals_at_z[layout.sorted_rw];

    let sorted_val_combined_z = if layout.num_val_columns == 1 {
        perm_col_evals_at_z[layout.sorted_val_start].clone()
    } else {
        let delta4 = delta2.mul(&delta2);
        let mut combined = Scalar::zero(curve);
        let mut delta_pow = Scalar::one(curve);
        for v in 0..layout.num_val_columns {
            combined = combined.add(&delta_pow.mul(&perm_col_evals_at_z[layout.sorted_val_start + v]));
            delta_pow = delta_pow.mul(&delta4);
        }
        combined
    };

    let denom_z = gamma.add(sorted_addr_z)
        .add(&delta.mul(&sorted_val_combined_z))
        .add(&delta2.mul(sorted_ts_z))
        .add(&delta3.mul(sorted_rw_z));

    // Grand product transition: Z(ω·z) · denom(z) - Z(z) · numer(z) = 0
    let z_at_z = &perm_col_evals_at_z[layout.z_column];
    let z_at_wz = &perm_shifted_evals[layout.z_column];

    let body_gp = z_at_wz.mul(&denom_z).sub(&z_at_z.mul(&numer_z));
    result = result.add(&ap.mul(&body_gp));
    ap = ap.mul(alpha);

    // Boundary: L_0(z) · (Z(z) - 1) = 0
    // L_0(z) = (z^n - 1) / (n · (z - 1))
    let n = domain_size;
    let mut z_n = Scalar::one(curve);
    let mut base = z.clone();
    let mut exp = n;
    while exp > 0 {
        if exp & 1 == 1 {
            z_n = z_n.mul(&base);
        }
        base = base.mul(&base);
        exp >>= 1;
    }
    let z_n_minus_1 = z_n.sub(&one);
    let n_scalar = Scalar::from_u64(n, curve);
    let z_minus_1 = z.sub(&one);
    // L_0(z) = (z^n - 1) / (n * (z - 1))
    let denom_l0 = n_scalar.mul(&z_minus_1);
    let l0_z = if !denom_l0.is_zero() {
        z_n_minus_1.mul(&denom_l0.inverse())
    } else {
        // z = 1 edge case: L_0(1) = 1
        one.clone()
    };

    let body_boundary = l0_z.mul(&z_at_z.sub(&one));
    result = result.add(&ap.mul(&body_boundary));

    result
}

/// Number of constraints produced by the permutation argument.
///
/// 1 (is_same_addr binary) + 2 (addr consistency) + num_val_columns (read consistency)
/// + 1 (grand product transition) + 1 (boundary Z(1)=1)
/// = 5 + num_val_columns
pub fn num_permutation_constraints(layout: &MemoryPermutationLayout) -> usize {
    5 + layout.num_val_columns
}

// ── Register file permutation ──────────────────────────────────────────

/// Layout for register file permutation auxiliary columns.
///
/// For k register ports per row, the sorted table stores k entries per row
/// in interleaved order. Each entry has (reg, val, ts, rw) = 4 columns.
///
/// Column layout (offsets from start of register permutation columns):
/// - `k` lanes × 4 columns each = `4k` sorted columns
/// - 1 Z accumulator
/// - `k` is_same_reg columns (one per transition)
/// - `k` inv_reg_diff columns
/// - 1 row_ts column (row index polynomial for timestamp computation)
/// Total: 6k + 2
///
/// The numer side uses trace column evaluations (already committed) for reg/val,
/// and the row_ts polynomial for timestamps. Port rw is a known constant.
#[derive(Debug, Clone)]
pub struct RegisterPermutationLayout {
    /// Number of register ports.
    pub num_ports: usize,
    /// Offset of Z accumulator.
    pub z_column: usize,
    /// Offset of first is_same_reg column.
    pub is_same_reg_start: usize,
    /// Offset of first inv_reg_diff column.
    pub inv_reg_diff_start: usize,
    /// Offset of the row timestamp column (row_ts[i] = i).
    pub row_ts: usize,
    /// Total number of register permutation columns.
    pub num_columns: usize,
}

impl RegisterPermutationLayout {
    /// Create a layout for `num_ports` register access ports.
    ///
    /// RISC-V (3 ports): 20 columns.
    /// SBF (2 ports): 14 columns.
    pub fn new(num_ports: usize) -> Self {
        let sorted_cols = 4 * num_ports; // k lanes × 4 columns each
        let z_column = sorted_cols;
        let is_same_reg_start = z_column + 1;
        let inv_reg_diff_start = is_same_reg_start + num_ports;
        let row_ts = inv_reg_diff_start + num_ports;
        let num_columns = row_ts + 1;

        RegisterPermutationLayout {
            num_ports,
            z_column,
            is_same_reg_start,
            inv_reg_diff_start,
            row_ts,
            num_columns,
        }
    }

    /// Offset of sorted_reg column for lane `lane`.
    pub fn sorted_reg(&self, lane: usize) -> usize {
        4 * lane
    }
    /// Offset of sorted_val column for lane `lane`.
    pub fn sorted_val(&self, lane: usize) -> usize {
        4 * lane + 1
    }
    /// Offset of sorted_ts column for lane `lane`.
    pub fn sorted_ts(&self, lane: usize) -> usize {
        4 * lane + 2
    }
    /// Offset of sorted_rw column for lane `lane`.
    pub fn sorted_rw(&self, lane: usize) -> usize {
        4 * lane + 3
    }
}

/// Number of constraints for register file permutation.
///
/// Per lane (k total):
///   - 1 is_same_reg binary check
///   - 1 address continuity (cross-row within lane or inter-lane)
///   - 1 address difference inverse
///   - 1 read consistency
/// Plus shared:
///   - 1 grand product transition
///   - 1 boundary Z(1) = 1
/// Total: 4k + 2
pub fn num_register_perm_constraints(layout: &RegisterPermutationLayout) -> usize {
    4 * layout.num_ports + 2
}

/// Evaluate register permutation constraints at a single point z.
///
/// The numer side uses trace column evaluations (reg, val already committed)
/// and the row_ts polynomial for timestamps. Port rw is a known constant.
///
/// Returns the combined constraint value at z, using alpha powers starting
/// from `alpha_offset`.
pub fn evaluate_register_perm_at_point(
    reg_evals_at_z: &[Scalar],
    reg_shifted_evals: &[Scalar],
    col_evals_at_z: &[Scalar],
    layout: &RegisterPermutationLayout,
    reg_ports: &[(usize, usize, bool)],
    gamma: &Scalar,
    delta: &Scalar,
    z: &Scalar,
    omega_n_minus_1: &Scalar,
    domain_size: u64,
    alpha: &Scalar,
    alpha_offset: usize,
) -> Scalar {
    let curve = alpha.curve_type();
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let delta2 = delta.mul(delta);
    let delta3 = delta2.mul(delta);
    let num_ports = layout.num_ports;

    let mut ap = Scalar::one(curve);
    for _ in 0..alpha_offset {
        ap = ap.mul(alpha);
    }
    let mut result = zero.clone();

    let n = domain_size;
    let n_scalar = Scalar::from_u64(n, curve);

    // z^n - 1
    let mut z_n = Scalar::one(curve);
    {
        let mut base = z.clone();
        let mut exp = n;
        while exp > 0 {
            if exp & 1 == 1 { z_n = z_n.mul(&base); }
            base = base.mul(&base);
            exp >>= 1;
        }
    }
    let z_n_minus_1 = z_n.sub(&one);

    let exclusion = z.sub(omega_n_minus_1);

    // Per-lane sorted consistency constraints (4 per lane)
    for lane in 0..num_ports {
        let isa_z = &reg_evals_at_z[layout.is_same_reg_start + lane];
        let isa_shifted = &reg_shifted_evals[layout.is_same_reg_start + lane];
        let inv_z = &reg_evals_at_z[layout.inv_reg_diff_start + lane];

        let sr_z = &reg_evals_at_z[layout.sorted_reg(lane)];
        let sr_shifted = &reg_shifted_evals[layout.sorted_reg(lane)];
        let sv_z = &reg_evals_at_z[layout.sorted_val(lane)];
        let sv_shifted = &reg_shifted_evals[layout.sorted_val(lane)];
        let srw_shifted = &reg_shifted_evals[layout.sorted_rw(lane)];

        // is_same_reg binary: is_same_reg * (is_same_reg - 1) = 0
        let body_bin = isa_z.mul(&isa_z.sub(&one));
        result = result.add(&ap.mul(&body_bin));
        ap = ap.mul(alpha);

        // Address continuity: is_same_reg(ω·z) * (sorted_reg(ω·z) - sorted_reg(z)) * exclusion = 0
        let reg_diff = sr_shifted.sub(sr_z);
        let body_cont = isa_shifted.mul(&reg_diff).mul(&exclusion);
        result = result.add(&ap.mul(&body_cont));
        ap = ap.mul(alpha);

        // Address difference: (1 - is_same_reg(ω·z)) * (1 - diff * inv_diff) * exclusion = 0
        let one_minus_isa = one.sub(isa_shifted);
        let diff_times_inv = reg_diff.mul(inv_z);
        let one_minus_prod = one.sub(&diff_times_inv);
        let body_inv = one_minus_isa.mul(&one_minus_prod).mul(&exclusion);
        result = result.add(&ap.mul(&body_inv));
        ap = ap.mul(alpha);

        // Read consistency: is_same(ω·z) * (1 - rw(ω·z)) * (val(ω·z) - val(z)) * exclusion = 0
        let one_minus_rw = one.sub(srw_shifted);
        let val_diff = sv_shifted.sub(sv_z);
        let body_read = isa_shifted.mul(&one_minus_rw).mul(&val_diff).mul(&exclusion);
        result = result.add(&ap.mul(&body_read));
        ap = ap.mul(alpha);
    }

    // Grand product transition: Z(ω·z) * ∏ denom - Z(z) * ∏ numer = 0
    let z_at_z = &reg_evals_at_z[layout.z_column];
    let z_at_omega_z = &reg_shifted_evals[layout.z_column];
    let row_ts_z = &reg_evals_at_z[layout.row_ts];
    let p_scalar = Scalar::from_u64(num_ports as u64, curve);

    let mut numer_product = Scalar::one(curve);
    let mut denom_product = Scalar::one(curve);

    for (port_idx, &(reg_col, val_col, is_write)) in reg_ports.iter().enumerate() {
        // numer: γ + trace_reg(z) + δ·trace_val(z) + δ²·(P·row_ts(z) + p) + δ³·rw
        let port_reg_z = &col_evals_at_z[reg_col];
        let port_val_z = &col_evals_at_z[val_col];
        let port_ts_z = p_scalar.mul(row_ts_z).add(&Scalar::from_u64(port_idx as u64, curve));
        let rw_val = if is_write { one.clone() } else { zero.clone() };

        let numer = gamma.add(port_reg_z)
            .add(&delta.mul(port_val_z))
            .add(&delta2.mul(&port_ts_z))
            .add(&delta3.mul(&rw_val));
        numer_product = numer_product.mul(&numer);

        // denom: γ + sorted_reg_lane(z) + δ·sorted_val_lane(z) + δ²·sorted_ts_lane(z) + δ³·sorted_rw_lane(z)
        let s_reg = &reg_evals_at_z[layout.sorted_reg(port_idx)];
        let s_val = &reg_evals_at_z[layout.sorted_val(port_idx)];
        let s_ts = &reg_evals_at_z[layout.sorted_ts(port_idx)];
        let s_rw = &reg_evals_at_z[layout.sorted_rw(port_idx)];
        let denom = gamma.add(s_reg)
            .add(&delta.mul(s_val))
            .add(&delta2.mul(s_ts))
            .add(&delta3.mul(s_rw));
        denom_product = denom_product.mul(&denom);
    }

    let gp_body = z_at_omega_z.mul(&denom_product).sub(&z_at_z.mul(&numer_product));
    result = result.add(&ap.mul(&gp_body));
    ap = ap.mul(alpha);

    // Boundary: L_0(z) * (Z(z) - 1) = 0
    let z_minus_1 = z.sub(&one);
    let denom_l0 = n_scalar.mul(&z_minus_1);
    let l0_z = if !denom_l0.is_zero() {
        z_n_minus_1.mul(&denom_l0.inverse())
    } else {
        one.clone()
    };
    let body_boundary = l0_z.mul(&z_at_z.sub(&one));
    result = result.add(&ap.mul(&body_boundary));

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;

    #[test]
    fn test_layout_riscv_sbf() {
        let layout = MemoryPermutationLayout::new(1);
        assert_eq!(layout.num_columns, 8);
        assert_eq!(layout.sorted_addr, 0);
        assert_eq!(layout.sorted_val_start, 1);
        assert_eq!(layout.sorted_ts, 2);
        assert_eq!(layout.sorted_rw, 3);
        assert_eq!(layout.z_column, 4);
        assert_eq!(layout.is_same_addr, 5);
        assert_eq!(layout.inv_addr_diff, 6);
        assert_eq!(layout.original_ts, 7);
    }

    #[test]
    fn test_layout_evm() {
        let layout = MemoryPermutationLayout::new(4);
        assert_eq!(layout.num_columns, 11);
        assert_eq!(layout.sorted_val_start, 1);
        assert_eq!(layout.sorted_ts, 5);
        assert_eq!(layout.sorted_rw, 6);
        assert_eq!(layout.z_column, 7);
        assert_eq!(layout.original_ts, 10);
    }

    #[test]
    fn test_sort_and_aux() {
        let curve = CurveType::Bls48581;
        let accesses = vec![
            MemoryAccess { addr: 100, values: vec![42], timestamp: 0, rw: 1 },
            MemoryAccess { addr: 100, values: vec![42], timestamp: 1, rw: 0 },
            MemoryAccess { addr: 200, values: vec![99], timestamp: 2, rw: 1 },
            MemoryAccess { addr: 100, values: vec![55], timestamp: 3, rw: 1 },
        ];

        let (sorted, is_same, inv_diff) = sort_and_compute_aux(&accesses, curve);

        // Should be sorted by (addr, ts): (100,0), (100,1), (100,3), (200,2)
        assert_eq!(sorted[0].addr, 100);
        assert_eq!(sorted[0].timestamp, 0);
        assert_eq!(sorted[1].addr, 100);
        assert_eq!(sorted[1].timestamp, 1);
        assert_eq!(sorted[2].addr, 100);
        assert_eq!(sorted[2].timestamp, 3);
        assert_eq!(sorted[3].addr, 200);
        assert_eq!(sorted[3].timestamp, 2);

        // is_same_addr: [0, 1, 1, 0]
        assert!(is_same[0].is_zero());
        assert!(is_same[1].is_one());
        assert!(is_same[2].is_one());
        assert!(is_same[3].is_zero());

        // inv_addr_diff: [0, 0, 0, (200-100)^{-1}]
        assert!(inv_diff[0].is_zero());
        assert!(inv_diff[1].is_zero());
        assert!(inv_diff[2].is_zero());
        // Check inv_diff[3] is inverse of 100
        let hundred = Scalar::from_u64(100, curve);
        let prod = hundred.mul(&inv_diff[3]);
        assert!(prod.is_one());
    }

    #[test]
    fn test_grand_product_identity() {
        let curve = CurveType::Bls48581;
        let accesses = vec![
            MemoryAccess { addr: 100, values: vec![42], timestamp: 0, rw: 1 },
            MemoryAccess { addr: 200, values: vec![99], timestamp: 1, rw: 1 },
            MemoryAccess { addr: 100, values: vec![42], timestamp: 2, rw: 0 },
            MemoryAccess { addr: 200, values: vec![99], timestamp: 3, rw: 0 },
        ];

        let (sorted, _, _) = sort_and_compute_aux(&accesses, curve);
        let gamma = Scalar::from_u64(17, curve);
        let delta = Scalar::from_u64(31, curve);

        let z = compute_grand_product(&accesses, &sorted, &gamma, &delta, curve);

        // Z[0] should be 1
        assert!(z[0].is_one());

        // The grand product should "close" — the product of all numer/denom
        // should equal 1 since original and sorted are the same multiset.
        // Z[n-1] * numer[n-1] / denom[n-1] should equal 1.
        let delta2 = delta.mul(&delta);
        let delta3 = delta2.mul(&delta);
        let last_numer = compute_tuple_hash(
            &accesses[3], &gamma, &delta, &delta2, &delta3, curve,
        );
        let last_denom = compute_tuple_hash(
            &sorted[3], &gamma, &delta, &delta2, &delta3, curve,
        );
        let final_val = z[3].mul(&last_numer).mul(&last_denom.inverse());
        assert!(final_val.is_one(), "Grand product should close to 1");
    }

    #[test]
    fn test_num_constraints() {
        let layout1 = MemoryPermutationLayout::new(1);
        assert_eq!(num_permutation_constraints(&layout1), 6); // 5 + 1 val col

        let layout4 = MemoryPermutationLayout::new(4);
        assert_eq!(num_permutation_constraints(&layout4), 9); // 5 + 4 val cols
    }
}
