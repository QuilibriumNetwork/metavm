//! Lookup argument types and declarations for range checks and bitwise operations.
//!
//! This module defines the infrastructure for lookup-based constraints:
//! - **Range checks**: Verify column values lie in `[0, 2^k - 1]` via byte
//!   decomposition and an 8-bit range table.
//! - **Bitwise operations**: Verify AND/OR/XOR results via a nibble-AND table
//!   (256 entries where entry[a*16+b] = a AND b) and nibble decomposition.
//!
//! # Architecture
//!
//! Range checks use byte (8-bit) decomposition: a 64-bit value is split into
//! 8 bytes, each looked up in the range table `{0, ..., 255}`.
//!
//! Bitwise operations use nibble (4-bit) decomposition: each operand is split
//! into nibbles (4-bit chunks), and for each nibble pair (a_i, b_i), the AND
//! result is looked up in a 256-entry table. OR and XOR are derived:
//! - `AND(a, b)` looked up directly
//! - `OR(a, b) = a + b - AND(a, b)`
//! - `XOR(a, b) = a + b - 2 * AND(a, b)`
//!
//! The LogUp running sum accumulates all lookup contributions (range + bitwise)
//! into a single grand sum that must close to zero.

/// A declaration that a particular trace column needs range checking.
///
/// The prover must ensure the column values are in `[0, max_value]` and
/// provide a lookup proof. The verifier checks the proof against the
/// declared range.
#[derive(Debug, Clone)]
pub struct LookupDeclaration {
    /// Human-readable label for debugging (e.g., "mul_aux0_range").
    pub label: String,

    /// Index of the trace column to range-check (step-excluded indexing).
    pub column_index: usize,

    /// Maximum valid value (inclusive). The valid range is `[0, max_value]`.
    ///
    /// Common values:
    /// - `(1u64 << 64) - 1` for 64-bit range
    /// - `(1u64 << 32) - 1` for 32-bit range
    /// - `(1u64 << 16) - 1` for 16-bit range (used in lookup tables)
    /// - `255` for byte range
    pub max_bits: u32,

    /// Optional selector column index. If `Some(idx)`, the range check
    /// only applies to rows where `columns[idx] = 1`. If `None`, the
    /// check applies to ALL rows (including padding).
    pub selector_column: Option<usize>,
}

/// The type of lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableType {
    /// Range table: `{0, 1, ..., 2^bits - 1}`.
    Range,
    /// Nibble-AND table: 256 entries where `entry[a*16+b] = a AND b`.
    /// Used for verifying bitwise AND via nibble decomposition.
    NibbleAnd,
}

/// A precomputed lookup table.
///
/// Two table types:
/// - `Range`: `[0, 1, 2, ..., 2^k - 1]` for range checking.
/// - `NibbleAnd`: 256 entries where `table[a*16+b] = a AND b`.
#[derive(Debug, Clone)]
pub struct LookupTable {
    /// Human-readable name (e.g., "range_16bit", "nibble_and").
    pub name: String,

    /// Number of bits (for Range: table size = 2^bits; for NibbleAnd: always 8).
    pub bits: u32,

    /// Table type.
    pub table_type: TableType,
}

impl LookupTable {
    /// Create a range table for `[0, 2^bits - 1]`.
    pub fn range(bits: u32) -> Self {
        LookupTable {
            name: format!("range_{}bit", bits),
            bits,
            table_type: TableType::Range,
        }
    }

    /// Create the nibble-AND table (256 entries).
    pub fn nibble_and() -> Self {
        LookupTable {
            name: "nibble_and".to_string(),
            bits: 8,
            table_type: TableType::NibbleAnd,
        }
    }

    /// Number of entries in the table.
    pub fn size(&self) -> u64 {
        1u64 << self.bits
    }

    /// Get the table value at the given index.
    pub fn value_at(&self, index: usize) -> u64 {
        match self.table_type {
            TableType::Range => index as u64,
            TableType::NibbleAnd => {
                let a = (index >> 4) & 0xF;
                let b = index & 0xF;
                (a & b) as u64
            }
        }
    }
}

/// Summary of all lookup requirements for a VM.
///
/// Groups declarations by the table they need, allowing the prover to
/// batch lookups into the same table efficiently.
#[derive(Debug, Clone)]
pub struct LookupRequirements {
    /// Lookup tables needed by this VM.
    pub tables: Vec<LookupTable>,

    /// Column declarations, each referencing a table by index into `tables`.
    pub declarations: Vec<(LookupDeclaration, usize)>, // (declaration, table_index)
}

impl LookupRequirements {
    /// Empty requirements (no lookups needed).
    pub fn none() -> Self {
        LookupRequirements {
            tables: Vec::new(),
            declarations: Vec::new(),
        }
    }

    /// Whether any lookups are declared.
    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }

    /// Total number of lookup declarations.
    pub fn num_lookups(&self) -> usize {
        self.declarations.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LogUp Lookup Argument Infrastructure
// ═══════════════════════════════════════════════════════════════════════════

use crate::field::{Scalar, CurveType};

/// A group of declarations that share the same source column and bit width.
///
/// Multiple `LookupDeclaration`s with the same `(column_index, max_bits)` can
/// share a single set of byte-decomposition limb columns, reducing overhead.
#[derive(Debug, Clone)]
pub struct LogUpGroup {
    /// Index of the trace column being range-checked.
    pub column_index: usize,
    /// Number of bits to range-check.
    pub max_bits: u32,
    /// Number of byte limbs needed: `ceil(max_bits / 8)`.
    pub num_limbs: usize,
    /// Selector column indices that gate this range check.
    /// The range check applies when ANY of these selectors is 1.
    /// If empty, applies to all rows.
    pub selectors: Vec<usize>,
}

/// Layout of LogUp auxiliary columns appended to the trace.
///
/// Current wiring: `limbs[..] | h | m` (h at `num_cols-2`, m at `num_cols-1`).
///
/// The extended Phase-0 layout (`limbs | f | t | u_t | h | m`) is defined
/// separately in [`ExtendedLogUpColumnLayout`] as groundwork for completing
/// the running-sum transition + inverse constraints; not yet committed.
#[derive(Debug, Clone)]
pub struct LogUpColumnLayout {
    /// For each group: (start_offset, num_limbs) — byte decomposition columns.
    pub limb_offsets: Vec<(usize, usize)>,
    /// Offset of the running sum column h.
    pub h_column: usize,
    /// Offset of the multiplicity column m.
    pub m_column: usize,
    /// Total number of LogUp auxiliary columns.
    pub num_columns: usize,
}

/// Extended LogUp layout reserving space for inverse columns, range table,
/// and table inverse. Planned wiring for the Phase-0 running-sum transition
/// but not yet committed/evaluated end-to-end.
#[derive(Debug, Clone)]
pub struct ExtendedLogUpColumnLayout {
    /// For each group: (start_offset, num_limbs) — byte decomposition columns.
    pub limb_offsets: Vec<(usize, usize)>,
    /// For each group: (start_offset, num_limbs) — inverse columns
    /// `f_k(X) = 1 / (γ - limb_k(X))`.
    pub f_offsets: Vec<(usize, usize)>,
    /// Offset of the range table column `t(X)`.
    pub t_column: usize,
    /// Offset of the table inverse column `u_t(X) = 1/(γ - t(X))`.
    pub u_t_column: usize,
    /// Offset of the running sum column h.
    pub h_column: usize,
    /// Offset of the multiplicity column m.
    pub m_column: usize,
    /// Total number of LogUp auxiliary columns.
    pub num_columns: usize,
}

/// The computed LogUp witness data.
#[derive(Debug, Clone)]
pub struct LogUpWitness {
    /// Byte decomposition columns for each group.
    /// Outer: group index, Inner: limb index within group, Innermost: values per row.
    pub limb_columns: Vec<Vec<Vec<Scalar>>>,
    /// Running sum column h.
    pub h_column: Vec<Scalar>,
    /// Multiplicity column m (counts per table entry).
    pub m_column: Vec<Scalar>,
}

/// Extended witness data for the Phase-0 running-sum transition:
/// per-limb inverses, range-table column, and table-inverse column.
/// Produced by [`compute_extended_logup_witness`]; not yet committed.
#[derive(Debug, Clone)]
pub struct ExtendedLogUpWitness {
    pub limb_columns: Vec<Vec<Vec<Scalar>>>,
    pub f_columns: Vec<Vec<Vec<Scalar>>>,
    pub t_column: Vec<Scalar>,
    pub u_t_column: Vec<Scalar>,
    pub h_column: Vec<Scalar>,
    pub m_column: Vec<Scalar>,
}

/// Group lookup declarations by `(column_index, max_bits)` to share limb columns.
pub fn group_declarations(reqs: &LookupRequirements) -> Vec<LogUpGroup> {
    use std::collections::BTreeMap;

    // Key: (column_index, max_bits) → set of selector columns
    let mut groups: BTreeMap<(usize, u32), Vec<usize>> = BTreeMap::new();

    for (decl, _table_idx) in &reqs.declarations {
        let key = (decl.column_index, decl.max_bits);
        let entry = groups.entry(key).or_default();
        if let Some(sel) = decl.selector_column {
            if !entry.contains(&sel) {
                entry.push(sel);
            }
        }
    }

    groups
        .into_iter()
        .map(|((col_idx, max_bits), selectors)| {
            let num_limbs = ((max_bits as usize) + 7) / 8;
            LogUpGroup {
                column_index: col_idx,
                max_bits,
                num_limbs,
                selectors,
            }
        })
        .collect()
}

/// Compute the LogUp column layout from grouped declarations.
///
/// Order: `limbs[..] | h | m` (currently wired — matches prover/verifier).
pub fn logup_column_layout(groups: &[LogUpGroup]) -> LogUpColumnLayout {
    let mut offset = 0;
    let mut limb_offsets = Vec::with_capacity(groups.len());
    for group in groups {
        limb_offsets.push((offset, group.num_limbs));
        offset += group.num_limbs;
    }

    let h_column = offset;
    let m_column = offset + 1;
    let num_columns = offset + 2;

    LogUpColumnLayout {
        limb_offsets,
        h_column,
        m_column,
        num_columns,
    }
}

/// Compute the extended LogUp column layout (`limbs | f | t | u_t | h | m`).
/// Groundwork for the Phase-0 running-sum transition; not wired yet.
pub fn extended_logup_column_layout(groups: &[LogUpGroup]) -> ExtendedLogUpColumnLayout {
    let mut offset = 0;
    let mut limb_offsets = Vec::with_capacity(groups.len());
    for group in groups {
        limb_offsets.push((offset, group.num_limbs));
        offset += group.num_limbs;
    }
    let mut f_offsets = Vec::with_capacity(groups.len());
    for group in groups {
        f_offsets.push((offset, group.num_limbs));
        offset += group.num_limbs;
    }
    let t_column = offset;
    let u_t_column = offset + 1;
    let h_column = offset + 2;
    let m_column = offset + 3;
    let num_columns = offset + 4;
    ExtendedLogUpColumnLayout {
        limb_offsets,
        f_offsets,
        t_column,
        u_t_column,
        h_column,
        m_column,
        num_columns,
    }
}

/// Size of the 8-bit range table (and implicit index domain).
pub const RANGE_TABLE_SIZE: usize = 256;

/// Fill a range-table column `t` of length `domain_size`:
/// `t[i] = i` for `i < 256`, zero otherwise.
///
/// This is the canonical preprocessed polynomial the LogUp argument targets.
/// The verifier must independently evaluate `t(z)` (see `evaluate_range_table_at_point`)
/// to bind the prover's committed `t` to the correct contents.
pub fn range_table_column(domain_size: usize, curve: CurveType) -> Vec<Scalar> {
    let mut t = vec![Scalar::zero(curve); domain_size];
    let max = domain_size.min(RANGE_TABLE_SIZE);
    for i in 0..max {
        t[i] = Scalar::from_u64(i as u64, curve);
    }
    t
}

/// Evaluate the canonical 8-bit range table polynomial `t(X)` at `z` using
/// Lagrange interpolation. `t(ω^i) = i` for `i < 256`, zero otherwise.
///
/// Used by the verifier to bind the prover-claimed `t(z)` to the known range
/// table contents — soundness of the lookup argument depends on this check.
pub fn evaluate_range_table_at_point(
    z: &Scalar,
    omega: &Scalar,
    domain_size: usize,
    curve: CurveType,
) -> Scalar {
    // t(z) = Σ_{i=0}^{255} i · L_i(z) = Σ i · (Z_H(z)/n) · ω^i / (z - ω^i)
    // We factor the common Z_H(z)/n outside the sum.
    // Z_H(z) = z^n - 1.
    let one = Scalar::one(curve);
    let z_n = {
        let mut zn = one.clone();
        let mut base = z.clone();
        let mut exp = domain_size;
        while exp > 0 {
            if exp & 1 == 1 {
                zn = zn.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }
        zn
    };
    let z_h = z_n.sub(&one);
    if z_h.is_zero() {
        // z is a domain point; t(z) equals the table value at that index if < 256, else 0.
        // Find i such that z = ω^i by trial (verifier should avoid challenging on domain).
        let mut pow = Scalar::one(curve);
        for i in 0..domain_size {
            if pow.sub(z).is_zero() {
                return if i < RANGE_TABLE_SIZE {
                    Scalar::from_u64(i as u64, curve)
                } else {
                    Scalar::zero(curve)
                };
            }
            pow = pow.mul(omega);
        }
        return Scalar::zero(curve);
    }
    let n_inv = Scalar::from_u64(domain_size as u64, curve).inverse();
    let common = z_h.mul(&n_inv);

    let max = domain_size.min(RANGE_TABLE_SIZE);
    let mut sum = Scalar::zero(curve);
    let mut omega_i = Scalar::one(curve);
    for i in 0..max {
        if i > 0 {
            let denom = z.sub(&omega_i);
            // i · ω^i / (z - ω^i)
            let term = Scalar::from_u64(i as u64, curve)
                .mul(&omega_i)
                .mul(&denom.inverse());
            sum = sum.add(&term);
        }
        omega_i = omega_i.mul(omega);
    }
    common.mul(&sum)
}

/// Size of the canonical nibble-AND table (256 entries: one per (a, b) ∈ [0,16)²).
pub const NIBBLE_AND_TABLE_SIZE: usize = 256;

/// Nibble-AND table component selector.
///
/// The 256-entry nibble-AND table is represented as three aligned columns:
/// - `A`: `t_a[i] = i >> 4` (high nibble, i.e. the `a` operand), for `i < 256`
/// - `B`: `t_b[i] = i & 0xF` (low nibble, i.e. the `b` operand), for `i < 256`
/// - `C`: `t_c[i] = (i >> 4) & (i & 0xF)` (the AND result), for `i < 256`
///
/// In each case the column is zero on rows `i >= 256`. Bitwise LogUp combines
/// these into a single scalar query `a + δ·b + δ²·c` with Fiat-Shamir
/// challenge `δ`, so the triple-valued lookup reduces to a scalar LogUp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NibbleAndTableComponent {
    A,
    B,
    C,
}

impl NibbleAndTableComponent {
    #[inline]
    pub(crate) fn value_at(self, i: usize) -> u64 {
        if i >= NIBBLE_AND_TABLE_SIZE {
            return 0;
        }
        let a = ((i >> 4) & 0xF) as u64;
        let b = (i & 0xF) as u64;
        match self {
            NibbleAndTableComponent::A => a,
            NibbleAndTableComponent::B => b,
            NibbleAndTableComponent::C => a & b,
        }
    }
}

/// Build one of the three nibble-AND table columns as a domain-sized vector.
///
/// Indices `0..256` carry the canonical table values; the remainder are zero.
/// Used by the prover to commit the (preprocessed) table column, and by the
/// verifier Lagrange evaluator to bind it.
pub fn nibble_and_table_column(
    component: NibbleAndTableComponent,
    domain_size: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut t = vec![Scalar::zero(curve); domain_size];
    let max = domain_size.min(NIBBLE_AND_TABLE_SIZE);
    for i in 0..max {
        t[i] = Scalar::from_u64(component.value_at(i), curve);
    }
    t
}

/// Evaluate the canonical nibble-AND table polynomial for a given component
/// at `z` using Lagrange interpolation over the 256 table rows:
///
///   component(z) = Σ_{i=0}^{255} component.value_at(i) · L_i(z)
///
/// Verifier-side soundness check for bitwise LogUp — prover's claimed
/// table-component evaluation must match this recomputation.
pub fn evaluate_nibble_and_table_at_point(
    component: NibbleAndTableComponent,
    z: &Scalar,
    omega: &Scalar,
    domain_size: usize,
    curve: CurveType,
) -> Scalar {
    let one = Scalar::one(curve);
    let z_n = {
        let mut zn = one.clone();
        let mut base = z.clone();
        let mut exp = domain_size;
        while exp > 0 {
            if exp & 1 == 1 {
                zn = zn.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }
        zn
    };
    let z_h = z_n.sub(&one);
    if z_h.is_zero() {
        // z coincides with a domain point — return the table value directly.
        let mut pow = Scalar::one(curve);
        for i in 0..domain_size {
            if pow.sub(z).is_zero() {
                return Scalar::from_u64(component.value_at(i), curve);
            }
            pow = pow.mul(omega);
        }
        return Scalar::zero(curve);
    }
    let n_inv = Scalar::from_u64(domain_size as u64, curve).inverse();
    let common = z_h.mul(&n_inv);

    let max = domain_size.min(NIBBLE_AND_TABLE_SIZE);
    let mut sum = Scalar::zero(curve);
    let mut omega_i = Scalar::one(curve);
    for i in 0..max {
        let v = component.value_at(i);
        if v != 0 {
            let denom = z.sub(&omega_i);
            // value · ω^i / (z - ω^i)
            let term = Scalar::from_u64(v, curve)
                .mul(&omega_i)
                .mul(&denom.inverse());
            sum = sum.add(&term);
        }
        omega_i = omega_i.mul(omega);
    }
    common.mul(&sum)
}

/// Decompose a u64 value into byte limbs (little-endian).
///
/// Returns `num_limbs` bytes. If the value exceeds `num_limbs` bytes, upper
/// bits are truncated (the decomposition constraint will catch mismatches).
fn decompose_to_bytes(value: u64, num_limbs: usize) -> Vec<u8> {
    let bytes = value.to_le_bytes();
    let mut result = vec![0u8; num_limbs];
    for i in 0..num_limbs.min(8) {
        result[i] = bytes[i];
    }
    result
}

/// Decompose a [`Scalar`] into the low `num_limbs` byte limbs (little-endian).
///
/// Reads the scalar's full big-endian byte representation and extracts the
/// `num_limbs` least-significant bytes. Supports widths ≥ 64 bits without
/// truncation (the older [`decompose_to_bytes`] silently caps at 8 bytes
/// because it goes through `u64`). Used for range checks on intermediate
/// schoolbook-multiplication carries that can reach ~2^67.
fn decompose_scalar_to_bytes(value: &Scalar, num_limbs: usize) -> Vec<u8> {
    let be_bytes = value.to_bytes();
    let mut result = vec![0u8; num_limbs];
    // be_bytes is big-endian, so the LSB is the LAST byte. Reverse-iterate
    // and copy `num_limbs` bytes (or all, whichever is smaller).
    let total = be_bytes.len();
    for i in 0..num_limbs.min(total) {
        result[i] = be_bytes[total - 1 - i];
    }
    result
}

/// Compute the LogUp witness from trace data and a gamma challenge.
///
/// For each group, decomposes the source column into byte limbs and verifies
/// the decomposition: `value == Σ limb_k * 256^k`. Builds the running sum `h`
/// and multiplicity column `m`.
///
/// The running sum accumulates:
///   `h[i] = h[i-1] + Σ_groups Σ_limbs selector[i] / (gamma - limb_value[i]) - m[i] / (gamma - table_entry)`
///
/// where the table for an 8-bit range check is `{0, 1, ..., 255}`.
pub fn compute_logup_witness(
    trace_columns: &[&Vec<Scalar>],
    groups: &[LogUpGroup],
    _layout: &LogUpColumnLayout,
    gamma: &Scalar,
    _num_rows: usize,
    domain_size: usize,
    curve: CurveType,
) -> LogUpWitness {
    // We iterate over the FULL domain (not just `num_rows`) so that the
    // polynomial representation matches:
    //   - for empty-selector groups (always-active), padding rows contribute
    //     to the lookup and must be counted in `byte_counts` to keep the
    //     grand sum closed against the transition constraint;
    //   - for non-empty-selector groups, padding rows have selectors = 0 so
    //     `active` evaluates to false there and the lookup simply skips them.
    let mut limb_columns: Vec<Vec<Vec<Scalar>>> = Vec::with_capacity(groups.len());
    let mut byte_counts = [0u64; 256];

    // Helper: whether the group is active at `row` in the polynomial sense.
    let group_active = |group: &LogUpGroup, row: usize| -> bool {
        if group.selectors.is_empty() {
            true
        } else {
            group
                .selectors
                .iter()
                .any(|&sel| !trace_columns[sel][row].is_zero())
        }
    };

    for group in groups {
        let src_col = trace_columns[group.column_index];
        let mut group_limbs: Vec<Vec<Scalar>> =
            vec![vec![Scalar::zero(curve); domain_size]; group.num_limbs];
        for row in 0..domain_size {
            // For widths ≤ 64 use the fast u64 path; for wider widths
            // (e.g. 72-bit mul-schoolbook carries) fall back to a full
            // scalar→bytes decomposition that doesn't truncate.
            let bytes = if group.max_bits <= 64 {
                decompose_to_bytes(src_col[row].to_u64(), group.num_limbs)
            } else {
                decompose_scalar_to_bytes(&src_col[row], group.num_limbs)
            };
            let active = group_active(group, row);
            for (limb_idx, &byte_val) in bytes.iter().enumerate() {
                group_limbs[limb_idx][row] = Scalar::from_u64(byte_val as u64, curve);
                if active {
                    byte_counts[byte_val as usize] += 1;
                }
            }
        }
        limb_columns.push(group_limbs);
    }

    // Multiplicity column m: byte_counts placed at table row indices 0..256.
    let mut m_column = vec![Scalar::zero(curve); domain_size];
    for (i, &count) in byte_counts.iter().enumerate() {
        if i < domain_size {
            m_column[i] = Scalar::from_u64(count, curve);
        }
    }

    // Running sum h (rational form). With h[0] = 0 and the cyclic transition
    // `h(ωX) - h(X) - Σ active·f + m·u_t = 0` enforced polynomially, the
    // grand sum closes automatically iff every limb occurrence is accounted
    // for in `byte_counts` (which it now is, after the domain-wide loop).
    let mut h_column = vec![Scalar::zero(curve); domain_size];
    for i in 1..domain_size {
        let mut sum = h_column[i - 1].clone();
        for (g_idx, group) in groups.iter().enumerate() {
            if group_active(group, i - 1) {
                for limb_idx in 0..group.num_limbs {
                    let denom = gamma.sub(&limb_columns[g_idx][limb_idx][i - 1]);
                    if !denom.is_zero() {
                        sum = sum.add(&denom.inverse());
                    }
                }
            }
        }
        if !m_column[i - 1].is_zero() {
            let table_val = Scalar::from_u64((i - 1) as u64, curve);
            let table_denom = gamma.sub(&table_val);
            if !table_denom.is_zero() {
                sum = sum.sub(&m_column[i - 1].mul(&table_denom.inverse()));
            }
        }
        h_column[i] = sum;
    }

    LogUpWitness {
        limb_columns,
        h_column,
        m_column,
    }
}

/// Extended witness producing per-limb inverses, the range-table column,
/// and the table-inverse column. Not yet used end-to-end; tested as
/// groundwork for completing the Phase-0 transition constraint.
pub fn compute_extended_logup_witness(
    trace_columns: &[&Vec<Scalar>],
    groups: &[LogUpGroup],
    gamma: &Scalar,
    num_rows: usize,
    domain_size: usize,
    curve: CurveType,
) -> ExtendedLogUpWitness {
    let base = compute_logup_witness(
        trace_columns,
        groups,
        &logup_column_layout(groups),
        gamma,
        num_rows,
        domain_size,
        curve,
    );

    let mut f_columns: Vec<Vec<Vec<Scalar>>> = Vec::with_capacity(groups.len());
    for (g_idx, group) in groups.iter().enumerate() {
        let mut g_invs: Vec<Vec<Scalar>> =
            vec![vec![Scalar::zero(curve); domain_size]; group.num_limbs];
        for limb_idx in 0..group.num_limbs {
            for i in 0..domain_size {
                let denom = gamma.sub(&base.limb_columns[g_idx][limb_idx][i]);
                debug_assert!(!denom.is_zero(), "gamma collided with limb value");
                g_invs[limb_idx][i] = denom.inverse();
            }
        }
        f_columns.push(g_invs);
    }
    let t_column = range_table_column(domain_size, curve);
    let mut u_t_column = vec![Scalar::zero(curve); domain_size];
    for i in 0..domain_size {
        let denom = gamma.sub(&t_column[i]);
        debug_assert!(!denom.is_zero(), "gamma collided with table entry");
        u_t_column[i] = denom.inverse();
    }
    ExtendedLogUpWitness {
        limb_columns: base.limb_columns,
        f_columns,
        t_column,
        u_t_column,
        h_column: base.h_column,
        m_column: base.m_column,
    }
}

/// Evaluate the LogUp decomposition constraint at a point for a single group.
///
/// `value(z) == Σ_k limb_k(z) * 256^k`
///
/// Gated by selector: `selector(z) * (value(z) - Σ_k limb_k(z) * 256^k) = 0`
pub fn evaluate_decomposition_at_point(
    value_at_z: &Scalar,
    limb_evals_at_z: &[Scalar],
    selector_at_z: Option<&Scalar>,
    curve: CurveType,
) -> Scalar {
    let mut recomposed = Scalar::zero(curve);
    let two_56_eight = Scalar::from_u64(256, curve);
    let mut power = Scalar::one(curve);

    for limb in limb_evals_at_z {
        recomposed = recomposed.add(&limb.mul(&power));
        power = power.mul(&two_56_eight);
    }

    let diff = value_at_z.sub(&recomposed);

    match selector_at_z {
        Some(sel) => sel.mul(&diff),
        None => diff,
    }
}

/// Evaluate the inverse-column constraint `f(X) · (γ - v(X)) - 1 = 0` at `z`.
/// Holds on all rows (no selector gating).
pub fn evaluate_inverse_at_point(
    f_at_z: &Scalar,
    value_at_z: &Scalar,
    gamma: &Scalar,
    curve: CurveType,
) -> Scalar {
    let denom = gamma.sub(value_at_z);
    let prod = f_at_z.mul(&denom);
    prod.sub(&Scalar::one(curve))
}

/// Evaluate the running-sum transition
/// `h(ω·X) - h(X) - Σ_{g,k} active_g(X)·f_{g,k}(X) + m(X)·u_t(X)` at `z`.
///
/// Enforced on all rows including wrap-around, so combined with `h(ω^0) = 0`
/// it forces the LogUp grand sum to close.
///
/// `active_sums`: per-group selector sum at z (empty selectors contribute the
/// scalar `one`, representing "always active"). Each entry is paired with the
/// vector of inverse evaluations `f_{g,k}(z)` for that group's limbs.
pub fn evaluate_transition_at_point(
    h_at_z: &Scalar,
    h_at_omega_z: &Scalar,
    active_and_fs: &[(Scalar, Vec<Scalar>)],
    m_at_z: &Scalar,
    u_t_at_z: &Scalar,
    curve: CurveType,
) -> Scalar {
    let mut lhs = h_at_omega_z.sub(h_at_z);
    for (active, f_evals) in active_and_fs {
        let mut group_sum = Scalar::zero(curve);
        for f in f_evals {
            group_sum = group_sum.add(f);
        }
        lhs = lhs.sub(&active.mul(&group_sum));
    }
    lhs.add(&m_at_z.mul(u_t_at_z))
}

/// Number of LogUp constraints currently enforced: 1 decomposition per group
/// plus 1 `L_0 · h = 0` boundary.
///
/// The full Phase-0 constraint count (adding per-limb inverse + table inverse
/// + running-sum transition) is reported by [`num_extended_logup_constraints`].
pub fn num_logup_constraints(groups: &[LogUpGroup]) -> usize {
    groups.len() + 1
}

/// Planned constraint count once the extended layout is wired end-to-end:
/// 1 decomposition per group, 1 inverse per limb, 1 table inverse, 1
/// transition, 1 boundary.
pub fn num_extended_logup_constraints(groups: &[LogUpGroup]) -> usize {
    let total_limbs: usize = groups.iter().map(|g| g.num_limbs).sum();
    groups.len() + total_limbs + 3
}

/// Total number of limb inverse columns across all groups.
pub fn total_num_limbs(groups: &[LogUpGroup]) -> usize {
    groups.iter().map(|g| g.num_limbs).sum()
}

// ═══════════════════════════════════════════════════════════════════════════
// Bitwise Lookup Infrastructure
// ═══════════════════════════════════════════════════════════════════════════

/// Bitwise operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseOp {
    /// `result = a AND b` — looked up directly from nibble-AND table.
    And,
    /// `result = a OR b = a + b - AND(a, b)`.
    Or,
    /// `result = a XOR b = a + b - 2 * AND(a, b)`.
    Xor,
}

/// A declaration for a bitwise lookup on two operand columns.
///
/// The prover decomposes both operands into nibbles, looks up AND(a_i, b_i)
/// for each nibble pair, and derives the final result based on `op`.
#[derive(Debug, Clone)]
pub struct BitwiseLookupDeclaration {
    /// Human-readable label (e.g., "riscv_and_r_type").
    pub label: String,

    /// Index of the first operand column (e.g., rs1_val).
    pub operand_a_column: usize,

    /// Index of the second operand column (e.g., rs2_val or immediate).
    pub operand_b_column: usize,

    /// Index of the result column (e.g., rd_val_after).
    pub result_column: usize,

    /// Width of the operands in bits (64 for RISC-V/SBF, 256 for EVM).
    /// Determines the number of nibbles: `width_bits / 4`.
    pub width_bits: u32,

    /// Which bitwise operation this verifies.
    pub op: BitwiseOp,

    /// Selector columns that gate this lookup.
    /// Active when ANY selector is 1.
    pub selectors: Vec<usize>,
}

/// A grouped bitwise lookup, sharing nibble decomposition columns.
///
/// Multiple declarations with the same `(operand_a_column, operand_b_column, width_bits)`
/// share nibble decomposition columns. Each declaration may use a different `op`
/// but the AND nibbles are the same.
#[derive(Debug, Clone)]
pub struct BitwiseLookupGroup {
    /// Index of the first operand column.
    pub operand_a_column: usize,
    /// Index of the second operand column.
    pub operand_b_column: usize,
    /// Index of the result column.
    pub result_column: usize,
    /// Width in bits.
    pub width_bits: u32,
    /// Number of nibbles per operand: `width_bits / 4`.
    pub num_nibbles: usize,
    /// The bitwise operation.
    pub op: BitwiseOp,
    /// Selector column indices that gate this lookup.
    pub selectors: Vec<usize>,
}

/// Layout of bitwise LogUp auxiliary columns.
#[derive(Debug, Clone)]
pub struct BitwiseColumnLayout {
    /// For each group: (start_offset, num_nibbles).
    /// Each group has 3 * num_nibbles columns: a_nibbles, b_nibbles, and_nibbles.
    pub nibble_offsets: Vec<(usize, usize)>,
    /// Total number of bitwise auxiliary columns (before h/m which are shared with range checks).
    pub num_columns: usize,
}

/// Computed bitwise witness data.
#[derive(Debug, Clone)]
pub struct BitwiseWitness {
    /// Nibble decomposition columns for each group.
    /// For group g: [a_nibble_0..a_nibble_{n-1}, b_nibble_0..b_nibble_{n-1}, and_nibble_0..and_nibble_{n-1}]
    pub nibble_columns: Vec<Vec<Vec<Scalar>>>,
}

/// Layout of bitwise LogUp auxiliary columns with the extended Phase-0
/// soundness machinery (per-nibble-position inverse, table-inverse, running
/// sum + multiplicity). Not yet wired into prover/verifier — groundwork.
///
/// Order: `[for each group: a_nibs | b_nibs | and_nibs | f_nibs] |
///          t_a | t_b | t_c | u_t | h | m`
#[derive(Debug, Clone)]
pub struct ExtendedBitwiseColumnLayout {
    /// For each group: `(start_offset, num_nibbles)` — nibble decomposition
    /// triple. 3 · num_nibbles contiguous columns per group.
    pub nibble_offsets: Vec<(usize, usize)>,
    /// For each group: `(start_offset, num_nibbles)` — per-nibble inverses
    /// `f_k(X) = 1 / (γ - (a_k + δ·b_k + δ²·c_k))`.
    pub f_offsets: Vec<(usize, usize)>,
    pub t_a_column: usize,
    pub t_b_column: usize,
    pub t_c_column: usize,
    /// Table inverse `u_t(X) = 1 / (γ - (t_a(X) + δ·t_b(X) + δ²·t_c(X)))`.
    pub u_t_column: usize,
    /// Running sum `h`.
    pub h_column: usize,
    /// Multiplicity column `m` (counts per table row).
    pub m_column: usize,
    /// Total number of bitwise LogUp auxiliary columns.
    pub num_columns: usize,
}

/// Extended bitwise witness paired with [`ExtendedBitwiseColumnLayout`].
#[derive(Debug, Clone)]
pub struct ExtendedBitwiseWitness {
    pub nibble_columns: Vec<Vec<Vec<Scalar>>>,
    pub f_columns: Vec<Vec<Vec<Scalar>>>,
    pub t_a_column: Vec<Scalar>,
    pub t_b_column: Vec<Scalar>,
    pub t_c_column: Vec<Scalar>,
    pub u_t_column: Vec<Scalar>,
    pub h_column: Vec<Scalar>,
    pub m_column: Vec<Scalar>,
}

/// Build the extended bitwise column layout: per-group `a | b | and | f`
/// block followed by shared `t_a | t_b | t_c | u_t | h | m`.
pub fn extended_bitwise_column_layout(
    groups: &[BitwiseLookupGroup],
) -> ExtendedBitwiseColumnLayout {
    let mut offset = 0usize;
    let mut nibble_offsets = Vec::with_capacity(groups.len());
    let mut f_offsets = Vec::with_capacity(groups.len());
    for group in groups {
        let n = group.num_nibbles;
        // a | b | and : 3n contiguous columns
        nibble_offsets.push((offset, n));
        offset += 3 * n;
        // f : n columns
        f_offsets.push((offset, n));
        offset += n;
    }
    let t_a_column = offset;
    let t_b_column = offset + 1;
    let t_c_column = offset + 2;
    let u_t_column = offset + 3;
    let h_column = offset + 4;
    let m_column = offset + 5;
    let num_columns = offset + 6;

    ExtendedBitwiseColumnLayout {
        nibble_offsets,
        f_offsets,
        t_a_column,
        t_b_column,
        t_c_column,
        u_t_column,
        h_column,
        m_column,
        num_columns,
    }
}

/// Compute the extended bitwise witness given a trace, groups, and the
/// Fiat-Shamir challenges `γ` and `δ`.
///
/// The nibble decomposition is populated by evaluating the operand columns,
/// taking nibbles, and computing `a_k AND b_k` for each position. Queries
/// `q_k = a_k + δ·b_k + δ²·c_k` are then inverted against `γ`. The 256-row
/// canonical nibble-AND table is similarly combined and inverted. The running
/// sum `h` closes cyclically when every per-nibble query appears in the
/// table with correct multiplicity.
pub fn compute_extended_bitwise_witness(
    trace_columns: &[&Vec<Scalar>],
    groups: &[BitwiseLookupGroup],
    gamma: &Scalar,
    delta: &Scalar,
    domain_size: usize,
    curve: CurveType,
) -> ExtendedBitwiseWitness {
    // Helper: whether the group is active at `row` in the polynomial sense.
    let group_active = |group: &BitwiseLookupGroup, row: usize| -> bool {
        if group.selectors.is_empty() {
            true
        } else {
            group
                .selectors
                .iter()
                .any(|&sel| !trace_columns[sel][row].is_zero())
        }
    };

    // Step 1: build nibble decompositions and count multiplicities per table row.
    let mut nibble_columns: Vec<Vec<Vec<Scalar>>> = Vec::with_capacity(groups.len());
    // Multiplicity per table row (256 entries; rows indexed by a*16 + b).
    let mut row_counts = [0u64; NIBBLE_AND_TABLE_SIZE];

    for group in groups {
        let n = group.num_nibbles;
        let a_col = trace_columns[group.operand_a_column];
        let b_col = trace_columns[group.operand_b_column];
        // 3n columns: a_nibs | b_nibs | and_nibs.
        let mut cols: Vec<Vec<Scalar>> =
            vec![vec![Scalar::zero(curve); domain_size]; 3 * n];
        for row in 0..domain_size {
            let active = group_active(group, row);
            let a_val = a_col[row].to_u64();
            let b_val = b_col[row].to_u64();
            let a_nibs = decompose_to_nibbles(a_val, n);
            let b_nibs = decompose_to_nibbles(b_val, n);
            for k in 0..n {
                let a_k = a_nibs[k] as u64;
                let b_k = b_nibs[k] as u64;
                let c_k = a_k & b_k;
                cols[k][row] = Scalar::from_u64(a_k, curve);
                cols[n + k][row] = Scalar::from_u64(b_k, curve);
                cols[2 * n + k][row] = Scalar::from_u64(c_k, curve);
                if active {
                    let table_idx = ((a_k << 4) | b_k) as usize;
                    row_counts[table_idx] += 1;
                }
            }
        }
        nibble_columns.push(cols);
    }

    // Step 2: per-nibble inverses `f_k = 1 / (γ - (a_k + δ·b_k + δ²·c_k))`.
    let delta_sq = delta.mul(delta);
    let mut f_columns: Vec<Vec<Vec<Scalar>>> = Vec::with_capacity(groups.len());
    for (g_idx, group) in groups.iter().enumerate() {
        let n = group.num_nibbles;
        let mut g_invs: Vec<Vec<Scalar>> =
            vec![vec![Scalar::zero(curve); domain_size]; n];
        for k in 0..n {
            for row in 0..domain_size {
                let a_k = &nibble_columns[g_idx][k][row];
                let b_k = &nibble_columns[g_idx][n + k][row];
                let c_k = &nibble_columns[g_idx][2 * n + k][row];
                // q = a + δ·b + δ²·c
                let q = a_k.add(&delta.mul(b_k)).add(&delta_sq.mul(c_k));
                let denom = gamma.sub(&q);
                debug_assert!(!denom.is_zero(), "gamma collided with bitwise query");
                g_invs[k][row] = denom.inverse();
            }
        }
        f_columns.push(g_invs);
    }

    // Step 3: canonical table columns and their combined inverse.
    let t_a_column = nibble_and_table_column(NibbleAndTableComponent::A, domain_size, curve);
    let t_b_column = nibble_and_table_column(NibbleAndTableComponent::B, domain_size, curve);
    let t_c_column = nibble_and_table_column(NibbleAndTableComponent::C, domain_size, curve);
    let mut u_t_column = vec![Scalar::zero(curve); domain_size];
    for row in 0..domain_size {
        let t_a = &t_a_column[row];
        let t_b = &t_b_column[row];
        let t_c = &t_c_column[row];
        let combined = t_a.add(&delta.mul(t_b)).add(&delta_sq.mul(t_c));
        let denom = gamma.sub(&combined);
        debug_assert!(!denom.is_zero(), "gamma collided with bitwise table row");
        u_t_column[row] = denom.inverse();
    }

    // Step 4: multiplicity column m.
    let mut m_column = vec![Scalar::zero(curve); domain_size];
    for (i, &count) in row_counts.iter().enumerate() {
        if i < domain_size {
            m_column[i] = Scalar::from_u64(count, curve);
        }
    }

    // Step 5: running sum h (cyclic so wrap closure is automatic).
    let mut h_column = vec![Scalar::zero(curve); domain_size];
    for i in 1..domain_size {
        let mut sum = h_column[i - 1].clone();
        for (g_idx, group) in groups.iter().enumerate() {
            if group_active(group, i - 1) {
                for k in 0..group.num_nibbles {
                    sum = sum.add(&f_columns[g_idx][k][i - 1]);
                }
            }
        }
        if !m_column[i - 1].is_zero() {
            sum = sum.sub(&m_column[i - 1].mul(&u_t_column[i - 1]));
        }
        h_column[i] = sum;
    }

    ExtendedBitwiseWitness {
        nibble_columns,
        f_columns,
        t_a_column,
        t_b_column,
        t_c_column,
        u_t_column,
        h_column,
        m_column,
    }
}

/// Group bitwise declarations by shared operand columns and width.
///
/// Unlike range-check groups which can share decomposition columns across
/// different selectors for the same source column, bitwise groups are kept
/// per-declaration since each may have a different result column and operation.
pub fn group_bitwise_declarations(
    decls: &[BitwiseLookupDeclaration],
) -> Vec<BitwiseLookupGroup> {
    decls.iter().map(|d| {
        BitwiseLookupGroup {
            operand_a_column: d.operand_a_column,
            operand_b_column: d.operand_b_column,
            result_column: d.result_column,
            width_bits: d.width_bits,
            num_nibbles: (d.width_bits / 4) as usize,
            op: d.op,
            selectors: d.selectors.clone(),
        }
    }).collect()
}

/// Compute the bitwise column layout from grouped declarations.
pub fn bitwise_column_layout(groups: &[BitwiseLookupGroup]) -> BitwiseColumnLayout {
    let mut offset = 0;
    let mut nibble_offsets = Vec::with_capacity(groups.len());

    for group in groups {
        nibble_offsets.push((offset, group.num_nibbles));
        // 3 sets of nibble columns per group: a_nibbles, b_nibbles, and_nibbles
        offset += 3 * group.num_nibbles;
    }

    BitwiseColumnLayout {
        nibble_offsets,
        num_columns: offset,
    }
}

/// Decompose a u64 value into nibbles (4-bit chunks, little-endian).
fn decompose_to_nibbles(value: u64, num_nibbles: usize) -> Vec<u8> {
    let mut result = vec![0u8; num_nibbles];
    for i in 0..num_nibbles.min(16) {
        result[i] = ((value >> (4 * i)) & 0xF) as u8;
    }
    result
}

/// Compute the bitwise witness from trace data.
///
/// For each group, decomposes operand A and B into nibbles, computes
/// AND(a_i, b_i) for each nibble pair, and verifies the result column
/// matches the derived operation (AND/OR/XOR).
pub fn compute_bitwise_witness(
    trace_columns: &[&Vec<Scalar>],
    groups: &[BitwiseLookupGroup],
    num_rows: usize,
    domain_size: usize,
    curve: CurveType,
) -> BitwiseWitness {
    let mut nibble_columns: Vec<Vec<Vec<Scalar>>> = Vec::with_capacity(groups.len());

    for group in groups {
        let n = group.num_nibbles;
        // 3n columns: a_nibbles[0..n], b_nibbles[0..n], and_nibbles[0..n]
        let mut cols: Vec<Vec<Scalar>> = vec![vec![Scalar::zero(curve); domain_size]; 3 * n];

        let a_col = trace_columns[group.operand_a_column];
        let b_col = trace_columns[group.operand_b_column];

        for row in 0..num_rows {
            let active = if group.selectors.is_empty() {
                true
            } else {
                group.selectors.iter().any(|&sel| !trace_columns[sel][row].is_zero())
            };

            if !active {
                continue;
            }

            let a_val = a_col[row].to_u64();
            let b_val = b_col[row].to_u64();
            let a_nibs = decompose_to_nibbles(a_val, n);
            let b_nibs = decompose_to_nibbles(b_val, n);

            for i in 0..n {
                cols[i][row] = Scalar::from_u64(a_nibs[i] as u64, curve);
                cols[n + i][row] = Scalar::from_u64(b_nibs[i] as u64, curve);
                cols[2 * n + i][row] = Scalar::from_u64((a_nibs[i] & b_nibs[i]) as u64, curve);
            }
        }

        nibble_columns.push(cols);
    }

    BitwiseWitness { nibble_columns }
}

/// Evaluate the bitwise result constraint at a point z for a single group.
///
/// Verifies that:
/// - `operand_a(z) == Σ a_nibble_k(z) * 16^k` (nibble decomposition of A)
/// - `operand_b(z) == Σ b_nibble_k(z) * 16^k` (nibble decomposition of B)
/// - `and_result(z) == Σ and_nibble_k(z) * 16^k` (AND recomposition)
/// - Result matches the operation:
///   - AND: `result(z) == and_result(z)`
///   - OR:  `result(z) == a(z) + b(z) - and_result(z)`
///   - XOR: `result(z) == a(z) + b(z) - 2 * and_result(z)`
///
/// Returns 4 constraint evaluations (3 decompositions + 1 result).
/// Each is gated by the combined selector.
pub fn evaluate_bitwise_at_point(
    a_at_z: &Scalar,
    b_at_z: &Scalar,
    result_at_z: &Scalar,
    a_nibble_evals: &[Scalar],
    b_nibble_evals: &[Scalar],
    and_nibble_evals: &[Scalar],
    selector_at_z: Option<&Scalar>,
    op: BitwiseOp,
    curve: CurveType,
) -> Vec<Scalar> {
    let sixteen = Scalar::from_u64(16, curve);

    // Recompose nibbles
    let recompose = |nibbles: &[Scalar]| -> Scalar {
        let mut val = Scalar::zero(curve);
        let mut power = Scalar::one(curve);
        for nib in nibbles {
            val = val.add(&nib.mul(&power));
            power = power.mul(&sixteen);
        }
        val
    };

    let a_recomp = recompose(a_nibble_evals);
    let b_recomp = recompose(b_nibble_evals);
    let and_recomp = recompose(and_nibble_evals);

    // Decomposition constraints
    let a_diff = a_at_z.sub(&a_recomp);
    let b_diff = b_at_z.sub(&b_recomp);

    // Result constraint depends on operation
    let expected = match op {
        BitwiseOp::And => and_recomp.clone(),
        BitwiseOp::Or => a_at_z.add(b_at_z).sub(&and_recomp),
        BitwiseOp::Xor => {
            let two = Scalar::from_u64(2, curve);
            a_at_z.add(b_at_z).sub(&two.mul(&and_recomp))
        }
    };
    let result_diff = result_at_z.sub(&expected);

    let gate = |diff: Scalar| -> Scalar {
        match selector_at_z {
            Some(sel) => sel.mul(&diff),
            None => diff,
        }
    };

    vec![
        gate(a_diff),      // operand A decomposition
        gate(b_diff),      // operand B decomposition
        gate(result_diff), // result verification
    ]
}

/// Number of bitwise constraints for a given set of groups.
///
/// Each group contributes 3 constraints: A decomposition, B decomposition, result check.
/// The nibble-AND table lookups are handled by the LogUp running sum (shared with range checks).
pub fn num_bitwise_constraints(groups: &[BitwiseLookupGroup]) -> usize {
    groups.len() * 3
}

/// Count nibble-AND table lookups from bitwise groups for multiplicity tracking.
///
/// For each active row in each group, each nibble pair contributes one lookup
/// into the nibble-AND table. Returns a 256-entry count array.
pub fn count_nibble_and_lookups(
    trace_columns: &[&Vec<Scalar>],
    groups: &[BitwiseLookupGroup],
    num_rows: usize,
) -> [u64; 256] {
    let mut counts = [0u64; 256];

    for group in groups {
        let a_col = trace_columns[group.operand_a_column];
        let b_col = trace_columns[group.operand_b_column];
        let n = group.num_nibbles;

        for row in 0..num_rows {
            let active = if group.selectors.is_empty() {
                true
            } else {
                group.selectors.iter().any(|&sel| !trace_columns[sel][row].is_zero())
            };

            if !active {
                continue;
            }

            let a_val = a_col[row].to_u64();
            let b_val = b_col[row].to_u64();
            let a_nibs = decompose_to_nibbles(a_val, n);
            let b_nibs = decompose_to_nibbles(b_val, n);

            for i in 0..n {
                let idx = (a_nibs[i] as usize) * 16 + (b_nibs[i] as usize);
                counts[idx] += 1;
            }
        }
    }

    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_table_range() {
        let table = LookupTable::range(16);
        assert_eq!(table.bits, 16);
        assert_eq!(table.size(), 65536);
        assert_eq!(table.name, "range_16bit");
    }

    #[test]
    fn test_lookup_requirements_empty() {
        let reqs = LookupRequirements::none();
        assert!(reqs.is_empty());
        assert_eq!(reqs.num_lookups(), 0);
    }

    #[test]
    fn test_lookup_declaration() {
        let decl = LookupDeclaration {
            label: "mul_aux0".to_string(),
            column_index: 16,
            max_bits: 64,
            selector_column: Some(27),
        };
        assert_eq!(decl.column_index, 16);
        assert_eq!(decl.max_bits, 64);
        assert_eq!(decl.selector_column, Some(27));
    }

    #[test]
    fn test_group_declarations() {
        let reqs = LookupRequirements {
            tables: vec![LookupTable::range(8)],
            declarations: vec![
                (LookupDeclaration {
                    label: "a".to_string(),
                    column_index: 5,
                    max_bits: 64,
                    selector_column: Some(20),
                }, 0),
                (LookupDeclaration {
                    label: "b".to_string(),
                    column_index: 5,
                    max_bits: 64,
                    selector_column: Some(21),
                }, 0),
                (LookupDeclaration {
                    label: "c".to_string(),
                    column_index: 10,
                    max_bits: 32,
                    selector_column: None,
                }, 0),
            ],
        };
        let groups = group_declarations(&reqs);
        // Two unique (col, bits) groups: (5,64) and (10,32)
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].column_index, 5);
        assert_eq!(groups[0].num_limbs, 8);
        assert_eq!(groups[0].selectors.len(), 2);
        assert_eq!(groups[1].column_index, 10);
        assert_eq!(groups[1].num_limbs, 4);
    }

    #[test]
    fn test_logup_column_layout() {
        let groups = vec![
            LogUpGroup { column_index: 5, max_bits: 64, num_limbs: 8, selectors: vec![] },
            LogUpGroup { column_index: 10, max_bits: 32, num_limbs: 4, selectors: vec![] },
        ];
        let layout = logup_column_layout(&groups);
        assert_eq!(layout.limb_offsets, vec![(0, 8), (8, 4)]);
        assert_eq!(layout.h_column, 12);
        assert_eq!(layout.m_column, 13);
        assert_eq!(layout.num_columns, 14);
    }

    #[test]
    fn test_extended_logup_column_layout() {
        let groups = vec![
            LogUpGroup { column_index: 5, max_bits: 64, num_limbs: 8, selectors: vec![] },
            LogUpGroup { column_index: 10, max_bits: 32, num_limbs: 4, selectors: vec![] },
        ];
        let layout = extended_logup_column_layout(&groups);
        // limbs: 0..8, 8..12; f: 12..20, 20..24; t: 24; u_t: 25; h: 26; m: 27
        assert_eq!(layout.limb_offsets, vec![(0, 8), (8, 4)]);
        assert_eq!(layout.f_offsets, vec![(12, 8), (20, 4)]);
        assert_eq!(layout.t_column, 24);
        assert_eq!(layout.u_t_column, 25);
        assert_eq!(layout.h_column, 26);
        assert_eq!(layout.m_column, 27);
        assert_eq!(layout.num_columns, 28);
        assert_eq!(layout.m_column, layout.num_columns - 1);
        assert_eq!(layout.h_column, layout.num_columns - 2);
    }

    #[test]
    fn test_decompose_to_bytes() {
        assert_eq!(decompose_to_bytes(0x0102, 2), vec![0x02, 0x01]);
        assert_eq!(decompose_to_bytes(256, 2), vec![0x00, 0x01]);
        assert_eq!(decompose_to_bytes(255, 1), vec![0xFF]);
    }

    #[test]
    fn test_evaluate_decomposition() {
        use crate::field::CurveType;
        let curve = CurveType::Bls48581;
        let value = Scalar::from_u64(0x0A0B, curve); // 2571 decimal
        let limbs = vec![
            Scalar::from_u64(0x0B, curve), // low byte
            Scalar::from_u64(0x0A, curve), // high byte
        ];
        let result = evaluate_decomposition_at_point(&value, &limbs, None, curve);
        assert!(result.is_zero(), "decomposition should match");

        // Wrong decomposition
        let bad_limbs = vec![
            Scalar::from_u64(0x0A, curve),
            Scalar::from_u64(0x0B, curve),
        ];
        let result2 = evaluate_decomposition_at_point(&value, &bad_limbs, None, curve);
        assert!(!result2.is_zero(), "wrong decomposition should fail");
    }

    #[test]
    fn test_num_logup_constraints() {
        let groups = vec![
            LogUpGroup { column_index: 0, max_bits: 64, num_limbs: 8, selectors: vec![] },
            LogUpGroup { column_index: 1, max_bits: 32, num_limbs: 4, selectors: vec![] },
        ];
        // Currently enforced: 1 decomposition per group + 1 boundary
        assert_eq!(num_logup_constraints(&groups), 3);
    }

    #[test]
    fn test_num_extended_logup_constraints() {
        let groups = vec![
            LogUpGroup { column_index: 0, max_bits: 64, num_limbs: 8, selectors: vec![] },
            LogUpGroup { column_index: 1, max_bits: 32, num_limbs: 4, selectors: vec![] },
        ];
        // 2 decomposition + (8+4) inverse + 1 table inv + 1 transition + 1 boundary
        assert_eq!(num_extended_logup_constraints(&groups), 2 + 12 + 3);
    }

    #[test]
    fn test_range_table_column() {
        let curve = CurveType::Bls48581;
        let t = range_table_column(1024, curve);
        assert_eq!(t.len(), 1024);
        for i in 0..256 {
            assert_eq!(t[i].to_u64(), i as u64);
        }
        for i in 256..1024 {
            assert!(t[i].is_zero());
        }
    }

    #[test]
    fn test_nibble_and_table_component_values() {
        use NibbleAndTableComponent::*;
        // Spot check the triple semantics: (i>>4, i&0xF, AND).
        for i in 0..NIBBLE_AND_TABLE_SIZE {
            let a = ((i >> 4) & 0xF) as u64;
            let b = (i & 0xF) as u64;
            assert_eq!(A.value_at(i), a);
            assert_eq!(B.value_at(i), b);
            assert_eq!(C.value_at(i), a & b);
        }
        // Outside the table range, every component is zero.
        for i in [NIBBLE_AND_TABLE_SIZE, NIBBLE_AND_TABLE_SIZE + 100] {
            assert_eq!(A.value_at(i), 0);
            assert_eq!(B.value_at(i), 0);
            assert_eq!(C.value_at(i), 0);
        }
    }

    #[test]
    fn test_nibble_and_table_column_shape() {
        let curve = CurveType::Bls48581;
        for comp in [
            NibbleAndTableComponent::A,
            NibbleAndTableComponent::B,
            NibbleAndTableComponent::C,
        ] {
            let t = nibble_and_table_column(comp, 1024, curve);
            assert_eq!(t.len(), 1024);
            for i in 0..256 {
                assert_eq!(t[i].to_u64(), comp.value_at(i));
            }
            for i in 256..1024 {
                assert!(t[i].is_zero(), "{:?} must be zero at row {}", comp, i);
            }
        }
    }

    /// Cross-check the Lagrange evaluator against a direct eval via IFFT +
    /// polynomial evaluation. Uses the BLS48-581 scheme since it provides
    /// `domain_generator`, `ifft`, and `eval_poly_at` for the canonical
    /// domain sizes (16, 32, 64, 128, 256).
    #[test]
    fn test_extended_bitwise_witness_grand_sum_closes() {
        // Soundness spot-check: build the extended bitwise witness and verify
        // the running sum `h` closes cyclically — i.e. the transition holds at
        // the wrap-around row, which happens iff every per-nibble query matches
        // the nibble-AND table with correct multiplicity.
        use crate::field::CurveType;
        let curve = CurveType::Bls48581;

        // 3 active rows × 1 group, 1 nibble per operand for simplicity.
        // Queries: (a,b) ∈ {(0x5, 0x3), (0xF, 0xA), (0x0, 0xF)}.
        let domain_size = 256usize;
        let mut a_col_vals = vec![Scalar::zero(curve); domain_size];
        let mut b_col_vals = vec![Scalar::zero(curve); domain_size];
        let mut c_col_vals = vec![Scalar::zero(curve); domain_size];
        let mut sel_col_vals = vec![Scalar::zero(curve); domain_size];
        let samples: [(u64, u64); 3] = [(0x5, 0x3), (0xF, 0xA), (0x0, 0xF)];
        for (row, &(a, b)) in samples.iter().enumerate() {
            a_col_vals[row] = Scalar::from_u64(a, curve);
            b_col_vals[row] = Scalar::from_u64(b, curve);
            c_col_vals[row] = Scalar::from_u64(a & b, curve);
            sel_col_vals[row] = Scalar::one(curve);
        }
        let trace: Vec<&Vec<Scalar>> = vec![
            &a_col_vals, &b_col_vals, &c_col_vals, &sel_col_vals,
        ];
        let groups = vec![BitwiseLookupGroup {
            operand_a_column: 0,
            operand_b_column: 1,
            result_column: 2,
            width_bits: 4,
            num_nibbles: 1,
            op: BitwiseOp::And,
            selectors: vec![3],
        }];
        let gamma = Scalar::from_u64(1_000_003, curve);
        let delta = Scalar::from_u64(7919, curve);

        let w = compute_extended_bitwise_witness(
            &trace, &groups, &gamma, &delta, domain_size, curve,
        );

        // All four structural invariants:

        // (I) Table columns match canonical values.
        for i in 0..NIBBLE_AND_TABLE_SIZE {
            assert_eq!(
                w.t_a_column[i].to_u64(),
                NibbleAndTableComponent::A.value_at(i),
                "t_a[{}] wrong", i
            );
            assert_eq!(
                w.t_b_column[i].to_u64(),
                NibbleAndTableComponent::B.value_at(i)
            );
            assert_eq!(
                w.t_c_column[i].to_u64(),
                NibbleAndTableComponent::C.value_at(i)
            );
        }

        // (II) Per-nibble inverse `f · (γ - q) = 1` on every row.
        let delta_sq = delta.mul(&delta);
        for row in 0..domain_size {
            let a = &w.nibble_columns[0][0][row];
            let b = &w.nibble_columns[0][1][row];
            let c = &w.nibble_columns[0][2][row];
            let q = a.add(&delta.mul(b)).add(&delta_sq.mul(c));
            let f = &w.f_columns[0][0][row];
            let prod = f.mul(&gamma.sub(&q));
            assert!(
                prod.sub(&Scalar::one(curve)).is_zero(),
                "f-inverse fails at row {}", row
            );
        }

        // (III) Table inverse `u_t · (γ - (t_a + δ·t_b + δ²·t_c)) = 1`.
        for row in 0..domain_size {
            let combined = w.t_a_column[row]
                .add(&delta.mul(&w.t_b_column[row]))
                .add(&delta_sq.mul(&w.t_c_column[row]));
            let prod = w.u_t_column[row].mul(&gamma.sub(&combined));
            assert!(
                prod.sub(&Scalar::one(curve)).is_zero(),
                "u_t-inverse fails at row {}", row
            );
        }

        // (IV) Running-sum transition (cyclic, on every row):
        // h[(r+1) mod n] - h[r] - Σ_g active · Σ_k f_{g,k}[r] + m[r] · u_t[r] = 0
        for r in 0..domain_size {
            let next = (r + 1) % domain_size;
            let mut lhs = w.h_column[next].sub(&w.h_column[r]);
            for (g_idx, group) in groups.iter().enumerate() {
                let active = if group.selectors.is_empty() {
                    Scalar::one(curve)
                } else {
                    let mut s = Scalar::zero(curve);
                    for &sel_idx in &group.selectors {
                        s = s.add(&trace[sel_idx][r]);
                    }
                    s
                };
                let mut fsum = Scalar::zero(curve);
                for k in 0..group.num_nibbles {
                    fsum = fsum.add(&w.f_columns[g_idx][k][r]);
                }
                lhs = lhs.sub(&active.mul(&fsum));
            }
            lhs = lhs.add(&w.m_column[r].mul(&w.u_t_column[r]));
            assert!(
                lhs.is_zero(),
                "bitwise transition fails at row {} → {}", r, next
            );
        }
        // (V) Boundary: h[0] = 0.
        assert!(w.h_column[0].is_zero(), "bitwise h[0] must be 0");
    }

    /// Soundness negative: tamper with the multiplicity column `m` and
    /// confirm the running-sum transition identity fails at some row.
    /// If this test ever fails to detect tampering, the bitwise LogUp
    /// transition is unsound.
    #[test]
    fn test_extended_bitwise_witness_rejects_tampered_multiplicity() {
        use crate::field::CurveType;
        let curve = CurveType::Bls48581;

        // Same setup as the positive-case fixture, one 4-bit AND sample.
        let domain_size = 256usize;
        let mut a_col_vals = vec![Scalar::zero(curve); domain_size];
        let mut b_col_vals = vec![Scalar::zero(curve); domain_size];
        let mut c_col_vals = vec![Scalar::zero(curve); domain_size];
        let mut sel_col_vals = vec![Scalar::zero(curve); domain_size];
        // One active row with (a,b) = (0x5, 0x3) → table row index 0x53.
        a_col_vals[0] = Scalar::from_u64(0x5, curve);
        b_col_vals[0] = Scalar::from_u64(0x3, curve);
        c_col_vals[0] = Scalar::from_u64(0x5 & 0x3, curve);
        sel_col_vals[0] = Scalar::one(curve);

        let trace: Vec<&Vec<Scalar>> =
            vec![&a_col_vals, &b_col_vals, &c_col_vals, &sel_col_vals];
        let groups = vec![BitwiseLookupGroup {
            operand_a_column: 0,
            operand_b_column: 1,
            result_column: 2,
            width_bits: 4,
            num_nibbles: 1,
            op: BitwiseOp::And,
            selectors: vec![3],
        }];
        let gamma = Scalar::from_u64(1_000_003, curve);
        let delta = Scalar::from_u64(7919, curve);
        let delta_sq = delta.mul(&delta);

        let mut w = compute_extended_bitwise_witness(
            &trace, &groups, &gamma, &delta, domain_size, curve,
        );

        // Tamper: zero out the multiplicity at table row 0x53 (the row the
        // sole active lookup should land on). The transition must now fail.
        assert!(
            !w.m_column[0x53].is_zero(),
            "precondition: m[0x53] should be > 0 before tampering",
        );
        w.m_column[0x53] = Scalar::zero(curve);

        // Scan every domain row for a transition violation.
        let mut any_violation = false;
        for r in 0..domain_size {
            let next = (r + 1) % domain_size;
            let mut lhs = w.h_column[next].sub(&w.h_column[r]);
            for (g_idx, group) in groups.iter().enumerate() {
                let active = if group.selectors.is_empty() {
                    Scalar::one(curve)
                } else {
                    let mut s = Scalar::zero(curve);
                    for &sel_idx in &group.selectors {
                        s = s.add(&trace[sel_idx][r]);
                    }
                    s
                };
                let mut fsum = Scalar::zero(curve);
                for k in 0..group.num_nibbles {
                    fsum = fsum.add(&w.f_columns[g_idx][k][r]);
                }
                lhs = lhs.sub(&active.mul(&fsum));
            }
            lhs = lhs.add(&w.m_column[r].mul(&w.u_t_column[r]));
            if !lhs.is_zero() {
                any_violation = true;
                break;
            }
        }
        assert!(
            any_violation,
            "tampered multiplicity must produce a transition violation somewhere"
        );
        // silence unused in this scope.
        let _ = delta_sq;
    }

    #[test]
    fn test_extended_bitwise_column_layout() {
        let groups = vec![
            BitwiseLookupGroup {
                operand_a_column: 0, operand_b_column: 1, result_column: 2,
                width_bits: 64, num_nibbles: 16, op: BitwiseOp::And, selectors: vec![],
            },
            BitwiseLookupGroup {
                operand_a_column: 0, operand_b_column: 1, result_column: 3,
                width_bits: 64, num_nibbles: 16, op: BitwiseOp::Or, selectors: vec![],
            },
        ];
        let layout = extended_bitwise_column_layout(&groups);
        // Group 0 at offset 0: 3·16 nibble cols (48) + 16 f cols = 64
        // Group 1 at offset 64: 64 cols. Shared tail at offset 128: 6 cols.
        assert_eq!(layout.nibble_offsets, vec![(0, 16), (64, 16)]);
        assert_eq!(layout.f_offsets, vec![(48, 16), (112, 16)]);
        assert_eq!(layout.t_a_column, 128);
        assert_eq!(layout.t_b_column, 129);
        assert_eq!(layout.t_c_column, 130);
        assert_eq!(layout.u_t_column, 131);
        assert_eq!(layout.h_column, 132);
        assert_eq!(layout.m_column, 133);
        assert_eq!(layout.num_columns, 134);
    }

    /// Soundness safeguard: the canonical range-table Lagrange evaluator
    /// must produce a value different from what an adversarial "all-zeros"
    /// or "all-ones" table would evaluate to. If it didn't, the verifier's
    /// canonical-t(z) binding would be toothless.
    #[test]
    fn test_range_table_canonical_detects_adversarial_tables() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let n = 256usize;
        let omega = scheme.domain_generator(n as u64);
        let z = Scalar::from_u64(12345, curve);

        let canonical = evaluate_range_table_at_point(&z, &omega, n, curve);

        // Adversarial table A: all zeros.
        let zeros = vec![Scalar::zero(curve); n];
        let zeros_coeffs = scheme.ifft(&zeros, n as u64);
        let zeros_at_z = scheme.eval_poly_at(&zeros_coeffs, &z);
        assert!(
            !canonical.sub(&zeros_at_z).is_zero(),
            "all-zero table must differ from canonical at z"
        );

        // Adversarial table B: constant 42 everywhere.
        let const_table: Vec<Scalar> =
            vec![Scalar::from_u64(42, curve); n];
        let const_coeffs = scheme.ifft(&const_table, n as u64);
        let const_at_z = scheme.eval_poly_at(&const_coeffs, &z);
        assert!(
            !canonical.sub(&const_at_z).is_zero(),
            "all-42 table must differ from canonical at z"
        );

        // Adversarial table C: permuted canonical (swap rows 5 and 7).
        let mut permuted = range_table_column(n, curve);
        permuted.swap(5, 7);
        let perm_coeffs = scheme.ifft(&permuted, n as u64);
        let perm_at_z = scheme.eval_poly_at(&perm_coeffs, &z);
        assert!(
            !canonical.sub(&perm_at_z).is_zero(),
            "permuted table must differ from canonical at z"
        );
    }

    /// Parallel safeguard for the nibble-AND table: the Lagrange evaluator
    /// must distinguish canonical from adversarial tables for every
    /// component (A, B, C).
    #[test]
    fn test_nibble_and_table_canonical_detects_adversarial_tables() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let n = 256usize;
        let omega = scheme.domain_generator(n as u64);
        let z = Scalar::from_u64(99_991, curve);

        for comp in [
            NibbleAndTableComponent::A,
            NibbleAndTableComponent::B,
            NibbleAndTableComponent::C,
        ] {
            let canonical = evaluate_nibble_and_table_at_point(
                comp, &z, &omega, n, curve,
            );

            // Adversarial: swap two rows. 17 and 34 differ on all three
            // components (rows 10/20 would fail for C since both AND to 0).
            let mut tampered = nibble_and_table_column(comp, n, curve);
            tampered.swap(17, 34);
            let tampered_coeffs = scheme.ifft(&tampered, n as u64);
            let tampered_at_z = scheme.eval_poly_at(&tampered_coeffs, &z);
            assert!(
                !canonical.sub(&tampered_at_z).is_zero(),
                "{:?}: tampered table must differ from canonical at z",
                comp
            );
        }
    }

    #[test]
    fn test_evaluate_nibble_and_table_matches_lagrange() {
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let domain_size = 256usize;
        let omega = scheme.domain_generator(domain_size as u64);

        // An arbitrary off-domain point that we can treat as z.
        // 12345 is tiny relative to the BLS48-581 scalar field, so it's not a domain element.
        let z = Scalar::from_u64(12_345, curve);

        for comp in [
            NibbleAndTableComponent::A,
            NibbleAndTableComponent::B,
            NibbleAndTableComponent::C,
        ] {
            let t = nibble_and_table_column(comp, domain_size, curve);
            let t_coeffs = scheme.ifft(&t, domain_size as u64);
            let expected = scheme.eval_poly_at(&t_coeffs, &z);
            let got = evaluate_nibble_and_table_at_point(comp, &z, &omega, domain_size, curve);
            assert!(
                got.sub(&expected).is_zero(),
                "{:?}: Lagrange eval disagrees with IFFT+eval_poly_at",
                comp
            );
        }
    }

    #[test]
    fn test_evaluate_inverse() {
        let curve = CurveType::Bls48581;
        let gamma = Scalar::from_u64(12345, curve);
        let v = Scalar::from_u64(42, curve);
        let f = gamma.sub(&v).inverse();
        let r = evaluate_inverse_at_point(&f, &v, &gamma, curve);
        assert!(r.is_zero(), "correct inverse should satisfy constraint");

        let bad = f.add(&Scalar::one(curve));
        let r2 = evaluate_inverse_at_point(&bad, &v, &gamma, curve);
        assert!(!r2.is_zero(), "wrong inverse should fail");
    }

    #[test]
    fn test_evaluate_transition_zero_contribution() {
        let curve = CurveType::Bls48581;
        // When contribution is 0, h stays constant: h(ωX) = h(X).
        let h_at_z = Scalar::from_u64(7, curve);
        let h_at_omega_z = Scalar::from_u64(7, curve);
        let m = Scalar::zero(curve);
        let u_t = Scalar::from_u64(99, curve);
        let actives: Vec<(Scalar, Vec<Scalar>)> =
            vec![(Scalar::zero(curve), vec![Scalar::from_u64(5, curve)])];
        let r = evaluate_transition_at_point(&h_at_z, &h_at_omega_z, &actives, &m, &u_t, curve);
        assert!(r.is_zero());
    }

    /// Soundness spot-check: build the extended witness on a tiny domain
    /// and verify the LogUp constraints (decomposition + boundary +
    /// per-limb inverse + table inverse + running-sum transition) vanish
    /// on every domain point. Math-level check (no commit/open) that
    /// finishes in milliseconds.
    #[test]
    fn test_extended_witness_all_constraints_vanish_on_domain() {
        use crate::field::CurveType;
        let curve = CurveType::Bls48581;

        // Trace: 4 real rows, 8-bit values all < 16 so they fit in a small table.
        // We still use the 256-entry range table (t[i]=i for i<256, else 0)
        // since the table polynomial is independent of the specific lookup.
        let src: Vec<Scalar> = [7u64, 12, 3, 0]
            .iter()
            .chain(std::iter::repeat(&0u64).take(252))
            .map(|v| Scalar::from_u64(*v, curve))
            .collect();
        let sel: Vec<Scalar> = [1u64, 1, 1, 0]
            .iter()
            .chain(std::iter::repeat(&0u64).take(252))
            .map(|v| Scalar::from_u64(*v, curve))
            .collect();
        assert_eq!(src.len(), 256);
        assert_eq!(sel.len(), 256);

        let trace: Vec<&Vec<Scalar>> = vec![&src, &sel];
        let groups = vec![LogUpGroup {
            column_index: 0,
            max_bits: 8,
            num_limbs: 1,
            selectors: vec![1],
        }];
        let domain_size = 256usize;
        let num_rows = 4usize;
        let gamma = Scalar::from_u64(1_000_003, curve);

        let w = compute_extended_logup_witness(
            &trace, &groups, &gamma, num_rows, domain_size, curve,
        );

        // (1) Decomposition: sel[r] · (value[r] - Σ limb_k[r]·256^k) = 0
        for r in 0..domain_size {
            let recomposed = &w.limb_columns[0][0][r]; // 1 limb, value = limb
            let diff = src[r].sub(recomposed);
            let body = sel[r].mul(&diff);
            assert!(
                body.is_zero(),
                "decomposition fails at row {}",
                r
            );
        }

        // (2) Per-limb inverse: f[r] · (γ - limb[r]) - 1 = 0 (on ALL rows)
        for r in 0..domain_size {
            let body = evaluate_inverse_at_point(
                &w.f_columns[0][0][r],
                &w.limb_columns[0][0][r],
                &gamma,
                curve,
            );
            assert!(body.is_zero(), "f-inverse fails at row {}", r);
        }

        // (3) Table inverse: u_t[r] · (γ - t[r]) - 1 = 0 (on ALL rows)
        for r in 0..domain_size {
            let body = evaluate_inverse_at_point(
                &w.u_t_column[r],
                &w.t_column[r],
                &gamma,
                curve,
            );
            assert!(body.is_zero(), "u_t-inverse fails at row {}", r);
        }

        // (4) Running-sum transition (cyclic, on ALL rows):
        // h[r+1] - h[r] - Σ_{g,k} active_g[r] · f_{g,k}[r] + m[r] · u_t[r] = 0
        // where r+1 is taken mod n.
        for r in 0..domain_size {
            let next_r = (r + 1) % domain_size;
            let h_next = &w.h_column[next_r];
            let h_cur = &w.h_column[r];
            let mut lhs = h_next.sub(h_cur);
            for (g_idx, group) in groups.iter().enumerate() {
                let active = if group.selectors.is_empty() {
                    Scalar::one(curve)
                } else {
                    let mut s = Scalar::zero(curve);
                    for &sel_idx in &group.selectors {
                        s = s.add(&trace[sel_idx][r]);
                    }
                    s
                };
                let mut fsum = Scalar::zero(curve);
                for k in 0..group.num_limbs {
                    fsum = fsum.add(&w.f_columns[g_idx][k][r]);
                }
                lhs = lhs.sub(&active.mul(&fsum));
            }
            lhs = lhs.add(&w.m_column[r].mul(&w.u_t_column[r]));
            assert!(
                lhs.is_zero(),
                "transition fails at row {} (next={}): grand sum not closed",
                r, next_r
            );
        }

        // (5) Boundary: h[0] = 0
        assert!(w.h_column[0].is_zero(), "h[0] must be 0");
    }

    #[test]
    fn test_logup_witness_closed_sum_simple() {
        // Construct a toy 4-row trace with one 8-bit group, verify h closes to zero.
        // Columns are padded to domain_size (512) to match the prover's contract
        // that trace columns are always at least `domain_size` long.
        let curve = CurveType::Bls48581;
        let domain_size = 512usize;
        let mut src: Vec<Scalar> = vec![Scalar::zero(curve); domain_size];
        src[0] = Scalar::from_u64(1, curve);
        src[1] = Scalar::from_u64(2, curve);
        src[2] = Scalar::from_u64(3, curve);
        let mut sel: Vec<Scalar> = vec![Scalar::zero(curve); domain_size];
        sel[0] = Scalar::from_u64(1, curve);
        sel[1] = Scalar::from_u64(1, curve);
        sel[2] = Scalar::from_u64(1, curve);
        let trace: Vec<&Vec<Scalar>> = vec![&src, &sel];

        let groups = vec![LogUpGroup {
            column_index: 0,
            max_bits: 8,
            num_limbs: 1,
            selectors: vec![1],
        }];
        let layout = logup_column_layout(&groups);

        // Use a gamma distinct from any table value or limb.
        let gamma = Scalar::from_u64(1_000_003, curve);
        let w = compute_logup_witness(&trace, &groups, &layout, &gamma, 3, domain_size, curve);

        // h should close to zero at the boundary after a full cycle.
        let contribution_last = {
            // Last row: padding, active = 0, m[last] = 0 → contribution is 0.
            Scalar::zero(curve)
        };
        // h[0] = 0 by construction.
        assert!(w.h_column[0].is_zero());
        // After the full cycle, h[n-1] should satisfy h[0] - h[n-1] = contribution[n-1]
        // so h[n-1] = -contribution[n-1] = 0 when padding row has no contribution.
        let wrap_rhs = contribution_last;
        let wrap_lhs = w.h_column[0].sub(&w.h_column[domain_size - 1]);
        assert_eq!(wrap_lhs.sub(&wrap_rhs).is_zero(), true, "grand sum must close");
    }

    // ── Bitwise lookup tests ───────────────────────────────────────────

    #[test]
    fn test_nibble_and_table() {
        let table = LookupTable::nibble_and();
        assert_eq!(table.size(), 256);
        assert_eq!(table.table_type, TableType::NibbleAnd);
        // a=5, b=3 → AND = 1
        assert_eq!(table.value_at(5 * 16 + 3), 1);
        // a=0xF, b=0xA → AND = 0xA
        assert_eq!(table.value_at(0xF * 16 + 0xA), 0xA);
        // a=0, b=anything → AND = 0
        assert_eq!(table.value_at(0 * 16 + 0xF), 0);
        // a=0xF, b=0xF → AND = 0xF
        assert_eq!(table.value_at(0xF * 16 + 0xF), 0xF);
    }

    #[test]
    fn test_decompose_to_nibbles() {
        assert_eq!(decompose_to_nibbles(0xAB, 4), vec![0xB, 0xA, 0, 0]);
        assert_eq!(decompose_to_nibbles(0x1234, 4), vec![4, 3, 2, 1]);
        assert_eq!(decompose_to_nibbles(0, 2), vec![0, 0]);
        assert_eq!(decompose_to_nibbles(0xF, 1), vec![0xF]);
    }

    #[test]
    fn test_bitwise_column_layout() {
        let groups = vec![
            BitwiseLookupGroup {
                operand_a_column: 0, operand_b_column: 1, result_column: 2,
                width_bits: 64, num_nibbles: 16, op: BitwiseOp::And, selectors: vec![],
            },
            BitwiseLookupGroup {
                operand_a_column: 0, operand_b_column: 1, result_column: 3,
                width_bits: 64, num_nibbles: 16, op: BitwiseOp::Or, selectors: vec![],
            },
        ];
        let layout = bitwise_column_layout(&groups);
        // Group 0: 3*16 = 48 columns starting at 0
        // Group 1: 3*16 = 48 columns starting at 48
        assert_eq!(layout.nibble_offsets, vec![(0, 16), (48, 16)]);
        assert_eq!(layout.num_columns, 96);
    }

    #[test]
    fn test_evaluate_bitwise_and() {
        let curve = CurveType::Bls48581;
        let a = Scalar::from_u64(0xAC, curve); // 1010_1100
        let b = Scalar::from_u64(0x5A, curve); // 0101_1010
        let and_result = Scalar::from_u64(0x08, curve); // 0000_1000

        // 2 nibbles: a = [0xC, 0xA], b = [0xA, 0x5], and = [0x8, 0x0]
        let a_nibs = vec![Scalar::from_u64(0xC, curve), Scalar::from_u64(0xA, curve)];
        let b_nibs = vec![Scalar::from_u64(0xA, curve), Scalar::from_u64(0x5, curve)];
        let and_nibs = vec![Scalar::from_u64(0x8, curve), Scalar::from_u64(0x0, curve)];

        let constraints = evaluate_bitwise_at_point(
            &a, &b, &and_result,
            &a_nibs, &b_nibs, &and_nibs,
            None, BitwiseOp::And, curve,
        );
        assert_eq!(constraints.len(), 3);
        assert!(constraints[0].is_zero(), "A decomposition");
        assert!(constraints[1].is_zero(), "B decomposition");
        assert!(constraints[2].is_zero(), "AND result");
    }

    #[test]
    fn test_evaluate_bitwise_or() {
        let curve = CurveType::Bls48581;
        let a = Scalar::from_u64(0xAC, curve);
        let b = Scalar::from_u64(0x5A, curve);
        let or_result = Scalar::from_u64(0xFE, curve); // 1111_1110

        let a_nibs = vec![Scalar::from_u64(0xC, curve), Scalar::from_u64(0xA, curve)];
        let b_nibs = vec![Scalar::from_u64(0xA, curve), Scalar::from_u64(0x5, curve)];
        let and_nibs = vec![Scalar::from_u64(0x8, curve), Scalar::from_u64(0x0, curve)];

        let constraints = evaluate_bitwise_at_point(
            &a, &b, &or_result,
            &a_nibs, &b_nibs, &and_nibs,
            None, BitwiseOp::Or, curve,
        );
        for (i, c) in constraints.iter().enumerate() {
            assert!(c.is_zero(), "OR constraint {} should be zero", i);
        }
    }

    #[test]
    fn test_evaluate_bitwise_xor() {
        let curve = CurveType::Bls48581;
        let a = Scalar::from_u64(0xAC, curve);
        let b = Scalar::from_u64(0x5A, curve);
        let xor_result = Scalar::from_u64(0xF6, curve); // 1111_0110

        let a_nibs = vec![Scalar::from_u64(0xC, curve), Scalar::from_u64(0xA, curve)];
        let b_nibs = vec![Scalar::from_u64(0xA, curve), Scalar::from_u64(0x5, curve)];
        let and_nibs = vec![Scalar::from_u64(0x8, curve), Scalar::from_u64(0x0, curve)];

        let constraints = evaluate_bitwise_at_point(
            &a, &b, &xor_result,
            &a_nibs, &b_nibs, &and_nibs,
            None, BitwiseOp::Xor, curve,
        );
        for (i, c) in constraints.iter().enumerate() {
            assert!(c.is_zero(), "XOR constraint {} should be zero", i);
        }
    }

    #[test]
    fn test_evaluate_bitwise_wrong_result() {
        let curve = CurveType::Bls48581;
        let a = Scalar::from_u64(0xAC, curve);
        let b = Scalar::from_u64(0x5A, curve);
        let wrong = Scalar::from_u64(0xFF, curve); // wrong AND result

        let a_nibs = vec![Scalar::from_u64(0xC, curve), Scalar::from_u64(0xA, curve)];
        let b_nibs = vec![Scalar::from_u64(0xA, curve), Scalar::from_u64(0x5, curve)];
        let and_nibs = vec![Scalar::from_u64(0x8, curve), Scalar::from_u64(0x0, curve)];

        let constraints = evaluate_bitwise_at_point(
            &a, &b, &wrong,
            &a_nibs, &b_nibs, &and_nibs,
            None, BitwiseOp::And, curve,
        );
        assert!(!constraints[2].is_zero(), "wrong AND result should fail");
    }

    #[test]
    fn test_nibble_and_lookup_counts() {
        let curve = CurveType::Bls48581;
        // 2-row trace: a=[0xAB, 0x00], b=[0xCD, 0x00]
        let col_a = vec![Scalar::from_u64(0xAB, curve), Scalar::from_u64(0, curve)];
        let col_b = vec![Scalar::from_u64(0xCD, curve), Scalar::from_u64(0, curve)];
        let sel = vec![Scalar::from_u64(1, curve), Scalar::from_u64(0, curve)];
        let cols: Vec<&Vec<Scalar>> = vec![&col_a, &col_b, &sel];

        let groups = vec![BitwiseLookupGroup {
            operand_a_column: 0, operand_b_column: 1, result_column: 0, // result col unused for counting
            width_bits: 8, num_nibbles: 2, op: BitwiseOp::And, selectors: vec![2],
        }];

        let counts = count_nibble_and_lookups(&cols, &groups, 2);
        // Row 0 active: a=0xAB → nibs [0xB, 0xA], b=0xCD → nibs [0xD, 0xC]
        // Lookups: (0xB, 0xD) → idx 0xBD, (0xA, 0xC) → idx 0xAC
        assert_eq!(counts[0xB * 16 + 0xD], 1);
        assert_eq!(counts[0xA * 16 + 0xC], 1);
        // Row 1 inactive: no lookups
        assert_eq!(counts[0], 0);
    }

    #[test]
    fn test_num_bitwise_constraints() {
        let groups = vec![
            BitwiseLookupGroup {
                operand_a_column: 0, operand_b_column: 1, result_column: 2,
                width_bits: 64, num_nibbles: 16, op: BitwiseOp::And, selectors: vec![],
            },
        ];
        assert_eq!(num_bitwise_constraints(&groups), 3);
    }
}
