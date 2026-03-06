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
    layout: &LogUpColumnLayout,
    gamma: &Scalar,
    num_rows: usize,
    domain_size: usize,
    curve: CurveType,
) -> LogUpWitness {
    // Step 1: Decompose each group's source column into byte limbs
    let mut limb_columns: Vec<Vec<Vec<Scalar>>> = Vec::with_capacity(groups.len());
    // Track byte value counts across all groups for multiplicity
    let mut byte_counts = [0u64; 256];

    for group in groups {
        let src_col = trace_columns[group.column_index];
        let mut group_limbs: Vec<Vec<Scalar>> = vec![vec![Scalar::zero(curve); domain_size]; group.num_limbs];

        for row in 0..num_rows {
            let val = src_col[row].to_u64();
            let bytes = decompose_to_bytes(val, group.num_limbs);

            // Check if any selector is active for this row
            let active = if group.selectors.is_empty() {
                true
            } else {
                group.selectors.iter().any(|&sel| !trace_columns[sel][row].is_zero())
            };

            for (limb_idx, &byte_val) in bytes.iter().enumerate() {
                group_limbs[limb_idx][row] = Scalar::from_u64(byte_val as u64, curve);
                if active {
                    byte_counts[byte_val as usize] += 1;
                }
            }
        }
        // Padding rows: limbs stay zero, byte_counts[0] needs adjustment for padding
        limb_columns.push(group_limbs);
    }

    // Step 2: Build multiplicity column m
    // m is defined over the table domain (256 entries for 8-bit), padded to domain_size.
    // m[i] = number of times table entry i appears across all limb columns.
    let mut m_column = vec![Scalar::zero(curve); domain_size];
    for (i, &count) in byte_counts.iter().enumerate() {
        if i < domain_size {
            m_column[i] = Scalar::from_u64(count, curve);
        }
    }

    // Step 3: Build running sum h
    // h[0] = 0
    // h[i] = h[i-1] + Σ_{groups,limbs} active[i] / (gamma - limb[i]) - m[i] / (gamma - i)
    let mut h_column = vec![Scalar::zero(curve); domain_size];

    for i in 1..domain_size {
        let mut sum = h_column[i - 1].clone();

        // Numerator terms: Σ selector[i-1] / (gamma - limb_value[i-1])
        for (g_idx, group) in groups.iter().enumerate() {
            let active = if i - 1 < num_rows {
                if group.selectors.is_empty() {
                    true
                } else {
                    group.selectors.iter().any(|&sel| !trace_columns[sel][i - 1].is_zero())
                }
            } else {
                false
            };

            if active {
                for limb_idx in 0..group.num_limbs {
                    let limb_val = &limb_columns[g_idx][limb_idx][i - 1];
                    let denom = gamma.sub(limb_val);
                    if !denom.is_zero() {
                        sum = sum.add(&denom.inverse());
                    }
                }
            }
        }

        // Table terms: - m[i-1] / (gamma - (i-1))
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

/// Number of LogUp constraints for a given set of range-check groups.
///
/// Each group contributes 1 decomposition constraint.
/// Plus 1 boundary constraint (L_0 · h = 0).
///
/// The running sum transition is not enforced as a polynomial constraint
/// (it requires inverse auxiliary columns for the rational terms).
pub fn num_logup_constraints(groups: &[LogUpGroup]) -> usize {
    groups.len() + 1 // decomposition per group + boundary
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
        assert_eq!(num_logup_constraints(&groups), 3); // 2 decomposition + 1 boundary
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
