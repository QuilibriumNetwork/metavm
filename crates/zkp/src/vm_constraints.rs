//! VM-agnostic constraint system trait.
//!
//! This trait defines the interface between VM-specific constraint evaluation
//! and the generic proving pipeline. Each VM (RISC-V, EVM, SBF) implements
//! this trait to define its own correctness constraints.

use crate::field::Scalar;
use crate::lookup::{LookupRequirements, BitwiseLookupDeclaration};

/// VM-specific constraint system that defines correctness conditions.
///
/// Implementations evaluate algebraic constraints over execution traces,
/// producing polynomials that vanish on valid executions. The proving
/// pipeline uses these to construct quotient polynomials and proofs.
pub trait VmConstraintSystem: Send + Sync {
    /// Number of distinct constraints in this system.
    fn num_constraints(&self) -> usize;

    /// Human-readable labels for each constraint (for debugging).
    fn constraint_labels(&self) -> Vec<String>;

    /// Evaluate all constraints on the full trace domain.
    ///
    /// Given `columns` (one `Vec<Scalar>` per trace column) and `num_rows`
    /// (the number of actual execution steps before padding), returns one
    /// `Vec<Scalar>` per constraint. Each output vector should be zero at
    /// every row where the constraint is satisfied.
    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        num_rows: usize,
    ) -> Vec<Vec<Scalar>>;

    /// Evaluate the combined constraint at a single point `z`.
    ///
    /// Given `col_evals_at_z` (the evaluation of each trace column at `z`)
    /// and `alpha` (the random linear combination challenge), returns the
    /// combined constraint value C(z) = Σ α^i · constraint_i(z).
    ///
    /// Uses algebraic selector multiplication (no integer branching) so that
    /// the result is correct at arbitrary field elements, not just trace rows.
    fn evaluate_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        alpha: &Scalar,
    ) -> Scalar;

    /// Column indices for selector columns, in order (s_0, s_1, ..., s_{K-1}).
    ///
    /// These are the columns in the trace that contain one-hot instruction type
    /// selectors. The indices refer to the data column layout (step excluded).
    ///
    /// Default: empty (no selectors). Override for selector-based constraint systems.
    fn selector_column_indices(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Which selector column should be set to 1 on padding rows.
    ///
    /// Padding rows have all-zero data columns. The selector sum-to-one
    /// constraint requires exactly one selector to be 1 on every row,
    /// including padding. This method returns the index of the selector
    /// whose gated constraint body evaluates to zero when all data columns
    /// are zero (e.g., STOP for EVM, EXIT for SBF, SYSTEM for RISC-V).
    ///
    /// Default: returns the first selector column index, or `None` if no selectors.
    fn padding_selector_column(&self) -> Option<usize> {
        self.selector_column_indices().first().copied()
    }

    /// Build the combined constraint polynomial C(x) in coefficient form.
    ///
    /// Given `column_coeffs[i]` as the coefficient-form polynomial for data
    /// column i (step excluded), the random combination challenge `alpha`,
    /// and the `domain_size`, returns C(x) = Σ α^k · s_k(x) · constraint_k(x)
    /// at full algebraic degree (up to 3(n-1) for quadratic constraints gated
    /// by selectors).
    ///
    /// The verifier does NOT call this — instead it calls `evaluate_at_point`
    /// to recompute C(z) from column evaluations. This method is used by the
    /// prover to construct the quotient polynomial Q(x) = C(x) / Z(x).
    ///
    /// Default: returns a zero polynomial. Override for selector-based constraint systems.
    fn build_constraint_polynomial(
        &self,
        _column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        vec![Scalar::zero(alpha.curve_type())]
    }

    /// Fix trace column values on padding rows for cross-row constraint satisfaction.
    ///
    /// Called by the prover after selector padding is applied but before columns
    /// are committed. VMs with cross-row constraints (e.g., PC continuity) should
    /// override this to set padding-row values that satisfy those constraints.
    ///
    /// - `columns`: mutable evaluation-form columns (without step column).
    /// - `num_rows`: the number of real trace rows.
    /// - `padded_size`: the total domain size (power of 2).
    fn fix_trace_padding(
        &self,
        _columns: &mut [Vec<Scalar>],
        _num_rows: usize,
        _padded_size: usize,
    ) {}

    // ── Cross-row (shifted) constraint support ──────────────────────────────

    /// Column indices that need shifted (ω·z) evaluations for cross-row constraints.
    ///
    /// Returns the indices of columns whose values at the *next* domain point
    /// (ω·X) are referenced by transition constraints (e.g., PC continuity:
    /// `col_pc(ω·X) == col_next_pc(X)`).
    ///
    /// Default: empty (no cross-row constraints).
    fn shifted_column_indices(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Number of cross-row (shifted) constraints.
    ///
    /// These constraints use alpha powers starting from `num_constraints()`.
    fn num_shifted_constraints(&self) -> usize {
        0
    }

    /// Evaluate cross-row constraints at a single point z.
    ///
    /// Returns the contribution of shifted constraints to C(z):
    ///   Σ α^(offset+i) · constraint_i(z)
    ///
    /// Each cross-row constraint is multiplied by `(z - ω^{n-1})` to exclude
    /// the last domain row (wrap-around).
    ///
    /// - `col_evals_at_z`: all column evaluations at z (same as evaluate_at_point)
    /// - `shifted_evals`: evaluations at ω·z for columns in shifted_column_indices()
    /// - `z`: the challenge point
    /// - `omega_n_minus_1`: ω^{n-1} (the last domain element)
    /// - `alpha`: random combination challenge
    /// - `alpha_offset`: starting alpha power index (typically num_constraints())
    fn evaluate_shifted_at_point(
        &self,
        _col_evals_at_z: &[Scalar],
        _shifted_evals: &[Scalar],
        _z: &Scalar,
        _omega_n_minus_1: &Scalar,
        _alpha: &Scalar,
        _alpha_offset: usize,
    ) -> Scalar {
        Scalar::zero(_alpha.curve_type())
    }

    /// Build the shifted constraint polynomial contribution in coefficient form.
    ///
    /// Returns the polynomial for cross-row constraints, already multiplied by
    /// `(X - ω^{n-1})` so they vanish on the full domain (enabling division by
    /// Z_H(X) = X^n - 1).
    ///
    /// Uses alpha powers starting from `alpha_offset` (typically num_constraints()).
    fn build_shifted_constraint_polynomial(
        &self,
        _column_coeffs: &[Vec<Scalar>],
        _alpha: &Scalar,
        _domain_size: u64,
        _omega: &Scalar,
        _alpha_offset: usize,
    ) -> Vec<Scalar> {
        vec![Scalar::zero(_alpha.curve_type())]
    }

    // ── Lookup / range check declarations ────────────────────────────────

    /// Declare which columns need range checking via lookup arguments.
    ///
    /// Returns a [`LookupRequirements`] describing which trace columns must
    /// have values within specific ranges, and which lookup tables to use.
    ///
    /// The prover uses these declarations to construct lookup proofs
    /// (e.g., LogUp or Plookup). The verifier uses them to verify the proofs.
    ///
    /// Default: no lookups required.
    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements::none()
    }

    /// Declare which trace columns need bitwise verification via nibble-AND lookup.
    ///
    /// Returns declarations for AND/OR/XOR operations. Each declaration specifies
    /// two operand columns, a result column, and the operation type. The prover
    /// decomposes operands into nibbles and looks up AND(a_i, b_i) in a 256-entry
    /// nibble-AND table. OR and XOR are derived algebraically.
    ///
    /// Default: no bitwise lookups.
    fn bitwise_lookup_declarations(&self) -> Vec<BitwiseLookupDeclaration> {
        Vec::new()
    }

    // ── Memory permutation declarations ──────────────────────────────

    /// Column indices for the memory permutation argument.
    ///
    /// Returns `(addr_col, val_cols, load_sels, store_sels)` where:
    /// - `addr_col`: index of the memory address column
    /// - `val_cols`: indices of value column(s) (1 for RISC-V/SBF, 4 for EVM limbs)
    /// - `load_sels`: indices of load selector columns (any=1 → read)
    /// - `store_sels`: indices of store selector columns (any=1 → write, rw=1)
    ///
    /// Rows where any load_sel=1 are reads (rw=0), rows where any store_sel=1
    /// are writes (rw=1). Rows where neither is 1 are non-memory operations
    /// (contribute dummy entries).
    ///
    /// Default: None (no memory permutation).
    fn memory_columns(&self) -> Option<(usize, Vec<usize>, Vec<usize>, Vec<usize>)> {
        None
    }

    // ── Register file permutation declarations ──────────────────────────

    /// Column indices for register file access.
    ///
    /// Returns a list of register access ports. Each port is a
    /// `(reg_col, val_col, is_write)` tuple where:
    /// - `reg_col`: index of the register number column
    /// - `val_col`: index of the register value column
    /// - `is_write`: true if this port writes to the register file
    ///
    /// RISC-V: 3 ports (rs1 read, rs2 read, rd write).
    /// SBF: 2 ports (src read, dst read/write).
    /// EVM: 0 ports (stack-based, no explicit register file).
    ///
    /// The prover builds a multi-port grand product from these declarations.
    /// Each port contributes one (reg, val, timestamp, rw) tuple per row.
    ///
    /// Default: empty (no register file permutation).
    fn register_ports(&self) -> Vec<(usize, usize, bool)> {
        Vec::new()
    }

    // ── Oracle public-input declarations ─────────────────────────────────

    /// Column indices for oracle-verified operations.
    ///
    /// Returns a list of `(selector_col, input_cols, output_cols)` tuples
    /// describing operations whose correctness is verified externally by
    /// the verifier rather than by polynomial constraints.
    ///
    /// The prover collects all oracle operations and includes them in the
    /// proof as public inputs. The constraint system's zero-body constraints
    /// accept any output; the verifier re-executes the operations to check
    /// correctness.
    ///
    /// Default: empty (no oracle operations).
    fn oracle_selectors(&self) -> Vec<usize> {
        Vec::new()
    }
}
