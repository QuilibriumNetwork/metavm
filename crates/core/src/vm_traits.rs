//! VM-agnostic traits for execution and trace generation.
//!
//! These traits define the interface between a VM executor (RISC-V, EVM, Solana SBF)
//! and the proving pipeline. Any VM that implements these traits can be proven.

/// Column-oriented execution trace (VM-agnostic).
///
/// A trace is a matrix of u64 values where each column represents a specific
/// aspect of the computation (e.g., program counter, register values, memory
/// addresses). The number and meaning of columns is VM-specific.
pub trait VmTrace: Send {
    /// Number of execution steps (rows) in the trace.
    fn num_steps(&self) -> usize;

    /// Number of columns in the trace.
    fn num_columns(&self) -> usize;

    /// Get a single column by index.
    fn column(&self, index: usize) -> &[u64];

    /// Get all columns as a vector of slices.
    fn columns(&self) -> Vec<&[u64]>;

    /// Names of each column (for debugging/display).
    fn column_names(&self) -> Vec<&'static str>;
}

/// VM executor that produces traces and state hashes.
///
/// Implementations wrap a specific VM (RISC-V, EVM, SBF) and provide
/// a uniform step-by-step execution interface with trace recording.
pub trait VmExecutor: Send {
    /// The trace type this executor produces.
    type Trace: VmTrace;

    /// Execute one instruction. Returns Ok(true) if there are more instructions,
    /// Ok(false) if execution completed normally, or Err on error.
    fn step(&mut self) -> Result<bool, Box<dyn std::error::Error>>;

    /// Check if the VM has halted.
    fn is_halted(&self) -> bool;

    /// Get the current program counter.
    fn pc(&self) -> u64;

    /// Begin recording a new trace segment.
    fn begin_trace(&mut self);

    /// End trace recording and return the recorded trace.
    fn end_trace(&mut self) -> Self::Trace;

    /// Compute a state hash for the current VM state.
    /// Used for chunk boundary state chain verification.
    fn state_hash(&self, step_number: u64) -> [u8; 32];

    /// Drain any output produced during execution (e.g., UART bytes).
    fn drain_output(&mut self) -> Vec<u8>;
}
