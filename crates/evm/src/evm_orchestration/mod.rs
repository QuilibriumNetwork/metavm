//! # EVM-side gadget orchestration helper
//!
//! Sibling to `metavm_zkp::ultimate_joint_prove`: assembles witnesses
//! for all **EVM-crate** gadget AIRs from a single
//! [`crate::trace::EvmTraceColumns`] input. The zkp-side
//! `ultimate_joint_prove::OrchestrationBundle` only covers AIRs that
//! live in `metavm-zkp` (storage, account, address-keccak, byte-memory,
//! block_header, MPT, beacon-chain). The EVM-crate gadgets
//! (`env_air`, `log_air`, `blockhash_history_air`, `call_frame_air`,
//! `calldata_byte_air`, `exp_air`, `byte_air`, `gas_tracking_air`,
//! `jumpdest_table_air`) **cannot** be referenced from that crate
//! because of the unidirectional `metavm-evm → metavm-zkp` dependency.
//!
//! This module bundles them on the EVM side.
//!
//! ## Scope
//! - Witness assembly from a real EVM trace via each AIR's
//!   `from_evm_trace` / `from_bytecode` / `from_calldata` /
//!   `from_inputs` builder.
//! - Constraint-system construction with the correct
//!   `omega` / `domain_size` for each gadget's padded trace.
//! - Cross-AIR LogUp descriptor wiring from EVM main → gadget.
//!
//! ## Out of scope (in this scaffold)
//! - `joint_prove` / `joint_verify` execution — gated as slow.
//! - Soft selector binding gaps (no must-fire constraints).
//! - Bytecode-table / static-gas-table sub-tables.
//! - MSIZE watermark, JUMPI taken-branch synthetic selector.

use crate::trace::EvmTraceColumns;
use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;
use metavm_zkp::field::CurveType;
use metavm_zkp::scheme::CommitmentScheme;
use metavm_zkp::trace::TracePolynomials;
use metavm_zkp::vm_constraints::VmConstraintSystem;
use revm::primitives::U256;

use crate::blockhash_history_air as bh;
use crate::byte_air as byte_op;
use crate::call_frame_air as cf;
use crate::calldata_byte_air as cd;
use crate::env_air as env;
use crate::exp_air;
use crate::exp_constraints::EvmExpConstraintSystem;
use crate::gas_tracking_air as gt;
use crate::jumpdest_table_air as jdt;
use crate::log_air as log;

// ─── Layer indices (stable; baked into descriptor wiring) ────────────

/// Layer 0 reserved for the EVM main trace (constructed externally
/// when wiring this bundle into a `joint_prove` invocation).
pub const LAYER_EVM_MAIN: usize = 0;
/// ENV opcodes gadget (11 KIND_* opcodes).
pub const LAYER_ENV: usize = 1;
/// LOG topic0 gadget.
pub const LAYER_LOG: usize = 2;
/// BLOCKHASH history.
pub const LAYER_BLOCKHASH: usize = 3;
/// CALL / CREATE / RETURN frame transitions.
pub const LAYER_CALL_FRAME: usize = 4;
/// CALLDATALOAD byte-level table.
pub const LAYER_CALLDATA_BYTE: usize = 5;
/// EXP gadget (multi-invocation square-and-multiply).
pub const LAYER_EXP: usize = 6;
/// BYTE opcode 32-case extractor.
pub const LAYER_BYTE_OP: usize = 7;
/// Gas-tracking (opcode → static + dynamic cost).
pub const LAYER_GAS_TRACKING: usize = 8;
/// JUMPDEST validity table (one row per byte position).
pub const LAYER_JUMPDEST_TABLE: usize = 9;

/// Number of gadget layers in the EVM-side bundle (excluding the
/// reserved layer 0 for the EVM main trace).
pub const NUM_EVM_GADGET_LAYERS: usize = 9;

// ─── Bundle ──────────────────────────────────────────────────────────

/// All witnesses + traces + constraint systems composing the EVM-side
/// gadget bundle. The ordering of [`Self::traces`] matches the
/// `LAYER_*` constants (starting at [`LAYER_ENV`]).
#[allow(dead_code)]
pub struct EvmGadgetBundle {
    pub curve: CurveType,

    pub env_trace: TracePolynomials,
    pub env_cs: env::EnvConstraintSystem,

    pub log_trace: TracePolynomials,
    pub log_cs: log::LogConstraintSystem,

    pub bh_trace: TracePolynomials,
    pub bh_cs: bh::HistoryConstraintSystem,

    pub cf_trace: TracePolynomials,
    pub cf_cs: cf::CallFrameConstraintSystem,

    pub cd_trace: TracePolynomials,
    pub cd_cs: cd::CalldataConstraintSystem,

    pub exp_trace: TracePolynomials,
    pub exp_cs: EvmExpConstraintSystem,

    pub byte_trace: TracePolynomials,
    pub byte_cs: byte_op::ByteOpConstraintSystem,

    pub gt_trace: TracePolynomials,
    pub gt_cs: gt::GasTrackingConstraintSystem,

    pub jdt_trace: TracePolynomials,
    pub jdt_cs: jdt::JumpdestTableConstraintSystem,
}

impl EvmGadgetBundle {
    /// Borrow `(trace, constraint_system)` pairs in `LAYER_*` order
    /// starting at layer index 1 (layer 0 is reserved for the
    /// caller-supplied EVM main trace).
    pub fn traces<'a>(&'a self) -> Vec<(&'a TracePolynomials, &'a dyn VmConstraintSystem)> {
        vec![
            (&self.env_trace, &self.env_cs as &dyn VmConstraintSystem),
            (&self.log_trace, &self.log_cs),
            (&self.bh_trace, &self.bh_cs),
            (&self.cf_trace, &self.cf_cs),
            (&self.cd_trace, &self.cd_cs),
            (&self.exp_trace, &self.exp_cs),
            (&self.byte_trace, &self.byte_cs),
            (&self.gt_trace, &self.gt_cs),
            (&self.jdt_trace, &self.jdt_cs),
        ]
    }

    /// Borrow only the constraint systems, in the same order.
    pub fn cs_refs<'a>(&'a self) -> Vec<&'a dyn VmConstraintSystem> {
        vec![
            &self.env_cs,
            &self.log_cs,
            &self.bh_cs,
            &self.cf_cs,
            &self.cd_cs,
            &self.exp_cs,
            &self.byte_cs,
            &self.gt_cs,
            &self.jdt_cs,
        ]
    }
}

// ─── EXP gadget trace helper ─────────────────────────────────────────

/// Extract (base, exponent) invocations from an EVM trace by scanning
/// `sel_exp` rows. Returns the input tuples in trace order.
pub fn extract_exp_invocations(cols: &EvmTraceColumns) -> Vec<(U256, U256)> {
    let n = cols.step.len();
    let mut out = Vec::new();
    for r in 0..n {
        if cols.sel_exp[r] != 1 {
            continue;
        }
        let base = U256::from_limbs([
            cols.input0[0][r],
            cols.input0[1][r],
            cols.input0[2][r],
            cols.input0[3][r],
        ]);
        let exponent = U256::from_limbs([
            cols.input1[0][r],
            cols.input1[1][r],
            cols.input1[2][r],
            cols.input1[3][r],
        ]);
        out.push((base, exponent));
    }
    out
}

/// Extract BYTE-opcode (index_limb0, value_limbs) invocations from an
/// EVM trace by scanning `sel_byte_op` rows. The index is forced down
/// to a single limb (BYTE's stack input is always < 256 to do anything
/// meaningful; out-of-range indices return 0 algebraically).
pub fn extract_byte_op_invocations(cols: &EvmTraceColumns) -> Vec<(u64, [u64; 4])> {
    let n = cols.step.len();
    let mut out = Vec::new();
    for r in 0..n {
        if cols.sel_byte_op[r] != 1 {
            continue;
        }
        let index = cols.input0[0][r];
        let value = [
            cols.input1[0][r],
            cols.input1[1][r],
            cols.input1[2][r],
            cols.input1[3][r],
        ];
        out.push((index, value));
    }
    out
}

/// Build an EXP gadget [`TracePolynomials`] from a list of
/// `(base, exponent)` invocations. Uses
/// [`exp_air::populate_multi_trace`] for N>0 invocations and falls
/// back to a single dummy invocation (`(0, 0)`) for empty traces (so
/// the bundle stays well-formed even when no EXP rows are present).
fn build_exp_trace(invocations: &[(U256, U256)], curve: CurveType) -> TracePolynomials {
    let invs: Vec<(U256, U256)> = if invocations.is_empty() {
        vec![(U256::ZERO, U256::ZERO)]
    } else {
        invocations.to_vec()
    };
    let columns = exp_air::populate_multi_trace(&invs, curve);
    let total_rows = invs.len() * exp_air::NUM_STEPS;
    let padded = metavm_zkp::trace::nearest_power_of_two(total_rows.max(1));
    let polys: Vec<metavm_zkp::trace::Polynomial> = columns
        .into_iter()
        .map(|mut evals| {
            if evals.len() < padded {
                evals.resize(padded, metavm_zkp::field::Scalar::zero(curve));
            }
            metavm_zkp::trace::Polynomial {
                evaluations: evals,
                degree: total_rows,
            }
        })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows: total_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Bundle assembler ────────────────────────────────────────────────

/// Build the entire EVM gadget bundle from a single EVM trace + the
/// matching bytecode and calldata. **No proving is performed.**
///
/// Selecting `BLS48-581` matches the rest of the EVM crate's defaults;
/// callers that need wider FFT widths (e.g. multi-invocation EXP with
/// many opcodes) should construct individual witnesses manually with
/// `BLS12-381` per the
/// [`evm_exp_multi_invocation`](../../evm_exp_multi_invocation.md) note.
pub fn assemble_from_trace(
    scheme: &dyn CommitmentScheme,
    cols: &EvmTraceColumns,
    bytecode: &[u8],
    calldata: &[u8],
) -> EvmGadgetBundle {
    let curve = CurveType::Bls48581;

    // ── ENV ───────────────────────────────────────────────────────
    let env_w = env::from_evm_trace(cols);
    let env_trace = env::build_trace_polynomials(&env_w, curve);
    let env_omega = scheme.domain_generator(env_trace.padded_size);
    let env_cs = env::EnvConstraintSystem::new(env_trace.num_rows)
        .with_omega_and_domain(env_omega, env_trace.padded_size);

    // ── LOG ───────────────────────────────────────────────────────
    let log_w = log::from_evm_trace(cols);
    let log_trace = log::build_trace_polynomials(&log_w, curve);
    let log_omega = scheme.domain_generator(log_trace.padded_size);
    let log_cs = log::LogConstraintSystem::new(log_trace.num_rows)
        .with_omega_and_domain(log_omega, log_trace.padded_size);

    // ── BLOCKHASH history (empty by default — real wiring needs an
    //    oracle that knows the last-256-blocks hashes). ────────────
    let bh_w = bh::HistoryWitness::default();
    let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
    let bh_omega = scheme.domain_generator(bh_trace.padded_size);
    let bh_cs = bh::HistoryConstraintSystem::new(bh_trace.num_rows)
        .with_omega_and_domain(bh_omega, bh_trace.padded_size);

    // ── Call-frame transitions ────────────────────────────────────
    let cf_w = cf::from_evm_trace(cols);
    let cf_trace = cf::build_trace_polynomials(&cf_w, curve);
    let cf_omega = scheme.domain_generator(cf_trace.padded_size);
    let cf_cs = cf::CallFrameConstraintSystem::new(cf_trace.num_rows)
        .with_omega_and_domain(cf_omega, cf_trace.padded_size);

    // ── Calldata byte table ───────────────────────────────────────
    let cd_w = cd::CalldataWitness::from_calldata(calldata);
    let cd_trace = cd::build_trace_polynomials(&cd_w, curve);
    let cd_omega = scheme.domain_generator(cd_trace.padded_size);
    let cd_cs = cd::CalldataConstraintSystem::new(cd_trace.num_rows)
        .with_omega_and_domain(cd_omega, cd_trace.padded_size);

    // ── EXP gadget (multi-invocation) ─────────────────────────────
    let exp_invs = extract_exp_invocations(cols);
    let exp_trace = build_exp_trace(&exp_invs, curve);
    let exp_cs = EvmExpConstraintSystem::with_num_rows(exp_trace.num_rows);

    // ── BYTE opcode gadget ────────────────────────────────────────
    let byte_invs = extract_byte_op_invocations(cols);
    let byte_w = if byte_invs.is_empty() {
        byte_op::ByteOpWitness::default()
    } else {
        byte_op::ByteOpWitness::from_inputs(&byte_invs)
    };
    let byte_trace = byte_op::build_trace_polynomials(&byte_w, curve);
    let byte_omega = scheme.domain_generator(byte_trace.padded_size);
    let byte_cs = byte_op::ByteOpConstraintSystem::new(byte_trace.num_rows)
        .with_omega_and_domain(byte_omega, byte_trace.padded_size);

    // ── Gas tracking ──────────────────────────────────────────────
    let gt_w = gt::from_evm_trace(cols);
    let gt_trace = gt::build_trace_polynomials(&gt_w, curve);
    let gt_omega = scheme.domain_generator(gt_trace.padded_size);
    let gt_cs = gt::GasTrackingConstraintSystem::new(gt_trace.num_rows)
        .with_omega_and_domain(gt_omega, gt_trace.padded_size);

    // ── JUMPDEST table (derived from bytecode) ────────────────────
    let jdt_w = jdt::JumpdestTableWitness::from_bytecode(bytecode);
    let jdt_trace = jdt::build_trace_polynomials(&jdt_w, curve);
    let jdt_omega = scheme.domain_generator(jdt_trace.padded_size);
    let jdt_cs = jdt::JumpdestTableConstraintSystem::new(jdt_trace.num_rows)
        .with_omega_and_domain(jdt_omega, jdt_trace.padded_size);

    EvmGadgetBundle {
        curve,
        env_trace,
        env_cs,
        log_trace,
        log_cs,
        bh_trace,
        bh_cs,
        cf_trace,
        cf_cs,
        cd_trace,
        cd_cs,
        exp_trace,
        exp_cs,
        byte_trace,
        byte_cs,
        gt_trace,
        gt_cs,
        jdt_trace,
        jdt_cs,
    }
}

// ─── Descriptor wiring ───────────────────────────────────────────────

/// Collect every cross-AIR LogUp descriptor that wires the EVM main
/// trace to one of the gadget AIRs in this bundle.
///
/// Layer indices follow the [`LAYER_*`] constants. `evm_layer_index`
/// is the layer index of the EVM main trace in the caller's
/// `joint_prove` traces vector (typically `LAYER_EVM_MAIN = 0`); each
/// gadget layer is computed as `gadget_base + LAYER_*`, where
/// `gadget_base` is the offset into the caller's traces vector at
/// which the bundle starts.
///
/// In most cases callers pass `gadget_base = 1` (gadgets immediately
/// follow the EVM main trace).
pub fn collect_evm_descriptors(evm_layer_index: usize, gadget_base: usize) -> Vec<CrossAirLogUpDescriptor> {
    let env_l = gadget_base + (LAYER_ENV - 1);
    let log_l = gadget_base + (LAYER_LOG - 1);
    let bh_l = gadget_base + (LAYER_BLOCKHASH - 1);
    let cf_l = gadget_base + (LAYER_CALL_FRAME - 1);
    let cd_l = gadget_base + (LAYER_CALLDATA_BYTE - 1);
    let exp_l = gadget_base + (LAYER_EXP - 1);
    let byte_l = gadget_base + (LAYER_BYTE_OP - 1);
    let gt_l = gadget_base + (LAYER_GAS_TRACKING - 1);
    let jdt_l = gadget_base + (LAYER_JUMPDEST_TABLE - 1);

    vec![
        // ── ENV opcodes (11 descriptors) ─────────────────────────
        env::make_address_descriptor(evm_layer_index, env_l),
        env::make_caller_descriptor(evm_layer_index, env_l),
        env::make_callvalue_descriptor(evm_layer_index, env_l),
        env::make_origin_descriptor(evm_layer_index, env_l),
        env::make_gasprice_descriptor(evm_layer_index, env_l),
        env::make_calldatasize_descriptor(evm_layer_index, env_l),
        env::make_codesize_descriptor(evm_layer_index, env_l),
        env::make_returndatasize_descriptor(evm_layer_index, env_l),
        env::make_pc_descriptor(evm_layer_index, env_l),
        env::make_gas_descriptor(evm_layer_index, env_l),
        env::make_msize_descriptor(evm_layer_index, env_l),

        // ── LOG topics (4 descriptors) ───────────────────────────
        log::make_log1_descriptor(evm_layer_index, log_l),
        log::make_log2_descriptor(evm_layer_index, log_l),
        log::make_log3_descriptor(evm_layer_index, log_l),
        log::make_log4_descriptor(evm_layer_index, log_l),

        // ── BLOCKHASH ────────────────────────────────────────────
        bh::make_blockhash_descriptor(evm_layer_index, bh_l),

        // ── Call-frame: CALL family + RETURN/REVERT/STOP ─────────
        cf::make_call_descriptor(evm_layer_index, cf_l),
        cf::make_evm_to_call_frame_air_descriptor(
            "evm_callcode_to_call_frame_air_v1",
            evm_layer_index,
            cf_l,
            crate::trace::COL_SEL_CALLCODE,
            cf::COL_SEL_CALLCODE,
        ),
        cf::make_evm_to_call_frame_air_descriptor(
            "evm_delegatecall_to_call_frame_air_v1",
            evm_layer_index,
            cf_l,
            crate::trace::COL_SEL_DELEGATECALL,
            cf::COL_SEL_DELEGATECALL,
        ),
        cf::make_evm_to_call_frame_air_descriptor(
            "evm_staticcall_to_call_frame_air_v1",
            evm_layer_index,
            cf_l,
            crate::trace::COL_SEL_STATICCALL,
            cf::COL_SEL_STATICCALL,
        ),
        cf::make_evm_to_call_frame_air_descriptor(
            "evm_create_to_call_frame_air_v1",
            evm_layer_index,
            cf_l,
            crate::trace::COL_SEL_CREATE,
            cf::COL_SEL_CREATE,
        ),
        cf::make_evm_to_call_frame_air_descriptor(
            "evm_create2_to_call_frame_air_v1",
            evm_layer_index,
            cf_l,
            crate::trace::COL_SEL_CREATE2,
            cf::COL_SEL_CREATE2,
        ),
        cf::make_evm_to_call_frame_air_descriptor(
            "evm_revert_to_call_frame_air_v1",
            evm_layer_index,
            cf_l,
            crate::trace::COL_SEL_REVERT,
            cf::COL_SEL_REVERT,
        ),
        cf::make_evm_to_call_frame_air_descriptor(
            "evm_stop_pop_to_call_frame_air_v1",
            evm_layer_index,
            cf_l,
            crate::trace::COL_SEL_STOP_POP,
            cf::COL_SEL_STOP_POP,
        ),

        // ── Calldata byte ────────────────────────────────────────
        cd::make_calldataload_byte_descriptor(evm_layer_index, cd_l, 0),

        // ── EXP gadget (multi-invocation) ────────────────────────
        crate::cross_air_linkage::make_evm_exp_linkage_descriptor(evm_layer_index, exp_l),

        // ── BYTE opcode ─────────────────────────────────────────
        byte_op::make_evm_byte_linkage_descriptor(evm_layer_index, byte_l),

        // ── Gas tracking ────────────────────────────────────────
        gt::make_evm_to_gas_tracking_descriptor(evm_layer_index, gt_l),

        // ── JUMPDEST table ──────────────────────────────────────
        jdt::make_evm_jump_to_jumpdest_table_descriptor(evm_layer_index, jdt_l),
    ]
}

// ─── Constraint counting ─────────────────────────────────────────────

/// Sum of `NUM_ROW_CONSTRAINTS` across every EVM gadget AIR in the
/// bundle. Coarse complexity metric; actual prove-time cost scales
/// with `padded_size * num_constraints * num_quotient_columns`.
pub fn count_evm_constraints() -> usize {
    env::NUM_ROW_CONSTRAINTS
        + log::NUM_ROW_CONSTRAINTS
        + bh::NUM_ROW_CONSTRAINTS
        + cf::NUM_ROW_CONSTRAINTS
        + cd::NUM_ROW_CONSTRAINTS
        + crate::exp_constraints::NUM_ROW_CONSTRAINTS
        + byte_op::NUM_ROW_CONSTRAINTS
        + gt::NUM_ROW_CONSTRAINTS
        + jdt::NUM_ROW_CONSTRAINTS
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;
    use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;

    fn make_scheme() -> Bls48581Scheme {
        let s = Bls48581Scheme::new();
        s.init();
        s
    }

    #[test]
    fn bundle_assembles_from_simple_trace() {
        // PUSH1 1; PUSH1 2; ADD; STOP
        let bc = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let scheme = make_scheme();
        let bundle = assemble_from_trace(&scheme, &cols, &bc, &[]);
        let traces = bundle.traces();
        assert_eq!(traces.len(), NUM_EVM_GADGET_LAYERS);
        assert_eq!(bundle.cs_refs().len(), NUM_EVM_GADGET_LAYERS);
    }

    #[test]
    fn bundle_assembles_from_sstore_sload_trace() {
        // PUSH1 0x42; PUSH1 0x00; SSTORE; PUSH1 0x00; SLOAD; STOP
        let bc = vec![0x60, 0x42, 0x60, 0x00, 0x55, 0x60, 0x00, 0x54, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let scheme = make_scheme();
        let bundle = assemble_from_trace(&scheme, &cols, &bc, &[]);
        assert_eq!(bundle.traces().len(), NUM_EVM_GADGET_LAYERS);
    }

    #[test]
    fn bundle_assembles_from_log1_trace() {
        // PUSH32 topic; PUSH1 0; PUSH1 0; LOG1; STOP
        let mut bc = vec![0x7F];
        bc.extend_from_slice(&[0xAA; 32]);
        bc.extend_from_slice(&[0x60, 0x00, 0x60, 0x00, 0xA1, 0x00]);
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let scheme = make_scheme();
        let bundle = assemble_from_trace(&scheme, &cols, &bc, &[]);
        assert_eq!(bundle.traces().len(), NUM_EVM_GADGET_LAYERS);
        // LOG row was extracted on log_air side.
        let log_w = log::from_evm_trace(&cols);
        assert_eq!(log_w.rows.len(), 1);
    }

    #[test]
    fn descriptor_count_exceeds_twenty() {
        let descriptors = collect_evm_descriptors(LAYER_EVM_MAIN, 1);
        assert!(
            descriptors.len() >= 20,
            "expected >=20 descriptors, got {}",
            descriptors.len(),
        );
    }

    #[test]
    fn constraint_sum_exceeds_fifty() {
        let total = count_evm_constraints();
        assert!(
            total >= 50,
            "expected >=50 row-constraints, got {}",
            total,
        );
        eprintln!("[evm_orchestration] total row-constraints = {}", total);
    }

    #[test]
    fn bundle_handles_empty_trace_gracefully() {
        // STOP only — minimal trace, no EXP/BYTE/LOG/SSTORE rows.
        let bc = vec![0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let scheme = make_scheme();
        let bundle = assemble_from_trace(&scheme, &cols, &bc, &[]);
        // All gadget layers still present with valid (possibly empty
        // or single-padded) traces.
        assert_eq!(bundle.traces().len(), NUM_EVM_GADGET_LAYERS);
        // EXP trace falls back to a dummy single invocation when no
        // EVM EXP rows are present.
        assert_eq!(bundle.exp_trace.num_rows, exp_air::NUM_STEPS);
    }

    #[test]
    fn descriptors_reference_valid_layer_indices() {
        let descriptors = collect_evm_descriptors(LAYER_EVM_MAIN, 1);
        for d in &descriptors {
            assert_eq!(
                d.a_layer_index, LAYER_EVM_MAIN,
                "descriptor {} should source from EVM main layer",
                d.label,
            );
            assert!(
                d.b_layer_index >= 1
                    && d.b_layer_index <= NUM_EVM_GADGET_LAYERS,
                "descriptor {} b_layer_index={} out of range",
                d.label,
                d.b_layer_index,
            );
        }
    }

    #[test]
    fn descriptors_have_nonempty_columns() {
        for d in collect_evm_descriptors(LAYER_EVM_MAIN, 1) {
            assert!(
                !d.a_columns.is_empty(),
                "descriptor {} has empty a_columns",
                d.label,
            );
            assert!(
                !d.b_columns.is_empty(),
                "descriptor {} has empty b_columns",
                d.label,
            );
        }
    }

    #[test]
    fn extract_exp_invocations_simple() {
        // PUSH1 5; PUSH1 2; EXP; STOP (computes 2^5 = 32)
        let bc = vec![0x60, 0x05, 0x60, 0x02, 0x0A, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let invs = extract_exp_invocations(&cols);
        assert_eq!(invs.len(), 1);
        assert_eq!(invs[0].0, U256::from(2u64));
        assert_eq!(invs[0].1, U256::from(5u64));
    }

    #[test]
    fn extract_byte_op_invocations_simple() {
        // PUSH32 (some value); PUSH1 31; BYTE; STOP → byte at index 31 (last)
        let mut bc = vec![0x7F];
        let mut v = [0u8; 32];
        v[31] = 0x42;
        bc.extend_from_slice(&v);
        bc.extend_from_slice(&[0x60, 0x1F, 0x1A, 0x00]);
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let invs = extract_byte_op_invocations(&cols);
        assert_eq!(invs.len(), 1);
        assert_eq!(invs[0].0, 31);
    }
}
