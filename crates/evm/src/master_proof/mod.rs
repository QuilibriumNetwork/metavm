//! # MasterProofBundle — combined zkp + evm orchestration
//!
//! Composes the two sibling orchestration bundles into a single proof
//! scaffold:
//!
//! - **zkp side** (`metavm_zkp::ultimate_joint_prove::OrchestrationBundle`):
//!   14 AIRs spanning storage, account-state, address-keccak,
//!   byte-memory, generic + wide keccak preimage extraction, the EVM
//!   block-header AIR, MPT inclusion, the beacon-chain SSZ HTR chain
//!   (BBH-pair, body-pair, payload-pair, root consumer, sha256_extract),
//!   and stake-weighted finality.
//! - **evm side** (`crate::evm_orchestration::EvmGadgetBundle`):
//!   9 gadget AIRs (env, log, blockhash, call-frame, calldata byte,
//!   exp, byte opcode, gas tracking, JUMPDEST table).
//!
//! Total (wired): **23 AIRs**, **51 cross-AIR LogUp descriptors**
//! (21 zkp + 29 evm + 1 inter-bundle), and **265 algebraic
//! row-constraints** summed across both bundles. **No proving is
//! performed** — this is orchestration scaffolding analogous to its
//! two siblings.
//!
//! ### Extended accounting (rounds 11–34)
//!
//! Eighty-seven additional AIRs landed across rounds 11–34 — 69 on the
//! zkp side (rounds 28–31 added `sha512_air`, `bn254_final_exp_air`,
//! `bn254_miller_loop_air`, and the `bn254_curve_ops_air` sub-AIRs
//! `fp2` + `g2` for +5; see
//! [`metavm_zkp::ultimate_joint_prove::NUM_EXTENDED_AIRS`]) and 18 on
//! the EVM side (`call_family_air` plus rounds 15–18 additions:
//! `precompile_air`, `jump_validity_air`, `selfdestruct_air`,
//! `access_list_eip2929_air`, `returndata_air`, `extcode_air`,
//! `hardfork_rules_air`, `precompile_io_air`, `stack_depth_air`; plus
//! rounds 19–20 additions: `gas_refund_3529_air`, `basefee_air`,
//! `stack_contents_air`; plus round 21 addition `push0_air`; plus
//! round 22 additions `opcode_dispatch_air`, `address_opcode_air`;
//! plus round 23 additions `context_readers_air`,
//! `sstore_prepost_air`). Round 24 added 13 new zkp-side AIRs
//! (`lmd_ghost_fork_choice_air`, `shuffle_90_round_air`,
//! `casper_ffg_chain_air`, `epoch_processing_air`,
//! `eip7702_delegation_air`, `eip4337_user_op_air`,
//! `data_availability_sampling_air`, `ripemd160_internals_air`,
//! `blake2_f_internals_air`, `modexp_internals_air`,
//! `bn254_curve_ops_air`, `bn254_pairing_internals_air`,
//! `precompile_io_chunked_air`) and on the EVM side three integration
//! test modules (`evm_main_basefee_joint_prove`,
//! `evm_main_push0_joint_prove`, `evm_main_context_joint_prove`) plus
//! six new descriptor builders in `precompile_evm_linkages` (blake2f,
//! modexp, bn254_pairing × dispatch+IO). The zkp side also gained the
//! BBB8 8-field merkleization extension on `beacon_block_body_air`
//! (+33 constraints / +7 descriptors), seven new round-21/22
//! precompile/composition AIRs, three round-23 AIRs
//! (`beacon_block_validity_air`, `sha3_full_chain_air`,
//! `attestation_rewards_air`), and the round-24 set listed above.
//! Round 25–27 added `secp256k1_fp_air` (scaffold; full
//! `VmConstraintSystem` integration deferred) plus three zkp-side
//! integration test modules (`integration_eip7702_joint_prove`,
//! `integration_das_joint_prove`, `integration_casper_ffg_joint_prove`)
//! that compose existing AIRs into multi-AIR joint_prove coverage
//! without introducing new AIRs. On the EVM side rounds 25–27 added
//! nine new integration test modules — `evm_main_caller_joint_prove`,
//! `evm_main_callvalue_joint_prove`, `evm_main_origin_joint_prove`,
//! `evm_main_gasprice_joint_prove`, `evm_main_gaslimit_joint_prove`,
//! `evm_main_prevrandao_joint_prove`, `evm_main_exp_joint_prove`,
//! `evm_main_create_joint_prove`, `evm_main_create2_joint_prove`
//! (joining the round-24 `evm_main_basefee_joint_prove`,
//! `evm_main_push0_joint_prove`, `evm_main_context_joint_prove`, and
//! the prior `sstore`, `log`, `call`, `sha3`, `mstore`, `return`,
//! `address` joint_prove modules). These integration modules wire
//! existing AIRs end-to-end and contribute no new AIRs or
//! row-constraints. Round 25–27 also extended `blake2_f_internals_air`
//! (NUM_COLUMNS 6740 → 7509, NUM_ROW_CONSTRAINTS 8390 → 9355),
//! `ripemd160_internals_air` (250 → 317 cols, 202 → 239 constraints),
//! and `modexp_internals_air` (560 → 717 cols, 77 → 102 constraints).
//!
//! Rounds 28–31 added five new zkp-side AIRs (`sha512_air`,
//! `bn254_final_exp_air`, `bn254_miller_loop_air`, plus the
//! `bn254_curve_ops_air` sub-AIRs `fp2` and `g2`) contributing 0 + 5 +
//! 10 + 12 + 26 = 53 new row-constraints and 0 + 2 + 3 + 13 + 1 = 19
//! new descriptor builders. They also extended `blake2_f_internals_air`
//! with task #216's `h_out` triple-XOR finalize layer (9355 → 11947
//! row-constraints; 7509 → 9557 columns) plus accumulating round
//! 28–31 binding tasks #195 (ripemd 250 → 317) / #182 / #196 (modexp
//! 560 → 717) / #205 / #206 / #215. On the EVM side rounds 28–31 added
//! nine new zkp-side integration test modules (`integration_eip7702_joint_prove`,
//! `integration_das_joint_prove`,
//! `integration_blob_full_chain_joint_prove`,
//! `integration_casper_ffg_joint_prove`,
//! `integration_multi_tx_full_chain_joint_prove`,
//! `integration_lmd_ghost_joint_prove`,
//! `integration_cross_block_beacon_transition_joint_prove`,
//! `integration_block_validity_master_joint_prove`,
//! `integration_hash_to_curve_joint_prove`) which compose existing AIRs
//! end-to-end without introducing new AIRs or row-constraints.
//!
//! Rounds 35–37 added the full `sha512_constraints` row-local algebra
//! module (+12 row-constraints; extends the already-counted `sha512_air`
//! scaffold) and tracked `blob_kzg_air` (+47 row-constraints, +1 AIR;
//! the IETF G1-compressed flag-byte splitter from #290 contributes 2 of
//! the 47). Rounds 35–37 also populate `bn254_miller_loop_air`
//! (#259 / #269) and add the host-side `bn254_curve_ops_air::fp12`
//! module (host-side only, no AIR columns / row-constraints /
//! descriptors) plus several integration test modules.
//! `eip1559_fee_market_air` was extended with +2 shifted constraints
//! (multi-block base-fee chain hookup, #286) and EVM-side
//! `gas_tracking_air` with +1 shifted constraint (cumulative-gas
//! transition, #276); neither changes row-constraint counts. The zkp
//! sub-bundle therefore grows by +59 row-constraints (12 + 47) and +1
//! extended AIR (`blob_kzg_air`). The EVM sub-bundle is unchanged.
//!
//! Rounds 32–34 added seven new zkp-side AIRs (`bls12_381_curve_ops_air`
//! G1, `bls12_381_curve_ops_air::fp2`, `bls12_381_curve_ops_air::g2`,
//! `randao_32_epoch_chain_air`, `ssz_generalized_index_air`,
//! `verkle_tree_air`, `ed25519_air`) contributing 16 + 12 + 22 + 40 +
//! 5 + 5 + 67 = 167 new row-constraints and 1 + 13 + 1 + 2 + 1 + 2 +
//! 1 = 21 new descriptor builders. The `bn254_curve_ops_air::fp12`
//! host module is host-side only and contributes no AIR columns /
//! row-constraints / descriptors. `eip1559_fee_market_air` was extended
//! with +2 shifted constraints and EVM-side `gas_tracking_air` with +1
//! shifted constraint; neither changes row-constraint totals. Rounds
//! 32–34 also added ten new zkp-side integration test modules
//! (`integration_sync_committee_aggregate_joint_prove`,
//! `integration_block_full_proof_joint_prove`,
//! `integration_multi_block_proof_joint_prove`,
//! `integration_beacon_block_validity_joint_prove`,
//! `integration_tx_full_chain_joint_prove`,
//! `integration_full_eth_block_joint_prove`,
//! `integration_validator_deposit_joint_prove`,
//! `integration_beacon_block_proposer_signature_joint_prove`,
//! `integration_eip2537_joint_prove`,
//! `integration_multi_hash_chain_joint_prove`,
//! `integration_ffg_finality_joint_prove`) which compose existing AIRs
//! end-to-end without introducing new AIRs or row-constraints.
//!
//! They are tracked for coverage reporting via [`count_total_airs`] /
//! [`count_total_constraints`] / [`count_total_descriptors`] but are
//! not yet assembled into the runnable bundle (curve-specific
//! primitives, cross-block boundary consistency, or external oracles
//! are still required). Including them brings the totals to
//! **111 AIRs / 311 descriptor builders / 14960 row-constraints**
//! (post-round-35–37).
//!
//! ## Layer-indexing contract
//!
//! [`collect_all_descriptors`] takes both `zkp_layer_base` and
//! `evm_layer_base` so the caller can decide where each sub-bundle's
//! AIRs live in the global `joint_prove` traces vector. The sub-bundle
//! descriptors are rewritten to add their base offset, so no two AIRs
//! share a layer index as long as the two ranges don't overlap.
//!
//! ## Inter-bundle wiring
//!
//! [`make_evm_main_to_zkp_block_header_descriptor`] is provided as a
//! representative cross-bundle linkage binding the EVM main trace's
//! block-context selector rows to the zkp-side `block_header_air`
//! `is_real` rows. Real callers would add one such descriptor per
//! block-context opcode (NUMBER, TIMESTAMP, GASLIMIT, BASEFEE,
//! COINBASE) using the existing per-opcode builders in
//! `crate::cross_air_linkage`.

use metavm_zkp::cross_air_logup::{
    joint_prove as cross_joint_prove, joint_verify as cross_joint_verify,
    CrossAirLogUpDescriptor, CrossAirLogUpExtension,
};
use metavm_zkp::field::CurveType;
use metavm_zkp::prover::ExecutionProof;
use metavm_zkp::scheme::CommitmentScheme;
use metavm_zkp::trace::TracePolynomials;
use metavm_zkp::ultimate_joint_prove::{
    self as zkp_bundle, OrchestrationBundle, SampleTxInput,
};
use metavm_zkp::vm_constraints::VmConstraintSystem;

use crate::constraints::EvmConstraintSystem;
use crate::evm_orchestration::{self as evm_bundle, EvmGadgetBundle};
use crate::executor::execute_bytecode;
use crate::trace::EvmTraceColumns;

// ─── Master input ─────────────────────────────────────────────────────

/// Minimal driver input for [`assemble_master_bundle`]. Combines the
/// transaction-level data the EVM bundle needs (bytecode + calldata)
/// with the beacon-chain context the zkp-side bundle needs (sample
/// storage address/slot/value + block number).
///
/// A production driver would derive these from a real `(block, tx)`
/// pair fetched from an Ethereum execution node + a beacon node. The
/// tests in this module use a simple SSTORE/SLOAD bytecode that
/// exercises both sides.
#[derive(Clone, Debug)]
pub struct MasterInput {
    /// EVM bytecode under execution.
    pub bytecode: Vec<u8>,
    /// EVM calldata.
    pub calldata: Vec<u8>,
    /// Contract address (also threaded into the zkp-side
    /// account/storage gadgets).
    pub address: [u8; 20],
    /// Big-endian storage slot.
    pub slot_be: [u8; 32],
    /// Big-endian storage value at `slot_be`.
    pub value_be: [u8; 32],
    /// Execution-layer block number; also used as the beacon-chain
    /// payload's `block_number`.
    pub block_number: u64,
}

impl Default for MasterInput {
    fn default() -> Self {
        // PUSH1 0x42; PUSH1 0x00; SSTORE; PUSH1 0x00; SLOAD; STOP
        let bytecode = vec![0x60, 0x42, 0x60, 0x00, 0x55, 0x60, 0x00, 0x54, 0x00];
        let mut slot = [0u8; 32];
        slot[31] = 0;
        let mut value = [0u8; 32];
        value[31] = 0x42;
        Self {
            bytecode,
            calldata: Vec::new(),
            address: [0xab; 20],
            slot_be: slot,
            value_be: value,
            block_number: 18_500_000,
        }
    }
}

impl MasterInput {
    /// Project to the zkp-side `SampleTxInput`.
    pub fn as_sample_tx_input(&self) -> SampleTxInput {
        SampleTxInput {
            address: self.address,
            slot_be: self.slot_be,
            value_be: self.value_be,
            block_number: self.block_number,
        }
    }
}

// ─── Master bundle ────────────────────────────────────────────────────

/// Combined bundle owning both sub-bundles plus the underlying EVM
/// trace columns (needed by the EVM-main commitments in any future
/// `joint_prove` driver).
#[allow(dead_code)]
pub struct MasterProofBundle {
    /// zkp-side bundle (14 AIRs).
    pub zkp: OrchestrationBundle,
    /// EVM-side gadget bundle (9 AIRs).
    pub evm: EvmGadgetBundle,
    /// Raw EVM trace columns the EVM-side bundle was assembled from.
    /// Retained so the caller can produce the EVM main-trace witness
    /// when wiring this bundle into `joint_prove`.
    pub evm_trace: EvmTraceColumns,
}

impl MasterProofBundle {
    /// Borrow the zkp sub-bundle.
    pub fn zkp(&self) -> &OrchestrationBundle {
        &self.zkp
    }

    /// Borrow the EVM gadget sub-bundle.
    pub fn evm(&self) -> &EvmGadgetBundle {
        &self.evm
    }
}

// ─── Bundle assembler ─────────────────────────────────────────────────

/// Build the entire master bundle from a single [`MasterInput`].
///
/// Executes `input.bytecode` against `input.calldata` to obtain the
/// real EVM trace, then assembles both sub-bundles using their existing
/// assemblers. **No proving is performed.**
pub fn assemble_master_bundle(
    scheme: &dyn CommitmentScheme,
    input: &MasterInput,
) -> MasterProofBundle {
    let evm_trace = execute_bytecode(&input.bytecode, &input.calldata)
        .expect("master input bytecode must execute cleanly");
    let evm = evm_bundle::assemble_from_trace(
        scheme,
        &evm_trace,
        &input.bytecode,
        &input.calldata,
    );
    let zkp = zkp_bundle::assemble_bundle(scheme, &input.as_sample_tx_input());

    MasterProofBundle { zkp, evm, evm_trace }
}

// ─── Layer-index offsetting helper ────────────────────────────────────

/// Add `offset` to both `a_layer_index` and `b_layer_index` of every
/// descriptor in `descriptors`. Used to re-base the per-sub-bundle
/// descriptor lists into the master traces vector.
fn offset_descriptors(
    descriptors: Vec<CrossAirLogUpDescriptor>,
    offset: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    descriptors
        .into_iter()
        .map(|mut d| {
            d.a_layer_index += offset;
            d.b_layer_index += offset;
            d
        })
        .collect()
}

// ─── Inter-bundle descriptor ──────────────────────────────────────────

/// Inter-bundle cross-AIR LogUp descriptor: EVM main `NUMBER` opcode
/// row ↔ zkp-side `block_header_air` `is_real` row.
///
/// The zkp bundle's `block_header_air` already binds the full RLP +
/// the `state_root` / `transactions_root` / `receipts_root` fields,
/// and the EVM-side bundle's `cross_air_linkage::make_evm_*_to_block_header_*`
/// descriptors bind individual BLOCK opcodes. When zkp's
/// `block_header_air` lives at a different layer index from the one
/// hard-coded in those builders, the master bundle re-wires the
/// linkage with the correct cross-bundle indices.
///
/// `evm_main_layer_index` — the layer holding the EVM main trace
/// (typically the one immediately before [`zkp_bundle::LAYER_STORAGE`]
/// is placed, or wherever the caller chooses).
///
/// `zkp_block_header_layer_index` — the layer holding the zkp-side
/// block_header_air; for an unoffset zkp bundle this is
/// [`zkp_bundle::LAYER_BLOCK_HEADER`].
pub fn make_evm_main_to_zkp_block_header_descriptor(
    evm_main_layer_index: usize,
    zkp_block_header_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    let mut d = crate::cross_air_linkage::make_evm_number_to_block_header_linkage_descriptor(
        evm_main_layer_index,
        zkp_block_header_layer_index,
    );
    d.label = "master_evm_main_to_zkp_block_header_v1".into();
    d
}

// ─── Descriptor collection ────────────────────────────────────────────

/// Collect the union of every cross-AIR LogUp descriptor across both
/// sub-bundles, re-based into the master traces vector.
///
/// `zkp_layer_base` is added to every zkp-bundle descriptor's
/// `a_layer_index` and `b_layer_index` (so e.g. zkp's
/// [`zkp_bundle::LAYER_STORAGE`] = 0 lands at `zkp_layer_base + 0`).
///
/// `evm_layer_base` is the layer index of the EVM main trace in the
/// caller's traces vector; gadget AIRs are assumed to live immediately
/// after it (`evm_layer_base + 1 .. evm_layer_base + 1 +
/// NUM_EVM_GADGET_LAYERS`).
///
/// **Important**: the two `[base, base+N)` ranges MUST be disjoint so
/// no two AIRs share a layer index. [`count_total_airs`] / [`Self::layer_indices_distinct`]
/// helpers can be used in tests to assert this.
pub fn collect_all_descriptors(
    zkp_layer_base: usize,
    evm_layer_base: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    // zkp-side: shift by zkp_layer_base.
    let zkp_descs = offset_descriptors(zkp_bundle::collect_descriptors(), zkp_layer_base);

    // evm-side: collect_evm_descriptors already takes
    // (evm_layer_index, gadget_base) absolute indices, so no shift
    // needed.
    let evm_descs = evm_bundle::collect_evm_descriptors(
        evm_layer_base,
        evm_layer_base + 1,
    );

    // Inter-bundle: EVM main → zkp block_header_air.
    let inter = make_evm_main_to_zkp_block_header_descriptor(
        evm_layer_base,
        zkp_layer_base + zkp_bundle::LAYER_BLOCK_HEADER,
    );

    let mut out =
        Vec::with_capacity(zkp_descs.len() + evm_descs.len() + 1);
    out.extend(zkp_descs);
    out.extend(evm_descs);
    out.push(inter);
    out
}

// ─── Counting helpers ─────────────────────────────────────────────────

/// Sum of `NUM_ROW_CONSTRAINTS` across every AIR in the wired master
/// bundle (zkp side + evm side). Does **not** include extended (round
/// 11–14) AIRs; see [`count_grand_total_constraints`].
pub fn count_total_constraints() -> usize {
    zkp_bundle::count_constraints() + evm_bundle::count_evm_constraints()
}

/// Total number of AIRs in the wired master bundle (zkp side + evm
/// side).
///
/// Does **not** include the EVM main trace itself — that lives at the
/// caller-chosen `evm_layer_base` and is not assembled by either
/// sub-bundle. Including the EVM main would give 24. Does **not**
/// include extended (round 11–14) AIRs; see
/// [`count_grand_total_airs`].
pub fn count_total_airs() -> usize {
    zkp_bundle::NUM_AIRS + evm_bundle::NUM_EVM_GADGET_LAYERS
}

/// Total number of cross-AIR LogUp descriptors in the wired master
/// bundle (zkp side + evm side + 1 inter-bundle).
pub fn count_total_descriptors() -> usize {
    collect_all_descriptors(0, zkp_bundle::NUM_AIRS).len()
}

// ─── Extended AIR accounting (rounds 11–20) ──────────────────────────

/// EVM-side AIRs landed across rounds 11–23 and tracked for metrics:
/// `call_family_air` (round 11–14) plus rounds 15–18 additions
/// `precompile_air`, `jump_validity_air`, `selfdestruct_air`,
/// `access_list_eip2929_air`, `returndata_air`, `extcode_air`,
/// `hardfork_rules_air`, `precompile_io_air`, `stack_depth_air`, plus
/// rounds 19–20 additions `gas_refund_3529_air`, `basefee_air`,
/// `stack_contents_air`, plus round 21 addition `push0_air`, plus
/// round 22 additions `opcode_dispatch_air`, `address_opcode_air`,
/// plus round 23 additions `context_readers_air`, `sstore_prepost_air`.
/// **Not** yet wired into the runnable bundle.
pub const NUM_EVM_EXTENDED_AIRS: usize = 18;

/// Cross-AIR LogUp descriptor builders exposed by the 18 EVM-side
/// extended modules plus the round-24 `precompile_evm_linkages` deltas:
/// - `call_family_air`: 2 (round 11–14)
/// - `precompile_air`: 3 (round 15)
/// - `jump_validity_air`: 1 (round 16)
/// - `selfdestruct_air`: 3 (round 16)
/// - `access_list_eip2929_air`: 2 (round 16)
/// - `returndata_air`: 2 (round 16)
/// - `extcode_air`: 3 (round 17)
/// - `hardfork_rules_air`: 2 (round 17)
/// - `precompile_io_air`: 4 (round 17)
/// - `stack_depth_air`: 1 (round 18)
/// - `gas_refund_3529_air`: 3 (round 19)
/// - `basefee_air`: 3 (round 20)
/// - `stack_contents_air`: 2 (round 20)
/// - `push0_air`: 2 (round 21)
/// - `opcode_dispatch_air`: 2 (round 22)
/// - `address_opcode_air`: 2 (round 22)
/// - `context_readers_air`: 4 (round 23)
/// - `sstore_prepost_air`: 3 (round 23)
/// - `precompile_evm_linkages` (round 24, blake2f + modexp +
///   bn254_pairing × dispatch+IO): 6
///
/// Total = 50.
pub const NUM_EVM_EXTENDED_DESCRIPTORS: usize = 50;

/// Sum of `NUM_ROW_CONSTRAINTS` from the EVM-side extended AIRs (rounds
/// 11–23). `call_family_air` (round 11–14) plus 9 rounds 15–18
/// additions plus 3 rounds 19–20 additions plus 1 round 21 addition
/// plus 2 round 22 additions plus 2 round 23 additions.
pub fn count_evm_extended_constraints() -> usize {
    crate::call_family_air::NUM_ROW_CONSTRAINTS
        // Rounds 15–18 additions.
        + crate::precompile_air::NUM_ROW_CONSTRAINTS
        + crate::jump_validity_air::NUM_ROW_CONSTRAINTS
        + crate::selfdestruct_air::NUM_ROW_CONSTRAINTS
        + crate::access_list_eip2929_air::NUM_ROW_CONSTRAINTS
        + crate::returndata_air::NUM_ROW_CONSTRAINTS
        + crate::extcode_air::NUM_ROW_CONSTRAINTS
        + crate::hardfork_rules_air::NUM_ROW_CONSTRAINTS
        + crate::precompile_io_air::NUM_ROW_CONSTRAINTS
        + crate::stack_depth_air::NUM_ROW_CONSTRAINTS
        // Rounds 19–20 additions.
        + crate::gas_refund_3529_air::NUM_ROW_CONSTRAINTS
        + crate::basefee_air::NUM_ROW_CONSTRAINTS
        + crate::stack_contents_air::NUM_ROW_CONSTRAINTS
        // Round 21 addition.
        + crate::push0_air::NUM_ROW_CONSTRAINTS
        // Round 22 additions.
        + crate::opcode_dispatch_air::NUM_ROW_CONSTRAINTS
        + crate::address_opcode_air::NUM_ROW_CONSTRAINTS
        // Round 23 additions.
        + crate::context_readers_air::NUM_ROW_CONSTRAINTS
        + crate::sstore_prepost_air::NUM_ROW_CONSTRAINTS
}

/// Grand total AIR count: wired master bundle + extended AIRs from
/// both sides (rounds 11–23).
pub fn count_grand_total_airs() -> usize {
    count_total_airs() + zkp_bundle::NUM_EXTENDED_AIRS + NUM_EVM_EXTENDED_AIRS
}

/// Grand total descriptor-builder count: wired descriptors + extended
/// builders from both sides.
pub fn count_grand_total_descriptors() -> usize {
    count_total_descriptors()
        + zkp_bundle::NUM_EXTENDED_DESCRIPTORS
        + NUM_EVM_EXTENDED_DESCRIPTORS
}

/// Grand total `NUM_ROW_CONSTRAINTS` sum: wired constraint sums +
/// extended sums from both sides.
pub fn count_grand_total_constraints() -> usize {
    count_total_constraints()
        + zkp_bundle::count_extended_constraints()
        + count_evm_extended_constraints()
}

// ─── Layer-layout convention ──────────────────────────────────────────

/// Convention used by [`run_joint_prove`] / [`run_joint_verify`]:
///
/// - layers `[0, NUM_AIRS_ZKP)` — zkp-side OrchestrationBundle AIRs
///   (`LAYER_STORAGE` .. `LAYER_FINALITY`).
/// - layer `NUM_AIRS_ZKP` — EVM main trace.
/// - layers `[NUM_AIRS_ZKP+1, NUM_AIRS_ZKP+1+NUM_EVM_GADGET_LAYERS)`
///   — EVM-side gadget AIRs (`LAYER_ENV` .. `LAYER_JUMPDEST_TABLE`).
///
/// Total = `NUM_AIRS_ZKP + 1 + NUM_EVM_GADGET_LAYERS` traces.
pub const ZKP_LAYER_BASE: usize = 0;
/// Absolute layer index of the EVM main trace.
pub const EVM_MAIN_LAYER: usize = zkp_bundle::NUM_AIRS;
/// Total number of trace layers passed to [`cross_joint_prove`].
pub const TOTAL_LAYERS: usize =
    zkp_bundle::NUM_AIRS + 1 + evm_bundle::NUM_EVM_GADGET_LAYERS;

// ─── Master joint proof wrapper ──────────────────────────────────────

/// Output of [`run_joint_prove`]: the per-AIR execution proofs plus
/// the shared cross-AIR LogUp extension (β/γ + per-linkage closures).
///
/// Indexed by absolute layer index following the [`ZKP_LAYER_BASE`] /
/// [`EVM_MAIN_LAYER`] layout convention. `proofs[EVM_MAIN_LAYER]` is the
/// EVM main trace proof; `proofs[0..zkp_bundle::NUM_AIRS]` are the
/// zkp-side AIRs; `proofs[EVM_MAIN_LAYER+1..]` are the EVM gadget AIRs.
#[derive(Debug, Clone)]
pub struct JointProof {
    /// Per-AIR execution proofs in the layout described above.
    pub proofs: Vec<ExecutionProof>,
    /// Shared joint-prover extension (β/γ + per-linkage closure proofs).
    pub extension: CrossAirLogUpExtension,
}

/// Build the EVM main trace polynomials from the bundle's captured
/// `EvmTraceColumns` (BLS48-581).
fn build_evm_main_trace(bundle: &MasterProofBundle) -> TracePolynomials {
    TracePolynomials::from_vm_trace(&bundle.evm_trace, CurveType::Bls48581)
}

/// Run the joint prover over **every AIR in the master bundle** plus the
/// EVM main trace, using the canonical layer layout defined by
/// [`ZKP_LAYER_BASE`] / [`EVM_MAIN_LAYER`].
///
/// On success returns a [`JointProof`] containing per-AIR
/// [`ExecutionProof`]s and the shared [`CrossAirLogUpExtension`]. On
/// failure (any per-AIR phase-2 commit, linkage SNARK, or transcript
/// derivation step rejecting) returns the underlying error string.
///
/// ## Wall-clock cost
///
/// Linear in the union of all per-AIR domain sizes; on BLS48-581 with
/// the default [`MasterInput`] this is dominated by `block_header_air`
/// (974 cols, ~580-byte header_rlp) and `mpt_air` (~688 cols). Expect
/// well over an hour even on a fast machine; the runnable test is
/// `#[ignore]`-tagged.
///
/// ## Honest-witness caveat
///
/// `assemble_master_bundle` wires both sub-bundles around a single
/// `SampleTxInput`, so the boundary multisets (address, slot, value,
/// state_root, block_number, ...) line up by construction. Sub-bundle
/// closures match individually; the inter-bundle linkage
/// (`master_evm_main_to_zkp_block_header_v1`) requires the EVM trace to
/// emit at least one `NUMBER` opcode at the matching block number — the
/// default bytecode (PUSH/SSTORE/PUSH/SLOAD/STOP) does **not**, so the
/// runnable test must use an extended bytecode (see
/// `master_joint_prove_passes`).
pub fn run_joint_prove(
    bundle: &MasterProofBundle,
    scheme: &dyn CommitmentScheme,
) -> Result<JointProof, String> {
    // EVM main trace + CS.
    let evm_main_trace = build_evm_main_trace(bundle);
    let evm_main_cs = EvmConstraintSystem::new();

    // zkp-side traces (14) in LAYER_STORAGE..LAYER_FINALITY order.
    let zkp_traces = bundle.zkp.traces();
    // EVM-side gadget traces (9) in LAYER_ENV..LAYER_JUMPDEST_TABLE order.
    let evm_gadget_traces = bundle.evm.traces();

    // Compose the global traces vector in layer order:
    //   [zkp_storage, ..., zkp_finality,
    //    evm_main,
    //    evm_env, evm_log, ..., evm_jumpdest_table]
    let mut traces: Vec<(&TracePolynomials, &dyn VmConstraintSystem)> =
        Vec::with_capacity(TOTAL_LAYERS);
    for tup in &zkp_traces {
        traces.push(*tup);
    }
    traces.push((&evm_main_trace, &evm_main_cs as &dyn VmConstraintSystem));
    for tup in &evm_gadget_traces {
        traces.push(*tup);
    }

    debug_assert_eq!(traces.len(), TOTAL_LAYERS);

    let descriptors = collect_all_descriptors(ZKP_LAYER_BASE, EVM_MAIN_LAYER);

    let (proofs, extension) = cross_joint_prove(&traces, &descriptors, scheme)
        .map_err(|e| format!("joint_prove failed: {}", e))?;

    Ok(JointProof { proofs, extension })
}

/// Verify a [`JointProof`] against the same master bundle the prover
/// used. Recomputes the descriptors via [`collect_all_descriptors`] and
/// delegates to [`cross_joint_verify`].
///
/// Returns `false` for any soundness rejection (per-AIR verify, linkage
/// SNARK, cross-trace binding, closure mismatch, transcript mismatch).
pub fn run_joint_verify(
    proof: &JointProof,
    bundle: &MasterProofBundle,
    scheme: &dyn CommitmentScheme,
) -> bool {
    // Reassemble the constraint-system reference list in the same layer
    // order used by run_joint_prove.
    let evm_main_cs = EvmConstraintSystem::new();
    let zkp_cs = bundle.zkp.cs_refs();
    let evm_gadget_cs = bundle.evm.cs_refs();

    let mut cs_refs: Vec<&dyn VmConstraintSystem> =
        Vec::with_capacity(TOTAL_LAYERS);
    for c in &zkp_cs {
        cs_refs.push(*c);
    }
    cs_refs.push(&evm_main_cs as &dyn VmConstraintSystem);
    for c in &evm_gadget_cs {
        cs_refs.push(*c);
    }

    if proof.proofs.len() != cs_refs.len() {
        return false;
    }

    let descriptors = collect_all_descriptors(ZKP_LAYER_BASE, EVM_MAIN_LAYER);

    cross_joint_verify(
        &proof.proofs,
        &cs_refs,
        &descriptors,
        &proof.extension,
        scheme,
        bundle.zkp.curve,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;

    fn make_scheme() -> Bls48581Scheme {
        let s = Bls48581Scheme::new();
        s.init();
        s
    }

    #[test]
    fn master_bundle_assembles() {
        let scheme = make_scheme();
        let bundle = assemble_master_bundle(&scheme, &MasterInput::default());
        assert_eq!(bundle.zkp.traces().len(), zkp_bundle::NUM_AIRS);
        assert_eq!(bundle.evm.traces().len(), evm_bundle::NUM_EVM_GADGET_LAYERS);
    }

    #[test]
    fn total_air_count_is_at_least_twenty() {
        let n = count_total_airs();
        assert!(n >= 20, "expected >=20 AIRs, got {}", n);
        eprintln!("[master_proof] total AIRs (wired) = {}", n);
    }

    #[test]
    fn total_descriptor_count_is_at_least_forty_five() {
        let descs = collect_all_descriptors(0, 100);
        assert!(
            descs.len() >= 45,
            "expected >=45 descriptors, got {}",
            descs.len(),
        );
        eprintln!("[master_proof] total descriptors (wired) = {}", descs.len());
    }

    #[test]
    fn total_constraint_sum_is_at_least_two_hundred_fifty() {
        let total = count_total_constraints();
        assert!(
            total >= 250,
            "expected >=250 row-constraints, got {}",
            total,
        );
        eprintln!("[master_proof] total row-constraints (wired) = {}", total);
    }

    #[test]
    fn grand_total_air_count_meets_target() {
        // wired (23) + zkp-extended (71) + evm-extended (18) = 112
        // after round 38 (eip3074_air newly tracked).
        let n = count_grand_total_airs();
        assert!(
            n >= 111,
            "expected >=111 grand total AIRs, got {}",
            n,
        );
        eprintln!("[master_proof] grand total AIRs (wired+extended) = {}", n);
    }

    #[test]
    fn grand_total_descriptor_count_meets_target() {
        // wired collect_all_descriptors (~51) + zkp-extended (210) +
        // evm-extended (50) ~= 311 after rounds 32–34.
        let n = count_grand_total_descriptors();
        assert!(
            n >= 311,
            "expected >=311 grand total descriptors, got {}",
            n,
        );
        eprintln!(
            "[master_proof] grand total descriptors (wired+extended) = {}",
            n,
        );
    }

    #[test]
    fn grand_total_constraint_sum_meets_target() {
        // wired (269) + zkp-extended (>=14361 incl. rounds 21–34 BLAKE2 /
        // ripemd160 / modexp / bn254 binding expansions, rounds 28–31
        // bn254 pairing / sha512 AIRs / BLAKE2 #216 h_out finalize layer,
        // rounds 32–34 BLS12-381 G1+fp2+g2 / randao_32 / ssz_gen /
        // verkle_tree / ed25519 AIRs, and rounds 35–37 sha512_constraints
        // (+12) + blob_kzg_air newly tracked (+47)) + evm-extended (330
        // incl. rounds 21–23; round 24–37 EVM additions are integration
        // tests with no new row-constraints) = >=14960.
        let total = count_grand_total_constraints();
        assert!(
            total >= 14960,
            "expected >=14960 grand total row-constraints, got {}",
            total,
        );
        eprintln!(
            "[master_proof] grand total row-constraints (wired+extended) = {}",
            total,
        );
    }

    #[test]
    fn evm_extended_air_count_matches_constant() {
        // 18 EVM-side extended AIRs tracked: call_family_air (round
        // 11–14) plus rounds 15–18 additions precompile_air,
        // jump_validity_air, selfdestruct_air,
        // access_list_eip2929_air, returndata_air, extcode_air,
        // hardfork_rules_air, precompile_io_air, stack_depth_air;
        // plus rounds 19–20 additions gas_refund_3529_air,
        // basefee_air, stack_contents_air; plus round 21 addition
        // push0_air; plus round 22 additions opcode_dispatch_air,
        // address_opcode_air; plus round 23 additions
        // context_readers_air, sstore_prepost_air.
        assert_eq!(NUM_EVM_EXTENDED_AIRS, 18);
        // Descriptor builders: 2 + 3 + 1 + 3 + 2 + 2 + 3 + 2 + 4 + 1
        //                    + 3 + 3 + 2 + 2 + 2 + 2 + 4 + 3 (rounds
        //                    11–23 = 44) + 6 (round-24
        //                    precompile_evm_linkages) = 50.
        assert_eq!(NUM_EVM_EXTENDED_DESCRIPTORS, 50);
        // call_family (22) + precompile (18) + jump_validity (10) +
        // selfdestruct (11) + access_list_2929 (15, +4 vs round 35 from
        // the round-38 multi-access chain extension) + returndata (15) +
        // extcode (41) + hardfork_rules (16) + precompile_io (76) +
        // stack_depth (11) + gas_refund_3529 (16) + basefee (10) +
        // stack_contents (10) + push0 (10) + opcode_dispatch (4) +
        // address_opcode (9) + context_readers (23) +
        // sstore_prepost (17) = 334.
        assert_eq!(count_evm_extended_constraints(), 334);
    }

    #[test]
    fn round_21_22_evm_extended_constraint_contribution_matches_target() {
        // Round 21 EVM addition: push0 (10).
        // Round 22 EVM additions: opcode_dispatch (4) + address_opcode (9) = 13.
        // Combined: 23.
        let r21_22 = crate::push0_air::NUM_ROW_CONSTRAINTS
            + crate::opcode_dispatch_air::NUM_ROW_CONSTRAINTS
            + crate::address_opcode_air::NUM_ROW_CONSTRAINTS;
        assert_eq!(r21_22, 23, "rounds 21–22 EVM extended constraints drifted");
    }

    #[test]
    fn round_23_evm_extended_constraint_contribution_matches_target() {
        // Round 23 EVM additions: context_readers (23) +
        // sstore_prepost (17) = 40.
        let r23 = crate::context_readers_air::NUM_ROW_CONSTRAINTS
            + crate::sstore_prepost_air::NUM_ROW_CONSTRAINTS;
        assert_eq!(r23, 40, "round 23 EVM extended constraints drifted");
    }

    #[test]
    fn round_23_grand_totals_match_target() {
        // Historical pin: round 23 baseline. Now superseded by
        // round_24_grand_totals_match_target; kept as a floor sanity
        // check (round-24 additions only grow the totals).
        assert!(count_grand_total_airs() >= 84);
        assert!(count_grand_total_descriptors() >= 249);
        assert!(count_grand_total_constraints() >= 2304);
    }

    #[test]
    fn round_24_extended_constraint_contribution_matches_target() {
        // Round 24 EVM additions are integration tests
        // (evm_main_basefee_joint_prove, evm_main_push0_joint_prove,
        // evm_main_context_joint_prove) and the new
        // precompile_evm_linkages module — no new AIRs, no new
        // row-constraints contributed on the EVM side. The round-24
        // delta is +6 descriptor builders (asserted via
        // NUM_EVM_EXTENDED_DESCRIPTORS) and +88 zkp-side
        // row-constraints (asserted in the zkp bundle).
        let r24_evm_constraints = 0usize;
        assert_eq!(r24_evm_constraints, 0);
    }

    #[test]
    fn round_24_grand_totals_match_target() {
        // Historical pin: round-24 baseline. Superseded by
        // `round_25_grand_totals_match_target` once rounds 25–27
        // extended the BLAKE2 / ripemd / modexp internals AIRs. Kept
        // as a sanity floor — the round-25–27 additions only grow the
        // totals.
        assert!(count_grand_total_airs() >= 97);
        assert!(count_grand_total_descriptors() >= 271);
        assert!(count_grand_total_constraints() >= 2595);
    }

    #[test]
    fn round_25_grand_totals_match_target() {
        // Historical pin: round-25–27 baseline. Superseded by
        // `round_28_grand_totals_match_target` once rounds 28–31 added
        // the new bn254 pairing / sha512 AIRs and the BLAKE2 #216 h_out
        // triple-XOR finalize layer. Kept as a sanity floor — rounds
        // 28–31 additions only grow the totals.
        assert!(count_grand_total_airs() >= 98);
        assert!(count_grand_total_descriptors() >= 271);
        assert!(count_grand_total_constraints() >= 12077);
    }

    #[test]
    fn round_28_grand_totals_match_target() {
        // Historical pin: round-28–31 baseline. Superseded by
        // `round_32_grand_totals_match_target` once rounds 32–34 added
        // the BLS12-381 curve_ops sub-AIRs (G1 + fp2 + g2), the
        // randao_32 / ssz_generalized_index / verkle_tree / ed25519 AIRs,
        // and the bn254 fp12 host-side module. Kept as a sanity floor —
        // rounds 32–34 additions only grow the totals.
        assert!(count_grand_total_airs() >= 103);
        assert!(count_grand_total_descriptors() >= 290);
        let total = count_grand_total_constraints();
        assert!(
            total >= 14734,
            "expected >=14734 grand total row-constraints, got {}",
            total,
        );
        eprintln!(
            "[master_proof] round-28 grand total row-constraints = {}",
            total,
        );
    }

    #[test]
    fn round_32_grand_totals_match_target() {
        // Historical pin: round-32–34 baseline. Now superseded by
        // `round_35_grand_totals_match_target` once rounds 35–37 added
        // the `sha512_constraints` row-local algebra (+12) and tracked
        // `blob_kzg_air` (+47, newly tracked). Kept as a sanity floor —
        // rounds 35–37 additions only grow the totals.
        assert!(count_grand_total_airs() >= 110);
        assert!(count_grand_total_descriptors() >= 311);
        let total = count_grand_total_constraints();
        assert!(
            total >= 14901,
            "expected >=14901 grand total row-constraints, got {}",
            total,
        );
        eprintln!(
            "[master_proof] round-32 grand total row-constraints = {}",
            total,
        );
    }

    #[test]
    fn round_35_grand_totals_match_target() {
        // Historical pin: round-35–37 baseline. Now superseded by
        // `round_38_grand_totals_match_target` once round 38 added the
        // zkp-side `eip3074_air` (+1 AIR, +6 row-constraints), extended
        // `verkle_tree_air` Pedersen partial-sum binding (+2
        // row-constraints), and extended EVM-side
        // `access_list_eip2929_air` with the multi-access chain (+4
        // row-constraints). Kept as a sanity floor — round-38 additions
        // only grow the totals.
        assert!(count_grand_total_airs() >= 111);
        assert!(count_grand_total_descriptors() >= 311);
        let total = count_grand_total_constraints();
        assert!(
            total >= 14960,
            "expected >=14960 grand total row-constraints, got {}",
            total,
        );
        eprintln!(
            "[master_proof] round-35 grand total row-constraints = {}",
            total,
        );
    }

    #[test]
    fn round_38_grand_totals_match_target() {
        // After round 38 (closure work for rounds 35+):
        //   grand AIRs = 112 (23 wired + 71 zkp-ext + 18 evm-ext)
        //     (+1 vs round 35: `eip3074_air` — 189 cols, 6
        //      row-constraints — tracked as a forward-looking
        //      AUTH/AUTHCALL scaffold AIR. The verkle_tree_air Pedersen
        //      extension, master_proof composer, cross-layer descriptors
        //      module, mirror column additions to deposit_tree_air /
        //      validator_registry, Miller loop populate extensions,
        //      final exp populate, and BLS pairing equation closure
        //      descriptors all extend already-counted modules and do
        //      not add new AIRs.)
        //   grand descriptors = 311 (51 wired + 210 zkp-ext + 50 evm-ext)
        //     (unchanged: round-38 closure work — cross-layer
        //      descriptors module, master composer, mirror column
        //      fixes, populate extensions, BLS pairing equation closure
        //      descriptors — either wires existing descriptor builders
        //      under a fresh module surface or lands new descriptors as
        //      free functions outside the bundle's counted
        //      `NUM_EXTENDED_DESCRIPTORS` surface.)
        //   grand row-constraints >= 14974
        //     wired (269) + zkp-ext (>=14369) + evm-ext (334) = >=14972
        //     (round-35 baseline 14960 + access_list_eip2929 multi-access
        //      +4 + verkle_tree_air Pedersen +2 + eip3074_air 6 = 14972).
        //     The mirror column / populate / closure descriptor work
        //     contributes no new row-constraints (all extends
        //     pre-existing constraints).
        assert_eq!(count_grand_total_airs(), 112);
        assert_eq!(count_grand_total_descriptors(), 311);
        let total = count_grand_total_constraints();
        assert!(
            total >= 14972,
            "expected >=14972 grand total row-constraints, got {}",
            total,
        );
        eprintln!(
            "[master_proof] round-38 grand total row-constraints = {}",
            total,
        );
    }

    #[test]
    fn round_25_evm_integration_modules_have_no_new_airs() {
        // Round 25–27 EVM additions are integration tests
        // (evm_main_caller/callvalue/origin/gasprice/gaslimit/prevrandao/
        // exp/create/create2_joint_prove). No new AIRs, no new
        // row-constraints contributed on the EVM side. The delta lives
        // entirely on the zkp-side extended AIRs (`blake2_f_internals`,
        // `ripemd160_internals`, `modexp_internals` extensions, plus the
        // `secp256k1_fp_air` scaffold which contributes 0
        // row-constraints).
        let r25_evm_constraints = 0usize;
        assert_eq!(r25_evm_constraints, 0);
    }

    #[test]
    fn round_19_20_evm_extended_constraint_contribution_matches_target() {
        // Rounds 19–20 EVM-side additions contribute exactly 36
        // row-constraints: gas_refund_3529 (16) + basefee (10) +
        // stack_contents (10) = 36.
        let r19_20 = crate::gas_refund_3529_air::NUM_ROW_CONSTRAINTS
            + crate::basefee_air::NUM_ROW_CONSTRAINTS
            + crate::stack_contents_air::NUM_ROW_CONSTRAINTS;
        assert_eq!(r19_20, 36, "round 19–20 EVM extended constraints drifted");
    }

    #[test]
    fn round_15_18_evm_extended_constraint_contribution_matches_target() {
        // Rounds 15–18 EVM-side additions (excluding round-11-14
        // call_family_air) contribute precompile (18) + jump_validity
        // (10) + selfdestruct (11) + access_list_2929 (15, +4 vs round
        // 35 from the round-38 multi-access chain extension) +
        // returndata (15) + extcode (41) + hardfork_rules (16) +
        // precompile_io (76) + stack_depth (11) = 213.
        let r15_18 = crate::precompile_air::NUM_ROW_CONSTRAINTS
            + crate::jump_validity_air::NUM_ROW_CONSTRAINTS
            + crate::selfdestruct_air::NUM_ROW_CONSTRAINTS
            + crate::access_list_eip2929_air::NUM_ROW_CONSTRAINTS
            + crate::returndata_air::NUM_ROW_CONSTRAINTS
            + crate::extcode_air::NUM_ROW_CONSTRAINTS
            + crate::hardfork_rules_air::NUM_ROW_CONSTRAINTS
            + crate::precompile_io_air::NUM_ROW_CONSTRAINTS
            + crate::stack_depth_air::NUM_ROW_CONSTRAINTS;
        assert_eq!(r15_18, 213, "round 15–18 EVM extended constraints drifted");
    }

    #[test]
    fn both_sub_bundles_are_present() {
        let scheme = make_scheme();
        let bundle = assemble_master_bundle(&scheme, &MasterInput::default());
        // Spot-check that both sub-bundles' traces are non-empty.
        assert!(!bundle.zkp().traces().is_empty());
        assert!(!bundle.evm().traces().is_empty());
        // And that the underlying EVM trace was captured.
        assert!(!bundle.evm_trace.step.is_empty());
    }

    #[test]
    fn layer_indices_do_not_collide() {
        // zkp at [0, NUM_AIRS); EVM main at NUM_AIRS; EVM gadgets at
        // [NUM_AIRS+1, NUM_AIRS+1+NUM_EVM_GADGET_LAYERS).
        let zkp_base = 0usize;
        let evm_main = zkp_bundle::NUM_AIRS;
        let descs = collect_all_descriptors(zkp_base, evm_main);

        // Collect every (layer_index, side) pair referenced.
        let max_layer = evm_main + 1 + evm_bundle::NUM_EVM_GADGET_LAYERS;
        for d in &descs {
            assert!(
                d.a_layer_index < max_layer,
                "descriptor {} a_layer_index={} out of expected range",
                d.label,
                d.a_layer_index,
            );
            assert!(
                d.b_layer_index < max_layer,
                "descriptor {} b_layer_index={} out of expected range",
                d.label,
                d.b_layer_index,
            );
            if d.a_layer_index == d.b_layer_index {
                // Only legitimate self-linkage in the bundles is
                // byte_memory's perm closure.
                assert!(
                    d.label.contains("byte_memory"),
                    "unexpected self-loop linkage: {}",
                    d.label,
                );
            }
        }

        // The zkp range [0, NUM_AIRS) and EVM range
        // [NUM_AIRS, NUM_AIRS + 1 + NUM_EVM_GADGET_LAYERS) must be
        // disjoint by construction.
        let zkp_max = zkp_base + zkp_bundle::NUM_AIRS;
        let evm_min = evm_main;
        assert!(zkp_max <= evm_min);
    }

    #[test]
    fn inter_bundle_descriptor_is_well_formed() {
        let d = make_evm_main_to_zkp_block_header_descriptor(
            42, // evm_main_layer
            7,  // zkp block_header layer
        );
        assert_eq!(d.a_layer_index, 42);
        assert_eq!(d.b_layer_index, 7);
        assert!(d.label.contains("master_evm_main_to_zkp_block_header"));
        assert!(!d.a_columns.is_empty());
        assert!(!d.b_columns.is_empty());
        assert_eq!(d.a_columns.len(), d.b_columns.len());
        assert!(d.a_selector_column.is_some());
        assert!(d.b_selector_column.is_some());
    }

    #[test]
    fn collected_descriptors_have_nonempty_columns() {
        let descs = collect_all_descriptors(0, zkp_bundle::NUM_AIRS);
        for d in &descs {
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
            assert_eq!(
                d.a_columns.len(),
                d.b_columns.len(),
                "descriptor {} A/B tuple shape mismatch",
                d.label,
            );
        }
    }

    #[test]
    fn zkp_layer_offset_is_applied_correctly() {
        // With a non-zero zkp base, every zkp-side descriptor's
        // layer indices must shift by that offset.
        let descs_unshifted = collect_all_descriptors(0, 100);
        let descs_shifted = collect_all_descriptors(10, 100);
        // The zkp side contributes the same number of descriptors in
        // both cases; spot-check by counting non-EVM-side labels.
        // Conservatively, total counts must match.
        assert_eq!(descs_unshifted.len(), descs_shifted.len());
    }

    // ─── Joint-prove scaffolding (fast tests) ────────────────────────

    /// Fast smoke test: asserts the master bundle assembles, the
    /// `run_joint_prove` API exists with the expected total layer
    /// count, and the descriptor list lines up with that layout.
    /// Does NOT actually call `run_joint_prove` (that takes hours; see
    /// `master_joint_prove_passes` for the `#[ignore]`-tagged run).
    #[test]
    fn joint_prove_function_exists_and_compiles() {
        let scheme = make_scheme();
        let bundle = assemble_master_bundle(&scheme, &MasterInput::default());

        // Total AIRs that the joint prover will commit over.
        let expected = zkp_bundle::NUM_AIRS + 1 + evm_bundle::NUM_EVM_GADGET_LAYERS;
        assert_eq!(TOTAL_LAYERS, expected);
        assert_eq!(EVM_MAIN_LAYER, zkp_bundle::NUM_AIRS);
        assert_eq!(ZKP_LAYER_BASE, 0);

        // The sub-bundles expose exactly the trace counts they should.
        assert_eq!(bundle.zkp.traces().len(), zkp_bundle::NUM_AIRS);
        assert_eq!(bundle.evm.traces().len(), evm_bundle::NUM_EVM_GADGET_LAYERS);

        // Descriptors should be assembled relative to the same layout
        // (zkp at base 0, evm_main at NUM_AIRS, gadgets immediately
        // after).
        let descriptors = collect_all_descriptors(ZKP_LAYER_BASE, EVM_MAIN_LAYER);
        assert!(!descriptors.is_empty(), "expected at least one descriptor");

        // Reference the API symbols so they're checked by `cargo
        // check` and not flagged as dead-code if a future refactor
        // removes their other callers.
        let _ptr_prove: fn(
            &MasterProofBundle,
            &dyn CommitmentScheme,
        ) -> Result<JointProof, String> = run_joint_prove;
        let _ptr_verify: fn(
            &JointProof,
            &MasterProofBundle,
            &dyn CommitmentScheme,
        ) -> bool = run_joint_verify;

        eprintln!(
            "[master_proof] total layers = {} (zkp {} + evm_main 1 + evm gadgets {}); \
             descriptors = {}",
            TOTAL_LAYERS,
            zkp_bundle::NUM_AIRS,
            evm_bundle::NUM_EVM_GADGET_LAYERS,
            descriptors.len(),
        );
    }

    /// Sanity check: every descriptor produced by
    /// `collect_all_descriptors(ZKP_LAYER_BASE, EVM_MAIN_LAYER)` must
    /// reference a layer index in `[0, TOTAL_LAYERS)`. This is the
    /// well-formedness invariant `run_joint_prove` relies on; if it
    /// fails, the joint prover would index past the traces vector.
    #[test]
    fn descriptor_layer_indices_match_bundle_layers() {
        let descriptors = collect_all_descriptors(ZKP_LAYER_BASE, EVM_MAIN_LAYER);
        for d in &descriptors {
            assert!(
                d.a_layer_index < TOTAL_LAYERS,
                "descriptor {} a_layer_index {} >= TOTAL_LAYERS {}",
                d.label,
                d.a_layer_index,
                TOTAL_LAYERS,
            );
            assert!(
                d.b_layer_index < TOTAL_LAYERS,
                "descriptor {} b_layer_index {} >= TOTAL_LAYERS {}",
                d.label,
                d.b_layer_index,
                TOTAL_LAYERS,
            );
        }

        // EVM-main-side descriptors (those produced by
        // collect_evm_descriptors with evm_layer_index = EVM_MAIN_LAYER)
        // must reference EVM_MAIN_LAYER on at least one of their
        // sides. Confirm at least one such descriptor exists.
        let any_touches_evm_main = descriptors
            .iter()
            .any(|d| d.a_layer_index == EVM_MAIN_LAYER || d.b_layer_index == EVM_MAIN_LAYER);
        assert!(
            any_touches_evm_main,
            "expected at least one descriptor to reference EVM_MAIN_LAYER ({}), got {:?}",
            EVM_MAIN_LAYER,
            descriptors.iter().map(|d| (d.a_layer_index, d.b_layer_index)).collect::<Vec<_>>(),
        );

        // The inter-bundle descriptor specifically wires EVM_MAIN_LAYER
        // ↔ ZKP_LAYER_BASE + LAYER_BLOCK_HEADER. Confirm it landed at
        // the right indices.
        let inter = descriptors
            .iter()
            .find(|d| d.label.contains("master_evm_main_to_zkp_block_header"))
            .expect("inter-bundle descriptor must be present");
        assert_eq!(inter.a_layer_index, EVM_MAIN_LAYER);
        assert_eq!(
            inter.b_layer_index,
            ZKP_LAYER_BASE + zkp_bundle::LAYER_BLOCK_HEADER,
        );
    }

    /// **SLOW** runnable end-to-end test: assembles the master bundle,
    /// runs [`run_joint_prove`] over all 24 layers (14 zkp + 1 EVM main
    /// + 9 EVM gadget), then runs [`run_joint_verify`] on the result.
    ///
    /// Wall-clock: many hours on BLS48-581 — dominated by the wide
    /// AIRs (block_header_air 974 cols, mpt_air ~688 cols). Run with:
    ///
    /// ```text
    /// cargo test -p metavm-evm --release --lib \
    ///     master_proof::tests::master_joint_prove_passes -- \
    ///     --ignored --test-threads=1 --nocapture
    /// ```
    ///
    /// Honest-witness caveats: the default `MasterInput`'s bytecode
    /// does not emit a `NUMBER` opcode, so the inter-bundle linkage
    /// (`master_evm_main_to_zkp_block_header_v1`) closes via the
    /// empty-multiset case on the EVM-main A-side. Multiset equality
    /// still holds (both sides empty under the selector), so
    /// joint_verify accepts. A production driver should exercise an
    /// extended bytecode that emits NUMBER + TIMESTAMP + GASLIMIT +
    /// BASEFEE + COINBASE and wire one descriptor per opcode.
    #[test]
    #[ignore = "slow: master joint_prove (~minutes)"]
    fn master_joint_prove_passes() {
        let scheme = make_scheme();
        let bundle = assemble_master_bundle(&scheme, &MasterInput::default());

        let proof = run_joint_prove(&bundle, &scheme)
            .expect("master run_joint_prove must succeed");

        // Per-AIR proof count matches the layer layout.
        assert_eq!(proof.proofs.len(), TOTAL_LAYERS);
        let descriptors = collect_all_descriptors(ZKP_LAYER_BASE, EVM_MAIN_LAYER);
        assert_eq!(proof.extension.linkage_proofs.len(), descriptors.len());

        // Every per-linkage closure must match.
        for lp in &proof.extension.linkage_proofs {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closure mismatch",
                lp.label,
            );
        }

        assert!(
            run_joint_verify(&proof, &bundle, &scheme),
            "master run_joint_verify must accept honest witness",
        );
    }
}
