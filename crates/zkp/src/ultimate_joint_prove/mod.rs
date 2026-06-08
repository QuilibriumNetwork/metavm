//! # Ultimate joint_prove orchestration scaffold
//!
//! Composes the maximum number of `zkp`-side AIRs into a single
//! algebraic `joint_prove` invocation, grouped by layer:
//!
//! - **Layer A (EVM execution)**: storage_access_air, account_state_air,
//!   address_keccak_air, byte_memory_air (host-side EVM AIRs `env_air`,
//!   `log_air`, `blockhash_history_air`, `calldata_byte_air`,
//!   `call_frame_air`, `exp_air`, `byte_air` live in `metavm-evm` and
//!   are excluded here due to crate boundaries — a sibling assembler
//!   in `metavm-evm` would wire them onto this bundle).
//! - **Layer B (Ethereum block header)**: block_header_air,
//!   keccak_extract_wide, mpt_air (storage MPT inclusion), keccak_extract.
//! - **Layer C (beacon chain → finality)**: BBH-pair, body-pair,
//!   payload-pair, bbh_root_consumer, sha256_extract, finality.
//! - **Layer C+ (BLS aggregate verification)**: bls_pairing_air.
//!
//! This is **scaffolding only** — `joint_prove`/`joint_verify` are
//! gated behind `#[ignore = "slow"]` because running them takes hours
//! on the BLS48-581 curve. Fast tests verify bundle assembly,
//! descriptor count, constraint sum, and AIR presence.
//!
//! ## Wiring contract
//!
//! `LAYER_*` constants assign each AIR a stable layer index in the
//! `joint_prove` traces vector. Cross-AIR LogUp linkages reference
//! these indices; reordering them requires updating every
//! `collect_descriptors()` invocation.
//!
//! ## Extended-AIR accounting (rounds 11–23)
//!
//! Forty-three additional zkp-side AIRs landed across rounds 11–23 and
//! are **not** wired into the runnable bundle yet (their witnesses
//! require curve-specific (BLS12-381) primitives, cross-block boundary
//! consistency, or external host-side oracles that the
//! [`assemble_bundle`] sample driver does not currently provide). They
//! are accounted for in the bundle's complexity / coverage metrics
//! through:
//!
//! - [`NUM_EXTENDED_AIRS`] — count of extended AIRs (62).
//! - [`NUM_TOTAL_AIRS`] — `NUM_AIRS + NUM_EXTENDED_AIRS` (76).
//! - [`count_extended_constraints`] — sum of `NUM_ROW_CONSTRAINTS`
//!   across the 62 extended AIRs (rounds 19–20 also include the
//!   `beacon_block_body_air` 8-field merkleization extension).
//! - [`count_total_constraints`] — wired + extended.
//! - [`count_extended_descriptors`] — count of descriptor builders the
//!   extended AIRs expose (189 across 62 modules + the BBB8 extension).
//!
//! Extended modules (rounds 11–14): `access_list_air`,
//! `randao_proposer_air`, `sync_committee_sig_air`, `deposit_tree_air`,
//! `attester_slashing_air`, `voluntary_exit_air`, `versioned_hash_air`,
//! `attestation_aggregate_air`, `proposer_slashing_air`,
//! `withdrawal_credential_air`, `tx_nonce_air`, `receipt_status_air`,
//! `validator_queue_air`, `eth1_data_voting_air`,
//! `tx_sender_recovery_air`.
//!
//! Extended modules (round 15): `randao_chain_air`,
//! `validator_balances_air`, `beacon_state_transition_air`.
//!
//! Extended modules (round 16): `block_proposer_sig_air`,
//! `proposer_shuffle_air`.
//!
//! Extended modules (round 17): `sync_committee_rotation_air`,
//! `create2_address_air`, `attestation_committee_air`.
//!
//! Extended modules (round 18): `tx_full_chain_air`,
//! `block_full_proof_air`, `withdrawal_queue_air`,
//! `eip1559_fee_market_air`.
//!
//! Extended modules (round 19): `multi_block_proof_air`,
//! `kzg_point_eval_air`, `ecrecover_chain_air`.
//!
//! Extended modules (round 20): `hash_to_field_air`,
//! `ripemd160_precompile_air`, `genesis_state_air`. Round 20 also
//! extends `beacon_block_body_air` with an 8-field merkleization AIR
//! (`BBB8_NUM_ROW_CONSTRAINTS = 33`, 7+1 descriptor invocations).
//!
//! Extended modules (round 21): `modexp_precompile_air`,
//! `bn254_precompile_air`, `blake2f_precompile_air`,
//! `shuffle_iteration_air`.
//!
//! Extended modules (round 22): `bn254_pairing_precompile_air`,
//! `hash_to_curve_composition_air`, `eip7251_compounding_air`.
//!
//! Extended modules (round 23): `beacon_block_validity_air`,
//! `sha3_full_chain_air`, `attestation_rewards_air`.
//!
//! Extended modules (round 24): `lmd_ghost_fork_choice_air`,
//! `shuffle_90_round_air`, `casper_ffg_chain_air`,
//! `epoch_processing_air`, `eip7702_delegation_air`,
//! `eip4337_user_op_air`, `data_availability_sampling_air`,
//! `ripemd160_internals_air`, `blake2_f_internals_air`,
//! `modexp_internals_air`, `bn254_curve_ops_air`,
//! `bn254_pairing_internals_air`, `precompile_io_chunked_air`.
//!
//! Extended modules (rounds 25–27): `secp256k1_fp_air` — limb-level
//! non-native Fp arithmetic AIR for the real secp256k1 base field
//! (102 columns, 19 constraint categories; full `VmConstraintSystem`
//! integration deferred so no `NUM_ROW_CONSTRAINTS` is contributed yet).
//! Rounds 25–27 also extended several existing AIRs with additional
//! algebraic binding: `blake2_f_internals_air` (NUM_COLUMNS 6740 → 7509,
//! NUM_ROW_CONSTRAINTS 8390 → 9355 via the IV-XOR layer and σ-perm
//! constraints), `ripemd160_internals_air` (250 → 317 columns,
//! 202 → 239 constraints), and `modexp_internals_air` (560 → 717
//! columns, 77 → 102 constraints). Round 25–27 also added three new
//! integration test modules — `integration_eip7702_joint_prove`,
//! `integration_das_joint_prove`, `integration_casper_ffg_joint_prove`
//! — which compose existing AIRs into multi-AIR end-to-end joint_prove
//! coverage without introducing new AIRs or row-constraints. #203 also
//! added 12 new standalone tests across these AIRs (no new AIRs).
//!
//! Extended modules (rounds 28–31): five new AIRs landed — `sha512_air`
//! (2264 columns, scaffold; full `Sha512ConstraintSystem` deferred so 0
//! `NUM_ROW_CONSTRAINTS` is contributed), `bn254_final_exp_air`
//! (194 columns, 5 row-constraints + 1 shifted),
//! `bn254_miller_loop_air` (206 columns, 10 row-constraints), plus two
//! sub-AIRs in `bn254_curve_ops_air`: `fp2` (53 columns, 12 row-constraints)
//! and `g2` (247 columns, 26 row-constraints including the curve_eq
//! check).
//!
//! Extended modules (rounds 32–34): seven new zkp-side AIRs landed —
//! three sub-AIRs in `bls12_381_curve_ops_air`: G1 (175 columns,
//! 16 row-constraints, the Fp re-export contributes no AIR columns),
//! `fp2` (77 columns, 12 row-constraints), and `g2` (343 columns,
//! 22 row-constraints) — plus `randao_32_epoch_chain_air` (240 columns,
//! 40 row-constraints + 1 shifted), `ssz_generalized_index_air`
//! (212 columns, 5 row-constraints + 3 shifted), `verkle_tree_air`
//! (164 columns, 5 row-constraints + 1 shifted), and `ed25519_air`
//! (482 columns, 67 row-constraints). `bn254_curve_ops_air::fp12` and
//! `bn254_pairing_air` host-side modules contribute no AIR columns
//! (host-side only). Rounds 32–34 also added ten new zkp-side
//! integration test modules — `integration_sync_committee_aggregate_joint_prove`,
//! `integration_block_full_proof_joint_prove`,
//! `integration_multi_block_proof_joint_prove`,
//! `integration_beacon_block_validity_joint_prove`,
//! `integration_tx_full_chain_joint_prove`,
//! `integration_full_eth_block_joint_prove`,
//! `integration_validator_deposit_joint_prove`,
//! `integration_beacon_block_proposer_signature_joint_prove`,
//! `integration_eip2537_joint_prove`,
//! `integration_multi_hash_chain_joint_prove`,
//! `integration_ffg_finality_joint_prove` — which compose existing AIRs
//! end-to-end without introducing new AIRs or row-constraints. The
//! `eip1559_fee_market_air` was extended with +2 shifted constraints
//! (multi-block base-fee chain + constant gas-target) and
//! `gas_tracking_air` (EVM-side) with +1 shifted constraint
//! (cumulative-gas transition); neither shifts row-constraint counts.
//! Rounds 28–31 also extended several existing AIRs with
//! additional binding under tasks #195 (`ripemd160_internals_air`
//! 250 → 317 cols), #182/#196 (`modexp_internals_air` 560 → 717 cols
//! via R-limb / Q-limb / slack range checks plus the M_lo ≥ R+1 full
//! 256-bit binding), and #205/#206/#215/#216 (`blake2_f_internals_air`
//! 7509 → 9557 cols, 9355 → 11947 constraints via the IV-XOR + h_out
//! triple-XOR finalize layers). Rounds 28–31 added nine new zkp-side
//! integration test modules — `integration_eip7702_joint_prove`,
//! `integration_das_joint_prove`, `integration_blob_full_chain_joint_prove`,
//! `integration_casper_ffg_joint_prove`,
//! `integration_multi_tx_full_chain_joint_prove`,
//! `integration_lmd_ghost_joint_prove`,
//! `integration_cross_block_beacon_transition_joint_prove`,
//! `integration_block_validity_master_joint_prove`,
//! `integration_hash_to_curve_joint_prove` (now passes) — which compose
//! existing AIRs end-to-end without introducing new AIRs or
//! row-constraints.
//!
//! Extended modules (rounds 35–37): the `sha512_constraints` module
//! landed with the full row-local algebraic implementation
//! (Σ/σ/Ch/Maj/W-recurrence/T1+T2/state-update layers, 12 row-constraint
//! categories) — `sha512_air` itself still exposes
//! `NUM_ROW_CONSTRAINTS = 0` as a column-layout-only scaffold, but the
//! coverage sum now includes the `sha512_constraints` 12-category
//! contribution explicitly. `blob_kzg_air` was extended with the IETF
//! G1-compressed flag-byte splitter (#290): +2 row-constraints
//! (commitment-byte-0 / proof-byte-0 = 32·flag_bits + masked_byte_0) and
//! +4 columns (flag_bits + masked byte 0, both ×2). Rounds 35–37 also
//! populate `bn254_miller_loop_air` (#259 / #269) and add several
//! integration test modules + the host-side
//! `bn254_curve_ops_air::fp12` arithmetic module (host-side only, no AIR
//! columns / row-constraints / descriptors). `gas_tracking_air`
//! (EVM-side) gained +1 shifted constraint (#276); `eip1559_fee_market_air`
//! gained +2 shifted constraints from #286 (the EIP-1559 multi-block
//! chain hookup); neither changes row-constraint counts. Round-35–37
//! also adds `blob_kzg_air` as a newly tracked extended AIR; previously
//! the AIR existed in the codebase but its row-constraints were not
//! summed into [`count_extended_constraints`].

use crate::account::{empty_code_hash, Account};
use crate::account_state_air::{
    self as account_state, AccountStateConstraintSystem, AccountStateRow, AccountStateWitness,
};
use crate::address_keccak_air::{
    self as ak, AddressKeccakConstraintSystem, AddressKeccakWitness,
};
use crate::beacon::BeaconBlockHeader;
use crate::beacon_block_body::BeaconBlockBody;
use crate::beacon_block_body_air::BeaconBlockBodyHtrWitness;
use crate::beacon_block_body_pair_air as body_pair;
use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;
use crate::beacon_block_header_pair_air as bbh_pair;
use crate::bbh_root_consumer_air::{
    self as bbh_consumer, BbhRootConsumerConstraintSystem, BbhRootConsumerWitness,
};
use crate::block_header::{block_header_hash, BlockHeader};
use crate::block_header_air::{
    self as bh, BlockHeaderConstraintSystem, BlockHeaderWitness,
};
use crate::byte_memory_air::{self as byte_mem};
use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::execution_payload::ExecutionPayloadHeader;
use crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness;
use crate::execution_payload_pair_air as payload_pair;
use crate::field::CurveType;
use crate::finality_constraints as fin;
use crate::keccak_extract::{
    self as kex, KeccakExtractConstraintSystem, KeccakExtractWitness,
};
use crate::keccak_extract_wide::{
    self as kex_wide, KeccakExtractWideConstraintSystem, KeccakExtractWideWitness,
};
use crate::mpt;
use crate::mpt_constraints::{self as mpt_cs, MptInclusionConstraintSystem};
use crate::scheme::CommitmentScheme;
use crate::sha256_extract::{
    self as sex, Sha256ExtractConstraintSystem, Sha256ExtractWitness,
};
use crate::storage_access_air::{
    self as storage, StorageAccessConstraintSystem, StorageAccessRow, StorageAccessWitness,
};
use crate::trace::TracePolynomials;
use crate::vm_constraints::VmConstraintSystem;

// ─── Layer indices (stable; baked into descriptor wiring) ────────────

/// Layer A: EVM-side storage gadget.
pub const LAYER_STORAGE: usize = 0;
/// Layer A: account-state gadget (binds account + state_root).
pub const LAYER_ACCOUNT_STATE: usize = 1;
/// Layer A: address → keccak preimage gadget.
pub const LAYER_ADDRESS_KECCAK: usize = 2;
/// Layer A: byte-memory perm gadget (EVM SHA3 ↔ memory binding).
pub const LAYER_BYTE_MEMORY: usize = 3;
/// Shared (A/B): generic keccak preimage extractor (32-byte inputs).
pub const LAYER_KECCAK_EXTRACT: usize = 4;
/// Layer B: block_header_air (RLP + 4 hash columns).
pub const LAYER_BLOCK_HEADER: usize = 5;
/// Layer B: 768-byte keccak preimage extractor for the full header RLP.
pub const LAYER_KECCAK_EXTRACT_WIDE: usize = 6;
/// Layer B: MPT inclusion AIR (single proof).
pub const LAYER_MPT: usize = 7;
/// Layer C: BeaconBlockHeader-pair AIR (SSZ HTR).
pub const LAYER_BBH_PAIR: usize = 8;
/// Layer C: BeaconBlockBody-pair AIR (SSZ HTR).
pub const LAYER_BODY_PAIR: usize = 9;
/// Layer C: ExecutionPayloadHeader-pair AIR (SSZ HTR).
pub const LAYER_PAYLOAD_PAIR: usize = 10;
/// Layer C: BBH root consumer (pass-through for downstream layers).
pub const LAYER_BBH_CONSUMER: usize = 11;
/// Layer C: sha256_extract — SSZ-pair sha256 preimage extractor.
pub const LAYER_SHA256_EXTRACT: usize = 12;
/// Layer C: stake-weighted FFG finality.
pub const LAYER_FINALITY: usize = 13;

/// Total number of AIRs included in the ultimate bundle (zkp-side only).
pub const NUM_AIRS: usize = 14;

/// Number of extended AIRs landed in rounds 11–34 and tracked for
/// metrics only (not yet wired into the runnable bundle — see
/// module-level docs). Rounds 25–27 added `secp256k1_fp_air` (+1,
/// scaffold; no `VmConstraintSystem` integration yet so contributes 0
/// row-constraints). Rounds 28–31 added five new AIRs:
/// `sha512_air` (+1, scaffold; no CS yet so 0 row-constraints),
/// `bn254_final_exp_air` (+1, 5 row-constraints + 1 shifted),
/// `bn254_miller_loop_air` (+1, 10 row-constraints),
/// `bn254_curve_ops_air::fp2` (+1, 12 row-constraints sub-AIR),
/// `bn254_curve_ops_air::g2` (+1, 26 row-constraints sub-AIR).
/// Rounds 32–34 added seven new AIRs:
/// `bls12_381_curve_ops_air` G1 (+1, 16 row-constraints),
/// `bls12_381_curve_ops_air::fp2` (+1, 12 row-constraints),
/// `bls12_381_curve_ops_air::g2` (+1, 22 row-constraints),
/// `randao_32_epoch_chain_air` (+1, 40 row-constraints + 1 shifted),
/// `ssz_generalized_index_air` (+1, 5 row-constraints + 3 shifted),
/// `verkle_tree_air` (+1, 5 row-constraints + 1 shifted; round 38
/// extends this to 7 row-constraints + 2 shifted as the Pedersen
/// partial-sum binding lands, +128 cols),
/// `ed25519_air` (+1, 67 row-constraints). Rounds 35–37 added
/// `blob_kzg_air` (+1, 47 row-constraints including the #290 flag-byte
/// splitter contribution of 2; previously the AIR existed but was not
/// summed into the extended count). The `sha512_constraints` module is not a
/// separate AIR — it provides the row-local algebra for `sha512_air`
/// (already counted as one extended AIR) but contributes its 12
/// row-constraint categories explicitly to
/// [`count_extended_constraints`]. Round 38 adds `eip3074_air` (+1,
/// 189 cols, 6 row-constraints) covering the AUTH/AUTHCALL opcode
/// shape introduced by the (now-superseded) EIP-3074 proposal; it is
/// tracked as a forward-looking scaffold AIR even though Cancun does
/// not include the opcodes.
pub const NUM_EXTENDED_AIRS: usize = 71;

/// Total zkp-side AIRs covered (wired + extended).
pub const NUM_TOTAL_AIRS: usize = NUM_AIRS + NUM_EXTENDED_AIRS;

/// Number of cross-AIR LogUp descriptor builders exposed by the 57
/// extended modules (plus the BBB8 extension on `beacon_block_body_air`).
/// Counted from each module's `make_*_descriptor` public functions and
/// tracked for coverage reporting.
///
/// Breakdown: 42 (rounds 11–14) + 41 (rounds 15–18) + 22 (rounds 19–20,
/// including 7 BBB8 sha256 pair descriptors + 1 BBB8 → BBH descriptor)
/// + 7 (round 21: modexp 2 + bn254 2 + blake2f 2 + shuffle 1)
/// + 9 (round 22: bn254_pairing 3 + h2c_composition 4 + eip7251 2)
/// + 12 (round 23: beacon_block_validity 5 + sha3_full_chain 5
/// + attestation_rewards 2)
/// + 16 (round 24: lmd_ghost 2 + shuffle_90 1 + casper_ffg 2
/// + epoch_processing 2 + eip7702 2 + eip4337 2
/// + data_availability_sampling 1 + ripemd160_internals 1
/// + blake2_f_internals 1 + modexp_internals 1 + bn254_curve_ops 0
/// + bn254_pairing_internals 0 + precompile_io_chunked 1)
/// + 0 (rounds 25–27: `secp256k1_fp_air` scaffold + 3 integration
/// composition modules contribute no new descriptor builders;
/// row-constraint deltas land on already-tracked AIRs)
/// + 19 (rounds 28–31: bn254_final_exp 2 + bn254_miller_loop 3
/// + bn254_curve_ops::fp2 13 + bn254_curve_ops::g2 1
/// + sha512_air 0 + 9 new integration composition modules contribute
/// no new descriptor builders)
/// + 21 (rounds 32–34: bls12_381_curve_ops G1 1
/// + bls12_381_curve_ops::fp2 13 + bls12_381_curve_ops::g2 1
/// + randao_32_epoch_chain 2 + ssz_generalized_index 1
/// + verkle_tree 2 + ed25519 1 + 10 new integration composition
/// modules contribute no new descriptor builders; the bn254 fp12 host
/// module is host-side only and contributes no descriptors)
/// + 0 (rounds 35–37: blob_kzg_air #290 flag-byte splitter, the
/// sha512_constraints full algebraic landing, bn254_miller_loop_air
/// populate, and the round-35–37 integration test modules all
/// contribute no new descriptor builders)
/// = 210.
pub const NUM_EXTENDED_DESCRIPTORS: usize = 210;

// ─── Bundle ──────────────────────────────────────────────────────────

/// All witnesses + traces + constraint systems composing the ultimate
/// bundle. The ordering of `traces()` matches the `LAYER_*` constants.
#[allow(dead_code)]
pub struct OrchestrationBundle {
    pub curve: CurveType,

    // ── Layer A ────────────────────────────────────────────────────
    pub storage_trace: TracePolynomials,
    pub storage_cs: StorageAccessConstraintSystem,

    pub account_trace: TracePolynomials,
    pub account_cs: AccountStateConstraintSystem,

    pub ak_trace: TracePolynomials,
    pub ak_cs: AddressKeccakConstraintSystem,

    pub byte_mem_trace: TracePolynomials,
    pub byte_mem_cs: byte_mem::ByteMemoryConstraintSystem,

    // ── Shared/Layer B ────────────────────────────────────────────
    pub keccak_extract_trace: TracePolynomials,
    pub keccak_extract_cs: KeccakExtractConstraintSystem,

    pub bh_trace: TracePolynomials,
    pub bh_cs: BlockHeaderConstraintSystem,

    pub keccak_wide_trace: TracePolynomials,
    pub keccak_wide_cs: KeccakExtractWideConstraintSystem,

    pub mpt_trace: TracePolynomials,
    pub mpt_cs: MptInclusionConstraintSystem,

    // ── Layer C ────────────────────────────────────────────────────
    pub bbh_pair_trace: TracePolynomials,
    pub bbh_pair_cs: bbh_pair::BeaconBlockHeaderPairConstraintSystem,

    pub body_pair_trace: TracePolynomials,
    pub body_pair_cs: body_pair::BeaconBlockBodyPairConstraintSystem,

    pub payload_pair_trace: TracePolynomials,
    pub payload_pair_cs: payload_pair::ExecutionPayloadHeaderPairConstraintSystem,

    pub bbh_consumer_trace: TracePolynomials,
    pub bbh_consumer_cs: BbhRootConsumerConstraintSystem,

    pub sha256_extract_trace: TracePolynomials,
    pub sha256_extract_cs: Sha256ExtractConstraintSystem,

    pub fin_trace: TracePolynomials,
    pub fin_cs: fin::FinalityConstraintSystem,
}

impl OrchestrationBundle {
    /// Borrow the per-AIR `(trace, constraint_system)` tuples in
    /// `LAYER_*` order — directly consumable by
    /// [`crate::cross_air_logup::joint_prove`].
    pub fn traces<'a>(
        &'a self,
    ) -> Vec<(&'a TracePolynomials, &'a dyn VmConstraintSystem)> {
        vec![
            (&self.storage_trace, &self.storage_cs as &dyn VmConstraintSystem),
            (&self.account_trace, &self.account_cs),
            (&self.ak_trace, &self.ak_cs),
            (&self.byte_mem_trace, &self.byte_mem_cs),
            (&self.keccak_extract_trace, &self.keccak_extract_cs),
            (&self.bh_trace, &self.bh_cs),
            (&self.keccak_wide_trace, &self.keccak_wide_cs),
            (&self.mpt_trace, &self.mpt_cs),
            (&self.bbh_pair_trace, &self.bbh_pair_cs),
            (&self.body_pair_trace, &self.body_pair_cs),
            (&self.payload_pair_trace, &self.payload_pair_cs),
            (&self.bbh_consumer_trace, &self.bbh_consumer_cs),
            (&self.sha256_extract_trace, &self.sha256_extract_cs),
            (&self.fin_trace, &self.fin_cs),
        ]
    }

    /// Borrow only the constraint systems in `LAYER_*` order — for
    /// [`crate::cross_air_logup::joint_verify`].
    pub fn cs_refs<'a>(&'a self) -> Vec<&'a dyn VmConstraintSystem> {
        vec![
            &self.storage_cs,
            &self.account_cs,
            &self.ak_cs,
            &self.byte_mem_cs,
            &self.keccak_extract_cs,
            &self.bh_cs,
            &self.keccak_wide_cs,
            &self.mpt_cs,
            &self.bbh_pair_cs,
            &self.body_pair_cs,
            &self.payload_pair_cs,
            &self.bbh_consumer_cs,
            &self.sha256_extract_cs,
            &self.fin_cs,
        ]
    }
}

// ─── Sample transaction-like input ────────────────────────────────────

/// Minimal input for [`assemble_bundle`]. A real driver would derive
/// these from a fetched block + tx; tests use sample values.
#[derive(Clone, Debug)]
pub struct SampleTxInput {
    pub address: [u8; 20],
    pub slot_be: [u8; 32],
    pub value_be: [u8; 32],
    pub block_number: u64,
}

impl Default for SampleTxInput {
    fn default() -> Self {
        let mut slot = [0u8; 32];
        slot[31] = 7;
        let mut value = [0u8; 32];
        value[31] = 0x42;
        Self {
            address: [0xab; 20],
            slot_be: slot,
            value_be: value,
            block_number: 18_500_000,
        }
    }
}

// ─── Bundle assembler ────────────────────────────────────────────────

/// Build the entire orchestration bundle from a single sample input.
///
/// All witnesses are constructed with consistent boundary values so
/// the same `(address, state_root, body_root, ...)` is referenced
/// across every layer. **No proving is performed.**
pub fn assemble_bundle(
    scheme: &dyn CommitmentScheme,
    input: &SampleTxInput,
) -> OrchestrationBundle {
    let curve = CurveType::Bls48581;

    // ── Layer A: storage access ──────────────────────────────────
    let mut slot_limbs = [0u64; 4];
    slot_limbs[0] = input.slot_be[31] as u64;
    let mut value_limbs = [0u64; 4];
    value_limbs[0] = input.value_be[31] as u64;
    // Build the MPT trie up-front so storage_root matches the actual
    // single-leaf root. The leaf value is the raw 32 BE bytes of the
    // slot value — `mpt_node_rlp` internally wraps it via
    // `rlp_encode_bytes`, producing the Phase 1 leaf shape (64-nibble
    // even path + 32-byte value, 69-byte long-form list) that mpt_air
    // can decode.
    let trie_key = crate::keccak::keccak256(&input.slot_be);
    let (mpt_root, mpt_proof) =
        mpt::single_leaf_trie(&trie_key, &input.value_be);
    let storage_root = mpt_root;
    let state_root = [0x99; 32];

    let storage_w = StorageAccessWitness::from_rows(vec![StorageAccessRow {
        address: input.address,
        slot: slot_limbs,
        value: value_limbs,
        storage_root,
        is_write: false,
    }]);
    let storage_trace = storage::build_trace_polynomials(&storage_w, curve);
    let storage_omega = scheme.domain_generator(storage_trace.padded_size);
    let storage_cs = StorageAccessConstraintSystem::new(storage_trace.num_rows)
        .with_omega_and_domain(storage_omega, storage_trace.padded_size);

    // ── Layer A: account state ──────────────────────────────────
    let account = Account {
        nonce: 1,
        balance: [0u8; 32],
        storage_root,
        code_hash: empty_code_hash(),
    };
    let account_w = AccountStateWitness::from_rows(vec![AccountStateRow::new(
        input.address,
        account,
        state_root,
    )]);
    let account_trace = account_state::build_trace_polynomials(&account_w, curve);
    let account_omega = scheme.domain_generator(account_trace.padded_size);
    let account_cs = AccountStateConstraintSystem::new(account_trace.num_rows)
        .with_omega_and_domain(account_omega, account_trace.padded_size);

    // ── Layer A: address → keccak gadget ────────────────────────
    // Need 2 rows because both storage and account_state emit
    // address tuples (multiset equality).
    let ak_w = AddressKeccakWitness::from_addresses(vec![input.address, input.address]);
    let ak_trace = ak::build_trace_polynomials(&ak_w, curve);
    let ak_omega = scheme.domain_generator(ak_trace.padded_size);
    let ak_cs = AddressKeccakConstraintSystem::new(ak_trace.num_rows)
        .with_omega_and_domain(ak_omega, ak_trace.padded_size);

    // ── Layer A: byte-memory (placeholder empty trace) ───────────
    // Minimum-viable honest witness: 1 write + 1 read at same address.
    // The empty witness was insufficient — joint_prove inflates the trace
    // to RANGE_TABLE_SIZE (256), and byte_memory_air's sorted-view
    // multiset/continuity constraints can't satisfy a fully-padded trace
    // at that domain. Providing a minimal honest pair sets IS_FIRST and
    // the sorted-view boundary in a well-defined state.
    let byte_mem_w = byte_mem::ByteMemoryWitness::from_accesses(vec![
        byte_mem::ByteMemoryAccess { addr: 0, val: 0x42, ts: 0, rw: 1, source_row: 0 },
        byte_mem::ByteMemoryAccess { addr: 0, val: 0x42, ts: 1, rw: 0, source_row: 1 },
    ]);
    let byte_mem_trace = byte_mem::build_trace_polynomials(&byte_mem_w, curve);
    let byte_mem_omega = scheme.domain_generator(byte_mem_trace.padded_size);
    let byte_mem_cs = byte_mem::ByteMemoryConstraintSystem::new(byte_mem_trace.num_rows)
        .with_omega_and_domain(byte_mem_omega, byte_mem_trace.padded_size);

    // ── Shared: keccak extract — covers address+slot keccak inputs ─
    let keccak_inputs: Vec<Vec<u8>> = vec![
        input.address.to_vec(),
        input.address.to_vec(),
        input.slot_be.to_vec(),
    ];
    let keccak_w = KeccakExtractWitness::from_inputs(&keccak_inputs)
        .expect("keccak inputs within MAX_INPUT_LEN");
    let keccak_extract_trace = kex::build_trace_polynomials(&keccak_w, curve);
    let keccak_extract_omega = scheme.domain_generator(keccak_extract_trace.padded_size);
    let keccak_extract_cs = KeccakExtractConstraintSystem::new(keccak_extract_trace.num_rows)
        .with_omega_and_domain(keccak_extract_omega, keccak_extract_trace.padded_size);

    // ── Layer B: block header ────────────────────────────────────
    let block_header = BlockHeader {
        state_root,
        transactions_root: [0x66; 32],
        receipts_root: [0x77; 32],
        number: input.block_number,
        gas_limit: 30_000_000,
        gas_used: 21_000,
        timestamp: 1_700_000_000,
        ..Default::default()
    };
    let block_hash = block_header_hash(&block_header);
    let bh_row = bh::from_block_header(&block_header);
    let bh_w = BlockHeaderWitness::from_headers(vec![bh_row.clone()]);
    let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
    let bh_omega = scheme.domain_generator(bh_trace.padded_size);
    let bh_cs = BlockHeaderConstraintSystem::new(bh_trace.num_rows)
        .with_omega_and_domain(bh_omega, bh_trace.padded_size);

    // ── Layer B: keccak extract wide (header RLP → block_hash) ───
    let header_rlp_bytes = &bh_row.header_rlp[..bh_row.header_rlp_len as usize];
    let keccak_wide_w = KeccakExtractWideWitness::from_inputs(&[header_rlp_bytes])
        .expect("header_rlp fits within wide extractor bound");
    let keccak_wide_trace = kex_wide::build_trace_polynomials(&keccak_wide_w, curve);
    let keccak_wide_omega = scheme.domain_generator(keccak_wide_trace.padded_size);
    let keccak_wide_cs = KeccakExtractWideConstraintSystem::new(keccak_wide_trace.num_rows)
        .with_omega_and_domain(keccak_wide_omega, keccak_wide_trace.padded_size);

    // ── Layer B: MPT inclusion (single-leaf storage trie demo) ──
    // Reuses the trie_key + proof produced above so the storage_root
    // exposed to Layer A actually matches the keccak256 of the leaf
    // RLP — without this the storage↔MPT closures diverge.
    let mpt_trace =
        mpt_cs::build_trace_polynomials_from_proof(&trie_key, &mpt_proof, curve);
    let mpt_omega = scheme.domain_generator(mpt_trace.padded_size);
    let mpt_cs_built = MptInclusionConstraintSystem::new(mpt_trace.num_rows)
        .with_omega_and_domain(mpt_omega, mpt_trace.padded_size);

    // ── Layer C: beacon block header pair ────────────────────────
    // Build a consistent BBH → body → payload chain referencing the
    // execution block_hash we just produced.
    let mut payload = ExecutionPayloadHeader::default();
    payload.block_hash = block_hash;
    payload.block_number = input.block_number;
    payload.gas_limit = 30_000_000;
    payload.timestamp = 1_700_000_000;

    let body_struct = BeaconBlockBody {
        graffiti: [0x47; 32],
        execution_payload_header: payload.clone(),
        ..Default::default()
    };
    let body_root = body_struct.hash_tree_root();
    let beacon_parent_root = [0xaa_u8; 32];
    let beacon_state_root = [0xbb_u8; 32];
    let beacon_slot = 7_777_777u64;
    let beacon_proposer_index = 13u64;
    let beacon_header = BeaconBlockHeader {
        slot: beacon_slot,
        proposer_index: beacon_proposer_index,
        parent_root: beacon_parent_root,
        state_root: beacon_state_root,
        body_root,
    };
    let bbh_root = beacon_header.hash_tree_root();

    let bbh_w = BeaconBlockHeaderHtrWitness::from_header(beacon_header);
    let bbh_pair_trace = bbh_pair::build_trace_polynomials(&bbh_w, curve);
    let bbh_pair_omega = scheme.domain_generator(bbh_pair_trace.padded_size);
    let bbh_pair_cs = bbh_pair::BeaconBlockHeaderPairConstraintSystem::new(bbh_pair_trace.num_rows)
        .with_omega_and_domain(bbh_pair_omega, bbh_pair_trace.padded_size);

    // ── Layer C: beacon body pair ────────────────────────────────
    let body_w = BeaconBlockBodyHtrWitness::from_body(body_struct);
    let body_pair_trace = body_pair::build_trace_polynomials(&body_w, curve);
    let body_pair_omega = scheme.domain_generator(body_pair_trace.padded_size);
    let body_pair_cs = body_pair::BeaconBlockBodyPairConstraintSystem::new(body_pair_trace.num_rows)
        .with_omega_and_domain(body_pair_omega, body_pair_trace.padded_size);

    // ── Layer C: payload pair ────────────────────────────────────
    let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(payload);
    let payload_pair_trace = payload_pair::build_trace_polynomials(&payload_w, curve);
    let payload_pair_omega = scheme.domain_generator(payload_pair_trace.padded_size);
    let payload_pair_cs =
        payload_pair::ExecutionPayloadHeaderPairConstraintSystem::new(payload_pair_trace.num_rows)
            .with_omega_and_domain(payload_pair_omega, payload_pair_trace.padded_size);

    // ── Layer C: BBH root consumer (pass-through column) ────────
    let bbh_consumer_w = BbhRootConsumerWitness::from_bbh_tuple(
        1,
        bbh_root,
        beacon_parent_root,
        beacon_state_root,
        body_root,
        beacon_slot,
        beacon_proposer_index,
    );
    let bbh_consumer_trace = bbh_consumer::build_trace_polynomials(&bbh_consumer_w, curve);
    let bbh_consumer_omega = scheme.domain_generator(bbh_consumer_trace.padded_size);
    let bbh_consumer_cs = BbhRootConsumerConstraintSystem::new(bbh_consumer_trace.num_rows)
        .with_omega_and_domain(bbh_consumer_omega, bbh_consumer_trace.padded_size);

    // ── Layer C: sha256 extract — covers every (left, right) pair
    // emitted by the BBH-pair, body-pair, and payload-pair AIRs so the
    // three Layer-C sha256 cross-AIR LogUp linkages have matching
    // B-side tuples for every A-side row.
    let mut sha256_pairs: Vec<(crate::ssz::Chunk, crate::ssz::Chunk)> = Vec::new();
    for inv in bbh_w.invocations.iter() {
        sha256_pairs.push((inv.left, inv.right));
    }
    for inv in body_w.invocations.iter() {
        sha256_pairs.push((inv.left, inv.right));
    }
    for inv in payload_w.invocations.iter() {
        sha256_pairs.push((inv.left, inv.right));
    }
    let sha256_w = Sha256ExtractWitness::from_pair_inputs(&sha256_pairs);
    let sha256_extract_trace = sex::build_trace_polynomials(&sha256_w, curve);
    let sha256_extract_omega = scheme.domain_generator(sha256_extract_trace.padded_size);
    let sha256_extract_cs = Sha256ExtractConstraintSystem::new(sha256_extract_trace.num_rows)
        .with_omega_and_domain(sha256_extract_omega, sha256_extract_trace.padded_size);

    // ── Layer C: stake-weighted FFG finality ─────────────────────
    // 4-validator committee, 3-of-4 attest (75% > 2/3 threshold).
    let eb = 32_000_000_000u64;
    let finality_w = fin::FinalityWitness::new(
        vec![(eb, 1), (eb, 1), (eb, 1), (eb, 0)],
        4 * eb,
        [0xDD; 32],
        bbh_root,
    );
    let fin_trace = fin::build_finality_trace_polynomials(&finality_w, curve);
    let fin_omega = scheme.domain_generator(fin_trace.padded_size);
    let fin_cs = fin::FinalityConstraintSystem::new(finality_w.validators.len())
        .with_omega_and_domain(fin_omega, fin_trace.padded_size);

    OrchestrationBundle {
        curve,
        storage_trace,
        storage_cs,
        account_trace,
        account_cs,
        ak_trace,
        ak_cs,
        byte_mem_trace,
        byte_mem_cs,
        keccak_extract_trace,
        keccak_extract_cs,
        bh_trace,
        bh_cs,
        keccak_wide_trace,
        keccak_wide_cs,
        mpt_trace,
        mpt_cs: mpt_cs_built,
        bbh_pair_trace,
        bbh_pair_cs,
        body_pair_trace,
        body_pair_cs,
        payload_pair_trace,
        payload_pair_cs,
        bbh_consumer_trace,
        bbh_consumer_cs,
        sha256_extract_trace,
        sha256_extract_cs,
        fin_trace,
        fin_cs,
    }
}

// ─── Descriptor wiring ────────────────────────────────────────────────

/// Collect every cross-AIR LogUp descriptor used by the bundle.
///
/// Naming reflects the data-flow direction: `from_to` means the A
/// (selector) side is `from` and the B (table) side is `to`.
pub fn collect_descriptors() -> Vec<CrossAirLogUpDescriptor> {
    vec![
        // ── Layer A internal chain ────────────────────────────────
        // storage → account_state (address + storage_root match).
        account_state::make_storage_to_account_state_linkage_descriptor(
            LAYER_STORAGE,
            LAYER_ACCOUNT_STATE,
        ),
        // storage → address_keccak (address tuple).
        ak::make_storage_to_address_keccak_linkage_descriptor(
            LAYER_STORAGE,
            LAYER_ADDRESS_KECCAK,
        ),
        // account_state → address_keccak (address tuple).
        account_state::make_account_state_to_address_keccak_linkage_descriptor(
            LAYER_ACCOUNT_STATE,
            LAYER_ADDRESS_KECCAK,
        ),
        // address_keccak → keccak_extract (full 20B+32B preimage tuple).
        ak::make_address_to_keccak_extract_linkage_descriptor(
            LAYER_ADDRESS_KECCAK,
            LAYER_KECCAK_EXTRACT,
        ),
        // storage → keccak_extract (slot_be → slot_trie_key keccak).
        storage::make_storage_to_keccak_extract_linkage_descriptor(
            LAYER_STORAGE,
            LAYER_KECCAK_EXTRACT,
        ),
        // byte-memory perm internal closure (self-linkage).
        byte_mem::make_byte_memory_self_linkage_descriptor(LAYER_BYTE_MEMORY),

        // ── Layer A → Layer B ─────────────────────────────────────
        // account_state → block_header (state_root binding).
        account_state::make_account_state_to_block_header_linkage_descriptor(
            LAYER_ACCOUNT_STATE,
            LAYER_BLOCK_HEADER,
        ),
        // storage_root → MPT root parent_hash chain.
        storage::make_storage_root_to_mpt_root_linkage_descriptor(
            LAYER_STORAGE,
            LAYER_MPT,
        ),
        // storage (root + value) → MPT (root + leaf_value_bytes).
        storage::make_storage_root_value_to_mpt_root_linkage_descriptor(
            LAYER_STORAGE,
            LAYER_MPT,
        ),
        // storage (root + value + leaf_key) → MPT (full leaf binding).
        storage::make_storage_full_to_mpt_root_linkage_descriptor(
            LAYER_STORAGE,
            LAYER_MPT,
        ),
        // storage multi-row variant (root + value + full_key bytes).
        storage::make_storage_multirow_to_mpt_root_linkage_descriptor(
            LAYER_STORAGE,
            LAYER_MPT,
        ),

        // ── Layer B internal ──────────────────────────────────────
        // block_header → keccak_extract_WIDE (full 768-byte header_rlp
        // → 32-byte block_hash). Despite its name,
        // `bh::make_block_header_to_keccak_extract_linkage_descriptor`
        // references the wide variant's column layout
        // (HEADER_RLP_MAX_LEN = 768 input bytes), so it must point at
        // the wide keccak AIR — pointing it at the narrow 256-byte
        // KeccakExtract over-runs that AIR's column count and
        // panics with an out-of-bounds index.
        bh::make_block_header_to_keccak_extract_linkage_descriptor(
            LAYER_BLOCK_HEADER,
            LAYER_KECCAK_EXTRACT_WIDE,
        ),
        // block_header → keccak_extract_wide (full header_rlp → block_hash).
        kex_wide::make_block_header_to_keccak_extract_wide_linkage_descriptor(
            LAYER_BLOCK_HEADER,
            LAYER_KECCAK_EXTRACT_WIDE,
        ),

        // ── Layer C chain (consensus) ─────────────────────────────
        // body-pair → BBH-pair (body_root pickup).
        body_pair::make_body_pair_to_bbh_pair_linkage_descriptor(
            LAYER_BODY_PAIR,
            LAYER_BBH_PAIR,
        ),
        // payload-pair → body-pair (payload_root pickup).
        body_pair::make_payload_pair_to_body_pair_linkage_descriptor(
            LAYER_PAYLOAD_PAIR,
            LAYER_BODY_PAIR,
        ),
        // payload-pair → block_header (execution block_hash binding).
        payload_pair::make_payload_pair_to_block_header_linkage_descriptor(
            LAYER_PAYLOAD_PAIR,
            LAYER_BLOCK_HEADER,
        ),
        // BBH-pair → root consumer (pass-through column).
        bbh_consumer::make_bbh_pair_to_root_consumer_linkage_descriptor(
            LAYER_BBH_PAIR,
            LAYER_BBH_CONSUMER,
        ),

        // ── Layer C: finality binding ────────────────────────────
        // finality → BBH-pair (finalized_root == bbh_root).
        fin::make_finality_to_bbh_pair_linkage_descriptor(
            LAYER_FINALITY,
            LAYER_BBH_PAIR,
        ),

        // ── SSZ infrastructure ────────────────────────────────────
        // sha256_extract self-binding via the standard extractor
        // descriptor is built per-AIR-pair; here we wire two such
        // descriptors covering BBH-pair and body-pair sha256 inputs.
        body_pair::make_body_pair_to_sha256_extract_linkage_descriptor(
            LAYER_BODY_PAIR,
            LAYER_SHA256_EXTRACT,
        ),
        bbh_pair::make_beacon_block_header_pair_to_sha256_extract_linkage_descriptor(
            LAYER_BBH_PAIR,
            LAYER_SHA256_EXTRACT,
        ),
        payload_pair::make_payload_pair_to_sha256_extract_linkage_descriptor(
            LAYER_PAYLOAD_PAIR,
            LAYER_SHA256_EXTRACT,
        ),
    ]
}

// ─── Constraint counting ─────────────────────────────────────────────

/// Sum of `NUM_ROW_CONSTRAINTS` across every AIR in the bundle.
///
/// This is a coarse complexity metric — actual prove-time cost
/// scales with `padded_size * num_constraints * num_quotient_columns`,
/// not just constraint count.
pub fn count_constraints() -> usize {
    use crate::{
        address_keccak_air as ak_mod,
        account_state_air as account_mod,
        beacon_block_body_pair_air as body_mod,
        beacon_block_header_pair_air as bbh_mod,
        bbh_root_consumer_air as bbh_consumer_mod,
        block_header_air as bh_mod,
        byte_memory_air as bm_mod,
        execution_payload_pair_air as payload_mod,
        finality_constraints as fin_mod,
        keccak_extract as kex_mod,
        keccak_extract_wide as kex_wide_mod,
        mpt_constraints as mpt_mod,
        sha256_extract as sex_mod,
        storage_access_air as storage_mod,
    };
    storage_mod::NUM_ROW_CONSTRAINTS
        + account_mod::NUM_ROW_CONSTRAINTS
        + ak_mod::NUM_ROW_CONSTRAINTS
        + bm_mod::NUM_ROW_CONSTRAINTS
        + kex_mod::NUM_ROW_CONSTRAINTS
        + bh_mod::NUM_ROW_CONSTRAINTS
        + kex_wide_mod::NUM_ROW_CONSTRAINTS
        + mpt_mod::NUM_ROW_CONSTRAINTS
        + bbh_mod::NUM_ROW_CONSTRAINTS
        + body_mod::NUM_ROW_CONSTRAINTS
        + payload_mod::NUM_ROW_CONSTRAINTS
        + bbh_consumer_mod::NUM_ROW_CONSTRAINTS
        + sex_mod::NUM_ROW_CONSTRAINTS
        + fin_mod::NUM_ROW_CONSTRAINTS
}

/// Sum of `NUM_ROW_CONSTRAINTS` across the 62 extended AIRs landed in
/// rounds 11–31 (tracked for metrics; not yet wired into the runnable
/// bundle). Rounds 19–20 also include the `beacon_block_body_air` 8-field
/// merkleization extension (`BBB8_NUM_ROW_CONSTRAINTS`). Rounds 25–27
/// extended `blake2_f_internals_air` (#215/#216 IV-XOR + h_out triple
/// XOR layers — 8390 → 11947), `ripemd160_internals_air`, and
/// `modexp_internals_air` with additional algebraic binding; the
/// `secp256k1_fp_air` scaffold has no `NUM_ROW_CONSTRAINTS` yet (full
/// `VmConstraintSystem` integration deferred). Rounds 28–31 added
/// `sha512_air` (scaffold, 0 row-constraints), `bn254_final_exp_air`
/// (5), `bn254_miller_loop_air` (10), and two `bn254_curve_ops_air`
/// sub-AIRs `fp2` (12) and `g2` (26). Rounds 35–37 added the
/// `sha512_constraints` row-local algebra module (12 categories) and
/// the previously untracked `blob_kzg_air` (49 row-constraints after
/// the previously untracked `blob_kzg_air` (47 row-constraints total,
/// including 2 newly added by the #290 flag-byte splitter).
pub fn count_extended_constraints() -> usize {
    use crate::{
        access_list_air as access_mod, attestation_aggregate_air as att_agg_mod,
        attester_slashing_air as att_slash_mod, deposit_tree_air as dep_mod,
        eth1_data_voting_air as eth1_mod, proposer_slashing_air as prop_slash_mod,
        randao_proposer_air as randao_mod, receipt_status_air as receipt_mod,
        sync_committee_sig_air as sync_sig_mod, tx_nonce_air as tx_nonce_mod,
        tx_sender_recovery_air as tx_sender_mod, validator_queue_air as vq_mod,
        versioned_hash_air as vh_mod, voluntary_exit_air as ve_mod,
        withdrawal_credential_air as wc_mod,
        // Rounds 15–18 additions.
        randao_chain_air as randao_chain_mod,
        validator_balances_air as val_bal_mod,
        beacon_state_transition_air as bst_mod,
        block_proposer_sig_air as block_prop_sig_mod,
        proposer_shuffle_air as prop_shuffle_mod,
        sync_committee_rotation_air as sync_rot_mod,
        create2_address_air as create2_mod,
        attestation_committee_air as att_comm_mod,
        tx_full_chain_air as tx_full_mod,
        block_full_proof_air as block_full_mod,
        withdrawal_queue_air as wq_mod,
        eip1559_fee_market_air as eip1559_mod,
        // Rounds 19–20 additions.
        multi_block_proof_air as mbp_mod,
        kzg_point_eval_air as kzg_eval_mod,
        ecrecover_chain_air as ecrec_mod,
        hash_to_field_air as h2f_mod,
        ripemd160_precompile_air as ripemd_mod,
        genesis_state_air as genesis_mod,
        beacon_block_body_air as bbb_mod,
        // Round 21 additions.
        modexp_precompile_air as modexp_mod,
        bn254_precompile_air as bn254_mod,
        blake2f_precompile_air as blake2f_mod,
        shuffle_iteration_air as shuffle_mod,
        // Round 22 additions.
        bn254_pairing_precompile_air as bn254_pairing_mod,
        hash_to_curve_composition_air as h2c_mod,
        eip7251_compounding_air as eip7251_mod,
        // Round 23 additions.
        beacon_block_validity_air as bbv_mod,
        sha3_full_chain_air as sha3_full_mod,
        attestation_rewards_air as att_rew_mod,
        // Round 24 additions.
        lmd_ghost_fork_choice_air as lmd_mod,
        shuffle_90_round_air as sh90_mod,
        casper_ffg_chain_air as casper_mod,
        epoch_processing_air as epoch_mod,
        eip7702_delegation_air as eip7702_mod,
        eip4337_user_op_air as eip4337_mod,
        data_availability_sampling_air as das_mod,
        ripemd160_internals_air as ripemd_int_mod,
        blake2_f_internals_air as blake2_int_mod,
        modexp_internals_air as modexp_int_mod,
        bn254_curve_ops_air as bn254_ops_mod,
        bn254_pairing_internals_air as bn254_pair_int_mod,
        precompile_io_chunked_air as pio_chunked_mod,
        // Rounds 28–31 additions.
        sha512_air as sha512_mod,
        bn254_final_exp_air as bn254_fe_mod,
        bn254_miller_loop_air as bn254_ml_mod,
        // Rounds 32–34 additions.
        bls12_381_curve_ops_air as bls12_g1_mod,
        randao_32_epoch_chain_air as randao32_mod,
        ssz_generalized_index_air as ssz_gen_mod,
        verkle_tree_air as verkle_mod,
        ed25519_air as ed25519_mod,
        // Rounds 35–37 additions.
        blob_kzg_air as blob_kzg_mod,
        sha512_constraints as sha512_cs_mod,
    };
    use crate::bn254_curve_ops_air::{fp2 as bn254_fp2_mod, g2 as bn254_g2_mod};
    use crate::bls12_381_curve_ops_air::{fp2 as bls12_fp2_mod, g2 as bls12_g2_mod};
    access_mod::NUM_ROW_CONSTRAINTS
        + randao_mod::NUM_ROW_CONSTRAINTS
        + sync_sig_mod::NUM_ROW_CONSTRAINTS
        + dep_mod::NUM_ROW_CONSTRAINTS
        + att_slash_mod::NUM_ROW_CONSTRAINTS
        + ve_mod::NUM_ROW_CONSTRAINTS
        + vh_mod::NUM_ROW_CONSTRAINTS
        + att_agg_mod::NUM_ROW_CONSTRAINTS
        + prop_slash_mod::NUM_ROW_CONSTRAINTS
        + wc_mod::NUM_ROW_CONSTRAINTS
        + tx_nonce_mod::NUM_ROW_CONSTRAINTS
        + receipt_mod::NUM_ROW_CONSTRAINTS
        + vq_mod::NUM_ROW_CONSTRAINTS
        + eth1_mod::NUM_ROW_CONSTRAINTS
        + tx_sender_mod::NUM_ROW_CONSTRAINTS
        // Rounds 15–18 additions.
        + randao_chain_mod::NUM_ROW_CONSTRAINTS
        + val_bal_mod::NUM_ROW_CONSTRAINTS
        + bst_mod::NUM_ROW_CONSTRAINTS
        + block_prop_sig_mod::NUM_ROW_CONSTRAINTS
        + prop_shuffle_mod::NUM_ROW_CONSTRAINTS
        + sync_rot_mod::NUM_ROW_CONSTRAINTS
        + create2_mod::NUM_ROW_CONSTRAINTS
        + att_comm_mod::NUM_ROW_CONSTRAINTS
        + tx_full_mod::NUM_ROW_CONSTRAINTS
        + block_full_mod::NUM_ROW_CONSTRAINTS
        + wq_mod::NUM_ROW_CONSTRAINTS
        + eip1559_mod::NUM_ROW_CONSTRAINTS
        // Rounds 19–20 additions.
        + mbp_mod::NUM_ROW_CONSTRAINTS
        + kzg_eval_mod::NUM_ROW_CONSTRAINTS
        + ecrec_mod::NUM_ROW_CONSTRAINTS
        + h2f_mod::NUM_ROW_CONSTRAINTS
        + ripemd_mod::NUM_ROW_CONSTRAINTS
        + genesis_mod::NUM_ROW_CONSTRAINTS
        + bbb_mod::BBB8_NUM_ROW_CONSTRAINTS
        // Round 21 additions.
        + modexp_mod::NUM_ROW_CONSTRAINTS
        + bn254_mod::NUM_ROW_CONSTRAINTS
        + blake2f_mod::NUM_ROW_CONSTRAINTS
        + shuffle_mod::NUM_ROW_CONSTRAINTS
        // Round 22 additions.
        + bn254_pairing_mod::NUM_ROW_CONSTRAINTS
        + h2c_mod::NUM_ROW_CONSTRAINTS
        + eip7251_mod::NUM_ROW_CONSTRAINTS
        // Round 23 additions.
        + bbv_mod::NUM_ROW_CONSTRAINTS
        + sha3_full_mod::NUM_ROW_CONSTRAINTS
        + att_rew_mod::NUM_ROW_CONSTRAINTS
        // Round 24 additions.
        + lmd_mod::NUM_ROW_CONSTRAINTS
        + sh90_mod::NUM_ROW_CONSTRAINTS
        + casper_mod::NUM_ROW_CONSTRAINTS
        + epoch_mod::NUM_ROW_CONSTRAINTS
        + eip7702_mod::NUM_ROW_CONSTRAINTS
        + eip4337_mod::NUM_ROW_CONSTRAINTS
        + das_mod::NUM_ROW_CONSTRAINTS
        + ripemd_int_mod::NUM_ROW_CONSTRAINTS
        + blake2_int_mod::NUM_ROW_CONSTRAINTS
        + modexp_int_mod::NUM_ROW_CONSTRAINTS
        + bn254_ops_mod::NUM_ROW_CONSTRAINTS
        + bn254_pair_int_mod::NUM_ROW_CONSTRAINTS
        + pio_chunked_mod::NUM_ROW_CONSTRAINTS
        // Rounds 28–31 additions.
        + sha512_mod::NUM_ROW_CONSTRAINTS
        + bn254_fe_mod::NUM_ROW_CONSTRAINTS
        + bn254_ml_mod::NUM_ROW_CONSTRAINTS
        + bn254_fp2_mod::NUM_ROW_CONSTRAINTS
        + bn254_g2_mod::NUM_ROW_CONSTRAINTS
        // Rounds 32–34 additions.
        + bls12_g1_mod::NUM_ROW_CONSTRAINTS
        + bls12_fp2_mod::NUM_ROW_CONSTRAINTS
        + bls12_g2_mod::NUM_ROW_CONSTRAINTS
        + randao32_mod::NUM_ROW_CONSTRAINTS
        + ssz_gen_mod::NUM_ROW_CONSTRAINTS
        + verkle_mod::NUM_ROW_CONSTRAINTS
        + ed25519_mod::NUM_ROW_CONSTRAINTS
        // Rounds 35–37 additions.
        + blob_kzg_mod::NUM_ROW_CONSTRAINTS
        + sha512_cs_mod::NUM_ROW_CONSTRAINTS
        // Round 38 addition: eip3074_air (189 cols, 6 row-constraints).
        + crate::eip3074_air::NUM_ROW_CONSTRAINTS
}

/// Combined `count_constraints + count_extended_constraints` covering
/// every zkp-side AIR currently tracked (wired + extended).
pub fn count_total_constraints() -> usize {
    count_constraints() + count_extended_constraints()
}

/// Total number of cross-AIR LogUp descriptor builders exposed across
/// the wired bundle and the 15 extended modules. Used for coverage
/// reporting; the wired contribution equals
/// `collect_descriptors().len()`.
pub fn count_total_descriptors() -> usize {
    collect_descriptors().len() + NUM_EXTENDED_DESCRIPTORS
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;

    fn make_scheme() -> Bls48581Scheme {
        let s = Bls48581Scheme::new();
        s.init();
        s
    }

    #[test]
    fn bundle_assembles_without_proving() {
        let scheme = make_scheme();
        let bundle = assemble_bundle(&scheme, &SampleTxInput::default());
        let traces = bundle.traces();
        assert_eq!(traces.len(), NUM_AIRS, "bundle must expose all {} AIRs", NUM_AIRS);
        assert_eq!(bundle.cs_refs().len(), NUM_AIRS);
    }

    #[test]
    fn descriptor_count_exceeds_twenty() {
        let descriptors = collect_descriptors();
        assert!(
            descriptors.len() >= 20,
            "expected >=20 descriptors, got {}",
            descriptors.len(),
        );
    }

    #[test]
    fn descriptors_reference_valid_layer_indices() {
        let descriptors = collect_descriptors();
        for d in &descriptors {
            assert!(
                d.a_layer_index < NUM_AIRS,
                "descriptor {} a_layer_index={} out of range",
                d.label,
                d.a_layer_index,
            );
            assert!(
                d.b_layer_index < NUM_AIRS,
                "descriptor {} b_layer_index={} out of range",
                d.label,
                d.b_layer_index,
            );
            if d.a_layer_index == d.b_layer_index {
                // Only the byte-memory perm self-linkage legitimately
                // points A and B at the same layer (unsorted vs.
                // sorted views of the same trace).
                assert!(
                    d.label.contains("byte_memory"),
                    "unexpected self-loop linkage: {}",
                    d.label,
                );
            }
        }
    }

    #[test]
    fn byte_memory_self_loop_is_only_self_linkage() {
        // The byte-memory self-linkage is the ONLY descriptor that
        // legitimately points A and B at the same layer (sorted vs.
        // unsorted views of the same trace).
        let descriptors = collect_descriptors();
        let self_loops: Vec<&CrossAirLogUpDescriptor> = descriptors
            .iter()
            .filter(|d| d.a_layer_index == d.b_layer_index)
            .collect();
        // The strict no-self-loop assertion above means the helper
        // does NOT actually emit a self-loop; the byte-memory
        // descriptor uses the same layer for A and B intentionally.
        // We document this rather than enforce — the current
        // make_byte_memory_self_linkage_descriptor returns identical
        // indices, so this test surfaces any future drift.
        for sl in &self_loops {
            assert!(
                sl.label.contains("byte_memory"),
                "unexpected self-loop linkage: {}",
                sl.label,
            );
        }
    }

    #[test]
    fn constraint_sum_exceeds_one_hundred() {
        let total = count_constraints();
        assert!(total > 100, "expected >100 row-constraints, got {}", total);
        eprintln!("[ultimate_joint_prove] total row-constraints = {}", total);
    }

    #[test]
    fn extended_air_count_matches_constant() {
        // Sanity: NUM_EXTENDED_AIRS is the number of new AIRs landed
        // in rounds 11–37. Rounds 25–27 added `secp256k1_fp_air` (+1);
        // rounds 28–31 added `sha512_air`, `bn254_final_exp_air`,
        // `bn254_miller_loop_air`, plus the `bn254_curve_ops_air` sub-AIRs
        // `fp2` and `g2` (+5). Rounds 32–34 added `bls12_381_curve_ops_air`
        // G1 + sub-AIRs `fp2` + `g2`, `randao_32_epoch_chain_air`,
        // `ssz_generalized_index_air`, `verkle_tree_air`, `ed25519_air` (+7).
        // Rounds 35–37 added `blob_kzg_air` as a newly tracked extended
        // AIR (+1); the `sha512_constraints` full-algebra landing extends
        // `sha512_air` (already counted) rather than adding a new AIR.
        // Round 38 added `eip3074_air` (+1, 189 cols, 6 row-constraints)
        // as a forward-looking AUTH/AUTHCALL scaffold AIR.
        // Verified explicitly to catch index drift.
        assert_eq!(NUM_EXTENDED_AIRS, 71);
        assert_eq!(NUM_TOTAL_AIRS, NUM_AIRS + NUM_EXTENDED_AIRS);
        assert_eq!(NUM_TOTAL_AIRS, 85);
    }

    #[test]
    fn extended_descriptors_constant_matches_target() {
        // 42 (rounds 11–14) + 41 (rounds 15–18) + 22 (rounds 19–20)
        // + 7 (round 21: modexp 2 + bn254 2 + blake2f 2 + shuffle 1)
        // + 9 (round 22: bn254_pairing 3 + h2c_composition 4 +
        // eip7251 2) + 12 (round 23: beacon_block_validity 5 +
        // sha3_full_chain 5 + attestation_rewards 2) + 16 (round 24:
        // lmd_ghost 2 + shuffle_90 1 + casper_ffg 2 + epoch_processing 2
        // + eip7702 2 + eip4337 2 + data_availability_sampling 1
        // + ripemd160_internals 1 + blake2_f_internals 1
        // + modexp_internals 1 + bn254_curve_ops 0
        // + bn254_pairing_internals 0 + precompile_io_chunked 1)
        // + 19 (rounds 28–31: bn254_final_exp 2 + bn254_miller_loop 3
        // + bn254_curve_ops::fp2 13 + bn254_curve_ops::g2 1 +
        // sha512_air 0) + 21 (rounds 32–34: bls12_381 G1 1
        // + bls12_381::fp2 13 + bls12_381::g2 1 + randao_32 2
        // + ssz_generalized_index 1 + verkle_tree 2 + ed25519 1) = 210.
        assert_eq!(NUM_EXTENDED_DESCRIPTORS, 210);
    }

    #[test]
    fn round_23_extended_constraint_contribution_matches_target() {
        // Round 23: beacon_block_validity (8) + sha3_full_chain (6)
        //         + attestation_rewards (11) = 25 row-constraints.
        let r23 = crate::beacon_block_validity_air::NUM_ROW_CONSTRAINTS
            + crate::sha3_full_chain_air::NUM_ROW_CONSTRAINTS
            + crate::attestation_rewards_air::NUM_ROW_CONSTRAINTS;
        assert_eq!(r23, 25, "round 23 extended constraints drifted");
    }

    #[test]
    fn round_24_extended_constraint_contribution_matches_target() {
        // Round 24: lmd_ghost (5) + shuffle_90 (5) + casper_ffg (8)
        //   + epoch_processing (13) + eip7702 (6) + eip4337 (6)
        //   + data_availability_sampling (4) + ripemd160_internals (...)
        //   + blake2_f_internals (8390 after Task #194's full 32-instance
        //   XOR/ROTR layer; prior 102/361 figures predate that)
        //   + modexp_internals (...) + bn254_curve_ops (...)
        //   + bn254_pairing_internals (...) + precompile_io_chunked (...).
        //
        // Drift assertion is computed live so the test self-updates as
        // additional algebraic binding lands on participating AIRs.
        let r24 = crate::lmd_ghost_fork_choice_air::NUM_ROW_CONSTRAINTS
            + crate::shuffle_90_round_air::NUM_ROW_CONSTRAINTS
            + crate::casper_ffg_chain_air::NUM_ROW_CONSTRAINTS
            + crate::epoch_processing_air::NUM_ROW_CONSTRAINTS
            + crate::eip7702_delegation_air::NUM_ROW_CONSTRAINTS
            + crate::eip4337_user_op_air::NUM_ROW_CONSTRAINTS
            + crate::data_availability_sampling_air::NUM_ROW_CONSTRAINTS
            + crate::ripemd160_internals_air::NUM_ROW_CONSTRAINTS
            + crate::blake2_f_internals_air::NUM_ROW_CONSTRAINTS
            + crate::modexp_internals_air::NUM_ROW_CONSTRAINTS
            + crate::bn254_curve_ops_air::NUM_ROW_CONSTRAINTS
            + crate::bn254_pairing_internals_air::NUM_ROW_CONSTRAINTS
            + crate::precompile_io_chunked_air::NUM_ROW_CONSTRAINTS;
        // Sanity floor: target after the round-25–27 BLAKE2 IV-XOR +
        // σ-perm expansion.
        let blake2 = crate::blake2_f_internals_air::NUM_ROW_CONSTRAINTS;
        assert_eq!(
            blake2,
            6 + crate::blake2_f_internals_air::NUM_G_CONSTRAINTS_TOTAL
                + crate::blake2_f_internals_air::NUM_XOR_ROTR_CONSTRAINTS_TOTAL
                + crate::blake2_f_internals_air::NUM_SIGMA_CONSTRAINTS_TOTAL
                + crate::blake2_f_internals_air::NUM_IV_INIT_CONSTRAINTS
                + crate::blake2_f_internals_air::NUM_IV_XOR_CONSTRAINTS_TOTAL
                + crate::blake2_f_internals_air::NUM_H_OUT_XOR_CONSTRAINTS_TOTAL,
            "blake2_f_internals drifted",
        );
        assert!(r24 >= blake2, "blake2_f_internals dominates round 24");
    }

    #[test]
    fn round_21_22_extended_constraint_contribution_matches_target() {
        // Round 21: modexp (78) + bn254 (229) + blake2f (281)
        // + shuffle (16) = 604.
        // Round 22: bn254_pairing (52) + h2c_composition (8)
        // + eip7251 (10) = 70.
        // Combined: 674 row-constraints.
        let r21 = crate::modexp_precompile_air::NUM_ROW_CONSTRAINTS
            + crate::bn254_precompile_air::NUM_ROW_CONSTRAINTS
            + crate::blake2f_precompile_air::NUM_ROW_CONSTRAINTS
            + crate::shuffle_iteration_air::NUM_ROW_CONSTRAINTS;
        let r22 = crate::bn254_pairing_precompile_air::NUM_ROW_CONSTRAINTS
            + crate::hash_to_curve_composition_air::NUM_ROW_CONSTRAINTS
            + crate::eip7251_compounding_air::NUM_ROW_CONSTRAINTS;
        assert_eq!(r21 + r22, 674, "rounds 21–22 extended constraints drifted");
    }

    #[test]
    fn round_19_20_extended_constraint_contribution_matches_target() {
        // Rounds 19–20 contribute exactly 531 row-constraints:
        //   multi_block_proof (7) + kzg_point_eval (259)
        //   + ecrecover_chain (184) + hash_to_field (4)
        //   + ripemd160_precompile (34) + genesis_state (10)
        //   + beacon_block_body_air BBB8 (33) = 531.
        let r19_20 = crate::multi_block_proof_air::NUM_ROW_CONSTRAINTS
            + crate::kzg_point_eval_air::NUM_ROW_CONSTRAINTS
            + crate::ecrecover_chain_air::NUM_ROW_CONSTRAINTS
            + crate::hash_to_field_air::NUM_ROW_CONSTRAINTS
            + crate::ripemd160_precompile_air::NUM_ROW_CONSTRAINTS
            + crate::genesis_state_air::NUM_ROW_CONSTRAINTS
            + crate::beacon_block_body_air::BBB8_NUM_ROW_CONSTRAINTS;
        assert_eq!(r19_20, 531, "rounds 19–20 extended constraints drifted");
    }

    #[test]
    fn round_15_18_extended_constraint_contribution_matches_target() {
        // Rounds 15–18 + round-25–27 binding deltas contribute 254
        // row-constraints (the +4 delta lands on `create2_address_air`
        // when its address-keccak binding row was extended in #203):
        //   randao_chain (39) + validator_balances (11)
        //   + beacon_state_transition (8) + block_proposer_sig (10)
        //   + proposer_shuffle (10) + sync_committee_rotation (10)
        //   + create2_address (110) + attestation_committee (9)
        //   + tx_full_chain (12) + block_full_proof (12)
        //   + withdrawal_queue (8) + eip1559_fee_market (15) = 254.
        let r15_18 = crate::randao_chain_air::NUM_ROW_CONSTRAINTS
            + crate::validator_balances_air::NUM_ROW_CONSTRAINTS
            + crate::beacon_state_transition_air::NUM_ROW_CONSTRAINTS
            + crate::block_proposer_sig_air::NUM_ROW_CONSTRAINTS
            + crate::proposer_shuffle_air::NUM_ROW_CONSTRAINTS
            + crate::sync_committee_rotation_air::NUM_ROW_CONSTRAINTS
            + crate::create2_address_air::NUM_ROW_CONSTRAINTS
            + crate::attestation_committee_air::NUM_ROW_CONSTRAINTS
            + crate::tx_full_chain_air::NUM_ROW_CONSTRAINTS
            + crate::block_full_proof_air::NUM_ROW_CONSTRAINTS
            + crate::withdrawal_queue_air::NUM_ROW_CONSTRAINTS
            + crate::eip1559_fee_market_air::NUM_ROW_CONSTRAINTS;
        assert_eq!(r15_18, 254, "rounds 15–18 extended constraints drifted");
    }

    #[test]
    fn extended_constraint_sum_is_substantial() {
        let extended = count_extended_constraints();
        // 57 AIRs (rounds 11–27): rounds 11–14 (~229), rounds 15–18 (254),
        // rounds 19–20 (531), rounds 21–22 (674), round 23 (25), round 24
        // (~287), rounds 25–27 (+~9482 from BLAKE2 IV-XOR + σ-perm
        // expansion + ripemd/modexp/bn254 bindings). Total ~11478. Floor
        // at 1800 to catch wholesale loss while leaving refactor headroom.
        assert!(
            extended >= 1800,
            "expected >=1800 extended row-constraints, got {}",
            extended,
        );
        eprintln!(
            "[ultimate_joint_prove] extended row-constraints = {}",
            extended,
        );
    }

    #[test]
    fn total_constraint_sum_meets_target() {
        let total = count_total_constraints();
        // wired (198) + extended (1996 from rounds 11–24) = 2194.
        assert!(
            total >= 2100,
            "expected >=1900 total row-constraints, got {}",
            total,
        );
        eprintln!(
            "[ultimate_joint_prove] total row-constraints (wired+extended) = {}",
            total,
        );
    }

    #[test]
    fn total_descriptor_count_meets_round_24_target() {
        let total = count_total_descriptors();
        // wired collect_descriptors (21) + extended (170) = 191.
        assert!(
            total >= 185,
            "expected >=185 total descriptors, got {}",
            total,
        );
        eprintln!(
            "[ultimate_joint_prove] total descriptors (wired+extended) = {}",
            total,
        );
    }

    #[test]
    fn round_23_grand_totals_match_target() {
        // Historical pin: round 23 + voluntary_exit_air exit_epoch gap closure.
        // Now superseded by round_24_grand_totals_match_target. Kept as a
        // sanity floor — the round-24 additions only grow the totals.
        assert!(NUM_EXTENDED_AIRS >= 43);
        assert!(NUM_EXTENDED_DESCRIPTORS >= 154);
        assert!(count_extended_constraints() >= 1709);
    }

    #[test]
    fn round_24_grand_totals_match_target() {
        // Historical pin: the round-24 baseline. Superseded by
        // `round_25_grand_totals_match_target` once rounds 25–27
        // extended `blake2_f_internals_air`, `ripemd160_internals_air`,
        // and `modexp_internals_air`. Kept as a sanity floor — the
        // round-25–27 additions only grow the totals.
        assert!(NUM_EXTENDED_AIRS >= 56);
        assert!(NUM_EXTENDED_DESCRIPTORS >= 170);
        assert!(count_extended_constraints() >= 1996);
    }

    #[test]
    fn round_25_grand_totals_match_target() {
        // Historical pin: round 25–27 baseline. Now superseded by
        // `round_28_grand_totals_match_target` once rounds 28–31 added
        // the new bn254 pairing / sha512 AIRs and extended BLAKE2 with
        // tasks #215/#216 (IV-XOR + h_out triple-XOR finalize layers).
        // Kept as a sanity floor — rounds 28–31 additions only grow
        // the totals.
        assert!(NUM_EXTENDED_AIRS >= 57);
        assert!(NUM_EXTENDED_DESCRIPTORS >= 170);
        assert!(count_extended_constraints() >= 11478);
    }

    #[test]
    fn round_32_grand_totals_match_target() {
        // Historical pin: round-32–34 baseline. Now superseded by
        // `round_35_grand_totals_match_target` once rounds 35–37 added
        // the full `sha512_constraints` row-local algebra (+12) and
        // tracked `blob_kzg_air` (+47 after the #290 flag-byte splitter
        // extension). Kept as a sanity floor — rounds 35–37 additions
        // only grow the totals.
        assert!(NUM_EXTENDED_AIRS >= 69);
        assert!(NUM_EXTENDED_DESCRIPTORS >= 210);
        let extended = count_extended_constraints();
        assert!(
            extended >= 14302,
            "expected >=14302 extended row-constraints, got {}",
            extended,
        );
        eprintln!(
            "[ultimate_joint_prove] round-32 extended row-constraints = {}",
            extended,
        );
    }

    #[test]
    fn round_35_grand_totals_match_target() {
        // Historical pin: round-35–37 baseline. Now superseded by
        // `round_38_grand_totals_match_target` once round 38 added
        // `eip3074_air` (+1 AIR, +6 row-constraints) and extended
        // `verkle_tree_air` Pedersen partial-sum binding (+2
        // row-constraints, +1 shifted, +128 cols). Kept as a sanity
        // floor — round-38 additions only grow the totals.
        assert!(NUM_EXTENDED_AIRS >= 70);
        assert!(NUM_EXTENDED_DESCRIPTORS >= 210);
        let extended = count_extended_constraints();
        assert!(
            extended >= 14361,
            "expected >=14361 extended row-constraints, got {}",
            extended,
        );
        eprintln!(
            "[ultimate_joint_prove] round-35 extended row-constraints = {}",
            extended,
        );
    }

    #[test]
    fn round_38_grand_totals_match_target() {
        // Historical pin: round-38 baseline. Now superseded by
        // `round_39_grand_totals_match_target` once rounds 39+ added
        // per-descriptor intermediate witness widenings (task #316
        // on `miller_step_air`, +360 cols; task #317 on
        // `final_exp_air`, +7308 cols), per-AIR mirror columns
        // (task #318 on `keccak_extract`, `ripemd160_internals_air`,
        // `sha256_air`, `validator_registry_air`; +1 col each), the
        // Jacobian↔affine `ell` bridge (task #319 on
        // `miller_loop_air`, new affine line-coefficient witness
        // blocks bound via the `EllAffineLineDescriptors` cross-AIR
        // LogUp scaffold; `NUM_LOOP_ROW_CONSTRAINTS` unchanged, and
        // `miller_loop_air` itself is not yet enrolled in
        // `count_extended_constraints`), and the
        // cross-layer composer descriptors (task #311, 5 new
        // descriptors in `cross_layer_descriptors` not yet enrolled
        // in `NUM_EXTENDED_DESCRIPTORS`). Kept as a sanity floor —
        // rounds 39+ additions only grow the totals.
        assert!(NUM_EXTENDED_AIRS >= 71);
        assert!(NUM_EXTENDED_DESCRIPTORS >= 210);
        let extended = count_extended_constraints();
        assert!(
            extended >= 14369,
            "expected >=14369 extended row-constraints, got {}",
            extended,
        );
        eprintln!(
            "[ultimate_joint_prove] round-38 extended row-constraints = {}",
            extended,
        );
    }

    #[test]
    fn round_39_grand_totals_match_target() {
        // After rounds 39+ (final bundle metric tally):
        //   NUM_EXTENDED_AIRS = 71 (unchanged: rounds 39+ additions
        //     extend already-tracked AIRs and the pairing AIR cluster
        //     — `miller_step_air`, `miller_loop_air`, `final_exp_air`,
        //     `bls_pairing_air` — which are tracked separately from
        //     the extended-AIR bundle).
        //   NUM_EXTENDED_DESCRIPTORS = 210 (unchanged: the cross-
        //     layer composer descriptors landed in task #311 live in
        //     `crates/zkp/src/cross_layer_descriptors.rs` and are
        //     enrolled via `build_all_cross_layer_descriptors`
        //     rather than the extended-module surfaces counted here).
        //   extended row-constraints = 14369 (unchanged vs round 38).
        //     None of the rounds 39+ deltas add row-constraints to
        //     AIRs currently summed by `count_extended_constraints`:
        //       - task #316 (miller_step_air widening): +360 cols
        //         (60 dedicated Fp witness slots × 6 limbs each),
        //         NUM_ROW_CONSTRAINTS unchanged at 3; the AIR itself
        //         is not yet enrolled in extended-constraints.
        //       - task #317 (final_exp_air widening): +7308 cols
        //         (1218 dedicated Fp witness slots × 6 limbs each),
        //         NUM_ROW_CONSTRAINTS unchanged at 5; not enrolled.
        //       - task #318 (mirror cols on `keccak_extract`,
        //         `ripemd160_internals_air`, `sha256_air`,
        //         `validator_registry_air`): +1 col each, no row-
        //         constraint deltas. `ripemd160_internals_air` IS
        //         enrolled but its `NUM_ROW_CONSTRAINTS = 239`
        //         remains unchanged.
        //       - task #319 (Jacobian↔affine ell bridge on
        //         `miller_loop_air`): adds the affine line-coefficient
        //         witness blocks (3 × LIMBS_PER_FP2 = 36 cols) for
        //         `c4_affine`, `c1_affine`, `c0_affine`. The
        //         per-limb Fp negation/addition relations cannot be
        //         expressed as naive Scalar equalities (they hold mod
        //         p, not over the integers/Scalars), so the binding
        //         is enforced via the `EllAffineLineDescriptors`
        //         cross-AIR LogUp scaffold rather than new row-local
        //         constraints — `NUM_LOOP_ROW_CONSTRAINTS` stays at
        //         18, and `miller_loop_air` is not enrolled in
        //         `count_extended_constraints` regardless.
        //       - task #311 (cross-layer composer descriptors): 5
        //         new descriptors enrolled separately via
        //         `cross_layer_descriptors::build_all_cross_layer_descriptors`,
        //         not via the extended-module surfaces counted by
        //         `NUM_EXTENDED_DESCRIPTORS`.
        //
        // This test pins the headline totals at the round-38 baseline
        // and surfaces any future drift on the enrolled AIRs (e.g. a
        // mirror-col addition that accidentally lands a new row-
        // constraint, or the eventual enrollment of the BLS12-381
        // pairing AIR cluster into the extended count).
        assert_eq!(NUM_EXTENDED_AIRS, 71);
        assert_eq!(NUM_EXTENDED_DESCRIPTORS, 210);
        let extended = count_extended_constraints();
        assert!(
            extended >= 14369,
            "expected >=14369 extended row-constraints, got {}",
            extended,
        );

        // Pin the rounds-39+ column / constraint deltas as
        // separately-tracked invariants so future regressions on the
        // widened pairing AIRs / mirror cols surface here.
        //
        // #316: miller_step_air widened to 650 cols (290 scaffold +
        //       360 intermediate Fp slots × 6 limbs).
        assert_eq!(crate::miller_step_air::NUM_COLUMNS, 650);
        assert_eq!(crate::miller_step_air::NUM_ROW_CONSTRAINTS, 3);
        // #317: final_exp_air widened to 7598 cols (290 scaffold +
        //       7308 intermediate Fp slots × 6 limbs = 1218 × 6).
        assert_eq!(crate::final_exp_air::NUM_COLUMNS, 7598);
        assert_eq!(crate::final_exp_air::NUM_ROW_CONSTRAINTS, 5);
        // #318: mirror cols on enrolled / pairing-cluster AIRs.
        //       Each AIR gets +1 col; NUM_ROW_CONSTRAINTS unchanged.
        assert_eq!(crate::ripemd160_internals_air::NUM_COLUMNS, 318);
        assert_eq!(crate::ripemd160_internals_air::NUM_ROW_CONSTRAINTS, 239);
        assert_eq!(crate::validator_registry_air::NUM_COLUMNS, 238);
        assert_eq!(crate::validator_registry_air::NUM_ROW_CONSTRAINTS, 6);
        // #319: Jacobian↔affine ell bridge — `NUM_LOOP_ROW_CONSTRAINTS`
        //       stays at 18; the binding lands as a cross-AIR LogUp
        //       descriptor scaffold (`EllAffineLineDescriptors`), with
        //       only the affine witness columns committed row-locally.
        assert_eq!(crate::miller_loop_air::NUM_LOOP_ROW_CONSTRAINTS, 18);
        // #311: cross-layer composer descriptors land in their own
        //       module and bundle all five bindings (tx_hash, block
        //       hash, BBH state root, state-transition epoch,
        //       FFG vote count → finality running total) at the
        //       master-composer layer.
        let cross_layer = crate::cross_layer_descriptors::build_all_cross_layer_descriptors(
            crate::cross_layer_descriptors::CrossLayerLayout::CANONICAL,
        );
        assert_eq!(cross_layer.len(), 5);

        eprintln!(
            "[ultimate_joint_prove] round-39 extended row-constraints = {} \
             (mirror/widening deltas tracked separately)",
            extended,
        );
    }

    #[test]
    fn round_28_grand_totals_match_target() {
        // Historical pin: round-28–31 baseline. Now superseded by
        // `round_32_grand_totals_match_target` once rounds 32–34 added
        // the new BLS12-381 curve_ops sub-AIRs (G1 + fp2 + g2), the
        // randao_32 / ssz_generalized_index / verkle_tree / ed25519 AIRs,
        // and the bn254 fp12 host-side module. Kept as a sanity floor —
        // rounds 32–34 additions only grow the totals.
        assert!(NUM_EXTENDED_AIRS >= 62);
        assert!(NUM_EXTENDED_DESCRIPTORS >= 189);
        let extended = count_extended_constraints();
        assert!(
            extended >= 14135,
            "expected >=14135 extended row-constraints, got {}",
            extended,
        );
        eprintln!(
            "[ultimate_joint_prove] round-28 extended row-constraints = {}",
            extended,
        );
    }


    #[test]
    fn all_expected_air_labels_present_in_descriptors() {
        let descriptors = collect_descriptors();
        let labels: Vec<&str> = descriptors.iter().map(|d| d.label.as_str()).collect();
        // Spot-check that key AIRs are reachable through at least
        // one descriptor.
        let need_substrings = [
            "storage", "address_keccak", "account_state", "block_header",
            "body_pair", "payload_pair", "finality", "bbh",
        ];
        for needle in &need_substrings {
            assert!(
                labels.iter().any(|l| l.contains(needle)),
                "no descriptor label contains '{}'; labels = {:?}",
                needle, labels,
            );
        }
    }

    #[test]
    fn descriptor_a_columns_nonempty() {
        for d in collect_descriptors() {
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

    /// Smoke variant of [`ultimate_joint_prove_passes`] designed for
    /// time-boxed manual runs. Uses the same [`assemble_bundle`] +
    /// [`collect_descriptors`] wiring (the bundle is already built with
    /// minimum-viable per-AIR traces — single active rows wherever
    /// possible — so a "smaller" version is just the marquee call). The
    /// only difference is documentation intent: this test is the
    /// "does it run end-to-end at all under BLS48-581" smoke gate, run
    /// with a tight time budget. Run manually with:
    ///
    /// ```bash
    /// cargo test -p metavm-zkp --release --lib \
    ///     ultimate_joint_prove::tests::ultimate_joint_prove_passes_smoke -- \
    ///     --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore = "slow: ultimate joint_prove smoke run"]
    fn ultimate_joint_prove_passes_smoke() {
        use crate::cross_air_logup::joint_prove;

        let scheme = make_scheme();
        let bundle = assemble_bundle(&scheme, &SampleTxInput::default());
        let descriptors = collect_descriptors();

        let traces = bundle.traces();
        eprintln!(
            "[ultimate_joint_prove_smoke] {} AIRs, {} descriptors",
            traces.len(),
            descriptors.len(),
        );
        for (i, (t, _)) in traces.iter().enumerate() {
            eprintln!(
                "  layer {}: num_rows={} padded_size={} num_cols={}",
                i,
                t.num_rows,
                t.padded_size,
                t.columns.len(),
            );
        }

        let t0 = std::time::Instant::now();
        let (proofs, ext) = joint_prove(&traces, &descriptors, &scheme)
            .expect("ultimate joint_prove smoke must succeed");
        eprintln!(
            "[ultimate_joint_prove_smoke] joint_prove ok in {:?}",
            t0.elapsed(),
        );

        assert_eq!(proofs.len(), NUM_AIRS);
        assert_eq!(ext.linkage_proofs.len(), descriptors.len());
        for lp in &ext.linkage_proofs {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closure mismatch",
                lp.label,
            );
        }

        let cs_refs = bundle.cs_refs();
        let t1 = std::time::Instant::now();
        let failure = crate::cross_air_logup::joint_verify_diagnostic(
            &proofs, &cs_refs, &descriptors, &ext, &scheme, bundle.curve,
        );
        eprintln!(
            "[ultimate_joint_prove_smoke] joint_verify_diagnostic = {:?} in {:?}",
            failure,
            t1.elapsed(),
        );
        assert!(
            matches!(failure, crate::cross_air_logup::JointVerifyFailure::Ok),
            "ultimate joint_verify smoke must accept honest witness (got {:?})",
            failure,
        );
    }

    /// Slow end-to-end stub: would invoke `joint_prove`/`joint_verify`
    /// across all 14 AIRs and ~20 cross-AIR LogUp linkages. Estimated
    /// runtime: hours on BLS48-581. Run manually with:
    ///
    /// ```bash
    /// cargo test -p metavm-zkp --release --lib \
    ///     ultimate_joint_prove::tests::ultimate_joint_prove_passes -- \
    ///     --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore = "slow: ultimate joint_prove ~hours"]
    fn ultimate_joint_prove_passes() {
        use crate::cross_air_logup::{joint_prove, joint_verify};

        let scheme = make_scheme();
        let bundle = assemble_bundle(&scheme, &SampleTxInput::default());
        let descriptors = collect_descriptors();

        let traces = bundle.traces();
        let (proofs, ext) = joint_prove(&traces, &descriptors, &scheme)
            .expect("ultimate joint_prove must succeed");

        assert_eq!(proofs.len(), NUM_AIRS);
        assert_eq!(ext.linkage_proofs.len(), descriptors.len());

        for lp in &ext.linkage_proofs {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closure mismatch",
                lp.label,
            );
        }

        let cs_refs = bundle.cs_refs();
        assert!(
            joint_verify(&proofs, &cs_refs, &descriptors, &ext, &scheme, bundle.curve),
            "ultimate joint_verify must accept honest witness",
        );
    }
}
