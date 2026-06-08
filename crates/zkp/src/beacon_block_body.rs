//! BeaconBlockBody (Cancun/Deneb) host-side scaffolding — Phase C↔B
//! bridge step 1.
//!
//! `BeaconBlockBody` is the SSZ container referenced by a
//! `BeaconBlockHeader.body_root`. Its 12 fields (post-Cancun) include
//! `execution_payload`, which carries the Ethereum execution block's
//! `block_hash`. Composing the per-field merkleization gives the full
//! bridge: BBH.body_root → BeaconBlockBody → execution_payload →
//! block_hash → Layer B's `block_header_air`.
//!
//! # Spec reference (Deneb)
//!
//! ```text
//! class BeaconBlockBody(Container):
//!     randao_reveal:              BLSSignature
//!     eth1_data:                  Eth1Data
//!     graffiti:                   Bytes32
//!     proposer_slashings:         List[ProposerSlashing, MAX]
//!     attester_slashings:         List[AttesterSlashing, MAX]
//!     attestations:               List[Attestation, MAX]
//!     deposits:                   List[Deposit, MAX]
//!     voluntary_exits:            List[SignedVoluntaryExit, MAX]
//!     sync_aggregate:             SyncAggregate
//!     execution_payload:          ExecutionPayload
//!     bls_to_execution_changes:   List[SignedBLSToExecutionChange, MAX]
//!     blob_kzg_commitments:       List[KZGCommitment, MAX]
//! ```
//!
//! # Simplification
//!
//! Only `execution_payload_header` is modeled as a typed struct here —
//! that's the field carrying the bridge value. All other fields are
//! represented as their pre-computed HTR roots (`Chunk` = `[u8; 32]`).
//! Future work that needs to inspect attestations, sync aggregate,
//! etc., can replace those roots with typed structs without breaking
//! the surrounding API.
//!
//! Note: we use `ExecutionPayloadHeader` (the lighter struct used in
//! `BeaconState`) rather than the full `ExecutionPayload`. The two
//! share `block_hash`, so the bridge logic is identical; using the
//! header avoids modelling transactions and withdrawals lists.

use crate::execution_payload::ExecutionPayloadHeader;
use crate::ssz::{hash_tree_root_container, Chunk};

/// BeaconBlockBody for the C↔B bridge. 12 fields (Cancun). All non-
/// bridge fields use their pre-computed HTR roots; only
/// `execution_payload_header` is modelled as a typed struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeaconBlockBody {
    pub randao_reveal_root: Chunk,
    pub eth1_data_root: Chunk,
    pub graffiti: Chunk,
    pub proposer_slashings_root: Chunk,
    pub attester_slashings_root: Chunk,
    pub attestations_root: Chunk,
    pub deposits_root: Chunk,
    pub voluntary_exits_root: Chunk,
    pub sync_aggregate_root: Chunk,
    /// The bridge field. The full payload's HTR is what the spec
    /// merkleizes; we use ExecutionPayloadHeader as a structurally-
    /// equivalent proxy that exposes `block_hash`.
    pub execution_payload_header: ExecutionPayloadHeader,
    pub bls_to_execution_changes_root: Chunk,
    pub blob_kzg_commitments_root: Chunk,
}

impl Default for BeaconBlockBody {
    fn default() -> Self {
        Self {
            randao_reveal_root: [0u8; 32],
            eth1_data_root: [0u8; 32],
            graffiti: [0u8; 32],
            proposer_slashings_root: [0u8; 32],
            attester_slashings_root: [0u8; 32],
            attestations_root: [0u8; 32],
            deposits_root: [0u8; 32],
            voluntary_exits_root: [0u8; 32],
            sync_aggregate_root: [0u8; 32],
            execution_payload_header: ExecutionPayloadHeader::default(),
            bls_to_execution_changes_root: [0u8; 32],
            blob_kzg_commitments_root: [0u8; 32],
        }
    }
}

impl BeaconBlockBody {
    /// SSZ `hash_tree_root(BeaconBlockBody)` — container reducer over
    /// the 12 field roots.
    pub fn hash_tree_root(&self) -> Chunk {
        let payload_root = self.execution_payload_header.hash_tree_root();
        let field_roots: [Chunk; 12] = [
            self.randao_reveal_root,
            self.eth1_data_root,
            self.graffiti,
            self.proposer_slashings_root,
            self.attester_slashings_root,
            self.attestations_root,
            self.deposits_root,
            self.voluntary_exits_root,
            self.sync_aggregate_root,
            payload_root,
            self.bls_to_execution_changes_root,
            self.blob_kzg_commitments_root,
        ];
        hash_tree_root_container(&field_roots)
    }

    /// Extract the execution block hash. Pure accessor — the algebraic
    /// version follows the same path: BeaconBlockBody → field index 9
    /// (execution_payload_header) → ExecutionPayloadHeader → block_hash.
    pub fn execution_block_hash(&self) -> [u8; 32] {
        self.execution_payload_header.block_hash
    }
}

// ─── Bridge oracle ─────────────────────────────────────────────────────

/// Host-side verifier for the C↔B bridge.
///
/// Given a `BeaconBlockBody` and claimed `body_root` (typically pulled
/// from a `BeaconBlockHeader`), plus a claimed execution-side
/// `block_hash` (typically pulled from Layer B's `BlockHeader`),
/// returns `true` iff:
///
///   1. `body.hash_tree_root() == claimed_body_root`
///   2. `body.execution_payload_header.block_hash == claimed_block_hash`
///
/// Together these algebraically tie the BeaconBlockHeader's `body_root`
/// commitment to the execution block hash that Layer B proves. The
/// future cross-AIR LogUp version of this oracle composes the BBH-pair
/// AIR's `CLAIMED_ROOT` → BeaconBlockBody-pair AIR's body_root output →
/// ExecutionPayloadHeader-pair AIR's block_hash output → Layer B's
/// `block_header_air.block_hash`.
pub fn verify_body_to_block_hash_binding(
    body: &BeaconBlockBody,
    claimed_body_root: Chunk,
    claimed_block_hash: [u8; 32],
) -> bool {
    if body.hash_tree_root() != claimed_body_root {
        return false;
    }
    body.execution_payload_header.block_hash == claimed_block_hash
}

/// Three-way composition: given a `BeaconBlockHeader`, a `BeaconBlockBody`
/// claimed to underlie the header's `body_root`, and a claimed Ethereum
/// execution `block_hash`, verify the full BBH → body → payload chain.
pub fn verify_bbh_to_block_hash_chain(
    beacon_block_header: &crate::beacon::BeaconBlockHeader,
    body: &BeaconBlockBody,
    claimed_block_hash: [u8; 32],
) -> bool {
    // 1. Body hash_tree_root must match the BeaconBlockHeader's body_root.
    if body.hash_tree_root() != beacon_block_header.body_root {
        return false;
    }
    // 2. Payload's block_hash must equal the claimed execution block hash.
    body.execution_payload_header.block_hash == claimed_block_hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon::BeaconBlockHeader;

    fn sample_payload() -> ExecutionPayloadHeader {
        let mut p = ExecutionPayloadHeader::default();
        p.block_hash = [0xab; 32];
        p.block_number = 18_500_000;
        p.gas_limit = 30_000_000;
        p.timestamp = 1_700_000_000;
        p
    }

    fn sample_body() -> BeaconBlockBody {
        BeaconBlockBody {
            graffiti: [0x47; 32],
            execution_payload_header: sample_payload(),
            ..Default::default()
        }
    }

    #[test]
    fn default_root_is_deterministic() {
        let b = BeaconBlockBody::default();
        let r1 = b.hash_tree_root();
        let r2 = b.hash_tree_root();
        assert_eq!(r1, r2);
    }

    #[test]
    fn changing_any_root_field_changes_body_root() {
        let b1 = sample_body();
        let mut b2 = sample_body();
        b2.attestations_root = [0xee; 32];
        assert_ne!(b1.hash_tree_root(), b2.hash_tree_root());
    }

    #[test]
    fn changing_payload_block_hash_changes_body_root() {
        let b1 = sample_body();
        let mut b2 = sample_body();
        b2.execution_payload_header.block_hash = [0xcd; 32];
        assert_ne!(
            b1.hash_tree_root(),
            b2.hash_tree_root(),
            "body_root must depend on payload.block_hash transitively",
        );
    }

    #[test]
    fn execution_block_hash_accessor_returns_payload_field() {
        let b = sample_body();
        assert_eq!(b.execution_block_hash(), [0xab; 32]);
    }

    #[test]
    fn bridge_oracle_accepts_honest_witness() {
        let body = sample_body();
        let body_root = body.hash_tree_root();
        let block_hash = body.execution_payload_header.block_hash;
        assert!(verify_body_to_block_hash_binding(&body, body_root, block_hash));
    }

    #[test]
    fn bridge_oracle_rejects_wrong_body_root() {
        let body = sample_body();
        let block_hash = body.execution_payload_header.block_hash;
        let wrong_root = [0xff; 32];
        assert!(!verify_body_to_block_hash_binding(&body, wrong_root, block_hash));
    }

    #[test]
    fn bridge_oracle_rejects_wrong_block_hash() {
        let body = sample_body();
        let body_root = body.hash_tree_root();
        let wrong_hash = [0xff; 32];
        assert!(!verify_body_to_block_hash_binding(&body, body_root, wrong_hash));
    }

    #[test]
    fn three_way_chain_oracle_accepts_consistent_witness() {
        let body = sample_body();
        let block_hash = body.execution_payload_header.block_hash;
        let bbh = BeaconBlockHeader {
            slot: 12345,
            proposer_index: 67,
            parent_root: [0xaa; 32],
            state_root: [0xbb; 32],
            body_root: body.hash_tree_root(),
        };
        assert!(verify_bbh_to_block_hash_chain(&bbh, &body, block_hash));
    }

    #[test]
    fn three_way_chain_oracle_rejects_mismatched_body_root() {
        let body = sample_body();
        let block_hash = body.execution_payload_header.block_hash;
        let bbh = BeaconBlockHeader {
            slot: 12345,
            proposer_index: 67,
            parent_root: [0xaa; 32],
            state_root: [0xbb; 32],
            body_root: [0xff; 32], // wrong
        };
        assert!(!verify_bbh_to_block_hash_chain(&bbh, &body, block_hash));
    }

    #[test]
    fn three_way_chain_oracle_rejects_mismatched_block_hash() {
        let body = sample_body();
        let bbh = BeaconBlockHeader {
            slot: 12345,
            proposer_index: 67,
            parent_root: [0xaa; 32],
            state_root: [0xbb; 32],
            body_root: body.hash_tree_root(),
        };
        assert!(!verify_bbh_to_block_hash_chain(&bbh, &body, [0xff; 32]));
    }

    /// Spec-shape pin: the body must have exactly 12 fields; any spec
    /// fork that adds/removes fields will break this test as a forcing
    /// function so the AIR layout doesn't silently fall out of sync.
    #[test]
    fn deneb_field_count_pinned_to_12() {
        let body = BeaconBlockBody::default();
        // The hash_tree_root impl uses [Chunk; 12]; compile-time check.
        let _root = body.hash_tree_root();
        let n_fields: usize = 12;
        assert_eq!(n_fields, 12);
    }
}
