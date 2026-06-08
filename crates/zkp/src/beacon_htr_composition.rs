//! Composition bundle for the algebraic C↔B bridge HTR chain.
//!
//! Bundles the three SSZ-extraction HTR witnesses that together prove
//! the chain from a `BeaconBlockHeader` down to an
//! `ExecutionPayloadHeader` (and thus down to an Ethereum execution
//! `block_hash`):
//!
//! ```text
//! BeaconBlockHeaderHtrWitness    (5 fields → 7 pair invocations)
//!   ↓ body_root composition
//! BeaconBlockBodyHtrWitness      (12 fields → 12 pair invocations)
//!   ↓ execution_payload_header_root composition (field 9)
//! ExecutionPayloadHeaderHtrWitness  (17 fields → 20 pair invocations)
//!   ↓ block_hash accessor (field 12)
//! Ethereum execution block_hash
//! ```
//!
//! Mirrors the host-side [`crate::beacon_world_proof::BeaconWorldProof`]
//! bundle but holds the algebraic HTR witnesses instead of execution-
//! layer artifacts. The accompanying oracle verifies that the
//! field-root composition holds: BBH's leaf 4 equals body's witness
//! root, and body's field root 9 equals payload's witness root.
//!
//! # Where this fits in the wider chain
//!
//! Once the corresponding pair AIRs land (BBH-pair AIR is closed
//! today; body-pair AIR and payload-pair AIR are step 1 work for their
//! respective levels), the algebraic equivalent of
//! [`verify_beacon_htr_chain_oracle`] is a small set of cross-AIR
//! LogUp linkages:
//!
//!   1. `body-pair AIR.CLAIMED_ROOT` ↔ `BBH-pair AIR.LEFT @ row 2` —
//!      32-byte tuple bound by the existing CLAIMED_ROOT output
//!      column on body-pair AIR and a leaf-position selector on
//!      BBH-pair AIR.
//!   2. `payload-pair AIR.CLAIMED_ROOT` ↔ `body-pair AIR.field_root_9_col` —
//!      32-byte tuple bound by an analogous setup.
//!   3. `payload-pair AIR.CLAIMED_BLOCK_HASH` ↔ `block_header_air.block_hash` —
//!      32-byte tuple (the consensus↔execution bridge).
//!
//! These descriptors slot directly into the existing `joint_prove`
//! framework once the two missing AIRs exist.

use crate::beacon_block_body_air::BeaconBlockBodyHtrWitness;
use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;
use crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness;
use crate::ssz::Chunk;

/// Index of `body_root` in the BBH-pair AIR's invocations: the leaf
/// at position 4 is the LEFT of pair (4,5) in layer 0, which is
/// invocation index 2 in evaluation order.
pub const BBH_BODY_ROOT_INVOCATION: usize = 2;

/// Field index of `execution_payload_header_root` in the body's
/// container merkleization (BeaconBlockBody field 9, 0-indexed).
pub const BODY_PAYLOAD_FIELD_INDEX: usize = 9;

/// Bundle of the three HTR witnesses that algebraically compose the
/// beacon header → body → payload chain.
#[derive(Clone, Debug)]
pub struct BeaconHtrComposition {
    pub bbh_witness: BeaconBlockHeaderHtrWitness,
    pub body_witness: BeaconBlockBodyHtrWitness,
    pub payload_witness: ExecutionPayloadHeaderHtrWitness,
}

/// Verify the field-root composition between the 3 HTR witnesses.
///
/// On `Ok(())`:
///   1. BBH's leaf-4 (= `body_root`) equals `body_witness.root`.
///      Concretely: `bbh_witness.invocations[BBH_BODY_ROOT_INVOCATION].left
///      == body_witness.root`.
///   2. Body's field root at index 9 (= `execution_payload_header_root`)
///      equals `payload_witness.root`.
///
/// This pins the bottom-up chain: payload root → body field root →
/// body witness root → BBH leaf → BBH witness root.
pub fn verify_beacon_htr_chain_oracle(comp: &BeaconHtrComposition) -> Result<(), String> {
    // 1. BBH.body_root chain.
    let bbh_body_field = comp.bbh_witness.invocations[BBH_BODY_ROOT_INVOCATION].left;
    if bbh_body_field != comp.body_witness.root {
        return Err(format!(
            "BBH.body_root != body witness root: bbh_field={:?} body_root={:?}",
            bbh_body_field, comp.body_witness.root,
        ));
    }
    // 2. Body.execution_payload_header_root chain.
    let body_payload_field = comp.body_witness.field_roots[BODY_PAYLOAD_FIELD_INDEX];
    if body_payload_field != comp.payload_witness.root {
        return Err(format!(
            "Body.payload_root field != payload witness root: body_field={:?} payload_root={:?}",
            body_payload_field, comp.payload_witness.root,
        ));
    }
    Ok(())
}

impl BeaconHtrComposition {
    /// Build a composition from a `BeaconBlockHeader` and matching
    /// body + payload. Constructs each HTR witness internally.
    pub fn from_header(
        bbh: crate::beacon::BeaconBlockHeader,
        body: crate::beacon_block_body::BeaconBlockBody,
    ) -> Self {
        let payload = body.execution_payload_header.clone();
        Self {
            bbh_witness: BeaconBlockHeaderHtrWitness::from_header(bbh),
            body_witness: BeaconBlockBodyHtrWitness::from_body(body),
            payload_witness: ExecutionPayloadHeaderHtrWitness::from_payload(payload),
        }
    }

    /// The BBH root computed by the witness (= claimed beacon block root).
    pub fn beacon_block_root(&self) -> Chunk { self.bbh_witness.root }

    /// The body root computed by the witness.
    pub fn body_root(&self) -> Chunk { self.body_witness.root }

    /// The payload root computed by the witness.
    pub fn payload_root(&self) -> Chunk { self.payload_witness.root }

    /// The execution `block_hash` carried inside the payload. This is
    /// the bridge value bound to Layer B's `BlockHeader.block_hash`.
    pub fn execution_block_hash(&self) -> [u8; 32] {
        self.payload_witness.payload.block_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon::BeaconBlockHeader;
    use crate::beacon_block_body::BeaconBlockBody;
    use crate::execution_payload::ExecutionPayloadHeader;

    fn sample_payload() -> ExecutionPayloadHeader {
        let mut p = ExecutionPayloadHeader::default();
        p.block_hash = [0xab; 32];
        p.block_number = 19_000_000;
        p.state_root = [0x55; 32];
        p
    }

    fn sample_body() -> BeaconBlockBody {
        BeaconBlockBody {
            graffiti: [0x47; 32],
            attestations_root: [0xee; 32],
            execution_payload_header: sample_payload(),
            ..Default::default()
        }
    }

    fn sample_bbh_for(body: &BeaconBlockBody) -> BeaconBlockHeader {
        BeaconBlockHeader {
            slot: 1_234_567,
            proposer_index: 42,
            parent_root: [0xaa; 32],
            state_root: [0xbb; 32],
            body_root: body.hash_tree_root(),
        }
    }

    fn build_consistent_composition() -> BeaconHtrComposition {
        let body = sample_body();
        let bbh = sample_bbh_for(&body);
        BeaconHtrComposition::from_header(bbh, body)
    }

    #[test]
    fn chain_oracle_accepts_consistent_composition() {
        let comp = build_consistent_composition();
        assert!(
            verify_beacon_htr_chain_oracle(&comp).is_ok(),
            "consistent HTR chain must verify",
        );
    }

    #[test]
    fn chain_oracle_rejects_bbh_body_field_mismatch() {
        let body = sample_body();
        // Build BBH with a WRONG body_root.
        let bbh = BeaconBlockHeader {
            slot: 1,
            proposer_index: 0,
            parent_root: [0u8; 32],
            state_root: [0u8; 32],
            body_root: [0xff; 32], // wrong
        };
        let comp = BeaconHtrComposition::from_header(bbh, body);
        let err = verify_beacon_htr_chain_oracle(&comp).unwrap_err();
        assert!(err.contains("BBH.body_root"), "error message should mention BBH.body_root: {}", err);
    }

    #[test]
    fn chain_oracle_rejects_payload_field_mismatch_via_swap() {
        // Construct a body whose internal payload differs from a
        // separately-supplied payload witness, by manually wiring two
        // different payloads through the bundle.
        let body = sample_body();
        let bbh = sample_bbh_for(&body);
        let mut comp = BeaconHtrComposition::from_header(bbh, body);
        // Tamper: replace payload_witness with a different one.
        let mut other_payload = sample_payload();
        other_payload.block_hash[0] ^= 0xff;
        comp.payload_witness =
            ExecutionPayloadHeaderHtrWitness::from_payload(other_payload);
        let err = verify_beacon_htr_chain_oracle(&comp).unwrap_err();
        assert!(err.contains("payload"), "error should mention payload mismatch: {}", err);
    }

    #[test]
    fn beacon_block_root_accessor_matches_canonical() {
        let comp = build_consistent_composition();
        assert_eq!(
            comp.beacon_block_root(),
            comp.bbh_witness.header.hash_tree_root(),
        );
    }

    #[test]
    fn execution_block_hash_accessor_returns_payload_field() {
        let comp = build_consistent_composition();
        assert_eq!(comp.execution_block_hash(), [0xab; 32]);
    }

    #[test]
    fn bbh_body_root_invocation_index_is_two() {
        // Pin: leaf 4 is invocation 2 LEFT in BBH-pair AIR's evaluation
        // order. If this changes, every cross-AIR LogUp descriptor
        // built off this constant breaks.
        let comp = build_consistent_composition();
        assert_eq!(
            comp.bbh_witness.invocations[BBH_BODY_ROOT_INVOCATION].left,
            comp.body_witness.root,
        );
    }

    #[test]
    fn body_payload_field_index_is_nine() {
        // Pin: payload root is body field 9. If the body's field
        // ordering changes, the cross-AIR LogUp descriptor for the
        // body→payload binding must update.
        let comp = build_consistent_composition();
        assert_eq!(
            comp.body_witness.field_roots[BODY_PAYLOAD_FIELD_INDEX],
            comp.payload_witness.root,
        );
    }

    #[test]
    fn changing_payload_block_hash_propagates_through_chain() {
        // A small change in payload.block_hash propagates all the way
        // up to BBH root.
        let body1 = sample_body();
        let bbh1 = sample_bbh_for(&body1);
        let comp1 = BeaconHtrComposition::from_header(bbh1, body1);

        let mut payload2 = sample_payload();
        payload2.block_hash[0] ^= 0xff;
        let body2 = BeaconBlockBody {
            execution_payload_header: payload2,
            ..sample_body()
        };
        let bbh2 = sample_bbh_for(&body2);
        let comp2 = BeaconHtrComposition::from_header(bbh2, body2);

        assert_ne!(comp1.payload_root(), comp2.payload_root());
        assert_ne!(comp1.body_root(), comp2.body_root());
        assert_ne!(comp1.beacon_block_root(), comp2.beacon_block_root());
        // Both compositions must still verify (each is self-consistent).
        assert!(verify_beacon_htr_chain_oracle(&comp1).is_ok());
        assert!(verify_beacon_htr_chain_oracle(&comp2).is_ok());
    }
}
