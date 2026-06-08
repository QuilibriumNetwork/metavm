//! End-to-end host-side composition: from a finalized
//! `BeaconBlockHeader` down to a single EVM transaction's execution
//! against its world state.
//!
//! This module extends [`crate::world_proof`] (which already verifies
//! Layer A's tx/receipt/account/storage inclusion against a block
//! header) with the **C↔B bridge**:
//!
//!   BeaconBlockHeader.body_root
//!     → BeaconBlockBody.execution_payload_header.block_hash
//!     → Layer B's BlockHeader.block_hash (= keccak256(rlp(header)))
//!     → Layer A's WorldProof (tx + receipt + account + storage)
//!
//! Verifying this oracle is the host-side spec for "given a finalized
//! beacon block, every committed transaction executed against the
//! world state committed in its header." The algebraic version (still
//! to be wired) composes the same chain via cross-AIR LogUp:
//!
//!   BBH-pair AIR (CLAIMED_ROOT)
//!     ↔ BeaconBlockBody-pair AIR (body_root + payload_root)
//!     ↔ ExecutionPayloadHeader-pair AIR (block_hash output)
//!     ↔ BlockHeader AIR (block_hash via KeccakExtract)
//!     ↔ EVM main + storage chain (existing Layer A wiring)
//!
//! Once those AIRs land, this oracle's checks become redundant — the
//! algebraic proof binds the same chain.

use crate::beacon::{BeaconBlockHeader, Checkpoint, Epoch, IndexedAttestation};
use crate::beacon_block_body::BeaconBlockBody;
use crate::finality::{stake_weighted_finalization_check, ValidatorRegistry};
use crate::ssz::Chunk;
use crate::world_proof::{verify_world_proof_oracle, WorldProof};

/// Full host-side composition: a finalized `BeaconBlockHeader` +
/// its body + the execution-side world proof for one tx in that
/// execution payload.
#[derive(Clone, Debug)]
pub struct BeaconWorldProof {
    /// The finalized beacon block header. The verifier's anchor; in
    /// production its root would be cross-checked against the Casper
    /// FFG finalized checkpoint via the existing Finality AIR.
    pub beacon_block_header: BeaconBlockHeader,
    /// The body referenced by `beacon_block_header.body_root`.
    pub body: BeaconBlockBody,
    /// Layer A world proof for one tx executed in the payload.
    pub world: WorldProof,
}

/// Verify the full beacon → execution → tx chain.
///
/// On `Ok(())`:
///   1. `body.hash_tree_root() == beacon_block_header.body_root`
///   2. `body.execution_payload_header.block_hash == world.block_hash`
///   3. World proof passes: block header keccak matches `block_hash`,
///      tx inclusion under `transactions_root`, receipt inclusion
///      under `receipts_root`, per-contract account inclusion under
///      `state_root`, per-contract storage inclusion under each
///      contract's `storage_root`.
///
/// Returns descriptive `Err(...)` on any mismatch.
pub fn verify_beacon_world_proof_oracle(proof: &BeaconWorldProof) -> Result<(), String> {
    // 1. BBH ↔ body chain.
    let body_root = proof.body.hash_tree_root();
    if body_root != proof.beacon_block_header.body_root {
        return Err(format!(
            "Body root mismatch: BBH.body_root={:?} body.hash_tree_root()={:?}",
            proof.beacon_block_header.body_root, body_root,
        ));
    }
    // 2. Body's execution_payload_header.block_hash ↔ WorldProof.block_hash.
    let payload_block_hash = proof.body.execution_payload_header.block_hash;
    if payload_block_hash != proof.world.block_hash {
        return Err(format!(
            "Payload block_hash mismatch: payload={:?} world={:?}",
            payload_block_hash, proof.world.block_hash,
        ));
    }
    // 3. Layer A world proof (existing oracle).
    verify_world_proof_oracle(&proof.world)
        .map_err(|e| format!("World proof failed: {}", e))?;
    Ok(())
}

/// Extract the claimed BeaconBlockHeader root from the composition.
/// Downstream consumers (e.g. finality AIR linkages) use this as the
/// public commitment.
pub fn beacon_block_header_root(proof: &BeaconWorldProof) -> Chunk {
    proof.beacon_block_header.hash_tree_root()
}

// ─── FFG finality witness + oracle ────────────────────────────────────

/// Host-side witness that a particular `BeaconBlockHeader` is FFG-
/// finalized by ≥ `threshold_numerator / threshold_denominator` of the
/// total active effective balance at the relevant epoch.
#[derive(Clone, Debug)]
pub struct FinalityWitness {
    /// Validator registry as of the finalization epoch.
    pub registry: ValidatorRegistry,
    /// Epoch at which `registry.total_active_balance` is computed (the
    /// epoch the finalized checkpoint anchors).
    pub epoch: Epoch,
    /// Attestations (potentially from multiple aggregates) targeting
    /// the claimed finalized checkpoint.
    pub attestations: Vec<IndexedAttestation>,
    /// The checkpoint claimed to be FFG-finalized. Its `root` MUST
    /// equal `beacon_block_header.hash_tree_root()` for the witness
    /// to bind to the proof.
    pub claimed_finalized: Checkpoint,
}

/// Verify the FFG finality witness against a beacon block header.
///
/// On `Ok(())`:
///   1. `witness.claimed_finalized.root == bbh.hash_tree_root()` —
///      the checkpoint points at the correct beacon block.
///   2. The unique attesters targeting `claimed_finalized` accumulate
///      stake `≥ threshold_num/threshold_den` of the registry's total
///      active balance at `witness.epoch`.
///
/// Standard FFG threshold is `(2, 3)`. `(1, 1)` would require unanimity.
pub fn verify_ffg_finality_oracle(
    witness: &FinalityWitness,
    bbh: &BeaconBlockHeader,
    threshold_num: u64,
    threshold_den: u64,
) -> Result<(), String> {
    if threshold_den == 0 {
        return Err("threshold denominator must be non-zero".into());
    }
    // 1. Checkpoint root binding.
    let bbh_root = bbh.hash_tree_root();
    if witness.claimed_finalized.root != bbh_root {
        return Err(format!(
            "Checkpoint root mismatch: claimed={:?} bbh.hash_tree_root()={:?}",
            witness.claimed_finalized.root, bbh_root,
        ));
    }
    // 2. Stake threshold via the existing finality primitive.
    let output = stake_weighted_finalization_check(
        &witness.claimed_finalized,
        &witness.registry,
        &witness.attestations,
    );
    let total = witness.registry.total_active_balance(witness.epoch);
    // Cross-multiply to avoid floating point / division precision loss:
    //   weight / total >= num / den  ⇔  weight * den >= num * total
    let weight_x_den = (output.total_effective_balance_weight as u128) * threshold_den as u128;
    let num_x_total = (threshold_num as u128) * total as u128;
    if weight_x_den < num_x_total {
        return Err(format!(
            "FFG stake threshold not met: weight={} (gwei) total={} (gwei) threshold={}/{}",
            output.total_effective_balance_weight, total,
            threshold_num, threshold_den,
        ));
    }
    Ok(())
}

/// Full beacon-finalized + execution-bound composition.
///
/// Given a `BeaconWorldProof` (BBH + body + Layer A WorldProof) and a
/// `FinalityWitness` (registry + attestations + claimed checkpoint),
/// returns `Ok(())` iff:
///   - The execution chain (BBH → body → payload → block_header → tx)
///     all binds correctly (via `verify_beacon_world_proof_oracle`).
///   - The BBH is FFG-finalized by `threshold_num/threshold_den` of
///     the active validator stake (via `verify_ffg_finality_oracle`).
///
/// This is the **host-side spec for the user's overarching goal**:
/// "transaction's successful execution, mutation of world state,
/// inclusion in the beacon chain, and economic finality."
pub fn verify_finalized_beacon_world_proof_oracle(
    proof: &BeaconWorldProof,
    finality: &FinalityWitness,
    threshold_num: u64,
    threshold_den: u64,
) -> Result<(), String> {
    verify_beacon_world_proof_oracle(proof)
        .map_err(|e| format!("Beacon world proof failed: {}", e))?;
    verify_ffg_finality_oracle(finality, &proof.beacon_block_header, threshold_num, threshold_den)
        .map_err(|e| format!("FFG finality failed: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::Account;
    use crate::block_header::{block_header_hash, BlockHeader};
    use crate::execution_payload::ExecutionPayloadHeader;
    use crate::mpt::single_leaf_trie;
    use crate::receipt::{Receipt, ReceiptType};
    use crate::transaction::{LegacyTx, Transaction};
    use crate::world_proof::{ContractProof, StorageProof};

    /// Build a minimal-but-real composition: one contract with one
    /// storage slot, one tx, one receipt, packaged with matching
    /// payload + body + BBH.
    fn build_consistent_proof() -> BeaconWorldProof {
        // ── Layer A: one contract, one slot, one tx, one receipt ──
        let address = [0xab_u8; 20];
        let mut slot_be = [0u8; 32];
        slot_be[31] = 7;
        let mut value_be = [0u8; 32];
        value_be[31] = 0x42;

        let trie_key = crate::keccak::keccak256(&slot_be);
        let value_rlp = crate::rlp::rlp_encode_u256(&value_be);
        let (storage_root, storage_proof) = single_leaf_trie(&trie_key, &value_rlp);

        let account = Account {
            nonce: 1,
            balance: [0u8; 32],
            storage_root,
            code_hash: crate::account::empty_code_hash(),
        };
        let account_trie_key = crate::keccak::keccak256(&address);
        let account_rlp = crate::account::account_rlp(&account);
        let (state_root, account_proof) =
            single_leaf_trie(&account_trie_key, &account_rlp);

        let tx = Transaction::Legacy(LegacyTx {
            nonce: 1,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        });
        let tx_wire = tx.wire_encoding();
        let tx_index = 0u64;
        let tx_trie_key = crate::rlp::rlp_encode_uint(tx_index);
        let (transactions_root, tx_proof) =
            single_leaf_trie(&tx_trie_key, &tx_wire);

        let receipt = Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        };
        let receipt_wire = receipt.wire_encoding();
        let (receipts_root, receipt_proof) =
            single_leaf_trie(&tx_trie_key, &receipt_wire);

        let block_header = BlockHeader {
            state_root,
            transactions_root,
            receipts_root,
            ..Default::default()
        };
        let block_hash = block_header_hash(&block_header);

        let world = WorldProof {
            block_header,
            block_hash,
            tx_index,
            transaction: tx,
            transaction_proof: tx_proof,
            receipt,
            receipt_proof,
            contracts: vec![ContractProof {
                address,
                account,
                account_proof,
                storage_proofs: vec![StorageProof {
                    slot: slot_be,
                    value: value_be,
                    proof: storage_proof,
                }],
            }],
        };

        // ── C↔B bridge: build matching ExecutionPayloadHeader + body ──
        let mut payload = ExecutionPayloadHeader::default();
        payload.block_hash = block_hash;
        let body = BeaconBlockBody {
            execution_payload_header: payload,
            ..Default::default()
        };

        // ── Beacon block header committing to that body ──
        let bbh = BeaconBlockHeader {
            slot: 1234,
            proposer_index: 42,
            parent_root: [0u8; 32],
            state_root: [0u8; 32],
            body_root: body.hash_tree_root(),
        };

        BeaconWorldProof {
            beacon_block_header: bbh,
            body,
            world,
        }
    }

    #[test]
    fn full_chain_accepts_consistent_proof() {
        let p = build_consistent_proof();
        assert!(
            verify_beacon_world_proof_oracle(&p).is_ok(),
            "honest full-chain proof must verify",
        );
    }

    #[test]
    fn full_chain_rejects_tampered_body_root() {
        let mut p = build_consistent_proof();
        // Tamper the BBH's body_root — body.hash_tree_root() no longer matches.
        p.beacon_block_header.body_root[0] ^= 0xff;
        assert!(
            verify_beacon_world_proof_oracle(&p).is_err(),
            "must reject mismatched body_root",
        );
    }

    #[test]
    fn full_chain_rejects_tampered_payload_block_hash() {
        let mut p = build_consistent_proof();
        // Tamper payload.block_hash — no longer matches the Layer B block_hash.
        // We must also rebuild body_root so step 1 still passes; then step 2 fires.
        p.body.execution_payload_header.block_hash[0] ^= 0xff;
        p.beacon_block_header.body_root = p.body.hash_tree_root();
        assert!(
            verify_beacon_world_proof_oracle(&p).is_err(),
            "must reject payload.block_hash != world.block_hash",
        );
    }

    #[test]
    fn full_chain_rejects_tampered_world_proof() {
        let mut p = build_consistent_proof();
        // Tamper an MPT root in the block header — keccak no longer matches
        // the block_hash, so verify_world_proof_oracle fires.
        p.world.block_header.state_root[0] ^= 0xff;
        assert!(
            verify_beacon_world_proof_oracle(&p).is_err(),
            "must reject tampered world proof",
        );
    }

    #[test]
    fn beacon_block_header_root_accessor_matches_canonical() {
        let p = build_consistent_proof();
        assert_eq!(
            beacon_block_header_root(&p),
            p.beacon_block_header.hash_tree_root(),
        );
    }

    // ── FFG finality oracle ──────────────────────────────────────────

    use crate::beacon::{AttestationData, IndexedAttestation, Validator};

    const FAR_FUTURE_EPOCH: u64 = 1u64 << 40;

    fn mk_validator(eff_balance: u64) -> Validator {
        Validator {
            pubkey: [0u8; 48],
            withdrawal_credentials: [0u8; 32],
            effective_balance: eff_balance,
            slashed: false,
            activation_eligibility_epoch: 0,
            activation_epoch: 0,
            exit_epoch: FAR_FUTURE_EPOCH,
            withdrawable_epoch: FAR_FUTURE_EPOCH + 256,
        }
    }

    /// Build a 4-validator registry where each validator has 32 ETH
    /// (32_000_000_000 gwei) effective balance.
    fn mk_registry(n: usize) -> ValidatorRegistry {
        let validators: Vec<Validator> = (0..n).map(|_| mk_validator(32_000_000_000)).collect();
        let balances = vec![32_000_000_000u64; n];
        ValidatorRegistry::new(validators, balances)
    }

    fn build_finality_witness(
        proof: &BeaconWorldProof,
        registry: ValidatorRegistry,
        attester_indices: Vec<u64>,
    ) -> FinalityWitness {
        let bbh_root = proof.beacon_block_header.hash_tree_root();
        let finalized = Checkpoint { epoch: 100, root: bbh_root };
        let source = Checkpoint { epoch: 99, root: [0xaa; 32] };
        let data = AttestationData {
            slot: 100 * 32,
            index: 0,
            beacon_block_root: bbh_root,
            source,
            target: finalized.clone(),
        };
        let att = IndexedAttestation {
            attesting_indices: attester_indices,
            data,
            signature: [0u8; 96],
        };
        FinalityWitness {
            registry,
            epoch: 100,
            attestations: vec![att],
            claimed_finalized: finalized,
        }
    }

    #[test]
    fn ffg_oracle_accepts_unanimous_attesters() {
        let p = build_consistent_proof();
        let registry = mk_registry(4);
        let witness = build_finality_witness(&p, registry, vec![0, 1, 2, 3]);
        // 4/4 attesting → 100% stake — passes 2/3 threshold.
        assert!(
            verify_ffg_finality_oracle(&witness, &p.beacon_block_header, 2, 3).is_ok(),
            "100% stake must pass 2/3 threshold",
        );
    }

    #[test]
    fn ffg_oracle_accepts_three_of_four_attesters() {
        let p = build_consistent_proof();
        let registry = mk_registry(4);
        let witness = build_finality_witness(&p, registry, vec![0, 1, 2]);
        // 3/4 = 75% > 2/3 ≈ 66.7%.
        assert!(
            verify_ffg_finality_oracle(&witness, &p.beacon_block_header, 2, 3).is_ok(),
            "75% stake must pass 2/3 threshold",
        );
    }

    #[test]
    fn ffg_oracle_rejects_below_threshold() {
        let p = build_consistent_proof();
        let registry = mk_registry(4);
        let witness = build_finality_witness(&p, registry, vec![0, 1]);
        // 2/4 = 50% < 2/3.
        let res = verify_ffg_finality_oracle(&witness, &p.beacon_block_header, 2, 3);
        assert!(res.is_err(), "50% stake must FAIL 2/3 threshold: {:?}", res);
    }

    #[test]
    fn ffg_oracle_rejects_wrong_checkpoint_root() {
        let p = build_consistent_proof();
        let registry = mk_registry(4);
        let mut witness = build_finality_witness(&p, registry, vec![0, 1, 2, 3]);
        // Tamper the checkpoint root — must no longer match BBH.
        witness.claimed_finalized.root[0] ^= 0xff;
        assert!(
            verify_ffg_finality_oracle(&witness, &p.beacon_block_header, 2, 3).is_err(),
            "wrong checkpoint root must fail",
        );
    }

    #[test]
    fn full_finalized_oracle_accepts_honest_witness() {
        let p = build_consistent_proof();
        let registry = mk_registry(4);
        let witness = build_finality_witness(&p, registry, vec![0, 1, 2, 3]);
        assert!(
            verify_finalized_beacon_world_proof_oracle(&p, &witness, 2, 3).is_ok(),
            "honest full-stack proof must verify",
        );
    }

    #[test]
    fn full_finalized_oracle_rejects_unfinalized_witness() {
        let p = build_consistent_proof();
        let registry = mk_registry(4);
        let witness = build_finality_witness(&p, registry, vec![0]); // only 25%.
        assert!(
            verify_finalized_beacon_world_proof_oracle(&p, &witness, 2, 3).is_err(),
            "below-threshold stake must fail full oracle",
        );
    }

    #[test]
    fn full_finalized_oracle_rejects_tampered_execution_chain() {
        let mut p = build_consistent_proof();
        // Tamper the block header's state root — Layer A world proof fails.
        p.world.block_header.state_root[0] ^= 0xff;
        let registry = mk_registry(4);
        let witness = build_finality_witness(&p, registry, vec![0, 1, 2, 3]);
        assert!(
            verify_finalized_beacon_world_proof_oracle(&p, &witness, 2, 3).is_err(),
            "tampered execution chain must fail even with full stake finality",
        );
    }

    #[test]
    fn ffg_oracle_rejects_zero_denominator() {
        let p = build_consistent_proof();
        let registry = mk_registry(4);
        let witness = build_finality_witness(&p, registry, vec![0, 1, 2, 3]);
        assert!(
            verify_ffg_finality_oracle(&witness, &p.beacon_block_header, 2, 0).is_err(),
            "zero denominator must be rejected",
        );
    }
}
