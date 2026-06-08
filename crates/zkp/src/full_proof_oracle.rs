//! Full proof oracle: EVM execution → economic finality.
//!
//! Composes block_execution_proof (Layer A+B) with
//! e2e_finality_oracle (Layer C) into a single verification
//! function covering the user's overarching goal:
//!
//! "transaction's successful execution, mutation of world state,
//! inclusion in the beacon chain, and economic finality"

use crate::beacon::BeaconBlockHeader;
use crate::beacon_block_body::BeaconBlockBody;
use crate::block_execution_proof::{BlockExecutionProof, verify_block_execution_proof};
use crate::e2e_finality_oracle::{FinalityBundle, verify_finality_e2e};
use crate::execution_payload::ExecutionPayloadHeader;
use crate::validator_set::ValidatorState;

pub struct FullProof {
    pub execution: BlockExecutionProof,
    pub beacon_header: BeaconBlockHeader,
    pub beacon_body: BeaconBlockBody,
    pub payload: ExecutionPayloadHeader,
}

pub fn verify_full_proof(
    proof: &FullProof,
    validators: &[ValidatorState],
    participating: &[bool],
    epoch: u64,
) -> Result<(), String> {
    // Layer A+B: verify transaction execution + inclusion
    let block_hash = verify_block_execution_proof(&proof.execution)
        .map_err(|e| format!("execution proof: {}", e))?;

    // Bridge: verify payload.block_hash matches execution block_hash
    if proof.payload.block_hash != block_hash {
        return Err(format!(
            "bridge mismatch: payload.block_hash != execution block_hash"
        ));
    }

    // Verify slot/epoch consistency
    crate::epoch::verify_slot_epoch_consistency(proof.beacon_header.slot, epoch)
        .map_err(|e| format!("epoch: {}", e))?;

    // Layer C: verify beacon chain finality
    let finality_bundle = FinalityBundle {
        header: proof.beacon_header.clone(),
        body: proof.beacon_body.clone(),
        payload: proof.payload.clone(),
        block_header: proof.execution.header.clone(),
    };
    verify_finality_e2e(&finality_bundle, validators, participating, epoch)
        .map_err(|e| format!("finality: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt;
    use crate::receipt::{Receipt, ReceiptType};
    use crate::transaction::{LegacyTx, Transaction};
    use crate::block_header::BlockHeader;

    fn make_full_proof() -> (FullProof, Vec<ValidatorState>, Vec<bool>) {
        let tx = Transaction::Legacy(LegacyTx {
            nonce: 0, gas_price: [0u8; 32], gas_limit: 21000,
            to: Some([0x42; 20]), value: [0u8; 32], data: vec![],
            v: 27, r: [0xAA; 32], s: [0xBB; 32],
        });
        let receipt = Receipt {
            ty: ReceiptType::Legacy, status: 1,
            cumulative_gas_used: 21_000, logs_bloom: [0u8; 256], logs: Vec::new(),
        };

        let tx_key = crate::transaction::tx_trie_key(0);
        let tx_value = tx.wire_encoding();
        let (tx_root, tx_proof) = mpt::single_leaf_trie(&tx_key, &tx_value);

        let rx_key = crate::receipt::receipt_trie_key(0);
        let rx_value = receipt.wire_encoding();
        let (rx_root, rx_proof) = mpt::single_leaf_trie(&rx_key, &rx_value);

        let mut header = BlockHeader::default();
        header.transactions_root = tx_root;
        header.receipts_root = rx_root;
        header.number = 100;
        header.timestamp = 1700000000;
        header.gas_limit = 30_000_000;
        header.gas_used = 21_000;

        let block_hash = crate::block_header::block_header_hash(&header);

        let payload = ExecutionPayloadHeader {
            parent_hash: header.parent_hash,
            fee_recipient: header.beneficiary,
            state_root: header.state_root,
            receipts_root: header.receipts_root,
            logs_bloom: header.logs_bloom,
            prev_randao: header.mix_hash,
            block_number: header.number,
            gas_limit: header.gas_limit,
            gas_used: header.gas_used,
            timestamp: header.timestamp,
            extra_data: header.extra_data.clone(),
            base_fee_per_gas: header.base_fee_per_gas.unwrap_or([0u8; 32]),
            block_hash,
            transactions_root: header.transactions_root,
            withdrawals_root: header.withdrawals_root.unwrap_or([0u8; 32]),
            blob_gas_used: 0,
            excess_blob_gas: 0,
        };

        let body = BeaconBlockBody {
            randao_reveal_root: [0u8; 32],
            eth1_data_root: [0u8; 32],
            graffiti: [0u8; 32],
            proposer_slashings_root: [0u8; 32],
            attester_slashings_root: [0u8; 32],
            attestations_root: [0u8; 32],
            deposits_root: [0u8; 32],
            voluntary_exits_root: [0u8; 32],
            sync_aggregate_root: [0u8; 32],
            execution_payload_header: payload.clone(),
            bls_to_execution_changes_root: [0u8; 32],
            blob_kzg_commitments_root: [0u8; 32],
        };
        let body_root = body.hash_tree_root();

        let beacon_header = BeaconBlockHeader {
            slot: 320, // epoch 10 = slots 320..351
            proposer_index: 42,
            parent_root: [0u8; 32],
            state_root: [0xAA; 32],
            body_root,
        };

        let exec = BlockExecutionProof {
            header, tx_index: 0, tx, tx_proof, receipt, receipt_proof: rx_proof,
        };

        let validators: Vec<ValidatorState> = (0..100).map(|i| ValidatorState {
            pubkey: { let mut p = [0u8; 48]; p[0] = i as u8; p },
            effective_balance: 32_000_000_000,
            slashed: false, activation_epoch: 0, exit_epoch: u64::MAX,
        }).collect();

        let participating: Vec<bool> = (0..100).map(|i| i < 70).collect();

        let proof = FullProof {
            execution: exec,
            beacon_header,
            beacon_body: body,
            payload,
        };
        (proof, validators, participating)
    }

    #[test]
    fn full_proof_passes() {
        let (p, v, part) = make_full_proof();
        verify_full_proof(&p, &v, &part, 10).unwrap();
    }

    #[test]
    fn full_proof_tampered_tx_fails() {
        let (mut p, v, part) = make_full_proof();
        p.execution.header.transactions_root[0] ^= 0xff;
        assert!(verify_full_proof(&p, &v, &part, 10).is_err());
    }

    #[test]
    fn full_proof_bridge_mismatch_fails() {
        let (mut p, v, part) = make_full_proof();
        p.payload.block_hash[0] ^= 0xff;
        assert!(verify_full_proof(&p, &v, &part, 10).is_err());
    }

    #[test]
    fn full_proof_weak_attestation_fails() {
        let (p, v, _) = make_full_proof();
        let weak: Vec<bool> = (0..100).map(|i| i < 60).collect();
        assert!(verify_full_proof(&p, &v, &weak, 10).is_err());
    }

    #[test]
    fn full_proof_wrong_epoch_fails() {
        let (p, v, part) = make_full_proof();
        // slot 320 = epoch 10, but claiming epoch 5
        assert!(verify_full_proof(&p, &v, &part, 5).is_err());
    }
}
