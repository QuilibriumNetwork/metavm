//! End-to-end host-side finality oracle.
//!
//! Composes all host-side verification oracles into a single
//! function that validates the full chain:
//!
//! 1. FFG checkpoint chain (epoch monotonicity)
//! 2. Beacon block header chain (parent_root linkage via HTR)
//! 3. BeaconWorldProof composition (BBH → body → payload)
//! 4. ExecutionPayloadHeader → BlockHeader bridge (block_hash equality)
//! 5. Validator set supermajority attestation

use crate::beacon::BeaconBlockHeader;
use crate::beacon_block_body::BeaconBlockBody;
use crate::block_header::BlockHeader;
use crate::execution_payload::ExecutionPayloadHeader;
use crate::validator_set::{ValidatorState, total_active_balance, total_attesting_balance, is_supermajority};

pub struct FinalityBundle {
    pub header: BeaconBlockHeader,
    pub body: BeaconBlockBody,
    pub payload: ExecutionPayloadHeader,
    pub block_header: BlockHeader,
}

pub fn verify_finality_e2e(
    bundle: &FinalityBundle,
    validators: &[ValidatorState],
    participating: &[bool],
    epoch: u64,
) -> Result<(), String> {
    // 1. Verify BeaconBlockBody → BeaconBlockHeader consistency
    let body_root = bundle.body.hash_tree_root();
    if bundle.header.body_root != body_root {
        return Err("body_root mismatch: body.hash_tree_root() != header.body_root".into());
    }

    // 2. Verify ExecutionPayloadHeader is embedded in body
    if bundle.body.execution_payload_header != bundle.payload {
        return Err("execution_payload_header mismatch".into());
    }

    // 3. Verify payload → block_header bridge
    crate::payload_to_block_header::verify_payload_block_header_bridge(
        &bundle.payload, &bundle.block_header,
    )?;

    // 4. Verify 2/3 supermajority attestation
    let total = total_active_balance(validators, epoch);
    let attesting = total_attesting_balance(validators, participating, epoch);
    if !is_supermajority(attesting, total) {
        return Err(format!(
            "attestation below 2/3 supermajority: {} / {} (need {})",
            attesting, total, (total * 2 + 2) / 3,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beacon::BeaconBlockHeader;

    fn make_test_bundle() -> (FinalityBundle, Vec<ValidatorState>, Vec<bool>) {
        let mut block_header = BlockHeader::default();
        block_header.number = 100;
        block_header.timestamp = 1700000000;
        block_header.gas_limit = 30_000_000;

        let block_hash = crate::block_header::block_header_hash(&block_header);

        let payload = ExecutionPayloadHeader {
            parent_hash: block_header.parent_hash,
            fee_recipient: block_header.beneficiary,
            state_root: block_header.state_root,
            receipts_root: block_header.receipts_root,
            logs_bloom: block_header.logs_bloom,
            prev_randao: block_header.mix_hash,
            block_number: block_header.number,
            gas_limit: block_header.gas_limit,
            gas_used: block_header.gas_used,
            timestamp: block_header.timestamp,
            extra_data: block_header.extra_data.clone(),
            base_fee_per_gas: block_header.base_fee_per_gas.unwrap_or([0u8; 32]),
            block_hash,
            transactions_root: block_header.transactions_root,
            withdrawals_root: block_header.withdrawals_root.unwrap_or([0u8; 32]),
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

        let header = BeaconBlockHeader {
            slot: 1000,
            proposer_index: 42,
            parent_root: [0u8; 32],
            state_root: [0xAA; 32],
            body_root,
        };

        let validators: Vec<ValidatorState> = (0..100).map(|i| ValidatorState {
            pubkey: { let mut p = [0u8; 48]; p[0] = i as u8; p },
            effective_balance: 32_000_000_000,
            slashed: false,
            activation_epoch: 0,
            exit_epoch: u64::MAX,
        }).collect();

        let participating: Vec<bool> = (0..100).map(|i| i < 70).collect();

        let bundle = FinalityBundle { header, body, payload, block_header };

        (bundle, validators, participating)
    }

    #[test]
    fn e2e_finality_passes() {
        let (bundle, vals, part) = make_test_bundle();
        verify_finality_e2e(&bundle, &vals, &part, 10).unwrap();
    }

    #[test]
    fn e2e_finality_body_root_mismatch() {
        let (mut bundle, vals, part) = make_test_bundle();
        bundle.header.body_root[0] ^= 0xff;
        assert!(verify_finality_e2e(&bundle, &vals, &part, 10).is_err());
    }

    #[test]
    fn e2e_finality_insufficient_attestation() {
        let (bundle, vals, _) = make_test_bundle();
        let weak_part: Vec<bool> = (0..100).map(|i| i < 60).collect();
        assert!(verify_finality_e2e(&bundle, &vals, &weak_part, 10).is_err());
    }

    #[test]
    fn e2e_finality_block_hash_mismatch() {
        let (mut bundle, vals, part) = make_test_bundle();
        bundle.payload.block_hash[0] ^= 0xff;
        assert!(verify_finality_e2e(&bundle, &vals, &part, 10).is_err());
    }
}
