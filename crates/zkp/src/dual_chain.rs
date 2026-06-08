//! Dual-chain consistency oracle.
//!
//! Verifies that a beacon header chain and an execution block chain
//! are consistent: each beacon block's execution payload block_hash
//! matches the corresponding execution block's keccak256(rlp(header)).

use crate::beacon::BeaconBlockHeader;
use crate::beacon_block_body::BeaconBlockBody;
use crate::block_header::{BlockHeader, block_header_hash};

pub struct DualChainEntry {
    pub beacon_header: BeaconBlockHeader,
    pub beacon_body: BeaconBlockBody,
    pub execution_header: BlockHeader,
}

pub fn verify_dual_chain_consistency(entries: &[DualChainEntry]) -> Result<(), String> {
    for (i, entry) in entries.iter().enumerate() {
        let body_root = entry.beacon_body.hash_tree_root();
        if entry.beacon_header.body_root != body_root {
            return Err(format!("entry[{}]: body_root mismatch", i));
        }

        let exec_hash = block_header_hash(&entry.execution_header);
        let payload_hash = entry.beacon_body.execution_block_hash();
        if exec_hash != payload_hash {
            return Err(format!(
                "entry[{}]: execution block_hash != payload block_hash",
                i,
            ));
        }
    }

    // Verify beacon parent_root chain.
    for i in 1..entries.len() {
        let parent_htr = entries[i - 1].beacon_header.hash_tree_root();
        if entries[i].beacon_header.parent_root != parent_htr {
            return Err(format!(
                "entry[{}]: beacon parent_root != HTR(entry[{}])",
                i, i - 1,
            ));
        }
    }

    // Verify execution parent_hash chain.
    for i in 1..entries.len() {
        let parent_hash = block_header_hash(&entries[i - 1].execution_header);
        if entries[i].execution_header.parent_hash != parent_hash {
            return Err(format!(
                "entry[{}]: execution parent_hash != hash(entry[{}])",
                i, i - 1,
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_payload::ExecutionPayloadHeader;

    fn make_entry(
        slot: u64,
        number: u64,
        beacon_parent: [u8; 32],
        exec_parent: [u8; 32],
    ) -> DualChainEntry {
        let mut exec = BlockHeader::default();
        exec.number = number;
        exec.timestamp = 1700000000 + number * 12;
        exec.parent_hash = exec_parent;
        exec.gas_limit = 30_000_000;

        let exec_hash = block_header_hash(&exec);
        let payload = ExecutionPayloadHeader {
            block_hash: exec_hash,
            block_number: number,
            timestamp: exec.timestamp,
            gas_limit: exec.gas_limit,
            parent_hash: exec.parent_hash,
            fee_recipient: exec.beneficiary,
            state_root: exec.state_root,
            receipts_root: exec.receipts_root,
            logs_bloom: exec.logs_bloom,
            prev_randao: exec.mix_hash,
            gas_used: exec.gas_used,
            extra_data: exec.extra_data.clone(),
            base_fee_per_gas: exec.base_fee_per_gas.unwrap_or([0u8; 32]),
            transactions_root: exec.transactions_root,
            withdrawals_root: exec.withdrawals_root.unwrap_or([0u8; 32]),
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
            execution_payload_header: payload,
            bls_to_execution_changes_root: [0u8; 32],
            blob_kzg_commitments_root: [0u8; 32],
        };
        let body_root = body.hash_tree_root();

        let beacon = BeaconBlockHeader {
            slot,
            proposer_index: 0,
            parent_root: beacon_parent,
            state_root: [0u8; 32],
            body_root,
        };

        DualChainEntry { beacon_header: beacon, beacon_body: body, execution_header: exec }
    }

    #[test]
    fn single_entry_consistent() {
        let entry = make_entry(100, 50, [0u8; 32], [0u8; 32]);
        verify_dual_chain_consistency(&[entry]).unwrap();
    }

    #[test]
    fn two_entry_chain() {
        let e0 = make_entry(100, 50, [0u8; 32], [0u8; 32]);
        let beacon_parent = e0.beacon_header.hash_tree_root();
        let exec_parent = block_header_hash(&e0.execution_header);
        let e1 = make_entry(101, 51, beacon_parent, exec_parent);
        verify_dual_chain_consistency(&[e0, e1]).unwrap();
    }

    #[test]
    fn broken_body_root_fails() {
        let mut entry = make_entry(100, 50, [0u8; 32], [0u8; 32]);
        entry.beacon_header.body_root[0] ^= 0xff;
        assert!(verify_dual_chain_consistency(&[entry]).is_err());
    }

    #[test]
    fn broken_beacon_parent_fails() {
        let e0 = make_entry(100, 50, [0u8; 32], [0u8; 32]);
        let exec_parent = block_header_hash(&e0.execution_header);
        let e1 = make_entry(101, 51, [0xBB; 32], exec_parent); // wrong beacon parent
        assert!(verify_dual_chain_consistency(&[e0, e1]).is_err());
    }
}
