//! ExecutionPayloadHeader → BlockHeader bridge.
//!
//! Verifies consistency between an ExecutionPayloadHeader (Layer C) and
//! a BlockHeader (Layer B). The payload's block_hash must equal the
//! header's computed block_hash (= keccak256(rlp(header))), and the
//! shared fields (state_root, receipts_root, etc.) must match.

use crate::block_header::BlockHeader;
use crate::execution_payload::ExecutionPayloadHeader;

pub fn verify_payload_block_header_bridge(
    payload: &ExecutionPayloadHeader,
    header: &BlockHeader,
) -> Result<(), String> {
    let header_hash = crate::block_header::block_header_hash(header);
    if payload.block_hash != header_hash {
        return Err(format!(
            "block_hash mismatch: payload.block_hash != keccak256(rlp(header))"
        ));
    }
    if payload.parent_hash != header.parent_hash {
        return Err("parent_hash mismatch".into());
    }
    if payload.state_root != header.state_root {
        return Err("state_root mismatch".into());
    }
    if payload.receipts_root != header.receipts_root {
        return Err("receipts_root mismatch".into());
    }
    if payload.transactions_root != header.transactions_root {
        return Err("transactions_root mismatch".into());
    }
    if payload.fee_recipient != header.beneficiary {
        return Err("fee_recipient/beneficiary mismatch".into());
    }
    if payload.block_number != header.number {
        return Err("block_number/number mismatch".into());
    }
    if payload.gas_limit != header.gas_limit {
        return Err("gas_limit mismatch".into());
    }
    if payload.gas_used != header.gas_used {
        return Err("gas_used mismatch".into());
    }
    if payload.timestamp != header.timestamp {
        return Err("timestamp mismatch".into());
    }
    if payload.prev_randao != header.mix_hash {
        return Err("prev_randao/mix_hash mismatch".into());
    }
    if payload.base_fee_per_gas != header.base_fee_per_gas.unwrap_or([0u8; 32]) {
        return Err("base_fee_per_gas mismatch".into());
    }
    if let Some(wr) = header.withdrawals_root {
        if payload.withdrawals_root != wr {
            return Err("withdrawals_root mismatch".into());
        }
    }
    if payload.logs_bloom != header.logs_bloom {
        return Err("logs_bloom mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_consistent_pair() -> (ExecutionPayloadHeader, BlockHeader) {
        let mut header = BlockHeader::default();
        header.state_root = [0x11; 32];
        header.transactions_root = [0x22; 32];
        header.receipts_root = [0x33; 32];
        header.number = 12345;
        header.timestamp = 1700000000;
        header.gas_limit = 30_000_000;
        header.gas_used = 15_000_000;
        header.beneficiary = [0x42; 20];
        header.mix_hash = [0x55; 32];
        header.base_fee_per_gas = Some([0u8; 32]);
        header.withdrawals_root = Some([0x77; 32]);
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
            base_fee_per_gas: [0u8; 32],
            block_hash,
            transactions_root: header.transactions_root,
            withdrawals_root: [0x77; 32],
            blob_gas_used: 0,
            excess_blob_gas: 0,
        };
        (payload, header)
    }

    #[test]
    fn consistent_pair_passes() {
        let (p, h) = make_consistent_pair();
        verify_payload_block_header_bridge(&p, &h).unwrap();
    }

    #[test]
    fn block_hash_mismatch_fails() {
        let (mut p, h) = make_consistent_pair();
        p.block_hash[0] ^= 0xff;
        assert!(verify_payload_block_header_bridge(&p, &h).is_err());
    }

    #[test]
    fn state_root_mismatch_fails() {
        let (mut p, h) = make_consistent_pair();
        p.state_root[0] ^= 0xff;
        assert!(verify_payload_block_header_bridge(&p, &h).is_err());
    }

    #[test]
    fn timestamp_mismatch_fails() {
        let (mut p, h) = make_consistent_pair();
        p.timestamp += 1;
        assert!(verify_payload_block_header_bridge(&p, &h).is_err());
    }
}
