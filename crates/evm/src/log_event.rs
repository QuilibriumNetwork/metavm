//! LOG opcode event extraction from EVM execution traces.
//!
//! Extracts LOG0..LOG4 events from the EVM trace columns and builds
//! a witness usable by the receipt/bloom AIR pipeline.

use crate::trace::EvmTraceColumns;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogEvent {
    pub address: [u8; 20],
    pub topics: Vec<[u8; 32]>,
    pub data: Vec<u8>,
    pub log_index: usize,
    pub mem_offset: u64,
    pub mem_size: u64,
}

/// Extract LOG event metadata from the EVM trace.
///
/// Current limitation: the inspector captures `input0 = stack[0]` (memory
/// offset) and `input1 = stack[1]` (size) for LOG rows. Topic values live at
/// stack positions 2..2+n which are NOT currently captured in trace columns.
/// Full topic extraction requires adding dedicated topic witness columns to
/// the trace (deferred). This function captures the event's existence,
/// topic count, and memory region.
pub fn extract_log_events(cols: &EvmTraceColumns) -> Vec<LogEvent> {
    let n = cols.step.len();
    let mut events = Vec::new();
    for r in 0..n {
        if cols.sel_log[r] != 1 { continue; }
        let topic_count = cols.funct[r] as usize;
        let mem_offset = cols.input0[0][r];
        let mem_size = cols.input1[0][r];

        let mut topics = Vec::with_capacity(topic_count);
        if topic_count >= 1 {
            // topic0 captured in immediate by inspector (LOG1..LOG4)
            topics.push(limbs_to_be32(
                cols.immediate[0][r], cols.immediate[1][r],
                cols.immediate[2][r], cols.immediate[3][r],
            ));
        }
        // topics 1-3 need additional trace columns (deferred)
        for _ in 1..topic_count {
            topics.push([0u8; 32]);
        }

        events.push(LogEvent {
            address: [0u8; 20],
            topics,
            data: Vec::new(),
            log_index: events.len(),
            mem_offset,
            mem_size,
        });
    }
    events
}

fn limbs_to_be32(l0: u64, l1: u64, l2: u64, l3: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[24..32].copy_from_slice(&l0.to_be_bytes());
    out[16..24].copy_from_slice(&l1.to_be_bytes());
    out[8..16].copy_from_slice(&l2.to_be_bytes());
    out[0..8].copy_from_slice(&l3.to_be_bytes());
    out
}

/// Verify that the bloom filter in a receipt matches the given log events.
pub fn verify_logs_bloom(
    logs_bloom: &[u8; 256],
    logs: &[(/* address */ [u8; 20], /* topics */ Vec<[u8; 32]>)],
) -> Result<(), String> {
    let expected = metavm_zkp::bloom::logs_bloom_for_block(logs);
    if &expected != logs_bloom {
        return Err(format!("logs_bloom mismatch: computed vs provided"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    #[test]
    fn extract_log0_event() {
        // LOG0: PUSH1 0x00 (size), PUSH1 0x00 (offset), LOG0, STOP
        let bytecode = vec![0x60, 0x00, 0x60, 0x00, 0xA0, 0x00];
        let cols = execute_bytecode(&bytecode, &[]).unwrap();
        let events = extract_log_events(&cols);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].topics.len(), 0);
        assert_eq!(events[0].log_index, 0);
    }

    #[test]
    fn log_per_variant_selectors() {
        // LOG0: PUSH1 0; PUSH1 0; LOG0; STOP
        let bc = vec![0x60, 0x00, 0x60, 0x00, 0xA0, 0x00];
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let n = cols.step.len();
        let mut found_log0 = false;
        for r in 0..n {
            if cols.opcode[r] == 0xA0 {
                assert_eq!(cols.sel_log[r], 1);
                assert_eq!(cols.sel_log0[r], 1);
                assert_eq!(cols.sel_log1[r], 0);
                found_log0 = true;
            }
        }
        assert!(found_log0, "LOG0 not found");
    }

    #[test]
    fn extract_log1_event_with_topic() {
        // LOG1: PUSH32 topic; PUSH1 0x00 (size); PUSH1 0x00 (offset); LOG1; STOP
        let mut bc = vec![0x7F]; // PUSH32
        let topic = [0xAA; 32];
        bc.extend_from_slice(&topic);
        bc.extend_from_slice(&[0x60, 0x00, 0x60, 0x00, 0xA1, 0x00]);
        let cols = execute_bytecode(&bc, &[]).unwrap();
        let events = extract_log_events(&cols);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].topics.len(), 1);
        // topic0 is now captured via inspector's immediate field
        assert_eq!(events[0].topics[0], topic);
        assert_eq!(events[0].mem_offset, 0);
        assert_eq!(events[0].mem_size, 0);
    }

    #[test]
    fn bloom_verification_smoke() {
        let addr = [0x42u8; 20];
        let topics = vec![[0xAA; 32]];
        let bloom = metavm_zkp::bloom::logs_bloom_for_block(&[(addr, topics.clone())]);
        verify_logs_bloom(&bloom, &[(addr, topics)]).unwrap();
    }

    #[test]
    fn bloom_verification_mismatch_fails() {
        let bloom = [0u8; 256];
        let addr = [0x42u8; 20];
        let topics = vec![[0xAA; 32]];
        assert!(verify_logs_bloom(&bloom, &[(addr, topics)]).is_err());
    }

    #[test]
    fn empty_logs_empty_bloom() {
        let bloom = [0u8; 256];
        verify_logs_bloom(&bloom, &[]).unwrap();
    }
}
