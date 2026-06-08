//! Receipt RLP composition oracle.
//!
//! Decomposes a `Receipt` (no-logs case) into per-field RLP encodings
//! using the session's gadgets and verifies the result matches
//! `Receipt::wire_encoding()`.

use crate::receipt::{Log, Receipt, ReceiptType};
use crate::u64_rlp_air::rlp_encode_u64;

/// Verify receipt RLP composition for a receipt with no logs.
/// Receipts with logs need a dedicated Log RLP gadget (deferred).
pub fn verify_receipt_rlp_composition(receipt: &Receipt) -> Result<Vec<u8>, String> {
    if !receipt.logs.is_empty() {
        return Err("receipt_rlp_composition only handles no-logs receipts; Log RLP gadget needed".into());
    }
    let canonical = receipt.wire_encoding();

    // Use rlp_var_bytes for short fields, canonical rlp_encode_bytes
    // for logs_bloom (256 bytes > our 32-byte gadget limit).
    let fields: Vec<Vec<u8>> = vec![
        crate::rlp_var_bytes_air::rlp_encode_bytes(&[receipt.status]),
        rlp_encode_u64(receipt.cumulative_gas_used),
        crate::rlp::rlp_encode_bytes(&receipt.logs_bloom),
        vec![0xc0], // empty list encoding
    ];

    let payload_len: usize = fields.iter().map(|f| f.len()).sum();
    let mut body = Vec::with_capacity(3 + payload_len);
    if payload_len < 56 {
        body.push(0xc0 + payload_len as u8);
    } else {
        let mut len_be = Vec::new();
        let mut n = payload_len;
        while n > 0 { len_be.push((n & 0xff) as u8); n >>= 8; }
        len_be.reverse();
        body.push(0xf7 + len_be.len() as u8);
        body.extend_from_slice(&len_be);
    }
    for f in &fields { body.extend_from_slice(f); }

    let assembled = match receipt.ty.type_byte() {
        None => body,
        Some(tb) => { let mut out = vec![tb]; out.extend_from_slice(&body); out }
    };

    if assembled != canonical {
        return Err(format!("receipt RLP mismatch: {} vs {} bytes", assembled.len(), canonical.len()));
    }
    Ok(assembled)
}

/// Verify log RLP composition using our per-field gadgets.
pub fn verify_log_rlp_composition(log: &Log) -> Result<Vec<u8>, String> {
    let canonical = log.rlp_encode();
    let address_enc = crate::rlp::rlp_encode_bytes(&log.address);
    let topics_items: Vec<Vec<u8>> = log.topics.iter()
        .map(|t| { let mut e = vec![crate::fixed_rlp_air::RLP32_PREFIX]; e.extend_from_slice(t); e })
        .collect();
    let topics_enc = crate::rlp::rlp_encode_list(&topics_items);
    let data_enc = crate::rlp::rlp_encode_bytes(&log.data);
    let fields = vec![address_enc, topics_enc, data_enc];
    let payload_len: usize = fields.iter().map(|f| f.len()).sum();
    let mut assembled = Vec::with_capacity(3 + payload_len);
    if payload_len < 56 {
        assembled.push(0xc0 + payload_len as u8);
    } else {
        let mut len_be = Vec::new(); let mut n = payload_len;
        while n > 0 { len_be.push((n & 0xff) as u8); n >>= 8; }
        len_be.reverse();
        assembled.push(0xf7 + len_be.len() as u8);
        assembled.extend_from_slice(&len_be);
    }
    for f in &fields { assembled.extend_from_slice(f); }
    if assembled != canonical {
        return Err(format!("log RLP mismatch: {} vs {} bytes", assembled.len(), canonical.len()));
    }
    Ok(assembled)
}

/// Verify receipt RLP composition with logs.
pub fn verify_receipt_with_logs_rlp_composition(receipt: &Receipt) -> Result<Vec<u8>, String> {
    let canonical = receipt.wire_encoding();
    let logs_items: Vec<Vec<u8>> = receipt.logs.iter()
        .map(|l| verify_log_rlp_composition(l))
        .collect::<Result<_, _>>()?;
    let logs_enc = crate::rlp::rlp_encode_list(&logs_items);

    let fields: Vec<Vec<u8>> = vec![
        crate::rlp_var_bytes_air::rlp_encode_bytes(&[receipt.status]),
        rlp_encode_u64(receipt.cumulative_gas_used),
        crate::rlp::rlp_encode_bytes(&receipt.logs_bloom),
        logs_enc,
    ];
    let payload_len: usize = fields.iter().map(|f| f.len()).sum();
    let mut body = Vec::with_capacity(3 + payload_len);
    if payload_len < 56 {
        body.push(0xc0 + payload_len as u8);
    } else {
        let mut len_be = Vec::new(); let mut n = payload_len;
        while n > 0 { len_be.push((n & 0xff) as u8); n >>= 8; }
        len_be.reverse();
        body.push(0xf7 + len_be.len() as u8);
        body.extend_from_slice(&len_be);
    }
    for f in &fields { body.extend_from_slice(f); }
    let assembled = match receipt.ty.type_byte() {
        None => body,
        Some(tb) => { let mut out = vec![tb]; out.extend_from_slice(&body); out }
    };
    if assembled != canonical {
        return Err(format!("receipt+logs RLP mismatch: {} vs {} bytes", assembled.len(), canonical.len()));
    }
    Ok(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_receipt_no_logs() {
        let r = Receipt {
            ty: ReceiptType::Legacy, status: 1,
            cumulative_gas_used: 21_000, logs_bloom: [0u8; 256], logs: Vec::new(),
        };
        verify_receipt_rlp_composition(&r).unwrap();
    }

    #[test]
    fn eip1559_receipt_no_logs() {
        let r = Receipt {
            ty: ReceiptType::Eip1559, status: 1,
            cumulative_gas_used: 100_000, logs_bloom: [0x11; 256], logs: Vec::new(),
        };
        let assembled = verify_receipt_rlp_composition(&r).unwrap();
        assert_eq!(assembled[0], 0x02); // type byte
        assert_eq!(assembled, r.wire_encoding());
    }

    #[test]
    fn log_rlp_composition() {
        let log = Log {
            address: [0x42u8; 20],
            topics: vec![[0xaa; 32], [0xbb; 32]],
            data: vec![0xde, 0xad, 0xbe, 0xef],
        };
        verify_log_rlp_composition(&log).unwrap();
    }

    #[test]
    fn receipt_with_logs() {
        let r = Receipt {
            ty: ReceiptType::Legacy, status: 1,
            cumulative_gas_used: 50_000, logs_bloom: [0u8; 256],
            logs: vec![
                Log { address: [0x11; 20], topics: vec![[0xcc; 32]], data: vec![0x01, 0x02] },
                Log { address: [0x22; 20], topics: vec![], data: vec![] },
            ],
        };
        verify_receipt_with_logs_rlp_composition(&r).unwrap();
    }

    #[test]
    fn failed_receipt() {
        let r = Receipt {
            ty: ReceiptType::Legacy, status: 0,
            cumulative_gas_used: 50_000, logs_bloom: [0u8; 256], logs: Vec::new(),
        };
        verify_receipt_rlp_composition(&r).unwrap();
    }
}
