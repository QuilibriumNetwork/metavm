//! End-to-end receipt-with-logs composition chain.
//!
//! Wires
//!
//! ```text
//!   logs_rlp_air         (per-log RLP encoding)
//!         ↓ multiset on (log_index, byte_value)
//!   logs_list_concat_air (byte-granular concat over all logs)
//!         ↓ multiset on (absolute_offset_in_logs_list, byte_value)
//!   rlp_byte_concat_air  (byte-level offset chain — outer receipt payload)
//!         ↓ multiset on (absolute_offset, byte_value)
//!   receipt_rlp_air      (outer receipt RLP)
//! ```
//!
//! so that a receipt with N logs is algebraically composable from the
//! per-log encodings → concatenated logs list → receipt payload region.
//!
//! # Logs region (the "what" the chain pins)
//!
//! Ethereum receipt RLP shape (logs-present case):
//!
//! ```text
//! Receipt.wire_encoding() =
//!   [type_byte]?
//! || RLP-list-prefix
//! || RLP(status)
//! || RLP(cumulative_gas_used)
//! || RLP(logs_bloom)                    (259 bytes)
//! || RLP-list-prefix(logs)              (1 or 2 bytes — `0xc0+N` or `0xf8 N`)
//! || encoded(log_0) || encoded(log_1) || ...
//! ```
//!
//! The empty-logs case shrinks the logs region to a single `0xc0` byte
//! and is the only case [`crate::receipt_rlp_air`] currently pins
//! algebraically (it hard-codes `EMPTY_LOGS_BYTE = 0xc0`). For the
//! logs-present case the receipt AIR would need to be widened to expose
//! a variable-length logs region as a byte stream + accept arbitrary
//! `payload_len`. **The current chain documents this gap algebraically
//! via host-side cross-checks + descriptor scaffolding**; the per-log
//! → concat side of the chain is already fully algebraic.
//!
//! # Limitations recorded
//!
//! - [`crate::receipt_rlp_air`] pins `EMPTY_LOGS_BYTE = 0xc0` and
//!   `payload_len = status + cumul + 259 + 1` — the logs-present case
//!   requires widening that AIR. The chain still assembles a logs-side
//!   witness (per-log + concat + byte-concat) for any N ≥ 0 logs; for
//!   N = 0 the receipt AIR's row-local constraints fully bind the
//!   trailing `0xc0`; for N ≥ 1 the receipt-side binding is
//!   non-algebraic until the AIR is widened.
//! - The `logs_list_concat_air` descriptors point at `logs_rlp_air`'s
//!   1-row-per-log layout (~261 cols × 1 row, with `encoded_bytes` held
//!   in `LogsRlpRow` but not yet exposed as per-byte trace columns).
//!   The per-log-byte binding requires an adapter AIR unfolding the
//!   per-log row into byte rows; the descriptors here record the SHAPE
//!   CONTRACT.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::receipt::Receipt;

// Per-AIR types referenced by the assembler.
use crate::logs_list_concat_air::{
    LogsListConcatWitness, COL_ABSOLUTE_OFFSET as LLC_ABS_OFFSET,
    COL_BYTE_VALUE as LLC_BYTE_VALUE, COL_IS_REAL as LLC_IS_REAL,
    COL_LOG_ENCODED_LEN as LLC_LOG_ENC_LEN, COL_LOG_INDEX as LLC_LOG_INDEX,
};
use crate::logs_rlp_air::LogsRlpWitness;
use crate::receipt_rlp_air::ReceiptRlpWitness;
use crate::rlp_byte_concat_air::{
    RlpByteConcatWitness, COL_ABSOLUTE_OFFSET as RBC_ABS_OFFSET,
    COL_BYTE_VALUE as RBC_BYTE_VALUE, COL_IS_REAL as RBC_IS_REAL,
};

// ─────────────────────────────────────────────────────────────────────
// Witness bundle
// ─────────────────────────────────────────────────────────────────────

/// Composite witness threading a receipt through the
/// `logs_rlp_air → logs_list_concat_air → rlp_byte_concat_air →
/// receipt_rlp_air` chain.
///
/// The four sub-witnesses are the existing per-AIR witness types so
/// the standard `build_trace_polynomials` paths apply unchanged. The
/// chain assembler additionally caches:
///   - `receipt_payload_bytes`: the host-side byte stream of the
///     receipt-level RLP payload (status + cumul + bloom + logs region).
///     This is the byte stream the [`crate::rlp_byte_concat_air`]
///     witness covers.
///   - `logs_list_prefix_len`: 1 (short-list, `< 56` payload bytes) or
///     2 (long-list, `0xf8 N`). Threaded into the concat witness so
///     `absolute_offset_in_logs_list` is referenced from the same
///     origin in both the concat AIR and the receipt-level byte concat.
///   - `logs_list_payload_len`: the total bytes of `Σ_i encoded(log_i)`.
#[derive(Clone, Debug)]
pub struct ReceiptWithLogsChain {
    /// The receipt being chain-composed.
    pub receipt: Receipt,
    /// Per-log [`LogsRlpWitness`] (one witness covering all N logs;
    /// `rows[i]` = log `i`). Empty when the receipt has no logs.
    pub logs: LogsRlpWitness,
    /// Byte-granular logs-list witness with one row per byte of each
    /// log encoding, grouped by `log_index`. `list_prefix_len` is set
    /// so the first log's bytes start at offset `list_prefix_len`.
    pub logs_concat: LogsListConcatWitness,
    /// Byte-level offset chain covering the receipt's RLP payload.
    /// Fields appear in canonical receipt order:
    ///   `field 0`: status RLP
    ///   `field 1`: cumulative_gas_used RLP
    ///   `field 2`: logs_bloom RLP (259 bytes)
    ///   `field 3`: logs region RLP (`prefix || Σ encoded(log_i)`)
    pub byte_concat: RlpByteConcatWitness,
    /// The (no-logs) outer receipt AIR witness. For N ≥ 1 logs this is
    /// built only when the receipt has zero logs (matching the current
    /// AIR's no-logs-only constraint set). For non-empty logs this
    /// field carries `None`; the outer receipt-side binding is deferred
    /// until the AIR is widened (see module docs).
    pub receipt_outer: Option<ReceiptRlpWitness>,
    /// Host-side cached receipt payload bytes (status + cumul + bloom +
    /// logs region).
    pub receipt_payload_bytes: Vec<u8>,
    /// Length of the RLP list prefix for the logs list (1 or 2 bytes).
    pub logs_list_prefix_len: u64,
    /// Sum of `encoded(log_i)` over all logs (excludes the list prefix).
    pub logs_list_payload_len: u64,
    /// Absolute byte offset (within `receipt_payload_bytes`) at which
    /// the logs region (prefix + payload) begins.
    pub logs_region_offset: u64,
}

impl ReceiptWithLogsChain {
    /// Number of logs in the receipt.
    pub fn log_count(&self) -> usize { self.receipt.logs.len() }

    /// `true` iff (a) the host-side concat witness's offset equation
    /// holds, (b) its log grouping is consistent, and (c) the concat
    /// byte stream matches `Σ_i encoded(log_i)`.
    pub fn is_host_consistent(&self) -> bool {
        if self.logs_concat.verify_offset_equation().is_err() { return false; }
        if self.logs_concat.verify_log_grouping().is_err() { return false; }
        let mut payload = Vec::new();
        for l in &self.receipt.logs {
            payload.extend_from_slice(&l.rlp_encode());
        }
        if self.logs_concat.verify_against_canonical_payload(&payload).is_err() {
            return false;
        }
        // logs_list_prefix_len matches the canonical RLP prefix shape.
        let canonical_prefix_len = canonical_logs_list_prefix_len(self.logs_list_payload_len);
        if canonical_prefix_len != self.logs_list_prefix_len {
            return false;
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────
// Assembler
// ─────────────────────────────────────────────────────────────────────

/// Canonical RLP list-prefix length for a logs list whose payload (sum
/// of all encoded logs) is `payload_len` bytes:
///   - `< 56` → 1 byte (`0xc0 + len`)
///   - `56..=255` → 2 bytes (`0xf8 || len`)
///   - else → 3 bytes (`0xf9 || len_hi || len_lo`). Receipts almost
///     never hit this branch in practice but we handle it for
///     completeness.
fn canonical_logs_list_prefix_len(payload_len: u64) -> u64 {
    if payload_len < 56 { 1 } else if payload_len < 256 { 2 } else { 3 }
}

/// Emit the canonical RLP list prefix for a logs list of payload
/// length `payload_len`.
fn canonical_logs_list_prefix_bytes(payload_len: u64) -> Vec<u8> {
    if payload_len < 56 {
        vec![0xc0 + payload_len as u8]
    } else if payload_len < 256 {
        vec![0xf8, payload_len as u8]
    } else {
        vec![0xf9, (payload_len >> 8) as u8, (payload_len & 0xff) as u8]
    }
}

/// Build a [`ReceiptWithLogsChain`] from a host-supplied receipt.
///
/// The four sub-witnesses are populated as follows:
///   - `logs`: one [`crate::logs_rlp_air::LogsRlpRow`] per log.
///   - `logs_concat`: byte-granular concat of all per-log encodings,
///     with `list_prefix_len` set to the canonical RLP prefix size
///     of the logs list.
///   - `byte_concat`: byte-granular concat over the four receipt
///     payload fields, with `absolute_offset` indexing into the
///     receipt's RLP payload (NOT the wire encoding — the wire
///     encoding prepends an optional 1-byte type tag + 3-byte list
///     prefix, both of which the receipt AIR already handles).
///   - `receipt_outer`: `Some(ReceiptRlpWitness)` when `logs` is empty
///     (matching the current no-logs constraint set); `None` otherwise.
pub fn assemble_chain(receipt: &Receipt) -> ReceiptWithLogsChain {
    // Logs (per-log RLP AIR witness).
    let logs = LogsRlpWitness::from_logs(&receipt.logs)
        .expect("log topics/data within configured limits");

    // Sum of per-log encoded lengths = the logs list payload length.
    let logs_list_payload_len: u64 = receipt
        .logs
        .iter()
        .map(|l| l.rlp_encode().len() as u64)
        .sum();
    let logs_list_prefix_len = canonical_logs_list_prefix_len(logs_list_payload_len);
    let logs_list_prefix_bytes = canonical_logs_list_prefix_bytes(logs_list_payload_len);

    // Concat witness (byte-granular). list_prefix_len threads so the
    // first log's bytes start at offset `logs_list_prefix_len`.
    let logs_concat =
        LogsListConcatWitness::from_logs_with_prefix(&receipt.logs, logs_list_prefix_len);

    // Build the receipt payload bytes: status || cumul || bloom ||
    // logs_region. status / cumul use canonical u64-RLP. logs_bloom
    // uses the fixed 259-byte (0xb9 0x01 0x00 || 256 bytes) encoding.
    let status_enc = crate::u64_rlp_air::rlp_encode_u64(receipt.status as u64);
    let cumul_enc = crate::u64_rlp_air::rlp_encode_u64(receipt.cumulative_gas_used);
    let mut bloom_enc = Vec::with_capacity(crate::rlp_logs_bloom_air::ENCODED_LEN);
    bloom_enc.extend_from_slice(&crate::rlp_logs_bloom_air::PREFIX_BYTES);
    bloom_enc.extend_from_slice(&receipt.logs_bloom);

    let mut logs_region = Vec::with_capacity(
        logs_list_prefix_bytes.len() + logs_list_payload_len as usize,
    );
    logs_region.extend_from_slice(&logs_list_prefix_bytes);
    for l in &receipt.logs {
        logs_region.extend_from_slice(&l.rlp_encode());
    }
    let logs_region_offset = (status_enc.len() + cumul_enc.len() + bloom_enc.len()) as u64;

    let mut receipt_payload_bytes = Vec::new();
    receipt_payload_bytes.extend_from_slice(&status_enc);
    receipt_payload_bytes.extend_from_slice(&cumul_enc);
    receipt_payload_bytes.extend_from_slice(&bloom_enc);
    receipt_payload_bytes.extend_from_slice(&logs_region);

    // Build the byte-concat witness across the four payload fields.
    let mut running: u64 = 0;
    let fields: Vec<(u64, u64, Vec<u8>)> = {
        let mut v = Vec::with_capacity(4);
        let mut push = |idx: u64, bytes: Vec<u8>, running: &mut u64| {
            let len = bytes.len() as u64;
            v.push((idx, *running, bytes));
            *running += len;
        };
        push(0, status_enc.clone(), &mut running);
        push(1, cumul_enc.clone(), &mut running);
        push(2, bloom_enc.clone(), &mut running);
        push(3, logs_region.clone(), &mut running);
        v
    };
    let byte_concat = RlpByteConcatWitness::from_field_encodings(&fields);

    // Outer receipt AIR witness: only valid for the no-logs case under
    // the current constraint set.
    let receipt_outer = if receipt.logs.is_empty() {
        ReceiptRlpWitness::from_receipt(receipt).ok()
    } else {
        None
    };

    ReceiptWithLogsChain {
        receipt: receipt.clone(),
        logs,
        logs_concat,
        byte_concat,
        receipt_outer,
        receipt_payload_bytes,
        logs_list_prefix_len,
        logs_list_payload_len,
        logs_region_offset,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Cross-AIR LogUp descriptors
// ─────────────────────────────────────────────────────────────────────

/// Bind a per-log [`crate::logs_rlp_air`] row's emitted encoded bytes
/// to the [`crate::logs_list_concat_air`] rows for that log.
///
/// **Shape contract** — `logs_rlp_air` currently exposes per-log
/// `encoded_bytes` only via the witness (`LogsRlpRow::encoded_bytes`),
/// not as per-byte trace columns. A future adapter AIR will unfold
/// the row into `(byte_in_log, byte_value)` rows; that adapter's
/// `(log_index, byte_value)` tuple is the A-side of this descriptor
/// and the concat AIR's `(log_index, byte_value)` is the B-side.
///
/// This convenience wrapper calls
/// [`crate::logs_list_concat_air::make_per_log_to_concat_descriptor`]
/// with the canonical concat-side selector
/// ([`crate::logs_list_concat_air::COL_IS_REAL`]).
pub fn make_per_log_to_logs_concat_descriptor(
    log_index: u64,
    per_log_layer_index: usize,
    per_log_byte_value_cols: Vec<usize>,
    per_log_selector_col: Option<usize>,
    concat_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    crate::logs_list_concat_air::make_per_log_to_concat_descriptor(
        log_index,
        per_log_layer_index,
        per_log_byte_value_cols,
        per_log_selector_col,
        concat_layer_index,
        Some(LLC_IS_REAL),
    )
}

/// Bind [`crate::logs_list_concat_air`]'s
/// `(absolute_offset_in_logs_list, byte_value)` rows to
/// [`crate::rlp_byte_concat_air`]'s `(absolute_offset, byte_value)`
/// rows for the logs region.
///
/// Both AIRs reference the same `absolute_offset`, but the concat
/// AIR indexes from `0` at the **start of the logs list** while the
/// byte concat indexes from `0` at the **start of the receipt
/// payload**. Therefore the binding shifts by
/// `logs_region_offset` — the host-side offset of the logs region
/// within the receipt payload. Since cross-AIR LogUp tuples are
/// **equality-of-multisets**, the actual algebraic binding requires
/// an offset-shifted byte_concat selector (a per-row gate that fires
/// only for `field_index = 3` and adds `logs_region_offset` to the
/// absolute offset). The descriptor here documents the shape; the
/// in-flight offset rewriting is **deferred** until the byte_concat
/// AIR exposes a `field_3_logs_region_offset` column or a separate
/// adapter row.
///
/// **Tuple**: `(absolute_offset_in_logs_list, byte_value)` (2 cols
/// on each side).
pub fn make_logs_concat_to_byte_concat_descriptor(
    concat_layer_index: usize,
    byte_concat_layer_index: usize,
    byte_concat_logs_field_selector: Option<usize>,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "receipt_with_logs_concat_to_byte_concat_v1".into(),
        a_layer_index: concat_layer_index,
        a_columns: vec![LLC_ABS_OFFSET, LLC_BYTE_VALUE],
        a_selector_column: Some(LLC_IS_REAL),
        b_layer_index: byte_concat_layer_index,
        b_columns: vec![RBC_ABS_OFFSET, RBC_BYTE_VALUE],
        b_selector_column: byte_concat_logs_field_selector,
    }
}

/// Bind [`crate::rlp_byte_concat_air`]'s `(absolute_offset,
/// byte_value)` rows for the entire receipt payload to a future
/// widened receipt AIR's per-byte payload columns.
///
/// **Shape contract** — the receipt AIR currently exposes only
/// fixed-position single-byte columns (`LIST_PREFIX_0`,
/// `EMPTY_LOGS_BYTE`, …). Once it grows a contiguous
/// `payload_byte[0..MAX_PAYLOAD_LEN]` buffer (deferred follow-up),
/// the B side of this descriptor wires into that buffer.
///
/// **Tuple**: `(absolute_offset, byte_value)` (2 cols each side).
pub fn make_byte_concat_to_receipt_descriptor(
    byte_concat_layer_index: usize,
    receipt_layer_index: usize,
    receipt_position_col: usize,
    receipt_byte_value_col: usize,
    receipt_selector_col: Option<usize>,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "receipt_with_logs_byte_concat_to_receipt_v1".into(),
        a_layer_index: byte_concat_layer_index,
        a_columns: vec![RBC_ABS_OFFSET, RBC_BYTE_VALUE],
        a_selector_column: Some(RBC_IS_REAL),
        b_layer_index: receipt_layer_index,
        b_columns: vec![receipt_position_col, receipt_byte_value_col],
        b_selector_column: receipt_selector_col,
    }
}

/// Convenience wrapper over
/// [`crate::logs_list_concat_air::make_concat_to_per_log_len_descriptor`]
/// binding this AIR's per-log `(log_index, log_encoded_len)` to a
/// per-log AIR row.
pub fn make_logs_concat_to_per_log_len_descriptor(
    concat_layer_index: usize,
    per_log_layer_index: usize,
    per_log_log_index_col: usize,
    per_log_encoded_len_col: usize,
    per_log_selector_col: Option<usize>,
) -> CrossAirLogUpDescriptor {
    crate::logs_list_concat_air::make_concat_to_per_log_len_descriptor(
        concat_layer_index,
        per_log_layer_index,
        per_log_log_index_col,
        per_log_encoded_len_col,
        per_log_selector_col,
    )
}

/// Collect the chain descriptors at concrete layer indices.
///
/// Layer indices: `(logs_rlp, logs_concat, byte_concat, receipt_rlp)`.
///
/// For a chain with `n_logs` logs the returned vector has length
/// `n_logs + 3`:
///   - one `per_log_i → logs_concat` descriptor per log (`n_logs`)
///   - `logs_concat → byte_concat` (1)
///   - `byte_concat → receipt_rlp` (1)
///   - `logs_concat → per_log[0] len` (1)
///
/// The per-log byte-value column lists are provided by the caller; the
/// caller must thread the per-log AIR's per-byte trace columns once an
/// adapter AIR exists. For the shape contract test below we pass an
/// empty `per_log_byte_value_cols` and `None` selectors.
pub fn collect_descriptors(
    n_logs: usize,
    logs_rlp_layer_index: usize,
    logs_concat_layer_index: usize,
    byte_concat_layer_index: usize,
    receipt_layer_index: usize,
    per_log_byte_value_cols: &[Vec<usize>],
    per_log_selector_cols: &[Option<usize>],
    byte_concat_logs_field_selector: Option<usize>,
    receipt_position_col: usize,
    receipt_byte_value_col: usize,
    receipt_selector_col: Option<usize>,
    per_log_log_index_col: usize,
    per_log_encoded_len_col: usize,
    per_log_concat_len_selector: Option<usize>,
) -> Vec<CrossAirLogUpDescriptor> {
    let mut out = Vec::with_capacity(n_logs + 3);
    for i in 0..n_logs {
        let cols = per_log_byte_value_cols
            .get(i)
            .cloned()
            .unwrap_or_default();
        let sel = per_log_selector_cols.get(i).copied().unwrap_or(None);
        out.push(make_per_log_to_logs_concat_descriptor(
            i as u64,
            logs_rlp_layer_index,
            cols,
            sel,
            logs_concat_layer_index,
        ));
    }
    out.push(make_logs_concat_to_byte_concat_descriptor(
        logs_concat_layer_index,
        byte_concat_layer_index,
        byte_concat_logs_field_selector,
    ));
    out.push(make_byte_concat_to_receipt_descriptor(
        byte_concat_layer_index,
        receipt_layer_index,
        receipt_position_col,
        receipt_byte_value_col,
        receipt_selector_col,
    ));
    out.push(make_logs_concat_to_per_log_len_descriptor(
        logs_concat_layer_index,
        logs_rlp_layer_index,
        per_log_log_index_col,
        per_log_encoded_len_col,
        per_log_concat_len_selector,
    ));
    out
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipt::{Log, Receipt, ReceiptType};

    fn empty_receipt() -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }
    }

    fn log_simple() -> Log {
        Log {
            address: [0x11; 20],
            topics: vec![[0xaa; 32]],
            data: vec![0xde, 0xad, 0xbe, 0xef],
        }
    }

    fn log_no_topics() -> Log {
        Log { address: [0x22; 20], topics: vec![], data: vec![] }
    }

    fn log_two_topics() -> Log {
        Log {
            address: [0x33; 20],
            topics: vec![[0x55; 32], [0x66; 32]],
            data: vec![0xff; 16],
        }
    }

    fn receipt_with_logs(logs: Vec<Log>) -> Receipt {
        Receipt {
            ty: ReceiptType::Eip1559,
            status: 1,
            cumulative_gas_used: 100_000,
            logs_bloom: [0x11; 256],
            logs,
        }
    }

    // ── Chain assembly ───────────────────────────────────────────────

    #[test]
    fn chain_assembles_for_zero_logs() {
        let r = empty_receipt();
        let chain = assemble_chain(&r);
        assert_eq!(chain.log_count(), 0);
        assert!(chain.logs.rows.is_empty());
        assert!(chain.logs_concat.rows.is_empty());
        // byte_concat covers status + cumul + bloom + 1-byte logs region (0xc0).
        let expected_payload_len = chain.receipt_payload_bytes.len();
        assert_eq!(chain.byte_concat.rows.len(), expected_payload_len);
        // Empty-logs region = single 0xc0 byte.
        assert_eq!(chain.logs_list_payload_len, 0);
        assert_eq!(chain.logs_list_prefix_len, 1);
        // receipt_outer present (no-logs case is supported by the AIR).
        assert!(chain.receipt_outer.is_some());
        assert!(chain.is_host_consistent());
    }

    #[test]
    fn chain_assembles_for_one_log() {
        let r = receipt_with_logs(vec![log_simple()]);
        let chain = assemble_chain(&r);
        assert_eq!(chain.log_count(), 1);
        assert_eq!(chain.logs.rows.len(), 1);
        let expected_log_len = r.logs[0].rlp_encode().len() as u64;
        assert_eq!(chain.logs_list_payload_len, expected_log_len);
        // logs_concat rows = bytes of the (one) log.
        assert_eq!(chain.logs_concat.rows.len() as u64, expected_log_len);
        // First log row indexes at absolute = list_prefix_len.
        let first = &chain.logs_concat.rows[0];
        assert_eq!(first.log_index, 0);
        assert_eq!(first.byte_in_log, 0);
        assert_eq!(first.absolute_offset_in_logs_list, chain.logs_list_prefix_len);
        // receipt_outer is None (current AIR is no-logs only).
        assert!(chain.receipt_outer.is_none());
        assert!(chain.is_host_consistent());
    }

    #[test]
    fn chain_assembles_for_three_logs() {
        let r = receipt_with_logs(vec![log_no_topics(), log_simple(), log_two_topics()]);
        let chain = assemble_chain(&r);
        assert_eq!(chain.log_count(), 3);
        assert_eq!(chain.logs.rows.len(), 3);
        let lens: Vec<u64> = r.logs.iter().map(|l| l.rlp_encode().len() as u64).collect();
        let total: u64 = lens.iter().sum();
        assert_eq!(chain.logs_list_payload_len, total);
        assert_eq!(chain.logs_concat.rows.len() as u64, total);
        // Per-log encoded lens cached.
        assert_eq!(chain.logs_concat.log_encoded_lens, lens);
        // logs_list_prefix_len is 1 (short list, payload < 56) or 2.
        let expected_prefix = if total < 56 { 1 } else if total < 256 { 2 } else { 3 };
        assert_eq!(chain.logs_list_prefix_len, expected_prefix);
        assert!(chain.is_host_consistent());
    }

    // ── Descriptor count grows with log count ─────────────────────────

    #[test]
    fn descriptor_count_grows_with_log_count() {
        for n_logs in 0..=5 {
            let descs = collect_descriptors(
                n_logs,
                0, 1, 2, 3,
                &[],
                &[],
                None,
                0,
                0,
                None,
                0,
                0,
                None,
            );
            // n_logs (per-log → concat) + concat→byte_concat + byte_concat→receipt + concat→per_log_len = n_logs + 3.
            assert_eq!(descs.len(), n_logs + 3, "n_logs = {}", n_logs);
        }
    }

    // ── All descriptors distinct labels ──────────────────────────────

    #[test]
    fn descriptors_have_distinct_labels() {
        let descs = collect_descriptors(
            4,
            0, 1, 2, 3,
            &[],
            &[],
            None,
            0,
            0,
            None,
            0,
            0,
            None,
        );
        let labels: Vec<&str> = descs.iter().map(|d| d.label.as_str()).collect();
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                assert_ne!(
                    labels[i], labels[j],
                    "duplicate descriptor label at positions {} and {}: {}",
                    i, j, labels[i],
                );
            }
        }
        // Spot-check expected labels.
        assert!(labels.iter().any(|l| l == &"logs_list_concat_log_0_byte_v1"));
        assert!(labels.iter().any(|l| l == &"logs_list_concat_log_3_byte_v1"));
        assert!(labels.iter().any(|l| l == &"receipt_with_logs_concat_to_byte_concat_v1"));
        assert!(labels.iter().any(|l| l == &"receipt_with_logs_byte_concat_to_receipt_v1"));
        assert!(labels.iter().any(|l| l == &"logs_list_concat_to_per_log_len_v1"));
    }

    // ── Chain witness consistency ────────────────────────────────────

    #[test]
    fn chain_witness_consistency_three_logs() {
        let r = receipt_with_logs(vec![log_simple(), log_no_topics(), log_two_topics()]);
        let chain = assemble_chain(&r);
        // Host-side: each per-log row's encoded length matches the concat
        // AIR's cached per-log length.
        for (i, log_row) in chain.logs.rows.iter().enumerate() {
            assert_eq!(
                log_row.encoded_len as u64,
                chain.logs_concat.log_encoded_lens[i],
            );
        }
        // byte_concat covers exactly status + cumul + bloom + logs_region.
        let status_enc = crate::u64_rlp_air::rlp_encode_u64(r.status as u64);
        let cumul_enc = crate::u64_rlp_air::rlp_encode_u64(r.cumulative_gas_used);
        let bloom_enc_len = crate::rlp_logs_bloom_air::ENCODED_LEN;
        let prefix_bytes = canonical_logs_list_prefix_bytes(chain.logs_list_payload_len);
        let logs_region_len = prefix_bytes.len() as u64 + chain.logs_list_payload_len;
        let expected_payload_len = (status_enc.len() + cumul_enc.len()) as u64
            + bloom_enc_len as u64
            + logs_region_len;
        assert_eq!(chain.byte_concat.rows.len() as u64, expected_payload_len);
        // logs_region_offset within receipt payload is fixed.
        assert_eq!(
            chain.logs_region_offset,
            (status_enc.len() + cumul_enc.len() + bloom_enc_len) as u64,
        );
        // Last byte_concat row's absolute_offset equals payload_len - 1.
        let last = chain.byte_concat.rows.last().unwrap();
        assert_eq!(last.absolute_offset, expected_payload_len - 1);
        // Last byte_concat row's field_index = 3 (the logs region).
        assert_eq!(last.field_index, 3);
        // logs_concat byte stream matches canonical Σ_i encoded(log_i).
        let mut canonical_payload = Vec::new();
        for l in &r.logs {
            canonical_payload.extend_from_slice(&l.rlp_encode());
        }
        chain
            .logs_concat
            .verify_against_canonical_payload(&canonical_payload)
            .unwrap();
    }

    // ── Per-log descriptor shape ─────────────────────────────────────

    #[test]
    fn per_log_descriptor_uses_concat_is_real_selector() {
        let d = make_per_log_to_logs_concat_descriptor(
            7,
            1,
            vec![100, 101],
            Some(99),
            0,
        );
        assert_eq!(d.label, "logs_list_concat_log_7_byte_v1");
        assert_eq!(d.a_layer_index, 1);
        assert_eq!(d.a_columns, vec![100, 101]);
        assert_eq!(d.a_selector_column, Some(99));
        assert_eq!(d.b_layer_index, 0);
        assert_eq!(d.b_selector_column, Some(LLC_IS_REAL));
    }

    // ── concat → byte_concat descriptor shape ────────────────────────

    #[test]
    fn concat_to_byte_concat_descriptor_well_formed() {
        let d = make_logs_concat_to_byte_concat_descriptor(0, 1, Some(42));
        assert_eq!(d.label, "receipt_with_logs_concat_to_byte_concat_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.a_columns, vec![LLC_ABS_OFFSET, LLC_BYTE_VALUE]);
        assert_eq!(d.a_selector_column, Some(LLC_IS_REAL));
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.b_columns, vec![RBC_ABS_OFFSET, RBC_BYTE_VALUE]);
        assert_eq!(d.b_selector_column, Some(42));
        assert_eq!(d.a_columns.len(), d.b_columns.len());
    }

    // ── byte_concat → receipt descriptor shape ───────────────────────

    #[test]
    fn byte_concat_to_receipt_descriptor_well_formed() {
        let d = make_byte_concat_to_receipt_descriptor(2, 3, 555, 777, Some(8));
        assert_eq!(d.label, "receipt_with_logs_byte_concat_to_receipt_v1");
        assert_eq!(d.a_layer_index, 2);
        assert_eq!(d.a_columns, vec![RBC_ABS_OFFSET, RBC_BYTE_VALUE]);
        assert_eq!(d.a_selector_column, Some(RBC_IS_REAL));
        assert_eq!(d.b_layer_index, 3);
        assert_eq!(d.b_columns, vec![555, 777]);
        assert_eq!(d.b_selector_column, Some(8));
    }

    // ── concat → per-log length convenience wrapper ──────────────────

    #[test]
    fn concat_to_per_log_len_wrapper_matches_underlying() {
        let d = make_logs_concat_to_per_log_len_descriptor(0, 1, 10, 11, Some(12));
        assert_eq!(d.label, "logs_list_concat_to_per_log_len_v1");
        assert_eq!(d.a_columns, vec![LLC_LOG_INDEX, LLC_LOG_ENC_LEN]);
        assert_eq!(d.b_columns, vec![10, 11]);
        assert_eq!(d.b_selector_column, Some(12));
    }

    // ── Canonical prefix-len helper ──────────────────────────────────

    #[test]
    fn canonical_logs_list_prefix_len_branches() {
        assert_eq!(canonical_logs_list_prefix_len(0), 1);
        assert_eq!(canonical_logs_list_prefix_len(1), 1);
        assert_eq!(canonical_logs_list_prefix_len(55), 1);
        assert_eq!(canonical_logs_list_prefix_len(56), 2);
        assert_eq!(canonical_logs_list_prefix_len(255), 2);
        assert_eq!(canonical_logs_list_prefix_len(256), 3);
    }

    #[test]
    fn canonical_logs_list_prefix_bytes_branches() {
        assert_eq!(canonical_logs_list_prefix_bytes(0), vec![0xc0]);
        assert_eq!(canonical_logs_list_prefix_bytes(40), vec![0xc0 + 40]);
        assert_eq!(canonical_logs_list_prefix_bytes(100), vec![0xf8, 100]);
        assert_eq!(canonical_logs_list_prefix_bytes(300), vec![0xf9, 0x01, 0x2c]);
    }
}
