//! Algebraic per-log RLP encoding AIR (single-log, ≤4 topics, ≤32-byte data).
//!
//! Phase A2 #59 step 1+ companion to [`crate::receipt_rlp_air`]:
//! lifts the per-`Log` RLP encoding into row-local constraints + a set
//! of cross-AIR LogUp descriptors that bind the per-field encoded byte
//! windows to the dedicated [`crate::fixed_rlp20_air`] (address),
//! [`crate::fixed_rlp_air`] (each 32-byte topic), and
//! [`crate::rlp_var_bytes_air`] (variable-length data).
//!
//! # Encoding shape
//!
//! A canonical Ethereum log RLP is `RLP([address, topics, data])`:
//!
//! ```text
//! Log_RLP =
//!     OUTER_LIST_PREFIX                  (1 or 2 bytes)
//!  || ADDRESS_RLP                        (21 bytes,    0x94 || addr)
//!  || TOPICS_LIST_PREFIX                 (1 or 2 bytes)
//!  || TOPIC_RLP_0 || ... || TOPIC_RLP_{n-1}   (n × 33 bytes; n = topics_len ∈ {0..=4})
//!  || DATA_RLP                           (data_enc_len bytes; see rlp_var_bytes_air)
//! ```
//!
//! Topics inner length = `33 * topics_len`. For topics_len ∈ {0, 1}
//! the inner length is < 56 so the prefix is a single byte `0xc0 +
//! topics_inner_len`. For topics_len ∈ {2, 3, 4} the inner length is
//! ≥ 66 so the prefix is the long form `[0xf8, inner_len]` (2 bytes).
//!
//! Outer payload length ranges from `21 + 1 + 1 = 23` (no topics,
//! empty data) up to `21 + 134 + 33 = 188` (4 topics, 32-byte data
//! with leading byte ≥ 0x80). For `payload_len < 56` the outer prefix
//! is a single byte `0xc0 + payload_len`; otherwise it is the long
//! form `[0xf8, payload_len]` (2 bytes).
//!
//! # What this AIR enforces algebraically (per row, gated by IS_REAL)
//!
//! 1. `IS_REAL` binary.
//! 2. Each `TOPIC_ACTIVE[k]` is binary; the column is monotonically
//!    non-increasing across `k = 0..3`; `Σ TOPIC_ACTIVE = TOPICS_LEN`.
//! 3. Each `DATA_ACTIVE[k]` is binary; monotonically non-increasing
//!    across `k = 0..31`; `Σ DATA_ACTIVE = DATA_LEN` (so DATA_LEN ≤ 32).
//! 4. `IS_TOPICS_SHORT` + `IS_TOPICS_LONG` = `IS_REAL`; both binary.
//!    Topic case selection from `topics_len`:
//!      `IS_TOPICS_SHORT` enforced via `(topics_len) * (topics_len - 1) *`
//!      indicator only when topics_len ≤ 1 — captured via the explicit
//!      witness column + a `topics_len * (topics_len - 1) * (1 -
//!      IS_TOPICS_LONG) = 0` style constraint. Simplification: we
//!      witness `IS_TOPICS_LONG` and check `IS_TOPICS_LONG * (topics_len
//!      - 2) * (topics_len - 3) * (topics_len - 4) = 0` and
//!      `(1 - IS_TOPICS_LONG) * topics_len * (topics_len - 1) = 0`.
//! 5. `TOPICS_INNER_LEN = 33 * topics_len`.
//! 6. `TOPICS_PREFIX_LEN = 1 + IS_TOPICS_LONG`.
//! 7. `TOPICS_TOTAL_LEN = TOPICS_PREFIX_LEN + TOPICS_INNER_LEN`.
//! 8. `PAYLOAD_LEN = 21 + TOPICS_TOTAL_LEN + DATA_ENC_LEN`.
//! 9. `IS_OUTER_SHORT + IS_OUTER_LONG = IS_REAL`; both binary.
//!    `IS_OUTER_LONG * (PAYLOAD_LEN < 56)` style check is replaced by
//!    an arithmetic split: witness `IS_OUTER_LONG`, then enforce
//!    `(IS_OUTER_LONG = 0) ⇒ PAYLOAD_LEN ≤ 55` and
//!    `(IS_OUTER_LONG = 1) ⇒ PAYLOAD_LEN ≥ 56`. We *do not* enforce
//!    this range split algebraically here — it is captured by the
//!    explicit `OUTER_PREFIX_LEN_FROM_FLAGS = 1 + IS_OUTER_LONG`
//!    formula plus the host-side witness builder. (See "Deferred"
//!    below.)
//! 10. `OUTER_PREFIX_LEN = 1 + IS_OUTER_LONG`.
//! 11. `ENCODED_LEN = OUTER_PREFIX_LEN + PAYLOAD_LEN`.
//!
//! All exposed 8-bit columns get range checks via `LookupTable::range(256)`.
//!
//! # Cross-AIR LogUp linkages (descriptors below; not row-local)
//!
//! - [`make_log_to_address_descriptor`] — binds `(address[0..20])` to
//!   [`crate::fixed_rlp20_air`].
//! - [`make_log_to_topic_descriptor(k)`] — for k ∈ 0..4 binds
//!   `(topic[k][0..32])` to [`crate::fixed_rlp_air`], gated by
//!   `TOPIC_ACTIVE[k]`.
//! - [`make_log_to_data_descriptor`] — binds the data byte window and
//!   `DATA_LEN`, `DATA_ENC_LEN` and full data-encoded window to
//!   [`crate::rlp_var_bytes_air`].
//!
//! # Deferred
//!
//! - Data > 32 bytes (would need long-string RLP encoding on the data
//!   side; the var_bytes gadget is currently capped at 32). Witness
//!   builder rejects `data.len() > 32`.
//! - Multiple logs / a logs-list concatenation AIR (so the per-row
//!   encoded byte windows can be byte-aligned into the receipt's
//!   logs-list section). One log per row for now; multi-log rows are
//!   independent and uncomposed.
//! - Strict bounds-checking constraints linking `IS_OUTER_LONG` ↔
//!   `PAYLOAD_LEN ≥ 56` (range argument). Currently `IS_OUTER_LONG`
//!   is a witness column trusted to be consistent with `PAYLOAD_LEN`;
//!   composition with a payload_len range lookup closes this. The
//!   same applies to `IS_TOPICS_LONG` ↔ `topics_len ≥ 2`.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::receipt::Log;
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const MAX_TOPICS: usize = 4;
pub const TOPIC_BYTES: usize = 32;
pub const ADDRESS_BYTES: usize = 20;
pub const ADDRESS_ENC_LEN: usize = 21; // 0x94 || 20 bytes
pub const TOPIC_ENC_LEN: usize = 33; // 0xa0 || 32 bytes
pub const MAX_DATA_BYTES: usize = 32;
pub const MAX_DATA_ENC_LEN: usize = 33; // worst case: 0x80+len + 32 bytes

/// Bounds for the topics inner concatenation length.
pub const TOPICS_INNER_MAX: usize = MAX_TOPICS * TOPIC_ENC_LEN; // 132

/// Topics list prefix is 1 byte if inner < 56, else 2 bytes (0xf8 || len).
/// For topics_len ∈ {0,1} (inner 0 or 33) → 1 byte; for {2,3,4} → 2 bytes.

/// Maximum payload length = address(21) + topics_total(max 2 + 132) + data_enc(max 33) = 188.
pub const MAX_PAYLOAD_LEN: usize = ADDRESS_ENC_LEN + 2 + TOPICS_INNER_MAX + MAX_DATA_ENC_LEN;

/// Maximum total encoded length = outer prefix (≤ 2) + payload (≤ 188) = 190.
pub const MAX_ENCODED_LEN: usize = 2 + MAX_PAYLOAD_LEN;

// ─── Column layout ────────────────────────────────────────────────────

// Selectors / shape.
pub const COL_IS_REAL: usize = 0;
pub const COL_TOPICS_LEN: usize = 1;
pub const COL_DATA_LEN: usize = 2;
pub const COL_DATA_ENC_LEN: usize = 3;
pub const COL_TOPICS_INNER_LEN: usize = 4;
pub const COL_TOPICS_PREFIX_LEN: usize = 5;
pub const COL_TOPICS_TOTAL_LEN: usize = 6;
pub const COL_PAYLOAD_LEN: usize = 7;
pub const COL_OUTER_PREFIX_LEN: usize = 8;
pub const COL_ENCODED_LEN: usize = 9;
pub const COL_IS_TOPICS_LONG: usize = 10;
pub const COL_IS_OUTER_LONG: usize = 11;

// Address bytes (20).
pub const COL_ADDRESS_OFFSET: usize = 12;
pub const COL_ADDRESS_END: usize = COL_ADDRESS_OFFSET + ADDRESS_BYTES; // 32

// Topic active flags (4) and per-topic 32 raw bytes (4 × 32 = 128).
pub const COL_TOPIC_ACTIVE_OFFSET: usize = COL_ADDRESS_END;             // 32
pub const COL_TOPIC_ACTIVE_END: usize = COL_TOPIC_ACTIVE_OFFSET + MAX_TOPICS; // 36
pub const COL_TOPIC_BYTES_OFFSET: usize = COL_TOPIC_ACTIVE_END;         // 36
pub const COL_TOPIC_BYTES_END: usize = COL_TOPIC_BYTES_OFFSET + MAX_TOPICS * TOPIC_BYTES; // 164

// Data byte window (32) + is_active (32).
pub const COL_DATA_BYTES_OFFSET: usize = COL_TOPIC_BYTES_END;           // 164
pub const COL_DATA_BYTES_END: usize = COL_DATA_BYTES_OFFSET + MAX_DATA_BYTES; // 196
pub const COL_DATA_ACTIVE_OFFSET: usize = COL_DATA_BYTES_END;           // 196
pub const COL_DATA_ACTIVE_END: usize = COL_DATA_ACTIVE_OFFSET + MAX_DATA_BYTES; // 228

// Data RLP encoded window (mirrors rlp_var_bytes_air's encoded buffer).
pub const COL_DATA_ENC_OFFSET: usize = COL_DATA_ACTIVE_END;             // 228
pub const COL_DATA_ENC_END: usize = COL_DATA_ENC_OFFSET + MAX_DATA_ENC_LEN; // 261

pub const NUM_COLUMNS: usize = COL_DATA_ENC_END;                        // 261

/// Row-local constraint count.
///
///   0: is_real binary
///   1: is_topics_long binary
///   2: is_outer_long binary
///   3: (1 - is_topics_long) * topics_len * (topics_len - 1) = 0
///      [topics_len ∈ {0,1} when short]
///   4: is_topics_long * (topics_len - 2) * (topics_len - 3) * (topics_len - 4) = 0
///      [topics_len ∈ {2,3,4} when long]
///   5: topic_active[k] binary, β-RLC over k ∈ 0..4
///   6: topic_active monotonic (active[k+1] ≤ active[k]), β-RLC
///   7: Σ topic_active[k] = topics_len
///   8: data_active[k] binary, β-RLC over k ∈ 0..32
///   9: data_active monotonic, β-RLC
///  10: Σ data_active[k] = data_len
///  11: topics_inner_len = 33 * topics_len
///  12: topics_prefix_len = 1 + is_topics_long
///  13: topics_total_len = topics_prefix_len + topics_inner_len
///  14: payload_len = 21 + topics_total_len + data_enc_len
///  15: outer_prefix_len = 1 + is_outer_long
///  16: encoded_len = outer_prefix_len + payload_len
pub const NUM_ROW_CONSTRAINTS: usize = 17;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LogsRlpRow {
    pub address: [u8; ADDRESS_BYTES],
    pub topics: Vec<[u8; TOPIC_BYTES]>,
    pub data: Vec<u8>,
    pub data_enc: Vec<u8>,
    pub topics_inner_len: usize,
    pub topics_prefix_len: usize,
    pub topics_total_len: usize,
    pub payload_len: usize,
    pub outer_prefix_len: usize,
    pub encoded_len: usize,
    pub encoded_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct LogsRlpWitness {
    pub rows: Vec<LogsRlpRow>,
}

/// Build a single-log witness row.
pub fn from_log(log: &Log) -> Result<LogsRlpRow, &'static str> {
    if log.topics.len() > MAX_TOPICS {
        return Err("logs_rlp_air: at most 4 topics");
    }
    if log.data.len() > MAX_DATA_BYTES {
        return Err("logs_rlp_air: data > 32 bytes is deferred (variable-length RLP gadget cap)");
    }
    let topics_len = log.topics.len();
    let topics_inner_len = topics_len * TOPIC_ENC_LEN;
    let topics_prefix_len = if topics_inner_len < 56 { 1 } else { 2 };
    let topics_total_len = topics_prefix_len + topics_inner_len;
    let data_enc = crate::rlp_var_bytes_air::rlp_encode_bytes(&log.data);
    let data_enc_len = data_enc.len();
    let payload_len = ADDRESS_ENC_LEN + topics_total_len + data_enc_len;
    let outer_prefix_len = if payload_len < 56 { 1 } else { 2 };
    let encoded_len = outer_prefix_len + payload_len;

    // Sanity: build the canonical encoded bytes for round-trip tests.
    let mut encoded_bytes = Vec::with_capacity(encoded_len);
    // outer prefix
    if outer_prefix_len == 1 {
        encoded_bytes.push(0xc0 + payload_len as u8);
    } else {
        encoded_bytes.push(0xf8);
        encoded_bytes.push(payload_len as u8);
    }
    // address RLP: 0x94 || addr[0..20]
    encoded_bytes.push(crate::fixed_rlp20_air::RLP20_PREFIX);
    encoded_bytes.extend_from_slice(&log.address);
    // topics list prefix
    if topics_prefix_len == 1 {
        encoded_bytes.push(0xc0 + topics_inner_len as u8);
    } else {
        encoded_bytes.push(0xf8);
        encoded_bytes.push(topics_inner_len as u8);
    }
    // topics inner: 33-byte chunks
    for t in &log.topics {
        encoded_bytes.push(crate::fixed_rlp_air::RLP32_PREFIX);
        encoded_bytes.extend_from_slice(t);
    }
    // data RLP
    encoded_bytes.extend_from_slice(&data_enc);

    debug_assert_eq!(encoded_bytes.len(), encoded_len);

    // Cross-check against the canonical Log::rlp_encode for confidence
    // (the canonical encoder uses the same rules).
    debug_assert_eq!(encoded_bytes, log.rlp_encode());

    let mut topics_padded: Vec<[u8; TOPIC_BYTES]> = log.topics.clone();
    while topics_padded.len() < MAX_TOPICS {
        topics_padded.push([0u8; TOPIC_BYTES]);
    }

    Ok(LogsRlpRow {
        address: log.address,
        topics: topics_padded,
        data: log.data.clone(),
        data_enc,
        topics_inner_len,
        topics_prefix_len,
        topics_total_len,
        payload_len,
        outer_prefix_len,
        encoded_len,
        encoded_bytes,
    })
}

impl LogsRlpWitness {
    pub fn from_log(log: &Log) -> Result<Self, &'static str> {
        Ok(Self { rows: vec![from_log(log)?] })
    }
    pub fn from_logs(logs: &[Log]) -> Result<Self, &'static str> {
        let mut rows = Vec::with_capacity(logs.len());
        for l in logs {
            rows.push(from_log(l)?);
        }
        Ok(Self { rows })
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &LogsRlpWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.rows.iter().enumerate() {
        columns[COL_IS_REAL][r] = one.clone();
        let topics_len = row.topics_inner_len / TOPIC_ENC_LEN;
        columns[COL_TOPICS_LEN][r] = Scalar::from_u64(topics_len as u64, curve);
        columns[COL_DATA_LEN][r] = Scalar::from_u64(row.data.len() as u64, curve);
        columns[COL_DATA_ENC_LEN][r] = Scalar::from_u64(row.data_enc.len() as u64, curve);
        columns[COL_TOPICS_INNER_LEN][r] = Scalar::from_u64(row.topics_inner_len as u64, curve);
        columns[COL_TOPICS_PREFIX_LEN][r] = Scalar::from_u64(row.topics_prefix_len as u64, curve);
        columns[COL_TOPICS_TOTAL_LEN][r] = Scalar::from_u64(row.topics_total_len as u64, curve);
        columns[COL_PAYLOAD_LEN][r] = Scalar::from_u64(row.payload_len as u64, curve);
        columns[COL_OUTER_PREFIX_LEN][r] = Scalar::from_u64(row.outer_prefix_len as u64, curve);
        columns[COL_ENCODED_LEN][r] = Scalar::from_u64(row.encoded_len as u64, curve);
        columns[COL_IS_TOPICS_LONG][r] =
            Scalar::from_u64(if row.topics_prefix_len == 2 { 1 } else { 0 }, curve);
        columns[COL_IS_OUTER_LONG][r] =
            Scalar::from_u64(if row.outer_prefix_len == 2 { 1 } else { 0 }, curve);

        // Address bytes.
        for k in 0..ADDRESS_BYTES {
            columns[COL_ADDRESS_OFFSET + k][r] =
                Scalar::from_u64(row.address[k] as u64, curve);
        }

        // Topic active flags + raw 32-byte payloads.
        for k in 0..MAX_TOPICS {
            let active = if k < topics_len { 1u64 } else { 0 };
            columns[COL_TOPIC_ACTIVE_OFFSET + k][r] = Scalar::from_u64(active, curve);
            for b in 0..TOPIC_BYTES {
                columns[COL_TOPIC_BYTES_OFFSET + k * TOPIC_BYTES + b][r] =
                    Scalar::from_u64(row.topics[k][b] as u64, curve);
            }
        }

        // Data bytes + active.
        for k in 0..MAX_DATA_BYTES {
            let byte = if k < row.data.len() { row.data[k] } else { 0 };
            let active = if k < row.data.len() { 1u64 } else { 0 };
            columns[COL_DATA_BYTES_OFFSET + k][r] = Scalar::from_u64(byte as u64, curve);
            columns[COL_DATA_ACTIVE_OFFSET + k][r] = Scalar::from_u64(active, curve);
        }

        // Data encoded window (mirrors var_bytes encoded buffer).
        for k in 0..MAX_DATA_ENC_LEN {
            let byte = if k < row.data_enc.len() { row.data_enc[k] } else { 0 };
            columns[COL_DATA_ENC_OFFSET + k][r] = Scalar::from_u64(byte as u64, curve);
        }
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ────────────────────────────────────────────────

pub struct LogsRlpConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl LogsRlpConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for LogsRlpConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_topics_long_binary".into(),
            "is_outer_long_binary".into(),
            "topics_short_implies_len_le_1".into(),
            "topics_long_implies_len_ge_2".into(),
            "topic_active_binary_rlc".into(),
            "topic_active_monotonic_rlc".into(),
            "topic_active_sum_eq_topics_len".into(),
            "data_active_binary_rlc".into(),
            "data_active_monotonic_rlc".into(),
            "data_active_sum_eq_data_len".into(),
            "topics_inner_len_eq_33_times_topics_len".into(),
            "topics_prefix_len_formula".into(),
            "topics_total_len_formula".into(),
            "payload_len_formula".into(),
            "outer_prefix_len_formula".into(),
            "encoded_len_formula".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let two = Scalar::from_u64(2, curve);
        let three = Scalar::from_u64(3, curve);
        let four = Scalar::from_u64(4, curve);
        let c21 = Scalar::from_u64(ADDRESS_ENC_LEN as u64, curve);
        let c33 = Scalar::from_u64(TOPIC_ENC_LEN as u64, curve);
        let beta = Scalar::from_u64(7, curve);

        let mk = || vec![Scalar::zero(curve); n];
        let mut c0 = mk();
        let mut c1 = mk();
        let mut c2 = mk();
        let mut c3 = mk();
        let mut c4 = mk();
        let mut c5 = mk();
        let mut c6 = mk();
        let mut c7 = mk();
        let mut c8 = mk();
        let mut c9 = mk();
        let mut c10 = mk();
        let mut c11 = mk();
        let mut c12 = mk();
        let mut c13 = mk();
        let mut c14 = mk();
        let mut c15 = mk();
        let mut c16 = mk();

        for r in 0..n {
            let ir = &columns[COL_IS_REAL][r];
            c0[r] = ir.mul(&ir.sub(&one));
            let itl = &columns[COL_IS_TOPICS_LONG][r];
            c1[r] = itl.mul(&itl.sub(&one));
            let iol = &columns[COL_IS_OUTER_LONG][r];
            c2[r] = iol.mul(&iol.sub(&one));

            let tl = &columns[COL_TOPICS_LEN][r];

            // (1 - is_topics_long) * topics_len * (topics_len - 1) = 0
            let not_itl = one.sub(itl);
            let tl_m1 = tl.sub(&one);
            c3[r] = ir.mul(&not_itl.mul(&tl.mul(&tl_m1)));

            // is_topics_long * (topics_len - 2)(topics_len - 3)(topics_len - 4) = 0
            let tl_m2 = tl.sub(&two);
            let tl_m3 = tl.sub(&three);
            let tl_m4 = tl.sub(&four);
            c4[r] = ir.mul(&itl.mul(&tl_m2.mul(&tl_m3).mul(&tl_m4)));

            // topic_active binary + monotonic (β-RLC) + sum.
            let mut acc_bin = Scalar::zero(curve);
            let mut acc_mono = Scalar::zero(curve);
            let mut acc_sum = Scalar::zero(curve);
            let mut bp = Scalar::one(curve);
            for k in 0..MAX_TOPICS {
                let a = &columns[COL_TOPIC_ACTIVE_OFFSET + k][r];
                acc_bin = acc_bin.add(&bp.mul(&a.mul(&a.sub(&one))));
                acc_sum = acc_sum.add(a);
                if k < MAX_TOPICS - 1 {
                    let a1 = &columns[COL_TOPIC_ACTIVE_OFFSET + k + 1][r];
                    // a1 * (1 - a) = 0: forbids 0→1 transitions
                    acc_mono = acc_mono.add(&bp.mul(&a1.mul(&one.sub(a))));
                }
                bp = bp.mul(&beta);
            }
            c5[r] = ir.mul(&acc_bin);
            c6[r] = ir.mul(&acc_mono);
            c7[r] = ir.mul(&acc_sum.sub(tl));

            // data_active binary + monotonic + sum = data_len.
            let dl = &columns[COL_DATA_LEN][r];
            let mut d_bin = Scalar::zero(curve);
            let mut d_mono = Scalar::zero(curve);
            let mut d_sum = Scalar::zero(curve);
            let mut bp2 = Scalar::one(curve);
            for k in 0..MAX_DATA_BYTES {
                let a = &columns[COL_DATA_ACTIVE_OFFSET + k][r];
                d_bin = d_bin.add(&bp2.mul(&a.mul(&a.sub(&one))));
                d_sum = d_sum.add(a);
                if k < MAX_DATA_BYTES - 1 {
                    let a1 = &columns[COL_DATA_ACTIVE_OFFSET + k + 1][r];
                    d_mono = d_mono.add(&bp2.mul(&a1.mul(&one.sub(a))));
                }
                bp2 = bp2.mul(&beta);
            }
            c8[r] = ir.mul(&d_bin);
            c9[r] = ir.mul(&d_mono);
            c10[r] = ir.mul(&d_sum.sub(dl));

            // topics_inner_len = 33 * topics_len
            let til = &columns[COL_TOPICS_INNER_LEN][r];
            c11[r] = ir.mul(&til.sub(&c33.mul(tl)));

            // topics_prefix_len = 1 + is_topics_long
            let tpl = &columns[COL_TOPICS_PREFIX_LEN][r];
            c12[r] = ir.mul(&tpl.sub(&one.add(itl)));

            // topics_total_len = topics_prefix_len + topics_inner_len
            let ttl = &columns[COL_TOPICS_TOTAL_LEN][r];
            c13[r] = ir.mul(&ttl.sub(&tpl.add(til)));

            // payload_len = 21 + topics_total_len + data_enc_len
            let pl = &columns[COL_PAYLOAD_LEN][r];
            let del = &columns[COL_DATA_ENC_LEN][r];
            let expected_pl = c21.add(ttl).add(del);
            c14[r] = ir.mul(&pl.sub(&expected_pl));

            // outer_prefix_len = 1 + is_outer_long
            let opl = &columns[COL_OUTER_PREFIX_LEN][r];
            c15[r] = ir.mul(&opl.sub(&one.add(iol)));

            // encoded_len = outer_prefix_len + payload_len
            let el = &columns[COL_ENCODED_LEN][r];
            c16[r] = ir.mul(&el.sub(&opl.add(pl)));
        }

        vec![c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let two = Scalar::from_u64(2, curve);
        let three = Scalar::from_u64(3, curve);
        let four = Scalar::from_u64(4, curve);
        let c21 = Scalar::from_u64(ADDRESS_ENC_LEN as u64, curve);
        let c33 = Scalar::from_u64(TOPIC_ENC_LEN as u64, curve);

        let ir = &col_evals[COL_IS_REAL];
        let itl = &col_evals[COL_IS_TOPICS_LONG];
        let iol = &col_evals[COL_IS_OUTER_LONG];
        let tl = &col_evals[COL_TOPICS_LEN];
        let dl = &col_evals[COL_DATA_LEN];
        let del = &col_evals[COL_DATA_ENC_LEN];
        let til = &col_evals[COL_TOPICS_INNER_LEN];
        let tpl = &col_evals[COL_TOPICS_PREFIX_LEN];
        let ttl = &col_evals[COL_TOPICS_TOTAL_LEN];
        let pl = &col_evals[COL_PAYLOAD_LEN];
        let opl = &col_evals[COL_OUTER_PREFIX_LEN];
        let el = &col_evals[COL_ENCODED_LEN];

        let mut bodies = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(ir.mul(&ir.sub(&one)));
        bodies.push(itl.mul(&itl.sub(&one)));
        bodies.push(iol.mul(&iol.sub(&one)));

        let not_itl = one.sub(itl);
        bodies.push(ir.mul(&not_itl.mul(&tl.mul(&tl.sub(&one)))));
        let tl_m2 = tl.sub(&two);
        let tl_m3 = tl.sub(&three);
        let tl_m4 = tl.sub(&four);
        bodies.push(ir.mul(&itl.mul(&tl_m2.mul(&tl_m3).mul(&tl_m4))));

        let mut acc_bin = Scalar::zero(curve);
        let mut acc_mono = Scalar::zero(curve);
        let mut acc_sum = Scalar::zero(curve);
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_TOPICS {
            let a = &col_evals[COL_TOPIC_ACTIVE_OFFSET + k];
            acc_bin = acc_bin.add(&bp.mul(&a.mul(&a.sub(&one))));
            acc_sum = acc_sum.add(a);
            if k < MAX_TOPICS - 1 {
                let a1 = &col_evals[COL_TOPIC_ACTIVE_OFFSET + k + 1];
                acc_mono = acc_mono.add(&bp.mul(&a1.mul(&one.sub(a))));
            }
            bp = bp.mul(alpha);
        }
        bodies.push(ir.mul(&acc_bin));
        bodies.push(ir.mul(&acc_mono));
        bodies.push(ir.mul(&acc_sum.sub(tl)));

        let mut d_bin = Scalar::zero(curve);
        let mut d_mono = Scalar::zero(curve);
        let mut d_sum = Scalar::zero(curve);
        let mut bp2 = Scalar::one(curve);
        for k in 0..MAX_DATA_BYTES {
            let a = &col_evals[COL_DATA_ACTIVE_OFFSET + k];
            d_bin = d_bin.add(&bp2.mul(&a.mul(&a.sub(&one))));
            d_sum = d_sum.add(a);
            if k < MAX_DATA_BYTES - 1 {
                let a1 = &col_evals[COL_DATA_ACTIVE_OFFSET + k + 1];
                d_mono = d_mono.add(&bp2.mul(&a1.mul(&one.sub(a))));
            }
            bp2 = bp2.mul(alpha);
        }
        bodies.push(ir.mul(&d_bin));
        bodies.push(ir.mul(&d_mono));
        bodies.push(ir.mul(&d_sum.sub(dl)));

        bodies.push(ir.mul(&til.sub(&c33.mul(tl))));
        bodies.push(ir.mul(&tpl.sub(&one.add(itl))));
        bodies.push(ir.mul(&ttl.sub(&tpl.add(til))));
        let expected_pl = c21.add(ttl).add(del);
        bodies.push(ir.mul(&pl.sub(&expected_pl)));
        bodies.push(ir.mul(&opl.sub(&one.add(iol))));
        bodies.push(ir.mul(&el.sub(&opl.add(pl))));

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let two_poly = vec![Scalar::from_u64(2, curve)];
        let three_poly = vec![Scalar::from_u64(3, curve)];
        let four_poly = vec![Scalar::from_u64(4, curve)];
        let c21_poly = vec![Scalar::from_u64(ADDRESS_ENC_LEN as u64, curve)];
        let c33_poly = vec![Scalar::from_u64(TOPIC_ENC_LEN as u64, curve)];

        let ir = &col_coeffs[COL_IS_REAL];
        let itl = &col_coeffs[COL_IS_TOPICS_LONG];
        let iol = &col_coeffs[COL_IS_OUTER_LONG];
        let tl = &col_coeffs[COL_TOPICS_LEN];
        let dl = &col_coeffs[COL_DATA_LEN];
        let del = &col_coeffs[COL_DATA_ENC_LEN];
        let til = &col_coeffs[COL_TOPICS_INNER_LEN];
        let tpl = &col_coeffs[COL_TOPICS_PREFIX_LEN];
        let ttl = &col_coeffs[COL_TOPICS_TOTAL_LEN];
        let pl = &col_coeffs[COL_PAYLOAD_LEN];
        let opl = &col_coeffs[COL_OUTER_PREFIX_LEN];
        let el = &col_coeffs[COL_ENCODED_LEN];

        let bin = |x: &Vec<Scalar>| poly_mul(x, &poly_sub(x, &one_poly, curve), curve);

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(bin(ir));
        bodies.push(bin(itl));
        bodies.push(bin(iol));

        // (1 - itl) * tl * (tl - 1), times ir
        let not_itl = poly_sub(&one_poly, itl, curve);
        let tl_m1 = poly_sub(tl, &one_poly, curve);
        let tl_tl_m1 = poly_mul(tl, &tl_m1, curve);
        let body3 = poly_mul(&not_itl, &tl_tl_m1, curve);
        bodies.push(poly_mul(ir, &body3, curve));

        // itl * (tl - 2)(tl - 3)(tl - 4), times ir
        let tl_m2 = poly_sub(tl, &two_poly, curve);
        let tl_m3 = poly_sub(tl, &three_poly, curve);
        let tl_m4 = poly_sub(tl, &four_poly, curve);
        let tm23 = poly_mul(&tl_m2, &tl_m3, curve);
        let tm234 = poly_mul(&tm23, &tl_m4, curve);
        let body4 = poly_mul(itl, &tm234, curve);
        bodies.push(poly_mul(ir, &body4, curve));

        // topic_active binary / monotonic / sum
        let mut acc_bin = vec![Scalar::zero(curve)];
        let mut acc_mono = vec![Scalar::zero(curve)];
        let mut acc_sum = vec![Scalar::zero(curve)];
        let mut bp = Scalar::one(curve);
        for k in 0..MAX_TOPICS {
            let a = &col_coeffs[COL_TOPIC_ACTIVE_OFFSET + k];
            acc_bin = poly_add(&acc_bin, &poly_scalar_mul(&bin(a), &bp), curve);
            acc_sum = poly_add(&acc_sum, a, curve);
            if k < MAX_TOPICS - 1 {
                let a1 = &col_coeffs[COL_TOPIC_ACTIVE_OFFSET + k + 1];
                let term = poly_mul(a1, &poly_sub(&one_poly, a, curve), curve);
                acc_mono = poly_add(&acc_mono, &poly_scalar_mul(&term, &bp), curve);
            }
            bp = bp.mul(alpha);
        }
        bodies.push(poly_mul(ir, &acc_bin, curve));
        bodies.push(poly_mul(ir, &acc_mono, curve));
        bodies.push(poly_mul(ir, &poly_sub(&acc_sum, tl, curve), curve));

        // data_active binary / monotonic / sum
        let mut d_bin = vec![Scalar::zero(curve)];
        let mut d_mono = vec![Scalar::zero(curve)];
        let mut d_sum = vec![Scalar::zero(curve)];
        let mut bp2 = Scalar::one(curve);
        for k in 0..MAX_DATA_BYTES {
            let a = &col_coeffs[COL_DATA_ACTIVE_OFFSET + k];
            d_bin = poly_add(&d_bin, &poly_scalar_mul(&bin(a), &bp2), curve);
            d_sum = poly_add(&d_sum, a, curve);
            if k < MAX_DATA_BYTES - 1 {
                let a1 = &col_coeffs[COL_DATA_ACTIVE_OFFSET + k + 1];
                let term = poly_mul(a1, &poly_sub(&one_poly, a, curve), curve);
                d_mono = poly_add(&d_mono, &poly_scalar_mul(&term, &bp2), curve);
            }
            bp2 = bp2.mul(alpha);
        }
        bodies.push(poly_mul(ir, &d_bin, curve));
        bodies.push(poly_mul(ir, &d_mono, curve));
        bodies.push(poly_mul(ir, &poly_sub(&d_sum, dl, curve), curve));

        // topics_inner_len = 33 * topics_len
        let c33_tl = poly_mul(&c33_poly, tl, curve);
        bodies.push(poly_mul(ir, &poly_sub(til, &c33_tl, curve), curve));

        // topics_prefix_len = 1 + is_topics_long
        let one_plus_itl = poly_add(&one_poly, itl, curve);
        bodies.push(poly_mul(ir, &poly_sub(tpl, &one_plus_itl, curve), curve));

        // topics_total_len = topics_prefix_len + topics_inner_len
        let sum_t = poly_add(tpl, til, curve);
        bodies.push(poly_mul(ir, &poly_sub(ttl, &sum_t, curve), curve));

        // payload_len = 21 + topics_total_len + data_enc_len
        let expected_pl = poly_add(&poly_add(&c21_poly, ttl, curve), del, curve);
        bodies.push(poly_mul(ir, &poly_sub(pl, &expected_pl, curve), curve));

        // outer_prefix_len = 1 + is_outer_long
        let one_plus_iol = poly_add(&one_poly, iol, curve);
        bodies.push(poly_mul(ir, &poly_sub(opl, &one_plus_iol, curve), curve));

        // encoded_len = outer_prefix_len + payload_len
        let sum_e = poly_add(opl, pl, curve);
        bodies.push(poly_mul(ir, &poly_sub(el, &sum_e, curve), curve));

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();

        for k in 0..ADDRESS_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("logs_rlp_address_{}_8bit", k),
                    column_index: COL_ADDRESS_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..(MAX_TOPICS * TOPIC_BYTES) {
            declarations.push((
                LookupDeclaration {
                    label: format!("logs_rlp_topic_byte_{}_8bit", k),
                    column_index: COL_TOPIC_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MAX_DATA_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("logs_rlp_data_byte_{}_8bit", k),
                    column_index: COL_DATA_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..MAX_DATA_ENC_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("logs_rlp_data_enc_{}_8bit", k),
                    column_index: COL_DATA_ENC_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }

        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Binds the 20 address bytes on this AIR to
/// [`crate::fixed_rlp20_air`]'s field byte columns. Combined with that
/// AIR's RLP constraints, this pins the address RLP encoding window
/// `[0x94, addr[0], ..., addr[19]]` inside the log's encoded byte
/// stream.
pub fn make_log_to_address_descriptor(
    logs_layer_index: usize,
    fixed_rlp20_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(ADDRESS_BYTES);
    let mut b_columns = Vec::with_capacity(ADDRESS_BYTES);
    for k in 0..ADDRESS_BYTES {
        a_columns.push(COL_ADDRESS_OFFSET + k);
        b_columns.push(crate::fixed_rlp20_air::COL_FIELD_BYTE_OFFSET + k);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "logs_rlp_to_fixed_rlp20_address_v1".into(),
        a_layer_index: logs_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: fixed_rlp20_layer_index,
        b_columns,
        b_selector_column: Some(crate::fixed_rlp20_air::COL_IS_REAL),
    }
}

/// Binds the 32 raw bytes of topic slot `topic_index ∈ 0..4` on this
/// AIR to [`crate::fixed_rlp_air`]'s field byte columns, gated by the
/// per-topic `TOPIC_ACTIVE[topic_index]` flag on the A-side.
/// Combined with the fixed_rlp32 RLP constraints this pins the topic's
/// RLP encoding `[0xa0, t[0], ..., t[31]]`.
pub fn make_log_to_topic_descriptor(
    topic_index: usize,
    logs_layer_index: usize,
    fixed_rlp32_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    assert!(topic_index < MAX_TOPICS, "topic_index out of range");
    let mut a_columns = Vec::with_capacity(TOPIC_BYTES);
    let mut b_columns = Vec::with_capacity(TOPIC_BYTES);
    for b in 0..TOPIC_BYTES {
        a_columns.push(COL_TOPIC_BYTES_OFFSET + topic_index * TOPIC_BYTES + b);
        b_columns.push(crate::fixed_rlp_air::COL_FIELD_BYTE_OFFSET + b);
    }
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: format!("logs_rlp_to_fixed_rlp32_topic_{}_v1", topic_index),
        a_layer_index: logs_layer_index,
        a_columns,
        a_selector_column: Some(COL_TOPIC_ACTIVE_OFFSET + topic_index),
        b_layer_index: fixed_rlp32_layer_index,
        b_columns,
        b_selector_column: Some(crate::fixed_rlp_air::COL_IS_REAL),
    }
}

/// Binds the data byte window + length + data RLP encoded window on
/// this AIR to [`crate::rlp_var_bytes_air`]'s columns of the same
/// shape (data bytes 0..32, data_len, encoded bytes 0..33,
/// encoded_len). Combined with the var_bytes AIR's constraints, this
/// pins the data RLP encoding to its unique canonical form.
pub fn make_log_to_data_descriptor(
    logs_layer_index: usize,
    var_bytes_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let tuple_len = MAX_DATA_BYTES + 1 + MAX_DATA_ENC_LEN + 1;
    let mut a_columns = Vec::with_capacity(tuple_len);
    let mut b_columns = Vec::with_capacity(tuple_len);
    for k in 0..MAX_DATA_BYTES {
        a_columns.push(COL_DATA_BYTES_OFFSET + k);
        b_columns.push(crate::rlp_var_bytes_air::COL_DATA_OFFSET + k);
    }
    a_columns.push(COL_DATA_LEN);
    b_columns.push(crate::rlp_var_bytes_air::COL_DATA_LEN);
    for k in 0..MAX_DATA_ENC_LEN {
        a_columns.push(COL_DATA_ENC_OFFSET + k);
        b_columns.push(crate::rlp_var_bytes_air::COL_ENCODED_OFFSET + k);
    }
    a_columns.push(COL_DATA_ENC_LEN);
    b_columns.push(crate::rlp_var_bytes_air::COL_ENCODED_LEN);

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "logs_rlp_to_rlp_var_bytes_data_v1".into(),
        a_layer_index: logs_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: var_bytes_layer_index,
        b_columns,
        b_selector_column: Some(crate::rlp_var_bytes_air::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_no_topics_empty_data() -> Log {
        Log {
            address: [0x11u8; 20],
            topics: vec![],
            data: vec![],
        }
    }

    fn log_four_topics_short_data() -> Log {
        Log {
            address: [0x22u8; 20],
            topics: vec![[0xaau8; 32], [0xbbu8; 32], [0xccu8; 32], [0xddu8; 32]],
            data: vec![0xde, 0xad, 0xbe, 0xef],
        }
    }

    fn build_for(logs: &[Log]) -> TracePolynomials {
        let w = LogsRlpWitness::from_logs(logs).unwrap();
        build_trace_polynomials(&w, CurveType::Bls48581)
    }

    #[test]
    fn log_with_zero_topics_encodes_canonically() {
        let l = log_no_topics_empty_data();
        let row = from_log(&l).unwrap();
        // Canonical Log RLP for empty topics + empty data:
        //   outer payload = 21 (addr) + 1 (0xc0 empty topics list) + 1 (0x80 empty data) = 23
        //   outer prefix  = 0xc0 + 23 = 0xd7 (1 byte)
        //   encoded_len   = 1 + 23 = 24
        assert_eq!(row.topics_inner_len, 0);
        assert_eq!(row.topics_prefix_len, 1);
        assert_eq!(row.topics_total_len, 1);
        assert_eq!(row.data_enc.len(), 1);
        assert_eq!(row.payload_len, 23);
        assert_eq!(row.outer_prefix_len, 1);
        assert_eq!(row.encoded_len, 24);
        assert_eq!(row.encoded_bytes, l.rlp_encode());
        assert_eq!(row.encoded_bytes.len(), 24);
        assert_eq!(row.encoded_bytes[0], 0xd7);
    }

    #[test]
    fn log_with_four_topics_encodes_canonically() {
        let l = log_four_topics_short_data();
        let row = from_log(&l).unwrap();
        // topics_inner = 4 * 33 = 132, prefix = 2 (long form 0xf8 0x84)
        // topics_total = 134
        // data = 4 bytes (none < 0x80 leading? 0xde ≥ 0x80 → multi-byte) → enc_len = 5
        // payload = 21 + 134 + 5 = 160; outer prefix = 2 (0xf8 0xa0)
        // encoded_len = 162
        assert_eq!(row.topics_inner_len, 132);
        assert_eq!(row.topics_prefix_len, 2);
        assert_eq!(row.topics_total_len, 134);
        assert_eq!(row.data_enc.len(), 5);
        assert_eq!(row.payload_len, 160);
        assert_eq!(row.outer_prefix_len, 2);
        assert_eq!(row.encoded_len, 162);
        assert_eq!(row.encoded_bytes, l.rlp_encode());
        assert_eq!(row.encoded_bytes[0], 0xf8);
        assert_eq!(row.encoded_bytes[1], 160);
    }

    #[test]
    fn constraints_zero_on_honest_witnesses() {
        let logs = vec![
            log_no_topics_empty_data(),
            log_four_topics_short_data(),
            // 1 topic, no data
            Log {
                address: [0x33u8; 20],
                topics: vec![[0xee; 32]],
                data: vec![],
            },
            // 2 topics, 32-byte data (max data)
            Log {
                address: [0x44u8; 20],
                topics: vec![[0x55; 32], [0x66; 32]],
                data: vec![0xff; 32],
            },
        ];
        let trace = build_for(&logs);
        let cs = LogsRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(res.len(), NUM_ROW_CONSTRAINTS);
        for (i, body) in res.iter().enumerate() {
            for (r, val) in body.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} (label {}) at row {} = {:?}",
                    i,
                    cs.constraint_labels()[i],
                    r,
                    val,
                );
            }
        }
    }

    #[test]
    fn tampered_encoded_len_detected() {
        let trace = build_for(&[log_four_topics_short_data()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let cur = cols[COL_ENCODED_LEN][0].to_u64();
        cols[COL_ENCODED_LEN][0] = Scalar::from_u64(cur + 1, CurveType::Bls48581);
        let cs = LogsRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 16 = encoded_len_formula
        assert!(!res[16][0].is_zero(), "encoded_len_formula must fire");
    }

    #[test]
    fn tampered_topics_inner_len_detected() {
        let trace = build_for(&[log_four_topics_short_data()]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let cur = cols[COL_TOPICS_INNER_LEN][0].to_u64();
        cols[COL_TOPICS_INNER_LEN][0] =
            Scalar::from_u64(cur + 33, CurveType::Bls48581);
        let cs = LogsRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 11 = topics_inner_len_eq_33_times_topics_len
        assert!(!res[11][0].is_zero(), "topics_inner_len_formula must fire");
    }

    #[test]
    fn tampered_topic_active_monotonic_detected() {
        let l = Log {
            address: [0x77u8; 20],
            topics: vec![[0x88; 32], [0x99; 32]],
            data: vec![],
        };
        let trace = build_for(&[l]);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Original active = [1, 1, 0, 0]. Force a 0→1 transition at
        // k=2→3 by setting active[3] = 1, so the monotonic constraint
        // (active[3] * (1 - active[2])) fires.
        cols[COL_TOPIC_ACTIVE_OFFSET + 3][0] =
            Scalar::from_u64(1, CurveType::Bls48581);
        let cs = LogsRlpConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let res = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // constraint 6 = topic_active_monotonic_rlc
        assert!(!res[6][0].is_zero(), "topic_active_monotonic must fire");
    }

    #[test]
    fn descriptors_well_formed() {
        let d_addr = make_log_to_address_descriptor(0, 1);
        assert_eq!(d_addr.label, "logs_rlp_to_fixed_rlp20_address_v1");
        assert_eq!(d_addr.a_columns.len(), ADDRESS_BYTES);
        assert_eq!(d_addr.b_columns.len(), ADDRESS_BYTES);
        assert_eq!(d_addr.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_addr.b_selector_column,
            Some(crate::fixed_rlp20_air::COL_IS_REAL),
        );

        for k in 0..MAX_TOPICS {
            let d_t = make_log_to_topic_descriptor(k, 0, 2);
            assert_eq!(d_t.label, format!("logs_rlp_to_fixed_rlp32_topic_{}_v1", k));
            assert_eq!(d_t.a_columns.len(), TOPIC_BYTES);
            assert_eq!(d_t.b_columns.len(), TOPIC_BYTES);
            assert_eq!(
                d_t.a_selector_column,
                Some(COL_TOPIC_ACTIVE_OFFSET + k),
            );
        }

        let d_data = make_log_to_data_descriptor(0, 3);
        assert_eq!(d_data.label, "logs_rlp_to_rlp_var_bytes_data_v1");
        // 32 data + 1 data_len + 33 enc + 1 enc_len = 67
        let expected_tuple_len = MAX_DATA_BYTES + 1 + MAX_DATA_ENC_LEN + 1;
        assert_eq!(d_data.a_columns.len(), expected_tuple_len);
        assert_eq!(d_data.b_columns.len(), expected_tuple_len);
    }

    #[test]
    fn rejects_oversized_data() {
        let l = Log {
            address: [0u8; 20],
            topics: vec![],
            data: vec![0u8; 33],
        };
        let err = from_log(&l).unwrap_err();
        assert!(err.contains("data > 32"), "got: {}", err);
    }

    #[test]
    fn rejects_too_many_topics() {
        let l = Log {
            address: [0u8; 20],
            topics: vec![[0u8; 32]; 5],
            data: vec![],
        };
        let err = from_log(&l).unwrap_err();
        assert!(err.contains("at most 4 topics"), "got: {}", err);
    }

    #[test]
    fn num_columns_pinned() {
        assert_eq!(NUM_COLUMNS, 261);
        assert_eq!(MAX_PAYLOAD_LEN, 188);
        assert_eq!(MAX_ENCODED_LEN, 190);
        assert_eq!(ADDRESS_ENC_LEN, 21);
        assert_eq!(TOPIC_ENC_LEN, 33);
    }

    #[test]
    fn encoded_len_correct_for_various_logs() {
        // Comprehensive table of (log, expected encoded_len) — pin
        // sizes against the canonical encoder.
        let logs = vec![
            // 0 topics, 0 data → 24
            (log_no_topics_empty_data(), 24),
            // 4 topics, 4-byte multi-data → 162
            (log_four_topics_short_data(), 162),
            // 1 topic, 32-byte data with leading-byte 0xff (multi) →
            //   addr(21) + topics_prefix(1) + topic(33) + data_enc(33) = 88; outer prefix = 2; total = 90
            (
                Log {
                    address: [0xab; 20],
                    topics: vec![[0xcd; 32]],
                    data: vec![0xff; 32],
                },
                90,
            ),
            // 2 topics, empty data:
            //   addr(21) + topics_prefix(2) + topics_inner(66) + data(1) = 90; outer prefix = 2; total = 92
            (
                Log {
                    address: [0x01; 20],
                    topics: vec![[0x02; 32], [0x03; 32]],
                    data: vec![],
                },
                92,
            ),
        ];
        for (l, expected_len) in logs {
            let row = from_log(&l).unwrap();
            assert_eq!(
                row.encoded_len, expected_len,
                "log encoded_len mismatch for {:?}",
                l,
            );
            assert_eq!(row.encoded_bytes.len(), expected_len);
            assert_eq!(row.encoded_bytes, l.rlp_encode());
        }
    }
}
