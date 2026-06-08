//! SHA3 input ↔ EVM byte-memory binding oracle (Phase A1b-mem step 0).
//!
//! Phase A1b's joint_prove validated the chain:
//!   EVM main SHA3 output ↔ SHA3 input gadget ↔ KeccakExtract,
//! algebraically committing the SHA3 *preimage* as audit-able witness
//! data. But the gadget's `INPUT_BYTE` columns are still oracle relative
//! to the EVM memory model — a malicious prover could declare any
//! preimage; only the hash-output binding back to the EVM stack is
//! constrained.
//!
//! A1b-mem closes that gap by binding the gadget's `INPUT_BYTE[i]` to
//! `evm_byte_memory[offset + i]` at the row where the SHA3 opcode
//! executes. The full algebraic binding will need a byte-level memory
//! permutation argument (the existing one is U256-aligned). This module
//! is **step 0**: a host-side oracle that walks the EVM trace, replays
//! all MSTORE/MSTORE8 writes as a byte-addressable memory image, and
//! emits — for each SHA3 row — the slice of bytes the SHA3 input gadget
//! must hash. The oracle becomes the spec that an algebraic step-1
//! constraint must enforce.
//!
//! Usage in tests:
//!   let snapshots = sha3_input_bytes_from_trace(&evm_cols);
//!   let gadget_w = Sha3InputWitness::from_inputs(&snapshots).unwrap();
//!   // → identical to constructing the gadget directly from the
//!   //   bytecode-level inputs.

use std::collections::BTreeMap;

use crate::trace::EvmTraceColumns;

/// Kind of byte-level memory access. The future algebraic permutation
/// will use this as a 0/1 selector (1 = write) to gate the read-
/// consistency constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteRw {
    Read,
    Write,
}

/// One byte-level memory access: `(addr, val, ts, rw)`. This is the
/// tuple the future byte-memory permutation argument will commit to.
/// `ts` is a host-assigned global counter that monotonically increases
/// across the trace; it disambiguates multiple accesses at the same
/// `addr` and pins ordering when the permutation re-sorts by `(addr,
/// ts)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteAccess {
    pub addr: u64,
    pub val: u8,
    pub ts: u64,
    pub rw: ByteRw,
    /// Which EVM trace row generated this access. Used by step-1
    /// cross-AIR LogUp to bind accesses back to the source row.
    pub source_row: usize,
}

/// Big-endian byte decomposition of a U256 value held as 4 little-endian
/// u64 limbs (limb 0 = least significant 64 bits).
fn u256_be_bytes(limbs: [u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    // limb 3 holds bytes 0..8 (most significant), limb 0 holds bytes 24..32.
    for (i, limb) in limbs.iter().rev().enumerate() {
        let be = limb.to_be_bytes();
        out[i * 8..(i + 1) * 8].copy_from_slice(&be);
    }
    out
}

/// Result of replaying byte-level memory across an EVM trace.
///
/// `snapshots[k]` is the slice of bytes the k-th SHA3 row read from
/// memory at the moment that row executed: `mem[offset..offset+len]`
/// where `offset = trace.input0[0][row]`, `len = trace.input1[0][row]`.
/// Bytes are returned in memory order (i.e. the order they would be fed
/// into keccak256).
///
/// Only the low limb of `offset`/`len` is honored. Larger values are an
/// error condition (real EVM enforces a byte-level gas cost that limits
/// these in practice; tests should stay well under 2^32).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sha3MemorySnapshots {
    /// One Vec<u8> per SHA3 row in trace order. Length equals the count
    /// of rows where `sel_keccak == 1`.
    pub snapshots: Vec<Vec<u8>>,
    /// Indexes (into trace rows) of the SHA3 rows the snapshots came
    /// from, in 1:1 correspondence with `snapshots`.
    pub sha3_row_indexes: Vec<usize>,
}

/// Walk the EVM trace and replay a byte-level memory image. For each
/// MSTORE row, write 32 BE bytes of the value at `mem_offset`. For each
/// MSTORE8 row, write the low byte of `mem_value[0]` at `mem_offset`.
/// At each SHA3 row, snapshot `mem[offset..offset+len]` (filling unwritten
/// bytes with 0, matching EVM's zero-initialized memory semantics).
///
/// Returns one snapshot per SHA3 row in trace order.
pub fn replay_byte_memory_at_sha3(trace: &EvmTraceColumns) -> Sha3MemorySnapshots {
    let n_rows = trace.step.len();
    let mut byte_mem: BTreeMap<u64, u8> = BTreeMap::new();
    let mut snapshots: Vec<Vec<u8>> = Vec::new();
    let mut sha3_rows: Vec<usize> = Vec::new();

    for i in 0..n_rows {
        // Apply byte-level writes BEFORE checking for SHA3 read at this
        // row. In the real EVM step model, SHA3 *reads* memory written by
        // prior steps; the inspector pushes one row per opcode and
        // mem_offset/mem_value are populated in step_end (after the
        // store completes). A SHA3 row at position i has access to all
        // writes from rows 0..=i-1.
        if trace.sel_mstore[i] == 1 {
            let offset = trace.mem_offset[i];
            let val_limbs = [
                trace.mem_value[0][i],
                trace.mem_value[1][i],
                trace.mem_value[2][i],
                trace.mem_value[3][i],
            ];
            let bytes = u256_be_bytes(val_limbs);
            for (k, b) in bytes.iter().enumerate() {
                byte_mem.insert(offset.wrapping_add(k as u64), *b);
            }
        } else if trace.sel_mstore8[i] == 1 {
            let offset = trace.mem_offset[i];
            // MSTORE8 writes the low byte of the U256 value popped from
            // the stack (input1 in the inspector convention).
            let byte = (trace.mem_value[0][i] & 0xff) as u8;
            byte_mem.insert(offset, byte);
        }

        if trace.sel_keccak[i] == 1 {
            // SHA3: input0 = offset, input1 = len (both U256, low limb only).
            let offset = trace.input0[0][i];
            let len = trace.input1[0][i] as usize;
            let mut snap = Vec::with_capacity(len);
            for k in 0..len {
                let addr = offset.wrapping_add(k as u64);
                snap.push(*byte_mem.get(&addr).unwrap_or(&0u8));
            }
            snapshots.push(snap);
            sha3_rows.push(i);
        }
    }

    Sha3MemorySnapshots {
        snapshots,
        sha3_row_indexes: sha3_rows,
    }
}

/// Build the byte-level memory access trace for an EVM run.
///
/// For each EVM trace row, emit the byte-granularity accesses that row
/// performed:
/// - MSTORE @ offset, value V → 32 Write accesses `(offset+k, byte_k(V))` for k=0..32
/// - MSTORE8 @ offset, value V → 1 Write access `(offset, V & 0xff)`
/// - SHA3 @ offset, len L → L Read accesses `(offset+k, mem[offset+k])` for k=0..L
///
/// Each access gets a monotonically-increasing `ts` (timestamp) so that
/// sorting by `(addr, ts)` recovers a per-address chronological access
/// list — the basis for the read-consistency permutation argument the
/// step-1 algebraic binding will use.
///
/// For SHA3 reads, the `val` is taken from the current byte-memory
/// image at that row, matching what the SHA3 input gadget's
/// `INPUT_BYTE[k]` is *required* to equal.
pub fn build_byte_access_trace(trace: &EvmTraceColumns) -> Vec<ByteAccess> {
    let n_rows = trace.step.len();
    let mut byte_mem: BTreeMap<u64, u8> = BTreeMap::new();
    let mut accesses: Vec<ByteAccess> = Vec::new();
    let mut ts: u64 = 0;

    for i in 0..n_rows {
        if trace.sel_mstore[i] == 1 {
            let offset = trace.mem_offset[i];
            let val_limbs = [
                trace.mem_value[0][i],
                trace.mem_value[1][i],
                trace.mem_value[2][i],
                trace.mem_value[3][i],
            ];
            let bytes = u256_be_bytes(val_limbs);
            for (k, b) in bytes.iter().enumerate() {
                let addr = offset.wrapping_add(k as u64);
                byte_mem.insert(addr, *b);
                accesses.push(ByteAccess {
                    addr,
                    val: *b,
                    ts,
                    rw: ByteRw::Write,
                    source_row: i,
                });
                ts += 1;
            }
        } else if trace.sel_mstore8[i] == 1 {
            let offset = trace.mem_offset[i];
            let byte = (trace.mem_value[0][i] & 0xff) as u8;
            byte_mem.insert(offset, byte);
            accesses.push(ByteAccess {
                addr: offset,
                val: byte,
                ts,
                rw: ByteRw::Write,
                source_row: i,
            });
            ts += 1;
        }

        if trace.sel_keccak[i] == 1 {
            let offset = trace.input0[0][i];
            let len = trace.input1[0][i] as usize;
            for k in 0..len {
                let addr = offset.wrapping_add(k as u64);
                let val = *byte_mem.get(&addr).unwrap_or(&0u8);
                accesses.push(ByteAccess {
                    addr,
                    val,
                    ts,
                    rw: ByteRw::Read,
                    source_row: i,
                });
                ts += 1;
            }
        }
    }

    accesses
}

/// Convert host-side EVM byte accesses into the zkp crate's
/// [`metavm_zkp::byte_memory_air::ByteMemoryAccess`] tuples — the
/// witness type the byte-memory AIR consumes. Encodes
/// `ByteRw::Write → 1`, `ByteRw::Read → 0`.
pub fn to_byte_memory_witness(
    accesses: &[ByteAccess],
) -> metavm_zkp::byte_memory_air::ByteMemoryWitness {
    use metavm_zkp::byte_memory_air::{ByteMemoryAccess, ByteMemoryWitness};
    let mapped: Vec<ByteMemoryAccess> = accesses
        .iter()
        .map(|a| ByteMemoryAccess {
            addr: a.addr,
            val: a.val,
            ts: a.ts,
            rw: match a.rw {
                ByteRw::Read => 0,
                ByteRw::Write => 1,
            },
            source_row: a.source_row as u64,
        })
        .collect();
    ByteMemoryWitness::from_accesses(mapped)
}

/// Verify that a byte access trace satisfies *read consistency*: when
/// sorted by `(addr, ts)`, every read at address `a` returns the value
/// of the most recent write to `a` (or 0 if no prior write).
///
/// Returns `Ok(())` on success or `Err(diagnostic)` describing the
/// first violation. This is the algebraic invariant the step-1
/// permutation argument will need to enforce; here we check it
/// host-side as the spec for that constraint.
pub fn verify_read_consistency(accesses: &[ByteAccess]) -> Result<(), String> {
    let mut sorted: Vec<ByteAccess> = accesses.to_vec();
    sorted.sort_by_key(|a| (a.addr, a.ts));

    let mut prev: Option<ByteAccess> = None;
    for cur in &sorted {
        match prev {
            None => {
                // First access at this addr (lowest ts).
                if cur.rw == ByteRw::Read && cur.val != 0 {
                    return Err(format!(
                        "first access at addr {} is a Read with val={} \
                         (expected 0 for uninitialized byte memory)",
                        cur.addr, cur.val
                    ));
                }
            }
            Some(p) => {
                if p.addr == cur.addr {
                    match cur.rw {
                        ByteRw::Read => {
                            if cur.val != p.val {
                                return Err(format!(
                                    "read at addr={} ts={} returned val={} \
                                     but most recent prior access (ts={}, \
                                     rw={:?}) had val={}",
                                    cur.addr, cur.ts, cur.val, p.ts, p.rw, p.val
                                ));
                            }
                        }
                        ByteRw::Write => {
                            // Write at same addr — fine, mutates state.
                        }
                    }
                } else {
                    // New address — must start with either a Write or a
                    // Read of 0 (default memory).
                    if cur.rw == ByteRw::Read && cur.val != 0 {
                        return Err(format!(
                            "first access at addr {} is a Read with val={} \
                             (expected 0 for uninitialized byte memory)",
                            cur.addr, cur.val
                        ));
                    }
                }
            }
        }
        prev = Some(*cur);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::execute_bytecode;

    /// The A1b joint_prove test bytecode: MSTORE8 ×4 of [0xab, 0xcd,
    /// 0xef, 0x12] at offsets 0..=3, then SHA3(offset=0, len=4).
    fn a1b_bytecode() -> Vec<u8> {
        vec![
            // PUSH1 0xab; PUSH1 0x00; MSTORE8
            0x60, 0xab,  0x60, 0x00,  0x53,
            // PUSH1 0xcd; PUSH1 0x01; MSTORE8
            0x60, 0xcd,  0x60, 0x01,  0x53,
            // PUSH1 0xef; PUSH1 0x02; MSTORE8
            0x60, 0xef,  0x60, 0x02,  0x53,
            // PUSH1 0x12; PUSH1 0x03; MSTORE8
            0x60, 0x12,  0x60, 0x03,  0x53,
            // PUSH1 0x04; PUSH1 0x00; SHA3; STOP
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ]
    }

    #[test]
    fn replay_matches_a1b_input_bytes() {
        let evm_cols = execute_bytecode(&a1b_bytecode(), &[]).unwrap();
        let snaps = replay_byte_memory_at_sha3(&evm_cols);
        assert_eq!(snaps.snapshots.len(), 1, "exactly one SHA3 row");
        assert_eq!(snaps.snapshots[0], vec![0xab, 0xcd, 0xef, 0x12]);
    }

    #[test]
    fn snapshot_matches_gadget_witness_input() {
        use metavm_zkp::sha3_input_air::Sha3InputWitness;

        let evm_cols = execute_bytecode(&a1b_bytecode(), &[]).unwrap();
        let snaps = replay_byte_memory_at_sha3(&evm_cols);
        // The gadget's input field is byte-padded to INPUT_BYTE_WIDTH
        // (256) with trailing zeros; the live `input_len` records the
        // actual byte count.
        let gadget = Sha3InputWitness::from_inputs(&snaps.snapshots).unwrap();
        assert_eq!(gadget.invocations.len(), 1);
        assert_eq!(gadget.invocations[0].input_len, 4);
        assert_eq!(&gadget.invocations[0].input[..4], &[0xab, 0xcd, 0xef, 0x12]);
        // Trailing bytes are zero-padded.
        assert!(gadget.invocations[0].input[4..].iter().all(|&b| b == 0));
    }

    #[test]
    fn mstore_word_full_32_bytes_then_sha3() {
        // PUSH32 0x0102...20 ; PUSH1 0x00 ; MSTORE
        // PUSH1 0x20 ; PUSH1 0x00 ; SHA3 ; STOP
        let mut bc = vec![0x7F]; // PUSH32
        for k in 0..32u8 {
            bc.push(k + 1); // value is 0x01..0x20
        }
        bc.extend_from_slice(&[
            0x60, 0x00,  // PUSH1 0
            0x52,        // MSTORE
            0x60, 0x20,  // PUSH1 32
            0x60, 0x00,  // PUSH1 0
            0x20,        // SHA3
            0x00,        // STOP
        ]);
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();
        let snaps = replay_byte_memory_at_sha3(&evm_cols);
        assert_eq!(snaps.snapshots.len(), 1);
        let want: Vec<u8> = (1..=32u8).collect();
        assert_eq!(snaps.snapshots[0], want);
    }

    #[test]
    fn unwritten_memory_reads_as_zero() {
        // SHA3(offset=0, len=4) with NO prior MSTORE — reads default 0s.
        let bc = vec![
            0x60, 0x04,  // PUSH1 4
            0x60, 0x00,  // PUSH1 0
            0x20,        // SHA3
            0x00,        // STOP
        ];
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();
        let snaps = replay_byte_memory_at_sha3(&evm_cols);
        assert_eq!(snaps.snapshots.len(), 1);
        assert_eq!(snaps.snapshots[0], vec![0u8; 4]);
    }

    #[test]
    fn empty_input_sha3_returns_empty_snapshot() {
        // SHA3(offset=0, len=0) — keccak256 of empty bytes.
        let bc = vec![
            0x60, 0x00,  // PUSH1 0 (len)
            0x60, 0x00,  // PUSH1 0 (offset)
            0x20,        // SHA3
            0x00,        // STOP
        ];
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();
        let snaps = replay_byte_memory_at_sha3(&evm_cols);
        assert_eq!(snaps.snapshots.len(), 1);
        assert_eq!(snaps.snapshots[0], Vec::<u8>::new());
    }

    #[test]
    fn mstore8_overwrite_takes_latest_value() {
        // Write 0xaa at offset 0, overwrite with 0xbb, SHA3(offset=0, len=1).
        let bc = vec![
            0x60, 0xaa,  0x60, 0x00,  0x53,
            0x60, 0xbb,  0x60, 0x00,  0x53,
            0x60, 0x01,  0x60, 0x00,  0x20,  0x00,
        ];
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();
        let snaps = replay_byte_memory_at_sha3(&evm_cols);
        assert_eq!(snaps.snapshots.len(), 1);
        assert_eq!(snaps.snapshots[0], vec![0xbb]);
    }

    #[test]
    fn snapshot_matches_gadget_witness_for_full_word() {
        use metavm_zkp::sha3_input_air::Sha3InputWitness;
        let mut bc = vec![0x7F];
        for k in 0..32u8 {
            bc.push(k + 0x10);
        }
        bc.extend_from_slice(&[
            0x60, 0x00,  0x52,
            0x60, 0x20,  0x60, 0x00,  0x20,  0x00,
        ]);
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();
        let snaps = replay_byte_memory_at_sha3(&evm_cols);
        let gadget = Sha3InputWitness::from_inputs(&snaps.snapshots).unwrap();
        assert_eq!(gadget.invocations[0].input_len, 32);
        let expected: Vec<u8> = (0..32u8).map(|k| k + 0x10).collect();
        assert_eq!(&gadget.invocations[0].input[..32], &expected[..]);
    }

    #[test]
    fn byte_access_trace_a1b_layout() {
        // A1b: 4 MSTORE8 writes at offsets 0..=3, 1 SHA3(offset=0, len=4).
        // Expected access trace: 4 Writes followed by 4 Reads.
        let evm_cols = execute_bytecode(&a1b_bytecode(), &[]).unwrap();
        let trace = build_byte_access_trace(&evm_cols);
        assert_eq!(trace.len(), 8);
        let writes: Vec<&ByteAccess> = trace.iter().filter(|a| a.rw == ByteRw::Write).collect();
        let reads: Vec<&ByteAccess> = trace.iter().filter(|a| a.rw == ByteRw::Read).collect();
        assert_eq!(writes.len(), 4);
        assert_eq!(reads.len(), 4);
        // Writes in trace order: 0xab @ 0, 0xcd @ 1, 0xef @ 2, 0x12 @ 3.
        assert_eq!((writes[0].addr, writes[0].val), (0, 0xab));
        assert_eq!((writes[1].addr, writes[1].val), (1, 0xcd));
        assert_eq!((writes[2].addr, writes[2].val), (2, 0xef));
        assert_eq!((writes[3].addr, writes[3].val), (3, 0x12));
        // Reads from SHA3 row in offset order, returning the bytes that
        // were just written.
        assert_eq!((reads[0].addr, reads[0].val), (0, 0xab));
        assert_eq!((reads[1].addr, reads[1].val), (1, 0xcd));
        assert_eq!((reads[2].addr, reads[2].val), (2, 0xef));
        assert_eq!((reads[3].addr, reads[3].val), (3, 0x12));
        // Timestamps are monotonically increasing.
        for w in trace.windows(2) {
            assert!(w[1].ts > w[0].ts, "ts must be monotonic: {} >! {}", w[0].ts, w[1].ts);
        }
    }

    #[test]
    fn byte_access_trace_mstore_word_expands_to_32() {
        let mut bc = vec![0x7F];
        for k in 0..32u8 {
            bc.push(k + 1);
        }
        bc.extend_from_slice(&[
            0x60, 0x00,  0x52,
            0x60, 0x20,  0x60, 0x00,  0x20,  0x00,
        ]);
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();
        let trace = build_byte_access_trace(&evm_cols);
        let writes: Vec<&ByteAccess> = trace.iter().filter(|a| a.rw == ByteRw::Write).collect();
        let reads: Vec<&ByteAccess> = trace.iter().filter(|a| a.rw == ByteRw::Read).collect();
        assert_eq!(writes.len(), 32, "MSTORE expands to 32 byte writes");
        assert_eq!(reads.len(), 32, "SHA3 of len=32 emits 32 byte reads");
        for k in 0..32 {
            assert_eq!(writes[k].addr, k as u64);
            assert_eq!(writes[k].val, (k as u8) + 1);
        }
    }

    #[test]
    fn read_consistency_holds_for_honest_a1b_trace() {
        let evm_cols = execute_bytecode(&a1b_bytecode(), &[]).unwrap();
        let trace = build_byte_access_trace(&evm_cols);
        verify_read_consistency(&trace).expect("honest A1b trace must satisfy read consistency");
    }

    #[test]
    fn read_consistency_holds_for_default_zero_read() {
        let bc = vec![
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ];
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();
        let trace = build_byte_access_trace(&evm_cols);
        verify_read_consistency(&trace).expect("default-zero reads satisfy consistency");
    }

    #[test]
    fn read_consistency_detects_tampered_read_value() {
        let evm_cols = execute_bytecode(&a1b_bytecode(), &[]).unwrap();
        let mut trace = build_byte_access_trace(&evm_cols);
        // Tamper: change the first read's value from 0xab to 0xff.
        let first_read = trace.iter_mut().find(|a| a.rw == ByteRw::Read).unwrap();
        first_read.val = 0xff;
        let err = verify_read_consistency(&trace).unwrap_err();
        assert!(err.contains("returned val=255"), "got: {}", err);
    }

    #[test]
    fn read_consistency_detects_uninit_nonzero_read() {
        // Hand-build: a single Read access at addr=42 with val=0x77.
        let trace = vec![ByteAccess {
            addr: 42,
            val: 0x77,
            ts: 0,
            rw: ByteRw::Read,
            source_row: 0,
        }];
        let err = verify_read_consistency(&trace).unwrap_err();
        assert!(err.contains("expected 0"), "got: {}", err);
    }

    #[test]
    fn to_byte_memory_witness_a1b_round_trip() {
        let evm_cols = execute_bytecode(&a1b_bytecode(), &[]).unwrap();
        let trace = build_byte_access_trace(&evm_cols);
        let witness = to_byte_memory_witness(&trace);
        // The zkp-side witness sees the same access count.
        assert_eq!(witness.accesses.len(), 8);
        // ZKP-side read consistency oracle agrees.
        witness.verify_read_consistency().unwrap();
        // First write maps to (addr=0, val=0xab, ts=0, rw=1).
        let w0 = witness.accesses[0];
        assert_eq!((w0.addr, w0.val, w0.ts, w0.rw), (0, 0xab, 0, 1));
        // First read (4th access in trace order) maps to rw=0.
        let r0 = witness.accesses[4];
        assert_eq!((r0.addr, r0.val, r0.rw), (0, 0xab, 0));
    }

    #[test]
    fn to_byte_memory_witness_full_word_round_trip() {
        let mut bc = vec![0x7F];
        for k in 0..32u8 {
            bc.push(k + 1);
        }
        bc.extend_from_slice(&[
            0x60, 0x00,  0x52,
            0x60, 0x20,  0x60, 0x00,  0x20,  0x00,
        ]);
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();
        let trace = build_byte_access_trace(&evm_cols);
        let witness = to_byte_memory_witness(&trace);
        assert_eq!(witness.accesses.len(), 64); // 32 writes + 32 reads
        witness.verify_read_consistency().unwrap();
    }

    #[test]
    fn u256_be_bytes_round_trip() {
        // U256 value 0x0102030405060708_090a0b0c0d0e0f10_1112131415161718_191a1b1c1d1e1f20
        // → BE bytes should be 0x01..0x20 in order.
        let limbs = [
            0x191a1b1c1d1e1f20u64, // limb 0 (low)
            0x1112131415161718u64,
            0x090a0b0c0d0e0f10u64,
            0x0102030405060708u64, // limb 3 (high)
        ];
        let be = u256_be_bytes(limbs);
        let want: Vec<u8> = (1..=32u8).collect();
        assert_eq!(&be[..], &want[..]);
    }
}
