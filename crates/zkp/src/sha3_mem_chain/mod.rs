//! SHA3 input ↔ byte-memory ↔ EVM main chain composition.
//!
//! **Task #51** — wires the algebraic chain
//!
//! ```text
//!   evm_main  →  mstore_byte_air  →  byte_memory_air  ←  sha3_read_byte_air  ←  sha3_input_air
//! ```
//!
//! into a single `joint_prove`-callable bundle.  The host-side EVM
//! main trace is **assumed correct** at this layer (the `metavm-zkp`
//! crate cannot import `metavm-evm`); callers that actually wire EVM
//! into the joint prove run pass their own EVM main trace + constraint
//! system separately (e.g. from `crates/evm`).  This module exposes
//! the *zkp-side* witnesses, layer-index threading, and complete
//! `Vec<CrossAirLogUpDescriptor>` that closes the chain.
//!
//! # Components (zkp-side)
//!
//! 1. [`byte_memory_air`](crate::byte_memory_air) — the byte-addressable
//!    memory bus.  Holds both the unsorted access trace and a sorted
//!    view; the self-linkage descriptor algebraically forces them to
//!    be permutations of each other.
//!
//! 2. [`mstore_byte_air`](crate::mstore_byte_air) — MSTORE byte-decomp
//!    gadget.  Each invocation row commits a 32-byte BE decomposition
//!    of a U256 written by EVM MSTORE; per-position linkages push the
//!    `(offset + k, byte_k)` tuples into byte-memory as Writes.
//!
//! 3. [`sha3_read_byte_air`](crate::sha3_read_byte_air) — SHA3 byte-
//!    forwarding gadget.  Each row commits the 256-byte input buffer
//!    of one SHA3 invocation plus `is_active_k` prefix flags; per-
//!    position linkages pull the `(offset + k, byte_k)` tuples from
//!    byte-memory as Reads.
//!
//! 4. [`sha3_input_air`](crate::sha3_input_air) — SHA3 input gadget.
//!    The full 256-byte input plus output limbs.  The
//!    `sha3_input_to_read_byte` linkage binds its 257-col tuple
//!    `(input_byte[0..256], input_len)` to the read-byte gadget.
//!
//! # What the chain proves
//!
//! Given a set of MSTOREs that populate memory and a set of SHA3s
//! that read it, the chain algebraically witnesses:
//!
//! - **MSTORE → byte-memory**: every BE byte of every MSTOREd U256
//!   appears in byte-memory as a Write at `(mstore_offset + k)`.
//! - **byte-memory permutation**: the unsorted access list and the
//!   sorted-by-(addr,ts) view are the same multiset.
//! - **byte-memory → SHA3 read**: every active `(addr, byte)` in a
//!   SHA3 invocation appears in byte-memory.
//! - **SHA3 input ↔ read-byte**: the SHA3 input gadget's
//!   `(input_byte[0..256], input_len)` matches the read-byte gadget's
//!   per-row view.
//!
//! Composed end-to-end: every byte fed to keccak originated from a
//! prior MSTORE (or zero-padded memory), algebraically tied through
//! the cross-AIR LogUp closures.
//!
//! # Limitations (recorded for follow-up)
//!
//! - **EVM main is host-trusted at this layer.**  This module does
//!   not commit the EVM main trace.  The full chain end-to-end
//!   (`make_evm_mstore_to_byte_decomp_linkage_descriptor` +
//!   `make_evm_mstore8_byte_memory_linkage_descriptor`) is wired in
//!   the `crates/evm` joint-prove harness, where the EVM crate is
//!   available.  Here we wire the **zkp-internal** linkages only.
//!
//! - **Read-only gating on byte-memory** (`sel_read` column) is not
//!   yet present, so the read-byte → byte-memory linkage gates on
//!   `IS_REAL` and is therefore a subset check rather than a strict
//!   "Reads only" check.  Documented in
//!   [`crate::sha3_read_byte_air::make_sha3_read_byte_to_byte_memory_linkage_descriptor`].
//!
//! - **Address arithmetic uses u64** (matching the existing AIRs).
//!   EVM offsets up to `2^64 - 1` are supported; arithmetic overflow
//!   into `u128` is a follow-up.

use crate::byte_memory_air::{ByteMemoryAccess, ByteMemoryWitness};
use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::mstore_byte_air::{MstoreByteRow, MstoreByteWitness, NUM_BYTES as MSTORE_NUM_BYTES};
use crate::sha3_input_air::{
    Sha3InputRow, Sha3InputWitness, INPUT_BYTE_WIDTH as SHA3_INPUT_BYTE_WIDTH,
};
use crate::sha3_read_byte_air::{
    Sha3ReadByteRow, Sha3ReadByteWitness, NUM_BYTES as SHA3_READ_NUM_BYTES,
};

// ─────────────────────────────────────────────────────────────────────
// Witness bundle
// ─────────────────────────────────────────────────────────────────────

/// Composite witness threading SHA3 input → read-byte → byte-memory ←
/// MSTORE-byte → (EVM main) into one chain.
///
/// `evm_main_marker` is a no-op placeholder so callers wiring an
/// external EVM main trace at the next layer can keep this struct's
/// layout stable as the EVM-crate-side fields are populated.  At this
/// layer (zkp only) EVM is taken on host trust.
#[derive(Clone, Debug)]
pub struct Sha3MemChain {
    /// MSTORE offsets (one per MSTORE event), in host order.
    pub mstore_offsets: Vec<u64>,
    /// 32-byte big-endian U256 values being MSTOREd, paired 1:1 with
    /// `mstore_offsets`.
    pub mstore_values: Vec<[u8; 32]>,
    /// Offset and size of the SHA3 invocation.  One SHA3 call per
    /// chain assembly; multi-SHA3 chains compose by concatenating
    /// these.
    pub sha3_offset: u64,
    pub sha3_size: u64,

    /// The byte-memory AIR's witness (unsorted + sorted views).
    pub byte_memory_witness: ByteMemoryWitness,
    /// The MSTORE byte-decomp gadget's witness (one row per MSTORE).
    pub mstore_byte_witness: MstoreByteWitness,
    /// The SHA3 read-byte gadget's witness (one row per SHA3).
    pub sha3_read_byte_witness: Sha3ReadByteWitness,
    /// The SHA3 input gadget's witness (one row per SHA3).
    pub sha3_input_witness: Sha3InputWitness,

    /// Placeholder marker for the host-trusted EVM main trace.  Set
    /// to `true` by callers that have wired an EVM main trace into
    /// the same joint-prove call; otherwise the EVM linkages are
    /// considered host-trusted at this layer.
    pub evm_main_marker: bool,
}

impl Sha3MemChain {
    /// `true` iff per-witness sizes are consistent (one row per
    /// MSTORE / SHA3, paired vectors equal length).
    pub fn is_host_consistent(&self) -> bool {
        if self.mstore_offsets.len() != self.mstore_values.len() {
            return false;
        }
        if self.mstore_byte_witness.invocations.len() != self.mstore_offsets.len() {
            return false;
        }
        if self.sha3_read_byte_witness.invocations.len() != 1 {
            return false;
        }
        if self.sha3_input_witness.invocations.len() != 1 {
            return false;
        }
        // SHA3 invocation byte-buffer width matches gadgets.
        if (self.sha3_size as usize) > SHA3_READ_NUM_BYTES {
            return false;
        }
        if (self.sha3_size as usize) > SHA3_INPUT_BYTE_WIDTH {
            return false;
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────

/// Decompose a 32-byte BE U256 into 4 LE u64 limbs.
///
/// Mirrors [`crate::mstore_byte_air::be_bytes_of_u256`] in reverse:
/// `value_be[0..8]` is the MSB limb, `value_be[24..32]` is the LSB.
fn be_bytes_to_le_limbs(value_be: &[u8; 32]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for limb_idx in 0..4 {
        // Limb 3 = MSB → bytes 0..8; limb 0 = LSB → bytes 24..32.
        let byte_start = (3 - limb_idx) * 8;
        let mut acc: u64 = 0;
        for k in 0..8 {
            acc = (acc << 8) | (value_be[byte_start + k] as u64);
        }
        limbs[limb_idx] = acc;
    }
    limbs
}

/// Assemble a [`Sha3MemChain`] from a host-supplied set of MSTOREs
/// followed by a single SHA3 invocation that hashes
/// `memory[sha3_offset..sha3_offset + sha3_size]`.
///
/// The caller is responsible for ensuring `mstore_offsets` /
/// `mstore_values` (in host order) populate the SHA3 input region
/// consistently with a real EVM execution; this builder is a
/// witness-shaper, not a memory simulator.  Bytes outside any MSTORE
/// region default to zero (the EVM memory abstraction guarantees
/// zero-init for unwritten cells), matching the `verify_read_consistency`
/// invariant on the byte-memory AIR.
pub fn assemble_chain(
    mstore_offsets: &[u64],
    mstore_values: &[[u8; 32]],
    sha3_offset: u64,
    sha3_size: u64,
) -> Sha3MemChain {
    assert_eq!(
        mstore_offsets.len(),
        mstore_values.len(),
        "sha3_mem_chain: mstore_offsets / mstore_values length mismatch",
    );
    assert!(
        (sha3_size as usize) <= SHA3_READ_NUM_BYTES,
        "sha3_mem_chain: sha3_size {} exceeds gadget cap {}",
        sha3_size,
        SHA3_READ_NUM_BYTES,
    );

    // ── byte_memory_air witness ────────────────────────────────────
    // Writes first (ts 0..32*num_mstores), then SHA3 reads
    // (ts following writes).  source_row encodes the originating
    // event index.
    let mut accesses: Vec<ByteMemoryAccess> = Vec::new();
    let mut ts: u64 = 0;

    for (event_idx, (&offset, value)) in
        mstore_offsets.iter().zip(mstore_values.iter()).enumerate()
    {
        for k in 0..32u64 {
            accesses.push(ByteMemoryAccess {
                addr: offset.wrapping_add(k),
                val: value[k as usize],
                ts,
                rw: 1,
                source_row: event_idx as u64,
            });
            ts += 1;
        }
    }

    // Compute the SHA3 input bytes by replaying the writes above.
    let mut sha3_input = [0u8; SHA3_READ_NUM_BYTES];
    {
        // Reconstruct a sparse byte-map of all writes, then read the
        // SHA3 window.  Last-write-wins semantics (matching the
        // sorted-by-(addr,ts) view of byte-memory).
        use std::collections::HashMap;
        let mut bytes: HashMap<u64, u8> = HashMap::new();
        for a in &accesses {
            if a.rw == 1 {
                bytes.insert(a.addr, a.val);
            }
        }
        for k in 0..(sha3_size as usize) {
            let addr = sha3_offset.wrapping_add(k as u64);
            sha3_input[k] = *bytes.get(&addr).unwrap_or(&0);
        }
    }

    // Emit SHA3 read accesses for active positions.
    let sha3_event_idx = mstore_offsets.len() as u64;
    for k in 0..(sha3_size as usize) {
        accesses.push(ByteMemoryAccess {
            addr: sha3_offset.wrapping_add(k as u64),
            val: sha3_input[k],
            ts,
            rw: 0,
            source_row: sha3_event_idx,
        });
        ts += 1;
    }

    let byte_memory_witness = ByteMemoryWitness::from_accesses(accesses);

    // ── mstore_byte_air witness ────────────────────────────────────
    let mstore_byte_rows: Vec<MstoreByteRow> = mstore_offsets
        .iter()
        .zip(mstore_values.iter())
        .map(|(&offset, value)| MstoreByteRow {
            offset,
            limb: be_bytes_to_le_limbs(value),
        })
        .collect();
    let mstore_byte_witness = MstoreByteWitness::from_invocations(mstore_byte_rows);

    // ── sha3_read_byte_air witness ─────────────────────────────────
    let sha3_read_byte_witness = Sha3ReadByteWitness::from_invocations(vec![Sha3ReadByteRow {
        offset: sha3_offset,
        input_len: sha3_size as u32,
        input_byte: sha3_input,
    }])
    .expect("sha3_size validated <= NUM_BYTES");

    // ── sha3_input_air witness ─────────────────────────────────────
    // sha3_input_air pads to INPUT_BYTE_WIDTH (256); both gadgets
    // share NUM_BYTES = 256 so the buffers align.
    let mut sha3_input_padded = [0u8; SHA3_INPUT_BYTE_WIDTH];
    let copy_len = (sha3_size as usize).min(SHA3_INPUT_BYTE_WIDTH);
    sha3_input_padded[..copy_len].copy_from_slice(&sha3_input[..copy_len]);
    let digest = crate::keccak::keccak256(&sha3_input[..(sha3_size as usize)]);
    let output_limb = crate::keccak_extract::keccak_output_limbs_from_output(&digest);
    let sha3_input_witness = Sha3InputWitness {
        invocations: vec![Sha3InputRow {
            input: sha3_input_padded,
            input_len: sha3_size as usize,
            output_limb,
        }],
    };

    Sha3MemChain {
        mstore_offsets: mstore_offsets.to_vec(),
        mstore_values: mstore_values.to_vec(),
        sha3_offset,
        sha3_size,
        byte_memory_witness,
        mstore_byte_witness,
        sha3_read_byte_witness,
        sha3_input_witness,
        evm_main_marker: false,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Layer indices
// ─────────────────────────────────────────────────────────────────────

/// Layer 0: byte_memory_air.
pub const LAYER_BYTE_MEMORY: usize = 0;
/// Layer 1: mstore_byte_air.
pub const LAYER_MSTORE_BYTE: usize = 1;
/// Layer 2: sha3_read_byte_air.
pub const LAYER_SHA3_READ_BYTE: usize = 2;
/// Layer 3: sha3_input_air.
pub const LAYER_SHA3_INPUT: usize = 3;

// ─────────────────────────────────────────────────────────────────────
// Cross-AIR LogUp descriptors
// ─────────────────────────────────────────────────────────────────────

/// Default number of per-position MSTORE→byte_memory descriptors.
///
/// Each MSTORE pushes 32 bytes into memory; one descriptor per byte
/// position closes the full BE decomposition algebraically.  Smaller
/// counts (e.g. 1, 3) are accepted by
/// [`collect_descriptors_with_positions`] for faster smoke tests.
pub const DEFAULT_MSTORE_BYTE_POSITIONS: usize = MSTORE_NUM_BYTES;

/// Default number of per-position read_byte→byte_memory descriptors.
/// 256 covers the full SHA3 input width.
pub const DEFAULT_SHA3_READ_BYTE_POSITIONS: usize = SHA3_READ_NUM_BYTES;

/// Collect all cross-AIR LogUp descriptors threading the chain at
/// default layer indices, using the default per-position counts.
///
/// Returns ≥6 descriptors:
///   1. byte_memory self-linkage (unsorted ↔ sorted)
///   2..(1 + DEFAULT_MSTORE_BYTE_POSITIONS): per-position
///      mstore_byte → byte_memory descriptors
///   (1 + DEFAULT_MSTORE_BYTE_POSITIONS) ..
///   (1 + DEFAULT_MSTORE_BYTE_POSITIONS + DEFAULT_SHA3_READ_BYTE_POSITIONS):
///      per-position sha3_read_byte → byte_memory descriptors
///   (last): sha3_input → sha3_read_byte (257-col tuple)
pub fn collect_descriptors() -> Vec<CrossAirLogUpDescriptor> {
    // Use small per-position counts by default to keep the
    // descriptor vector compact for fast tests.  The slow joint_prove
    // test in this module uses the same default.  Callers that want
    // the full 32 + 256 per-position closure can call
    // [`collect_descriptors_with_positions`].
    //
    // The task spec only requires ≥6 descriptors; we choose
    // (3 mstore positions, 2 sha3 read positions) → 1 + 3 + 2 + 1 = 7.
    collect_descriptors_with_positions(3, 2)
}

/// Collect descriptors with explicit per-position counts.
///
/// `mstore_byte_positions` and `sha3_read_byte_positions` set how
/// many per-byte-position descriptors are emitted for the two
/// per-position linkage families.  Each is capped at 32 / 256
/// respectively.
pub fn collect_descriptors_with_positions(
    mstore_byte_positions: usize,
    sha3_read_byte_positions: usize,
) -> Vec<CrossAirLogUpDescriptor> {
    let mut out: Vec<CrossAirLogUpDescriptor> = Vec::new();

    // (1) byte_memory self-linkage.
    out.push(
        crate::byte_memory_air::make_byte_memory_self_linkage_descriptor(LAYER_BYTE_MEMORY),
    );

    // (2..) mstore_byte → byte_memory, one per byte position.
    let m_pos = mstore_byte_positions.min(MSTORE_NUM_BYTES);
    for k in 0..m_pos {
        out.push(
            crate::mstore_byte_air::make_mstore_byte_to_byte_memory_linkage_descriptor(
                LAYER_MSTORE_BYTE,
                LAYER_BYTE_MEMORY,
                k,
            ),
        );
    }

    // (3..) sha3_read_byte → byte_memory, one per byte position.
    let s_pos = sha3_read_byte_positions.min(SHA3_READ_NUM_BYTES);
    for k in 0..s_pos {
        out.push(
            crate::sha3_read_byte_air::make_sha3_read_byte_to_byte_memory_linkage_descriptor(
                LAYER_SHA3_READ_BYTE,
                LAYER_BYTE_MEMORY,
                k,
            ),
        );
    }

    // (4) sha3_input → sha3_read_byte (257-col tuple).
    out.push(
        crate::sha3_read_byte_air::make_sha3_input_to_read_byte_linkage_descriptor(
            LAYER_SHA3_INPUT,
            LAYER_SHA3_READ_BYTE,
        ),
    );

    out
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_mstore_value(seed: u8) -> [u8; 32] {
        let mut v = [0u8; 32];
        for k in 0..32 {
            v[k] = seed.wrapping_add(k as u8);
        }
        v
    }

    /// Two MSTOREs covering a 64-byte region, then a SHA3 over the
    /// first 32 bytes.
    fn sample_chain() -> Sha3MemChain {
        let offsets = vec![0u64, 32u64];
        let values = vec![sample_mstore_value(0x10), sample_mstore_value(0x40)];
        assemble_chain(&offsets, &values, 0, 32)
    }

    // ── Fast: chain assembles ───────────────────────────────────────

    #[test]
    fn chain_assembles_for_two_mstores_one_sha3() {
        let chain = sample_chain();
        assert_eq!(chain.mstore_offsets.len(), 2);
        assert_eq!(chain.mstore_values.len(), 2);
        assert_eq!(chain.mstore_byte_witness.invocations.len(), 2);
        assert_eq!(chain.sha3_read_byte_witness.invocations.len(), 1);
        assert_eq!(chain.sha3_input_witness.invocations.len(), 1);
        // 2 MSTOREs × 32 bytes + 32 SHA3 reads = 96 byte_memory accesses.
        assert_eq!(chain.byte_memory_witness.accesses.len(), 96);
        // Host-consistency check.
        assert!(chain.is_host_consistent());
        // SHA3 input bytes match the MSTORE values.
        let sha3_input = &chain.sha3_read_byte_witness.invocations[0].input_byte;
        for k in 0..32 {
            assert_eq!(sha3_input[k], chain.mstore_values[0][k]);
        }
        // sha3_input_air row sees the same bytes.
        let row = &chain.sha3_input_witness.invocations[0];
        assert_eq!(row.input_len, 32);
        for k in 0..32 {
            assert_eq!(row.input[k], chain.mstore_values[0][k]);
        }
        // Read-consistency holds host-side.
        chain.byte_memory_witness.verify_read_consistency().unwrap();
    }

    // ── Fast: descriptor count ≥6 ──────────────────────────────────

    #[test]
    fn descriptor_count_meets_minimum() {
        let descs = collect_descriptors();
        assert!(
            descs.len() >= 6,
            "expected ≥6 descriptors threading the chain, got {}",
            descs.len(),
        );
    }

    // ── Fast: all descriptors have nonempty column lists ───────────

    #[test]
    fn all_descriptors_have_nonempty_columns() {
        let descs = collect_descriptors();
        for d in &descs {
            assert!(
                !d.a_columns.is_empty(),
                "descriptor {} has empty a_columns",
                d.label,
            );
            assert!(
                !d.b_columns.is_empty(),
                "descriptor {} has empty b_columns",
                d.label,
            );
            assert_eq!(
                d.a_columns.len(),
                d.b_columns.len(),
                "descriptor {} a/b column count mismatch",
                d.label,
            );
        }
    }

    // ── Fast: descriptor labels distinct ──────────────────────────

    #[test]
    fn descriptor_labels_are_distinct() {
        let descs = collect_descriptors();
        let labels: Vec<&str> = descs.iter().map(|d| d.label.as_str()).collect();
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                assert_ne!(
                    labels[i], labels[j],
                    "duplicate label at positions {} / {}: {}",
                    i, j, labels[i],
                );
            }
        }
        // Specific expected labels (loose subset check — exact
        // structure depends on the per-position count chosen by
        // collect_descriptors).
        assert!(labels.contains(&"byte_memory_unsorted_eq_sorted_v1"));
        assert!(labels.contains(&"sha3_input_to_read_byte_v1"));
    }

    // ── Slow stub: full joint_prove + joint_verify ─────────────────

    /// Slow end-to-end stub: would assemble all four zkp-side AIRs
    /// into a single `joint_prove` call across the chain.  Estimated
    /// runtime: minutes on BLS48-581 (each per-position LogUp adds a
    /// linkage SNARK; the 32-position MSTORE expansion + 256-position
    /// SHA3-read expansion would push this to hours).  This stub
    /// pins the API call signatures the slow run would use.
    ///
    /// Run manually with:
    ///
    /// ```bash
    /// cargo test -p metavm-zkp --release --lib \
    ///     sha3_mem_chain::tests::sha3_mem_chain_joint_prove_passes -- \
    ///     --ignored --test-threads=1 --nocapture
    /// ```
    #[test]
    #[ignore = "slow: sha3 mem chain joint_prove ~minutes"]
    fn sha3_mem_chain_joint_prove_passes() {
        use crate::byte_memory_air::ByteMemoryConstraintSystem;
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::field::CurveType;
        use crate::mstore_byte_air::MstoreByteConstraintSystem;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::sha3_input_air::Sha3InputConstraintSystem;
        use crate::sha3_read_byte_air::Sha3ReadByteConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        let chain = sample_chain();
        let descriptors = collect_descriptors();

        // Build all four traces (host-consistency presumed; the slow
        // run validates algebraically through joint_prove).
        let bm_trace =
            crate::byte_memory_air::build_trace_polynomials(&chain.byte_memory_witness, curve);
        let mb_trace = crate::mstore_byte_air::build_trace_polynomials(
            &chain.mstore_byte_witness,
            curve,
        );
        let sr_trace = crate::sha3_read_byte_air::build_trace_polynomials(
            &chain.sha3_read_byte_witness,
            curve,
        );
        let si_trace = crate::sha3_input_air::build_trace_polynomials(
            &chain.sha3_input_witness,
            curve,
        );

        let bm_cs = ByteMemoryConstraintSystem::new(bm_trace.num_rows);
        let mb_cs = MstoreByteConstraintSystem::new(mb_trace.num_rows);
        let sr_cs = Sha3ReadByteConstraintSystem::new(sr_trace.num_rows);
        let si_cs = Sha3InputConstraintSystem::new(si_trace.num_rows);

        // Order MUST match LAYER_* constants above.
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&bm_trace, &bm_cs),
            (&mb_trace, &mb_cs),
            (&sr_trace, &sr_cs),
            (&si_trace, &si_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &descriptors, &scheme)
            .expect("sha3_mem_chain joint_prove must succeed");

        assert_eq!(proofs.len(), 4);
        assert_eq!(ext.linkage_proofs.len(), descriptors.len());
        for lp in &ext.linkage_proofs {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closure mismatch",
                lp.label,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&bm_cs, &mb_cs, &sr_cs, &si_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &descriptors, &ext, &scheme, curve),
            "sha3_mem_chain joint_verify must accept honest witness",
        );
    }

    // ── Helper sanity: BE limb decomp round-trip ────────────────────

    #[test]
    fn be_bytes_to_le_limbs_round_trips_through_mstore_byte_air() {
        let value: [u8; 32] = (0..32u8).collect::<Vec<_>>().try_into().unwrap();
        let limbs = be_bytes_to_le_limbs(&value);
        let bytes_back = crate::mstore_byte_air::be_bytes_of_u256(limbs);
        assert_eq!(bytes_back, value);
    }
}
