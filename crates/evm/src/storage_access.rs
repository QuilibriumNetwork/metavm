//! Storage access witness + host-side MPT oracle — Phase A2 step 0.
//!
//! Captures per-row witness data for EVM `SLOAD`/`SSTORE` opcodes that
//! the future Storage gadget AIR (step 1+) will algebraically commit
//! to. The witness pairs each storage access with the MPT inclusion
//! proof anchoring its `(slot, value)` to the contract's
//! `storage_root`, which itself will eventually be bound to
//! `state_root` via a parallel account-state MPT chain (step 3).
//!
//! Step 0 (this module) is host-side only: build witness, verify
//! against existing MPT inclusion verifier ([`metavm_zkp::mpt::verify_mpt_inclusion`]),
//! and extract from a real EVM trace. No algebraic constraints yet.
//!
//! Ethereum storage MPT value convention: storage trie values are
//! `rlp(value_be_minus_leading_zeros)`. The "empty slot" → "not in
//! trie" (proof of non-inclusion is needed for a 0-valued SLOAD on
//! an empty slot, which step 1+ will address). For now this oracle
//! treats `value == 0` as "absent from trie" and skips verification
//! of those accesses (caveat for follow-up).

use revm::primitives::U256;

use crate::storage_slot::{slot_to_be_bytes, storage_slot_trie_key};
use crate::trace::EvmTraceColumns;

/// One SLOAD or SSTORE event from EVM execution, paired with the MPT
/// inclusion proof that anchors `(slot, value)` to `storage_root`.
#[derive(Debug, Clone)]
pub struct StorageAccess {
    /// 20-byte contract address whose storage trie is being accessed.
    pub contract_address: [u8; 20],
    /// Storage slot key (the U256 popped from the stack).
    pub slot: U256,
    /// Value read (SLOAD) or written (SSTORE).
    pub value: U256,
    /// `true` for SSTORE, `false` for SLOAD.
    pub is_write: bool,
    /// 32-byte root hash of the contract's storage trie at the time of
    /// the access. For SLOAD this is the pre-state storage root; for
    /// SSTORE this is also the pre-state root (the value being
    /// overwritten lives at this slot in the pre-state).
    pub storage_root: [u8; 32],
    /// Sequence of RLP-encoded MPT nodes forming a proof from
    /// `storage_root` to the leaf containing `(trie_key, rlp(value))`,
    /// where `trie_key = keccak256(slot.to_be_bytes::<32>())`.
    pub mpt_proof: Vec<Vec<u8>>,
}

/// Encode `value` as the bytes Ethereum stores in the storage trie:
/// RLP of the big-endian U256 with leading zeros stripped. The zero
/// value is encoded as the empty string `0x80` per RLP convention.
pub fn storage_value_to_rlp(value: U256) -> Vec<u8> {
    let be = slot_to_be_bytes(value);
    metavm_zkp::rlp::rlp_encode_u256(&be)
}

/// Verify a single storage access against its declared storage root.
///
/// On `Ok(())` the MPT inclusion proof verifies against the root and
/// the value matches what's stored at the slot.
///
/// Returns `Err(diagnostic)` for any failure mode. **Caveat (step 0)**:
/// a `value == 0` SLOAD against a slot that's not in the trie is
/// represented in real Ethereum by absence-of-inclusion (a non-
/// inclusion proof, which has a different shape). This oracle does
/// not yet handle that case — it only validates *positive* inclusion.
pub fn verify_storage_access_oracle(access: &StorageAccess) -> Result<(), String> {
    let trie_key = storage_slot_trie_key(access.slot);
    if access.value == U256::ZERO {
        // Step 0 skip — non-inclusion proofs are step 1+.
        return Ok(());
    }
    let value_bytes = storage_value_to_rlp(access.value);
    if !metavm_zkp::mpt::verify_mpt_inclusion(
        access.storage_root,
        &trie_key,
        &value_bytes,
        &access.mpt_proof,
    ) {
        return Err(format!(
            "MPT inclusion proof failed for slot={:?} value={:?} \
             storage_root={:?} (trie_key={:?})",
            access.slot, access.value, access.storage_root, trie_key,
        ));
    }
    Ok(())
}

/// Walk an EVM trace and extract one [`StorageAccess`] entry per
/// SLOAD/SSTORE row, **with empty MPT proof + zero storage root**.
/// The caller is responsible for filling `storage_root` and
/// `mpt_proof` from an external state DB or test fixture.
///
/// This is the bridge from real EVM execution to the storage witness;
/// the algebraic step-1 gadget AIR will then commit to the
/// `storage_root` and the proof.
pub fn extract_storage_accesses_from_trace(trace: &EvmTraceColumns) -> Vec<StorageAccess> {
    let n = trace.step.len();
    let mut out = Vec::new();
    for r in 0..n {
        let is_load = trace.opcode[r] == 0x54;
        let is_store = trace.opcode[r] == 0x55;
        if !is_load && !is_store {
            continue;
        }
        let slot_limbs = [
            trace.input0[0][r],
            trace.input0[1][r],
            trace.input0[2][r],
            trace.input0[3][r],
        ];
        let value_limbs = if is_load {
            [
                trace.output0[0][r],
                trace.output0[1][r],
                trace.output0[2][r],
                trace.output0[3][r],
            ]
        } else {
            // SSTORE: input1 holds the value being stored (pre-popped
            // from stack[1]).
            [
                trace.input1[0][r],
                trace.input1[1][r],
                trace.input1[2][r],
                trace.input1[3][r],
            ]
        };
        // Contract address = the executing frame's callee (already in
        // the frame_callee columns of the EVM trace).
        let mut contract_address = [0u8; 20];
        let callee_be = u256_be_from_limbs([
            trace.frame_callee[0][r],
            trace.frame_callee[1][r],
            trace.frame_callee[2][r],
            trace.frame_callee[3][r],
        ]);
        // Address occupies low 20 bytes of the 32-byte BE rep.
        contract_address.copy_from_slice(&callee_be[12..32]);

        out.push(StorageAccess {
            contract_address,
            slot: U256::from_limbs(slot_limbs),
            value: U256::from_limbs(value_limbs),
            is_write: is_store,
            storage_root: [0u8; 32], // caller fills
            mpt_proof: Vec::new(),    // caller fills
        });
    }
    out
}

/// Convert a [`StorageAccess`] into the zkp crate's
/// [`metavm_zkp::storage_access_air::StorageAccessRow`] shape — the
/// witness type the storage gadget AIR consumes. Throws away the
/// `mpt_proof` field (kept on the [`StorageAccess`] for the host-side
/// MPT oracle but not consumed by the gadget AIR itself, which only
/// commits to the access tuple).
pub fn to_storage_gadget_row(
    access: &StorageAccess,
) -> metavm_zkp::storage_access_air::StorageAccessRow {
    metavm_zkp::storage_access_air::StorageAccessRow {
        address: access.contract_address,
        slot: access.slot.as_limbs().to_owned(),
        value: access.value.as_limbs().to_owned(),
        storage_root: access.storage_root,
        is_write: access.is_write,
    }
}

/// Build the full storage gadget witness from an EVM trace + caller-
/// supplied storage root and MPT proof per access. The proof is needed
/// per access (different slots/values produce different proofs).
///
/// This is the bridge from "real EVM execution" to "gadget AIR
/// witness ready for joint_prove". Step 1b's EVM↔gadget cross-AIR
/// LogUp linkage uses these gadget rows as the B side; the A side
/// would be the EVM main trace gated by `sel_sload`/`sel_sstore`
/// (those selectors are not yet wired — see `storage_access_air.md`).
pub fn build_storage_gadget_witness_from_trace(
    trace: &EvmTraceColumns,
    storage_roots: &[[u8; 32]],
) -> Result<metavm_zkp::storage_access_air::StorageAccessWitness, String> {
    let mut accesses = extract_storage_accesses_from_trace(trace);
    if accesses.len() != storage_roots.len() {
        return Err(format!(
            "build_storage_gadget_witness: trace has {} storage accesses but \
             caller supplied {} storage_roots",
            accesses.len(),
            storage_roots.len(),
        ));
    }
    for (a, r) in accesses.iter_mut().zip(storage_roots.iter()) {
        a.storage_root = *r;
    }
    let rows: Vec<_> = accesses.iter().map(to_storage_gadget_row).collect();
    Ok(metavm_zkp::storage_access_air::StorageAccessWitness::from_rows(rows))
}

/// Build the MPT inclusion AIR witness rows for a single storage
/// access. Bridges this crate's `StorageAccess` (slot, value, proof)
/// to the zkp crate's MPT inclusion AIR witness format.
///
/// Each row in the returned `Vec` corresponds to one node along the
/// MPT inclusion path from `storage_root` (row 0) to the leaf
/// containing the slot's value (last row).
///
/// This is the host-side step-1d-bridge: it produces the witness
/// data that a future cross-AIR LogUp linkage (storage gadget ↔ MPT
/// inclusion AIR) will algebraically commit to. The current
/// implementation uses the existing `mpt_air::inclusion_witness`
/// helper which already validates the proof against the key as it
/// walks the path.
pub fn build_mpt_inclusion_witness_for_access(
    access: &StorageAccess,
) -> Vec<metavm_zkp::mpt_air::InclusionRow> {
    let trie_key = storage_slot_trie_key(access.slot);
    metavm_zkp::mpt_air::inclusion_witness(&trie_key, &access.mpt_proof)
}

/// Verify the END-TO-END storage proof chain at the witness level:
///   1. The trie key = keccak256(slot_be).
///   2. The MPT proof verifies against `storage_root` for `(trie_key,
///      rlp(value))`.
///   3. The MPT inclusion AIR witness row 0's `parent_hash` equals
///      `storage_root`.
///
/// This is the host-side oracle for the full step-1 chain. The
/// future joint_prove with cross-AIR LogUps will commit to each
/// piece of this oracle algebraically; until then, this validates
/// that the witness construction is correct end-to-end.
pub fn verify_storage_proof_chain_oracle(
    access: &StorageAccess,
) -> Result<(), String> {
    // Step 1: trie key derivation.
    let trie_key = storage_slot_trie_key(access.slot);
    let recomputed = metavm_zkp::keccak::keccak256(&slot_to_be_bytes(access.slot));
    if recomputed != trie_key {
        return Err("trie key derivation mismatch (impossible if storage_slot_trie_key is honest)".into());
    }
    // Step 2: MPT inclusion oracle.
    verify_storage_access_oracle(access)?;
    // Step 3: MPT inclusion witness row 0 binding.
    if access.value != revm::primitives::U256::ZERO {
        let rows = build_mpt_inclusion_witness_for_access(access);
        if rows.is_empty() {
            return Err("MPT inclusion witness builder returned no rows".into());
        }
        if rows[0].parent_hash != access.storage_root {
            return Err(format!(
                "MPT inclusion witness row 0 parent_hash {:?} != storage_root {:?}",
                rows[0].parent_hash, access.storage_root,
            ));
        }
    }
    Ok(())
}

fn u256_be_from_limbs(limbs: [u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for k in 0..4 {
        let start = (3 - k) * 8;
        out[start..start + 8].copy_from_slice(&limbs[k].to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use metavm_zkp::mpt::single_leaf_trie;

    /// Single-leaf storage trie: one slot in the trie, the oracle
    /// accepts an honest SLOAD against it.
    #[test]
    fn oracle_accepts_honest_single_leaf_sload() {
        let slot = U256::from(0u64);
        let value = U256::from(42u64);
        let trie_key = storage_slot_trie_key(slot);
        let value_rlp = storage_value_to_rlp(value);
        let (root, proof) = single_leaf_trie(&trie_key, &value_rlp);
        let access = StorageAccess {
            contract_address: [0xab; 20],
            slot,
            value,
            is_write: false,
            storage_root: root,
            mpt_proof: proof,
        };
        verify_storage_access_oracle(&access).unwrap();
    }

    /// Tampered value: oracle rejects.
    #[test]
    fn oracle_rejects_tampered_value() {
        let slot = U256::from(0u64);
        let value = U256::from(42u64);
        let trie_key = storage_slot_trie_key(slot);
        let value_rlp = storage_value_to_rlp(value);
        let (root, proof) = single_leaf_trie(&trie_key, &value_rlp);
        // Tamper: claim a different value with the same proof.
        let access = StorageAccess {
            contract_address: [0xab; 20],
            slot,
            value: U256::from(99u64),
            is_write: false,
            storage_root: root,
            mpt_proof: proof,
        };
        let err = verify_storage_access_oracle(&access).unwrap_err();
        assert!(err.contains("MPT inclusion proof failed"), "got: {}", err);
    }

    /// Tampered storage root: oracle rejects.
    #[test]
    fn oracle_rejects_tampered_storage_root() {
        let slot = U256::from(0u64);
        let value = U256::from(42u64);
        let trie_key = storage_slot_trie_key(slot);
        let value_rlp = storage_value_to_rlp(value);
        let (_root, proof) = single_leaf_trie(&trie_key, &value_rlp);
        let access = StorageAccess {
            contract_address: [0xab; 20],
            slot,
            value,
            is_write: false,
            storage_root: [0xff; 32], // wrong root
            mpt_proof: proof,
        };
        let err = verify_storage_access_oracle(&access).unwrap_err();
        assert!(err.contains("MPT inclusion proof failed"), "got: {}", err);
    }

    /// Zero-value SLOAD: step-0 oracle accepts (non-inclusion is
    /// deferred). Documents the caveat as a test.
    #[test]
    fn oracle_skips_zero_value_sload_caveat() {
        let access = StorageAccess {
            contract_address: [0; 20],
            slot: U256::from(7u64),
            value: U256::ZERO,
            is_write: false,
            storage_root: [0xff; 32], // arbitrary; not checked when value==0
            mpt_proof: Vec::new(),
        };
        verify_storage_access_oracle(&access).unwrap();
    }

    /// Storage value RLP encoding: zero is 0x80, small values are
    /// single bytes, large values are length-prefixed.
    #[test]
    fn storage_value_rlp_encoding() {
        // Zero → 0x80 (empty string in RLP).
        assert_eq!(storage_value_to_rlp(U256::ZERO), vec![0x80]);
        // 0x42 → 0x42 (single byte < 0x80).
        assert_eq!(storage_value_to_rlp(U256::from(0x42u64)), vec![0x42]);
        // 0x80 → length prefix 0x81 + value 0x80.
        assert_eq!(storage_value_to_rlp(U256::from(0x80u64)), vec![0x81, 0x80]);
        // 0xFF → length prefix 0x81 + value 0xff.
        assert_eq!(storage_value_to_rlp(U256::from(0xffu64)), vec![0x81, 0xff]);
    }

    /// Trie-key known-value oracle: slot 0 → known keccak256 hash.
    #[test]
    fn storage_slot_trie_key_for_slot_zero() {
        let key = storage_slot_trie_key(U256::ZERO);
        // keccak256(0x00 * 32) = 0x290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563
        let want = [
            0x29, 0x0d, 0xec, 0xd9, 0x54, 0x8b, 0x62, 0xa8,
            0xd6, 0x03, 0x45, 0xa9, 0x88, 0x38, 0x6f, 0xc8,
            0x4b, 0xa6, 0xbc, 0x95, 0x48, 0x40, 0x08, 0xf6,
            0x36, 0x2f, 0x93, 0x16, 0x0e, 0xf3, 0xe5, 0x63,
        ];
        assert_eq!(key, want);
    }

    /// MPT inclusion witness builder: build a single-leaf storage
    /// trie, then verify the storage gadget + MPT chain agree on
    /// (root, key, value) at the witness level.
    #[test]
    fn mpt_inclusion_witness_for_single_leaf_trie() {
        let slot = U256::from(0u64);
        let value = U256::from(42u64);
        let trie_key = storage_slot_trie_key(slot);
        let value_rlp = storage_value_to_rlp(value);
        let (root, proof) = metavm_zkp::mpt::single_leaf_trie(&trie_key, &value_rlp);
        let access = StorageAccess {
            contract_address: [0xab; 20],
            slot,
            value,
            is_write: false,
            storage_root: root,
            mpt_proof: proof.clone(),
        };
        let rows = build_mpt_inclusion_witness_for_access(&access);
        assert!(!rows.is_empty(), "MPT witness should have rows");
        // Row 0's parent_hash must equal the storage root.
        assert_eq!(rows[0].parent_hash, root);
        // Last row should be terminal (a leaf).
        let last = rows.last().unwrap();
        assert_eq!(last.is_terminal, 1, "last row should be leaf/terminal");
    }

    /// Full chain oracle: trie key + MPT inclusion + witness row 0
    /// parent_hash binding all agree.
    #[test]
    fn verify_storage_proof_chain_oracle_honest() {
        let slot = U256::from(7u64);
        let value = U256::from(0x1234u64);
        let trie_key = storage_slot_trie_key(slot);
        let value_rlp = storage_value_to_rlp(value);
        let (root, proof) = metavm_zkp::mpt::single_leaf_trie(&trie_key, &value_rlp);
        let access = StorageAccess {
            contract_address: [0; 20],
            slot,
            value,
            is_write: true,
            storage_root: root,
            mpt_proof: proof,
        };
        verify_storage_proof_chain_oracle(&access).unwrap();
    }

    /// **End-to-end integration**: run a real EVM SSTORE+SLOAD
    /// bytecode through the inspector, extract the storage accesses,
    /// build single-leaf storage tries containing each access, build
    /// a complete `WorldProof` bundle (block header + account +
    /// storage), and verify it host-side.
    ///
    /// This validates the entire host-side proof chain for an actual
    /// EVM execution — the algebraic equivalent (when fully wired)
    /// will compose the same pieces via cross-AIR LogUp linkages.
    #[test]
    fn world_proof_built_from_real_evm_execution_verifies() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::account::Account;
        use metavm_zkp::block_header::{block_header_hash, BlockHeader};
        use metavm_zkp::keccak::keccak256;
        use metavm_zkp::mpt::single_leaf_trie;
        use metavm_zkp::receipt::{Receipt, ReceiptType};
        use metavm_zkp::rlp::{rlp_encode_u256, rlp_encode_uint};
        use metavm_zkp::transaction::{LegacyTx, Transaction};
        use metavm_zkp::world_proof::{ContractProof, StorageProof, WorldProof, verify_world_proof_oracle};

        // PUSH1 0x42; PUSH1 0x05; SSTORE; PUSH1 0x05; SLOAD; STOP
        let bytecode = vec![
            0x60, 0x42,  0x60, 0x05,  0x55,
            0x60, 0x05,  0x54,  0x00,
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let storage_accesses = extract_storage_accesses_from_trace(&evm_cols);
        assert_eq!(storage_accesses.len(), 2, "1 SSTORE + 1 SLOAD = 2 accesses");

        // Use the SLOAD access (post-state). Slot=5, value=0x42.
        // For this synthetic test we build a single-leaf storage
        // trie containing just (slot=5 → value=0x42).
        let sload = &storage_accesses[1];
        assert!(!sload.is_write);
        let mut slot_be = [0u8; 32];
        slot_be[31] = 0x05;
        let mut value_be = [0u8; 32];
        value_be[31] = 0x42;
        let trie_key = keccak256(&slot_be);
        let value_rlp = rlp_encode_u256(&value_be);
        let (storage_root, storage_proof) = single_leaf_trie(&trie_key, &value_rlp);

        // Build the contract account with this storage_root.
        let address = sload.contract_address;
        let account = Account {
            nonce: 1,
            balance: [0u8; 32],
            storage_root,
            code_hash: metavm_zkp::account::empty_code_hash(),
        };
        let address_trie_key = keccak256(&address);
        let account_rlp = metavm_zkp::account::account_rlp(&account);
        let (state_root, account_proof) =
            single_leaf_trie(&address_trie_key, &account_rlp);

        // Synthetic transaction (a placeholder; real production
        // would derive from the actual transaction inputs).
        let tx = Transaction::Legacy(LegacyTx {
            nonce: 1,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some(address),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        });
        let tx_index = 0u64;
        let tx_trie_key = rlp_encode_uint(tx_index);
        let (transactions_root, tx_proof) =
            single_leaf_trie(&tx_trie_key, &tx.wire_encoding());

        // Synthetic receipt.
        let receipt = Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: 21_000,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        };
        let (receipts_root, receipt_proof) =
            single_leaf_trie(&tx_trie_key, &receipt.wire_encoding());

        // Block header anchoring all four roots.
        let block_header = BlockHeader {
            state_root,
            transactions_root,
            receipts_root,
            ..Default::default()
        };
        let block_hash = block_header_hash(&block_header);

        // Compose the full proof bundle.
        let proof = WorldProof {
            block_header,
            block_hash,
            tx_index,
            transaction: tx,
            transaction_proof: tx_proof,
            receipt,
            receipt_proof,
            contracts: vec![ContractProof {
                address,
                account,
                account_proof,
                storage_proofs: vec![StorageProof {
                    slot: slot_be,
                    value: value_be,
                    proof: storage_proof,
                }],
            }],
        };

        // Verify the entire bundle.
        verify_world_proof_oracle(&proof).unwrap();
    }

    /// EVM-trace → storage gadget witness round-trip: run an
    /// SSTORE+SLOAD bytecode through the inspector, build the gadget
    /// witness, and verify the gadget AIR's witness columns have the
    /// expected `(slot, value, is_write)` per access.
    #[test]
    fn build_storage_gadget_witness_from_real_evm_trace() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::storage_access_air::{
            build_trace_polynomials, COL_IS_WRITE, COL_SLOT_L0, COL_VALUE_L0,
        };

        // PUSH1 0x42; PUSH1 0x05; SSTORE; PUSH1 0x05; SLOAD; STOP
        let bytecode = vec![
            0x60, 0x42,  0x60, 0x05,  0x55,
            0x60, 0x05,  0x54,  0x00,
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();

        // Fake roots: the gadget AIR commits to these but doesn't yet
        // verify them against an MPT (that's step 1d).
        let storage_roots = vec![[0u8; 32]; 2];
        let witness = build_storage_gadget_witness_from_trace(
            &evm_cols, &storage_roots,
        )
        .unwrap();
        assert_eq!(witness.invocations.len(), 2);

        // Row 0 is SSTORE (slot=5, value=0x42, is_write=1).
        assert_eq!(witness.invocations[0].slot[0], 0x05);
        assert_eq!(witness.invocations[0].value[0], 0x42);
        assert!(witness.invocations[0].is_write);

        // Row 1 is SLOAD (slot=5, value=0x42, is_write=0).
        assert_eq!(witness.invocations[1].slot[0], 0x05);
        assert_eq!(witness.invocations[1].value[0], 0x42);
        assert!(!witness.invocations[1].is_write);

        // Build the gadget trace and verify column shapes.
        let trace = build_trace_polynomials(&witness, CurveType::Bls48581);
        assert_eq!(trace.num_rows, 2);
        // is_write: row 0 = 1 (SSTORE), row 1 = 0 (SLOAD).
        assert_eq!(trace.columns[COL_IS_WRITE].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_IS_WRITE].evaluations[1].to_u64(), 0);
        // slot[L0]: both rows = 5.
        assert_eq!(trace.columns[COL_SLOT_L0].evaluations[0].to_u64(), 5);
        assert_eq!(trace.columns[COL_SLOT_L0].evaluations[1].to_u64(), 5);
        // value[L0]: both rows = 0x42.
        assert_eq!(trace.columns[COL_VALUE_L0].evaluations[0].to_u64(), 0x42);
        assert_eq!(trace.columns[COL_VALUE_L0].evaluations[1].to_u64(), 0x42);
    }

    /// Extract storage accesses from a real EVM trace.
    #[test]
    fn extract_from_trace_picks_sload_sstore_rows() {
        use crate::executor::execute_bytecode;
        // PUSH1 0x42; PUSH1 0x05; SSTORE; PUSH1 0x05; SLOAD; STOP
        let bytecode = vec![
            0x60, 0x42, // PUSH1 0x42 (value)
            0x60, 0x05, // PUSH1 0x05 (slot)
            0x55,       // SSTORE
            0x60, 0x05, // PUSH1 0x05 (slot)
            0x54,       // SLOAD
            0x00,       // STOP
        ];
        let trace = execute_bytecode(&bytecode, &[]).unwrap();
        let accesses = extract_storage_accesses_from_trace(&trace);
        assert_eq!(accesses.len(), 2);

        // First access is SSTORE.
        assert!(accesses[0].is_write);
        assert_eq!(accesses[0].slot, U256::from(0x05u64));
        assert_eq!(accesses[0].value, U256::from(0x42u64));

        // Second is SLOAD that reads the stored value back.
        assert!(!accesses[1].is_write);
        assert_eq!(accesses[1].slot, U256::from(0x05u64));
        assert_eq!(accesses[1].value, U256::from(0x42u64));
    }
}
