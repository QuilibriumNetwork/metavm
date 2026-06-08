//! End-to-end integration test: FFG-finalized beacon block → execution
//! block_hash, validating every host-side oracle and witness module
//! built in the session.
//!
//! This single test exercises:
//!   1. Per-field RLP encoding (all 20 fields via 7 gadgets)
//!   2. RLP list concat (running-offset chain)
//!   3. Assembled RLP = canonical `block_header_rlp()`
//!   4. `keccak256(assembled) = block_header_hash()`
//!   5. ExecutionPayloadHeader HTR witness (20 pair invocations)
//!   6. BeaconBlockBody HTR witness (12 pair invocations)
//!   7. BeaconBlockHeader HTR witness (7 pair invocations)
//!   8. BeaconHtrComposition chain oracle (BBH→body→payload)
//!   9. BeaconWorldProof oracle (BBH→body→payload→block_header→tx/receipt/account/storage)
//!  10. FFG finality oracle (stake-weighted threshold)
//!  11. LogsBloom sub-tree HTR witness (7 pair invocations)
//!  12. ExtraData sub-tree HTR witness (1 pair invocation)

#[cfg(test)]
mod tests {
    use crate::account::Account;
    use crate::beacon::BeaconBlockHeader;
    use crate::beacon_block_body::BeaconBlockBody;
    use crate::beacon_block_body_air::BeaconBlockBodyHtrWitness;
    use crate::beacon_block_header_air::BeaconBlockHeaderHtrWitness;
    use crate::beacon_htr_composition::{BeaconHtrComposition, verify_beacon_htr_chain_oracle};
    use crate::beacon_world_proof::{
        BeaconWorldProof, FinalityWitness,
        verify_beacon_world_proof_oracle, verify_finalized_beacon_world_proof_oracle,
    };
    use crate::block_header::{block_header_hash, block_header_rlp, BlockHeader};
    use crate::execution_payload::ExecutionPayloadHeader;
    use crate::execution_payload_air::ExecutionPayloadHeaderHtrWitness;
    use crate::extra_data_air::ExtraDataHtrWitness;
    use crate::finality::ValidatorRegistry;
    use crate::keccak::keccak256;
    use crate::logs_bloom_air::LogsBloomHtrWitness;
    use crate::mpt::single_leaf_trie;
    use crate::receipt::{Receipt, ReceiptType};
    use crate::rlp_header_composition::verify_header_rlp_composition;
    use crate::rlp_list_concat_air::RlpListConcatWitness;
    use crate::transaction::{LegacyTx, Transaction};
    use crate::world_proof::{ContractProof, StorageProof, WorldProof};

    /// **Master integration test**: validates every host-side module
    /// from this session in a single consistent composition.
    #[test]
    fn e2e_finalized_beacon_to_block_hash() {
        // ══════════════════════════════════════════════════════════════
        // Step 1: Build an Ethereum execution block header (Layer B)
        // ══════════════════════════════════════════════════════════════
        let address = [0xab_u8; 20];
        let mut slot_be = [0u8; 32]; slot_be[31] = 7;
        let mut value_be = [0u8; 32]; value_be[31] = 0x42;
        let trie_key = keccak256(&slot_be);
        let value_rlp = crate::rlp::rlp_encode_u256(&value_be);
        let (storage_root, storage_proof) = single_leaf_trie(&trie_key, &value_rlp);
        let account = Account {
            nonce: 1, balance: [0u8; 32], storage_root,
            code_hash: crate::account::empty_code_hash(),
        };
        let account_trie_key = keccak256(&address);
        let account_rlp = crate::account::account_rlp(&account);
        let (state_root, account_proof) = single_leaf_trie(&account_trie_key, &account_rlp);

        let tx = Transaction::Legacy(LegacyTx {
            nonce: 1, gas_price: [0u8; 32], gas_limit: 21_000,
            to: Some([0x42u8; 20]), value: [0u8; 32], data: Vec::new(),
            v: 27, r: [0x11u8; 32], s: [0x22u8; 32],
        });
        let tx_wire = tx.wire_encoding();
        let tx_trie_key = crate::rlp::rlp_encode_uint(0u64);
        let (tx_root, tx_proof) = single_leaf_trie(&tx_trie_key, &tx_wire);

        let receipt = Receipt {
            ty: ReceiptType::Legacy, status: 1,
            cumulative_gas_used: 21_000, logs_bloom: [0u8; 256], logs: Vec::new(),
        };
        let receipt_wire = receipt.wire_encoding();
        let (receipts_root, receipt_proof) = single_leaf_trie(&tx_trie_key, &receipt_wire);

        let block_header = BlockHeader {
            state_root, transactions_root: tx_root, receipts_root,
            number: 18_500_000, gas_limit: 30_000_000, gas_used: 21_000,
            timestamp: 1_700_000_000,
            logs_bloom: [0x77; 256],
            extra_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            base_fee_per_gas: Some({
                let mut b = [0u8; 32];
                b[24..32].copy_from_slice(&15_000_000_000u64.to_be_bytes()); b
            }),
            withdrawals_root: Some([0x99; 32]),
            blob_gas_used: Some(393_216),
            excess_blob_gas: Some(786_432),
            parent_beacon_block_root: Some([0xaa; 32]),
            ..Default::default()
        };
        let block_hash = block_header_hash(&block_header);

        // ══════════════════════════════════════════════════════════════
        // Step 2: Validate per-field RLP encoding (all 20 fields)
        // ══════════════════════════════════════════════════════════════
        let assembled_rlp = verify_header_rlp_composition(&block_header)
            .expect("per-field RLP gadgets must produce canonical RLP");
        assert_eq!(assembled_rlp, block_header_rlp(&block_header));
        assert_eq!(keccak256(&assembled_rlp), block_hash);

        // ══════════════════════════════════════════════════════════════
        // Step 3: Validate RLP list concat running-offset chain
        // ══════════════════════════════════════════════════════════════
        let concat_w = RlpListConcatWitness::from_block_header(&block_header);
        concat_w.verify_offset_chain().expect("offset chain valid");
        concat_w.verify_byte_alignment(&assembled_rlp).expect("byte alignment valid");

        // ══════════════════════════════════════════════════════════════
        // Step 4: Build the C↔B bridge (payload → body → BBH)
        // ══════════════════════════════════════════════════════════════
        let mut payload = ExecutionPayloadHeader::default();
        payload.block_hash = block_hash;
        payload.logs_bloom = block_header.logs_bloom;
        payload.extra_data = block_header.extra_data.clone();
        payload.block_number = block_header.number;
        payload.gas_limit = block_header.gas_limit;
        payload.timestamp = block_header.timestamp;

        let body = BeaconBlockBody {
            execution_payload_header: payload.clone(),
            ..Default::default()
        };
        let bbh = BeaconBlockHeader {
            slot: 7_777_777, proposer_index: 13,
            parent_root: [0xaa; 32], state_root: [0xbb; 32],
            body_root: body.hash_tree_root(),
        };

        // ══════════════════════════════════════════════════════════════
        // Step 5: Validate all HTR witnesses
        // ══════════════════════════════════════════════════════════════
        let payload_w = ExecutionPayloadHeaderHtrWitness::from_payload(payload.clone());
        assert_eq!(payload_w.root, payload.hash_tree_root());
        assert_eq!(payload_w.invocations.len(), 20);

        let body_w = BeaconBlockBodyHtrWitness::from_body(body.clone());
        assert_eq!(body_w.root, body.hash_tree_root());
        assert_eq!(body_w.invocations.len(), 12);

        let bbh_w = BeaconBlockHeaderHtrWitness::from_header(bbh);
        assert_eq!(bbh_w.root, bbh.hash_tree_root());
        assert_eq!(bbh_w.invocations.len(), 7);

        // Sub-tree witnesses.
        let logs_w = LogsBloomHtrWitness::from_logs_bloom(payload.logs_bloom);
        assert_eq!(logs_w.root, payload_w.field_roots[4]);

        let extra_w = ExtraDataHtrWitness::from_extra_data(payload.extra_data.clone());
        assert_eq!(extra_w.root, payload_w.field_roots[10]);

        // ══════════════════════════════════════════════════════════════
        // Step 6: Validate BeaconHtrComposition chain
        // ══════════════════════════════════════════════════════════════
        let comp = BeaconHtrComposition::from_header(bbh, body.clone());
        verify_beacon_htr_chain_oracle(&comp).expect("HTR chain valid");
        assert_eq!(comp.execution_block_hash(), block_hash);

        // ══════════════════════════════════════════════════════════════
        // Step 7: Validate full BeaconWorldProof
        // ══════════════════════════════════════════════════════════════
        let world = WorldProof {
            block_header, block_hash, tx_index: 0,
            transaction: tx, transaction_proof: tx_proof,
            receipt, receipt_proof,
            contracts: vec![ContractProof {
                address, account, account_proof,
                storage_proofs: vec![StorageProof { slot: slot_be, value: value_be, proof: storage_proof }],
            }],
        };
        let bwp = BeaconWorldProof {
            beacon_block_header: bbh,
            body,
            world,
        };
        verify_beacon_world_proof_oracle(&bwp).expect("BeaconWorldProof valid");

        // ══════════════════════════════════════════════════════════════
        // Step 8: Validate FFG finality
        // ══════════════════════════════════════════════════════════════
        let eb = 32_000_000_000u64;
        let bbh_root = bbh.hash_tree_root();
        let finality_w = FinalityWitness {
            registry: ValidatorRegistry::new(
                (0..4).map(|_| crate::beacon::Validator {
                    pubkey: [0u8; 48], withdrawal_credentials: [0u8; 32],
                    effective_balance: eb, slashed: false,
                    activation_eligibility_epoch: 0, activation_epoch: 0,
                    exit_epoch: 1u64 << 40, withdrawable_epoch: (1u64 << 40) + 256,
                }).collect(),
                vec![eb; 4],
            ),
            epoch: 100,
            attestations: vec![crate::beacon::IndexedAttestation {
                attesting_indices: vec![0, 1, 2],
                data: crate::beacon::AttestationData {
                    slot: 100 * 32, index: 0, beacon_block_root: bbh_root,
                    source: crate::beacon::Checkpoint { epoch: 99, root: [0xaa; 32] },
                    target: crate::beacon::Checkpoint { epoch: 100, root: bbh_root },
                },
                signature: [0u8; 96],
            }],
            claimed_finalized: crate::beacon::Checkpoint { epoch: 100, root: bbh_root },
        };
        verify_finalized_beacon_world_proof_oracle(&bwp, &finality_w, 2, 3)
            .expect("finalized BeaconWorldProof valid");

        eprintln!("[e2e] All 12 pipeline stages validated successfully:");
        eprintln!("  ✅ 20/20 per-field RLP encoding gadgets");
        eprintln!("  ✅ RLP list concat running-offset chain");
        eprintln!("  ✅ Assembled RLP = canonical block_header_rlp");
        eprintln!("  ✅ keccak256(RLP) = block_header_hash");
        eprintln!("  ✅ ExecutionPayloadHeader HTR (20 invocations)");
        eprintln!("  ✅ BeaconBlockBody HTR (12 invocations)");
        eprintln!("  ✅ BeaconBlockHeader HTR (7 invocations)");
        eprintln!("  ✅ LogsBloom + ExtraData sub-tree HTR");
        eprintln!("  ✅ BeaconHtrComposition chain (BBH→body→payload)");
        eprintln!("  ✅ BeaconWorldProof (tx/receipt/account/storage)");
        eprintln!("  ✅ FFG finality (3/4 validators, 75% > 2/3)");
        eprintln!("  ✅ block_hash = {:?}", block_hash);
    }
}
