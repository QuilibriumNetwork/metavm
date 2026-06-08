//! End-to-end reference integration — execution → block → consensus →
//! finality — stitched together from every major module in this crate.
//!
//! This is a demonstration artifact, not an AIR and not a proof. It builds
//! a plausible-but-synthetic Ethereum scenario using real data structures
//! (RLP, MPT, block header, bloom, SSZ, beacon containers, BLS signatures,
//! Casper FFG finality) and checks that they're mutually consistent.
//!
//! The single test below walks:
//! 1. Execution layer — craft a legacy transaction, RLP-encode + keccak256
//!    for `tx_hash`. Build a one-tx MPT and recover `transactions_root`.
//! 2. Receipts — a status-1 receipt with no logs; one-leaf MPT root.
//! 3. Block header — assemble the post-merge 15-field shape, RLP + keccak
//!    for `block_hash`.
//! 4. Consensus — treat `block_hash` as the `body_root` of a beacon block
//!    header; compute its SSZ hash-tree-root.
//! 5. Attestations — 4 synthetic validators sign the beacon header root
//!    under the beacon DST; 3 of 4 aggregate and verify.
//! 6. Finality — stake-weighted check over the 3 signers finalizes the
//!    beacon block root; assert weight = 3 * 32 ETH (Gwei).
//! 7. Print a one-line summary of the public outputs.

#[cfg(test)]
mod tests {
    use crate::beacon::{
        AttestationData, BeaconBlockHeader, Checkpoint, IndexedAttestation, Validator,
    };
    use crate::block_header::{
        block_header_hash, block_header_rlp, empty_list_hash, BlockHeader, LOGS_BLOOM_LEN,
    };
    use crate::bloom::logs_bloom_for_block;
    use crate::bls_sig::{
        aggregate_pubkeys, aggregate_sigs, fast_aggregate_verify, verify, PublicKey, SecretKey,
        Signature,
    };
    use crate::finality::{
        stake_weighted_finalization_check, ValidatorRegistry, BEACON_DST,
    };
    use crate::keccak::keccak256;
    use crate::mpt::{mpt_node_hash, mpt_node_rlp, verify_mpt_inclusion, MptNode, Nibbles};
    use crate::rlp::{rlp_encode_bytes, rlp_encode_list, rlp_encode_u256, rlp_encode_uint};

    /// Hex-encode a byte slice (test-only).
    fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for &x in b {
            s.push_str(&format!("{:02x}", x));
        }
        s
    }

    /// Build a 32-byte big-endian scalar from a seed byte (1..=255) by
    /// placing the seed in the last byte. Valid for small seeds; this is
    /// the same trick the `bls_sig` tests use.
    fn deterministic_sk(seed: u8) -> SecretKey {
        let mut bytes = [0u8; 32];
        bytes[31] = seed;
        SecretKey::from_bytes(&bytes).expect("seed must be a valid non-zero BLS scalar")
    }

    /// Encode a 256-bit big-endian integer from a u64 low-word into [u8; 32].
    fn u256_be_from_u64(v: u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[24..].copy_from_slice(&v.to_be_bytes());
        out
    }

    /// RLP-encode a legacy transaction (type 0x00): rlp([nonce, gasPrice,
    /// gasLimit, to, value, data, v, r, s]). For a pre-signed synthetic test
    /// we fill (v, r, s) with placeholder minimal integers — the goal is a
    /// well-formed RLP list, not a real ECDSA-valid tx.
    fn encode_legacy_tx(
        nonce: u64,
        gas_price: u64,
        gas_limit: u64,
        to: &[u8; 20],
        value_wei: &[u8; 32],
        data: &[u8],
        v: u64,
        r: &[u8; 32],
        s: &[u8; 32],
    ) -> Vec<u8> {
        let items = vec![
            rlp_encode_uint(nonce),
            rlp_encode_uint(gas_price),
            rlp_encode_uint(gas_limit),
            rlp_encode_bytes(to),
            rlp_encode_u256(value_wei),
            rlp_encode_bytes(data),
            rlp_encode_uint(v),
            rlp_encode_u256(r),
            rlp_encode_u256(s),
        ];
        rlp_encode_list(&items)
    }

    /// RLP-encode a legacy receipt: rlp([status, cumulative_gas_used,
    /// logs_bloom, logs]). `logs_bloom` is 256 bytes; `logs` is a list of
    /// `[address, topics, data]`. Here we emit a zero bloom and empty logs.
    fn encode_legacy_receipt(
        status: u64,
        cumulative_gas_used: u64,
        logs_bloom: &[u8; 256],
        logs: &[(Vec<u8>, Vec<Vec<u8>>, Vec<u8>)],
    ) -> Vec<u8> {
        let encoded_logs: Vec<Vec<u8>> = logs
            .iter()
            .map(|(addr, topics, data)| {
                let topics_enc: Vec<Vec<u8>> = topics.iter().map(|t| rlp_encode_bytes(t)).collect();
                rlp_encode_list(&[
                    rlp_encode_bytes(addr),
                    rlp_encode_list(&topics_enc),
                    rlp_encode_bytes(data),
                ])
            })
            .collect();
        let items = vec![
            rlp_encode_uint(status),
            rlp_encode_uint(cumulative_gas_used),
            rlp_encode_bytes(logs_bloom),
            rlp_encode_list(&encoded_logs),
        ];
        rlp_encode_list(&items)
    }

    /// Build a one-leaf MPT trie whose single key/value pair is `(key, value)`.
    /// Returns `(root, leaf_rlp)`. The root is `keccak256(leaf_rlp)` so the
    /// caller can hand `[leaf_rlp]` back to `verify_mpt_inclusion`.
    fn single_leaf_trie(key: &[u8], value: &[u8]) -> ([u8; 32], Vec<u8>) {
        let leaf = MptNode::Leaf {
            path: Nibbles::from_bytes(key),
            value: value.to_vec(),
        };
        let leaf_rlp = mpt_node_rlp(&leaf);
        let root = mpt_node_hash(&leaf);
        (root, leaf_rlp)
    }

    // =========================================================================
    // The full-stack reference test.
    // =========================================================================

    #[test]
    fn e2e_reference_execution_to_finality() {
        // ----------------------------------------------------------------
        // Step 1: Execution layer — one legacy transaction.
        // ----------------------------------------------------------------
        let sender_addr: [u8; 20] = [
            0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef, 0xde, 0xad,
            0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef,
        ];
        let _ = sender_addr; // anchor comment — not encoded in the unsigned tx
        let recipient: [u8; 20] = [
            0xca, 0xfe, 0xba, 0xbe, 0xca, 0xfe, 0xba, 0xbe, 0xca, 0xfe,
            0xba, 0xbe, 0xca, 0xfe, 0xba, 0xbe, 0xca, 0xfe, 0xba, 0xbe,
        ];
        let one_eth_wei: [u8; 32] = {
            // 1 ETH = 10^18 wei = 0x0de0b6b3a7640000.
            let mut v = [0u8; 32];
            v[24..].copy_from_slice(&1_000_000_000_000_000_000u64.to_be_bytes());
            v
        };

        // Synthetic signature placeholders (v=27, r=s=1): the RLP is
        // well-formed but we don't claim this is an ECDSA-valid signature.
        let r_placeholder = u256_be_from_u64(1);
        let s_placeholder = u256_be_from_u64(1);

        let tx_rlp = encode_legacy_tx(
            0,              // nonce
            20_000_000_000, // gas price = 20 gwei
            21_000,         // gas limit
            &recipient,
            &one_eth_wei,
            &[], // no calldata
            27,
            &r_placeholder,
            &s_placeholder,
        );
        let tx_hash = keccak256(&tx_rlp);

        // Cross-check: re-RLP and re-hash produces the same thing.
        let tx_rlp_again = encode_legacy_tx(
            0,
            20_000_000_000,
            21_000,
            &recipient,
            &one_eth_wei,
            &[],
            27,
            &r_placeholder,
            &s_placeholder,
        );
        assert_eq!(tx_rlp, tx_rlp_again, "tx RLP encoding must be deterministic");
        assert_eq!(
            keccak256(&tx_rlp_again),
            tx_hash,
            "re-hashing the same tx RLP must yield the same tx_hash"
        );

        // Build the one-transaction MPT: key = rlp(tx_index = 0) = [0x80],
        // value = tx_rlp. Root = keccak256(leaf).
        let tx_index_key = rlp_encode_uint(0);
        assert_eq!(tx_index_key, vec![0x80], "rlp(0) = 0x80 (empty string)");
        let (tx_root, tx_leaf_rlp) = single_leaf_trie(&tx_index_key, &tx_rlp);

        // Inclusion check: the tx must verify under `tx_root`.
        let tx_proof = vec![tx_leaf_rlp.clone()];
        assert!(
            verify_mpt_inclusion(tx_root, &tx_index_key, &tx_rlp, &tx_proof),
            "transaction must be included under tx_root"
        );

        // Negative check: wrong value must fail.
        assert!(
            !verify_mpt_inclusion(tx_root, &tx_index_key, b"not-the-tx", &tx_proof),
            "wrong value must not verify"
        );

        // ----------------------------------------------------------------
        // Step 2: Receipts — one status-1 receipt with no logs.
        // ----------------------------------------------------------------
        let zero_bloom: [u8; 256] = [0u8; 256];
        let receipt_rlp = encode_legacy_receipt(1, 21_000, &zero_bloom, &[]);
        let (receipts_root, receipt_leaf_rlp) = single_leaf_trie(&tx_index_key, &receipt_rlp);
        let receipt_proof = vec![receipt_leaf_rlp.clone()];
        assert!(
            verify_mpt_inclusion(receipts_root, &tx_index_key, &receipt_rlp, &receipt_proof),
            "receipt must be included under receipts_root"
        );

        // ----------------------------------------------------------------
        // Step 3: Block header.
        // ----------------------------------------------------------------
        // Canonical empty-ommers hash: keccak256(rlp([])) = keccak256(&[0xc0]).
        let ommers_hash = empty_list_hash();
        let expected_ommers: [u8; 32] = [
            0x1d, 0xcc, 0x4d, 0xe8, 0xde, 0xc7, 0x5d, 0x7a, 0xab, 0x85,
            0xb5, 0x67, 0xb6, 0xcc, 0xd4, 0x1a, 0xd3, 0x12, 0x45, 0x1b,
            0x94, 0x8a, 0x74, 0x13, 0xf0, 0xa1, 0x42, 0xfd, 0x40, 0xd4,
            0x93, 0x47,
        ];
        assert_eq!(
            ommers_hash, expected_ommers,
            "canonical empty-ommers hash must match the well-known constant"
        );

        // Block-level bloom must equal [0; 256] since we have zero logs.
        let block_bloom = logs_bloom_for_block(&[]);
        assert_eq!(
            block_bloom,
            [0u8; LOGS_BLOOM_LEN],
            "block with no logs must have an all-zero bloom"
        );

        let header = BlockHeader {
            parent_hash: [0u8; 32], // block-1 of a synthetic chain
            ommers_hash,
            beneficiary: [0u8; 20],
            state_root: [0u8; 32], // synthetic
            transactions_root: tx_root,
            receipts_root,
            logs_bloom: block_bloom,
            difficulty: [0u8; 32],
            number: 1,
            gas_limit: 30_000_000,
            gas_used: 21_000,
            timestamp: 1_700_000_000,
            extra_data: Vec::new(),
            mix_hash: [0u8; 32],
            nonce: [0u8; 8],
            base_fee_per_gas: None,
            withdrawals_root: None,
            blob_gas_used: None,
            excess_blob_gas: None,
            parent_beacon_block_root: None,
        };
        let block_hash = block_header_hash(&header);

        // Cross-check: re-encoding + re-hashing the header yields the same hash.
        let header_rlp_a = block_header_rlp(&header);
        let header_rlp_b = block_header_rlp(&header);
        assert_eq!(header_rlp_a, header_rlp_b, "block header RLP must be deterministic");
        assert_eq!(keccak256(&header_rlp_a), block_hash);

        // ----------------------------------------------------------------
        // Step 4: Consensus layer — stand-in beacon block header whose
        // body_root is our execution block hash.
        // ----------------------------------------------------------------
        let beacon_block_header = BeaconBlockHeader {
            slot: 32, // first slot of epoch 1
            proposer_index: 0,
            parent_root: [0u8; 32],
            state_root: [0u8; 32],
            body_root: block_hash, // anchors execution ← consensus
        };
        let beacon_block_root = beacon_block_header.hash_tree_root();

        // Determinism cross-check.
        assert_eq!(
            beacon_block_header.hash_tree_root(),
            beacon_block_root,
            "beacon block root must be deterministic"
        );

        // ----------------------------------------------------------------
        // Step 5: Validator registry + attestations.
        // ----------------------------------------------------------------
        const NUM_VALIDATORS: usize = 4;
        const EFFECTIVE_BALANCE: u64 = 32_000_000_000; // 32 ETH in Gwei

        // Generate 4 deterministic (sk, pk) pairs.
        let sks: Vec<SecretKey> = (1..=NUM_VALIDATORS as u8).map(deterministic_sk).collect();
        let pks: Vec<PublicKey> = sks.iter().map(|sk| sk.public_key()).collect();

        let validators: Vec<Validator> = pks
            .iter()
            .map(|pk| Validator {
                pubkey: pk.0,
                withdrawal_credentials: [0u8; 32],
                effective_balance: EFFECTIVE_BALANCE,
                slashed: false,
                activation_eligibility_epoch: 0,
                activation_epoch: 0,
                exit_epoch: 1u64 << 40, // far-future
                withdrawable_epoch: (1u64 << 40) + 256,
            })
            .collect();
        let balances = vec![EFFECTIVE_BALANCE; NUM_VALIDATORS];
        let registry = ValidatorRegistry::new(validators, balances);

        // Build the attestation: target.root = beacon_block_root.
        let attestation_data = AttestationData {
            slot: 32,
            index: 0,
            beacon_block_root,
            source: Checkpoint {
                epoch: 0,
                root: [0u8; 32],
            },
            target: Checkpoint {
                epoch: 1,
                root: beacon_block_root,
            },
        };
        let att_signing_root = attestation_data.hash_tree_root();

        // 3 of 4 validators sign the attestation root.
        let signing_indices: Vec<u64> = vec![0, 1, 2];
        let signer_sks: Vec<&SecretKey> = signing_indices
            .iter()
            .map(|&i| &sks[i as usize])
            .collect();
        let signer_pks: Vec<PublicKey> = signing_indices
            .iter()
            .map(|&i| pks[i as usize])
            .collect();

        // Individually sign + individually verify each share.
        let individual_sigs: Vec<Signature> = signer_sks
            .iter()
            .map(|sk| sk.sign(&att_signing_root, BEACON_DST))
            .collect();
        for (pk, sig) in signer_pks.iter().zip(individual_sigs.iter()) {
            assert!(
                verify(pk, &att_signing_root, sig, BEACON_DST),
                "individual BLS signature must verify"
            );
        }

        // Aggregate pubkeys + signatures, then fast-aggregate-verify.
        let agg_sig = aggregate_sigs(&individual_sigs).expect("aggregate 3 signatures");
        let _agg_pk = aggregate_pubkeys(&signer_pks).expect("aggregate 3 pubkeys");
        assert!(
            fast_aggregate_verify(&signer_pks, &att_signing_root, &agg_sig, BEACON_DST),
            "aggregate BLS signature must verify under beacon DST"
        );

        // Negative cross-check: wrong DST must reject.
        assert!(
            !fast_aggregate_verify(
                &signer_pks,
                &att_signing_root,
                &agg_sig,
                b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_",
            ),
            "aggregate signature must fail under a wrong DST"
        );

        // ----------------------------------------------------------------
        // Step 6: Stake-weighted finality check.
        // ----------------------------------------------------------------
        let finalized = Checkpoint {
            epoch: 1,
            root: beacon_block_root,
        };
        let indexed_attestation = IndexedAttestation {
            attesting_indices: signing_indices.clone(),
            data: attestation_data,
            signature: agg_sig.0,
        };
        let finality_output = stake_weighted_finalization_check(
            &finalized,
            &registry,
            &[indexed_attestation],
        );

        assert_eq!(
            finality_output.finalized_block_root, beacon_block_root,
            "finalized block root must equal the beacon block root"
        );
        assert_eq!(
            finality_output.total_effective_balance_weight,
            3 * EFFECTIVE_BALANCE,
            "finality weight must equal 3 * 32 ETH in Gwei"
        );
        assert_eq!(
            finality_output.num_unique_attesters, 3,
            "3 unique attesters"
        );

        // Consistency: the stake-weighted output's weight must equal the
        // registry's total_balance over the signing indices.
        assert_eq!(
            finality_output.total_effective_balance_weight,
            registry.total_balance(&signing_indices),
            "finality weight must equal registry.total_balance(signers)"
        );

        // ----------------------------------------------------------------
        // Step 7: build a LayerChain from the boundary values, verify it.
        //
        // This is the cross-layer rollup interface — each layer's claim
        // captures its public input/output, and the chain checks the
        // boundaries match. Once the four AIRs all wire to a prover, this
        // chain becomes the "single statement" the production proof binds.
        // ----------------------------------------------------------------
        let att_data_root = att_signing_root;
        let boundaries = crate::layer_chain::ChainBoundaries {
            block_hash,
            beacon_block_root,
            attestation_data_root: att_data_root,
            num_attesters: finality_output.num_unique_attesters,
            finalized_root: finality_output.finalized_block_root,
            total_effective_balance_gwei: finality_output.total_effective_balance_weight,
        };
        let chain = crate::layer_chain::LayerChain::from_boundaries(&boundaries);
        chain.verify_consistency()
            .expect("layer chain must verify against e2e values");
        let final_out = chain.final_output().expect("final output");
        assert_eq!(final_out.block_hash, block_hash);
        assert_eq!(final_out.beacon_block_root, beacon_block_root);
        assert_eq!(final_out.finalized_root, beacon_block_root);
        assert_eq!(
            final_out.total_effective_balance_gwei,
            3 * EFFECTIVE_BALANCE
        );

        // Single-hash commitment over all four layer boundaries — the
        // public-input root a recursive proof would expose. Cross-check
        // that LayerChain::commitment() and FinalOutput::commitment_with()
        // agree on the same chain.
        let chain_commit = chain.commitment().expect("commitment");
        let final_commit = final_out.commitment_with(
            &boundaries.attestation_data_root,
            boundaries.num_attesters,
        );
        assert_eq!(
            chain_commit, final_commit,
            "chain and final-output commitments must agree",
        );

        // Wrap as a LayerChainProof envelope (reference-only — per-layer
        // AIR proofs not yet plugged in for every layer). Verifies via
        // chain consistency + per-layer proof verification (no-op on
        // empty proof_bytes by design).
        let chain_proof = crate::layer_chain::LayerChainProof::new(
            chain
                .claims
                .iter()
                .cloned()
                .map(crate::layer_chain::LayerProof::reference_only)
                .collect(),
        );
        chain_proof.verify().expect("layer chain proof must verify");
        assert_eq!(
            chain_proof.commitment().expect("commitment"),
            chain_commit,
            "LayerChainProof commitment must equal LayerChain commitment",
        );

        // Exercise the closure-based per-layer verifier dispatch path.
        // Every layer is reference-only here (no AIR proof bytes attached
        // in the reference walk), so the closure should never fire — we
        // assert this with a counter. Once individual layers carry real
        // proofs, the closure dispatches to per-AIR verifiers.
        use std::cell::Cell;
        let dispatch_calls = Cell::new(0u32);
        chain_proof
            .verify_with_layer_verifier(|_layer| {
                dispatch_calls.set(dispatch_calls.get() + 1);
                Ok(())
            })
            .expect("dispatched verify on reference-only chain must pass");
        assert_eq!(
            dispatch_calls.get(),
            0,
            "no real proof bytes attached → dispatch closure must not fire",
        );

        // ----------------------------------------------------------------
        // Step 8: one-line summary of the proof's public outputs.
        // ----------------------------------------------------------------
        eprintln!(
            "e2e_reference: execution_block_hash=0x{} beacon_block_root=0x{} \
             finality_weight_gwei={}",
            hex(&block_hash),
            hex(&beacon_block_root),
            finality_output.total_effective_balance_weight,
        );
    }

    /// End-to-end with MULTIPLE real per-layer proofs in a single
    /// `LayerChainProof`. Each layer that's wired through to a prover
    /// gets a real `ExecutionProof` attached and verified through the
    /// chain's closure-based dispatch — first multi-kind composition
    /// test, validates the envelope under realistic load.
    ///
    /// Layers covered (3 of 4):
    ///   * `BlockBinding` → `LayerProofKind::Keccak` (block-header keccak256)
    ///   * `Attestation`  → `LayerProofKind::NonnativeFp` (BLS pairing
    ///     stand-in — real BLS verify trace would be ~107k FpOps which
    ///     can't be proven directly today; uses a small synthetic Fp trace)
    ///   * `Finality`     → `LayerProofKind::Ssz` (validator-registry
    ///     merkleization)
    ///
    /// `Execution` stays reference-only since RISC-V/EVM/SBF VM AIRs
    /// produce per-chunk proofs (covered by the existing tree_fold
    /// machinery), not single ExecutionProofs that fit this slot.
    ///
    /// Total: 3 real proofs aggregated through the envelope. Fast path is
    /// dominated by the per-AIR prove/verify costs (Keccak ~5min,
    /// NonnativeFp ~9min, Ssz ~10s in release mode).
    #[test]
    #[ignore = "slow: 3 separate prove/verify roundtrips; run with --release --ignored"]
    fn multi_air_proofs_flow_through_layer_chain_envelope() {
        use crate::field::CurveType;
        use crate::keccak::{keccak_f1600_witness, State};
        use crate::keccak_constraints::{
            build_trace_polynomials_from_rounds as keccak_build_trace,
            KeccakConstraintSystem,
        };
        use crate::layer_chain::{
            ChainBoundaries, LayerChain, LayerChainProof, LayerProof, LayerProofKind,
        };
        use crate::nonnative_fp::Fp;
        use crate::nonnative_fp_air::FpOp;
        use crate::nonnative_fp_constraints::{
            build_fp_trace_polynomials, NonnativeFpConstraintSystem,
        };
        use crate::scheme::{bls48581_scheme::Bls48581Scheme, CommitmentScheme};
        use crate::ssz::Chunk;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as ssz_build_trace, SszConstraintSystem,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ---- 1. Keccak proof (BlockBinding layer) -----------------------
        let state: State = [[0u64; 5]; 5];
        let keccak_rounds = keccak_f1600_witness(state);
        let keccak_trace = keccak_build_trace(&keccak_rounds, curve);
        let keccak_domain = keccak_trace.padded_size;
        let keccak_omega = scheme.domain_generator(keccak_domain);
        let keccak_cs = KeccakConstraintSystem::new(keccak_rounds.len())
            .with_omega_and_domain(keccak_omega.clone(), keccak_domain);
        let keccak_proof =
            crate::prover::prove_with_scheme(&keccak_trace, &keccak_cs, &scheme);
        let keccak_bytes = keccak_proof.to_bytes();
        let keccak_num_rounds = keccak_rounds.len();

        // ---- 2. NonnativeFp proof (Attestation layer) -------------------
        let fp_ops = vec![
            FpOp::Add { a: Fp::from_u64(3), b: Fp::from_u64(7) },
            FpOp::Mul { a: Fp::from_u64(5), b: Fp::from_u64(11) },
            FpOp::Inv { a: Fp::from_u64(0xC0FFEE) },
        ];
        let fp_trace = build_fp_trace_polynomials(&fp_ops, curve);
        let fp_cs = NonnativeFpConstraintSystem::new();
        let fp_proof = crate::prover::prove_with_scheme(&fp_trace, &fp_cs, &scheme);
        let fp_bytes = fp_proof.to_bytes();

        // ---- 3. SSZ proof (Finality layer) ------------------------------
        fn ssz_chunk(byte: u8) -> Chunk {
            let mut c = [0u8; 32];
            c[0] = byte;
            c
        }
        let ssz_rows = merkleize_witness(
            &[ssz_chunk(0xA0), ssz_chunk(0xB0), ssz_chunk(0xC0), ssz_chunk(0xD0)],
            None,
        );
        let ssz_trace = ssz_build_trace(&ssz_rows, curve);
        let ssz_domain = ssz_trace.padded_size;
        let ssz_omega = scheme.domain_generator(ssz_domain);
        let ssz_cs = SszConstraintSystem::new(ssz_rows.len())
            .with_omega_and_domain(ssz_omega.clone(), ssz_domain);
        let ssz_proof = crate::prover::prove_with_scheme(&ssz_trace, &ssz_cs, &scheme);
        let ssz_bytes = ssz_proof.to_bytes();
        let ssz_num_rows = ssz_rows.len();

        // ---- 4. Build a LayerChainProof with all three attached ---------
        let boundaries = ChainBoundaries {
            block_hash: [0xBB; 32],
            beacon_block_root: [0xCC; 32],
            attestation_data_root: [0xDD; 32],
            num_attesters: 1,
            finalized_root: [0xCC; 32],
            total_effective_balance_gwei: 32_000_000_000,
        };
        let chain = LayerChain::from_boundaries(&boundaries);
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| match i {
                0 => LayerProof::reference_only(claim),
                1 => LayerProof::with_proof(claim, LayerProofKind::Keccak, keccak_bytes.clone()),
                2 => LayerProof::with_proof(claim, LayerProofKind::NonnativeFp, fp_bytes.clone()),
                3 => LayerProof::with_proof(claim, LayerProofKind::Ssz, ssz_bytes.clone()),
                _ => unreachable!(),
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        // ---- 5. Closure-based dispatch on each kind ---------------------
        use std::cell::Cell;
        let calls = Cell::new(0u32);
        let result = chain_proof.verify_with_layer_verifier(|layer| {
            calls.set(calls.get() + 1);
            match layer.kind {
                LayerProofKind::Keccak => {
                    let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                        .map_err(|e| format!("keccak decode: {:?}", e))?;
                    let cs = KeccakConstraintSystem::new(keccak_num_rounds)
                        .with_omega_and_domain(keccak_omega.clone(), keccak_domain);
                    if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                        Ok(())
                    } else {
                        Err("keccak proof did not verify".to_string())
                    }
                }
                LayerProofKind::NonnativeFp => {
                    let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                        .map_err(|e| format!("fp decode: {:?}", e))?;
                    let cs = NonnativeFpConstraintSystem::new();
                    if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                        Ok(())
                    } else {
                        Err("fp proof did not verify".to_string())
                    }
                }
                LayerProofKind::Ssz => {
                    let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                        .map_err(|e| format!("ssz decode: {:?}", e))?;
                    let cs = SszConstraintSystem::new(ssz_num_rows)
                        .with_omega_and_domain(ssz_omega.clone(), ssz_domain);
                    if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                        Ok(())
                    } else {
                        Err("ssz proof did not verify".to_string())
                    }
                }
                LayerProofKind::ReferenceOnly => Ok(()),
                other => Err(format!("unsupported kind {}", other.as_str())),
            }
        });
        assert_eq!(result, Ok(()), "multi-AIR chain must verify end-to-end");
        assert_eq!(calls.get(), 3, "exactly 3 layers carry real proofs");

        let original_commit = chain_proof.commitment().expect("commitment");
        eprintln!(
            "multi_air_proofs_flow: chain commitment = 0x{}",
            hex(&original_commit),
        );

        // ── Persistence: round-trip the LayerChainProof through bytes ──
        //
        // Validates the full flow: serialize the entire envelope (claims +
        // kind tags + per-layer ExecutionProof bytes), reload from bytes,
        // and re-verify against fresh constraint systems. This closes the
        // loop on the persistence story — a real consumer can prove on one
        // machine, save to disk, ship over the wire, reload elsewhere, and
        // verify everything against the chain commitment.
        let serialized = chain_proof.to_bytes();
        eprintln!(
            "multi_air_proofs_flow: serialized envelope = {} bytes",
            serialized.len(),
        );
        let reloaded = crate::layer_chain::LayerChainProof::from_bytes(&serialized)
            .expect("envelope from_bytes must succeed");
        assert_eq!(
            reloaded.commitment().expect("reloaded commitment"),
            original_commit,
            "commitment must survive serialize → from_bytes round-trip",
        );
        // Re-verify the reloaded chain: same closure dispatch, fresh
        // constraint systems built from scratch each call.
        let reload_result = reloaded.verify_with_layer_verifier(|layer| {
            match layer.kind {
                LayerProofKind::Keccak => {
                    let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                        .map_err(|e| format!("keccak decode: {:?}", e))?;
                    let cs = KeccakConstraintSystem::new(keccak_num_rounds)
                        .with_omega_and_domain(keccak_omega.clone(), keccak_domain);
                    if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                        Ok(())
                    } else {
                        Err("reloaded keccak did not verify".to_string())
                    }
                }
                LayerProofKind::NonnativeFp => {
                    let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                        .map_err(|e| format!("fp decode: {:?}", e))?;
                    let cs = NonnativeFpConstraintSystem::new();
                    if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                        Ok(())
                    } else {
                        Err("reloaded fp did not verify".to_string())
                    }
                }
                LayerProofKind::Ssz => {
                    let p = crate::prover::ExecutionProof::from_bytes(&layer.proof_bytes)
                        .map_err(|e| format!("ssz decode: {:?}", e))?;
                    let cs = SszConstraintSystem::new(ssz_num_rows)
                        .with_omega_and_domain(ssz_omega.clone(), ssz_domain);
                    if crate::verifier::verify_with_scheme(&p, &cs, &scheme, curve) {
                        Ok(())
                    } else {
                        Err("reloaded ssz did not verify".to_string())
                    }
                }
                LayerProofKind::ReferenceOnly => Ok(()),
                other => Err(format!("unsupported kind {}", other.as_str())),
            }
        });
        assert_eq!(
            reload_result,
            Ok(()),
            "reloaded multi-AIR chain must verify end-to-end",
        );
    }

    /// Cross-AIR cryptographic fold: prove two different AIRs as
    /// ChunkProofs with chain-derived state hashes, fold them through
    /// the existing IVC accumulator, and verify the accumulated pairing
    /// in one shot. Demonstrates that
    /// `LayerChainFolder::fold_into_recursive_proof_scheme` produces a
    /// `RecursiveProof` whose `verify_final_scheme` returns true.
    ///
    /// Uses SSZ + SHA-256 (fastest pair) so the test runs in ~85s
    /// release-mode. Skipped by default; run with `--release --ignored`.
    #[test]
    #[ignore = "slow: 2 separate prove/verify roundtrips; run with --release --ignored"]
    fn cross_air_fold_chunked_layers() {
        use crate::field::CurveType;
        use crate::keccak::keccak256;
        use crate::layer_chain::{
            LayerChainFolder, LayerChainProof, LayerProof, LayerProofKind,
        };
        use crate::prover::{prove_chunk_with_scheme, ExecutionProof};
        use crate::scheme::{bls48581_scheme::Bls48581Scheme, CommitmentScheme};
        use crate::sha256::sha256_witness;
        use crate::sha256_constraints::{
            build_trace_polynomials_from_hash, Sha256ConstraintSystem,
        };
        use crate::ssz::Chunk;
        use crate::ssz_air::merkleize_witness;
        use crate::ssz_constraints::{
            build_trace_polynomials_from_rows as ssz_build_trace, SszConstraintSystem,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ---- 1. Build the chain (boundaries) BEFORE proving so we can
        //         derive deterministic state hashes that both the prover
        //         and the verifier (fold) use. -----------------------------
        let block_hash = [0xBB; 32];
        let beacon_block_root = [0xCC; 32];
        let attestation_data_root = [0xDD; 32];
        let chain = crate::layer_chain::LayerChain::from_boundaries(
            &crate::layer_chain::ChainBoundaries {
                block_hash,
                beacon_block_root,
                attestation_data_root,
                num_attesters: 1,
                finalized_root: [0xCC; 32],
                total_effective_balance_gwei: 32_000_000_000,
            },
        );
        let chain_commitment = chain.commitment().expect("chain commitment");

        // ---- 2. Derive per-provable-layer state hashes: must match the
        //         derivation in fold_into_recursive_proof_scheme. ----------
        // SSZ goes at provable index 0 (BlockBinding layer, chain index 1),
        // SHA-256 goes at provable index 1 (Attestation layer, chain index 2).
        let derive_next = |prev: &[u8; 32], layer_idx: usize, kind_tag: u8| -> [u8; 32] {
            let mut buf = Vec::with_capacity(32 + 1 + 1 + 8);
            buf.extend_from_slice(prev);
            buf.push(0xF1);
            buf.push(kind_tag);
            buf.extend_from_slice(&(layer_idx as u64).to_be_bytes());
            keccak256(&buf)
        };
        // kind_tag mapping (from layer_chain::encode_kind):
        //   Ssz = 4, Sha256 = 3.
        let s0 = chain_commitment;
        let s1 = derive_next(&s0, 1, 4); // SSZ at chain index 1
        let s2 = derive_next(&s1, 2, 3); // SHA-256 at chain index 2

        // ---- 3. Build SSZ proof as a ChunkProof at chain_idx=1, prov=0 --
        let ssz_rows = merkleize_witness(
            &[Chunk::default(), Chunk::default(), Chunk::default(), Chunk::default()],
            None,
        );
        let ssz_num_rows = ssz_rows.len();
        let ssz_trace = ssz_build_trace(&ssz_rows, curve);
        let ssz_domain = ssz_trace.padded_size;
        let ssz_omega = scheme.domain_generator(ssz_domain);
        let ssz_cs = SszConstraintSystem::new(ssz_num_rows)
            .with_omega_and_domain(ssz_omega.clone(), ssz_domain);
        let ssz_chunk = prove_chunk_with_scheme(
            &ssz_trace,
            &ssz_cs,
            0, // provable index in chain
            &s0,
            &s1,
            &scheme,
        );
        let ssz_bytes = ssz_chunk.execution_proof.to_bytes();

        // ---- 4. Build SHA-256 proof as a ChunkProof at chain_idx=2, prov=1
        let sha_hash = sha256_witness(b"hello world");
        let sha_num_rounds = sha_hash.blocks.len() * 64;
        let sha_trace = build_trace_polynomials_from_hash(&sha_hash, curve);
        let sha_domain = sha_trace.padded_size;
        let sha_omega = scheme.domain_generator(sha_domain);
        let sha_cs = Sha256ConstraintSystem::new(sha_num_rounds)
            .with_omega_and_domain(sha_omega.clone(), sha_domain);
        let sha_chunk = prove_chunk_with_scheme(
            &sha_trace,
            &sha_cs,
            1, // provable index in chain
            &s1,
            &s2,
            &scheme,
        );
        let sha_bytes = sha_chunk.execution_proof.to_bytes();

        // Sanity: ExecutionProofs round-trip through bytes.
        let _ = ExecutionProof::from_bytes(&ssz_bytes).expect("ssz exec from_bytes");
        let _ = ExecutionProof::from_bytes(&sha_bytes).expect("sha exec from_bytes");

        // ---- 5. Build the LayerChainProof ------------------------------
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| match i {
                1 => LayerProof::with_proof(claim, LayerProofKind::Ssz, ssz_bytes.clone()),
                2 => LayerProof::with_proof(claim, LayerProofKind::Sha256, sha_bytes.clone()),
                _ => LayerProof::reference_only(claim),
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        // ---- 6. Cross-AIR fold + single pairing check ------------------
        let recursive = LayerChainFolder::fold_into_recursive_proof_scheme(
            &chain_proof,
            &scheme,
            curve,
        )
        .expect("cross-AIR fold");
        assert_eq!(recursive.depth, 2, "2 provable layers folded");
        assert_eq!(recursive.accumulator.num_folded, 2);
        assert!(
            crate::recursive::verify_final_scheme(&recursive, &scheme),
            "cross-AIR recursive proof must verify in one pairing check",
        );
        eprintln!(
            "cross_air_fold_chunked: depth={} num_folded={} ok",
            recursive.depth, recursive.accumulator.num_folded,
        );

        // ── Full-fold variant ────────────────────────────────────────────
        //
        // Aggregate ALL per-layer KZG opening pairings (main + shifted +
        // logup + bitwise + perm + reg_perm + frame_perm and their
        // shifted variants — whichever the proof carries) into the same
        // (L_acc, R_acc) accumulator via the meta-challenge ξ. One
        // pairing check then certifies every aggregated opening for both
        // layers simultaneously, eliminating the need for per-layer
        // verify_with_scheme calls (modulo the per-AIR constraint identity
        // scalar, which is tracked separately for the pending scalar
        // accumulator integration).
        let ssz_num_rows_for_dispatch = ssz_num_rows;
        let ssz_omega_for_dispatch = ssz_omega.clone();
        let ssz_domain_for_dispatch = ssz_domain;
        let sha_num_rounds_for_dispatch = sha_num_rounds;
        let sha_omega_for_dispatch = sha_omega.clone();
        let sha_domain_for_dispatch = sha_domain;
        let recursive_full = LayerChainFolder::fold_into_recursive_proof_full_scheme(
            &chain_proof,
            &scheme,
            curve,
            |kind| -> Box<dyn crate::vm_constraints::VmConstraintSystem> {
                match kind {
                    LayerProofKind::Ssz => Box::new(
                        SszConstraintSystem::new(ssz_num_rows_for_dispatch)
                            .with_omega_and_domain(
                                ssz_omega_for_dispatch.clone(),
                                ssz_domain_for_dispatch,
                            ),
                    ),
                    LayerProofKind::Sha256 => Box::new(
                        Sha256ConstraintSystem::new(sha_num_rounds_for_dispatch)
                            .with_omega_and_domain(
                                sha_omega_for_dispatch.clone(),
                                sha_domain_for_dispatch,
                            ),
                    ),
                    other => panic!("cs_dispatch: unsupported kind {}", other.as_str()),
                }
            },
        )
        .expect("full-fold cross-AIR fold");
        assert_eq!(recursive_full.depth, 2);
        assert_eq!(recursive_full.accumulator.num_folded, 2);
        assert!(
            crate::recursive::verify_final_scheme(&recursive_full, &scheme),
            "full-fold cross-AIR recursive proof must verify in one pairing check",
        );
        eprintln!(
            "cross_air_fold_full_chunked: depth={} num_folded={} ok",
            recursive_full.depth, recursive_full.accumulator.num_folded,
        );
    }

    /// Unified Eth → beacon → finality proof demo.
    ///
    /// Walks the four-layer architecture end-to-end with the layers
    /// available today wired as real `ChunkProof`s and the rest as
    /// `ReferenceOnly` placeholders, then collapses everything into a
    /// single [`crate::recursive::RecursiveProof`] via
    /// [`crate::layer_chain::LayerChainFolder::fold_into_recursive_proof_full_scheme`].
    /// One [`crate::recursive::verify_final_scheme`] call decides every
    /// layer's pairing aggregation plus the per-AIR constraint identity
    /// scalar.
    ///
    /// This is the zkp-crate-local version of the demo (no `metavm_evm`
    /// dependency, so L1 Execution is `ReferenceOnly` and L3 Attestation
    /// is also `ReferenceOnly` to avoid the BLS-aggregation prover cost).
    /// L2 (SSZ HTR via SHA-256) and L4 (Finality) carry real proofs.
    ///
    /// For the all-four-real-layers version see
    /// `metavm_evm::constraints::tests::unified_eth_finalized_block_proof_all_real_layers`.
    #[test]
    #[ignore = "slow: 2 real prove/verify roundtrips + 4-layer fold; run with --release --ignored"]
    fn unified_eth_finalized_block_proof() {
        use crate::field::CurveType;
        use crate::finality_constraints::{
            build_finality_trace_polynomials, FinalityConstraintSystem, FinalityWitness,
        };
        use crate::keccak::keccak256;
        use crate::layer_chain::{
            ChainBoundaries, LayerChain, LayerChainFolder, LayerChainProof, LayerProof,
            LayerProofKind,
        };
        use crate::prover::{prove_chunk_with_scheme, ExecutionProof};
        use crate::scheme::{bls48581_scheme::Bls48581Scheme, CommitmentScheme};
        use crate::sha256::sha256_witness;
        use crate::sha256_constraints::{
            build_trace_polynomials_from_hash, Sha256ConstraintSystem,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ---- Synthetic boundary values (an integrator would pull these
        //      from a real Ethereum block + beacon chain context). -----
        let block_hash = [0xBB; 32];
        let beacon_block_root = [0xCC; 32];
        let attestation_data_root = [0xDD; 32];
        let finalized_root = [0xEE; 32];

        // Finality witness (4-validator toy committee, all attesting).
        let finality_witness = FinalityWitness {
            validators: vec![
                (32_000_000_000, 1),
                (32_000_000_000, 1),
                (32_000_000_000, 1),
                (32_000_000_000, 0),
            ],
            total_active_balance_gwei: 4 * 32_000_000_000,
            attestation_data_root,
            finalized_root,
        };
        let num_attesters = finality_witness
            .validators
            .iter()
            .filter(|(_, b)| *b == 1)
            .count() as u64;

        let chain = LayerChain::from_boundaries(&ChainBoundaries {
            block_hash,
            beacon_block_root,
            attestation_data_root,
            num_attesters,
            finalized_root,
            total_effective_balance_gwei: finality_witness.total_attesting_balance(),
        });
        let chain_commitment = chain.commitment().expect("well-formed chain");

        // Derive deterministic per-provable-layer state hashes — must
        // match `LayerChainFolder::fold_into_recursive_proof_full_scheme`.
        let derive_next =
            |prev: &[u8; 32], layer_idx: usize, kind_tag: u8| -> [u8; 32] {
                let mut buf = Vec::with_capacity(42);
                buf.extend_from_slice(prev);
                buf.push(0xF1);
                buf.push(kind_tag);
                buf.extend_from_slice(&(layer_idx as u64).to_be_bytes());
                keccak256(&buf)
            };
        // kind_tag mapping: Sha256 = 3 (L2), Finality = 11 (L4).
        let s0 = chain_commitment;
        let s1 = derive_next(&s0, 1, 3); // L2 SHA-256 at chain idx 1
        let s2 = derive_next(&s1, 3, 11); // L4 Finality at chain idx 3

        // ---- L2 SHA-256 / SSZ HTR proof ---------------------------------
        let sha_hash = sha256_witness(b"unified_eth_finalized_block_demo");
        let sha_num_rounds = sha_hash.blocks.len() * 64;
        let sha_trace = build_trace_polynomials_from_hash(&sha_hash, curve);
        let sha_domain = sha_trace.padded_size;
        let sha_omega = scheme.domain_generator(sha_domain);
        let sha_cs = Sha256ConstraintSystem::new(sha_num_rounds)
            .with_omega_and_domain(sha_omega.clone(), sha_domain);
        let sha_chunk = prove_chunk_with_scheme(
            &sha_trace, &sha_cs, 0, // first provable layer
            &s0, &s1, &scheme,
        );
        let sha_bytes = sha_chunk.execution_proof.to_bytes();

        // ---- L4 Finality proof ------------------------------------------
        let fin_trace = build_finality_trace_polynomials(&finality_witness, curve);
        let fin_domain = fin_trace.padded_size;
        let fin_omega = scheme.domain_generator(fin_domain);
        let fin_cs = FinalityConstraintSystem::new(finality_witness.validators.len())
            .with_omega_and_domain(fin_omega.clone(), fin_domain);
        let fin_chunk = prove_chunk_with_scheme(
            &fin_trace, &fin_cs, 1, // second provable layer
            &s1, &s2, &scheme,
        );
        let fin_bytes = fin_chunk.execution_proof.to_bytes();

        // Sanity: both proofs round-trip through bytes.
        let _ = ExecutionProof::from_bytes(&sha_bytes).expect("sha exec from_bytes");
        let _ = ExecutionProof::from_bytes(&fin_bytes).expect("fin exec from_bytes");

        // ---- Build the 4-layer chain proof ------------------------------
        let layers: Vec<LayerProof> = chain
            .claims
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, claim)| match i {
                // L1 Execution: reference-only (placeholder until the
                // production `prove-evm` output is folded in).
                0 => LayerProof::reference_only(claim),
                // L2 BlockBinding: real SHA-256 proof.
                1 => LayerProof::with_proof(claim, LayerProofKind::Sha256, sha_bytes.clone()),
                // L3 Attestation: reference-only here to keep the test
                // fast (real BLS aggregation is exercised separately in
                // bls_sig_constraints).
                2 => LayerProof::reference_only(claim),
                // L4 Finality: real Finality proof.
                3 => LayerProof::with_proof(claim, LayerProofKind::Finality, fin_bytes.clone()),
                _ => unreachable!(),
            })
            .collect();
        let chain_proof = LayerChainProof::new(layers);

        // ---- Cross-AIR full-fold + single pairing decision --------------
        let sha_num_rounds_dispatch = sha_num_rounds;
        let sha_omega_dispatch = sha_omega.clone();
        let sha_domain_dispatch = sha_domain;
        let fin_committee_size = finality_witness.validators.len();
        let fin_omega_dispatch = fin_omega.clone();
        let fin_domain_dispatch = fin_domain;
        let recursive = LayerChainFolder::fold_into_recursive_proof_full_scheme(
            &chain_proof,
            &scheme,
            curve,
            |kind| -> Box<dyn crate::vm_constraints::VmConstraintSystem> {
                match kind {
                    LayerProofKind::Sha256 => Box::new(
                        Sha256ConstraintSystem::new(sha_num_rounds_dispatch)
                            .with_omega_and_domain(
                                sha_omega_dispatch.clone(),
                                sha_domain_dispatch,
                            ),
                    ),
                    LayerProofKind::Finality => Box::new(
                        FinalityConstraintSystem::new(fin_committee_size)
                            .with_omega_and_domain(
                                fin_omega_dispatch.clone(),
                                fin_domain_dispatch,
                            ),
                    ),
                    other => panic!("cs_dispatch: unsupported kind {}", other.as_str()),
                }
            },
        )
        .expect("4-layer cross-AIR fold");

        assert_eq!(recursive.depth, 2, "2 provable layers folded (L2 + L4)");
        assert_eq!(recursive.accumulator.num_folded, 2);
        assert!(
            crate::recursive::verify_final_scheme(&recursive, &scheme),
            "unified Eth-finalized-block proof must verify in one pairing check",
        );

        // ---- Persistence: round-trip the whole RecursiveProof ----------
        let recursive_bytes = recursive.to_bytes();
        eprintln!(
            "unified_eth_finalized_block: chain_commitment=0x{} \
             single_proof_bytes={} verified=true",
            hex(&chain_commitment),
            recursive_bytes.len(),
        );
        let reloaded = crate::recursive::RecursiveProof::from_bytes(&recursive_bytes)
            .expect("RecursiveProof from_bytes");
        assert!(
            crate::recursive::verify_final_scheme(&reloaded, &scheme),
            "reloaded unified proof must still verify",
        );
    }
}
