//! Task #305 — finality-proof master `joint_prove` composer driving
//! **all four layers** of the consensus → execution stack in ONE
//! joint_prove call.
//!
//! This is the top-level composer that, when fully validated under
//! `joint_prove` / `joint_verify`, attests that a **single transaction
//! inside a single execution block** belongs to a **beacon block** that
//! is **finalised by Casper FFG with a 2/3 stake-weighted
//! supermajority**. The witness in this harness is the canonical
//! 1-tx-1-block-1-BBH-1-finalisation pattern.
//!
//! ## 10-AIR composition
//!
//! - layer 0: [`crate::tx_full_chain_air`] (Layer A — per-tx summary)
//! - layer 1: [`crate::receipt_status_air`] (Layer A+ — receipt status
//!   & cumulative-gas chain)
//! - layer 2: [`crate::block_header_air`] (Layer B — Ethereum block
//!   header)
//! - layer 3: [`crate::multi_block_proof_air`] (Layer B+ — parent-hash
//!   chain of blocks)
//! - layer 4: [`crate::bbh_root_consumer_air`] (Layer C — carries the
//!   beacon block header `state_root` for binding)
//! - layer 5: [`crate::beacon_state_transition_air`] (Layer C+ —
//!   slot-level state-root carry-forward)
//! - layer 6: [`crate::block_full_proof_air`] (Layer C++ — block-proof
//!   AIR with `COL_BLOCK_HASH_OFFSET`; substituted for
//!   `execution_payload_air` because the latter is an HTR witness
//!   builder, not a constraint-system AIR, and does not publish a
//!   row-level `block_hash` column)
//! - layer 7: [`crate::casper_ffg_chain_air`] (Layer D — FFG
//!   justification & finalisation)
//! - layer 8: [`crate::finality_constraints`] (Layer D+ —
//!   stake-weighted finality)
//! - layer 9: [`crate::withdrawal_root_air`] (Layer A+ — Shanghai
//!   withdrawals MPT, included so the 10-AIR composition is closed at
//!   the executable-block boundary)
//!
//! ## 7 cross-layer descriptors (single-column tuples)
//!
//! All descriptors are single-column tuples (the `joint_prove` API
//! supports `a_columns.len() == 1` only — see
//! `cross_air_logup::build_linkage_trace`'s assertion). Anchor bytes
//! reference the byte-0 column of any 32-byte hash field.
//!
//!  * **D0 (A→B)** — `tx_full_chain.COL_TX_INDEX` ↔
//!    `block_header.COL_TRANSACTIONS_ROOT_OFFSET[0]` (anchor: tx_index
//!    matches byte-0 of the block's transactions_root).
//!
//!  * **D1 (B→B+)** — `block_header.COL_BLOCK_HASH_OFFSET[0]` ↔
//!    `multi_block.COL_BLOCK_HASH_OFFSET[0]` (anchor: byte-0 of the
//!    block hash agrees across the two views of the same block).
//!
//!  * **D2 (B→C)** — `block_header.COL_STATE_ROOT_OFFSET[0]` ↔
//!    `bbh_root_consumer.COL_STATE_ROOT_OFFSET[0]` (anchor: byte-0 of
//!    the EL state_root inside the BBH `body_root` projection).
//!
//!  * **D3 (C→C+)** — `bbh_root_consumer.COL_STATE_ROOT_OFFSET[0]` ↔
//!    `beacon_state_transition.COL_POST_STATE_ROOT_OFFSET[0]` (anchor:
//!    byte-0 of the post-state-root carried into the next BBH).
//!
//!  * **D4 (C++→B)** — `block_full_proof.COL_BLOCK_HASH_OFFSET[0]` ↔
//!    `block_header.COL_BLOCK_HASH_OFFSET[0]` (anchor: byte-0 of the
//!    block hash agrees across the block-proof AIR and the canonical
//!    block-header AIR).
//!
//!  * **D5 (C→D)** — `beacon_state_transition.COL_EPOCH` gated by
//!    `IS_EPOCH_BOUNDARY` ↔ `casper_ffg.COL_TARGET_EPOCH` gated by
//!    `IS_FINALIZED` (anchor: the epoch at the boundary row equals the
//!    target epoch the FFG row finalises).
//!
//!  * **D6 (D→D+)** — `casper_ffg.COL_VOTE_COUNT` gated by
//!    `IS_FINALIZED` ↔ `finality.COL_RUNNING_TOTAL` gated by
//!    `SEL_THRESHOLD` (anchor: the FFG vote count equals the
//!    stake-weighted running-total accumulator on the finality
//!    threshold row).
//!
//! ## Witness shape
//!
//! - One transaction in one block. The block sits in a 3-block
//!   multi-block chain (rows 0..3) at slots `(61, 62, 63)` so slot 63
//!   is the unique epoch-boundary row (epoch = 1).
//! - One BBH row carrying the EL state_root of the target block.
//! - Three FFG rows so row 0 finalises (`source=0, target=1`).
//! - Stake-weighted finality with a single attesting validator.
//!
//! ## Tests
//!
//! - [`descriptor_consistency`] (fast) — descriptor well-formedness,
//!   trace shapes, single-column-tuple invariant, and per-descriptor
//!   on-row closure equality at row 0 (or the active row on each
//!   side).
//! - [`honest_round_trip`] (`#[ignore]`, slow) — full `joint_prove` +
//!   `joint_verify` on BLS48-581.
//! - [`tampered_closure_rejected`] (`#[ignore]`, slow) — mutates one
//!   closure and confirms the verifier rejects.

#[cfg(test)]
mod tests {
    use crate::beacon_state_transition_air as bst;
    use crate::bbh_root_consumer_air as bbh;
    use crate::block_full_proof_air as bfp;
    use crate::block_header::BlockHeader;
    use crate::block_header_air as bh;
    use crate::casper_ffg_chain_air as ffg;
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::finality_constraints as fc;
    use crate::multi_block_proof_air as mb;
    use crate::receipt::{Receipt, ReceiptType};
    use crate::receipt_status_air as rs;
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;
    use crate::transaction::{LegacyTx, Transaction};
    use crate::tx_full_chain_air as tf;
    use crate::withdrawal::Withdrawal;
    use crate::withdrawal_root_air as wr;

    // ─── Honest scalar pins ───────────────────────────────────────────

    /// First slot in the 3-block multi-block chain (chosen so slot 63
    /// is the epoch-1 boundary row, mirroring the
    /// `integration_cross_block_beacon_transition` pattern).
    const FIRST_SLOT: u64 = 61;
    const CHAIN_LENGTH: usize = 3;

    /// Target block within the chain (index 2 — the slot-63
    /// epoch-boundary row). The transaction lives in this block.
    const TARGET_BLOCK_IDX: usize = 2;

    /// FFG finalised target epoch (= D5 honest scalar).
    const FINALIZED_TARGET_EPOCH: u64 = 1;

    /// FFG vote count = stake-weighted finality running total at the
    /// threshold row (= D6 honest scalar). Both sides project the same
    /// u64. Must satisfy `3 · vote_count ≥ 2 · total_active` so the
    /// FFG supermajority constraint holds; we use a small attesting
    /// stake of 1 gwei against a total-active of 1 gwei so the
    /// finality threshold row also passes (3 · 1 ≥ 2 · 1).
    const FFG_VOTE_COUNT: u64 = 1;
    const FFG_TOTAL_ACTIVE: u64 = 1;

    /// First-receipt cumulative gas, also reused as the tx's gas_used
    /// (1 tx → cumulative = gas_used). Kept ≤ 255 so single-byte tuple
    /// anchors line up with full u64 columns.
    const HONEST_GAS_USED: u64 = 21;

    // ─── Helpers ──────────────────────────────────────────────────────

    fn h(seed: u8) -> [u8; mb::HASH_LEN] {
        let mut x = [0u8; mb::HASH_LEN];
        for k in 0..mb::HASH_LEN {
            x[k] = seed.wrapping_add(k as u8);
        }
        x
    }

    /// Per-block block hashes for the 3-block chain. Distinct in byte
    /// 0 to keep the per-row D1 multiset projection well-defined.
    fn block_hashes() -> Vec<[u8; mb::HASH_LEN]> {
        (0..CHAIN_LENGTH).map(|i| h(0x10u8 + i as u8)).collect()
    }

    /// The single transaction belonging to the target block.
    fn tx0() -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce: 0,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [0x11u8; 32],
            s: [0x22u8; 32],
        })
    }

    fn honest_withdrawals() -> [Withdrawal; 1] {
        [Withdrawal {
            index: 0,
            validator_index: 7,
            address: [0x77u8; 20],
            amount: 1_000,
        }]
    }

    fn honest_receipt() -> Receipt {
        Receipt {
            ty: ReceiptType::Legacy,
            status: 1,
            cumulative_gas_used: HONEST_GAS_USED,
            logs_bloom: [0u8; 256],
            logs: Vec::new(),
        }
    }

    /// Synthesise the target-block header so cross-layer byte-0
    /// anchors hold:
    ///
    ///  * `transactions_root[0] == tx_index (= 0)` for D0
    ///  * `block_hash[0] == block_hashes()[TARGET_BLOCK_IDX][0]` for D1
    ///    and D4 — but `block_hash` is computed from the header, so we
    ///    instead route D1/D4 honest equality through the
    ///    `multi_block` / `block_full_proof` witnesses which both
    ///    project the same per-block hash byte. Concretely we **do
    ///    not** force the header-derived `block_hash[0]` to match —
    ///    the descriptor consistency test below uses the derived
    ///    header hash on both sides (since the closure equality runs
    ///    on the active row tuple, and the header AIR's `block_hash`
    ///    column is populated by `from_block_header` which calls
    ///    `block_header_hash`).
    ///  * `state_root[0] == bbh.state_root[0]` for D2 — we hard-code
    ///    state_root[0] to 0xAA so both the header and BBH project the
    ///    same byte.
    fn synth_target_header() -> BlockHeader {
        let mut transactions_root = [0u8; 32];
        // D0 honest: tx_index == 0, transactions_root[0] == 0.
        transactions_root[0] = 0;
        let mut state_root = [0u8; 32];
        state_root[0] = 0xAA;
        let mut receipts_root = [0u8; 32];
        receipts_root[0] = HONEST_GAS_USED as u8;
        let wr_root = crate::withdrawal_root_air::compute_withdrawals_root(
            &honest_withdrawals(),
        );

        let mut hdr = BlockHeader::default();
        hdr.number = FIRST_SLOT + TARGET_BLOCK_IDX as u64; // slot 63
        hdr.timestamp = 1_700_000_000;
        hdr.gas_limit = 30_000_000;
        hdr.gas_used = HONEST_GAS_USED;
        hdr.parent_hash = h(0x10u8 + (TARGET_BLOCK_IDX - 1) as u8);
        hdr.state_root = state_root;
        hdr.transactions_root = transactions_root;
        hdr.receipts_root = receipts_root;
        hdr.withdrawals_root = Some(wr_root);
        hdr
    }

    // ─── Witness builders ─────────────────────────────────────────────

    /// 1-row tx_full_chain witness for the single tx in the target
    /// block. `tx_index = 0` matches the synth header's
    /// `transactions_root[0] = 0` for D0.
    fn build_tf_witness() -> tf::TxFullChainWitness {
        tf::TxFullChainWitness::from_transaction(
            0,
            &tx0(),
            [0u8; 20],
            HONEST_GAS_USED,
            HONEST_GAS_USED,
            1,
        )
    }

    fn build_rs_witness() -> rs::ReceiptStatusWitness {
        rs::ReceiptStatusWitness::from_receipts(&[honest_receipt()])
    }

    fn build_bh_witness() -> bh::BlockHeaderWitness {
        let row = bh::from_block_header(&synth_target_header());
        bh::BlockHeaderWitness::from_headers(vec![row])
    }

    /// 3-row multi-block chain with chained parent/child hashes. The
    /// `state_root` per row matches the same byte-0 anchor as the
    /// header's state_root on the target row.
    fn build_mb_witness() -> mb::MultiBlockProofWitness {
        let bhs = block_hashes();
        // Override the target-block hash to match the canonical
        // header-derived hash so D1 / D4 anchor on the actual
        // block_header_hash byte 0.
        let target_hdr = synth_target_header();
        let target_hash = crate::block_header::block_header_hash(&target_hdr);
        let mut chain_hashes = bhs.clone();
        chain_hashes[TARGET_BLOCK_IDX] = target_hash;

        let mut rows: Vec<(u64, [u8; mb::HASH_LEN], [u8; mb::HASH_LEN],
                           [u8; mb::HASH_LEN], [u8; mb::HASH_LEN])> =
            Vec::with_capacity(CHAIN_LENGTH);
        let mut parent = h(0xee);
        for i in 0..CHAIN_LENGTH {
            let mut sr = [0u8; 32];
            if i == TARGET_BLOCK_IDX {
                sr[0] = 0xAA; // matches header state_root[0]
            } else {
                sr[0] = 0xC0u8.wrapping_add(i as u8);
            }
            let psr = h(0x80u8 + i as u8);
            rows.push((FIRST_SLOT + i as u64, chain_hashes[i], parent, sr, psr));
            parent = chain_hashes[i];
        }
        mb::MultiBlockProofWitness::from_chain(&rows)
    }

    /// Single-row BBH-root consumer witness carrying the target
    /// block's state_root in the BBH's `state_root` column.
    fn build_bbh_witness() -> bbh::BbhRootConsumerWitness {
        let mut sr = [0u8; 32];
        sr[0] = 0xAA; // D2 honest equality
        bbh::BbhRootConsumerWitness::from_bbh_tuple(
            1,
            [0u8; 32], // claimed_root
            [0u8; 32], // parent_root
            sr,        // state_root
            [0u8; 32], // body_root
            FIRST_SLOT + TARGET_BLOCK_IDX as u64,
            0, // proposer_index
        )
    }

    /// 3-row beacon state-transition witness. Slot 63 is the lone
    /// epoch-boundary row (epoch = 1). `post_state_root[0]` on the
    /// boundary row matches the BBH state_root[0] for D3.
    fn build_bst_witness() -> bst::BeaconStateTransitionWitness {
        let mut blocks: Vec<(u64, [u8; 32], [u8; 32])> =
            Vec::with_capacity(CHAIN_LENGTH);
        for i in 0..CHAIN_LENGTH {
            let block_root = h(0x10u8 + i as u8);
            let mut post_sr = [0u8; 32];
            if i == TARGET_BLOCK_IDX {
                post_sr[0] = 0xAA; // matches bbh.state_root[0]
            } else {
                post_sr[0] = 0xA0u8.wrapping_add(i as u8);
            }
            blocks.push((FIRST_SLOT + i as u64, block_root, post_sr));
        }
        bst::BeaconStateTransitionWitness::from_chain(h(0xc0), &blocks)
    }

    fn build_bfp_witness() -> bfp::BlockFullProofWitness {
        let hdr = synth_target_header();
        bfp::BlockFullProofWitness::from_block(
            &hdr,
            &[tx0()],
            &honest_withdrawals(),
            0, // proposer_index
        )
    }

    /// 3-epoch FFG witness: row 0 (`source=0,target=1`) finalises.
    fn build_ffg_witness() -> ffg::CasperFfgChainWitness {
        let tallies = vec![
            (0u64, FINALIZED_TARGET_EPOCH, FFG_VOTE_COUNT, FFG_TOTAL_ACTIVE),
            (1u64, 2u64, FFG_VOTE_COUNT, FFG_TOTAL_ACTIVE),
            (5u64, 6u64, FFG_VOTE_COUNT, FFG_TOTAL_ACTIVE),
        ];
        ffg::CasperFfgChainWitness::from_tallies(&tallies)
    }

    /// Single-validator finality witness. `RUNNING_TOTAL` at the
    /// threshold row equals the validator's attesting balance =
    /// `FFG_VOTE_COUNT`.
    fn build_fin_witness() -> fc::FinalityWitness {
        let validators = vec![(FFG_VOTE_COUNT, 1u8)];
        let mut finalized_root = [0u8; 32];
        finalized_root[0] = 0xAA;
        fc::FinalityWitness::new(
            validators,
            FFG_TOTAL_ACTIVE,
            [0u8; 32],
            finalized_root,
        )
    }

    fn build_wr_witness() -> wr::WithdrawalRootWitness {
        wr::WithdrawalRootWitness::from_withdrawals(&honest_withdrawals())
    }

    // ─── Descriptors ──────────────────────────────────────────────────

    /// D0: tx_full_chain.COL_TX_INDEX ↔ block_header.COL_TRANSACTIONS_ROOT_OFFSET[0].
    fn d0_tx_index_to_tx_root_byte0() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "finality_master_d0_tx_index_to_tx_root_byte0_v1".into(),
            a_layer_index: 0,
            a_columns: vec![tf::COL_TX_INDEX],
            a_selector_column: Some(tf::COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![bh::COL_TRANSACTIONS_ROOT_OFFSET],
            b_selector_column: Some(bh::COL_IS_REAL),
        }
    }

    /// D1: block_header.COL_BLOCK_HASH_OFFSET[0] ↔ multi_block.COL_BLOCK_HASH_OFFSET[0].
    fn d1_bh_to_mb_block_hash_byte0() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "finality_master_d1_bh_to_mb_block_hash_byte0_v1".into(),
            a_layer_index: 2,
            a_columns: vec![bh::COL_BLOCK_HASH_OFFSET],
            a_selector_column: Some(bh::COL_IS_REAL),
            b_layer_index: 3,
            b_columns: vec![mb::COL_BLOCK_HASH_OFFSET],
            b_selector_column: Some(mb::COL_IS_REAL),
        }
    }

    /// D2: block_header.COL_STATE_ROOT_OFFSET[0] ↔ bbh_root_consumer.COL_STATE_ROOT_OFFSET[0].
    fn d2_bh_to_bbh_state_root_byte0() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "finality_master_d2_bh_to_bbh_state_root_byte0_v1".into(),
            a_layer_index: 2,
            a_columns: vec![bh::COL_STATE_ROOT_OFFSET],
            a_selector_column: Some(bh::COL_IS_REAL),
            b_layer_index: 4,
            b_columns: vec![bbh::COL_STATE_ROOT_OFFSET],
            b_selector_column: Some(bbh::COL_IS_REAL),
        }
    }

    /// D3: bbh_root_consumer.COL_STATE_ROOT_OFFSET[0] ↔
    /// beacon_state_transition.COL_POST_STATE_ROOT_OFFSET[0] (on the
    /// epoch-boundary row of the BST trace via IS_EPOCH_BOUNDARY).
    fn d3_bbh_to_bst_post_state_root_byte0() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "finality_master_d3_bbh_to_bst_post_state_root_byte0_v1".into(),
            a_layer_index: 4,
            a_columns: vec![bbh::COL_STATE_ROOT_OFFSET],
            a_selector_column: Some(bbh::COL_IS_REAL),
            b_layer_index: 5,
            b_columns: vec![bst::COL_POST_STATE_ROOT_OFFSET],
            b_selector_column: Some(bst::COL_IS_EPOCH_BOUNDARY),
        }
    }

    /// D4: block_full_proof.COL_BLOCK_HASH_OFFSET[0] ↔ block_header.COL_BLOCK_HASH_OFFSET[0].
    fn d4_bfp_to_bh_block_hash_byte0() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "finality_master_d4_bfp_to_bh_block_hash_byte0_v1".into(),
            a_layer_index: 6,
            a_columns: vec![bfp::COL_BLOCK_HASH_OFFSET],
            a_selector_column: Some(bfp::COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![bh::COL_BLOCK_HASH_OFFSET],
            b_selector_column: Some(bh::COL_IS_REAL),
        }
    }

    /// D5: beacon_state_transition.COL_EPOCH (gated by IS_EPOCH_BOUNDARY) ↔
    /// casper_ffg.COL_TARGET_EPOCH (gated by IS_FINALIZED).
    fn d5_bst_epoch_to_ffg_target() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "finality_master_d5_bst_epoch_to_ffg_target_v1".into(),
            a_layer_index: 5,
            a_columns: vec![bst::COL_EPOCH],
            a_selector_column: Some(bst::COL_IS_EPOCH_BOUNDARY),
            b_layer_index: 7,
            b_columns: vec![ffg::COL_TARGET_EPOCH],
            b_selector_column: Some(ffg::COL_IS_FINALIZED),
        }
    }

    /// D6: casper_ffg.COL_VOTE_COUNT (gated by IS_FINALIZED) ↔
    /// finality.COL_RUNNING_TOTAL (gated by SEL_THRESHOLD).
    fn d6_ffg_vote_count_to_finality_running_total() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "finality_master_d6_ffg_vote_count_to_finality_running_total_v1"
                .into(),
            a_layer_index: 7,
            a_columns: vec![ffg::COL_VOTE_COUNT],
            a_selector_column: Some(ffg::COL_IS_FINALIZED),
            b_layer_index: 8,
            b_columns: vec![fc::COL_RUNNING_TOTAL],
            b_selector_column: Some(fc::COL_SEL_THRESHOLD),
        }
    }

    fn all_descriptors() -> Vec<CrossAirLogUpDescriptor> {
        vec![
            d0_tx_index_to_tx_root_byte0(),
            d1_bh_to_mb_block_hash_byte0(),
            d2_bh_to_bbh_state_root_byte0(),
            d3_bbh_to_bst_post_state_root_byte0(),
            d4_bfp_to_bh_block_hash_byte0(),
            d5_bst_epoch_to_ffg_target(),
            d6_ffg_vote_count_to_finality_running_total(),
        ]
    }

    // ─── Fast static check ────────────────────────────────────────────

    #[test]
    fn descriptor_consistency() {
        let curve = CurveType::Bls48581;

        let descriptors = all_descriptors();
        assert_eq!(descriptors.len(), 7, "7 cross-layer descriptors");

        // Single-column tuple invariant + layer-index bounds.
        for d in &descriptors {
            assert_eq!(d.a_columns.len(), 1, "joint_prove v0 → 1-col tuples");
            assert_eq!(d.b_columns.len(), 1, "joint_prove v0 → 1-col tuples");
            assert!(d.a_layer_index < 10);
            assert!(d.b_layer_index < 10);
            assert_ne!(d.a_layer_index, d.b_layer_index);
        }

        // Build all 10 witnesses and traces.
        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let bh_w = build_bh_witness();
        let mb_w = build_mb_witness();
        let bbh_w = build_bbh_witness();
        let bst_w = build_bst_witness();
        let bfp_w = build_bfp_witness();
        let ffg_w = build_ffg_witness();
        let fin_w = build_fin_witness();
        let wr_w = build_wr_witness();

        let tf_trace = tf::build_trace_polynomials(&tf_w, curve);
        let rs_trace = rs::build_trace_polynomials(&rs_w, curve);
        let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
        let mb_trace = mb::build_trace_polynomials(&mb_w, curve);
        let bbh_trace = bbh::build_trace_polynomials(&bbh_w, curve);
        let bst_trace = bst::build_trace_polynomials(&bst_w, curve);
        let bfp_trace = bfp::build_trace_polynomials(&bfp_w, curve);
        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);
        let fin_trace = fc::build_finality_trace_polynomials(&fin_w, curve);
        let wr_trace = wr::build_trace_polynomials(&wr_w, curve);

        // Witness/trace shape sanity.
        assert_eq!(tf_w.rows.len(), 1);
        assert_eq!(rs_w.rows.len(), 1);
        assert_eq!(bh_w.headers.len(), 1);
        assert_eq!(mb_w.rows.len(), CHAIN_LENGTH);
        assert_eq!(bbh_w.rows.len(), 1);
        assert_eq!(bst_w.rows.len(), CHAIN_LENGTH);
        assert_eq!(ffg_w.rows.len(), 3);

        // Slot 63 is the lone epoch-boundary row with epoch=1.
        assert!(bst_w.rows[TARGET_BLOCK_IDX].is_epoch_boundary);
        assert_eq!(bst_w.rows[TARGET_BLOCK_IDX].epoch, 1);

        // Row 0 of FFG finalises with target_epoch=1.
        assert!(ffg_w.rows[0].is_finalized);
        assert_eq!(ffg_w.rows[0].target_epoch, FINALIZED_TARGET_EPOCH);

        // ─── Column bounds against published constants ──────────────
        let d = &descriptors;
        assert!(d[0].a_columns[0] < tf::NUM_COLUMNS);
        assert!(d[0].b_columns[0] < bh::NUM_COLUMNS);
        assert!(d[1].a_columns[0] < bh::NUM_COLUMNS);
        assert!(d[1].b_columns[0] < mb::NUM_COLUMNS);
        assert!(d[2].a_columns[0] < bh::NUM_COLUMNS);
        assert!(d[2].b_columns[0] < bbh::NUM_COLUMNS);
        assert!(d[3].a_columns[0] < bbh::NUM_COLUMNS);
        assert!(d[3].b_columns[0] < bst::NUM_COLUMNS);
        assert!(d[4].a_columns[0] < bfp::NUM_COLUMNS);
        assert!(d[4].b_columns[0] < bh::NUM_COLUMNS);
        assert!(d[5].a_columns[0] < bst::NUM_COLUMNS);
        assert!(d[5].b_columns[0] < ffg::NUM_COLUMNS);
        assert!(d[6].a_columns[0] < ffg::NUM_COLUMNS);
        assert!(d[6].b_columns[0] < fc::NUM_COLUMNS);

        // ─── Per-descriptor honest closure equality on active rows ──

        // D0: tx_index (row 0) == transactions_root[0] (row 0).
        let d0_a = tf_trace.columns[tf::COL_TX_INDEX].evaluations[0].to_u64();
        let d0_b = bh_trace.columns[bh::COL_TRANSACTIONS_ROOT_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(d0_a, d0_b, "D0: tx_index ↔ transactions_root[0]");
        assert_eq!(d0_a, 0);

        // D1: bh.block_hash[0] (row 0) == mb.block_hash[0] on target row.
        let d1_a = bh_trace.columns[bh::COL_BLOCK_HASH_OFFSET]
            .evaluations[0]
            .to_u64();
        let d1_b = mb_trace.columns[mb::COL_BLOCK_HASH_OFFSET]
            .evaluations[TARGET_BLOCK_IDX]
            .to_u64();
        assert_eq!(d1_a, d1_b, "D1: bh.block_hash[0] ↔ mb.block_hash[0]");

        // D2: bh.state_root[0] (row 0) == bbh.state_root[0] (row 0).
        let d2_a = bh_trace.columns[bh::COL_STATE_ROOT_OFFSET]
            .evaluations[0]
            .to_u64();
        let d2_b = bbh_trace.columns[bbh::COL_STATE_ROOT_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(d2_a, d2_b, "D2: bh.state_root[0] ↔ bbh.state_root[0]");
        assert_eq!(d2_a, 0xAA);

        // D3: bbh.state_root[0] (row 0) ==
        //     bst.post_state_root[0] on the epoch-boundary row.
        let d3_a = bbh_trace.columns[bbh::COL_STATE_ROOT_OFFSET]
            .evaluations[0]
            .to_u64();
        let d3_b = bst_trace.columns[bst::COL_POST_STATE_ROOT_OFFSET]
            .evaluations[TARGET_BLOCK_IDX]
            .to_u64();
        assert_eq!(d3_a, d3_b, "D3: bbh.state_root[0] ↔ bst.post_state_root[0]");

        // D4: bfp.block_hash[0] (row 0 = header row) == bh.block_hash[0].
        // Find the header row of bfp (where IS_HEADER == 1) — by
        // construction row 0 is the header row.
        let d4_a = bfp_trace.columns[bfp::COL_BLOCK_HASH_OFFSET]
            .evaluations[0]
            .to_u64();
        let d4_b = bh_trace.columns[bh::COL_BLOCK_HASH_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(d4_a, d4_b, "D4: bfp.block_hash[0] ↔ bh.block_hash[0]");

        // D5: bst.epoch on boundary row == ffg.target_epoch on finalised row.
        let d5_a = bst_trace.columns[bst::COL_EPOCH]
            .evaluations[TARGET_BLOCK_IDX]
            .to_u64();
        let d5_b = ffg_trace.columns[ffg::COL_TARGET_EPOCH]
            .evaluations[0]
            .to_u64();
        assert_eq!(d5_a, d5_b, "D5: bst.epoch ↔ ffg.target_epoch");
        assert_eq!(d5_a, FINALIZED_TARGET_EPOCH);

        // D6: ffg.vote_count on finalised row == finality.running_total on threshold row.
        let threshold_row = fin_w.validators.len();
        let d6_a = ffg_trace.columns[ffg::COL_VOTE_COUNT]
            .evaluations[0]
            .to_u64();
        let d6_b = fin_trace.columns[fc::COL_RUNNING_TOTAL]
            .evaluations[threshold_row]
            .to_u64();
        assert_eq!(d6_a, d6_b, "D6: ffg.vote_count ↔ finality.running_total");
        assert_eq!(d6_a, FFG_VOTE_COUNT);

        // ─── Selector active-row sanity ─────────────────────────────
        assert_eq!(
            bst_trace.columns[bst::COL_IS_EPOCH_BOUNDARY]
                .evaluations[TARGET_BLOCK_IDX]
                .to_u64(),
            1,
        );
        assert_eq!(
            ffg_trace.columns[ffg::COL_IS_FINALIZED]
                .evaluations[0]
                .to_u64(),
            1,
        );
        assert_eq!(
            fin_trace.columns[fc::COL_SEL_THRESHOLD]
                .evaluations[threshold_row]
                .to_u64(),
            1,
        );

        // ─── joint_prove input shape (10 traces + 7 linkages) ──────
        let tf_cs = tf::TxFullChainConstraintSystem::new(tf_trace.num_rows);
        let rs_cs = rs::ReceiptStatusConstraintSystem::new(rs_trace.num_rows);
        let bh_cs = bh::BlockHeaderConstraintSystem::new(bh_trace.num_rows);
        let mb_cs = mb::MultiBlockProofConstraintSystem::new(mb_trace.num_rows);
        let bbh_cs = bbh::BbhRootConsumerConstraintSystem::new(bbh_trace.num_rows);
        let bst_cs = bst::BeaconStateTransitionConstraintSystem::new(bst_trace.num_rows);
        let bfp_cs = bfp::BlockFullProofConstraintSystem::new(bfp_trace.num_rows);
        let ffg_cs = ffg::CasperFfgChainConstraintSystem::new(ffg_trace.num_rows);
        let fin_cs = fc::FinalityConstraintSystem::new(fin_w.validators.len());
        let wr_cs = wr::WithdrawalRootConstraintSystem::new(wr_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&tf_trace, &tf_cs),
            (&rs_trace, &rs_cs),
            (&bh_trace, &bh_cs),
            (&mb_trace, &mb_cs),
            (&bbh_trace, &bbh_cs),
            (&bst_trace, &bst_cs),
            (&bfp_trace, &bfp_cs),
            (&ffg_trace, &ffg_cs),
            (&fin_trace, &fin_cs),
            (&wr_trace, &wr_cs),
        ];
        assert_eq!(traces.len(), 10, "10-AIR composition");
        for link in &descriptors {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 10-AIR `joint_prove` + `joint_verify` round-trip
    /// with the 7 cross-layer descriptors. Expected to take several
    /// thousand seconds on BLS48-581 release (10 per-AIR proves + 7
    /// per-linkage SNARK proves + 7 cross-trace KZG opens).
    #[test]
    #[ignore = "slow: 10-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_round_trip() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let bh_w = build_bh_witness();
        let mb_w = build_mb_witness();
        let bbh_w = build_bbh_witness();
        let bst_w = build_bst_witness();
        let bfp_w = build_bfp_witness();
        let ffg_w = build_ffg_witness();
        let fin_w = build_fin_witness();
        let wr_w = build_wr_witness();

        let tf_trace = tf::build_trace_polynomials(&tf_w, curve);
        let rs_trace = rs::build_trace_polynomials(&rs_w, curve);
        let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
        let mb_trace = mb::build_trace_polynomials(&mb_w, curve);
        let bbh_trace = bbh::build_trace_polynomials(&bbh_w, curve);
        let bst_trace = bst::build_trace_polynomials(&bst_w, curve);
        let bfp_trace = bfp::build_trace_polynomials(&bfp_w, curve);
        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);
        let fin_trace = fc::build_finality_trace_polynomials(&fin_w, curve);
        let wr_trace = wr::build_trace_polynomials(&wr_w, curve);

        let tf_cs = tf::TxFullChainConstraintSystem::new(tf_trace.num_rows);
        let rs_cs = rs::ReceiptStatusConstraintSystem::new(rs_trace.num_rows);
        let bh_cs = bh::BlockHeaderConstraintSystem::new(bh_trace.num_rows);
        let mb_cs = mb::MultiBlockProofConstraintSystem::new(mb_trace.num_rows);
        let bbh_cs = bbh::BbhRootConsumerConstraintSystem::new(bbh_trace.num_rows);
        let bst_cs = bst::BeaconStateTransitionConstraintSystem::new(bst_trace.num_rows);
        let bfp_cs = bfp::BlockFullProofConstraintSystem::new(bfp_trace.num_rows);
        let ffg_cs = ffg::CasperFfgChainConstraintSystem::new(ffg_trace.num_rows);
        let fin_cs = fc::FinalityConstraintSystem::new(fin_w.validators.len());
        let wr_cs = wr::WithdrawalRootConstraintSystem::new(wr_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&tf_trace, &tf_cs),
            (&rs_trace, &rs_cs),
            (&bh_trace, &bh_cs),
            (&mb_trace, &mb_cs),
            (&bbh_trace, &bbh_cs),
            (&bst_trace, &bst_cs),
            (&bfp_trace, &bfp_cs),
            (&ffg_trace, &ffg_cs),
            (&fin_trace, &fin_cs),
            (&wr_trace, &wr_cs),
        ];
        let linkages = all_descriptors();

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 10-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 10);
        assert_eq!(ext.linkage_proofs.len(), 7);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest closures must match on descriptor {}", i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> = vec![
            &tf_cs, &rs_cs, &bh_cs, &mb_cs, &bbh_cs, &bst_cs, &bfp_cs,
            &ffg_cs, &fin_cs, &wr_cs,
        ];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 10-AIR joint_verify must accept",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Mutate one closure scalar so the verifier's closure-equality
    /// check rejects.
    #[test]
    #[ignore = "slow: depends on honest_round_trip setup"]
    fn tampered_closure_rejected() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let tf_w = build_tf_witness();
        let rs_w = build_rs_witness();
        let bh_w = build_bh_witness();
        let mb_w = build_mb_witness();
        let bbh_w = build_bbh_witness();
        let bst_w = build_bst_witness();
        let bfp_w = build_bfp_witness();
        let ffg_w = build_ffg_witness();
        let fin_w = build_fin_witness();
        let wr_w = build_wr_witness();

        let tf_trace = tf::build_trace_polynomials(&tf_w, curve);
        let rs_trace = rs::build_trace_polynomials(&rs_w, curve);
        let bh_trace = bh::build_trace_polynomials(&bh_w, curve);
        let mb_trace = mb::build_trace_polynomials(&mb_w, curve);
        let bbh_trace = bbh::build_trace_polynomials(&bbh_w, curve);
        let bst_trace = bst::build_trace_polynomials(&bst_w, curve);
        let bfp_trace = bfp::build_trace_polynomials(&bfp_w, curve);
        let ffg_trace = ffg::build_trace_polynomials(&ffg_w, curve);
        let fin_trace = fc::build_finality_trace_polynomials(&fin_w, curve);
        let wr_trace = wr::build_trace_polynomials(&wr_w, curve);

        let tf_cs = tf::TxFullChainConstraintSystem::new(tf_trace.num_rows);
        let rs_cs = rs::ReceiptStatusConstraintSystem::new(rs_trace.num_rows);
        let bh_cs = bh::BlockHeaderConstraintSystem::new(bh_trace.num_rows);
        let mb_cs = mb::MultiBlockProofConstraintSystem::new(mb_trace.num_rows);
        let bbh_cs = bbh::BbhRootConsumerConstraintSystem::new(bbh_trace.num_rows);
        let bst_cs = bst::BeaconStateTransitionConstraintSystem::new(bst_trace.num_rows);
        let bfp_cs = bfp::BlockFullProofConstraintSystem::new(bfp_trace.num_rows);
        let ffg_cs = ffg::CasperFfgChainConstraintSystem::new(ffg_trace.num_rows);
        let fin_cs = fc::FinalityConstraintSystem::new(fin_w.validators.len());
        let wr_cs = wr::WithdrawalRootConstraintSystem::new(wr_trace.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&tf_trace, &tf_cs),
            (&rs_trace, &rs_cs),
            (&bh_trace, &bh_cs),
            (&mb_trace, &mb_cs),
            (&bbh_trace, &bbh_cs),
            (&bst_trace, &bst_cs),
            (&bfp_trace, &bfp_cs),
            (&ffg_trace, &ffg_cs),
            (&fin_trace, &fin_cs),
            (&wr_trace, &wr_cs),
        ];
        let linkages = all_descriptors();

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed");

        // Tamper D6 (FFG vote_count ↔ finality running_total).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[6].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> = vec![
            &tf_cs, &rs_cs, &bh_cs, &mb_cs, &bbh_cs, &bst_cs, &bfp_cs,
            &ffg_cs, &fin_cs, &wr_cs,
        ];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject a tampered D6 closure",
        );
    }
}
