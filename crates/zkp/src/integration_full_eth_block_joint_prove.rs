//! Task #284 — Full Ethereum block per-block validity 7-AIR
//! `joint_prove` / `joint_verify` integration scaffold.
//!
//! Extends [`crate::integration_block_validity_master_joint_prove`]
//! (#251) from 1 active tx to **3 transactions** by composing one
//! [`crate::block_header_air`] hub against **three**
//! [`crate::tx_full_chain_air`] instances (one per tx), one shared
//! [`crate::receipt_status_air`] (3 rows, cross-row cumulative-gas
//! chain), and one shared [`crate::withdrawal_root_air`] (2 rows, 2
//! Shanghai withdrawals).  The 5th root commitment (state_root) is
//! bound **host-side** via [`crate::state_root_transition`]'s
//! `verify_state_transition` oracle on a pre/post account inclusion
//! pair anchored to `header.state_root` (no AIR descriptor — the
//! oracle is a host-side host-trie inclusion check around the same
//! state_root scalar that is written into `block_header_air`).
//!
//! ## AIR layer layout (6 traces)
//!
//!   - Layer 0: `block_header_air` — single executable header row
//!     carrying `state_root`, `transactions_root`, `receipts_root`,
//!     `withdrawals_root`, `number`, `timestamp`, …
//!   - Layer 1: `tx_full_chain_air` for `tx[0]` (1 row).
//!   - Layer 2: `tx_full_chain_air` for `tx[1]` (1 row).
//!   - Layer 3: `tx_full_chain_air` for `tx[2]` (1 row).
//!   - Layer 4: `receipt_status_air` (3 rows; the per-row shifted
//!     constraint `prev[i+1] = cumulative[i]` algebraically enforces
//!     the cumulative-gas chain).
//!   - Layer 5: `withdrawal_root_air` (2 rows; `IS_FIRST` gates row 0).
//!
//! ## Cross-AIR LogUp descriptors (6 single-column tuples → 5 roots)
//!
//! All descriptors are **single-column tuples** — the only shape
//! `cross_air_logup::joint_prove` accepts. The hub layer is 0
//! (`block_header_air`).
//!
//!   * **D0 / D1 / D2** (transactions_root distribution): bind
//!     `block_header_air.COL_TRANSACTIONS_ROOT_OFFSET + i` (byte `i`
//!     of `transactions_root`, gated by `IS_REAL`) ↔
//!     `tx_full_chain_air[i].COL_TX_HASH_OFFSET` (byte 0 of that
//!     tx's hash, gated by `IS_REAL`).  Synthetic anchor:
//!     `header.transactions_root[i] = tx[i].hash()[0]`.  Distributes
//!     the **transactions_root** commitment across the 3 per-tx
//!     composer witnesses.
//!
//!   * **D3** (receipts_root): bind
//!     `block_header_air.COL_RECEIPTS_ROOT_OFFSET` (byte 0, gated by
//!     `IS_REAL`) ↔ `receipt_status_air.COL_CUMULATIVE_GAS` (first
//!     row, gated by `IS_FIRST`).  Honest: `header.receipts_root[0] =
//!     receipt[0].cumulative_gas_used` (with `cumulative_gas_used ≤
//!     255` so the byte equals the full u64 column).
//!
//!   * **D4** (withdrawals_root): bind
//!     `block_header_air.COL_WITHDRAWALS_ROOT_OFFSET` (byte 0, gated
//!     by `IS_REAL`) ↔ `withdrawal_root_air.COL_WITHDRAWALS_ROOT_OFFSET`
//!     (byte 0, gated by `IS_FIRST`).  Honest: `header.withdrawals_root
//!     = withdrawals_root_air.witness.withdrawals_root`.
//!
//!   * **D5** (state_root → host-side, NOT a joint_prove descriptor):
//!     bound via [`crate::state_root_transition::verify_state_transition`]
//!     against the same `header.state_root` scalar that
//!     `block_header_air` commits in `COL_STATE_ROOT_OFFSET..+32`.
//!     The single-leaf pre/post tries built host-side make the oracle
//!     reject any state_root drift.  Documented in the fast test as a
//!     **5th root** check on top of the 4 algebraic descriptors.
//!
//! ## Witness shape
//!
//!   - 1 Ethereum block, 3 transactions, 2 withdrawals.
//!   - `block_header_air`: 1 row.
//!   - 3 × `tx_full_chain_air`: 1 row each.
//!   - `receipt_status_air`: 3 rows (one per tx; D3 gates row 0 via
//!     `IS_FIRST`).
//!   - `withdrawal_root_air`: 2 rows (D4 gates row 0 via `IS_FIRST`).
//!
//! ## Cross-references
//!
//!   - [`crate::integration_block_validity_master_joint_prove`] —
//!     #251 1-tx ancestor; same hub layer + single-column tuple
//!     protocol.
//!   - [`crate::integration_multi_tx_full_chain_joint_prove`] —
//!     existing 3 × `tx_full_chain_air` harness; this module shares
//!     the per-tx `TxFullChainWitness::from_transaction` pattern.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::account::{empty_code_hash, empty_storage_root, Account};
    use crate::block_header::BlockHeader;
    use crate::block_header_air::{
        build_trace_polynomials as build_bh_trace, from_block_header,
        BlockHeaderConstraintSystem, BlockHeaderWitness,
        COL_IS_REAL as BH_COL_IS_REAL,
        COL_RECEIPTS_ROOT_OFFSET as BH_COL_RECEIPTS_ROOT_OFFSET,
        COL_STATE_ROOT_OFFSET as BH_COL_STATE_ROOT_OFFSET,
        COL_TRANSACTIONS_ROOT_OFFSET as BH_COL_TRANSACTIONS_ROOT_OFFSET,
        COL_WITHDRAWALS_ROOT_OFFSET as BH_COL_WITHDRAWALS_ROOT_OFFSET,
    };
    use crate::receipt::{Receipt, ReceiptType};
    use crate::receipt_status_air::{
        build_trace_polynomials as build_rs_trace, ReceiptStatusConstraintSystem,
        ReceiptStatusWitness, COL_CUMULATIVE_GAS as RS_COL_CUMULATIVE_GAS,
        COL_IS_FIRST as RS_COL_IS_FIRST,
    };
    use crate::state_root_transition::{verify_state_transition, AccountInclusion};
    use crate::transaction::{LegacyTx, Transaction};
    use crate::tx_full_chain_air::{
        build_trace_polynomials as build_tf_trace, TxFullChainConstraintSystem,
        TxFullChainWitness, ADDR_LEN, COL_IS_REAL as TF_COL_IS_REAL,
        COL_TX_HASH_OFFSET as TF_COL_TX_HASH_OFFSET,
    };
    use crate::withdrawal::Withdrawal;
    use crate::withdrawal_root_air::{
        build_trace_polynomials as build_wr_trace, WithdrawalRootConstraintSystem,
        WithdrawalRootWitness, COL_IS_FIRST as WR_COL_IS_FIRST,
        COL_WITHDRAWALS_ROOT_OFFSET as WR_COL_WITHDRAWALS_ROOT_OFFSET,
    };

    // ─── Honest witness scalars ──────────────────────────────────────────

    /// Receipt cumulative-gas chain. Row 0 ≤ 255 so the
    /// receipts_root[0] byte in `block_header_air` matches the full u64
    /// scalar in `receipt_status_air.COL_CUMULATIVE_GAS` on D3.
    const HONEST_CUMUL: [u64; 3] = [50, 100, 150];
    /// Honest per-tx gas_used so that cumulative[i] - cumulative[i-1] =
    /// gas_used[i]. The `receipt_status_air` cross-row shifted
    /// constraint pins this chain on rows 1 and 2.
    const HONEST_GAS_USED: [u64; 3] = [50, 50, 50];

    const HONEST_SENDER: [u8; ADDR_LEN] = [0xCAu8; ADDR_LEN];

    // ─── Honest transactions ────────────────────────────────────────────

    fn tx_for(nonce: u64, to_byte: u8) -> Transaction {
        Transaction::Legacy(LegacyTx {
            nonce,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([to_byte; 20]),
            value: [0u8; 32],
            data: Vec::new(),
            v: 27,
            r: [(0x10u8 + to_byte) & 0x7f; 32],
            s: [(0x20u8 + to_byte) & 0x7f; 32],
        })
    }

    fn txs() -> [Transaction; 3] {
        [tx_for(0, 0x42), tx_for(1, 0x43), tx_for(2, 0x44)]
    }

    fn honest_withdrawals() -> [Withdrawal; 2] {
        [
            Withdrawal { index: 0, validator_index: 7, address: [0x77u8; 20], amount: 1_000 },
            Withdrawal { index: 1, validator_index: 8, address: [0x88u8; 20], amount: 2_000 },
        ]
    }

    fn honest_receipts() -> [Receipt; 3] {
        [
            Receipt {
                ty: ReceiptType::Legacy, status: 1,
                cumulative_gas_used: HONEST_CUMUL[0],
                logs_bloom: [0u8; 256], logs: Vec::new(),
            },
            Receipt {
                ty: ReceiptType::Legacy, status: 1,
                cumulative_gas_used: HONEST_CUMUL[1],
                logs_bloom: [0u8; 256], logs: Vec::new(),
            },
            Receipt {
                ty: ReceiptType::Legacy, status: 1,
                cumulative_gas_used: HONEST_CUMUL[2],
                logs_bloom: [0u8; 256], logs: Vec::new(),
            },
        ]
    }

    // ─── Host-side state_root oracle (5th root commitment, D5) ──────────

    /// Build a single-leaf account-state trie for an honest account
    /// at address `addr`. Returns the `(inclusion, root)` pair for use
    /// with [`verify_state_transition`].
    fn make_state_inclusion(addr: [u8; 20], nonce: u64) -> (AccountInclusion, [u8; 32]) {
        let account = Account {
            nonce,
            balance: [0u8; 32],
            storage_root: empty_storage_root(),
            code_hash: empty_code_hash(),
        };
        let trie_key = crate::account::account_trie_key(&addr);
        let value = crate::account::account_rlp(&account);
        let (root, proof) = crate::mpt::single_leaf_trie(&trie_key, &value);
        (AccountInclusion { address: addr, account, proof }, root)
    }

    /// Honest pre/post-state inclusion pair around the synthesised
    /// block header. Returns `(pre_root, post_root, pre_incl, post_incl)`.
    /// The host-side oracle binds the 5th root (state_root) by
    /// verifying both inclusions against the same scalar that
    /// `block_header_air` commits in `state_root[0..32]`.
    fn honest_state_pair() -> ([u8; 32], [u8; 32], AccountInclusion, AccountInclusion) {
        let (pre_incl, pre_root) = make_state_inclusion(HONEST_SENDER, 0);
        let (post_incl, post_root) = make_state_inclusion(HONEST_SENDER, 3);
        (pre_root, post_root, pre_incl, post_incl)
    }

    // ─── Witness builders ──────────────────────────────────────────────

    /// Header whose four root columns are wired to match the
    /// downstream AIR witnesses on the 4 algebraic descriptors and
    /// whose `state_root` is the honest **post**-state root (5th root
    /// commitment, bound host-side via `verify_state_transition`).
    fn synth_block_header() -> BlockHeader {
        let txs = txs();
        let mut transactions_root = [0u8; 32];
        // D0/D1/D2: transactions_root[i] = tx[i].hash()[0].
        transactions_root[0] = txs[0].hash()[0];
        transactions_root[1] = txs[1].hash()[0];
        transactions_root[2] = txs[2].hash()[0];

        let mut receipts_root = [0u8; 32];
        receipts_root[0] = HONEST_CUMUL[0] as u8;

        let withdrawals = honest_withdrawals();
        let wr_root = crate::withdrawal_root_air::compute_withdrawals_root(&withdrawals);

        let (_, post_root, _, _) = honest_state_pair();

        let mut h = BlockHeader::default();
        h.number = 100;
        h.timestamp = 1_700_000_000;
        h.gas_limit = 30_000_000;
        h.gas_used = HONEST_CUMUL[2];
        h.parent_hash = [0xAAu8; 32];
        h.state_root = post_root;
        h.transactions_root = transactions_root;
        h.receipts_root = receipts_root;
        h.withdrawals_root = Some(wr_root);
        h
    }

    fn build_bh_witness() -> BlockHeaderWitness {
        let row = from_block_header(&synth_block_header());
        BlockHeaderWitness::from_headers(vec![row])
    }

    fn build_tf_witness(i: usize) -> TxFullChainWitness {
        let txs = txs();
        TxFullChainWitness::from_transaction(
            i as u64,
            &txs[i],
            HONEST_SENDER,
            HONEST_GAS_USED[i],
            HONEST_CUMUL[i],
            1,
        )
    }

    fn build_rs_witness() -> ReceiptStatusWitness {
        ReceiptStatusWitness::from_receipts(&honest_receipts())
    }

    fn build_wr_witness() -> WithdrawalRootWitness {
        WithdrawalRootWitness::from_withdrawals(&honest_withdrawals())
    }

    // ─── Descriptor builders ──────────────────────────────────────────

    /// `block_header_air.TRANSACTIONS_ROOT[byte_idx]` ↔
    /// `tx_full_chain_air[layer_idx].TX_HASH[0]`.
    fn descriptor_bh_to_tx(byte_idx: usize, layer_idx: usize, label: &str) -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: label.into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_TRANSACTIONS_ROOT_OFFSET + byte_idx],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: layer_idx,
            b_columns: vec![TF_COL_TX_HASH_OFFSET],
            b_selector_column: Some(TF_COL_IS_REAL),
        }
    }

    fn descriptor_bh_to_tx0() -> CrossAirLogUpDescriptor {
        descriptor_bh_to_tx(0, 1, "full_eth_block_bh_to_tx0_v1")
    }
    fn descriptor_bh_to_tx1() -> CrossAirLogUpDescriptor {
        descriptor_bh_to_tx(1, 2, "full_eth_block_bh_to_tx1_v1")
    }
    fn descriptor_bh_to_tx2() -> CrossAirLogUpDescriptor {
        descriptor_bh_to_tx(2, 3, "full_eth_block_bh_to_tx2_v1")
    }

    /// D3: `block_header_air.RECEIPTS_ROOT[0]` ↔
    /// `receipt_status_air.CUMULATIVE_GAS` (first row, `IS_FIRST`).
    fn descriptor_bh_to_rs() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "full_eth_block_bh_to_rs_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_RECEIPTS_ROOT_OFFSET],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 4,
            b_columns: vec![RS_COL_CUMULATIVE_GAS],
            b_selector_column: Some(RS_COL_IS_FIRST),
        }
    }

    /// D4: `block_header_air.WITHDRAWALS_ROOT[0]` ↔
    /// `withdrawal_root_air.WITHDRAWALS_ROOT[0]` (first row, `IS_FIRST`).
    fn descriptor_bh_to_wr() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "full_eth_block_bh_to_wr_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BH_COL_WITHDRAWALS_ROOT_OFFSET],
            a_selector_column: Some(BH_COL_IS_REAL),
            b_layer_index: 5,
            b_columns: vec![WR_COL_WITHDRAWALS_ROOT_OFFSET],
            b_selector_column: Some(WR_COL_IS_FIRST),
        }
    }

    fn all_descriptors() -> Vec<CrossAirLogUpDescriptor> {
        vec![
            descriptor_bh_to_tx0(),
            descriptor_bh_to_tx1(),
            descriptor_bh_to_tx2(),
            descriptor_bh_to_rs(),
            descriptor_bh_to_wr(),
        ]
    }

    // ─── Fast static check (un-ignored) ──────────────────────────────

    /// 6-trace shape + 5-descriptor sanity check, plus the host-side
    /// `state_root` 5th-root oracle pin via
    /// [`verify_state_transition`]. Validates every published
    /// descriptor's column constants, selectors, layer indices, and
    /// the honest single-column tuple equalities on the active rows.
    /// Does **not** call `joint_prove`, so runs in CI.
    #[test]
    fn descriptor_consistency_full_eth_block() {
        let descs = all_descriptors();
        assert_eq!(descs.len(), 5);

        // All descriptors are single-column tuples (joint_prove invariant).
        for d in &descs {
            assert_eq!(d.a_columns.len(), 1);
            assert_eq!(d.b_columns.len(), 1);
        }

        // Hub is layer 0; descriptors bind layer 0 ↔ {1, 2, 3, 4, 5}.
        assert_eq!(descs[0].b_layer_index, 1);
        assert_eq!(descs[1].b_layer_index, 2);
        assert_eq!(descs[2].b_layer_index, 3);
        assert_eq!(descs[3].b_layer_index, 4);
        assert_eq!(descs[4].b_layer_index, 5);
        for d in &descs {
            assert_eq!(d.a_layer_index, 0);
        }

        // Selectors: BH side IS_REAL; B-side per AIR.
        for d in &descs {
            assert_eq!(d.a_selector_column, Some(BH_COL_IS_REAL));
        }
        assert_eq!(descs[0].b_selector_column, Some(TF_COL_IS_REAL));
        assert_eq!(descs[1].b_selector_column, Some(TF_COL_IS_REAL));
        assert_eq!(descs[2].b_selector_column, Some(TF_COL_IS_REAL));
        assert_eq!(descs[3].b_selector_column, Some(RS_COL_IS_FIRST));
        assert_eq!(descs[4].b_selector_column, Some(WR_COL_IS_FIRST));

        // Column constants pinned to published COL_* (catches drift).
        assert_eq!(descs[0].a_columns[0], BH_COL_TRANSACTIONS_ROOT_OFFSET);
        assert_eq!(descs[1].a_columns[0], BH_COL_TRANSACTIONS_ROOT_OFFSET + 1);
        assert_eq!(descs[2].a_columns[0], BH_COL_TRANSACTIONS_ROOT_OFFSET + 2);
        assert_eq!(descs[3].a_columns[0], BH_COL_RECEIPTS_ROOT_OFFSET);
        assert_eq!(descs[4].a_columns[0], BH_COL_WITHDRAWALS_ROOT_OFFSET);
        assert_eq!(descs[0].b_columns[0], TF_COL_TX_HASH_OFFSET);
        assert_eq!(descs[1].b_columns[0], TF_COL_TX_HASH_OFFSET);
        assert_eq!(descs[2].b_columns[0], TF_COL_TX_HASH_OFFSET);
        assert_eq!(descs[3].b_columns[0], RS_COL_CUMULATIVE_GAS);
        assert_eq!(descs[4].b_columns[0], WR_COL_WITHDRAWALS_ROOT_OFFSET);

        // ─── Build all 6 per-AIR witnesses + traces at BLS48-581. ───
        let curve = CurveType::Bls48581;
        let bh_w = build_bh_witness();
        let tf0_w = build_tf_witness(0);
        let tf1_w = build_tf_witness(1);
        let tf2_w = build_tf_witness(2);
        let rs_w = build_rs_witness();
        let wr_w = build_wr_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tf0 = build_tf_trace(&tf0_w, curve);
        let trace_tf1 = build_tf_trace(&tf1_w, curve);
        let trace_tf2 = build_tf_trace(&tf2_w, curve);
        let trace_rs = build_rs_trace(&rs_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tf0 = TxFullChainConstraintSystem::new(trace_tf0.num_rows);
        let cs_tf1 = TxFullChainConstraintSystem::new(trace_tf1.num_rows);
        let cs_tf2 = TxFullChainConstraintSystem::new(trace_tf2.num_rows);
        let cs_rs = ReceiptStatusConstraintSystem::new(trace_rs.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tf0, &cs_tf0),
            (&trace_tf1, &cs_tf1),
            (&trace_tf2, &cs_tf2),
            (&trace_rs, &cs_rs),
            (&trace_wr, &cs_wr),
        ];

        assert_eq!(traces.len(), 6);
        for link in &descs {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // Witness shape: 3 txs, 2 withdrawals, 3 receipts.
        assert_eq!(tf0_w.rows.len(), 1);
        assert_eq!(tf1_w.rows.len(), 1);
        assert_eq!(tf2_w.rows.len(), 1);
        assert_eq!(rs_w.rows.len(), 3);
        assert_eq!(wr_w.rows.len(), 2);

        // ─── Honest single-column tuple equalities ─────────────────

        let tx_hashes = txs().map(|t| t.hash());

        // D0/D1/D2: header.transactions_root[i] == tx[i].hash()[0].
        for i in 0..3 {
            let tf_trace = [&trace_tf0, &trace_tf1, &trace_tf2][i];
            let expected = tx_hashes[i][0] as u64;
            assert_eq!(
                trace_bh.columns[BH_COL_TRANSACTIONS_ROOT_OFFSET + i].evaluations[0].to_u64(),
                expected,
                "header.transactions_root[{}] must equal tx[{}].hash()[0]",
                i, i,
            );
            assert_eq!(
                tf_trace.columns[TF_COL_TX_HASH_OFFSET].evaluations[0].to_u64(),
                expected,
                "tx_full_chain[{}].tx_hash[0] must equal tx[{}].hash()[0]",
                i, i,
            );
        }

        // D3: header.receipts_root[0] == receipt[0].cumulative_gas.
        assert_eq!(
            trace_bh.columns[BH_COL_RECEIPTS_ROOT_OFFSET].evaluations[0].to_u64(),
            HONEST_CUMUL[0],
        );
        assert_eq!(
            trace_rs.columns[RS_COL_CUMULATIVE_GAS].evaluations[0].to_u64(),
            HONEST_CUMUL[0],
        );
        // IS_FIRST=1 on row 0, 0 on rows 1, 2.
        assert_eq!(trace_rs.columns[RS_COL_IS_FIRST].evaluations[0].to_u64(), 1);
        assert_eq!(trace_rs.columns[RS_COL_IS_FIRST].evaluations[1].to_u64(), 0);
        assert_eq!(trace_rs.columns[RS_COL_IS_FIRST].evaluations[2].to_u64(), 0);

        // D4: header.withdrawals_root[0] == wr_witness.withdrawals_root[0].
        let wr_byte0 = wr_w.withdrawals_root[0] as u64;
        assert_eq!(
            trace_bh.columns[BH_COL_WITHDRAWALS_ROOT_OFFSET].evaluations[0].to_u64(),
            wr_byte0,
        );
        assert_eq!(
            trace_wr.columns[WR_COL_WITHDRAWALS_ROOT_OFFSET].evaluations[0].to_u64(),
            wr_byte0,
        );
        // IS_FIRST=1 on row 0, 0 on row 1.
        assert_eq!(trace_wr.columns[WR_COL_IS_FIRST].evaluations[0].to_u64(), 1);
        assert_eq!(trace_wr.columns[WR_COL_IS_FIRST].evaluations[1].to_u64(), 0);

        // ─── D5: state_root host-side oracle binding. ──────────────
        //
        // The 5th root commitment is bound host-side via
        // `verify_state_transition` on a single-leaf pre/post pair.
        // The `state_root` scalar that `block_header_air` commits in
        // `COL_STATE_ROOT_OFFSET..+32` is the *post* root of the
        // honest transition.  The oracle accepts (Ok(())); flipping
        // either inclusion's account or the root must reject.
        let (pre_root, post_root, pre_incl, post_incl) = honest_state_pair();
        verify_state_transition(pre_root, post_root, &[pre_incl.clone()], &[post_incl.clone()])
            .expect("honest state-root transition must verify");
        // The header's state_root scalar equals the honest post_root byte 0.
        assert_eq!(
            trace_bh.columns[BH_COL_STATE_ROOT_OFFSET].evaluations[0].to_u64(),
            post_root[0] as u64,
        );
        // A tampered post-state inclusion must reject (5th-root
        // soundness check, host-side).
        let mut bad_post = post_incl.clone();
        bad_post.account.nonce ^= 1;
        assert!(
            verify_state_transition(pre_root, post_root, &[pre_incl.clone()], &[bad_post]).is_err(),
            "tampered post-state nonce must reject the 5th-root oracle",
        );

        // Cumulative-gas chain visible in the receipt-status trace:
        // prev[i+1] = cumulative[i].
        for i in 0..2 {
            let cum_i = trace_rs.columns[RS_COL_CUMULATIVE_GAS].evaluations[i].to_u64();
            let prev_next = trace_rs.columns
                [crate::receipt_status_air::COL_PREV_CUMULATIVE_GAS]
                .evaluations[i + 1]
                .to_u64();
            assert_eq!(
                cum_i, prev_next,
                "receipt_status_air cross-row chain: prev[i+1] must equal cum[i]",
            );
        }
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Honest 6-AIR `joint_prove` + `joint_verify` round-trip with all
    /// 5 cross-AIR LogUp descriptors. Run via `cargo test --release
    /// --ignored honest_full_eth_block_joint_verify_true`.
    /// Expected runtime: 6 per-AIR proves + 5 per-linkage SNARK
    /// proves + 5 cross-trace KZG opens on BLS48-581 — comfortably
    /// > 120 s, so `#[ignore]`d.
    #[test]
    #[ignore = "slow: 6-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_full_eth_block_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Host-side 5th-root binding (state_root) precondition.
        let (pre_root, post_root, pre_incl, post_incl) = honest_state_pair();
        verify_state_transition(pre_root, post_root, &[pre_incl], &[post_incl])
            .expect("honest state_root transition must verify before joint_prove");

        let bh_w = build_bh_witness();
        let tf0_w = build_tf_witness(0);
        let tf1_w = build_tf_witness(1);
        let tf2_w = build_tf_witness(2);
        let rs_w = build_rs_witness();
        let wr_w = build_wr_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tf0 = build_tf_trace(&tf0_w, curve);
        let trace_tf1 = build_tf_trace(&tf1_w, curve);
        let trace_tf2 = build_tf_trace(&tf2_w, curve);
        let trace_rs = build_rs_trace(&rs_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tf0 = TxFullChainConstraintSystem::new(trace_tf0.num_rows);
        let cs_tf1 = TxFullChainConstraintSystem::new(trace_tf1.num_rows);
        let cs_tf2 = TxFullChainConstraintSystem::new(trace_tf2.num_rows);
        let cs_rs = ReceiptStatusConstraintSystem::new(trace_rs.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tf0, &cs_tf0),
            (&trace_tf1, &cs_tf1),
            (&trace_tf2, &cs_tf2),
            (&trace_rs, &cs_rs),
            (&trace_wr, &cs_wr),
        ];
        let linkages = all_descriptors();

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 6-AIR joint_prove must succeed");

        assert_eq!(proofs.len(), 6);
        assert_eq!(ext.linkage_proofs.len(), 5);

        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_bh, &cs_tf0, &cs_tf1, &cs_tf2, &cs_rs, &cs_wr];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 6-AIR joint_verify must accept matching tuples on all 5 descriptors",
        );
    }

    // ─── Slow tampered round-trip (ignored) ──────────────────────────

    /// Tamper `closure_a` on D4 (withdrawals-root linkage) after a
    /// successful `joint_prove`. `joint_verify` must reject.
    #[test]
    #[ignore = "slow: depends on honest_full_eth_block_joint_verify_true setup"]
    fn tampered_full_eth_block_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let bh_w = build_bh_witness();
        let tf0_w = build_tf_witness(0);
        let tf1_w = build_tf_witness(1);
        let tf2_w = build_tf_witness(2);
        let rs_w = build_rs_witness();
        let wr_w = build_wr_witness();

        let trace_bh = build_bh_trace(&bh_w, curve);
        let trace_tf0 = build_tf_trace(&tf0_w, curve);
        let trace_tf1 = build_tf_trace(&tf1_w, curve);
        let trace_tf2 = build_tf_trace(&tf2_w, curve);
        let trace_rs = build_rs_trace(&rs_w, curve);
        let trace_wr = build_wr_trace(&wr_w, curve);

        let cs_bh = BlockHeaderConstraintSystem::new(trace_bh.num_rows);
        let cs_tf0 = TxFullChainConstraintSystem::new(trace_tf0.num_rows);
        let cs_tf1 = TxFullChainConstraintSystem::new(trace_tf1.num_rows);
        let cs_tf2 = TxFullChainConstraintSystem::new(trace_tf2.num_rows);
        let cs_rs = ReceiptStatusConstraintSystem::new(trace_rs.num_rows);
        let cs_wr = WithdrawalRootConstraintSystem::new(trace_wr.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_bh, &cs_bh),
            (&trace_tf0, &cs_tf0),
            (&trace_tf1, &cs_tf1),
            (&trace_tf2, &cs_tf2),
            (&trace_rs, &cs_rs),
            (&trace_wr, &cs_wr),
        ];
        let linkages = all_descriptors();

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed on honest inputs");

        // Tamper descriptor index 4 (D4 — withdrawals-root) closure_a.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[4].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_bh, &cs_tf0, &cs_tf1, &cs_tf2, &cs_rs, &cs_wr];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched closures on D4 (withdrawals-root)",
        );
    }
}
