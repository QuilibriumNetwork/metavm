//! Cross-layer LogUp descriptors for the master composer.
//!
//! Wires the five top-level bindings that thread Layer A (EVM /
//! transaction execution) → Layer B (block header) → Layer C (beacon
//! execution payload + state transition) → Layer D (Casper FFG +
//! finality) through the cross-AIR LogUp protocol.
//!
//! Each descriptor uses the **single-column tuple** convention compatible
//! with [`crate::cross_air_logup::joint_prove`]: one column per side,
//! both gated by the natural `is_real` / phase selector. Hash bindings
//! pin the **first byte of the hash window** as the representative
//! tuple column; the byte-wise multiset closure of the full 32-byte
//! window is left to dedicated MPT / Keccak chain descriptors that
//! these single-column linkages anchor against.
//!
//! ## The five bindings
//!
//! | # | Binding | A side | B side |
//! |---|---------|--------|--------|
//! | 1 | A→B   | `tx_full_chain.COL_TX_HASH_OFFSET`             | `block_header.COL_TRANSACTIONS_ROOT_OFFSET`           |
//! | 2 | B→C   | `block_header.COL_BLOCK_HASH_OFFSET`           | `execution_payload_pair.COL_CLAIMED_BLOCK_HASH_OFFSET` |
//! | 3 | C→C+  | `bbh_root_consumer.COL_STATE_ROOT_OFFSET`      | `beacon_state_transition.COL_POST_STATE_ROOT_OFFSET`  |
//! | 4 | C→D   | `beacon_state_transition.COL_EPOCH`            | `casper_ffg.COL_TARGET_EPOCH`                         |
//! | 5 | D     | `casper_ffg.COL_VOTE_COUNT`                    | `finality.COL_RUNNING_TOTAL`                          |
//!
//! ## Scope
//!
//! * **In scope (this module)**: descriptor builders parameterised by
//!   layer index, with stable labels, gating selectors, and a
//!   builder bundling all five into a single `Vec` for the master
//!   composer.
//! * **Out of scope**: the algebraic bytewise closure of hash
//!   bindings (#1, #2, #3) — those compose with the existing MPT /
//!   Keccak / SSZ pair AIRs whose dedicated descriptors close the
//!   full 32-tuple. The single-column anchors here merely declare
//!   the cross-layer relationship the master composer must enforce.
//! * **Out of scope**: epoch-boundary gating for #4 (left to the
//!   beacon state transition AIR's `COL_IS_EPOCH_BOUNDARY` once a
//!   dedicated finalised-epoch selector lands on the FFG side).

use crate::cross_air_logup::CrossAirLogUpDescriptor;

use crate::bbh_root_consumer_air as bbh;
use crate::beacon_state_transition_air as bst;
use crate::block_header_air as bh;
use crate::casper_ffg_chain_air as ffg;
use crate::execution_payload_pair_air as epp;
use crate::finality_constraints as fc;
use crate::tx_full_chain_air as tx;

/// Stable label root: every descriptor's label is `format!("{LABEL_ROOT}_<n>_v1")`.
const LABEL_ROOT: &str = "cross_layer";

// ─── Builders ─────────────────────────────────────────────────────────

/// **Binding #1 — Layer A → Layer B.**
///
/// Anchors `tx_full_chain.COL_TX_HASH_OFFSET[0]` (the first byte of the
/// per-transaction hash) to `block_header.COL_TRANSACTIONS_ROOT_OFFSET[0]`
/// (the first byte of the block's `transactionsRoot`). The full
/// algebraic binding (`tx_hash ∈ tx MPT(tx_root)`) is closed by the
/// dedicated tx MPT inclusion descriptors layered on top of this
/// anchor.
pub fn tx_hash_to_block_tx_root_descriptor(
    tx_full_chain_layer_index: usize,
    block_header_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: format!("{LABEL_ROOT}_1_tx_hash_to_block_tx_root_v1"),
        a_layer_index: tx_full_chain_layer_index,
        a_columns: vec![tx::COL_TX_HASH_OFFSET],
        a_selector_column: Some(tx::COL_IS_REAL),
        b_layer_index: block_header_layer_index,
        b_columns: vec![bh::COL_TRANSACTIONS_ROOT_OFFSET],
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

/// **Binding #2 — Layer B → Layer C.**
///
/// Anchors `block_header.COL_BLOCK_HASH_OFFSET[0]` to the beacon
/// execution payload pair AIR's `COL_CLAIMED_BLOCK_HASH_OFFSET[0]`
/// (the bound block hash exposed by the payload merkleization).
pub fn block_hash_to_execution_payload_descriptor(
    block_header_layer_index: usize,
    execution_payload_pair_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: format!("{LABEL_ROOT}_2_block_hash_to_execution_payload_v1"),
        a_layer_index: block_header_layer_index,
        a_columns: vec![bh::COL_BLOCK_HASH_OFFSET],
        a_selector_column: Some(bh::COL_IS_REAL),
        b_layer_index: execution_payload_pair_layer_index,
        b_columns: vec![epp::COL_CLAIMED_BLOCK_HASH_OFFSET],
        b_selector_column: Some(epp::COL_IS_BLOCK_HASH_BOUND_AT),
    }
}

/// **Binding #3 — Layer C → Layer C+.**
///
/// Anchors `bbh_root_consumer.COL_STATE_ROOT_OFFSET[0]` to
/// `beacon_state_transition.COL_POST_STATE_ROOT_OFFSET[0]`, binding the
/// BeaconBlockHeader's exposed state root to the post-state-root
/// produced by the beacon state transition AIR.
pub fn bbh_state_root_to_state_transition_descriptor(
    bbh_root_consumer_layer_index: usize,
    beacon_state_transition_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: format!("{LABEL_ROOT}_3_bbh_state_root_to_state_transition_v1"),
        a_layer_index: bbh_root_consumer_layer_index,
        a_columns: vec![bbh::COL_STATE_ROOT_OFFSET],
        a_selector_column: Some(bbh::COL_IS_REAL),
        b_layer_index: beacon_state_transition_layer_index,
        b_columns: vec![bst::COL_POST_STATE_ROOT_OFFSET],
        b_selector_column: Some(bst::COL_IS_REAL),
    }
}

/// **Binding #4 — Layer C → Layer D.**
///
/// Anchors `beacon_state_transition.COL_EPOCH` (gated by
/// `COL_IS_EPOCH_BOUNDARY` so only epoch-boundary rows participate) to
/// `casper_ffg.COL_TARGET_EPOCH` (gated by `COL_IS_FINALIZED`, i.e.
/// the epoch that just finalised). On the boundary row the BST's
/// committed `EPOCH` must equal the finalising FFG row's target
/// epoch.
pub fn state_transition_epoch_to_casper_target_descriptor(
    beacon_state_transition_layer_index: usize,
    casper_ffg_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: format!("{LABEL_ROOT}_4_state_transition_epoch_to_casper_target_v1"),
        a_layer_index: beacon_state_transition_layer_index,
        a_columns: vec![bst::COL_EPOCH],
        a_selector_column: Some(bst::COL_IS_EPOCH_BOUNDARY),
        b_layer_index: casper_ffg_layer_index,
        b_columns: vec![ffg::COL_TARGET_EPOCH],
        b_selector_column: Some(ffg::COL_IS_FINALIZED),
    }
}

/// **Binding #5 — Layer D internal.**
///
/// Anchors `casper_ffg.COL_VOTE_COUNT` (gated by `COL_IS_FINALIZED`,
/// so only the finalising row's vote tally participates) to
/// `finality.COL_RUNNING_TOTAL` (gated by `COL_SEL_THRESHOLD`, the
/// row at which the running attestation balance crosses the 2/3
/// threshold). Both sides reference the same scalar — the total
/// effective balance backing the finalised checkpoint — so the
/// multiset closure pins the FFG vote count to the finality AIR's
/// terminal running total.
pub fn casper_vote_count_to_finality_running_total_descriptor(
    casper_ffg_layer_index: usize,
    finality_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: format!("{LABEL_ROOT}_5_casper_vote_count_to_finality_running_total_v1"),
        a_layer_index: casper_ffg_layer_index,
        a_columns: vec![ffg::COL_VOTE_COUNT],
        a_selector_column: Some(ffg::COL_IS_FINALIZED),
        b_layer_index: finality_layer_index,
        b_columns: vec![fc::COL_RUNNING_TOTAL],
        b_selector_column: Some(fc::COL_SEL_THRESHOLD),
    }
}

/// Per-AIR layer indices for the master-composer cross-layer bundle.
///
/// The five descriptors share seven layer references; this struct
/// gives a single point at which the master composer assigns each AIR
/// a layer index in its [`crate::layer_chain::LayerChainProof`].
#[derive(Debug, Clone, Copy)]
pub struct CrossLayerLayout {
    pub tx_full_chain: usize,
    pub block_header: usize,
    pub execution_payload_pair: usize,
    pub bbh_root_consumer: usize,
    pub beacon_state_transition: usize,
    pub casper_ffg: usize,
    pub finality: usize,
}

impl CrossLayerLayout {
    /// A canonical layer assignment: `tx_full_chain=0`,
    /// `block_header=1`, `execution_payload_pair=2`,
    /// `bbh_root_consumer=3`, `beacon_state_transition=4`,
    /// `casper_ffg=5`, `finality=6`.
    pub const CANONICAL: Self = Self {
        tx_full_chain: 0,
        block_header: 1,
        execution_payload_pair: 2,
        bbh_root_consumer: 3,
        beacon_state_transition: 4,
        casper_ffg: 5,
        finality: 6,
    };
}

/// Build all five cross-layer descriptors with the given layer
/// assignment. Order matches the binding-number table at the top of
/// this module.
pub fn build_all_cross_layer_descriptors(
    layout: CrossLayerLayout,
) -> Vec<CrossAirLogUpDescriptor> {
    vec![
        tx_hash_to_block_tx_root_descriptor(layout.tx_full_chain, layout.block_header),
        block_hash_to_execution_payload_descriptor(
            layout.block_header,
            layout.execution_payload_pair,
        ),
        bbh_state_root_to_state_transition_descriptor(
            layout.bbh_root_consumer,
            layout.beacon_state_transition,
        ),
        state_transition_epoch_to_casper_target_descriptor(
            layout.beacon_state_transition,
            layout.casper_ffg,
        ),
        casper_vote_count_to_finality_running_total_descriptor(
            layout.casper_ffg,
            layout.finality,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each descriptor must use a single-column tuple per side and a
    /// non-empty unique label.
    fn assert_well_formed_single_column(d: &CrossAirLogUpDescriptor) {
        assert!(!d.label.is_empty(), "label must not be empty");
        assert_eq!(d.a_columns.len(), 1, "single-column tuple required on A");
        assert_eq!(d.b_columns.len(), 1, "single-column tuple required on B");
        assert!(
            d.a_selector_column.is_some(),
            "A side must be gated by a selector column"
        );
        assert!(
            d.b_selector_column.is_some(),
            "B side must be gated by a selector column"
        );
        assert_ne!(
            d.a_layer_index, d.b_layer_index,
            "cross-layer descriptor must reference two distinct AIRs"
        );
    }

    /// Assert that a column index falls within the AIR's `NUM_COLUMNS`
    /// (catches typos / drift in column constants).
    fn assert_col_in_range(col: usize, num_cols: usize, what: &str) {
        assert!(
            col < num_cols,
            "{what}: col {col} out of range (num_cols = {num_cols})"
        );
    }

    #[test]
    fn binding_1_tx_hash_to_block_tx_root_well_formed() {
        let d = tx_hash_to_block_tx_root_descriptor(0, 1);
        assert_well_formed_single_column(&d);
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns[0], tx::COL_TX_HASH_OFFSET);
        assert_eq!(d.b_columns[0], bh::COL_TRANSACTIONS_ROOT_OFFSET);
        assert_eq!(d.a_selector_column, Some(tx::COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(bh::COL_IS_REAL));
        assert_col_in_range(d.a_columns[0], tx::NUM_COLUMNS, "tx_full_chain.tx_hash");
        assert_col_in_range(d.b_columns[0], bh::NUM_COLUMNS, "block_header.tx_root");
    }

    #[test]
    fn binding_2_block_hash_to_execution_payload_well_formed() {
        let d = block_hash_to_execution_payload_descriptor(1, 2);
        assert_well_formed_single_column(&d);
        assert_eq!(d.a_columns[0], bh::COL_BLOCK_HASH_OFFSET);
        assert_eq!(d.b_columns[0], epp::COL_CLAIMED_BLOCK_HASH_OFFSET);
        assert_eq!(d.a_selector_column, Some(bh::COL_IS_REAL));
        assert_eq!(
            d.b_selector_column,
            Some(epp::COL_IS_BLOCK_HASH_BOUND_AT)
        );
        assert_col_in_range(d.a_columns[0], bh::NUM_COLUMNS, "block_header.block_hash");
        // execution_payload_pair_air is a multi-section AIR; just sanity
        // check the column constant is sensible (> 0 and the selector is
        // a distinct, smaller column index than the start of the hash
        // window — both are encoded in the AIR's layout).
        assert!(d.b_columns[0] > 0);
    }

    #[test]
    fn binding_3_bbh_state_root_to_state_transition_well_formed() {
        let d = bbh_state_root_to_state_transition_descriptor(3, 4);
        assert_well_formed_single_column(&d);
        assert_eq!(d.a_columns[0], bbh::COL_STATE_ROOT_OFFSET);
        assert_eq!(d.b_columns[0], bst::COL_POST_STATE_ROOT_OFFSET);
        assert_eq!(d.a_selector_column, Some(bbh::COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(bst::COL_IS_REAL));
        assert_col_in_range(
            d.a_columns[0],
            bbh::NUM_COLUMNS,
            "bbh_root_consumer.state_root",
        );
        assert_col_in_range(
            d.b_columns[0],
            bst::NUM_COLUMNS,
            "beacon_state_transition.post_state_root",
        );
    }

    #[test]
    fn binding_4_state_transition_epoch_to_casper_target_well_formed() {
        let d = state_transition_epoch_to_casper_target_descriptor(4, 5);
        assert_well_formed_single_column(&d);
        assert_eq!(d.a_columns[0], bst::COL_EPOCH);
        assert_eq!(d.b_columns[0], ffg::COL_TARGET_EPOCH);
        assert_eq!(d.a_selector_column, Some(bst::COL_IS_EPOCH_BOUNDARY));
        assert_eq!(d.b_selector_column, Some(ffg::COL_IS_FINALIZED));
        assert_col_in_range(
            d.a_columns[0],
            bst::NUM_COLUMNS,
            "beacon_state_transition.epoch",
        );
        assert_col_in_range(d.b_columns[0], ffg::NUM_COLUMNS, "casper_ffg.target_epoch");
    }

    #[test]
    fn binding_5_casper_vote_count_to_finality_running_total_well_formed() {
        let d = casper_vote_count_to_finality_running_total_descriptor(5, 6);
        assert_well_formed_single_column(&d);
        assert_eq!(d.a_columns[0], ffg::COL_VOTE_COUNT);
        assert_eq!(d.b_columns[0], fc::COL_RUNNING_TOTAL);
        assert_eq!(d.a_selector_column, Some(ffg::COL_IS_FINALIZED));
        assert_eq!(d.b_selector_column, Some(fc::COL_SEL_THRESHOLD));
        assert_col_in_range(d.a_columns[0], ffg::NUM_COLUMNS, "casper_ffg.vote_count");
        assert_col_in_range(d.b_columns[0], fc::NUM_COLUMNS, "finality.running_total");
    }

    #[test]
    fn build_all_returns_five_descriptors_in_documented_order() {
        let ds = build_all_cross_layer_descriptors(CrossLayerLayout::CANONICAL);
        assert_eq!(ds.len(), 5);
        for d in &ds {
            assert_well_formed_single_column(d);
        }
        // Labels are ordered and unique.
        for (i, d) in ds.iter().enumerate() {
            let expected_prefix = format!("{LABEL_ROOT}_{}_", i + 1);
            assert!(
                d.label.starts_with(&expected_prefix),
                "descriptor {i} label `{}` does not start with `{expected_prefix}`",
                d.label
            );
        }
        let mut labels: Vec<&str> = ds.iter().map(|d| d.label.as_str()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), 5, "labels must be unique");
    }

    #[test]
    fn canonical_layout_assigns_seven_distinct_layers() {
        let l = CrossLayerLayout::CANONICAL;
        let layers = [
            l.tx_full_chain,
            l.block_header,
            l.execution_payload_pair,
            l.bbh_root_consumer,
            l.beacon_state_transition,
            l.casper_ffg,
            l.finality,
        ];
        let mut sorted = layers.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 7, "all 7 AIR layer indices must be distinct");
    }

    #[test]
    fn canonical_layer_assignment_matches_descriptors() {
        let l = CrossLayerLayout::CANONICAL;
        let ds = build_all_cross_layer_descriptors(l);
        // #1 A=tx_full_chain, B=block_header
        assert_eq!(ds[0].a_layer_index, l.tx_full_chain);
        assert_eq!(ds[0].b_layer_index, l.block_header);
        // #2 A=block_header, B=execution_payload_pair
        assert_eq!(ds[1].a_layer_index, l.block_header);
        assert_eq!(ds[1].b_layer_index, l.execution_payload_pair);
        // #3 A=bbh_root_consumer, B=beacon_state_transition
        assert_eq!(ds[2].a_layer_index, l.bbh_root_consumer);
        assert_eq!(ds[2].b_layer_index, l.beacon_state_transition);
        // #4 A=beacon_state_transition, B=casper_ffg
        assert_eq!(ds[3].a_layer_index, l.beacon_state_transition);
        assert_eq!(ds[3].b_layer_index, l.casper_ffg);
        // #5 A=casper_ffg, B=finality
        assert_eq!(ds[4].a_layer_index, l.casper_ffg);
        assert_eq!(ds[4].b_layer_index, l.finality);
    }
}
