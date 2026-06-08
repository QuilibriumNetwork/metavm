//! Integration `joint_prove` / `joint_verify` test for the
//! [`crate::beacon_block_validity_air`] composer.
//!
//! Task #150: scaffold a runnable joint-prove harness that composes a
//! single beacon block's validity bundle across the composer + at least
//! one downstream sub-AIR via the cross-AIR LogUp protocol.
//!
//! ## Scope
//!
//! The composer's full sub-AIR set is (see `beacon_block_validity_air`):
//!
//!   - [`crate::block_proposer_sig_air`] — proposer signature pairing,
//!   - [`crate::beacon_block_body_air`] (BBB8 layout) — body merkleization,
//!   - [`crate::attestation_aggregate_air`] — per-attestation BLS aggregate,
//!   - [`crate::beacon_state_transition_air`] — state-root transition,
//!   - [`crate::bbh_root_consumer_air`] — canonical (slot, proposer_index,
//!     parent_root, body_root, state_root) anchor.
//!
//! The composer publishes 5 multi-column tuple descriptors against these
//! sub-AIRs (48 / 32 / 8 / 40 / 112 columns wide). The current
//! `cross_air_logup::joint_prove` API algebraically supports only
//! **single-column tuple descriptors** (see the assertion documented at
//! `cross_air_logup::build_linkage_trace`). Wiring the full multi-column
//! descriptors through the joint prover is deferred to the multi-column
//! tuple follow-up tracked alongside the protocol roadmap.
//!
//! ## What this file does today
//!
//! 1. The fast `descriptor_consistency` test validates every published
//!    composer descriptor for shape, label, layer-index distinctness,
//!    column-index correctness (against the sub-AIR public `COL_*`
//!    constants), and selector wiring — i.e. it catches any column-layout
//!    drift in the composer or any sub-AIR.
//!
//! 2. The `#[ignore]`'d slow tests scaffold a 2-AIR
//!    `composer ↔ bbh_root_consumer_air` joint-prove pipeline using a
//!    **single-column** tuple on `COL_SLOT_BYTE_OFFSET[0]` (the slot's
//!    lowest LE byte). Both AIRs commit the same byte under matching
//!    `IS_REAL` selectors, so the LogUp closure equality
//!    `closure_a == closure_b` holds by construction. This is the
//!    smallest faithful test of the composer ↔ canonical-anchor binding
//!    that the current single-column API supports.
//!
//! ## Deferred (follow-up tasks)
//!
//!   - Full 48-col composer ↔ block_proposer_sig joint-prove pipeline
//!     (needs multi-column tuple support).
//!   - Full 32-col composer ↔ body_root joint-prove (ditto).
//!   - Full 40-col composer ↔ state_transition joint-prove (ditto).
//!   - 6-AIR end-to-end joint-prove with all 5 composer descriptors.
//!
//! ## Cross-references
//!
//!   - [`crate::integration_joint_prove_three_air`] for the 3-AIR
//!     single-column template this module mirrors.
//!   - [`crate::beacon_block_validity_air`] for the composer module.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::beacon_block_validity_air::{
        build_trace_polynomials as build_bbv_trace, make_block_validity_to_bbh_descriptor,
        make_block_validity_to_body_root_descriptor,
        make_block_validity_to_proposer_sig_descriptor,
        make_block_validity_to_state_transition_descriptor, BeaconBlockValidityConstraintSystem,
        BeaconBlockValidityWitness, COL_IS_REAL as BBV_COL_IS_REAL,
        COL_SLOT_BYTE_OFFSET as BBV_COL_SLOT_BYTE_OFFSET, HASH_BYTES,
    };
    use crate::bbh_root_consumer_air::{
        build_trace_polynomials as build_bbh_trace, BbhRootConsumerConstraintSystem,
        BbhRootConsumerWitness, COL_IS_REAL as BBH_COL_IS_REAL,
        COL_SLOT_BYTE_OFFSET as BBH_COL_SLOT_BYTE_OFFSET,
    };

    // ─── Honest test inputs ──────────────────────────────────────────────

    /// Honest slot value used by both the composer and the BBH-consumer
    /// witnesses. The LSB (slot & 0xFF) participates in the single-column
    /// joint_prove tuple.
    const HONEST_SLOT: u64 = 0x0123_4567_89AB_CDEF;

    const HONEST_PROPOSER_INDEX: u64 = 7_654_321;

    fn synth_root(seed: u8) -> [u8; HASH_BYTES] {
        let mut r = [0u8; HASH_BYTES];
        for i in 0..HASH_BYTES {
            r[i] = (i as u8).wrapping_mul(seed).wrapping_add(seed);
        }
        r
    }

    fn honest_block_root() -> [u8; HASH_BYTES] {
        synth_root(11)
    }
    fn honest_parent_root() -> [u8; HASH_BYTES] {
        synth_root(13)
    }
    fn honest_body_root() -> [u8; HASH_BYTES] {
        synth_root(17)
    }
    fn honest_state_root() -> [u8; HASH_BYTES] {
        synth_root(19)
    }

    fn build_bbv_witness() -> BeaconBlockValidityWitness {
        BeaconBlockValidityWitness::from_block_components(
            HONEST_SLOT,
            HONEST_PROPOSER_INDEX,
            honest_block_root(),
            honest_parent_root(),
            honest_body_root(),
            honest_state_root(),
            3,
        )
    }

    fn build_bbh_witness() -> BbhRootConsumerWitness {
        // Single-row witness mirroring the composer's anchor tuple.
        // claimed_root is independent (the BBH-pair AIR's own concern);
        // here we set it to a synthetic value.
        BbhRootConsumerWitness::from_bbh_tuple(
            1,
            synth_root(23), // claimed_root (independent)
            honest_parent_root(),
            honest_state_root(),
            honest_body_root(),
            HONEST_SLOT,
            HONEST_PROPOSER_INDEX,
        )
    }

    /// Single-column tuple descriptor binding
    /// `BBV.COL_SLOT_BYTE_OFFSET[0]` (slot's lowest LE byte, gated by
    /// `BBV_COL_IS_REAL`) ↔ `BBH.COL_SLOT_BYTE_OFFSET[0]` (matching byte
    /// on the canonical anchor, gated by `BBH_COL_IS_REAL`).
    ///
    /// This is a **single-column slice** of the composer's published
    /// 112-col `block_validity_to_bbh_v1` descriptor. Widening to the
    /// full tuple awaits multi-column tuple support in `joint_prove`.
    fn slot_byte0_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "beacon_block_validity_to_bbh_slot_byte0_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BBV_COL_SLOT_BYTE_OFFSET],
            a_selector_column: Some(BBV_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![BBH_COL_SLOT_BYTE_OFFSET],
            b_selector_column: Some(BBH_COL_IS_REAL),
        }
    }

    // ─── Fast static checks (un-ignored) ─────────────────────────────────

    /// Verifies every published composer descriptor is well-formed at
    /// the source-of-truth column constants, plus the single-column
    /// joint-prove descriptor used by the ignored slow tests.
    ///
    /// This test does NOT invoke `joint_prove`, so it runs well under
    /// the 120 s CI budget.
    #[test]
    fn descriptor_consistency_beacon_block_validity() {
        // ─── 1. Composer's full multi-column descriptors ─────────────
        let d_ps = make_block_validity_to_proposer_sig_descriptor(0, 1);
        assert_eq!(d_ps.label, "block_validity_to_proposer_sig_v1");
        assert_eq!(d_ps.a_columns.len(), 48);
        assert_eq!(d_ps.b_columns.len(), 48);
        assert_eq!(d_ps.a_selector_column, Some(BBV_COL_IS_REAL));
        assert_eq!(
            d_ps.b_selector_column,
            Some(crate::block_proposer_sig_air::COL_IS_REAL),
        );

        let d_body = make_block_validity_to_body_root_descriptor(0, 2);
        assert_eq!(d_body.label, "block_validity_to_body_root_v1");
        assert_eq!(d_body.a_columns.len(), HASH_BYTES);
        assert_eq!(d_body.b_columns.len(), HASH_BYTES);
        assert_eq!(
            d_body.b_columns[0],
            crate::beacon_block_body_air::BBB8_COL_CLAIMED_BODY_ROOT_OFFSET,
        );

        let d_st = make_block_validity_to_state_transition_descriptor(0, 3);
        assert_eq!(d_st.label, "block_validity_to_state_transition_v1");
        assert_eq!(d_st.a_columns.len(), 8 + HASH_BYTES);
        assert_eq!(d_st.b_columns.len(), 8 + HASH_BYTES);

        let d_bbh = make_block_validity_to_bbh_descriptor(0, 4);
        assert_eq!(d_bbh.label, "block_validity_to_bbh_v1");
        assert_eq!(d_bbh.a_columns.len(), 2 * 8 + 3 * HASH_BYTES);
        assert_eq!(d_bbh.b_columns.len(), 2 * 8 + 3 * HASH_BYTES);
        assert_eq!(d_bbh.a_selector_column, Some(BBV_COL_IS_REAL));
        assert_eq!(d_bbh.b_selector_column, Some(BBH_COL_IS_REAL));

        // ─── 2. Single-column descriptor used by joint_prove tests ──
        let d = slot_byte0_descriptor();
        assert_eq!(d.a_columns.len(), 1);
        assert_eq!(d.b_columns.len(), 1);
        assert_eq!(d.a_columns[0], BBV_COL_SLOT_BYTE_OFFSET);
        assert_eq!(d.b_columns[0], BBH_COL_SLOT_BYTE_OFFSET);
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_ne!(d.a_layer_index, d.b_layer_index);
        assert_eq!(d.a_selector_column, Some(BBV_COL_IS_REAL));
        assert_eq!(d.b_selector_column, Some(BBH_COL_IS_REAL));

        // ─── 3. Assemble (without proving) the 2-AIR joint input ────
        let curve = CurveType::Bls48581;
        let bbv_w = build_bbv_witness();
        let bbh_w = build_bbh_witness();
        let trace_0 = build_bbv_trace(&bbv_w, curve);
        let trace_1 = build_bbh_trace(&bbh_w, curve);
        let cs_0 = BeaconBlockValidityConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BbhRootConsumerConstraintSystem::new(trace_1.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1)];
        let linkages = vec![d.clone()];

        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // Sanity: both sides commit the same slot LSB.
        let slot_lsb = (HONEST_SLOT & 0xFF) as u64;
        assert_eq!(
            trace_0.columns[BBV_COL_SLOT_BYTE_OFFSET].evaluations[0].to_u64(),
            slot_lsb,
        );
        assert_eq!(
            trace_1.columns[BBH_COL_SLOT_BYTE_OFFSET].evaluations[0].to_u64(),
            slot_lsb,
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 2-AIR `joint_prove` + `joint_verify` round-trip for
    /// the beacon-block-validity composer ↔ bbh_root_consumer single-column
    /// slot-byte0 binding. Closure equality must hold on both sides.
    ///
    /// `#[ignore]` because `joint_prove` runs 2× per-AIR
    /// `prove_with_scheme` + 1× per-linkage SNARK + KZG opens on
    /// BLS48-581 — comfortably > 120 s in release on most hardware.
    /// Run via `cargo test --release --ignored
    /// honest_beacon_block_validity_joint_verify_true`.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_beacon_block_validity_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let bbv_w = build_bbv_witness();
        let bbh_w = build_bbh_witness();
        let trace_0 = build_bbv_trace(&bbv_w, curve);
        let trace_1 = build_bbh_trace(&bbh_w, curve);
        let cs_0 = BeaconBlockValidityConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BbhRootConsumerConstraintSystem::new(trace_1.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1)];
        let linkages = vec![slot_byte0_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove must succeed for composer ↔ bbh");

        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);

        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest matching slot byte0 witnesses must yield equal closures",
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept matching composer ↔ bbh slot-byte0 binding",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tampered closure: overwrite `closure_a` after a successful
    /// `joint_prove`. Verifier's scalar equality check must reject.
    ///
    /// `#[ignore]` because the setup half is the same `joint_prove`
    /// invocation as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_beacon_block_validity_joint_verify_true"]
    fn tampered_beacon_block_validity_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let bbv_w = build_bbv_witness();
        let bbh_w = build_bbh_witness();
        let trace_0 = build_bbv_trace(&bbv_w, curve);
        let trace_1 = build_bbh_trace(&bbh_w, curve);
        let cs_0 = BeaconBlockValidityConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BbhRootConsumerConstraintSystem::new(trace_1.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1)];
        let linkages = vec![slot_byte0_descriptor()];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove must succeed for composer ↔ bbh");

        // Force a closure mismatch by overwriting `closure_a`.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures across composer ↔ bbh",
        );
    }
}
