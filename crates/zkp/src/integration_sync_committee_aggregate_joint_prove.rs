//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! for the **`sync_committee_aggregate_composer_air` 4-phase composer**.
//!
//! Task #149: wire the composition AIR into a 5-layer chain together
//! with its four sub-AIRs:
//!
//!   - layer 0: [`crate::sync_committee_aggregate_composer_air`]
//!     (composer; 4 consecutive rows, one per phase).
//!   - layer 1: [`crate::sync_committee_filter_air`] (phase 0 sub-AIR).
//!   - layer 2: [`crate::hash_to_g2_air`]              (phase 1 sub-AIR).
//!   - layer 3: [`crate::sync_committee_sig_air`]      (phase 2 sub-AIR).
//!   - layer 4: [`crate::bls_pairing_air`]             (phase 3 sub-AIR).
//!
//! Four cross-AIR LogUp descriptors are wired, one per phase, each as
//! a **single-column tuple** (the `joint_prove` API currently asserts
//! `a_columns.len() == 1 && b_columns.len() == 1`):
//!
//! - D0 (phase 0 / filter):
//!     composer `aggregate_pubkey[0]` ↔ filter AIR `pubkey[0]`,
//!     gated by composer's filter-phase ↔ filter AIR's IS_REAL.
//! - D1 (phase 1 / hash_to_g2):
//!     composer `msg[0]` ↔ h2g2 AIR `msg[0]`, gated by composer's
//!     h2g2-phase ↔ h2g2 AIR's IS_REAL.
//! - D2 (phase 2 / agg-pubkey):
//!     composer `aggregate_pubkey[0]` ↔ sync-sig AIR
//!     `agg_pk_x_bytes[0]`, gated by composer's agg-pubkey phase ↔
//!     sync-sig AIR's IS_REAL.
//! - D3 (phase 3 / pairing):
//!     composer `aggregate_pubkey[0]` ↔ pairing AIR
//!     `pk_compressed[0]`, gated by composer's pairing-phase ↔
//!     pairing AIR's IS_REAL.
//!
//! Because the composer's own descriptors are multi-column shapes and
//! `joint_prove` currently consumes only single-column tuples, we build
//! single-column anchor projections here. The honest pipeline binds
//! these anchors across all four phases of the composer.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::bls_pairing_air::{
        build_trace_polynomials as build_pair_trace, BlsPairingConstraintSystem,
        BlsPairingWitness, COL_IS_REAL as PAIR_COL_IS_REAL,
        COL_PK_COMPRESSED_OFFSET as PAIR_COL_PK_COMPRESSED_OFFSET,
        NUM_COLUMNS as PAIR_NUM_COLUMNS,
    };
    use crate::hash_to_g2_air::{
        build_trace_polynomials as build_h2g2_trace, HashToG2ConstraintSystem, HashToG2Witness,
        COL_IS_REAL as H2G2_COL_IS_REAL, COL_MSG_OFFSET as H2G2_COL_MSG_OFFSET,
        NUM_COLUMNS as H2G2_NUM_COLUMNS,
    };
    use crate::sync_committee_aggregate_composer_air::{
        build_trace_polynomials as build_sca_trace,
        SyncCommitteeAggregateComposerConstraintSystem,
        SyncCommitteeAggregateComposerWitness,
        COL_AGG_PK_X_BYTE0_MIRROR as SCA_COL_AGG_PK_X_BYTE0_MIRROR,
        COL_AGG_PUBKEY_OFFSET as SCA_COL_AGG_PUBKEY_OFFSET,
        COL_FIRST_MEMBER_PK_BYTE0_MIRROR as SCA_COL_FIRST_MEMBER_PK_BYTE0_MIRROR,
        COL_IS_AGG_PUBKEY_PHASE as SCA_COL_IS_AGG_PUBKEY_PHASE,
        COL_IS_FILTER_PHASE as SCA_COL_IS_FILTER_PHASE,
        COL_IS_HASH_TO_G2_PHASE as SCA_COL_IS_HASH_TO_G2_PHASE,
        COL_IS_PAIRING_PHASE as SCA_COL_IS_PAIRING_PHASE,
        COL_MSG_OFFSET as SCA_COL_MSG_OFFSET, NUM_COLUMNS as SCA_NUM_COLUMNS,
        ROWS_PER_INSTANCE as SCA_ROWS_PER_INSTANCE,
    };
    use crate::sync_committee_filter_air::{
        build_trace_polynomials as build_filter_trace, SyncCommitteeFilterConstraintSystem,
        SyncCommitteeFilterWitness, COL_IS_REAL as FILTER_COL_IS_REAL,
        COL_PUBKEY_OFFSET as FILTER_COL_PUBKEY_OFFSET, NUM_COLUMNS as FILTER_NUM_COLUMNS,
    };
    use crate::sync_committee_sig_air::{
        build_trace_polynomials as build_ssig_trace, SyncCommitteeSigConstraintSystem,
        SyncCommitteeSigWitness, COL_AGG_PK_X_BYTES_OFFSET as SSIG_COL_AGG_PK_X_BYTES_OFFSET,
        COL_IS_REAL as SSIG_COL_IS_REAL, NUM_COLUMNS as SSIG_NUM_COLUMNS,
    };

    use crate::bls_sig as bs;

    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
    const HONEST_MSG: [u8; 32] = *b"sync-aggregate-joint-prove-test!";

    // ─── Single-column descriptor builders ────────────────────────────

    /// D0: composer `first_member_pk_byte0` mirror ↔ filter AIR
    /// `pubkey[0]`, gated by composer's filter-phase ↔ filter AIR's
    /// IS_REAL.
    ///
    /// Task #190: the composer commits the **aggregate** pubkey on
    /// `COL_AGG_PUBKEY_OFFSET`, while the filter AIR commits **per-
    /// member** pubkey bytes on `COL_PUBKEY_OFFSET`. Those values are
    /// disjoint, so the LogUp multiset-equality check would always
    /// reject. The composer's `COL_FIRST_MEMBER_PK_BYTE0_MIRROR`
    /// witness column holds byte 0 of the **first committee member's
    /// pubkey** — the same scalar the filter AIR commits on its
    /// row-0 `pubkey[0]` — so the single-column LogUp closure holds
    /// honestly. Soundness flows from the LogUp closure on the
    /// mirror: any prover that tampers with the mirror column breaks
    /// the descriptor's closure equality.
    fn d0_filter_pubkey_anchor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "sync_composer_to_filter_anchor_v1".into(),
            a_layer_index: 0,
            a_columns: vec![SCA_COL_FIRST_MEMBER_PK_BYTE0_MIRROR],
            a_selector_column: Some(SCA_COL_IS_FILTER_PHASE),
            b_layer_index: 1,
            b_columns: vec![FILTER_COL_PUBKEY_OFFSET],
            b_selector_column: Some(FILTER_COL_IS_REAL),
        }
    }

    /// D1: composer `msg[0]` ↔ h2g2 AIR `msg[0]`, gated by composer's
    /// hash_to_g2 phase ↔ h2g2's IS_REAL.
    fn d1_hash_to_g2_msg_anchor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "sync_composer_to_hash_to_g2_anchor_v1".into(),
            a_layer_index: 0,
            a_columns: vec![SCA_COL_MSG_OFFSET],
            a_selector_column: Some(SCA_COL_IS_HASH_TO_G2_PHASE),
            b_layer_index: 2,
            b_columns: vec![H2G2_COL_MSG_OFFSET],
            b_selector_column: Some(H2G2_COL_IS_REAL),
        }
    }

    /// D2: composer `agg_pk_x_byte0` mirror ↔ sync-sig AIR
    /// `agg_pk_x_bytes[0]`, gated by composer's agg-pubkey phase ↔
    /// sync-sig AIR's IS_REAL.
    ///
    /// Task #190: the composer's raw `aggregate_pubkey[0]` carries
    /// the IETF 3 flag bits in its top nibble; the sync-sig AIR
    /// stores byte 0 with those flag bits **masked off**
    /// (`& 0x1f`) so the canonical Fp byte form decodes cleanly.
    /// The two would be unequal for any real BLS pubkey, so we anchor
    /// the LogUp tuple on the composer's
    /// `COL_AGG_PK_X_BYTE0_MIRROR` mirror column, which holds the
    /// same flag-masked byte the sync-sig AIR commits. The closure
    /// pins the mirror algebraically.
    fn d2_sync_sig_pubkey_anchor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "sync_composer_to_sync_sig_anchor_v1".into(),
            a_layer_index: 0,
            a_columns: vec![SCA_COL_AGG_PK_X_BYTE0_MIRROR],
            a_selector_column: Some(SCA_COL_IS_AGG_PUBKEY_PHASE),
            b_layer_index: 3,
            b_columns: vec![SSIG_COL_AGG_PK_X_BYTES_OFFSET],
            b_selector_column: Some(SSIG_COL_IS_REAL),
        }
    }

    /// D3: composer `aggregate_pubkey[0]` ↔ pairing AIR
    /// `pk_compressed[0]`, gated by composer's pairing-phase ↔ pairing
    /// AIR's IS_REAL.
    fn d3_pairing_pubkey_anchor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "sync_composer_to_pairing_anchor_v1".into(),
            a_layer_index: 0,
            a_columns: vec![SCA_COL_AGG_PUBKEY_OFFSET],
            a_selector_column: Some(SCA_COL_IS_PAIRING_PHASE),
            b_layer_index: 4,
            b_columns: vec![PAIR_COL_PK_COMPRESSED_OFFSET],
            b_selector_column: Some(PAIR_COL_IS_REAL),
        }
    }

    // ─── Witness builders ─────────────────────────────────────────────

    /// Two-signer honest BLS aggregate over the same `(msg, DST)` pair.
    /// Returns (msg, pk0, pk1, agg_pubkey, agg_sig).
    fn honest_aggregate() -> (
        [u8; 32],
        bs::PublicKey,
        bs::PublicKey,
        bs::PublicKey,
        bs::Signature,
    ) {
        let msg = HONEST_MSG;
        let sk0 = bs::SecretKey::from_u8_seed(0xa1);
        let sk1 = bs::SecretKey::from_u8_seed(0xb2);
        let pk0 = sk0.public_key();
        let pk1 = sk1.public_key();
        let s0 = sk0.sign(&msg, POP_DST);
        let s1 = sk1.sign(&msg, POP_DST);
        let agg_sig = bs::aggregate_sigs(&[s0, s1]).expect("agg sig");
        let agg_pk = bs::aggregate_pubkeys(&[pk0.clone(), pk1.clone()]).expect("agg pk");
        (msg, pk0, pk1, agg_pk, agg_sig)
    }

    fn build_composer_witness() -> SyncCommitteeAggregateComposerWitness {
        let (msg, pk0, pk1, _, agg_sig) = honest_aggregate();
        SyncCommitteeAggregateComposerWitness::from_aggregate(
            msg,
            &[pk0.0, pk1.0],
            agg_sig.0,
            2,
        )
    }

    /// Tiny 2-row committee witness with both bits set (both members
    /// participate). The filter AIR's NUM_COLUMNS is 54 and the trace
    /// is auto-inflated by joint_prove to the LogUp range-table size
    /// (256) when running the slow tests.
    fn build_filter_witness() -> SyncCommitteeFilterWitness {
        let (_, pk0, pk1, _, _) = honest_aggregate();
        SyncCommitteeFilterWitness::from_committee(&[pk0.0, pk1.0], &[true, true])
    }

    fn build_h2g2_witness() -> HashToG2Witness {
        HashToG2Witness::from_message(&HONEST_MSG, POP_DST)
            .expect("h2g2 witness builds for honest msg")
    }

    fn build_sync_sig_witness() -> SyncCommitteeSigWitness {
        let (msg, pk0, pk1, _, agg_sig) = honest_aggregate();
        SyncCommitteeSigWitness::from_aggregate(&[pk0.0, pk1.0], msg, agg_sig.0)
            .expect("sync_sig witness builds for honest agg")
    }

    fn build_pairing_witness() -> BlsPairingWitness {
        let (msg, _, _, agg_pk, agg_sig) = honest_aggregate();
        BlsPairingWitness::from_decoded(agg_pk.0, agg_sig.0, msg)
            .expect("pairing witness decodes honest (agg_pk, agg_sig, msg)")
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check across all four descriptors plus a
    /// non-prove sanity check on the 5-trace shape that `joint_prove`
    /// receives. Does NOT run `joint_prove`, so it stays under 120s
    /// and runs in CI.
    #[test]
    fn descriptor_consistency_sync_committee_aggregate_composer() {
        let d0 = d0_filter_pubkey_anchor();
        let d1 = d1_hash_to_g2_msg_anchor();
        let d2 = d2_sync_sig_pubkey_anchor();
        let d3 = d3_pairing_pubkey_anchor();

        // Single-column-tuple invariant — joint_prove asserts this.
        for d in [&d0, &d1, &d2, &d3] {
            assert_eq!(d.a_columns.len(), 1, "joint_prove requires single-column tuples (a)");
            assert_eq!(d.b_columns.len(), 1, "joint_prove requires single-column tuples (b)");
            assert_ne!(d.a_layer_index, d.b_layer_index, "descriptor spans two layers");
            assert!(d.a_selector_column.is_some(), "A side must be gated");
            assert!(d.b_selector_column.is_some(), "B side must be gated");
        }

        // Layer indices: composer = 0; filter=1, h2g2=2, sig=3, pair=4.
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.b_layer_index, 2);
        assert_eq!(d2.b_layer_index, 3);
        assert_eq!(d3.b_layer_index, 4);

        // A side selectors: each descriptor uses a distinct phase
        // selector on the composer, exercising the phase-disjoint
        // multi-row-per-instance pattern.
        assert_eq!(d0.a_selector_column, Some(SCA_COL_IS_FILTER_PHASE));
        assert_eq!(d1.a_selector_column, Some(SCA_COL_IS_HASH_TO_G2_PHASE));
        assert_eq!(d2.a_selector_column, Some(SCA_COL_IS_AGG_PUBKEY_PHASE));
        assert_eq!(d3.a_selector_column, Some(SCA_COL_IS_PAIRING_PHASE));
        // B side selectors: each sub-AIR's IS_REAL.
        assert_eq!(d0.b_selector_column, Some(FILTER_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(H2G2_COL_IS_REAL));
        assert_eq!(d2.b_selector_column, Some(SSIG_COL_IS_REAL));
        assert_eq!(d3.b_selector_column, Some(PAIR_COL_IS_REAL));

        // Labels are unique.
        let labels = [
            d0.label.as_str(),
            d1.label.as_str(),
            d2.label.as_str(),
            d3.label.as_str(),
        ];
        let mut sorted = labels.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "descriptor labels must be unique");

        // Build the per-AIR witnesses + traces at BLS48-581 and confirm
        // the 5-trace orchestrator inputs are constructible.
        let curve = CurveType::Bls48581;
        let sca_w = build_composer_witness();
        assert_eq!(sca_w.rows.len(), SCA_ROWS_PER_INSTANCE, "composer emits 4 rows");
        let filter_w = build_filter_witness();
        let h2g2_w = build_h2g2_witness();
        let ssig_w = build_sync_sig_witness();
        let pair_w = build_pairing_witness();

        // Honest pairing_result must verify on the composer row 0
        // (host-side `fast_aggregate_verify` succeeded).
        assert_eq!(
            sca_w.rows[0].pairing_result, 1,
            "host-side pairing_result must be 1 for honest 2-signer aggregate",
        );

        let trace_0 = build_sca_trace(&sca_w, curve);
        let trace_1 = build_filter_trace(&filter_w, curve);
        let trace_2 = build_h2g2_trace(&h2g2_w, curve);
        let trace_3 = build_ssig_trace(&ssig_w, curve);
        let trace_4 = build_pair_trace(&pair_w, curve);

        // Per-AIR num_rows is positive and trace columns match the
        // published NUM_COLUMNS constants.
        assert!(trace_0.num_rows >= SCA_ROWS_PER_INSTANCE);
        assert_eq!(trace_0.columns.len(), SCA_NUM_COLUMNS);
        assert_eq!(trace_1.columns.len(), FILTER_NUM_COLUMNS);
        assert_eq!(trace_2.columns.len(), H2G2_NUM_COLUMNS);
        assert_eq!(trace_3.columns.len(), SSIG_NUM_COLUMNS);
        assert_eq!(trace_4.columns.len(), PAIR_NUM_COLUMNS);

        let cs_0 = SyncCommitteeAggregateComposerConstraintSystem::new(trace_0.num_rows);
        let cs_1 = SyncCommitteeFilterConstraintSystem::new(trace_1.num_rows);
        let cs_2 = HashToG2ConstraintSystem::new(trace_2.num_rows);
        let cs_3 = SyncCommitteeSigConstraintSystem::new(trace_3.num_rows);
        let cs_4 = BlsPairingConstraintSystem::new(trace_4.num_rows);

        // Assemble the input shape `joint_prove` takes.
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
            (&trace_4, &cs_4),
        ];
        let linkages = vec![d0.clone(), d1.clone(), d2.clone(), d3.clone()];

        assert_eq!(traces.len(), 5, "5-AIR joint_prove input must have 5 traces");
        assert_eq!(linkages.len(), 4, "must wire exactly 4 descriptors");

        // Layer index bounds + per-side column bounds.
        let num_cols_per_layer = [
            SCA_NUM_COLUMNS,
            FILTER_NUM_COLUMNS,
            H2G2_NUM_COLUMNS,
            SSIG_NUM_COLUMNS,
            PAIR_NUM_COLUMNS,
        ];
        for link in &linkages {
            assert!(link.a_layer_index < traces.len(), "a_layer in bounds");
            assert!(link.b_layer_index < traces.len(), "b_layer in bounds");
            for &c in &link.a_columns {
                assert!(
                    c < num_cols_per_layer[link.a_layer_index],
                    "a_column {} out of range for layer {} (max {})",
                    c, link.a_layer_index, num_cols_per_layer[link.a_layer_index],
                );
            }
            for &c in &link.b_columns {
                assert!(
                    c < num_cols_per_layer[link.b_layer_index],
                    "b_column {} out of range for layer {} (max {})",
                    c, link.b_layer_index, num_cols_per_layer[link.b_layer_index],
                );
            }
            if let Some(sa) = link.a_selector_column {
                assert!(sa < num_cols_per_layer[link.a_layer_index]);
            }
            if let Some(sb) = link.b_selector_column {
                assert!(sb < num_cols_per_layer[link.b_layer_index]);
            }
        }

        // Confirm phase-selector firing per row on the composer trace.
        let sca_cols: Vec<&Vec<Scalar>> =
            trace_0.columns.iter().map(|p| &p.evaluations).collect();
        assert!(sca_cols[SCA_COL_IS_FILTER_PHASE][0].is_one());
        assert!(sca_cols[SCA_COL_IS_HASH_TO_G2_PHASE][1].is_one());
        assert!(sca_cols[SCA_COL_IS_AGG_PUBKEY_PHASE][2].is_one());
        assert!(sca_cols[SCA_COL_IS_PAIRING_PHASE][3].is_one());
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 5-AIR `joint_prove` + `joint_verify` round-trip with
    /// 4 cross-AIR LogUp descriptors. Marked `#[ignore]` for the same
    /// reasons as [`crate::integration_hash_to_curve_joint_prove`].
    #[test]
    #[ignore = "slow: 5-AIR joint_prove + joint_verify under BLS48-581 (9 inner prove calls)"]
    fn honest_sync_committee_aggregate_composer_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let sca_w = build_composer_witness();
        let filter_w = build_filter_witness();
        let h2g2_w = build_h2g2_witness();
        let ssig_w = build_sync_sig_witness();
        let pair_w = build_pairing_witness();

        let trace_0 = build_sca_trace(&sca_w, curve);
        let trace_1 = build_filter_trace(&filter_w, curve);
        let trace_2 = build_h2g2_trace(&h2g2_w, curve);
        let trace_3 = build_ssig_trace(&ssig_w, curve);
        let trace_4 = build_pair_trace(&pair_w, curve);

        let cs_0 = SyncCommitteeAggregateComposerConstraintSystem::new(trace_0.num_rows);
        let cs_1 = SyncCommitteeFilterConstraintSystem::new(trace_1.num_rows);
        let cs_2 = HashToG2ConstraintSystem::new(trace_2.num_rows);
        let cs_3 = SyncCommitteeSigConstraintSystem::new(trace_3.num_rows);
        let cs_4 = BlsPairingConstraintSystem::new(trace_4.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
            (&trace_4, &cs_4),
        ];
        let linkages = vec![
            d0_filter_pubkey_anchor(),
            d1_hash_to_g2_msg_anchor(),
            d2_sync_sig_pubkey_anchor(),
            d3_pairing_pubkey_anchor(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 5-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 5, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            4,
            "expected one CrossAirLogUpProof per descriptor",
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2, &cs_3, &cs_4];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 5-AIR joint_verify must accept",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper the second descriptor's `closure_a` — `joint_verify` must
    /// reject. Marked `#[ignore]` because the setup half is the same
    /// `joint_prove` call as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_sync_committee_aggregate_composer_joint_verify_true"]
    fn tampered_sync_committee_aggregate_composer_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let sca_w = build_composer_witness();
        let filter_w = build_filter_witness();
        let h2g2_w = build_h2g2_witness();
        let ssig_w = build_sync_sig_witness();
        let pair_w = build_pairing_witness();

        let trace_0 = build_sca_trace(&sca_w, curve);
        let trace_1 = build_filter_trace(&filter_w, curve);
        let trace_2 = build_h2g2_trace(&h2g2_w, curve);
        let trace_3 = build_ssig_trace(&ssig_w, curve);
        let trace_4 = build_pair_trace(&pair_w, curve);

        let cs_0 = SyncCommitteeAggregateComposerConstraintSystem::new(trace_0.num_rows);
        let cs_1 = SyncCommitteeFilterConstraintSystem::new(trace_1.num_rows);
        let cs_2 = HashToG2ConstraintSystem::new(trace_2.num_rows);
        let cs_3 = SyncCommitteeSigConstraintSystem::new(trace_3.num_rows);
        let cs_4 = BlsPairingConstraintSystem::new(trace_4.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
            (&trace_4, &cs_4),
        ];
        let linkages = vec![
            d0_filter_pubkey_anchor(),
            d1_hash_to_g2_msg_anchor(),
            d2_sync_sig_pubkey_anchor(),
            d3_pairing_pubkey_anchor(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 5-AIR joint_prove must succeed");

        // Tamper the SECOND descriptor's closure_a (h2g2 msg anchor).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[1].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2, &cs_3, &cs_4];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the h2g2 descriptor",
        );
    }
}
