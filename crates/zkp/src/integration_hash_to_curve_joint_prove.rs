//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! for the **`hash_to_curve_composition_air` 4-phase composer**.
//!
//! Task #148: wire the composition AIR into a 5-layer chain together
//! with its four sub-AIRs:
//!
//!   - layer 0: [`crate::hash_to_curve_composition_air`] (composer; 4
//!     consecutive rows, one per phase).
//!   - layer 1: [`crate::hash_to_field_air`] (phase 0 sub-AIR).
//!   - layer 2: [`crate::hash_to_g2_air`] (phase 1 sub-AIR).
//!   - layer 3: [`crate::isogeny_map_air`] (phase 2 sub-AIR).
//!   - layer 4: [`crate::g2_cofactor_clear_air`] (phase 3 sub-AIR).
//!
//! Four cross-AIR LogUp descriptors are wired, one per phase, each as
//! a **single-column tuple** (the `joint_prove` API currently asserts
//! `a_columns.len() == 1 && b_columns.len() == 1`):
//!
//! - D0 (phase 0 / h2f): anchors the first byte of `msg` on both sides
//!   (`composition.COL_MSG_OFFSET` ↔ `hash_to_field.COL_MSG_OFFSET`),
//!   gated by `COL_IS_H2F_PHASE` ↔ `hash_to_field.COL_IS_REAL`.
//! - D1 (phase 1 / sswu): anchors the first byte of `field_elements` on
//!   the composer side ↔ the first Fp limb of `u0_c0` in the
//!   hash_to_g2 AIR, gated by `COL_IS_SSWU_PHASE` ↔
//!   `hash_to_g2.COL_IS_SSWU_PHASE`.
//! - D2 (phase 2 / isogeny): anchors the first byte of `iso_outputs`
//!   on the composer side ↔ the first Fp limb of `out_x_c0` on the
//!   isogeny AIR, gated by `COL_IS_ISOGENY_PHASE` ↔
//!   `isogeny_map.COL_IS_REAL`.
//! - D3 (phase 3 / cofactor): anchors the first byte of `final_g2` on
//!   the composer side ↔ the first Fp limb of `out_x_c0` on the
//!   cofactor AIR, gated by `COL_IS_COFACTOR_PHASE` ↔
//!   `g2_cofactor_clear.COL_IS_REAL`.
//!
//! Because the column-shape descriptors emitted by the composition AIR
//! itself are multi-column (and joint_prove currently only consumes
//! single-column tuples), we build single-column projections here. This
//! mirrors the `joint_prove` smoke-test convention established by
//! [`crate::integration_joint_prove_three_air`].
//!
//! ## Witness alignment
//!
//! The composer witness is built from `(msg, dst)` via blst's
//! `hash_to_g2_affine`; the four sub-AIR witnesses are built from the
//! **same** `(msg, dst)` pair via their respective `from_message`
//! constructors. All four sub-AIRs treat the SSWU/isogeny intermediates
//! as identity-passthrough at this stage (the algebraic Stage-2
//! reduction is deferred), so each sub-AIR commits the final G2 point
//! bytes into its `in/out` slabs. This keeps the per-phase byte-anchor
//! columns on both sides of every descriptor consistent and lets the
//! single-column closure-equality close on the honest path.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::hash_to_curve_composition_air::{
        build_trace_polynomials as build_h2c_trace, HashToCurveCompositionConstraintSystem,
        HashToCurveCompositionWitness,
        COL_COF_OUT_X_C0_LIMB0_MIRROR as H2C_COL_COF_OUT_X_C0_LIMB0_MIRROR,
        COL_ISO_OUT_X_C0_LIMB0_MIRROR as H2C_COL_ISO_OUT_X_C0_LIMB0_MIRROR,
        COL_IS_COFACTOR_PHASE as H2C_COL_IS_COFACTOR_PHASE,
        COL_IS_H2F_PHASE as H2C_COL_IS_H2F_PHASE,
        COL_IS_ISOGENY_PHASE as H2C_COL_IS_ISOGENY_PHASE,
        COL_IS_SSWU_PHASE as H2C_COL_IS_SSWU_PHASE,
        COL_MSG_OFFSET as H2C_COL_MSG_OFFSET,
        COL_U0_C0_LIMB0_MIRROR as H2C_COL_U0_C0_LIMB0_MIRROR,
        NUM_COLUMNS as H2C_NUM_COLUMNS,
        ROWS_PER_INSTANCE as H2C_ROWS_PER_INSTANCE,
    };
    use crate::hash_to_field_air::{
        build_trace_polynomials as build_h2f_trace, HashToFieldConstraintSystem,
        HashToFieldWitness, COL_IS_REAL as H2F_COL_IS_REAL,
        COL_MSG_OFFSET as H2F_COL_MSG_OFFSET, NUM_COLUMNS as H2F_NUM_COLUMNS,
    };
    use crate::hash_to_g2_air::{
        build_trace_polynomials as build_h2g2_trace, HashToG2ConstraintSystem, HashToG2Witness,
        COL_IS_SSWU_PHASE as H2G2_COL_IS_SSWU_PHASE,
        COL_U0_C0_LIMB_OFFSET as H2G2_COL_U0_C0_LIMB_OFFSET,
        NUM_COLUMNS as H2G2_NUM_COLUMNS,
    };
    use crate::isogeny_map_air::{
        build_trace_polynomials as build_iso_trace, IsogenyMapConstraintSystem, IsogenyMapWitness,
        COL_IS_REAL as ISO_COL_IS_REAL,
        COL_OUT_X_C0_LIMB_OFFSET as ISO_COL_OUT_X_C0_LIMB_OFFSET,
        NUM_COLUMNS as ISO_NUM_COLUMNS,
    };
    use crate::g2_cofactor_clear_air::{
        build_trace_polynomials as build_cof_trace, G2CofactorClearConstraintSystem,
        G2CofactorClearWitness, COL_IS_REAL as COF_COL_IS_REAL,
        COL_OUT_X_C0_LIMB_OFFSET as COF_COL_OUT_X_C0_LIMB_OFFSET,
        NUM_COLUMNS as COF_NUM_COLUMNS,
    };

    /// Beacon-chain ciphersuite DST (POP variant) — matches the composer
    /// + sub-AIR oracles' default DST.
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
    const HONEST_MSG: &[u8] = b"h2c-composer-joint-prove";

    // ─── Single-column descriptor builders ────────────────────────────

    /// D0: composer `msg[0]` ↔ hash_to_field `msg[0]`, gated by the
    /// composer's H2F phase selector on side A and the h2f AIR's
    /// `IS_REAL` on side B.
    fn d0_h2f_msg_anchor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "h2c_composer_to_h2f_msg_anchor_v1".into(),
            a_layer_index: 0,
            a_columns: vec![H2C_COL_MSG_OFFSET],
            a_selector_column: Some(H2C_COL_IS_H2F_PHASE),
            b_layer_index: 1,
            b_columns: vec![H2F_COL_MSG_OFFSET],
            b_selector_column: Some(H2F_COL_IS_REAL),
        }
    }

    /// D1: composer's `u0.c0.limbs[0]` mirror ↔ hash_to_g2 `u0_c0`
    /// first limb, gated by the composer's SSWU phase selector ↔ the
    /// h2g2 AIR's SSWU-phase selector. Both sides commit the SAME u64
    /// scalar (Task #189 — the previous `field_elements[0]` byte vs
    /// u64 limb mismatch made multiset equality impossible).
    fn d1_sswu_limb_anchor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "h2c_composer_to_sswu_limb_anchor_v1".into(),
            a_layer_index: 0,
            a_columns: vec![H2C_COL_U0_C0_LIMB0_MIRROR],
            a_selector_column: Some(H2C_COL_IS_SSWU_PHASE),
            b_layer_index: 2,
            b_columns: vec![H2G2_COL_U0_C0_LIMB_OFFSET],
            b_selector_column: Some(H2G2_COL_IS_SSWU_PHASE),
        }
    }

    /// D2: composer's `iso_out_x_c0.limbs[0]` mirror ↔ isogeny_map
    /// `out_x_c0` first limb, gated by the composer's isogeny phase
    /// selector ↔ `isogeny_map.IS_REAL`. Both sides commit the SAME
    /// u64 scalar (Task #189).
    fn d2_isogeny_limb_anchor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "h2c_composer_to_isogeny_limb_anchor_v1".into(),
            a_layer_index: 0,
            a_columns: vec![H2C_COL_ISO_OUT_X_C0_LIMB0_MIRROR],
            a_selector_column: Some(H2C_COL_IS_ISOGENY_PHASE),
            b_layer_index: 3,
            b_columns: vec![ISO_COL_OUT_X_C0_LIMB_OFFSET],
            b_selector_column: Some(ISO_COL_IS_REAL),
        }
    }

    /// D3: composer's `cof_out_x_c0.limbs[0]` mirror ↔
    /// g2_cofactor_clear `out_x_c0` first limb, gated by the
    /// composer's cofactor phase selector ↔
    /// `g2_cofactor_clear.IS_REAL`. Both sides commit the SAME u64
    /// scalar (Task #189).
    fn d3_cofactor_limb_anchor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "h2c_composer_to_cofactor_limb_anchor_v1".into(),
            a_layer_index: 0,
            a_columns: vec![H2C_COL_COF_OUT_X_C0_LIMB0_MIRROR],
            a_selector_column: Some(H2C_COL_IS_COFACTOR_PHASE),
            b_layer_index: 4,
            b_columns: vec![COF_COL_OUT_X_C0_LIMB_OFFSET],
            b_selector_column: Some(COF_COL_IS_REAL),
        }
    }

    // ─── Witness builders ─────────────────────────────────────────────

    fn build_h2c_witness() -> HashToCurveCompositionWitness {
        HashToCurveCompositionWitness::from_message(HONEST_MSG, POP_DST)
            .expect("composer witness builds for honest msg")
    }

    fn build_h2f_witness() -> HashToFieldWitness {
        HashToFieldWitness::from_message(HONEST_MSG, POP_DST)
    }

    fn build_h2g2_witness() -> HashToG2Witness {
        HashToG2Witness::from_message(HONEST_MSG, POP_DST)
            .expect("hash_to_g2 witness builds for honest msg")
    }

    fn build_iso_witness() -> IsogenyMapWitness {
        IsogenyMapWitness::from_message(HONEST_MSG, POP_DST)
            .expect("isogeny_map witness builds for honest msg")
    }

    fn build_cof_witness() -> G2CofactorClearWitness {
        G2CofactorClearWitness::from_message(HONEST_MSG, POP_DST)
            .expect("g2_cofactor_clear witness builds for honest msg")
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check across all four descriptors plus a
    /// non-prove sanity check on the 5-trace shape that `joint_prove`
    /// receives. Does NOT run `joint_prove`, so it stays well under
    /// 120s and runs in CI.
    #[test]
    fn descriptor_consistency_hash_to_curve_composer() {
        let d0 = d0_h2f_msg_anchor();
        let d1 = d1_sswu_limb_anchor();
        let d2 = d2_isogeny_limb_anchor();
        let d3 = d3_cofactor_limb_anchor();

        // Single-column-tuple invariant — joint_prove asserts this.
        for d in [&d0, &d1, &d2, &d3] {
            assert_eq!(d.a_columns.len(), 1, "joint_prove requires single-column tuples (a)");
            assert_eq!(d.b_columns.len(), 1, "joint_prove requires single-column tuples (b)");
            assert_ne!(d.a_layer_index, d.b_layer_index, "descriptor spans two layers");
            assert!(d.a_selector_column.is_some(), "A side must be gated");
            assert!(d.b_selector_column.is_some(), "B side must be gated");
        }

        // Layer indices: composer = 0; h2f=1, h2g2=2, iso=3, cof=4.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 2);
        assert_eq!(d2.a_layer_index, 0);
        assert_eq!(d2.b_layer_index, 3);
        assert_eq!(d3.a_layer_index, 0);
        assert_eq!(d3.b_layer_index, 4);

        // A side selectors: each descriptor uses a distinct phase
        // selector on the composer, exercising the phase-disjoint
        // multi-row-per-instance pattern.
        assert_eq!(d0.a_selector_column, Some(H2C_COL_IS_H2F_PHASE));
        assert_eq!(d1.a_selector_column, Some(H2C_COL_IS_SSWU_PHASE));
        assert_eq!(d2.a_selector_column, Some(H2C_COL_IS_ISOGENY_PHASE));
        assert_eq!(d3.a_selector_column, Some(H2C_COL_IS_COFACTOR_PHASE));
        // B side selectors: per-sub-AIR is_real (or h2g2's SSWU phase).
        assert_eq!(d0.b_selector_column, Some(H2F_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(H2G2_COL_IS_SSWU_PHASE));
        assert_eq!(d2.b_selector_column, Some(ISO_COL_IS_REAL));
        assert_eq!(d3.b_selector_column, Some(COF_COL_IS_REAL));

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
        // the 5-trace orchestrator inputs are constructible. Column
        // bounds for each descriptor's a_columns / b_columns are
        // checked against the per-AIR NUM_COLUMNS so any layout drift
        // surfaces here.
        let curve = CurveType::Bls48581;
        let h2c_w = build_h2c_witness();
        assert_eq!(h2c_w.rows.len(), H2C_ROWS_PER_INSTANCE, "composer emits 4 rows");
        let h2f_w = build_h2f_witness();
        let h2g2_w = build_h2g2_witness();
        let iso_w = build_iso_witness();
        let cof_w = build_cof_witness();

        let trace_0 = build_h2c_trace(&h2c_w, curve);
        let trace_1 = build_h2f_trace(&h2f_w, curve);
        let trace_2 = build_h2g2_trace(&h2g2_w, curve);
        let trace_3 = build_iso_trace(&iso_w, curve);
        let trace_4 = build_cof_trace(&cof_w, curve);

        // Per-AIR num_rows is positive and trace columns match the
        // published NUM_COLUMNS constants.
        assert!(trace_0.num_rows >= H2C_ROWS_PER_INSTANCE);
        assert_eq!(trace_0.columns.len(), H2C_NUM_COLUMNS);
        assert_eq!(trace_1.columns.len(), H2F_NUM_COLUMNS);
        assert_eq!(trace_2.columns.len(), H2G2_NUM_COLUMNS);
        assert_eq!(trace_3.columns.len(), ISO_NUM_COLUMNS);
        assert_eq!(trace_4.columns.len(), COF_NUM_COLUMNS);

        let cs_0 = HashToCurveCompositionConstraintSystem::new(trace_0.num_rows);
        let cs_1 = HashToFieldConstraintSystem::new(trace_1.num_rows);
        let cs_2 = HashToG2ConstraintSystem::new(trace_2.num_rows);
        let cs_3 = IsogenyMapConstraintSystem::new(trace_3.num_rows);
        let cs_4 = G2CofactorClearConstraintSystem::new(trace_4.num_rows);

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

        // Every linkage layer index must be in range — this is the
        // bounds check `joint_prove` performs at the top of its loop.
        // Additionally check column indices are within the side's
        // committed NUM_COLUMNS.
        let num_cols_per_layer = [
            H2C_NUM_COLUMNS,
            H2F_NUM_COLUMNS,
            H2G2_NUM_COLUMNS,
            ISO_NUM_COLUMNS,
            COF_NUM_COLUMNS,
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
            // Selector columns must also be in range for their layer.
            if let Some(sa) = link.a_selector_column {
                assert!(sa < num_cols_per_layer[link.a_layer_index]);
            }
            if let Some(sb) = link.b_selector_column {
                assert!(sb < num_cols_per_layer[link.b_layer_index]);
            }
        }

        // Confirm at least one row of the composer is gated on each of
        // the four phase selectors (one per row), and is_real = 1.
        let h2c_cols: Vec<&Vec<Scalar>> =
            trace_0.columns.iter().map(|p| &p.evaluations).collect();
        // Row 0 = h2f phase: COL_IS_H2F_PHASE == 1; others zero.
        assert!(h2c_cols[H2C_COL_IS_H2F_PHASE][0].is_one());
        assert!(h2c_cols[H2C_COL_IS_SSWU_PHASE][1].is_one());
        assert!(h2c_cols[H2C_COL_IS_ISOGENY_PHASE][2].is_one());
        assert!(h2c_cols[H2C_COL_IS_COFACTOR_PHASE][3].is_one());
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 5-AIR `joint_prove` + `joint_verify` round-trip with
    /// 4 cross-AIR LogUp descriptors. Marked `#[ignore]` because each
    /// per-AIR `prove_with_scheme` plus 4 inner linkage SNARKs adds up
    /// to many minutes under BLS48-581 even with tiny per-AIR
    /// witnesses (every AIR declares 8-bit byte range lookups so the
    /// LogUp domain auto-inflates to 256).
    #[test]
    #[ignore = "slow: 5-AIR joint_prove + joint_verify under BLS48-581 (9 inner prove calls)"]
    fn honest_hash_to_curve_composer_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let h2c_w = build_h2c_witness();
        let h2f_w = build_h2f_witness();
        let h2g2_w = build_h2g2_witness();
        let iso_w = build_iso_witness();
        let cof_w = build_cof_witness();

        let trace_0 = build_h2c_trace(&h2c_w, curve);
        let trace_1 = build_h2f_trace(&h2f_w, curve);
        let trace_2 = build_h2g2_trace(&h2g2_w, curve);
        let trace_3 = build_iso_trace(&iso_w, curve);
        let trace_4 = build_cof_trace(&cof_w, curve);

        let cs_0 = HashToCurveCompositionConstraintSystem::new(trace_0.num_rows);
        let cs_1 = HashToFieldConstraintSystem::new(trace_1.num_rows);
        let cs_2 = HashToG2ConstraintSystem::new(trace_2.num_rows);
        let cs_3 = IsogenyMapConstraintSystem::new(trace_3.num_rows);
        let cs_4 = G2CofactorClearConstraintSystem::new(trace_4.num_rows);

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
            d0_h2f_msg_anchor(),
            d1_sswu_limb_anchor(),
            d2_isogeny_limb_anchor(),
            d3_cofactor_limb_anchor(),
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
    #[ignore = "slow: depends on the joint_prove setup of honest_hash_to_curve_composer_joint_verify_true"]
    fn tampered_hash_to_curve_composer_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let h2c_w = build_h2c_witness();
        let h2f_w = build_h2f_witness();
        let h2g2_w = build_h2g2_witness();
        let iso_w = build_iso_witness();
        let cof_w = build_cof_witness();

        let trace_0 = build_h2c_trace(&h2c_w, curve);
        let trace_1 = build_h2f_trace(&h2f_w, curve);
        let trace_2 = build_h2g2_trace(&h2g2_w, curve);
        let trace_3 = build_iso_trace(&iso_w, curve);
        let trace_4 = build_cof_trace(&cof_w, curve);

        let cs_0 = HashToCurveCompositionConstraintSystem::new(trace_0.num_rows);
        let cs_1 = HashToFieldConstraintSystem::new(trace_1.num_rows);
        let cs_2 = HashToG2ConstraintSystem::new(trace_2.num_rows);
        let cs_3 = IsogenyMapConstraintSystem::new(trace_3.num_rows);
        let cs_4 = G2CofactorClearConstraintSystem::new(trace_4.num_rows);

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
            d0_h2f_msg_anchor(),
            d1_sswu_limb_anchor(),
            d2_isogeny_limb_anchor(),
            d3_cofactor_limb_anchor(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 5-AIR joint_prove must succeed");

        // Tamper the SECOND descriptor's closure_a (SSWU limb anchor).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[1].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2, &cs_3, &cs_4];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the SSWU descriptor",
        );
    }
}
