//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! for the Data-Availability-Sampling (DAS) chain:
//!
//! - layer 0: [`crate::data_availability_sampling_air`] — DAS cell
//!   (commitment, cell_index, cell_value, cell_proof, is_valid, is_real)
//!   gated by [`crate::data_availability_sampling_air::COL_IS_REAL`].
//! - layer 1: [`crate::kzg_point_eval_air`] — EIP-4844 point-eval
//!   precompile witness; selector
//!   [`crate::kzg_point_eval_air::COL_IS_REAL`].
//! - layer 2: [`crate::blob_kzg_air`] — blob KZG verification row;
//!   selector [`crate::blob_kzg_air::COL_IS_REAL`].
//!
//! ## Descriptors
//!
//! Two single-column-tuple cross-AIR LogUp descriptors:
//!
//! - **D0** binds the FIRST byte of the blob/KZG commitment between
//!   DAS layer 0 and point-eval layer 1:
//!     - A side: `das.COL_BLOB_COMMITMENT_OFFSET + 0`
//!       (gated by `das.COL_IS_REAL`),
//!     - B side: `kzg_point_eval.COL_COMMITMENT_OFFSET + 0`
//!       (gated by `kzg_point_eval.COL_IS_REAL`).
//!
//! - **D1** binds the cell-index scalar to the corresponding LSB
//!   of the BE-encoded `z` scalar consumed by the point-eval AIR:
//!     - A side: `das.COL_CELL_INDEX`
//!       (gated by `das.COL_IS_REAL`),
//!     - B side: `kzg_point_eval.COL_Z_OFFSET + 31`
//!       (least-significant BE byte; gated by
//!       `kzg_point_eval.COL_IS_REAL`).
//!
//!   For `cell_index = 1` the scalar `1` equals the LSB byte `1`,
//!   so the honest closure pair matches on a single real row.
//!
//! Layer 2 (`blob_kzg_air`) is included to stress the same
//! 3-AIR `joint_prove` orchestrator surface as
//! [`crate::integration_joint_prove_three_air`] without wiring an
//! additional descriptor — it exercises the per-AIR `prove_with_scheme`
//! + per-trace domain inflation paths.
//!
//! ## Static blob witness
//!
//! We use a dummy static witness — the per-AIR row-local constraints
//! for all three AIRs only check structural/byte-decomp invariants
//! (the algebraic KZG pairing equation is an oracle gap deferred to
//! a future pairing AIR), so we do NOT need a real blst KZG proof
//! here. The point of this integration test is exclusively to
//! exercise the cross-AIR LogUp orchestrator over the DAS chain.
//!
//! Honest witness values:
//!   - `cell_index = 1`,
//!   - `blob_commitment[0..48]` = `[0xa1, 0x00, ..., 0x00]`,
//!   - `cell_value[0..32]`      = `[0x00, ..., 0x00, 0x07]`,
//!   - `cell_proof[0..48]`      = `[0xb2, 0x00, ..., 0x00]`.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::blob_kzg_air::{
        build_trace_polynomials as build_bk_trace, BlobKzgConstraintSystem, BlobKzgWitness,
        COL_IS_REAL as BK_COL_IS_REAL, G1_BYTES as BK_G1_BYTES, SCALAR_BYTES as BK_SCALAR_BYTES,
    };
    use crate::data_availability_sampling_air::{
        build_trace_polynomials as build_das_trace, from_cell as das_from_cell,
        DasCellConstraintSystem, CELL_VALUE_LEN, COL_BLOB_COMMITMENT_OFFSET, COL_CELL_INDEX,
        COL_IS_REAL as DAS_COL_IS_REAL, COMMITMENT_LEN, PROOF_LEN,
    };
    use crate::kzg_point_eval_air::{
        build_trace_polynomials as build_kpe_trace, KzgPointEvalConstraintSystem,
        KzgPointEvalWitness, COL_COMMITMENT_OFFSET as KPE_COL_COMMITMENT_OFFSET,
        COL_IS_REAL as KPE_COL_IS_REAL, COL_Z_OFFSET as KPE_COL_Z_OFFSET, G1_LEN as KPE_G1_LEN,
        SCALAR_LEN as KPE_SCALAR_LEN, VERSIONED_HASH_LEN as KPE_VH_LEN,
    };

    // ─── Honest witness constants ─────────────────────────────────────

    const HONEST_CELL_INDEX: u64 = 1;
    /// The last (LSB) byte of the BE-encoded `z` scalar, i.e.
    /// `kzg_point_eval.COL_Z_OFFSET + (KPE_SCALAR_LEN - 1)`. For
    /// `cell_index = 1` this byte equals 1, matching the
    /// DAS `COL_CELL_INDEX` scalar value of 1.
    const KPE_COL_Z_LSB: usize = KPE_COL_Z_OFFSET + KPE_SCALAR_LEN - 1;

    fn honest_blob_commitment() -> [u8; COMMITMENT_LEN] {
        let mut c = [0u8; COMMITMENT_LEN];
        c[0] = 0xa1;
        c
    }

    fn honest_cell_value() -> [u8; CELL_VALUE_LEN] {
        let mut v = [0u8; CELL_VALUE_LEN];
        v[CELL_VALUE_LEN - 1] = 0x07;
        v
    }

    fn honest_cell_proof() -> [u8; PROOF_LEN] {
        let mut p = [0u8; PROOF_LEN];
        p[0] = 0xb2;
        p
    }

    fn honest_versioned_hash() -> [u8; KPE_VH_LEN] {
        let mut vh = [0u8; KPE_VH_LEN];
        vh[0] = 0x01; // EIP-4844 versioned-hash tag byte.
        vh
    }

    fn honest_z_be() -> [u8; KPE_SCALAR_LEN] {
        // BE-encoded `cell_index = 1`: 32 bytes, all zero except LSB = 1.
        let mut z = [0u8; KPE_SCALAR_LEN];
        z[KPE_SCALAR_LEN - 1] = HONEST_CELL_INDEX as u8;
        z
    }

    fn honest_y_be() -> [u8; KPE_SCALAR_LEN] {
        // Matches DAS cell_value (last byte 0x07, rest zero).
        let mut y = [0u8; KPE_SCALAR_LEN];
        y[KPE_SCALAR_LEN - 1] = 0x07;
        y
    }

    fn honest_kpe_commitment() -> [u8; KPE_G1_LEN] {
        // The first byte (col COL_COMMITMENT_OFFSET + 0) must match
        // DAS `COL_BLOB_COMMITMENT_OFFSET + 0`, so we set the leading
        // byte to 0xa1 just like the DAS row.
        let mut c = [0u8; KPE_G1_LEN];
        c[0] = 0xa1;
        c
    }

    fn honest_kpe_proof() -> [u8; KPE_G1_LEN] {
        let mut p = [0u8; KPE_G1_LEN];
        p[0] = 0xb2;
        p
    }

    fn honest_blob_kzg_commitment() -> [u8; BK_G1_BYTES] {
        // Top 3 flag bits cleared so the per-row limb-decomposition
        // constraint round-trips: the BlobKzg trace builder commits
        // `commitment_x = commitment & 0x1f` to the limb columns while
        // committing raw `commitment` bytes to the byte columns. The
        // constraint requires limb == Σ byte * weight, so byte 0's
        // flag bits MUST be zero or the constraint cannot hold.
        let mut c = [0u8; BK_G1_BYTES];
        c[0] = 0xa1 & 0x1f;
        c
    }

    fn honest_blob_kzg_proof() -> [u8; BK_G1_BYTES] {
        // Top 3 flag bits cleared; see `honest_blob_kzg_commitment`.
        let mut p = [0u8; BK_G1_BYTES];
        p[0] = 0xb2 & 0x1f;
        p
    }

    fn honest_blob_kzg_z() -> [u8; BK_SCALAR_BYTES] {
        let mut z = [0u8; BK_SCALAR_BYTES];
        z[BK_SCALAR_BYTES - 1] = HONEST_CELL_INDEX as u8;
        z
    }

    fn honest_blob_kzg_y() -> [u8; BK_SCALAR_BYTES] {
        let mut y = [0u8; BK_SCALAR_BYTES];
        y[BK_SCALAR_BYTES - 1] = 0x07;
        y
    }

    // ─── Descriptor builders ──────────────────────────────────────────

    /// D0: `das.blob_commitment[0]` (layer 0) ↔
    /// `kpe.commitment[0]` (layer 1). Single-column tuple.
    fn commitment_byte0_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "das_to_kpe_commitment_byte0_v1".into(),
            a_layer_index: 0,
            a_columns: vec![COL_BLOB_COMMITMENT_OFFSET],
            a_selector_column: Some(DAS_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![KPE_COL_COMMITMENT_OFFSET],
            b_selector_column: Some(KPE_COL_IS_REAL),
        }
    }

    /// D1: `das.cell_index` (layer 0) ↔ `kpe.z_be[LSB]` (layer 1).
    /// For `cell_index = 1` the scalar 1 equals the LSB byte value 1.
    fn cell_index_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "das_to_kpe_cell_index_v1".into(),
            a_layer_index: 0,
            a_columns: vec![COL_CELL_INDEX],
            a_selector_column: Some(DAS_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![KPE_COL_Z_LSB],
            b_selector_column: Some(KPE_COL_IS_REAL),
        }
    }

    // ─── Witness builders ─────────────────────────────────────────────

    fn build_das_witness() -> crate::data_availability_sampling_air::DasCellWitness {
        das_from_cell(
            honest_blob_commitment(),
            HONEST_CELL_INDEX,
            honest_cell_value(),
            honest_cell_proof(),
            true,
        )
    }

    fn build_kpe_witness() -> KzgPointEvalWitness {
        KzgPointEvalWitness::from_inputs(
            honest_versioned_hash(),
            honest_z_be(),
            honest_y_be(),
            honest_kpe_commitment(),
            honest_kpe_proof(),
        )
    }

    fn build_bk_witness() -> BlobKzgWitness {
        let evaluation = honest_blob_kzg_y(); // y == evaluation (row-local constraint).
        BlobKzgWitness::from_commitment_proof(
            &honest_blob_kzg_commitment(),
            &honest_blob_kzg_proof(),
            &honest_blob_kzg_z(),
            &honest_blob_kzg_y(),
            &evaluation,
        )
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check for both descriptors and the
    /// 3-trace orchestrator input shape, mirroring the pattern in
    /// [`crate::integration_joint_prove_three_air`]. Does NOT call
    /// `joint_prove`, so it runs comfortably under 120s in CI.
    #[test]
    fn descriptor_consistency() {
        let d0 = commitment_byte0_descriptor();
        let d1 = cell_index_descriptor();

        // Single-column tuples (the only shape `joint_prove` wires).
        assert_eq!(d0.a_columns.len(), 1);
        assert_eq!(d0.b_columns.len(), 1);
        assert_eq!(d1.a_columns.len(), 1);
        assert_eq!(d1.b_columns.len(), 1);

        // Layer wiring: both descriptors connect layer 0 ↔ layer 1.
        // Layer 2 (blob_kzg) is included as an inert AIR exercising
        // the orchestrator without a descriptor of its own.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert_ne!(d0.a_layer_index, d0.b_layer_index);
        assert_ne!(d1.a_layer_index, d1.b_layer_index);

        // Selectors wired on both sides of both descriptors.
        assert_eq!(d0.a_selector_column, Some(DAS_COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(KPE_COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(DAS_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(KPE_COL_IS_REAL));

        // Column-index pinning: any layout drift in the per-AIR
        // modules will surface here.
        assert_eq!(d0.a_columns[0], COL_BLOB_COMMITMENT_OFFSET);
        assert_eq!(d0.b_columns[0], KPE_COL_COMMITMENT_OFFSET);
        assert_eq!(d1.a_columns[0], COL_CELL_INDEX);
        assert_eq!(d1.b_columns[0], KPE_COL_Z_OFFSET + KPE_SCALAR_LEN - 1);

        // Build the three per-AIR witnesses + traces at BLS48-581 and
        // confirm the 3-trace orchestrator input shape is constructible.
        let curve = CurveType::Bls48581;
        let das_w = build_das_witness();
        let kpe_w = build_kpe_witness();
        let bk_w = build_bk_witness();

        let trace_0 = build_das_trace(&das_w, curve);
        let trace_1 = build_kpe_trace(&kpe_w, curve);
        let trace_2 = build_bk_trace(&bk_w, curve);

        let cs_0 = DasCellConstraintSystem::new(trace_0.num_rows);
        let cs_1 = KzgPointEvalConstraintSystem::new(trace_1.num_rows);
        let cs_2 = BlobKzgConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![d0.clone(), d1.clone()];

        assert_eq!(traces.len(), 3, "3-AIR joint_prove input must have 3 traces");
        assert_eq!(linkages.len(), 2, "must wire exactly 2 descriptors");

        // Every linkage layer index must be in range — same bounds
        // check `joint_prove` performs at the top of its loop.
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // Numerical sanity for the honest closure equality on both
        // descriptors. The DAS row commits `blob_commitment[0] = 0xa1`
        // and `cell_index = 1`; the KPE row commits the same byte at
        // its commitment column and `z_be[LSB] = 1`.
        let das_cols = &trace_0.columns;
        let kpe_cols = &trace_1.columns;
        let bk_cols = &trace_2.columns;

        // D0 sanity on row 0.
        assert_eq!(
            das_cols[COL_BLOB_COMMITMENT_OFFSET].evaluations[0].to_bytes(),
            kpe_cols[KPE_COL_COMMITMENT_OFFSET].evaluations[0].to_bytes(),
            "honest D0 commitment-byte tuple must match on the real row",
        );
        // D1 sanity on row 0.
        assert_eq!(
            das_cols[COL_CELL_INDEX].evaluations[0].to_bytes(),
            kpe_cols[KPE_COL_Z_LSB].evaluations[0].to_bytes(),
            "honest D1 cell_index↔z_lsb tuple must match on the real row",
        );

        // Layer 2 sanity: blob_kzg_air's IS_REAL is set on row 0.
        assert_eq!(
            bk_cols[BK_COL_IS_REAL].evaluations[0].to_bytes(),
            Scalar::one(curve).to_bytes(),
            "blob_kzg_air must commit a real row at row 0",
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 3-AIR `joint_prove` + `joint_verify` round-trip with
    /// 2 cross-AIR LogUp descriptors, all under BLS48-581.
    ///
    /// Marked `#[ignore]` because `joint_prove` runs 3× per-AIR
    /// `prove_with_scheme` + 2× per-linkage `prove_with_scheme` +
    /// cross-trace and closure-wrap KZG opens; even on single-row
    /// witnesses the kzg_point_eval AIR commits 450 columns and
    /// declares 8-bit byte range lookups, inflating its per-AIR
    /// domain to 256. Expected release runtime is in the high hundreds
    /// of seconds. Run via `--ignored --release`.
    #[test]
    #[ignore = "slow: DAS 3-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_das_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let das_w = build_das_witness();
        let kpe_w = build_kpe_witness();
        let bk_w = build_bk_witness();

        let trace_0 = build_das_trace(&das_w, curve);
        let trace_1 = build_kpe_trace(&kpe_w, curve);
        let trace_2 = build_bk_trace(&bk_w, curve);

        // Compute target_padded matching joint_prove's auto-inflation:
        // every AIR here declares 8-bit byte range lookups, so each is
        // boosted to max(padded, RANGE_TABLE_SIZE=256); target_padded =
        // max across all AIRs. Pre-wire the CS omega/domain to match so
        // per-AIR verify sees the same domain as the prover.
        let target_padded: u64 = [
            trace_0.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_1.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_2.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
        ]
        .into_iter()
        .max()
        .unwrap();
        let omega = scheme.domain_generator(target_padded);

        let cs_0 = DasCellConstraintSystem::new(trace_0.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_1 = KzgPointEvalConstraintSystem::new(trace_1.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_2 = BlobKzgConstraintSystem::new(trace_2.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![commitment_byte0_descriptor(), cell_index_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest DAS 3-AIR joint_prove must succeed");

        assert_eq!(proofs.len(), 3, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            2,
            "expected one CrossAirLogUpProof per descriptor",
        );

        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest DAS 3-AIR joint_verify must accept matching tuples on both descriptors",
        );
    }

    /// Diagnostic surface of the DAS 3-AIR joint proof: runs the
    /// honest setup, then calls `joint_verify_diagnostic` to report
    /// which check (if any) fails. Useful for isolating cross-AIR
    /// LogUp regressions (per-AIR verify vs linkage SNARK vs
    /// cross-trace binding vs closure binding vs Fiat-Shamir).
    #[test]
    #[ignore = "slow: DAS 3-AIR diagnostic verify under BLS48-581"]
    fn honest_das_joint_verify_diagnostic() {
        use crate::cross_air_logup::{joint_verify_diagnostic, JointVerifyFailure};
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let das_w = build_das_witness();
        let kpe_w = build_kpe_witness();
        let bk_w = build_bk_witness();

        let trace_0 = build_das_trace(&das_w, curve);
        let trace_1 = build_kpe_trace(&kpe_w, curve);
        let trace_2 = build_bk_trace(&bk_w, curve);

        let target_padded: u64 = [
            trace_0.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_1.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_2.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
        ]
        .into_iter()
        .max()
        .unwrap();
        let omega = scheme.domain_generator(target_padded);

        let cs_0 = DasCellConstraintSystem::new(trace_0.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_1 = KzgPointEvalConstraintSystem::new(trace_1.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_2 = BlobKzgConstraintSystem::new(trace_2.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![commitment_byte0_descriptor(), cell_index_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest DAS 3-AIR joint_prove must succeed");
        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        let diag = joint_verify_diagnostic(
            &proofs, &cs_refs, &linkages, &ext, &scheme, curve,
        );
        // Print before asserting so the failing variant is visible on stderr.
        eprintln!("DAS joint_verify_diagnostic = {:?}", diag);
        assert_eq!(
            diag,
            JointVerifyFailure::Ok,
            "joint_verify_diagnostic should return Ok on honest DAS proof; got {:?}",
            diag,
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tampered closure: corrupt `closure_a` on descriptor D1
    /// (cell_index↔z_lsb). The verifier's scalar `closure_a ==
    /// closure_b` check must reject.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_das_joint_verify_true"]
    fn tampered_das_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let das_w = build_das_witness();
        let kpe_w = build_kpe_witness();
        let bk_w = build_bk_witness();

        let trace_0 = build_das_trace(&das_w, curve);
        let trace_1 = build_kpe_trace(&kpe_w, curve);
        let trace_2 = build_bk_trace(&bk_w, curve);

        let target_padded: u64 = [
            trace_0.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_1.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_2.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
        ]
        .into_iter()
        .max()
        .unwrap();
        let omega = scheme.domain_generator(target_padded);

        let cs_0 = DasCellConstraintSystem::new(trace_0.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_1 = KzgPointEvalConstraintSystem::new(trace_1.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_2 = BlobKzgConstraintSystem::new(trace_2.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![commitment_byte0_descriptor(), cell_index_descriptor()];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest DAS 3-AIR joint_prove must succeed");

        // Tamper with the second descriptor's closure_a so we exercise
        // the per-descriptor rejection path (mirrors the convention
        // used in `integration_joint_prove_three_air`).
        let bogus = Scalar::one(curve).add(&Scalar::one(curve)).to_bytes();
        ext.linkage_proofs[1].closure_a = bogus;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the second descriptor",
        );
    }
}
