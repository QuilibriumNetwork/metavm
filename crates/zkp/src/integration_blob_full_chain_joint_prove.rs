//! Integration `joint_prove` / `joint_verify` test for the **full EIP-4844
//! blob chain**: ties together
//!
//!   - layer 0: [`crate::versioned_hash_air`]      — commitment ↔ versioned-hash,
//!   - layer 1: [`crate::blob_kzg_air`]            — blob KZG opening row,
//!   - layer 2: [`crate::kzg_point_eval_air`]      — point-eval precompile,
//!   - layer 3: [`crate::sha256_extract`]          — per-invocation byte view of
//!                                                   `sha256(commitment) → digest`.
//!
//! Task #235.
//!
//! ## Descriptors
//!
//! Three single-column-tuple cross-AIR LogUp descriptors:
//!
//!   - **D0** (`commitment_byte0_vh_to_bk`):
//!     `versioned_hash.COL_COMMITMENT[0]` (layer 0, gated by VH `COL_IS_REAL`)
//!     ↔ `blob_kzg.COL_COMMITMENT_BYTES[0]` (layer 1, gated by BK `COL_IS_REAL`).
//!
//!   - **D1** (`commitment_byte0_bk_to_kpe`):
//!     `blob_kzg.COL_COMMITMENT_BYTES[0]` (layer 1, gated by BK `COL_IS_REAL`)
//!     ↔ `kzg_point_eval.COL_COMMITMENT[0]` (layer 2, gated by KPE `COL_IS_REAL`).
//!
//!   - **D2** (`versioned_hash_byte0_vh_to_sha256_extract`):
//!     `versioned_hash.COL_VERSIONED_HASH[0]` (layer 0, gated by VH `COL_IS_REAL`)
//!     ↔ `sha256_extract.COL_OUTPUT_BYTE[0]` (layer 3, gated by `COL_IS_REAL`).
//!
//!     The versioned-hash AIR's row-local constraint pins
//!     `versioned_hash[0] = 0x01` (the EIP-4844 `VERSIONED_HASH_VERSION_KZG`
//!     magic). The Sha256Extract witness is **synthetic** for this row:
//!     since that AIR only enforces `is_real ∈ {0, 1}`, the prover commits
//!     `output[0] = 0x01` directly so the D2 closure matches. The
//!     algebraic binding that `versioned_hash[0]` must agree with the
//!     SHA-256 *digest*'s first byte (after the magic stamp) is part of
//!     the wider chain covered by `make_versioned_hash_to_sha256_descriptor`;
//!     the magic-byte coincidence on byte 0 is the single-column slice we
//!     exercise here.
//!
//! ## Real witness construction (task #255)
//!
//! The 4-element blob is a real BLS12-381 polynomial `p(x) = 1 + 2x +
//! 3x^2 + 4x^3` (coefficient form). `commitment = commit_coefficients(blob)`
//! is a real 48-byte compressed BLS12-381 G1 point computed via blst on
//! the embedded trusted setup. The opening point `z = 7` produces
//! `y = p(7) = 1 + 14 + 147 + 1372 = 1534`; the quotient
//! `q(x) = (p(x) - y) / (x - z)` is committed via `commit_coefficients`
//! to yield the real 48-byte KZG `proof`. `versioned_hash =
//! 0x01 || sha256(commitment)[1..32]` is computed by
//! `VersionedHashWitness::from_commitment`. The Sha256Extract row remains
//! synthetic: it mirrors `output[0] = 0x01` so D2 closes (the SE AIR only
//! enforces `is_real ∈ {0,1}` — full sha256-byte binding lives in the
//! wider chain via `make_versioned_hash_to_sha256_descriptor`).
//!
//! ## What this file does
//!
//! 1. The fast `descriptor_consistency` test pins all three descriptors'
//!    shapes (single-column tuples, layer wiring, column constants,
//!    selectors) and builds all four traces to confirm the honest
//!    closures match.
//!
//! 2. Two `#[ignore]`'d slow tests run the full 4-AIR
//!    `joint_prove` / `joint_verify` round-trip and a tampered-closure
//!    rejection.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::blob_kzg_air::{
        build_trace_polynomials as build_bk_trace, BlobKzgConstraintSystem, BlobKzgWitness,
        COL_COMMITMENT_BYTES_OFFSET as BK_COL_COMMITMENT_OFFSET, COL_IS_REAL as BK_COL_IS_REAL,
        G1_BYTES as BK_G1_BYTES, SCALAR_BYTES as BK_SCALAR_BYTES,
    };
    use crate::kzg_point_eval_air::{
        build_trace_polynomials as build_kpe_trace, KzgPointEvalConstraintSystem,
        KzgPointEvalWitness, COL_COMMITMENT_OFFSET as KPE_COL_COMMITMENT_OFFSET,
        COL_IS_REAL as KPE_COL_IS_REAL, G1_LEN as KPE_G1_LEN, SCALAR_LEN as KPE_SCALAR_LEN,
        VERSIONED_HASH_LEN as KPE_VH_LEN,
    };
    use crate::sha256_extract::{
        build_trace_polynomials as build_se_trace, Sha256ExtractConstraintSystem,
        Sha256ExtractRow, Sha256ExtractWitness,
        COL_IS_REAL as SE_COL_IS_REAL, COL_OUTPUT_BYTE_OFFSET as SE_COL_OUTPUT_OFFSET,
    };
    use crate::versioned_hash_air::{
        build_trace_polynomials as build_vh_trace, VersionedHashConstraintSystem,
        VersionedHashRow, VersionedHashWitness, COL_COMMITMENT_OFFSET as VH_COL_COMMITMENT_OFFSET,
        COL_IS_REAL as VH_COL_IS_REAL, COL_VERSIONED_HASH_OFFSET as VH_COL_VERSIONED_HASH_OFFSET,
        COMMITMENT_LEN as VH_COMMITMENT_LEN, DIGEST_LEN as VH_DIGEST_LEN,
        VERSIONED_HASH_VERSION_KZG,
    };

    // ─── Real witness construction (blst BLS12-381 KZG) ───────────────

    /// Build a real BLS12-381 KZG (commitment, proof, z, y) tuple from
    /// the 4-element blob `p(x) = 1 + 2x + 3x^2 + 4x^3` opened at z = 7.
    ///
    /// Returns `(commitment_bytes, proof_bytes, z_be, y_be)` where:
    /// - `commitment_bytes` = 48-byte BLS12-381 G1 compressed (real,
    ///   computed by `commit_coefficients` via the embedded trusted setup).
    /// - `proof_bytes` = 48-byte BLS12-381 G1 compressed commitment of
    ///   the quotient polynomial `q(x) = (p(x) - y) / (x - z)`.
    /// - `z_be`, `y_be` = 32-byte big-endian scalar encodings of z and y.
    fn real_kzg_witness() -> ([u8; 48], [u8; 48], [u8; 32], [u8; 32]) {
        use crate::scheme::bls12381_scheme::Bls12381Scheme;
        use crate::scheme::CommitmentScheme;

        let curve_kzg = CurveType::Bls12381;
        let scheme = Bls12381Scheme::new();
        scheme.init();

        // Blob = polynomial p(x) = 1 + 2x + 3x^2 + 4x^3 in coefficient form.
        let blob: Vec<Scalar> = (1u64..=4u64)
            .map(|v| Scalar::from_u64(v, curve_kzg))
            .collect();

        // Real commitment via embedded BLS12-381 trusted setup.
        let commitment_vec = scheme.commit_coefficients(&blob);
        let mut commitment = [0u8; 48];
        commitment.copy_from_slice(&commitment_vec);

        // Open at z = 7: y = p(7) = 1 + 14 + 147 + 1372 = 1534.
        let z = Scalar::from_u64(7, curve_kzg);
        let y = scheme.eval_poly_at(&blob, &z);

        // Quotient q(x) = (p(x) - y) / (x - z), then commit.
        let mut shifted = blob.clone();
        shifted[0] = shifted[0].sub(&y);
        let quotient = scheme.div_by_linear(&shifted, &z);
        let proof_vec = scheme.commit_coefficients(&quotient);
        let mut proof = [0u8; 48];
        proof.copy_from_slice(&proof_vec);

        // blst encodes scalars little-endian; the EIP-4844 wire format
        // (and the AIR byte columns) consume big-endian, so reverse.
        let z_le = z.to_bytes();
        let y_le = y.to_bytes();
        let mut z_be = [0u8; 32];
        let mut y_be = [0u8; 32];
        for i in 0..32 {
            z_be[i] = z_le[31 - i];
            y_be[i] = y_le[31 - i];
        }

        (commitment, proof, z_be, y_be)
    }

    /// Cached real KZG tuple — derived once per test invocation.
    fn honest_commitment_and_proof() -> ([u8; 48], [u8; 48], [u8; 32], [u8; 32]) {
        real_kzg_witness()
    }

    fn honest_commitment_vh() -> [u8; VH_COMMITMENT_LEN] {
        let (c, _, _, _) = honest_commitment_and_proof();
        debug_assert_eq!(VH_COMMITMENT_LEN, 48);
        c
    }

    fn honest_commitment_bk() -> [u8; BK_G1_BYTES] {
        let (c, _, _, _) = honest_commitment_and_proof();
        debug_assert_eq!(BK_G1_BYTES, 48);
        c
    }

    fn honest_commitment_kpe() -> [u8; KPE_G1_LEN] {
        let (c, _, _, _) = honest_commitment_and_proof();
        debug_assert_eq!(KPE_G1_LEN, 48);
        c
    }

    fn honest_proof_bk() -> [u8; BK_G1_BYTES] {
        let (_, p, _, _) = honest_commitment_and_proof();
        p
    }

    fn honest_proof_kpe() -> [u8; KPE_G1_LEN] {
        let (_, p, _, _) = honest_commitment_and_proof();
        p
    }

    fn honest_z_be() -> [u8; KPE_SCALAR_LEN] {
        let (_, _, z, _) = honest_commitment_and_proof();
        debug_assert_eq!(KPE_SCALAR_LEN, 32);
        z
    }

    fn honest_y_be() -> [u8; KPE_SCALAR_LEN] {
        let (_, _, _, y) = honest_commitment_and_proof();
        debug_assert_eq!(KPE_SCALAR_LEN, 32);
        y
    }

    fn honest_bk_z() -> [u8; BK_SCALAR_BYTES] {
        let (_, _, z, _) = honest_commitment_and_proof();
        debug_assert_eq!(BK_SCALAR_BYTES, 32);
        z
    }

    fn honest_bk_y() -> [u8; BK_SCALAR_BYTES] {
        let (_, _, _, y) = honest_commitment_and_proof();
        debug_assert_eq!(BK_SCALAR_BYTES, 32);
        y
    }

    fn honest_versioned_hash() -> [u8; KPE_VH_LEN] {
        // versioned_hash = 0x01 || sha256(commitment)[1..32].
        let digest = crate::sha256::sha256(&honest_commitment_vh());
        let mut vh = digest;
        vh[0] = VERSIONED_HASH_VERSION_KZG;
        vh
    }

    // ─── Witness builders ─────────────────────────────────────────────

    fn build_vh_witness() -> VersionedHashWitness {
        // Use the public single-row constructor — it computes
        // sha256(commitment) and stamps byte 0 = 0x01.
        VersionedHashWitness::from_commitment(honest_commitment_vh())
    }

    fn build_bk_witness() -> BlobKzgWitness {
        let evaluation = honest_bk_y(); // y == evaluation (row-local equality constraint).
        BlobKzgWitness::from_commitment_proof(
            &honest_commitment_bk(),
            &honest_proof_bk(),
            &honest_bk_z(),
            &honest_bk_y(),
            &evaluation,
        )
    }

    fn build_kpe_witness() -> KzgPointEvalWitness {
        KzgPointEvalWitness::from_inputs(
            honest_versioned_hash(),
            honest_z_be(),
            honest_y_be(),
            honest_commitment_kpe(),
            honest_proof_kpe(),
        )
    }

    fn build_se_witness() -> Sha256ExtractWitness {
        // SYNTHETIC: the sha256_extract AIR only enforces `is_real
        // ∈ {0, 1}`. We commit the VH AIR's *versioned hash* (with
        // byte 0 = 0x01) as the "output", so D2 closure (VH versioned-
        // hash byte 0 ↔ SE output byte 0) matches with both equal to
        // 0x01. The input mirrors the real commitment bytes for
        // documentary clarity; it is not algebraically bound here.
        let commitment = honest_commitment_vh();
        let mut input = [0u8; 64];
        input[..48].copy_from_slice(&commitment);
        let digest = crate::sha256::sha256(&commitment);
        let mut output = digest;
        output[0] = VERSIONED_HASH_VERSION_KZG;
        Sha256ExtractWitness {
            invocations: vec![Sha256ExtractRow { input, output }],
            mirror_byte0: Vec::new(),
        }
    }

    // ─── Descriptor builders ──────────────────────────────────────────

    /// D0: `versioned_hash.commitment[0]` (layer 0) ↔
    /// `blob_kzg.commitment_bytes[0]` (layer 1). Single-column tuple.
    fn d0_commitment_byte0_vh_to_bk() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "blob_full_chain_commitment_byte0_vh_to_bk_v1".into(),
            a_layer_index: 0,
            a_columns: vec![VH_COL_COMMITMENT_OFFSET],
            a_selector_column: Some(VH_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![BK_COL_COMMITMENT_OFFSET],
            b_selector_column: Some(BK_COL_IS_REAL),
        }
    }

    /// D1: `blob_kzg.commitment_bytes[0]` (layer 1) ↔
    /// `kzg_point_eval.commitment[0]` (layer 2). Single-column tuple.
    fn d1_commitment_byte0_bk_to_kpe() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "blob_full_chain_commitment_byte0_bk_to_kpe_v1".into(),
            a_layer_index: 1,
            a_columns: vec![BK_COL_COMMITMENT_OFFSET],
            a_selector_column: Some(BK_COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![KPE_COL_COMMITMENT_OFFSET],
            b_selector_column: Some(KPE_COL_IS_REAL),
        }
    }

    /// D2: `versioned_hash.versioned_hash[0]` (layer 0) ↔
    /// `sha256_extract.output_byte[0]` (layer 3). Both equal the EIP-4844
    /// magic byte `0x01` on honest rows.
    fn d2_versioned_hash_byte0_vh_to_sha256_extract() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "blob_full_chain_versioned_hash_byte0_vh_to_sha256_extract_v1".into(),
            a_layer_index: 0,
            a_columns: vec![VH_COL_VERSIONED_HASH_OFFSET],
            a_selector_column: Some(VH_COL_IS_REAL),
            b_layer_index: 3,
            b_columns: vec![SE_COL_OUTPUT_OFFSET],
            b_selector_column: Some(SE_COL_IS_REAL),
        }
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness + honest-closure check for all three
    /// descriptors and the 4-trace orchestrator input shape. Mirrors
    /// the canonical pattern from
    /// [`crate::integration_das_joint_prove`]. Does NOT call
    /// `joint_prove`.
    #[test]
    fn descriptor_consistency() {
        let d0 = d0_commitment_byte0_vh_to_bk();
        let d1 = d1_commitment_byte0_bk_to_kpe();
        let d2 = d2_versioned_hash_byte0_vh_to_sha256_extract();

        // Single-column tuples (the only shape `joint_prove` wires).
        assert_eq!(d0.a_columns.len(), 1);
        assert_eq!(d0.b_columns.len(), 1);
        assert_eq!(d1.a_columns.len(), 1);
        assert_eq!(d1.b_columns.len(), 1);
        assert_eq!(d2.a_columns.len(), 1);
        assert_eq!(d2.b_columns.len(), 1);

        // Layer wiring.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 1);
        assert_eq!(d1.b_layer_index, 2);
        assert_eq!(d2.a_layer_index, 0);
        assert_eq!(d2.b_layer_index, 3);
        for d in [&d0, &d1, &d2] {
            assert_ne!(d.a_layer_index, d.b_layer_index);
        }

        // Column pinning.
        assert_eq!(d0.a_columns[0], VH_COL_COMMITMENT_OFFSET);
        assert_eq!(d0.b_columns[0], BK_COL_COMMITMENT_OFFSET);
        assert_eq!(d1.a_columns[0], BK_COL_COMMITMENT_OFFSET);
        assert_eq!(d1.b_columns[0], KPE_COL_COMMITMENT_OFFSET);
        assert_eq!(d2.a_columns[0], VH_COL_VERSIONED_HASH_OFFSET);
        assert_eq!(d2.b_columns[0], SE_COL_OUTPUT_OFFSET);

        // Selector pinning on every side.
        assert_eq!(d0.a_selector_column, Some(VH_COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(BK_COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(BK_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(KPE_COL_IS_REAL));
        assert_eq!(d2.a_selector_column, Some(VH_COL_IS_REAL));
        assert_eq!(d2.b_selector_column, Some(SE_COL_IS_REAL));

        // Witness-level honest-closure sanity at row 0 across all 4 traces.
        let curve = CurveType::Bls48581;
        let vh_w = build_vh_witness();
        let bk_w = build_bk_witness();
        let kpe_w = build_kpe_witness();
        let se_w = build_se_witness();

        let trace_0 = build_vh_trace(&vh_w, curve);
        let trace_1 = build_bk_trace(&bk_w, curve);
        let trace_2 = build_kpe_trace(&kpe_w, curve);
        let trace_3 = build_se_trace(&se_w, curve);

        let cs_0 = VersionedHashConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BlobKzgConstraintSystem::new(trace_1.num_rows);
        let cs_2 = KzgPointEvalConstraintSystem::new(trace_2.num_rows);
        let cs_3 = Sha256ExtractConstraintSystem::new(trace_3.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
        ];
        let linkages = vec![d0.clone(), d1.clone(), d2.clone()];

        assert_eq!(traces.len(), 4, "4-AIR joint_prove input must have 4 traces");
        assert_eq!(linkages.len(), 3, "must wire exactly 3 descriptors");

        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        let vh_cols = &trace_0.columns;
        let bk_cols = &trace_1.columns;
        let kpe_cols = &trace_2.columns;
        let se_cols = &trace_3.columns;

        // D0: VH commitment[0] ↔ BK commitment_bytes[0].
        assert_eq!(
            vh_cols[VH_COL_COMMITMENT_OFFSET].evaluations[0].to_bytes(),
            bk_cols[BK_COL_COMMITMENT_OFFSET].evaluations[0].to_bytes(),
            "honest D0 commitment-byte tuple must match on the real row",
        );

        // D1: BK commitment_bytes[0] ↔ KPE commitment[0].
        assert_eq!(
            bk_cols[BK_COL_COMMITMENT_OFFSET].evaluations[0].to_bytes(),
            kpe_cols[KPE_COL_COMMITMENT_OFFSET].evaluations[0].to_bytes(),
            "honest D1 commitment-byte tuple must match on the real row",
        );

        // D2: VH versioned_hash[0] ↔ SE output_byte[0], both = 0x01.
        let expected_magic = Scalar::from_u64(VERSIONED_HASH_VERSION_KZG as u64, curve);
        assert_eq!(
            vh_cols[VH_COL_VERSIONED_HASH_OFFSET].evaluations[0].to_bytes(),
            expected_magic.to_bytes(),
            "VH versioned_hash byte 0 must equal the EIP-4844 magic 0x01",
        );
        assert_eq!(
            se_cols[SE_COL_OUTPUT_OFFSET].evaluations[0].to_bytes(),
            expected_magic.to_bytes(),
            "SE output byte 0 must equal the EIP-4844 magic 0x01 (synthetic)",
        );
        assert_eq!(
            vh_cols[VH_COL_VERSIONED_HASH_OFFSET].evaluations[0].to_bytes(),
            se_cols[SE_COL_OUTPUT_OFFSET].evaluations[0].to_bytes(),
            "honest D2 versioned-hash↔sha256-extract output tuple must match",
        );

        // Selectors all set on the real row.
        let one = Scalar::one(curve).to_bytes();
        assert_eq!(vh_cols[VH_COL_IS_REAL].evaluations[0].to_bytes(), one);
        assert_eq!(bk_cols[BK_COL_IS_REAL].evaluations[0].to_bytes(), one);
        assert_eq!(kpe_cols[KPE_COL_IS_REAL].evaluations[0].to_bytes(), one);
        assert_eq!(se_cols[SE_COL_IS_REAL].evaluations[0].to_bytes(), one);

        // Witness row count + VH row identifies cleanly.
        assert_eq!(vh_w.rows.len(), 1);
        let _: &VersionedHashRow = &vh_w.rows[0];

        // Pin the digest length to catch any layout drift.
        assert_eq!(VH_DIGEST_LEN, 32);
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 4-AIR `joint_prove` + `joint_verify` round-trip under
    /// BLS48-581 with 3 cross-AIR LogUp descriptors. `#[ignore]` because
    /// 4× per-AIR `prove_with_scheme` + 3× per-linkage SNARK + KZG opens
    /// is comfortably > 120 s in release on most hardware. Run via
    /// `cargo test --release --ignored
    /// honest_blob_full_chain_joint_verify_true`.
    #[test]
    #[ignore = "slow: 4-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_blob_full_chain_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let vh_w = build_vh_witness();
        let bk_w = build_bk_witness();
        let kpe_w = build_kpe_witness();
        let se_w = build_se_witness();

        let trace_0 = build_vh_trace(&vh_w, curve);
        let trace_1 = build_bk_trace(&bk_w, curve);
        let trace_2 = build_kpe_trace(&kpe_w, curve);
        let trace_3 = build_se_trace(&se_w, curve);

        // Pre-wire CS omega/domain to the joint_prove auto-inflation
        // target (LogUp boost → 256). Per #278: every AIR here declares
        // 8-bit byte range lookups, so each is boosted to
        // max(padded, RANGE_TABLE_SIZE=256); target_padded = max across
        // all AIRs.
        let target_padded: u64 = [
            trace_0.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_1.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_2.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_3.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
        ]
        .into_iter()
        .max()
        .unwrap();
        let omega = scheme.domain_generator(target_padded);

        let cs_0 = VersionedHashConstraintSystem::new(trace_0.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_1 = BlobKzgConstraintSystem::new(trace_1.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_2 = KzgPointEvalConstraintSystem::new(trace_2.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_3 = Sha256ExtractConstraintSystem::new(trace_3.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
        ];
        let linkages = vec![
            d0_commitment_byte0_vh_to_bk(),
            d1_commitment_byte0_bk_to_kpe(),
            d2_versioned_hash_byte0_vh_to_sha256_extract(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 4-AIR joint_prove must succeed for blob full chain");

        assert_eq!(proofs.len(), 4, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            3,
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
            vec![&cs_0, &cs_1, &cs_2, &cs_3];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 4-AIR joint_verify must accept matching tuples on all 3 descriptors",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tampered closure: overwrite `closure_a` on descriptor D2 (the
    /// versioned-hash ↔ sha256-extract linkage) after a successful
    /// `joint_prove`. Verifier must reject.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_blob_full_chain_joint_verify_true"]
    fn tampered_blob_full_chain_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let vh_w = build_vh_witness();
        let bk_w = build_bk_witness();
        let kpe_w = build_kpe_witness();
        let se_w = build_se_witness();

        let trace_0 = build_vh_trace(&vh_w, curve);
        let trace_1 = build_bk_trace(&bk_w, curve);
        let trace_2 = build_kpe_trace(&kpe_w, curve);
        let trace_3 = build_se_trace(&se_w, curve);

        let target_padded: u64 = [
            trace_0.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_1.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_2.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
            trace_3.padded_size.max(crate::lookup::RANGE_TABLE_SIZE as u64),
        ]
        .into_iter()
        .max()
        .unwrap();
        let omega = scheme.domain_generator(target_padded);

        let cs_0 = VersionedHashConstraintSystem::new(trace_0.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_1 = BlobKzgConstraintSystem::new(trace_1.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_2 = KzgPointEvalConstraintSystem::new(trace_2.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);
        let cs_3 = Sha256ExtractConstraintSystem::new(trace_3.num_rows)
            .with_omega_and_domain(omega.clone(), target_padded);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&trace_0, &cs_0),
            (&trace_1, &cs_1),
            (&trace_2, &cs_2),
            (&trace_3, &cs_3),
        ];
        let linkages = vec![
            d0_commitment_byte0_vh_to_bk(),
            d1_commitment_byte0_bk_to_kpe(),
            d2_versioned_hash_byte0_vh_to_sha256_extract(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 4-AIR joint_prove must succeed for blob full chain");

        // Tamper with descriptor D2's closure_a.
        let two_bytes = Scalar::from_u64(2, curve).to_bytes();
        ext.linkage_proofs[2].closure_a = two_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2, &cs_3];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the versioned-hash ↔ sha256-extract descriptor",
        );
    }
}
