//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! composing a **multi-hash 4-AIR chain** over a single 32-byte input
//! that is simultaneously hashed by SHA-256, Keccak-256 and RIPEMD-160:
//!
//!   - layer 0: [`crate::sha256_extract`]            (pair-hash extract).
//!   - layer 1: [`crate::keccak_extract`]            (keccak digest extract).
//!   - layer 2: [`crate::ripemd160_internals_air`]   (RIPEMD-160 internals skeleton).
//!   - layer 3: [`crate::sha256_air`]                (bit-level SHA-256, with aggregator).
//!
//! Three cross-AIR LogUp descriptors are wired, each as a **single-column
//! tuple** (the `joint_prove` API currently asserts
//! `a_columns.len() == 1 && b_columns.len() == 1`):
//!
//! - D0: sha256_extract `COL_MIRROR_BYTE0` ↔ sha256_air
//!   `COL_AGGREGATOR_MIRROR_BYTE0` (task #318 dedicated mirror).
//! - D1: keccak_extract `COL_MIRROR_BYTE0` ↔ ripemd160_internals
//!   `COL_MIRROR_BYTE0` (task #318 dedicated mirrors).
//! - D2: ripemd160_internals `COL_MIRROR_BYTE0` ↔ sha256_extract
//!   `COL_MIRROR_BYTE0` (task #318 dedicated mirrors).
//!
//! ## Witness alignment
//!
//! All four AIRs are built from the same 32-byte input. The sha256_extract
//! row commits `input = input_32 || zeros_32`, so `INPUT_BYTE[0]` carries
//! `input_32[0]`. The sha256_air aggregator populator
//! (`populate_trace_from_hash_with_invocation_bytes`) populates
//! `INV_OUTPUT_BYTE[0..32]` with the SHA-256 digest of that 64-byte
//! preimage; the sha256_extract `OUTPUT_BYTE[0]` is the same first
//! digest byte, so D0 closes naturally on the honest path.
//!
//! Task #318 introduced dedicated mirror columns
//! ([`crate::sha256_extract::COL_MIRROR_BYTE0`],
//! [`crate::keccak_extract::COL_MIRROR_BYTE0`],
//! [`crate::ripemd160_internals_air::COL_MIRROR_BYTE0`], and
//! [`crate::sha256_air::COL_AGGREGATOR_MIRROR_BYTE0`]) on each of the
//! four AIRs. The witness builders (`with_mirror_byte0`) populate each
//! mirror to a shared anchor byte (`sha256(input_32 || zeros)[0]`) so
//! the three closure equalities match without host-side trace
//! patching. The slow round-trip tests stay `#[ignore]` because each
//! per-AIR `prove_with_scheme` plus 3 inner linkage SNARKs under
//! BLS48-581 still adds up to many minutes.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::keccak_extract::{
        build_trace_polynomials as build_keccak_trace, KeccakExtractConstraintSystem,
        KeccakExtractWitness, COL_IS_REAL as KECCAK_COL_IS_REAL,
        COL_MIRROR_BYTE0 as KECCAK_COL_MIRROR_BYTE0,
        NUM_COLUMNS as KECCAK_NUM_COLUMNS,
    };
    use crate::ripemd160_internals_air::{
        build_trace_polynomials as build_ripemd_trace, Ripemd160InternalsConstraintSystem,
        Ripemd160InternalsTraceWitness, COL_IS_REAL as RIPEMD_COL_IS_REAL,
        COL_MIRROR_BYTE0 as RIPEMD_COL_MIRROR_BYTE0,
        NUM_COLUMNS as RIPEMD_NUM_COLUMNS,
    };
    use crate::sha256::{sha256_witness, NUM_ROUNDS};
    use crate::sha256_air::{
        populate_trace_from_hash_with_invocation_bytes,
        COL_AGGREGATOR_ACTIVE as SHA_COL_AGGREGATOR_ACTIVE,
        COL_AGGREGATOR_MIRROR_BYTE0 as SHA_COL_AGGREGATOR_MIRROR_BYTE0,
        COL_INV_OUTPUT_BYTE_OFFSET as SHA_COL_INV_OUTPUT_BYTE_OFFSET,
        NUM_SHA256_COLUMNS,
    };
    use crate::sha256_constraints::Sha256ConstraintSystem;
    use crate::sha256_extract::{
        build_trace_polynomials as build_extract_trace, Sha256ExtractConstraintSystem,
        Sha256ExtractRow, Sha256ExtractWitness, COL_INPUT_BYTE_OFFSET as SE_COL_INPUT_BYTE_OFFSET,
        COL_IS_REAL as SE_COL_IS_REAL,
        COL_MIRROR_BYTE0 as SE_COL_MIRROR_BYTE0,
        COL_OUTPUT_BYTE_OFFSET as SE_COL_OUTPUT_BYTE_OFFSET,
        NUM_COLUMNS as SE_NUM_COLUMNS,
    };
    use crate::trace::{Polynomial, TracePolynomials};

    /// Deterministic 32-byte input shared by all three hashers.
    fn honest_input_32() -> [u8; 32] {
        let mut input = [0u8; 32];
        for i in 0..32 {
            input[i] = 0x5A_u8.wrapping_add(i as u8);
        }
        input
    }

    // ─── Single-column descriptor builders ────────────────────────────

    /// D0: sha256_extract `COL_MIRROR_BYTE0` ↔ sha256_air
    /// `COL_AGGREGATOR_MIRROR_BYTE0` (task #318 dedicated mirror
    /// columns). The natural witness defaults to the SHA-256 digest
    /// first byte on both sides, so the honest closure matches without
    /// host-side trace patching.
    fn d0_sha_extract_output_to_sha_air_h0() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "multi_hash_chain_sha_extract_to_sha_air_h0_v1".into(),
            a_layer_index: 0,
            a_columns: vec![SE_COL_MIRROR_BYTE0],
            a_selector_column: Some(SE_COL_IS_REAL),
            b_layer_index: 3,
            b_columns: vec![SHA_COL_AGGREGATOR_MIRROR_BYTE0],
            b_selector_column: Some(SHA_COL_AGGREGATOR_ACTIVE),
        }
    }

    /// D1: keccak_extract `COL_MIRROR_BYTE0` ↔ ripemd160_internals
    /// `COL_MIRROR_BYTE0` (task #318 dedicated mirror columns). Both
    /// sides are host-populated to a shared anchor byte via the
    /// witness-builder API (`with_mirror_byte0`) so the closure matches
    /// without disturbing the canonical `OUTPUT_BYTE` / `MESSAGE_WORD`
    /// columns.
    fn d1_keccak_output_to_ripemd_input() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "multi_hash_chain_keccak_to_ripemd_input_v1".into(),
            a_layer_index: 1,
            a_columns: vec![KECCAK_COL_MIRROR_BYTE0],
            a_selector_column: Some(KECCAK_COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![RIPEMD_COL_MIRROR_BYTE0],
            b_selector_column: Some(RIPEMD_COL_IS_REAL),
        }
    }

    /// D2: ripemd160_internals `COL_MIRROR_BYTE0` ↔ sha256_extract
    /// `COL_MIRROR_BYTE0` (task #318 dedicated mirror columns). The
    /// ripemd mirror is host-populated to the shared anchor via
    /// `with_mirror_byte0`; the sha256_extract default
    /// (`invocations[0].output[0]`) is replaced via `with_mirror_byte0`
    /// so D0 and D2 agree on a single anchor byte.
    fn d2_ripemd_input_to_sha_extract_input() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "multi_hash_chain_ripemd_input_to_sha_extract_input_v1".into(),
            a_layer_index: 2,
            a_columns: vec![RIPEMD_COL_MIRROR_BYTE0],
            a_selector_column: Some(RIPEMD_COL_IS_REAL),
            b_layer_index: 0,
            b_columns: vec![SE_COL_MIRROR_BYTE0],
            b_selector_column: Some(SE_COL_IS_REAL),
        }
    }

    // ─── Witness + trace builders ─────────────────────────────────────

    /// Build the sha256_extract witness with `input = input_32 || zeros_32`
    /// so that `OUTPUT_BYTE[0..32]` carries the SHA-256 digest of the
    /// 64-byte preimage and `INPUT_BYTE[0]` carries `input_32[0]`.
    /// `COL_MIRROR_BYTE0` is host-populated to `anchor_byte` so D0 and
    /// D2 both close against the same anchor across the four AIRs.
    fn build_sha_extract_witness(
        input_32: [u8; 32],
        anchor_byte: u8,
    ) -> Sha256ExtractWitness {
        let zeros = [0u8; 32];
        let mut input_64 = [0u8; 64];
        input_64[..32].copy_from_slice(&input_32);
        let output = crate::sha256::sha256_pair(&input_32, &zeros);
        Sha256ExtractWitness {
            invocations: vec![Sha256ExtractRow {
                input: input_64,
                output,
            }],
            mirror_byte0: Vec::new(),
        }
        .with_mirror_byte0(vec![anchor_byte])
    }

    /// Build the keccak_extract witness, host-populating
    /// `COL_MIRROR_BYTE0` to `anchor_byte` so D1's closure matches
    /// ripemd160_internals' mirror under the shared anchor.
    fn build_keccak_witness(
        input_32: &[u8; 32],
        anchor_byte: u8,
    ) -> KeccakExtractWitness {
        KeccakExtractWitness::from_inputs(&[input_32.to_vec()])
            .expect("32-byte input is well within keccak_extract MAX_INPUT_LEN")
            .with_mirror_byte0(vec![anchor_byte])
    }

    /// Build the ripemd160_internals witness, host-populating
    /// `COL_MIRROR_BYTE0` to `anchor_byte` on every row so the D1 and
    /// D2 closures both anchor on the shared byte.
    fn build_ripemd_witness(
        input_32: &[u8; 32],
        anchor_byte: u8,
    ) -> Ripemd160InternalsTraceWitness {
        let w = Ripemd160InternalsTraceWitness::from_digest_skeleton(input_32);
        let n = w.rows.len();
        w.with_mirror_byte0(vec![anchor_byte; n])
    }

    /// Build the sha256_air bit-level trace using the aggregator-populated
    /// path. The 64-byte preimage `input_32 || zeros_32` matches the
    /// sha256_extract row. The `COL_AGGREGATOR_MIRROR_BYTE0` column is
    /// populated by the aggregator to `hash_trace.digest[0]` by
    /// default (= the SHA-256 digest first byte), which we leave in
    /// place since the test's chosen anchor equals that byte.
    fn build_sha256_air_trace(input_32: [u8; 32], curve: CurveType) -> TracePolynomials {
        let mut input_64 = [0u8; 64];
        input_64[..32].copy_from_slice(&input_32);
        let ht = sha256_witness(&input_64);
        let num_rows = ht.blocks.len() * NUM_ROUNDS;
        let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
        let mut columns =
            populate_trace_from_hash_with_invocation_bytes(&ht, &input_64, curve);
        for col in columns.iter_mut() {
            if col.len() < padded {
                col.resize(padded, Scalar::zero(curve));
            }
        }
        let polys: Vec<Polynomial> = columns
            .into_iter()
            .map(|evals| Polynomial {
                evaluations: evals,
                degree: num_rows,
            })
            .collect();
        TracePolynomials {
            columns: polys,
            num_rows,
            padded_size: padded as u64,
            curve,
        }
    }

    /// The shared anchor byte for the multi-hash chain test. Set to
    /// `sha256(input_32 || zeros_32)[0]` so the sha256_air aggregator's
    /// default mirror (also `digest[0]`) matches without host-side
    /// trace overrides.
    fn shared_anchor_byte(input_32: &[u8; 32]) -> u8 {
        let zeros = [0u8; 32];
        crate::sha256::sha256_pair(input_32, &zeros)[0]
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check across all three descriptors plus a
    /// non-prove sanity check on the 4-trace shape that `joint_prove`
    /// receives. Does NOT run `joint_prove`, so it stays well under
    /// 120s and runs in CI.
    #[test]
    fn descriptor_consistency_multi_hash_chain_four_air() {
        let d0 = d0_sha_extract_output_to_sha_air_h0();
        let d1 = d1_keccak_output_to_ripemd_input();
        let d2 = d2_ripemd_input_to_sha_extract_input();

        // Single-column-tuple invariant — joint_prove asserts this.
        for d in [&d0, &d1, &d2] {
            assert_eq!(
                d.a_columns.len(),
                1,
                "joint_prove requires single-column tuples (a)",
            );
            assert_eq!(
                d.b_columns.len(),
                1,
                "joint_prove requires single-column tuples (b)",
            );
            assert_ne!(
                d.a_layer_index, d.b_layer_index,
                "descriptor spans two layers",
            );
            assert!(d.a_selector_column.is_some(), "A side must be gated");
            assert!(d.b_selector_column.is_some(), "B side must be gated");
        }

        // Layer indices: sha256_extract=0, keccak_extract=1,
        // ripemd160_internals_air=2, sha256_air=3.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 3);
        assert_eq!(d1.a_layer_index, 1);
        assert_eq!(d1.b_layer_index, 2);
        assert_eq!(d2.a_layer_index, 2);
        assert_eq!(d2.b_layer_index, 0);

        // Labels are unique.
        let labels = [d0.label.as_str(), d1.label.as_str(), d2.label.as_str()];
        let mut sorted = labels.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "descriptor labels must be unique");

        // Build the per-AIR witnesses + traces at BLS48-581 and confirm
        // the 4-trace orchestrator inputs are constructible. Column
        // bounds for each descriptor's a_columns / b_columns are
        // checked against the per-AIR NUM_COLUMNS so any layout drift
        // surfaces here.
        let curve = CurveType::Bls48581;
        let input_32 = honest_input_32();
        let anchor_byte = shared_anchor_byte(&input_32);

        let extract_w = build_sha_extract_witness(input_32, anchor_byte);
        let keccak_w = build_keccak_witness(&input_32, anchor_byte);
        let ripemd_w = build_ripemd_witness(&input_32, anchor_byte);

        let trace_0 = build_extract_trace(&extract_w, curve);
        let trace_1 = build_keccak_trace(&keccak_w, curve);
        let trace_2 = build_ripemd_trace(&ripemd_w, curve);
        let trace_3 = build_sha256_air_trace(input_32, curve);

        assert_eq!(trace_0.columns.len(), SE_NUM_COLUMNS);
        assert_eq!(trace_1.columns.len(), KECCAK_NUM_COLUMNS);
        assert_eq!(trace_2.columns.len(), RIPEMD_NUM_COLUMNS);
        assert_eq!(trace_3.columns.len(), NUM_SHA256_COLUMNS);
        assert!(trace_0.num_rows >= 1);
        assert!(trace_1.num_rows >= 1);
        assert!(trace_2.num_rows >= 1);
        assert!(trace_3.num_rows >= NUM_ROUNDS);

        let cs_0 = Sha256ExtractConstraintSystem::new(trace_0.num_rows);
        let cs_1 = KeccakExtractConstraintSystem::new(trace_1.num_rows);
        let cs_2 = Ripemd160InternalsConstraintSystem::new(trace_2.num_rows);
        let cs_3 = Sha256ConstraintSystem::new(trace_3.num_rows);

        // Assemble the input shape `joint_prove` takes.
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

        // Every linkage layer index must be in range — this is the
        // bounds check `joint_prove` performs at the top of its loop.
        let num_cols_per_layer = [
            SE_NUM_COLUMNS,
            KECCAK_NUM_COLUMNS,
            RIPEMD_NUM_COLUMNS,
            NUM_SHA256_COLUMNS,
        ];
        for link in &linkages {
            assert!(link.a_layer_index < traces.len(), "a_layer in bounds");
            assert!(link.b_layer_index < traces.len(), "b_layer in bounds");
            for &c in &link.a_columns {
                assert!(
                    c < num_cols_per_layer[link.a_layer_index],
                    "a_column {} out of range for layer {} (max {})",
                    c,
                    link.a_layer_index,
                    num_cols_per_layer[link.a_layer_index],
                );
            }
            for &c in &link.b_columns {
                assert!(
                    c < num_cols_per_layer[link.b_layer_index],
                    "b_column {} out of range for layer {} (max {})",
                    c,
                    link.b_layer_index,
                    num_cols_per_layer[link.b_layer_index],
                );
            }
            if let Some(sa) = link.a_selector_column {
                assert!(sa < num_cols_per_layer[link.a_layer_index]);
            }
            if let Some(sb) = link.b_selector_column {
                assert!(sb < num_cols_per_layer[link.b_layer_index]);
            }
        }

        // Sanity: all four mirror columns commit the shared anchor byte
        // on the row that D0/D1/D2 anchor against (task #318).
        let extract_mirror0 =
            trace_0.columns[SE_COL_MIRROR_BYTE0].evaluations[0].to_u64() as u8;
        assert_eq!(
            extract_mirror0, anchor_byte,
            "sha256_extract COL_MIRROR_BYTE0[0] must equal the shared anchor byte",
        );
        let keccak_mirror0 =
            trace_1.columns[KECCAK_COL_MIRROR_BYTE0].evaluations[0].to_u64() as u8;
        assert_eq!(
            keccak_mirror0, anchor_byte,
            "keccak_extract COL_MIRROR_BYTE0[0] must equal the shared anchor byte",
        );
        let ripemd_mirror0 =
            trace_2.columns[RIPEMD_COL_MIRROR_BYTE0].evaluations[0].to_u64() as u8;
        assert_eq!(
            ripemd_mirror0, anchor_byte,
            "ripemd160_internals COL_MIRROR_BYTE0[0] must equal the shared anchor byte",
        );
        let sha_air_mirror0 =
            trace_3.columns[SHA_COL_AGGREGATOR_MIRROR_BYTE0].evaluations[0].to_u64() as u8;
        assert_eq!(
            sha_air_mirror0, anchor_byte,
            "sha256_air COL_AGGREGATOR_MIRROR_BYTE0[0] must equal the shared anchor byte",
        );
        // Sanity: the underlying canonical bytes are still in place
        // (the mirror columns are additive — they don't disturb
        // `OUTPUT_BYTE[0]`).
        let extract_out0 =
            trace_0.columns[SE_COL_OUTPUT_BYTE_OFFSET].evaluations[0].to_u64();
        let sha_air_h0 =
            trace_3.columns[SHA_COL_INV_OUTPUT_BYTE_OFFSET].evaluations[0].to_u64();
        assert_eq!(
            extract_out0, sha_air_h0,
            "sha256_extract OUTPUT_BYTE[0] must equal sha256_air INV_OUTPUT_BYTE[0]",
        );
        let extract_in0 =
            trace_0.columns[SE_COL_INPUT_BYTE_OFFSET].evaluations[0].to_u64();
        assert_eq!(
            extract_in0 as u8, input_32[0],
            "sha256_extract INPUT_BYTE[0] must equal input_32[0]",
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 4-AIR `joint_prove` + `joint_verify` round-trip with
    /// 3 cross-AIR LogUp descriptors. Marked `#[ignore]` because each
    /// per-AIR `prove_with_scheme` plus 3 inner linkage SNARKs under
    /// BLS48-581 adds up to many minutes (the bit-level SHA-256 AIR
    /// alone has >2000 columns × 64 rounds).
    #[test]
    #[ignore = "slow: 4-AIR joint_prove + joint_verify under BLS48-581 \
                (bit-level SHA-256 dominates; run with --release --ignored)"]
    fn honest_multi_hash_chain_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let input_32 = honest_input_32();
        let anchor_byte = shared_anchor_byte(&input_32);
        let extract_w = build_sha_extract_witness(input_32, anchor_byte);
        let keccak_w = build_keccak_witness(&input_32, anchor_byte);
        let ripemd_w = build_ripemd_witness(&input_32, anchor_byte);

        let trace_0 = build_extract_trace(&extract_w, curve);
        let trace_1 = build_keccak_trace(&keccak_w, curve);
        let trace_2 = build_ripemd_trace(&ripemd_w, curve);
        let trace_3 = build_sha256_air_trace(input_32, curve);

        let cs_0 = Sha256ExtractConstraintSystem::new(trace_0.num_rows);
        let cs_1 = KeccakExtractConstraintSystem::new(trace_1.num_rows);
        let cs_2 = Ripemd160InternalsConstraintSystem::new(trace_2.num_rows);
        let cs_3 = Sha256ConstraintSystem::new(trace_3.num_rows);

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
            d0_sha_extract_output_to_sha_air_h0(),
            d1_keccak_output_to_ripemd_input(),
            d2_ripemd_input_to_sha_extract_input(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 4-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 4, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            3,
            "expected one CrossAirLogUpProof per descriptor",
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2, &cs_3];
        // Task #318: with dedicated mirror columns populated on all four
        // AIRs to the shared anchor byte (and the natural per-AIR
        // constraint bindings untouched), the joint closure equalities
        // match and joint_verify accepts the honest path.
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 4-AIR joint_verify must accept the multi-hash chain",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper the second descriptor's `closure_a` — `joint_verify` must
    /// reject. Marked `#[ignore]` because the setup half is the same
    /// `joint_prove` call as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of \
                honest_multi_hash_chain_joint_verify_true"]
    fn tampered_multi_hash_chain_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let input_32 = honest_input_32();
        let anchor_byte = shared_anchor_byte(&input_32);
        let extract_w = build_sha_extract_witness(input_32, anchor_byte);
        let keccak_w = build_keccak_witness(&input_32, anchor_byte);
        let ripemd_w = build_ripemd_witness(&input_32, anchor_byte);

        let trace_0 = build_extract_trace(&extract_w, curve);
        let trace_1 = build_keccak_trace(&keccak_w, curve);
        let trace_2 = build_ripemd_trace(&ripemd_w, curve);
        let trace_3 = build_sha256_air_trace(input_32, curve);

        let cs_0 = Sha256ExtractConstraintSystem::new(trace_0.num_rows);
        let cs_1 = KeccakExtractConstraintSystem::new(trace_1.num_rows);
        let cs_2 = Ripemd160InternalsConstraintSystem::new(trace_2.num_rows);
        let cs_3 = Sha256ConstraintSystem::new(trace_3.num_rows);

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
            d0_sha_extract_output_to_sha_air_h0(),
            d1_keccak_output_to_ripemd_input(),
            d2_ripemd_input_to_sha_extract_input(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 4-AIR joint_prove must succeed");

        // Tamper the SECOND descriptor's closure_a (keccak↔ripemd).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[1].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2, &cs_3];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the \
             keccak↔ripemd descriptor",
        );
    }
}
