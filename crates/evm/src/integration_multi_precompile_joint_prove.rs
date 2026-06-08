//! 4-AIR cross-AIR LogUp `joint_prove` / `joint_verify` integration
//! test composing the EVM precompile dispatch + IO scaffold with two
//! per-precompile gadget AIRs:
//!
//!   - layer 0: [`crate::precompile_air`] — dispatch AIR holding one
//!     row per precompile invocation observed in the EVM trace, with
//!     the appropriate `sel_*` selector set per row.
//!   - layer 1: [`crate::precompile_io_air`] — single-row-per-call IO
//!     scaffold; today only `Identity` and `Sha256` rows are supported
//!     by this AIR (`MAX_INPUT_LENGTH = 64`, two selector columns).
//!   - layer 2: [`metavm_zkp::ecrecover_chain_air`] — the per-row
//!     ECRECOVER (callee `0x01`) gadget AIR.
//!   - layer 3: [`metavm_zkp::ripemd160_precompile_air`] — the per-row
//!     RIPEMD-160 (callee `0x03`) gadget AIR.
//!
//! The witness commits **four** precompile dispatches in one logical
//! EVM trace:
//!
//!   - row 0: `SHA256`     (callee `0x02`)
//!   - row 1: `RIPEMD160`  (callee `0x03`)
//!   - row 2: `IDENTITY`   (callee `0x04`)
//!   - row 3: `ECRECOVER`  (callee `0x01`)
//!
//! Each row sets a distinct selector on `precompile_air` (the AIR's
//! `sel_sum_eq_is_real` constraint enforces one-hot selection). The
//! `precompile_io_air` only carries `SHA256` + `IDENTITY` rows because
//! today it does not commit selectors for the other precompiles
//! (`MAX_INPUT_LENGTH = 64`, `sel_identity`/`sel_sha256` only). The
//! `ecrecover_chain_air` and `ripemd160_precompile_air` each carry
//! exactly one real row matching their respective dispatch.
//!
//! ## Cross-AIR LogUp descriptors (4 total, all single-column tuples)
//!
//!   - **D0** (`ecrecover ↔ dispatch`): layer 2 `COL_IS_REAL` gated by
//!     `COL_IS_REAL` ↔ layer 0 `COL_SEL_ECRECOVER` gated by
//!     `COL_SEL_ECRECOVER`. Both sides commit a multiset `{1}` on
//!     their one ECRECOVER row.
//!   - **D1** (`ripemd  ↔ dispatch`): layer 3 `COL_IS_REAL` gated by
//!     `COL_IS_REAL` ↔ layer 0 `COL_SEL_RIPEMD` gated by
//!     `COL_SEL_RIPEMD`. Both sides commit `{1}` on their one RIPEMD
//!     row.
//!   - **D2** (`io_sha256 ↔ dispatch_sha256`): layer 1 `COL_SEL_SHA256`
//!     gated by `COL_SEL_SHA256` ↔ layer 0 `COL_SEL_SHA256` gated by
//!     `COL_SEL_SHA256`. Both `{1}`.
//!   - **D3** (`io_identity ↔ dispatch_identity`): layer 1
//!     `COL_SEL_IDENTITY` gated by `COL_SEL_IDENTITY` ↔ layer 0
//!     `COL_SEL_IDENTITY` gated by `COL_SEL_IDENTITY`. Both `{1}`.
//!
//! With identical anchor values `{1}` on each side of every descriptor
//! the running-sum closures match by construction, so the joint
//! verifier's per-descriptor `closure_a == closure_b` checks accept
//! and the orchestrator's per-AIR proofs validate independently.
//!
//! ## Cross-references
//!
//! - `crate::evm_main_sha3_joint_prove` — canonical 2-AIR template
//!   used as the structural anchor for this file.
//! - `metavm_zkp::integration_joint_prove_three_air` — 3-AIR
//!   multi-descriptor template demonstrating the shared-middle-layer
//!   pattern this file generalises to four layers and four
//!   descriptors.
//! - `crate::precompile_evm_linkages` — EVM-side descriptor builders
//!   wrapping the zkp-side stub patterns; this test uses bespoke
//!   single-column wrappers (the wider tuples in
//!   `precompile_evm_linkages` are kept for future multi-column
//!   `joint_prove` extensions).
//!
//! The fast `descriptor_consistency` test validates well-formedness +
//! trace construction shape without invoking the prover; the two
//! `#[ignore]`-gated slow tests run the honest 4-AIR round-trip and a
//! tampered-closure rejection under BLS48-581.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::precompile_air as pa;
use crate::precompile_io_air as pio;
use metavm_zkp::ecrecover_chain_air as ecr;
use metavm_zkp::ripemd160_precompile_air as ripemd;

// ─── Descriptor builders ──────────────────────────────────────────────

/// D0: ECRECOVER chain `COL_IS_REAL` ↔ dispatch `COL_SEL_ECRECOVER`.
pub fn make_ecrecover_to_dispatch_descriptor(
    ecrecover_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "multi_pc_ecrecover_to_dispatch_v1".into(),
        a_layer_index: ecrecover_layer_index,
        a_columns: vec![ecr::COL_IS_REAL],
        a_selector_column: Some(ecr::COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns: vec![pa::COL_SEL_ECRECOVER],
        b_selector_column: Some(pa::COL_SEL_ECRECOVER),
    }
}

/// D1: RIPEMD-160 `COL_IS_REAL` ↔ dispatch `COL_SEL_RIPEMD`.
pub fn make_ripemd_to_dispatch_descriptor(
    ripemd_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "multi_pc_ripemd_to_dispatch_v1".into(),
        a_layer_index: ripemd_layer_index,
        a_columns: vec![ripemd::COL_IS_REAL],
        a_selector_column: Some(ripemd::COL_IS_REAL),
        b_layer_index: precompile_layer_index,
        b_columns: vec![pa::COL_SEL_RIPEMD],
        b_selector_column: Some(pa::COL_SEL_RIPEMD),
    }
}

/// D2: precompile_io_air `COL_SEL_SHA256` ↔ dispatch `COL_SEL_SHA256`.
pub fn make_io_sha256_to_dispatch_descriptor(
    precompile_io_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "multi_pc_io_sha256_to_dispatch_v1".into(),
        a_layer_index: precompile_io_layer_index,
        a_columns: vec![pio::COL_SEL_SHA256],
        a_selector_column: Some(pio::COL_SEL_SHA256),
        b_layer_index: precompile_layer_index,
        b_columns: vec![pa::COL_SEL_SHA256],
        b_selector_column: Some(pa::COL_SEL_SHA256),
    }
}

/// D3: precompile_io_air `COL_SEL_IDENTITY` ↔ dispatch
/// `COL_SEL_IDENTITY`.
pub fn make_io_identity_to_dispatch_descriptor(
    precompile_io_layer_index: usize,
    precompile_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "multi_pc_io_identity_to_dispatch_v1".into(),
        a_layer_index: precompile_io_layer_index,
        a_columns: vec![pio::COL_SEL_IDENTITY],
        a_selector_column: Some(pio::COL_SEL_IDENTITY),
        b_layer_index: precompile_layer_index,
        b_columns: vec![pa::COL_SEL_IDENTITY],
        b_selector_column: Some(pa::COL_SEL_IDENTITY),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    use crate::precompile_air::{
        build_trace_polynomials as build_pa_trace, expected_gas_cost,
        PrecompileConstraintSystem, PrecompileTraceWitness, PrecompileWitness, PC_ECRECOVER,
        PC_IDENTITY, PC_RIPEMD160, PC_SHA256,
    };
    use crate::precompile_io_air::{
        build_trace_polynomials as build_pio_trace, from_call as pio_from_call,
        PrecompileIoConstraintSystem, PrecompileIoTraceWitness, PrecompileKind,
    };
    use metavm_zkp::ecrecover_chain_air::{
        build_trace_polynomials as build_ecr_trace, EcrecoverChainConstraintSystem,
        EcrecoverChainWitness, INPUT_LEN as ECR_INPUT_LEN,
    };
    use metavm_zkp::ripemd160_precompile_air::{
        build_trace_polynomials as build_ripemd_trace, Ripemd160PrecompileConstraintSystem,
        Ripemd160PrecompileTraceWitness, Ripemd160PrecompileWitness,
    };

    // ─── Witness fixtures ──────────────────────────────────────────────

    /// A 4-byte input used uniformly for the SHA-256, IDENTITY, and
    /// RIPEMD-160 dispatches. Short enough to fit in a single
    /// `ceil(len/32) = 1` gas chunk and well below
    /// `precompile_io_air::MAX_INPUT_LENGTH`.
    const HASHABLE_INPUT: [u8; 4] = [0xde, 0xad, 0xbe, 0xef];

    /// 128-byte zero buffer fed to ECRECOVER. With `v_byte = 0` (since
    /// `input[63] = 0`) this fails the `v ∈ {27, 28}` check and
    /// produces an `is_valid = 0` row. `is_real` is still 1, which is
    /// what descriptor D0 binds, so the closure equality holds.
    const ECRECOVER_INPUT: [u8; ECR_INPUT_LEN] = [0u8; ECR_INPUT_LEN];

    /// Build the layer-0 `precompile_air` trace witness with four
    /// distinct dispatch rows, one per supported callee.
    fn build_pa_witness() -> PrecompileTraceWitness {
        let sha_in_len = HASHABLE_INPUT.len() as u64;
        let identity_in_len = HASHABLE_INPUT.len() as u64;
        let ripemd_in_len = HASHABLE_INPUT.len() as u64;

        let rows = vec![
            PrecompileWitness {
                callee: [PC_SHA256, 0, 0, 0],
                precompile_id: PC_SHA256,
                input_length: sha_in_len,
                output_length: 32,
                gas_cost: expected_gas_cost(PC_SHA256, sha_in_len).unwrap(),
            },
            PrecompileWitness {
                callee: [PC_RIPEMD160, 0, 0, 0],
                precompile_id: PC_RIPEMD160,
                input_length: ripemd_in_len,
                output_length: 32,
                gas_cost: expected_gas_cost(PC_RIPEMD160, ripemd_in_len).unwrap(),
            },
            PrecompileWitness {
                callee: [PC_IDENTITY, 0, 0, 0],
                precompile_id: PC_IDENTITY,
                input_length: identity_in_len,
                // IDENTITY echoes its input: output_length == input_length.
                output_length: identity_in_len,
                gas_cost: expected_gas_cost(PC_IDENTITY, identity_in_len).unwrap(),
            },
            PrecompileWitness {
                callee: [PC_ECRECOVER, 0, 0, 0],
                precompile_id: PC_ECRECOVER,
                // ECRECOVER is always 128-byte input / 32-byte output / flat 3000 gas.
                input_length: ECR_INPUT_LEN as u64,
                output_length: 32,
                gas_cost: expected_gas_cost(PC_ECRECOVER, ECR_INPUT_LEN as u64).unwrap(),
            },
        ];
        PrecompileTraceWitness::from_rows(rows)
    }

    /// Build the layer-1 `precompile_io_air` witness with one SHA-256
    /// row + one IDENTITY row. RIPEMD-160 / ECRECOVER cannot be
    /// represented in today's IO AIR (which only commits `sel_sha256`
    /// + `sel_identity`), so they're carried only by their dedicated
    /// gadget AIRs in layers 2 and 3.
    fn build_pio_witness() -> PrecompileIoTraceWitness {
        let sha_out = metavm_zkp::sha256::sha256(&HASHABLE_INPUT);
        let sha_row = pio_from_call(
            PrecompileKind::Sha256,
            &HASHABLE_INPUT,
            &sha_out,
            0,
            32,
        );
        let identity_row = pio_from_call(
            PrecompileKind::Identity,
            &HASHABLE_INPUT,
            &HASHABLE_INPUT,
            0,
            32,
        );
        PrecompileIoTraceWitness::from_rows(vec![sha_row, identity_row])
    }

    /// Build the layer-2 `ecrecover_chain_air` witness with a single
    /// invocation. The input is all-zero, which fails the `v ∈ {27,
    /// 28}` check and produces an `is_valid = 0` failure row. The
    /// `is_real = 1` column is what descriptor D0 binds, so closure
    /// equality is preserved.
    fn build_ecr_witness() -> EcrecoverChainWitness {
        EcrecoverChainWitness::from_input(ECRECOVER_INPUT)
    }

    /// Build the layer-3 `ripemd160_precompile_air` witness with a
    /// single honest RIPEMD-160 invocation over `HASHABLE_INPUT`.
    fn build_ripemd_witness() -> Ripemd160PrecompileTraceWitness {
        let row = Ripemd160PrecompileWitness::from_input(&HASHABLE_INPUT);
        Ripemd160PrecompileTraceWitness::from_rows(vec![row])
    }

    // ─── Fast static check (un-ignored) ────────────────────────────────

    /// Static well-formedness + trace-shape sanity for the 4-AIR / 4-
    /// descriptor composition. Does **not** invoke `joint_prove`, so
    /// stays well under the 120s CI budget.
    #[test]
    fn descriptor_consistency() {
        // ─── Descriptor shape ──────────────────────────────────────
        let d0 = make_ecrecover_to_dispatch_descriptor(2, 0);
        let d1 = make_ripemd_to_dispatch_descriptor(3, 0);
        let d2 = make_io_sha256_to_dispatch_descriptor(1, 0);
        let d3 = make_io_identity_to_dispatch_descriptor(1, 0);

        for d in [&d0, &d1, &d2, &d3] {
            assert_eq!(
                d.a_columns.len(),
                1,
                "joint_prove currently requires single-column tuples for descriptor {:?}",
                d.label,
            );
            assert_eq!(d.b_columns.len(), 1);
            assert_ne!(
                d.a_layer_index, d.b_layer_index,
                "linkage layers must differ for descriptor {:?}",
                d.label,
            );
            assert!(d.a_selector_column.is_some());
            assert!(d.b_selector_column.is_some());
            // No sentinels — all column indices map to real AIR
            // columns.
            assert_ne!(d.a_columns[0], usize::MAX);
            assert_ne!(d.b_columns[0], usize::MAX);
        }

        // ─── Pin to AIR layout constants ───────────────────────────
        assert_eq!(d0.a_columns[0], ecr::COL_IS_REAL);
        assert_eq!(d0.b_columns[0], pa::COL_SEL_ECRECOVER);
        assert_eq!(d1.a_columns[0], ripemd::COL_IS_REAL);
        assert_eq!(d1.b_columns[0], pa::COL_SEL_RIPEMD);
        assert_eq!(d2.a_columns[0], pio::COL_SEL_SHA256);
        assert_eq!(d2.b_columns[0], pa::COL_SEL_SHA256);
        assert_eq!(d3.a_columns[0], pio::COL_SEL_IDENTITY);
        assert_eq!(d3.b_columns[0], pa::COL_SEL_IDENTITY);

        // Per-AIR column bounds.
        assert!(d0.b_columns[0] < pa::NUM_COLUMNS);
        assert!(d1.b_columns[0] < pa::NUM_COLUMNS);
        assert!(d2.b_columns[0] < pa::NUM_COLUMNS);
        assert!(d3.b_columns[0] < pa::NUM_COLUMNS);
        assert!(d2.a_columns[0] < pio::NUM_COLUMNS);
        assert!(d3.a_columns[0] < pio::NUM_COLUMNS);
        assert!(d0.a_columns[0] < ecr::NUM_COLUMNS);
        assert!(d1.a_columns[0] < ripemd::NUM_COLUMNS);

        // ─── Build all four witnesses + traces at BLS48-581 ────────
        let curve = CurveType::Bls48581;
        let pa_w = build_pa_witness();
        let pio_w = build_pio_witness();
        let ecr_w = build_ecr_witness();
        let ripemd_w = build_ripemd_witness();

        let pa_trace = build_pa_trace(&pa_w, curve);
        let pio_trace = build_pio_trace(&pio_w, curve);
        let ecr_trace = build_ecr_trace(&ecr_w, curve);
        let ripemd_trace = build_ripemd_trace(&ripemd_w, curve);

        let pa_cs = PrecompileConstraintSystem::new(pa_trace.num_rows);
        let pio_cs = PrecompileIoConstraintSystem::new(pio_trace.num_rows);
        let ecr_cs = EcrecoverChainConstraintSystem::new(ecr_trace.num_rows);
        let ripemd_cs = Ripemd160PrecompileConstraintSystem::new(ripemd_trace.num_rows);

        // ─── Sanity: each AIR commits its expected real rows ───────
        let one_bytes = Scalar::one(curve).to_bytes();
        assert_eq!(pa_trace.num_rows, 4, "precompile_air must hold 4 dispatch rows");
        assert_eq!(pio_trace.num_rows, 2, "precompile_io_air holds SHA256 + IDENTITY rows");
        assert_eq!(ecr_trace.num_rows, 1, "ecrecover_chain_air commits 1 row");
        assert_eq!(ripemd_trace.num_rows, 1, "ripemd160_precompile_air commits 1 row");

        // Confirm the 4 dispatch selectors are set as expected.
        let row_for = |sel: usize| -> Option<usize> {
            let col = &pa_trace.columns[sel].evaluations;
            (0..pa_trace.num_rows).find(|&r| col[r].to_bytes() == one_bytes)
        };
        let sha_row = row_for(pa::COL_SEL_SHA256).expect("SHA256 dispatch row missing");
        let ripemd_row = row_for(pa::COL_SEL_RIPEMD).expect("RIPEMD dispatch row missing");
        let identity_row = row_for(pa::COL_SEL_IDENTITY).expect("IDENTITY dispatch row missing");
        let ecrecover_row = row_for(pa::COL_SEL_ECRECOVER).expect("ECRECOVER dispatch row missing");

        // All four rows are distinct (one-hot over real rows).
        let mut seen = std::collections::HashSet::new();
        for r in [sha_row, ripemd_row, identity_row, ecrecover_row] {
            assert!(
                seen.insert(r),
                "two dispatch rows share the same row index ({})",
                r,
            );
        }

        // Confirm `is_real = 1` on each dispatch row.
        let is_real_pa = &pa_trace.columns[pa::COL_IS_REAL].evaluations;
        for r in [sha_row, ripemd_row, identity_row, ecrecover_row] {
            assert_eq!(
                is_real_pa[r].to_bytes(),
                one_bytes,
                "dispatch row {} must have is_real = 1",
                r,
            );
        }

        // ─── Assemble the 4-trace input shape `joint_prove` consumes ─
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&pa_trace, &pa_cs),
            (&pio_trace, &pio_cs),
            (&ecr_trace, &ecr_cs),
            (&ripemd_trace, &ripemd_cs),
        ];
        let linkages = vec![d0.clone(), d1.clone(), d2.clone(), d3.clone()];
        assert_eq!(traces.len(), 4, "4-AIR joint_prove input has 4 traces");
        assert_eq!(linkages.len(), 4, "4 cross-AIR LogUp descriptors total");

        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // Labels must not carry stub markers — these are concrete
        // bindings, not zkp-side placeholders.
        for d in &linkages {
            assert!(
                !d.label.contains("_stub"),
                "descriptor {:?} still carries a stub label",
                d.label,
            );
        }
    }

    // ─── Slow honest round-trip (ignored) ──────────────────────────────

    /// Full honest 4-AIR `joint_prove` + `joint_verify` round-trip
    /// across precompile_air + precompile_io_air + ecrecover_chain_air
    /// + ripemd160_precompile_air with 4 cross-AIR LogUp descriptors.
    ///
    /// Marked `#[ignore]` because `joint_prove` runs 4 per-AIR
    /// `prove_with_scheme` calls + 4 per-linkage `prove_with_scheme`
    /// calls + KZG opens, which together easily exceed the 120s CI
    /// budget under BLS48-581.
    #[test]
    #[ignore = "slow: 4-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_multi_precompile_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let pa_trace = build_pa_trace(&build_pa_witness(), curve);
        let pio_trace = build_pio_trace(&build_pio_witness(), curve);
        let ecr_trace = build_ecr_trace(&build_ecr_witness(), curve);
        let ripemd_trace = build_ripemd_trace(&build_ripemd_witness(), curve);

        let pa_cs = PrecompileConstraintSystem::new(pa_trace.num_rows);
        let pio_cs = PrecompileIoConstraintSystem::new(pio_trace.num_rows);
        let ecr_cs = EcrecoverChainConstraintSystem::new(ecr_trace.num_rows);
        let ripemd_cs = Ripemd160PrecompileConstraintSystem::new(ripemd_trace.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&pa_trace, &pa_cs),
            (&pio_trace, &pio_cs),
            (&ecr_trace, &ecr_cs),
            (&ripemd_trace, &ripemd_cs),
        ];
        let linkages = vec![
            make_ecrecover_to_dispatch_descriptor(2, 0),
            make_ripemd_to_dispatch_descriptor(3, 0),
            make_io_sha256_to_dispatch_descriptor(1, 0),
            make_io_identity_to_dispatch_descriptor(1, 0),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 4-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 4, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            4,
            "expected one CrossAirLogUpProof per descriptor",
        );

        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&pa_cs, &pio_cs, &ecr_cs, &ripemd_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 4-AIR joint_verify must accept matching tuples on all descriptors",
        );
    }

    // ─── Slow tampered round-trip (ignored) ────────────────────────────

    /// Tampered closure: corrupt `closure_a` on descriptor D2
    /// (`io_sha256 ↔ dispatch_sha256`) so the joint verifier's
    /// `closure_a == closure_b` scalar equality must reject.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_multi_precompile_joint_verify_true"]
    fn tampered_multi_precompile_joint_verify_false() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let pa_trace = build_pa_trace(&build_pa_witness(), curve);
        let pio_trace = build_pio_trace(&build_pio_witness(), curve);
        let ecr_trace = build_ecr_trace(&build_ecr_witness(), curve);
        let ripemd_trace = build_ripemd_trace(&build_ripemd_witness(), curve);

        let pa_cs = PrecompileConstraintSystem::new(pa_trace.num_rows);
        let pio_cs = PrecompileIoConstraintSystem::new(pio_trace.num_rows);
        let ecr_cs = EcrecoverChainConstraintSystem::new(ecr_trace.num_rows);
        let ripemd_cs = Ripemd160PrecompileConstraintSystem::new(ripemd_trace.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&pa_trace, &pa_cs),
            (&pio_trace, &pio_cs),
            (&ecr_trace, &ecr_cs),
            (&ripemd_trace, &ripemd_cs),
        ];
        let linkages = vec![
            make_ecrecover_to_dispatch_descriptor(2, 0),
            make_ripemd_to_dispatch_descriptor(3, 0),
            make_io_sha256_to_dispatch_descriptor(1, 0),
            make_io_identity_to_dispatch_descriptor(1, 0),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 4-AIR joint_prove must succeed before tampering");

        // Mutate descriptor D2's closure_a so the per-descriptor
        // closure equality check fires.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[2].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&pa_cs, &pio_cs, &ecr_cs, &ripemd_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject the mismatching closure on D2 (io_sha256↔dispatch_sha256)",
        );
    }
}
