//! End-to-end `joint_prove` / `joint_verify` integration smoke test for
//! [`crate::multi_block_proof_air`].
//!
//! `multi_block_proof_air` composes a chain of N consecutive Ethereum
//! blocks into a single AIR with row-local and **cross-row** shifted
//! bodies enforcing `parent_hash(ω·X) == block_hash(X)` and
//! `block_number(ω·X) == block_number(X) + 1`. This integration test
//! drives the AIR through the full cross-AIR LogUp orchestrator on a
//! **3-block chain** witness so any drift in:
//!
//!   1. the shifted constraint polynomial composition under
//!      `prove_with_scheme`,
//!   2. the per-AIR `prove_with_scheme` LogUp domain auto-inflation
//!      inside `joint_prove`, and
//!   3. the cross-AIR LogUp descriptor multiset binding,
//!
//! surfaces here. Two identical 3-block chains are paired via a
//! self-linkage on `COL_BLOCK_NUMBER` (the simplest single-column
//! tuple that's also a u64 value — matches the existing `joint_prove`
//! single-column-tuple shape).
//!
//! ## Topology
//!
//! - layer 0: `multi_block_proof_air` (148 cols, 7 row-local + 2
//!   shifted constraints), gated by `COL_IS_REAL`, with 3 active rows
//!   for blocks `(N, N+1, N+2)`.
//! - layer 1: a second `multi_block_proof_air` instance with the same
//!   3-block chain.
//! - descriptor D0: single-column tuple on `COL_BLOCK_NUMBER` gated by
//!   `COL_IS_REAL`. With identical chains on both sides, the multisets
//!   `{N, N+1, N+2}` match by construction.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{
        joint_prove, joint_verify, CrossAirLogUpDescriptor,
    };
    use crate::field::{CurveType, Scalar};
    use crate::multi_block_proof_air::{
        build_trace_polynomials as build_mb_trace, MultiBlockProofConstraintSystem,
        MultiBlockProofWitness, COL_BLOCK_NUMBER as MB_COL_BLOCK_NUMBER,
        COL_IS_REAL as MB_COL_IS_REAL, HASH_LEN,
    };
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    const CHAIN_LENGTH: usize = 3;
    const START_NUMBER: u64 = 100;

    fn h(seed: u8) -> [u8; HASH_LEN] {
        let mut x = [0u8; HASH_LEN];
        for k in 0..HASH_LEN {
            x[k] = seed.wrapping_add(k as u8);
        }
        x
    }

    /// 3-block honest chain at numbers `[START_NUMBER, START_NUMBER+1,
    /// START_NUMBER+2]` rooted at synthetic genesis parent.
    fn honest_chain()
        -> Vec<(u64, [u8; HASH_LEN], [u8; HASH_LEN], [u8; HASH_LEN], [u8; HASH_LEN])>
    {
        let mut out = Vec::with_capacity(CHAIN_LENGTH);
        let mut parent = h(0xee);
        for i in 0..CHAIN_LENGTH {
            let bh = h(0x10u8.wrapping_add(i as u8));
            let sr = h(0x40u8.wrapping_add(i as u8));
            let psr = h(0x80u8.wrapping_add(i as u8));
            out.push((START_NUMBER + i as u64, bh, parent, sr, psr));
            parent = bh;
        }
        out
    }

    fn build_witness() -> MultiBlockProofWitness {
        MultiBlockProofWitness::from_chain(&honest_chain())
    }

    /// D0: self-linkage on `multi_block_proof_air.COL_BLOCK_NUMBER`
    /// gated by `COL_IS_REAL`. With identical 3-block chains, the
    /// multisets match by construction.
    fn block_number_self_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "multi_block_self_block_number_v1".into(),
            a_layer_index: 0,
            a_columns: vec![MB_COL_BLOCK_NUMBER],
            a_selector_column: Some(MB_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![MB_COL_BLOCK_NUMBER],
            b_selector_column: Some(MB_COL_IS_REAL),
        }
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Confirm the 3-block witness shape, the descriptor's
    /// well-formedness, and the 2-trace orchestrator input. Does NOT
    /// invoke `joint_prove`.
    #[test]
    fn descriptor_and_3block_chain_shapes() {
        let curve = CurveType::Bls48581;

        let w = build_witness();
        assert_eq!(w.rows.len(), CHAIN_LENGTH);
        for i in 0..CHAIN_LENGTH {
            assert_eq!(w.rows[i].block_number, START_NUMBER + i as u64);
        }
        for i in 1..CHAIN_LENGTH {
            assert_eq!(
                w.rows[i].parent_hash, w.rows[i - 1].block_hash,
                "honest chain row {} parent must equal row {} block_hash",
                i, i - 1,
            );
        }
        assert!(w.rows[0].is_first);
        for i in 1..CHAIN_LENGTH {
            assert!(!w.rows[i].is_first);
        }

        let trace_0 = build_mb_trace(&w, curve);
        let trace_1 = build_mb_trace(&w, curve);
        assert_eq!(trace_0.num_rows, CHAIN_LENGTH);
        assert_eq!(trace_1.num_rows, CHAIN_LENGTH);
        // 3 rows → padded to minimum FFT width 16 by
        // `nearest_power_of_two`.
        assert_eq!(trace_0.padded_size, 16);
        assert_eq!(trace_1.padded_size, 16);

        let d0 = block_number_self_descriptor();
        assert_eq!(d0.label, "multi_block_self_block_number_v1");
        assert_eq!(d0.a_columns.len(), 1);
        assert_eq!(d0.b_columns.len(), 1);
        assert_eq!(d0.a_columns[0], MB_COL_BLOCK_NUMBER);
        assert_eq!(d0.b_columns[0], MB_COL_BLOCK_NUMBER);
        assert_eq!(d0.a_selector_column, Some(MB_COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(MB_COL_IS_REAL));
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);

        let cs_0 = MultiBlockProofConstraintSystem::new(trace_0.num_rows);
        let cs_1 = MultiBlockProofConstraintSystem::new(trace_1.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1)];
        let linkages = vec![d0.clone()];

        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }

        // The honest single-row block-number values on both sides match.
        for i in 0..CHAIN_LENGTH {
            assert_eq!(
                trace_0.columns[MB_COL_BLOCK_NUMBER].evaluations[i].to_u64(),
                START_NUMBER + i as u64,
            );
            assert_eq!(
                trace_1.columns[MB_COL_BLOCK_NUMBER].evaluations[i].to_u64(),
                START_NUMBER + i as u64,
            );
        }
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 2-AIR `joint_prove` + `joint_verify` round-trip on
    /// two 3-block chains. The per-AIR `prove_with_scheme` exercises
    /// `multi_block_proof_air`'s 2 shifted constraints
    /// (`parent_chain_eq_block_hash` + `block_number_increment`) on
    /// the real 3-block witness, plus the LogUp byte-range table
    /// auto-inflation to a 256-row commitment domain.
    ///
    /// Marked `#[ignore]` — expected runtime under BLS48-581 is in the
    /// high hundreds of seconds on release builds.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove + joint_verify under BLS48-581 with 3-block shifted bodies"]
    fn honest_multi_block_self_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = build_witness();
        let trace_0 = build_mb_trace(&w, curve);
        let trace_1 = build_mb_trace(&w, curve);

        let cs_0 = MultiBlockProofConstraintSystem::new(trace_0.num_rows);
        let cs_1 = MultiBlockProofConstraintSystem::new(trace_1.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1)];
        let linkages = vec![block_number_self_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest multi_block joint_prove must succeed");

        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);

        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest multi_block joint_verify must accept matching block-number multisets",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper `closure_a` and confirm `joint_verify` rejects.
    #[test]
    #[ignore = "slow: depends on honest_multi_block_self_joint_verify_true setup"]
    fn tampered_multi_block_self_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let w = build_witness();
        let trace_0 = build_mb_trace(&w, curve);
        let trace_1 = build_mb_trace(&w, curve);

        let cs_0 = MultiBlockProofConstraintSystem::new(trace_0.num_rows);
        let cs_1 = MultiBlockProofConstraintSystem::new(trace_1.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1)];
        let linkages = vec![block_number_self_descriptor()];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed on honest inputs");

        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched closures",
        );
    }
}
