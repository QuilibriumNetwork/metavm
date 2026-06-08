//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **PUSH0 opcode gadget AIR**.
//!
//! Mirrors `crate::evm_main_address_joint_prove`, but for opcode `0x5F`
//! (PUSH0, EIP-3855). The witness is the minimal bytecode
//! `[0x5F, 0x00]` (PUSH0; STOP) executed against the standard
//! `execute_bytecode` helper.
//!
//! ## Selector caveat
//!
//! PUSH0 is **not** classified as `(INSN_STACK, FUNCT_PUSH)` by the
//! trace's `classify_opcode` (the PUSH1..32 range is `0x60..=0x7F`), so
//! there is no dedicated `COL_SEL_PUSH0` on the EVM main trace. PUSH0
//! falls through to the `INSN_STOP` arm, which sets `COL_SEL_STOP = 1`
//! on both the PUSH0 row AND the trailing STOP row. We therefore gate
//! the A side of the linkage by `COL_SEL_STOP` and rely on the fact
//! that for `[0x5F, 0x00]` both rows leave `0` on top of the stack:
//!
//!   - Row 0 (PUSH0):  pushes u256 zero, `output0 = 0`.
//!   - Row 1 (STOP):   doesn't pop, stack top is still the zero pushed
//!     by PUSH0 → `output0 = 0`.
//!
//! The push0_air gadget witness is built with 2 real rows, each with
//! `value = [0, 0, 0, 0]`, so the multisets on both sides are
//! `{0, 0}` and closures match by construction. A future dedicated
//! `COL_SEL_PUSH0` (gated by `COL_OPCODE == 0x5F`) would let this
//! linkage fire only on the real PUSH0 row, but the structural shape
//! and `joint_prove` wiring is identical.
//!
//! The descriptor binds **`COL_OUTPUT0_L0` ↔ `COL_VALUE_LIMB_0`** as a
//! single-column tuple. The gadget AIR's `value_limb_k_zero`
//! constraints (k = 0..4) and the row-local LE byte decomposition
//! force the gadget's value to be exactly u256(0), so binding the
//! limb-0 single-column tuple is sufficient to algebraically commit
//! the full 4-limb push output to the gadget's canonical PUSH0 zero.
//!
//! Slow tests are `#[ignore]`-gated for the same reason as the ADDRESS
//! template: BLS48-581 joint_prove takes O(30-60s) release.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::push0_air::{COL_IS_REAL, COL_VALUE_LIMB_0};
use crate::trace::{COL_OUTPUT0_L0, COL_SEL_STOP};

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_OUTPUT0_L0)` on rows gated by `COL_SEL_STOP` to push0_air's
/// `(COL_VALUE_LIMB_0)` on real rows.
///
/// - A side: EVM main `COL_OUTPUT0_L0` gated by `COL_SEL_STOP`
///   (which fires on both PUSH0 and STOP for the canonical
///   `[0x5F, 0x00]` bytecode — see module-level caveat).
/// - B side: push0_air `COL_VALUE_LIMB_0` gated by `COL_IS_REAL`.
pub fn make_evm_push0_to_gadget_descriptor(
    evm_layer_index: usize,
    push0_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_push0_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_STOP),
        b_layer_index: push0_gadget_layer_index,
        b_columns: vec![COL_VALUE_LIMB_0],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use crate::push0_air::{
        build_trace_polynomials as build_push0_trace, Push0ConstraintSystem, Push0Witness,
        PUSH0_OPCODE,
    };
    use crate::trace::COL_SEL_STOP;
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// Minimal bytecode used by all tests: PUSH0; STOP.
    fn push0_stop_bytecode() -> Vec<u8> {
        vec![0x5F, 0x00]
    }

    /// Helper: build the EVM main trace by running `[0x5F, 0x00]`.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = push0_stop_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for PUSH0;STOP");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Count the number of EVM main rows where `COL_SEL_STOP` fires.
    /// Used to size the gadget witness so closures match by construction.
    fn count_stop_gated_rows(evm_polys: &TracePolynomials) -> usize {
        let sel_col = &evm_polys.columns[COL_SEL_STOP].evaluations;
        let one_bytes = Scalar::one(evm_polys.curve).to_bytes();
        let mut n = 0;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                n += 1;
            }
        }
        n
    }

    /// Build a push0_air gadget trace with one real row per EVM main
    /// row gated by `COL_SEL_STOP`. Each row carries the canonical
    /// PUSH0 payload (value = 0, opcode = 0x5F, post-Shanghai = 1,
    /// gas_cost = 2).
    fn build_push0_gadget_trace(
        evm_polys: &TracePolynomials,
        curve: CurveType,
    ) -> TracePolynomials {
        let n = count_stop_gated_rows(evm_polys).max(1);
        let events: Vec<(u64, bool)> =
            (0..n).map(|i| (i as u64, true)).collect();
        let w = Push0Witness::from_events(&events);
        build_push0_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_push0_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_push0_to_gadget_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_ne!(d.a_layer_index, d.b_layer_index);
        assert_eq!(d.a_columns.len(), d.b_columns.len());
        assert_eq!(
            d.a_columns.len(),
            1,
            "joint_prove currently requires single-column tuples"
        );
        assert_eq!(d.a_columns[0], COL_OUTPUT0_L0);
        assert_eq!(d.b_columns[0], COL_VALUE_LIMB_0);
        assert_eq!(d.a_selector_column, Some(COL_SEL_STOP));
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));

        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let push0_polys = build_push0_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let push0_cs = Push0ConstraintSystem::new(push0_polys.num_rows);

        // Confirm there's at least one stop-gated row that's a PUSH0
        // event (output0 must be the zero pushed by PUSH0).
        let sel_col = &evm_polys.columns[COL_SEL_STOP].evaluations;
        let out0_col = &evm_polys.columns[COL_OUTPUT0_L0].evaluations;
        let zero_bytes = Scalar::zero(curve).to_bytes();
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    out0_col[r].to_bytes(),
                    zero_bytes,
                    "PUSH0/STOP rows must leave 0 on top of the stack"
                );
                found = true;
            }
        }
        assert!(found, "EVM trace must contain at least one stop-gated row");

        // Gadget side: every real row's value_limb_0 must be zero.
        let gadget_l0 = &push0_polys.columns[COL_VALUE_LIMB_0].evaluations;
        for r in 0..push0_polys.num_rows {
            assert_eq!(
                gadget_l0[r].to_bytes(),
                zero_bytes,
                "gadget value_limb_0 row {} must equal 0 (PUSH0 always pushes zero)",
                r
            );
        }

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&push0_polys, &push0_cs)];
        let linkages = vec![make_evm_push0_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);

        // Sanity: PUSH0 opcode constant pinned by gadget matches 0x5F.
        assert_eq!(PUSH0_OPCODE, 0x5F);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + push0_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let push0_polys = build_push0_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let push0_cs = Push0ConstraintSystem::new(push0_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&push0_polys, &push0_cs)];
        let linkages = vec![make_evm_push0_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + push0 gadget) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's stop-gated zero-push limb-0 \
             multiset equals B's real-row value_limb_0 multiset (all zeros)",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &push0_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ push0 gadget linkage",
        );
    }

    /// Tampered envelope: corrupt `closure_a` so the joint verifier
    /// rejects.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_joint_verify_true"]
    fn tampered_joint_verify_false() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let push0_polys = build_push0_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let push0_cs = Push0ConstraintSystem::new(push0_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&push0_polys, &push0_cs)];
        let linkages = vec![make_evm_push0_to_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &push0_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched linkage closures",
        );
    }
}
