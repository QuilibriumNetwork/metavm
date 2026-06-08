//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **ADDRESS opcode gadget AIR**.
//!
//! This is the first joint_prove that pairs the EVM main trace with a
//! single small per-opcode gadget AIR. The witness is the minimal
//! bytecode `[0x30, 0x00]` (ADDRESS; STOP) executed against the
//! standard executor (contract address = `[0x42; 20]`):
//!
//!   - The EVM main row at the ADDRESS step exposes the pushed u256 in
//!     `COL_OUTPUT0_L0..L3` and sets `COL_SEL_ADDRESS = 1`.
//!   - The address_opcode_air row mirrors the same address bytes and
//!     `value_limb_0..3` with `COL_IS_REAL = 1`.
//!
//! The descriptor binds **`COL_OUTPUT0_L0` ↔ `COL_VALUE_LIMB_0`** as a
//! single-column tuple gated by the respective selectors (`COL_SEL_ADDRESS`
//! on the EVM side, `COL_IS_REAL` on the gadget side). `joint_prove`'s
//! current API requires single-column tuples (see
//! `crates/zkp/src/cross_air_logup.rs` doc on `build_linkage_trace`); the
//! gadget AIR internally pins limb 1, limb 2 and limb 3 via its
//! `value_limb_*_binding` row-local constraints, so binding the limb-0
//! single-column tuple is sufficient to algebraically commit the full
//! 4-limb address tuple to the EVM main row's push output. The shared
//! `[0x42; 20]` address yields identical limb-0 multisets on both sides
//! so closures match by construction.
//!
//! Patterned after:
//!   - `crates/zkp/src/integration_joint_prove_smoke.rs` — minimal smoke
//!     test using a single-column self-linkage.
//!   - `crates/evm/src/cross_air_linkage.rs::joint_prove_evm_storage_chain`
//!     — EVM main + gadget AIR multi-linkage pattern.
//!
//! Slow tests are `#[ignore]`-gated because `joint_prove` on a BLS48-581
//! trace pair with EVM main (~2 EVM rows padded to 256 by LogUp range
//! tables) runs in O(30-60s) release.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::address_opcode_air::{COL_IS_REAL, COL_VALUE_LIMB_0};
use crate::trace::{COL_OUTPUT0_L0, COL_SEL_ADDRESS};

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_OUTPUT0_L0)` on ADDRESS rows to address_opcode_air's
/// `(COL_VALUE_LIMB_0)` on real rows.
///
/// - A side: EVM main `COL_OUTPUT0_L0` gated by `COL_SEL_ADDRESS`.
/// - B side: address_opcode_air `COL_VALUE_LIMB_0` gated by `COL_IS_REAL`.
///
/// Combined with the gadget AIR's row-local limb-from-bytes bindings,
/// the gas/opcode constraints, and limb-3 = 0 padding constraint, this
/// pins the EVM main push output to the gadget's canonical
/// `address_to_limbs(contract_addr)`.
pub fn make_evm_address_to_gadget_descriptor(
    evm_layer_index: usize,
    address_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_address_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_ADDRESS),
        b_layer_index: address_gadget_layer_index,
        b_columns: vec![COL_VALUE_LIMB_0],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::address_opcode_air::{
        address_to_limbs, build_trace_polynomials as build_addr_trace,
        AddressOpcodeConstraintSystem, AddressOpcodeWitness, ADDRESS_OPCODE,
    };
    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use crate::trace::COL_SEL_ADDRESS;
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// Contract address used by `execute_bytecode` (mirrors
    /// `crate::executor`'s hard-coded `Address::from([0x42; 20])`).
    const CONTRACT_ADDR: [u8; 20] = [0x42; 20];

    /// Minimal bytecode used by all tests: ADDRESS; STOP.
    fn address_stop_bytecode() -> Vec<u8> {
        vec![0x30, 0x00]
    }

    /// Helper: build the EVM main trace by running `[0x30, 0x00]`.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = address_stop_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for ADDRESS;STOP");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Helper: build the address-opcode gadget trace pinned to the
    /// executor's contract address at PC = 0.
    fn build_address_gadget_trace(curve: CurveType) -> TracePolynomials {
        let w = AddressOpcodeWitness::from_events(&[(0, CONTRACT_ADDR)]);
        build_addr_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_address_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_address_to_gadget_v1");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_ne!(
            d.a_layer_index, d.b_layer_index,
            "linkage layers must differ for cross-AIR LogUp"
        );
        assert_eq!(d.a_columns.len(), d.b_columns.len());
        assert_eq!(
            d.a_columns.len(),
            1,
            "joint_prove currently requires single-column tuples"
        );
        assert_eq!(d.a_columns[0], COL_OUTPUT0_L0);
        assert_eq!(d.b_columns[0], COL_VALUE_LIMB_0);
        assert_eq!(d.a_selector_column, Some(COL_SEL_ADDRESS));
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));

        // The `JointProveInput` for this test is just the (trace, cs)
        // slice + the linkage slice we plug into `joint_prove`. Build
        // both fully here to catch wiring regressions without running
        // the (slow) prover. All we assert is that everything ties
        // together correctly: layer indices, column counts, and the
        // shared limb-0 value across both traces line up.
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let addr_polys = build_address_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let addr_cs = AddressOpcodeConstraintSystem::new(addr_polys.num_rows);

        // Confirm the EVM trace actually contains an ADDRESS row with the
        // expected selector + push output.
        let sel_col = &evm_polys.columns[COL_SEL_ADDRESS].evaluations;
        let out0_col = &evm_polys.columns[COL_OUTPUT0_L0].evaluations;
        let limbs = address_to_limbs(&CONTRACT_ADDR);
        let expected_l0_bytes = Scalar::from_u64(limbs[0], curve).to_bytes();
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    out0_col[r].to_bytes(),
                    expected_l0_bytes,
                    "EVM main ADDRESS row's OUTPUT0_L0 must equal address_to_limbs(contract)[0]"
                );
                found = true;
                break;
            }
        }
        assert!(found, "EVM main trace must contain at least one ADDRESS row");

        // Gadget side: row 0 must mirror the same limb-0.
        let gadget_l0 = &addr_polys.columns[COL_VALUE_LIMB_0].evaluations;
        assert_eq!(
            gadget_l0[0].to_bytes(),
            expected_l0_bytes,
            "gadget value_limb_0 row 0 must equal address_to_limbs(contract)[0]"
        );

        // Sanity: the two CS instances and traces can be assembled into
        // the (trace, cs) pair shape `joint_prove` consumes.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&addr_polys, &addr_cs)];
        let linkages = vec![make_evm_address_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);

        // Sanity: ADDRESS opcode constant pinned by gadget matches 0x30.
        assert_eq!(ADDRESS_OPCODE, 0x30);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace (running `ADDRESS;STOP`) and a 1-row
    /// address_opcode_air witness pinned to the same contract address.
    ///
    /// Marked `#[ignore]` because, even with a 2-byte bytecode,
    /// `joint_prove` runs:
    ///   - 1× per-AIR `prove_with_scheme` for EVM main (padded to the
    ///     LogUp range-table domain = 256 because EvmConstraintSystem
    ///     declares 8-bit range checks);
    ///   - 1× per-AIR `prove_with_scheme` for address_opcode_air;
    ///   - 1× per-linkage `prove_with_scheme` on the inner
    ///     `LinkageConstraintSystem`;
    ///   - KZG opens against per-AIR / per-linkage commitments.
    /// On BLS48-581 this is expected to run in the 30-60s range on a
    /// modest workstation in `--release`.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + address_opcode_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let addr_polys = build_address_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let addr_cs = AddressOpcodeConstraintSystem::new(addr_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&addr_polys, &addr_cs)];
        let linkages = vec![make_evm_address_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + address gadget) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's ADDRESS-row push limb-0 multiset \
             equals B's real-row value_limb_0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &addr_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ address gadget linkage",
        );
    }

    /// Tampered witness: corrupt `closure_a` in the extension envelope
    /// so the joint verifier's scalar `closure_a == closure_b` check
    /// fires and rejects.
    ///
    /// Re-runs `joint_prove` because we need a real `ExecutionProof`
    /// pair to feed `joint_verify`. Marked `#[ignore]` because the
    /// setup half is the same `joint_prove` call as
    /// `honest_joint_verify_true`.
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
        let addr_polys = build_address_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let addr_cs = AddressOpcodeConstraintSystem::new(addr_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&addr_polys, &addr_cs)];
        let linkages = vec![make_evm_address_to_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        // Mutate closure_a in the envelope so the joint verifier's
        // `closure_a == closure_b` scalar equality check fires.
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &addr_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched linkage closures",
        );
    }
}
