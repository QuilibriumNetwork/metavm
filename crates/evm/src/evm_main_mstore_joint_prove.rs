//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **MSTORE byte-decomposition gadget AIR**
//! (`metavm_zkp::mstore_byte_air`).
//!
//! Mirrors `crate::evm_main_address_joint_prove`. The witness is the
//! bytecode
//!
//! ```text
//!   PUSH1 0x42  PUSH1 0x00  MSTORE  PUSH1 0x00  MLOAD  STOP
//!   = [0x60, 0x42, 0x60, 0x00, 0x52, 0x60, 0x00, 0x51, 0x00]
//! ```
//!
//! executed against the standard `execute_bytecode` helper:
//!
//!   - The EVM main row at the MSTORE step exposes the memory write
//!     offset in `COL_MEM_OFFSET` (= 0) and the stored value in
//!     `COL_MEM_VALUE_L0..L3` (= 0x42), with `COL_SEL_MSTORE = 1`.
//!   - The `mstore_byte_air` witness mirrors this one MSTORE event as a
//!     single row: `offset = 0`, `limb = [0x42, 0, 0, 0]`, plus its 32
//!     BE-byte decomposition columns. `COL_IS_REAL = 1`.
//!
//! The descriptor binds **`COL_MEM_OFFSET` ↔ `COL_OFFSET`** as a single-
//! column tuple gated by the respective selectors (`COL_SEL_MSTORE` on
//! the EVM side, `COL_IS_REAL` on the gadget side). `joint_prove`'s
//! current API requires single-column tuples; the gadget AIR internally
//! pins each of the 4 value limbs and the 32 BE bytes via its row-local
//! `limb_decomp_*` constraints, so binding the offset single-column
//! tuple is the minimal hook that ties the EVM MSTORE row to a canonical
//! byte-decomposed gadget row at the same offset. The shared offset
//! (= 0) yields identical multisets on both sides on the one MSTORE row
//! each side commits, so closures match by construction. The MLOAD row
//! is filtered out by the `COL_SEL_MSTORE` gate on the EVM side, so it
//! does not contribute to the linkage trace.
//!
//! Patterned after:
//!   - `crate::evm_main_address_joint_prove` — minimal 2-AIR
//!     joint_prove template.
//!   - `crate::cross_air_linkage::make_evm_mstore_to_byte_decomp_linkage_descriptor`
//!     — the multi-column version (5 cols) used by the slow
//!     `joint_prove_evm_mstore_full_chain` test.
//!
//! Slow tests are `#[ignore]`-gated for the same reason as the ADDRESS /
//! BASEFEE templates: `joint_prove` on a ~5-row EVM main trace padded
//! by the byte-range LogUp tables plus the 1-row gadget runs in the
//! O(30-60s) range on BLS48-581 in `--release`.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::trace::{COL_MEM_OFFSET, COL_SEL_MSTORE};

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_MEM_OFFSET)` on MSTORE rows to `mstore_byte_air`'s
/// `(COL_OFFSET)` on real rows.
///
/// - A side: EVM main `COL_MEM_OFFSET` gated by `COL_SEL_MSTORE`.
/// - B side: `mstore_byte_air` `COL_OFFSET` gated by `COL_IS_REAL`.
///
/// Combined with the gadget AIR's row-local limb-from-bytes bindings
/// and per-byte addr derivation, this pins the EVM main MSTORE row's
/// memory offset to the gadget's canonical BE byte decomposition.
pub fn make_evm_mstore_to_gadget_descriptor(
    evm_layer_index: usize,
    mstore_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_mstore_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_MEM_OFFSET],
        a_selector_column: Some(COL_SEL_MSTORE),
        b_layer_index: mstore_gadget_layer_index,
        b_columns: vec![metavm_zkp::mstore_byte_air::COL_OFFSET],
        b_selector_column: Some(metavm_zkp::mstore_byte_air::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::mstore_byte_air::{
        build_trace_polynomials as build_gadget_trace, MstoreByteConstraintSystem,
        MstoreByteRow, MstoreByteWitness, COL_IS_REAL as GADGET_COL_IS_REAL,
        COL_OFFSET as GADGET_COL_OFFSET,
    };
    use metavm_zkp::trace::TracePolynomials;

    /// PUSH1 0x42 PUSH1 0x00 MSTORE PUSH1 0x00 MLOAD STOP.
    fn mstore_mload_bytecode() -> Vec<u8> {
        vec![0x60, 0x42, 0x60, 0x00, 0x52, 0x60, 0x00, 0x51, 0x00]
    }

    /// Helper: build the EVM main trace by running the MSTORE / MLOAD
    /// bytecode.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = mstore_mload_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for MSTORE/MLOAD bytecode");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Build the gadget trace: 1 row mirroring the MSTORE at offset 0
    /// with value 0x42 (LE limbs `[0x42, 0, 0, 0]`).
    fn build_mstore_gadget_trace(curve: CurveType) -> TracePolynomials {
        let w = MstoreByteWitness::from_invocations(vec![MstoreByteRow {
            offset: 0,
            limb: [0x42, 0, 0, 0],
        }]);
        build_gadget_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces; the MSTORE
    /// row's offset is shared between the EVM main and gadget traces.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_mstore_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_mstore_to_gadget_v1");
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
        assert_eq!(d.a_columns[0], COL_MEM_OFFSET);
        assert_eq!(d.b_columns[0], GADGET_COL_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_SEL_MSTORE));
        assert_eq!(d.b_selector_column, Some(GADGET_COL_IS_REAL));

        // Build both traces and confirm the MSTORE row offset matches
        // on both sides (sanity check on the column wiring before the
        // slow joint_prove tests).
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_mstore_gadget_trace(curve);
        let _evm_cs = EvmConstraintSystem::new();
        let _gadget_cs = MstoreByteConstraintSystem::new(gadget_polys.num_rows);

        // Confirm at least one MSTORE row on the EVM side with
        // COL_MEM_OFFSET = 0.
        let sel_col = &evm_polys.columns[COL_SEL_MSTORE].evaluations;
        let off_col = &evm_polys.columns[COL_MEM_OFFSET].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let expected_off_bytes = Scalar::from_u64(0, curve).to_bytes();
        let mut evm_found = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    off_col[r].to_bytes(),
                    expected_off_bytes,
                    "EVM main MSTORE row's COL_MEM_OFFSET must equal 0",
                );
                evm_found = true;
                break;
            }
        }
        assert!(
            evm_found,
            "EVM main trace must contain at least one MSTORE row",
        );

        // Gadget side: row 0 must be the one real MSTORE row.
        let gadget_real = &gadget_polys.columns[GADGET_COL_IS_REAL].evaluations;
        let gadget_off = &gadget_polys.columns[GADGET_COL_OFFSET].evaluations;
        assert_eq!(
            gadget_real[0].to_bytes(),
            one_bytes,
            "gadget row 0 must be the real MSTORE row",
        );
        assert_eq!(
            gadget_off[0].to_bytes(),
            expected_off_bytes,
            "gadget COL_OFFSET on row 0 must equal 0",
        );

        // Assemble the (trace, cs) pair shape `joint_prove` consumes.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &_evm_cs), (&gadget_polys, &_gadget_cs)];
        let linkages = vec![make_evm_mstore_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace and the 1-row `mstore_byte_air` witness.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + mstore_byte_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_mstore_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = MstoreByteConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_mstore_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + mstore_byte gadget) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's MSTORE-row offset multiset \
             equals B's real-row offset multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ mstore_byte gadget linkage",
        );
    }

    /// Tampered envelope: corrupt `closure_a` so the joint verifier's
    /// scalar `closure_a == closure_b` check fires and rejects.
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
        let gadget_polys = build_mstore_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = MstoreByteConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_mstore_to_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched linkage closures",
        );
    }
}
