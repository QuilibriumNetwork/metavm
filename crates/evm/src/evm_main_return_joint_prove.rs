//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **returndata gadget AIR**
//! (`crate::returndata_air`).
//!
//! Mirrors `crate::evm_main_address_joint_prove`. The witness is the
//! minimal top-level RETURN bytecode
//!
//! ```text
//!   PUSH1 0x20  PUSH1 0x00  RETURN
//!   = [0x60, 0x20, 0x60, 0x00, 0xF3]
//! ```
//!
//! executed against the standard `execute_bytecode` helper:
//!
//!   - The EVM main row at the RETURN step exposes the popped
//!     memory-offset in `COL_INPUT0_L0` (= 0x00) and the popped length
//!     in `COL_INPUT1_L0` (= 0x20). Top-level RETURN (depth = 0)
//!     routes to the umbrella `COL_SEL_CALL = 1` selector (the
//!     algebraic `COL_SEL_CALL_RETURN` only fires for nested
//!     depth ≥ 1 RETURN; see `crate::trace::COL_SEL_CALL_RETURN`).
//!   - The `returndata_air` witness mirrors this one RETURN event as
//!     a single row: `event_type = KIND_RETURN`, `mem_offset = 0`,
//!     `length = 0x20`, with `COL_SEL_RETURN = COL_IS_REAL = 1`.
//!
//! The descriptor binds **`COL_INPUT0_L0` ↔ `COL_MEM_OFFSET`** as a
//! single-column tuple gated by the respective selectors
//! (`COL_SEL_CALL` on the EVM side because our bytecode only contains
//! RETURN among the call-family opcodes, `COL_SEL_RETURN` on the
//! gadget side). `joint_prove`'s current API requires single-column
//! tuples; the gadget AIR internally pins the length, returndata
//! buffer-size threading and per-byte memory-bound checks via its
//! row-local constraints, so binding the memory-offset single-column
//! tuple is sufficient as the minimal RETURN hook. The shared offset
//! (= 0) yields identical multisets on both sides on the one RETURN
//! row each side commits, so closures match by construction. The
//! PUSH1 rows are filtered out by the selector gates and do not
//! contribute to the linkage trace.
//!
//! Patterned after:
//!   - `crate::evm_main_address_joint_prove` — minimal 2-AIR
//!     joint_prove template.
//!   - `crate::returndata_air::make_returndata_to_call_family_descriptor`
//!     — the analogous 2-column gadget→family descriptor.
//!
//! Slow tests are `#[ignore]`-gated for the same reason as the ADDRESS /
//! BASEFEE templates: `joint_prove` on a ~3-row EVM main trace padded
//! by the byte-range LogUp tables plus the 1-row gadget runs in the
//! O(30-60s) range on BLS48-581 in `--release`.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::returndata_air::{COL_MEM_OFFSET as GADGET_COL_MEM_OFFSET, COL_SEL_RETURN};
use crate::trace::{COL_INPUT0_L0, COL_SEL_CALL};

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_INPUT0_L0)` on RETURN-style rows to `returndata_air`'s
/// `(COL_MEM_OFFSET)` on RETURN-event rows.
///
/// - A side: EVM main `COL_INPUT0_L0` gated by `COL_SEL_CALL` (the
///   umbrella call-family oracle that fires for top-level RETURN at
///   depth 0; see `crate::trace::COL_SEL_CALL_RETURN` for the nested
///   depth ≥ 1 variant).
/// - B side: `returndata_air` `COL_MEM_OFFSET` gated by
///   `COL_SEL_RETURN`.
///
/// Combined with the gadget AIR's row-local
/// `return_revert_sets_post_eq_length` and byte-range constraints,
/// this pins the EVM main RETURN row's memory offset to the gadget's
/// canonical view of the RETURN event.
pub fn make_evm_return_to_gadget_descriptor(
    evm_layer_index: usize,
    returndata_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_return_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_INPUT0_L0],
        a_selector_column: Some(COL_SEL_CALL),
        b_layer_index: returndata_gadget_layer_index,
        b_columns: vec![GADGET_COL_MEM_OFFSET],
        b_selector_column: Some(COL_SEL_RETURN),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use crate::returndata_air::{
        build_trace_polynomials as build_returndata_trace, from_events,
        ReturnDataConstraintSystem, ReturnDataEvent,
    };
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// PUSH1 0x20 PUSH1 0x00 RETURN.
    fn return_bytecode() -> Vec<u8> {
        vec![0x60, 0x20, 0x60, 0x00, 0xF3]
    }

    /// Helper: build the EVM main trace by running the RETURN bytecode.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = return_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for top-level RETURN bytecode");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Build the gadget trace: 1 row mirroring the RETURN at mem_offset = 0,
    /// length = 0x20, depth = 0 (top-level), buffer-size pre = 0 → post
    /// = length.
    fn build_returndata_gadget_trace(curve: CurveType) -> TracePolynomials {
        let w = from_events(&[ReturnDataEvent::ret(0, 0x20, 0, 0)]);
        build_returndata_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces; the RETURN
    /// row's mem_offset is shared between the EVM main and gadget
    /// traces.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_return_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_return_to_gadget_v1");
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
        assert_eq!(d.a_columns[0], COL_INPUT0_L0);
        assert_eq!(d.b_columns[0], GADGET_COL_MEM_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_SEL_CALL));
        assert_eq!(d.b_selector_column, Some(COL_SEL_RETURN));

        // Build both traces and confirm the RETURN row mem_offset
        // matches on both sides (sanity check on the column wiring
        // before the slow joint_prove tests).
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_returndata_gadget_trace(curve);
        let _evm_cs = EvmConstraintSystem::new();
        let _gadget_cs = ReturnDataConstraintSystem::new(gadget_polys.num_rows);

        // Confirm at least one sel_call row on the EVM side (the
        // top-level RETURN) with COL_INPUT0_L0 = 0.
        let sel_col = &evm_polys.columns[COL_SEL_CALL].evaluations;
        let in0_col = &evm_polys.columns[COL_INPUT0_L0].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let expected_off_bytes = Scalar::from_u64(0, curve).to_bytes();
        let mut evm_found = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    in0_col[r].to_bytes(),
                    expected_off_bytes,
                    "EVM main RETURN row's COL_INPUT0_L0 must equal 0",
                );
                evm_found = true;
                break;
            }
        }
        assert!(
            evm_found,
            "EVM main trace must contain at least one COL_SEL_CALL row \
             (the top-level RETURN)",
        );

        // Gadget side: row 0 must be the one RETURN row.
        let gadget_sel = &gadget_polys.columns[COL_SEL_RETURN].evaluations;
        let gadget_off = &gadget_polys.columns[GADGET_COL_MEM_OFFSET].evaluations;
        assert_eq!(
            gadget_sel[0].to_bytes(),
            one_bytes,
            "gadget row 0 must be the RETURN row",
        );
        assert_eq!(
            gadget_off[0].to_bytes(),
            expected_off_bytes,
            "gadget COL_MEM_OFFSET on row 0 must equal 0",
        );

        // Assemble the (trace, cs) pair shape `joint_prove` consumes.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &_evm_cs), (&gadget_polys, &_gadget_cs)];
        let linkages = vec![make_evm_return_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace and the 1-row `returndata_air` witness.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + returndata_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_returndata_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = ReturnDataConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_return_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + returndata gadget) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's RETURN-row mem_offset multiset \
             equals B's RETURN-event row mem_offset multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ returndata gadget linkage",
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
        let gadget_polys = build_returndata_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = ReturnDataConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_return_to_gadget_descriptor(0, 1)];

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
