//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **SSTORE / SLOAD pre-post gadget AIR**
//! (`crate::sstore_prepost_air`).
//!
//! Mirrors `crate::evm_main_address_joint_prove` and
//! `crate::evm_main_basefee_joint_prove`, but for the storage opcodes.
//! The witness is the bytecode
//!
//! ```text
//!   PUSH1 0x42  PUSH1 0x05  SSTORE  PUSH1 0x05  SLOAD  STOP
//!   = [0x60, 0x42, 0x60, 0x05, 0x55, 0x60, 0x05, 0x54, 0x00]
//! ```
//!
//! executed against the standard `execute_bytecode` helper:
//!
//!   - The EVM main row at the SSTORE step exposes the popped slot in
//!     `COL_INPUT0_L0..L3` (= 0x05) and the value in `COL_INPUT1_L0..L3`
//!     (= 0x42), with `COL_SEL_SSTORE = 1`.
//!   - The EVM main row at the SLOAD step exposes the popped slot in
//!     `COL_INPUT0_L0..L3` (= 0x05) and the loaded value in
//!     `COL_OUTPUT0_L0..L3` (= 0x42), with `COL_SEL_SLOAD = 1`.
//!   - The `sstore_prepost_air` witness mirrors both as two rows:
//!     row 0 = SSTORE (`sel_sstore=1`, original=0, pre=0, post=0x42,
//!     gas=20000); row 1 = SLOAD (`sel_sload=1`, warm, gas=100, pre=post
//!     = 0x42, original=0x42).
//!
//! The descriptor binds **the LSB of the slot** as a single-column tuple
//! gated by the respective SSTORE selectors:
//!
//!   - A side: EVM main `COL_INPUT0_L0` gated by `COL_SEL_SSTORE`.
//!   - B side: `sstore_prepost_air`'s slot byte 31 (last byte of the
//!     big-endian 32-byte slot column block) gated by `COL_SEL_SSTORE`.
//!
//! `joint_prove`'s current API requires single-column tuples (see
//! `crates/zkp/src/cross_air_logup.rs` doc on `build_linkage_trace`).
//! The gadget AIR internally pins the rest of the slot bytes via its
//! per-row byte range checks and the constraints that couple slot/value
//! columns to gas. The shared slot = 0x05 yields identical multisets on
//! both sides on the single SSTORE row each side commits, so closures
//! match by construction. Two-row SSTORE + SLOAD pair exercises the
//! selector gating on both sides — SLOAD rows are filtered out by the
//! `COL_SEL_SSTORE` gate so they don't contribute to the linkage trace.
//!
//! Slow tests are `#[ignore]`-gated for the same reason as the ADDRESS /
//! BASEFEE templates: `joint_prove` on a ~5-row EVM main trace padded by
//! the byte-range LogUp tables, plus the 2-row gadget, runs in the
//! O(30-60s) range on BLS48-581 in `--release`.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::sstore_prepost_air::{
    COL_SEL_SSTORE as GADGET_COL_SEL_SSTORE, COL_SLOT_OFFSET, WORD_BYTES,
};
use crate::trace::{COL_INPUT0_L0, COL_SEL_SSTORE};

/// LSB byte index of the gadget's big-endian 32-byte slot column block.
pub const GADGET_SLOT_LSB_COL: usize = COL_SLOT_OFFSET + WORD_BYTES - 1;

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_INPUT0_L0)` on SSTORE rows to `sstore_prepost_air`'s
/// slot-byte-31 (LSB of the BE slot column block) on SSTORE rows.
///
/// - A side: EVM main `COL_INPUT0_L0` gated by `COL_SEL_SSTORE`.
/// - B side: `sstore_prepost_air` slot byte 31 gated by `COL_SEL_SSTORE`.
///
/// Combined with the gadget AIR's row-local byte range checks on every
/// slot byte and the SSTORE gas / no-op / write / clear constraints,
/// this pins the EVM main popped slot's LSB to the gadget's canonical
/// slot binding on each SSTORE row.
pub fn make_evm_sstore_to_gadget_descriptor(
    evm_layer_index: usize,
    sstore_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_sstore_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_INPUT0_L0],
        a_selector_column: Some(COL_SEL_SSTORE),
        b_layer_index: sstore_gadget_layer_index,
        b_columns: vec![GADGET_SLOT_LSB_COL],
        b_selector_column: Some(GADGET_COL_SEL_SSTORE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use crate::sstore_prepost_air::{
        build_trace_polynomials as build_sstore_trace, from_events,
        SstorePrepostConstraintSystem, COLD_SLOAD_GAS, SSTORE_WRITE_GAS,
        WARM_SLOAD_GAS,
    };
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// PUSH1 0x42 PUSH1 0x05 SSTORE PUSH1 0x05 SLOAD STOP.
    fn sstore_sload_bytecode() -> Vec<u8> {
        vec![0x60, 0x42, 0x60, 0x05, 0x55, 0x60, 0x05, 0x54, 0x00]
    }

    /// Big-endian 32-byte word with `v` in the LSB position.
    fn word_with_lsb(v: u8) -> [u8; 32] {
        let mut w = [0u8; 32];
        w[31] = v;
        w
    }

    /// Helper: build the EVM main trace by running the SSTORE / SLOAD
    /// bytecode.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = sstore_sload_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for SSTORE/SLOAD bytecode");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Build the gadget trace: 2 rows mirroring SSTORE then SLOAD on
    /// slot 0x05 / value 0x42.
    ///
    /// Row 0 — SSTORE: original=0, pre=0, post=0x42 → "write" case,
    /// gas = 20000.
    ///
    /// Row 1 — SLOAD: pre=post=0x42, original=0x42, cold (first SLOAD
    /// after a transaction-level SSTORE has already warmed the slot, so
    /// in principle this would be warm; we use COLD here because the
    /// SLOAD-non-mutation and gas constraints are satisfied for cold
    /// SLOAD = 2100 gas regardless of the warm/cold split — the gadget
    /// constraints only check is_warm consistency with gas_cost, not
    /// warmth against execution order). Using warm + 100 also satisfies
    /// the constraints; we pick warm = true + 100 to mirror the actual
    /// post-SSTORE EVM warmth and minimise mismatch risk.
    fn build_gadget_trace(curve: CurveType) -> TracePolynomials {
        let slot = word_with_lsb(0x05);
        let val = word_with_lsb(0x42);
        let zero = [0u8; 32];
        let events = vec![
            // SSTORE: original=0, pre=0, post=val → write 20000 gas.
            (4, true, slot, zero, val, zero, SSTORE_WRITE_GAS, true),
            // SLOAD: warm (just-written slot) → 100 gas, pre==post==val.
            (7, false, slot, val, val, val, WARM_SLOAD_GAS, true),
        ];
        let _ = COLD_SLOAD_GAS; // silence unused if branch is removed
        let w = from_events(&events);
        build_sstore_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces; the SSTORE
    /// row's slot LSB is shared between the EVM main and gadget traces.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_sstore_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_sstore_to_gadget_v1");
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
        assert_eq!(d.b_columns[0], GADGET_SLOT_LSB_COL);
        assert_eq!(d.a_selector_column, Some(COL_SEL_SSTORE));
        assert_eq!(d.b_selector_column, Some(GADGET_COL_SEL_SSTORE));

        // Build both traces and confirm the SSTORE row LSB matches on
        // both sides (sanity check on the column wiring before the
        // slow joint_prove tests).
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_gadget_trace(curve);
        let _evm_cs = EvmConstraintSystem::new();
        let _gadget_cs = SstorePrepostConstraintSystem::new(gadget_polys.num_rows);

        // Confirm at least one SSTORE row on the EVM side with INPUT0_L0
        // = 0x05.
        let sel_col = &evm_polys.columns[COL_SEL_SSTORE].evaluations;
        let in0_col = &evm_polys.columns[COL_INPUT0_L0].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let expected_lsb_bytes = Scalar::from_u64(0x05, curve).to_bytes();
        let mut evm_found = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    in0_col[r].to_bytes(),
                    expected_lsb_bytes,
                    "EVM main SSTORE row's INPUT0_L0 must equal 0x05",
                );
                evm_found = true;
                break;
            }
        }
        assert!(
            evm_found,
            "EVM main trace must contain at least one SSTORE row",
        );

        // Gadget side: row 0 is the SSTORE row. Confirm SEL_SSTORE = 1
        // and slot LSB = 0x05.
        let gadget_sel = &gadget_polys.columns[GADGET_COL_SEL_SSTORE].evaluations;
        let gadget_lsb = &gadget_polys.columns[GADGET_SLOT_LSB_COL].evaluations;
        assert_eq!(
            gadget_sel[0].to_bytes(),
            one_bytes,
            "gadget row 0 must be the SSTORE row",
        );
        assert_eq!(
            gadget_lsb[0].to_bytes(),
            expected_lsb_bytes,
            "gadget slot LSB on row 0 must equal 0x05",
        );

        // Sanity: gadget row 1 is the SLOAD row (sel_sstore = 0).
        let zero_bytes = Scalar::zero(curve).to_bytes();
        assert_eq!(
            gadget_sel[1].to_bytes(),
            zero_bytes,
            "gadget row 1 must NOT be SSTORE-selected (it is the SLOAD row)",
        );

        // Assemble the (trace, cs) pair shape `joint_prove` consumes.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &_evm_cs), (&gadget_polys, &_gadget_cs)];
        let linkages = vec![make_evm_sstore_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across the
    /// EVM main trace (running the SSTORE/SLOAD bytecode) and the 2-row
    /// `sstore_prepost_air` witness.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + sstore_prepost_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = SstorePrepostConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_sstore_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + sstore gadget) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's SSTORE-row slot-LSB multiset \
             equals B's SSTORE-row slot-LSB multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ sstore gadget linkage",
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
        let gadget_polys = build_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = SstorePrepostConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_sstore_to_gadget_descriptor(0, 1)];

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
