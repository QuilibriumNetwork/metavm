//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **BASEFEE opcode gadget AIR**.
//!
//! Mirrors `crate::evm_main_address_joint_prove`, but for opcode `0x48`
//! (BASEFEE). The witness is the minimal bytecode `[0x48, 0x00]`
//! (BASEFEE; STOP) executed against the standard `execute_bytecode`
//! helper:
//!
//!   - The EVM main row at the BASEFEE step exposes the pushed u256 in
//!     `COL_OUTPUT0_L0..L3` and sets `COL_SEL_BASEFEE = 1`.
//!   - The basefee_air row mirrors the same value bytes / limbs with
//!     `COL_IS_REAL = 1`. To make closures match by construction, the
//!     gadget witness is populated by reading the EVM main row's
//!     OUTPUT0_L0 back out of the trace and feeding it to
//!     `BaseFeeWitness::from_events`.
//!
//! The descriptor binds **`COL_OUTPUT0_L0` ↔ `COL_BASE_FEE_VAL_L0`** as a
//! single-column tuple gated by the respective selectors (`COL_SEL_BASEFEE`
//! on the EVM side, `COL_IS_REAL` on the gadget side). The gadget AIR's
//! `base_fee_eq_block_lk` constraints (k = 0..4) and LE byte-decomposition
//! constraints pin the higher limbs, so binding the limb-0 single-column
//! tuple is sufficient to algebraically commit the full 4-limb push
//! output to the gadget's canonical block-base-fee binding.
//!
//! Slow tests are `#[ignore]`-gated for the same reason as the ADDRESS
//! template: `joint_prove` on a 2-row EVM main trace (padded to 256 by
//! the byte-range LogUp tables) plus the small basefee gadget runs in
//! the O(30-60s) range on BLS48-581 in `--release`.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::basefee_air::{COL_BASE_FEE_VAL_L0, COL_IS_REAL};
use crate::trace::{COL_OUTPUT0_L0, COL_SEL_BASEFEE};

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_OUTPUT0_L0)` on BASEFEE rows to basefee_air's
/// `(COL_BASE_FEE_VAL_L0)` on real rows.
///
/// - A side: EVM main `COL_OUTPUT0_L0` gated by `COL_SEL_BASEFEE`.
/// - B side: basefee_air `COL_BASE_FEE_VAL_L0` gated by `COL_IS_REAL`.
///
/// Combined with the gadget AIR's row-local LE byte decomposition and
/// `base_fee_eq_block` limb-equality constraints, this pins the EVM
/// main push output to the gadget's canonical 4-limb base-fee binding.
pub fn make_evm_basefee_to_gadget_descriptor(
    evm_layer_index: usize,
    basefee_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_basefee_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_BASEFEE),
        b_layer_index: basefee_gadget_layer_index,
        b_columns: vec![COL_BASE_FEE_VAL_L0],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::basefee_air::{
        build_trace_polynomials as build_basefee_trace, BaseFeeConstraintSystem, BaseFeeWitness,
    };
    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use crate::trace::COL_SEL_BASEFEE;
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// Minimal bytecode used by all tests: BASEFEE; STOP.
    fn basefee_stop_bytecode() -> Vec<u8> {
        vec![0x48, 0x00]
    }

    /// Helper: build the EVM main trace by running `[0x48, 0x00]`.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = basefee_stop_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for BASEFEE;STOP");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Locate the BASEFEE row in the EVM main trace and return the
    /// LE u64 limb-0 of OUTPUT0 (matches whatever revm's BlockEnv
    /// returns for `base_fee_per_gas`).
    fn extract_basefee_value(evm_polys: &TracePolynomials) -> Result<u64, &'static str> {
        let sel_col = &evm_polys.columns[COL_SEL_BASEFEE].evaluations;
        let out0_col = &evm_polys.columns[COL_OUTPUT0_L0].evaluations;
        let curve = evm_polys.curve;
        let one_bytes = Scalar::one(curve).to_bytes();
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                // Round-trip through field bytes to extract the low u64.
                // OUTPUT0_L0 is canonically a u64 cell, so the BE-bytes
                // tail holds the value.
                let be = out0_col[r].to_bytes();
                let mut tail = [0u8; 8];
                tail.copy_from_slice(&be[be.len() - 8..]);
                return Ok(u64::from_be_bytes(tail));
            }
        }
        Err("EVM main trace must contain at least one BASEFEE row")
    }

    /// Build a gadget witness pinned to the EVM main's pushed base-fee.
    fn build_basefee_gadget_trace(
        evm_polys: &TracePolynomials,
        curve: CurveType,
    ) -> TracePolynomials {
        let v = extract_basefee_value(evm_polys)
            .expect("EVM main trace must contain at least one BASEFEE row");
        let mut be = [0u8; 32];
        be[24..32].copy_from_slice(&v.to_be_bytes());
        let w = BaseFeeWitness::from_events(&[(0, be)]);
        build_basefee_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces; closures align
    /// on the EVM main BASEFEE row's pushed value.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_basefee_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_basefee_to_gadget_v1");
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
        assert_eq!(d.b_columns[0], COL_BASE_FEE_VAL_L0);
        assert_eq!(d.a_selector_column, Some(COL_SEL_BASEFEE));
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));

        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let basefee_polys = build_basefee_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let basefee_cs = BaseFeeConstraintSystem::new(basefee_polys.num_rows);

        // Confirm the EVM trace actually contains a BASEFEE row.
        let sel_col = &evm_polys.columns[COL_SEL_BASEFEE].evaluations;
        let out0_col = &evm_polys.columns[COL_OUTPUT0_L0].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found = None;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                found = Some(out0_col[r].to_bytes());
                break;
            }
        }
        let evm_l0_bytes =
            found.expect("EVM main trace must contain at least one BASEFEE row");

        // Gadget side: row 0 limb-0 must mirror it.
        let gadget_l0 = &basefee_polys.columns[COL_BASE_FEE_VAL_L0].evaluations;
        assert_eq!(
            gadget_l0[0].to_bytes(),
            evm_l0_bytes,
            "gadget base_fee_value_l0 row 0 must equal EVM main BASEFEE row's OUTPUT0_L0"
        );

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&basefee_polys, &basefee_cs)];
        let linkages = vec![make_evm_basefee_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace (running `BASEFEE;STOP`) and a 1-row
    /// basefee_air witness pinned to the same pushed value.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + basefee_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let basefee_polys = build_basefee_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let basefee_cs = BaseFeeConstraintSystem::new(basefee_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&basefee_polys, &basefee_cs)];
        let linkages = vec![make_evm_basefee_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + basefee gadget) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's BASEFEE-row push limb-0 multiset \
             equals B's real-row base_fee_value_l0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &basefee_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ basefee gadget linkage",
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
        let basefee_polys = build_basefee_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let basefee_cs = BaseFeeConstraintSystem::new(basefee_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&basefee_polys, &basefee_cs)];
        let linkages = vec![make_evm_basefee_to_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &basefee_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched linkage closures",
        );
    }
}
