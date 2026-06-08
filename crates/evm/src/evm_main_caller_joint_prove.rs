//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **context-readers gadget AIR**
//! on the CALLER opcode (`0x33`).
//!
//! Mirrors `crate::evm_main_context_joint_prove`, which pairs the same
//! gadget AIR with EVM main on CHAINID. Bytecode `[0x33, 0x00]`
//! (CALLER; STOP).
//!
//!   - The EVM main row at the CALLER step exposes the pushed u256 in
//!     `COL_OUTPUT0_L0..L3` and sets `COL_SEL_CALLER = 1`. The gadget
//!     witness reads the OUTPUT0_L0 value out of the EVM trace, so the
//!     test is robust to the executor's default caller address.
//!   - The context_readers_air row mirrors the same value bytes/limbs
//!     with `sel_caller = 1`, `is_real = 1`.
//!
//! The descriptor binds **`COL_OUTPUT0_L0` ↔ `COL_VALUE_LIMB_0`** as a
//! single-column tuple. The gadget AIR's row-local constraints pin the
//! higher limbs to the caller address bytes (and the address byte sum
//! to `value_limb_*`), so the limb-0 binding is sufficient to bind the
//! full 4-limb push output.
//!
//! Slow tests are `#[ignore]`-gated for the same reason as the ADDRESS
//! template: BLS48-581 joint_prove takes O(30-60s) release.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::context_readers_air::{COL_SEL_CALLER as GADGET_COL_SEL_CALLER, COL_VALUE_LIMB_0};
use crate::trace::{COL_OUTPUT0_L0, COL_SEL_CALLER};

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_OUTPUT0_L0)` on CALLER rows to context_readers_air's
/// `(COL_VALUE_LIMB_0)` on caller-selector rows.
///
/// - A side: EVM main `COL_OUTPUT0_L0` gated by `COL_SEL_CALLER`.
/// - B side: context_readers_air `COL_VALUE_LIMB_0` gated by
///   `COL_SEL_CALLER` (gadget side).
pub fn make_evm_caller_to_context_gadget_descriptor(
    evm_layer_index: usize,
    context_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_caller_to_context_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_CALLER),
        b_layer_index: context_gadget_layer_index,
        b_columns: vec![COL_VALUE_LIMB_0],
        b_selector_column: Some(GADGET_COL_SEL_CALLER),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::context_readers_air::{
        build_trace_polynomials as build_context_trace, ContextReadersConstraintSystem,
        ContextReadersWitness, CALLER_OPCODE,
    };
    use crate::executor::execute_bytecode;
    use crate::trace::COL_SEL_CALLER;
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// Bytecode used by all tests: CALLER; STOP.
    fn caller_stop_bytecode() -> Vec<u8> {
        vec![0x33, 0x00]
    }

    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = caller_stop_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for CALLER;STOP");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Pull the 32-byte value (BE) representing the caller address from
    /// the EVM main's CALLER row and re-pack it into the `[u8; 32]`
    /// little-endian limb-byte layout the gadget witness expects.
    fn extract_caller_value(
        evm_polys: &TracePolynomials,
    ) -> Result<[u8; 32], &'static str> {
        use crate::trace::{COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3};
        let sel_col = &evm_polys.columns[COL_SEL_CALLER].evaluations;
        let curve = evm_polys.curve;
        let one_bytes = Scalar::one(curve).to_bytes();
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                let mut val = [0u8; 32];
                for (i, col) in [
                    COL_OUTPUT0_L0,
                    COL_OUTPUT0_L1,
                    COL_OUTPUT0_L2,
                    COL_OUTPUT0_L3,
                ]
                .iter()
                .enumerate()
                {
                    let be = evm_polys.columns[*col].evaluations[r].to_bytes();
                    let mut tail = [0u8; 8];
                    tail.copy_from_slice(&be[be.len() - 8..]);
                    let limb = u64::from_be_bytes(tail);
                    val[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
                }
                return Ok(val);
            }
        }
        Err("EVM main trace must contain at least one CALLER row")
    }

    fn build_context_gadget_trace(
        evm_polys: &TracePolynomials,
        curve: CurveType,
    ) -> TracePolynomials {
        let value = extract_caller_value(evm_polys)
            .expect("EVM main trace must contain at least one CALLER row");
        let w = ContextReadersWitness::from_events(&[(CALLER_OPCODE, 0u64, value)]);
        build_context_trace(&w, curve)
    }

    /// Fast: descriptor structure + closure alignment.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_caller_to_context_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_caller_to_context_gadget_v1");
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
        assert_eq!(d.a_selector_column, Some(COL_SEL_CALLER));
        assert_eq!(d.b_selector_column, Some(GADGET_COL_SEL_CALLER));

        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let ctx_polys = build_context_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let ctx_cs = ContextReadersConstraintSystem::new(ctx_polys.num_rows);

        let sel_col = &evm_polys.columns[COL_SEL_CALLER].evaluations;
        let out0_col = &evm_polys.columns[COL_OUTPUT0_L0].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found = None;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                found = Some(out0_col[r].to_bytes());
                break;
            }
        }
        let evm_l0_bytes = found.expect("EVM main trace must contain at least one CALLER row");

        let gadget_l0 = &ctx_polys.columns[COL_VALUE_LIMB_0].evaluations;
        assert_eq!(
            gadget_l0[0].to_bytes(),
            evm_l0_bytes,
            "gadget value_limb_0 row 0 must equal EVM main CALLER row's OUTPUT0_L0"
        );

        let gadget_sel = &ctx_polys.columns[GADGET_COL_SEL_CALLER].evaluations;
        assert_eq!(
            gadget_sel[0].to_bytes(),
            one_bytes,
            "gadget sel_caller must fire on the CALLER event row"
        );

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&ctx_polys, &ctx_cs)];
        let linkages = vec![make_evm_caller_to_context_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);

        assert_eq!(CALLER_OPCODE, 0x33);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + context_readers_air CALLER (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let ctx_polys = build_context_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let ctx_cs = ContextReadersConstraintSystem::new(ctx_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&ctx_polys, &ctx_cs)];
        let linkages = vec![make_evm_caller_to_context_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + context CALLER) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's CALLER-row push limb-0 multiset \
             equals B's caller-row value_limb_0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &ctx_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ context CALLER linkage",
        );
    }

    /// Tampered envelope: corrupt `closure_a` so the joint verifier rejects.
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
        let ctx_polys = build_context_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let ctx_cs = ContextReadersConstraintSystem::new(ctx_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&ctx_polys, &ctx_cs)];
        let linkages = vec![make_evm_caller_to_context_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &ctx_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched linkage closures",
        );
    }
}
