//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **block-context-readers gadget
//! AIR** on the ORIGIN opcode (`0x32`).
//!
//! Mirrors `crate::evm_main_context_joint_prove`. Bytecode
//! `[0x32, 0x00]` (ORIGIN; STOP).
//!
//!   - The EVM main row at the ORIGIN step exposes the pushed u256 in
//!     `COL_OUTPUT0_L0..L3` and sets `COL_SEL_ORIGIN = 1`. The gadget
//!     witness reads the value out of the EVM trace so the test is
//!     robust to the executor's default tx-origin address.
//!   - The block_context_readers_air row mirrors the same value
//!     bytes/limbs with `sel_origin = 1`, `is_real = 1`.
//!
//! The descriptor binds **`COL_OUTPUT0_L0` ↔ `COL_VALUE_LIMB_0`** as a
//! single-column tuple. The gadget AIR's row-local constraints pin the
//! higher limbs to the origin address bytes via `tx_origin`, so the
//! limb-0 binding is sufficient to bind the full 4-limb push output.
//!
//! Slow tests are `#[ignore]`-gated for the same reason as the ADDRESS
//! template: BLS48-581 joint_prove takes O(30-60s) release.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::block_context_readers_air::{
    COL_SEL_ORIGIN as GADGET_COL_SEL_ORIGIN, COL_VALUE_LIMB_0, COL_VALUE_LIMB_1, COL_VALUE_LIMB_2,
    COL_VALUE_LIMB_3,
};
use crate::trace::{
    COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_ORIGIN,
};

/// Build the **4-limb tuple** descriptor binding EVM main's
/// `(COL_OUTPUT0_L0..L3)` on ORIGIN rows to block_context_readers_air's
/// `(COL_VALUE_LIMB_0..L3)` on origin-selector rows.
///
/// **Widened (task #211)** from single-column `(L0 ↔ L0)` to the full
/// 4-limb β-RLC tuple binding — closes the residual soundness gap on
/// higher u256 limbs of the tx origin address.
pub fn make_evm_origin_to_block_context_gadget_descriptor(
    evm_layer_index: usize,
    block_context_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_origin_to_block_context_gadget_v2".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![
            COL_OUTPUT0_L0,
            COL_OUTPUT0_L1,
            COL_OUTPUT0_L2,
            COL_OUTPUT0_L3,
        ],
        a_selector_column: Some(COL_SEL_ORIGIN),
        b_layer_index: block_context_gadget_layer_index,
        b_columns: vec![
            COL_VALUE_LIMB_0,
            COL_VALUE_LIMB_1,
            COL_VALUE_LIMB_2,
            COL_VALUE_LIMB_3,
        ],
        b_selector_column: Some(GADGET_COL_SEL_ORIGIN),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::block_context_readers_air::{
        build_trace_polynomials as build_block_context_trace, BlockContextReadersConstraintSystem,
        BlockContextReadersWitness, ORIGIN_OPCODE,
    };
    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use crate::block_context_readers_air::{COL_VALUE_LIMB_1, COL_VALUE_LIMB_2, COL_VALUE_LIMB_3};
    use crate::trace::{COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_ORIGIN};
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    fn origin_stop_bytecode() -> Vec<u8> {
        vec![0x32, 0x00]
    }

    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = origin_stop_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for ORIGIN;STOP");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Extract the 4 LE u64 limbs from the EVM main's ORIGIN row and
    /// pack them into the `[u8; 32]` layout the gadget witness expects.
    fn extract_origin_value(
        evm_polys: &TracePolynomials,
    ) -> Result<[u8; 32], &'static str> {
        let sel_col = &evm_polys.columns[COL_SEL_ORIGIN].evaluations;
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
        Err("EVM main trace must contain at least one ORIGIN row")
    }

    fn build_block_context_gadget_trace(
        evm_polys: &TracePolynomials,
        curve: CurveType,
    ) -> TracePolynomials {
        let value = extract_origin_value(evm_polys)
            .expect("EVM main trace must contain at least one ORIGIN row");
        let w = BlockContextReadersWitness::from_events(&[(ORIGIN_OPCODE, 0u64, value)]);
        build_block_context_trace(&w, curve)
    }

    #[test]
    fn descriptor_consistency() {
        let d = make_evm_origin_to_block_context_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_origin_to_block_context_gadget_v2");
        assert_eq!(d.a_layer_index, 0);
        assert_eq!(d.b_layer_index, 1);
        assert_ne!(d.a_layer_index, d.b_layer_index);
        assert_eq!(d.a_columns.len(), d.b_columns.len());
        assert_eq!(d.a_columns.len(), 4);
        assert_eq!(
            d.a_columns,
            vec![COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3]
        );
        assert_eq!(
            d.b_columns,
            vec![COL_VALUE_LIMB_0, COL_VALUE_LIMB_1, COL_VALUE_LIMB_2, COL_VALUE_LIMB_3]
        );
        assert_eq!(d.a_selector_column, Some(COL_SEL_ORIGIN));
        assert_eq!(d.b_selector_column, Some(GADGET_COL_SEL_ORIGIN));

        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let ctx_polys = build_block_context_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let ctx_cs = BlockContextReadersConstraintSystem::new(ctx_polys.num_rows);

        let sel_col = &evm_polys.columns[COL_SEL_ORIGIN].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let evm_limb_cols = [
            COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3,
        ];
        let gadget_limb_cols = [
            COL_VALUE_LIMB_0, COL_VALUE_LIMB_1, COL_VALUE_LIMB_2, COL_VALUE_LIMB_3,
        ];
        let mut evm_row = None;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                evm_row = Some(r);
                break;
            }
        }
        let evm_row = evm_row.expect("EVM main trace must contain at least one ORIGIN row");
        // Widened (task #211): all 4 limbs must match across EVM main
        // and the block_context gadget. For ORIGIN the value is a 20-byte
        // address — L2 high half and L3 are zero, but binding them
        // algebraically prevents a tampered EVM trace from forging
        // nonzero high limbs.
        for k in 0..4 {
            let evm_bytes = evm_polys.columns[evm_limb_cols[k]].evaluations[evm_row].to_bytes();
            let gadget_bytes = ctx_polys.columns[gadget_limb_cols[k]].evaluations[0].to_bytes();
            assert_eq!(
                evm_bytes, gadget_bytes,
                "limb {} must match across EVM main and block_context gadget",
                k,
            );
        }

        let gadget_sel = &ctx_polys.columns[GADGET_COL_SEL_ORIGIN].evaluations;
        assert_eq!(gadget_sel[0].to_bytes(), one_bytes);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&ctx_polys, &ctx_cs)];
        let linkages = vec![make_evm_origin_to_block_context_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);

        assert_eq!(ORIGIN_OPCODE, 0x32);
    }

    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + block_context_readers_air ORIGIN (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let ctx_polys = build_block_context_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let ctx_cs = BlockContextReadersConstraintSystem::new(ctx_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&ctx_polys, &ctx_cs)];
        let linkages = vec![make_evm_origin_to_block_context_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + block_context ORIGIN) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &ctx_cs];
        assert!(joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve));
    }

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
        let ctx_polys = build_block_context_gadget_trace(&evm_polys, curve);
        let evm_cs = EvmConstraintSystem::new();
        let ctx_cs = BlockContextReadersConstraintSystem::new(ctx_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&ctx_polys, &ctx_cs)];
        let linkages = vec![make_evm_origin_to_block_context_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &ctx_cs];
        assert!(!joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve));
    }
}
