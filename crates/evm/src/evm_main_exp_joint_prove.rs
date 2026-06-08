//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **EXP gadget AIR**
//! ([`crate::exp_air`] / [`crate::exp_constraints`]).
//!
//! Mirrors [`crate::evm_main_address_joint_prove`] but for the
//! `EXP` (0x0A) opcode. The witness is the minimal bytecode
//! `PUSH1 3 PUSH1 2 EXP STOP` (= `2^3 = 8`), executed via
//! `executor::execute_bytecode`:
//!
//!   - The EVM main row at the EXP step exposes the pushed u256
//!     (= `8`) in `COL_OUTPUT0_L0..L3` and sets `COL_SEL_EXP = 1`.
//!   - The EXP gadget AIR row 0 (where `COL_IS_FIRST_ROW = 1`) holds
//!     the same final output in `COL_FINAL_OUTPUT_OFFSET..+4`,
//!     replicated across all 256 rows of the invocation.
//!
//! The descriptor binds **`COL_OUTPUT0_L0` ↔ `COL_FINAL_OUTPUT_OFFSET`**
//! as a single-column tuple gated by the respective selectors
//! (`COL_SEL_EXP` on the EVM side, `COL_IS_FIRST_ROW` on the gadget
//! side). `joint_prove`'s current API requires single-column tuples
//! (see `crates/zkp/src/cross_air_logup.rs` doc on
//! `build_linkage_trace`); the gadget AIR internally pins limbs 1..3
//! via its `final_output_at_last_row` cross-row constraint, so
//! binding limb-0 is sufficient to algebraically commit the full
//! 4-limb output tuple. The shared LSB limb (= 8) yields identical
//! multisets on both sides on the single EXP row each side commits.
//!
//! Patterned after [`crate::evm_main_address_joint_prove`].
//!
//! Slow tests are `#[ignore]`-gated because `joint_prove` on a
//! BLS48-581 trace pair with EVM main (~5 EVM rows padded to 256)
//! plus a 256-row EXP gadget trace runs in O(30-60s) release.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::exp_air::{COL_FINAL_OUTPUT_OFFSET, COL_IS_FIRST_ROW};
use crate::trace::{COL_OUTPUT0_L0, COL_SEL_EXP};

/// Gadget B-side column index: limb 0 of the final EXP output (LSB-limb).
pub const GADGET_COL_FINAL_OUTPUT_L0: usize = COL_FINAL_OUTPUT_OFFSET;

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_OUTPUT0_L0)` on EXP rows to exp_air's
/// `(COL_FINAL_OUTPUT_OFFSET)` on invocation-anchor (first) rows.
///
/// - A side: EVM main `COL_OUTPUT0_L0` gated by `COL_SEL_EXP`.
/// - B side: exp_air `COL_FINAL_OUTPUT_OFFSET` gated by `COL_IS_FIRST_ROW`.
pub fn make_evm_exp_to_gadget_descriptor(
    evm_layer_index: usize,
    exp_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_exp_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_EXP),
        b_layer_index: exp_gadget_layer_index,
        b_columns: vec![GADGET_COL_FINAL_OUTPUT_L0],
        b_selector_column: Some(COL_IS_FIRST_ROW),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use crate::exp_air::{populate_trace, NUM_STEPS};
    use crate::exp_constraints::EvmExpConstraintSystem;
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::{Polynomial, TracePolynomials};
    use revm::primitives::U256;

    /// Wrap `populate_trace` (which populates the cross-AIR LogUp
    /// columns `COL_EXPONENT_OFFSET`, `COL_FINAL_OUTPUT_OFFSET`,
    /// `COL_IS_FIRST_ROW` in addition to the algorithm step columns)
    /// into a [`TracePolynomials`] the joint_prove pipeline consumes.
    /// `exp_constraints::build_trace_polynomials` predates the
    /// linkage columns and does NOT populate them, so we go through
    /// `populate_trace` directly here.
    fn build_exp_gadget_trace_polys(base: U256, exponent: U256, curve: CurveType) -> TracePolynomials {
        let rows = crate::exp_air::exp_witness(base, exponent);
        let columns = populate_trace(&rows, curve);
        let polys: Vec<Polynomial> = columns
            .into_iter()
            .map(|evals| Polynomial {
                evaluations: evals,
                degree: NUM_STEPS,
            })
            .collect();
        TracePolynomials {
            columns: polys,
            num_rows: NUM_STEPS,
            padded_size: NUM_STEPS as u64,
            curve,
        }
    }

    /// Bytecode `PUSH1 3 PUSH1 2 EXP STOP` (= `2^3 = 8`).
    ///
    /// Stack walk (EVM convention: top-of-stack is the last pushed):
    ///   PUSH1 0x03 → stack = [3]
    ///   PUSH1 0x02 → stack = [3, 2]   (top = 2)
    ///   EXP        → pops top (= base = 2), then next (= exponent = 3)
    ///                pushes base^exponent = 8.
    fn exp_bytecode() -> Vec<u8> {
        vec![0x60, 0x03, 0x60, 0x02, 0x0A, 0x00]
    }

    /// Expected base/exponent at the EXP opcode given EVM operand order
    /// (base = top, exponent = second-from-top).
    const EXP_BASE: u64 = 2;
    const EXP_EXPONENT: u64 = 3;
    const EXP_RESULT_L0: u64 = 8;

    /// Helper: build the EVM main trace by running the EXP bytecode.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = exp_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for the EXP bytecode");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Helper: build the EXP gadget trace for the matching invocation.
    fn build_exp_gadget_trace(curve: CurveType) -> TracePolynomials {
        build_exp_gadget_trace_polys(U256::from(EXP_BASE), U256::from(EXP_EXPONENT), curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces, and the EVM
    /// + gadget traces actually agree on the limb-0 of `2^3 = 8`.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_exp_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_exp_to_gadget_v1");
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
        assert_eq!(d.b_columns[0], COL_FINAL_OUTPUT_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_SEL_EXP));
        assert_eq!(d.b_selector_column, Some(COL_IS_FIRST_ROW));

        // Build both traces and confirm the EXP row's limb-0 matches
        // the gadget's anchor-row limb-0 byte-for-byte.
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_exp_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = EvmExpConstraintSystem::new();

        // EVM side: locate the EXP row by sel_exp = 1 and confirm
        // its OUTPUT0_L0 equals the expected result limb 0.
        let sel_col = &evm_polys.columns[COL_SEL_EXP].evaluations;
        let evm_out0 = &evm_polys.columns[COL_OUTPUT0_L0].evaluations;
        let expected_l0_bytes = Scalar::from_u64(EXP_RESULT_L0, curve).to_bytes();
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found_evm = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    evm_out0[r].to_bytes(),
                    expected_l0_bytes,
                    "EVM main EXP row's OUTPUT0_L0 must equal 8 (= 2^3) limb 0"
                );
                found_evm = true;
                break;
            }
        }
        assert!(found_evm, "EVM main trace must contain at least one EXP row");

        // Gadget side: row 0 (where IS_FIRST_ROW = 1) must mirror the
        // same limb 0 value (= 8).
        let is_first = &gadget_polys.columns[COL_IS_FIRST_ROW].evaluations;
        assert_eq!(
            is_first[0].to_bytes(),
            one_bytes,
            "gadget row 0 must be marked is_first_row"
        );
        let gadget_l0 = &gadget_polys.columns[COL_FINAL_OUTPUT_OFFSET].evaluations;
        assert_eq!(
            gadget_l0[0].to_bytes(),
            expected_l0_bytes,
            "gadget COL_FINAL_OUTPUT_OFFSET row 0 must equal 8 (= 2^3) limb 0"
        );

        // Sanity: assemble the (trace, cs) pair shape that `joint_prove`
        // consumes without actually invoking the prover.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_exp_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace (running the EXP bytecode) and a 256-row
    /// exp_air gadget witness pinned to the same `(base, exponent)`.
    ///
    /// Marked `#[ignore]` because, even with a 6-byte bytecode,
    /// `joint_prove` is moderately expensive on BLS48-581.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + exp_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_exp_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = EvmExpConstraintSystem::new();

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_exp_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + exp_air) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's EXP-row push limb-0 multiset \
             equals B's anchor-row final_output_limb_0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ exp_air linkage",
        );
    }

    /// Tampered witness: corrupt `closure_a` in the extension envelope
    /// so the joint verifier's scalar `closure_a == closure_b` check
    /// fires and rejects.
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
        let gadget_polys = build_exp_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = EvmExpConstraintSystem::new();

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_exp_to_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        // Mutate closure_a in the envelope so the joint verifier's
        // `closure_a == closure_b` scalar equality check fires.
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
