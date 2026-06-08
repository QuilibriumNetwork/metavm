//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **LOG opcode gadget AIR**
//! (`crate::log_air`) on a LOG2 row.
//!
//! Witness: the minimal bytecode
//!
//!   PUSH1 0xAA  PUSH1 0xBB  PUSH1 0x00  PUSH1 0x00  LOG2  STOP
//!   = [0x60, 0xAA, 0x60, 0xBB, 0x60, 0x00, 0x60, 0x00, 0xA2, 0x00]
//!
//! The stack at LOG2 (top first) is `[offset=0, length=0, topic0=0xBB,
//! topic1=0xAA]`. The EVM inspector captures `topic0 = stack[2] = 0xBB`
//! in the `immediate` column for LOG1+ rows (see `crate::inspector`
//! around the `0xA1..=0xA4` arm), so the LOG2 row in the EVM main trace
//! has:
//!
//!   - `COL_SEL_LOG2 = 1`
//!   - `COL_IMMEDIATE_L0 = 0xBB`, `COL_IMMEDIATE_L{1,2,3} = 0`
//!
//! The companion `log_air` gadget AIR is populated with a single
//! `LogRow { kind: 2, topic0: [0xBB, 0, 0, 0], expected: [0xBB, 0, 0, 0] }`,
//! so row 0 has `COL_TOPIC0_L0 = 0xBB` and `COL_IS_REAL = 1`.
//!
//! `joint_prove`'s current API only accepts single-column tuples (see
//! `crates/zkp/src/cross_air_logup.rs` docs on `build_linkage_trace`),
//! so this module exposes a **single-column** descriptor binding
//! `COL_IMMEDIATE_L0` ↔ `COL_TOPIC0_L0`. The gadget AIR's row-local
//! `topic0 == expected` constraints already pin the remaining three
//! limbs of `topic0` to `expected`, so binding limb-0 is sufficient
//! algebraic provenance for the LOG2 topic0 oracle witness on the
//! shared-LSB witness used here. The slow honest / tampered tests
//! follow the same shape as `evm_main_address_joint_prove`.
//!
//! Patterned after:
//!   - `crates/evm/src/evm_main_address_joint_prove.rs` — sibling
//!     joint_prove on ADDRESS;STOP for the address gadget AIR.
//!
//! Slow tests are `#[ignore]`-gated because `joint_prove` on a BLS48-581
//! trace pair with EVM main runs in the tens of seconds in `--release`.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::log_air::{COL_IS_REAL, COL_TOPIC0_L0};
use crate::trace::{COL_IMMEDIATE_L0, COL_SEL_LOG2};

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_IMMEDIATE_L0)` on LOG2 rows to log_air's `(COL_TOPIC0_L0)` on
/// real rows.
///
/// - A side: EVM main `COL_IMMEDIATE_L0` gated by `COL_SEL_LOG2`.
/// - B side: log_air `COL_TOPIC0_L0` gated by `COL_IS_REAL`.
///
/// Combined with the gadget AIR's row-local `topic0_l{0..3} ==
/// expected_l{0..3}` equality constraints, this pins the EVM main
/// LOG2-row immediate-limb-0 to the gadget's canonical topic0
/// witness.
pub fn make_evm_log2_to_gadget_descriptor(
    evm_layer_index: usize,
    log_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_log2_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_IMMEDIATE_L0],
        a_selector_column: Some(COL_SEL_LOG2),
        b_layer_index: log_gadget_layer_index,
        b_columns: vec![COL_TOPIC0_L0],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use crate::log_air::{
        build_trace_polynomials as build_log_trace, LogConstraintSystem, LogRow, LogWitness,
    };
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// topic0 value as encoded by the bytecode (PUSH1 0xBB pushed before
    /// the two zero offsets/lengths, leaving it at stack[2] at LOG2).
    const TOPIC0_LIMB0: u64 = 0xBB;

    /// Minimal bytecode used by all tests:
    /// PUSH1 0xAA; PUSH1 0xBB; PUSH1 0x00; PUSH1 0x00; LOG2; STOP.
    fn log2_stop_bytecode() -> Vec<u8> {
        vec![0x60, 0xAA, 0x60, 0xBB, 0x60, 0x00, 0x60, 0x00, 0xA2, 0x00]
    }

    /// Helper: build the EVM main trace by running the LOG2 bytecode.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = log2_stop_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for LOG2;STOP");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Helper: build the log_air gadget trace pinned to a single LOG2
    /// row whose topic0 mirrors the EVM main immediate captured on the
    /// LOG2 step.
    fn build_log_gadget_trace(curve: CurveType) -> TracePolynomials {
        let topic = [TOPIC0_LIMB0, 0, 0, 0];
        let w = LogWitness::from_rows(vec![LogRow {
            kind: 2,
            topic0: topic,
            expected: topic,
        }]);
        build_log_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_log2_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_log2_to_gadget_v1");
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
        assert_eq!(d.a_columns[0], COL_IMMEDIATE_L0);
        assert_eq!(d.b_columns[0], COL_TOPIC0_L0);
        assert_eq!(d.a_selector_column, Some(COL_SEL_LOG2));
        assert_eq!(d.b_selector_column, Some(COL_IS_REAL));

        // Build both traces and verify the LOG2 row's immediate-limb-0
        // matches the gadget topic0-limb-0 so closures will match by
        // construction. This catches inspector/encoding regressions
        // without paying for the prover.
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let log_polys = build_log_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let log_cs = LogConstraintSystem::new(log_polys.num_rows);

        let sel_col = &evm_polys.columns[COL_SEL_LOG2].evaluations;
        let imm_col = &evm_polys.columns[COL_IMMEDIATE_L0].evaluations;
        let expected_bytes = Scalar::from_u64(TOPIC0_LIMB0, curve).to_bytes();
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    imm_col[r].to_bytes(),
                    expected_bytes,
                    "EVM main LOG2 row's IMMEDIATE_L0 must equal topic0 LSB \
                     (0xBB pushed at stack[2] before LOG2)"
                );
                found = true;
                break;
            }
        }
        assert!(found, "EVM main trace must contain at least one LOG2 row");

        // Gadget side: row 0 must mirror the same topic0-limb-0.
        let gadget_l0 = &log_polys.columns[COL_TOPIC0_L0].evaluations;
        assert_eq!(
            gadget_l0[0].to_bytes(),
            expected_bytes,
            "gadget topic0_l0 row 0 must equal TOPIC0_LIMB0"
        );

        // Sanity: the two CS instances and traces can be assembled into
        // the (trace, cs) pair shape `joint_prove` consumes.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&log_polys, &log_cs)];
        let linkages = vec![make_evm_log2_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace (running the LOG2 bytecode) and a 1-row
    /// log_air gadget witness pinned to the same topic0.
    ///
    /// Marked `#[ignore]` because `joint_prove` runs a per-AIR
    /// `prove_with_scheme` for each of EVM main + log_air plus a
    /// per-linkage prover, taking tens of seconds on BLS48-581 in
    /// `--release`.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + log_air (BLS48-581, tens of seconds release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let log_polys = build_log_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let log_cs = LogConstraintSystem::new(log_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&log_polys, &log_cs)];
        let linkages = vec![make_evm_log2_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + log gadget) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's LOG2-row immediate-limb-0 multiset \
             equals B's real-row topic0-limb-0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &log_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ log gadget linkage",
        );
    }

    /// Tampered witness: corrupt `closure_a` in the extension envelope
    /// so the joint verifier's `closure_a == closure_b` check fires
    /// and rejects.
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
        let log_polys = build_log_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let log_cs = LogConstraintSystem::new(log_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&log_polys, &log_cs)];
        let linkages = vec![make_evm_log2_to_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed before tampering");

        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[0].closure_a = one_bytes;

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &log_cs];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatched linkage closures",
        );
    }
}
