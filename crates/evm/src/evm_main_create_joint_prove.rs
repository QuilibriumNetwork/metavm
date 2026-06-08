//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **`metavm_zkp::evm_create_rlp_air`**
//! gadget AIR.
//!
//! Mirrors [`crate::evm_main_address_joint_prove`] but for the
//! `CREATE` (0xF0) opcode. The witness is the minimal bytecode
//! `PUSH1 0 PUSH1 0 PUSH1 0 CREATE STOP` (CREATE with `value = 0`,
//! `mem_offset = 0`, `mem_length = 0` ⇒ empty init code):
//!
//!   - The EVM main row at the CREATE step sets `COL_SEL_CREATE = 1`,
//!     exposes the sender (caller) address in `COL_FRAME_CALLER_*` as
//!     4 LE u64 limbs.
//!   - `evm_create_rlp_air` exposes the same sender as 4 LE u64 limbs
//!     at `COL_SENDER_LIMB_OFFSET..+4`. Both encodings agree by
//!     construction (`metavm_zkp::evm_create_rlp_air::sender_limbs_from_bytes`
//!     mirrors `crate::inspector::address_to_limbs`).
//!
//! The descriptor binds **`COL_FRAME_CALLEE_L0` ↔
//! `COL_SENDER_LIMB_OFFSET`** as a single-column tuple gated by the
//! respective selectors (`COL_SEL_CREATE` on the EVM side,
//! `COL_IS_REAL` on the gadget side). `joint_prove`'s current API
//! requires single-column tuples (see
//! `crates/zkp/src/cross_air_logup.rs` doc on `build_linkage_trace`);
//! `make_evm_main_create_rlp_linkage_descriptor` already exists in the
//! zkp crate but binds a 5-tuple `(sender_l0..l3, nonce)`; this
//! module reduces to the limb-0 single-tuple shape `joint_prove`
//! consumes. The gadget AIR internally pins limbs 1..3 via its
//! `sender_limb_binding` row-local constraints, so limb-0 alone is
//! sufficient to algebraically commit the full sender tuple.
//!
//! Patterned after [`crate::evm_main_address_joint_prove`] and the
//! cross-crate precedent in [`crate::evm_main_sha3_joint_prove`].
//!
//! Slow tests are `#[ignore]`-gated because `joint_prove` on a
//! BLS48-581 trace pair with EVM main plus a CREATE-RLP gadget trace
//! runs in O(30-60s) release.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;
use metavm_zkp::evm_create_rlp_air::{
    COL_IS_REAL as GADGET_COL_IS_REAL,
    COL_SENDER_LIMB_OFFSET as GADGET_COL_SENDER_LIMB_OFFSET,
};

use crate::trace::{COL_FRAME_CALLEE_L0, COL_SEL_CREATE};

// The CREATE sender (= the contract that issued CREATE) lives in the
// EVM main row's `COL_FRAME_CALLEE_L0..L3`: the current frame's
// `callee` is the address whose code is executing, which IS the
// CREATE sender per EVM semantics. `COL_FRAME_CALLER_L0` instead
// names the address that CALLED into the current frame (the EOA for
// the top-level frame), which is NOT the CREATE sender.

/// Gadget B-side column index: limb 0 of the sender's 4 LE u64 limbs.
pub const GADGET_COL_SENDER_LIMB_0: usize = GADGET_COL_SENDER_LIMB_OFFSET;

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_FRAME_CALLEE_L0)` on CREATE rows to evm_create_rlp_air's
/// `(COL_SENDER_LIMB_OFFSET)` on real rows.
///
/// - A side: EVM main `COL_FRAME_CALLEE_L0` gated by `COL_SEL_CREATE`.
/// - B side: evm_create_rlp_air `COL_SENDER_LIMB_OFFSET` gated by
///   `COL_IS_REAL`.
///
/// Combined with the gadget AIR's per-limb byte-decomposition and
/// nonce-bit-decomposition row-local constraints, this pins the EVM
/// main CREATE row's sender to the gadget's canonical
/// `sender_limbs_from_bytes(sender)`.
pub fn make_evm_create_to_gadget_descriptor(
    evm_layer_index: usize,
    create_rlp_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_create_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_FRAME_CALLEE_L0],
        a_selector_column: Some(COL_SEL_CREATE),
        b_layer_index: create_rlp_gadget_layer_index,
        b_columns: vec![GADGET_COL_SENDER_LIMB_0],
        b_selector_column: Some(GADGET_COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::address_opcode_air::address_to_limbs;
    use crate::executor::execute_bytecode;
    use metavm_zkp::evm_create_rlp_air::{
        build_trace_polynomials as build_create_rlp_trace, CreateRlpConstraintSystem,
        CreateRlpWitness,
    };
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// Contract address used by `execute_bytecode` (mirrors
    /// `crate::executor`'s hard-coded `Address::from([0x42; 20])`).
    /// This is the *sender* of the CREATE opcode (the calling contract).
    const SENDER_ADDR: [u8; 20] = [0x42; 20];

    /// Minimal CREATE bytecode: pushes value=0, offset=0, length=0
    /// onto the stack then issues CREATE.
    ///
    /// Stack walk (EVM convention: top-of-stack is the last pushed):
    ///   PUSH1 0x00 → stack = [0]
    ///   PUSH1 0x00 → stack = [0, 0]
    ///   PUSH1 0x00 → stack = [0, 0, 0]
    ///   CREATE     → pops (value, offset, length) = (0, 0, 0)
    ///                creates a contract with empty init code
    fn create_bytecode() -> Vec<u8> {
        vec![0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF0, 0x00]
    }

    /// Helper: build the EVM main trace by running the CREATE bytecode.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = create_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for the CREATE bytecode");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Helper: build the evm_create_rlp_air gadget trace mirroring the
    /// CREATE invocation in the EVM trace. The pre-bump sender nonce
    /// for the first CREATE issued by this contract is 0 (the
    /// executor's fresh sender starts with nonce = 0; the inspector's
    /// `create_nonce_hint` column captures this pre-bump value).
    fn build_create_rlp_gadget_trace(curve: CurveType) -> TracePolynomials {
        let w = CreateRlpWitness::from_inputs(&[(&SENDER_ADDR, 0u64)])
            .expect("nonce 0 is within MAX_NONCE_EXCLUSIVE");
        build_create_rlp_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces, and the EVM
    /// + gadget traces actually agree on the limb-0 of the sender.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_create_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_create_to_gadget_v1");
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
        assert_eq!(d.a_columns[0], COL_FRAME_CALLEE_L0);
        assert_eq!(d.b_columns[0], GADGET_COL_SENDER_LIMB_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_SEL_CREATE));
        assert_eq!(d.b_selector_column, Some(GADGET_COL_IS_REAL));

        // Build both traces and confirm the CREATE row's caller-limb-0
        // matches the gadget's row-0 sender-limb-0 byte-for-byte.
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_create_rlp_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = CreateRlpConstraintSystem::new(gadget_polys.num_rows);

        // EVM side: locate the CREATE row by sel_create = 1 and confirm
        // its FRAME_CALLEE_L0 equals limb 0 of the sender.
        let sel_col = &evm_polys.columns[COL_SEL_CREATE].evaluations;
        let evm_caller = &evm_polys.columns[COL_FRAME_CALLEE_L0].evaluations;
        let expected_limbs = address_to_limbs(&SENDER_ADDR);
        let expected_l0_bytes = Scalar::from_u64(expected_limbs[0], curve).to_bytes();
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found_evm = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    evm_caller[r].to_bytes(),
                    expected_l0_bytes,
                    "EVM main CREATE row's FRAME_CALLEE_L0 must equal sender limb 0"
                );
                found_evm = true;
                break;
            }
        }
        assert!(
            found_evm,
            "EVM main trace must contain at least one CREATE (sel_create=1) row"
        );

        // Gadget side: row 0 must mirror the same limb-0.
        let is_real = &gadget_polys.columns[GADGET_COL_IS_REAL].evaluations;
        assert_eq!(
            is_real[0].to_bytes(),
            one_bytes,
            "gadget row 0 must be marked is_real"
        );
        let gadget_l0 = &gadget_polys.columns[GADGET_COL_SENDER_LIMB_OFFSET].evaluations;
        assert_eq!(
            gadget_l0[0].to_bytes(),
            expected_l0_bytes,
            "gadget COL_SENDER_LIMB_OFFSET row 0 must equal sender limb 0"
        );

        // Sanity: assemble the (trace, cs) pair shape that `joint_prove`
        // consumes without actually invoking the prover.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_create_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace (running the CREATE bytecode) and a 1-row
    /// evm_create_rlp_air witness pinned to the same sender + nonce.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + evm_create_rlp_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_create_rlp_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = CreateRlpConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_create_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + evm_create_rlp_air) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's CREATE-row caller limb-0 multiset \
             equals B's real-row sender_limb_0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ evm_create_rlp_air linkage",
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
        let gadget_polys = build_create_rlp_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = CreateRlpConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_create_to_gadget_descriptor(0, 1)];

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
