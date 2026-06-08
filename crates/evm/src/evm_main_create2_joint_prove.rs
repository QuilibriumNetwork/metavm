//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **`metavm_zkp::create2_address_air`**
//! gadget AIR.
//!
//! Mirrors [`crate::evm_main_address_joint_prove`] but for the
//! `CREATE2` (0xF5) opcode. The witness is the minimal bytecode
//! `PUSH1 0 PUSH1 0 PUSH1 0 PUSH1 0 CREATE2 STOP` (CREATE2 with
//! `salt = 0`, `value = 0`, `mem_offset = 0`, `mem_length = 0`
//! ⇒ empty init code, zero salt):
//!
//!   - The EVM main row at the CREATE2 step sets `COL_SEL_CREATE2 = 1`,
//!     exposes the sender (caller) address in `COL_FRAME_CALLER_*` as
//!     4 LE u64 limbs, and captures the salt + initcode hash hints in
//!     the dedicated `COL_CREATE2_SALT_HINT_*` /
//!     `COL_CREATE2_INITCODE_HASH_HINT_*` columns.
//!   - `create2_address_air` exposes the same sender as 20 BE bytes
//!     at `COL_SENDER_OFFSET..+20` and the derived address as 20 BE
//!     bytes at `COL_DERIVED_ADDRESS_OFFSET..+20`.
//!
//! ## Encoding adapter
//!
//! The EVM main trace commits the sender as 4 LE u64 limbs at
//! `COL_FRAME_CALLEE_L0..L3` (see `crate::inspector::address_to_limbs`).
//! The `create2_address_air` gadget commits sender as 20 BE bytes at
//! `COL_SENDER_OFFSET..+20`, AND mirrors the same scalar in 4 LE u64
//! limb adapter columns `COL_SENDER_LIMB_OFFSET..+4`, with 4 dedicated
//! row-local byte-decomposition constraints algebraically binding the
//! adapter to the BE byte view. This lets the cross-AIR LogUp
//! descriptor bind on a single-column tuple
//! **`COL_FRAME_CALLEE_L0` ↔ `COL_SENDER_LIMB_OFFSET`**
//! byte-for-byte, mirroring [`crate::evm_main_create_joint_prove`].
//!
//! Patterned after [`crate::evm_main_address_joint_prove`],
//! [`crate::evm_main_create_joint_prove`], and the cross-crate
//! precedent in [`crate::evm_main_sha3_joint_prove`].

use metavm_zkp::create2_address_air::{
    COL_IS_REAL as GADGET_COL_IS_REAL, COL_SENDER_LIMB_OFFSET as GADGET_COL_SENDER_LIMB_OFFSET,
};
use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;

use crate::trace::{COL_FRAME_CALLEE_L0, COL_SEL_CREATE2};

// The CREATE2 sender (= the contract that issued CREATE2) lives in
// the EVM main row's `COL_FRAME_CALLEE_L0..L3`: the current frame's
// `callee` is the address whose code is executing, which IS the
// CREATE2 sender per EVM semantics.

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_FRAME_CALLEE_L0)` on CREATE2 rows to create2_address_air's
/// `(COL_SENDER_LIMB_OFFSET)` (i.e. sender limb 0) on real rows.
///
/// - A side: EVM main `COL_FRAME_CALLEE_L0` gated by `COL_SEL_CREATE2`.
/// - B side: create2_address_air `COL_SENDER_LIMB_OFFSET` gated by
///   `COL_IS_REAL`.
///
/// Both sides commit the same LE u64 limb-0 scalar of the 20-byte
/// sender address (see module docs); the gadget's adapter columns are
/// algebraically pinned to the BE byte view via dedicated row-local
/// byte-decomposition constraints.
pub fn make_evm_create2_to_gadget_descriptor(
    evm_layer_index: usize,
    create2_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_create2_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_FRAME_CALLEE_L0],
        a_selector_column: Some(COL_SEL_CREATE2),
        b_layer_index: create2_gadget_layer_index,
        b_columns: vec![GADGET_COL_SENDER_LIMB_OFFSET],
        b_selector_column: Some(GADGET_COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use metavm_zkp::create2_address_air::{
        build_trace_polynomials as build_create2_trace, Create2AddressConstraintSystem,
        Create2AddressWitness, SALT_LEN, SENDER_LEN,
    };
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::trace::TracePolynomials;

    /// Contract address used by `execute_bytecode` (mirrors
    /// `crate::executor`'s hard-coded `Address::from([0x42; 20])`).
    /// This is the *sender* of the CREATE2 opcode (the calling contract).
    const SENDER_ADDR: [u8; SENDER_LEN] = [0x42; SENDER_LEN];

    /// Salt argument the bytecode pushes (= 0).
    const SALT: [u8; SALT_LEN] = [0u8; SALT_LEN];

    /// Empty init code.
    fn init_code() -> Vec<u8> {
        Vec::new()
    }

    /// Minimal CREATE2 bytecode: pushes salt=0, value=0, offset=0,
    /// length=0 onto the stack then issues CREATE2.
    ///
    /// EVM CREATE2 pops `(value, offset, length, salt)` in that order;
    /// since the stack is LIFO with top-of-stack = last pushed, we
    /// push them in REVERSE order: salt, length, offset, value.
    ///
    /// Stack walk:
    ///   PUSH1 0x00 → stack = [salt]
    ///   PUSH1 0x00 → stack = [salt, length]
    ///   PUSH1 0x00 → stack = [salt, length, offset]
    ///   PUSH1 0x00 → stack = [salt, length, offset, value]
    ///   CREATE2    → pops value, offset, length, salt
    fn create2_bytecode() -> Vec<u8> {
        vec![
            0x60, 0x00, // PUSH1 0 (salt)
            0x60, 0x00, // PUSH1 0 (length)
            0x60, 0x00, // PUSH1 0 (offset)
            0x60, 0x00, // PUSH1 0 (value)
            0xF5, // CREATE2
            0x00, // STOP
        ]
    }

    /// Helper: build the EVM main trace by running the CREATE2 bytecode.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = create2_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for the CREATE2 bytecode");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Helper: build the create2_address_air gadget trace mirroring
    /// the CREATE2 invocation (sender = 0x42…42, salt = 0,
    /// init_code = empty).
    fn build_create2_gadget_trace(curve: CurveType) -> TracePolynomials {
        let w = Create2AddressWitness::from_input_tuples(&[(SENDER_ADDR, SALT, init_code())]);
        build_create2_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces. Both traces
    /// build cleanly and contain the matching selector rows. Per the
    /// module-level encoding caveat we DO NOT assert byte-equality of
    /// the bound scalars (LE u64 limb on A side vs BE byte on B side).
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_create2_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_create2_to_gadget_v1");
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
        assert_eq!(d.a_selector_column, Some(COL_SEL_CREATE2));
        assert_eq!(d.b_selector_column, Some(GADGET_COL_IS_REAL));

        // Build both traces and confirm each side contains the
        // selector-active row that `joint_prove`'s linkage trace
        // builder will pick up.
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_create2_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = Create2AddressConstraintSystem::new(gadget_polys.num_rows);

        // EVM side: at least one row has sel_create2 = 1.
        let sel_col = &evm_polys.columns[COL_SEL_CREATE2].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let mut found_evm = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                found_evm = true;
                break;
            }
        }
        assert!(
            found_evm,
            "EVM main trace must contain at least one CREATE2 (sel_create2=1) row"
        );

        // Gadget side: row 0 must be marked is_real.
        let is_real = &gadget_polys.columns[GADGET_COL_IS_REAL].evaluations;
        assert_eq!(
            is_real[0].to_bytes(),
            one_bytes,
            "gadget row 0 must be marked is_real"
        );

        // Sanity: gadget row 0's sender limb 0 equals the LE u64
        // packing of SENDER_ADDR[0..8]. Mirrors the EVM main side's
        // `address_to_limbs(SENDER_ADDR)[0]`.
        use metavm_zkp::create2_address_air::sender_limbs_from_bytes;
        let expected_l0 = sender_limbs_from_bytes(&SENDER_ADDR)[0];
        let sender_l0 = &gadget_polys.columns[GADGET_COL_SENDER_LIMB_OFFSET].evaluations;
        assert_eq!(
            sender_l0[0].to_bytes(),
            Scalar::from_u64(expected_l0, curve).to_bytes(),
            "gadget COL_SENDER_LIMB_OFFSET row 0 must equal sender limb 0"
        );

        // And the gadget's COL_FRAME_CALLEE_L0 on the EVM side must
        // match the same limb on every CREATE2-selected row.
        let evm_callee_l0 = &evm_polys.columns[COL_FRAME_CALLEE_L0].evaluations;
        let expected_l0_bytes = Scalar::from_u64(expected_l0, curve).to_bytes();
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    evm_callee_l0[r].to_bytes(),
                    expected_l0_bytes,
                    "EVM main CREATE2 row's FRAME_CALLEE_L0 must equal sender limb 0"
                );
            }
        }

        // Sanity: assemble the (trace, cs) pair shape that `joint_prove`
        // consumes without actually invoking the prover.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_create2_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace (running the CREATE2 bytecode) and a 1-row
    /// create2_address_air witness pinned to the same sender + salt +
    /// init_code.
    ///
    /// Both sides commit the sender limb 0 as the same LE u64 scalar
    /// (the gadget's `COL_SENDER_LIMB_OFFSET` adapter is algebraically
    /// pinned to its BE byte view via dedicated row-local byte
    /// decomposition constraints), so the linkage closures match
    /// byte-for-byte.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + create2_address_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_create2_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = Create2AddressConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_create2_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + create2_address_air) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's CREATE2-row caller limb-0 multiset \
             equals B's real-row sender_limb_0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must accept the honest 2-AIR proof",
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
        let gadget_polys = build_create2_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = Create2AddressConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_create2_to_gadget_descriptor(0, 1)];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed structurally before tampering");

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
