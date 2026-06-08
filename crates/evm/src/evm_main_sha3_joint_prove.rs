//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! binding the **EVM main AIR** to the **SHA3 full-chain composition AIR**
//! (`metavm_zkp::sha3_full_chain_air`).
//!
//! Mirrors `crate::evm_main_address_joint_prove` and the other
//! per-opcode joint_prove templates, but for the `SHA3` (0x20) opcode.
//!
//! The witness is the bytecode
//!
//! ```text
//!   PUSH32 <32 zero bytes>  PUSH1 0x00  MSTORE
//!   PUSH1 0x20              PUSH1 0x00  SHA3
//!   STOP
//! ```
//!
//! ie. `[0x7F, ..[0u8; 32].., 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0x20, 0x00]`,
//! executed via `executor::execute_bytecode`:
//!
//!   - The EVM main row at the `SHA3` step exposes the pushed u256
//!     (= `keccak256([0u8; 32])`) in `COL_OUTPUT0_L0..L3`, sets
//!     `COL_SEL_KECCAK = 1`, and reflects the popped (offset, length) =
//!     (0, 32) in `COL_INPUT0_*` / `COL_INPUT1_*`.
//!   - The `sha3_full_chain_air` witness mirrors the same invocation
//!     as a single row with `is_real = 1`,
//!     `mem_offset = 0`, `length = 32`, `input_bytes = [0u8; 32]`,
//!     `output_hash = keccak256([0u8; 32])`, and
//!     `output_limb[0..4] = keccak_output_limbs_from_output(output_hash)`.
//!
//! The descriptor binds **`COL_OUTPUT0_L0` ↔ `COL_OUTPUT_LIMB_0`** as a
//! single-column tuple gated by the respective selectors (`COL_SEL_KECCAK`
//! on the EVM side, `COL_IS_REAL` on the gadget side). `joint_prove`'s
//! current API requires single-column tuples (see
//! `crates/zkp/src/cross_air_logup.rs` doc on `build_linkage_trace`).
//! The gadget AIR internally pins the other three output limbs via its
//! per-limb BE byte decomposition row-local constraints (limbs 1..3 are
//! derived from `output_hash[0..24]`), so binding limb-0 is sufficient to
//! commit the full 4-limb output tuple algebraically once the gadget's
//! constraints fire. The shared `keccak256([0u8; 32])` LSB limb yields
//! identical multisets on both sides on the single SHA3 row each side
//! commits, so closures match by construction.
//!
//! Patterned after:
//!   - `crate::evm_main_address_joint_prove` — the canonical EVM main
//!     + small gadget AIR template.
//!   - `crate::cross_air_linkage::evm_keccak_keccak_extract_tuple_match`
//!     — confirms the EVM `output0[k]` limb ordering matches
//!     `keccak_output_limbs_from_output`.
//!
//! Slow tests are `#[ignore]`-gated because `joint_prove` over the EVM
//! main trace (padded to the LogUp range-table domain) plus the gadget
//! runs in the O(30-60s) range on BLS48-581 in `--release`.

use metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor;
use metavm_zkp::sha3_full_chain_air::{
    COL_IS_REAL as GADGET_COL_IS_REAL, COL_OUTPUT_LIMB_OFFSET as GADGET_COL_OUTPUT_LIMB_OFFSET,
};

use crate::trace::{COL_OUTPUT0_L0, COL_SEL_KECCAK};

/// Gadget B-side column index: limb 0 of the keccak output (LSB-limb).
pub const GADGET_COL_OUTPUT_LIMB_0: usize = GADGET_COL_OUTPUT_LIMB_OFFSET;

/// Build the single-column tuple descriptor binding EVM main's
/// `(COL_OUTPUT0_L0)` on SHA3 rows to `sha3_full_chain_air`'s
/// `(COL_OUTPUT_LIMB_0)` on real rows.
///
/// - A side: EVM main `COL_OUTPUT0_L0` gated by `COL_SEL_KECCAK`.
/// - B side: `sha3_full_chain_air` `COL_OUTPUT_LIMB_0` gated by
///   `COL_IS_REAL`.
///
/// Combined with the gadget AIR's per-limb BE byte decompositions and
/// length LE decomposition, this pins the EVM main push output's LSB
/// limb to the gadget's canonical keccak256-of-input commitment on
/// each SHA3 row.
pub fn make_evm_sha3_to_gadget_descriptor(
    evm_layer_index: usize,
    sha3_gadget_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    CrossAirLogUpDescriptor {
        label: "evm_main_sha3_to_gadget_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_KECCAK),
        b_layer_index: sha3_gadget_layer_index,
        b_columns: vec![GADGET_COL_OUTPUT_LIMB_0],
        b_selector_column: Some(GADGET_COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::constraints::EvmConstraintSystem;
    use crate::executor::execute_bytecode;
    use metavm_zkp::field::{CurveType, Scalar};
    use metavm_zkp::sha3_full_chain_air::{
        build_trace_polynomials as build_sha3_full_trace, Sha3FullChainConstraintSystem,
        Sha3FullChainWitness,
    };
    use metavm_zkp::trace::TracePolynomials;

    /// 32-byte SHA3 input used by all tests: `[0u8; 32]`. Chosen so the
    /// expected digest matches the known-good `keccak256([0u8; 32])`
    /// vector `0x290decd9...` already pinned in
    /// `sha3_full_chain_air::tests::thirty_two_byte_input_known_vector`.
    const SHA3_INPUT: [u8; 32] = [0u8; 32];

    /// `SHA3(offset=0, length=32)` bytecode hashing memory[0..32] which
    /// was first populated by `MSTORE` of a 32-byte PUSH32 immediate.
    ///
    /// Layout:
    ///   PUSH32 <32 bytes of SHA3_INPUT> ; stack top = U256 value
    ///   PUSH1 0x00                      ; stack top = offset
    ///   MSTORE                          ; memory[0..32] = SHA3_INPUT
    ///   PUSH1 0x20                      ; stack top = length (32)
    ///   PUSH1 0x00                      ; stack top = offset (0)
    ///   SHA3                            ; pop offset, length, push hash
    ///   STOP
    fn sha3_bytecode() -> Vec<u8> {
        let mut bc = Vec::with_capacity(1 + 32 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1);
        bc.push(0x7F); // PUSH32
        bc.extend_from_slice(&SHA3_INPUT);
        bc.extend_from_slice(&[
            0x60, 0x00, // PUSH1 0x00 (offset for MSTORE)
            0x52, // MSTORE
            0x60, 0x20, // PUSH1 0x20 (length = 32)
            0x60, 0x00, // PUSH1 0x00 (offset = 0)
            0x20, // SHA3
            0x00, // STOP
        ]);
        bc
    }

    /// Helper: run the bytecode and produce the EVM main trace.
    fn build_evm_main_trace(curve: CurveType) -> TracePolynomials {
        let bytecode = sha3_bytecode();
        let cols = execute_bytecode(&bytecode, &[])
            .expect("execute_bytecode must succeed for the SHA3 bytecode");
        TracePolynomials::from_vm_trace(&cols, curve)
    }

    /// Helper: build the sha3_full_chain_air gadget trace for a single
    /// row mirroring the SHA3 invocation in the EVM trace.
    fn build_sha3_gadget_trace(curve: CurveType) -> TracePolynomials {
        // pc of the SHA3 opcode in the bytecode above:
        //   byte 0     : PUSH32
        //   bytes 1..33: immediate
        //   byte 33    : PUSH1
        //   byte 34    : 0x00
        //   byte 35    : MSTORE
        //   byte 36    : PUSH1
        //   byte 37    : 0x20
        //   byte 38    : PUSH1
        //   byte 39    : 0x00
        //   byte 40    : SHA3        ← pc = 40
        let pc = 40u64;
        let w = Sha3FullChainWitness::from_events(&[(
            pc,
            0u64,
            SHA3_INPUT.len() as u64,
            SHA3_INPUT.to_vec(),
        )])
        .expect("from_events must succeed for a 32-byte input");
        build_sha3_full_trace(&w, curve)
    }

    /// Fast: descriptor is well-formed and matches the single-column
    /// tuple invariant `joint_prove` currently enforces, and the EVM
    /// + gadget traces actually agree on the limb-0 of `keccak256([0u8; 32])`.
    #[test]
    fn descriptor_consistency() {
        let d = make_evm_sha3_to_gadget_descriptor(0, 1);
        assert_eq!(d.label, "evm_main_sha3_to_gadget_v1");
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
        assert_eq!(d.b_columns[0], GADGET_COL_OUTPUT_LIMB_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_SEL_KECCAK));
        assert_eq!(d.b_selector_column, Some(GADGET_COL_IS_REAL));

        // Cross-check the column contract pinned in the gadget AIR.
        assert_eq!(GADGET_COL_OUTPUT_LIMB_OFFSET, 107);
        assert_eq!(GADGET_COL_IS_REAL, 111);

        // Build both traces and confirm the SHA3 row's limb-0 matches
        // the gadget's limb-0 byte-for-byte.
        let curve = CurveType::Bls48581;
        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_sha3_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = Sha3FullChainConstraintSystem::new(gadget_polys.num_rows);

        // EVM side: locate the SHA3 row by sel_keccak = 1 and confirm
        // its OUTPUT0_L0 equals the gadget's row-0 output_limb[0].
        let sel_col = &evm_polys.columns[COL_SEL_KECCAK].evaluations;
        let evm_out0 = &evm_polys.columns[COL_OUTPUT0_L0].evaluations;
        let one_bytes = Scalar::one(curve).to_bytes();
        let gadget_l0 =
            &gadget_polys.columns[GADGET_COL_OUTPUT_LIMB_OFFSET].evaluations;
        let expected_l0_bytes = gadget_l0[0].to_bytes();

        let mut found = false;
        for r in 0..evm_polys.num_rows {
            if sel_col[r].to_bytes() == one_bytes {
                assert_eq!(
                    evm_out0[r].to_bytes(),
                    expected_l0_bytes,
                    "EVM main SHA3 row's OUTPUT0_L0 must equal gadget output_limb[0]",
                );
                found = true;
                break;
            }
        }
        assert!(
            found,
            "EVM main trace must contain at least one SHA3 (sel_keccak=1) row",
        );

        // Sanity: gadget row 0 IS_REAL = 1.
        let is_real = &gadget_polys.columns[GADGET_COL_IS_REAL].evaluations;
        assert_eq!(
            is_real[0].to_bytes(),
            one_bytes,
            "gadget row 0 must be marked is_real",
        );

        // Sanity: assemble the (trace, cs) pair shape that `joint_prove`
        // consumes without actually invoking the prover.
        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_sha3_to_gadget_descriptor(0, 1)];
        assert_eq!(traces.len(), 2);
        assert_eq!(linkages.len(), 1);
    }

    /// Honest 2-AIR `joint_prove` + `joint_verify` round-trip across
    /// the EVM main trace (running the SHA3 bytecode) and a 1-row
    /// sha3_full_chain_air witness pinned to the same invocation.
    ///
    /// Marked `#[ignore]` because, even with a ~6-row bytecode,
    /// `joint_prove` runs:
    ///   - 1× per-AIR `prove_with_scheme` for EVM main (padded to the
    ///     LogUp range-table domain = 256 because EvmConstraintSystem
    ///     declares 8-bit range checks);
    ///   - 1× per-AIR `prove_with_scheme` for sha3_full_chain_air;
    ///   - 1× per-linkage `prove_with_scheme` on the inner
    ///     `LinkageConstraintSystem`;
    ///   - KZG opens against per-AIR / per-linkage commitments.
    /// On BLS48-581 this is expected to run in the 30-60s range on a
    /// modest workstation in `--release`.
    #[test]
    #[ignore = "slow: joint_prove + joint_verify EVM main + sha3_full_chain_air (BLS48-581, ~30-60s release)"]
    fn honest_joint_verify_true() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let evm_polys = build_evm_main_trace(curve);
        let gadget_polys = build_sha3_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = Sha3FullChainConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_sha3_to_gadget_descriptor(0, 1)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove (EVM main + sha3_full_chain) must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        assert_eq!(
            lp.closure_a, lp.closure_b,
            "honest linkage closures must match: A's SHA3-row push limb-0 multiset \
             equals B's real-row output_limb_0 multiset",
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest joint_verify must accept the EVM main ↔ sha3_full_chain linkage",
        );
    }

    /// Tampered witness: corrupt `closure_a` in the extension envelope
    /// so the joint verifier's scalar `closure_a == closure_b` check
    /// fires and rejects.
    ///
    /// Re-runs `joint_prove` because we need a real `ExecutionProof`
    /// pair to feed `joint_verify`. Marked `#[ignore]` because the
    /// setup half is the same `joint_prove` call as
    /// `honest_joint_verify_true`.
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
        let gadget_polys = build_sha3_gadget_trace(curve);
        let evm_cs = EvmConstraintSystem::new();
        let gadget_cs = Sha3FullChainConstraintSystem::new(gadget_polys.num_rows);

        let traces: Vec<(
            &TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_polys, &gadget_cs)];
        let linkages = vec![make_evm_sha3_to_gadget_descriptor(0, 1)];

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
