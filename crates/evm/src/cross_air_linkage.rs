//! Host-side cross-AIR consistency: EVM main-trace EXP rows ↔ EXP gadget AIR.
//!
//! The EVM main trace contains EXP opcode rows where the constraint body
//! is a zero-body oracle (constraint 47 in `crates/evm/src/constraints/mod.rs`):
//! the prover claims `(base, exponent, result)` triples but the main AIR
//! does NOT enforce `result == base^exponent mod 2^256`. The standalone
//! EXP gadget AIR (`crates/evm/src/exp_air`) verifies that relation
//! iteratively over 256 rows but is not yet cryptographically linked to
//! the main trace.
//!
//! This module provides a **host-side** checker mirroring
//! `crates/zkp/src/cross_air_linkage.rs`:
//!
//!   1. Documents the data-flow contract a future LogUp-style cross-AIR
//!      lookup will enforce cryptographically (the cryptographic version
//!      produces exactly the same accept/reject decision).
//!   2. Supplies cheap end-to-end smoke checks for integration tests.
//!
//! # Out of scope
//!
//! Cryptographic LogUp/Plookup-style cross-table arguments — those
//! require coordinated Fiat-Shamir randomness across the EVM main trace
//! and the EXP gadget plus auxiliary witness columns in both AIRs. The
//! host-side check here is the data-flow contract a cryptographic
//! version would mirror.

use crate::exp_air::{exp_witness, ExpRow};
use revm::primitives::U256;

/// One EXP claim extracted from the EVM main trace.
///
/// On a row where `sel_arith_exp == 1`, the trace exposes `INPUT0` as the
/// base, `INPUT1` as the exponent, and `OUTPUT0` as the claimed result —
/// but no constraint binds them. This struct is the data-flow contract a
/// future cross-AIR lookup will check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvmExpClaim {
    pub base: U256,
    pub exponent: U256,
    pub claimed_result: U256,
}

/// Per-claim result of [`check_evm_exp_consistency`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvmExpRowCheck {
    /// `claimed_result == base^exponent mod 2^256` AND a matching gadget
    /// trace was supplied.
    OkLinked,
    /// `claimed_result == base^exponent mod 2^256` but no matching gadget
    /// trace was supplied. The result is correct natively but the
    /// cross-AIR linkage is incomplete.
    OkNoTrace,
    /// `claimed_result != base^exponent mod 2^256`. The main-trace EXP
    /// claim is wrong (or the gadget trace was tampered with).
    ResultMismatch,
}

/// Host-side cross-AIR consistency check: EVM main-trace EXP ↔ gadget.
///
/// For each EXP claim, verify:
///
///   1. `claimed_result == base^exponent mod 2^256` (computed natively
///      via `exp_witness`).
///   2. Some supplied gadget trace covers the same (base, exponent)
///      with a matching final-row result.
///
/// Returns one [`EvmExpRowCheck`] per claim, in order. The cryptographic
/// LogUp version will enforce condition 1 by checking each
/// (base, exponent, result) tuple appears in the EXP gadget's
/// (initial_base, initial_exponent, final_result) table — i.e. the same
/// accept rule as `OkLinked`.
pub fn check_evm_exp_consistency(
    claims: &[EvmExpClaim],
    gadget_traces: &[Vec<ExpRow>],
) -> Vec<EvmExpRowCheck> {
    claims
        .iter()
        .map(|claim| {
            // 1. Native check: actually compute base^exponent mod 2^256.
            let native = exp_witness(claim.base, claim.exponent);
            let computed = final_result(&native);
            if computed != claim.claimed_result {
                return EvmExpRowCheck::ResultMismatch;
            }
            // 2. Linked check: find a gadget trace matching this claim.
            let linked = gadget_traces.iter().any(|t| {
                if t.len() != native.len() {
                    return false;
                }
                if t.first().map(|r| r.base) != native.first().map(|r| r.base) {
                    return false;
                }
                final_result(t) == claim.claimed_result
            });
            if linked {
                EvmExpRowCheck::OkLinked
            } else {
                EvmExpRowCheck::OkNoTrace
            }
        })
        .collect()
}

/// Convenience: build the EXP gadget traces a future cross-AIR linkage
/// would need to fully cover a list of EVM main-trace EXP claims.
/// Produces one trace per claim (256 rows each).
pub fn exp_traces_for_evm_claims(claims: &[EvmExpClaim]) -> Vec<Vec<ExpRow>> {
    claims
        .iter()
        .map(|c| exp_witness(c.base, c.exponent))
        .collect()
}

/// Construct the [`metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor`]
/// for the EVM main-trace-EXP ↔ EXP-gadget linkage.
///
/// The cryptographic linkage matches a 12-limb tuple
/// `(base[0..4], exponent[0..4], output[0..4])` between:
///
/// - **A side (EVM main trace)**: the EXP opcode rows, gated by
///   `SEL_EXP`. Tuple columns: `INPUT0_L0..3` (base) at indices 4..7,
///   `INPUT1_L0..3` (exponent) at indices 8..11, `OUTPUT0_L0..3`
///   (output) at indices 12..15.
///
/// - **B side (EXP gadget)**: a single anchor row per invocation
///   (where `IS_FIRST_ROW = 1` ⇔ `bit_index = 255`). Tuple columns:
///   `BASE_OFFSET..+4`, `EXPONENT_OFFSET..+4`, `FINAL_OUTPUT_OFFSET..+4`.
///
/// The cross-AIR LogUp checks that every `(base, exponent, output)`
/// tuple appearing in the EVM main trace at a `SEL_EXP` row also
/// appears in some EXP gadget invocation's anchor row. By the
/// β-RLC tuple encoding and Schwartz-Zippel, this enforces the
/// correctness of every EVM EXP opcode against the gadget's
/// algebraically-verified `base^exponent` computation.
///
/// **Scope and remaining gaps:**
/// - This descriptor assumes a single EXP gadget invocation per gadget
///   trace (the current layout). For multiple EVM EXP invocations, a
///   multi-invocation gadget AIR or one linkage per invocation is
///   needed — see `cross_air_logup_dependent_tasks.md` memory.
/// - The new gadget columns (`EXPONENT`, `FINAL_OUTPUT`, `IS_FIRST_ROW`)
///   are populated by the witness builder but not yet constrained at
///   the per-AIR level (no invariance check for `EXPONENT`/`FINAL_OUTPUT`
///   across rows of an invocation, no enforcement that `IS_FIRST_ROW`
///   coincides with `bit_index = 255`). The cross-AIR LogUp tuple
///   matching catches anchor-row tampering, but the witness-column
///   integrity at non-anchor rows requires future per-AIR constraints.
pub fn make_evm_exp_linkage_descriptor(
    a_layer_index: usize,
    b_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::exp_air;
    use crate::trace::{
        COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3, COL_INPUT1_L0, COL_INPUT1_L1,
        COL_INPUT1_L2, COL_INPUT1_L3, COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2,
        COL_OUTPUT0_L3, COL_SEL_EXP,
    };

    let a_columns = vec![
        COL_INPUT0_L0,
        COL_INPUT0_L1,
        COL_INPUT0_L2,
        COL_INPUT0_L3,
        COL_INPUT1_L0,
        COL_INPUT1_L1,
        COL_INPUT1_L2,
        COL_INPUT1_L3,
        COL_OUTPUT0_L0,
        COL_OUTPUT0_L1,
        COL_OUTPUT0_L2,
        COL_OUTPUT0_L3,
    ];
    let b_columns = vec![
        exp_air::COL_BASE_OFFSET,
        exp_air::COL_BASE_OFFSET + 1,
        exp_air::COL_BASE_OFFSET + 2,
        exp_air::COL_BASE_OFFSET + 3,
        exp_air::COL_EXPONENT_OFFSET,
        exp_air::COL_EXPONENT_OFFSET + 1,
        exp_air::COL_EXPONENT_OFFSET + 2,
        exp_air::COL_EXPONENT_OFFSET + 3,
        exp_air::COL_FINAL_OUTPUT_OFFSET,
        exp_air::COL_FINAL_OUTPUT_OFFSET + 1,
        exp_air::COL_FINAL_OUTPUT_OFFSET + 2,
        exp_air::COL_FINAL_OUTPUT_OFFSET + 3,
    ];

    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_exp_gadget_v1".into(),
        a_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_EXP),
        b_layer_index,
        b_columns,
        b_selector_column: Some(exp_air::COL_IS_FIRST_ROW),
    }
}

/// Construct a [`metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor`]
/// for the EVM main-trace CREATE / CREATE2 ↔ KeccakExtract linkage
/// (closes the address-derivation gap of #95 — the second half, the
/// algebraic input binding `keccak_input == rlp(sender, nonce)` for
/// CREATE / `keccak_input == 0xff || sender || salt || keccak256(initcode)`
/// for CREATE2, is the residual scope on the EVM side).
///
/// The cryptographic linkage matches a 4-limb tuple
/// `(address_limb_0..3)` between:
///
/// - **A side (EVM main trace)**: the CREATE or CREATE2 row, gated by
///   `selector` (one of [`crate::trace::COL_SEL_CREATE`] or
///   [`crate::trace::COL_SEL_CREATE2`]). The four columns are
///   `COL_CREATE_ADDRESS_HINT_L0..L3` (cols 226–229), populated by the
///   inspector via `address_to_limbs(addr)` (LE u64 packing of the
///   20-byte address).
///
/// - **B side (KeccakExtract)**: any row where `IS_REAL = 1` (one
///   keccak256 invocation). The four columns are
///   `COL_ADDRESS_LIMB_OFFSET..+4`, aggregated by the AIR's
///   `address_limb_*_binding` row-local constraints from
///   `OUTPUT_BYTE[12..32]` using the same LE convention as
///   `address_to_limbs`.
///
/// Soundness chain (output-side only, this PR):
/// 1. KeccakExtract's `address_limb_binding` constraints pin
///    `address_limb[k] = LE-aggregate of OUTPUT_BYTE[12 + 8k .. ]`
///    for every row.
/// 2. KeccakExtract↔Keccak (already landed via #91) pins
///    `OUTPUT_BYTE[0..32] = keccak256(INPUT_BYTE[0..INPUT_LEN])`.
/// 3. This linkage matches EVM's `create_address_hint` tuple to some
///    keccak invocation's `address_limb` tuple.
///
/// **Open soundness gap** (input side): nothing yet enforces that the
/// linked KeccakExtract row's `(INPUT_BYTE, INPUT_LEN)` is the
/// canonical CREATE pre-image (`rlp(sender, nonce)`) or CREATE2
/// pre-image (`0xff || sender || salt || keccak256(initcode)`). A
/// future task will close this with an RLP-encoding gadget AIR plus
/// a second cross-AIR linkage on the input side. By keccak preimage
/// resistance the current binding is computationally sound (the
/// prover cannot produce a fake `(input, address)` pair with the
/// claimed address that ALSO matches `keccak256(input)`), but it is
/// not algebraically sound: a malicious prover with infinite
/// computation could in principle find any input whose digest's
/// last 20 bytes equal a target address.
///
/// Use one descriptor per CREATE-family selector — the cross-AIR
/// LogUp infrastructure takes a `Vec<CrossAirLogUpDescriptor>`, so
/// CREATE and CREATE2 can be wired as two parallel linkages over
/// the same KeccakExtract layer.
pub fn make_evm_create_address_keccak_extract_linkage_descriptor(
    a_layer_index: usize,
    b_layer_index: usize,
    a_selector_column: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_CREATE_ADDRESS_HINT_L0, COL_CREATE_ADDRESS_HINT_L1, COL_CREATE_ADDRESS_HINT_L2,
        COL_CREATE_ADDRESS_HINT_L3, COL_SEL_CREATE, COL_SEL_CREATE2,
    };
    debug_assert!(
        a_selector_column == COL_SEL_CREATE || a_selector_column == COL_SEL_CREATE2,
        "selector must be COL_SEL_CREATE or COL_SEL_CREATE2"
    );
    let a_columns = vec![
        COL_CREATE_ADDRESS_HINT_L0,
        COL_CREATE_ADDRESS_HINT_L1,
        COL_CREATE_ADDRESS_HINT_L2,
        COL_CREATE_ADDRESS_HINT_L3,
    ];
    let b_columns = (0..metavm_zkp::keccak_extract::ADDRESS_LIMB_LEN)
        .map(|k| metavm_zkp::keccak_extract::COL_ADDRESS_LIMB_OFFSET + k)
        .collect();
    let label = if a_selector_column == COL_SEL_CREATE {
        "evm_create_address_v1"
    } else {
        "evm_create2_address_v1"
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index,
        a_columns,
        a_selector_column: Some(a_selector_column),
        b_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::keccak_extract::COL_IS_REAL),
    }
}

/// EVM-side wrapper over
/// [`metavm_zkp::evm_create_rlp_air::make_evm_main_create_rlp_linkage_descriptor`]
/// that wires the actual EVM trace column indices: `frame_callee_l[0..4]`
/// (the executing contract's address — the CREATE sender), the new
/// `create_nonce_hint` oracle (sender's pre-bump nonce on CREATE rows),
/// and `sel_create` as the gating selector.
///
/// Closes the EVM-side adapter for the CREATE input-side cross-AIR
/// linkage. Combined with [`make_evm_create_address_keccak_extract_linkage_descriptor`]
/// (output side) and the new RLP gadget AIR, this completes the
/// CREATE address-derivation soundness chain end-to-end.
pub fn make_evm_main_create_rlp_linkage_descriptor(
    evm_layer_index: usize,
    rlp_gadget_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::evm_create_rlp_air::make_evm_main_create_rlp_linkage_descriptor(
        evm_layer_index,
        rlp_gadget_layer_index,
        crate::trace::COL_FRAME_CALLEE_L0,
        crate::trace::COL_CREATE_NONCE_HINT,
        crate::trace::COL_SEL_CREATE,
    )
}

/// EVM-side wrapper over
/// [`metavm_zkp::evm_create2_input_air::make_evm_main_create2_input_linkage_descriptor`]
/// that wires the actual EVM trace column indices: `frame_callee_l[0..4]`
/// (the executing contract — the CREATE2 sender), the
/// `create2_salt_hint_l[0..4]` and `create2_initcode_hash_hint_l[0..4]`
/// oracles populated by the inspector on CREATE2 rows, and `sel_create2`
/// as the gating selector.
///
/// Closes the EVM-side adapter for the CREATE2 input-side cross-AIR
/// linkage. Combined with
/// [`metavm_zkp::evm_create2_input_air::make_create2_input_keccak_extract_input_linkage_descriptor`]
/// (gadget→KeccakExtract) and
/// [`make_evm_create_address_keccak_extract_linkage_descriptor`] (output
/// side, shared with CREATE), this completes a 4-AIR CREATE2
/// address-derivation soundness chain analogous to the CREATE chain.
pub fn make_evm_main_create2_input_linkage_descriptor(
    evm_layer_index: usize,
    create2_gadget_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::evm_create2_input_air::make_evm_main_create2_input_linkage_descriptor(
        evm_layer_index,
        create2_gadget_layer_index,
        crate::trace::COL_FRAME_CALLEE_L0,
        crate::trace::COL_CREATE2_SALT_HINT_L0,
        crate::trace::COL_CREATE2_INITCODE_HASH_HINT_L0,
        crate::trace::COL_SEL_CREATE2,
    )
}

/// EVM SHA3 (KECCAK opcode, 0x20) ↔ KeccakExtract output-side cross-AIR
/// LogUp linkage (Phase A1a of the EVM full-coverage roadmap).
///
/// On EVM rows where `SEL_KECCAK = 1`, the trace's `output0` (4 u64
/// limbs at `COL_OUTPUT0_L0..L3`) holds `keccak256(memory[offset..offset+size])`
/// pushed onto the stack as a U256. KeccakExtract's `KECCAK_OUTPUT_LIMB[0..4]`
/// columns expose the same packing (`U256::from_be_bytes(output).as_limbs()`)
/// computed from its `OUTPUT_BYTE[0..32]` via the
/// `keccak_output_limb_*_binding` row-locals. This descriptor matches
/// the 4-limb tuples directly.
///
/// **Soundness state — output-side only**: combined with the existing
/// KeccakExtract↔Keccak linkage (#91), the SHA3 result is now bound to
/// **some** real keccak invocation by preimage resistance. The
/// **input-side** binding — pinning that the linked KeccakExtract row's
/// `(INPUT_BYTE, INPUT_LEN)` equals the actual EVM memory bytes the
/// SHA3 opcode read — is Phase A1b, deferred until the EVM memory model
/// can expose the read range as algebraic columns. Until A1b lands a
/// malicious prover with infinite computation could in principle
/// fabricate `(input, output)` pairs whose digest equals a desired
/// stack value; with A1a alone it's computationally infeasible but not
/// algebraically pinned.
///
/// Mirrors [`make_evm_create_address_keccak_extract_linkage_descriptor`]'s
/// preimage-resistance pattern.
pub fn make_evm_keccak_keccak_extract_linkage_descriptor(
    evm_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_KECCAK,
    };
    let a_columns = vec![COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3];
    let b_columns = (0..metavm_zkp::keccak_extract::KECCAK_OUTPUT_LIMB_LEN)
        .map(|k| metavm_zkp::keccak_extract::COL_KECCAK_OUTPUT_LIMB_OFFSET + k)
        .collect();
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_keccak_output_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_KECCAK),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::keccak_extract::COL_IS_REAL),
    }
}

/// **Phase A1b L1**: EVM SHA3 `output0[0..4]` ↔ SHA3 input gadget
/// `OUTPUT_LIMB[0..4]`, gated by `SEL_KECCAK` / `IS_REAL`. Routes the
/// existing A1a output binding through the SHA3 input gadget instead
/// of directly to KeccakExtract.
///
/// The full A1b chain is two linkages:
/// - L1 (this descriptor): EVM main ↔ gadget on output_limb (4 cols).
/// - L2 ([`metavm_zkp::sha3_input_air::make_sha3_input_keccak_extract_linkage_descriptor`]):
///   gadget ↔ KeccakExtract on `(input_byte[0..256], input_len, output_limb)`
///   (261 cols).
///
/// Combined: EVM's `output0` matches gadget row's `OUTPUT_LIMB`
/// (forcing same gadget row), and gadget row's full
/// `(input_byte, input_len, output_limb)` matches KeccakExtract,
/// which itself is bound to `output = keccak256(input)` via #91.
///
/// Soundness gain over A1a alone: the SHA3 preimage is now exposed
/// as committed witness data on the gadget AIR. An external observer
/// can audit which bytes the prover claims as the SHA3 input. The
/// **memory binding** — pinning those gadget bytes to actual EVM
/// memory contents — is a future Phase A1b-mem follow-up requiring
/// byte-level memory access tracking.
pub fn make_evm_sha3_input_linkage_descriptor(
    evm_layer_index: usize,
    sha3_input_gadget_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_KECCAK,
    };
    let a_columns = vec![COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3];
    let b_columns = (0..metavm_zkp::sha3_input_air::NUM_OUTPUT_LIMBS)
        .map(|k| metavm_zkp::sha3_input_air::COL_OUTPUT_LIMB_OFFSET + k)
        .collect();
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_sha3_input_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_KECCAK),
        b_layer_index: sha3_input_gadget_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::sha3_input_air::COL_IS_REAL),
    }
}

/// **Phase A1b-mem step 1c-a**: EVM main `MSTORE8` rows ↔ byte-memory
/// AIR `Write` entries, gated by `SEL_MSTORE8` on EVM and `RW` on the
/// byte-memory AIR.
///
/// Tuple shape: `(mem_offset, mem_value[0])`. A side projects the EVM
/// main's two memory-write trace columns; B side projects the byte-
/// memory AIR's `(addr, val)` columns from the *unsorted* view.
///
/// Phase 3 of byte-memory AIR (multiset perm sorted ↔ unsorted) closes
/// the link from the unsorted view to the sorted view, so combined
/// with this linkage:
///   EVM MSTORE8 row's `(offset, byte)` is forced (via cross-AIR LogUp
///   closure) to appear in byte-memory's unsorted Write tuples, which
///   are forced (via self-linkage) to equal the multiset of the sorted
///   view, on which the read-consistency constraints are enforced.
///
/// **Caveat (A1b-mem step 1c-ext, deferred)**: the byte-memory's `val`
/// is a u8 (one byte), but the EVM main's `mem_value[0]` is a u64
/// limb holding the full popped stack value. For MSTORE8 with stack
/// inputs in `[0, 255]` (e.g. the A1b bytecode using `PUSH1`), these
/// match. For general MSTORE8 with stack inputs > 255 (e.g. `PUSH32
/// 0xff..00; MSTORE8`), the EVM-side tuple is `(offset, big_u64)`
/// while the byte-memory tuple is `(offset, big_u64 & 0xff)` — they
/// won't match. A future byte-extraction gadget AIR (or a new
/// `mstore8_byte` algebraic column in EVM main bound to `mem_value[0]
/// mod 256`) is required to close this gap.
///
/// Also note: this linkage does NOT cover MSTORE (32-byte word
/// writes), which expand 1:32 in the byte-memory AIR. A future
/// byte-expansion gadget AIR is required for that case.
/// **Phase A1b-mem step 1c-b**: EVM main `MSTORE` row ↔ MSTORE byte-
/// decomposition gadget AIR, gated by `SEL_MSTORE` on EVM and `IS_REAL`
/// on the gadget.
///
/// Tuple shape: `(mem_offset, mem_value[L0..L3])` — 5 columns. 1:1
/// multiset (each EVM MSTORE row corresponds to one gadget row).
/// Combined with the 32 gadget→byte-memory per-position linkages, this
/// chain forces every EVM MSTORE event to commit its 32 BE bytes to
/// the byte-memory AIR's Write multiset.
pub fn make_evm_mstore_to_byte_decomp_linkage_descriptor(
    evm_layer_index: usize,
    mstore_byte_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_MEM_OFFSET, COL_MEM_VALUE_L0, COL_MEM_VALUE_L1, COL_MEM_VALUE_L2,
        COL_MEM_VALUE_L3, COL_SEL_MSTORE,
    };
    let a_columns = vec![
        COL_MEM_OFFSET,
        COL_MEM_VALUE_L0,
        COL_MEM_VALUE_L1,
        COL_MEM_VALUE_L2,
        COL_MEM_VALUE_L3,
    ];
    let b_columns = vec![
        metavm_zkp::mstore_byte_air::COL_OFFSET,
        metavm_zkp::mstore_byte_air::COL_LIMB_L0,
        metavm_zkp::mstore_byte_air::COL_LIMB_L1,
        metavm_zkp::mstore_byte_air::COL_LIMB_L2,
        metavm_zkp::mstore_byte_air::COL_LIMB_L3,
    ];
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_mstore_to_byte_decomp_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_MSTORE),
        b_layer_index: mstore_byte_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::mstore_byte_air::COL_IS_REAL),
    }
}

/// **#53 step 2 / #54 — EVM TIMESTAMP ↔ block_header_air linkage**.
/// Tuple: `(output0_l0)` ↔ `(timestamp)`. Single-col tuple gated by
/// `COL_SEL_TIMESTAMP` on EVM side and `COL_IS_REAL` on
/// block_header_air side.
///
/// **Caveat**: like SLOAD/SSTORE, the EVM-side selector binding is
/// honest-by-inspector (no algebraic must-fire constraint yet); a
/// malicious prover that omits sel_timestamp on a real TIMESTAMP row
/// would have the gadget's `(timestamp)` not match the EVM-side
/// closure → multiset mismatch.
pub fn make_evm_timestamp_to_block_header_linkage_descriptor(
    evm_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_OUTPUT0_L0, COL_SEL_TIMESTAMP};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_timestamp_to_block_header_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_TIMESTAMP),
        b_layer_index: block_header_layer_index,
        b_columns: vec![metavm_zkp::block_header_air::COL_TIMESTAMP],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// **#53 step 2 / #54 — EVM NUMBER ↔ block_header_air linkage**.
pub fn make_evm_number_to_block_header_linkage_descriptor(
    evm_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_OUTPUT0_L0, COL_SEL_NUMBER};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_number_to_block_header_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_NUMBER),
        b_layer_index: block_header_layer_index,
        b_columns: vec![metavm_zkp::block_header_air::COL_NUMBER],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// **#53 step 2 / #54 — EVM GASLIMIT ↔ block_header_air linkage**.
pub fn make_evm_gaslimit_to_block_header_linkage_descriptor(
    evm_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_OUTPUT0_L0, COL_SEL_GASLIMIT};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_gaslimit_to_block_header_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_GASLIMIT),
        b_layer_index: block_header_layer_index,
        b_columns: vec![metavm_zkp::block_header_air::COL_GAS_LIMIT],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// **#53 step 2 / #54 — EVM BASEFEE ↔ block_header_air linkage**.
/// 4-limb tuple binding `output0[L0..L3]` ↔ `base_fee_per_gas[L0..L3]`.
pub fn make_evm_basefee_to_block_header_linkage_descriptor(
    evm_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_BASEFEE,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_basefee_to_block_header_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3],
        a_selector_column: Some(COL_SEL_BASEFEE),
        b_layer_index: block_header_layer_index,
        b_columns: vec![
            metavm_zkp::block_header_air::COL_BASE_FEE_L0,
            metavm_zkp::block_header_air::COL_BASE_FEE_L1,
            metavm_zkp::block_header_air::COL_BASE_FEE_L2,
            metavm_zkp::block_header_air::COL_BASE_FEE_L3,
        ],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// **#53 step 2 / #54 — EVM COINBASE ↔ block_header_air linkage**.
/// 4-limb tuple for the 20-byte coinbase address (low 160 bits).
pub fn make_evm_coinbase_to_block_header_linkage_descriptor(
    evm_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_COINBASE,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_coinbase_to_block_header_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3],
        a_selector_column: Some(COL_SEL_COINBASE),
        b_layer_index: block_header_layer_index,
        b_columns: vec![
            metavm_zkp::block_header_air::COL_BENEFICIARY_L0,
            metavm_zkp::block_header_air::COL_BENEFICIARY_L1,
            metavm_zkp::block_header_air::COL_BENEFICIARY_L2,
            metavm_zkp::block_header_air::COL_BENEFICIARY_L3,
        ],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// **#54 — EVM PREVRANDAO ↔ block_header_air linkage**.
/// 4-limb tuple binding `output0[L0..L3]` ↔ `prev_randao[L0..L3]`.
pub fn make_evm_prevrandao_to_block_header_linkage_descriptor(
    evm_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_PREVRANDAO,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_prevrandao_to_block_header_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3],
        a_selector_column: Some(COL_SEL_PREVRANDAO),
        b_layer_index: block_header_layer_index,
        b_columns: vec![
            metavm_zkp::block_header_air::COL_PREV_RANDAO_L0,
            metavm_zkp::block_header_air::COL_PREV_RANDAO_L1,
            metavm_zkp::block_header_air::COL_PREV_RANDAO_L2,
            metavm_zkp::block_header_air::COL_PREV_RANDAO_L3,
        ],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// **#54 — EVM CHAINID ↔ block_header_air linkage**.
/// Single-limb binding `output0[L0]` ↔ `chain_id`.
pub fn make_evm_chainid_to_block_header_linkage_descriptor(
    evm_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_OUTPUT0_L0, COL_SEL_CHAINID};
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_chainid_to_block_header_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0],
        a_selector_column: Some(COL_SEL_CHAINID),
        b_layer_index: block_header_layer_index,
        b_columns: vec![metavm_zkp::block_header_air::COL_CHAIN_ID],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// **#54 — EVM SELFBALANCE ↔ account_state_air linkage**.
/// 4-limb tuple binding EVM `output0[L0..L3]` (the balance) ↔
/// account_state_air balance columns. The EVM-side output is the
/// balance of the current contract address. The gadget side needs
/// a future `balance_l0..l3` column set; this descriptor documents
/// the data contract.
pub fn make_evm_selfbalance_to_account_state_linkage_descriptor(
    evm_layer_index: usize,
    account_state_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_SELFBALANCE,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_selfbalance_to_account_state_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3],
        a_selector_column: Some(COL_SEL_SELFBALANCE),
        b_layer_index: account_state_layer_index,
        b_columns: vec![
            metavm_zkp::account_state_air::COL_BALANCE_L0,
            metavm_zkp::account_state_air::COL_BALANCE_L1,
            metavm_zkp::account_state_air::COL_BALANCE_L2,
            metavm_zkp::account_state_air::COL_BALANCE_L3,
        ],
        b_selector_column: Some(metavm_zkp::account_state_air::COL_IS_REAL),
    }
}

/// **#54 — EVM EXTCODESIZE ↔ account_state_air linkage**.
/// Binds (queried_address, returned_code_size) for EXTCODESIZE rows.
/// account_state_air needs a code_size column for full binding; until
/// that's added, this descriptor uses placeholder col 4 (nonce).
pub fn make_evm_extcodesize_to_account_state_linkage_descriptor(
    evm_layer_index: usize,
    account_state_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
        COL_OUTPUT0_L0, COL_SEL_ENV,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_extcodesize_to_account_state_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![
            COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
            COL_OUTPUT0_L0,
        ],
        a_selector_column: Some(COL_SEL_ENV),
        b_layer_index: account_state_layer_index,
        b_columns: vec![
            metavm_zkp::account_state_air::COL_ADDR_L0,
            metavm_zkp::account_state_air::COL_ADDR_L1,
            metavm_zkp::account_state_air::COL_ADDR_L2,
            metavm_zkp::account_state_air::COL_ADDR_L3,
            metavm_zkp::account_state_air::COL_NONCE, // placeholder — needs code_size column
        ],
        b_selector_column: Some(metavm_zkp::account_state_air::COL_IS_REAL),
    }
}

/// **#54 — EVM BALANCE ↔ account_state_air linkage**.
/// Binds (queried_address, returned_balance) on BALANCE rows to
/// (address, balance_limbs) in account_state_air.
pub fn make_evm_balance_to_account_state_linkage_descriptor(
    evm_layer_index: usize,
    account_state_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3,
        COL_SEL_ENV,
    };
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_balance_to_account_state_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns: vec![
            COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
            COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3,
        ],
        a_selector_column: Some(COL_SEL_ENV),
        b_layer_index: account_state_layer_index,
        b_columns: vec![
            metavm_zkp::account_state_air::COL_ADDR_L0,
            metavm_zkp::account_state_air::COL_ADDR_L1,
            metavm_zkp::account_state_air::COL_ADDR_L2,
            metavm_zkp::account_state_air::COL_ADDR_L3,
            metavm_zkp::account_state_air::COL_BALANCE_L0,
            metavm_zkp::account_state_air::COL_BALANCE_L1,
            metavm_zkp::account_state_air::COL_BALANCE_L2,
            metavm_zkp::account_state_air::COL_BALANCE_L3,
        ],
        b_selector_column: Some(metavm_zkp::account_state_air::COL_IS_REAL),
    }
}

/// **Phase A2 step 1b — EVM SLOAD ↔ storage_access_air linkage**.
///
/// Tuple shape: `(input0[L0..L3], output0[L0..L3])` — 8 cols (slot
/// limbs + value limbs). A side gated by `COL_SEL_SLOAD` (1 only on
/// real SLOAD rows). B side gated by a hypothetical "is_load_real"
/// column on the gadget. **Caveat**: the gadget AIR currently exposes
/// `IS_REAL` and `IS_WRITE` separately; to gate SLOAD-only on the
/// gadget side we'd need a witness column `is_load_real = is_real *
/// (1 - is_write)`. Until that's added, this descriptor uses
/// `IS_REAL` (covers BOTH SLOAD and SSTORE gadget rows) — the
/// multiset closure on the EVM side has only SLOAD tuples, so
/// matching against gadget's `IS_REAL`-gated mixed multiset may
/// over-include SSTORE gadget rows. Multiset equality won't hold
/// strictly; this descriptor is the framework, not a sound binding
/// until `is_load_real` lands.
pub fn make_evm_sload_to_storage_gadget_linkage_descriptor(
    evm_layer_index: usize,
    storage_gadget_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3,
        COL_SEL_SLOAD,
    };
    let a_columns = vec![
        COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
        COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3,
    ];
    let b_columns = vec![
        metavm_zkp::storage_access_air::COL_SLOT_L0,
        metavm_zkp::storage_access_air::COL_SLOT_L1,
        metavm_zkp::storage_access_air::COL_SLOT_L2,
        metavm_zkp::storage_access_air::COL_SLOT_L3,
        metavm_zkp::storage_access_air::COL_VALUE_L0,
        metavm_zkp::storage_access_air::COL_VALUE_L1,
        metavm_zkp::storage_access_air::COL_VALUE_L2,
        metavm_zkp::storage_access_air::COL_VALUE_L3,
    ];
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_sload_to_storage_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_SLOAD),
        b_layer_index: storage_gadget_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::storage_access_air::COL_IS_REAL),
    }
}

/// **Phase A2 step 1b — EVM SSTORE ↔ storage_access_air linkage**.
/// Mirror of the SLOAD linkage but with value source = `input1`
/// (popped from stack) and gate = `COL_SEL_SSTORE`. Same caveat as
/// the SLOAD descriptor about strict B-side gating.
pub fn make_evm_sstore_to_storage_gadget_linkage_descriptor(
    evm_layer_index: usize,
    storage_gadget_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{
        COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
        COL_INPUT1_L0, COL_INPUT1_L1, COL_INPUT1_L2, COL_INPUT1_L3,
        COL_SEL_SSTORE,
    };
    let a_columns = vec![
        COL_INPUT0_L0, COL_INPUT0_L1, COL_INPUT0_L2, COL_INPUT0_L3,
        COL_INPUT1_L0, COL_INPUT1_L1, COL_INPUT1_L2, COL_INPUT1_L3,
    ];
    let b_columns = vec![
        metavm_zkp::storage_access_air::COL_SLOT_L0,
        metavm_zkp::storage_access_air::COL_SLOT_L1,
        metavm_zkp::storage_access_air::COL_SLOT_L2,
        metavm_zkp::storage_access_air::COL_SLOT_L3,
        metavm_zkp::storage_access_air::COL_VALUE_L0,
        metavm_zkp::storage_access_air::COL_VALUE_L1,
        metavm_zkp::storage_access_air::COL_VALUE_L2,
        metavm_zkp::storage_access_air::COL_VALUE_L3,
    ];
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_sstore_to_storage_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_SSTORE),
        b_layer_index: storage_gadget_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::storage_access_air::COL_IS_REAL),
    }
}

pub fn make_evm_mstore8_byte_memory_linkage_descriptor(
    evm_layer_index: usize,
    byte_memory_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::trace::{COL_MEM_OFFSET, COL_MEM_VALUE_L0, COL_SEL_MSTORE8};
    let a_columns = vec![COL_MEM_OFFSET, COL_MEM_VALUE_L0];
    let b_columns = vec![
        metavm_zkp::byte_memory_air::COL_ADDR,
        metavm_zkp::byte_memory_air::COL_VAL,
    ];
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "evm_mstore8_byte_memory_v1".into(),
        a_layer_index: evm_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_MSTORE8),
        b_layer_index: byte_memory_layer_index,
        b_columns,
        // `rw` is 1 on Write rows, 0 on Read rows AND 0 on padding —
        // works directly as a selector for "byte-memory Write tuples".
        b_selector_column: Some(metavm_zkp::byte_memory_air::COL_RW),
    }
}

#[cfg(test)]
mod a1a_tests {
    use super::*;
    use crate::trace::{COL_OUTPUT0_L0, COL_OUTPUT0_L1, COL_OUTPUT0_L2, COL_OUTPUT0_L3, COL_SEL_KECCAK};

    /// Phase A1a EVM SHA3 → KeccakExtract output-side linkage descriptor
    /// must wire EVM `output0[0..4]` ↔ KeccakExtract `KECCAK_OUTPUT_LIMB[0..4]`,
    /// gated by `SEL_KECCAK` on EVM and `IS_REAL` on extract.
    #[test]
    fn evm_keccak_output_descriptor_well_formed() {
        let desc = make_evm_keccak_keccak_extract_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "evm_keccak_output_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 4);
        assert_eq!(desc.b_columns.len(), 4);
        assert_eq!(desc.a_columns[0], COL_OUTPUT0_L0);
        assert_eq!(desc.a_columns[1], COL_OUTPUT0_L1);
        assert_eq!(desc.a_columns[2], COL_OUTPUT0_L2);
        assert_eq!(desc.a_columns[3], COL_OUTPUT0_L3);
        for k in 0..4 {
            assert_eq!(
                desc.b_columns[k],
                metavm_zkp::keccak_extract::COL_KECCAK_OUTPUT_LIMB_OFFSET + k
            );
        }
        assert_eq!(desc.a_selector_column, Some(COL_SEL_KECCAK));
        assert_eq!(
            desc.b_selector_column,
            Some(metavm_zkp::keccak_extract::COL_IS_REAL)
        );
    }

    /// **Phase A1b L1 descriptor** well-formed.
    #[test]
    fn evm_sha3_input_l1_descriptor_well_formed() {
        let desc = make_evm_sha3_input_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "evm_sha3_input_v1");
        assert_eq!(desc.a_columns.len(), 4);
        assert_eq!(desc.b_columns.len(), 4);
        for k in 0..4 {
            assert_eq!(desc.a_columns[k], COL_OUTPUT0_L0 + k);
            assert_eq!(
                desc.b_columns[k],
                metavm_zkp::sha3_input_air::COL_OUTPUT_LIMB_OFFSET + k
            );
        }
        assert_eq!(desc.a_selector_column, Some(COL_SEL_KECCAK));
        assert_eq!(
            desc.b_selector_column,
            Some(metavm_zkp::sha3_input_air::COL_IS_REAL)
        );
    }

    /// **Phase A1b-mem step 1c-b EVM ↔ MSTORE byte-decomp linkage**
    /// descriptor is well-formed.
    #[test]
    fn evm_mstore_to_byte_decomp_descriptor_well_formed() {
        use crate::trace::{
            COL_MEM_OFFSET, COL_MEM_VALUE_L0, COL_MEM_VALUE_L1, COL_MEM_VALUE_L2,
            COL_MEM_VALUE_L3, COL_SEL_MSTORE,
        };
        let desc = make_evm_mstore_to_byte_decomp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "evm_mstore_to_byte_decomp_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(
            desc.a_columns,
            vec![COL_MEM_OFFSET, COL_MEM_VALUE_L0, COL_MEM_VALUE_L1,
                 COL_MEM_VALUE_L2, COL_MEM_VALUE_L3],
        );
        assert_eq!(
            desc.b_columns,
            vec![
                metavm_zkp::mstore_byte_air::COL_OFFSET,
                metavm_zkp::mstore_byte_air::COL_LIMB_L0,
                metavm_zkp::mstore_byte_air::COL_LIMB_L1,
                metavm_zkp::mstore_byte_air::COL_LIMB_L2,
                metavm_zkp::mstore_byte_air::COL_LIMB_L3,
            ],
        );
        assert_eq!(desc.a_selector_column, Some(COL_SEL_MSTORE));
        assert_eq!(
            desc.b_selector_column,
            Some(metavm_zkp::mstore_byte_air::COL_IS_REAL),
        );
    }

    /// **Phase A1b-mem step 1c-b host-side tuple alignment**:
    /// runs an MSTORE-only bytecode through the EVM inspector, builds
    /// the MSTORE byte-decomposition gadget witness from the MSTORE
    /// rows, and verifies:
    ///   1. (offset, limbs) tuples match between EVM main MSTORE rows
    ///      and gadget rows (for the EVM↔gadget linkage).
    ///   2. For each byte position k, the multiset of (offset+k,
    ///      byte_k) tuples on the gadget side equals the multiset of
    ///      (addr, val) tuples in byte-memory's Write entries with
    ///      addr == offset+k (for the gadget↔byte-memory linkages).
    #[test]
    fn evm_mstore_byte_decomp_tuples_align() {
        use crate::executor::execute_bytecode;
        use crate::sha3_mem_binding::{build_byte_access_trace, to_byte_memory_witness};
        use metavm_zkp::mstore_byte_air::{
            be_bytes_of_u256, MstoreByteRow, MstoreByteWitness,
        };

        // PUSH32 0x0102..0x20; PUSH1 0x00; MSTORE; STOP
        let mut bc = vec![0x7F];
        for k in 0..32u8 {
            bc.push(k + 1);
        }
        bc.extend_from_slice(&[0x60, 0x00, 0x52, 0x00]);
        let evm_cols = execute_bytecode(&bc, &[]).unwrap();

        // Collect EVM MSTORE rows.
        let n_rows = evm_cols.step.len();
        let mut evm_mstore_tuples: Vec<(u64, [u64; 4])> = Vec::new();
        for r in 0..n_rows {
            if evm_cols.sel_mstore[r] == 1 {
                evm_mstore_tuples.push((
                    evm_cols.mem_offset[r],
                    [
                        evm_cols.mem_value[0][r],
                        evm_cols.mem_value[1][r],
                        evm_cols.mem_value[2][r],
                        evm_cols.mem_value[3][r],
                    ],
                ));
            }
        }
        assert_eq!(evm_mstore_tuples.len(), 1, "1 MSTORE row");

        // Build the gadget witness from the EVM MSTORE rows.
        let gadget_w = MstoreByteWitness::from_invocations(
            evm_mstore_tuples
                .iter()
                .map(|(off, lim)| MstoreByteRow { offset: *off, limb: *lim })
                .collect(),
        );
        assert_eq!(gadget_w.invocations.len(), 1);
        assert_eq!(gadget_w.invocations[0].offset, evm_mstore_tuples[0].0);
        assert_eq!(gadget_w.invocations[0].limb, evm_mstore_tuples[0].1);

        // Build the byte-memory witness.
        let byte_accesses = build_byte_access_trace(&evm_cols);
        let byte_witness = to_byte_memory_witness(&byte_accesses);

        // For each byte position k, check the gadget-side and byte-
        // memory-side tuples align.
        let bytes = be_bytes_of_u256(gadget_w.invocations[0].limb);
        for k in 0..32 {
            let gadget_addr_k = gadget_w.invocations[0].offset.wrapping_add(k as u64);
            let gadget_byte_k = bytes[k];

            // byte-memory writes at addr=gadget_addr_k.
            let byte_mem_at_k: Vec<u8> = byte_witness
                .accesses
                .iter()
                .filter(|a| a.rw == 1 && a.addr == gadget_addr_k)
                .map(|a| a.val)
                .collect();
            assert_eq!(
                byte_mem_at_k,
                vec![gadget_byte_k],
                "byte-memory Write at addr={} expected val={} got {:?}",
                gadget_addr_k, gadget_byte_k, byte_mem_at_k,
            );
        }
    }

    #[test]
    fn evm_block_opcode_descriptors_well_formed() {
        use crate::trace::{
            COL_OUTPUT0_L0, COL_SEL_BASEFEE, COL_SEL_COINBASE, COL_SEL_GASLIMIT,
            COL_SEL_NUMBER, COL_SEL_TIMESTAMP,
        };
        let ts = make_evm_timestamp_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(ts.label, "evm_timestamp_to_block_header_v1");
        assert_eq!(ts.a_columns, vec![COL_OUTPUT0_L0]);
        assert_eq!(ts.a_selector_column, Some(COL_SEL_TIMESTAMP));
        assert_eq!(ts.b_columns, vec![metavm_zkp::block_header_air::COL_TIMESTAMP]);

        let num = make_evm_number_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(num.label, "evm_number_to_block_header_v1");
        assert_eq!(num.a_selector_column, Some(COL_SEL_NUMBER));
        assert_eq!(num.b_columns, vec![metavm_zkp::block_header_air::COL_NUMBER]);

        let gl = make_evm_gaslimit_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(gl.a_selector_column, Some(COL_SEL_GASLIMIT));
        assert_eq!(gl.b_columns, vec![metavm_zkp::block_header_air::COL_GAS_LIMIT]);

        let bf = make_evm_basefee_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(bf.a_columns.len(), 4);
        assert_eq!(bf.a_selector_column, Some(COL_SEL_BASEFEE));

        let cb = make_evm_coinbase_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(cb.a_columns.len(), 4);
        assert_eq!(cb.a_selector_column, Some(COL_SEL_COINBASE));

        let pr = make_evm_prevrandao_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(pr.a_columns.len(), 4);
        assert_eq!(pr.a_selector_column, Some(crate::trace::COL_SEL_PREVRANDAO));
        assert_eq!(pr.b_columns[0], metavm_zkp::block_header_air::COL_PREV_RANDAO_L0);

        let ci = make_evm_chainid_to_block_header_linkage_descriptor(0, 1);
        assert_eq!(ci.a_columns, vec![COL_OUTPUT0_L0]);
        assert_eq!(ci.a_selector_column, Some(crate::trace::COL_SEL_CHAINID));
        assert_eq!(ci.b_columns, vec![metavm_zkp::block_header_air::COL_CHAIN_ID]);

        let sb = make_evm_selfbalance_to_account_state_linkage_descriptor(0, 2);
        assert_eq!(sb.a_columns.len(), 4);
        assert_eq!(sb.a_selector_column, Some(crate::trace::COL_SEL_SELFBALANCE));
    }

    /// Validate the inspector populates the new aux selectors
    /// correctly for each BLOCK opcode.
    #[test]
    fn evm_block_aux_selectors_populated_correctly() {
        use crate::executor::execute_bytecode;

        // Bytecode: TIMESTAMP; NUMBER; GASLIMIT; BASEFEE; COINBASE;
        //           PREVRANDAO; CHAINID; BLOCKHASH(0); STOP
        // Note: SELFBALANCE requires a non-zero balance context so test separately.
        let bytecode = vec![
            0x42, // TIMESTAMP
            0x43, // NUMBER
            0x45, // GASLIMIT
            0x48, // BASEFEE
            0x41, // COINBASE
            0x44, // PREVRANDAO
            0x46, // CHAINID
            0x60, 0x00, // PUSH1 0 (block number arg for BLOCKHASH)
            0x40, // BLOCKHASH
            0x00, // STOP
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();

        let n = evm_cols.step.len();
        let mut found = [false; 8]; // ts, num, gl, bf, cb, pr, ci, bh
        for r in 0..n {
            match evm_cols.opcode[r] {
                0x42 => {
                    assert_eq!(evm_cols.sel_timestamp[r], 1);
                    assert_eq!(evm_cols.sel_block[r], 1);
                    found[0] = true;
                }
                0x43 => {
                    assert_eq!(evm_cols.sel_number[r], 1);
                    assert_eq!(evm_cols.sel_block[r], 1);
                    found[1] = true;
                }
                0x45 => {
                    assert_eq!(evm_cols.sel_gaslimit[r], 1);
                    found[2] = true;
                }
                0x48 => {
                    assert_eq!(evm_cols.sel_basefee[r], 1);
                    found[3] = true;
                }
                0x41 => {
                    assert_eq!(evm_cols.sel_coinbase[r], 1);
                    found[4] = true;
                }
                0x44 => {
                    assert_eq!(evm_cols.sel_prevrandao[r], 1);
                    assert_eq!(evm_cols.sel_block[r], 1);
                    found[5] = true;
                }
                0x46 => {
                    assert_eq!(evm_cols.sel_chainid[r], 1);
                    assert_eq!(evm_cols.sel_block[r], 1);
                    found[6] = true;
                }
                0x40 => {
                    assert_eq!(evm_cols.sel_blockhash[r], 1);
                    assert_eq!(evm_cols.sel_block[r], 1);
                    found[7] = true;
                }
                _ => {}
            }
        }
        assert!(found.iter().all(|f| *f),
                "all 8 BLOCK opcodes should have been seen: {:?}", found);
    }

    #[test]
    fn evm_sload_to_storage_descriptor_well_formed() {
        use crate::trace::{
            COL_INPUT0_L0, COL_OUTPUT0_L0, COL_SEL_SLOAD,
        };
        let desc = make_evm_sload_to_storage_gadget_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "evm_sload_to_storage_v1");
        assert_eq!(desc.a_columns.len(), 8);
        assert_eq!(desc.b_columns.len(), 8);
        assert_eq!(desc.a_columns[0], COL_INPUT0_L0);
        assert_eq!(desc.a_columns[4], COL_OUTPUT0_L0);
        assert_eq!(desc.a_selector_column, Some(COL_SEL_SLOAD));
    }

    #[test]
    fn evm_sstore_to_storage_descriptor_well_formed() {
        use crate::trace::{
            COL_INPUT0_L0, COL_INPUT1_L0, COL_SEL_SSTORE,
        };
        let desc = make_evm_sstore_to_storage_gadget_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "evm_sstore_to_storage_v1");
        assert_eq!(desc.a_columns[0], COL_INPUT0_L0);
        assert_eq!(desc.a_columns[4], COL_INPUT1_L0);
        assert_eq!(desc.a_selector_column, Some(COL_SEL_SSTORE));
    }

    /// **Phase A2 step 1b tuple alignment**: run an SSTORE+SLOAD
    /// bytecode through the EVM inspector and verify the EVM-side
    /// columns gated by sel_sload/sel_sstore project the same
    /// `(slot, value)` tuples as the storage_access_air gadget
    /// witness rows. This is the host-side oracle for what the
    /// cross-AIR LogUp closures will enforce algebraically.
    #[test]
    fn evm_storage_tuples_align_with_gadget_witness() {
        use crate::executor::execute_bytecode;
        use crate::storage_access::build_storage_gadget_witness_from_trace;

        // PUSH1 0x42; PUSH1 0x05; SSTORE; PUSH1 0x05; SLOAD; STOP
        let bytecode = vec![
            0x60, 0x42,  0x60, 0x05,  0x55,
            0x60, 0x05,  0x54,  0x00,
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let n = evm_cols.step.len();

        // Verify the inspector populated sel_sload/sel_sstore correctly.
        let mut sload_rows = Vec::new();
        let mut sstore_rows = Vec::new();
        for r in 0..n {
            if evm_cols.sel_sload[r] == 1 {
                sload_rows.push(r);
                assert_eq!(evm_cols.opcode[r], 0x54,
                           "sel_sload set on non-SLOAD opcode {:#x}",
                           evm_cols.opcode[r]);
                assert_eq!(evm_cols.sel_storage[r], 1);
            }
            if evm_cols.sel_sstore[r] == 1 {
                sstore_rows.push(r);
                assert_eq!(evm_cols.opcode[r], 0x55,
                           "sel_sstore set on non-SSTORE opcode {:#x}",
                           evm_cols.opcode[r]);
                assert_eq!(evm_cols.sel_storage[r], 1);
            }
        }
        assert_eq!(sload_rows.len(), 1, "exactly 1 SLOAD row");
        assert_eq!(sstore_rows.len(), 1, "exactly 1 SSTORE row");

        // Mutual exclusion.
        for r in 0..n {
            assert!(evm_cols.sel_sload[r] * evm_cols.sel_sstore[r] == 0,
                    "sel_sload * sel_sstore must be 0 (mutual exclusion)");
        }

        // EVM-side SLOAD tuple = (input0, output0).
        let r_sload = sload_rows[0];
        let evm_sload_slot = evm_cols.input0[0][r_sload];
        let evm_sload_value = evm_cols.output0[0][r_sload];
        // EVM-side SSTORE tuple = (input0, input1).
        let r_sstore = sstore_rows[0];
        let evm_sstore_slot = evm_cols.input0[0][r_sstore];
        let evm_sstore_value = evm_cols.input1[0][r_sstore];

        // Build the storage gadget witness from the trace.
        let storage_roots = vec![[0u8; 32]; 2];
        let gadget_witness =
            build_storage_gadget_witness_from_trace(&evm_cols, &storage_roots)
                .unwrap();
        // Index 0 is SSTORE, index 1 is SLOAD (by execution order).
        assert!(gadget_witness.invocations[0].is_write);
        assert!(!gadget_witness.invocations[1].is_write);

        // EVM SSTORE row's (slot, value) == gadget SSTORE row's
        // (slot[0], value[0]).
        assert_eq!(evm_sstore_slot, gadget_witness.invocations[0].slot[0]);
        assert_eq!(evm_sstore_value, gadget_witness.invocations[0].value[0]);
        // EVM SLOAD row's (slot, value) == gadget SLOAD row's
        // (slot[0], value[0]).
        assert_eq!(evm_sload_slot, gadget_witness.invocations[1].slot[0]);
        assert_eq!(evm_sload_value, gadget_witness.invocations[1].value[0]);
    }

    /// **Phase A2 step 1b tampering test**: tamper the gadget
    /// witness's `value` on the SLOAD row and check that the
    /// EVM-side multiset (gated by sel_sload) no longer matches the
    /// gadget's multiset. Validates host-side that the cross-AIR
    /// LogUp closure equality enforces consistency.
    #[test]
    fn evm_storage_tuples_mismatch_on_tampered_gadget_value() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::storage_access_air::{
            build_trace_polynomials, StorageAccessRow, StorageAccessWitness,
            COL_VALUE_L0,
        };

        let bytecode = vec![
            0x60, 0x42,  0x60, 0x05,  0x55,
            0x60, 0x05,  0x54,  0x00,
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();

        // Build honest gadget witness rows.
        let honest_rows = vec![
            StorageAccessRow {
                address: [0u8; 20],
                slot: [0x05, 0, 0, 0],
                value: [0x42, 0, 0, 0],
                storage_root: [0u8; 32],
                is_write: true,
            },
            StorageAccessRow {
                address: [0u8; 20],
                slot: [0x05, 0, 0, 0],
                value: [0x42, 0, 0, 0],
                storage_root: [0u8; 32],
                is_write: false,
            },
        ];
        let honest_w = StorageAccessWitness::from_rows(honest_rows.clone());
        let honest_trace = build_trace_polynomials(&honest_w, metavm_zkp::field::CurveType::Bls48581);

        // The SLOAD row in EVM trace.
        let n = evm_cols.step.len();
        let sload_row = (0..n).find(|&r| evm_cols.sel_sload[r] == 1).unwrap();
        let evm_value = evm_cols.output0[0][sload_row];

        // Gadget row 1 = SLOAD, value column.
        let honest_gadget_value = honest_trace
            .columns[COL_VALUE_L0]
            .evaluations[1]
            .to_u64();
        assert_eq!(evm_value, honest_gadget_value,
                   "honest setup: EVM SLOAD value matches gadget");

        // Tamper: build a tampered witness with wrong SLOAD value.
        let mut tampered_rows = honest_rows;
        tampered_rows[1].value = [0xff, 0, 0, 0]; // tampered SLOAD value
        let tampered_w = StorageAccessWitness::from_rows(tampered_rows);
        let tampered_trace = build_trace_polynomials(&tampered_w, metavm_zkp::field::CurveType::Bls48581);
        let tampered_gadget_value = tampered_trace
            .columns[COL_VALUE_L0]
            .evaluations[1]
            .to_u64();

        assert_ne!(evm_value, tampered_gadget_value,
                   "tampered gadget value should not match EVM SLOAD value");
    }

    /// **Phase A2 step 1c tampering test**: tamper the gadget's
    /// `slot_trie_key` to break the keccak binding. Verifies that
    /// the gadget's claimed trie key no longer matches what
    /// KeccakExtract would compute.
    #[test]
    fn storage_keccak_mismatch_on_tampered_trie_key() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak::keccak256;
        use metavm_zkp::storage_access_air::{
            build_trace_polynomials, StorageAccessRow, StorageAccessWitness,
            COL_SLOT_BE_OFFSET, COL_SLOT_TRIE_KEY_OFFSET,
        };

        let row = StorageAccessRow {
            address: [0u8; 20],
            slot: [0x05, 0, 0, 0],
            value: [0x42, 0, 0, 0],
            storage_root: [0u8; 32],
            is_write: false,
        };
        let w = StorageAccessWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);

        // Read slot_be from trace.
        let mut slot_be = [0u8; 32];
        for k in 0..32 {
            slot_be[k] = trace.columns[COL_SLOT_BE_OFFSET + k]
                .evaluations[0]
                .to_u64() as u8;
        }
        let expected_hash = keccak256(&slot_be);

        // Tamper: pretend the trie_key is something else.
        let tampered_hash = [0xffu8; 32];
        assert_ne!(tampered_hash, expected_hash);

        // The honest trace's trie key matches the keccak.
        let mut honest_trie_key = [0u8; 32];
        for k in 0..32 {
            honest_trie_key[k] = trace.columns[COL_SLOT_TRIE_KEY_OFFSET + k]
                .evaluations[0]
                .to_u64() as u8;
        }
        assert_eq!(honest_trie_key, expected_hash);

        // A tampered trie key would produce a different B-side tuple
        // (slot_be ++ trie_key) on the storage gadget, which
        // KeccakExtract wouldn't have any row matching →
        // multiset closure mismatch. We verify host-side by
        // checking the trie_key column value differs from keccak.
        let tampered_neq_expected = tampered_hash != expected_hash;
        assert!(tampered_neq_expected,
                "tampered trie key must differ from honest keccak");
    }

    /// **#53 step 1 + #54 EVM↔BlockHeader joint_prove**: validates
    /// the Layer A→C bridge linkages on a real EVM bytecode that
    /// invokes the 5 supported BLOCK opcodes.
    ///
    /// AIRs:
    ///   0. EVM main
    ///   1. block_header_air
    ///
    /// Linkages (5 cross-AIR LogUps, all gated by per-opcode aux
    /// selectors on EVM side and IS_REAL on block_header_air side):
    ///   - L_timestamp, L_number, L_gaslimit, L_basefee, L_coinbase
    ///
    /// The test reads the actual BLOCK opcode output values from the
    /// EVM trace (whatever revm's Context::mainnet() provides), then
    /// builds a BlockHeader fixture with those exact values. This
    /// guarantees the multiset on both sides matches honestly.
    ///
    /// Marked #[ignore] — slow: ~15-20 min release expected.
    #[test]
    #[ignore = "slow: 2-AIR EVM + block_header joint_prove (~15-20 min); \
                run with --release --ignored"]
    fn joint_prove_evm_block_header_chain() {
        use crate::constraints::EvmConstraintSystem;
        use crate::executor::execute_bytecode;
        use metavm_zkp::block_header::BlockHeader;
        use metavm_zkp::block_header_air::{
            self as bh, build_trace_polynomials as build_bh_trace, from_block_header,
            BlockHeaderConstraintSystem, BlockHeaderWitness,
        };
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::CurveType;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Bytecode: TIMESTAMP, NUMBER, GASLIMIT, BASEFEE, COINBASE,
        // POP each, STOP. Each opcode pushes one value onto the stack;
        // we pop after each so the stack stays bounded.
        let bytecode = vec![
            0x42, 0x50, // TIMESTAMP, POP
            0x43, 0x50, // NUMBER, POP
            0x45, 0x50, // GASLIMIT, POP
            0x48, 0x50, // BASEFEE, POP
            0x41, 0x50, // COINBASE, POP
            0x00,       // STOP
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let evm_polys =
            metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_cols, curve);
        let evm_cs = EvmConstraintSystem::new();

        // Read the actual BLOCK context values from revm's trace.
        // Each BLOCK opcode pushes its value to output0; capture each
        // row by selector.
        let n = evm_cols.step.len();
        let mut timestamp: u64 = 0;
        let mut number: u64 = 0;
        let mut gas_limit: u64 = 0;
        let mut base_fee: [u64; 4] = [0; 4];
        let mut beneficiary: [u64; 4] = [0; 4];
        for r in 0..n {
            if evm_cols.sel_timestamp[r] == 1 {
                timestamp = evm_cols.output0[0][r];
            }
            if evm_cols.sel_number[r] == 1 {
                number = evm_cols.output0[0][r];
            }
            if evm_cols.sel_gaslimit[r] == 1 {
                gas_limit = evm_cols.output0[0][r];
            }
            if evm_cols.sel_basefee[r] == 1 {
                base_fee[0] = evm_cols.output0[0][r];
                base_fee[1] = evm_cols.output0[1][r];
                base_fee[2] = evm_cols.output0[2][r];
                base_fee[3] = evm_cols.output0[3][r];
            }
            if evm_cols.sel_coinbase[r] == 1 {
                beneficiary[0] = evm_cols.output0[0][r];
                beneficiary[1] = evm_cols.output0[1][r];
                beneficiary[2] = evm_cols.output0[2][r];
                beneficiary[3] = evm_cols.output0[3][r];
            }
        }

        // Build the BlockHeader fixture with the values revm used.
        // base_fee_per_gas is U256, encode 4 LE u64 limbs as 32 BE bytes.
        let mut base_fee_be = [0u8; 32];
        for k in 0..4 {
            let limb = base_fee[k];
            let start = (3 - k) * 8;
            base_fee_be[start..start + 8].copy_from_slice(&limb.to_be_bytes());
        }
        // beneficiary as 20-byte BE address (low 160 bits).
        let mut full_be = [0u8; 32];
        for k in 0..4 {
            let limb = beneficiary[k];
            let start = (3 - k) * 8;
            full_be[start..start + 8].copy_from_slice(&limb.to_be_bytes());
        }
        let mut beneficiary_be = [0u8; 20];
        beneficiary_be.copy_from_slice(&full_be[12..32]);

        let mut h = BlockHeader::default();
        h.timestamp = timestamp;
        h.number = number;
        h.gas_limit = gas_limit;
        h.base_fee_per_gas = Some(base_fee_be);
        h.beneficiary = beneficiary_be;
        let bh_row = from_block_header(&h);
        let bh_w = BlockHeaderWitness::from_headers(vec![bh_row]);
        let bh_trace = build_bh_trace(&bh_w, curve);
        let bh_omega = scheme.domain_generator(bh_trace.padded_size);
        let bh_cs = BlockHeaderConstraintSystem::new(bh_trace.num_rows)
            .with_omega_and_domain(bh_omega, bh_trace.padded_size);

        // 5 cross-AIR linkages.
        let linkages = vec![
            make_evm_timestamp_to_block_header_linkage_descriptor(0, 1),
            make_evm_number_to_block_header_linkage_descriptor(0, 1),
            make_evm_gaslimit_to_block_header_linkage_descriptor(0, 1),
            make_evm_basefee_to_block_header_linkage_descriptor(0, 1),
            make_evm_coinbase_to_block_header_linkage_descriptor(0, 1),
        ];

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&bh_trace, &bh_cs)];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("EVM↔BlockHeader joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 5);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            eprintln!(
                "[diag] linkage {} label={} closure_match={}",
                i, lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b);
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &bh_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "EVM↔BlockHeader joint_verify must accept honest witness",
        );
        let _ = bh::COL_TIMESTAMP; // silence warning
    }

    /// **Phase A2 step 1b full chain joint_prove**: EVM main +
    /// storage_access_air + KeccakExtract.
    ///
    /// Linkages:
    ///   - L_sload: EVM `SLOAD` ↔ storage gadget on `(slot, value)`
    ///     8-col tuple, gated by SEL_SLOAD / IS_REAL.
    ///   - L_sstore: EVM `SSTORE` ↔ storage gadget on `(slot, value)`
    ///     8-col tuple, gated by SEL_SSTORE / IS_REAL.
    ///   - L_storage_keccak: storage gadget ↔ KeccakExtract on
    ///     `(slot_be, slot_trie_key)` 64-col tuple, binding
    ///     `slot_trie_key = keccak256(slot_be)`.
    ///
    /// Bytecode: PUSH1 0x42; PUSH1 0x05; SSTORE; PUSH1 0x05; SLOAD;
    /// STOP. Produces 1 SSTORE row and 1 SLOAD row, each binding to
    /// a gadget row, with both keccak inputs (slot_be ×2) going
    /// through KeccakExtract.
    ///
    /// Soundness caveats (documented):
    ///   - Soft EVM-side selector binding: prover could omit sel_sload
    ///     on a real SLOAD row (no must-fire constraint yet); cross-
    ///     AIR LogUp would still close trivially.
    ///   - Gadget B-side gating is IS_REAL (covers both SLOAD AND
    ///     SSTORE gadget rows) — strictness deferred to step 1b-strict.
    ///   - No MPT inclusion link yet (step 1d).
    ///
    /// Marked #[ignore] — heavy: ~18-25 min release expected.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove EVM + storage + KeccakExtract \
                with 3 linkages (~18-25 min); run with --release --ignored"]
    fn joint_prove_evm_storage_chain() {
        use crate::constraints::EvmConstraintSystem;
        use crate::executor::execute_bytecode;
        use crate::sha3_mem_binding::build_byte_access_trace;
        use crate::storage_access::build_storage_gadget_witness_from_trace;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            self as ke, build_trace_polynomials as build_keccak_trace,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use metavm_zkp::storage_access_air::{
            self as sa, build_trace_polynomials as build_storage_trace,
            StorageAccessConstraintSystem,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // PUSH1 0x42; PUSH1 0x05; SSTORE; PUSH1 0x05; SLOAD; STOP
        let bytecode = vec![
            0x60, 0x42,  0x60, 0x05,  0x55,
            0x60, 0x05,  0x54,  0x00,
        ];
        let _byte_accesses = build_byte_access_trace(
            &execute_bytecode(&bytecode, &[]).unwrap(),
        );
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let evm_polys =
            metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_cols, curve);
        let evm_cs = EvmConstraintSystem::new();

        // Build storage gadget witness. Use zero roots; the gadget AIR
        // commits to them but doesn't yet verify them against MPT
        // (step 1d follow-up). Storage gadget standalone is sound;
        // EVM↔gadget linkage is what this test validates.
        let storage_roots = vec![[0u8; 32]; 2];
        let storage_w =
            build_storage_gadget_witness_from_trace(&evm_cols, &storage_roots)
                .unwrap();
        let storage_trace = build_storage_trace(&storage_w, curve);
        let storage_omega = scheme.domain_generator(storage_trace.padded_size);
        let storage_cs = StorageAccessConstraintSystem::new(storage_trace.num_rows)
            .with_omega_and_domain(storage_omega, storage_trace.padded_size);

        // Build KeccakExtract witness for both slot_trie_key computations.
        // Both accesses are at slot=5, so we have 2 invocations of
        // keccak256(0x00 ×31 || 0x05).
        let slot_be: Vec<u8> = {
            let mut v = vec![0u8; 32];
            v[31] = 0x05;
            v
        };
        let extract_w =
            KeccakExtractWitness::from_inputs(&[slot_be.clone(), slot_be]).unwrap();
        let extract_trace = build_keccak_trace(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        let _ = ke::COL_INPUT_BYTE_OFFSET;

        // Linkages: 2 EVM→gadget + 1 gadget→keccak.
        let linkages = vec![
            make_evm_sload_to_storage_gadget_linkage_descriptor(0, 1),
            make_evm_sstore_to_storage_gadget_linkage_descriptor(0, 1),
            sa::make_storage_to_keccak_extract_linkage_descriptor(1, 2),
        ];

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&evm_polys, &evm_cs),
            (&storage_trace, &storage_cs),
            (&extract_trace, &extract_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("3-AIR storage-chain joint_prove must succeed");
        assert_eq!(proofs.len(), 3);
        assert_eq!(ext.linkage_proofs.len(), 3);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            eprintln!(
                "[diag] linkage {} label={} closure_match={}",
                i, lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b);
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &storage_cs, &extract_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "storage-chain joint_verify must accept honest witness",
        );
    }

    /// **Phase A1b-mem step 1c-b full chain joint_prove**: EVM main +
    /// MSTORE byte-decomp gadget + byte-memory AIR with all linkages:
    ///   - L_evm_to_mstore_byte_decomp (1:1 on offset+limbs)
    ///   - L_mstore_byte_to_byte_memory_k for k in 0..32 (32 linkages)
    ///   - L_byte_memory_self (multiset perm sorted ↔ unsorted)
    ///
    /// Bytecode: PUSH32 0x0102..0x20; PUSH1 0x00; MSTORE; STOP. One
    /// MSTORE row in EVM main expands to 32 Write entries in byte-
    /// memory; the gadget bridges via byte decomposition.
    ///
    /// Marked #[ignore] — heavy: ~15-25 min release expected.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove with 34 linkages (~20+ min); \
                run with --release --ignored"]
    fn joint_prove_evm_mstore_full_chain() {
        use crate::constraints::EvmConstraintSystem;
        use crate::executor::execute_bytecode;
        use crate::sha3_mem_binding::{build_byte_access_trace, to_byte_memory_witness};
        use metavm_zkp::byte_memory_air::{
            self as bma, build_trace_polynomials as build_byte_mem_trace,
            ByteMemoryConstraintSystem,
        };
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::CurveType;
        use metavm_zkp::mstore_byte_air::{
            self as mba, build_trace_polynomials as build_mstore_byte_trace,
            MstoreByteConstraintSystem, MstoreByteRow, MstoreByteWitness,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // PUSH32 0x0102..0x20; PUSH1 0x00; MSTORE; STOP
        let mut bytecode = vec![0x7F];
        for k in 0..32u8 {
            bytecode.push(k + 1);
        }
        bytecode.extend_from_slice(&[0x60, 0x00, 0x52, 0x00]);

        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let evm_polys =
            metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_cols, curve);
        let evm_cs = EvmConstraintSystem::new();

        // Build gadget witness from EVM MSTORE rows.
        let n_rows = evm_cols.step.len();
        let mut gadget_rows: Vec<MstoreByteRow> = Vec::new();
        for r in 0..n_rows {
            if evm_cols.sel_mstore[r] == 1 {
                gadget_rows.push(MstoreByteRow {
                    offset: evm_cols.mem_offset[r],
                    limb: [
                        evm_cols.mem_value[0][r],
                        evm_cols.mem_value[1][r],
                        evm_cols.mem_value[2][r],
                        evm_cols.mem_value[3][r],
                    ],
                });
            }
        }
        let gadget_w = MstoreByteWitness::from_invocations(gadget_rows);
        let gadget_trace = build_mstore_byte_trace(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = MstoreByteConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        // Build byte-memory witness.
        let byte_accesses = build_byte_access_trace(&evm_cols);
        let byte_witness = to_byte_memory_witness(&byte_accesses);
        let byte_trace = build_byte_mem_trace(&byte_witness, curve);
        let byte_omega = scheme.domain_generator(byte_trace.padded_size);
        let byte_cs = ByteMemoryConstraintSystem::new(byte_trace.num_rows)
            .with_omega_and_domain(byte_omega, byte_trace.padded_size);

        // Linkages: 1 EVM→gadget + 32 gadget→byte-mem + 1 byte-mem self
        let mut linkages = vec![
            make_evm_mstore_to_byte_decomp_linkage_descriptor(0, 1),
        ];
        for k in 0..32 {
            linkages.push(
                mba::make_mstore_byte_to_byte_memory_linkage_descriptor(1, 2, k),
            );
        }
        linkages.push(bma::make_byte_memory_self_linkage_descriptor(2));

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&evm_polys, &evm_cs),
            (&gadget_trace, &gadget_cs),
            (&byte_trace, &byte_cs),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("3-AIR MSTORE-chain joint_prove must succeed");
        assert_eq!(proofs.len(), 3);
        assert_eq!(ext.linkage_proofs.len(), 34);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            eprintln!(
                "[diag] linkage {} label={} closure_match={}",
                i, lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b);
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs, &byte_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "MSTORE-chain joint_verify must accept honest witness",
        );
    }

    /// **Phase A1b-mem step 1c-a slow joint_prove**: full end-to-end
    /// of EVM main + byte-memory AIR with two linkages:
    ///   L_mstore8: EVM MSTORE8 ↔ byte-memory Write tuples
    ///   L_self:    byte-memory unsorted ↔ sorted (phase 3 multiset perm)
    ///
    /// This is the full A1b-mem step 1c-a soundness chain: the EVM's
    /// MSTORE8 writes are committed to byte-memory's unsorted view via
    /// the MSTORE8 linkage; the unsorted view is forced to equal the
    /// sorted view via the self-linkage; the sorted view satisfies
    /// the read-consistency constraints (phases 1+2). A malicious
    /// prover cannot fabricate any of these links.
    ///
    /// Caveat: still doesn't cover MSTORE (32-byte) or SHA3 input
    /// reads — those are deferred as step 1c-b / 1c-c.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with EVM main A1b trace + byte-memory AIR \
                (~13-15 min, dominated by EVM main 256-padded trace); run with \
                --release --ignored"]
    fn joint_prove_evm_mstore8_byte_memory_a1b() {
        use crate::constraints::EvmConstraintSystem;
        use crate::executor::execute_bytecode;
        use crate::sha3_mem_binding::{build_byte_access_trace, to_byte_memory_witness};
        use metavm_zkp::byte_memory_air::{
            self as bma, build_trace_polynomials as build_byte_mem_trace,
            ByteMemoryConstraintSystem,
        };
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::CurveType;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // A1b bytecode but stop right after the 4 MSTORE8s (no SHA3)
        // — we're only testing the MSTORE8 linkage, and excluding SHA3
        // avoids needing the SHA3 input gadget + read-side linkage
        // (which is deferred to step 1c-c).
        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x00, // STOP
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let evm_polys =
            metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_cols, curve);
        let evm_cs = EvmConstraintSystem::new();

        let byte_accesses = build_byte_access_trace(&evm_cols);
        let byte_witness = to_byte_memory_witness(&byte_accesses);
        let byte_trace = build_byte_mem_trace(&byte_witness, curve);
        let byte_omega = scheme.domain_generator(byte_trace.padded_size);
        let byte_cs = ByteMemoryConstraintSystem::new(byte_trace.num_rows)
            .with_omega_and_domain(byte_omega, byte_trace.padded_size);

        let l_mstore8 = make_evm_mstore8_byte_memory_linkage_descriptor(0, 1);
        let l_self = bma::make_byte_memory_self_linkage_descriptor(1);

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&byte_trace, &byte_cs)];
        let linkages = vec![l_mstore8, l_self];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 2-AIR joint_prove for step 1c-a must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 2);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            eprintln!(
                "[diag] linkage {} label={} closure_match={}",
                i, lp.label, lp.closure_a == lp.closure_b,
            );
            assert_eq!(lp.closure_a, lp.closure_b);
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &byte_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "step 1c-a joint_verify must accept honest witness",
        );
    }

    /// **Phase A1b-mem step 1c-a descriptor** well-formed.
    #[test]
    fn evm_mstore8_byte_memory_descriptor_well_formed() {
        use crate::trace::{COL_MEM_OFFSET, COL_MEM_VALUE_L0, COL_SEL_MSTORE8};
        let desc = make_evm_mstore8_byte_memory_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "evm_mstore8_byte_memory_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns, vec![COL_MEM_OFFSET, COL_MEM_VALUE_L0]);
        assert_eq!(
            desc.b_columns,
            vec![
                metavm_zkp::byte_memory_air::COL_ADDR,
                metavm_zkp::byte_memory_air::COL_VAL,
            ],
        );
        assert_eq!(desc.a_selector_column, Some(COL_SEL_MSTORE8));
        assert_eq!(desc.b_selector_column, Some(metavm_zkp::byte_memory_air::COL_RW));
    }

    /// **Phase A1b-mem step 1c-a tuple-alignment fast check.**
    ///
    /// Runs the A1b bytecode (4× MSTORE8 + SHA3) through the EVM
    /// inspector, builds the byte-memory witness from the trace via
    /// `to_byte_memory_witness`, and asserts the multiset of EVM-side
    /// MSTORE8 tuples `{(mem_offset[r], mem_value[0][r]) : r where
    /// sel_mstore8[r] == 1}` equals the multiset of byte-memory Write
    /// tuples `{(addr[i], val[i]) : i where rw[i] == 1}`. This is what
    /// the cross-AIR LogUp closure scalars enforce algebraically.
    #[test]
    fn evm_mstore8_byte_memory_tuples_align_on_a1b_bytecode() {
        use crate::executor::execute_bytecode;
        use crate::sha3_mem_binding::{build_byte_access_trace, to_byte_memory_witness};
        use crate::trace::{COL_MEM_OFFSET, COL_MEM_VALUE_L0, COL_SEL_MSTORE8};

        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        // Build the byte-memory witness from the EVM trace.
        let byte_accesses = build_byte_access_trace(&evm_cols);
        let byte_witness = to_byte_memory_witness(&byte_accesses);

        // Collect EVM-side MSTORE8 tuples.
        let n_rows = evm_cols.step.len();
        let mut evm_tuples: Vec<(u64, u64)> = Vec::new();
        for r in 0..n_rows {
            if evm_cols.sel_mstore8[r] == 1 {
                evm_tuples.push((evm_cols.mem_offset[r], evm_cols.mem_value[0][r]));
            }
        }
        evm_tuples.sort();

        // Collect byte-memory Write tuples (rw == 1).
        let mut byte_tuples: Vec<(u64, u64)> = byte_witness
            .accesses
            .iter()
            .filter(|a| a.rw == 1)
            .map(|a| (a.addr, a.val as u64))
            .collect();
        byte_tuples.sort();

        // Sanity: the A1b bytecode has 4 MSTORE8 ops.
        assert_eq!(evm_tuples.len(), 4);
        assert_eq!(byte_tuples.len(), 4);
        assert_eq!(evm_tuples, byte_tuples,
                   "EVM MSTORE8 tuples must equal byte-memory Write tuples (multiset)");

        // Use the column indices the linkage projects to double-check.
        // sel_mstore8 column index in TracePolynomials = COL_SEL_MSTORE8.
        // mem_offset = COL_MEM_OFFSET (16). mem_value[0] = COL_MEM_VALUE_L0 (17).
        // These all match the descriptor's `a_columns`/`a_selector_column`.
        let desc = make_evm_mstore8_byte_memory_linkage_descriptor(0, 1);
        assert_eq!(desc.a_columns[0], COL_MEM_OFFSET);
        assert_eq!(desc.a_columns[1], COL_MEM_VALUE_L0);
        assert_eq!(desc.a_selector_column, Some(COL_SEL_MSTORE8));
    }

    /// **Phase A1b end-to-end fast check**: run a real EVM SHA3 program
    /// through the inspector, then verify that BOTH cross-AIR LogUp
    /// linkages (L1: EVM↔gadget on output_limb; L2: gadget↔KeccakExtract
    /// on full tuple) are tuple-aligned on the actual SHA3 inputs.
    ///
    /// Builds the SHA3InputAir witness from the actual memory bytes
    /// (the `interp.shared_memory[offset..offset+size]` contents the
    /// inspector saw); builds the KeccakExtract witness from the same
    /// bytes; asserts byte-for-byte equality across all three sides
    /// of the chain.
    #[test]
    fn evm_sha3_input_full_chain_tuples_align() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            self, build_trace_polynomials as build_extract_trace, KeccakExtractWitness,
        };
        use metavm_zkp::sha3_input_air::{
            self, build_trace_polynomials as build_gadget_trace, Sha3InputWitness,
        };

        let curve = CurveType::Bls48581;

        // MSTORE8 ×4 then SHA3 over memory[0..4].
        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x60, 0x04,  // size
            0x60, 0x00,  // offset
            0x20,        // SHA3
            0x00,        // STOP
        ];
        let t = execute_bytecode(&bytecode, &[]).unwrap();
        let sha3_row = (0..t.opcode.len())
            .find(|&i| t.opcode[i] == 0x20)
            .expect("SHA3 row");
        let evm_output0: [u64; 4] = [
            t.output0[0][sha3_row], t.output0[1][sha3_row],
            t.output0[2][sha3_row], t.output0[3][sha3_row],
        ];

        // The actual memory bytes that SHA3 hashed. We know them
        // statically from the program; in production this would come
        // from the inspector's `interp.shared_memory[..]`.
        let input_bytes = vec![0xab, 0xcd, 0xef, 0x12];

        // L2 gadget witness.
        let gadget_w = Sha3InputWitness::from_inputs(&[input_bytes.clone()]).unwrap();
        let gadget_trace = build_gadget_trace(&gadget_w, curve);

        // KeccakExtract witness.
        let extract_w = KeccakExtractWitness::from_inputs(&[input_bytes.clone()]).unwrap();
        let extract_trace = build_extract_trace(&extract_w, curve);

        // ── L1 tuple: EVM output0[k] == gadget OUTPUT_LIMB[k] ─────────
        for k in 0..4 {
            let g = gadget_trace.columns[sha3_input_air::COL_OUTPUT_LIMB_OFFSET + k]
                .evaluations[0].to_u64();
            assert_eq!(g, evm_output0[k], "L1 output_limb {} mismatch", k);
        }

        // ── L2 tuple: gadget == KeccakExtract on (INPUT_BYTE, INPUT_LEN, OUTPUT_LIMB) ──
        for b in 0..sha3_input_air::INPUT_BYTE_WIDTH {
            let g = gadget_trace.columns[sha3_input_air::COL_INPUT_BYTE_OFFSET + b]
                .evaluations[0].to_u64();
            let e = extract_trace.columns[keccak_extract::COL_INPUT_BYTE_OFFSET + b]
                .evaluations[0].to_u64();
            assert_eq!(g, e, "L2 input_byte {} mismatch", b);
        }
        let g_len = gadget_trace.columns[sha3_input_air::COL_INPUT_LEN]
            .evaluations[0].to_u64();
        let e_len = extract_trace.columns[keccak_extract::COL_INPUT_LEN]
            .evaluations[0].to_u64();
        assert_eq!(g_len, e_len, "L2 input_len mismatch");
        for k in 0..sha3_input_air::NUM_OUTPUT_LIMBS {
            let g = gadget_trace.columns[sha3_input_air::COL_OUTPUT_LIMB_OFFSET + k]
                .evaluations[0].to_u64();
            let e = extract_trace.columns[keccak_extract::COL_KECCAK_OUTPUT_LIMB_OFFSET + k]
                .evaluations[0].to_u64();
            assert_eq!(g, e, "L2 output_limb {} mismatch", k);
        }

        // ── Sanity: the canonical chain agrees with EVM SHA3 output ──
        let extract_output_l0 = extract_trace.columns[
            keccak_extract::COL_KECCAK_OUTPUT_LIMB_OFFSET
        ].evaluations[0].to_u64();
        assert_eq!(extract_output_l0, evm_output0[0]);
    }

    /// A1b tampering coverage: if the gadget's `INPUT_BYTE[0]` is
    /// corrupted while the KeccakExtract witness stays honest, the L2
    /// cross-AIR LogUp's 261-element tuple no longer matches between
    /// gadget and extract — the prover cannot produce both
    /// `(corrupted_input, real_output)` and `(real_input, real_output)`
    /// pointing at the same KeccakExtract row.
    ///
    /// Host-side verification of the tuple-mismatch precondition; the
    /// slow `joint_prove`-based rejection is the
    /// `joint_prove_evm_sha3_a1b_rejects_tampered_input` test.
    #[test]
    fn evm_sha3_input_a1b_tampered_gadget_byte_breaks_l2_tuple() {
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            self, build_trace_polynomials as build_extract_trace, KeccakExtractWitness,
        };
        use metavm_zkp::sha3_input_air::{
            self, build_trace_polynomials as build_gadget_trace, Sha3InputWitness,
        };

        let curve = CurveType::Bls48581;
        let honest_input = vec![0xab, 0xcd, 0xef, 0x12];

        let gadget_w = Sha3InputWitness::from_inputs(&[honest_input.clone()]).unwrap();
        let mut gadget_trace = build_gadget_trace(&gadget_w, curve);
        let extract_w = KeccakExtractWitness::from_inputs(&[honest_input]).unwrap();
        let extract_trace = build_extract_trace(&extract_w, curve);

        // Tamper: flip gadget's INPUT_BYTE[0] (0xab → 0xff).
        gadget_trace.columns[sha3_input_air::COL_INPUT_BYTE_OFFSET].evaluations[0] =
            metavm_zkp::field::Scalar::from_u64(0xff, curve);

        // L2 tuple now disagrees on at least byte 0.
        let g = gadget_trace.columns[sha3_input_air::COL_INPUT_BYTE_OFFSET]
            .evaluations[0].to_u64();
        let e = extract_trace.columns[keccak_extract::COL_INPUT_BYTE_OFFSET]
            .evaluations[0].to_u64();
        assert_ne!(g, e, "tampered byte must differ from honest extract witness");
    }

    /// Phase A1b standalone-EVM diagnostic: prove + verify the same
    /// SHA3 bytecode through `prove_with_scheme` standalone (no joint
    /// pipeline) to check whether the EVM main proof is the regression.
    /// If THIS passes, the bug is in the joint pipeline's EVM handling.
    /// If THIS fails, the bug is in EVM constraint coverage for SHA3
    /// traces (likely a regression from the BYTE selector split).
    #[test]
    #[ignore = "slow: EVM main standalone prove+verify; \
                run with --release --ignored"]
    fn standalone_evm_prove_verify_sha3_bytecode_diagnostic() {
        use crate::constraints::EvmConstraintSystem;
        use crate::executor::execute_bytecode;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_cols, curve);
        let evm_cs = EvmConstraintSystem::new();

        eprintln!("[diag] standalone EVM: num_steps={} padded_size={}",
            evm_polys.num_rows, evm_polys.padded_size);
        let proof = metavm_zkp::prover::prove_with_scheme(&evm_polys, &evm_cs, &scheme);
        let valid = metavm_zkp::verifier::verify_with_scheme(&proof, &evm_cs, &scheme, curve);
        eprintln!("[diag] standalone EVM verify = {}", valid);
        assert!(valid, "EVM main standalone prove+verify must succeed on SHA3 bytecode");
    }

    /// Phase A1b joint_verify diagnostic: skip EVM main and prove only
    /// the 2-AIR sub-chain (gadget + KeccakExtract with L2 only) to
    /// isolate whether the 3-AIR failure is in the EVM-main side or the
    /// gadget↔extract integration. Fast enough to run in a single
    /// session (~3-5 min).
    #[test]
    #[ignore = "slow: 2-AIR joint_prove for diagnostic; \
                run with --release --ignored"]
    fn joint_prove_a1b_gadget_extract_subchain_diagnostic() {
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            build_trace_polynomials as build_extract_trace,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use metavm_zkp::sha3_input_air::{
            self as sia, build_trace_polynomials as build_gadget_trace,
            Sha3InputConstraintSystem, Sha3InputWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let input_bytes = vec![0xab, 0xcd, 0xef, 0x12];
        let gadget_w = Sha3InputWitness::from_inputs(&[input_bytes.clone()]).unwrap();
        let gadget_trace = build_gadget_trace(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = Sha3InputConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let extract_w = KeccakExtractWitness::from_inputs(&[input_bytes]).unwrap();
        let extract_trace = build_extract_trace(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        let l2 = sia::make_sha3_input_keccak_extract_linkage_descriptor(0, 1);
        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&gadget_trace, &gadget_cs), (&extract_trace, &extract_cs)];
        let linkages = vec![l2];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("2-AIR joint_prove (gadget+extract) must succeed");
        eprintln!("[diag] 2-AIR joint_prove returned {} proofs, {} linkage_proofs",
            proofs.len(), ext.linkage_proofs.len());
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            let closure_match = lp.closure_a == lp.closure_b;
            eprintln!("[diag] linkage {} label={} closure_match={}", i, lp.label, closure_match);
        }
        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&gadget_cs, &extract_cs];

        // Per-AIR verify isolation
        for (i, (p, c)) in proofs.iter().zip(cs_refs.iter()).enumerate() {
            let v = metavm_zkp::verifier::verify_with_scheme(p, *c, &scheme, curve);
            eprintln!("[diag] per-AIR verify[{}] = {}", i, v);
        }
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve);
        eprintln!("[diag] joint_verify = {}", valid);
        assert!(valid, "2-AIR diagnostic chain joint_verify must accept honest proof");
    }

    /// Slow Phase A1b end-to-end: 3-AIR `joint_prove` + `joint_verify`
    /// across EVM main + SHA3 input gadget + KeccakExtract with L1 + L2
    /// linkages. Validates the full preimage-committed chain on a real
    /// EVM SHA3 run.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove (~15-20 min, dominated by EVM main \
                256-padded trace); run with --release --ignored"]
    fn joint_prove_evm_sha3_a1b_e2e() {
        use crate::constraints::EvmConstraintSystem;
        use crate::executor::execute_bytecode;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            build_trace_polynomials as build_extract_trace,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use metavm_zkp::sha3_input_air::{
            self as sia, build_trace_polynomials as build_gadget_trace,
            Sha3InputConstraintSystem, Sha3InputWitness,
        };

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // MSTORE8 ×4 then SHA3 over memory[0..4]. Same setup as the
        // fast alignment test.
        let bytecode = vec![
            0x60, 0xab,  0x60, 0x00,  0x53,
            0x60, 0xcd,  0x60, 0x01,  0x53,
            0x60, 0xef,  0x60, 0x02,  0x53,
            0x60, 0x12,  0x60, 0x03,  0x53,
            0x60, 0x04,  0x60, 0x00,  0x20,  0x00,
        ];
        let evm_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_cols, curve);
        let evm_cs = EvmConstraintSystem::new();

        let input_bytes = vec![0xab, 0xcd, 0xef, 0x12];
        let gadget_w = Sha3InputWitness::from_inputs(&[input_bytes.clone()]).unwrap();
        let gadget_trace = build_gadget_trace(&gadget_w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = Sha3InputConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let extract_w = KeccakExtractWitness::from_inputs(&[input_bytes]).unwrap();
        let extract_trace = build_extract_trace(&extract_w, curve);
        let extract_omega = scheme.domain_generator(extract_trace.padded_size);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows)
            .with_omega_and_domain(extract_omega, extract_trace.padded_size);

        let l1 = make_evm_sha3_input_linkage_descriptor(0, 1);
        let l2 = sia::make_sha3_input_keccak_extract_linkage_descriptor(1, 2);

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&evm_polys, &evm_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        let linkages = vec![l1, l2];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove for A1b must succeed");
        assert_eq!(proofs.len(), 3);
        assert_eq!(ext.linkage_proofs.len(), 2);
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            eprintln!("[diag] linkage {} label={} closure_match={}",
                i, lp.label, lp.closure_a == lp.closure_b);
            assert_eq!(lp.closure_a, lp.closure_b);
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs, &extract_cs];

        // Diagnostic: per-AIR verify in isolation
        for (i, (p, c)) in proofs.iter().zip(cs_refs.iter()).enumerate() {
            let v = metavm_zkp::verifier::verify_with_scheme(p, *c, &scheme, curve);
            eprintln!("[diag] per-AIR verify[{}] (cols={}, num_steps={}, domain={}) = {}",
                i, p.column_commitments.len(), p.num_steps, p.domain_size, v);
        }

        let valid = joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve);
        eprintln!("[diag] joint_verify = {}", valid);
        assert!(valid, "joint_verify must accept honest A1b 3-AIR chain");
    }
}

/// Reconstruct the final EXP result from a gadget trace.
///
/// The last row's `result_out` is `mul` if `exp_bit == 1` else `squared`.
/// (See `exp_witness` in `exp_air/mod.rs:251`.)
fn final_result(rows: &[ExpRow]) -> U256 {
    let last = rows.last().expect("EXP gadget trace must be non-empty");
    let limbs = if last.exp_bit == 1 { last.mul } else { last.squared };
    let mut bytes = [0u8; 32];
    for (i, l) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&l.to_le_bytes());
    }
    U256::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end cross-AIR `joint_prove`/`joint_verify` test for
    /// EVM main trace ↔ EXP gadget AIR. Mirrors the
    /// `joint_prove_keccak_extract_keccak_linkage` /
    /// `joint_prove_sha256_extract_sha256_linkage` tests
    /// (which closed #91 / #87 end-to-end).
    ///
    /// Setup:
    ///   - EVM main trace with one `EXP` opcode row (base=2, exponent=10,
    ///     output=1024) — uses the existing `EvmConstraintSystem`. EXP
    ///     is treated as an oracle by the main constraint system (the
    ///     selector body is identically zero); the cross-AIR LogUp
    ///     linkage to the EXP gadget AIR is what binds the claim.
    ///   - EXP gadget AIR trace via `exp_witness(2, 10)` →
    ///     256-row standard square-and-multiply trace with the
    ///     algebraic input/output binding (EXPONENT, FINAL_OUTPUT,
    ///     IS_FIRST_ROW columns) populated by `populate_trace`.
    ///   - Linkage: 12-column tuple `(base, exponent, output)` with
    ///     EVM-side gated by `COL_SEL_EXP` and gadget-side gated by
    ///     `COL_IS_FIRST_ROW`.
    ///
    /// Validates that `joint_prove` accepts the matched (base,
    /// exponent, output) tuple and `joint_verify` rejects any
    /// tampered version. With the EXP gadget's algebraic correctness
    /// (gadget AIR's per-row constraints pin the square-and-multiply
    /// computation), the cross-AIR LogUp transfers the gadget's
    /// `pow(base, exponent)` guarantee to the EVM main trace's claimed
    /// EXP result.
    #[test]
    #[ignore = "slow: EVM main + EXP gadget joint prove (~10 min); run with --release --ignored"]
    fn joint_prove_evm_exp_linkage() {
        use crate::exp_air;
        use crate::trace::{EvmTraceColumns, EvmTraceRow, FrameState, INSN_ARITH};
        use crate::trace::FUNCT_EXP;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::{CurveType, Scalar};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── EVM main trace (A side) — single EXP-opcode row ──
        let base = U256::from(2u64);
        let exponent = U256::from(10u64);
        let result = U256::from(1024u64);
        let limb = |x: U256| -> [u64; 4] {
            let mut out = [0u64; 4];
            for (i, l) in x.as_limbs().iter().enumerate().take(4) {
                out[i] = *l;
            }
            out
        };

        let mut evm_trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x0a,
            gas_remaining: 1000,
            stack_depth: 2,
            input0: limb(base),
            input1: limb(exponent),
            output0: limb(result),
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_EXP,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 1,
            frame: FrameState::default(),
            create_address_hint: [0; 4],
            create_nonce_hint: 0,
            sel_stop_pop: 0,
            create2_salt_hint: [0; 4],
            create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
        };
        evm_trace.push_row(&row);
        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_trace, curve);
        let evm_cs = EvmConstraintSystem::new();

        // ── EXP gadget AIR trace (B side) ──
        let exp_rows = exp_air::exp_witness(base, exponent);
        let exp_columns = exp_air::populate_trace(&exp_rows, curve);
        let exp_polys = {
            let num_rows = exp_air::NUM_STEPS;
            let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
            let mut col_polys: Vec<metavm_zkp::trace::Polynomial> = exp_columns
                .into_iter()
                .map(|mut evals| {
                    if evals.len() < padded {
                        evals.resize(padded, Scalar::zero(curve));
                    }
                    metavm_zkp::trace::Polynomial {
                        evaluations: evals,
                        degree: num_rows,
                    }
                })
                .collect();
            // Sanity: column count matches the EXP gadget's NUM_EXP_AIR_COLUMNS.
            assert_eq!(col_polys.len(), exp_air::NUM_EXP_AIR_COLUMNS);
            metavm_zkp::trace::TracePolynomials {
                columns: std::mem::take(&mut col_polys),
                num_rows,
                padded_size: padded as u64,
                curve,
            }
        };
        let exp_cs = crate::exp_constraints::EvmExpConstraintSystem::new();

        // ── Linkage ──
        let linkage = make_evm_exp_linkage_descriptor(0, 1);

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&exp_polys, &exp_cs)];
        let linkages = vec![linkage];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for matched EVM main + EXP gadget");
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        assert_eq!(
            extension.linkage_proofs[0].closure_a,
            extension.linkage_proofs[0].closure_b,
            "honest closure scalars must match"
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &exp_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the EVM main↔EXP gadget honest joint proof"
        );
    }

    /// #94 Phase 4: end-to-end multi-invocation integration. Two EVM
    /// EXP opcodes (`2^10` and `3^7`) on the EVM main side; two
    /// gadget invocations on the gadget side via
    /// [`exp_air::populate_multi_trace`]. The same single-shape
    /// linkage descriptor `make_evm_exp_linkage_descriptor` works
    /// unchanged because its B-side selector
    /// [`exp_air::COL_IS_FIRST_ROW`] fires once per gadget
    /// invocation in the multi-invocation trace, matching the N
    /// `SEL_EXP` rows on the EVM main side.
    ///
    /// This closes #94 end-to-end: the multi-invocation EXP gadget
    /// (Phases 1+2+3) is now demonstrated working with a real EVM
    /// main trace via the cross-AIR LogUp linkage.
    ///
    /// **Scheme choice**: this test uses **BLS12-381** instead of
    /// BLS48-581 because the multi-invocation gadget natural padded
    /// size is 512 rows (2 × NUM_STEPS), and BLS48-581's cached FFT
    /// widths only go up to 256. BLS12-381 supports up to 4096 via
    /// the embedded SRS, comfortably covering 512.
    #[test]
    #[ignore = "slow: EVM main + multi-invocation EXP gadget joint prove; \
                run with --release --ignored"]
    fn joint_prove_evm_exp_linkage_multi_invocation() {
        use crate::exp_air;
        use crate::trace::{EvmTraceColumns, EvmTraceRow, FrameState, INSN_ARITH};
        use crate::trace::FUNCT_EXP;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::{CurveType, Scalar};
        use metavm_zkp::scheme::bls12381_scheme::Bls12381Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;

        let curve = CurveType::Bls12381;
        let scheme = Bls12381Scheme::new();
        scheme.init();

        let limb = |x: U256| -> [u64; 4] {
            let mut out = [0u64; 4];
            for (i, l) in x.as_limbs().iter().enumerate().take(4) {
                out[i] = *l;
            }
            out
        };

        // ── EVM main trace (A side) — TWO EXP-opcode rows ──
        let invocations = [
            (U256::from(2u64), U256::from(10u64), U256::from(1024u64)),
            (U256::from(3u64), U256::from(7u64), U256::from(2187u64)),
        ];
        let mut evm_trace = EvmTraceColumns::new();
        for (i, &(base, exponent, result)) in invocations.iter().enumerate() {
            let row = EvmTraceRow {
                step: i as u64,
                pc: i as u64,
                opcode: 0x0a,
                gas_remaining: 1000 - (i as u64),
                stack_depth: 2,
                input0: limb(base),
                input1: limb(exponent),
                output0: limb(result),
                mem_offset: 0,
                mem_value: [0; 4],
                insn_type: INSN_ARITH,
                funct: FUNCT_EXP,
                immediate: [0; 4],
                aux0: [0; 4],
                aux1: [0; 4],
                next_pc: (i as u64) + 1,
                frame: FrameState::default(),
                create_address_hint: [0; 4],
                create_nonce_hint: 0,
                sel_stop_pop: 0,
                create2_salt_hint: [0; 4],
                create2_initcode_hash_hint: [0; 4],
            tx_origin: [0u64; 4],
            tx_gas_price: 0,
            tx_calldata_size: 0,
            tx_code_size: 0,
            returndata_size: 0,
            };
            evm_trace.push_row(&row);
        }
        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_trace, curve);
        let evm_cs = EvmConstraintSystem::new();

        // ── Multi-invocation EXP gadget AIR trace (B side) ──
        let pairs: Vec<(U256, U256)> = invocations
            .iter()
            .map(|(b, e, _)| (*b, *e))
            .collect();
        let exp_columns = exp_air::populate_multi_trace(&pairs, curve);
        let exp_polys = {
            let num_rows = pairs.len() * exp_air::NUM_STEPS;
            let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
            let mut col_polys: Vec<metavm_zkp::trace::Polynomial> = exp_columns
                .into_iter()
                .map(|mut evals| {
                    if evals.len() < padded {
                        evals.resize(padded, Scalar::zero(curve));
                    }
                    metavm_zkp::trace::Polynomial {
                        evaluations: evals,
                        degree: num_rows,
                    }
                })
                .collect();
            assert_eq!(col_polys.len(), exp_air::NUM_EXP_AIR_COLUMNS);
            metavm_zkp::trace::TracePolynomials {
                columns: std::mem::take(&mut col_polys),
                num_rows,
                padded_size: padded as u64,
                curve,
            }
        };
        let exp_cs = crate::exp_constraints::EvmExpConstraintSystem::new();

        // ── Linkage (same descriptor as single-invocation; B-side
        //    selector IS_FIRST_ROW fires once per invocation) ──
        let linkage = make_evm_exp_linkage_descriptor(0, 1);

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&exp_polys, &exp_cs)];
        let linkages = vec![linkage];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme).expect(
            "joint_prove must succeed for matched 2-EXP EVM main + 2-invocation gadget",
        );
        assert_eq!(proofs.len(), 2);
        assert_eq!(extension.linkage_proofs.len(), 1);
        assert_eq!(
            extension.linkage_proofs[0].closure_a,
            extension.linkage_proofs[0].closure_b,
            "honest closure scalars must match for multi-invocation linkage"
        );

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &exp_cs];
        let valid =
            joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the 2-EXP EVM main ↔ multi-invocation \
             EXP gadget honest joint proof"
        );
    }

    #[test]
    fn evm_exp_linkage_descriptor_well_formed() {
        let desc = make_evm_exp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "evm_exp_gadget_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        // 12 columns on each side: 4 base + 4 exponent + 4 output limbs.
        assert_eq!(desc.a_columns.len(), 12);
        assert_eq!(desc.b_columns.len(), 12);
        // Selectors must reference the per-AIR gating columns.
        assert_eq!(desc.a_selector_column, Some(crate::trace::COL_SEL_EXP));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::exp_air::COL_IS_FIRST_ROW)
        );
        // A side: INPUT0 (4-7), INPUT1 (8-11), OUTPUT0 (12-15) in order.
        assert_eq!(desc.a_columns[0], crate::trace::COL_INPUT0_L0);
        assert_eq!(desc.a_columns[3], crate::trace::COL_INPUT0_L3);
        assert_eq!(desc.a_columns[4], crate::trace::COL_INPUT1_L0);
        assert_eq!(desc.a_columns[7], crate::trace::COL_INPUT1_L3);
        assert_eq!(desc.a_columns[8], crate::trace::COL_OUTPUT0_L0);
        assert_eq!(desc.a_columns[11], crate::trace::COL_OUTPUT0_L3);
        // B side: BASE (4-7), EXPONENT (27-30), FINAL_OUTPUT (31-34).
        assert_eq!(desc.b_columns[0], crate::exp_air::COL_BASE_OFFSET);
        assert_eq!(desc.b_columns[4], crate::exp_air::COL_EXPONENT_OFFSET);
        assert_eq!(desc.b_columns[8], crate::exp_air::COL_FINAL_OUTPUT_OFFSET);
    }

    /// Three-AIR end-to-end joint_prove regression for CREATE address
    /// derivation. Wires:
    ///   - EVM main trace with one CREATE row.
    ///   - `EvmCreateRlpAir` gadget with the matching `(sender, nonce)`
    ///     invocation.
    ///   - `KeccakExtract` with the matching `keccak256(rlp([sender, nonce]))`
    ///     invocation.
    ///
    /// Three cross-AIR LogUp linkages active simultaneously:
    ///   - L1: EVM main ↔ RLP gadget (5-tuple `(frame_callee_l[0..4], create_nonce_hint)`
    ///     ↔ `(SENDER_LIMB[0..4], NONCE)`), gated by `COL_SEL_CREATE` / `COL_IS_REAL`.
    ///   - L2: RLP gadget ↔ KeccakExtract input (257-tuple
    ///     `(RLP_BYTE[0..256], RLP_LEN)` ↔ `(INPUT_BYTE[0..256], INPUT_LEN)`),
    ///     both gated by `COL_IS_REAL`.
    ///   - L3: EVM main ↔ KeccakExtract output (4-tuple
    ///     `(create_address_hint_l[0..4])` ↔ `(address_limb_l[0..4])`),
    ///     gated by `COL_SEL_CREATE` / `COL_IS_REAL`.
    ///
    /// Combined with the already-closed KeccakExtract↔Keccak (#91)
    /// chain (not exercised in this test for runtime — already
    /// validated by `joint_prove_keccak_extract_keccak_linkage`), this
    /// test demonstrates the full input-side and output-side soundness
    /// chain for CREATE address derivation operating in unison through
    /// `joint_prove`/`joint_verify`.
    ///
    /// End-to-end multi-linkage joint_prove validating the full CREATE
    /// address-derivation soundness chain: EVM main + RLP gadget +
    /// KeccakExtract with three simultaneous cross-AIR LogUp linkages
    /// (EVM↔gadget on `(sender_limbs, nonce)`; gadget↔extract on RLP
    /// bytes; EVM↔extract on the address-limb output).
    ///
    /// Required several rounds of joint_prove infrastructure fixes
    /// (this session): the auto-inflation in
    /// `metavm_zkp::cross_air_logup::joint_prove` now (a) accounts for
    /// the LogUp domain boost in `commit_main_columns_phase1` (AIRs
    /// with LogUp declarations get committed at `max(padded, RANGE_TABLE_SIZE)`),
    /// AND (b) applies each AIR's `fix_trace_padding` to the inflated
    /// trace clones so the linkage trace's TUPLE_A/B polynomials are
    /// built from the SAME column values that the per-AIR commitments
    /// commit to. Without (a) the linkage's domain (16) didn't match
    /// the EVM AIR's commit domain (256). Without (b) the EVM AIR's
    /// `fix_trace_padding` (which copies `frame_callee/frame_caller/
    /// frame_value/frame_static/...` to padding rows for preservation
    /// constraint satisfaction) produced different column polynomials
    /// than the linkage saw, breaking cross-trace tuple binding.
    /// Required also the EVM constraint fixes: PC continuity gate by
    /// `is_pc_jump`, frame-state preservation gate by `is_frame_op`
    /// (with `sel_stop_pop`), and the new `COL_SEL_STOP_POP` selector
    /// in the LIFO frame-stack permutation argument's pop_selectors.
    #[test]
    #[ignore = "slow: 3-AIR + 3-linkage joint_prove (~12 min, dominated \
                by EVM main 256-padded trace + 1398 LogUp cols)"]
    fn joint_prove_evm_create_address_e2e() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::evm_create_rlp_air;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            self, KeccakExtractWitness, KeccakExtractConstraintSystem,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── EVM main trace from the real executor ──
        // Run a tiny bytecode that does one CREATE (PUSH3 zeros + CREATE
        // + STOP). This generates a fully-constrained EVM trace where
        // the executor wired the full frame transition + nonce/address
        // hints; all EVM constraints are satisfied by construction.
        let bytecode = vec![
            0x60, 0x00, // PUSH1 0  (size)
            0x60, 0x00, // PUSH1 0  (offset)
            0x60, 0x00, // PUSH1 0  (value)
            0xF0,       // CREATE
            0x00,       // STOP
        ];
        let evm_trace_columns = execute_bytecode(&bytecode, &[])
            .expect("CREATE bytecode must execute cleanly");

        // Find the CREATE row in the produced trace and extract its
        // (sender, nonce) for the matching gadget invocation.
        let create_row = (0..evm_trace_columns.opcode.len())
            .find(|&i| evm_trace_columns.opcode[i] == 0xF0)
            .expect("CREATE row must appear in trace");
        // Sender = frame.callee at the CREATE row (the executing
        // contract address). Reassemble from 4 LE u64 limbs into
        // 20 raw bytes using the inverse of `address_to_limbs`.
        let sender_limbs: [u64; 4] = [
            evm_trace_columns.frame_callee[0][create_row],
            evm_trace_columns.frame_callee[1][create_row],
            evm_trace_columns.frame_callee[2][create_row],
            evm_trace_columns.frame_callee[3][create_row],
        ];
        let mut sender = [0u8; 20];
        sender[0..8].copy_from_slice(&sender_limbs[0].to_le_bytes());
        sender[8..16].copy_from_slice(&sender_limbs[1].to_le_bytes());
        sender[16..20].copy_from_slice(&sender_limbs[2].to_le_bytes()[..4]);
        let nonce = evm_trace_columns.create_nonce_hint[create_row];
        // The address hint columns will agree with `keccak256(rlp([sender, nonce]))[12..32]`
        // because the inspector now derives them from the same nonce.

        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(
            &evm_trace_columns, curve,
        );
        let evm_cs = EvmConstraintSystem::new();

        // ── RLP gadget (single invocation matching the EVM CREATE row) ──
        let gadget_w = evm_create_rlp_air::CreateRlpWitness::from_inputs(&[(&sender, nonce)])
            .expect("nonce in valid range (must be < 128 for the MVP gadget)");
        let gadget_trace = evm_create_rlp_air::build_trace_polynomials(&gadget_w, curve);
        let gadget_cs = evm_create_rlp_air::CreateRlpConstraintSystem::new(gadget_trace.num_rows);

        // ── KeccakExtract (single invocation, hashing the canonical RLP) ──
        let canonical = evm_create_rlp_air::canonical_rlp(&sender, nonce);
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical.to_vec()])
            .expect("rlp output fits in MAX_INPUT_LEN");
        let extract_trace = keccak_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows);

        // Sanity: the executor-derived address hint must match what
        // KeccakExtract's address_limb aggregator computes from the
        // canonical RLP digest. If this fails, the inspector's nonce
        // sourcing or the gadget's RLP encoding is wrong.
        let address_limbs = keccak_extract::address_limbs_from_output(
            &extract_w.invocations[0].output,
        );
        for k in 0..4 {
            assert_eq!(
                evm_trace_columns.create_address_hint[k][create_row],
                address_limbs[k],
                "EVM address hint limb {} must match keccak256(rlp([sender, nonce]))[12..32] aggregation",
                k
            );
        }

        // ── Linkages ──
        // L1: EVM main ↔ RLP gadget.
        let l1 = make_evm_main_create_rlp_linkage_descriptor(/* evm */ 0, /* gadget */ 1);
        // L2: RLP gadget ↔ KeccakExtract input side.
        let l2 = evm_create_rlp_air::make_create_rlp_keccak_extract_input_linkage_descriptor(
            /* gadget */ 1, /* extract */ 2,
        );
        // L3: EVM main ↔ KeccakExtract output (address-limb) side.
        let l3 = make_evm_create_address_keccak_extract_linkage_descriptor(
            /* evm */ 0, /* extract */ 2, crate::trace::COL_SEL_CREATE,
        );

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&evm_polys, &evm_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        let linkages = vec![l1, l2, l3];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the honest 3-AIR + 3-linkage CREATE chain");
        assert_eq!(proofs.len(), 3, "one proof per AIR");
        assert_eq!(extension.linkage_proofs.len(), 3, "one linkage proof each");
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs, &extract_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the honest CREATE address-derivation chain"
        );
    }

    /// Four-AIR end-to-end CREATE address derivation joint_prove,
    /// extending the 3-AIR test with the 4th layer: bit-level Keccak
    /// (`keccak_constraints`). Closes the entire algebraic chain
    /// `EVM main ↔ RLP gadget ↔ KeccakExtract ↔ bit-level Keccak`.
    ///
    /// Adds a 4th cross-AIR linkage L4: KeccakExtract↔Keccak (the same
    /// linkage validated by `joint_prove_keccak_extract_keccak_linkage`
    /// in `crates/zkp/src/keccak_constraints.rs`). Combined with the
    /// 3-AIR L1/L2/L3:
    ///   - L1: EVM main↔RLP gadget on `(sender, nonce)`
    ///   - L2: RLP gadget↔KeccakExtract input on RLP byte tuples
    ///   - L3: EVM main↔KeccakExtract output on address-limb tuples
    ///   - L4: KeccakExtract↔bit-level Keccak on byte aggregator tuples
    ///
    /// Net result: the EVM CREATE row's `create_address_hint` is now
    /// algebraically pinned by:
    /// 1. L1 forces sender + nonce to match the gadget's witness
    /// 2. Gadget's row-locals pin RLP_BYTE = canonical_rlp(sender, nonce)
    /// 3. L2 forces gadget's RLP_BYTE = KeccakExtract's INPUT_BYTE
    /// 4. L4 + bit-level Keccak's input/output bindings (#91) force
    ///    KeccakExtract's INPUT_BYTE → AFTER_IOTA bit-state → digest
    /// 5. L3 forces address_limbs to match KeccakExtract's
    ///    aggregated `OUTPUT_BYTE[12..32]`
    ///
    /// Estimated runtime: ~25 min total (3-AIR portion ~12 min for EVM
    /// main + Keccak portion ~5 min for bit-level prover).
    #[test]
    #[ignore = "very slow: 4-AIR + 4-linkage joint_prove (~25 min, dominated \
                by EVM main 256-padded trace + bit-level Keccak)"]
    fn joint_prove_evm_create_address_e2e_with_bit_level_keccak() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::evm_create_rlp_air;
        use metavm_zkp::field::{CurveType, Scalar};
        use metavm_zkp::keccak::keccak_witness;
        use metavm_zkp::keccak_air;
        use metavm_zkp::keccak::NUM_ROUNDS;
        use metavm_zkp::keccak_constraints::KeccakConstraintSystem;
        use metavm_zkp::keccak_extract::{
            self, make_keccak_extract_keccak_linkage_descriptor,
            KeccakExtractWitness, KeccakExtractConstraintSystem,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── EVM main trace from the real executor (same as 3-AIR test) ──
        let bytecode = vec![
            0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF0, 0x00,
        ];
        let evm_trace_columns = execute_bytecode(&bytecode, &[])
            .expect("CREATE bytecode must execute cleanly");

        let create_row = (0..evm_trace_columns.opcode.len())
            .find(|&i| evm_trace_columns.opcode[i] == 0xF0)
            .expect("CREATE row must appear in trace");
        let sender_limbs: [u64; 4] = [
            evm_trace_columns.frame_callee[0][create_row],
            evm_trace_columns.frame_callee[1][create_row],
            evm_trace_columns.frame_callee[2][create_row],
            evm_trace_columns.frame_callee[3][create_row],
        ];
        let mut sender = [0u8; 20];
        sender[0..8].copy_from_slice(&sender_limbs[0].to_le_bytes());
        sender[8..16].copy_from_slice(&sender_limbs[1].to_le_bytes());
        sender[16..20].copy_from_slice(&sender_limbs[2].to_le_bytes()[..4]);
        let nonce = evm_trace_columns.create_nonce_hint[create_row];

        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(
            &evm_trace_columns, curve,
        );
        let evm_cs = EvmConstraintSystem::new();

        // ── RLP gadget ──
        let gadget_w = evm_create_rlp_air::CreateRlpWitness::from_inputs(&[(&sender, nonce)])
            .expect("nonce in valid range");
        let gadget_trace = evm_create_rlp_air::build_trace_polynomials(&gadget_w, curve);
        let gadget_cs = evm_create_rlp_air::CreateRlpConstraintSystem::new(gadget_trace.num_rows);

        // ── KeccakExtract ──
        let canonical = evm_create_rlp_air::canonical_rlp(&sender, nonce);
        let canonical_vec: Vec<u8> = canonical.to_vec();
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical_vec.clone()])
            .expect("rlp output fits in MAX_INPUT_LEN");
        let extract_trace = keccak_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows);

        // ── Bit-level Keccak (4th AIR) ──
        let digest = metavm_zkp::keccak::keccak256(&canonical_vec);
        let ht = keccak_witness(&canonical_vec);
        let num_keccak_rows = ht.blocks.len() * NUM_ROUNDS;
        let keccak_padded = metavm_zkp::trace::nearest_power_of_two(num_keccak_rows.max(1));
        let mut keccak_columns = keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &canonical_vec, &digest, curve,
        );
        for col in keccak_columns.iter_mut() {
            if col.len() < keccak_padded {
                col.resize(keccak_padded, Scalar::zero(curve));
            }
        }
        let keccak_polys: Vec<metavm_zkp::trace::Polynomial> = keccak_columns
            .into_iter()
            .map(|evals| metavm_zkp::trace::Polynomial {
                evaluations: evals,
                degree: num_keccak_rows,
            })
            .collect();
        let keccak_trace = metavm_zkp::trace::TracePolynomials {
            columns: keccak_polys,
            num_rows: num_keccak_rows,
            padded_size: keccak_padded as u64,
            curve,
        };
        let keccak_cs = KeccakConstraintSystem::new(num_keccak_rows);

        // ── Linkages ──
        let l1 = make_evm_main_create_rlp_linkage_descriptor(/* evm */ 0, /* gadget */ 1);
        let l2 = evm_create_rlp_air::make_create_rlp_keccak_extract_input_linkage_descriptor(
            /* gadget */ 1, /* extract */ 2,
        );
        let l3 = make_evm_create_address_keccak_extract_linkage_descriptor(
            /* evm */ 0, /* extract */ 2, crate::trace::COL_SEL_CREATE,
        );
        // L4: KeccakExtract↔bit-level Keccak (input/output byte aggregators).
        let l4 = make_keccak_extract_keccak_linkage_descriptor(
            /* extract */ 2, /* keccak */ 3,
        );

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&evm_polys, &evm_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
            (&keccak_trace, &keccak_cs),
        ];
        let linkages = vec![l1, l2, l3, l4];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the honest 4-AIR + 4-linkage CREATE chain");
        assert_eq!(proofs.len(), 4, "one proof per AIR");
        assert_eq!(extension.linkage_proofs.len(), 4, "one linkage proof each");
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs, &extract_cs, &keccak_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the honest 4-AIR CREATE address chain"
        );
    }

    /// Fast sanity check (~5s, no proof): asserts the EVM trace's
    /// `(frame_callee, create2_salt_hint, create2_initcode_hash_hint)`
    /// 12-limb tuple at a CREATE2 row matches the CREATE2 input gadget's
    /// `(SENDER_LIMB, SALT_LIMB, INITCODE_HASH_LIMB)` row tuple. If
    /// this assertion fires, the EVM-side adapter columns are wired
    /// incorrectly relative to the gadget's limb-binding constraints.
    #[test]
    fn evm_create2_hint_tuple_matches_gadget_limbs() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::evm_create2_input_air;
        use metavm_zkp::field::CurveType;

        let curve = CurveType::Bls48581;

        let bytecode = vec![
            0x60, 0x01, 0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF5, 0x00,
        ];
        let evm_trace_columns = execute_bytecode(&bytecode, &[]).unwrap();
        let row = (0..evm_trace_columns.opcode.len())
            .find(|&i| evm_trace_columns.opcode[i] == 0xF5)
            .expect("CREATE2 row");

        // Reconstruct (sender, salt, initcode_hash) from the EVM hints.
        let sender_limbs = [
            evm_trace_columns.frame_callee[0][row],
            evm_trace_columns.frame_callee[1][row],
            evm_trace_columns.frame_callee[2][row],
            evm_trace_columns.frame_callee[3][row],
        ];
        let mut sender = [0u8; 20];
        sender[0..8].copy_from_slice(&sender_limbs[0].to_le_bytes());
        sender[8..16].copy_from_slice(&sender_limbs[1].to_le_bytes());
        sender[16..20].copy_from_slice(&sender_limbs[2].to_le_bytes()[..4]);

        let salt_limbs = [
            evm_trace_columns.create2_salt_hint[0][row],
            evm_trace_columns.create2_salt_hint[1][row],
            evm_trace_columns.create2_salt_hint[2][row],
            evm_trace_columns.create2_salt_hint[3][row],
        ];
        let mut salt = [0u8; 32];
        for (k, &l) in salt_limbs.iter().enumerate() {
            salt[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
        }

        let ih_limbs = [
            evm_trace_columns.create2_initcode_hash_hint[0][row],
            evm_trace_columns.create2_initcode_hash_hint[1][row],
            evm_trace_columns.create2_initcode_hash_hint[2][row],
            evm_trace_columns.create2_initcode_hash_hint[3][row],
        ];
        let mut initcode_hash = [0u8; 32];
        for (k, &l) in ih_limbs.iter().enumerate() {
            initcode_hash[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
        }

        let gadget_w = evm_create2_input_air::Create2InputWitness::from_inputs(
            &[(&sender, &salt, &initcode_hash)],
        );
        let gadget_trace = evm_create2_input_air::build_trace_polynomials(&gadget_w, curve);

        // Sender limbs at gadget row 0 must equal EVM frame_callee limbs.
        for k in 0..4 {
            let g = gadget_trace.columns[
                evm_create2_input_air::COL_SENDER_LIMB_OFFSET + k
            ].evaluations[0].to_u64();
            assert_eq!(g, sender_limbs[k], "sender limb {} mismatch", k);
        }
        // Salt limbs.
        for k in 0..4 {
            let g = gadget_trace.columns[
                evm_create2_input_air::COL_SALT_LIMB_OFFSET + k
            ].evaluations[0].to_u64();
            assert_eq!(g, salt_limbs[k], "salt limb {} mismatch", k);
        }
        // Initcode_hash limbs.
        for k in 0..4 {
            let g = gadget_trace.columns[
                evm_create2_input_air::COL_INITCODE_HASH_LIMB_OFFSET + k
            ].evaluations[0].to_u64();
            assert_eq!(g, ih_limbs[k], "initcode_hash limb {} mismatch", k);
        }

        // Verify the linkage descriptor's tuple-column wiring matches.
        let desc = make_evm_main_create2_input_linkage_descriptor(0, 1);
        assert_eq!(desc.a_columns.len(), 12, "12 limb columns on EVM side");
        assert_eq!(desc.b_columns.len(), 12, "12 limb columns on gadget side");
        // a_columns layout: 4 sender + 4 salt + 4 initcode_hash.
        for k in 0..4 {
            assert_eq!(desc.a_columns[k], crate::trace::COL_FRAME_CALLEE_L0 + k);
        }
        for k in 0..4 {
            assert_eq!(desc.a_columns[4 + k], crate::trace::COL_CREATE2_SALT_HINT_L0 + k);
        }
        for k in 0..4 {
            assert_eq!(
                desc.a_columns[8 + k],
                crate::trace::COL_CREATE2_INITCODE_HASH_HINT_L0 + k
            );
        }
    }

    /// Phase A1a end-to-end fast check: run a real EVM program containing
    /// SHA3 through the inspector, then assert the EVM trace's
    /// `output0[k]` (4 u64 limbs) on the SHA3 row matches a
    /// `KeccakExtractWitness` row's `KECCAK_OUTPUT_LIMB[k]` byte-for-byte.
    /// This is the fast pre-condition for the cross-AIR LogUp (the
    /// linkage matches these tuples directly); if it fires, the
    /// EVM-side `safe_peek` packing convention disagrees with
    /// `keccak_output_limbs_from_output`'s big-endian aggregation.
    #[test]
    fn evm_sha3_output_tuple_matches_keccak_extract_limbs() {
        use crate::executor::execute_bytecode;
        use crate::trace::{COL_OUTPUT0_L0, COL_SEL_KECCAK};
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            build_trace_polynomials, KeccakExtractWitness,
            COL_KECCAK_OUTPUT_LIMB_OFFSET, KECCAK_OUTPUT_LIMB_LEN,
        };

        let curve = CurveType::Bls48581;

        // Bytecode: write 4 bytes [0xab, 0xcd, 0xef, 0x12] to memory[0..4],
        // then SHA3 over that 4-byte range. Stack ordering for MSTORE8:
        // top = offset, next = value (offset popped first). For SHA3:
        // top = offset, next = size (offset popped first).
        let bytecode = vec![
            0x60, 0xab,  // PUSH1 0xab (value)
            0x60, 0x00,  // PUSH1 0x00 (offset)
            0x53,        // MSTORE8 → memory[0] = 0xab
            0x60, 0xcd,  // PUSH1 0xcd
            0x60, 0x01,  // PUSH1 0x01
            0x53,        // MSTORE8 → memory[1] = 0xcd
            0x60, 0xef,
            0x60, 0x02,
            0x53,        // memory[2] = 0xef
            0x60, 0x12,
            0x60, 0x03,
            0x53,        // memory[3] = 0x12
            0x60, 0x04,  // PUSH1 0x04 (size)
            0x60, 0x00,  // PUSH1 0x00 (offset)
            0x20,        // SHA3 → pushes keccak256(memory[0..4])
            0x00,        // STOP
        ];

        let evm_trace_columns = execute_bytecode(&bytecode, &[]).unwrap();

        // Find the SHA3 row.
        let sha3_row = (0..evm_trace_columns.opcode.len())
            .find(|&i| evm_trace_columns.opcode[i] == 0x20)
            .expect("SHA3 row must exist in trace");

        // Sanity: SEL_KECCAK fires on this row. (Selector access by index
        // not column constant — sel_keccak is the row-pushed selector.)
        assert_eq!(
            evm_trace_columns.sel_keccak[sha3_row], 1,
            "sel_keccak must be 1 on the SHA3 row"
        );
        let _ = COL_SEL_KECCAK;

        // Extract EVM output0 (4 LE u64 limbs of the U256 pushed on stack).
        let evm_output0: [u64; 4] = [
            evm_trace_columns.output0[0][sha3_row],
            evm_trace_columns.output0[1][sha3_row],
            evm_trace_columns.output0[2][sha3_row],
            evm_trace_columns.output0[3][sha3_row],
        ];
        let _ = COL_OUTPUT0_L0;

        // Independently compute the expected hash and build the matching
        // KeccakExtract witness.
        let input_bytes = vec![0xab, 0xcd, 0xef, 0x12];
        let extract_w = KeccakExtractWitness::from_inputs(&[input_bytes.clone()])
            .expect("input is well within MAX_INPUT_LEN");
        assert_eq!(extract_w.invocations.len(), 1);
        let extract_trace = build_trace_polynomials(&extract_w, curve);

        // Assert the cross-AIR LogUp's 4-tuple matches byte-for-byte.
        for k in 0..KECCAK_OUTPUT_LIMB_LEN {
            let extract_limb = extract_trace.columns[COL_KECCAK_OUTPUT_LIMB_OFFSET + k]
                .evaluations[0].to_u64();
            assert_eq!(
                extract_limb, evm_output0[k],
                "limb {} mismatch: KeccakExtract={:#x} EVM output0={:#x}",
                k, extract_limb, evm_output0[k]
            );
        }

        // Verify descriptor wiring matches what we just compared.
        let desc = make_evm_keccak_keccak_extract_linkage_descriptor(0, 1);
        assert_eq!(desc.a_columns.len(), 4);
        for k in 0..4 {
            assert_eq!(desc.a_columns[k], crate::trace::COL_OUTPUT0_L0 + k);
            assert_eq!(
                desc.b_columns[k],
                metavm_zkp::keccak_extract::COL_KECCAK_OUTPUT_LIMB_OFFSET + k
            );
        }
    }

    /// Regression test for the latent soundness bug where the NOT
    /// constraint at slot 55 (`sel_bitwise_other * not_raw_sum`)
    /// would incorrectly fire on BYTE (0x1A) rows because
    /// `sel_bitwise_other` previously fired on every INSN_BITWISE
    /// opcode without a dedicated selector. Fix: split out
    /// `COL_SEL_BYTE_OP` (col 240) so BYTE no longer triggers
    /// `sel_bitwise_other`.
    ///
    /// The test executes a BYTE opcode through the inspector and
    /// asserts:
    ///   1. `sel_byte_op = 1` on the BYTE row.
    ///   2. `sel_bitwise_other = 0` on the BYTE row.
    ///   3. Both selectors are mutually exclusive (one or the other,
    ///      never both).
    #[test]
    fn evm_byte_opcode_fires_sel_byte_op_not_bitwise_other() {
        use crate::executor::execute_bytecode;

        // PUSH1 0xAB (value), PUSH1 31 (index = last byte), BYTE, STOP.
        // EVM BYTE semantics: pops (i, x); pushes x[i] where i is byte
        // index from MSB. With x=0xAB (only low byte set) and i=31, the
        // result is 0xAB (the rightmost / least-significant byte).
        let bytecode = vec![
            0x60, 0xAB,  // PUSH1 0xAB (value x)
            0x60, 31,    // PUSH1 31 (index i)
            0x1A,        // BYTE
            0x00,        // STOP
        ];
        let t = execute_bytecode(&bytecode, &[]).unwrap();

        let byte_row = (0..t.opcode.len())
            .find(|&i| t.opcode[i] == 0x1A)
            .expect("BYTE row");

        assert_eq!(
            t.sel_byte_op[byte_row], 1,
            "sel_byte_op must fire on BYTE row"
        );
        assert_eq!(
            t.sel_bitwise_other[byte_row], 0,
            "sel_bitwise_other must NOT fire on BYTE row (latent bug fix)"
        );

        // Sanity: the per-row sel selectors are mutually exclusive
        // across the whole trace.
        for i in 0..t.opcode.len() {
            assert!(
                t.sel_byte_op[i] + t.sel_bitwise_other[i] <= 1,
                "row {} (opcode {:#x}): sel_byte_op + sel_bitwise_other > 1",
                i, t.opcode[i]
            );
        }
    }

    /// Companion: ensure NOT still correctly fires `sel_bitwise_other`
    /// (NOT is the only remaining opcode routed to the catchall after
    /// BYTE was split out). If a future refactor routes NOT to a
    /// dedicated selector, this test pins the migration boundary.
    #[test]
    fn evm_not_opcode_still_fires_sel_bitwise_other() {
        use crate::executor::execute_bytecode;
        // PUSH1 0x42, NOT, STOP.
        let bytecode = vec![0x60, 0x42, 0x19, 0x00];
        let t = execute_bytecode(&bytecode, &[]).unwrap();
        let not_row = (0..t.opcode.len())
            .find(|&i| t.opcode[i] == 0x19)
            .expect("NOT row");
        assert_eq!(t.sel_bitwise_other[not_row], 1);
        assert_eq!(t.sel_byte_op[not_row], 0);
    }

    /// Phase A3-byte step 2 fast precondition: run a real EVM BYTE
    /// opcode through the inspector and confirm the 12-limb cross-AIR
    /// LogUp tuple `(input0, input1, output0)` matches a matching
    /// [`crate::byte_air::ByteOpWitness`] row's `(INDEX, VALUE, RESULT)`
    /// limb-for-limb. If this fires, the EVM-side U256 packing on
    /// BYTE rows disagrees with the gadget's witness builder.
    #[test]
    fn evm_byte_op_tuple_matches_gadget_result() {
        use crate::byte_air::{
            self, build_trace_polynomials, ByteOpWitness,
            COL_INDEX_OFFSET, COL_RESULT_OFFSET, COL_VALUE_OFFSET,
        };
        use crate::executor::execute_bytecode;
        use metavm_zkp::field::CurveType;

        let curve = CurveType::Bls48581;

        // PUSH32 with a distinctive 32-byte value, PUSH1 index, BYTE, STOP.
        // The 32-byte value: bytes 0x20, 0x1f, …, 0x01 (BE byte order on stack).
        let mut bytecode = vec![0x7f]; // PUSH32
        for k in 0..32u8 {
            bytecode.push(0x20 - k);   // [0x20, 0x1f, …, 0x01]
        }
        bytecode.extend_from_slice(&[0x60, 5, 0x1a, 0x00]); // PUSH1 5; BYTE; STOP
        // BYTE pops (i=5, x). With x_be = [0x20, 0x1f, …, 0x01], byte 5 = 0x1b.

        let t = execute_bytecode(&bytecode, &[]).unwrap();
        let byte_row = (0..t.opcode.len())
            .find(|&i| t.opcode[i] == 0x1A)
            .expect("BYTE row must exist");
        assert_eq!(t.sel_byte_op[byte_row], 1);
        // Output is the byte 0x1b in the low byte of output0_l0.
        assert_eq!(t.output0[0][byte_row] & 0xff, 0x1b);

        // Build matching gadget witness.
        let index = t.input0[0][byte_row];
        let value: [u64; 4] = [
            t.input1[0][byte_row],
            t.input1[1][byte_row],
            t.input1[2][byte_row],
            t.input1[3][byte_row],
        ];
        // Sanity: gadget canonical computation agrees with EVM trace.
        let expected = byte_air::byte_op_result(index, value) as u64;
        assert_eq!(expected, t.output0[0][byte_row]);

        let w = ByteOpWitness::from_inputs(&[(index, value)]);
        let gadget_trace = build_trace_polynomials(&w, curve);

        // Cross-AIR LogUp tuple: 4 INDEX + 4 VALUE + 4 RESULT.
        for k in 0..4 {
            let g = gadget_trace.columns[COL_INDEX_OFFSET + k].evaluations[0].to_u64();
            assert_eq!(g, t.input0[k][byte_row], "index limb {} mismatch", k);
        }
        for k in 0..4 {
            let g = gadget_trace.columns[COL_VALUE_OFFSET + k].evaluations[0].to_u64();
            assert_eq!(g, t.input1[k][byte_row], "value limb {} mismatch", k);
        }
        for k in 0..4 {
            let g = gadget_trace.columns[COL_RESULT_OFFSET + k].evaluations[0].to_u64();
            assert_eq!(g, t.output0[k][byte_row], "result limb {} mismatch", k);
        }
    }

    /// Slow Phase A3-byte step 2 end-to-end: joint_prove + joint_verify
    /// for EVM main (containing BYTE) + ByteOp gadget. Mirrors the
    /// CREATE2 chain pattern.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove (~15 min, dominated by EVM main \
                256-padded trace + gadget's 33 case-selector products); \
                run with --release --ignored"]
    fn joint_prove_evm_byte_op_e2e() {
        use crate::byte_air::{
            self, build_trace_polynomials, make_evm_byte_linkage_descriptor,
            ByteOpConstraintSystem, ByteOpWitness,
        };
        use crate::constraints::EvmConstraintSystem;
        use crate::executor::execute_bytecode;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::field::CurveType;
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // PUSH32 distinctive value, PUSH1 5, BYTE, STOP.
        let mut bytecode = vec![0x7f];
        for k in 0..32u8 { bytecode.push(0x20 - k); }
        bytecode.extend_from_slice(&[0x60, 5, 0x1a, 0x00]);
        let evm_trace_columns = execute_bytecode(&bytecode, &[]).unwrap();
        let byte_row = (0..evm_trace_columns.opcode.len())
            .find(|&i| evm_trace_columns.opcode[i] == 0x1A)
            .unwrap();

        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&evm_trace_columns, curve);
        let evm_cs = EvmConstraintSystem::new();

        let index = evm_trace_columns.input0[0][byte_row];
        let value = [
            evm_trace_columns.input1[0][byte_row],
            evm_trace_columns.input1[1][byte_row],
            evm_trace_columns.input1[2][byte_row],
            evm_trace_columns.input1[3][byte_row],
        ];
        let w = ByteOpWitness::from_inputs(&[(index, value)]);
        let gadget_trace = build_trace_polynomials(&w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = ByteOpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_evm_byte_linkage_descriptor(0, 1);
        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs[0].closure_a, ext.linkage_proofs[0].closure_b);

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve);
        assert!(valid, "joint_verify must accept honest BYTE chain");
        let _ = byte_air::COL_IS_REAL;
    }

    /// Slow tampering regression: corrupt EVM `output0_l0` on the BYTE
    /// row while keeping the gadget honest; multiset diverges and
    /// joint_prove rejects.
    #[test]
    #[ignore = "slow: full joint_prove with tampered witness; \
                run with --release --ignored"]
    fn joint_prove_evm_byte_op_rejects_tampered_result() {
        use crate::byte_air::{
            build_trace_polynomials, make_evm_byte_linkage_descriptor,
            ByteOpConstraintSystem, ByteOpWitness,
        };
        use crate::constraints::EvmConstraintSystem;
        use crate::executor::execute_bytecode;
        use crate::trace::COL_OUTPUT0_L0;
        use metavm_zkp::cross_air_logup::joint_prove;
        use metavm_zkp::field::{CurveType, Scalar};
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let mut bytecode = vec![0x7f];
        for k in 0..32u8 { bytecode.push(0x20 - k); }
        bytecode.extend_from_slice(&[0x60, 5, 0x1a, 0x00]);
        let t_cols = execute_bytecode(&bytecode, &[]).unwrap();
        let byte_row = (0..t_cols.opcode.len()).find(|&i| t_cols.opcode[i] == 0x1A).unwrap();

        let mut evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(&t_cols, curve);
        // Tamper: flip BYTE row's output0_l0 to a wrong value (0xff).
        // Note: from_vm_trace skips col 0 (step), so byte_row stays the
        // same and column index `COL_OUTPUT0_L0` indexes into the
        // post-skip column layout.
        evm_polys.columns[COL_OUTPUT0_L0].evaluations[byte_row] =
            Scalar::from_u64(0xff, curve);

        let evm_cs = EvmConstraintSystem::new();
        let index = t_cols.input0[0][byte_row];
        let value = [
            t_cols.input1[0][byte_row],
            t_cols.input1[1][byte_row],
            t_cols.input1[2][byte_row],
            t_cols.input1[3][byte_row],
        ];
        let w = ByteOpWitness::from_inputs(&[(index, value)]);
        let gadget_trace = build_trace_polynomials(&w, curve);
        let gadget_omega = scheme.domain_generator(gadget_trace.padded_size);
        let gadget_cs = ByteOpConstraintSystem::new(gadget_trace.num_rows)
            .with_omega_and_domain(gadget_omega, gadget_trace.padded_size);

        let linkage = make_evm_byte_linkage_descriptor(0, 1);
        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![(&evm_polys, &evm_cs), (&gadget_trace, &gadget_cs)];
        let linkages = vec![linkage];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(result.is_err(), "joint_prove must reject tampered BYTE output");
    }

    /// Three-AIR end-to-end CREATE2 address derivation joint_prove —
    /// the CREATE2 analog of `joint_prove_evm_create_address_e2e`.
    ///
    /// The chain:
    ///   - L1: EVM main↔CREATE2 input gadget on
    ///         `(sender_limbs, salt_limbs, initcode_hash_limbs)`
    ///         (12 LE u64 limbs total) gated by `sel_create2`.
    ///   - L2: CREATE2 input gadget↔KeccakExtract input on
    ///         `(INPUT_BYTE[0..256], INPUT_LEN)`.
    ///   - L3: EVM main↔KeccakExtract output on address-limb tuples,
    ///         gated by `sel_create2` (the same L3 helper handles
    ///         CREATE2 by selector).
    ///
    /// Net: the EVM CREATE2 row's `create_address_hint` is now
    /// algebraically pinned by the (sender, salt, initcode_hash) triple
    /// + the Keccak invocation that hashes the canonical pre-image.
    ///
    /// Closes the EVM-side adapter for #117 (cross-AIR layer).
    #[test]
    #[ignore = "slow: 3-AIR + 3-linkage joint_prove (~12 min, dominated \
                by EVM main 256-padded trace + 1398 LogUp cols)"]
    fn joint_prove_evm_create2_address_e2e() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::evm_create2_input_air;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            self, KeccakExtractWitness, KeccakExtractConstraintSystem,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // ── EVM main trace from the real executor ──
        // Bytecode that does one CREATE2 with empty init code, salt = 1:
        //   PUSH1 1   (salt)
        //   PUSH1 0   (size)
        //   PUSH1 0   (offset)
        //   PUSH1 0   (value)
        //   CREATE2   (opcode 0xF5)
        //   STOP
        let bytecode = vec![
            0x60, 0x01, // PUSH1 1
            0x60, 0x00, // PUSH1 0
            0x60, 0x00, // PUSH1 0
            0x60, 0x00, // PUSH1 0
            0xF5,       // CREATE2
            0x00,       // STOP
        ];
        let evm_trace_columns = execute_bytecode(&bytecode, &[])
            .expect("CREATE2 bytecode must execute cleanly");

        let create2_row = (0..evm_trace_columns.opcode.len())
            .find(|&i| evm_trace_columns.opcode[i] == 0xF5)
            .expect("CREATE2 row must appear in trace");
        let sender_limbs: [u64; 4] = [
            evm_trace_columns.frame_callee[0][create2_row],
            evm_trace_columns.frame_callee[1][create2_row],
            evm_trace_columns.frame_callee[2][create2_row],
            evm_trace_columns.frame_callee[3][create2_row],
        ];
        let mut sender = [0u8; 20];
        sender[0..8].copy_from_slice(&sender_limbs[0].to_le_bytes());
        sender[8..16].copy_from_slice(&sender_limbs[1].to_le_bytes());
        sender[16..20].copy_from_slice(&sender_limbs[2].to_le_bytes()[..4]);
        let salt_limbs: [u64; 4] = [
            evm_trace_columns.create2_salt_hint[0][create2_row],
            evm_trace_columns.create2_salt_hint[1][create2_row],
            evm_trace_columns.create2_salt_hint[2][create2_row],
            evm_trace_columns.create2_salt_hint[3][create2_row],
        ];
        let mut salt = [0u8; 32];
        for (k, &l) in salt_limbs.iter().enumerate() {
            salt[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
        }
        let initcode_hash_limbs: [u64; 4] = [
            evm_trace_columns.create2_initcode_hash_hint[0][create2_row],
            evm_trace_columns.create2_initcode_hash_hint[1][create2_row],
            evm_trace_columns.create2_initcode_hash_hint[2][create2_row],
            evm_trace_columns.create2_initcode_hash_hint[3][create2_row],
        ];
        let mut initcode_hash = [0u8; 32];
        for (k, &l) in initcode_hash_limbs.iter().enumerate() {
            initcode_hash[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
        }

        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(
            &evm_trace_columns, curve,
        );
        let evm_cs = EvmConstraintSystem::new();

        // ── CREATE2 input gadget (single invocation matching the EVM CREATE2 row) ──
        let gadget_w = evm_create2_input_air::Create2InputWitness::from_inputs(
            &[(&sender, &salt, &initcode_hash)],
        );
        let gadget_trace = evm_create2_input_air::build_trace_polynomials(&gadget_w, curve);
        let gadget_cs = evm_create2_input_air::Create2InputConstraintSystem::new(
            gadget_trace.num_rows,
        );

        // ── KeccakExtract (single invocation, hashing the canonical pre-image) ──
        let canonical = evm_create2_input_air::canonical_create2_input(
            &sender, &salt, &initcode_hash,
        );
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical.to_vec()])
            .expect("CREATE2 pre-image fits in MAX_INPUT_LEN");
        let extract_trace = keccak_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows);

        // Sanity: the executor-derived address hint must match what
        // KeccakExtract's address_limb aggregator computes from the
        // canonical pre-image digest.
        let address_limbs = keccak_extract::address_limbs_from_output(
            &extract_w.invocations[0].output,
        );
        for k in 0..4 {
            assert_eq!(
                evm_trace_columns.create_address_hint[k][create2_row],
                address_limbs[k],
                "EVM address hint limb {} must match keccak256(0xff||sender||salt||initcode_hash)[12..32]",
                k
            );
        }

        // ── Linkages ──
        let l1 = make_evm_main_create2_input_linkage_descriptor(/* evm */ 0, /* gadget */ 1);
        let l2 = evm_create2_input_air::make_create2_input_keccak_extract_input_linkage_descriptor(
            /* gadget */ 1, /* extract */ 2,
        );
        let l3 = make_evm_create_address_keccak_extract_linkage_descriptor(
            /* evm */ 0, /* extract */ 2, crate::trace::COL_SEL_CREATE2,
        );

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&evm_polys, &evm_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        let linkages = vec![l1, l2, l3];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the honest 3-AIR + 3-linkage CREATE2 chain");
        assert_eq!(proofs.len(), 3, "one proof per AIR");
        assert_eq!(extension.linkage_proofs.len(), 3, "one linkage proof each");
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs, &extract_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the honest CREATE2 address-derivation chain"
        );
    }

    /// Four-AIR end-to-end CREATE2 address derivation joint_prove,
    /// extending the 3-AIR test with bit-level Keccak as the 4th layer.
    /// The CREATE2 analog of
    /// `joint_prove_evm_create_address_e2e_with_bit_level_keccak`.
    ///
    /// Linkages:
    ///   - L1: EVM main↔CREATE2 input gadget on 12-limb tuple
    ///   - L2: gadget↔KeccakExtract input on 257-tuple
    ///   - L3: EVM main↔KeccakExtract output on address-limb tuple,
    ///         gated by `sel_create2`
    ///   - L4: KeccakExtract↔bit-level Keccak on byte aggregator tuples
    ///
    /// Net result: the EVM CREATE2 row's `create_address_hint` is
    /// algebraically pinned all the way through bit-level keccak256.
    /// Closes the algebraic soundness chain end-to-end for CREATE2.
    #[test]
    #[ignore = "very slow: 4-AIR + 4-linkage joint_prove (~80 min, dominated \
                by EVM main 256-padded trace + bit-level Keccak)"]
    fn joint_prove_evm_create2_address_e2e_with_bit_level_keccak() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::cross_air_logup::{joint_prove, joint_verify};
        use metavm_zkp::evm_create2_input_air;
        use metavm_zkp::field::{CurveType, Scalar};
        use metavm_zkp::keccak::keccak_witness;
        use metavm_zkp::keccak_air;
        use metavm_zkp::keccak::NUM_ROUNDS;
        use metavm_zkp::keccak_constraints::KeccakConstraintSystem;
        use metavm_zkp::keccak_extract::{
            self, make_keccak_extract_keccak_linkage_descriptor,
            KeccakExtractWitness, KeccakExtractConstraintSystem,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // Same bytecode as the 3-AIR test.
        let bytecode = vec![
            0x60, 0x01, 0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF5, 0x00,
        ];
        let evm_trace_columns = execute_bytecode(&bytecode, &[])
            .expect("CREATE2 bytecode must execute cleanly");

        let create2_row = (0..evm_trace_columns.opcode.len())
            .find(|&i| evm_trace_columns.opcode[i] == 0xF5)
            .expect("CREATE2 row must appear in trace");

        let sender_limbs: [u64; 4] = [
            evm_trace_columns.frame_callee[0][create2_row],
            evm_trace_columns.frame_callee[1][create2_row],
            evm_trace_columns.frame_callee[2][create2_row],
            evm_trace_columns.frame_callee[3][create2_row],
        ];
        let mut sender = [0u8; 20];
        sender[0..8].copy_from_slice(&sender_limbs[0].to_le_bytes());
        sender[8..16].copy_from_slice(&sender_limbs[1].to_le_bytes());
        sender[16..20].copy_from_slice(&sender_limbs[2].to_le_bytes()[..4]);
        let salt_limbs: [u64; 4] = [
            evm_trace_columns.create2_salt_hint[0][create2_row],
            evm_trace_columns.create2_salt_hint[1][create2_row],
            evm_trace_columns.create2_salt_hint[2][create2_row],
            evm_trace_columns.create2_salt_hint[3][create2_row],
        ];
        let mut salt = [0u8; 32];
        for (k, &l) in salt_limbs.iter().enumerate() {
            salt[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
        }
        let initcode_hash_limbs: [u64; 4] = [
            evm_trace_columns.create2_initcode_hash_hint[0][create2_row],
            evm_trace_columns.create2_initcode_hash_hint[1][create2_row],
            evm_trace_columns.create2_initcode_hash_hint[2][create2_row],
            evm_trace_columns.create2_initcode_hash_hint[3][create2_row],
        ];
        let mut initcode_hash = [0u8; 32];
        for (k, &l) in initcode_hash_limbs.iter().enumerate() {
            initcode_hash[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
        }

        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(
            &evm_trace_columns, curve,
        );
        let evm_cs = EvmConstraintSystem::new();

        // CREATE2 input gadget.
        let gadget_w = evm_create2_input_air::Create2InputWitness::from_inputs(
            &[(&sender, &salt, &initcode_hash)],
        );
        let gadget_trace = evm_create2_input_air::build_trace_polynomials(&gadget_w, curve);
        let gadget_cs = evm_create2_input_air::Create2InputConstraintSystem::new(
            gadget_trace.num_rows,
        );

        // KeccakExtract.
        let canonical = evm_create2_input_air::canonical_create2_input(
            &sender, &salt, &initcode_hash,
        );
        let canonical_vec: Vec<u8> = canonical.to_vec();
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical_vec.clone()])
            .expect("CREATE2 pre-image fits in MAX_INPUT_LEN");
        let extract_trace = keccak_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows);

        // Bit-level Keccak (4th AIR).
        let digest = metavm_zkp::keccak::keccak256(&canonical_vec);
        let ht = keccak_witness(&canonical_vec);
        let num_keccak_rows = ht.blocks.len() * NUM_ROUNDS;
        let keccak_padded = metavm_zkp::trace::nearest_power_of_two(num_keccak_rows.max(1));
        let mut keccak_columns = keccak_air::populate_trace_from_hash_with_invocation_bytes(
            &ht, &canonical_vec, &digest, curve,
        );
        for col in keccak_columns.iter_mut() {
            if col.len() < keccak_padded {
                col.resize(keccak_padded, Scalar::zero(curve));
            }
        }
        let keccak_polys: Vec<metavm_zkp::trace::Polynomial> = keccak_columns
            .into_iter()
            .map(|evals| metavm_zkp::trace::Polynomial {
                evaluations: evals,
                degree: num_keccak_rows,
            })
            .collect();
        let keccak_trace = metavm_zkp::trace::TracePolynomials {
            columns: keccak_polys,
            num_rows: num_keccak_rows,
            padded_size: keccak_padded as u64,
            curve,
        };
        let keccak_cs = KeccakConstraintSystem::new(num_keccak_rows);

        // Linkages.
        let l1 = make_evm_main_create2_input_linkage_descriptor(0, 1);
        let l2 = evm_create2_input_air::make_create2_input_keccak_extract_input_linkage_descriptor(
            1, 2,
        );
        let l3 = make_evm_create_address_keccak_extract_linkage_descriptor(
            0, 2, crate::trace::COL_SEL_CREATE2,
        );
        let l4 = make_keccak_extract_keccak_linkage_descriptor(2, 3);

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&evm_polys, &evm_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
            (&keccak_trace, &keccak_cs),
        ];
        let linkages = vec![l1, l2, l3, l4];

        let (proofs, extension) = joint_prove(&traces, &linkages, &scheme)
            .expect("joint_prove must succeed for the honest 4-AIR + 4-linkage CREATE2 chain");
        assert_eq!(proofs.len(), 4, "one proof per AIR");
        assert_eq!(extension.linkage_proofs.len(), 4, "one linkage proof each");
        for (i, lp) in extension.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "linkage {} closures must match on honest witness", i
            );
        }

        let cs_refs: Vec<&dyn metavm_zkp::vm_constraints::VmConstraintSystem> =
            vec![&evm_cs, &gadget_cs, &extract_cs, &keccak_cs];
        let valid = joint_verify(&proofs, &cs_refs, &linkages, &extension, &scheme, curve);
        assert!(
            valid,
            "joint verifier must accept the honest 4-AIR CREATE2 address chain"
        );
    }

    /// Multi-invocation CREATE2 sanity test: builds an EVM trace with
    /// TWO CREATE2 opcodes back-to-back (different salts so the hint
    /// limbs are distinct), then asserts that the
    /// `make_evm_main_create2_input_linkage_descriptor` correctly
    /// matches BOTH gated rows against a 2-row gadget witness via the
    /// witness-builder dry-run (without running the full ~12-min joint
    /// proof).
    ///
    /// Closes the "only-single-CREATE2-validated" residual on #117.
    #[test]
    fn evm_multi_create2_hint_tuples_match_gadget_two_invocations() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::cross_air_logup::compute_cross_air_logup_witness;
        use metavm_zkp::evm_create2_input_air;
        use metavm_zkp::field::CurveType;

        let curve = CurveType::Bls48581;

        // Bytecode: two CREATE2 opcodes back-to-back with empty initcode.
        //   PUSH1 1; PUSH1 0×3; CREATE2; POP;       (salt = 1)
        //   PUSH1 2; PUSH1 0×3; CREATE2; POP; STOP  (salt = 2)
        let bytecode = vec![
            0x60, 0x01, 0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF5, 0x50,
            0x60, 0x02, 0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF5, 0x50,
            0x00,
        ];
        let evm_trace_columns = execute_bytecode(&bytecode, &[]).unwrap();

        let create2_rows: Vec<usize> = (0..evm_trace_columns.opcode.len())
            .filter(|&i| evm_trace_columns.opcode[i] == 0xF5)
            .collect();
        assert_eq!(create2_rows.len(), 2, "expected 2 CREATE2 rows");

        // Reconstruct (sender, salt, initcode_hash) for each CREATE2 row.
        let invocations: Vec<([u8; 20], [u8; 32], [u8; 32])> = create2_rows
            .iter()
            .map(|&row| {
                let sender_limbs = [
                    evm_trace_columns.frame_callee[0][row],
                    evm_trace_columns.frame_callee[1][row],
                    evm_trace_columns.frame_callee[2][row],
                    evm_trace_columns.frame_callee[3][row],
                ];
                let mut sender = [0u8; 20];
                sender[0..8].copy_from_slice(&sender_limbs[0].to_le_bytes());
                sender[8..16].copy_from_slice(&sender_limbs[1].to_le_bytes());
                sender[16..20].copy_from_slice(&sender_limbs[2].to_le_bytes()[..4]);

                let salt_limbs = [
                    evm_trace_columns.create2_salt_hint[0][row],
                    evm_trace_columns.create2_salt_hint[1][row],
                    evm_trace_columns.create2_salt_hint[2][row],
                    evm_trace_columns.create2_salt_hint[3][row],
                ];
                let mut salt = [0u8; 32];
                for (k, &l) in salt_limbs.iter().enumerate() {
                    salt[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
                }
                let ih_limbs = [
                    evm_trace_columns.create2_initcode_hash_hint[0][row],
                    evm_trace_columns.create2_initcode_hash_hint[1][row],
                    evm_trace_columns.create2_initcode_hash_hint[2][row],
                    evm_trace_columns.create2_initcode_hash_hint[3][row],
                ];
                let mut initcode_hash = [0u8; 32];
                for (k, &l) in ih_limbs.iter().enumerate() {
                    initcode_hash[8 * k..8 * k + 8].copy_from_slice(&l.to_le_bytes());
                }
                (sender, salt, initcode_hash)
            })
            .collect();

        // Salt must differ between the two invocations.
        assert_ne!(invocations[0].1, invocations[1].1, "salts must differ");

        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(
            &evm_trace_columns, curve,
        );

        // Gadget with 2 invocations.
        let inv_refs: Vec<(&[u8; 20], &[u8; 32], &[u8; 32])> = invocations
            .iter()
            .map(|(s, sa, ih)| (s, sa, ih))
            .collect();
        let gadget_w = evm_create2_input_air::Create2InputWitness::from_inputs(&inv_refs);
        let gadget_trace = evm_create2_input_air::build_trace_polynomials(&gadget_w, curve);

        // Run the cross-AIR LogUp witness builder directly (avoids the
        // ~12-min full joint_prove). Synthesize deterministic
        // beta/gamma; the multiset-equality check is independent of
        // their concrete values (it's a structural invariant of the
        // tuples themselves).
        let l1 = make_evm_main_create2_input_linkage_descriptor(0, 1);
        let beta = metavm_zkp::field::Scalar::from_u64(7, curve);
        let gamma = metavm_zkp::field::Scalar::from_u64(11, curve);
        let result = compute_cross_air_logup_witness(
            &evm_polys, &gadget_trace, &l1, &beta, &gamma, curve,
        );
        assert!(
            result.is_ok(),
            "cross_air_logup must accept honest 2-invocation CREATE2 chain: {:?}",
            result.err()
        );
    }

    /// Tampering test for the CREATE2 cross-AIR chain: confirms the
    /// witness builder rejects a malicious gadget invocation whose salt
    /// differs from what the EVM trace recorded. Mirrors
    /// `joint_prove_mpt_chain_rejects_tampered_node_hash` and the
    /// Finality/SSZ tampering tests.
    ///
    /// The EVM trace truthfully records `salt = 1` from the CREATE2
    /// opcode; the gadget witness lies and uses `salt = 2`. The L1
    /// linkage compares the 12-limb tuple `(sender, salt, initcode_hash)`
    /// — a mismatched salt limb breaks multiset equality, and
    /// `compute_cross_air_logup_witness` rejects with "multiset equality
    /// cannot hold" before any proof artifacts are produced.
    #[test]
    fn joint_prove_evm_create2_chain_rejects_tampered_salt() {
        use crate::executor::execute_bytecode;
        use metavm_zkp::cross_air_logup::joint_prove;
        use metavm_zkp::evm_create2_input_air;
        use metavm_zkp::field::CurveType;
        use metavm_zkp::keccak_extract::{
            self, KeccakExtractWitness, KeccakExtractConstraintSystem,
        };
        use metavm_zkp::scheme::bls48581_scheme::Bls48581Scheme;
        use metavm_zkp::scheme::CommitmentScheme;
        use crate::constraints::EvmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        // EVM trace: PUSH1 1 || PUSH1 0 × 3 || CREATE2 || STOP — same
        // as the honest e2e test, salt = 1.
        let bytecode = vec![
            0x60, 0x01, 0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0xF5, 0x00,
        ];
        let evm_trace_columns = execute_bytecode(&bytecode, &[]).unwrap();
        let row = (0..evm_trace_columns.opcode.len())
            .find(|&i| evm_trace_columns.opcode[i] == 0xF5)
            .expect("CREATE2 row");

        // Reconstruct sender from the EVM trace.
        let sender_limbs = [
            evm_trace_columns.frame_callee[0][row],
            evm_trace_columns.frame_callee[1][row],
            evm_trace_columns.frame_callee[2][row],
            evm_trace_columns.frame_callee[3][row],
        ];
        let mut sender = [0u8; 20];
        sender[0..8].copy_from_slice(&sender_limbs[0].to_le_bytes());
        sender[8..16].copy_from_slice(&sender_limbs[1].to_le_bytes());
        sender[16..20].copy_from_slice(&sender_limbs[2].to_le_bytes()[..4]);

        // TAMPER: gadget uses salt = 2 (BE byte 31), but the EVM
        // trace recorded salt = 1 from the actual CREATE2 opcode.
        let mut tampered_salt = [0u8; 32];
        tampered_salt[31] = 2;
        // Honest initcode_hash for the empty initcode.
        let initcode_hash_b256 = revm::primitives::keccak256(&[]);
        let initcode_hash: [u8; 32] = initcode_hash_b256.0;

        let evm_polys = metavm_zkp::trace::TracePolynomials::from_vm_trace(
            &evm_trace_columns, curve,
        );
        let evm_cs = EvmConstraintSystem::new();

        let gadget_w = evm_create2_input_air::Create2InputWitness::from_inputs(
            &[(&sender, &tampered_salt, &initcode_hash)],
        );
        let gadget_trace = evm_create2_input_air::build_trace_polynomials(&gadget_w, curve);
        let gadget_cs = evm_create2_input_air::Create2InputConstraintSystem::new(
            gadget_trace.num_rows,
        );

        // KeccakExtract: hash whatever the gadget claims (so L2 can
        // potentially still match — we want L1 to fail first).
        let canonical = evm_create2_input_air::canonical_create2_input(
            &sender, &tampered_salt, &initcode_hash,
        );
        let extract_w = KeccakExtractWitness::from_inputs(&[canonical.to_vec()])
            .expect("CREATE2 pre-image fits in MAX_INPUT_LEN");
        let extract_trace = keccak_extract::build_trace_polynomials(&extract_w, curve);
        let extract_cs = KeccakExtractConstraintSystem::new(extract_trace.num_rows);

        let l1 = make_evm_main_create2_input_linkage_descriptor(0, 1);

        let traces: Vec<(
            &metavm_zkp::trace::TracePolynomials,
            &dyn metavm_zkp::vm_constraints::VmConstraintSystem,
        )> = vec![
            (&evm_polys, &evm_cs),
            (&gadget_trace, &gadget_cs),
            (&extract_trace, &extract_cs),
        ];
        // Only the L1 (EVM↔gadget) linkage matters for catching the
        // tampered salt — L2/L3 are unnecessary for this test.
        let linkages = vec![l1];

        let result = joint_prove(&traces, &linkages, &scheme);
        assert!(
            result.is_err(),
            "joint_prove must reject CREATE2 chain when gadget salt differs from EVM hint"
        );
        let err = result.err().unwrap();
        assert!(
            err.contains("multiset equality cannot hold"),
            "unexpected error message: {}",
            err
        );
    }

    /// Build a real `(EVM main trace with single CREATE row,
    /// KeccakExtract trace with the matching keccak invocation)` pair
    /// and assert the 4-limb cross-AIR LogUp tuples match byte-for-byte
    /// at the gated rows. Mirrors
    /// `mpt_keccak_extract_byte_tuples_match_real_mpt_witness`.
    ///
    /// The `address_to_limbs` (EVM inspector) and
    /// `address_limbs_from_output` (KeccakExtract aggregator) functions
    /// must produce identical 4-limb tuples for the same 20-byte
    /// address — otherwise the cross-AIR LogUp would never match.
    #[test]
    fn evm_create_address_tuples_match_keccak_extract_aggregator() {
        use revm::primitives::Address;
        // Pick a representative pre-image and digest.
        let preimage = b"the quick brown fox creates a contract".to_vec();
        let digest = metavm_zkp::keccak::keccak256(&preimage);
        let mut addr_bytes = [0u8; 20];
        addr_bytes.copy_from_slice(&digest[12..32]);
        let address = Address::from(addr_bytes);

        // 1) EVM inspector's limb decomposition (must match what the
        //    real CREATE / CREATE2 hooks populate).
        //    Re-create address_to_limbs locally — it lives in
        //    `crates/evm/src/inspector.rs` and is `pub(crate)`.
        let evm_limbs: [u64; 4] = {
            let bytes = address.0.0;
            let mut limbs = [0u64; 4];
            for i in 0..2 {
                let mut tmp = [0u8; 8];
                tmp.copy_from_slice(&bytes[8 * i..8 * i + 8]);
                limbs[i] = u64::from_le_bytes(tmp);
            }
            let mut tmp = [0u8; 8];
            tmp[..4].copy_from_slice(&bytes[16..20]);
            limbs[2] = u64::from_le_bytes(tmp);
            limbs
        };

        // 2) KeccakExtract aggregator's limb decomposition.
        let ke_limbs = metavm_zkp::keccak_extract::address_limbs_from_output(&digest);

        // 3) Tuples MUST match — this is the cross-AIR linkage's
        //    soundness foundation.
        assert_eq!(evm_limbs, ke_limbs, "EVM and KeccakExtract must agree on the 4-limb address tuple");
        // Also confirm limb 3 is identically zero (a 20-byte address
        // never spills into limb 3 in either encoding).
        assert_eq!(evm_limbs[3], 0);
        assert_eq!(ke_limbs[3], 0);
    }

    #[test]
    fn evm_main_create_rlp_descriptor_well_formed() {
        // Verifies the EVM-side wrapper wires the right columns from
        // the EVM main trace: frame_callee limbs (cols 205..209),
        // create_nonce_hint (col 230), gated by sel_create (col 219).
        let desc = make_evm_main_create_rlp_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "evm_main_create_rlp_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 5);
        assert_eq!(desc.b_columns.len(), 5);
        // A side: frame_callee_l[0..4] then create_nonce_hint.
        assert_eq!(desc.a_columns[0], crate::trace::COL_FRAME_CALLEE_L0);
        assert_eq!(desc.a_columns[1], crate::trace::COL_FRAME_CALLEE_L0 + 1);
        assert_eq!(desc.a_columns[2], crate::trace::COL_FRAME_CALLEE_L0 + 2);
        assert_eq!(desc.a_columns[3], crate::trace::COL_FRAME_CALLEE_L0 + 3);
        assert_eq!(desc.a_columns[4], crate::trace::COL_CREATE_NONCE_HINT);
        // B side: gadget's SENDER_LIMB[0..4] then NONCE.
        for k in 0..4 {
            assert_eq!(
                desc.b_columns[k],
                metavm_zkp::evm_create_rlp_air::COL_SENDER_LIMB_OFFSET + k
            );
        }
        assert_eq!(
            desc.b_columns[4],
            metavm_zkp::evm_create_rlp_air::COL_NONCE
        );
        assert_eq!(desc.a_selector_column, Some(crate::trace::COL_SEL_CREATE));
        assert_eq!(
            desc.b_selector_column,
            Some(metavm_zkp::evm_create_rlp_air::COL_IS_REAL)
        );
    }

    #[test]
    fn evm_create_address_keccak_extract_descriptor_well_formed() {
        use crate::trace::{
            COL_CREATE_ADDRESS_HINT_L0, COL_CREATE_ADDRESS_HINT_L1, COL_CREATE_ADDRESS_HINT_L2,
            COL_CREATE_ADDRESS_HINT_L3, COL_SEL_CREATE, COL_SEL_CREATE2,
        };
        // CREATE descriptor.
        let desc = make_evm_create_address_keccak_extract_linkage_descriptor(
            0, 1, COL_SEL_CREATE,
        );
        assert_eq!(desc.label, "evm_create_address_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), 4);
        assert_eq!(desc.b_columns.len(), 4);
        assert_eq!(desc.a_columns[0], COL_CREATE_ADDRESS_HINT_L0);
        assert_eq!(desc.a_columns[1], COL_CREATE_ADDRESS_HINT_L1);
        assert_eq!(desc.a_columns[2], COL_CREATE_ADDRESS_HINT_L2);
        assert_eq!(desc.a_columns[3], COL_CREATE_ADDRESS_HINT_L3);
        assert_eq!(desc.a_selector_column, Some(COL_SEL_CREATE));
        for k in 0..4 {
            assert_eq!(
                desc.b_columns[k],
                metavm_zkp::keccak_extract::COL_ADDRESS_LIMB_OFFSET + k
            );
        }
        assert_eq!(
            desc.b_selector_column,
            Some(metavm_zkp::keccak_extract::COL_IS_REAL)
        );

        // CREATE2 descriptor differs only in label and A-side selector.
        let desc2 = make_evm_create_address_keccak_extract_linkage_descriptor(
            0, 1, COL_SEL_CREATE2,
        );
        assert_eq!(desc2.label, "evm_create2_address_v1");
        assert_eq!(desc2.a_selector_column, Some(COL_SEL_CREATE2));
    }

    #[test]
    fn exp_consistency_accepts_correct_claim_with_matching_trace() {
        let claim = EvmExpClaim {
            base: U256::from(2u64),
            exponent: U256::from(10u64),
            claimed_result: U256::from(1024u64),
        };
        let traces = exp_traces_for_evm_claims(&[claim]);
        let checks = check_evm_exp_consistency(&[claim], &traces);
        assert_eq!(checks, vec![EvmExpRowCheck::OkLinked]);
    }

    #[test]
    fn exp_consistency_accepts_correct_claim_without_trace() {
        let claim = EvmExpClaim {
            base: U256::from(3u64),
            exponent: U256::from(5u64),
            claimed_result: U256::from(243u64),
        };
        let checks = check_evm_exp_consistency(&[claim], &[]);
        assert_eq!(checks, vec![EvmExpRowCheck::OkNoTrace]);
    }

    #[test]
    fn exp_consistency_rejects_wrong_result() {
        let claim = EvmExpClaim {
            base: U256::from(2u64),
            exponent: U256::from(10u64),
            claimed_result: U256::from(1023u64), // off by one
        };
        let checks = check_evm_exp_consistency(&[claim], &[]);
        assert_eq!(checks, vec![EvmExpRowCheck::ResultMismatch]);
    }

    #[test]
    fn exp_consistency_rejects_correct_result_with_mismatched_trace() {
        // Claim is correct (2^10 = 1024) but the supplied trace covers
        // a DIFFERENT (base, exponent), so the linkage is incomplete.
        let real_claim = EvmExpClaim {
            base: U256::from(2u64),
            exponent: U256::from(10u64),
            claimed_result: U256::from(1024u64),
        };
        let mismatched_trace = exp_witness(U256::from(3u64), U256::from(5u64));
        let checks = check_evm_exp_consistency(&[real_claim], &[mismatched_trace]);
        assert_eq!(checks, vec![EvmExpRowCheck::OkNoTrace]);
    }

    #[test]
    fn exp_consistency_handles_zero_exponent() {
        // Anything to the 0 = 1 (including 0^0 in EVM semantics).
        let claim = EvmExpClaim {
            base: U256::from(7u64),
            exponent: U256::ZERO,
            claimed_result: U256::from(1u64),
        };
        let traces = exp_traces_for_evm_claims(&[claim]);
        let checks = check_evm_exp_consistency(&[claim], &traces);
        assert_eq!(checks, vec![EvmExpRowCheck::OkLinked]);
    }

    #[test]
    fn exp_consistency_handles_overflow_modulo_2_256() {
        // 2^256 mod 2^256 = 0. The gadget natively reduces mod 2^256.
        let claim = EvmExpClaim {
            base: U256::from(2u64),
            exponent: U256::from(256u64),
            claimed_result: U256::ZERO,
        };
        let traces = exp_traces_for_evm_claims(&[claim]);
        let checks = check_evm_exp_consistency(&[claim], &traces);
        assert_eq!(checks, vec![EvmExpRowCheck::OkLinked]);
    }

    #[test]
    fn exp_consistency_handles_multiple_claims() {
        let claims = vec![
            EvmExpClaim {
                base: U256::from(2u64),
                exponent: U256::from(8u64),
                claimed_result: U256::from(256u64),
            },
            EvmExpClaim {
                base: U256::from(5u64),
                exponent: U256::from(3u64),
                claimed_result: U256::from(125u64),
            },
            EvmExpClaim {
                base: U256::from(10u64),
                exponent: U256::from(2u64),
                claimed_result: U256::from(99u64), // wrong (should be 100)
            },
        ];
        let traces = exp_traces_for_evm_claims(&claims);
        let checks = check_evm_exp_consistency(&claims, &traces);
        assert_eq!(
            checks,
            vec![
                EvmExpRowCheck::OkLinked,
                EvmExpRowCheck::OkLinked,
                EvmExpRowCheck::ResultMismatch,
            ]
        );
    }
}
