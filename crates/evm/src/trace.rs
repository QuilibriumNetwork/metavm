//! EVM execution trace with 256-bit limb decomposition.
//!
//! Each 256-bit EVM value is decomposed into 4 x u64 limbs (little-endian:
//! limb0 = least significant 64 bits).

use metavm_core::vm_traits::VmTrace;
use metavm_zkp::field::CurveType;
use metavm_zkp::trace::TracePolynomials;
use sha3::{Sha3_256, Digest};

/// Number of data columns (excluding step).
///
/// Layout: 77 base columns (through next_pc) + 32 SIGNEXTEND case selectors
/// + 8 byte-decomp witnesses + 1 low7 + 1 sign_bit (= 119)
/// + 35 MULMOD witnesses (q 4 + product 8 + prod_c 7 + sum_c 7 + slack 4
///   + slack_borrow 4 + n_is_zero 1) = 154
/// + 46 ADDMOD witnesses (q 4 + s_low 4 + s_high 1 + ab_carry 4 + qn_p 8
///   + qn_carry 7 + sum_carry 9 + slack 4 + slack_borrow 4 + n_is_zero 1) = 200
/// + 17 frame-stack columns (depth 1 + caller 4 + callee 4 + value 4 + gas 1
///   + return_pc 1 + return_offset 1 + return_size 1) = 217
/// + 2 split-call selectors (sel_call_push_frame, sel_call_return) = 219
/// + 6 phase-5 second-pass call-family selectors (CREATE, CALLCODE,
///   DELEGATECALL, CREATE2, STATICCALL, REVERT) = 225
/// + 1 frame_static flag (sticky static-mode bit) = 226
/// + 4 create_address_hint limbs (oracle: contract address derived by inspector
///   for CREATE/CREATE2 — the in-circuit constraint binds the next frame's
///   callee to this hint; the actual derivation is delegated to a future
///   keccak cross-AIR linkage) = 230.
/// + 1 create_nonce_hint (oracle: sender account's live nonce at the moment
///   of the CREATE opcode; consumed by the cross-AIR LogUp linkage to the
///   `EvmCreateRlpAir` gadget so the gadget's RLP encoding can be tied to
///   the canonical CREATE pre-image. Zero on non-CREATE rows.) = 231.
/// + 1 sel_stop_pop (witness oracle: 1 iff opcode==STOP AND frame.depth ≥ 1.
///   Marks STOP rows where the next row is in a popped-into frame
///   (init code completion or callee STOP-as-implicit-RETURN). Used as
///   a pop_selector in the LIFO frame-stack permutation argument and in
///   PC-continuity / frame-state-preservation gates. Soundness gap:
///   currently an unbound oracle — a malicious prover could set it to 1
///   on non-STOP or depth=0 rows and bypass continuity/preservation.
///   Future work: add row-local binarity + sel_stop_pop·(1-sel_stop)=0 +
///   a depth-nonzero binding via a depth_inv witness column.) = 232.
/// + 4 create2_salt_hint limbs (oracle: salt argument from stack[3] of
///   CREATE2 opcode, as 4 LE u64 limbs. Zero on non-CREATE2 rows.) = 236.
/// + 4 create2_initcode_hash_hint limbs (oracle: keccak256 of the init
///   code at memory[offset..offset+size] for CREATE2, as 4 LE u64 limbs.
///   Zero on non-CREATE2 rows. Used together with salt_hint to build the
///   canonical CREATE2 pre-image `0xff || sender || salt || initcode_hash`
///   via the cross-AIR linkage to `EvmCreate2InputAir`.) = 240.
/// Per-variant LOG selectors for cross-AIR LogUp gating.
/// LOGn (0xA0..0xA4) fires `sel_log` (umbrella) + `sel_log{n}`.
pub const COL_SEL_LOG0: usize = 252;
pub const COL_SEL_LOG1: usize = 253;
pub const COL_SEL_LOG2: usize = 254;
pub const COL_SEL_LOG3: usize = 255;
pub const COL_SEL_LOG4: usize = 256;
/// Per-opcode ENV selectors for algebraic constraint gating.
pub const COL_SEL_ADDRESS: usize = 257;
pub const COL_SEL_CALLER: usize = 258;
pub const COL_SEL_CALLVALUE: usize = 259;
pub const COL_SEL_ORIGIN: usize = 260;
pub const COL_SEL_CALLDATASIZE: usize = 261;
pub const COL_SEL_CODESIZE: usize = 262;
pub const COL_SEL_GASPRICE: usize = 263;

/// Per-tx/per-block context columns for ENV opcode algebraic binding.
/// These are constant across all rows of a single transaction execution.
/// - tx_origin: 4 LE u64 limbs (low 160 bits = address)
/// - tx_gas_price: u64
/// - tx_calldata_size: u64
/// - tx_code_size: u64
/// - returndata_size: u64 (updated by inspector at CALL boundaries)
pub const COL_TX_ORIGIN_L0: usize = 264;
pub const COL_TX_ORIGIN_L1: usize = 265;
pub const COL_TX_ORIGIN_L2: usize = 266;
pub const COL_TX_ORIGIN_L3: usize = 267;
pub const COL_TX_GAS_PRICE: usize = 268;
pub const COL_TX_CALLDATA_SIZE: usize = 269;
pub const COL_TX_CODE_SIZE: usize = 270;
pub const COL_RETURNDATA_SIZE: usize = 271;

/// Per-opcode selectors for PC, GAS, MSIZE (algebraically bound via env_air pattern).
pub const COL_SEL_PC: usize = 272;
pub const COL_SEL_GAS: usize = 273;
pub const COL_SEL_MSIZE_OP: usize = 274;

pub const NUM_EVM_COLUMNS: usize = 275;

// Column index constants for the EVM trace layout.
// These index into the data columns (step is column 0 in VmTrace but excluded
// from TracePolynomials).
pub const COL_PC: usize = 0;
pub const COL_OPCODE: usize = 1;
pub const COL_GAS_REMAINING: usize = 2;
pub const COL_STACK_DEPTH: usize = 3;
// input0: U256 as 4 limbs
pub const COL_INPUT0_L0: usize = 4;
pub const COL_INPUT0_L1: usize = 5;
pub const COL_INPUT0_L2: usize = 6;
pub const COL_INPUT0_L3: usize = 7;
// input1: U256 as 4 limbs
pub const COL_INPUT1_L0: usize = 8;
pub const COL_INPUT1_L1: usize = 9;
pub const COL_INPUT1_L2: usize = 10;
pub const COL_INPUT1_L3: usize = 11;
// output0: U256 as 4 limbs
pub const COL_OUTPUT0_L0: usize = 12;
pub const COL_OUTPUT0_L1: usize = 13;
pub const COL_OUTPUT0_L2: usize = 14;
pub const COL_OUTPUT0_L3: usize = 15;
// memory
pub const COL_MEM_OFFSET: usize = 16;
pub const COL_MEM_VALUE_L0: usize = 17;
pub const COL_MEM_VALUE_L1: usize = 18;
pub const COL_MEM_VALUE_L2: usize = 19;
pub const COL_MEM_VALUE_L3: usize = 20;
// selectors
pub const COL_INSN_TYPE: usize = 21;
pub const COL_FUNCT: usize = 22;
// immediate (PUSH value): U256 as 4 limbs
pub const COL_IMMEDIATE_L0: usize = 23;
pub const COL_IMMEDIATE_L1: usize = 24;
pub const COL_IMMEDIATE_L2: usize = 25;
pub const COL_IMMEDIATE_L3: usize = 26;
// aux0: U256 as 4 limbs (overflow, carry, intermediate)
pub const COL_AUX0_L0: usize = 27;
pub const COL_AUX0_L1: usize = 28;
pub const COL_AUX0_L2: usize = 29;
pub const COL_AUX0_L3: usize = 30;
// aux1: U256 as 4 limbs (borrow, remainder)
pub const COL_AUX1_L0: usize = 31;
pub const COL_AUX1_L1: usize = 32;
pub const COL_AUX1_L2: usize = 33;
pub const COL_AUX1_L3: usize = 34;
// Selector columns: one-hot encoding of (insn_type, funct) groups (41 selectors)
pub const COL_SEL_STOP: usize = 35;
pub const COL_SEL_ARITH_ADD: usize = 36;
pub const COL_SEL_ARITH_SUB: usize = 37;
pub const COL_SEL_ARITH_MUL: usize = 38;
pub const COL_SEL_ARITH_DIV: usize = 39;
pub const COL_SEL_MOD: usize = 40;
pub const COL_SEL_SDIV: usize = 41;
pub const COL_SEL_SMOD: usize = 42;
pub const COL_SEL_ADDMOD: usize = 43;
pub const COL_SEL_MULMOD: usize = 44;
pub const COL_SEL_EXP: usize = 45;
pub const COL_SEL_SIGNEXTEND: usize = 46;
pub const COL_SEL_LT: usize = 47;
pub const COL_SEL_GT: usize = 48;
pub const COL_SEL_EQ: usize = 49;
pub const COL_SEL_ISZERO: usize = 50;
pub const COL_SEL_COMPARE_OTHER: usize = 51;
pub const COL_SEL_AND: usize = 52;
pub const COL_SEL_OR: usize = 53;
pub const COL_SEL_XOR: usize = 54;
pub const COL_SEL_BITWISE_OTHER: usize = 55;
pub const COL_SEL_SHL: usize = 56;
pub const COL_SEL_SHR: usize = 57;
pub const COL_SEL_SAR: usize = 58;
pub const COL_SEL_KECCAK: usize = 59;
pub const COL_SEL_ENV: usize = 60;
pub const COL_SEL_BLOCK: usize = 61;
pub const COL_SEL_PUSH: usize = 62;
pub const COL_SEL_DUP: usize = 63;
pub const COL_SEL_POP: usize = 64;
pub const COL_SEL_SWAP: usize = 65;
pub const COL_SEL_STACK_OTHER: usize = 66;
pub const COL_SEL_MLOAD: usize = 67;
pub const COL_SEL_MSTORE: usize = 68;
pub const COL_SEL_MSTORE8: usize = 69;
pub const COL_SEL_MSIZE: usize = 70;
pub const COL_SEL_MEMORY_OTHER: usize = 71;
pub const COL_SEL_STORAGE: usize = 72;
pub const COL_SEL_JUMP: usize = 73;
pub const COL_SEL_LOG: usize = 74;
pub const COL_SEL_CALL: usize = 75;
// Next PC for cross-row PC continuity constraint
pub const COL_NEXT_PC: usize = 76;

// ─── SIGNEXTEND case selectors and witness columns ───────────────────────
// 32 case selectors: one is 1 when sel_signextend=1, picking the byte
// position b (b=0..30, or b>=31 for passthrough). At most one is 1 per row;
// all are 0 when sel_signextend=0. Binding: sel_signextend = Σ sel_se_k.
pub const COL_SEL_SE_0: usize = 77;
// … COL_SEL_SE_0 + k for k = 0..30
// COL_SEL_SE_0 + 31 = passthrough (b >= 31)
pub const COL_SEL_SE_GE31: usize = 108;

/// 8-byte decomposition of the input1 limb containing the sign byte (mixed limb).
/// When sel_signextend=1 and b=0..30, these must equal the bytes of input1_l[b/8]
/// (little-endian: byte_0 is LSB of limb, byte_7 is MSB). Each is range-checked
/// to [0, 255] via the 8-bit range table (LogUp).
pub const COL_SE_BYTE_0: usize = 109;
pub const COL_SE_BYTE_1: usize = 110;
pub const COL_SE_BYTE_2: usize = 111;
pub const COL_SE_BYTE_3: usize = 112;
pub const COL_SE_BYTE_4: usize = 113;
pub const COL_SE_BYTE_5: usize = 114;
pub const COL_SE_BYTE_6: usize = 115;
pub const COL_SE_BYTE_7: usize = 116;

/// Low 7 bits of the sign byte (byte at position b%8 within the mixed limb).
/// Must satisfy se_sign_byte = sign_bit * 128 + se_low7, with se_low7 ∈ [0, 127].
pub const COL_SE_LOW7: usize = 117;

/// Binary witness: the sign bit (top bit of sign byte). Used to fill high bytes
/// of the output with either 0 (sign_bit=0) or 0xFF (sign_bit=1).
pub const COL_SE_SIGN_BIT: usize = 118;

// ─── MULMOD witness columns ──────────────────────────────────────────────
// MULMOD(a, b, n) = (a*b) mod n. Stack layout:
//   input0 = a, input1 = b, immediate = n (captured by inspector),
//   output0 = r. Algebraic form: p = a*b (8 limbs), p = q*n + r, r < n.
// Special case n == 0 forces r = 0.

/// Quotient q: 4 u64 limbs (256-bit unsigned).
pub const COL_MULMOD_Q_L0: usize = 119;
pub const COL_MULMOD_Q_L1: usize = 120;
pub const COL_MULMOD_Q_L2: usize = 121;
pub const COL_MULMOD_Q_L3: usize = 122;
/// 512-bit product p = a*b: 8 u64 limbs (little-endian).
pub const COL_MULMOD_P_L0: usize = 123;
pub const COL_MULMOD_P_L1: usize = 124;
pub const COL_MULMOD_P_L2: usize = 125;
pub const COL_MULMOD_P_L3: usize = 126;
pub const COL_MULMOD_P_L4: usize = 127;
pub const COL_MULMOD_P_L5: usize = 128;
pub const COL_MULMOD_P_L6: usize = 129;
pub const COL_MULMOD_P_L7: usize = 130;
/// Carries for a*b schoolbook: carry out of equation positions 0..6 (7 total).
pub const COL_MULMOD_PC_0: usize = 131;
pub const COL_MULMOD_PC_1: usize = 132;
pub const COL_MULMOD_PC_2: usize = 133;
pub const COL_MULMOD_PC_3: usize = 134;
pub const COL_MULMOD_PC_4: usize = 135;
pub const COL_MULMOD_PC_5: usize = 136;
pub const COL_MULMOD_PC_6: usize = 137;
/// Carries for q*n + r = p schoolbook: carry out of positions 0..6 (7 total).
pub const COL_MULMOD_SC_0: usize = 138;
pub const COL_MULMOD_SC_1: usize = 139;
pub const COL_MULMOD_SC_2: usize = 140;
pub const COL_MULMOD_SC_3: usize = 141;
pub const COL_MULMOD_SC_4: usize = 142;
pub const COL_MULMOD_SC_5: usize = 143;
pub const COL_MULMOD_SC_6: usize = 144;
/// Slack limbs: n - r - 1 = slack with borrow chain (proves r < n when n != 0).
pub const COL_MULMOD_SLACK_L0: usize = 145;
pub const COL_MULMOD_SLACK_L1: usize = 146;
pub const COL_MULMOD_SLACK_L2: usize = 147;
pub const COL_MULMOD_SLACK_L3: usize = 148;
/// Borrow bits for the r < n slack chain (4 binary witnesses).
pub const COL_MULMOD_SLACK_B0: usize = 149;
pub const COL_MULMOD_SLACK_B1: usize = 150;
pub const COL_MULMOD_SLACK_B2: usize = 151;
pub const COL_MULMOD_SLACK_B3: usize = 152;
/// Binary witness: 1 iff n == 0.
pub const COL_MULMOD_N_IS_ZERO: usize = 153;

// ─── ADDMOD witness columns ──────────────────────────────────────────────
// ADDMOD(a, b, n) = (a+b) mod n. Stack layout:
//   input0 = a, input1 = b, immediate = n (captured by inspector),
//   output0 = r. Algebraic form: a + b = s_low + s_high·2^256 (257 bits),
//   s = q·n + r with 0 ≤ r < n. Special case n == 0 forces r = 0.

/// Quotient q: 4 u64 limbs (256-bit unsigned).
pub const COL_ADDMOD_Q_L0: usize = 154;
pub const COL_ADDMOD_Q_L1: usize = 155;
pub const COL_ADDMOD_Q_L2: usize = 156;
pub const COL_ADDMOD_Q_L3: usize = 157;
/// Low 256 bits of a + b: 4 u64 limbs (little-endian).
pub const COL_ADDMOD_S_LOW_L0: usize = 158;
pub const COL_ADDMOD_S_LOW_L1: usize = 159;
pub const COL_ADDMOD_S_LOW_L2: usize = 160;
pub const COL_ADDMOD_S_LOW_L3: usize = 161;
/// Top carry bit of a + b (binary).
pub const COL_ADDMOD_S_HIGH: usize = 162;
/// Carry bits for the a + b limb chain (4 binary witnesses).
pub const COL_ADDMOD_AB_CARRY_0: usize = 163;
pub const COL_ADDMOD_AB_CARRY_1: usize = 164;
pub const COL_ADDMOD_AB_CARRY_2: usize = 165;
pub const COL_ADDMOD_AB_CARRY_3: usize = 166;
/// Product q·n as 8-limb little-endian.
pub const COL_ADDMOD_QN_P_L0: usize = 167;
pub const COL_ADDMOD_QN_P_L1: usize = 168;
pub const COL_ADDMOD_QN_P_L2: usize = 169;
pub const COL_ADDMOD_QN_P_L3: usize = 170;
pub const COL_ADDMOD_QN_P_L4: usize = 171;
pub const COL_ADDMOD_QN_P_L5: usize = 172;
pub const COL_ADDMOD_QN_P_L6: usize = 173;
pub const COL_ADDMOD_QN_P_L7: usize = 174;
/// Carries for q·n schoolbook (7 positions 0..6).
pub const COL_ADDMOD_QN_CARRY_0: usize = 175;
pub const COL_ADDMOD_QN_CARRY_1: usize = 176;
pub const COL_ADDMOD_QN_CARRY_2: usize = 177;
pub const COL_ADDMOD_QN_CARRY_3: usize = 178;
pub const COL_ADDMOD_QN_CARRY_4: usize = 179;
pub const COL_ADDMOD_QN_CARRY_5: usize = 180;
pub const COL_ADDMOD_QN_CARRY_6: usize = 181;
/// Carries for q·n + r = s_low + s_high·2^256 chain (9 positions 0..8).
pub const COL_ADDMOD_SUM_CARRY_0: usize = 182;
pub const COL_ADDMOD_SUM_CARRY_1: usize = 183;
pub const COL_ADDMOD_SUM_CARRY_2: usize = 184;
pub const COL_ADDMOD_SUM_CARRY_3: usize = 185;
pub const COL_ADDMOD_SUM_CARRY_4: usize = 186;
pub const COL_ADDMOD_SUM_CARRY_5: usize = 187;
pub const COL_ADDMOD_SUM_CARRY_6: usize = 188;
pub const COL_ADDMOD_SUM_CARRY_7: usize = 189;
pub const COL_ADDMOD_SUM_CARRY_8: usize = 190;
/// Slack limbs: n - r - 1 = slack with borrow chain (proves r < n when n != 0).
pub const COL_ADDMOD_SLACK_L0: usize = 191;
pub const COL_ADDMOD_SLACK_L1: usize = 192;
pub const COL_ADDMOD_SLACK_L2: usize = 193;
pub const COL_ADDMOD_SLACK_L3: usize = 194;
/// Borrow bits for the r < n slack chain (4 binary witnesses).
pub const COL_ADDMOD_SLACK_B0: usize = 195;
pub const COL_ADDMOD_SLACK_B1: usize = 196;
pub const COL_ADDMOD_SLACK_B2: usize = 197;
pub const COL_ADDMOD_SLACK_B3: usize = 198;
/// Binary witness: 1 iff n == 0.
pub const COL_ADDMOD_N_IS_ZERO: usize = 199;

// ─── Frame-stack columns (CALL/RETURN frame state) ──────────────────────
//
// Each row carries the *current* frame-state (the frame the executing
// instruction belongs to). Cross-row transitions on rows where
// sel_call_push_frame / sel_call_return are set verify that the next row's
// frame-state correctly reflects a CALL push or RETURN pop.
//
// For the umbrella `sel_call` (CREATE/CALLCODE/DELEGATECALL/CREATE2/STATICCALL/
// REVERT) the frame columns are still oracles — no algebraic constraints are
// emitted in this first-pass. Subsequent passes will tighten those.

/// Current frame depth (u32 stored as scalar). 0 = top-level.
pub const COL_FRAME_DEPTH: usize = 200;
/// Caller address as 4 u64 limbs (low 256 bits; addresses fit in 160 bits so
/// limbs 2/3 are zero in practice).
pub const COL_FRAME_CALLER_L0: usize = 201;
pub const COL_FRAME_CALLER_L1: usize = 202;
pub const COL_FRAME_CALLER_L2: usize = 203;
pub const COL_FRAME_CALLER_L3: usize = 204;
/// Callee address (the contract whose code is currently executing) as 4 limbs.
pub const COL_FRAME_CALLEE_L0: usize = 205;
pub const COL_FRAME_CALLEE_L1: usize = 206;
pub const COL_FRAME_CALLEE_L2: usize = 207;
pub const COL_FRAME_CALLEE_L3: usize = 208;
/// Value transferred into this frame (U256 as 4 limbs).
pub const COL_FRAME_VALUE_L0: usize = 209;
pub const COL_FRAME_VALUE_L1: usize = 210;
pub const COL_FRAME_VALUE_L2: usize = 211;
pub const COL_FRAME_VALUE_L3: usize = 212;
/// Gas allotted to the current frame.
pub const COL_FRAME_GAS: usize = 213;
/// Return PC: the PC the executor will resume at in the *caller* once this
/// frame returns. For the top-level frame this column is 0.
pub const COL_FRAME_RETURN_PC: usize = 214;
/// Memory offset (in caller's memory) where returndata should be written.
pub const COL_FRAME_RETURN_OFFSET: usize = 215;
/// Maximum size (bytes) of returndata to copy back to the caller.
pub const COL_FRAME_RETURN_SIZE: usize = 216;

// ─── Split-call selectors ────────────────────────────────────────────────
//
// The umbrella `sel_call` selector at index 75 historically fired for ALL of
// 0xF0/0xF1/0xF2/0xF3/0xF4/0xF5/0xFA. We now split out 0xF1 (CALL) and 0xF3
// (RETURN) into their own selectors so they can carry algebraic frame-stack
// transition constraints. The remaining 5 opcodes still set `sel_call`
// (oracle, body 0). Selector sum-to-one over the now 43 selectors.

/// Selector for opcode 0xF1 (CALL): pushes a new frame on the frame stack.
pub const COL_SEL_CALL_PUSH_FRAME: usize = 217;
/// Selector for opcode 0xF3 (RETURN): pops the current frame.
pub const COL_SEL_CALL_RETURN: usize = 218;

// ─── Remaining call-family selectors ────────────────────────────────────
//
// CREATE, CALLCODE, DELEGATECALL, CREATE2, STATICCALL push a new frame just
// like CALL but with different parent-vs-child caller / value relationships.
// REVERT pops the current frame like RETURN. Each gets its own selector so
// the cross-row frame-state transition can carry the correct algebraic
// relations.
pub const COL_SEL_CREATE: usize = 219;       // 0xF0
pub const COL_SEL_CALLCODE: usize = 220;     // 0xF2
pub const COL_SEL_DELEGATECALL: usize = 221; // 0xF4
pub const COL_SEL_CREATE2: usize = 222;      // 0xF5
pub const COL_SEL_STATICCALL: usize = 223;   // 0xFA
pub const COL_SEL_REVERT: usize = 224;       // 0xFD

/// Sticky static-mode flag. Set to 1 by STATICCALL on the pushed (child) frame
/// and propagated through nested CALL/CALLCODE/DELEGATECALL/STATICCALL frames
/// (a static frame can never escape back to non-static). On the top-level
/// frame this column is 0 unless the transaction was launched from a STATICCALL
/// (which never happens in normal Ethereum execution; left at 0).
pub const COL_FRAME_STATIC: usize = 225;

/// Oracle column populated by the inspector with the contract address that
/// CREATE/CREATE2 just derived (Keccak of RLP-encoded sender+nonce, or
/// Keccak of salt+initcode hash). The shifted-constraint `frame_callee(ω·X) -
/// create_address_hint(X) = 0` binds the next frame's callee to this hint.
/// The actual address derivation is NOT verified algebraically at this layer
/// — that's a future cross-AIR linkage with keccak_constraints (mirroring how
/// MPT/SSZ delegate hash checks).
pub const COL_CREATE_ADDRESS_HINT_L0: usize = 226;
pub const COL_CREATE_ADDRESS_HINT_L1: usize = 227;
pub const COL_CREATE_ADDRESS_HINT_L2: usize = 228;
pub const COL_CREATE_ADDRESS_HINT_L3: usize = 229;

/// Oracle column populated by the inspector with the sender account's
/// live nonce at the moment of the CREATE opcode. Used by the
/// cross-AIR LogUp linkage to [`metavm_zkp::evm_create_rlp_air`] so
/// the gadget's RLP encoding's nonce input is tied to what the EVM
/// trace actually saw. Zero on non-CREATE rows. The CREATE2 opcode
/// does NOT use this column (CREATE2's address derivation does not
/// involve the nonce).
pub const COL_CREATE_NONCE_HINT: usize = 230;

/// Witness oracle column: 1 iff the row is a STOP at depth ≥ 1 (i.e.,
/// init code completion or callee STOP-as-implicit-RETURN). Used as a
/// pop selector in the LIFO frame-stack permutation argument and to
/// gate off PC continuity / frame-state preservation on these rows
/// (where the next row is in a popped-into frame). Distinct from
/// `sel_stop` which fires on EVERY STOP including top-level (depth=0)
/// STOP that ends the transaction with no parent frame.
pub const COL_SEL_STOP_POP: usize = 231;

/// Oracle columns populated by the inspector with the CREATE2 `salt`
/// argument (from stack position 3 at the CREATE2 opcode). 4 LE u64
/// limbs encoding the 32-byte salt. Zero on non-CREATE2 rows. Consumed
/// by the cross-AIR LogUp linkage to `EvmCreate2InputAir` so the
/// gadget's canonical pre-image construction `0xff || sender || salt
/// || initcode_hash` is tied to what the EVM trace actually saw.
pub const COL_CREATE2_SALT_HINT_L0: usize = 232;
pub const COL_CREATE2_SALT_HINT_L1: usize = 233;
pub const COL_CREATE2_SALT_HINT_L2: usize = 234;
pub const COL_CREATE2_SALT_HINT_L3: usize = 235;

/// Oracle columns populated by the inspector with `keccak256(initcode)`
/// at the CREATE2 opcode site. 4 LE u64 limbs encoding the 32-byte
/// digest of the init code bytes at memory[offset..offset+size]. Zero
/// on non-CREATE2 rows.
pub const COL_CREATE2_INITCODE_HASH_HINT_L0: usize = 236;
pub const COL_CREATE2_INITCODE_HASH_HINT_L1: usize = 237;
pub const COL_CREATE2_INITCODE_HASH_HINT_L2: usize = 238;
pub const COL_CREATE2_INITCODE_HASH_HINT_L3: usize = 239;

/// Selector for the BYTE opcode (0x1A). Split out from
/// `COL_SEL_BITWISE_OTHER` to fix the latent soundness bug where the
/// NOT constraint at slot 55 (gated by `sel_bitwise_other`) would
/// incorrectly fire on BYTE rows: BYTE has different semantics from
/// NOT and the constraint body would be non-zero, making any BYTE-
/// containing trace unprovable. With `sel_byte_op` carrying its own
/// selector, `sel_bitwise_other` now fires only on NOT in practice
/// (the only opcode left in INSN_BITWISE with no dedicated selector).
/// BYTE itself remains an oracle for now (Phase A3 follow-up will add
/// the byte-extraction algebraic constraint).
pub const COL_SEL_BYTE_OP: usize = 240;

/// **Phase A2 step 1b**: per-opcode storage selectors split out from
/// the umbrella `COL_SEL_STORAGE`. Unlike the byte-op split, these are
/// AUXILIARY witness columns — they do NOT replace `COL_SEL_STORAGE`
/// in the one-hot selector list. They fire when `sel_storage == 1`
/// AND opcode matches: `sel_sload = 1` iff opcode == 0x54,
/// `sel_sstore = 1` iff opcode == 0x55. On TLOAD (0x5C) / TSTORE
/// (0x5D) rows, both remain 0 while `sel_storage = 1`.
///
/// Future algebraic constraints (step 1b-strict) will bind them:
///   `sel_sload * (opcode - 0x54) = 0`,
///   `sel_sstore * (opcode - 0x55) = 0`,
///   `sel_sload * (1 - sel_storage) = 0`, etc.
/// The current implementation treats them as oracle witness columns,
/// honest by inspector convention; the cross-AIR LogUp linkage to the
/// storage_access_air gadget provides multiset-equality binding even
/// without the must-fire constraint (a malicious prover that omits a
/// gadget row for a real SLOAD would still have closure mismatch).
pub const COL_SEL_SLOAD: usize = 241;
pub const COL_SEL_SSTORE: usize = 242;

/// **#53 step 2 / #54 — per-opcode BLOCK selectors** split out from
/// the umbrella `COL_SEL_BLOCK` (col 61). Same pattern as
/// SLOAD/SSTORE: auxiliary witness columns gating cross-AIR LogUp
/// linkages from EVM main BLOCK opcodes to `block_header_air`. Each
/// fires iff `sel_block == 1` AND `opcode == matching value`.
///
/// - `sel_coinbase`     iff opcode == 0x41 (COINBASE)
/// - `sel_timestamp`    iff opcode == 0x42 (TIMESTAMP)
/// - `sel_number`       iff opcode == 0x43 (NUMBER)
/// - `sel_gaslimit`     iff opcode == 0x45 (GASLIMIT)
/// - `sel_basefee`      iff opcode == 0x48 (BASEFEE)
///
/// - `sel_prevrandao`   iff opcode == 0x44 (PREVRANDAO)
/// - `sel_chainid`      iff opcode == 0x46 (CHAINID)
/// - `sel_blockhash`    iff opcode == 0x40 (BLOCKHASH)
/// - `sel_selfbalance`  iff opcode == 0x47 (SELFBALANCE)
///
/// Honest-by-inspector convention; algebraic must-fire
/// constraints are deferred (same as SLOAD/SSTORE pattern).
pub const COL_SEL_TIMESTAMP: usize = 243;
pub const COL_SEL_NUMBER: usize = 244;
pub const COL_SEL_GASLIMIT: usize = 245;
pub const COL_SEL_BASEFEE: usize = 246;
pub const COL_SEL_COINBASE: usize = 247;
pub const COL_SEL_PREVRANDAO: usize = 248;
pub const COL_SEL_CHAINID: usize = 249;
pub const COL_SEL_BLOCKHASH: usize = 250;
pub const COL_SEL_SELFBALANCE: usize = 251;

/// EVM instruction type selectors.
pub const INSN_STOP: u8 = 0;
pub const INSN_ARITH: u8 = 1;
pub const INSN_COMPARE: u8 = 2;
pub const INSN_BITWISE: u8 = 3;
pub const INSN_KECCAK: u8 = 4;
pub const INSN_ENV: u8 = 5;
pub const INSN_BLOCK: u8 = 6;
pub const INSN_STACK: u8 = 7;
pub const INSN_MEMORY: u8 = 8;
pub const INSN_STORAGE: u8 = 9;
pub const INSN_JUMP: u8 = 10;
pub const INSN_LOG: u8 = 11;
pub const INSN_CALL: u8 = 12;

/// Funct codes for arithmetic sub-variants.
pub const FUNCT_ADD: u8 = 0;
pub const FUNCT_MUL: u8 = 1;
pub const FUNCT_SUB: u8 = 2;
pub const FUNCT_DIV: u8 = 3;
pub const FUNCT_SDIV: u8 = 4;
pub const FUNCT_MOD: u8 = 5;
pub const FUNCT_SMOD: u8 = 6;
pub const FUNCT_ADDMOD: u8 = 7;
pub const FUNCT_MULMOD: u8 = 8;
pub const FUNCT_EXP: u8 = 9;
pub const FUNCT_SIGNEXTEND: u8 = 10;

/// Funct codes for comparison sub-variants.
pub const FUNCT_LT: u8 = 0;
pub const FUNCT_GT: u8 = 1;
pub const FUNCT_SLT: u8 = 2;
pub const FUNCT_SGT: u8 = 3;
pub const FUNCT_EQ: u8 = 4;
pub const FUNCT_ISZERO: u8 = 5;

/// Funct codes for bitwise sub-variants.
pub const FUNCT_AND: u8 = 0;
pub const FUNCT_OR: u8 = 1;
pub const FUNCT_XOR: u8 = 2;
pub const FUNCT_NOT: u8 = 3;
pub const FUNCT_BYTE: u8 = 4;
pub const FUNCT_SHL: u8 = 5;
pub const FUNCT_SHR: u8 = 6;
pub const FUNCT_SAR: u8 = 7;

/// Funct codes for stack operations.
pub const FUNCT_POP: u8 = 0;
pub const FUNCT_PUSH: u8 = 1;
pub const FUNCT_DUP: u8 = 2;
pub const FUNCT_SWAP: u8 = 3;

/// Funct codes for memory operations.
pub const FUNCT_MLOAD: u8 = 0;
pub const FUNCT_MSTORE: u8 = 1;
pub const FUNCT_MSTORE8: u8 = 2;
pub const FUNCT_MSIZE: u8 = 3;

/// Funct codes for jump operations.
pub const FUNCT_JUMP: u8 = 0;
pub const FUNCT_JUMPI: u8 = 1;
pub const FUNCT_JUMPDEST: u8 = 2;
pub const FUNCT_PC: u8 = 3;

/// Funct codes for storage operations.
pub const FUNCT_SLOAD: u8 = 0;
pub const FUNCT_SSTORE: u8 = 1;
pub const FUNCT_TLOAD: u8 = 2;
pub const FUNCT_TSTORE: u8 = 3;

/// Funct codes for BLOCK opcodes (matches `opcode - 0x40` from
/// `classify_opcode`). `0x40..=0x48 => (INSN_BLOCK, opcode - 0x40)`.
pub const FUNCT_BLOCKHASH: u8 = 0;
pub const FUNCT_COINBASE: u8 = 1;
pub const FUNCT_TIMESTAMP: u8 = 2;
pub const FUNCT_NUMBER: u8 = 3;
pub const FUNCT_PREVRANDAO: u8 = 4;
pub const FUNCT_GASLIMIT: u8 = 5;
pub const FUNCT_CHAINID: u8 = 6;
pub const FUNCT_SELFBALANCE: u8 = 7;
pub const FUNCT_BASEFEE: u8 = 8;

/// Decompose a U256 (represented as [u64; 4] in little-endian) into 4 u64 limbs.
pub fn u256_to_limbs(limbs: [u64; 4]) -> [u64; 4] {
    limbs // Already in the right format for revm's Uint<256, 4>
}

/// Classify an EVM opcode into (insn_type, funct) pair.
pub fn classify_opcode(opcode: u8) -> (u8, u8) {
    match opcode {
        0x00 => (INSN_STOP, 0),           // STOP
        0xFE => (INSN_STOP, 1),           // INVALID
        0xFD => (INSN_STOP, 2),           // REVERT
        0xFF => (INSN_STOP, 3),           // SELFDESTRUCT

        0x01 => (INSN_ARITH, FUNCT_ADD),
        0x02 => (INSN_ARITH, FUNCT_MUL),
        0x03 => (INSN_ARITH, FUNCT_SUB),
        0x04 => (INSN_ARITH, FUNCT_DIV),
        0x05 => (INSN_ARITH, FUNCT_SDIV),
        0x06 => (INSN_ARITH, FUNCT_MOD),
        0x07 => (INSN_ARITH, FUNCT_SMOD),
        0x08 => (INSN_ARITH, FUNCT_ADDMOD),
        0x09 => (INSN_ARITH, FUNCT_MULMOD),
        0x0A => (INSN_ARITH, FUNCT_EXP),
        0x0B => (INSN_ARITH, FUNCT_SIGNEXTEND),

        0x10 => (INSN_COMPARE, FUNCT_LT),
        0x11 => (INSN_COMPARE, FUNCT_GT),
        0x12 => (INSN_COMPARE, FUNCT_SLT),
        0x13 => (INSN_COMPARE, FUNCT_SGT),
        0x14 => (INSN_COMPARE, FUNCT_EQ),
        0x15 => (INSN_COMPARE, FUNCT_ISZERO),

        0x16 => (INSN_BITWISE, FUNCT_AND),
        0x17 => (INSN_BITWISE, FUNCT_OR),
        0x18 => (INSN_BITWISE, FUNCT_XOR),
        0x19 => (INSN_BITWISE, FUNCT_NOT),
        0x1A => (INSN_BITWISE, FUNCT_BYTE),
        0x1B => (INSN_BITWISE, FUNCT_SHL),
        0x1C => (INSN_BITWISE, FUNCT_SHR),
        0x1D => (INSN_BITWISE, FUNCT_SAR),

        0x20 => (INSN_KECCAK, 0),         // SHA3

        0x30..=0x3F => (INSN_ENV, opcode - 0x30),
        0x40..=0x48 => (INSN_BLOCK, opcode - 0x40),

        0x50 => (INSN_STACK, FUNCT_POP),
        0x60..=0x7F => (INSN_STACK, FUNCT_PUSH),  // PUSH1-PUSH32
        0x80..=0x8F => (INSN_STACK, FUNCT_DUP),   // DUP1-DUP16
        0x90..=0x9F => (INSN_STACK, FUNCT_SWAP),   // SWAP1-SWAP16

        0x51 => (INSN_MEMORY, FUNCT_MLOAD),
        0x52 => (INSN_MEMORY, FUNCT_MSTORE),
        0x53 => (INSN_MEMORY, FUNCT_MSTORE8),
        0x59 => (INSN_MEMORY, FUNCT_MSIZE),

        0x54 => (INSN_STORAGE, 0),         // SLOAD
        0x55 => (INSN_STORAGE, 1),         // SSTORE
        0x5C => (INSN_STORAGE, 2),         // TLOAD
        0x5D => (INSN_STORAGE, 3),         // TSTORE

        0x56 => (INSN_JUMP, FUNCT_JUMP),
        0x57 => (INSN_JUMP, FUNCT_JUMPI),
        0x5B => (INSN_JUMP, FUNCT_JUMPDEST),
        0x58 => (INSN_JUMP, FUNCT_PC),
        0x5A => (INSN_JUMP, 4), // GAS

        0xA0..=0xA4 => (INSN_LOG, opcode - 0xA0),

        0xF0 | 0xF1 | 0xF2 | 0xF4 | 0xF5 | 0xF3 | 0xFA => (INSN_CALL, opcode - 0xF0),

        _ => (INSN_STOP, 0xFF),  // Unknown opcodes treated as stop
    }
}

/// Frame-stack state recorded on every row of the EVM trace.
///
/// This snapshots the *current* frame the executing instruction belongs to.
/// On a CALL row the next row's frame state must be the freshly-pushed frame;
/// on a RETURN row the next row's frame state must match the popped frame.
#[derive(Clone, Debug, Default)]
pub struct FrameState {
    /// 0 = top-level transaction. Increments on each CALL.
    pub depth: u64,
    /// Address of the contract that initiated this frame. As 4×u64 little-endian
    /// limbs; high limbs are zero for normal 160-bit Ethereum addresses.
    pub caller: [u64; 4],
    /// Address of the contract whose code is currently executing.
    pub callee: [u64; 4],
    /// Value (wei) transferred into this frame.
    pub value: [u64; 4],
    /// Gas allotted to this frame.
    pub gas: u64,
    /// PC in the caller to resume at when this frame returns. 0 for top-level.
    pub return_pc: u64,
    /// Offset in caller's memory where returndata should be written.
    pub return_offset: u64,
    /// Maximum bytes of returndata to copy back to the caller.
    pub return_size: u64,
    /// Sticky static-mode flag: 1 iff this frame was entered via STATICCALL or
    /// is the descendant of a STATICCALL frame. Once 1 it never goes back to 0
    /// in any nested child frame.
    pub is_static: u64,
}

/// One row of the EVM execution trace.
#[derive(Clone, Debug)]
pub struct EvmTraceRow {
    pub step: u64,
    pub pc: u64,
    pub opcode: u8,
    pub gas_remaining: u64,
    pub stack_depth: u64,
    pub input0: [u64; 4],
    pub input1: [u64; 4],
    pub output0: [u64; 4],
    pub mem_offset: u64,
    pub mem_value: [u64; 4],
    pub insn_type: u8,
    pub funct: u8,
    pub immediate: [u64; 4],
    pub aux0: [u64; 4],
    pub aux1: [u64; 4],
    /// Expected PC of the next instruction (for PC continuity constraint).
    /// Set by the executor/inspector based on opcode semantics.
    pub next_pc: u64,
    /// Current frame state for this row.
    pub frame: FrameState,
    /// Oracle hint populated by the inspector on CREATE/CREATE2 rows with
    /// the just-derived contract address (4 limbs, little-endian). Zero on
    /// all non-CREATE rows.
    pub create_address_hint: [u64; 4],
    /// Oracle hint populated by the inspector on CREATE rows with the
    /// sender account's nonce at the moment of the CREATE opcode (read
    /// from the live revm journal). Zero on all non-CREATE rows. Used
    /// by the cross-AIR LogUp linkage to the CREATE pre-image RLP gadget
    /// (`metavm_zkp::evm_create_rlp_air`).
    pub create_nonce_hint: u64,
    /// Computed at row build time: 1 iff opcode == STOP (0x00) AND
    /// frame.depth ≥ 1 (init code completion or callee
    /// STOP-as-implicit-RETURN). Used as a pop_selector in the LIFO
    /// frame-stack permutation argument and to gate off PC continuity /
    /// frame-state preservation on these rows.
    pub sel_stop_pop: u64,
    /// Oracle hint populated by the inspector on CREATE2 rows with the
    /// `salt` argument (stack[3]) as 4 LE u64 limbs. Zero on non-CREATE2
    /// rows. Consumed by the cross-AIR LogUp linkage to the CREATE2
    /// pre-image gadget (`metavm_zkp::evm_create2_input_air`).
    pub create2_salt_hint: [u64; 4],
    /// Oracle hint populated by the inspector on CREATE2 rows with
    /// `keccak256(initcode)` as 4 LE u64 limbs (matches `address_to_limbs`
    /// little-endian convention). Zero on non-CREATE2 rows.
    pub create2_initcode_hash_hint: [u64; 4],
    /// Per-tx/per-block context for ENV opcode algebraic binding (constant per row).
    pub tx_origin: [u64; 4],
    pub tx_gas_price: u64,
    pub tx_calldata_size: u64,
    pub tx_code_size: u64,
    pub returndata_size: u64,
}

/// MULMOD witness values for a single row. All zero unless the row is MULMOD.
///
/// Algebraic decomposition of MULMOD(a, b, n) = (a*b) mod n:
/// - `p[0..8]` = a*b as 512-bit value (8 little-endian limbs)
/// - `pc[0..7]` = carry chain for the a*b schoolbook (carries out of positions 0..6)
/// - `q[0..4]` = quotient such that p = q*n + r (with r = output)
/// - `sc[0..7]` = carry chain for q*n + r = p schoolbook
/// - `slack[0..4]` with `slack_borrow[0..4]` = non-borrow chain proving r < n
/// - `n_is_zero` = 1 iff n == 0
#[derive(Clone, Debug, Default)]
pub struct MulmodWitness {
    pub q: [u64; 4],
    pub p: [u64; 8],
    pub pc: [u64; 7],
    pub sc: [u64; 7],
    pub slack: [u64; 4],
    pub slack_borrow: [u64; 4],
    pub n_is_zero: u64,
}

/// ADDMOD witness values for a single row. All zero unless the row is ADDMOD.
///
/// Algebraic decomposition of ADDMOD(a, b, n) = (a+b) mod n:
/// - `s_low[0..4]` = low 256 bits of a+b
/// - `s_high` = top carry bit of a+b (binary)
/// - `ab_carry[0..4]` = per-limb carry chain for a+b (each binary)
/// - `q[0..4]` = quotient such that s = q*n + r (with r = output), s the 257-bit sum
/// - `qn_p[0..8]` = 8-limb product q*n
/// - `qn_carry[0..7]` = carry chain for the q*n schoolbook (positions 0..6)
/// - `sum_carry[0..9]` = carry chain for q*n + r = s_low + s_high*2^256 (positions 0..8)
/// - `slack[0..4]` with `slack_borrow[0..4]` = non-borrow chain proving r < n
/// - `n_is_zero` = 1 iff n == 0
#[derive(Clone, Debug, Default)]
pub struct AddmodWitness {
    pub q: [u64; 4],
    pub s_low: [u64; 4],
    pub s_high: u64,
    pub ab_carry: [u64; 4],
    pub qn_p: [u64; 8],
    pub qn_carry: [u64; 7],
    pub sum_carry: [u64; 9],
    pub slack: [u64; 4],
    pub slack_borrow: [u64; 4],
    pub n_is_zero: u64,
}

/// Column-oriented EVM trace.
pub struct EvmTraceColumns {
    pub step: Vec<u64>,
    pub pc: Vec<u64>,
    pub opcode: Vec<u64>,
    pub gas_remaining: Vec<u64>,
    pub stack_depth: Vec<u64>,
    pub input0: [Vec<u64>; 4],
    pub input1: [Vec<u64>; 4],
    pub output0: [Vec<u64>; 4],
    pub mem_offset: Vec<u64>,
    pub mem_value: [Vec<u64>; 4],
    pub insn_type: Vec<u64>,
    pub funct: Vec<u64>,
    pub immediate: [Vec<u64>; 4],
    pub aux0: [Vec<u64>; 4],
    pub aux1: [Vec<u64>; 4],
    pub sel_stop: Vec<u64>,
    pub sel_arith_add: Vec<u64>,
    pub sel_arith_sub: Vec<u64>,
    pub sel_arith_mul: Vec<u64>,
    pub sel_arith_div: Vec<u64>,
    pub sel_mod: Vec<u64>,
    pub sel_sdiv: Vec<u64>,
    pub sel_smod: Vec<u64>,
    pub sel_addmod: Vec<u64>,
    pub sel_mulmod: Vec<u64>,
    pub sel_exp: Vec<u64>,
    pub sel_signextend: Vec<u64>,
    pub sel_lt: Vec<u64>,
    pub sel_gt: Vec<u64>,
    pub sel_eq: Vec<u64>,
    pub sel_iszero: Vec<u64>,
    pub sel_compare_other: Vec<u64>,
    pub sel_and: Vec<u64>,
    pub sel_or: Vec<u64>,
    pub sel_xor: Vec<u64>,
    pub sel_bitwise_other: Vec<u64>,
    pub sel_shl: Vec<u64>,
    pub sel_shr: Vec<u64>,
    pub sel_sar: Vec<u64>,
    pub sel_keccak: Vec<u64>,
    pub sel_env: Vec<u64>,
    pub sel_block: Vec<u64>,
    pub sel_push: Vec<u64>,
    pub sel_dup: Vec<u64>,
    pub sel_pop: Vec<u64>,
    pub sel_swap: Vec<u64>,
    pub sel_stack_other: Vec<u64>,
    pub sel_mload: Vec<u64>,
    pub sel_mstore: Vec<u64>,
    pub sel_mstore8: Vec<u64>,
    pub sel_msize: Vec<u64>,
    pub sel_memory_other: Vec<u64>,
    pub sel_storage: Vec<u64>,
    /// Phase A2 step 1b: SLOAD-specific auxiliary selector (gate for
    /// the storage_access_air cross-AIR LogUp).
    pub sel_sload: Vec<u64>,
    /// Phase A2 step 1b: SSTORE-specific auxiliary selector.
    pub sel_sstore: Vec<u64>,
    /// #53 step 2 / #54: per-opcode BLOCK aux selectors for the
    /// EVM ↔ block_header_air cross-AIR LogUp linkages.
    pub sel_timestamp: Vec<u64>,
    pub sel_number: Vec<u64>,
    pub sel_gaslimit: Vec<u64>,
    pub sel_basefee: Vec<u64>,
    pub sel_coinbase: Vec<u64>,
    pub sel_prevrandao: Vec<u64>,
    pub sel_chainid: Vec<u64>,
    pub sel_blockhash: Vec<u64>,
    pub sel_selfbalance: Vec<u64>,
    pub sel_log0: Vec<u64>,
    pub sel_log1: Vec<u64>,
    pub sel_log2: Vec<u64>,
    pub sel_log3: Vec<u64>,
    pub sel_log4: Vec<u64>,
    pub sel_address: Vec<u64>,
    pub sel_caller: Vec<u64>,
    pub sel_callvalue: Vec<u64>,
    pub sel_origin: Vec<u64>,
    pub sel_calldatasize: Vec<u64>,
    pub sel_codesize: Vec<u64>,
    pub sel_gasprice: Vec<u64>,
    pub tx_origin: [Vec<u64>; 4],
    pub tx_gas_price: Vec<u64>,
    pub tx_calldata_size: Vec<u64>,
    pub tx_code_size: Vec<u64>,
    pub returndata_size: Vec<u64>,
    pub sel_pc: Vec<u64>,
    pub sel_gas: Vec<u64>,
    pub sel_msize_op: Vec<u64>,
    pub sel_jump: Vec<u64>,
    pub sel_log: Vec<u64>,
    pub sel_call: Vec<u64>,
    pub next_pc: Vec<u64>,
    /// 32 SIGNEXTEND case selectors, indexed 0..=31. Index 31 = passthrough (b >= 31).
    pub sel_se: [Vec<u64>; 32],
    /// Byte decomposition of the input1 limb containing the sign byte.
    pub se_byte: [Vec<u64>; 8],
    /// Low 7 bits of the sign byte.
    pub se_low7: Vec<u64>,
    /// Top bit of the sign byte (binary).
    pub se_sign_bit: Vec<u64>,
    // ─── MULMOD witness columns ───────────────────────────────────────────
    /// MULMOD quotient q (4 limbs).
    pub mulmod_q: [Vec<u64>; 4],
    /// 512-bit product p = a*b (8 limbs, little-endian).
    pub mulmod_p: [Vec<u64>; 8],
    /// Carries for a*b schoolbook (positions 0..6).
    pub mulmod_pc: [Vec<u64>; 7],
    /// Carries for q*n + r = p schoolbook (positions 0..6).
    pub mulmod_sc: [Vec<u64>; 7],
    /// Slack = n - r - 1 (per-limb, non-borrow chain witnesses).
    pub mulmod_slack: [Vec<u64>; 4],
    /// Borrow bits for the r < n slack chain.
    pub mulmod_slack_borrow: [Vec<u64>; 4],
    /// Binary witness: 1 iff n == 0.
    pub mulmod_n_is_zero: Vec<u64>,
    // ─── ADDMOD witness columns ───────────────────────────────────────────
    /// ADDMOD quotient q (4 limbs).
    pub addmod_q: [Vec<u64>; 4],
    /// Low 256 bits of a + b (4 limbs).
    pub addmod_s_low: [Vec<u64>; 4],
    /// Top carry bit of a + b.
    pub addmod_s_high: Vec<u64>,
    /// Carry bits for the a + b limb chain (4 binary witnesses).
    pub addmod_ab_carry: [Vec<u64>; 4],
    /// 8-limb product q * n.
    pub addmod_qn_p: [Vec<u64>; 8],
    /// Carries for the q * n schoolbook (positions 0..6).
    pub addmod_qn_carry: [Vec<u64>; 7],
    /// Carries for q*n + r = s chain (9 positions 0..8).
    pub addmod_sum_carry: [Vec<u64>; 9],
    /// Slack = n - r - 1 (per-limb, non-borrow chain witnesses).
    pub addmod_slack: [Vec<u64>; 4],
    /// Borrow bits for the r < n slack chain.
    pub addmod_slack_borrow: [Vec<u64>; 4],
    /// Binary witness: 1 iff n == 0.
    pub addmod_n_is_zero: Vec<u64>,
    // ─── Frame-stack columns ──────────────────────────────────────────────
    pub frame_depth: Vec<u64>,
    pub frame_caller: [Vec<u64>; 4],
    pub frame_callee: [Vec<u64>; 4],
    pub frame_value: [Vec<u64>; 4],
    pub frame_gas: Vec<u64>,
    pub frame_return_pc: Vec<u64>,
    pub frame_return_offset: Vec<u64>,
    pub frame_return_size: Vec<u64>,
    /// Selector for opcode 0xF1 (CALL).
    pub sel_call_push_frame: Vec<u64>,
    /// Selector for opcode 0xF3 (RETURN).
    pub sel_call_return: Vec<u64>,
    /// Selector for opcode 0xF0 (CREATE).
    pub sel_create: Vec<u64>,
    /// Selector for opcode 0xF2 (CALLCODE).
    pub sel_callcode: Vec<u64>,
    /// Selector for opcode 0xF4 (DELEGATECALL).
    pub sel_delegatecall: Vec<u64>,
    /// Selector for opcode 0xF5 (CREATE2).
    pub sel_create2: Vec<u64>,
    /// Selector for opcode 0xFA (STATICCALL).
    pub sel_staticcall: Vec<u64>,
    /// Selector for opcode 0xFD (REVERT).
    pub sel_revert: Vec<u64>,
    /// Selector for opcode 0x1A (BYTE). Split out from
    /// `sel_bitwise_other` to fix a latent soundness bug — see
    /// [`COL_SEL_BYTE_OP`].
    pub sel_byte_op: Vec<u64>,
    /// Sticky static-mode flag (per-row snapshot from the current frame).
    pub frame_static: Vec<u64>,
    /// Oracle hint: contract address CREATE/CREATE2 just derived.
    pub create_address_hint: [Vec<u64>; 4],
    /// Oracle hint: sender account nonce at the moment of the CREATE
    /// opcode (zero on non-CREATE rows; CREATE2 also leaves it zero
    /// since CREATE2 address derivation does not use the nonce).
    pub create_nonce_hint: Vec<u64>,
    /// Witness oracle: 1 iff row is STOP at depth ≥ 1 (implicit frame pop).
    pub sel_stop_pop: Vec<u64>,
    /// Oracle hint: CREATE2 `salt` argument as 4 LE u64 limbs. Zero on
    /// non-CREATE2 rows.
    pub create2_salt_hint: [Vec<u64>; 4],
    /// Oracle hint: keccak256(initcode) for CREATE2 as 4 LE u64 limbs.
    /// Zero on non-CREATE2 rows.
    pub create2_initcode_hash_hint: [Vec<u64>; 4],
}

impl EvmTraceColumns {
    pub fn new() -> Self {
        EvmTraceColumns {
            step: Vec::new(),
            pc: Vec::new(),
            opcode: Vec::new(),
            gas_remaining: Vec::new(),
            stack_depth: Vec::new(),
            input0: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            input1: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            output0: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            mem_offset: Vec::new(),
            mem_value: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            insn_type: Vec::new(),
            funct: Vec::new(),
            immediate: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            aux0: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            aux1: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            sel_stop: Vec::new(),
            sel_arith_add: Vec::new(),
            sel_arith_sub: Vec::new(),
            sel_arith_mul: Vec::new(),
            sel_arith_div: Vec::new(),
            sel_mod: Vec::new(),
            sel_sdiv: Vec::new(),
            sel_smod: Vec::new(),
            sel_addmod: Vec::new(),
            sel_mulmod: Vec::new(),
            sel_exp: Vec::new(),
            sel_signextend: Vec::new(),
            sel_lt: Vec::new(),
            sel_gt: Vec::new(),
            sel_eq: Vec::new(),
            sel_iszero: Vec::new(),
            sel_compare_other: Vec::new(),
            sel_and: Vec::new(),
            sel_or: Vec::new(),
            sel_xor: Vec::new(),
            sel_bitwise_other: Vec::new(),
            sel_shl: Vec::new(),
            sel_shr: Vec::new(),
            sel_sar: Vec::new(),
            sel_keccak: Vec::new(),
            sel_env: Vec::new(),
            sel_block: Vec::new(),
            sel_push: Vec::new(),
            sel_dup: Vec::new(),
            sel_pop: Vec::new(),
            sel_swap: Vec::new(),
            sel_stack_other: Vec::new(),
            sel_mload: Vec::new(),
            sel_mstore: Vec::new(),
            sel_mstore8: Vec::new(),
            sel_msize: Vec::new(),
            sel_memory_other: Vec::new(),
            sel_storage: Vec::new(),
            sel_sload: Vec::new(),
            sel_sstore: Vec::new(),
            sel_timestamp: Vec::new(),
            sel_number: Vec::new(),
            sel_gaslimit: Vec::new(),
            sel_basefee: Vec::new(),
            sel_coinbase: Vec::new(),
            sel_prevrandao: Vec::new(),
            sel_chainid: Vec::new(),
            sel_blockhash: Vec::new(),
            sel_selfbalance: Vec::new(),
            sel_log0: Vec::new(),
            sel_log1: Vec::new(),
            sel_log2: Vec::new(),
            sel_log3: Vec::new(),
            sel_log4: Vec::new(),
            sel_address: Vec::new(),
            sel_caller: Vec::new(),
            sel_callvalue: Vec::new(),
            sel_origin: Vec::new(),
            sel_calldatasize: Vec::new(),
            sel_codesize: Vec::new(),
            sel_gasprice: Vec::new(),
            tx_origin: std::array::from_fn(|_| Vec::new()),
            tx_gas_price: Vec::new(),
            tx_calldata_size: Vec::new(),
            tx_code_size: Vec::new(),
            returndata_size: Vec::new(),
            sel_pc: Vec::new(),
            sel_gas: Vec::new(),
            sel_msize_op: Vec::new(),
            sel_jump: Vec::new(),
            sel_log: Vec::new(),
            sel_call: Vec::new(),
            next_pc: Vec::new(),
            sel_se: std::array::from_fn(|_| Vec::new()),
            se_byte: std::array::from_fn(|_| Vec::new()),
            se_low7: Vec::new(),
            se_sign_bit: Vec::new(),
            mulmod_q: std::array::from_fn(|_| Vec::new()),
            mulmod_p: std::array::from_fn(|_| Vec::new()),
            mulmod_pc: std::array::from_fn(|_| Vec::new()),
            mulmod_sc: std::array::from_fn(|_| Vec::new()),
            mulmod_slack: std::array::from_fn(|_| Vec::new()),
            mulmod_slack_borrow: std::array::from_fn(|_| Vec::new()),
            mulmod_n_is_zero: Vec::new(),
            addmod_q: std::array::from_fn(|_| Vec::new()),
            addmod_s_low: std::array::from_fn(|_| Vec::new()),
            addmod_s_high: Vec::new(),
            addmod_ab_carry: std::array::from_fn(|_| Vec::new()),
            addmod_qn_p: std::array::from_fn(|_| Vec::new()),
            addmod_qn_carry: std::array::from_fn(|_| Vec::new()),
            addmod_sum_carry: std::array::from_fn(|_| Vec::new()),
            addmod_slack: std::array::from_fn(|_| Vec::new()),
            addmod_slack_borrow: std::array::from_fn(|_| Vec::new()),
            addmod_n_is_zero: Vec::new(),
            frame_depth: Vec::new(),
            frame_caller: std::array::from_fn(|_| Vec::new()),
            frame_callee: std::array::from_fn(|_| Vec::new()),
            frame_value: std::array::from_fn(|_| Vec::new()),
            frame_gas: Vec::new(),
            frame_return_pc: Vec::new(),
            frame_return_offset: Vec::new(),
            frame_return_size: Vec::new(),
            sel_call_push_frame: Vec::new(),
            sel_call_return: Vec::new(),
            sel_create: Vec::new(),
            sel_callcode: Vec::new(),
            sel_delegatecall: Vec::new(),
            sel_create2: Vec::new(),
            sel_staticcall: Vec::new(),
            sel_revert: Vec::new(),
            sel_byte_op: Vec::new(),
            frame_static: Vec::new(),
            create_address_hint: std::array::from_fn(|_| Vec::new()),
            create_nonce_hint: Vec::new(),
            sel_stop_pop: Vec::new(),
            create2_salt_hint: std::array::from_fn(|_| Vec::new()),
            create2_initcode_hash_hint: std::array::from_fn(|_| Vec::new()),
        }
    }

    /// Extract rows `[start..end)` into a new trace with step renumbered from 0.
    pub fn slice_rows(&self, start: usize, end: usize) -> Self {
        let end = end.min(self.step.len());
        assert!(start <= end, "slice_rows: start {} > end {}", start, end);
        let len = end - start;

        macro_rules! slice_vec {
            ($v:expr) => { $v[start..end].to_vec() };
        }
        macro_rules! slice_arr4 {
            ($a:expr) => {
                [slice_vec!($a[0]), slice_vec!($a[1]), slice_vec!($a[2]), slice_vec!($a[3])]
            };
        }

        let mut result = EvmTraceColumns {
            step: (0..len as u64).collect(),
            pc: slice_vec!(self.pc),
            opcode: slice_vec!(self.opcode),
            gas_remaining: slice_vec!(self.gas_remaining),
            stack_depth: slice_vec!(self.stack_depth),
            input0: slice_arr4!(self.input0),
            input1: slice_arr4!(self.input1),
            output0: slice_arr4!(self.output0),
            mem_offset: slice_vec!(self.mem_offset),
            mem_value: slice_arr4!(self.mem_value),
            insn_type: slice_vec!(self.insn_type),
            funct: slice_vec!(self.funct),
            immediate: slice_arr4!(self.immediate),
            aux0: slice_arr4!(self.aux0),
            aux1: slice_arr4!(self.aux1),
            sel_stop: slice_vec!(self.sel_stop),
            sel_arith_add: slice_vec!(self.sel_arith_add),
            sel_arith_sub: slice_vec!(self.sel_arith_sub),
            sel_arith_mul: slice_vec!(self.sel_arith_mul),
            sel_arith_div: slice_vec!(self.sel_arith_div),
            sel_mod: slice_vec!(self.sel_mod),
            sel_sdiv: slice_vec!(self.sel_sdiv),
            sel_smod: slice_vec!(self.sel_smod),
            sel_addmod: slice_vec!(self.sel_addmod),
            sel_mulmod: slice_vec!(self.sel_mulmod),
            sel_exp: slice_vec!(self.sel_exp),
            sel_signextend: slice_vec!(self.sel_signextend),
            sel_lt: slice_vec!(self.sel_lt),
            sel_gt: slice_vec!(self.sel_gt),
            sel_eq: slice_vec!(self.sel_eq),
            sel_iszero: slice_vec!(self.sel_iszero),
            sel_compare_other: slice_vec!(self.sel_compare_other),
            sel_and: slice_vec!(self.sel_and),
            sel_or: slice_vec!(self.sel_or),
            sel_xor: slice_vec!(self.sel_xor),
            sel_bitwise_other: slice_vec!(self.sel_bitwise_other),
            sel_shl: slice_vec!(self.sel_shl),
            sel_shr: slice_vec!(self.sel_shr),
            sel_sar: slice_vec!(self.sel_sar),
            sel_keccak: slice_vec!(self.sel_keccak),
            sel_env: slice_vec!(self.sel_env),
            sel_block: slice_vec!(self.sel_block),
            sel_push: slice_vec!(self.sel_push),
            sel_dup: slice_vec!(self.sel_dup),
            sel_pop: slice_vec!(self.sel_pop),
            sel_swap: slice_vec!(self.sel_swap),
            sel_stack_other: slice_vec!(self.sel_stack_other),
            sel_mload: slice_vec!(self.sel_mload),
            sel_mstore: slice_vec!(self.sel_mstore),
            sel_mstore8: slice_vec!(self.sel_mstore8),
            sel_msize: slice_vec!(self.sel_msize),
            sel_memory_other: slice_vec!(self.sel_memory_other),
            sel_storage: slice_vec!(self.sel_storage),
            sel_sload: slice_vec!(self.sel_sload),
            sel_sstore: slice_vec!(self.sel_sstore),
            sel_timestamp: slice_vec!(self.sel_timestamp),
            sel_number: slice_vec!(self.sel_number),
            sel_gaslimit: slice_vec!(self.sel_gaslimit),
            sel_basefee: slice_vec!(self.sel_basefee),
            sel_coinbase: slice_vec!(self.sel_coinbase),
            sel_prevrandao: slice_vec!(self.sel_prevrandao),
            sel_chainid: slice_vec!(self.sel_chainid),
            sel_blockhash: slice_vec!(self.sel_blockhash),
            sel_selfbalance: slice_vec!(self.sel_selfbalance),
            sel_log0: slice_vec!(self.sel_log0),
            sel_log1: slice_vec!(self.sel_log1),
            sel_log2: slice_vec!(self.sel_log2),
            sel_log3: slice_vec!(self.sel_log3),
            sel_log4: slice_vec!(self.sel_log4),
            sel_address: slice_vec!(self.sel_address),
            sel_caller: slice_vec!(self.sel_caller),
            sel_callvalue: slice_vec!(self.sel_callvalue),
            sel_origin: slice_vec!(self.sel_origin),
            sel_calldatasize: slice_vec!(self.sel_calldatasize),
            sel_codesize: slice_vec!(self.sel_codesize),
            sel_gasprice: slice_vec!(self.sel_gasprice),
            tx_origin: std::array::from_fn(|i| slice_vec!(self.tx_origin[i])),
            tx_gas_price: slice_vec!(self.tx_gas_price),
            tx_calldata_size: slice_vec!(self.tx_calldata_size),
            tx_code_size: slice_vec!(self.tx_code_size),
            returndata_size: slice_vec!(self.returndata_size),
            sel_pc: slice_vec!(self.sel_pc),
            sel_gas: slice_vec!(self.sel_gas),
            sel_msize_op: slice_vec!(self.sel_msize_op),
            sel_jump: slice_vec!(self.sel_jump),
            sel_log: slice_vec!(self.sel_log),
            sel_call: slice_vec!(self.sel_call),
            next_pc: slice_vec!(self.next_pc),
            sel_se: std::array::from_fn(|i| slice_vec!(self.sel_se[i])),
            se_byte: std::array::from_fn(|i| slice_vec!(self.se_byte[i])),
            se_low7: slice_vec!(self.se_low7),
            se_sign_bit: slice_vec!(self.se_sign_bit),
            mulmod_q: std::array::from_fn(|i| slice_vec!(self.mulmod_q[i])),
            mulmod_p: std::array::from_fn(|i| slice_vec!(self.mulmod_p[i])),
            mulmod_pc: std::array::from_fn(|i| slice_vec!(self.mulmod_pc[i])),
            mulmod_sc: std::array::from_fn(|i| slice_vec!(self.mulmod_sc[i])),
            mulmod_slack: std::array::from_fn(|i| slice_vec!(self.mulmod_slack[i])),
            mulmod_slack_borrow: std::array::from_fn(|i| slice_vec!(self.mulmod_slack_borrow[i])),
            mulmod_n_is_zero: slice_vec!(self.mulmod_n_is_zero),
            addmod_q: std::array::from_fn(|i| slice_vec!(self.addmod_q[i])),
            addmod_s_low: std::array::from_fn(|i| slice_vec!(self.addmod_s_low[i])),
            addmod_s_high: slice_vec!(self.addmod_s_high),
            addmod_ab_carry: std::array::from_fn(|i| slice_vec!(self.addmod_ab_carry[i])),
            addmod_qn_p: std::array::from_fn(|i| slice_vec!(self.addmod_qn_p[i])),
            addmod_qn_carry: std::array::from_fn(|i| slice_vec!(self.addmod_qn_carry[i])),
            addmod_sum_carry: std::array::from_fn(|i| slice_vec!(self.addmod_sum_carry[i])),
            addmod_slack: std::array::from_fn(|i| slice_vec!(self.addmod_slack[i])),
            addmod_slack_borrow: std::array::from_fn(|i| slice_vec!(self.addmod_slack_borrow[i])),
            addmod_n_is_zero: slice_vec!(self.addmod_n_is_zero),
            frame_depth: slice_vec!(self.frame_depth),
            frame_caller: std::array::from_fn(|i| slice_vec!(self.frame_caller[i])),
            frame_callee: std::array::from_fn(|i| slice_vec!(self.frame_callee[i])),
            frame_value: std::array::from_fn(|i| slice_vec!(self.frame_value[i])),
            frame_gas: slice_vec!(self.frame_gas),
            frame_return_pc: slice_vec!(self.frame_return_pc),
            frame_return_offset: slice_vec!(self.frame_return_offset),
            frame_return_size: slice_vec!(self.frame_return_size),
            sel_call_push_frame: slice_vec!(self.sel_call_push_frame),
            sel_call_return: slice_vec!(self.sel_call_return),
            sel_create: slice_vec!(self.sel_create),
            sel_callcode: slice_vec!(self.sel_callcode),
            sel_delegatecall: slice_vec!(self.sel_delegatecall),
            sel_create2: slice_vec!(self.sel_create2),
            sel_staticcall: slice_vec!(self.sel_staticcall),
            sel_revert: slice_vec!(self.sel_revert),
            sel_byte_op: slice_vec!(self.sel_byte_op),
            frame_static: slice_vec!(self.frame_static),
            create_address_hint: std::array::from_fn(|i| slice_vec!(self.create_address_hint[i])),
            create_nonce_hint: slice_vec!(self.create_nonce_hint),
            sel_stop_pop: slice_vec!(self.sel_stop_pop),
            create2_salt_hint: std::array::from_fn(|i| slice_vec!(self.create2_salt_hint[i])),
            create2_initcode_hash_hint: std::array::from_fn(|i| slice_vec!(self.create2_initcode_hash_hint[i])),
        };
        let _ = &mut result; // suppress unused_mut
        result
    }

    pub fn push_row(&mut self, row: &EvmTraceRow) {
        self.step.push(row.step);
        self.pc.push(row.pc);
        self.opcode.push(row.opcode as u64);
        self.gas_remaining.push(row.gas_remaining);
        self.stack_depth.push(row.stack_depth);
        for j in 0..4 {
            self.input0[j].push(row.input0[j]);
            self.input1[j].push(row.input1[j]);
            self.output0[j].push(row.output0[j]);
            self.mem_value[j].push(row.mem_value[j]);
            self.immediate[j].push(row.immediate[j]);
            self.aux0[j].push(row.aux0[j]);
            self.aux1[j].push(row.aux1[j]);
        }
        self.mem_offset.push(row.mem_offset);
        self.insn_type.push(row.insn_type as u64);
        self.funct.push(row.funct as u64);

        // Set selector columns based on (insn_type, funct).
        // Slots 0..=40: 41 base selectors. Slots 41/42: split-call selectors
        // (sel_call_push_frame, sel_call_return). Slots 43..=48: per-opcode
        // call-family selectors (CREATE, CALLCODE, DELEGATECALL, CREATE2,
        // STATICCALL, REVERT).
        let mut sel = [0u64; 76];
        match row.insn_type {
            // Special-case opcode 0xFD (REVERT): currently classified as
            // (INSN_STOP, funct=2) but needs its own selector so the
            // depth-decrement frame transition can be enforced like RETURN.
            INSN_STOP if row.opcode == 0xFD && row.frame.depth >= 1 => sel[48] = 1,
            INSN_STOP => sel[0] = 1,
            INSN_ARITH => match row.funct {
                FUNCT_ADD => sel[1] = 1,
                FUNCT_SUB => sel[2] = 1,
                FUNCT_MUL => sel[3] = 1,
                FUNCT_DIV => sel[4] = 1,
                FUNCT_MOD => sel[5] = 1,
                FUNCT_SDIV => sel[6] = 1,
                FUNCT_SMOD => sel[7] = 1,
                FUNCT_ADDMOD => sel[8] = 1,
                FUNCT_MULMOD => sel[9] = 1,
                FUNCT_EXP => sel[10] = 1,
                FUNCT_SIGNEXTEND => sel[11] = 1,
                _ => sel[0] = 1,
            },
            INSN_COMPARE => match row.funct {
                FUNCT_LT => sel[12] = 1,
                FUNCT_GT => sel[13] = 1,
                FUNCT_EQ => sel[14] = 1,
                FUNCT_ISZERO => sel[15] = 1,
                _ => sel[16] = 1,
            },
            INSN_BITWISE => match row.funct {
                FUNCT_AND => sel[17] = 1,
                FUNCT_OR => sel[18] = 1,
                FUNCT_XOR => sel[19] = 1,
                FUNCT_NOT => sel[20] = 1,    // sel_bitwise_other (NOT only — see COL_SEL_BYTE_OP)
                FUNCT_BYTE => sel[49] = 1,   // sel_byte_op (split out from bitwise_other)
                FUNCT_SHL => sel[21] = 1,
                FUNCT_SHR => sel[22] = 1,
                FUNCT_SAR => sel[23] = 1,
                _ => sel[20] = 1,
            },
            INSN_KECCAK => sel[24] = 1,
            INSN_ENV => {
                sel[25] = 1;
                match row.funct {
                    0 => sel[66] = 1,  // ADDRESS
                    2 => sel[69] = 1,  // ORIGIN
                    3 => sel[67] = 1,  // CALLER
                    4 => sel[68] = 1,  // CALLVALUE
                    6 => sel[70] = 1,  // CALLDATASIZE
                    8 => sel[71] = 1,  // CODESIZE
                    10 => sel[72] = 1, // GASPRICE
                    _ => {}
                }
            }
            INSN_BLOCK => {
                sel[26] = 1; // Umbrella sel_block (one-hot list)
                // #53 step 2 / #54: per-opcode aux selectors used as
                // gates for the EVM ↔ block_header_air cross-AIR
                // LogUp linkages. Honest-by-inspector convention,
                // similar to sel_sload/sel_sstore.
                match row.funct {
                    FUNCT_TIMESTAMP => sel[52] = 1,
                    FUNCT_NUMBER => sel[53] = 1,
                    FUNCT_GASLIMIT => sel[54] = 1,
                    FUNCT_BASEFEE => sel[55] = 1,
                    FUNCT_COINBASE => sel[56] = 1,
                    FUNCT_PREVRANDAO => sel[57] = 1,
                    FUNCT_CHAINID => sel[58] = 1,
                    FUNCT_BLOCKHASH => sel[59] = 1,
                    FUNCT_SELFBALANCE => sel[60] = 1,
                    _ => {}
                }
            }
            INSN_STACK => match row.funct {
                FUNCT_PUSH => sel[27] = 1,
                FUNCT_DUP => sel[28] = 1,
                FUNCT_POP => sel[29] = 1,
                FUNCT_SWAP => sel[30] = 1,
                _ => sel[31] = 1,
            },
            INSN_MEMORY => {
                match row.funct {
                    FUNCT_MLOAD => sel[32] = 1,
                    FUNCT_MSTORE => sel[33] = 1,
                    FUNCT_MSTORE8 => sel[34] = 1,
                    FUNCT_MSIZE => { sel[35] = 1; sel[75] = 1; } // MSIZE also fires sel_msize_op
                    _ => sel[36] = 1,
                }
            },
            INSN_STORAGE => {
                sel[37] = 1; // Umbrella sel_storage (unchanged for one-hot list)
                // Auxiliary per-opcode selectors used as gates for the
                // storage_access_air cross-AIR LogUp linkage. See Phase
                // A2 step 1b — these don't participate in the one-hot
                // sum but are honest-by-inspector and cross-AIR-bound.
                match row.funct {
                    FUNCT_SLOAD => sel[50] = 1,
                    FUNCT_SSTORE => sel[51] = 1,
                    _ => {} // TLOAD/TSTORE: both aux selectors stay 0
                }
            }
            INSN_JUMP => {
                sel[38] = 1;
                match row.funct {
                    FUNCT_PC => sel[73] = 1,
                    4 => sel[74] = 1, // GAS
                    _ => {}
                }
            }
            INSN_LOG => {
                sel[39] = 1;
                match row.funct {
                    0 => sel[61] = 1, // LOG0
                    1 => sel[62] = 1, // LOG1
                    2 => sel[63] = 1, // LOG2
                    3 => sel[64] = 1, // LOG3
                    4 => sel[65] = 1, // LOG4
                    _ => {}
                }
            }
            // INSN_CALL covers 7 opcodes 0xF0..0xFA; each gets its own
            // selector so the cross-row frame transition can carry per-
            // opcode algebraic relations. Top-level RETURN (depth=0) ends
            // the transaction with no caller frame to pop into, so it
            // routes to the umbrella sel_call oracle (the depth-decrement
            // constraint can't satisfy a -1 transition).
            // Nested RETURN (depth ≥ 1) gets the algebraic sel_call_return.
            INSN_CALL => match row.opcode {
                0xF0 => sel[43] = 1, // sel_create
                0xF1 => sel[41] = 1, // sel_call_push_frame (CALL)
                0xF2 => sel[44] = 1, // sel_callcode
                0xF3 if row.frame.depth >= 1 => sel[42] = 1, // sel_call_return
                0xF4 => sel[45] = 1, // sel_delegatecall
                0xF5 => sel[46] = 1, // sel_create2
                0xFA => sel[47] = 1, // sel_staticcall
                _ => sel[40] = 1,    // sel_call (umbrella oracle, e.g. top-level RETURN)
            },
            _ => sel[0] = 1,
        }
        self.sel_stop.push(sel[0]);
        self.sel_arith_add.push(sel[1]);
        self.sel_arith_sub.push(sel[2]);
        self.sel_arith_mul.push(sel[3]);
        self.sel_arith_div.push(sel[4]);
        self.sel_mod.push(sel[5]);
        self.sel_sdiv.push(sel[6]);
        self.sel_smod.push(sel[7]);
        self.sel_addmod.push(sel[8]);
        self.sel_mulmod.push(sel[9]);
        self.sel_exp.push(sel[10]);
        self.sel_signextend.push(sel[11]);
        self.sel_lt.push(sel[12]);
        self.sel_gt.push(sel[13]);
        self.sel_eq.push(sel[14]);
        self.sel_iszero.push(sel[15]);
        self.sel_compare_other.push(sel[16]);
        self.sel_and.push(sel[17]);
        self.sel_or.push(sel[18]);
        self.sel_xor.push(sel[19]);
        self.sel_bitwise_other.push(sel[20]);
        self.sel_shl.push(sel[21]);
        self.sel_shr.push(sel[22]);
        self.sel_sar.push(sel[23]);
        self.sel_keccak.push(sel[24]);
        self.sel_env.push(sel[25]);
        self.sel_block.push(sel[26]);
        self.sel_push.push(sel[27]);
        self.sel_dup.push(sel[28]);
        self.sel_pop.push(sel[29]);
        self.sel_swap.push(sel[30]);
        self.sel_stack_other.push(sel[31]);
        self.sel_mload.push(sel[32]);
        self.sel_mstore.push(sel[33]);
        self.sel_mstore8.push(sel[34]);
        self.sel_msize.push(sel[35]);
        self.sel_memory_other.push(sel[36]);
        self.sel_storage.push(sel[37]);
        self.sel_sload.push(sel[50]);
        self.sel_sstore.push(sel[51]);
        self.sel_timestamp.push(sel[52]);
        self.sel_number.push(sel[53]);
        self.sel_gaslimit.push(sel[54]);
        self.sel_basefee.push(sel[55]);
        self.sel_coinbase.push(sel[56]);
        self.sel_prevrandao.push(sel[57]);
        self.sel_chainid.push(sel[58]);
        self.sel_blockhash.push(sel[59]);
        self.sel_selfbalance.push(sel[60]);
        self.sel_log0.push(sel[61]);
        self.sel_log1.push(sel[62]);
        self.sel_log2.push(sel[63]);
        self.sel_log3.push(sel[64]);
        self.sel_log4.push(sel[65]);
        self.sel_address.push(sel[66]);
        self.sel_caller.push(sel[67]);
        self.sel_callvalue.push(sel[68]);
        self.sel_origin.push(sel[69]);
        self.sel_calldatasize.push(sel[70]);
        self.sel_codesize.push(sel[71]);
        self.sel_gasprice.push(sel[72]);
        // tx context columns: populated by inspector via row.tx_context_hint, defaulting to 0
        for j in 0..4 { self.tx_origin[j].push(row.tx_origin[j]); }
        self.tx_gas_price.push(row.tx_gas_price);
        self.tx_calldata_size.push(row.tx_calldata_size);
        self.tx_code_size.push(row.tx_code_size);
        self.returndata_size.push(row.returndata_size);
        self.sel_pc.push(sel[73]);
        self.sel_gas.push(sel[74]);
        self.sel_msize_op.push(sel[75]);
        self.sel_jump.push(sel[38]);
        self.sel_log.push(sel[39]);
        self.sel_call.push(sel[40]);
        self.sel_call_push_frame.push(sel[41]);
        self.sel_call_return.push(sel[42]);
        self.sel_create.push(sel[43]);
        self.sel_callcode.push(sel[44]);
        self.sel_delegatecall.push(sel[45]);
        self.sel_create2.push(sel[46]);
        self.sel_staticcall.push(sel[47]);
        self.sel_revert.push(sel[48]);
        self.sel_byte_op.push(sel[49]);
        self.next_pc.push(row.next_pc);

        // Frame-stack columns.
        self.frame_depth.push(row.frame.depth);
        for j in 0..4 {
            self.frame_caller[j].push(row.frame.caller[j]);
            self.frame_callee[j].push(row.frame.callee[j]);
            self.frame_value[j].push(row.frame.value[j]);
        }
        self.frame_gas.push(row.frame.gas);
        self.frame_return_pc.push(row.frame.return_pc);
        self.frame_return_offset.push(row.frame.return_offset);
        self.frame_return_size.push(row.frame.return_size);
        self.frame_static.push(row.frame.is_static);
        for j in 0..4 {
            self.create_address_hint[j].push(row.create_address_hint[j]);
            self.create2_salt_hint[j].push(row.create2_salt_hint[j]);
            self.create2_initcode_hash_hint[j].push(row.create2_initcode_hash_hint[j]);
        }
        self.create_nonce_hint.push(row.create_nonce_hint);
        // Compute sel_stop_pop deterministically from opcode + depth so
        // the inspector and synthetic constructors don't both have to
        // remember to set it.
        let sel_stop_pop = if row.opcode == 0x00 && row.frame.depth >= 1 { 1 } else { 0 };
        self.sel_stop_pop.push(sel_stop_pop);

        // ─── SIGNEXTEND case selectors and witness columns ───
        //
        // sel_se[k] for k in 0..=30 is 1 iff sel_signextend=1 AND b equals k.
        // sel_se[31] is 1 iff sel_signextend=1 AND b >= 31 (passthrough).
        //
        // Also populate byte decomposition of the mixed limb (input1_l[b/8])
        // and the sign-bit extraction.
        let is_signextend = sel[11] == 1;
        let mut case_sel = [0u64; 32];
        let mut se_bytes = [0u64; 8];
        let mut se_low7 = 0u64;
        let mut se_sign_bit = 0u64;

        if is_signextend {
            // Determine the effective case index from input0 (b).
            // b is a U256; EVM only uses its low byte. If b >= 31 OR any upper
            // limb is nonzero, we take the passthrough case (31).
            let b_is_passthrough = row.input0[1] != 0
                || row.input0[2] != 0
                || row.input0[3] != 0
                || row.input0[0] >= 31;
            let case_idx: usize = if b_is_passthrough { 31 } else { row.input0[0] as usize };
            case_sel[case_idx] = 1;

            if case_idx < 31 {
                // Determine mixed-limb index and byte offset within it.
                let mixed_limb_idx = case_idx / 8;
                let byte_offset = case_idx % 8;

                // Byte-decompose input1[mixed_limb_idx] into 8 bytes
                // (little-endian: byte_0 = LSB).
                let limb_val = row.input1[mixed_limb_idx];
                for i in 0..8 {
                    se_bytes[i] = (limb_val >> (8 * i)) & 0xFF;
                }

                // Extract sign-byte info.
                let sign_byte = se_bytes[byte_offset];
                se_sign_bit = (sign_byte >> 7) & 1;
                se_low7 = sign_byte & 0x7F;
            }
            // For passthrough case (31), all witness columns remain 0.
        }

        for k in 0..32 {
            self.sel_se[k].push(case_sel[k]);
        }
        for i in 0..8 {
            self.se_byte[i].push(se_bytes[i]);
        }
        self.se_low7.push(se_low7);
        self.se_sign_bit.push(se_sign_bit);

        // ─── MULMOD witness columns ───
        //
        // Populated when sel_mulmod=1. Inputs: a = input0, b = input1, n =
        // immediate (captured by inspector from stack position 2), r = output0.
        let is_mulmod = sel[9] == 1;
        let mulmod_w = if is_mulmod {
            compute_mulmod_witness(row.input0, row.input1, row.immediate, row.output0)
        } else {
            MulmodWitness::default()
        };
        for i in 0..4 {
            self.mulmod_q[i].push(mulmod_w.q[i]);
            self.mulmod_slack[i].push(mulmod_w.slack[i]);
            self.mulmod_slack_borrow[i].push(mulmod_w.slack_borrow[i]);
        }
        for i in 0..8 {
            self.mulmod_p[i].push(mulmod_w.p[i]);
        }
        for i in 0..7 {
            self.mulmod_pc[i].push(mulmod_w.pc[i]);
            self.mulmod_sc[i].push(mulmod_w.sc[i]);
        }
        self.mulmod_n_is_zero.push(mulmod_w.n_is_zero);

        // ─── ADDMOD witness columns ───
        //
        // Populated when sel_addmod=1. Inputs: a = input0, b = input1, n =
        // immediate (captured by inspector from stack position 2), r = output0.
        let is_addmod = sel[8] == 1;
        let addmod_w = if is_addmod {
            compute_addmod_witness(row.input0, row.input1, row.immediate, row.output0)
        } else {
            AddmodWitness::default()
        };
        for i in 0..4 {
            self.addmod_q[i].push(addmod_w.q[i]);
            self.addmod_s_low[i].push(addmod_w.s_low[i]);
            self.addmod_ab_carry[i].push(addmod_w.ab_carry[i]);
            self.addmod_slack[i].push(addmod_w.slack[i]);
            self.addmod_slack_borrow[i].push(addmod_w.slack_borrow[i]);
        }
        self.addmod_s_high.push(addmod_w.s_high);
        for i in 0..8 {
            self.addmod_qn_p[i].push(addmod_w.qn_p[i]);
        }
        for i in 0..7 {
            self.addmod_qn_carry[i].push(addmod_w.qn_carry[i]);
        }
        for i in 0..9 {
            self.addmod_sum_carry[i].push(addmod_w.sum_carry[i]);
        }
        self.addmod_n_is_zero.push(addmod_w.n_is_zero);
    }
}

/// Compute all MULMOD auxiliary witnesses for a single row.
///
/// Given a, b, n, r: compute
///   - p = a*b (full 512-bit product, 8 limbs)
///   - pc = carries for the schoolbook a*b multiplication
///   - q such that p = q*n + r when n != 0, else q = 0
///   - sc = carries for the q*n + r = p schoolbook
///   - slack + slack_borrow: non-borrow chain for (n - r - 1) when n != 0
///   - n_is_zero = 1 iff n == 0
pub fn compute_mulmod_witness(
    a: [u64; 4],
    b: [u64; 4],
    n: [u64; 4],
    r: [u64; 4],
) -> MulmodWitness {
    let n_is_zero = if n == [0u64; 4] { 1 } else { 0 };

    // (1) Compute p = a*b as 8 limbs via 4x4 schoolbook with carry chain.
    // At each output position k in 0..8:
    //   sum_k = (Σ_{i+j=k, i<4, j<4} a_i * b_j) + pc[k-1]
    //   p_k   = sum_k mod 2^64
    //   pc[k] = sum_k div 2^64   (for k < 7; pc[7] is implicit and must be 0)
    //
    // We accumulate each partial-product sum as a u128 with a separate "upper"
    // tracker to ensure the running value can exceed 2^128 (four products can
    // sum to up to 2^130 + 1). We use a simple u256 accumulator via two u128s.
    let mut p = [0u64; 8];
    let mut pc = [0u64; 7];
    let mut prev_carry_lo: u128 = 0;
    let mut prev_carry_hi: u128 = 0;
    for k in 0..8usize {
        // Sum all partial products a_i * b_j with i+j == k (i,j < 4).
        let mut lo: u128 = prev_carry_lo;
        let mut hi: u128 = prev_carry_hi;
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                let prod = (a[i] as u128) * (b[j] as u128);
                let old_lo = lo;
                lo = lo.wrapping_add(prod);
                if lo < old_lo {
                    hi = hi.wrapping_add(1);
                }
            }
        }
        // Extract low 64 bits for p_k, shift remaining to form carry for next k.
        p[k] = lo as u64;
        // Combined (hi, lo) as a 256-bit value; shift right by 64.
        // lo after shift: (lo >> 64) | (hi << 64); hi after shift: hi >> 64.
        let new_lo = (lo >> 64) | (hi << 64);
        let new_hi = hi >> 64;
        if k < 7 {
            // The carry out of position k must fit in 64 bits (it's bounded by
            // the number of partial products: 4 * 2^128 / 2^64 + 2^64 < 2^66,
            // but in practice carries fit in u64 for correctly computed
            // products — see DIV/MUL constraints for similar claim).
            pc[k] = new_lo as u64;
            debug_assert!(new_hi == 0, "mulmod product carry overflow");
            prev_carry_lo = new_lo;
            prev_carry_hi = new_hi;
        }
    }

    // (2) Compute q from p and n: q = p / n (large-integer division).
    // r is already given (the opcode output); q must satisfy p = q*n + r.
    let (q, _r_actual) = if n == [0u64; 4] {
        ([0u64; 4], [0u64; 4])
    } else {
        u512_divmod_u256(&p, &n)
    };

    // (3) Compute sc = carries for q*n + r = p schoolbook (8 positions, 4x4).
    let mut sc = [0u64; 7];
    let mut prev_lo: u128 = 0;
    let mut prev_hi: u128 = 0;
    for k in 0..8usize {
        let mut lo: u128 = prev_lo;
        let mut hi: u128 = prev_hi;
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                let prod = (q[i] as u128) * (n[j] as u128);
                let old = lo;
                lo = lo.wrapping_add(prod);
                if lo < old { hi = hi.wrapping_add(1); }
            }
        }
        if k < 4 {
            let old = lo;
            lo = lo.wrapping_add(r[k] as u128);
            if lo < old { hi = hi.wrapping_add(1); }
        }
        // Subtract p[k] (expected to be in the low 64 bits).
        // But the identity is: lo's low 64 bits must equal p[k]; the upper
        // part is the new carry.
        let new_lo = (lo >> 64) | (hi << 64);
        let new_hi = hi >> 64;
        if k < 7 {
            sc[k] = new_lo as u64;
            debug_assert!(new_hi == 0, "mulmod sum carry overflow");
            prev_lo = new_lo;
            prev_hi = new_hi;
        } else {
            // At position 7, the carry out must be zero for a consistent
            // decomposition: q*n + r must equal p exactly as a 512-bit value.
            debug_assert!(new_lo == 0 && new_hi == 0, "mulmod sum final carry nonzero");
        }
    }

    // (4) Slack chain: slack = n - r - 1 with borrow chain (only meaningful
    // when n != 0; when n == 0, the chain is gated off algebraically).
    // Per-limb: slack[k] = n[k] - r[k] - slack_borrow[k-1] + slack_borrow[k]*2^64
    //   (slack_borrow[-1] = 1 because we subtract 1 from n - r)
    let mut slack = [0u64; 4];
    let mut slack_borrow = [0u64; 4];
    if n_is_zero == 0 {
        let mut borrow: u64 = 1; // start with -1 in the sub chain (r + 1)
        for k in 0..4usize {
            let (d1, b1) = n[k].overflowing_sub(r[k]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            slack[k] = d2;
            borrow = (b1 as u64) + (b2 as u64);
            slack_borrow[k] = borrow;
        }
        // Final borrow must be 0 if r < n (which holds when n != 0).
        // If r == n or r > n, borrow would be 1 at the end — that would be
        // an invalid witness (EVM ensures r < n).
    }

    MulmodWitness {
        q,
        p,
        pc,
        sc,
        slack,
        slack_borrow,
        n_is_zero,
    }
}

/// Compute all ADDMOD auxiliary witnesses for a single row.
///
/// Given a, b, n, r compute:
///   - s_low, s_high: low 256 bits and top carry of a+b
///   - ab_carry: per-limb carry chain for a+b
///   - q such that (a+b) = q*n + r when n != 0, else q = 0
///   - qn_p: 8-limb product q*n
///   - qn_carry: carry chain for the q*n schoolbook (positions 0..6)
///   - sum_carry: 9-limb carry chain for q*n + r = s_low + s_high*2^256
///   - slack + slack_borrow: non-borrow chain proving r < n when n != 0
///   - n_is_zero = 1 iff n == 0
pub fn compute_addmod_witness(
    a: [u64; 4],
    b: [u64; 4],
    n: [u64; 4],
    r: [u64; 4],
) -> AddmodWitness {
    let n_is_zero = if n == [0u64; 4] { 1 } else { 0 };

    // (1) Compute a + b as (s_low, s_high) with ab_carry chain.
    // Per-limb: a[k] + b[k] + carry_in = s_low[k] + carry_out * 2^64
    //   ab_carry[k] = carry_out at position k
    //   s_high = ab_carry[3] (final carry out of limb 3)
    let mut s_low = [0u64; 4];
    let mut ab_carry = [0u64; 4];
    let mut carry: u64 = 0;
    for k in 0..4 {
        let (s1, c1) = a[k].overflowing_add(b[k]);
        let (s2, c2) = s1.overflowing_add(carry);
        s_low[k] = s2;
        carry = (c1 as u64) + (c2 as u64);
        ab_carry[k] = carry;
    }
    let s_high = ab_carry[3];
    debug_assert!(s_high <= 1, "addmod s_high must fit in 1 bit");

    // (2) Compute q from s = s_low + s_high*2^256 and n (large integer division).
    // Represent s as 8 little-endian u64 limbs (s_low in limbs 0..3; s_high in limb 4).
    let s_as_512: [u64; 8] = [
        s_low[0], s_low[1], s_low[2], s_low[3],
        s_high,   0,        0,        0,
    ];
    let (q, _r_actual) = if n_is_zero == 1 {
        ([0u64; 4], [0u64; 4])
    } else {
        u512_divmod_u256(&s_as_512, &n)
    };

    // (3) Compute qn_p = q * n via 4x4 schoolbook, capturing limb and carry chain.
    // At output position k in 0..8:
    //   sum_k = Σ_{i+j=k, i<4, j<4} q[i]*n[j] + qn_carry[k-1]
    //   qn_p[k] = sum_k mod 2^64
    //   qn_carry[k] = sum_k div 2^64  (for k < 7; qn_carry[7] must be 0 implicitly)
    let mut qn_p = [0u64; 8];
    let mut qn_carry = [0u64; 7];
    let mut prev_lo: u128 = 0;
    let mut prev_hi: u128 = 0;
    for k in 0..8usize {
        let mut lo: u128 = prev_lo;
        let mut hi: u128 = prev_hi;
        for i in 0..4usize {
            if k >= i && k - i < 4 {
                let j = k - i;
                let prod = (q[i] as u128) * (n[j] as u128);
                let old_lo = lo;
                lo = lo.wrapping_add(prod);
                if lo < old_lo {
                    hi = hi.wrapping_add(1);
                }
            }
        }
        qn_p[k] = lo as u64;
        let new_lo = (lo >> 64) | (hi << 64);
        let new_hi = hi >> 64;
        if k < 7 {
            qn_carry[k] = new_lo as u64;
            debug_assert!(new_hi == 0, "addmod qn carry overflow");
            prev_lo = new_lo;
            prev_hi = new_hi;
        } else {
            // qn_p[7] is free; the outer sum chain handles limbs 4..7 vs s_high.
            debug_assert!(new_lo == 0 && new_hi == 0, "addmod qn final carry nonzero");
        }
    }

    // (4) sum_carry: carry chain enforcing qn_p + r = s_low + s_high*2^256.
    //
    // Per-limb identity (k in 0..9):
    //   qn_p_k + r_k + sum_carry[k-1] = rhs_k + sum_carry[k] * 2^64
    // where:
    //   qn_p_k = qn_p[k] for k<8, else 0
    //   r_k    = r[k] for k<4, else 0
    //   rhs_k  = s_low[k] for k<4, s_high for k==4, 0 for k>4
    //   sum_carry[-1] = 0
    //
    // This identity holds when n != 0 and r is the correct remainder. When
    // n == 0 the constraint is gated off, so we populate zeros. When r is
    // incorrect (invalid witness) the computation below may produce
    // inconsistent limbs — that's expected; the constraint body will reflect
    // the inconsistency.
    let mut sum_carry = [0u64; 9];
    if n_is_zero == 0 {
        let mut carry: i128 = 0;
        for k in 0..9usize {
            let qn_k: i128 = if k < 8 { qn_p[k] as i128 } else { 0 };
            let r_k: i128 = if k < 4 { r[k] as i128 } else { 0 };
            let rhs_k: i128 = if k < 4 {
                s_low[k] as i128
            } else if k == 4 {
                s_high as i128
            } else {
                0
            };
            let sum = qn_k + r_k + carry - rhs_k;
            // For a valid witness the low 64 bits of `sum` are 0 and carry is
            // nonneg. We extract the carry via arithmetic right shift; the
            // low-64 residue is discarded (it will be reflected in the
            // constraint body for an invalid witness, which is still a valid
            // witness from the trace's standpoint — just not a passing one).
            carry = sum >> 64;
            // Clamp to u64 range when storing (negative / out-of-range values
            // indicate an invalid witness; the stored value is just a
            // deterministic function so the trace is reproducible).
            sum_carry[k] = (carry as i64 as u64).wrapping_add(0);
        }
    }

    // (5) Slack chain: slack = n - r - 1 with borrow chain.
    let mut slack = [0u64; 4];
    let mut slack_borrow = [0u64; 4];
    if n_is_zero == 0 {
        let mut borrow: u64 = 1; // start with -1 in the sub chain (r + 1)
        for k in 0..4usize {
            let (d1, b1) = n[k].overflowing_sub(r[k]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            slack[k] = d2;
            borrow = (b1 as u64) + (b2 as u64);
            slack_borrow[k] = borrow;
        }
    }

    AddmodWitness {
        q,
        s_low,
        s_high,
        ab_carry,
        qn_p,
        qn_carry,
        sum_carry,
        slack,
        slack_borrow,
        n_is_zero,
    }
}

/// Compute (quotient, remainder) = dividend / divisor where dividend is 512-bit
/// (8 u64 limbs, little-endian) and divisor is 256-bit (4 u64 limbs).
/// Returns ([0;4], [0;4]) if divisor is zero.
fn u512_divmod_u256(dividend: &[u64; 8], divisor: &[u64; 4]) -> ([u64; 4], [u64; 4]) {
    if *divisor == [0u64; 4] {
        return ([0u64; 4], [0u64; 4]);
    }
    // Binary long division: iterate bits 511..0.
    let mut q = [0u64; 4];
    let mut rem = [0u64; 4]; // running remainder (256-bit)
    for bit in (0..512).rev() {
        // Left-shift rem by 1, bringing in the bit from dividend.
        let new_rem3 = (rem[3] << 1) | (rem[2] >> 63);
        let new_rem2 = (rem[2] << 1) | (rem[1] >> 63);
        let new_rem1 = (rem[1] << 1) | (rem[0] >> 63);
        let new_rem0 = rem[0] << 1;
        rem = [new_rem0, new_rem1, new_rem2, new_rem3];
        let limb_idx = bit / 64;
        let bit_idx = bit % 64;
        let bit_val = (dividend[limb_idx] >> bit_idx) & 1;
        rem[0] |= bit_val;
        // If rem >= divisor, subtract and set quotient bit.
        if !lt_u256_cmp(&rem, divisor) {
            rem = sub_u256_arr(&rem, divisor);
            if bit < 256 {
                let qlimb = bit / 64;
                let qbit = bit % 64;
                q[qlimb] |= 1u64 << qbit;
            }
            // If bit >= 256, it means dividend bit was too high to fit in a
            // 256-bit quotient. For a valid MULMOD witness this does not
            // happen (q < 2^256 follows from p < n * 2^256).
        }
    }
    ([q[0], q[1], q[2], q[3]], rem)
}

fn lt_u256_cmp(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

fn sub_u256_arr(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (s1, b1) = a[i].overflowing_sub(b[i]);
        let (s2, b2) = s1.overflowing_sub(borrow);
        result[i] = s2;
        borrow = (b1 as u64) + (b2 as u64);
    }
    result
}

impl VmTrace for EvmTraceColumns {
    fn num_steps(&self) -> usize {
        self.step.len()
    }

    fn num_columns(&self) -> usize {
        1 + NUM_EVM_COLUMNS
    }

    fn column(&self, index: usize) -> &[u64] {
        match index {
            0 => &self.step,
            1 => &self.pc,
            2 => &self.opcode,
            3 => &self.gas_remaining,
            4 => &self.stack_depth,
            5 => &self.input0[0],
            6 => &self.input0[1],
            7 => &self.input0[2],
            8 => &self.input0[3],
            9 => &self.input1[0],
            10 => &self.input1[1],
            11 => &self.input1[2],
            12 => &self.input1[3],
            13 => &self.output0[0],
            14 => &self.output0[1],
            15 => &self.output0[2],
            16 => &self.output0[3],
            17 => &self.mem_offset,
            18 => &self.mem_value[0],
            19 => &self.mem_value[1],
            20 => &self.mem_value[2],
            21 => &self.mem_value[3],
            22 => &self.insn_type,
            23 => &self.funct,
            24 => &self.immediate[0],
            25 => &self.immediate[1],
            26 => &self.immediate[2],
            27 => &self.immediate[3],
            28 => &self.aux0[0],
            29 => &self.aux0[1],
            30 => &self.aux0[2],
            31 => &self.aux0[3],
            32 => &self.aux1[0],
            33 => &self.aux1[1],
            34 => &self.aux1[2],
            35 => &self.aux1[3],
            36 => &self.sel_stop,
            37 => &self.sel_arith_add,
            38 => &self.sel_arith_sub,
            39 => &self.sel_arith_mul,
            40 => &self.sel_arith_div,
            41 => &self.sel_mod,
            42 => &self.sel_sdiv,
            43 => &self.sel_smod,
            44 => &self.sel_addmod,
            45 => &self.sel_mulmod,
            46 => &self.sel_exp,
            47 => &self.sel_signextend,
            48 => &self.sel_lt,
            49 => &self.sel_gt,
            50 => &self.sel_eq,
            51 => &self.sel_iszero,
            52 => &self.sel_compare_other,
            53 => &self.sel_and,
            54 => &self.sel_or,
            55 => &self.sel_xor,
            56 => &self.sel_bitwise_other,
            57 => &self.sel_shl,
            58 => &self.sel_shr,
            59 => &self.sel_sar,
            60 => &self.sel_keccak,
            61 => &self.sel_env,
            62 => &self.sel_block,
            63 => &self.sel_push,
            64 => &self.sel_dup,
            65 => &self.sel_pop,
            66 => &self.sel_swap,
            67 => &self.sel_stack_other,
            68 => &self.sel_mload,
            69 => &self.sel_mstore,
            70 => &self.sel_mstore8,
            71 => &self.sel_msize,
            72 => &self.sel_memory_other,
            73 => &self.sel_storage,
            74 => &self.sel_jump,
            75 => &self.sel_log,
            76 => &self.sel_call,
            77 => &self.next_pc,
            // SIGNEXTEND case selectors: 78..=109 (data indices 77..=108)
            78..=109 => &self.sel_se[index - 78],
            // SIGNEXTEND byte decomposition: 110..=117 (data indices 109..=116)
            110..=117 => &self.se_byte[index - 110],
            118 => &self.se_low7,
            119 => &self.se_sign_bit,
            // ─── MULMOD columns (data indices 119..=153) ───
            // VmTrace index = data index + 1.
            // q: 120..=123 (data 119..=122)
            120..=123 => &self.mulmod_q[index - 120],
            // p: 124..=131 (data 123..=130)
            124..=131 => &self.mulmod_p[index - 124],
            // pc: 132..=138 (data 131..=137)
            132..=138 => &self.mulmod_pc[index - 132],
            // sc: 139..=145 (data 138..=144)
            139..=145 => &self.mulmod_sc[index - 139],
            // slack: 146..=149 (data 145..=148)
            146..=149 => &self.mulmod_slack[index - 146],
            // slack_borrow: 150..=153 (data 149..=152)
            150..=153 => &self.mulmod_slack_borrow[index - 150],
            // n_is_zero: 154 (data 153)
            154 => &self.mulmod_n_is_zero,
            // ─── ADDMOD columns (data indices 154..=199) ───
            // VmTrace index = data index + 1.
            // q: 155..=158 (data 154..=157)
            155..=158 => &self.addmod_q[index - 155],
            // s_low: 159..=162 (data 158..=161)
            159..=162 => &self.addmod_s_low[index - 159],
            // s_high: 163 (data 162)
            163 => &self.addmod_s_high,
            // ab_carry: 164..=167 (data 163..=166)
            164..=167 => &self.addmod_ab_carry[index - 164],
            // qn_p: 168..=175 (data 167..=174)
            168..=175 => &self.addmod_qn_p[index - 168],
            // qn_carry: 176..=182 (data 175..=181)
            176..=182 => &self.addmod_qn_carry[index - 176],
            // sum_carry: 183..=191 (data 182..=190)
            183..=191 => &self.addmod_sum_carry[index - 183],
            // slack: 192..=195 (data 191..=194)
            192..=195 => &self.addmod_slack[index - 192],
            // slack_borrow: 196..=199 (data 195..=198)
            196..=199 => &self.addmod_slack_borrow[index - 196],
            // n_is_zero: 200 (data 199)
            200 => &self.addmod_n_is_zero,
            // ─── Frame-stack columns (data indices 200..=216) ───
            // VmTrace index = data index + 1.
            // depth: 201 (data 200)
            201 => &self.frame_depth,
            // caller: 202..=205 (data 201..=204)
            202..=205 => &self.frame_caller[index - 202],
            // callee: 206..=209 (data 205..=208)
            206..=209 => &self.frame_callee[index - 206],
            // value: 210..=213 (data 209..=212)
            210..=213 => &self.frame_value[index - 210],
            // gas: 214 (data 213)
            214 => &self.frame_gas,
            // return_pc: 215 (data 214)
            215 => &self.frame_return_pc,
            // return_offset: 216 (data 215)
            216 => &self.frame_return_offset,
            // return_size: 217 (data 216)
            217 => &self.frame_return_size,
            // sel_call_push_frame: 218 (data 217)
            218 => &self.sel_call_push_frame,
            // sel_call_return: 219 (data 218)
            219 => &self.sel_call_return,
            // ─── Per-opcode call-family selectors (data 219..=224) ───
            220 => &self.sel_create,        // data 219
            221 => &self.sel_callcode,      // data 220
            222 => &self.sel_delegatecall,  // data 221
            223 => &self.sel_create2,       // data 222
            224 => &self.sel_staticcall,    // data 223
            225 => &self.sel_revert,        // data 224
            // frame_static (data 225)
            226 => &self.frame_static,
            // create_address_hint limbs (data 226..=229)
            227..=230 => &self.create_address_hint[index - 227],
            // create_nonce_hint (data 230)
            231 => &self.create_nonce_hint,
            // sel_stop_pop (data 231)
            232 => &self.sel_stop_pop,
            // create2_salt_hint limbs (data 232..=235)
            233..=236 => &self.create2_salt_hint[index - 233],
            // create2_initcode_hash_hint limbs (data 236..=239)
            237..=240 => &self.create2_initcode_hash_hint[index - 237],
            // sel_byte_op (data 240)
            241 => &self.sel_byte_op,
            // Phase A2 step 1b: sel_sload (data 241), sel_sstore (data 242)
            242 => &self.sel_sload,
            243 => &self.sel_sstore,
            // #53 step 2 / #54: per-opcode BLOCK selectors (data 243..248)
            244 => &self.sel_timestamp,
            245 => &self.sel_number,
            246 => &self.sel_gaslimit,
            247 => &self.sel_basefee,
            248 => &self.sel_coinbase,
            249 => &self.sel_prevrandao,
            250 => &self.sel_chainid,
            251 => &self.sel_blockhash,
            252 => &self.sel_selfbalance,
            253 => &self.sel_log0,
            254 => &self.sel_log1,
            255 => &self.sel_log2,
            256 => &self.sel_log3,
            257 => &self.sel_log4,
            258 => &self.sel_address,
            259 => &self.sel_caller,
            260 => &self.sel_callvalue,
            261 => &self.sel_origin,
            262 => &self.sel_calldatasize,
            263 => &self.sel_codesize,
            264 => &self.sel_gasprice,
            265..=268 => &self.tx_origin[index - 265],
            269 => &self.tx_gas_price,
            270 => &self.tx_calldata_size,
            271 => &self.tx_code_size,
            272 => &self.returndata_size,
            273 => &self.sel_pc,
            274 => &self.sel_gas,
            275 => &self.sel_msize_op,
            _ => panic!("Column index {} out of range", index),
        }
    }

    fn columns(&self) -> Vec<&[u64]> {
        (0..self.num_columns()).map(|i| self.column(i)).collect()
    }

    fn column_names(&self) -> Vec<&'static str> {
        vec![
            "step", "pc", "opcode", "gas_remaining", "stack_depth",
            "input0_l0", "input0_l1", "input0_l2", "input0_l3",
            "input1_l0", "input1_l1", "input1_l2", "input1_l3",
            "output0_l0", "output0_l1", "output0_l2", "output0_l3",
            "mem_offset",
            "mem_value_l0", "mem_value_l1", "mem_value_l2", "mem_value_l3",
            "insn_type", "funct",
            "immediate_l0", "immediate_l1", "immediate_l2", "immediate_l3",
            "aux0_l0", "aux0_l1", "aux0_l2", "aux0_l3",
            "aux1_l0", "aux1_l1", "aux1_l2", "aux1_l3",
            "sel_stop", "sel_arith_add", "sel_arith_sub",
            "sel_arith_mul", "sel_arith_div",
            "sel_mod", "sel_sdiv", "sel_smod", "sel_addmod",
            "sel_mulmod", "sel_exp", "sel_signextend",
            "sel_lt", "sel_gt", "sel_eq", "sel_iszero", "sel_compare_other",
            "sel_and", "sel_or", "sel_xor", "sel_bitwise_other",
            "sel_shl", "sel_shr", "sel_sar",
            "sel_keccak", "sel_env", "sel_block",
            "sel_push", "sel_dup", "sel_pop", "sel_swap", "sel_stack_other",
            "sel_mload", "sel_mstore", "sel_mstore8", "sel_msize", "sel_memory_other",
            "sel_storage", "sel_jump", "sel_log", "sel_call",
            "next_pc",
            "sel_se_0", "sel_se_1", "sel_se_2", "sel_se_3",
            "sel_se_4", "sel_se_5", "sel_se_6", "sel_se_7",
            "sel_se_8", "sel_se_9", "sel_se_10", "sel_se_11",
            "sel_se_12", "sel_se_13", "sel_se_14", "sel_se_15",
            "sel_se_16", "sel_se_17", "sel_se_18", "sel_se_19",
            "sel_se_20", "sel_se_21", "sel_se_22", "sel_se_23",
            "sel_se_24", "sel_se_25", "sel_se_26", "sel_se_27",
            "sel_se_28", "sel_se_29", "sel_se_30", "sel_se_ge31",
            "se_byte_0", "se_byte_1", "se_byte_2", "se_byte_3",
            "se_byte_4", "se_byte_5", "se_byte_6", "se_byte_7",
            "se_low7", "se_sign_bit",
            "mulmod_q0", "mulmod_q1", "mulmod_q2", "mulmod_q3",
            "mulmod_p0", "mulmod_p1", "mulmod_p2", "mulmod_p3",
            "mulmod_p4", "mulmod_p5", "mulmod_p6", "mulmod_p7",
            "mulmod_pc0", "mulmod_pc1", "mulmod_pc2", "mulmod_pc3",
            "mulmod_pc4", "mulmod_pc5", "mulmod_pc6",
            "mulmod_sc0", "mulmod_sc1", "mulmod_sc2", "mulmod_sc3",
            "mulmod_sc4", "mulmod_sc5", "mulmod_sc6",
            "mulmod_slack0", "mulmod_slack1", "mulmod_slack2", "mulmod_slack3",
            "mulmod_slack_borrow0", "mulmod_slack_borrow1",
            "mulmod_slack_borrow2", "mulmod_slack_borrow3",
            "mulmod_n_is_zero",
            "addmod_q0", "addmod_q1", "addmod_q2", "addmod_q3",
            "addmod_s_low0", "addmod_s_low1", "addmod_s_low2", "addmod_s_low3",
            "addmod_s_high",
            "addmod_ab_carry0", "addmod_ab_carry1",
            "addmod_ab_carry2", "addmod_ab_carry3",
            "addmod_qn_p0", "addmod_qn_p1", "addmod_qn_p2", "addmod_qn_p3",
            "addmod_qn_p4", "addmod_qn_p5", "addmod_qn_p6", "addmod_qn_p7",
            "addmod_qn_carry0", "addmod_qn_carry1", "addmod_qn_carry2",
            "addmod_qn_carry3", "addmod_qn_carry4", "addmod_qn_carry5",
            "addmod_qn_carry6",
            "addmod_sum_carry0", "addmod_sum_carry1", "addmod_sum_carry2",
            "addmod_sum_carry3", "addmod_sum_carry4", "addmod_sum_carry5",
            "addmod_sum_carry6", "addmod_sum_carry7", "addmod_sum_carry8",
            "addmod_slack0", "addmod_slack1", "addmod_slack2", "addmod_slack3",
            "addmod_slack_borrow0", "addmod_slack_borrow1",
            "addmod_slack_borrow2", "addmod_slack_borrow3",
            "addmod_n_is_zero",
            "frame_depth",
            "frame_caller_l0", "frame_caller_l1", "frame_caller_l2", "frame_caller_l3",
            "frame_callee_l0", "frame_callee_l1", "frame_callee_l2", "frame_callee_l3",
            "frame_value_l0", "frame_value_l1", "frame_value_l2", "frame_value_l3",
            "frame_gas", "frame_return_pc",
            "frame_return_offset", "frame_return_size",
            "sel_call_push_frame", "sel_call_return",
            "sel_create", "sel_callcode", "sel_delegatecall",
            "sel_create2", "sel_staticcall", "sel_revert",
            "frame_static",
            "create_address_hint_l0", "create_address_hint_l1",
            "create_address_hint_l2", "create_address_hint_l3",
            "create_nonce_hint",
            "sel_stop_pop",
            "create2_salt_hint_l0", "create2_salt_hint_l1",
            "create2_salt_hint_l2", "create2_salt_hint_l3",
            "create2_initcode_hash_hint_l0", "create2_initcode_hash_hint_l1",
            "create2_initcode_hash_hint_l2", "create2_initcode_hash_hint_l3",
            "sel_byte_op",
            "sel_sload",
            "sel_sstore",
            "sel_timestamp",
            "sel_number",
            "sel_gaslimit",
            "sel_basefee",
            "sel_coinbase",
            "sel_prevrandao",
            "sel_chainid",
            "sel_blockhash",
            "sel_selfbalance",
            "sel_log0",
            "sel_log1",
            "sel_log2",
            "sel_log3",
            "sel_log4",
            "sel_address",
            "sel_caller",
            "sel_callvalue",
            "sel_origin",
            "sel_calldatasize",
            "sel_codesize",
            "sel_gasprice",
            "tx_origin_l0", "tx_origin_l1", "tx_origin_l2", "tx_origin_l3",
            "tx_gas_price", "tx_calldata_size", "tx_code_size", "returndata_size",
            "sel_pc", "sel_gas", "sel_msize_op",
        ]
    }
}

/// Extract the effective shift amount from a U256 input1 (clamped to 0..=256).
/// If input1 >= 256 (any upper limb nonzero or limb0 >= 256), returns 256.
fn shift_amount_u256(input1: [u64; 4]) -> u32 {
    if input1[1] != 0 || input1[2] != 0 || input1[3] != 0 || input1[0] >= 256 {
        256
    } else {
        input1[0] as u32
    }
}

/// Compute 2^k as 4 x u64 limbs. If k >= 256, returns [0,0,0,0].
fn power_of_two_limbs(k: u32) -> [u64; 4] {
    if k >= 256 {
        return [0, 0, 0, 0];
    }
    let limb_idx = (k / 64) as usize;
    let bit_idx = k % 64;
    let mut result = [0u64; 4];
    result[limb_idx] = 1u64 << bit_idx;
    result
}

/// Compute the 2^k limb representation for use in shift constraint immediate columns.
/// The shift amount is taken from input0 (top of stack in EVM shift operations).
pub fn shift_power_of_two(shift_amount: [u64; 4]) -> [u64; 4] {
    power_of_two_limbs(shift_amount_u256(shift_amount))
}

/// Compute auxiliary values for an EVM instruction.
///
/// For ADD: aux0 = per-limb carry chain [carry0..carry3], aux1 = 0
/// For SUB: aux0 = per-limb borrow chain [borrow0..borrow3], aux1 = 0
/// For MUL: aux0 = high 256 bits of 512-bit product, aux1[0] = high 64 bits of input0_l0*input1_l0
/// For DIV: aux0 = remainder, aux1[0] = carry from limb 0 of quotient*divisor+remainder
/// For LT (0x10): aux1[0] = lt result (1 if input0 < input1)
/// For GT (0x11): aux1[0] = gt result (1 if input0 > input1)
/// For ISZERO (0x15): aux1 = 0 (no extra witness needed)
/// For MSTORE8 (0x53): aux0[0] = input1_l0 / 256 (quotient for byte extraction)
pub fn compute_evm_aux(
    opcode: u8,
    input0: [u64; 4],
    input1: [u64; 4],
    output: [u64; 4],
) -> ([u64; 4], [u64; 4]) {
    let zero = [0u64; 4];
    match opcode {
        0x01 => {
            // ADD: per-limb carry chain
            let mut aux0 = [0u64; 4];
            let mut carry = 0u64;
            for i in 0..4 {
                let (s1, c1) = input0[i].overflowing_add(input1[i]);
                let (_, c2) = s1.overflowing_add(carry);
                carry = (c1 as u64) + (c2 as u64);
                aux0[i] = carry;
            }
            (aux0, zero)
        }
        0x03 => {
            // SUB: per-limb borrow chain
            let mut aux0 = [0u64; 4];
            let mut borrow = 0u64;
            for i in 0..4 {
                let (s1, b1) = output[i].overflowing_add(input1[i]);
                let (_, b2) = s1.overflowing_add(borrow);
                borrow = (b1 as u64) + (b2 as u64);
                aux0[i] = borrow;
            }
            (aux0, zero)
        }
        0x02 => {
            // MUL: compute high 256 bits of 512-bit product (stored in aux0)
            // and the full carry chain for schoolbook multiplication (stored in aux1).
            let mut full = [0u128; 8];
            for i in 0..4 {
                let mut carry = 0u128;
                for j in 0..4 {
                    full[i + j] += (input0[i] as u128) * (input1[j] as u128) + carry;
                    carry = full[i + j] >> 64;
                    full[i + j] &= 0xFFFF_FFFF_FFFF_FFFF;
                }
                if i + 4 < 8 {
                    full[i + 4] += carry;
                }
            }
            let mut aux0 = [0u64; 4];
            for i in 0..4 {
                aux0[i] = full[i + 4] as u64;
            }
            let mut aux1 = [0u64; 4];
            let mut acc = (input0[0] as u128) * (input1[0] as u128);
            aux1[0] = (acc >> 64) as u64;
            acc = (input0[0] as u128) * (input1[1] as u128)
                + (input0[1] as u128) * (input1[0] as u128)
                + (aux1[0] as u128);
            aux1[1] = (acc >> 64) as u64;
            acc = (input0[0] as u128) * (input1[2] as u128)
                + (input0[1] as u128) * (input1[1] as u128)
                + (input0[2] as u128) * (input1[0] as u128)
                + (aux1[1] as u128);
            aux1[2] = (acc >> 64) as u64;
            acc = (input0[0] as u128) * (input1[3] as u128)
                + (input0[1] as u128) * (input1[2] as u128)
                + (input0[2] as u128) * (input1[1] as u128)
                + (input0[3] as u128) * (input1[0] as u128)
                + (aux1[2] as u128);
            aux1[3] = (acc >> 64) as u64;
            (aux0, aux1)
        }
        0x04 | 0x05 => {
            // DIV/SDIV: aux0 = remainder = dividend - quotient * divisor (mod 2^256)
            // aux1 = full carry chain for (quotient * divisor + remainder) per limb
            // The algebraic identity quotient*divisor+remainder = dividend holds in two's complement.
            let remainder = compute_remainder(input0, input1, output);
            let mut aux1 = [0u64; 4];
            let mut acc = (output[0] as u128) * (input1[0] as u128) + (remainder[0] as u128);
            aux1[0] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (input1[1] as u128)
                + (output[1] as u128) * (input1[0] as u128)
                + (remainder[1] as u128) + (aux1[0] as u128);
            aux1[1] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (input1[2] as u128)
                + (output[1] as u128) * (input1[1] as u128)
                + (output[2] as u128) * (input1[0] as u128)
                + (remainder[2] as u128) + (aux1[1] as u128);
            aux1[2] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (input1[3] as u128)
                + (output[1] as u128) * (input1[2] as u128)
                + (output[2] as u128) * (input1[1] as u128)
                + (output[3] as u128) * (input1[0] as u128)
                + (remainder[3] as u128) + (aux1[2] as u128);
            aux1[3] = (acc >> 64) as u64;
            (remainder, aux1)
        }
        0x06 | 0x07 => {
            // MOD/SMOD: output = remainder, aux0 = quotient
            // For SMOD: quotient is the signed quotient (two's complement).
            // The algebraic identity quotient*divisor+remainder = dividend holds mod 2^256.
            if input1 == [0, 0, 0, 0] {
                (zero, zero)
            } else {
                // For MOD (0x06), quotient is unsigned. For SMOD (0x07), compute signed quotient.
                let quotient = if opcode == 0x07 {
                    sdiv_u256(input0, input1)
                } else {
                    div_u256(input0, input1)
                };
                let mut aux1 = [0u64; 4];
                let mut acc = (quotient[0] as u128) * (input1[0] as u128) + (output[0] as u128);
                aux1[0] = (acc >> 64) as u64;
                acc = (quotient[0] as u128) * (input1[1] as u128)
                    + (quotient[1] as u128) * (input1[0] as u128)
                    + (output[1] as u128) + (aux1[0] as u128);
                aux1[1] = (acc >> 64) as u64;
                acc = (quotient[0] as u128) * (input1[2] as u128)
                    + (quotient[1] as u128) * (input1[1] as u128)
                    + (quotient[2] as u128) * (input1[0] as u128)
                    + (output[2] as u128) + (aux1[1] as u128);
                aux1[2] = (acc >> 64) as u64;
                acc = (quotient[0] as u128) * (input1[3] as u128)
                    + (quotient[1] as u128) * (input1[2] as u128)
                    + (quotient[2] as u128) * (input1[1] as u128)
                    + (quotient[3] as u128) * (input1[0] as u128)
                    + (output[3] as u128) + (aux1[2] as u128);
                aux1[3] = (acc >> 64) as u64;
                (quotient, aux1)
            }
        }
        0x10 => {
            // LT: subtract input0 - input1 with borrow chain
            let mut aux0 = [0u64; 4];
            let mut aux1 = [0u64; 4];
            let mut borrow = 0u64;
            for i in 0..4 {
                let (d1, b1) = input0[i].overflowing_sub(input1[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                aux1[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
                aux0[i] = borrow;
            }
            (aux0, aux1)
        }
        0x11 => {
            // GT: subtract input1 - input0 with borrow chain
            let mut aux0 = [0u64; 4];
            let mut aux1 = [0u64; 4];
            let mut borrow = 0u64;
            for i in 0..4 {
                let (d1, b1) = input1[i].overflowing_sub(input0[i]);
                let (d2, b2) = d1.overflowing_sub(borrow);
                aux1[i] = d2;
                borrow = (b1 as u64) + (b2 as u64);
                aux0[i] = borrow;
            }
            (aux0, aux1)
        }
        0x14 => {
            // EQ: aux1[0..3] = diff limbs (input0 - input1), aux0 = 0
            let mut aux1 = [0u64; 4];
            for i in 0..4 {
                aux1[i] = input0[i].wrapping_sub(input1[i]);
            }
            (zero, aux1)
        }
        0x15 => {
            // ISZERO: no extra witness needed
            (zero, zero)
        }
        0x16 | 0x17 | 0x18 => {
            // AND/OR/XOR: aux0 = per-limb AND(input0, input1)
            let mut aux0 = [0u64; 4];
            for i in 0..4 {
                aux0[i] = input0[i] & input1[i];
            }
            (aux0, zero)
        }
        0x1B => {
            // SHL: output = input1 << input0 (mod 2^256)
            let k = shift_amount_u256(input0);
            let pow2 = power_of_two_limbs(k);
            let mut aux0 = [0u64; 4];
            let mut acc = (input1[0] as u128) * (pow2[0] as u128);
            aux0[0] = (acc >> 64) as u64;
            acc = (input1[0] as u128) * (pow2[1] as u128)
                + (input1[1] as u128) * (pow2[0] as u128)
                + (aux0[0] as u128);
            aux0[1] = (acc >> 64) as u64;
            acc = (input1[0] as u128) * (pow2[2] as u128)
                + (input1[1] as u128) * (pow2[1] as u128)
                + (input1[2] as u128) * (pow2[0] as u128)
                + (aux0[1] as u128);
            aux0[2] = (acc >> 64) as u64;
            acc = (input1[0] as u128) * (pow2[3] as u128)
                + (input1[1] as u128) * (pow2[2] as u128)
                + (input1[2] as u128) * (pow2[1] as u128)
                + (input1[3] as u128) * (pow2[0] as u128)
                + (aux0[2] as u128);
            aux0[3] = (acc >> 64) as u64;
            (aux0, zero)
        }
        0x1C | 0x1D => {
            // SHR/SAR: output = input1 >> input0 (logical or arithmetic)
            let k = shift_amount_u256(input0);
            let pow2 = power_of_two_limbs(k);
            let product = mul_u256_low(output, pow2);
            let remainder = sub_u256(input1, product);
            let mut aux1 = [0u64; 4];
            let mut acc = (output[0] as u128) * (pow2[0] as u128) + (remainder[0] as u128);
            aux1[0] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (pow2[1] as u128)
                + (output[1] as u128) * (pow2[0] as u128)
                + (remainder[1] as u128) + (aux1[0] as u128);
            aux1[1] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (pow2[2] as u128)
                + (output[1] as u128) * (pow2[1] as u128)
                + (output[2] as u128) * (pow2[0] as u128)
                + (remainder[2] as u128) + (aux1[1] as u128);
            aux1[2] = (acc >> 64) as u64;
            acc = (output[0] as u128) * (pow2[3] as u128)
                + (output[1] as u128) * (pow2[2] as u128)
                + (output[2] as u128) * (pow2[1] as u128)
                + (output[3] as u128) * (pow2[0] as u128)
                + (remainder[3] as u128) + (aux1[2] as u128);
            aux1[3] = (acc >> 64) as u64;
            (remainder, aux1)
        }
        0x53 => {
            // MSTORE8: aux0[0] = input1_l0 / 256 (quotient for byte extraction)
            // Constraint: aux0_l0 * 256 + mem_val_l0 - input1_l0 = 0
            let aux0 = [input1[0] / 256, 0, 0, 0];
            (aux0, zero)
        }
        _ => (zero, zero),
    }
}

/// Add two 256-bit numbers, returning (result, carry).
#[allow(dead_code)]
fn add_u256_with_carry(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], bool) {
    let mut result = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        result[i] = s2;
        carry = (c1 as u64) + (c2 as u64);
    }
    (result, carry > 0)
}

/// Compare two u256 values: a < b.
fn lt_u256(a: [u64; 4], b: [u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

/// Compute remainder = dividend - quotient * divisor (for DIV constraint).
fn compute_remainder(dividend: [u64; 4], divisor: [u64; 4], quotient: [u64; 4]) -> [u64; 4] {
    let product = mul_u256_low(quotient, divisor);
    sub_u256(dividend, product)
}

/// Multiply two u256 values, returning only the low 256 bits.
pub fn mul_u256_low(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            if i + j >= 4 { break; }
            let prod = (a[i] as u128) * (b[j] as u128) + (result[i + j] as u128) + carry;
            result[i + j] = prod as u64;
            carry = prod >> 64;
        }
    }
    result
}

/// Divide a by b (unsigned 256-bit), returning the quotient.
/// Returns [0,0,0,0] if b == 0.
fn div_u256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    if b == [0, 0, 0, 0] {
        return [0, 0, 0, 0];
    }
    let mut quotient = [0u64; 4];
    let mut remainder = [0u64; 4];
    for bit in (0..256).rev() {
        let carry3 = remainder[3] >> 63;
        remainder[3] = (remainder[3] << 1) | (remainder[2] >> 63);
        remainder[2] = (remainder[2] << 1) | (remainder[1] >> 63);
        remainder[1] = (remainder[1] << 1) | (remainder[0] >> 63);
        remainder[0] <<= 1;
        let _ = carry3;
        let limb_idx = bit / 64;
        let bit_idx = bit % 64;
        remainder[0] |= (a[limb_idx] >> bit_idx) & 1;
        if !lt_u256(remainder, b) {
            remainder = sub_u256(remainder, b);
            quotient[limb_idx] |= 1u64 << bit_idx;
        }
    }
    quotient
}

/// Signed division of two 256-bit two's complement values, returning the signed quotient
/// in two's complement representation (mod 2^256).
fn sdiv_u256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    if b == [0, 0, 0, 0] {
        return [0, 0, 0, 0];
    }
    let a_neg = (a[3] >> 63) != 0;
    let b_neg = (b[3] >> 63) != 0;
    let abs_a = if a_neg { negate_u256(a) } else { a };
    let abs_b = if b_neg { negate_u256(b) } else { b };
    let q = div_u256(abs_a, abs_b);
    if a_neg != b_neg { negate_u256(q) } else { q }
}

/// Negate a 256-bit two's complement value: result = (2^256 - val) mod 2^256 = !val + 1.
fn negate_u256(val: [u64; 4]) -> [u64; 4] {
    let mut result = [!val[0], !val[1], !val[2], !val[3]];
    let mut carry = 1u64;
    for i in 0..4 {
        let (s, c) = result[i].overflowing_add(carry);
        result[i] = s;
        carry = c as u64;
    }
    result
}

/// Subtract b from a (mod 2^256).
fn sub_u256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (s1, b1) = a[i].overflowing_sub(b[i]);
        let (s2, b2) = s1.overflowing_sub(borrow);
        result[i] = s2;
        borrow = (b1 as u64) + (b2 as u64);
    }
    result
}

/// Compute a SHA3-256 hash of the EVM execution state at a given row.
pub fn evm_state_hash(trace: &EvmTraceColumns, row: usize) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(trace.step[row].to_le_bytes());
    hasher.update(trace.pc[row].to_le_bytes());
    hasher.update(trace.opcode[row].to_le_bytes());
    hasher.update(trace.gas_remaining[row].to_le_bytes());
    hasher.update(trace.stack_depth[row].to_le_bytes());
    hasher.finalize().into()
}

/// Hash for the initial state (before any execution).
pub fn evm_initial_state_hash() -> [u8; 32] {
    let hasher = Sha3_256::new();
    hasher.finalize().into()
}

/// Hash for the final state after all execution.
pub fn evm_final_state_hash(trace: &EvmTraceColumns) -> [u8; 32] {
    if trace.step.is_empty() {
        return evm_initial_state_hash();
    }
    let last = trace.step.len() - 1;
    let mut hasher = Sha3_256::new();
    hasher.update(b"final");
    hasher.update(trace.step[last].to_le_bytes());
    hasher.update(trace.pc[last].to_le_bytes());
    hasher.update(trace.opcode[last].to_le_bytes());
    hasher.update(trace.gas_remaining[last].to_le_bytes());
    hasher.update(trace.stack_depth[last].to_le_bytes());
    hasher.finalize().into()
}

/// Convenience wrapper: build TracePolynomials from an EVM trace with the given curve.
pub fn evm_trace_polys_with_curve(trace: &EvmTraceColumns, curve: CurveType) -> TracePolynomials {
    TracePolynomials::from_vm_trace(trace, curve)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_opcode_add() {
        assert_eq!(classify_opcode(0x01), (INSN_ARITH, FUNCT_ADD));
    }

    #[test]
    fn test_classify_opcode_push1() {
        assert_eq!(classify_opcode(0x60), (INSN_STACK, FUNCT_PUSH));
    }

    #[test]
    fn test_classify_opcode_stop() {
        assert_eq!(classify_opcode(0x00), (INSN_STOP, 0));
    }

    #[test]
    fn test_classify_opcode_jump() {
        assert_eq!(classify_opcode(0x56), (INSN_JUMP, FUNCT_JUMP));
    }

    #[test]
    fn test_u256_add_no_carry() {
        let a = [10, 0, 0, 0];
        let b = [20, 0, 0, 0];
        let (result, carry) = add_u256_with_carry(a, b);
        assert_eq!(result, [30, 0, 0, 0]);
        assert!(!carry);
    }

    #[test]
    fn test_u256_add_with_carry() {
        let a = [u64::MAX, 0, 0, 0];
        let b = [1, 0, 0, 0];
        let (result, carry) = add_u256_with_carry(a, b);
        assert_eq!(result, [0, 1, 0, 0]);
        assert!(!carry);
    }

    #[test]
    fn test_u256_add_overflow() {
        let a = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let b = [1, 0, 0, 0];
        let (result, carry) = add_u256_with_carry(a, b);
        assert_eq!(result, [0, 0, 0, 0]);
        assert!(carry);
    }

    #[test]
    fn test_u256_sub() {
        let a = [30, 0, 0, 0];
        let b = [10, 0, 0, 0];
        let result = sub_u256(a, b);
        assert_eq!(result, [20, 0, 0, 0]);
    }

    #[test]
    fn test_u256_mul() {
        let a = [3, 0, 0, 0];
        let b = [7, 0, 0, 0];
        let result = mul_u256_low(a, b);
        assert_eq!(result, [21, 0, 0, 0]);
    }

    #[test]
    fn test_vm_trace_columns() {
        let mut trace = EvmTraceColumns::new();
        let row = EvmTraceRow {
            step: 0,
            pc: 0,
            opcode: 0x01, // ADD
            gas_remaining: 1000,
            stack_depth: 2,
            input0: [10, 0, 0, 0],
            input1: [20, 0, 0, 0],
            output0: [30, 0, 0, 0],
            mem_offset: 0,
            mem_value: [0; 4],
            insn_type: INSN_ARITH,
            funct: FUNCT_ADD,
            immediate: [0; 4],
            aux0: [0; 4],
            aux1: [0; 4],
            next_pc: 0,
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
        trace.push_row(&row);

        assert_eq!(trace.num_steps(), 1);
        assert_eq!(trace.num_columns(), 1 + NUM_EVM_COLUMNS);
        assert_eq!(trace.column(1)[0], 0);  // pc
        assert_eq!(trace.column(2)[0], 0x01);  // opcode
    }

    #[test]
    fn test_compute_aux_add() {
        let input0 = [10, 0, 0, 0];
        let input1 = [20, 0, 0, 0];
        let output = [30, 0, 0, 0];
        let (aux0, aux1) = compute_evm_aux(0x01, input0, input1, output);
        assert_eq!(aux0[0], 0); // no carry
        assert_eq!(aux1, [0; 4]);
    }

    #[test]
    fn test_compute_aux_add_overflow() {
        let input0 = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let input1 = [1, 0, 0, 0];
        let output = [0, 0, 0, 0];
        let (aux0, _) = compute_evm_aux(0x01, input0, input1, output);
        assert_eq!(aux0[0], 1); // carry
    }

    #[test]
    fn test_compute_aux_mstore8() {
        // MSTORE8: stores byte at input1_l0 & 0xFF
        let input0 = [0x100, 0, 0, 0]; // offset
        let input1 = [0x1234, 0, 0, 0]; // value (low byte = 0x34)
        let output = [0; 4];
        let (aux0, _) = compute_evm_aux(0x53, input0, input1, output);
        // aux0[0] = input1_l0 / 256 = 0x1234 / 256 = 0x12
        assert_eq!(aux0[0], 0x12);
    }
}
