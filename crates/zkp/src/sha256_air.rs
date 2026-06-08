//! Bit-level AIR for the SHA-256 compression function.
//!
//! This module implements a sound constraint system for the 64-round
//! SHA-256 compression function by representing every 32-bit word as 32
//! individual 0/1 bit columns. Unlike the Keccak AIR (which is fully
//! bit-level over XOR/AND/rotate), SHA-256 also has modular 32-bit
//! additions for T₁, T₂, new-a and new-e. Those adds are witnessed with
//! explicit small carry columns so that every constraint stays
//! algebraically expressible as a degree-≤2 polynomial:
//!
//!   XOR(a, b) = a + b − 2·a·b
//!   NOT(a)    = 1 − a
//!   AND(a, b) = a · b
//!   rotate-by-k = permute bit positions (structural — no arithmetic)
//!   32-bit add = sum_bits + carry·2^32 = Σ inputs  (range-checked via
//!     binary carry columns)
//!
//! Trade-off: ~1,000 bit columns per row. Much smaller than the Keccak
//! AIR (~13k bits) because SHA-256's state is only 256 working bits plus
//! change, and the 64 rounds share a single row-local constraint template.
//!
//! # Row layout
//!
//! One row per SHA-256 round. 64 rounds per block; a multi-block hash
//! produces `blocks.len() * 64` rows. Cross-row constraints bind the
//! post-round working variables of row `r` to the pre-round working
//! variables of row `r+1`, *except* at block boundaries (`r mod 64 == 63`)
//! where the next block starts from its own `state_in`.
//!
//! # Column layout
//!
//! All columns hold 0/1 scalar values. The layout is (all bit widths are
//! 32 unless noted):
//!
//! ```text
//! index range                   size  meaning
//! 0..256      BEFORE (a..h)      256   pre-round working variables
//! 256..512    AFTER  (a'..h')    256   post-round working variables
//! 512..544    W                   32   message-schedule word W[t]
//! 544..576    K                   32   round constant K[t]
//! 576..608    BIG_SIGMA0_A        32   Σ₀(a)
//! 608..640    XOR01_S0            32   aux: rotr(a,2) XOR rotr(a,13)
//! 640..672    BIG_SIGMA1_E        32   Σ₁(e)
//! 672..704    XOR01_S1            32   aux: rotr(e,6) XOR rotr(e,11)
//! 704..736    EF                  32   aux: e AND f
//! 736..768    NOT_E_G             32   aux: (NOT e) AND g
//! 768..800    CH_EFG              32   Ch(e,f,g)
//! 800..832    AB                  32   aux: a AND b
//! 832..864    AC                  32   aux: a AND c
//! 864..896    BC                  32   aux: b AND c
//! 896..928    XOR_AB_AC           32   aux: AB XOR AC
//! 928..960    MAJ_ABC             32   Maj(a,b,c)
//! 960..963    T1_CARRY             3   T₁ carry bits (sum can be up to
//!                                      5·(2^32−1) < 2^35 → 3-bit carry)
//! 963..964    T2_CARRY             1   T₂ carry bit
//! 964..965    A_NEW_CARRY          1   new-a carry bit
//! 965..966    E_NEW_CARRY          1   new-e carry bit
//! 966..1030   SEL_ROUND           64   one-hot round selector
//! 1030..1094  INV_INPUT_BYTE      64   per-invocation input bytes
//! 1094..1126  INV_OUTPUT_BYTE     32   per-invocation output bytes (digest)
//! 1126..1127  IS_FIRST_INV_ROW     1   anchor: 1 only on row 0 of invocation
//! 1127..1128  IS_FIRST_BLOCK       1   1 on every round of block 0, 0 elsewhere
//! 1128..1136  STATE_IN_WORD        8   chaining state_in[v] for the current block
//! 1136..1144  CHAIN_CARRY_BIT      8   1-bit carry for cross-block state chain
//! 1144..1152  BINDING_CARRY_BIT    8   1-bit carry for final digest reconstruction
//! 1152..1153  AGGREGATOR_ACTIVE    1   1 on aggregator-populated traces, 0 on legacy
//! 1153..1154  IS_LAST_BLOCK        1   1 on the LAST block's rows, 0 elsewhere
//! ```
//!
//! Total: 1,096 data (incl. 128-col message-schedule σ helpers + 2-bit
//! W[16] recurrence carry) + 64 selectors + 97 invocation aggregator + 1
//! IS_FIRST_BLOCK + 24 multi-block output-binding aggregator + 1
//! AGGREGATOR_ACTIVE + 1 IS_LAST_BLOCK = **1,284 columns**.
//!
//! # Constraints (all bit-level where not noted)
//!
//!  1. Bit validity: every data column satisfies `b·(b−1) = 0`.
//!  2. Σ₀(a) definition via two-stage XOR:
//!     `xor_01_s0[i] = rot(a,2)[i] XOR rot(a,13)[i]`
//!     `big_sigma0_a[i] = xor_01_s0[i] XOR rot(a,22)[i]`
//!  3. Σ₁(e) definition via two-stage XOR with rotations 6, 11, 25.
//!  4. Ch definition:
//!     `ef[i] = e[i]·f[i]`
//!     `not_e_g[i] = (1 − e[i])·g[i]`
//!     `ch_efg[i] = ef[i] XOR not_e_g[i]`  (degree-2 XOR)
//!  5. Maj definition:
//!     `ab[i] = a[i]·b[i]`, `ac[i] = a[i]·c[i]`, `bc[i] = b[i]·c[i]`
//!     `xor_ab_ac[i] = ab[i] XOR ac[i]`
//!     `maj_abc[i] = xor_ab_ac[i] XOR bc[i]`
//!  6. T₁ summation: `t1_value = Σ a'[i]·2^i` where `a' = new-a`. Actually
//!     the T₁ 32-bit word isn't a column — instead, we recover it via
//!     `new_a - t2 mod 2^32` implicitly, and we fold the T₁ add and the
//!     new-a add into a single relation. Equivalently, we impose:
//!
//!     `a_new_value + t1_carry·2^32 + t2_value + a_new_carry·2^32
//!        = h + Σ₁(e) + Ch + K + W + t2_value + a_new_carry·2^32`
//!
//!     Which simplifies to: on each row we introduce a 32-bit `T1_value`
//!     virtually equal to the sum-minus-carries. To keep the algebra
//!     simple we instead materialize T₁ and T₂ as separate witness
//!     values reconstructed from the post-round e and a:
//!
//!         T₁ = new_e − d  (mod 2^32)   (= e_new_carry·2^32 + new_e − d)
//!         T₂ = new_a − T₁ (mod 2^32)   (= a_new_carry·2^32 + new_a − T₁)
//!
//!     Concretely, the two summation constraints we impose are:
//!         (S1) Σ₁(e) + Ch + h + K + W
//!              = (new_e − d) + e_new_carry·2^32 + t1_carry·2^32
//!         (S2) Σ₀(a) + Maj = T₂ + t2_carry·2^32
//!              where T₂ is taken to be (new_a − T₁) reconstructed as
//!              (new_a − new_e + d + e_new_carry·2^32) mod 2^32,
//!              plus a_new_carry·2^32.
//!
//!     The practical circuit therefore expresses the SHA-256 round as
//!     one big algebraic relation per summation, reading the values
//!     directly off the bit columns of new_a, new_e, d, etc. Each bit
//!     column is a 0/1 value and the "value" is Σ bit[i]·2^i evaluated
//!     on the fly.
//!
//!  7. Passthrough:
//!     `b' = a, c' = b, d' = c, f' = e, g' = f, h' = g`
//!     enforced bit-by-bit.
//!  8. K binding: `Σ_k sel_round_k · (K_bit[i] − RC_k_bit[i]) = 0`
//!     for every i.
//!  9. Selector validity: `sel_round_k · (sel_round_k − 1) = 0`, and
//!     `(Σ sel_round_k) · (Σ sel_round_k − 1) = 0` on every row.
//!
//! Cross-row (transition) constraints bind `after(row r)[i] ==
//! before(row r+1)[i]` bit-by-bit, except at block boundaries (row
//! indices r where (r+1) mod 64 == 0).
//!
//! # Scope cuts
//!
//! - Message schedule (the W[t] recurrence for t ≥ 16) is **not**
//!   constrained here. W is treated as an assumed-correct column. A
//!   separate AIR will constrain its derivation.
//! - Cross-block chaining (`state_in[b+1] = state_in[b] + last-round's
//!   working vars of block b`) is likewise not wired in this row-local
//!   AIR. It is enforced externally by binding the first row of each
//!   block to its public `state_in`.

use crate::field::{CurveType, Scalar};
use crate::sha256::{HashTrace, RoundTrace, NUM_ROUNDS, ROUND_CONSTANTS};

// ──── Column offsets ──────────────────────────────────────────────────

pub const BITS_PER_WORD: usize = 32;
pub const NUM_WORKING_VARS: usize = 8;         // a..h
pub const BITS_PER_STATE: usize = 256;         // 8 × 32

pub const COL_BEFORE_OFFSET:        usize = 0;
pub const COL_AFTER_OFFSET:         usize = COL_BEFORE_OFFSET + BITS_PER_STATE;
pub const COL_W_OFFSET:             usize = COL_AFTER_OFFSET + BITS_PER_STATE;
pub const COL_K_OFFSET:             usize = COL_W_OFFSET + BITS_PER_WORD;
pub const COL_BIG_SIGMA0_A_OFFSET:  usize = COL_K_OFFSET + BITS_PER_WORD;
pub const COL_XOR01_S0_OFFSET:      usize = COL_BIG_SIGMA0_A_OFFSET + BITS_PER_WORD;
pub const COL_BIG_SIGMA1_E_OFFSET:  usize = COL_XOR01_S0_OFFSET + BITS_PER_WORD;
pub const COL_XOR01_S1_OFFSET:      usize = COL_BIG_SIGMA1_E_OFFSET + BITS_PER_WORD;
pub const COL_EF_OFFSET:            usize = COL_XOR01_S1_OFFSET + BITS_PER_WORD;
pub const COL_NOT_E_G_OFFSET:       usize = COL_EF_OFFSET + BITS_PER_WORD;
pub const COL_CH_EFG_OFFSET:        usize = COL_NOT_E_G_OFFSET + BITS_PER_WORD;
pub const COL_AB_OFFSET:            usize = COL_CH_EFG_OFFSET + BITS_PER_WORD;
pub const COL_AC_OFFSET:            usize = COL_AB_OFFSET + BITS_PER_WORD;
pub const COL_BC_OFFSET:            usize = COL_AC_OFFSET + BITS_PER_WORD;
pub const COL_XOR_AB_AC_OFFSET:     usize = COL_BC_OFFSET + BITS_PER_WORD;
pub const COL_MAJ_ABC_OFFSET:       usize = COL_XOR_AB_AC_OFFSET + BITS_PER_WORD;

pub const T1_CARRY_BITS: usize = 3;
pub const COL_T1_CARRY_OFFSET:      usize = COL_MAJ_ABC_OFFSET + BITS_PER_WORD;
pub const COL_T2_CARRY_OFFSET:      usize = COL_T1_CARRY_OFFSET + T1_CARRY_BITS;
pub const COL_A_NEW_CARRY_OFFSET:   usize = COL_T2_CARRY_OFFSET + 1;
pub const COL_E_NEW_CARRY_OFFSET:   usize = COL_A_NEW_CARRY_OFFSET + 1;

// ── Message-schedule σ helper columns (per-row witness for σ0(W), σ1(W)) ──
//
// SHA-256 message schedule uses two helper functions on 32-bit words:
//   σ0(x) = ROTR_7(x)  XOR ROTR_18(x) XOR SHR_3(x)
//   σ1(x) = ROTR_17(x) XOR ROTR_19(x) XOR SHR_10(x)
//
// Each row commits σ0(W) and σ1(W) as 32-bit bit-decomposed words, with
// a 32-bit XOR-stage intermediate. ROTR is a permutation (no separate
// witness column needed — we reference rotated W bits directly). SHR_k
// is "permutation with high-k bits forced to zero" (also no separate
// witness column — bits i ≥ 32-k contribute the literal 0).
//
// Constraint (per bit i, two-stage XOR — matches Σ₀/Σ₁ pattern):
//   xor01_ss0[i] = W[(i+7) mod 32] XOR W[(i+18) mod 32]
//   small_sigma0_w[i] = xor01_ss0[i] XOR (W[i+3] if i<29 else 0)
// and analogously for σ1 with rotations 17, 19 and shift 10.
//
// These per-row helpers algebraically pin σ0(W) and σ1(W) as functions of
// W on every row. The FULL message-schedule recurrence W[t] = σ1(W[t-2])
// + W[t-7] + σ0(W[t-15]) + W[t-16] mod 2^32 for t ≥ 16 requires cross-row
// access to W[t-k] for k ∈ {2, 7, 15, 16}, which is implemented in the
// shifted constraint (`w_recurrence_at_t16`) for the **first** anchor row
// only (t = 16, SEL_ROUND[16] = 1). Generalizing to all t ∈ {16..64}
// follows the same pattern with one carry column + one shifted body per
// recurrence offset; see the module-level doc for the deferred pattern.
pub const COL_XOR01_SS0_OFFSET:        usize = COL_E_NEW_CARRY_OFFSET + 1;
pub const COL_SMALL_SIGMA0_W_OFFSET:   usize = COL_XOR01_SS0_OFFSET + BITS_PER_WORD;
pub const COL_XOR01_SS1_OFFSET:        usize = COL_SMALL_SIGMA0_W_OFFSET + BITS_PER_WORD;
pub const COL_SMALL_SIGMA1_W_OFFSET:   usize = COL_XOR01_SS1_OFFSET + BITS_PER_WORD;

/// 2-bit carry column for the W[16] recurrence: 4 32-bit values sum to at
/// most ~4·2^32 < 2^34, so a 2-bit carry suffices. Witness-populated only
/// on the anchor row (SEL_ROUND[16] = 1) and zero elsewhere.
pub const W_RECURRENCE_CARRY_BITS: usize = 2;
pub const COL_W_RECURRENCE_CARRY_OFFSET: usize =
    COL_SMALL_SIGMA1_W_OFFSET + BITS_PER_WORD;

pub const NUM_DATA_COLUMNS: usize = COL_W_RECURRENCE_CARRY_OFFSET + W_RECURRENCE_CARRY_BITS;

pub const COL_SEL_ROUND_OFFSET:     usize = NUM_DATA_COLUMNS;
pub const NUM_SEL_ROUND:            usize = NUM_ROUNDS; // 64

// ── Per-invocation byte-aggregation columns ───────────────────────────
//
// A SHA-256 AIR trace covers ONE invocation (one call to `sha256(input)` /
// `sha256_pair(...)`). The bytes of the input and the 32-byte output
// digest are committed as per-invocation aggregator columns: they hold
// the SAME value on every row of the trace (invariance constraint). A
// cross-AIR LogUp linkage from [`crate::sha256_extract`] picks the
// anchor row (the one where `IS_FIRST_INV_ROW = 1`) and matches its
// `(INPUT_BYTE[..], OUTPUT_BYTE[..])` tuple against the extract AIR's
// per-row tuple.
//
// Soundness scope (current): the bytes are witness-populated and
// invariance-enforced. Algebraic binding to the bit-level trace
// (matching `INV_INPUT_BYTE` to W bits at rounds 0..15 and
// `INV_OUTPUT_BYTE` to the post-compression state) is a documented
// follow-up — see `cross_air_logup_dependent_tasks.md`. Today the
// linkage proves "the SHA-256 AIR committed *some* (input, output)
// tuple that matches the extract AIR's claim", which is the exact
// guarantee the extract AIR already has internally — so the binding
// adds no algebraic strength yet, only the structural plumbing.
pub const INV_INPUT_LEN: usize = 64;
pub const INV_OUTPUT_LEN: usize = 32;
pub const COL_INV_INPUT_BYTE_OFFSET:  usize = COL_SEL_ROUND_OFFSET + NUM_SEL_ROUND;
pub const COL_INV_OUTPUT_BYTE_OFFSET: usize = COL_INV_INPUT_BYTE_OFFSET + INV_INPUT_LEN;
pub const COL_IS_FIRST_INV_ROW:       usize = COL_INV_OUTPUT_BYTE_OFFSET + INV_OUTPUT_LEN;

/// Indicator: `1` on every round of block 0 (rows 0..NUM_ROUNDS-1),
/// `0` on every round of subsequent blocks. Witness-populated;
/// algebraically constrained by:
///   - Binarity (row-local)
///   - Pinned to 1 at row 0 via `IS_FIRST_INV_ROW · (1 − IS_FIRST_BLOCK)
///     = 0` (row-local)
///   - "Invariant within a block, free at block boundary" via
///     `(1 − SEL_ROUND[NUM_ROUNDS-1]) · (IS_FIRST_BLOCK(ω·X) −
///     IS_FIRST_BLOCK(X)) = 0` (shifted; excluded at domain wrap)
///
/// The invariance constraint forces IS_FIRST_BLOCK to remain
/// constant across the 63 internal-block transitions (rows where
/// `SEL_ROUND[NUM_ROUNDS-1] = 0`) and only allows it to change at the
/// block-end transition (`SEL_ROUND[NUM_ROUNDS-1] = 1`). Combined with
/// the `IS_FIRST_INV_ROW`-pinning at row 0, this guarantees
/// IS_FIRST_BLOCK = 1 on all 64 rows of block 0 in any valid trace.
pub const COL_IS_FIRST_BLOCK:         usize = COL_IS_FIRST_INV_ROW + 1;

/// Per-working-variable 32-bit word value of the SHA-256 chaining state
/// (`state_in`) at the START of the block this row belongs to.
///
///   - At row 0 (the start of block 0): `STATE_IN_WORD[v] = INITIAL_HASH[v]`
///     (algebraically pinned via `IS_FIRST_INV_ROW · (STATE_IN_WORD[v] −
///     INITIAL_HASH[v]) = 0`).
///   - Within a block (rows where `SEL_ROUND[NUM_ROUNDS-1] = 0`):
///     invariant (cross-row shifted constraint).
///   - At a block boundary (rows where `SEL_ROUND[NUM_ROUNDS-1] = 1`):
///     `STATE_IN_WORD[v](ω·X) = STATE_IN_WORD[v](X) + AFTER_word[v](X)
///     mod 2^32` (cross-row chaining; gated by `AGGREGATOR_ACTIVE` to
///     avoid firing on legacy traces).
///
/// This generalizes the previous sha256_pair-only `BLOCK_0_AFTER_WORD`
/// to arbitrary multi-block sha256 traces. For 1-block traces
/// (input ≤ 55 bytes), `STATE_IN_WORD = INITIAL_HASH` throughout, and
/// the digest reconstruction at row 63 gives
/// `INITIAL_HASH + AFTER_word[63] mod 2^32`. For 2-block sha256_pair,
/// `STATE_IN_WORD = INITIAL_HASH` on rows 0..63 and
/// `INITIAL_HASH + AFTER_word[63]` on rows 64..127, so the binding at
/// row 127 gives `(INITIAL_HASH + AFTER[63]) + AFTER[127] mod 2^32`,
/// matching the previous formula.
pub const NUM_OUTPUT_WORDS: usize = NUM_WORKING_VARS; // 8
pub const COL_STATE_IN_WORD_OFFSET: usize = COL_IS_FIRST_BLOCK + 1;

/// 1-bit carry per working variable for the cross-block chaining
/// `STATE_IN_WORD_next = STATE_IN_WORD_curr + AFTER_word_curr mod 2^32`.
/// Sum of 2 32-bit values < 2^33, so a single binary carry per v
/// suffices.
pub const COL_CHAIN_CARRY_BIT_OFFSET: usize =
    COL_STATE_IN_WORD_OFFSET + NUM_OUTPUT_WORDS;

/// 1-bit carry per working variable for the final digest reconstruction
/// `digest_word_v + carry_v · 2^32 = STATE_IN_WORD[v] + AFTER_word[v]`.
/// Sum of 2 32-bit values < 2^33, so a single binary carry per v
/// suffices.
pub const COL_BINDING_CARRY_BIT_OFFSET: usize =
    COL_CHAIN_CARRY_BIT_OFFSET + NUM_OUTPUT_WORDS;

/// Indicator: `1` on every row of an aggregator-populated trace, `0` on
/// legacy traces (where `populate_trace_from_hash` is used and the
/// per-invocation aggregator + binding columns are left zero). The
/// `output_byte_binding` constraint multiplies by this column to
/// avoid firing on legacy traces (where `INV_OUTPUT_BYTE` is
/// witness-zero, producing a constraint mismatch with the actual
/// digest reconstruction).
///
/// Witness-set:
///   - Aggregator populator (`populate_trace_from_hash_with_invocation_bytes`):
///     `1` on every row.
///   - Legacy populator (`populate_trace_from_hash`): leaves `0`.
///
/// Algebraically constrained by binarity (row-local) and within-trace
/// invariance (shifted; same exclusion pattern as the aggregator
/// invariance — only last-real + wrap excluded).
pub const COL_AGGREGATOR_ACTIVE: usize =
    COL_BINDING_CARRY_BIT_OFFSET + NUM_OUTPUT_WORDS;

/// Indicator: `1` on every row of the LAST block of an aggregator-
/// populated trace, `0` elsewhere. For 1-block traces,
/// `IS_LAST_BLOCK = IS_FIRST_BLOCK = 1` throughout. For sha256_pair
/// (2 blocks), `IS_LAST_BLOCK = 1` on rows 64..127. For N-block
/// traces, `IS_LAST_BLOCK = 1` on the last 64 rows.
///
/// Algebraically constrained by binarity (row-local) + within-block
/// invariance (shifted; same gate as IS_FIRST_BLOCK invariance).
/// Sum-to-NUM_ROUNDS is NOT enforced algebraically (collision
/// resistance + cross-AIR LogUp consistency makes the soundness
/// argument: a malicious prover setting IS_LAST_BLOCK on a non-last
/// block would force the digest claim to equal an intermediate
/// state_out, which by SHA-256 collision-resistance cannot match the
/// canonical hash claimed by the extract AIR).
pub const COL_IS_LAST_BLOCK: usize = COL_AGGREGATOR_ACTIVE + 1;

// ── W[t] recurrence aggregator (word-valued) ──────────────────────────
//
// Task #179 (followup to #170), generalized in Task #193: algebraically
// pin the message-schedule recurrence
//   W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16] (mod 2^32)
// for **every t ∈ 16..64** of every block, by committing the four
// summands at each recurrence row as word-valued aggregator columns
// (Option A — cumulative redundancy). The names retain the row-16
// origin (`W0_WORD`, `W9_WORD`, `SIGMA0_W1_WORD`, `SIGMA1_W14_WORD`)
// but the columns are now reused per-t to hold the row-t-relative
// addends `(W[t-16], W[t-7], σ0(W[t-15]), σ1(W[t-2]))`.
//
// These columns are NOT bit-decomposed (each holds a 32-bit integer as
// a single field element); they are excluded from `bit_validity` by
// virtue of sitting after `NUM_DATA_COLUMNS`. They are witness-populated
// on every recurrence row (t ∈ 16..64) of each block, and zero on the
// rest. The row-local `w_recurrence` constraint (in
// `sha256_constraints.rs`) gates by `Σ_{t∈16..64} SEL_ROUND[t]`, so by
// `sel_sum_01` at most one t fires per row, enforcing
//
//   W_word = σ1_addend + W9_addend + σ0_addend + W0_addend + carry · 2^32
//
// where `W_word` is reconstructed from the row's W bit columns and
// `carry` from the 2-bit `W_RECURRENCE_CARRY` column.
//
// **Soundness scope** (#193): this pins the *value* of the recurrence
// sum to W[t] on every recurrence row. The four committed addends
// remain UNBOUND to the actual W on rows t-16/t-7 and to the
// σ-helper definitions on rows t-15/t-2 (those bindings need
// cross-row / multi-shift constraints, which exceed the current
// single-shift `NUM_SHIFTED` machinery). A malicious prover can pick
// any quadruple that sums to W[t] mod 2^32 at each row — the
// constraint adds no algebraic strength beyond the within-row
// identity until the multi-shift binding lands (deferred follow-up:
// a multi-row LogUp gadget that ties each addend column on row t to
// the corresponding W (or σ-output) value on its source row).
// What this DOES pin: the 2-bit carry column is algebraically forced
// to be the correct carry of the 4-term sum against the committed
// summands, on all 48 recurrence rows per block.
pub const COL_W_RECURRENCE_W0_WORD:        usize = COL_IS_LAST_BLOCK + 1;
pub const COL_W_RECURRENCE_W9_WORD:        usize = COL_W_RECURRENCE_W0_WORD + 1;
pub const COL_W_RECURRENCE_SIGMA0_W1_WORD: usize = COL_W_RECURRENCE_W9_WORD + 1;
pub const COL_W_RECURRENCE_SIGMA1_W14_WORD: usize =
    COL_W_RECURRENCE_SIGMA0_W1_WORD + 1;

/// Task #318 / #309 / #190 mirror column: per-row byte anchor for
/// cross-AIR LogUp descriptors that want to bind an arbitrary host-derived
/// byte to this AIR's row without requiring B-side overrides on
/// `INV_OUTPUT_BYTE[0]`. Populated by
/// [`populate_trace_from_hash_with_invocation_bytes`] on every row;
/// defaults to `hash_trace.digest[0]` (back-compat). No row-local
/// constraint binds this column — soundness flows from the cross-AIR
/// LogUp closure.
pub const COL_AGGREGATOR_MIRROR_BYTE0: usize =
    COL_W_RECURRENCE_SIGMA1_W14_WORD + 1;

pub const NUM_SHA256_COLUMNS: usize = COL_AGGREGATOR_MIRROR_BYTE0 + 1;

// ──── Column index helpers ────────────────────────────────────────────

/// Index of working-var `w` (0..=7) bit `bit` (0..32) in the pre-round state.
#[inline]
pub fn before(var: usize, bit: usize) -> usize {
    debug_assert!(var < NUM_WORKING_VARS && bit < BITS_PER_WORD);
    COL_BEFORE_OFFSET + var * BITS_PER_WORD + bit
}

/// Index of working-var `w` (0..=7) bit `bit` (0..32) in the post-round state.
#[inline]
pub fn after(var: usize, bit: usize) -> usize {
    debug_assert!(var < NUM_WORKING_VARS && bit < BITS_PER_WORD);
    COL_AFTER_OFFSET + var * BITS_PER_WORD + bit
}

#[inline]
pub fn w_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_W_OFFSET + bit
}

#[inline]
pub fn k_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_K_OFFSET + bit
}

#[inline]
pub fn big_sigma0_a_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_BIG_SIGMA0_A_OFFSET + bit
}

#[inline]
pub fn xor01_s0_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_XOR01_S0_OFFSET + bit
}

#[inline]
pub fn big_sigma1_e_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_BIG_SIGMA1_E_OFFSET + bit
}

#[inline]
pub fn xor01_s1_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_XOR01_S1_OFFSET + bit
}

#[inline]
pub fn ef_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_EF_OFFSET + bit
}

#[inline]
pub fn not_e_g_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_NOT_E_G_OFFSET + bit
}

#[inline]
pub fn ch_efg_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_CH_EFG_OFFSET + bit
}

#[inline]
pub fn ab_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_AB_OFFSET + bit
}

#[inline]
pub fn ac_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_AC_OFFSET + bit
}

#[inline]
pub fn bc_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_BC_OFFSET + bit
}

#[inline]
pub fn xor_ab_ac_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_XOR_AB_AC_OFFSET + bit
}

#[inline]
pub fn maj_abc_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_MAJ_ABC_OFFSET + bit
}

#[inline]
pub fn t1_carry_bit(bit: usize) -> usize {
    debug_assert!(bit < T1_CARRY_BITS);
    COL_T1_CARRY_OFFSET + bit
}

#[inline]
pub fn xor01_ss0_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_XOR01_SS0_OFFSET + bit
}

#[inline]
pub fn small_sigma0_w_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_SMALL_SIGMA0_W_OFFSET + bit
}

#[inline]
pub fn xor01_ss1_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_XOR01_SS1_OFFSET + bit
}

#[inline]
pub fn small_sigma1_w_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_SMALL_SIGMA1_W_OFFSET + bit
}

#[inline]
pub fn w_recurrence_carry_bit(bit: usize) -> usize {
    debug_assert!(bit < W_RECURRENCE_CARRY_BITS);
    COL_W_RECURRENCE_CARRY_OFFSET + bit
}

/// Word-valued aggregator column for W[0] (the row's block) at the
/// per-block anchor row. Zero on non-anchor rows.
#[inline]
pub fn w_recurrence_w0_word() -> usize { COL_W_RECURRENCE_W0_WORD }

/// Word-valued aggregator column for W[9] at the anchor row.
#[inline]
pub fn w_recurrence_w9_word() -> usize { COL_W_RECURRENCE_W9_WORD }

/// Word-valued aggregator column for σ0(W[1]) at the anchor row.
#[inline]
pub fn w_recurrence_sigma0_w1_word() -> usize { COL_W_RECURRENCE_SIGMA0_W1_WORD }

/// Word-valued aggregator column for σ1(W[14]) at the anchor row.
#[inline]
pub fn w_recurrence_sigma1_w14_word() -> usize { COL_W_RECURRENCE_SIGMA1_W14_WORD }

#[inline]
pub fn t2_carry() -> usize {
    COL_T2_CARRY_OFFSET
}

#[inline]
pub fn a_new_carry() -> usize {
    COL_A_NEW_CARRY_OFFSET
}

#[inline]
pub fn e_new_carry() -> usize {
    COL_E_NEW_CARRY_OFFSET
}

#[inline]
pub fn sel_round(round: usize) -> usize {
    debug_assert!(round < NUM_SEL_ROUND);
    COL_SEL_ROUND_OFFSET + round
}

#[inline]
pub fn inv_input_byte(b: usize) -> usize {
    debug_assert!(b < INV_INPUT_LEN);
    COL_INV_INPUT_BYTE_OFFSET + b
}

#[inline]
pub fn inv_output_byte(b: usize) -> usize {
    debug_assert!(b < INV_OUTPUT_LEN);
    COL_INV_OUTPUT_BYTE_OFFSET + b
}

#[inline]
pub fn state_in_word(v: usize) -> usize {
    debug_assert!(v < NUM_OUTPUT_WORDS);
    COL_STATE_IN_WORD_OFFSET + v
}

#[inline]
pub fn chain_carry_bit(v: usize) -> usize {
    debug_assert!(v < NUM_OUTPUT_WORDS);
    COL_CHAIN_CARRY_BIT_OFFSET + v
}

#[inline]
pub fn binding_carry_bit(v: usize) -> usize {
    debug_assert!(v < NUM_OUTPUT_WORDS);
    COL_BINDING_CARRY_BIT_OFFSET + v
}

// Convenience named indices for the working-variable positions.
pub const VAR_A: usize = 0;
pub const VAR_B: usize = 1;
pub const VAR_C: usize = 2;
pub const VAR_D: usize = 3;
pub const VAR_E: usize = 4;
pub const VAR_F: usize = 5;
pub const VAR_G: usize = 6;
pub const VAR_H: usize = 7;

// ──── Witness population ──────────────────────────────────────────────

/// Initialize `columns` to have `NUM_SHA256_COLUMNS` empty vectors each of
/// `num_rows` `Scalar::zero(curve)` entries. Convenience for tests.
pub fn alloc_trace(num_rows: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..NUM_SHA256_COLUMNS)
        .map(|_| vec![zero.clone(); num_rows])
        .collect()
}

/// Write the 32 LSB-first bits of `word` into `row` starting at `offset`.
fn write_word_bits(
    columns: &mut [Vec<Scalar>],
    row: usize,
    offset: usize,
    word: u32,
    one: &Scalar,
    zero: &Scalar,
) {
    for bit in 0..BITS_PER_WORD {
        let v = (word >> bit) & 1;
        columns[offset + bit][row] = if v == 1 { one.clone() } else { zero.clone() };
    }
}

/// Write 8 words (a..h, 256 bits) starting at `offset`.
fn write_working_bits(
    columns: &mut [Vec<Scalar>],
    row: usize,
    offset: usize,
    words: &[u32; 8],
    one: &Scalar,
    zero: &Scalar,
) {
    for (i, &w) in words.iter().enumerate() {
        write_word_bits(columns, row, offset + i * BITS_PER_WORD, w, one, zero);
    }
}

/// Populate a single row from a `RoundTrace`.
pub fn populate_round(
    columns: &mut [Vec<Scalar>],
    row: usize,
    rt: &RoundTrace,
    curve: CurveType,
) {
    assert!(columns.len() == NUM_SHA256_COLUMNS, "columns has wrong shape");
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    // BEFORE / AFTER working variables.
    write_working_bits(columns, row, COL_BEFORE_OFFSET, &rt.before, &one, &zero);
    write_working_bits(columns, row, COL_AFTER_OFFSET,  &rt.after,  &one, &zero);

    // W and K.
    write_word_bits(columns, row, COL_W_OFFSET, rt.w, &one, &zero);
    write_word_bits(columns, row, COL_K_OFFSET, rt.k, &one, &zero);

    // Σ₀(a) + its XOR intermediate.
    let a = rt.before[VAR_A];
    let xor01_s0 = a.rotate_right(2) ^ a.rotate_right(13);
    write_word_bits(columns, row, COL_XOR01_S0_OFFSET, xor01_s0, &one, &zero);
    write_word_bits(columns, row, COL_BIG_SIGMA0_A_OFFSET, rt.big_sigma0_a, &one, &zero);

    // Σ₁(e) + its XOR intermediate.
    let e = rt.before[VAR_E];
    let xor01_s1 = e.rotate_right(6) ^ e.rotate_right(11);
    write_word_bits(columns, row, COL_XOR01_S1_OFFSET, xor01_s1, &one, &zero);
    write_word_bits(columns, row, COL_BIG_SIGMA1_E_OFFSET, rt.big_sigma1_e, &one, &zero);

    // Ch intermediates: ef, not_e_g, ch_efg.
    let f = rt.before[VAR_F];
    let g = rt.before[VAR_G];
    let ef = e & f;
    let not_e_g = (!e) & g;
    write_word_bits(columns, row, COL_EF_OFFSET, ef, &one, &zero);
    write_word_bits(columns, row, COL_NOT_E_G_OFFSET, not_e_g, &one, &zero);
    write_word_bits(columns, row, COL_CH_EFG_OFFSET, rt.ch_efg, &one, &zero);

    // Maj intermediates: ab, ac, bc, xor_ab_ac, maj_abc.
    let b = rt.before[VAR_B];
    let c = rt.before[VAR_C];
    let ab = a & b;
    let ac = a & c;
    let bc = b & c;
    let xor_ab_ac = ab ^ ac;
    write_word_bits(columns, row, COL_AB_OFFSET, ab, &one, &zero);
    write_word_bits(columns, row, COL_AC_OFFSET, ac, &one, &zero);
    write_word_bits(columns, row, COL_BC_OFFSET, bc, &one, &zero);
    write_word_bits(columns, row, COL_XOR_AB_AC_OFFSET, xor_ab_ac, &one, &zero);
    write_word_bits(columns, row, COL_MAJ_ABC_OFFSET, rt.maj_abc, &one, &zero);

    // ── Carry columns ──
    // T₁ sum (5 terms): h + Σ₁(e) + Ch + K + W.
    // Each term ≤ 2^32 − 1, so sum ≤ 5·(2^32 − 1) < 5·2^32.
    let h = rt.before[VAR_H];
    let t1_full_sum: u64 = (h as u64)
        + (rt.big_sigma1_e as u64)
        + (rt.ch_efg as u64)
        + (rt.k as u64)
        + (rt.w as u64);
    let t1_carry_value: u64 = t1_full_sum >> 32;
    debug_assert!(t1_carry_value < 8, "T1 carry should fit in 3 bits");
    for bit in 0..T1_CARRY_BITS {
        let v = (t1_carry_value >> bit) & 1;
        columns[COL_T1_CARRY_OFFSET + bit][row] =
            if v == 1 { one.clone() } else { zero.clone() };
    }

    // T₂ sum (2 terms): Σ₀(a) + Maj.
    let t2_full_sum: u64 = (rt.big_sigma0_a as u64) + (rt.maj_abc as u64);
    let t2_carry_value: u64 = t2_full_sum >> 32;
    debug_assert!(t2_carry_value <= 1, "T2 carry is 1 bit");
    columns[COL_T2_CARRY_OFFSET][row] =
        if t2_carry_value == 1 { one.clone() } else { zero.clone() };

    // new-a = T₁ + T₂ (2 terms, 1-bit carry).
    let new_a_sum: u64 = (rt.t1 as u64) + (rt.t2 as u64);
    let a_new_carry_value: u64 = new_a_sum >> 32;
    debug_assert!(a_new_carry_value <= 1);
    columns[COL_A_NEW_CARRY_OFFSET][row] =
        if a_new_carry_value == 1 { one.clone() } else { zero.clone() };

    // new-e = d + T₁ (2 terms, 1-bit carry).
    let d = rt.before[VAR_D];
    let new_e_sum: u64 = (d as u64) + (rt.t1 as u64);
    let e_new_carry_value: u64 = new_e_sum >> 32;
    debug_assert!(e_new_carry_value <= 1);
    columns[COL_E_NEW_CARRY_OFFSET][row] =
        if e_new_carry_value == 1 { one.clone() } else { zero.clone() };

    // ── Message-schedule σ helpers for this row's W ──
    // σ0(W) = ROTR_7(W) XOR ROTR_18(W) XOR SHR_3(W)
    // σ1(W) = ROTR_17(W) XOR ROTR_19(W) XOR SHR_10(W)
    let w = rt.w;
    let small_sigma0_w = w.rotate_right(7) ^ w.rotate_right(18) ^ (w >> 3);
    let small_sigma1_w = w.rotate_right(17) ^ w.rotate_right(19) ^ (w >> 10);
    let xor01_ss0 = w.rotate_right(7) ^ w.rotate_right(18);
    let xor01_ss1 = w.rotate_right(17) ^ w.rotate_right(19);
    write_word_bits(columns, row, COL_XOR01_SS0_OFFSET, xor01_ss0, &one, &zero);
    write_word_bits(columns, row, COL_SMALL_SIGMA0_W_OFFSET, small_sigma0_w, &one, &zero);
    write_word_bits(columns, row, COL_XOR01_SS1_OFFSET, xor01_ss1, &one, &zero);
    write_word_bits(columns, row, COL_SMALL_SIGMA1_W_OFFSET, small_sigma1_w, &one, &zero);

    // One-hot round selector.
    columns[sel_round(rt.round)][row] = one.clone();
}

/// Populate a full `HashTrace` into a freshly allocated column set. The
/// returned trace has one row per round across all blocks
/// (`blocks.len() * 64` rows).
///
/// Per-invocation byte aggregator columns
/// ([`COL_INV_INPUT_BYTE_OFFSET`], [`COL_INV_OUTPUT_BYTE_OFFSET`],
/// [`COL_IS_FIRST_INV_ROW`]) are populated from
/// `invocation_input_bytes` (the 64-byte conceptual "input" — for
/// `sha256_pair(left, right)` this is `left || right`; for
/// general-purpose `sha256(input)` this is the first 64 bytes of the
/// padded input) and `hash_trace.digest`. The same input/output bytes
/// are written to every row (invariance enforced by
/// `sha256_constraints`); `IS_FIRST_INV_ROW` is `1` only on row 0.
pub fn populate_trace_from_hash_with_invocation_bytes(
    hash_trace: &HashTrace,
    invocation_input_bytes: &[u8; INV_INPUT_LEN],
    curve: CurveType,
) -> Vec<Vec<Scalar>> {
    let num_rows = hash_trace.blocks.len() * NUM_ROUNDS;
    let mut columns = alloc_trace(num_rows, curve);
    let mut row = 0usize;
    for block in &hash_trace.blocks {
        for rt in &block.rounds {
            populate_round(&mut columns, row, rt, curve);
            row += 1;
        }
    }
    debug_assert_eq!(row, num_rows);

    // Fill per-invocation byte aggregator columns. Same input/output on
    // every row (invariance constraint enforces this in the AIR).
    let one = Scalar::one(curve);
    for r in 0..num_rows {
        for b in 0..INV_INPUT_LEN {
            columns[COL_INV_INPUT_BYTE_OFFSET + b][r] =
                Scalar::from_u64(invocation_input_bytes[b] as u64, curve);
        }
        for b in 0..INV_OUTPUT_LEN {
            columns[COL_INV_OUTPUT_BYTE_OFFSET + b][r] =
                Scalar::from_u64(hash_trace.digest[b] as u64, curve);
        }
    }
    // IS_FIRST_INV_ROW = 1 only on row 0.
    if num_rows > 0 {
        columns[COL_IS_FIRST_INV_ROW][0] = one.clone();
    }
    // IS_FIRST_BLOCK = 1 on every round of block 0, 0 elsewhere.
    for r in 0..num_rows.min(NUM_ROUNDS) {
        columns[COL_IS_FIRST_BLOCK][r] = one.clone();
    }

    // ── Output-binding aggregator columns (general multi-block) ──
    //
    // STATE_IN_WORD[v] at row r = state_in for the block containing
    //   row r. Computed by chaining: state_in[block 0] = INITIAL_HASH;
    //   state_in[block k+1] = state_in[block k] + AFTER_word[63 of
    //   block k] mod 2^32 componentwise.
    //
    // CHAIN_CARRY_BIT[v] at row r = block-end transition carry for v
    //   between row r and row r+1, when r is a block-end row.
    //   Specifically: chain_carry[v][r] = 1 iff state_in[curr block] +
    //   AFTER_word[v][r] >= 2^32. Other rows have chain_carry = 0
    //   (witness; constraint only fires at block-end gates).
    //
    // BINDING_CARRY_BIT[v] at row r = final digest reconstruction
    //   carry for v at the LAST block's last round. Computed at row
    //   = (num_blocks-1)*64 + 63. Other rows have value 0.
    let num_blocks = hash_trace.blocks.len();
    if num_blocks >= 1 {
        // state_in chain: starts at INITIAL_HASH; updated at each block
        // boundary by adding AFTER_word[63 of that block].
        let mut state_in: [u64; NUM_OUTPUT_WORDS] = {
            let mut s = [0u64; NUM_OUTPUT_WORDS];
            for v in 0..NUM_OUTPUT_WORDS {
                s[v] = crate::sha256::INITIAL_HASH[v] as u64;
            }
            s
        };
        for block_idx in 0..num_blocks {
            // Populate STATE_IN_WORD on rows of this block.
            let row_base = block_idx * NUM_ROUNDS;
            for v in 0..NUM_OUTPUT_WORDS {
                let val = Scalar::from_u64(state_in[v], curve);
                for r in row_base..row_base + NUM_ROUNDS {
                    if r < num_rows {
                        columns[COL_STATE_IN_WORD_OFFSET + v][r] = val.clone();
                    }
                }
            }
            let after_v = &hash_trace.blocks[block_idx].rounds[NUM_ROUNDS - 1].after;
            // Compute chain_carry / binding_carry at the block-end row
            // (row_base + NUM_ROUNDS - 1).
            let block_end_row = row_base + NUM_ROUNDS - 1;
            if block_idx + 1 < num_blocks {
                // Cross-block chain: state_in_next = state_in + AFTER_word
                // mod 2^32. Carry per v.
                for v in 0..NUM_OUTPUT_WORDS {
                    let after = after_v[v] as u64;
                    let sum = state_in[v] + after;
                    let carry = (sum >> 32) as u64;
                    debug_assert!(carry <= 1);
                    let scalar = if carry == 1 {
                        one.clone()
                    } else {
                        Scalar::zero(curve)
                    };
                    if block_end_row < num_rows {
                        columns[COL_CHAIN_CARRY_BIT_OFFSET + v][block_end_row] =
                            scalar;
                    }
                    // Update state_in for next block.
                    state_in[v] = sum & 0xFFFF_FFFF;
                }
            } else {
                // Last block: compute binding carry.
                for v in 0..NUM_OUTPUT_WORDS {
                    let after = after_v[v] as u64;
                    let sum = state_in[v] + after;
                    let carry = (sum >> 32) as u64;
                    debug_assert!(carry <= 1);
                    let scalar = if carry == 1 {
                        one.clone()
                    } else {
                        Scalar::zero(curve)
                    };
                    if block_end_row < num_rows {
                        columns[COL_BINDING_CARRY_BIT_OFFSET + v][block_end_row] =
                            scalar;
                    }
                }
            }
        }

        // IS_LAST_BLOCK = 1 on the last block's NUM_ROUNDS rows.
        let last_block_row_base = (num_blocks - 1) * NUM_ROUNDS;
        for r in last_block_row_base..num_rows {
            columns[COL_IS_LAST_BLOCK][r] = one.clone();
        }
    }
    // AGGREGATOR_ACTIVE = 1 on every row of an aggregator-populated trace.
    for r in 0..num_rows {
        columns[COL_AGGREGATOR_ACTIVE][r] = one.clone();
    }
    // Task #318 mirror: per-row byte anchor for cross-AIR LogUp
    // descriptors. Defaults to `hash_trace.digest[0]` on every active
    // row; callers wanting a different byte may overwrite this column
    // host-side after populating the trace.
    let mirror_byte = hash_trace.digest[0];
    let mirror_scalar = Scalar::from_u64(mirror_byte as u64, curve);
    for r in 0..num_rows {
        columns[COL_AGGREGATOR_MIRROR_BYTE0][r] = mirror_scalar.clone();
    }
    populate_w_recurrence_carry(&mut columns, hash_trace, curve);
    columns
}

/// Backwards-compatible populator that leaves the per-invocation byte
/// aggregator columns zero. Use this when the caller doesn't need the
/// cross-AIR LogUp linkage to [`crate::sha256_extract`].
pub fn populate_trace_from_hash(
    hash_trace: &HashTrace,
    curve: CurveType,
) -> Vec<Vec<Scalar>> {
    let num_rows = hash_trace.blocks.len() * NUM_ROUNDS;
    let mut columns = alloc_trace(num_rows, curve);
    let mut row = 0usize;
    for block in &hash_trace.blocks {
        for rt in &block.rounds {
            populate_round(&mut columns, row, rt, curve);
            row += 1;
        }
    }
    debug_assert_eq!(row, num_rows);
    populate_w_recurrence_carry(&mut columns, hash_trace, curve);
    columns
}

/// Witness-populate the 2-bit `W_RECURRENCE_CARRY` column AND the four
/// word-valued aggregator columns for EVERY recurrence row t ∈ 16..64
/// of each block (Task #193). At row `block_idx * NUM_ROUNDS + t`,
/// commits the addends
///   σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16] = W[t] + carry·2^32
/// as the four `COL_W_RECURRENCE_*_WORD` columns plus the 2-bit carry.
/// Non-recurrence rows (t ∈ 0..16) are left zero, and the corresponding
/// row-local `w_recurrence` constraint is gated by `Σ_{t∈16..64}
/// SEL_ROUND[t]` so the body vanishes there.
///
/// Naming retained from the row-16 anchor pattern (#179): the same four
/// columns are reused on every recurrence row to hold the row-specific
/// addends. By `sel_sum_01`, at most one selector is active per row, so
/// at most one t's recurrence fires.
fn populate_w_recurrence_carry(
    columns: &mut [Vec<Scalar>],
    hash_trace: &HashTrace,
    curve: CurveType,
) {
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);
    for (block_idx, block) in hash_trace.blocks.iter().enumerate() {
        if block.rounds.len() <= 16 {
            continue;
        }
        for t in 16..NUM_ROUNDS.min(block.rounds.len()) {
            // Addends use the row-t-relative offsets.
            let w_tm16 = block.rounds[t - 16].w as u64;
            let w_tm15 = block.rounds[t - 15].w as u32;
            let w_tm7 = block.rounds[t - 7].w as u64;
            let w_tm2 = block.rounds[t - 2].w as u32;
            let w_t = block.rounds[t].w as u64;
            let s0 = (w_tm15.rotate_right(7)
                ^ w_tm15.rotate_right(18)
                ^ (w_tm15 >> 3)) as u64;
            let s1 = (w_tm2.rotate_right(17)
                ^ w_tm2.rotate_right(19)
                ^ (w_tm2 >> 10)) as u64;
            let full_sum = s1 + w_tm7 + s0 + w_tm16;
            let carry = full_sum >> 32;
            debug_assert!(carry < 4, "W[{}] recurrence carry must fit in 2 bits", t);
            debug_assert_eq!(
                full_sum & 0xFFFF_FFFF,
                w_t,
                "W[{}] recurrence violated", t
            );
            let row = block_idx * NUM_ROUNDS + t;
            if row >= columns[0].len() {
                continue;
            }
            for bit in 0..W_RECURRENCE_CARRY_BITS {
                let v = (carry >> bit) & 1;
                columns[COL_W_RECURRENCE_CARRY_OFFSET + bit][row] =
                    if v == 1 { one.clone() } else { zero.clone() };
            }
            // Word-valued aggregator columns (Task #179 → #193 generalized):
            // commit the four summands at row t. Columns are reused per t.
            columns[COL_W_RECURRENCE_W0_WORD][row]        = Scalar::from_u64(w_tm16, curve);
            columns[COL_W_RECURRENCE_W9_WORD][row]        = Scalar::from_u64(w_tm7, curve);
            columns[COL_W_RECURRENCE_SIGMA0_W1_WORD][row] = Scalar::from_u64(s0, curve);
            columns[COL_W_RECURRENCE_SIGMA1_W14_WORD][row] = Scalar::from_u64(s1, curve);
        }
    }
}

// ──── Constraint evaluation ───────────────────────────────────────────

/// A single constraint body label + its per-row evaluations.
pub struct ConstraintEval {
    pub label: String,
    pub values: Vec<Scalar>,
}

/// Helper: XOR(a, b) = a + b − 2·a·b as a field expression.
#[inline]
fn xor_expr(a: &Scalar, b: &Scalar, two: &Scalar) -> Scalar {
    let ab = a.mul(b);
    a.add(b).sub(&two.mul(&ab))
}

/// Helper: evaluate the 32-bit integer value of a word's bit columns at
/// `row` via Σ bit[i]·2^i, given a starting `offset` and length `n`.
fn word_value_at(
    columns: &[&Vec<Scalar>],
    offset: usize,
    n: usize,
    row: usize,
    curve: CurveType,
) -> Scalar {
    let two = Scalar::from_u64(2, curve);
    let mut pow = Scalar::one(curve);
    let mut acc = Scalar::zero(curve);
    for bit in 0..n {
        let c = &columns[offset + bit][row];
        acc = acc.add(&pow.mul(c));
        pow = pow.mul(&two);
    }
    acc
}

/// Evaluate the SHA-256 AIR constraint bodies on a populated trace.
///
/// Returns a vector of `(label, per-row values)`. A valid witness
/// produces zero at every row for every constraint. The caller is
/// responsible for passing a Fiat-Shamir challenge `beta` (used to
/// linearly combine the many bit-level sub-constraints inside each
/// category so the returned vector stays a manageable size).
pub fn evaluate_constraints(
    columns: &[&Vec<Scalar>],
    beta: &Scalar,
) -> Vec<ConstraintEval> {
    assert!(
        columns.len() == NUM_SHA256_COLUMNS,
        "expected {} columns, got {}", NUM_SHA256_COLUMNS, columns.len()
    );
    let num_rows = columns[0].len();
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one  = Scalar::one(curve);
    let two  = Scalar::from_u64(2, curve);
    // 2^32 as a field scalar.
    let two_pow_32 = Scalar::from_u64(1u64 << 32, curve);

    let mut result: Vec<ConstraintEval> = Vec::new();

    // ── 1. Bit validity: b·(b − 1) = 0 for every data column. ──
    let mut bit_validity = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for col_idx in 0..NUM_DATA_COLUMNS {
            let col = columns[col_idx];
            for (row, cell) in col.iter().enumerate() {
                let diff = cell.sub(&one);
                let body = cell.mul(&diff); // b·(b − 1)
                let term = beta_pow.mul(&body);
                bit_validity[row] = bit_validity[row].add(&term);
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "bit_validity".into(), values: bit_validity });

    // ── 2. Σ₀(a) definition: two-stage XOR. ──
    // xor_01_s0[i]    = rot(a,2)[i]  XOR rot(a,13)[i]
    //                = a[(i−2) mod 32] XOR a[(i−13) mod 32]
    // big_sigma0_a[i] = xor_01_s0[i] XOR rot(a,22)[i]
    //                = xor_01_s0[i] XOR a[(i−22) mod 32]
    //
    // (For rotate-right by k, bit i comes from bit (i − k) mod 32 of the
    // original, because rotate_right(x, k) has its LSB drawn from bit k.)
    let mut sigma0_def = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for bit in 0..BITS_PER_WORD {
            // rot(a, 2) at output position bit ← a at (bit + 2) mod 32.
            let a_rot2  = columns[before(VAR_A, (bit + 2)  % BITS_PER_WORD)];
            let a_rot13 = columns[before(VAR_A, (bit + 13) % BITS_PER_WORD)];
            let a_rot22 = columns[before(VAR_A, (bit + 22) % BITS_PER_WORD)];
            let xor01   = columns[xor01_s0_bit(bit)];
            let sig0    = columns[big_sigma0_a_bit(bit)];
            for row in 0..num_rows {
                // Stage 1: xor01 = a_rot2 XOR a_rot13
                let x1 = xor_expr(&a_rot2[row], &a_rot13[row], &two);
                let body1 = xor01[row].sub(&x1);
                sigma0_def[row] = sigma0_def[row].add(&beta_pow.mul(&body1));
            }
            beta_pow = beta_pow.mul(beta);
            for row in 0..num_rows {
                // Stage 2: sig0 = xor01 XOR a_rot22
                let x2 = xor_expr(&xor01[row], &a_rot22[row], &two);
                let body2 = sig0[row].sub(&x2);
                sigma0_def[row] = sigma0_def[row].add(&beta_pow.mul(&body2));
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "big_sigma0_a_definition".into(), values: sigma0_def });

    // ── 3. Σ₁(e) definition: two-stage XOR with rotations 6, 11, 25. ──
    let mut sigma1_def = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for bit in 0..BITS_PER_WORD {
            let e_rot6  = columns[before(VAR_E, (bit + 6)  % BITS_PER_WORD)];
            let e_rot11 = columns[before(VAR_E, (bit + 11) % BITS_PER_WORD)];
            let e_rot25 = columns[before(VAR_E, (bit + 25) % BITS_PER_WORD)];
            let xor01   = columns[xor01_s1_bit(bit)];
            let sig1    = columns[big_sigma1_e_bit(bit)];
            for row in 0..num_rows {
                let x1 = xor_expr(&e_rot6[row], &e_rot11[row], &two);
                let body1 = xor01[row].sub(&x1);
                sigma1_def[row] = sigma1_def[row].add(&beta_pow.mul(&body1));
            }
            beta_pow = beta_pow.mul(beta);
            for row in 0..num_rows {
                let x2 = xor_expr(&xor01[row], &e_rot25[row], &two);
                let body2 = sig1[row].sub(&x2);
                sigma1_def[row] = sigma1_def[row].add(&beta_pow.mul(&body2));
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "big_sigma1_e_definition".into(), values: sigma1_def });

    // ── 4. Ch definition. ──
    // ef[i]       = e[i]·f[i]
    // not_e_g[i]  = (1 − e[i])·g[i]
    // ch_efg[i]   = ef[i] XOR not_e_g[i]
    let mut ch_def = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for bit in 0..BITS_PER_WORD {
            let e  = columns[before(VAR_E, bit)];
            let f  = columns[before(VAR_F, bit)];
            let g  = columns[before(VAR_G, bit)];
            let ef = columns[ef_bit(bit)];
            let ng = columns[not_e_g_bit(bit)];
            let ch = columns[ch_efg_bit(bit)];
            for row in 0..num_rows {
                // ef = e·f
                let body1 = ef[row].sub(&e[row].mul(&f[row]));
                ch_def[row] = ch_def[row].add(&beta_pow.mul(&body1));
            }
            beta_pow = beta_pow.mul(beta);
            for row in 0..num_rows {
                // not_e_g = (1 − e)·g
                let not_e = one.sub(&e[row]);
                let body2 = ng[row].sub(&not_e.mul(&g[row]));
                ch_def[row] = ch_def[row].add(&beta_pow.mul(&body2));
            }
            beta_pow = beta_pow.mul(beta);
            for row in 0..num_rows {
                // ch_efg = ef XOR not_e_g
                let x = xor_expr(&ef[row], &ng[row], &two);
                let body3 = ch[row].sub(&x);
                ch_def[row] = ch_def[row].add(&beta_pow.mul(&body3));
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "ch_definition".into(), values: ch_def });

    // ── 5. Maj definition. ──
    // ab[i]       = a[i]·b[i]
    // ac[i]       = a[i]·c[i]
    // bc[i]       = b[i]·c[i]
    // xor_ab_ac[i] = ab[i] XOR ac[i]
    // maj_abc[i]  = xor_ab_ac[i] XOR bc[i]
    let mut maj_def = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for bit in 0..BITS_PER_WORD {
            let a  = columns[before(VAR_A, bit)];
            let b  = columns[before(VAR_B, bit)];
            let c  = columns[before(VAR_C, bit)];
            let ab = columns[ab_bit(bit)];
            let ac = columns[ac_bit(bit)];
            let bc = columns[bc_bit(bit)];
            let xab_ac = columns[xor_ab_ac_bit(bit)];
            let mj = columns[maj_abc_bit(bit)];
            for row in 0..num_rows {
                let body1 = ab[row].sub(&a[row].mul(&b[row]));
                maj_def[row] = maj_def[row].add(&beta_pow.mul(&body1));
            }
            beta_pow = beta_pow.mul(beta);
            for row in 0..num_rows {
                let body2 = ac[row].sub(&a[row].mul(&c[row]));
                maj_def[row] = maj_def[row].add(&beta_pow.mul(&body2));
            }
            beta_pow = beta_pow.mul(beta);
            for row in 0..num_rows {
                let body3 = bc[row].sub(&b[row].mul(&c[row]));
                maj_def[row] = maj_def[row].add(&beta_pow.mul(&body3));
            }
            beta_pow = beta_pow.mul(beta);
            for row in 0..num_rows {
                let x = xor_expr(&ab[row], &ac[row], &two);
                let body4 = xab_ac[row].sub(&x);
                maj_def[row] = maj_def[row].add(&beta_pow.mul(&body4));
            }
            beta_pow = beta_pow.mul(beta);
            for row in 0..num_rows {
                let x = xor_expr(&xab_ac[row], &bc[row], &two);
                let body5 = mj[row].sub(&x);
                maj_def[row] = maj_def[row].add(&beta_pow.mul(&body5));
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "maj_definition".into(), values: maj_def });

    // ── 6. T₁, T₂, new-a, new-e summations (one consolidated category). ──
    //
    // Let value(X) = Σ X[i]·2^i be the 32-bit value reconstruction on a
    // row. We relate the 32-bit values directly — no intermediate T1/T2
    // witness word is materialized; we recover them via:
    //
    //   T1_value = (new_e_value + e_new_carry·2^32) − d_value
    //   T2_value = (new_a_value + a_new_carry·2^32) − T1_value
    //
    // Then impose:
    //   (A) h + Σ₁(e) + Ch + K + W
    //         = T1_value + t1_carry_value · 2^32         (5-term add)
    //   (B) Σ₀(a) + Maj
    //         = T2_value + t2_carry · 2^32               (2-term add)
    //
    // These two relations together with the four carry-bit columns
    // exactly encode the row update a' = T1 + T2 and e' = d + T1 modulo
    // 2^32.
    let mut adds = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for row in 0..num_rows {
            // Reconstruct the relevant 32-bit values for relation (A).
            let h_val   = word_value_at(columns, COL_BEFORE_OFFSET + VAR_H * BITS_PER_WORD, BITS_PER_WORD, row, curve);
            let d_val   = word_value_at(columns, COL_BEFORE_OFFSET + VAR_D * BITS_PER_WORD, BITS_PER_WORD, row, curve);
            let s1_val  = word_value_at(columns, COL_BIG_SIGMA1_E_OFFSET, BITS_PER_WORD, row, curve);
            let ch_val  = word_value_at(columns, COL_CH_EFG_OFFSET, BITS_PER_WORD, row, curve);
            let k_val   = word_value_at(columns, COL_K_OFFSET, BITS_PER_WORD, row, curve);
            let w_val   = word_value_at(columns, COL_W_OFFSET, BITS_PER_WORD, row, curve);
            let new_e_val = word_value_at(columns, COL_AFTER_OFFSET + VAR_E * BITS_PER_WORD, BITS_PER_WORD, row, curve);

            // T1 carry value = Σ bit[i]·2^i across the 3 T1 carry columns.
            let mut t1c = zero.clone();
            let mut p = one.clone();
            for bit in 0..T1_CARRY_BITS {
                t1c = t1c.add(&p.mul(&columns[COL_T1_CARRY_OFFSET + bit][row]));
                p = p.mul(&two);
            }
            let enc = &columns[COL_E_NEW_CARRY_OFFSET][row];

            // T1_value = new_e_value + e_new_carry·2^32 − d_value.
            let t1_val = new_e_val.add(&enc.mul(&two_pow_32)).sub(&d_val);

            // (A) lhs = h + Σ₁(e) + Ch + K + W
            //     rhs = T1_value + t1_carry_value · 2^32
            let lhs_a = h_val.add(&s1_val).add(&ch_val).add(&k_val).add(&w_val);
            let rhs_a = t1_val.add(&t1c.mul(&two_pow_32));
            let body_a = lhs_a.sub(&rhs_a);
            adds[row] = adds[row].add(&beta_pow.mul(&body_a));
        }
        beta_pow = beta_pow.mul(beta);
        for row in 0..num_rows {
            let s0_val  = word_value_at(columns, COL_BIG_SIGMA0_A_OFFSET, BITS_PER_WORD, row, curve);
            let maj_val = word_value_at(columns, COL_MAJ_ABC_OFFSET, BITS_PER_WORD, row, curve);
            let d_val   = word_value_at(columns, COL_BEFORE_OFFSET + VAR_D * BITS_PER_WORD, BITS_PER_WORD, row, curve);
            let new_a_val = word_value_at(columns, COL_AFTER_OFFSET + VAR_A * BITS_PER_WORD, BITS_PER_WORD, row, curve);
            let new_e_val = word_value_at(columns, COL_AFTER_OFFSET + VAR_E * BITS_PER_WORD, BITS_PER_WORD, row, curve);
            let t2c   = &columns[COL_T2_CARRY_OFFSET][row];
            let anc   = &columns[COL_A_NEW_CARRY_OFFSET][row];
            let enc   = &columns[COL_E_NEW_CARRY_OFFSET][row];

            let t1_val = new_e_val.add(&enc.mul(&two_pow_32)).sub(&d_val);
            let t2_val = new_a_val.add(&anc.mul(&two_pow_32)).sub(&t1_val);

            // (B) lhs = Σ₀(a) + Maj
            //     rhs = T2_value + t2_carry · 2^32
            let lhs_b = s0_val.add(&maj_val);
            let rhs_b = t2_val.add(&t2c.mul(&two_pow_32));
            let body_b = lhs_b.sub(&rhs_b);
            adds[row] = adds[row].add(&beta_pow.mul(&body_b));
        }
    }
    result.push(ConstraintEval { label: "round_additions".into(), values: adds });

    // ── 7. Passthrough: b' = a, c' = b, d' = c, f' = e, g' = f, h' = g. ──
    // Enforced bit-by-bit.
    let passthrough_pairs: [(usize, usize); 6] = [
        (VAR_B, VAR_A),
        (VAR_C, VAR_B),
        (VAR_D, VAR_C),
        (VAR_F, VAR_E),
        (VAR_G, VAR_F),
        (VAR_H, VAR_G),
    ];
    let mut passthrough = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for &(dst, src) in &passthrough_pairs {
            for bit in 0..BITS_PER_WORD {
                let src_col = columns[before(src, bit)];
                let dst_col = columns[after(dst, bit)];
                for row in 0..num_rows {
                    let body = dst_col[row].sub(&src_col[row]);
                    let term = beta_pow.mul(&body);
                    passthrough[row] = passthrough[row].add(&term);
                }
                beta_pow = beta_pow.mul(beta);
            }
        }
    }
    result.push(ConstraintEval { label: "passthrough".into(), values: passthrough });

    // ── 8. K binding: for each bit i,
    //   Σ_k sel_round_k · (k_bit[i] − RC_k_bit[i]) = 0
    // If exactly one selector is active (which the selector-sum constraint
    // already ensures for active rows), this pins k_bit to RC[round][i].
    let mut k_binding = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for bit in 0..BITS_PER_WORD {
            let kb = columns[k_bit(bit)];
            for k in 0..NUM_ROUNDS {
                let rc_bit = ((ROUND_CONSTANTS[k] >> bit) & 1) as u64;
                let rc_scalar = if rc_bit == 1 { one.clone() } else { zero.clone() };
                let sel = columns[sel_round(k)];
                for row in 0..num_rows {
                    let diff = kb[row].sub(&rc_scalar);
                    let gated = sel[row].mul(&diff);
                    let term = beta_pow.mul(&gated);
                    k_binding[row] = k_binding[row].add(&term);
                }
                beta_pow = beta_pow.mul(beta);
            }
        }
    }
    result.push(ConstraintEval { label: "k_binding".into(), values: k_binding });

    // ── 9a. Selector binary: sel_round_k · (sel_round_k − 1) = 0. ──
    let mut sel_binary = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for k in 0..NUM_SEL_ROUND {
            let col = columns[sel_round(k)];
            for (row, cell) in col.iter().enumerate() {
                let body = cell.mul(&cell.sub(&one));
                let term = beta_pow.mul(&body);
                sel_binary[row] = sel_binary[row].add(&term);
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "sel_binary".into(), values: sel_binary });

    // ── 9b. Selector sum ∈ {0, 1} per row:
    // (Σ sel_round_k) · (Σ sel_round_k − 1) = 0. ──
    let mut sel_sum = vec![zero.clone(); num_rows];
    for row in 0..num_rows {
        let mut sum = zero.clone();
        for k in 0..NUM_SEL_ROUND {
            sum = sum.add(&columns[sel_round(k)][row]);
        }
        let body = sum.mul(&sum.sub(&one));
        sel_sum[row] = body;
    }
    result.push(ConstraintEval { label: "sel_sum_01".into(), values: sel_sum });

    // ── 10. small_sigma0(W) definition: two-stage XOR. ──
    //   xor01_ss0[i] = W[(i+7) mod 32] XOR W[(i+18) mod 32]
    //   small_sigma0_w[i] = xor01_ss0[i] XOR SHR_3(W)[i]
    // where SHR_3(W)[i] = W[i+3] if i < 29, else 0.
    let mut ss0_def = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for bit in 0..BITS_PER_WORD {
            let w_rot7 = columns[w_bit((bit + 7) % BITS_PER_WORD)];
            let w_rot18 = columns[w_bit((bit + 18) % BITS_PER_WORD)];
            let xor01 = columns[xor01_ss0_bit(bit)];
            let ss0 = columns[small_sigma0_w_bit(bit)];
            for row in 0..num_rows {
                // Stage 1: xor01 = w_rot7 XOR w_rot18.
                let x1 = xor_expr(&w_rot7[row], &w_rot18[row], &two);
                let body1 = xor01[row].sub(&x1);
                ss0_def[row] = ss0_def[row].add(&beta_pow.mul(&body1));
            }
            beta_pow = beta_pow.mul(beta);
            // Stage 2: ss0 = xor01 XOR SHR_3(W)[i].
            if bit + 3 < BITS_PER_WORD {
                let w_shr3 = columns[w_bit(bit + 3)];
                for row in 0..num_rows {
                    let x2 = xor_expr(&xor01[row], &w_shr3[row], &two);
                    let body2 = ss0[row].sub(&x2);
                    ss0_def[row] = ss0_def[row].add(&beta_pow.mul(&body2));
                }
            } else {
                // SHR_3 bit is forced 0 (high 3 bits). XOR with 0 = identity:
                // ss0[i] = xor01[i].
                for row in 0..num_rows {
                    let body2 = ss0[row].sub(&xor01[row]);
                    ss0_def[row] = ss0_def[row].add(&beta_pow.mul(&body2));
                }
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "small_sigma0_w_definition".into(), values: ss0_def });

    // ── 11. small_sigma1(W) definition: two-stage XOR with rotations 17/19 and shift 10. ──
    let mut ss1_def = vec![zero.clone(); num_rows];
    {
        let mut beta_pow = one.clone();
        for bit in 0..BITS_PER_WORD {
            let w_rot17 = columns[w_bit((bit + 17) % BITS_PER_WORD)];
            let w_rot19 = columns[w_bit((bit + 19) % BITS_PER_WORD)];
            let xor01 = columns[xor01_ss1_bit(bit)];
            let ss1 = columns[small_sigma1_w_bit(bit)];
            for row in 0..num_rows {
                let x1 = xor_expr(&w_rot17[row], &w_rot19[row], &two);
                let body1 = xor01[row].sub(&x1);
                ss1_def[row] = ss1_def[row].add(&beta_pow.mul(&body1));
            }
            beta_pow = beta_pow.mul(beta);
            if bit + 10 < BITS_PER_WORD {
                let w_shr10 = columns[w_bit(bit + 10)];
                for row in 0..num_rows {
                    let x2 = xor_expr(&xor01[row], &w_shr10[row], &two);
                    let body2 = ss1[row].sub(&x2);
                    ss1_def[row] = ss1_def[row].add(&beta_pow.mul(&body2));
                }
            } else {
                for row in 0..num_rows {
                    let body2 = ss1[row].sub(&xor01[row]);
                    ss1_def[row] = ss1_def[row].add(&beta_pow.mul(&body2));
                }
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    result.push(ConstraintEval { label: "small_sigma1_w_definition".into(), values: ss1_def });

    result
}

/// Evaluate cross-row (transition) constraints: `after(row r)[var][bit]
/// == before(row r+1)[var][bit]` for all 8 working variables, except on
/// rows that are block boundaries (where `(r+1) mod 64 == 0`).
///
/// Returns a single consolidated per-row vector (length `num_rows`). The
/// entry at row r represents the constraint between r and r+1; the last
/// row's entry is always zero.
pub fn evaluate_cross_row(
    columns: &[&Vec<Scalar>],
    beta: &Scalar,
) -> ConstraintEval {
    let num_rows = columns[0].len();
    let curve = beta.curve_type();
    let zero = Scalar::zero(curve);
    let one  = Scalar::one(curve);

    let mut values = vec![zero.clone(); num_rows];
    let mut beta_pow = one.clone();
    for var in 0..NUM_WORKING_VARS {
        for bit in 0..BITS_PER_WORD {
            let af = columns[after(var, bit)];
            let bf = columns[before(var, bit)];
            for row in 0..num_rows.saturating_sub(1) {
                // Skip block boundary: next row starts a fresh block whose
                // BEFORE values are its public `state_in`, not this row's
                // AFTER (chained only externally).
                if (row + 1) % NUM_ROUNDS == 0 { continue; }
                let body = af[row].sub(&bf[row + 1]);
                let term = beta_pow.mul(&body);
                values[row] = values[row].add(&term);
            }
            beta_pow = beta_pow.mul(beta);
        }
    }
    ConstraintEval { label: "cross_row_transition".into(), values }
}

/// Number of consolidated (β-RLC) constraint categories returned by
/// [`evaluate_constraints`]. Plus one cross-row category from
/// [`evaluate_cross_row`].
///
/// Categories 10/11 are the per-row σ0(W)/σ1(W) definitional constraints
/// added for the message-schedule recurrence.
pub const NUM_CONSTRAINT_CATEGORIES: usize = 12;

// ──── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::CurveType;
    use crate::sha256::{sha256_compress_witness, sha256_witness, INITIAL_HASH};

    fn beta_challenge() -> Scalar {
        // A fixed non-zero challenge. (Testing determinism; avoids Fiat-Shamir.)
        Scalar::from_u64(12345, CurveType::Bls48581)
    }

    fn col_refs(columns: &[Vec<Scalar>]) -> Vec<&Vec<Scalar>> {
        columns.iter().collect()
    }

    /// One compression of INITIAL_HASH on the "abc" padded block; handy
    /// deterministic driver.
    fn abc_rounds() -> Vec<RoundTrace> {
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[63] = 24;
        sha256_compress_witness(INITIAL_HASH, block)
    }

    #[test]
    fn sha256_air_column_layout_is_consistent() {
        assert_eq!(COL_BEFORE_OFFSET, 0);
        assert_eq!(COL_AFTER_OFFSET, BITS_PER_STATE);
        assert_eq!(COL_W_OFFSET, 2 * BITS_PER_STATE);
        assert_eq!(COL_K_OFFSET, 2 * BITS_PER_STATE + BITS_PER_WORD);
        // 2 working states (before + after) + 18 aux words + T1 (3) +
        // T2/Aneg/Eneg (3 singletons) + 2 W_RECURRENCE_CARRY bits = data columns.
        //
        // 18 aux words = W, K, sigma0_a, xor01_s0, sigma1_e, xor01_s1,
        //                ef, not_e_g, ch_efg, ab, ac, bc, xor_ab_ac, maj_abc,
        //                xor01_ss0, small_sigma0_w, xor01_ss1, small_sigma1_w
        let expected_data = 2 * BITS_PER_STATE + 18 * BITS_PER_WORD
            + T1_CARRY_BITS + 1 + 1 + 1 + W_RECURRENCE_CARRY_BITS;
        assert_eq!(NUM_DATA_COLUMNS, expected_data);
        assert_eq!(NUM_DATA_COLUMNS, 1096);
        // Total = data + selectors + invocation aggregator (97 cols:
        // 64 input bytes + 32 output bytes + 1 IS_FIRST_INV_ROW) +
        // IS_FIRST_BLOCK (1 col) + multi-block output binding
        // (24 cols: 8 STATE_IN_WORD + 8 CHAIN_CARRY + 8 BINDING_CARRY)
        // + AGGREGATOR_ACTIVE (1) + IS_LAST_BLOCK (1) + W[16] recurrence
        // word aggregator (4: W0, W9, σ0(W1), σ1(W14)) — Task #179 +
        // COL_AGGREGATOR_MIRROR_BYTE0 (1: task #318 mirror).
        assert_eq!(
            NUM_SHA256_COLUMNS,
            NUM_DATA_COLUMNS
                + NUM_SEL_ROUND
                + INV_INPUT_LEN
                + INV_OUTPUT_LEN
                + 1
                + 1
                + 3 * NUM_OUTPUT_WORDS
                + 1
                + 1
                + 4
                + 1
        );
        assert_eq!(NUM_SHA256_COLUMNS, 1289);

        // Bit helpers land in the right offsets.
        assert_eq!(before(VAR_A, 0), 0);
        assert_eq!(before(VAR_H, 31), 8 * BITS_PER_WORD - 1);
        assert_eq!(after(VAR_A, 0), BITS_PER_STATE);
        assert_eq!(w_bit(0), COL_W_OFFSET);
        assert_eq!(k_bit(0), COL_K_OFFSET);
        assert_eq!(sel_round(0), NUM_DATA_COLUMNS);
        assert_eq!(sel_round(NUM_ROUNDS - 1), NUM_DATA_COLUMNS + NUM_SEL_ROUND - 1);
        // Invocation aggregator helpers.
        assert_eq!(inv_input_byte(0), COL_INV_INPUT_BYTE_OFFSET);
        assert_eq!(inv_input_byte(INV_INPUT_LEN - 1), COL_INV_OUTPUT_BYTE_OFFSET - 1);
        assert_eq!(inv_output_byte(0), COL_INV_OUTPUT_BYTE_OFFSET);
        assert_eq!(inv_output_byte(INV_OUTPUT_LEN - 1), COL_IS_FIRST_INV_ROW - 1);
        // Multi-block output-binding aggregator helpers.
        assert_eq!(state_in_word(0), COL_STATE_IN_WORD_OFFSET);
        assert_eq!(
            state_in_word(NUM_OUTPUT_WORDS - 1),
            COL_CHAIN_CARRY_BIT_OFFSET - 1
        );
        assert_eq!(chain_carry_bit(0), COL_CHAIN_CARRY_BIT_OFFSET);
        assert_eq!(binding_carry_bit(0), COL_BINDING_CARRY_BIT_OFFSET);
        assert_eq!(
            binding_carry_bit(NUM_OUTPUT_WORDS - 1),
            COL_AGGREGATOR_ACTIVE - 1
        );
        assert_eq!(COL_AGGREGATOR_ACTIVE, COL_IS_LAST_BLOCK - 1);
        // IS_LAST_BLOCK is followed by 4 W[16]-recurrence word aggregator
        // columns (Task #179), so it's no longer the very last column.
        assert_eq!(COL_W_RECURRENCE_W0_WORD, COL_IS_LAST_BLOCK + 1);
        assert_eq!(COL_W_RECURRENCE_SIGMA1_W14_WORD, NUM_SHA256_COLUMNS - 2);
        assert_eq!(COL_AGGREGATOR_MIRROR_BYTE0, NUM_SHA256_COLUMNS - 1);
    }

    #[test]
    fn sha256_air_witness_populates_fully_single_block() {
        let rounds = abc_rounds();
        assert_eq!(rounds.len(), NUM_ROUNDS);

        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        // Spot-check a few bits against the reference round trace.
        for (row, rt) in rounds.iter().enumerate() {
            // Full BEFORE/AFTER round-trip.
            for var in 0..NUM_WORKING_VARS {
                for bit in [0usize, 1, 7, 15, 23, 30, 31] {
                    let expected_before = ((rt.before[var] >> bit) & 1) as u64;
                    let got_before = columns[before(var, bit)][row].to_u64();
                    assert_eq!(got_before, expected_before,
                        "before mismatch row {} var {} bit {}", row, var, bit);
                    let expected_after = ((rt.after[var] >> bit) & 1) as u64;
                    let got_after = columns[after(var, bit)][row].to_u64();
                    assert_eq!(got_after, expected_after,
                        "after mismatch row {} var {} bit {}", row, var, bit);
                }
            }
            // W and K.
            for bit in 0..BITS_PER_WORD {
                assert_eq!(columns[w_bit(bit)][row].to_u64(),
                           ((rt.w >> bit) & 1) as u64, "W bit mismatch");
                assert_eq!(columns[k_bit(bit)][row].to_u64(),
                           ((rt.k >> bit) & 1) as u64, "K bit mismatch");
            }
            // Σ₀(a), Σ₁(e), Ch, Maj reference values.
            for bit in 0..BITS_PER_WORD {
                assert_eq!(columns[big_sigma0_a_bit(bit)][row].to_u64(),
                           ((rt.big_sigma0_a >> bit) & 1) as u64);
                assert_eq!(columns[big_sigma1_e_bit(bit)][row].to_u64(),
                           ((rt.big_sigma1_e >> bit) & 1) as u64);
                assert_eq!(columns[ch_efg_bit(bit)][row].to_u64(),
                           ((rt.ch_efg >> bit) & 1) as u64);
                assert_eq!(columns[maj_abc_bit(bit)][row].to_u64(),
                           ((rt.maj_abc >> bit) & 1) as u64);
            }
            // Exactly one selector active.
            let mut active = 0usize;
            for k in 0..NUM_ROUNDS {
                if !columns[sel_round(k)][row].is_zero() { active += 1; }
            }
            assert_eq!(active, 1, "row {} must have exactly one active selector", row);
            assert!(!columns[sel_round(rt.round)][row].is_zero());
        }
    }

    #[test]
    fn sha256_air_witness_populates_fully_multi_block() {
        // 56-byte input forces padding spill → two blocks → 128 rows.
        let input = vec![0xABu8; 56];
        let trace = sha256_witness(&input);
        assert_eq!(trace.blocks.len(), 2);
        let curve = CurveType::Bls48581;
        let columns = populate_trace_from_hash(&trace, curve);
        assert_eq!(columns.len(), NUM_SHA256_COLUMNS);
        let expected_rows = 2 * NUM_ROUNDS;
        for col in &columns {
            assert_eq!(col.len(), expected_rows);
        }
        // Every row has exactly one active selector.
        for row in 0..expected_rows {
            let mut active = 0usize;
            for k in 0..NUM_ROUNDS {
                if !columns[sel_round(k)][row].is_zero() { active += 1; }
            }
            assert_eq!(active, 1, "row {} selector count", row);
        }
    }

    #[test]
    fn sha256_air_constraints_vanish_on_valid_witness_single_block() {
        let rounds = abc_rounds();
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        for ce in &evals {
            for (row, v) in ce.values.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint `{}` fired at row {} (valid witness)",
                    ce.label, row
                );
            }
        }
        // Cross-row within the block must also vanish.
        let cross = evaluate_cross_row(&refs, &beta);
        for (row, v) in cross.values.iter().enumerate() {
            assert!(v.is_zero(), "cross-row fired at row {}", row);
        }
    }

    #[test]
    fn sha256_air_constraints_vanish_on_valid_witness_hash() {
        // Multi-block: 56-byte input → 2 blocks × 64 rounds = 128 rows.
        let input = vec![0xABu8; 56];
        let trace = sha256_witness(&input);
        assert_eq!(trace.blocks.len(), 2);

        let curve = CurveType::Bls48581;
        let columns = populate_trace_from_hash(&trace, curve);
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        for ce in &evals {
            for (row, v) in ce.values.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint `{}` fired at row {} (hash witness)",
                    ce.label, row
                );
            }
        }
        // Cross-row: within each 64-row block the state chains; the row
        // 63→64 transition is excluded by design.
        let cross = evaluate_cross_row(&refs, &beta);
        for (row, v) in cross.values.iter().enumerate() {
            assert!(v.is_zero(), "cross-row fired at row {}", row);
        }
    }

    #[test]
    fn sha256_air_constraints_reject_tampered_before_bit() {
        // Flip a pre-round `a` bit — Σ₀ / Maj / addition paths all
        // reference it, so some constraint must fire.
        let rounds = abc_rounds();
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        let col = before(VAR_A, 0);
        let one = Scalar::one(curve);
        let old = columns[col][5].clone();
        columns[col][5] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let any_fired = evals.iter().any(|ce| !ce.values[5].is_zero());
        assert!(any_fired, "tampered BEFORE bit should trigger some constraint");
    }

    #[test]
    fn sha256_air_constraints_reject_tampered_carry() {
        // Flip a T₁ carry bit — the round_additions constraint must fire.
        let rounds = abc_rounds();
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        // Toggle t1_carry bit 0 at row 3.
        let col = t1_carry_bit(0);
        let one = Scalar::one(curve);
        let old = columns[col][3].clone();
        columns[col][3] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let cat = evals.iter().find(|ce| ce.label == "round_additions").unwrap();
        assert!(
            !cat.values[3].is_zero(),
            "flipped T1 carry bit must fail round_additions"
        );
    }

    #[test]
    fn sha256_air_constraints_reject_tampered_k_binding() {
        // Flip a K bit — k_binding must fire on that row.
        let rounds = abc_rounds();
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        // Pick a row whose K differs on bit 0 from the active round's K.
        // On row 0, round 0, K[0] = 0x428a2f98, bit 0 = 0. Flip to 1.
        let col = k_bit(0);
        let one = Scalar::one(curve);
        let old = columns[col][0].clone();
        columns[col][0] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let cat = evals.iter().find(|ce| ce.label == "k_binding").unwrap();
        assert!(
            !cat.values[0].is_zero(),
            "flipped K bit must fail k_binding on row 0"
        );
    }

    #[test]
    fn sha256_air_small_sigma_w_helpers_populate_correctly() {
        // Verify that for each round, the small_sigma0_w / small_sigma1_w
        // columns hold the canonical σ0(W) / σ1(W) values matching the
        // reference SHA-256 message-schedule helpers.
        let rounds = abc_rounds();
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        for (row, rt) in rounds.iter().enumerate() {
            let w = rt.w;
            let expected_s0 =
                w.rotate_right(7) ^ w.rotate_right(18) ^ (w >> 3);
            let expected_s1 =
                w.rotate_right(17) ^ w.rotate_right(19) ^ (w >> 10);
            let mut got_s0: u32 = 0;
            let mut got_s1: u32 = 0;
            for bit in 0..BITS_PER_WORD {
                let s0_bit = columns[small_sigma0_w_bit(bit)][row].to_u64() as u32;
                let s1_bit = columns[small_sigma1_w_bit(bit)][row].to_u64() as u32;
                got_s0 |= s0_bit << bit;
                got_s1 |= s1_bit << bit;
            }
            assert_eq!(got_s0, expected_s0,
                "σ0(W) mismatch at row {}: W = 0x{:08x}", row, w);
            assert_eq!(got_s1, expected_s1,
                "σ1(W) mismatch at row {}: W = 0x{:08x}", row, w);
        }
    }

    #[test]
    fn sha256_air_w_recurrence_holds_at_anchor() {
        // Witness-level oracle test: on every block's anchor row 16, the
        // recurrence W[16] = σ1(W[14]) + W[9] + σ0(W[1]) + W[0]
        // (mod 2^32) holds, and the 2-bit W_RECURRENCE_CARRY column is
        // populated correctly.
        //
        // **Promoted algebraically (Task #179, generalized in #193)** by
        // the `w_recurrence` row-local constraint in
        // [`crate::sha256_constraints`]. See
        // `sha256_cs_w_recurrence_rejects_tampered_w16` etc. for the
        // algebraic-tampering tests. This host-side oracle stays as a
        // witness-population sanity check.
        let input = vec![0xABu8; 56]; // 2 blocks
        let trace = crate::sha256::sha256_witness(&input);
        assert_eq!(trace.blocks.len(), 2);
        let curve = CurveType::Bls48581;
        let columns = populate_trace_from_hash(&trace, curve);
        for (block_idx, block) in trace.blocks.iter().enumerate() {
            let anchor_row = block_idx * NUM_ROUNDS + 16;
            let w0 = block.rounds[0].w;
            let w1 = block.rounds[1].w;
            let w9 = block.rounds[9].w;
            let w14 = block.rounds[14].w;
            let w16 = block.rounds[16].w;
            let s0 = w1.rotate_right(7) ^ w1.rotate_right(18) ^ (w1 >> 3);
            let s1 = w14.rotate_right(17) ^ w14.rotate_right(19) ^ (w14 >> 10);
            let full_sum: u64 = (s1 as u64) + (w9 as u64) + (s0 as u64) + (w0 as u64);
            assert_eq!(
                (full_sum & 0xFFFF_FFFF) as u32,
                w16,
                "block {} W[16] recurrence violated", block_idx
            );
            let expected_carry = full_sum >> 32;
            // Reconstruct the 2-bit carry from the witness column.
            let mut got_carry: u64 = 0;
            for bit in 0..W_RECURRENCE_CARRY_BITS {
                let v = columns[COL_W_RECURRENCE_CARRY_OFFSET + bit][anchor_row].to_u64();
                got_carry |= v << bit;
            }
            assert_eq!(got_carry, expected_carry,
                "W_RECURRENCE_CARRY mismatch at block {} (row {})", block_idx, anchor_row);
        }
    }

    #[test]
    fn sha256_air_w_recurrence_holds_at_all_recurrence_rows() {
        // Task #193: extended witness-level oracle test that the full
        // message-schedule recurrence
        //   W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16] (mod 2^32)
        // holds for every t ∈ 16..64 of every block, with the aggregator
        // word columns + 2-bit W_RECURRENCE_CARRY populated correctly at
        // each recurrence row.
        let input = vec![0xCDu8; 56]; // 2 blocks
        let trace = crate::sha256::sha256_witness(&input);
        assert_eq!(trace.blocks.len(), 2);
        let curve = CurveType::Bls48581;
        let columns = populate_trace_from_hash(&trace, curve);
        for (block_idx, block) in trace.blocks.iter().enumerate() {
            for t in 16..NUM_ROUNDS {
                let row = block_idx * NUM_ROUNDS + t;
                let w_tm16 = block.rounds[t - 16].w;
                let w_tm15 = block.rounds[t - 15].w;
                let w_tm7 = block.rounds[t - 7].w;
                let w_tm2 = block.rounds[t - 2].w;
                let w_t = block.rounds[t].w;
                let s0 = w_tm15.rotate_right(7)
                    ^ w_tm15.rotate_right(18)
                    ^ (w_tm15 >> 3);
                let s1 = w_tm2.rotate_right(17)
                    ^ w_tm2.rotate_right(19)
                    ^ (w_tm2 >> 10);
                let full_sum: u64 =
                    (s1 as u64) + (w_tm7 as u64) + (s0 as u64) + (w_tm16 as u64);
                assert_eq!(
                    (full_sum & 0xFFFF_FFFF) as u32,
                    w_t,
                    "block {} W[{}] recurrence violated", block_idx, t
                );
                let expected_carry = full_sum >> 32;
                let mut got_carry: u64 = 0;
                for bit in 0..W_RECURRENCE_CARRY_BITS {
                    let v = columns[COL_W_RECURRENCE_CARRY_OFFSET + bit][row].to_u64();
                    got_carry |= v << bit;
                }
                assert_eq!(got_carry, expected_carry,
                    "W_RECURRENCE_CARRY mismatch at block {} row {} (t={})",
                    block_idx, row, t);
                // Aggregator word columns also match.
                assert_eq!(
                    columns[COL_W_RECURRENCE_W0_WORD][row].to_u64(),
                    w_tm16 as u64,
                    "W[t-16] aggregator mismatch at block {} t={}", block_idx, t,
                );
                assert_eq!(
                    columns[COL_W_RECURRENCE_W9_WORD][row].to_u64(),
                    w_tm7 as u64,
                    "W[t-7] aggregator mismatch at block {} t={}", block_idx, t,
                );
                assert_eq!(
                    columns[COL_W_RECURRENCE_SIGMA0_W1_WORD][row].to_u64(),
                    s0 as u64,
                    "σ0(W[t-15]) aggregator mismatch at block {} t={}", block_idx, t,
                );
                assert_eq!(
                    columns[COL_W_RECURRENCE_SIGMA1_W14_WORD][row].to_u64(),
                    s1 as u64,
                    "σ1(W[t-2]) aggregator mismatch at block {} t={}", block_idx, t,
                );
            }
        }
    }

    #[test]
    fn sha256_air_constraints_reject_tampered_small_sigma0_w_bit() {
        // Flip a bit of `small_sigma0_w` — the small_sigma0_w_definition
        // constraint must fire at that row.
        let rounds = abc_rounds();
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }
        // Toggle small_sigma0_w bit 0 at row 4.
        let col = small_sigma0_w_bit(0);
        let one = Scalar::one(curve);
        let old = columns[col][4].clone();
        columns[col][4] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let cat = evals.iter()
            .find(|ce| ce.label == "small_sigma0_w_definition")
            .expect("small_sigma0_w_definition category exists");
        assert!(
            !cat.values[4].is_zero(),
            "flipped small_sigma0_w bit must fail definition constraint"
        );
    }

    #[test]
    fn sha256_air_constraints_reject_tampered_small_sigma1_w_bit() {
        // Flip a bit of `small_sigma1_w` — the small_sigma1_w_definition
        // constraint must fire at that row.
        let rounds = abc_rounds();
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }
        let col = small_sigma1_w_bit(0);
        let one = Scalar::one(curve);
        let old = columns[col][7].clone();
        columns[col][7] = one.sub(&old);

        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let cat = evals.iter()
            .find(|ce| ce.label == "small_sigma1_w_definition")
            .expect("small_sigma1_w_definition category exists");
        assert!(
            !cat.values[7].is_zero(),
            "flipped small_sigma1_w bit must fail definition constraint"
        );
    }

    #[test]
    fn sha256_air_w_recurrence_carry_zero_off_recurrence_rows() {
        // Task #193: the W_RECURRENCE_CARRY column should be zero on every
        // row except the per-block recurrence rows (t ∈ 16..64 within each
        // block).
        let input = vec![0xABu8; 56]; // 2 blocks
        let trace = crate::sha256::sha256_witness(&input);
        let curve = CurveType::Bls48581;
        let columns = populate_trace_from_hash(&trace, curve);
        let num_rows = columns[0].len();
        for row in 0..num_rows {
            let t = row % NUM_ROUNDS;
            let is_recurrence = (16..NUM_ROUNDS).contains(&t);
            for bit in 0..W_RECURRENCE_CARRY_BITS {
                let v = &columns[COL_W_RECURRENCE_CARRY_OFFSET + bit][row];
                if !is_recurrence {
                    assert!(
                        v.is_zero(),
                        "W_RECURRENCE_CARRY bit {} non-zero on non-recurrence row {}",
                        bit, row
                    );
                }
            }
        }
    }

    #[test]
    fn sha256_air_round_constants_wire_through() {
        // Populate a valid witness and verify that each row's K column
        // decodes to ROUND_CONSTANTS[row].
        let rounds = abc_rounds();
        let curve = CurveType::Bls48581;
        let mut columns = alloc_trace(NUM_ROUNDS, curve);
        for (row, rt) in rounds.iter().enumerate() {
            populate_round(&mut columns, row, rt, curve);
        }

        for row in 0..NUM_ROUNDS {
            let mut k_val: u32 = 0;
            for bit in 0..BITS_PER_WORD {
                let v = columns[k_bit(bit)][row].to_u64() as u32;
                k_val |= v << bit;
            }
            assert_eq!(k_val, ROUND_CONSTANTS[row],
                "K column mismatch at row {}", row);
        }

        // And the k_binding constraint passes on the valid witness.
        let beta = beta_challenge();
        let refs = col_refs(&columns);
        let evals = evaluate_constraints(&refs, &beta);
        let cat = evals.iter().find(|ce| ce.label == "k_binding").unwrap();
        for (row, v) in cat.values.iter().enumerate() {
            assert!(v.is_zero(), "k_binding fired at row {}", row);
        }
    }
}
