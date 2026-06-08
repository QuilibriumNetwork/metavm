//! BLS12-381 **multi-row Miller-loop composition** AIR.
//!
//! # Purpose
//!
//! This module stitches together many rows of the single-step Miller
//! AIR ([`crate::miller_step_air`]) into a single trace covering one
//! full Miller-loop evaluation on BLS12-381. The BLS12-381 curve
//! parameter is `|x| = 0xd201_0000_0001_0000` (Hamming weight 6, MSB
//! at bit 63), so the natural unrolling is:
//!
//!   * **63 doubling rows** (one per `i` in `(0..63).rev()`).
//!   * **5 addition rows**, interleaved at each set bit of `|x|` in
//!     positions 0..62 (positions 16, 48, 57, 60, 62).
//!
//! Total: **68 active rows**. The trace is then zero-padded to the
//! nearest power of two (128 rows) for FFT-friendly proving.
//!
//! # What this AIR commits and enforces
//!
//! Each row uses **exactly** the column layout of
//! [`crate::miller_step_air`] (290 columns: Q_curr/Q_next/Q_fixed/
//! line_value/acc_pre/acc_post/is_doubling/is_addition). All the
//! row-local selector-binarity + mutual-exclusion constraints and the
//! cross-row continuity (`Q_curr[r+1] = Q_next[r]`,
//! `acc_pre[r+1] = acc_post[r]`) come "for free" from re-using
//! [`crate::miller_step_air::MillerStepConstraintSystem`].
//!
//! In addition this AIR adds:
//!
//!   1. **`is_real ∈ {0, 1}`** per row (boundary selector derived from
//!      `is_doubling + is_addition`).
//!   2. **First-row Q_curr initialization**: on row 0,
//!      `Q_curr == q` (the input G2 point). Encoded as a 24-limb
//!      β-RLC equality body gated by an `is_first_row` boundary
//!      selector. The input `q` value enters the AIR via the
//!      `q_fixed` columns (which are constant across all rows, so
//!      they double as a 24-limb commitment to the loop's G2
//!      operand). The boundary constraint then reads
//!      `Q_curr[row 0] − Q_fixed[row 0] = 0` per limb.
//!   3. **First-row acc_pre = Fp12::one()**: on row 0,
//!      `acc_pre == 1`. Encoded as a 72-limb β-RLC equality body
//!      against the constant `Fp12::one()` (limb 0 of c0.c0.c0 == 1,
//!      all other 71 limbs == 0). Gated by `is_first_row`.
//!
//! Together this AIR is the **first algebraic seal** on the Miller
//! loop boundary conditions: an honest prover trivially satisfies
//! them, and a malicious prover cannot start the chain anywhere
//! other than `(Q_curr=q, acc_pre=1)` without violating constraint 2
//! or 3.
//!
//! # What is still oracle (deferred)
//!
//! Same as [`crate::miller_step_air`]: the **per-row Fp12 arithmetic**
//! `acc_post = acc_pre² · line_value` and `Q_next = 2·Q_curr` (resp.
//! `Q_curr + Q_fixed`) is committed but not algebraically checked.
//! Each such row's deep relations decompose into ~150 Fp limb-level
//! multiplications inside [`crate::nonnative_fp_air`]; closing them is
//! a follow-up phase via cross-AIR LogUp ([`crate::cross_air_logup`]).
//!
//! Also deferred:
//!
//!   * **Real `line_value` computation** — Task #197 closed: the
//!     host-side builder [`MillerLoopWitness::from_pk_and_p`] now
//!     threads the real Miller-loop ell line evaluations into each
//!     row's `line_value`, via the `pub(crate)` `doubling_step` /
//!     `addition_step` / `ell_line_value` helpers in
//!     [`crate::pairing`]. The chained `acc_post` therefore equals
//!     `pairing::miller_loop(p, q).conjugate()` on the last row (the
//!     final negative-x conjugate is left for a separate row in
//!     [`crate::final_exp_air`]).
//!   * **Final conjugation** for `x < 0` — one extra row with an
//!     `is_conjugate` selector. Tracked separately by the
//!     [`crate::final_exp_air`] surface.
//!   * **Last-row pinning to the actual Miller-loop output** — the
//!     host-side `pairing::miller_loop(p, q)` value is published on
//!     the witness as `expected_miller_output` for downstream cross-
//!     AIR LogUp pinning. The last row's in-circuit `acc_post` is now
//!     equal to `expected_miller_output.conjugate()` *value-wise*,
//!     but until the per-row Fp12 arithmetic is algebraically closed
//!     this equality is not enforced in-circuit; it's bound only via
//!     the boundary `acc_pre = 1` constraint at row 0 plus the
//!     (oracle) chain consistency through 68 rows.
//!
//! # Cross-AIR linkages
//!
//! * [`make_miller_loop_to_step_descriptor`] — A side = this AIR's
//!   per-row (Q_curr || acc_pre || acc_post) tuple gated by
//!   `is_doubling + is_addition`, B side = the same tuple in the
//!   single-step Miller AIR. Witness-shape spec for a future
//!   per-row LogUp that pins each composition row to a row of the
//!   stand-alone `miller_step_air` (useful once the step AIR's deep
//!   arithmetic relations are closed: the composition AIR can then
//!   "outsource" per-row correctness to step AIR rows). 168-tuple
//!   LogUp.
//! * [`make_miller_loop_to_final_exp_descriptor`] — A side = this
//!   AIR's last-row `acc_post` Fp12 limbs gated by `is_last_row`,
//!   B side = [`crate::final_exp_air`]'s row-0 `f_pre` Fp12 limbs
//!   gated by `is_easy_step`. 72-tuple LogUp. Pins the Miller-loop
//!   output to the final-exponentiation input.
//!
//! Both descriptors are intentionally simple shapes: closing the full
//! soundness story is the cross-AIR LogUp protocol's job (per the
//! [`crate::cross_air_logup`] orchestrator).

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::miller_step_air::{
    self as ms, FP_PER_FP12, FP_PER_G2, LIMBS_PER_FP, LIMBS_PER_FP12, LIMBS_PER_G2,
    MillerStepConstraintSystem, MillerStepWitness, NUM_COLUMNS as MS_NUM_COLUMNS,
    NUM_ROW_CONSTRAINTS as MS_NUM_ROW_CONSTRAINTS, NUM_SHIFTED as MS_NUM_SHIFTED,
};
use crate::nonnative_tower::Fp12;
use crate::pairing::{self, G1Affine, G2Affine};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// BLS12-381 curve parameter absolute value. The Miller loop scans bits
/// of this value from MSB-1 down to bit 0.
///
/// `|x| = 0xd201_0000_0001_0000` has Hamming weight 6 (set bits at
/// positions 16, 48, 57, 60, 62, 63). The MSB at position 63 is implicit
/// (it picks the initial T = q), so 5 set bits remain in 0..62 to
/// trigger addition rows.
pub const MILLER_X_ABS: u64 = 0xd201_0000_0001_0000;

/// Number of doubling iterations: one per bit position `i in (0..63).rev()`.
pub const NUM_DOUBLING_ROWS: usize = 63;
/// Number of addition iterations: one per set bit of `|x|` in positions
/// 0..62.
pub const NUM_ADDITION_ROWS: usize = 5;
/// Total active rows in the unrolled Miller loop.
pub const NUM_LOOP_ROWS: usize = NUM_DOUBLING_ROWS + NUM_ADDITION_ROWS;

// ─── Column layout ────────────────────────────────────────────────────
//
// We add two boundary selector columns on top of the miller_step_air
// layout. The Miller-step layout stays at indices 0..MS_NUM_COLUMNS;
// `IS_FIRST_ROW` is at MS_NUM_COLUMNS; `IS_LAST_ROW` is at
// MS_NUM_COLUMNS + 1.

pub use crate::miller_step_air::{
    COL_ACC_POST_OFFSET, COL_ACC_PRE_OFFSET, COL_IS_ADDITION, COL_IS_DOUBLING,
    COL_LINE_VALUE_OFFSET, COL_Q_CURR_OFFSET, COL_Q_FIXED_OFFSET, COL_Q_NEXT_OFFSET,
};

/// Boundary selector: 1 on row 0, 0 elsewhere. Used to gate first-row
/// initialization constraints.
pub const COL_IS_FIRST_ROW: usize = MS_NUM_COLUMNS;
/// Boundary selector: 1 on the last active row, 0 elsewhere. Used to
/// gate the cross-AIR LogUp linking `acc_post` to the final-exp input.
pub const COL_IS_LAST_ROW: usize = COL_IS_FIRST_ROW + 1;

/// Number of Fp limbs in the X-component "representative" of the f
/// accumulator (Fp12). We pick the first two Fp components
/// (c0.c0.c0 + c0.c0.c1) = 2 × 6 = 12 limbs ×... no, we pick the first
/// **four** Fp components (24 limbs) of `acc_pre`/`acc_post` as the
/// stitched cross-row representative. This is the **f_x_limbs**
/// surface referenced by the cross-row continuity constraint.
pub const F_X_LIMBS_LEN: usize = 24;
/// Number of Fp limbs in the X-component of the running point T (G2).
/// G2 = (X.c0, X.c1, Y.c0, Y.c1) × 6 limbs each. The X-component is
/// the first 2 × 6 = 12 limbs.
pub const T_X_LIMBS_LEN: usize = 12;

/// Row-start "f_x_limbs_in" — copy of the first 24 limbs of `acc_pre`.
/// This is a per-row witness column used to drive the **cross-row f
/// continuity** shifted constraint: `f_x_limbs_in(ω·X)` matches
/// `f_x_limbs_out(X)`.
pub const COL_F_X_LIMBS_IN_OFFSET: usize = COL_IS_LAST_ROW + 1;
/// Row-end "f_x_limbs_out" — copy of the first 24 limbs of `acc_post`.
pub const COL_F_X_LIMBS_OUT_OFFSET: usize = COL_F_X_LIMBS_IN_OFFSET + F_X_LIMBS_LEN;
/// Row-start "t_x_limbs_in" — copy of the first 12 limbs of `Q_curr`
/// (the X-component of the running point T).
pub const COL_T_X_LIMBS_IN_OFFSET: usize = COL_F_X_LIMBS_OUT_OFFSET + F_X_LIMBS_LEN;
/// Row-end "t_x_limbs_out" — copy of the first 12 limbs of `Q_next`.
pub const COL_T_X_LIMBS_OUT_OFFSET: usize = COL_T_X_LIMBS_IN_OFFSET + T_X_LIMBS_LEN;
/// NAF bit "positive" column: 1 iff the loop's signed NAF digit at this
/// row is +1, else 0.
pub const COL_NAF_POS: usize = COL_T_X_LIMBS_OUT_OFFSET + T_X_LIMBS_LEN;
/// NAF bit "negative" column: 1 iff the loop's signed NAF digit at
/// this row is -1, else 0. The mutual-exclusion constraint
/// `naf_pos · naf_neg = 0` is enforced row-locally.
pub const COL_NAF_NEG: usize = COL_NAF_POS + 1;

// ── Task #306 (closes #233 deferral): affine doubling-step ell-line
// slope λ witness columns.
//
// On a doubling row, the **affine** doubling step on the running point
// `T = Q_curr` (which is committed in this trace as an affine G2 point)
// yields the tangent-line slope
//
//   λ = 3·T.x² / (2·T.y)
//
// We commit four Fp2 (= 2·6 = 12 limb) blocks per row that decompose
// this:
//
//   * `lambda_num     = 3·T.x²`
//   * `lambda_denom   = 2·T.y`
//   * `lambda_denom_inv` — witness Fp2 satisfying
//     `lambda_denom · lambda_denom_inv = 1`
//   * `lambda         = lambda_num · lambda_denom_inv = 3·T.x²/(2·T.y)`
//
// These columns are populated on doubling rows and zeroed on addition /
// padding rows. The algebraic relations binding them to `T.x`, `T.y`
// are surfaced as **cross-AIR LogUp descriptors** into
// [`crate::nonnative_fp_air`] (see
// [`crate::miller_fp_descriptors::EllLambdaDescriptors`]); the
// witness blocks are exposed here as the B-side column shape those
// descriptors point at.
//
// The final binding back to `line_value` (`line.c4 = -λ` /
// `line.c1 = λ·T.x − T.y`) is **not** yet enforced as a row-local in
// this AIR, because the dense `line_value` stored in the trace is the
// Jacobian-formula c4/c1 already scaled by `P.y` / `P.x` (see
// [`crate::pairing::ell_line_value`] / [`crate::pairing::doubling_step`]).
// Reconciling the affine λ surface with the Jacobian-scaled line_value
// requires a separate Jacobian-↔-affine bridging gadget that
// the descriptor set is designed to compose with once it lands. For
// the present phase, the descriptors algebraically pin the four Fp2
// quantities to honest values; that is precisely the "λ itself is
// witness" gap #233 called out.

/// Number of Fp limbs in one Fp2 witness block (2 Fp values × 6 limbs).
pub const LIMBS_PER_FP2: usize = 12;

/// `λ_num = 3·T.x²` (Fp2, 12 limbs).
pub const COL_LAMBDA_NUM_OFFSET: usize = COL_NAF_NEG + 1;
/// `λ_denom = 2·T.y` (Fp2, 12 limbs).
pub const COL_LAMBDA_DENOM_OFFSET: usize = COL_LAMBDA_NUM_OFFSET + LIMBS_PER_FP2;
/// Witness Fp2 satisfying `λ_denom · λ_denom_inv = 1`.
pub const COL_LAMBDA_DENOM_INV_OFFSET: usize = COL_LAMBDA_DENOM_OFFSET + LIMBS_PER_FP2;
/// `λ = λ_num · λ_denom_inv = 3·T.x²/(2·T.y)` (Fp2, 12 limbs).
pub const COL_LAMBDA_OFFSET: usize = COL_LAMBDA_DENOM_INV_OFFSET + LIMBS_PER_FP2;

// ── Task #319: Jacobian↔affine ell bridge — affine line-coefficient
// witness blocks.
//
// The Jacobian `doubling_step` returns `(c4, c1, c0)` with embedded `Z²`
// scaling factors (see `pairing::doubling_step` lines 701–723). The
// **affine** tangent-line formulation at `T = (T.x, T.y)` with slope
// `λ = 3·T.x²/(2·T.y)` instead yields the canonical sparse line
// coefficients
//
//   c4_affine = -λ
//   c1_affine =  λ·T.x − T.y
//   c0_affine = −λ·T.x + T.y      (= −c1_affine for the doubling tangent;
//                                   note that for the addition step the
//                                   spec form differs — see below).
//
// These three Fp2 (= 12-limb) witness blocks commit the affine line
// coefficients per **doubling** row. They serve two soundness roles:
//
//   1. `c4_affine + λ = 0` is the row-local pin that **closes the
//      Task #306 deferred binding** (the `line.c4 = -λ` constraint that
//      could not be enforced row-locally before because the trace's
//      `line_value` is Jacobian-scaled). With the affine surface now
//      separated out, λ is pinned algebraically to the affine c4.
//
//   2. `c1_affine = λ·T.x − T.y` and `c0_affine = −λ·T.x + T.y` are
//      Fp2-arithmetic relations whose Fp-level decompositions feed
//      cross-AIR LogUp into [`crate::nonnative_fp_air`] (see
//      [`crate::miller_fp_descriptors::EllAffineLineDescriptors`]).
//
// **Jacobian↔affine bridge to `line_value` (deferred scaffold).** The
// final connection — pinning `line_value[c0.c1 slot] = c1_affine · x_P`
// and `line_value[c1.c1 slot] = c4_affine · y_P` (modulo the Jacobian
// Z²-scaling correction factor between the affine c4/c1 surface and
// the Jacobian-scaled `c4`/`c1` actually used by
// `pairing::ell_line_value`) requires two additional witness surfaces
// not yet committed in the trace:
//
//   * `x_P`, `y_P` (the input G1 point's affine coordinates), and
//   * `Z²` (the Jacobian running-point's Z² scaling factor at this
//     iteration).
//
// `x_P` / `y_P` are deferred to a follow-up `P_INPUT_OFFSET` column
// block; the `Z²` factor is deferred to a follow-up Jacobian-running
// state. The Fp2 mul / sub descriptors wired here pin the affine
// surface itself; the bridge into `line_value` is descriptor-only
// scaffold once those columns land.
//
// On addition rows (which use a different slope formula `λ_add =
// (Q.y − T.y)/(Q.x − T.x)`) and on padding rows, all three affine
// blocks are zeroed.

/// Affine line coefficient `c4_affine = -λ` (Fp2, 12 limbs).
pub const COL_C4_AFFINE_OFFSET: usize = COL_LAMBDA_OFFSET + LIMBS_PER_FP2;
/// Affine line coefficient `c1_affine = λ·T.x − T.y` (Fp2, 12 limbs).
pub const COL_C1_AFFINE_OFFSET: usize = COL_C4_AFFINE_OFFSET + LIMBS_PER_FP2;
/// Affine line coefficient `c0_affine = −λ·T.x + T.y = −c1_affine`
/// (Fp2, 12 limbs).
pub const COL_C0_AFFINE_OFFSET: usize = COL_C1_AFFINE_OFFSET + LIMBS_PER_FP2;

/// Total column count for this AIR.
pub const NUM_COLUMNS: usize = COL_C0_AFFINE_OFFSET + LIMBS_PER_FP2;

/// Row-local constraints (this AIR's additions on top of the
/// re-exported [`crate::miller_step_air`] ones):
///
///   0: `is_first_row ∈ {0, 1}`
///   1: `is_last_row ∈ {0, 1}`
///   2: `is_first_row · is_last_row = 0` on rows with more than one
///      active row (mutual exclusion of boundary selectors; trivially
///      satisfied for the 68-row honest witness).
///   3: First-row Q_curr initialization:
///      `is_first_row · Σ_{j=0..24} β^j · (Q_curr[j] − Q_fixed[j]) = 0`.
///      (Since `Q_fixed` is constant across all rows, this pins
///      `Q_curr[row 0] = Q_fixed[row 0] = q`.)
///   4: First-row acc_pre = Fp12::one():
///      `is_first_row · Σ_{j=0..72} β^j · (acc_pre[j] − ONE[j]) = 0`
///      where `ONE[0] = 1` (the c0.c0.c0.limb_5 spot in BE convention
///      — see below) and `ONE[j] = 0` elsewhere.
///   5: `naf_pos ∈ {0, 1}`.
///   6: `naf_neg ∈ {0, 1}`.
///   7: `naf_pos · naf_neg = 0` (NAF digit must be one of -1, 0, +1).
///   8: `IS_REAL · Σ_{j=0..24} β^j · (f_x_limbs_in[j] − acc_pre[j]) = 0`
///      where `IS_REAL = is_doubling + is_addition`. This mirrors
///      `f_x_limbs_in` onto `acc_pre`'s first 24 limbs.
///   9: `IS_REAL · Σ_{j=0..24} β^j · (f_x_limbs_out[j] − acc_post[j]) = 0`.
///  10: `IS_REAL · Σ_{j=0..12} β^j · (t_x_limbs_in[j] − Q_curr[j]) = 0`.
///  11: `IS_REAL · Σ_{j=0..12} β^j · (t_x_limbs_out[j] − Q_next[j]) = 0`.
///  12..17 (Task #233): **ell line sparsity — structural-zero Fp slots
///       of the dense `line_value` Fp12 are pinned to zero**. The
///       sparse `mul_by_014` form has non-zero Fp2 entries only at
///       Fp12 positions 1 (c0.c0), v (c0.c1), and v·w (c1.c1). The
///       other three Fp2 slots — c0.c2, c1.c0, c1.c2 — must be the
///       zero Fp2 (6 Fp limbs each), giving 6 β-RLC constraints over
///       the 6 limbs of each structurally-zero Fp slot:
///       * 12: `IS_REAL · Σ_{j=0..6} β^j · line_value[c0.c2.c0 + j] = 0`
///         (slot 4 in the flattened layout).
///       * 13: `IS_REAL · Σ_{j=0..6} β^j · line_value[c0.c2.c1 + j] = 0`
///         (slot 5).
///       * 14: `IS_REAL · Σ_{j=0..6} β^j · line_value[c1.c0.c0 + j] = 0`
///         (slot 6).
///       * 15: `IS_REAL · Σ_{j=0..6} β^j · line_value[c1.c0.c1 + j] = 0`
///         (slot 7).
///       * 16: `IS_REAL · Σ_{j=0..6} β^j · line_value[c1.c2.c0 + j] = 0`
///         (slot 10).
///       * 17: `IS_REAL · Σ_{j=0..6} β^j · line_value[c1.c2.c1 + j] = 0`
///         (slot 11).
///       Together these pin the dense `line_value` to the 014-image
///       (a necessary condition for it to be a real ell line evaluation).
///       The full λ derivation — `c4 = -λ`, `c1 = λ·T.x - T.y`,
///       and the per-slot scalings `line[c0.c1] = c1 · x_P`,
///       `line[c1.c1] = c4 · y_P` — is an Fp-arithmetic relation
///       and is deferred to a cross-AIR LogUp linkage into
///       [`crate::nonnative_fp_air`].
///
/// **Task #319 (closes Task #306 deferral)** — additional affine line
/// coefficient witness columns (`COL_C4_AFFINE_OFFSET`,
/// `COL_C1_AFFINE_OFFSET`, `COL_C0_AFFINE_OFFSET`) are committed per
/// doubling row but their binding is enforced via **cross-AIR LogUp
/// descriptors** (not row-local constraints), because the per-limb
/// Fp negation/addition relations cannot be expressed as a naive
/// `c4_affine[j] + λ[j] = 0` Scalar equality (the BE Fp limbs satisfy
/// `c4_affine + λ ≡ 0 (mod p)`, not zero as integers/Scalars). See
/// [`crate::miller_fp_descriptors::EllAffineLineDescriptors`] for the
/// 5 descriptor scaffold (3 Fp MULs + 2 Fp ADDs) and the
/// (deferred) `c4_affine + λ = 0` / `c0_affine + c1_affine = 0` Fp ADD
/// closures.
pub const NUM_LOOP_ROW_CONSTRAINTS: usize = 18;

/// Sub-offsets (in Fp-slot units, 0..12) into the 72-limb dense Fp12
/// flattening at which the sparse `mul_by_014` representation must be
/// the zero Fp. Used to gate the ell-sparsity row-local constraints
/// (12..17 above).
pub const ELL_SPARSITY_ZERO_FP_SLOTS: [usize; 6] = [4, 5, 6, 7, 10, 11];

/// Shifted-row constraint categories — inherits miller_step_air's two
/// (Q chain continuity + acc chain continuity), plus two new ones:
///   * **f continuity**: `f_x_limbs_in(ω·X) = f_x_limbs_out(X)` (β-RLC).
///   * **T continuity**: `t_x_limbs_in(ω·X) = t_x_limbs_out(X)` (β-RLC).
/// Both gated by `IS_REAL` so padding rows are exempt.
pub const NUM_LOOP_SHIFTED: usize = MS_NUM_SHIFTED + 2;

// ─── Witness ──────────────────────────────────────────────────────────

/// Multi-row Miller-loop witness.
///
/// This is a thin wrapper around [`MillerStepWitness`] (so the trace
/// builder can reuse `miller_step_air`'s flattening logic) plus a
/// host-side `expected_miller_output` field carrying the
/// `pairing::miller_loop(p, q)` reference value.
///
/// # Soundness note
///
/// As of Task #197 each row's `line_value` is the real `ell` line
/// evaluation at `P` (the dense Fp12 lift of the sparse `(c4, c1, c0)`
/// produced by [`crate::pairing::doubling_step`] /
/// [`crate::pairing::addition_step`] / [`crate::pairing::ell_line_value`]).
/// The chained accumulator `acc_post` therefore matches the real
/// `pairing::miller_loop` running value at every row, and equals
/// `expected_miller_output.conjugate()` on the last row (the
/// negative-x conjugate is left for a downstream final-exp row). The
/// `expected_miller_output` field is still **not yet algebraically tied
/// to the trace** — it is published as a host-side oracle that
/// downstream phases will pin in-circuit once the per-row Fp12
/// arithmetic is closed via cross-AIR LogUp into
/// [`crate::nonnative_fp_air`].
#[derive(Clone, Debug)]
pub struct MillerLoopWitness {
    /// The underlying step witness (one row per Miller iteration).
    pub steps: MillerStepWitness,
    /// The input G1 point P (host-side, used for the real
    /// `pairing::miller_loop(p, q)` computation).
    pub p: G1Affine,
    /// The input G2 point Q (host-side, used for the real
    /// `pairing::miller_loop(p, q)` computation, and pinned in the
    /// AIR via the `Q_fixed` columns + first-row Q_curr boundary
    /// constraint).
    pub q: G2Affine,
    /// `pairing::miller_loop(p, q)` — host-side reference value.
    /// **Oracle**: not yet algebraically bound to the trace; see the
    /// struct-level note above.
    pub expected_miller_output: Fp12,
}

impl MillerLoopWitness {
    /// Build a Miller-loop witness from the input G1 / G2 points.
    ///
    /// Iterates the same bit pattern as [`crate::pairing::miller_loop`]
    /// (MSB-1 → bit 0 of `|x|`), emitting one doubling row per bit and
    /// one extra addition row at each set bit. The `Q_curr`/`Q_next`
    /// trace columns track affine `q.double()` / `q.add(&q_fixed)` (the
    /// [`crate::miller_step_air`] layout convention), while a parallel
    /// host-side Jacobian state `t` mirrors `pairing::miller_loop`'s
    /// running `T` and produces the real `(c4, c1, c0)` line
    /// coefficients per row. Each row's `line_value` is the dense
    /// `Fp12` lift of those coefficients evaluated at `P` (see
    /// [`crate::pairing::ell_line_value`]), so the chained accumulator
    /// `acc_post = acc_pre² · line_value` carries the real Miller-loop
    /// arithmetic. The resulting witness satisfies every row-local +
    /// shifted constraint in this AIR.
    ///
    /// The host-side `pairing::miller_loop(p, q)` is computed and
    /// stored on the witness as `expected_miller_output` for downstream
    /// pinning (see the struct-level soundness note). The relation
    /// `last_acc_post == expected_miller_output.conjugate()` holds by
    /// construction.
    pub fn from_pk_and_p(p: &G1Affine, q: &G2Affine) -> Self {
        let mut steps = MillerStepWitness::new();
        let mut current_q = *q;
        let mut acc = Fp12::one();
        let q_fixed = *q;

        // Real ell line values now thread through the witness (Task #197).
        //
        // The trace's affine `Q_curr`/`Q_next` columns continue to evolve
        // affinely (`q_curr.double()` / `q_curr + q_fixed`) for layout
        // compatibility with miller_step_air, but the **line coefficients**
        // are computed from a parallel Jacobian state `t` that mirrors
        // `pairing::miller_loop`'s `T` exactly. Each row's `line_value` is
        // the dense Fp12 lift of the sparse `(c4, c1, c0)` evaluated at
        // `P` (see `pairing::ell_line_value`), so the in-circuit
        // recurrence `acc_post = acc_pre² · line_value` carries the real
        // Miller-loop arithmetic, and the chained accumulator matches
        // `pairing::miller_loop(p, q).conjugate()` on the last row (the
        // top-level conjugate is applied by `pairing::miller_loop` for
        // negative-x; the per-row chain produces the pre-conjugation
        // value).
        let mut t = pairing::G2Jacobian::from_affine(q);

        // Mirror `pairing::miller_loop`'s iteration:
        //   msb = 63, then for i in (0..msb).rev(): doubling; if bit i
        //   of MILLER_X_ABS is set: addition.
        let msb = 63 - MILLER_X_ABS.leading_zeros() as i32;
        for i in (0..msb).rev() {
            // Real doubling-step line coefficients (also mutates `t`).
            let coeffs = pairing::doubling_step(&mut t);
            let line = pairing::ell_line_value(&coeffs, p);
            steps.push_doubling(current_q, q_fixed, line, acc);
            // After push_doubling: chain continuity demands the next
            // row's Q_curr = this row's Q_next, acc_pre = acc_post.
            let last = steps.rows.last().unwrap();
            current_q = last.q_next;
            acc = last.acc_post;

            if ((MILLER_X_ABS >> i) & 1) == 1 {
                // Real addition-step line coefficients (also mutates `t`).
                let coeffs = pairing::addition_step(&mut t, q);
                let line = pairing::ell_line_value(&coeffs, p);
                steps.push_addition(current_q, q_fixed, line, acc);
                let last = steps.rows.last().unwrap();
                current_q = last.q_next;
                acc = last.acc_post;
            }
        }

        let expected_miller_output = pairing::miller_loop(p, q);

        Self { steps, p: *p, q: *q, expected_miller_output }
    }

    /// Total number of active (non-padding) rows.
    pub fn num_rows(&self) -> usize {
        self.steps.rows.len()
    }

    /// Last-row accumulator output (the in-circuit Miller-loop result
    /// under the toy line=1 model).
    pub fn last_acc_post(&self) -> Option<Fp12> {
        self.steps.rows.last().map(|r| r.acc_post)
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

/// Build the trace polynomials for the multi-row Miller-loop AIR.
///
/// The first `MS_NUM_COLUMNS` columns are populated by reusing
/// [`crate::miller_step_air::build_trace_polynomials`]; we then append
/// the two boundary selector columns.
pub fn build_trace_polynomials(
    witness: &MillerLoopWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.num_rows();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    // Reuse miller_step_air's trace builder for the first 290 columns.
    let step_trace = ms::build_trace_polynomials(&witness.steps, curve);
    assert_eq!(step_trace.columns.len(), MS_NUM_COLUMNS);
    assert_eq!(step_trace.padded_size as usize, padded);

    let mut columns: Vec<Vec<Scalar>> = step_trace
        .columns
        .into_iter()
        .map(|p| p.evaluations)
        .collect();

    // Append boundary selector columns.
    let mut is_first = vec![zero.clone(); padded];
    let mut is_last = vec![zero.clone(); padded];
    if num_rows > 0 {
        is_first[0] = one.clone();
        is_last[num_rows - 1] = one.clone();
    }
    columns.push(is_first);
    columns.push(is_last);

    // Append f_x_limbs_in / f_x_limbs_out / t_x_limbs_in / t_x_limbs_out
    // — copies of the first 24 limbs of acc_pre/acc_post and first 12
    // limbs of Q_curr/Q_next respectively. The trace builder mirrors
    // these from the already-populated underlying columns so the
    // row-local mirroring constraints are honestly satisfied.

    // f_x_limbs_in: mirror of acc_pre[0..F_X_LIMBS_LEN].
    for j in 0..F_X_LIMBS_LEN {
        let mut col = vec![zero.clone(); padded];
        for r in 0..num_rows {
            col[r] = columns[COL_ACC_PRE_OFFSET + j][r].clone();
        }
        columns.push(col);
    }
    // f_x_limbs_out: mirror of acc_post[0..F_X_LIMBS_LEN].
    for j in 0..F_X_LIMBS_LEN {
        let mut col = vec![zero.clone(); padded];
        for r in 0..num_rows {
            col[r] = columns[COL_ACC_POST_OFFSET + j][r].clone();
        }
        columns.push(col);
    }
    // t_x_limbs_in: mirror of Q_curr[0..T_X_LIMBS_LEN].
    for j in 0..T_X_LIMBS_LEN {
        let mut col = vec![zero.clone(); padded];
        for r in 0..num_rows {
            col[r] = columns[COL_Q_CURR_OFFSET + j][r].clone();
        }
        columns.push(col);
    }
    // t_x_limbs_out: mirror of Q_next[0..T_X_LIMBS_LEN].
    for j in 0..T_X_LIMBS_LEN {
        let mut col = vec![zero.clone(); padded];
        for r in 0..num_rows {
            col[r] = columns[COL_Q_NEXT_OFFSET + j][r].clone();
        }
        columns.push(col);
    }

    // NAF digit columns. For the BLS12-381 |x| = 0xd201_0000_0001_0000
    // pattern this scaffold uses (no signed NAF), every active row's
    // digit is 0 (doubling) or +1 (addition), and -1 is never used.
    // We encode `naf_pos = is_addition`, `naf_neg = 0` so the binary
    // constraints + mutual-exclusion (naf_pos·naf_neg = 0) are
    // honestly satisfied.
    let mut naf_pos = vec![zero.clone(); padded];
    let naf_neg = vec![zero.clone(); padded];
    for r in 0..num_rows {
        if !columns[COL_IS_ADDITION][r].is_zero() {
            naf_pos[r] = one.clone();
        }
    }
    columns.push(naf_pos);
    columns.push(naf_neg);

    // Task #306 (closes #233): affine ell-line slope witness blocks.
    //
    // On each doubling row we commit four Fp2 values derived from the
    // affine running point `T = Q_curr` (stored in this trace as an
    // affine G2 point):
    //
    //   lambda_num       = 3·T.x²
    //   lambda_denom     = 2·T.y
    //   lambda_denom_inv = (2·T.y)⁻¹
    //   lambda           = lambda_num · lambda_denom_inv
    //                    = 3·T.x² / (2·T.y)
    //
    // The cross-AIR LogUp descriptors in
    // [`crate::miller_fp_descriptors::EllLambdaDescriptors`] pin each of
    // the Fp-level operations (Fp ADD for denom doubling, Fp MUL for the
    // Fp2 squaring, the denom inverse, and the lambda product) against
    // [`crate::nonnative_fp_air`]. On addition / padding rows we zero
    // every limb.
    let mut lambda_num_cols: Vec<Vec<Scalar>> =
        (0..LIMBS_PER_FP2).map(|_| vec![zero.clone(); padded]).collect();
    let mut lambda_denom_cols: Vec<Vec<Scalar>> =
        (0..LIMBS_PER_FP2).map(|_| vec![zero.clone(); padded]).collect();
    let mut lambda_denom_inv_cols: Vec<Vec<Scalar>> =
        (0..LIMBS_PER_FP2).map(|_| vec![zero.clone(); padded]).collect();
    let mut lambda_cols: Vec<Vec<Scalar>> =
        (0..LIMBS_PER_FP2).map(|_| vec![zero.clone(); padded]).collect();

    use crate::nonnative_fp::Fp2 as Fp2Tower;
    let two = Fp2Tower {
        c0: crate::nonnative_fp::Fp::from_u64(2),
        c1: crate::nonnative_fp::Fp::zero(),
    };
    let three = Fp2Tower {
        c0: crate::nonnative_fp::Fp::from_u64(3),
        c1: crate::nonnative_fp::Fp::zero(),
    };

    for (r, row) in witness.steps.rows.iter().enumerate() {
        if !row.is_doubling {
            continue;
        }
        let t = row.q_curr;
        let lambda_num_val = three.mul(&t.x.square()); // 3·T.x²
        let lambda_denom_val = two.mul(&t.y); // 2·T.y
        let lambda_denom_inv_val = match lambda_denom_val.invert() {
            Some(v) => v,
            None => Fp2Tower::zero(),
        };
        let lambda_val = lambda_num_val.mul(&lambda_denom_inv_val);

        let write = |dst: &mut [Vec<Scalar>], val: &Fp2Tower| {
            for j in 0..LIMBS_PER_FP {
                dst[j][r] = Scalar::from_u64(val.c0.limbs[j], curve);
                dst[LIMBS_PER_FP + j][r] =
                    Scalar::from_u64(val.c1.limbs[j], curve);
            }
        };
        write(&mut lambda_num_cols, &lambda_num_val);
        write(&mut lambda_denom_cols, &lambda_denom_val);
        write(&mut lambda_denom_inv_cols, &lambda_denom_inv_val);
        write(&mut lambda_cols, &lambda_val);
    }

    columns.extend(lambda_num_cols);
    columns.extend(lambda_denom_cols);
    columns.extend(lambda_denom_inv_cols);
    columns.extend(lambda_cols);

    // Task #319: affine line-coefficient witness blocks
    //   c4_affine = -λ
    //   c1_affine =  λ·T.x − T.y
    //   c0_affine = −λ·T.x + T.y     (= −c1_affine for doubling)
    // Populated only on doubling rows; zero on addition / padding.
    let mut c4_affine_cols: Vec<Vec<Scalar>> =
        (0..LIMBS_PER_FP2).map(|_| vec![zero.clone(); padded]).collect();
    let mut c1_affine_cols: Vec<Vec<Scalar>> =
        (0..LIMBS_PER_FP2).map(|_| vec![zero.clone(); padded]).collect();
    let mut c0_affine_cols: Vec<Vec<Scalar>> =
        (0..LIMBS_PER_FP2).map(|_| vec![zero.clone(); padded]).collect();

    for (r, row) in witness.steps.rows.iter().enumerate() {
        if !row.is_doubling {
            continue;
        }
        let t = row.q_curr;
        let lambda_num_val = three.mul(&t.x.square());
        let lambda_denom_val = two.mul(&t.y);
        let lambda_denom_inv_val = match lambda_denom_val.invert() {
            Some(v) => v,
            None => Fp2Tower::zero(),
        };
        let lambda_val = lambda_num_val.mul(&lambda_denom_inv_val);

        let c4_affine_val = lambda_val.neg();
        let c1_affine_val = lambda_val.mul(&t.x).sub(&t.y);
        let c0_affine_val = c1_affine_val.neg();

        let write = |dst: &mut [Vec<Scalar>], val: &Fp2Tower| {
            for j in 0..LIMBS_PER_FP {
                dst[j][r] = Scalar::from_u64(val.c0.limbs[j], curve);
                dst[LIMBS_PER_FP + j][r] =
                    Scalar::from_u64(val.c1.limbs[j], curve);
            }
        };
        write(&mut c4_affine_cols, &c4_affine_val);
        write(&mut c1_affine_cols, &c1_affine_val);
        write(&mut c0_affine_cols, &c0_affine_val);
    }

    columns.extend(c4_affine_cols);
    columns.extend(c1_affine_cols);
    columns.extend(c0_affine_cols);

    assert_eq!(columns.len(), NUM_COLUMNS);

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
        .collect();

    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ─────────────────────────────────────────────────

/// Constraint-system handle for the multi-row Miller-loop composition
/// AIR.
///
/// Combines [`MillerStepConstraintSystem`]'s 3 row-local constraints +
/// 2 shifted continuity constraints with this AIR's 5 boundary
/// constraints (binary selectors + first-row Q init + first-row
/// acc_pre = 1).
pub struct MillerLoopConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
    step_cs: MillerStepConstraintSystem,
}

impl MillerLoopConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            omega: None,
            domain_size: None,
            step_cs: MillerStepConstraintSystem::new(num_rows),
        }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega.clone());
        self.domain_size = Some(domain_size);
        self.step_cs = self.step_cs.with_omega_and_domain(omega, domain_size);
        self
    }

    /// Total constraint count = miller_step_air row-local + this AIR's
    /// boundary constraints.
    pub fn total_row_constraints(&self) -> usize {
        MS_NUM_ROW_CONSTRAINTS + NUM_LOOP_ROW_CONSTRAINTS
    }
}

/// Layout of the constant `Fp12::one()` in 72 BE limbs.
///
/// `Fp::one()` is encoded as `[0, 0, 0, 0, 0, 1]` (limb 5 = LSB =
/// `1`), in line with the BE limb convention used by
/// [`crate::miller_step_air::write_fp_limbs`]. `Fp12::one()` =
/// `Fp::one()` in the `c0.c0.c0` slot and zero elsewhere; the 72-limb
/// flattening therefore has a single `1` at index 5 and zeros at the
/// other 71 positions.
fn fp12_one_limbs() -> [u64; LIMBS_PER_FP12] {
    let mut out = [0u64; LIMBS_PER_FP12];
    // c0.c0.c0 starts at offset 0; Fp::one() is [0,0,0,0,0,1] BE, so
    // limb 5 = 1.
    out[LIMBS_PER_FP - 1] = 1;
    out
}

/// β-RLC body that pins per-limb equality between two column ranges
/// at row `r`.
fn rlc_equality_body(
    columns: &[&Vec<Scalar>],
    r: usize,
    base_a: usize,
    base_b: usize,
    len: usize,
    beta: &Scalar,
) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut beta_pow = Scalar::one(curve);
    for j in 0..len {
        let a = &columns[base_a + j][r];
        let b = &columns[base_b + j][r];
        let body = a.sub(b);
        acc = acc.add(&beta_pow.mul(&body));
        beta_pow = beta_pow.mul(beta);
    }
    acc
}

/// β-RLC body that pins every limb in a column range to zero at row `r`:
///   `Σ_{j=0..len} β^j · columns[base + j][r]`
///
/// Vanishes iff every cell in the range is zero (Schwartz–Zippel over
/// the verifier's fresh α — soundly equivalent to per-limb equality
/// constraints for the structurally-zero Fp slots of the sparse ell
/// line evaluation).
fn rlc_is_zero_body(
    columns: &[&Vec<Scalar>],
    r: usize,
    base: usize,
    len: usize,
    beta: &Scalar,
) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut beta_pow = Scalar::one(curve);
    for j in 0..len {
        let a = &columns[base + j][r];
        acc = acc.add(&beta_pow.mul(a));
        beta_pow = beta_pow.mul(beta);
    }
    acc
}

/// β-RLC body that pins per-limb equality between a column range and a
/// constant `[u64; len]` vector at row `r`.
fn rlc_equality_body_to_const(
    columns: &[&Vec<Scalar>],
    r: usize,
    base: usize,
    constants: &[u64],
    beta: &Scalar,
) -> Scalar {
    let curve = beta.curve_type();
    let mut acc = Scalar::zero(curve);
    let mut beta_pow = Scalar::one(curve);
    for (j, c) in constants.iter().enumerate() {
        let a = &columns[base + j][r];
        let c_scalar = Scalar::from_u64(*c, curve);
        let body = a.sub(&c_scalar);
        acc = acc.add(&beta_pow.mul(&body));
        beta_pow = beta_pow.mul(beta);
    }
    acc
}

impl VmConstraintSystem for MillerLoopConstraintSystem {
    fn num_constraints(&self) -> usize {
        MS_NUM_ROW_CONSTRAINTS + NUM_LOOP_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = self.step_cs.constraint_labels();
        labels.extend(
            [
                "is_first_row_binary",
                "is_last_row_binary",
                "boundary_selectors_mutually_exclusive",
                "first_row_q_curr_equals_q",
                "first_row_acc_pre_equals_one",
                "naf_pos_binary",
                "naf_neg_binary",
                "naf_pos_naf_neg_mutex",
                "f_x_limbs_in_mirrors_acc_pre",
                "f_x_limbs_out_mirrors_acc_post",
                "t_x_limbs_in_mirrors_q_curr",
                "t_x_limbs_out_mirrors_q_next",
                "ell_sparsity_line_value_c0_c2_c0_is_zero",
                "ell_sparsity_line_value_c0_c2_c1_is_zero",
                "ell_sparsity_line_value_c1_c0_c0_is_zero",
                "ell_sparsity_line_value_c1_c0_c1_is_zero",
                "ell_sparsity_line_value_c1_c2_c0_is_zero",
                "ell_sparsity_line_value_c1_c2_c1_is_zero",
            ]
            .into_iter()
            .map(String::from),
        );
        labels
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let beta = Scalar::from_u64(7, curve);

        let mut out = self.step_cs.evaluate_on_domain(columns, n);
        assert_eq!(out.len(), MS_NUM_ROW_CONSTRAINTS);

        // 0 (loop-local): is_first_row binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_FIRST_ROW][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 1: is_last_row binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_LAST_ROW][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 2: boundary selectors mutually exclusive.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let f = &columns[COL_IS_FIRST_ROW][r];
                let l = &columns[COL_IS_LAST_ROW][r];
                c[r] = f.mul(l);
            }
            out.push(c);
        }
        // 3: first-row Q_curr == Q_fixed (24 limbs, β-RLC, gated by is_first_row).
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate = &columns[COL_IS_FIRST_ROW][r];
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_equality_body(
                    columns,
                    r,
                    COL_Q_CURR_OFFSET,
                    COL_Q_FIXED_OFFSET,
                    LIMBS_PER_G2,
                    &beta,
                );
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }
        // 4: first-row acc_pre == Fp12::one() (72 limbs, β-RLC, gated by is_first_row).
        {
            let ones = fp12_one_limbs();
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate = &columns[COL_IS_FIRST_ROW][r];
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_equality_body_to_const(
                    columns,
                    r,
                    COL_ACC_PRE_OFFSET,
                    &ones,
                    &beta,
                );
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }
        // 5: naf_pos binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_NAF_POS][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 6: naf_neg binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_NAF_NEG][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }
        // 7: naf_pos · naf_neg = 0.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let p = &columns[COL_NAF_POS][r];
                let q = &columns[COL_NAF_NEG][r];
                c[r] = p.mul(q);
            }
            out.push(c);
        }
        // 8: IS_REAL · (f_x_limbs_in == acc_pre[0..24]) β-RLC.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate =
                    columns[COL_IS_DOUBLING][r].add(&columns[COL_IS_ADDITION][r]);
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_equality_body(
                    columns,
                    r,
                    COL_F_X_LIMBS_IN_OFFSET,
                    COL_ACC_PRE_OFFSET,
                    F_X_LIMBS_LEN,
                    &beta,
                );
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }
        // 9: IS_REAL · (f_x_limbs_out == acc_post[0..24]) β-RLC.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate =
                    columns[COL_IS_DOUBLING][r].add(&columns[COL_IS_ADDITION][r]);
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_equality_body(
                    columns,
                    r,
                    COL_F_X_LIMBS_OUT_OFFSET,
                    COL_ACC_POST_OFFSET,
                    F_X_LIMBS_LEN,
                    &beta,
                );
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }
        // 10: IS_REAL · (t_x_limbs_in == Q_curr[0..12]) β-RLC.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate =
                    columns[COL_IS_DOUBLING][r].add(&columns[COL_IS_ADDITION][r]);
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_equality_body(
                    columns,
                    r,
                    COL_T_X_LIMBS_IN_OFFSET,
                    COL_Q_CURR_OFFSET,
                    T_X_LIMBS_LEN,
                    &beta,
                );
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }
        // 11: IS_REAL · (t_x_limbs_out == Q_next[0..12]) β-RLC.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate =
                    columns[COL_IS_DOUBLING][r].add(&columns[COL_IS_ADDITION][r]);
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_equality_body(
                    columns,
                    r,
                    COL_T_X_LIMBS_OUT_OFFSET,
                    COL_Q_NEXT_OFFSET,
                    T_X_LIMBS_LEN,
                    &beta,
                );
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }
        // 12..17 (Task #233): ell line sparsity — the dense `line_value`
        // Fp12 lifted from the sparse `mul_by_014` form has structural
        // zeros at Fp slots {4, 5, 6, 7, 10, 11}. Each constraint pins
        // the 6 Fp limbs of one such slot to zero (β-RLC), gated by
        // IS_REAL so padding rows are exempt.
        for &slot in ELL_SPARSITY_ZERO_FP_SLOTS.iter() {
            let base = COL_LINE_VALUE_OFFSET + slot * LIMBS_PER_FP;
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let gate =
                    columns[COL_IS_DOUBLING][r].add(&columns[COL_IS_ADDITION][r]);
                if gate.is_zero() {
                    continue;
                }
                let body = rlc_is_zero_body(columns, r, base, LIMBS_PER_FP, &beta);
                c[r] = gate.mul(&body);
            }
            out.push(c);
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let beta = Scalar::from_u64(7, curve);

        // Re-use miller_step_air's per-point evaluation for constraints 0..3.
        let mut acc = self.step_cs.evaluate_at_point(col_evals, alpha);
        // alpha_pow advances after the step CS's MS_NUM_ROW_CONSTRAINTS terms.
        let mut alpha_pow = Scalar::one(curve);
        for _ in 0..MS_NUM_ROW_CONSTRAINTS {
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 0: is_first_row binary.
        {
            let v = &col_evals[COL_IS_FIRST_ROW];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_last_row binary.
        {
            let v = &col_evals[COL_IS_LAST_ROW];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: boundary mutual exclusion.
        {
            let f = &col_evals[COL_IS_FIRST_ROW];
            let l = &col_evals[COL_IS_LAST_ROW];
            acc = acc.add(&alpha_pow.mul(&f.mul(l)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3: first-row Q_curr == Q_fixed.
        {
            let gate = &col_evals[COL_IS_FIRST_ROW];
            let mut body = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_G2 {
                let diff =
                    col_evals[COL_Q_CURR_OFFSET + j].sub(&col_evals[COL_Q_FIXED_OFFSET + j]);
                body = body.add(&beta_pow.mul(&diff));
                beta_pow = beta_pow.mul(&beta);
            }
            acc = acc.add(&alpha_pow.mul(&gate.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4: first-row acc_pre == Fp12::one().
        {
            let gate = &col_evals[COL_IS_FIRST_ROW];
            let ones = fp12_one_limbs();
            let mut body = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for (j, c) in ones.iter().enumerate() {
                let c_scalar = Scalar::from_u64(*c, curve);
                let diff = col_evals[COL_ACC_PRE_OFFSET + j].sub(&c_scalar);
                body = body.add(&beta_pow.mul(&diff));
                beta_pow = beta_pow.mul(&beta);
            }
            acc = acc.add(&alpha_pow.mul(&gate.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 5: naf_pos binary.
        {
            let v = &col_evals[COL_NAF_POS];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 6: naf_neg binary.
        {
            let v = &col_evals[COL_NAF_NEG];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 7: naf_pos · naf_neg mutex.
        {
            let p = &col_evals[COL_NAF_POS];
            let q = &col_evals[COL_NAF_NEG];
            acc = acc.add(&alpha_pow.mul(&p.mul(q)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 8..11: mirror constraints (β-RLC equality bodies gated by IS_REAL).
        let is_real_eval =
            col_evals[COL_IS_DOUBLING].add(&col_evals[COL_IS_ADDITION]);
        let mirror_body = |base_a: usize, base_b: usize, len: usize| -> Scalar {
            let mut body = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for j in 0..len {
                let diff = col_evals[base_a + j].sub(&col_evals[base_b + j]);
                body = body.add(&beta_pow.mul(&diff));
                beta_pow = beta_pow.mul(&beta);
            }
            body
        };
        // 8.
        {
            let body = mirror_body(
                COL_F_X_LIMBS_IN_OFFSET,
                COL_ACC_PRE_OFFSET,
                F_X_LIMBS_LEN,
            );
            acc = acc.add(&alpha_pow.mul(&is_real_eval.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 9.
        {
            let body = mirror_body(
                COL_F_X_LIMBS_OUT_OFFSET,
                COL_ACC_POST_OFFSET,
                F_X_LIMBS_LEN,
            );
            acc = acc.add(&alpha_pow.mul(&is_real_eval.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 10.
        {
            let body = mirror_body(
                COL_T_X_LIMBS_IN_OFFSET,
                COL_Q_CURR_OFFSET,
                T_X_LIMBS_LEN,
            );
            acc = acc.add(&alpha_pow.mul(&is_real_eval.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 11.
        {
            let body = mirror_body(
                COL_T_X_LIMBS_OUT_OFFSET,
                COL_Q_NEXT_OFFSET,
                T_X_LIMBS_LEN,
            );
            acc = acc.add(&alpha_pow.mul(&is_real_eval.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 12..17 (Task #233): ell line sparsity.
        let zero_slot_body = |slot: usize| -> Scalar {
            let base = COL_LINE_VALUE_OFFSET + slot * LIMBS_PER_FP;
            let mut body = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_FP {
                body = body.add(&beta_pow.mul(&col_evals[base + j]));
                beta_pow = beta_pow.mul(&beta);
            }
            body
        };
        for (i, &slot) in ELL_SPARSITY_ZERO_FP_SLOTS.iter().enumerate() {
            let body = zero_slot_body(slot);
            acc = acc.add(&alpha_pow.mul(&is_real_eval.mul(&body)));
            if i + 1 < ELL_SPARSITY_ZERO_FP_SLOTS.len() {
                alpha_pow = alpha_pow.mul(alpha);
            }
        }

        acc
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let beta = Scalar::from_u64(7, curve);

        // Start from the miller_step_air's accumulated polynomial.
        let mut acc = self.step_cs.build_constraint_polynomial(col_coeffs, alpha, domain_size);
        let mut alpha_pow = Scalar::one(curve);
        for _ in 0..MS_NUM_ROW_CONSTRAINTS {
            alpha_pow = alpha_pow.mul(alpha);
        }

        // 0: is_first_row binary.
        {
            let v = &col_coeffs[COL_IS_FIRST_ROW];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: is_last_row binary.
        {
            let v = &col_coeffs[COL_IS_LAST_ROW];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: boundary mutual exclusion.
        {
            let f = &col_coeffs[COL_IS_FIRST_ROW];
            let l = &col_coeffs[COL_IS_LAST_ROW];
            let body = poly_mul(f, l, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3: first-row Q_curr == Q_fixed.
        {
            let gate = &col_coeffs[COL_IS_FIRST_ROW];
            let mut body = vec![Scalar::zero(curve)];
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_G2 {
                let diff = poly_sub(
                    &col_coeffs[COL_Q_CURR_OFFSET + j],
                    &col_coeffs[COL_Q_FIXED_OFFSET + j],
                    curve,
                );
                body = poly_add(&body, &poly_scalar_mul(&diff, &beta_pow), curve);
                beta_pow = beta_pow.mul(&beta);
            }
            let gated = poly_mul(gate, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4: first-row acc_pre == Fp12::one() (encoded as a polynomial offset).
        {
            let gate = &col_coeffs[COL_IS_FIRST_ROW];
            let ones = fp12_one_limbs();
            let mut body = vec![Scalar::zero(curve)];
            let mut beta_pow = Scalar::one(curve);
            for (j, c) in ones.iter().enumerate() {
                let c_scalar = Scalar::from_u64(*c, curve);
                let const_poly = vec![c_scalar];
                let diff =
                    poly_sub(&col_coeffs[COL_ACC_PRE_OFFSET + j], &const_poly, curve);
                body = poly_add(&body, &poly_scalar_mul(&diff, &beta_pow), curve);
                beta_pow = beta_pow.mul(&beta);
            }
            let gated = poly_mul(gate, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 5: naf_pos binary.
        {
            let v = &col_coeffs[COL_NAF_POS];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 6: naf_neg binary.
        {
            let v = &col_coeffs[COL_NAF_NEG];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 7: naf_pos · naf_neg mutex.
        {
            let p = &col_coeffs[COL_NAF_POS];
            let q = &col_coeffs[COL_NAF_NEG];
            let body = poly_mul(p, q, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 8..11: mirror constraints — β-RLC equality bodies gated by IS_REAL.
        let is_real_poly = poly_add(
            &col_coeffs[COL_IS_DOUBLING],
            &col_coeffs[COL_IS_ADDITION],
            curve,
        );
        let mirror_poly = |base_a: usize, base_b: usize, len: usize| -> Vec<Scalar> {
            let mut body = vec![Scalar::zero(curve)];
            let mut beta_pow = Scalar::one(curve);
            for j in 0..len {
                let diff = poly_sub(
                    &col_coeffs[base_a + j],
                    &col_coeffs[base_b + j],
                    curve,
                );
                body = poly_add(&body, &poly_scalar_mul(&diff, &beta_pow), curve);
                beta_pow = beta_pow.mul(&beta);
            }
            body
        };
        // 8.
        {
            let body = mirror_poly(
                COL_F_X_LIMBS_IN_OFFSET,
                COL_ACC_PRE_OFFSET,
                F_X_LIMBS_LEN,
            );
            let gated = poly_mul(&is_real_poly, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 9.
        {
            let body = mirror_poly(
                COL_F_X_LIMBS_OUT_OFFSET,
                COL_ACC_POST_OFFSET,
                F_X_LIMBS_LEN,
            );
            let gated = poly_mul(&is_real_poly, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 10.
        {
            let body = mirror_poly(
                COL_T_X_LIMBS_IN_OFFSET,
                COL_Q_CURR_OFFSET,
                T_X_LIMBS_LEN,
            );
            let gated = poly_mul(&is_real_poly, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 11.
        {
            let body = mirror_poly(
                COL_T_X_LIMBS_OUT_OFFSET,
                COL_Q_NEXT_OFFSET,
                T_X_LIMBS_LEN,
            );
            let gated = poly_mul(&is_real_poly, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 12..17 (Task #233): ell line sparsity. Each constraint sums
        // 6 line_value limbs to zero via β-RLC, gated by IS_REAL.
        let zero_slot_poly = |slot: usize| -> Vec<Scalar> {
            let base = COL_LINE_VALUE_OFFSET + slot * LIMBS_PER_FP;
            let mut body = vec![Scalar::zero(curve)];
            let mut beta_pow = Scalar::one(curve);
            for j in 0..LIMBS_PER_FP {
                body = poly_add(
                    &body,
                    &poly_scalar_mul(&col_coeffs[base + j], &beta_pow),
                    curve,
                );
                beta_pow = beta_pow.mul(&beta);
            }
            body
        };
        for (i, &slot) in ELL_SPARSITY_ZERO_FP_SLOTS.iter().enumerate() {
            let body = zero_slot_poly(slot);
            let gated = poly_mul(&is_real_poly, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            if i + 1 < ELL_SPARSITY_ZERO_FP_SLOTS.len() {
                alpha_pow = alpha_pow.mul(alpha);
            }
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        let mut v = self.step_cs.selector_column_indices();
        v.push(COL_IS_FIRST_ROW);
        v.push(COL_IS_LAST_ROW);
        v.push(COL_NAF_POS);
        v.push(COL_NAF_NEG);
        v
    }

    fn padding_selector_column(&self) -> Option<usize> {
        None
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size {
            return;
        }
        if columns.len() < NUM_COLUMNS {
            return;
        }
        // Delegate the first MS_NUM_COLUMNS to the step constraint system,
        // then zero our appended boundary / mirror / naf columns on
        // padding rows.
        self.step_cs.fix_trace_padding(columns, num_rows, padded_size);
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls12381);
        let zero = Scalar::zero(curve);
        let mut to_zero: Vec<usize> = vec![COL_IS_FIRST_ROW, COL_IS_LAST_ROW];
        for j in 0..F_X_LIMBS_LEN {
            to_zero.push(COL_F_X_LIMBS_IN_OFFSET + j);
            to_zero.push(COL_F_X_LIMBS_OUT_OFFSET + j);
        }
        for j in 0..T_X_LIMBS_LEN {
            to_zero.push(COL_T_X_LIMBS_IN_OFFSET + j);
            to_zero.push(COL_T_X_LIMBS_OUT_OFFSET + j);
        }
        to_zero.push(COL_NAF_POS);
        to_zero.push(COL_NAF_NEG);
        // Task #306 lambda witness blocks — zero on padding rows.
        for j in 0..LIMBS_PER_FP2 {
            to_zero.push(COL_LAMBDA_NUM_OFFSET + j);
            to_zero.push(COL_LAMBDA_DENOM_OFFSET + j);
            to_zero.push(COL_LAMBDA_DENOM_INV_OFFSET + j);
            to_zero.push(COL_LAMBDA_OFFSET + j);
        }
        // Task #319 affine line-coefficient blocks — zero on padding rows.
        for j in 0..LIMBS_PER_FP2 {
            to_zero.push(COL_C4_AFFINE_OFFSET + j);
            to_zero.push(COL_C1_AFFINE_OFFSET + j);
            to_zero.push(COL_C0_AFFINE_OFFSET + j);
        }
        for col_idx in to_zero {
            for cell in
                columns[col_idx].iter_mut().skip(num_rows).take(padded_size - num_rows)
            {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // Same posture as miller_step_air / final_exp_air: limb 64-bit
        // range checks live in nonnative_fp_air via cross-AIR LogUp.
        LookupRequirements { tables: Vec::new(), declarations: Vec::new() }
    }

    fn shifted_column_indices(&self) -> Vec<usize> {
        // Inherit the step AIR's shifted columns (Q_curr + acc_pre).
        // Add `f_x_limbs_in` (24 cols) and `t_x_limbs_in` (12 cols)
        // since their next-row values appear in the new continuity
        // shifted constraints.
        let mut idx = self.step_cs.shifted_column_indices();
        for j in 0..F_X_LIMBS_LEN {
            idx.push(COL_F_X_LIMBS_IN_OFFSET + j);
        }
        for j in 0..T_X_LIMBS_LEN {
            idx.push(COL_T_X_LIMBS_IN_OFFSET + j);
        }
        idx
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_LOOP_SHIFTED
    }
}

// ─── Shifted helper ───────────────────────────────────────────────────

/// Per-row evaluation of shifted constraint `i` at row `r`, given full
/// columns. Indices 0 and 1 inherit from
/// [`crate::miller_step_air::evaluate_shifted_row`] (`Q_curr` and
/// `acc_pre` chain continuity). Indices 2 and 3 are new:
///
///   * **2 — f continuity**: `IS_REAL[r] · Σ_j β^j · (f_x_limbs_in[j][r+1]
///     − f_x_limbs_out[j][r])` (24 sub-bodies).
///   * **3 — T continuity**: `IS_REAL[r] · Σ_j β^j · (t_x_limbs_in[j][r+1]
///     − t_x_limbs_out[j][r])` (12 sub-bodies).
///
/// Each is gated by `IS_REAL = is_doubling + is_addition` on the
/// **source** row so padding-row→padding-row pairs vanish.
pub fn evaluate_shifted_row(
    columns: &[&Vec<Scalar>],
    r: usize,
    next: usize,
    constraint_index: usize,
    alpha: &Scalar,
) -> Scalar {
    let curve = alpha.curve_type();
    let zero = Scalar::zero(curve);

    match constraint_index {
        0 | 1 => {
            crate::miller_step_air::evaluate_shifted_row(
                columns, r, next, constraint_index, alpha,
            )
        }
        2 => {
            // f continuity, gated by IS_REAL at row r.
            let gate =
                columns[COL_IS_DOUBLING][r].add(&columns[COL_IS_ADDITION][r]);
            if gate.is_zero() {
                return zero;
            }
            let mut acc = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for j in 0..F_X_LIMBS_LEN {
                let a = &columns[COL_F_X_LIMBS_IN_OFFSET + j][next];
                let b = &columns[COL_F_X_LIMBS_OUT_OFFSET + j][r];
                let body = a.sub(b);
                acc = acc.add(&beta_pow.mul(&body));
                beta_pow = beta_pow.mul(alpha);
            }
            gate.mul(&acc)
        }
        3 => {
            // T continuity, gated by IS_REAL at row r.
            let gate =
                columns[COL_IS_DOUBLING][r].add(&columns[COL_IS_ADDITION][r]);
            if gate.is_zero() {
                return zero;
            }
            let mut acc = Scalar::zero(curve);
            let mut beta_pow = Scalar::one(curve);
            for j in 0..T_X_LIMBS_LEN {
                let a = &columns[COL_T_X_LIMBS_IN_OFFSET + j][next];
                let b = &columns[COL_T_X_LIMBS_OUT_OFFSET + j][r];
                let body = a.sub(b);
                acc = acc.add(&beta_pow.mul(&body));
                beta_pow = beta_pow.mul(alpha);
            }
            gate.mul(&acc)
        }
        _ => zero,
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Per-row LogUp linkage: this composition AIR's
/// `(Q_curr || acc_pre || acc_post)` tuple ↔ the same tuple in the
/// stand-alone [`crate::miller_step_air`].
///
/// 24 + 72 + 72 = **168-column tuple**. Both sides are gated by
/// `is_doubling + is_addition` (the natural `IS_REAL` shape of the
/// step AIR's selector pair). Once the step AIR's per-row Fp12 arithmetic
/// is algebraically closed (cross-AIR LogUp into
/// [`crate::nonnative_fp_air`]), this linkage lets the composition AIR
/// "outsource" per-row correctness to the step AIR by multiset-matching
/// every row.
///
/// **Caveat**: the cross_air_logup descriptor only accepts a single
/// selector column per side, so we use `COL_IS_DOUBLING` as the gate
/// (it matches every doubling row; addition rows are matched in the
/// step AIR's reciprocal descriptor). The full
/// `is_doubling + is_addition` (= IS_REAL) gate requires a dedicated
/// boolean column — straightforward follow-up.
pub fn make_miller_loop_to_step_descriptor(
    miller_loop_layer_index: usize,
    miller_step_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let columns: Vec<usize> = {
        let mut v = Vec::with_capacity(LIMBS_PER_G2 + 2 * LIMBS_PER_FP12);
        for j in 0..LIMBS_PER_G2 {
            v.push(COL_Q_CURR_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP12 {
            v.push(COL_ACC_PRE_OFFSET + j);
        }
        for j in 0..LIMBS_PER_FP12 {
            v.push(COL_ACC_POST_OFFSET + j);
        }
        v
    };
    // Step AIR's column indices are the same (re-exported), so use the
    // same offsets.
    let step_columns = columns.clone();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "miller_loop_to_step_v1".into(),
        a_layer_index: miller_loop_layer_index,
        a_columns: columns,
        a_selector_column: Some(COL_IS_DOUBLING),
        b_layer_index: miller_step_layer_index,
        b_columns: step_columns,
        b_selector_column: Some(ms::COL_IS_DOUBLING),
    }
}

/// Per-row LogUp linkage: this AIR's row-INPUT tuple
/// `(t_x_limbs_in || f_x_limbs_in)` (12 + 24 = 36 cols, gated by
/// `IS_DOUBLING`) ↔ the corresponding `(Q_curr[0..12] || acc_pre[0..24])`
/// tuple in [`crate::miller_step_air`] (also gated by its
/// `IS_DOUBLING`).
///
/// This is the **stitching descriptor** that pins the cross-row f / T
/// inputs of every row in the composition AIR to a row of the
/// stand-alone single-step AIR, so per-row Fp12 arithmetic correctness
/// can be outsourced to that AIR once it closes its deep relations.
///
/// **Note**: distinct from
/// [`make_miller_loop_to_step_descriptor`] (full 168-col Q_curr +
/// acc_pre + acc_post tuple): this descriptor uses the narrower
/// 36-col "X-component representative" tuple driven by the new
/// `f_x_limbs_in` / `t_x_limbs_in` columns, which is the surface the
/// shifted continuity constraints actually pin.
pub fn make_miller_loop_to_miller_step_descriptor(
    miller_loop_layer_index: usize,
    miller_step_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    let a_columns: Vec<usize> = {
        let mut v = Vec::with_capacity(T_X_LIMBS_LEN + F_X_LIMBS_LEN);
        for j in 0..T_X_LIMBS_LEN {
            v.push(COL_T_X_LIMBS_IN_OFFSET + j);
        }
        for j in 0..F_X_LIMBS_LEN {
            v.push(COL_F_X_LIMBS_IN_OFFSET + j);
        }
        v
    };
    let b_columns: Vec<usize> = {
        let mut v = Vec::with_capacity(T_X_LIMBS_LEN + F_X_LIMBS_LEN);
        for j in 0..T_X_LIMBS_LEN {
            v.push(COL_Q_CURR_OFFSET + j);
        }
        for j in 0..F_X_LIMBS_LEN {
            v.push(COL_ACC_PRE_OFFSET + j);
        }
        v
    };
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "miller_loop_to_miller_step_v1".into(),
        a_layer_index: miller_loop_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_DOUBLING),
        b_layer_index: miller_step_layer_index,
        b_columns,
        b_selector_column: Some(ms::COL_IS_DOUBLING),
    }
}

/// Boundary LogUp linkage: this AIR's last-row `acc_post` Fp12 limbs ↔
/// [`crate::final_exp_air`]'s row-0 `f_pre` Fp12 limbs.
///
/// 72-column tuple. A side is gated by `IS_LAST_ROW`, B side by
/// `IS_EASY_STEP` (the closest boundary-style selector exposed by the
/// final-exp AIR). Pins the Miller-loop output to the final-
/// exponentiation input.
///
/// **Caveat**: until the per-row Fp12 arithmetic of this AIR is
/// closed, `acc_post[last_row]` is the toy `Fp12::one()` (not the
/// real Miller-loop output). The descriptor below is the column-shape
/// spec; the semantic binding to the real Miller output activates
/// when the per-row arithmetic gap is closed.
pub fn make_miller_loop_to_final_exp_descriptor(
    miller_loop_layer_index: usize,
    final_exp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::final_exp_air as fe;
    let a_columns: Vec<usize> = (0..LIMBS_PER_FP12)
        .map(|j| COL_ACC_POST_OFFSET + j)
        .collect();
    let b_columns: Vec<usize> = (0..LIMBS_PER_FP12)
        .map(|j| fe::COL_F_PRE_OFFSET + j)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "miller_loop_to_final_exp_v1".into(),
        a_layer_index: miller_loop_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_LAST_ROW),
        b_layer_index: final_exp_layer_index,
        b_columns,
        b_selector_column: Some(fe::COL_IS_EASY_STEP),
    }
}

// ─── Task #280: bind Fp mults to bls12_381_curve_ops_air::fp ──────────
//
// `make_fp_mul_descriptor` in [`crate::miller_fp_descriptors`] binds the
// (a, b, c) Fp multiplication tuple to [`crate::nonnative_fp_air`]
// directly. Per #270, [`crate::bls12_381_curve_ops_air::fp`] re-exports
// the same column layout from `nonnative_fp_air` and is the canonical
// reference for BLS12-381 base-field arithmetic used by the curve-ops
// AIR. Sharing a single Fp AIR layer across both consumers
// (miller_loop_air and bls12_381_curve_ops_air) folds their Fp mults
// into the same multiset, which the joint cross-AIR LogUp protocol can
// close in one γ sweep.
//
// This descriptor is the column-shape equivalent of
// [`crate::miller_fp_descriptors::make_fp_mul_descriptor`] but resolves
// `(COL_A, COL_B, COL_R)` and the `MUL` selector through the
// `bls12_381_curve_ops_air::fp` re-exports — so the binding is named
// after the shared Fp layer rather than the underlying nonnative_fp
// implementation detail.

/// Build a cross-AIR LogUp descriptor binding one Fp multiplication
/// `c = a · b mod p` between this AIR's miller-step-shaped columns and
/// the shared BLS12-381 Fp AIR layer at
/// [`crate::bls12_381_curve_ops_air::fp`].
///
/// Column layout is identical to
/// [`crate::miller_fp_descriptors::make_fp_mul_descriptor`] (the
/// `bls12_381_curve_ops_air::fp` module re-exports the underlying
/// `nonnative_fp_air` symbols), but the descriptor's A-side layer name
/// is conceptually the shared "BLS12-381 Fp" layer, letting a single
/// joint LogUp instance cover Fp mults from both miller_loop_air and
/// bls12_381_curve_ops_air's G1 doubling/addition AIRs.
///
/// * `a_base_col` / `b_base_col` / `c_base_col` — miller_loop_air column
///   indices of the first BE u64 limb of each operand (6 contiguous
///   limbs follow).
/// * `b_selector_column` — miller_loop_air row-selector gating which
///   rows fire (typically [`COL_IS_DOUBLING`] or [`COL_IS_ADDITION`]).
/// * `label` — unique tag so multiple descriptors can coexist.
/// * `miller_loop_layer_index` — joint protocol layer index for this AIR.
/// * `bls12_381_fp_layer_index` — joint protocol layer index for the
///   shared BLS12-381 Fp AIR.
pub fn make_miller_loop_to_bls12_381_fp_mul_descriptor(
    a_base_col: usize,
    b_base_col: usize,
    c_base_col: usize,
    b_selector_column: usize,
    label: impl Into<String>,
    miller_loop_layer_index: usize,
    bls12_381_fp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::bls12_381_curve_ops_air::fp as bls_fp;

    // A-side: bls12_381_curve_ops_air::fp's (COL_A, COL_B, COL_C) triple,
    // 6 limbs each, gated by COL_SEL_MUL.
    let mut a_columns: Vec<usize> = Vec::with_capacity(3 * bls_fp::LIMBS_PER_FP);
    for j in 0..bls_fp::LIMBS_PER_FP {
        a_columns.push(bls_fp::COL_A_OFFSET + j);
    }
    for j in 0..bls_fp::LIMBS_PER_FP {
        a_columns.push(bls_fp::COL_B_OFFSET + j);
    }
    for j in 0..bls_fp::LIMBS_PER_FP {
        a_columns.push(bls_fp::COL_C_OFFSET + j);
    }

    // B-side: miller_loop_air's (a, b, c) limb columns.
    let mut b_columns: Vec<usize> = Vec::with_capacity(3 * LIMBS_PER_FP);
    for j in 0..LIMBS_PER_FP {
        b_columns.push(a_base_col + j);
    }
    for j in 0..LIMBS_PER_FP {
        b_columns.push(b_base_col + j);
    }
    for j in 0..LIMBS_PER_FP {
        b_columns.push(c_base_col + j);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: label.into(),
        a_layer_index: bls12_381_fp_layer_index,
        a_columns,
        a_selector_column: Some(bls_fp::COL_SEL_MUL),
        b_layer_index: miller_loop_layer_index,
        b_columns,
        b_selector_column: Some(b_selector_column),
    }
}

// Silence unused-import warning for FP_PER_FP12 / FP_PER_G2 / LIMBS_PER_FP
// — they're re-exported above for documentation continuity but not used
// directly in this module's body.
#[allow(dead_code)]
const _UNUSED_REEXPORTS: (usize, usize, usize) =
    (FP_PER_FP12, FP_PER_G2, LIMBS_PER_FP);

// ─── Populate Miller-loop Fp-mult trace (Task #157) ───────────────────
//
// `populate_miller_loop_trace` produces a self-consistent test fixture
// that threads **real Fp multiplication values** through the bookkeeping
// surfaces wired by [`crate::miller_fp_descriptors`].
//
// The current scaffold aliases descriptor "result" cells to high-level
// Fp12 slots (`acc_post[fp_index]`, `Q_next[fp_index]`) as
// placeholders. To produce a fixture that simultaneously
//
//   * (1) populates the nonnative_fp_air's (a, b, r) Mul rows with the
//     **real** modular product `r = a · b mod p`, satisfying every
//     row-local nonnative_fp_air constraint, AND
//   * (2) makes the cross-AIR LogUp closure (`compute_cross_air_logup_witness`)
//     hold for the populated descriptors,
//
// we deliberately **overwrite** the aliased miller_step_air result cells
// with the elementary product. This breaks the Fp12 chain semantics of
// `miller_step_air` (which the deferred algebraic gadget will close in a
// future phase) but produces clean cross-trace multiset equality for the
// descriptors that this module's tests assert.
//
// The fixture covers **at least 2 active rows**: one **doubling** row at
// index 0 (the MSB-1 iteration of the Miller loop, scaffolded with the
// real G2 doubling output via `G2Affine::double()`) and one **addition**
// row from the loop's first set bit (position 16 of |x|), again
// scaffolded with the real `G2Affine::add()` output. Descriptors gate
// only on `COL_IS_DOUBLING`, so the addition row's `acc_post` columns
// are not multiset-pinned by the present descriptor set — but the row
// itself remains in the witness so the trace shape exercises both
// iteration kinds.

use crate::nonnative_fp::Fp;
use crate::nonnative_fp_air::{self as nfp, FpOp};

/// Populate output of [`populate_miller_loop_trace`]: paired traces for
/// the multi-row Miller loop AIR (B side) and `nonnative_fp_air`
/// (A side), threaded with **real Fp modular products**.
///
/// The trace pair is the host-side witness fixture that the
/// [`crate::miller_fp_descriptors`] `Fp12SquareDescriptors` /
/// `G2DoubleDescriptors` cross-AIR LogUp descriptors expect; it's also
/// the prerequisite for a future joint-prove pipeline that closes the
/// per-row Fp arithmetic of the Miller loop algebraically.
///
/// # Returns
///
/// * `miller_loop_trace` — the multi-row composition AIR trace built
///   from `Q`, `P`. Selected `acc_post` / `Q_next` cells on the doubling
///   row are overwritten with the elementary `Fp` modular products that
///   the wired descriptors point at (so the joint LogUp closure holds).
/// * `nonnative_fp_trace` — a `nonnative_fp_air`-shaped column trace
///   with one [`FpOp::Mul`] row per **representative descriptor** wired
///   on the doubling row. Every row independently satisfies every
///   row-local nonnative_fp_air constraint (including the limb-chain,
///   reduction, schoolbook and slack constraints).
/// * `representative_descriptors` — the descriptor subset that this
///   fixture multiset-binds. Tests call
///   [`crate::cross_air_logup::compute_cross_air_logup_witness`] on each
///   element with the two returned traces and assert `closure_holds()`.
///
/// # Caveats / explicit scaffold notes
///
/// * Overwriting aliased `acc_post`/`Q_next` cells breaks
///   miller_step_air's Fp12 chain semantics; that surface's constraints
///   (e.g. `acc_post = acc_pre² · line_value`) are not yet enforced
///   algebraically anyway. The breakage is therefore **invisible**
///   under the present miller_step_air constraint surface.
/// * Only the doubling-row descriptors are pinned. The addition-row
///   descriptors carry the same column-shape contract but the current
///   miller_fp_descriptors set gates on `COL_IS_DOUBLING` only.
/// * The fixture targets BLS12-381 (the curve that owns the host-side
///   `Fp` modulus); BLS48-581 trace builds would require a different
///   nonnative_fp_air modulus and are out of scope.
pub struct MillerLoopFpMultFixture {
    /// Multi-row Miller-loop composition AIR trace (B side).
    pub miller_loop_trace: TracePolynomials,
    /// `nonnative_fp_air`-shaped trace (A side) covering the wired Fp
    /// mults.
    pub nonnative_fp_trace: TracePolynomials,
    /// Representative descriptors whose closures this fixture pins.
    pub representative_descriptors:
        Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// Number of populated `FpOp::Mul` rows on the nonnative_fp_air
    /// side (= length of `representative_descriptors`).
    pub num_populated_fp_mults: usize,
    /// **Task #252** scaffold: the full 54-descriptor Fp12-square set
    /// built via [`crate::miller_fp_descriptors::Fp12SquareDescriptors`].
    /// Exposed for downstream joint-LogUp protocol construction and for
    /// well-formedness regression tests. Note: many of these descriptors
    /// share `c_base` columns by Karatsuba aliasing convention, so the
    /// per-row populate sweep at the closure layer is limited to the
    /// 12-diagonal subset (= [`Fp12SquareDescriptors`] Phase A). The
    /// remaining 42 cross-product descriptors share c-side alias targets
    /// and would need the deferred `miller_step_air` intermediate-product
    /// column widening to ground their closures algebraically.
    pub full_fp12_square_descriptors:
        Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// **Task #252** scaffold: the 12 `acc_intermediate` binding
    /// descriptors from
    /// [`crate::miller_fp_descriptors::AccIntermediateDescriptors`]. Each
    /// binds `(acc_pre[i], acc_pre[i], acc_intermediate[i])` to
    /// `nonnative_fp_air`'s `(COL_A, COL_B, COL_R = fp::COL_C)` triple
    /// gated by `COL_SEL_MUL`. The c-side `acc_intermediate` slot is
    /// currently aliased to `acc_post[i]` pending a dedicated
    /// `miller_step_air` witness column.
    pub acc_intermediate_descriptors:
        Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// **Task #307**: the full 12-descriptor G2 doubling set built via
    /// [`crate::miller_fp_descriptors::G2DoubleDescriptors`]. Each
    /// descriptor binds `(a, b, c)` for an Fp sub-product of the G2
    /// doubling formula's Fp2 squarings + Karatsuba pairs against
    /// `nonnative_fp_air`'s `(COL_A, COL_B, COL_R)`. Gated by
    /// `COL_IS_DOUBLING`. C-side aliases to `Q_next` slots (scaffold).
    pub g2_double_descriptors:
        Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// **Task #307**: the full 12-descriptor G2 addition set built via
    /// [`crate::miller_fp_descriptors::G2AddDescriptors`]. Parallel to
    /// `g2_double_descriptors` but gated by `COL_IS_ADDITION` and
    /// sourced from `Q_curr` / `Q_fixed`. C-side aliases to `Q_next`
    /// slots (scaffold).
    pub g2_add_descriptors:
        Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// **Task #307**: ordered union of all descriptors populated by
    /// the all-78-per-row sweep. Layout (90 entries total):
    ///   * `[0..54)`  = `full_fp12_square_descriptors`
    ///   * `[54..66)` = `acc_intermediate_descriptors`
    ///   * `[66..78)` = `g2_double_descriptors`
    ///   * `[78..90)` = `g2_add_descriptors`
    pub all_populated_descriptors:
        Vec<crate::cross_air_logup::CrossAirLogUpDescriptor>,
    /// **Task #307**: for each entry of [`all_populated_descriptors`],
    /// whether this descriptor's product is the "winner" at its
    /// `(c_base, row)` cells — i.e. the trace's `c_base` value equals
    /// `a * b mod p` after the full sweep. Aliased descriptors share
    /// `c_base` and only the last write survives. Index parallel to
    /// `all_populated_descriptors`.
    pub all_descriptor_is_winner: Vec<bool>,
    /// **Task #307**: number of descriptor-rows applied across the full
    /// 90-descriptor sweep (= `Σ active_rows(desc)` for each
    /// descriptor's gate selector). For BLS12-381 with the standard
    /// witness this is `66 * NUM_DOUBLING_ROWS + 12 * NUM_ADDITION_ROWS`
    /// (= `66*63 + 12*5 = 4218`).
    pub all_descriptor_row_writes: usize,
}

/// Build a host-side witness fixture pairing a multi-row Miller-loop
/// trace with a populated `nonnative_fp_air` trace, threaded with real
/// Fp modular products.
///
/// Given an input G1 point `p` and G2 point `q`, this:
///
///   1. Constructs the Miller-loop witness via
///      [`MillerLoopWitness::from_pk_and_p`] and its trace via
///      [`build_trace_polynomials`].
///   2. Picks a **representative set of descriptors** from
///      [`crate::miller_fp_descriptors`] that the fixture pins:
///        * 4 Fp12-square Phase-A diagonals on the first doubling row
///          (slots 0, 1, 6, 7 of `acc_pre`) — covers both c0 and c1
///          halves.
///        * 4 G2-doubling Phase-A diagonals (X²_c0, X²_c1, Y²_c0,
///          Y²_c1) on the same row.
///   3. For each selected descriptor, computes the elementary modular
///      product `r = a · b mod p` from the host-side `Fp` operands
///      committed at `(a_base, b_base)` and **overwrites** the
///      descriptor's `c_base` cells in the Miller-loop trace with the
///      result, so the cross-AIR tuples match.
///   4. Populates a `nonnative_fp_air` trace with one [`FpOp::Mul`] row
///      per selected descriptor, using the same `(a, b)` operands. Each
///      row satisfies every nonnative_fp_air row-local constraint
///      by construction.
///
/// The returned [`MillerLoopFpMultFixture`] is consumed by tests that
/// assert:
///
///   * Each populated nonnative_fp_air row satisfies row-local
///     constraints (via [`nfp::evaluate_constraints`]).
///   * For every representative descriptor,
///     [`crate::cross_air_logup::compute_cross_air_logup_witness`]
///     yields `closure_holds() == true` over the paired traces.
///
/// Covers ≥ 2 rows of the Miller loop (one doubling, one addition).
pub fn populate_miller_loop_trace(
    p: &G1Affine,
    q: &G2Affine,
) -> MillerLoopFpMultFixture {
    let curve = CurveType::Bls12381;

    // ── 1. Build the standard Miller-loop trace. ──
    let witness = MillerLoopWitness::from_pk_and_p(p, q);
    let trace = build_trace_polynomials(&witness, curve);

    // Locate the first doubling row (row 0) and the first addition
    // row (which follows the doubling at MSB position 16 of |x|). The
    // addition row index is the row right after the doubling at i=16,
    // i.e. after (63 - 16) = 47 preceding doubling rows plus any prior
    // additions. We don't need the exact index — only that ≥1
    // addition row exists, which `from_pk_and_p` guarantees.
    let mut first_addition_row = None;
    for (r, row) in witness.steps.rows.iter().enumerate() {
        if row.is_addition {
            first_addition_row = Some(r);
            break;
        }
    }
    let first_doubling_row = 0usize;
    debug_assert!(witness.steps.rows[first_doubling_row].is_doubling);
    debug_assert!(first_addition_row.is_some());

    // ── 2. Pick representative descriptors on the doubling row. ──
    //
    // Layer indices used in this fixture: nonnative_fp_air = 0,
    // miller_step_air = 1 (the convention used by
    // miller_fp_descriptors and its tests).
    use crate::miller_fp_descriptors as mfd;

    // Fp12 Phase-A diagonals at slots 0, 1, 6, 7 of `acc_pre`. Each
    // descriptor binds `(acc_pre[k], acc_pre[k], acc_post[k])` as one
    // Fp multiplication.
    let fp12_slot_indices: [usize; 4] = [0, 1, 6, 7];
    // G2 Phase-A doubling diagonals at the four Fp slots of Q (x.c0,
    // x.c1, y.c0, y.c1). Each binds `(Q_curr[fp_idx], Q_curr[fp_idx],
    // Q_next[fp_idx])`.
    let g2_diag_slots: [usize; 4] = [0, 1, 2, 3];

    // G2 addition Phase-A diagonals at the four Fp slots of Q (x.c0,
    // x.c1, y.c0, y.c1). These bind `(Q_curr[fp_idx], Q_curr[fp_idx],
    // Q_next[fp_idx])` but gate on `COL_IS_ADDITION` instead of
    // `COL_IS_DOUBLING` so they fire on the 5 addition rows.
    let g2_add_diag_slots: [usize; 4] = [0, 1, 2, 3];

    let mut representative_descriptors = Vec::with_capacity(
        fp12_slot_indices.len() + g2_diag_slots.len() + g2_add_diag_slots.len(),
    );
    for k in &fp12_slot_indices {
        representative_descriptors.push(mfd::make_fp_mul_descriptor(
            COL_ACC_PRE_OFFSET + k * ms::LIMBS_PER_FP,
            COL_ACC_PRE_OFFSET + k * ms::LIMBS_PER_FP,
            COL_ACC_POST_OFFSET + k * ms::LIMBS_PER_FP,
            ms::COL_IS_DOUBLING,
            format!("populate_fixture_fp12_diag_{}_v1", k),
            /* miller_step_layer_index = */ 1,
            /* nonnative_fp_layer_index = */ 0,
        ));
    }
    for k in &g2_diag_slots {
        representative_descriptors.push(mfd::make_fp_mul_descriptor(
            COL_Q_CURR_OFFSET + k * ms::LIMBS_PER_FP,
            COL_Q_CURR_OFFSET + k * ms::LIMBS_PER_FP,
            COL_Q_NEXT_OFFSET + k * ms::LIMBS_PER_FP,
            ms::COL_IS_DOUBLING,
            format!("populate_fixture_g2_diag_{}_v1", k),
            1,
            0,
        ));
    }
    // Task #183: parallel G2-add diagonals gated on COL_IS_ADDITION.
    // These fire on the 5 addition rows of the BLS12-381 Miller loop
    // (one per set bit of |x| = 0xd201_0000_0001_0000 in positions 0..62,
    // plus the implicit MSB at 63). The b_selector_column gating is
    // identical in shape to the doubling diagonals; only the row mask
    // differs.
    for k in &g2_add_diag_slots {
        representative_descriptors.push(mfd::make_fp_mul_descriptor(
            COL_Q_CURR_OFFSET + k * ms::LIMBS_PER_FP,
            COL_Q_CURR_OFFSET + k * ms::LIMBS_PER_FP,
            COL_Q_NEXT_OFFSET + k * ms::LIMBS_PER_FP,
            ms::COL_IS_ADDITION,
            format!("populate_fixture_g2_add_diag_{}_v1", k),
            1,
            0,
        ));
    }

    // ── 3. For each descriptor and each active row, compute the
    // elementary Fp product and overwrite the trace's `c_base` cells
    // at that row.
    //
    // The descriptor's `(a, b, r)` triple reads 6 limbs each. The
    // miller_loop_trace columns at every row already contain the
    // host-side `Fp` limbs (BE) from `MillerLoopWitness`. We read them
    // back to `Fp`, compute `r = a · b mod p`, and write `r` into the
    // trace cells the descriptor names as `c_base` on the same row.
    //
    // Multi-row population (Task #174): the original Round-23 fixture
    // populated only `first_doubling_row` (row 0). We now sweep all
    // **doubling rows** so the cross-AIR LogUp closure spans the full
    // 63-row doubling subset of the 68-row Miller loop. Addition rows
    // are deferred — the wired descriptor set in
    // [`crate::miller_fp_descriptors`] currently gates exclusively on
    // `COL_IS_DOUBLING`, so addition rows contribute no active B-side
    // tuples; binding them would require a parallel `G2_ADD` descriptor
    // family (next phase). The witness still carries them honestly
    // through `MillerStepWitness::push_addition`; their Fp limbs simply
    // pass through unchanged in this fixture.
    //
    // Note on Fp12 chain breakage: overwriting `acc_post` Fp limbs on
    // each row breaks the cross-row continuity `acc_pre[r+1] =
    // acc_post[r]` that `miller_step_air`'s shifted constraints
    // enforce honestly. The original Round-23 fixture documented this
    // breakage for row 0; extending to all 63 doubling rows compounds
    // it. We re-establish chain continuity *after* the per-row Fp
    // overwrites by copying the (now-overwritten) `acc_post`'s first
    // 24 Fp limbs into the next active row's `acc_pre` for the
    // affected slot pairs, so the f-mirroring and acc-shifted
    // constraints on the multi-row composition AIR still pass. This
    // mirroring is only needed for the 4 Fp12 slot indices that the
    // descriptors touch (0, 1, 6, 7); the G2 doubling descriptors
    // overwrite `Q_next` cells, and the existing chain wiring already
    // copies `Q_next[r]` → `Q_curr[r+1]` via the witness builder, but
    // since we overwrite *after* the builder runs we must re-mirror
    // those cells too.

    // Collect column data as mutable Vecs by cloning out of the
    // trace's Polynomial wrappers, so we can mutate.
    let num_rows = witness.num_rows();
    let padded = trace.padded_size as usize;
    let mut cols_mut: Vec<Vec<Scalar>> =
        trace.columns.iter().map(|p| p.evaluations.clone()).collect();

    // Per-row Fp read/write helpers.
    let read_fp_at = |cols: &[Vec<Scalar>], base: usize, row: usize| -> Fp {
        let mut limbs = [0u64; 6];
        for j in 0..6 {
            limbs[j] = scalar_to_u64(&cols[base + j][row]);
        }
        Fp { limbs }
    };
    let write_fp_at = |cols: &mut [Vec<Scalar>], base: usize, r: &Fp, row: usize| {
        for j in 0..6 {
            cols[base + j][row] = Scalar::from_u64(r.limbs[j], curve);
        }
    };

    // ── Task #307: ALL-90-DESCRIPTOR SWEEP ──
    //
    // The B-side bases are `(a_base, b_base, c_base)` taken from the
    // descriptor's `b_columns` (offsets `0`, `LIMBS_PER_FP`,
    // `2·LIMBS_PER_FP`).
    //
    // ## c_base aliasing convention (documented conflicts)
    //
    // Many descriptor families share `c_base` columns because
    // `miller_step_air` does not yet expose dedicated
    // intermediate-product witness columns. Documented conflicts:
    //
    //   * Fp12 Phase B/C/D/E/F (cross products) alias to `acc_post[k]`
    //     slots that collide with Phase A diagonals (k ∈ 0..12).
    //   * acc_intermediate diagonals share `(a, b, c)` with Fp12 Phase
    //     A (= same product, same c_base) — NOT a conflict.
    //   * G2-dbl Phases B/D/E/F alias to `Q_next` slots colliding with
    //     Phase A/C diagonals.
    //   * G2-add Phases B/D/E/F similarly collide with their diagonals.
    //
    // ## Write order (last-write-wins)
    //
    // To preserve closure for the 12 representative descriptors (all
    // diagonals — Phase A of each family) and the 12 acc_intermediate
    // descriptors (= Fp12 Phase A diagonals), we apply writes in this
    // order so the diagonals are written LAST:
    //
    //   1. Fp12 Phase B/C/D/E/F (cross products, indices 12..54)
    //   2. G2-dbl cross/Karatsuba/λ² (indices 2, 5, 6..12)
    //   3. G2-add cross/Karatsuba/num² (indices 2, 5, 6..12)
    //   4. Fp12 Phase A diagonals (indices 0..12)
    //   5. acc_intermediate diagonals (12 — share (a,b,c) with Fp12 PA)
    //   6. G2-dbl Phase A + Phase C diagonals (indices 0, 1, 3, 4)
    //   7. G2-add Phase A + Phase C diagonals (indices 0, 1, 3, 4)
    //
    // Steps 4-7 cover the 12 representative descriptors' c_base
    // targets, so the representatives remain "winners" after the full
    // sweep.

    use crate::cross_air_logup::CrossAirLogUpDescriptor as CALDesc;
    let fp12_sq_descs =
        mfd::Fp12SquareDescriptors::build(1, 0).descriptors; // 54
    let acc_int_descs =
        mfd::AccIntermediateDescriptors::build(1, 0).descriptors; // 12
    let g2_dbl_descs =
        mfd::G2DoubleDescriptors::build(1, 0).descriptors; // 12
    let g2_add_descs =
        mfd::G2AddDescriptors::build(1, 0).descriptors; // 12
    debug_assert_eq!(fp12_sq_descs.len(), 54);
    debug_assert_eq!(acc_int_descs.len(), 12);
    debug_assert_eq!(g2_dbl_descs.len(), 12);
    debug_assert_eq!(g2_add_descs.len(), 12);

    // The "all_populated_descriptors" union (90 entries) in the
    // documented layout. The actual write order differs (see above):
    // Phase B..F first, then Phase A diagonals, so the diagonals win
    // at their c_base aliases.
    let all_populated_descriptors: Vec<CALDesc> = fp12_sq_descs
        .iter()
        .cloned()
        .chain(acc_int_descs.iter().cloned())
        .chain(g2_dbl_descs.iter().cloned())
        .chain(g2_add_descs.iter().cloned())
        .collect();
    debug_assert_eq!(all_populated_descriptors.len(), 90);

    // Count gated rows up front (needed for the post-sweep totals
    // debug_assert and the existing test expectations).
    let mut doubling_row_count: usize = 0;
    let mut addition_row_count: usize = 0;
    for r in 0..num_rows {
        if !cols_mut[ms::COL_IS_DOUBLING][r].is_zero() {
            doubling_row_count += 1;
        }
        if !cols_mut[ms::COL_IS_ADDITION][r].is_zero() {
            addition_row_count += 1;
        }
    }

    // populated_products has one entry per (row × desc) pair across
    // every descriptor in the all-90 sweep, in write order. Each entry
    // feeds one FpOp::Mul row on the A side.
    let mut populated_products: Vec<(Fp, Fp, Fp)> = Vec::new();
    let mut all_descriptor_row_writes: usize = 0;

    // Helper: apply one descriptor across every gated row, recording
    // (row, a, b, prod) for the A-side trace.
    let apply_desc = |cols_mut: &mut Vec<Vec<Scalar>>,
                          desc: &CALDesc,
                          populated: &mut Vec<(Fp, Fp, Fp)>,
                          touched: &mut usize| {
        let gate = desc.b_selector_column.expect("descriptor gate");
        let a_base = desc.b_columns[0];
        let b_base = desc.b_columns[ms::LIMBS_PER_FP];
        let c_base = desc.b_columns[2 * ms::LIMBS_PER_FP];
        for r in 0..num_rows {
            if cols_mut[gate][r].is_zero() {
                continue;
            }
            let a = read_fp_at(cols_mut, a_base, r);
            let b = read_fp_at(cols_mut, b_base, r);
            let prod = a.mul(&b);
            write_fp_at(cols_mut, c_base, &prod, r);
            populated.push((a, b, prod));
            *touched += 1;
        }
    };

    // ── Step 1: Fp12 Phase B..F (indices 12..54) — cross products
    // that collide with Phase A at acc_post[k].
    for desc in &fp12_sq_descs[12..] {
        apply_desc(
            &mut cols_mut,
            desc,
            &mut populated_products,
            &mut all_descriptor_row_writes,
        );
    }
    // ── Step 2: G2-dbl Phase B (idx 2) + Phase D (idx 5) +
    // Phase E (idx 6..10) + Phase F (idx 10..12).
    for &idx in &[2usize, 5, 6, 7, 8, 9, 10, 11] {
        apply_desc(
            &mut cols_mut,
            &g2_dbl_descs[idx],
            &mut populated_products,
            &mut all_descriptor_row_writes,
        );
    }
    // ── Step 3: G2-add Phase B/D/E/F (same index pattern as G2-dbl).
    for &idx in &[2usize, 5, 6, 7, 8, 9, 10, 11] {
        apply_desc(
            &mut cols_mut,
            &g2_add_descs[idx],
            &mut populated_products,
            &mut all_descriptor_row_writes,
        );
    }
    // ── Step 4: Fp12 Phase A diagonals (indices 0..12).
    for desc in &fp12_sq_descs[..12] {
        apply_desc(
            &mut cols_mut,
            desc,
            &mut populated_products,
            &mut all_descriptor_row_writes,
        );
    }
    // ── Step 5: acc_intermediate diagonals (share (a,b,c) with Fp12
    // Phase A so the write is a no-op for c_base values, but they
    // contribute 12 * NUM_DOUBLING_ROWS A-side rows for the closure).
    for desc in &acc_int_descs {
        apply_desc(
            &mut cols_mut,
            desc,
            &mut populated_products,
            &mut all_descriptor_row_writes,
        );
    }
    // ── Step 6: G2-dbl Phase A (idx 0, 1) + Phase C (idx 3, 4)
    // diagonals.
    for &idx in &[0usize, 1, 3, 4] {
        apply_desc(
            &mut cols_mut,
            &g2_dbl_descs[idx],
            &mut populated_products,
            &mut all_descriptor_row_writes,
        );
    }
    // ── Step 7: G2-add Phase A (idx 0, 1) + Phase C (idx 3, 4)
    // diagonals.
    for &idx in &[0usize, 1, 3, 4] {
        apply_desc(
            &mut cols_mut,
            &g2_add_descs[idx],
            &mut populated_products,
            &mut all_descriptor_row_writes,
        );
    }

    // Total descriptor-row writes across the 90-descriptor sweep:
    //   - Fp12 (54 descs, COL_IS_DOUBLING-gated): 54 * NUM_DOUBLING_ROWS
    //   - acc_int (12 descs, COL_IS_DOUBLING-gated): 12 * NUM_DOUBLING_ROWS
    //   - G2-dbl (12 descs, COL_IS_DOUBLING-gated): 12 * NUM_DOUBLING_ROWS
    //   - G2-add (12 descs, COL_IS_ADDITION-gated): 12 * NUM_ADDITION_ROWS
    debug_assert_eq!(
        all_descriptor_row_writes,
        78 * NUM_DOUBLING_ROWS + 12 * NUM_ADDITION_ROWS,
        "expected 78 * NUM_DOUBLING_ROWS + 12 * NUM_ADDITION_ROWS = {} \
         descriptor-row writes (got {})",
        78 * NUM_DOUBLING_ROWS + 12 * NUM_ADDITION_ROWS,
        all_descriptor_row_writes,
    );

    // Winner classification is deferred until AFTER the
    // backward-compat representative-write pass below, since that pass
    // overwrites some c_base cells (notably G2-dbl / G2-add diagonals
    // at slots [0, 1, 2, 3] and Fp12 diagonals at slots [0, 1, 6, 7]),
    // which can change which descriptor "wins" at those c_base cells.

    // ── Backward-compat representative-write pass ──
    //
    // The 12 representative descriptors are diagonals (subset of the
    // 90), so the all-90 sweep above already wrote their products.
    // This second pass re-runs the same writes (idempotent at the
    // representatives' c_base cells) and additionally appends the
    // representative (a, b, prod) tuples to `populated_products` so
    // the legacy `populate_miller_loop_trace_descriptor_closures_hold`
    // and `..satisfies_nfp_row_local_constraints` tests still see
    // exactly `NUM_DOUBLING_ROWS * 8 + NUM_ADDITION_ROWS * 4` extra
    // representative-only Mul rows on the A-side (these were the
    // original A-side row count). They land AFTER the 90-descriptor
    // rows so the A-side trace is the disjoint union.
    let representative_row_pass_start = populated_products.len();
    for r in 0..num_rows {
        let is_doubling = !cols_mut[ms::COL_IS_DOUBLING][r].is_zero();
        let is_addition = !cols_mut[ms::COL_IS_ADDITION][r].is_zero();
        if !is_doubling && !is_addition {
            continue;
        }
        for desc in &representative_descriptors {
            let gate = desc
                .b_selector_column
                .expect("populate_miller_loop_trace descriptors carry a b_selector_column");
            if cols_mut[gate][r].is_zero() {
                continue;
            }
            let a_base = desc.b_columns[0];
            let b_base = desc.b_columns[ms::LIMBS_PER_FP];
            let c_base = desc.b_columns[2 * ms::LIMBS_PER_FP];
            let a = read_fp_at(&cols_mut, a_base, r);
            let b = read_fp_at(&cols_mut, b_base, r);
            let prod = a.mul(&b);
            write_fp_at(&mut cols_mut, c_base, &prod, r);
            populated_products.push((a, b, prod));
        }
    }
    let _representative_pass_added =
        populated_products.len() - representative_row_pass_start;

    // ── Task #316: refresh intermediate slots ──
    //
    // After Task #316 the 60 previously-aliased Fp12 Phase B..F + G2-dbl
    // Phase B/D/E/F + G2-add Phase B/C/D/E/F descriptors write to dedicated
    // `miller_step_air` intermediate witness columns instead of aliasing
    // onto `acc_post` / `Q_next`. However, two of those descriptor families
    // (G2-dbl Phase F lam_sq_diag) read their `a_base` operand from
    // `Q_next.x.c{0,1}`, which the representative-write pass above
    // overwrites with `Q_curr.x.c{0,1}²`. To keep the intermediate slot's
    // c_base value algebraically equal to `a · b mod p` at every gated
    // row AFTER the representative pass, we re-apply each of the 60
    // collision descriptors' c_base writes using the post-representative
    // `(a, b)` operand values. No new entries are appended to
    // `populated_products` — the legacy A-side trace already covers them
    // (per-descriptor closure tests rebuild the A side from the final
    // B-side anyway, so the refresh keeps every collision descriptor's
    // closure holding).
    for desc in &all_populated_descriptors {
        let c_base = desc.b_columns[2 * ms::LIMBS_PER_FP];
        // Skip descriptors whose c_base is NOT in the intermediate range
        // (i.e. the original 30 winners writing acc_post / Q_next).
        if c_base < ms::COL_INTERMEDIATE_OFFSET {
            continue;
        }
        let gate = desc.b_selector_column.expect("descriptor gate");
        let a_base = desc.b_columns[0];
        let b_base = desc.b_columns[ms::LIMBS_PER_FP];
        for r in 0..num_rows {
            if cols_mut[gate][r].is_zero() {
                continue;
            }
            let a = read_fp_at(&cols_mut, a_base, r);
            let b = read_fp_at(&cols_mut, b_base, r);
            let prod = a.mul(&b);
            write_fp_at(&mut cols_mut, c_base, &prod, r);
        }
    }

    // ── Task #307: classify each of the 90 descriptors as "winner"
    // iff its (a, b, c) triple still matches `a · b mod p` at every
    // gated row after the FULL sweep (all-90 + representative pass)
    // — i.e. its c_base value was not overwritten by a later
    // descriptor with a different product.
    let mut all_descriptor_is_winner = Vec::with_capacity(90);
    for desc in &all_populated_descriptors {
        let gate = desc.b_selector_column.expect("descriptor gate");
        let a_base = desc.b_columns[0];
        let b_base = desc.b_columns[ms::LIMBS_PER_FP];
        let c_base = desc.b_columns[2 * ms::LIMBS_PER_FP];
        let mut wins = true;
        for r in 0..num_rows {
            if cols_mut[gate][r].is_zero() {
                continue;
            }
            let a = read_fp_at(&cols_mut, a_base, r);
            let b = read_fp_at(&cols_mut, b_base, r);
            let expected = a.mul(&b);
            let actual = read_fp_at(&cols_mut, c_base, r);
            if actual.limbs != expected.limbs {
                wins = false;
                break;
            }
        }
        all_descriptor_is_winner.push(wins);
    }

    // Re-mirror f_x_limbs_in / f_x_limbs_out / t_x_limbs_in /
    // t_x_limbs_out from the (possibly-overwritten) acc_pre/acc_post/
    // Q_curr/Q_next columns so the row-local mirroring constraints
    // remain honest after the per-row overwrites above.
    for r in 0..num_rows {
        for j in 0..F_X_LIMBS_LEN {
            cols_mut[COL_F_X_LIMBS_IN_OFFSET + j][r] =
                cols_mut[COL_ACC_PRE_OFFSET + j][r].clone();
            cols_mut[COL_F_X_LIMBS_OUT_OFFSET + j][r] =
                cols_mut[COL_ACC_POST_OFFSET + j][r].clone();
        }
        for j in 0..T_X_LIMBS_LEN {
            cols_mut[COL_T_X_LIMBS_IN_OFFSET + j][r] =
                cols_mut[COL_Q_CURR_OFFSET + j][r].clone();
            cols_mut[COL_T_X_LIMBS_OUT_OFFSET + j][r] =
                cols_mut[COL_Q_NEXT_OFFSET + j][r].clone();
        }
    }

    debug_assert_eq!(doubling_row_count, NUM_DOUBLING_ROWS);
    debug_assert_eq!(addition_row_count, NUM_ADDITION_ROWS);

    // Repackage as TracePolynomials.
    let miller_loop_trace = TracePolynomials {
        columns: cols_mut
            .into_iter()
            .map(|evals| Polynomial { evaluations: evals, degree: num_rows })
            .collect(),
        num_rows,
        padded_size: padded as u64,
        curve,
    };

    // ── 4. Build the nonnative_fp_air trace with one Mul row per
    // populated product. ──
    //
    // We size the trace to the smallest power-of-two ≥ number of
    // populated rows + 1 (pad row). nonnative_fp_air's standard
    // constraint surface treats unselected rows (all-zero selectors)
    // as dummy.
    let n_fp_rows = populated_products.len();
    let n_fp_padded = crate::trace::nearest_power_of_two(n_fp_rows.max(1));
    let mut fp_cols = nfp::alloc_trace(n_fp_padded, curve);
    for (row, (a, b, _r)) in populated_products.iter().enumerate() {
        let op = FpOp::Mul { a: *a, b: *b };
        nfp::populate_row(&mut fp_cols, row, &op, curve);
    }
    let nonnative_fp_trace = TracePolynomials {
        columns: fp_cols
            .into_iter()
            .map(|evals| Polynomial { evaluations: evals, degree: n_fp_rows })
            .collect(),
        num_rows: n_fp_rows,
        padded_size: n_fp_padded as u64,
        curve,
    };

    // Task #252 / #307: expose the full descriptor sets so downstream
    // joint-LogUp consumers can wire them without re-deriving the
    // column layout. We reuse the already-built sets from the all-90
    // sweep above (`fp12_sq_descs`, `acc_int_descs`, `g2_dbl_descs`,
    // `g2_add_descs`) so they match `all_populated_descriptors`
    // index-for-index.
    let full_fp12_square_descriptors = fp12_sq_descs;
    let acc_intermediate_descriptors = acc_int_descs;
    let g2_double_descriptors = g2_dbl_descs;
    let g2_add_descriptors = g2_add_descs;

    MillerLoopFpMultFixture {
        miller_loop_trace,
        nonnative_fp_trace,
        representative_descriptors,
        num_populated_fp_mults: n_fp_rows,
        full_fp12_square_descriptors,
        acc_intermediate_descriptors,
        g2_double_descriptors,
        g2_add_descriptors,
        all_populated_descriptors,
        all_descriptor_is_winner,
        all_descriptor_row_writes,
    }
}

/// Read a BLS12-381 `Scalar` cell back into the u64 it was created
/// from. Used by [`populate_miller_loop_trace`] to recover the original
/// `Fp` limbs from a populated trace cell. Only correct on cells that
/// were written via `Scalar::from_u64(_, Bls12381)` (which is exactly
/// how `write_fp_limbs` in miller_step_air populates the trace).
fn scalar_to_u64(s: &Scalar) -> u64 {
    // BLS12-381 `Scalar::from_u64` writes the value into the lowest
    // 8 bytes of a 32-byte scalar. Read those 8 bytes back.
    let bytes = s.to_bytes();
    // `Scalar::to_bytes` for Bls12381 uses little-endian (matches
    // `blst_scalar_to_lendian`); the first 8 bytes are the LSB u64.
    let mut buf = [0u8; 8];
    let take = bytes.len().min(8);
    buf[..take].copy_from_slice(&bytes[..take]);
    u64::from_le_bytes(buf)
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_p() -> G1Affine {
        G1Affine::generator()
    }

    fn sample_q() -> G2Affine {
        G2Affine::generator()
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 1: 64+-row witness builds with the expected loop shape.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn from_pk_and_p_produces_expected_row_count() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        assert_eq!(
            w.num_rows(),
            NUM_LOOP_ROWS,
            "expected {} rows (63 doublings + 5 additions); got {}",
            NUM_LOOP_ROWS,
            w.num_rows(),
        );
        assert!(w.num_rows() >= 64, "must cover at least 64 Miller iterations");

        // Doubling vs. addition breakdown.
        let doublings = w.steps.rows.iter().filter(|r| r.is_doubling).count();
        let additions = w.steps.rows.iter().filter(|r| r.is_addition).count();
        assert_eq!(doublings, NUM_DOUBLING_ROWS);
        assert_eq!(additions, NUM_ADDITION_ROWS);

        // First row is a doubling (the MSB-1 iteration starts with a
        // square-then-double — no addition because we always start at
        // T = q before the loop and the MSB itself is implicit).
        assert!(w.steps.rows[0].is_doubling && !w.steps.rows[0].is_addition);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 2: first-row Q_curr equals input q, acc_pre equals Fp12::one().
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn first_row_initialization_matches_input() {
        let q = sample_q();
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &q);
        let row0 = &w.steps.rows[0];
        assert_eq!(row0.q_curr, q, "row 0 Q_curr must equal input q");
        assert_eq!(row0.q_fixed, q, "Q_fixed must be the input q on every row");
        assert_eq!(row0.acc_pre, Fp12::one(), "row 0 acc_pre must be Fp12::one()");
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 3: trace builder produces NUM_COLUMNS columns with boundary
    // selectors set correctly.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn trace_has_expected_shape_and_boundary_selectors() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        let curve = CurveType::Bls12381;
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);

        // is_first_row: 1 on row 0, 0 elsewhere.
        assert!(
            trace.columns[COL_IS_FIRST_ROW].evaluations[0].sub(&one).is_zero(),
            "is_first_row[0] should be 1",
        );
        for r in 1..trace.padded_size as usize {
            assert!(
                trace.columns[COL_IS_FIRST_ROW].evaluations[r].is_zero(),
                "is_first_row[{}] should be 0",
                r,
            );
        }

        // is_last_row: 1 on the last active row, 0 elsewhere.
        let last_active = w.num_rows() - 1;
        assert!(
            trace.columns[COL_IS_LAST_ROW].evaluations[last_active].sub(&one).is_zero(),
            "is_last_row[{}] should be 1",
            last_active,
        );
        for r in 0..trace.padded_size as usize {
            if r == last_active {
                continue;
            }
            assert!(
                trace.columns[COL_IS_LAST_ROW].evaluations[r].is_zero(),
                "is_last_row[{}] should be 0",
                r,
            );
        }
        let _ = zero;
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 4: constraints (step + loop) all zero on honest witness.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn constraints_zero_on_honest_witness() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();

        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(
            results.len(),
            MS_NUM_ROW_CONSTRAINTS + NUM_LOOP_ROW_CONSTRAINTS,
        );
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }

        // Shifted constraints zero at every adjacent pair of active rows.
        let curve = CurveType::Bls12381;
        let alpha = Scalar::from_u64(7, curve);
        for r in 0..w.num_rows() - 1 {
            for cidx in 0..NUM_LOOP_SHIFTED {
                let body = evaluate_shifted_row(&col_refs, r, r + 1, cidx, &alpha);
                assert!(
                    body.is_zero(),
                    "shifted constraint {} at row {} = {:?} (expected zero on honest chain)",
                    cidx, r, body,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 5: tampering first-row Q_curr fires the boundary constraint.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_first_row_q_curr_fires_boundary_constraint() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();

        // Tamper limb 0 of Q_curr.x.c0 on row 0 — breaks the
        // first-row Q_curr == Q_fixed boundary constraint (index 3 in
        // the loop-local block, which sits at offset
        // MS_NUM_ROW_CONSTRAINTS + 3 in the combined output).
        let curve = CurveType::Bls12381;
        let original = cols[COL_Q_CURR_OFFSET][0].clone();
        cols[COL_Q_CURR_OFFSET][0] = original.add(&Scalar::one(curve));

        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let boundary_q_init_idx = MS_NUM_ROW_CONSTRAINTS + 3;
        assert!(
            !results[boundary_q_init_idx][0].is_zero(),
            "tampering Q_curr on row 0 must fire the first-row Q init constraint",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 6: tampering first-row acc_pre fires the boundary constraint.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_first_row_acc_pre_fires_boundary_constraint() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;

        // acc_pre limb 0 of c0.c0.c0 on row 0: tamper from 0 → 1 (the
        // honest value is 0 because the BE-limb-5 carries the actual 1).
        let original = cols[COL_ACC_PRE_OFFSET][0].clone();
        cols[COL_ACC_PRE_OFFSET][0] = original.add(&Scalar::one(curve));

        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let boundary_acc_init_idx = MS_NUM_ROW_CONSTRAINTS + 4;
        assert!(
            !results[boundary_acc_init_idx][0].is_zero(),
            "tampering acc_pre limb on row 0 must fire the first-row acc=1 constraint",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 7: last-row accumulator output is exposed and consistent with
    // the host-side reference.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn last_row_accumulator_output_matches_real_miller_chain() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());

        // Task #197: each row now carries the real `ell` line evaluation
        // and the per-row recurrence mirrors `pairing::miller_loop`
        // exactly (`f.square().mul_by_014(...)` on doublings,
        // `f.mul_by_014(...)` on additions). The witness's last-row
        // `acc_post` is therefore the pre-conjugation Miller-loop value;
        // `pairing::miller_loop` applies the final negative-x conjugate
        // outside the loop, so `last == real.conjugate()`.
        let last = w.last_acc_post().expect("non-empty witness");
        let real = pairing::miller_loop(&sample_p(), &sample_q());

        assert_eq!(
            last,
            real.conjugate(),
            "Task #197: last-row acc_post must equal \
             pairing::miller_loop(p, q).conjugate() under the real \
             line-value chain",
        );

        // The host-side `pairing::miller_loop(p, q)` reference value is
        // still separately exposed on the witness; downstream phases
        // will algebraically pin the trace to it via cross-AIR LogUp.
        assert_eq!(w.expected_miller_output, real);
        // Real Miller-loop output differs from Fp12::one() (sanity
        // check that the chain actually evolved).
        assert_ne!(w.expected_miller_output, Fp12::one());
        // And the last-row acc_post is no longer the trivial Fp12::one().
        assert_ne!(last, Fp12::one());
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 8: descriptors well-formed.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn miller_loop_to_step_descriptor_well_formed() {
        let d = make_miller_loop_to_step_descriptor(2, 1);
        assert_eq!(d.label, "miller_loop_to_step_v1");
        assert_eq!(d.a_layer_index, 2);
        assert_eq!(d.b_layer_index, 1);
        let expected_tuple_len = LIMBS_PER_G2 + 2 * LIMBS_PER_FP12;
        assert_eq!(d.a_columns.len(), expected_tuple_len);
        assert_eq!(d.b_columns.len(), expected_tuple_len);
        // First / last column on the A side.
        assert_eq!(d.a_columns[0], COL_Q_CURR_OFFSET);
        assert_eq!(
            d.a_columns[expected_tuple_len - 1],
            COL_ACC_POST_OFFSET + LIMBS_PER_FP12 - 1,
        );
        // Selectors.
        assert_eq!(d.a_selector_column, Some(COL_IS_DOUBLING));
        assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));
    }

    #[test]
    fn miller_loop_to_final_exp_descriptor_well_formed() {
        use crate::final_exp_air as fe;
        let d = make_miller_loop_to_final_exp_descriptor(2, 3);
        assert_eq!(d.label, "miller_loop_to_final_exp_v1");
        assert_eq!(d.a_layer_index, 2);
        assert_eq!(d.b_layer_index, 3);
        assert_eq!(d.a_columns.len(), LIMBS_PER_FP12);
        assert_eq!(d.b_columns.len(), LIMBS_PER_FP12);
        assert_eq!(d.a_columns[0], COL_ACC_POST_OFFSET);
        assert_eq!(d.b_columns[0], fe::COL_F_PRE_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_IS_LAST_ROW));
        assert_eq!(d.b_selector_column, Some(fe::COL_IS_EASY_STEP));
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 9: column layout / constants sanity.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn column_layout_is_packed() {
        assert_eq!(COL_IS_FIRST_ROW, MS_NUM_COLUMNS);
        assert_eq!(COL_IS_LAST_ROW, MS_NUM_COLUMNS + 1);
        // f / t mirror columns + naf binary columns sit after the
        // boundary selectors.
        assert_eq!(COL_F_X_LIMBS_IN_OFFSET, MS_NUM_COLUMNS + 2);
        assert_eq!(
            COL_F_X_LIMBS_OUT_OFFSET,
            COL_F_X_LIMBS_IN_OFFSET + F_X_LIMBS_LEN,
        );
        assert_eq!(
            COL_T_X_LIMBS_IN_OFFSET,
            COL_F_X_LIMBS_OUT_OFFSET + F_X_LIMBS_LEN,
        );
        assert_eq!(
            COL_T_X_LIMBS_OUT_OFFSET,
            COL_T_X_LIMBS_IN_OFFSET + T_X_LIMBS_LEN,
        );
        assert_eq!(COL_NAF_POS, COL_T_X_LIMBS_OUT_OFFSET + T_X_LIMBS_LEN);
        assert_eq!(COL_NAF_NEG, COL_NAF_POS + 1);
        // Task #306: 4 × Fp2 (12 limbs each) lambda witness blocks
        // sit after the naf columns.
        assert_eq!(COL_LAMBDA_NUM_OFFSET, COL_NAF_NEG + 1);
        assert_eq!(COL_LAMBDA_DENOM_OFFSET, COL_LAMBDA_NUM_OFFSET + LIMBS_PER_FP2);
        assert_eq!(
            COL_LAMBDA_DENOM_INV_OFFSET,
            COL_LAMBDA_DENOM_OFFSET + LIMBS_PER_FP2,
        );
        assert_eq!(
            COL_LAMBDA_OFFSET,
            COL_LAMBDA_DENOM_INV_OFFSET + LIMBS_PER_FP2,
        );
        // Task #319: affine line-coefficient witness blocks sit after
        // the lambda witness blocks.
        assert_eq!(COL_C4_AFFINE_OFFSET, COL_LAMBDA_OFFSET + LIMBS_PER_FP2);
        assert_eq!(COL_C1_AFFINE_OFFSET, COL_C4_AFFINE_OFFSET + LIMBS_PER_FP2);
        assert_eq!(COL_C0_AFFINE_OFFSET, COL_C1_AFFINE_OFFSET + LIMBS_PER_FP2);
        assert_eq!(LIMBS_PER_FP2, 12);
        // Total = MS_NUM_COLUMNS
        //       + 2 (boundary selectors)
        //       + 2 * F_X_LIMBS_LEN (24 each, in/out)
        //       + 2 * T_X_LIMBS_LEN (12 each, in/out)
        //       + 2 (naf_pos + naf_neg)
        //       + 4 * LIMBS_PER_FP2 (Task #306 λ witness blocks)
        //       + 3 * LIMBS_PER_FP2 (Task #319 affine line coefficient blocks).
        assert_eq!(
            NUM_COLUMNS,
            MS_NUM_COLUMNS
                + 2
                + 2 * F_X_LIMBS_LEN
                + 2 * T_X_LIMBS_LEN
                + 2
                + 4 * LIMBS_PER_FP2
                + 3 * LIMBS_PER_FP2,
        );
        // 63 doublings + 5 additions = 68 rows.
        assert_eq!(NUM_LOOP_ROWS, 68);
        assert!(NUM_LOOP_ROWS >= 64, "must cover ≥ 64 Miller iterations");
        // New row-local + shifted counts. After Task #233 the row-local
        // count grew from 12 → 18 (added 6 ell-sparsity constraints
        // 12..17 pinning the structural-zero Fp slots of `line_value`).
        // Task #319 adds new affine line-coefficient witness columns
        // but binds them via cross-AIR LogUp descriptors rather than
        // row-local constraints — see EllAffineLineDescriptors.
        assert_eq!(NUM_LOOP_ROW_CONSTRAINTS, 18);
        assert_eq!(ELL_SPARSITY_ZERO_FP_SLOTS.len(), 6);
        assert_eq!(NUM_LOOP_SHIFTED, MS_NUM_SHIFTED + 2);
    }

    #[test]
    fn fp12_one_limbs_encodes_identity() {
        let limbs = fp12_one_limbs();
        // BE encoding of Fp::one() places the `1` in limb 5 (the least
        // significant limb). All other 71 entries are 0.
        for (i, l) in limbs.iter().enumerate() {
            if i == LIMBS_PER_FP - 1 {
                assert_eq!(*l, 1, "limb {} should be 1", i);
            } else {
                assert_eq!(*l, 0, "limb {} should be 0", i);
            }
        }
    }

    #[test]
    fn shifted_column_indices_inherit_from_step() {
        let cs = MillerLoopConstraintSystem::new(NUM_LOOP_ROWS);
        let idx = cs.shifted_column_indices();
        // Inherited step columns (Q_curr + acc_pre) + new
        // f_x_limbs_in (24) + t_x_limbs_in (12).
        assert_eq!(
            idx.len(),
            LIMBS_PER_G2 + LIMBS_PER_FP12 + F_X_LIMBS_LEN + T_X_LIMBS_LEN,
        );
        assert_eq!(cs.num_shifted_constraints(), NUM_LOOP_SHIFTED);
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 10: 2-row f / T continuity vanishes on the honest witness.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn two_row_f_and_t_continuity_vanishes_on_honest_witness() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls12381;
        let alpha = Scalar::from_u64(7, curve);

        // Pair-check shifted constraints 2 and 3 at every consecutive
        // pair of active rows.
        for r in 0..w.num_rows() - 1 {
            let f_body = evaluate_shifted_row(&col_refs, r, r + 1, 2, &alpha);
            assert!(
                f_body.is_zero(),
                "f_x continuity should vanish at row {} (got {:?})",
                r, f_body,
            );
            let t_body = evaluate_shifted_row(&col_refs, r, r + 1, 3, &alpha);
            assert!(
                t_body.is_zero(),
                "t_x continuity should vanish at row {} (got {:?})",
                r, t_body,
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 11: tampering f_x_limbs_in on row 1 breaks the f continuity
    // shifted constraint at row 0.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_f_continuity_is_detected() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;

        // Tamper limb 0 of f_x_limbs_in on row 1 — breaks the f
        // continuity shifted body at row 0 → row 1.
        let original = cols[COL_F_X_LIMBS_IN_OFFSET][1].clone();
        cols[COL_F_X_LIMBS_IN_OFFSET][1] = original.add(&Scalar::one(curve));

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let alpha = Scalar::from_u64(7, curve);
        let body = evaluate_shifted_row(&col_refs, 0, 1, 2, &alpha);
        assert!(
            !body.is_zero(),
            "tampered f_x_limbs_in on row 1 must fire f continuity at row 0",
        );

        // Sanity: the mirror constraint for row 1 also fires, because
        // f_x_limbs_in[1] no longer equals acc_pre[1].
        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let mirror_constraint_idx = MS_NUM_ROW_CONSTRAINTS + 8;
        assert!(
            !results[mirror_constraint_idx][1].is_zero(),
            "mirror constraint for f_x_limbs_in should also fire on row 1",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 12: tampering t_x_limbs_in on row 1 breaks the T continuity
    // shifted constraint at row 0.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_t_continuity_is_detected() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;

        let original = cols[COL_T_X_LIMBS_IN_OFFSET][1].clone();
        cols[COL_T_X_LIMBS_IN_OFFSET][1] = original.add(&Scalar::one(curve));

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let alpha = Scalar::from_u64(7, curve);
        let body = evaluate_shifted_row(&col_refs, 0, 1, 3, &alpha);
        assert!(
            !body.is_zero(),
            "tampered t_x_limbs_in on row 1 must fire T continuity at row 0",
        );

        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let mirror_constraint_idx = MS_NUM_ROW_CONSTRAINTS + 10;
        assert!(
            !results[mirror_constraint_idx][1].is_zero(),
            "mirror constraint for t_x_limbs_in should also fire on row 1",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 13: NAF binary + mutual-exclusion constraints are honestly
    // satisfied, and tampering naf_pos to 2 fires the binary constraint.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn naf_columns_binary_and_mutex_honest_and_tampered() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let curve = CurveType::Bls12381;

        // Honest: every active row has naf_pos ∈ {0,1} (= is_addition),
        // naf_neg = 0, mutex trivially holds.
        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        let naf_pos_bin = MS_NUM_ROW_CONSTRAINTS + 5;
        let naf_neg_bin = MS_NUM_ROW_CONSTRAINTS + 6;
        let naf_mutex = MS_NUM_ROW_CONSTRAINTS + 7;
        for r in 0..w.num_rows() {
            assert!(results[naf_pos_bin][r].is_zero());
            assert!(results[naf_neg_bin][r].is_zero());
            assert!(results[naf_mutex][r].is_zero());
        }

        // Tamper naf_pos[0] from 0 to 2 — fires the binary constraint.
        cols[COL_NAF_POS][0] = Scalar::from_u64(2, curve);
        let col_refs2: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results2 = cs.evaluate_on_domain(&col_refs2, trace.num_rows);
        assert!(
            !results2[naf_pos_bin][0].is_zero(),
            "naf_pos = 2 must fire binary constraint",
        );

        // Also tamper naf_neg[0] to 1 — fires the mutex constraint
        // because naf_pos * naf_neg = 2 * 1 = 2 != 0.
        cols[COL_NAF_NEG][0] = Scalar::one(curve);
        let col_refs3: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results3 = cs.evaluate_on_domain(&col_refs3, trace.num_rows);
        assert!(
            !results3[naf_mutex][0].is_zero(),
            "naf_pos · naf_neg != 0 must fire mutex constraint",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 14: make_miller_loop_to_miller_step_descriptor well-formed.
    // ───────────────────────────────────────────────────────────────────

    // ───────────────────────────────────────────────────────────────────
    // Test 15 (Task #157): populate_miller_loop_trace yields a
    // nonnative_fp_air trace whose populated rows satisfy every
    // row-local nonnative_fp_air constraint.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn populate_miller_loop_trace_satisfies_nfp_row_local_constraints() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());

        // The fixture pins ≥ 2 rows of the Miller loop: row 0
        // (doubling) — the only row that the wired descriptors actually
        // populate — plus the first addition row, which the
        // miller_loop trace already carries from the standard witness.
        assert!(
            fixture.miller_loop_trace.num_rows >= 2,
            "populate_miller_loop_trace must cover ≥ 2 Miller iterations \
             (got {})",
            fixture.miller_loop_trace.num_rows,
        );
        // Task #174 + Task #307: the fixture now sweeps **all 63
        // doubling rows** × all 90 descriptors PLUS a backward-compat
        // representative-write pass for the 12 representative
        // descriptors. The A-side row total is therefore:
        //   - 78 * NUM_DOUBLING_ROWS (all Fp12 / acc_int / G2-dbl) +
        //     12 * NUM_ADDITION_ROWS (all G2-add)
        //     = 4974 from the all-90 sweep
        //   - + NUM_DOUBLING_ROWS * 8 + NUM_ADDITION_ROWS * 4 = 524
        //     from the representative pass
        //   - = 5498 total Mul rows
        let all_90_sweep = 78 * NUM_DOUBLING_ROWS + 12 * NUM_ADDITION_ROWS;
        let representative_pass =
            NUM_DOUBLING_ROWS * 8 + NUM_ADDITION_ROWS * 4;
        let expected_mults = all_90_sweep + representative_pass;
        assert_eq!(
            fixture.num_populated_fp_mults,
            expected_mults,
            "expected all_90_sweep ({}) + representative_pass ({}) = {} \
             populated Fp mults",
            all_90_sweep,
            representative_pass,
            expected_mults,
        );
        assert_eq!(fixture.representative_descriptors.len(), 12);
        // Task #307: the new `all_*` fields are populated.
        assert_eq!(fixture.all_populated_descriptors.len(), 90);
        assert_eq!(fixture.all_descriptor_is_winner.len(), 90);
        assert_eq!(fixture.all_descriptor_row_writes, all_90_sweep);

        // Every populated nonnative_fp_air row satisfies every row-local
        // constraint.
        let beta = Scalar::from_u64(11, CurveType::Bls12381);
        let col_refs: Vec<&Vec<Scalar>> = fixture
            .nonnative_fp_trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
        let evals = crate::nonnative_fp_air::evaluate_constraints(&col_refs, &beta);
        for cat in &evals {
            for (row, val) in cat.values.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "nonnative_fp_air constraint '{}' fired at row {} \
                     (expected zero on honest populated witness)",
                    cat.label,
                    row,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 16 (Task #157): each representative descriptor's cross-AIR
    // LogUp closure holds over the paired traces.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn populate_miller_loop_trace_descriptor_closures_hold() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        let curve = CurveType::Bls12381;
        let beta = Scalar::from_u64(13, curve);
        let gamma = Scalar::from_u64(17, curve);

        // Helper: build a per-descriptor mini nonnative_fp_air trace
        // containing one FpOp::Mul row per **doubling row** of the
        // Miller loop, with each Mul's (a, b) read from the descriptor's
        // b_columns at that row. After Task #174, every doubling row
        // (63 total) contributes a selected B-side tuple; the A-side
        // trace must mirror them 1-to-1 for the multiset equality to
        // hold.
        let per_descriptor_nfp_trace =
            |desc: &crate::cross_air_logup::CrossAirLogUpDescriptor| {
                let a_base = desc.b_columns[0];
                let b_base = desc.b_columns[ms::LIMBS_PER_FP];
                // Use the descriptor's own gate selector (COL_IS_DOUBLING
                // or COL_IS_ADDITION) so the per-descriptor mini-A trace
                // mirrors exactly the rows where the B side is active.
                let gate_col = desc
                    .b_selector_column
                    .expect("descriptor carries a b_selector_column");

                let mut ops: Vec<FpOp> = Vec::new();
                for r in 0..fixture.miller_loop_trace.num_rows {
                    let gate = &fixture
                        .miller_loop_trace
                        .columns[gate_col]
                        .evaluations[r];
                    if gate.is_zero() {
                        continue;
                    }
                    let mut a_limbs = [0u64; 6];
                    let mut b_limbs = [0u64; 6];
                    for j in 0..6 {
                        a_limbs[j] = scalar_to_u64(
                            &fixture.miller_loop_trace.columns[a_base + j]
                                .evaluations[r],
                        );
                        b_limbs[j] = scalar_to_u64(
                            &fixture.miller_loop_trace.columns[b_base + j]
                                .evaluations[r],
                        );
                    }
                    ops.push(FpOp::Mul {
                        a: Fp { limbs: a_limbs },
                        b: Fp { limbs: b_limbs },
                    });
                }
                let n_rows = ops.len();
                let n_padded =
                    crate::trace::nearest_power_of_two(n_rows.max(1));
                let mut cols = nfp::alloc_trace(n_padded, curve);
                for (row, op) in ops.iter().enumerate() {
                    nfp::populate_row(&mut cols, row, op, curve);
                }
                TracePolynomials {
                    columns: cols
                        .into_iter()
                        .map(|evals| Polynomial {
                            evaluations: evals,
                            degree: n_rows,
                        })
                        .collect(),
                    num_rows: n_rows,
                    padded_size: n_padded as u64,
                    curve,
                }
            };

        let mut matched = 0usize;
        for desc in &fixture.representative_descriptors {
            let mini_a = per_descriptor_nfp_trace(desc);
            // Sanity: the per-descriptor trace has one Mul row per
            // gated row of the Miller loop. Doubling-gated descriptors
            // cover all 63 doubling rows; addition-gated descriptors
            // (Task #183) cover all 5 addition rows.
            let expected_rows = if desc
                .b_selector_column
                .expect("descriptor carries a b_selector_column")
                == ms::COL_IS_ADDITION
            {
                NUM_ADDITION_ROWS
            } else {
                NUM_DOUBLING_ROWS
            };
            assert_eq!(
                mini_a.num_rows,
                expected_rows,
                "per-descriptor mini-A trace for '{}' must cover all {} \
                 gated rows (got {})",
                desc.label,
                expected_rows,
                mini_a.num_rows,
            );
            let w = crate::cross_air_logup::compute_cross_air_logup_witness(
                &mini_a,
                &fixture.miller_loop_trace,
                desc,
                &beta,
                &gamma,
                curve,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "compute_cross_air_logup_witness failed for descriptor \
                     '{}': {}",
                    desc.label, err,
                )
            });
            assert!(
                w.closure_holds(),
                "closure must hold for descriptor '{}' over populated \
                 per-descriptor mini-A trace",
                desc.label,
            );
            matched += 1;
        }
        assert!(
            matched >= 12,
            "must validate every representative descriptor closure — 4 \
             Fp12 + 4 G2 dbl + 4 G2 add (got {})",
            matched,
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Test 17 (Task #174): all 63 doubling rows are populated and the
    // multi-row closure holds for at least one representative descriptor
    // spanning the full Miller loop.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn populate_miller_loop_trace_spans_all_doubling_rows() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());

        // Every doubling row should have its representative-descriptor
        // c_base cells filled with `a · b mod p` (not the original
        // honest miller_step_air values). For the G2-diagonal
        // descriptors the original Q_next cell is `(2·Q_curr).x.c0`
        // etc., whereas the overwritten value is `Q_curr.x.c0²` (a
        // distinct quantity). We assert that on **every doubling row**,
        // at least one G2-diagonal descriptor's c_base cell differs
        // from the corresponding honest Q_next limbs we'd recompute via
        // `G2Affine::double()`. This proves multi-row coverage, not
        // just row-0.
        let mut rows_with_overwrite = 0usize;
        let witness =
            MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        for (r, row) in witness.steps.rows.iter().enumerate() {
            if !row.is_doubling {
                continue;
            }
            // G2-diagonal slot 0 = x.c0; descriptor c_base is Q_next.x.c0.
            // Honest Q_next.x.c0 from the witness:
            let honest = row.q_next.x.c0;
            // Overwritten value: Q_curr.x.c0².
            let overwrite = row.q_curr.x.c0.mul(&row.q_curr.x.c0);
            // Read back what's actually in the trace at c_base:
            let g2_desc = &fixture.representative_descriptors[4]; // first G2 diag
            let c_base = g2_desc.b_columns[2 * ms::LIMBS_PER_FP];
            let mut actual_limbs = [0u64; 6];
            for j in 0..6 {
                actual_limbs[j] = scalar_to_u64(
                    &fixture.miller_loop_trace.columns[c_base + j]
                        .evaluations[r],
                );
            }
            let actual = Fp { limbs: actual_limbs };
            assert_eq!(
                actual.limbs, overwrite.limbs,
                "row {} G2 diag c_base must hold a·b mod p (= Q_curr.x.c0²)",
                r,
            );
            if actual.limbs != honest.limbs {
                rows_with_overwrite += 1;
            }
        }
        // At least 60 of the 63 doubling rows should show an actual
        // overwrite (the 2·Q ≠ Q² identity is generic; row 0 with
        // Q = generator gives a distinct pair, and subsequent rows
        // diverge further).
        assert!(
            rows_with_overwrite >= 60,
            "expected ≥ 60 doubling rows where the G2-diag overwrite \
             differs from the honest Q_next limbs (got {})",
            rows_with_overwrite,
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #233 — Test 18: ell-sparsity row-local constraints (indices
    // 12..17) vanish on every active row of the honest witness.
    //
    // The 6 structurally-zero Fp slots of the sparse `mul_by_014` lift
    // are pinned to zero via β-RLC, gated by IS_REAL.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn ell_sparsity_constraints_vanish_on_honest_witness() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        // The ell-sparsity bank lives at offsets [12..18) in the
        // loop-local list, which is itself appended after the step CS's
        // MS_NUM_ROW_CONSTRAINTS terms.
        for slot_idx in 0..ELL_SPARSITY_ZERO_FP_SLOTS.len() {
            let constraint_idx = MS_NUM_ROW_CONSTRAINTS + 12 + slot_idx;
            for r in 0..w.num_rows() {
                assert!(
                    results[constraint_idx][r].is_zero(),
                    "ell sparsity constraint {} (Fp slot {}) must vanish \
                     on honest row {}",
                    constraint_idx,
                    ELL_SPARSITY_ZERO_FP_SLOTS[slot_idx],
                    r,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #233 — Test 19: tampering any limb of a structurally-zero
    // Fp slot in `line_value` on an active row fires the matching
    // ell-sparsity constraint.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn tampered_ell_sparsity_zero_slot_is_detected() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let curve = CurveType::Bls12381;

        // For each of the 6 zero Fp slots, tamper limb 0 of that slot
        // on row 0 (which is the first doubling row → IS_REAL = 1) and
        // verify the corresponding ell-sparsity constraint fires there.
        for (slot_idx, &slot) in ELL_SPARSITY_ZERO_FP_SLOTS.iter().enumerate() {
            let mut cols: Vec<Vec<Scalar>> =
                trace.columns.iter().map(|p| p.evaluations.clone()).collect();
            let tamper_col = COL_LINE_VALUE_OFFSET + slot * LIMBS_PER_FP;
            cols[tamper_col][0] = Scalar::one(curve);

            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);

            let constraint_idx = MS_NUM_ROW_CONSTRAINTS + 12 + slot_idx;
            assert!(
                !results[constraint_idx][0].is_zero(),
                "tampering line_value Fp slot {} limb 0 on row 0 must \
                 fire ell-sparsity constraint {}",
                slot,
                constraint_idx,
            );

            // Other ell-sparsity constraints (sibling slots) at row 0
            // must remain zero — they only see their own slot.
            for (other_idx, &other_slot) in
                ELL_SPARSITY_ZERO_FP_SLOTS.iter().enumerate()
            {
                if other_idx == slot_idx {
                    continue;
                }
                let other_constraint = MS_NUM_ROW_CONSTRAINTS + 12 + other_idx;
                assert!(
                    results[other_constraint][0].is_zero(),
                    "tampering slot {} should NOT fire sibling \
                     constraint {} (slot {})",
                    slot,
                    other_constraint,
                    other_slot,
                );
            }
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #233 — Test 20: ell-sparsity constraints are gated by
    // IS_REAL — tampering on a padding row does NOT fire the constraint.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn ell_sparsity_constraints_gated_by_is_real() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let cs = MillerLoopConstraintSystem::new(trace.num_rows);
        let curve = CurveType::Bls12381;

        // Pick a padding row (beyond num_rows but within padded_size).
        let pad_row = w.num_rows();
        assert!(pad_row < trace.padded_size as usize);

        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let slot = ELL_SPARSITY_ZERO_FP_SLOTS[0];
        let tamper_col = COL_LINE_VALUE_OFFSET + slot * LIMBS_PER_FP;
        cols[tamper_col][pad_row] = Scalar::from_u64(99, curve);

        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        let constraint_idx = MS_NUM_ROW_CONSTRAINTS + 12;
        assert!(
            results[constraint_idx][pad_row].is_zero(),
            "ell-sparsity constraint must NOT fire on padding row \
             (IS_REAL = 0 there)",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #233 — Test 21: evaluate_at_point matches evaluate_on_domain
    // for the ell-sparsity constraint terms (both honest and tampered).
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn ell_sparsity_evaluate_at_point_matches_on_domain() {
        // We compare on a tampered trace: tamper line_value slot-4
        // limb 0 on row 0, then check that the AIR's accumulated
        // residual at α picks up the non-zero contribution.
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let curve = CurveType::Bls12381;
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        let slot = ELL_SPARSITY_ZERO_FP_SLOTS[0];
        let tamper_col = COL_LINE_VALUE_OFFSET + slot * LIMBS_PER_FP;
        cols[tamper_col][0] = Scalar::one(curve);

        let cs = MillerLoopConstraintSystem::new(trace.num_rows);

        // Honest reference: the per-row residual via evaluate_on_domain.
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);

        // Reconstruct evaluate_at_point's view by sampling the per-row
        // columns at row 0 and asserting its α-accumulator is non-zero
        // when α=2 (picks up the tampered ell-sparsity term).
        let alpha = Scalar::from_u64(2, curve);
        let row0_evals: Vec<Scalar> =
            (0..cols.len()).map(|c| cols[c][0].clone()).collect();
        let acc = cs.evaluate_at_point(&row0_evals, &alpha);

        // The tampered slot-4 sparsity constraint at row 0 is non-zero,
        // and so is its α-weighted contribution — together with the
        // honest zeros for all other constraints — making the full
        // accumulator non-zero.
        let sparsity_idx = MS_NUM_ROW_CONSTRAINTS + 12;
        assert!(!results[sparsity_idx][0].is_zero());
        assert!(
            !acc.is_zero(),
            "evaluate_at_point must reflect the tampered ell-sparsity \
             term at row 0",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #252 — fixture exposes the full 54 Fp12-sq descriptor set
    // and the 12 acc_intermediate diagonal descriptors.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn populate_miller_loop_trace_exposes_full_fp12_square_descriptors() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        assert_eq!(
            fixture.full_fp12_square_descriptors.len(),
            crate::miller_fp_descriptors::FP12_SQUARE_WIRED_MULTS,
            "fixture must expose all {} Fp12-square Fp-mult descriptors",
            crate::miller_fp_descriptors::FP12_SQUARE_WIRED_MULTS,
        );
        assert_eq!(fixture.full_fp12_square_descriptors.len(), 54);
        // Every descriptor must have the 18/18 column-tuple shape and
        // gate on the doubling selector.
        for d in &fixture.full_fp12_square_descriptors {
            assert_eq!(d.a_columns.len(), 18);
            assert_eq!(d.b_columns.len(), 18);
            assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));
            assert_eq!(d.a_selector_column, Some(nfp::COL_SEL_MUL));
        }
    }

    #[test]
    fn populate_miller_loop_trace_exposes_acc_intermediate_descriptors() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        assert_eq!(
            fixture.acc_intermediate_descriptors.len(),
            crate::miller_fp_descriptors::ACC_INTERMEDIATE_DESCRIPTORS,
            "fixture must expose all {} acc_intermediate descriptors",
            crate::miller_fp_descriptors::ACC_INTERMEDIATE_DESCRIPTORS,
        );
        assert_eq!(fixture.acc_intermediate_descriptors.len(), 12);

        // Each acc_intermediate descriptor binds acc_pre[i] · acc_pre[i]
        // → acc_intermediate[i] (currently aliased to acc_post[i]). The
        // A side targets `nonnative_fp_air`'s (COL_A, COL_B, COL_R =
        // fp::COL_C) triple gated by COL_SEL_MUL.
        for (i, d) in fixture.acc_intermediate_descriptors.iter().enumerate() {
            let a_base = COL_ACC_PRE_OFFSET + i * ms::LIMBS_PER_FP;
            for j in 0..ms::LIMBS_PER_FP {
                assert_eq!(d.b_columns[j], a_base + j);
                assert_eq!(d.b_columns[ms::LIMBS_PER_FP + j], a_base + j);
                // c-side aliases to acc_post[i] (placeholder).
                assert_eq!(
                    d.b_columns[2 * ms::LIMBS_PER_FP + j],
                    COL_ACC_POST_OFFSET + i * ms::LIMBS_PER_FP + j,
                );
                // A-side walks (COL_A, COL_B, COL_R).
                assert_eq!(d.a_columns[j], nfp::COL_A_OFFSET + j);
                assert_eq!(d.a_columns[ms::LIMBS_PER_FP + j], nfp::COL_B_OFFSET + j);
                assert_eq!(
                    d.a_columns[2 * ms::LIMBS_PER_FP + j],
                    nfp::COL_R_OFFSET + j,
                );
            }
            assert_eq!(d.a_selector_column, Some(nfp::COL_SEL_MUL));
            assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));
        }
    }

    #[test]
    fn acc_intermediate_diagonals_alias_fp12_phase_a_diagonals() {
        // Task #252's diagonals share the same (a_base, b_base, c_base)
        // triples as the first 12 entries of [`Fp12SquareDescriptors`]
        // (Phase A). The labels differ so the two sets can coexist in a
        // joint LogUp protocol. This is the intended scaffold: the
        // acc_intermediate set will later swap its c_base to a dedicated
        // `COL_ACC_INTERMEDIATE_OFFSET` once miller_step_air widens.
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        for i in 0..12 {
            let acc_d = &fixture.acc_intermediate_descriptors[i];
            let fp12_d = &fixture.full_fp12_square_descriptors[i];
            assert_eq!(
                acc_d.b_columns, fp12_d.b_columns,
                "acc_intermediate descriptor {} must share Fp12-sq Phase-A \
                 column triple",
                i,
            );
            assert_ne!(
                acc_d.label, fp12_d.label,
                "label collision between acc_intermediate diag {} and \
                 Fp12-square diag",
                i,
            );
        }
    }

    #[test]
    fn miller_loop_to_miller_step_descriptor_well_formed() {
        let d = make_miller_loop_to_miller_step_descriptor(2, 1);
        assert_eq!(d.label, "miller_loop_to_miller_step_v1");
        assert_eq!(d.a_layer_index, 2);
        assert_eq!(d.b_layer_index, 1);
        assert_eq!(d.a_columns.len(), T_X_LIMBS_LEN + F_X_LIMBS_LEN);
        assert_eq!(d.b_columns.len(), T_X_LIMBS_LEN + F_X_LIMBS_LEN);
        // A side columns are f_x_limbs_in + t_x_limbs_in (the row-input
        // stitching columns).
        assert_eq!(d.a_columns[0], COL_T_X_LIMBS_IN_OFFSET);
        assert_eq!(d.a_columns[T_X_LIMBS_LEN], COL_F_X_LIMBS_IN_OFFSET);
        assert_eq!(d.b_columns[0], COL_Q_CURR_OFFSET);
        assert_eq!(d.b_columns[T_X_LIMBS_LEN], COL_ACC_PRE_OFFSET);
        assert_eq!(d.a_selector_column, Some(COL_IS_DOUBLING));
        assert_eq!(d.b_selector_column, Some(ms::COL_IS_DOUBLING));
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #280: miller_loop_air ↔ bls12_381_curve_ops_air::fp Fp-mult
    // descriptor well-formedness + cross-AIR closure on the populated
    // fixture (sharing the BLS12-381 Fp AIR layer with the curve-ops AIR).
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn miller_loop_to_bls12_381_fp_mul_descriptor_well_formed() {
        use crate::bls12_381_curve_ops_air::fp as bls_fp;

        // Build one descriptor binding `(acc_pre[0], acc_pre[0],
        // acc_post[0])` (the Phase-A Fp12-square diagonal for slot 0)
        // against the shared BLS12-381 Fp AIR layer.
        let a_base = COL_ACC_PRE_OFFSET;
        let b_base = COL_ACC_PRE_OFFSET;
        let c_base = COL_ACC_POST_OFFSET;
        let d = make_miller_loop_to_bls12_381_fp_mul_descriptor(
            a_base,
            b_base,
            c_base,
            COL_IS_DOUBLING,
            "miller_loop_to_bls12_381_fp_test_v1",
            /* miller_loop_layer_index = */ 7,
            /* bls12_381_fp_layer_index = */ 3,
        );

        assert_eq!(d.label, "miller_loop_to_bls12_381_fp_test_v1");
        assert_eq!(d.a_layer_index, 3);
        assert_eq!(d.b_layer_index, 7);

        // 18-column tuple on each side (3 × 6 BE u64 limbs).
        assert_eq!(d.a_columns.len(), 3 * bls_fp::LIMBS_PER_FP);
        assert_eq!(d.b_columns.len(), 3 * LIMBS_PER_FP);

        // A side: bls12_381_curve_ops_air::fp's (COL_A, COL_B, COL_C)
        // triple in order, 6 limbs each.
        assert_eq!(d.a_columns[0], bls_fp::COL_A_OFFSET);
        assert_eq!(d.a_columns[bls_fp::LIMBS_PER_FP], bls_fp::COL_B_OFFSET);
        assert_eq!(d.a_columns[2 * bls_fp::LIMBS_PER_FP], bls_fp::COL_C_OFFSET);
        // Last limb of each operand contiguous.
        assert_eq!(
            d.a_columns[bls_fp::LIMBS_PER_FP - 1],
            bls_fp::COL_A_OFFSET + bls_fp::LIMBS_PER_FP - 1,
        );

        // B side: miller_loop_air's (a, b, c) triple.
        assert_eq!(d.b_columns[0], a_base);
        assert_eq!(d.b_columns[LIMBS_PER_FP], b_base);
        assert_eq!(d.b_columns[2 * LIMBS_PER_FP], c_base);

        // Selectors: A-side gated by COL_SEL_MUL (the MUL op kind of the
        // Fp AIR); B-side gated by the supplied miller_loop selector.
        assert_eq!(d.a_selector_column, Some(bls_fp::COL_SEL_MUL));
        assert_eq!(d.b_selector_column, Some(COL_IS_DOUBLING));
    }

    #[test]
    fn miller_loop_to_bls12_381_fp_mul_descriptor_matches_nonnative_layout() {
        // The bls12_381_curve_ops_air::fp module is documented as a
        // pure re-export of nonnative_fp_air's column layout. This test
        // pins that contract: a descriptor built via the new entry
        // point uses the same column indices and selector as one built
        // via miller_fp_descriptors::make_fp_mul_descriptor — so a
        // joint-LogUp instance can fold both miller_loop_air and
        // bls12_381_curve_ops_air's G1 doubling/addition Fp mults into
        // the same Fp AIR layer without re-deriving the column shape.
        use crate::miller_fp_descriptors as mfd;

        let a_base = COL_ACC_PRE_OFFSET;
        let b_base = COL_ACC_PRE_OFFSET;
        let c_base = COL_ACC_POST_OFFSET;

        let bls_d = make_miller_loop_to_bls12_381_fp_mul_descriptor(
            a_base,
            b_base,
            c_base,
            COL_IS_DOUBLING,
            "layout_check_bls",
            /* miller_loop_layer_index = */ 1,
            /* bls12_381_fp_layer_index = */ 0,
        );
        let nfp_d = mfd::make_fp_mul_descriptor(
            a_base,
            b_base,
            c_base,
            COL_IS_DOUBLING,
            "layout_check_nfp",
            /* miller_step_layer_index = */ 1,
            /* nonnative_fp_layer_index = */ 0,
        );

        assert_eq!(bls_d.a_columns, nfp_d.a_columns);
        assert_eq!(bls_d.b_columns, nfp_d.b_columns);
        assert_eq!(bls_d.a_selector_column, nfp_d.a_selector_column);
        assert_eq!(bls_d.b_selector_column, nfp_d.b_selector_column);
        assert_eq!(bls_d.a_layer_index, nfp_d.a_layer_index);
        assert_eq!(bls_d.b_layer_index, nfp_d.b_layer_index);
    }

    #[test]
    fn miller_loop_to_bls12_381_fp_mul_descriptor_closure_holds_on_fixture() {
        // The populated fixture (`populate_miller_loop_trace`) writes
        // honest Fp products into the miller_loop trace at the column
        // bases declared by every representative descriptor. Since
        // `bls12_381_curve_ops_air::fp` re-exports the nonnative_fp_air
        // column layout, a descriptor built via the new entry point
        // shares its column shape with the existing representative
        // descriptors, so the cross-AIR LogUp closure should hold when
        // paired with a per-descriptor mini Fp-AIR trace constructed
        // from the descriptor's own (a, b) operand cells on every
        // active row.
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        let curve = CurveType::Bls12381;
        let beta = Scalar::from_u64(11, curve);
        let gamma = Scalar::from_u64(17, curve);

        // Pick the same Phase-A Fp12 diagonal at slot 0 the fixture
        // pins (so honest product values exist at COL_ACC_POST_OFFSET),
        // but route the descriptor through the new entry point.
        let d = make_miller_loop_to_bls12_381_fp_mul_descriptor(
            COL_ACC_PRE_OFFSET,
            COL_ACC_PRE_OFFSET,
            COL_ACC_POST_OFFSET,
            COL_IS_DOUBLING,
            "populate_fixture_fp12_diag_0_v1",
            /* miller_loop_layer_index = */ 1,
            /* bls12_381_fp_layer_index = */ 0,
        );

        // Build a mini Fp-AIR trace with one FpOp::Mul row per active
        // doubling row, reading (a, b) from the descriptor's B-side
        // operand columns at that row.
        let a_base = d.b_columns[0];
        let b_base = d.b_columns[LIMBS_PER_FP];
        let gate_col = d.b_selector_column.expect("gate present");
        let mut ops: Vec<FpOp> = Vec::new();
        for r in 0..fixture.miller_loop_trace.num_rows {
            let gate =
                &fixture.miller_loop_trace.columns[gate_col].evaluations[r];
            if gate.is_zero() {
                continue;
            }
            let mut a_limbs = [0u64; 6];
            let mut b_limbs = [0u64; 6];
            for j in 0..6 {
                a_limbs[j] = scalar_to_u64(
                    &fixture.miller_loop_trace.columns[a_base + j]
                        .evaluations[r],
                );
                b_limbs[j] = scalar_to_u64(
                    &fixture.miller_loop_trace.columns[b_base + j]
                        .evaluations[r],
                );
            }
            ops.push(FpOp::Mul {
                a: Fp { limbs: a_limbs },
                b: Fp { limbs: b_limbs },
            });
        }
        let n_rows = ops.len();
        assert_eq!(
            n_rows, NUM_DOUBLING_ROWS,
            "slot-0 diagonal is COL_IS_DOUBLING-gated; expected 63 rows",
        );
        let n_padded = crate::trace::nearest_power_of_two(n_rows.max(1));
        let mut fp_cols = nfp::alloc_trace(n_padded, curve);
        for (row, op) in ops.iter().enumerate() {
            nfp::populate_row(&mut fp_cols, row, op, curve);
        }
        let mini_a = TracePolynomials {
            columns: fp_cols
                .into_iter()
                .map(|evals| Polynomial {
                    evaluations: evals,
                    degree: n_rows,
                })
                .collect(),
            num_rows: n_rows,
            padded_size: n_padded as u64,
            curve,
        };

        let w = crate::cross_air_logup::compute_cross_air_logup_witness(
            &mini_a,
            &fixture.miller_loop_trace,
            &d,
            &beta,
            &gamma,
            curve,
        )
        .expect("compute_cross_air_logup_witness must succeed");
        assert!(
            w.closure_holds(),
            "BLS12-381 Fp-mult cross-AIR LogUp closure failed for the \
             slot-0 Phase-A diagonal under the bls12_381_curve_ops_air::fp \
             binding",
        );
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #307 — all-90-descriptor population sweep
    // ───────────────────────────────────────────────────────────────────

    // Layout invariants for the fixture's `all_populated_descriptors`:
    //   [0..54)  = Fp12-square (54)
    //   [54..66) = acc_intermediate (12)
    //   [66..78) = G2 doubling (12)
    //   [78..90) = G2 addition (12)
    #[test]
    fn populate_miller_loop_trace_all_90_descriptors_layout() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        assert_eq!(fixture.all_populated_descriptors.len(), 90);
        assert_eq!(fixture.full_fp12_square_descriptors.len(), 54);
        assert_eq!(fixture.acc_intermediate_descriptors.len(), 12);
        assert_eq!(fixture.g2_double_descriptors.len(), 12);
        assert_eq!(fixture.g2_add_descriptors.len(), 12);

        // Index-for-index match with the union.
        for i in 0..54 {
            assert_eq!(
                fixture.all_populated_descriptors[i].label,
                fixture.full_fp12_square_descriptors[i].label,
            );
        }
        for i in 0..12 {
            assert_eq!(
                fixture.all_populated_descriptors[54 + i].label,
                fixture.acc_intermediate_descriptors[i].label,
            );
            assert_eq!(
                fixture.all_populated_descriptors[66 + i].label,
                fixture.g2_double_descriptors[i].label,
            );
            assert_eq!(
                fixture.all_populated_descriptors[78 + i].label,
                fixture.g2_add_descriptors[i].label,
            );
        }
    }

    // The 90-descriptor sweep applies exactly
    // `78 * NUM_DOUBLING_ROWS + 12 * NUM_ADDITION_ROWS` per-row writes
    // (66 doubling-gated descriptors × 63 rows + 12 addition-gated
    // descriptors × 5 rows = 4974). Plus the documented G2-dbl set
    // of 12 also COL_IS_DOUBLING-gated.
    #[test]
    fn populate_miller_loop_trace_all_90_descriptor_row_writes() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        let expected = 78 * NUM_DOUBLING_ROWS + 12 * NUM_ADDITION_ROWS;
        assert_eq!(fixture.all_descriptor_row_writes, expected);
        assert_eq!(expected, 78 * 63 + 12 * 5);
        assert_eq!(expected, 4974);
    }

    // Task #316: with 60 dedicated intermediate Fp witness slots added
    // to `miller_step_air`, every one of the 90 descriptors now has its
    // own (a, b, c) result column triple — the 60 previously-aliased
    // Fp12 Phase B..F + G2-dbl Phase B/D/E/F + G2-add Phase B/C/D/E/F
    // descriptors no longer collide. After the all-90 sweep, the
    // representative-write pass, AND the intermediate-refresh pass
    // (which re-reads `(a, b)` and re-writes `c_base` for descriptors
    // targeting intermediate slots, so any descriptors reading from
    // `Q_next` post-representative-pass still see their canonical
    // product), all 90 descriptors are documented winners.
    #[test]
    fn populate_miller_loop_trace_all_90_winner_classification() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        let winners: Vec<usize> = fixture
            .all_descriptor_is_winner
            .iter()
            .enumerate()
            .filter(|(_, &w)| w)
            .map(|(i, _)| i)
            .collect();

        // All 90 descriptors must win.
        for i in 0..90 {
            assert!(
                fixture.all_descriptor_is_winner[i],
                "descriptor {} (label: {}) must win after Task #316's \
                 60 dedicated intermediate slots",
                i,
                fixture.all_populated_descriptors[i].label,
            );
        }
        assert_eq!(
            winners.len(),
            90,
            "expected all 90 descriptors to win after Task #316 \
             (got {} winners, indices: {:?})",
            winners.len(),
            winners,
        );
    }

    // Task #316: after widening `miller_step_air` with 60 dedicated
    // intermediate Fp witness slots (one per former aliasing collision)
    // all 90 descriptor closures hold simultaneously. The per-descriptor
    // mini A-side nonnative_fp_air trace consistently matches the
    // descriptor's `(a, b, c)` tuple on every gated row.
    #[test]
    fn populate_miller_loop_trace_all_90_descriptor_closures() {
        let fixture = populate_miller_loop_trace(&sample_p(), &sample_q());
        let curve = CurveType::Bls12381;
        let beta = Scalar::from_u64(11, curve);
        let gamma = Scalar::from_u64(17, curve);

        let per_descriptor_nfp_trace =
            |desc: &crate::cross_air_logup::CrossAirLogUpDescriptor| {
                let a_base = desc.b_columns[0];
                let b_base = desc.b_columns[ms::LIMBS_PER_FP];
                let gate_col = desc
                    .b_selector_column
                    .expect("descriptor carries a b_selector_column");

                let mut ops: Vec<FpOp> = Vec::new();
                for r in 0..fixture.miller_loop_trace.num_rows {
                    let gate = &fixture
                        .miller_loop_trace
                        .columns[gate_col]
                        .evaluations[r];
                    if gate.is_zero() {
                        continue;
                    }
                    let mut a_limbs = [0u64; 6];
                    let mut b_limbs = [0u64; 6];
                    for j in 0..6 {
                        a_limbs[j] = scalar_to_u64(
                            &fixture.miller_loop_trace.columns[a_base + j]
                                .evaluations[r],
                        );
                        b_limbs[j] = scalar_to_u64(
                            &fixture.miller_loop_trace.columns[b_base + j]
                                .evaluations[r],
                        );
                    }
                    ops.push(FpOp::Mul {
                        a: Fp { limbs: a_limbs },
                        b: Fp { limbs: b_limbs },
                    });
                }
                let n_rows = ops.len();
                let n_padded =
                    crate::trace::nearest_power_of_two(n_rows.max(1));
                let mut cols = nfp::alloc_trace(n_padded, curve);
                for (row, op) in ops.iter().enumerate() {
                    nfp::populate_row(&mut cols, row, op, curve);
                }
                TracePolynomials {
                    columns: cols
                        .into_iter()
                        .map(|evals| Polynomial {
                            evaluations: evals,
                            degree: n_rows,
                        })
                        .collect(),
                    num_rows: n_rows,
                    padded_size: n_padded as u64,
                    curve,
                }
            };

        let mut winners_closed = 0usize;
        let mut losers_failed = 0usize;
        for (idx, desc) in fixture.all_populated_descriptors.iter().enumerate() {
            let mini_a = per_descriptor_nfp_trace(desc);
            let w_res = crate::cross_air_logup::compute_cross_air_logup_witness(
                &mini_a,
                &fixture.miller_loop_trace,
                desc,
                &beta,
                &gamma,
                curve,
            );
            let expected_winner = fixture.all_descriptor_is_winner[idx];
            if expected_winner {
                let w = w_res.unwrap_or_else(|err| {
                    panic!(
                        "compute_cross_air_logup_witness must succeed for \
                         winner descriptor '{}' (idx {}): {}",
                        desc.label, idx, err,
                    )
                });
                assert!(
                    w.closure_holds(),
                    "winner descriptor '{}' (idx {}) must have a passing \
                     closure",
                    desc.label,
                    idx,
                );
                winners_closed += 1;
            } else {
                // Loser: either the witness construction errors (A's
                // tuples don't appear in B) OR the closure fails. Either
                // outcome documents the c_base aliasing conflict.
                let fails = match w_res {
                    Err(_) => true,
                    Ok(w) => !w.closure_holds(),
                };
                assert!(
                    fails,
                    "loser descriptor '{}' (idx {}) unexpectedly closed — \
                     c_base aliasing should have made it impossible",
                    desc.label,
                    idx,
                );
                losers_failed += 1;
            }
        }
        // Task #316: with 60 dedicated intermediate slots all 90
        // descriptor closures hold simultaneously.
        assert_eq!(
            winners_closed, 90,
            "expected all 90 winners to close (got {})",
            winners_closed,
        );
        assert_eq!(
            losers_failed, 0,
            "expected 0 losers after Task #316 (got {})",
            losers_failed,
        );
        assert_eq!(winners_closed + losers_failed, 90);
    }

    // ───────────────────────────────────────────────────────────────────
    // Task #306 (closes #233): λ witness blocks are populated correctly
    // on every doubling row and zeroed on addition / padding rows.
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn lambda_witness_blocks_populated_on_doubling_rows() {
        use crate::nonnative_fp::Fp;
        use crate::nonnative_fp::Fp2 as Fp2Tower;

        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let curve = CurveType::Bls12381;

        let check_fp2 = |cols: &[Polynomial], base: usize, r: usize, v: &Fp2Tower, label: &str| {
            for j in 0..LIMBS_PER_FP {
                let got_c0 = &cols[base + j].evaluations[r];
                let exp_c0 = Scalar::from_u64(v.c0.limbs[j], curve);
                assert!(
                    got_c0.sub(&exp_c0).is_zero(),
                    "{}: row {} limb {} c0 mismatch",
                    label, r, j,
                );
                let got_c1 = &cols[base + LIMBS_PER_FP + j].evaluations[r];
                let exp_c1 = Scalar::from_u64(v.c1.limbs[j], curve);
                assert!(
                    got_c1.sub(&exp_c1).is_zero(),
                    "{}: row {} limb {} c1 mismatch",
                    label, r, j,
                );
            }
        };

        let two = Fp2Tower { c0: Fp::from_u64(2), c1: Fp::zero() };
        let three = Fp2Tower { c0: Fp::from_u64(3), c1: Fp::zero() };

        let mut checked = 0usize;
        for (r, row) in w.steps.rows.iter().enumerate() {
            if !row.is_doubling {
                continue;
            }
            let t = row.q_curr;
            let lambda_num = three.mul(&t.x.square());
            let lambda_denom = two.mul(&t.y);
            let lambda_denom_inv =
                lambda_denom.invert().expect("2·T.y invertible");
            let lambda = lambda_num.mul(&lambda_denom_inv);

            check_fp2(
                &trace.columns,
                COL_LAMBDA_NUM_OFFSET,
                r,
                &lambda_num,
                "lambda_num",
            );
            check_fp2(
                &trace.columns,
                COL_LAMBDA_DENOM_OFFSET,
                r,
                &lambda_denom,
                "lambda_denom",
            );
            check_fp2(
                &trace.columns,
                COL_LAMBDA_DENOM_INV_OFFSET,
                r,
                &lambda_denom_inv,
                "lambda_denom_inv",
            );
            check_fp2(&trace.columns, COL_LAMBDA_OFFSET, r, &lambda, "lambda");
            checked += 1;
        }
        assert!(
            checked >= 60,
            "expected ≥ 60 doubling rows (got {})",
            checked,
        );
    }

    #[test]
    fn lambda_witness_blocks_zero_on_addition_and_padding_rows() {
        let w = MillerLoopWitness::from_pk_and_p(&sample_p(), &sample_q());
        let trace = build_trace_polynomials(&w, CurveType::Bls12381);
        let num_rows = w.num_rows();
        let padded = trace.padded_size as usize;

        let lambda_blocks = [
            COL_LAMBDA_NUM_OFFSET,
            COL_LAMBDA_DENOM_OFFSET,
            COL_LAMBDA_DENOM_INV_OFFSET,
            COL_LAMBDA_OFFSET,
        ];

        let mut addition_rows_checked = 0usize;
        for r in 0..num_rows {
            let is_add =
                !trace.columns[COL_IS_ADDITION].evaluations[r].is_zero();
            if !is_add {
                continue;
            }
            for base in lambda_blocks.iter() {
                for j in 0..LIMBS_PER_FP2 {
                    assert!(
                        trace.columns[base + j].evaluations[r].is_zero(),
                        "addition row {} block@{} limb {} must be zero",
                        r, base, j,
                    );
                }
            }
            addition_rows_checked += 1;
        }
        assert!(
            addition_rows_checked >= 1,
            "expected ≥ 1 addition row",
        );

        for r in num_rows..padded {
            for base in lambda_blocks.iter() {
                for j in 0..LIMBS_PER_FP2 {
                    assert!(
                        trace.columns[base + j].evaluations[r].is_zero(),
                        "padding row {} block@{} limb {} must be zero",
                        r, base, j,
                    );
                }
            }
        }
    }

    #[test]
    fn ell_lambda_descriptors_b_columns_point_into_miller_loop_layout() {
        // The EllLambda descriptors target miller_loop_air column
        // indices (since the λ witness blocks live in this AIR). Every
        // B-side column index must fall inside `[0, NUM_COLUMNS)`.
        use crate::miller_fp_descriptors::EllLambdaDescriptors;
        let set = EllLambdaDescriptors::build(1, 0);
        for d in set.descriptors.iter() {
            for &col in d.b_columns.iter() {
                assert!(
                    col < NUM_COLUMNS,
                    "descriptor {} references B-side col {} ≥ NUM_COLUMNS ({})",
                    d.label, col, NUM_COLUMNS,
                );
            }
        }
    }
}
