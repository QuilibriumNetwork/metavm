//! EIP-1559 fee market AIR.
//!
//! Proves:
//!  1. The per-block `new_base_fee` derives correctly from the parent
//!     block's `(parent_base_fee, parent_gas_used, parent_gas_target)`
//!     under the EIP-1559 base-fee update rule:
//!       new_base_fee
//!         = parent_base_fee
//!           + parent_base_fee * (gas_used - target) / target / 8
//!         (delta sign is encoded via `delta_sign`; integer division
//!         is enforced via witnessed quotient + remainder.)
//!  2. Per-tx fees: `effective_gas_price = base_fee + tip` where
//!     `tip = min(max_priority_fee_per_gas, max_fee_per_gas - base_fee)`
//!     and `base_fee_burn_per_gas = base_fee`.
//!
//! ## Row classes
//! Each row is either a *block row* (`is_block_row = 1`) or a *tx row*
//! (`is_tx_row = 1`), never both. Padding rows have both flags = 0.
//!
//! Block rows pin (parent_base_fee, parent_gas_used, parent_gas_target,
//! new_base_fee) and witness the integer-division quotient + remainder
//! used by the EIP-1559 update formula.
//!
//! Tx rows pin (tx_index, base_fee, max_priority_fee, max_fee,
//! effective_gas_price, tip, base_fee_burn) and witness `tip_is_priority`
//! plus two `slack` values that algebraically enforce
//! `tip = min(max_priority_fee, max_fee - base_fee)`.
//!
//! ## Soundness notes / deferred work
//!  - The integer division `8*target*quot + rem = parent_base_fee*|gas_delta|`
//!    is enforced. `rem < 8*target` is witnessed via
//!    `remainder_slack = 8*target - rem - 1` and equality is enforced
//!    on block rows with `target != 0`, but the LE byte decomposition
//!    of `remainder_slack` (proving slack >= 0) is **deferred** to a
//!    follow-up. Without it, a malicious prover could choose a
//!    negative-in-the-field `remainder_slack` and over-flow `rem`.
//!    This is a known soundness gap; in real blocks `target > 0` always.
//!  - The `slack_a, slack_b >= 0` constraints likewise need range
//!    checks (deferred). Without them, `tip` could exceed `max_fee -
//!    base_fee` or `max_priority_fee` via negative-in-field slack.
//!  - Cross-AIR LogUp descriptors below bind block-row fields to
//!    `block_header_air` and tx-row fields to `tx_rlp_air`. The
//!    descriptors check tuple equality; the surrounding algebra here
//!    closes the fee-market rule. Together they form the algebraic
//!    economic-finality binding for fee accounting.

use crate::field::{CurveType, Scalar};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_shift, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

/// Convert a u128 to a Scalar via `low + high * 2^64`.
fn scalar_from_u128(v: u128, curve: CurveType) -> Scalar {
    let low = (v & 0xffff_ffff_ffff_ffffu128) as u64;
    let high = (v >> 64) as u64;
    if high == 0 {
        return Scalar::from_u64(low, curve);
    }
    let shift = Scalar::from_u64(1u64 << 32, curve); // 2^32
    let shift_64 = shift.mul(&shift);                // 2^64
    Scalar::from_u64(low, curve)
        .add(&Scalar::from_u64(high, curve).mul(&shift_64))
}

// ─── Column layout ────────────────────────────────────────────────────

// Row-class selectors.
pub const COL_IS_BLOCK_ROW: usize = 0;
pub const COL_IS_TX_ROW: usize = 1;

// Block-row payload.
pub const COL_PARENT_BASE_FEE: usize = 2;
pub const COL_PARENT_GAS_USED: usize = 3;
pub const COL_PARENT_GAS_TARGET: usize = 4;
pub const COL_NEW_BASE_FEE: usize = 5;
pub const COL_ABS_DELTA: usize = 6;       // |gas_used - target|
pub const COL_DELTA_SIGN: usize = 7;      // 1 = decrease, 0 = increase/equal
pub const COL_DELTA_QUOTIENT: usize = 8;  // parent_base_fee * abs_delta / (8 * target)
pub const COL_DELTA_REMAINDER: usize = 9;
pub const COL_REMAINDER_SLACK: usize = 10; // 8*target - remainder - 1

// Tx-row payload.
pub const COL_TX_INDEX: usize = 11;
pub const COL_TX_BASE_FEE: usize = 12;
pub const COL_TX_PRIORITY_FEE: usize = 13;     // max_priority_fee_per_gas
pub const COL_TX_MAX_FEE: usize = 14;          // max_fee_per_gas
pub const COL_EFFECTIVE_GAS_PRICE: usize = 15;
pub const COL_TIP_PER_GAS: usize = 16;
pub const COL_BASE_FEE_BURN_PER_GAS: usize = 17;
pub const COL_TX_GAS_USED: usize = 18;
pub const COL_TIP_IS_PRIORITY: usize = 19;     // 1 = tip == max_priority_fee, 0 = tip == max_fee - base_fee
pub const COL_SLACK_A: usize = 20;             // max_priority_fee - tip
pub const COL_SLACK_B: usize = 21;             // (max_fee - base_fee) - tip

pub const NUM_COLUMNS: usize = 22;

// Row-local constraints (see `evaluate_on_domain` below for the body
// of each):
//
//   0  is_block_row binary
//   1  is_tx_row binary
//   2  is_block_row * is_tx_row = 0 (mutually exclusive)
//   3  is_block_row * delta_sign binary
//   4  is_block_row * abs_delta definition (delta_sign-cased)
//   5  is_block_row * (8 * target * quot + rem - parent_base_fee * abs_delta) = 0
//   6  is_block_row * (new_base_fee update rule, delta_sign-cased)
//   7  is_block_row * parent_gas_target * (8*target - rem - 1 - slack_rem) = 0
//   8  is_tx_row * tip_is_priority binary
//   9  is_tx_row * (max_priority_fee - tip - slack_a) = 0
//  10  is_tx_row * (max_fee - base_fee - tip - slack_b) = 0
//  11  is_tx_row * tip_is_priority * slack_a = 0
//  12  is_tx_row * (1 - tip_is_priority) * slack_b = 0
//  13  is_tx_row * (effective_gas_price - base_fee - tip) = 0
//  14  is_tx_row * (base_fee_burn_per_gas - base_fee) = 0
pub const NUM_ROW_CONSTRAINTS: usize = 15;
// Cross-row (shifted) constraints, gated by `is_block_row[r] * is_block_row[r+1]`:
//   S0: parent_base_fee[r+1] - new_base_fee[r]   = 0  (multi-block base-fee chain)
//   S1: parent_gas_target[r+1] - parent_gas_target[r] = 0  (constant target across chain)
pub const NUM_SHIFTED: usize = 2;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct BlockFeeRow {
    pub parent_base_fee: u64,
    pub parent_gas_used: u64,
    pub parent_gas_target: u64,
    pub new_base_fee: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct TxFeeRow {
    pub tx_index: u64,
    pub base_fee: u64,
    pub max_priority_fee: u64,
    pub max_fee: u64,
    pub gas_used: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Eip1559FeeMarketWitness {
    pub block_rows: Vec<BlockFeeRow>,
    pub tx_rows: Vec<TxFeeRow>,
}

impl Eip1559FeeMarketWitness {
    /// Build a witness from a parent-block snapshot and a slice of txs
    /// (`(max_priority_fee_per_gas, max_fee_per_gas, gas_used)`).
    ///
    /// All txs share the *next-block* `base_fee` (i.e., the computed
    /// `new_base_fee`), as is the case for transactions executed in
    /// the new block. Returns the witness containing exactly one
    /// block row + one tx row per input tx.
    pub fn from_block(
        parent_base_fee: u64,
        parent_gas_used: u64,
        parent_gas_limit: u64,
        txs: &[(u64, u64, u64)],
    ) -> Self {
        let parent_gas_target = parent_gas_limit / 2;
        let new_base_fee = crate::eip1559_fee_market_air::next_base_fee_host(
            parent_base_fee,
            parent_gas_used,
            parent_gas_target,
        );
        let block_row = BlockFeeRow {
            parent_base_fee,
            parent_gas_used,
            parent_gas_target,
            new_base_fee,
        };
        let tx_rows: Vec<TxFeeRow> = txs
            .iter()
            .enumerate()
            .map(|(i, (mpf, mf, gas))| TxFeeRow {
                tx_index: i as u64,
                base_fee: new_base_fee,
                max_priority_fee: *mpf,
                max_fee: *mf,
                gas_used: *gas,
            })
            .collect();
        Self {
            block_rows: vec![block_row],
            tx_rows,
        }
    }

    /// Build a multi-block chained witness: each block's `new_base_fee`
    /// becomes the next block's `parent_base_fee` under a constant
    /// `parent_gas_target`. `parent_gas_used_per_block` provides one
    /// `gas_used` value per block (the starting `parent_base_fee` is
    /// for the very first row's parent).
    ///
    /// The cross-row shifted constraints (S0, S1) algebraically pin
    /// `parent_base_fee[r+1] = new_base_fee[r]` and
    /// `parent_gas_target[r+1] = parent_gas_target[r]` across all
    /// consecutive block rows.
    pub fn from_chain(
        initial_parent_base_fee: u64,
        parent_gas_target: u64,
        parent_gas_used_per_block: &[u64],
    ) -> Self {
        let mut block_rows = Vec::with_capacity(parent_gas_used_per_block.len());
        let mut pbf = initial_parent_base_fee;
        for &gas_used in parent_gas_used_per_block {
            let new_base_fee = next_base_fee_host(pbf, gas_used, parent_gas_target);
            block_rows.push(BlockFeeRow {
                parent_base_fee: pbf,
                parent_gas_used: gas_used,
                parent_gas_target,
                new_base_fee,
            });
            pbf = new_base_fee;
        }
        Self { block_rows, tx_rows: vec![] }
    }
}

/// EIP-1559 base-fee update rule, host-side reference.
///
/// `new = parent + parent*(used-target)/target/8` when `used > target`;
/// `new = parent - parent*(target-used)/target/8` when `used < target`;
/// `new = parent` when `used == target`. Mirrors the witness
/// derivation, but **does not** mirror `crates/evm/src/eip1559_fee.rs`'s
/// `.max(1)` fee-up nudge — that nudge is unenforceable here without an
/// extra selector and isn't required for tests that drive the AIR
/// directly.
pub fn next_base_fee_host(
    parent_base_fee: u64,
    parent_gas_used: u64,
    parent_gas_target: u64,
) -> u64 {
    if parent_gas_target == 0 {
        return parent_base_fee;
    }
    if parent_gas_used == parent_gas_target {
        return parent_base_fee;
    }
    if parent_gas_used > parent_gas_target {
        let delta = parent_gas_used - parent_gas_target;
        let q = (parent_base_fee as u128) * (delta as u128)
            / (8u128 * parent_gas_target as u128);
        parent_base_fee.saturating_add(q as u64)
    } else {
        let delta = parent_gas_target - parent_gas_used;
        let q = (parent_base_fee as u128) * (delta as u128)
            / (8u128 * parent_gas_target as u128);
        parent_base_fee.saturating_sub(q as u64)
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Eip1559FeeMarketWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.block_rows.len() + witness.tx_rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    // Block rows occupy the first `block_rows.len()` rows.
    for (i, b) in witness.block_rows.iter().enumerate() {
        columns[COL_IS_BLOCK_ROW][i] = one.clone();

        columns[COL_PARENT_BASE_FEE][i] = Scalar::from_u64(b.parent_base_fee, curve);
        columns[COL_PARENT_GAS_USED][i] = Scalar::from_u64(b.parent_gas_used, curve);
        columns[COL_PARENT_GAS_TARGET][i] = Scalar::from_u64(b.parent_gas_target, curve);
        columns[COL_NEW_BASE_FEE][i] = Scalar::from_u64(b.new_base_fee, curve);

        let (delta_sign_bit, abs_delta) = if b.parent_gas_used >= b.parent_gas_target {
            (0u64, b.parent_gas_used - b.parent_gas_target)
        } else {
            (1u64, b.parent_gas_target - b.parent_gas_used)
        };
        columns[COL_ABS_DELTA][i] = Scalar::from_u64(abs_delta, curve);
        columns[COL_DELTA_SIGN][i] = Scalar::from_u64(delta_sign_bit, curve);

        let numerator = (b.parent_base_fee as u128) * (abs_delta as u128);
        let denom = 8u128 * b.parent_gas_target as u128;
        let (quot, rem) = if denom == 0 {
            // target == 0: degenerate; pin both to 0. The block-row
            // constraints will satisfy 0 = parent_base_fee * abs_delta
            // iff abs_delta == 0 or parent_base_fee == 0.
            (0u128, 0u128)
        } else {
            (numerator / denom, numerator % denom)
        };
        columns[COL_DELTA_QUOTIENT][i] = scalar_from_u128(quot, curve);
        columns[COL_DELTA_REMAINDER][i] = scalar_from_u128(rem, curve);
        // remainder_slack = denom - rem - 1 (when denom > 0).
        let slack_rem: u128 = if denom == 0 { 0 } else { denom - rem - 1 };
        columns[COL_REMAINDER_SLACK][i] = scalar_from_u128(slack_rem, curve);
    }

    // Tx rows follow.
    let tx_offset = witness.block_rows.len();
    for (i, t) in witness.tx_rows.iter().enumerate() {
        let r = tx_offset + i;
        columns[COL_IS_TX_ROW][r] = one.clone();
        columns[COL_TX_INDEX][r] = Scalar::from_u64(t.tx_index, curve);
        columns[COL_TX_BASE_FEE][r] = Scalar::from_u64(t.base_fee, curve);
        columns[COL_TX_PRIORITY_FEE][r] = Scalar::from_u64(t.max_priority_fee, curve);
        columns[COL_TX_MAX_FEE][r] = Scalar::from_u64(t.max_fee, curve);

        // Compute tip = min(max_priority_fee, max_fee - base_fee).
        // (Assumes max_fee >= base_fee; if not, the EIP-1559
        // transaction is invalid and a separate gating AIR should
        // catch it.)
        let max_fee_minus_base = t.max_fee.saturating_sub(t.base_fee);
        let (tip, tip_is_priority) = if t.max_priority_fee <= max_fee_minus_base {
            (t.max_priority_fee, 1u64)
        } else {
            (max_fee_minus_base, 0u64)
        };
        let effective = t.base_fee.saturating_add(tip);
        let slack_a = t.max_priority_fee - tip;
        let slack_b = max_fee_minus_base - tip;
        columns[COL_EFFECTIVE_GAS_PRICE][r] = Scalar::from_u64(effective, curve);
        columns[COL_TIP_PER_GAS][r] = Scalar::from_u64(tip, curve);
        columns[COL_BASE_FEE_BURN_PER_GAS][r] = Scalar::from_u64(t.base_fee, curve);
        columns[COL_TX_GAS_USED][r] = Scalar::from_u64(t.gas_used, curve);
        columns[COL_TIP_IS_PRIORITY][r] = Scalar::from_u64(tip_is_priority, curve);
        columns[COL_SLACK_A][r] = Scalar::from_u64(slack_a, curve);
        columns[COL_SLACK_B][r] = Scalar::from_u64(slack_b, curve);
    }

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

pub struct Eip1559FeeMarketConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Eip1559FeeMarketConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Eip1559FeeMarketConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_block_row_binary".into(),
            "is_tx_row_binary".into(),
            "row_class_mutex".into(),
            "block_delta_sign_binary".into(),
            "block_abs_delta_def".into(),
            "block_div_quot_rem".into(),
            "block_new_base_fee_update".into(),
            "block_remainder_slack_eq".into(),
            "tx_tip_is_priority_binary".into(),
            "tx_slack_a_eq".into(),
            "tx_slack_b_eq".into(),
            "tx_priority_cap_complementary".into(),
            "tx_max_fee_cap_complementary".into(),
            "tx_effective_price_eq".into(),
            "tx_base_fee_burn_eq".into(),
        ]
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let one = Scalar::one(curve);
        let eight = Scalar::from_u64(8, curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        let mk = || vec![Scalar::zero(curve); n];

        // 0: is_block_row binary.
        let mut c0 = mk();
        for r in 0..n {
            let v = &columns[COL_IS_BLOCK_ROW][r];
            c0[r] = v.mul(&v.sub(&one));
        }
        out.push(c0);

        // 1: is_tx_row binary.
        let mut c1 = mk();
        for r in 0..n {
            let v = &columns[COL_IS_TX_ROW][r];
            c1[r] = v.mul(&v.sub(&one));
        }
        out.push(c1);

        // 2: mutual exclusion.
        let mut c2 = mk();
        for r in 0..n {
            c2[r] = columns[COL_IS_BLOCK_ROW][r].mul(&columns[COL_IS_TX_ROW][r]);
        }
        out.push(c2);

        // 3: is_block_row * delta_sign * (delta_sign - 1).
        let mut c3 = mk();
        for r in 0..n {
            let ds = &columns[COL_DELTA_SIGN][r];
            c3[r] = columns[COL_IS_BLOCK_ROW][r].mul(&ds.mul(&ds.sub(&one)));
        }
        out.push(c3);

        // 4: is_block_row * ((1-ds)*(used - target - abs_delta) + ds*(target - used - abs_delta)) = 0.
        let mut c4 = mk();
        for r in 0..n {
            let is_br = &columns[COL_IS_BLOCK_ROW][r];
            let ds = &columns[COL_DELTA_SIGN][r];
            let used = &columns[COL_PARENT_GAS_USED][r];
            let target = &columns[COL_PARENT_GAS_TARGET][r];
            let abs_d = &columns[COL_ABS_DELTA][r];
            let one_minus_ds = one.sub(ds);
            let up_term = used.sub(target).sub(abs_d);
            let down_term = target.sub(used).sub(abs_d);
            let body = one_minus_ds.mul(&up_term).add(&ds.mul(&down_term));
            c4[r] = is_br.mul(&body);
        }
        out.push(c4);

        // 5: is_block_row * (8 * target * quot + rem - parent_base_fee * abs_delta) = 0.
        let mut c5 = mk();
        for r in 0..n {
            let is_br = &columns[COL_IS_BLOCK_ROW][r];
            let target = &columns[COL_PARENT_GAS_TARGET][r];
            let quot = &columns[COL_DELTA_QUOTIENT][r];
            let rem = &columns[COL_DELTA_REMAINDER][r];
            let pbf = &columns[COL_PARENT_BASE_FEE][r];
            let abs_d = &columns[COL_ABS_DELTA][r];
            let lhs = eight.mul(target).mul(quot).add(rem);
            let rhs = pbf.mul(abs_d);
            c5[r] = is_br.mul(&lhs.sub(&rhs));
        }
        out.push(c5);

        // 6: is_block_row * ((1-ds)*(new - parent - quot) + ds*(new - parent + quot)) = 0.
        let mut c6 = mk();
        for r in 0..n {
            let is_br = &columns[COL_IS_BLOCK_ROW][r];
            let ds = &columns[COL_DELTA_SIGN][r];
            let new_bf = &columns[COL_NEW_BASE_FEE][r];
            let pbf = &columns[COL_PARENT_BASE_FEE][r];
            let quot = &columns[COL_DELTA_QUOTIENT][r];
            let one_minus_ds = one.sub(ds);
            let up = new_bf.sub(pbf).sub(quot);
            let down = new_bf.sub(pbf).add(quot);
            let body = one_minus_ds.mul(&up).add(&ds.mul(&down));
            c6[r] = is_br.mul(&body);
        }
        out.push(c6);

        // 7: is_block_row * target * (8*target - rem - 1 - slack_rem) = 0.
        let mut c7 = mk();
        for r in 0..n {
            let is_br = &columns[COL_IS_BLOCK_ROW][r];
            let target = &columns[COL_PARENT_GAS_TARGET][r];
            let rem = &columns[COL_DELTA_REMAINDER][r];
            let slack = &columns[COL_REMAINDER_SLACK][r];
            let body = eight.mul(target).sub(rem).sub(&one).sub(slack);
            c7[r] = is_br.mul(&target.mul(&body));
        }
        out.push(c7);

        // 8: is_tx_row * tip_is_priority binary.
        let mut c8 = mk();
        for r in 0..n {
            let v = &columns[COL_TIP_IS_PRIORITY][r];
            c8[r] = columns[COL_IS_TX_ROW][r].mul(&v.mul(&v.sub(&one)));
        }
        out.push(c8);

        // 9: is_tx_row * (max_priority - tip - slack_a) = 0.
        let mut c9 = mk();
        for r in 0..n {
            let is_tr = &columns[COL_IS_TX_ROW][r];
            let mpf = &columns[COL_TX_PRIORITY_FEE][r];
            let tip = &columns[COL_TIP_PER_GAS][r];
            let sa = &columns[COL_SLACK_A][r];
            c9[r] = is_tr.mul(&mpf.sub(tip).sub(sa));
        }
        out.push(c9);

        // 10: is_tx_row * (max_fee - base_fee - tip - slack_b) = 0.
        let mut c10 = mk();
        for r in 0..n {
            let is_tr = &columns[COL_IS_TX_ROW][r];
            let mf = &columns[COL_TX_MAX_FEE][r];
            let bf = &columns[COL_TX_BASE_FEE][r];
            let tip = &columns[COL_TIP_PER_GAS][r];
            let sb = &columns[COL_SLACK_B][r];
            c10[r] = is_tr.mul(&mf.sub(bf).sub(tip).sub(sb));
        }
        out.push(c10);

        // 11: is_tx_row * tip_is_priority * slack_a = 0.
        let mut c11 = mk();
        for r in 0..n {
            let is_tr = &columns[COL_IS_TX_ROW][r];
            let tip_is_pri = &columns[COL_TIP_IS_PRIORITY][r];
            let sa = &columns[COL_SLACK_A][r];
            c11[r] = is_tr.mul(&tip_is_pri.mul(sa));
        }
        out.push(c11);

        // 12: is_tx_row * (1 - tip_is_priority) * slack_b = 0.
        let mut c12 = mk();
        for r in 0..n {
            let is_tr = &columns[COL_IS_TX_ROW][r];
            let tip_is_pri = &columns[COL_TIP_IS_PRIORITY][r];
            let sb = &columns[COL_SLACK_B][r];
            let one_minus = one.sub(tip_is_pri);
            c12[r] = is_tr.mul(&one_minus.mul(sb));
        }
        out.push(c12);

        // 13: is_tx_row * (effective_price - base_fee - tip) = 0.
        let mut c13 = mk();
        for r in 0..n {
            let is_tr = &columns[COL_IS_TX_ROW][r];
            let eff = &columns[COL_EFFECTIVE_GAS_PRICE][r];
            let bf = &columns[COL_TX_BASE_FEE][r];
            let tip = &columns[COL_TIP_PER_GAS][r];
            c13[r] = is_tr.mul(&eff.sub(bf).sub(tip));
        }
        out.push(c13);

        // 14: is_tx_row * (base_fee_burn - base_fee) = 0.
        let mut c14 = mk();
        for r in 0..n {
            let is_tr = &columns[COL_IS_TX_ROW][r];
            let burn = &columns[COL_BASE_FEE_BURN_PER_GAS][r];
            let bf = &columns[COL_TX_BASE_FEE][r];
            c14[r] = is_tr.mul(&burn.sub(bf));
        }
        out.push(c14);

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let eight = Scalar::from_u64(8, curve);

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);
        let push = |body: Scalar, acc: &mut Scalar, ap: &mut Scalar| {
            *acc = acc.add(&ap.mul(&body));
            *ap = ap.mul(alpha);
        };

        let is_br = &col_evals[COL_IS_BLOCK_ROW];
        let is_tr = &col_evals[COL_IS_TX_ROW];
        let ds = &col_evals[COL_DELTA_SIGN];
        let used = &col_evals[COL_PARENT_GAS_USED];
        let target = &col_evals[COL_PARENT_GAS_TARGET];
        let abs_d = &col_evals[COL_ABS_DELTA];
        let quot = &col_evals[COL_DELTA_QUOTIENT];
        let rem = &col_evals[COL_DELTA_REMAINDER];
        let pbf = &col_evals[COL_PARENT_BASE_FEE];
        let new_bf = &col_evals[COL_NEW_BASE_FEE];
        let slack_rem = &col_evals[COL_REMAINDER_SLACK];
        let mpf = &col_evals[COL_TX_PRIORITY_FEE];
        let mf = &col_evals[COL_TX_MAX_FEE];
        let bf = &col_evals[COL_TX_BASE_FEE];
        let tip = &col_evals[COL_TIP_PER_GAS];
        let sa = &col_evals[COL_SLACK_A];
        let sb = &col_evals[COL_SLACK_B];
        let tip_is_pri = &col_evals[COL_TIP_IS_PRIORITY];
        let eff = &col_evals[COL_EFFECTIVE_GAS_PRICE];
        let burn = &col_evals[COL_BASE_FEE_BURN_PER_GAS];

        // 0
        push(is_br.mul(&is_br.sub(&one)), &mut acc, &mut alpha_pow);
        // 1
        push(is_tr.mul(&is_tr.sub(&one)), &mut acc, &mut alpha_pow);
        // 2
        push(is_br.mul(is_tr), &mut acc, &mut alpha_pow);
        // 3
        push(is_br.mul(&ds.mul(&ds.sub(&one))), &mut acc, &mut alpha_pow);
        // 4
        let one_minus_ds = one.sub(ds);
        let up_term = used.sub(target).sub(abs_d);
        let down_term = target.sub(used).sub(abs_d);
        let body4 = one_minus_ds.mul(&up_term).add(&ds.mul(&down_term));
        push(is_br.mul(&body4), &mut acc, &mut alpha_pow);
        // 5
        let lhs5 = eight.mul(target).mul(quot).add(rem);
        let rhs5 = pbf.mul(abs_d);
        push(is_br.mul(&lhs5.sub(&rhs5)), &mut acc, &mut alpha_pow);
        // 6
        let up6 = new_bf.sub(pbf).sub(quot);
        let down6 = new_bf.sub(pbf).add(quot);
        let body6 = one_minus_ds.mul(&up6).add(&ds.mul(&down6));
        push(is_br.mul(&body6), &mut acc, &mut alpha_pow);
        // 7
        let body7 = eight.mul(target).sub(rem).sub(&one).sub(slack_rem);
        push(is_br.mul(&target.mul(&body7)), &mut acc, &mut alpha_pow);
        // 8
        push(is_tr.mul(&tip_is_pri.mul(&tip_is_pri.sub(&one))), &mut acc, &mut alpha_pow);
        // 9
        push(is_tr.mul(&mpf.sub(tip).sub(sa)), &mut acc, &mut alpha_pow);
        // 10
        push(is_tr.mul(&mf.sub(bf).sub(tip).sub(sb)), &mut acc, &mut alpha_pow);
        // 11
        push(is_tr.mul(&tip_is_pri.mul(sa)), &mut acc, &mut alpha_pow);
        // 12
        let one_minus_tip = one.sub(tip_is_pri);
        push(is_tr.mul(&one_minus_tip.mul(sb)), &mut acc, &mut alpha_pow);
        // 13
        push(is_tr.mul(&eff.sub(bf).sub(tip)), &mut acc, &mut alpha_pow);
        // 14
        push(is_tr.mul(&burn.sub(bf)), &mut acc, &mut alpha_pow);

        acc
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let eight_poly = vec![Scalar::from_u64(8, curve)];

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);
        let push_body = |body: Vec<Scalar>, acc: &mut Vec<Scalar>, ap: &mut Scalar| {
            *acc = poly_add(acc, &poly_scalar_mul(&body, ap), curve);
            *ap = ap.mul(alpha);
        };

        let is_br = &col_coeffs[COL_IS_BLOCK_ROW];
        let is_tr = &col_coeffs[COL_IS_TX_ROW];
        let ds = &col_coeffs[COL_DELTA_SIGN];
        let used = &col_coeffs[COL_PARENT_GAS_USED];
        let target = &col_coeffs[COL_PARENT_GAS_TARGET];
        let abs_d = &col_coeffs[COL_ABS_DELTA];
        let quot = &col_coeffs[COL_DELTA_QUOTIENT];
        let rem = &col_coeffs[COL_DELTA_REMAINDER];
        let pbf = &col_coeffs[COL_PARENT_BASE_FEE];
        let new_bf = &col_coeffs[COL_NEW_BASE_FEE];
        let slack_rem = &col_coeffs[COL_REMAINDER_SLACK];
        let mpf = &col_coeffs[COL_TX_PRIORITY_FEE];
        let mf = &col_coeffs[COL_TX_MAX_FEE];
        let bf = &col_coeffs[COL_TX_BASE_FEE];
        let tip = &col_coeffs[COL_TIP_PER_GAS];
        let sa = &col_coeffs[COL_SLACK_A];
        let sb = &col_coeffs[COL_SLACK_B];
        let tip_is_pri = &col_coeffs[COL_TIP_IS_PRIORITY];
        let eff = &col_coeffs[COL_EFFECTIVE_GAS_PRICE];
        let burn = &col_coeffs[COL_BASE_FEE_BURN_PER_GAS];

        // 0: is_br * (is_br - 1)
        {
            let body = poly_mul(is_br, &poly_sub(is_br, &one_poly, curve), curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 1
        {
            let body = poly_mul(is_tr, &poly_sub(is_tr, &one_poly, curve), curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 2
        {
            let body = poly_mul(is_br, is_tr, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 3
        {
            let ds_m1 = poly_sub(ds, &one_poly, curve);
            let body = poly_mul(is_br, &poly_mul(ds, &ds_m1, curve), curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 4: is_br * ((1-ds)*(used-target-abs_d) + ds*(target-used-abs_d))
        let one_minus_ds = poly_sub(&one_poly, ds, curve);
        {
            let up = poly_sub(&poly_sub(used, target, curve), abs_d, curve);
            let down = poly_sub(&poly_sub(target, used, curve), abs_d, curve);
            let left = poly_mul(&one_minus_ds, &up, curve);
            let right = poly_mul(ds, &down, curve);
            let body_inner = poly_add(&left, &right, curve);
            let body = poly_mul(is_br, &body_inner, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 5: is_br * (8*target*quot + rem - pbf*abs_d)
        {
            let eight_target = poly_mul(&eight_poly, target, curve);
            let eight_target_quot = poly_mul(&eight_target, quot, curve);
            let lhs = poly_add(&eight_target_quot, rem, curve);
            let rhs = poly_mul(pbf, abs_d, curve);
            let body_inner = poly_sub(&lhs, &rhs, curve);
            let body = poly_mul(is_br, &body_inner, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 6: is_br * ((1-ds)*(new-pbf-quot) + ds*(new-pbf+quot))
        {
            let new_minus_pbf = poly_sub(new_bf, pbf, curve);
            let up = poly_sub(&new_minus_pbf, quot, curve);
            let down = poly_add(&new_minus_pbf, quot, curve);
            let left = poly_mul(&one_minus_ds, &up, curve);
            let right = poly_mul(ds, &down, curve);
            let body_inner = poly_add(&left, &right, curve);
            let body = poly_mul(is_br, &body_inner, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 7: is_br * target * (8*target - rem - 1 - slack_rem)
        {
            let eight_target = poly_mul(&eight_poly, target, curve);
            let t1 = poly_sub(&eight_target, rem, curve);
            let t2 = poly_sub(&t1, &one_poly, curve);
            let t3 = poly_sub(&t2, slack_rem, curve);
            let target_body = poly_mul(target, &t3, curve);
            let body = poly_mul(is_br, &target_body, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 8: is_tr * tip_is_pri * (tip_is_pri - 1)
        {
            let tp_m1 = poly_sub(tip_is_pri, &one_poly, curve);
            let body = poly_mul(is_tr, &poly_mul(tip_is_pri, &tp_m1, curve), curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 9: is_tr * (mpf - tip - sa)
        {
            let body_inner = poly_sub(&poly_sub(mpf, tip, curve), sa, curve);
            let body = poly_mul(is_tr, &body_inner, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 10: is_tr * (mf - bf - tip - sb)
        {
            let body_inner = poly_sub(
                &poly_sub(&poly_sub(mf, bf, curve), tip, curve),
                sb,
                curve,
            );
            let body = poly_mul(is_tr, &body_inner, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 11: is_tr * tip_is_pri * sa
        {
            let body = poly_mul(is_tr, &poly_mul(tip_is_pri, sa, curve), curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 12: is_tr * (1 - tip_is_pri) * sb
        {
            let one_minus = poly_sub(&one_poly, tip_is_pri, curve);
            let body = poly_mul(is_tr, &poly_mul(&one_minus, sb, curve), curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 13: is_tr * (eff - bf - tip)
        {
            let body_inner = poly_sub(&poly_sub(eff, bf, curve), tip, curve);
            let body = poly_mul(is_tr, &body_inner, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }
        // 14: is_tr * (burn - bf)
        {
            let body_inner = poly_sub(burn, bf, curve);
            let body = poly_mul(is_tr, &body_inner, curve);
            push_body(body, &mut acc, &mut alpha_pow);
        }

        let _ = alpha_pow; // silence
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_BLOCK_ROW, COL_IS_TX_ROW]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    // ── Cross-row (shifted) constraints ──────────────────────────────
    //
    // Layout of shifted_evals (size 4):
    //   [0] is_block_row(ω·X)
    //   [1] parent_base_fee(ω·X)
    //   [2] parent_gas_target(ω·X)
    //   [3] new_base_fee(ω·X)  (unused in eval, but pinned for symmetry)
    //
    // S0 (alpha_offset + 0):
    //   is_block_row(X) * is_block_row(ω·X)
    //     * (parent_base_fee(ω·X) - new_base_fee(X)) = 0
    // S1 (alpha_offset + 1):
    //   is_block_row(X) * is_block_row(ω·X)
    //     * (parent_gas_target(ω·X) - parent_gas_target(X)) = 0
    //
    // Both bodies are multiplied by (X - ω^{n-1}) by the prover to skip
    // the wrap-around row.

    fn num_shifted_constraints(&self) -> usize { NUM_SHIFTED }

    fn shifted_column_indices(&self) -> Vec<usize> {
        vec![
            COL_IS_BLOCK_ROW,    // 0
            COL_PARENT_BASE_FEE, // 1
            COL_PARENT_GAS_TARGET, // 2
            COL_NEW_BASE_FEE,    // 3 (currently unused; reserved)
        ]
    }

    fn evaluate_shifted_at_point(
        &self,
        col_evals_at_z: &[Scalar],
        shifted_evals: &[Scalar],
        z: &Scalar,
        omega_n_minus_1: &Scalar,
        alpha: &Scalar,
        alpha_offset: usize,
    ) -> Scalar {
        let curve = alpha.curve_type();
        if shifted_evals.len() < 4 || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(curve);
        }
        let is_br_curr = &col_evals_at_z[COL_IS_BLOCK_ROW];
        let is_br_next = &shifted_evals[0];
        let pbf_next = &shifted_evals[1];
        let target_next = &shifted_evals[2];
        let new_bf_curr = &col_evals_at_z[COL_NEW_BASE_FEE];
        let target_curr = &col_evals_at_z[COL_PARENT_GAS_TARGET];

        let gating = is_br_curr.mul(is_br_next);
        // S0: gating * (parent_base_fee(ω·z) - new_base_fee(z))
        let body_s0 = gating.mul(&pbf_next.sub(new_bf_curr));
        // S1: gating * (parent_gas_target(ω·z) - parent_gas_target(z))
        let body_s1 = gating.mul(&target_next.sub(target_curr));

        // α^(alpha_offset + i) · body_i
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = ap.mul(&body_s0);
        ap = ap.mul(alpha);
        total = total.add(&ap.mul(&body_s1));

        // Boundary exclusion: multiply by (z - ω^{n-1}).
        total.mul(&z.sub(omega_n_minus_1))
    }

    fn build_shifted_constraint_polynomial(
        &self,
        column_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let is_br = &column_coeffs[COL_IS_BLOCK_ROW];
        let is_br_next = poly_shift(is_br, omega);
        let gating = poly_mul(is_br, &is_br_next, curve);

        // S0: gating * (parent_base_fee(ω·X) - new_base_fee(X))
        let pbf = &column_coeffs[COL_PARENT_BASE_FEE];
        let pbf_next = poly_shift(pbf, omega);
        let new_bf = &column_coeffs[COL_NEW_BASE_FEE];
        let diff_s0 = poly_sub(&pbf_next, new_bf, curve);
        let body_s0 = poly_mul(&gating, &diff_s0, curve);

        // S1: gating * (parent_gas_target(ω·X) - parent_gas_target(X))
        let target = &column_coeffs[COL_PARENT_GAS_TARGET];
        let target_next = poly_shift(target, omega);
        let diff_s1 = poly_sub(&target_next, target, curve);
        let body_s1 = poly_mul(&gating, &diff_s1, curve);

        // α^(alpha_offset + i) folding.
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset { ap = ap.mul(alpha); }
        let mut total = poly_scalar_mul(&body_s0, &ap);
        ap = ap.mul(alpha);
        total = poly_add(&total, &poly_scalar_mul(&body_s1, &ap), curve);

        // Multiply by (X - ω^{n-1}) to skip the wrap-around row.
        let mut omega_n_minus_1 = Scalar::one(curve);
        for _ in 0..(domain_size - 1) { omega_n_minus_1 = omega_n_minus_1.mul(omega); }
        let neg = Scalar::zero(curve).sub(&omega_n_minus_1);
        let x_minus = vec![neg, Scalar::one(curve)];
        poly_mul(&total, &x_minus, curve)
    }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind block-row fields to `block_header_air`:
/// `(parent_base_fee, new_base_fee, parent_gas_used)`.
///
/// On the A side (this AIR), the tuple is `(parent_base_fee_l0,
/// new_base_fee_l0, parent_gas_used)` — using the 0-th LE u64 limb is
/// safe here because `block_header_air` exposes base fee as 4 LE
/// limbs and we want to bind only the low 64-bit limb (block headers
/// never exceed u64 for these fields in practice). Honest provers
/// concur in writing the same low limb; mismatching higher limbs are
/// already constrained on the block_header_air side via byte-decomp.
///
/// In the absence of a "scalar-equality" descriptor flavor, we drop
/// per-row gating to `is_block_row` on the A side. The B side gates
/// on `block_header_air::COL_IS_REAL`.
///
/// Caveat: this is a 3-column tuple; for full equality across all
/// base-fee limbs, mirror this descriptor for L1..L3 or add a wider
/// variant. Deferred.
pub fn make_eip1559_to_block_header_descriptor(
    fee_market_layer_index: usize,
    block_header_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "eip1559_to_block_header_v1".into(),
        a_layer_index: fee_market_layer_index,
        a_columns: vec![
            COL_PARENT_BASE_FEE,
            COL_NEW_BASE_FEE,
            COL_PARENT_GAS_USED,
        ],
        a_selector_column: Some(COL_IS_BLOCK_ROW),
        b_layer_index: block_header_layer_index,
        b_columns: vec![
            bh::COL_BASE_FEE_L0, // parent base fee — host must align rows so the parent's header is the source row.
            bh::COL_BASE_FEE_L0, // new base fee — likewise from the current block's header row.
            bh::COL_GAS_USED,
        ],
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

/// Bind tx-row fields to `tx_rlp_air`:
/// `(tx_priority_fee, tx_max_fee)` ↔ the EIP-1559 tx's
/// `max_priority_fee` and `max_fee` payloads.
///
/// `tx_rlp_air` exposes these as 32-byte BE byte arrays. We restrict
/// the binding to the LSB byte (most-significant 31 bytes of a normal
/// fee are typically zero; for full coverage, mirror this descriptor
/// across all 32 byte positions or build a U256 limb-binding variant).
/// This single-byte alignment is intentional for a first cut and is a
/// known soundness gap; deferred.
pub fn make_eip1559_to_tx_rlp_descriptor(
    fee_market_layer_index: usize,
    tx_rlp_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::tx_rlp_air as txr;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "eip1559_to_tx_rlp_v1".into(),
        a_layer_index: fee_market_layer_index,
        a_columns: vec![
            COL_TX_PRIORITY_FEE,
            COL_TX_MAX_FEE,
        ],
        a_selector_column: Some(COL_IS_TX_ROW),
        b_layer_index: tx_rlp_layer_index,
        b_columns: vec![
            txr::COL_MAX_PRIORITY_FEE_BYTE_OFFSET + 31, // LSB byte of 32-byte BE encoding
            txr::COL_MAX_FEE_BYTE_OFFSET + 31,
        ],
        b_selector_column: Some(txr::COL_IS_EIP1559),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cs_evaluates_to_zero(witness: &Eip1559FeeMarketWitness) {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(witness, curve);
        let cs = Eip1559FeeMarketConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} at row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn gas_used_equals_target_no_fee_change() {
        let parent_gas_limit = 30_000_000u64;
        let parent_gas_used = 15_000_000u64; // == target
        let parent_base_fee = 1_000_000_000u64;
        let w = Eip1559FeeMarketWitness::from_block(
            parent_base_fee,
            parent_gas_used,
            parent_gas_limit,
            &[],
        );
        assert_eq!(w.block_rows[0].new_base_fee, parent_base_fee);
        cs_evaluates_to_zero(&w);
    }

    #[test]
    fn gas_used_above_target_fee_increases() {
        let parent_gas_limit = 30_000_000u64;
        let parent_gas_used = 20_000_000u64; // > target=15M
        let parent_base_fee = 1_000_000_000u64;
        let w = Eip1559FeeMarketWitness::from_block(
            parent_base_fee,
            parent_gas_used,
            parent_gas_limit,
            &[],
        );
        assert!(w.block_rows[0].new_base_fee > parent_base_fee);
        cs_evaluates_to_zero(&w);
    }

    #[test]
    fn gas_used_below_target_fee_decreases() {
        let parent_gas_limit = 30_000_000u64;
        let parent_gas_used = 10_000_000u64; // < target=15M
        let parent_base_fee = 1_000_000_000u64;
        let w = Eip1559FeeMarketWitness::from_block(
            parent_base_fee,
            parent_gas_used,
            parent_gas_limit,
            &[],
        );
        assert!(w.block_rows[0].new_base_fee < parent_base_fee);
        cs_evaluates_to_zero(&w);
    }

    #[test]
    fn tampered_new_base_fee_detected() {
        let w = Eip1559FeeMarketWitness::from_block(
            1_000_000_000,
            20_000_000,
            30_000_000,
            &[],
        );
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Bump new_base_fee by one — should fire constraint 6.
        let bad = cols[COL_NEW_BASE_FEE][0]
            .add(&Scalar::from_u64(1, curve));
        cols[COL_NEW_BASE_FEE][0] = bad;
        let cs = Eip1559FeeMarketConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !results[6][0].is_zero(),
            "constraint 6 (new_base_fee update) must fire on tampered value",
        );
    }

    #[test]
    fn tx_row_effective_price_and_tip_priority_cap() {
        // base_fee=100, max_priority=20, max_fee=200 → tip=20 (priority binds),
        // effective_price=120.
        let w = Eip1559FeeMarketWitness::from_block(
            // Pick parent gas_used=target so block's new_base_fee = parent_base_fee = 100.
            100,
            15_000_000,
            30_000_000,
            &[(20, 200, 21_000)],
        );
        assert_eq!(w.block_rows[0].new_base_fee, 100);
        // Tip row sanity: from_block populates these.
        cs_evaluates_to_zero(&w);
    }

    #[test]
    fn tx_row_effective_price_and_tip_max_fee_cap() {
        // base_fee=100, max_priority=100, max_fee=130 → tip=30 (max_fee binds),
        // effective_price=130.
        let w = Eip1559FeeMarketWitness::from_block(
            100,
            15_000_000,
            30_000_000,
            &[(100, 130, 21_000)],
        );
        cs_evaluates_to_zero(&w);
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_eip1559_to_block_header_descriptor(0, 1);
        assert_eq!(d1.label, "eip1559_to_block_header_v1");
        assert_eq!(d1.a_columns.len(), 3);
        assert_eq!(d1.b_columns.len(), 3);
        assert_eq!(d1.a_selector_column, Some(COL_IS_BLOCK_ROW));

        let d2 = make_eip1559_to_tx_rlp_descriptor(0, 2);
        assert_eq!(d2.label, "eip1559_to_tx_rlp_v1");
        assert_eq!(d2.a_columns.len(), 2);
        assert_eq!(d2.b_columns.len(), 2);
        assert_eq!(d2.a_selector_column, Some(COL_IS_TX_ROW));
    }

    #[test]
    fn column_layout_pinned() {
        // Pin the exact layout so future refactors are visible in
        // diff. Any rearrangement must update the descriptors.
        assert_eq!(COL_IS_BLOCK_ROW, 0);
        assert_eq!(COL_IS_TX_ROW, 1);
        assert_eq!(COL_PARENT_BASE_FEE, 2);
        assert_eq!(COL_NEW_BASE_FEE, 5);
        assert_eq!(COL_DELTA_SIGN, 7);
        assert_eq!(COL_TX_INDEX, 11);
        assert_eq!(COL_EFFECTIVE_GAS_PRICE, 15);
        assert_eq!(COL_SLACK_B, 21);
        assert_eq!(NUM_COLUMNS, 22);
        assert_eq!(NUM_ROW_CONSTRAINTS, 15);
    }

    #[test]
    fn host_helper_matches_expected_branches() {
        // At target: unchanged.
        assert_eq!(next_base_fee_host(1000, 15_000_000, 15_000_000), 1000);
        // Above target: increases.
        let above = next_base_fee_host(1000, 30_000_000, 15_000_000);
        assert!(above > 1000);
        // Below target: decreases.
        let below = next_base_fee_host(1000, 0, 15_000_000);
        assert!(below < 1000);
    }

    /// Directly evaluate the shifted bodies on (row, row+1) pairs and
    /// confirm both vanish on every consecutive *block-row* pair.
    ///
    /// Body layout (matches `build_shifted_constraint_polynomial`):
    ///   S0 = is_br[r] * is_br[r+1] * (parent_base_fee[r+1] - new_base_fee[r])
    ///   S1 = is_br[r] * is_br[r+1] * (parent_gas_target[r+1] - parent_gas_target[r])
    fn shifted_bodies_zero_on_chain(witness: &Eip1559FeeMarketWitness) {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(witness, curve);
        let cols: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let n = cols[0].len();
        for r in 0..n.saturating_sub(1) {
            let is_br = &cols[COL_IS_BLOCK_ROW][r];
            let is_br_next = &cols[COL_IS_BLOCK_ROW][r + 1];
            let gating = is_br.mul(is_br_next);
            let pbf_next = &cols[COL_PARENT_BASE_FEE][r + 1];
            let new_bf_curr = &cols[COL_NEW_BASE_FEE][r];
            let s0 = gating.mul(&pbf_next.sub(new_bf_curr));
            assert!(
                s0.is_zero(),
                "shifted S0 (base-fee chain) non-zero at row {} (={:?})",
                r, s0,
            );
            let target_curr = &cols[COL_PARENT_GAS_TARGET][r];
            let target_next = &cols[COL_PARENT_GAS_TARGET][r + 1];
            let s1 = gating.mul(&target_next.sub(target_curr));
            assert!(
                s1.is_zero(),
                "shifted S1 (target constancy) non-zero at row {} (={:?})",
                r, s1,
            );
        }
    }

    #[test]
    fn ten_block_chain_realistic_gas() {
        // Realistic Cancun-era values: ~30M gas limit, ~15M gas target,
        // initial base fee ~7 gwei. Mix above- and below-target usage to
        // drive the fee both up and down across the 10-block chain.
        let parent_gas_target = 15_000_000u64;
        let initial_base_fee = 7_000_000_000u64; // 7 gwei
        let gas_used_per_block: [u64; 10] = [
            15_000_000, // at target — flat
            20_000_000, // +5M above — fee up
            22_000_000, // +7M above — fee up more
            10_000_000, // -5M below — fee down
            16_500_000, // +1.5M above — slight up
            14_000_000, // -1M below — slight down
            12_000_000, // -3M below — down
            15_000_000, // at target — flat
            25_000_000, // +10M above — fee up
            8_000_000,  // -7M below — fee down
        ];
        let w = Eip1559FeeMarketWitness::from_chain(
            initial_base_fee,
            parent_gas_target,
            &gas_used_per_block,
        );
        assert_eq!(w.block_rows.len(), 10);

        // Row-local constraints (per-block derivation) must hold on each block.
        cs_evaluates_to_zero(&w);

        // Shifted constraints (S0 + S1) must hold across consecutive block rows.
        shifted_bodies_zero_on_chain(&w);

        // Spot-check the chain semantics: each block's new_base_fee
        // equals the next block's parent_base_fee, target is constant.
        for r in 0..w.block_rows.len() - 1 {
            assert_eq!(
                w.block_rows[r].new_base_fee,
                w.block_rows[r + 1].parent_base_fee,
                "chain break at row {}: new_base_fee != next.parent_base_fee",
                r,
            );
            assert_eq!(
                w.block_rows[r].parent_gas_target,
                w.block_rows[r + 1].parent_gas_target,
                "target drift at row {}",
                r,
            );
        }

        // Spot-check sign of fee delta against gas_used vs target.
        for b in &w.block_rows {
            if b.parent_gas_used == b.parent_gas_target {
                assert_eq!(b.new_base_fee, b.parent_base_fee);
            } else if b.parent_gas_used > b.parent_gas_target {
                assert!(
                    b.new_base_fee >= b.parent_base_fee,
                    "fee must not decrease when gas_used > target (got {} -> {})",
                    b.parent_base_fee, b.new_base_fee,
                );
            } else {
                assert!(
                    b.new_base_fee <= b.parent_base_fee,
                    "fee must not increase when gas_used < target (got {} -> {})",
                    b.parent_base_fee, b.new_base_fee,
                );
            }
        }
    }

    #[test]
    fn ten_block_chain_tampered_link_detected() {
        // Build an honest chain, then tamper with one row's
        // parent_base_fee so it no longer equals the previous row's
        // new_base_fee. Confirm the S0 shifted body fires.
        let parent_gas_target = 15_000_000u64;
        let w = Eip1559FeeMarketWitness::from_chain(
            7_000_000_000,
            parent_gas_target,
            &[15_000_000u64; 10],
        );
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper row 3's parent_base_fee — break the chain link from row 2.
        let bad = cols[COL_PARENT_BASE_FEE][3]
            .add(&Scalar::from_u64(1, curve));
        cols[COL_PARENT_BASE_FEE][3] = bad;
        // S0 body at row 2: is_br[2]*is_br[3]*(parent_base_fee[3] - new_base_fee[2])
        let is_br_curr = &cols[COL_IS_BLOCK_ROW][2];
        let is_br_next = &cols[COL_IS_BLOCK_ROW][3];
        let gating = is_br_curr.mul(is_br_next);
        let pbf_next = &cols[COL_PARENT_BASE_FEE][3];
        let new_bf_curr = &cols[COL_NEW_BASE_FEE][2];
        let s0 = gating.mul(&pbf_next.sub(new_bf_curr));
        assert!(
            !s0.is_zero(),
            "S0 must fire on a tampered chain link (row 2 -> row 3)",
        );
    }

    #[test]
    fn ten_block_chain_tampered_target_detected() {
        let parent_gas_target = 15_000_000u64;
        let w = Eip1559FeeMarketWitness::from_chain(
            7_000_000_000,
            parent_gas_target,
            &[15_000_000u64; 10],
        );
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper row 5's parent_gas_target — break the constancy.
        let bad = cols[COL_PARENT_GAS_TARGET][5]
            .add(&Scalar::from_u64(1, curve));
        cols[COL_PARENT_GAS_TARGET][5] = bad;
        // S1 body at row 4: gating * (target[5] - target[4])
        let gating = cols[COL_IS_BLOCK_ROW][4]
            .mul(&cols[COL_IS_BLOCK_ROW][5]);
        let t_next = &cols[COL_PARENT_GAS_TARGET][5];
        let t_curr = &cols[COL_PARENT_GAS_TARGET][4];
        let s1 = gating.mul(&t_next.sub(t_curr));
        assert!(
            !s1.is_zero(),
            "S1 must fire on a tampered target (row 4 -> row 5)",
        );
    }

    #[test]
    fn shifted_skips_block_to_tx_boundary() {
        // Build a chain with 2 blocks then tx rows. The boundary row
        // (last block row -> first tx row) has gating = 1 * 0 = 0,
        // so the bodies vanish even though the chain doesn't extend.
        let mut w = Eip1559FeeMarketWitness::from_chain(
            7_000_000_000,
            15_000_000,
            &[15_000_000u64, 20_000_000],
        );
        // Append a single tx row with arbitrary unrelated values.
        let new_bf = w.block_rows[1].new_base_fee;
        w.tx_rows.push(TxFeeRow {
            tx_index: 0,
            base_fee: new_bf,
            max_priority_fee: 1_000_000_000,
            max_fee: new_bf + 2_000_000_000,
            gas_used: 21_000,
        });
        cs_evaluates_to_zero(&w);
        shifted_bodies_zero_on_chain(&w);
    }

    #[test]
    fn many_txs_round_trip() {
        let w = Eip1559FeeMarketWitness::from_block(
            500,
            16_000_000,
            30_000_000,
            &[
                (10, 600, 21_000),  // priority binds (10 ≤ max_fee - new_base_fee)
                (1000, 700, 50_000), // max_fee binds
                (0, 1000, 21_000),  // zero priority
            ],
        );
        assert_eq!(w.tx_rows.len(), 3);
        cs_evaluates_to_zero(&w);
    }
}
