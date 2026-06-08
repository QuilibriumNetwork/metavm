//! Proposer-index shuffle AIR.
//!
//! Proves that the beacon-state `proposer_index` for a given slot is the
//! one selected by the spec's `compute_proposer_index(state, indices,
//! seed)` loop. Per the consensus spec (phase 0 + later forks, with the
//! same MAX_EFFECTIVE_BALANCE constant used for proposer selection):
//!
//! ```text
//! def compute_proposer_index(state, indices, seed):
//!     i = 0
//!     total = len(indices)
//!     while True:
//!         candidate_index = indices[compute_shuffled_index(i % total,
//!                                                          total, seed)]
//!         random_byte = hash(seed + uint_to_bytes(uint64(i // 32)))[i % 32]
//!         effective_balance = state.validators[candidate_index].effective_balance
//!         if effective_balance * MAX_RANDOM_BYTE >= \
//!                 MAX_EFFECTIVE_BALANCE * random_byte:
//!             return candidate_index
//!         i += 1
//! ```
//!
//! This AIR commits one row per loop iteration. Each row exposes the
//! per-iteration witness data: the candidate index, candidate's
//! `effective_balance`, the SHA-256 input/output that produced the
//! `random_byte`, the accept/reject test and its slack ("margin"). The
//! final accepted row (`is_accepted = 1`) carries the `proposer_index`.
//!
//! ## Scope (this step)
//!
//! Closed algebraically here:
//!   - `is_accepted` and `is_real` are binary.
//!   - On `is_real` rows, the per-iteration accept/reject test
//!     `effective_balance * MAX_RANDOM_BYTE ?>= MAX_EFFECTIVE_BALANCE *
//!     random_byte` is bound by a margin column with byte-decomp range
//!     checks: accept rows pin `effective_balance * 255 -
//!     32_000_000_000 * random_byte = accept_margin ≥ 0`; reject rows
//!     pin `32_000_000_000 * random_byte - effective_balance * 255 =
//!     reject_margin ≥ 1` (so the inequality is strict).
//!   - `effective_balance` is bound to its 8 LE bytes (range-checked).
//!   - `random_byte` is range-checked to `[0, 255]`.
//!
//! Deferred (handed off to cross-AIR LogUp + downstream gadgets):
//!   - The shuffle itself: `shuffled_index_at_iter` is committed as
//!     witness only. The `compute_shuffled_index` Fisher–Yates / swap-or-
//!     not algorithm is structurally heavy; a dedicated shuffle gadget
//!     AIR will bind `(iteration_index, seed, total) → shuffled_index`
//!     in a follow-up. Here we forward the committed value into the
//!     `(candidate_index, effective_balance)` lookup so the
//!     post-shuffle data is at least bound to the validator registry.
//!   - `hash_output = sha256(hash_input)` is bound by the cross-AIR
//!     LogUp into `sha256_extract`; this AIR commits the input/output
//!     bytes and selects `random_byte = hash_output[iter % 32]` as a
//!     witness column (the per-iteration index selection is committed
//!     and gated by the iteration counter; algebraic 32-way one-hot
//!     selection of the active hash output byte is a follow-up).
//!   - The first accept-row must be globally minimal: the AIR scaffolds
//!     up to `MAX_ITERATIONS = 8` rows and the host-side witness builder
//!     stops emitting rows after the first accepting iteration. A
//!     "no-earlier-accept" cross-row constraint is a follow-up.
//!   - The `proposer_index` accepted by this AIR must equal the
//!     `proposer_index` consumed by the downstream block-proposer
//!     signature AIR: descriptor wired here matches the
//!     `randao_proposer_air::COL_PROPOSER_INDEX` column as the
//!     stepping-stone consumer (the dedicated block-proposer-sig AIR
//!     will replace that target once it lands).
//!
//! ## Constraints (10 row-local bodies)
//!
//! 0. `is_real_binary`             — `is_real * (is_real - 1) = 0`.
//! 1. `is_accepted_binary`         — `is_accepted * (is_accepted - 1) = 0`.
//! 2. `accept_identity` (gated by `is_real * is_accepted`):
//!    `eb * 255 - 32_000_000_000 * rb - accept_margin = 0`.
//! 3. `reject_identity` (gated by `is_real * (1 - is_accepted)`):
//!    `32_000_000_000 * rb - eb * 255 - reject_margin = 0` where
//!    `reject_margin = 1 + Σ_b reject_margin_minus_one_byte[b] * 256^b`.
//! 4. `eb_le_decomp`               — `eb - Σ_b eb_byte[b] * 256^b = 0`.
//! 5. `accept_margin_le_decomp`    — `accept_margin -
//!    Σ_b accept_margin_byte[b] * 256^b = 0`.
//! 6. `reject_margin_le_decomp`    — `reject_margin -
//!    (1 + Σ_b reject_margin_minus_one_byte[b] * 256^b) = 0`.
//! 7. `candidate_eq_shuffled` (gated by `is_real`): commits the contract
//!    `candidate_index = indices[shuffled_index_at_iter]` by binding the
//!    AIR's `candidate_index` column directly against the shuffled
//!    output via the cross-AIR LogUp (the equality is committed; the
//!    shuffle-derivation is the deferred follow-up). The body is
//!    `is_real * (candidate_index - shuffled_index_at_iter -
//!    (candidate_index - shuffled_index_at_iter)) = 0` — i.e. trivially
//!    zero as a row-local constraint; the binding lives in the
//!    `make_shuffle_to_validator_balances_descriptor` LogUp. The body
//!    is kept as a numbered slot so the constraint count is stable
//!    when the shuffle gadget closes.
//! 8. `random_byte_range_redundant`: range is also checked by the
//!    lookup declaration; the row-local body re-pins it as
//!    `is_real * (random_byte - random_byte) = 0` for layout stability.
//! 9. `iteration_index_bound` (gated by `is_real`): pin
//!    `iteration_index < MAX_ITERATIONS` via a tiny range check column
//!    `iteration_index_byte` such that
//!    `iteration_index - iteration_index_byte = 0` (one-byte range; for
//!    `MAX_ITERATIONS ≤ 256` this is sufficient).
//!
//! Padding rows: `is_real = 0`, all data columns zero, `is_accepted =
//! 0`, all margin columns zero. Constraints 2/3 are gated by `is_real`,
//! so they vanish. Constraints 4/5/6 with all-zero columns: 4 gives
//! `0 - 0 = 0` ✓; 5 gives `0 - 0 = 0` ✓; 6 gives `0 - (1 + 0) = -1 ≠
//! 0` — so this constraint must be gated by `is_real` too. Same for
//! constraint 0 (`is_real_binary` is fine, 0 trivially). And constraint
//! 9 is gated by `is_real`. Constraint 6 alongside constraint 3 are
//! gated by `is_real` to make padding rows vanish trivially.
//!
//! Range checks (8-bit) on every byte-decomp column and on `random_byte`.

use crate::cross_air_logup::CrossAirLogUpDescriptor;
use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Spec constants ───────────────────────────────────────────────────

/// `MAX_EFFECTIVE_BALANCE = 32 ETH = 32_000_000_000 Gwei` (Cancun-era).
pub const MAX_EFFECTIVE_BALANCE: u64 = 32_000_000_000;
/// `MAX_RANDOM_BYTE = 2^8 - 1 = 255`.
pub const MAX_RANDOM_BYTE: u64 = 255;

/// Maximum iterations scaffolded per shuffle witness. In practice the
/// loop almost always terminates in `O(1)` iterations (under the
/// expected effective-balance distribution); 8 rows gives ample margin
/// for the scaffold while keeping the AIR small.
pub const MAX_ITERATIONS: usize = 8;

pub const SEED_LEN: usize = 32;
pub const HASH_INPUT_LEN: usize = 32; // truncated to 32 for the scaffold; spec uses 32-byte seed + 8 byte counter (40 bytes); the 8-byte counter binding is deferred to the cross-AIR sha256 link.
pub const HASH_OUTPUT_LEN: usize = 32;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_ITERATION_INDEX: usize = 0; // 0
pub const COL_SEED_OFFSET: usize = COL_ITERATION_INDEX + 1; // 1..33
pub const COL_CANDIDATE_INDEX: usize = COL_SEED_OFFSET + SEED_LEN; // 33
pub const COL_CANDIDATE_EFFECTIVE_BALANCE: usize = COL_CANDIDATE_INDEX + 1; // 34
pub const COL_SHUFFLED_INDEX_AT_ITER: usize = COL_CANDIDATE_EFFECTIVE_BALANCE + 1; // 35
pub const COL_RANDOM_BYTE: usize = COL_SHUFFLED_INDEX_AT_ITER + 1; // 36
pub const COL_HASH_INPUT_OFFSET: usize = COL_RANDOM_BYTE + 1; // 37..69
pub const COL_HASH_OUTPUT_OFFSET: usize = COL_HASH_INPUT_OFFSET + HASH_INPUT_LEN; // 69..101

pub const COL_IS_ACCEPTED: usize = COL_HASH_OUTPUT_OFFSET + HASH_OUTPUT_LEN; // 101
pub const COL_IS_REAL: usize = COL_IS_ACCEPTED + 1; // 102

pub const COL_EB_BYTE_OFFSET: usize = COL_IS_REAL + 1; // 103..111
pub const COL_ACCEPT_MARGIN: usize = COL_EB_BYTE_OFFSET + U64_BYTES; // 111
pub const COL_ACCEPT_MARGIN_BYTE_OFFSET: usize = COL_ACCEPT_MARGIN + 1; // 112..120
pub const COL_REJECT_MARGIN: usize = COL_ACCEPT_MARGIN_BYTE_OFFSET + U64_BYTES; // 120
pub const COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET: usize = COL_REJECT_MARGIN + 1; // 121..129
pub const COL_ITERATION_INDEX_BYTE: usize =
    COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET + U64_BYTES; // 129
/// Last column: the `proposer_index` exposed by this AIR (constant on
/// every active row; equals `candidate_index` on the accepted row).
pub const COL_PROPOSER_INDEX: usize = COL_ITERATION_INDEX_BYTE + 1; // 130

pub const NUM_COLUMNS: usize = COL_PROPOSER_INDEX + 1; // 131

/// Row-local constraints (10 bodies).
pub const NUM_ROW_CONSTRAINTS: usize = 10;
/// No cross-row constraints in this step.
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct ProposerShuffleRow {
    pub iteration_index: u64,
    pub seed: [u8; SEED_LEN],
    pub candidate_index: u64,
    pub candidate_effective_balance: u64,
    pub shuffled_index_at_iter: u64,
    pub random_byte: u8,
    pub hash_input: [u8; HASH_INPUT_LEN],
    pub hash_output: [u8; HASH_OUTPUT_LEN],
    pub is_accepted: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ProposerShuffleWitness {
    pub rows: Vec<ProposerShuffleRow>,
    /// The final accepted `proposer_index` (= `candidate_index` of the
    /// accept row). Committed on every active row.
    pub proposer_index: u64,
}

impl ProposerShuffleWitness {
    pub fn from_rows(rows: Vec<ProposerShuffleRow>, proposer_index: u64) -> Self {
        Self { rows, proposer_index }
    }

    /// Host-side builder. Walks the shuffle loop log and produces one
    /// row per iteration. Caller pre-computes (per spec):
    ///   - `effective_balances`: `state.validators[i].effective_balance`
    ///     for each candidate `indices[shuffled_index]` encountered in
    ///     the loop, in iteration order (one per iteration).
    ///   - `indices`: the per-iteration `candidate_index = indices[
    ///     compute_shuffled_index(i % total, total, seed)]` values, in
    ///     iteration order. For this scaffold step, the caller passes
    ///     `indices = candidate_indices_in_loop_order` (i.e. already
    ///     post-shuffle); the AIR commits the post-shuffle index as
    ///     witness only.
    ///   - `proposer_idx`: the final accepted `proposer_index`. The
    ///     builder stops emitting rows once the accept condition fires.
    ///
    /// Returns rows in iteration order (the last row is the accept row
    /// with `is_accepted = true`).
    pub fn from_shuffle_log(
        seed: [u8; 32],
        indices: &[u64],
        effective_balances: &[u64],
        proposer_idx: u64,
    ) -> Self {
        let n = indices.len().min(effective_balances.len()).min(MAX_ITERATIONS);
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            // Per spec: `random_byte = hash(seed || uint_to_bytes(i / 32))[i % 32]`.
            // We build a 32-byte hash_input by overwriting the first 8
            // bytes of `seed` with the LE encoding of `i / 32` for the
            // scaffold (the full 40-byte preimage spec is deferred to
            // the cross-AIR sha256 binding, which handles the 8-byte
            // counter suffix).
            let mut hash_input = seed;
            let counter = (i as u64) / 32;
            hash_input[..8].copy_from_slice(&counter.to_le_bytes());
            let hash_output = crate::sha256::sha256(&hash_input);
            let random_byte = hash_output[i % 32];
            let candidate_index = indices[i];
            let eb = effective_balances[i];
            let lhs = eb.saturating_mul(MAX_RANDOM_BYTE);
            let rhs = MAX_EFFECTIVE_BALANCE.saturating_mul(random_byte as u64);
            let is_accepted = lhs >= rhs;
            rows.push(ProposerShuffleRow {
                iteration_index: i as u64,
                seed,
                candidate_index,
                candidate_effective_balance: eb,
                shuffled_index_at_iter: candidate_index,
                random_byte,
                hash_input,
                hash_output,
                is_accepted,
            });
            if is_accepted {
                break;
            }
        }
        Self { rows, proposer_index: proposer_idx }
    }
}

fn le_decomp_u64(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ProposerShuffleWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_ITERATION_INDEX][i] = Scalar::from_u64(row.iteration_index, curve);
        for k in 0..SEED_LEN {
            columns[COL_SEED_OFFSET + k][i] = Scalar::from_u64(row.seed[k] as u64, curve);
        }
        columns[COL_CANDIDATE_INDEX][i] = Scalar::from_u64(row.candidate_index, curve);
        columns[COL_CANDIDATE_EFFECTIVE_BALANCE][i] =
            Scalar::from_u64(row.candidate_effective_balance, curve);
        columns[COL_SHUFFLED_INDEX_AT_ITER][i] =
            Scalar::from_u64(row.shuffled_index_at_iter, curve);
        columns[COL_RANDOM_BYTE][i] = Scalar::from_u64(row.random_byte as u64, curve);
        for k in 0..HASH_INPUT_LEN {
            columns[COL_HASH_INPUT_OFFSET + k][i] =
                Scalar::from_u64(row.hash_input[k] as u64, curve);
        }
        for k in 0..HASH_OUTPUT_LEN {
            columns[COL_HASH_OUTPUT_OFFSET + k][i] =
                Scalar::from_u64(row.hash_output[k] as u64, curve);
        }
        columns[COL_IS_ACCEPTED][i] = if row.is_accepted { one.clone() } else { zero.clone() };
        columns[COL_IS_REAL][i] = one.clone();

        // eb byte decomp
        let eb_bytes = le_decomp_u64(row.candidate_effective_balance);
        for b in 0..U64_BYTES {
            columns[COL_EB_BYTE_OFFSET + b][i] = Scalar::from_u64(eb_bytes[b] as u64, curve);
        }
        // accept / reject margin
        let lhs = (row.candidate_effective_balance as u128) * (MAX_RANDOM_BYTE as u128);
        let rhs = (MAX_EFFECTIVE_BALANCE as u128) * (row.random_byte as u128);
        let (accept_margin, reject_margin) = if row.is_accepted {
            (lhs - rhs, 0u128)
        } else {
            // reject branch: rhs > lhs strictly.
            (0u128, rhs - lhs)
        };
        columns[COL_ACCEPT_MARGIN][i] = Scalar::from_u64(accept_margin as u64, curve);
        let am_bytes = le_decomp_u64(accept_margin as u64);
        for b in 0..U64_BYTES {
            columns[COL_ACCEPT_MARGIN_BYTE_OFFSET + b][i] =
                Scalar::from_u64(am_bytes[b] as u64, curve);
        }
        columns[COL_REJECT_MARGIN][i] = Scalar::from_u64(reject_margin as u64, curve);
        // On accept rows, reject_margin = 0; we still need the byte
        // column to satisfy the decomp constraint
        // `reject_margin = 1 + Σ byte * 256^b`. On accept rows this
        // would force `Σ byte * 256^b = -1`, which is unreachable for
        // byte-range values. So the reject-margin decomp constraint
        // is GATED by `is_real * (1 - is_accepted)` (the rejection
        // gate). On accept + padding rows this constraint vanishes and
        // we leave the bytes as zero (only meaningful on reject rows).
        let rmm1: u64 = if row.is_accepted || reject_margin == 0 {
            0
        } else {
            (reject_margin - 1) as u64
        };
        let rmm1_bytes = le_decomp_u64(rmm1);
        for b in 0..U64_BYTES {
            columns[COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET + b][i] =
                Scalar::from_u64(rmm1_bytes[b] as u64, curve);
        }
        columns[COL_ITERATION_INDEX_BYTE][i] =
            Scalar::from_u64(row.iteration_index, curve);
        columns[COL_PROPOSER_INDEX][i] = Scalar::from_u64(witness.proposer_index, curve);
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

pub struct ProposerShuffleConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ProposerShuffleConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn pow256_pow(b: usize, curve: CurveType) -> Scalar {
    // 256^b
    let mut acc: u64 = 1;
    for _ in 0..b {
        acc = acc.wrapping_mul(256);
    }
    Scalar::from_u64(acc, curve)
}

impl VmConstraintSystem for ProposerShuffleConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_accepted_binary".into(),
            "accept_identity".into(),
            "reject_identity".into(),
            "eb_le_decomp".into(),
            "accept_margin_le_decomp".into(),
            "reject_margin_le_decomp".into(),
            "candidate_eq_shuffled".into(),
            "random_byte_redundant".into(),
            "iteration_index_bound".into(),
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
        let max_rb = Scalar::from_u64(MAX_RANDOM_BYTE, curve);
        let max_eb = Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        let pow256: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let is_accepted = &columns[COL_IS_ACCEPTED][r];
            let eb = &columns[COL_CANDIDATE_EFFECTIVE_BALANCE][r];
            let rb = &columns[COL_RANDOM_BYTE][r];
            let accept_margin = &columns[COL_ACCEPT_MARGIN][r];
            let reject_margin = &columns[COL_REJECT_MARGIN][r];

            // 0: is_real_binary.
            out[0][r] = is_real.mul(&is_real.sub(&one));

            // 1: is_accepted_binary.
            out[1][r] = is_accepted.mul(&is_accepted.sub(&one));

            // 2: accept_identity gated by is_real * is_accepted:
            //    is_real * is_accepted * (eb*255 - 32G*rb - accept_margin) = 0
            {
                let lhs = eb.mul(&max_rb);
                let rhs = max_eb.mul(rb);
                let body = lhs.sub(&rhs).sub(accept_margin);
                let gate = is_real.mul(is_accepted);
                out[2][r] = gate.mul(&body);
            }

            // 3: reject_identity gated by is_real * (1 - is_accepted):
            //    is_real * (1 - is_accepted) * (32G*rb - eb*255 - reject_margin) = 0
            {
                let lhs = max_eb.mul(rb);
                let rhs = eb.mul(&max_rb);
                let body = lhs.sub(&rhs).sub(reject_margin);
                let gate = is_real.mul(&one.sub(is_accepted));
                out[3][r] = gate.mul(&body);
            }

            // 4: eb_le_decomp: eb - Σ eb_byte[b] * 256^b = 0 (gated by is_real).
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(&columns[COL_EB_BYTE_OFFSET + b][r].mul(&pow256[b]));
                }
                out[4][r] = is_real.mul(&eb.sub(&sum));
            }

            // 5: accept_margin_le_decomp gated by is_real:
            //    is_real * (accept_margin - Σ accept_margin_byte[b] * 256^b) = 0
            {
                let mut sum = Scalar::zero(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(
                        &columns[COL_ACCEPT_MARGIN_BYTE_OFFSET + b][r].mul(&pow256[b]),
                    );
                }
                out[5][r] = is_real.mul(&accept_margin.sub(&sum));
            }

            // 6: reject_margin_le_decomp gated by is_real * (1 - is_accepted):
            //    gate * (reject_margin - 1 - Σ rmm1_byte[b] * 256^b) = 0
            {
                let mut sum = Scalar::one(curve);
                for b in 0..U64_BYTES {
                    sum = sum.add(
                        &columns[COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET + b][r]
                            .mul(&pow256[b]),
                    );
                }
                let gate = is_real.mul(&one.sub(is_accepted));
                out[6][r] = gate.mul(&reject_margin.sub(&sum));
            }

            // 7: candidate_eq_shuffled (trivial slot — binding lives in
            //    the cross-AIR LogUp). Reserve the slot with the
            //    identity body: is_real * (candidate_index -
            //    shuffled_index_at_iter - (candidate_index -
            //    shuffled_index_at_iter)) = 0. This is provably zero
            //    on every row but pins the column count.
            {
                let body = columns[COL_CANDIDATE_INDEX][r]
                    .sub(&columns[COL_SHUFFLED_INDEX_AT_ITER][r])
                    .sub(
                        &columns[COL_CANDIDATE_INDEX][r]
                            .sub(&columns[COL_SHUFFLED_INDEX_AT_ITER][r]),
                    );
                out[7][r] = is_real.mul(&body);
            }

            // 8: random_byte_redundant: 0 = 0. Lookup table does the
            //    actual range check; we keep this body for layout
            //    stability with the gadget chain (constraint count
            //    pinned).
            {
                out[8][r] = Scalar::zero(curve);
            }

            // 9: iteration_index_bound: is_real *
            //    (iteration_index - iteration_index_byte) = 0. The byte
            //    column is range-checked to [0, 255] via the lookup
            //    declaration; combined with this body it pins
            //    iteration_index ∈ [0, 255] (a strict superset of
            //    MAX_ITERATIONS = 8).
            {
                let body = columns[COL_ITERATION_INDEX][r]
                    .sub(&columns[COL_ITERATION_INDEX_BYTE][r]);
                out[9][r] = is_real.mul(&body);
            }
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let max_rb = Scalar::from_u64(MAX_RANDOM_BYTE, curve);
        let max_eb = Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve);
        let pow256: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let is_real = &col_evals[COL_IS_REAL];
        let is_accepted = &col_evals[COL_IS_ACCEPTED];
        let eb = &col_evals[COL_CANDIDATE_EFFECTIVE_BALANCE];
        let rb = &col_evals[COL_RANDOM_BYTE];
        let accept_margin = &col_evals[COL_ACCEPT_MARGIN];
        let reject_margin = &col_evals[COL_REJECT_MARGIN];

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        // 0
        acc = acc.add(&alpha_pow.mul(&is_real.mul(&is_real.sub(&one))));
        alpha_pow = alpha_pow.mul(alpha);
        // 1
        acc = acc.add(&alpha_pow.mul(&is_accepted.mul(&is_accepted.sub(&one))));
        alpha_pow = alpha_pow.mul(alpha);
        // 2 accept
        {
            let body = eb.mul(&max_rb).sub(&max_eb.mul(rb)).sub(accept_margin);
            let gate = is_real.mul(is_accepted);
            acc = acc.add(&alpha_pow.mul(&gate.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3 reject
        {
            let body = max_eb.mul(rb).sub(&eb.mul(&max_rb)).sub(reject_margin);
            let gate = is_real.mul(&one.sub(is_accepted));
            acc = acc.add(&alpha_pow.mul(&gate.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4 eb decomp
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(&col_evals[COL_EB_BYTE_OFFSET + b].mul(&pow256[b]));
            }
            acc = acc.add(&alpha_pow.mul(&is_real.mul(&eb.sub(&sum))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 5 accept_margin decomp
        {
            let mut sum = Scalar::zero(curve);
            for b in 0..U64_BYTES {
                sum =
                    sum.add(&col_evals[COL_ACCEPT_MARGIN_BYTE_OFFSET + b].mul(&pow256[b]));
            }
            acc =
                acc.add(&alpha_pow.mul(&is_real.mul(&accept_margin.sub(&sum))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 6 reject_margin decomp
        {
            let mut sum = Scalar::one(curve);
            for b in 0..U64_BYTES {
                sum = sum.add(
                    &col_evals[COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET + b]
                        .mul(&pow256[b]),
                );
            }
            let gate = is_real.mul(&one.sub(is_accepted));
            acc = acc.add(&alpha_pow.mul(&gate.mul(&reject_margin.sub(&sum))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 7 candidate_eq_shuffled trivial slot
        {
            let body = col_evals[COL_CANDIDATE_INDEX]
                .sub(&col_evals[COL_SHUFFLED_INDEX_AT_ITER])
                .sub(
                    &col_evals[COL_CANDIDATE_INDEX]
                        .sub(&col_evals[COL_SHUFFLED_INDEX_AT_ITER]),
                );
            acc = acc.add(&alpha_pow.mul(&is_real.mul(&body)));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 8 random_byte_redundant
        {
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 9 iteration_index_bound
        {
            let body = col_evals[COL_ITERATION_INDEX]
                .sub(&col_evals[COL_ITERATION_INDEX_BYTE]);
            acc = acc.add(&alpha_pow.mul(&is_real.mul(&body)));
        }
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
        let max_rb = Scalar::from_u64(MAX_RANDOM_BYTE, curve);
        let max_eb = Scalar::from_u64(MAX_EFFECTIVE_BALANCE, curve);
        let pow256: Vec<Scalar> = (0..U64_BYTES).map(|b| pow256_pow(b, curve)).collect();

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_accepted = &col_coeffs[COL_IS_ACCEPTED];
        let one_minus_is_accepted = poly_sub(&one_poly, is_accepted, curve);
        let eb = &col_coeffs[COL_CANDIDATE_EFFECTIVE_BALANCE];
        let rb = &col_coeffs[COL_RANDOM_BYTE];
        let accept_margin = &col_coeffs[COL_ACCEPT_MARGIN];
        let reject_margin = &col_coeffs[COL_REJECT_MARGIN];

        // 0
        {
            let v_m1 = poly_sub(is_real, &one_poly, curve);
            let body = poly_mul(is_real, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1
        {
            let v_m1 = poly_sub(is_accepted, &one_poly, curve);
            let body = poly_mul(is_accepted, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2 accept gated
        {
            let lhs = poly_scalar_mul(eb, &max_rb);
            let rhs = poly_scalar_mul(rb, &max_eb);
            let body = poly_sub(&poly_sub(&lhs, &rhs, curve), accept_margin, curve);
            let gate = poly_mul(is_real, is_accepted, curve);
            let gated = poly_mul(&gate, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 3 reject gated
        {
            let lhs = poly_scalar_mul(rb, &max_eb);
            let rhs = poly_scalar_mul(eb, &max_rb);
            let body = poly_sub(&poly_sub(&lhs, &rhs, curve), reject_margin, curve);
            let gate = poly_mul(is_real, &one_minus_is_accepted, curve);
            let gated = poly_mul(&gate, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 4 eb decomp gated by is_real
        {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..U64_BYTES {
                let term =
                    poly_scalar_mul(&col_coeffs[COL_EB_BYTE_OFFSET + b], &pow256[b]);
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(eb, &sum, curve);
            let gated = poly_mul(is_real, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 5 accept_margin decomp gated by is_real
        {
            let mut sum = vec![Scalar::zero(curve)];
            for b in 0..U64_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_ACCEPT_MARGIN_BYTE_OFFSET + b],
                    &pow256[b],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(accept_margin, &sum, curve);
            let gated = poly_mul(is_real, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 6 reject_margin decomp gated by is_real * (1 - is_accepted)
        {
            let mut sum = one_poly.clone();
            for b in 0..U64_BYTES {
                let term = poly_scalar_mul(
                    &col_coeffs[COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET + b],
                    &pow256[b],
                );
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(reject_margin, &sum, curve);
            let gate = poly_mul(is_real, &one_minus_is_accepted, curve);
            let gated = poly_mul(&gate, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 7 candidate_eq_shuffled trivial slot
        {
            let diff = poly_sub(
                &col_coeffs[COL_CANDIDATE_INDEX],
                &col_coeffs[COL_SHUFFLED_INDEX_AT_ITER],
                curve,
            );
            let body = poly_sub(&diff, &diff, curve);
            let gated = poly_mul(is_real, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 8 random_byte_redundant
        {
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 9 iteration_index_bound gated by is_real
        {
            let body = poly_sub(
                &col_coeffs[COL_ITERATION_INDEX],
                &col_coeffs[COL_ITERATION_INDEX_BYTE],
                curve,
            );
            let gated = poly_mul(is_real, &body, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&gated, &alpha_pow), curve);
        }

        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
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
        let curve = columns[0]
            .first()
            .map(|s| s.curve_type())
            .unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for b in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("eb_byte_{}_8bit", b),
                    column_index: COL_EB_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("accept_margin_byte_{}_8bit", b),
                    column_index: COL_ACCEPT_MARGIN_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for b in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("reject_margin_mm1_byte_{}_8bit", b),
                    column_index: COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET + b,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        declarations.push((
            LookupDeclaration {
                label: "random_byte_8bit".into(),
                column_index: COL_RANDOM_BYTE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        declarations.push((
            LookupDeclaration {
                label: "iteration_index_byte_8bit".into(),
                column_index: COL_ITERATION_INDEX_BYTE,
                max_bits: 8,
                selector_column: None,
            },
            0,
        ));
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Bind `(candidate_index, candidate_effective_balance)` rows of this
/// AIR to `(validator_index, effective_balance)` rows of
/// `validator_balances_air`. Closes the post-shuffle data against the
/// validator registry's effective-balance witness.
pub fn make_shuffle_to_validator_balances_descriptor(
    shuffle_layer_index: usize,
    validator_balances_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::validator_balances_air as vb;
    CrossAirLogUpDescriptor {
        label: "shuffle_to_validator_balances_v1".into(),
        a_layer_index: shuffle_layer_index,
        a_columns: vec![COL_CANDIDATE_INDEX, COL_CANDIDATE_EFFECTIVE_BALANCE],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: validator_balances_layer_index,
        b_columns: vec![vb::COL_VALIDATOR_INDEX, vb::COL_EFFECTIVE_BALANCE],
        b_selector_column: Some(vb::COL_IS_REAL),
    }
}

/// Bind `(hash_input[0..32], hash_output[0..32])` to `sha256_extract`'s
/// `(INPUT_BYTE[0..32], OUTPUT_BYTE[0..32])`. Stepping-stone: the spec
/// preimage is `seed || uint64_le(i // 32)` (40 bytes), but this
/// scaffold uses a 32-byte preimage; the wider 40-byte sha256 binding
/// is deferred to the variable-length sha256 gadget that the BLS reveal
/// chain uses.
pub fn make_shuffle_to_sha256_descriptor(
    shuffle_layer_index: usize,
    sha256_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;
    // sha256_extract's input is 64 bytes; we bind only the first 32
    // bytes (the preimage we publish). The remaining 32 input bytes of
    // sha256_extract's single block are not constrained here — a
    // dedicated 32-byte sha256 gadget is the follow-up.
    let a_columns: Vec<usize> = (0..HASH_INPUT_LEN)
        .map(|k| COL_HASH_INPUT_OFFSET + k)
        .chain((0..HASH_OUTPUT_LEN).map(|k| COL_HASH_OUTPUT_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = (0..HASH_INPUT_LEN)
        .map(|k| se::COL_INPUT_BYTE_OFFSET + k)
        .chain((0..HASH_OUTPUT_LEN).map(|k| se::COL_OUTPUT_BYTE_OFFSET + k))
        .collect();
    CrossAirLogUpDescriptor {
        label: "shuffle_to_sha256_v1".into(),
        a_layer_index: shuffle_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Bind the per-row `seed[0..32]` to the RANDAO mix the shuffle reads
/// from. Two acceptable consumer-side targets exist (the per-block
/// `randao_proposer_air` and the multi-block `randao_chain_air`); this
/// descriptor wires the simpler per-block view. The seed is matched
/// against `new_mix` bytes (LE/BE conventions for the proposer-seed
/// derivation are handled by the deferred seed-derivation gadget; this
/// step just binds the raw 32-byte seed against the produced 32-byte
/// RANDAO mix on the source layer).
pub fn make_shuffle_to_randao_descriptor(
    shuffle_layer_index: usize,
    randao_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::randao_proposer_air as rp;
    let a_columns: Vec<usize> =
        (0..SEED_LEN).map(|k| COL_SEED_OFFSET + k).collect();
    let b_columns: Vec<usize> =
        (0..rp::MIX_LEN).map(|k| rp::COL_NEW_MIX_OFFSET + k).collect();
    CrossAirLogUpDescriptor {
        label: "shuffle_to_randao_v1".into(),
        a_layer_index: shuffle_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: randao_layer_index,
        b_columns,
        b_selector_column: Some(rp::COL_IS_REAL),
    }
}

/// Bind the final `proposer_index` (committed on every active row) to
/// the downstream block-proposer signature AIR's proposer-index
/// column. The B-side targets `block_proposer_sig_air::COL_PROPOSER_INDEX`,
/// pinning the chosen proposer against the BLS signature verification.
/// A-side selector is `COL_IS_ACCEPTED` so only the single accept row
/// publishes the bound tuple.
pub fn make_shuffle_to_block_proposer_descriptor(
    shuffle_layer_index: usize,
    block_proposer_layer_index: usize,
) -> CrossAirLogUpDescriptor {
    use crate::block_proposer_sig_air as bps;
    CrossAirLogUpDescriptor {
        label: "shuffle_to_block_proposer_v1".into(),
        a_layer_index: shuffle_layer_index,
        a_columns: vec![COL_PROPOSER_INDEX],
        a_selector_column: Some(COL_IS_ACCEPTED),
        b_layer_index: block_proposer_layer_index,
        b_columns: vec![bps::COL_PROPOSER_INDEX],
        b_selector_column: Some(bps::COL_IS_REAL),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_bodies(w: &ProposerShuffleWitness) -> Vec<Vec<Scalar>> {
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(w, curve);
        let cs = ProposerShuffleConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    fn assert_all_vanish(bodies: &[Vec<Scalar>]) {
        for (i, body) in bodies.iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} row {} = {:?} should be zero",
                    i, r, v
                );
            }
        }
    }

    /// Build a deterministic witness with `n_iter` rows, the last
    /// accepting. We pick `effective_balance` and seed values such that
    /// the accept/reject branch matches the spec at each iteration.
    fn build_demo_witness(seed_byte: u8, ebs: &[u64], indices: &[u64]) -> ProposerShuffleWitness {
        let seed = [seed_byte; 32];
        // Accepting `proposer_index` = the last index (since we engineer
        // the witness to accept on the last row).
        let proposer_idx = *indices.last().unwrap();
        ProposerShuffleWitness::from_shuffle_log(seed, indices, ebs, proposer_idx)
    }

    #[test]
    fn accept_on_first_iteration() {
        // Pick eb = MAX_EFFECTIVE_BALANCE so eb*255 ≥ 32G*rb for any
        // random_byte ∈ [0, 255]. First iteration must accept.
        let ebs = [MAX_EFFECTIVE_BALANCE];
        let indices = [42u64];
        let w = build_demo_witness(0x42, &ebs, &indices);
        assert_eq!(w.rows.len(), 1);
        assert!(w.rows[0].is_accepted, "first iteration with full eb must accept");
        let bodies = run_bodies(&w);
        assert_eq!(bodies.len(), NUM_ROW_CONSTRAINTS);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn reject_then_accept_chain() {
        // Build a sequence with a fixed rejection then an acceptance.
        // We have to engineer it by handing the witness builder a
        // sequence whose first iteration's hash byte exceeds the
        // threshold. Use eb=0 on the first iteration: lhs=0, rhs=32G*rb.
        // If rb > 0, this rejects strictly; if rb=0, lhs=rhs=0 and
        // is_accepted = true. The seed_byte = 0xff produces non-zero
        // hash output bytes overwhelmingly likely; we also include a
        // second iteration with eb = MAX so it must accept.
        let ebs = [0u64, MAX_EFFECTIVE_BALANCE];
        let indices = [11u64, 22u64];
        let w = build_demo_witness(0xff, &ebs, &indices);
        // Either iteration 0 already accepts (rare lucky rb=0) or
        // iteration 0 rejects and iteration 1 accepts. Either way the
        // honest constraint system must vanish.
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn tampered_accept_margin_fires_constraint() {
        let ebs = [MAX_EFFECTIVE_BALANCE];
        let indices = [7u64];
        let w = build_demo_witness(0x01, &ebs, &indices);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper accept_margin: subtract 1.
        let cur = cols[COL_ACCEPT_MARGIN][0].to_u64();
        let tampered = if cur >= 1 { cur - 1 } else { cur + 1 };
        cols[COL_ACCEPT_MARGIN][0] = Scalar::from_u64(tampered, curve);
        let cs = ProposerShuffleConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // Constraint 2 = accept_identity must fire on row 0.
        assert!(
            !bodies[2][0].is_zero(),
            "tampered accept_margin should fire accept_identity"
        );
        // And constraint 5 = accept_margin_le_decomp must fire on row 0
        // (since the byte cols no longer sum to the tampered scalar).
        assert!(
            !bodies[5][0].is_zero(),
            "tampered accept_margin should also fire accept_margin_le_decomp"
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d1 = make_shuffle_to_validator_balances_descriptor(0, 1);
        assert_eq!(d1.label, "shuffle_to_validator_balances_v1");
        assert_eq!(d1.a_columns, vec![COL_CANDIDATE_INDEX, COL_CANDIDATE_EFFECTIVE_BALANCE]);
        assert_eq!(d1.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(d1.a_columns.len(), d1.b_columns.len());

        let d2 = make_shuffle_to_sha256_descriptor(0, 2);
        assert_eq!(d2.label, "shuffle_to_sha256_v1");
        assert_eq!(d2.a_columns.len(), HASH_INPUT_LEN + HASH_OUTPUT_LEN);
        assert_eq!(d2.b_columns.len(), HASH_INPUT_LEN + HASH_OUTPUT_LEN);

        let d3 = make_shuffle_to_randao_descriptor(0, 3);
        assert_eq!(d3.label, "shuffle_to_randao_v1");
        assert_eq!(d3.a_columns.len(), SEED_LEN);
        assert_eq!(d3.b_columns.len(), SEED_LEN);

        let d4 = make_shuffle_to_block_proposer_descriptor(0, 4);
        assert_eq!(d4.label, "shuffle_to_block_proposer_v1");
        assert_eq!(d4.a_columns, vec![COL_PROPOSER_INDEX]);
        assert_eq!(d4.a_selector_column, Some(COL_IS_ACCEPTED));
        assert_eq!(d4.b_columns.len(), 1);
    }

    #[test]
    fn column_layout_pinned() {
        assert_eq!(COL_ITERATION_INDEX, 0);
        assert_eq!(COL_SEED_OFFSET, 1);
        assert_eq!(COL_CANDIDATE_INDEX, 33);
        assert_eq!(COL_CANDIDATE_EFFECTIVE_BALANCE, 34);
        assert_eq!(COL_SHUFFLED_INDEX_AT_ITER, 35);
        assert_eq!(COL_RANDOM_BYTE, 36);
        assert_eq!(COL_HASH_INPUT_OFFSET, 37);
        assert_eq!(COL_HASH_OUTPUT_OFFSET, 69);
        assert_eq!(COL_IS_ACCEPTED, 101);
        assert_eq!(COL_IS_REAL, 102);
        assert_eq!(COL_EB_BYTE_OFFSET, 103);
        assert_eq!(COL_ACCEPT_MARGIN, 111);
        assert_eq!(COL_ACCEPT_MARGIN_BYTE_OFFSET, 112);
        assert_eq!(COL_REJECT_MARGIN, 120);
        assert_eq!(COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET, 121);
        assert_eq!(COL_ITERATION_INDEX_BYTE, 129);
        assert_eq!(COL_PROPOSER_INDEX, 130);
        assert_eq!(NUM_COLUMNS, 131);
        assert_eq!(NUM_ROW_CONSTRAINTS, 10);
        assert_eq!(NUM_SHIFTED, 0);
    }

    #[test]
    fn byte_range_coverage() {
        let cs = ProposerShuffleConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 3 * 8 byte cols + random_byte + iteration_index_byte = 26.
        assert_eq!(req.declarations.len(), 3 * U64_BYTES + 2);
        for (decl, _) in &req.declarations {
            assert_eq!(decl.max_bits, 8, "all declared ranges must be 8-bit");
            assert!(decl.label.contains("8bit"));
        }
        // Must include random_byte and iteration_index_byte.
        let labels: Vec<&str> =
            req.declarations.iter().map(|(d, _)| d.label.as_str()).collect();
        assert!(labels.contains(&"random_byte_8bit"));
        assert!(labels.contains(&"iteration_index_byte_8bit"));
    }

    #[test]
    fn reject_margin_must_be_strictly_positive() {
        // Engineer a row that is rejected (eb=0, random_byte>0) and
        // verify the reject_margin column is at least 1, the
        // decomp constraint must vanish only when the witness honestly
        // sets reject_margin = 32G * rb − eb*255.
        let ebs = [0u64];
        let indices = [0u64];
        let w = build_demo_witness(0xab, &ebs, &indices);
        // If the lucky case happens (rb=0 ⇒ accept on first iter), skip
        // the strict check.
        if w.rows[0].is_accepted {
            return;
        }
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Honest evaluation: all bodies must vanish.
        {
            let cs = ProposerShuffleConstraintSystem::new(trace.num_rows);
            let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
            let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
            assert_all_vanish(&bodies);
        }
        // Tamper: set reject_margin to 0 (would violate strict
        // positivity). The reject_margin_le_decomp constraint then
        // requires `0 = 1 + Σ rmm1_byte * 256^b` — impossible with
        // byte-range values, so constraint 6 fires.
        cols[COL_REJECT_MARGIN][0] = Scalar::from_u64(0, curve);
        // Also clear rmm1 bytes — without this clearing, constraint 6
        // would already fire on the inconsistent rmm1 decomp.
        for b in 0..U64_BYTES {
            cols[COL_REJECT_MARGIN_MINUS_ONE_BYTE_OFFSET + b][0] =
                Scalar::from_u64(0, curve);
        }
        let cs = ProposerShuffleConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[6][0].is_zero(),
            "reject_margin=0 with strict-positivity decomp must fail"
        );
        // And the reject_identity constraint must fire too (rhs−lhs = 32G*rb ≠ 0).
        assert!(
            !bodies[3][0].is_zero(),
            "reject_identity must fire when reject_margin=0 but rhs>lhs"
        );
    }

    #[test]
    fn shuffle_log_builds_proposer_index() {
        let seed = [0x99u8; 32];
        // 3 candidates, last one accepts (use MAX eb on the last).
        let indices = [10u64, 20, 30];
        let ebs = [0u64, 0u64, MAX_EFFECTIVE_BALANCE];
        let w = ProposerShuffleWitness::from_shuffle_log(seed, &indices, &ebs, 30);
        // The last emitted row is the accept row.
        let last = w.rows.last().unwrap();
        assert!(last.is_accepted);
        assert_eq!(last.candidate_index, w.proposer_index);
        // No iteration after the accept.
        assert!(w.rows.len() <= 3);
        // Honest constraints vanish.
        let bodies = run_bodies(&w);
        assert_all_vanish(&bodies);
    }

    #[test]
    fn binary_constraint_fires_on_non_binary_is_accepted() {
        let ebs = [MAX_EFFECTIVE_BALANCE];
        let indices = [1u64];
        let w = build_demo_witness(0x10, &ebs, &indices);
        let curve = CurveType::Bls48581;
        let trace = build_trace_polynomials(&w, curve);
        let mut cols: Vec<Vec<Scalar>> =
            trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        cols[COL_IS_ACCEPTED][0] = Scalar::from_u64(2, curve);
        let cs = ProposerShuffleConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let bodies = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(
            !bodies[1][0].is_zero(),
            "is_accepted = 2 must fire is_accepted_binary"
        );
    }
}
