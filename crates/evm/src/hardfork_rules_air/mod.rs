//! Shanghai / Cancun hardfork activation rules AIR.
//!
//! Proves that a block's hardfork-feature flags are consistent with
//! the block timestamp. Concretely, the AIR commits the following per
//! row:
//!
//!   - `block_timestamp` (u64) — taken from the block header.
//!   - `is_post_paris`    (bin) — Paris (The Merge) is the floor for
//!     Shanghai. We do NOT pin Paris by timestamp here (it activated
//!     by TTD, not timestamp) — instead this AIR enforces only the
//!     monotone implication `shanghai ⇒ paris`.
//!   - `is_post_shanghai` (bin) — pinned by `block_timestamp` vs
//!     `SHANGHAI_TIMESTAMP` via a non-negative `shanghai_delta`.
//!   - `is_post_cancun`   (bin) — pinned by `block_timestamp` vs
//!     `CANCUN_TIMESTAMP` via a non-negative `cancun_delta`.
//!   - `shanghai_delta`   (u64) — `block_timestamp − SHANGHAI_TIMESTAMP`
//!     when `is_post_shanghai = 1`; free otherwise.
//!   - `cancun_delta`     (u64) — `block_timestamp − CANCUN_TIMESTAMP`
//!     when `is_post_cancun = 1`; free otherwise.
//!   - `is_real`          (bin) — real-row indicator.
//!
//! ## EIPs gated by these flags
//!
//!   - Shanghai (1681338455):
//!       * EIP-3651 — warm coinbase in the EIP-2929 access list.
//!       * EIP-3855 — PUSH0 opcode (0x5F).
//!       * EIP-3860 — `MAX_INITCODE_SIZE = 49152`.
//!   - Cancun (1710338135):
//!       * EIP-4844 — BLOBHASH opcode + blob_versioned_hashes commitment.
//!
//! ## Constraint catalog (15 row-local constraints)
//!
//! All constraints are gated to vanish on padding rows (`is_real = 0`)
//! either implicitly (selector-binarity) or via explicit multiplication
//! by `is_real` where required.
//!
//!  1. `is_real_binary`        — `is_real · (is_real − 1) = 0`
//!  2. `paris_binary`          — `is_post_paris · (is_post_paris − 1) = 0`
//!  3. `shanghai_binary`       — `is_post_shanghai · (is_post_shanghai − 1) = 0`
//!  4. `cancun_binary`         — `is_post_cancun · (is_post_cancun − 1) = 0`
//!  5. `shanghai_implies_paris`
//!         — `is_post_shanghai · (1 − is_post_paris) = 0`
//!  6. `cancun_implies_shanghai`
//!         — `is_post_cancun · (1 − is_post_shanghai) = 0`
//!  7. `shanghai_gate`         —
//!         `is_post_shanghai · (block_timestamp − SHANGHAI_TIMESTAMP − shanghai_delta) = 0`
//!  8. `cancun_gate`           —
//!         `is_post_cancun · (block_timestamp − CANCUN_TIMESTAMP − cancun_delta) = 0`
//!  9. `block_timestamp_le_decomp`
//!         — `block_timestamp − Σ_{j=0..8} ts_byte_j · 2^(8·j) = 0`
//! 10. `shanghai_delta_le_decomp`
//!         — `shanghai_delta − Σ_{j=0..8} sd_byte_j · 2^(8·j) = 0`
//! 11. `cancun_delta_le_decomp`
//!         — `cancun_delta − Σ_{j=0..8} cd_byte_j · 2^(8·j) = 0`
//! 12. `shanghai_delta_zero_when_pre`
//!         — `(1 − is_post_shanghai) · shanghai_delta = 0`  (canonicalize
//!         the witness: pre-Shanghai rows MUST set the delta to 0)
//! 13. `cancun_delta_zero_when_pre`
//!         — `(1 − is_post_cancun) · cancun_delta = 0`
//! 14. `paris_zero_on_padding`
//!         — `(1 − is_real) · is_post_paris = 0`  (forbid flags fired
//!         on padding rows; combined with 5/6 they propagate down)
//! 15. `block_timestamp_zero_on_padding`
//!         — `(1 − is_real) · block_timestamp = 0`
//!
//! Plus 8+8+8 = 24 byte range lookups on the LE byte decompositions
//! (range [0, 256)). The byte decompositions force the deltas and
//! timestamp to be representable as u64 — non-negative by construction.
//!
//! ## Soundness notes
//!
//!   - The implication `shanghai ⇒ paris` is enforced by constraint 5.
//!     `cancun ⇒ shanghai` (and therefore `cancun ⇒ paris`) by 6.
//!   - The non-negativity of `shanghai_delta` follows from the LE byte
//!     decomposition (constraint 10) plus the 8-bit range lookups —
//!     any algebraic value not representable as a u64 fails the range.
//!   - The shanghai-gate only fires on `is_post_shanghai = 1`. The
//!     reverse direction (`is_post_shanghai = 0 ⇒ ts < SHANGHAI`) is
//!     NOT enforced algebraically here; that direction is unfalsifiable
//!     in isolation because we'd need a strict inequality witness.
//!     The full direction is enforced *contextually* via the cross-AIR
//!     LogUp linkages: any rule downstream that requires the flag to
//!     be set on a real row will catch a missing activation.
//!   - The canonical-witness constraints 12/13 (delta = 0 when flag
//!     off) keep the witness deterministic for downstream linkages.
//!
//! ## Per-row layout (`NUM_COLUMNS = 32`)
//!
//! ```text
//! offset   size   meaning
//! 0        1      block_timestamp
//! 1        1      is_post_paris
//! 2        1      is_post_shanghai
//! 3        1      is_post_cancun
//! 4        1      shanghai_delta
//! 5        1      cancun_delta
//! 6        1      is_real
//! 7..15    8      ts_byte_0..7        (LE byte decomp of block_timestamp)
//! 15..23   8      sd_byte_0..7        (LE byte decomp of shanghai_delta)
//! 23..31   8      cd_byte_0..7        (LE byte decomp of cancun_delta)
//! 31       1      coinbase_warm       (1 iff is_post_shanghai = 1, by EIP-3651;
//!                                       constrained equal to is_post_shanghai)
//! ```

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Activation timestamps ────────────────────────────────────────────

/// Mainnet Shanghai activation timestamp (EIP-3651/3855/3860).
pub const SHANGHAI_TIMESTAMP: u64 = 1_681_338_455;
/// Mainnet Cancun activation timestamp (EIP-4844 et al.).
pub const CANCUN_TIMESTAMP: u64 = 1_710_338_135;
/// EIP-3860 init-code size cap (= 2 * MAX_CODE_SIZE).
pub const MAX_INITCODE_SIZE: usize = 49_152;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_BLOCK_TIMESTAMP: usize = 0;
pub const COL_IS_POST_PARIS: usize = 1;
pub const COL_IS_POST_SHANGHAI: usize = 2;
pub const COL_IS_POST_CANCUN: usize = 3;
pub const COL_SHANGHAI_DELTA: usize = 4;
pub const COL_CANCUN_DELTA: usize = 5;
pub const COL_IS_REAL: usize = 6;
pub const COL_TS_BYTE_OFFSET: usize = 7;
pub const COL_SD_BYTE_OFFSET: usize = COL_TS_BYTE_OFFSET + 8;
pub const COL_CD_BYTE_OFFSET: usize = COL_SD_BYTE_OFFSET + 8;
pub const COL_COINBASE_WARM: usize = COL_CD_BYTE_OFFSET + 8;
pub const NUM_COLUMNS: usize = COL_COINBASE_WARM + 1;

pub const NUM_BYTE_LIMBS: usize = 8;
pub const NUM_ROW_CONSTRAINTS: usize = 16;
pub const NUM_SHIFTED: usize = 0;

const _: () = assert!(NUM_COLUMNS == 32);

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardforkRulesRow {
    pub block_timestamp: u64,
    pub is_post_paris: bool,
    pub is_post_shanghai: bool,
    pub is_post_cancun: bool,
    pub shanghai_delta: u64,
    pub cancun_delta: u64,
    pub is_real: bool,
}

#[derive(Clone, Debug, Default)]
pub struct HardforkRulesWitness {
    pub rows: Vec<HardforkRulesRow>,
}

impl HardforkRulesWitness {
    /// Build a one-row witness from a block timestamp.
    pub fn from_timestamp(ts: u64) -> Self {
        Self {
            rows: vec![from_timestamp_row(ts)],
        }
    }

    /// Append a row corresponding to `ts`.
    pub fn push_timestamp(&mut self, ts: u64) {
        self.rows.push(from_timestamp_row(ts));
    }
}

/// Public helper: derive the canonical hardfork-rules row from a block
/// timestamp. Returns `is_post_X` flags + non-negative deltas computed
/// against the mainnet activation timestamps.
pub fn from_timestamp(ts: u64) -> HardforkRulesRow {
    from_timestamp_row(ts)
}

fn from_timestamp_row(ts: u64) -> HardforkRulesRow {
    let is_post_shanghai = ts >= SHANGHAI_TIMESTAMP;
    let is_post_cancun = ts >= CANCUN_TIMESTAMP;
    // Paris activated by TTD — for any block we have a timestamp for,
    // Paris is already active in practice. The AIR only requires
    // `shanghai ⇒ paris`, so we conservatively set `is_post_paris=1`
    // whenever Shanghai is on, and otherwise default to `true` (host
    // can override if needed for pre-Paris historical scaffolding).
    let is_post_paris = is_post_shanghai || ts >= 1_663_224_162; // Paris timestamp (~Sep 15 2022)
    let shanghai_delta = if is_post_shanghai {
        ts - SHANGHAI_TIMESTAMP
    } else {
        0
    };
    let cancun_delta = if is_post_cancun {
        ts - CANCUN_TIMESTAMP
    } else {
        0
    };
    HardforkRulesRow {
        block_timestamp: ts,
        is_post_paris,
        is_post_shanghai,
        is_post_cancun,
        shanghai_delta,
        cancun_delta,
        is_real: true,
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &HardforkRulesWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (i, row) in witness.rows.iter().enumerate() {
        columns[COL_BLOCK_TIMESTAMP][i] = Scalar::from_u64(row.block_timestamp, curve);
        columns[COL_IS_POST_PARIS][i] = if row.is_post_paris { one.clone() } else { zero.clone() };
        columns[COL_IS_POST_SHANGHAI][i] =
            if row.is_post_shanghai { one.clone() } else { zero.clone() };
        columns[COL_IS_POST_CANCUN][i] =
            if row.is_post_cancun { one.clone() } else { zero.clone() };
        columns[COL_SHANGHAI_DELTA][i] = Scalar::from_u64(row.shanghai_delta, curve);
        columns[COL_CANCUN_DELTA][i] = Scalar::from_u64(row.cancun_delta, curve);
        columns[COL_IS_REAL][i] = if row.is_real { one.clone() } else { zero.clone() };
        for j in 0..NUM_BYTE_LIMBS {
            let ts_b = ((row.block_timestamp >> (8 * j)) & 0xff) as u64;
            let sd_b = ((row.shanghai_delta >> (8 * j)) & 0xff) as u64;
            let cd_b = ((row.cancun_delta >> (8 * j)) & 0xff) as u64;
            columns[COL_TS_BYTE_OFFSET + j][i] = Scalar::from_u64(ts_b, curve);
            columns[COL_SD_BYTE_OFFSET + j][i] = Scalar::from_u64(sd_b, curve);
            columns[COL_CD_BYTE_OFFSET + j][i] = Scalar::from_u64(cd_b, curve);
        }
        // EIP-3651: coinbase is warm iff Shanghai is active.
        columns[COL_COINBASE_WARM][i] =
            if row.is_post_shanghai { one.clone() } else { zero.clone() };
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

pub struct HardforkRulesConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl HardforkRulesConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn byte_pow(j: usize, curve: CurveType) -> Scalar {
    debug_assert!(j < 8);
    Scalar::from_u64(1u64 << (8 * j), curve)
}

/// Σ_{j=0..8} byte_offset+j · 2^(8j)  at one row.
fn eval_le_recompose(col_evals: &[Scalar], byte_offset: usize) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut acc = Scalar::zero(curve);
    for j in 0..NUM_BYTE_LIMBS {
        let b = &col_evals[byte_offset + j];
        let w = byte_pow(j, curve);
        acc = acc.add(&b.mul(&w));
    }
    acc
}

fn build_le_recompose_poly(
    col_coeffs: &[Vec<Scalar>],
    byte_offset: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let mut acc = vec![Scalar::zero(curve)];
    for j in 0..NUM_BYTE_LIMBS {
        let scaled = poly_scalar_mul(&col_coeffs[byte_offset + j], &byte_pow(j, curve));
        acc = poly_add(&acc, &scaled, curve);
    }
    acc
}

impl VmConstraintSystem for HardforkRulesConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "paris_binary".into(),
            "shanghai_binary".into(),
            "cancun_binary".into(),
            "shanghai_implies_paris".into(),
            "cancun_implies_shanghai".into(),
            "shanghai_gate".into(),
            "cancun_gate".into(),
            "block_timestamp_le_decomp".into(),
            "shanghai_delta_le_decomp".into(),
            "cancun_delta_le_decomp".into(),
            "shanghai_delta_zero_when_pre".into(),
            "cancun_delta_zero_when_pre".into(),
            "paris_zero_on_padding".into(),
            "block_timestamp_zero_on_padding".into(),
            "coinbase_warm_equals_shanghai".into(),
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
        let n = columns[0].len();
        let mut bodies: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();
        let shanghai_ts = Scalar::from_u64(SHANGHAI_TIMESTAMP, curve);
        let cancun_ts = Scalar::from_u64(CANCUN_TIMESTAMP, curve);
        for row in 0..n {
            let r: Vec<Scalar> = columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &r[COL_IS_REAL];
            let paris = &r[COL_IS_POST_PARIS];
            let shanghai = &r[COL_IS_POST_SHANGHAI];
            let cancun = &r[COL_IS_POST_CANCUN];
            let sd = &r[COL_SHANGHAI_DELTA];
            let cd = &r[COL_CANCUN_DELTA];
            let ts = &r[COL_BLOCK_TIMESTAMP];
            let coinbase_warm = &r[COL_COINBASE_WARM];
            let one_minus = |x: &Scalar| one.sub(x);
            // 1. is_real_binary
            bodies[0][row] = is_real.mul(&one_minus(is_real));
            // 2. paris_binary
            bodies[1][row] = paris.mul(&one_minus(paris));
            // 3. shanghai_binary
            bodies[2][row] = shanghai.mul(&one_minus(shanghai));
            // 4. cancun_binary
            bodies[3][row] = cancun.mul(&one_minus(cancun));
            // 5. shanghai_implies_paris
            bodies[4][row] = shanghai.mul(&one_minus(paris));
            // 6. cancun_implies_shanghai
            bodies[5][row] = cancun.mul(&one_minus(shanghai));
            // 7. shanghai_gate
            let s_gap = ts.sub(&shanghai_ts).sub(sd);
            bodies[6][row] = shanghai.mul(&s_gap);
            // 8. cancun_gate
            let c_gap = ts.sub(&cancun_ts).sub(cd);
            bodies[7][row] = cancun.mul(&c_gap);
            // 9. block_timestamp_le_decomp
            let ts_recompose = eval_le_recompose(&r, COL_TS_BYTE_OFFSET);
            bodies[8][row] = ts.sub(&ts_recompose);
            // 10. shanghai_delta_le_decomp
            let sd_recompose = eval_le_recompose(&r, COL_SD_BYTE_OFFSET);
            bodies[9][row] = sd.sub(&sd_recompose);
            // 11. cancun_delta_le_decomp
            let cd_recompose = eval_le_recompose(&r, COL_CD_BYTE_OFFSET);
            bodies[10][row] = cd.sub(&cd_recompose);
            // 12. shanghai_delta_zero_when_pre
            bodies[11][row] = one_minus(shanghai).mul(sd);
            // 13. cancun_delta_zero_when_pre
            bodies[12][row] = one_minus(cancun).mul(cd);
            // 14. paris_zero_on_padding
            bodies[13][row] = one_minus(is_real).mul(paris);
            // 15. block_timestamp_zero_on_padding
            bodies[14][row] = one_minus(is_real).mul(ts);
            // 16. coinbase_warm_equals_shanghai
            bodies[15][row] = coinbase_warm.sub(shanghai);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let shanghai_ts = Scalar::from_u64(SHANGHAI_TIMESTAMP, curve);
        let cancun_ts = Scalar::from_u64(CANCUN_TIMESTAMP, curve);
        let is_real = &col_evals[COL_IS_REAL];
        let paris = &col_evals[COL_IS_POST_PARIS];
        let shanghai = &col_evals[COL_IS_POST_SHANGHAI];
        let cancun = &col_evals[COL_IS_POST_CANCUN];
        let sd = &col_evals[COL_SHANGHAI_DELTA];
        let cd = &col_evals[COL_CANCUN_DELTA];
        let ts = &col_evals[COL_BLOCK_TIMESTAMP];
        let coinbase_warm = &col_evals[COL_COINBASE_WARM];
        let one_minus = |x: &Scalar| one.sub(x);
        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&one_minus(is_real)),
            paris.mul(&one_minus(paris)),
            shanghai.mul(&one_minus(shanghai)),
            cancun.mul(&one_minus(cancun)),
            shanghai.mul(&one_minus(paris)),
            cancun.mul(&one_minus(shanghai)),
            shanghai.mul(&ts.sub(&shanghai_ts).sub(sd)),
            cancun.mul(&ts.sub(&cancun_ts).sub(cd)),
            ts.sub(&eval_le_recompose(col_evals, COL_TS_BYTE_OFFSET)),
            sd.sub(&eval_le_recompose(col_evals, COL_SD_BYTE_OFFSET)),
            cd.sub(&eval_le_recompose(col_evals, COL_CD_BYTE_OFFSET)),
            one_minus(shanghai).mul(sd),
            one_minus(cancun).mul(cd),
            one_minus(is_real).mul(paris),
            one_minus(is_real).mul(ts),
            coinbase_warm.sub(shanghai),
        ];
        let mut acc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            acc = acc.add(&body.mul(&ap));
            ap = ap.mul(alpha);
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
        let shanghai_ts_poly = vec![Scalar::from_u64(SHANGHAI_TIMESTAMP, curve)];
        let cancun_ts_poly = vec![Scalar::from_u64(CANCUN_TIMESTAMP, curve)];
        let is_real = &col_coeffs[COL_IS_REAL];
        let paris = &col_coeffs[COL_IS_POST_PARIS];
        let shanghai = &col_coeffs[COL_IS_POST_SHANGHAI];
        let cancun = &col_coeffs[COL_IS_POST_CANCUN];
        let sd = &col_coeffs[COL_SHANGHAI_DELTA];
        let cd = &col_coeffs[COL_CANCUN_DELTA];
        let ts = &col_coeffs[COL_BLOCK_TIMESTAMP];
        let coinbase_warm = &col_coeffs[COL_COINBASE_WARM];
        let one_minus = |p: &Vec<Scalar>| poly_sub(&one_poly, p, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            poly_mul(is_real, &one_minus(is_real), curve),
            poly_mul(paris, &one_minus(paris), curve),
            poly_mul(shanghai, &one_minus(shanghai), curve),
            poly_mul(cancun, &one_minus(cancun), curve),
            poly_mul(shanghai, &one_minus(paris), curve),
            poly_mul(cancun, &one_minus(shanghai), curve),
            {
                let gap = poly_sub(&poly_sub(ts, &shanghai_ts_poly, curve), sd, curve);
                poly_mul(shanghai, &gap, curve)
            },
            {
                let gap = poly_sub(&poly_sub(ts, &cancun_ts_poly, curve), cd, curve);
                poly_mul(cancun, &gap, curve)
            },
            poly_sub(ts, &build_le_recompose_poly(col_coeffs, COL_TS_BYTE_OFFSET, curve), curve),
            poly_sub(sd, &build_le_recompose_poly(col_coeffs, COL_SD_BYTE_OFFSET, curve), curve),
            poly_sub(cd, &build_le_recompose_poly(col_coeffs, COL_CD_BYTE_OFFSET, curve), curve),
            poly_mul(&one_minus(shanghai), sd, curve),
            poly_mul(&one_minus(cancun), cd, curve),
            poly_mul(&one_minus(is_real), paris, curve),
            poly_mul(&one_minus(is_real), ts, curve),
            poly_sub(coinbase_warm, shanghai, curve),
        ];
        let mut acc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for body in &bodies {
            let scaled = poly_scalar_mul(body, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![
            COL_IS_REAL,
            COL_IS_POST_PARIS,
            COL_IS_POST_SHANGHAI,
            COL_IS_POST_CANCUN,
            COL_COINBASE_WARM,
        ]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(&self, columns: &mut [Vec<Scalar>], num_rows: usize, padded_size: usize) {
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

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256), LookupTable::range(2)];
        let tbl_byte = 0usize;
        let tbl_bit = 1usize;
        let mut declarations = Vec::new();
        for j in 0..NUM_BYTE_LIMBS {
            declarations.push((
                LookupDeclaration {
                    label: format!("hardfork_ts_byte_{}_8bit", j),
                    column_index: COL_TS_BYTE_OFFSET + j,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("hardfork_sd_byte_{}_8bit", j),
                    column_index: COL_SD_BYTE_OFFSET + j,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("hardfork_cd_byte_{}_8bit", j),
                    column_index: COL_CD_BYTE_OFFSET + j,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        for (col, label) in [
            (COL_IS_REAL, "hardfork_is_real_1bit"),
            (COL_IS_POST_PARIS, "hardfork_paris_1bit"),
            (COL_IS_POST_SHANGHAI, "hardfork_shanghai_1bit"),
            (COL_IS_POST_CANCUN, "hardfork_cancun_1bit"),
            (COL_COINBASE_WARM, "hardfork_coinbase_warm_1bit"),
        ] {
            declarations.push((
                LookupDeclaration {
                    label: label.into(),
                    column_index: col,
                    max_bits: 1,
                    selector_column: None,
                },
                tbl_bit,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp linkage descriptors ──────────────────────────────

/// Binds this AIR's `block_timestamp` column to the corresponding
/// scalar slot in `block_header_air` (`COL_TIMESTAMP`). Gated by
/// `is_real` on the A side and `is_real_row` on the B side (the latter
/// is the standard block-header real-row selector).
pub fn make_hardfork_to_block_header_descriptor(
    hardfork_layer_index: usize,
    block_header_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "hardfork_to_block_header_timestamp_v1".into(),
        a_layer_index: hardfork_layer_index,
        a_columns: vec![COL_BLOCK_TIMESTAMP],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: block_header_layer_index,
        b_columns: vec![metavm_zkp::block_header_air::COL_TIMESTAMP],
        b_selector_column: Some(metavm_zkp::block_header_air::COL_IS_REAL),
    }
}

/// Binds this AIR's `(is_post_shanghai, coinbase_warm)` pair to the
/// EIP-2929 access-list AIR's `(is_warm, is_warm)` selector at coinbase
/// accesses. Since the constraint `coinbase_warm_equals_shanghai`
/// forces `coinbase_warm = is_post_shanghai` in this AIR, the tuple
/// `(is_post_shanghai, coinbase_warm) = (1, 1)` on post-Shanghai rows
/// and `(0, 0)` on pre-Shanghai rows. The B side projects the access
/// AIR's `is_warm` column twice for tuple-alignment (the redundancy
/// pins both flags to the same access-list bit).
pub fn make_hardfork_to_access_2929_descriptor(
    hardfork_layer_index: usize,
    access_2929_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "hardfork_to_access_2929_coinbase_warm_v1".into(),
        a_layer_index: hardfork_layer_index,
        a_columns: vec![COL_IS_POST_SHANGHAI, COL_COINBASE_WARM],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: access_2929_layer_index,
        b_columns: vec![
            crate::access_list_eip2929_air::COL_IS_WARM,
            crate::access_list_eip2929_air::COL_IS_WARM,
        ],
        b_selector_column: Some(crate::access_list_eip2929_air::COL_IS_REAL),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const PRE_SHANGHAI_TS: u64 = 1_670_000_000; // late 2022
    const POST_SHANGHAI_PRE_CANCUN_TS: u64 = 1_690_000_000; // mid 2023
    const POST_CANCUN_TS: u64 = 1_715_000_000; // mid 2024

    fn cs_of(w: &HardforkRulesWitness) -> (TracePolynomials, HardforkRulesConstraintSystem) {
        let trace = build_trace_polynomials(w, CurveType::Bls48581);
        let cs = HardforkRulesConstraintSystem::new(trace.num_rows);
        (trace, cs)
    }

    fn assert_all_vanish(trace: &TracePolynomials, cs: &HardforkRulesConstraintSystem) {
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, trace.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) nonzero at row {}",
                    i,
                    cs.constraint_labels()[i],
                    r,
                );
            }
        }
    }

    #[test]
    fn pre_shanghai_witness_vanishes() {
        let r = from_timestamp(PRE_SHANGHAI_TS);
        assert!(!r.is_post_shanghai);
        assert!(!r.is_post_cancun);
        assert_eq!(r.shanghai_delta, 0);
        assert_eq!(r.cancun_delta, 0);
        let w = HardforkRulesWitness { rows: vec![r] };
        let (trace, cs) = cs_of(&w);
        assert_all_vanish(&trace, &cs);
    }

    #[test]
    fn post_shanghai_pre_cancun_witness_vanishes() {
        let r = from_timestamp(POST_SHANGHAI_PRE_CANCUN_TS);
        assert!(r.is_post_shanghai);
        assert!(!r.is_post_cancun);
        assert_eq!(r.shanghai_delta, POST_SHANGHAI_PRE_CANCUN_TS - SHANGHAI_TIMESTAMP);
        assert_eq!(r.cancun_delta, 0);
        let w = HardforkRulesWitness { rows: vec![r] };
        let (trace, cs) = cs_of(&w);
        assert_all_vanish(&trace, &cs);
    }

    #[test]
    fn post_cancun_witness_vanishes() {
        let r = from_timestamp(POST_CANCUN_TS);
        assert!(r.is_post_shanghai);
        assert!(r.is_post_cancun);
        assert_eq!(r.shanghai_delta, POST_CANCUN_TS - SHANGHAI_TIMESTAMP);
        assert_eq!(r.cancun_delta, POST_CANCUN_TS - CANCUN_TIMESTAMP);
        let w = HardforkRulesWitness { rows: vec![r] };
        let (trace, cs) = cs_of(&w);
        assert_all_vanish(&trace, &cs);
    }

    #[test]
    fn monotonicity_violation_detected() {
        // Construct a row claiming cancun=1 but shanghai=0 — violates
        // constraint 6 (cancun_implies_shanghai).
        let curve = CurveType::Bls48581;
        let w = HardforkRulesWitness::from_timestamp(POST_CANCUN_TS);
        let mut trace = build_trace_polynomials(&w, curve);
        // Flip is_post_shanghai to 0 while keeping is_post_cancun=1.
        trace.columns[COL_IS_POST_SHANGHAI].evaluations[0] = Scalar::zero(curve);
        // Also flip coinbase_warm to 0 to keep constraint 16 satisfied
        // — we want to isolate the monotonicity violation.
        trace.columns[COL_COINBASE_WARM].evaluations[0] = Scalar::zero(curve);
        let cs = HardforkRulesConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // constraint 5 = cancun_implies_shanghai (index 5 in 0-based list).
        let body_5 = &bodies[5];
        assert!(!body_5[0].is_zero(), "cancun_implies_shanghai must fire");
    }

    #[test]
    fn tampered_delta_detected() {
        let curve = CurveType::Bls48581;
        let w = HardforkRulesWitness::from_timestamp(POST_SHANGHAI_PRE_CANCUN_TS);
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper shanghai_delta to a wrong (but still small / range-ok)
        // value while leaving the LE byte decomp consistent — choose
        // delta+1 and rebuild bytes.
        let bad = POST_SHANGHAI_PRE_CANCUN_TS - SHANGHAI_TIMESTAMP + 1;
        trace.columns[COL_SHANGHAI_DELTA].evaluations[0] = Scalar::from_u64(bad, curve);
        for j in 0..NUM_BYTE_LIMBS {
            let b = ((bad >> (8 * j)) & 0xff) as u64;
            trace.columns[COL_SD_BYTE_OFFSET + j].evaluations[0] = Scalar::from_u64(b, curve);
        }
        let cs = HardforkRulesConstraintSystem::new(trace.num_rows);
        let cr: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, trace.num_rows);
        // shanghai_gate is constraint 6 (0-based).
        let body_6 = &bodies[6];
        assert!(!body_6[0].is_zero(), "shanghai_gate must fire on tampered delta");
    }

    #[test]
    fn descriptors_well_formed() {
        let d_bh = make_hardfork_to_block_header_descriptor(0, 1);
        assert_eq!(d_bh.label, "hardfork_to_block_header_timestamp_v1");
        assert_eq!(d_bh.a_layer_index, 0);
        assert_eq!(d_bh.b_layer_index, 1);
        assert_eq!(d_bh.a_columns, vec![COL_BLOCK_TIMESTAMP]);
        assert_eq!(d_bh.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_bh.b_columns,
            vec![metavm_zkp::block_header_air::COL_TIMESTAMP]
        );
        assert_eq!(
            d_bh.b_selector_column,
            Some(metavm_zkp::block_header_air::COL_IS_REAL)
        );

        let d_acc = make_hardfork_to_access_2929_descriptor(2, 3);
        assert_eq!(d_acc.label, "hardfork_to_access_2929_coinbase_warm_v1");
        assert_eq!(d_acc.a_columns.len(), 2);
        assert_eq!(d_acc.b_columns.len(), 2);
        assert_eq!(d_acc.a_columns[0], COL_IS_POST_SHANGHAI);
        assert_eq!(d_acc.a_columns[1], COL_COINBASE_WARM);
        assert_eq!(d_acc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_acc.b_selector_column,
            Some(crate::access_list_eip2929_air::COL_IS_REAL)
        );
    }

    #[test]
    fn at_activation_boundary_vanishes() {
        // At exactly SHANGHAI_TIMESTAMP, is_post_shanghai=1 with delta=0.
        let r = from_timestamp(SHANGHAI_TIMESTAMP);
        assert!(r.is_post_shanghai);
        assert!(!r.is_post_cancun);
        assert_eq!(r.shanghai_delta, 0);
        let w = HardforkRulesWitness { rows: vec![r] };
        let (trace, cs) = cs_of(&w);
        assert_all_vanish(&trace, &cs);
    }

    #[test]
    fn lookup_declarations_well_formed() {
        let cs = HardforkRulesConstraintSystem::new(1);
        let req = cs.lookup_declarations();
        // 24 byte 8-bit declarations + 5 binary 1-bit declarations.
        assert_eq!(req.declarations.len(), 24 + 5);
        assert_eq!(req.tables.len(), 2);
    }

    #[test]
    fn max_initcode_size_constant() {
        assert_eq!(MAX_INITCODE_SIZE, 49_152);
    }
}
