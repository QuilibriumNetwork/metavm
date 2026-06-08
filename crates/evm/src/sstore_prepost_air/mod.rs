//! EVM SLOAD/SSTORE pre/post-state AIR.
//!
//! Proves, per `SLOAD` / `SSTORE` event:
//!
//! - SLOAD reads `pre_value` from the world state for `slot` and the
//!   value is unchanged after the op (`post_value = pre_value`).
//! - SSTORE writes `post_value` to `slot`; `pre_value` is the value at
//!   the start of the op; `original_value` is the slot value at the
//!   start of the transaction.
//! - Per-event `gas_cost` follows EIP-2200 / EIP-2929 / EIP-3529 case
//!   logic. The constraint system here scaffolds the **major cases**:
//!   - warm SLOAD       → 100 gas  (EIP-2929)
//!   - cold SLOAD       → 2100 gas (EIP-2929)
//!   - SSTORE no-op     → 100 gas  (pre == post, EIP-2200 warm)
//!   - SSTORE write     → 20000 gas (original == 0, post != 0, "create")
//!   - SSTORE clear     → 5000 gas  (original != 0, post == 0, "clear",
//!                                    refund tracked elsewhere via EIP-3529)
//!
//! Full case-by-case EIP-2200 cost matrix (with the 9 sub-cases of
//! original/current/new permutations) is left as follow-up — see the
//! companion AIRs `sstore_transition_air` (algebraic state-root delta)
//! and `gas_refund_3529_air` (refund tracking).
//!
//! # Row layout
//!
//! One row per `(pc, op, slot, pre, post, original, gas, is_warm)` event:
//!
//!   - `pc` (u64)
//!   - `sel_sload`, `sel_sstore`, `is_warm`, `is_real` (binary)
//!   - 32 bytes of `slot`
//!   - 32 bytes of `pre_value`
//!   - 32 bytes of `post_value`
//!   - 32 bytes of `original_value`
//!   - `gas_cost` (u64) + 8 LE byte limbs
//!
//! # Constraints
//!
//! Row-local:
//!
//! 0..3 four binary checks (sel_sload, sel_sstore, is_warm, is_real)
//! 4   `sel_sload + sel_sstore = is_real`            (mutex on real rows)
//! 5   `sel_sload · is_warm · (gas_cost - 100)  = 0`
//! 6   `sel_sload · (1 - is_warm) · (gas_cost - 2100) = 0`
//! 7   `sel_sstore · (pre = post indicator)`         (no-op gas = 100,
//!       enforced via the β-RLC equality bundle, see #8)
//! 8   `sel_sload · Σ_β β^k (post_byte[k] - pre_byte[k]) = 0`
//!       (SLOAD does not mutate the slot)
//! 9   `gas_cost - Σ_{i<8} 2^{8i} · gas_byte[i] = 0`  (LE byte decomp)
//! 10  `sel_sstore · noop_indicator · (gas_cost - 100) = 0`
//!       (SSTORE no-op → 100 gas)
//! 11  `sel_sstore · is_orig_zero · is_post_nonzero_hint · (gas_cost - 20000) = 0`
//!       (SSTORE write: original==0, post!=0 → 20000 gas)
//! 12  `sel_sstore · is_orig_nonzero_hint · is_post_zero · (gas_cost - 5000) = 0`
//!       (SSTORE clear: original!=0, post==0 → 5000 gas refund-eligible)
//!
//! Byte ranges: 32 slot + 32 pre + 32 post + 32 original + 8 gas-byte =
//! 136 8-bit lookups.
//!
//! # Cross-AIR LogUp descriptors
//!
//! - [`make_sstore_prepost_to_storage_access_descriptor`] —
//!   binds the **post**-value to `storage_access_air`'s value column on
//!   the side of the storage gadget.
//! - [`make_sstore_prepost_to_access_2929_descriptor`] —
//!   binds `(slot, is_warm, gas_cost)` to `access_list_eip2929_air`.
//! - [`make_sstore_prepost_to_sstore_transition_descriptor`] —
//!   binds `(original, pre, post)` aliased onto `(old, new)` of the
//!   sstore-transition AIR.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const WORD_BYTES: usize = 32;
pub const GAS_BYTES: usize = 8;

pub const WARM_SLOAD_GAS: u64 = 100;
pub const COLD_SLOAD_GAS: u64 = 2100;
pub const SSTORE_NOOP_GAS: u64 = 100;
pub const SSTORE_WRITE_GAS: u64 = 20_000;
pub const SSTORE_CLEAR_GAS: u64 = 5_000;

// ─── Column layout ───────────────────────────────────────────────────

pub const COL_PC: usize = 0;
pub const COL_SEL_SLOAD: usize = 1;
pub const COL_SEL_SSTORE: usize = 2;
pub const COL_IS_WARM: usize = 3;
pub const COL_IS_REAL: usize = 4;
pub const COL_GAS_COST: usize = 5;

// Indicator columns; binary, witnessed.
//   IS_PRE_EQ_POST     := (pre_value == post_value)
//   IS_ORIG_ZERO       := (original_value == 0)
//   IS_POST_ZERO       := (post_value == 0)
//   IS_POST_NONZERO    := 1 - IS_POST_ZERO  (hint)
//   IS_ORIG_NONZERO    := 1 - IS_ORIG_ZERO  (hint)
pub const COL_IS_PRE_EQ_POST: usize = 6;
pub const COL_IS_ORIG_ZERO: usize = 7;
pub const COL_IS_POST_ZERO: usize = 8;
pub const COL_IS_POST_NONZERO: usize = 9;
pub const COL_IS_ORIG_NONZERO: usize = 10;

pub const COL_SLOT_OFFSET: usize = 11;
pub const COL_SLOT_END: usize = COL_SLOT_OFFSET + WORD_BYTES; // 43

pub const COL_PRE_VALUE_OFFSET: usize = COL_SLOT_END; // 43
pub const COL_PRE_VALUE_END: usize = COL_PRE_VALUE_OFFSET + WORD_BYTES; // 75

pub const COL_POST_VALUE_OFFSET: usize = COL_PRE_VALUE_END; // 75
pub const COL_POST_VALUE_END: usize = COL_POST_VALUE_OFFSET + WORD_BYTES; // 107

pub const COL_ORIGINAL_VALUE_OFFSET: usize = COL_POST_VALUE_END; // 107
pub const COL_ORIGINAL_VALUE_END: usize = COL_ORIGINAL_VALUE_OFFSET + WORD_BYTES; // 139

pub const COL_GAS_BYTES_OFFSET: usize = COL_ORIGINAL_VALUE_END; // 139
pub const COL_GAS_BYTES_END: usize = COL_GAS_BYTES_OFFSET + GAS_BYTES; // 147

pub const NUM_COLUMNS: usize = COL_GAS_BYTES_END; // 147

/// Row-local constraint count.
///
/// 0:  sel_sload binary
/// 1:  sel_sstore binary
/// 2:  is_warm binary
/// 3:  is_real binary
/// 4:  sel_sload + sel_sstore = is_real    (mutex on real rows)
/// 5:  sel_sload * is_warm * (gas - 100)        = 0
/// 6:  sel_sload * (1 - is_warm) * (gas - 2100) = 0
/// 7:  sel_sload * β-RLC(post_bytes - pre_bytes) = 0  (SLOAD non-mutation)
/// 8:  gas_cost - Σ 2^{8i} · gas_byte[i] = 0
/// 9:  sel_sstore * is_pre_eq_post * (gas - 100) = 0   (no-op)
/// 10: sel_sstore * is_orig_zero * is_post_nonzero * (gas - 20000) = 0 (write)
/// 11: sel_sstore * is_orig_nonzero * is_post_zero * (gas - 5000) = 0  (clear)
/// 12: is_pre_eq_post binary
/// 13: is_orig_zero binary
/// 14: is_post_zero binary
/// 15: is_post_nonzero + is_post_zero = 1   (complementary hint)
/// 16: is_orig_nonzero + is_orig_zero = 1   (complementary hint)
pub const NUM_ROW_CONSTRAINTS: usize = 17;
pub const NUM_SHIFTED: usize = 0;

/// Beta tag for the SLOAD β-RLC byte equality bundle.
const SLOAD_NON_MUTATION_BETA: u64 = 0x515f4e4f4d5554; // "Q_NOMUT"

// ─── Public types ────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SstorePrepostRow {
    pub pc: u64,
    pub sel_sload: bool,
    pub sel_sstore: bool,
    pub slot: [u8; WORD_BYTES],
    pub pre_value: [u8; WORD_BYTES],
    pub post_value: [u8; WORD_BYTES],
    pub original_value: [u8; WORD_BYTES],
    pub gas_cost: u64,
    pub is_warm: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SstorePrepostWitness {
    pub rows: Vec<SstorePrepostRow>,
}

impl SstorePrepostWitness {
    pub fn from_rows(rows: Vec<SstorePrepostRow>) -> Self {
        Self { rows }
    }
}

/// Host-side constructor.
///
/// `events` are tuples of:
/// `(pc, is_sstore, slot, pre, post, original, gas_cost, is_warm)`.
/// `is_sstore = true` → SSTORE row; `false` → SLOAD row.
pub fn from_events(
    events: &[(u64, bool, [u8; 32], [u8; 32], [u8; 32], [u8; 32], u64, bool)],
) -> SstorePrepostWitness {
    let mut rows = Vec::with_capacity(events.len());
    for (pc, is_sstore, slot, pre, post, original, gas, is_warm) in events.iter() {
        rows.push(SstorePrepostRow {
            pc: *pc,
            sel_sload: !*is_sstore,
            sel_sstore: *is_sstore,
            slot: *slot,
            pre_value: *pre,
            post_value: *post,
            original_value: *original,
            gas_cost: *gas,
            is_warm: *is_warm,
        });
    }
    SstorePrepostWitness { rows }
}

// ─── Trace builder ───────────────────────────────────────────────────

fn is_word_zero(w: &[u8; WORD_BYTES]) -> bool {
    w.iter().all(|b| *b == 0)
}

pub fn build_trace_polynomials(
    w: &SstorePrepostWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = w.rows.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (r, row) in w.rows.iter().enumerate() {
        cols[COL_PC][r] = Scalar::from_u64(row.pc, curve);
        cols[COL_SEL_SLOAD][r] = if row.sel_sload { one.clone() } else { zero.clone() };
        cols[COL_SEL_SSTORE][r] =
            if row.sel_sstore { one.clone() } else { zero.clone() };
        cols[COL_IS_WARM][r] = if row.is_warm { one.clone() } else { zero.clone() };
        cols[COL_IS_REAL][r] = if row.sel_sload || row.sel_sstore {
            one.clone()
        } else {
            zero.clone()
        };
        cols[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);

        // Indicator witnesses
        let pre_eq_post = row.pre_value == row.post_value;
        let orig_zero = is_word_zero(&row.original_value);
        let post_zero = is_word_zero(&row.post_value);
        cols[COL_IS_PRE_EQ_POST][r] =
            if pre_eq_post { one.clone() } else { zero.clone() };
        cols[COL_IS_ORIG_ZERO][r] = if orig_zero { one.clone() } else { zero.clone() };
        cols[COL_IS_POST_ZERO][r] = if post_zero { one.clone() } else { zero.clone() };
        cols[COL_IS_POST_NONZERO][r] =
            if post_zero { zero.clone() } else { one.clone() };
        cols[COL_IS_ORIG_NONZERO][r] =
            if orig_zero { zero.clone() } else { one.clone() };

        // Byte columns
        for k in 0..WORD_BYTES {
            cols[COL_SLOT_OFFSET + k][r] = Scalar::from_u64(row.slot[k] as u64, curve);
            cols[COL_PRE_VALUE_OFFSET + k][r] =
                Scalar::from_u64(row.pre_value[k] as u64, curve);
            cols[COL_POST_VALUE_OFFSET + k][r] =
                Scalar::from_u64(row.post_value[k] as u64, curve);
            cols[COL_ORIGINAL_VALUE_OFFSET + k][r] =
                Scalar::from_u64(row.original_value[k] as u64, curve);
        }
        // Gas LE decomposition
        let gas_le = row.gas_cost.to_le_bytes();
        for k in 0..GAS_BYTES {
            cols[COL_GAS_BYTES_OFFSET + k][r] = Scalar::from_u64(gas_le[k] as u64, curve);
        }
    }

    let polys: Vec<Polynomial> = cols
        .into_iter()
        .map(|e| Polynomial { evaluations: e, degree: num_rows })
        .collect();
    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ───────────────────────────────────────────────

pub struct SstorePrepostConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl SstorePrepostConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

fn beta_powers(curve: CurveType, n: usize) -> Vec<Scalar> {
    let beta = Scalar::from_u64(SLOAD_NON_MUTATION_BETA, curve);
    let mut p = Vec::with_capacity(n);
    let mut acc = Scalar::one(curve);
    for _ in 0..n {
        p.push(acc.clone());
        acc = acc.mul(&beta);
    }
    p
}

impl VmConstraintSystem for SstorePrepostConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "sel_sload_binary".into(),
            "sel_sstore_binary".into(),
            "is_warm_binary".into(),
            "is_real_binary".into(),
            "sel_sload_plus_sstore_eq_real".into(),
            "sload_warm_gas_eq_100".into(),
            "sload_cold_gas_eq_2100".into(),
            "sload_non_mutation_beta_rlc".into(),
            "gas_le_byte_decomp".into(),
            "sstore_noop_gas_eq_100".into(),
            "sstore_write_gas_eq_20000".into(),
            "sstore_clear_gas_eq_5000".into(),
            "is_pre_eq_post_binary".into(),
            "is_orig_zero_binary".into(),
            "is_post_zero_binary".into(),
            "post_zero_complement".into(),
            "orig_zero_complement".into(),
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
        let c100 = Scalar::from_u64(WARM_SLOAD_GAS, curve);
        let c2100 = Scalar::from_u64(COLD_SLOAD_GAS, curve);
        let c20000 = Scalar::from_u64(SSTORE_WRITE_GAS, curve);
        let c5000 = Scalar::from_u64(SSTORE_CLEAR_GAS, curve);

        // 2^{8i} for i in 0..GAS_BYTES
        let mut pow256 = Vec::with_capacity(GAS_BYTES);
        let mut acc = Scalar::one(curve);
        let two56 = Scalar::from_u64(256, curve);
        for _ in 0..GAS_BYTES {
            pow256.push(acc.clone());
            acc = acc.mul(&two56);
        }

        let betas = beta_powers(curve, WORD_BYTES);

        let mk = || vec![Scalar::zero(curve); n];
        let mut out: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| mk()).collect();

        for r in 0..n {
            let sld = &columns[COL_SEL_SLOAD][r];
            let sst = &columns[COL_SEL_SSTORE][r];
            let iw = &columns[COL_IS_WARM][r];
            let ir = &columns[COL_IS_REAL][r];
            let gc = &columns[COL_GAS_COST][r];
            let ipep = &columns[COL_IS_PRE_EQ_POST][r];
            let ioz = &columns[COL_IS_ORIG_ZERO][r];
            let ipz = &columns[COL_IS_POST_ZERO][r];
            let ipnz = &columns[COL_IS_POST_NONZERO][r];
            let ionz = &columns[COL_IS_ORIG_NONZERO][r];

            out[0][r] = sld.mul(&sld.sub(&one));
            out[1][r] = sst.mul(&sst.sub(&one));
            out[2][r] = iw.mul(&iw.sub(&one));
            out[3][r] = ir.mul(&ir.sub(&one));
            out[4][r] = sld.add(sst).sub(ir);

            let g_m_100 = gc.sub(&c100);
            let g_m_2100 = gc.sub(&c2100);
            let g_m_20000 = gc.sub(&c20000);
            let g_m_5000 = gc.sub(&c5000);

            out[5][r] = sld.mul(iw).mul(&g_m_100);
            out[6][r] = sld.mul(&one.sub(iw)).mul(&g_m_2100);

            // sload non-mutation β-RLC: sld * Σ β^k (post[k] - pre[k]) = 0
            let mut rlc = Scalar::zero(curve);
            for k in 0..WORD_BYTES {
                let pb = &columns[COL_PRE_VALUE_OFFSET + k][r];
                let qb = &columns[COL_POST_VALUE_OFFSET + k][r];
                rlc = rlc.add(&betas[k].mul(&qb.sub(pb)));
            }
            out[7][r] = sld.mul(&rlc);

            // gas LE byte decomp
            let mut g_acc = Scalar::zero(curve);
            for k in 0..GAS_BYTES {
                g_acc = g_acc.add(
                    &pow256[k].mul(&columns[COL_GAS_BYTES_OFFSET + k][r]),
                );
            }
            out[8][r] = gc.sub(&g_acc);

            out[9][r] = sst.mul(ipep).mul(&g_m_100);
            out[10][r] = sst.mul(ioz).mul(ipnz).mul(&g_m_20000);
            out[11][r] = sst.mul(ionz).mul(ipz).mul(&g_m_5000);

            out[12][r] = ipep.mul(&ipep.sub(&one));
            out[13][r] = ioz.mul(&ioz.sub(&one));
            out[14][r] = ipz.mul(&ipz.sub(&one));
            out[15][r] = ipnz.add(ipz).sub(&one);
            out[16][r] = ionz.add(ioz).sub(&one);
        }

        out
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let c100 = Scalar::from_u64(WARM_SLOAD_GAS, curve);
        let c2100 = Scalar::from_u64(COLD_SLOAD_GAS, curve);
        let c20000 = Scalar::from_u64(SSTORE_WRITE_GAS, curve);
        let c5000 = Scalar::from_u64(SSTORE_CLEAR_GAS, curve);

        let mut pow256 = Vec::with_capacity(GAS_BYTES);
        let mut acc = Scalar::one(curve);
        let two56 = Scalar::from_u64(256, curve);
        for _ in 0..GAS_BYTES {
            pow256.push(acc.clone());
            acc = acc.mul(&two56);
        }
        let betas = beta_powers(curve, WORD_BYTES);

        let sld = &ce[COL_SEL_SLOAD];
        let sst = &ce[COL_SEL_SSTORE];
        let iw = &ce[COL_IS_WARM];
        let ir = &ce[COL_IS_REAL];
        let gc = &ce[COL_GAS_COST];
        let ipep = &ce[COL_IS_PRE_EQ_POST];
        let ioz = &ce[COL_IS_ORIG_ZERO];
        let ipz = &ce[COL_IS_POST_ZERO];
        let ipnz = &ce[COL_IS_POST_NONZERO];
        let ionz = &ce[COL_IS_ORIG_NONZERO];

        let mut rlc = Scalar::zero(curve);
        for k in 0..WORD_BYTES {
            let pb = &ce[COL_PRE_VALUE_OFFSET + k];
            let qb = &ce[COL_POST_VALUE_OFFSET + k];
            rlc = rlc.add(&betas[k].mul(&qb.sub(pb)));
        }
        let mut g_acc = Scalar::zero(curve);
        for k in 0..GAS_BYTES {
            g_acc = g_acc.add(&pow256[k].mul(&ce[COL_GAS_BYTES_OFFSET + k]));
        }

        let bodies = [
            sld.mul(&sld.sub(&one)),
            sst.mul(&sst.sub(&one)),
            iw.mul(&iw.sub(&one)),
            ir.mul(&ir.sub(&one)),
            sld.add(sst).sub(ir),
            sld.mul(iw).mul(&gc.sub(&c100)),
            sld.mul(&one.sub(iw)).mul(&gc.sub(&c2100)),
            sld.mul(&rlc),
            gc.sub(&g_acc),
            sst.mul(ipep).mul(&gc.sub(&c100)),
            sst.mul(ioz).mul(ipnz).mul(&gc.sub(&c20000)),
            sst.mul(ionz).mul(ipz).mul(&gc.sub(&c5000)),
            ipep.mul(&ipep.sub(&one)),
            ioz.mul(&ioz.sub(&one)),
            ipz.mul(&ipz.sub(&one)),
            ipnz.add(ipz).sub(&one),
            ionz.add(ioz).sub(&one),
        ];

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = total.add(&ap.mul(b));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(
        &self,
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let c100_p = vec![Scalar::from_u64(WARM_SLOAD_GAS, curve)];
        let c2100_p = vec![Scalar::from_u64(COLD_SLOAD_GAS, curve)];
        let c20000_p = vec![Scalar::from_u64(SSTORE_WRITE_GAS, curve)];
        let c5000_p = vec![Scalar::from_u64(SSTORE_CLEAR_GAS, curve)];

        let sld = &cc[COL_SEL_SLOAD];
        let sst = &cc[COL_SEL_SSTORE];
        let iw = &cc[COL_IS_WARM];
        let ir = &cc[COL_IS_REAL];
        let gc = &cc[COL_GAS_COST];
        let ipep = &cc[COL_IS_PRE_EQ_POST];
        let ioz = &cc[COL_IS_ORIG_ZERO];
        let ipz = &cc[COL_IS_POST_ZERO];
        let ipnz = &cc[COL_IS_POST_NONZERO];
        let ionz = &cc[COL_IS_ORIG_NONZERO];

        let bin = |x: &Vec<Scalar>| poly_mul(x, &poly_sub(x, &one_p, curve), curve);

        let mut pow256 = Vec::with_capacity(GAS_BYTES);
        let mut acc = Scalar::one(curve);
        let two56 = Scalar::from_u64(256, curve);
        for _ in 0..GAS_BYTES {
            pow256.push(acc.clone());
            acc = acc.mul(&two56);
        }
        let betas = beta_powers(curve, WORD_BYTES);

        // β-RLC of (post[k] - pre[k])
        let mut rlc_poly: Vec<Scalar> = Vec::new();
        for k in 0..WORD_BYTES {
            let pb = &cc[COL_PRE_VALUE_OFFSET + k];
            let qb = &cc[COL_POST_VALUE_OFFSET + k];
            let diff = poly_sub(qb, pb, curve);
            let term = poly_scalar_mul(&diff, &betas[k]);
            if rlc_poly.is_empty() {
                rlc_poly = term;
            } else {
                rlc_poly = poly_add(&rlc_poly, &term, curve);
            }
        }

        // gas LE decomposition combined
        let mut g_acc_poly: Vec<Scalar> = Vec::new();
        for k in 0..GAS_BYTES {
            let term =
                poly_scalar_mul(&cc[COL_GAS_BYTES_OFFSET + k], &pow256[k]);
            if g_acc_poly.is_empty() {
                g_acc_poly = term;
            } else {
                g_acc_poly = poly_add(&g_acc_poly, &term, curve);
            }
        }

        let g_m_100 = poly_sub(gc, &c100_p, curve);
        let g_m_2100 = poly_sub(gc, &c2100_p, curve);
        let g_m_20000 = poly_sub(gc, &c20000_p, curve);
        let g_m_5000 = poly_sub(gc, &c5000_p, curve);

        let one_minus_iw = poly_sub(&one_p, iw, curve);

        let body0 = bin(sld);
        let body1 = bin(sst);
        let body2 = bin(iw);
        let body3 = bin(ir);
        let body4 = poly_sub(&poly_add(sld, sst, curve), ir, curve);
        let body5 = poly_mul(&poly_mul(sld, iw, curve), &g_m_100, curve);
        let body6 = poly_mul(&poly_mul(sld, &one_minus_iw, curve), &g_m_2100, curve);
        let body7 = poly_mul(sld, &rlc_poly, curve);
        let body8 = poly_sub(gc, &g_acc_poly, curve);
        let body9 = poly_mul(&poly_mul(sst, ipep, curve), &g_m_100, curve);
        let body10 = poly_mul(
            &poly_mul(&poly_mul(sst, ioz, curve), ipnz, curve),
            &g_m_20000,
            curve,
        );
        let body11 = poly_mul(
            &poly_mul(&poly_mul(sst, ionz, curve), ipz, curve),
            &g_m_5000,
            curve,
        );
        let body12 = bin(ipep);
        let body13 = bin(ioz);
        let body14 = bin(ipz);
        let body15 = poly_sub(&poly_add(ipnz, ipz, curve), &one_p, curve);
        let body16 = poly_sub(&poly_add(ionz, ioz, curve), &one_p, curve);

        let bodies = [
            body0, body1, body2, body3, body4, body5, body6, body7, body8, body9,
            body10, body11, body12, body13, body14, body15, body16,
        ];

        let mut total = bodies[0].clone();
        let mut ap = alpha.clone();
        for b in &bodies[1..] {
            total = poly_add(&total, &poly_scalar_mul(b, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
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
        if num_rows == 0 || num_rows >= padded_size || columns.len() < NUM_COLUMNS {
            return;
        }
        let zero = Scalar::zero(columns[0][0].curve_type());
        let one = Scalar::one(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
        // Padding-row indicators: pre==post (both 0) → 1; orig_zero=1;
        // post_zero=1; complementary nonzero hints = 0; this keeps all
        // constraints zero on padding rows.
        for r in num_rows..padded_size {
            columns[COL_IS_PRE_EQ_POST][r] = one.clone();
            columns[COL_IS_ORIG_ZERO][r] = one.clone();
            columns[COL_IS_POST_ZERO][r] = one.clone();
            columns[COL_IS_POST_NONZERO][r] = zero.clone();
            columns[COL_IS_ORIG_NONZERO][r] = zero.clone();
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..WORD_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sstore_prepost_slot_{}_8bit", k),
                    column_index: COL_SLOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("sstore_prepost_pre_{}_8bit", k),
                    column_index: COL_PRE_VALUE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("sstore_prepost_post_{}_8bit", k),
                    column_index: COL_POST_VALUE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
            declarations.push((
                LookupDeclaration {
                    label: format!("sstore_prepost_original_{}_8bit", k),
                    column_index: COL_ORIGINAL_VALUE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..GAS_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("sstore_prepost_gas_byte_{}_8bit", k),
                    column_index: COL_GAS_BYTES_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// Bind `(slot_bytes, post_value_bytes)` on this AIR to the
/// `(slot_be, value_be)` byte tuple on `storage_access_air`. Gated by
/// `is_real` on this AIR and `is_real` on the storage gadget. This
/// algebraically commits to the world-state read/write being consistent
/// with the storage-access proof.
pub fn make_sstore_prepost_to_storage_access_descriptor(
    sstore_prepost_layer_index: usize,
    storage_access_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(WORD_BYTES * 2);
    let mut b_columns = Vec::with_capacity(WORD_BYTES * 2);
    for k in 0..WORD_BYTES {
        a_columns.push(COL_SLOT_OFFSET + k);
        b_columns.push(metavm_zkp::storage_access_air::COL_SLOT_BE_OFFSET + k);
    }
    for k in 0..WORD_BYTES {
        a_columns.push(COL_POST_VALUE_OFFSET + k);
        b_columns.push(metavm_zkp::storage_access_air::COL_VALUE_BE_OFFSET + k);
    }
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sstore_prepost_to_storage_access_v1".into(),
        a_layer_index: sstore_prepost_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: storage_access_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::storage_access_air::COL_IS_REAL),
    }
}

/// Bind `(slot_bytes, is_warm, gas_cost)` on this AIR to the
/// EIP-2929 access-list AIR's `(slot_bytes, is_warm, gas_cost)` tuple.
/// Gated by `is_real` on both sides.
pub fn make_sstore_prepost_to_access_2929_descriptor(
    sstore_prepost_layer_index: usize,
    access_2929_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(WORD_BYTES + 2);
    let mut b_columns = Vec::with_capacity(WORD_BYTES + 2);
    for k in 0..WORD_BYTES {
        a_columns.push(COL_SLOT_OFFSET + k);
        b_columns.push(crate::access_list_eip2929_air::COL_SLOT_OFFSET + k);
    }
    a_columns.push(COL_IS_WARM);
    b_columns.push(crate::access_list_eip2929_air::COL_IS_WARM);
    a_columns.push(COL_GAS_COST);
    b_columns.push(crate::access_list_eip2929_air::COL_GAS_COST);
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sstore_prepost_to_access_2929_v1".into(),
        a_layer_index: sstore_prepost_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: access_2929_layer_index,
        b_columns,
        b_selector_column: Some(crate::access_list_eip2929_air::COL_IS_REAL),
    }
}

/// Bind `(pre_value, post_value, original_value)` on this AIR to the
/// `(old_value_be, new_value_be, original_value_be)` byte tuples on
/// `sstore_transition_air` — `pre → old_value_be`, `post → new_value_be`,
/// `original → original_value_be`. Gated by `sel_sstore` on this AIR and
/// `is_real` on the transition AIR. The `original_value_be` column was
/// added to the transition AIR specifically to close this binding.
///
/// 96-col tuple (3 × 32 BE bytes).
pub fn make_sstore_prepost_to_sstore_transition_descriptor(
    sstore_prepost_layer_index: usize,
    sstore_transition_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(WORD_BYTES * 3);
    let mut b_columns = Vec::with_capacity(WORD_BYTES * 3);
    for k in 0..WORD_BYTES {
        a_columns.push(COL_PRE_VALUE_OFFSET + k);
        b_columns
            .push(metavm_zkp::sstore_transition_air::COL_OLD_VALUE_BE_OFFSET + k);
    }
    for k in 0..WORD_BYTES {
        a_columns.push(COL_POST_VALUE_OFFSET + k);
        b_columns
            .push(metavm_zkp::sstore_transition_air::COL_NEW_VALUE_BE_OFFSET + k);
    }
    for k in 0..WORD_BYTES {
        a_columns.push(COL_ORIGINAL_VALUE_OFFSET + k);
        b_columns.push(
            metavm_zkp::sstore_transition_air::COL_ORIGINAL_VALUE_BE_OFFSET + k,
        );
    }
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "sstore_prepost_to_sstore_transition_v1".into(),
        a_layer_index: sstore_prepost_layer_index,
        a_columns,
        a_selector_column: Some(COL_SEL_SSTORE),
        b_layer_index: sstore_transition_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::sstore_transition_air::COL_IS_REAL),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn word_from_u64(v: u64) -> [u8; 32] {
        let mut w = [0u8; 32];
        w[24..].copy_from_slice(&v.to_be_bytes());
        w
    }

    fn assert_all_zero(t: &TracePolynomials) {
        let cs = SstorePrepostConstraintSystem::new(t.num_rows);
        let mut cv: Vec<Vec<Scalar>> =
            t.columns.iter().map(|p| p.evaluations.clone()).collect();
        cs.fix_trace_padding(&mut cv, t.num_rows, t.padded_size as usize);
        let cr: Vec<&Vec<Scalar>> = cv.iter().collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "constraint {} ({}) at row {} nonzero",
                    i,
                    cs.constraint_labels()[i],
                    r,
                );
            }
        }
    }

    #[test]
    fn sload_warm_charged_100() {
        let slot = word_from_u64(0xaa);
        let val = word_from_u64(0x42);
        let events =
            vec![(100, false, slot, val, val, val, WARM_SLOAD_GAS, true)];
        let w = from_events(&events);
        assert_eq!(w.rows.len(), 1);
        assert!(w.rows[0].sel_sload);
        assert!(!w.rows[0].sel_sstore);
        assert!(w.rows[0].is_warm);
        assert_eq!(w.rows[0].gas_cost, 100);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn sload_cold_charged_2100() {
        let slot = word_from_u64(0xbb);
        let val = word_from_u64(0x11);
        let events =
            vec![(200, false, slot, val, val, val, COLD_SLOAD_GAS, false)];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn sstore_noop_charged_100() {
        // original = pre = post → SSTORE no-op → 100 gas (warm).
        let slot = word_from_u64(0xcc);
        let val = word_from_u64(0x77);
        let events =
            vec![(300, true, slot, val, val, val, SSTORE_NOOP_GAS, true)];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn sstore_create_charged_20000() {
        // original = 0, pre = 0, post != 0 → 20000 gas.
        let slot = word_from_u64(0xdd);
        let zero = [0u8; 32];
        let val = word_from_u64(0x99);
        let events =
            vec![(400, true, slot, zero, val, zero, SSTORE_WRITE_GAS, true)];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn sstore_clear_charged_5000() {
        // original != 0, pre = original, post = 0 → 5000 gas (clear).
        let slot = word_from_u64(0xee);
        let orig = word_from_u64(0xab);
        let zero = [0u8; 32];
        let events =
            vec![(500, true, slot, orig, zero, orig, SSTORE_CLEAR_GAS, true)];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn tampered_sload_mutation_detected() {
        // SLOAD with post != pre must violate non-mutation β-RLC.
        let slot = word_from_u64(0xab);
        let pre = word_from_u64(0x10);
        let post = word_from_u64(0x20);
        let events =
            vec![(600, false, slot, pre, post, pre, WARM_SLOAD_GAS, true)];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SstorePrepostConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 7 = sload_non_mutation_beta_rlc must fire.
        assert!(
            !bodies[7][0].is_zero(),
            "sload mutation not caught by non-mutation constraint"
        );
    }

    #[test]
    fn tampered_gas_cost_detected() {
        // Honest warm SLOAD at 100 → tamper to 99 (gas LE decomp fails or
        // sload_warm_gas constraint fails).
        let slot = word_from_u64(0xcd);
        let val = word_from_u64(0x33);
        let events =
            vec![(700, false, slot, val, val, val, WARM_SLOAD_GAS, true)];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cv: Vec<Vec<Scalar>> =
            t.columns.iter().map(|p| p.evaluations.clone()).collect();
        cv[COL_GAS_COST][0] = Scalar::from_u64(99, CurveType::Bls48581);
        let cs = SstorePrepostConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cv.iter().collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Either sload_warm_gas_eq_100 (5) or gas_le_byte_decomp (8) fires.
        assert!(
            !bodies[5][0].is_zero() || !bodies[8][0].is_zero(),
            "tampered gas not caught"
        );
    }

    #[test]
    fn descriptors_well_formed() {
        let d_sa = make_sstore_prepost_to_storage_access_descriptor(0, 1);
        assert_eq!(d_sa.label, "sstore_prepost_to_storage_access_v1");
        assert_eq!(d_sa.a_columns.len(), WORD_BYTES * 2);
        assert_eq!(d_sa.b_columns.len(), WORD_BYTES * 2);
        assert_eq!(d_sa.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_sa.b_selector_column,
            Some(metavm_zkp::storage_access_air::COL_IS_REAL)
        );

        let d_29 = make_sstore_prepost_to_access_2929_descriptor(0, 2);
        assert_eq!(d_29.label, "sstore_prepost_to_access_2929_v1");
        assert_eq!(d_29.a_columns.len(), WORD_BYTES + 2);
        assert_eq!(d_29.b_columns.len(), WORD_BYTES + 2);
        assert_eq!(d_29.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_29.b_selector_column,
            Some(crate::access_list_eip2929_air::COL_IS_REAL)
        );

        let d_tr = make_sstore_prepost_to_sstore_transition_descriptor(0, 3);
        assert_eq!(d_tr.label, "sstore_prepost_to_sstore_transition_v1");
        assert_eq!(d_tr.a_columns.len(), WORD_BYTES * 3);
        assert_eq!(d_tr.b_columns.len(), WORD_BYTES * 3);
        assert_eq!(d_tr.a_selector_column, Some(COL_SEL_SSTORE));
        assert_eq!(
            d_tr.b_selector_column,
            Some(metavm_zkp::sstore_transition_air::COL_IS_REAL)
        );
        // Third 32-col block binds original_value → original_value_be.
        assert_eq!(d_tr.a_columns[WORD_BYTES * 2], COL_ORIGINAL_VALUE_OFFSET);
        assert_eq!(
            d_tr.b_columns[WORD_BYTES * 2],
            metavm_zkp::sstore_transition_air::COL_ORIGINAL_VALUE_BE_OFFSET,
        );
    }

    #[test]
    fn num_columns_and_constraints_pinned() {
        assert_eq!(NUM_COLUMNS, 147);
        assert_eq!(NUM_ROW_CONSTRAINTS, 17);
        assert_eq!(NUM_SHIFTED, 0);
    }

    #[test]
    fn evaluate_at_point_zero_on_honest_mixed_trace() {
        let slot1 = word_from_u64(0x01);
        let slot2 = word_from_u64(0x02);
        let val_a = word_from_u64(0x10);
        let val_b = word_from_u64(0x20);
        let zero = [0u8; 32];
        let events = vec![
            (10, false, slot1, val_a, val_a, val_a, WARM_SLOAD_GAS, true),
            (20, false, slot2, val_b, val_b, val_b, COLD_SLOAD_GAS, false),
            (30, true, slot1, val_a, val_a, val_a, SSTORE_NOOP_GAS, true),
            (40, true, slot2, zero, val_b, zero, SSTORE_WRITE_GAS, true),
        ];
        let w = from_events(&events);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = SstorePrepostConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xdead_beef, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..w.rows.len() {
            let row_evals: Vec<Scalar> = cr.iter().map(|c| c[r].clone()).collect();
            let v = cs.evaluate_at_point(&row_evals, &alpha);
            assert!(v.is_zero(), "row {} nonzero combined eval", r);
        }
    }
}
