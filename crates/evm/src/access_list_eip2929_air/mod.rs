//! EIP-2929 warm/cold access list AIR.
//!
//! Proves the per-access gas semantics for EIP-2929 storage slot and
//! address accesses, with optional EIP-2930 access-list pre-warming.
//!
//! # Per-access gas semantics
//!
//! - Cold storage slot read (SLOAD on a slot not yet touched in tx) = 2100 gas.
//! - Warm storage slot read (SLOAD on a slot already touched) = 100 gas.
//! - Cold address access (extcodesize/balance/call/...) = 2600 gas.
//! - Warm address access = 100 gas.
//!
//! An EIP-2930 access list pre-warms `(address, slot)` pairs *before*
//! the transaction starts executing. Such pre-warmed accesses must
//! therefore be billed as warm even on first touch.
//!
//! # Row layout
//!
//! One row per `(tx_index, address, slot, access_type)` access. The AIR
//! commits per-row:
//!
//!   - `access_type` ∈ {0 = storage slot, 1 = address}
//!   - 20 bytes of `address`
//!   - 32 bytes of `slot` (zero for `access_type = 1`)
//!   - `tx_index` (u64)
//!   - `is_cold`, `is_warm`, `prewarm_from_eip2930` (binary)
//!   - `gas_cost` (u64)
//!   - `is_real` (binary)
//!
//! # Row-local constraints
//!
//! 1.  `access_type` binary.
//! 2.  `is_cold` binary.
//! 3.  `is_warm` binary.
//! 4.  `prewarm_from_eip2930` binary.
//! 5.  `is_real` binary.
//! 6.  `is_cold + is_warm = is_real`  (exclusive on real rows; zero on padding).
//! 7.  `prewarm_from_eip2930 · (1 - is_warm) = 0`  (pre-warmed ⇒ warm).
//! 8.  `is_real · (1 - access_type) · is_cold · (gas_cost - 2100) = 0`
//!     (storage cold → 2100 gas).
//! 9.  `is_real · (1 - access_type) · is_warm · (gas_cost - 100) = 0`
//!     (storage warm → 100 gas).
//! 10. `is_real · access_type · is_cold · (gas_cost - 2600) = 0`
//!     (address cold → 2600 gas).
//! 11. `is_real · access_type · is_warm · (gas_cost - 100) = 0`
//!     (address warm → 100 gas).
//!
//! All 20 address bytes and 32 slot bytes get 8-bit range checks.
//!
//! # Shifted constraint — multi-access pre-warm chain
//!
//! Two row-local witness columns drive the cross-row chain:
//!
//!   - `is_chain` (binary): set by the witness builder to `1` on row `i`
//!     when row `i+1` is a later access to the **same** `(access_type,
//!     address, slot)` entry within the same tx. Padding rows and the
//!     last row of any chain set this to `0`.
//!   - `is_cold_to_warm_promotion` (binary): set to `is_cold` on real
//!     rows; equals `1` exactly when this row's access is the cold first
//!     touch that promotes the entry to warm.
//!
//! Shifted constraint (row 12):
//!
//!     is_chain[i] · ( is_warm[i+1] − max(is_warm[i], is_cold_to_warm_promotion[i]) ) = 0
//!
//! where `max(a,b) = a + b − a·b` for binary `a, b`. Together with the
//! row-local invariant `is_cold + is_warm = is_real`, this means: on any
//! real row in a chain, after the access the entry is warm
//! (`max(is_warm[i], is_cold_to_warm_promotion[i]) = 1`), so the next
//! row's `is_warm[i+1]` is forced to `1` — exactly the EIP-2929
//! semantics of "first touch cold (2100 / 2600), every subsequent touch
//! of the same entry warm (100)".
//!
//! A complete soundness chain that *forces* `is_chain` to be honest via
//! a sorted-by-(access_type, addr, slot, tx_index) view + permutation
//! still relies on the cross-AIR LogUp scaffold below — see
//! `make_access_2929_to_access_list_descriptor`. The shifted constraint
//! here closes the cold-to-warm promotion **conditionally on** the
//! witness-supplied chain bit.
//!
//! # Cross-AIR LogUp descriptors
//!
//! - [`make_access_2929_to_access_list_descriptor`] — binds rows where
//!   `prewarm_from_eip2930 = 1` to the EIP-2930 access-list AIR's
//!   `(address, storage_key)` columns, proving the pre-warm flag is
//!   honest.
//! - [`make_access_2929_to_gas_tracking_descriptor`] — binds
//!   `(gas_cost)` on this AIR to the gas-tracking AIR's `static_cost`
//!   column, gated by `is_real`.

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const ADDRESS_BYTES: usize = 20;
pub const SLOT_BYTES: usize = 32;

pub const COLD_STORAGE_GAS: u64 = 2100;
pub const WARM_GAS: u64 = 100;
pub const COLD_ADDRESS_GAS: u64 = 2600;

// ─── Column layout ───────────────────────────────────────────────────

pub const COL_ACCESS_TYPE: usize = 0;
pub const COL_TX_INDEX: usize = 1;
pub const COL_GAS_COST: usize = 2;
pub const COL_IS_COLD: usize = 3;
pub const COL_IS_WARM: usize = 4;
pub const COL_PREWARM_FROM_EIP2930: usize = 5;
pub const COL_IS_REAL: usize = 6;
/// Witness-supplied bit: 1 iff row i+1 is a later access to the **same**
/// `(access_type, address, slot)` entry within the same tx. The shifted
/// constraint uses this to propagate the warm flag forward.
pub const COL_IS_CHAIN: usize = 7;
/// Witness-supplied bit: 1 iff this access promotes the entry from cold
/// to warm (i.e., the first touch of `(access_type, address, slot)` in
/// the tx). On real rows this equals `is_cold`.
pub const COL_IS_COLD_TO_WARM_PROMOTION: usize = 8;

pub const COL_ADDRESS_OFFSET: usize = 9;
pub const COL_ADDRESS_END: usize = COL_ADDRESS_OFFSET + ADDRESS_BYTES; // 29

pub const COL_SLOT_OFFSET: usize = COL_ADDRESS_END;                    // 29
pub const COL_SLOT_END: usize = COL_SLOT_OFFSET + SLOT_BYTES;          // 61

pub const NUM_COLUMNS: usize = COL_SLOT_END;                           // 61

/// Row-local constraint count.
///
/// 0:  access_type binary
/// 1:  is_cold binary
/// 2:  is_warm binary
/// 3:  prewarm_from_eip2930 binary
/// 4:  is_real binary
/// 5:  is_cold + is_warm = is_real
/// 6:  prewarm_from_eip2930 * (1 - is_warm) = 0
/// 7:  is_real * (1 - access_type) * is_cold * (gas_cost - 2100) = 0
/// 8:  is_real * (1 - access_type) * is_warm * (gas_cost - 100)  = 0
/// 9:  is_real *      access_type  * is_cold * (gas_cost - 2600) = 0
/// 10: is_real *      access_type  * is_warm * (gas_cost - 100)  = 0
/// 11: is_chain binary
/// 12: is_cold_to_warm_promotion binary
/// 13: is_real * (is_cold_to_warm_promotion - is_cold) = 0
///     (on real rows, the promotion bit equals `is_cold`)
/// 14: (1 - is_real) * is_chain = 0
///     (padding rows cannot start a chain)
pub const NUM_ROW_CONSTRAINTS: usize = 15;

/// Shifted constraint count.
///
/// 0:  is_chain[i] * (is_warm[i+1] − max(is_warm[i], is_cold_to_warm_promotion[i])) = 0
///     where `max(a,b) = a + b − a·b` for binary a,b.
pub const NUM_SHIFTED: usize = 1;

// ─── Public types ────────────────────────────────────────────────────

/// What kind of resource is being accessed under EIP-2929.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AccessKind {
    /// `(address, slot)` storage-slot access (SLOAD / SSTORE).
    StorageSlot,
    /// Address-only access (EXTCODESIZE / EXTCODEHASH / BALANCE / CALL family).
    Address,
}

impl AccessKind {
    pub fn as_u64(self) -> u64 {
        match self {
            AccessKind::StorageSlot => 0,
            AccessKind::Address => 1,
        }
    }
}

/// One row of the access-list-EIP-2929 witness.
#[derive(Clone, Debug)]
pub struct AccessListEip2929Row {
    pub access_type: AccessKind,
    pub address: [u8; ADDRESS_BYTES],
    pub slot: [u8; SLOT_BYTES],
    pub tx_index: u64,
    pub is_cold: bool,
    pub is_warm: bool,
    pub prewarm_from_eip2930: bool,
    pub gas_cost: u64,
    /// Cross-row chain bit. Set to `true` on row `i` when row `i+1` is a
    /// later access to the same `(access_type, address, slot)` entry in
    /// the same tx. Always `false` on the last row of a chain.
    pub is_chain: bool,
    /// Equals `is_cold` on real rows: the cold-to-warm promotion bit
    /// driving the shifted warm-propagation constraint.
    pub is_cold_to_warm_promotion: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AccessListEip2929Witness {
    pub rows: Vec<AccessListEip2929Row>,
}

impl AccessListEip2929Witness {
    pub fn from_rows(rows: Vec<AccessListEip2929Row>) -> Self {
        Self { rows }
    }
}

/// Build a witness from a sequence of accesses, threading per-tx warm
/// sets to assign `is_cold` / `is_warm` / `gas_cost`. Each entry:
/// `(access_type, address, slot, prewarm_from_eip2930)`.
///
/// Pre-warmed `(address, slot)` pairs are treated as already in the warm
/// set at tx start, so even the first touch within the tx is billed
/// warm.
///
/// All accesses must share the same `tx_index` semantics — this helper
/// takes a flat list and treats them as one tx (the host may compose
/// per-tx invocations explicitly to scope warm-sets per tx).
pub fn from_accesses(
    accesses: &[(AccessKind, [u8; ADDRESS_BYTES], [u8; SLOT_BYTES], bool)],
) -> AccessListEip2929Witness {
    from_accesses_with_tx_index(accesses, 0)
}

/// Same as [`from_accesses`] but tags every emitted row with `tx_index`
/// (the warm-set is still scoped to this invocation: callers building
/// multi-tx witnesses should concatenate per-tx invocations).
pub fn from_accesses_with_tx_index(
    accesses: &[(AccessKind, [u8; ADDRESS_BYTES], [u8; SLOT_BYTES], bool)],
    tx_index: u64,
) -> AccessListEip2929Witness {
    let mut warm_storage: std::collections::HashSet<(
        [u8; ADDRESS_BYTES],
        [u8; SLOT_BYTES],
    )> = std::collections::HashSet::new();
    let mut warm_address: std::collections::HashSet<[u8; ADDRESS_BYTES]> =
        std::collections::HashSet::new();

    // Pre-warm phase: walk once and pre-mark every prewarm entry.
    for (kind, addr, slot, prewarm) in accesses.iter() {
        if *prewarm {
            match kind {
                AccessKind::StorageSlot => {
                    warm_storage.insert((*addr, *slot));
                    warm_address.insert(*addr);
                }
                AccessKind::Address => {
                    warm_address.insert(*addr);
                }
            }
        }
    }

    let mut rows = Vec::with_capacity(accesses.len());
    for (kind, addr, slot, prewarm) in accesses.iter() {
        let is_warm = match kind {
            AccessKind::StorageSlot => warm_storage.contains(&(*addr, *slot)),
            AccessKind::Address => warm_address.contains(addr),
        };
        let is_cold = !is_warm;
        let gas_cost = match (kind, is_cold) {
            (AccessKind::StorageSlot, true) => COLD_STORAGE_GAS,
            (AccessKind::StorageSlot, false) => WARM_GAS,
            (AccessKind::Address, true) => COLD_ADDRESS_GAS,
            (AccessKind::Address, false) => WARM_GAS,
        };
        rows.push(AccessListEip2929Row {
            access_type: *kind,
            address: *addr,
            slot: *slot,
            tx_index,
            is_cold,
            is_warm,
            prewarm_from_eip2930: *prewarm,
            gas_cost,
            // Provisional values; the chain bit is back-patched after
            // the full row sequence is known.
            is_chain: false,
            // On real rows the promotion bit equals `is_cold`.
            is_cold_to_warm_promotion: is_cold,
        });
        // After this access, mark the touched resource as warm for the
        // next access in the same tx.
        match kind {
            AccessKind::StorageSlot => {
                warm_storage.insert((*addr, *slot));
                warm_address.insert(*addr);
            }
            AccessKind::Address => {
                warm_address.insert(*addr);
            }
        }
    }

    // Back-patch the chain bit: row i has `is_chain = true` iff some
    // later row j > i references the same `(access_type, address, slot)`
    // AND the *immediate* next row i+1 also references the same entry
    // (i.e., they are adjacent). The shifted constraint binds adjacent
    // rows, so the chain bit must mark adjacency, not arbitrary repeats.
    let n = rows.len();
    for i in 0..n.saturating_sub(1) {
        let same_entry = rows[i].access_type == rows[i + 1].access_type
            && rows[i].address == rows[i + 1].address
            && rows[i].slot == rows[i + 1].slot
            && rows[i].tx_index == rows[i + 1].tx_index;
        rows[i].is_chain = same_entry;
    }

    AccessListEip2929Witness { rows }
}

// ─── Trace builder ───────────────────────────────────────────────────

pub fn build_trace_polynomials(
    w: &AccessListEip2929Witness,
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
        cols[COL_ACCESS_TYPE][r] = Scalar::from_u64(row.access_type.as_u64(), curve);
        cols[COL_TX_INDEX][r] = Scalar::from_u64(row.tx_index, curve);
        cols[COL_GAS_COST][r] = Scalar::from_u64(row.gas_cost, curve);
        cols[COL_IS_COLD][r] = if row.is_cold { one.clone() } else { zero.clone() };
        cols[COL_IS_WARM][r] = if row.is_warm { one.clone() } else { zero.clone() };
        cols[COL_PREWARM_FROM_EIP2930][r] =
            if row.prewarm_from_eip2930 { one.clone() } else { zero.clone() };
        cols[COL_IS_REAL][r] = one.clone();
        cols[COL_IS_CHAIN][r] = if row.is_chain { one.clone() } else { zero.clone() };
        cols[COL_IS_COLD_TO_WARM_PROMOTION][r] =
            if row.is_cold_to_warm_promotion { one.clone() } else { zero.clone() };
        for k in 0..ADDRESS_BYTES {
            cols[COL_ADDRESS_OFFSET + k][r] =
                Scalar::from_u64(row.address[k] as u64, curve);
        }
        for k in 0..SLOT_BYTES {
            cols[COL_SLOT_OFFSET + k][r] = Scalar::from_u64(row.slot[k] as u64, curve);
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

pub struct AccessListEip2929ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AccessListEip2929ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for AccessListEip2929ConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "access_type_binary".into(),
            "is_cold_binary".into(),
            "is_warm_binary".into(),
            "prewarm_from_eip2930_binary".into(),
            "is_real_binary".into(),
            "cold_plus_warm_eq_real".into(),
            "prewarm_implies_warm".into(),
            "storage_cold_gas_eq_2100".into(),
            "storage_warm_gas_eq_100".into(),
            "address_cold_gas_eq_2600".into(),
            "address_warm_gas_eq_100".into(),
            "is_chain_binary".into(),
            "is_cold_to_warm_promotion_binary".into(),
            "promotion_eq_cold_on_real".into(),
            "padding_has_no_chain".into(),
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
        let c100 = Scalar::from_u64(WARM_GAS, curve);
        let c2100 = Scalar::from_u64(COLD_STORAGE_GAS, curve);
        let c2600 = Scalar::from_u64(COLD_ADDRESS_GAS, curve);

        let mk = || vec![Scalar::zero(curve); n];
        let mut out: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| mk()).collect();

        for r in 0..n {
            let at = &columns[COL_ACCESS_TYPE][r];
            let cold = &columns[COL_IS_COLD][r];
            let warm = &columns[COL_IS_WARM][r];
            let pw = &columns[COL_PREWARM_FROM_EIP2930][r];
            let ir = &columns[COL_IS_REAL][r];
            let gc = &columns[COL_GAS_COST][r];
            let ic = &columns[COL_IS_CHAIN][r];
            let pr = &columns[COL_IS_COLD_TO_WARM_PROMOTION][r];

            out[0][r] = at.mul(&at.sub(&one));
            out[1][r] = cold.mul(&cold.sub(&one));
            out[2][r] = warm.mul(&warm.sub(&one));
            out[3][r] = pw.mul(&pw.sub(&one));
            out[4][r] = ir.mul(&ir.sub(&one));

            // is_cold + is_warm = is_real
            out[5][r] = cold.add(warm).sub(ir);

            // prewarm * (1 - warm) = 0
            out[6][r] = pw.mul(&one.sub(warm));

            let not_at = one.sub(at);

            // is_real * (1 - access_type) * is_cold * (gas - 2100) = 0
            let g_m_2100 = gc.sub(&c2100);
            out[7][r] = ir.mul(&not_at).mul(cold).mul(&g_m_2100);

            // is_real * (1 - access_type) * is_warm * (gas - 100) = 0
            let g_m_100 = gc.sub(&c100);
            out[8][r] = ir.mul(&not_at).mul(warm).mul(&g_m_100);

            // is_real * access_type * is_cold * (gas - 2600) = 0
            let g_m_2600 = gc.sub(&c2600);
            out[9][r] = ir.mul(at).mul(cold).mul(&g_m_2600);

            // is_real * access_type * is_warm * (gas - 100) = 0
            out[10][r] = ir.mul(at).mul(warm).mul(&g_m_100);

            // 11: is_chain binary
            out[11][r] = ic.mul(&ic.sub(&one));
            // 12: is_cold_to_warm_promotion binary
            out[12][r] = pr.mul(&pr.sub(&one));
            // 13: is_real * (is_cold_to_warm_promotion - is_cold) = 0
            out[13][r] = ir.mul(&pr.sub(cold));
            // 14: (1 - is_real) * is_chain = 0  (no chains start on padding)
            out[14][r] = one.sub(ir).mul(ic);
        }

        out
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let c100 = Scalar::from_u64(WARM_GAS, curve);
        let c2100 = Scalar::from_u64(COLD_STORAGE_GAS, curve);
        let c2600 = Scalar::from_u64(COLD_ADDRESS_GAS, curve);

        let at = &ce[COL_ACCESS_TYPE];
        let cold = &ce[COL_IS_COLD];
        let warm = &ce[COL_IS_WARM];
        let pw = &ce[COL_PREWARM_FROM_EIP2930];
        let ir = &ce[COL_IS_REAL];
        let gc = &ce[COL_GAS_COST];
        let ic = &ce[COL_IS_CHAIN];
        let pr = &ce[COL_IS_COLD_TO_WARM_PROMOTION];
        let not_at = one.sub(at);

        let bodies = [
            at.mul(&at.sub(&one)),
            cold.mul(&cold.sub(&one)),
            warm.mul(&warm.sub(&one)),
            pw.mul(&pw.sub(&one)),
            ir.mul(&ir.sub(&one)),
            cold.add(warm).sub(ir),
            pw.mul(&one.sub(warm)),
            ir.mul(&not_at).mul(cold).mul(&gc.sub(&c2100)),
            ir.mul(&not_at).mul(warm).mul(&gc.sub(&c100)),
            ir.mul(at).mul(cold).mul(&gc.sub(&c2600)),
            ir.mul(at).mul(warm).mul(&gc.sub(&c100)),
            ic.mul(&ic.sub(&one)),
            pr.mul(&pr.sub(&one)),
            ir.mul(&pr.sub(cold)),
            one.sub(ir).mul(ic),
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
        let c100_p = vec![Scalar::from_u64(WARM_GAS, curve)];
        let c2100_p = vec![Scalar::from_u64(COLD_STORAGE_GAS, curve)];
        let c2600_p = vec![Scalar::from_u64(COLD_ADDRESS_GAS, curve)];

        let at = &cc[COL_ACCESS_TYPE];
        let cold = &cc[COL_IS_COLD];
        let warm = &cc[COL_IS_WARM];
        let pw = &cc[COL_PREWARM_FROM_EIP2930];
        let ir = &cc[COL_IS_REAL];
        let gc = &cc[COL_GAS_COST];
        let ic = &cc[COL_IS_CHAIN];
        let pr = &cc[COL_IS_COLD_TO_WARM_PROMOTION];

        let bin = |x: &Vec<Scalar>| poly_mul(x, &poly_sub(x, &one_p, curve), curve);

        let at_bin = bin(at);
        let cold_bin = bin(cold);
        let warm_bin = bin(warm);
        let pw_bin = bin(pw);
        let ir_bin = bin(ir);
        let ic_bin = bin(ic);
        let pr_bin = bin(pr);

        // ir * (pr - cold)
        let pr_minus_cold = poly_sub(pr, cold, curve);
        let promotion_eq_cold = poly_mul(ir, &pr_minus_cold, curve);

        // (1 - ir) * ic
        let one_minus_ir = poly_sub(&one_p, ir, curve);
        let padding_no_chain = poly_mul(&one_minus_ir, ic, curve);

        // cold + warm - ir
        let cw = poly_add(cold, warm, curve);
        let cw_eq_ir = poly_sub(&cw, ir, curve);

        // pw * (1 - warm)
        let one_minus_warm = poly_sub(&one_p, warm, curve);
        let pw_implies_warm = poly_mul(pw, &one_minus_warm, curve);

        let not_at = poly_sub(&one_p, at, curve);

        let g_m_2100 = poly_sub(gc, &c2100_p, curve);
        let g_m_100 = poly_sub(gc, &c100_p, curve);
        let g_m_2600 = poly_sub(gc, &c2600_p, curve);

        // is_real * (1 - access_type) * is_cold * (gas - 2100)
        let body7 = poly_mul(
            &poly_mul(&poly_mul(ir, &not_at, curve), cold, curve),
            &g_m_2100,
            curve,
        );
        // is_real * (1 - access_type) * is_warm * (gas - 100)
        let body8 = poly_mul(
            &poly_mul(&poly_mul(ir, &not_at, curve), warm, curve),
            &g_m_100,
            curve,
        );
        // is_real * access_type * is_cold * (gas - 2600)
        let body9 = poly_mul(
            &poly_mul(&poly_mul(ir, at, curve), cold, curve),
            &g_m_2600,
            curve,
        );
        // is_real * access_type * is_warm * (gas - 100)
        let body10 = poly_mul(
            &poly_mul(&poly_mul(ir, at, curve), warm, curve),
            &g_m_100,
            curve,
        );

        let bodies = [
            at_bin,
            cold_bin,
            warm_bin,
            pw_bin,
            ir_bin,
            cw_eq_ir,
            pw_implies_warm,
            body7,
            body8,
            body9,
            body10,
            ic_bin,
            pr_bin,
            promotion_eq_cold,
            padding_no_chain,
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
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..ADDRESS_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("access_2929_address_{}_8bit", k),
                    column_index: COL_ADDRESS_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..SLOT_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("access_2929_slot_{}_8bit", k),
                    column_index: COL_SLOT_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }

    // ─── Shifted constraint — multi-access pre-warm chain ────────────
    //
    //   is_chain[i] · ( is_warm[i+1] − max(is_warm[i], promotion[i]) ) = 0
    //
    // with max(a,b) = a + b − a·b for binary a, b. Excludes the wrap row
    // via the standard (X − ω^{n-1}) factor.

    fn shifted_column_indices(&self) -> Vec<usize> {
        // shifted[0] = IS_WARM_next
        vec![COL_IS_WARM]
    }

    fn num_shifted_constraints(&self) -> usize {
        NUM_SHIFTED
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
        if shifted_evals.is_empty() || col_evals_at_z.len() < NUM_COLUMNS {
            return Scalar::zero(curve);
        }
        let warm_next = &shifted_evals[0];
        let warm = &col_evals_at_z[COL_IS_WARM];
        let promo = &col_evals_at_z[COL_IS_COLD_TO_WARM_PROMOTION];
        let chain = &col_evals_at_z[COL_IS_CHAIN];

        // max(warm, promo) = warm + promo - warm*promo
        let max_wp = warm.add(promo).sub(&warm.mul(promo));
        let body = warm_next.sub(&max_wp);
        let row = chain.mul(&body);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        ap.mul(&row).mul(&z.sub(omega_n_minus_1))
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
        let warm = &column_coeffs[COL_IS_WARM];
        let warm_next = poly_shift(warm, omega);
        let promo = &column_coeffs[COL_IS_COLD_TO_WARM_PROMOTION];
        let chain = &column_coeffs[COL_IS_CHAIN];

        // max(warm, promo) = warm + promo - warm*promo
        let warm_promo = poly_mul(warm, promo, curve);
        let warm_plus_promo = poly_add(warm, promo, curve);
        let max_wp = poly_sub(&warm_plus_promo, &warm_promo, curve);

        // body = warm_next - max_wp
        let body = poly_sub(&warm_next, &max_wp, curve);

        // row = chain * body
        let row = poly_mul(chain, &body, curve);

        // Multiply by (X − ω^{n-1}) to exclude the wrap row, matching the
        // verifier's `(z − ω^{n-1})` factor in `evaluate_shifted_at_point`.
        let n_minus_1 = domain_size.saturating_sub(1);
        let mut omega_n_minus_1 = Scalar::one(curve);
        let mut e = n_minus_1;
        let mut base_pow = omega.clone();
        while e > 0 {
            if e & 1 == 1 {
                omega_n_minus_1 = omega_n_minus_1.mul(&base_pow);
            }
            base_pow = base_pow.mul(&base_pow);
            e >>= 1;
        }
        let row_excluded = poly_mul_linear(&row, &omega_n_minus_1);

        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        poly_scalar_mul(&row_excluded, &ap)
    }
}

// ─── Cross-AIR LogUp descriptors ─────────────────────────────────────

/// Binds rows with `prewarm_from_eip2930 = 1` on this AIR to the
/// EIP-2930 access-list AIR's `(address, storage_key_0)` columns. This
/// proves the host-side pre-warm flag is honest: only `(address, slot)`
/// pairs actually present in the transaction's EIP-2930 access list are
/// marked as pre-warmed.
///
/// Note: a full multi-key access-list entry exposes up to 8 storage
/// keys; this descriptor binds only against slot 0 of each entry as a
/// scaffold. Multi-key per-entry binding requires either widening this
/// AIR to materialize each `(address, key_k)` pair as a separate row
/// or splitting the descriptor by key index — both deferred.
pub fn make_access_2929_to_access_list_descriptor(
    access_2929_layer_index: usize,
    access_list_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns = Vec::with_capacity(ADDRESS_BYTES + SLOT_BYTES);
    let mut b_columns = Vec::with_capacity(ADDRESS_BYTES + SLOT_BYTES);
    for k in 0..ADDRESS_BYTES {
        a_columns.push(COL_ADDRESS_OFFSET + k);
        b_columns.push(metavm_zkp::access_list_air::COL_ADDRESS_OFFSET + k);
    }
    for k in 0..SLOT_BYTES {
        a_columns.push(COL_SLOT_OFFSET + k);
        b_columns.push(metavm_zkp::access_list_air::COL_KEY_BYTES_OFFSET + k);
    }
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "access_2929_to_access_list_prewarm_v1".into(),
        a_layer_index: access_2929_layer_index,
        a_columns,
        a_selector_column: Some(COL_PREWARM_FROM_EIP2930),
        b_layer_index: access_list_layer_index,
        b_columns,
        b_selector_column: Some(metavm_zkp::access_list_air::COL_IS_REAL),
    }
}

/// Binds this AIR's `(opcode_marker = 0, gas_cost)` tuple to the
/// gas-tracking AIR's `(opcode, static_cost)`. Because the EIP-2929
/// charges are layered on top of the per-opcode static cost and depend
/// on dynamic per-tx context (warm/cold), the natural binding is to
/// project just the `gas_cost` column onto the gas-tracking AIR's
/// `static_cost` slot — leaving the host-side oracle to reconcile
/// per-opcode dispatch. This is a single-column binding gated by
/// `is_real`.
pub fn make_access_2929_to_gas_tracking_descriptor(
    access_2929_layer_index: usize,
    gas_tracking_layer_index: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "access_2929_to_gas_tracking_v1".into(),
        a_layer_index: access_2929_layer_index,
        a_columns: vec![COL_GAS_COST],
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: gas_tracking_layer_index,
        b_columns: vec![crate::gas_tracking_air::COL_STATIC_COST],
        b_selector_column: Some(crate::gas_tracking_air::COL_IS_REAL),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(b: u8) -> [u8; 20] {
        [b; 20]
    }
    fn slot(b: u8) -> [u8; 32] {
        [b; 32]
    }

    fn assert_all_zero(t: &TracePolynomials) {
        let cs = AccessListEip2929ConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
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
    fn cold_address_access_charged_2600() {
        let accesses =
            vec![(AccessKind::Address, addr(0x11), slot(0), false)];
        let w = from_accesses(&accesses);
        assert_eq!(w.rows.len(), 1);
        assert!(w.rows[0].is_cold);
        assert!(!w.rows[0].is_warm);
        assert!(!w.rows[0].prewarm_from_eip2930);
        assert_eq!(w.rows[0].gas_cost, 2600);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn warm_repeat_address_charged_100() {
        let accesses = vec![
            (AccessKind::Address, addr(0x22), slot(0), false),
            (AccessKind::Address, addr(0x22), slot(0), false),
        ];
        let w = from_accesses(&accesses);
        assert_eq!(w.rows[0].gas_cost, 2600);
        assert!(w.rows[0].is_cold);
        assert_eq!(w.rows[1].gas_cost, 100);
        assert!(w.rows[1].is_warm);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn cold_then_warm_storage_slot() {
        let accesses = vec![
            (AccessKind::StorageSlot, addr(0x33), slot(0x01), false),
            (AccessKind::StorageSlot, addr(0x33), slot(0x01), false),
            // Different slot on same address is still cold for the slot.
            (AccessKind::StorageSlot, addr(0x33), slot(0x02), false),
        ];
        let w = from_accesses(&accesses);
        assert_eq!(w.rows[0].gas_cost, 2100);
        assert!(w.rows[0].is_cold);
        assert_eq!(w.rows[1].gas_cost, 100);
        assert!(w.rows[1].is_warm);
        assert_eq!(w.rows[2].gas_cost, 2100);
        assert!(w.rows[2].is_cold);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn eip2930_prewarmed_storage_is_warm_on_first_touch() {
        let accesses =
            vec![(AccessKind::StorageSlot, addr(0x44), slot(0x55), true)];
        let w = from_accesses(&accesses);
        assert!(w.rows[0].prewarm_from_eip2930);
        assert!(w.rows[0].is_warm);
        assert!(!w.rows[0].is_cold);
        assert_eq!(w.rows[0].gas_cost, 100);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }

    #[test]
    fn tampered_gas_cost_detected() {
        // Honest cold-storage access at 2100 → tamper to 999.
        let accesses =
            vec![(AccessKind::StorageSlot, addr(0x66), slot(0x77), false)];
        let w = from_accesses(&accesses);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cv: Vec<Vec<Scalar>> =
            t.columns.iter().map(|p| p.evaluations.clone()).collect();
        cv[COL_GAS_COST][0] = Scalar::from_u64(999, CurveType::Bls48581);
        let cs = AccessListEip2929ConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cv.iter().collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 7 = storage_cold_gas_eq_2100 must fire.
        assert!(
            !bodies[7][0].is_zero(),
            "tampered storage-cold gas not caught"
        );
    }

    #[test]
    fn tampered_prewarm_without_warm_detected() {
        // Manually build a row where prewarm = 1 but is_warm = 0 (impossible).
        let curve = CurveType::Bls48581;
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        let n = 2;
        let mut cv: Vec<Vec<Scalar>> =
            (0..NUM_COLUMNS).map(|_| vec![zero.clone(); n]).collect();
        // Row 0: claim cold storage at 2100 but mark prewarm=1 (should
        // force is_warm=1 → contradiction).
        cv[COL_ACCESS_TYPE][0] = zero.clone(); // storage
        cv[COL_GAS_COST][0] = Scalar::from_u64(2100, curve);
        cv[COL_IS_COLD][0] = one.clone();
        cv[COL_IS_WARM][0] = zero.clone();
        cv[COL_PREWARM_FROM_EIP2930][0] = one.clone();
        cv[COL_IS_REAL][0] = one.clone();
        let cs = AccessListEip2929ConstraintSystem::new(n);
        let cr: Vec<&Vec<Scalar>> = cv.iter().collect();
        let bodies = cs.evaluate_on_domain(&cr, n);
        // Constraint 6 = prewarm_implies_warm must fire.
        assert!(
            !bodies[6][0].is_zero(),
            "prewarm-implies-warm constraint not caught"
        );
    }

    #[test]
    fn tampered_cold_and_warm_both_set_detected() {
        let accesses =
            vec![(AccessKind::Address, addr(0x88), slot(0), false)];
        let w = from_accesses(&accesses);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cv: Vec<Vec<Scalar>> =
            t.columns.iter().map(|p| p.evaluations.clone()).collect();
        let one = Scalar::one(CurveType::Bls48581);
        // Originally is_cold=1, is_warm=0; force is_warm=1 too.
        cv[COL_IS_WARM][0] = one.clone();
        let cs = AccessListEip2929ConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = cv.iter().collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // Constraint 5 = cold_plus_warm_eq_real (1+1 != 1).
        assert!(!bodies[5][0].is_zero(), "both-set not caught");
    }

    #[test]
    fn descriptors_well_formed() {
        let d_al = make_access_2929_to_access_list_descriptor(0, 1);
        assert_eq!(d_al.label, "access_2929_to_access_list_prewarm_v1");
        assert_eq!(d_al.a_columns.len(), ADDRESS_BYTES + SLOT_BYTES);
        assert_eq!(d_al.b_columns.len(), ADDRESS_BYTES + SLOT_BYTES);
        assert_eq!(d_al.a_layer_index, 0);
        assert_eq!(d_al.b_layer_index, 1);
        assert_eq!(d_al.a_selector_column, Some(COL_PREWARM_FROM_EIP2930));
        assert_eq!(
            d_al.b_selector_column,
            Some(metavm_zkp::access_list_air::COL_IS_REAL),
        );

        let d_gt = make_access_2929_to_gas_tracking_descriptor(0, 2);
        assert_eq!(d_gt.label, "access_2929_to_gas_tracking_v1");
        assert_eq!(d_gt.a_columns.len(), 1);
        assert_eq!(d_gt.b_columns.len(), 1);
        assert_eq!(d_gt.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            d_gt.b_selector_column,
            Some(crate::gas_tracking_air::COL_IS_REAL),
        );
    }

    #[test]
    fn num_columns_and_constraints_pinned() {
        assert_eq!(NUM_COLUMNS, 61);
        assert_eq!(NUM_ROW_CONSTRAINTS, 15);
        assert_eq!(NUM_SHIFTED, 1);
        assert_eq!(COL_SLOT_END, 61);
        assert_eq!(COL_IS_CHAIN, 7);
        assert_eq!(COL_IS_COLD_TO_WARM_PROMOTION, 8);
    }

    #[test]
    fn evaluate_at_point_zero_on_honest_mixed_trace() {
        let accesses = vec![
            (AccessKind::Address, addr(0xaa), slot(0), false),
            (AccessKind::Address, addr(0xaa), slot(0), false),
            (AccessKind::StorageSlot, addr(0xbb), slot(0x01), true),
            (AccessKind::StorageSlot, addr(0xbb), slot(0x02), false),
        ];
        let w = from_accesses(&accesses);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AccessListEip2929ConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0x4242, CurveType::Bls48581);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for r in 0..w.rows.len() {
            let row_evals: Vec<Scalar> =
                cr.iter().map(|c| c[r].clone()).collect();
            let v = cs.evaluate_at_point(&row_evals, &alpha);
            assert!(v.is_zero(), "row {} nonzero", r);
        }
    }

    // ─── Multi-access pre-warm chain tests ───────────────────────────

    /// 5-access trace validating the cold→warm chain end-to-end.
    ///
    /// Five SLOAD-style accesses to the same `(address, slot)`:
    ///   row 0 = cold (first touch, 2100 gas, promotion=1)
    ///   row 1..4 = warm (100 gas, promotion=0)
    ///
    /// The witness builder sets `is_chain[i]` = 1 for i ∈ {0,1,2,3} and
    /// `is_chain[4] = 0` (last row of chain). The shifted constraint
    /// must vanish on every real row and the row-local pre-warm chain
    /// constraints (11..14) must all evaluate to zero.
    #[test]
    fn multi_access_prewarm_chain_5_accesses() {
        let a = addr(0xcc);
        let s = slot(0xde);
        let accesses = vec![
            (AccessKind::StorageSlot, a, s, false),
            (AccessKind::StorageSlot, a, s, false),
            (AccessKind::StorageSlot, a, s, false),
            (AccessKind::StorageSlot, a, s, false),
            (AccessKind::StorageSlot, a, s, false),
        ];
        let w = from_accesses(&accesses);
        assert_eq!(w.rows.len(), 5);
        // Row 0 cold.
        assert!(w.rows[0].is_cold);
        assert_eq!(w.rows[0].gas_cost, 2100);
        assert!(w.rows[0].is_cold_to_warm_promotion);
        // Rows 1..4 warm.
        for i in 1..5 {
            assert!(w.rows[i].is_warm, "row {} should be warm", i);
            assert_eq!(w.rows[i].gas_cost, 100);
            assert!(!w.rows[i].is_cold_to_warm_promotion);
        }
        // Chain bits: 1 on rows 0..3, 0 on row 4.
        for i in 0..4 {
            assert!(w.rows[i].is_chain, "row {} chain bit must be 1", i);
        }
        assert!(!w.rows[4].is_chain, "row 4 chain bit must be 0 (tail)");

        let curve = CurveType::Bls48581;
        let t = build_trace_polynomials(&w, curve);
        let _cs = AccessListEip2929ConstraintSystem::new(t.num_rows);

        // Row-local constraints all vanish.
        assert_all_zero(&t);

        // Shifted constraint: walk adjacent real-row pairs (excluding wrap).
        let cols: Vec<&Vec<Scalar>> =
            t.columns.iter().map(|p| &p.evaluations).collect();
        let n_real = w.rows.len();
        for i in 0..n_real.saturating_sub(1) {
            let chain = &cols[COL_IS_CHAIN][i];
            let warm = &cols[COL_IS_WARM][i];
            let promo = &cols[COL_IS_COLD_TO_WARM_PROMOTION][i];
            let warm_next = &cols[COL_IS_WARM][i + 1];
            // max(warm, promo) = warm + promo - warm*promo
            let max_wp = warm.add(promo).sub(&warm.mul(promo));
            let body = warm_next.sub(&max_wp);
            let row = chain.mul(&body);
            assert!(
                row.is_zero(),
                "shifted body must vanish at row {} (chain={:?}, warm[{}]={:?}, promo[{}]={:?}, warm[{}]={:?})",
                i,
                chain,
                i,
                warm,
                i,
                promo,
                i + 1,
                warm_next,
            );
        }

        // Tamper test: flip warm[1] to 0 (still consistent with row-local
        // is_cold+is_warm=is_real if we also flip cold[1]=1, but then the
        // row-local promotion constraint and the shifted chain both fire).
        let mut cv: Vec<Vec<Scalar>> =
            t.columns.iter().map(|p| p.evaluations.clone()).collect();
        let one = Scalar::one(curve);
        let zero = Scalar::zero(curve);
        cv[COL_IS_WARM][1] = zero.clone();
        cv[COL_IS_COLD][1] = one.clone();
        // Tampered gas (warm → cold gas) to defeat constraint 8, leaving
        // the shifted chain as the catching constraint.
        cv[COL_GAS_COST][1] = Scalar::from_u64(2100, curve);
        // Update promotion bit too so the row-local constraint 13
        // (ir * (promo - cold)) still vanishes — this isolates the
        // shifted chain as the catching constraint.
        cv[COL_IS_COLD_TO_WARM_PROMOTION][1] = one.clone();

        // Now check chain shifted body at row 0:
        // chain[0]=1, warm[0]=0, promo[0]=1 ⇒ max=1, warm_next[1]=0 ⇒ body=-1
        // chain * body = -1 ≠ 0.
        let chain0 = &cv[COL_IS_CHAIN][0];
        let warm0 = &cv[COL_IS_WARM][0];
        let promo0 = &cv[COL_IS_COLD_TO_WARM_PROMOTION][0];
        let warm1 = &cv[COL_IS_WARM][1];
        let max_wp = warm0.add(promo0).sub(&warm0.mul(promo0));
        let body = warm1.sub(&max_wp);
        let row = chain0.mul(&body);
        assert!(
            !row.is_zero(),
            "tampered warm[1]=0 must violate the shifted chain at row 0"
        );
    }

    /// Mixed-entry chain: distinct `(address, slot)` interleaved with
    /// repeats. The witness builder marks chain only where adjacent rows
    /// share `(access_type, address, slot, tx_index)`.
    #[test]
    fn multi_access_chain_mixed_entries() {
        let a1 = addr(0x01);
        let a2 = addr(0x02);
        let s1 = slot(0x10);
        let s2 = slot(0x20);
        let accesses = vec![
            (AccessKind::StorageSlot, a1, s1, false), // cold
            (AccessKind::StorageSlot, a1, s1, false), // warm (chain w/ row 0)
            (AccessKind::StorageSlot, a2, s2, false), // cold (different entry)
            (AccessKind::StorageSlot, a2, s2, false), // warm (chain w/ row 2)
            (AccessKind::StorageSlot, a1, s1, false), // warm (a1/s1 still warm — but not chain w/ row 3)
        ];
        let w = from_accesses(&accesses);
        // Chain bits: only adjacent same-entry rows.
        assert!(w.rows[0].is_chain);
        assert!(!w.rows[1].is_chain); // row 2 is different entry
        assert!(w.rows[2].is_chain);
        assert!(!w.rows[3].is_chain); // row 4 is different entry
        assert!(!w.rows[4].is_chain); // tail
        // Warm/cold semantics still honored across the whole tx.
        assert_eq!(w.rows[0].gas_cost, 2100);
        assert_eq!(w.rows[1].gas_cost, 100);
        assert_eq!(w.rows[2].gas_cost, 2100);
        assert_eq!(w.rows[3].gas_cost, 100);
        assert_eq!(w.rows[4].gas_cost, 100); // a1/s1 was warmed earlier
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_all_zero(&t);
    }
}
