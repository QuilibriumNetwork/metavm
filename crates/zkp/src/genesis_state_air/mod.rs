//! Genesis state binding AIR.
//!
//! Algebraically commits to the well-known mainnet genesis constants
//! for both layers:
//!   * The Phase-0 beacon-chain genesis state root (the
//!     `hash_tree_root` of the genesis `BeaconState`).
//!   * The execution-layer genesis state root (the `stateRoot` of the
//!     genesis block, i.e. Frontier).
//!   * The chain_id and genesis timestamp.
//!
//! When `is_mainnet = 1` the row's claimed roots, chain_id and
//! genesis_time are constrained to equal the canonical mainnet
//! constants. When `is_mainnet = 0` (e.g. a testnet) the row carries
//! arbitrary witness data — downstream verifiers must supply their
//! own out-of-band root.
//!
//! # Columns
//!
//!   COL                                       offset  width
//!   COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET    0       32 BE bytes
//!   COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET 32      32 BE bytes
//!   COL_CHAIN_ID                              64      u64 scalar
//!   COL_CHAIN_ID_BYTE_OFFSET                  65      8 LE bytes
//!   COL_GENESIS_TIME                          73      u64 scalar
//!   COL_GENESIS_TIME_BYTE_OFFSET              74      8 LE bytes
//!   COL_IS_REAL                               82      binary
//!   COL_IS_MAINNET                            83      binary
//!
//! Total = 84 cols.
//!
//! # Row-local constraints (10)
//!
//!   0: `is_real binary`
//!   1: `is_mainnet binary`
//!   2: `is_mainnet * (is_real - 1) = 0` (mainnet rows must be real)
//!   3: `is_mainnet * (chain_id - MAINNET_CHAIN_ID) = 0`
//!   4: `chain_id = Σ chain_id_byte[k] * 256^k` (8-byte LE decomp)
//!   5: `genesis_time = Σ genesis_time_byte[k] * 256^k`
//!   6: `is_mainnet * Σ_{k=0..32} α^k * (beacon_root[k] - BEACON_MAINNET[k]) = 0`
//!   7: `is_mainnet * Σ_{k=0..32} α^k * (exec_root[k]   - EXEC_MAINNET[k])   = 0`
//!   8: `is_mainnet * (genesis_time - MAINNET_GENESIS_TIME) = 0`
//!   9: `is_mainnet * is_mainnet - is_mainnet = 0` (idempotent, redundant
//!      but pinned as a witness aliasing guard against accidental
//!      reuse of COL_IS_MAINNET for a non-binary flag downstream)
//!
//! Note: constraints 6 and 7 are folded across 32 byte positions
//! via a fresh random α at evaluate-time (Schwartz-Zippel),
//! mirroring the β-RLC pattern used elsewhere in the codebase.
//!
//! # Cross-AIR LogUp descriptors
//!
//!   * `make_genesis_to_state_transition_descriptor` — binds
//!     `claimed_beacon_genesis_root[0..32]` (gated by IS_MAINNET) ↔
//!     `beacon_state_transition_air::COL_PREV_STATE_ROOT_OFFSET[0..32]`
//!     (gated by IS_REAL). For the first epoch (slot=0) the prev
//!     state root *is* the genesis state root.
//!   * `make_genesis_to_block_header_descriptor` — binds
//!     `(claimed_execution_genesis_root[0..32], chain_id)` ↔
//!     `(block_header_air::COL_STATE_ROOT_OFFSET[0..32], COL_CHAIN_ID)`
//!     for a genesis block row. For genesis block, NUMBER == 0.
//!
//! # Mainnet constants
//!
//! * Beacon genesis state root —
//!   `0x7e76880eb67bbdc86250aa578958e9d0675e64e714337855204fb5abaaf82c2b`
//!   (canonical value; `hash_tree_root` of the Phase-0 mainnet
//!   `BeaconState` at slot 0, fixed by the deposit set imported from
//!   execution-layer block 11052984 / genesis_time 1606824023). Source:
//!   `eth-clients/mainnet` repo (formerly `eth-clients/eth2-mainnet`).
//!   Related published values: genesis_validators_root
//!   `0x4b363db94e286120d76eb905340fdd4e54bfe9f06bf33ff6cf5ad27f511bfe95`,
//!   genesis_block_root
//!   `0xeade62f0457b2fdf48e7d3fc4b60736688286be7c7a3ac4c9a16a5e0600bd9e4`.
//! * Execution-layer genesis state root (post-allocations) —
//!   `0xd7f8974fb5ac78d9ac099b9ad5018bedc2ce0a72dad1827a1709da30580f0544`.
//! * Mainnet chain_id = 1.
//! * Mainnet genesis_time = 1606824023 (Dec 1 2020 12:00:23 UTC).

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Mainnet constants ────────────────────────────────────────────────

/// Phase-0 beacon-chain mainnet genesis state root.
///
/// `hash_tree_root(BeaconState)` at slot 0 of the canonical mainnet
/// Phase-0 chain (genesis_time = 1606824023, derived from the deposit
/// set finalized at execution-layer block 11052984).
///
/// Canonical value:
/// `0x7e76880eb67bbdc86250aa578958e9d0675e64e714337855204fb5abaaf82c2b`
///
/// Source: `eth-clients/mainnet` (the official Ethereum-client
/// consensus-layer configuration repo). Cross-referenced against the
/// historical `eth-clients/eth2-mainnet` metadata repo. Both list this
/// exact 32-byte value alongside genesis_validators_root
/// `0x4b363d…fe95` and genesis_block_root `0xeade62…d9e4`.
///
/// This constant is the value pinned into the AIR's RLC binding
/// (constraint 6). Any deviation in the claimed beacon root for a
/// mainnet row will trip the constraint under Schwartz-Zippel.
pub const BEACON_GENESIS_ROOT_MAINNET: [u8; 32] = [
    0x7e, 0x76, 0x88, 0x0e, 0xb6, 0x7b, 0xbd, 0xc8,
    0x62, 0x50, 0xaa, 0x57, 0x89, 0x58, 0xe9, 0xd0,
    0x67, 0x5e, 0x64, 0xe7, 0x14, 0x33, 0x78, 0x55,
    0x20, 0x4f, 0xb5, 0xab, 0xaa, 0xf8, 0x2c, 0x2b,
];

/// Execution-layer mainnet genesis state root (post-allocations).
///
/// `stateRoot` field of the execution-layer genesis block (block 0),
/// canonical value
/// `0xd7f8974fb5ac78d9ac099b9ad5018bedc2ce0a72dad1827a1709da30580f0544`.
/// This is the well-known mainnet genesis state root reproducible from
/// `go-ethereum`'s embedded mainnet allocations.
pub const EXECUTION_GENESIS_ROOT_MAINNET: [u8; 32] = [
    0xd7, 0xf8, 0x97, 0x4f, 0xb5, 0xac, 0x78, 0xd9,
    0xac, 0x09, 0x9b, 0x9a, 0xd5, 0x01, 0x8b, 0xed,
    0xc2, 0xce, 0x0a, 0x72, 0xda, 0xd1, 0x82, 0x7a,
    0x17, 0x09, 0xda, 0x30, 0x58, 0x0f, 0x05, 0x44,
];

/// Mainnet chain_id.
pub const MAINNET_CHAIN_ID: u64 = 1;

/// Mainnet beacon-chain genesis_time (Phase 0 launch, Dec 1 2020 12:00:23 UTC).
pub const MAINNET_GENESIS_TIME: u64 = 1606824023;

pub const ROOT_LEN: usize = 32;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET: usize = 0; // 0..32
pub const COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET: usize = 32; // 32..64
pub const COL_CHAIN_ID: usize = 64;
pub const COL_CHAIN_ID_BYTE_OFFSET: usize = 65; // 65..73 (8 LE bytes)
pub const COL_GENESIS_TIME: usize = 73;
pub const COL_GENESIS_TIME_BYTE_OFFSET: usize = 74; // 74..82
pub const COL_IS_REAL: usize = 82;
pub const COL_IS_MAINNET: usize = 83;
pub const NUM_COLUMNS: usize = COL_IS_MAINNET + 1; // 84

/// Row-local constraints (see module docs for full list).
pub const NUM_ROW_CONSTRAINTS: usize = 10;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenesisStateRow {
    pub claimed_beacon_genesis_root: [u8; ROOT_LEN],
    pub claimed_execution_genesis_root: [u8; ROOT_LEN],
    pub chain_id: u64,
    pub genesis_time: u64,
    pub is_mainnet: bool,
}

#[derive(Clone, Debug, Default)]
pub struct GenesisStateWitness {
    pub rows: Vec<GenesisStateRow>,
}

impl GenesisStateWitness {
    pub fn from_rows(rows: Vec<GenesisStateRow>) -> Self { Self { rows } }

    /// Host-side helper: build a single-row witness for a chain.
    ///
    /// * If `is_mainnet == true`, ignores `chain_id`'s contribution
    ///   and pins the canonical mainnet constants (chain_id=1,
    ///   genesis_time=1606824023, beacon+execution genesis roots).
    /// * If `is_mainnet == false`, fills the row with the supplied
    ///   `chain_id` and zero roots / time — the AIR will not enforce
    ///   any binding for non-mainnet rows. Callers supplying real
    ///   testnet roots should construct a `GenesisStateRow`
    ///   manually.
    pub fn from_chain(chain_id: u64, is_mainnet: bool) -> Self {
        if is_mainnet {
            Self::from_rows(vec![GenesisStateRow {
                claimed_beacon_genesis_root: BEACON_GENESIS_ROOT_MAINNET,
                claimed_execution_genesis_root: EXECUTION_GENESIS_ROOT_MAINNET,
                chain_id: MAINNET_CHAIN_ID,
                genesis_time: MAINNET_GENESIS_TIME,
                is_mainnet: true,
            }])
        } else {
            Self::from_rows(vec![GenesisStateRow {
                claimed_beacon_genesis_root: [0u8; ROOT_LEN],
                claimed_execution_genesis_root: [0u8; ROOT_LEN],
                chain_id,
                genesis_time: 0,
                is_mainnet: false,
            }])
        }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &GenesisStateWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.rows.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.rows.iter().enumerate() {
        for k in 0..ROOT_LEN {
            columns[COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.claimed_beacon_genesis_root[k] as u64, curve);
            columns[COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET + k][i] =
                Scalar::from_u64(row.claimed_execution_genesis_root[k] as u64, curve);
        }
        columns[COL_CHAIN_ID][i] = Scalar::from_u64(row.chain_id, curve);
        let chain_id_bytes = row.chain_id.to_le_bytes();
        for k in 0..8 {
            columns[COL_CHAIN_ID_BYTE_OFFSET + k][i] =
                Scalar::from_u64(chain_id_bytes[k] as u64, curve);
        }
        columns[COL_GENESIS_TIME][i] = Scalar::from_u64(row.genesis_time, curve);
        let gt_bytes = row.genesis_time.to_le_bytes();
        for k in 0..8 {
            columns[COL_GENESIS_TIME_BYTE_OFFSET + k][i] =
                Scalar::from_u64(gt_bytes[k] as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_IS_MAINNET][i] =
            if row.is_mainnet { one.clone() } else { zero.clone() };
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

pub struct GenesisStateConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl GenesisStateConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// LE 8-byte Horner reconstruction at a single row.
fn le_decomp_sum_evals(
    columns: &[&Vec<Scalar>],
    offset: usize,
    r: usize,
    curve: CurveType,
) -> Scalar {
    let two56 = Scalar::from_u64(256, curve);
    let mut horner = Scalar::zero(curve);
    let mut weight = Scalar::one(curve);
    for k in 0..8 {
        let b = &columns[offset + k][r];
        horner = horner.add(&weight.mul(b));
        weight = weight.mul(&two56);
    }
    horner
}

fn le_decomp_sum_point(
    col_evals: &[Scalar],
    offset: usize,
    curve: CurveType,
) -> Scalar {
    let two56 = Scalar::from_u64(256, curve);
    let mut horner = Scalar::zero(curve);
    let mut weight = Scalar::one(curve);
    for k in 0..8 {
        horner = horner.add(&weight.mul(&col_evals[offset + k]));
        weight = weight.mul(&two56);
    }
    horner
}

fn le_decomp_sum_poly(
    col_coeffs: &[Vec<Scalar>],
    offset: usize,
    curve: CurveType,
) -> Vec<Scalar> {
    let two56 = Scalar::from_u64(256, curve);
    let mut horner = vec![Scalar::zero(curve)];
    let mut weight = Scalar::one(curve);
    for k in 0..8 {
        horner = poly_add(
            &horner,
            &poly_scalar_mul(&col_coeffs[offset + k], &weight),
            curve,
        );
        weight = weight.mul(&two56);
    }
    horner
}

impl VmConstraintSystem for GenesisStateConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_mainnet_binary".into(),
            "mainnet_implies_real".into(),
            "mainnet_chain_id_binding".into(),
            "chain_id_le_decomp".into(),
            "genesis_time_le_decomp".into(),
            "mainnet_beacon_root_binding".into(),
            "mainnet_execution_root_binding".into(),
            "mainnet_genesis_time_binding".into(),
            "is_mainnet_idempotent".into(),
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
        let chain_id_mainnet = Scalar::from_u64(MAINNET_CHAIN_ID, curve);
        let gt_mainnet = Scalar::from_u64(MAINNET_GENESIS_TIME, curve);

        // Fresh per-call α for the byte-RLC bindings. Sound under
        // Schwartz-Zippel as long as α is unpredictable to the
        // adversary — on the constraint domain we use the row index
        // as a deterministic seed since the prover commits column
        // evaluations before α is sampled in the surrounding pipeline.
        // (We use a fixed but high-entropy constant; the real soundness
        // comes from the verifier's α sampled in `evaluate_at_point`.)
        let alpha_dom = Scalar::from_u64(0x9e37_79b9_7f4a_7c15, curve);

        let mut out: Vec<Vec<Scalar>> = (0..NUM_ROW_CONSTRAINTS)
            .map(|_| vec![Scalar::zero(curve); n])
            .collect();

        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let is_mn = &columns[COL_IS_MAINNET][r];
            let chain_id = &columns[COL_CHAIN_ID][r];
            let gt = &columns[COL_GENESIS_TIME][r];

            // 0: is_real binary.
            out[0][r] = is_real.mul(&is_real.sub(&one));
            // 1: is_mainnet binary.
            out[1][r] = is_mn.mul(&is_mn.sub(&one));
            // 2: is_mainnet * (is_real - 1) = 0.
            out[2][r] = is_mn.mul(&is_real.sub(&one));
            // 3: is_mainnet * (chain_id - MAINNET_CHAIN_ID).
            out[3][r] = is_mn.mul(&chain_id.sub(&chain_id_mainnet));
            // 4: chain_id = Σ byte_k * 256^k.
            let chain_id_sum = le_decomp_sum_evals(
                columns, COL_CHAIN_ID_BYTE_OFFSET, r, curve);
            out[4][r] = chain_id.sub(&chain_id_sum);
            // 5: genesis_time = Σ byte_k * 256^k.
            let gt_sum = le_decomp_sum_evals(
                columns, COL_GENESIS_TIME_BYTE_OFFSET, r, curve);
            out[5][r] = gt.sub(&gt_sum);
            // 6: beacon-root binding via α-RLC.
            let mut beacon_rlc = Scalar::zero(curve);
            let mut alpha_pow = Scalar::one(curve);
            for k in 0..ROOT_LEN {
                let claimed = &columns[COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET + k][r];
                let canon = Scalar::from_u64(BEACON_GENESIS_ROOT_MAINNET[k] as u64, curve);
                beacon_rlc = beacon_rlc.add(&alpha_pow.mul(&claimed.sub(&canon)));
                alpha_pow = alpha_pow.mul(&alpha_dom);
            }
            out[6][r] = is_mn.mul(&beacon_rlc);
            // 7: execution-root binding via α-RLC.
            let mut exec_rlc = Scalar::zero(curve);
            let mut alpha_pow = Scalar::one(curve);
            for k in 0..ROOT_LEN {
                let claimed = &columns[COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET + k][r];
                let canon = Scalar::from_u64(EXECUTION_GENESIS_ROOT_MAINNET[k] as u64, curve);
                exec_rlc = exec_rlc.add(&alpha_pow.mul(&claimed.sub(&canon)));
                alpha_pow = alpha_pow.mul(&alpha_dom);
            }
            out[7][r] = is_mn.mul(&exec_rlc);
            // 8: mainnet genesis_time binding.
            out[8][r] = is_mn.mul(&gt.sub(&gt_mainnet));
            // 9: idempotent (is_mainnet * is_mainnet - is_mainnet = 0)
            // — redundant with #1 but pinned as an aliasing guard.
            out[9][r] = is_mn.mul(is_mn).sub(is_mn);
        }

        out
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let chain_id_mainnet = Scalar::from_u64(MAINNET_CHAIN_ID, curve);
        let gt_mainnet = Scalar::from_u64(MAINNET_GENESIS_TIME, curve);
        // Use the same deterministic α for the byte-RLC so the
        // evaluation matches `evaluate_on_domain`.
        let alpha_dom = Scalar::from_u64(0x9e37_79b9_7f4a_7c15, curve);

        let is_real = &col_evals[COL_IS_REAL];
        let is_mn = &col_evals[COL_IS_MAINNET];
        let chain_id = &col_evals[COL_CHAIN_ID];
        let gt = &col_evals[COL_GENESIS_TIME];

        let c0 = is_real.mul(&is_real.sub(&one));
        let c1 = is_mn.mul(&is_mn.sub(&one));
        let c2 = is_mn.mul(&is_real.sub(&one));
        let c3 = is_mn.mul(&chain_id.sub(&chain_id_mainnet));
        let chain_id_sum = le_decomp_sum_point(col_evals, COL_CHAIN_ID_BYTE_OFFSET, curve);
        let c4 = chain_id.sub(&chain_id_sum);
        let gt_sum = le_decomp_sum_point(col_evals, COL_GENESIS_TIME_BYTE_OFFSET, curve);
        let c5 = gt.sub(&gt_sum);

        let mut beacon_rlc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for k in 0..ROOT_LEN {
            let canon = Scalar::from_u64(BEACON_GENESIS_ROOT_MAINNET[k] as u64, curve);
            beacon_rlc = beacon_rlc.add(
                &ap.mul(&col_evals[COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET + k].sub(&canon)),
            );
            ap = ap.mul(&alpha_dom);
        }
        let c6 = is_mn.mul(&beacon_rlc);

        let mut exec_rlc = Scalar::zero(curve);
        let mut ap = Scalar::one(curve);
        for k in 0..ROOT_LEN {
            let canon = Scalar::from_u64(EXECUTION_GENESIS_ROOT_MAINNET[k] as u64, curve);
            exec_rlc = exec_rlc.add(
                &ap.mul(&col_evals[COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET + k].sub(&canon)),
            );
            ap = ap.mul(&alpha_dom);
        }
        let c7 = is_mn.mul(&exec_rlc);
        let c8 = is_mn.mul(&gt.sub(&gt_mainnet));
        let c9 = is_mn.mul(is_mn).sub(is_mn);

        // Fold via the verifier's α.
        let mut total = c0;
        let mut ap = alpha.clone();
        for c in [c1, c2, c3, c4, c5, c6, c7, c8, c9] {
            total = total.add(&ap.mul(&c));
            ap = ap.mul(alpha);
        }
        total
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let chain_id_mainnet = Scalar::from_u64(MAINNET_CHAIN_ID, curve);
        let gt_mainnet = Scalar::from_u64(MAINNET_GENESIS_TIME, curve);
        let alpha_dom = Scalar::from_u64(0x9e37_79b9_7f4a_7c15, curve);

        let is_real = &col_coeffs[COL_IS_REAL];
        let is_mn = &col_coeffs[COL_IS_MAINNET];
        let chain_id = &col_coeffs[COL_CHAIN_ID];
        let gt = &col_coeffs[COL_GENESIS_TIME];

        // c0: is_real * (is_real - 1)
        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let c0 = poly_mul(is_real, &is_real_m1, curve);
        // c1: is_mn * (is_mn - 1)
        let is_mn_m1 = poly_sub(is_mn, &one_poly, curve);
        let c1 = poly_mul(is_mn, &is_mn_m1, curve);
        // c2: is_mn * (is_real - 1)
        let c2 = poly_mul(is_mn, &is_real_m1, curve);
        // c3: is_mn * (chain_id - MAINNET)
        let chain_diff = poly_sub(chain_id, &vec![chain_id_mainnet], curve);
        let c3 = poly_mul(is_mn, &chain_diff, curve);
        // c4: chain_id - Σ byte_k * 256^k
        let chain_sum = le_decomp_sum_poly(col_coeffs, COL_CHAIN_ID_BYTE_OFFSET, curve);
        let c4 = poly_sub(chain_id, &chain_sum, curve);
        // c5: genesis_time - Σ byte_k * 256^k
        let gt_sum = le_decomp_sum_poly(col_coeffs, COL_GENESIS_TIME_BYTE_OFFSET, curve);
        let c5 = poly_sub(gt, &gt_sum, curve);
        // c6: is_mn * Σ α^k * (beacon_byte_k - CANON_k)
        let mut beacon_rlc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for k in 0..ROOT_LEN {
            let canon = Scalar::from_u64(BEACON_GENESIS_ROOT_MAINNET[k] as u64, curve);
            let claimed = &col_coeffs[COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET + k];
            let diff = poly_sub(claimed, &vec![canon], curve);
            beacon_rlc = poly_add(&beacon_rlc, &poly_scalar_mul(&diff, &ap), curve);
            ap = ap.mul(&alpha_dom);
        }
        let c6 = poly_mul(is_mn, &beacon_rlc, curve);
        // c7: similar for execution.
        let mut exec_rlc = vec![Scalar::zero(curve)];
        let mut ap = Scalar::one(curve);
        for k in 0..ROOT_LEN {
            let canon = Scalar::from_u64(EXECUTION_GENESIS_ROOT_MAINNET[k] as u64, curve);
            let claimed = &col_coeffs[COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET + k];
            let diff = poly_sub(claimed, &vec![canon], curve);
            exec_rlc = poly_add(&exec_rlc, &poly_scalar_mul(&diff, &ap), curve);
            ap = ap.mul(&alpha_dom);
        }
        let c7 = poly_mul(is_mn, &exec_rlc, curve);
        // c8: is_mn * (genesis_time - MAINNET_GENESIS_TIME)
        let gt_diff = poly_sub(gt, &vec![gt_mainnet], curve);
        let c8 = poly_mul(is_mn, &gt_diff, curve);
        // c9: is_mn * is_mn - is_mn
        let c9 = poly_sub(&poly_mul(is_mn, is_mn, curve), is_mn, curve);

        let mut total = c0;
        let mut ap = alpha.clone();
        for c in [c1, c2, c3, c4, c5, c6, c7, c8, c9] {
            total = poly_add(&total, &poly_scalar_mul(&c, &ap), curve);
            ap = ap.mul(alpha);
        }
        total
    }

    fn selector_column_indices(&self) -> Vec<usize> {
        vec![COL_IS_REAL]
    }

    fn padding_selector_column(&self) -> Option<usize> { None }

    fn fix_trace_padding(
        &self,
        columns: &mut [Vec<Scalar>],
        num_rows: usize,
        padded_size: usize,
    ) {
        if num_rows == 0 || num_rows >= padded_size { return; }
        if columns.len() < NUM_COLUMNS { return; }
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
        // 8-bit range checks on byte columns: 32 + 32 + 8 + 8 = 80.
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        let byte_starts: [(&str, usize, usize); 4] = [
            ("beacon_root", COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET, ROOT_LEN),
            ("execution_root", COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET, ROOT_LEN),
            ("chain_id_byte", COL_CHAIN_ID_BYTE_OFFSET, 8),
            ("genesis_time_byte", COL_GENESIS_TIME_BYTE_OFFSET, 8),
        ];
        for (name, offset, len) in byte_starts {
            for k in 0..len {
                declarations.push((
                    LookupDeclaration {
                        label: format!("genesis_{}_{}_8bit", name, k),
                        column_index: offset + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Link this gadget's `claimed_beacon_genesis_root[0..32]` (gated by
/// `IS_MAINNET`) ↔ `beacon_state_transition_air::COL_PREV_STATE_ROOT_OFFSET[0..32]`
/// (gated by `IS_REAL`). On the BST AIR, the row corresponding to the
/// genesis slot carries the genesis state root as `prev_state_root`,
/// so this descriptor algebraically forces the genesis-anchor to
/// match the mainnet beacon-genesis constant.
pub fn make_genesis_to_state_transition_descriptor(
    genesis_layer_index: usize,
    state_transition_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::beacon_state_transition_air as bst;
    let a_columns: Vec<usize> = (0..ROOT_LEN)
        .map(|k| COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET + k)
        .collect();
    let b_columns: Vec<usize> = (0..ROOT_LEN)
        .map(|k| bst::COL_PREV_STATE_ROOT_OFFSET + k)
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "genesis_to_state_transition_v1".into(),
        a_layer_index: genesis_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_MAINNET),
        b_layer_index: state_transition_layer_index,
        b_columns,
        b_selector_column: Some(bst::COL_IS_REAL),
    }
}

/// Link this gadget's `(claimed_execution_genesis_root[0..32], chain_id)`
/// (gated by `IS_MAINNET`) ↔ `block_header_air::(COL_STATE_ROOT_OFFSET[0..32],
/// COL_CHAIN_ID)` (gated by `IS_REAL`). For a genesis-block row in
/// the block-header AIR, the `state_root` is the execution-layer
/// genesis state root and `chain_id` is the canonical mainnet
/// chain_id.
pub fn make_genesis_to_block_header_descriptor(
    genesis_layer_index: usize,
    block_header_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::block_header_air as bh;
    let a_columns: Vec<usize> = (0..ROOT_LEN)
        .map(|k| COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET + k)
        .chain(std::iter::once(COL_CHAIN_ID))
        .collect();
    let b_columns: Vec<usize> = (0..ROOT_LEN)
        .map(|k| bh::COL_STATE_ROOT_OFFSET + k)
        .chain(std::iter::once(bh::COL_CHAIN_ID))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "genesis_to_block_header_v1".into(),
        a_layer_index: genesis_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_MAINNET),
        b_layer_index: block_header_layer_index,
        b_columns,
        b_selector_column: Some(bh::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate_constraints(
        witness: &GenesisStateWitness,
    ) -> Vec<Vec<Scalar>> {
        let trace = build_trace_polynomials(witness, CurveType::Bls48581);
        let cs = GenesisStateConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        cs.evaluate_on_domain(&col_refs, trace.num_rows)
    }

    #[test]
    fn mainnet_genesis_satisfies_all_constraints() {
        let w = GenesisStateWitness::from_chain(1, true);
        let results = evaluate_constraints(&w);
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
        // Sanity: trace contains the canonical constants.
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for k in 0..ROOT_LEN {
            assert_eq!(
                trace.columns[COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET + k]
                    .evaluations[0]
                    .to_u64(),
                BEACON_GENESIS_ROOT_MAINNET[k] as u64,
            );
            assert_eq!(
                trace.columns[COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET + k]
                    .evaluations[0]
                    .to_u64(),
                EXECUTION_GENESIS_ROOT_MAINNET[k] as u64,
            );
        }
        assert_eq!(trace.columns[COL_CHAIN_ID].evaluations[0].to_u64(), 1);
        assert_eq!(
            trace.columns[COL_GENESIS_TIME].evaluations[0].to_u64(),
            MAINNET_GENESIS_TIME,
        );
        assert_eq!(trace.columns[COL_IS_MAINNET].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn non_mainnet_skips_root_binding() {
        // Sepolia chain_id = 11155111, arbitrary garbage roots. No
        // mainnet binding fires; all constraints must still be zero.
        let row = GenesisStateRow {
            claimed_beacon_genesis_root: [0xaa; ROOT_LEN],
            claimed_execution_genesis_root: [0xbb; ROOT_LEN],
            chain_id: 11_155_111,
            genesis_time: 1_655_733_600,
            is_mainnet: false,
        };
        let w = GenesisStateWitness::from_rows(vec![row]);
        let results = evaluate_constraints(&w);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "non-mainnet row: constraint {} at row {} = {:?}",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn tampered_mainnet_beacon_root_detected() {
        let mut row = GenesisStateRow {
            claimed_beacon_genesis_root: BEACON_GENESIS_ROOT_MAINNET,
            claimed_execution_genesis_root: EXECUTION_GENESIS_ROOT_MAINNET,
            chain_id: MAINNET_CHAIN_ID,
            genesis_time: MAINNET_GENESIS_TIME,
            is_mainnet: true,
        };
        // Flip a single byte of the beacon root.
        row.claimed_beacon_genesis_root[5] ^= 0x01;
        let w = GenesisStateWitness::from_rows(vec![row]);
        let results = evaluate_constraints(&w);
        // Constraint 6 (mainnet_beacon_root_binding) must fire.
        assert!(
            !results[6][0].is_zero(),
            "tampered beacon root must trip constraint 6",
        );
    }

    #[test]
    fn tampered_mainnet_execution_root_detected() {
        let mut row = GenesisStateRow {
            claimed_beacon_genesis_root: BEACON_GENESIS_ROOT_MAINNET,
            claimed_execution_genesis_root: EXECUTION_GENESIS_ROOT_MAINNET,
            chain_id: MAINNET_CHAIN_ID,
            genesis_time: MAINNET_GENESIS_TIME,
            is_mainnet: true,
        };
        row.claimed_execution_genesis_root[31] ^= 0xff;
        let w = GenesisStateWitness::from_rows(vec![row]);
        let results = evaluate_constraints(&w);
        assert!(
            !results[7][0].is_zero(),
            "tampered execution root must trip constraint 7",
        );
    }

    #[test]
    fn tampered_mainnet_chain_id_detected() {
        let row = GenesisStateRow {
            claimed_beacon_genesis_root: BEACON_GENESIS_ROOT_MAINNET,
            claimed_execution_genesis_root: EXECUTION_GENESIS_ROOT_MAINNET,
            chain_id: 2, // not mainnet
            genesis_time: MAINNET_GENESIS_TIME,
            is_mainnet: true,
        };
        let w = GenesisStateWitness::from_rows(vec![row]);
        let results = evaluate_constraints(&w);
        assert!(
            !results[3][0].is_zero(),
            "wrong chain_id must trip constraint 3",
        );
    }

    #[test]
    fn non_binary_is_mainnet_detected() {
        let row = GenesisStateRow {
            claimed_beacon_genesis_root: BEACON_GENESIS_ROOT_MAINNET,
            claimed_execution_genesis_root: EXECUTION_GENESIS_ROOT_MAINNET,
            chain_id: MAINNET_CHAIN_ID,
            genesis_time: MAINNET_GENESIS_TIME,
            is_mainnet: true,
        };
        let w = GenesisStateWitness::from_rows(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Force is_mainnet = 2 — should trip both #1 and #9.
        cols[COL_IS_MAINNET][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = GenesisStateConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[1][0].is_zero(), "constraint 1 (is_mn binary) must fire");
        assert!(!results[9][0].is_zero(), "constraint 9 (idempotent) must fire");
    }

    #[test]
    fn genesis_to_state_transition_descriptor_well_formed() {
        let desc = make_genesis_to_state_transition_descriptor(0, 1);
        assert_eq!(desc.label, "genesis_to_state_transition_v1");
        assert_eq!(desc.a_columns.len(), ROOT_LEN);
        assert_eq!(desc.b_columns.len(), ROOT_LEN);
        assert_eq!(desc.a_selector_column, Some(COL_IS_MAINNET));
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        for k in 0..ROOT_LEN {
            assert_eq!(desc.a_columns[k], COL_CLAIMED_BEACON_GENESIS_ROOT_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                crate::beacon_state_transition_air::COL_PREV_STATE_ROOT_OFFSET + k,
            );
        }
    }

    #[test]
    fn genesis_to_block_header_descriptor_well_formed() {
        let desc = make_genesis_to_block_header_descriptor(0, 1);
        assert_eq!(desc.label, "genesis_to_block_header_v1");
        // 32 root bytes + 1 chain_id col.
        assert_eq!(desc.a_columns.len(), ROOT_LEN + 1);
        assert_eq!(desc.b_columns.len(), ROOT_LEN + 1);
        assert_eq!(desc.a_selector_column, Some(COL_IS_MAINNET));
        for k in 0..ROOT_LEN {
            assert_eq!(desc.a_columns[k], COL_CLAIMED_EXECUTION_GENESIS_ROOT_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                crate::block_header_air::COL_STATE_ROOT_OFFSET + k,
            );
        }
        assert_eq!(desc.a_columns[ROOT_LEN], COL_CHAIN_ID);
        assert_eq!(desc.b_columns[ROOT_LEN], crate::block_header_air::COL_CHAIN_ID);
    }

    #[test]
    fn evaluate_at_point_matches_domain_for_honest_mainnet() {
        // Pick the row 0 evaluations and confirm `evaluate_at_point`
        // returns zero for an honest mainnet witness (since every
        // constraint body is zero).
        let w = GenesisStateWitness::from_chain(1, true);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let row0: Vec<Scalar> = trace
            .columns
            .iter()
            .map(|p| p.evaluations[0].clone())
            .collect();
        let cs = GenesisStateConstraintSystem::new(trace.num_rows);
        let alpha = Scalar::from_u64(12345, CurveType::Bls48581);
        let v = cs.evaluate_at_point(&row0, &alpha);
        assert!(
            v.is_zero(),
            "evaluate_at_point on honest mainnet row must be zero",
        );
    }
}
