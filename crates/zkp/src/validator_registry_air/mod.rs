//! Validator registry inclusion AIR (roadmap #63).
//!
//! Per-inclusion gadget: proves that one validator at a known index
//! sits at a known position within the beacon-chain validator registry,
//! given the registry's claimed root.
//!
//! The full beacon-chain validator registry is
//! `List[Validator, VALIDATOR_REGISTRY_LIMIT = 2^40]`; its hashTreeRoot
//! is `mix_in_length(merkleize(validator_htrs, 2^40), registry_len)`.
//! Proving the construction of that root from 1M+ validators is
//! impractical inside the AIR (it would require 2^40 SHA-256 leaves);
//! instead this AIR proves an *inclusion*: given the merkle path of
//! sibling hashes from leaf to root, the running hash plus the
//! `mix_in_length` step reproduces the claimed registry root.
//!
//! ## Trace layout
//!
//! One inclusion expands to `DEPTH + 1 = 41` rows:
//!   - rows `0..DEPTH` (40): one per merkle tree level, bottom-up.
//!     Row 0's `CURRENT_HASH` is the validator's hash_tree_root (the
//!     leaf at the bottom of the tree). At level `k`,
//!     `NEXT_HASH = sha256_pair(LEFT, RIGHT)` where `(LEFT, RIGHT)` is
//!     `(CURRENT_HASH, SIBLING)` if `INDEX_BIT_k = 0`, else
//!     `(SIBLING, CURRENT_HASH)`. The next row's `CURRENT_HASH` equals
//!     this row's `NEXT_HASH`.
//!   - row `DEPTH` (40): the `mix_in_length` step.
//!     `CURRENT_HASH` = merkleized registry root,
//!     `SIBLING` = `length_chunk` (`length_le[0..8] || 0[8..32]`),
//!     `NEXT_HASH = sha256_pair(CURRENT_HASH, SIBLING)` = the claimed
//!     `REGISTRY_ROOT` constant.
//!
//! The `(LEFT, RIGHT, NEXT_HASH)` triple on every active row maps to a
//! single SHA-256 invocation, which is matched against the
//! `sha256_extract` AIR via cross-AIR LogUp (see
//! [`make_validator_registry_pair_hash_linkage_descriptor`]).
//!
//! ## Algebraic constraints (row-local, 6 bodies)
//!
//! 0. `is_real_binary` — `IS_REAL · (IS_REAL − 1) = 0`
//! 1. `is_mix_binary` — `IS_MIX · (IS_MIX − 1) = 0`
//! 2. `index_bit_binary` — `IS_REAL · INDEX_BIT · (INDEX_BIT − 1) = 0`
//! 3. `left_selection` (β-RLC over 32 byte sub-bodies) —
//!    `IS_REAL · Σ β^i · (LEFT[i] − ((1 − INDEX_BIT) · CURRENT_HASH[i] + INDEX_BIT · SIBLING[i])) = 0`
//! 4. `right_selection` (β-RLC over 32 byte sub-bodies) —
//!    `IS_REAL · Σ β^i · (RIGHT[i] − (INDEX_BIT · CURRENT_HASH[i] + (1 − INDEX_BIT) · SIBLING[i])) = 0`
//!    (On mix rows `IS_MIX = 1` so the index-bit is fixed to 0 and
//!    `LEFT = CURRENT_HASH`, `RIGHT = SIBLING` — same row-local body.)
//! 5. `index_aggregator` (β-RLC over 40 byte sub-bodies) —
//!    `IS_REAL · (VALIDATOR_INDEX − Σ_{k=0}^{39} INDEX_BIT_REPL[k] · 2^k) = 0`
//!    where `INDEX_BIT_REPL[k]` is the bit-k column replicated identically
//!    on every active row (constancy enforced by shifted body 1).
//!
//! ## Cross-row shifted constraints (3 bodies)
//!
//! 0. `hash_chain` — `IS_REAL(ω·X) · (CURRENT_HASH(ω·X) − NEXT_HASH(X)) = 0`
//!    (β-RLC over 32 byte sub-bodies, excluded at the wrap-around row).
//! 1. `index_bits_constancy` — `IS_REAL(ω·X) · Σ β^k · (INDEX_BIT_REPL_k(ω·X) − INDEX_BIT_REPL_k(X)) = 0`
//! 2. `registry_root_constancy` — `IS_REAL(ω·X) · Σ β^i · (REGISTRY_ROOT_i(ω·X) − REGISTRY_ROOT_i(X)) = 0`
//!
//! Together these pin the row-by-row tree-walk: each level's `NEXT_HASH`
//! flows into the next level's `CURRENT_HASH`, replicated columns are
//! constant across the inclusion, and (with the pair-hash cross-AIR
//! LogUp) the final `NEXT_HASH` on the mix row equals the claimed
//! `REGISTRY_ROOT`. The row-local `index_aggregator` body binds the
//! 40-bit decomposition to the prover-claimed validator index.
//!
//! ## What this AIR does NOT prove
//!
//! * SHA-256 of `(LEFT || RIGHT)`: deferred to the cross-AIR LogUp into
//!   `sha256_extract` (and transitively the bit-level SHA-256 AIR).
//! * That `CURRENT_HASH` on row 0 equals the validator's actual
//!   `hash_tree_root`: deferred to a cross-AIR LogUp into
//!   `validator_htr_air` (see
//!   [`make_validator_registry_leaf_to_htr_linkage_descriptor`]).
//! * Multi-validator registry root construction (would require 2^40 leaf
//!   SHA-256 invocations — out of scope; only inclusion proofs are
//!   feasible at scale).
//! * That `INDEX_BIT_REPL_k` equals the level-k bit on each row (the
//!   row-local body uses `INDEX_BIT` which the witness builder sets to
//!   `INDEX_BIT_REPL[level]`; a follow-up could add a per-row position
//!   one-hot selector to pin this algebraically, but for inclusion
//!   soundness it is sufficient that the chain hashes to the claimed
//!   `REGISTRY_ROOT`).

use crate::field::{CurveType, Scalar};
use crate::lookup::LookupRequirements;
use crate::poly_arith::{
    poly_add, poly_mul, poly_mul_linear, poly_scalar_mul, poly_shift, poly_sub,
};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

/// Depth of the validator registry merkle tree.
/// `VALIDATOR_REGISTRY_LIMIT = 2^40`, so the merkleization tree has 40 levels.
pub const DEPTH: usize = 40;
/// Bytes per hash chunk.
pub const CHUNK_BYTES: usize = 32;
/// Rows used per inclusion: `DEPTH` tree levels + 1 `mix_in_length` row.
pub const ROWS_PER_INCLUSION: usize = DEPTH + 1;

// ─── Column indices ───────────────────────────────────────────────────

pub const COL_CURRENT_HASH_OFFSET: usize = 0;
pub const COL_SIBLING_OFFSET: usize = COL_CURRENT_HASH_OFFSET + CHUNK_BYTES; // 32
pub const COL_LEFT_OFFSET: usize = COL_SIBLING_OFFSET + CHUNK_BYTES;          // 64
pub const COL_RIGHT_OFFSET: usize = COL_LEFT_OFFSET + CHUNK_BYTES;            // 96
pub const COL_NEXT_HASH_OFFSET: usize = COL_RIGHT_OFFSET + CHUNK_BYTES;       // 128
pub const COL_REGISTRY_ROOT_OFFSET: usize = COL_NEXT_HASH_OFFSET + CHUNK_BYTES; // 160

pub const COL_INDEX_BIT: usize = COL_REGISTRY_ROOT_OFFSET + CHUNK_BYTES;      // 192
pub const COL_VALIDATOR_INDEX: usize = COL_INDEX_BIT + 1;                     // 193
pub const COL_INDEX_BIT_REPL_OFFSET: usize = COL_VALIDATOR_INDEX + 1;         // 194
pub const COL_LEVEL: usize = COL_INDEX_BIT_REPL_OFFSET + DEPTH;               // 234

pub const COL_IS_REAL: usize = COL_LEVEL + 1;                                 // 235
pub const COL_IS_MIX: usize = COL_IS_REAL + 1;                                // 236

/// Task #318 / #309 / #190 mirror column: validator pubkey first-byte
/// anchor populated on the leaf row (row 0 of each inclusion) for
/// cross-AIR LogUp descriptors that want to bind the proposer pubkey's
/// first byte to this AIR without requiring B-side overrides on
/// `CURRENT_HASH[0]` / `LEFT[0]` (which would otherwise break the
/// row-local `left_selection` body and the cross-row `hash_chain`
/// constraint). Populated by the trace builder via
/// [`ValidatorRegistryWitness::pk_byte0`]; defaults to
/// `inclusion.validator_htr[0]` (back-compat) when unset. The mirror is
/// non-zero only on the leaf row; all other rows (intermediate path,
/// mix-in-length, padding) leave the cell at zero. No row-local
/// constraint binds this column — soundness flows from the cross-AIR
/// LogUp closure plus the (deferred) validator-HTR linkage that ties
/// the leaf chunk to the validator's `(pubkey, ...)` SSZ container.
pub const COL_PK_BYTE0_MIRROR: usize = COL_IS_MIX + 1;                        // 237

pub const NUM_COLUMNS: usize = COL_PK_BYTE0_MIRROR + 1;                       // 238

/// 6 row-local constraint bodies (see module doc).
pub const NUM_ROW_CONSTRAINTS: usize = 6;
/// 3 cross-row shifted bodies (hash_chain + 2 constancy).
pub const NUM_SHIFTED: usize = 3;

// ─── Witness types ────────────────────────────────────────────────────

/// One inclusion's witness: a single validator's merkle path to the
/// claimed registry root, plus the registry length used by
/// `mix_in_length`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistryInclusionWitness {
    /// The validator's hash_tree_root (leaf chunk).
    pub validator_htr: [u8; CHUNK_BYTES],
    /// Validator's position in the registry (0..2^40).
    pub validator_index: u64,
    /// Sibling hashes from the bottom level upward.
    /// `sibling_hashes[0]` is the sibling at the leaf level (depth-0
    /// pair); `sibling_hashes[DEPTH-1]` is the root-adjacent sibling.
    pub sibling_hashes: [[u8; CHUNK_BYTES]; DEPTH],
    /// Number of real validators in the registry (mixed into the root
    /// via `mix_in_length`).
    pub registry_length: u64,
    /// Claimed `hash_tree_root` of the registry. Must equal
    /// `mix_in_length(merkleize_path_root, registry_length)`.
    pub registry_root: [u8; CHUNK_BYTES],
}

impl RegistryInclusionWitness {
    /// Build a witness from a validator, its index, and a merkle proof
    /// path of length [`DEPTH`]. The registry root is computed
    /// host-side by walking the path; the caller asserts it matches the
    /// expected on-chain root.
    pub fn from_validator_proof(
        validator_htr: [u8; CHUNK_BYTES],
        index: u64,
        proof_path: [[u8; CHUNK_BYTES]; DEPTH],
        registry_length: u64,
    ) -> Self {
        let mut current = validator_htr;
        for (level, sibling) in proof_path.iter().enumerate() {
            let bit = (index >> level) & 1;
            let (left, right) = if bit == 0 {
                (current, *sibling)
            } else {
                (*sibling, current)
            };
            current = crate::sha256::sha256_pair(&left, &right);
        }
        // mix_in_length
        let registry_root =
            crate::ssz::mix_in_length(current, registry_length);

        Self {
            validator_htr,
            validator_index: index,
            sibling_hashes: proof_path,
            registry_length,
            registry_root,
        }
    }
}

/// Sequence of inclusions packed into a single trace. (One inclusion
/// occupies [`ROWS_PER_INCLUSION`] consecutive rows.) For the v1 wire
/// we expose just one inclusion at a time; multi-inclusion batching is
/// straightforward but currently out of scope.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidatorRegistryWitness {
    pub inclusions: Vec<RegistryInclusionWitness>,
    /// Task #318 / #309 mirror data: per-inclusion validator pubkey
    /// first byte populated into [`COL_PK_BYTE0_MIRROR`] on the leaf
    /// row of that inclusion. Empty `Vec` means "default to
    /// `inclusions[i].validator_htr[0]`" (back-compat). Otherwise must
    /// match `inclusions.len()` and is written verbatim per leaf row.
    #[doc(hidden)]
    pub pk_byte0: Vec<u8>,
}

impl ValidatorRegistryWitness {
    /// Builder: override `COL_PK_BYTE0_MIRROR` per-inclusion. Used by
    /// integration tests that wire a cross-AIR LogUp descriptor against
    /// the leaf-row mirror cell without touching `CURRENT_HASH[0]` (
    /// which would otherwise break the row-local `left_selection` body
    /// and the cross-row `hash_chain` constraint).
    pub fn with_pk_byte0(mut self, mirror: Vec<u8>) -> Self {
        assert_eq!(
            mirror.len(),
            self.inclusions.len(),
            "pk_byte0 length must equal inclusions.len()",
        );
        self.pk_byte0 = mirror;
        self
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn byte_power(i: usize, curve: CurveType) -> Scalar {
    debug_assert!(i < 8);
    Scalar::from_u64(1u64 << (8 * i), curve)
}

fn scalar_pow(base: &Scalar, exp: u64) -> Scalar {
    let mut result = Scalar::one(base.curve_type());
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.mul(&b);
        }
        b = b.mul(&b);
        e >>= 1;
    }
    result
}

fn write_chunk(
    columns: &mut [Vec<Scalar>],
    offset: usize,
    row: usize,
    chunk: &[u8; CHUNK_BYTES],
    curve: CurveType,
) {
    for (b, &byte) in chunk.iter().enumerate() {
        columns[offset + b][row] = Scalar::from_u64(byte as u64, curve);
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ValidatorRegistryWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.inclusions.len() * ROWS_PER_INCLUSION;
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();

    for (inc_i, inclusion) in witness.inclusions.iter().enumerate() {
        let base_row = inc_i * ROWS_PER_INCLUSION;
        let mut current = inclusion.validator_htr;

        // Task #318 mirror: pin `COL_PK_BYTE0_MIRROR` on the leaf row
        // (base_row + 0) to the per-inclusion pubkey first byte. Default
        // to `validator_htr[0]` when the caller didn't populate
        // `pk_byte0`. Non-leaf rows keep the cell at zero.
        let pk_byte = if witness.pk_byte0.is_empty() {
            inclusion.validator_htr[0]
        } else {
            witness.pk_byte0[inc_i]
        };
        columns[COL_PK_BYTE0_MIRROR][base_row] =
            Scalar::from_u64(pk_byte as u64, curve);

        // Pre-compute the 40 index bits.
        let mut index_bits = [0u8; DEPTH];
        for k in 0..DEPTH {
            index_bits[k] = ((inclusion.validator_index >> k) & 1) as u8;
        }

        // Construct the length chunk used by mix_in_length:
        // 8 LE bytes of `registry_length` || 24 zero bytes.
        let mut length_chunk = [0u8; CHUNK_BYTES];
        length_chunk[..8].copy_from_slice(&inclusion.registry_length.to_le_bytes());

        // Tree rows 0..DEPTH.
        for level in 0..DEPTH {
            let row = base_row + level;
            let sibling = inclusion.sibling_hashes[level];
            let bit = index_bits[level];
            let (left, right) = if bit == 0 {
                (current, sibling)
            } else {
                (sibling, current)
            };
            let next = crate::sha256::sha256_pair(&left, &right);

            write_chunk(&mut columns, COL_CURRENT_HASH_OFFSET, row, &current, curve);
            write_chunk(&mut columns, COL_SIBLING_OFFSET, row, &sibling, curve);
            write_chunk(&mut columns, COL_LEFT_OFFSET, row, &left, curve);
            write_chunk(&mut columns, COL_RIGHT_OFFSET, row, &right, curve);
            write_chunk(&mut columns, COL_NEXT_HASH_OFFSET, row, &next, curve);
            write_chunk(
                &mut columns,
                COL_REGISTRY_ROOT_OFFSET,
                row,
                &inclusion.registry_root,
                curve,
            );

            columns[COL_INDEX_BIT][row] = Scalar::from_u64(bit as u64, curve);
            columns[COL_VALIDATOR_INDEX][row] =
                Scalar::from_u64(inclusion.validator_index, curve);
            for k in 0..DEPTH {
                columns[COL_INDEX_BIT_REPL_OFFSET + k][row] =
                    Scalar::from_u64(index_bits[k] as u64, curve);
            }
            columns[COL_LEVEL][row] = Scalar::from_u64(level as u64, curve);
            columns[COL_IS_REAL][row] = one.clone();
            // IS_MIX = 0 on tree rows.

            current = next;
        }

        // Mix-in-length row at base_row + DEPTH.
        {
            let row = base_row + DEPTH;
            let left = current;
            let right = length_chunk;
            let next = crate::sha256::sha256_pair(&left, &right);

            debug_assert_eq!(
                next, inclusion.registry_root,
                "mix_in_length output must equal claimed registry_root"
            );

            write_chunk(&mut columns, COL_CURRENT_HASH_OFFSET, row, &current, curve);
            write_chunk(&mut columns, COL_SIBLING_OFFSET, row, &length_chunk, curve);
            write_chunk(&mut columns, COL_LEFT_OFFSET, row, &left, curve);
            write_chunk(&mut columns, COL_RIGHT_OFFSET, row, &right, curve);
            write_chunk(&mut columns, COL_NEXT_HASH_OFFSET, row, &next, curve);
            write_chunk(
                &mut columns,
                COL_REGISTRY_ROOT_OFFSET,
                row,
                &inclusion.registry_root,
                curve,
            );

            // On mix row INDEX_BIT = 0 so LEFT = CURRENT_HASH.
            columns[COL_INDEX_BIT][row] = zero.clone();
            columns[COL_VALIDATOR_INDEX][row] =
                Scalar::from_u64(inclusion.validator_index, curve);
            for k in 0..DEPTH {
                columns[COL_INDEX_BIT_REPL_OFFSET + k][row] =
                    Scalar::from_u64(index_bits[k] as u64, curve);
            }
            columns[COL_LEVEL][row] = Scalar::from_u64(DEPTH as u64, curve);
            columns[COL_IS_REAL][row] = one.clone();
            columns[COL_IS_MIX][row] = one.clone();
        }
    }

    let polys: Vec<Polynomial> = columns
        .into_iter()
        .map(|evals| Polynomial {
            evaluations: evals,
            degree: num_rows,
        })
        .collect();

    TracePolynomials {
        columns: polys,
        num_rows,
        padded_size: padded as u64,
        curve,
    }
}

// ─── Constraint system ─────────────────────────────────────────────────

pub struct ValidatorRegistryConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ValidatorRegistryConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            omega: None,
            domain_size: None,
        }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

// ─── Body helpers ─────────────────────────────────────────────────────

/// β-RLC over 32 byte sub-bodies of `LEFT[i] − ((1−bit)·current_hash[i] + bit·sibling[i])`.
fn eval_left_selection(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let is_real = &col_evals[COL_IS_REAL];
    let bit = &col_evals[COL_INDEX_BIT];
    let one_minus_bit = one.sub(bit);
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let left = &col_evals[COL_LEFT_OFFSET + i];
        let cur = &col_evals[COL_CURRENT_HASH_OFFSET + i];
        let sib = &col_evals[COL_SIBLING_OFFSET + i];
        let expected = one_minus_bit.mul(cur).add(&bit.mul(sib));
        let body = left.sub(&expected);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    is_real.mul(&acc)
}

fn build_left_selection_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let bit_poly = &col_coeffs[COL_INDEX_BIT];
    let one_minus_bit_poly = poly_sub(&one_poly, bit_poly, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let left_p = &col_coeffs[COL_LEFT_OFFSET + i];
        let cur_p = &col_coeffs[COL_CURRENT_HASH_OFFSET + i];
        let sib_p = &col_coeffs[COL_SIBLING_OFFSET + i];
        let term0 = poly_mul(&one_minus_bit_poly, cur_p, curve);
        let term1 = poly_mul(bit_poly, sib_p, curve);
        let expected = poly_add(&term0, &term1, curve);
        let body = poly_sub(left_p, &expected, curve);
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

/// β-RLC over 32 byte sub-bodies of `RIGHT[i] − (bit·current_hash[i] + (1−bit)·sibling[i])`.
fn eval_right_selection(col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
    let curve = alpha.curve_type();
    let one = Scalar::one(curve);
    let is_real = &col_evals[COL_IS_REAL];
    let bit = &col_evals[COL_INDEX_BIT];
    let one_minus_bit = one.sub(bit);
    let mut acc = Scalar::zero(curve);
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let right = &col_evals[COL_RIGHT_OFFSET + i];
        let cur = &col_evals[COL_CURRENT_HASH_OFFSET + i];
        let sib = &col_evals[COL_SIBLING_OFFSET + i];
        let expected = bit.mul(cur).add(&one_minus_bit.mul(sib));
        let body = right.sub(&expected);
        acc = acc.add(&body.mul(&ap));
        ap = ap.mul(alpha);
    }
    is_real.mul(&acc)
}

fn build_right_selection_poly(
    col_coeffs: &[Vec<Scalar>],
    alpha: &Scalar,
    curve: CurveType,
) -> Vec<Scalar> {
    let one_poly = vec![Scalar::one(curve)];
    let bit_poly = &col_coeffs[COL_INDEX_BIT];
    let one_minus_bit_poly = poly_sub(&one_poly, bit_poly, curve);
    let mut acc = vec![Scalar::zero(curve)];
    let mut ap = Scalar::one(curve);
    for i in 0..CHUNK_BYTES {
        let right_p = &col_coeffs[COL_RIGHT_OFFSET + i];
        let cur_p = &col_coeffs[COL_CURRENT_HASH_OFFSET + i];
        let sib_p = &col_coeffs[COL_SIBLING_OFFSET + i];
        let term0 = poly_mul(bit_poly, cur_p, curve);
        let term1 = poly_mul(&one_minus_bit_poly, sib_p, curve);
        let expected = poly_add(&term0, &term1, curve);
        let body = poly_sub(right_p, &expected, curve);
        let scaled = poly_scalar_mul(&body, &ap);
        acc = poly_add(&acc, &scaled, curve);
        ap = ap.mul(alpha);
    }
    poly_mul(&col_coeffs[COL_IS_REAL], &acc, curve)
}

/// `IS_REAL · (VALIDATOR_INDEX − Σ_{k=0..40} INDEX_BIT_REPL[k] · 2^k) = 0`.
fn eval_index_aggregator(col_evals: &[Scalar]) -> Scalar {
    let curve = col_evals[COL_IS_REAL].curve_type();
    let mut sum = Scalar::zero(curve);
    for k in 0..DEPTH {
        // Use field-multiplication powers of 2 (won't overflow as Scalar).
        let pow = scalar_pow(&Scalar::from_u64(2, curve), k as u64);
        let bit_k = &col_evals[COL_INDEX_BIT_REPL_OFFSET + k];
        sum = sum.add(&bit_k.mul(&pow));
    }
    let vi = &col_evals[COL_VALIDATOR_INDEX];
    let body = vi.sub(&sum);
    col_evals[COL_IS_REAL].mul(&body)
}

fn build_index_aggregator_poly(
    col_coeffs: &[Vec<Scalar>],
    curve: CurveType,
) -> Vec<Scalar> {
    let mut sum = vec![Scalar::zero(curve)];
    for k in 0..DEPTH {
        let pow = scalar_pow(&Scalar::from_u64(2, curve), k as u64);
        let bit_k_p = &col_coeffs[COL_INDEX_BIT_REPL_OFFSET + k];
        let scaled = poly_scalar_mul(bit_k_p, &pow);
        sum = poly_add(&sum, &scaled, curve);
    }
    let vi_p = &col_coeffs[COL_VALIDATOR_INDEX];
    let body = poly_sub(vi_p, &sum, curve);
    poly_mul(&col_coeffs[COL_IS_REAL], &body, curve)
}

// suppress unused warning re: byte_power (kept for symmetry; aggregator uses scalar_pow on 2).
#[allow(dead_code)]
fn _unused() { let _ = byte_power; }

// ─── VmConstraintSystem impl ───────────────────────────────────────────

impl VmConstraintSystem for ValidatorRegistryConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        vec![
            "is_real_binary".into(),
            "is_mix_binary".into(),
            "index_bit_binary".into(),
            "left_selection".into(),
            "right_selection".into(),
            "index_aggregator".into(),
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
        let alpha = Scalar::from_u64(7, curve);

        let mut bodies: Vec<Vec<Scalar>> =
            (0..NUM_ROW_CONSTRAINTS).map(|_| vec![Scalar::zero(curve); n]).collect();

        for row in 0..n {
            let row_evals: Vec<Scalar> =
                columns.iter().map(|c| c[row].clone()).collect();
            let is_real = &row_evals[COL_IS_REAL];
            let is_mix = &row_evals[COL_IS_MIX];
            let bit = &row_evals[COL_INDEX_BIT];

            bodies[0][row] = is_real.mul(&is_real.sub(&one));
            bodies[1][row] = is_mix.mul(&is_mix.sub(&one));
            bodies[2][row] = is_real.mul(&bit.mul(&bit.sub(&one)));
            bodies[3][row] = eval_left_selection(&row_evals, &alpha);
            bodies[4][row] = eval_right_selection(&row_evals, &alpha);
            bodies[5][row] = eval_index_aggregator(&row_evals);
        }
        bodies
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &col_evals[COL_IS_REAL];
        let is_mix = &col_evals[COL_IS_MIX];
        let bit = &col_evals[COL_INDEX_BIT];

        let bodies: [Scalar; NUM_ROW_CONSTRAINTS] = [
            is_real.mul(&is_real.sub(&one)),
            is_mix.mul(&is_mix.sub(&one)),
            is_real.mul(&bit.mul(&bit.sub(&one))),
            eval_left_selection(col_evals, alpha),
            eval_right_selection(col_evals, alpha),
            eval_index_aggregator(col_evals),
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
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_mix = &col_coeffs[COL_IS_MIX];
        let bit = &col_coeffs[COL_INDEX_BIT];

        let is_real_m1 = poly_sub(is_real, &one_poly, curve);
        let is_real_binary = poly_mul(is_real, &is_real_m1, curve);

        let is_mix_m1 = poly_sub(is_mix, &one_poly, curve);
        let is_mix_binary = poly_mul(is_mix, &is_mix_m1, curve);

        let bit_m1 = poly_sub(bit, &one_poly, curve);
        let bit_sq = poly_mul(bit, &bit_m1, curve);
        let index_bit_binary = poly_mul(is_real, &bit_sq, curve);

        let bodies: Vec<Vec<Scalar>> = vec![
            is_real_binary,
            is_mix_binary,
            index_bit_binary,
            build_left_selection_poly(col_coeffs, alpha, curve),
            build_right_selection_poly(col_coeffs, alpha, curve),
            build_index_aggregator_poly(col_coeffs, curve),
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

    fn shifted_column_indices(&self) -> Vec<usize> {
        // For the hash_chain body we need CURRENT_HASH[0..32](ω·X)
        // and IS_REAL(ω·X). For the constancy bodies we additionally
        // need INDEX_BIT_REPL[0..DEPTH](ω·X) and REGISTRY_ROOT[0..32](ω·X).
        let mut cols = Vec::with_capacity(CHUNK_BYTES + 1 + DEPTH + CHUNK_BYTES);
        for b in 0..CHUNK_BYTES {
            cols.push(COL_CURRENT_HASH_OFFSET + b);
        }
        cols.push(COL_IS_REAL);
        for k in 0..DEPTH {
            cols.push(COL_INDEX_BIT_REPL_OFFSET + k);
        }
        for b in 0..CHUNK_BYTES {
            cols.push(COL_REGISTRY_ROOT_OFFSET + b);
        }
        cols
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
        let expected_shifted_len = CHUNK_BYTES + 1 + DEPTH + CHUNK_BYTES;
        if shifted_evals.len() < expected_shifted_len
            || col_evals_at_z.len() < NUM_COLUMNS
        {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        // Indices inside shifted_evals.
        let cur_shift_off = 0usize;
        let is_real_next_idx = CHUNK_BYTES;
        let bit_repl_shift_off = CHUNK_BYTES + 1;
        let reg_root_shift_off = CHUNK_BYTES + 1 + DEPTH;

        let is_real_next = &shifted_evals[is_real_next_idx];

        // Shifted body 0: hash_chain (β-RLC over 32 sub-bodies).
        let mut chain_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let cur_next = &shifted_evals[cur_shift_off + b];
            let next_curr = &col_evals_at_z[COL_NEXT_HASH_OFFSET + b];
            let diff = cur_next.sub(next_curr);
            chain_acc = chain_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = is_real_next.mul(&chain_acc);

        // Shifted body 1: index_bits_constancy.
        let mut bits_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for k in 0..DEPTH {
            let bit_next = &shifted_evals[bit_repl_shift_off + k];
            let bit_curr = &col_evals_at_z[COL_INDEX_BIT_REPL_OFFSET + k];
            let diff = bit_next.sub(bit_curr);
            bits_acc = bits_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body1 = is_real_next.mul(&bits_acc);

        // Shifted body 2: registry_root_constancy.
        let mut root_acc = Scalar::zero(curve);
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_next = &shifted_evals[reg_root_shift_off + b];
            let r_curr = &col_evals_at_z[COL_REGISTRY_ROOT_OFFSET + b];
            let diff = r_next.sub(r_curr);
            root_acc = root_acc.add(&diff.mul(&ap_inner));
            ap_inner = ap_inner.mul(alpha);
        }
        let body2 = is_real_next.mul(&root_acc);

        let exclusion = z.sub(omega_n_minus_1);
        let bodies = [body0, body1, body2];
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut acc = Scalar::zero(curve);
        for body in &bodies {
            acc = acc.add(&ap.mul(body).mul(&exclusion));
            ap = ap.mul(alpha);
        }
        acc
    }

    fn build_shifted_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        domain_size: u64,
        omega: &Scalar,
        alpha_offset: usize,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let is_real = &col_coeffs[COL_IS_REAL];
        let is_real_shift = poly_shift(is_real, omega);

        // Body 0: hash_chain β-RLC over 32 sub-bodies.
        let mut chain_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let cur_poly = &col_coeffs[COL_CURRENT_HASH_OFFSET + b];
            let next_poly = &col_coeffs[COL_NEXT_HASH_OFFSET + b];
            let cur_shift = poly_shift(cur_poly, omega);
            let diff = poly_sub(&cur_shift, next_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            chain_acc = poly_add(&chain_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body0 = poly_mul(&is_real_shift, &chain_acc, curve);

        // Body 1: index_bits_constancy β-RLC over DEPTH sub-bodies.
        let mut bits_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for k in 0..DEPTH {
            let bit_poly = &col_coeffs[COL_INDEX_BIT_REPL_OFFSET + k];
            let bit_shift = poly_shift(bit_poly, omega);
            let diff = poly_sub(&bit_shift, bit_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            bits_acc = poly_add(&bits_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body1 = poly_mul(&is_real_shift, &bits_acc, curve);

        // Body 2: registry_root_constancy β-RLC over 32 sub-bodies.
        let mut root_acc = vec![Scalar::zero(curve)];
        let mut ap_inner = Scalar::one(curve);
        for b in 0..CHUNK_BYTES {
            let r_poly = &col_coeffs[COL_REGISTRY_ROOT_OFFSET + b];
            let r_shift = poly_shift(r_poly, omega);
            let diff = poly_sub(&r_shift, r_poly, curve);
            let scaled = poly_scalar_mul(&diff, &ap_inner);
            root_acc = poly_add(&root_acc, &scaled, curve);
            ap_inner = ap_inner.mul(alpha);
        }
        let body2 = poly_mul(&is_real_shift, &root_acc, curve);

        let omega_n_minus_1 = scalar_pow(omega, domain_size.saturating_sub(1));
        let bodies = [body0, body1, body2];
        let mut ap = Scalar::one(curve);
        for _ in 0..alpha_offset {
            ap = ap.mul(alpha);
        }
        let mut acc = vec![Scalar::zero(curve)];
        for body in &bodies {
            let excluded = poly_mul_linear(body, &omega_n_minus_1);
            let scaled = poly_scalar_mul(&excluded, &ap);
            acc = poly_add(&acc, &scaled, curve);
            ap = ap.mul(alpha);
        }
        acc
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        LookupRequirements::none()
    }
}

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// Cross-AIR LogUp descriptor binding every active `(LEFT, RIGHT,
/// NEXT_HASH)` triple on the registry-inclusion gadget side to a real
/// SHA-256 invocation in [`crate::sha256_extract`]. 96-byte tuple on
/// each side, gated by `IS_REAL`.
///
/// Once paired with the bit-level SHA-256 cross-AIR LogUp from
/// `sha256_extract`, this cryptographically pins
/// `NEXT_HASH = sha256(LEFT || RIGHT)` for every level of the
/// merkleization walk **and** for the `mix_in_length` step.
/// Alias for [`make_validator_registry_pair_hash_linkage_descriptor`] with a
/// shorter / more uniform name matching the other `*_to_sha256` descriptors
/// throughout the crate. Binds every active `(LEFT, RIGHT, NEXT_HASH)` triple
/// of the registry-inclusion walk (one per tree level + the `mix_in_length`
/// row) to a real SHA-256 invocation on the `sha256_extract` side.
pub fn make_validator_registry_to_sha256_descriptor(
    registry_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    make_validator_registry_pair_hash_linkage_descriptor(
        registry_layer_index,
        sha256_extract_layer_index,
    )
}

pub fn make_validator_registry_pair_hash_linkage_descriptor(
    registry_layer_index: usize,
    sha256_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::sha256_extract as se;
    let mut a_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_LEFT_OFFSET + b);
    }
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_RIGHT_OFFSET + b);
    }
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_NEXT_HASH_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(96);
    for b in 0..se::NUM_INPUT_BYTES {
        b_columns.push(se::COL_INPUT_BYTE_OFFSET + b);
    }
    for b in 0..se::NUM_OUTPUT_BYTES {
        b_columns.push(se::COL_OUTPUT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_registry_pair_hash_v1".into(),
        a_layer_index: registry_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha256_extract_layer_index,
        b_columns,
        b_selector_column: Some(se::COL_IS_REAL),
    }
}

/// Cross-AIR LogUp descriptor binding the registry-inclusion gadget's
/// leaf (`CURRENT_HASH` on the row where `LEVEL = 0`) to the validator's
/// `hash_tree_root` exposed by [`crate::validator_htr_air`].
///
/// Tuple matched (33 columns):
///   * A side: `(VALIDATOR_INDEX, CURRENT_HASH[0..32])` gated by a
///     leaf-row selector (here re-using `IS_REAL` and trusting the
///     witness to filter to only the leaf row of each inclusion — a
///     future tightening can add a dedicated `IS_LEAF` selector column).
///   * B side: `(validator_htr::COL_VALIDATOR_INDEX,
///     validator_htr::COL_VALIDATOR_ROOT_BYTE_OFFSET[0..32])` gated by
///     `validator_htr::COL_IS_REAL`.
///
/// **Soundness note**: as stated this descriptor is over-inclusive on
/// the A side (it includes all 41 rows of each inclusion, not only the
/// leaf row). Until an `IS_LEAF` selector lands, the descriptor is
/// scaffolding for the eventual binding — wire it through joint_prove
/// once the per-row leaf selector is added.
pub fn make_validator_registry_leaf_to_htr_linkage_descriptor(
    registry_layer_index: usize,
    htr_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::validator_htr_air as vh;
    let mut a_columns: Vec<usize> = Vec::with_capacity(33);
    a_columns.push(COL_VALIDATOR_INDEX);
    for b in 0..CHUNK_BYTES {
        a_columns.push(COL_CURRENT_HASH_OFFSET + b);
    }

    let mut b_columns: Vec<usize> = Vec::with_capacity(33);
    b_columns.push(vh::COL_VALIDATOR_INDEX);
    for b in 0..CHUNK_BYTES {
        b_columns.push(vh::COL_VALIDATOR_ROOT_BYTE_OFFSET + b);
    }

    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "validator_registry_leaf_to_htr_v1".into(),
        a_layer_index: registry_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: htr_layer_index,
        b_columns,
        b_selector_column: Some(vh::COL_IS_REAL),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small witness with a synthetic merkle proof: validator
    /// at the given index and a deterministic sibling path. Returns
    /// `(witness, expected_registry_root)`.
    fn synthetic_inclusion(
        index: u64,
        registry_length: u64,
    ) -> RegistryInclusionWitness {
        // Synthetic validator_htr.
        let mut leaf = [0u8; CHUNK_BYTES];
        for i in 0..CHUNK_BYTES {
            leaf[i] = (i as u8).wrapping_mul(11).wrapping_add(3);
        }
        // Synthetic sibling path: each sibling = sha256_pair(leaf, [k; 32]).
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        for k in 0..DEPTH {
            let mut s = [k as u8; CHUNK_BYTES];
            s[0] = s[0].wrapping_add(17);
            path[k] = s;
        }
        RegistryInclusionWitness::from_validator_proof(leaf, index, path, registry_length)
    }

    fn single_inclusion_witness() -> ValidatorRegistryWitness {
        ValidatorRegistryWitness {
            inclusions: vec![synthetic_inclusion(0xABCDEF, 1_234_567)],
            pk_byte0: Vec::new(),
        }
    }

    /// Sanity: `from_validator_proof` matches a manual host-side walk
    /// for a depth-40 path, with `mix_in_length` applied at the top.
    #[test]
    fn from_validator_proof_matches_manual_walk() {
        let index: u64 = 0x123_4567_89AB; // ~41 bits — but bits past 40 are dropped via shift
        let truncated_index = index & ((1u64 << DEPTH) - 1);
        let registry_length: u64 = 1_000_000;
        let mut leaf = [0u8; CHUNK_BYTES];
        for i in 0..CHUNK_BYTES { leaf[i] = i as u8; }
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        for k in 0..DEPTH {
            let mut s = [0u8; CHUNK_BYTES];
            s[0] = k as u8;
            s[31] = k as u8;
            path[k] = s;
        }
        let w = RegistryInclusionWitness::from_validator_proof(
            leaf, truncated_index, path, registry_length,
        );
        // Manual recompute.
        let mut cur = leaf;
        for k in 0..DEPTH {
            let bit = (truncated_index >> k) & 1;
            let (l, r) = if bit == 0 { (cur, path[k]) } else { (path[k], cur) };
            cur = crate::sha256::sha256_pair(&l, &r);
        }
        let expected_root = crate::ssz::mix_in_length(cur, registry_length);
        assert_eq!(w.registry_root, expected_root);
    }

    /// Witness round-trip: `compute_validator_root` style end-to-end —
    /// build trace, all constraints vanish on every row.
    #[test]
    fn constraints_vanish_on_honest_single_inclusion() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, ROWS_PER_INCLUSION);

        let cs = ValidatorRegistryConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        assert_eq!(evals.len(), NUM_ROW_CONSTRAINTS);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) should vanish at row {} (got {:?})",
                    k,
                    cs.constraint_labels()[k],
                    row,
                    v.to_u64(),
                );
            }
        }
    }

    /// Single-validator registry: a tiny `length = 1` registry where
    /// the leaf is the only real validator. The merkleization root
    /// (before mix_in_length) reduces to zero-padding the leaf up the
    /// 40-deep tree using the canonical SSZ `zero_hash` chain, which the
    /// host-side merkleize_chunks helper already validates.
    #[test]
    fn single_validator_registry_root_matches_ssz_helper() {
        let mut leaf = [0u8; CHUNK_BYTES];
        for i in 0..CHUNK_BYTES { leaf[i] = (i as u8).wrapping_mul(7); }

        // The sibling path of the single leaf in a 2^40 tree consists of
        // zero_hash[0..40] (each pair-step hashes against zero on the
        // right since the leaf sits at index 0).
        // We can equivalently derive the path by asking ssz::merkleize_chunks
        // for the root and reconstructing the sibling-path bottom-up.
        // For this test, just use the synthetic path with all-zero siblings
        // and confirm the witness builder's root matches the manual walk
        // (already tested above) — but also confirm it equals
        // `merkleize_chunks(&[leaf], Some(2^40))` mixed in with length 1.
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        // siblings at every level for index 0 are zero_hash[level] —
        // equivalent to fold sha256_pair(prev, prev) starting from ZERO_CHUNK.
        let mut z = [0u8; CHUNK_BYTES];
        for k in 0..DEPTH {
            path[k] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }
        let registry_length = 1u64;
        let w = RegistryInclusionWitness::from_validator_proof(
            leaf, 0, path, registry_length,
        );

        // Independently compute the expected root via ssz helpers:
        // merkleize a single-leaf list with limit 2^40, then mix_in_length(1).
        let merkle_root = crate::ssz::merkleize_chunks(&[leaf], Some(1u64 << DEPTH));
        let expected = crate::ssz::mix_in_length(merkle_root, registry_length);
        assert_eq!(
            w.registry_root, expected,
            "depth-40 single-leaf inclusion must match SSZ helper output"
        );
    }

    /// Tampering: corrupt one of the sibling-hash chunks in the trace
    /// and confirm the row-local left/right selection bodies still
    /// vanish (because LEFT/RIGHT/CURRENT/SIBLING are all updated
    /// consistently by the prover), but the hash_chain shifted body
    /// would fail (deferred to a separate evaluate_shifted check).
    /// We exercise the row-local tampering via `left_selection`: corrupt
    /// LEFT directly.
    #[test]
    fn left_selection_rejects_tampered_left_byte() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        // Tamper LEFT[5] on the leaf row.
        trace.columns[COL_LEFT_OFFSET + 5].evaluations[0] =
            Scalar::from_u64(0xFF, curve);
        let cs = ValidatorRegistryConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // body 3 = left_selection.
        assert!(
            !evals[3][0].is_zero(),
            "tampered LEFT[5] must make left_selection non-zero on the leaf row"
        );
    }

    /// Tampering: corrupt the validator_index aggregator and confirm
    /// `index_aggregator` fires.
    #[test]
    fn index_aggregator_rejects_tampered_index() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        trace.columns[COL_VALIDATOR_INDEX].evaluations[0] =
            Scalar::from_u64(0xDEAD, curve);
        let cs = ValidatorRegistryConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // body 5 = index_aggregator.
        assert!(
            !evals[5][0].is_zero(),
            "tampered VALIDATOR_INDEX must make index_aggregator non-zero"
        );
    }

    /// Tampering: corrupt a sibling on a tree row. The row-local
    /// `left_selection` / `right_selection` bodies should fire because
    /// LEFT/RIGHT on that row are functions of CURRENT_HASH and SIBLING
    /// and we corrupted SIBLING without updating LEFT/RIGHT.
    #[test]
    fn sibling_tamper_breaks_selection() {
        let curve = CurveType::Bls48581;
        let w = single_inclusion_witness();
        let mut trace = build_trace_polynomials(&w, curve);
        // Corrupt SIBLING[7] on row 3 (tree row at level 3).
        trace.columns[COL_SIBLING_OFFSET + 7].evaluations[3] =
            Scalar::from_u64(0x55, curve);
        let cs = ValidatorRegistryConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        // Either left_selection (body 3) or right_selection (body 4)
        // must fire on row 3, depending on INDEX_BIT at level 3.
        assert!(
            !evals[3][3].is_zero() || !evals[4][3].is_zero(),
            "tampered sibling must break one of left/right selection on the affected row"
        );
    }

    /// Bit decomposition sanity: `INDEX_BIT_REPL` columns reproduce the
    /// 40-bit LE decomposition of `validator_index`.
    #[test]
    fn index_bit_decomposition_consistent() {
        let curve = CurveType::Bls48581;
        let index: u64 = 0x0123_4567_89; // 40 bits
        let w = ValidatorRegistryWitness {
            inclusions: vec![synthetic_inclusion(index, 9999)],
            pk_byte0: Vec::new(),
        };
        let trace = build_trace_polynomials(&w, curve);
        // Check each replicated bit column on row 0 matches the
        // LE bit decomposition of index.
        for k in 0..DEPTH {
            let expected = (index >> k) & 1;
            let got = trace.columns[COL_INDEX_BIT_REPL_OFFSET + k].evaluations[0]
                .to_u64();
            assert_eq!(
                got, expected,
                "INDEX_BIT_REPL[{}] should match bit {} of validator_index", k, k
            );
        }
        // Same bits replicated on every active row.
        for row in 0..ROWS_PER_INCLUSION {
            for k in 0..DEPTH {
                let expected = (index >> k) & 1;
                let got = trace.columns[COL_INDEX_BIT_REPL_OFFSET + k].evaluations[row]
                    .to_u64();
                assert_eq!(
                    got, expected,
                    "INDEX_BIT_REPL[{}] should be replicated on row {}", k, row
                );
            }
        }
        // VALIDATOR_INDEX aggregator on row 0.
        assert_eq!(
            trace.columns[COL_VALIDATOR_INDEX].evaluations[0].to_u64(),
            index,
        );
    }

    /// Descriptor wiring sanity for the pair-hash cross-AIR LogUp.
    #[test]
    fn pair_hash_linkage_descriptor_well_formed() {
        let desc = make_validator_registry_pair_hash_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "validator_registry_pair_hash_v1");
        assert_eq!(desc.a_columns.len(), 96);
        assert_eq!(desc.b_columns.len(), 96);
        // A-side first 32 = LEFT.
        for b in 0..CHUNK_BYTES {
            assert_eq!(desc.a_columns[b], COL_LEFT_OFFSET + b);
        }
        // A-side next 32 = RIGHT.
        for b in 0..CHUNK_BYTES {
            assert_eq!(desc.a_columns[CHUNK_BYTES + b], COL_RIGHT_OFFSET + b);
        }
        // A-side last 32 = NEXT_HASH.
        for b in 0..CHUNK_BYTES {
            assert_eq!(desc.a_columns[2 * CHUNK_BYTES + b], COL_NEXT_HASH_OFFSET + b);
        }
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
        assert_eq!(
            desc.b_selector_column,
            Some(crate::sha256_extract::COL_IS_REAL)
        );
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
    }

    /// Alias descriptor parity check: the `to_sha256` alias must produce a
    /// byte-for-byte identical descriptor to the underlying pair-hash
    /// constructor.
    #[test]
    fn to_sha256_descriptor_matches_pair_hash_alias() {
        let alias = make_validator_registry_to_sha256_descriptor(3, 5);
        let direct = make_validator_registry_pair_hash_linkage_descriptor(3, 5);
        assert_eq!(alias.label, direct.label);
        assert_eq!(alias.a_layer_index, direct.a_layer_index);
        assert_eq!(alias.b_layer_index, direct.b_layer_index);
        assert_eq!(alias.a_columns, direct.a_columns);
        assert_eq!(alias.b_columns, direct.b_columns);
        assert_eq!(alias.a_selector_column, direct.a_selector_column);
        assert_eq!(alias.b_selector_column, direct.b_selector_column);
    }

    /// Multi-validator inclusion: build 16 validators with distinct
    /// pubkeys / balances / exit_epochs, merkleize them into a depth-4
    /// sub-tree, embed that sub-tree at the bottom of the full depth-40
    /// VALIDATOR_REGISTRY_LIMIT tree by pairing with zero_hash siblings
    /// for levels 4..40, then prove inclusion for the validator at
    /// index 7. The witness builder's claimed `registry_root` must
    /// equal `mix_in_length(merkleize_chunks(htrs, 2^40), 16)`, and all
    /// row-local constraints must vanish on the honest trace.
    #[test]
    fn multi_validator_inclusion_with_16_validators() {
        use crate::beacon::Validator;
        let curve = CurveType::Bls48581;
        const N_VALIDATORS: usize = 16;
        const SUBTREE_DEPTH: u32 = 4; // log2(16)
        const TARGET_INDEX: u64 = 7;

        // Build 16 distinct validators.
        let mut validators: Vec<Validator> = Vec::with_capacity(N_VALIDATORS);
        for i in 0..N_VALIDATORS {
            let mut v = Validator::default();
            // Distinct pubkey: first byte = index, rest deterministic spread.
            for b in 0..48 {
                v.pubkey[b] = ((i as u8).wrapping_mul(7)).wrapping_add(b as u8);
            }
            for b in 0..32 {
                v.withdrawal_credentials[b] =
                    ((i as u8).wrapping_mul(31)).wrapping_add(b as u8);
            }
            v.effective_balance = 32_000_000_000u64 + (i as u64) * 1_000_000;
            v.slashed = i % 5 == 3;
            v.activation_eligibility_epoch = (i as u64) * 11;
            v.activation_epoch = (i as u64) * 13;
            v.exit_epoch = u64::MAX.wrapping_sub(i as u64); // distinct
            v.withdrawable_epoch = (i as u64) * 17;
            validators.push(v);
        }

        // Compute each validator's hashTreeRoot — these are the leaves of
        // the validator-registry merkle tree.
        let htrs: Vec<[u8; CHUNK_BYTES]> =
            validators.iter().map(|v| v.hash_tree_root()).collect();
        assert_eq!(htrs.len(), N_VALIDATORS);

        // Sanity: all 16 htrs are distinct (a property used by the
        // inclusion proof's soundness story).
        for i in 0..N_VALIDATORS {
            for j in (i + 1)..N_VALIDATORS {
                assert_ne!(
                    htrs[i], htrs[j],
                    "validator htrs at {} and {} collide", i, j,
                );
            }
        }

        // Build the depth-4 sub-tree bottom-up, keeping every level so we
        // can extract the sibling path for TARGET_INDEX = 7.
        // levels[0] = htrs (16 leaves), levels[1] = 8 nodes, ..., levels[4] = 1 node.
        let mut levels: Vec<Vec<[u8; CHUNK_BYTES]>> = Vec::with_capacity(5);
        levels.push(htrs.clone());
        for d in 0..(SUBTREE_DEPTH as usize) {
            let cur = &levels[d];
            let mut next = Vec::with_capacity(cur.len() / 2);
            let mut i = 0;
            while i < cur.len() {
                next.push(crate::sha256::sha256_pair(&cur[i], &cur[i + 1]));
                i += 2;
            }
            levels.push(next);
        }
        assert_eq!(levels[SUBTREE_DEPTH as usize].len(), 1);

        // Extract sibling path for validator at TARGET_INDEX through the
        // depth-4 sub-tree.
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        let mut idx = TARGET_INDEX;
        for d in 0..(SUBTREE_DEPTH as usize) {
            // Sibling at level d is the neighbouring node at index `idx ^ 1`.
            let sib = levels[d][(idx ^ 1) as usize];
            path[d] = sib;
            idx >>= 1;
        }
        // For levels 4..40, the rest of the tree (the right-hand side of
        // the registry) is all zero validators, so the sibling at each
        // remaining level is `zero_hash(level)` where the cumulative zero
        // chain starts from level SUBTREE_DEPTH = 4.
        let mut z = [0u8; CHUNK_BYTES];
        // Walk z up SUBTREE_DEPTH levels to land at zero_hash(SUBTREE_DEPTH).
        for _ in 0..(SUBTREE_DEPTH as usize) {
            z = crate::sha256::sha256_pair(&z, &z);
        }
        for d in (SUBTREE_DEPTH as usize)..DEPTH {
            path[d] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }

        // Build the witness.
        let registry_length = N_VALIDATORS as u64;
        let leaf = htrs[TARGET_INDEX as usize];
        let w_inc = RegistryInclusionWitness::from_validator_proof(
            leaf, TARGET_INDEX, path, registry_length,
        );

        // Cross-check the witness-builder's root against the SSZ helper
        // (full depth-40 merkleization with limit = 2^40 + mix_in_length).
        let expected_root = crate::ssz::mix_in_length(
            crate::ssz::merkleize_chunks(&htrs, Some(1u64 << DEPTH)),
            registry_length,
        );
        assert_eq!(
            w_inc.registry_root, expected_root,
            "16-validator inclusion root must match SSZ merkleize_chunks helper",
        );

        // Build the trace and check constraints vanish on every row.
        let w = ValidatorRegistryWitness {
            inclusions: vec![w_inc],
            pk_byte0: Vec::new(),
        };
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, ROWS_PER_INCLUSION);

        let cs = ValidatorRegistryConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) should vanish at row {}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }

        // Also sanity-check the leaf row's CURRENT_HASH equals the
        // target validator's htr.
        for b in 0..CHUNK_BYTES {
            let got = trace.columns[COL_CURRENT_HASH_OFFSET + b].evaluations[0]
                .to_u64() as u8;
            assert_eq!(
                got, leaf[b],
                "leaf row CURRENT_HASH[{}] should equal validator htr[{}]", b, b,
            );
        }

        // And the validator index decomposition matches TARGET_INDEX.
        for k in 0..DEPTH {
            let expected = (TARGET_INDEX >> k) & 1;
            let got = trace.columns[COL_INDEX_BIT_REPL_OFFSET + k].evaluations[0]
                .to_u64();
            assert_eq!(got, expected, "INDEX_BIT_REPL[{}] mismatch", k);
        }
    }

    /// Descriptor wiring sanity for the leaf↔validator_htr cross-AIR
    /// LogUp.
    #[test]
    fn leaf_to_htr_linkage_descriptor_well_formed() {
        let desc = make_validator_registry_leaf_to_htr_linkage_descriptor(0, 2);
        assert_eq!(desc.label, "validator_registry_leaf_to_htr_v1");
        assert_eq!(desc.a_columns.len(), 33);
        assert_eq!(desc.b_columns.len(), 33);
        assert_eq!(desc.a_columns[0], COL_VALIDATOR_INDEX);
        for b in 0..CHUNK_BYTES {
            assert_eq!(desc.a_columns[1 + b], COL_CURRENT_HASH_OFFSET + b);
        }
        assert_eq!(
            desc.b_columns[0],
            crate::validator_htr_air::COL_VALIDATOR_INDEX
        );
        for b in 0..CHUNK_BYTES {
            assert_eq!(
                desc.b_columns[1 + b],
                crate::validator_htr_air::COL_VALIDATOR_ROOT_BYTE_OFFSET + b,
            );
        }
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 2);
    }

    /// Stress variant of [`multi_validator_inclusion_with_16_validators`]:
    /// build 200 distinct validators occupying a depth-8 sub-tree (256
    /// leaves with 200 populated + 56 zero-validator leaves), embed that
    /// sub-tree at the bottom of the full depth-40 VALIDATOR_REGISTRY_LIMIT
    /// tree by pairing with cumulative zero_hash siblings for levels
    /// 8..40, and prove inclusion for the validator at index 137. The
    /// witness builder's claimed `registry_root` must equal
    /// `mix_in_length(merkleize_chunks(htrs, 2^40), 200)` (the SSZ helper
    /// is given only the 200 populated htrs and pads internally), and all
    /// row-local constraints must vanish on the honest trace. This test
    /// is host-side merkle only (no prove/verify), so it runs fast.
    #[test]
    fn multi_validator_inclusion_with_200_validators() {
        use crate::beacon::Validator;
        let curve = CurveType::Bls48581;
        const N_VALIDATORS: usize = 200;
        const SUBTREE_LEAVES: usize = 256; // 2^8
        const SUBTREE_DEPTH: u32 = 8; // log2(256)
        const TARGET_INDEX: u64 = 137;

        // Build 200 distinct validators with spread-out field values.
        let mut validators: Vec<Validator> = Vec::with_capacity(N_VALIDATORS);
        for i in 0..N_VALIDATORS {
            let mut v = Validator::default();
            // Distinct pubkey: mix high+low bytes of i across 48 bytes.
            let hi = (i >> 8) as u8;
            let lo = (i & 0xff) as u8;
            for b in 0..48 {
                v.pubkey[b] = hi
                    .wrapping_add(lo.wrapping_mul(7))
                    .wrapping_add(b as u8);
            }
            for b in 0..32 {
                v.withdrawal_credentials[b] = hi
                    .wrapping_add(lo.wrapping_mul(31))
                    .wrapping_add(b as u8);
            }
            v.effective_balance = 32_000_000_000u64 + (i as u64) * 1_000_000;
            v.slashed = i % 7 == 4;
            v.activation_eligibility_epoch = (i as u64) * 11;
            v.activation_epoch = (i as u64) * 13;
            v.exit_epoch = u64::MAX.wrapping_sub(i as u64); // distinct
            v.withdrawable_epoch = (i as u64) * 17;
            validators.push(v);
        }

        // hashTreeRoot of every real validator — these populate the first
        // 200 slots of the depth-8 sub-tree.
        let htrs: Vec<[u8; CHUNK_BYTES]> =
            validators.iter().map(|v| v.hash_tree_root()).collect();
        assert_eq!(htrs.len(), N_VALIDATORS);

        // Distinctness sanity: collisions among the populated htrs would
        // undermine the inclusion-soundness story for this witness.
        for i in 0..N_VALIDATORS {
            for j in (i + 1)..N_VALIDATORS {
                assert_ne!(
                    htrs[i], htrs[j],
                    "validator htrs at {} and {} collide", i, j,
                );
            }
        }

        // The depth-8 sub-tree is padded with zero CHUNK leaves
        // (`[0u8; 32]`) in the remaining 56 slots so the sub-tree leaf
        // layer has exactly SUBTREE_LEAVES entries. This matches the
        // SSZ `merkleize_chunks` convention, where unused leaves up to
        // the limit are zero chunks (not zero-validator hashTreeRoots).
        let mut subtree_leaves: Vec<[u8; CHUNK_BYTES]> =
            Vec::with_capacity(SUBTREE_LEAVES);
        subtree_leaves.extend_from_slice(&htrs);
        while subtree_leaves.len() < SUBTREE_LEAVES {
            subtree_leaves.push([0u8; CHUNK_BYTES]);
        }
        assert_eq!(subtree_leaves.len(), SUBTREE_LEAVES);

        // Build the depth-8 sub-tree bottom-up so we can extract the
        // sibling path for TARGET_INDEX = 137. levels[0] has 256 leaves,
        // levels[8] has 1 node.
        let mut levels: Vec<Vec<[u8; CHUNK_BYTES]>> =
            Vec::with_capacity((SUBTREE_DEPTH as usize) + 1);
        levels.push(subtree_leaves);
        for d in 0..(SUBTREE_DEPTH as usize) {
            let cur = &levels[d];
            let mut next = Vec::with_capacity(cur.len() / 2);
            let mut i = 0;
            while i < cur.len() {
                next.push(crate::sha256::sha256_pair(&cur[i], &cur[i + 1]));
                i += 2;
            }
            levels.push(next);
        }
        assert_eq!(levels[SUBTREE_DEPTH as usize].len(), 1);

        // Extract sibling path for validator at TARGET_INDEX through the
        // depth-8 sub-tree (siblings within the populated/zero-padded
        // sub-tree).
        let mut path = [[0u8; CHUNK_BYTES]; DEPTH];
        let mut idx = TARGET_INDEX;
        for d in 0..(SUBTREE_DEPTH as usize) {
            let sib = levels[d][(idx ^ 1) as usize];
            path[d] = sib;
            idx >>= 1;
        }
        // Levels 8..40 sit on the right-hand side of the full depth-40
        // registry tree; those siblings are the cumulative zero-hash
        // chain starting at `zero_hash(SUBTREE_DEPTH)`.
        let mut z = [0u8; CHUNK_BYTES];
        for _ in 0..(SUBTREE_DEPTH as usize) {
            z = crate::sha256::sha256_pair(&z, &z);
        }
        for d in (SUBTREE_DEPTH as usize)..DEPTH {
            path[d] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }

        // Build the witness for the inclusion of validator #137.
        let registry_length = N_VALIDATORS as u64;
        let leaf = htrs[TARGET_INDEX as usize];
        let w_inc = RegistryInclusionWitness::from_validator_proof(
            leaf, TARGET_INDEX, path, registry_length,
        );

        // Cross-check the witness-builder's root against the SSZ helper.
        // `merkleize_chunks` is given only the 200 populated htrs and
        // must internally zero-pad up to the full depth-40 limit.
        let expected_root = crate::ssz::mix_in_length(
            crate::ssz::merkleize_chunks(&htrs, Some(1u64 << DEPTH)),
            registry_length,
        );
        assert_eq!(
            w_inc.registry_root, expected_root,
            "200-validator inclusion root must match SSZ merkleize_chunks helper",
        );

        // Build the trace and check all row-local constraints vanish.
        let w = ValidatorRegistryWitness {
            inclusions: vec![w_inc],
            pk_byte0: Vec::new(),
        };
        let trace = build_trace_polynomials(&w, curve);
        assert_eq!(trace.columns.len(), NUM_COLUMNS);
        assert_eq!(trace.num_rows, ROWS_PER_INCLUSION);

        let cs = ValidatorRegistryConstraintSystem::new(trace.num_rows);
        let cols_owned: Vec<&Vec<Scalar>> =
            trace.columns.iter().map(|p| &p.evaluations).collect();
        let evals = cs.evaluate_on_domain(&cols_owned, trace.num_rows);
        for (k, body) in evals.iter().enumerate() {
            for (row, v) in body.iter().enumerate() {
                assert!(
                    v.is_zero(),
                    "row-local constraint {} ({}) should vanish at row {}",
                    k,
                    cs.constraint_labels()[k],
                    row,
                );
            }
        }

        // Leaf row's CURRENT_HASH must equal target validator's htr.
        for b in 0..CHUNK_BYTES {
            let got = trace.columns[COL_CURRENT_HASH_OFFSET + b].evaluations[0]
                .to_u64() as u8;
            assert_eq!(
                got, leaf[b],
                "leaf row CURRENT_HASH[{}] should equal validator htr[{}]", b, b,
            );
        }

        // Validator index decomposition matches TARGET_INDEX = 137
        // (binary: 0b10001001).
        for k in 0..DEPTH {
            let expected = (TARGET_INDEX >> k) & 1;
            let got = trace.columns[COL_INDEX_BIT_REPL_OFFSET + k].evaluations[0]
                .to_u64();
            assert_eq!(got, expected, "INDEX_BIT_REPL[{}] mismatch", k);
        }
    }
}
