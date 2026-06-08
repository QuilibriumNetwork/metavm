//! Address-keccak gadget AIR — Phase A2 step 3 stepping stone.
//!
//! Bridges 20-byte Ethereum addresses to their world-state MPT trie
//! keys via `trie_key = keccak256(address)`. One row per address that
//! the surrounding chain needs to hash (typically one per storage
//! access by a unique contract; deduplication is a future
//! optimization).
//!
//! Columns:
//!   - `address_be[0..20]` — 20 BE bytes of the address.
//!   - `address_limb[0..4]` — 4 LE u64 limbs (top 12 bytes / 96 bits
//!     zero), bound to `address_be` via byte-decomp constraints. The
//!     limb form is what `storage_access_air::COL_ADDR_L0..L3` and
//!     other gadgets expose; the byte form is what KeccakExtract
//!     takes as input.
//!   - `address_trie_key[0..32]` — `keccak256(address)`, the world
//!     MPT key for the account state.
//!   - `is_real` — 1 on real rows, 0 on padding.
//!
//! Row-local constraints:
//!   - `is_real` binary.
//!   - 4 limb-decomp equations: `address_limb_j = Σ address_be[…] *
//!     2^…` (BE bytes 0..7 → limb 3 MSB, bytes 12..20 → limb 0 LSB).
//!     Top 12 bytes (positions 0..11) are constrained to 0 via the
//!     limb-decomp itself (since they decompose into limb 3 high
//!     bytes which must equal 0 for a valid 20-byte address).
//!
//! Linkages (descriptors below):
//!   - L_addr_to_keccak: this gadget's `(address_be[0..20],
//!     address_trie_key[0..32])` ↔ KeccakExtract's
//!     `(INPUT_BYTE[0..20], OUTPUT_BYTE[0..32])` — 52-col tuple.
//!     Binds `address_trie_key = keccak256(address)`.
//!   - L_storage_to_addr_keccak (storage side, in cross_air_linkage):
//!     storage gadget's address limbs ↔ this gadget's
//!     `address_limb[0..4]` — binds the storage gadget's contract
//!     address to its keccak hash.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

pub const ADDRESS_LEN: usize = 20;
pub const TRIE_KEY_LEN: usize = 32;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_ADDRESS_BE_OFFSET: usize = 0; // 0..20
pub const COL_ADDRESS_LIMB_L0: usize = 20;
pub const COL_ADDRESS_LIMB_L1: usize = 21;
pub const COL_ADDRESS_LIMB_L2: usize = 22;
pub const COL_ADDRESS_LIMB_L3: usize = 23;
pub const COL_ADDRESS_TRIE_KEY_OFFSET: usize = 24; // 24..56
pub const COL_IS_REAL: usize = 56;
pub const NUM_COLUMNS: usize = 57;

/// Row-local constraints:
///   0: is_real binary
///   1..5: 4 limb-decomp equations (Σ bytes * 2^k = limb)
pub const NUM_ROW_CONSTRAINTS: usize = 1 + 4;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct AddressKeccakRow {
    pub address: [u8; ADDRESS_LEN],
}

#[derive(Clone, Debug, Default)]
pub struct AddressKeccakWitness {
    pub invocations: Vec<AddressKeccakRow>,
}

impl AddressKeccakWitness {
    pub fn from_addresses(addresses: Vec<[u8; ADDRESS_LEN]>) -> Self {
        Self {
            invocations: addresses
                .into_iter()
                .map(|a| AddressKeccakRow { address: a })
                .collect(),
        }
    }
}

/// 20-byte BE address → 4 LE u64 limbs (top 12 bytes / 96 bits = 0).
/// Mirrors `storage_access_air::address_to_limbs`.
fn address_to_limbs(address: [u8; 20]) -> [u64; 4] {
    let mut full = [0u8; 32];
    full[12..32].copy_from_slice(&address);
    [
        u64::from_be_bytes([
            full[24], full[25], full[26], full[27],
            full[28], full[29], full[30], full[31],
        ]),
        u64::from_be_bytes([
            full[16], full[17], full[18], full[19],
            full[20], full[21], full[22], full[23],
        ]),
        u64::from_be_bytes([
            full[8],  full[9],  full[10], full[11],
            full[12], full[13], full[14], full[15],
        ]),
        u64::from_be_bytes([
            full[0], full[1], full[2], full[3],
            full[4], full[5], full[6], full[7],
        ]),
    ]
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &AddressKeccakWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.invocations.iter().enumerate() {
        for k in 0..ADDRESS_LEN {
            columns[COL_ADDRESS_BE_OFFSET + k][i] =
                Scalar::from_u64(row.address[k] as u64, curve);
        }
        let limbs = address_to_limbs(row.address);
        for j in 0..4 {
            columns[COL_ADDRESS_LIMB_L0 + j][i] = Scalar::from_u64(limbs[j], curve);
        }
        let hash = crate::keccak::keccak256(&row.address);
        for k in 0..TRIE_KEY_LEN {
            columns[COL_ADDRESS_TRIE_KEY_OFFSET + k][i] =
                Scalar::from_u64(hash[k] as u64, curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
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

pub struct AddressKeccakConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl AddressKeccakConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// `(byte_index_in_be, weight)` for the `limb_idx`-th LE u64 limb of a
/// 32-byte BE quantity (used to constrain address limbs against
/// address_be when address occupies the LOW 20 bytes; the high 12
/// bytes are all zero, so limb 3 ends up = 0 for valid 20-byte
/// addresses).
fn limb_decomp_targets_for_address(limb_idx: usize) -> Vec<(usize, u64)> {
    // The 32-byte BE form of the address: positions 0..12 are zero
    // (padding), positions 12..32 are the actual address bytes.
    // Limb 0 (LSB) ↔ BE bytes 24..32 → address bytes 12..20 (i.e.,
    //   address_be[12..20] = original address[12..20])
    //   Wait — address_be IS the original 20-byte address (we don't
    //   store the 32-byte zero-padded form; we only store the 20
    //   actual bytes). So address_be[0] is the first byte of the
    //   address (which corresponds to full[12] in the 32-byte form).
    //
    // Re-derived mapping:
    //   limb 0 (low 64 bits of U256) = full[24..32] BE
    //     = address[12..20] BE
    //     = address_be[12..20] (since address_be == address)
    //     = byte_idx 12,13,14,15,16,17,18,19 with weights 2^56..2^0
    //   limb 1 = full[16..24] = address[4..12] = address_be[4..12]
    //   limb 2 = full[8..16] = (full[8..12] are zero, address[0..4]
    //                          = address_be[0..4])
    //     Position 8 of full = 0 (out of address range)
    //     Position 9..12 of full = 0 (still out of address range)
    //     Position 12..16 of full = address[0..4] = address_be[0..4]
    //     So limb 2 contains 4 zero bytes then 4 address bytes:
    //     limb 2 = address_be[0]*2^24 + address_be[1]*2^16 +
    //              address_be[2]*2^8 + address_be[3]*2^0
    //     (only positions 12..16 of full contribute, i.e., address_be[0..4]).
    //   limb 3 = full[0..8] = all zero (out of address range)
    //
    // The constraints reference address_be indices, not the 32-byte
    // full form, so:
    //   limb 0 = Σ_{k=12..20} address_be[k] * 2^(8*(31-pos))
    //     where pos is the 32-byte position. For k=12, full pos=24,
    //     weight = 2^(8*7) = 2^56. For k=19, full pos=31, weight=2^0.
    //   ...

    match limb_idx {
        0 => (0..8).map(|k| (12 + k, 1u64 << (8 * (7 - k)))).collect(),
        1 => (0..8).map(|k| (4 + k, 1u64 << (8 * (7 - k)))).collect(),
        2 => (0..4).map(|k| (k, 1u64 << (8 * (3 - k)))).collect(),
        3 => Vec::new(), // limb 3 == 0 always for 20-byte addresses
        _ => unreachable!(),
    }
}

impl VmConstraintSystem for AddressKeccakConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = vec!["is_real_binary".into()];
        for j in 0..4 { labels.push(format!("address_limb_{}_decomp", j)); }
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
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real binary.
        {
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1..5: limb_j_decomp.
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets_for_address(limb_idx);
            let mut c = vec![Scalar::zero(curve); n];
            for r in 0..n {
                let mut sum = Scalar::zero(curve);
                for (byte_idx, pow) in &targets {
                    let b = &columns[COL_ADDRESS_BE_OFFSET + byte_idx][r];
                    sum = sum.add(&b.mul(&Scalar::from_u64(*pow, curve)));
                }
                c[r] = columns[COL_ADDRESS_LIMB_L0 + limb_idx][r].sub(&sum);
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

        let mut acc = Scalar::zero(curve);
        let mut alpha_pow = Scalar::one(curve);

        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets_for_address(limb_idx);
            let mut sum = Scalar::zero(curve);
            for (byte_idx, pow) in &targets {
                sum = sum.add(
                    &col_evals[COL_ADDRESS_BE_OFFSET + byte_idx]
                        .mul(&Scalar::from_u64(*pow, curve)),
                );
            }
            let body = col_evals[COL_ADDRESS_LIMB_L0 + limb_idx].sub(&sum);
            acc = acc.add(&alpha_pow.mul(&body));
            alpha_pow = alpha_pow.mul(alpha);
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

        let mut acc = vec![Scalar::zero(curve)];
        let mut alpha_pow = Scalar::one(curve);

        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        for limb_idx in 0..4 {
            let targets = limb_decomp_targets_for_address(limb_idx);
            let mut sum = vec![Scalar::zero(curve)];
            for (byte_idx, pow) in &targets {
                let b = &col_coeffs[COL_ADDRESS_BE_OFFSET + byte_idx];
                let term = poly_scalar_mul(b, &Scalar::from_u64(*pow, curve));
                sum = poly_add(&sum, &term, curve);
            }
            let body = poly_sub(&col_coeffs[COL_ADDRESS_LIMB_L0 + limb_idx], &sum, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        acc
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
        let curve = columns[0].first().map(|s| s.curve_type()).unwrap_or(CurveType::Bls48581);
        let zero = Scalar::zero(curve);
        for col in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in col.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        // 8-bit range check on every byte column (address + trie_key).
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        for k in 0..ADDRESS_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("address_be_{}_8bit", k),
                    column_index: COL_ADDRESS_BE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        for k in 0..TRIE_KEY_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("address_trie_key_{}_8bit", k),
                    column_index: COL_ADDRESS_TRIE_KEY_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                0,
            ));
        }
        LookupRequirements { tables, declarations }
    }
}

// ─── Linkage descriptors ──────────────────────────────────────────────

/// Link to KeccakExtract: this gadget's `(address_be[0..20],
/// address_trie_key[0..32])` ↔ KeccakExtract's `(INPUT_BYTE[0..20],
/// OUTPUT_BYTE[0..32])`. Binds `address_trie_key = keccak256(address)`.
pub fn make_address_to_keccak_extract_linkage_descriptor(
    address_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract as ke;
    let a_columns: Vec<usize> = (0..ADDRESS_LEN)
        .map(|k| COL_ADDRESS_BE_OFFSET + k)
        .chain((0..TRIE_KEY_LEN).map(|k| COL_ADDRESS_TRIE_KEY_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = (0..ADDRESS_LEN)
        .map(|k| ke::COL_INPUT_BYTE_OFFSET + k)
        .chain((0..TRIE_KEY_LEN).map(|k| ke::COL_OUTPUT_BYTE_OFFSET + k))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "address_keccak_v1".into(),
        a_layer_index: address_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

/// Link from storage gadget to this address-keccak gadget on the 4
/// address limbs. Forces storage_access_air's contract address to
/// have a corresponding row in this gadget (whose own constraints +
/// keccak link bind its byte form to a known address_trie_key).
pub fn make_storage_to_address_keccak_linkage_descriptor(
    storage_layer_index: usize,
    address_keccak_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::storage_access_air as sa;
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "storage_to_address_keccak_v1".into(),
        a_layer_index: storage_layer_index,
        a_columns: vec![sa::COL_ADDR_L0, sa::COL_ADDR_L1, sa::COL_ADDR_L2, sa::COL_ADDR_L3],
        a_selector_column: Some(sa::COL_IS_REAL),
        b_layer_index: address_keccak_layer_index,
        b_columns: vec![
            COL_ADDRESS_LIMB_L0,
            COL_ADDRESS_LIMB_L1,
            COL_ADDRESS_LIMB_L2,
            COL_ADDRESS_LIMB_L3,
        ],
        b_selector_column: Some(COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_builder_populates_address_and_trie_key() {
        let address = [
            0xab, 0xcd, 0xef, 0x01, 0x02, 0x03, 0x04, 0x05,
            0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11,
        ];
        let w = AddressKeccakWitness::from_addresses(vec![address]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        for k in 0..ADDRESS_LEN {
            assert_eq!(
                trace.columns[COL_ADDRESS_BE_OFFSET + k].evaluations[0].to_u64(),
                address[k] as u64,
            );
        }
        // Trie key first/last bytes match keccak256(address).
        let expected = crate::keccak::keccak256(&address);
        assert_eq!(
            trace.columns[COL_ADDRESS_TRIE_KEY_OFFSET].evaluations[0].to_u64(),
            expected[0] as u64,
        );
        assert_eq!(
            trace.columns[COL_ADDRESS_TRIE_KEY_OFFSET + 31].evaluations[0].to_u64(),
            expected[31] as u64,
        );
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let address = [0xab; 20];
        let w = AddressKeccakWitness::from_addresses(vec![address]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = AddressKeccakConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
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
    fn limb_decomp_fires_on_tampered_byte() {
        let address = [0u8; 20];
        let w = AddressKeccakWitness::from_addresses(vec![address]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        // Tamper byte 19 (last byte of address, LSB of limb 0).
        cols[COL_ADDRESS_BE_OFFSET + 19][0] = Scalar::from_u64(0xff, CurveType::Bls48581);
        let cs = AddressKeccakConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // limb 0 decomp is constraint index 1.
        assert!(!results[1][0].is_zero(), "limb_0 decomp should fire");
    }

    #[test]
    fn keccak_linkage_descriptor_well_formed() {
        let desc = make_address_to_keccak_extract_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "address_keccak_v1");
        assert_eq!(desc.a_columns.len(), 52);
        assert_eq!(desc.b_columns.len(), 52);
        // First 20 = address bytes.
        for k in 0..ADDRESS_LEN {
            assert_eq!(desc.a_columns[k], COL_ADDRESS_BE_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                crate::keccak_extract::COL_INPUT_BYTE_OFFSET + k,
            );
        }
        // Next 32 = trie key.
        for k in 0..TRIE_KEY_LEN {
            assert_eq!(desc.a_columns[20 + k], COL_ADDRESS_TRIE_KEY_OFFSET + k);
            assert_eq!(
                desc.b_columns[20 + k],
                crate::keccak_extract::COL_OUTPUT_BYTE_OFFSET + k,
            );
        }
    }

    #[test]
    fn storage_linkage_descriptor_well_formed() {
        let desc = make_storage_to_address_keccak_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "storage_to_address_keccak_v1");
        assert_eq!(desc.a_columns.len(), 4);
        assert_eq!(desc.b_columns.len(), 4);
    }

    #[test]
    fn limb_decomp_targets_top_limb_is_empty() {
        let targets = limb_decomp_targets_for_address(3);
        assert!(targets.is_empty(),
                "limb 3 has no address-byte contributions");
    }

    #[test]
    fn address_to_limbs_zero_address() {
        let limbs = address_to_limbs([0u8; 20]);
        assert_eq!(limbs, [0u64, 0, 0, 0]);
    }

    #[test]
    fn standalone_prove_with_scheme_passes() {
        use crate::prover::prove_with_scheme;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::verifier::verify_with_scheme;

        let scheme = Bls48581Scheme::new();
        scheme.init();
        let curve = CurveType::Bls48581;

        let address = [0xab; 20];
        let w = AddressKeccakWitness::from_addresses(vec![address]);
        let trace = build_trace_polynomials(&w, curve);
        let omega = scheme.domain_generator(trace.padded_size);
        let cs = AddressKeccakConstraintSystem::new(trace.num_rows)
            .with_omega_and_domain(omega, trace.padded_size);
        let proof = prove_with_scheme(&trace, &cs, &scheme);
        assert!(
            verify_with_scheme(&proof, &cs, &scheme, curve),
            "standalone address_keccak_air proof must verify",
        );
    }

    /// **Phase A2 step 3a joint_prove**: 2-AIR `address_keccak_air` ↔
    /// `KeccakExtract` via the 52-col tuple linkage. Validates that
    /// `address_trie_key = keccak256(address)` is bound algebraically.
    #[test]
    #[ignore = "slow: 2-AIR joint_prove (~3-5 min); run with --release --ignored"]
    fn joint_prove_address_to_keccak_extract() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak_extract::{
            self as ke, build_trace_polynomials as build_keccak_trace,
            KeccakExtractConstraintSystem, KeccakExtractWitness,
        };
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let address = [
            0xab, 0xcd, 0xef, 0x01, 0x02, 0x03, 0x04, 0x05,
            0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11,
        ];

        let addr_w = AddressKeccakWitness::from_addresses(vec![address]);
        let addr_trace = build_trace_polynomials(&addr_w, curve);
        let addr_omega = scheme.domain_generator(addr_trace.padded_size);
        let addr_cs = AddressKeccakConstraintSystem::new(addr_trace.num_rows)
            .with_omega_and_domain(addr_omega, addr_trace.padded_size);

        let keccak_w = KeccakExtractWitness::from_inputs(&[address.to_vec()]).unwrap();
        let keccak_trace = build_keccak_trace(&keccak_w, curve);
        let keccak_omega = scheme.domain_generator(keccak_trace.padded_size);
        let keccak_cs = KeccakExtractConstraintSystem::new(keccak_trace.num_rows)
            .with_omega_and_domain(keccak_omega, keccak_trace.padded_size);

        let _ = ke::COL_INPUT_BYTE_OFFSET;

        let linkage = make_address_to_keccak_extract_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&addr_trace, &addr_cs), (&keccak_trace, &keccak_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("2-AIR address_keccak joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&addr_cs, &keccak_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
            "joint_verify must accept honest witness",
        );
    }

    #[test]
    fn address_to_limbs_max_address() {
        let limbs = address_to_limbs([0xffu8; 20]);
        // limb 3 = 0 (top 96 bits of zero-padded form).
        assert_eq!(limbs[3], 0);
        // limb 0 (BE bytes 24..32 of full = address[12..20] = all 0xff) = 0xffff..ff (u64::MAX).
        assert_eq!(limbs[0], u64::MAX);
        assert_eq!(limbs[1], u64::MAX);
        // limb 2 = full[8..16] = (4 zeros, then address[0..4] = 0xff*4)
        // = 0x00000000_ffffffff
        assert_eq!(limbs[2], 0x0000_0000_ffff_ffff);
    }
}
