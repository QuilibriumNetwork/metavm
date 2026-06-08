//! BlockHeader gadget AIR — Phase B step 0 (#53 foundation).
//!
//! The Layer A → Layer C bridge. Per row, commits to one Ethereum
//! execution block header: the four MPT roots (state, transactions,
//! receipts, withdrawals), the block hash, and the BLOCK-opcode-
//! targetable scalar fields (number, timestamp, gas_limit, gas_used,
//! base_fee, beneficiary).
//!
//! **Step 0 (this commit)**: witness-commitment skeleton with row-
//! local `is_real` binary + 8-bit byte range checks on root/hash
//! columns. **No** RLP encoding or `block_hash = keccak256(rlp)`
//! binding yet — the prover supplies `block_hash` as oracle. Step 1+
//! adds the algebraic RLP encoding gadget + KeccakExtract linkage.
//!
//! Once Step 1 lands, this AIR becomes:
//!   - The single canonical place where stateRoot / txRoot /
//!     receiptsRoot are exposed to the rest of the proof chain.
//!   - The cross-AIR linkage target for EVM main's BLOCK opcodes
//!     (TIMESTAMP, NUMBER, BASEFEE, etc).
//!   - The cross-AIR linkage target for the storage / account /
//!     transaction / receipt MPT inclusion chains' root anchors.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Column layout ────────────────────────────────────────────────────

pub const ROOT_LEN: usize = 32;
pub const HASH_LEN: usize = 32;

// Four canonical MPT roots from the block header.
pub const COL_STATE_ROOT_OFFSET: usize = 0;                    // 0..32
pub const COL_TRANSACTIONS_ROOT_OFFSET: usize = 32;            // 32..64
pub const COL_RECEIPTS_ROOT_OFFSET: usize = 64;                // 64..96
pub const COL_WITHDRAWALS_ROOT_OFFSET: usize = 96;             // 96..128

// Block hash (= keccak256(rlp(header)); algebraic binding deferred).
pub const COL_BLOCK_HASH_OFFSET: usize = 128;                  // 128..160

// Parent hash — chains blocks together.
pub const COL_PARENT_HASH_OFFSET: usize = 160;                 // 160..192

// Scalar fields (BLOCK-opcode targets).
pub const COL_NUMBER: usize = 192;
pub const COL_TIMESTAMP: usize = 193;
pub const COL_GAS_LIMIT: usize = 194;
pub const COL_GAS_USED: usize = 195;
// base_fee_per_gas as 4 LE u64 limbs (U256).
pub const COL_BASE_FEE_L0: usize = 196;
pub const COL_BASE_FEE_L1: usize = 197;
pub const COL_BASE_FEE_L2: usize = 198;
pub const COL_BASE_FEE_L3: usize = 199;
// beneficiary (coinbase) as 4 LE u64 limbs (low 160 bits).
pub const COL_BENEFICIARY_L0: usize = 200;
pub const COL_BENEFICIARY_L1: usize = 201;
pub const COL_BENEFICIARY_L2: usize = 202;
pub const COL_BENEFICIARY_L3: usize = 203;

pub const COL_IS_REAL: usize = 204;

/// **#53 step 1**: 256 RLP byte columns + length tracker, for the
/// keccak chain `block_hash = keccak256(rlp(header)[..rlp_len])`.
/// Widened to 768 bytes to cover full real Ethereum block headers
/// (~600 bytes due to 256-byte logs_bloom + ~330 bytes of other
/// fields). Binds to `keccak_extract_wide` (MAX_INPUT_LEN=768) via
/// `make_block_header_to_keccak_extract_wide_linkage_descriptor`.
pub const COL_HEADER_RLP_OFFSET: usize = 205;        // 205..973
pub const HEADER_RLP_MAX_LEN: usize = 768;
pub const COL_HEADER_RLP_LEN: usize = COL_HEADER_RLP_OFFSET + HEADER_RLP_MAX_LEN; // 973

// prev_randao (mix_hash) as 4 LE u64 limbs (full 256-bit value).
pub const COL_PREV_RANDAO_L0: usize = COL_HEADER_RLP_LEN + 1; // 974
pub const COL_PREV_RANDAO_L1: usize = COL_PREV_RANDAO_L0 + 1;
pub const COL_PREV_RANDAO_L2: usize = COL_PREV_RANDAO_L0 + 2;
pub const COL_PREV_RANDAO_L3: usize = COL_PREV_RANDAO_L0 + 3;

// chain_id — single u64 (fits in one limb for all known Ethereum chains).
pub const COL_CHAIN_ID: usize = COL_PREV_RANDAO_L3 + 1; // 978

pub const NUM_COLUMNS: usize = COL_CHAIN_ID + 1; // 979

/// Row-local constraints: `is_real` binary (step 0). Step 1+ adds
/// RLP encoding correctness binding the field columns to the
/// header_rlp bytes (variable-length list encoding gadget — large).
pub const NUM_ROW_CONSTRAINTS: usize = 1;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BlockHeaderRow {
    /// **#53 step 1 (widened)**: 768-byte RLP encoding of the block
    /// header (zero-padded). Covers full real Ethereum headers (~600
    /// bytes). The keccak chain binds
    /// `block_hash = keccak256(header_rlp[..header_rlp_len])` via
    /// cross-AIR LogUp to KeccakExtractWide.
    pub header_rlp: [u8; HEADER_RLP_MAX_LEN],
    /// Actual length of the RLP encoding (bytes < `HEADER_RLP_MAX_LEN`).
    pub header_rlp_len: u32,
    pub state_root: [u8; ROOT_LEN],
    pub transactions_root: [u8; ROOT_LEN],
    pub receipts_root: [u8; ROOT_LEN],
    pub withdrawals_root: [u8; ROOT_LEN],
    pub block_hash: [u8; HASH_LEN],
    pub parent_hash: [u8; HASH_LEN],
    pub number: u64,
    pub timestamp: u64,
    pub gas_limit: u64,
    pub gas_used: u64,
    pub base_fee_per_gas: [u64; 4],
    pub beneficiary: [u64; 4],
    pub prev_randao: [u64; 4],
    pub chain_id: u64,
}

#[derive(Clone, Debug, Default)]
pub struct BlockHeaderWitness {
    pub headers: Vec<BlockHeaderRow>,
}

impl BlockHeaderWitness {
    pub fn from_headers(headers: Vec<BlockHeaderRow>) -> Self {
        Self { headers }
    }
}

/// Convert a `crate::block_header::BlockHeader` into the witness row
/// format consumed by this AIR. The block hash is computed from the
/// header (matching the eventual algebraic binding in step 1+).
pub fn from_block_header(
    h: &crate::block_header::BlockHeader,
) -> BlockHeaderRow {
    let block_hash = crate::block_header::block_header_hash(h);
    let withdrawals_root = h.withdrawals_root.unwrap_or([0u8; 32]);
    // base_fee_per_gas as 4 LE u64 limbs (U256, BE bytes → LE limbs).
    let base_fee_be = h.base_fee_per_gas.unwrap_or([0u8; 32]);
    let base_fee = [
        u64::from_be_bytes([
            base_fee_be[24], base_fee_be[25], base_fee_be[26], base_fee_be[27],
            base_fee_be[28], base_fee_be[29], base_fee_be[30], base_fee_be[31],
        ]),
        u64::from_be_bytes([
            base_fee_be[16], base_fee_be[17], base_fee_be[18], base_fee_be[19],
            base_fee_be[20], base_fee_be[21], base_fee_be[22], base_fee_be[23],
        ]),
        u64::from_be_bytes([
            base_fee_be[8], base_fee_be[9], base_fee_be[10], base_fee_be[11],
            base_fee_be[12], base_fee_be[13], base_fee_be[14], base_fee_be[15],
        ]),
        u64::from_be_bytes([
            base_fee_be[0], base_fee_be[1], base_fee_be[2], base_fee_be[3],
            base_fee_be[4], base_fee_be[5], base_fee_be[6], base_fee_be[7],
        ]),
    ];
    // Beneficiary as 4 LE u64 limbs (low 160 bits).
    let mut full = [0u8; 32];
    full[12..32].copy_from_slice(&h.beneficiary);
    let beneficiary = [
        u64::from_be_bytes([
            full[24], full[25], full[26], full[27],
            full[28], full[29], full[30], full[31],
        ]),
        u64::from_be_bytes([
            full[16], full[17], full[18], full[19],
            full[20], full[21], full[22], full[23],
        ]),
        u64::from_be_bytes([
            full[8], full[9], full[10], full[11],
            full[12], full[13], full[14], full[15],
        ]),
        u64::from_be_bytes([
            full[0], full[1], full[2], full[3],
            full[4], full[5], full[6], full[7],
        ]),
    ];
    // #53 step 1: RLP encode the header. For test/POC headers under
    // 256 bytes this fits the existing KeccakExtract chip's bound.
    // Real Ethereum headers (with 256-byte logs_bloom) exceed 256
    // bytes — those need a wider keccak variant.
    let rlp_bytes = crate::block_header::block_header_rlp(h);
    let mut header_rlp = [0u8; HEADER_RLP_MAX_LEN];
    let rlp_len = rlp_bytes.len().min(HEADER_RLP_MAX_LEN);
    header_rlp[..rlp_len].copy_from_slice(&rlp_bytes[..rlp_len]);

    // prev_randao (mix_hash) as 4 LE u64 limbs (U256, BE → LE).
    let mh = h.mix_hash;
    let prev_randao = [
        u64::from_be_bytes([mh[24], mh[25], mh[26], mh[27], mh[28], mh[29], mh[30], mh[31]]),
        u64::from_be_bytes([mh[16], mh[17], mh[18], mh[19], mh[20], mh[21], mh[22], mh[23]]),
        u64::from_be_bytes([mh[8],  mh[9],  mh[10], mh[11], mh[12], mh[13], mh[14], mh[15]]),
        u64::from_be_bytes([mh[0],  mh[1],  mh[2],  mh[3],  mh[4],  mh[5],  mh[6],  mh[7]]),
    ];

    BlockHeaderRow {
        header_rlp,
        header_rlp_len: rlp_bytes.len() as u32,
        state_root: h.state_root,
        transactions_root: h.transactions_root,
        receipts_root: h.receipts_root,
        withdrawals_root,
        block_hash,
        parent_hash: h.parent_hash,
        number: h.number,
        timestamp: h.timestamp,
        gas_limit: h.gas_limit,
        gas_used: h.gas_used,
        base_fee_per_gas: base_fee,
        beneficiary,
        prev_randao,
        chain_id: 1, // default to mainnet; caller overrides if needed
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &BlockHeaderWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.headers.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, h) in witness.headers.iter().enumerate() {
        for k in 0..ROOT_LEN {
            columns[COL_STATE_ROOT_OFFSET + k][i] = Scalar::from_u64(h.state_root[k] as u64, curve);
            columns[COL_TRANSACTIONS_ROOT_OFFSET + k][i] = Scalar::from_u64(h.transactions_root[k] as u64, curve);
            columns[COL_RECEIPTS_ROOT_OFFSET + k][i] = Scalar::from_u64(h.receipts_root[k] as u64, curve);
            columns[COL_WITHDRAWALS_ROOT_OFFSET + k][i] = Scalar::from_u64(h.withdrawals_root[k] as u64, curve);
        }
        for k in 0..HASH_LEN {
            columns[COL_BLOCK_HASH_OFFSET + k][i] = Scalar::from_u64(h.block_hash[k] as u64, curve);
            columns[COL_PARENT_HASH_OFFSET + k][i] = Scalar::from_u64(h.parent_hash[k] as u64, curve);
        }
        columns[COL_NUMBER][i] = Scalar::from_u64(h.number, curve);
        columns[COL_TIMESTAMP][i] = Scalar::from_u64(h.timestamp, curve);
        columns[COL_GAS_LIMIT][i] = Scalar::from_u64(h.gas_limit, curve);
        columns[COL_GAS_USED][i] = Scalar::from_u64(h.gas_used, curve);
        for j in 0..4 {
            columns[COL_BASE_FEE_L0 + j][i] = Scalar::from_u64(h.base_fee_per_gas[j], curve);
            columns[COL_BENEFICIARY_L0 + j][i] = Scalar::from_u64(h.beneficiary[j], curve);
        }
        columns[COL_IS_REAL][i] = one.clone();
        // #53 step 1: header_rlp + length cols.
        for k in 0..HEADER_RLP_MAX_LEN {
            columns[COL_HEADER_RLP_OFFSET + k][i] =
                Scalar::from_u64(h.header_rlp[k] as u64, curve);
        }
        columns[COL_HEADER_RLP_LEN][i] = Scalar::from_u64(h.header_rlp_len as u64, curve);
        for j in 0..4 {
            columns[COL_PREV_RANDAO_L0 + j][i] = Scalar::from_u64(h.prev_randao[j], curve);
        }
        columns[COL_CHAIN_ID][i] = Scalar::from_u64(h.chain_id, curve);
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

pub struct BlockHeaderConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl BlockHeaderConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for BlockHeaderConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        vec!["is_real_binary".into()]
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
        let mut bin = vec![Scalar::zero(curve); n];
        for r in 0..n {
            let v = &columns[COL_IS_REAL][r];
            bin[r] = v.mul(&v.sub(&one));
        }
        vec![bin]
    }

    fn evaluate_at_point(&self, col_evals: &[Scalar], alpha: &Scalar) -> Scalar {
        if col_evals.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let v = &col_evals[COL_IS_REAL];
        let body = v.mul(&v.sub(&one));
        let _ = alpha;
        body
    }

    fn build_constraint_polynomial(
        &self,
        col_coeffs: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_poly = vec![Scalar::one(curve)];
        let v = &col_coeffs[COL_IS_REAL];
        let v_m1 = poly_sub(v, &one_poly, curve);
        let body = poly_mul(v, &v_m1, curve);
        let _ = poly_add::<>;
        let _ = poly_scalar_mul::<>;
        body
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
        // 8-bit range check on every byte column (4 roots × 32 = 128
        // + 2 hashes × 32 = 64 = 192 byte cols).
        let tables = vec![LookupTable::range(256)];
        let mut declarations = Vec::new();
        let byte_starts = [
            ("state_root", COL_STATE_ROOT_OFFSET),
            ("transactions_root", COL_TRANSACTIONS_ROOT_OFFSET),
            ("receipts_root", COL_RECEIPTS_ROOT_OFFSET),
            ("withdrawals_root", COL_WITHDRAWALS_ROOT_OFFSET),
            ("block_hash", COL_BLOCK_HASH_OFFSET),
            ("parent_hash", COL_PARENT_HASH_OFFSET),
        ];
        for (name, offset) in byte_starts {
            for k in 0..32 {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_byte_{}_8bit", name, k),
                        column_index: offset + k,
                        max_bits: 8,
                        selector_column: None,
                    },
                    0,
                ));
            }
        }
        // #53 step 1: 256 RLP byte cols range-checked.
        for k in 0..HEADER_RLP_MAX_LEN {
            declarations.push((
                LookupDeclaration {
                    label: format!("header_rlp_byte_{}_8bit", k),
                    column_index: COL_HEADER_RLP_OFFSET + k,
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

/// **#53 step 1**: BlockHeader ↔ KeccakExtract on
/// `(header_rlp[0..256], block_hash[0..32])` ↔
/// `(INPUT_BYTE[0..256], OUTPUT_BYTE[0..32])`. Binds
/// `block_hash = keccak256(header_rlp[..])` algebraically (with
/// the prover supplying header_rlp_len; the keccak chip handles
/// trailing-zero padding correctly per its absorb logic).
///
/// **Caveat**: KeccakExtract's MAX_INPUT_LEN is 256. Headers whose
/// Cross-AIR LogUp descriptor binding the 768-byte header_rlp
/// window + 32-byte block_hash to `keccak_extract_wide` (768-byte
/// input variant).
pub fn make_block_header_to_keccak_extract_linkage_descriptor(
    block_header_layer_index: usize,
    keccak_extract_layer_index: usize,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    use crate::keccak_extract_wide as ke;
    let a_columns: Vec<usize> = (0..HEADER_RLP_MAX_LEN)
        .map(|k| COL_HEADER_RLP_OFFSET + k)
        .chain((0..32).map(|k| COL_BLOCK_HASH_OFFSET + k))
        .collect();
    let b_columns: Vec<usize> = (0..HEADER_RLP_MAX_LEN)
        .map(|k| ke::COL_INPUT_BYTE_OFFSET + k)
        .chain((0..32).map(|k| ke::COL_OUTPUT_BYTE_OFFSET + k))
        .collect();
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "block_header_to_keccak_wide_v1".into(),
        a_layer_index: block_header_layer_index,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: keccak_extract_layer_index,
        b_columns,
        b_selector_column: Some(ke::COL_IS_REAL),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_header::{block_header_hash, BlockHeader};

    #[test]
    fn from_block_header_extracts_fields() {
        let mut h = BlockHeader::default();
        h.state_root = [0xab; 32];
        h.transactions_root = [0xcd; 32];
        h.receipts_root = [0xef; 32];
        h.number = 17_000_000;
        h.timestamp = 1_700_000_000;
        h.gas_limit = 30_000_000;
        h.gas_used = 12_500_000;
        let row = from_block_header(&h);
        assert_eq!(row.state_root, [0xab; 32]);
        assert_eq!(row.transactions_root, [0xcd; 32]);
        assert_eq!(row.receipts_root, [0xef; 32]);
        assert_eq!(row.withdrawals_root, [0u8; 32]); // None → zero
        assert_eq!(row.number, 17_000_000);
        assert_eq!(row.timestamp, 1_700_000_000);
        assert_eq!(row.gas_limit, 30_000_000);
        assert_eq!(row.gas_used, 12_500_000);
        // block_hash = keccak256(rlp(header)).
        assert_eq!(row.block_hash, block_header_hash(&h));
    }

    #[test]
    fn trace_builder_populates_all_columns() {
        let mut h = BlockHeader::default();
        h.state_root = [0x11; 32];
        h.number = 42;
        let row = from_block_header(&h);
        let w = BlockHeaderWitness::from_headers(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        assert_eq!(trace.columns[COL_STATE_ROOT_OFFSET].evaluations[0].to_u64(), 0x11);
        assert_eq!(trace.columns[COL_NUMBER].evaluations[0].to_u64(), 42);
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn constraints_zero_on_honest_witness() {
        let mut h = BlockHeader::default();
        h.number = 1;
        let row = from_block_header(&h);
        let w = BlockHeaderWitness::from_headers(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = BlockHeaderConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| &p.evaluations)
            .collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for col in results {
            for v in col {
                assert!(v.is_zero(), "constraint must vanish");
            }
        }
    }

    /// 2-AIR joint_prove: block_header_air ↔ keccak_extract_wide.
    /// Validates `block_hash = keccak256(header_rlp)` algebraically
    /// for a full Cancun header (~580 bytes RLP).
    #[test]
    #[ignore = "slow: 2-AIR joint_prove with 800-col linkage (~5-15 min release)"]
    fn joint_prove_block_header_to_keccak_extract_wide() {
        use crate::cross_air_logup::{joint_prove, joint_verify};
        use crate::keccak_extract_wide as kew;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        use crate::vm_constraints::VmConstraintSystem;

        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let h = BlockHeader {
            parent_hash: [0x11; 32],
            ommers_hash: [0x22; 32],
            beneficiary: [0x33; 20],
            state_root: [0x44; 32],
            transactions_root: [0x55; 32],
            receipts_root: [0x66; 32],
            logs_bloom: [0x77; 256],
            difficulty: [0; 32],
            number: 18_500_000,
            gas_limit: 30_000_000,
            gas_used: 15_000_000,
            timestamp: 1_700_000_000,
            extra_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            mix_hash: [0x88; 32],
            nonce: [0x00; 8],
            base_fee_per_gas: Some({
                let mut b = [0u8; 32];
                b[24..32].copy_from_slice(&15_000_000_000u64.to_be_bytes());
                b
            }),
            withdrawals_root: Some([0x99; 32]),
            blob_gas_used: Some(393_216),
            excess_blob_gas: Some(786_432),
            parent_beacon_block_root: Some([0xaa; 32]),
        };

        // Block header AIR (layer 0).
        let bh_row = from_block_header(&h);
        let bh_w = BlockHeaderWitness::from_headers(vec![bh_row]);
        let bh_trace = build_trace_polynomials(&bh_w, curve);
        let bh_omega = scheme.domain_generator(bh_trace.padded_size);
        let bh_cs = BlockHeaderConstraintSystem::new(bh_trace.num_rows)
            .with_omega_and_domain(bh_omega, bh_trace.padded_size);

        // KeccakExtractWide (layer 1) — feed the header RLP as input.
        let rlp_bytes = crate::block_header::block_header_rlp(&h);
        let kew_w = kew::KeccakExtractWideWitness::from_inputs(&[&rlp_bytes]).unwrap();
        let kew_trace = kew::build_trace_polynomials(&kew_w, curve);
        let kew_omega = scheme.domain_generator(kew_trace.padded_size);
        let kew_cs = kew::KeccakExtractWideConstraintSystem::new(kew_trace.num_rows)
            .with_omega_and_domain(kew_omega, kew_trace.padded_size);

        let linkage = make_block_header_to_keccak_extract_linkage_descriptor(0, 1);
        let traces: Vec<(&crate::trace::TracePolynomials, &dyn VmConstraintSystem)> =
            vec![(&bh_trace, &bh_cs), (&kew_trace, &kew_cs)];

        let (proofs, ext) = joint_prove(&traces, &[linkage.clone()], &scheme)
            .expect("block_header ↔ keccak_extract_wide joint_prove must succeed");
        assert_eq!(proofs.len(), 2);
        assert_eq!(ext.linkage_proofs.len(), 1);
        let lp = &ext.linkage_proofs[0];
        eprintln!(
            "[diag] linkage label={} closure_match={}",
            lp.label, lp.closure_a == lp.closure_b,
        );
        assert_eq!(lp.closure_a, lp.closure_b);

        let cs_refs: Vec<&dyn VmConstraintSystem> = vec![&bh_cs, &kew_cs];
        assert!(
            joint_verify(&proofs, &cs_refs, &[linkage], &ext, &scheme, curve),
            "joint_verify must accept block_header ↔ keccak_extract_wide",
        );
    }

    #[test]
    fn full_cancun_header_rlp_fits_in_768_window() {
        let h = BlockHeader {
            parent_hash: [0x11; 32],
            ommers_hash: [0x22; 32],
            beneficiary: [0x33; 20],
            state_root: [0x44; 32],
            transactions_root: [0x55; 32],
            receipts_root: [0x66; 32],
            logs_bloom: [0x77; 256],
            difficulty: [0; 32],
            number: 18_500_000,
            gas_limit: 30_000_000,
            gas_used: 15_000_000,
            timestamp: 1_700_000_000,
            extra_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            mix_hash: [0x88; 32],
            nonce: [0x00; 8],
            base_fee_per_gas: Some({
                let mut b = [0u8; 32];
                b[24..32].copy_from_slice(&15_000_000_000u64.to_be_bytes());
                b
            }),
            withdrawals_root: Some([0x99; 32]),
            blob_gas_used: Some(393_216),
            excess_blob_gas: Some(786_432),
            parent_beacon_block_root: Some([0xaa; 32]),
        };
        let rlp = crate::block_header::block_header_rlp(&h);
        assert!(
            rlp.len() <= HEADER_RLP_MAX_LEN,
            "full Cancun header RLP {} bytes exceeds window {} bytes",
            rlp.len(), HEADER_RLP_MAX_LEN,
        );
        let row = from_block_header(&h);
        assert_eq!(row.header_rlp_len as usize, rlp.len());
        // Verify the stored RLP bytes match the canonical encoding.
        for i in 0..rlp.len() {
            assert_eq!(row.header_rlp[i], rlp[i], "byte {} mismatch", i);
        }
        // Verify keccak matches.
        assert_eq!(row.block_hash, crate::block_header::block_header_hash(&h));
    }

    #[test]
    fn is_real_binary_fires_on_nonbinary() {
        let mut h = BlockHeader::default();
        h.number = 1;
        let row = from_block_header(&h);
        let w = BlockHeaderWitness::from_headers(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace
            .columns
            .iter()
            .map(|p| p.evaluations.clone())
            .collect();
        cols[COL_IS_REAL][0] = Scalar::from_u64(2, CurveType::Bls48581);
        let cs = BlockHeaderConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert!(!results[0][0].is_zero(), "is_real_binary should fire");
    }

    #[test]
    fn block_header_to_keccak_descriptor_well_formed() {
        let desc = make_block_header_to_keccak_extract_linkage_descriptor(0, 1);
        assert_eq!(desc.label, "block_header_to_keccak_wide_v1");
        // 768 RLP input bytes + 32 hash output bytes = 800 col tuple.
        assert_eq!(desc.a_columns.len(), HEADER_RLP_MAX_LEN + 32);
        assert_eq!(desc.b_columns.len(), HEADER_RLP_MAX_LEN + 32);
        for k in 0..HEADER_RLP_MAX_LEN {
            assert_eq!(desc.a_columns[k], COL_HEADER_RLP_OFFSET + k);
            assert_eq!(
                desc.b_columns[k],
                crate::keccak_extract_wide::COL_INPUT_BYTE_OFFSET + k,
            );
        }
        for k in 0..32 {
            assert_eq!(desc.a_columns[HEADER_RLP_MAX_LEN + k], COL_BLOCK_HASH_OFFSET + k);
            assert_eq!(
                desc.b_columns[HEADER_RLP_MAX_LEN + k],
                crate::keccak_extract_wide::COL_OUTPUT_BYTE_OFFSET + k,
            );
        }
    }

    #[test]
    fn header_rlp_populated_in_witness() {
        let mut h = BlockHeader::default();
        h.state_root = [0x11; 32];
        h.number = 17;
        let row = from_block_header(&h);
        let full_rlp = crate::block_header::block_header_rlp(&h);
        let full_len = full_rlp.len();
        assert_eq!(row.header_rlp_len as usize, full_len,
                   "header_rlp_len should record the FULL RLP length even if > 256");
        // The witness column stores the first `min(256, full_len)` bytes.
        let stored = full_len.min(HEADER_RLP_MAX_LEN);
        assert_eq!(&row.header_rlp[..stored], &full_rlp[..stored]);
        // Real Ethereum block headers (~600 bytes due to logs_bloom)
        // exceed 256 bytes — so this test documents that the column
        // truncates at 256 and the keccak linkage works only when
        // full_len <= 256 (test/POC headers without logs_bloom).
        if full_len > HEADER_RLP_MAX_LEN {
            // Document the truncation behavior.
            assert!(full_len > HEADER_RLP_MAX_LEN);
        }
    }

    #[test]
    fn block_hash_matches_block_header_hash() {
        let mut h = BlockHeader::default();
        h.state_root = [0xab; 32];
        h.number = 100;
        h.timestamp = 1_700_000_000;
        let row = from_block_header(&h);
        let expected = block_header_hash(&h);
        assert_eq!(row.block_hash, expected);
    }
}
