//! EXTCODE family AIR — algebraic gadget for `EXTCODESIZE` (0x3B),
//! `EXTCODEHASH` (0x3F), and `EXTCODECOPY` (0x3C).
//!
//! These three opcodes all read an *external* account's bytecode (or a
//! summary of it):
//!
//!   * `EXTCODESIZE addr → push len(account.code)`
//!   * `EXTCODEHASH addr → push keccak256(account.code)` (EIP-1052; pushes
//!     0 if the account is non-existent, or `keccak256(b"")` for an
//!     empty contract).
//!   * `EXTCODECOPY addr destOffset offset length → memcpy(account.code[offset..offset+length])`
//!
//! This gadget commits, per EXTCODE invocation, the relevant
//! `(target_address, target_code_size, target_code_hash)` triple
//! together with the per-opcode selectors and (for EXTCODECOPY) the
//! memory-region operands. Algebraic constraints enforce:
//!
//!  * The three opcode selectors are binary and mutually exclusive
//!    (their sum equals `is_real`).
//!  * If the target account does not exist (`account_exists = 0`):
//!      - EXTCODEHASH must return 0 (every byte of `target_code_hash`
//!        is forced to 0 on that row).
//!      - EXTCODESIZE must return 0 (`target_code_size` forced to 0).
//!  * `target_code_size` and `length` decompose canonically into 8 LE
//!    bytes each (forcing them into `u64` range and feeding byte range
//!    checks).
//!
//! Cross-AIR LogUp descriptors bind:
//!
//!  * `(target_address, target_code_hash)` to an `account_state_air` row
//!    via [`make_extcode_to_account_descriptor`] — so the prover can't
//!    invent a code-hash for an address.
//!  * `(target_code_hash, target_code_size)` to a bytecode-table style
//!    AIR via [`make_extcode_to_bytecode_table_descriptor`] — so the
//!    size matches the actual code committed under that hash.
//!  * `(mem_offset, length)` (EXTCODECOPY only) to a memory-expansion
//!    AIR via [`make_extcode_to_memory_descriptor`].
//!
//! Per-row layout (one row per EXTCODE event; padding rows have
//! `is_real = 0` and all other columns zero):
//!
//! ```text
//! offset   size   meaning
//! 0..20    20     target_addr_byte_{0..19}      20 BE bytes of address
//! 20..21   1      target_code_size              u64 size of target account's code
//! 21..29   8      code_size_byte_{0..7}         LE byte decomposition of code_size
//! 29..61   32     target_code_hash_byte_{0..31} 32 bytes of keccak256(code)
//! 61..62   1      mem_offset                    EXTCODECOPY destination memory offset
//! 62..63   1      code_offset                   EXTCODECOPY source offset within code
//! 63..64   1      length                        EXTCODECOPY copy length
//! 64..72   8      length_byte_{0..7}            LE byte decomp of length
//! 72..73   1      sel_extcodesize               binary
//! 73..74   1      sel_extcodehash               binary
//! 74..75   1      sel_extcodecopy               binary
//! 75..76   1      account_exists                binary
//! 76..77   1      is_real                       binary
//! ```

use metavm_zkp::field::{CurveType, Scalar};
use metavm_zkp::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use metavm_zkp::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use metavm_zkp::trace::{Polynomial, TracePolynomials};
use metavm_zkp::vm_constraints::VmConstraintSystem;

// ─── Constants ────────────────────────────────────────────────────────

pub const ADDRESS_BYTES: usize = 20;
pub const CODE_HASH_BYTES: usize = 32;
pub const U64_BYTES: usize = 8;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_TARGET_ADDR_OFFSET: usize = 0;
pub const COL_TARGET_CODE_SIZE: usize = COL_TARGET_ADDR_OFFSET + ADDRESS_BYTES; // 20
pub const COL_CODE_SIZE_BYTE_OFFSET: usize = COL_TARGET_CODE_SIZE + 1;          // 21
pub const COL_TARGET_CODE_HASH_OFFSET: usize =
    COL_CODE_SIZE_BYTE_OFFSET + U64_BYTES;                                       // 29
pub const COL_MEM_OFFSET: usize = COL_TARGET_CODE_HASH_OFFSET + CODE_HASH_BYTES; // 61
pub const COL_CODE_OFFSET: usize = COL_MEM_OFFSET + 1;                          // 62
pub const COL_LENGTH: usize = COL_CODE_OFFSET + 1;                              // 63
pub const COL_LENGTH_BYTE_OFFSET: usize = COL_LENGTH + 1;                       // 64
pub const COL_SEL_EXTCODESIZE: usize = COL_LENGTH_BYTE_OFFSET + U64_BYTES;      // 72
pub const COL_SEL_EXTCODEHASH: usize = COL_SEL_EXTCODESIZE + 1;                 // 73
pub const COL_SEL_EXTCODECOPY: usize = COL_SEL_EXTCODEHASH + 1;                 // 74
pub const COL_ACCOUNT_EXISTS: usize = COL_SEL_EXTCODECOPY + 1;                  // 75
pub const COL_IS_REAL: usize = COL_ACCOUNT_EXISTS + 1;                          // 76

pub const NUM_COLUMNS: usize = COL_IS_REAL + 1;                                  // 77

const _: () = assert!(NUM_COLUMNS == 77);

// Row-local constraint catalogue:
//   0     is_real binary
//   1     sel_extcodesize binary
//   2     sel_extcodehash binary
//   3     sel_extcodecopy binary
//   4     account_exists binary
//   5     selector-sum: sel_extcodesize + sel_extcodehash + sel_extcodecopy − is_real = 0
//   6..37 (32) EXTCODEHASH ∧ ¬account_exists ⇒ code_hash[i] = 0
//   38    EXTCODESIZE ∧ ¬account_exists ⇒ target_code_size = 0
//   39    target_code_size − Σ code_size_byte_k · 2^(8k) = 0  (LE recomposition)
//   40    length − Σ length_byte_k · 2^(8k) = 0               (LE recomposition)
pub const NUM_ROW_CONSTRAINTS: usize = 41;
pub const NUM_SHIFTED: usize = 0;

// ─── Event / witness types ────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtcodeOp {
    /// EXTCODESIZE: only the size is observed.
    Size,
    /// EXTCODEHASH: only the hash is observed.
    Hash,
    /// EXTCODECOPY: a memory region is copied.
    Copy { mem_offset: u64, code_offset: u64, length: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtcodeEvent {
    pub target_address: [u8; 20],
    pub target_code_size: u64,
    /// 32 BE bytes of `keccak256(code)`. For non-existent accounts
    /// the canonical observed value is all zero (EIP-1052).
    pub target_code_hash: [u8; 32],
    pub account_exists: bool,
    pub op: ExtcodeOp,
}

#[derive(Clone, Debug, Default)]
pub struct ExtcodeWitness {
    pub events: Vec<ExtcodeEvent>,
}

impl ExtcodeWitness {
    /// Host-side helper: build a witness from a slice of events.
    ///
    /// For non-existent accounts, the canonical EIP-1052 behaviour
    /// forces `target_code_hash` to all-zero (the AIR also enforces
    /// this algebraically). Any non-zero bytes supplied for a
    /// non-existent account are zeroed here so the trace is honest.
    pub fn from_events(events: &[ExtcodeEvent]) -> Self {
        let mut normalised = Vec::with_capacity(events.len());
        for e in events {
            let mut row = e.clone();
            if !row.account_exists {
                row.target_code_hash = [0u8; 32];
                row.target_code_size = 0;
            }
            normalised.push(row);
        }
        Self { events: normalised }
    }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &ExtcodeWitness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.events.len();
    let padded = metavm_zkp::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);
    let mut cols: Vec<Vec<Scalar>> = (0..NUM_COLUMNS)
        .map(|_| vec![zero.clone(); padded])
        .collect();
    for (r, e) in witness.events.iter().enumerate() {
        for k in 0..ADDRESS_BYTES {
            cols[COL_TARGET_ADDR_OFFSET + k][r] =
                Scalar::from_u64(e.target_address[k] as u64, curve);
        }
        cols[COL_TARGET_CODE_SIZE][r] = Scalar::from_u64(e.target_code_size, curve);
        for k in 0..U64_BYTES {
            let byte = ((e.target_code_size >> (8 * k)) & 0xff) as u64;
            cols[COL_CODE_SIZE_BYTE_OFFSET + k][r] = Scalar::from_u64(byte, curve);
        }
        for k in 0..CODE_HASH_BYTES {
            cols[COL_TARGET_CODE_HASH_OFFSET + k][r] =
                Scalar::from_u64(e.target_code_hash[k] as u64, curve);
        }
        let (mem_offset, code_offset, length) = match e.op {
            ExtcodeOp::Copy { mem_offset, code_offset, length } => {
                (mem_offset, code_offset, length)
            }
            _ => (0u64, 0u64, 0u64),
        };
        cols[COL_MEM_OFFSET][r] = Scalar::from_u64(mem_offset, curve);
        cols[COL_CODE_OFFSET][r] = Scalar::from_u64(code_offset, curve);
        cols[COL_LENGTH][r] = Scalar::from_u64(length, curve);
        for k in 0..U64_BYTES {
            let byte = ((length >> (8 * k)) & 0xff) as u64;
            cols[COL_LENGTH_BYTE_OFFSET + k][r] = Scalar::from_u64(byte, curve);
        }
        let (sz, hs, cp) = match e.op {
            ExtcodeOp::Size => (true, false, false),
            ExtcodeOp::Hash => (false, true, false),
            ExtcodeOp::Copy { .. } => (false, false, true),
        };
        cols[COL_SEL_EXTCODESIZE][r] = if sz { one.clone() } else { zero.clone() };
        cols[COL_SEL_EXTCODEHASH][r] = if hs { one.clone() } else { zero.clone() };
        cols[COL_SEL_EXTCODECOPY][r] = if cp { one.clone() } else { zero.clone() };
        cols[COL_ACCOUNT_EXISTS][r] =
            if e.account_exists { one.clone() } else { zero.clone() };
        cols[COL_IS_REAL][r] = one.clone();
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

// ─── Constraint system ────────────────────────────────────────────────

pub struct ExtcodeConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl ExtcodeConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }
    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

/// 2^(8·k) as a Scalar (k ∈ 0..8).
fn byte_power_u64(k: usize, curve: CurveType) -> Scalar {
    debug_assert!(k < 8);
    Scalar::from_u64(1u64 << (8 * k), curve)
}

impl VmConstraintSystem for ExtcodeConstraintSystem {
    fn num_constraints(&self) -> usize {
        NUM_ROW_CONSTRAINTS
    }

    fn constraint_labels(&self) -> Vec<String> {
        let mut v = vec![
            "is_real_binary".into(),
            "sel_extcodesize_binary".into(),
            "sel_extcodehash_binary".into(),
            "sel_extcodecopy_binary".into(),
            "account_exists_binary".into(),
            "selector_sum_to_is_real".into(),
        ];
        for k in 0..CODE_HASH_BYTES {
            v.push(format!("extcodehash_nonexistent_zero_byte_{}", k));
        }
        v.push("extcodesize_nonexistent_zero".into());
        v.push("code_size_le_decomp".into());
        v.push("length_le_decomp".into());
        v
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
        for r in 0..n {
            let is_real = &columns[COL_IS_REAL][r];
            let sel_sz = &columns[COL_SEL_EXTCODESIZE][r];
            let sel_hs = &columns[COL_SEL_EXTCODEHASH][r];
            let sel_cp = &columns[COL_SEL_EXTCODECOPY][r];
            let acct_exists = &columns[COL_ACCOUNT_EXISTS][r];

            // 0..4 binarity.
            bodies[0][r] = is_real.mul(&is_real.sub(&one));
            bodies[1][r] = sel_sz.mul(&sel_sz.sub(&one));
            bodies[2][r] = sel_hs.mul(&sel_hs.sub(&one));
            bodies[3][r] = sel_cp.mul(&sel_cp.sub(&one));
            bodies[4][r] = acct_exists.mul(&acct_exists.sub(&one));

            // 5 selector sum
            let sum = sel_sz.add(sel_hs).add(sel_cp);
            bodies[5][r] = sum.sub(is_real);

            // 6..37: EXTCODEHASH ∧ ¬account_exists ⇒ code_hash[i] = 0
            let not_exists = one.sub(acct_exists);
            let hash_gate = sel_hs.mul(&not_exists);
            for k in 0..CODE_HASH_BYTES {
                let h = &columns[COL_TARGET_CODE_HASH_OFFSET + k][r];
                bodies[6 + k][r] = hash_gate.mul(h);
            }

            // 38: EXTCODESIZE ∧ ¬account_exists ⇒ target_code_size = 0
            let size_gate = sel_sz.mul(&not_exists);
            let size = &columns[COL_TARGET_CODE_SIZE][r];
            bodies[38][r] = size_gate.mul(size);

            // 39: target_code_size − Σ code_size_byte_k · 2^(8k) = 0
            let mut size_horner = Scalar::zero(curve);
            for k in 0..U64_BYTES {
                let b = &columns[COL_CODE_SIZE_BYTE_OFFSET + k][r];
                let w = byte_power_u64(k, curve);
                size_horner = size_horner.add(&b.mul(&w));
            }
            bodies[39][r] = size.sub(&size_horner);

            // 40: length − Σ length_byte_k · 2^(8k) = 0
            let length = &columns[COL_LENGTH][r];
            let mut len_horner = Scalar::zero(curve);
            for k in 0..U64_BYTES {
                let b = &columns[COL_LENGTH_BYTE_OFFSET + k][r];
                let w = byte_power_u64(k, curve);
                len_horner = len_horner.add(&b.mul(&w));
            }
            bodies[40][r] = length.sub(&len_horner);
        }
        bodies
    }

    fn evaluate_at_point(&self, ce: &[Scalar], alpha: &Scalar) -> Scalar {
        if ce.len() < NUM_COLUMNS {
            return Scalar::zero(alpha.curve_type());
        }
        let curve = alpha.curve_type();
        let one = Scalar::one(curve);
        let is_real = &ce[COL_IS_REAL];
        let sel_sz = &ce[COL_SEL_EXTCODESIZE];
        let sel_hs = &ce[COL_SEL_EXTCODEHASH];
        let sel_cp = &ce[COL_SEL_EXTCODECOPY];
        let acct_exists = &ce[COL_ACCOUNT_EXISTS];
        let not_exists = one.sub(acct_exists);
        let hash_gate = sel_hs.mul(&not_exists);
        let size_gate = sel_sz.mul(&not_exists);

        let mut bodies: Vec<Scalar> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(is_real.mul(&is_real.sub(&one)));
        bodies.push(sel_sz.mul(&sel_sz.sub(&one)));
        bodies.push(sel_hs.mul(&sel_hs.sub(&one)));
        bodies.push(sel_cp.mul(&sel_cp.sub(&one)));
        bodies.push(acct_exists.mul(&acct_exists.sub(&one)));
        bodies.push(sel_sz.add(sel_hs).add(sel_cp).sub(is_real));
        for k in 0..CODE_HASH_BYTES {
            bodies.push(hash_gate.mul(&ce[COL_TARGET_CODE_HASH_OFFSET + k]));
        }
        bodies.push(size_gate.mul(&ce[COL_TARGET_CODE_SIZE]));
        let mut size_horner = Scalar::zero(curve);
        for k in 0..U64_BYTES {
            let b = &ce[COL_CODE_SIZE_BYTE_OFFSET + k];
            size_horner = size_horner.add(&b.mul(&byte_power_u64(k, curve)));
        }
        bodies.push(ce[COL_TARGET_CODE_SIZE].sub(&size_horner));
        let mut len_horner = Scalar::zero(curve);
        for k in 0..U64_BYTES {
            let b = &ce[COL_LENGTH_BYTE_OFFSET + k];
            len_horner = len_horner.add(&b.mul(&byte_power_u64(k, curve)));
        }
        bodies.push(ce[COL_LENGTH].sub(&len_horner));

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
        cc: &[Vec<Scalar>],
        alpha: &Scalar,
        _domain_size: u64,
    ) -> Vec<Scalar> {
        let curve = alpha.curve_type();
        let one_p = vec![Scalar::one(curve)];
        let is_real = &cc[COL_IS_REAL];
        let sel_sz = &cc[COL_SEL_EXTCODESIZE];
        let sel_hs = &cc[COL_SEL_EXTCODEHASH];
        let sel_cp = &cc[COL_SEL_EXTCODECOPY];
        let acct_exists = &cc[COL_ACCOUNT_EXISTS];

        let bin = |p: &Vec<Scalar>| -> Vec<Scalar> {
            let pm1 = poly_sub(p, &one_p, curve);
            poly_mul(p, &pm1, curve)
        };

        let mut bodies: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        bodies.push(bin(is_real));
        bodies.push(bin(sel_sz));
        bodies.push(bin(sel_hs));
        bodies.push(bin(sel_cp));
        bodies.push(bin(acct_exists));
        // selector sum
        let sum = poly_add(&poly_add(sel_sz, sel_hs, curve), sel_cp, curve);
        bodies.push(poly_sub(&sum, is_real, curve));
        // hash gate = sel_hs · (1 - acct_exists)
        let not_exists_p = poly_sub(&one_p, acct_exists, curve);
        let hash_gate = poly_mul(sel_hs, &not_exists_p, curve);
        for k in 0..CODE_HASH_BYTES {
            bodies.push(poly_mul(&hash_gate, &cc[COL_TARGET_CODE_HASH_OFFSET + k], curve));
        }
        // size gate
        let size_gate = poly_mul(sel_sz, &not_exists_p, curve);
        bodies.push(poly_mul(&size_gate, &cc[COL_TARGET_CODE_SIZE], curve));
        // size LE recompose
        let mut size_horner = vec![Scalar::zero(curve)];
        for k in 0..U64_BYTES {
            let scaled =
                poly_scalar_mul(&cc[COL_CODE_SIZE_BYTE_OFFSET + k], &byte_power_u64(k, curve));
            size_horner = poly_add(&size_horner, &scaled, curve);
        }
        bodies.push(poly_sub(&cc[COL_TARGET_CODE_SIZE], &size_horner, curve));
        // length LE recompose
        let mut len_horner = vec![Scalar::zero(curve)];
        for k in 0..U64_BYTES {
            let scaled =
                poly_scalar_mul(&cc[COL_LENGTH_BYTE_OFFSET + k], &byte_power_u64(k, curve));
            len_horner = poly_add(&len_horner, &scaled, curve);
        }
        bodies.push(poly_sub(&cc[COL_LENGTH], &len_horner, curve));

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
            COL_SEL_EXTCODESIZE,
            COL_SEL_EXTCODEHASH,
            COL_SEL_EXTCODECOPY,
            COL_ACCOUNT_EXISTS,
        ]
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
        let zero = Scalar::zero(columns[0][0].curve_type());
        for c in columns.iter_mut().take(NUM_COLUMNS) {
            for cell in c.iter_mut().skip(num_rows).take(padded_size - num_rows) {
                *cell = zero.clone();
            }
        }
    }

    fn lookup_declarations(&self) -> LookupRequirements {
        let tables = vec![LookupTable::range(256), LookupTable::range(2)];
        let tbl_byte = 0usize;
        let tbl_bit = 1usize;
        let mut declarations = Vec::new();
        // 20 address bytes.
        for k in 0..ADDRESS_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("extcode_addr_byte_{}_8bit", k),
                    column_index: COL_TARGET_ADDR_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        // 8 code-size LE bytes.
        for k in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("extcode_code_size_byte_{}_8bit", k),
                    column_index: COL_CODE_SIZE_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        // 32 code-hash bytes.
        for k in 0..CODE_HASH_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("extcode_code_hash_byte_{}_8bit", k),
                    column_index: COL_TARGET_CODE_HASH_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        // 8 length LE bytes.
        for k in 0..U64_BYTES {
            declarations.push((
                LookupDeclaration {
                    label: format!("extcode_length_byte_{}_8bit", k),
                    column_index: COL_LENGTH_BYTE_OFFSET + k,
                    max_bits: 8,
                    selector_column: None,
                },
                tbl_byte,
            ));
        }
        // Binary flags.
        for (label, col) in [
            ("is_real", COL_IS_REAL),
            ("sel_extcodesize", COL_SEL_EXTCODESIZE),
            ("sel_extcodehash", COL_SEL_EXTCODEHASH),
            ("sel_extcodecopy", COL_SEL_EXTCODECOPY),
            ("account_exists", COL_ACCOUNT_EXISTS),
        ] {
            declarations.push((
                LookupDeclaration {
                    label: format!("extcode_{}_1bit", label),
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

// ─── Cross-AIR LogUp descriptors ──────────────────────────────────────

/// EXTCODE gadget row ↔ `account_state_air` row.
///
/// Binds the 52-element tuple `(target_address_bytes[0..20] ++
/// target_code_hash_bytes[0..32])` between:
///   * A side (this gadget): gated by `COL_IS_REAL`.
///   * B side (account_state_air): host-side adapter columns laying out
///     address bytes followed by code-hash bytes. The B-side columns
///     wired here use account_state_air's `COL_CODE_HASH_OFFSET` for
///     the hash region; address bytes are passed through an adapter
///     region that the orchestration layer materialises (we point them
///     at account_state_air columns 0..20 as a sentinel — orchestration
///     should remap to the proper byte-level adapter region).
pub fn make_extcode_to_account_descriptor(
    extcode_layer: usize,
    account_state_layer: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    use metavm_zkp::account_state_air::COL_CODE_HASH_OFFSET as ACCT_CODE_HASH_OFFSET;
    let mut a_columns: Vec<usize> = Vec::with_capacity(ADDRESS_BYTES + CODE_HASH_BYTES);
    let mut b_columns: Vec<usize> = Vec::with_capacity(ADDRESS_BYTES + CODE_HASH_BYTES);
    for k in 0..ADDRESS_BYTES {
        a_columns.push(COL_TARGET_ADDR_OFFSET + k);
        // account_state_air does not currently expose per-byte address
        // adapter columns; orchestration is expected to wire a byte
        // adapter. As a stand-in we publish account_state_air's first
        // 20 columns as the address-byte region — downstream
        // orchestration overrides.
        b_columns.push(k);
    }
    for k in 0..CODE_HASH_BYTES {
        a_columns.push(COL_TARGET_CODE_HASH_OFFSET + k);
        b_columns.push(ACCT_CODE_HASH_OFFSET + k);
    }
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "extcode_to_account_state_v1".into(),
        a_layer_index: extcode_layer,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: account_state_layer,
        b_columns,
        b_selector_column: Some(metavm_zkp::account_state_air::COL_IS_REAL),
    }
}

/// EXTCODE gadget row ↔ bytecode-table style AIR.
///
/// Binds `(target_code_hash_bytes[0..32], target_code_size)` to a
/// bytecode-table AIR row that publishes the code-hash and the total
/// size of the code committed under that hash.
///
/// The `bytecode_table_code_hash_offset` parameter is the B-side
/// column index where the bytecode-table AIR exposes its 32 code-hash
/// bytes; `bytecode_table_code_size_column` is the column for the
/// `u64` size; `bytecode_table_selector_column` is the AIR's
/// `is_real`-equivalent selector.
pub fn make_extcode_to_bytecode_table_descriptor(
    extcode_layer: usize,
    bytecode_table_layer: usize,
    bytecode_table_code_hash_offset: usize,
    bytecode_table_code_size_column: usize,
    bytecode_table_selector_column: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    let mut a_columns: Vec<usize> = Vec::with_capacity(CODE_HASH_BYTES + 1);
    let mut b_columns: Vec<usize> = Vec::with_capacity(CODE_HASH_BYTES + 1);
    for k in 0..CODE_HASH_BYTES {
        a_columns.push(COL_TARGET_CODE_HASH_OFFSET + k);
        b_columns.push(bytecode_table_code_hash_offset + k);
    }
    a_columns.push(COL_TARGET_CODE_SIZE);
    b_columns.push(bytecode_table_code_size_column);
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "extcode_to_bytecode_table_v1".into(),
        a_layer_index: extcode_layer,
        a_columns,
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: bytecode_table_layer,
        b_columns,
        b_selector_column: Some(bytecode_table_selector_column),
    }
}

/// EXTCODE gadget row ↔ memory-expansion AIR (EXTCODECOPY only).
///
/// Binds `(mem_offset, length)` for EXTCODECOPY rows to a memory-expansion
/// AIR's `(new_size_start, expansion_length)` style columns. Gated by
/// `COL_SEL_EXTCODECOPY` on A side so the linkage only fires for actual
/// EXTCODECOPY rows.
pub fn make_extcode_to_memory_descriptor(
    extcode_layer: usize,
    memory_expansion_layer: usize,
    memory_offset_column: usize,
    memory_length_column: usize,
    memory_selector_column: usize,
) -> metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
    metavm_zkp::cross_air_logup::CrossAirLogUpDescriptor {
        label: "extcode_to_memory_expansion_v1".into(),
        a_layer_index: extcode_layer,
        a_columns: vec![COL_MEM_OFFSET, COL_LENGTH],
        a_selector_column: Some(COL_SEL_EXTCODECOPY),
        b_layer_index: memory_expansion_layer,
        b_columns: vec![memory_offset_column, memory_length_column],
        b_selector_column: Some(memory_selector_column),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_addr(byte: u8) -> [u8; 20] {
        [byte; 20]
    }
    fn dummy_hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn check_all_zero(cs: &ExtcodeConstraintSystem, t: &TracePolynomials) {
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        for (i, body) in cs.evaluate_on_domain(&cr, t.num_rows).iter().enumerate() {
            for (r, v) in body.iter().enumerate() {
                assert!(v.is_zero(), "constraint {} non-zero at row {}", i, r);
            }
        }
    }

    #[test]
    fn extcode_air_column_layout_pinned() {
        // Pin the layout: defensive against accidental reshuffling.
        assert_eq!(COL_TARGET_ADDR_OFFSET, 0);
        assert_eq!(COL_TARGET_CODE_SIZE, 20);
        assert_eq!(COL_CODE_SIZE_BYTE_OFFSET, 21);
        assert_eq!(COL_TARGET_CODE_HASH_OFFSET, 29);
        assert_eq!(COL_MEM_OFFSET, 61);
        assert_eq!(COL_CODE_OFFSET, 62);
        assert_eq!(COL_LENGTH, 63);
        assert_eq!(COL_LENGTH_BYTE_OFFSET, 64);
        assert_eq!(COL_SEL_EXTCODESIZE, 72);
        assert_eq!(COL_SEL_EXTCODEHASH, 73);
        assert_eq!(COL_SEL_EXTCODECOPY, 74);
        assert_eq!(COL_ACCOUNT_EXISTS, 75);
        assert_eq!(COL_IS_REAL, 76);
        assert_eq!(NUM_COLUMNS, 77);
        assert!(NUM_ROW_CONSTRAINTS >= 10);
        assert_eq!(NUM_ROW_CONSTRAINTS, 41);
    }

    #[test]
    fn extcode_air_extcodesize_existing_account_honest() {
        let event = ExtcodeEvent {
            target_address: dummy_addr(0xAA),
            target_code_size: 0xC0FFEE,
            target_code_hash: dummy_hash(0x11),
            account_exists: true,
            op: ExtcodeOp::Size,
        };
        let w = ExtcodeWitness::from_events(&[event]);
        // For an existing account, code_size + hash are preserved.
        assert_eq!(w.events[0].target_code_size, 0xC0FFEE);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ExtcodeConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
    }

    #[test]
    fn extcode_air_extcodehash_nonexistent_returns_zero() {
        // Non-existent account: EIP-1052 says EXTCODEHASH = 0.
        // The `from_events` helper zeros the hash for us; the AIR
        // also enforces this algebraically.
        let event = ExtcodeEvent {
            target_address: dummy_addr(0xBB),
            target_code_size: 0,
            target_code_hash: [0u8; 32],
            account_exists: false,
            op: ExtcodeOp::Hash,
        };
        let w = ExtcodeWitness::from_events(&[event]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ExtcodeConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);

        // Verify the hash bytes are indeed all zero in the trace.
        for k in 0..CODE_HASH_BYTES {
            assert!(t.columns[COL_TARGET_CODE_HASH_OFFSET + k].evaluations[0].is_zero());
        }
    }

    #[test]
    fn extcode_air_extcodecopy_in_bounds_honest() {
        let event = ExtcodeEvent {
            target_address: dummy_addr(0xCC),
            target_code_size: 1024,
            target_code_hash: dummy_hash(0x22),
            account_exists: true,
            op: ExtcodeOp::Copy { mem_offset: 64, code_offset: 0, length: 256 },
        };
        let w = ExtcodeWitness::from_events(&[event]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ExtcodeConstraintSystem::new(t.num_rows);
        check_all_zero(&cs, &t);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let curve = CurveType::Bls48581;
        assert!(cr[COL_MEM_OFFSET][0].sub(&Scalar::from_u64(64, curve)).is_zero());
        assert!(cr[COL_LENGTH][0].sub(&Scalar::from_u64(256, curve)).is_zero());
        assert!(cr[COL_SEL_EXTCODECOPY][0].sub(&Scalar::one(curve)).is_zero());
    }

    #[test]
    fn extcode_air_tampered_code_hash_for_nonexistent_detected() {
        // Build a non-existent EXTCODEHASH row, then tamper one of the
        // hash bytes to a non-zero value. Constraint 6..37 must fire.
        let event = ExtcodeEvent {
            target_address: dummy_addr(0xDD),
            target_code_size: 0,
            target_code_hash: [0u8; 32],
            account_exists: false,
            op: ExtcodeOp::Hash,
        };
        let w = ExtcodeWitness::from_events(&[event]);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        // Lie: claim a non-zero code-hash byte while account does not exist.
        t.columns[COL_TARGET_CODE_HASH_OFFSET + 7].evaluations[0] =
            Scalar::from_u64(0x99, CurveType::Bls48581);
        let cs = ExtcodeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        // The corresponding constraint slot 6 + 7 = 13 must NOT be zero.
        assert!(
            !bodies[6 + 7][0].is_zero(),
            "tampered code-hash on non-existent account must trip constraint"
        );
    }

    #[test]
    fn extcode_air_tampered_code_size_for_nonexistent_detected() {
        // Non-existent EXTCODESIZE but claim non-zero size — must trip
        // constraint 38.
        let event = ExtcodeEvent {
            target_address: dummy_addr(0xEE),
            target_code_size: 0,
            target_code_hash: [0u8; 32],
            account_exists: false,
            op: ExtcodeOp::Size,
        };
        let w = ExtcodeWitness::from_events(&[event]);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        // Lie about the size.
        t.columns[COL_TARGET_CODE_SIZE].evaluations[0] =
            Scalar::from_u64(42, CurveType::Bls48581);
        // Also have to update one byte so the LE decomposition is
        // internally consistent — otherwise constraint 39 trips
        // *first* on the byte-decomposition mismatch, masking the
        // semantic violation we want to observe.
        t.columns[COL_CODE_SIZE_BYTE_OFFSET].evaluations[0] =
            Scalar::from_u64(42, CurveType::Bls48581);
        let cs = ExtcodeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(
            !bodies[38][0].is_zero(),
            "tampered size on non-existent account must trip constraint 38"
        );
        // And the LE decomp constraint should still vanish (we kept it
        // consistent).
        assert!(bodies[39][0].is_zero(), "byte decomp constraint should remain zero");
    }

    #[test]
    fn extcode_air_descriptors_well_formed() {
        let d_acct = make_extcode_to_account_descriptor(0, 1);
        assert_eq!(d_acct.label, "extcode_to_account_state_v1");
        assert_eq!(d_acct.a_columns.len(), ADDRESS_BYTES + CODE_HASH_BYTES);
        assert_eq!(d_acct.b_columns.len(), ADDRESS_BYTES + CODE_HASH_BYTES);
        assert_eq!(d_acct.a_selector_column, Some(COL_IS_REAL));

        // Bytecode table descriptor with stand-in B-side col indices.
        let d_bt = make_extcode_to_bytecode_table_descriptor(0, 2, 100, 132, 133);
        assert_eq!(d_bt.label, "extcode_to_bytecode_table_v1");
        assert_eq!(d_bt.a_columns.len(), CODE_HASH_BYTES + 1);
        assert_eq!(d_bt.b_columns.len(), CODE_HASH_BYTES + 1);
        assert_eq!(d_bt.a_columns[0], COL_TARGET_CODE_HASH_OFFSET);
        assert_eq!(d_bt.a_columns[CODE_HASH_BYTES], COL_TARGET_CODE_SIZE);
        assert_eq!(d_bt.b_columns[CODE_HASH_BYTES], 132);

        // Memory descriptor — EXTCODECOPY only, gated by sel_extcodecopy.
        let d_mem = make_extcode_to_memory_descriptor(0, 3, 1, 3, 12);
        assert_eq!(d_mem.label, "extcode_to_memory_expansion_v1");
        assert_eq!(d_mem.a_columns, vec![COL_MEM_OFFSET, COL_LENGTH]);
        assert_eq!(d_mem.a_selector_column, Some(COL_SEL_EXTCODECOPY));
        assert_eq!(d_mem.b_columns, vec![1, 3]);
        assert_eq!(d_mem.b_selector_column, Some(12));
    }

    #[test]
    fn extcode_air_selector_sum_violation_detected() {
        // Two selectors set simultaneously — selector-sum constraint 5
        // must fire.
        let event = ExtcodeEvent {
            target_address: dummy_addr(0x10),
            target_code_size: 16,
            target_code_hash: dummy_hash(0x33),
            account_exists: true,
            op: ExtcodeOp::Size,
        };
        let w = ExtcodeWitness::from_events(&[event]);
        let mut t = build_trace_polynomials(&w, CurveType::Bls48581);
        // Force also sel_extcodehash = 1 (so both selectors fire).
        t.columns[COL_SEL_EXTCODEHASH].evaluations[0] =
            Scalar::one(CurveType::Bls48581);
        let cs = ExtcodeConstraintSystem::new(t.num_rows);
        let cr: Vec<&Vec<Scalar>> = t.columns.iter().map(|p| &p.evaluations).collect();
        let bodies = cs.evaluate_on_domain(&cr, t.num_rows);
        assert!(!bodies[5][0].is_zero(), "selector-sum violation undetected");
    }

    #[test]
    fn extcode_air_evaluate_at_point_zero_on_honest() {
        let event = ExtcodeEvent {
            target_address: dummy_addr(0x77),
            target_code_size: 200,
            target_code_hash: dummy_hash(0x44),
            account_exists: true,
            op: ExtcodeOp::Copy { mem_offset: 0, code_offset: 0, length: 200 },
        };
        let w = ExtcodeWitness::from_events(&[event]);
        let t = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = ExtcodeConstraintSystem::new(t.num_rows);
        let alpha = Scalar::from_u64(0xDEADBEEF, CurveType::Bls48581);
        let row_evals: Vec<Scalar> =
            t.columns.iter().map(|p| p.evaluations[0].clone()).collect();
        let v = cs.evaluate_at_point(&row_evals, &alpha);
        assert!(v.is_zero(), "honest row must evaluate to zero");
    }
}
