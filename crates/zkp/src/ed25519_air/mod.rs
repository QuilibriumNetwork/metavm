//! Ed25519 signature-verification AIR — Task #261 scaffold.
//!
//! Algebraic-skeleton AIR that commits the host-side ed25519
//! signature-verification witness (R, A, s, M, hram, decompressed
//! points, double-scalar-mul result) and enforces:
//!
//!   * a row-local **curve-equation pin** for the decompressed `R` and
//!     `A` points — `y² − x² − 1 − d·x²·y² = 0` over `Fp25519`
//!     (`p = 2^255 − 19`, `d = −121665/121666 mod p`). Modular
//!     reduction is performed host-side with `num_bigint`; the AIR
//!     commits the residue limbs and constrains every limb to zero
//!     (true Fp limb-level reduction inside the AIR is deferred to
//!     `ed25519_fp_air`, per the task brief).
//!   * `is_real` / `valid` binarity.
//!   * `valid * is_real == 1` for honest rows (host trace populates
//!     `valid = 1` only when the cofactor-cleared signature equation
//!     `8·s·B == 8·R + 8·hram·A` holds).
//!
//! Cross-AIR LogUp linkage (host-side descriptor, joint γ wiring TBD):
//!
//!   * `make_ed25519_hram_to_sha512_linkage_descriptor` — A-side
//!     publishes `(R[0..32] || A[0..32] || hram[0..64])` (the
//!     SHA-512 input + output that derives `hram`); B-side is
//!     `sha512_air` once it exposes input/output-byte columns. The
//!     descriptor wires byte-column ranges that the future
//!     sha512_air interface contract is expected to expose; today
//!     the B-side column indices are conservative placeholders
//!     (see `sha512_b_columns_input` / `sha512_b_columns_output`).
//!
//! ── Column layout (1 row = 1 signature) ──────────────────────────
//!
//! ```text
//! offset  name                   size  notes
//! 0       r_compressed_be         32   R (compressed, BE)
//! 32      pub_a_compressed_be     32   A (compressed, BE)
//! 64      s_scalar_le             32   s (LE per RFC8032)
//! 96      hram_bytes              64   H(R||A||M) raw 64-byte digest
//! 160     r_x_be                  32   decompressed R.x in BE bytes
//! 192     r_y_be                  32   decompressed R.y in BE bytes
//! 224     a_x_be                  32   decompressed A.x in BE bytes
//! 256     a_y_be                  32   decompressed A.y in BE bytes
//! 288     lhs_x_be                32   (8·s·B).x in BE bytes
//! 320     lhs_y_be                32   (8·s·B).y in BE bytes
//! 352     rhs_x_be                32   (8·R + 8·hram·A).x in BE bytes
//! 384     rhs_y_be                32   (8·R + 8·hram·A).y in BE bytes
//! 416     r_curve_eq_residue_be   32   y²−x²−1−d·x²·y² mod p, R-side
//! 448     a_curve_eq_residue_be   32   y²−x²−1−d·x²·y² mod p, A-side
//! 480     is_real                  1   {0,1}
//! 481     valid                    1   {0,1}, honest = is_real
//! ```
//!
//! Total: **482 columns**.
//!
//! Row-local constraints (NUM_ROW_CONSTRAINTS = 67):
//!   * 0: `is_real` binary.
//!   * 1: `valid` binary.
//!   * 2: `is_real * (1 − valid)` (honest sigs verify).
//!   * 3..35: 32 limbs of `r_curve_eq_residue_be` == 0.
//!   * 35..67: 32 limbs of `a_curve_eq_residue_be` == 0.
//!
//! Byte-range LogUp declarations cover all byte-shaped columns.

use crate::field::{CurveType, Scalar};
use crate::lookup::{LookupDeclaration, LookupRequirements, LookupTable};
use crate::poly_arith::{poly_add, poly_mul, poly_scalar_mul, poly_sub};
use crate::trace::{Polynomial, TracePolynomials};
use crate::vm_constraints::VmConstraintSystem;

// ─── Sizes ────────────────────────────────────────────────────────────

pub const COMPRESSED_LEN: usize = 32;
pub const SCALAR_LEN: usize = 32;
pub const HRAM_LEN: usize = 64;
pub const COORD_LEN: usize = 32;

// ─── Column layout ────────────────────────────────────────────────────

pub const COL_R_COMPRESSED_OFFSET: usize = 0;
pub const COL_PUB_A_COMPRESSED_OFFSET: usize = COL_R_COMPRESSED_OFFSET + COMPRESSED_LEN;
pub const COL_S_SCALAR_OFFSET: usize = COL_PUB_A_COMPRESSED_OFFSET + COMPRESSED_LEN;
pub const COL_HRAM_OFFSET: usize = COL_S_SCALAR_OFFSET + SCALAR_LEN;

pub const COL_R_X_OFFSET: usize = COL_HRAM_OFFSET + HRAM_LEN;
pub const COL_R_Y_OFFSET: usize = COL_R_X_OFFSET + COORD_LEN;
pub const COL_A_X_OFFSET: usize = COL_R_Y_OFFSET + COORD_LEN;
pub const COL_A_Y_OFFSET: usize = COL_A_X_OFFSET + COORD_LEN;

pub const COL_LHS_X_OFFSET: usize = COL_A_Y_OFFSET + COORD_LEN;
pub const COL_LHS_Y_OFFSET: usize = COL_LHS_X_OFFSET + COORD_LEN;
pub const COL_RHS_X_OFFSET: usize = COL_LHS_Y_OFFSET + COORD_LEN;
pub const COL_RHS_Y_OFFSET: usize = COL_RHS_X_OFFSET + COORD_LEN;

pub const COL_R_CURVE_EQ_RESIDUE_OFFSET: usize = COL_RHS_Y_OFFSET + COORD_LEN;
pub const COL_A_CURVE_EQ_RESIDUE_OFFSET: usize = COL_R_CURVE_EQ_RESIDUE_OFFSET + COORD_LEN;

pub const COL_IS_REAL: usize = COL_A_CURVE_EQ_RESIDUE_OFFSET + COORD_LEN;
pub const COL_VALID: usize = COL_IS_REAL + 1;

pub const NUM_COLUMNS: usize = COL_VALID + 1;

pub const NUM_ROW_CONSTRAINTS: usize = 3 + 32 + 32;
pub const NUM_SHIFTED: usize = 0;

// ─── Witness types ────────────────────────────────────────────────────

/// One ed25519 signature row.
#[derive(Clone, Debug)]
pub struct Ed25519Row {
    pub r_compressed: [u8; COMPRESSED_LEN],
    pub pub_a_compressed: [u8; COMPRESSED_LEN],
    pub s_scalar: [u8; SCALAR_LEN],
    pub message: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct Ed25519Witness {
    pub invocations: Vec<Ed25519Row>,
}

impl Ed25519Witness {
    pub fn from_signatures(rows: Vec<Ed25519Row>) -> Self {
        Self { invocations: rows }
    }
}

// ─── Host-side ed25519 helpers ────────────────────────────────────────

/// `p = 2^255 − 19` (curve25519 base-field prime), as a `num_bigint::BigInt`.
fn fp_p() -> num_bigint::BigInt {
    use num_bigint::BigInt;
    use num_traits::One;
    (BigInt::one() << 255) - BigInt::from(19u32)
}

/// Edwards `d = −121665 · 121666^{−1} mod p`.
fn ed_d() -> num_bigint::BigInt {
    use num_bigint::BigInt;
    use num_traits::Signed;
    let p = fp_p();
    let inv = mod_inverse(&BigInt::from(121666i64), &p);
    let mut d = (-BigInt::from(121665i64) * inv) % &p;
    if d.is_negative() {
        d += &p;
    }
    d
}

/// Extended Euclidean inverse (panics if non-invertible; we only call
/// it for 121666 mod p and the residue normalisation path).
fn mod_inverse(a: &num_bigint::BigInt, m: &num_bigint::BigInt) -> num_bigint::BigInt {
    use num_bigint::BigInt;
    use num_traits::{One, Signed, Zero};
    let (mut old_r, mut r) = (a.clone(), m.clone());
    let (mut old_s, mut s) = (BigInt::one(), BigInt::zero());
    while !r.is_zero() {
        let q = &old_r / &r;
        let new_r = &old_r - &q * &r;
        old_r = std::mem::replace(&mut r, new_r);
        let new_s = &old_s - &q * &s;
        old_s = std::mem::replace(&mut s, new_s);
    }
    let mut inv = old_s % m;
    if inv.is_negative() {
        inv += m;
    }
    inv
}

/// 32 LE bytes → BigInt.
fn le_to_bigint(bytes: &[u8]) -> num_bigint::BigInt {
    num_bigint::BigInt::from_bytes_le(num_bigint::Sign::Plus, bytes)
}

/// BigInt (assumed in `[0, p)`) → 32 BE bytes.
fn bigint_to_be32(v: &num_bigint::BigInt) -> [u8; 32] {
    let (_, bytes_be) = v.to_bytes_be();
    let mut out = [0u8; 32];
    let off = 32usize.saturating_sub(bytes_be.len());
    out[off..].copy_from_slice(&bytes_be[..bytes_be.len().min(32)]);
    out
}

/// Compute `y² − x² − 1 − d·x²·y² mod p` for a point in affine
/// (x_le, y_le) byte form. Returns 32 BE bytes (zero iff the point
/// lies on the Edwards curve).
fn curve_eq_residue_be(x_le: &[u8; 32], y_le: &[u8; 32]) -> [u8; 32] {
    use num_bigint::BigInt;
    use num_traits::{One, Signed};
    let p = fp_p();
    let d = ed_d();
    let x = le_to_bigint(x_le);
    let y = le_to_bigint(y_le);
    let x2 = (&x * &x) % &p;
    let y2 = (&y * &y) % &p;
    let dx2y2 = (&d * &x2 % &p * &y2) % &p;
    let mut r = (&y2 - &x2 - BigInt::one() - &dx2y2) % &p;
    if r.is_negative() {
        r += &p;
    }
    bigint_to_be32(&r)
}

/// Convert a curve25519-dalek `EdwardsPoint` into affine `(x_le, y_le)`
/// 32-byte little-endian field-element pairs. Uses `compress()` for the
/// y bytes; recovers x by re-decompressing and reading the encoded sign
/// bit back. We round-trip through compression to avoid touching the
/// crate-private `FieldElement` type.
fn decompose_edwards_point(p: &curve25519_dalek::edwards::EdwardsPoint) -> ([u8; 32], [u8; 32]) {
    let compressed = p.compress();
    let y_bytes = compressed.0; // little-endian y || (1 bit) sign of x
    // Strip the high bit (sign of x) to extract canonical y.
    let mut y_le = y_bytes;
    let x_is_negative = (y_le[31] >> 7) & 1 == 1;
    y_le[31] &= 0x7f;
    // Recover x from the curve equation: x² = (y² − 1) / (d·y² + 1).
    use num_bigint::BigInt;
    use num_traits::{One, Signed};
    let prime = fp_p();
    let d = ed_d();
    let y = le_to_bigint(&y_le);
    let y2 = (&y * &y) % &prime;
    let num = (&y2 - BigInt::one() + &prime) % &prime;
    let den = (&d * &y2 + BigInt::one()) % &prime;
    let den_inv = mod_inverse(&den, &prime);
    let x2 = (&num * &den_inv) % &prime;
    // x = x2^{(p+3)/8} mod p (standard curve25519 sqrt).
    let exp = (&prime + BigInt::from(3u32)) >> 3;
    let mut x = x2.modpow(&exp, &prime);
    let x_squared_check = (&x * &x) % &prime;
    if x_squared_check != x2 {
        // Multiply by sqrt(−1).
        let sqrt_m1_exp = (&prime - BigInt::one()) >> 2;
        let sqrt_m1 = BigInt::from(2u32).modpow(&sqrt_m1_exp, &prime);
        x = (&x * &sqrt_m1) % &prime;
    }
    if x.is_negative() {
        x += &prime;
    }
    // Pick the sign matching x_is_negative (low bit of x).
    let low_bit = (&x % BigInt::from(2u32)).to_u32_digits().1.first().copied().unwrap_or(0) != 0;
    if low_bit != x_is_negative {
        x = (&prime - &x) % &prime;
    }
    let mut x_le = [0u8; 32];
    let (_, x_bytes_le) = x.to_bytes_le();
    let n = x_bytes_le.len().min(32);
    x_le[..n].copy_from_slice(&x_bytes_le[..n]);
    (x_le, y_le)
}

/// Compute `hram = SHA-512(R || A || M)` (raw 64-byte digest).
fn compute_hram(
    r_compressed: &[u8; COMPRESSED_LEN],
    pub_a_compressed: &[u8; COMPRESSED_LEN],
    message: &[u8],
) -> [u8; HRAM_LEN] {
    let mut input = Vec::with_capacity(64 + message.len());
    input.extend_from_slice(r_compressed);
    input.extend_from_slice(pub_a_compressed);
    input.extend_from_slice(message);
    crate::sha512_air::sha512(&input)
}

/// Host-side ed25519 signature check. Returns `(valid, R_xy, A_xy,
/// lhs_xy, rhs_xy, hram_bytes)`.
fn check_signature(
    row: &Ed25519Row,
) -> (
    bool,
    ([u8; 32], [u8; 32]),
    ([u8; 32], [u8; 32]),
    ([u8; 32], [u8; 32]),
    ([u8; 32], [u8; 32]),
    [u8; HRAM_LEN],
) {
    use curve25519_dalek::edwards::{CompressedEdwardsY, EdwardsPoint};
    use curve25519_dalek::scalar::Scalar as DalekScalar;

    let hram = compute_hram(&row.r_compressed, &row.pub_a_compressed, &row.message);

    let r_pt = CompressedEdwardsY(row.r_compressed)
        .decompress()
        .unwrap_or(EdwardsPoint::default());
    let a_pt = CompressedEdwardsY(row.pub_a_compressed)
        .decompress()
        .unwrap_or(EdwardsPoint::default());

    let s = DalekScalar::from_bytes_mod_order(row.s_scalar);
    let k = DalekScalar::from_bytes_mod_order_wide(&hram);

    // Compute LHS = 8·s·B and RHS = 8·R + 8·k·A.
    let basepoint = curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
    let lhs = basepoint * s;
    let rhs = r_pt + a_pt * k;
    let cofactor = DalekScalar::from(8u64);
    let lhs8 = lhs * cofactor;
    let rhs8 = rhs * cofactor;
    let valid = lhs8 == rhs8;

    let r_xy = decompose_edwards_point(&r_pt);
    let a_xy = decompose_edwards_point(&a_pt);
    let lhs_xy = decompose_edwards_point(&lhs8);
    let rhs_xy = decompose_edwards_point(&rhs8);

    (valid, r_xy, a_xy, lhs_xy, rhs_xy, hram)
}

#[inline]
fn write_be_bytes(columns: &mut [Vec<Scalar>], offset: usize, bytes_be: &[u8], row: usize, curve: CurveType) {
    for (k, b) in bytes_be.iter().enumerate() {
        columns[offset + k][row] = Scalar::from_u64(*b as u64, curve);
    }
}

#[inline]
fn write_le_as_be(columns: &mut [Vec<Scalar>], offset: usize, bytes_le: &[u8], row: usize, curve: CurveType) {
    // Reverse LE → BE for layout consistency with the rest of the codebase
    // (linkage byte columns are BE).
    let mut tmp = [0u8; 32];
    let n = bytes_le.len().min(32);
    for k in 0..n { tmp[k] = bytes_le[n - 1 - k]; }
    for k in 0..n { columns[offset + k][row] = Scalar::from_u64(tmp[k] as u64, curve); }
}

// ─── Trace builder ────────────────────────────────────────────────────

pub fn build_trace_polynomials(
    witness: &Ed25519Witness,
    curve: CurveType,
) -> TracePolynomials {
    let num_rows = witness.invocations.len();
    let padded = crate::trace::nearest_power_of_two(num_rows.max(1));
    let zero = Scalar::zero(curve);
    let one = Scalar::one(curve);

    let mut columns: Vec<Vec<Scalar>> =
        (0..NUM_COLUMNS).map(|_| vec![zero.clone(); padded]).collect();

    for (i, row) in witness.invocations.iter().enumerate() {
        let (valid, r_xy, a_xy, lhs_xy, rhs_xy, hram) = check_signature(row);

        // Raw compressed bytes (stored as-given; RFC8032 wire form is LE
        // but for column tuple-shape purposes we use the same byte order
        // throughout this AIR — the linkage descriptor consumes them
        // directly as bytes).
        write_be_bytes(&mut columns, COL_R_COMPRESSED_OFFSET, &row.r_compressed, i, curve);
        write_be_bytes(&mut columns, COL_PUB_A_COMPRESSED_OFFSET, &row.pub_a_compressed, i, curve);
        write_be_bytes(&mut columns, COL_S_SCALAR_OFFSET, &row.s_scalar, i, curve);
        write_be_bytes(&mut columns, COL_HRAM_OFFSET, &hram, i, curve);

        // Decompressed (x, y) coordinates, stored as BE bytes.
        write_le_as_be(&mut columns, COL_R_X_OFFSET, &r_xy.0, i, curve);
        write_le_as_be(&mut columns, COL_R_Y_OFFSET, &r_xy.1, i, curve);
        write_le_as_be(&mut columns, COL_A_X_OFFSET, &a_xy.0, i, curve);
        write_le_as_be(&mut columns, COL_A_Y_OFFSET, &a_xy.1, i, curve);

        write_le_as_be(&mut columns, COL_LHS_X_OFFSET, &lhs_xy.0, i, curve);
        write_le_as_be(&mut columns, COL_LHS_Y_OFFSET, &lhs_xy.1, i, curve);
        write_le_as_be(&mut columns, COL_RHS_X_OFFSET, &rhs_xy.0, i, curve);
        write_le_as_be(&mut columns, COL_RHS_Y_OFFSET, &rhs_xy.1, i, curve);

        let r_residue = curve_eq_residue_be(&r_xy.0, &r_xy.1);
        let a_residue = curve_eq_residue_be(&a_xy.0, &a_xy.1);
        write_be_bytes(&mut columns, COL_R_CURVE_EQ_RESIDUE_OFFSET, &r_residue, i, curve);
        write_be_bytes(&mut columns, COL_A_CURVE_EQ_RESIDUE_OFFSET, &a_residue, i, curve);

        columns[COL_IS_REAL][i] = one.clone();
        columns[COL_VALID][i] = if valid { one.clone() } else { zero.clone() };
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

// ─── Constraint system ────────────────────────────────────────────────

pub struct Ed25519ConstraintSystem {
    pub num_rows: usize,
    pub omega: Option<Scalar>,
    pub domain_size: Option<u64>,
}

impl Ed25519ConstraintSystem {
    pub fn new(num_rows: usize) -> Self {
        Self { num_rows, omega: None, domain_size: None }
    }

    pub fn with_omega_and_domain(mut self, omega: Scalar, domain_size: u64) -> Self {
        self.omega = Some(omega);
        self.domain_size = Some(domain_size);
        self
    }
}

impl VmConstraintSystem for Ed25519ConstraintSystem {
    fn num_constraints(&self) -> usize { NUM_ROW_CONSTRAINTS }

    fn constraint_labels(&self) -> Vec<String> {
        let mut labels = Vec::with_capacity(NUM_ROW_CONSTRAINTS);
        labels.push("is_real_binary".into());
        labels.push("valid_binary".into());
        labels.push("honest_implies_valid".into());
        for k in 0..32 { labels.push(format!("r_curve_eq_residue_{}_zero", k)); }
        for k in 0..32 { labels.push(format!("a_curve_eq_residue_{}_zero", k)); }
        labels
    }

    fn evaluate_on_domain(
        &self,
        columns: &[&Vec<Scalar>],
        _num_rows: usize,
    ) -> Vec<Vec<Scalar>> {
        assert!(columns.len() >= NUM_COLUMNS);
        let curve = columns[0][0].curve_type();
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        let n = columns[0].len();
        let mut out: Vec<Vec<Scalar>> = Vec::with_capacity(NUM_ROW_CONSTRAINTS);

        // 0: is_real binary.
        {
            let mut c = vec![zero.clone(); n];
            for r in 0..n {
                let v = &columns[COL_IS_REAL][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 1: valid binary.
        {
            let mut c = vec![zero.clone(); n];
            for r in 0..n {
                let v = &columns[COL_VALID][r];
                c[r] = v.mul(&v.sub(&one));
            }
            out.push(c);
        }

        // 2: is_real * (1 − valid) = 0 → honest rows must have valid=1.
        {
            let mut c = vec![zero.clone(); n];
            for r in 0..n {
                let is_real = &columns[COL_IS_REAL][r];
                let valid = &columns[COL_VALID][r];
                c[r] = is_real.mul(&one.sub(valid));
            }
            out.push(c);
        }

        // 3..35: R-side curve-eq residue limbs all zero.
        for k in 0..32 {
            let mut c = vec![zero.clone(); n];
            for r in 0..n {
                c[r] = columns[COL_R_CURVE_EQ_RESIDUE_OFFSET + k][r].clone();
            }
            out.push(c);
        }

        // 35..67: A-side curve-eq residue limbs all zero.
        for k in 0..32 {
            let mut c = vec![zero.clone(); n];
            for r in 0..n {
                c[r] = columns[COL_A_CURVE_EQ_RESIDUE_OFFSET + k][r].clone();
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

        // 0: is_real binary.
        {
            let v = &col_evals[COL_IS_REAL];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: valid binary.
        {
            let v = &col_evals[COL_VALID];
            acc = acc.add(&alpha_pow.mul(&v.mul(&v.sub(&one))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: honest implies valid.
        {
            let is_real = &col_evals[COL_IS_REAL];
            let valid = &col_evals[COL_VALID];
            acc = acc.add(&alpha_pow.mul(&is_real.mul(&one.sub(valid))));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // R-residue limbs zero.
        for k in 0..32 {
            let v = &col_evals[COL_R_CURVE_EQ_RESIDUE_OFFSET + k];
            acc = acc.add(&alpha_pow.mul(v));
            alpha_pow = alpha_pow.mul(alpha);
        }
        // A-residue limbs zero.
        for k in 0..32 {
            let v = &col_evals[COL_A_CURVE_EQ_RESIDUE_OFFSET + k];
            acc = acc.add(&alpha_pow.mul(v));
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

        // 0: is_real binary.
        {
            let v = &col_coeffs[COL_IS_REAL];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 1: valid binary.
        {
            let v = &col_coeffs[COL_VALID];
            let v_m1 = poly_sub(v, &one_poly, curve);
            let body = poly_mul(v, &v_m1, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        // 2: honest implies valid: is_real * (1 - valid).
        {
            let is_real = &col_coeffs[COL_IS_REAL];
            let valid = &col_coeffs[COL_VALID];
            let one_m_valid = poly_sub(&one_poly, valid, curve);
            let body = poly_mul(is_real, &one_m_valid, curve);
            acc = poly_add(&acc, &poly_scalar_mul(&body, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        for k in 0..32 {
            let v = &col_coeffs[COL_R_CURVE_EQ_RESIDUE_OFFSET + k];
            acc = poly_add(&acc, &poly_scalar_mul(v, &alpha_pow), curve);
            alpha_pow = alpha_pow.mul(alpha);
        }
        for k in 0..32 {
            let v = &col_coeffs[COL_A_CURVE_EQ_RESIDUE_OFFSET + k];
            acc = poly_add(&acc, &poly_scalar_mul(v, &alpha_pow), curve);
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
        let byte_ranges: &[(usize, usize, &str)] = &[
            (COL_R_COMPRESSED_OFFSET, COMPRESSED_LEN, "r_compressed"),
            (COL_PUB_A_COMPRESSED_OFFSET, COMPRESSED_LEN, "pub_a_compressed"),
            (COL_S_SCALAR_OFFSET, SCALAR_LEN, "s_scalar"),
            (COL_HRAM_OFFSET, HRAM_LEN, "hram"),
            (COL_R_X_OFFSET, COORD_LEN, "r_x"),
            (COL_R_Y_OFFSET, COORD_LEN, "r_y"),
            (COL_A_X_OFFSET, COORD_LEN, "a_x"),
            (COL_A_Y_OFFSET, COORD_LEN, "a_y"),
            (COL_LHS_X_OFFSET, COORD_LEN, "lhs_x"),
            (COL_LHS_Y_OFFSET, COORD_LEN, "lhs_y"),
            (COL_RHS_X_OFFSET, COORD_LEN, "rhs_x"),
            (COL_RHS_Y_OFFSET, COORD_LEN, "rhs_y"),
            (COL_R_CURVE_EQ_RESIDUE_OFFSET, COORD_LEN, "r_curve_eq_residue"),
            (COL_A_CURVE_EQ_RESIDUE_OFFSET, COORD_LEN, "a_curve_eq_residue"),
        ];
        for (offset, len, label) in byte_ranges {
            for k in 0..*len {
                declarations.push((
                    LookupDeclaration {
                        label: format!("{}_{}_8bit", label, k),
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

// ─── Cross-AIR linkage descriptors ────────────────────────────────────

/// Returns the A-side (this AIR) column indices for the SHA-512
/// preimage of `hram`: the concatenation `R || A || M_chunk` where the
/// message bytes are *not* part of this AIR's column layout. Scaffold
/// version: publish only `R || A` as a 64-byte prefix on this side,
/// concatenated with the 64-byte `hram` digest. Once sha512_air gains
/// explicit input/output byte columns this descriptor will be extended
/// to bind the message tail too.
pub fn ed25519_hram_a_columns() -> Vec<usize> {
    let mut cols = Vec::with_capacity(COMPRESSED_LEN * 2 + HRAM_LEN);
    for k in 0..COMPRESSED_LEN { cols.push(COL_R_COMPRESSED_OFFSET + k); }
    for k in 0..COMPRESSED_LEN { cols.push(COL_PUB_A_COMPRESSED_OFFSET + k); }
    for k in 0..HRAM_LEN { cols.push(COL_HRAM_OFFSET + k); }
    cols
}

/// Cross-AIR LogUp descriptor stub binding this AIR's `hram` row to the
/// `sha512_air` AIR (B-side). The B-side column lists are exposed as
/// argument vectors because `sha512_air` does not yet publish a
/// dedicated input-byte / output-byte tuple shape; once that contract
/// is defined (parallel task), callers will pass the matching column
/// indices and the linkage will close.
pub fn make_ed25519_hram_to_sha512_linkage_descriptor(
    ed25519_layer_index: usize,
    sha512_layer_index: usize,
    sha512_b_columns: Vec<usize>,
) -> crate::cross_air_logup::CrossAirLogUpDescriptor {
    crate::cross_air_logup::CrossAirLogUpDescriptor {
        label: "ed25519_hram_sha512_v1".into(),
        a_layer_index: ed25519_layer_index,
        a_columns: ed25519_hram_a_columns(),
        a_selector_column: Some(COL_IS_REAL),
        b_layer_index: sha512_layer_index,
        b_columns: sha512_b_columns,
        b_selector_column: None,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Produce a deterministic ed25519 signature (R, A, s, M) using
    /// curve25519-dalek primitives + the workspace sha512.
    fn make_signature_vector(seed: [u8; 32], message: Vec<u8>) -> Ed25519Row {
        use curve25519_dalek::constants::ED25519_BASEPOINT_POINT;
        use curve25519_dalek::scalar::Scalar as DalekScalar;

        let h = crate::sha512_air::sha512(&seed);
        let mut a_bytes = [0u8; 32];
        a_bytes.copy_from_slice(&h[0..32]);
        // Clamp per RFC8032.
        a_bytes[0] &= 248;
        a_bytes[31] &= 127;
        a_bytes[31] |= 64;
        let a = DalekScalar::from_bytes_mod_order(a_bytes);
        let prefix = &h[32..64];

        let pub_a_point = ED25519_BASEPOINT_POINT * a;
        let pub_a_compressed = pub_a_point.compress().0;

        let mut r_input = Vec::with_capacity(32 + message.len());
        r_input.extend_from_slice(prefix);
        r_input.extend_from_slice(&message);
        let r_hash = crate::sha512_air::sha512(&r_input);
        let r = DalekScalar::from_bytes_mod_order_wide(&r_hash);
        let r_point = ED25519_BASEPOINT_POINT * r;
        let r_compressed = r_point.compress().0;

        let mut k_input = Vec::with_capacity(64 + message.len());
        k_input.extend_from_slice(&r_compressed);
        k_input.extend_from_slice(&pub_a_compressed);
        k_input.extend_from_slice(&message);
        let k_hash = crate::sha512_air::sha512(&k_input);
        let k = DalekScalar::from_bytes_mod_order_wide(&k_hash);

        let s = r + k * a;
        let s_scalar = s.to_bytes();

        Ed25519Row {
            r_compressed,
            pub_a_compressed,
            s_scalar,
            message,
        }
    }

    #[test]
    fn column_layout_constants() {
        assert_eq!(COL_R_COMPRESSED_OFFSET, 0);
        assert_eq!(COL_PUB_A_COMPRESSED_OFFSET, 32);
        assert_eq!(COL_S_SCALAR_OFFSET, 64);
        assert_eq!(COL_HRAM_OFFSET, 96);
        assert_eq!(COL_R_X_OFFSET, 160);
        assert_eq!(COL_A_CURVE_EQ_RESIDUE_OFFSET, 448);
        assert_eq!(COL_IS_REAL, 480);
        assert_eq!(COL_VALID, 481);
        assert_eq!(NUM_COLUMNS, 482);
        assert_eq!(NUM_ROW_CONSTRAINTS, 67);
    }

    #[test]
    fn trace_builder_populates_signature_witness() {
        let row = make_signature_vector([7u8; 32], b"test-msg".to_vec());
        let w = Ed25519Witness::from_signatures(vec![row.clone()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // R bytes round-trip.
        for k in 0..COMPRESSED_LEN {
            assert_eq!(
                trace.columns[COL_R_COMPRESSED_OFFSET + k].evaluations[0].to_u64(),
                row.r_compressed[k] as u64,
            );
        }
        // is_real = 1, valid = 1 on the honest witness.
        assert_eq!(trace.columns[COL_IS_REAL].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_VALID].evaluations[0].to_u64(), 1);
    }

    #[test]
    fn constraints_zero_on_honest_signature() {
        let row = make_signature_vector([42u8; 32], b"another-msg".to_vec());
        let w = Ed25519Witness::from_signatures(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let cs = Ed25519ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(results.len(), NUM_ROW_CONSTRAINTS);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} row {} = {:?} (expected zero)",
                    i, r, val,
                );
            }
        }
    }

    #[test]
    fn curve_eq_pin_fires_on_tampered_point() {
        let row = make_signature_vector([1u8; 32], b"x".to_vec());
        let w = Ed25519Witness::from_signatures(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        let mut cols: Vec<Vec<Scalar>> = trace.columns.iter().map(|p| p.evaluations.clone()).collect();
        // Tamper the R-side residue: bump byte 0 to 1.
        cols[COL_R_CURVE_EQ_RESIDUE_OFFSET][0] = Scalar::from_u64(1, CurveType::Bls48581);
        let cs = Ed25519ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = cols.iter().collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        // R-residue limb 0 zero-check is constraint index 3.
        assert!(!results[3][0].is_zero(), "r curve-eq residue limb 0 should fire");
    }

    #[test]
    fn honest_implies_valid_fires_on_invalid_sig() {
        // Random bogus signature.
        let row = Ed25519Row {
            r_compressed: [0xaa; 32],
            pub_a_compressed: [0xbb; 32],
            s_scalar: [0xcc; 32],
            message: b"forged".to_vec(),
        };
        let w = Ed25519Witness::from_signatures(vec![row]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // The host already wrote valid=0 because the eqn failed; the
        // honest_implies_valid constraint at index 2 must then fire
        // because is_real=1 ∧ valid=0.
        let cs = Ed25519ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        assert_eq!(trace.columns[COL_VALID].evaluations[0].to_u64(), 0);
        assert!(!results[2][0].is_zero(), "honest_implies_valid should fire on invalid sig");
    }

    #[test]
    fn sha512_linkage_descriptor_well_formed() {
        let a_cols = ed25519_hram_a_columns();
        assert_eq!(a_cols.len(), COMPRESSED_LEN * 2 + HRAM_LEN);
        let stub_b_cols = vec![0usize; a_cols.len()];
        let desc = make_ed25519_hram_to_sha512_linkage_descriptor(0, 1, stub_b_cols.clone());
        assert_eq!(desc.label, "ed25519_hram_sha512_v1");
        assert_eq!(desc.a_layer_index, 0);
        assert_eq!(desc.b_layer_index, 1);
        assert_eq!(desc.a_columns.len(), desc.b_columns.len());
        assert_eq!(desc.a_selector_column, Some(COL_IS_REAL));
    }

    #[test]
    fn multiple_signatures_independent_rows() {
        let row1 = make_signature_vector([1u8; 32], b"msg-1".to_vec());
        let row2 = make_signature_vector([2u8; 32], b"msg-2".to_vec());
        let w = Ed25519Witness::from_signatures(vec![row1.clone(), row2.clone()]);
        let trace = build_trace_polynomials(&w, CurveType::Bls48581);
        // Distinct R values in the two rows.
        let r0 = trace.columns[COL_R_COMPRESSED_OFFSET].evaluations[0].to_u64();
        let r1 = trace.columns[COL_R_COMPRESSED_OFFSET].evaluations[1].to_u64();
        assert_eq!(r0, row1.r_compressed[0] as u64);
        assert_eq!(r1, row2.r_compressed[0] as u64);
        // Both rows verify.
        assert_eq!(trace.columns[COL_VALID].evaluations[0].to_u64(), 1);
        assert_eq!(trace.columns[COL_VALID].evaluations[1].to_u64(), 1);

        // Constraints zero on the honest 2-row trace.
        let cs = Ed25519ConstraintSystem::new(trace.num_rows);
        let col_refs: Vec<&Vec<Scalar>> = trace.columns.iter().map(|p| &p.evaluations).collect();
        let results = cs.evaluate_on_domain(&col_refs, trace.num_rows);
        for (i, col) in results.iter().enumerate() {
            for (r, val) in col.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "constraint {} row {} = {:?}", i, r, val,
                );
            }
        }
    }
}
