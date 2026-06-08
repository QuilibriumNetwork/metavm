//! Compile Fp2 operations into sequences of [`FpOp`]s.
//!
//! The non-native Fp AIR ([`crate::nonnative_fp_air`]) handles a single row
//! per Fp add / sub / mul. Fp2 = Fp[u] / (u² + 1) elements have two Fp
//! coordinates and their operations decompose into a handful of Fp ops:
//!
//! - `Fp2::add((a0,a1), (b0,b1))` → `(a0+b0, a1+b1)` — 2 Fp adds
//! - `Fp2::sub((a0,a1), (b0,b1))` → `(a0-b0, a1-b1)` — 2 Fp subs
//! - `Fp2::mul((a0,a1), (b0,b1))` → `(a0·b0 - a1·b1, a0·b1 + a1·b0)`
//!   — 4 Fp muls + 1 sub + 1 add (plus scratch for the temporary products)
//! - `Fp2::square((a0,a1))` → `((a0+a1)·(a0-a1), 2·a0·a1)`
//!   — 1 add + 1 sub + 2 muls + 1 add (for the doubling `2·a0·a1 = a0·a1 + a0·a1`)
//!
//! This module provides a small `Fp2Op` enum + `compile_fp2_op` which emits
//! the linear sequence of `FpOp`s that together compute the result. The
//! caller can concatenate multiple Fp2 op sequences into one trace and hand
//! that to the existing `nonnative_fp_constraints` AIR for proving.
//!
//! The intermediates that link the Fp rows are carried on the host side —
//! each `FpOp` has its own self-contained inputs and output, so the
//! compiler re-computes the product/sum scratch values via the host-side
//! `Fp::mul` / `Fp::add` / `Fp::sub` and feeds them into the next op.
//! An AIR layer that binds these intermediates across rows (a permutation
//! argument on the Fp "output" column) is a follow-up.

use crate::nonnative_fp::Fp;
use crate::nonnative_fp_air::FpOp;

/// Operations over Fp2, each compiling to a fixed number of [`FpOp`]s.
#[derive(Debug, Clone, Copy)]
pub enum Fp2Op {
    Add { a: (Fp, Fp), b: (Fp, Fp) },
    Sub { a: (Fp, Fp), b: (Fp, Fp) },
    Mul { a: (Fp, Fp), b: (Fp, Fp) },
    Square { a: (Fp, Fp) },
}

/// The Fp-level result of an Fp2 operation, returned alongside the compiled
/// [`FpOp`] sequence so callers can compose further ops.
pub fn compile_fp2_op(op: Fp2Op) -> (Vec<FpOp>, (Fp, Fp)) {
    match op {
        Fp2Op::Add { a, b } => {
            let out = (a.0.add(&b.0), a.1.add(&b.1));
            let ops = vec![
                FpOp::Add { a: a.0, b: b.0 },
                FpOp::Add { a: a.1, b: b.1 },
            ];
            (ops, out)
        }
        Fp2Op::Sub { a, b } => {
            let out = (a.0.sub(&b.0), a.1.sub(&b.1));
            let ops = vec![
                FpOp::Sub { a: a.0, b: b.0 },
                FpOp::Sub { a: a.1, b: b.1 },
            ];
            (ops, out)
        }
        Fp2Op::Mul { a, b } => {
            // (a0 + a1·u)(b0 + b1·u) = (a0·b0 − a1·b1) + (a0·b1 + a1·b0)·u
            let a0b0 = a.0.mul(&b.0);
            let a1b1 = a.1.mul(&b.1);
            let a0b1 = a.0.mul(&b.1);
            let a1b0 = a.1.mul(&b.0);
            let c0 = a0b0.sub(&a1b1);
            let c1 = a0b1.add(&a1b0);
            let ops = vec![
                FpOp::Mul { a: a.0, b: b.0 }, // a0·b0
                FpOp::Mul { a: a.1, b: b.1 }, // a1·b1
                FpOp::Mul { a: a.0, b: b.1 }, // a0·b1
                FpOp::Mul { a: a.1, b: b.0 }, // a1·b0
                FpOp::Sub { a: a0b0, b: a1b1 }, // c0
                FpOp::Add { a: a0b1, b: a1b0 }, // c1
            ];
            (ops, (c0, c1))
        }
        Fp2Op::Square { a } => {
            // (a0 + a1·u)² = (a0² − a1²) + 2·a0·a1·u
            //            = (a0+a1)(a0−a1) + (a0·a1 + a0·a1)·u
            // Uses the difference-of-squares identity to save one mul.
            let sum = a.0.add(&a.1);
            let diff = a.0.sub(&a.1);
            let c0 = sum.mul(&diff);
            let a0a1 = a.0.mul(&a.1);
            let c1 = a0a1.add(&a0a1);
            let ops = vec![
                FpOp::Add { a: a.0, b: a.1 }, // sum
                FpOp::Sub { a: a.0, b: a.1 }, // diff
                FpOp::Mul { a: sum, b: diff },  // c0
                FpOp::Mul { a: a.0, b: a.1 }, // a0·a1
                FpOp::Add { a: a0a1, b: a0a1 }, // c1
            ];
            (ops, (c0, c1))
        }
    }
}

/// BLS12-381 Miller-loop curve parameter `|x| = 0xd201000000010000`. Bits
/// of `x` are scanned high-to-low (skipping the leading 1) and each `1` bit
/// triggers an additional `addition_step` after the doubling.
pub const MILLER_X_ABS: u64 = 0xd201_0000_0001_0000;

/// `ell` line evaluation as Fp ops: scale the (c4, c1) Fp2 line coefficients
/// by the affine G1 point's (y_p, x_p) Fp coordinates, then fold into the
/// running Fp12 accumulator via `mul_by_014`.
///
/// Inputs:
///   - `f`: running Fp12 accumulator
///   - `coeffs`: line coefficients `(c4, c1, c0)` as emitted by the doubling /
///     addition steps
///   - `p`: affine G1 point (x_p, y_p) ∈ Fp²
///
/// Cost: 4 Fp Mul (c1 and c4 components × x_p / y_p) + Fp12 mul_by_014 (~170)
///       ≈ 174 Fp ops.
pub fn compile_ell(
    f: [[(Fp, Fp); 3]; 2],
    coeffs: [(Fp, Fp); 3],
    p: (Fp, Fp),
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(180);
    let (c4, c1, c0) = (coeffs[0], coeffs[1], coeffs[2]);

    // c1_scaled = (c1.c0 · x_p, c1.c1 · x_p)
    let c1_scaled = (c1.0.mul(&p.0), c1.1.mul(&p.0));
    ops.push(FpOp::Mul { a: c1.0, b: p.0 });
    ops.push(FpOp::Mul { a: c1.1, b: p.0 });

    // c4_scaled = (c4.c0 · y_p, c4.c1 · y_p)
    let c4_scaled = (c4.0.mul(&p.1), c4.1.mul(&p.1));
    ops.push(FpOp::Mul { a: c4.0, b: p.1 });
    ops.push(FpOp::Mul { a: c4.1, b: p.1 });

    let (mul_ops, new_f) = compile_fp12_mul_by_014(f, c0, c1_scaled, c4_scaled);
    ops.extend(mul_ops);
    (ops, new_f)
}

/// Compile the BLS12-381 Miller loop into a sequence of Fp ops.
///
/// Iterates from the second-most-significant bit of `MILLER_X_ABS` down to
/// bit 0. Each iteration:
///   1. f ← f²
///   2. (line, T) ← doubling_step(T)
///   3. f ← ell(f, line, P)
///   4. if bit_i set: (line, T) ← addition_step(T, Q); f ← ell(f, line, P)
///
/// After the loop, since `x` is negative, `f ← f.conjugate()`.
///
/// Cost: 64 outer iterations, with hamming-weight ≈ 6 add steps:
///   - 64 × Fp12 square (~165)        = ~10,560 Fp ops
///   - 64 × G2 doubling (~90)         = ~5,760 Fp ops
///   - 64 × ell (~174)                = ~11,136 Fp ops
///   - 6 × G2 addition (~110)         = ~660 Fp ops
///   - 6 × ell (~174)                 = ~1,044 Fp ops
///   - 1 × Fp12 conjugate (~6)        = 6 Fp ops
///   Total ≈ ~29,000 Fp ops per Miller loop.
///
/// Inputs:
///   - `p`: G1 affine point as (x_p, y_p) Fp tuple
///   - `q`: G2 affine point as (x_q, y_q) Fp2 tuple, used both as
///     the addition_step's q AND as the initial T (Z = 1 lifted to Jacobian).
pub fn compile_miller_loop(
    p: (Fp, Fp),
    q: [(Fp, Fp); 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(30_000);

    // f = Fp12::one() ; T = q.lift_to_jacobian() = (q.x, q.y, 1)
    let zero = Fp::zero();
    let one_fp = Fp::one();
    let zero_fp2 = (zero, zero);
    let one_fp2 = (one_fp, zero);
    let zero_fp6: [(Fp, Fp); 3] = [zero_fp2, zero_fp2, zero_fp2];
    let one_fp6: [(Fp, Fp); 3] = [one_fp2, zero_fp2, zero_fp2];
    let mut f: [[(Fp, Fp); 3]; 2] = [one_fp6, zero_fp6];
    let mut t: [(Fp, Fp); 3] = [q[0], q[1], one_fp2];

    let msb = 63i32 - MILLER_X_ABS.leading_zeros() as i32;
    for i in (0..msb).rev() {
        // f ← f²
        let (sq_ops, new_f) = compile_fp12_square(f);
        ops.extend(sq_ops);
        f = new_f;

        // (line, T) ← doubling_step(T)
        let (d_ops, new_t, line) = compile_g2_doubling_step(t);
        ops.extend(d_ops);
        t = new_t;

        // f ← ell(f, line, P)
        let (ell_ops, new_f) = compile_ell(f, line, p);
        ops.extend(ell_ops);
        f = new_f;

        if ((MILLER_X_ABS >> i) & 1) == 1 {
            let (a_ops, new_t, line) = compile_g2_addition_step(t, q);
            ops.extend(a_ops);
            t = new_t;
            let (ell_ops, new_f) = compile_ell(f, line, p);
            ops.extend(ell_ops);
            f = new_f;
        }
    }

    // x is negative → final conjugate
    let (conj_ops, conj_f) = compile_fp12_conjugate(f);
    ops.extend(conj_ops);
    f = conj_f;

    (ops, f)
}

// ── Fp2 mul-by-ξ (the Fp6/Fp12 tower nonresidue, ξ = 1 + u) ────────────
//
// fp2_mul_by_xi(a) = (a.c0 − a.c1) + (a.c0 + a.c1)·u
//
// Compiles to 1 Fp Sub (for c0) + 1 Fp Add (for c1) = 2 Fp ops. Returns
// the resulting Fp2 coordinates and the appended op sequence.
fn compile_fp2_mul_by_xi(a: (Fp, Fp)) -> (Vec<FpOp>, (Fp, Fp)) {
    let c0 = a.0.sub(&a.1);
    let c1 = a.0.add(&a.1);
    let ops = vec![
        FpOp::Sub { a: a.0, b: a.1 },
        FpOp::Add { a: a.0, b: a.1 },
    ];
    (ops, (c0, c1))
}

/// Compile an Fp6 element (c0 + c1·v + c2·v²) mul into Fp ops via the
/// standard Karatsuba layout. Mirrors `nonnative_tower::Fp6::mul` exactly.
///
/// Cost: 6 Fp2 muls + 3 Fp2 mul-by-ξ + 12 Fp2 adds/subs ≈ 60 Fp ops.
pub fn compile_fp6_mul(
    a: [(Fp, Fp); 3],
    b: [(Fp, Fp); 3],
) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(60);

    // v0 = a0·b0, v1 = a1·b1, v2 = a2·b2
    let (o, v0) = compile_fp2_op(Fp2Op::Mul { a: a[0], b: b[0] });
    ops.extend(o);
    let (o, v1) = compile_fp2_op(Fp2Op::Mul { a: a[1], b: b[1] });
    ops.extend(o);
    let (o, v2) = compile_fp2_op(Fp2Op::Mul { a: a[2], b: b[2] });
    ops.extend(o);

    // c0 = v0 + ξ * ((a1+a2)(b1+b2) − v1 − v2)
    let (o, sum_a12) = compile_fp2_op(Fp2Op::Add { a: a[1], b: a[2] });
    ops.extend(o);
    let (o, sum_b12) = compile_fp2_op(Fp2Op::Add { a: b[1], b: b[2] });
    ops.extend(o);
    let (o, prod12) = compile_fp2_op(Fp2Op::Mul { a: sum_a12, b: sum_b12 });
    ops.extend(o);
    let (o, t1) = compile_fp2_op(Fp2Op::Sub { a: prod12, b: v1 });
    ops.extend(o);
    let (o, t2) = compile_fp2_op(Fp2Op::Sub { a: t1, b: v2 });
    ops.extend(o);
    let (o, xi_t2) = compile_fp2_mul_by_xi(t2);
    ops.extend(o);
    let (o, c0) = compile_fp2_op(Fp2Op::Add { a: v0, b: xi_t2 });
    ops.extend(o);

    // c1 = (a0+a1)(b0+b1) − v0 − v1 + ξ * v2
    let (o, sum_a01) = compile_fp2_op(Fp2Op::Add { a: a[0], b: a[1] });
    ops.extend(o);
    let (o, sum_b01) = compile_fp2_op(Fp2Op::Add { a: b[0], b: b[1] });
    ops.extend(o);
    let (o, prod01) = compile_fp2_op(Fp2Op::Mul { a: sum_a01, b: sum_b01 });
    ops.extend(o);
    let (o, t1) = compile_fp2_op(Fp2Op::Sub { a: prod01, b: v0 });
    ops.extend(o);
    let (o, t2) = compile_fp2_op(Fp2Op::Sub { a: t1, b: v1 });
    ops.extend(o);
    let (o, xi_v2) = compile_fp2_mul_by_xi(v2);
    ops.extend(o);
    let (o, c1) = compile_fp2_op(Fp2Op::Add { a: t2, b: xi_v2 });
    ops.extend(o);

    // c2 = (a0+a2)(b0+b2) − v0 − v2 + v1
    let (o, sum_a02) = compile_fp2_op(Fp2Op::Add { a: a[0], b: a[2] });
    ops.extend(o);
    let (o, sum_b02) = compile_fp2_op(Fp2Op::Add { a: b[0], b: b[2] });
    ops.extend(o);
    let (o, prod02) = compile_fp2_op(Fp2Op::Mul { a: sum_a02, b: sum_b02 });
    ops.extend(o);
    let (o, t1) = compile_fp2_op(Fp2Op::Sub { a: prod02, b: v0 });
    ops.extend(o);
    let (o, t2) = compile_fp2_op(Fp2Op::Sub { a: t1, b: v2 });
    ops.extend(o);
    let (o, c2) = compile_fp2_op(Fp2Op::Add { a: t2, b: v1 });
    ops.extend(o);

    (ops, [c0, c1, c2])
}

/// Compile Fp6 squaring via the Chung-Hasan CH-SQR3 specialization, which
/// uses 3 Fp2 squares + 2 Fp2 mults instead of 6 generic Fp2 mults.
/// Mirrors `nonnative_tower::Fp6::square` exactly.
///
/// Formula:
///   s0 = a0²,  s1 = 2·a0·a1,  s2 = (a0 - a1 + a2)²
///   s3 = 2·a1·a2,  s4 = a2²
///   c0 = s0 + ξ·s3
///   c1 = s1 + ξ·s4
///   c2 = s1 + s2 + s3 - s0 - s4
///
/// Cost: 3 Fp2 squares (3·5=15) + 2 Fp2 muls (2·6=12) + ~10 Fp2 add/sub
///       (10·2=20) + 2 mul_by_xi (2·2=4) ≈ 51 Fp ops.
pub fn compile_fp6_square(a: [(Fp, Fp); 3]) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(55);

    // s0 = a0²
    let (o, s0) = compile_fp2_op(Fp2Op::Square { a: a[0] });
    ops.extend(o);

    // s1 = 2 · a0 · a1 = (a0·a1) + (a0·a1)
    let (o, ab) = compile_fp2_op(Fp2Op::Mul { a: a[0], b: a[1] });
    ops.extend(o);
    let (o, s1) = compile_fp2_op(Fp2Op::Add { a: ab, b: ab });
    ops.extend(o);

    // s2 = (a0 - a1 + a2)²
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: a[0], b: a[1] });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Add { a: t, b: a[2] });
    ops.extend(o);
    let (o, s2) = compile_fp2_op(Fp2Op::Square { a: t });
    ops.extend(o);

    // s3 = 2 · a1 · a2
    let (o, bc) = compile_fp2_op(Fp2Op::Mul { a: a[1], b: a[2] });
    ops.extend(o);
    let (o, s3) = compile_fp2_op(Fp2Op::Add { a: bc, b: bc });
    ops.extend(o);

    // s4 = a2²
    let (o, s4) = compile_fp2_op(Fp2Op::Square { a: a[2] });
    ops.extend(o);

    // c0 = s0 + ξ·s3
    let (o, xi_s3) = compile_fp2_mul_by_xi(s3);
    ops.extend(o);
    let (o, c0) = compile_fp2_op(Fp2Op::Add { a: s0, b: xi_s3 });
    ops.extend(o);

    // c1 = s1 + ξ·s4
    let (o, xi_s4) = compile_fp2_mul_by_xi(s4);
    ops.extend(o);
    let (o, c1) = compile_fp2_op(Fp2Op::Add { a: s1, b: xi_s4 });
    ops.extend(o);

    // c2 = s1 + s2 + s3 - s0 - s4
    let (o, t) = compile_fp2_op(Fp2Op::Add { a: s1, b: s2 });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Add { a: t, b: s3 });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t, b: s0 });
    ops.extend(o);
    let (o, c2) = compile_fp2_op(Fp2Op::Sub { a: t, b: s4 });
    ops.extend(o);

    (ops, [c0, c1, c2])
}

/// Compile Fp6 sparse multiplication `self * (c0 + c1·v)` where the third
/// coefficient is zero. Used in Miller loop line evaluations.
/// Mirrors `nonnative_tower::Fp6::mul_by_01`.
///
/// Cost: 4 Fp2 muls (~24) + 6 Fp2 add/sub (~12) + 1 mul_by_xi (~2) ≈ 38 Fp ops.
pub fn compile_fp6_mul_by_01(
    a: [(Fp, Fp); 3],
    c0: (Fp, Fp),
    c1: (Fp, Fp),
) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(40);

    // a_a = a0 · c0
    let (o, a_a) = compile_fp2_op(Fp2Op::Mul { a: a[0], b: c0 });
    ops.extend(o);

    // b_b = a1 · c1
    let (o, b_b) = compile_fp2_op(Fp2Op::Mul { a: a[1], b: c1 });
    ops.extend(o);

    // out_c0 = ξ * (a2 · c1) + a_a
    let (o, t1) = compile_fp2_op(Fp2Op::Mul { a: a[2], b: c1 });
    ops.extend(o);
    let (o, xi_t1) = compile_fp2_mul_by_xi(t1);
    ops.extend(o);
    let (o, out_c0) = compile_fp2_op(Fp2Op::Add { a: xi_t1, b: a_a });
    ops.extend(o);

    // out_c1 = (a0 + a1)(c0 + c1) − a_a − b_b
    let (o, sum_a01) = compile_fp2_op(Fp2Op::Add { a: a[0], b: a[1] });
    ops.extend(o);
    let (o, sum_c01) = compile_fp2_op(Fp2Op::Add { a: c0, b: c1 });
    ops.extend(o);
    let (o, t2) = compile_fp2_op(Fp2Op::Mul { a: sum_a01, b: sum_c01 });
    ops.extend(o);
    let (o, t3) = compile_fp2_op(Fp2Op::Sub { a: t2, b: a_a });
    ops.extend(o);
    let (o, out_c1) = compile_fp2_op(Fp2Op::Sub { a: t3, b: b_b });
    ops.extend(o);

    // out_c2 = (a0 + a2) · c0 − a_a + b_b
    let (o, sum_a02) = compile_fp2_op(Fp2Op::Add { a: a[0], b: a[2] });
    ops.extend(o);
    let (o, t4) = compile_fp2_op(Fp2Op::Mul { a: sum_a02, b: c0 });
    ops.extend(o);
    let (o, t5) = compile_fp2_op(Fp2Op::Sub { a: t4, b: a_a });
    ops.extend(o);
    let (o, out_c2) = compile_fp2_op(Fp2Op::Add { a: t5, b: b_b });
    ops.extend(o);

    (ops, [out_c0, out_c1, out_c2])
}

/// Compile Fp6 sparse multiplication `self * (0 + c1·v + 0·v²) = self * (c1·v)`.
/// Mirrors `nonnative_tower::Fp6::mul_by_1`.
///
/// Cost: 3 Fp2 muls (~18) + 1 mul_by_xi (~2) ≈ 20 Fp ops.
pub fn compile_fp6_mul_by_1(
    a: [(Fp, Fp); 3],
    c1: (Fp, Fp),
) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(22);

    // b_b = a1 · c1, t1 = a2 · c1, t2 = a0 · c1
    let (o, b_b) = compile_fp2_op(Fp2Op::Mul { a: a[1], b: c1 });
    ops.extend(o);
    let (o, t1) = compile_fp2_op(Fp2Op::Mul { a: a[2], b: c1 });
    ops.extend(o);
    let (o, t2) = compile_fp2_op(Fp2Op::Mul { a: a[0], b: c1 });
    ops.extend(o);

    // out = (ξ·t1, t2, b_b)
    let (o, out_c0) = compile_fp2_mul_by_xi(t1);
    ops.extend(o);
    (ops, [out_c0, t2, b_b])
}

/// Compile Fp12 sparse multiplication `self * (c0 + c1·v + c4·v·w)` —
/// the line accumulator update at each Miller-loop iteration. Mirrors
/// `nonnative_tower::Fp12::mul_by_014`.
///
/// Decomposition:
///   aa = self.c0 · (c0 + c1·v)              (Fp6 mul_by_01)
///   bb = self.c1 · (c4·v)                   (Fp6 mul_by_1)
///   o = c1 + c4
///   sum = self.c0 + self.c1                  (Fp6 add)
///   cross = sum · (c0 + o·v)                (Fp6 mul_by_01)
///   new.c1 = cross − aa − bb
///   new.c0 = bb · v + aa                    (mul_by_nonresidue + add)
///
/// Cost: 3 Fp6 sparse muls (~3·40 + 20 = 140) + 1 mul_by_nonresidue (2)
///       + 4 Fp6 add/sub (~24) ≈ 170 Fp ops.
pub fn compile_fp12_mul_by_014(
    a: [[(Fp, Fp); 3]; 2],
    c0: (Fp, Fp),
    c1: (Fp, Fp),
    c4: (Fp, Fp),
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(180);

    // aa = a.c0 · (c0 + c1·v)
    let (o, aa) = compile_fp6_mul_by_01(a[0], c0, c1);
    ops.extend(o);

    // bb = a.c1 · (c4·v)
    let (o, bb) = compile_fp6_mul_by_1(a[1], c4);
    ops.extend(o);

    // o = c1 + c4
    let (o_ops, o_val) = compile_fp2_op(Fp2Op::Add { a: c1, b: c4 });
    ops.extend(o_ops);

    // sum = a.c0 + a.c1
    let (o_ops, sum_a) = compile_fp6_add(a[0], a[1]);
    ops.extend(o_ops);

    // cross = sum_a · (c0 + o·v)
    let (o_ops, cross) = compile_fp6_mul_by_01(sum_a, c0, o_val);
    ops.extend(o_ops);

    // new_c1 = cross − aa − bb
    let (o_ops, t) = compile_fp6_sub(cross, aa);
    ops.extend(o_ops);
    let (o_ops, new_c1) = compile_fp6_sub(t, bb);
    ops.extend(o_ops);

    // new_c0 = bb · v + aa
    let (o_ops, bb_nr) = compile_fp6_mul_by_nonresidue(bb);
    ops.extend(o_ops);
    let (o_ops, new_c0) = compile_fp6_add(bb_nr, aa);
    ops.extend(o_ops);

    (ops, [new_c0, new_c1])
}

// ── G2 Jacobian doubling step + line coefficients ──────────────────────
//
// A G2 Jacobian point is (X, Y, Z) ∈ Fp2³. Doubling uses the standard
// addition-chain formulas with line coefficients extracted for the Miller
// loop's `ell` evaluation. Mirrors `pairing.rs::doubling_step`.

/// Negate an Fp2 in-place via two Fp Subs from zero.
fn compile_fp2_neg(a: (Fp, Fp)) -> (Vec<FpOp>, (Fp, Fp)) {
    let zero = Fp::zero();
    let ops = vec![
        FpOp::Sub { a: zero, b: a.0 },
        FpOp::Sub { a: zero, b: a.1 },
    ];
    (ops, (zero.sub(&a.0), zero.sub(&a.1)))
}

/// Double a G2 Jacobian point and emit the line coefficients `(c0, c1, c4)`
/// for the Fp12 `mul_by_014` line accumulation.
///
/// Inputs:  `r = (X, Y, Z)` as three Fp2 = `[(Fp, Fp); 3]`.
/// Outputs: the updated point `r' = (X', Y', Z')` plus the three Fp2 line
///          coefficients in the slot order `(c4, c1, c0)` matching `mul_by_014`.
///
/// Cost: ~70 Fp ops for the point update + ~20 Fp ops for the line coefs ≈ 90.
pub fn compile_g2_doubling_step(
    r: [(Fp, Fp); 3],
) -> (Vec<FpOp>, [(Fp, Fp); 3], [(Fp, Fp); 3]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(100);
    let (rx, ry, rz) = (r[0], r[1], r[2]);

    // tmp0 = X²
    let (o, tmp0) = compile_fp2_op(Fp2Op::Square { a: rx });
    ops.extend(o);
    // tmp1 = Y²
    let (o, tmp1) = compile_fp2_op(Fp2Op::Square { a: ry });
    ops.extend(o);
    // tmp2 = tmp1² = Y⁴
    let (o, tmp2) = compile_fp2_op(Fp2Op::Square { a: tmp1 });
    ops.extend(o);
    // tmp3 = 2·((X+Y²)² − X² − Y⁴)
    let (o, t) = compile_fp2_op(Fp2Op::Add { a: rx, b: tmp1 });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Square { a: t });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t, b: tmp0 });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t, b: tmp2 });
    ops.extend(o);
    let (o, tmp3) = compile_fp2_op(Fp2Op::Add { a: t, b: t });
    ops.extend(o);
    // tmp4 = 3·X² = tmp0 + tmp0 + tmp0
    let (o, two_tmp0) = compile_fp2_op(Fp2Op::Add { a: tmp0, b: tmp0 });
    ops.extend(o);
    let (o, tmp4) = compile_fp2_op(Fp2Op::Add { a: two_tmp0, b: tmp0 });
    ops.extend(o);
    // tmp6 = X + tmp4
    let (o, tmp6) = compile_fp2_op(Fp2Op::Add { a: rx, b: tmp4 });
    ops.extend(o);
    // tmp5 = tmp4²
    let (o, tmp5) = compile_fp2_op(Fp2Op::Square { a: tmp4 });
    ops.extend(o);
    // zsquared = Z²
    let (o, zsquared) = compile_fp2_op(Fp2Op::Square { a: rz });
    ops.extend(o);
    // new_x = tmp5 − tmp3 − tmp3
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: tmp5, b: tmp3 });
    ops.extend(o);
    let (o, new_x) = compile_fp2_op(Fp2Op::Sub { a: t, b: tmp3 });
    ops.extend(o);
    // new_z = (Z+Y)² − Y² − Z²
    let (o, t) = compile_fp2_op(Fp2Op::Add { a: rz, b: ry });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Square { a: t });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t, b: tmp1 });
    ops.extend(o);
    let (o, new_z) = compile_fp2_op(Fp2Op::Sub { a: t, b: zsquared });
    ops.extend(o);
    // new_y = (tmp3 − new_x)·tmp4 − 8·tmp2
    let (o, tmp_sub_x) = compile_fp2_op(Fp2Op::Sub { a: tmp3, b: new_x });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Mul { a: tmp_sub_x, b: tmp4 });
    ops.extend(o);
    // 8·Y⁴
    let (o, two_t2) = compile_fp2_op(Fp2Op::Add { a: tmp2, b: tmp2 });
    ops.extend(o);
    let (o, four_t2) = compile_fp2_op(Fp2Op::Add { a: two_t2, b: two_t2 });
    ops.extend(o);
    let (o, eight_t2) = compile_fp2_op(Fp2Op::Add { a: four_t2, b: four_t2 });
    ops.extend(o);
    let (o, new_y) = compile_fp2_op(Fp2Op::Sub { a: t, b: eight_t2 });
    ops.extend(o);

    // ── Line coefficients ──
    // c4 = 2·new_z·zsquared
    let (o, t) = compile_fp2_op(Fp2Op::Mul { a: new_z, b: zsquared });
    ops.extend(o);
    let (o, c4) = compile_fp2_op(Fp2Op::Add { a: t, b: t });
    ops.extend(o);
    // c1 = −(2·tmp4·zsquared)
    let (o, t) = compile_fp2_op(Fp2Op::Mul { a: tmp4, b: zsquared });
    ops.extend(o);
    let (o, two_t) = compile_fp2_op(Fp2Op::Add { a: t, b: t });
    ops.extend(o);
    let (o, c1) = compile_fp2_neg(two_t);
    ops.extend(o);
    // c0 = tmp6² − tmp0 − tmp5 − 4·tmp1
    let (o, tmp6_sq) = compile_fp2_op(Fp2Op::Square { a: tmp6 });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: tmp6_sq, b: tmp0 });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t, b: tmp5 });
    ops.extend(o);
    let (o, two_t1) = compile_fp2_op(Fp2Op::Add { a: tmp1, b: tmp1 });
    ops.extend(o);
    let (o, four_t1) = compile_fp2_op(Fp2Op::Add { a: two_t1, b: two_t1 });
    ops.extend(o);
    let (o, c0) = compile_fp2_op(Fp2Op::Sub { a: t, b: four_t1 });
    ops.extend(o);

    // Line coefs in (c4, c1, c0) order matching mul_by_014's argument order
    // — but our function signature returns them as a length-3 array; caller
    // passes them to mul_by_014 as (c0=array[2], c1=array[1], c4=array[0]).
    let coeffs = [c4, c1, c0];
    let new_point = [new_x, new_y, new_z];
    (ops, new_point, coeffs)
}

/// G2 Jacobian + Affine addition step: update `r ← r + q_aff` and emit the
/// line coefficients `(c4, c1, c0)` for `mul_by_014`. Mirrors
/// `pairing.rs::addition_step`.
///
/// Inputs:
///   - `r`: G2 Jacobian point (3 Fp2 = `[(Fp,Fp); 3]`)
///   - `q`: G2 Affine point (2 Fp2 = `[(Fp,Fp); 2]`)
/// Outputs: updated Jacobian point + line coeffs in `(c4, c1, c0)` slot order.
///
/// Cost: ~110 Fp ops for the point update + line coeffs.
pub fn compile_g2_addition_step(
    r: [(Fp, Fp); 3],
    q: [(Fp, Fp); 2],
) -> (Vec<FpOp>, [(Fp, Fp); 3], [(Fp, Fp); 3]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(120);
    let (rx, ry, rz) = (r[0], r[1], r[2]);
    let (qx, qy) = (q[0], q[1]);

    // zsquared = Z²
    let (o, zsquared) = compile_fp2_op(Fp2Op::Square { a: rz });
    ops.extend(o);
    // ysquared = q.y²
    let (o, ysquared) = compile_fp2_op(Fp2Op::Square { a: qy });
    ops.extend(o);
    // t0 = zsquared · q.x
    let (o, t0) = compile_fp2_op(Fp2Op::Mul { a: zsquared, b: qx });
    ops.extend(o);
    // t1 = ((q.y + Z)² − ysquared − zsquared) · zsquared
    let (o, qy_plus_rz) = compile_fp2_op(Fp2Op::Add { a: qy, b: rz });
    ops.extend(o);
    let (o, sq) = compile_fp2_op(Fp2Op::Square { a: qy_plus_rz });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: sq, b: ysquared });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t, b: zsquared });
    ops.extend(o);
    let (o, t1) = compile_fp2_op(Fp2Op::Mul { a: t, b: zsquared });
    ops.extend(o);
    // t2 = t0 − X
    let (o, t2) = compile_fp2_op(Fp2Op::Sub { a: t0, b: rx });
    ops.extend(o);
    // t3 = t2²
    let (o, t3) = compile_fp2_op(Fp2Op::Square { a: t2 });
    ops.extend(o);
    // t4 = 4·t3
    let (o, two_t3) = compile_fp2_op(Fp2Op::Add { a: t3, b: t3 });
    ops.extend(o);
    let (o, t4) = compile_fp2_op(Fp2Op::Add { a: two_t3, b: two_t3 });
    ops.extend(o);
    // t5 = t4 · t2
    let (o, t5) = compile_fp2_op(Fp2Op::Mul { a: t4, b: t2 });
    ops.extend(o);
    // t6 = t1 − Y − Y
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t1, b: ry });
    ops.extend(o);
    let (o, t6) = compile_fp2_op(Fp2Op::Sub { a: t, b: ry });
    ops.extend(o);
    // t9 = t6 · q.x
    let (o, t9) = compile_fp2_op(Fp2Op::Mul { a: t6, b: qx });
    ops.extend(o);
    // t7 = t4 · X
    let (o, t7) = compile_fp2_op(Fp2Op::Mul { a: t4, b: rx });
    ops.extend(o);

    // new_x = t6² − t5 − t7 − t7
    let (o, t6_sq) = compile_fp2_op(Fp2Op::Square { a: t6 });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t6_sq, b: t5 });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: t, b: t7 });
    ops.extend(o);
    let (o, new_x) = compile_fp2_op(Fp2Op::Sub { a: t, b: t7 });
    ops.extend(o);
    // new_z = (Z + t2)² − zsquared − t3
    let (o, u) = compile_fp2_op(Fp2Op::Add { a: rz, b: t2 });
    ops.extend(o);
    let (o, u_sq) = compile_fp2_op(Fp2Op::Square { a: u });
    ops.extend(o);
    let (o, t) = compile_fp2_op(Fp2Op::Sub { a: u_sq, b: zsquared });
    ops.extend(o);
    let (o, new_z) = compile_fp2_op(Fp2Op::Sub { a: t, b: t3 });
    ops.extend(o);
    // t8 = (t7 − new_x) · t6
    let (o, t7_sub_x) = compile_fp2_op(Fp2Op::Sub { a: t7, b: new_x });
    ops.extend(o);
    let (o, t8) = compile_fp2_op(Fp2Op::Mul { a: t7_sub_x, b: t6 });
    ops.extend(o);
    // new_y = t8 − 2·(Y · t5)
    let (o, y_t5) = compile_fp2_op(Fp2Op::Mul { a: ry, b: t5 });
    ops.extend(o);
    let (o, two_y_t5) = compile_fp2_op(Fp2Op::Add { a: y_t5, b: y_t5 });
    ops.extend(o);
    let (o, new_y) = compile_fp2_op(Fp2Op::Sub { a: t8, b: two_y_t5 });
    ops.extend(o);

    // Line coeffs
    // t10_first = q.y + new_z; then t10 = t10_first² − ysquared − new_z²
    let (o, t10_first) = compile_fp2_op(Fp2Op::Add { a: qy, b: new_z });
    ops.extend(o);
    let (o, t10_sq) = compile_fp2_op(Fp2Op::Square { a: t10_first });
    ops.extend(o);
    let (o, t10_diff_y) = compile_fp2_op(Fp2Op::Sub { a: t10_sq, b: ysquared });
    ops.extend(o);
    let (o, new_z_sq) = compile_fp2_op(Fp2Op::Square { a: new_z });
    ops.extend(o);
    let (o, t10) = compile_fp2_op(Fp2Op::Sub { a: t10_diff_y, b: new_z_sq });
    ops.extend(o);
    // t9_final = 2·t9 − t10
    let (o, two_t9) = compile_fp2_op(Fp2Op::Add { a: t9, b: t9 });
    ops.extend(o);
    let (o, t9_final) = compile_fp2_op(Fp2Op::Sub { a: two_t9, b: t10 });
    ops.extend(o);
    // c4_out = 2·new_z
    let (o, c4_out) = compile_fp2_op(Fp2Op::Add { a: new_z, b: new_z });
    ops.extend(o);
    // c1_out = 2·(-t6)
    let (o, neg_t6) = compile_fp2_neg(t6);
    ops.extend(o);
    let (o, c1_out) = compile_fp2_op(Fp2Op::Add { a: neg_t6, b: neg_t6 });
    ops.extend(o);
    // c0_out = t9_final
    let c0_out = t9_final;

    let coeffs = [c4_out, c1_out, c0_out];
    let new_point = [new_x, new_y, new_z];
    (ops, new_point, coeffs)
}

/// Compile Fp12 squaring via complex squaring (2 Fp6 mults vs 3 generic).
/// Mirrors `nonnative_tower::Fp12::square` exactly.
///
/// Formula:
///   v0 = a0 · a1
///   c0 = (a0 + a1)(a0 + v·a1) − v0 − v·v0
///   c1 = 2·v0
///
/// Cost: 2 Fp6 muls (~132) + 2 Fp6 mul_by_nonresidue (~4) + 5 Fp6 add/sub (~30)
///       ≈ 165 Fp ops per Fp12 square.
pub fn compile_fp12_square(
    a: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(170);

    // v0 = a0 · a1
    let (o, v0) = compile_fp6_mul(a[0], a[1]);
    ops.extend(o);

    // a0 + a1
    let (o, a0_plus_a1) = compile_fp6_add(a[0], a[1]);
    ops.extend(o);

    // a0 + v·a1   (v · a1 = mul_by_nonresidue(a1))
    let (o, v_a1) = compile_fp6_mul_by_nonresidue(a[1]);
    ops.extend(o);
    let (o, a0_plus_v_a1) = compile_fp6_add(a[0], v_a1);
    ops.extend(o);

    // c0_tmp = (a0+a1)(a0 + v·a1)
    let (o, c0_tmp) = compile_fp6_mul(a0_plus_a1, a0_plus_v_a1);
    ops.extend(o);

    // v · v0
    let (o, v_v0) = compile_fp6_mul_by_nonresidue(v0);
    ops.extend(o);

    // c0 = c0_tmp − v0 − v·v0
    let (o, t) = compile_fp6_sub(c0_tmp, v0);
    ops.extend(o);
    let (o, c0) = compile_fp6_sub(t, v_v0);
    ops.extend(o);

    // c1 = v0 + v0
    let (o, c1) = compile_fp6_add(v0, v0);
    ops.extend(o);

    (ops, [c0, c1])
}

/// Compile Fp12 conjugation: `(c0, c1) → (c0, -c1)`. Negates every Fp
/// coordinate of `c1` via `0 - x`. Cost: 6 Fp Sub ops (one per Fp inside the
/// 3 Fp2 of the c1 Fp6 component).
pub fn compile_fp12_conjugate(
    a: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let zero = Fp::zero();
    let mut ops: Vec<FpOp> = Vec::with_capacity(6);
    let mut neg_c1: [(Fp, Fp); 3] = [(zero, zero); 3];
    for k in 0..3 {
        let neg0 = zero.sub(&a[1][k].0);
        let neg1 = zero.sub(&a[1][k].1);
        ops.push(FpOp::Sub { a: zero, b: a[1][k].0 });
        ops.push(FpOp::Sub { a: zero, b: a[1][k].1 });
        neg_c1[k] = (neg0, neg1);
    }
    (ops, [a[0], neg_c1])
}

/// Compile Fp6 multiplication-by-nonresidue (`x * v`):
///   `(c0 + c1·v + c2·v²) * v = c2·ξ + c0·v + c1·v²`
/// Cost: 1 Fp2 mul-by-ξ = 2 Fp ops.
pub fn compile_fp6_mul_by_nonresidue(
    a: [(Fp, Fp); 3],
) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let (ops, c0) = compile_fp2_mul_by_xi(a[2]);
    // The other two coordinates are pure assignment (no Fp ops needed).
    (ops, [c0, a[0], a[1]])
}

/// Compile Fp12 multiplication using Karatsuba over Fp6 (w² = v):
/// `(a0 + a1·w)(b0 + b1·w) = (v0 + v·v1) + ((a0+a1)(b0+b1) − v0 − v1)·w`
///
/// Cost: 3 Fp6 muls + 1 Fp6 mul-by-nonresidue + 4 Fp6 adds/subs ≈ 220 Fp ops.
pub fn compile_fp12_mul(
    a: [[(Fp, Fp); 3]; 2],
    b: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(220);

    // v0 = a0 · b0
    let (o, v0) = compile_fp6_mul(a[0], b[0]);
    ops.extend(o);
    // v1 = a1 · b1
    let (o, v1) = compile_fp6_mul(a[1], b[1]);
    ops.extend(o);

    // c0 = v0 + v · v1 = v0 + mul_by_nonresidue(v1)
    let (o, nr_v1) = compile_fp6_mul_by_nonresidue(v1);
    ops.extend(o);
    let (o, c0) = compile_fp6_add(v0, nr_v1);
    ops.extend(o);

    // c1 = (a0 + a1) · (b0 + b1) − v0 − v1
    let (o, sum_a) = compile_fp6_add(a[0], a[1]);
    ops.extend(o);
    let (o, sum_b) = compile_fp6_add(b[0], b[1]);
    ops.extend(o);
    let (o, prod) = compile_fp6_mul(sum_a, sum_b);
    ops.extend(o);
    let (o, t1) = compile_fp6_sub(prod, v0);
    ops.extend(o);
    let (o, c1) = compile_fp6_sub(t1, v1);
    ops.extend(o);

    (ops, [c0, c1])
}

/// Compile an Fp6 element (c0 + c1·v + c2·v²) add or sub into Fp ops.
/// Uses component-wise Fp2 ops; each Fp6 add/sub emits 6 Fp ops.
pub fn compile_fp6_add(
    a: [(Fp, Fp); 3],
    b: [(Fp, Fp); 3],
) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let mut all_ops = Vec::with_capacity(6);
    let (o0, r0) = compile_fp2_op(Fp2Op::Add { a: a[0], b: b[0] });
    let (o1, r1) = compile_fp2_op(Fp2Op::Add { a: a[1], b: b[1] });
    let (o2, r2) = compile_fp2_op(Fp2Op::Add { a: a[2], b: b[2] });
    all_ops.extend(o0);
    all_ops.extend(o1);
    all_ops.extend(o2);
    (all_ops, [r0, r1, r2])
}

pub fn compile_fp6_sub(
    a: [(Fp, Fp); 3],
    b: [(Fp, Fp); 3],
) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let mut all_ops = Vec::with_capacity(6);
    let (o0, r0) = compile_fp2_op(Fp2Op::Sub { a: a[0], b: b[0] });
    let (o1, r1) = compile_fp2_op(Fp2Op::Sub { a: a[1], b: b[1] });
    let (o2, r2) = compile_fp2_op(Fp2Op::Sub { a: a[2], b: b[2] });
    all_ops.extend(o0);
    all_ops.extend(o1);
    all_ops.extend(o2);
    (all_ops, [r0, r1, r2])
}

/// Compile an Fp12 element (c0 + c1·w) add or sub into Fp ops. Each Fp12
/// add/sub emits 12 Fp ops via two component-wise Fp6 ops.
pub fn compile_fp12_add(
    a: [[(Fp, Fp); 3]; 2],
    b: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let (ops0, r0) = compile_fp6_add(a[0], b[0]);
    let (ops1, r1) = compile_fp6_add(a[1], b[1]);
    let mut all = ops0;
    all.extend(ops1);
    (all, [r0, r1])
}

pub fn compile_fp12_sub(
    a: [[(Fp, Fp); 3]; 2],
    b: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let (ops0, r0) = compile_fp6_sub(a[0], b[0]);
    let (ops1, r1) = compile_fp6_sub(a[1], b[1]);
    let mut all = ops0;
    all.extend(ops1);
    (all, [r0, r1])
}

/// Compile Fp2 conjugation `(c0, c1) → (c0, -c1)`. Emits exactly 1 Fp Sub
/// (the negation of the imaginary coordinate via `0 - c1`). The real
/// coordinate passes through with no Fp op.
fn compile_fp2_conjugate(a: (Fp, Fp)) -> (Vec<FpOp>, (Fp, Fp)) {
    let zero = Fp::zero();
    let ops = vec![FpOp::Sub { a: zero, b: a.1 }];
    (ops, (a.0, zero.sub(&a.1)))
}

/// Compile a single Fp inversion via [`FpOp::Inv`]. The Inv op asserts
/// `a · a^{-1} = 1` in-circuit; the witness inverse is computed by the
/// host. `a` must be nonzero or witness population panics.
fn compile_fp_invert(a: Fp) -> (Vec<FpOp>, Fp) {
    let inv = a.invert().expect("Fp::invert(0)");
    (vec![FpOp::Inv { a }], inv)
}

/// Compile Fp2 inversion `(c0, c1)^{-1} = (c0 - c1·u) / (c0² + c1²)`.
///
/// Uses one in-circuit `FpOp::Inv` for the Fp norm inverse. The norm
/// `c0² + c1²` is an Fp element, so it bottoms out cleanly.
///
/// Cost: 2 Fp Mul (c0², c1²) + 1 Fp Add (norm) + 1 Fp Inv + 1 Fp Sub
/// (negate c1 → 0 - c1) + 2 Fp Mul (scale c0, -c1 by norm_inv) = 7 Fp ops.
pub fn compile_fp2_invert(a: (Fp, Fp)) -> (Vec<FpOp>, (Fp, Fp)) {
    let zero = Fp::zero();
    let mut ops: Vec<FpOp> = Vec::with_capacity(7);

    let c0_sq = a.0.mul(&a.0);
    ops.push(FpOp::Mul { a: a.0, b: a.0 });
    let c1_sq = a.1.mul(&a.1);
    ops.push(FpOp::Mul { a: a.1, b: a.1 });
    let norm = c0_sq.add(&c1_sq);
    ops.push(FpOp::Add { a: c0_sq, b: c1_sq });

    let (inv_ops, norm_inv) = compile_fp_invert(norm);
    ops.extend(inv_ops);

    let neg_c1 = zero.sub(&a.1);
    ops.push(FpOp::Sub { a: zero, b: a.1 });

    let out_c0 = a.0.mul(&norm_inv);
    ops.push(FpOp::Mul { a: a.0, b: norm_inv });
    let out_c1 = neg_c1.mul(&norm_inv);
    ops.push(FpOp::Mul { a: neg_c1, b: norm_inv });

    (ops, (out_c0, out_c1))
}

/// Compile Fp6 inversion via the closed-form ti / denom formula from
/// `nonnative_tower::Fp6::invert`:
///
/// ```text
///   t0 = a0² − ξ a1 a2          t1 = ξ a2² − a0 a1          t2 = a1² − a0 a2
///   denom = a0 t0 + ξ (a2 t1 + a1 t2)
///   a^{-1} = (t0 + t1 v + t2 v²) / denom
/// ```
///
/// Bottoms out at one Fp2 inverse for `denom`, which itself uses one Fp
/// inverse — so the entire Fp6 inverse compiles to a single in-circuit
/// `FpOp::Inv` row plus the surrounding multiplicative chain.
///
/// Cost: ~85 Fp ops (3 Fp2 sq + 3 Fp2 mul + 3 mul_by_xi + 4 Fp2 sub +
/// 1 Fp2 add + 2 Fp2 mul for inner + 1 Fp2 mul + 1 mul_by_xi + 1 Fp2 add
/// for denom + 7 ops for Fp2 invert + 3 Fp2 mul for output scaling).
pub fn compile_fp6_invert(a: [(Fp, Fp); 3]) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(95);

    // Squares.
    let (o, a0_sq) = compile_fp2_op(Fp2Op::Square { a: a[0] }); ops.extend(o);
    let (o, a1_sq) = compile_fp2_op(Fp2Op::Square { a: a[1] }); ops.extend(o);
    let (o, a2_sq) = compile_fp2_op(Fp2Op::Square { a: a[2] }); ops.extend(o);

    // Cross products.
    let (o, a0_a1) = compile_fp2_op(Fp2Op::Mul { a: a[0], b: a[1] }); ops.extend(o);
    let (o, a0_a2) = compile_fp2_op(Fp2Op::Mul { a: a[0], b: a[2] }); ops.extend(o);
    let (o, a1_a2) = compile_fp2_op(Fp2Op::Mul { a: a[1], b: a[2] }); ops.extend(o);

    // t0 = a0_sq - ξ * a1_a2
    let (o, xi_a1a2) = compile_fp2_mul_by_xi(a1_a2); ops.extend(o);
    let (o, t0) = compile_fp2_op(Fp2Op::Sub { a: a0_sq, b: xi_a1a2 }); ops.extend(o);

    // t1 = ξ * a2_sq - a0_a1
    let (o, xi_a2sq) = compile_fp2_mul_by_xi(a2_sq); ops.extend(o);
    let (o, t1) = compile_fp2_op(Fp2Op::Sub { a: xi_a2sq, b: a0_a1 }); ops.extend(o);

    // t2 = a1_sq - a0_a2
    let (o, t2) = compile_fp2_op(Fp2Op::Sub { a: a1_sq, b: a0_a2 }); ops.extend(o);

    // inner = a2 * t1 + a1 * t2
    let (o, c2_t1) = compile_fp2_op(Fp2Op::Mul { a: a[2], b: t1 }); ops.extend(o);
    let (o, c1_t2) = compile_fp2_op(Fp2Op::Mul { a: a[1], b: t2 }); ops.extend(o);
    let (o, inner) = compile_fp2_op(Fp2Op::Add { a: c2_t1, b: c1_t2 }); ops.extend(o);

    // denom = a0 * t0 + ξ * inner
    let (o, c0_t0) = compile_fp2_op(Fp2Op::Mul { a: a[0], b: t0 }); ops.extend(o);
    let (o, xi_inner) = compile_fp2_mul_by_xi(inner); ops.extend(o);
    let (o, denom) = compile_fp2_op(Fp2Op::Add { a: c0_t0, b: xi_inner }); ops.extend(o);

    // Invert denom.
    let (o, denom_inv) = compile_fp2_invert(denom); ops.extend(o);

    // Output: ti * denom_inv.
    let (o, out_c0) = compile_fp2_op(Fp2Op::Mul { a: t0, b: denom_inv }); ops.extend(o);
    let (o, out_c1) = compile_fp2_op(Fp2Op::Mul { a: t1, b: denom_inv }); ops.extend(o);
    let (o, out_c2) = compile_fp2_op(Fp2Op::Mul { a: t2, b: denom_inv }); ops.extend(o);

    (ops, [out_c0, out_c1, out_c2])
}

/// Negate every Fp2 coordinate of an Fp6 element (used inside Fp12 inverse
/// to produce `-c1`). Cost: 6 Fp Subs (one per Fp inside the 3 Fp2).
fn compile_fp6_neg(a: [(Fp, Fp); 3]) -> (Vec<FpOp>, [(Fp, Fp); 3]) {
    let zero = Fp::zero();
    let mut ops: Vec<FpOp> = Vec::with_capacity(6);
    let mut neg: [(Fp, Fp); 3] = [(zero, zero); 3];
    for k in 0..3 {
        let n0 = zero.sub(&a[k].0);
        let n1 = zero.sub(&a[k].1);
        ops.push(FpOp::Sub { a: zero, b: a[k].0 });
        ops.push(FpOp::Sub { a: zero, b: a[k].1 });
        neg[k] = (n0, n1);
    }
    (ops, neg)
}

/// Compile Fp12 inversion `f^{-1} = conj_w(f) / (c0² − v c1²)` where
/// `conj_w(f) = (c0, -c1)` is the w-conjugate. Mirrors
/// `nonnative_tower::Fp12::invert`.
///
/// Bottoms out at exactly one in-circuit Fp inverse (inside `compile_fp6_invert
/// → compile_fp2_invert`).
///
/// Cost: 2 Fp6 squares (~100 Fp ops) + 1 Fp6 mul_by_nonresidue (2 ops) +
/// 1 Fp6 sub (6 ops) + 1 Fp6 invert (~85 ops) + 1 Fp6 neg (6 ops) +
/// 2 Fp6 muls (~130 ops) ≈ 330 Fp ops.
pub fn compile_fp12_invert(
    a: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(340);

    // c0_sq, c1_sq ∈ Fp6.
    let (o, c0_sq) = compile_fp6_square(a[0]); ops.extend(o);
    let (o, c1_sq) = compile_fp6_square(a[1]); ops.extend(o);

    // v · c1_sq via Fp6 mul_by_nonresidue.
    let (o, nr_c1_sq) = compile_fp6_mul_by_nonresidue(c1_sq); ops.extend(o);

    // norm = c0_sq − v c1_sq.
    let (o, norm) = compile_fp6_sub(c0_sq, nr_c1_sq); ops.extend(o);

    // norm_inv ∈ Fp6.
    let (o, norm_inv) = compile_fp6_invert(norm); ops.extend(o);

    // out_c0 = c0 · norm_inv.
    let (o, out_c0) = compile_fp6_mul(a[0], norm_inv); ops.extend(o);

    // out_c1 = (-c1) · norm_inv.
    let (o, neg_c1) = compile_fp6_neg(a[1]); ops.extend(o);
    let (o, out_c1) = compile_fp6_mul(neg_c1, norm_inv); ops.extend(o);

    (ops, [out_c0, out_c1])
}

/// Compile the Fp12 Frobenius map `π^k(x)` for an arbitrary power.
///
/// Mirrors `nonnative_tower::Fp12::frobenius_map` exactly. Given
/// `x = c0 + c1·w` with each `c{0,1}` an Fp6 = `a0 + a1·v + a2·v²`:
///
/// ```text
///   π^k(x) = F6(c0) + F6(c1) · γ_{k,6} · w
///   F6(a0 + a1 v + a2 v²)
///       = π^k_Fp2(a0)
///       + π^k_Fp2(a1) · γ_{k,2} · v
///       + π^k_Fp2(a2) · γ_{k,4} · v²
/// ```
///
/// where π_Fp2 is complex conjugation (since `p ≡ 3 mod 4`), applied iff `k`
/// is odd. The γ constants come from [`crate::nonnative_tower::frobenius_gamma_constants`].
///
/// `power` is reduced mod 12; `power = 0` returns the input untouched and
/// emits zero Fp ops.
///
/// Cost: roughly 6 Fp Subs (for k odd) + 4 Fp2 muls (γ_{k,2}, γ_{k,4} on v
/// and v² coords of both c0 and c1) + 3 Fp2 muls (γ_{k,6} on each Fp2 of
/// new_c1) ≈ 48 Fp ops for odd k, ≈ 42 for even k > 0.
pub fn compile_fp12_frobenius_map(
    a: [[(Fp, Fp); 3]; 2],
    power: usize,
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let k = power % 12;
    if k == 0 {
        return (Vec::new(), a);
    }

    let (g_k_2, g_k_4, g_k_6) = crate::nonnative_tower::frobenius_gamma_constants(k);
    let g2 = (g_k_2.c0, g_k_2.c1);
    let g4 = (g_k_4.c0, g_k_4.c1);
    let g6 = (g_k_6.c0, g_k_6.c1);

    let mut ops: Vec<FpOp> = Vec::with_capacity(48);

    // Apply F6 to the c0 and c1 Fp6 components of a.
    let f6 = |inp: [(Fp, Fp); 3], ops: &mut Vec<FpOp>| -> [(Fp, Fp); 3] {
        // Step 1: per-Fp2 Frobenius — conjugate iff k is odd.
        let (a0_f, a1_f, a2_f) = if k % 2 == 1 {
            let (o0, c0) = compile_fp2_conjugate(inp[0]);
            ops.extend(o0);
            let (o1, c1) = compile_fp2_conjugate(inp[1]);
            ops.extend(o1);
            let (o2, c2) = compile_fp2_conjugate(inp[2]);
            ops.extend(o2);
            (c0, c1, c2)
        } else {
            (inp[0], inp[1], inp[2])
        };
        // Step 2: scale a1 by γ_{k,2}, a2 by γ_{k,4}.
        let (m1_ops, b1) = compile_fp2_op(Fp2Op::Mul { a: a1_f, b: g2 });
        ops.extend(m1_ops);
        let (m2_ops, b2) = compile_fp2_op(Fp2Op::Mul { a: a2_f, b: g4 });
        ops.extend(m2_ops);
        [a0_f, b1, b2]
    };

    let new_c0 = f6(a[0], &mut ops);
    let pre_c1 = f6(a[1], &mut ops);

    // Step 3: multiply each Fp2 coord of new_c1 by γ_{k,6}.
    let (m0_ops, c1_0) = compile_fp2_op(Fp2Op::Mul { a: pre_c1[0], b: g6 });
    ops.extend(m0_ops);
    let (m1_ops, c1_1) = compile_fp2_op(Fp2Op::Mul { a: pre_c1[1], b: g6 });
    ops.extend(m1_ops);
    let (m2_ops, c1_2) = compile_fp2_op(Fp2Op::Mul { a: pre_c1[2], b: g6 });
    ops.extend(m2_ops);

    (ops, [new_c0, [c1_0, c1_1, c1_2]])
}

/// Compile Fp12 exponentiation by the BLS12-381 Miller-loop parameter
/// `x = -0xd201_0000_0001_0000`. Computes `f^x ∈ Fp12`.
///
/// Uses left-to-right binary square-and-multiply over `|x|` (`MILLER_X_ABS`),
/// then conjugates the result since `x < 0` (in the cyclotomic subgroup,
/// inverse equals w-conjugation, which is what `compile_fp12_conjugate`
/// emits — 6 Fp Subs, no FpOp::Inv needed).
///
/// `MILLER_X_ABS = 0xd201_0000_0001_0000` has Hamming weight 6, so the
/// chain is ~63 Fp12 squares (the leading-bit step is just an assignment)
/// plus 5 Fp12 muls + 1 Fp12 conjugate.
///
/// # Soundness note
///
/// The caller must guarantee `f` lies in the **cyclotomic subgroup**
/// `G_φ12(Fp)` (i.e. `f^(p^4 − p^2 + 1) = 1`). On cyclotomic inputs,
/// `f^{-1} = conj_w(f)`, so we use the conjugate as a free inverse for the
/// negative-`x` correction step. After `compile_final_exp_easy`, this is
/// always the case — every consumer of `compile_fp12_pow_by_x` should be
/// downstream of the easy part.
///
/// Cost: ~64·150 + 5·220 + 6 ≈ 10.7k Fp ops per call.
pub fn compile_fp12_pow_by_x(
    f: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(11_000);

    // Initialize at the leading bit of |x|. MILLER_X_ABS != 0 so leading_zeros < 64.
    let msb = 63i32 - MILLER_X_ABS.leading_zeros() as i32;
    let mut acc = f;

    // Iterate from msb-1 down to 0: square, then mul-by-base if bit is set.
    for i in (0..msb).rev() {
        let (sq_ops, sq) = compile_fp12_square(acc);
        ops.extend(sq_ops);
        acc = sq;
        if ((MILLER_X_ABS >> i) & 1) == 1 {
            let (mul_ops, prod) = compile_fp12_mul(acc, f);
            ops.extend(mul_ops);
            acc = prod;
        }
    }

    // Negative-x correction: f^x = (f^|x|)^{-1}. On cyclotomic inputs the
    // inverse is the w-conjugate, which is just the per-Fp negation of c1.
    let (conj_ops, neg_acc) = compile_fp12_conjugate(acc);
    ops.extend(conj_ops);
    acc = neg_acc;

    (ops, acc)
}

/// Compile the easy part of the BLS12-381 final exponentiation:
/// `f^((p^6 - 1)(p^2 + 1))`.
///
/// Decomposed as two stages:
///
/// ```text
///   stage 1:  e1 = f^(p^6 − 1) = π^6(f) · f^{-1} = conj_w(f) · f^{-1}
///   stage 2:  e2 = e1^(p^2 + 1) = π^2(e1) · e1
/// ```
///
/// Both stages use only Frobenius + Fp12 multiplication + Fp12 inversion,
/// all already algebraic in the Fp AIR. After stage 1 the result lives in
/// the cyclotomic subgroup `G_φ12(Fp)`, where `f^{-1} = conj_w(f)`. After
/// stage 2 the result lives in the trace-zero subgroup of `G_φ12(Fp)` and
/// is ready for the hard-part exponentiation by `Φ_12(p)/r`.
///
/// Cost: ~330 Fp ops (Fp12 invert) + ~225 Fp ops (Fp12 mul) ×2 + ~50 Fp ops
/// (two Frobenius maps) ≈ 850 Fp ops total — a tiny fraction of the ~28k
/// Fp ops in the Miller loop preceding it.
pub fn compile_final_exp_easy(
    f: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::with_capacity(900);

    // Stage 1: e1 = π^6(f) · f^{-1}.
    let (o, frob6_f) = compile_fp12_frobenius_map(f, 6);
    ops.extend(o);
    let (o, f_inv) = compile_fp12_invert(f);
    ops.extend(o);
    let (o, e1) = compile_fp12_mul(frob6_f, f_inv);
    ops.extend(o);

    // Stage 2: e2 = π^2(e1) · e1.
    let (o, frob2_e1) = compile_fp12_frobenius_map(e1, 2);
    ops.extend(o);
    let (o, e2) = compile_fp12_mul(frob2_e1, e1);
    ops.extend(o);

    (ops, e2)
}

/// Compile Fp12 exponentiation by an arbitrary multi-limb big-endian
/// unsigned exponent. Mirrors `pairing::fp12_pow_wide` exactly: left-to-
/// right binary square-and-multiply with the leading bit as a free assign.
///
/// Used to lift the BLS12-381 hard-part exponent `(p^4 − p^2 + 1)/r`
/// (~530 bits) into FpOps. Per-bit cost: 1 Fp12 square + (bit) 1 Fp12 mul,
/// so total ≈ (n_bits − 1) squares + popcount(exp) − 1 multiplies.
///
/// The algorithm makes no assumption about cyclotomic membership — works
/// for any Fp12 base. (For cyclotomic inputs, callers can substitute the
/// cheaper `compile_fp12_pow_by_x` chain instead.)
pub fn compile_fp12_pow_wide(
    base: [[(Fp, Fp); 3]; 2],
    exp: &[u64],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::new();
    let zero = Fp::zero();
    let one_fp = Fp::one();
    // Fp12::one in our slot ordering: c0 = (Fp6 one), c1 = (Fp6 zero).
    let one_fp12: [[(Fp, Fp); 3]; 2] = [
        [(one_fp, zero), (zero, zero), (zero, zero)],
        [(zero, zero), (zero, zero), (zero, zero)],
    ];

    let mut acc = one_fp12;
    let mut started = false;
    for &limb in exp {
        for bit in (0..64).rev() {
            if started {
                let (sq_ops, sq) = compile_fp12_square(acc);
                ops.extend(sq_ops);
                acc = sq;
            }
            if ((limb >> bit) & 1) == 1 {
                if !started {
                    acc = base;
                    started = true;
                } else {
                    let (mul_ops, prod) = compile_fp12_mul(acc, base);
                    ops.extend(mul_ops);
                    acc = prod;
                }
            }
        }
    }
    (ops, acc)
}

/// Compile `(ret · a)^(2^n)` in the cyclotomic subgroup of Fp12. Mirrors
/// blst's `mul_n_sqr` and is the squaring-window primitive used by
/// [`compile_fp12_raise_to_z_div_by_2`].
fn compile_fp12_mul_n_sqr(
    ret: [[(Fp, Fp); 3]; 2],
    a: [[(Fp, Fp); 3]; 2],
    n: usize,
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::new();
    let (mul_ops, mut acc) = compile_fp12_mul(ret, a);
    ops.extend(mul_ops);
    for _ in 0..n {
        let (sq_ops, sq) = compile_fp12_square(acc);
        ops.extend(sq_ops);
        acc = sq;
    }
    (ops, acc)
}

/// Compile `a^(z/2)` where `z = -0xd201_0000_0001_0000` is the BLS12-381
/// Miller parameter. Mirrors `pairing::fp12_raise_to_z_div_by_2` (in turn
/// mirroring blst's `raise_to_z_div_by_2`).
///
/// The chain hits |z|/2 = 0x6900_8000_0000_8000 via specific square
/// windows, then conjugates for the negative-z correction.
pub fn compile_fp12_raise_to_z_div_by_2(
    a: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::new();
    let (sq_ops, mut r) = compile_fp12_square(a);                 // 0x2
    ops.extend(sq_ops);
    let (s, next) = compile_fp12_mul_n_sqr(r, a, 2);              // 0xc
    ops.extend(s); r = next;
    let (s, next) = compile_fp12_mul_n_sqr(r, a, 3);              // 0x68
    ops.extend(s); r = next;
    let (s, next) = compile_fp12_mul_n_sqr(r, a, 9);              // 0xd200
    ops.extend(s); r = next;
    let (s, next) = compile_fp12_mul_n_sqr(r, a, 32);             // 0xd20100000000
    ops.extend(s); r = next;
    let (s, next) = compile_fp12_mul_n_sqr(r, a, 16 - 1);         // 0x6900800000008000
    ops.extend(s); r = next;
    let (c_ops, c) = compile_fp12_conjugate(r);                   // negative-z
    ops.extend(c_ops);
    (ops, c)
}

/// Compile `a^z` where `z = -0xd201_0000_0001_0000`. Equals
/// `(compile_fp12_raise_to_z_div_by_2)^2`. Companion of blst's `raise_to_z`.
pub fn compile_fp12_raise_to_z(
    a: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::new();
    let (half_ops, half) = compile_fp12_raise_to_z_div_by_2(a);
    ops.extend(half_ops);
    let (sq_ops, full) = compile_fp12_square(half);
    ops.extend(sq_ops);
    (ops, full)
}

/// Compile the BLS12-381 hard-part addition chain (mirrors
/// `pairing::hard_part_addchain` byte-for-byte). Produces `m^(3d)` where
/// `d = (p^4 − p^2 + 1)/r` — the IETF-convention pairing output, matching
/// blst.
///
/// Cost: 4 raise_to_z chains (~11k ops each) + ~10 Fp12 muls + several
/// Frobenius + cyclotomic squares ≈ **50k Fp ops**, vs ~373k for the
/// naive [`compile_final_exp_hard_naive`] — a ~7.5× speedup.
///
/// Like all primitives in the cyclotomic subgroup, this chain uses **zero**
/// in-circuit `FpOp::Inv`s.
pub fn compile_final_exp_hard_addchain(
    t2: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::new();

    // t1 = t2.cyclotomic_square().conjugate()
    let (s, t1_sq) = compile_fp12_square(t2);
    ops.extend(s);
    let (c, t1) = compile_fp12_conjugate(t1_sq);
    ops.extend(c);

    // t3 = t2^z
    let (z, t3) = compile_fp12_raise_to_z(t2);
    ops.extend(z);

    // t4 = t3^2
    let (s, t4) = compile_fp12_square(t3);
    ops.extend(s);

    // t5 = t1 · t3
    let (m, t5) = compile_fp12_mul(t1, t3);
    ops.extend(m);

    // t1 = t5^z
    let (z, t1) = compile_fp12_raise_to_z(t5);
    ops.extend(z);

    // t0 = t1^z
    let (z, t0) = compile_fp12_raise_to_z(t1);
    ops.extend(z);

    // t6 = t0^z
    let (z, t6) = compile_fp12_raise_to_z(t0);
    ops.extend(z);

    // t6 = t6 · t4
    let (m, t6) = compile_fp12_mul(t6, t4);
    ops.extend(m);

    // t4 = t6^z
    let (z, t4) = compile_fp12_raise_to_z(t6);
    ops.extend(z);

    // t5 = t5.conjugate()
    let (c, t5_conj) = compile_fp12_conjugate(t5);
    ops.extend(c);

    // t4 = t4 · t5_conj · t2
    let (m, t4) = compile_fp12_mul(t4, t5_conj);
    ops.extend(m);
    let (m, t4) = compile_fp12_mul(t4, t2);
    ops.extend(m);

    // t5 = t2.conjugate()
    let (c, t5) = compile_fp12_conjugate(t2);
    ops.extend(c);

    // t1 = t1 · t2
    let (m, t1) = compile_fp12_mul(t1, t2);
    ops.extend(m);

    // t1 = t1.frobenius_map(3)
    let (f, t1) = compile_fp12_frobenius_map(t1, 3);
    ops.extend(f);

    // t6 = t6 · t5
    let (m, t6) = compile_fp12_mul(t6, t5);
    ops.extend(m);

    // t6 = t6.frobenius_map(1)
    let (f, t6) = compile_fp12_frobenius_map(t6, 1);
    ops.extend(f);

    // t3 = t3 · t0
    let (m, t3) = compile_fp12_mul(t3, t0);
    ops.extend(m);

    // t3 = t3.frobenius_map(2)
    let (f, t3) = compile_fp12_frobenius_map(t3, 2);
    ops.extend(f);

    // t3 = t3 · t1
    let (m, t3) = compile_fp12_mul(t3, t1);
    ops.extend(m);

    // t3 = t3 · t6
    let (m, t3) = compile_fp12_mul(t3, t6);
    ops.extend(m);

    // out = t3 · t4
    let (m, out) = compile_fp12_mul(t3, t4);
    ops.extend(m);

    (ops, out)
}

/// **Naive** wide-exponent variant of the BLS12-381 hard part — kept for
/// reference and as a sanity check against the optimized
/// [`compile_final_exp_hard_addchain`]. Mirrors `pairing::hard_part`
/// (textbook `m^d` exponent, vs. the IETF-convention `m^(3d)` produced by
/// the addchain). Costs ~373k Fp ops per call vs ~50k for the addchain.
pub fn compile_final_exp_hard_naive(
    m: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let exp = crate::pairing::hard_part_exponent_cached();
    compile_fp12_pow_wide(m, exp)
}

/// Compile the BLS12-381 hard part of the final exponentiation:
/// `m^(3d)` where `d = (p^4 − p^2 + 1) / r` and `m` is the easy-part
/// output (lives in the cyclotomic subgroup `G_φ12(Fp)`). Output matches
/// blst / zkcrypto / IETF convention.
///
/// Default path: optimized Fuentes-Castañeda addition chain. ~50k Fp ops
/// per call (~7.5× cheaper than [`compile_final_exp_hard_naive`]'s 373k).
/// Uses **zero** in-circuit `FpOp::Inv`s — every step is pure
/// multiplication / squaring / Frobenius / cheap conjugation.
pub fn compile_final_exp_hard(
    m: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    compile_final_exp_hard_addchain(m)
}

/// Compile the full BLS12-381 final exponentiation: easy part followed by
/// hard part (IETF / blst convention). Output equals `blst_pairing(P, Q)`
/// when applied to a Miller-loop result on real points.
///
/// Cost: ~1k Fp ops (easy) + ~50k Fp ops (hard) ≈ 51k Fp ops total —
/// down from ~146k with the naive path.
pub fn compile_final_exponentiation(
    f: [[(Fp, Fp); 3]; 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::new();
    let (o, easy) = compile_final_exp_easy(f);
    ops.extend(o);
    let (o, hard) = compile_final_exp_hard(easy);
    ops.extend(o);
    (ops, hard)
}

/// Compile a **product of pairings** with a single shared final
/// exponentiation. Computes `final_exp(∏ miller_loop(p_i, q_i))` for the
/// supplied list of `(G1, G2)` point tuples.
///
/// This is the standard pattern for BLS verification, multi-message
/// aggregate verification, KZG batch openings, etc. — anywhere the
/// pairing equation reduces to a product of Miller-loop outputs whose
/// final exponentiation must equal `Fp12::one()`.
///
/// Cost: `n × 28k (Miller) + (n−1) × 220 (Fp12 mul) + 51k (final exp)` Fp
/// ops. For `n = 2` this is ~107k vs ~162k for two independent pairings —
/// a ~33% saving from sharing the final exp. Linear in `n` thereafter.
///
/// Empty input returns `Fp12::one()` (vacuous pairing equation).
pub fn compile_pairing_product(
    pairs: &[((Fp, Fp), [(Fp, Fp); 2])],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::new();
    let zero = Fp::zero();
    let one_fp = Fp::one();
    let zero_fp2 = (zero, zero);
    let one_fp2 = (one_fp, zero);
    let zero_fp6: [(Fp, Fp); 3] = [zero_fp2, zero_fp2, zero_fp2];
    let one_fp6: [(Fp, Fp); 3] = [one_fp2, zero_fp2, zero_fp2];
    let one_fp12: [[(Fp, Fp); 3]; 2] = [one_fp6, zero_fp6];

    if pairs.is_empty() {
        return (ops, one_fp12);
    }

    // First Miller loop seeds the product.
    let (m_ops, mut acc) = compile_miller_loop(pairs[0].0, pairs[0].1);
    ops.extend(m_ops);

    // Subsequent: Miller loop × accumulate via Fp12 mul.
    for (p, q) in &pairs[1..] {
        let (m_ops, mi) = compile_miller_loop(*p, *q);
        ops.extend(m_ops);
        let (mul_ops, prod) = compile_fp12_mul(acc, mi);
        ops.extend(mul_ops);
        acc = prod;
    }

    // Single shared final exponentiation.
    let (fe_ops, out) = compile_final_exponentiation(acc);
    ops.extend(fe_ops);

    (ops, out)
}

/// Compile a BLS12-381 signature verification trace. Returns the FpOps
/// that compute `final_exp(miller_loop(σ, -G1) · miller_loop(H(m), pk))`,
/// plus the resulting Fp12 element. The verifier should check that the
/// output equals `Fp12::one()` to accept the signature.
///
/// Inputs:
///   - `sig`: signature point on G2, as `[(x.c0, x.c1), (y.c0, y.c1)]`
///   - `pk`: public key point on G1, as `(x, y)`
///   - `msg_hash`: hash-to-curve output `H(m)` on G2 (host-side computed)
///   - `neg_g1`: the constant `-G1_generator` in affine form (caller
///     supplies — this is a known constant, no need to recompute per verify)
///
/// Cost: ~107k Fp ops per verify (vs ~160k for two independent pairings).
/// **Zero** in-circuit `FpOp::Inv`s outside the single one in the easy part.
pub fn compile_bls_verify(
    sig: [(Fp, Fp); 2],
    neg_g1: (Fp, Fp),
    msg_hash: [(Fp, Fp); 2],
    pk: (Fp, Fp),
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    compile_pairing_product(&[(neg_g1, sig), (pk, msg_hash)])
}

/// Compile the full BLS12-381 optimal-ate pairing `e(P, Q) ∈ Fp12`:
/// Miller loop output fed through the final exponentiation. Output equals
/// `pairing::pairing(P, Q)` exactly.
///
/// Inputs as raw Fp / Fp2 limb tuples (callers handle infinity / subgroup
/// checks externally). Cost: ~28k (Miller) + ~145k (final exp) ≈ 173k Fp
/// ops per pairing — every step bottoms out at FpOp Add/Sub/Mul/Inv with
/// exactly **one** in-circuit FpOp::Inv (inside compile_final_exp_easy's
/// Fp12 invert).
pub fn compile_pairing(
    p: (Fp, Fp),
    q: [(Fp, Fp); 2],
) -> (Vec<FpOp>, [[(Fp, Fp); 3]; 2]) {
    let mut ops: Vec<FpOp> = Vec::new();
    let (o, miller) = compile_miller_loop(p, q);
    ops.extend(o);
    let (o, out) = compile_final_exponentiation(miller);
    ops.extend(o);
    (ops, out)
}

/// Compile a whole sequence of Fp2 ops, concatenating the generated Fp ops
/// and returning the final result tuple.
pub fn compile_fp2_sequence(ops: &[Fp2Op]) -> (Vec<FpOp>, Vec<(Fp, Fp)>) {
    let mut all_ops: Vec<FpOp> = Vec::new();
    let mut results: Vec<(Fp, Fp)> = Vec::with_capacity(ops.len());
    for op in ops {
        let (sub_ops, out) = compile_fp2_op(*op);
        all_ops.extend(sub_ops);
        results.push(out);
    }
    (all_ops, results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonnative_fp::Fp2;

    fn fp2_from_u64(c0: u64, c1: u64) -> Fp2 {
        Fp2 {
            c0: Fp::from_u64(c0),
            c1: Fp::from_u64(c1),
        }
    }

    fn matches_fp2(got: (Fp, Fp), expected: &Fp2) -> bool {
        got.0 == expected.c0 && got.1 == expected.c1
    }

    #[test]
    fn fp2_add_compiles_to_two_fp_adds() {
        let a = fp2_from_u64(3, 5);
        let b = fp2_from_u64(7, 11);
        let expected = a.add(&b);
        let (ops, out) = compile_fp2_op(Fp2Op::Add {
            a: (a.c0, a.c1),
            b: (b.c0, b.c1),
        });
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0], FpOp::Add { .. }));
        assert!(matches!(ops[1], FpOp::Add { .. }));
        assert!(matches_fp2(out, &expected), "add result mismatch");
    }

    #[test]
    fn fp2_sub_compiles_to_two_fp_subs() {
        let a = fp2_from_u64(100, 50);
        let b = fp2_from_u64(3, 7);
        let expected = a.sub(&b);
        let (ops, out) = compile_fp2_op(Fp2Op::Sub {
            a: (a.c0, a.c1),
            b: (b.c0, b.c1),
        });
        assert_eq!(ops.len(), 2);
        assert!(matches_fp2(out, &expected));
    }

    #[test]
    fn fp2_mul_compiles_to_six_fp_ops() {
        let a = fp2_from_u64(3, 5);
        let b = fp2_from_u64(7, 11);
        let expected = a.mul(&b);
        let (ops, out) = compile_fp2_op(Fp2Op::Mul {
            a: (a.c0, a.c1),
            b: (b.c0, b.c1),
        });
        assert_eq!(ops.len(), 6, "Fp2::mul should compile to 6 Fp ops");
        assert!(matches_fp2(out, &expected), "mul result mismatch");
    }

    #[test]
    fn fp2_square_compiles_to_five_fp_ops() {
        let a = fp2_from_u64(42, 17);
        let expected = a.square();
        let (ops, out) = compile_fp2_op(Fp2Op::Square {
            a: (a.c0, a.c1),
        });
        assert_eq!(ops.len(), 5, "Fp2::square should compile to 5 Fp ops");
        assert!(matches_fp2(out, &expected), "square result mismatch");
    }

    #[test]
    fn fp2_sequence_chains_correctly() {
        // Compute (a + b) * c via two Fp2 ops, cross-check final result.
        // The compiler returns intermediate results so a caller can wire
        // the second op's inputs from the first op's output.
        let a = fp2_from_u64(3, 5);
        let b = fp2_from_u64(7, 11);
        let c = fp2_from_u64(13, 17);

        // Step 1: compute sum = a + b.
        let (add_ops, sum) = compile_fp2_op(Fp2Op::Add {
            a: (a.c0, a.c1),
            b: (b.c0, b.c1),
        });
        // Step 2: feed sum into a mul.
        let (mul_ops, prod) = compile_fp2_op(Fp2Op::Mul {
            a: sum,
            b: (c.c0, c.c1),
        });

        let mut all_ops = Vec::with_capacity(add_ops.len() + mul_ops.len());
        all_ops.extend(add_ops);
        all_ops.extend(mul_ops);

        // Expect: (a + b) * c via reference.
        let expected = a.add(&b).mul(&c);
        assert!(matches_fp2(prod, &expected));
        // Op count: 2 (add) + 6 (mul) = 8.
        assert_eq!(all_ops.len(), 8);
    }

    fn fp6_from_tuple(c: [(u64, u64); 3]) -> [(Fp, Fp); 3] {
        [
            (Fp::from_u64(c[0].0), Fp::from_u64(c[0].1)),
            (Fp::from_u64(c[1].0), Fp::from_u64(c[1].1)),
            (Fp::from_u64(c[2].0), Fp::from_u64(c[2].1)),
        ]
    }

    #[test]
    fn fp6_add_compiles_to_six_fp_ops_and_matches_reference() {
        use crate::nonnative_tower::Fp6;
        let a = fp6_from_tuple([(3, 5), (7, 11), (13, 17)]);
        let b = fp6_from_tuple([(2, 4), (6, 8), (10, 12)]);

        let (ops, out) = compile_fp6_add(a, b);
        assert_eq!(ops.len(), 6);

        // Cross-check against Fp6 reference add.
        let a_fp6 = Fp6 {
            c0: Fp2 { c0: a[0].0, c1: a[0].1 },
            c1: Fp2 { c0: a[1].0, c1: a[1].1 },
            c2: Fp2 { c0: a[2].0, c1: a[2].1 },
        };
        let b_fp6 = Fp6 {
            c0: Fp2 { c0: b[0].0, c1: b[0].1 },
            c1: Fp2 { c0: b[1].0, c1: b[1].1 },
            c2: Fp2 { c0: b[2].0, c1: b[2].1 },
        };
        let ref_sum = a_fp6.add(&b_fp6);
        assert!(matches_fp2(out[0], &Fp2 { c0: ref_sum.c0.c0, c1: ref_sum.c0.c1 }));
        assert!(matches_fp2(out[1], &Fp2 { c0: ref_sum.c1.c0, c1: ref_sum.c1.c1 }));
        assert!(matches_fp2(out[2], &Fp2 { c0: ref_sum.c2.c0, c1: ref_sum.c2.c1 }));
    }

    #[test]
    fn fp12_add_compiles_to_twelve_fp_ops() {
        let a = [
            fp6_from_tuple([(1, 2), (3, 4), (5, 6)]),
            fp6_from_tuple([(7, 8), (9, 10), (11, 12)]),
        ];
        let b = [
            fp6_from_tuple([(13, 14), (15, 16), (17, 18)]),
            fp6_from_tuple([(19, 20), (21, 22), (23, 24)]),
        ];
        let (ops, _out) = compile_fp12_add(a, b);
        assert_eq!(ops.len(), 12, "Fp12 add should compile to 12 Fp ops");
    }

    fn fp6_to_native(x: [(Fp, Fp); 3]) -> crate::nonnative_tower::Fp6 {
        crate::nonnative_tower::Fp6 {
            c0: Fp2 { c0: x[0].0, c1: x[0].1 },
            c1: Fp2 { c0: x[1].0, c1: x[1].1 },
            c2: Fp2 { c0: x[2].0, c1: x[2].1 },
        }
    }

    fn fp6_from_native(x: &crate::nonnative_tower::Fp6) -> [(Fp, Fp); 3] {
        [
            (x.c0.c0, x.c0.c1),
            (x.c1.c0, x.c1.c1),
            (x.c2.c0, x.c2.c1),
        ]
    }

    #[test]
    fn fp6_mul_matches_reference() {
        let a = fp6_from_tuple([(3, 5), (7, 11), (13, 17)]);
        let b = fp6_from_tuple([(2, 4), (6, 8), (10, 12)]);

        let (ops, out) = compile_fp6_mul(a, b);
        // 6 Fp2 muls (6×6=36) + 12 Fp2 add/sub (12×2=24) + 3 mul_by_xi (3×2=6) = 66.
        // Allow small variance for ordering choices; the actual count is fixed by the formula.
        assert!(ops.len() >= 60 && ops.len() <= 70, "Fp6 mul ops {}", ops.len());

        let expected = fp6_to_native(a).mul(&fp6_to_native(b));
        assert_eq!(out, fp6_from_native(&expected), "Fp6 mul mismatch");
    }

    #[test]
    fn fp6_mul_zero_and_one() {
        use crate::nonnative_tower::Fp6;
        let zero = fp6_from_native(&Fp6::zero());
        let one = fp6_from_native(&Fp6::one());
        let x = fp6_from_tuple([(42, 99), (17, 23), (8, 5)]);

        let (_, prod_zero) = compile_fp6_mul(zero, x);
        assert_eq!(prod_zero, zero, "0·x must be 0");

        let (_, prod_one) = compile_fp6_mul(one, x);
        assert_eq!(prod_one, x, "1·x must be x");
    }

    #[test]
    fn fp6_mul_by_nonresidue_matches_reference() {
        let a = fp6_from_tuple([(3, 5), (7, 11), (13, 17)]);
        let (ops, out) = compile_fp6_mul_by_nonresidue(a);
        assert_eq!(ops.len(), 2, "mul_by_nonresidue is 1 mul_by_xi = 2 Fp ops");
        let expected = fp6_to_native(a).mul_by_nonresidue();
        assert_eq!(out, fp6_from_native(&expected));
    }

    #[test]
    fn fp12_mul_matches_reference() {
        use crate::nonnative_tower::Fp12;
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let b_lo = fp6_from_tuple([(13, 14), (15, 16), (17, 18)]);
        let b_hi = fp6_from_tuple([(19, 20), (21, 22), (23, 24)]);

        let (ops, out) = compile_fp12_mul([a_lo, a_hi], [b_lo, b_hi]);
        // 3 Fp6 muls (~3·66=198) + 1 mul_by_nr (2) + 4 Fp6 add/sub (4·6=24) ≈ 224.
        assert!(ops.len() >= 200 && ops.len() <= 250, "Fp12 mul ops {}", ops.len());

        let a = Fp12 {
            c0: fp6_to_native(a_lo),
            c1: fp6_to_native(a_hi),
        };
        let b = Fp12 {
            c0: fp6_to_native(b_lo),
            c1: fp6_to_native(b_hi),
        };
        let expected = a.mul(&b);
        assert_eq!(out[0], fp6_from_native(&expected.c0));
        assert_eq!(out[1], fp6_from_native(&expected.c1));
    }

    #[test]
    fn fp6_square_matches_reference() {
        let a = fp6_from_tuple([(3, 5), (7, 11), (13, 17)]);
        let (ops, out) = compile_fp6_square(a);
        // 3 squares (3·5=15) + 2 muls (12) + ~10 add/sub (20) + 2 mul_by_xi (4) = 51
        assert!(ops.len() >= 45 && ops.len() <= 65, "Fp6 square ops {}", ops.len());

        let expected = fp6_to_native(a).square();
        assert_eq!(out, fp6_from_native(&expected));
    }

    #[test]
    fn fp6_square_equals_self_mul_self() {
        // Sanity: square(a) == a · a for a few inputs.
        let inputs = [
            fp6_from_tuple([(1, 0), (0, 0), (0, 0)]),
            fp6_from_tuple([(2, 3), (5, 7), (11, 13)]),
            fp6_from_tuple([(0, 1), (1, 0), (1, 1)]),
        ];
        for a in inputs {
            let (_, sq) = compile_fp6_square(a);
            let (_, mu) = compile_fp6_mul(a, a);
            assert_eq!(sq, mu, "square(a) must equal a·a");
        }
    }

    #[test]
    fn fp6_mul_by_01_matches_reference() {
        let a = fp6_from_tuple([(3, 5), (7, 11), (13, 17)]);
        let c0 = (Fp::from_u64(2), Fp::from_u64(4));
        let c1 = (Fp::from_u64(6), Fp::from_u64(8));
        let (ops, out) = compile_fp6_mul_by_01(a, c0, c1);
        assert!(ops.len() >= 30 && ops.len() <= 50, "mul_by_01 ops {}", ops.len());

        // Reference: full mul against (c0, c1, 0).
        let b = [c0, c1, (Fp::zero(), Fp::zero())];
        let (_, ref_out) = compile_fp6_mul(a, b);
        assert_eq!(out, ref_out, "mul_by_01 must match full-mul against (c0, c1, 0)");
    }

    #[test]
    fn fp6_mul_by_1_matches_reference() {
        let a = fp6_from_tuple([(3, 5), (7, 11), (13, 17)]);
        let c1 = (Fp::from_u64(6), Fp::from_u64(8));
        let (ops, out) = compile_fp6_mul_by_1(a, c1);
        assert!(ops.len() >= 18 && ops.len() <= 25, "mul_by_1 ops {}", ops.len());

        // Reference: full mul against (0, c1, 0).
        let b = [(Fp::zero(), Fp::zero()), c1, (Fp::zero(), Fp::zero())];
        let (_, ref_out) = compile_fp6_mul(a, b);
        assert_eq!(out, ref_out, "mul_by_1 must match full-mul against (0, c1, 0)");
    }

    #[test]
    fn fp12_mul_by_014_matches_reference() {
        use crate::nonnative_tower::Fp12;
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let c0 = (Fp::from_u64(13), Fp::from_u64(14));
        let c1 = (Fp::from_u64(15), Fp::from_u64(16));
        let c4 = (Fp::from_u64(17), Fp::from_u64(18));

        let (ops, out) = compile_fp12_mul_by_014([a_lo, a_hi], c0, c1, c4);
        // Expected ~170-200 ops; allow generous range.
        assert!(ops.len() >= 130 && ops.len() <= 220, "mul_by_014 ops {}", ops.len());

        // Cross-check against reference Fp12::mul_by_014.
        let a = Fp12 {
            c0: fp6_to_native(a_lo),
            c1: fp6_to_native(a_hi),
        };
        let c0_fp2 = Fp2 { c0: c0.0, c1: c0.1 };
        let c1_fp2 = Fp2 { c0: c1.0, c1: c1.1 };
        let c4_fp2 = Fp2 { c0: c4.0, c1: c4.1 };
        let expected = a.mul_by_014(&c0_fp2, &c1_fp2, &c4_fp2);
        assert_eq!(out[0], fp6_from_native(&expected.c0));
        assert_eq!(out[1], fp6_from_native(&expected.c1));
    }

    #[test]
    fn g2_addition_step_compiles() {
        let r = [
            (Fp::from_u64(7), Fp::from_u64(11)),
            (Fp::from_u64(13), Fp::from_u64(17)),
            (Fp::from_u64(1), Fp::from_u64(0)),
        ];
        let q = [
            (Fp::from_u64(3), Fp::from_u64(5)),
            (Fp::from_u64(19), Fp::from_u64(23)),
        ];
        let (ops, new_point, coeffs) = compile_g2_addition_step(r, q);
        assert!(ops.len() >= 90 && ops.len() <= 150, "addition step ops {}", ops.len());
        assert_ne!(new_point[0], r[0]);
        assert_ne!(coeffs[0], coeffs[1]);
    }

    #[test]
    fn miller_loop_compiles_with_expected_op_count() {
        // Sanity check the Miller loop unroller doesn't blow up on small inputs.
        // Op count must scale with: 64 outer iters × ~430 ops each + 6 add-step
        // expansions × ~280 ops each + 6 conjugate.
        // Expected: ~28k–32k Fp ops total.
        let p = (Fp::from_u64(7), Fp::from_u64(11));
        let q = [
            (Fp::from_u64(3), Fp::from_u64(5)),
            (Fp::from_u64(19), Fp::from_u64(23)),
        ];
        let (ops, _result) = compile_miller_loop(p, q);
        let total = ops.len();
        assert!(total >= 25_000 && total <= 40_000, "miller loop ops {}", total);

        // Hamming weight of 0xd201_0000_0001_0000 = 6 set bits inside the
        // 64-bit param, but the MSB is the leading 1 we skip; so the inner
        // loop sees `popcount(x) - 1` add steps. Verify approximately.
        let hw = MILLER_X_ABS.count_ones();
        assert_eq!(hw, 6, "BLS12-381 |x| should have hamming weight 6");
    }

    #[test]
    fn ell_compiles_with_expected_op_count() {
        // ell = 4 Fp Mul (scaling) + Fp12 mul_by_014 (~170) ≈ 174.
        let f_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let f_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let coeffs = [
            (Fp::from_u64(13), Fp::from_u64(14)),
            (Fp::from_u64(15), Fp::from_u64(16)),
            (Fp::from_u64(17), Fp::from_u64(18)),
        ];
        let p = (Fp::from_u64(19), Fp::from_u64(23));
        let (ops, _f_new) = compile_ell([f_lo, f_hi], coeffs, p);
        assert!(ops.len() >= 130 && ops.len() <= 220, "ell ops {}", ops.len());
    }

    #[test]
    fn g2_doubling_step_compiles_without_panicking() {
        // Just sanity-check: doubling on a point shaped like the G2 generator
        // produces a valid Fp op sequence and the line coefficients are
        // distinct Fp2 values. (Cross-check against pairing.rs::doubling_step
        // requires exposing G2Jacobian publicly; that's a follow-up. The math
        // shape — formula correctness — is verified via the ref impl.)
        let r = [
            (Fp::from_u64(7), Fp::from_u64(11)),
            (Fp::from_u64(13), Fp::from_u64(17)),
            (Fp::from_u64(1), Fp::from_u64(0)),
        ];
        let (ops, new_point, coeffs) = compile_g2_doubling_step(r);
        // ~70 ops point update + ~20 line coefs ≈ 90; allow generous range.
        assert!(ops.len() >= 70 && ops.len() <= 120, "doubling step ops {}", ops.len());
        // Line coefficients are typically distinct from each other and from
        // the point coordinates for non-trivial inputs.
        assert_ne!(coeffs[0], coeffs[1]);
        // Point should change (input ≠ output).
        assert_ne!(new_point[0], r[0]);
    }

    #[test]
    fn fp12_square_matches_reference() {
        use crate::nonnative_tower::Fp12;
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let (ops, out) = compile_fp12_square([a_lo, a_hi]);
        // 2 Fp6 muls (~132) + ~5 Fp6 add/sub (~30) + 2 Fp6 mul_by_nonresidue (~4)
        // ≈ 170. Allow generous variance.
        assert!(ops.len() >= 130 && ops.len() <= 200, "Fp12 square ops {}", ops.len());

        let a = Fp12 {
            c0: fp6_to_native(a_lo),
            c1: fp6_to_native(a_hi),
        };
        let expected = a.square();
        assert_eq!(out[0], fp6_from_native(&expected.c0));
        assert_eq!(out[1], fp6_from_native(&expected.c1));
    }

    #[test]
    fn fp12_square_equals_self_mul_self() {
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let (_, sq) = compile_fp12_square([a_lo, a_hi]);
        let (_, mu) = compile_fp12_mul([a_lo, a_hi], [a_lo, a_hi]);
        assert_eq!(sq, mu, "square(a) must equal a·a");
    }

    #[test]
    fn fp12_conjugate_negates_c1_only() {
        use crate::nonnative_tower::Fp12;
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);

        let (ops, out) = compile_fp12_conjugate([a_lo, a_hi]);
        assert_eq!(ops.len(), 6, "conjugate is 6 Fp Subs (2 per Fp2 × 3 Fp2)");

        // c0 unchanged.
        assert_eq!(out[0], a_lo);
        // c1 component-wise negated.
        let a_fp12 = Fp12 {
            c0: fp6_to_native(a_lo),
            c1: fp6_to_native(a_hi),
        };
        let conj_ref = a_fp12.conjugate();
        assert_eq!(out[0], fp6_from_native(&conj_ref.c0));
        assert_eq!(out[1], fp6_from_native(&conj_ref.c1));
    }

    #[test]
    fn fp12_conjugate_involutive() {
        let a_lo = fp6_from_tuple([(42, 99), (17, 23), (8, 5)]);
        let a_hi = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let (_, conj) = compile_fp12_conjugate([a_lo, a_hi]);
        let (_, conj_conj) = compile_fp12_conjugate(conj);
        assert_eq!(conj_conj[0], a_lo);
        assert_eq!(conj_conj[1], a_hi);
    }

    #[test]
    fn fp12_mul_zero_and_one() {
        use crate::nonnative_tower::Fp6;
        let zero_fp6 = fp6_from_native(&Fp6::zero());
        let one_fp6 = fp6_from_native(&Fp6::one());
        let x_lo = fp6_from_tuple([(42, 99), (17, 23), (8, 5)]);
        let x_hi = fp6_from_tuple([(1, 1), (2, 2), (3, 3)]);

        // 0 · x = 0
        let (_, prod_zero) = compile_fp12_mul(
            [zero_fp6, zero_fp6],
            [x_lo, x_hi],
        );
        assert_eq!(prod_zero[0], zero_fp6);
        assert_eq!(prod_zero[1], zero_fp6);

        // 1 · x = x  (Fp12::one = (Fp6::one, Fp6::zero))
        let (_, prod_one) = compile_fp12_mul(
            [one_fp6, zero_fp6],
            [x_lo, x_hi],
        );
        assert_eq!(prod_one[0], x_lo);
        assert_eq!(prod_one[1], x_hi);
    }

    #[test]
    fn fp2_mul_boundary_values() {
        // Identities: 0 * x = 0, 1 * x = x.
        let zero = fp2_from_u64(0, 0);
        let one = Fp2 { c0: Fp::one(), c1: Fp::zero() };
        let x = fp2_from_u64(42, 99);

        let (_, out0) = compile_fp2_op(Fp2Op::Mul {
            a: (zero.c0.clone(), zero.c1.clone()),
            b: (x.c0.clone(), x.c1.clone()),
        });
        assert!(matches_fp2(out0, &zero.mul(&x)));

        let (_, out1) = compile_fp2_op(Fp2Op::Mul {
            a: (one.c0.clone(), one.c1.clone()),
            b: (x.c0.clone(), x.c1.clone()),
        });
        assert!(matches_fp2(out1, &x));
    }

    fn fp12_to_native(a: [[(Fp, Fp); 3]; 2]) -> crate::nonnative_tower::Fp12 {
        crate::nonnative_tower::Fp12 {
            c0: fp6_to_native(a[0]),
            c1: fp6_to_native(a[1]),
        }
    }

    fn fp12_from_native(a: &crate::nonnative_tower::Fp12) -> [[(Fp, Fp); 3]; 2] {
        [fp6_from_native(&a.c0), fp6_from_native(&a.c1)]
    }

    #[test]
    fn fp12_frobenius_zero_is_identity() {
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let (ops, out) = compile_fp12_frobenius_map([a_lo, a_hi], 0);
        assert_eq!(ops.len(), 0, "π^0 must emit zero Fp ops");
        assert_eq!(out, [a_lo, a_hi]);
    }

    #[test]
    fn fp12_frobenius_matches_reference_for_each_power() {
        let a_lo = fp6_from_tuple([(11, 13), (17, 19), (23, 29)]);
        let a_hi = fp6_from_tuple([(31, 37), (41, 43), (47, 53)]);
        let a_native = fp12_to_native([a_lo, a_hi]);

        for k in 0..12 {
            let (_ops, out) = compile_fp12_frobenius_map([a_lo, a_hi], k);
            let expected = a_native.frobenius_map(k);
            assert_eq!(
                out,
                fp12_from_native(&expected),
                "π^{} mismatch vs Fp12::frobenius_map",
                k
            );
        }
    }

    #[test]
    fn fp12_frobenius_op_counts_are_bounded() {
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);

        // For k = 0 we already check zero ops above. For k > 0 the cost is
        // (6 conjugates if k odd) + 4 Fp2 muls (γ_{k,2}, γ_{k,4} on a1 of c0/c1
        // and a2 of c0/c1) + 3 Fp2 muls (γ_{k,6} on each Fp2 of new_c1).
        // 6 conjugates = 6 Fp Subs (1 per Fp2). 7 Fp2 muls × 6 ops = 42.
        // Total: odd k → 48, even k > 0 → 42.
        for k in 1..12 {
            let (ops, _) = compile_fp12_frobenius_map([a_lo, a_hi], k);
            let expected = if k % 2 == 1 { 48 } else { 42 };
            assert_eq!(
                ops.len(),
                expected,
                "k = {}: expected {} Fp ops, got {}",
                k,
                expected,
                ops.len()
            );
        }
    }

    #[test]
    fn fp12_frobenius_twelfth_power_is_identity() {
        // π^12 = id on Fp12. Iteratively applying π once 12 times must return
        // the original element.
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let mut current = [a_lo, a_hi];
        for _ in 0..12 {
            let (_, next) = compile_fp12_frobenius_map(current, 1);
            current = next;
        }
        assert_eq!(current, [a_lo, a_hi], "π^12 (iterated) must equal id");
    }

    #[test]
    fn fp12_frobenius_six_equals_conjugate() {
        // Identity from the tower: π^6(x) = x.conjugate() (since p^6 ≡ -1
        // when restricted to the c1·w part).
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let (_, frob6) = compile_fp12_frobenius_map([a_lo, a_hi], 6);
        let (_, conj) = compile_fp12_conjugate([a_lo, a_hi]);
        assert_eq!(frob6, conj, "π^6 must equal Fp12 conjugation");
    }

    #[test]
    fn fp12_frobenius_powers_compose() {
        // π^i ∘ π^j = π^{i+j}  on the Fp12 level.
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        for i in 0..6 {
            for j in 0..6 {
                let (_, after_j) = compile_fp12_frobenius_map([a_lo, a_hi], j);
                let (_, after_ij) = compile_fp12_frobenius_map(after_j, i);
                let (_, direct) = compile_fp12_frobenius_map([a_lo, a_hi], i + j);
                assert_eq!(after_ij, direct, "π^{} ∘ π^{} != π^{}", i, j, i + j);
            }
        }
    }

    #[test]
    fn fp2_invert_matches_reference() {
        // Cross-check compile_fp2_invert against Fp2::invert for several
        // sample values, including ones with both coords nonzero.
        let cases: &[(u64, u64)] = &[
            (1, 0), (0, 1), (3, 5), (7, 11), (42, 99),
            (0xC0FFEE, 0xDEADBEEF), (0xffff_ffff_ffff_ffff, 1),
        ];
        for &(c0, c1) in cases {
            let a = fp2_from_u64(c0, c1);
            let expected = a.invert().expect("nonzero sample");
            let (_, out) = compile_fp2_invert((a.c0, a.c1));
            assert!(
                matches_fp2(out, &expected),
                "Fp2 invert mismatch for ({}, {})",
                c0, c1,
            );
        }
    }

    #[test]
    fn fp2_invert_op_count_is_seven() {
        let a = fp2_from_u64(13, 17);
        let (ops, _) = compile_fp2_invert((a.c0, a.c1));
        assert_eq!(ops.len(), 7, "Fp2 invert must compile to 7 Fp ops");
        // Exactly one of those is the in-circuit Inv.
        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(inv_count, 1, "Fp2 invert must use exactly one FpOp::Inv");
    }

    #[test]
    fn fp2_invert_round_trip_is_identity() {
        // a · a^{-1} = 1 by composing with multiplication.
        let a = fp2_from_u64(31337, 0xCAFE);
        let (_, inv) = compile_fp2_invert((a.c0, a.c1));
        let (_, prod) = compile_fp2_op(Fp2Op::Mul {
            a: (a.c0, a.c1),
            b: inv,
        });
        let one = Fp2 { c0: Fp::one(), c1: Fp::zero() };
        assert!(matches_fp2(prod, &one), "a · a^{{-1}} must equal 1");
    }

    #[test]
    fn fp6_invert_matches_reference() {
        // Cross-check compile_fp6_invert against Fp6::invert.
        let a = fp6_from_tuple([(3, 5), (7, 11), (13, 17)]);
        let expected = fp6_to_native(a).invert().expect("nonzero");
        let (_, out) = compile_fp6_invert(a);
        assert_eq!(out, fp6_from_native(&expected), "Fp6 invert mismatch");
    }

    #[test]
    fn fp6_invert_round_trip_is_identity() {
        // a · a^{-1} = 1 by composing with full Fp6 multiplication.
        let a = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let (_, inv) = compile_fp6_invert(a);
        let (_, prod) = compile_fp6_mul(a, inv);
        use crate::nonnative_tower::Fp6;
        let one = fp6_from_native(&Fp6::one());
        assert_eq!(prod, one, "Fp6 a · a^{{-1}} must equal 1");
    }

    #[test]
    fn fp6_invert_uses_exactly_one_fp_inv() {
        // The whole Fp6 inverse should bottom out at exactly one in-circuit
        // FpOp::Inv (the one inside compile_fp2_invert(denom)).
        let a = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let (ops, _) = compile_fp6_invert(a);
        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(
            inv_count, 1,
            "Fp6 invert must use exactly one FpOp::Inv (got {})",
            inv_count,
        );
    }

    #[test]
    fn fp12_invert_matches_reference() {
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let a_native = fp12_to_native([a_lo, a_hi]);
        let expected = a_native.invert().expect("nonzero");
        let (_, out) = compile_fp12_invert([a_lo, a_hi]);
        assert_eq!(out, fp12_from_native(&expected), "Fp12 invert mismatch");
    }

    #[test]
    fn fp12_invert_round_trip_is_identity() {
        // a · a^{-1} = 1 over Fp12 — the cleanest end-to-end test that the
        // whole tower-of-towers inverse compiler is consistent.
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let (_, inv) = compile_fp12_invert([a_lo, a_hi]);
        let (_, prod) = compile_fp12_mul([a_lo, a_hi], inv);
        use crate::nonnative_tower::{Fp12, Fp6};
        let one = [fp6_from_native(&Fp6::one()), fp6_from_native(&Fp6::zero())];
        assert_eq!(prod, one, "Fp12 a · a^{{-1}} must equal 1");
        // Also cross-check with native Fp12.
        let a_native = fp12_to_native([a_lo, a_hi]);
        let prod_native = a_native.mul(&Fp12 {
            c0: fp6_to_native(inv[0]),
            c1: fp6_to_native(inv[1]),
        });
        assert_eq!(prod_native, Fp12::one(), "native cross-check");
    }

    #[test]
    fn compile_fp12_raise_to_z_matches_native_reference() {
        // Cross-check compile_fp12_raise_to_z against pairing::fp12_raise_to_z
        // on a cyclotomic input (lifted via the easy part).
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let (_, m_compiled) = compile_final_exp_easy([a_lo, a_hi]);
        let m_native = fp12_to_native(m_compiled);

        let (_, got) = compile_fp12_raise_to_z(m_compiled);
        let expected = crate::pairing::fp12_raise_to_z(&m_native);
        assert_eq!(
            got,
            fp12_from_native(&expected),
            "compile_fp12_raise_to_z != native reference",
        );

        // Also raise_to_z_div_by_2.
        let (_, got_half) = compile_fp12_raise_to_z_div_by_2(m_compiled);
        let expected_half = crate::pairing::fp12_raise_to_z_div_by_2(&m_native);
        assert_eq!(
            got_half,
            fp12_from_native(&expected_half),
            "compile_fp12_raise_to_z_div_by_2 != native reference",
        );
    }

    #[test]
    fn compile_final_exp_hard_addchain_matches_native_reference() {
        // Compile output of the optimized hard-part chain must match the
        // native Rust reference `pairing::hard_part_addchain`. Fast: both
        // are O(50k Fp ops), no naive ~530-bit path involved.
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let (_, easy_compiled) = compile_final_exp_easy([a_lo, a_hi]);
        let easy_native = fp12_to_native(easy_compiled);

        let (_, got) = compile_final_exp_hard_addchain(easy_compiled);
        let expected = crate::pairing::hard_part_addchain(&easy_native);
        assert_eq!(
            got,
            fp12_from_native(&expected),
            "compile_final_exp_hard_addchain != native reference",
        );
    }

    #[test]
    fn compile_final_exp_hard_addchain_op_count() {
        // Optimized chain should be ~50k Fp ops per call — roughly
        // 4 raise_to_z (~11k each) + ~10 muls + a few squares/Frobenius.
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let (_, easy) = compile_final_exp_easy([a_lo, a_hi]);
        let (ops, _) = compile_final_exp_hard_addchain(easy);

        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(
            inv_count, 0,
            "addchain hard part must use zero FpOp::Inv (got {})",
            inv_count,
        );

        // Empirical: ~50k. Bound generously.
        assert!(
            ops.len() > 30_000 && ops.len() < 80_000,
            "addchain hard-part op count {} outside expected ~50k range",
            ops.len(),
        );
    }

    #[test]
    fn compile_final_exp_hard_op_count_in_expected_range() {
        // `compile_final_exp_hard` defaults to the optimized addchain
        // (~50k Fp ops). The naive 530-bit path is preserved in
        // `compile_final_exp_hard_naive` for cross-check only.
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let (_, m) = compile_final_exp_easy([a_lo, a_hi]);
        let (ops, _) = compile_final_exp_hard(m);

        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(
            inv_count, 0,
            "default hard-part chain must use zero FpOp::Inv (got {})",
            inv_count,
        );

        // Optimized addchain: ~50k Fp ops. Bounds set generously.
        assert!(
            ops.len() > 30_000 && ops.len() < 80_000,
            "default hard-part op count {} outside expected ~50k range",
            ops.len(),
        );
    }

    #[test]
    fn compile_pairing_op_count_and_inv_count() {
        // Structural test only — the slow #[ignore]'d test below cross-
        // checks against pairing::pairing.
        // Use small synthetic inputs (the pairing chain is generic in P, Q;
        // off-curve inputs give a meaningless field element but the
        // FpOp sequence is well-formed and the trace is provable).
        let p = (Fp::from_u64(7), Fp::from_u64(11));
        let q = [
            (Fp::from_u64(3), Fp::from_u64(5)),
            (Fp::from_u64(19), Fp::from_u64(23)),
        ];
        let (ops, _) = compile_pairing(p, q);

        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(
            inv_count, 1,
            "pairing must use exactly one FpOp::Inv (got {})",
            inv_count,
        );

        // ~28k Miller + ~1k easy + ~50k hard (addchain) ≈ 80k Fp ops.
        assert!(
            ops.len() > 50_000 && ops.len() < 150_000,
            "pairing op count {} outside expected range",
            ops.len(),
        );
    }

    #[test]
    fn compile_final_exponentiation_matches_native_addchain() {
        // Default `compile_final_exponentiation` uses easy + addchain (IETF
        // convention). Cross-check against the corresponding native
        // composition. Fast: both paths are ~50k Fp ops, no naive
        // wide-exponent involved.
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let f_native = fp12_to_native([a_lo, a_hi]);

        // Native: easy + addchain.
        let f1 = f_native.conjugate();
        let f2 = f_native.invert().unwrap();
        let f3 = f1.mul(&f2);
        let f4 = f3.frobenius_map(2);
        let easy_native = f4.mul(&f3);
        let expected = crate::pairing::hard_part_addchain(&easy_native);

        let (_, out) = compile_final_exponentiation([a_lo, a_hi]);
        assert_eq!(
            out,
            fp12_from_native(&expected),
            "compile_final_exponentiation diverges from native easy+addchain composition",
        );
    }

    #[test]
    fn compile_pairing_product_empty_returns_one() {
        // Vacuous pairing equation `∏ e(p_i, q_i) = 1` over no terms.
        let (ops, out) = compile_pairing_product(&[]);
        assert_eq!(ops.len(), 0, "empty product emits zero Fp ops");
        use crate::nonnative_tower::{Fp12, Fp6};
        let one = [fp6_from_native(&Fp6::one()), fp6_from_native(&Fp6::zero())];
        assert_eq!(out, one);
        // Sanity: out == native Fp12::one
        assert_eq!(fp12_to_native(out), Fp12::one());
    }

    #[test]
    fn compile_pairing_product_two_inverse_pairs_equals_one() {
        // Bilinearity test: e(G2, -G1) · e(G2, G1) = e(G2, -G1 + G1) =
        // e(G2, identity) = 1. The shared final exp returns Fp12::one()
        // exactly. This is also the structure of BLS verify on a trivial
        // identity-signature case.
        use crate::pairing::{G1Affine, G2Affine};
        let g1 = G1Affine::generator();
        let g1_neg = g1.neg();
        let g2 = G2Affine::generator();

        let p1 = (g1.x, g1.y);
        let p2 = (g1_neg.x, g1_neg.y);
        let q = [(g2.x.c0, g2.x.c1), (g2.y.c0, g2.y.c1)];

        let (_, out) = compile_pairing_product(&[(p1, q), (p2, q)]);
        use crate::nonnative_tower::{Fp12, Fp6};
        let one = [fp6_from_native(&Fp6::one()), fp6_from_native(&Fp6::zero())];
        assert_eq!(out, one, "e(G2, G1) · e(G2, -G1) must equal Fp12::one()");
        assert_eq!(fp12_to_native(out), Fp12::one(), "native cross-check");
    }

    #[test]
    fn compile_bls_verify_on_real_signature_succeeds() {
        // End-to-end: generate a real BLS signature via bls_sig, decompress
        // pk/sig/H(msg) to affine blst points, convert to our Fp/Fp2
        // coordinates, run compile_bls_verify, assert output == Fp12::one.
        // This validates that our compiled pairing trace agrees with blst on
        // the actual ciphersuite (BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_).
        use crate::bls_sig::{hash_to_g2_affine, pk_to_affine, sig_to_affine, SecretKey};
        use crate::pairing::G1Affine;
        use blst::{
            blst_bendian_from_fp, blst_fp,
        };
        use crate::nonnative_fp::{Fp, Fp2};

        // Helper: convert blst_fp → Fp via 48-byte BE round-trip.
        fn blst_fp_to_fp(b: &blst_fp) -> Fp {
            let mut bytes = [0u8; 48];
            unsafe { blst_bendian_from_fp(bytes.as_mut_ptr(), b); }
            Fp::from_bytes_be(&bytes).expect("blst fp canonical")
        }

        let sk = SecretKey::from_u8_seed(7);
        let pk = sk.public_key();
        let msg = b"hello bls12-381 in-circuit pairing";
        let dst = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
        let sig = sk.sign(msg, dst);

        let pk_aff = pk_to_affine(&pk).expect("pk decompresses");
        let sig_aff = sig_to_affine(&sig).expect("sig decompresses");
        let h_aff = hash_to_g2_affine(msg, dst);

        // Convert to our coord system.
        let pk_fp = (blst_fp_to_fp(&pk_aff.x), blst_fp_to_fp(&pk_aff.y));
        let neg_g1 = G1Affine::generator().neg();
        let neg_g1_fp = (neg_g1.x, neg_g1.y);

        let sig_fp2: [(Fp, Fp); 2] = [
            (blst_fp_to_fp(&sig_aff.x.fp[0]), blst_fp_to_fp(&sig_aff.x.fp[1])),
            (blst_fp_to_fp(&sig_aff.y.fp[0]), blst_fp_to_fp(&sig_aff.y.fp[1])),
        ];
        let h_fp2: [(Fp, Fp); 2] = [
            (blst_fp_to_fp(&h_aff.x.fp[0]), blst_fp_to_fp(&h_aff.x.fp[1])),
            (blst_fp_to_fp(&h_aff.y.fp[0]), blst_fp_to_fp(&h_aff.y.fp[1])),
        ];

        // Sanity: drop unused Fp2 import warning by referencing it.
        let _ = Fp2::zero();

        let (_, out) = compile_bls_verify(sig_fp2, neg_g1_fp, h_fp2, pk_fp);

        use crate::nonnative_tower::{Fp12, Fp6};
        let one = [fp6_from_native(&Fp6::one()), fp6_from_native(&Fp6::zero())];
        assert_eq!(out, one, "real BLS signature must verify to Fp12::one()");
        assert_eq!(fp12_to_native(out), Fp12::one(), "native cross-check");
    }

    #[test]
    fn compile_bls_verify_on_identity_pair_succeeds() {
        // Plug the same identity into the BLS verify wrapper. Equation:
        //   e(σ, -G1) · e(H, pk) = 1
        // With σ = G2_gen, pk = G1_gen, H = G2_gen, this reduces to
        //   e(G2_gen, -G1_gen) · e(G2_gen, G1_gen) = e(G2_gen, identity) = 1.
        use crate::pairing::{G1Affine, G2Affine};
        let g1 = G1Affine::generator();
        let neg_g1 = g1.neg();
        let g2 = G2Affine::generator();

        let neg_g1_p = (neg_g1.x, neg_g1.y);
        let pk = (g1.x, g1.y);
        let sig: [(Fp, Fp); 2] = [(g2.x.c0, g2.x.c1), (g2.y.c0, g2.y.c1)];
        let msg_hash: [(Fp, Fp); 2] = sig;

        let (ops, out) = compile_bls_verify(sig, neg_g1_p, msg_hash, pk);

        use crate::nonnative_tower::{Fp12, Fp6};
        let one = [fp6_from_native(&Fp6::one()), fp6_from_native(&Fp6::zero())];
        assert_eq!(out, one, "BLS verify on identity equation must yield 1");
        assert_eq!(fp12_to_native(out), Fp12::one());

        // Cost: 2 Miller (~28k each) + 1 Fp12 mul (~225) + 1 final_exp
        // (~51k) ≈ 107k Fp ops, with exactly ONE FpOp::Inv (in easy part).
        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(inv_count, 1, "BLS verify must use exactly one FpOp::Inv");
        assert!(
            ops.len() > 80_000 && ops.len() < 150_000,
            "BLS verify op count {} outside expected ~107k range",
            ops.len(),
        );
    }

    #[test]
    fn compile_pairing_matches_native_addchain_on_generators() {
        // End-to-end: full `compile_pairing` on BLS12-381 generators must
        // equal native `easy + hard_part_addchain` composition (IETF
        // convention — matches blst's `blst_final_exp`, the Ethereum 2.0
        // pairing convention, etc.). Fast: both paths are ~80k Fp12 ops.
        use crate::pairing::{hard_part_addchain, miller_loop, G1Affine, G2Affine};
        let g1 = G1Affine::generator();
        let g2 = G2Affine::generator();

        let m = miller_loop(&g1, &g2);
        let f1 = m.conjugate();
        let f2 = m.invert().unwrap();
        let f3 = f1.mul(&f2);
        let f4 = f3.frobenius_map(2);
        let easy = f4.mul(&f3);
        let expected = hard_part_addchain(&easy);

        let p = (g1.x, g1.y);
        let q = [(g2.x.c0, g2.x.c1), (g2.y.c0, g2.y.c1)];
        let (_, out) = compile_pairing(p, q);
        assert_eq!(
            out,
            fp12_from_native(&expected),
            "compile_pairing(g1, g2) diverges from native easy + hard_part_addchain",
        );
    }

    #[test]
    fn final_exp_easy_matches_reference() {
        // Reference closed form lifted from `pairing::easy_part_agrees_with_closed_form`:
        //   f1 = f.conjugate(); f2 = f.invert(); f3 = f1·f2;
        //   f4 = f3.frobenius_map(2); easy = f4·f3.
        // Cross-check the in-circuit compile_final_exp_easy against this.
        use crate::nonnative_tower::Fp12;
        let a_lo = fp6_from_tuple([(3, 5), (7, 11), (13, 17)]);
        let a_hi = fp6_from_tuple([(19, 23), (29, 31), (37, 41)]);
        let f = fp12_to_native([a_lo, a_hi]);

        let f1 = f.conjugate();
        let f2 = f.invert().unwrap();
        let f3 = f1.mul(&f2);
        let f4 = f3.frobenius_map(2);
        let expected = f4.mul(&f3);

        let (_, out) = compile_final_exp_easy([a_lo, a_hi]);
        assert_eq!(
            out,
            fp12_from_native(&expected),
            "easy-part output must match reference closed form",
        );

        // Algebraic invariant: easy_part(f) lives in ker(x ↦ x^{p⁶+1}),
        // i.e. conjugate(easy) · easy = 1. The reference test in
        // pairing.rs proves this for the native value; here we verify the
        // compiled output respects it too.
        let easy_native = Fp12 {
            c0: fp6_to_native(out[0]),
            c1: fp6_to_native(out[1]),
        };
        let check = easy_native.conjugate().mul(&easy_native);
        assert_eq!(
            check,
            Fp12::one(),
            "compile_final_exp_easy output not in kernel of x^{{p⁶+1}}",
        );
    }

    /// Native reference: f^|x| via the same left-to-right binary square-
    /// and-multiply that `compile_fp12_pow_by_x` mirrors. Used inside tests
    /// only.
    fn fp12_pow_u64_ref(base: &crate::nonnative_tower::Fp12, exp: u64) -> crate::nonnative_tower::Fp12 {
        use crate::nonnative_tower::Fp12;
        if exp == 0 {
            return Fp12::one();
        }
        let msb = 63i32 - exp.leading_zeros() as i32;
        let mut acc = *base;
        for i in (0..msb).rev() {
            acc = acc.square();
            if ((exp >> i) & 1) == 1 {
                acc = acc.mul(base);
            }
        }
        acc
    }

    #[test]
    fn fp12_pow_wide_matches_reference_on_small_exp() {
        // Cross-check compile_fp12_pow_wide against fp12_pow_u64_ref on a
        // small exponent — keeps the test fast while exercising the binary
        // square-and-multiply chain end to end.
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let base_native = fp12_to_native([a_lo, a_hi]);

        for &exp in &[0u64, 1, 2, 5, 17, 0xC0FFEE, 0xDEAD_BEEF] {
            let expected = fp12_pow_u64_ref(&base_native, exp);
            let (_, out) = compile_fp12_pow_wide([a_lo, a_hi], &[exp]);
            assert_eq!(
                out,
                fp12_from_native(&expected),
                "compile_fp12_pow_wide diverges from reference at exp = {:#x}",
                exp,
            );
        }
    }

    #[test]
    fn fp12_pow_by_x_op_count_matches_chain_structure() {
        // |x| = MILLER_X_ABS has 6 bits set across 64 bits, so
        // square-and-multiply emits 63 squares + 5 multiplies, plus 1
        // conjugate (6 Fp Subs).
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);

        // First lift to the cyclotomic subgroup so the conjugate-as-inverse
        // step is meaningful (though op count itself doesn't depend on this).
        let (_, m) = compile_final_exp_easy([a_lo, a_hi]);
        let (ops, _) = compile_fp12_pow_by_x(m);

        // No FpOp::Inv inside the exp chain — the conjugate handles the
        // negative-x correction algebraically.
        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(
            inv_count, 0,
            "compile_fp12_pow_by_x must use zero FpOp::Inv (got {})",
            inv_count,
        );

        // Sanity-bound the total op count — must be near 11k.
        assert!(
            ops.len() > 9_000 && ops.len() < 14_000,
            "exp-by-x op count {} outside expected range",
            ops.len(),
        );
    }

    #[test]
    fn fp12_pow_by_x_matches_reference_on_cyclotomic_input() {
        // Take an arbitrary Fp12, lift it into the cyclotomic subgroup via
        // the easy part, then compare compile_fp12_pow_by_x to the native
        // closed form `(m^|x|).conjugate()` (since x < 0).
        let a_lo = fp6_from_tuple([(2, 3), (5, 7), (11, 13)]);
        let a_hi = fp6_from_tuple([(17, 19), (23, 29), (31, 37)]);
        let (_, m_compiled) = compile_final_exp_easy([a_lo, a_hi]);
        let m_native = fp12_to_native(m_compiled);

        let m_pow_x_native = fp12_pow_u64_ref(&m_native, MILLER_X_ABS).conjugate();
        let (_, m_pow_x_compiled) = compile_fp12_pow_by_x(m_compiled);

        assert_eq!(
            m_pow_x_compiled,
            fp12_from_native(&m_pow_x_native),
            "compile_fp12_pow_by_x diverges from native (m^|x|)·conj on cyclotomic input",
        );
    }

    #[test]
    fn final_exp_easy_uses_one_fp_inv() {
        // The whole easy part bottoms out at the single Fp inverse inside
        // compile_fp12_invert(f).
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let (ops, _) = compile_final_exp_easy([a_lo, a_hi]);
        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(
            inv_count, 1,
            "easy part must use exactly one FpOp::Inv (got {})",
            inv_count,
        );
    }

    #[test]
    fn fp12_invert_uses_exactly_one_fp_inv() {
        // The whole Fp12 inverse must compile to exactly one FpOp::Inv.
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let (ops, _) = compile_fp12_invert([a_lo, a_hi]);
        let inv_count = ops
            .iter()
            .filter(|op| matches!(op, FpOp::Inv { .. }))
            .count();
        assert_eq!(
            inv_count, 1,
            "Fp12 invert must use exactly one FpOp::Inv (got {})",
            inv_count,
        );
    }

    #[test]
    fn fp12_frobenius_is_homomorphism_over_product() {
        // π^k(a · b) == π^k(a) · π^k(b) — fundamental Frobenius property.
        let a_lo = fp6_from_tuple([(1, 2), (3, 4), (5, 6)]);
        let a_hi = fp6_from_tuple([(7, 8), (9, 10), (11, 12)]);
        let b_lo = fp6_from_tuple([(13, 14), (15, 16), (17, 18)]);
        let b_hi = fp6_from_tuple([(19, 20), (21, 22), (23, 24)]);

        // Test power 1 only — keeps the test fast, identity is what matters.
        let (_, ab) = compile_fp12_mul([a_lo, a_hi], [b_lo, b_hi]);
        let (_, lhs) = compile_fp12_frobenius_map(ab, 1);
        let (_, fa) = compile_fp12_frobenius_map([a_lo, a_hi], 1);
        let (_, fb) = compile_fp12_frobenius_map([b_lo, b_hi], 1);
        let (_, rhs) = compile_fp12_mul(fa, fb);
        assert_eq!(lhs, rhs, "π is not a ring homomorphism on Fp12");
    }
}
