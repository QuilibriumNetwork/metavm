//! Bit-level AIR scaffold for the SHA-512 compression function.
//!
//! This is the SHA-512 counterpart of [`crate::sha256_air`], operating on
//! **64-bit** words instead of 32-bit. SHA-512 is used by:
//! - **BLS12-381 `hash_to_field`** (RFC 9380 §5.3) — required to algebraically
//!   close `hash_to_curve` for BLS signature verification (currently
//!   host-side oracle).
//! - **Ed25519** — `H(R || A || M)` challenge inside the signature
//!   verification equation `[8s]B = [8]R + [8k]A`.
//!
//! # Differences from SHA-256
//!
//! - Words are 64-bit (8 bytes each); state is 8 × 64 = 512 bits.
//! - 80 rounds (vs 64), 80 round constants `K_512[t]` (vs 64).
//! - 128-byte block (vs 64-byte); message schedule has 80 words W[0..80].
//! - Rotation amounts in σ/Σ change:
//!     σ0(x) = ROTR_1(x)  XOR ROTR_8(x)  XOR SHR_7(x)
//!     σ1(x) = ROTR_19(x) XOR ROTR_61(x) XOR SHR_6(x)
//!     Σ0(x) = ROTR_28(x) XOR ROTR_34(x) XOR ROTR_39(x)
//!     Σ1(x) = ROTR_14(x) XOR ROTR_18(x) XOR ROTR_41(x)
//! - Padding length field is 128-bit big-endian (vs 64-bit).
//!
//! # Scaffold scope (this task)
//!
//! 1. Host-side reference SHA-512 (`sha512`, `sha512_compress`,
//!    `sha512_witness`) — used to derive AIR witness data.
//! 2. AIR column layout for one round (per-bit decomposition of all
//!    working variables + helpers), mirroring the SHA-256 layout. **The
//!    full 80-round expansion is documented as a deferred follow-up** —
//!    this module pins the scaffold (constants, types, column offsets,
//!    one-round populator) so the algebraic constraint construction can
//!    proceed incrementally.
//! 3. Witness population for a single round.
//! 4. Known-answer tests for the empty string + a few standard vectors.
//!
//! # Deferred follow-ups
//!
//! - Full `sha512_constraints` module (row-local Σ/σ/Ch/Maj algebra,
//!   round transition, cross-block chaining).
//! - Message-schedule recurrence binding for t ∈ 16..80.
//! - Per-invocation byte aggregator + cross-AIR LogUp linkage to
//!   `hash_to_field` / Ed25519 verification AIRs.
//! - Carry-column widths for the 64-bit add (T₁ sum has 5 terms each
//!   ≤ 2^64−1 → carry needs 3 bits, same as SHA-256 case).

use crate::field::{CurveType, Scalar};

// ──── Constants ───────────────────────────────────────────────────────

/// Number of compression rounds per block.
pub const NUM_ROUNDS: usize = 80;

/// Block size in bytes (= 1024 bits).
pub const BLOCK_BYTES: usize = 128;

/// Bits per 64-bit word.
pub const BITS_PER_WORD: usize = 64;

/// Number of working variables a..h.
pub const NUM_WORKING_VARS: usize = 8;

/// Total bits per state (8 × 64).
pub const BITS_PER_STATE: usize = NUM_WORKING_VARS * BITS_PER_WORD;

/// Initial hash values H(0): first 64 bits of the fractional parts of
/// the square roots of the first 8 primes (2..19). FIPS 180-4 §5.3.5.
pub const INITIAL_HASH: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// Round constants K_512[t]: first 64 bits of the fractional parts of the
/// cube roots of the first 80 primes (2..409). FIPS 180-4 §4.2.3.
pub const ROUND_CONSTANTS: [u64; NUM_ROUNDS] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

// ──── Host-side reference SHA-512 ────────────────────────────────────

#[inline]
fn ch(x: u64, y: u64, z: u64) -> u64 {
    (x & y) ^ (!x & z)
}

#[inline]
fn maj(x: u64, y: u64, z: u64) -> u64 {
    (x & y) ^ (x & z) ^ (y & z)
}

#[inline]
pub fn big_sigma0(x: u64) -> u64 {
    x.rotate_right(28) ^ x.rotate_right(34) ^ x.rotate_right(39)
}

#[inline]
pub fn big_sigma1(x: u64) -> u64 {
    x.rotate_right(14) ^ x.rotate_right(18) ^ x.rotate_right(41)
}

#[inline]
pub fn small_sigma0(x: u64) -> u64 {
    x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7)
}

#[inline]
pub fn small_sigma1(x: u64) -> u64 {
    x.rotate_right(19) ^ x.rotate_right(61) ^ (x >> 6)
}

/// Expand a 1024-bit block into the 80-word message schedule W[0..80].
/// Words 0..16 are the block parsed as big-endian u64; the remainder are
/// derived via W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16].
fn message_schedule(block: &[u8; BLOCK_BYTES]) -> [u64; NUM_ROUNDS] {
    let mut w = [0u64; NUM_ROUNDS];
    for i in 0..16 {
        w[i] = u64::from_be_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
    }
    for t in 16..NUM_ROUNDS {
        w[t] = small_sigma1(w[t - 2])
            .wrapping_add(w[t - 7])
            .wrapping_add(small_sigma0(w[t - 15]))
            .wrapping_add(w[t - 16]);
    }
    w
}

/// Apply one SHA-512 block compression to `state` using the 128-byte `block`.
pub fn sha512_compress(state: &mut [u64; 8], block: &[u8; BLOCK_BYTES]) {
    let w = message_schedule(block);

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for t in 0..NUM_ROUNDS {
        let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(ROUND_CONSTANTS[t])
            .wrapping_add(w[t]);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Compute SHA-512 of `input` with standard Merkle–Damgård padding.
/// Padding rule: append 0x80, then zero bytes, then the **128-bit
/// big-endian bit length** (high 64 bits zero for `input.len() <
/// 2^61`), so that the total length is a multiple of 128 bytes.
pub fn sha512(input: &[u8]) -> [u8; 64] {
    let mut state = INITIAL_HASH;

    let mut offset = 0;
    while input.len() - offset >= BLOCK_BYTES {
        let block: &[u8; BLOCK_BYTES] =
            input[offset..offset + BLOCK_BYTES].try_into().unwrap();
        sha512_compress(&mut state, block);
        offset += BLOCK_BYTES;
    }

    let remaining = &input[offset..];
    let bit_len = (input.len() as u128).wrapping_mul(8);

    // Worst case: tail spans two 128-byte blocks (1 + 16 = 17 bytes
    // minimum overhead).
    let mut tail = [0u8; 256];
    tail[..remaining.len()].copy_from_slice(remaining);
    tail[remaining.len()] = 0x80;

    let pad_to_blocks = if remaining.len() + 1 + 16 <= BLOCK_BYTES {
        1
    } else {
        2
    };
    let total = pad_to_blocks * BLOCK_BYTES;
    tail[total - 16..total].copy_from_slice(&bit_len.to_be_bytes());

    for i in 0..pad_to_blocks {
        let block: &[u8; BLOCK_BYTES] =
            tail[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES].try_into().unwrap();
        sha512_compress(&mut state, block);
    }

    let mut out = [0u8; 64];
    for (i, word) in state.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Snapshot of a single SHA-512 round; consumed by the AIR populator.
#[derive(Debug, Clone)]
pub struct RoundTrace {
    pub round: usize,
    /// Message-schedule word W[t].
    pub w: u64,
    /// Round constant K_512[t].
    pub k: u64,
    /// Working vars (a,b,c,d,e,f,g,h) before this round.
    pub before: [u64; 8],
    /// Σ₁(e).
    pub big_sigma1_e: u64,
    /// Ch(e,f,g).
    pub ch_efg: u64,
    /// Σ₀(a).
    pub big_sigma0_a: u64,
    /// Maj(a,b,c).
    pub maj_abc: u64,
    /// T₁ = h + Σ₁(e) + Ch + K + W (mod 2^64).
    pub t1: u64,
    /// T₂ = Σ₀(a) + Maj (mod 2^64).
    pub t2: u64,
    /// Working vars after this round.
    pub after: [u64; 8],
}

/// One block's worth of compression-function witness data.
#[derive(Debug, Clone)]
pub struct BlockTrace {
    pub block: [u8; BLOCK_BYTES],
    pub state_in: [u64; 8],
    pub state_out: [u64; 8],
    pub rounds: Vec<RoundTrace>,
}

/// Full witness for `sha512(input)`.
#[derive(Debug, Clone)]
pub struct HashTrace {
    pub input_len: usize,
    pub blocks: Vec<BlockTrace>,
    pub digest: [u8; 64],
}

/// Per-round witness producer: replays `sha512_compress` and records
/// every round's intermediate values.
fn sha512_compress_witness(
    state_in: [u64; 8],
    block: [u8; BLOCK_BYTES],
) -> Vec<RoundTrace> {
    let w = message_schedule(&block);
    let mut a = state_in[0];
    let mut b = state_in[1];
    let mut c = state_in[2];
    let mut d = state_in[3];
    let mut e = state_in[4];
    let mut f = state_in[5];
    let mut g = state_in[6];
    let mut h = state_in[7];
    let mut rounds = Vec::with_capacity(NUM_ROUNDS);
    for t in 0..NUM_ROUNDS {
        let before = [a, b, c, d, e, f, g, h];
        let s1 = big_sigma1(e);
        let chv = ch(e, f, g);
        let s0 = big_sigma0(a);
        let mjv = maj(a, b, c);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(chv)
            .wrapping_add(ROUND_CONSTANTS[t])
            .wrapping_add(w[t]);
        let t2 = s0.wrapping_add(mjv);
        let new_h = g;
        let new_g = f;
        let new_f = e;
        let new_e = d.wrapping_add(t1);
        let new_d = c;
        let new_c = b;
        let new_b = a;
        let new_a = t1.wrapping_add(t2);
        let after = [new_a, new_b, new_c, new_d, new_e, new_f, new_g, new_h];
        rounds.push(RoundTrace {
            round: t,
            w: w[t],
            k: ROUND_CONSTANTS[t],
            before,
            big_sigma1_e: s1,
            ch_efg: chv,
            big_sigma0_a: s0,
            maj_abc: mjv,
            t1,
            t2,
            after,
        });
        a = new_a;
        b = new_b;
        c = new_c;
        d = new_d;
        e = new_e;
        f = new_f;
        g = new_g;
        h = new_h;
    }
    rounds
}

/// Produce a full-hash witness for `sha512(input)`.
pub fn sha512_witness(input: &[u8]) -> HashTrace {
    let mut state = INITIAL_HASH;
    let mut blocks: Vec<BlockTrace> = Vec::new();

    let mut offset = 0;
    while input.len() - offset >= BLOCK_BYTES {
        let block: [u8; BLOCK_BYTES] =
            input[offset..offset + BLOCK_BYTES].try_into().unwrap();
        let state_in = state;
        let rounds = sha512_compress_witness(state_in, block);
        sha512_compress(&mut state, &block);
        blocks.push(BlockTrace {
            block,
            state_in,
            state_out: state,
            rounds,
        });
        offset += BLOCK_BYTES;
    }

    let remaining = &input[offset..];
    let bit_len = (input.len() as u128).wrapping_mul(8);
    let mut tail = [0u8; 256];
    tail[..remaining.len()].copy_from_slice(remaining);
    tail[remaining.len()] = 0x80;
    let pad_to_blocks = if remaining.len() + 1 + 16 <= BLOCK_BYTES {
        1
    } else {
        2
    };
    let total = pad_to_blocks * BLOCK_BYTES;
    tail[total - 16..total].copy_from_slice(&bit_len.to_be_bytes());
    for i in 0..pad_to_blocks {
        let block: [u8; BLOCK_BYTES] =
            tail[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES].try_into().unwrap();
        let state_in = state;
        let rounds = sha512_compress_witness(state_in, block);
        sha512_compress(&mut state, &block);
        blocks.push(BlockTrace {
            block,
            state_in,
            state_out: state,
            rounds,
        });
    }

    let mut digest = [0u8; 64];
    for (i, word) in state.iter().enumerate() {
        digest[i * 8..i * 8 + 8].copy_from_slice(&word.to_be_bytes());
    }

    HashTrace {
        input_len: input.len(),
        blocks,
        digest,
    }
}

// ──── AIR column layout ───────────────────────────────────────────────
//
// Mirrors `sha256_air` with all word widths doubled. One row per
// SHA-512 round. 80 rounds per block; a multi-block hash produces
// `blocks.len() * 80` rows.
//
// All bit columns hold 0/1 scalar values. Column index helpers map
// (working-var, bit) → column index.
//
// ┌─ index range ───────────┬──── size ─┬── meaning ───────────────────
// │ 0..512                  │ 512       │ BEFORE (a..h)
// │ 512..1024               │ 512       │ AFTER  (a'..h')
// │ 1024..1088              │ 64        │ W
// │ 1088..1152              │ 64        │ K
// │ 1152..1216              │ 64        │ Σ₀(a)
// │ 1216..1280              │ 64        │ aux: rot28(a) XOR rot34(a)
// │ 1280..1344              │ 64        │ Σ₁(e)
// │ 1344..1408              │ 64        │ aux: rot14(e) XOR rot18(e)
// │ 1408..1472              │ 64        │ aux: e AND f
// │ 1472..1536              │ 64        │ aux: (NOT e) AND g
// │ 1536..1600              │ 64        │ Ch(e,f,g)
// │ 1600..1664              │ 64        │ aux: a AND b
// │ 1664..1728              │ 64        │ aux: a AND c
// │ 1728..1792              │ 64        │ aux: b AND c
// │ 1792..1856              │ 64        │ aux: ab XOR ac
// │ 1856..1920              │ 64        │ Maj(a,b,c)
// │ 1920..1923              │ 3         │ T₁ carry (5-term sum, ≤2^66)
// │ 1923..1924              │ 1         │ T₂ carry (2-term sum)
// │ 1924..1925              │ 1         │ new-a carry
// │ 1925..1926              │ 1         │ new-e carry
// │ 1926..1990              │ 64        │ σ0(W) two-stage XOR aux
// │ 1990..2054              │ 64        │ σ0(W)
// │ 2054..2118              │ 64        │ σ1(W) two-stage XOR aux
// │ 2118..2182              │ 64        │ σ1(W)
// │ 2182..2184              │ 2         │ W[t] recurrence carry (2-bit)
// │ 2184..NUM_DATA_COLUMNS  │           │ (end of data columns)
// │ ...                     │ 80        │ SEL_ROUND[t]
// └─────────────────────────┴───────────┴──────────────────────────────
//
// T₁ carry width: 5 terms each ≤ 2^64−1 → sum ≤ 5·(2^64−1) < 5·2^64 <
// 2^66.5, so carry < 8 → 3 bits suffice (same as SHA-256).

pub const COL_BEFORE_OFFSET:        usize = 0;
pub const COL_AFTER_OFFSET:         usize = COL_BEFORE_OFFSET + BITS_PER_STATE;
pub const COL_W_OFFSET:             usize = COL_AFTER_OFFSET + BITS_PER_STATE;
pub const COL_K_OFFSET:             usize = COL_W_OFFSET + BITS_PER_WORD;
pub const COL_BIG_SIGMA0_A_OFFSET:  usize = COL_K_OFFSET + BITS_PER_WORD;
pub const COL_XOR01_S0_OFFSET:      usize = COL_BIG_SIGMA0_A_OFFSET + BITS_PER_WORD;
pub const COL_BIG_SIGMA1_E_OFFSET:  usize = COL_XOR01_S0_OFFSET + BITS_PER_WORD;
pub const COL_XOR01_S1_OFFSET:      usize = COL_BIG_SIGMA1_E_OFFSET + BITS_PER_WORD;
pub const COL_EF_OFFSET:            usize = COL_XOR01_S1_OFFSET + BITS_PER_WORD;
pub const COL_NOT_E_G_OFFSET:       usize = COL_EF_OFFSET + BITS_PER_WORD;
pub const COL_CH_EFG_OFFSET:        usize = COL_NOT_E_G_OFFSET + BITS_PER_WORD;
pub const COL_AB_OFFSET:            usize = COL_CH_EFG_OFFSET + BITS_PER_WORD;
pub const COL_AC_OFFSET:            usize = COL_AB_OFFSET + BITS_PER_WORD;
pub const COL_BC_OFFSET:            usize = COL_AC_OFFSET + BITS_PER_WORD;
pub const COL_XOR_AB_AC_OFFSET:     usize = COL_BC_OFFSET + BITS_PER_WORD;
pub const COL_MAJ_ABC_OFFSET:       usize = COL_XOR_AB_AC_OFFSET + BITS_PER_WORD;

pub const T1_CARRY_BITS: usize = 3;
pub const COL_T1_CARRY_OFFSET:      usize = COL_MAJ_ABC_OFFSET + BITS_PER_WORD;
pub const COL_T2_CARRY_OFFSET:      usize = COL_T1_CARRY_OFFSET + T1_CARRY_BITS;
pub const COL_A_NEW_CARRY_OFFSET:   usize = COL_T2_CARRY_OFFSET + 1;
pub const COL_E_NEW_CARRY_OFFSET:   usize = COL_A_NEW_CARRY_OFFSET + 1;

/// σ0/σ1(W) helper columns: same pattern as SHA-256, widened to 64-bit.
pub const COL_XOR01_SS0_OFFSET:        usize = COL_E_NEW_CARRY_OFFSET + 1;
pub const COL_SMALL_SIGMA0_W_OFFSET:   usize = COL_XOR01_SS0_OFFSET + BITS_PER_WORD;
pub const COL_XOR01_SS1_OFFSET:        usize = COL_SMALL_SIGMA0_W_OFFSET + BITS_PER_WORD;
pub const COL_SMALL_SIGMA1_W_OFFSET:   usize = COL_XOR01_SS1_OFFSET + BITS_PER_WORD;

/// 2-bit carry for W[t] recurrence: 4 64-bit summands < 4·2^64 < 2^66,
/// carry fits in 2 bits.
pub const W_RECURRENCE_CARRY_BITS: usize = 2;
pub const COL_W_RECURRENCE_CARRY_OFFSET: usize =
    COL_SMALL_SIGMA1_W_OFFSET + BITS_PER_WORD;

pub const NUM_DATA_COLUMNS: usize =
    COL_W_RECURRENCE_CARRY_OFFSET + W_RECURRENCE_CARRY_BITS;

pub const COL_SEL_ROUND_OFFSET: usize = NUM_DATA_COLUMNS;
pub const NUM_SEL_ROUND: usize = NUM_ROUNDS; // 80

pub const NUM_SHA512_COLUMNS: usize = COL_SEL_ROUND_OFFSET + NUM_SEL_ROUND;

/// Alias for [`NUM_SHA512_COLUMNS`] matching the standard AIR-module
/// convention used by `count_extended_constraints` in
/// `crate::ultimate_joint_prove`. Round 28–31 scaffold: 2264 cols.
pub const NUM_COLUMNS: usize = NUM_SHA512_COLUMNS;

/// Row-local algebraic constraints currently enforced by the SHA-512
/// scaffold. The full constraint system (round + selector binding,
/// working-var recurrence, Σ0/Σ1/Ch/Maj decomp closure, W-recurrence,
/// final-add carry chain) is deferred; this AIR contributes 0
/// row-constraints to `count_extended_constraints` until a
/// `Sha512ConstraintSystem` lands. The witness builder + KAT tests
/// already exist, so the scaffold can be upgraded incrementally.
pub const NUM_ROW_CONSTRAINTS: usize = 0;

/// Number of shifted (cross-row) constraints. Reserved 0 until the
/// hash-update chain that binds `after = before(ω·X)` is wired.
pub const NUM_SHIFTED: usize = 0;

// Convenience named indices for the working-variable positions.
pub const VAR_A: usize = 0;
pub const VAR_B: usize = 1;
pub const VAR_C: usize = 2;
pub const VAR_D: usize = 3;
pub const VAR_E: usize = 4;
pub const VAR_F: usize = 5;
pub const VAR_G: usize = 6;
pub const VAR_H: usize = 7;

// ──── Column index helpers ────────────────────────────────────────────

#[inline]
pub fn before(var: usize, bit: usize) -> usize {
    debug_assert!(var < NUM_WORKING_VARS && bit < BITS_PER_WORD);
    COL_BEFORE_OFFSET + var * BITS_PER_WORD + bit
}

#[inline]
pub fn after(var: usize, bit: usize) -> usize {
    debug_assert!(var < NUM_WORKING_VARS && bit < BITS_PER_WORD);
    COL_AFTER_OFFSET + var * BITS_PER_WORD + bit
}

#[inline]
pub fn w_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_W_OFFSET + bit
}

#[inline]
pub fn k_bit(bit: usize) -> usize {
    debug_assert!(bit < BITS_PER_WORD);
    COL_K_OFFSET + bit
}

#[inline]
pub fn sel_round(round: usize) -> usize {
    debug_assert!(round < NUM_SEL_ROUND);
    COL_SEL_ROUND_OFFSET + round
}

// ──── Witness population ──────────────────────────────────────────────

/// Allocate a trace shape with `num_rows` rows × `NUM_SHA512_COLUMNS`
/// columns, each cell zero.
pub fn alloc_trace(num_rows: usize, curve: CurveType) -> Vec<Vec<Scalar>> {
    let zero = Scalar::zero(curve);
    (0..NUM_SHA512_COLUMNS)
        .map(|_| vec![zero.clone(); num_rows])
        .collect()
}

fn write_word_bits(
    columns: &mut [Vec<Scalar>],
    row: usize,
    offset: usize,
    word: u64,
    one: &Scalar,
    zero: &Scalar,
) {
    for bit in 0..BITS_PER_WORD {
        let v = (word >> bit) & 1;
        columns[offset + bit][row] = if v == 1 { one.clone() } else { zero.clone() };
    }
}

fn write_working_bits(
    columns: &mut [Vec<Scalar>],
    row: usize,
    offset: usize,
    words: &[u64; 8],
    one: &Scalar,
    zero: &Scalar,
) {
    for (i, &w) in words.iter().enumerate() {
        write_word_bits(columns, row, offset + i * BITS_PER_WORD, w, one, zero);
    }
}

/// Populate a single row from a [`RoundTrace`]. Mirrors
/// [`crate::sha256_air::populate_round`] with widths doubled. Carry
/// columns are derived from `rt` directly using the same 5-term/2-term
/// decomposition.
pub fn populate_round(
    columns: &mut [Vec<Scalar>],
    row: usize,
    rt: &RoundTrace,
    curve: CurveType,
) {
    assert!(columns.len() == NUM_SHA512_COLUMNS, "columns has wrong shape");
    let one = Scalar::one(curve);
    let zero = Scalar::zero(curve);

    // BEFORE / AFTER working variables.
    write_working_bits(columns, row, COL_BEFORE_OFFSET, &rt.before, &one, &zero);
    write_working_bits(columns, row, COL_AFTER_OFFSET, &rt.after, &one, &zero);

    // W and K.
    write_word_bits(columns, row, COL_W_OFFSET, rt.w, &one, &zero);
    write_word_bits(columns, row, COL_K_OFFSET, rt.k, &one, &zero);

    // Σ₀(a) + its 2-stage XOR intermediate (rot28 XOR rot34).
    let a = rt.before[VAR_A];
    let xor01_s0 = a.rotate_right(28) ^ a.rotate_right(34);
    write_word_bits(columns, row, COL_XOR01_S0_OFFSET, xor01_s0, &one, &zero);
    write_word_bits(columns, row, COL_BIG_SIGMA0_A_OFFSET, rt.big_sigma0_a, &one, &zero);

    // Σ₁(e) + its 2-stage XOR intermediate (rot14 XOR rot18).
    let e = rt.before[VAR_E];
    let xor01_s1 = e.rotate_right(14) ^ e.rotate_right(18);
    write_word_bits(columns, row, COL_XOR01_S1_OFFSET, xor01_s1, &one, &zero);
    write_word_bits(columns, row, COL_BIG_SIGMA1_E_OFFSET, rt.big_sigma1_e, &one, &zero);

    // Ch intermediates: ef, not_e_g, ch_efg.
    let f = rt.before[VAR_F];
    let g = rt.before[VAR_G];
    let ef = e & f;
    let not_e_g = (!e) & g;
    write_word_bits(columns, row, COL_EF_OFFSET, ef, &one, &zero);
    write_word_bits(columns, row, COL_NOT_E_G_OFFSET, not_e_g, &one, &zero);
    write_word_bits(columns, row, COL_CH_EFG_OFFSET, rt.ch_efg, &one, &zero);

    // Maj intermediates: ab, ac, bc, xor_ab_ac, maj_abc.
    let b = rt.before[VAR_B];
    let c = rt.before[VAR_C];
    let ab = a & b;
    let ac = a & c;
    let bc = b & c;
    let xor_ab_ac = ab ^ ac;
    write_word_bits(columns, row, COL_AB_OFFSET, ab, &one, &zero);
    write_word_bits(columns, row, COL_AC_OFFSET, ac, &one, &zero);
    write_word_bits(columns, row, COL_BC_OFFSET, bc, &one, &zero);
    write_word_bits(columns, row, COL_XOR_AB_AC_OFFSET, xor_ab_ac, &one, &zero);
    write_word_bits(columns, row, COL_MAJ_ABC_OFFSET, rt.maj_abc, &one, &zero);

    // ── Carry columns ──
    // T₁ sum (5 terms ≤ 2^64−1): use u128 to avoid overflow.
    let h = rt.before[VAR_H];
    let t1_full_sum: u128 = (h as u128)
        + (rt.big_sigma1_e as u128)
        + (rt.ch_efg as u128)
        + (rt.k as u128)
        + (rt.w as u128);
    let t1_carry_value: u128 = t1_full_sum >> 64;
    debug_assert!(t1_carry_value < 8, "T1 carry fits in 3 bits");
    for bit in 0..T1_CARRY_BITS {
        let v = ((t1_carry_value >> bit) & 1) as u64;
        columns[COL_T1_CARRY_OFFSET + bit][row] =
            if v == 1 { one.clone() } else { zero.clone() };
    }

    // T₂ sum (2 terms).
    let t2_full_sum: u128 = (rt.big_sigma0_a as u128) + (rt.maj_abc as u128);
    let t2_carry_value: u128 = t2_full_sum >> 64;
    debug_assert!(t2_carry_value <= 1);
    columns[COL_T2_CARRY_OFFSET][row] =
        if t2_carry_value == 1 { one.clone() } else { zero.clone() };

    // new-a = T₁ + T₂ (2 terms, 1-bit carry).
    let new_a_sum: u128 = (rt.t1 as u128) + (rt.t2 as u128);
    let a_new_carry_value: u128 = new_a_sum >> 64;
    debug_assert!(a_new_carry_value <= 1);
    columns[COL_A_NEW_CARRY_OFFSET][row] =
        if a_new_carry_value == 1 { one.clone() } else { zero.clone() };

    // new-e = d + T₁.
    let d = rt.before[VAR_D];
    let new_e_sum: u128 = (d as u128) + (rt.t1 as u128);
    let e_new_carry_value: u128 = new_e_sum >> 64;
    debug_assert!(e_new_carry_value <= 1);
    columns[COL_E_NEW_CARRY_OFFSET][row] =
        if e_new_carry_value == 1 { one.clone() } else { zero.clone() };

    // ── σ0/σ1(W) helpers for this row's W ──
    // σ0(W) = ROTR_1(W)  XOR ROTR_8(W)  XOR SHR_7(W)
    // σ1(W) = ROTR_19(W) XOR ROTR_61(W) XOR SHR_6(W)
    let w = rt.w;
    let small_sigma0_w = w.rotate_right(1) ^ w.rotate_right(8) ^ (w >> 7);
    let small_sigma1_w = w.rotate_right(19) ^ w.rotate_right(61) ^ (w >> 6);
    let xor01_ss0 = w.rotate_right(1) ^ w.rotate_right(8);
    let xor01_ss1 = w.rotate_right(19) ^ w.rotate_right(61);
    write_word_bits(columns, row, COL_XOR01_SS0_OFFSET, xor01_ss0, &one, &zero);
    write_word_bits(columns, row, COL_SMALL_SIGMA0_W_OFFSET, small_sigma0_w, &one, &zero);
    write_word_bits(columns, row, COL_XOR01_SS1_OFFSET, xor01_ss1, &one, &zero);
    write_word_bits(columns, row, COL_SMALL_SIGMA1_W_OFFSET, small_sigma1_w, &one, &zero);

    // One-hot round selector.
    columns[sel_round(rt.round)][row] = one.clone();
}

// ──── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a hex string into bytes. Helper for KAT inputs.
    fn unhex(s: &str) -> Vec<u8> {
        assert!(s.len() % 2 == 0);
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// FIPS 180-4 §B.3 / NIST KAT: SHA-512("") = empty-string digest.
    #[test]
    fn sha512_air_empty_string_kat() {
        let expected = unhex(
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
             47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
        );
        let got = sha512(&[]);
        assert_eq!(&got[..], &expected[..], "SHA-512(\"\") mismatch");
    }

    /// FIPS 180-4 §C.1: SHA-512("abc").
    #[test]
    fn sha512_air_abc_kat() {
        let expected = unhex(
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        );
        let got = sha512(b"abc");
        assert_eq!(&got[..], &expected[..], "SHA-512(\"abc\") mismatch");
    }

    /// FIPS 180-4 §C.2: 896-bit message that crosses one block boundary.
    #[test]
    fn sha512_air_two_block_kat() {
        let msg = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno\
                    ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
        let expected = unhex(
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018\
             501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
        );
        let got = sha512(msg);
        assert_eq!(&got[..], &expected[..], "two-block KAT mismatch");
    }

    /// Cross-check: `sha512_witness(input).digest == sha512(input)`.
    #[test]
    fn sha512_witness_matches_sha512() {
        for input in [&b""[..], &b"a"[..], &b"abc"[..], &[0x42u8; 200][..]] {
            let direct = sha512(input);
            let wit = sha512_witness(input);
            assert_eq!(direct, wit.digest, "witness vs direct mismatch");
            // Cross-check chaining: each block's state_out must thread to
            // the next block's state_in.
            for i in 1..wit.blocks.len() {
                assert_eq!(
                    wit.blocks[i - 1].state_out,
                    wit.blocks[i].state_in,
                    "chaining broken between blocks {} and {}",
                    i - 1,
                    i
                );
            }
        }
    }

    /// Sanity: σ/Σ helper functions match their canonical definitions.
    #[test]
    fn sha512_air_sigma_helpers() {
        // Known value cross-checks: pick a fixed non-trivial word.
        let x: u64 = 0x0123_4567_89ab_cdef;
        assert_eq!(
            big_sigma0(x),
            x.rotate_right(28) ^ x.rotate_right(34) ^ x.rotate_right(39),
        );
        assert_eq!(
            big_sigma1(x),
            x.rotate_right(14) ^ x.rotate_right(18) ^ x.rotate_right(41),
        );
        assert_eq!(small_sigma0(x), x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7));
        assert_eq!(small_sigma1(x), x.rotate_right(19) ^ x.rotate_right(61) ^ (x >> 6));
    }

    /// AIR-shape sanity: column count, constant widths, populate one
    /// round and assert: (i) all written cells are 0 or 1; (ii) bit
    /// columns recover the input word; (iii) selector column is one-hot.
    #[test]
    fn sha512_air_populate_round_smoke() {
        let curve = CurveType::Bls12381;
        let wit = sha512_witness(b"abc");
        let first_round = &wit.blocks[0].rounds[0];

        // Layout sanity.
        assert_eq!(BITS_PER_STATE, 512);
        assert_eq!(NUM_ROUNDS, 80);
        assert_eq!(BLOCK_BYTES, 128);
        assert_eq!(NUM_SEL_ROUND, 80);

        // Populate row 0 of a fresh trace.
        let mut cols = alloc_trace(NUM_ROUNDS, curve);
        populate_round(&mut cols, 0, first_round, curve);

        // All bit columns are 0 or 1.
        let zero = Scalar::zero(curve);
        let one = Scalar::one(curve);
        for col in 0..NUM_DATA_COLUMNS {
            let v = &cols[col][0];
            let is_zero = v.to_bytes() == zero.to_bytes();
            let is_one = v.to_bytes() == one.to_bytes();
            assert!(
                is_zero || is_one,
                "bit column {} on row 0 is neither 0 nor 1",
                col,
            );
        }

        // Recover `a` from BEFORE bit columns and check it matches.
        let mut recovered_a: u64 = 0;
        for bit in 0..BITS_PER_WORD {
            let idx = before(VAR_A, bit);
            if cols[idx][0].to_bytes() == one.to_bytes() {
                recovered_a |= 1u64 << bit;
            }
        }
        assert_eq!(recovered_a, first_round.before[VAR_A], "a-bit decoding wrong");

        // Recover W and check.
        let mut recovered_w: u64 = 0;
        for bit in 0..BITS_PER_WORD {
            if cols[w_bit(bit)][0].to_bytes() == one.to_bytes() {
                recovered_w |= 1u64 << bit;
            }
        }
        assert_eq!(recovered_w, first_round.w, "W-bit decoding wrong");

        // Selector for round 0 is set to one on row 0; others zero.
        for t in 0..NUM_SEL_ROUND {
            let want = if t == 0 { &one } else { &zero };
            assert_eq!(
                cols[sel_round(t)][0].to_bytes(),
                want.to_bytes(),
                "selector column for round {} on row 0",
                t,
            );
        }

        // Column count is the published constant.
        assert_eq!(cols.len(), NUM_SHA512_COLUMNS);
    }

    /// Cross-check Ch / Maj round-by-round on the first 8 rounds of the
    /// "abc" trace: ensures the AIR-side round trace agrees with the
    /// canonical Ch/Maj definitions per round.
    #[test]
    fn sha512_air_first_8_rounds_consistency() {
        let wit = sha512_witness(b"abc");
        let rounds = &wit.blocks[0].rounds;
        for t in 0..8 {
            let r = &rounds[t];
            let [a, b, c, _d, e, f, g, _h] = r.before;
            assert_eq!(r.ch_efg, (e & f) ^ (!e & g), "ch mismatch round {}", t);
            assert_eq!(
                r.maj_abc,
                (a & b) ^ (a & c) ^ (b & c),
                "maj mismatch round {}",
                t,
            );
            assert_eq!(r.big_sigma0_a, big_sigma0(a), "Σ₀ mismatch round {}", t);
            assert_eq!(r.big_sigma1_e, big_sigma1(e), "Σ₁ mismatch round {}", t);
            assert_eq!(r.k, ROUND_CONSTANTS[t], "K mismatch round {}", t);
        }
    }

    /// Constants: IV / round-constant first/last entries match FIPS 180-4.
    #[test]
    fn sha512_air_constants() {
        assert_eq!(INITIAL_HASH[0], 0x6a09e667f3bcc908);
        assert_eq!(INITIAL_HASH[7], 0x5be0cd19137e2179);
        assert_eq!(ROUND_CONSTANTS[0], 0x428a2f98d728ae22);
        assert_eq!(ROUND_CONSTANTS[79], 0x6c44198c4a475817);
        assert_eq!(ROUND_CONSTANTS.len(), NUM_ROUNDS);
    }
}
