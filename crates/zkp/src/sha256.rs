//! SHA-256 compression function — reference implementation.
//!
//! Foundation for hashing circuits across the project:
//! - Ethereum consensus SSZ merkleization (beacon-chain state root, block root)
//! - Validator registry / randao / deposit Merkle trees
//! - Generic binary Merkle tree inclusion proofs over 32-byte leaves
//!
//! This module implements the raw SHA-256 compression function defined by
//! FIPS 180-4, plus a streaming wrapper that applies standard Merkle–Damgård
//! padding. The big-endian byte/word layout contrasts with Keccak's
//! little-endian lane layout — see [`crate::keccak`] for the latter.
//!
//! # AIR wiring (future)
//! The SHA-256 circuit will represent each round as a row containing the
//! 8 working variables (a..h) as 32-bit words, the message-schedule word
//! W[t], and witness columns for Σ₀/Σ₁/Ch/Maj/T₁/T₂. Bitwise AND / XOR /
//! rotations will be backed by a nibble-decomposition lookup table.

/// Number of compression rounds per block.
pub const NUM_ROUNDS: usize = 64;

/// Initial hash values H(0), the first 32 bits of the fractional parts of
/// the square roots of the first 8 primes (2..19). FIPS 180-4 §5.3.3.
pub const INITIAL_HASH: [u32; 8] = [
    0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
    0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
];

/// Round constants K[t], the first 32 bits of the fractional parts of the
/// cube roots of the first 64 primes (2..311). FIPS 180-4 §4.2.2.
pub const ROUND_CONSTANTS: [u32; NUM_ROUNDS] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5,
    0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
    0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
    0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
    0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc,
    0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
    0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
    0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
    0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3,
    0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5,
    0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
    0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
];

#[inline]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    // Choose: for each bit, pick y if x==1 else z.
    (x & y) ^ (!x & z)
}

#[inline]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    // Majority: 1 iff at least two of x,y,z are 1.
    (x & y) ^ (x & z) ^ (y & z)
}

#[inline]
fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

#[inline]
fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

#[inline]
fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

#[inline]
fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// Expand a 512-bit block into the 64-word message schedule W[0..64].
/// Words 0..16 are the block parsed as big-endian u32; the remainder are
/// derived via W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16].
fn message_schedule(block: &[u8; 64]) -> [u32; NUM_ROUNDS] {
    let mut w = [0u32; NUM_ROUNDS];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
    }
    for t in 16..NUM_ROUNDS {
        w[t] = small_sigma1(w[t - 2])
            .wrapping_add(w[t - 7])
            .wrapping_add(small_sigma0(w[t - 15]))
            .wrapping_add(w[t - 16]);
    }
    w
}

/// Apply one SHA-256 block compression to `state` using the 512-bit `block`.
///
/// Mutates `state` in place: on return it holds H(i+1) = H(i) + working-vars.
pub fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
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

/// Convenience: `sha256(left || right)` for merkle-style pair hashing.
/// Used by the SSZ merkleization primitives.
pub fn sha256_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    sha256(&buf)
}

/// Compute SHA-256 of `input` with standard Merkle–Damgård padding:
/// append 0x80, then zero bytes, then the 64-bit big-endian bit length,
/// so that the total length is a multiple of 64 bytes.
pub fn sha256(input: &[u8]) -> [u8; 32] {
    let mut state = INITIAL_HASH;

    // Absorb full 64-byte blocks.
    let mut offset = 0;
    while input.len() - offset >= 64 {
        let block: &[u8; 64] = input[offset..offset + 64].try_into().unwrap();
        sha256_compress(&mut state, block);
        offset += 64;
    }

    // Build the final padded block(s). We always need at least the 0x80
    // byte and 8 length bytes; if they don't fit in the current tail we
    // emit two blocks instead of one.
    let remaining = &input[offset..];
    let bit_len = (input.len() as u64).wrapping_mul(8);

    let mut tail = [0u8; 128];
    tail[..remaining.len()].copy_from_slice(remaining);
    tail[remaining.len()] = 0x80;

    let pad_to_blocks = if remaining.len() + 1 + 8 <= 64 { 1 } else { 2 };
    let total = pad_to_blocks * 64;
    tail[total - 8..total].copy_from_slice(&bit_len.to_be_bytes());

    for i in 0..pad_to_blocks {
        let block: &[u8; 64] = tail[i * 64..(i + 1) * 64].try_into().unwrap();
        sha256_compress(&mut state, block);
    }

    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// One block's worth of compression-function witness data: the raw 64-byte
/// block that was absorbed, the chaining state before and after, and the
/// per-round trace. The block-level AIR binds the chaining chain:
/// `blocks[i].state_in == blocks[i-1].state_out`, and the first block's
/// `state_in == INITIAL_HASH`.
#[derive(Debug, Clone)]
pub struct BlockTrace {
    /// Padded block actually fed into `sha256_compress`.
    pub block: [u8; 64],
    /// Chaining state before absorbing this block.
    pub state_in: [u32; 8],
    /// Chaining state after absorbing this block.
    pub state_out: [u32; 8],
    /// 64 per-round traces from `sha256_compress_witness`.
    pub rounds: Vec<RoundTrace>,
}

/// Witness for a full `sha256(input)` — all padded blocks, each with its
/// chaining-state pair and 64 round traces. The last block's `state_out`
/// serialized big-endian is the final 32-byte digest.
///
/// This is the AIR-shaped data for a SHA-256 proof: every input byte is
/// either in the original input or produced by the canonical padding rule
/// (0x80 terminator, zero fill, 8-byte big-endian bit length in the tail).
/// The AIR will constrain both the compression function (via RoundTrace)
/// and the padding derivation (which bytes are input, which are 0x80/0/len).
#[derive(Debug, Clone)]
pub struct HashTrace {
    /// Original pre-padding input length in bytes.
    pub input_len: usize,
    /// Padded blocks absorbed, in order.
    pub blocks: Vec<BlockTrace>,
    /// Final 32-byte digest (redundant with `blocks.last().state_out` but
    /// cached for convenience).
    pub digest: [u8; 32],
}

/// Produce a full-hash witness for `sha256(input)`.
///
/// Mirrors the padding and block absorption in `sha256()` exactly; cross-
/// checked by `test_hash_witness_matches_sha256` so drift in either
/// implementation is caught immediately.
pub fn sha256_witness(input: &[u8]) -> HashTrace {
    let mut state = INITIAL_HASH;
    let mut blocks: Vec<BlockTrace> = Vec::new();

    // Absorb full 64-byte blocks from the raw input.
    let mut offset = 0;
    while input.len() - offset >= 64 {
        let block: [u8; 64] = input[offset..offset + 64].try_into().unwrap();
        let state_in = state;
        let rounds = sha256_compress_witness(state_in, block);
        sha256_compress(&mut state, &block);
        blocks.push(BlockTrace {
            block,
            state_in,
            state_out: state,
            rounds,
        });
        offset += 64;
    }

    // Build the padded tail (one or two blocks).
    let remaining = &input[offset..];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut tail = [0u8; 128];
    tail[..remaining.len()].copy_from_slice(remaining);
    tail[remaining.len()] = 0x80;
    let pad_to_blocks = if remaining.len() + 1 + 8 <= 64 { 1 } else { 2 };
    let total = pad_to_blocks * 64;
    tail[total - 8..total].copy_from_slice(&bit_len.to_be_bytes());
    for i in 0..pad_to_blocks {
        let block: [u8; 64] = tail[i * 64..(i + 1) * 64].try_into().unwrap();
        let state_in = state;
        let rounds = sha256_compress_witness(state_in, block);
        sha256_compress(&mut state, &block);
        blocks.push(BlockTrace {
            block,
            state_in,
            state_out: state,
            rounds,
        });
    }

    let mut digest = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }

    HashTrace {
        input_len: input.len(),
        blocks,
        digest,
    }
}

/// Snapshot of a single SHA-256 round.
/// Consumed by the SHA-256 AIR to populate trace columns per round.
#[derive(Debug, Clone)]
pub struct RoundTrace {
    pub round: usize,
    /// Message-schedule word W[t] used by this round.
    pub w: u32,
    /// Round constant K[t].
    pub k: u32,
    /// Working variables (a,b,c,d,e,f,g,h) before this round.
    pub before: [u32; 8],
    /// Σ₁(e) computed from the pre-round e.
    pub big_sigma1_e: u32,
    /// Ch(e,f,g).
    pub ch_efg: u32,
    /// Σ₀(a) computed from the pre-round a.
    pub big_sigma0_a: u32,
    /// Maj(a,b,c).
    pub maj_abc: u32,
    /// T₁ = h + Σ₁(e) + Ch(e,f,g) + K[t] + W[t] (mod 2^32).
    pub t1: u32,
    /// T₂ = Σ₀(a) + Maj(a,b,c) (mod 2^32).
    pub t2: u32,
    /// Working variables after this round.
    pub after: [u32; 8],
}

/// Run one SHA-256 block compression and return a [`RoundTrace`] per round
/// with every intermediate value the AIR will need to constrain.
///
/// The returned trace has exactly [`NUM_ROUNDS`] (= 64) entries. The final
/// round's `after` values, added component-wise to the input `state`, give
/// the post-compression hash words.
pub fn sha256_compress_witness(state: [u32; 8], block: [u8; 64]) -> Vec<RoundTrace> {
    let w_sched = message_schedule(&block);
    let mut rounds = Vec::with_capacity(NUM_ROUNDS);

    let mut vars = state;
    for t in 0..NUM_ROUNDS {
        let before = vars;
        let [a, b, c, d, e, f, g, h] = vars;

        let big_sigma1_e = big_sigma1(e);
        let ch_efg = ch(e, f, g);
        let big_sigma0_a = big_sigma0(a);
        let maj_abc = maj(a, b, c);

        let t1 = h
            .wrapping_add(big_sigma1_e)
            .wrapping_add(ch_efg)
            .wrapping_add(ROUND_CONSTANTS[t])
            .wrapping_add(w_sched[t]);
        let t2 = big_sigma0_a.wrapping_add(maj_abc);

        let new_a = t1.wrapping_add(t2);
        let new_e = d.wrapping_add(t1);
        let after = [new_a, a, b, c, new_e, e, f, g];

        rounds.push(RoundTrace {
            round: t,
            w: w_sched[t],
            k: ROUND_CONSTANTS[t],
            before,
            big_sigma1_e,
            ch_efg,
            big_sigma0_a,
            maj_abc,
            t1,
            t2,
            after,
        });

        vars = after;
    }
    rounds
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a lowercase hex string into a byte vector (test-only helper).
    fn hex_to_bytes(s: &str) -> Vec<u8> {
        assert!(s.len() % 2 == 0, "odd-length hex");
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// The full-hash witness must reproduce `sha256(input)` exactly.
    #[test]
    fn test_hash_witness_matches_sha256() {
        // Three input sizes: empty, short (one padded block), long (two blocks).
        for input_len in [0usize, 3, 55, 56, 64, 100, 256] {
            let input: Vec<u8> = (0..input_len).map(|i| (i * 7 + 1) as u8).collect();
            let trace = sha256_witness(&input);
            assert_eq!(trace.input_len, input.len());
            assert_eq!(trace.digest, sha256(&input), "len={}", input_len);
            // Chain invariant: each block's state_in equals the previous
            // block's state_out (or INITIAL_HASH for the first).
            let mut expected_state_in = INITIAL_HASH;
            for b in &trace.blocks {
                assert_eq!(b.state_in, expected_state_in);
                expected_state_in = b.state_out;
                // Rounds internally consistent: after absorbing b.block, the
                // final round's chaining output + state_in component-wise
                // equals b.state_out.
                let last_after = b.rounds.last().expect("64 rounds").after;
                for i in 0..8 {
                    assert_eq!(
                        b.state_out[i],
                        b.state_in[i].wrapping_add(last_after[i]),
                        "block chain broken at word {}",
                        i
                    );
                }
            }
        }
    }

    /// Rate-boundary case: input exactly 56 bytes means padding spills into
    /// a second block (since 56 + 1 + 8 > 64), producing 2 blocks total.
    #[test]
    fn test_hash_witness_padding_spill() {
        let trace = sha256_witness(&vec![0xABu8; 56]);
        assert_eq!(trace.blocks.len(), 2, "56-byte input must produce 2 blocks");
        assert_eq!(trace.digest, sha256(&vec![0xABu8; 56]));
    }

    /// FIPS 180-4 / RFC 6234 test vector: sha256("").
    #[test]
    fn test_sha256_empty() {
        let h = sha256(b"");
        let expected =
            hex_to_bytes("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(h.to_vec(), expected);
    }

    /// FIPS 180-4 / RFC 6234 test vector: sha256("abc").
    #[test]
    fn test_sha256_abc() {
        let h = sha256(b"abc");
        let expected =
            hex_to_bytes("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(h.to_vec(), expected);
    }

    /// FIPS 180-4 / RFC 6234 test vector: the 448-bit "abcdbcde..." input.
    /// This forces the padding to spill into a second block.
    #[test]
    fn test_sha256_two_block_message() {
        let input = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(input.len(), 56);
        let h = sha256(input);
        let expected =
            hex_to_bytes("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        assert_eq!(h.to_vec(), expected);
    }

    /// A block-boundary case: input exactly 64 bytes → padding spills to
    /// a second block by itself.
    #[test]
    fn test_sha256_64_byte_input() {
        let input = vec![0x42u8; 64];
        let h = sha256(&input);
        let h63 = sha256(&vec![0x42u8; 63]);
        let h65 = sha256(&vec![0x42u8; 65]);
        assert_ne!(h, h63);
        assert_ne!(h, h65);
        assert_ne!(h63, h65);
    }

    /// Witness trace's final-round working variables, added to the initial
    /// state, must match the direct `sha256_compress` output word-for-word.
    #[test]
    fn test_witness_matches_direct_compression() {
        let state = INITIAL_HASH;
        // Padded block for input "abc" (length 24 bits).
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[63] = 24;

        let mut direct = state;
        sha256_compress(&mut direct, &block);

        let trace = sha256_compress_witness(state, block);
        assert_eq!(trace.len(), NUM_ROUNDS);
        assert_eq!(trace[0].before, state);

        let mut expected_final = state;
        for i in 0..8 {
            expected_final[i] = expected_final[i].wrapping_add(trace[NUM_ROUNDS - 1].after[i]);
        }
        assert_eq!(expected_final, direct);

        // Each round's `before` equals the previous round's `after`.
        for i in 1..NUM_ROUNDS {
            assert_eq!(trace[i].before, trace[i - 1].after);
        }
    }

    /// Ch/Maj/Σ₀/Σ₁/T₁/T₂ must satisfy their defining equations inside the
    /// witness, and the round state update must be consistent.
    #[test]
    fn test_witness_round_internal_consistency() {
        let state = INITIAL_HASH;
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[63] = 24;

        let trace = sha256_compress_witness(state, block);
        for rt in &trace {
            let [a, b, c, d, e, f, g, h] = rt.before;

            assert_eq!(rt.ch_efg, (e & f) ^ (!e & g), "round {}: Ch", rt.round);
            assert_eq!(
                rt.maj_abc,
                (a & b) ^ (a & c) ^ (b & c),
                "round {}: Maj",
                rt.round
            );
            assert_eq!(
                rt.big_sigma1_e,
                e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25),
                "round {}: Σ₁",
                rt.round
            );
            assert_eq!(
                rt.big_sigma0_a,
                a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22),
                "round {}: Σ₀",
                rt.round
            );

            let t1 = h
                .wrapping_add(rt.big_sigma1_e)
                .wrapping_add(rt.ch_efg)
                .wrapping_add(rt.k)
                .wrapping_add(rt.w);
            let t2 = rt.big_sigma0_a.wrapping_add(rt.maj_abc);
            assert_eq!(rt.t1, t1, "round {}: T₁", rt.round);
            assert_eq!(rt.t2, t2, "round {}: T₂", rt.round);

            // State update: (a',b',c',d',e',f',g',h') =
            // (T1+T2, a, b, c, d+T1, e, f, g).
            assert_eq!(rt.after[0], rt.t1.wrapping_add(rt.t2));
            assert_eq!(rt.after[1], a);
            assert_eq!(rt.after[2], b);
            assert_eq!(rt.after[3], c);
            assert_eq!(rt.after[4], d.wrapping_add(rt.t1));
            assert_eq!(rt.after[5], e);
            assert_eq!(rt.after[6], f);
            assert_eq!(rt.after[7], g);

            assert_eq!(rt.k, ROUND_CONSTANTS[rt.round]);
        }
    }

    /// Message-schedule words beyond index 15 satisfy
    /// W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16].
    #[test]
    fn test_message_schedule_recurrence() {
        // Arbitrary non-trivial block.
        let mut block = [0u8; 64];
        for (i, b) in block.iter_mut().enumerate() {
            *b = ((i as u32).wrapping_mul(31) ^ 0xA5) as u8;
        }
        let w = message_schedule(&block);
        for t in 16..NUM_ROUNDS {
            let expected = small_sigma1(w[t - 2])
                .wrapping_add(w[t - 7])
                .wrapping_add(small_sigma0(w[t - 15]))
                .wrapping_add(w[t - 16]);
            assert_eq!(w[t], expected, "W[{}] recurrence mismatch", t);
        }
        // First 16 words are the block parsed big-endian.
        for i in 0..16 {
            let expected =
                u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(w[i], expected);
        }
    }

    /// Ch/Maj satisfy their boolean-logic identities on sampled inputs.
    #[test]
    fn test_ch_maj_semantics() {
        let samples: &[(u32, u32, u32)] = &[
            (0, 0, 0),
            (u32::MAX, 0, 0),
            (0, u32::MAX, 0),
            (0, 0, u32::MAX),
            (0xAAAA_AAAA, 0x5555_5555, 0xF0F0_F0F0),
            (0x1234_5678, 0x9ABC_DEF0, 0xDEAD_BEEF),
        ];
        for &(x, y, z) in samples {
            // Ch picks y where x=1, else z.
            let expected_ch = (x & y) | (!x & z);
            assert_eq!(ch(x, y, z), expected_ch);

            // Maj equals 1 iff at least two of x,y,z are 1.
            let expected_maj = (x & y) | (x & z) | (y & z);
            assert_eq!(maj(x, y, z), expected_maj);
        }
    }

    #[test]
    fn test_round_constants_count() {
        assert_eq!(ROUND_CONSTANTS.len(), NUM_ROUNDS);
        assert_eq!(INITIAL_HASH.len(), 8);
    }

    /// Sanity: a one-block message ("abc") and the manual single-block
    /// call with the same padding produce identical digests.
    #[test]
    fn test_sha256_matches_single_block_compress() {
        let mut state = INITIAL_HASH;
        let mut block = [0u8; 64];
        block[..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[63] = 24;
        sha256_compress(&mut state, &block);

        let mut direct = [0u8; 32];
        for (i, word) in state.iter().enumerate() {
            direct[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        assert_eq!(direct, sha256(b"abc"));
    }
}
