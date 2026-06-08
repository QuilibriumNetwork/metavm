//! Keccak-f[1600] permutation — reference implementation.
//!
//! Foundation for hashing circuits across the project:
//! - EVM SHA3 / KECCAK256 opcode
//! - Ethereum block header / state-root / tx-root / receipts-root hashing
//! - Merkle Patricia Trie (MPT) node hashing (keccak256 on RLP)
//!
//! This module implements the permutation used by all keccak256 / SHA-3
//! variants. The absorb-squeeze sponge wrapper lives in [`keccak256`].
//!
//! # AIR wiring (future)
//! The Keccak circuit will represent the 1600-bit state as 25 × 64-bit
//! lanes across trace columns, with round-phase selectors for θ, ρ, π, χ, ι.
//! The bitwise XOR / AND / NOT operations of χ reuse the nibble-AND lookup
//! from `lookup.rs`.

/// Keccak-f[1600] state: 5 × 5 lanes of 64 bits.
pub type State = [[u64; 5]; 5];

/// Number of rounds in Keccak-f[1600].
pub const NUM_ROUNDS: usize = 24;

/// Round constants RC[i] XOR'd into A[0][0] during ι.
pub const ROUND_CONSTANTS: [u64; NUM_ROUNDS] = [
    0x0000_0000_0000_0001, 0x0000_0000_0000_8082,
    0x8000_0000_0000_808a, 0x8000_0000_8000_8000,
    0x0000_0000_0000_808b, 0x0000_0000_8000_0001,
    0x8000_0000_8000_8081, 0x8000_0000_0000_8009,
    0x0000_0000_0000_008a, 0x0000_0000_0000_0088,
    0x0000_0000_8000_8009, 0x0000_0000_8000_000a,
    0x0000_0000_8000_808b, 0x8000_0000_0000_008b,
    0x8000_0000_0000_8089, 0x8000_0000_0000_8003,
    0x8000_0000_0000_8002, 0x8000_0000_0000_0080,
    0x0000_0000_0000_800a, 0x8000_0000_8000_000a,
    0x8000_0000_8000_8081, 0x8000_0000_0000_8080,
    0x0000_0000_8000_0001, 0x8000_0000_8000_8008,
];

/// Left-rotation offsets for ρ, indexed as `RHO_OFFSETS[x][y]`.
pub const RHO_OFFSETS: [[u32; 5]; 5] = [
    [ 0, 36,  3, 41, 18],
    [ 1, 44, 10, 45,  2],
    [62,  6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39,  8, 14],
];

#[inline]
fn rotl(x: u64, n: u32) -> u64 {
    if n == 0 { x } else { x.rotate_left(n) }
}

/// θ: XOR each column's parity into every lane.
fn theta(a: &mut State) {
    let mut c = [0u64; 5];
    for x in 0..5 {
        c[x] = a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4];
    }
    let mut d = [0u64; 5];
    for x in 0..5 {
        d[x] = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1);
    }
    for x in 0..5 {
        for y in 0..5 {
            a[x][y] ^= d[x];
        }
    }
}

/// ρ: rotate each lane by `RHO_OFFSETS[x][y]`.
fn rho(a: &mut State) {
    for x in 0..5 {
        for y in 0..5 {
            a[x][y] = rotl(a[x][y], RHO_OFFSETS[x][y]);
        }
    }
}

/// π: permute lane positions so that `new[y][(2x+3y) mod 5] = a[x][y]`.
fn pi(a: &mut State) {
    let mut b = [[0u64; 5]; 5];
    for x in 0..5 {
        for y in 0..5 {
            b[y][(2 * x + 3 * y) % 5] = a[x][y];
        }
    }
    *a = b;
}

/// χ: A[x][y] = A[x][y] XOR ((NOT A[x+1][y]) AND A[x+2][y]).
fn chi(a: &mut State) {
    for y in 0..5 {
        let row: [u64; 5] = [a[0][y], a[1][y], a[2][y], a[3][y], a[4][y]];
        for x in 0..5 {
            a[x][y] = row[x] ^ ((!row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
    }
}

/// ι: XOR round constant into A[0][0].
fn iota(a: &mut State, round: usize) {
    a[0][0] ^= ROUND_CONSTANTS[round];
}

/// Apply the full Keccak-f[1600] permutation (24 rounds) to `state`.
pub fn keccak_f1600(state: &mut State) {
    for round in 0..NUM_ROUNDS {
        theta(state);
        rho(state);
        pi(state);
        chi(state);
        iota(state, round);
    }
}

/// State snapshot at each phase boundary within a single round.
/// Consumed by the Keccak AIR to populate trace columns per round.
#[derive(Debug, Clone)]
pub struct RoundTrace {
    pub round: usize,
    /// State before this round's θ.
    pub before: State,
    /// Column parities `C[x] = XOR_y A[x][y]` computed during θ.
    pub c: [u64; 5],
    /// `D[x] = C[x-1] XOR rotl(C[x+1], 1)` — θ's diffusion term.
    pub d: [u64; 5],
    /// State after θ (before ρ).
    pub after_theta: State,
    /// State after ρ (before π).
    pub after_rho: State,
    /// State after π (before χ).
    pub after_pi: State,
    /// State after χ (before ι).
    pub after_chi: State,
    /// State after ι — i.e. the round's output.
    pub after_iota: State,
}

/// Run Keccak-f[1600] and return a `RoundTrace` per round with every
/// intermediate state that the AIR will need to constrain.
pub fn keccak_f1600_witness(mut state: State) -> Vec<RoundTrace> {
    let mut rounds = Vec::with_capacity(NUM_ROUNDS);
    for round in 0..NUM_ROUNDS {
        let before = state;

        // θ: recompute C/D without mutating yet so we can capture them.
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1);
        }
        let mut after_theta = state;
        for x in 0..5 {
            for y in 0..5 {
                after_theta[x][y] ^= d[x];
            }
        }

        let mut after_rho = after_theta;
        rho(&mut after_rho);

        let mut after_pi = after_rho;
        pi(&mut after_pi);

        let mut after_chi = after_pi;
        chi(&mut after_chi);

        let mut after_iota = after_chi;
        iota(&mut after_iota, round);

        rounds.push(RoundTrace {
            round,
            before,
            c,
            d,
            after_theta,
            after_rho,
            after_pi,
            after_chi,
            after_iota,
        });

        state = after_iota;
    }
    rounds
}

/// Compute the keccak256 hash of `input` using the Ethereum variant
/// (rate = 1088 bits = 136 bytes, capacity = 512 bits, pad10*1 with
/// domain byte 0x01 — this is the pre-NIST Keccak used by Ethereum,
/// NOT SHA3-256 which uses 0x06).
pub fn keccak256(input: &[u8]) -> [u8; 32] {
    const RATE_BYTES: usize = 136;
    let mut state: State = [[0u64; 5]; 5];

    // Absorb full blocks.
    let mut offset = 0;
    while input.len() - offset >= RATE_BYTES {
        absorb_block(&mut state, &input[offset..offset + RATE_BYTES]);
        keccak_f1600(&mut state);
        offset += RATE_BYTES;
    }

    // Absorb final block with pad10*1 + 0x01 domain byte (Ethereum keccak256).
    let remaining = &input[offset..];
    let mut block = [0u8; RATE_BYTES];
    block[..remaining.len()].copy_from_slice(remaining);
    block[remaining.len()] ^= 0x01; // Ethereum keccak256 domain byte
    block[RATE_BYTES - 1] ^= 0x80;
    absorb_block(&mut state, &block);
    keccak_f1600(&mut state);

    // Squeeze 32 bytes.
    let mut out = [0u8; 32];
    for i in 0..4 {
        let lane = state[i][0];
        out[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
    }
    out
}

/// Per-block witness for the keccak256 sponge: the padded 136-byte block,
/// state before and after absorbing it, and the 24 round traces from
/// the post-absorb `keccak_f1600` permutation. The AIR binds
/// `blocks[i].state_in == blocks[i-1].state_out` and the first block's
/// `state_in == [[0u64; 5]; 5]`.
#[derive(Debug, Clone)]
pub struct BlockTrace {
    /// Padded 136-byte rate block that was absorbed.
    pub block: [u8; 136],
    /// State before the block was XOR'd in (i.e. after the previous
    /// permutation, or all-zero for the first block).
    pub state_in: State,
    /// State immediately after XORing the block lanes in (pre-permutation).
    pub state_after_absorb: State,
    /// State after the 24-round `keccak_f1600` permutation.
    pub state_out: State,
    /// 24 per-round traces for this block's permutation.
    pub rounds: Vec<RoundTrace>,
}

/// Full-hash witness for `keccak256(input)`. Contains every padded block
/// absorbed, the permutation trace for each, and the 32-byte output.
///
/// The AIR version will constrain:
/// 1. Padding derivation: the final block is built from `input[offset..]`
///    plus the 0x01 domain byte at position `input.len() mod 136` and the
///    0x80 terminator at position 135.
/// 2. Chain invariant: `blocks[i].state_in == blocks[i-1].state_out`.
/// 3. Absorb: `state_after_absorb.lane[i] == state_in.lane[i] XOR block.lane[i]`
///    for i in 0..17, and `== state_in.lane[i]` for i in 17..25 (capacity lanes).
/// 4. Permutation: `state_out == keccak_f1600(state_after_absorb)` via the
///    per-round `RoundTrace`s.
/// 5. Output: `digest[0..32]` equals the first 4 rate lanes of the final
///    block's `state_out`, little-endian.
#[derive(Debug, Clone)]
pub struct HashTrace {
    /// Original pre-padding input length in bytes.
    pub input_len: usize,
    /// Padded-and-absorbed blocks in order.
    pub blocks: Vec<BlockTrace>,
    /// Final 32-byte digest (redundant with the first 4 rate lanes of the
    /// last block's `state_out`, but cached for convenience).
    pub digest: [u8; 32],
}

/// Produce a full-hash witness for `keccak256(input)`.
///
/// Mirrors the absorb/pad logic in [`keccak256`] exactly; cross-checked by
/// `test_hash_witness_matches_keccak256` so drift in either implementation
/// is caught.
pub fn keccak_witness(input: &[u8]) -> HashTrace {
    const RATE_BYTES: usize = 136;
    let mut state: State = [[0u64; 5]; 5];
    let mut blocks: Vec<BlockTrace> = Vec::new();

    // Helper: absorb a block and return (state_after_absorb, state_out, rounds).
    fn absorb_and_permute(state: &mut State, block_bytes: &[u8; 136])
        -> (State, State, Vec<RoundTrace>)
    {
        let state_in = *state;
        absorb_block(state, block_bytes);
        let state_after_absorb = *state;
        // Capture the round trace BEFORE mutating state.
        let rounds = keccak_f1600_witness(state_after_absorb);
        keccak_f1600(state);
        let _ = state_in;
        (state_after_absorb, *state, rounds)
    }

    // Absorb full 136-byte blocks.
    let mut offset = 0;
    while input.len() - offset >= RATE_BYTES {
        let mut block = [0u8; RATE_BYTES];
        block.copy_from_slice(&input[offset..offset + RATE_BYTES]);
        let state_in = state;
        let (state_after_absorb, state_out, rounds) = absorb_and_permute(&mut state, &block);
        blocks.push(BlockTrace {
            block,
            state_in,
            state_after_absorb,
            state_out,
            rounds,
        });
        offset += RATE_BYTES;
    }

    // Final block with pad10*1 + 0x01 Ethereum keccak256 domain byte.
    let remaining = &input[offset..];
    let mut block = [0u8; RATE_BYTES];
    block[..remaining.len()].copy_from_slice(remaining);
    block[remaining.len()] ^= 0x01;
    block[RATE_BYTES - 1] ^= 0x80;
    let state_in = state;
    let (state_after_absorb, state_out, rounds) = absorb_and_permute(&mut state, &block);
    blocks.push(BlockTrace {
        block,
        state_in,
        state_after_absorb,
        state_out,
        rounds,
    });

    // Squeeze 32 bytes.
    let mut digest = [0u8; 32];
    for i in 0..4 {
        let lane = state[i][0];
        digest[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
    }

    HashTrace {
        input_len: input.len(),
        blocks,
        digest,
    }
}

fn absorb_block(state: &mut State, block: &[u8]) {
    debug_assert_eq!(block.len(), 136);
    for i in 0..17 {
        let lane = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
        let x = i % 5;
        let y = i / 5;
        state[x][y] ^= lane;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 202 Appendix A: keccak-f[1600] applied to the zero state.
    /// First 8 bytes (little-endian of lane (0,0)) after one permutation.
    #[test]
    fn test_keccak_f1600_zero_state_lane_0_0() {
        let mut s: State = [[0u64; 5]; 5];
        keccak_f1600(&mut s);
        // Expected lane (0,0) after keccak-f[1600] on zero state.
        // Derived from the NIST FIPS 202 reference test vectors.
        assert_eq!(s[0][0], 0xF1258F7940E1DDE7);
        assert_eq!(s[1][0], 0x84D5CCF933C0478A);
        assert_eq!(s[2][0], 0xD598261EA65AA9EE);
        assert_eq!(s[3][0], 0xBD1547306F80494D);
        assert_eq!(s[4][0], 0x8B284E056253D057);
    }

    /// The full-hash witness must reproduce `keccak256(input)` exactly and
    /// satisfy its chain/absorb/permutation invariants.
    #[test]
    fn test_hash_witness_matches_keccak256() {
        // Cover rate-boundary cases: empty (1 block), short (1), exactly 135
        // (1 block with 0x01^0x80 XOR into same byte — trickier padding),
        // exactly 136 (2 blocks: one full + one all-padding),
        // and 272 (3 blocks).
        for input_len in [0usize, 3, 32, 135, 136, 137, 272] {
            let input: Vec<u8> = (0..input_len).map(|i| (i * 13 + 7) as u8).collect();
            let trace = keccak_witness(&input);
            assert_eq!(trace.input_len, input.len(), "input_len mismatch");
            assert_eq!(trace.digest, keccak256(&input), "digest mismatch, len={}", input_len);

            // Expected block count: ceil((len + 1) / 136), but at least 1.
            let expected_blocks = (input_len / 136) + 1;
            assert_eq!(trace.blocks.len(), expected_blocks, "block count wrong for len={}", input_len);

            // Chain invariant: each block's state_in equals the previous
            // block's state_out (or all-zero for the first).
            let mut expected_state_in: State = [[0u64; 5]; 5];
            for b in &trace.blocks {
                assert_eq!(b.state_in, expected_state_in, "chain broken at len={}", input_len);
                // Absorb invariant: state_after_absorb lanes 0..17 equal
                // state_in XOR'd with the block's lanes; lanes 17..25
                // (capacity) are unchanged.
                for i in 0..17 {
                    let x = i % 5;
                    let y = i / 5;
                    let block_lane = u64::from_le_bytes(
                        b.block[i * 8..i * 8 + 8].try_into().unwrap(),
                    );
                    assert_eq!(
                        b.state_after_absorb[x][y],
                        b.state_in[x][y] ^ block_lane,
                        "absorb broken at lane {}",
                        i
                    );
                }
                for i in 17..25 {
                    let x = i % 5;
                    let y = i / 5;
                    assert_eq!(
                        b.state_after_absorb[x][y], b.state_in[x][y],
                        "capacity lane {} touched by absorb",
                        i
                    );
                }
                // Round count
                assert_eq!(b.rounds.len(), NUM_ROUNDS, "wrong round count");
                expected_state_in = b.state_out;
            }

            // Final digest matches first 4 rate lanes of last block's state_out.
            let last = trace.blocks.last().unwrap();
            for i in 0..4 {
                let lane_bytes = last.state_out[i][0].to_le_bytes();
                assert_eq!(&trace.digest[i * 8..i * 8 + 8], &lane_bytes);
            }
        }
    }

    /// keccak256("") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
    #[test]
    fn test_keccak256_empty() {
        let h = keccak256(b"");
        let expected = [
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
            0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
            0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
            0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
        ];
        assert_eq!(h, expected);
    }

    /// keccak256("abc") = 4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
    #[test]
    fn test_keccak256_abc() {
        let h = keccak256(b"abc");
        let expected = [
            0x4e, 0x03, 0x65, 0x7a, 0xea, 0x45, 0xa9, 0x4f,
            0xc7, 0xd4, 0x7b, 0xa8, 0x26, 0xc8, 0xd6, 0x67,
            0xc0, 0xd1, 0xe6, 0xe3, 0x3a, 0x64, 0xa0, 0x36,
            0xec, 0x44, 0xf5, 0x8f, 0xa1, 0x2d, 0x6c, 0x45,
        ];
        assert_eq!(h, expected);
    }

    /// A rate-boundary case: input exactly 136 bytes forces two absorb blocks.
    #[test]
    fn test_keccak256_136_byte_input() {
        let input = vec![0x42u8; 136];
        let h = keccak256(&input);
        // Sanity: different from the 135-byte and 137-byte cases.
        let h135 = keccak256(&vec![0x42u8; 135]);
        let h137 = keccak256(&vec![0x42u8; 137]);
        assert_ne!(h, h135);
        assert_ne!(h, h137);
        assert_ne!(h135, h137);
    }

    #[test]
    fn test_round_constants_count() {
        assert_eq!(ROUND_CONSTANTS.len(), NUM_ROUNDS);
    }

    #[test]
    fn test_rho_offsets_symmetric() {
        // (0,0) offset is always 0 by definition.
        assert_eq!(RHO_OFFSETS[0][0], 0);
        // All offsets are < 64.
        for row in &RHO_OFFSETS {
            for &off in row {
                assert!(off < 64);
            }
        }
    }

    #[test]
    fn test_rotl_edges() {
        assert_eq!(rotl(1, 0), 1);
        assert_eq!(rotl(1, 1), 2);
        assert_eq!(rotl(0x8000_0000_0000_0000, 1), 1);
        assert_eq!(rotl(0xFFFF_FFFF_FFFF_FFFF, 37), 0xFFFF_FFFF_FFFF_FFFF);
    }

    #[test]
    fn test_pi_permutation_bijective() {
        // Start with distinct values and confirm π is a bijection.
        let mut a: State = [[0u64; 5]; 5];
        for x in 0..5 {
            for y in 0..5 {
                a[x][y] = (x * 5 + y + 1) as u64;
            }
        }
        let mut b = a;
        pi(&mut b);
        let mut values: Vec<u64> = b.iter().flat_map(|r| r.iter().copied()).collect();
        values.sort_unstable();
        let expected: Vec<u64> = (1..=25u64).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn test_witness_matches_direct_permutation() {
        // Cross-check: running keccak_f1600 and taking the final round's
        // `after_iota` from the witness must agree.
        let mut state: State = [[0u64; 5]; 5];
        state[0][0] = 0x0123_4567_89ab_cdef;
        state[2][3] = 0xdead_beef_cafe_babe;

        let mut direct = state;
        keccak_f1600(&mut direct);

        let trace = keccak_f1600_witness(state);
        assert_eq!(trace.len(), NUM_ROUNDS);
        assert_eq!(trace[0].before, state);
        assert_eq!(trace[NUM_ROUNDS - 1].after_iota, direct);

        // Each round's `before` must equal the previous round's `after_iota`.
        for i in 1..NUM_ROUNDS {
            assert_eq!(trace[i].before, trace[i - 1].after_iota);
        }
    }

    #[test]
    fn test_witness_c_d_internal_consistency() {
        // For any state, `D[x] = C[x-1] XOR rotl(C[x+1], 1)` must hold,
        // and applying D to `before` must produce `after_theta`.
        let state: State = [[0u64; 5]; 5];
        let trace = keccak_f1600_witness(state);
        for round in &trace {
            for x in 0..5 {
                assert_eq!(
                    round.d[x],
                    round.c[(x + 4) % 5] ^ round.c[(x + 1) % 5].rotate_left(1),
                    "round {}: D[{}] inconsistent", round.round, x
                );
            }
            for x in 0..5 {
                for y in 0..5 {
                    assert_eq!(
                        round.after_theta[x][y],
                        round.before[x][y] ^ round.d[x],
                        "round {}: after_theta[{}][{}] inconsistent",
                        round.round, x, y
                    );
                }
            }
        }
    }

    #[test]
    fn test_chi_zero_on_zero() {
        let mut a: State = [[0u64; 5]; 5];
        chi(&mut a);
        for row in &a {
            for &lane in row {
                assert_eq!(lane, 0);
            }
        }
    }
}
