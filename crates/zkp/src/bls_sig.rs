//! BLS12-381 signature / aggregation reference module.
//!
//! Host-side reference implementation used as a building block for
//! attestation aggregation. Follows the IETF `bls-signatures` draft as
//! specialised by the Ethereum beacon chain:
//!
//!   ciphersuite: BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_
//!   - public keys in G1 (48 bytes compressed)
//!   - signatures   in G2 (96 bytes compressed)
//!   - hash-to-curve in G2 via SSWU + XMD:SHA-256
//!
//! This is *not* an in-circuit pairing; it's the plain-curve reference
//! that the eventual circuit will be constrained to match.
//!
//! No new crate dependencies are introduced: only the `blst` crate that is
//! already used by `scheme::bls12381_scheme`. All calls go through the raw
//! FFI bindings from `blst::*`.

use blst::*;

/// BLS12-381 subgroup order `r` (big-endian). Used by the boundary-case
/// test that checks `from_bytes(r)` is rejected. We rely on `blst`'s
/// `blst_scalar_fr_check` for the runtime check itself; this constant
/// exists only to make the test's intent explicit.
///
/// r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
#[cfg(test)]
const CURVE_ORDER_BE: [u8; 32] = [
    0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48,
    0x33, 0x39, 0xd8, 0x08, 0x09, 0xa1, 0xd8, 0x05,
    0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe, 0x5b, 0xfe,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
];

/// Errors produced by this module. Deliberately coarse — this is a
/// reference module, not a user-facing API.
#[derive(Debug, PartialEq, Eq)]
pub enum BlsError {
    /// Secret-key scalar is not in the range `[1, r)`, or is all-zero.
    InvalidSecretKey,
    /// G1 (public key) decompression failed or point is not in the
    /// prime-order subgroup.
    InvalidPublicKey,
    /// G2 (signature) decompression failed or point is not in the
    /// prime-order subgroup.
    InvalidSignature,
    /// Attempt to aggregate an empty slice. The IETF draft leaves this
    /// case implementation-defined; we surface it as an explicit error
    /// so the caller can decide.
    EmptyAggregation,
    /// `aggregate_verify` received mismatched pks / messages lengths.
    LengthMismatch,
}

/// A 32-byte BLS12-381 scalar, known to be in `[1, r)`.
#[derive(Clone)]
pub struct SecretKey(pub blst_scalar);

impl core::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Do not leak key material; `blst_scalar` has no public Debug impl
        // anyway so we can't defer to one.
        f.write_str("SecretKey(..redacted..)")
    }
}

/// G1 public key, compressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicKey(pub [u8; 48]);

/// G2 signature, compressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Signature(pub [u8; 96]);

// -------------------------------------------------------------------------
// Low-level helpers
// -------------------------------------------------------------------------

/// Decompress a 48-byte G1 public key to an affine point and perform a
/// prime-order subgroup check.
pub(crate) fn pk_to_affine(pk: &PublicKey) -> Result<blst_p1_affine, BlsError> {
    let mut aff = blst_p1_affine::default();
    let err = unsafe { blst_p1_uncompress(&mut aff, pk.0.as_ptr()) };
    if err != BLST_ERROR::BLST_SUCCESS {
        return Err(BlsError::InvalidPublicKey);
    }
    // Subgroup check: BLS signatures require pk ∈ G1 (prime-order subgroup),
    // not just on-curve. blst exposes this via `blst_p1_affine_in_g1`.
    if !unsafe { blst_p1_affine_in_g1(&aff) } {
        return Err(BlsError::InvalidPublicKey);
    }
    Ok(aff)
}

/// Decompress a 96-byte G2 signature to an affine point and perform a
/// prime-order subgroup check.
pub(crate) fn sig_to_affine(sig: &Signature) -> Result<blst_p2_affine, BlsError> {
    let mut aff = blst_p2_affine::default();
    let err = unsafe { blst_p2_uncompress(&mut aff, sig.0.as_ptr()) };
    if err != BLST_ERROR::BLST_SUCCESS {
        return Err(BlsError::InvalidSignature);
    }
    if !unsafe { blst_p2_affine_in_g2(&aff) } {
        return Err(BlsError::InvalidSignature);
    }
    Ok(aff)
}

/// Hash a message to an affine G2 point per
/// `hash_to_curve(BLS12381G2_XMD:SHA-256_SSWU_RO_)`.
pub(crate) fn hash_to_g2_affine(msg: &[u8], dst: &[u8]) -> blst_p2_affine {
    let mut p = blst_p2::default();
    unsafe {
        blst_hash_to_g2(
            &mut p,
            msg.as_ptr(),
            msg.len(),
            dst.as_ptr(),
            dst.len(),
            core::ptr::null(),
            0,
        );
    }
    let mut aff = blst_p2_affine::default();
    unsafe { blst_p2_to_affine(&mut aff, &p); }
    aff
}

/// The BLS12-381 G1 generator, as a projective point.
fn g1_generator() -> blst_p1 {
    let mut g = blst_p1::default();
    // blst exposes `BLS12_381_G1` as a static `blst_p1_affine`.
    unsafe { blst_p1_from_affine(&mut g, &BLS12_381_G1); }
    g
}

/// Compress a projective G1 point to 48 bytes.
fn compress_g1(p: &blst_p1) -> [u8; 48] {
    let mut out = [0u8; 48];
    unsafe { blst_p1_compress(out.as_mut_ptr(), p); }
    out
}

/// Compress a projective G2 point to 96 bytes.
fn compress_g2(p: &blst_p2) -> [u8; 96] {
    let mut out = [0u8; 96];
    unsafe { blst_p2_compress(out.as_mut_ptr(), p); }
    out
}

// -------------------------------------------------------------------------
// SecretKey
// -------------------------------------------------------------------------

impl SecretKey {
    /// Build a deterministic test secret key from a small seed byte.
    /// `seed` becomes the last byte of the scalar (so `from_u8_seed(1)` is
    /// `0x0000...0001`). Intended for unit tests and fixtures — DO NOT use
    /// for any real key material.
    pub fn from_u8_seed(seed: u8) -> Self {
        let mut bytes = [0u8; 32];
        bytes[31] = seed;
        // `seed = 0` would fail `from_bytes`; bump it to 1.
        if bytes.iter().all(|&b| b == 0) {
            bytes[31] = 1;
        }
        Self::from_bytes(&bytes).expect("seed produces a valid non-zero scalar")
    }

    /// Parse a big-endian 32-byte scalar. Rejects `0` and scalars `>= r`.
    pub fn from_bytes(bytes: &[u8; 32]) -> Result<Self, BlsError> {
        let mut sk = blst_scalar::default();
        unsafe { blst_scalar_from_bendian(&mut sk, bytes.as_ptr()); }

        // Must be strictly less than r.
        if !unsafe { blst_scalar_fr_check(&sk) } {
            return Err(BlsError::InvalidSecretKey);
        }
        // Must be non-zero (all-zero scalars degenerate pk to identity).
        if sk.b.iter().all(|&byte| byte == 0) {
            return Err(BlsError::InvalidSecretKey);
        }
        Ok(SecretKey(sk))
    }

    /// Derive the corresponding public key as `sk · G1`.
    pub fn public_key(&self) -> PublicKey {
        let g = g1_generator();
        let mut pk = blst_p1::default();
        // 255 bits is the full scalar size for BLS12-381 `r`.
        unsafe { blst_p1_mult(&mut pk, &g, self.0.b.as_ptr(), 255); }
        PublicKey(compress_g1(&pk))
    }

    /// Sign a message: `sig = sk · H(msg)`.
    ///
    /// `domain_separation_tag` is the DST string for the hash-to-curve
    /// ciphersuite; for the beacon-chain POP scheme this is
    /// `b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_"`.
    pub fn sign(&self, msg: &[u8], domain_separation_tag: &[u8]) -> Signature {
        let h_aff = hash_to_g2_affine(msg, domain_separation_tag);
        let mut h = blst_p2::default();
        unsafe { blst_p2_from_affine(&mut h, &h_aff); }
        let mut sig = blst_p2::default();
        unsafe { blst_p2_mult(&mut sig, &h, self.0.b.as_ptr(), 255); }
        Signature(compress_g2(&sig))
    }
}

// -------------------------------------------------------------------------
// Single-signature verify
// -------------------------------------------------------------------------

/// Verify a single BLS signature: checks `e(G1, sig) == e(pk, H(msg))`.
///
/// Returns `false` on any parse / subgroup / pairing failure.
pub fn verify(pk: &PublicKey, msg: &[u8], sig: &Signature, dst: &[u8]) -> bool {
    let pk_aff = match pk_to_affine(pk) { Ok(a) => a, Err(_) => return false };
    let sig_aff = match sig_to_affine(sig) { Ok(a) => a, Err(_) => return false };
    let h_aff = hash_to_g2_affine(msg, dst);

    // Pairing equation (rearranged to a single final-exponentiation):
    //   e(-G1, sig) · e(pk, H(msg)) == 1
    // We negate G1 because blst miller_loop takes (G2, G1) as (q, p).
    let mut neg_g1 = g1_generator();
    unsafe { blst_p1_cneg(&mut neg_g1, true); }
    let mut neg_g1_aff = blst_p1_affine::default();
    unsafe { blst_p1_to_affine(&mut neg_g1_aff, &neg_g1); }

    let ml1 = blst_fp12::miller_loop(&sig_aff, &neg_g1_aff);
    let ml2 = blst_fp12::miller_loop(&h_aff, &pk_aff);
    let product = ml1 * ml2;
    let mut fe = blst_fp12::default();
    unsafe { blst_final_exp(&mut fe, &product); }
    unsafe { blst_fp12_is_equal(&fe, blst_fp12_one()) }
}

// -------------------------------------------------------------------------
// Aggregation
// -------------------------------------------------------------------------

/// Aggregate a non-empty slice of signatures: `agg = Σ sig_i`.
///
/// Returns `BlsError::EmptyAggregation` on an empty slice (caller chooses
/// whether to substitute a zero-sig or bubble up). Returns
/// `BlsError::InvalidSignature` if any sig fails to decompress.
pub fn aggregate_sigs(sigs: &[Signature]) -> Result<Signature, BlsError> {
    if sigs.is_empty() {
        return Err(BlsError::EmptyAggregation);
    }
    let mut acc = blst_p2::default(); // identity
    for sig in sigs {
        let aff = sig_to_affine(sig)?;
        let mut p = blst_p2::default();
        unsafe { blst_p2_from_affine(&mut p, &aff); }
        unsafe { blst_p2_add_or_double(&mut acc, &acc, &p); }
    }
    Ok(Signature(compress_g2(&acc)))
}

/// Aggregate a non-empty slice of public keys: `agg = Σ pk_i`.
pub fn aggregate_pubkeys(pks: &[PublicKey]) -> Result<PublicKey, BlsError> {
    if pks.is_empty() {
        return Err(BlsError::EmptyAggregation);
    }
    let mut acc = blst_p1::default();
    for pk in pks {
        let aff = pk_to_affine(pk)?;
        let mut p = blst_p1::default();
        unsafe { blst_p1_from_affine(&mut p, &aff); }
        unsafe { blst_p1_add_or_double(&mut acc, &acc, &p); }
    }
    Ok(PublicKey(compress_g1(&acc)))
}

/// Beacon-chain "fast aggregate verify": all signers signed the same
/// message. Aggregate the public keys, then do a single pairing check.
///
/// This is the hot path for attestation / sync-committee aggregation.
pub fn fast_aggregate_verify(
    pks: &[PublicKey],
    msg: &[u8],
    agg_sig: &Signature,
    dst: &[u8],
) -> bool {
    let agg_pk = match aggregate_pubkeys(pks) { Ok(pk) => pk, Err(_) => return false };
    verify(&agg_pk, msg, agg_sig, dst)
}

/// Distinct-message aggregate verify:
///   `e(G1, agg_sig) == Π_i e(pk_i, H(msg_i))`
///
/// Less common on the beacon chain (where FAV dominates), but this is the
/// standard IETF API. Runs one Miller loop per (pk, msg) pair plus one for
/// the aggregated signature, then a single final exponentiation.
pub fn aggregate_verify(
    pks_and_msgs: &[(PublicKey, &[u8])],
    agg_sig: &Signature,
    dst: &[u8],
) -> bool {
    if pks_and_msgs.is_empty() {
        return false;
    }

    let sig_aff = match sig_to_affine(agg_sig) { Ok(a) => a, Err(_) => return false };

    // Accumulate Π e(pk_i, H(msg_i)) · e(-G1, agg_sig) and check == 1.
    let mut neg_g1 = g1_generator();
    unsafe { blst_p1_cneg(&mut neg_g1, true); }
    let mut neg_g1_aff = blst_p1_affine::default();
    unsafe { blst_p1_to_affine(&mut neg_g1_aff, &neg_g1); }

    let mut accum = blst_fp12::miller_loop(&sig_aff, &neg_g1_aff);
    for (pk, msg) in pks_and_msgs {
        let pk_aff = match pk_to_affine(pk) { Ok(a) => a, Err(_) => return false };
        let h_aff = hash_to_g2_affine(msg, dst);
        let ml = blst_fp12::miller_loop(&h_aff, &pk_aff);
        accum = accum * ml;
    }

    let mut fe = blst_fp12::default();
    unsafe { blst_final_exp(&mut fe, &accum); }
    unsafe { blst_fp12_is_equal(&fe, blst_fp12_one()) }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Beacon-chain ciphersuite DST (POP variant).
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    fn sk_from_byte(b: u8) -> SecretKey {
        let mut seed = [0u8; 32];
        seed[31] = b;
        SecretKey::from_bytes(&seed).unwrap()
    }

    // ---------------------------------------------------------------------
    // Test 1: known-good signature (self-generated; round-trip + stability)
    // ---------------------------------------------------------------------
    //
    // We cross-check against the externally-known pubkey for `sk = 1`:
    //   pk(sk=1) = G1 generator compressed =
    //     0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
    // (this is the canonical BLS12-381 G1 compressed generator used by the
    // Ethereum trusted setup and the consensus-specs `bls/sk_to_pk/sk_1`
    // test vector).
    #[test]
    fn sk_one_yields_g1_generator() {
        let sk = sk_from_byte(1);
        let pk = sk.public_key();
        let expected: [u8; 48] = [
            0x97, 0xf1, 0xd3, 0xa7, 0x31, 0x97, 0xd7, 0x94,
            0x26, 0x95, 0x63, 0x8c, 0x4f, 0xa9, 0xac, 0x0f,
            0xc3, 0x68, 0x8c, 0x4f, 0x97, 0x74, 0xb9, 0x05,
            0xa1, 0x4e, 0x3a, 0x3f, 0x17, 0x1b, 0xac, 0x58,
            0x6c, 0x55, 0xe8, 0x3f, 0xf9, 0x7a, 0x1a, 0xef,
            0xfb, 0x3a, 0xf0, 0x0a, 0xdb, 0x22, 0xc6, 0xbb,
        ];
        assert_eq!(pk.0, expected, "sk=1 must yield the canonical G1 generator");
    }

    #[test]
    fn sign_verify_round_trip() {
        let sk = sk_from_byte(0x2a);
        let pk = sk.public_key();
        let msg = b"hello";
        let sig = sk.sign(msg, POP_DST);
        assert!(verify(&pk, msg, &sig, POP_DST));
    }

    #[test]
    fn sign_is_deterministic() {
        // Signing the same (sk, msg, dst) must be deterministic: no RNG
        // enters the signing path. This catches accidental reintroduction
        // of randomness.
        let sk = sk_from_byte(0x2a);
        let a = sk.sign(b"hello", POP_DST);
        let b = sk.sign(b"hello", POP_DST);
        assert_eq!(a.0, b.0);
    }

    // ---------------------------------------------------------------------
    // Test 2: empty aggregation
    // ---------------------------------------------------------------------
    //
    // Decision documented here: we return `BlsError::EmptyAggregation` for
    // both `aggregate_sigs(&[])` and `aggregate_pubkeys(&[])`. The IETF
    // draft leaves the empty case implementation-defined; returning an
    // explicit error lets callers choose whether to substitute the identity
    // element (which would silently produce an all-zero "infinity"
    // signature).
    #[test]
    fn empty_aggregation_errors() {
        assert_eq!(aggregate_sigs(&[]).unwrap_err(), BlsError::EmptyAggregation);
        assert_eq!(aggregate_pubkeys(&[]).unwrap_err(), BlsError::EmptyAggregation);
    }

    // ---------------------------------------------------------------------
    // Test 3: fast aggregate verify, same message, 3 keys
    // ---------------------------------------------------------------------
    #[test]
    fn fast_aggregate_verify_accepts_valid() {
        let sks: Vec<SecretKey> = (1..=3).map(sk_from_byte).collect();
        let pks: Vec<PublicKey> = sks.iter().map(|s| s.public_key()).collect();
        let msg: &[u8] = b"attestation";
        let sigs: Vec<Signature> = sks.iter().map(|s| s.sign(msg, POP_DST)).collect();
        let agg = aggregate_sigs(&sigs).unwrap();

        assert!(fast_aggregate_verify(&pks, msg, &agg, POP_DST));
    }

    #[test]
    fn fast_aggregate_verify_rejects_tampered_sig() {
        let sks: Vec<SecretKey> = (1..=3).map(sk_from_byte).collect();
        let pks: Vec<PublicKey> = sks.iter().map(|s| s.public_key()).collect();
        let msg: &[u8] = b"attestation";
        let sigs: Vec<Signature> = sks.iter().map(|s| s.sign(msg, POP_DST)).collect();
        let mut agg = aggregate_sigs(&sigs).unwrap();

        // Flip a bit in a "free" byte of the compressed signature (avoid
        // the sign/infinity flag bits in byte 0 so we still get a valid
        // on-curve point, not just a parse failure).
        agg.0[50] ^= 0x01;
        assert!(!fast_aggregate_verify(&pks, msg, &agg, POP_DST));
    }

    #[test]
    fn fast_aggregate_verify_rejects_tampered_msg() {
        let sks: Vec<SecretKey> = (1..=3).map(sk_from_byte).collect();
        let pks: Vec<PublicKey> = sks.iter().map(|s| s.public_key()).collect();
        let msg: &[u8] = b"attestation";
        let sigs: Vec<Signature> = sks.iter().map(|s| s.sign(msg, POP_DST)).collect();
        let agg = aggregate_sigs(&sigs).unwrap();

        assert!(!fast_aggregate_verify(&pks, b"attestatioN", &agg, POP_DST));
    }

    #[test]
    fn fast_aggregate_verify_rejects_tampered_pk() {
        let sks: Vec<SecretKey> = (1..=3).map(sk_from_byte).collect();
        let mut pks: Vec<PublicKey> = sks.iter().map(|s| s.public_key()).collect();
        let msg: &[u8] = b"attestation";
        let sigs: Vec<Signature> = sks.iter().map(|s| s.sign(msg, POP_DST)).collect();
        let agg = aggregate_sigs(&sigs).unwrap();

        // Swap a pk for a different (but valid) one.
        pks[1] = sk_from_byte(99).public_key();
        assert!(!fast_aggregate_verify(&pks, msg, &agg, POP_DST));
    }

    // ---------------------------------------------------------------------
    // Test 4: distinct-message aggregate verify, 3 keys, 3 messages
    // ---------------------------------------------------------------------
    #[test]
    fn aggregate_verify_distinct_messages_accepts_valid() {
        let sks: Vec<SecretKey> = (1..=3).map(sk_from_byte).collect();
        let pks: Vec<PublicKey> = sks.iter().map(|s| s.public_key()).collect();
        let msgs: [&[u8]; 3] = [b"one", b"two", b"three"];
        let sigs: Vec<Signature> = sks.iter().zip(msgs.iter())
            .map(|(sk, m)| sk.sign(m, POP_DST)).collect();
        let agg = aggregate_sigs(&sigs).unwrap();

        let pairs: Vec<(PublicKey, &[u8])> = pks.iter().zip(msgs.iter())
            .map(|(pk, m)| (*pk, *m)).collect();
        assert!(aggregate_verify(&pairs, &agg, POP_DST));
    }

    #[test]
    fn aggregate_verify_distinct_messages_rejects_tampering() {
        let sks: Vec<SecretKey> = (1..=3).map(sk_from_byte).collect();
        let pks: Vec<PublicKey> = sks.iter().map(|s| s.public_key()).collect();
        let msgs: [&[u8]; 3] = [b"one", b"two", b"three"];
        let sigs: Vec<Signature> = sks.iter().zip(msgs.iter())
            .map(|(sk, m)| sk.sign(m, POP_DST)).collect();
        let agg = aggregate_sigs(&sigs).unwrap();

        // Tamper with one of the messages.
        let tampered_msgs: [&[u8]; 3] = [b"one", b"TWO", b"three"];
        let pairs: Vec<(PublicKey, &[u8])> = pks.iter().zip(tampered_msgs.iter())
            .map(|(pk, m)| (*pk, *m)).collect();
        assert!(!aggregate_verify(&pairs, &agg, POP_DST));

        // Tamper with the aggregated signature.
        let mut bad_agg = agg;
        bad_agg.0[80] ^= 0x01;
        let pairs: Vec<(PublicKey, &[u8])> = pks.iter().zip(msgs.iter())
            .map(|(pk, m)| (*pk, *m)).collect();
        assert!(!aggregate_verify(&pairs, &bad_agg, POP_DST));

        // Tamper with one public key.
        let mut bad_pks = pks.clone();
        bad_pks[0] = sk_from_byte(77).public_key();
        let pairs: Vec<(PublicKey, &[u8])> = bad_pks.iter().zip(msgs.iter())
            .map(|(pk, m)| (*pk, *m)).collect();
        assert!(!aggregate_verify(&pairs, &agg, POP_DST));
    }

    // ---------------------------------------------------------------------
    // Test 5: wrong DST
    // ---------------------------------------------------------------------
    #[test]
    fn wrong_dst_rejects() {
        let sk = sk_from_byte(7);
        let pk = sk.public_key();
        let msg: &[u8] = b"hello";
        let sig = sk.sign(msg, POP_DST);
        assert!(verify(&pk, msg, &sig, POP_DST));
        assert!(!verify(&pk, msg, &sig, b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_"));
    }

    // ---------------------------------------------------------------------
    // Secret-key edge cases
    // ---------------------------------------------------------------------
    #[test]
    fn sk_zero_rejected() {
        assert_eq!(
            SecretKey::from_bytes(&[0u8; 32]).unwrap_err(),
            BlsError::InvalidSecretKey
        );
    }

    #[test]
    fn sk_order_or_above_rejected() {
        // Exactly `r` must be rejected (scalars are in `[1, r)`).
        assert_eq!(
            SecretKey::from_bytes(&CURVE_ORDER_BE).unwrap_err(),
            BlsError::InvalidSecretKey
        );
        // All-ones is > r.
        assert_eq!(
            SecretKey::from_bytes(&[0xffu8; 32]).unwrap_err(),
            BlsError::InvalidSecretKey
        );
    }

    // ---------------------------------------------------------------------
    // Invalid inputs
    // ---------------------------------------------------------------------
    #[test]
    fn verify_rejects_garbage_pk() {
        let pk = PublicKey([0xff; 48]); // not a valid compressed G1 point
        let sk = sk_from_byte(3);
        let sig = sk.sign(b"x", POP_DST);
        assert!(!verify(&pk, b"x", &sig, POP_DST));
    }

    #[test]
    fn verify_rejects_garbage_sig() {
        let sk = sk_from_byte(3);
        let pk = sk.public_key();
        let sig = Signature([0xff; 96]);
        assert!(!verify(&pk, b"x", &sig, POP_DST));
    }
}
