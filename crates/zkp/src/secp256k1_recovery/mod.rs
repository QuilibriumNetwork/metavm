//! secp256k1 ECDSA signature recovery — Ethereum sender binding.
//!
//! Today the EVM trace's `tx_origin` columns (cols 264..267 in
//! `metavm-evm`) take the sender address on host-side trust: the
//! prover writes whatever `tx.from` the inspector saw, with no
//! algebraic check that the address actually corresponds to the
//! signer of the signed transaction payload.
//!
//! This module is the first step toward closing that gap:
//!   1. A **host-side oracle** ([`recover_sender`] /
//!      [`verify_tx_sender`]) that uses the `k256` crate to recover
//!      the signer address from `(v, r, s, msg_hash)` and check it
//!      against a claimed sender. This already lets composition
//!      pipelines that trust the host run a sound *external* check
//!      before the proof is generated, and is the reference
//!      implementation used to test the algebraic side.
//!   2. A **skeleton AIR** ([`recovery_air`]) that exposes the same
//!      witness shape (`msg_hash`, `r`, `s`, `v`, recovered point,
//!      recovered address) with one demonstration of nonnative-field
//!      style binding (the recovered address is bound to the keccak
//!      of the recovered uncompressed pubkey via a fresh
//!      Schwartz-Zippel check on a single representative row). The
//!      full secp256k1 group law would take weeks of constraints —
//!      this scaffold establishes the column layout and the
//!      cross-AIR linkage to the EVM main trace so the in-circuit
//!      ECDSA gadget can be slotted in incrementally.
//!   3. A [`make_tx_sender_recovery_descriptor`] cross-AIR LogUp
//!      descriptor that pins the EVM's `tx_origin` limbs to this
//!      gadget's recovered address (gated by a synthetic
//!      `is_first_row` selector — the sender is constant for the
//!      whole transaction, so binding once per tx is sufficient).
//!
//! ## Soundness scope (current)
//!
//! - The host-side helpers are cryptographically sound: `k256`
//!   performs the full ECDSA recovery and we compare 20-byte
//!   addresses byte-for-byte.
//! - The algebraic AIR currently proves *only* the
//!   `recovered_addr == keccak256(recovered_pk_uncompressed[1..])[12..32]`
//!   relationship between its own columns (plus byte range checks
//!   and binary `is_real`). It does **not** prove that
//!   `recovered_pk` is the ECDSA recovery of `(v, r, s, msg_hash)` —
//!   that requires a full secp256k1 scalar-mul + point-add gadget
//!   over a nonnative 256-bit prime, which is deferred.
//! - The descriptor in this module therefore turns the EVM's
//!   trusted-input on `tx_origin` into a *trusted-input on the
//!   `recovered_pk` columns of this AIR*. The remaining gap is
//!   closed by future ECDSA-in-circuit work; the column layout and
//!   linkages here are designed so that landing the ECDSA gadget is
//!   a drop-in upgrade rather than a re-plumb.

use crate::keccak::keccak256;
use crate::transaction::{Eip1559Tx, LegacyTx, Transaction};
use crate::tx_sig_hash::signing_hash;

use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};

pub mod recovery_air;

pub use recovery_air::{
    make_tx_sender_recovery_descriptor, RecoveryAirConstraintSystem, RecoveryAirWitness,
    RecoveryRow, COL_IS_FIRST_ROW, COL_IS_REAL, COL_MSG_HASH_OFFSET, COL_R_OFFSET,
    COL_RECOVERED_ADDR_OFFSET, COL_RECOVERED_X_OFFSET, COL_RECOVERED_Y_OFFSET, COL_S_OFFSET,
    COL_V, NUM_COLUMNS,
};

/// Decode an Ethereum-style `v` into the canonical `y_parity` (0 or 1).
///
/// - Legacy (pre-EIP-155): `v = 27` or `28` → parity = `v - 27`.
/// - EIP-155: `v = 2*chain_id + 35 + parity` → parity = `(v - 35) % 2`
///   and the encoded chain id must match `chain_id` if it is supplied.
/// - EIP-1559 / typed: `v` is already `y_parity` (0 or 1).
pub fn decode_v(v: u64, chain_id: Option<u64>) -> Result<u8, String> {
    if v == 0 || v == 1 {
        return Ok(v as u8);
    }
    if v == 27 || v == 28 {
        return Ok((v - 27) as u8);
    }
    if v >= 35 {
        let encoded_chain = (v - 35) / 2;
        if let Some(cid) = chain_id {
            if encoded_chain != cid {
                return Err(format!(
                    "v={} encodes chain_id={} but expected {}",
                    v, encoded_chain, cid
                ));
            }
        }
        return Ok(((v - 35) % 2) as u8);
    }
    Err(format!("unrecognized v value: {}", v))
}

/// Host-side ECDSA recovery: given `(v, r, s)` over `msg_hash`,
/// return the 20-byte Ethereum address of the signer.
pub fn recover_sender(
    v: u64,
    r: [u8; 32],
    s: [u8; 32],
    msg_hash: [u8; 32],
    chain_id: Option<u64>,
) -> Result<[u8; 20], String> {
    let parity = decode_v(v, chain_id)?;
    let recovery_id =
        RecoveryId::try_from(parity).map_err(|e| format!("bad recovery id: {:?}", e))?;

    let mut sig_bytes = [0u8; 64];
    sig_bytes[0..32].copy_from_slice(&r);
    sig_bytes[32..64].copy_from_slice(&s);
    let signature = Signature::from_slice(&sig_bytes)
        .map_err(|e| format!("invalid (r,s) signature bytes: {:?}", e))?;

    let vk = VerifyingKey::recover_from_prehash(&msg_hash, &signature, recovery_id)
        .map_err(|e| format!("ECDSA recovery failed: {:?}", e))?;

    let encoded = vk.to_encoded_point(false); // 65 bytes: 0x04 || X || Y
    let bytes = encoded.as_bytes();
    if bytes.len() != 65 || bytes[0] != 0x04 {
        return Err(format!(
            "unexpected pubkey encoding: len={}, first={:02x}",
            bytes.len(),
            bytes.first().copied().unwrap_or(0)
        ));
    }
    let hash = keccak256(&bytes[1..]); // hash of X||Y (64 bytes)
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..32]);
    Ok(addr)
}

/// Same as [`recover_sender`] but also returns the recovered
/// uncompressed `(X, Y)` coordinates — exposed for AIR witness
/// construction so the algebraic side can commit to them.
pub fn recover_sender_full(
    v: u64,
    r: [u8; 32],
    s: [u8; 32],
    msg_hash: [u8; 32],
    chain_id: Option<u64>,
) -> Result<([u8; 20], [u8; 32], [u8; 32]), String> {
    let parity = decode_v(v, chain_id)?;
    let recovery_id =
        RecoveryId::try_from(parity).map_err(|e| format!("bad recovery id: {:?}", e))?;

    let mut sig_bytes = [0u8; 64];
    sig_bytes[0..32].copy_from_slice(&r);
    sig_bytes[32..64].copy_from_slice(&s);
    let signature = Signature::from_slice(&sig_bytes)
        .map_err(|e| format!("invalid (r,s) signature bytes: {:?}", e))?;

    let vk = VerifyingKey::recover_from_prehash(&msg_hash, &signature, recovery_id)
        .map_err(|e| format!("ECDSA recovery failed: {:?}", e))?;

    let encoded = vk.to_encoded_point(false);
    let bytes = encoded.as_bytes();
    if bytes.len() != 65 || bytes[0] != 0x04 {
        return Err("unexpected pubkey encoding".to_string());
    }
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&bytes[1..33]);
    y.copy_from_slice(&bytes[33..65]);
    let hash = keccak256(&bytes[1..]);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..32]);
    Ok((addr, x, y))
}

/// Extract `(v, r, s)` from a [`Transaction`], normalizing across the
/// legacy and EIP-1559 variants.
pub fn extract_vrs(tx: &Transaction) -> (u64, [u8; 32], [u8; 32]) {
    match tx {
        Transaction::Legacy(LegacyTx { v, r, s, .. }) => (*v, *r, *s),
        Transaction::Eip1559(Eip1559Tx { y_parity, r, s, .. }) => (*y_parity, *r, *s),
    }
}

/// Host-side oracle: compute the signing hash for a transaction,
/// ECDSA-recover the sender, and check it matches `claimed_sender`.
///
/// - For Legacy + EIP-155, pass `chain_id = Some(cid)`.
/// - For Legacy pre-EIP-155, pass `chain_id = None` (the signing
///   hash will omit the chain-id triple).
/// - For EIP-1559, `chain_id` is taken from the tx itself for the
///   signing hash, but the `v` (= `y_parity`) does not encode it,
///   so passing `None` for `chain_id` to this helper is fine.
pub fn verify_tx_sender(
    tx: &Transaction,
    claimed_sender: [u8; 20],
    chain_id: Option<u64>,
) -> Result<(), String> {
    let msg_hash = signing_hash(tx, chain_id);
    let (v, r, s) = extract_vrs(tx);
    // For EIP-1559 v is already 0/1; chain_id decoding is N/A.
    let v_chain = match tx {
        Transaction::Legacy(_) => chain_id,
        Transaction::Eip1559(_) => None,
    };
    let recovered = recover_sender(v, r, s, msg_hash, v_chain)?;
    if recovered != claimed_sender {
        return Err(format!(
            "sender mismatch: recovered=0x{} claimed=0x{}",
            hex_lower(&recovered),
            hex_lower(&claimed_sender),
        ));
    }
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::{Eip1559Tx, LegacyTx};
    use k256::ecdsa::{signature::hazmat::PrehashSigner, SigningKey};

    /// Deterministic test signing key (NOT for production).
    fn test_signing_key() -> SigningKey {
        // 32 nonzero bytes; well below curve order.
        let mut sk_bytes = [0u8; 32];
        for i in 0..32 {
            sk_bytes[i] = (i as u8) + 1;
        }
        SigningKey::from_bytes((&sk_bytes).into()).expect("valid signing key")
    }

    fn test_address(sk: &SigningKey) -> [u8; 20] {
        let vk = sk.verifying_key();
        let encoded = vk.to_encoded_point(false);
        let hash = keccak256(&encoded.as_bytes()[1..]);
        let mut a = [0u8; 20];
        a.copy_from_slice(&hash[12..32]);
        a
    }

    /// Sign `msg_hash` with `sk` and return `(v_parity_only, r, s)`.
    /// `v` here is 0 or 1 (y_parity); legacy v = v_parity + 27,
    /// EIP-155 v = chain_id*2 + 35 + v_parity.
    fn raw_sign(sk: &SigningKey, msg_hash: &[u8; 32]) -> (u8, [u8; 32], [u8; 32]) {
        let (sig, rid): (Signature, RecoveryId) =
            sk.sign_prehash(msg_hash).expect("sign_prehash");
        let bytes = sig.to_bytes();
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        r.copy_from_slice(&bytes[0..32]);
        s.copy_from_slice(&bytes[32..64]);
        (rid.to_byte(), r, s)
    }

    #[test]
    fn known_good_signature_recovers_correct_address() {
        let sk = test_signing_key();
        let expected_addr = test_address(&sk);
        let msg_hash = [0x42u8; 32];
        let (parity, r, s) = raw_sign(&sk, &msg_hash);

        // Pre-EIP-155 legacy: v = parity + 27.
        let v = parity as u64 + 27;
        let recovered = recover_sender(v, r, s, msg_hash, None).expect("recover");
        assert_eq!(recovered, expected_addr);
    }

    #[test]
    fn mismatched_address_fails_verify() {
        let sk = test_signing_key();
        let msg_hash = [0x12u8; 32];
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        let v = parity as u64 + 27;
        let wrong = [0xffu8; 20];
        // recover_sender succeeds (it returns the *real* signer),
        // but a comparison against a wrong claimed address must fail.
        let recovered = recover_sender(v, r, s, msg_hash, None).expect("recover");
        assert_ne!(recovered, wrong);
    }

    #[test]
    fn recovery_is_deterministic() {
        let sk = test_signing_key();
        let msg_hash = [0x77u8; 32];
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        let v = parity as u64 + 27;
        let a = recover_sender(v, r, s, msg_hash, None).expect("a");
        let b = recover_sender(v, r, s, msg_hash, None).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn verify_legacy_tx_sender() {
        let sk = test_signing_key();
        let addr = test_address(&sk);

        // Build a legacy tx (without v/r/s yet), compute its signing
        // hash (pre-EIP-155), sign, and stuff (v, r, s) back in.
        let mut tx = LegacyTx {
            nonce: 9,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 0,
            r: [0u8; 32],
            s: [0u8; 32],
        };
        let msg_hash = signing_hash(&Transaction::Legacy(tx.clone()), None);
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        tx.v = parity as u64 + 27;
        tx.r = r;
        tx.s = s;
        verify_tx_sender(&Transaction::Legacy(tx), addr, None).expect("verify");
    }

    #[test]
    fn verify_eip155_legacy_tx_sender() {
        let sk = test_signing_key();
        let addr = test_address(&sk);
        let chain_id = 1u64;

        let mut tx = LegacyTx {
            nonce: 1,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x99u8; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 0,
            r: [0u8; 32],
            s: [0u8; 32],
        };
        let msg_hash = signing_hash(&Transaction::Legacy(tx.clone()), Some(chain_id));
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        tx.v = 35 + 2 * chain_id + parity as u64;
        tx.r = r;
        tx.s = s;
        verify_tx_sender(&Transaction::Legacy(tx), addr, Some(chain_id)).expect("verify");
    }

    #[test]
    fn verify_eip1559_tx_sender() {
        let sk = test_signing_key();
        let addr = test_address(&sk);

        let mut tx = Eip1559Tx {
            chain_id: 1,
            nonce: 0,
            max_priority_fee_per_gas: [0u8; 32],
            max_fee_per_gas: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: vec![],
            access_list_rlp: vec![0xc0],
            y_parity: 0,
            r: [0u8; 32],
            s: [0u8; 32],
        };
        let msg_hash = signing_hash(&Transaction::Eip1559(tx.clone()), None);
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        tx.y_parity = parity as u64;
        tx.r = r;
        tx.s = s;
        verify_tx_sender(&Transaction::Eip1559(tx), addr, None).expect("verify");
    }

    #[test]
    fn verify_tx_sender_rejects_wrong_claim() {
        let sk = test_signing_key();
        let mut tx = LegacyTx {
            nonce: 0,
            gas_price: [0u8; 32],
            gas_limit: 21_000,
            to: Some([0x42u8; 20]),
            value: [0u8; 32],
            data: vec![],
            v: 0,
            r: [0u8; 32],
            s: [0u8; 32],
        };
        let msg_hash = signing_hash(&Transaction::Legacy(tx.clone()), None);
        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        tx.v = parity as u64 + 27;
        tx.r = r;
        tx.s = s;
        let wrong = [0xaau8; 20];
        let err = verify_tx_sender(&Transaction::Legacy(tx), wrong, None).unwrap_err();
        assert!(err.contains("sender mismatch"), "got: {}", err);
    }

    #[test]
    fn decode_v_handles_all_encodings() {
        assert_eq!(decode_v(0, None).unwrap(), 0);
        assert_eq!(decode_v(1, None).unwrap(), 1);
        assert_eq!(decode_v(27, None).unwrap(), 0);
        assert_eq!(decode_v(28, None).unwrap(), 1);
        // EIP-155 chain_id = 1: 37 or 38.
        assert_eq!(decode_v(37, Some(1)).unwrap(), 0);
        assert_eq!(decode_v(38, Some(1)).unwrap(), 1);
        // Mismatched chain_id rejected.
        assert!(decode_v(37, Some(5)).is_err());
        // Bogus v rejected.
        assert!(decode_v(2, None).is_err());
    }
}
