//! Sync-committee-shape subset BLS aggregate witness scaffolding.
//!
//! Beacon-chain `sync_aggregate` verification (Altair+) selects a subset
//! of the current sync committee's 512 pubkeys via a 512-bit
//! participation bitmap, aggregates those pubkeys, and verifies the
//! aggregate signature over the slot's signing root. Building the AIR
//! infrastructure for this requires several pieces:
//!
//!   1. Algebraically prove the bitmap-driven subset filtering
//!      (selected = {pk_i : bitmap_i = 1}).
//!   2. Aggregate the filtered subset to `agg_pk = Σ pk_i`.
//!   3. Verify `agg_pk + agg_sig` via pairing (today: host-side oracle
//!      via [`crate::bls_sig::fast_aggregate_verify`]).
//!   4. Bind the full `pubkeys[0..512]` list to the beacon-state
//!      committed sync committee Merkle root (future work, beacon-state
//!      SSZ extraction).
//!
//! This module supplies the **host-side data shape + bitmap-filter
//! preprocessing** for piece (1) and reuses
//! [`crate::bls_sig_constraints::BlsSigWitness`] for piece (2). The
//! algebraic binding of pieces (1) and (3) and (4) are deferred to
//! follow-up phases.
//!
//! # Soundness state
//!
//! With this scaffold alone, an external observer can audit:
//!   - The `full_pubkeys` list the prover claims as the sync committee.
//!   - The `participation_bitmap` they claim selects participants.
//!   - The `agg_sig` and `msg` they claim were verified.
//!
//! The trace builder refuses to construct a witness if
//! `fast_aggregate_verify` fails on the filtered subset (matching
//! existing `BlsSigWitness` behavior). So the host-side oracle
//! soundness gain is: prover commits to a specific (committee, bitmap,
//! signature) tuple that DOES verify under fast_aggregate_verify.
//! Algebraic binding of the filter step is the next phase.

use crate::bls_sig::{fast_aggregate_verify, PublicKey, Signature};
use crate::bls_sig_constraints::BlsSigWitness;

/// Standard Ethereum sync committee size (Altair+).
pub const SYNC_COMMITTEE_SIZE: usize = 512;

/// Errors specific to subset construction (filtering / verification).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubsetBlsError {
    /// `participation_bitmap.len() != full_pubkeys.len()`.
    BitmapLengthMismatch { pubkeys: usize, bitmap: usize },
    /// No bits set in the bitmap → no participating pubkeys.
    NoParticipants,
    /// Host-side `fast_aggregate_verify` rejected the (subset, msg, agg_sig).
    HostAggregateVerifyFailed,
}

/// Sync-committee-shape subset aggregate witness.
///
/// Host-side preprocessing wrapper over [`BlsSigWitness`]: takes the
/// full pubkey list + participation bitmap + aggregate signature, and
/// derives the filtered participating subset on-the-fly via
/// [`Self::to_bls_sig_witness`].
///
/// `full_pubkeys.len()` need not be exactly [`SYNC_COMMITTEE_SIZE`] —
/// any consistent-length pubkey list + bitmap pair works (smaller for
/// tests, 512 for production sync committee). The constant is provided
/// for production use; tests typically use small values.
#[derive(Debug, Clone)]
pub struct SubsetBlsSigWitness {
    /// Full pubkey list (e.g. all 512 sync committee members).
    pub full_pubkeys: Vec<PublicKey>,
    /// Per-pubkey participation flag. Same length as `full_pubkeys`.
    pub participation_bitmap: Vec<bool>,
    /// Aggregate signature claimed for the participating subset.
    pub agg_sig: Signature,
    /// Message that was signed (typically the slot's signing root).
    pub msg: Vec<u8>,
    /// BLS hash-to-curve domain-separation tag.
    pub dst: Vec<u8>,
}

impl SubsetBlsSigWitness {
    /// Number of participating members (set bits in the bitmap).
    pub fn num_participants(&self) -> usize {
        self.participation_bitmap.iter().filter(|b| **b).count()
    }

    /// Filter `full_pubkeys` by `participation_bitmap` (keeping the
    /// original order) and produce a [`BlsSigWitness`] over the
    /// participating subset. Verifies the aggregate signature on the
    /// way; returns `Err` if the host-side oracle rejects.
    pub fn to_bls_sig_witness(&self) -> Result<BlsSigWitness, SubsetBlsError> {
        if self.participation_bitmap.len() != self.full_pubkeys.len() {
            return Err(SubsetBlsError::BitmapLengthMismatch {
                pubkeys: self.full_pubkeys.len(),
                bitmap: self.participation_bitmap.len(),
            });
        }
        let participating: Vec<PublicKey> = self
            .full_pubkeys
            .iter()
            .zip(&self.participation_bitmap)
            .filter_map(|(pk, &b)| if b { Some(pk.clone()) } else { None })
            .collect();
        if participating.is_empty() {
            return Err(SubsetBlsError::NoParticipants);
        }
        if !fast_aggregate_verify(&participating, &self.msg, &self.agg_sig, &self.dst) {
            return Err(SubsetBlsError::HostAggregateVerifyFailed);
        }
        Ok(BlsSigWitness {
            pubkeys: participating,
            msg: self.msg.clone(),
            agg_sig: self.agg_sig.clone(),
            dst: self.dst.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bls_sig::{aggregate_sigs, SecretKey};

    /// Beacon-chain DST (POP variant), matching the existing
    /// `bls_sig_constraints` test convention.
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

    fn build_committee(n: u8) -> (Vec<SecretKey>, Vec<PublicKey>) {
        let sks: Vec<SecretKey> = (1..=n).map(SecretKey::from_u8_seed).collect();
        let pks: Vec<PublicKey> = sks.iter().map(|s| s.public_key()).collect();
        (sks, pks)
    }

    /// Build a SubsetBlsSigWitness over an n-member committee with the
    /// `participating_indices` set as the participating subset.
    fn build_subset_witness(
        committee_size: u8,
        participating_indices: &[usize],
        msg: &[u8],
    ) -> SubsetBlsSigWitness {
        let (sks, pks) = build_committee(committee_size);
        let mut bitmap = vec![false; pks.len()];
        for &i in participating_indices {
            bitmap[i] = true;
        }
        let participating_sigs: Vec<Signature> = participating_indices
            .iter()
            .map(|&i| sks[i].sign(msg, POP_DST))
            .collect();
        let agg_sig = aggregate_sigs(&participating_sigs).unwrap();
        SubsetBlsSigWitness {
            full_pubkeys: pks,
            participation_bitmap: bitmap,
            agg_sig,
            msg: msg.to_vec(),
            dst: POP_DST.to_vec(),
        }
    }

    #[test]
    fn full_participation_filters_to_full_committee() {
        let w = build_subset_witness(4, &[0, 1, 2, 3], b"slot 100");
        assert_eq!(w.num_participants(), 4);
        let sub = w.to_bls_sig_witness().expect("full participation must verify");
        assert_eq!(sub.pubkeys.len(), 4);
        assert_eq!(&sub.pubkeys[..], &w.full_pubkeys[..]);
    }

    #[test]
    fn partial_participation_filters_correctly() {
        // 5-member committee; only members 0, 2, 4 participate.
        let w = build_subset_witness(5, &[0, 2, 4], b"slot 200");
        assert_eq!(w.num_participants(), 3);
        let sub = w.to_bls_sig_witness().expect("partial participation must verify");
        assert_eq!(sub.pubkeys.len(), 3);
        assert_eq!(sub.pubkeys[0], w.full_pubkeys[0]);
        assert_eq!(sub.pubkeys[1], w.full_pubkeys[2]);
        assert_eq!(sub.pubkeys[2], w.full_pubkeys[4]);
    }

    #[test]
    fn empty_bitmap_rejected() {
        let mut w = build_subset_witness(3, &[0, 1, 2], b"x");
        w.participation_bitmap = vec![false; 3];
        let r = w.to_bls_sig_witness();
        assert!(matches!(r, Err(SubsetBlsError::NoParticipants)));
    }

    #[test]
    fn bitmap_length_mismatch_rejected() {
        let mut w = build_subset_witness(3, &[0, 1, 2], b"x");
        w.participation_bitmap.push(false); // length 4 vs pubkeys 3
        let r = w.to_bls_sig_witness();
        assert!(matches!(
            r,
            Err(SubsetBlsError::BitmapLengthMismatch { pubkeys: 3, bitmap: 4 })
        ));
    }

    #[test]
    fn tampered_aggregate_signature_rejected() {
        let mut w = build_subset_witness(4, &[0, 1, 2, 3], b"slot 5");
        // Flip a byte in the aggregate signature; fast_aggregate_verify
        // must reject.
        w.agg_sig.0[20] ^= 0x01;
        let r = w.to_bls_sig_witness();
        assert!(matches!(r, Err(SubsetBlsError::HostAggregateVerifyFailed)));
    }

    /// Critical regression: passing the WRONG committee subset (the
    /// agg_sig matches members 0+1+2 but the bitmap claims 0+1+3) must
    /// be rejected. Demonstrates the host-side oracle catches
    /// mismatches between the claimed participating subset and the
    /// signed-by subset.
    #[test]
    fn bitmap_mismatch_with_signed_subset_rejected() {
        let (sks, pks) = build_committee(4);
        let msg = b"slot 7";
        // Sign with members 0, 1, 2.
        let sigs: Vec<Signature> = [0, 1, 2]
            .iter()
            .map(|&i| sks[i].sign(msg, POP_DST))
            .collect();
        let agg_sig = aggregate_sigs(&sigs).unwrap();
        // Claim members 0, 1, 3 participated (mismatched).
        let bitmap = vec![true, true, false, true];
        let w = SubsetBlsSigWitness {
            full_pubkeys: pks,
            participation_bitmap: bitmap,
            agg_sig,
            msg: msg.to_vec(),
            dst: POP_DST.to_vec(),
        };
        let r = w.to_bls_sig_witness();
        assert!(
            matches!(r, Err(SubsetBlsError::HostAggregateVerifyFailed)),
            "bitmap mismatching signed subset must be caught by host-side verify"
        );
    }

    /// SubsetBlsSigWitness can be threaded into the existing trace
    /// builder via `to_bls_sig_witness()`. Smoke check.
    #[test]
    fn subset_witness_threads_into_existing_trace_builder() {
        use crate::bls_sig_constraints::build_bls_sig_trace_polynomials;
        use crate::field::CurveType;

        let w = build_subset_witness(3, &[0, 2], b"slot 42");
        let sub = w.to_bls_sig_witness().unwrap();
        let trace = build_bls_sig_trace_polynomials(&sub, CurveType::Bls48581);
        assert!(trace.num_rows >= 1);
    }
}
