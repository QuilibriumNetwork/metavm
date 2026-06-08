//! Ethereum beacon-chain SSZ container types.
//!
//! Pure reference implementations of the Phase-0 beacon-chain containers
//! consumed by the consensus-layer verifier. Each type carries a
//! `hash_tree_root()` that mirrors the SSZ spec exactly, layered on top
//! of the primitives in [`crate::ssz`].
//!
//! These are strictly data types — no AIR, no witness, no prover hooks. They
//! exist so that the Gasper / finality circuit has a well-typed anchor for its
//! public inputs, and so that test vectors from `consensus-specs` can be
//! checked against a single source of truth.
//!
//! # References
//! * <https://github.com/ethereum/consensus-specs/blob/dev/specs/phase0/beacon-chain.md>
//! * `tests/core/pyspec_tests/phase0/ssz_static/` in the consensus-specs repo.

use crate::ssz::{
    hash_tree_root_bytes_fixed, hash_tree_root_bytes, hash_tree_root_bool,
    hash_tree_root_container, hash_tree_root_uint, merkleize_chunks,
    mix_in_length, pack_bytes, Chunk,
};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

/// A slot number — 12-second window in the beacon chain.
pub type Slot = u64;
/// An epoch number — 32 slots.
pub type Epoch = u64;
/// Index into the validator registry.
pub type ValidatorIndex = u64;
/// Index into a beacon committee.
pub type CommitteeIndex = u64;
/// Effective balance / stake unit, denominated in Gwei (10^-9 ETH).
pub type Gwei = u64;

/// Phase 0 limit: `MAX_VALIDATORS_PER_COMMITTEE = 2048`. Used as the `List[..]`
/// cap for `IndexedAttestation.attesting_indices`.
pub const MAX_VALIDATORS_PER_COMMITTEE: u64 = 2048;

/// Phase 0 constant: `VALIDATOR_REGISTRY_LIMIT = 2^40`.
pub const VALIDATOR_REGISTRY_LIMIT: u64 = 1u64 << 40;

/// Phase 0 constant: `JUSTIFICATION_BITS_LENGTH = 4`.
pub const JUSTIFICATION_BITS_LENGTH: u64 = 4;

// ---------------------------------------------------------------------------
// Local helpers (SSZ primitives not present in crate::ssz)
// ---------------------------------------------------------------------------

/// hashTreeRoot(List[uint64, N]):
///
/// Pack the u64s (LE, 8 bytes each -> 4 per 32-byte chunk), merkleize into at
/// most `ceil(max_length * 8 / 32)` chunks, then mix in the actual element
/// count.
///
/// `values` are the logical u64 elements (not bytes); `max_length` is the
/// type-level element cap.
pub fn hash_tree_root_list_uint64(values: &[u64], max_length: u64) -> Chunk {
    // Serialize values as a single LE byte stream (8 bytes per u64).
    let mut buf = Vec::with_capacity(values.len() * 8);
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    // chunks_limit counts 32-byte chunks; each chunk holds 4 u64s, i.e. 32 bytes.
    // ceil((max_length * 8) / 32) = ceil(max_length / 4).
    let chunks_limit = (max_length * 8).div_ceil(32);
    let chunks = pack_bytes(&buf);
    let root = merkleize_chunks(&chunks, Some(chunks_limit));
    mix_in_length(root, values.len() as u64)
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// Gasper finality checkpoint: `(epoch, beacon-block-root)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Checkpoint {
    pub epoch: Epoch,
    pub root: [u8; 32],
}

impl Checkpoint {
    pub fn hash_tree_root(&self) -> Chunk {
        merkleize_chunks(&[hash_tree_root_uint(self.epoch), self.root], None)
    }
}

// ---------------------------------------------------------------------------
// AttestationData
// ---------------------------------------------------------------------------

/// The attested-to consensus data. Paired with an aggregate signature this
/// forms an attestation. Five fields, so it pads to a depth-3 merkle tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttestationData {
    pub slot: Slot,
    pub index: CommitteeIndex,
    pub beacon_block_root: [u8; 32],
    pub source: Checkpoint,
    pub target: Checkpoint,
}

impl AttestationData {
    pub fn hash_tree_root(&self) -> Chunk {
        hash_tree_root_container(&[
            hash_tree_root_uint(self.slot),
            hash_tree_root_uint(self.index),
            self.beacon_block_root,
            self.source.hash_tree_root(),
            self.target.hash_tree_root(),
        ])
    }
}

/// Type alias for clarity — the root of an `AttestationData` as consumed by
/// signature verification / aggregation.
pub type AttestationDataRoot = Chunk;

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// A single entry in the beacon-chain validator registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Validator {
    pub pubkey: [u8; 48],
    pub withdrawal_credentials: [u8; 32],
    pub effective_balance: Gwei,
    pub slashed: bool,
    pub activation_eligibility_epoch: Epoch,
    pub activation_epoch: Epoch,
    pub exit_epoch: Epoch,
    pub withdrawable_epoch: Epoch,
}

impl Default for Validator {
    fn default() -> Self {
        Self {
            pubkey: [0u8; 48],
            withdrawal_credentials: [0u8; 32],
            effective_balance: 0,
            slashed: false,
            activation_eligibility_epoch: 0,
            activation_epoch: 0,
            exit_epoch: 0,
            withdrawable_epoch: 0,
        }
    }
}

impl Validator {
    /// Set the `pubkey` field from a `bls_sig::PublicKey` (just copies the
    /// 48 compressed bytes). Convenience for wiring BLS pubkeys into the
    /// validator registry.
    pub fn with_pubkey(mut self, pk: &crate::bls_sig::PublicKey) -> Self {
        self.pubkey = pk.0;
        self
    }

    pub fn hash_tree_root(&self) -> Chunk {
        // Bytes48: 48 bytes pack to 2 chunks (32 + 16 zero-padded), limit = ceil(48/32) = 2.
        let pubkey_root = hash_tree_root_bytes_fixed(&self.pubkey, 2);
        hash_tree_root_container(&[
            pubkey_root,
            self.withdrawal_credentials,
            hash_tree_root_uint(self.effective_balance),
            hash_tree_root_bool(self.slashed),
            hash_tree_root_uint(self.activation_eligibility_epoch),
            hash_tree_root_uint(self.activation_epoch),
            hash_tree_root_uint(self.exit_epoch),
            hash_tree_root_uint(self.withdrawable_epoch),
        ])
    }
}

// ---------------------------------------------------------------------------
// BeaconBlockHeader
// ---------------------------------------------------------------------------

/// The header portion of a signed beacon block (5 fields: slot, proposer_index,
/// parent_root, state_root, body_root). Pads to a depth-3 tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BeaconBlockHeader {
    pub slot: Slot,
    pub proposer_index: ValidatorIndex,
    pub parent_root: [u8; 32],
    pub state_root: [u8; 32],
    pub body_root: [u8; 32],
}

impl BeaconBlockHeader {
    pub fn hash_tree_root(&self) -> Chunk {
        hash_tree_root_container(&[
            hash_tree_root_uint(self.slot),
            hash_tree_root_uint(self.proposer_index),
            self.parent_root,
            self.state_root,
            self.body_root,
        ])
    }
}

// ---------------------------------------------------------------------------
// SignedBeaconBlockHeader
// ---------------------------------------------------------------------------

/// A beacon block header with its BLS aggregate signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedBeaconBlockHeader {
    pub message: BeaconBlockHeader,
    pub signature: [u8; 96],
}

impl Default for SignedBeaconBlockHeader {
    fn default() -> Self {
        Self { message: BeaconBlockHeader::default(), signature: [0u8; 96] }
    }
}

impl SignedBeaconBlockHeader {
    pub fn hash_tree_root(&self) -> Chunk {
        // Bytes96: 96 bytes pack to exactly 3 chunks (no padding of bytes, but
        // the merkle tree pads 3 -> 4 leaves).
        let sig_root = hash_tree_root_bytes_fixed(&self.signature, 3);
        hash_tree_root_container(&[self.message.hash_tree_root(), sig_root])
    }
}

// ---------------------------------------------------------------------------
// IndexedAttestation
// ---------------------------------------------------------------------------

/// Attestation keyed by explicit validator indices (as opposed to a committee
/// bitfield). Three fields: list of indices, attestation data, aggregate sig.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedAttestation {
    /// `List[ValidatorIndex, MAX_VALIDATORS_PER_COMMITTEE = 2048]`.
    pub attesting_indices: Vec<ValidatorIndex>,
    pub data: AttestationData,
    pub signature: [u8; 96],
}

impl Default for IndexedAttestation {
    fn default() -> Self {
        Self {
            attesting_indices: Vec::new(),
            data: AttestationData::default(),
            signature: [0u8; 96],
        }
    }
}

impl IndexedAttestation {
    pub fn hash_tree_root(&self) -> Chunk {
        let indices_root =
            hash_tree_root_list_uint64(&self.attesting_indices, MAX_VALIDATORS_PER_COMMITTEE);
        let data_root = self.data.hash_tree_root();
        let sig_root = hash_tree_root_bytes_fixed(&self.signature, 3);
        hash_tree_root_container(&[indices_root, data_root, sig_root])
    }
}

// ---------------------------------------------------------------------------
// Minimal BeaconState (Gasper-relevant subset)
// ---------------------------------------------------------------------------

/// A deliberately minimal `BeaconState`-shaped container exposing only the
/// fields the Phase 10 Gasper / stake-weighting circuit needs. The real
/// beacon state has 30+ fields; this projection keeps the validator registry,
/// balances, justification bits, and the three Gasper checkpoints.
///
/// `hash_tree_root()` on this type is intentionally NOT a drop-in replacement
/// for the full `BeaconState` root — it's a 6-field container root over the
/// projected fields only. Consumers should treat it as an internal commitment
/// to the Gasper-relevant view of state, not as the canonical state root.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GasperStateView {
    /// `List[Validator, VALIDATOR_REGISTRY_LIMIT]`.
    pub validators: Vec<Validator>,
    /// `List[Gwei, VALIDATOR_REGISTRY_LIMIT]`.
    pub balances: Vec<Gwei>,
    /// `Bitvector[JUSTIFICATION_BITS_LENGTH = 4]` — packed into the low nibble
    /// of a single byte (bit `i` at position `i`).
    pub justification_bits: u8,
    pub previous_justified_checkpoint: Checkpoint,
    pub current_justified_checkpoint: Checkpoint,
    pub finalized_checkpoint: Checkpoint,
}

impl GasperStateView {
    /// Root of the validator list: `merkleize(pack(validator_roots), VALIDATOR_REGISTRY_LIMIT)`
    /// then `mix_in_length(num_validators)`.
    pub fn validators_root(&self) -> Chunk {
        let leaves: Vec<Chunk> = self.validators.iter().map(|v| v.hash_tree_root()).collect();
        let root = merkleize_chunks(&leaves, Some(VALIDATOR_REGISTRY_LIMIT));
        mix_in_length(root, self.validators.len() as u64)
    }

    /// Root of the balances list: List[uint64, VALIDATOR_REGISTRY_LIMIT].
    pub fn balances_root(&self) -> Chunk {
        hash_tree_root_list_uint64(&self.balances, VALIDATOR_REGISTRY_LIMIT)
    }

    /// Root of the 4-bit Bitvector. Bitvector is merkleized without mixing in
    /// length (unlike Bitlist); 4 bits pack to 1 byte, pad to one chunk.
    pub fn justification_bits_root(&self) -> Chunk {
        // Mask to the low 4 bits for safety.
        let byte = self.justification_bits & 0x0f;
        hash_tree_root_bytes(&[byte])
    }

    pub fn hash_tree_root(&self) -> Chunk {
        hash_tree_root_container(&[
            self.validators_root(),
            self.balances_root(),
            self.justification_bits_root(),
            self.previous_justified_checkpoint.hash_tree_root(),
            self.current_justified_checkpoint.hash_tree_root(),
            self.finalized_checkpoint.hash_tree_root(),
        ])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::sha256_pair;
    use crate::ssz::ZERO_CHUNK;

    fn hex(chunk: &Chunk) -> String {
        let mut s = String::with_capacity(64);
        for b in chunk {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    /// `zero_hash[d]` computed inline — mirrors `crate::ssz::zero_hash` (which
    /// is module-private). Used to lock test vectors to the canonical SSZ
    /// zero-tree values.
    fn zh(depth: u32) -> Chunk {
        let mut z = ZERO_CHUNK;
        for _ in 0..depth {
            z = sha256_pair(&z, &z);
        }
        z
    }

    // -----------------------------------------------------------------------
    // Checkpoint
    // -----------------------------------------------------------------------

    #[test]
    fn checkpoint_all_zero_is_zero_hash_1() {
        // Two fields, both zero: root = sha256(0^64) = zero_hash[1].
        let c = Checkpoint { epoch: 0, root: [0u8; 32] };
        assert_eq!(
            hex(&c.hash_tree_root()),
            "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        );
        assert_eq!(c.hash_tree_root(), zh(1));
    }

    #[test]
    fn checkpoint_manual_merkleization() {
        let c = Checkpoint {
            epoch: 42,
            root: [0xabu8; 32],
        };
        let expected = sha256_pair(&hash_tree_root_uint(42), &[0xabu8; 32]);
        assert_eq!(c.hash_tree_root(), expected);
    }

    #[test]
    fn checkpoint_tamper_changes_root() {
        let base = Checkpoint { epoch: 5, root: [1u8; 32] };
        let e_tampered = Checkpoint { epoch: 6, root: [1u8; 32] };
        let r_tampered = Checkpoint { epoch: 5, root: [2u8; 32] };
        assert_ne!(base.hash_tree_root(), e_tampered.hash_tree_root());
        assert_ne!(base.hash_tree_root(), r_tampered.hash_tree_root());
    }

    // -----------------------------------------------------------------------
    // AttestationData
    // -----------------------------------------------------------------------

    #[test]
    fn attestation_data_all_zero_matches_manual() {
        let d = AttestationData::default();
        let roots = [
            hash_tree_root_uint(0), // slot
            hash_tree_root_uint(0), // index
            [0u8; 32],              // beacon_block_root
            zh(1),                  // source Checkpoint all-zero
            zh(1),                  // target Checkpoint all-zero
        ];
        // 5 fields -> pad to 8. Compute manually:
        //   L1 = pair(roots[0], roots[1])  = pair(0^32, 0^32) = zh(1)
        //   L2 = pair(roots[2], roots[3])  = pair(0^32, zh(1))
        //   L3 = pair(roots[4], 0^32)      = pair(zh(1), 0^32)
        //   L4 = pair(0^32, 0^32)          = zh(1)
        //   T1 = pair(L1, L2); T2 = pair(L3, L4); root = pair(T1, T2)
        let l1 = sha256_pair(&roots[0], &roots[1]);
        let l2 = sha256_pair(&roots[2], &roots[3]);
        let l3 = sha256_pair(&roots[4], &ZERO_CHUNK);
        let l4 = sha256_pair(&ZERO_CHUNK, &ZERO_CHUNK);
        let t1 = sha256_pair(&l1, &l2);
        let t2 = sha256_pair(&l3, &l4);
        let expected = sha256_pair(&t1, &t2);
        assert_eq!(d.hash_tree_root(), expected);
    }

    #[test]
    fn attestation_data_tamper_each_field() {
        let base = AttestationData {
            slot: 10,
            index: 3,
            beacon_block_root: [1u8; 32],
            source: Checkpoint { epoch: 1, root: [2u8; 32] },
            target: Checkpoint { epoch: 2, root: [3u8; 32] },
        };
        let base_root = base.hash_tree_root();

        let mut t = base;
        t.slot = 11;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.index = 4;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.beacon_block_root[0] = 0xff;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.source.epoch = 99;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.target.root[31] = 0xff;
        assert_ne!(base_root, t.hash_tree_root());
    }

    // -----------------------------------------------------------------------
    // Validator
    // -----------------------------------------------------------------------

    #[test]
    fn validator_all_zero_regression_vector() {
        // 8 fields, all zero:
        //   pubkey_root = Bytes48 all-zero = zero_hash[1] (2 chunks -> 1 pair hash).
        //   withdrawal_credentials = ZERO_CHUNK
        //   effective_balance = ZERO_CHUNK (uint64 0)
        //   slashed = ZERO_CHUNK (bool false)
        //   4x epoch = ZERO_CHUNK
        //
        // 8 leaves, next_pow_of_two = 8, no padding. The tree reduces:
        //   pair(zh(1), 0) = sha256(zh(1)||0)      [leaves 0..1]
        //   pair(0, 0) = zh(1)                     [leaves 2..3]
        //   pair(0, 0) = zh(1)                     [leaves 4..5]
        //   pair(0, 0) = zh(1)                     [leaves 6..7]
        //   L = pair(pair(zh1,0), zh1)
        //   R = pair(zh1, zh1) = zh(2)
        //   root = pair(L, R)
        let v = Validator::default();
        let pk = zh(1);
        let l1 = sha256_pair(&pk, &ZERO_CHUNK);
        let l2 = sha256_pair(&ZERO_CHUNK, &ZERO_CHUNK); // = zh(1)
        let l3 = l2;
        let l4 = l2;
        let t1 = sha256_pair(&l1, &l2);
        let t2 = sha256_pair(&l3, &l4); // = zh(2)
        let expected = sha256_pair(&t1, &t2);
        assert_eq!(v.hash_tree_root(), expected);

        // Regression anchor: lock the exact hex so future refactors can't
        // silently change the root.
        assert_eq!(
            hex(&v.hash_tree_root()),
            "fa324a462bcb0f10c24c9e17c326a4e0ebad204feced523eccaf346c686f06ee"
        );
    }

    #[test]
    fn validator_tamper_each_field() {
        let base = Validator {
            pubkey: [0xaau8; 48],
            withdrawal_credentials: [0xbbu8; 32],
            effective_balance: 32_000_000_000,
            slashed: false,
            activation_eligibility_epoch: 1,
            activation_epoch: 2,
            exit_epoch: u64::MAX,
            withdrawable_epoch: u64::MAX,
        };
        let base_root = base.hash_tree_root();

        let mut t = base;
        t.pubkey[0] = 0xcc;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.withdrawal_credentials[0] = 0xcc;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.effective_balance += 1;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.slashed = true;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.activation_eligibility_epoch = 999;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.activation_epoch = 999;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.exit_epoch = 0;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.withdrawable_epoch = 0;
        assert_ne!(base_root, t.hash_tree_root());
    }

    // -----------------------------------------------------------------------
    // BeaconBlockHeader
    // -----------------------------------------------------------------------

    #[test]
    fn beacon_block_header_all_zero_manual() {
        // 5 fields all zero -> pad to 8, same shape as the all-zero
        // AttestationData but with 3 ZERO_CHUNKs and 2 ZERO_CHUNKs mixed in.
        // In fact all 5 field roots are ZERO_CHUNK (slot=0, proposer=0, 3x
        // 32-byte zero), so root = zh(3).
        let h = BeaconBlockHeader::default();
        assert_eq!(h.hash_tree_root(), zh(3));
        assert_eq!(
            hex(&h.hash_tree_root()),
            "c78009fdf07fc56a11f122370658a353aaa542ed63e44c4bc15ff4cd105ab33c"
        );
    }

    #[test]
    fn beacon_block_header_tamper_each_field() {
        let base = BeaconBlockHeader {
            slot: 100,
            proposer_index: 7,
            parent_root: [1u8; 32],
            state_root: [2u8; 32],
            body_root: [3u8; 32],
        };
        let base_root = base.hash_tree_root();

        let mut t = base;
        t.slot = 101;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.proposer_index = 8;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.parent_root[0] = 0xff;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.state_root[15] = 0xff;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.body_root[31] = 0xff;
        assert_ne!(base_root, t.hash_tree_root());
    }

    // -----------------------------------------------------------------------
    // SignedBeaconBlockHeader
    // -----------------------------------------------------------------------

    #[test]
    fn signed_beacon_block_header_all_zero_manual() {
        let sbh = SignedBeaconBlockHeader::default();
        // 2 fields, both zero-derived:
        //   message_root = zh(3) (from BeaconBlockHeader all-zero)
        //   sig_root     = Bytes96 all-zero, 3 chunks pad to 4 -> zh(2)
        // container root = sha256(message_root || sig_root)
        let expected = sha256_pair(&zh(3), &zh(2));
        assert_eq!(sbh.hash_tree_root(), expected);
    }

    #[test]
    fn signed_beacon_block_header_tamper_signature() {
        let base = SignedBeaconBlockHeader {
            message: BeaconBlockHeader {
                slot: 1,
                proposer_index: 2,
                parent_root: [0u8; 32],
                state_root: [0u8; 32],
                body_root: [0u8; 32],
            },
            signature: [0u8; 96],
        };
        let base_root = base.hash_tree_root();

        let mut t = base;
        t.signature[0] = 0xff;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.signature[95] = 0xff;
        assert_ne!(base_root, t.hash_tree_root());

        let mut t = base;
        t.message.slot = 2;
        assert_ne!(base_root, t.hash_tree_root());
    }

    // -----------------------------------------------------------------------
    // IndexedAttestation — exercises mix_in_length on the list field.
    // -----------------------------------------------------------------------

    #[test]
    fn indexed_attestation_small_list_manual() {
        let ia = IndexedAttestation {
            attesting_indices: vec![1, 2, 3],
            data: AttestationData::default(),
            signature: [0u8; 96],
        };

        // Expected indices_root:
        //   Pack [1,2,3] LE: 24 bytes in one 32-byte chunk.
        //   chunks_limit = ceil(2048*8/32) = 512 chunks, depth 9.
        //   merkleize([chunk], Some(512)) then mix_in_length(3).
        let mut packed = [0u8; 32];
        packed[..8].copy_from_slice(&1u64.to_le_bytes());
        packed[8..16].copy_from_slice(&2u64.to_le_bytes());
        packed[16..24].copy_from_slice(&3u64.to_le_bytes());
        let expected_indices_root =
            mix_in_length(merkleize_chunks(&[packed], Some(512)), 3);

        // Expected container root: merkleize([indices_root, data_root, sig_root]).
        let data_root = ia.data.hash_tree_root();
        let sig_root = hash_tree_root_bytes_fixed(&[0u8; 96], 3);
        let expected = hash_tree_root_container(&[expected_indices_root, data_root, sig_root]);
        assert_eq!(ia.hash_tree_root(), expected);
    }

    #[test]
    fn indexed_attestation_empty_list_differs_from_nonempty() {
        let empty = IndexedAttestation::default();
        let nonempty = IndexedAttestation {
            attesting_indices: vec![0], // single zero index
            ..Default::default()
        };
        // The packed bytes of [0u64] are all zero, same as the empty list's
        // merkle root before length-mixing — but mix_in_length differentiates
        // them (0 vs 1).
        assert_ne!(empty.hash_tree_root(), nonempty.hash_tree_root());
    }

    #[test]
    fn indexed_attestation_index_order_matters() {
        let a = IndexedAttestation {
            attesting_indices: vec![1, 2, 3],
            ..Default::default()
        };
        let b = IndexedAttestation {
            attesting_indices: vec![3, 2, 1],
            ..Default::default()
        };
        assert_ne!(a.hash_tree_root(), b.hash_tree_root());
    }

    // -----------------------------------------------------------------------
    // hash_tree_root_list_uint64 — direct tests.
    // -----------------------------------------------------------------------

    #[test]
    fn list_uint64_empty_max_4() {
        // Empty list, max=4 -> chunks_limit = ceil(4*8/32) = 1, merkleize
        // &[] with Some(1) = ZERO_CHUNK, mix_in_length(0) = zh(1).
        let root = hash_tree_root_list_uint64(&[], 4);
        assert_eq!(root, zh(1));
    }

    #[test]
    fn list_uint64_fits_one_chunk() {
        // [1,2,3,4] packs exactly into one 32-byte chunk.
        let values = [1u64, 2, 3, 4];
        let mut packed = [0u8; 32];
        for (i, v) in values.iter().enumerate() {
            packed[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
        }
        let expected = mix_in_length(packed, 4);
        assert_eq!(hash_tree_root_list_uint64(&values, 4), expected);
    }

    #[test]
    fn list_uint64_max_caps_tree_depth() {
        // Same payload, different max => different tree depth => different
        // merkle root before mixing => different final root.
        let values = [1u64, 2, 3];
        let r_small = hash_tree_root_list_uint64(&values, 4);
        let r_large = hash_tree_root_list_uint64(&values, 2048);
        assert_ne!(r_small, r_large);
    }

    // -----------------------------------------------------------------------
    // GasperStateView — smoke tests only (minimal spec subset).
    // -----------------------------------------------------------------------

    #[test]
    fn gasper_state_view_empty_is_deterministic() {
        let s = GasperStateView::default();
        let r1 = s.hash_tree_root();
        let r2 = s.hash_tree_root();
        assert_eq!(r1, r2);
    }

    #[test]
    fn gasper_state_view_adding_validator_changes_root() {
        let empty = GasperStateView::default();
        let with_v = GasperStateView {
            validators: vec![Validator::default()],
            ..Default::default()
        };
        assert_ne!(empty.hash_tree_root(), with_v.hash_tree_root());
    }

    #[test]
    fn gasper_state_view_justification_bits_masked() {
        // Only the low 4 bits of justification_bits are meaningful; higher
        // bits should be ignored.
        let a = GasperStateView {
            justification_bits: 0b0000_1111,
            ..Default::default()
        };
        let b = GasperStateView {
            justification_bits: 0b1111_1111,
            ..Default::default()
        };
        assert_eq!(a.hash_tree_root(), b.hash_tree_root());
    }

    #[test]
    fn gasper_state_view_finalized_checkpoint_tamper() {
        let base = GasperStateView::default();
        let tampered = GasperStateView {
            finalized_checkpoint: Checkpoint { epoch: 1, root: [0u8; 32] },
            ..Default::default()
        };
        assert_ne!(base.hash_tree_root(), tampered.hash_tree_root());
    }
}
