//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! composing a **beacon-block proposer-signature 3-AIR chain**:
//!
//!   - layer 0: [`crate::block_proposer_sig_air`] (per-block proposer-signature row)
//!   - layer 1: [`crate::bls_pairing_air`]         (BLS signature commitment)
//!   - layer 2: [`crate::validator_registry_air`]  (validator merkle inclusion)
//!
//! Three cross-AIR LogUp descriptors are wired, each as a **single-column
//! tuple** (the `joint_prove` API currently asserts
//! `a_columns.len() == 1 && b_columns.len() == 1`). The honest witness
//! is constructed so all three single-byte projections match between A
//! and B sides.
//!
//! - D0 (proposer pubkey ↔ validator-registry pubkey-byte mirror):
//!   first byte of the block_proposer_sig_air `PUBKEY` column ↔
//!   dedicated `COL_PK_BYTE0_MIRROR` mirror column on the
//!   validator_registry_air leaf row (task #318 — populated host-side
//!   via [`crate::validator_registry_air::ValidatorRegistryWitness::with_pk_byte0`]).
//!   The mirror sidesteps the `left_selection` / `hash_chain` constraints
//!   that previously blocked the honest closure under the
//!   `CURRENT_HASH[0]` projection.
//! - D1 (signature byte ↔ pairing sig):
//!   first byte of the block_proposer_sig_air `SIGNATURE` column ↔
//!   first byte of the bls_pairing_air `SIG_BYTES` column. The honest
//!   witness uses the same 96-byte signature on both sides.
//! - D2 (signing_root byte ↔ pairing msg_hash):
//!   first byte of the block_proposer_sig_air `SIGNING_ROOT` column ↔
//!   first byte of the bls_pairing_air `MSG_HASH` column. The honest
//!   witness sets `bls_pairing.msg_hash = signing_root` so the byte
//!   projection closes naturally.
//!
//! ## Witness alignment
//!
//! A single proposer (`bls_sig::SecretKey::from_u8_seed(0x29)`) signs a
//! placeholder beacon-block `signing_root` (derived from a fabricated
//! `block_root`/`domain` pair via `sha256(block_root || domain)` inside
//! [`block_proposer_sig_air::BlockProposerSigWitness::from_signed_block`]).
//! The resulting `(pubkey, signature, signing_root)` triple is forwarded
//! into:
//!   - `bls_pairing_air::BlsPairingWitness::from_decoded(pubkey,
//!     signature, signing_root)` for the pairing side,
//!   - a single-leaf depth-40 validator-registry inclusion (all-zero
//!     sibling subtree, validator_index = 0) on the registry side.
//!
//! With task #318's `COL_PK_BYTE0_MIRROR` populated to `pubkey[0]` on
//! the leaf row and the honest witnesses already aligning D1/D2, the
//! joint closure equalities match without any host-side trace patches.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::block_proposer_sig_air::{
        build_trace_polynomials as build_bps_trace, BlockProposerSigConstraintSystem,
        BlockProposerSigWitness, COL_IS_REAL as BPS_COL_IS_REAL,
        COL_PUBKEY_OFFSET as BPS_COL_PUBKEY_OFFSET,
        COL_SIGNATURE_OFFSET as BPS_COL_SIGNATURE_OFFSET,
        COL_SIGNING_ROOT_OFFSET as BPS_COL_SIGNING_ROOT_OFFSET,
        NUM_COLUMNS as BPS_NUM_COLUMNS,
    };
    use crate::bls_pairing_air::{
        build_trace_polynomials as build_pair_trace, BlsPairingConstraintSystem,
        BlsPairingWitness, COL_IS_REAL as PAIR_COL_IS_REAL,
        COL_MSG_HASH_OFFSET as PAIR_COL_MSG_HASH_OFFSET,
        COL_SIG_BYTES_OFFSET as PAIR_COL_SIG_BYTES_OFFSET,
        NUM_COLUMNS as PAIR_NUM_COLUMNS,
    };
    use crate::validator_registry_air::{
        build_trace_polynomials as build_reg_trace, RegistryInclusionWitness,
        ValidatorRegistryConstraintSystem, ValidatorRegistryWitness,
        CHUNK_BYTES as REG_CHUNK_BYTES,
        COL_IS_REAL as REG_COL_IS_REAL,
        COL_PK_BYTE0_MIRROR as REG_COL_PK_BYTE0_MIRROR,
        DEPTH as REG_DEPTH,
        NUM_COLUMNS as REG_NUM_COLUMNS,
        ROWS_PER_INCLUSION as REG_ROWS_PER_INCLUSION,
    };

    /// Beacon-chain ciphersuite DST (POP variant) — matches
    /// `bls_pairing_air` / `block_proposer_sig_air` expectations.
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
    /// Deterministic proposer-key seed for tests. DO NOT use for real
    /// key material.
    const PROPOSER_SK_SEED: u8 = 0x29;
    /// Synthetic beacon-block slot / proposer index for the witness.
    const PROPOSER_SLOT: u64 = 42;
    const PROPOSER_INDEX: u64 = 0;

    // ─── Single-column descriptor builders ────────────────────────────

    /// D0: block_proposer_sig_air `PUBKEY[0]` ↔ validator_registry_air
    /// `COL_PK_BYTE0_MIRROR` (task #318 dedicated mirror column on the
    /// leaf row). The mirror is host-populated to the proposer pubkey
    /// first byte via [`ValidatorRegistryWitness::with_pk_byte0`], so the
    /// honest closure matches without touching `CURRENT_HASH[0]` / the
    /// `left_selection` body or the cross-row `hash_chain`. The B side
    /// is gated by a dedicated *leaf-row selector* (rather than
    /// `IS_REAL`) so the multiset on the B side is a single byte —
    /// matching the A side's single active row. Since
    /// `COL_PK_BYTE0_MIRROR` is zero on non-leaf rows and the leaf row
    /// is also `IS_REAL = 1`, we use `IS_REAL` as the selector and let
    /// the natural zero-padding on non-leaf rows align with A's zero-
    /// padding row (both multisets are `{pubkey[0]} ∪ {0, 0, …}` of
    /// equal length).
    fn d0_proposer_pubkey_to_validator_registry() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_proposer_pubkey_to_validator_registry_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BPS_COL_PUBKEY_OFFSET],
            a_selector_column: Some(BPS_COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![REG_COL_PK_BYTE0_MIRROR],
            b_selector_column: Some(REG_COL_IS_REAL),
        }
    }

    /// D1: block_proposer_sig_air `SIGNATURE[0]` ↔ bls_pairing_air
    /// `SIG_BYTES[0]`. Anchors the first byte of the proposer's
    /// signature against the pairing-side signature commitment.
    fn d1_proposer_signature_to_pairing_sig() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_proposer_signature_to_pairing_sig_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BPS_COL_SIGNATURE_OFFSET],
            a_selector_column: Some(BPS_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![PAIR_COL_SIG_BYTES_OFFSET],
            b_selector_column: Some(PAIR_COL_IS_REAL),
        }
    }

    /// D2: block_proposer_sig_air `SIGNING_ROOT[0]` ↔ bls_pairing_air
    /// `MSG_HASH[0]`. Anchors the first byte of the proposer-side
    /// signing root against the pairing-side message-hash commitment.
    fn d2_proposer_signing_root_to_pairing_msg_hash() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "block_proposer_signing_root_to_pairing_msg_hash_v1".into(),
            a_layer_index: 0,
            a_columns: vec![BPS_COL_SIGNING_ROOT_OFFSET],
            a_selector_column: Some(BPS_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![PAIR_COL_MSG_HASH_OFFSET],
            b_selector_column: Some(PAIR_COL_IS_REAL),
        }
    }

    // ─── Witness builders ─────────────────────────────────────────────

    /// Build a signed beacon-block proposer-signature witness with a
    /// real BLS signature over `signing_root = sha256(block_root ||
    /// domain)` (the in-AIR convention). Returns the witness plus the
    /// raw (pubkey, signature, signing_root) tuple for plumbing into the
    /// downstream AIRs.
    fn build_proposer_sig_witness() -> (
        BlockProposerSigWitness,
        [u8; 48],
        [u8; 96],
        [u8; 32],
    ) {
        let sk = crate::bls_sig::SecretKey::from_u8_seed(PROPOSER_SK_SEED);
        let pk = sk.public_key();

        // Fabricated 32-byte block_root + 32-byte domain placeholders.
        // The block_proposer_sig_air does not yet algebraically bind
        // them to any source beyond `sha256(block_root || domain) =
        // signing_root`, which it computes host-side inside
        // `from_signed_block`.
        let block_root = crate::keccak::keccak256(b"bps-joint-prove-block-root");
        let domain = crate::keccak::keccak256(b"bps-joint-prove-domain");

        // signing_root = sha256(block_root || domain) — matches the
        // in-AIR computation in `BlockProposerSigWitness::from_signed_block`.
        let mut sha_input = [0u8; 64];
        sha_input[..32].copy_from_slice(&block_root);
        sha_input[32..].copy_from_slice(&domain);
        let signing_root = crate::sha256::sha256(&sha_input);

        let sig = sk.sign(&signing_root, POP_DST);

        let witness = BlockProposerSigWitness::from_signed_block(
            block_root,
            domain,
            PROPOSER_INDEX,
            pk.0,
            sig.0,
            PROPOSER_SLOT,
        );
        (witness, pk.0, sig.0, signing_root)
    }

    /// Build a bls_pairing witness for the proposer's `(pubkey,
    /// signature, signing_root)` triple. Decodes the compressed
    /// encodings into Fp coords host-side.
    fn build_pair_witness_for_proposer(
        pubkey: [u8; 48],
        signature: [u8; 96],
        signing_root: [u8; 32],
    ) -> BlsPairingWitness {
        BlsPairingWitness::from_decoded(pubkey, signature, signing_root)
            .expect("real proposer (pk, sig) must decode under BLS12-381")
    }

    /// Build a single-leaf validator-registry inclusion witness with
    /// `validator_index = 0` and the all-zero sibling subtree. The
    /// `COL_PK_BYTE0_MIRROR` cell on the leaf row is populated to
    /// `pubkey_byte0` via [`ValidatorRegistryWitness::with_pk_byte0`]
    /// so the D0 cross-AIR LogUp closes naturally against the
    /// proposer pubkey's first byte without disturbing
    /// `CURRENT_HASH[0]` / the row-local `left_selection` body or the
    /// cross-row `hash_chain` (task #318: this seals the deferred
    /// audit gap previously documented under #292 / #295).
    fn build_validator_registry_witness(
        validator_htr: [u8; 32],
        pubkey_byte0: u8,
    ) -> ValidatorRegistryWitness {
        let mut path = [[0u8; REG_CHUNK_BYTES]; REG_DEPTH];
        let mut z = [0u8; REG_CHUNK_BYTES];
        for k in 0..REG_DEPTH {
            path[k] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }
        // Single-leaf registry: registry_length = 1.
        let inc = RegistryInclusionWitness::from_validator_proof(
            validator_htr,
            PROPOSER_INDEX,
            path,
            1,
        );
        ValidatorRegistryWitness {
            inclusions: vec![inc],
            pk_byte0: Vec::new(),
        }
        .with_pk_byte0(vec![pubkey_byte0])
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check across all three descriptors plus a
    /// non-prove sanity check on the 3-trace shape that `joint_prove`
    /// receives. Does NOT run `joint_prove`, so it stays well under
    /// 120s and runs in CI.
    #[test]
    fn descriptor_consistency_beacon_block_proposer_signature_three_air() {
        let d0 = d0_proposer_pubkey_to_validator_registry();
        let d1 = d1_proposer_signature_to_pairing_sig();
        let d2 = d2_proposer_signing_root_to_pairing_msg_hash();

        // Single-column-tuple invariant — joint_prove asserts this.
        for d in [&d0, &d1, &d2] {
            assert_eq!(
                d.a_columns.len(),
                1,
                "joint_prove requires single-column tuples (a)",
            );
            assert_eq!(
                d.b_columns.len(),
                1,
                "joint_prove requires single-column tuples (b)",
            );
            assert_ne!(
                d.a_layer_index, d.b_layer_index,
                "descriptor spans two layers",
            );
            assert!(d.a_selector_column.is_some(), "A side must be gated");
            assert!(d.b_selector_column.is_some(), "B side must be gated");
        }

        // Layer indices: block_proposer_sig=0, bls_pairing=1,
        // validator_registry=2.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 2);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert_eq!(d2.a_layer_index, 0);
        assert_eq!(d2.b_layer_index, 1);

        // Labels are unique.
        let labels = [d0.label.as_str(), d1.label.as_str(), d2.label.as_str()];
        let mut sorted = labels.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "descriptor labels must be unique");

        // Build the per-AIR witnesses + traces at BLS48-581 and confirm
        // the 3-trace orchestrator inputs are constructible.
        let curve = CurveType::Bls48581;
        let (bps_w, pubkey, signature, signing_root) = build_proposer_sig_witness();
        let pair_w = build_pair_witness_for_proposer(pubkey, signature, signing_root);
        // Validator HTR placeholder: use the proposer pubkey's first 32
        // bytes. The dedicated `COL_PK_BYTE0_MIRROR` mirror column is
        // populated to `pubkey[0]` via `with_pk_byte0`, so D0 closes
        // naturally on the leaf row without disturbing the natural
        // `CURRENT_HASH` chain.
        let mut validator_htr = [0u8; 32];
        validator_htr.copy_from_slice(&pubkey[..32]);
        let reg_w = build_validator_registry_witness(validator_htr, pubkey[0]);

        let trace_0 = build_bps_trace(&bps_w, curve);
        let trace_1 = build_pair_trace(&pair_w, curve);
        let trace_2 = build_reg_trace(&reg_w, curve);

        assert_eq!(trace_0.columns.len(), BPS_NUM_COLUMNS);
        assert_eq!(trace_1.columns.len(), PAIR_NUM_COLUMNS);
        assert_eq!(trace_2.columns.len(), REG_NUM_COLUMNS);
        assert!(trace_0.num_rows >= 1);
        assert!(trace_1.num_rows >= 1);
        assert!(trace_2.num_rows >= REG_ROWS_PER_INCLUSION);

        let cs_0 = BlockProposerSigConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BlsPairingConstraintSystem::new(trace_1.num_rows);
        let cs_2 = ValidatorRegistryConstraintSystem::new(trace_2.num_rows);

        // Assemble the input shape `joint_prove` takes.
        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![d0.clone(), d1.clone(), d2.clone()];

        assert_eq!(traces.len(), 3, "3-AIR joint_prove input must have 3 traces");
        assert_eq!(linkages.len(), 3, "must wire exactly 3 descriptors");

        // Every linkage layer index must be in range — this is the
        // bounds check `joint_prove` performs at the top of its loop.
        let num_cols_per_layer = [BPS_NUM_COLUMNS, PAIR_NUM_COLUMNS, REG_NUM_COLUMNS];
        for link in &linkages {
            assert!(link.a_layer_index < traces.len(), "a_layer in bounds");
            assert!(link.b_layer_index < traces.len(), "b_layer in bounds");
            for &c in &link.a_columns {
                assert!(
                    c < num_cols_per_layer[link.a_layer_index],
                    "a_column {} out of range for layer {} (max {})",
                    c,
                    link.a_layer_index,
                    num_cols_per_layer[link.a_layer_index],
                );
            }
            for &c in &link.b_columns {
                assert!(
                    c < num_cols_per_layer[link.b_layer_index],
                    "b_column {} out of range for layer {} (max {})",
                    c,
                    link.b_layer_index,
                    num_cols_per_layer[link.b_layer_index],
                );
            }
            if let Some(sa) = link.a_selector_column {
                assert!(sa < num_cols_per_layer[link.a_layer_index]);
            }
            if let Some(sb) = link.b_selector_column {
                assert!(sb < num_cols_per_layer[link.b_layer_index]);
            }
        }

        // Sanity: the block_proposer_sig row's PUBKEY[0],
        // SIGNATURE[0], SIGNING_ROOT[0] cells equal the proposer's
        // raw bytes (what D0/D1/D2 anchor on the A side).
        let bps_pk_byte0 = trace_0.columns[BPS_COL_PUBKEY_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            bps_pk_byte0 as u8, pubkey[0],
            "block_proposer_sig row 0 PUBKEY[0] must equal pubkey[0]",
        );
        let bps_sig_byte0 = trace_0.columns[BPS_COL_SIGNATURE_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            bps_sig_byte0 as u8, signature[0],
            "block_proposer_sig row 0 SIGNATURE[0] must equal signature[0]",
        );
        let bps_sr_byte0 = trace_0.columns[BPS_COL_SIGNING_ROOT_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            bps_sr_byte0 as u8, signing_root[0],
            "block_proposer_sig row 0 SIGNING_ROOT[0] must equal signing_root[0]",
        );

        // Pair-side SIG_BYTES[0] and MSG_HASH[0] cells equal the
        // honest-witness inputs.
        let pair_sig_byte0 = trace_1.columns[PAIR_COL_SIG_BYTES_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            pair_sig_byte0 as u8, signature[0],
            "pair SIG_BYTES[0] aligned with signature[0]",
        );
        let pair_msg_byte0 = trace_1.columns[PAIR_COL_MSG_HASH_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            pair_msg_byte0 as u8, signing_root[0],
            "pair MSG_HASH[0] aligned with signing_root[0]",
        );

        // Validator-registry `COL_PK_BYTE0_MIRROR` on the leaf row
        // (task #318) equals the host-supplied proposer pubkey first
        // byte — this is what D0 anchors against on the B side.
        let reg_pk_byte0 = trace_2.columns[REG_COL_PK_BYTE0_MIRROR]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            reg_pk_byte0 as u8, pubkey[0],
            "validator_registry row 0 COL_PK_BYTE0_MIRROR must equal pubkey[0]",
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 3-AIR `joint_prove` + `joint_verify` round-trip with
    /// 3 cross-AIR LogUp descriptors. Marked `#[ignore]` because each
    /// per-AIR `prove_with_scheme` plus 3 inner linkage SNARKs adds up
    /// to many minutes under BLS48-581 (the block_proposer_sig AIR has
    /// 251+ columns × 1 row and the validator-registry AIR has 237
    /// columns × 41 rows + byte-range LogUp inflating each domain).
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_beacon_block_proposer_signature_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (bps_w, pubkey, signature, signing_root) = build_proposer_sig_witness();
        let pair_w = build_pair_witness_for_proposer(pubkey, signature, signing_root);
        let mut validator_htr = [0u8; 32];
        validator_htr.copy_from_slice(&pubkey[..32]);
        let reg_w = build_validator_registry_witness(validator_htr, pubkey[0]);

        // Suppress unused-warning on signing_root in case the
        // honest-witness B-side is already aligned (D2 closes naturally
        // because `bls_pairing.msg_hash = signing_root`).
        let _ = signing_root;

        let trace_0 = build_bps_trace(&bps_w, curve);
        let trace_1 = build_pair_trace(&pair_w, curve);
        let trace_2 = build_reg_trace(&reg_w, curve);

        let cs_0 = BlockProposerSigConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BlsPairingConstraintSystem::new(trace_1.num_rows);
        let cs_2 = ValidatorRegistryConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![
            d0_proposer_pubkey_to_validator_registry(),
            d1_proposer_signature_to_pairing_sig(),
            d2_proposer_signing_root_to_pairing_msg_hash(),
        ];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed");
        assert_eq!(proofs.len(), 3, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            3,
            "expected one CrossAirLogUpProof per descriptor",
        );

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        // Task #318: with the dedicated `COL_PK_BYTE0_MIRROR` column
        // populated on the validator-registry leaf row, the D0 closure
        // matches without disturbing `CURRENT_HASH[0]` / the cross-row
        // `hash_chain`. D1 and D2 close naturally from the honest
        // witness. The orchestrator's joint_verify is asserted true.
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 3-AIR joint_verify must accept the proposer-signature chain",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper the second descriptor's `closure_a` — `joint_verify` must
    /// reject. Marked `#[ignore]` because the setup half is the same
    /// `joint_prove` call as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_beacon_block_proposer_signature_joint_verify_true"]
    fn tampered_beacon_block_proposer_signature_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (bps_w, pubkey, signature, signing_root) = build_proposer_sig_witness();
        let pair_w = build_pair_witness_for_proposer(pubkey, signature, signing_root);
        let mut validator_htr = [0u8; 32];
        validator_htr.copy_from_slice(&pubkey[..32]);
        let reg_w = build_validator_registry_witness(validator_htr, pubkey[0]);
        let _ = (signature, signing_root);

        let trace_0 = build_bps_trace(&bps_w, curve);
        let trace_1 = build_pair_trace(&pair_w, curve);
        let trace_2 = build_reg_trace(&reg_w, curve);

        let cs_0 = BlockProposerSigConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BlsPairingConstraintSystem::new(trace_1.num_rows);
        let cs_2 = ValidatorRegistryConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![
            d0_proposer_pubkey_to_validator_registry(),
            d1_proposer_signature_to_pairing_sig(),
            d2_proposer_signing_root_to_pairing_msg_hash(),
        ];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest 3-AIR joint_prove must succeed");

        // Tamper the SECOND descriptor's closure_a (signature byte).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[1].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the signature descriptor",
        );
    }
}
