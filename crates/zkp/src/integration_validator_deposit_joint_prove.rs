//! Cross-AIR LogUp `joint_prove` / `joint_verify` integration test
//! composing a **validator-deposit 3-AIR chain**:
//!
//!   - layer 0: [`crate::deposit_tree_air`]    (depth-32 merkle inclusion).
//!   - layer 1: [`crate::bls_pairing_air`]      (BLS signature commitment).
//!   - layer 2: [`crate::sha256_extract`]       (SHA-256 pair-hash extraction).
//!
//! Three cross-AIR LogUp descriptors are wired, each as a **single-column
//! tuple** (the `joint_prove` API currently asserts
//! `a_columns.len() == 1 && b_columns.len() == 1`). The honest witness
//! is constructed so all three single-byte projections match between A
//! and B sides.
//!
//! - D0 (deposit_root ↔ block-header eth1_data deposit_root, via mix):
//!   first byte of the deposit_tree_air `DEPOSIT_ROOT` column ↔ first
//!   byte of the bls_pairing_air `MSG_HASH` column. This anchors the
//!   "deposit_root mixed into the block-header `eth1_data` field" byte
//!   commitment using `msg_hash` as the block-side placeholder until a
//!   block-header AIR is wired in.
//! - D1 (signature byte, task #309): first byte of the deposit_tree_air
//!   `COL_SIG_BYTE0_MIRROR` mirror column (populated algebraically by
//!   the witness builder with `deposit.signature[0]`) ↔ first byte of
//!   the bls_pairing_air `SIG_BYTES` column. The mirror column replaces
//!   the scaffold-era host-side `SIBLING[0]` / `RIGHT[0]` overrides,
//!   which clobbered cells bound by the AIR's row-local selection body.
//! - D2 (leaf_hash ↔ sha256_extract mirror, task #309): first byte of
//!   the deposit_tree_air `DEPOSIT_ROOT` column on the `IS_TOP` row ↔
//!   first byte of the sha256_extract `COL_MIRROR_BYTE0` mirror column
//!   (populated via [`crate::sha256_extract::Sha256ExtractWitness::with_mirror_byte0`]
//!   to carry `deposit_root[0]`). The mirror column replaces the
//!   scaffold-era host-side `OUTPUT_BYTE[0]` override.
//!
//! ## Witness alignment
//!
//! The deposit uses `bls_sig::SecretKey::from_u8_seed(0x37)` to derive a
//! pubkey + produce a real BLS signature over a 32-byte beacon-style
//! `signing_root` placeholder. The deposit_tree_air witness builds a
//! single-leaf depth-32 inclusion with the all-zero subtree siblings.
//! D0 closes because the pair witness sets `msg_hash = deposit_root`.
//! D1 closes because the deposit_tree witness builder mirrors
//! `deposit.signature[0]` into `COL_SIG_BYTE0_MIRROR` on every active
//! row (gated on `IS_TOP` for single-row multiset alignment with the
//! pair AIR's single row), and the pair witness pins `SIG_BYTES[0] =
//! deposit.signature[0]` naturally. D2 closes because the
//! sha256_extract witness's `mirror_byte0` field is set to
//! `[deposit_root[0]]` via `with_mirror_byte0`.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::deposit::DepositData;
    use crate::deposit_tree_air::{
        build_trace_polynomials as build_dep_trace, from_deposit_proof,
        DepositTreeConstraintSystem,
        COL_CURRENT_HASH_OFFSET as DEP_COL_CURRENT_HASH_OFFSET,
        COL_DEPOSIT_ROOT_OFFSET as DEP_COL_DEPOSIT_ROOT_OFFSET,
        COL_IS_TOP as DEP_COL_IS_TOP,
        COL_SIG_BYTE0_MIRROR as DEP_COL_SIG_BYTE0_MIRROR,
        CHUNK_BYTES as DEP_CHUNK_BYTES,
        DEPTH as DEP_DEPTH,
        NUM_COLUMNS as DEP_NUM_COLUMNS,
    };
    use crate::bls_pairing_air::{
        build_trace_polynomials as build_pair_trace, BlsPairingConstraintSystem,
        BlsPairingWitness, COL_IS_REAL as PAIR_COL_IS_REAL,
        COL_MSG_HASH_OFFSET as PAIR_COL_MSG_HASH_OFFSET,
        COL_SIG_BYTES_OFFSET as PAIR_COL_SIG_BYTES_OFFSET,
        NUM_COLUMNS as PAIR_NUM_COLUMNS,
    };
    use crate::sha256_extract::{
        build_trace_polynomials as build_sha_trace, Sha256ExtractConstraintSystem,
        Sha256ExtractWitness, COL_IS_REAL as SHA_COL_IS_REAL,
        COL_MIRROR_BYTE0 as SHA_COL_MIRROR_BYTE0,
        NUM_COLUMNS as SHA_NUM_COLUMNS,
    };

    /// Beacon-chain ciphersuite DST (POP variant).
    const POP_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
    /// Deterministic deposit-key seed for tests. DO NOT use for real
    /// key material.
    const DEPOSIT_SK_SEED: u8 = 0x37;
    /// Deposit amount in Gwei (32 ETH, max effective balance).
    const DEPOSIT_AMOUNT_GWEI: u64 = 32_000_000_000;
    /// Deposit index in the contract incremental merkle tree.
    const DEPOSIT_INDEX: u64 = 0;

    /// Beacon-chain deposit domain type (4 bytes, little-endian per spec).
    /// `DOMAIN_DEPOSIT = 0x03000000`.
    const DOMAIN_DEPOSIT: [u8; 4] = [0x03, 0x00, 0x00, 0x00];
    /// Genesis fork version on mainnet beacon chain (4 bytes, LE).
    const GENESIS_FORK_VERSION: [u8; 4] = [0x00, 0x00, 0x00, 0x00];

    /// Compute the 32-byte SSZ `hash_tree_root` of the deposit_message =
    /// (pubkey: Bytes48, withdrawal_credentials: Bytes32, amount: uint64).
    fn deposit_message_root(
        pubkey: &[u8; 48],
        withdrawal_credentials: &[u8; 32],
        amount_gwei: u64,
    ) -> [u8; 32] {
        let field_roots: [[u8; 32]; 3] = [
            crate::ssz::hash_tree_root_bytes_fixed(pubkey, 2),
            *withdrawal_credentials,
            crate::ssz::hash_tree_root_uint(amount_gwei),
        ];
        crate::ssz::merkleize_chunks(&field_roots, None)
    }

    /// Build the deposit signing-root domain. For deposits the beacon
    /// spec uses a special "genesis" fork data root: the `current_version`
    /// is the genesis fork version, and the `genesis_validators_root` is
    /// the zero hash (since deposits happen before genesis is finalized).
    /// `domain = DOMAIN_DEPOSIT || fork_data_root[0..28]`.
    fn compute_deposit_domain() -> [u8; 32] {
        // fork_data_root = HTR((current_version: Bytes4, genesis_validators_root: Bytes32)).
        let fork_version_root = crate::ssz::hash_tree_root_bytes_fixed(&GENESIS_FORK_VERSION, 1);
        let genesis_validators_root = [0u8; 32];
        let fork_data_root = crate::ssz::merkleize_chunks(
            &[fork_version_root, genesis_validators_root],
            None,
        );
        let mut domain = [0u8; 32];
        domain[0..4].copy_from_slice(&DOMAIN_DEPOSIT);
        domain[4..32].copy_from_slice(&fork_data_root[0..28]);
        domain
    }

    /// `signing_root = sha256(deposit_message_root || domain)`.
    fn compute_deposit_signing_root(
        message_root: &[u8; 32],
        domain: &[u8; 32],
    ) -> [u8; 32] {
        crate::sha256::sha256_pair(message_root, domain)
    }

    // ─── Single-column descriptor builders ────────────────────────────

    /// D0: deposit_tree_air `DEPOSIT_ROOT[0]` ↔ bls_pairing_air
    /// `MSG_HASH[0]`. Anchors the first byte of the claimed deposit
    /// root against the bls-pairing-side message-hash byte (acting as
    /// the block-header `eth1_data.deposit_root` placeholder).
    fn d0_deposit_root_to_block_eth1_data() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "validator_deposit_root_to_block_eth1_data_v1".into(),
            a_layer_index: 0,
            a_columns: vec![DEP_COL_DEPOSIT_ROOT_OFFSET],
            // Gate on IS_TOP (single row, the merkle-root row) so the
            // per-tuple A multiset has cardinality 1 — matching the
            // single B row. Gating on IS_REAL would project 32 copies
            // (one per merkle level) versus the pair AIR's single row
            // and witness-building would abort.
            a_selector_column: Some(DEP_COL_IS_TOP),
            b_layer_index: 1,
            b_columns: vec![PAIR_COL_MSG_HASH_OFFSET],
            b_selector_column: Some(PAIR_COL_IS_REAL),
        }
    }

    /// D1: deposit_tree_air `COL_SIG_BYTE0_MIRROR` ↔ bls_pairing_air
    /// `SIG_BYTES[0]`. Task #309: previously projected
    /// `DEP_COL_SIBLING_OFFSET` and host-side overrode SIBLING[0]
    /// (and the row-local-bound RIGHT[0]) to carry
    /// `deposit.signature[0]`. The clobber violated the AIR's
    /// `right_selection` body and was patched twice over. We now
    /// commit the signature first byte in a dedicated mirror column
    /// populated algebraically by the witness builder (see
    /// [`DepositInclusionWitness::sig_byte0`]). No row-local
    /// constraint binds the mirror; cross-AIR LogUp closure equality
    /// against the pair AIR's `SIG_BYTES[0]` provides the soundness.
    /// Same scaffolding pattern as
    /// [`crate::sync_committee_aggregate_composer_air`]'s
    /// `COL_AGG_PK_X_BYTE0_MIRROR`.
    fn d1_deposit_signature_byte_to_pairing_sig() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "validator_deposit_signature_byte_to_pairing_v1".into(),
            a_layer_index: 0,
            a_columns: vec![DEP_COL_SIG_BYTE0_MIRROR],
            // Gate on IS_TOP so A and B have matching multiset
            // cardinality 1. Mirror column carries the same
            // `deposit.signature[0]` byte on every active row; gating
            // on the single top row collapses the A multiset to one
            // copy, matching the pair AIR's single-row B side.
            a_selector_column: Some(DEP_COL_IS_TOP),
            b_layer_index: 1,
            b_columns: vec![PAIR_COL_SIG_BYTES_OFFSET],
            b_selector_column: Some(PAIR_COL_IS_REAL),
        }
    }

    /// D2: deposit_tree_air `DEPOSIT_ROOT[0]` ↔ sha256_extract
    /// `COL_MIRROR_BYTE0`. Task #309: previously targeted
    /// `SHA_COL_OUTPUT_BYTE_OFFSET`, but the natural
    /// `sha256_pair(leaf, leaf)` doesn't yield `deposit_root[0]` as
    /// its first byte, so the patch hand-overwrote `OUTPUT_BYTE[0]`
    /// (clobbering an algebraically committed cell). We now retarget
    /// to a dedicated mirror column populated by the witness builder
    /// via [`Sha256ExtractWitness::with_mirror_byte0`]; the honest
    /// path sets it to `deposit_root[0]` so the cross-AIR LogUp
    /// closure matches. No row-local constraint binds the mirror;
    /// soundness flows from the closure equality.
    fn d2_leaf_hash_to_sha256_output() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "validator_deposit_leaf_to_sha256_output_v1".into(),
            a_layer_index: 0,
            // Project DEPOSIT_ROOT[0] (constant across all merkle
            // rows by the deposit_root_constancy shifted body) gated
            // on IS_TOP for cardinality-1 multiset alignment.
            a_columns: vec![DEP_COL_DEPOSIT_ROOT_OFFSET],
            a_selector_column: Some(DEP_COL_IS_TOP),
            b_layer_index: 2,
            b_columns: vec![SHA_COL_MIRROR_BYTE0],
            b_selector_column: Some(SHA_COL_IS_REAL),
        }
    }

    // ─── Witness builders ─────────────────────────────────────────────

    /// Build a deposit with a real BLS signature over the canonical
    /// beacon-chain deposit `signing_root` derived from
    /// `hash_tree_root(deposit_message) || domain`. Returns the deposit,
    /// the signing root, and the host-derived deposit-tree root (for the
    /// all-zero-subtree siblings inclusion at `DEPOSIT_INDEX = 0`).
    fn build_signed_deposit() -> (DepositData, [u8; 32], [u8; 32]) {
        let sk = crate::bls_sig::SecretKey::from_u8_seed(DEPOSIT_SK_SEED);
        let pk = sk.public_key();
        let withdrawal_credentials = {
            let mut wc = [0u8; 32];
            wc[0] = 0x01; // execution-address (Eth1) prefix.
            wc
        };
        // Canonical beacon-chain deposit signing root:
        //   message_root = HTR(DepositMessage { pubkey, withdrawal_credentials, amount })
        //   domain       = DOMAIN_DEPOSIT(4) || fork_data_root[0..28]
        //   signing_root = sha256(message_root || domain)
        let message_root = deposit_message_root(
            &pk.0,
            &withdrawal_credentials,
            DEPOSIT_AMOUNT_GWEI,
        );
        let domain = compute_deposit_domain();
        let signing_root = compute_deposit_signing_root(&message_root, &domain);
        let sig = sk.sign(&signing_root, POP_DST);
        // Cross-check: the signature must verify under the canonical
        // BLS verify with the same DST.
        debug_assert!(
            crate::bls_sig::verify(&pk, &signing_root, &sig, POP_DST),
            "real BLS deposit signature must verify host-side",
        );
        let deposit = DepositData {
            pubkey: pk.0,
            withdrawal_credentials,
            amount: DEPOSIT_AMOUNT_GWEI,
            signature: sig.0,
        };
        // All-zero-subtree siblings for a single-leaf depth-32 tree at
        // index 0: yields a well-defined deposit root.
        let mut path = [[0u8; DEP_CHUNK_BYTES]; DEP_DEPTH];
        let mut z = [0u8; DEP_CHUNK_BYTES];
        for k in 0..DEP_DEPTH {
            path[k] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }
        let w = from_deposit_proof(&deposit, DEPOSIT_INDEX, &path);
        let deposit_root = w.inclusions[0].deposit_root;
        (deposit, signing_root, deposit_root)
    }

    /// Build the bls_pairing witness for the deposit. The witness's
    /// `msg_hash` is overridden host-side to equal the deposit_root so
    /// the D0 single-byte projection closes.
    fn build_pair_witness_for_deposit(
        deposit: &DepositData,
        deposit_root: [u8; 32],
    ) -> BlsPairingWitness {
        BlsPairingWitness::from_decoded(deposit.pubkey, deposit.signature, deposit_root)
            .expect("real deposit (pk, sig) must decode under BLS12-381")
    }

    /// Build a SHA-256 extract witness whose first invocation outputs a
    /// 32-byte hash and whose mirror column commits the deposit-tree
    /// root's first byte. Task #309: the mirror column replaces the
    /// scaffold-era host-side `OUTPUT_BYTE[0]` override so the
    /// cross-AIR LogUp D2 closure equality holds without clobbering
    /// an algebraically committed cell. The exact preimage doesn't
    /// matter for this descriptor — only the mirror byte does.
    fn build_sha_witness_for_leaf(leaf: [u8; 32], deposit_root_byte0: u8) -> Sha256ExtractWitness {
        Sha256ExtractWitness::from_pair_inputs(&[(leaf, leaf)])
            .with_mirror_byte0(vec![deposit_root_byte0])
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check across all three descriptors plus a
    /// non-prove sanity check on the 3-trace shape that `joint_prove`
    /// receives. Does NOT run `joint_prove`, so it stays well under
    /// 120s and runs in CI.
    #[test]
    fn descriptor_consistency_validator_deposit_three_air() {
        let d0 = d0_deposit_root_to_block_eth1_data();
        let d1 = d1_deposit_signature_byte_to_pairing_sig();
        let d2 = d2_leaf_hash_to_sha256_output();

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

        // Layer indices: deposit_tree=0, bls_pairing=1, sha256_extract=2.
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 1);
        assert_eq!(d2.a_layer_index, 0);
        assert_eq!(d2.b_layer_index, 2);

        // Labels are unique.
        let labels = [d0.label.as_str(), d1.label.as_str(), d2.label.as_str()];
        let mut sorted = labels.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "descriptor labels must be unique");

        // Build the per-AIR witnesses + traces at BLS48-581 and confirm
        // the 3-trace orchestrator inputs are constructible. Column
        // bounds for each descriptor's a_columns / b_columns are
        // checked against the per-AIR NUM_COLUMNS.
        let curve = CurveType::Bls48581;
        let (deposit, _signing_root, deposit_root) = build_signed_deposit();
        let dep_w = from_deposit_proof(
            &deposit,
            DEPOSIT_INDEX,
            &{
                let mut path = [[0u8; DEP_CHUNK_BYTES]; DEP_DEPTH];
                let mut z = [0u8; DEP_CHUNK_BYTES];
                for k in 0..DEP_DEPTH {
                    path[k] = z;
                    z = crate::sha256::sha256_pair(&z, &z);
                }
                path
            },
        );
        let pair_w = build_pair_witness_for_deposit(&deposit, deposit_root);
        let leaf = dep_w.inclusions[0].leaf;
        let sha_w = build_sha_witness_for_leaf(leaf, deposit_root[0]);

        let trace_0 = build_dep_trace(&dep_w, curve);
        let trace_1 = build_pair_trace(&pair_w, curve);
        let trace_2 = build_sha_trace(&sha_w, curve);

        assert_eq!(trace_0.columns.len(), DEP_NUM_COLUMNS);
        assert_eq!(trace_1.columns.len(), PAIR_NUM_COLUMNS);
        assert_eq!(trace_2.columns.len(), SHA_NUM_COLUMNS);
        assert!(trace_0.num_rows >= DEP_DEPTH);
        assert!(trace_1.num_rows >= 1);
        assert!(trace_2.num_rows >= 1);

        let cs_0 = DepositTreeConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BlsPairingConstraintSystem::new(trace_1.num_rows);
        let cs_2 = Sha256ExtractConstraintSystem::new(trace_2.num_rows);

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
        let num_cols_per_layer = [DEP_NUM_COLUMNS, PAIR_NUM_COLUMNS, SHA_NUM_COLUMNS];
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

        // Sanity: the deposit-tree leaf row's CURRENT_HASH[0] equals
        // the deposit's `hash_tree_root` first byte, which is what
        // descriptor D2 anchors on the A side.
        let leaf_byte0 = leaf[0];
        let dep_current_hash_byte0 = trace_0.columns[DEP_COL_CURRENT_HASH_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            dep_current_hash_byte0 as u8, leaf_byte0,
            "deposit_tree row 0 CURRENT_HASH[0] must equal leaf[0]",
        );
        // And the DEPOSIT_ROOT[0] cell equals deposit_root[0] (what
        // D0 anchors on the A side).
        let dep_root_byte0 = trace_0.columns[DEP_COL_DEPOSIT_ROOT_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            dep_root_byte0 as u8, deposit_root[0],
            "deposit_tree row 0 DEPOSIT_ROOT[0] must equal deposit_root[0]",
        );

        // Pair-side MSG_HASH[0] cell is initially the host-supplied
        // msg_hash[0] = deposit_root[0]; SIG_BYTES[0] is the deposit
        // signature's first byte.
        let pair_msg_byte0 = trace_1.columns[PAIR_COL_MSG_HASH_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            pair_msg_byte0 as u8, deposit_root[0],
            "pair MSG_HASH[0] aligned with deposit_root[0]",
        );
        let pair_sig_byte0 = trace_1.columns[PAIR_COL_SIG_BYTES_OFFSET]
            .evaluations[0]
            .to_u64();
        assert_eq!(
            pair_sig_byte0 as u8, deposit.signature[0],
            "pair SIG_BYTES[0] equals deposit.signature[0]",
        );
    }

    /// Fast: the real BLS signature produced by `build_signed_deposit`
    /// verifies under the canonical beacon-chain `signing_root` derived
    /// from `hash_tree_root(deposit_message) || domain`. Pins the
    /// host-side signing contract Phase 0 implements.
    #[test]
    fn real_bls_deposit_signature_verifies() {
        let (deposit, signing_root, _deposit_root) = build_signed_deposit();
        // Recompute the signing root from the deposit's public fields
        // and confirm it matches what `build_signed_deposit` signed.
        let recomputed_message_root = deposit_message_root(
            &deposit.pubkey,
            &deposit.withdrawal_credentials,
            deposit.amount,
        );
        let domain = compute_deposit_domain();
        let recomputed_signing_root =
            compute_deposit_signing_root(&recomputed_message_root, &domain);
        assert_eq!(
            recomputed_signing_root, signing_root,
            "signing_root must be reproducible from deposit_message + domain",
        );
        // Domain prefix: first 4 bytes are DOMAIN_DEPOSIT (0x03000000).
        assert_eq!(&domain[0..4], &DOMAIN_DEPOSIT);
        // The real BLS signature must verify under POP_DST.
        let pk = crate::bls_sig::PublicKey(deposit.pubkey);
        let sig = crate::bls_sig::Signature(deposit.signature);
        assert!(
            crate::bls_sig::verify(&pk, &signing_root, &sig, POP_DST),
            "real BLS deposit signature must verify against derived signing_root",
        );
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 3-AIR `joint_prove` + `joint_verify` round-trip with
    /// 3 cross-AIR LogUp descriptors. Marked `#[ignore]` because each
    /// per-AIR `prove_with_scheme` plus 3 inner linkage SNARKs adds up
    /// to many minutes under BLS48-581 (the deposit-tree AIR alone has
    /// 229 columns × 32 rows + byte-range LogUp inflating the domain
    /// to 256).
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify under BLS48-581"]
    fn honest_validator_deposit_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (deposit, _signing_root, deposit_root) = build_signed_deposit();
        let mut path = [[0u8; DEP_CHUNK_BYTES]; DEP_DEPTH];
        let mut z = [0u8; DEP_CHUNK_BYTES];
        for k in 0..DEP_DEPTH {
            path[k] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }
        let dep_w = from_deposit_proof(&deposit, DEPOSIT_INDEX, &path);
        let pair_w = build_pair_witness_for_deposit(&deposit, deposit_root);
        let leaf = dep_w.inclusions[0].leaf;
        let sha_w = build_sha_witness_for_leaf(leaf, deposit_root[0]);

        // Task #309: the patch_traces_for_single_byte_alignment hack is
        // gone — D1 now targets `DEP_COL_SIG_BYTE0_MIRROR` (witness-
        // populated with deposit.signature[0]), D2 now targets
        // `SHA_COL_MIRROR_BYTE0` (witness-populated with deposit_root[0])
        // via `with_mirror_byte0`, and D0 closes naturally because both
        // `DEPOSIT_ROOT[0]` and `PAIR_MSG_HASH[0]` equal deposit_root[0]
        // on the gated rows.
        let trace_0 = build_dep_trace(&dep_w, curve);
        let trace_1 = build_pair_trace(&pair_w, curve);
        let trace_2 = build_sha_trace(&sha_w, curve);

        let cs_0 = DepositTreeConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BlsPairingConstraintSystem::new(trace_1.num_rows);
        let cs_2 = Sha256ExtractConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![
            d0_deposit_root_to_block_eth1_data(),
            d1_deposit_signature_byte_to_pairing_sig(),
            d2_leaf_hash_to_sha256_output(),
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
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 3-AIR joint_verify must accept",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tamper the second descriptor's `closure_a` — `joint_verify` must
    /// reject. Marked `#[ignore]` because the setup half is the same
    /// `joint_prove` call as the honest test.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_validator_deposit_joint_verify_true"]
    fn tampered_validator_deposit_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (deposit, _signing_root, deposit_root) = build_signed_deposit();
        let mut path = [[0u8; DEP_CHUNK_BYTES]; DEP_DEPTH];
        let mut z = [0u8; DEP_CHUNK_BYTES];
        for k in 0..DEP_DEPTH {
            path[k] = z;
            z = crate::sha256::sha256_pair(&z, &z);
        }
        let dep_w = from_deposit_proof(&deposit, DEPOSIT_INDEX, &path);
        let pair_w = build_pair_witness_for_deposit(&deposit, deposit_root);
        let leaf = dep_w.inclusions[0].leaf;
        let sha_w = build_sha_witness_for_leaf(leaf, deposit_root[0]);

        // Task #309: see `honest_validator_deposit_joint_verify_true`
        // for the mirror-column wiring that retired the patch hack.
        let trace_0 = build_dep_trace(&dep_w, curve);
        let trace_1 = build_pair_trace(&pair_w, curve);
        let trace_2 = build_sha_trace(&sha_w, curve);

        let cs_0 = DepositTreeConstraintSystem::new(trace_0.num_rows);
        let cs_1 = BlsPairingConstraintSystem::new(trace_1.num_rows);
        let cs_2 = Sha256ExtractConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![
            d0_deposit_root_to_block_eth1_data(),
            d1_deposit_signature_byte_to_pairing_sig(),
            d2_leaf_hash_to_sha256_output(),
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
