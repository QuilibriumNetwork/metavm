//! 3-AIR cross-AIR LogUp `joint_prove` / `joint_verify` integration
//! exercising the EIP-7702 set-code authorization chain:
//!
//!   layer 0 — [`crate::eip7702_delegation_air`] (125 cols, gated by
//!             [`crate::eip7702_delegation_air::COL_IS_REAL`]),
//!   layer 1 — [`crate::secp256k1_recovery::recovery_air`]
//!             (`RecoveryAir`, gated by
//!             [`crate::secp256k1_recovery::COL_IS_REAL`]),
//!   layer 2 — [`crate::tx_rlp_air`] (gated by
//!             [`crate::tx_rlp_air::COL_IS_REAL`]).
//!
//! ## Descriptors
//!
//! The current `joint_prove` API only wires single-column tuples on
//! each side, so we bind on the first byte / first column of each
//! algebraic target. (The full multi-byte / multi-column binding for
//! the EIP-7702 authorization-list lives in
//! [`crate::eip7702_delegation_air::make_eip7702_to_secp256k1_recovery_descriptor`]
//! and
//! [`crate::eip7702_delegation_air::make_eip7702_to_tx_rlp_descriptor`],
//! which together cover 20 + 2 columns; those need a multi-column
//! LogUp encoding extension to join-prove and are exercised
//! algebraically only at the per-AIR layer in this round.)
//!
//! - **D0** (`authority_address[0]` ↔ `recovered_addr[0]`): wires the
//!   first byte of the EIP-7702 authority address (layer 0,
//!   `COL_AUTHORITY_ADDR_OFFSET`) to the first byte of the
//!   secp256k1-recovered Ethereum address (layer 1,
//!   `COL_RECOVERED_ADDR_OFFSET`).
//!
//! - **D1** (`chain_id` ↔ `chain_id`): wires the EIP-7702
//!   authorization's `chain_id` (layer 0, `COL_CHAIN_ID`) to the
//!   surrounding tx's `chain_id` (layer 2,
//!   [`crate::tx_rlp_air::COL_CHAIN_ID`]).
//!
//! ## Witness alignment
//!
//! We deterministically generate a secp256k1 test key, sign the
//! EIP-7702 auth-tuple keccak `MAGIC || rlp([chain_id, delegate,
//! nonce])`, and thread:
//!
//!   - the recovered authority address into the delegation AIR (so
//!     `auth_addr[0] == recovered_addr[0]`),
//!   - a common `chain_id` (= 1) into both the delegation AIR and a
//!     companion EIP-1559 `LegacyTx`-style tx into the `tx_rlp_air`.
//!
//! ## Soundness scope
//!
//! Like `integration_joint_prove_three_air.rs`, this test wires
//! `joint_prove` + `joint_verify` end-to-end. It does not subsume the
//! per-AIR full-row checks (`recovered_addr == keccak256(X || Y)[12..]`
//! for layer 1 is enforced by `RecoveryAir`; full RLP layout for
//! layer 2 is enforced by `TxRlpConstraintSystem`), and the
//! single-column tuple binding is a smoke test of the orchestrator on
//! a heterogeneous 3-AIR chain — the multi-column closure for the
//! full 20-byte address ↔ recovered-address binding is reserved for a
//! follow-up.

#[cfg(test)]
mod tests {
    use crate::cross_air_logup::{joint_prove, joint_verify, CrossAirLogUpDescriptor};
    use crate::field::{CurveType, Scalar};
    use crate::scheme::bls48581_scheme::Bls48581Scheme;
    use crate::scheme::CommitmentScheme;

    use crate::eip7702_delegation_air::{
        auth_message_preimage, build_trace_polynomials as build_eip_trace,
        from_authorization as build_eip_witness, Eip7702DelegationConstraintSystem,
        ADDR_LEN as EIP_ADDR_LEN, COL_AUTHORITY_ADDR_OFFSET as EIP_COL_AUTHORITY_ADDR_OFFSET,
        COL_CHAIN_ID as EIP_COL_CHAIN_ID, COL_IS_REAL as EIP_COL_IS_REAL, SIG_LEN as EIP_SIG_LEN,
    };
    use crate::keccak::keccak256;
    use crate::secp256k1_recovery::{
        recovery_air::build_trace_polynomials as build_rec_trace, RecoveryAirConstraintSystem,
        RecoveryAirWitness, COL_IS_REAL as REC_COL_IS_REAL,
        COL_RECOVERED_ADDR_OFFSET as REC_COL_RECOVERED_ADDR_OFFSET,
    };
    use crate::transaction::Eip1559Tx;
    use crate::tx_rlp_air::{
        build_trace_polynomials as build_txr_trace, TxRlpConstraintSystem, TxRlpRow, TxRlpWitness,
        COL_CHAIN_ID as TXR_COL_CHAIN_ID, COL_IS_REAL as TXR_COL_IS_REAL,
    };

    use k256::ecdsa::{
        signature::hazmat::PrehashSigner, RecoveryId, Signature, SigningKey,
    };

    // ─── Test fixtures ────────────────────────────────────────────────

    const HONEST_CHAIN_ID: u64 = 1;
    const HONEST_NONCE: u64 = 7;

    /// Deterministic test signing key derived from a single u8 seed
    /// (NOT for production).
    fn test_signing_key(seed: u8) -> SigningKey {
        let mut sk_bytes = [0u8; 32];
        for i in 0..32 {
            // Ensure all-nonzero and well below the curve order.
            sk_bytes[i] = (i as u8).wrapping_add(seed).wrapping_add(1);
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

    /// Sign `msg_hash` with `sk` and return `(parity, r, s)`.
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

    // ─── Descriptor builders (single-column tuples) ───────────────────

    /// D0: `eip7702_delegation_air.COL_AUTHORITY_ADDR_OFFSET` (first
    /// byte of authority address, layer 0, gated by `COL_IS_REAL`)
    /// ↔ `secp256k1_recovery.recovery_air.COL_RECOVERED_ADDR_OFFSET`
    /// (first byte of recovered address, layer 1, gated by
    /// `COL_IS_REAL`).
    ///
    /// The full 20-byte binding is the multi-column descriptor in
    /// [`crate::eip7702_delegation_air::make_eip7702_to_secp256k1_recovery_descriptor`];
    /// here we restrict to the first byte to match the current
    /// `joint_prove` single-column API.
    fn authority_address_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "integration_eip7702_authority_address_byte0_v1".into(),
            a_layer_index: 0,
            a_columns: vec![EIP_COL_AUTHORITY_ADDR_OFFSET],
            a_selector_column: Some(EIP_COL_IS_REAL),
            b_layer_index: 1,
            b_columns: vec![REC_COL_RECOVERED_ADDR_OFFSET],
            b_selector_column: Some(REC_COL_IS_REAL),
        }
    }

    /// D1: `eip7702_delegation_air.COL_CHAIN_ID` (layer 0, gated by
    /// `COL_IS_REAL`) ↔ `tx_rlp_air.COL_CHAIN_ID` (layer 2, gated by
    /// `COL_IS_REAL`).
    fn chain_id_descriptor() -> CrossAirLogUpDescriptor {
        CrossAirLogUpDescriptor {
            label: "integration_eip7702_chain_id_v1".into(),
            a_layer_index: 0,
            a_columns: vec![EIP_COL_CHAIN_ID],
            a_selector_column: Some(EIP_COL_IS_REAL),
            b_layer_index: 2,
            b_columns: vec![TXR_COL_CHAIN_ID],
            b_selector_column: Some(TXR_COL_IS_REAL),
        }
    }

    // ─── Honest witness construction ──────────────────────────────────

    /// Build the three honest per-AIR witnesses, returning each one
    /// together with the shared (chain_id, authority_address) tuple
    /// they were keyed on. The authority address is the secp256k1
    /// recovery of the EIP-7702 auth tuple's keccak signed by a
    /// deterministic test key, so all three traces share matching
    /// values on the linked columns.
    fn build_honest_traces() -> (
        crate::eip7702_delegation_air::Eip7702DelegationWitness,
        RecoveryAirWitness,
        TxRlpWitness,
    ) {
        let sk = test_signing_key(0x13);
        let authority = test_address(&sk);

        let delegated_to = {
            let mut d = [0u8; EIP_ADDR_LEN];
            d[19] = 0xee;
            d
        };

        // Build the EIP-7702 auth-tuple preimage:
        // MAGIC || rlp([chain_id, delegated_to, nonce])
        let preimage = auth_message_preimage(HONEST_CHAIN_ID, &delegated_to, HONEST_NONCE);
        let msg_hash = keccak256(&preimage);

        let (parity, r, s) = raw_sign(&sk, &msg_hash);
        let mut signature = [0u8; EIP_SIG_LEN];
        signature[0..32].copy_from_slice(&r);
        signature[32..64].copy_from_slice(&s);
        signature[64] = parity; // EIP-7702 stores raw y_parity (0/1).

        // Layer 0 — EIP-7702 delegation witness.
        let eip_w = build_eip_witness(
            authority,
            HONEST_CHAIN_ID,
            HONEST_NONCE,
            delegated_to,
            signature,
            true,
        );

        // Layer 1 — secp256k1 recovery witness (using the raw v = parity,
        // EIP-7702 uses y_parity directly).
        let rec_w =
            RecoveryAirWitness::from_signature(msg_hash, parity as u64, r, s, None)
                .expect("recovery from_signature must succeed for honest sig");

        // Layer 2 — companion EIP-1559 tx whose chain_id matches.
        let tx = Eip1559Tx {
            chain_id: HONEST_CHAIN_ID,
            nonce: HONEST_NONCE,
            max_priority_fee_per_gas: [0u8; 32],
            max_fee_per_gas: [0u8; 32],
            gas_limit: 21_000,
            to: Some(delegated_to),
            value: [0u8; 32],
            data: Vec::new(),
            access_list_rlp: vec![0xc0],
            y_parity: 0,
            r: [0u8; 32],
            s: [0u8; 32],
        };
        let txr_row = TxRlpRow::from_eip1559_tx(&tx).expect("eip1559 tx_rlp row");
        let txr_w = TxRlpWitness { rows: vec![txr_row] };

        (eip_w, rec_w, txr_w)
    }

    // ─── Fast static check (un-ignored) ───────────────────────────────

    /// Static well-formedness check across both descriptors plus a
    /// non-prove sanity check on the 3-trace shape that `joint_prove`
    /// receives. Does NOT run `joint_prove`, so it stays well under
    /// 120s and runs in CI.
    #[test]
    fn descriptor_consistency() {
        let d0 = authority_address_descriptor();
        let d1 = chain_id_descriptor();

        // Single-column tuples (current joint_prove API limit).
        assert_eq!(d0.a_columns.len(), 1);
        assert_eq!(d0.b_columns.len(), 1);
        assert_eq!(d1.a_columns.len(), 1);
        assert_eq!(d1.b_columns.len(), 1);

        // Layer 0 is the shared "middle" trace participating in both
        // descriptors (D0: 0↔1, D1: 0↔2).
        assert_eq!(d0.a_layer_index, 0);
        assert_eq!(d0.b_layer_index, 1);
        assert_eq!(d1.a_layer_index, 0);
        assert_eq!(d1.b_layer_index, 2);
        assert_ne!(d0.a_layer_index, d0.b_layer_index);
        assert_ne!(d1.a_layer_index, d1.b_layer_index);

        // Selectors must be wired on both sides of both descriptors.
        assert_eq!(d0.a_selector_column, Some(EIP_COL_IS_REAL));
        assert_eq!(d0.b_selector_column, Some(REC_COL_IS_REAL));
        assert_eq!(d1.a_selector_column, Some(EIP_COL_IS_REAL));
        assert_eq!(d1.b_selector_column, Some(TXR_COL_IS_REAL));

        // Column indices reference the AIR-published `COL_*` constants.
        assert_eq!(d0.a_columns[0], EIP_COL_AUTHORITY_ADDR_OFFSET);
        assert_eq!(d0.b_columns[0], REC_COL_RECOVERED_ADDR_OFFSET);
        assert_eq!(d1.a_columns[0], EIP_COL_CHAIN_ID);
        assert_eq!(d1.b_columns[0], TXR_COL_CHAIN_ID);

        // Honest witness construction: confirm the recovered address
        // first byte equals the EIP-7702 authority address first byte
        // (so D0's per-row closure will match algebraically), and the
        // chain_id matches across layer 0 and layer 2.
        let (eip_w, rec_w, txr_w) = build_honest_traces();
        assert_eq!(
            eip_w.rows[0].authority_address[0], rec_w.rows[0].recovered_addr[0],
            "honest auth_addr[0] must equal recovered_addr[0]",
        );
        assert_eq!(eip_w.rows[0].chain_id, HONEST_CHAIN_ID);
        assert_eq!(txr_w.rows[0].chain_id, HONEST_CHAIN_ID);

        // Build per-AIR traces + constraint systems and verify the
        // orchestrator's input shape can be assembled.
        let curve = CurveType::Bls48581;
        let trace_0 = build_eip_trace(&eip_w, curve);
        let trace_1 = build_rec_trace(&rec_w, curve);
        let trace_2 = build_txr_trace(&txr_w, curve);

        let cs_0 = Eip7702DelegationConstraintSystem::new(trace_0.num_rows);
        let cs_1 = RecoveryAirConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxRlpConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![d0.clone(), d1.clone()];

        assert_eq!(traces.len(), 3, "3-AIR joint_prove input must have 3 traces");
        assert_eq!(linkages.len(), 2, "must wire exactly 2 descriptors");

        for link in &linkages {
            assert!(link.a_layer_index < traces.len());
            assert!(link.b_layer_index < traces.len());
        }
    }

    // ─── Slow honest round-trip (ignored) ─────────────────────────────

    /// Full honest 3-AIR `joint_prove` + `joint_verify` round-trip
    /// with 2 cross-AIR LogUp descriptors. Marked `#[ignore]`
    /// because the per-AIR `prove_with_scheme` calls (each AIR
    /// declares 8-bit byte range lookups → LogUp inflates each
    /// per-AIR domain to 256) plus the 2 per-linkage prove calls
    /// blow past 120s under BLS48-581.
    #[test]
    #[ignore = "slow: 3-AIR joint_prove + joint_verify under BLS48-581 (5 inner prove calls)"]
    fn honest_eip7702_joint_verify_true() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (eip_w, rec_w, txr_w) = build_honest_traces();

        let trace_0 = build_eip_trace(&eip_w, curve);
        let trace_1 = build_rec_trace(&rec_w, curve);
        let trace_2 = build_txr_trace(&txr_w, curve);

        let cs_0 = Eip7702DelegationConstraintSystem::new(trace_0.num_rows);
        let cs_1 = RecoveryAirConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxRlpConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![authority_address_descriptor(), chain_id_descriptor()];

        let (proofs, ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest EIP-7702 3-AIR joint_prove must succeed");

        assert_eq!(proofs.len(), 3, "expected one ExecutionProof per AIR");
        assert_eq!(
            ext.linkage_proofs.len(),
            2,
            "expected one CrossAirLogUpProof per descriptor",
        );

        // Honest closure-equality across both descriptors.
        for (i, lp) in ext.linkage_proofs.iter().enumerate() {
            assert_eq!(
                lp.closure_a, lp.closure_b,
                "honest joint_prove must produce matching closures on descriptor {}",
                i,
            );
        }

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        assert!(
            joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "honest 3-AIR joint_verify must accept matching tuples on both descriptors",
        );
    }

    // ─── Slow tampered round-trip (ignored) ───────────────────────────

    /// Tampered closure: corrupt `closure_a` on descriptor D1 (the
    /// chain_id linkage between layer 0 and layer 2). The verifier's
    /// `closure_a == closure_b` scalar equality must reject.
    #[test]
    #[ignore = "slow: depends on the joint_prove setup of honest_eip7702_joint_verify_true"]
    fn tampered_eip7702_joint_verify_false() {
        let curve = CurveType::Bls48581;
        let scheme = Bls48581Scheme::new();
        scheme.init();

        let (eip_w, rec_w, txr_w) = build_honest_traces();

        let trace_0 = build_eip_trace(&eip_w, curve);
        let trace_1 = build_rec_trace(&rec_w, curve);
        let trace_2 = build_txr_trace(&txr_w, curve);

        let cs_0 = Eip7702DelegationConstraintSystem::new(trace_0.num_rows);
        let cs_1 = RecoveryAirConstraintSystem::new(trace_1.num_rows);
        let cs_2 = TxRlpConstraintSystem::new(trace_2.num_rows);

        let traces: Vec<(
            &crate::trace::TracePolynomials,
            &dyn crate::vm_constraints::VmConstraintSystem,
        )> = vec![(&trace_0, &cs_0), (&trace_1, &cs_1), (&trace_2, &cs_2)];
        let linkages = vec![authority_address_descriptor(), chain_id_descriptor()];

        let (proofs, mut ext) = joint_prove(&traces, &linkages, &scheme)
            .expect("honest EIP-7702 3-AIR joint_prove must succeed");

        // Tamper with the SECOND descriptor's closure_a (chain_id
        // linkage).
        let one_bytes = Scalar::one(curve).to_bytes();
        ext.linkage_proofs[1].closure_a = one_bytes;

        let cs_refs: Vec<&dyn crate::vm_constraints::VmConstraintSystem> =
            vec![&cs_0, &cs_1, &cs_2];
        assert!(
            !joint_verify(&proofs, &cs_refs, &linkages, &ext, &scheme, curve),
            "joint_verify must reject mismatching closures on the chain_id descriptor",
        );
    }
}
