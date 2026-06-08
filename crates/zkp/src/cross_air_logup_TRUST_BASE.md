# Cross-AIR LogUp Descriptor Library — Trust Base

**Status**: normative. Read this BEFORE auditing any cross-AIR linkage proof
or adding a new `make_*_linkage_descriptor` constructor.

This document is the security contract for the
`CrossAirLogUpDescriptor` ecosystem defined in
[`cross_air_logup.rs`](./cross_air_logup.rs). It exists because the
cryptographic soundness of every joint proof in this codebase
*assumes* the descriptors are semantically correct. The proof system
does **not** detect a descriptor that mis-maps columns; it only proves
the multiset equality that the descriptor *literally describes*. A
buggy descriptor produces a fully verifying proof of a statement
nobody intended to prove.

Treat descriptor construction the way you would treat a trusted setup.

---

## 1. What a descriptor declares

A `CrossAirLogUpDescriptor` (see `cross_air_logup.rs:847`) is a
five-tuple per side:

```text
CrossAirLogUpDescriptor {
    label: String,                          // transcript domain separator
    a_layer_index: usize,                   // source AIR's slot in LayerChainProof
    a_columns: Vec<usize>,                  // source column indices (the "tuple shape")
    a_selector_column: Option<usize>,       // gating column on A
    b_layer_index: usize,                   // target AIR's slot in LayerChainProof
    b_columns: Vec<usize>,                  // target column indices (matching tuple shape)
    b_selector_column: Option<usize>,       // gating column on B
}
```

The descriptor is consumed by `joint_prove` /
`joint_verify` (`cross_air_logup.rs:1030`, `:1583`) which:

1. Hashes `label`, all `a_columns`, all `b_columns`, the layer indices,
   and both selectors into the joint transcript to derive β (the
   tuple-folding challenge) and γ (the LogUp challenge).
2. Computes per-row tuples
   `tuple_A(r) = Σ_k β^k · trace_A[a_columns[k]](r)`
   and similarly for B.
3. Builds the LogUp witness `(f_A, m_B, f_B, h_A, h_B)`, opens
   `h_A(ω^{n-1})` and `h_B(ω^{n-1})` against per-linkage SNARK
   commitments, and enforces `closure_A == closure_B` algebraically.

The result is a proof that the multiset
`{ tuple_A(r) : active_A(r) }` equals
`{ tuple_B(r) : active_B(r) }` (with multiplicity `m_B`).

---

## 2. What the proof DOES guarantee

For a verifying `CrossAirLogUpProof` over descriptor `D` and two
verifying single-AIR proofs `π_A`, `π_B`:

> **For the multiset of row vectors
> `(trace_A[D.a_columns](r))` taken over rows where
> `trace_A[D.a_selector_column](r) ≠ 0`, there exists a multiplicity
> assignment to rows of `B` such that this multiset equals the
> corresponding multiset on side B.**

This is enforced cryptographically under standard
KZG / Schwartz–Zippel assumptions, provided the joint γ is unknown to
the prover before main commits land in the transcript (this is what
the phase-split prover in `joint_prove` exists to ensure).

## 3. What the proof DOES NOT guarantee

The LogUp argument is **purely a multiset equality over scalar
tuples**. It is blind to:

- Whether `a_columns` points at the columns the *user* intended.
- Whether the columns' encodings on A and B are semantically aligned
  (e.g. byte-order, endianness, limb decomposition, padding).
- Whether the *selectors* fire on the rows that matter.
- Whether the AIRs themselves correctly bind those columns to the
  semantic objects the proof system claims to talk about.

A typo of `0..32` instead of `1..33`, a swap of `b_columns` and
`a_columns`, a selector that is always `0`, or two AIRs that store
"the SHA-256 input bytes" in two different orderings will all produce
**a verifying proof of the wrong statement**. The verifier has no
oracle for "what statement should this be?" — it has only the
descriptor.

---

## 4. The trust base: what descriptor authors MUST get right

A descriptor is **correct** iff all of the following hold. Reviewers:
check every item, every time.

### 4.1 Column-index correctness

- `a_columns[k]` indexes the column on AIR A that carries the *k*-th
  limb / byte / scalar of the semantic value being linked.
- `b_columns[k]` indexes the column on AIR B that carries the same
  *k*-th limb / byte / scalar, **in the same on-wire layout** (same
  endianness, same packing, same range, same padding convention).
- `a_columns.len() == b_columns.len()`. Mis-matched lengths are
  rejected by `joint_prove`, but matching the count alone is not
  sufficient — the *meaning* must align element-wise.

### 4.2 Selector correctness

- `a_selector_column` is `1` on every row whose tuple should appear in
  the source multiset and `0` everywhere else. In particular, it must
  be `0` on padding rows, on rows that "share" a multi-row gadget but
  don't publish a tuple, and on rows belonging to disjoint operations.
- `b_selector_column` is `1` on every row whose tuple should appear in
  the target multiset.
- A `None` selector means "every row". This is almost never right;
  use it only when the AIR's row schema guarantees every row carries a
  semantically valid tuple.
- A selector that is always `0` makes the descriptor trivially
  satisfiable (`closure = 0` on both sides). This is a critical
  silent-acceptance failure mode.

### 4.3 Label uniqueness and stability

- `label` is absorbed into the joint Fiat–Shamir transcript and
  domain-separates β and γ. Two distinct descriptors with the same
  label collapse into a single transcript binding — the joint γ no
  longer uniquely characterises either, and adaptive attacks become
  possible.
- Labels must be **globally unique** across the entire descriptor
  library. Convention: `"<gadget>_<operation>_v<n>"`, e.g.
  `"ssz_sha256_pair_v1"`, `"evm_mstore_byte_decomp_v1"`. Bump the
  version on any breaking change to column layout or selector
  semantics.
- Labels must be **stable**: changing a label silently invalidates
  every previously generated proof against the old descriptor. Treat
  label changes as soundness-sensitive migrations.

### 4.4 Layer index correctness

- `a_layer_index` / `b_layer_index` must match the actual position of
  the AIRs A and B inside the `LayerChainProof` envelope at prove and
  verify time. A mismatched index opens commits from a different AIR,
  which usually causes verification to fail loudly, but in pathological
  cases (two AIRs sharing column layout by coincidence) could silently
  prove a meaningless statement.

---

## 5. Failure modes if the descriptor is wrong

| Failure mode | Symptom | Severity |
|---|---|---|
| Wrong column index on one side | Verifies; proves multiset equality of garbage values | Critical, silent |
| Swapped `a_columns` and `b_columns` | Verifies if multisets coincidentally equal | Critical, silent on toy inputs |
| Selector always 0 | Verifies with `closure = 0 = 0`; proves nothing | Critical, silent |
| Selector fires on padding rows | Verifies, but padding-row tuples leak into multiset | High, silent |
| Mismatched endian / limb decomposition | Verifies as multiset equality over re-interpreted scalars | Critical, silent |
| Duplicate label | Two linkages share β/γ; weakens domain separation | Medium, silent under honest prover |
| Wrong layer index | Usually loud verify-time failure; rarely silent | Medium |
| `a_columns.len() != b_columns.len()` | Loud `joint_prove` panic / error | Low (caught) |

The **silent** rows are why descriptor authoring is part of the trust
base: there is no test-suite-shaped oracle that will catch them.

---

## 6. Recommendations

1. **Treat the descriptor library as a trusted module.** Every
   `pub fn make_*_linkage_descriptor` (and every analog in the
   `*_descriptors.rs` files) needs the same review rigour as a
   constraint-system constructor.
2. **Reviewers required.** No new descriptor should land without a
   second reviewer eyeballing the column-index ↔ semantic-value
   mapping on both the A side and the B side. Where possible, reference
   the column-layout `COL_*` constants by name instead of by literal
   index.
3. **Pin column layouts.** Every AIR that exposes columns to a
   descriptor SHOULD have a `// EXPOSED TO CROSS-AIR LINKAGE` comment
   block listing the descriptor labels that reference it and the
   column indices used. Renumbering a column requires sweeping every
   descriptor that touches it.
4. **Tampering tests are mandatory.** Each non-trivial descriptor
   should ship with at least one negative test that mutates a tuple
   value on one side and asserts the closure mismatch is detected.
   Without a negative test, a typo'd descriptor that happens to
   coincide on golden inputs is indistinguishable from a correct one.
5. **Beware of `None` selectors.** Default to *some* selector and
   only fall back to `None` when the row schema mathematically
   guarantees every row is meaningful. Document the reasoning.

---

## 7. Trust-base manifest: where descriptors live

These are the modules whose `pub fn make_*` constructors collectively
form the descriptor library. Anything verifying a multi-AIR proof in
this repo transitively depends on the semantic correctness of these
functions. As of this writing, **129 source files** define one or more
`pub fn make_*_(linkage_)?descriptor` constructors — the list below
covers the *organising* modules; individual gadget AIRs each contribute
their own `make_*_linkage_descriptor` next to the AIR definition.

### Centralised descriptor modules

These files exist purely to hold descriptor constructors and are the
highest-leverage review surface:

- `cross_layer_descriptors.rs` — block-level linkages
  (block_header ↔ payload, header ↔ tx-root, header ↔ receipts-root,
  etc.). Every full-Ethereum-block proof flows through these.
- `miller_fp_descriptors.rs` — BLS12-381 / BN254 Miller loop ↔ Fp
  arithmetic AIR linkages. Underlies every pairing-based proof
  (BLS verify, KZG point eval, sync-committee signature).
- `final_exp_descriptors.rs` — BLS12-381 final exponentiation ↔ Fp
  tower linkages. Pairs with `miller_fp_descriptors.rs` to close
  pairing soundness.
- `secp256k1_group_descriptors.rs` — secp256k1 group law ↔ Fp
  arithmetic linkages. Underlies tx-sender recovery (`ecrecover`),
  EIP-1559 / 4844 / 7702 signature validation.
- `mpt_inclusion_descriptors.rs` — MPT row-type linkages (leaf /
  extension / branch ↔ keccak). Underlies every state / storage /
  receipt / transaction inclusion proof.

### Distributed per-AIR descriptor functions (selected)

Most gadget AIRs publish their own descriptors as
`pub fn make_<gadget>_to_<consumer>_linkage_descriptor`. Notable
clusters:

- **EVM core ↔ gadget** linkages: `mstore_byte_air.rs`,
  `sha3_read_byte_air.rs`, `sha3_input_air.rs`, `storage_access_air.rs`,
  `address_keccak_air.rs`, `byte_memory_air.rs`,
  `evm_create_rlp_air.rs`, `evm_create2_input_air.rs`,
  `account_state_air.rs`.
- **MPT row ↔ keccak / RLP** linkages: `mpt_air.rs`,
  `mpt_rlp_air.rs`, `mpt_short_leaf_rlp_air.rs`,
  `mpt_six_nibble_leaf_rlp_air.rs`, `mpt_eight_nibble_leaf_rlp_air.rs`,
  `mpt_odd_leaf_rlp_air.rs`, `mpt_extension_rlp_air.rs`,
  `mpt_branch_05_rlp_air.rs`, `mpt_branch_15_rlp_air.rs`.
- **Beacon / consensus** linkages:
  `beacon_block_body_air.rs`, `beacon_block_body_pair_air.rs`,
  `beacon_block_header_pair_air.rs`, `bbh_root_consumer_air.rs`,
  `execution_payload_pair_air.rs`, `extra_data_pair_air.rs`,
  `logs_bloom_pair_air.rs`, `validator_htr_air.rs`,
  `validator_extract.rs`, `sync_committee_filter_air.rs`,
  `finality_constraints.rs`.
- **Hash / RLP extract** linkages: `keccak_extract.rs`,
  `keccak_extract_wide.rs`, `keccak_constraints.rs`,
  `sha256_extract.rs`, `sha256_constraints.rs`,
  `fixed_rlp_air.rs`, `block_header_air.rs`,
  `receipt_with_logs_chain.rs`.

To regenerate the canonical list:

```sh
grep -rln 'pub fn make_.*_descriptor\|pub fn make_.*_linkage_descriptor' \
     crates/zkp/src/ | sort
```

Any module on that list is part of the trust base.

---

## 8. Audit checklist (per new descriptor)

When reviewing a PR that adds or modifies a descriptor, confirm:

- [ ] `label` is globally unique and version-suffixed.
- [ ] `a_columns` reference named `COL_*` constants, not magic literals.
- [ ] `b_columns` reference named `COL_*` constants, not magic literals.
- [ ] `a_columns.len() == b_columns.len()` and element-wise semantic
  alignment is explained in a comment.
- [ ] Endianness / packing convention on A matches the convention on B
  (cite both AIRs' COL_* documentation).
- [ ] `a_selector_column` is set; the gated rows match the linkage's
  semantic intent; padding rows are excluded.
- [ ] `b_selector_column` is set with the same rigour.
- [ ] `a_layer_index` / `b_layer_index` match how callers wire the
  `LayerChainProof`.
- [ ] At least one positive `joint_prove` + `joint_verify` test exists.
- [ ] At least one tampering test exists that mutates a tuple value on
  one side and asserts `joint_verify` rejects.
- [ ] Both source AIRs have a `// EXPOSED TO CROSS-AIR LINKAGE` comment
  block updated with the new descriptor's label and column indices.

A descriptor that fails any item above should be assumed
soundness-broken until proven otherwise.
