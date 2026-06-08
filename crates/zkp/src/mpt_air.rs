//! MPT inclusion-proof AIR (witness layer).
//!
//! Models a Merkle Patricia Trie inclusion proof as a sequence of "node
//! visit" rows. Each row records one node along the inclusion path from
//! root to leaf:
//!
//! ```text
//!   row 0:  root node           (parent_hash = root)
//!   row 1:  child of row 0
//!   ...
//!   row k:  leaf node           (is_terminal = 1, node_kind = 2)
//! ```
//!
//! For the AIR scope, this module treats the per-node hash check
//! (`keccak256(node_rlp) == node_hash`) as a primitive provable
//! separately by `keccak_air` / `keccak_constraints`. The MPT AIR's
//! job is the **tree-structure logic**: that the chain of node hashes
//! follows the proof, that depth increments by exactly 1 per row, and
//! that the terminal row is a leaf.
//!
//! # Soundness gaps (deferred)
//!
//! 1. **Per-node hash check** `node_hash == keccak256(node_rlp)` — out
//!    of scope; delegated to a separate `keccak_constraints` AIR via
//!    cross-AIR linkage (future task).
//! 2. **RLP-decoding consistency**: that `node_kind` in fact matches
//!    the RLP-decoded node type, that `path_nibble` matches the nibble
//!    consumed at this branch step, and that the child slot referenced
//!    contains exactly the next row's `node_hash`. These require an
//!    RLP-decode AIR (not yet built); host-side they are enforced by
//!    [`inclusion_witness`] which derives the witness rows from
//!    `verify_mpt_inclusion`-style walking.
//! 3. **Root binding**: row 0's `parent_hash` is the trie root, which
//!    must be bound externally to whatever the caller is proving (e.g.
//!    block-header `state_root`).

use crate::keccak::keccak256;
use crate::mpt::{Nibbles, MptNode, mpt_node_rlp};
use crate::rlp::{rlp_decode, RlpItem};

/// Maximum RLP-encoded node length supported by the AIR. Covers MPT
/// leaf and extension nodes (typically ≤ 64 bytes); MPT branch nodes
/// (potentially up to ~564 bytes in adversarial cases, ~150–250 bytes
/// in practice) need a separate larger-bound AIR variant or a
/// different binding strategy.
pub const MAX_RLP_LEN: usize = 256;

/// Witness row for a single MPT node along the inclusion path.
#[derive(Debug, Clone)]
pub struct InclusionRow {
    /// keccak256 of this node's RLP encoding.
    pub node_hash: [u8; 32],
    /// `node_hash` of the parent node in the inclusion path (the previous
    /// row's `node_hash`); for row 0 this equals the externally-supplied
    /// root (which itself equals `node_hash` for row 0 in any valid proof,
    /// since proof[0] hashes to the root). The cross-row constraint
    /// `parent_hash(ω·X) − node_hash(X) = 0` enforces this binding.
    pub parent_hash: [u8; 32],
    /// 0 = branch, 1 = extension, 2 = leaf.
    pub node_kind: u8,
    /// Nibble of the key consumed at this step (4 bits, only meaningful
    /// for branch nodes; 0 otherwise).
    pub path_nibble: u8,
    /// 0 = root, increments at each step.
    pub depth: u32,
    /// 1 iff this is a leaf row (last row in the path), 0 otherwise.
    pub is_terminal: u8,
    /// RLP encoding of this node, zero-padded to [`MAX_RLP_LEN`].
    /// Cross-AIR LogUp linkage to KeccakExtract matches against
    /// `(node_rlp[..node_rlp_len], node_hash)` to bind
    /// `node_hash = keccak256(node_rlp[..node_rlp_len])`.
    pub node_rlp: [u8; MAX_RLP_LEN],
    /// Actual RLP length (`node_rlp[node_rlp_len..]` is zero padding).
    pub node_rlp_len: usize,
    /// Decoded leaf path bytes (HP-packed, 2 nibbles per byte) — only
    /// populated when this row matches the Phase 1 leaf gadget shape;
    /// otherwise zero. Bound algebraically to `node_rlp[4..36]` via the
    /// cross-AIR LogUp linkage to [`crate::mpt_rlp_air`].
    pub key_path_bytes: [u8; 32],
    /// Decoded leaf value bytes — same gating as `key_path_bytes`.
    pub value: [u8; 32],
    /// 1 iff this row's RLP exactly matches the Phase 1 leaf gadget
    /// shape (64-nibble even-length path, 32-byte value, 69-byte
    /// long-form list). 0 otherwise.
    pub is_phase1_leaf_shape: u8,
    /// 1 iff this row's RLP exactly matches the Phase 3 short-leaf
    /// gadget shape (4-nibble even-length path, 32-byte value, 38-byte
    /// short-form list). 0 otherwise. Mutually exclusive with
    /// `is_phase1_leaf_shape`.
    pub is_phase3_leaf_shape: u8,
    /// 1 iff this row's RLP exactly matches the Phase 4 extension
    /// gadget shape (4-nibble even-length path, 32-byte child hash,
    /// 38-byte short-form list, HP prefix 0x00). 0 otherwise.
    /// Mutually exclusive with the Phase 1 / Phase 3 leaf selectors.
    pub is_phase4_ext_shape: u8,
    /// Leading path nibble for odd-length-path rows (Phase 6). The
    /// low nibble of the HP byte. 0 on rows that don't match an
    /// odd-length-path shape.
    pub leading_path_nibble: u8,
    /// 1 iff this row's RLP matches the Phase 6 odd-length-path leaf
    /// shape (3-nibble odd path, 32-byte value, 37-byte short-form
    /// list, HP prefix `0x30 | leading_nibble`). Mutually exclusive
    /// with all other Phase selectors.
    pub is_phase6_odd_leaf_shape: u8,
    /// 1 iff this row's RLP matches the Phase 7 6-nibble even-length
    /// leaf shape (6-nibble even path, 32-byte value, 39-byte
    /// short-form list, HP prefix 0x20). Mutually exclusive with all
    /// other Phase selectors.
    pub is_phase7_leaf_6n_shape: u8,
    /// 1 iff this row's RLP matches the Phase 8a branch shape:
    /// 17-list, children at slots 0 and 1 only, 14 empty child
    /// slots, empty value slot, 83-byte long-form-list RLP.
    /// Mutually exclusive with all other Phase selectors.
    pub is_phase8_branch_01_shape: u8,
    /// Decoded child-hash bytes for slot 0 of a Phase 8a branch.
    /// Populated from `encoded[3..35]` on Phase 8a rows; zero
    /// elsewhere.
    pub branch_child_0_hash_bytes: [u8; 32],
    /// Decoded "second occupied child" hash bytes. On Phase 8a rows
    /// holds the slot-1 child hash (from `encoded[36..68]`); on
    /// Phase 9 rows holds the slot-5 child hash (from
    /// `encoded[40..72]`). Zero on rows matching neither branch shape.
    pub branch_child_1_hash_bytes: [u8; 32],
    /// 1 iff this row's RLP matches the Phase 9 branch shape
    /// (children at slots 0 and 5, 83-byte long-form list with
    /// shifted internal byte positions vs Phase 8a). Mutually
    /// exclusive with other Phase selectors.
    pub is_phase9_branch_05_shape: u8,
    /// 1 iff this row's RLP matches the Phase 10 branch shape
    /// (children at slots 1 and 5, with slot 0 empty — first
    /// occupied slot is NOT 0). Mutually exclusive with other Phase
    /// selectors.
    pub is_phase10_branch_15_shape: u8,
    /// 1 iff this row's RLP matches the Phase 11 8-nibble even-length
    /// leaf shape (8-nibble even path, 32-byte value, 40-byte
    /// short-form list, HP prefix 0x20). Mutually exclusive with all
    /// other Phase selectors.
    pub is_phase11_leaf_8n_shape: u8,
    /// **A2 step 1d (#52 step 1d) selector**: 1 iff this is the FIRST
    /// row of an inclusion-proof chain (depth = 0). Used as the gate
    /// for cross-AIR LogUp linkages from gadgets that need to bind
    /// the trie root (e.g. storage_access_air's storage_root). On
    /// row 0 of each chain, `parent_hash` equals the trie root, so
    /// projecting `(parent_hash[0..32])` gated by `is_root` yields
    /// one tuple per inclusion proof.
    pub is_root: u8,
    /// **A2 step 1d-leaf-summary witness column**: 32 bytes of the
    /// final leaf's value, populated on EVERY row of the chain by
    /// the witness builder (constancy across the chain). The future
    /// cross-row constancy constraint plus a leaf-row equality
    /// constraint will algebraically bind it to `value_byte` at the
    /// leaf row. The column is populated host-side for now, with
    /// constraints deferred — see linkage descriptor in
    /// `storage_access_air::make_storage_root_value_to_mpt_root_linkage_descriptor`.
    pub claimed_leaf_value_bytes: [u8; 32],
    /// **A2 step 1d-leaf-key-summary witness column**: 32 bytes of
    /// the final leaf's `key_path_bytes`, propagated constant across
    /// the chain. For single-leaf tries this equals the full storage
    /// trie key (= keccak256(slot_be)); for multi-row chains it is
    /// the leaf's REMAINING key suffix after branch consumption.
    /// Bound algebraically via row-local constraint (leaf-row
    /// equality with `key_path_bytes`) + shifted constraint
    /// (cross-row constancy gated by `1 - is_terminal`).
    pub claimed_leaf_key_bytes: [u8; 32],
    /// **A2 step 1d-full-key witness column** (multi-row prefix
    /// accumulator scaffold): 32 bytes of the FULL trie key being
    /// proven. For single-leaf tries this equals
    /// `claimed_leaf_key_bytes`. For multi-row tries this includes
    /// the consumed-nibble prefix from branch/extension nodes plus
    /// the leaf's suffix.
    pub claimed_full_key_bytes: [u8; 32],
    /// **A2 step 1d-multirow nibble decomposition**: 64 nibbles of
    /// `claimed_full_key_bytes` (2 nibbles per byte, high-then-low).
    /// `claimed_full_key_nibbles[2i]` = high nibble of byte i,
    /// `claimed_full_key_nibbles[2i+1]` = low nibble of byte i.
    /// 4-bit range-checked via LogUp. Constrained by 32 byte-decomp
    /// equations `byte_i = 16 * nib_{2i} + nib_{2i+1}`.
    pub claimed_full_key_nibbles: [u8; 64],
    /// **A2 step 1d-multirow per-depth selectors**: 64 binary one-hot
    /// cols. `is_depth_eq[k] = 1` iff `depth == k`. Constrained by:
    /// (a) binary check for each k, (b) sum-to-one Σ is_depth_eq = 1,
    /// (c) linear-combination Σ k * is_depth_eq[k] = depth.
    /// Used in the per-branch nibble-equality constraint to extract
    /// `nibble_at_depth = Σ is_depth_eq[k] * claimed_full_key_nibbles[k]`.
    pub is_depth_eq: [u8; 64],
}

// ---------------------------------------------------------------------------
// Trace-column layout
// ---------------------------------------------------------------------------

/// Number of bytes in a node hash (keccak256 output).
pub const HASH_BYTES: usize = 32;

/// Column offsets in the MPT inclusion AIR trace.
pub mod col {
    use super::{HASH_BYTES, MAX_RLP_LEN};
    pub const NODE_HASH_OFFSET: usize = 0;
    pub const PARENT_HASH_OFFSET: usize = NODE_HASH_OFFSET + HASH_BYTES; // 32
    pub const NODE_KIND: usize = PARENT_HASH_OFFSET + HASH_BYTES;        // 64
    pub const PATH_NIBBLE: usize = NODE_KIND + 1;                        // 65
    pub const DEPTH: usize = PATH_NIBBLE + 1;                            // 66
    pub const IS_TERMINAL: usize = DEPTH + 1;                            // 67
    /// Per-row RLP encoding (256 bytes, zero-padded).
    pub const NODE_RLP_OFFSET: usize = IS_TERMINAL + 1;                  // 68
    /// Per-row actual RLP length (so the cross-AIR LogUp tuple to
    /// KeccakExtract can disambiguate inputs of different lengths).
    pub const NODE_RLP_LEN: usize = NODE_RLP_OFFSET + MAX_RLP_LEN;       // 324
    /// Decoded leaf path bytes (32 bytes; populated only on rows with
    /// `IS_PHASE1_LEAF_SHAPE = 1`, zero elsewhere). Bound to the gadget
    /// AIR (`crate::mpt_rlp_air`) via cross-AIR LogUp.
    pub const KEY_PATH_BYTE_OFFSET: usize = NODE_RLP_LEN + 1;            // 325
    /// Decoded leaf value bytes (32 bytes; same gating as KEY_PATH_BYTE).
    pub const VALUE_BYTE_OFFSET: usize = KEY_PATH_BYTE_OFFSET + 32;       // 357
    /// Selector firing on rows whose RLP exactly matches the Phase 1
    /// leaf gadget shape (64-nibble even path, 32-byte value, 69-byte
    /// long-form list). The cross-AIR LogUp to `mpt_rlp_air` uses this
    /// column as its A-side selector. Binary-constrained algebraically
    /// in `mpt_constraints`.
    pub const IS_PHASE1_LEAF_SHAPE: usize = VALUE_BYTE_OFFSET + 32;       // 389
    /// Selector firing on rows whose RLP exactly matches the Phase 3
    /// short-leaf gadget shape (4-nibble even path, 32-byte value,
    /// 38-byte short-form list). The cross-AIR LogUp to
    /// `mpt_short_leaf_rlp_air` uses this column as its A-side selector.
    /// On Phase-3 rows the witness populates `KEY_PATH_BYTE[0..2]` from
    /// the leaf's path bytes and zero-pads `KEY_PATH_BYTE[2..32]`;
    /// `VALUE_BYTE[0..32]` carries the 32-byte value as usual.
    pub const IS_PHASE3_LEAF_SHAPE: usize = IS_PHASE1_LEAF_SHAPE + 1;    // 390
    /// Selector firing on rows whose RLP exactly matches the Phase 4
    /// extension gadget shape (4-nibble even path, 32-byte child-hash
    /// pointer, 38-byte short-form list). Same byte layout as Phase 3
    /// except HP prefix = 0x00 (extension flag) instead of 0x20 (leaf
    /// flag), and decoded NODE_KIND = 1 (extension) instead of 2 (leaf).
    /// On Phase-4 rows the witness populates `KEY_PATH_BYTE[0..2]` from
    /// the extension's path bytes and `VALUE_BYTE[0..32]` from the
    /// 32-byte child hash. (The "VALUE_BYTE" column is reused as the
    /// generic "decoded trailing 32-byte string" — semantics depend on
    /// which shape selector fires.)
    pub const IS_PHASE4_EXT_SHAPE: usize = IS_PHASE3_LEAF_SHAPE + 1;     // 391
    /// Decoded "leading path nibble" for odd-length-path rows. On
    /// Phase 6 (3-nibble odd leaf), the HP prefix byte is
    /// `0x30 | leading_nibble` where leading_nibble ∈ [0, 16). The
    /// cross-AIR LogUp to `mpt_odd_leaf_rlp_air` includes this column
    /// in its tuple. Zero on rows that don't match an odd-length-path
    /// shape. 4-bit range-checked.
    pub const LEADING_PATH_NIBBLE: usize = IS_PHASE4_EXT_SHAPE + 1;       // 392
    /// Selector firing on rows whose RLP exactly matches the Phase 6
    /// odd-length-path leaf gadget shape (3-nibble odd path, 32-byte
    /// value, 37-byte short-form list). Witness populates
    /// `LEADING_PATH_NIBBLE` from the HP byte's low nibble,
    /// `KEY_PATH_BYTE[0]` from the packed remaining-nibble byte, and
    /// `VALUE_BYTE[0..32]` from the leaf value.
    pub const IS_PHASE6_ODD_LEAF_SHAPE: usize = LEADING_PATH_NIBBLE + 1;  // 393
    /// Selector firing on rows whose RLP matches the Phase 7 6-nibble
    /// even-length-path leaf gadget shape (6-nibble even path, 32-byte
    /// value, 39-byte short-form list). Witness populates
    /// `KEY_PATH_BYTE[0..3]` from the leaf's path bytes (zero-pads
    /// indices 3..32) and `VALUE_BYTE[0..32]` from the leaf value.
    pub const IS_PHASE7_LEAF_6N_SHAPE: usize = IS_PHASE6_ODD_LEAF_SHAPE + 1; // 394
    /// Selector firing on rows whose RLP matches the Phase 8a branch
    /// shape: 17-list with EXACTLY children at slots 0 and 1, all
    /// other 14 child slots empty, value slot empty (83-byte
    /// long-form-list RLP). This is the simplest non-trivial branch
    /// shape, naturally produced by a 2-leaf trie whose keys differ
    /// at nibble 0 with values 0 and 1.
    pub const IS_PHASE8_BRANCH_01_SHAPE: usize = IS_PHASE7_LEAF_6N_SHAPE + 1; // 395
    /// Decoded child-hash bytes for slot 0 of a Phase 8a branch (32
    /// bytes). Populated only on Phase 8a rows; zero elsewhere. The
    /// Phase 8b cross-row constraint pins
    /// `(1 − PATH_NIBBLE) · BRANCH_CHILD_0_HASH_BYTE +
    ///  PATH_NIBBLE     · BRANCH_CHILD_1_HASH_BYTE
    ///  = NODE_HASH(ω·X)` — i.e. the path-nibble-selected child hash
    /// equals the next row's node hash, closing the inclusion chain
    /// through branches.
    pub const BRANCH_CHILD_0_HASH_BYTE_OFFSET: usize = IS_PHASE8_BRANCH_01_SHAPE + 1; // 396
    /// Decoded child-hash bytes for slot 1 of a Phase 8a branch.
    /// REUSED for Phase 9: holds the slot-5 child hash on Phase 9 rows.
    /// Generic name "second occupied child"; the active shape selector
    /// (`IS_PHASE8_BRANCH_01_SHAPE` vs `IS_PHASE9_BRANCH_05_SHAPE`)
    /// disambiguates which slot it represents.
    pub const BRANCH_CHILD_1_HASH_BYTE_OFFSET: usize = BRANCH_CHILD_0_HASH_BYTE_OFFSET + 32; // 428
    /// Selector firing on rows whose RLP matches the Phase 9 branch
    /// shape: 17-list with EXACTLY children at slots 0 and 5, all
    /// other 14 child slots empty, value slot empty (also 83-byte
    /// long-form-list RLP, since payload is still 81 bytes — the
    /// child-hash bytes occupy different positions than Phase 8a).
    /// Mutually exclusive with all other shape selectors.
    pub const IS_PHASE9_BRANCH_05_SHAPE: usize = BRANCH_CHILD_1_HASH_BYTE_OFFSET + 32; // 460
    /// Selector firing on rows whose RLP matches the Phase 10 branch
    /// shape: 17-list with EXACTLY children at slots 1 and 5, all
    /// other 14 child slots empty, value slot empty (83-byte
    /// long-form-list RLP). The first occupied slot is at position 1
    /// (NOT 0) — demonstrates the Lagrange chain-binding pattern
    /// works for slot pairs not anchored at 0. BRANCH_CHILD_0 holds
    /// the slot-1 hash; BRANCH_CHILD_1 holds the slot-5 hash.
    pub const IS_PHASE10_BRANCH_15_SHAPE: usize = IS_PHASE9_BRANCH_05_SHAPE + 1; // 461
    /// Selector firing on rows whose RLP matches the Phase 11 8-nibble
    /// even-length-path leaf gadget shape (8-nibble even path, 32-byte
    /// value, 40-byte short-form list, HP prefix 0x20). Witness
    /// populates `KEY_PATH_BYTE[0..4]` from the leaf's path bytes
    /// (zero-pads indices 4..32) and `VALUE_BYTE[0..32]` from the leaf
    /// value. Mutually exclusive with all other Phase selectors.
    pub const IS_PHASE11_LEAF_8N_SHAPE: usize = IS_PHASE10_BRANCH_15_SHAPE + 1; // 462
    /// A2 step 1d gate: 1 iff this is the first row of a proof chain.
    pub const IS_ROOT: usize = IS_PHASE11_LEAF_8N_SHAPE + 1;                 // 463
    /// A2 step 1d leaf-summary: 32 bytes of the leaf value, propagated
    /// constant across every row of the chain. Algebraically bound
    /// via row-local constraint 16 + shifted body 5.
    pub const CLAIMED_LEAF_VALUE_BYTE_OFFSET: usize = IS_ROOT + 1;           // 464
    /// A2 step 1d leaf-key-summary: 32 bytes of the leaf's `key_path_bytes`,
    /// propagated constant across the chain. For single-leaf tries =
    /// full storage trie key; for multi-row chains = leaf suffix only.
    /// Bound via row-local constraint 17 + shifted body 6.
    pub const CLAIMED_LEAF_KEY_BYTE_OFFSET: usize = CLAIMED_LEAF_VALUE_BYTE_OFFSET + 32; // 496
    /// A2 step 1d-full-key (multi-row prefix accumulator scaffold):
    /// 32 bytes of the FULL trie key, propagated constant across
    /// the chain. Bound via shifted body 7 (cross-row constancy).
    pub const CLAIMED_FULL_KEY_BYTE_OFFSET: usize = CLAIMED_LEAF_KEY_BYTE_OFFSET + 32;  // 528
    /// A2 step 1d-multirow: 64 nibbles of `claimed_full_key_bytes`,
    /// 2 per byte (high-then-low). Cols 560..624.
    pub const CLAIMED_FULL_KEY_NIBBLE_OFFSET: usize = CLAIMED_FULL_KEY_BYTE_OFFSET + 32; // 560
    /// A2 step 1d-multirow: 64 per-depth one-hot selectors.
    /// Cols 624..688.
    pub const IS_DEPTH_EQ_OFFSET: usize = CLAIMED_FULL_KEY_NIBBLE_OFFSET + 64; // 624
    pub const NUM_COLUMNS: usize = IS_DEPTH_EQ_OFFSET + 64;                  // 688
}

/// Build the inclusion-proof witness rows by walking `proof` against `key`.
///
/// `proof` is a sequence of RLP-encoded MPT nodes, with `proof[0]`
/// hashing to the trie root and each subsequent node being the one
/// pointed to by the previous step at the next nibble. The last node
/// must be a leaf whose stored value matches the proven value (this
/// helper does not check the value — that is the caller's job).
///
/// Returns one row per proof node. Returns an empty `Vec` if the proof
/// is malformed (RLP decode failure, unsupported branch slot type, etc.).
pub fn inclusion_witness(key: &[u8], proof: &[Vec<u8>]) -> Vec<InclusionRow> {
    let target_nibbles = Nibbles::from_bytes(key);
    let mut path = &target_nibbles.0[..];
    let mut rows: Vec<InclusionRow> = Vec::with_capacity(proof.len());

    // The first row's parent_hash equals the keccak256 of the first
    // proof node — that is the trie root and equals row 0's node_hash.
    let mut parent_hash: [u8; 32] = if proof.is_empty() {
        [0u8; 32]
    } else {
        keccak256(&proof[0])
    };

    for (i, encoded) in proof.iter().enumerate() {
        // Reject nodes whose RLP exceeds MAX_RLP_LEN — the AIR cannot
        // commit them. (MPT branch nodes can exceed 256 bytes; callers
        // hitting this need a wider AIR variant.)
        if encoded.len() > MAX_RLP_LEN {
            return Vec::new();
        }
        let node_hash = keccak256(encoded);
        // Decode the node to determine kind and (if branch) which nibble
        // is consumed.
        let (item, consumed) = match rlp_decode(encoded) {
            Ok(x) => x,
            Err(_) => return Vec::new(),
        };
        if consumed != encoded.len() {
            return Vec::new();
        }
        let list = match &item {
            RlpItem::List(l) => l,
            _ => return Vec::new(),
        };

        let is_last = i + 1 == proof.len();
        let mut node_kind: u8;
        let mut path_nibble: u8 = 0;

        match list.len() {
            2 => {
                // Leaf or extension. Inspect the HP prefix.
                let hp = match &list[0] {
                    RlpItem::Bytes(b) => b,
                    _ => return Vec::new(),
                };
                let (nibbles, is_leaf) = match Nibbles::from_encoded_path(hp) {
                    Some(p) => p,
                    None => return Vec::new(),
                };
                if path.len() < nibbles.0.len()
                    || path[..nibbles.0.len()] != nibbles.0[..]
                {
                    return Vec::new();
                }
                path = &path[nibbles.0.len()..];
                if is_leaf {
                    node_kind = 2;
                } else {
                    node_kind = 1;
                    // Sanity: extension's child slot must be a 32-byte hash.
                    let child = match &list[1] {
                        RlpItem::Bytes(b) => b,
                        _ => return Vec::new(),
                    };
                    if child.len() != 32 {
                        return Vec::new();
                    }
                }
            }
            17 => {
                node_kind = 0;
                if path.is_empty() {
                    // Branch-with-value: terminal at this branch. The AIR
                    // does not currently distinguish branch-with-value
                    // from leaf-terminal; we coerce node_kind = 2 below
                    // for the terminal-is-leaf check.
                } else {
                    let nib = path[0];
                    path_nibble = nib;
                    // Sanity: child slot must be a 32-byte hash.
                    let slot = &list[nib as usize];
                    let b = match slot {
                        RlpItem::Bytes(b) => b,
                        _ => return Vec::new(),
                    };
                    if b.len() != 32 {
                        return Vec::new();
                    }
                    path = &path[1..];
                }
            }
            _ => return Vec::new(),
        }

        let is_terminal = is_last as u8;
        // Force terminal-row semantics: if this is the last row but the
        // RLP shape isn't a 2-list-leaf (e.g. branch-with-value),
        // synthesize node_kind = 2 to satisfy the terminal check —
        // documented soundness gap (RLP/structural binding deferred).
        if is_last {
            node_kind = 2;
        }

        let mut node_rlp = [0u8; MAX_RLP_LEN];
        node_rlp[..encoded.len()].copy_from_slice(encoded);

        // Detect Phase 1 leaf gadget shape (see crate::mpt_rlp_air for
        // the canonical 69-byte layout). Conditions:
        //   - terminal leaf (node_kind == 2, is_terminal == 1)
        //   - encoded.len() == 69 (long-form list with 33-byte HP path
        //     and 32-byte value)
        //   - structural bytes match the canonical shape
        // We populate `key_path_bytes` from RLP bytes [4..36] and
        // `value` from RLP bytes [37..69]. On any mismatch we leave the
        // decoded fields zero and the selector off.
        let mut key_path_bytes = [0u8; 32];
        let mut value_bytes = [0u8; 32];
        let mut is_phase1_leaf_shape: u8 = 0;
        let mut is_phase3_leaf_shape: u8 = 0;
        let mut is_phase4_ext_shape: u8 = 0;
        let mut leading_path_nibble: u8 = 0;
        let mut is_phase6_odd_leaf_shape: u8 = 0;
        let mut is_phase7_leaf_6n_shape: u8 = 0;
        let mut is_phase8_branch_01_shape: u8 = 0;
        let mut is_phase9_branch_05_shape: u8 = 0;
        let mut is_phase10_branch_15_shape: u8 = 0;
        let mut is_phase11_leaf_8n_shape: u8 = 0;
        let mut branch_child_0_hash_bytes = [0u8; 32];
        let mut branch_child_1_hash_bytes = [0u8; 32];
        if is_last
            && node_kind == 2
            && encoded.len() == 69
            && encoded[0] == 0xf8
            && encoded[1] == 0x43
            && encoded[2] == 0xa1
            && encoded[3] == 0x20
            && encoded[36] == 0xa0
        {
            // Phase 1: 64-nibble even-length path + 32-byte value,
            // 69-byte long-form list.
            key_path_bytes.copy_from_slice(&encoded[4..36]);
            value_bytes.copy_from_slice(&encoded[37..69]);
            is_phase1_leaf_shape = 1;
        } else if is_last
            && node_kind == 2
            && encoded.len() == 38
            && encoded[0] == 0xe5
            && encoded[1] == 0x83
            && encoded[2] == 0x20
            && encoded[5] == 0xa0
        {
            // Phase 3: 4-nibble even-length path + 32-byte value,
            // 38-byte short-form list. Layout:
            //   [0]   0xe5  short-list header (0xc0 + 37)
            //   [1]   0x83  HP-string header (0x80 + 3)
            //   [2]   0x20  HP prefix (leaf, even, no leading nibble)
            //   [3..5]      2 path bytes (4 packed nibbles)
            //   [5]   0xa0  value-string header (0x80 + 32)
            //   [6..38]     32 value bytes
            // KEY_PATH_BYTE bytes 2..32 are zero (the gadget's row-locals
            // pin them to zero so the cross-AIR LogUp tuple matches).
            key_path_bytes[..2].copy_from_slice(&encoded[3..5]);
            value_bytes.copy_from_slice(&encoded[6..38]);
            is_phase3_leaf_shape = 1;
        } else if !is_last
            && node_kind == 1
            && encoded.len() == 38
            && encoded[0] == 0xe5
            && encoded[1] == 0x83
            && encoded[2] == 0x00
            && encoded[5] == 0xa0
        {
            // Phase 4: 4-nibble even-length extension + 32-byte child
            // hash, 38-byte short-form list. Same byte layout as
            // Phase 3 except HP prefix = 0x00 (extension flag) instead
            // of 0x20 (leaf flag), and node_kind = 1 (extension).
            // VALUE_BYTE columns are reused for the 32-byte child hash
            // on Phase 4 rows; the IS_PHASE4_EXT_SHAPE selector
            // disambiguates them from leaf-row VALUE_BYTE semantics.
            key_path_bytes[..2].copy_from_slice(&encoded[3..5]);
            value_bytes.copy_from_slice(&encoded[6..38]);
            is_phase4_ext_shape = 1;
        } else if is_last
            && node_kind == 2
            && encoded.len() == 37
            && encoded[0] == 0xe4
            && encoded[1] == 0x82
            && (encoded[2] & 0xf0) == 0x30
            && encoded[4] == 0xa0
        {
            // Phase 6: 3-nibble odd-length leaf + 32-byte value,
            // 37-byte short-form list. Layout:
            //   [0]   0xe4         short-list header (0xc0 + 36)
            //   [1]   0x82         HP-string header (0x80 + 2)
            //   [2]   0x30 | leading_nibble  (HP prefix: leaf+odd+nibble)
            //   [3]   packed byte (mid_nibble << 4 | last_nibble)
            //   [4]   0xa0         value-string header
            //   [5..37]            32 value bytes
            // Decoded fields populated:
            //   - LEADING_PATH_NIBBLE = encoded[2] & 0xf
            //   - KEY_PATH_BYTE[0] = encoded[3]; KEY_PATH_BYTE[1..32] = 0
            //   - VALUE_BYTE[0..32] = encoded[5..37]
            leading_path_nibble = encoded[2] & 0xf;
            key_path_bytes[0] = encoded[3];
            value_bytes.copy_from_slice(&encoded[5..37]);
            is_phase6_odd_leaf_shape = 1;
        } else if is_last
            && node_kind == 2
            && encoded.len() == 39
            && encoded[0] == 0xe6
            && encoded[1] == 0x84
            && encoded[2] == 0x20
            && encoded[6] == 0xa0
        {
            // Phase 7: 6-nibble even-length leaf + 32-byte value,
            // 39-byte short-form list. Layout:
            //   [0]   0xe6   short-list header (0xc0 + 38)
            //   [1]   0x84   HP-string header (0x80 + 4)
            //   [2]   0x20   HP prefix (leaf+even, no leading nibble)
            //   [3..6]       3 path bytes (6 packed nibbles)
            //   [6]   0xa0   value-string header
            //   [7..39]      32 value bytes
            key_path_bytes[..3].copy_from_slice(&encoded[3..6]);
            value_bytes.copy_from_slice(&encoded[7..39]);
            is_phase7_leaf_6n_shape = 1;
        } else if is_last
            && node_kind == 2
            && encoded.len() == 40
            && encoded[0] == 0xe7
            && encoded[1] == 0x85
            && encoded[2] == 0x20
            && encoded[7] == 0xa0
        {
            // Phase 11: 8-nibble even-length leaf + 32-byte value,
            // 40-byte short-form list. Layout:
            //   [0]   0xe7   short-list header (0xc0 + 39)
            //   [1]   0x85   HP-string header (0x80 + 5)
            //   [2]   0x20   HP prefix (leaf+even, no leading nibble)
            //   [3..7]       4 path bytes (8 packed nibbles)
            //   [7]   0xa0   value-string header
            //   [8..40]      32 value bytes
            key_path_bytes[..4].copy_from_slice(&encoded[3..7]);
            value_bytes.copy_from_slice(&encoded[8..40]);
            is_phase11_leaf_8n_shape = 1;
        } else if node_kind == 0
            && encoded.len() == 83
            && encoded[0] == 0xf8
            && encoded[1] == 0x51
            && encoded[2] == 0xa0
            && encoded[35] == 0xa0
            && (68..82).all(|i| encoded[i] == 0x80)
            && encoded[82] == 0x80
        {
            // Phase 8a: branch with exactly children at slots 0 and 1,
            // 14 empty child slots, empty value slot. 83-byte
            // long-form-list RLP. Layout:
            //   [0]   0xf8   long-list header
            //   [1]   0x51   payload length = 81
            //   [2]   0xa0   slot 0 child-hash header
            //   [3..35]      32 bytes child_hash_0
            //   [35]  0xa0   slot 1 child-hash header
            //   [36..68]     32 bytes child_hash_1
            //   [68..82]     14 × 0x80 (empty child slots 2..15)
            //   [82]  0x80   empty value slot
            // Phase 8b extracts the child hashes into MPT-side
            // columns so the cross-row chain constraint can pin the
            // path-nibble-selected hash to the next row's NODE_HASH.
            is_phase8_branch_01_shape = 1;
            branch_child_0_hash_bytes.copy_from_slice(&encoded[3..35]);
            branch_child_1_hash_bytes.copy_from_slice(&encoded[36..68]);
        } else if node_kind == 0
            && encoded.len() == 83
            && encoded[0] == 0xf8
            && encoded[1] == 0x51
            && encoded[2] == 0xa0
            && (35..39).all(|i| encoded[i] == 0x80)
            && encoded[39] == 0xa0
            && (72..82).all(|i| encoded[i] == 0x80)
            && encoded[82] == 0x80
        {
            // Phase 9: branch with EXACTLY children at slots 0 and 5,
            // 14 empty child slots, empty value slot. 83-byte
            // long-form-list RLP. Layout:
            //   [0]   0xf8   long-list header
            //   [1]   0x51   payload length = 81
            //   [2]   0xa0   slot 0 child-hash header
            //   [3..35]      32 bytes child_hash_0
            //   [35..39]     4 × 0x80 (empty slots 1..4)
            //   [39]  0xa0   slot 5 child-hash header
            //   [40..72]     32 bytes child_hash_5
            //   [72..82]     10 × 0x80 (empty slots 6..15)
            //   [82]  0x80   empty value slot
            // The 32-byte slot-5 hash is stored in
            // `branch_child_1_hash_bytes` (the generic "second
            // occupied child" column, reused from Phase 8a).
            is_phase9_branch_05_shape = 1;
            branch_child_0_hash_bytes.copy_from_slice(&encoded[3..35]);
            branch_child_1_hash_bytes.copy_from_slice(&encoded[40..72]);
        } else if node_kind == 0
            && encoded.len() == 83
            && encoded[0] == 0xf8
            && encoded[1] == 0x51
            && encoded[2] == 0x80
            && encoded[3] == 0xa0
            && (36..39).all(|i| encoded[i] == 0x80)
            && encoded[39] == 0xa0
            && (72..82).all(|i| encoded[i] == 0x80)
            && encoded[82] == 0x80
        {
            // Phase 10: branch with EXACTLY children at slots 1 and 5
            // (slot 0 empty — first occupied slot is NOT 0).
            // 83-byte long-form-list RLP. Layout:
            //   [0]   0xf8   long-list header
            //   [1]   0x51   payload length = 81
            //   [2]   0x80   empty slot 0
            //   [3]   0xa0   slot 1 child-hash header
            //   [4..36]      32 bytes child_hash_1
            //   [36..39]     3 × 0x80 (empty slots 2..4)
            //   [39]  0xa0   slot 5 child-hash header
            //   [40..72]     32 bytes child_hash_5
            //   [72..82]     10 × 0x80 (empty slots 6..15)
            //   [82]  0x80   empty value slot
            // BRANCH_CHILD_0 = slot-1 hash; BRANCH_CHILD_1 = slot-5 hash.
            is_phase10_branch_15_shape = 1;
            branch_child_0_hash_bytes.copy_from_slice(&encoded[4..36]);
            branch_child_1_hash_bytes.copy_from_slice(&encoded[40..72]);
        }

        rows.push(InclusionRow {
            node_hash,
            parent_hash,
            node_kind,
            path_nibble,
            depth: i as u32,
            is_terminal,
            node_rlp,
            node_rlp_len: encoded.len(),
            key_path_bytes,
            value: value_bytes,
            is_phase1_leaf_shape,
            is_phase3_leaf_shape,
            is_phase4_ext_shape,
            leading_path_nibble,
            is_phase6_odd_leaf_shape,
            is_phase7_leaf_6n_shape,
            is_phase8_branch_01_shape,
            branch_child_0_hash_bytes,
            branch_child_1_hash_bytes,
            is_phase9_branch_05_shape,
            is_phase10_branch_15_shape,
            is_phase11_leaf_8n_shape,
            // A2 step 1d: is_root = 1 iff this is the first row of
            // the proof chain (depth = 0, parent_hash = trie root).
            is_root: if i == 0 { 1 } else { 0 },
            // claimed_leaf_value_bytes / claimed_leaf_key_bytes /
            // claimed_full_key_bytes are populated in a second pass
            // below after the leaf is found. The full key for now is
            // taken from the input `key` parameter directly (which
            // is the claimed key being proven).
            claimed_leaf_value_bytes: [0u8; 32],
            claimed_leaf_key_bytes: [0u8; 32],
            claimed_full_key_bytes: [0u8; 32],
            claimed_full_key_nibbles: [0u8; 64],
            is_depth_eq: [0u8; 64],
        });

        // The next row's parent_hash is this row's node_hash — the
        // cross-row constraint binds them.
        parent_hash = node_hash;
    }
    // A2 step 1d-leaf-summary: backfill `claimed_leaf_value_bytes`,
    // `claimed_leaf_key_bytes`, and `claimed_full_key_bytes` on EVERY
    // row.
    //
    // claimed_full_key_bytes = the input `key` parameter (the claimed
    // full trie key being proven). For single-leaf tries this equals
    // `key_path_bytes`; for multi-row tries this is the full key
    // including consumed prefix.
    let mut claimed_full_key = [0u8; 32];
    let key_len = key.len().min(32);
    claimed_full_key[..key_len].copy_from_slice(&key[..key_len]);
    // Decompose into 64 nibbles, high-then-low per byte.
    let mut claimed_full_key_nibbles = [0u8; 64];
    for i in 0..32 {
        claimed_full_key_nibbles[2 * i] = (claimed_full_key[i] >> 4) & 0xf;
        claimed_full_key_nibbles[2 * i + 1] = claimed_full_key[i] & 0xf;
    }
    if let Some(leaf) = rows.last() {
        let leaf_value = leaf.value;
        let leaf_key = leaf.key_path_bytes;
        for row in &mut rows {
            row.claimed_leaf_value_bytes = leaf_value;
            row.claimed_leaf_key_bytes = leaf_key;
            row.claimed_full_key_bytes = claimed_full_key;
            row.claimed_full_key_nibbles = claimed_full_key_nibbles;
            // is_depth_eq one-hot for this row's depth.
            let d = row.depth as usize;
            if d < 64 {
                row.is_depth_eq[d] = 1;
            }
        }
    }
    rows
}

/// Convenience: build an inclusion witness for `(key, value)` against a
/// single-leaf trie. Returns the witness rows; useful for tests.
pub fn single_leaf_inclusion_witness(key: &[u8], value: &[u8]) -> Vec<InclusionRow> {
    let leaf = MptNode::Leaf {
        path: Nibbles::from_bytes(key),
        value: value.to_vec(),
    };
    let leaf_rlp = mpt_node_rlp(&leaf);
    inclusion_witness(key, &[leaf_rlp])
}

/// Populate AIR trace columns from a row sequence. `columns.len()` must
/// be at least `col::NUM_COLUMNS`; each inner Vec must be at least
/// `rows.len()` long (the caller pads to a power of two for FFT).
pub fn populate_trace(rows: &[InclusionRow], columns: &mut [Vec<u64>]) {
    assert!(columns.len() >= col::NUM_COLUMNS);
    for (row_idx, row) in rows.iter().enumerate() {
        for i in 0..HASH_BYTES {
            columns[col::NODE_HASH_OFFSET + i][row_idx] = row.node_hash[i] as u64;
            columns[col::PARENT_HASH_OFFSET + i][row_idx] = row.parent_hash[i] as u64;
        }
        columns[col::NODE_KIND][row_idx] = row.node_kind as u64;
        columns[col::PATH_NIBBLE][row_idx] = row.path_nibble as u64;
        columns[col::DEPTH][row_idx] = row.depth as u64;
        columns[col::IS_TERMINAL][row_idx] = row.is_terminal as u64;
        for b in 0..MAX_RLP_LEN {
            columns[col::NODE_RLP_OFFSET + b][row_idx] = row.node_rlp[b] as u64;
        }
        columns[col::NODE_RLP_LEN][row_idx] = row.node_rlp_len as u64;
        for b in 0..32 {
            columns[col::KEY_PATH_BYTE_OFFSET + b][row_idx] = row.key_path_bytes[b] as u64;
            columns[col::VALUE_BYTE_OFFSET + b][row_idx] = row.value[b] as u64;
        }
        columns[col::IS_PHASE1_LEAF_SHAPE][row_idx] = row.is_phase1_leaf_shape as u64;
        columns[col::IS_PHASE3_LEAF_SHAPE][row_idx] = row.is_phase3_leaf_shape as u64;
        columns[col::IS_PHASE4_EXT_SHAPE][row_idx] = row.is_phase4_ext_shape as u64;
        columns[col::LEADING_PATH_NIBBLE][row_idx] = row.leading_path_nibble as u64;
        columns[col::IS_PHASE6_ODD_LEAF_SHAPE][row_idx] = row.is_phase6_odd_leaf_shape as u64;
        columns[col::IS_PHASE7_LEAF_6N_SHAPE][row_idx] = row.is_phase7_leaf_6n_shape as u64;
        columns[col::IS_PHASE8_BRANCH_01_SHAPE][row_idx] = row.is_phase8_branch_01_shape as u64;
        for b in 0..32 {
            columns[col::BRANCH_CHILD_0_HASH_BYTE_OFFSET + b][row_idx] =
                row.branch_child_0_hash_bytes[b] as u64;
            columns[col::BRANCH_CHILD_1_HASH_BYTE_OFFSET + b][row_idx] =
                row.branch_child_1_hash_bytes[b] as u64;
        }
        columns[col::IS_PHASE9_BRANCH_05_SHAPE][row_idx] = row.is_phase9_branch_05_shape as u64;
        columns[col::IS_PHASE10_BRANCH_15_SHAPE][row_idx] = row.is_phase10_branch_15_shape as u64;
        columns[col::IS_PHASE11_LEAF_8N_SHAPE][row_idx] = row.is_phase11_leaf_8n_shape as u64;
        columns[col::IS_ROOT][row_idx] = row.is_root as u64;
        for b in 0..32 {
            columns[col::CLAIMED_LEAF_VALUE_BYTE_OFFSET + b][row_idx] =
                row.claimed_leaf_value_bytes[b] as u64;
            columns[col::CLAIMED_LEAF_KEY_BYTE_OFFSET + b][row_idx] =
                row.claimed_leaf_key_bytes[b] as u64;
            columns[col::CLAIMED_FULL_KEY_BYTE_OFFSET + b][row_idx] =
                row.claimed_full_key_bytes[b] as u64;
        }
        for k in 0..64 {
            columns[col::CLAIMED_FULL_KEY_NIBBLE_OFFSET + k][row_idx] =
                row.claimed_full_key_nibbles[k] as u64;
            columns[col::IS_DEPTH_EQ_OFFSET + k][row_idx] =
                row.is_depth_eq[k] as u64;
        }
    }
}

/// Host-side chain-consistency check: row r's `parent_hash` matches row
/// r-1's `node_hash` (root assumed verified externally for row 0). The
/// last row must additionally have `is_terminal == 1` and
/// `node_kind == 2`.
pub fn rows_chain_consistent(rows: &[InclusionRow]) -> bool {
    if rows.is_empty() {
        return false;
    }
    for r in 1..rows.len() {
        if rows[r].parent_hash != rows[r - 1].node_hash {
            return false;
        }
    }
    let last = &rows[rows.len() - 1];
    last.is_terminal == 1 && last.node_kind == 2
}

/// Host-side depth-monotonicity check: depth strictly increases (by
/// exactly 1 per row, starting at 0).
pub fn rows_depth_monotone(rows: &[InclusionRow]) -> bool {
    if rows.is_empty() {
        return true;
    }
    if rows[0].depth != 0 {
        return false;
    }
    for r in 1..rows.len() {
        if rows[r].depth != rows[r - 1].depth + 1 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpt::{single_leaf_trie, verify_mpt_inclusion};

    #[test]
    fn witness_single_leaf_one_row() {
        let key = vec![0xab, 0xcd];
        let value = b"payload".to_vec();
        let (root, proof) = single_leaf_trie(&key, &value);
        // Sanity: the proof verifies via the reference walker.
        assert!(verify_mpt_inclusion(root, &key, &value, &proof));

        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].depth, 0);
        assert_eq!(rows[0].is_terminal, 1);
        assert_eq!(rows[0].node_kind, 2);
        assert_eq!(rows[0].parent_hash, root);
        assert_eq!(rows[0].node_hash, root);
        assert!(rows_chain_consistent(&rows));
        assert!(rows_depth_monotone(&rows));
    }

    #[test]
    fn is_root_set_only_on_first_row() {
        // Single-leaf trie: 1 row, is_root = 1.
        let key = vec![0xab, 0xcd];
        let (root, proof) = single_leaf_trie(&key, b"v");
        let rows = inclusion_witness(&key, &proof);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].is_root, 1);
        let _ = root;

        // Two-leaf trie via two_leaf_trie_distinct_first_nibble:
        // 2 rows, is_root = 1 only on row 0 (the branch).
        let k1 = vec![0x05u8, 0xab];
        let k2 = vec![0x16u8, 0xcd];
        let (_, proof2) =
            crate::mpt::two_leaf_trie_distinct_first_nibble(&k1, b"alpha", &k2, b"beta");
        let rows2 = inclusion_witness(&k1, &proof2);
        assert_eq!(rows2.len(), 2);
        assert_eq!(rows2[0].is_root, 1);
        assert_eq!(rows2[1].is_root, 0, "leaf row should have is_root = 0");
    }

    #[test]
    fn is_root_column_in_trace() {
        let key = vec![0xab, 0xcd];
        let (_, proof) = single_leaf_trie(&key, b"v");
        let rows = inclusion_witness(&key, &proof);
        let n = rows.len().max(1);
        let mut columns: Vec<Vec<u64>> = (0..col::NUM_COLUMNS).map(|_| vec![0u64; n]).collect();
        populate_trace(&rows, &mut columns);
        assert_eq!(columns[col::IS_ROOT][0], 1);
    }

    #[test]
    fn witness_branch_then_leaf_two_rows() {
        // Mirror `mpt::tests::test_branch_then_leaf_inclusion`.
        let key_a = vec![0x1a, 0xbc];
        let val_a = b"value-a".to_vec();
        let _key_b = vec![0x2d, 0xef];
        let val_b = b"value-b".to_vec();

        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xa, 0xb, 0xc]),
            value: val_a.clone(),
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xd, 0xe, 0xf]),
            value: val_b.clone(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let leaf_a_hash = keccak256(&leaf_a_rlp);
        let leaf_b_hash = keccak256(&leaf_b_rlp);

        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(leaf_a_hash);
        children[2] = Some(leaf_b_hash);
        let branch = MptNode::Branch {
            children,
            value: None,
        };
        let branch_rlp = mpt_node_rlp(&branch);

        let proof_a = vec![branch_rlp.clone(), leaf_a_rlp.clone()];
        let rows = inclusion_witness(&key_a, &proof_a);
        assert_eq!(rows.len(), 2);
        // Row 0: branch.
        assert_eq!(rows[0].depth, 0);
        assert_eq!(rows[0].is_terminal, 0);
        assert_eq!(rows[0].node_kind, 0);
        assert_eq!(rows[0].path_nibble, 1);
        // Row 1: leaf.
        assert_eq!(rows[1].depth, 1);
        assert_eq!(rows[1].is_terminal, 1);
        assert_eq!(rows[1].node_kind, 2);
        // Chain: row 1's parent == row 0's node hash.
        assert_eq!(rows[1].parent_hash, rows[0].node_hash);
        assert!(rows_chain_consistent(&rows));
        assert!(rows_depth_monotone(&rows));
    }

    #[test]
    fn witness_extension_branch_leaf_three_rows() {
        let common_nibbles = vec![0xa, 0xb, 0xc, 0xd];
        let key_a = vec![0xab, 0xcd, 0x1a, 0xbc];
        let val_a = b"aa".to_vec();

        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xa, 0xb, 0xc]),
            value: val_a.clone(),
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xd, 0xe, 0xf]),
            value: b"bb".to_vec(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);

        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(keccak256(&leaf_a_rlp));
        children[2] = Some(keccak256(&leaf_b_rlp));
        let branch = MptNode::Branch {
            children,
            value: None,
        };
        let branch_rlp = mpt_node_rlp(&branch);
        let branch_hash = keccak256(&branch_rlp);

        let ext = MptNode::Extension {
            path: Nibbles::from_nibbles(common_nibbles),
            child: branch_hash,
        };
        let ext_rlp = mpt_node_rlp(&ext);

        let proof_a = vec![ext_rlp.clone(), branch_rlp.clone(), leaf_a_rlp.clone()];
        let rows = inclusion_witness(&key_a, &proof_a);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].node_kind, 1); // extension
        assert_eq!(rows[1].node_kind, 0); // branch
        assert_eq!(rows[2].node_kind, 2); // leaf (terminal)
        assert_eq!(rows[2].is_terminal, 1);
        assert!(rows_chain_consistent(&rows));
        assert!(rows_depth_monotone(&rows));
    }

    #[test]
    fn rows_chain_consistent_detects_broken_chain() {
        let key_a = vec![0x1a, 0xbc];
        let val_a = b"value-a".to_vec();
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xa, 0xb, 0xc]),
            value: val_a.clone(),
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xd, 0xe, 0xf]),
            value: b"vb".to_vec(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(keccak256(&leaf_a_rlp));
        children[2] = Some(keccak256(&leaf_b_rlp));
        let branch = MptNode::Branch { children, value: None };
        let branch_rlp = mpt_node_rlp(&branch);
        let proof_a = vec![branch_rlp.clone(), leaf_a_rlp.clone()];
        let mut rows = inclusion_witness(&key_a, &proof_a);
        // Tamper row 1's parent_hash → chain breaks.
        rows[1].parent_hash[0] ^= 0xff;
        assert!(!rows_chain_consistent(&rows));
    }

    #[test]
    fn trace_column_populator_round_trips() {
        let key = vec![0xab, 0xcd];
        let value = b"payload".to_vec();
        let (_root, proof) = single_leaf_trie(&key, &value);
        let rows = inclusion_witness(&key, &proof);
        let mut columns: Vec<Vec<u64>> =
            (0..col::NUM_COLUMNS).map(|_| vec![0u64; rows.len()]).collect();
        populate_trace(&rows, &mut columns);
        for (row_idx, row) in rows.iter().enumerate() {
            for i in 0..HASH_BYTES {
                assert_eq!(columns[col::NODE_HASH_OFFSET + i][row_idx], row.node_hash[i] as u64);
                assert_eq!(columns[col::PARENT_HASH_OFFSET + i][row_idx], row.parent_hash[i] as u64);
            }
            assert_eq!(columns[col::NODE_KIND][row_idx], row.node_kind as u64);
            assert_eq!(columns[col::PATH_NIBBLE][row_idx], row.path_nibble as u64);
            assert_eq!(columns[col::DEPTH][row_idx], row.depth as u64);
            assert_eq!(columns[col::IS_TERMINAL][row_idx], row.is_terminal as u64);
        }
    }

    #[test]
    fn rows_depth_monotone_detects_non_monotone() {
        let key = vec![0x1a, 0xbc];
        let leaf_a = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xa, 0xb, 0xc]),
            value: b"va".to_vec(),
        };
        let leaf_b = MptNode::Leaf {
            path: Nibbles::from_nibbles(vec![0xd, 0xe, 0xf]),
            value: b"vb".to_vec(),
        };
        let leaf_a_rlp = mpt_node_rlp(&leaf_a);
        let leaf_b_rlp = mpt_node_rlp(&leaf_b);
        let mut children: [Option<[u8; 32]>; 16] = Default::default();
        children[1] = Some(keccak256(&leaf_a_rlp));
        children[2] = Some(keccak256(&leaf_b_rlp));
        let branch = MptNode::Branch { children, value: None };
        let branch_rlp = mpt_node_rlp(&branch);
        let proof = vec![branch_rlp, leaf_a_rlp];
        let mut rows = inclusion_witness(&key, &proof);
        rows[1].depth = 5; // not monotone-by-1
        assert!(!rows_depth_monotone(&rows));
    }
}
