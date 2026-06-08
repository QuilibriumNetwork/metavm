//! Cross-layer claim chain.
//!
//! The MetaVM proof architecture stacks four layers:
//!
//! ```text
//!   1. Execution-layer trace      → produces  block_hash       (keccak256 of RLP header)
//!   2. Block-header binding       → produces  beacon_block_root (SSZ hash-tree-root)
//!   3. Consensus / attestation    → produces  attestation_data_root + agg sig
//!   4. Finality / stake weight    → produces  (finalized_root, total_effective_balance)
//! ```
//!
//! Each layer's proof exposes specific public outputs that bind to the
//! next layer's public inputs. This module defines the [`LayerClaim`] enum
//! capturing those boundary values and a [`LayerChain`] aggregator that
//! checks adjacent claims share the right boundary value, then extracts
//! the final tuple `(block_hash, beacon_root, finality_weight_gwei)` —
//! the "the network finalized this block under this much staked ETH"
//! statement the production proof binds.
//!
//! [`LayerChainFolder::fold_into_recursive_proof_scheme`] collapses every
//! provable layer's [`crate::prover::ExecutionProof`] into a single
//! [`crate::recursive::RecursiveProof`] via the IVC accumulator, and
//! [`crate::recursive::verify_final_scheme`] returns a single bool from
//! the accumulated `(L_acc, R_acc)` pairing plus the constraint identity
//! `scalar_acc`. The
//! [`LayerChainFolder::fold_into_recursive_proof_full_scheme`] variant
//! aggregates every per-layer KZG opening pairing and the per-AIR
//! constraint identity scalar so a single `verify_final_scheme` call
//! certifies each layer's full proof.
//!
//! Pairs naturally with [`crate::e2e_reference`] which walks all four
//! layers via their host-side reference implementations.

use crate::beacon::Gwei;
use crate::keccak::keccak256;

/// 32-byte hash type used at every layer boundary.
pub type Hash32 = [u8; 32];

/// One claim in the layer chain. Each variant carries the public input it
/// expects from the previous layer and the public output it commits to for
/// the next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerClaim {
    /// Execution-layer claim: a trace was correctly executed and its block
    /// header keccak256-hashes to `block_hash`.
    Execution {
        /// Output: the keccak256 of the RLP-encoded execution-layer header.
        block_hash: Hash32,
    },
    /// Block-binding claim: `block_hash` (input) is the
    /// `execution_payload_root` field of a beacon block whose
    /// SSZ hash-tree-root is `beacon_block_root`.
    BlockBinding {
        /// Input from the execution layer.
        block_hash: Hash32,
        /// Output: the beacon block's hash-tree-root.
        beacon_block_root: Hash32,
    },
    /// Attestation claim: an aggregate BLS signature over an
    /// `AttestationData` referencing `beacon_block_root` as its target.
    /// Public outputs: the attesting set's count + the merkle root of
    /// the attestation data.
    Attestation {
        /// Input.
        beacon_block_root: Hash32,
        /// Output: hash-tree-root of the AttestationData container.
        attestation_data_root: Hash32,
        /// Output: number of unique validator indices.
        num_attesters: u64,
    },
    /// Finality claim: validators backing `attestation_data_root` together
    /// hold `total_effective_balance_gwei` and constitute ≥ 2/3 of the
    /// active validator set, so the corresponding checkpoint is FFG-final.
    Finality {
        /// Input — the attestation root the finality argument consumes.
        attestation_data_root: Hash32,
        /// Output: the finalized block root.
        finalized_root: Hash32,
        /// Output: total effective balance backing finalization.
        total_effective_balance_gwei: Gwei,
    },
}

/// Errors for [`LayerChain`] consistency checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// Chain is empty.
    Empty,
    /// First claim must be `Execution`.
    FirstNotExecution,
    /// Last claim must be `Finality`.
    LastNotFinality,
    /// Claim at index `i` doesn't follow claim at index `i-1` by type.
    InvalidTransition { from_index: usize },
    /// Boundary value mismatch (e.g. `block_hash` output ≠ next claim's input).
    BoundaryMismatch { at_index: usize, what: &'static str },
}

/// An ordered chain of layer claims, expected to follow the canonical
/// `Execution → BlockBinding → Attestation → Finality` sequence.
#[derive(Debug, Clone, Default)]
pub struct LayerChain {
    pub claims: Vec<LayerClaim>,
}

impl LayerChain {
    pub fn new() -> Self {
        LayerChain { claims: Vec::new() }
    }

    pub fn push(&mut self, claim: LayerClaim) {
        self.claims.push(claim);
    }

    /// Check the chain is well-formed:
    /// - Starts with [`LayerClaim::Execution`].
    /// - Ends with [`LayerClaim::Finality`].
    /// - Adjacent claims have compatible types in the canonical order.
    /// - Each adjacent pair's boundary values agree.
    pub fn verify_consistency(&self) -> Result<(), ChainError> {
        if self.claims.is_empty() {
            return Err(ChainError::Empty);
        }
        if !matches!(self.claims[0], LayerClaim::Execution { .. }) {
            return Err(ChainError::FirstNotExecution);
        }
        if !matches!(self.claims.last(), Some(LayerClaim::Finality { .. })) {
            return Err(ChainError::LastNotFinality);
        }

        for i in 1..self.claims.len() {
            match (&self.claims[i - 1], &self.claims[i]) {
                (
                    LayerClaim::Execution { block_hash: out },
                    LayerClaim::BlockBinding { block_hash: inp, .. },
                ) => {
                    if out != inp {
                        return Err(ChainError::BoundaryMismatch {
                            at_index: i,
                            what: "block_hash",
                        });
                    }
                }
                (
                    LayerClaim::BlockBinding { beacon_block_root: out, .. },
                    LayerClaim::Attestation { beacon_block_root: inp, .. },
                ) => {
                    if out != inp {
                        return Err(ChainError::BoundaryMismatch {
                            at_index: i,
                            what: "beacon_block_root",
                        });
                    }
                }
                (
                    LayerClaim::Attestation { attestation_data_root: out, .. },
                    LayerClaim::Finality { attestation_data_root: inp, .. },
                ) => {
                    if out != inp {
                        return Err(ChainError::BoundaryMismatch {
                            at_index: i,
                            what: "attestation_data_root",
                        });
                    }
                }
                _ => {
                    return Err(ChainError::InvalidTransition { from_index: i - 1 });
                }
            }
        }
        Ok(())
    }

    /// Compute a single 32-byte commitment over every boundary value in
    /// the chain. This is the **public-input root** a recursive proof
    /// would commit to as its single output — once each layer's AIR is
    /// wired through a prover, a recursive fold can carry just this hash
    /// from layer to layer.
    ///
    /// Encoding (big-endian, total 168 bytes pre-hash):
    /// ```text
    ///   block_hash             (32 bytes)
    /// ‖ beacon_block_root      (32)
    /// ‖ attestation_data_root  (32)
    /// ‖ num_attesters          ( 8, BE u64)
    /// ‖ finalized_root         (32)
    /// ‖ total_effective_balance_gwei (8, BE u64)
    /// ```
    ///
    /// Returns `None` if the chain is malformed (use [`Self::final_output`]
    /// to detect this; `commitment` and `final_output` agree on
    /// validity).
    pub fn commitment(&self) -> Option<Hash32> {
        // Walk the chain (independently of `final_output` since we need
        // attestation_data_root + num_attesters which the FinalOutput
        // doesn't expose).
        self.verify_consistency().ok()?;
        let mut block_hash: Option<Hash32> = None;
        let mut beacon_root: Option<Hash32> = None;
        let mut att_root: Option<Hash32> = None;
        let mut num_atts: Option<u64> = None;
        let mut final_root: Option<Hash32> = None;
        let mut weight: Option<Gwei> = None;
        for c in &self.claims {
            match c {
                LayerClaim::Execution { block_hash: h } => block_hash = Some(*h),
                LayerClaim::BlockBinding { beacon_block_root: r, .. } => beacon_root = Some(*r),
                LayerClaim::Attestation {
                    attestation_data_root: r,
                    num_attesters: n,
                    ..
                } => {
                    att_root = Some(*r);
                    num_atts = Some(*n);
                }
                LayerClaim::Finality {
                    finalized_root: r,
                    total_effective_balance_gwei: w,
                    ..
                } => {
                    final_root = Some(*r);
                    weight = Some(*w);
                }
            }
        }
        let mut buf = Vec::with_capacity(32 * 4 + 8 * 2);
        buf.extend_from_slice(&block_hash?);
        buf.extend_from_slice(&beacon_root?);
        buf.extend_from_slice(&att_root?);
        buf.extend_from_slice(&num_atts?.to_be_bytes());
        buf.extend_from_slice(&final_root?);
        buf.extend_from_slice(&weight?.to_be_bytes());
        Some(keccak256(&buf))
    }

    /// Extract the final `(block_hash, beacon_block_root, finalized_root,
    /// total_effective_balance_gwei)` tuple — the public statement the
    /// production proof binds. Only valid for chains that pass
    /// [`Self::verify_consistency`].
    pub fn final_output(&self) -> Option<FinalOutput> {
        self.verify_consistency().ok()?;
        let mut block_hash: Option<Hash32> = None;
        let mut beacon_block_root: Option<Hash32> = None;
        let mut finalized_root: Option<Hash32> = None;
        let mut weight: Option<Gwei> = None;
        for c in &self.claims {
            match c {
                LayerClaim::Execution { block_hash: h } => block_hash = Some(*h),
                LayerClaim::BlockBinding { beacon_block_root: r, .. } => {
                    beacon_block_root = Some(*r)
                }
                LayerClaim::Finality {
                    finalized_root: r,
                    total_effective_balance_gwei: w,
                    ..
                } => {
                    finalized_root = Some(*r);
                    weight = Some(*w);
                }
                _ => {}
            }
        }
        Some(FinalOutput {
            block_hash: block_hash?,
            beacon_block_root: beacon_block_root?,
            finalized_root: finalized_root?,
            total_effective_balance_gwei: weight?,
        })
    }
}

/// All boundary values needed to assemble a canonical
/// `Execution → BlockBinding → Attestation → Finality` chain in one shot.
///
/// Convenience over hand-pushing four `LayerClaim` variants — used by both
/// the e2e_reference walk and any downstream caller that has the four
/// hashes plus weight/count to bind into a single chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainBoundaries {
    pub block_hash: Hash32,
    pub beacon_block_root: Hash32,
    pub attestation_data_root: Hash32,
    pub num_attesters: u64,
    pub finalized_root: Hash32,
    pub total_effective_balance_gwei: Gwei,
}

impl LayerChain {
    /// One-shot constructor: build the canonical 4-layer chain from a
    /// fully-populated [`ChainBoundaries`]. Equivalent to pushing four
    /// `LayerClaim` variants in order with the appropriate boundary values.
    pub fn from_boundaries(b: &ChainBoundaries) -> Self {
        let mut chain = LayerChain::new();
        chain.push(LayerClaim::Execution { block_hash: b.block_hash });
        chain.push(LayerClaim::BlockBinding {
            block_hash: b.block_hash,
            beacon_block_root: b.beacon_block_root,
        });
        chain.push(LayerClaim::Attestation {
            beacon_block_root: b.beacon_block_root,
            attestation_data_root: b.attestation_data_root,
            num_attesters: b.num_attesters,
        });
        chain.push(LayerClaim::Finality {
            attestation_data_root: b.attestation_data_root,
            finalized_root: b.finalized_root,
            total_effective_balance_gwei: b.total_effective_balance_gwei,
        });
        chain
    }
}

impl FinalOutput {
    /// Compute the same single-hash commitment as [`LayerChain::commitment`]
    /// but from the terminal `FinalOutput` plus the missing
    /// `attestation_data_root + num_attesters` (which `FinalOutput` itself
    /// does not carry — those live mid-chain). Returned hash is identical
    /// to `chain.commitment()` for the chain that produced this output.
    pub fn commitment_with(
        &self,
        attestation_data_root: &Hash32,
        num_attesters: u64,
    ) -> Hash32 {
        let mut buf = Vec::with_capacity(32 * 4 + 8 * 2);
        buf.extend_from_slice(&self.block_hash);
        buf.extend_from_slice(&self.beacon_block_root);
        buf.extend_from_slice(attestation_data_root);
        buf.extend_from_slice(&num_attesters.to_be_bytes());
        buf.extend_from_slice(&self.finalized_root);
        buf.extend_from_slice(&self.total_effective_balance_gwei.to_be_bytes());
        keccak256(&buf)
    }
}

/// Tag identifying which AIR (constraint system) produced a given layer's
/// proof bytes. Lets `LayerChainProof::verify_with_layer_verifier` and
/// downstream tooling dispatch on the right verifier without parsing the
/// bytes themselves. `ReferenceOnly` is the sentinel for layers that
/// haven't been wired to an AIR yet.
///
/// Each variant lists the matching constraint-system module:
///   * `NonnativeFp` → `crate::nonnative_fp_constraints`
///   * `Keccak`      → `crate::keccak_constraints`
///   * `Sha256`      → `crate::sha256_constraints`
///   * `Ssz`         → `crate::ssz_constraints`
///   * `Mpt`         → `crate::mpt_constraints`
///   * `EvmExp`      → `metavm_evm::exp_constraints` (256-row EXP gadget)
///   * `Vm{...}`     → execution-VM AIRs (RISC-V/EVM/SBF) — wired in their
///     own crates; `LayerProofKind` is just the tag here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerProofKind {
    ReferenceOnly,
    NonnativeFp,
    Keccak,
    Sha256,
    Ssz,
    Mpt,
    EvmExp,
    VmRiscv,
    VmEvm,
    VmSbf,
    /// Aggregate BLS signature verification (G1 pubkeys, G2 sigs).
    /// Wired via [`crate::bls_sig_constraints::BlsSigConstraintSystem`].
    BlsSig,
    /// Gasper finality + stake weighting (≥ 2/3 effective balance over a
    /// finalized checkpoint). Wired via
    /// [`crate::finality_constraints::FinalityConstraintSystem`].
    Finality,
}

impl LayerProofKind {
    /// Human-readable tag, useful for error messages and tracing.
    pub fn as_str(self) -> &'static str {
        match self {
            LayerProofKind::ReferenceOnly => "reference-only",
            LayerProofKind::NonnativeFp => "nonnative_fp",
            LayerProofKind::Keccak => "keccak",
            LayerProofKind::Sha256 => "sha256",
            LayerProofKind::Ssz => "ssz",
            LayerProofKind::Mpt => "mpt",
            LayerProofKind::EvmExp => "evm_exp",
            LayerProofKind::VmRiscv => "riscv",
            LayerProofKind::VmEvm => "evm",
            LayerProofKind::VmSbf => "sbf",
            LayerProofKind::BlsSig => "bls_sig",
            LayerProofKind::Finality => "finality",
        }
    }
}

/// One layer of a [`LayerChainProof`]: the public claim plus the bytes of
/// the per-layer execution proof that binds it.
///
/// The proof bytes are opaque from this module's perspective — verification
/// is delegated to the layer-specific verifier supplied via
/// [`LayerChainProof::verify_with_layer_verifier`] or
/// [`LayerChainFolder::fold_into_recursive_proof_full_scheme`]. Empty
/// `proof_bytes` (`reference_only`) means "trust the claim's public
/// values" — the chain commitment still binds them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerProof {
    pub claim: LayerClaim,
    /// Tag identifying which AIR produced [`Self::proof_bytes`].
    pub kind: LayerProofKind,
    /// Layer-specific proof bytes. Empty = "this layer is reference-checked
    /// only; trust the claim's public values." The chain-level commitment
    /// still binds the claim into the final hash.
    pub proof_bytes: Vec<u8>,
}

impl LayerProof {
    /// A reference-only layer (no AIR proof yet — only the public values).
    pub fn reference_only(claim: LayerClaim) -> Self {
        Self {
            claim,
            kind: LayerProofKind::ReferenceOnly,
            proof_bytes: Vec::new(),
        }
    }

    /// A layer with an actual proof attached, tagged by the AIR that
    /// produced the bytes.
    pub fn with_proof(claim: LayerClaim, kind: LayerProofKind, proof_bytes: Vec<u8>) -> Self {
        Self { claim, kind, proof_bytes }
    }

    /// Whether this layer carries an actual AIR proof. (`ReferenceOnly` and
    /// empty bytes are equivalent — both skip per-layer verification.)
    pub fn has_proof(&self) -> bool {
        !matches!(self.kind, LayerProofKind::ReferenceOnly) && !self.proof_bytes.is_empty()
    }
}

/// A chain of [`LayerProof`]s plus the matching consistency-checked
/// `LayerChain`. Verifying a `LayerChainProof` runs three things:
///
///   1. Every per-layer proof's claim equals the corresponding chain claim
///      (the chain is a strict view of the proofs' claims).
///   2. The chain itself passes [`LayerChain::verify_consistency`].
///   3. Every per-layer proof verifies against its layer-specific verifier
///      (supplied by the caller via
///      [`Self::verify_with_layer_verifier`] or
///      [`LayerChainFolder::fold`]; reference-only layers with empty
///      `proof_bytes` are accepted by design).
///
/// Recursive accumulation collapses every per-layer proof into a single
/// [`crate::recursive::RecursiveProof`] via
/// [`LayerChainFolder::fold_into_recursive_proof_full_scheme`], whose
/// `verify_final_scheme` decision binds [`LayerChainProof::commitment`]
/// — the chain's keccak commitment over every boundary value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerChainProof {
    pub layers: Vec<LayerProof>,
}

impl LayerChainProof {
    pub fn new(layers: Vec<LayerProof>) -> Self {
        Self { layers }
    }

    /// Serialize the full chain proof — claims, kind tags, and per-layer
    /// proof bytes — into a single self-delimited byte blob. Format
    /// (all big-endian):
    ///
    /// ```text
    ///   [u32 num_layers]
    ///   for each layer:
    ///     [u8  claim_tag]              (0=Execution, 1=BlockBinding,
    ///                                   2=Attestation, 3=Finality)
    ///     [claim payload]              variable per tag (see encode/decode)
    ///     [u8  kind_tag]               (0..=9 per LayerProofKind enum order)
    ///     [u32 proof_bytes_len][bytes]
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1024);
        out.extend_from_slice(&(self.layers.len() as u32).to_be_bytes());
        for layer in &self.layers {
            encode_claim(&mut out, &layer.claim);
            out.push(encode_kind(layer.kind));
            out.extend_from_slice(&(layer.proof_bytes.len() as u32).to_be_bytes());
            out.extend_from_slice(&layer.proof_bytes);
        }
        out
    }

    /// Deserialize a chain proof produced by [`Self::to_bytes`]. Strict:
    /// trailing bytes after a complete decode are an error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ChainDecodeError> {
        let mut r = ChainReader::new(bytes);
        let n = r.read_u32()? as usize;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let claim = decode_claim(&mut r)?;
            let kind_byte = r.read_u8()?;
            let kind = decode_kind(kind_byte)
                .ok_or(ChainDecodeError::InvalidKind(kind_byte))?;
            let len = r.read_u32()? as usize;
            let proof_bytes = r.take(len)?.to_vec();
            layers.push(LayerProof { claim, kind, proof_bytes });
        }
        let trailing = r.remaining();
        if trailing > 0 {
            return Err(ChainDecodeError::TrailingBytes(trailing));
        }
        Ok(LayerChainProof::new(layers))
    }

    /// View the proof envelope as a pure [`LayerChain`] of public claims.
    pub fn chain(&self) -> LayerChain {
        LayerChain {
            claims: self.layers.iter().map(|l| l.claim.clone()).collect(),
        }
    }

    /// Single-hash commitment over all four boundary values — equal to
    /// [`LayerChain::commitment`] on the inner chain.
    pub fn commitment(&self) -> Option<Hash32> {
        self.chain().commitment()
    }

    /// Verify chain consistency only — checks that adjacent claims share
    /// the right boundary value. Per-layer proof verification is **not**
    /// performed here; use [`Self::verify_with_layer_verifier`] (or
    /// [`LayerChainFolder::fold`] / `fold_into_recursive_proof_*_scheme`)
    /// for that.
    pub fn verify(&self) -> Result<(), ChainError> {
        self.chain().verify_consistency()
    }

    /// Total byte size of all per-layer proof payloads. Useful for
    /// reporting the on-wire cost of a `LayerChainProof` to users.
    /// Excludes the chain envelope's own framing overhead (claim values,
    /// kind tags) which is constant per layer.
    pub fn total_proof_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.proof_bytes.len()).sum()
    }

    /// Per-layer (kind, byte_count) accounting. Reference-only layers
    /// are included with byte_count = 0 so the caller can iterate the
    /// full chain shape. Order matches `self.layers`.
    pub fn byte_sizes(&self) -> Vec<(LayerProofKind, usize)> {
        self.layers
            .iter()
            .map(|l| (l.kind, l.proof_bytes.len()))
            .collect()
    }

    /// Human-readable summary suitable for logging or debugging.
    /// One line per layer: index, claim type, kind tag, byte count.
    /// Plus a footer with chain commitment (or "<malformed>" if the
    /// chain fails consistency) and total byte count.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "LayerChainProof ({} layers)\n",
            self.layers.len()
        ));
        for (i, layer) in self.layers.iter().enumerate() {
            let claim_kind = match &layer.claim {
                LayerClaim::Execution { .. } => "Execution",
                LayerClaim::BlockBinding { .. } => "BlockBinding",
                LayerClaim::Attestation { .. } => "Attestation",
                LayerClaim::Finality { .. } => "Finality",
            };
            out.push_str(&format!(
                "  layer {}: {:<14} kind={:<14} bytes={}\n",
                i,
                claim_kind,
                layer.kind.as_str(),
                layer.proof_bytes.len(),
            ));
        }
        let commitment_line = match self.commitment() {
            Some(c) => {
                let mut hex = String::with_capacity(64);
                for b in &c {
                    hex.push_str(&format!("{:02x}", b));
                }
                format!("  commitment: 0x{}\n", hex)
            }
            None => "  commitment: <malformed>\n".to_string(),
        };
        out.push_str(&format!(
            "  total proof bytes: {}\n",
            self.total_proof_bytes()
        ));
        out.push_str(&commitment_line);
        out
    }

    /// Verify the chain plus run a caller-supplied verifier against every
    /// layer carrying actual proof bytes. The closure receives the full
    /// `LayerProof` (claim + bytes), so a caller can dispatch on
    /// `claim` variant to pick the right per-layer verifier.
    ///
    /// Layers with empty `proof_bytes` are skipped (still reference-only —
    /// the chain commitment alone binds them). Once every layer has an
    /// AIR + serializable proof type, the closure runs for all of them
    /// and the chain becomes a single-statement-bound recursive root.
    pub fn verify_with_layer_verifier<F>(
        &self,
        verifier: F,
    ) -> Result<(), ProofChainError>
    where
        F: Fn(&LayerProof) -> Result<(), String>,
    {
        self.chain()
            .verify_consistency()
            .map_err(ProofChainError::Chain)?;
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.has_proof() {
                verifier(layer).map_err(|msg| ProofChainError::LayerProof {
                    at_index: i,
                    msg,
                })?;
            }
        }
        Ok(())
    }
}

/// Error type for [`LayerChainProof::verify_with_layer_verifier`]. Splits
/// "chain consistency broke" from "a layer's proof failed" so callers can
/// distinguish data-shape errors from cryptographic verification failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofChainError {
    /// The chain itself was malformed (boundary mismatch, wrong order, etc).
    Chain(ChainError),
    /// A specific layer's proof bytes failed verification.
    LayerProof {
        at_index: usize,
        msg: String,
    },
}

/// Result of running [`LayerChainFolder::fold`] over a [`LayerChainProof`].
///
/// Holds the chain's single-hash commitment plus a per-layer verification
/// status. This is the **public output** a downstream recursive folder
/// would commit to: the chain commitment binds every boundary value, and
/// the per-layer statuses witness that each AIR proof was either verified
/// or skipped (reference-only) by the appropriate per-AIR verifier.
///
/// # Status semantics
///
/// * `LayerStatus::ReferenceOnly` — the layer was tagged
///   [`LayerProofKind::ReferenceOnly`]; only its public claim values
///   were checked (via the chain consistency check).
/// * `LayerStatus::Verified { kind }` — the layer's proof bytes were
///   decoded and the per-AIR verifier accepted the proof.
/// * The fold short-circuits and returns `Err` on the first per-layer
///   verification failure, so a successful `FoldedLayerChain` implies
///   every wired layer's proof is cryptographically sound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldedLayerChain {
    /// The chain's single 32-byte commitment over all boundary values.
    pub commitment: Hash32,
    /// Per-layer verification status, one entry per `LayerChainProof::layers`.
    pub layer_statuses: Vec<LayerStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerStatus {
    ReferenceOnly,
    Verified { kind: LayerProofKind },
}

/// A recursive folder over a [`LayerChainProof`].
///
/// The folder takes a chain proof + a per-layer verifier closure (mirroring
/// [`LayerChainProof::verify_with_layer_verifier`]) and returns a
/// [`FoldedLayerChain`] capturing the chain commitment plus per-layer
/// verification statuses.
///
/// # What this folder does today
///
/// 1. Runs `LayerChain::verify_consistency` (boundary values agree).
/// 2. For every layer carrying real proof bytes, dispatches to a caller-
///    supplied closure that decodes via `ExecutionProof::from_bytes` and
///    verifies via the appropriate `verify_with_scheme` invocation.
/// 3. Returns a `FoldedLayerChain` recording each layer's status.
///
/// For cross-AIR cryptographic accumulation see
/// [`Self::fold_into_recursive_proof_scheme`] /
/// [`Self::fold_into_recursive_proof_full_scheme`].
pub struct LayerChainFolder;

impl LayerChainFolder {
    /// Fold a [`LayerChainProof`] into a [`FoldedLayerChain`]. The
    /// `verifier` closure is called for every layer carrying real proof
    /// bytes; reference-only layers are skipped.
    ///
    /// Returns `Err(ProofChainError::Chain(_))` if the chain is malformed
    /// or `Err(ProofChainError::LayerProof { .. })` on the first layer
    /// whose proof fails verification.
    pub fn fold<F>(
        proof: &LayerChainProof,
        verifier: F,
    ) -> Result<FoldedLayerChain, ProofChainError>
    where
        F: Fn(&LayerProof) -> Result<(), String>,
    {
        let chain = proof.chain();
        chain.verify_consistency().map_err(ProofChainError::Chain)?;
        let commitment = chain.commitment().ok_or_else(|| {
            // Should be unreachable since verify_consistency just passed,
            // but `commitment()` returns Option for safety; surface as a
            // chain error with the most informative variant available.
            ProofChainError::Chain(ChainError::Empty)
        })?;

        let mut statuses = Vec::with_capacity(proof.layers.len());
        for (i, layer) in proof.layers.iter().enumerate() {
            if layer.has_proof() {
                verifier(layer).map_err(|msg| ProofChainError::LayerProof {
                    at_index: i,
                    msg,
                })?;
                statuses.push(LayerStatus::Verified { kind: layer.kind });
            } else {
                statuses.push(LayerStatus::ReferenceOnly);
            }
        }

        Ok(FoldedLayerChain {
            commitment,
            layer_statuses: statuses,
        })
    }

    /// Cross-AIR cryptographic fold: collapse a [`LayerChainProof`] into a
    /// single [`crate::recursive::RecursiveProof`] by sequentially folding
    /// each provable layer's `ExecutionProof` through the existing IVC
    /// accumulator. Reference-only layers are skipped (their public values
    /// are already bound by the chain commitment).
    ///
    /// Each provable layer is wrapped in a synthetic `ChunkProof` whose
    /// state hashes are derived from the chain commitment so successive
    /// layers chain correctly:
    ///
    /// ```text
    ///   H_0 = chain_commitment
    ///   H_{i+1} = keccak256(H_i ‖ kind_tag ‖ layer_index_BE)
    ///   chunk[i].initial = H_i ; chunk[i].final = H_{i+1}
    /// ```
    ///
    /// The returned `RecursiveProof` carries a single `(L_acc, R_acc)`
    /// pairing accumulator that aggregates every layer's KZG base opening.
    /// `verify_final_scheme` produces a single bool from this accumulator.
    ///
    /// # Soundness scope
    ///
    /// Consistent with the existing single-AIR multi-chunk fold, this
    /// accumulator only folds the **base opening** pairing of each layer
    /// (`column_commitments + quotient_commitments` opened at `z`). The
    /// per-layer auxiliary openings (shifted, logup, bitwise, perm,
    /// reg_perm, frame_perm) and the per-AIR constraint arithmetic
    /// (`evaluate_at_point`) are **not** included in the accumulator;
    /// callers needing full per-layer verification use [`Self::fold`] or
    /// [`Self::fold_into_recursive_proof_full_scheme`] (which aggregates
    /// every opening pairing AND the constraint identity scalar).
    pub fn fold_into_recursive_proof_scheme(
        proof: &LayerChainProof,
        scheme: &dyn crate::scheme::CommitmentScheme,
        curve: crate::field::CurveType,
    ) -> Result<crate::recursive::RecursiveProof, ProofChainError> {
        use crate::keccak::keccak256;
        use crate::prover::ChunkProof;
        use crate::prover::ExecutionProof;
        use crate::recursive::{begin_chunk_scheme, fold_chunks_scheme};

        // 1. Chain consistency must pass (same precondition as `fold`).
        let chain = proof.chain();
        chain.verify_consistency().map_err(ProofChainError::Chain)?;
        let chain_commitment = chain
            .commitment()
            .ok_or(ProofChainError::Chain(ChainError::Empty))?;

        // 2. Collect provable layers (those with real proof bytes).
        let provable: Vec<(usize, &LayerProof)> = proof
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.has_proof())
            .collect();
        if provable.is_empty() {
            return Err(ProofChainError::LayerProof {
                at_index: 0,
                msg: "fold_into_recursive_proof_scheme: chain has no provable layers"
                    .to_string(),
            });
        }

        // 3. Helper: derive next state hash from prev hash + layer kind +
        //    layer index. Domain-separation byte 0xF1 distinguishes this
        //    derivation from any other keccak-based hash chain.
        let derive_next = |prev: &[u8; 32], layer_idx: usize, kind: LayerProofKind| -> [u8; 32] {
            let mut buf = Vec::with_capacity(32 + 1 + 1 + 8);
            buf.extend_from_slice(prev);
            buf.push(0xF1);
            buf.push(encode_kind(kind));
            buf.extend_from_slice(&(layer_idx as u64).to_be_bytes());
            keccak256(&buf)
        };

        // 4. Decode each layer's ExecutionProof and wrap in a synthetic
        //    ChunkProof. Sequential state hashes guarantee the fold's
        //    state-chain check passes.
        let mut state_hash: [u8; 32] = chain_commitment;
        let mut chunks: Vec<ChunkProof> = Vec::with_capacity(provable.len());
        for (chunk_idx, (layer_idx, layer)) in provable.iter().enumerate() {
            let exec_proof = ExecutionProof::from_bytes(&layer.proof_bytes).map_err(|e| {
                ProofChainError::LayerProof {
                    at_index: *layer_idx,
                    msg: format!(
                        "fold_into_recursive_proof_scheme: ExecutionProof::from_bytes failed: {:?}",
                        e
                    ),
                }
            })?;
            let next_hash = derive_next(&state_hash, *layer_idx, layer.kind);
            chunks.push(ChunkProof {
                execution_proof: exec_proof,
                initial_state_hash: state_hash,
                final_state_hash: next_hash,
                chunk_index: chunk_idx as u64,
            });
            state_hash = next_hash;
        }

        // 5. Begin with the first chunk, fold the rest sequentially.
        let mut chunks_iter = chunks.into_iter();
        let first = chunks_iter
            .next()
            .expect("provable.is_empty() check above guarantees at least one");
        let mut acc = begin_chunk_scheme(first, scheme, curve);
        for next in chunks_iter {
            let layer_idx_for_err = next.chunk_index as usize;
            acc = fold_chunks_scheme(&acc, next, scheme, curve).map_err(|msg| {
                ProofChainError::LayerProof {
                    at_index: layer_idx_for_err,
                    msg: format!("fold_chunks_scheme failed: {msg}"),
                }
            })?;
        }
        Ok(acc)
    }

    /// Full-fold cross-AIR cryptographic fold. Same shape as
    /// [`Self::fold_into_recursive_proof_scheme`] but uses
    /// [`crate::recursive::begin_chunk_full_scheme`] /
    /// [`crate::recursive::fold_chunks_full_scheme`] which aggregate
    /// every per-chunk KZG opening pairing (main + shifted + logup +
    /// bitwise + perm + reg_perm + frame_perm and their shifted
    /// variants) AND the per-AIR constraint identity scalar
    /// `c_check = Q(z)·Z_H(z) - C(z)`. The accumulated `(L_acc, R_acc)`
    /// + `scalar_acc` therefore certify every layer's full proof in one
    /// `verify_final_scheme` call.
    ///
    /// Each layer's constraint system is supplied via the `cs_dispatch`
    /// closure, mirroring the per-layer-verifier pattern in
    /// [`LayerChainProof::verify_with_layer_verifier`].
    pub fn fold_into_recursive_proof_full_scheme<F>(
        proof: &LayerChainProof,
        scheme: &dyn crate::scheme::CommitmentScheme,
        curve: crate::field::CurveType,
        cs_dispatch: F,
    ) -> Result<crate::recursive::RecursiveProof, ProofChainError>
    where
        F: Fn(LayerProofKind) -> Box<dyn crate::vm_constraints::VmConstraintSystem>,
    {
        use crate::keccak::keccak256;
        use crate::prover::{ChunkProof, ExecutionProof};
        use crate::recursive::{begin_chunk_full_scheme, fold_chunks_full_scheme};

        let chain = proof.chain();
        chain.verify_consistency().map_err(ProofChainError::Chain)?;
        let chain_commitment = chain
            .commitment()
            .ok_or(ProofChainError::Chain(ChainError::Empty))?;

        let provable: Vec<(usize, &LayerProof)> = proof
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.has_proof())
            .collect();
        if provable.is_empty() {
            return Err(ProofChainError::LayerProof {
                at_index: 0,
                msg: "fold_into_recursive_proof_full_scheme: no provable layers".to_string(),
            });
        }

        let derive_next = |prev: &[u8; 32], layer_idx: usize, kind: LayerProofKind| -> [u8; 32] {
            let mut buf = Vec::with_capacity(32 + 1 + 1 + 8);
            buf.extend_from_slice(prev);
            buf.push(0xF1);
            buf.push(encode_kind(kind));
            buf.extend_from_slice(&(layer_idx as u64).to_be_bytes());
            keccak256(&buf)
        };

        let mut state_hash: [u8; 32] = chain_commitment;
        let mut chunks: Vec<(ChunkProof, LayerProofKind, usize)> =
            Vec::with_capacity(provable.len());
        for (chunk_idx, (layer_idx, layer)) in provable.iter().enumerate() {
            let exec_proof = ExecutionProof::from_bytes(&layer.proof_bytes).map_err(|e| {
                ProofChainError::LayerProof {
                    at_index: *layer_idx,
                    msg: format!(
                        "fold_into_recursive_proof_full_scheme: ExecutionProof::from_bytes failed: {:?}",
                        e
                    ),
                }
            })?;
            let next_hash = derive_next(&state_hash, *layer_idx, layer.kind);
            chunks.push((
                ChunkProof {
                    execution_proof: exec_proof,
                    initial_state_hash: state_hash,
                    final_state_hash: next_hash,
                    chunk_index: chunk_idx as u64,
                },
                layer.kind,
                *layer_idx,
            ));
            state_hash = next_hash;
        }

        let mut chunks_iter = chunks.into_iter();
        let (first_chunk, first_kind, _first_layer_idx) = chunks_iter
            .next()
            .expect("provable.is_empty() check above guarantees at least one");
        let first_cs = cs_dispatch(first_kind);
        let mut acc = begin_chunk_full_scheme(first_chunk, scheme, curve, first_cs.as_ref());
        for (chunk, kind, layer_idx_for_err) in chunks_iter {
            let cs = cs_dispatch(kind);
            acc = fold_chunks_full_scheme(&acc, chunk, scheme, curve, cs.as_ref())
                .map_err(|msg| ProofChainError::LayerProof {
                    at_index: layer_idx_for_err,
                    msg: format!("fold_chunks_full_scheme failed: {msg}"),
                })?;
        }
        Ok(acc)
    }
}

/// Errors from [`LayerChainProof::from_bytes`] decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainDecodeError {
    /// Buffer truncated mid-field.
    Truncated { wanted: usize, available: usize },
    /// Length prefix declares more bytes than remain in the buffer.
    LengthOverflow,
    /// Unknown LayerClaim tag byte.
    InvalidClaimTag(u8),
    /// Unknown LayerProofKind tag byte.
    InvalidKind(u8),
    /// Invalid Gwei encoding or other field-level decode error.
    InvalidField(&'static str),
    /// Trailing bytes after a complete decode.
    TrailingBytes(usize),
}

// ── Claim / kind encoding helpers ─────────────────────────────────────

fn encode_claim(out: &mut Vec<u8>, claim: &LayerClaim) {
    match claim {
        LayerClaim::Execution { block_hash } => {
            out.push(0);
            out.extend_from_slice(block_hash);
        }
        LayerClaim::BlockBinding { block_hash, beacon_block_root } => {
            out.push(1);
            out.extend_from_slice(block_hash);
            out.extend_from_slice(beacon_block_root);
        }
        LayerClaim::Attestation { beacon_block_root, attestation_data_root, num_attesters } => {
            out.push(2);
            out.extend_from_slice(beacon_block_root);
            out.extend_from_slice(attestation_data_root);
            out.extend_from_slice(&num_attesters.to_be_bytes());
        }
        LayerClaim::Finality { attestation_data_root, finalized_root, total_effective_balance_gwei } => {
            out.push(3);
            out.extend_from_slice(attestation_data_root);
            out.extend_from_slice(finalized_root);
            out.extend_from_slice(&total_effective_balance_gwei.to_be_bytes());
        }
    }
}

fn decode_claim(r: &mut ChainReader) -> Result<LayerClaim, ChainDecodeError> {
    let tag = r.read_u8()?;
    match tag {
        0 => Ok(LayerClaim::Execution {
            block_hash: r.read_hash()?,
        }),
        1 => Ok(LayerClaim::BlockBinding {
            block_hash: r.read_hash()?,
            beacon_block_root: r.read_hash()?,
        }),
        2 => Ok(LayerClaim::Attestation {
            beacon_block_root: r.read_hash()?,
            attestation_data_root: r.read_hash()?,
            num_attesters: r.read_u64()?,
        }),
        3 => Ok(LayerClaim::Finality {
            attestation_data_root: r.read_hash()?,
            finalized_root: r.read_hash()?,
            total_effective_balance_gwei: r.read_u64()?,
        }),
        other => Err(ChainDecodeError::InvalidClaimTag(other)),
    }
}

fn encode_kind(kind: LayerProofKind) -> u8 {
    match kind {
        LayerProofKind::ReferenceOnly => 0,
        LayerProofKind::NonnativeFp => 1,
        LayerProofKind::Keccak => 2,
        LayerProofKind::Sha256 => 3,
        LayerProofKind::Ssz => 4,
        LayerProofKind::Mpt => 5,
        LayerProofKind::EvmExp => 6,
        LayerProofKind::VmRiscv => 7,
        LayerProofKind::VmEvm => 8,
        LayerProofKind::VmSbf => 9,
        LayerProofKind::BlsSig => 10,
        LayerProofKind::Finality => 11,
    }
}

fn decode_kind(byte: u8) -> Option<LayerProofKind> {
    match byte {
        0 => Some(LayerProofKind::ReferenceOnly),
        1 => Some(LayerProofKind::NonnativeFp),
        2 => Some(LayerProofKind::Keccak),
        3 => Some(LayerProofKind::Sha256),
        4 => Some(LayerProofKind::Ssz),
        5 => Some(LayerProofKind::Mpt),
        6 => Some(LayerProofKind::EvmExp),
        7 => Some(LayerProofKind::VmRiscv),
        8 => Some(LayerProofKind::VmEvm),
        9 => Some(LayerProofKind::VmSbf),
        10 => Some(LayerProofKind::BlsSig),
        11 => Some(LayerProofKind::Finality),
        _ => None,
    }
}

struct ChainReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ChainReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ChainDecodeError> {
        if self.pos + n > self.buf.len() {
            return Err(ChainDecodeError::Truncated {
                wanted: n,
                available: self.buf.len() - self.pos,
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8, ChainDecodeError> {
        Ok(self.take(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, ChainDecodeError> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, ChainDecodeError> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_be_bytes(a))
    }

    fn read_hash(&mut self) -> Result<Hash32, ChainDecodeError> {
        let s = self.take(32)?;
        let mut h = [0u8; 32];
        h.copy_from_slice(s);
        Ok(h)
    }
}

/// The terminal public output of a fully-verified layer chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalOutput {
    /// keccak256 of the execution-layer block header.
    pub block_hash: Hash32,
    /// SSZ hash-tree-root of the beacon block referencing the execution payload.
    pub beacon_block_root: Hash32,
    /// FFG-finalized checkpoint root.
    pub finalized_root: Hash32,
    /// Total effective balance (Gwei) backing finalization.
    pub total_effective_balance_gwei: Gwei,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(byte: u8) -> Hash32 {
        let mut x = [0u8; 32];
        x[0] = byte;
        x
    }

    fn build_valid_chain() -> LayerChain {
        let block_hash = h(0xBB);
        let beacon_root = h(0xCC);
        let att_root = h(0xDD);
        let final_root = h(0xCC); // same as beacon — finalizing this very block

        let mut chain = LayerChain::new();
        chain.push(LayerClaim::Execution { block_hash });
        chain.push(LayerClaim::BlockBinding {
            block_hash,
            beacon_block_root: beacon_root,
        });
        chain.push(LayerClaim::Attestation {
            beacon_block_root: beacon_root,
            attestation_data_root: att_root,
            num_attesters: 70,
        });
        chain.push(LayerClaim::Finality {
            attestation_data_root: att_root,
            finalized_root: final_root,
            total_effective_balance_gwei: 70 * 32_000_000_000,
        });
        chain
    }

    #[test]
    fn empty_chain_fails() {
        let chain = LayerChain::new();
        assert_eq!(chain.verify_consistency(), Err(ChainError::Empty));
    }

    #[test]
    fn valid_chain_verifies_and_extracts_output() {
        let chain = build_valid_chain();
        chain.verify_consistency().unwrap();
        let out = chain.final_output().unwrap();
        assert_eq!(out.block_hash, h(0xBB));
        assert_eq!(out.beacon_block_root, h(0xCC));
        assert_eq!(out.finalized_root, h(0xCC));
        assert_eq!(out.total_effective_balance_gwei, 70 * 32_000_000_000);
    }

    #[test]
    fn first_must_be_execution() {
        let mut chain = LayerChain::new();
        chain.push(LayerClaim::BlockBinding {
            block_hash: h(0),
            beacon_block_root: h(1),
        });
        assert_eq!(
            chain.verify_consistency(),
            Err(ChainError::FirstNotExecution)
        );
    }

    #[test]
    fn last_must_be_finality() {
        let mut chain = LayerChain::new();
        chain.push(LayerClaim::Execution { block_hash: h(0xBB) });
        // Stop at BlockBinding (no Finality).
        chain.push(LayerClaim::BlockBinding {
            block_hash: h(0xBB),
            beacon_block_root: h(0xCC),
        });
        assert_eq!(
            chain.verify_consistency(),
            Err(ChainError::LastNotFinality)
        );
    }

    #[test]
    fn block_hash_boundary_mismatch_caught() {
        let mut chain = build_valid_chain();
        // Tamper the BlockBinding's block_hash input.
        if let LayerClaim::BlockBinding { block_hash, .. } = &mut chain.claims[1] {
            *block_hash = h(0xFF);
        }
        assert_eq!(
            chain.verify_consistency(),
            Err(ChainError::BoundaryMismatch { at_index: 1, what: "block_hash" })
        );
    }

    #[test]
    fn beacon_root_boundary_mismatch_caught() {
        let mut chain = build_valid_chain();
        if let LayerClaim::Attestation { beacon_block_root, .. } = &mut chain.claims[2] {
            *beacon_block_root = h(0xFF);
        }
        assert_eq!(
            chain.verify_consistency(),
            Err(ChainError::BoundaryMismatch { at_index: 2, what: "beacon_block_root" })
        );
    }

    #[test]
    fn attestation_root_boundary_mismatch_caught() {
        let mut chain = build_valid_chain();
        if let LayerClaim::Finality { attestation_data_root, .. } = &mut chain.claims[3] {
            *attestation_data_root = h(0xFF);
        }
        assert_eq!(
            chain.verify_consistency(),
            Err(ChainError::BoundaryMismatch {
                at_index: 3,
                what: "attestation_data_root"
            })
        );
    }

    #[test]
    fn invalid_transition_caught() {
        // Execution → Attestation skipping BlockBinding.
        let mut chain = LayerChain::new();
        chain.push(LayerClaim::Execution { block_hash: h(0xBB) });
        chain.push(LayerClaim::Attestation {
            beacon_block_root: h(0xCC),
            attestation_data_root: h(0xDD),
            num_attesters: 1,
        });
        chain.push(LayerClaim::Finality {
            attestation_data_root: h(0xDD),
            finalized_root: h(0xEE),
            total_effective_balance_gwei: 1,
        });
        assert_eq!(
            chain.verify_consistency(),
            Err(ChainError::InvalidTransition { from_index: 0 })
        );
    }

    #[test]
    fn final_output_returns_none_on_invalid_chain() {
        let mut chain = build_valid_chain();
        if let LayerClaim::BlockBinding { block_hash, .. } = &mut chain.claims[1] {
            *block_hash = h(0xFF);
        }
        assert!(chain.final_output().is_none());
    }

    fn build_valid_boundaries() -> ChainBoundaries {
        ChainBoundaries {
            block_hash: h(0xBB),
            beacon_block_root: h(0xCC),
            attestation_data_root: h(0xDD),
            num_attesters: 70,
            finalized_root: h(0xCC),
            total_effective_balance_gwei: 70 * 32_000_000_000,
        }
    }

    #[test]
    fn from_boundaries_matches_hand_built() {
        let by_hand = build_valid_chain();
        let by_builder = LayerChain::from_boundaries(&build_valid_boundaries());
        assert_eq!(by_hand.claims, by_builder.claims);
        by_builder.verify_consistency().unwrap();
    }

    #[test]
    fn commitment_is_deterministic() {
        let chain = build_valid_chain();
        let c1 = chain.commitment().expect("valid chain");
        let c2 = chain.commitment().expect("valid chain");
        assert_eq!(c1, c2, "commitment must be deterministic");
    }

    #[test]
    fn commitment_returns_none_for_invalid_chain() {
        let mut chain = build_valid_chain();
        if let LayerClaim::BlockBinding { block_hash, .. } = &mut chain.claims[1] {
            *block_hash = h(0xFF);
        }
        assert!(chain.commitment().is_none(), "invalid chain has no commitment");
    }

    #[test]
    fn commitment_diverges_on_boundary_change() {
        let original = build_valid_chain().commitment().unwrap();
        let mut tweaked = build_valid_boundaries();
        tweaked.beacon_block_root = h(0xEE);
        let tweaked_commit = LayerChain::from_boundaries(&tweaked).commitment().unwrap();
        assert_ne!(original, tweaked_commit, "commitment must depend on every boundary");
    }

    #[test]
    fn final_output_commitment_matches_chain_commitment() {
        // FinalOutput::commitment_with(att_root, num_atts) must agree with
        // LayerChain::commitment() for the chain that produced it.
        let b = build_valid_boundaries();
        let chain = LayerChain::from_boundaries(&b);
        let chain_commit = chain.commitment().unwrap();
        let final_out = chain.final_output().unwrap();
        let final_commit = final_out.commitment_with(
            &b.attestation_data_root,
            b.num_attesters,
        );
        assert_eq!(
            chain_commit, final_commit,
            "FinalOutput::commitment_with must agree with chain.commitment()",
        );
    }

    #[test]
    fn commitment_distinguishes_num_attesters() {
        let mut b = build_valid_boundaries();
        let c1 = LayerChain::from_boundaries(&b).commitment().unwrap();
        b.num_attesters += 1;
        let c2 = LayerChain::from_boundaries(&b).commitment().unwrap();
        assert_ne!(c1, c2, "num_attesters must influence commitment");
    }

    #[test]
    fn commitment_distinguishes_finality_weight() {
        let mut b = build_valid_boundaries();
        let c1 = LayerChain::from_boundaries(&b).commitment().unwrap();
        b.total_effective_balance_gwei += 1;
        let c2 = LayerChain::from_boundaries(&b).commitment().unwrap();
        assert_ne!(c1, c2, "finality weight must influence commitment");
    }

    fn build_valid_proof_chain() -> LayerChainProof {
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .map(LayerProof::reference_only)
            .collect();
        LayerChainProof::new(layers)
    }

    #[test]
    fn layer_proof_reference_only_has_no_bytes() {
        let claim = LayerClaim::Execution { block_hash: h(0xBB) };
        let p = LayerProof::reference_only(claim.clone());
        assert!(!p.has_proof());
        assert!(p.proof_bytes.is_empty());
        assert_eq!(p.claim, claim);
    }

    #[test]
    fn layer_proof_with_proof_carries_bytes_and_kind() {
        let claim = LayerClaim::Execution { block_hash: h(0xBB) };
        let p = LayerProof::with_proof(claim, LayerProofKind::Sha256, vec![1, 2, 3]);
        assert!(p.has_proof());
        assert_eq!(p.proof_bytes, vec![1, 2, 3]);
        assert_eq!(p.kind, LayerProofKind::Sha256);
    }

    #[test]
    fn reference_only_layer_has_reference_only_kind() {
        let claim = LayerClaim::Execution { block_hash: h(0xBB) };
        let p = LayerProof::reference_only(claim);
        assert!(!p.has_proof());
        assert_eq!(p.kind, LayerProofKind::ReferenceOnly);
        assert_eq!(p.kind.as_str(), "reference-only");
    }

    #[test]
    fn layer_proof_kind_str_round_trip() {
        for kind in [
            LayerProofKind::ReferenceOnly,
            LayerProofKind::NonnativeFp,
            LayerProofKind::Keccak,
            LayerProofKind::Sha256,
            LayerProofKind::Ssz,
            LayerProofKind::Mpt,
            LayerProofKind::EvmExp,
            LayerProofKind::VmRiscv,
            LayerProofKind::VmEvm,
            LayerProofKind::VmSbf,
        ] {
            let s = kind.as_str();
            assert!(!s.is_empty(), "kind str must be non-empty");
        }
    }

    #[test]
    fn empty_bytes_with_non_reference_kind_still_has_no_proof() {
        // Defensive: even if a caller tags a layer Sha256 but supplies no
        // bytes, has_proof() returns false (so the verifier closure is
        // skipped and no spurious "verify empty bytes" call happens).
        let claim = LayerClaim::Execution { block_hash: h(0xBB) };
        let p = LayerProof::with_proof(claim, LayerProofKind::Sha256, Vec::new());
        assert!(!p.has_proof());
    }

    #[test]
    fn layer_chain_proof_commitment_matches_inner_chain() {
        let cp = build_valid_proof_chain();
        let inner_commit = cp.chain().commitment().unwrap();
        let outer_commit = cp.commitment().unwrap();
        assert_eq!(inner_commit, outer_commit);
    }

    #[test]
    fn layer_chain_proof_verify_succeeds_on_valid_chain() {
        let cp = build_valid_proof_chain();
        cp.verify().expect("valid reference-only chain must verify");
    }

    #[test]
    fn layer_chain_proof_verify_fails_on_inconsistent_chain() {
        let mut cp = build_valid_proof_chain();
        // Tamper the BlockBinding's block_hash input so the chain breaks.
        if let LayerClaim::BlockBinding { block_hash, .. } = &mut cp.layers[1].claim {
            *block_hash = h(0xFF);
        }
        assert_eq!(
            cp.verify(),
            Err(ChainError::BoundaryMismatch { at_index: 1, what: "block_hash" })
        );
    }

    #[test]
    fn layer_chain_proof_verify_with_layer_verifier_skips_reference_only() {
        let cp = build_valid_proof_chain();
        // Verifier should never run on reference-only layers — flip a flag
        // to detect any unexpected calls.
        use std::cell::Cell;
        let calls = Cell::new(0);
        let result = cp.verify_with_layer_verifier(|_layer| {
            calls.set(calls.get() + 1);
            Ok(())
        });
        assert_eq!(result, Ok(()));
        assert_eq!(calls.get(), 0, "no layer carried proof bytes; verifier must not run");
    }

    #[test]
    fn layer_chain_proof_verify_with_layer_verifier_runs_on_proven_layers() {
        let chain = build_valid_chain();
        // Attach proof bytes to layer 2 (Attestation).
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| {
                if i == 2 {
                    LayerProof::with_proof(
                        claim,
                        LayerProofKind::NonnativeFp,
                        vec![0xDE, 0xAD, 0xBE, 0xEF],
                    )
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let cp = LayerChainProof::new(layers);

        use std::cell::Cell;
        let bytes_seen: Cell<Option<Vec<u8>>> = Cell::new(None);
        let result = cp.verify_with_layer_verifier(|layer| {
            bytes_seen.set(Some(layer.proof_bytes.clone()));
            // Claim-aware dispatch demonstration.
            if matches!(layer.claim, LayerClaim::Attestation { .. })
                && layer.proof_bytes == vec![0xDE, 0xAD, 0xBE, 0xEF]
            {
                Ok(())
            } else {
                Err(format!("unexpected layer / bytes: {:?}", layer))
            }
        });
        assert_eq!(result, Ok(()));
        assert_eq!(bytes_seen.into_inner(), Some(vec![0xDE, 0xAD, 0xBE, 0xEF]));
    }

    #[test]
    fn layer_chain_proof_verify_with_layer_verifier_propagates_failure() {
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| {
                if i == 1 {
                    LayerProof::with_proof(claim, LayerProofKind::Sha256, vec![0xBA, 0xD0])
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let cp = LayerChainProof::new(layers);

        let result = cp.verify_with_layer_verifier(|_| {
            Err("simulated proof failure".to_string())
        });
        assert_eq!(
            result,
            Err(ProofChainError::LayerProof {
                at_index: 1,
                msg: "simulated proof failure".to_string(),
            })
        );
    }

    #[test]
    fn layer_chain_proof_verify_with_layer_verifier_propagates_chain_error() {
        // Even with a permissive verifier, an inconsistent chain fails.
        let mut cp = build_valid_proof_chain();
        if let LayerClaim::BlockBinding { block_hash, .. } = &mut cp.layers[1].claim {
            *block_hash = h(0xFF);
        }
        let result = cp.verify_with_layer_verifier(|_| Ok(()));
        assert_eq!(
            result,
            Err(ProofChainError::Chain(ChainError::BoundaryMismatch {
                at_index: 1,
                what: "block_hash",
            }))
        );
    }

    #[test]
    fn layer_chain_proof_serializes_round_trip() {
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| match i {
                0 => LayerProof::with_proof(claim, LayerProofKind::VmEvm, vec![1, 2, 3]),
                1 => LayerProof::with_proof(claim, LayerProofKind::Keccak, vec![10; 100]),
                2 => LayerProof::with_proof(claim, LayerProofKind::NonnativeFp, vec![0xCA, 0xFE]),
                3 => LayerProof::with_proof(claim, LayerProofKind::Ssz, vec![0xAB, 0xCD]),
                _ => unreachable!(),
            })
            .collect();
        let cp = LayerChainProof::new(layers);

        let bytes = cp.to_bytes();
        let decoded = LayerChainProof::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.layers.len(), cp.layers.len());
        for (a, b) in decoded.layers.iter().zip(cp.layers.iter()) {
            assert_eq!(a.claim, b.claim);
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.proof_bytes, b.proof_bytes);
        }
        assert_eq!(decoded.commitment(), cp.commitment());
        // Canonical encoding.
        assert_eq!(decoded.to_bytes(), bytes);
    }

    #[test]
    fn layer_chain_proof_serializes_reference_only_layers() {
        let cp = build_valid_proof_chain();
        let bytes = cp.to_bytes();
        let decoded = LayerChainProof::from_bytes(&bytes).expect("decode");
        for (a, b) in decoded.layers.iter().zip(cp.layers.iter()) {
            assert_eq!(a.claim, b.claim);
            assert_eq!(a.kind, b.kind);
            assert!(a.proof_bytes.is_empty());
            assert!(b.proof_bytes.is_empty());
        }
    }

    #[test]
    fn layer_chain_proof_decode_truncated_fails() {
        let cp = build_valid_proof_chain();
        let bytes = cp.to_bytes();
        for trunc in &[0usize, 1, 5, 20, bytes.len() - 1] {
            let r = LayerChainProof::from_bytes(&bytes[..*trunc]);
            assert!(r.is_err(), "truncated to {} bytes must error", trunc);
        }
    }

    #[test]
    fn layer_chain_proof_decode_trailing_bytes_fails() {
        let cp = build_valid_proof_chain();
        let mut bytes = cp.to_bytes();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        assert_eq!(
            LayerChainProof::from_bytes(&bytes).err(),
            Some(ChainDecodeError::TrailingBytes(2)),
        );
    }

    #[test]
    fn layer_chain_proof_decode_invalid_claim_tag_fails() {
        let cp = build_valid_proof_chain();
        let mut bytes = cp.to_bytes();
        // Byte 4 is the first claim's tag (after 4-byte num_layers prefix).
        bytes[4] = 0x99;
        assert_eq!(
            LayerChainProof::from_bytes(&bytes).err(),
            Some(ChainDecodeError::InvalidClaimTag(0x99)),
        );
    }

    #[test]
    fn layer_chain_proof_decode_invalid_kind_tag_fails() {
        let cp = build_valid_proof_chain();
        let bytes = cp.to_bytes();
        // The first layer's kind byte sits after num_layers (4) +
        // claim_tag (1) + Execution claim payload (32 bytes for
        // block_hash) = byte 37.
        let mut bad = bytes.clone();
        bad[37] = 99;
        assert_eq!(
            LayerChainProof::from_bytes(&bad).err(),
            Some(ChainDecodeError::InvalidKind(99)),
        );
    }

    #[test]
    fn total_proof_bytes_sums_across_layers() {
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| match i {
                0 => LayerProof::with_proof(claim, LayerProofKind::VmEvm, vec![1u8; 100]),
                1 => LayerProof::with_proof(claim, LayerProofKind::Sha256, vec![2u8; 250]),
                _ => LayerProof::reference_only(claim),
            })
            .collect();
        let cp = LayerChainProof::new(layers);
        assert_eq!(cp.total_proof_bytes(), 350);
    }

    #[test]
    fn byte_sizes_includes_reference_only_layers() {
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| match i {
                1 => LayerProof::with_proof(claim, LayerProofKind::Sha256, vec![0u8; 42]),
                _ => LayerProof::reference_only(claim),
            })
            .collect();
        let cp = LayerChainProof::new(layers);
        let sizes = cp.byte_sizes();
        assert_eq!(sizes.len(), 4);
        assert_eq!(sizes[0], (LayerProofKind::ReferenceOnly, 0));
        assert_eq!(sizes[1], (LayerProofKind::Sha256, 42));
        assert_eq!(sizes[2], (LayerProofKind::ReferenceOnly, 0));
        assert_eq!(sizes[3], (LayerProofKind::ReferenceOnly, 0));
    }

    #[test]
    fn summary_includes_layer_lines_total_and_commitment() {
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| match i {
                1 => LayerProof::with_proof(claim, LayerProofKind::Sha256, vec![0u8; 17]),
                _ => LayerProof::reference_only(claim),
            })
            .collect();
        let cp = LayerChainProof::new(layers);
        let s = cp.summary();
        assert!(s.contains("LayerChainProof (4 layers)"));
        assert!(s.contains("Execution"));
        assert!(s.contains("BlockBinding"));
        assert!(s.contains("Attestation"));
        assert!(s.contains("Finality"));
        assert!(s.contains("kind=sha256"));
        assert!(s.contains("bytes=17"));
        assert!(s.contains("total proof bytes: 17"));
        assert!(s.contains("commitment: 0x"));
        // The commitment hex should be 64 chars (32 bytes).
        let commit_line = s.lines().find(|l| l.contains("commitment:")).unwrap();
        let hex_part = commit_line.trim_start().trim_start_matches("commitment: 0x");
        assert_eq!(hex_part.len(), 64);
    }

    #[test]
    fn summary_marks_malformed_chain() {
        let mut cp = build_valid_proof_chain();
        if let LayerClaim::BlockBinding { block_hash, .. } = &mut cp.layers[1].claim {
            *block_hash = h(0xFF);
        }
        let s = cp.summary();
        assert!(s.contains("commitment: <malformed>"));
    }

    #[test]
    fn folder_returns_reference_only_statuses_on_valid_unproven_chain() {
        let cp = build_valid_proof_chain();
        let folded = LayerChainFolder::fold(&cp, |_| Ok(())).expect("valid");
        assert_eq!(folded.commitment, cp.commitment().unwrap());
        assert_eq!(folded.layer_statuses.len(), 4);
        for status in &folded.layer_statuses {
            assert!(matches!(status, LayerStatus::ReferenceOnly));
        }
    }

    #[test]
    fn fold_into_recursive_proof_rejects_chain_with_no_provable_layers() {
        // All-reference-only chain: cross-AIR fold has nothing to do, must
        // return a LayerProof error rather than producing a zero-depth
        // recursive proof (which `verify_final_scheme` would reject anyway).
        use crate::field::CurveType;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let cp = build_valid_proof_chain();
        let result = LayerChainFolder::fold_into_recursive_proof_scheme(
            &cp,
            &scheme,
            CurveType::Bls48581,
        );
        assert!(matches!(
            result,
            Err(ProofChainError::LayerProof { msg, .. })
                if msg.contains("no provable layers")
        ));
    }

    #[test]
    fn fold_into_recursive_proof_rejects_malformed_chain() {
        // Chain consistency must still be the gate — a chain with mismatched
        // boundaries can't be folded even if it has provable layers.
        use crate::field::CurveType;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let mut cp = build_valid_proof_chain();
        // Tamper with layer 1's claim so adjacency check fails.
        if let LayerClaim::BlockBinding { block_hash, .. } = &mut cp.layers[1].claim {
            *block_hash = h(0xFF);
        }
        // Attach a (garbage but non-empty) proof so the no-provable-layers
        // path doesn't fire first.
        cp.layers[0] = LayerProof::with_proof(
            cp.layers[0].claim.clone(),
            LayerProofKind::Keccak,
            vec![0x00, 0x01, 0x02],
        );
        let result = LayerChainFolder::fold_into_recursive_proof_scheme(
            &cp,
            &scheme,
            CurveType::Bls48581,
        );
        assert!(matches!(result, Err(ProofChainError::Chain(_))));
    }

    #[test]
    fn fold_into_recursive_proof_rejects_garbled_proof_bytes() {
        // A chain with consistent boundaries but a layer whose proof_bytes
        // can't be decoded by ExecutionProof::from_bytes must surface the
        // decode error as a LayerProof error at the right layer index.
        use crate::field::CurveType;
        use crate::scheme::bls48581_scheme::Bls48581Scheme;
        use crate::scheme::CommitmentScheme;
        let scheme = Bls48581Scheme::new();
        scheme.init();
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| {
                if i == 1 {
                    // Garbled (non-empty) proof bytes → decode failure.
                    LayerProof::with_proof(claim, LayerProofKind::Keccak, vec![0xFF; 8])
                } else {
                    LayerProof::reference_only(claim)
                }
            })
            .collect();
        let cp = LayerChainProof::new(layers);
        let result = LayerChainFolder::fold_into_recursive_proof_scheme(
            &cp,
            &scheme,
            CurveType::Bls48581,
        );
        match result {
            Err(ProofChainError::LayerProof { at_index, msg }) => {
                assert_eq!(at_index, 1);
                assert!(
                    msg.contains("ExecutionProof::from_bytes"),
                    "expected from_bytes failure, got: {msg}"
                );
            }
            other => panic!("expected LayerProof error from from_bytes, got {other:?}"),
        }
    }

    #[test]
    fn folder_records_verified_kind_for_proven_layers() {
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| match i {
                0 => LayerProof::with_proof(claim, LayerProofKind::VmEvm, vec![0xE0]),
                2 => LayerProof::with_proof(claim, LayerProofKind::NonnativeFp, vec![0xE2]),
                _ => LayerProof::reference_only(claim),
            })
            .collect();
        let cp = LayerChainProof::new(layers);
        let folded = LayerChainFolder::fold(&cp, |_| Ok(())).expect("valid");
        assert_eq!(
            folded.layer_statuses,
            vec![
                LayerStatus::Verified { kind: LayerProofKind::VmEvm },
                LayerStatus::ReferenceOnly,
                LayerStatus::Verified { kind: LayerProofKind::NonnativeFp },
                LayerStatus::ReferenceOnly,
            ]
        );
        assert_eq!(folded.commitment, cp.commitment().unwrap());
    }

    #[test]
    fn folder_propagates_chain_error() {
        let mut cp = build_valid_proof_chain();
        if let LayerClaim::BlockBinding { block_hash, .. } = &mut cp.layers[1].claim {
            *block_hash = h(0xFF);
        }
        let res = LayerChainFolder::fold(&cp, |_| Ok(()));
        assert!(matches!(res, Err(ProofChainError::Chain(_))));
    }

    #[test]
    fn folder_short_circuits_on_first_layer_failure() {
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| match i {
                1 => LayerProof::with_proof(claim, LayerProofKind::Sha256, vec![0xAA]),
                3 => LayerProof::with_proof(claim, LayerProofKind::Ssz, vec![0xBB]),
                _ => LayerProof::reference_only(claim),
            })
            .collect();
        let cp = LayerChainProof::new(layers);

        use std::cell::Cell;
        let calls = Cell::new(0u32);
        let res = LayerChainFolder::fold(&cp, |_layer| {
            calls.set(calls.get() + 1);
            Err("layer 1 always fails in this test".to_string())
        });
        assert_eq!(
            res,
            Err(ProofChainError::LayerProof {
                at_index: 1,
                msg: "layer 1 always fails in this test".to_string(),
            })
        );
        // Short-circuit: the second proven layer (index 3) should not have
        // been seen by the closure.
        assert_eq!(calls.get(), 1, "fold must short-circuit on first failure");
    }

    #[test]
    fn folder_commitment_matches_chain_commitment() {
        let cp = build_valid_proof_chain();
        let folded = LayerChainFolder::fold(&cp, |_| Ok(())).unwrap();
        assert_eq!(folded.commitment, cp.commitment().unwrap());
    }

    #[test]
    fn layer_chain_proof_verifier_dispatches_on_kind() {
        // A real recursive folder will dispatch on `kind` to pick the right
        // per-AIR verifier. Demonstrate the pattern.
        let chain = build_valid_chain();
        let layers: Vec<LayerProof> = chain
            .claims
            .into_iter()
            .enumerate()
            .map(|(i, claim)| match i {
                0 => LayerProof::with_proof(claim, LayerProofKind::VmEvm, vec![0xE0]),
                1 => LayerProof::with_proof(claim, LayerProofKind::Sha256, vec![0xE1]),
                2 => LayerProof::with_proof(claim, LayerProofKind::NonnativeFp, vec![0xE2]),
                _ => LayerProof::reference_only(claim),
            })
            .collect();
        let cp = LayerChainProof::new(layers);

        use std::cell::Cell;
        let counts = (Cell::new(0u32), Cell::new(0u32), Cell::new(0u32));
        let result = cp.verify_with_layer_verifier(|layer| {
            match layer.kind {
                LayerProofKind::VmEvm => counts.0.set(counts.0.get() + 1),
                LayerProofKind::Sha256 => counts.1.set(counts.1.get() + 1),
                LayerProofKind::NonnativeFp => counts.2.set(counts.2.get() + 1),
                _ => return Err(format!("unexpected kind {}", layer.kind.as_str())),
            }
            Ok(())
        });
        assert_eq!(result, Ok(()));
        // Each kind verified exactly once; reference-only layer skipped.
        assert_eq!(counts.0.get(), 1, "VmEvm dispatched once");
        assert_eq!(counts.1.get(), 1, "Sha256 dispatched once");
        assert_eq!(counts.2.get(), 1, "NonnativeFp dispatched once");
    }

    #[test]
    fn layer_chain_proof_view_equals_construction() {
        // Round-trip: build chain → wrap as proofs → view as chain again =
        // identical claims.
        let original = build_valid_chain();
        let layers: Vec<LayerProof> = original
            .claims
            .iter()
            .cloned()
            .map(LayerProof::reference_only)
            .collect();
        let cp = LayerChainProof::new(layers);
        assert_eq!(cp.chain().claims, original.claims);
    }
}
