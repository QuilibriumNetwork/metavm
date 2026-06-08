//! Execution proof generation.
//!
//! This module implements the prover side of the MetaVM execution proof
//! protocol. Given an execution trace (as [`TracePolynomials`]) and a
//! VM-specific constraint system (via [`VmConstraintSystem`]), it produces
//! an [`ExecutionProof`] that a verifier can check without re-executing.
//!
//! The prover builds the combined constraint polynomial C(x) at full algebraic
//! degree using `build_constraint_polynomial`, then divides by Z(x) = x^n - 1
//! to get the quotient Q(x). If deg(Q) > n-1, Q is split into chunks.
//! The verifier recomputes C(z) from column evaluations — no constraint
//! commitment is included in the proof.

use crate::commitment::{self, Commitment, BatchProof};
use crate::field::{Scalar, CurveType};
use crate::trace::TracePolynomials;
use crate::vm_constraints::VmConstraintSystem;
use bls48581::bls48581::big;
use bls48581::bls48581::rom;
use metavm_core::transcript::Transcript;
use rayon::prelude::*;

/// A proof of correct execution.
#[derive(Clone, Debug)]
pub struct ExecutionProof {
    /// Commitments to each trace column polynomial.
    pub column_commitments: Vec<Commitment>,
    /// Commitments to the quotient polynomial chunk(s).
    /// Q(x) = Q_0(x) + x^n * Q_1(x) when deg(Q) > n-1.
    pub quotient_commitments: Vec<Commitment>,
    /// Evaluations of all committed polynomials at the challenge point `z`.
    /// Layout: [col_0(z), ..., col_k(z), Q_0(z), Q_1(z), ...]
    /// Each entry is a serialized scalar.
    pub evaluations: Vec<Vec<u8>>,
    /// Batch opening proof covering all committed polynomials at `z`.
    pub opening_proof: BatchProof,
    /// Number of actual execution steps in the trace (before padding).
    pub num_steps: u64,
    /// Padded domain size (power of 2) used for polynomial commitments.
    pub domain_size: u64,
    /// Number of quotient polynomial chunks (1 or 2).
    pub num_quotient_chunks: u8,
    /// Evaluations of shifted columns at ω·z (for cross-row constraints).
    /// Layout: [col_{s0}(ω·z), col_{s1}(ω·z), ...] matching shifted_column_indices().
    /// Empty if no cross-row constraints.
    pub shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for shifted columns at ω·z.
    /// None if no cross-row constraints.
    pub shifted_opening_proof: Option<BatchProof>,
    // ── LogUp range check proof ────────────────────────────────────────
    /// Commitments to LogUp auxiliary columns (byte limbs + h + m).
    /// Empty if no lookup declarations.
    pub logup_commitments: Vec<Commitment>,
    /// Evaluations of LogUp auxiliary columns at z.
    /// Empty if no lookup declarations.
    pub logup_evaluations: Vec<Vec<u8>>,
    /// Evaluations of LogUp shifted columns (h) at ω·z.
    /// Empty if no lookup declarations.
    pub logup_shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for LogUp columns at z.
    pub logup_opening_proof: Option<BatchProof>,
    /// Batch opening proof for LogUp shifted columns at ω·z.
    pub logup_shifted_opening_proof: Option<BatchProof>,
    // ── Bitwise LogUp (nibble-AND table) proof ─────────────────────────
    /// Commitments to bitwise LogUp auxiliary columns
    /// (nibble triples + inverses + table components + u_t + h + m).
    /// Empty if no bitwise lookup declarations.
    pub bitwise_commitments: Vec<Commitment>,
    /// Evaluations of bitwise LogUp columns at z.
    pub bitwise_evaluations: Vec<Vec<u8>>,
    /// Evaluation of the bitwise h column at ω·z.
    pub bitwise_shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for bitwise LogUp columns at z.
    pub bitwise_opening_proof: Option<BatchProof>,
    /// Batch opening proof for bitwise shifted h at ω·z.
    pub bitwise_shifted_opening_proof: Option<BatchProof>,
    // ── Memory permutation proof ───────────────────────────────────────
    /// Commitments to permutation auxiliary columns (sorted, Z, is_same_addr, inv_addr_diff).
    /// Empty if no memory permutation.
    pub perm_commitments: Vec<Commitment>,
    /// Evaluations of permutation columns at z.
    pub perm_evaluations: Vec<Vec<u8>>,
    /// Evaluations of permutation shifted columns at ω·z.
    pub perm_shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for permutation columns at z.
    pub perm_opening_proof: Option<BatchProof>,
    /// Batch opening proof for permutation shifted columns at ω·z.
    pub perm_shifted_opening_proof: Option<BatchProof>,
    // ── Oracle public-input data ─────────────────────────────────────
    /// Serialized oracle operation entries for external verification.
    ///
    /// Each entry contains: [row_index (8 bytes LE), selector_col (8 bytes LE),
    /// data_col_0 (8 bytes LE), ..., data_col_k (8 bytes LE)] for rows where
    /// an oracle selector is active. Absorbed into the Fiat-Shamir transcript
    /// to bind the proof to specific oracle values.
    pub oracle_data: Vec<Vec<u8>>,
    // ── Register file permutation proof ──────────────────────────────
    /// Commitments to register permutation auxiliary columns.
    /// Empty if no register ports declared.
    pub reg_perm_commitments: Vec<Commitment>,
    /// Evaluations of register permutation columns at z.
    pub reg_perm_evaluations: Vec<Vec<u8>>,
    /// Evaluations of register permutation shifted columns at ω·z.
    pub reg_perm_shifted_evaluations: Vec<Vec<u8>>,
    /// Batch opening proof for register permutation columns at z.
    pub reg_perm_opening_proof: Option<BatchProof>,
    /// Batch opening proof for register permutation shifted columns at ω·z.
    pub reg_perm_shifted_opening_proof: Option<BatchProof>,
    // ── Frame-stack LIFO permutation proof ────────────────────────────
    /// Commitment to the Z accumulator column for the frame-stack
    /// multiset permutation argument. Empty when the AIR does not
    /// declare a `frame_perm_layout` (RISC-V, SBF).
    pub frame_perm_commitment: Option<Commitment>,
    /// Evaluation of Z at z.
    pub frame_perm_evaluation: Option<Vec<u8>>,
    /// Evaluation of Z at ω·z (shifted).
    pub frame_perm_shifted_evaluation: Option<Vec<u8>>,
    /// Opening proof for Z at z.
    pub frame_perm_opening_proof: Option<BatchProof>,
    /// Opening proof for Z at ω·z.
    pub frame_perm_shifted_opening_proof: Option<BatchProof>,
    /// Reserved: empty in the current frame-perm design (which evaluates
    /// `is_pop(z)` rather than `is_pop(ω·z)`). Kept in the wire format
    /// so the byte layout stays stable if a future variant needs them.
    pub frame_perm_pop_shifted_evaluations: Vec<Vec<u8>>,
    /// Reserved; see [`Self::frame_perm_pop_shifted_evaluations`].
    pub frame_perm_pop_shifted_opening_proof: Option<BatchProof>,
}

/// Errors from [`ExecutionProof::from_bytes`] decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofDecodeError {
    /// Buffer truncated mid-field.
    Truncated { wanted: usize, available: usize },
    /// Length prefix declares more bytes than remain in the buffer.
    LengthOverflow,
    /// Trailing bytes after the proof structure was fully decoded.
    TrailingBytes(usize),
    /// `num_quotient_chunks` byte was zero. Any non-zero value is accepted
    /// — the advanced prover path (logup + permutation arguments) can
    /// emit 3+ chunks for high-degree grand-product polynomials.
    InvalidQuotientChunks(u8),
    /// `Option<BatchProof>` tag byte was neither 0 nor 1.
    InvalidOptionTag(u8),
}

impl ExecutionProof {
    /// Serialize the proof into a self-delimited byte sequence using a
    /// length-prefixed framing. No external serde dependency — every field
    /// is already a byte vector (commitments are 74-byte compressed G1
    /// points wrapped in `Vec<u8>`; evaluations and BatchProofs are
    /// likewise byte payloads).
    ///
    /// Framing primitives (all big-endian):
    ///   * `bytes(v)`         → `[u32 len][len bytes]`
    ///   * `vec_bytes(vs)`    → `[u32 count][bytes(vs[0])][bytes(vs[1])]…`
    ///   * `vec_commit(cs)`   → `vec_bytes` over each commitment's inner bytes
    ///   * `option(opt)`      → `[u8 tag][BatchProof if tag==1]`
    ///   * `BatchProof(bp)`   → `bytes(bp.d) ‖ bytes(bp.proof)`
    ///   * scalar `u64`       → 8 BE bytes
    ///   * scalar `u8`        → 1 byte
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1024);
        write_vec_commit(&mut out, &self.column_commitments);
        write_vec_commit(&mut out, &self.quotient_commitments);
        write_vec_bytes(&mut out, &self.evaluations);
        write_batch_proof(&mut out, &self.opening_proof);
        out.extend_from_slice(&self.num_steps.to_be_bytes());
        out.extend_from_slice(&self.domain_size.to_be_bytes());
        out.push(self.num_quotient_chunks);
        write_vec_bytes(&mut out, &self.shifted_evaluations);
        write_option_batch_proof(&mut out, self.shifted_opening_proof.as_ref());
        write_vec_commit(&mut out, &self.logup_commitments);
        write_vec_bytes(&mut out, &self.logup_evaluations);
        write_vec_bytes(&mut out, &self.logup_shifted_evaluations);
        write_option_batch_proof(&mut out, self.logup_opening_proof.as_ref());
        write_option_batch_proof(&mut out, self.logup_shifted_opening_proof.as_ref());
        write_vec_commit(&mut out, &self.bitwise_commitments);
        write_vec_bytes(&mut out, &self.bitwise_evaluations);
        write_vec_bytes(&mut out, &self.bitwise_shifted_evaluations);
        write_option_batch_proof(&mut out, self.bitwise_opening_proof.as_ref());
        write_option_batch_proof(&mut out, self.bitwise_shifted_opening_proof.as_ref());
        write_vec_commit(&mut out, &self.perm_commitments);
        write_vec_bytes(&mut out, &self.perm_evaluations);
        write_vec_bytes(&mut out, &self.perm_shifted_evaluations);
        write_option_batch_proof(&mut out, self.perm_opening_proof.as_ref());
        write_option_batch_proof(&mut out, self.perm_shifted_opening_proof.as_ref());
        write_vec_bytes(&mut out, &self.oracle_data);
        write_vec_commit(&mut out, &self.reg_perm_commitments);
        write_vec_bytes(&mut out, &self.reg_perm_evaluations);
        write_vec_bytes(&mut out, &self.reg_perm_shifted_evaluations);
        write_option_batch_proof(&mut out, self.reg_perm_opening_proof.as_ref());
        write_option_batch_proof(&mut out, self.reg_perm_shifted_opening_proof.as_ref());
        // Frame-stack permutation fields.
        write_option_commit(&mut out, self.frame_perm_commitment.as_ref());
        write_option_bytes(&mut out, self.frame_perm_evaluation.as_deref());
        write_option_bytes(&mut out, self.frame_perm_shifted_evaluation.as_deref());
        write_option_batch_proof(&mut out, self.frame_perm_opening_proof.as_ref());
        write_option_batch_proof(&mut out, self.frame_perm_shifted_opening_proof.as_ref());
        write_vec_bytes(&mut out, &self.frame_perm_pop_shifted_evaluations);
        write_option_batch_proof(&mut out, self.frame_perm_pop_shifted_opening_proof.as_ref());
        out
    }

    /// Deserialize a proof produced by [`Self::to_bytes`]. Returns
    /// [`ProofDecodeError`] on malformed input. Strict: trailing bytes
    /// after a complete decode are an error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProofDecodeError> {
        let mut r = Reader::new(bytes);
        let column_commitments = r.read_vec_commit()?;
        let quotient_commitments = r.read_vec_commit()?;
        let evaluations = r.read_vec_bytes()?;
        let opening_proof = r.read_batch_proof()?;
        let num_steps = r.read_u64()?;
        let domain_size = r.read_u64()?;
        let num_quotient_chunks = r.read_u8()?;
        if num_quotient_chunks == 0 {
            return Err(ProofDecodeError::InvalidQuotientChunks(0));
        }
        let shifted_evaluations = r.read_vec_bytes()?;
        let shifted_opening_proof = r.read_option_batch_proof()?;
        let logup_commitments = r.read_vec_commit()?;
        let logup_evaluations = r.read_vec_bytes()?;
        let logup_shifted_evaluations = r.read_vec_bytes()?;
        let logup_opening_proof = r.read_option_batch_proof()?;
        let logup_shifted_opening_proof = r.read_option_batch_proof()?;
        let bitwise_commitments = r.read_vec_commit()?;
        let bitwise_evaluations = r.read_vec_bytes()?;
        let bitwise_shifted_evaluations = r.read_vec_bytes()?;
        let bitwise_opening_proof = r.read_option_batch_proof()?;
        let bitwise_shifted_opening_proof = r.read_option_batch_proof()?;
        let perm_commitments = r.read_vec_commit()?;
        let perm_evaluations = r.read_vec_bytes()?;
        let perm_shifted_evaluations = r.read_vec_bytes()?;
        let perm_opening_proof = r.read_option_batch_proof()?;
        let perm_shifted_opening_proof = r.read_option_batch_proof()?;
        let oracle_data = r.read_vec_bytes()?;
        let reg_perm_commitments = r.read_vec_commit()?;
        let reg_perm_evaluations = r.read_vec_bytes()?;
        let reg_perm_shifted_evaluations = r.read_vec_bytes()?;
        let reg_perm_opening_proof = r.read_option_batch_proof()?;
        let reg_perm_shifted_opening_proof = r.read_option_batch_proof()?;
        // Frame-stack permutation fields.
        let frame_perm_commitment = r.read_option_commit()?;
        let frame_perm_evaluation = r.read_option_bytes()?;
        let frame_perm_shifted_evaluation = r.read_option_bytes()?;
        let frame_perm_opening_proof = r.read_option_batch_proof()?;
        let frame_perm_shifted_opening_proof = r.read_option_batch_proof()?;
        let frame_perm_pop_shifted_evaluations = r.read_vec_bytes()?;
        let frame_perm_pop_shifted_opening_proof = r.read_option_batch_proof()?;

        let trailing = r.remaining();
        if trailing > 0 {
            return Err(ProofDecodeError::TrailingBytes(trailing));
        }

        Ok(ExecutionProof {
            column_commitments,
            quotient_commitments,
            evaluations,
            opening_proof,
            num_steps,
            domain_size,
            num_quotient_chunks,
            shifted_evaluations,
            shifted_opening_proof,
            logup_commitments,
            logup_evaluations,
            logup_shifted_evaluations,
            logup_opening_proof,
            logup_shifted_opening_proof,
            bitwise_commitments,
            bitwise_evaluations,
            bitwise_shifted_evaluations,
            bitwise_opening_proof,
            bitwise_shifted_opening_proof,
            perm_commitments,
            perm_evaluations,
            perm_shifted_evaluations,
            perm_opening_proof,
            perm_shifted_opening_proof,
            oracle_data,
            reg_perm_commitments,
            reg_perm_evaluations,
            reg_perm_shifted_evaluations,
            reg_perm_opening_proof,
            reg_perm_shifted_opening_proof,
            frame_perm_commitment,
            frame_perm_evaluation,
            frame_perm_shifted_evaluation,
            frame_perm_opening_proof,
            frame_perm_shifted_opening_proof,
            frame_perm_pop_shifted_evaluations,
            frame_perm_pop_shifted_opening_proof,
        })
    }
}

// ── Framing helpers (private) ─────────────────────────────────────────

fn write_bytes(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u32).to_be_bytes());
    out.extend_from_slice(b);
}

fn write_vec_bytes(out: &mut Vec<u8>, vs: &[Vec<u8>]) {
    out.extend_from_slice(&(vs.len() as u32).to_be_bytes());
    for v in vs {
        write_bytes(out, v);
    }
}

fn write_vec_commit(out: &mut Vec<u8>, cs: &[Commitment]) {
    out.extend_from_slice(&(cs.len() as u32).to_be_bytes());
    for c in cs {
        write_bytes(out, &c.0);
    }
}

fn write_batch_proof(out: &mut Vec<u8>, bp: &BatchProof) {
    write_bytes(out, &bp.d);
    write_bytes(out, &bp.proof);
}

fn write_option_batch_proof(out: &mut Vec<u8>, opt: Option<&BatchProof>) {
    match opt {
        None => out.push(0),
        Some(bp) => {
            out.push(1);
            write_batch_proof(out, bp);
        }
    }
}

fn write_option_commit(out: &mut Vec<u8>, opt: Option<&Commitment>) {
    match opt {
        None => out.push(0),
        Some(c) => {
            out.push(1);
            write_bytes(out, &c.0);
        }
    }
}

fn write_option_bytes(out: &mut Vec<u8>, opt: Option<&[u8]>) {
    match opt {
        None => out.push(0),
        Some(b) => {
            out.push(1);
            write_bytes(out, b);
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ProofDecodeError> {
        if self.pos + n > self.buf.len() {
            return Err(ProofDecodeError::Truncated {
                wanted: n,
                available: self.buf.len() - self.pos,
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8, ProofDecodeError> {
        Ok(self.take(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, ProofDecodeError> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, ProofDecodeError> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_be_bytes(a))
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, ProofDecodeError> {
        let len = self.read_u32()? as usize;
        if self.pos + len > self.buf.len() {
            return Err(ProofDecodeError::LengthOverflow);
        }
        Ok(self.take(len)?.to_vec())
    }

    fn read_vec_bytes(&mut self) -> Result<Vec<Vec<u8>>, ProofDecodeError> {
        let n = self.read_u32()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_bytes()?);
        }
        Ok(out)
    }

    fn read_vec_commit(&mut self) -> Result<Vec<Commitment>, ProofDecodeError> {
        let n = self.read_u32()? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(Commitment(self.read_bytes()?));
        }
        Ok(out)
    }

    fn read_batch_proof(&mut self) -> Result<BatchProof, ProofDecodeError> {
        let d = self.read_bytes()?;
        let proof = self.read_bytes()?;
        Ok(BatchProof { d, proof })
    }

    fn read_option_batch_proof(&mut self) -> Result<Option<BatchProof>, ProofDecodeError> {
        let tag = self.read_u8()?;
        match tag {
            0 => Ok(None),
            1 => Ok(Some(self.read_batch_proof()?)),
            other => Err(ProofDecodeError::InvalidOptionTag(other)),
        }
    }

    fn read_option_commit(&mut self) -> Result<Option<Commitment>, ProofDecodeError> {
        let tag = self.read_u8()?;
        match tag {
            0 => Ok(None),
            1 => Ok(Some(Commitment(self.read_bytes()?))),
            other => Err(ProofDecodeError::InvalidOptionTag(other)),
        }
    }

    fn read_option_bytes(&mut self) -> Result<Option<Vec<u8>>, ProofDecodeError> {
        let tag = self.read_u8()?;
        match tag {
            0 => Ok(None),
            1 => Ok(Some(self.read_bytes()?)),
            other => Err(ProofDecodeError::InvalidOptionTag(other)),
        }
    }
}

/// A chunk proof wrapping an execution proof with state chain metadata.
#[derive(Clone, Debug)]
pub struct ChunkProof {
    /// The underlying execution proof for this chunk's trace.
    pub execution_proof: ExecutionProof,
    /// SHA3-256 hash of the VM state at the beginning of this chunk.
    pub initial_state_hash: [u8; 32],
    /// SHA3-256 hash of the VM state at the end of this chunk.
    pub final_state_hash: [u8; 32],
    /// Sequential chunk index (0-based).
    pub chunk_index: u64,
}

impl ChunkProof {
    /// Serialize the chunk proof: `ExecutionProof bytes ‖ initial_state ‖
    /// final_state ‖ chunk_index_BE`. Reuses [`ExecutionProof::to_bytes`]
    /// for the inner proof body, then appends 32 + 32 + 8 = 72 bytes of
    /// chunk-chain metadata. Total framing makes the byte stream
    /// self-delimiting.
    pub fn to_bytes(&self) -> Vec<u8> {
        let inner = self.execution_proof.to_bytes();
        let mut out = Vec::with_capacity(inner.len() + 4 + 32 + 32 + 8);
        // Length-prefix the inner ExecutionProof so the decoder knows
        // where chunk metadata begins.
        out.extend_from_slice(&(inner.len() as u32).to_be_bytes());
        out.extend_from_slice(&inner);
        out.extend_from_slice(&self.initial_state_hash);
        out.extend_from_slice(&self.final_state_hash);
        out.extend_from_slice(&self.chunk_index.to_be_bytes());
        out
    }

    /// Decode a chunk proof produced by [`Self::to_bytes`]. Strict:
    /// trailing bytes after a complete decode are an error.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ProofDecodeError> {
        if bytes.len() < 4 {
            return Err(ProofDecodeError::Truncated {
                wanted: 4,
                available: bytes.len(),
            });
        }
        let inner_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let inner_end = 4 + inner_len;
        if bytes.len() < inner_end + 32 + 32 + 8 {
            return Err(ProofDecodeError::Truncated {
                wanted: inner_end + 32 + 32 + 8 - 4,
                available: bytes.len() - 4,
            });
        }
        let execution_proof = ExecutionProof::from_bytes(&bytes[4..inner_end])?;
        let initial_state_hash: [u8; 32] = bytes[inner_end..inner_end + 32]
            .try_into()
            .expect("32 bytes");
        let final_state_hash: [u8; 32] = bytes[inner_end + 32..inner_end + 64]
            .try_into()
            .expect("32 bytes");
        let chunk_index = u64::from_be_bytes(
            bytes[inner_end + 64..inner_end + 72].try_into().expect("8 bytes"),
        );
        let total = inner_end + 72;
        if bytes.len() != total {
            return Err(ProofDecodeError::TrailingBytes(bytes.len() - total));
        }
        Ok(ChunkProof {
            execution_proof,
            initial_state_hash,
            final_state_hash,
            chunk_index,
        })
    }
}

/// Convert 32 Fiat-Shamir challenge bytes to a BLS48-581 BIG field element.
pub fn challenge_to_big(challenge_bytes: &[u8; 32]) -> big::BIG {
    let mut z_padded = [0u8; big::MODBYTES];
    z_padded[big::MODBYTES - 32..].copy_from_slice(challenge_bytes);
    let mut z = big::BIG::frombytes(&z_padded);
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);
    z.rmod(&modulus);
    z
}

/// Convert 32 Fiat-Shamir challenge bytes to a Scalar for a given curve.
pub fn challenge_to_scalar(challenge_bytes: &[u8; 32], curve: CurveType) -> Scalar {
    Scalar::from_challenge_bytes(challenge_bytes, curve)
}

/// Serialize a BIG value to MODBYTES for inclusion in proof evaluations.
fn big_to_eval_bytes(val: &big::BIG) -> Vec<u8> {
    let mut buf = vec![0u8; big::MODBYTES];
    val.tobytes(&mut buf);
    buf
}

/// Core proving logic shared by `prove()` and `prove_chunk()`.
///
/// Currently only supports BLS48-581 for commitment operations.
fn prove_inner(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    transcript: &mut Transcript,
) -> ExecutionProof {
    let num_steps = trace.num_steps();
    let domain_size = trace.domain_size();
    let curve = trace.curve;
    let modulus = big::BIG::new_ints(&rom::CURVE_ORDER);

    assert_eq!(curve, CurveType::Bls48581, "Commitment currently only supports BLS48-581");

    // Get columns as BIG vectors for commitment operations
    let mut big_columns = trace.columns_as_bls48581();

    // Fix selector padding: set the designated "no-op" selector to 1 on
    // padding rows so that the sum-to-one constraint is satisfied everywhere.
    if let Some(padding_col) = constraints.padding_selector_column() {
        let one = big::BIG::new_int(1);
        let num_rows = trace.num_rows;
        let padded = domain_size as usize;
        if padding_col < big_columns.len() {
            for i in num_rows..padded {
                big_columns[padding_col][i] = big::BIG::new_copy(&one);
            }
        }
    }

    // Fix trace padding for cross-row constraint satisfaction (PC continuity).
    {
        let num_rows = trace.num_rows;
        let padded = domain_size as usize;
        let mut scalar_columns: Vec<Vec<Scalar>> = big_columns.iter()
            .map(|col| col.iter().map(|b| Scalar::Bls48581(big::BIG::new_copy(b))).collect())
            .collect();
        constraints.fix_trace_padding(&mut scalar_columns, num_rows, padded);
        // Copy back to big_columns
        for (col_idx, scalar_col) in scalar_columns.iter().enumerate() {
            for (row_idx, s) in scalar_col.iter().enumerate() {
                big_columns[col_idx][row_idx] = big::BIG::new_copy(s.as_bls48581());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 1: Commit to each trace column
    // -----------------------------------------------------------------------
    let mut column_commitments: Vec<Commitment> = Vec::with_capacity(big_columns.len());

    for col in &big_columns {
        let comm = commitment::commit_from_scalars(col, domain_size);
        column_commitments.push(comm);
    }

    // -----------------------------------------------------------------------
    // Step 2: Absorb commitments into Fiat-Shamir transcript
    // -----------------------------------------------------------------------
    transcript.append_u64(b"num_steps", num_steps);
    transcript.append_u64(b"domain_size", domain_size);

    for comm in &column_commitments {
        transcript.append_message(b"column_commitment", &comm.0);
    }

    // -----------------------------------------------------------------------
    // Step 3: Draw constraint combination challenge alpha
    // -----------------------------------------------------------------------
    let alpha_bytes = transcript.challenge_bytes(b"alpha");
    let alpha_big = big::BIG::frombytes(&alpha_bytes);
    let alpha = Scalar::Bls48581(big::BIG::new_copy(&alpha_big));

    // -----------------------------------------------------------------------
    // Step 4: Build C(x) — either via build_constraint_polynomial (selector-based)
    //         or via evaluate_on_domain + IFFT (fallback)
    //
    // We compute column_coeffs_big here and cache it for reuse in Step 8,
    // avoiding a redundant IFFT pass over all columns.
    // -----------------------------------------------------------------------
    let c_coeffs: Vec<big::BIG>;
    // Always use the algebraic-coefficient path so that shifted, LogUp,
    // perm, and reg-perm contributions are part of C(X). The earlier
    // gating on selector presence left selector-less AIRs (MPT, SSZ)
    // with a fallback path that omitted those contributions, and the
    // verifier's mirroring guard skipped the `Q(z)·Z(z) == C(z)` identity
    // entirely — meaning their constraints were never enforced. Each
    // selector-less AIR must additionally gate its shifted constraint
    // to vanish on padding-row transitions so `C / Z_H` is divisible.
    let use_build_poly = true;

    // Compute column coefficients ONCE — reused in Step 8 for evaluations at z
    let column_coeffs_big: Vec<Vec<big::BIG>> = big_columns.iter()
        .map(|col| commitment::eval_to_coeff(col, domain_size))
        .collect();

    // Check for cross-row constraints
    let shifted_indices = constraints.shifted_column_indices();
    let has_shifts = !shifted_indices.is_empty();

    if use_build_poly {
        // Selector-based VMs: build C(x) algebraically in coefficient form
        let column_coeffs_scalar: Vec<Vec<Scalar>> = column_coeffs_big.iter()
            .map(|coeffs| {
                coeffs.iter().map(|b| Scalar::Bls48581(big::BIG::new_copy(b))).collect()
            })
            .collect();

        let mut c_coeffs_scalar = constraints.build_constraint_polynomial(
            &column_coeffs_scalar, &alpha, domain_size,
        );

        // Add cross-row constraint polynomial if any
        if has_shifts {
            use bls48581::bls;
            let s = bls::singleton();
            let omega_big = s.RootsOfUnityBLS48581[&domain_size][1].clone();
            let omega = Scalar::Bls48581(omega_big);
            let c_shifted = constraints.build_shifted_constraint_polynomial(
                &column_coeffs_scalar, &alpha, domain_size,
                &omega,
                constraints.num_constraints(),
            );
            c_coeffs_scalar = crate::poly_arith::poly_add(&c_coeffs_scalar, &c_shifted, curve);
        }

        c_coeffs = c_coeffs_scalar.iter()
            .map(|s| big::BIG::new_copy(s.as_bls48581()))
            .collect();
    } else {
        // Fallback: evaluate constraints on domain, combine with alpha, IFFT to coeffs
        let columns_refs: Vec<&Vec<Scalar>> = trace.columns().into_iter().collect();
        let constraint_evals = constraints.evaluate_on_domain(&columns_refs, trace.num_rows);
        let n = domain_size as usize;

        // Combine: C_evals[j] = Σ alpha^i * constraint_i_evals[j]
        let mut c_evals_big = vec![big::BIG::new(); n];
        let mut alpha_power = big::BIG::new_int(1);
        for constraint in &constraint_evals {
            for j in 0..n {
                let val = if j < constraint.len() {
                    big::BIG::new_copy(constraint[j].as_bls48581())
                } else {
                    big::BIG::new()
                };
                let term = big::BIG::modmul(&alpha_power, &val, &modulus);
                c_evals_big[j] = big::BIG::modadd(&c_evals_big[j], &term, &modulus);
            }
            alpha_power = big::BIG::modmul(&alpha_power, &alpha_big, &modulus);
        }

        c_coeffs = commitment::eval_to_coeff(&c_evals_big, domain_size);
    };

    // -----------------------------------------------------------------------
    // Step 5: Divide C(x) by Z(x) = x^n - 1 to get Q(x)
    // -----------------------------------------------------------------------
    let n = domain_size as usize;
    let c_len = c_coeffs.len();

    // Pad to at least n if needed
    let mut dividend = c_coeffs.clone();
    while dividend.len() < n {
        dividend.push(big::BIG::new());
    }

    // Q degree = c_deg - n. If c_deg < n, Q = 0 (constraint is trivially divisible).
    let c_deg = if c_len > 0 { c_len - 1 } else { 0 };
    let q_deg = if c_deg >= n { c_deg - n } else { 0 };

    let mut quotient_coeffs_all = vec![big::BIG::new(); if c_deg >= n { q_deg + 1 } else { n }];

    if c_deg >= n {
        // Proper polynomial division: C(x) / (x^n - 1)
        // For i from c_deg down to n: q[i-n] = dividend[i]; dividend[i-n] += dividend[i]
        let mut div = dividend.clone();
        while div.len() <= c_deg {
            div.push(big::BIG::new());
        }
        for i in (n..=c_deg).rev() {
            quotient_coeffs_all[i - n] = big::BIG::new_copy(&div[i]);
            div[i - n] = big::BIG::modadd(&div[i - n], &div[i], &modulus);
        }
    } else {
        // C(x) degree < n: standard division where C has degree n-1
        // This handles the case from evaluate_on_domain (degree n-1 constraints)
        let mut div = dividend.clone();
        while div.len() < 2 * n {
            div.push(big::BIG::new());
        }
        quotient_coeffs_all = vec![big::BIG::new(); n];
        for i in (0..n).rev() {
            quotient_coeffs_all[i] = big::BIG::new_copy(&div[n + i]);
            div[i] = big::BIG::modadd(&div[i], &quotient_coeffs_all[i], &modulus);
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Split Q into chunks if needed, commit each
    // -----------------------------------------------------------------------
    let raw_num_q_chunks = if quotient_coeffs_all.len() > n {
        ((quotient_coeffs_all.len() - 1) / n) + 1
    } else {
        1
    };
    eprintln!("[prover-big] c_coeffs.len()={} c_deg={} q_deg={} raw_chunks={}",
        c_coeffs.len(), c_deg, q_deg, raw_num_q_chunks);
    let num_q_chunks = raw_num_q_chunks.min(2) as u8; // At most 2 chunks for our degree budget

    let mut quotient_commitments = Vec::new();
    let mut q_chunk_coeffs: Vec<Vec<big::BIG>> = Vec::new();

    for chunk_idx in 0..num_q_chunks as usize {
        let start = chunk_idx * n;
        let end = (start + n).min(quotient_coeffs_all.len());
        let mut chunk: Vec<big::BIG> = quotient_coeffs_all[start..end].to_vec();
        // Pad chunk to n for commitment
        while chunk.len() < n {
            chunk.push(big::BIG::new());
        }
        // Commit in coefficient form
        let comm_point = bls48581::commit_scalars_monomial(&chunk);
        quotient_commitments.push(Commitment(comm_point));
        q_chunk_coeffs.push(chunk);
    }

    // -----------------------------------------------------------------------
    // Step 7: Absorb quotient commitments, derive z
    // -----------------------------------------------------------------------
    for qc in &quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let z_bytes = transcript.challenge_bytes(b"z");
    let z = challenge_to_big(&z_bytes);

    // -----------------------------------------------------------------------
    // Step 8: Evaluate columns and Q chunks at z
    // (Reuses column_coeffs_big computed in Step 4 — no redundant IFFTs)
    // -----------------------------------------------------------------------
    let mut evaluations: Vec<Vec<u8>> = Vec::with_capacity(big_columns.len() + num_q_chunks as usize);

    for coeffs in &column_coeffs_big {
        let y = commitment::eval_poly_at(coeffs, &z);
        evaluations.push(big_to_eval_bytes(&y));
    }

    // Evaluate each Q chunk at z
    let mut q_chunk_evals = Vec::new();
    for chunk in &q_chunk_coeffs {
        let q_at_z = commitment::eval_poly_at(chunk, &z);
        evaluations.push(big_to_eval_bytes(&q_at_z));
        q_chunk_evals.push(q_at_z);
    }

    // -----------------------------------------------------------------------
    // Step 8b: Evaluate shifted columns at ω·z (for cross-row constraints)
    // -----------------------------------------------------------------------
    let mut shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_shifts {
        use bls48581::bls;
        let s = bls::singleton();
        let omega_big = s.RootsOfUnityBLS48581[&domain_size][1].clone();
        let omega_z = big::BIG::modmul(&omega_big, &z, &modulus);
        for &col_idx in &shifted_indices {
            let y = commitment::eval_poly_at(&column_coeffs_big[col_idx], &omega_z);
            shifted_evaluations.push(big_to_eval_bytes(&y));
        }
    }

    // -----------------------------------------------------------------------
    // Step 9: Absorb evaluations, derive batch opening challenge β
    // -----------------------------------------------------------------------
    for eval_bytes in &evaluations {
        transcript.append_message(b"evaluation", eval_bytes);
    }
    for se in &shifted_evaluations {
        transcript.append_message(b"shifted_evaluation", se);
    }
    let beta_bytes = transcript.challenge_bytes(b"beta");
    let beta = challenge_to_big(&beta_bytes);

    // -----------------------------------------------------------------------
    // Step 10: Batch open all polynomials at z using β
    // -----------------------------------------------------------------------
    let batch_size = n; // All coefficient arrays are padded to n
    let mut combined_coeffs = vec![big::BIG::new(); batch_size];
    let mut combined_y = big::BIG::new();
    let mut beta_power = big::BIG::new_int(1);

    // Combine column polynomials
    for (col_idx, coeffs) in column_coeffs_big.iter().enumerate() {
        let y_bytes = &evaluations[col_idx];
        let y = big::BIG::frombytes(y_bytes);
        let y_term = big::BIG::modmul(&beta_power, &y, &modulus);
        combined_y = big::BIG::modadd(&combined_y, &y_term, &modulus);

        for j in 0..coeffs.len().min(batch_size) {
            let term = big::BIG::modmul(&beta_power, &coeffs[j], &modulus);
            combined_coeffs[j] = big::BIG::modadd(&combined_coeffs[j], &term, &modulus);
        }
        beta_power = big::BIG::modmul(&beta_power, &beta, &modulus);
    }

    // Combine Q chunk polynomials
    for (chunk_idx, chunk) in q_chunk_coeffs.iter().enumerate() {
        let q_eval = &q_chunk_evals[chunk_idx];
        let y_term = big::BIG::modmul(&beta_power, q_eval, &modulus);
        combined_y = big::BIG::modadd(&combined_y, &y_term, &modulus);

        for j in 0..chunk.len().min(batch_size) {
            let term = big::BIG::modmul(&beta_power, &chunk[j], &modulus);
            combined_coeffs[j] = big::BIG::modadd(&combined_coeffs[j], &term, &modulus);
        }
        beta_power = big::BIG::modmul(&beta_power, &beta, &modulus);
    }

    // Subtract combined_y from constant term
    combined_coeffs[0] = big::BIG::modadd(
        &combined_coeffs[0],
        &big::BIG::modneg(&combined_y, &modulus),
        &modulus,
    );

    // Synthetic division by (x - z)
    let q_open_coeffs = commitment::div_by_linear(&combined_coeffs, &z);

    // Commit quotient in coefficient form via monomial SRS
    let proof_point = bls48581::commit_scalars_monomial(&q_open_coeffs);

    let opening_proof = BatchProof {
        d: vec![],
        proof: proof_point,
    };

    // -----------------------------------------------------------------------
    // Step 10b: Batch open shifted columns at ω·z (if any)
    // -----------------------------------------------------------------------
    let shifted_opening_proof = if has_shifts {
        use bls48581::bls;
        let beta_shifted_bytes = transcript.challenge_bytes(b"beta_shifted");
        let beta_shifted = challenge_to_big(&beta_shifted_bytes);
        let s = bls::singleton();
        let omega_big = s.RootsOfUnityBLS48581[&domain_size][1].clone();
        let omega_z = big::BIG::modmul(&omega_big, &z, &modulus);

        let mut combined_shifted = vec![big::BIG::new(); batch_size];
        let mut combined_shifted_y = big::BIG::new();
        let mut bs_power = big::BIG::new_int(1);

        for (i, &col_idx) in shifted_indices.iter().enumerate() {
            let y = big::BIG::frombytes(&shifted_evaluations[i]);
            let y_term = big::BIG::modmul(&bs_power, &y, &modulus);
            combined_shifted_y = big::BIG::modadd(&combined_shifted_y, &y_term, &modulus);

            let coeffs = &column_coeffs_big[col_idx];
            for j in 0..coeffs.len().min(batch_size) {
                let term = big::BIG::modmul(&bs_power, &coeffs[j], &modulus);
                combined_shifted[j] = big::BIG::modadd(&combined_shifted[j], &term, &modulus);
            }
            bs_power = big::BIG::modmul(&bs_power, &beta_shifted, &modulus);
        }

        combined_shifted[0] = big::BIG::modadd(
            &combined_shifted[0],
            &big::BIG::modneg(&combined_shifted_y, &modulus),
            &modulus,
        );
        let q_shifted = commitment::div_by_linear(&combined_shifted, &omega_z);
        let shifted_proof_point = bls48581::commit_scalars_monomial(&q_shifted);

        Some(BatchProof {
            d: vec![],
            proof: shifted_proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 11: Assemble and return the proof
    // -----------------------------------------------------------------------
    ExecutionProof {
        column_commitments,
        quotient_commitments,
        evaluations,
        opening_proof,
        num_steps,
        domain_size,
        num_quotient_chunks: num_q_chunks,
        shifted_evaluations,
        shifted_opening_proof,
        logup_commitments: Vec::new(),
        logup_evaluations: Vec::new(),
        logup_shifted_evaluations: Vec::new(),
        logup_opening_proof: None,
        logup_shifted_opening_proof: None,
        bitwise_commitments: Vec::new(),
        bitwise_evaluations: Vec::new(),
        bitwise_shifted_evaluations: Vec::new(),
        bitwise_opening_proof: None,
        bitwise_shifted_opening_proof: None,
        perm_commitments: Vec::new(),
        perm_evaluations: Vec::new(),
        perm_shifted_evaluations: Vec::new(),
        perm_opening_proof: None,
        perm_shifted_opening_proof: None,
        oracle_data: Vec::new(),
        reg_perm_commitments: Vec::new(),
        reg_perm_evaluations: Vec::new(),
        reg_perm_shifted_evaluations: Vec::new(),
        reg_perm_opening_proof: None,
        reg_perm_shifted_opening_proof: None,
        frame_perm_commitment: None,
        frame_perm_evaluation: None,
        frame_perm_shifted_evaluation: None,
        frame_perm_opening_proof: None,
        frame_perm_shifted_opening_proof: None,
        frame_perm_pop_shifted_evaluations: Vec::new(),
        frame_perm_pop_shifted_opening_proof: None,
    }
}

/// Generate an execution proof from a trace and constraint system.
pub fn prove(trace: &TracePolynomials, constraints: &dyn VmConstraintSystem) -> ExecutionProof {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    prove_inner(trace, constraints, &mut transcript)
}

/// Generate an execution proof using a generic CommitmentScheme.
///
/// Works with any curve type via the [`CommitmentScheme`] trait (BLS48-581 or BLS12-381).
pub fn prove_with_scheme(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    scheme: &dyn crate::scheme::CommitmentScheme,
) -> ExecutionProof {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    prove_inner_scheme(trace, constraints, &mut transcript, scheme)
}

/// Core proving logic using a generic CommitmentScheme.
/// Output of [`commit_main_columns_phase1`]: everything the rest of the
/// prover needs to continue past Step 2 (column commitments absorbed into
/// the Fiat-Shamir transcript). Designed to be opaque from the joint
/// prover's perspective — it threads the state from phase-1 back into
/// phase-2 (the as-yet-unexposed remainder of `prove_inner_scheme`).
pub struct MainCommitState {
    pub num_steps: u64,
    pub domain_size: u64,
    pub col_eval_forms: Vec<Vec<Scalar>>,
    pub column_coeffs_all: Vec<Vec<Scalar>>,
    pub column_commitments: Vec<Commitment>,
    /// Accumulated wall-clock for the IFFT step. Carried so phase-2 can
    /// keep producing the existing `[prover] ifft_trace=... commit_trace=...`
    /// diagnostic line without re-instrumenting.
    pub ifft_trace_ms: u128,
    pub commit_trace_ms: u128,
    pub prove_start: std::time::Instant,
}

/// Phase-1 of the prover: pad the trace, IFFT to coefficient form,
/// commit to each main trace column, and absorb commitments into the
/// supplied transcript.
///
/// After this call, `transcript` carries `num_steps + domain_size + every
/// column commitment`. The next protocol step (drawing α) is intentionally
/// NOT done here so that the **joint cross-AIR prover** can interleave a
/// joint-γ derivation between multiple AIRs' commit phases.
pub(crate) fn commit_main_columns_phase1(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    transcript: &mut Transcript,
    scheme: &dyn crate::scheme::CommitmentScheme,
) -> MainCommitState {
    use std::time::Instant;
    let prove_start = Instant::now();

    let num_steps = trace.num_steps();
    let raw_domain_size = trace.domain_size();
    let curve = trace.curve;

    // The 8-bit LogUp range table requires ≥ 256 domain slots. When LogUp
    // declarations are present, bump `domain_size` to `max(raw, 256)` and
    // re-pad all trace columns with zeros to the enlarged domain.
    let needs_logup_domain = !constraints.lookup_declarations().is_empty();
    let domain_size = if needs_logup_domain {
        raw_domain_size.max(crate::lookup::RANGE_TABLE_SIZE as u64)
    } else {
        raw_domain_size
    };

    let mut col_eval_forms: Vec<Vec<Scalar>> = trace.columns.iter()
        .map(|p| {
            let mut v = p.evaluations.clone();
            if (v.len() as u64) < domain_size {
                v.resize(domain_size as usize, Scalar::zero(curve));
            }
            v
        })
        .collect();

    // Fix selector padding: set the designated "no-op" selector to 1 on
    // padding rows so that the sum-to-one constraint is satisfied everywhere.
    if let Some(padding_col) = constraints.padding_selector_column() {
        let one = Scalar::one(curve);
        let num_rows = trace.num_rows;
        let padded = domain_size as usize;
        if padding_col < col_eval_forms.len() {
            for i in num_rows..padded {
                col_eval_forms[padding_col][i] = one.clone();
            }
        }
    }

    // Fix trace padding for cross-row constraint satisfaction (PC continuity).
    constraints.fix_trace_padding(
        &mut col_eval_forms,
        trace.num_rows,
        domain_size as usize,
    );

    // Convert each column to coefficient form for commitment.
    let t_ifft = Instant::now();
    let column_coeffs_all: Vec<Vec<Scalar>> = col_eval_forms.par_iter()
        .map(|col| scheme.ifft(col, domain_size))
        .collect();
    let ifft_trace_ms = t_ifft.elapsed().as_millis();

    // Step 1: Commit to each trace column (coefficient form).
    let t_commit_trace = Instant::now();
    let column_commitments: Vec<Commitment> = column_coeffs_all.par_iter()
        .map(|coeffs| Commitment(scheme.commit_coefficients(coeffs)))
        .collect();
    let commit_trace_ms = t_commit_trace.elapsed().as_millis();

    // Step 2: Absorb commitments into Fiat-Shamir transcript.
    transcript.append_u64(b"num_steps", num_steps);
    transcript.append_u64(b"domain_size", domain_size);
    for comm in &column_commitments {
        transcript.append_message(b"column_commitment", &comm.0);
    }

    MainCommitState {
        num_steps,
        domain_size,
        col_eval_forms,
        column_coeffs_all,
        column_commitments,
        ifft_trace_ms,
        commit_trace_ms,
        prove_start,
    }
}

fn prove_inner_scheme(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    transcript: &mut Transcript,
    scheme: &dyn crate::scheme::CommitmentScheme,
) -> ExecutionProof {
    let phase1 = commit_main_columns_phase1(trace, constraints, transcript, scheme);
    prove_phase2_from_main_commit(trace, constraints, transcript, scheme, phase1)
}

/// Phase-2 of the prover: takes the post-commit state from
/// [`commit_main_columns_phase1`] and a transcript that has already had any
/// joint cross-AIR challenges absorbed, and produces the final
/// [`ExecutionProof`] (α derivation through batch openings).
///
/// The split lets the joint cross-AIR LogUp prover interleave its γ
/// derivation between phase-1 and phase-2: each AIR's main commitments are
/// gathered first, a shared γ is derived from the union, that γ is absorbed
/// into each AIR's transcript, and only then is each AIR's phase-2 run.
pub(crate) fn prove_phase2_from_main_commit(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    transcript: &mut Transcript,
    scheme: &dyn crate::scheme::CommitmentScheme,
    phase1: MainCommitState,
) -> ExecutionProof {
    use std::time::Instant;
    let curve = trace.curve;
    let prove_start = phase1.prove_start;
    let num_steps = phase1.num_steps;
    let domain_size = phase1.domain_size;
    let col_eval_forms = phase1.col_eval_forms;
    let column_coeffs_all = phase1.column_coeffs_all;
    let column_commitments = phase1.column_commitments;
    let ifft_trace_ms = phase1.ifft_trace_ms;
    let commit_trace_ms = phase1.commit_trace_ms;

    // -----------------------------------------------------------------------
    // Step 3: Draw constraint combination challenge alpha
    // -----------------------------------------------------------------------
    let alpha_bytes = transcript.challenge_bytes(b"alpha");
    let alpha = Scalar::from_challenge_bytes(&alpha_bytes, curve);

    // -----------------------------------------------------------------------
    // Step 3b: LogUp range check witness computation (extended Phase-0 layout).
    //
    // Commits the full `limbs | f | t | u_t | h | m` layout. The downstream
    // constraint stage (below) adds per-limb inverse, table inverse, and
    // running-sum transition constraints so the LogUp argument is sound.
    //
    // Domain requirement: the 8-bit range table needs `domain_size >= 256`.
    // Enforced by bumping `domain_size` above when `has_logup` is set.
    // -----------------------------------------------------------------------
    let lookup_reqs = constraints.lookup_declarations();
    let logup_groups = crate::lookup::group_declarations(&lookup_reqs);
    let ext_logup_layout = crate::lookup::extended_logup_column_layout(&logup_groups);
    let has_logup = !logup_groups.is_empty();

    let mut logup_commitments: Vec<Commitment> = Vec::new();
    let mut logup_column_coeffs: Vec<Vec<Scalar>> = Vec::new();
    let mut logup_gamma_opt: Option<Scalar> = None;
    let t_logup = Instant::now();

    if has_logup {
        let gamma_bytes = transcript.challenge_bytes(b"logup_gamma");
        let gamma = Scalar::from_challenge_bytes(&gamma_bytes, curve);
        logup_gamma_opt = Some(gamma.clone());

        let col_refs: Vec<&Vec<Scalar>> = col_eval_forms.iter().collect();
        let witness = crate::lookup::compute_extended_logup_witness(
            &col_refs, &logup_groups,
            &gamma, trace.num_rows, domain_size as usize, curve,
        );

        // Emit in the order fixed by `extended_logup_column_layout`:
        // limbs[..] | f[..] | t | u_t | h | m
        let mut logup_eval_forms: Vec<Vec<Scalar>> = Vec::new();
        for group_limbs in &witness.limb_columns {
            for limb_col in group_limbs {
                logup_eval_forms.push(limb_col.clone());
            }
        }
        for group_fs in &witness.f_columns {
            for f_col in group_fs {
                logup_eval_forms.push(f_col.clone());
            }
        }
        logup_eval_forms.push(witness.t_column.clone());
        logup_eval_forms.push(witness.u_t_column.clone());
        logup_eval_forms.push(witness.h_column.clone());
        logup_eval_forms.push(witness.m_column.clone());
        debug_assert_eq!(logup_eval_forms.len(), ext_logup_layout.num_columns);

        // Convert to coefficient form and commit (parallel)
        let (comms, coeffs_list): (Vec<_>, Vec<_>) = logup_eval_forms.par_iter()
            .map(|col| {
                let coeffs = scheme.ifft(col, domain_size);
                let comm = Commitment(scheme.commit_coefficients(&coeffs));
                (comm, coeffs)
            })
            .unzip();
        logup_commitments = comms;
        logup_column_coeffs = coeffs_list;

        // Absorb LogUp commitments
        for comm in &logup_commitments {
            transcript.append_message(b"logup_column_commitment", &comm.0);
        }
    }
    let logup_ms = t_logup.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 3b': Bitwise LogUp witness computation (nibble-AND table).
    //
    // Commits the extended `[per group: a_nibs | b_nibs | and_nibs | f_nibs]
    // | t_a | t_b | t_c | u_t | h | m` layout. Each per-nibble query
    // `q_k = a_k + δ·b_k + δ²·c_k` is inverted against γ; the 256-row
    // preprocessed nibble-AND table gets the same treatment with
    // `t_a(z)`/`t_b(z)`/`t_c(z)` bound to their canonical Lagrange forms
    // in the verifier.
    // -----------------------------------------------------------------------
    let bitwise_decls_vec = constraints.bitwise_lookup_declarations();
    let bitwise_groups = crate::lookup::group_bitwise_declarations(&bitwise_decls_vec);
    let ext_bitwise_layout =
        crate::lookup::extended_bitwise_column_layout(&bitwise_groups);
    let has_bitwise = !bitwise_groups.is_empty();

    let mut bitwise_commitments: Vec<Commitment> = Vec::new();
    let mut bitwise_column_coeffs: Vec<Vec<Scalar>> = Vec::new();
    let mut bitwise_gamma_opt: Option<Scalar> = None;
    let mut bitwise_delta_opt: Option<Scalar> = None;

    if has_bitwise {
        let bw_gamma_bytes = transcript.challenge_bytes(b"bitwise_gamma");
        let bw_gamma = Scalar::from_challenge_bytes(&bw_gamma_bytes, curve);
        let bw_delta_bytes = transcript.challenge_bytes(b"bitwise_delta");
        let bw_delta = Scalar::from_challenge_bytes(&bw_delta_bytes, curve);
        bitwise_gamma_opt = Some(bw_gamma.clone());
        bitwise_delta_opt = Some(bw_delta.clone());

        let col_refs: Vec<&Vec<Scalar>> = col_eval_forms.iter().collect();
        let bw_witness = crate::lookup::compute_extended_bitwise_witness(
            &col_refs, &bitwise_groups,
            &bw_gamma, &bw_delta, domain_size as usize, curve,
        );

        // Emit in the order fixed by `extended_bitwise_column_layout`:
        // [per group: a_nibs | b_nibs | and_nibs | f_nibs] | t_a | t_b | t_c | u_t | h | m
        let mut bitwise_eval_forms: Vec<Vec<Scalar>> = Vec::new();
        for (g_idx, group) in bitwise_groups.iter().enumerate() {
            let n = group.num_nibbles;
            for k in 0..n { bitwise_eval_forms.push(bw_witness.nibble_columns[g_idx][k].clone()); }
            for k in 0..n { bitwise_eval_forms.push(bw_witness.nibble_columns[g_idx][n + k].clone()); }
            for k in 0..n { bitwise_eval_forms.push(bw_witness.nibble_columns[g_idx][2 * n + k].clone()); }
            for k in 0..n { bitwise_eval_forms.push(bw_witness.f_columns[g_idx][k].clone()); }
        }
        bitwise_eval_forms.push(bw_witness.t_a_column.clone());
        bitwise_eval_forms.push(bw_witness.t_b_column.clone());
        bitwise_eval_forms.push(bw_witness.t_c_column.clone());
        bitwise_eval_forms.push(bw_witness.u_t_column.clone());
        bitwise_eval_forms.push(bw_witness.h_column.clone());
        bitwise_eval_forms.push(bw_witness.m_column.clone());
        debug_assert_eq!(
            bitwise_eval_forms.len(),
            ext_bitwise_layout.num_columns
        );

        let (comms, coeffs_list): (Vec<_>, Vec<_>) = bitwise_eval_forms.par_iter()
            .map(|col| {
                let coeffs = scheme.ifft(col, domain_size);
                let comm = Commitment(scheme.commit_coefficients(&coeffs));
                (comm, coeffs)
            })
            .unzip();
        bitwise_commitments = comms;
        bitwise_column_coeffs = coeffs_list;

        for comm in &bitwise_commitments {
            transcript.append_message(b"bitwise_column_commitment", &comm.0);
        }
    }

    // -----------------------------------------------------------------------
    // Step 3c: Memory permutation witness computation
    // -----------------------------------------------------------------------
    let t_mem_perm = Instant::now();
    let mem_cols = constraints.memory_columns();
    let has_perm = mem_cols.is_some();
    let mut perm_commitments: Vec<Commitment> = Vec::new();
    let mut perm_column_coeffs: Vec<Vec<Scalar>> = Vec::new();
    let mut perm_layout: Option<crate::permutation::MemoryPermutationLayout> = None;
    let mut perm_gamma_opt: Option<Scalar> = None;
    let mut perm_delta_opt: Option<Scalar> = None;

    if let Some((addr_col, val_cols, load_sels, store_sels)) = &mem_cols {
        // Draw permutation challenges
        let perm_gamma_bytes = transcript.challenge_bytes(b"perm_gamma");
        let perm_gamma = Scalar::from_challenge_bytes(&perm_gamma_bytes, curve);
        let perm_delta_bytes = transcript.challenge_bytes(b"perm_delta");
        let perm_delta = Scalar::from_challenge_bytes(&perm_delta_bytes, curve);
        perm_gamma_opt = Some(perm_gamma.clone());
        perm_delta_opt = Some(perm_delta.clone());

        let num_rows = trace.num_rows;
        let n = domain_size as usize;

        // Extract memory accesses from trace columns
        let mut accesses: Vec<crate::permutation::MemoryAccess> = Vec::with_capacity(n);
        for row in 0..n {
            let is_load = if row < num_rows { load_sels.iter().any(|&s| !col_eval_forms[s][row].is_zero()) } else { false };
            let is_store = if row < num_rows { store_sels.iter().any(|&s| !col_eval_forms[s][row].is_zero()) } else { false };

            let (addr, values, rw) = if is_load || is_store {
                let addr = col_eval_forms[*addr_col][row].to_u64();
                let values: Vec<u64> = val_cols.iter().map(|&vc| col_eval_forms[vc][row].to_u64()).collect();
                let rw = if is_store { 1u64 } else { 0u64 };
                (addr, values, rw)
            } else {
                // Non-memory row: dummy entry at a sentinel address that
                // can't collide with real EVM memory accesses. Using
                // u64::MAX ensures all dummies cluster together in the
                // sorted order, separate from any real memory address.
                // Without this, MSTORE/MSTORE8 writes to addr 0 would
                // be followed (in sorted order) by dummy "reads" at
                // addr 0 with value 0, breaking read-consistency. The
                // verifier matches this convention via the effective_addr
                // formula in evaluate_grand_product_at_point. Fix landed
                // 2026-05-12 (see mstore8_memory_perm_fix.md).
                let values: Vec<u64> = vec![0u64; val_cols.len()];
                (u64::MAX, values, 0u64)
            };

            accesses.push(crate::permutation::MemoryAccess {
                addr,
                values,
                timestamp: row as u64,
                rw,
            });
        }

        // Sort and compute auxiliary columns
        let (sorted, is_same_addr, inv_addr_diff) =
            crate::permutation::sort_and_compute_aux(&accesses, curve);
        let z_col = crate::permutation::compute_grand_product(
            &accesses, &sorted, &perm_gamma, &perm_delta, curve,
        );

        let layout = crate::permutation::MemoryPermutationLayout::new(val_cols.len());

        // Build evaluation-form columns for sorted trace + auxiliary
        let mut perm_eval_forms: Vec<Vec<Scalar>> = Vec::with_capacity(layout.num_columns);

        // sorted_addr
        let mut sorted_addr_col = vec![Scalar::zero(curve); n];
        for (i, a) in sorted.iter().enumerate() {
            sorted_addr_col[i] = Scalar::from_u64(a.addr, curve);
        }
        perm_eval_forms.push(sorted_addr_col);

        // sorted_val columns
        for v_idx in 0..val_cols.len() {
            let mut sorted_val_col = vec![Scalar::zero(curve); n];
            for (i, a) in sorted.iter().enumerate() {
                sorted_val_col[i] = Scalar::from_u64(a.values[v_idx], curve);
            }
            perm_eval_forms.push(sorted_val_col);
        }

        // sorted_ts
        let mut sorted_ts_col = vec![Scalar::zero(curve); n];
        for (i, a) in sorted.iter().enumerate() {
            sorted_ts_col[i] = Scalar::from_u64(a.timestamp, curve);
        }
        perm_eval_forms.push(sorted_ts_col);

        // sorted_rw
        let mut sorted_rw_col = vec![Scalar::zero(curve); n];
        for (i, a) in sorted.iter().enumerate() {
            sorted_rw_col[i] = Scalar::from_u64(a.rw, curve);
        }
        perm_eval_forms.push(sorted_rw_col);

        // Z column
        let mut z_eval_col = vec![Scalar::zero(curve); n];
        for (i, z_val) in z_col.iter().enumerate() {
            z_eval_col[i] = z_val.clone();
        }
        perm_eval_forms.push(z_eval_col);

        // is_same_addr
        let mut isa_col = vec![Scalar::zero(curve); n];
        for (i, v) in is_same_addr.iter().enumerate() {
            isa_col[i] = v.clone();
        }
        perm_eval_forms.push(isa_col);

        // inv_addr_diff
        let mut iad_col = vec![Scalar::zero(curve); n];
        for (i, v) in inv_addr_diff.iter().enumerate() {
            iad_col[i] = v.clone();
        }
        perm_eval_forms.push(iad_col);

        // original_ts: row index for each row (needed for grand product constraint)
        let mut orig_ts_col = vec![Scalar::zero(curve); n];
        for i in 0..n {
            orig_ts_col[i] = Scalar::from_u64(i as u64, curve);
        }
        perm_eval_forms.push(orig_ts_col);

        // Convert to coefficient form and commit (parallel)
        let (comms, coeffs_list): (Vec<_>, Vec<_>) = perm_eval_forms.par_iter()
            .map(|col| {
                let coeffs = scheme.ifft(col, domain_size);
                let comm = Commitment(scheme.commit_coefficients(&coeffs));
                (comm, coeffs)
            })
            .unzip();
        perm_commitments = comms;
        perm_column_coeffs = coeffs_list;

        // Absorb permutation commitments
        for comm in &perm_commitments {
            transcript.append_message(b"perm_column_commitment", &comm.0);
        }

        perm_layout = Some(layout);
    }
    let mem_perm_ms = t_mem_perm.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 3d: Oracle public-input data collection
    // -----------------------------------------------------------------------
    let oracle_sels = constraints.oracle_selectors();
    let mut oracle_data_entries: Vec<Vec<u8>> = Vec::new();
    if !oracle_sels.is_empty() {
        let num_data_cols = constraints.selector_column_indices().first().copied().unwrap_or(col_eval_forms.len());
        for row in 0..trace.num_rows {
            for &sel_col in &oracle_sels {
                if sel_col < col_eval_forms.len() && !col_eval_forms[sel_col][row].is_zero() {
                    let mut entry = Vec::with_capacity(8 + 8 + 8 * num_data_cols);
                    entry.extend_from_slice(&(row as u64).to_le_bytes());
                    entry.extend_from_slice(&(sel_col as u64).to_le_bytes());
                    for c in 0..num_data_cols {
                        entry.extend_from_slice(&col_eval_forms[c][row].to_u64().to_le_bytes());
                    }
                    oracle_data_entries.push(entry);
                }
            }
        }
        // Absorb oracle data into transcript (only if entries exist)
        if !oracle_data_entries.is_empty() {
            transcript.append_u64(b"num_oracle_entries", oracle_data_entries.len() as u64);
            for entry in &oracle_data_entries {
                transcript.append_message(b"oracle_entry", entry);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3e: Register file permutation witness computation
    // -----------------------------------------------------------------------
    let t_reg_perm = Instant::now();
    let reg_ports = constraints.register_ports();
    let has_reg_perm = !reg_ports.is_empty();
    let mut reg_perm_commitments_vec: Vec<Commitment> = Vec::new();
    let mut reg_perm_column_coeffs: Vec<Vec<Scalar>> = Vec::new();
    let mut reg_perm_layout: Option<crate::permutation::RegisterPermutationLayout> = None;
    let mut reg_perm_gamma_opt: Option<Scalar> = None;
    let mut reg_perm_delta_opt: Option<Scalar> = None;

    if has_reg_perm {
        let num_ports = reg_ports.len();
        let n = domain_size as usize;

        // Draw register permutation challenges
        let rg_bytes = transcript.challenge_bytes(b"reg_perm_gamma");
        let reg_gamma = Scalar::from_challenge_bytes(&rg_bytes, curve);
        let rd_bytes = transcript.challenge_bytes(b"reg_perm_delta");
        let reg_delta = Scalar::from_challenge_bytes(&rd_bytes, curve);
        reg_perm_gamma_opt = Some(reg_gamma.clone());
        reg_perm_delta_opt = Some(reg_delta.clone());

        // Collect all register accesses: num_ports entries per row, interleaved
        let mut all_accesses: Vec<crate::permutation::MemoryAccess> = Vec::with_capacity(num_ports * n);
        for row in 0..n {
            for (port_idx, &(reg_col, val_col, is_write)) in reg_ports.iter().enumerate() {
                let reg_num = if row < trace.num_rows {
                    col_eval_forms[reg_col][row].to_u64()
                } else { 0 };
                let val = if row < trace.num_rows {
                    col_eval_forms[val_col][row].to_u64()
                } else { 0 };
                let ts = (row * num_ports + port_idx) as u64;
                let rw = if is_write { 1u64 } else { 0u64 };
                all_accesses.push(crate::permutation::MemoryAccess {
                    addr: reg_num,
                    values: vec![val],
                    timestamp: ts,
                    rw,
                });
            }
        }

        // Sort all accesses by (register, timestamp)
        let (sorted, _is_same_flat, _inv_diff_flat) =
            crate::permutation::sort_and_compute_aux(&all_accesses, curve);
        let z_all = crate::permutation::compute_grand_product(
            &all_accesses, &sorted, &reg_gamma, &reg_delta, curve,
        );

        let layout = crate::permutation::RegisterPermutationLayout::new(num_ports);
        let mut reg_eval_forms: Vec<Vec<Scalar>> = Vec::with_capacity(layout.num_columns);

        // Sorted columns: num_ports lanes × 4 (reg, val, ts, rw), interleaved
        for lane in 0..num_ports {
            for field_fn in [
                |a: &crate::permutation::MemoryAccess| a.addr,
                |a: &crate::permutation::MemoryAccess| a.values[0],
                |a: &crate::permutation::MemoryAccess| a.timestamp,
                |a: &crate::permutation::MemoryAccess| a.rw,
            ] {
                let mut col = vec![Scalar::zero(curve); n];
                for row in 0..n {
                    let idx = row * num_ports + lane;
                    if idx < sorted.len() {
                        col[row] = Scalar::from_u64(field_fn(&sorted[idx]), curve);
                    }
                }
                reg_eval_forms.push(col);
            }
        }

        // Z accumulator: Z[row] = ∏_{j=0}^{row*P-1} numer_j / denom_j
        // z_all[k] = ∏_{j=0}^{k-1} numer_j/denom_j, so Z[row] = z_all[row * P]
        let mut z_eval = vec![Scalar::zero(curve); n];
        z_eval[0] = Scalar::one(curve); // z_all[0] = 1
        for row in 1..n {
            let idx = row * num_ports;
            if idx < z_all.len() {
                z_eval[row] = z_all[idx].clone();
            }
        }
        reg_eval_forms.push(z_eval);

        // is_same_reg (backward-looking): isa[row] = 1 if sorted[row] same addr as sorted[row-1].
        // inv_reg_diff (forward-looking): inv[row] = 1/(addr[row+1] - addr[row]) when they differ.
        // The constraint accesses isa(ω·X) (shifted) and inv(X) (unshifted), so at X=ω^r:
        //   isa[r+1] tells if rows r+1 and r have same addr
        //   inv[r] must be the inverse of the addr diff between rows r+1 and r
        // Layout expects all is_same_reg columns contiguously, then all inv_reg_diff columns.
        let mut isa_cols: Vec<Vec<Scalar>> = Vec::with_capacity(num_ports);
        let mut inv_cols: Vec<Vec<Scalar>> = Vec::with_capacity(num_ports);
        for lane in 0..num_ports {
            let mut isa_col = vec![Scalar::zero(curve); n];
            let mut inv_col = vec![Scalar::zero(curve); n];
            // is_same_reg: backward-looking (row vs row-1)
            for row in 1..n {
                let curr_idx = row * num_ports + lane;
                let prev_idx = (row - 1) * num_ports + lane;
                if curr_idx < sorted.len() && prev_idx < sorted.len() {
                    if sorted[curr_idx].addr == sorted[prev_idx].addr {
                        isa_col[row] = Scalar::one(curve);
                    }
                }
            }
            // inv_reg_diff: forward-looking (row vs row+1)
            for row in 0..n.saturating_sub(1) {
                let curr_idx = row * num_ports + lane;
                let next_idx = (row + 1) * num_ports + lane;
                if next_idx < sorted.len() && curr_idx < sorted.len() {
                    if sorted[next_idx].addr != sorted[curr_idx].addr {
                        let diff = Scalar::from_u64(
                            sorted[next_idx].addr.wrapping_sub(sorted[curr_idx].addr), curve,
                        );
                        inv_col[row] = diff.inverse();
                    }
                }
            }
            isa_cols.push(isa_col);
            inv_cols.push(inv_col);
        }
        for col in isa_cols {
            reg_eval_forms.push(col);
        }
        for col in inv_cols {
            reg_eval_forms.push(col);
        }

        // Row timestamp column: row_ts[i] = i
        let mut row_ts_col = vec![Scalar::zero(curve); n];
        for i in 0..n {
            row_ts_col[i] = Scalar::from_u64(i as u64, curve);
        }
        reg_eval_forms.push(row_ts_col);

        assert_eq!(reg_eval_forms.len(), layout.num_columns,
            "register permutation column count mismatch");

        // Convert to coefficient form and commit (parallel)
        let (comms, coeffs_list): (Vec<_>, Vec<_>) = reg_eval_forms.par_iter()
            .map(|col| {
                let coeffs = scheme.ifft(col, domain_size);
                let comm = Commitment(scheme.commit_coefficients(&coeffs));
                (comm, coeffs)
            })
            .unzip();
        reg_perm_commitments_vec = comms;
        reg_perm_column_coeffs = coeffs_list;

        // Absorb register permutation commitments
        for comm in &reg_perm_commitments_vec {
            transcript.append_message(b"reg_perm_column_commitment", &comm.0);
        }

        reg_perm_layout = Some(layout);
    }
    let reg_perm_ms = t_reg_perm.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 3f: Frame-stack LIFO permutation witness computation.
    //
    // For VMs with a nested-call frame stack (EVM), commit a single Z
    // accumulator column whose grand-product closure proves that the
    // multiset of pushed parent-frame tuples equals the multiset of
    // popped parent-frame tuples. See `permutation::FrameStackPermLayout`.
    // -----------------------------------------------------------------------
    let frame_perm_layout_opt = constraints.frame_perm_layout();
    let has_frame_perm = frame_perm_layout_opt.is_some();
    let mut frame_perm_z_coeffs: Vec<Scalar> = Vec::new();
    let mut frame_perm_commitment_opt: Option<Commitment> = None;
    let mut frame_perm_gamma_opt: Option<Scalar> = None;
    let mut frame_perm_delta_opt: Option<Scalar> = None;

    if let Some(ref fp_layout) = frame_perm_layout_opt {
        let n = domain_size as usize;

        // Draw frame-perm challenges γ, δ.
        let fpg_bytes = transcript.challenge_bytes(b"frame_perm_gamma");
        let fp_gamma = Scalar::from_challenge_bytes(&fpg_bytes, curve);
        let fpd_bytes = transcript.challenge_bytes(b"frame_perm_delta");
        let fp_delta = Scalar::from_challenge_bytes(&fpd_bytes, curve);
        frame_perm_gamma_opt = Some(fp_gamma.clone());
        frame_perm_delta_opt = Some(fp_delta.clone());

        // Pre-compute the per-row tuple values (push side) and the
        // shifted-tuple values (pop side, i.e. tuple at row+1).
        let mut tuple_per_row: Vec<Vec<Scalar>> = Vec::with_capacity(n);
        let mut tuple_shifted_per_row: Vec<Vec<Scalar>> = Vec::with_capacity(n);
        for row in 0..n {
            let next = (row + 1) % n;
            let mut here: Vec<Scalar> = Vec::with_capacity(fp_layout.tuple_columns.len());
            let mut there: Vec<Scalar> = Vec::with_capacity(fp_layout.tuple_columns.len());
            for &c in &fp_layout.tuple_columns {
                here.push(col_eval_forms[c][row].clone());
                there.push(col_eval_forms[c][next].clone());
            }
            tuple_per_row.push(here);
            tuple_shifted_per_row.push(there);
        }

        // Per-row is_push and is_pop (sums of selector evaluations,
        // guaranteed binary by selector sum-to-one on the AIR).
        let mut is_push_evals = vec![Scalar::zero(curve); n];
        let mut is_pop_evals = vec![Scalar::zero(curve); n];
        for row in 0..n {
            let mut p = Scalar::zero(curve);
            for &s in &fp_layout.push_selectors {
                p = p.add(&col_eval_forms[s][row]);
            }
            is_push_evals[row] = p;
            let mut q = Scalar::zero(curve);
            for &s in &fp_layout.pop_selectors {
                q = q.add(&col_eval_forms[s][row]);
            }
            is_pop_evals[row] = q;
        }

        // Compute the Z accumulator column (eval form).
        let z_eval = crate::permutation::compute_frame_perm_z(
            &tuple_per_row, &tuple_shifted_per_row,
            &is_push_evals, &is_pop_evals,
            &fp_gamma, &fp_delta, curve,
        );

        // IFFT and commit.
        let z_coeffs = scheme.ifft(&z_eval, domain_size);
        let z_comm = Commitment(scheme.commit_coefficients(&z_coeffs));
        frame_perm_z_coeffs = z_coeffs;
        frame_perm_commitment_opt = Some(z_comm.clone());

        // Absorb commit so subsequent challenges (z, β, ...) bind to it.
        transcript.append_message(b"frame_perm_column_commitment", &z_comm.0);
    }

    // -----------------------------------------------------------------------
    // Step 4: Build C(x) — either via build_constraint_polynomial (selector-based)
    //         or via evaluate_on_domain + IFFT (fallback)
    // -----------------------------------------------------------------------
    let t_constraints = Instant::now();
    let c_coeffs: Vec<Scalar>;
    // See the matching note in the older `prove_inner` above.
    let use_build_poly = true;

    // Compute domain generator ω for cross-row constraints (if needed)
    let shifted_indices = constraints.shifted_column_indices();
    let has_shifts = !shifted_indices.is_empty();
    let omega = if has_shifts || has_logup || has_bitwise || has_perm || has_reg_perm || has_frame_perm {
        Some(scheme.domain_generator(domain_size))
    } else {
        None
    };

    if use_build_poly {
        // Selector-based VMs: build C(x) algebraically in coefficient form
        let mut c_intra = constraints.build_constraint_polynomial(
            &column_coeffs_all, &alpha, domain_size,
        );

        // Track alpha offset for LogUp/permutation constraints
        let mut alpha_offset = constraints.num_constraints();

        // Add cross-row constraint polynomial if any
        if has_shifts {
            let c_shifted = constraints.build_shifted_constraint_polynomial(
                &column_coeffs_all, &alpha, domain_size,
                omega.as_ref().unwrap(),
                alpha_offset,
            );
            c_intra = crate::poly_arith::poly_add(&c_intra, &c_shifted, curve);
            alpha_offset += constraints.num_shifted_constraints();
        }

        // Add LogUp decomposition constraints to C(x)
        if has_logup {
            // Compute alpha^alpha_offset
            let mut ap = Scalar::one(curve);
            for _ in 0..alpha_offset {
                ap = ap.mul(&alpha);
            }

            let lookup_reqs = constraints.lookup_declarations();
            let logup_groups_for_cx = crate::lookup::group_declarations(&lookup_reqs);
            let logup_layout_for_cx = crate::lookup::extended_logup_column_layout(&logup_groups_for_cx);

            // For each group: build sel(X) * (value(X) - Σ limb_k(X) * 256^k)
            for (g_idx, group) in logup_groups_for_cx.iter().enumerate() {
                let (start_offset, num_limbs) = logup_layout_for_cx.limb_offsets[g_idx];

                // Recompose: Σ limb_k(X) * 256^k in coefficient form
                let two56 = Scalar::from_u64(256, curve);
                let mut recomposed = vec![Scalar::zero(curve); domain_size as usize];
                let mut power = Scalar::one(curve);

                for l in 0..num_limbs {
                    let limb_coeffs = &logup_column_coeffs[start_offset + l];
                    let scaled = crate::poly_arith::poly_scalar_mul(limb_coeffs, &power);
                    recomposed = crate::poly_arith::poly_add(&recomposed, &scaled, curve);
                    power = power.mul(&two56);
                }

                // diff(X) = value(X) - recomposed(X)
                let value_coeffs = &column_coeffs_all[group.column_index];
                let diff = crate::poly_arith::poly_sub(value_coeffs, &recomposed, curve);

                // Gate by selector: combined_sel(X) = Σ sel_k(X)
                let body = if !group.selectors.is_empty() {
                    let mut sel_combined = vec![Scalar::zero(curve); domain_size as usize];
                    for &sel_idx in &group.selectors {
                        sel_combined = crate::poly_arith::poly_add(
                            &sel_combined, &column_coeffs_all[sel_idx], curve,
                        );
                    }
                    // body = sel_combined(X) * diff(X)
                    crate::poly_arith::poly_mul(&sel_combined, &diff, curve)
                } else {
                    diff
                };

                // Accumulate: alpha^k * body(X)
                let scaled_body = crate::poly_arith::poly_scalar_mul(&body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_body, curve);
                ap = ap.mul(&alpha);
            }

            alpha_offset += logup_groups_for_cx.len();

            let h_idx = logup_layout_for_cx.h_column;
            let m_idx = logup_layout_for_cx.m_column;
            let t_idx = logup_layout_for_cx.t_column;
            let u_t_idx = logup_layout_for_cx.u_t_column;

            // Running sum boundary: L_0(X) · h(X) = 0 (ensures h(ω^0) = 0).
            if h_idx < logup_column_coeffs.len() {
                let n = domain_size as usize;
                let n_inv = Scalar::from_u64(n as u64, curve).inverse();
                let l0_coeffs: Vec<Scalar> = vec![n_inv; n];
                let h_coeffs = &logup_column_coeffs[h_idx];
                let body_h_boundary = crate::poly_arith::poly_mul(&l0_coeffs, h_coeffs, curve);
                let scaled_h_boundary = crate::poly_arith::poly_scalar_mul(&body_h_boundary, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_h_boundary, curve);
                ap = ap.mul(&alpha);
                alpha_offset += 1;
            }

            // Phase-0 extended constraints: per-limb inverse, table inverse,
            // running-sum transition. Constraint count matches
            // `num_extended_logup_constraints(logup_groups)`.
            let gamma = logup_gamma_opt.clone()
                .expect("logup gamma must be set when has_logup");
            let n = domain_size as usize;
            let gamma_poly: Vec<Scalar> = {
                let mut v = vec![Scalar::zero(curve); n];
                v[0] = gamma.clone();
                v
            };
            let one_poly: Vec<Scalar> = {
                let mut v = vec![Scalar::zero(curve); n];
                v[0] = Scalar::one(curve);
                v
            };

            // (A) f_k(X) · (γ - ℓ_k(X)) - 1 = 0 for each limb across all groups.
            for (g_idx, _group) in logup_groups_for_cx.iter().enumerate() {
                let (limb_start, num_limbs) = logup_layout_for_cx.limb_offsets[g_idx];
                let (f_start, _) = logup_layout_for_cx.f_offsets[g_idx];
                for l in 0..num_limbs {
                    let limb_coeffs = &logup_column_coeffs[limb_start + l];
                    let f_coeffs = &logup_column_coeffs[f_start + l];
                    let gamma_minus_l = crate::poly_arith::poly_sub(&gamma_poly, limb_coeffs, curve);
                    let prod = crate::poly_arith::poly_mul(f_coeffs, &gamma_minus_l, curve);
                    let body = crate::poly_arith::poly_sub(&prod, &one_poly, curve);
                    let scaled = crate::poly_arith::poly_scalar_mul(&body, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                    ap = ap.mul(&alpha);
                    alpha_offset += 1;
                }
            }

            // (B) u_t(X) · (γ - t(X)) - 1 = 0
            if u_t_idx < logup_column_coeffs.len() && t_idx < logup_column_coeffs.len() {
                let t_coeffs = &logup_column_coeffs[t_idx];
                let u_t_coeffs = &logup_column_coeffs[u_t_idx];
                let gamma_minus_t = crate::poly_arith::poly_sub(&gamma_poly, t_coeffs, curve);
                let prod = crate::poly_arith::poly_mul(u_t_coeffs, &gamma_minus_t, curve);
                let body = crate::poly_arith::poly_sub(&prod, &one_poly, curve);
                let scaled = crate::poly_arith::poly_scalar_mul(&body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                ap = ap.mul(&alpha);
                alpha_offset += 1;
            }

            // (C) Cyclic running-sum transition:
            // h(ωX) - h(X) - Σ_{g,k} active_g(X)·f_{g,k}(X) + m(X)·u_t(X) = 0
            if h_idx < logup_column_coeffs.len() {
                let om = omega.as_ref().expect("omega required for transition");
                let h_coeffs = &logup_column_coeffs[h_idx];
                let m_coeffs = &logup_column_coeffs[m_idx];
                let u_t_coeffs = &logup_column_coeffs[u_t_idx];
                let h_shifted = crate::poly_arith::poly_shift(h_coeffs, om);
                let mut body = crate::poly_arith::poly_sub(&h_shifted, h_coeffs, curve);
                for (g_idx, group) in logup_groups_for_cx.iter().enumerate() {
                    let (f_start, num_limbs) = logup_layout_for_cx.f_offsets[g_idx];
                    let active_coeffs: Vec<Scalar> = if group.selectors.is_empty() {
                        one_poly.clone()
                    } else {
                        let mut s = vec![Scalar::zero(curve); n];
                        for &sel_idx in &group.selectors {
                            s = crate::poly_arith::poly_add(&s, &column_coeffs_all[sel_idx], curve);
                        }
                        s
                    };
                    for l in 0..num_limbs {
                        let f_coeffs = &logup_column_coeffs[f_start + l];
                        let term = crate::poly_arith::poly_mul(&active_coeffs, f_coeffs, curve);
                        body = crate::poly_arith::poly_sub(&body, &term, curve);
                    }
                }
                let mu_term = crate::poly_arith::poly_mul(m_coeffs, u_t_coeffs, curve);
                body = crate::poly_arith::poly_add(&body, &mu_term, curve);
                let scaled = crate::poly_arith::poly_scalar_mul(&body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                #[allow(unused_assignments)] {
                    ap = ap.mul(&alpha);
                }
                alpha_offset += 1;
            }
        }

        // Add bitwise LogUp constraints to C(x)
        if has_bitwise {
            let mut ap = Scalar::one(curve);
            for _ in 0..alpha_offset {
                ap = ap.mul(&alpha);
            }

            let bw_gamma = bitwise_gamma_opt.clone()
                .expect("bitwise gamma must be set when has_bitwise");
            let bw_delta = bitwise_delta_opt.clone()
                .expect("bitwise delta must be set when has_bitwise");
            let bw_delta_sq = bw_delta.mul(&bw_delta);
            let n = domain_size as usize;
            let one_poly: Vec<Scalar> = {
                let mut v = vec![Scalar::zero(curve); n];
                v[0] = Scalar::one(curve);
                v
            };
            let bw_gamma_poly: Vec<Scalar> = {
                let mut v = vec![Scalar::zero(curve); n];
                v[0] = bw_gamma.clone();
                v
            };

            // Pre-compute powers of 16 for nibble recomposition.
            let sixteen = Scalar::from_u64(16, curve);

            // (BW-A) Per-group nibble decomposition + result derivation
            for (g_idx, group) in bitwise_groups.iter().enumerate() {
                let (nib_start, num_nibs) = ext_bitwise_layout.nibble_offsets[g_idx];
                let a_start = nib_start;
                let b_start = nib_start + num_nibs;
                let c_start = nib_start + 2 * num_nibs;

                let combined_sel_coeffs: Vec<Scalar> = if group.selectors.is_empty() {
                    one_poly.clone()
                } else {
                    let mut s = vec![Scalar::zero(curve); n];
                    for &sel_idx in &group.selectors {
                        s = crate::poly_arith::poly_add(
                            &s, &column_coeffs_all[sel_idx], curve,
                        );
                    }
                    s
                };

                // Recompose a_nibbles → a_recomposed, b_nibbles → b_recomposed,
                // and_nibbles → c_recomposed (all in coefficient form).
                let recompose = |start: usize| -> Vec<Scalar> {
                    let mut acc = vec![Scalar::zero(curve); n];
                    let mut power = Scalar::one(curve);
                    for k in 0..num_nibs {
                        let nib_coeffs = &bitwise_column_coeffs[start + k];
                        let scaled = crate::poly_arith::poly_scalar_mul(nib_coeffs, &power);
                        acc = crate::poly_arith::poly_add(&acc, &scaled, curve);
                        power = power.mul(&sixteen);
                    }
                    acc
                };
                let a_recomposed = recompose(a_start);
                let b_recomposed = recompose(b_start);
                let c_recomposed = recompose(c_start);

                // Constraint: sel · (operand_a - a_recomposed) = 0
                let a_operand_coeffs = &column_coeffs_all[group.operand_a_column];
                let a_diff = crate::poly_arith::poly_sub(a_operand_coeffs, &a_recomposed, curve);
                let a_body = crate::poly_arith::poly_mul(&combined_sel_coeffs, &a_diff, curve);
                let scaled = crate::poly_arith::poly_scalar_mul(&a_body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                ap = ap.mul(&alpha);
                alpha_offset += 1;

                // Constraint: sel · (operand_b - b_recomposed) = 0
                let b_operand_coeffs = &column_coeffs_all[group.operand_b_column];
                let b_diff = crate::poly_arith::poly_sub(b_operand_coeffs, &b_recomposed, curve);
                let b_body = crate::poly_arith::poly_mul(&combined_sel_coeffs, &b_diff, curve);
                let scaled = crate::poly_arith::poly_scalar_mul(&b_body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                ap = ap.mul(&alpha);
                alpha_offset += 1;

                // Result derivation depending on op
                // AND: result = c_recomposed
                // OR:  result = a + b - c_recomposed
                // XOR: result = a + b - 2·c_recomposed
                let expected: Vec<Scalar> = match group.op {
                    crate::lookup::BitwiseOp::And => c_recomposed.clone(),
                    crate::lookup::BitwiseOp::Or => {
                        let ab = crate::poly_arith::poly_add(a_operand_coeffs, b_operand_coeffs, curve);
                        crate::poly_arith::poly_sub(&ab, &c_recomposed, curve)
                    }
                    crate::lookup::BitwiseOp::Xor => {
                        let ab = crate::poly_arith::poly_add(a_operand_coeffs, b_operand_coeffs, curve);
                        let two_c = crate::poly_arith::poly_scalar_mul(&c_recomposed, &Scalar::from_u64(2, curve));
                        crate::poly_arith::poly_sub(&ab, &two_c, curve)
                    }
                };
                let result_coeffs = &column_coeffs_all[group.result_column];
                let r_diff = crate::poly_arith::poly_sub(result_coeffs, &expected, curve);
                let r_body = crate::poly_arith::poly_mul(&combined_sel_coeffs, &r_diff, curve);
                let scaled = crate::poly_arith::poly_scalar_mul(&r_body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                ap = ap.mul(&alpha);
                alpha_offset += 1;
            }

            // (BW-B) Per-nibble-position inverse constraint:
            // f_k · (γ - (a_k + δ·b_k + δ²·c_k)) - 1 = 0
            for (g_idx, group) in bitwise_groups.iter().enumerate() {
                let (nib_start, num_nibs) = ext_bitwise_layout.nibble_offsets[g_idx];
                let (f_start, _) = ext_bitwise_layout.f_offsets[g_idx];
                for k in 0..num_nibs {
                    let a_coeffs = &bitwise_column_coeffs[nib_start + k];
                    let b_coeffs = &bitwise_column_coeffs[nib_start + num_nibs + k];
                    let c_coeffs = &bitwise_column_coeffs[nib_start + 2 * num_nibs + k];
                    let f_coeffs = &bitwise_column_coeffs[f_start + k];
                    // q = a + δ·b + δ²·c
                    let b_scaled = crate::poly_arith::poly_scalar_mul(b_coeffs, &bw_delta);
                    let c_scaled = crate::poly_arith::poly_scalar_mul(c_coeffs, &bw_delta_sq);
                    let ab = crate::poly_arith::poly_add(a_coeffs, &b_scaled, curve);
                    let q = crate::poly_arith::poly_add(&ab, &c_scaled, curve);
                    let gamma_minus_q = crate::poly_arith::poly_sub(&bw_gamma_poly, &q, curve);
                    let prod = crate::poly_arith::poly_mul(f_coeffs, &gamma_minus_q, curve);
                    let body = crate::poly_arith::poly_sub(&prod, &one_poly, curve);
                    let scaled = crate::poly_arith::poly_scalar_mul(&body, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                    ap = ap.mul(&alpha);
                    alpha_offset += 1;
                }
                let _ = group; // silence unused
            }

            // (BW-C) Table inverse: u_t · (γ - (t_a + δ·t_b + δ²·t_c)) - 1 = 0
            let t_a_coeffs = &bitwise_column_coeffs[ext_bitwise_layout.t_a_column];
            let t_b_coeffs = &bitwise_column_coeffs[ext_bitwise_layout.t_b_column];
            let t_c_coeffs = &bitwise_column_coeffs[ext_bitwise_layout.t_c_column];
            let u_t_bw_coeffs = &bitwise_column_coeffs[ext_bitwise_layout.u_t_column];
            {
                let t_b_scaled = crate::poly_arith::poly_scalar_mul(t_b_coeffs, &bw_delta);
                let t_c_scaled = crate::poly_arith::poly_scalar_mul(t_c_coeffs, &bw_delta_sq);
                let t_ab = crate::poly_arith::poly_add(t_a_coeffs, &t_b_scaled, curve);
                let t_combined = crate::poly_arith::poly_add(&t_ab, &t_c_scaled, curve);
                let gamma_minus_t = crate::poly_arith::poly_sub(&bw_gamma_poly, &t_combined, curve);
                let prod = crate::poly_arith::poly_mul(u_t_bw_coeffs, &gamma_minus_t, curve);
                let body = crate::poly_arith::poly_sub(&prod, &one_poly, curve);
                let scaled = crate::poly_arith::poly_scalar_mul(&body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                ap = ap.mul(&alpha);
                alpha_offset += 1;
            }

            // (BW-D) Running-sum transition (cyclic):
            // h(ωX) - h(X) - Σ_{g,k} active_g · f_{g,k} + m · u_t = 0
            let h_bw_coeffs = &bitwise_column_coeffs[ext_bitwise_layout.h_column];
            let m_bw_coeffs = &bitwise_column_coeffs[ext_bitwise_layout.m_column];
            {
                let om = omega.as_ref().expect("omega required for bitwise transition");
                let h_shifted = crate::poly_arith::poly_shift(h_bw_coeffs, om);
                let mut body = crate::poly_arith::poly_sub(&h_shifted, h_bw_coeffs, curve);
                for (g_idx, group) in bitwise_groups.iter().enumerate() {
                    let (f_start, num_nibs) = ext_bitwise_layout.f_offsets[g_idx];
                    let active_coeffs: Vec<Scalar> = if group.selectors.is_empty() {
                        one_poly.clone()
                    } else {
                        let mut s = vec![Scalar::zero(curve); n];
                        for &sel_idx in &group.selectors {
                            s = crate::poly_arith::poly_add(
                                &s, &column_coeffs_all[sel_idx], curve,
                            );
                        }
                        s
                    };
                    for k in 0..num_nibs {
                        let f_coeffs = &bitwise_column_coeffs[f_start + k];
                        let term = crate::poly_arith::poly_mul(&active_coeffs, f_coeffs, curve);
                        body = crate::poly_arith::poly_sub(&body, &term, curve);
                    }
                }
                let mu_term = crate::poly_arith::poly_mul(m_bw_coeffs, u_t_bw_coeffs, curve);
                body = crate::poly_arith::poly_add(&body, &mu_term, curve);
                let scaled = crate::poly_arith::poly_scalar_mul(&body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                ap = ap.mul(&alpha);
                alpha_offset += 1;
            }

            // (BW-E) Boundary: L_0(X) · h(X) = 0
            {
                let n_inv = Scalar::from_u64(n as u64, curve).inverse();
                let l0_coeffs: Vec<Scalar> = vec![n_inv; n];
                let body = crate::poly_arith::poly_mul(&l0_coeffs, h_bw_coeffs, curve);
                let scaled = crate::poly_arith::poly_scalar_mul(&body, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled, curve);
                #[allow(unused_assignments)] {
                    ap = ap.mul(&alpha);
                }
                alpha_offset += 1;
            }
        }

        // Add memory permutation constraints to C(x)
        if has_perm {
            if let Some(ref layout) = perm_layout {
                // Compute alpha^alpha_offset
                let mut ap = Scalar::one(curve);
                for _ in 0..alpha_offset {
                    ap = ap.mul(&alpha);
                }

                let om = omega.as_ref().unwrap();

                // Compute ω^{n-1}
                let n_minus_1 = domain_size - 1;
                let mut omega_n_minus_1 = Scalar::one(curve);
                let mut base_o = om.clone();
                let mut exp_o = n_minus_1;
                while exp_o > 0 {
                    if exp_o & 1 == 1 {
                        omega_n_minus_1 = omega_n_minus_1.mul(&base_o);
                    }
                    base_o = base_o.mul(&base_o);
                    exp_o >>= 1;
                }

                // Constraint 1: is_same_addr * (is_same_addr - 1) = 0
                let isa_coeffs = &perm_column_coeffs[layout.is_same_addr];
                let one_poly = {
                    let mut p = vec![Scalar::zero(curve); domain_size as usize];
                    p[0] = Scalar::one(curve);
                    p
                };
                let isa_minus_one = crate::poly_arith::poly_sub(isa_coeffs, &one_poly, curve);
                let body1 = crate::poly_arith::poly_mul(isa_coeffs, &isa_minus_one, curve);
                let scaled1 = crate::poly_arith::poly_scalar_mul(&body1, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled1, curve);
                ap = ap.mul(&alpha);

                // Exclusion factor polynomial: (X - ω^{n-1})
                // Cross-row constraints are multiplied by this to exclude wrap-around

                // Constraint 2: is_same_addr(ω·X) * (sorted_addr(ω·X) - sorted_addr(X)) = 0
                // Multiply by (X - ω^{n-1})
                let isa_shifted = crate::poly_arith::poly_shift(isa_coeffs, om);
                let sa_coeffs = &perm_column_coeffs[layout.sorted_addr];
                let sa_shifted = crate::poly_arith::poly_shift(sa_coeffs, om);
                let addr_diff = crate::poly_arith::poly_sub(&sa_shifted, sa_coeffs, curve);
                let body2_inner = crate::poly_arith::poly_mul(&isa_shifted, &addr_diff, curve);
                let body2 = crate::poly_arith::poly_mul_linear(&body2_inner, &omega_n_minus_1);
                let scaled2 = crate::poly_arith::poly_scalar_mul(&body2, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled2, curve);
                ap = ap.mul(&alpha);

                // Constraint 3: (1 - is_same_addr(ω·X)) * (1 - addr_diff · inv_addr_diff(ω·X)) = 0
                // Multiply by (X - ω^{n-1})
                // BUG FIX 2026-05-12: use SHIFTED inv_addr_diff so both
                // flags reference the same (X → ω·X) transition. See
                // memory/mstore8_memory_perm_fix.md.
                let one_minus_isa_shifted = crate::poly_arith::poly_sub(&one_poly, &isa_shifted, curve);
                let inv_diff_coeffs = &perm_column_coeffs[layout.inv_addr_diff];
                let inv_diff_shifted = crate::poly_arith::poly_shift(inv_diff_coeffs, om);
                let addr_diff_inv_prod = crate::poly_arith::poly_mul(&addr_diff, &inv_diff_shifted, curve);
                let one_minus_prod = crate::poly_arith::poly_sub(&one_poly, &addr_diff_inv_prod, curve);
                let body3_inner = crate::poly_arith::poly_mul(&one_minus_isa_shifted, &one_minus_prod, curve);
                let body3 = crate::poly_arith::poly_mul_linear(&body3_inner, &omega_n_minus_1);
                let scaled3 = crate::poly_arith::poly_scalar_mul(&body3, &ap);
                c_intra = crate::poly_arith::poly_add(&c_intra, &scaled3, curve);
                ap = ap.mul(&alpha);

                // Constraint 4: Read consistency per value column
                // is_same_addr(ω·X) * (1 - sorted_rw(ω·X)) * (sorted_val_k(ω·X) - sorted_val_k(X)) = 0
                let srw_coeffs = &perm_column_coeffs[layout.sorted_rw];
                let srw_shifted = crate::poly_arith::poly_shift(srw_coeffs, om);
                let one_minus_srw_shifted = crate::poly_arith::poly_sub(&one_poly, &srw_shifted, curve);
                let isa_times_rw = crate::poly_arith::poly_mul(&isa_shifted, &one_minus_srw_shifted, curve);

                for v in 0..layout.num_val_columns {
                    let sv_coeffs = &perm_column_coeffs[layout.sorted_val_start + v];
                    let sv_shifted = crate::poly_arith::poly_shift(sv_coeffs, om);
                    let val_diff = crate::poly_arith::poly_sub(&sv_shifted, sv_coeffs, curve);
                    let body4_inner = crate::poly_arith::poly_mul(&isa_times_rw, &val_diff, curve);
                    let body4 = crate::poly_arith::poly_mul_linear(&body4_inner, &omega_n_minus_1);
                    let scaled4 = crate::poly_arith::poly_scalar_mul(&body4, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled4, curve);
                    ap = ap.mul(&alpha);
                }

                // Constraint 5: Grand product transition
                // Z(ω·X) · denom(X) - Z(X) · numer(X) = 0
                // numer(X) = γ + addr(X) + δ·val(X) + δ²·ts(X) + δ³·rw(X)  (original trace)
                // denom(X) = γ + sorted_addr(X) + δ·sorted_val(X) + δ²·sorted_ts(X) + δ³·sorted_rw(X)
                if let (Some((addr_col, val_cols, load_sel, store_sel)), Some(ref pg), Some(ref pd)) =
                    (&mem_cols, &perm_gamma_opt, &perm_delta_opt)
                {
                    let delta2 = pd.mul(pd);
                    let delta3 = delta2.mul(pd);

                    // Build numer(X) polynomial
                    let gamma_poly = {
                        let mut p = vec![Scalar::zero(curve); domain_size as usize];
                        p[0] = pg.clone();
                        p
                    };
                    let mut numer_poly = gamma_poly.clone();
                    // effective_addr(X) = real_mem(X) · addr(X) + (1 - real_mem(X)) · SENTINEL
                    // real_mem(X) = Σ load_sels(X) + Σ store_sels(X) — binary
                    // (selectors are mutually exclusive). Matches the prover's
                    // u64::MAX dummy convention and the verifier's at-point
                    // reconstruction. See mstore8_memory_perm_fix.md.
                    let mut real_mem_poly: Vec<Scalar> = vec![Scalar::zero(curve)];
                    for &ls in load_sel {
                        real_mem_poly = crate::poly_arith::poly_add(&real_mem_poly, &column_coeffs_all[ls], curve);
                    }
                    for &ss in store_sel {
                        real_mem_poly = crate::poly_arith::poly_add(&real_mem_poly, &column_coeffs_all[ss], curve);
                    }
                    let sentinel_scalar = Scalar::from_u64(u64::MAX, curve);
                    let sentinel_poly = {
                        let mut p = vec![Scalar::zero(curve); domain_size as usize];
                        p[0] = sentinel_scalar;
                        p
                    };
                    let one_poly_const = {
                        let mut p = vec![Scalar::zero(curve); domain_size as usize];
                        p[0] = Scalar::one(curve);
                        p
                    };
                    let one_minus_real_mem = crate::poly_arith::poly_sub(&one_poly_const, &real_mem_poly, curve);
                    let real_addr_term = crate::poly_arith::poly_mul(&real_mem_poly, &column_coeffs_all[*addr_col], curve);
                    let sentinel_term = crate::poly_arith::poly_mul(&one_minus_real_mem, &sentinel_poly, curve);
                    let effective_addr_poly = crate::poly_arith::poly_add(&real_addr_term, &sentinel_term, curve);
                    numer_poly = crate::poly_arith::poly_add(&numer_poly, &effective_addr_poly, curve);

                    // δ · val_combined(X)
                    if val_cols.len() == 1 {
                        let scaled = crate::poly_arith::poly_scalar_mul(&column_coeffs_all[val_cols[0]], pd);
                        numer_poly = crate::poly_arith::poly_add(&numer_poly, &scaled, curve);
                    } else {
                        let delta4 = delta2.mul(&delta2);
                        let mut dp = pd.clone();
                        for (v_idx, &vc) in val_cols.iter().enumerate() {
                            let scaled = crate::poly_arith::poly_scalar_mul(&column_coeffs_all[vc], &dp);
                            numer_poly = crate::poly_arith::poly_add(&numer_poly, &scaled, curve);
                            if v_idx < val_cols.len() - 1 {
                                dp = dp.mul(&delta4);
                            }
                        }
                    }

                    // δ² · original_ts(X)
                    let ts_coeffs = &perm_column_coeffs[layout.original_ts];
                    let scaled_ts = crate::poly_arith::poly_scalar_mul(ts_coeffs, &delta2);
                    numer_poly = crate::poly_arith::poly_add(&numer_poly, &scaled_ts, curve);

                    // δ³ · rw(X) = sum of store_sel columns
                    let mut rw_coeffs = vec![Scalar::zero(curve)];
                    for &ss in store_sel {
                        rw_coeffs = crate::poly_arith::poly_add(&rw_coeffs, &column_coeffs_all[ss], curve);
                    }
                    let scaled_rw = crate::poly_arith::poly_scalar_mul(&rw_coeffs, &delta3);
                    numer_poly = crate::poly_arith::poly_add(&numer_poly, &scaled_rw, curve);

                    // Build denom(X) polynomial from sorted columns
                    let mut denom_poly = gamma_poly;
                    denom_poly = crate::poly_arith::poly_add(
                        &denom_poly, &perm_column_coeffs[layout.sorted_addr], curve,
                    );

                    if layout.num_val_columns == 1 {
                        let scaled = crate::poly_arith::poly_scalar_mul(
                            &perm_column_coeffs[layout.sorted_val_start], pd,
                        );
                        denom_poly = crate::poly_arith::poly_add(&denom_poly, &scaled, curve);
                    } else {
                        let delta4 = delta2.mul(&delta2);
                        let mut dp = pd.clone();
                        for v in 0..layout.num_val_columns {
                            let scaled = crate::poly_arith::poly_scalar_mul(
                                &perm_column_coeffs[layout.sorted_val_start + v], &dp,
                            );
                            denom_poly = crate::poly_arith::poly_add(&denom_poly, &scaled, curve);
                            if v < layout.num_val_columns - 1 {
                                dp = dp.mul(&delta4);
                            }
                        }
                    }

                    let sorted_ts_scaled = crate::poly_arith::poly_scalar_mul(
                        &perm_column_coeffs[layout.sorted_ts], &delta2,
                    );
                    denom_poly = crate::poly_arith::poly_add(&denom_poly, &sorted_ts_scaled, curve);
                    let sorted_rw_scaled = crate::poly_arith::poly_scalar_mul(
                        &perm_column_coeffs[layout.sorted_rw], &delta3,
                    );
                    denom_poly = crate::poly_arith::poly_add(&denom_poly, &sorted_rw_scaled, curve);

                    // Z(ω·X) · denom(X) - Z(X) · numer(X)
                    let z_coeffs = &perm_column_coeffs[layout.z_column];
                    let z_shifted = crate::poly_arith::poly_shift(z_coeffs, om);
                    let term1 = crate::poly_arith::poly_mul(&z_shifted, &denom_poly, curve);
                    let term2 = crate::poly_arith::poly_mul(z_coeffs, &numer_poly, curve);
                    let body_gp = crate::poly_arith::poly_sub(&term1, &term2, curve);
                    let scaled_gp = crate::poly_arith::poly_scalar_mul(&body_gp, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_gp, curve);
                    ap = ap.mul(&alpha);

                    // Constraint 6: Boundary Z(1) = 1
                    // L_0(X) · (Z(X) - 1) = 0, where L_0(X) = (1/n) · Σ X^k
                    let n = domain_size as usize;
                    let n_inv = Scalar::from_u64(n as u64, curve).inverse();
                    let l0_coeffs: Vec<Scalar> = vec![n_inv; n];
                    let z_minus_one = crate::poly_arith::poly_sub(z_coeffs, &one_poly, curve);
                    let body_boundary = crate::poly_arith::poly_mul(&l0_coeffs, &z_minus_one, curve);
                    let scaled_boundary = crate::poly_arith::poly_scalar_mul(&body_boundary, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_boundary, curve);
                    #[allow(unused_assignments)] {
                        ap = ap.mul(&alpha);
                    }
                }

                alpha_offset += crate::permutation::num_permutation_constraints(layout);
            }
        }

        // Add register file permutation constraints to C(x)
        if has_reg_perm {
            if let Some(ref layout) = reg_perm_layout {
                let mut ap = Scalar::one(curve);
                for _ in 0..alpha_offset {
                    ap = ap.mul(&alpha);
                }

                let om = omega.as_ref().unwrap();
                let num_ports = layout.num_ports;

                // Compute ω^{n-1}
                let n_minus_1 = domain_size - 1;
                let mut omega_n_minus_1 = Scalar::one(curve);
                let mut base_o = om.clone();
                let mut exp_o = n_minus_1;
                while exp_o > 0 {
                    if exp_o & 1 == 1 {
                        omega_n_minus_1 = omega_n_minus_1.mul(&base_o);
                    }
                    base_o = base_o.mul(&base_o);
                    exp_o >>= 1;
                }

                let one_poly = {
                    let mut p = vec![Scalar::zero(curve); domain_size as usize];
                    p[0] = Scalar::one(curve);
                    p
                };

                // Per-lane sorted consistency constraints (4 per lane)
                for lane in 0..num_ports {
                    let isa_coeffs = &reg_perm_column_coeffs[layout.is_same_reg_start + lane];
                    let inv_coeffs = &reg_perm_column_coeffs[layout.inv_reg_diff_start + lane];
                    let sr_coeffs = &reg_perm_column_coeffs[layout.sorted_reg(lane)];
                    let sv_coeffs = &reg_perm_column_coeffs[layout.sorted_val(lane)];
                    let srw_coeffs = &reg_perm_column_coeffs[layout.sorted_rw(lane)];

                    // 1. is_same_reg binary: isa * (isa - 1) = 0
                    let isa_minus_one = crate::poly_arith::poly_sub(isa_coeffs, &one_poly, curve);
                    let body_bin = crate::poly_arith::poly_mul(isa_coeffs, &isa_minus_one, curve);
                    let scaled_bin = crate::poly_arith::poly_scalar_mul(&body_bin, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_bin, curve);
                    ap = ap.mul(&alpha);

                    // Shifted versions for cross-row constraints
                    let isa_shifted = crate::poly_arith::poly_shift(isa_coeffs, om);
                    let sr_shifted = crate::poly_arith::poly_shift(sr_coeffs, om);
                    let sv_shifted = crate::poly_arith::poly_shift(sv_coeffs, om);
                    let srw_shifted = crate::poly_arith::poly_shift(srw_coeffs, om);

                    // 2. Address continuity: isa(ω·X) * (sr(ω·X) - sr(X)) * (X - ω^{n-1})
                    let reg_diff = crate::poly_arith::poly_sub(&sr_shifted, sr_coeffs, curve);
                    let body2_inner = crate::poly_arith::poly_mul(&isa_shifted, &reg_diff, curve);
                    let body2 = crate::poly_arith::poly_mul_linear(&body2_inner, &omega_n_minus_1);
                    let scaled2 = crate::poly_arith::poly_scalar_mul(&body2, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled2, curve);
                    ap = ap.mul(&alpha);

                    // 3. Difference inverse: (1 - isa(ω·X)) * (1 - diff * inv(X)) * (X - ω^{n-1})
                    let one_minus_isa = crate::poly_arith::poly_sub(&one_poly, &isa_shifted, curve);
                    let diff_inv_prod = crate::poly_arith::poly_mul(&reg_diff, inv_coeffs, curve);
                    let one_minus_prod = crate::poly_arith::poly_sub(&one_poly, &diff_inv_prod, curve);
                    let body3_inner = crate::poly_arith::poly_mul(&one_minus_isa, &one_minus_prod, curve);
                    let body3 = crate::poly_arith::poly_mul_linear(&body3_inner, &omega_n_minus_1);
                    let scaled3 = crate::poly_arith::poly_scalar_mul(&body3, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled3, curve);
                    ap = ap.mul(&alpha);

                    // 4. Read consistency: isa(ω·X) * (1 - rw(ω·X)) * (val(ω·X) - val(X)) * (X - ω^{n-1})
                    let one_minus_srw = crate::poly_arith::poly_sub(&one_poly, &srw_shifted, curve);
                    let isa_times_rw = crate::poly_arith::poly_mul(&isa_shifted, &one_minus_srw, curve);
                    let val_diff = crate::poly_arith::poly_sub(&sv_shifted, sv_coeffs, curve);
                    let body4_inner = crate::poly_arith::poly_mul(&isa_times_rw, &val_diff, curve);
                    let body4 = crate::poly_arith::poly_mul_linear(&body4_inner, &omega_n_minus_1);
                    let scaled4 = crate::poly_arith::poly_scalar_mul(&body4, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled4, curve);
                    ap = ap.mul(&alpha);
                }

                // Grand product transition: Z(ω·X) * ∏ denom_p(X) - Z(X) * ∏ numer_p(X) = 0
                if let (Some(ref rg), Some(ref rd)) = (&reg_perm_gamma_opt, &reg_perm_delta_opt) {
                    let delta2 = rd.mul(rd);
                    let delta3 = delta2.mul(rd);
                    let p_scalar = Scalar::from_u64(num_ports as u64, curve);

                    let z_coeffs = &reg_perm_column_coeffs[layout.z_column];
                    let z_shifted = crate::poly_arith::poly_shift(z_coeffs, om);
                    let row_ts_coeffs = &reg_perm_column_coeffs[layout.row_ts];

                    let gamma_poly = {
                        let mut p = vec![Scalar::zero(curve); domain_size as usize];
                        p[0] = rg.clone();
                        p
                    };

                    // Build product of numer and denom polynomials for all ports
                    let mut numer_product_poly = {
                        let mut p = vec![Scalar::zero(curve); domain_size as usize];
                        p[0] = Scalar::one(curve);
                        p
                    };
                    let mut denom_product_poly = numer_product_poly.clone();

                    for (port_idx, &(reg_col, val_col, is_write)) in reg_ports.iter().enumerate() {
                        // numer_p(X) = γ + trace_reg(X) + δ·trace_val(X) + δ²·(P·row_ts(X)+p) + δ³·rw
                        let mut numer_p = gamma_poly.clone();
                        numer_p = crate::poly_arith::poly_add(&numer_p, &column_coeffs_all[reg_col], curve);
                        let val_scaled = crate::poly_arith::poly_scalar_mul(&column_coeffs_all[val_col], rd);
                        numer_p = crate::poly_arith::poly_add(&numer_p, &val_scaled, curve);
                        // δ²·(P·row_ts + port_idx)
                        let p_row_ts = crate::poly_arith::poly_scalar_mul(row_ts_coeffs, &p_scalar);
                        let port_idx_scalar = Scalar::from_u64(port_idx as u64, curve);
                        let mut p_row_ts_plus_p = p_row_ts;
                        p_row_ts_plus_p[0] = p_row_ts_plus_p[0].add(&port_idx_scalar);
                        let ts_scaled = crate::poly_arith::poly_scalar_mul(&p_row_ts_plus_p, &delta2);
                        numer_p = crate::poly_arith::poly_add(&numer_p, &ts_scaled, curve);
                        // δ³·rw (constant)
                        let rw_val = if is_write { Scalar::one(curve) } else { Scalar::zero(curve) };
                        let rw_term = delta3.mul(&rw_val);
                        numer_p[0] = numer_p[0].add(&rw_term);

                        numer_product_poly = crate::poly_arith::poly_mul(&numer_product_poly, &numer_p, curve);

                        // denom_p(X) = γ + sorted_reg_lane(X) + δ·sorted_val_lane(X) + δ²·sorted_ts_lane(X) + δ³·sorted_rw_lane(X)
                        let mut denom_p = gamma_poly.clone();
                        denom_p = crate::poly_arith::poly_add(
                            &denom_p, &reg_perm_column_coeffs[layout.sorted_reg(port_idx)], curve,
                        );
                        let sv_scaled = crate::poly_arith::poly_scalar_mul(
                            &reg_perm_column_coeffs[layout.sorted_val(port_idx)], rd,
                        );
                        denom_p = crate::poly_arith::poly_add(&denom_p, &sv_scaled, curve);
                        let sts_scaled = crate::poly_arith::poly_scalar_mul(
                            &reg_perm_column_coeffs[layout.sorted_ts(port_idx)], &delta2,
                        );
                        denom_p = crate::poly_arith::poly_add(&denom_p, &sts_scaled, curve);
                        let srw_scaled = crate::poly_arith::poly_scalar_mul(
                            &reg_perm_column_coeffs[layout.sorted_rw(port_idx)], &delta3,
                        );
                        denom_p = crate::poly_arith::poly_add(&denom_p, &srw_scaled, curve);

                        denom_product_poly = crate::poly_arith::poly_mul(&denom_product_poly, &denom_p, curve);
                    }

                    // Z(ω·X) · ∏denom - Z(X) · ∏numer
                    let term1 = crate::poly_arith::poly_mul(&z_shifted, &denom_product_poly, curve);
                    let term2 = crate::poly_arith::poly_mul(z_coeffs, &numer_product_poly, curve);
                    let body_gp = crate::poly_arith::poly_sub(&term1, &term2, curve);
                    let scaled_gp = crate::poly_arith::poly_scalar_mul(&body_gp, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_gp, curve);
                    ap = ap.mul(&alpha);

                    // Boundary: L_0(X) · (Z(X) - 1) = 0
                    let n_inv = Scalar::from_u64(domain_size, curve).inverse();
                    let l0_coeffs: Vec<Scalar> = vec![n_inv; domain_size as usize];
                    let z_minus_one = crate::poly_arith::poly_sub(z_coeffs, &one_poly, curve);
                    let body_boundary = crate::poly_arith::poly_mul(&l0_coeffs, &z_minus_one, curve);
                    let scaled_boundary = crate::poly_arith::poly_scalar_mul(&body_boundary, &ap);
                    c_intra = crate::poly_arith::poly_add(&c_intra, &scaled_boundary, curve);
                    ap = ap.mul(&alpha);
                }

                alpha_offset += crate::permutation::num_register_perm_constraints(layout);
                let _ = ap;
            }
        }

        // Frame-stack permutation contribution.
        if let Some(ref fp_layout) = frame_perm_layout_opt {
            let fp_gamma = frame_perm_gamma_opt.as_ref().unwrap();
            let fp_delta = frame_perm_delta_opt.as_ref().unwrap();
            let om = omega.as_ref().unwrap();

            // Z polynomial in coefficient form (already computed above).
            let z_coeffs_fp: &[Scalar] = &frame_perm_z_coeffs;
            let z_shifted_coeffs = crate::poly_arith::poly_shift(z_coeffs_fp, om);

            // Tuple polynomial coefficients: each tuple column is a
            // trace polynomial; the shifted version is its poly_shift.
            let tuple_coeffs: Vec<Vec<Scalar>> = fp_layout.tuple_columns.iter()
                .map(|&c| column_coeffs_all[c].clone())
                .collect();
            let tuple_shifted_coeffs: Vec<Vec<Scalar>> = tuple_coeffs.iter()
                .map(|c| crate::poly_arith::poly_shift(c, om))
                .collect();

            // is_push(X) = Σ push_sel_k(X); is_pop(X) = Σ pop_sel_k(X).
            let n = domain_size as usize;
            let zero_poly: Vec<Scalar> = vec![Scalar::zero(curve); n];
            let mut is_push_coeffs = zero_poly.clone();
            for &s in &fp_layout.push_selectors {
                is_push_coeffs = crate::poly_arith::poly_add(
                    &is_push_coeffs, &column_coeffs_all[s], curve);
            }
            let mut is_pop_coeffs = zero_poly.clone();
            for &s in &fp_layout.pop_selectors {
                is_pop_coeffs = crate::poly_arith::poly_add(
                    &is_pop_coeffs, &column_coeffs_all[s], curve);
            }

            let body = crate::permutation::build_frame_perm_polynomial(
                z_coeffs_fp,
                &z_shifted_coeffs,
                &tuple_coeffs,
                &tuple_shifted_coeffs,
                &is_push_coeffs,
                &is_pop_coeffs,
                fp_gamma,
                fp_delta,
                domain_size,
                &alpha,
                alpha_offset,
            );
            c_intra = crate::poly_arith::poly_add(&c_intra, &body, curve);
            alpha_offset +=
                crate::permutation::FrameStackPermLayout::NUM_CONSTRAINTS;
        }

        let _ = alpha_offset;
        c_coeffs = c_intra;
    } else {
        // Fallback: evaluate constraints on domain, combine with alpha, IFFT to coeffs
        let columns_refs: Vec<&Vec<Scalar>> = col_eval_forms.iter().collect();
        let constraint_evals = constraints.evaluate_on_domain(&columns_refs, trace.num_rows);
        let n = domain_size as usize;

        // Combine: C_evals[j] = Σ alpha^i * constraint_i_evals[j]
        let mut c_evals = vec![Scalar::zero(curve); n];
        let mut alpha_power = Scalar::one(curve);
        for constraint in &constraint_evals {
            for j in 0..n {
                let val = if j < constraint.len() {
                    constraint[j].clone()
                } else {
                    Scalar::zero(curve)
                };
                let term = alpha_power.mul(&val);
                c_evals[j] = c_evals[j].add(&term);
            }
            alpha_power = alpha_power.mul(&alpha);
        }

        c_coeffs = scheme.ifft(&c_evals, domain_size);
    };
    let constraints_ms = t_constraints.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 5: Divide C(x) by Z(x) = x^n - 1 to get Q(x)
    // -----------------------------------------------------------------------
    let t_quotient = Instant::now();
    let n = domain_size as usize;
    let c_len = c_coeffs.len();
    let c_deg = if c_len > 0 { c_len - 1 } else { 0 };

    let quotient_coeffs_all: Vec<Scalar>;
    if c_deg >= n {
        // C(x) has degree >= n, do proper division
        let mut div = c_coeffs.clone();
        while div.len() <= c_deg {
            div.push(Scalar::zero(curve));
        }
        let q_deg = c_deg - n;
        let mut q = vec![Scalar::zero(curve); q_deg + 1];
        for i in (n..=c_deg).rev() {
            q[i - n] = div[i].clone();
            div[i - n] = div[i - n].add(&div[i]);
        }
        quotient_coeffs_all = q;
    } else {
        // C(x) degree < n: standard approach
        let mut div = c_coeffs.clone();
        while div.len() < 2 * n {
            div.push(Scalar::zero(curve));
        }
        let mut q = vec![Scalar::zero(curve); n];
        for i in (0..n).rev() {
            q[i] = div[n + i].clone();
            div[i] = div[i].add(&q[i]);
        }
        quotient_coeffs_all = q;
    }

    // -----------------------------------------------------------------------
    // Step 6: Split Q into chunks if needed, commit each
    // -----------------------------------------------------------------------
    let num_q_chunks = if quotient_coeffs_all.len() > n {
        ((quotient_coeffs_all.len() - 1) / n) + 1
    } else {
        1
    };
    let num_q_chunks = num_q_chunks as u8;

    let quotient_div_ms = t_quotient.elapsed().as_millis();

    let t_quotient_commit = Instant::now();
    let (quotient_commitments, q_chunk_coeffs): (Vec<_>, Vec<_>) = (0..num_q_chunks as usize)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * n;
            let end = (start + n).min(quotient_coeffs_all.len());
            let mut chunk: Vec<Scalar> = quotient_coeffs_all[start..end].to_vec();
            chunk.resize(n, Scalar::zero(curve));
            let comm = Commitment(scheme.commit_coefficients(&chunk));
            (comm, chunk)
        })
        .unzip();
    let quotient_commit_ms = t_quotient_commit.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 7: Absorb quotient commitments, derive z
    // -----------------------------------------------------------------------
    for qc in &quotient_commitments {
        transcript.append_message(b"quotient_commitment", &qc.0);
    }
    let z_bytes = transcript.challenge_bytes(b"z");
    let z = Scalar::from_challenge_bytes(&z_bytes, curve);

    // -----------------------------------------------------------------------
    // Step 8: Evaluate columns and Q chunks at z
    // -----------------------------------------------------------------------
    let t_evals = Instant::now();
    let mut evaluations: Vec<Vec<u8>> = Vec::with_capacity(col_eval_forms.len() + num_q_chunks as usize);

    for coeffs in &column_coeffs_all {
        let y = scheme.eval_poly_at(coeffs, &z);
        evaluations.push(y.to_bytes());
    }

    let mut q_chunk_evals = Vec::new();
    for chunk in &q_chunk_coeffs {
        let q_at_z = scheme.eval_poly_at(chunk, &z);
        evaluations.push(q_at_z.to_bytes());
        q_chunk_evals.push(q_at_z);
    }

    // -----------------------------------------------------------------------
    // Step 8b: Evaluate shifted columns at ω·z (for cross-row constraints)
    // -----------------------------------------------------------------------
    let mut shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_shifts {
        let omega_z = omega.as_ref().unwrap().mul(&z);
        for &col_idx in &shifted_indices {
            let y = scheme.eval_poly_at(&column_coeffs_all[col_idx], &omega_z);
            shifted_evaluations.push(y.to_bytes());
        }
    }

    // -----------------------------------------------------------------------
    // Step 8c: Evaluate LogUp columns at z and h at ω·z
    // -----------------------------------------------------------------------
    let mut logup_evaluations_vec: Vec<Vec<u8>> = Vec::new();
    let mut logup_shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_logup {
        for coeffs in &logup_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &z);
            logup_evaluations_vec.push(y.to_bytes());
        }
        // h column needs shifted evaluation for running sum transition
        let h_idx = logup_column_coeffs.len() - 2; // h is second-to-last
        let omega_z = omega.as_ref().unwrap().mul(&z);
        let h_shifted = scheme.eval_poly_at(&logup_column_coeffs[h_idx], &omega_z);
        logup_shifted_evaluations.push(h_shifted.to_bytes());
    }

    // -----------------------------------------------------------------------
    // Step 8c': Evaluate bitwise LogUp columns at z and bitwise h at ω·z
    // -----------------------------------------------------------------------
    let mut bitwise_evaluations_vec: Vec<Vec<u8>> = Vec::new();
    let mut bitwise_shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_bitwise {
        for coeffs in &bitwise_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &z);
            bitwise_evaluations_vec.push(y.to_bytes());
        }
        let h_bw_idx = ext_bitwise_layout.h_column;
        let omega_z = omega.as_ref().unwrap().mul(&z);
        let h_shifted = scheme.eval_poly_at(&bitwise_column_coeffs[h_bw_idx], &omega_z);
        bitwise_shifted_evaluations.push(h_shifted.to_bytes());
    }

    // -----------------------------------------------------------------------
    // Step 8d: Evaluate permutation columns at z and shifted columns at ω·z
    // -----------------------------------------------------------------------
    let mut perm_evaluations_vec: Vec<Vec<u8>> = Vec::new();
    let mut perm_shifted_evaluations: Vec<Vec<u8>> = Vec::new();
    if has_perm {
        for coeffs in &perm_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &z);
            perm_evaluations_vec.push(y.to_bytes());
        }
        // Shifted evaluations for cross-row constraints: all perm columns need ω·z
        let omega_z = omega.as_ref().unwrap().mul(&z);
        for coeffs in &perm_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &omega_z);
            perm_shifted_evaluations.push(y.to_bytes());
        }
    }

    // -----------------------------------------------------------------------
    // Step 8e: Evaluate register permutation columns at z and ω·z
    // -----------------------------------------------------------------------
    let mut reg_perm_evaluations_vec: Vec<Vec<u8>> = Vec::new();
    let mut reg_perm_shifted_evals_vec: Vec<Vec<u8>> = Vec::new();
    if has_reg_perm {
        for coeffs in &reg_perm_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &z);
            reg_perm_evaluations_vec.push(y.to_bytes());
        }
        let omega_z = omega.as_ref().unwrap().mul(&z);
        for coeffs in &reg_perm_column_coeffs {
            let y = scheme.eval_poly_at(coeffs, &omega_z);
            reg_perm_shifted_evals_vec.push(y.to_bytes());
        }
    }

    // -----------------------------------------------------------------------
    // Step 8f: Evaluate frame-perm Z column at z and ω·z, and pop
    // selectors at ω·z (needed by the verifier to reconstruct
    // pop_factor at z).
    // -----------------------------------------------------------------------
    let mut frame_perm_eval_z: Option<Vec<u8>> = None;
    let mut frame_perm_eval_omega_z: Option<Vec<u8>> = None;
    let frame_perm_pop_shifted_evals: Vec<Vec<u8>> = Vec::new();
    if has_frame_perm {
        let omega_z = omega.as_ref().unwrap().mul(&z);
        let z_at_z = scheme.eval_poly_at(&frame_perm_z_coeffs, &z);
        frame_perm_eval_z = Some(z_at_z.to_bytes());
        let z_at_omega = scheme.eval_poly_at(&frame_perm_z_coeffs, &omega_z);
        frame_perm_eval_omega_z = Some(z_at_omega.to_bytes());
    }
    let evals_ms = t_evals.elapsed().as_millis();

    // -----------------------------------------------------------------------
    // Step 9: Absorb evaluations, derive batch opening challenge β
    // -----------------------------------------------------------------------
    for eval_bytes in &evaluations {
        transcript.append_message(b"evaluation", eval_bytes);
    }
    for se in &shifted_evaluations {
        transcript.append_message(b"shifted_evaluation", se);
    }
    for le in &logup_evaluations_vec {
        transcript.append_message(b"logup_evaluation", le);
    }
    for lse in &logup_shifted_evaluations {
        transcript.append_message(b"logup_shifted_evaluation", lse);
    }
    for be in &bitwise_evaluations_vec {
        transcript.append_message(b"bitwise_evaluation", be);
    }
    for bse in &bitwise_shifted_evaluations {
        transcript.append_message(b"bitwise_shifted_evaluation", bse);
    }
    for pe in &perm_evaluations_vec {
        transcript.append_message(b"perm_evaluation", pe);
    }
    for pse in &perm_shifted_evaluations {
        transcript.append_message(b"perm_shifted_evaluation", pse);
    }
    for re in &reg_perm_evaluations_vec {
        transcript.append_message(b"reg_perm_evaluation", re);
    }
    for rse in &reg_perm_shifted_evals_vec {
        transcript.append_message(b"reg_perm_shifted_evaluation", rse);
    }
    if let Some(ref fpe) = frame_perm_eval_z {
        transcript.append_message(b"frame_perm_evaluation", fpe);
    }
    if let Some(ref fpse) = frame_perm_eval_omega_z {
        transcript.append_message(b"frame_perm_shifted_evaluation", fpse);
    }
    let _ = &frame_perm_pop_shifted_evals; // reserved
    let beta_bytes = transcript.challenge_bytes(b"beta");
    let beta = Scalar::from_challenge_bytes(&beta_bytes, curve);

    // -----------------------------------------------------------------------
    // Step 10: Batch open all polynomials at z using β
    // -----------------------------------------------------------------------
    let t_batch_open = Instant::now();
    let mut combined_coeffs_batch = vec![Scalar::zero(curve); n];
    let mut combined_y = Scalar::zero(curve);
    let mut beta_power = Scalar::one(curve);

    // Combine column polynomials
    for (col_idx, coeffs) in column_coeffs_all.iter().enumerate() {
        let y = Scalar::from_bytes(&evaluations[col_idx], curve);
        let y_term = beta_power.mul(&y);
        combined_y = combined_y.add(&y_term);

        for j in 0..coeffs.len().min(n) {
            let term = beta_power.mul(&coeffs[j]);
            combined_coeffs_batch[j] = combined_coeffs_batch[j].add(&term);
        }
        beta_power = beta_power.mul(&beta);
    }

    // Combine Q chunk polynomials
    for (chunk_idx, chunk) in q_chunk_coeffs.iter().enumerate() {
        let y_term = beta_power.mul(&q_chunk_evals[chunk_idx]);
        combined_y = combined_y.add(&y_term);

        for j in 0..chunk.len().min(n) {
            let term = beta_power.mul(&chunk[j]);
            combined_coeffs_batch[j] = combined_coeffs_batch[j].add(&term);
        }
        beta_power = beta_power.mul(&beta);
    }

    // Subtract combined_y from constant term
    combined_coeffs_batch[0] = combined_coeffs_batch[0].sub(&combined_y);

    // Synthetic division by (x - z)
    let q_open_coeffs = scheme.div_by_linear(&combined_coeffs_batch, &z);

    // Commit quotient in coefficient form via monomial SRS
    let proof_point = scheme.commit_coefficients(&q_open_coeffs);

    let opening_proof = BatchProof {
        d: vec![],
        proof: proof_point,
    };

    // -----------------------------------------------------------------------
    // Step 10b: Batch open shifted columns at ω·z (if any)
    // -----------------------------------------------------------------------
    let shifted_opening_proof = if has_shifts {
        let beta_shifted_bytes = transcript.challenge_bytes(b"beta_shifted");
        let beta_shifted = Scalar::from_challenge_bytes(&beta_shifted_bytes, curve);
        let omega_z = omega.as_ref().unwrap().mul(&z);

        let mut combined_shifted = vec![Scalar::zero(curve); n];
        let mut combined_shifted_y = Scalar::zero(curve);
        let mut bs_power = Scalar::one(curve);

        for (i, &col_idx) in shifted_indices.iter().enumerate() {
            let y = Scalar::from_bytes(&shifted_evaluations[i], curve);
            let y_term = bs_power.mul(&y);
            combined_shifted_y = combined_shifted_y.add(&y_term);

            let coeffs = &column_coeffs_all[col_idx];
            for j in 0..coeffs.len().min(n) {
                let term = bs_power.mul(&coeffs[j]);
                combined_shifted[j] = combined_shifted[j].add(&term);
            }
            bs_power = bs_power.mul(&beta_shifted);
        }

        combined_shifted[0] = combined_shifted[0].sub(&combined_shifted_y);
        let q_shifted = scheme.div_by_linear(&combined_shifted, &omega_z);
        let shifted_proof_point = scheme.commit_coefficients(&q_shifted);

        Some(BatchProof {
            d: vec![],
            proof: shifted_proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 10c: Batch open LogUp columns at z and h at ω·z (if any)
    // -----------------------------------------------------------------------
    let logup_opening_proof = if has_logup && !logup_column_coeffs.is_empty() {
        let beta_logup_bytes = transcript.challenge_bytes(b"beta_logup");
        let beta_logup = Scalar::from_challenge_bytes(&beta_logup_bytes, curve);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut bl_power = Scalar::one(curve);

        for (i, coeffs) in logup_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&logup_evaluations_vec[i], curve);
            let y_term = bl_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = bl_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            bl_power = bl_power.mul(&beta_logup);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_logup = scheme.div_by_linear(&combined, &z);
        let logup_proof_point = scheme.commit_coefficients(&q_logup);

        Some(BatchProof {
            d: vec![],
            proof: logup_proof_point,
        })
    } else {
        None
    };

    let logup_shifted_opening_proof = if has_logup && !logup_shifted_evaluations.is_empty() {
        // Advance the transcript by drawing β_logup_shifted even though we
        // never use the value — there is only one logup-shifted column (h)
        // so β-RLC is unnecessary, but skipping the draw would desync the
        // verifier's transcript replay.
        let _ = transcript.challenge_bytes(b"beta_logup_shifted");
        let omega_z = omega.as_ref().unwrap().mul(&z);
        let h_idx = logup_column_coeffs.len() - 2;

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let bl_power = Scalar::one(curve);

        let y = Scalar::from_bytes(&logup_shifted_evaluations[0], curve);
        combined_y = combined_y.add(&bl_power.mul(&y));
        let coeffs = &logup_column_coeffs[h_idx];
        for j in 0..coeffs.len().min(n) {
            combined[j] = combined[j].add(&bl_power.mul(&coeffs[j]));
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_shifted = scheme.div_by_linear(&combined, &omega_z);
        let proof_point = scheme.commit_coefficients(&q_shifted);

        Some(BatchProof {
            d: vec![],
            proof: proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 10c': Batch open bitwise LogUp columns at z and bitwise h at ω·z
    // -----------------------------------------------------------------------
    let bitwise_opening_proof = if has_bitwise && !bitwise_column_coeffs.is_empty() {
        let beta_bw_bytes = transcript.challenge_bytes(b"beta_bitwise");
        let beta_bw = Scalar::from_challenge_bytes(&beta_bw_bytes, curve);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut bb_power = Scalar::one(curve);

        for (i, coeffs) in bitwise_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&bitwise_evaluations_vec[i], curve);
            combined_y = combined_y.add(&bb_power.mul(&y));
            for j in 0..coeffs.len().min(n) {
                combined[j] = combined[j].add(&bb_power.mul(&coeffs[j]));
            }
            bb_power = bb_power.mul(&beta_bw);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_bw = scheme.div_by_linear(&combined, &z);
        let proof_point = scheme.commit_coefficients(&q_bw);
        Some(BatchProof { d: vec![], proof: proof_point })
    } else {
        None
    };

    let bitwise_shifted_opening_proof = if has_bitwise && !bitwise_shifted_evaluations.is_empty() {
        let beta_bw_shifted_bytes = transcript.challenge_bytes(b"beta_bitwise_shifted");
        let beta_bw_shifted = Scalar::from_challenge_bytes(&beta_bw_shifted_bytes, curve);
        let omega_z = omega.as_ref().unwrap().mul(&z);
        let h_bw_idx = ext_bitwise_layout.h_column;

        let mut combined = vec![Scalar::zero(curve); n];
        let bl_power = Scalar::one(curve);
        let y = Scalar::from_bytes(&bitwise_shifted_evaluations[0], curve);
        let combined_y = bl_power.mul(&y);
        let coeffs = &bitwise_column_coeffs[h_bw_idx];
        for j in 0..coeffs.len().min(n) {
            combined[j] = combined[j].add(&bl_power.mul(&coeffs[j]));
        }
        combined[0] = combined[0].sub(&combined_y);
        let q_shifted = scheme.div_by_linear(&combined, &omega_z);
        let proof_point = scheme.commit_coefficients(&q_shifted);
        let _ = beta_bw_shifted; // challenge absorbed via transcript
        Some(BatchProof { d: vec![], proof: proof_point })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 10d: Batch open permutation columns at z and ω·z (if any)
    // -----------------------------------------------------------------------
    let perm_opening_proof = if has_perm && !perm_column_coeffs.is_empty() {
        let beta_perm_bytes = transcript.challenge_bytes(b"beta_perm");
        let beta_perm = Scalar::from_challenge_bytes(&beta_perm_bytes, curve);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut bp_power = Scalar::one(curve);

        for (i, coeffs) in perm_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&perm_evaluations_vec[i], curve);
            let y_term = bp_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = bp_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            bp_power = bp_power.mul(&beta_perm);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_perm = scheme.div_by_linear(&combined, &z);
        let perm_proof_point = scheme.commit_coefficients(&q_perm);

        Some(BatchProof {
            d: vec![],
            proof: perm_proof_point,
        })
    } else {
        None
    };

    let perm_shifted_opening_proof = if has_perm && !perm_shifted_evaluations.is_empty() {
        let beta_perm_shifted_bytes = transcript.challenge_bytes(b"beta_perm_shifted");
        let beta_perm_shifted = Scalar::from_challenge_bytes(&beta_perm_shifted_bytes, curve);
        let omega_z = omega.as_ref().unwrap().mul(&z);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut bps_power = Scalar::one(curve);

        for (i, coeffs) in perm_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&perm_shifted_evaluations[i], curve);
            let y_term = bps_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = bps_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            bps_power = bps_power.mul(&beta_perm_shifted);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_perm_shifted = scheme.div_by_linear(&combined, &omega_z);
        let proof_point = scheme.commit_coefficients(&q_perm_shifted);

        Some(BatchProof {
            d: vec![],
            proof: proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 10e: Batch open register permutation columns at z and ω·z
    // -----------------------------------------------------------------------
    let reg_perm_opening = if has_reg_perm && !reg_perm_column_coeffs.is_empty() {
        let beta_rp_bytes = transcript.challenge_bytes(b"beta_reg_perm");
        let beta_rp = Scalar::from_challenge_bytes(&beta_rp_bytes, curve);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut brp_power = Scalar::one(curve);

        for (i, coeffs) in reg_perm_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&reg_perm_evaluations_vec[i], curve);
            let y_term = brp_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = brp_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            brp_power = brp_power.mul(&beta_rp);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_rp = scheme.div_by_linear(&combined, &z);
        let rp_proof_point = scheme.commit_coefficients(&q_rp);

        Some(BatchProof {
            d: vec![],
            proof: rp_proof_point,
        })
    } else {
        None
    };

    let reg_perm_shifted_opening = if has_reg_perm && !reg_perm_shifted_evals_vec.is_empty() {
        let beta_rps_bytes = transcript.challenge_bytes(b"beta_reg_perm_shifted");
        let beta_rps = Scalar::from_challenge_bytes(&beta_rps_bytes, curve);
        let omega_z = omega.as_ref().unwrap().mul(&z);

        let mut combined = vec![Scalar::zero(curve); n];
        let mut combined_y = Scalar::zero(curve);
        let mut brps_power = Scalar::one(curve);

        for (i, coeffs) in reg_perm_column_coeffs.iter().enumerate() {
            let y = Scalar::from_bytes(&reg_perm_shifted_evals_vec[i], curve);
            let y_term = brps_power.mul(&y);
            combined_y = combined_y.add(&y_term);

            for j in 0..coeffs.len().min(n) {
                let term = brps_power.mul(&coeffs[j]);
                combined[j] = combined[j].add(&term);
            }
            brps_power = brps_power.mul(&beta_rps);
        }

        combined[0] = combined[0].sub(&combined_y);
        let q_rps = scheme.div_by_linear(&combined, &omega_z);
        let rps_proof_point = scheme.commit_coefficients(&q_rps);

        Some(BatchProof {
            d: vec![],
            proof: rps_proof_point,
        })
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Step 10f: Open frame-perm Z at z, Z at ω·z, and pop selectors at ω·z.
    // -----------------------------------------------------------------------
    let frame_perm_open_z = if has_frame_perm {
        let _ = transcript.challenge_bytes(b"beta_frame_perm");
        let z_at_z_bytes = frame_perm_eval_z.as_ref().unwrap();
        let z_at_z = Scalar::from_bytes(z_at_z_bytes, curve);
        let mut combined = frame_perm_z_coeffs.clone();
        if combined.len() < n { combined.resize(n, Scalar::zero(curve)); }
        combined[0] = combined[0].sub(&z_at_z);
        let q = scheme.div_by_linear(&combined, &z);
        let p = scheme.commit_coefficients(&q);
        Some(BatchProof { d: vec![], proof: p })
    } else { None };

    let frame_perm_open_omega_z = if has_frame_perm {
        let _ = transcript.challenge_bytes(b"beta_frame_perm_shifted");
        let omega_z = omega.as_ref().unwrap().mul(&z);
        let z_at_omega_bytes = frame_perm_eval_omega_z.as_ref().unwrap();
        let z_at_omega = Scalar::from_bytes(z_at_omega_bytes, curve);
        let mut combined = frame_perm_z_coeffs.clone();
        if combined.len() < n { combined.resize(n, Scalar::zero(curve)); }
        combined[0] = combined[0].sub(&z_at_omega);
        let q = scheme.div_by_linear(&combined, &omega_z);
        let p = scheme.commit_coefficients(&q);
        Some(BatchProof { d: vec![], proof: p })
    } else { None };

    let frame_perm_pop_shifted_open: Option<BatchProof> = None;

    let batch_open_ms = t_batch_open.elapsed().as_millis();
    let total_ms = prove_start.elapsed().as_millis();

    // Print timing breakdown
    let num_logup_cols = logup_column_coeffs.len();
    let num_perm_cols = perm_column_coeffs.len();
    let num_reg_perm_cols = reg_perm_column_coeffs.len();
    eprintln!("[prover] domain={} steps={} cols={}+{}logup+{}perm+{}regperm+{}Q",
        domain_size, num_steps, col_eval_forms.len(),
        num_logup_cols, num_perm_cols, num_reg_perm_cols, num_q_chunks);
    eprintln!("[prover] ifft_trace={}ms commit_trace={}ms logup={}ms mem_perm={}ms reg_perm={}ms",
        ifft_trace_ms, commit_trace_ms, logup_ms, mem_perm_ms, reg_perm_ms);
    eprintln!("[prover] constraints={}ms quot_div={}ms quot_commit={}ms evals={}ms batch_open={}ms total={}ms",
        constraints_ms, quotient_div_ms, quotient_commit_ms, evals_ms, batch_open_ms, total_ms);

    // -----------------------------------------------------------------------
    // Step 11: Assemble and return the proof
    // -----------------------------------------------------------------------
    ExecutionProof {
        column_commitments,
        quotient_commitments,
        evaluations,
        opening_proof,
        num_steps,
        domain_size,
        num_quotient_chunks: num_q_chunks,
        shifted_evaluations,
        shifted_opening_proof,
        logup_commitments,
        logup_evaluations: logup_evaluations_vec,
        logup_shifted_evaluations,
        logup_opening_proof,
        logup_shifted_opening_proof,
        bitwise_commitments,
        bitwise_evaluations: bitwise_evaluations_vec,
        bitwise_shifted_evaluations,
        bitwise_opening_proof,
        bitwise_shifted_opening_proof,
        perm_commitments,
        perm_evaluations: perm_evaluations_vec,
        perm_shifted_evaluations,
        perm_opening_proof,
        perm_shifted_opening_proof,
        oracle_data: oracle_data_entries,
        reg_perm_commitments: reg_perm_commitments_vec,
        reg_perm_evaluations: reg_perm_evaluations_vec,
        reg_perm_shifted_evaluations: reg_perm_shifted_evals_vec,
        reg_perm_opening_proof: reg_perm_opening,
        reg_perm_shifted_opening_proof: reg_perm_shifted_opening,
        frame_perm_commitment: frame_perm_commitment_opt,
        frame_perm_evaluation: frame_perm_eval_z,
        frame_perm_shifted_evaluation: frame_perm_eval_omega_z,
        frame_perm_opening_proof: frame_perm_open_z,
        frame_perm_shifted_opening_proof: frame_perm_open_omega_z,
        frame_perm_pop_shifted_evaluations: frame_perm_pop_shifted_evals,
        frame_perm_pop_shifted_opening_proof: frame_perm_pop_shifted_open,
    }
}

/// Generate a chunk proof from a trace, constraint system, and state chain metadata.
pub fn prove_chunk(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    chunk_index: u64,
    initial_state_hash: &[u8; 32],
    final_state_hash: &[u8; 32],
) -> ChunkProof {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    transcript.append_u64(b"chunk_index", chunk_index);
    transcript.append_message(b"initial_state", initial_state_hash);
    transcript.append_message(b"final_state", final_state_hash);

    let execution_proof = prove_inner(trace, constraints, &mut transcript);

    ChunkProof {
        execution_proof,
        initial_state_hash: *initial_state_hash,
        final_state_hash: *final_state_hash,
        chunk_index,
    }
}

/// Generate a chunk proof using a generic CommitmentScheme.
///
/// Same as [`prove_chunk`] but works with any curve type via the
/// [`CommitmentScheme`] trait (BLS48-581 or BLS12-381).
pub fn prove_chunk_with_scheme(
    trace: &TracePolynomials,
    constraints: &dyn VmConstraintSystem,
    chunk_index: u64,
    initial_state_hash: &[u8; 32],
    final_state_hash: &[u8; 32],
    scheme: &dyn crate::scheme::CommitmentScheme,
) -> ChunkProof {
    let mut transcript = Transcript::new(b"metavm-execution-proof");
    transcript.append_u64(b"chunk_index", chunk_index);
    transcript.append_message(b"initial_state", initial_state_hash);
    transcript.append_message(b"final_state", final_state_hash);

    let execution_proof = prove_inner_scheme(trace, constraints, &mut transcript, scheme);

    ChunkProof {
        execution_proof,
        initial_state_hash: *initial_state_hash,
        final_state_hash: *final_state_hash,
        chunk_index,
    }
}

#[cfg(test)]
mod proof_serde_tests {
    use super::*;

    fn sample_proof() -> ExecutionProof {
        ExecutionProof {
            column_commitments: vec![Commitment(vec![1, 2, 3]), Commitment(vec![4, 5])],
            quotient_commitments: vec![Commitment(vec![9; 74])],
            evaluations: vec![vec![10, 11], vec![], vec![20]],
            opening_proof: BatchProof { d: vec![0xAA, 0xBB], proof: vec![0xCC] },
            num_steps: 0xDEAD_BEEF_FEED_FACE,
            domain_size: 256,
            num_quotient_chunks: 2,
            shifted_evaluations: vec![vec![1]],
            shifted_opening_proof: Some(BatchProof { d: vec![1], proof: vec![2, 3] }),
            logup_commitments: Vec::new(),
            logup_evaluations: Vec::new(),
            logup_shifted_evaluations: Vec::new(),
            logup_opening_proof: None,
            logup_shifted_opening_proof: None,
            bitwise_commitments: Vec::new(),
            bitwise_evaluations: Vec::new(),
            bitwise_shifted_evaluations: Vec::new(),
            bitwise_opening_proof: None,
            bitwise_shifted_opening_proof: None,
            perm_commitments: vec![Commitment(vec![7])],
            perm_evaluations: vec![vec![8]],
            perm_shifted_evaluations: Vec::new(),
            perm_opening_proof: Some(BatchProof { d: vec![5], proof: vec![6] }),
            perm_shifted_opening_proof: None,
            oracle_data: vec![vec![0xCA, 0xFE]],
            reg_perm_commitments: Vec::new(),
            reg_perm_evaluations: Vec::new(),
            reg_perm_shifted_evaluations: Vec::new(),
            reg_perm_opening_proof: None,
            reg_perm_shifted_opening_proof: None,
            frame_perm_commitment: None,
            frame_perm_evaluation: None,
            frame_perm_shifted_evaluation: None,
            frame_perm_opening_proof: None,
            frame_perm_shifted_opening_proof: None,
            frame_perm_pop_shifted_evaluations: Vec::new(),
            frame_perm_pop_shifted_opening_proof: None,
        }
    }

    fn assert_round_trip(p: &ExecutionProof) {
        let bytes = p.to_bytes();
        let decoded = ExecutionProof::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.column_commitments.len(), p.column_commitments.len());
        for (a, b) in decoded.column_commitments.iter().zip(&p.column_commitments) {
            assert_eq!(a.0, b.0);
        }
        assert_eq!(decoded.quotient_commitments.len(), p.quotient_commitments.len());
        for (a, b) in decoded.quotient_commitments.iter().zip(&p.quotient_commitments) {
            assert_eq!(a.0, b.0);
        }
        assert_eq!(decoded.evaluations, p.evaluations);
        assert_eq!(decoded.opening_proof.d, p.opening_proof.d);
        assert_eq!(decoded.opening_proof.proof, p.opening_proof.proof);
        assert_eq!(decoded.num_steps, p.num_steps);
        assert_eq!(decoded.domain_size, p.domain_size);
        assert_eq!(decoded.num_quotient_chunks, p.num_quotient_chunks);
        assert_eq!(decoded.shifted_evaluations, p.shifted_evaluations);
        assert_eq!(decoded.oracle_data, p.oracle_data);
        // Re-serializing the decoded proof must reproduce identical bytes.
        assert_eq!(decoded.to_bytes(), bytes, "encoding must be canonical");
    }

    #[test]
    fn execution_proof_round_trip_minimal() {
        assert_round_trip(&sample_proof());
    }

    #[test]
    fn execution_proof_round_trip_all_optional_present() {
        let mut p = sample_proof();
        p.shifted_opening_proof = Some(BatchProof { d: vec![1], proof: vec![2] });
        p.logup_opening_proof = Some(BatchProof { d: vec![3], proof: vec![4] });
        p.logup_shifted_opening_proof = Some(BatchProof { d: vec![5], proof: vec![6] });
        p.bitwise_opening_proof = Some(BatchProof { d: vec![7], proof: vec![8] });
        p.bitwise_shifted_opening_proof = Some(BatchProof { d: vec![9], proof: vec![10] });
        p.perm_shifted_opening_proof = Some(BatchProof { d: vec![11], proof: vec![12] });
        p.reg_perm_opening_proof = Some(BatchProof { d: vec![13], proof: vec![14] });
        p.reg_perm_shifted_opening_proof = Some(BatchProof { d: vec![15], proof: vec![16] });
        assert_round_trip(&p);
    }

    #[test]
    fn execution_proof_round_trip_all_optional_absent() {
        let mut p = sample_proof();
        p.shifted_opening_proof = None;
        p.perm_opening_proof = None;
        assert_round_trip(&p);
    }

    #[test]
    fn execution_proof_decode_truncated_fails() {
        let p = sample_proof();
        let bytes = p.to_bytes();
        for trunc in &[1usize, 5, 10, bytes.len() / 2, bytes.len() - 1] {
            let r = ExecutionProof::from_bytes(&bytes[..*trunc]);
            assert!(
                r.is_err(),
                "truncated to {} bytes must error, got {:?}",
                trunc,
                r,
            );
        }
    }

    #[test]
    fn execution_proof_decode_trailing_bytes_fails() {
        let p = sample_proof();
        let mut bytes = p.to_bytes();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        assert_eq!(
            ExecutionProof::from_bytes(&bytes).err(),
            Some(ProofDecodeError::TrailingBytes(2)),
        );
    }

    #[test]
    fn execution_proof_decode_invalid_quotient_chunks_fails() {
        // Zero chunks is the only invalid count (any positive value is
        // accepted — the advanced prover path emits 3+ chunks for
        // permutation-grand-product polynomials).
        let mut p = sample_proof();
        p.num_quotient_chunks = 0;
        let bytes = p.to_bytes();
        assert_eq!(
            ExecutionProof::from_bytes(&bytes).err(),
            Some(ProofDecodeError::InvalidQuotientChunks(0)),
        );
    }

    #[test]
    fn execution_proof_decode_accepts_high_quotient_chunk_counts() {
        // The prover's permutation/regperm path can emit 3-4 quotient
        // chunks. Make sure the decoder doesn't reject them.
        for chunks in &[3u8, 4, 7] {
            let mut p = sample_proof();
            p.num_quotient_chunks = *chunks;
            let bytes = p.to_bytes();
            let decoded = ExecutionProof::from_bytes(&bytes).expect("must decode");
            assert_eq!(decoded.num_quotient_chunks, *chunks);
        }
    }

    #[test]
    fn execution_proof_canonical_encoding_is_stable() {
        let p = sample_proof();
        assert_eq!(p.to_bytes(), p.to_bytes());
    }

    fn sample_chunk_proof() -> ChunkProof {
        ChunkProof {
            execution_proof: sample_proof(),
            initial_state_hash: [0xAA; 32],
            final_state_hash: [0xBB; 32],
            chunk_index: 0xCAFE_BEEF_DEAD_BABE,
        }
    }

    fn assert_chunk_round_trip(p: &ChunkProof) {
        let bytes = p.to_bytes();
        let decoded = ChunkProof::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.initial_state_hash, p.initial_state_hash);
        assert_eq!(decoded.final_state_hash, p.final_state_hash);
        assert_eq!(decoded.chunk_index, p.chunk_index);
        assert_eq!(decoded.execution_proof.num_steps, p.execution_proof.num_steps);
        assert_eq!(decoded.execution_proof.domain_size, p.execution_proof.domain_size);
        assert_eq!(decoded.to_bytes(), bytes, "encoding must be canonical");
    }

    #[test]
    fn chunk_proof_round_trip_minimal() {
        assert_chunk_round_trip(&sample_chunk_proof());
    }

    #[test]
    fn chunk_proof_decode_truncated_fails() {
        let p = sample_chunk_proof();
        let bytes = p.to_bytes();
        for trunc in &[0usize, 1, 3, 5, bytes.len() / 2, bytes.len() - 1] {
            let r = ChunkProof::from_bytes(&bytes[..*trunc]);
            assert!(
                r.is_err(),
                "truncated to {} bytes must error, got Ok",
                trunc,
            );
        }
    }

    #[test]
    fn chunk_proof_decode_trailing_bytes_fails() {
        let p = sample_chunk_proof();
        let mut bytes = p.to_bytes();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        assert_eq!(
            ChunkProof::from_bytes(&bytes).err(),
            Some(ProofDecodeError::TrailingBytes(2)),
        );
    }

    #[test]
    fn chunk_proof_inner_proof_decode_error_propagates() {
        let p = sample_chunk_proof();
        let mut bytes = p.to_bytes();
        // Corrupt the inner ExecutionProof body — the embedded
        // num_quotient_chunks byte. Find it by knowing the inner length
        // is the first 4 BE bytes.
        let inner_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        // Set a byte inside the inner range to a wild value that breaks
        // some interior length prefix.
        bytes[5] = 0xFF;
        bytes[6] = 0xFF;
        bytes[7] = 0xFF;
        bytes[8] = 0xFF;
        let r = ChunkProof::from_bytes(&bytes);
        assert!(r.is_err(), "corrupted inner proof must surface as error");
        // Sanity: untouched len prefix is the same.
        let bytes2 = p.to_bytes();
        assert_eq!(
            u32::from_be_bytes([bytes2[0], bytes2[1], bytes2[2], bytes2[3]]) as usize,
            inner_len,
        );
    }
}
