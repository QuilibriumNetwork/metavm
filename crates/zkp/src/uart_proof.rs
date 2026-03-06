//! UART output proof for verifying that a proven execution produced expected output.
//!
//! The UART output proof uses a "hash-and-reveal" approach:
//! 1. During execution, each chunk records UART write bytes (stores to `0x1000_0000`)
//! 2. The cumulative UART hash (SHA3-256) is included in each chunk's boundary state
//! 3. The final proof includes the `final_state_hash` which commits to the UART hash
//! 4. The verifier independently hashes the expected output and compares
//!
//! This avoids including the full UART output in the proof. The verifier only needs
//! the expected output bytes and the final state's UART hash to check correctness.

use sha3::{Sha3_256, Digest};
use crate::recursive::RecursiveProof;

/// UART address used by the MetaVM MMIO controller.
pub const UART_BASE: u64 = 0x1000_0000;

/// Compute the SHA3-256 hash of the expected UART output.
///
/// This is the hash the verifier computes independently from the claimed output
/// bytes. It must match the `uart_output_hash` in the final chunk's boundary state.
pub fn hash_uart_output(output: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(output);
    hasher.finalize().into()
}

/// Compute the SHA3-256 hash of UART output incrementally from chunks.
///
/// This mirrors the streaming prover's approach: each chunk appends its UART
/// bytes to the running hash. The final hash matches what the prover commits to.
pub struct UartHasher {
    hasher: Sha3_256,
    total_bytes: u64,
}

impl UartHasher {
    pub fn new() -> Self {
        UartHasher {
            hasher: Sha3_256::new(),
            total_bytes: 0,
        }
    }

    /// Append UART bytes from one chunk.
    pub fn append(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
        self.total_bytes += bytes.len() as u64;
    }

    /// Finalize and return the cumulative UART output hash.
    pub fn finalize(self) -> [u8; 32] {
        self.hasher.finalize().into()
    }

    /// Return the total number of UART bytes seen so far.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

/// Verification result for UART output proof.
#[derive(Debug, Clone)]
pub struct UartVerification {
    /// Whether the UART output hash matches the expected output.
    pub output_matches: bool,
    /// The expected UART hash (from claimed output bytes).
    pub expected_hash: [u8; 32],
    /// The actual UART hash (from the proof's final state).
    pub actual_hash: Option<[u8; 32]>,
}

/// Verify that a recursive proof's execution produced the expected UART output.
///
/// This extracts the UART output hash from the final state and compares it
/// against the hash of the expected output bytes.
///
/// Note: The `uart_hash_from_final_state` must be extracted from the final
/// chunk's boundary state separately, since the `RecursiveProof` only stores
/// the overall state hash (not individual fields). The caller must provide
/// the UART hash that was part of the final state.
pub fn verify_uart_output(
    expected_output: &[u8],
    uart_hash_from_final_state: &[u8; 32],
) -> UartVerification {
    let expected_hash = hash_uart_output(expected_output);
    let matches = expected_hash == *uart_hash_from_final_state;

    UartVerification {
        output_matches: matches,
        expected_hash,
        actual_hash: Some(*uart_hash_from_final_state),
    }
}

/// Check that a recursive proof has state hashes present (required for UART verification).
pub fn has_state_chain(proof: &RecursiveProof) -> bool {
    proof.initial_state_hash.is_some() && proof.final_state_hash.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_uart_output_empty() {
        let h1 = hash_uart_output(b"");
        let h2 = hash_uart_output(b"");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_uart_output_deterministic() {
        let msg = b"Hello from Linux!\n";
        let h1 = hash_uart_output(msg);
        let h2 = hash_uart_output(msg);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_uart_output_differs() {
        let h1 = hash_uart_output(b"Hello");
        let h2 = hash_uart_output(b"World");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_uart_hasher_incremental() {
        // Incremental hashing should match one-shot for the same bytes
        let full = b"Hello from Linux!\n";
        let h_oneshot = hash_uart_output(full);

        let mut hasher = UartHasher::new();
        hasher.append(b"Hello ");
        hasher.append(b"from ");
        hasher.append(b"Linux!\n");
        let h_incremental = hasher.finalize();

        assert_eq!(h_oneshot, h_incremental);
    }

    #[test]
    fn test_uart_hasher_total_bytes() {
        let mut hasher = UartHasher::new();
        assert_eq!(hasher.total_bytes(), 0);
        hasher.append(b"Hello");
        assert_eq!(hasher.total_bytes(), 5);
        hasher.append(b" World");
        assert_eq!(hasher.total_bytes(), 11);
    }

    #[test]
    fn test_verify_uart_output_match() {
        let output = b"Boot complete.\n";
        let hash = hash_uart_output(output);
        let result = verify_uart_output(output, &hash);
        assert!(result.output_matches);
        assert_eq!(result.expected_hash, hash);
    }

    #[test]
    fn test_verify_uart_output_mismatch() {
        let expected = b"Boot complete.\n";
        let wrong_hash = hash_uart_output(b"Wrong output");
        let result = verify_uart_output(expected, &wrong_hash);
        assert!(!result.output_matches);
    }
}
