use sha3::{Digest, Sha3_256};

use crate::field::FieldElement;

/// Fiat-Shamir transcript for generating non-interactive challenges.
/// Uses SHA3-256 for domain-separated hashing.
#[derive(Clone)]
pub struct Transcript {
    hasher: Sha3_256,
}

impl Transcript {
    /// Create a new transcript with the given domain separator label.
    pub fn new(label: &[u8]) -> Self {
        let mut hasher = Sha3_256::new();
        hasher.update(b"metavm-transcript");
        hasher.update(&(label.len() as u32).to_le_bytes());
        hasher.update(label);
        Transcript { hasher }
    }

    /// Append a labeled message to the transcript.
    pub fn append_message(&mut self, label: &[u8], message: &[u8]) {
        self.hasher.update(b"append");
        self.hasher.update(&(label.len() as u32).to_le_bytes());
        self.hasher.update(label);
        self.hasher.update(&(message.len() as u32).to_le_bytes());
        self.hasher.update(message);
    }

    /// Append a field element to the transcript.
    pub fn append_field_element(&mut self, label: &[u8], element: &FieldElement) {
        self.append_message(label, &element.to_bytes());
    }

    /// Append a u64 value to the transcript.
    pub fn append_u64(&mut self, label: &[u8], value: u64) {
        self.append_message(label, &value.to_le_bytes());
    }

    /// Generate a challenge field element from the current transcript state.
    /// This forks the transcript state so multiple challenges can be drawn.
    pub fn challenge(&mut self, label: &[u8]) -> FieldElement {
        let mut fork = self.hasher.clone();
        fork.update(b"challenge");
        fork.update(&(label.len() as u32).to_le_bytes());
        fork.update(label);
        let hash = fork.finalize();

        // Use the 32-byte hash output as a scalar (reduced mod field order)
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&hash);

        // Feed the challenge back into the transcript for chaining
        self.hasher.update(b"challenge-output");
        self.hasher.update(&bytes);

        FieldElement::from_bytes_mod_order(bytes)
    }

    /// Generate raw challenge bytes (32 bytes) from the current transcript state.
    pub fn challenge_bytes(&mut self, label: &[u8]) -> [u8; 32] {
        let mut fork = self.hasher.clone();
        fork.update(b"challenge-bytes");
        fork.update(&(label.len() as u32).to_le_bytes());
        fork.update(label);
        let hash = fork.finalize();

        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&hash);

        self.hasher.update(b"challenge-bytes-output");
        self.hasher.update(&bytes);

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_deterministic() {
        let mut t1 = Transcript::new(b"test");
        t1.append_message(b"data", b"hello");
        let c1 = t1.challenge(b"ch1");

        let mut t2 = Transcript::new(b"test");
        t2.append_message(b"data", b"hello");
        let c2 = t2.challenge(b"ch1");

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_transcript_different_labels() {
        let mut t1 = Transcript::new(b"test1");
        let c1 = t1.challenge(b"ch");

        let mut t2 = Transcript::new(b"test2");
        let c2 = t2.challenge(b"ch");

        assert_ne!(c1, c2);
    }

    #[test]
    fn test_transcript_different_data() {
        let mut t1 = Transcript::new(b"test");
        t1.append_message(b"data", b"hello");
        let c1 = t1.challenge(b"ch");

        let mut t2 = Transcript::new(b"test");
        t2.append_message(b"data", b"world");
        let c2 = t2.challenge(b"ch");

        assert_ne!(c1, c2);
    }

    #[test]
    fn test_transcript_chaining() {
        let mut t = Transcript::new(b"test");
        let c1 = t.challenge(b"first");
        let c2 = t.challenge(b"second");
        // Sequential challenges should be different
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_transcript_field_element() {
        let mut t = Transcript::new(b"test");
        let fe = FieldElement::from_u64(42);
        t.append_field_element(b"val", &fe);
        let c = t.challenge(b"ch");
        assert_ne!(c, FieldElement::ZERO);
    }
}
