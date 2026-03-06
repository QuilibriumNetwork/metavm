/// Trait for types that can produce a cryptographic commitment.
pub trait Committable {
    type Commitment;

    /// Produce a binding commitment to this value.
    fn commit(&self) -> Self::Commitment;
}

/// Trait for types that can generate and verify proofs.
pub trait Provable {
    type Proof;
    type Statement;
    type Error;

    /// Generate a proof for the given statement.
    fn prove(&self, statement: &Self::Statement) -> Result<Self::Proof, Self::Error>;

    /// Verify a proof against a statement.
    fn verify(proof: &Self::Proof, statement: &Self::Statement) -> Result<bool, Self::Error>;
}

/// Trait for types that can be secret-shared among parties.
pub trait SecretShareable {
    type Share;

    /// Split this value into `n` shares with threshold `t`.
    fn split(&self, n: usize, t: usize) -> Vec<Self::Share>;

    /// Reconstruct the value from a set of shares.
    fn reconstruct(shares: &[Self::Share]) -> Self;
}
