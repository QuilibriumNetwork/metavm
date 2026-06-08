//! Beacon chain deposit data oracle.
//!
//! Verifies the structure of deposit data from the Ethereum 1.0
//! deposit contract that feeds the beacon chain validator registry.

use crate::ssz;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepositData {
    pub pubkey: [u8; 48],
    pub withdrawal_credentials: [u8; 32],
    pub amount: u64, // in Gwei
    pub signature: [u8; 96],
}

impl DepositData {
    /// SSZ hash_tree_root over the 4 fields.
    pub fn hash_tree_root(&self) -> [u8; 32] {
        let field_roots: [[u8; 32]; 4] = [
            ssz::hash_tree_root_bytes_fixed(&self.pubkey, 2),
            self.withdrawal_credentials,
            ssz::hash_tree_root_uint(self.amount),
            ssz::hash_tree_root_bytes_fixed(&self.signature, 3),
        ];
        ssz::merkleize_chunks(&field_roots, None)
    }
}

pub const MIN_DEPOSIT_AMOUNT: u64 = 1_000_000_000; // 1 ETH in Gwei
pub const MAX_EFFECTIVE_BALANCE: u64 = 32_000_000_000; // 32 ETH in Gwei

pub fn validate_deposit_amount(amount: u64) -> Result<(), String> {
    if amount < MIN_DEPOSIT_AMOUNT {
        return Err(format!(
            "deposit amount {} below min {}",
            amount, MIN_DEPOSIT_AMOUNT,
        ));
    }
    Ok(())
}

pub fn effective_balance_from_deposit(amount: u64) -> u64 {
    amount.min(MAX_EFFECTIVE_BALANCE)
}

pub fn validate_withdrawal_credentials(wc: &[u8; 32]) -> Result<u8, String> {
    let prefix = wc[0];
    match prefix {
        0x00 => Ok(0), // BLS
        0x01 => Ok(1), // execution address (Eth1)
        0x02 => Ok(2), // compounding (EIP-7251)
        _ => Err(format!("invalid withdrawal credentials prefix: 0x{:02x}", prefix)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deposit_htr_deterministic() {
        let d = DepositData {
            pubkey: [0x11; 48],
            withdrawal_credentials: [0x22; 32],
            amount: 32_000_000_000,
            signature: [0x33; 96],
        };
        assert_eq!(d.hash_tree_root(), d.hash_tree_root());
    }

    #[test]
    fn different_deposits_different_roots() {
        let d1 = DepositData {
            pubkey: [0x11; 48],
            withdrawal_credentials: [0x22; 32],
            amount: 32_000_000_000,
            signature: [0x33; 96],
        };
        let mut d2 = d1.clone();
        d2.amount = 16_000_000_000;
        assert_ne!(d1.hash_tree_root(), d2.hash_tree_root());
    }

    #[test]
    fn min_amount_validation() {
        validate_deposit_amount(MIN_DEPOSIT_AMOUNT).unwrap();
        validate_deposit_amount(32_000_000_000).unwrap();
        assert!(validate_deposit_amount(MIN_DEPOSIT_AMOUNT - 1).is_err());
        assert!(validate_deposit_amount(0).is_err());
    }

    #[test]
    fn effective_balance_capped() {
        assert_eq!(effective_balance_from_deposit(32_000_000_000), 32_000_000_000);
        assert_eq!(effective_balance_from_deposit(64_000_000_000), 32_000_000_000);
        assert_eq!(effective_balance_from_deposit(1_000_000_000), 1_000_000_000);
    }

    #[test]
    fn withdrawal_credentials_valid() {
        let mut wc = [0u8; 32]; wc[0] = 0x00;
        assert_eq!(validate_withdrawal_credentials(&wc).unwrap(), 0);
        wc[0] = 0x01;
        assert_eq!(validate_withdrawal_credentials(&wc).unwrap(), 1);
        wc[0] = 0x02;
        assert_eq!(validate_withdrawal_credentials(&wc).unwrap(), 2);
    }

    #[test]
    fn withdrawal_credentials_invalid() {
        let mut wc = [0u8; 32]; wc[0] = 0xFF;
        assert!(validate_withdrawal_credentials(&wc).is_err());
    }
}
