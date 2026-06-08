//! Account state RLP composition oracle.

use crate::account::Account;
use crate::fixed_rlp_air::RLP32_PREFIX;
use crate::u64_rlp_air::rlp_encode_u64;
use crate::u256_rlp_air::rlp_encode_u256_be;

pub fn verify_account_rlp_composition(a: &Account) -> Result<Vec<u8>, String> {
    let canonical = crate::account::account_rlp(a);
    let fields: Vec<Vec<u8>> = vec![
        rlp_encode_u64(a.nonce),
        rlp_encode_u256_be(&a.balance),
        { let mut e = vec![RLP32_PREFIX]; e.extend_from_slice(&a.storage_root); e },
        { let mut e = vec![RLP32_PREFIX]; e.extend_from_slice(&a.code_hash); e },
    ];
    let payload_len: usize = fields.iter().map(|f| f.len()).sum();
    let mut assembled = Vec::with_capacity(3 + payload_len);
    if payload_len < 56 {
        assembled.push(0xc0 + payload_len as u8);
    } else {
        let mut len_be = Vec::new(); let mut n = payload_len;
        while n > 0 { len_be.push((n & 0xff) as u8); n >>= 8; }
        len_be.reverse();
        assembled.push(0xf7 + len_be.len() as u8);
        assembled.extend_from_slice(&len_be);
    }
    for f in &fields { assembled.extend_from_slice(f); }
    if assembled != canonical {
        return Err(format!("account RLP mismatch: {} vs {} bytes", assembled.len(), canonical.len()));
    }
    Ok(assembled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_account() {
        verify_account_rlp_composition(&Account::default()).unwrap();
    }

    #[test]
    fn account_with_balance() {
        let mut a = Account::default();
        a.nonce = 42;
        a.balance = { let mut b = [0u8; 32]; b[24..32].copy_from_slice(&1_000_000_000_000_000_000u64.to_be_bytes()); b };
        verify_account_rlp_composition(&a).unwrap();
    }

    #[test]
    fn contract_account() {
        let a = Account {
            nonce: 1, balance: [0u8; 32],
            storage_root: [0xaa; 32], code_hash: [0xbb; 32],
        };
        verify_account_rlp_composition(&a).unwrap();
    }
}
