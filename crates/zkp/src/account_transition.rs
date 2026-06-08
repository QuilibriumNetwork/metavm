//! Account state transition oracle.
//!
//! Models the state transition caused by a transaction on an account:
//! nonce increment, balance change (gas payment + value transfer),
//! and storage_root update (from SSTORE transitions).

use crate::account::Account;

#[derive(Clone, Debug)]
pub struct AccountTransition {
    pub address: [u8; 20],
    pub pre: Account,
    pub post: Account,
}

pub fn verify_nonce_increment(t: &AccountTransition) -> Result<(), String> {
    if t.post.nonce != t.pre.nonce + 1 {
        return Err(format!(
            "nonce not incremented: {} -> {} (expected {})",
            t.pre.nonce, t.post.nonce, t.pre.nonce + 1,
        ));
    }
    Ok(())
}

pub fn verify_code_hash_unchanged(t: &AccountTransition) -> Result<(), String> {
    if t.post.code_hash != t.pre.code_hash {
        return Err("code_hash changed (not expected for non-CREATE tx)".into());
    }
    Ok(())
}

pub fn verify_storage_root_transition(
    t: &AccountTransition,
    expected_post_storage_root: [u8; 32],
) -> Result<(), String> {
    if t.post.storage_root != expected_post_storage_root {
        return Err("post storage_root doesn't match SSTORE-computed root".into());
    }
    Ok(())
}

pub fn verify_sender_transition(
    t: &AccountTransition,
    gas_used: u64,
    gas_price: u64,
    value_sent: u64,
) -> Result<(), String> {
    verify_nonce_increment(t)?;
    verify_code_hash_unchanged(t)?;

    let gas_cost = gas_used.checked_mul(gas_price)
        .ok_or("gas cost overflow")?;
    let total_debit = gas_cost.checked_add(value_sent)
        .ok_or("total debit overflow")?;

    let pre_balance = u64::from_be_bytes([
        t.pre.balance[24], t.pre.balance[25], t.pre.balance[26], t.pre.balance[27],
        t.pre.balance[28], t.pre.balance[29], t.pre.balance[30], t.pre.balance[31],
    ]);
    let post_balance = u64::from_be_bytes([
        t.post.balance[24], t.post.balance[25], t.post.balance[26], t.post.balance[27],
        t.post.balance[28], t.post.balance[29], t.post.balance[30], t.post.balance[31],
    ]);

    if pre_balance < total_debit {
        return Err(format!("insufficient balance: {} < {}", pre_balance, total_debit));
    }
    if post_balance != pre_balance - total_debit {
        return Err(format!(
            "balance mismatch: {} - {} = {} but got {}",
            pre_balance, total_debit, pre_balance - total_debit, post_balance,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::{empty_code_hash, empty_storage_root};

    fn make_account(nonce: u64, balance: u64) -> Account {
        let mut bal = [0u8; 32];
        bal[24..32].copy_from_slice(&balance.to_be_bytes());
        Account {
            nonce,
            balance: bal,
            storage_root: empty_storage_root(),
            code_hash: empty_code_hash(),
        }
    }

    #[test]
    fn valid_sender_transition() {
        let pre = make_account(5, 1_000_000);
        let post = make_account(6, 1_000_000 - 21_000 * 10 - 100);
        let t = AccountTransition { address: [0x42; 20], pre, post };
        verify_sender_transition(&t, 21_000, 10, 100).unwrap();
    }

    #[test]
    fn wrong_nonce_fails() {
        let pre = make_account(5, 1_000_000);
        let post = make_account(5, 1_000_000); // nonce not incremented
        let t = AccountTransition { address: [0x42; 20], pre, post };
        assert!(verify_nonce_increment(&t).is_err());
    }

    #[test]
    fn wrong_balance_fails() {
        let pre = make_account(5, 1_000_000);
        let post = make_account(6, 999_999); // wrong debit
        let t = AccountTransition { address: [0x42; 20], pre, post };
        assert!(verify_sender_transition(&t, 21_000, 10, 100).is_err());
    }

    #[test]
    fn insufficient_balance_fails() {
        let pre = make_account(5, 100); // not enough
        let post = make_account(6, 0);
        let t = AccountTransition { address: [0x42; 20], pre, post };
        assert!(verify_sender_transition(&t, 21_000, 10, 0).is_err());
    }

    #[test]
    fn code_hash_changed_fails() {
        let pre = make_account(5, 1_000_000);
        let mut post = make_account(6, 1_000_000 - 210_000);
        post.code_hash[0] ^= 0xff;
        let t = AccountTransition { address: [0x42; 20], pre, post };
        assert!(verify_code_hash_unchanged(&t).is_err());
    }

    #[test]
    fn storage_root_transition() {
        let pre = make_account(5, 1_000_000);
        let mut post = make_account(6, 1_000_000);
        post.storage_root = [0xAA; 32];
        let t = AccountTransition { address: [0x42; 20], pre, post };
        verify_storage_root_transition(&t, [0xAA; 32]).unwrap();
        assert!(verify_storage_root_transition(&t, [0xBB; 32]).is_err());
    }
}
