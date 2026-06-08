//! World state root transition oracle.
//!
//! Verifies that a block's state_root correctly reflects account
//! state changes caused by transaction execution. Given pre-state
//! account inclusions and post-state account inclusions, verifies
//! the MPT roots match the block header's state_root fields.

use crate::account::{account_trie_key, Account, verify_account_inclusion_oracle};

#[derive(Clone, Debug)]
pub struct AccountInclusion {
    pub address: [u8; 20],
    pub account: Account,
    pub proof: Vec<Vec<u8>>,
}

pub fn verify_pre_state_inclusion(
    pre_state_root: [u8; 32],
    inclusion: &AccountInclusion,
) -> Result<(), String> {
    verify_account_inclusion_oracle(
        pre_state_root,
        &inclusion.address,
        &inclusion.account,
        &inclusion.proof,
    )
}

pub fn verify_post_state_inclusion(
    post_state_root: [u8; 32],
    inclusion: &AccountInclusion,
) -> Result<(), String> {
    verify_account_inclusion_oracle(
        post_state_root,
        &inclusion.address,
        &inclusion.account,
        &inclusion.proof,
    )
}

pub fn verify_state_transition(
    pre_state_root: [u8; 32],
    post_state_root: [u8; 32],
    pre_accounts: &[AccountInclusion],
    post_accounts: &[AccountInclusion],
) -> Result<(), String> {
    for (i, pre) in pre_accounts.iter().enumerate() {
        verify_pre_state_inclusion(pre_state_root, pre)
            .map_err(|e| format!("pre-state account[{}]: {}", i, e))?;
    }
    for (i, post) in post_accounts.iter().enumerate() {
        verify_post_state_inclusion(post_state_root, post)
            .map_err(|e| format!("post-state account[{}]: {}", i, e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::account::{account_rlp, empty_code_hash, empty_storage_root};
    use crate::mpt;

    fn make_inclusion(address: [u8; 20], account: Account) -> (AccountInclusion, [u8; 32]) {
        let trie_key = account_trie_key(&address);
        let value = account_rlp(&account);
        let (root, proof) = mpt::single_leaf_trie(&trie_key, &value);
        (AccountInclusion { address, account, proof }, root)
    }

    #[test]
    fn pre_state_inclusion_passes() {
        let account = Account {
            nonce: 0, balance: [0u8; 32],
            storage_root: empty_storage_root(), code_hash: empty_code_hash(),
        };
        let (incl, root) = make_inclusion([0x42; 20], account);
        verify_pre_state_inclusion(root, &incl).unwrap();
    }

    #[test]
    fn wrong_root_fails() {
        let account = Account {
            nonce: 0, balance: [0u8; 32],
            storage_root: empty_storage_root(), code_hash: empty_code_hash(),
        };
        let (incl, mut root) = make_inclusion([0x42; 20], account);
        root[0] ^= 0xff;
        assert!(verify_pre_state_inclusion(root, &incl).is_err());
    }

    #[test]
    fn state_transition_both_pass() {
        let pre_account = Account {
            nonce: 0, balance: [0u8; 32],
            storage_root: empty_storage_root(), code_hash: empty_code_hash(),
        };
        let mut post_account = pre_account.clone();
        post_account.nonce = 1;

        let (pre_incl, pre_root) = make_inclusion([0x42; 20], pre_account);
        let (post_incl, post_root) = make_inclusion([0x42; 20], post_account);

        verify_state_transition(pre_root, post_root, &[pre_incl], &[post_incl]).unwrap();
    }

    #[test]
    fn state_transition_mismatched_post_fails() {
        let account = Account {
            nonce: 0, balance: [0u8; 32],
            storage_root: empty_storage_root(), code_hash: empty_code_hash(),
        };
        let (pre_incl, pre_root) = make_inclusion([0x42; 20], account.clone());
        let (post_incl, _) = make_inclusion([0x42; 20], account);
        assert!(verify_state_transition(pre_root, [0xBB; 32], &[pre_incl], &[post_incl]).is_err());
    }
}
