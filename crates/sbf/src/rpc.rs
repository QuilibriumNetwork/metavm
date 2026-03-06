//! Solana JSON-RPC client for fetching slot and program data.

/// Transaction data from a Solana slot.
pub struct SlotTransaction {
    pub signature: String,
    pub program_id: String,
    pub accounts: Vec<String>,
    pub data: Vec<u8>,
    /// Which account indices are signers (from the outer tx header).
    pub signer_indices: Vec<usize>,
    /// Which account indices are writable (from the outer tx header).
    pub writable_indices: Vec<usize>,
}

/// Slot data from RPC.
pub struct SlotData {
    pub slot: u64,
    pub transactions: Vec<SlotTransaction>,
}

/// On-chain account data fetched via RPC.
pub struct AccountData {
    pub lamports: u64,
    pub data: Vec<u8>,
    pub owner: [u8; 32],
    pub executable: bool,
    pub rent_epoch: u64,
}

fn rpc_call(rpc_url: &str, method: &str, params: &serde_json::Value) -> Result<serde_json::Value, String> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let resp = ureq::post(rpc_url)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("HTTP error: {}", e))?;

    let json: serde_json::Value = resp.into_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;

    if let Some(error) = json.get("error") {
        return Err(format!("RPC error: {}", error));
    }

    json.get("result").cloned().ok_or_else(|| "No result in response".to_string())
}

/// Fetch a Solana block/slot with full transaction details.
pub fn fetch_block(rpc_url: &str, slot: u64) -> Result<SlotData, String> {
    let result = rpc_call(rpc_url, "getBlock", &serde_json::json!([
        slot,
        {
            "encoding": "json",
            "transactionDetails": "full",
            "maxSupportedTransactionVersion": 0
        }
    ]))?;

    if result.is_null() {
        return Err(format!("Slot {} not found or skipped", slot));
    }

    let txs = result.get("transactions")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter().filter_map(|tx_wrapper| {
                let tx = tx_wrapper.get("transaction")?;
                let msg = tx.get("message")?;

                // Get account keys
                let account_keys: Vec<String> = msg.get("accountKeys")
                    .and_then(|keys| keys.as_array())
                    .map(|arr| arr.iter().filter_map(|k| k.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default();

                // Parse tx header for signer/writable info
                let header = msg.get("header");
                let num_required_signatures = header
                    .and_then(|h| h.get("numRequiredSignatures"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let num_readonly_signed = header
                    .and_then(|h| h.get("numReadonlySignedAccounts"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let num_readonly_unsigned = header
                    .and_then(|h| h.get("numReadonlyUnsignedAccounts"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                let num_accounts = account_keys.len();

                // Get instructions
                let instructions = msg.get("instructions")
                    .and_then(|i| i.as_array())?;

                // Take the first non-system instruction
                for insn in instructions {
                    let program_id_idx = insn.get("programIdIndex")
                        .and_then(|i| i.as_u64())? as usize;
                    if program_id_idx >= num_accounts { continue; }
                    let program_id = &account_keys[program_id_idx];

                    // Skip system programs
                    if program_id == "11111111111111111111111111111111"
                        || program_id == "ComputeBudget111111111111111111111111111111"
                        || program_id == "Vote111111111111111111111111111111111111111" {
                        continue;
                    }

                    let data_b58 = insn.get("data").and_then(|d| d.as_str()).unwrap_or("");
                    let data = bs58_decode(data_b58).unwrap_or_default();

                    let insn_account_indices: Vec<usize> = insn.get("accounts")
                        .and_then(|a| a.as_array())
                        .map(|arr| arr.iter().filter_map(|idx| idx.as_u64().map(|i| i as usize)).collect())
                        .unwrap_or_default();

                    let insn_accounts: Vec<String> = insn_account_indices.iter()
                        .filter_map(|&i| account_keys.get(i).cloned())
                        .collect();

                    // Determine which instruction accounts are signers/writable
                    // based on their position in the global account_keys list.
                    // Signers: indices 0..num_required_signatures
                    // Writable signers: indices 0..num_required_signatures - num_readonly_signed
                    // Writable unsigned: indices num_required_signatures..num_accounts - num_readonly_unsigned
                    let signer_indices: Vec<usize> = insn_account_indices.iter().enumerate()
                        .filter(|(_, &global_idx)| global_idx < num_required_signatures)
                        .map(|(local_idx, _)| local_idx)
                        .collect();

                    let writable_signer_end = num_required_signatures.saturating_sub(num_readonly_signed);
                    let readonly_unsigned_start = num_accounts.saturating_sub(num_readonly_unsigned);
                    let writable_indices: Vec<usize> = insn_account_indices.iter().enumerate()
                        .filter(|(_, &global_idx)| {
                            // Writable if: writable signer OR writable unsigned
                            global_idx < writable_signer_end
                                || (global_idx >= num_required_signatures && global_idx < readonly_unsigned_start)
                        })
                        .map(|(local_idx, _)| local_idx)
                        .collect();

                    let sig = tx.get("signatures")
                        .and_then(|s| s.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();

                    return Some(SlotTransaction {
                        signature: sig,
                        program_id: program_id.clone(),
                        accounts: insn_accounts,
                        data,
                        signer_indices,
                        writable_indices,
                    });
                }
                None
            }).collect()
        })
        .unwrap_or_default();

    Ok(SlotData {
        slot,
        transactions: txs,
    })
}

/// Fetch a program's ELF binary from the chain.
pub fn fetch_program_elf(rpc_url: &str, program_id: &str) -> Result<Vec<u8>, String> {
    let result = rpc_call(rpc_url, "getAccountInfo", &serde_json::json!([
        program_id,
        {"encoding": "base64"}
    ]))?;

    let account = result.get("value").ok_or("Account not found")?;
    if account.is_null() {
        return Err(format!("Program account {} not found", program_id));
    }

    let data_arr = account.get("data")
        .and_then(|d| d.as_array())
        .ok_or("No data in account")?;

    let b64_data = data_arr.first()
        .and_then(|d| d.as_str())
        .ok_or("Invalid data format")?;

    let raw_data = base64_decode(b64_data)?;

    let owner = account.get("owner").and_then(|o| o.as_str()).unwrap_or("");

    // Check if this is an upgradeable program
    if owner == "BPFLoaderUpgradeab1e11111111111111111111111" {
        // The account data contains a programdata address
        // Layout: [4 bytes state][32 bytes programdata_address]
        if raw_data.len() < 36 {
            return Err("Invalid upgradeable program account".to_string());
        }
        let programdata_key = bs58_encode(&raw_data[4..36]);

        // Fetch the programdata account
        let pd_result = rpc_call(rpc_url, "getAccountInfo", &serde_json::json!([
            &programdata_key,
            {"encoding": "base64"}
        ]))?;

        let pd_account = pd_result.get("value").ok_or("Programdata account not found")?;
        let pd_data_arr = pd_account.get("data")
            .and_then(|d| d.as_array())
            .ok_or("No data in programdata account")?;
        let pd_b64 = pd_data_arr.first()
            .and_then(|d| d.as_str())
            .ok_or("Invalid programdata format")?;
        let pd_raw = base64_decode(pd_b64)?;

        // Skip the 45-byte programdata header to get the ELF
        if pd_raw.len() <= 45 {
            return Err("Programdata too small to contain ELF".to_string());
        }
        Ok(pd_raw[45..].to_vec())
    } else {
        // Direct BPF program (BPFLoader2)
        Ok(raw_data)
    }
}

/// Fetch account info from a Solana RPC endpoint.
pub fn fetch_account_info(rpc_url: &str, pubkey: &str) -> Result<AccountData, String> {
    let result = rpc_call(rpc_url, "getAccountInfo", &serde_json::json!([
        pubkey,
        {"encoding": "base64"}
    ]))?;

    let account = result.get("value").ok_or("Account not found")?;
    if account.is_null() {
        // Return a default empty account (common for PDAs that haven't been created yet)
        return Ok(AccountData {
            lamports: 0,
            data: Vec::new(),
            owner: [0u8; 32],
            executable: false,
            rent_epoch: 0,
        });
    }

    let lamports = account.get("lamports")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let data = account.get("data")
        .and_then(|d| d.as_array())
        .and_then(|arr| arr.first())
        .and_then(|d| d.as_str())
        .map(|s| base64_decode(s).unwrap_or_default())
        .unwrap_or_default();

    let owner_str = account.get("owner")
        .and_then(|o| o.as_str())
        .unwrap_or("11111111111111111111111111111111");
    let owner_bytes = bs58_decode(owner_str).unwrap_or_else(|_| vec![0u8; 32]);
    let mut owner = [0u8; 32];
    let copy_len = owner_bytes.len().min(32);
    owner[..copy_len].copy_from_slice(&owner_bytes[..copy_len]);

    let executable = account.get("executable")
        .and_then(|e| e.as_bool())
        .unwrap_or(false);

    let rent_epoch = account.get("rentEpoch")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    Ok(AccountData {
        lamports,
        data,
        owner,
        executable,
        rent_epoch,
    })
}

/// Serialize accounts and instruction data in Solana BPF input format.
///
/// This is the format that the Solana runtime passes to BPF programs at
/// MM_INPUT_START. Layout:
///
/// ```text
/// [8 bytes: num_accounts as u64 LE]
/// For each account:
///   [1 byte: dup_info (0xFF = not duplicate)]
///   [1 byte: is_signer (0 or 1)]
///   [1 byte: is_writable (0 or 1)]
///   [1 byte: is_executable (0 or 1)]
///   [4 bytes: padding (zeros)]
///   [32 bytes: account key (pubkey)]
///   [32 bytes: owner]
///   [8 bytes: lamports as u64 LE]
///   [8 bytes: data_len as u64 LE]
///   [data_len bytes: account data]
///   [padding to 8-byte alignment]
///   [8 bytes: rent_epoch as u64 LE]
/// [8 bytes: instruction_data_len as u64 LE]
/// [instruction_data_len bytes: instruction_data]
/// [32 bytes: program_id pubkey]
/// ```
pub fn serialize_bpf_input(
    accounts: &[(String, AccountData, bool, bool)], // (pubkey, data, is_signer, is_writable)
    instruction_data: &[u8],
    program_id: &str,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4096);

    // Number of accounts
    buf.extend_from_slice(&(accounts.len() as u64).to_le_bytes());

    for (pubkey_str, acct, is_signer, is_writable) in accounts {
        // dup_info: 0xFF = not a duplicate
        buf.push(0xFF);
        // is_signer
        buf.push(if *is_signer { 1 } else { 0 });
        // is_writable
        buf.push(if *is_writable { 1 } else { 0 });
        // is_executable
        buf.push(if acct.executable { 1 } else { 0 });
        // 4 bytes padding
        buf.extend_from_slice(&[0u8; 4]);

        // Account key (32 bytes)
        let key_bytes = bs58_decode(pubkey_str).unwrap_or_else(|_| vec![0u8; 32]);
        let mut key = [0u8; 32];
        let klen = key_bytes.len().min(32);
        key[..klen].copy_from_slice(&key_bytes[..klen]);
        buf.extend_from_slice(&key);

        // Owner (32 bytes)
        buf.extend_from_slice(&acct.owner);

        // Lamports
        buf.extend_from_slice(&acct.lamports.to_le_bytes());

        // Data length + data
        buf.extend_from_slice(&(acct.data.len() as u64).to_le_bytes());
        buf.extend_from_slice(&acct.data);

        // Pad to 8-byte alignment
        let padding = (8 - (acct.data.len() % 8)) % 8;
        for _ in 0..padding {
            buf.push(0);
        }

        // Rent epoch
        buf.extend_from_slice(&acct.rent_epoch.to_le_bytes());
    }

    // Instruction data
    buf.extend_from_slice(&(instruction_data.len() as u64).to_le_bytes());
    buf.extend_from_slice(instruction_data);

    // Program ID (32 bytes)
    let pid_bytes = bs58_decode(program_id).unwrap_or_else(|_| vec![0u8; 32]);
    let mut pid = [0u8; 32];
    let plen = pid_bytes.len().min(32);
    pid[..plen].copy_from_slice(&pid_bytes[..plen]);
    buf.extend_from_slice(&pid);

    buf
}

// Minimal base64 decode
fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(s)
        .map_err(|e| format!("Base64 decode error: {}", e))
}

// Minimal base58 decode (just enough for Solana addresses)
fn bs58_decode(s: &str) -> Result<Vec<u8>, String> {
    const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    let mut result = vec![0u8; 64]; // oversized buffer
    let mut result_len = 0usize;

    for &c in s.as_bytes() {
        let mut carry = ALPHABET.iter().position(|&a| a == c)
            .ok_or_else(|| format!("Invalid base58 character: {}", c as char))? as u32;

        for byte in result[..result_len].iter_mut().rev() {
            carry += 58 * (*byte as u32);
            *byte = (carry % 256) as u8;
            carry /= 256;
        }
        while carry > 0 {
            result_len += 1;
            if result_len > result.len() {
                result.push(0);
            }
            result[result_len - 1] = 0;
            // Shift right
            for i in (1..result_len).rev() {
                result[i] = result[i - 1];
            }
            result[0] = (carry % 256) as u8;
            carry /= 256;
        }
    }

    // Count leading '1's (leading zeros in base58)
    let leading_zeros = s.bytes().take_while(|&c| c == b'1').count();
    let mut out = vec![0u8; leading_zeros];
    out.extend_from_slice(&result[..result_len]);
    Ok(out)
}

fn bs58_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    if bytes.is_empty() { return String::new(); }

    let mut digits = vec![0u8; bytes.len() * 2];
    let mut digit_len = 0;

    for &byte in bytes {
        let mut carry = byte as u32;
        for d in digits[..digit_len].iter_mut() {
            carry += 256 * (*d as u32);
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            if digit_len >= digits.len() { digits.push(0); }
            digits[digit_len] = (carry % 58) as u8;
            digit_len += 1;
            carry /= 58;
        }
    }

    let leading_zeros = bytes.iter().take_while(|&&b| b == 0).count();
    let mut result = String::with_capacity(leading_zeros + digit_len);
    for _ in 0..leading_zeros {
        result.push('1');
    }
    for &d in digits[..digit_len].iter().rev() {
        result.push(ALPHABET[d as usize] as char);
    }
    result
}
