//! Ethereum JSON-RPC client for fetching block and state data.

use revm::database_interface::{Database, DatabaseRef};
use revm::primitives::{Address, Bytes, B256, U256};
use revm::state::{AccountInfo, Bytecode};
use std::cell::RefCell;
use std::collections::HashMap;

/// RPC error type for database operations.
#[derive(Debug)]
pub struct RpcError(pub String);

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RPC error: {}", self.0)
    }
}

impl std::error::Error for RpcError {}
impl revm::database_interface::DBErrorMarker for RpcError {}

/// Ethereum block data from RPC.
pub struct RpcBlock {
    pub number: u64,
    pub hash: B256,
    pub parent_hash: B256,
    pub timestamp: u64,
    pub gas_limit: u64,
    pub base_fee: u64,
    pub coinbase: Address,
    pub excess_blob_gas: Option<u64>,
    pub transactions: Vec<RpcTransaction>,
}

/// Ethereum transaction data from RPC.
pub struct RpcTransaction {
    pub from: Address,
    pub to: Option<Address>,
    pub value: U256,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub max_fee_per_gas: Option<u64>,
    pub max_priority_fee_per_gas: Option<u64>,
    pub nonce: u64,
    pub input: Bytes,
    pub tx_type: u8,
    pub max_fee_per_blob_gas: Option<u64>,
    pub blob_versioned_hashes: Vec<B256>,
}

/// Lazy-loading Ethereum database that fetches state from RPC on cache miss.
///
/// Uses interior mutability (`RefCell`) for caches so that it can implement
/// both `Database` (mutable) and `DatabaseRef` (shared) traits. The latter
/// is needed to wrap it in `CacheDB<RpcDatabase>` for cross-transaction
/// state persistence during block replay.
pub struct RpcDatabase {
    rpc_url: String,
    block_number: u64,
    account_cache: RefCell<HashMap<Address, AccountInfo>>,
    storage_cache: RefCell<HashMap<(Address, U256), U256>>,
    code_cache: RefCell<HashMap<B256, Bytecode>>,
    block_hash_cache: RefCell<HashMap<u64, B256>>,
}

impl RpcDatabase {
    pub fn new(rpc_url: &str, block_number: u64) -> Self {
        RpcDatabase {
            rpc_url: rpc_url.to_string(),
            block_number,
            account_cache: RefCell::new(HashMap::new()),
            storage_cache: RefCell::new(HashMap::new()),
            code_cache: RefCell::new(HashMap::new()),
            block_hash_cache: RefCell::new(HashMap::new()),
        }
    }

    fn rpc_call(&self, method: &str, params: &serde_json::Value) -> Result<serde_json::Value, RpcError> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        });

        let resp = ureq::post(&self.rpc_url)
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .map_err(|e| RpcError(format!("HTTP error: {}", e)))?;

        let json: serde_json::Value = resp.into_json()
            .map_err(|e| RpcError(format!("JSON parse error: {}", e)))?;

        if let Some(error) = json.get("error") {
            return Err(RpcError(format!("RPC error: {}", error)));
        }

        json.get("result").cloned().ok_or_else(|| RpcError("No result in response".to_string()))
    }

    fn block_hex(&self) -> String {
        format!("0x{:x}", self.block_number)
    }

    fn fetch_account(&self, address: Address) -> Result<AccountInfo, RpcError> {
        let addr_hex = format!("0x{}", hex::encode(address.as_slice()));
        let block_hex = self.block_hex();

        // Fetch balance
        let balance_result = self.rpc_call("eth_getBalance", &serde_json::json!([&addr_hex, &block_hex]))?;
        let balance = parse_u256_hex(balance_result.as_str().unwrap_or("0x0"));

        // Fetch nonce
        let nonce_result = self.rpc_call("eth_getTransactionCount", &serde_json::json!([&addr_hex, &block_hex]))?;
        let nonce = parse_u64_hex(nonce_result.as_str().unwrap_or("0x0"));

        // Fetch code
        let code_result = self.rpc_call("eth_getCode", &serde_json::json!([&addr_hex, &block_hex]))?;
        let code_hex = code_result.as_str().unwrap_or("0x");
        let code_bytes = hex_decode(code_hex);

        let code = if code_bytes.is_empty() {
            None
        } else {
            Some(Bytecode::new_legacy(Bytes::from(code_bytes)))
        };

        Ok(AccountInfo {
            balance,
            nonce,
            code_hash: Default::default(),
            account_id: None,
            code,
        })
    }

    fn fetch_storage(&self, address: Address, index: U256) -> Result<U256, RpcError> {
        let addr_hex = format!("0x{}", hex::encode(address.as_slice()));
        let index_hex = format!("0x{:064x}", index);
        let block_hex = self.block_hex();

        let result = self.rpc_call("eth_getStorageAt", &serde_json::json!([&addr_hex, &index_hex, &block_hex]))?;
        Ok(parse_u256_hex(result.as_str().unwrap_or("0x0")))
    }
}

impl DatabaseRef for RpcDatabase {
    type Error = RpcError;

    fn basic_ref(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        if let Some(info) = self.account_cache.borrow().get(&address) {
            return Ok(Some(info.clone()));
        }
        let info = self.fetch_account(address)?;
        self.account_cache.borrow_mut().insert(address, info.clone());
        Ok(Some(info))
    }

    fn code_by_hash_ref(&self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        if let Some(code) = self.code_cache.borrow().get(&code_hash) {
            return Ok(code.clone());
        }
        // Search account cache for an account with matching code
        let found = {
            let cache = self.account_cache.borrow();
            cache.values()
                .find(|info| info.code_hash == code_hash && info.code.is_some())
                .and_then(|info| info.code.clone())
        };
        if let Some(code) = found {
            self.code_cache.borrow_mut().insert(code_hash, code.clone());
            return Ok(code);
        }
        Ok(Bytecode::default())
    }

    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        let key = (address, index);
        if let Some(val) = self.storage_cache.borrow().get(&key) {
            return Ok(*val);
        }
        let val = self.fetch_storage(address, index)?;
        self.storage_cache.borrow_mut().insert(key, val);
        Ok(val)
    }

    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        if let Some(hash) = self.block_hash_cache.borrow().get(&number) {
            return Ok(*hash);
        }
        let block_hex = format!("0x{:x}", number);
        let result = self.rpc_call("eth_getBlockByNumber", &serde_json::json!([&block_hex, false]))?;
        if let Some(hash_str) = result.get("hash").and_then(|h| h.as_str()) {
            let hash = parse_b256_hex(hash_str);
            self.block_hash_cache.borrow_mut().insert(number, hash);
            Ok(hash)
        } else {
            Ok(B256::ZERO)
        }
    }
}

impl Database for RpcDatabase {
    type Error = RpcError;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        self.basic_ref(address)
    }

    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        self.code_by_hash_ref(code_hash)
    }

    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        self.storage_ref(address, index)
    }

    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        self.block_hash_ref(number)
    }
}

/// Fetch a block with full transaction objects from an Ethereum RPC endpoint.
pub fn fetch_block(rpc_url: &str, block_number: u64) -> Result<RpcBlock, String> {
    let block_hex = format!("0x{:x}", block_number);
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [&block_hex, true],
        "id": 1
    });

    let resp = ureq::post(rpc_url)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("HTTP error: {}", e))?;

    let json: serde_json::Value = resp.into_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;

    let result = json.get("result").ok_or("No result in response")?;
    if result.is_null() {
        return Err(format!("Block {} not found", block_number));
    }

    let hash = parse_b256_hex(result["hash"].as_str().unwrap_or("0x0"));
    let parent_hash = parse_b256_hex(result["parentHash"].as_str().unwrap_or("0x0"));
    let timestamp = parse_u64_hex(result["timestamp"].as_str().unwrap_or("0x0"));
    let gas_limit = parse_u64_hex(result["gasLimit"].as_str().unwrap_or("0x0"));
    let base_fee = parse_u64_hex(result.get("baseFeePerGas").and_then(|v| v.as_str()).unwrap_or("0x0"));
    let coinbase = parse_address_hex(result["miner"].as_str().unwrap_or("0x0000000000000000000000000000000000000000"));
    let excess_blob_gas = result.get("excessBlobGas").and_then(|v| v.as_str()).map(parse_u64_hex);

    let txs = result["transactions"].as_array()
        .map(|arr| arr.iter().map(|tx| {
            {
                let tx_type = parse_u64_hex(tx.get("type").and_then(|v| v.as_str()).unwrap_or("0x0")) as u8;
                let max_fee = tx.get("maxFeePerGas").and_then(|v| v.as_str()).map(parse_u64_hex);
                let max_priority = tx.get("maxPriorityFeePerGas").and_then(|v| v.as_str()).map(parse_u64_hex);
                let max_fee_per_blob_gas = tx.get("maxFeePerBlobGas").and_then(|v| v.as_str()).map(parse_u64_hex);
                let blob_versioned_hashes = tx.get("blobVersionedHashes")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|h| h.as_str().map(parse_b256_hex)).collect())
                    .unwrap_or_default();
                RpcTransaction {
                    from: parse_address_hex(tx["from"].as_str().unwrap_or("0x0")),
                    to: tx["to"].as_str().map(|s| parse_address_hex(s)),
                    value: parse_u256_hex(tx["value"].as_str().unwrap_or("0x0")),
                    gas_limit: parse_u64_hex(tx["gas"].as_str().unwrap_or("0x0")),
                    gas_price: parse_u64_hex(tx.get("gasPrice").and_then(|v| v.as_str()).unwrap_or("0x0")),
                    max_fee_per_gas: max_fee,
                    max_priority_fee_per_gas: max_priority,
                    nonce: parse_u64_hex(tx["nonce"].as_str().unwrap_or("0x0")),
                    input: Bytes::from(hex_decode(tx["input"].as_str().unwrap_or("0x"))),
                    tx_type,
                    max_fee_per_blob_gas,
                    blob_versioned_hashes,
                }
            }
        }).collect())
        .unwrap_or_default();

    Ok(RpcBlock {
        number: block_number,
        hash,
        parent_hash,
        timestamp,
        gas_limit,
        base_fee,
        coinbase,
        excess_blob_gas,
        transactions: txs,
    })
}

// Helper parsing functions
fn hex_decode(s: &str) -> Vec<u8> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.is_empty() { return Vec::new(); }
    hex::decode(s).unwrap_or_default()
}

fn parse_u64_hex(s: &str) -> u64 {
    let s = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(s, 16).unwrap_or(0)
}

fn parse_u256_hex(s: &str) -> U256 {
    let bytes = hex_decode(s);
    if bytes.is_empty() { return U256::ZERO; }
    U256::from_be_slice(&bytes)
}

fn parse_b256_hex(s: &str) -> B256 {
    let bytes = hex_decode(s);
    if bytes.len() != 32 { return B256::ZERO; }
    B256::from_slice(&bytes)
}

fn parse_address_hex(s: &str) -> Address {
    let bytes = hex_decode(s);
    if bytes.len() < 20 { return Address::ZERO; }
    Address::from_slice(&bytes[bytes.len()-20..])
}

// Minimal hex encoding (no external dep needed since we already have hex_decode)
mod hex {
    pub fn decode(s: &str) -> Result<Vec<u8>, ()> {
        if s.len() % 2 != 0 { return Err(()); }
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i+2], 16).map_err(|_| ()))
            .collect()
    }

    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}
