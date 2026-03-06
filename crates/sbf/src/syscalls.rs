//! Solana BPF syscall implementations.
//!
//! Provides the minimum set of syscalls needed for real Solana programs to
//! execute under solana_rbpf. These are registered as builtin functions on
//! the program loader.

use solana_rbpf::declare_builtin_function;
use solana_rbpf::error::EbpfError;
use solana_rbpf::memory_region::{AccessType, MemoryMapping};
use solana_rbpf::program::{BuiltinFunction, FunctionRegistry};
use solana_rbpf::vm::TestContextObject;

/// Register all Solana syscalls on a function registry.
pub fn register_syscalls(
    registry: &mut FunctionRegistry<BuiltinFunction<TestContextObject>>,
) -> Result<(), String> {
    let syscalls: &[(&[u8], BuiltinFunction<TestContextObject>)] = &[
        (b"abort", SyscallAbort::vm),
        (b"sol_log_", SyscallLog::vm),
        (b"sol_log_64_", SyscallLog64::vm),
        (b"sol_log_pubkey", SyscallLogPubkey::vm),
        (b"sol_log_compute_units_", SyscallLogCu::vm),
        (b"sol_memcpy_", SyscallMemcpy::vm),
        (b"sol_memset_", SyscallMemset::vm),
        (b"sol_memmove_", SyscallMemmove::vm),
        (b"sol_memcmp_", SyscallMemcmp::vm),
        (b"sol_sha256", SyscallSha256::vm),
        (b"sol_keccak256", SyscallKeccak256::vm),
        (b"sol_set_return_data", SyscallSetReturnData::vm),
        (b"sol_get_return_data", SyscallGetReturnData::vm),
        (b"sol_get_stack_height", SyscallGetStackHeight::vm),
        (b"sol_panic_", SyscallPanic::vm),
    ];

    for (name, func) in syscalls {
        registry
            .register_function_hashed(name.to_vec(), *func)
            .map_err(|e| format!("Failed to register syscall {:?}: {:?}", String::from_utf8_lossy(name), e))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Syscall implementations
// ---------------------------------------------------------------------------

declare_builtin_function!(
    /// Abort execution.
    SyscallAbort,
    fn rust(
        _context: &mut TestContextObject,
        _arg1: u64,
        _arg2: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        _memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        Err(Box::new(EbpfError::ExceededMaxInstructions))
    }
);

declare_builtin_function!(
    /// sol_log_(msg, msg_len): Log a UTF-8 string.
    SyscallLog,
    fn rust(
        _context: &mut TestContextObject,
        msg_addr: u64,
        msg_len: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if msg_len > 0 && msg_len < 32768 {
            let host_addr: Result<u64, EbpfError> =
                memory_mapping.map(AccessType::Load, msg_addr, msg_len).into();
            if let Ok(addr) = host_addr {
                let slice = unsafe { std::slice::from_raw_parts(addr as *const u8, msg_len as usize) };
                let msg = String::from_utf8_lossy(slice);
                eprintln!("[sbf-log] {}", msg);
            }
        }
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_log_64_(a, b, c, d, e): Log five u64 values.
    SyscallLog64,
    fn rust(
        _context: &mut TestContextObject,
        arg1: u64,
        arg2: u64,
        arg3: u64,
        arg4: u64,
        arg5: u64,
        _memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        eprintln!("[sbf-log] {}, {}, {}, {}, {}", arg1, arg2, arg3, arg4, arg5);
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_log_pubkey(pubkey_addr): Log a 32-byte public key.
    SyscallLogPubkey,
    fn rust(
        _context: &mut TestContextObject,
        pubkey_addr: u64,
        _arg2: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let host_addr: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Load, pubkey_addr, 32).into();
        if let Ok(addr) = host_addr {
            let slice = unsafe { std::slice::from_raw_parts(addr as *const u8, 32) };
            eprintln!("[sbf-log] pubkey: {}", bs58_encode(slice));
        }
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_log_compute_units_(): Log remaining compute units.
    SyscallLogCu,
    fn rust(
        context: &mut TestContextObject,
        _arg1: u64,
        _arg2: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        _memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        use solana_rbpf::vm::ContextObject;
        eprintln!("[sbf-log] compute units remaining: {}", context.get_remaining());
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_memcpy_(dst, src, len): Copy memory.
    SyscallMemcpy,
    fn rust(
        _context: &mut TestContextObject,
        dst_addr: u64,
        src_addr: u64,
        len: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if len == 0 {
            return Ok(0);
        }
        let src_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Load, src_addr, len).into();
        let src_host = src_host?;
        let dst_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Store, dst_addr, len).into();
        let dst_host = dst_host?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_host as *const u8,
                dst_host as *mut u8,
                len as usize,
            );
        }
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_memset_(dst, val, len): Fill memory with a byte value.
    SyscallMemset,
    fn rust(
        _context: &mut TestContextObject,
        dst_addr: u64,
        val: u64,
        len: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if len == 0 {
            return Ok(0);
        }
        let dst_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Store, dst_addr, len).into();
        let dst_host = dst_host?;
        unsafe {
            std::ptr::write_bytes(dst_host as *mut u8, val as u8, len as usize);
        }
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_memmove_(dst, src, len): Move memory (handles overlapping).
    SyscallMemmove,
    fn rust(
        _context: &mut TestContextObject,
        dst_addr: u64,
        src_addr: u64,
        len: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if len == 0 {
            return Ok(0);
        }
        let src_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Load, src_addr, len).into();
        let src_host = src_host?;
        let dst_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Store, dst_addr, len).into();
        let dst_host = dst_host?;
        unsafe {
            std::ptr::copy(src_host as *const u8, dst_host as *mut u8, len as usize);
        }
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_memcmp_(a, b, len, result): Compare memory, write result to *result.
    SyscallMemcmp,
    fn rust(
        _context: &mut TestContextObject,
        a_addr: u64,
        b_addr: u64,
        len: u64,
        result_addr: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if len == 0 {
            let result_host: Result<u64, EbpfError> =
                memory_mapping.map(AccessType::Store, result_addr, 4).into();
            let result_host = result_host?;
            unsafe { *(result_host as *mut i32) = 0; }
            return Ok(0);
        }
        let a_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Load, a_addr, len).into();
        let a_host = a_host?;
        let b_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Load, b_addr, len).into();
        let b_host = b_host?;
        let result_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Store, result_addr, 4).into();
        let result_host = result_host?;

        let a_slice = unsafe { std::slice::from_raw_parts(a_host as *const u8, len as usize) };
        let b_slice = unsafe { std::slice::from_raw_parts(b_host as *const u8, len as usize) };

        let cmp = a_slice.cmp(b_slice) as i32;
        unsafe { *(result_host as *mut i32) = cmp; }
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_sha256(vals, val_len, hash_result): Compute SHA-256 hash.
    SyscallSha256,
    fn rust(
        _context: &mut TestContextObject,
        vals_addr: u64,
        vals_len: u64,
        hash_result_addr: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        // vals is an array of (ptr, len) pairs
        let mut hasher = sha2_sha256_new();
        for i in 0..vals_len {
            let slice_desc_addr = vals_addr + i * 16;
            let desc_host: Result<u64, EbpfError> =
                memory_mapping.map(AccessType::Load, slice_desc_addr, 16).into();
            let desc_host = desc_host?;
            let (ptr, len) = unsafe {
                let p = *(desc_host as *const u64);
                let l = *((desc_host + 8) as *const u64);
                (p, l)
            };
            if len > 0 {
                let data_host: Result<u64, EbpfError> =
                    memory_mapping.map(AccessType::Load, ptr, len).into();
                let data_host = data_host?;
                let data = unsafe { std::slice::from_raw_parts(data_host as *const u8, len as usize) };
                hasher.update(data);
            }
        }
        let hash = hasher.finalize();
        let out_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Store, hash_result_addr, 32).into();
        let out_host = out_host?;
        unsafe {
            std::ptr::copy_nonoverlapping(hash.as_ptr(), out_host as *mut u8, 32);
        }
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_keccak256(vals, val_len, hash_result): Compute Keccak-256 hash.
    SyscallKeccak256,
    fn rust(
        _context: &mut TestContextObject,
        vals_addr: u64,
        vals_len: u64,
        hash_result_addr: u64,
        _arg4: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        use sha3::Digest;
        let mut hasher = sha3::Keccak256::new();
        for i in 0..vals_len {
            let slice_desc_addr = vals_addr + i * 16;
            let desc_host: Result<u64, EbpfError> =
                memory_mapping.map(AccessType::Load, slice_desc_addr, 16).into();
            let desc_host = desc_host?;
            let (ptr, len) = unsafe {
                let p = *(desc_host as *const u64);
                let l = *((desc_host + 8) as *const u64);
                (p, l)
            };
            if len > 0 {
                let data_host: Result<u64, EbpfError> =
                    memory_mapping.map(AccessType::Load, ptr, len).into();
                let data_host = data_host?;
                let data = unsafe { std::slice::from_raw_parts(data_host as *const u8, len as usize) };
                hasher.update(data);
            }
        }
        let hash = hasher.finalize();
        let out_host: Result<u64, EbpfError> =
            memory_mapping.map(AccessType::Store, hash_result_addr, 32).into();
        let out_host = out_host?;
        unsafe {
            std::ptr::copy_nonoverlapping(hash.as_ptr(), out_host as *mut u8, 32);
        }
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_set_return_data(data, len): Set program return data (no-op for tracing).
    SyscallSetReturnData,
    fn rust(
        _context: &mut TestContextObject,
        _data_addr: u64,
        _len: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        _memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_get_return_data(data, len, program_id): Get return data (returns 0 = no data).
    SyscallGetReturnData,
    fn rust(
        _context: &mut TestContextObject,
        _data_addr: u64,
        _len: u64,
        _program_id_addr: u64,
        _arg4: u64,
        _arg5: u64,
        _memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_get_stack_height(): Get current CPI stack height (always 0 for top-level).
    SyscallGetStackHeight,
    fn rust(
        _context: &mut TestContextObject,
        _arg1: u64,
        _arg2: u64,
        _arg3: u64,
        _arg4: u64,
        _arg5: u64,
        _memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(0)
    }
);

declare_builtin_function!(
    /// sol_panic_(file, file_len, line, column): Panic with location info.
    SyscallPanic,
    fn rust(
        _context: &mut TestContextObject,
        file_addr: u64,
        file_len: u64,
        line: u64,
        column: u64,
        _arg5: u64,
        memory_mapping: &mut MemoryMapping,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let file_str = if file_len > 0 && file_len < 4096 {
            let host_addr: Result<u64, EbpfError> =
                memory_mapping.map(AccessType::Load, file_addr, file_len).into();
            if let Ok(addr) = host_addr {
                let slice = unsafe { std::slice::from_raw_parts(addr as *const u8, file_len as usize) };
                String::from_utf8_lossy(slice).to_string()
            } else {
                "<unknown>".to_string()
            }
        } else {
            "<unknown>".to_string()
        };
        eprintln!("[sbf-panic] {}:{}:{}", file_str, line, column);
        Err(Box::new(EbpfError::ExceededMaxInstructions))
    }
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal SHA-256 using the sha2 crate (via sha3's Digest trait).
/// We use a simple software implementation since sha3 is already a dependency.
fn sha2_sha256_new() -> Sha256 {
    Sha256 { state: sha2_init(), data: Vec::new(), len: 0 }
}

/// Simple software SHA-256 implementation.
struct Sha256 {
    state: [u32; 8],
    data: Vec<u8>,
    len: u64,
}

impl Sha256 {
    fn update(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
        self.len += data.len() as u64;
        while self.data.len() >= 64 {
            let block: [u8; 64] = self.data[..64].try_into().unwrap();
            sha2_compress(&mut self.state, &block);
            self.data.drain(..64);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.len * 8;
        self.data.push(0x80);
        while self.data.len() % 64 != 56 {
            self.data.push(0);
        }
        self.data.extend_from_slice(&bit_len.to_be_bytes());
        for chunk in self.data.chunks(64) {
            let block: [u8; 64] = chunk.try_into().unwrap();
            sha2_compress(&mut self.state, &block);
        }
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i*4..(i+1)*4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

fn sha2_init() -> [u32; 8] {
    [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha2_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([block[i*4], block[i*4+1], block[i*4+2], block[i*4+3]]);
    }
    for i in 16..64 {
        let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
        let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
        w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g; g = f; f = e;
        e = d.wrapping_add(temp1);
        d = c; c = b; b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
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
    for _ in 0..leading_zeros { result.push('1'); }
    for &d in digits[..digit_len].iter().rev() {
        result.push(ALPHABET[d as usize] as char);
    }
    result
}
