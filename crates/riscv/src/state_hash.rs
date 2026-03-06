use std::collections::HashMap;
use sha3::{Sha3_256, Digest};
use crate::csr::addr;
use crate::memory::Memory;
use crate::vm::Vm;

/// Key CSR addresses captured in boundary state snapshots.
const KEY_CSRS: [u16; 20] = [
    addr::MSTATUS, addr::MTVEC, addr::MEPC, addr::MCAUSE,
    addr::MTVAL, addr::MIE, addr::MIP, addr::MSCRATCH,
    addr::MEDELEG, addr::MIDELEG, addr::MCOUNTEREN,
    addr::SSTATUS, addr::STVEC, addr::SEPC, addr::SCAUSE,
    addr::STVAL, addr::SIE, addr::SIP, addr::SATP,
    addr::STIMECMP,
];

/// Captures the complete VM boundary state at a chunk boundary.
/// Used for state chaining: each chunk proves state_initial -> state_final,
/// and adjacent chunks must have matching boundary hashes.
#[derive(Clone, Debug)]
pub struct ChunkBoundaryState {
    pub pc: u64,
    pub registers: [u64; 32],
    pub priv_mode: u8,
    pub key_csrs: [u64; 20],
    pub memory_hash: [u8; 32],
    pub uart_output_hash: [u8; 32],
    pub clint_mtime: u64,
    pub clint_mtimecmp: u64,
    pub step_number: u64,
}

/// Hash a boundary state into a 32-byte digest for state chaining.
pub fn hash_boundary_state(state: &ChunkBoundaryState) -> [u8; 32] {
    let mut hasher = Sha3_256::new();

    hasher.update(state.pc.to_le_bytes());
    for &reg in &state.registers {
        hasher.update(reg.to_le_bytes());
    }
    hasher.update([state.priv_mode]);
    for &csr in &state.key_csrs {
        hasher.update(csr.to_le_bytes());
    }
    hasher.update(state.memory_hash);
    hasher.update(state.uart_output_hash);
    hasher.update(state.clint_mtime.to_le_bytes());
    hasher.update(state.clint_mtimecmp.to_le_bytes());
    hasher.update(state.step_number.to_le_bytes());

    hasher.finalize().into()
}

/// Snapshot the current VM state into a ChunkBoundaryState.
pub fn snapshot_vm(
    vm: &Vm,
    memory_hash: [u8; 32],
    uart_output_hash: [u8; 32],
    step_number: u64,
) -> ChunkBoundaryState {
    let mut key_csrs = [0u64; 20];
    for (i, &csr_addr) in KEY_CSRS.iter().enumerate() {
        key_csrs[i] = vm.cpu.csrs.read_unchecked(csr_addr);
    }

    ChunkBoundaryState {
        pc: vm.cpu.pc,
        registers: *vm.cpu.registers(),
        priv_mode: vm.cpu.priv_mode as u8,
        key_csrs,
        memory_hash,
        uart_output_hash,
        clint_mtime: vm.mmio.clint.mtime,
        clint_mtimecmp: vm.mmio.clint.mtimecmp,
        step_number,
    }
}

/// Incrementally tracks memory page hashes, only re-hashing dirty pages.
pub struct IncrementalMemoryHash {
    page_hashes: HashMap<u64, [u8; 32]>,
}

impl IncrementalMemoryHash {
    pub fn new() -> Self {
        IncrementalMemoryHash {
            page_hashes: HashMap::new(),
        }
    }

    /// Update dirty page hashes and return the combined memory hash.
    /// Only re-hashes pages that were modified since the last call.
    pub fn update(&mut self, memory: &mut Memory) -> [u8; 32] {
        // Re-hash only dirty pages
        for page_num in memory.dirty_pages() {
            if let Some(data) = memory.page_data(page_num) {
                let mut hasher = Sha3_256::new();
                hasher.update(page_num.to_le_bytes());
                hasher.update(data);
                let hash: [u8; 32] = hasher.finalize().into();
                self.page_hashes.insert(page_num, hash);
            }
        }
        memory.clear_dirty();

        // Combine all page hashes in sorted order for determinism
        let mut sorted_pages: Vec<u64> = self.page_hashes.keys().copied().collect();
        sorted_pages.sort_unstable();

        let mut hasher = Sha3_256::new();
        hasher.update((sorted_pages.len() as u64).to_le_bytes());
        for page_num in &sorted_pages {
            hasher.update(self.page_hashes[page_num]);
        }

        hasher.finalize().into()
    }

    /// Initialize by hashing all currently allocated pages.
    pub fn initialize(&mut self, memory: &Memory) -> [u8; 32] {
        self.page_hashes.clear();

        for page_num in memory.page_numbers() {
            if let Some(data) = memory.page_data(page_num) {
                let mut hasher = Sha3_256::new();
                hasher.update(page_num.to_le_bytes());
                hasher.update(data);
                let hash: [u8; 32] = hasher.finalize().into();
                self.page_hashes.insert(page_num, hash);
            }
        }

        let mut sorted_pages: Vec<u64> = self.page_hashes.keys().copied().collect();
        sorted_pages.sort_unstable();

        let mut hasher = Sha3_256::new();
        hasher.update((sorted_pages.len() as u64).to_le_bytes());
        for page_num in &sorted_pages {
            hasher.update(self.page_hashes[page_num]);
        }

        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuState;

    #[test]
    fn test_hash_boundary_state_deterministic() {
        let state = ChunkBoundaryState {
            pc: 0x8000_0000,
            registers: [0u64; 32],
            priv_mode: 3,
            key_csrs: [0u64; 20],
            memory_hash: [0u8; 32],
            uart_output_hash: [0u8; 32],
            clint_mtime: 0,
            clint_mtimecmp: 0,
            step_number: 0,
        };

        let h1 = hash_boundary_state(&state);
        let h2 = hash_boundary_state(&state);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_boundary_state_differs_on_pc_change() {
        let mut state = ChunkBoundaryState {
            pc: 0x8000_0000,
            registers: [0u64; 32],
            priv_mode: 3,
            key_csrs: [0u64; 20],
            memory_hash: [0u8; 32],
            uart_output_hash: [0u8; 32],
            clint_mtime: 0,
            clint_mtimecmp: 0,
            step_number: 0,
        };

        let h1 = hash_boundary_state(&state);
        state.pc = 0x8000_0004;
        let h2 = hash_boundary_state(&state);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_incremental_memory_hash_empty() {
        let mut hasher = IncrementalMemoryHash::new();
        let mem = Memory::new();
        let h = hasher.initialize(&mem);
        // Should produce a deterministic hash for empty memory
        let h2 = hasher.initialize(&mem);
        assert_eq!(h, h2);
    }

    #[test]
    fn test_incremental_memory_hash_tracks_changes() {
        let mut hasher = IncrementalMemoryHash::new();
        let mut mem = Memory::new();

        let h1 = hasher.initialize(&mem);

        mem.store_word(0x1000, 0xDEADBEEF);
        let h2 = hasher.update(&mut mem);
        assert_ne!(h1, h2);

        // No changes: should return same hash
        let h3 = hasher.update(&mut mem);
        assert_eq!(h2, h3);
    }

    #[test]
    fn test_snapshot_vm() {
        let cpu = CpuState::with_pc(0x8000_0000);
        let mem = Memory::new();
        let vm = Vm::new(cpu, mem);

        let state = snapshot_vm(&vm, [0u8; 32], [0u8; 32], 0);
        assert_eq!(state.pc, 0x8000_0000);
        assert_eq!(state.priv_mode, 3); // Machine mode
        assert_eq!(state.step_number, 0);
    }
}
