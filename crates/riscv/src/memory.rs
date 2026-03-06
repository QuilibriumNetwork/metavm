use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MemoryError {
    #[error("misaligned access at address 0x{addr:x} for {size}-byte access")]
    MisalignedAccess { addr: u64, size: u64 },
}

const PAGE_BITS: usize = 12;
const PAGE_SIZE: usize = 1 << PAGE_BITS; // 4096
const PAGE_MASK: u64 = (PAGE_SIZE - 1) as u64;

type Page = Box<[u8; PAGE_SIZE]>;

fn new_page() -> Page {
    Box::new([0u8; PAGE_SIZE])
}

/// Page-based byte-addressable memory using 4KB pages.
/// Uses little-endian byte ordering. O(1) amortized access.
#[derive(Clone, Default)]
pub struct Memory {
    pages: HashMap<u64, Page>,
    dirty: HashSet<u64>,
}

impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory")
            .field("pages", &self.pages.len())
            .finish()
    }
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            pages: HashMap::new(),
            dirty: HashSet::new(),
        }
    }

    /// Pre-allocate pages for a memory region (avoids repeated allocation).
    pub fn pre_allocate(&mut self, base: u64, size: u64) {
        let start_page = base >> PAGE_BITS;
        let end_page = (base + size + PAGE_MASK) >> PAGE_BITS;
        for page_num in start_page..end_page {
            self.pages.entry(page_num).or_insert_with(new_page);
        }
    }

    /// Load a single byte from memory.
    #[inline(always)]
    pub fn load_byte(&self, addr: u64) -> u8 {
        let page_num = addr >> PAGE_BITS;
        match self.pages.get(&page_num) {
            Some(page) => page[(addr & PAGE_MASK) as usize],
            None => 0,
        }
    }

    /// Store a single byte to memory.
    #[inline(always)]
    pub fn store_byte(&mut self, addr: u64, val: u8) {
        let page_num = addr >> PAGE_BITS;
        let offset = (addr & PAGE_MASK) as usize;
        let page = self.pages.entry(page_num).or_insert_with(new_page);
        page[offset] = val;
        self.dirty.insert(page_num);
    }

    /// Load a 16-bit halfword (little-endian).
    #[inline(always)]
    pub fn load_half(&self, addr: u64) -> u16 {
        let page_num = addr >> PAGE_BITS;
        let offset = (addr & PAGE_MASK) as usize;
        if offset + 2 <= PAGE_SIZE {
            if let Some(page) = self.pages.get(&page_num) {
                return u16::from_le_bytes([page[offset], page[offset + 1]]);
            }
            return 0;
        }
        // Cross-page boundary (rare)
        let lo = self.load_byte(addr) as u16;
        let hi = self.load_byte(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    /// Store a 16-bit halfword (little-endian).
    #[inline(always)]
    pub fn store_half(&mut self, addr: u64, val: u16) {
        let page_num = addr >> PAGE_BITS;
        let offset = (addr & PAGE_MASK) as usize;
        if offset + 2 <= PAGE_SIZE {
            let page = self.pages.entry(page_num).or_insert_with(new_page);
            let bytes = val.to_le_bytes();
            page[offset] = bytes[0];
            page[offset + 1] = bytes[1];
            self.dirty.insert(page_num);
            return;
        }
        self.store_byte(addr, val as u8);
        self.store_byte(addr.wrapping_add(1), (val >> 8) as u8);
    }

    /// Load a 32-bit word (little-endian).
    #[inline(always)]
    pub fn load_word(&self, addr: u64) -> u32 {
        let page_num = addr >> PAGE_BITS;
        let offset = (addr & PAGE_MASK) as usize;
        if offset + 4 <= PAGE_SIZE {
            if let Some(page) = self.pages.get(&page_num) {
                let bytes: [u8; 4] = page[offset..offset + 4].try_into().unwrap();
                return u32::from_le_bytes(bytes);
            }
            return 0;
        }
        // Cross-page boundary
        let lo = self.load_half(addr) as u32;
        let hi = self.load_half(addr.wrapping_add(2)) as u32;
        lo | (hi << 16)
    }

    /// Store a 32-bit word (little-endian).
    #[inline(always)]
    pub fn store_word(&mut self, addr: u64, val: u32) {
        let page_num = addr >> PAGE_BITS;
        let offset = (addr & PAGE_MASK) as usize;
        if offset + 4 <= PAGE_SIZE {
            let page = self.pages.entry(page_num).or_insert_with(new_page);
            page[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
            self.dirty.insert(page_num);
            return;
        }
        self.store_half(addr, val as u16);
        self.store_half(addr.wrapping_add(2), (val >> 16) as u16);
    }

    /// Load a 64-bit doubleword (little-endian).
    #[inline(always)]
    pub fn load_double(&self, addr: u64) -> u64 {
        let page_num = addr >> PAGE_BITS;
        let offset = (addr & PAGE_MASK) as usize;
        if offset + 8 <= PAGE_SIZE {
            if let Some(page) = self.pages.get(&page_num) {
                let bytes: [u8; 8] = page[offset..offset + 8].try_into().unwrap();
                return u64::from_le_bytes(bytes);
            }
            return 0;
        }
        // Cross-page boundary
        let lo = self.load_word(addr) as u64;
        let hi = self.load_word(addr.wrapping_add(4)) as u64;
        lo | (hi << 32)
    }

    /// Store a 64-bit doubleword (little-endian).
    #[inline(always)]
    pub fn store_double(&mut self, addr: u64, val: u64) {
        let page_num = addr >> PAGE_BITS;
        let offset = (addr & PAGE_MASK) as usize;
        if offset + 8 <= PAGE_SIZE {
            let page = self.pages.entry(page_num).or_insert_with(new_page);
            page[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
            self.dirty.insert(page_num);
            return;
        }
        self.store_word(addr, val as u32);
        self.store_word(addr.wrapping_add(4), (val >> 32) as u32);
    }

    /// Load a program (raw binary bytes) into memory starting at `base_addr`.
    pub fn load_program(&mut self, base_addr: u64, bytes: &[u8]) {
        let mut addr = base_addr;
        let mut remaining = bytes;
        while !remaining.is_empty() {
            let page_num = addr >> PAGE_BITS;
            let offset = (addr & PAGE_MASK) as usize;
            let page = self.pages.entry(page_num).or_insert_with(new_page);
            let n = remaining.len().min(PAGE_SIZE - offset);
            page[offset..offset + n].copy_from_slice(&remaining[..n]);
            self.dirty.insert(page_num);
            addr += n as u64;
            remaining = &remaining[n..];
        }
    }

    /// Get sorted page numbers of all allocated pages.
    pub fn page_numbers(&self) -> Vec<u64> {
        let mut nums: Vec<u64> = self.pages.keys().copied().collect();
        nums.sort_unstable();
        nums
    }

    /// Get the raw data of a page by page number.
    pub fn page_data(&self, page_num: u64) -> Option<&[u8; PAGE_SIZE]> {
        self.pages.get(&page_num).map(|p| p.as_ref())
    }

    /// Get the set of dirty page numbers (modified since last clear).
    pub fn dirty_pages(&self) -> Vec<u64> {
        self.dirty.iter().copied().collect()
    }

    /// Clear the dirty page set.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Get the number of allocated pages.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get the number of non-zero bytes stored (expensive, for tests only).
    pub fn size(&self) -> usize {
        self.pages.values()
            .map(|page| page.iter().filter(|&&b| b != 0).count())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_load_store() {
        let mut mem = Memory::new();
        mem.store_byte(0x100, 0xAB);
        assert_eq!(mem.load_byte(0x100), 0xAB);
        assert_eq!(mem.load_byte(0x101), 0x00);
    }

    #[test]
    fn test_half_load_store() {
        let mut mem = Memory::new();
        mem.store_half(0x200, 0xBEEF);
        assert_eq!(mem.load_half(0x200), 0xBEEF);
        assert_eq!(mem.load_byte(0x200), 0xEF);
        assert_eq!(mem.load_byte(0x201), 0xBE);
    }

    #[test]
    fn test_word_load_store() {
        let mut mem = Memory::new();
        mem.store_word(0x300, 0xDEADBEEF);
        assert_eq!(mem.load_word(0x300), 0xDEADBEEF);
    }

    #[test]
    fn test_double_load_store() {
        let mut mem = Memory::new();
        mem.store_double(0x400, 0xCAFEBABE_DEADBEEF);
        assert_eq!(mem.load_double(0x400), 0xCAFEBABE_DEADBEEF);
        assert_eq!(mem.load_word(0x400), 0xDEADBEEF);
        assert_eq!(mem.load_word(0x404), 0xCAFEBABE);
    }

    #[test]
    fn test_load_program() {
        let mut mem = Memory::new();
        let program = [0x13, 0x05, 0x10, 0x00]; // ADDI x10, x0, 1
        mem.load_program(0x1000, &program);
        assert_eq!(mem.load_word(0x1000), 0x00100513);
    }

    #[test]
    fn test_sparse_memory() {
        let mut mem = Memory::new();
        mem.store_byte(0, 1);
        mem.store_byte(0xFFFF_FFFF_FFFF_FFFF, 2);
        assert_eq!(mem.size(), 2);
    }

    #[test]
    fn test_overwrite() {
        let mut mem = Memory::new();
        mem.store_word(0x100, 0xAAAA_BBBB);
        mem.store_word(0x100, 0xCCCC_DDDD);
        assert_eq!(mem.load_word(0x100), 0xCCCC_DDDD);
    }

    #[test]
    fn test_cross_page_load_store() {
        let mut mem = Memory::new();
        // Store a doubleword crossing a page boundary
        let addr = (PAGE_SIZE as u64) - 4; // 4 bytes before page end
        mem.store_double(addr, 0xDEADBEEF_CAFEBABE);
        assert_eq!(mem.load_double(addr), 0xDEADBEEF_CAFEBABE);
    }

    #[test]
    fn test_pre_allocate() {
        let mut mem = Memory::new();
        mem.pre_allocate(0x1000, 0x2000);
        assert_eq!(mem.page_count(), 2);
        // Pre-allocated pages should be zero-filled
        assert_eq!(mem.load_byte(0x1000), 0);
    }
}
