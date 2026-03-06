use crate::memory::Memory;

/// Type of memory access for permission checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessType {
    Execute,
    Read,
    Write,
}

/// Page fault information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageFault {
    pub access_type: AccessType,
    pub vaddr: u64,
}

impl PageFault {
    /// Return the exception cause code for this page fault.
    pub fn cause(&self) -> u64 {
        match self.access_type {
            AccessType::Execute => 12, // Instruction page fault
            AccessType::Read => 13,    // Load page fault
            AccessType::Write => 15,   // Store/AMO page fault
        }
    }
}

/// Page table entry with accessor methods.
#[derive(Clone, Copy, Debug)]
struct PageTableEntry(u64);

impl PageTableEntry {
    fn valid(&self) -> bool { self.0 & 1 != 0 }
    fn read(&self) -> bool { self.0 & (1 << 1) != 0 }
    fn write(&self) -> bool { self.0 & (1 << 2) != 0 }
    fn execute(&self) -> bool { self.0 & (1 << 3) != 0 }
    fn user(&self) -> bool { self.0 & (1 << 4) != 0 }
    fn accessed(&self) -> bool { self.0 & (1 << 6) != 0 }
    fn dirty(&self) -> bool { self.0 & (1 << 7) != 0 }

    fn is_leaf(&self) -> bool {
        self.read() || self.execute()
    }

    fn ppn(&self) -> u64 {
        (self.0 >> 10) & 0xFFF_FFFF_FFFF // 44 bits
    }

    fn ppn_i(&self, level: usize) -> u64 {
        match level {
            0 => (self.0 >> 10) & 0x1FF,
            1 => (self.0 >> 19) & 0x1FF,
            2 => (self.0 >> 28) & 0x3FF_FFFF, // 26 bits for level 2
            _ => 0,
        }
    }
}

/// Translate a virtual address using Sv39 page tables.
///
/// Returns `Ok((physical_addr, pte_update, pte_flags))` on success, where
/// `pte_update` is `Some((pte_addr, new_pte))` if the A/D bits need updating,
/// and `pte_flags` contains the raw PTE permission bits (R=bit1, W=bit2, X=bit3).
///
/// Arguments:
/// - `memory`: physical memory to read page tables from
/// - `satp`: the SATP register value (mode in bits[63:60], PPN in bits[43:0])
/// - `vaddr`: the virtual address to translate
/// - `access_type`: Read/Write/Execute
/// - `privilege`: the effective privilege for this access
/// - `sum`: mstatus.SUM (allow S-mode access to U pages)
/// - `mxr`: mstatus.MXR (make executable pages readable)
pub fn translate(
    memory: &Memory,
    satp: u64,
    vaddr: u64,
    access_type: AccessType,
    privilege: u8, // 0=U, 1=S, 3=M
    sum: bool,
    mxr: bool,
) -> Result<(u64, Option<(u64, u64)>, u8), PageFault> {
    let mode = (satp >> 60) & 0xF;

    // Mode 0 = Bare (no translation)
    if mode == 0 {
        return Ok((vaddr, None, 0xE)); // full RWX
    }

    // We only support Sv39 (mode 8)
    if mode != 8 {
        return Err(PageFault { access_type, vaddr });
    }

    // Check canonical address: bits [63:39] must all equal bit 38
    let bit38 = (vaddr >> 38) & 1;
    let upper = vaddr >> 39;
    if bit38 == 0 && upper != 0 {
        return Err(PageFault { access_type, vaddr });
    }
    if bit38 == 1 && upper != 0x1FFFFFF {
        return Err(PageFault { access_type, vaddr });
    }

    let root_ppn = satp & 0xFFF_FFFF_FFFF; // 44-bit PPN
    let vpn = [
        (vaddr >> 12) & 0x1FF, // VPN[0]
        (vaddr >> 21) & 0x1FF, // VPN[1]
        (vaddr >> 30) & 0x1FF, // VPN[2]
    ];

    let mut a = root_ppn << 12;

    for level in (0..3).rev() {
        let pte_addr = a + vpn[level] * 8;
        let pte_val = memory.load_double(pte_addr);
        let pte = PageTableEntry(pte_val);

        if !pte.valid() {
            return Err(PageFault { access_type, vaddr });
        }

        // Reserved: W=1, R=0
        if pte.write() && !pte.read() {
            return Err(PageFault { access_type, vaddr });
        }

        if pte.is_leaf() {
            // Superpage alignment check
            if level >= 1 && pte.ppn_i(0) != 0 {
                return Err(PageFault { access_type, vaddr });
            }
            if level == 2 && pte.ppn_i(1) != 0 {
                return Err(PageFault { access_type, vaddr });
            }

            // Permission checks
            match access_type {
                AccessType::Read => {
                    if !pte.read() && !(mxr && pte.execute()) {
                        return Err(PageFault { access_type, vaddr });
                    }
                }
                AccessType::Write => {
                    if !pte.write() {
                        return Err(PageFault { access_type, vaddr });
                    }
                }
                AccessType::Execute => {
                    if !pte.execute() {
                        return Err(PageFault { access_type, vaddr });
                    }
                }
            }

            // U-bit checks
            if pte.user() {
                if privilege == 1 && !sum {
                    // S-mode accessing U page without SUM
                    return Err(PageFault { access_type, vaddr });
                }
            } else {
                if privilege == 0 {
                    // U-mode accessing S page
                    return Err(PageFault { access_type, vaddr });
                }
            }

            // A/D bit management
            let needs_a = !pte.accessed();
            let needs_d = access_type == AccessType::Write && !pte.dirty();
            let pte_update = if needs_a || needs_d {
                let mut new_pte = pte_val | (1 << 6); // set A
                if needs_d {
                    new_pte |= 1 << 7; // set D
                }
                Some((pte_addr, new_pte))
            } else {
                None
            };

            // Construct physical address
            let page_offset = vaddr & 0xFFF;
            let pa = match level {
                0 => {
                    // 4KB page
                    (pte.ppn() << 12) | page_offset
                }
                1 => {
                    // 2MB megapage
                    (pte.ppn() & !0x1FF) << 12
                        | (vpn[0] << 12)
                        | page_offset
                }
                2 => {
                    // 1GB gigapage
                    (pte.ppn() & !0x3FFFF) << 12
                        | (vpn[1] << 21)
                        | (vpn[0] << 12)
                        | page_offset
                }
                _ => unreachable!(),
            };

            return Ok((pa, pte_update, (pte_val & 0xE) as u8));
        }

        // Non-leaf PTE: next level
        a = pte.ppn() << 12;
    }

    // Ran out of levels without finding a leaf
    Err(PageFault { access_type, vaddr })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;

    const PTE_V: u64 = 1 << 0;
    const PTE_R: u64 = 1 << 1;
    const PTE_W: u64 = 1 << 2;
    const PTE_X: u64 = 1 << 3;
    const PTE_U: u64 = 1 << 4;
    const PTE_A: u64 = 1 << 6;
    const PTE_D: u64 = 1 << 7;

    fn make_pte(ppn: u64, flags: u64) -> u64 {
        (ppn << 10) | flags
    }

    #[test]
    fn test_bare_mode() {
        let mem = Memory::new();
        let satp = 0; // mode=0 (Bare)
        let (pa, update, _) = translate(&mem, satp, 0x1234, AccessType::Read, 3, false, false).unwrap();
        assert_eq!(pa, 0x1234);
        assert!(update.is_none());
    }

    #[test]
    fn test_three_level_walk() {
        let mut mem = Memory::new();
        // Set up a 3-level page table
        let root_page = 0x8_0000; // PPN of root table
        let l1_page = 0x8_1000;   // physical address of level 1 table
        let l0_page = 0x8_2000;   // physical address of level 0 table
        let target_page = 0x8_3000; // physical address of mapped page

        // Root (level 2) PTE: points to l1_page
        let l1_ppn = l1_page >> 12;
        mem.store_double(root_page * 4096, make_pte(l1_ppn, PTE_V));

        // Level 1 PTE: points to l0_page
        let l0_ppn = l0_page >> 12;
        mem.store_double(l1_page, make_pte(l0_ppn, PTE_V));

        // Level 0 PTE: leaf page with RWX
        let target_ppn = target_page >> 12;
        mem.store_double(l0_page, make_pte(target_ppn, PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D));

        // satp: mode=8 (Sv39), PPN=root_page
        let satp = (8u64 << 60) | root_page;
        // Virtual address: VPN[2]=0, VPN[1]=0, VPN[0]=0, offset=0x42
        let vaddr = 0x42;

        let (pa, _, _) = translate(&mem, satp, vaddr, AccessType::Read, 1, false, false).unwrap();
        assert_eq!(pa, target_page + 0x42);
    }

    #[test]
    fn test_megapage() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        let l1_page = 0x8_1000;
        let target_ppn = 0x200; // Aligned to 2MB boundary (ppn[0] = 0)

        // Root PTE -> l1
        mem.store_double(root_page * 4096, make_pte(l1_page >> 12, PTE_V));
        // L1 PTE: leaf megapage
        mem.store_double(l1_page, make_pte(target_ppn << 9, PTE_V | PTE_R | PTE_W | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        // VPN[2]=0, VPN[1]=0, VPN[0]=5, offset=0x100
        let vaddr = (5u64 << 12) | 0x100;

        let (pa, _, _) = translate(&mem, satp, vaddr, AccessType::Read, 1, false, false).unwrap();
        // megapage: ppn[1:2] from PTE, vpn[0] from vaddr
        assert_eq!(pa, (target_ppn << 9 << 12) | (5 << 12) | 0x100);
    }

    #[test]
    fn test_permission_denied_write_to_readonly() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        // Single-level: root PTE is a gigapage (level 2 leaf)
        // ppn[1] and ppn[0] must be 0 for gigapage
        mem.store_double(root_page * 4096, make_pte(0, PTE_V | PTE_R | PTE_X | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        let result = translate(&mem, satp, 0x100, AccessType::Write, 1, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_permission_denied_exec_non_exec() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        mem.store_double(root_page * 4096, make_pte(0, PTE_V | PTE_R | PTE_W | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        let result = translate(&mem, satp, 0x100, AccessType::Execute, 1, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_u_bit_without_sum() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        mem.store_double(root_page * 4096, make_pte(0, PTE_V | PTE_R | PTE_W | PTE_X | PTE_U | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        // S-mode without SUM should fail on U page
        let result = translate(&mem, satp, 0x100, AccessType::Read, 1, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_u_bit_with_sum() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        mem.store_double(root_page * 4096, make_pte(0, PTE_V | PTE_R | PTE_W | PTE_X | PTE_U | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        // S-mode with SUM should succeed on U page
        let result = translate(&mem, satp, 0x100, AccessType::Read, 1, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mxr() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        // Page is X-only (no R)
        mem.store_double(root_page * 4096, make_pte(0, PTE_V | PTE_X | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        // Without MXR: read should fail
        let result = translate(&mem, satp, 0x100, AccessType::Read, 1, false, false);
        assert!(result.is_err());
        // With MXR: read should succeed
        let result = translate(&mem, satp, 0x100, AccessType::Read, 1, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_pte() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        // PTE with V=0
        mem.store_double(root_page * 4096, 0);

        let satp = (8u64 << 60) | root_page;
        let result = translate(&mem, satp, 0x100, AccessType::Read, 1, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_reserved_w_without_r() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        // W=1, R=0 is reserved
        mem.store_double(root_page * 4096, make_pte(0, PTE_V | PTE_W | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        let result = translate(&mem, satp, 0x100, AccessType::Read, 1, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_misaligned_superpage() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        let l1_page = 0x8_1000;

        mem.store_double(root_page * 4096, make_pte(l1_page >> 12, PTE_V));
        // Megapage with ppn[0] != 0 (misaligned)
        mem.store_double(l1_page, make_pte(1, PTE_V | PTE_R | PTE_W | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        let result = translate(&mem, satp, 0x100, AccessType::Read, 1, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_canonical_address() {
        let mem = Memory::new();
        let satp = (8u64 << 60) | 0x8_0000;
        // Non-canonical: bit 38 = 0 but upper bits != 0
        let vaddr = 0x0000_0100_0000_0000; // bit 40 set, bit 38 = 0
        let result = translate(&mem, satp, vaddr, AccessType::Read, 1, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_ad_bit_updates() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        // Page without A and D bits set
        mem.store_double(root_page * 4096, make_pte(0, PTE_V | PTE_R | PTE_W | PTE_X));

        let satp = (8u64 << 60) | root_page;
        let (_, update, _) = translate(&mem, satp, 0x100, AccessType::Write, 1, false, false).unwrap();
        // Should request A and D bits be set
        assert!(update.is_some());
        let (addr, new_pte) = update.unwrap();
        assert_ne!(new_pte & PTE_A, 0);
        assert_ne!(new_pte & PTE_D, 0);
    }

    #[test]
    fn test_u_mode_accessing_s_page_fails() {
        let mut mem = Memory::new();
        let root_page = 0x8_0000;
        // S-mode page (U bit not set)
        mem.store_double(root_page * 4096, make_pte(0, PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D));

        let satp = (8u64 << 60) | root_page;
        // U-mode should fail
        let result = translate(&mem, satp, 0x100, AccessType::Read, 0, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_page_fault_cause_codes() {
        let pf_exec = PageFault { access_type: AccessType::Execute, vaddr: 0 };
        assert_eq!(pf_exec.cause(), 12);
        let pf_read = PageFault { access_type: AccessType::Read, vaddr: 0 };
        assert_eq!(pf_read.cause(), 13);
        let pf_write = PageFault { access_type: AccessType::Write, vaddr: 0 };
        assert_eq!(pf_write.cause(), 15);
    }
}
