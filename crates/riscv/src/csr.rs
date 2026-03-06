use thiserror::Error;

/// RISC-V privilege levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum PrivilegeMode {
    User = 0,
    Supervisor = 1,
    Machine = 3,
}

impl PrivilegeMode {
    pub fn from_bits(bits: u64) -> Self {
        match bits & 0x3 {
            0 => PrivilegeMode::User,
            1 => PrivilegeMode::Supervisor,
            3 => PrivilegeMode::Machine,
            _ => PrivilegeMode::Machine,
        }
    }
}

/// CSR address constants.
pub mod addr {
    // Machine-level CSRs
    pub const MSTATUS: u16 = 0x300;
    pub const MISA: u16 = 0x301;
    pub const MEDELEG: u16 = 0x302;
    pub const MIDELEG: u16 = 0x303;
    pub const MIE: u16 = 0x304;
    pub const MTVEC: u16 = 0x305;
    pub const MCOUNTEREN: u16 = 0x306;
    pub const MSCRATCH: u16 = 0x340;
    pub const MEPC: u16 = 0x341;
    pub const MCAUSE: u16 = 0x342;
    pub const MTVAL: u16 = 0x343;
    pub const MIP: u16 = 0x344;
    pub const MHARTID: u16 = 0xF14;

    // Supervisor-level CSRs
    pub const SSTATUS: u16 = 0x100;
    pub const SIE: u16 = 0x104;
    pub const STVEC: u16 = 0x105;
    pub const SCOUNTEREN: u16 = 0x106;
    pub const SSCRATCH: u16 = 0x140;
    pub const SEPC: u16 = 0x141;
    pub const SCAUSE: u16 = 0x142;
    pub const STVAL: u16 = 0x143;
    pub const SIP: u16 = 0x144;
    pub const SATP: u16 = 0x180;

    // Supervisor timer compare (sstc extension)
    pub const STIMECMP: u16 = 0x14D;

    // User-level CSRs (read-only counters)
    pub const CYCLE: u16 = 0xC00;
    pub const TIME: u16 = 0xC01;
    pub const INSTRET: u16 = 0xC02;
}

/// mstatus field bit positions and masks.
pub mod mstatus {
    pub const SIE: u64 = 1 << 1;
    pub const MIE: u64 = 1 << 3;
    pub const SPIE: u64 = 1 << 5;
    pub const MPIE: u64 = 1 << 7;
    pub const SPP: u64 = 1 << 8;
    pub const MPP_MASK: u64 = 0x3 << 11;
    pub const MPP_SHIFT: u32 = 11;
    pub const MPRV: u64 = 1 << 17;
    pub const SUM: u64 = 1 << 18;
    pub const MXR: u64 = 1 << 19;
    pub const TVM: u64 = 1 << 20;
    pub const TW: u64 = 1 << 21;
    pub const TSR: u64 = 1 << 22;
    pub const UXL_MASK: u64 = 0x3 << 32;
    pub const SXL_MASK: u64 = 0x3 << 34;
    pub const SD: u64 = 1 << 63;

    /// Bits writable in mstatus.
    pub const WRITE_MASK: u64 = SIE | MIE | SPIE | MPIE | SPP | MPP_MASK
        | MPRV | SUM | MXR | TVM | TW | TSR | UXL_MASK | SXL_MASK;

    /// Bits visible through sstatus (supervisor view of mstatus).
    pub const SSTATUS_MASK: u64 = SIE | SPIE | SPP | SUM | MXR | UXL_MASK | SD;
}

/// Interrupt enable/pending bit positions (shared by mie/mip/sie/sip).
pub mod interrupt {
    pub const SSIP: u64 = 1 << 1;
    pub const MSIP: u64 = 1 << 3;
    pub const STIP: u64 = 1 << 5;
    pub const MTIP: u64 = 1 << 7;
    pub const SEIP: u64 = 1 << 9;
    pub const MEIP: u64 = 1 << 11;

    /// Bits visible through sie/sip (supervisor view of mie/mip).
    pub const S_MASK: u64 = SSIP | STIP | SEIP;
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CsrError {
    #[error("insufficient privilege for CSR 0x{addr:03x}: requires {required:?}, current {current:?}")]
    InsufficientPrivilege {
        addr: u16,
        required: PrivilegeMode,
        current: PrivilegeMode,
    },
    #[error("CSR 0x{0:03x} is read-only")]
    ReadOnly(u16),
    #[error("CSR 0x{0:03x} is not implemented")]
    Unimplemented(u16),
}

/// Returns true if the CSR address belongs to an unimplemented extension
/// whose probe CSRs should trap (to prevent false extension detection).
fn is_unimplemented_csr(csr_addr: u16) -> bool {
    match csr_addr {
        // smaia (Advanced Interrupt Architecture) — not implemented
        0x150..=0x157 => true, // siselect, sireg, sireg2-6
        0x350..=0x357 => true, // miselect, mireg, mireg2-6
        0xDB0 => true,         // stopi
        0xFB0 => true,         // mtopi
        // sscofpmf (Supervisor Counter Overflow PMF) — not implemented
        0xDA0 => true,         // scountovf
        // smstateen (State Enable) — not implemented
        0x10C..=0x10F => true, // sstateen0-3
        0x30C..=0x30F => true, // mstateen0-3
        0x31C..=0x31F => true, // mstateen0h-3h
        _ => false,
    }
}

/// The CSR register file.
#[derive(Clone)]
pub struct CsrFile {
    regs: [u64; 4096],
    pub instret: u64,
    /// Mirror of CLINT mtime, returned when TIME CSR is read.
    pub mtime: u64,
}

impl std::fmt::Debug for CsrFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsrFile")
            .field("instret", &self.instret)
            .field("mstatus", &self.regs[addr::MSTATUS as usize])
            .field("misa", &self.regs[addr::MISA as usize])
            .finish()
    }
}

/// misa bit for each extension letter: bit 0 = A, bit 2 = C, etc.
fn misa_bit(ch: char) -> u64 {
    1u64 << ((ch as u8 - b'A') as u64)
}

impl CsrFile {
    pub fn new() -> Self {
        let mut csrs = CsrFile {
            regs: [0u64; 4096],
            instret: 0,
            mtime: 0,
        };
        // misa: MXL=2 (64-bit) in bits [63:62], extensions I, M, A, C, S, U
        let misa = (2u64 << 62)
            | misa_bit('I')
            | misa_bit('M')
            | misa_bit('A')
            | misa_bit('C')
            | misa_bit('S')
            | misa_bit('U');
        csrs.regs[addr::MISA as usize] = misa;
        // mhartid = 0
        csrs.regs[addr::MHARTID as usize] = 0;
        // stimecmp: default to MAX (no timer pending)
        csrs.regs[addr::STIMECMP as usize] = u64::MAX;
        // Set UXL and SXL to 2 (64-bit) in mstatus
        let uxl_sxl = (2u64 << 32) | (2u64 << 34);
        csrs.regs[addr::MSTATUS as usize] = uxl_sxl;
        csrs
    }

    /// Read a CSR with privilege checking.
    pub fn read(&self, csr_addr: u16, priv_mode: PrivilegeMode) -> Result<u64, CsrError> {
        let required = required_privilege(csr_addr);
        if (priv_mode as u8) < (required as u8) {
            return Err(CsrError::InsufficientPrivilege {
                addr: csr_addr,
                required,
                current: priv_mode,
            });
        }

        if is_unimplemented_csr(csr_addr) {
            return Err(CsrError::Unimplemented(csr_addr));
        }

        Ok(self.read_unchecked(csr_addr))
    }

    /// Write a CSR with privilege checking.
    pub fn write(&mut self, csr_addr: u16, val: u64, priv_mode: PrivilegeMode) -> Result<(), CsrError> {
        let required = required_privilege(csr_addr);
        if (priv_mode as u8) < (required as u8) {
            return Err(CsrError::InsufficientPrivilege {
                addr: csr_addr,
                required,
                current: priv_mode,
            });
        }

        // Check read-only: bits [11:10] == 0b11
        if (csr_addr >> 10) & 0x3 == 0x3 {
            return Err(CsrError::ReadOnly(csr_addr));
        }

        if is_unimplemented_csr(csr_addr) {
            return Err(CsrError::Unimplemented(csr_addr));
        }

        self.write_unchecked(csr_addr, val);
        Ok(())
    }

    /// Read without privilege checking (for internal trap delivery).
    pub fn read_unchecked(&self, csr_addr: u16) -> u64 {
        match csr_addr {
            // sstatus is a shadow of mstatus
            addr::SSTATUS => self.regs[addr::MSTATUS as usize] & mstatus::SSTATUS_MASK,
            // sie is a shadow of mie
            addr::SIE => self.regs[addr::MIE as usize] & interrupt::S_MASK,
            // sip is a shadow of mip
            addr::SIP => self.regs[addr::MIP as usize] & interrupt::S_MASK,
            // Counter CSRs
            addr::CYCLE | addr::INSTRET => self.instret,
            addr::TIME => self.mtime,
            _ => self.regs[csr_addr as usize],
        }
    }

    /// Write without privilege checking (for internal trap delivery).
    pub fn write_unchecked(&mut self, csr_addr: u16, val: u64) {
        match csr_addr {
            // misa writes are ignored (WARL: we don't allow changing extensions)
            addr::MISA => {}
            // mstatus: apply WARL mask
            addr::MSTATUS => {
                let old = self.regs[addr::MSTATUS as usize];
                let new = (old & !mstatus::WRITE_MASK) | (val & mstatus::WRITE_MASK);
                self.regs[addr::MSTATUS as usize] = new;
            }
            // sstatus writes through to mstatus
            addr::SSTATUS => {
                let old = self.regs[addr::MSTATUS as usize];
                let new = (old & !mstatus::SSTATUS_MASK) | (val & mstatus::SSTATUS_MASK);
                self.regs[addr::MSTATUS as usize] = new;
            }
            // sie writes through to mie
            addr::SIE => {
                let old = self.regs[addr::MIE as usize];
                let new = (old & !interrupt::S_MASK) | (val & interrupt::S_MASK);
                self.regs[addr::MIE as usize] = new;
            }
            // sip: only SSIP is writable by software
            addr::SIP => {
                let old = self.regs[addr::MIP as usize];
                let new = (old & !interrupt::SSIP) | (val & interrupt::SSIP);
                self.regs[addr::MIP as usize] = new;
            }
            // satp: WARL — only mode 0 (Bare) and mode 8 (Sv39) are supported
            addr::SATP => {
                let mode = (val >> 60) & 0xF;
                if mode == 0 || mode == 8 {
                    self.regs[addr::SATP as usize] = val;
                } else {
                    // Unsupported mode: write 0 (Bare) to indicate not supported
                    self.regs[addr::SATP as usize] = 0;
                }
            }
            _ => {
                self.regs[csr_addr as usize] = val;
            }
        }
    }
}

/// Determine the minimum privilege level required to access a CSR.
/// CSR address bits [9:8] encode the privilege level.
fn required_privilege(csr_addr: u16) -> PrivilegeMode {
    match (csr_addr >> 8) & 0x3 {
        0 => PrivilegeMode::User,
        1 => PrivilegeMode::Supervisor,
        // 2 is reserved/hypervisor, treat as machine
        _ => PrivilegeMode::Machine,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_misa() {
        let csrs = CsrFile::new();
        let misa = csrs.read_unchecked(addr::MISA);
        // MXL = 2 (64-bit)
        assert_eq!((misa >> 62) & 0x3, 2);
        // Extensions I, M, A, S, U
        assert_ne!(misa & misa_bit('I'), 0);
        assert_ne!(misa & misa_bit('M'), 0);
        assert_ne!(misa & misa_bit('A'), 0);
        assert_ne!(misa & misa_bit('S'), 0);
        assert_ne!(misa & misa_bit('U'), 0);
    }

    #[test]
    fn test_misa_readonly() {
        let mut csrs = CsrFile::new();
        let original = csrs.read_unchecked(addr::MISA);
        csrs.write_unchecked(addr::MISA, 0);
        assert_eq!(csrs.read_unchecked(addr::MISA), original);
    }

    #[test]
    fn test_mstatus_warl() {
        let mut csrs = CsrFile::new();
        // Write all 1s to mstatus - only WRITE_MASK bits should be set
        csrs.write_unchecked(addr::MSTATUS, u64::MAX);
        let val = csrs.read_unchecked(addr::MSTATUS);
        assert_eq!(val & !mstatus::WRITE_MASK, csrs.regs[addr::MSTATUS as usize] & !mstatus::WRITE_MASK);
    }

    #[test]
    fn test_mstatus_mpp_field() {
        let mut csrs = CsrFile::new();
        // Set MPP to Supervisor (01)
        let val = 1u64 << mstatus::MPP_SHIFT;
        csrs.write_unchecked(addr::MSTATUS, val);
        let mstatus_val = csrs.read_unchecked(addr::MSTATUS);
        let mpp = (mstatus_val & mstatus::MPP_MASK) >> mstatus::MPP_SHIFT;
        assert_eq!(mpp, 1);
    }

    #[test]
    fn test_privilege_check_m_mode() {
        let csrs = CsrFile::new();
        // M-mode can read mstatus
        assert!(csrs.read(addr::MSTATUS, PrivilegeMode::Machine).is_ok());
        // S-mode cannot read mstatus
        assert!(csrs.read(addr::MSTATUS, PrivilegeMode::Supervisor).is_err());
        // U-mode cannot read mstatus
        assert!(csrs.read(addr::MSTATUS, PrivilegeMode::User).is_err());
    }

    #[test]
    fn test_privilege_check_s_mode() {
        let csrs = CsrFile::new();
        // S-mode can read sstatus
        assert!(csrs.read(addr::SSTATUS, PrivilegeMode::Supervisor).is_ok());
        // U-mode cannot read sstatus
        assert!(csrs.read(addr::SSTATUS, PrivilegeMode::User).is_err());
    }

    #[test]
    fn test_readonly_csr_write_fails() {
        let mut csrs = CsrFile::new();
        // mhartid (0xF14) is read-only (bits [11:10] == 11)
        let result = csrs.write(addr::MHARTID, 1, PrivilegeMode::Machine);
        assert!(matches!(result, Err(CsrError::ReadOnly(0xF14))));
    }

    #[test]
    fn test_sstatus_shadow_read() {
        let mut csrs = CsrFile::new();
        // Set SIE in mstatus
        csrs.write_unchecked(addr::MSTATUS, mstatus::SIE);
        let sstatus = csrs.read_unchecked(addr::SSTATUS);
        assert_ne!(sstatus & mstatus::SIE, 0);
        // MIE should NOT be visible in sstatus
        csrs.write_unchecked(addr::MSTATUS, mstatus::MIE | mstatus::SIE);
        let sstatus = csrs.read_unchecked(addr::SSTATUS);
        assert_eq!(sstatus & mstatus::MIE, 0);
    }

    #[test]
    fn test_sstatus_shadow_write() {
        let mut csrs = CsrFile::new();
        // Set MIE in mstatus first
        csrs.write_unchecked(addr::MSTATUS, mstatus::MIE);
        // Write SIE via sstatus
        csrs.write_unchecked(addr::SSTATUS, mstatus::SIE);
        let mstatus_val = csrs.read_unchecked(addr::MSTATUS);
        // SIE should be set
        assert_ne!(mstatus_val & mstatus::SIE, 0);
        // MIE should still be set (sstatus write shouldn't affect M-mode bits)
        assert_ne!(mstatus_val & mstatus::MIE, 0);
    }

    #[test]
    fn test_sie_shadow() {
        let mut csrs = CsrFile::new();
        // Write SSIP | STIP | SEIP | MTIP to mie
        let mie_val = interrupt::SSIP | interrupt::STIP | interrupt::SEIP | interrupt::MTIP;
        csrs.write_unchecked(addr::MIE, mie_val);
        // sie should only show S-mode bits
        let sie = csrs.read_unchecked(addr::SIE);
        assert_ne!(sie & interrupt::SSIP, 0);
        assert_ne!(sie & interrupt::STIP, 0);
        assert_ne!(sie & interrupt::SEIP, 0);
        assert_eq!(sie & interrupt::MTIP, 0);
    }

    #[test]
    fn test_sie_shadow_write() {
        let mut csrs = CsrFile::new();
        // Set MTIP in mie
        csrs.write_unchecked(addr::MIE, interrupt::MTIP);
        // Write SSIP via sie
        csrs.write_unchecked(addr::SIE, interrupt::SSIP);
        let mie = csrs.read_unchecked(addr::MIE);
        // SSIP should be set
        assert_ne!(mie & interrupt::SSIP, 0);
        // MTIP should still be set
        assert_ne!(mie & interrupt::MTIP, 0);
    }

    #[test]
    fn test_sip_shadow() {
        let mut csrs = CsrFile::new();
        csrs.write_unchecked(addr::MIP, interrupt::SSIP | interrupt::MTIP);
        let sip = csrs.read_unchecked(addr::SIP);
        assert_ne!(sip & interrupt::SSIP, 0);
        assert_eq!(sip & interrupt::MTIP, 0);
    }

    #[test]
    fn test_instret_counter() {
        let mut csrs = CsrFile::new();
        csrs.instret = 42;
        assert_eq!(csrs.read_unchecked(addr::INSTRET), 42);
        assert_eq!(csrs.read_unchecked(addr::CYCLE), 42);
    }

    #[test]
    fn test_initial_mhartid() {
        let csrs = CsrFile::new();
        assert_eq!(csrs.read_unchecked(addr::MHARTID), 0);
    }

    #[test]
    fn test_medeleg_mideleg() {
        let mut csrs = CsrFile::new();
        csrs.write_unchecked(addr::MEDELEG, 0xFFFF);
        assert_eq!(csrs.read_unchecked(addr::MEDELEG), 0xFFFF);
        csrs.write_unchecked(addr::MIDELEG, 0x222);
        assert_eq!(csrs.read_unchecked(addr::MIDELEG), 0x222);
    }

    #[test]
    fn test_privilege_mode_ordering() {
        assert!(PrivilegeMode::User < PrivilegeMode::Supervisor);
        assert!(PrivilegeMode::Supervisor < PrivilegeMode::Machine);
    }

    #[test]
    fn test_initial_uxl_sxl() {
        let csrs = CsrFile::new();
        let mstatus_val = csrs.read_unchecked(addr::MSTATUS);
        // UXL = 2 (bits [33:32])
        assert_eq!((mstatus_val >> 32) & 0x3, 2);
        // SXL = 2 (bits [35:34])
        assert_eq!((mstatus_val >> 34) & 0x3, 2);
    }
}
