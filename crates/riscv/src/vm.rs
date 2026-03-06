use crate::cpu::CpuState;
use crate::csr::{addr, mstatus, interrupt, PrivilegeMode};
use crate::decode::{self, DecodeError};
use crate::isa::Instruction;
use crate::memory::Memory;
use crate::mmio::MmioController;
use crate::mmu::{self, AccessType};
use crate::trap::{self, ExceptionCause, InterruptCause, TrapCause};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VmError {
    #[error("decode error: {0}")]
    Decode(#[from] DecodeError),
    #[error("VM is halted")]
    Halted,
    #[error("EBREAK at pc=0x{0:x}")]
    Ebreak(u64),
}

/// Simple direct-mapped TLB entry: maps a virtual page to a physical page.
#[derive(Clone, Copy)]
struct TlbEntry {
    vpn: u64,       // virtual page number (vaddr >> 12)
    ppn: u64,       // physical page number (paddr >> 12)
    satp: u64,      // satp value when this entry was created
    readable: bool,
    writable: bool,
    executable: bool,
}

const TLB_SIZE: usize = 256;

/// The local (plaintext) RV64IM virtual machine.
pub struct Vm {
    pub cpu: CpuState,
    pub memory: Memory,
    pub halted: bool,
    pub exit_code: Option<u64>,
    pub mmio: MmioController,
    /// When true, ECALL raises traps via the trap framework instead of legacy syscalls.
    pub privileged: bool,
    /// LR/SC reservation: (address, size in bytes)
    pub reservation: Option<(u64, u8)>,
    /// Last fetched instruction length (2 or 4 bytes).
    pub insn_len: u8,
    /// Last successfully fetched instruction (set by step_inner, None for trap deliveries).
    pub last_fetched: Option<(Instruction, u8)>,
    /// Simple direct-mapped TLB cache.
    tlb: [Option<TlbEntry>; TLB_SIZE],
}

impl Vm {
    pub fn new(cpu: CpuState, memory: Memory) -> Self {
        Vm {
            cpu,
            memory,
            halted: false,
            exit_code: None,
            mmio: MmioController::new(),
            privileged: false,
            reservation: None,
            insn_len: 4,
            last_fetched: None,
            tlb: [None; TLB_SIZE],
        }
    }

    pub fn new_with_mmio(cpu: CpuState, memory: Memory, mmio: MmioController) -> Self {
        Vm {
            cpu,
            memory,
            halted: false,
            exit_code: None,
            mmio,
            privileged: false,
            reservation: None,
            insn_len: 4,
            last_fetched: None,
            tlb: [None; TLB_SIZE],
        }
    }

    pub fn new_privileged(cpu: CpuState, memory: Memory) -> Self {
        Vm {
            cpu,
            memory,
            halted: false,
            exit_code: None,
            mmio: MmioController::new(),
            privileged: true,
            reservation: None,
            insn_len: 4,
            last_fetched: None,
            tlb: [None; TLB_SIZE],
        }
    }

    /// How often to tick devices (every N instructions).
    const TICK_INTERVAL: u64 = 64;

    /// mtime ticks per emulated instruction.  The DTB declares a 10 MHz
    /// timer.  At ~5 MIPS this gives a 20 MHz effective rate, which is
    /// close enough and — critically — means the timer fires every
    /// ~10 000 instructions (at HZ=250), leaving plenty of headroom for
    /// the interrupt handler.
    const MTIME_INCREMENT_PER_INSN: u64 = 4;

    /// Fetch, decode, and execute one instruction.
    pub fn step(&mut self) -> Result<(), VmError> {
        if self.halted {
            return Err(VmError::Halted);
        }

        // Tick devices and check interrupts (privileged mode only)
        if self.privileged {
            self.tick_devices();
            self.check_pending_interrupts();
        }

        self.step_inner()
    }

    /// Inner step: fetch, decode, execute (no device tick).
    #[inline(always)]
    fn step_inner(&mut self) -> Result<(), VmError> {
        self.last_fetched = None;
        let pc = self.cpu.pc;

        // Translate PC if paging is active
        let fetch_pc = if self.translation_active() {
            match self.translate_addr(pc, AccessType::Execute) {
                Ok(pa) => pa,
                Err(()) => return Ok(()), // page fault delivered
            }
        } else {
            pc
        };

        // Variable-length fetch
        let fetched = match decode::fetch_decode(&self.memory, fetch_pc) {
            Ok(f) => f,
            Err(e) => {
                if self.privileged {
                    // Deliver IllegalInstruction exception
                    let word = self.memory.load_word(fetch_pc);
                    trap::deliver_trap(
                        &mut self.cpu,
                        TrapCause::Exception(ExceptionCause::IllegalInstruction),
                        word as u64,
                        pc,
                    );
                    return Ok(());
                } else {
                    return Err(e.into());
                }
            }
        };

        self.insn_len = fetched.len;
        self.last_fetched = Some((fetched.instruction, fetched.len));
        self.execute(fetched.instruction)?;

        // Increment instret counter (privileged mode)
        if self.privileged {
            self.cpu.csrs.instret += 1;
        }

        Ok(())
    }

    /// Run until halt or step limit.
    pub fn run(&mut self, max_steps: u64) -> Result<u64, VmError> {
        let mut steps = 0u64;
        if self.privileged {
            // Optimized loop: tick devices every TICK_INTERVAL instructions
            while !self.halted && steps < max_steps {
                self.tick_devices();
                self.check_pending_interrupts();
                let batch_end = (steps + Self::TICK_INTERVAL).min(max_steps);
                while !self.halted && steps < batch_end {
                    self.step_inner()?;
                    steps += 1;
                }
            }
        } else {
            while !self.halted && steps < max_steps {
                self.step_inner()?;
                steps += 1;
            }
        }
        Ok(steps)
    }

    /// Tick CLINT, check PLIC, process VirtIO, and update mip accordingly.
    fn tick_devices(&mut self) {
        // Advance CLINT timer by MTIME_INCREMENT_PER_INSN * TICK_INTERVAL ticks
        // per batch.  At 5 MIPS this gives an effective 20 MHz mtime rate
        // (the DTB declares 10 MHz).  Using a lower per-instruction rate than
        // the original 25 avoids timer-interrupt storms: with 25, the timer
        // fires every ~1600 instructions which is fewer than a typical
        // tick_periodic handler, starving forward progress and eventually
        // causing the clockevent to shut down.
        self.mmio.clint.mtime = self.mmio.clint.mtime.wrapping_add(Self::MTIME_INCREMENT_PER_INSN * Self::TICK_INTERVAL);
        let timer_pending = self.mmio.clint.mtime >= self.mmio.clint.mtimecmp;
        let mut mip = self.cpu.csrs.read_unchecked(addr::MIP);
        if timer_pending {
            mip |= interrupt::MTIP;
        } else {
            mip &= !interrupt::MTIP;
        }
        // CLINT software interrupt
        if self.mmio.clint.msip_pending() {
            mip |= interrupt::MSIP;
        } else {
            mip &= !interrupt::MSIP;
        }

        // Check PLIC: only scan sources if any are pending (fast path)
        let has_any_pending = self.mmio.plic.pending[0] != 0 || self.mmio.plic.pending[1] != 0;
        if has_any_pending {
            if self.mmio.plic.has_pending(0) {
                mip |= interrupt::MEIP;
            } else {
                mip &= !interrupt::MEIP;
            }
            if self.mmio.plic.has_pending(1) {
                mip |= interrupt::SEIP;
            } else {
                mip &= !interrupt::SEIP;
            }
        } else {
            mip &= !(interrupt::MEIP | interrupt::SEIP);
        }

        // Update TIME CSR to mirror CLINT mtime
        self.cpu.csrs.mtime = self.mmio.clint.mtime;

        // sstc extension: set STIP when mtime >= stimecmp
        let stimecmp = self.cpu.csrs.read_unchecked(addr::STIMECMP);
        if self.mmio.clint.mtime >= stimecmp {
            mip |= interrupt::STIP;
        } else {
            mip &= !interrupt::STIP;
        }

        self.cpu.csrs.write_unchecked(addr::MIP, mip);

        // Process pending VirtIO requests
        for dev in &mut self.mmio.virtio_devices {
            if dev.needs_processing() {
                dev.process_queue(&mut self.memory);
            }
        }
    }

    /// Check for pending interrupts and deliver if enabled.
    fn check_pending_interrupts(&mut self) {
        let mip = self.cpu.csrs.read_unchecked(addr::MIP);
        let mie = self.cpu.csrs.read_unchecked(addr::MIE);
        let pending = mip & mie;
        if pending == 0 {
            return;
        }

        let ms = self.cpu.csrs.read_unchecked(addr::MSTATUS);
        let mideleg = self.cpu.csrs.read_unchecked(addr::MIDELEG);
        let priv_mode = self.cpu.priv_mode;

        // Priority order: MEI > MSI > MTI > SEI > SSI > STI
        let priorities = [
            (interrupt::MEIP, InterruptCause::MachineExternal),
            (interrupt::MSIP, InterruptCause::MachineSoftware),
            (interrupt::MTIP, InterruptCause::MachineTimer),
            (interrupt::SEIP, InterruptCause::SupervisorExternal),
            (interrupt::SSIP, InterruptCause::SupervisorSoftware),
            (interrupt::STIP, InterruptCause::SupervisorTimer),
        ];

        for (bit, cause) in priorities {
            if pending & bit == 0 {
                continue;
            }

            let delegated = (mideleg >> (cause as u64)) & 1 != 0;

            let should_take = if delegated {
                // Delegated to S-mode
                match priv_mode {
                    PrivilegeMode::User => true,
                    PrivilegeMode::Supervisor => (ms & mstatus::SIE) != 0,
                    PrivilegeMode::Machine => false,
                }
            } else {
                // Goes to M-mode
                match priv_mode {
                    PrivilegeMode::User | PrivilegeMode::Supervisor => true,
                    PrivilegeMode::Machine => (ms & mstatus::MIE) != 0,
                }
            };

            if should_take {
                let pc = self.cpu.pc;
                trap::deliver_trap(
                    &mut self.cpu,
                    TrapCause::Interrupt(cause),
                    0,
                    pc,
                );
                return;
            }
        }
    }

    /// Returns true if address translation is active for the current privilege.
    fn translation_active(&self) -> bool {
        if !self.privileged {
            return false;
        }
        let satp = self.cpu.csrs.read_unchecked(addr::SATP);
        let mode = (satp >> 60) & 0xF;
        if mode == 0 {
            return false;
        }
        // M-mode uses physical addresses (unless MPRV is set, handled separately)
        self.cpu.priv_mode != PrivilegeMode::Machine
    }

    /// Look up a virtual address in the TLB.
    fn tlb_lookup(&self, vaddr: u64, access_type: AccessType, satp: u64) -> Option<u64> {
        let vpn = vaddr >> 12;
        let idx = (vpn as usize) & (TLB_SIZE - 1);
        if let Some(entry) = &self.tlb[idx] {
            if entry.vpn == vpn && entry.satp == satp {
                let ok = match access_type {
                    AccessType::Execute => entry.executable,
                    AccessType::Read => entry.readable,
                    AccessType::Write => entry.writable,
                };
                if ok {
                    return Some((entry.ppn << 12) | (vaddr & 0xFFF));
                }
            }
        }
        None
    }

    /// Insert a TLB entry after a successful page table walk.
    fn tlb_insert(&mut self, vaddr: u64, paddr: u64, satp: u64, r: bool, w: bool, x: bool) {
        let vpn = vaddr >> 12;
        let idx = (vpn as usize) & (TLB_SIZE - 1);
        self.tlb[idx] = Some(TlbEntry {
            vpn,
            ppn: paddr >> 12,
            satp,
            readable: r,
            writable: w,
            executable: x,
        });
    }

    /// Flush the entire TLB.
    fn tlb_flush(&mut self) {
        self.tlb = [None; TLB_SIZE];
    }

    /// Translate a virtual address to physical. Returns Ok(physical) or delivers
    /// a page fault trap and returns Err(()).
    fn translate_addr(&mut self, vaddr: u64, access_type: AccessType) -> Result<u64, ()> {
        let satp = self.cpu.csrs.read_unchecked(addr::SATP);

        // Fast path: TLB hit
        if let Some(pa) = self.tlb_lookup(vaddr, access_type, satp) {
            return Ok(pa);
        }

        let ms = self.cpu.csrs.read_unchecked(addr::MSTATUS);
        let sum = (ms & mstatus::SUM) != 0;
        let mxr = (ms & mstatus::MXR) != 0;
        let priv_level = self.cpu.priv_mode as u8;

        match mmu::translate(&self.memory, satp, vaddr, access_type, priv_level, sum, mxr) {
            Ok((pa, pte_update, pte_flags)) => {
                if let Some((pte_addr, new_pte)) = pte_update {
                    self.memory.store_double(pte_addr, new_pte);
                }
                // Cache in TLB using PTE permission flags directly
                let page_pa = pa & !0xFFF;
                let page_va = vaddr & !0xFFF;
                let r = (pte_flags & 0x2) != 0 || (mxr && (pte_flags & 0x8) != 0);
                let w = (pte_flags & 0x4) != 0;
                let x = (pte_flags & 0x8) != 0;
                self.tlb_insert(page_va, page_pa, satp, r, w, x);
                Ok(pa)
            }
            Err(pf) => {
                let cause = match pf.access_type {
                    AccessType::Execute => ExceptionCause::InstructionPageFault,
                    AccessType::Read => ExceptionCause::LoadPageFault,
                    AccessType::Write => ExceptionCause::StorePageFault,
                };
                let pc = self.cpu.pc;
                trap::deliver_trap(
                    &mut self.cpu,
                    TrapCause::Exception(cause),
                    vaddr,
                    pc,
                );
                Err(())
            }
        }
    }

    /// Translate a virtual address for a load, returning the physical address.
    /// Returns Err(()) if a page fault was delivered.
    fn translate_load(&mut self, vaddr: u64) -> Result<u64, ()> {
        if self.translation_active() {
            self.translate_addr(vaddr, AccessType::Read)
        } else {
            Ok(vaddr)
        }
    }

    /// Translate a virtual address for a store, returning the physical address.
    /// Returns Err(()) if a page fault was delivered.
    fn translate_store(&mut self, vaddr: u64) -> Result<u64, ()> {
        if self.translation_active() {
            self.translate_addr(vaddr, AccessType::Write)
        } else {
            Ok(vaddr)
        }
    }

    /// RAM starts at 0x8000_0000; addresses at or above this skip MMIO dispatch.
    const RAM_BASE: u64 = 0x8000_0000;

    #[inline(always)]
    pub fn mmio_load_byte(&mut self, addr: u64) -> u8 {
        if addr >= Self::RAM_BASE {
            self.memory.load_byte(addr)
        } else {
            self.mmio.read_byte(addr).unwrap_or_else(|| self.memory.load_byte(addr))
        }
    }

    #[inline(always)]
    fn mmio_store_byte(&mut self, addr: u64, val: u8) {
        if addr >= Self::RAM_BASE {
            self.memory.store_byte(addr, val);
        } else if !self.mmio.write_byte(addr, val) {
            self.memory.store_byte(addr, val);
        }
    }

    #[inline(always)]
    pub fn mmio_load_half(&mut self, addr: u64) -> u16 {
        if addr >= Self::RAM_BASE {
            self.memory.load_half(addr)
        } else {
            let lo = self.mmio_load_byte(addr) as u16;
            let hi = self.mmio_load_byte(addr.wrapping_add(1)) as u16;
            lo | (hi << 8)
        }
    }

    #[inline(always)]
    fn mmio_store_half(&mut self, addr: u64, val: u16) {
        if addr >= Self::RAM_BASE {
            self.memory.store_half(addr, val);
        } else {
            self.mmio_store_byte(addr, val as u8);
            self.mmio_store_byte(addr.wrapping_add(1), (val >> 8) as u8);
        }
    }

    #[inline(always)]
    pub fn mmio_load_word(&mut self, addr: u64) -> u32 {
        if addr >= Self::RAM_BASE {
            self.memory.load_word(addr)
        } else {
            let lo = self.mmio_load_half(addr) as u32;
            let hi = self.mmio_load_half(addr.wrapping_add(2)) as u32;
            lo | (hi << 16)
        }
    }

    #[inline(always)]
    fn mmio_store_word(&mut self, addr: u64, val: u32) {
        if addr >= Self::RAM_BASE {
            self.memory.store_word(addr, val);
        } else {
            self.mmio_store_half(addr, val as u16);
            self.mmio_store_half(addr.wrapping_add(2), (val >> 16) as u16);
        }
    }

    #[inline(always)]
    pub fn mmio_load_double(&mut self, addr: u64) -> u64 {
        if addr >= Self::RAM_BASE {
            self.memory.load_double(addr)
        } else {
            let lo = self.mmio_load_word(addr) as u64;
            let hi = self.mmio_load_word(addr.wrapping_add(4)) as u64;
            lo | (hi << 32)
        }
    }

    #[inline(always)]
    fn mmio_store_double(&mut self, addr: u64, val: u64) {
        if addr >= Self::RAM_BASE {
            self.memory.store_double(addr, val);
        } else {
            self.mmio_store_word(addr, val as u32);
            self.mmio_store_word(addr.wrapping_add(4), (val >> 32) as u32);
        }
    }

    /// Clear reservation if a store overlaps the reserved address.
    fn check_reservation_on_store(&mut self, store_addr: u64, store_size: u8) {
        if let Some((res_addr, res_size)) = self.reservation {
            let store_end = store_addr.wrapping_add(store_size as u64);
            let res_end = res_addr.wrapping_add(res_size as u64);
            if store_addr < res_end && res_addr < store_end {
                self.reservation = None;
            }
        }
    }

    fn execute(&mut self, insn: Instruction) -> Result<(), VmError> {
        let pc = self.cpu.pc;
        let step = self.insn_len as u64;

        match insn {
            // === U-type ===
            Instruction::LUI { rd, imm } => {
                self.cpu.write_reg(rd, imm as i64 as u64);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AUIPC { rd, imm } => {
                self.cpu.write_reg(rd, pc.wrapping_add(imm as i64 as u64));
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === J-type ===
            Instruction::JAL { rd, imm } => {
                self.cpu.write_reg(rd, pc.wrapping_add(step));
                self.cpu.pc = pc.wrapping_add(imm as i64 as u64);
            }

            // === I-type (JALR) ===
            Instruction::JALR { rd, rs1, imm } => {
                let target = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64) & !1;
                self.cpu.write_reg(rd, pc.wrapping_add(step));
                self.cpu.pc = target;
            }

            // === B-type (branches) ===
            Instruction::BEQ { rs1, rs2, imm } => {
                if self.cpu.read_reg(rs1) == self.cpu.read_reg(rs2) {
                    self.cpu.pc = pc.wrapping_add(imm as i64 as u64);
                } else {
                    self.cpu.pc = pc.wrapping_add(step);
                }
            }
            Instruction::BNE { rs1, rs2, imm } => {
                if self.cpu.read_reg(rs1) != self.cpu.read_reg(rs2) {
                    self.cpu.pc = pc.wrapping_add(imm as i64 as u64);
                } else {
                    self.cpu.pc = pc.wrapping_add(step);
                }
            }
            Instruction::BLT { rs1, rs2, imm } => {
                if (self.cpu.read_reg(rs1) as i64) < (self.cpu.read_reg(rs2) as i64) {
                    self.cpu.pc = pc.wrapping_add(imm as i64 as u64);
                } else {
                    self.cpu.pc = pc.wrapping_add(step);
                }
            }
            Instruction::BGE { rs1, rs2, imm } => {
                if (self.cpu.read_reg(rs1) as i64) >= (self.cpu.read_reg(rs2) as i64) {
                    self.cpu.pc = pc.wrapping_add(imm as i64 as u64);
                } else {
                    self.cpu.pc = pc.wrapping_add(step);
                }
            }
            Instruction::BLTU { rs1, rs2, imm } => {
                if self.cpu.read_reg(rs1) < self.cpu.read_reg(rs2) {
                    self.cpu.pc = pc.wrapping_add(imm as i64 as u64);
                } else {
                    self.cpu.pc = pc.wrapping_add(step);
                }
            }
            Instruction::BGEU { rs1, rs2, imm } => {
                if self.cpu.read_reg(rs1) >= self.cpu.read_reg(rs2) {
                    self.cpu.pc = pc.wrapping_add(imm as i64 as u64);
                } else {
                    self.cpu.pc = pc.wrapping_add(step);
                }
            }

            // === Load instructions ===
            Instruction::LB { rd, rs1, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_byte(pa) as i8 as i64 as u64;
                self.cpu.write_reg(rd, val);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::LH { rd, rs1, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_half(pa) as i16 as i64 as u64;
                self.cpu.write_reg(rd, val);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::LW { rd, rs1, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_word(pa) as i32 as i64 as u64;
                self.cpu.write_reg(rd, val);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::LD { rd, rs1, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_double(pa);
                self.cpu.write_reg(rd, val);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::LBU { rd, rs1, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_byte(pa) as u64;
                self.cpu.write_reg(rd, val);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::LHU { rd, rs1, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_half(pa) as u64;
                self.cpu.write_reg(rd, val);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::LWU { rd, rs1, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_word(pa) as u64;
                self.cpu.write_reg(rd, val);
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === Store instructions ===
            Instruction::SB { rs1, rs2, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                self.mmio_store_byte(pa, self.cpu.read_reg(rs2) as u8);
                self.check_reservation_on_store(pa, 1);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SH { rs1, rs2, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                self.mmio_store_half(pa, self.cpu.read_reg(rs2) as u16);
                self.check_reservation_on_store(pa, 2);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SW { rs1, rs2, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                self.mmio_store_word(pa, self.cpu.read_reg(rs2) as u32);
                self.check_reservation_on_store(pa, 4);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SD { rs1, rs2, imm } => {
                let vaddr = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                self.mmio_store_double(pa, self.cpu.read_reg(rs2));
                self.check_reservation_on_store(pa, 8);
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === OP-IMM (immediate ALU) ===
            Instruction::ADDI { rd, rs1, imm } => {
                let result = self.cpu.read_reg(rs1).wrapping_add(imm as i64 as u64);
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SLTI { rd, rs1, imm } => {
                let result =
                    if (self.cpu.read_reg(rs1) as i64) < (imm as i64) { 1 } else { 0 };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SLTIU { rd, rs1, imm } => {
                let result =
                    if self.cpu.read_reg(rs1) < (imm as i64 as u64) { 1 } else { 0 };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::XORI { rd, rs1, imm } => {
                let result = self.cpu.read_reg(rs1) ^ (imm as i64 as u64);
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::ORI { rd, rs1, imm } => {
                let result = self.cpu.read_reg(rs1) | (imm as i64 as u64);
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::ANDI { rd, rs1, imm } => {
                let result = self.cpu.read_reg(rs1) & (imm as i64 as u64);
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SLLI { rd, rs1, imm } => {
                let shamt = (imm & 0x3F) as u32;
                let result = self.cpu.read_reg(rs1) << shamt;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SRLI { rd, rs1, imm } => {
                let shamt = (imm & 0x3F) as u32;
                let result = self.cpu.read_reg(rs1) >> shamt;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SRAI { rd, rs1, imm } => {
                let shamt = (imm & 0x3F) as u32;
                let result = ((self.cpu.read_reg(rs1) as i64) >> shamt) as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === OP (register ALU) ===
            Instruction::ADD { rd, rs1, rs2 } => {
                let result = self.cpu.read_reg(rs1).wrapping_add(self.cpu.read_reg(rs2));
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SUB { rd, rs1, rs2 } => {
                let result = self.cpu.read_reg(rs1).wrapping_sub(self.cpu.read_reg(rs2));
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SLL { rd, rs1, rs2 } => {
                let shamt = self.cpu.read_reg(rs2) & 0x3F;
                let result = self.cpu.read_reg(rs1) << shamt;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SLT { rd, rs1, rs2 } => {
                let result = if (self.cpu.read_reg(rs1) as i64) < (self.cpu.read_reg(rs2) as i64) {
                    1
                } else {
                    0
                };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SLTU { rd, rs1, rs2 } => {
                let result = if self.cpu.read_reg(rs1) < self.cpu.read_reg(rs2) {
                    1
                } else {
                    0
                };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::XOR { rd, rs1, rs2 } => {
                let result = self.cpu.read_reg(rs1) ^ self.cpu.read_reg(rs2);
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SRL { rd, rs1, rs2 } => {
                let shamt = self.cpu.read_reg(rs2) & 0x3F;
                let result = self.cpu.read_reg(rs1) >> shamt;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SRA { rd, rs1, rs2 } => {
                let shamt = self.cpu.read_reg(rs2) & 0x3F;
                let result = ((self.cpu.read_reg(rs1) as i64) >> shamt) as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::OR { rd, rs1, rs2 } => {
                let result = self.cpu.read_reg(rs1) | self.cpu.read_reg(rs2);
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AND { rd, rs1, rs2 } => {
                let result = self.cpu.read_reg(rs1) & self.cpu.read_reg(rs2);
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === M Extension ===
            Instruction::MUL { rd, rs1, rs2 } => {
                let result = self.cpu.read_reg(rs1).wrapping_mul(self.cpu.read_reg(rs2));
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::MULH { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as i64 as i128;
                let b = self.cpu.read_reg(rs2) as i64 as i128;
                let result = ((a * b) >> 64) as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::MULHSU { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as i64 as i128;
                let b = self.cpu.read_reg(rs2) as u128;
                let result = (((a as i128) * (b as i128)) >> 64) as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::MULHU { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as u128;
                let b = self.cpu.read_reg(rs2) as u128;
                let result = ((a * b) >> 64) as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::DIV { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as i64;
                let b = self.cpu.read_reg(rs2) as i64;
                let result = if b == 0 {
                    u64::MAX
                } else if a == i64::MIN && b == -1 {
                    a as u64
                } else {
                    (a / b) as u64
                };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::DIVU { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1);
                let b = self.cpu.read_reg(rs2);
                let result = if b == 0 { u64::MAX } else { a / b };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::REM { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as i64;
                let b = self.cpu.read_reg(rs2) as i64;
                let result = if b == 0 {
                    a as u64
                } else if a == i64::MIN && b == -1 {
                    0u64
                } else {
                    (a % b) as u64
                };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::REMU { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1);
                let b = self.cpu.read_reg(rs2);
                let result = if b == 0 { a } else { a % b };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === OP-IMM-32 (W variants) ===
            Instruction::ADDIW { rd, rs1, imm } => {
                let result = (self.cpu.read_reg(rs1) as i32).wrapping_add(imm) as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SLLIW { rd, rs1, imm } => {
                let shamt = (imm & 0x1F) as u32;
                let result = ((self.cpu.read_reg(rs1) as u32) << shamt) as i32 as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SRLIW { rd, rs1, imm } => {
                let shamt = (imm & 0x1F) as u32;
                let result = ((self.cpu.read_reg(rs1) as u32) >> shamt) as i32 as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SRAIW { rd, rs1, imm } => {
                let shamt = (imm & 0x1F) as u32;
                let result = ((self.cpu.read_reg(rs1) as i32) >> shamt) as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === OP-32 (W register variants) ===
            Instruction::ADDW { rd, rs1, rs2 } => {
                let result = (self.cpu.read_reg(rs1) as i32)
                    .wrapping_add(self.cpu.read_reg(rs2) as i32) as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SUBW { rd, rs1, rs2 } => {
                let result = (self.cpu.read_reg(rs1) as i32)
                    .wrapping_sub(self.cpu.read_reg(rs2) as i32) as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SLLW { rd, rs1, rs2 } => {
                let shamt = self.cpu.read_reg(rs2) & 0x1F;
                let result =
                    ((self.cpu.read_reg(rs1) as u32) << shamt) as i32 as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SRLW { rd, rs1, rs2 } => {
                let shamt = self.cpu.read_reg(rs2) & 0x1F;
                let result =
                    ((self.cpu.read_reg(rs1) as u32) >> shamt) as i32 as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SRAW { rd, rs1, rs2 } => {
                let shamt = self.cpu.read_reg(rs2) & 0x1F;
                let result =
                    ((self.cpu.read_reg(rs1) as i32) >> shamt) as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === M-extension W variants ===
            Instruction::MULW { rd, rs1, rs2 } => {
                let result = (self.cpu.read_reg(rs1) as i32)
                    .wrapping_mul(self.cpu.read_reg(rs2) as i32)
                    as i64 as u64;
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::DIVW { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as i32;
                let b = self.cpu.read_reg(rs2) as i32;
                let result = if b == 0 {
                    u64::MAX
                } else if a == i32::MIN && b == -1 {
                    a as i64 as u64
                } else {
                    (a / b) as i64 as u64
                };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::DIVUW { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as u32;
                let b = self.cpu.read_reg(rs2) as u32;
                let result = if b == 0 {
                    u64::MAX
                } else {
                    (a / b) as i32 as i64 as u64
                };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::REMW { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as i32;
                let b = self.cpu.read_reg(rs2) as i32;
                let result = if b == 0 {
                    a as i64 as u64
                } else if a == i32::MIN && b == -1 {
                    0u64
                } else {
                    (a % b) as i64 as u64
                };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::REMUW { rd, rs1, rs2 } => {
                let a = self.cpu.read_reg(rs1) as u32;
                let b = self.cpu.read_reg(rs2) as u32;
                let result = if b == 0 {
                    (a as i32) as i64 as u64
                } else {
                    (a % b) as i32 as i64 as u64
                };
                self.cpu.write_reg(rd, result);
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === System ===
            Instruction::FENCE { .. } | Instruction::FENCE_I => {
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::ECALL => {
                if self.privileged {
                    let cause = match self.cpu.priv_mode {
                        PrivilegeMode::User => ExceptionCause::EnvironmentCallFromU,
                        PrivilegeMode::Supervisor => ExceptionCause::EnvironmentCallFromS,
                        PrivilegeMode::Machine => ExceptionCause::EnvironmentCallFromM,
                    };
                    trap::deliver_trap(
                        &mut self.cpu,
                        TrapCause::Exception(cause),
                        0,
                        pc,
                    );
                } else {
                    // Legacy syscall dispatch
                    let syscall = self.cpu.read_reg(17);
                    match syscall {
                        0 => {
                            self.halted = true;
                            self.cpu.pc = pc.wrapping_add(step);
                        }
                        93 => {
                            self.exit_code = Some(self.cpu.read_reg(10));
                            self.halted = true;
                            self.cpu.pc = pc.wrapping_add(step);
                        }
                        64 => {
                            let _fd = self.cpu.read_reg(10);
                            let buf = self.cpu.read_reg(11);
                            let count = self.cpu.read_reg(12);
                            for i in 0..count {
                                let b = self.memory.load_byte(buf.wrapping_add(i));
                                self.mmio.uart.output.push(b);
                            }
                            self.cpu.write_reg(10, count);
                            self.cpu.pc = pc.wrapping_add(step);
                        }
                        _ => {
                            self.cpu.write_reg(10, (-38i64) as u64);
                            self.cpu.pc = pc.wrapping_add(step);
                        }
                    }
                }
            }
            Instruction::EBREAK => {
                if self.privileged {
                    trap::deliver_trap(
                        &mut self.cpu,
                        TrapCause::Exception(ExceptionCause::Breakpoint),
                        0,
                        pc,
                    );
                } else {
                    self.halted = true;
                    self.cpu.pc = pc.wrapping_add(step);
                    return Err(VmError::Ebreak(pc));
                }
            }

            // === CSR Instructions ===
            Instruction::CSRRW { rd, rs1, csr } => {
                match self.cpu.csrs.read(csr, self.cpu.priv_mode) {
                    Ok(old) => {
                        let src = self.cpu.read_reg(rs1);
                        if let Err(_) = self.cpu.csrs.write(csr, src, self.cpu.priv_mode) {
                            if self.privileged {
                                trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                                return Ok(());
                            }
                        }
                        if rd != 0 {
                            self.cpu.write_reg(rd, old);
                        }
                        self.cpu.pc = pc.wrapping_add(step);
                    }
                    Err(_) => {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                        }
                        return Ok(());
                    }
                }
            }
            Instruction::CSRRS { rd, rs1, csr } => {
                match self.cpu.csrs.read(csr, self.cpu.priv_mode) {
                    Ok(old) => {
                        if rs1 != 0 {
                            let mask = self.cpu.read_reg(rs1);
                            if let Err(_) = self.cpu.csrs.write(csr, old | mask, self.cpu.priv_mode) {
                                if self.privileged {
                                    trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                                    return Ok(());
                                }
                            }
                        }
                        self.cpu.write_reg(rd, old);
                        self.cpu.pc = pc.wrapping_add(step);
                    }
                    Err(_) => {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                        }
                        return Ok(());
                    }
                }
            }
            Instruction::CSRRC { rd, rs1, csr } => {
                match self.cpu.csrs.read(csr, self.cpu.priv_mode) {
                    Ok(old) => {
                        if rs1 != 0 {
                            let mask = self.cpu.read_reg(rs1);
                            if let Err(_) = self.cpu.csrs.write(csr, old & !mask, self.cpu.priv_mode) {
                                if self.privileged {
                                    trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                                    return Ok(());
                                }
                            }
                        }
                        self.cpu.write_reg(rd, old);
                        self.cpu.pc = pc.wrapping_add(step);
                    }
                    Err(_) => {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                        }
                        return Ok(());
                    }
                }
            }
            Instruction::CSRRWI { rd, uimm, csr } => {
                match self.cpu.csrs.read(csr, self.cpu.priv_mode) {
                    Ok(old) => {
                        let src = uimm as u64;
                        if let Err(_) = self.cpu.csrs.write(csr, src, self.cpu.priv_mode) {
                            if self.privileged {
                                trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                                return Ok(());
                            }
                        }
                        if rd != 0 {
                            self.cpu.write_reg(rd, old);
                        }
                        self.cpu.pc = pc.wrapping_add(step);
                    }
                    Err(_) => {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                        }
                        return Ok(());
                    }
                }
            }
            Instruction::CSRRSI { rd, uimm, csr } => {
                match self.cpu.csrs.read(csr, self.cpu.priv_mode) {
                    Ok(old) => {
                        if uimm != 0 {
                            let mask = uimm as u64;
                            if let Err(_) = self.cpu.csrs.write(csr, old | mask, self.cpu.priv_mode) {
                                if self.privileged {
                                    trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                                    return Ok(());
                                }
                            }
                        }
                        self.cpu.write_reg(rd, old);
                        self.cpu.pc = pc.wrapping_add(step);
                    }
                    Err(_) => {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                        }
                        return Ok(());
                    }
                }
            }
            Instruction::CSRRCI { rd, uimm, csr } => {
                match self.cpu.csrs.read(csr, self.cpu.priv_mode) {
                    Ok(old) => {
                        if uimm != 0 {
                            let mask = uimm as u64;
                            if let Err(_) = self.cpu.csrs.write(csr, old & !mask, self.cpu.priv_mode) {
                                if self.privileged {
                                    trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                                    return Ok(());
                                }
                            }
                        }
                        self.cpu.write_reg(rd, old);
                        self.cpu.pc = pc.wrapping_add(step);
                    }
                    Err(_) => {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                        }
                        return Ok(());
                    }
                }
            }

            // === Privileged Instructions ===
            Instruction::MRET => {
                let ms = self.cpu.csrs.read_unchecked(addr::MSTATUS);
                let mpp = (ms & mstatus::MPP_MASK) >> mstatus::MPP_SHIFT;
                let mpie = (ms & mstatus::MPIE) != 0;

                // Restore MIE from MPIE
                let mut new_ms = ms;
                if mpie { new_ms |= mstatus::MIE; } else { new_ms &= !mstatus::MIE; }
                new_ms |= mstatus::MPIE; // MPIE = 1
                new_ms &= !mstatus::MPP_MASK; // MPP = U (0)
                // If MPP != M, clear MPRV
                if mpp != PrivilegeMode::Machine as u64 {
                    new_ms &= !mstatus::MPRV;
                }
                self.cpu.csrs.write_unchecked(addr::MSTATUS, new_ms);

                self.cpu.priv_mode = PrivilegeMode::from_bits(mpp);
                self.cpu.pc = self.cpu.csrs.read_unchecked(addr::MEPC);
                self.tlb_flush();
            }
            Instruction::SRET => {
                // Check TSR for S-mode
                if self.cpu.priv_mode == PrivilegeMode::Supervisor {
                    let ms = self.cpu.csrs.read_unchecked(addr::MSTATUS);
                    if ms & mstatus::TSR != 0 {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                            return Ok(());
                        }
                    }
                }

                let ms = self.cpu.csrs.read_unchecked(addr::MSTATUS);
                let spp = if ms & mstatus::SPP != 0 { 1u64 } else { 0u64 };
                let spie = (ms & mstatus::SPIE) != 0;

                let mut new_ms = ms;
                if spie { new_ms |= mstatus::SIE; } else { new_ms &= !mstatus::SIE; }
                new_ms |= mstatus::SPIE; // SPIE = 1
                new_ms &= !mstatus::SPP; // SPP = U (0)
                // If SPP != M, clear MPRV
                new_ms &= !mstatus::MPRV;
                self.cpu.csrs.write_unchecked(addr::MSTATUS, new_ms);

                self.cpu.priv_mode = PrivilegeMode::from_bits(spp);
                self.cpu.pc = self.cpu.csrs.read_unchecked(addr::SEPC);
            }
            Instruction::WFI => {
                // Check TW for S-mode
                if self.cpu.priv_mode == PrivilegeMode::Supervisor {
                    let ms = self.cpu.csrs.read_unchecked(addr::MSTATUS);
                    if ms & mstatus::TW != 0 {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                            return Ok(());
                        }
                    }
                }
                // WFI is a NOP in our implementation
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SFENCE_VMA { .. } => {
                // Check TVM for S-mode
                if self.cpu.priv_mode == PrivilegeMode::Supervisor {
                    let ms = self.cpu.csrs.read_unchecked(addr::MSTATUS);
                    if ms & mstatus::TVM != 0 {
                        if self.privileged {
                            trap::deliver_trap(&mut self.cpu, TrapCause::Exception(ExceptionCause::IllegalInstruction), 0, pc);
                            return Ok(());
                        }
                    }
                }
                self.tlb_flush();
                self.cpu.pc = pc.wrapping_add(step);
            }

            // === A Extension (Atomics) ===
            Instruction::LR_W { rd, rs1, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_word(pa) as i32 as i64 as u64;
                self.cpu.write_reg(rd, val);
                self.reservation = Some((pa, 4));
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::LR_D { rd, rs1, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_load(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let val = self.mmio_load_double(pa);
                self.cpu.write_reg(rd, val);
                self.reservation = Some((pa, 8));
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SC_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                if self.reservation == Some((pa, 4)) {
                    self.mmio_store_word(pa, self.cpu.read_reg(rs2) as u32);
                    self.cpu.write_reg(rd, 0);
                } else {
                    self.cpu.write_reg(rd, 1);
                }
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::SC_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                if self.reservation == Some((pa, 8)) {
                    self.mmio_store_double(pa, self.cpu.read_reg(rs2));
                    self.cpu.write_reg(rd, 0);
                } else {
                    self.cpu.write_reg(rd, 1);
                }
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }

            // AMO.W operations
            Instruction::AMOSWAP_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa) as i32 as i64 as u64;
                self.mmio_store_word(pa, self.cpu.read_reg(rs2) as u32);
                self.cpu.write_reg(rd, old);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOADD_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa) as i32;
                let result = old.wrapping_add(self.cpu.read_reg(rs2) as i32);
                self.mmio_store_word(pa, result as u32);
                self.cpu.write_reg(rd, old as i64 as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOAND_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa);
                let result = old & self.cpu.read_reg(rs2) as u32;
                self.mmio_store_word(pa, result);
                self.cpu.write_reg(rd, old as i32 as i64 as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOOR_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa);
                let result = old | self.cpu.read_reg(rs2) as u32;
                self.mmio_store_word(pa, result);
                self.cpu.write_reg(rd, old as i32 as i64 as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOXOR_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa);
                let result = old ^ self.cpu.read_reg(rs2) as u32;
                self.mmio_store_word(pa, result);
                self.cpu.write_reg(rd, old as i32 as i64 as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOMAX_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa) as i32;
                let src = self.cpu.read_reg(rs2) as i32;
                let result = old.max(src);
                self.mmio_store_word(pa, result as u32);
                self.cpu.write_reg(rd, old as i64 as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOMIN_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa) as i32;
                let src = self.cpu.read_reg(rs2) as i32;
                let result = old.min(src);
                self.mmio_store_word(pa, result as u32);
                self.cpu.write_reg(rd, old as i64 as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOMAXU_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa);
                let src = self.cpu.read_reg(rs2) as u32;
                let result = old.max(src);
                self.mmio_store_word(pa, result);
                self.cpu.write_reg(rd, old as i32 as i64 as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOMINU_W { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_word(pa);
                let src = self.cpu.read_reg(rs2) as u32;
                let result = old.min(src);
                self.mmio_store_word(pa, result);
                self.cpu.write_reg(rd, old as i32 as i64 as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }

            // AMO.D operations
            Instruction::AMOSWAP_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa);
                self.mmio_store_double(pa, self.cpu.read_reg(rs2));
                self.cpu.write_reg(rd, old);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOADD_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa);
                let result = old.wrapping_add(self.cpu.read_reg(rs2));
                self.mmio_store_double(pa, result);
                self.cpu.write_reg(rd, old);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOAND_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa);
                let result = old & self.cpu.read_reg(rs2);
                self.mmio_store_double(pa, result);
                self.cpu.write_reg(rd, old);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOOR_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa);
                let result = old | self.cpu.read_reg(rs2);
                self.mmio_store_double(pa, result);
                self.cpu.write_reg(rd, old);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOXOR_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa);
                let result = old ^ self.cpu.read_reg(rs2);
                self.mmio_store_double(pa, result);
                self.cpu.write_reg(rd, old);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOMAX_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa) as i64;
                let src = self.cpu.read_reg(rs2) as i64;
                let result = old.max(src);
                self.mmio_store_double(pa, result as u64);
                self.cpu.write_reg(rd, old as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOMIN_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa) as i64;
                let src = self.cpu.read_reg(rs2) as i64;
                let result = old.min(src);
                self.mmio_store_double(pa, result as u64);
                self.cpu.write_reg(rd, old as u64);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOMAXU_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa);
                let src = self.cpu.read_reg(rs2);
                let result = old.max(src);
                self.mmio_store_double(pa, result);
                self.cpu.write_reg(rd, old);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
            Instruction::AMOMINU_D { rd, rs1, rs2, .. } => {
                let vaddr = self.cpu.read_reg(rs1);
                let pa = match self.translate_store(vaddr) { Ok(a) => a, Err(()) => return Ok(()) };
                let old = self.mmio_load_double(pa);
                let src = self.cpu.read_reg(rs2);
                let result = old.min(src);
                self.mmio_store_double(pa, result);
                self.cpu.write_reg(rd, old);
                self.reservation = None;
                self.cpu.pc = pc.wrapping_add(step);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vm(program: &[u32]) -> Vm {
        let mut memory = Memory::new();
        for (i, &word) in program.iter().enumerate() {
            memory.store_word((i * 4) as u64, word);
        }
        Vm::new(CpuState::new(), memory)
    }

    #[test]
    fn test_addi() {
        let mut vm = make_vm(&[0x02A00093, 0x00000073]);
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(1), 42);
    }

    #[test]
    fn test_add() {
        let mut vm = make_vm(&[
            0x00A00093, 0x01400113, 0x002081B3, 0x00000073,
        ]);
        vm.run(3).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 30);
    }

    #[test]
    fn test_sub() {
        let mut vm = make_vm(&[
            0x01E00093, 0x00A00113, 0x402081B3, 0x00000073,
        ]);
        vm.run(3).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 20);
    }

    #[test]
    fn test_mul() {
        let mut vm = make_vm(&[
            0x00600093, 0x00700113, 0x022081B3, 0x00000073,
        ]);
        vm.run(3).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 42);
    }

    #[test]
    fn test_div() {
        let mut vm = make_vm(&[
            0x02A00093, 0x00600113, 0x0220C1B3, 0x00000073,
        ]);
        vm.run(3).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 7);
    }

    #[test]
    fn test_div_by_zero() {
        let mut vm = make_vm(&[
            0x02A00093, 0x0200C1B3, 0x00000073,
        ]);
        vm.run(2).unwrap();
        assert_eq!(vm.cpu.read_reg(3), u64::MAX);
    }

    #[test]
    fn test_load_store() {
        let mut vm = make_vm(&[
            0x02A00093, 0x10000113, 0x00110023, 0x00010183, 0x00000073,
        ]);
        vm.run(4).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 42);
    }

    #[test]
    fn test_store_load_double() {
        let mut vm = make_vm(&[0x10000093, 0x00000137, 0x00000073]);
        vm.cpu.write_reg(1, 256);
        vm.cpu.write_reg(2, 0xCAFEBABE_DEADBEEFu64);
        vm.memory.store_double(256, 0xCAFEBABE_DEADBEEFu64);
        vm.memory.store_word(0, 0x0000B183);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(3), 0xCAFEBABE_DEADBEEFu64);
    }

    #[test]
    fn test_branch_taken() {
        let mut vm = make_vm(&[
            0x00A00093, 0x00A00113, 0x00208463, 0x00100193, 0x00200193, 0x00000073,
        ]);
        vm.run(4).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 2);
    }

    #[test]
    fn test_branch_not_taken() {
        let mut vm = make_vm(&[
            0x00A00093, 0x00B00113, 0x00208463, 0x00100193, 0x00000073,
        ]);
        vm.run(4).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 1);
    }

    #[test]
    fn test_jal() {
        let mut vm = make_vm(&[
            0x008000EF, 0x00000013, 0x02A00193, 0x00000073,
        ]);
        vm.run(3).unwrap();
        assert_eq!(vm.cpu.read_reg(1), 4);
        assert_eq!(vm.cpu.read_reg(3), 42);
    }

    #[test]
    fn test_lui() {
        let mut vm = make_vm(&[0x12345_0B7, 0x00000073]);
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(1), 0x12345000);
    }

    #[test]
    fn test_auipc() {
        let mut vm = make_vm(&[0x00001097, 0x00000073]);
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(1), 0x1000);
    }

    #[test]
    fn test_fibonacci() {
        let mut vm = make_vm(&[
            0x00A00093, 0x00000113, 0x00100193,
            0x00008C63, 0x00310233, 0x00018113,
            0x00020193, 0xFFF08093, 0xFE0006E3,
            0x00000073,
        ]);
        vm.run(200).unwrap();
        assert_eq!(vm.cpu.read_reg(2), 55);
    }

    #[test]
    fn test_ecall_halts() {
        let mut vm = make_vm(&[0x00000073]);
        vm.step().unwrap();
        assert!(vm.halted);
    }

    #[test]
    fn test_and_or_xor() {
        let mut vm = make_vm(&[
            0x0FF00093, 0x0F000113, 0x0020F1B3, 0x0020E233, 0x0020C2B3, 0x00000073,
        ]);
        vm.run(5).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 0xF0);
        assert_eq!(vm.cpu.read_reg(4), 0xFF);
        assert_eq!(vm.cpu.read_reg(5), 0x0F);
    }

    #[test]
    fn test_slt() {
        let mut vm = make_vm(&[
            0x00500093, 0x00A00113, 0x0020A1B3, 0x0010A233, 0x00000073,
        ]);
        vm.run(4).unwrap();
        assert_eq!(vm.cpu.read_reg(3), 1);
        assert_eq!(vm.cpu.read_reg(4), 0);
    }

    #[test]
    fn test_shifts() {
        let mut vm = make_vm(&[
            0x00100093, 0x01009113, 0x00815193, 0x00000073,
        ]);
        vm.run(3).unwrap();
        assert_eq!(vm.cpu.read_reg(2), 1 << 16);
        assert_eq!(vm.cpu.read_reg(3), 1 << 8);
    }

    #[test]
    fn test_word_operations() {
        let mut vm = make_vm(&[0x00000073]);
        vm.cpu.write_reg(1, 0x7FFFFFFF);
        vm.memory.store_word(0, 0x0010811B);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(2), 0xFFFF_FFFF_8000_0000u64);
    }

    #[test]
    fn test_ecall_exit_with_code() {
        let mut vm = make_vm(&[0x00000073]);
        vm.cpu.write_reg(17, 93);
        vm.cpu.write_reg(10, 42);
        vm.step().unwrap();
        assert!(vm.halted);
        assert_eq!(vm.exit_code, Some(42));
    }

    #[test]
    fn test_ecall_unknown_syscall() {
        let mut vm = make_vm(&[0x00000073, 0x00000073]);
        vm.cpu.write_reg(17, 999);
        vm.step().unwrap();
        assert!(!vm.halted);
        assert_eq!(vm.cpu.read_reg(10), (-38i64) as u64);
    }

    #[test]
    fn test_uart_mmio_write() {
        let mut vm = make_vm(&[0x00110023, 0x00000073]);
        vm.cpu.write_reg(1, b'Q' as u64);
        vm.cpu.write_reg(2, 0x1000_0000);
        vm.step().unwrap();
        assert_eq!(vm.mmio.uart.output, vec![b'Q']);
    }

    #[test]
    fn test_uart_mmio_read() {
        use crate::mmio::{MmioController, UartDevice};
        let uart = UartDevice::with_input(b"R".to_vec());
        let mmio = MmioController::with_uart(uart);
        let mut memory = Memory::new();
        memory.store_word(0, 0x0000C183);
        memory.store_word(4, 0x00000073);
        let mut vm = Vm::new_with_mmio(CpuState::new(), memory, mmio);
        vm.cpu.write_reg(1, 0x1000_0000);
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(3), b'R' as u64);
    }

    #[test]
    fn test_ecall_write_syscall() {
        let mut vm = make_vm(&[0x00000073]);
        let msg = b"Hello";
        for (i, &b) in msg.iter().enumerate() {
            vm.memory.store_byte(0x100 + i as u64, b);
        }
        vm.cpu.write_reg(17, 64);
        vm.cpu.write_reg(10, 1);
        vm.cpu.write_reg(11, 0x100);
        vm.cpu.write_reg(12, 5);
        vm.step().unwrap();
        assert!(!vm.halted);
        assert_eq!(vm.mmio.uart.output, b"Hello");
        assert_eq!(vm.cpu.read_reg(10), 5);
    }

    // -----------------------------------------------------------------------
    // CSR instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_csrrw_read_write() {
        let mut vm = make_vm(&[0x00000073]);
        // CSRRW x5, mscratch, x1  -- write x1 to mscratch, old value to x5
        // Encode: csr=0x340, rs1=1, funct3=1, rd=5, opcode=0x73
        let word = (0x340u32 << 20) | (1 << 15) | (1 << 12) | (5 << 7) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.write_reg(1, 0xCAFE);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(5), 0); // old mscratch was 0
        assert_eq!(vm.cpu.csrs.read_unchecked(0x340), 0xCAFE);
    }

    #[test]
    fn test_csrrs_set_bits() {
        let mut vm = make_vm(&[0x00000073]);
        vm.cpu.csrs.write_unchecked(0x340, 0x0F);
        // CSRRS x5, mscratch, x1  -- set bits from x1 into mscratch
        let word = (0x340u32 << 20) | (1 << 15) | (2 << 12) | (5 << 7) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.write_reg(1, 0xF0);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(5), 0x0F); // old value
        assert_eq!(vm.cpu.csrs.read_unchecked(0x340), 0xFF);
    }

    #[test]
    fn test_csrrc_clear_bits() {
        let mut vm = make_vm(&[0x00000073]);
        vm.cpu.csrs.write_unchecked(0x340, 0xFF);
        // CSRRC x5, mscratch, x1  -- clear bits from x1 in mscratch
        let word = (0x340u32 << 20) | (1 << 15) | (3 << 12) | (5 << 7) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.write_reg(1, 0x0F);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(5), 0xFF); // old value
        assert_eq!(vm.cpu.csrs.read_unchecked(0x340), 0xF0);
    }

    #[test]
    fn test_csrrs_rs1_zero_readonly() {
        let mut vm = make_vm(&[0x00000073]);
        vm.cpu.csrs.write_unchecked(0x340, 0x42);
        // CSRRS x5, mscratch, x0  -- rs1=0 means read-only
        let word = (0x340u32 << 20) | (0 << 15) | (2 << 12) | (5 << 7) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(5), 0x42);
        assert_eq!(vm.cpu.csrs.read_unchecked(0x340), 0x42); // unchanged
    }

    #[test]
    fn test_csrrwi() {
        let mut vm = make_vm(&[0x00000073]);
        // CSRRWI x5, mscratch, 7
        let word = (0x340u32 << 20) | (7 << 15) | (5 << 12) | (5 << 7) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.csrs.read_unchecked(0x340), 7);
    }

    // -----------------------------------------------------------------------
    // Privileged instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_privilege_starts_machine() {
        let vm = make_vm(&[0x00000073]);
        assert_eq!(vm.cpu.priv_mode, PrivilegeMode::Machine);
    }

    #[test]
    fn test_mret_returns_to_mepc() {
        let mut vm = make_vm(&[0x00000073]);
        vm.cpu.csrs.write_unchecked(addr::MEPC, 0x1000);
        vm.cpu.csrs.write_unchecked(addr::MSTATUS, mstatus::MPIE);
        // MRET encoding: funct7=0x18, rs2=2, funct3=0, opcode=0x73
        let word = (0x18u32 << 25) | (2 << 20) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.pc, 0x1000);
    }

    #[test]
    fn test_sret() {
        let mut vm = make_vm(&[0x00000073]);
        vm.cpu.csrs.write_unchecked(addr::SEPC, 0x2000);
        vm.cpu.csrs.write_unchecked(addr::MSTATUS, mstatus::SPIE);
        // SRET encoding: funct7=0x08, rs2=2, funct3=0, opcode=0x73
        let word = (0x08u32 << 25) | (2 << 20) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.pc, 0x2000);
    }

    #[test]
    fn test_wfi_is_nop() {
        let mut vm = make_vm(&[0x00000073]);
        // WFI encoding: funct7=0x08, rs2=5, funct3=0, opcode=0x73
        let word = (0x08u32 << 25) | (5 << 20) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.pc, 4);
    }

    #[test]
    fn test_sfence_vma_nop() {
        let mut vm = make_vm(&[0x00000073]);
        // SFENCE.VMA: funct7=0x09, rs1=10, rs2=11
        let word = (0x09u32 << 25) | (11 << 20) | (10 << 15) | 0x73;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.pc, 4);
    }

    // -----------------------------------------------------------------------
    // Privileged ECALL trap test
    // -----------------------------------------------------------------------

    #[test]
    fn test_ecall_privileged_delivers_trap() {
        let mut memory = Memory::new();
        memory.store_word(0, 0x00000073); // ECALL
        let mut cpu = CpuState::new();
        cpu.csrs.write_unchecked(addr::MTVEC, 0x8000_0000);
        let mut vm = Vm::new_privileged(cpu, memory);
        vm.cpu.priv_mode = PrivilegeMode::User;
        vm.step().unwrap();
        // Should have trapped to mtvec
        assert_eq!(vm.cpu.pc, 0x8000_0000);
        assert_eq!(vm.cpu.csrs.read_unchecked(addr::MCAUSE), 8); // ecall from U
    }

    #[test]
    fn test_non_privileged_preserves_legacy() {
        let mut vm = make_vm(&[0x00000073]); // ECALL with a7=0
        vm.step().unwrap();
        assert!(vm.halted);
    }

    // -----------------------------------------------------------------------
    // Atomic instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lr_sc_w_success() {
        let mut vm = make_vm(&[0x00000073]);
        vm.memory.store_word(0x100, 42);
        vm.cpu.write_reg(1, 0x100); // rs1 = addr
        vm.cpu.write_reg(2, 99);    // rs2 = new value

        // LR.W x3, (x1)
        let lr_w = (0x02u32 << 27) | (0 << 20) | (1 << 15) | (2 << 12) | (3 << 7) | 0x2F;
        vm.memory.store_word(0, lr_w);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(3), 42);
        assert_eq!(vm.reservation, Some((0x100, 4)));

        // SC.W x4, x2, (x1)
        let sc_w = (0x03u32 << 27) | (2 << 20) | (1 << 15) | (2 << 12) | (4 << 7) | 0x2F;
        vm.memory.store_word(4, sc_w);
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(4), 0); // success
        assert_eq!(vm.memory.load_word(0x100), 99);
        assert!(vm.reservation.is_none());
    }

    #[test]
    fn test_lr_sc_w_failure() {
        let mut vm = make_vm(&[0x00000073]);
        vm.memory.store_word(0x100, 42);
        vm.cpu.write_reg(1, 0x100);
        vm.cpu.write_reg(2, 99);

        // LR.W x3, (x1)
        let lr_w = (0x02u32 << 27) | (0 << 20) | (1 << 15) | (2 << 12) | (3 << 7) | 0x2F;
        vm.memory.store_word(0, lr_w);
        vm.cpu.pc = 0;
        vm.step().unwrap();

        // Intervening store clears reservation
        vm.reservation = None;

        // SC.W x4, x2, (x1)
        let sc_w = (0x03u32 << 27) | (2 << 20) | (1 << 15) | (2 << 12) | (4 << 7) | 0x2F;
        vm.memory.store_word(4, sc_w);
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(4), 1); // failure
        assert_eq!(vm.memory.load_word(0x100), 42); // unchanged
    }

    #[test]
    fn test_amoadd_w() {
        let mut vm = make_vm(&[0x00000073]);
        vm.memory.store_word(0x100, 10);
        vm.cpu.write_reg(1, 0x100); // addr
        vm.cpu.write_reg(2, 5);     // value to add

        // AMOADD.W x3, x2, (x1)
        let word = (0x00u32 << 27) | (2 << 20) | (1 << 15) | (2 << 12) | (3 << 7) | 0x2F;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(3), 10); // old value
        assert_eq!(vm.memory.load_word(0x100), 15); // 10 + 5
    }

    #[test]
    fn test_amoswap_d() {
        let mut vm = make_vm(&[0x00000073]);
        vm.memory.store_double(0x100, 0xDEADBEEF);
        vm.cpu.write_reg(1, 0x100);
        vm.cpu.write_reg(2, 0xCAFEBABE);

        // AMOSWAP.D x3, x2, (x1)
        let word = (0x01u32 << 27) | (2 << 20) | (1 << 15) | (3 << 12) | (3 << 7) | 0x2F;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(3), 0xDEADBEEF);
        assert_eq!(vm.memory.load_double(0x100), 0xCAFEBABE);
    }

    #[test]
    fn test_store_clears_reservation() {
        let mut vm = make_vm(&[0x00000073]);
        vm.reservation = Some((0x100, 4));
        vm.cpu.write_reg(1, 0x100);
        vm.cpu.write_reg(2, 42);

        // SW x2, 0(x1) -- store at the reservation address
        let word = (0u32 << 25) | (2 << 20) | (1 << 15) | (2 << 12) | 0x23;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert!(vm.reservation.is_none());
    }

    #[test]
    fn test_amoand_w() {
        let mut vm = make_vm(&[0x00000073]);
        vm.memory.store_word(0x100, 0xFF);
        vm.cpu.write_reg(1, 0x100);
        vm.cpu.write_reg(2, 0x0F);

        let word = (0x0Cu32 << 27) | (2 << 20) | (1 << 15) | (2 << 12) | (3 << 7) | 0x2F;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(3) as u32, 0xFF);
        assert_eq!(vm.memory.load_word(0x100), 0x0F);
    }

    #[test]
    fn test_amoor_w() {
        let mut vm = make_vm(&[0x00000073]);
        vm.memory.store_word(0x100, 0xF0);
        vm.cpu.write_reg(1, 0x100);
        vm.cpu.write_reg(2, 0x0F);

        let word = (0x08u32 << 27) | (2 << 20) | (1 << 15) | (2 << 12) | (3 << 7) | 0x2F;
        vm.memory.store_word(0, word);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.memory.load_word(0x100), 0xFF);
    }

    #[test]
    fn test_lr_sc_d() {
        let mut vm = make_vm(&[0x00000073]);
        vm.memory.store_double(0x100, 0xDEAD_BEEF_CAFE_BABEu64);
        vm.cpu.write_reg(1, 0x100);
        vm.cpu.write_reg(2, 0x1234_5678_9ABC_DEF0u64);

        // LR.D x3, (x1)
        let lr_d = (0x02u32 << 27) | (0 << 20) | (1 << 15) | (3 << 12) | (3 << 7) | 0x2F;
        vm.memory.store_word(0, lr_d);
        vm.cpu.pc = 0;
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(3), 0xDEAD_BEEF_CAFE_BABEu64);
        assert_eq!(vm.reservation, Some((0x100, 8)));

        // SC.D x4, x2, (x1)
        let sc_d = (0x03u32 << 27) | (2 << 20) | (1 << 15) | (3 << 12) | (4 << 7) | 0x2F;
        vm.memory.store_word(4, sc_d);
        vm.step().unwrap();
        assert_eq!(vm.cpu.read_reg(4), 0);
        assert_eq!(vm.memory.load_double(0x100), 0x1234_5678_9ABC_DEF0u64);
    }

    // -----------------------------------------------------------------------
    // Trap + MRET roundtrip test
    // -----------------------------------------------------------------------

    #[test]
    fn test_trap_mret_roundtrip() {
        let mut memory = Memory::new();
        // At 0x0: ECALL
        memory.store_word(0, 0x00000073);
        // At 0x8000_0000: MRET
        let mret_word = (0x18u32 << 25) | (2 << 20) | 0x73;
        memory.store_word(0x8000_0000, mret_word);
        // At 0x4: NOP (ADDI x0, x0, 0)
        memory.store_word(4, 0x00000013);

        let mut cpu = CpuState::new();
        cpu.csrs.write_unchecked(addr::MTVEC, 0x8000_0000);
        cpu.csrs.write_unchecked(addr::MSTATUS, mstatus::MIE);
        cpu.priv_mode = PrivilegeMode::User;

        let mut vm = Vm::new_privileged(cpu, memory);

        // Step 1: ECALL -> traps to 0x8000_0000
        vm.step().unwrap();
        assert_eq!(vm.cpu.pc, 0x8000_0000);
        assert_eq!(vm.cpu.priv_mode, PrivilegeMode::Machine);
        assert_eq!(vm.cpu.csrs.read_unchecked(addr::MEPC), 0); // ecall PC

        // Step 2: MRET -> returns to mepc (0), but we need to advance past ecall
        // First fix mepc to point past ecall
        vm.cpu.csrs.write_unchecked(addr::MEPC, 4);
        vm.step().unwrap();
        assert_eq!(vm.cpu.pc, 4);
        assert_eq!(vm.cpu.priv_mode, PrivilegeMode::User);
    }

    // -----------------------------------------------------------------------
    // Interrupt tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interrupt_pending_fires() {
        let mut memory = Memory::new();
        memory.store_word(0, 0x00000013); // NOP
        let mut cpu = CpuState::new();
        cpu.csrs.write_unchecked(addr::MTVEC, 0x100);
        cpu.csrs.write_unchecked(addr::MSTATUS, mstatus::MIE);
        cpu.csrs.write_unchecked(addr::MIE, interrupt::MTIP);
        cpu.csrs.write_unchecked(addr::MIP, interrupt::MTIP);

        let mut vm = Vm::new_privileged(cpu, memory);
        vm.step().unwrap();
        // Should have trapped
        assert_eq!(vm.cpu.pc, 0x100);
    }

    #[test]
    fn test_interrupt_disabled_no_fire() {
        let mut memory = Memory::new();
        memory.store_word(0, 0x00000013); // NOP
        let mut cpu = CpuState::new();
        cpu.csrs.write_unchecked(addr::MTVEC, 0x100);
        // MIE is NOT set, so interrupt shouldn't fire
        cpu.csrs.write_unchecked(addr::MSTATUS, 0);
        cpu.csrs.write_unchecked(addr::MIE, interrupt::MTIP);
        cpu.csrs.write_unchecked(addr::MIP, interrupt::MTIP);

        let mut vm = Vm::new_privileged(cpu, memory);
        vm.step().unwrap();
        // Should have executed the NOP, not trapped
        assert_eq!(vm.cpu.pc, 4);
    }
}
