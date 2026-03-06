use crate::cpu::CpuState;
use crate::csr::{addr, mstatus, PrivilegeMode};

/// Exception causes (synchronous traps).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum ExceptionCause {
    InstructionAddressMisaligned = 0,
    InstructionAccessFault = 1,
    IllegalInstruction = 2,
    Breakpoint = 3,
    LoadAddressMisaligned = 4,
    LoadAccessFault = 5,
    StoreAddressMisaligned = 6,
    StoreAccessFault = 7,
    EnvironmentCallFromU = 8,
    EnvironmentCallFromS = 9,
    EnvironmentCallFromM = 11,
    InstructionPageFault = 12,
    LoadPageFault = 13,
    StorePageFault = 15,
}

/// Interrupt causes (asynchronous traps).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum InterruptCause {
    SupervisorSoftware = 1,
    MachineSoftware = 3,
    SupervisorTimer = 5,
    MachineTimer = 7,
    SupervisorExternal = 9,
    MachineExternal = 11,
}

/// Combined trap cause (exception or interrupt).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrapCause {
    Exception(ExceptionCause),
    Interrupt(InterruptCause),
}

impl TrapCause {
    /// Encode as the xcause register value (bit 63 set for interrupts).
    pub fn to_cause_value(&self) -> u64 {
        match self {
            TrapCause::Exception(e) => *e as u64,
            TrapCause::Interrupt(i) => (1u64 << 63) | (*i as u64),
        }
    }
}

/// Deliver a trap to the CPU, updating privilege level and CSRs.
pub fn deliver_trap(cpu: &mut CpuState, cause: TrapCause, tval: u64, epc: u64) {
    let cause_val = cause.to_cause_value();
    let is_interrupt = matches!(cause, TrapCause::Interrupt(_));
    let cause_code = match cause {
        TrapCause::Exception(e) => e as u64,
        TrapCause::Interrupt(i) => i as u64,
    };

    // Check delegation: exceptions use medeleg, interrupts use mideleg.
    // Traps from M-mode are never delegated.
    let delegated = cpu.priv_mode != PrivilegeMode::Machine && {
        let deleg_reg = if is_interrupt { addr::MIDELEG } else { addr::MEDELEG };
        let deleg = cpu.csrs.read_unchecked(deleg_reg);
        (deleg >> cause_code) & 1 != 0
    };

    if delegated {
        // Deliver to S-mode
        cpu.csrs.write_unchecked(addr::SEPC, epc);
        cpu.csrs.write_unchecked(addr::SCAUSE, cause_val);
        cpu.csrs.write_unchecked(addr::STVAL, tval);

        // Update sstatus: SPIE = old SIE, SPP = old privilege, SIE = 0
        let old_mstatus = cpu.csrs.read_unchecked(addr::MSTATUS);
        let old_sie = (old_mstatus & mstatus::SIE) != 0;
        let mut new_mstatus = old_mstatus;
        if old_sie {
            new_mstatus |= mstatus::SPIE;
        } else {
            new_mstatus &= !mstatus::SPIE;
        }
        // SPP = old privilege (0=U, 1=S)
        if cpu.priv_mode == PrivilegeMode::Supervisor {
            new_mstatus |= mstatus::SPP;
        } else {
            new_mstatus &= !mstatus::SPP;
        }
        new_mstatus &= !mstatus::SIE; // disable interrupts
        cpu.csrs.write_unchecked(addr::MSTATUS, new_mstatus);

        // Jump to stvec
        let stvec = cpu.csrs.read_unchecked(addr::STVEC);
        let mode = stvec & 0x3;
        let base = stvec & !0x3;
        cpu.pc = if mode == 1 && is_interrupt {
            base.wrapping_add(cause_code * 4)
        } else {
            base
        };

        cpu.priv_mode = PrivilegeMode::Supervisor;
    } else {
        // Deliver to M-mode
        cpu.csrs.write_unchecked(addr::MEPC, epc);
        cpu.csrs.write_unchecked(addr::MCAUSE, cause_val);
        cpu.csrs.write_unchecked(addr::MTVAL, tval);

        // Update mstatus: MPIE = old MIE, MPP = old privilege, MIE = 0
        let old_mstatus = cpu.csrs.read_unchecked(addr::MSTATUS);
        let old_mie = (old_mstatus & mstatus::MIE) != 0;
        let mut new_mstatus = old_mstatus;
        if old_mie {
            new_mstatus |= mstatus::MPIE;
        } else {
            new_mstatus &= !mstatus::MPIE;
        }
        // MPP = old privilege level
        new_mstatus &= !mstatus::MPP_MASK;
        new_mstatus |= (cpu.priv_mode as u64) << mstatus::MPP_SHIFT;
        new_mstatus &= !mstatus::MIE; // disable interrupts
        cpu.csrs.write_unchecked(addr::MSTATUS, new_mstatus);

        // Jump to mtvec
        let mtvec = cpu.csrs.read_unchecked(addr::MTVEC);
        let mode = mtvec & 0x3;
        let base = mtvec & !0x3;
        cpu.pc = if mode == 1 && is_interrupt {
            base.wrapping_add(cause_code * 4)
        } else {
            base
        };

        cpu.priv_mode = PrivilegeMode::Machine;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::{addr, mstatus, interrupt};

    fn make_cpu() -> CpuState {
        CpuState::new()
    }

    #[test]
    fn test_exception_cause_encoding() {
        let cause = TrapCause::Exception(ExceptionCause::IllegalInstruction);
        assert_eq!(cause.to_cause_value(), 2);
    }

    #[test]
    fn test_interrupt_cause_encoding() {
        let cause = TrapCause::Interrupt(InterruptCause::MachineTimer);
        assert_eq!(cause.to_cause_value(), (1u64 << 63) | 7);
    }

    #[test]
    fn test_deliver_exception_to_m_mode() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::Machine;
        cpu.csrs.write_unchecked(addr::MTVEC, 0x8000_0000);
        cpu.csrs.write_unchecked(addr::MSTATUS, mstatus::MIE);

        deliver_trap(
            &mut cpu,
            TrapCause::Exception(ExceptionCause::IllegalInstruction),
            0xDEAD,
            0x1000,
        );

        assert_eq!(cpu.pc, 0x8000_0000);
        assert_eq!(cpu.csrs.read_unchecked(addr::MEPC), 0x1000);
        assert_eq!(cpu.csrs.read_unchecked(addr::MCAUSE), 2);
        assert_eq!(cpu.csrs.read_unchecked(addr::MTVAL), 0xDEAD);
        assert_eq!(cpu.priv_mode, PrivilegeMode::Machine);
        // MIE should be cleared, MPIE should be set
        let ms = cpu.csrs.read_unchecked(addr::MSTATUS);
        assert_eq!(ms & mstatus::MIE, 0);
        assert_ne!(ms & mstatus::MPIE, 0);
    }

    #[test]
    fn test_deliver_exception_delegated_to_s_mode() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::User;
        cpu.csrs.write_unchecked(addr::STVEC, 0x8020_0000);
        // Delegate illegal instruction (cause 2) to S-mode
        cpu.csrs.write_unchecked(addr::MEDELEG, 1 << 2);
        cpu.csrs.write_unchecked(addr::MSTATUS, mstatus::SIE);

        deliver_trap(
            &mut cpu,
            TrapCause::Exception(ExceptionCause::IllegalInstruction),
            0xBEEF,
            0x2000,
        );

        assert_eq!(cpu.pc, 0x8020_0000);
        assert_eq!(cpu.csrs.read_unchecked(addr::SEPC), 0x2000);
        assert_eq!(cpu.csrs.read_unchecked(addr::SCAUSE), 2);
        assert_eq!(cpu.csrs.read_unchecked(addr::STVAL), 0xBEEF);
        assert_eq!(cpu.priv_mode, PrivilegeMode::Supervisor);
        // SIE should be cleared, SPIE should be set
        let ms = cpu.csrs.read_unchecked(addr::MSTATUS);
        assert_eq!(ms & mstatus::SIE, 0);
        assert_ne!(ms & mstatus::SPIE, 0);
    }

    #[test]
    fn test_mstatus_mie_mpie_save_restore() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::Machine;
        cpu.csrs.write_unchecked(addr::MTVEC, 0x100);
        // MIE is set
        cpu.csrs.write_unchecked(addr::MSTATUS, mstatus::MIE);

        deliver_trap(
            &mut cpu,
            TrapCause::Exception(ExceptionCause::Breakpoint),
            0,
            0x50,
        );

        let ms = cpu.csrs.read_unchecked(addr::MSTATUS);
        // MIE cleared
        assert_eq!(ms & mstatus::MIE, 0);
        // MPIE set (old MIE was 1)
        assert_ne!(ms & mstatus::MPIE, 0);
        // MPP should be 3 (Machine)
        assert_eq!((ms & mstatus::MPP_MASK) >> mstatus::MPP_SHIFT, 3);
    }

    #[test]
    fn test_spp_save() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::Supervisor;
        cpu.csrs.write_unchecked(addr::STVEC, 0x200);
        cpu.csrs.write_unchecked(addr::MEDELEG, 1 << 8); // delegate ecall from U

        // But we're in S-mode, so ecall from S (cause 9) if delegated
        cpu.csrs.write_unchecked(addr::MEDELEG, 1 << 9);
        deliver_trap(
            &mut cpu,
            TrapCause::Exception(ExceptionCause::EnvironmentCallFromS),
            0,
            0x300,
        );

        let ms = cpu.csrs.read_unchecked(addr::MSTATUS);
        // SPP should be 1 (Supervisor)
        assert_ne!(ms & mstatus::SPP, 0);
    }

    #[test]
    fn test_mtvec_direct_mode() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::Machine;
        // Direct mode: mode=0
        cpu.csrs.write_unchecked(addr::MTVEC, 0x8000_0000);

        deliver_trap(
            &mut cpu,
            TrapCause::Interrupt(InterruptCause::MachineTimer),
            0,
            0x100,
        );

        assert_eq!(cpu.pc, 0x8000_0000);
    }

    #[test]
    fn test_mtvec_vectored_mode() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::Machine;
        // Vectored mode: mode=1
        cpu.csrs.write_unchecked(addr::MTVEC, 0x8000_0001);

        deliver_trap(
            &mut cpu,
            TrapCause::Interrupt(InterruptCause::MachineTimer), // cause 7
            0,
            0x100,
        );

        // Vectored: base + cause * 4 = 0x8000_0000 + 7 * 4 = 0x8000_001C
        assert_eq!(cpu.pc, 0x8000_001C);
    }

    #[test]
    fn test_vectored_mode_exception_uses_base() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::Machine;
        cpu.csrs.write_unchecked(addr::MTVEC, 0x8000_0001); // vectored

        deliver_trap(
            &mut cpu,
            TrapCause::Exception(ExceptionCause::IllegalInstruction),
            0,
            0x100,
        );

        // Exceptions always use base address, even in vectored mode
        assert_eq!(cpu.pc, 0x8000_0000);
    }

    #[test]
    fn test_delegation_checks() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::User;
        cpu.csrs.write_unchecked(addr::MTVEC, 0x1000);
        cpu.csrs.write_unchecked(addr::STVEC, 0x2000);
        // No delegation
        cpu.csrs.write_unchecked(addr::MEDELEG, 0);

        deliver_trap(
            &mut cpu,
            TrapCause::Exception(ExceptionCause::IllegalInstruction),
            0,
            0x100,
        );

        // Should go to M-mode (mtvec)
        assert_eq!(cpu.pc, 0x1000);
        assert_eq!(cpu.priv_mode, PrivilegeMode::Machine);
    }

    #[test]
    fn test_no_delegation_from_m_mode() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::Machine;
        cpu.csrs.write_unchecked(addr::MTVEC, 0x1000);
        cpu.csrs.write_unchecked(addr::STVEC, 0x2000);
        // Even with delegation set, M-mode traps go to M-mode
        cpu.csrs.write_unchecked(addr::MEDELEG, u64::MAX);

        deliver_trap(
            &mut cpu,
            TrapCause::Exception(ExceptionCause::Breakpoint),
            0,
            0x100,
        );

        assert_eq!(cpu.pc, 0x1000);
        assert_eq!(cpu.priv_mode, PrivilegeMode::Machine);
    }

    #[test]
    fn test_ecall_causes() {
        assert_eq!(ExceptionCause::EnvironmentCallFromU as u64, 8);
        assert_eq!(ExceptionCause::EnvironmentCallFromS as u64, 9);
        assert_eq!(ExceptionCause::EnvironmentCallFromM as u64, 11);
    }

    #[test]
    fn test_page_fault_causes() {
        assert_eq!(ExceptionCause::InstructionPageFault as u64, 12);
        assert_eq!(ExceptionCause::LoadPageFault as u64, 13);
        assert_eq!(ExceptionCause::StorePageFault as u64, 15);
    }

    #[test]
    fn test_interrupt_delegation() {
        let mut cpu = make_cpu();
        cpu.priv_mode = PrivilegeMode::Supervisor;
        cpu.csrs.write_unchecked(addr::MTVEC, 0x1000);
        cpu.csrs.write_unchecked(addr::STVEC, 0x2000);
        // Delegate supervisor timer interrupt (cause 5)
        cpu.csrs.write_unchecked(addr::MIDELEG, 1 << 5);

        deliver_trap(
            &mut cpu,
            TrapCause::Interrupt(InterruptCause::SupervisorTimer),
            0,
            0x100,
        );

        assert_eq!(cpu.pc, 0x2000);
        assert_eq!(cpu.priv_mode, PrivilegeMode::Supervisor);
    }
}
