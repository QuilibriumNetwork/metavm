use crate::csr::{CsrFile, PrivilegeMode};

/// CPU state for RV64IM: 32 general-purpose 64-bit registers + program counter.
/// Register x0 is hardwired to zero.
#[derive(Clone, Debug)]
pub struct CpuState {
    pub pc: u64,
    x: [u64; 32],
    pub priv_mode: PrivilegeMode,
    pub csrs: CsrFile,
}

impl CpuState {
    pub fn new() -> Self {
        CpuState {
            pc: 0,
            x: [0u64; 32],
            priv_mode: PrivilegeMode::Machine,
            csrs: CsrFile::new(),
        }
    }

    /// Create a CPU state with a given initial PC.
    pub fn with_pc(pc: u64) -> Self {
        CpuState {
            pc,
            x: [0u64; 32],
            priv_mode: PrivilegeMode::Machine,
            csrs: CsrFile::new(),
        }
    }

    /// Read register `reg`. x0 always returns 0.
    #[inline]
    pub fn read_reg(&self, reg: u8) -> u64 {
        if reg == 0 {
            0
        } else {
            self.x[reg as usize]
        }
    }

    /// Write register `reg`. Writes to x0 are silently ignored.
    #[inline]
    pub fn write_reg(&mut self, reg: u8, val: u64) {
        if reg != 0 {
            self.x[reg as usize] = val;
        }
    }

    /// Get a snapshot of all registers (for trace/debug).
    pub fn registers(&self) -> &[u64; 32] {
        &self.x
    }
}

impl Default for CpuState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x0_always_zero() {
        let mut cpu = CpuState::new();
        cpu.write_reg(0, 42);
        assert_eq!(cpu.read_reg(0), 0);
    }

    #[test]
    fn test_register_read_write() {
        let mut cpu = CpuState::new();
        cpu.write_reg(1, 100);
        assert_eq!(cpu.read_reg(1), 100);
        cpu.write_reg(31, 0xDEADBEEF);
        assert_eq!(cpu.read_reg(31), 0xDEADBEEF);
    }

    #[test]
    fn test_initial_state() {
        let cpu = CpuState::new();
        assert_eq!(cpu.pc, 0);
        for i in 0..32 {
            assert_eq!(cpu.read_reg(i), 0);
        }
    }

    #[test]
    fn test_with_pc() {
        let cpu = CpuState::with_pc(0x8000_0000);
        assert_eq!(cpu.pc, 0x8000_0000);
    }
}
