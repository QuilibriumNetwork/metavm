use crate::virtio::{self, VirtioBlockDevice};

/// QEMU virt-compatible UART base address (NS16550).
const UART_BASE: u64 = 0x1000_0000;
const UART_SIZE: u64 = 8;

/// NS16550-compatible UART device with full register tracking.
pub struct UartDevice {
    pub output: Vec<u8>,
    input: Vec<u8>,
    input_cursor: usize,
    /// Interrupt Enable Register (offset 1 when DLAB=0)
    ier: u8,
    /// Line Control Register (offset 3). Bit 7 = DLAB.
    lcr: u8,
    /// Modem Control Register (offset 4)
    mcr: u8,
    /// FIFO Control Register (offset 2 write). Bit 0 = FIFO enable.
    fcr: u8,
    /// Divisor Latch Low (offset 0 when DLAB=1)
    dll: u8,
    /// Divisor Latch High (offset 1 when DLAB=1)
    dlm: u8,
    /// Scratch register (offset 7)
    scratch: u8,
}

impl UartDevice {
    pub fn new() -> Self {
        UartDevice {
            output: Vec::new(),
            input: Vec::new(),
            input_cursor: 0,
            ier: 0,
            lcr: 0x03, // 8-N-1
            mcr: 0,
            fcr: 0,
            dll: 0x01, // 115200 baud default
            dlm: 0,
            scratch: 0,
        }
    }

    pub fn with_input(input: Vec<u8>) -> Self {
        UartDevice {
            output: Vec::new(),
            input,
            input_cursor: 0,
            ier: 0,
            lcr: 0x03,
            mcr: 0,
            fcr: 0,
            dll: 0x01,
            dlm: 0,
            scratch: 0,
        }
    }

    fn dlab(&self) -> bool {
        (self.lcr & 0x80) != 0
    }

    /// Read a UART register at the given offset (0..8).
    pub fn read(&self, offset: u64) -> u8 {
        match offset {
            0 => {
                if self.dlab() {
                    self.dll
                } else {
                    // RBR: non-destructive peek
                    if self.input_cursor < self.input.len() {
                        self.input[self.input_cursor]
                    } else {
                        0
                    }
                }
            }
            1 => {
                if self.dlab() {
                    self.dlm
                } else {
                    self.ier
                }
            }
            // IIR: Interrupt Identification Register
            // The serial8250 polling timer reads IIR and returns immediately
            // if bit 0 is set ("no interrupt pending").  We must report the
            // correct interrupt status so the polling path flushes the tty
            // transmit buffer.
            2 => {
                let fifo_bits = if (self.fcr & 1) != 0 { 0xC0 } else { 0 };
                // Priority (highest to lowest):
                //   RDA (received data available): IER bit 0 set + data ready
                //   THRE (transmit holding register empty): IER bit 1 set
                if (self.ier & 0x01) != 0 && self.input_cursor < self.input.len() {
                    // Received data available — IIR type = 0x04 (bit 0 = 0)
                    fifo_bits | 0x04
                } else if (self.ier & 0x02) != 0 {
                    // Transmitter holding register empty — IIR type = 0x02
                    fifo_bits | 0x02
                } else {
                    // No interrupt pending
                    fifo_bits | 0x01
                }
            }
            3 => self.lcr,
            4 => self.mcr,
            // LSR: Line Status Register
            5 => {
                let mut lsr = 0u8;
                if self.input_cursor < self.input.len() {
                    lsr |= 1; // Data Ready
                }
                lsr |= 1 << 5; // THR Empty
                lsr |= 1 << 6; // Transmitter Empty
                lsr
            }
            // MSR: Modem Status Register — CTS and DSR asserted
            6 => 0x30,
            7 => self.scratch,
            _ => 0,
        }
    }

    /// Read a UART register, consuming input if reading RBR.
    pub fn read_mut(&mut self, offset: u64) -> u8 {
        if offset == 0 && !self.dlab() {
            if self.input_cursor < self.input.len() {
                let b = self.input[self.input_cursor];
                self.input_cursor += 1;
                b
            } else {
                0
            }
        } else {
            self.read(offset)
        }
    }

    /// Write a UART register at the given offset.
    pub fn write(&mut self, offset: u64, val: u8) {
        match offset {
            0 => {
                if self.dlab() {
                    self.dll = val;
                } else {
                    // THR: Transmit Holding Register
                    self.output.push(val);
                }
            }
            1 => {
                if self.dlab() {
                    self.dlm = val;
                } else {
                    self.ier = val & 0x0F;
                }
            }
            2 => {
                // FCR: FIFO Control Register (write-only)
                self.fcr = val;
            }
            3 => {
                self.lcr = val;
            }
            4 => {
                self.mcr = val & 0x1F;
            }
            7 => {
                self.scratch = val;
            }
            _ => {} // LSR/MSR are read-only
        }
    }
}

// ---------------------------------------------------------------------------
// CLINT (Core Local Interruptor) — QEMU virt layout
// ---------------------------------------------------------------------------

pub const CLINT_BASE: u64 = 0x0200_0000;
pub const CLINT_SIZE: u64 = 0x1_0000;

const CLINT_MSIP: u64 = 0x0000;
const CLINT_MTIMECMP: u64 = 0x4000;
const CLINT_MTIME: u64 = 0xBFF8;

pub struct ClintDevice {
    pub msip: u32,
    pub mtimecmp: u64,
    pub mtime: u64,
}

impl ClintDevice {
    pub fn new() -> Self {
        ClintDevice {
            msip: 0,
            mtimecmp: 0,
            mtime: 0,
        }
    }

    /// Advance mtime by one tick. Returns `true` when mtime >= mtimecmp.
    pub fn tick(&mut self) -> bool {
        self.mtime = self.mtime.wrapping_add(1);
        self.mtime >= self.mtimecmp
    }

    /// Return whether the software-interrupt bit is set.
    pub fn msip_pending(&self) -> bool {
        (self.msip & 1) != 0
    }

    pub fn read_u8(&self, offset: u64) -> u8 {
        if offset >= CLINT_MSIP && offset < CLINT_MSIP + 4 {
            let byte_idx = (offset - CLINT_MSIP) as u32;
            (self.msip >> (byte_idx * 8)) as u8
        } else if offset >= CLINT_MTIMECMP && offset < CLINT_MTIMECMP + 8 {
            let byte_idx = (offset - CLINT_MTIMECMP) as u64;
            (self.mtimecmp >> (byte_idx * 8)) as u8
        } else if offset >= CLINT_MTIME && offset < CLINT_MTIME + 8 {
            let byte_idx = (offset - CLINT_MTIME) as u64;
            (self.mtime >> (byte_idx * 8)) as u8
        } else {
            0
        }
    }

    pub fn write_u8(&mut self, offset: u64, val: u8) {
        if offset >= CLINT_MSIP && offset < CLINT_MSIP + 4 {
            let byte_idx = (offset - CLINT_MSIP) as u32;
            let mask = !(0xFFu32 << (byte_idx * 8));
            self.msip = (self.msip & mask) | ((val as u32) << (byte_idx * 8));
            self.msip &= 1;
        } else if offset >= CLINT_MTIMECMP && offset < CLINT_MTIMECMP + 8 {
            let byte_idx = (offset - CLINT_MTIMECMP) as u64;
            let mask = !(0xFFu64 << (byte_idx * 8));
            self.mtimecmp = (self.mtimecmp & mask) | ((val as u64) << (byte_idx * 8));
        } else if offset >= CLINT_MTIME && offset < CLINT_MTIME + 8 {
            let byte_idx = (offset - CLINT_MTIME) as u64;
            let mask = !(0xFFu64 << (byte_idx * 8));
            self.mtime = (self.mtime & mask) | ((val as u64) << (byte_idx * 8));
        }
    }
}

// ---------------------------------------------------------------------------
// PLIC (Platform-Level Interrupt Controller) — QEMU virt layout
// ---------------------------------------------------------------------------

pub const PLIC_BASE: u64 = 0x0C00_0000;
pub const PLIC_SIZE: u64 = 0x0400_0000;

const PLIC_PRIORITY_BASE: u64 = 0x0000;
const PLIC_PENDING_BASE: u64 = 0x1000;
const PLIC_ENABLE_BASE: u64 = 0x2000;
const PLIC_ENABLE_STRIDE: u64 = 0x80;
const PLIC_CONTEXT_BASE: u64 = 0x20_0000;
const PLIC_CONTEXT_STRIDE: u64 = 0x1000;

pub struct PlicDevice {
    pub priority: [u32; 64],
    pub pending: [u32; 2],
    pub enable: [[u32; 2]; 2],
    pub threshold: [u32; 2],
    pub claimed: [u32; 2],
}

impl PlicDevice {
    pub fn new() -> Self {
        PlicDevice {
            priority: [0; 64],
            pending: [0; 2],
            enable: [[0; 2]; 2],
            threshold: [0; 2],
            claimed: [0; 2],
        }
    }

    pub fn set_pending(&mut self, source: u32) {
        if source == 0 || source >= 64 {
            return;
        }
        let word = (source / 32) as usize;
        let bit = source % 32;
        self.pending[word] |= 1 << bit;
    }

    pub fn has_pending(&self, context: usize) -> bool {
        if context >= 2 {
            return false;
        }
        for src in 1u32..64 {
            let word = (src / 32) as usize;
            let bit = src % 32;
            let is_pending = (self.pending[word] >> bit) & 1 != 0;
            let is_enabled = (self.enable[context][word] >> bit) & 1 != 0;
            if is_pending && is_enabled && self.priority[src as usize] > self.threshold[context] {
                return true;
            }
        }
        false
    }

    pub fn claim(&mut self, context: usize) -> u32 {
        if context >= 2 {
            return 0;
        }
        let mut best_src: u32 = 0;
        let mut best_pri: u32 = 0;
        for src in 1u32..64 {
            let word = (src / 32) as usize;
            let bit = src % 32;
            let is_pending = (self.pending[word] >> bit) & 1 != 0;
            let is_enabled = (self.enable[context][word] >> bit) & 1 != 0;
            let pri = self.priority[src as usize];
            if is_pending && is_enabled && pri > self.threshold[context] && pri > best_pri {
                best_pri = pri;
                best_src = src;
            }
        }
        if best_src != 0 {
            let word = (best_src / 32) as usize;
            let bit = best_src % 32;
            self.pending[word] &= !(1 << bit);
            self.claimed[context] = best_src;
        }
        best_src
    }

    pub fn complete(&mut self, _context: usize, _source: u32) {
        // Acknowledge completion. Could re-enable for future claims.
    }

    pub fn read_u32(&mut self, offset: u64) -> u32 {
        if offset < PLIC_PRIORITY_BASE + 64 * 4 {
            let idx = ((offset - PLIC_PRIORITY_BASE) / 4) as usize;
            return self.priority[idx];
        }
        if offset >= PLIC_PENDING_BASE && offset < PLIC_PENDING_BASE + 8 {
            let idx = ((offset - PLIC_PENDING_BASE) / 4) as usize;
            if idx < 2 { return self.pending[idx]; }
            return 0;
        }
        for ctx in 0..2usize {
            let base = PLIC_ENABLE_BASE + PLIC_ENABLE_STRIDE * ctx as u64;
            if offset >= base && offset < base + 8 {
                let idx = ((offset - base) / 4) as usize;
                if idx < 2 { return self.enable[ctx][idx]; }
                return 0;
            }
        }
        for ctx in 0..2usize {
            let ctx_base = PLIC_CONTEXT_BASE + PLIC_CONTEXT_STRIDE * ctx as u64;
            if offset == ctx_base {
                return self.threshold[ctx];
            }
            if offset == ctx_base + 4 {
                return self.claim(ctx);
            }
        }
        0
    }

    pub fn write_u32(&mut self, offset: u64, val: u32) {
        if offset < PLIC_PRIORITY_BASE + 64 * 4 {
            let idx = ((offset - PLIC_PRIORITY_BASE) / 4) as usize;
            self.priority[idx] = val;
            return;
        }
        if offset >= PLIC_PENDING_BASE && offset < PLIC_PENDING_BASE + 8 {
            return; // read-only
        }
        for ctx in 0..2usize {
            let base = PLIC_ENABLE_BASE + PLIC_ENABLE_STRIDE * ctx as u64;
            if offset >= base && offset < base + 8 {
                let idx = ((offset - base) / 4) as usize;
                if idx < 2 { self.enable[ctx][idx] = val; }
                return;
            }
        }
        for ctx in 0..2usize {
            let ctx_base = PLIC_CONTEXT_BASE + PLIC_CONTEXT_STRIDE * ctx as u64;
            if offset == ctx_base {
                self.threshold[ctx] = val;
                return;
            }
            if offset == ctx_base + 4 {
                self.complete(ctx, val);
                return;
            }
        }
    }

    pub fn read_u8(&mut self, offset: u64) -> u8 {
        let aligned = offset & !0x3;
        let byte_idx = (offset & 0x3) as u32;
        let word = self.read_u32_no_side_effect(aligned);
        (word >> (byte_idx * 8)) as u8
    }

    pub fn write_u8(&mut self, offset: u64, val: u8) {
        let aligned = offset & !0x3;
        let byte_idx = (offset & 0x3) as u32;
        let old = self.read_u32_no_side_effect(aligned);
        let mask = !(0xFFu32 << (byte_idx * 8));
        let new = (old & mask) | ((val as u32) << (byte_idx * 8));
        self.write_u32(aligned, new);
    }

    fn read_u32_no_side_effect(&self, offset: u64) -> u32 {
        if offset < PLIC_PRIORITY_BASE + 64 * 4 {
            let idx = ((offset - PLIC_PRIORITY_BASE) / 4) as usize;
            return self.priority[idx];
        }
        if offset >= PLIC_PENDING_BASE && offset < PLIC_PENDING_BASE + 8 {
            let idx = ((offset - PLIC_PENDING_BASE) / 4) as usize;
            if idx < 2 { return self.pending[idx]; }
            return 0;
        }
        for ctx in 0..2usize {
            let base = PLIC_ENABLE_BASE + PLIC_ENABLE_STRIDE * ctx as u64;
            if offset >= base && offset < base + 8 {
                let idx = ((offset - base) / 4) as usize;
                if idx < 2 { return self.enable[ctx][idx]; }
                return 0;
            }
        }
        for ctx in 0..2usize {
            let ctx_base = PLIC_CONTEXT_BASE + PLIC_CONTEXT_STRIDE * ctx as u64;
            if offset == ctx_base {
                return self.threshold[ctx];
            }
            if offset == ctx_base + 4 {
                return self.claimed[ctx];
            }
        }
        0
    }
}

// ---------------------------------------------------------------------------
// MMIO Controller — dispatches to all devices
// ---------------------------------------------------------------------------

/// MMIO controller dispatching to concrete devices.
pub struct MmioController {
    pub uart: UartDevice,
    pub clint: ClintDevice,
    pub plic: PlicDevice,
    pub virtio_devices: Vec<VirtioBlockDevice>,
}

impl MmioController {
    pub fn new() -> Self {
        MmioController {
            uart: UartDevice::new(),
            clint: ClintDevice::new(),
            plic: PlicDevice::new(),
            virtio_devices: Vec::new(),
        }
    }

    pub fn with_uart(uart: UartDevice) -> Self {
        MmioController {
            uart,
            clint: ClintDevice::new(),
            plic: PlicDevice::new(),
            virtio_devices: Vec::new(),
        }
    }

    pub fn new_with_devices(
        uart: UartDevice,
        clint: ClintDevice,
        plic: PlicDevice,
        virtio_devices: Vec<VirtioBlockDevice>,
    ) -> Self {
        MmioController { uart, clint, plic, virtio_devices }
    }

    /// Read a byte from an MMIO address. Returns `None` if the address is not mapped.
    pub fn read_byte(&mut self, addr: u64) -> Option<u8> {
        if addr >= UART_BASE && addr < UART_BASE + UART_SIZE {
            return Some(self.uart.read_mut(addr - UART_BASE));
        }
        if addr >= CLINT_BASE && addr < CLINT_BASE + CLINT_SIZE {
            return Some(self.clint.read_u8(addr - CLINT_BASE));
        }
        if addr >= PLIC_BASE && addr < PLIC_BASE + PLIC_SIZE {
            return Some(self.plic.read_u8(addr - PLIC_BASE));
        }
        for (i, dev) in self.virtio_devices.iter().enumerate() {
            let base = virtio::VIRTIO_BASE + (i as u64) * virtio::VIRTIO_SIZE;
            if addr >= base && addr < base + virtio::VIRTIO_SIZE {
                return Some(dev.read_u8(addr - base));
            }
        }
        None
    }

    /// Write a byte to an MMIO address. Returns `true` if the address was handled.
    pub fn write_byte(&mut self, addr: u64, val: u8) -> bool {
        if addr >= UART_BASE && addr < UART_BASE + UART_SIZE {
            self.uart.write(addr - UART_BASE, val);
            return true;
        }
        if addr >= CLINT_BASE && addr < CLINT_BASE + CLINT_SIZE {
            self.clint.write_u8(addr - CLINT_BASE, val);
            return true;
        }
        if addr >= PLIC_BASE && addr < PLIC_BASE + PLIC_SIZE {
            self.plic.write_u8(addr - PLIC_BASE, val);
            return true;
        }
        for (i, dev) in self.virtio_devices.iter_mut().enumerate() {
            let base = virtio::VIRTIO_BASE + (i as u64) * virtio::VIRTIO_SIZE;
            if addr >= base && addr < base + virtio::VIRTIO_SIZE {
                dev.write_u8(addr - base, val);
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uart_write() {
        let mut uart = UartDevice::new();
        uart.write(0, b'H');
        uart.write(0, b'i');
        assert_eq!(uart.output, b"Hi");
    }

    #[test]
    fn test_uart_read_with_input() {
        let mut uart = UartDevice::with_input(b"AB".to_vec());
        assert_eq!(uart.read_mut(0), b'A');
        assert_eq!(uart.read_mut(0), b'B');
        assert_eq!(uart.read_mut(0), 0);
    }

    #[test]
    fn test_uart_read_no_input() {
        let mut uart = UartDevice::new();
        assert_eq!(uart.read_mut(0), 0);
    }

    #[test]
    fn test_uart_lsr_data_ready() {
        let uart = UartDevice::with_input(b"X".to_vec());
        let lsr = uart.read(5);
        assert_ne!(lsr & 1, 0, "data ready should be set");
        assert_ne!(lsr & (1 << 5), 0, "THR empty should be set");

        let empty = UartDevice::new();
        let lsr = empty.read(5);
        assert_eq!(lsr & 1, 0, "data ready should NOT be set");
        assert_ne!(lsr & (1 << 5), 0, "THR empty should be set");
    }

    #[test]
    fn test_uart_dlab_mode() {
        let mut uart = UartDevice::new();
        // Set DLAB bit in LCR
        uart.write(3, 0x83); // DLAB=1, 8-N-1
        assert!(uart.dlab());
        // Write to offset 0 should go to DLL, not THR
        uart.write(0, 0x0C);
        assert_eq!(uart.dll, 0x0C);
        assert!(uart.output.is_empty(), "should not output with DLAB set");
        // Write to offset 1 should go to DLM
        uart.write(1, 0x00);
        assert_eq!(uart.dlm, 0x00);
        // Read offset 0 returns DLL
        assert_eq!(uart.read(0), 0x0C);
        // Clear DLAB
        uart.write(3, 0x03);
        assert!(!uart.dlab());
        // Now write to offset 0 goes to THR
        uart.write(0, b'A');
        assert_eq!(uart.output, b"A");
    }

    #[test]
    fn test_uart_ier_write() {
        let mut uart = UartDevice::new();
        uart.write(1, 0x0F);
        assert_eq!(uart.ier, 0x0F);
        assert_eq!(uart.read(1), 0x0F);
    }

    #[test]
    fn test_uart_scratch_register() {
        let mut uart = UartDevice::new();
        uart.write(7, 0xAA);
        assert_eq!(uart.read(7), 0xAA);
    }

    #[test]
    fn test_mmio_controller_uart_write() {
        let mut ctrl = MmioController::new();
        assert!(ctrl.write_byte(UART_BASE, b'Z'));
        assert_eq!(ctrl.uart.output, vec![b'Z']);
    }

    #[test]
    fn test_mmio_controller_uart_read() {
        let uart = UartDevice::with_input(b"Q".to_vec());
        let mut ctrl = MmioController::with_uart(uart);
        assert_eq!(ctrl.read_byte(UART_BASE), Some(b'Q'));
        assert_eq!(ctrl.read_byte(UART_BASE), Some(0));
    }

    #[test]
    fn test_mmio_controller_unmapped() {
        let mut ctrl = MmioController::new();
        assert_eq!(ctrl.read_byte(0x2000_0000), None);
        assert!(!ctrl.write_byte(0x2000_0000, 0xFF));
    }

    #[test]
    fn test_mmio_controller_boundary() {
        let mut ctrl = MmioController::new();
        // Last valid UART address
        assert_eq!(ctrl.read_byte(UART_BASE + UART_SIZE - 1), Some(0));
        // First address past UART (not mapped to UART)
        assert_eq!(ctrl.read_byte(UART_BASE + UART_SIZE), None);
        // Address before CLINT base is unmapped
        assert_eq!(ctrl.read_byte(CLINT_BASE - 1), None);
        // UART_BASE - 1 is within PLIC range, so it IS mapped
        assert_eq!(ctrl.read_byte(UART_BASE - 1), Some(0));
    }
}

#[cfg(test)]
mod clint_tests {
    use super::*;

    #[test]
    fn test_clint_initial_state() {
        let c = ClintDevice::new();
        assert_eq!(c.mtime, 0);
        assert_eq!(c.mtimecmp, 0);
        assert_eq!(c.msip, 0);
        assert!(!c.msip_pending());
    }

    #[test]
    fn test_clint_mtime_increments() {
        let mut c = ClintDevice::new();
        c.mtimecmp = u64::MAX;
        c.tick();
        assert_eq!(c.mtime, 1);
        c.tick();
        assert_eq!(c.mtime, 2);
    }

    #[test]
    fn test_clint_tick_returns_false_below_cmp() {
        let mut c = ClintDevice::new();
        c.mtimecmp = 10;
        for _ in 0..9 {
            assert!(!c.tick());
        }
    }

    #[test]
    fn test_clint_timer_interrupt_fires() {
        let mut c = ClintDevice::new();
        c.mtimecmp = 3;
        assert!(!c.tick()); // mtime = 1
        assert!(!c.tick()); // mtime = 2
        assert!(c.tick());  // mtime = 3 >= mtimecmp
        assert!(c.tick());  // mtime = 4, still >=
    }

    #[test]
    fn test_clint_mtimecmp_write_changes_threshold() {
        let mut c = ClintDevice::new();
        c.mtimecmp = 2;
        c.tick();
        c.tick();
        assert!(c.mtime >= c.mtimecmp);
        c.mtimecmp = 100;
        assert!(c.mtime < c.mtimecmp);
    }

    #[test]
    fn test_clint_msip_read_write() {
        let mut c = ClintDevice::new();
        assert!(!c.msip_pending());
        c.write_u8(CLINT_MSIP, 1);
        assert!(c.msip_pending());
        assert_eq!(c.read_u8(CLINT_MSIP), 1);
        c.write_u8(CLINT_MSIP, 0);
        assert!(!c.msip_pending());
    }

    #[test]
    fn test_clint_register_byte_read() {
        let mut c = ClintDevice::new();
        c.mtime = 0x0807_0605_0403_0201;
        assert_eq!(c.read_u8(CLINT_MTIME + 0), 0x01);
        assert_eq!(c.read_u8(CLINT_MTIME + 1), 0x02);
        assert_eq!(c.read_u8(CLINT_MTIME + 7), 0x08);
    }

    #[test]
    fn test_clint_partial_byte_write() {
        let mut c = ClintDevice::new();
        let bytes: [u8; 8] = [0xBE, 0xBA, 0xFE, 0xCA, 0xEF, 0xBE, 0xAD, 0xDE];
        for (i, &b) in bytes.iter().enumerate() {
            c.write_u8(CLINT_MTIME + i as u64, b);
        }
        assert_eq!(c.mtime, 0xDEAD_BEEF_CAFE_BABE);
    }
}

#[cfg(test)]
mod plic_tests {
    use super::*;

    #[test]
    fn test_plic_initial_state() {
        let p = PlicDevice::new();
        assert_eq!(p.priority, [0u32; 64]);
        assert_eq!(p.pending, [0u32; 2]);
        assert_eq!(p.enable, [[0u32; 2]; 2]);
        assert_eq!(p.threshold, [0u32; 2]);
    }

    #[test]
    fn test_plic_priority_read_write() {
        let mut p = PlicDevice::new();
        p.write_u32(0x04, 7);
        assert_eq!(p.read_u32_no_side_effect(0x04), 7);
        assert_eq!(p.priority[1], 7);
    }

    #[test]
    fn test_plic_enable_read_write() {
        let mut p = PlicDevice::new();
        p.write_u32(PLIC_ENABLE_BASE, 0x0000_0002);
        assert_eq!(p.enable[0][0], 0x0000_0002);
        p.write_u32(PLIC_ENABLE_BASE + PLIC_ENABLE_STRIDE, 0x0000_0004);
        assert_eq!(p.enable[1][0], 0x0000_0004);
    }

    #[test]
    fn test_plic_threshold_read_write() {
        let mut p = PlicDevice::new();
        p.write_u32(PLIC_CONTEXT_BASE, 5);
        assert_eq!(p.threshold[0], 5);
        p.write_u32(PLIC_CONTEXT_BASE + PLIC_CONTEXT_STRIDE, 3);
        assert_eq!(p.threshold[1], 3);
    }

    #[test]
    fn test_plic_set_pending_and_has_pending() {
        let mut p = PlicDevice::new();
        p.priority[1] = 5;
        p.enable[0][0] = 0x0000_0002;
        p.threshold[0] = 0;
        p.set_pending(1);
        assert!(p.has_pending(0));
    }

    #[test]
    fn test_plic_has_pending_false_when_threshold_high() {
        let mut p = PlicDevice::new();
        p.priority[1] = 3;
        p.enable[0][0] = 0x0000_0002;
        p.threshold[0] = 5;
        p.set_pending(1);
        assert!(!p.has_pending(0));
    }

    #[test]
    fn test_plic_claim_returns_highest_priority() {
        let mut p = PlicDevice::new();
        p.priority[1] = 3;
        p.priority[2] = 7;
        p.enable[0][0] = 0x0000_0006;
        p.threshold[0] = 0;
        p.set_pending(1);
        p.set_pending(2);
        assert_eq!(p.claim(0), 2);
    }

    #[test]
    fn test_plic_claim_clears_pending() {
        let mut p = PlicDevice::new();
        p.priority[1] = 5;
        p.enable[0][0] = 0x0000_0002;
        p.threshold[0] = 0;
        p.set_pending(1);
        assert!(p.has_pending(0));
        assert_eq!(p.claim(0), 1);
        assert!(!p.has_pending(0));
    }

    #[test]
    fn test_plic_complete_does_not_crash() {
        let mut p = PlicDevice::new();
        p.complete(0, 1);
        p.complete(1, 63);
    }

    #[test]
    fn test_plic_context_isolation() {
        let mut p = PlicDevice::new();
        p.priority[1] = 5;
        p.enable[0][0] = 0;
        p.enable[1][0] = 0x0000_0002;
        p.threshold[0] = 0;
        p.threshold[1] = 0;
        p.set_pending(1);
        assert!(!p.has_pending(0));
        assert!(p.has_pending(1));
        assert_eq!(p.claim(0), 0);
    }
}
