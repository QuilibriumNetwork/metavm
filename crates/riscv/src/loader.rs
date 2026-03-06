use crate::cpu::CpuState;
use crate::csr::PrivilegeMode;
use crate::dtb::{self, DtbConfig};
use crate::memory::Memory;
use crate::mmio::{ClintDevice, MmioController, PlicDevice, UartDevice};
use crate::virtio::VirtioBlockDevice;
use crate::vm::Vm;
use goblin::elf;
use thiserror::Error;

const DEFAULT_STACK_TOP: u64 = 0x8040_0000;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("ELF parse error: {0}")]
    ParseError(String),
    #[error("wrong machine type: expected RISC-V (243), got {0}")]
    WrongMachine(u16),
    #[error("not a 64-bit ELF")]
    Not64Bit,
    #[error("not little-endian")]
    NotLittleEndian,
    #[error("segment out of bounds: offset={offset}, filesz={filesz}, file_len={file_len}")]
    SegmentOutOfBounds {
        offset: u64,
        filesz: u64,
        file_len: usize,
    },
}

#[derive(Debug)]
pub struct LoadedElf {
    pub entry: u64,
    pub memory: Memory,
    pub load_base: u64,
    pub load_end: u64,
}

/// Parse an ELF binary and load PT_LOAD segments into memory.
pub fn load_elf(data: &[u8]) -> Result<LoadedElf, LoadError> {
    let elf = elf::Elf::parse(data).map_err(|e| LoadError::ParseError(e.to_string()))?;

    // Validate ELF properties
    if !elf.is_64 {
        return Err(LoadError::Not64Bit);
    }
    if !elf.little_endian {
        return Err(LoadError::NotLittleEndian);
    }
    if elf.header.e_machine != elf::header::EM_RISCV {
        return Err(LoadError::WrongMachine(elf.header.e_machine));
    }

    let mut memory = Memory::new();
    let mut load_base = u64::MAX;
    let mut load_end = 0u64;

    for phdr in &elf.program_headers {
        if phdr.p_type != elf::program_header::PT_LOAD {
            continue;
        }

        let offset = phdr.p_offset as u64;
        let filesz = phdr.p_filesz as u64;

        // Validate segment bounds against file
        if filesz > 0 {
            let end = offset.checked_add(filesz).ok_or(LoadError::SegmentOutOfBounds {
                offset,
                filesz,
                file_len: data.len(),
            })?;
            if end as usize > data.len() {
                return Err(LoadError::SegmentOutOfBounds {
                    offset,
                    filesz,
                    file_len: data.len(),
                });
            }

            let segment_data = &data[offset as usize..(offset + filesz) as usize];
            memory.load_program(phdr.p_vaddr, segment_data);
        }

        // Track load range
        let seg_start = phdr.p_vaddr;
        let seg_end = phdr.p_vaddr.wrapping_add(phdr.p_memsz);
        if seg_start < load_base {
            load_base = seg_start;
        }
        if seg_end > load_end {
            load_end = seg_end;
        }
    }

    if load_base == u64::MAX {
        load_base = 0;
    }

    Ok(LoadedElf {
        entry: elf.entry,
        memory,
        load_base,
        load_end,
    })
}

/// Load an ELF binary into a ready-to-run VM.
pub fn load_elf_into_vm(data: &[u8], stack_top: Option<u64>) -> Result<Vm, LoadError> {
    let loaded = load_elf(data)?;
    let mut cpu = CpuState::with_pc(loaded.entry);
    cpu.write_reg(2, stack_top.unwrap_or(DEFAULT_STACK_TOP)); // SP = x2
    Ok(Vm::new_with_mmio(cpu, loaded.memory, MmioController::new()))
}

/// Configuration for booting a Linux kernel.
pub struct LinuxBootConfig {
    pub kernel_data: Vec<u8>,
    pub initrd_data: Option<Vec<u8>>,
    pub bootargs: String,
    pub memory_size: u64,
    pub disk_image: Option<Vec<u8>>,
}

const KERNEL_LOAD_ADDR: u64 = 0x8020_0000;
const DTB_LOAD_ADDR: u64 = 0x8200_0000;
const INITRD_LOAD_ADDR: u64 = 0x8300_0000;
const MEMORY_BASE: u64 = 0x8000_0000;

/// Set up a privileged VM configured for Linux boot.
///
/// Loads the kernel (ELF or raw image) at 0x8020_0000, optionally loads an
/// initrd at 0x8300_0000, generates a device tree blob at 0x8200_0000, and
/// initializes all devices (CLINT, PLIC, UART, VirtIO block).
///
/// The VM starts in Machine mode at the kernel entry point with:
/// - a0 = 0 (hart ID)
/// - a1 = DTB address (0x8200_0000)
pub fn setup_linux_boot(config: LinuxBootConfig) -> Result<Vm, LoadError> {
    let mut memory = Memory::new();

    // Try to load as ELF first; fall back to raw image.
    let entry = match elf::Elf::parse(&config.kernel_data) {
        Ok(elf) if elf.is_64 && elf.little_endian && elf.header.e_machine == elf::header::EM_RISCV => {
            for phdr in &elf.program_headers {
                if phdr.p_type != elf::program_header::PT_LOAD {
                    continue;
                }
                let offset = phdr.p_offset as usize;
                let filesz = phdr.p_filesz as usize;
                if filesz > 0 && offset + filesz <= config.kernel_data.len() {
                    memory.load_program(phdr.p_vaddr, &config.kernel_data[offset..offset + filesz]);
                }
            }
            elf.entry
        }
        _ => {
            // Raw image: load at KERNEL_LOAD_ADDR
            memory.load_program(KERNEL_LOAD_ADDR, &config.kernel_data);
            KERNEL_LOAD_ADDR
        }
    };

    // Load initrd if provided
    let (initrd_start, initrd_end) = if let Some(ref initrd) = config.initrd_data {
        memory.load_program(INITRD_LOAD_ADDR, initrd);
        (Some(INITRD_LOAD_ADDR), Some(INITRD_LOAD_ADDR + initrd.len() as u64))
    } else {
        (None, None)
    };

    // Generate and load DTB
    let dtb_config = DtbConfig {
        memory_base: MEMORY_BASE,
        memory_size: config.memory_size,
        bootargs: config.bootargs,
        initrd_start,
        initrd_end,
    };
    let dtb_data = dtb::generate_dtb(&dtb_config);
    memory.load_program(DTB_LOAD_ADDR, &dtb_data);

    // Set up CPU: M-mode, a0=hartid(0), a1=DTB address
    let mut cpu = CpuState::with_pc(entry);
    cpu.priv_mode = PrivilegeMode::Machine;
    cpu.write_reg(10, 0);            // a0 = hart ID
    cpu.write_reg(11, DTB_LOAD_ADDR); // a1 = DTB address

    // Set up devices
    let mut virtio_devices = Vec::new();
    if let Some(disk) = config.disk_image {
        virtio_devices.push(VirtioBlockDevice::new(disk));
    }

    let mmio = MmioController::new_with_devices(
        UartDevice::new(),
        ClintDevice::new(),
        PlicDevice::new(),
        virtio_devices,
    );

    let mut vm = Vm::new_with_mmio(cpu, memory, mmio);
    vm.privileged = true;
    Ok(vm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid ELF64 RISC-V binary in memory.
    /// Contains one PT_LOAD segment with the given code bytes at vaddr.
    fn make_elf(entry: u64, vaddr: u64, code: &[u8]) -> Vec<u8> {
        let ehdr_size = 64u16;
        let phdr_size = 56u16;
        let phdr_offset = ehdr_size as u64;
        let code_offset = (ehdr_size + phdr_size) as u64;
        let total_size = code_offset as usize + code.len();

        let mut buf = vec![0u8; total_size];

        // ELF magic
        buf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        buf[4] = 2; // ELFCLASS64
        buf[5] = 1; // ELFDATA2LSB
        buf[6] = 1; // EV_CURRENT
        buf[7] = 0; // ELFOSABI_NONE

        // e_type = ET_EXEC (2)
        buf[16..18].copy_from_slice(&2u16.to_le_bytes());
        // e_machine = EM_RISCV (243)
        buf[18..20].copy_from_slice(&243u16.to_le_bytes());
        // e_version
        buf[20..24].copy_from_slice(&1u32.to_le_bytes());
        // e_entry
        buf[24..32].copy_from_slice(&entry.to_le_bytes());
        // e_phoff
        buf[32..40].copy_from_slice(&phdr_offset.to_le_bytes());
        // e_shoff = 0
        // e_flags = 0
        // e_ehsize
        buf[52..54].copy_from_slice(&ehdr_size.to_le_bytes());
        // e_phentsize
        buf[54..56].copy_from_slice(&phdr_size.to_le_bytes());
        // e_phnum = 1
        buf[56..58].copy_from_slice(&1u16.to_le_bytes());
        // e_shentsize, e_shnum, e_shstrndx = 0 (no sections)
        buf[58..60].copy_from_slice(&0u16.to_le_bytes());
        buf[60..62].copy_from_slice(&0u16.to_le_bytes());
        buf[62..64].copy_from_slice(&0u16.to_le_bytes());

        // Program header (PT_LOAD)
        let ph = phdr_offset as usize;
        // p_type = PT_LOAD (1)
        buf[ph..ph + 4].copy_from_slice(&1u32.to_le_bytes());
        // p_flags = PF_R | PF_X (5)
        buf[ph + 4..ph + 8].copy_from_slice(&5u32.to_le_bytes());
        // p_offset
        buf[ph + 8..ph + 16].copy_from_slice(&code_offset.to_le_bytes());
        // p_vaddr
        buf[ph + 16..ph + 24].copy_from_slice(&vaddr.to_le_bytes());
        // p_paddr
        buf[ph + 24..ph + 32].copy_from_slice(&vaddr.to_le_bytes());
        // p_filesz
        buf[ph + 32..ph + 40].copy_from_slice(&(code.len() as u64).to_le_bytes());
        // p_memsz
        buf[ph + 40..ph + 48].copy_from_slice(&(code.len() as u64).to_le_bytes());
        // p_align
        buf[ph + 48..ph + 56].copy_from_slice(&0x1000u64.to_le_bytes());

        // Code bytes
        buf[code_offset as usize..].copy_from_slice(code);

        buf
    }

    #[test]
    fn test_load_elf_entry_point() {
        let entry = 0x8000_0000u64;
        // ADDI x1, x0, 42 = 0x02A00093
        let code = 0x02A00093u32.to_le_bytes();
        let elf_data = make_elf(entry, entry, &code);
        let loaded = load_elf(&elf_data).unwrap();
        assert_eq!(loaded.entry, entry);
    }

    #[test]
    fn test_load_elf_memory_contents() {
        let vaddr = 0x8000_0000u64;
        let insn = 0x02A00093u32; // ADDI x1, x0, 42
        let code = insn.to_le_bytes();
        let elf_data = make_elf(vaddr, vaddr, &code);
        let loaded = load_elf(&elf_data).unwrap();
        assert_eq!(loaded.memory.load_word(vaddr), insn);
        assert_eq!(loaded.load_base, vaddr);
        assert_eq!(loaded.load_end, vaddr + code.len() as u64);
    }

    #[test]
    fn test_load_elf_wrong_machine() {
        let mut elf_data = make_elf(0, 0, &[0; 4]);
        // Overwrite e_machine (bytes 18-19) with x86 (3)
        elf_data[18..20].copy_from_slice(&3u16.to_le_bytes());
        let err = load_elf(&elf_data).unwrap_err();
        assert!(matches!(err, LoadError::WrongMachine(3)));
    }

    #[test]
    fn test_load_elf_into_vm_sets_sp() {
        let entry = 0x8000_0000u64;
        let code = 0x02A00093u32.to_le_bytes();
        let elf_data = make_elf(entry, entry, &code);
        let vm = load_elf_into_vm(&elf_data, None).unwrap();
        assert_eq!(vm.cpu.pc, entry);
        assert_eq!(vm.cpu.read_reg(2), DEFAULT_STACK_TOP);
    }

    #[test]
    fn test_load_elf_custom_stack() {
        let entry = 0x8000_0000u64;
        let code = 0x02A00093u32.to_le_bytes();
        let elf_data = make_elf(entry, entry, &code);
        let custom_sp = 0x9000_0000u64;
        let vm = load_elf_into_vm(&elf_data, Some(custom_sp)).unwrap();
        assert_eq!(vm.cpu.read_reg(2), custom_sp);
    }

    #[test]
    fn test_setup_linux_boot_registers() {
        // Raw image: just an ADDI x1, x0, 42
        let kernel = 0x02A00093u32.to_le_bytes().to_vec();
        let config = LinuxBootConfig {
            kernel_data: kernel,
            initrd_data: None,
            bootargs: "console=ttyS0".to_string(),
            memory_size: 128 * 1024 * 1024,
            disk_image: None,
        };
        let vm = setup_linux_boot(config).unwrap();
        assert_eq!(vm.cpu.pc, KERNEL_LOAD_ADDR);
        assert_eq!(vm.cpu.read_reg(10), 0);                // a0 = hart ID
        assert_eq!(vm.cpu.read_reg(11), DTB_LOAD_ADDR);    // a1 = DTB
        assert_eq!(vm.cpu.priv_mode, PrivilegeMode::Machine);
        assert!(vm.privileged);
    }

    #[test]
    fn test_setup_linux_boot_dtb_at_expected_addr() {
        let kernel = 0x02A00093u32.to_le_bytes().to_vec();
        let config = LinuxBootConfig {
            kernel_data: kernel,
            initrd_data: None,
            bootargs: "console=ttyS0".to_string(),
            memory_size: 128 * 1024 * 1024,
            disk_image: None,
        };
        let vm = setup_linux_boot(config).unwrap();
        // DTB magic at DTB_LOAD_ADDR: 0xD00DFEED big-endian = bytes D0 0D FE ED
        let magic = vm.memory.load_word(DTB_LOAD_ADDR);
        // Loaded as little-endian u32: 0xEDFE0DD0
        assert_eq!(magic, 0xEDFE_0DD0);
    }

    #[test]
    fn test_setup_linux_boot_with_clint() {
        // Simple M-mode program: ADDI x1, x0, 1 then ECALL
        let kernel = vec![
            0x93, 0x00, 0x10, 0x00, // ADDI x1, x0, 1
            0x73, 0x00, 0x00, 0x00, // ECALL
        ];
        let config = LinuxBootConfig {
            kernel_data: kernel,
            initrd_data: None,
            bootargs: "".to_string(),
            memory_size: 128 * 1024 * 1024,
            disk_image: None,
        };
        let vm = setup_linux_boot(config).unwrap();
        // Verify CLINT is accessible: mtime should start at 0
        assert_eq!(vm.mmio.clint.mtime, 0);
    }
}
