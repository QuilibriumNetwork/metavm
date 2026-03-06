use metavm_riscv::loader::{setup_linux_boot, LinuxBootConfig};
use std::io::Write;
use std::time::Instant;
use std::{env, fs, process};

fn flush_uart(vm: &mut metavm_riscv::vm::Vm) -> u64 {
    if !vm.mmio.uart.output.is_empty() {
        let bytes: Vec<u8> = vm.mmio.uart.output.drain(..).collect();
        let count = bytes.len() as u64;
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        let _ = lock.write_all(&bytes);
        let _ = lock.flush();
        count
    } else {
        0
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: metavm-run <kernel> [initrd] [disk.img]");
        eprintln!();
        eprintln!("  kernel   - RISC-V ELF or raw image (e.g. OpenSBI fw_payload.elf)");
        eprintln!("  initrd   - Optional initial ramdisk");
        eprintln!("  disk.img - Optional VirtIO block device image");
        process::exit(1);
    }

    let kernel_path = &args[1];
    eprintln!("[metavm] Loading kernel: {}", kernel_path);
    let kernel_data = fs::read(kernel_path).unwrap_or_else(|e| {
        eprintln!("Failed to read kernel: {}", e);
        process::exit(1);
    });
    eprintln!("[metavm] Kernel size: {} bytes", kernel_data.len());

    let initrd_data = args.get(2).map(|p| {
        eprintln!("[metavm] Loading initrd: {}", p);
        fs::read(p).unwrap_or_else(|e| {
            eprintln!("Failed to read initrd: {}", e);
            process::exit(1);
        })
    });

    let disk_image = args.get(3).map(|p| {
        eprintln!("[metavm] Loading disk image: {}", p);
        fs::read(p).unwrap_or_else(|e| {
            eprintln!("Failed to read disk image: {}", e);
            process::exit(1);
        })
    });

    let config = LinuxBootConfig {
        kernel_data,
        initrd_data,
        bootargs: "console=ttyS0 earlycon=uart8250,mmio,0x10000000,115200n8 nosoftlockup nohz=off nosmp rdinit=/init".to_string(),
        memory_size: 128 * 1024 * 1024,
        disk_image,
    };

    let mut vm = setup_linux_boot(config).unwrap_or_else(|e| {
        eprintln!("Failed to set up VM: {}", e);
        process::exit(1);
    });

    eprintln!("[metavm] VM initialized, entry=0x{:x}, mode={:?}", vm.cpu.pc, vm.cpu.priv_mode);
    eprintln!("[metavm] Starting execution...");

    let batch_size = 1_000_000u64;
    let mut total_steps = 0u64;
    let mut total_uart_bytes = 0u64;
    let start = Instant::now();
    let mut last_report = start;

    loop {
        match vm.run(batch_size) {
            Ok(steps) => {
                total_steps += steps;
                total_uart_bytes += flush_uart(&mut vm);

                if vm.halted {
                    let elapsed = start.elapsed().as_secs_f64();
                    eprintln!();
                    eprintln!("[metavm] VM halted after {} instructions ({:.1}s, {:.1} MIPS)",
                        total_steps, elapsed, total_steps as f64 / elapsed / 1_000_000.0);
                    if let Some(code) = vm.exit_code {
                        eprintln!("[metavm] Exit code: {}", code);
                        process::exit(code as i32);
                    }
                    break;
                }

                let now = Instant::now();
                if now.duration_since(last_report).as_secs() >= 5 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let scause = vm.cpu.csrs.read_unchecked(0x142); // scause
                    let stval = vm.cpu.csrs.read_unchecked(0x143);  // stval
                    let sepc = vm.cpu.csrs.read_unchecked(0x141);   // sepc
                    let _stvec = vm.cpu.csrs.read_unchecked(0x105);  // stvec
                    let _satp = vm.cpu.csrs.read_unchecked(0x180);   // satp
                    let mtime = vm.mmio.clint.mtime;
                    let stimecmp = vm.cpu.csrs.read_unchecked(0x14D); // stimecmp
                    let mie = vm.cpu.csrs.read_unchecked(0x304); // mie
                    let mip = vm.cpu.csrs.read_unchecked(0x344); // mip
                    let mstatus = vm.cpu.csrs.read_unchecked(0x300); // mstatus
                    eprintln!("[metavm] {} insns ({:.1}s, {:.1} MIPS) PC=0x{:x} priv={:?} scause=0x{:x} sepc=0x{:x} uart={}B mtime={} stimecmp={} mie=0x{:x} mip=0x{:x} mstatus=0x{:x}",
                        total_steps, elapsed, total_steps as f64 / elapsed / 1_000_000.0,
                        vm.cpu.pc, vm.cpu.priv_mode, scause, sepc, total_uart_bytes, mtime, stimecmp, mie, mip, mstatus);
                    last_report = now;
                }
            }
            Err(e) => {
                let _ = flush_uart(&mut vm);
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!();
                eprintln!("[metavm] VM error after {} instructions ({:.1}s): {}", total_steps, elapsed, e);
                eprintln!("[metavm] PC=0x{:016x}  priv={:?}", vm.cpu.pc, vm.cpu.priv_mode);
                for i in 0..32 {
                    let val = vm.cpu.read_reg(i);
                    if val != 0 {
                        eprintln!("[metavm]   x{:02}=0x{:016x}", i, val);
                    }
                }
                process::exit(1);
            }
        }
    }
}
