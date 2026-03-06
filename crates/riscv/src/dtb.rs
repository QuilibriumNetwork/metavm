use std::collections::HashMap;

/// Configuration for DTB generation.
pub struct DtbConfig {
    pub memory_base: u64,
    pub memory_size: u64,
    pub bootargs: String,
    pub initrd_start: Option<u64>,
    pub initrd_end: Option<u64>,
}

// FDT structure tokens.
const FDT_BEGIN_NODE: u32 = 0x0000_0001;
const FDT_END_NODE: u32 = 0x0000_0002;
const FDT_PROP: u32 = 0x0000_0003;
const FDT_END: u32 = 0x0000_0009;

// FDT header constants.
const FDT_MAGIC: u32 = 0xD00D_FEED;
const FDT_VERSION: u32 = 17;
const FDT_LAST_COMP_VERSION: u32 = 16;
const FDT_HEADER_SIZE: u32 = 40;

/// Builder for constructing an FDT (Flattened Device Tree) binary blob.
struct FdtBuilder {
    struct_buf: Vec<u8>,
    strings_buf: Vec<u8>,
    string_offsets: HashMap<String, u32>,
}

impl FdtBuilder {
    fn new() -> Self {
        Self {
            struct_buf: Vec::new(),
            strings_buf: Vec::new(),
            string_offsets: HashMap::new(),
        }
    }

    /// Write FDT_BEGIN_NODE followed by the null-terminated node name, padded to 4 bytes.
    fn begin_node(&mut self, name: &str) {
        self.struct_buf.extend_from_slice(&FDT_BEGIN_NODE.to_be_bytes());
        self.struct_buf.extend_from_slice(name.as_bytes());
        self.struct_buf.push(0); // null terminator
        self.pad_struct_to_4();
    }

    /// Write FDT_END_NODE.
    fn end_node(&mut self) {
        self.struct_buf.extend_from_slice(&FDT_END_NODE.to_be_bytes());
    }

    /// Write a property with a u32 value (single cell).
    fn prop_u32(&mut self, name: &str, val: u32) {
        self.prop_cells(name, &[val]);
    }

    /// Write a property with a u64 value encoded as two u32 cells (high, low).
    fn prop_u64(&mut self, name: &str, val: u64) {
        let high = (val >> 32) as u32;
        let low = val as u32;
        self.prop_cells(name, &[high, low]);
    }

    /// Write a property with a null-terminated string value.
    fn prop_string(&mut self, name: &str, val: &str) {
        let nameoff = self.add_string(name);
        let mut value_bytes = val.as_bytes().to_vec();
        value_bytes.push(0); // null terminator
        let len = value_bytes.len() as u32;

        self.struct_buf.extend_from_slice(&FDT_PROP.to_be_bytes());
        self.struct_buf.extend_from_slice(&len.to_be_bytes());
        self.struct_buf.extend_from_slice(&nameoff.to_be_bytes());
        self.struct_buf.extend_from_slice(&value_bytes);
        self.pad_struct_to_4();
    }

    /// Write a property with an array of u32 cells.
    fn prop_cells(&mut self, name: &str, cells: &[u32]) {
        let nameoff = self.add_string(name);
        let len = (cells.len() * 4) as u32;

        self.struct_buf.extend_from_slice(&FDT_PROP.to_be_bytes());
        self.struct_buf.extend_from_slice(&len.to_be_bytes());
        self.struct_buf.extend_from_slice(&nameoff.to_be_bytes());
        for &cell in cells {
            self.struct_buf.extend_from_slice(&cell.to_be_bytes());
        }
        self.pad_struct_to_4();
    }

    /// Write a property with zero-length value (boolean/empty property).
    fn prop_empty(&mut self, name: &str) {
        let nameoff = self.add_string(name);
        let len: u32 = 0;

        self.struct_buf.extend_from_slice(&FDT_PROP.to_be_bytes());
        self.struct_buf.extend_from_slice(&len.to_be_bytes());
        self.struct_buf.extend_from_slice(&nameoff.to_be_bytes());
        // No value bytes, no padding needed (already aligned).
    }

    /// Add a property name string to the strings block. Returns the offset.
    /// Deduplicates strings that have already been added.
    fn add_string(&mut self, name: &str) -> u32 {
        if let Some(&offset) = self.string_offsets.get(name) {
            return offset;
        }
        let offset = self.strings_buf.len() as u32;
        self.strings_buf.extend_from_slice(name.as_bytes());
        self.strings_buf.push(0); // null terminator
        self.string_offsets.insert(name.to_string(), offset);
        offset
    }

    /// Pad the structure buffer to a 4-byte boundary.
    fn pad_struct_to_4(&mut self) {
        let remainder = self.struct_buf.len() % 4;
        if remainder != 0 {
            let padding = 4 - remainder;
            for _ in 0..padding {
                self.struct_buf.push(0);
            }
        }
    }

    /// Assemble the final DTB: header + memory reservation block + struct block + strings block.
    fn finish(mut self) -> Vec<u8> {
        // Write FDT_END token to terminate the structure block.
        self.struct_buf.extend_from_slice(&FDT_END.to_be_bytes());

        // Memory reservation block: one empty entry (two u64 zeros = 16 bytes).
        let mem_rsvmap_size: u32 = 16;

        let off_mem_rsvmap = FDT_HEADER_SIZE;
        let off_dt_struct = off_mem_rsvmap + mem_rsvmap_size;
        let off_dt_strings = off_dt_struct + self.struct_buf.len() as u32;
        let totalsize = off_dt_strings + self.strings_buf.len() as u32;

        let mut blob = Vec::with_capacity(totalsize as usize);

        // Header (40 bytes).
        blob.extend_from_slice(&FDT_MAGIC.to_be_bytes());
        blob.extend_from_slice(&totalsize.to_be_bytes());
        blob.extend_from_slice(&off_dt_struct.to_be_bytes());
        blob.extend_from_slice(&off_dt_strings.to_be_bytes());
        blob.extend_from_slice(&off_mem_rsvmap.to_be_bytes());
        blob.extend_from_slice(&FDT_VERSION.to_be_bytes());
        blob.extend_from_slice(&FDT_LAST_COMP_VERSION.to_be_bytes());
        blob.extend_from_slice(&0u32.to_be_bytes()); // boot_cpuid_phys
        blob.extend_from_slice(&(self.strings_buf.len() as u32).to_be_bytes());
        blob.extend_from_slice(&(self.struct_buf.len() as u32).to_be_bytes());

        // Memory reservation block (one empty entry: 16 zero bytes).
        blob.extend_from_slice(&0u64.to_be_bytes());
        blob.extend_from_slice(&0u64.to_be_bytes());

        // Structure block.
        blob.extend_from_slice(&self.struct_buf);

        // Strings block.
        blob.extend_from_slice(&self.strings_buf);

        blob
    }
}

/// Generate a DTB (Device Tree Blob) for a RISC-V virtual machine.
///
/// The generated device tree describes a minimal RISC-V "virt" machine with:
/// - A single RV64IMAC CPU with SV39 MMU
/// - CLINT (Core Local Interruptor) at 0x2000000
/// - PLIC (Platform-Level Interrupt Controller) at 0xC000000
/// - NS16550A UART at 0x10000000
/// - VirtIO MMIO device at 0x10001000
/// - Memory region as specified in config
pub fn generate_dtb(config: &DtbConfig) -> Vec<u8> {
    let mut fdt = FdtBuilder::new();

    // Root node.
    fdt.begin_node("");
    fdt.prop_u32("#address-cells", 2);
    fdt.prop_u32("#size-cells", 2);
    fdt.prop_string("compatible", "riscv-virtio");
    fdt.prop_string("model", "riscv-virtio,qemu");

    // /chosen
    fdt.begin_node("chosen");
    fdt.prop_string("bootargs", &config.bootargs);
    fdt.prop_string("stdout-path", "/soc/uart@10000000");
    if let (Some(start), Some(end)) = (config.initrd_start, config.initrd_end) {
        fdt.prop_u64("linux,initrd-start", start);
        fdt.prop_u64("linux,initrd-end", end);
    }
    fdt.end_node(); // chosen

    // /cpus
    fdt.begin_node("cpus");
    fdt.prop_u32("#address-cells", 1);
    fdt.prop_u32("#size-cells", 0);
    fdt.prop_u32("timebase-frequency", 10_000_000);

    // /cpus/cpu@0
    fdt.begin_node("cpu@0");
    fdt.prop_string("device_type", "cpu");
    fdt.prop_u32("reg", 0);
    fdt.prop_string("compatible", "riscv");
    fdt.prop_string("riscv,isa", "rv64imac");
    fdt.prop_string("mmu-type", "riscv,sv39");
    fdt.prop_string("status", "okay");

    // /cpus/cpu@0/interrupt-controller
    fdt.begin_node("interrupt-controller");
    fdt.prop_u32("#interrupt-cells", 1);
    fdt.prop_string("compatible", "riscv,cpu-intc");
    fdt.prop_empty("interrupt-controller");
    fdt.prop_u32("phandle", 1);
    fdt.end_node(); // interrupt-controller

    fdt.end_node(); // cpu@0
    fdt.end_node(); // cpus

    // /memory@<base_hex>
    let memory_node_name = format!("memory@{:x}", config.memory_base);
    fdt.begin_node(&memory_node_name);
    fdt.prop_string("device_type", "memory");
    // reg: address-cells=2, size-cells=2 => 4 cells total.
    fdt.prop_cells(
        "reg",
        &[
            (config.memory_base >> 32) as u32,
            config.memory_base as u32,
            (config.memory_size >> 32) as u32,
            config.memory_size as u32,
        ],
    );
    fdt.end_node(); // memory

    // /soc
    fdt.begin_node("soc");
    fdt.prop_u32("#address-cells", 2);
    fdt.prop_u32("#size-cells", 2);
    fdt.prop_string("compatible", "simple-bus");
    fdt.prop_empty("ranges");

    // /soc/clint@2000000
    fdt.begin_node("clint@2000000");
    fdt.prop_string("compatible", "riscv,clint0");
    fdt.prop_cells("reg", &[0x0, 0x0200_0000, 0x0, 0x0001_0000]);
    // interrupts-extended: phandle=1 cause=3 (MSI), phandle=1 cause=7 (MTI)
    fdt.prop_cells("interrupts-extended", &[1, 3, 1, 7]);
    fdt.end_node(); // clint

    // /soc/plic@c000000
    fdt.begin_node("plic@c000000");
    fdt.prop_string("compatible", "sifive,plic-1.0.0");
    fdt.prop_cells("reg", &[0x0, 0x0c00_0000, 0x0, 0x0400_0000]);
    fdt.prop_u32("#interrupt-cells", 1);
    fdt.prop_empty("interrupt-controller");
    // interrupts-extended: phandle=1 cause=11 (MEI), phandle=1 cause=9 (SEI)
    fdt.prop_cells("interrupts-extended", &[1, 11, 1, 9]);
    fdt.prop_u32("riscv,ndev", 63);
    fdt.prop_u32("phandle", 2);
    fdt.end_node(); // plic

    // /soc/uart@10000000
    // No "interrupts" property: serial8250 will use timer-based polling
    // instead of IRQ-driven mode.  Our UART emulation does not wire
    // interrupts to the PLIC, so omitting the property avoids the
    // serial8250 driver hanging during its interrupt self-test while
    // holding the console lock (which blocks all printk output).
    fdt.begin_node("uart@10000000");
    fdt.prop_string("compatible", "ns16550a");
    fdt.prop_cells("reg", &[0x0, 0x1000_0000, 0x0, 0x8]);
    fdt.prop_u32("clock-frequency", 3_686_400);
    fdt.end_node(); // uart

    // /soc/virtio_mmio@10001000
    fdt.begin_node("virtio_mmio@10001000");
    fdt.prop_string("compatible", "virtio,mmio");
    fdt.prop_cells("reg", &[0x0, 0x1000_1000, 0x0, 0x200]);
    fdt.prop_u32("interrupts", 1);
    fdt.prop_u32("interrupt-parent", 2);
    fdt.end_node(); // virtio_mmio

    fdt.end_node(); // soc
    fdt.end_node(); // root

    fdt.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtb_magic() {
        let config = DtbConfig {
            memory_base: 0x8000_0000,
            memory_size: 128 * 1024 * 1024,
            bootargs: "console=ttyS0".to_string(),
            initrd_start: None,
            initrd_end: None,
        };
        let dtb = generate_dtb(&config);
        // Check FDT magic (big-endian).
        assert_eq!(&dtb[0..4], &[0xD0, 0x0D, 0xFE, 0xED]);
        // Check version field at offset 20.
        assert_eq!(&dtb[20..24], &0x00000011u32.to_be_bytes());
        // Check last compatible version at offset 24.
        assert_eq!(&dtb[24..28], &0x00000010u32.to_be_bytes());
        // Check that totalsize in header matches actual blob size.
        let totalsize =
            u32::from_be_bytes([dtb[4], dtb[5], dtb[6], dtb[7]]) as usize;
        assert_eq!(totalsize, dtb.len());
    }

    #[test]
    fn test_dtb_contains_nodes() {
        let config = DtbConfig {
            memory_base: 0x8000_0000,
            memory_size: 128 * 1024 * 1024,
            bootargs: "console=ttyS0".to_string(),
            initrd_start: None,
            initrd_end: None,
        };
        let dtb = generate_dtb(&config);
        // The DTB should contain these strings somewhere in the blob.
        let dtb_str = String::from_utf8_lossy(&dtb);
        assert!(dtb_str.contains("riscv-virtio"));
        assert!(dtb_str.contains("console=ttyS0"));
        assert!(dtb_str.contains("uart@10000000"));
        assert!(dtb_str.contains("cpu@0"));
        assert!(dtb_str.contains("riscv,clint0"));
        assert!(dtb_str.contains("sifive,plic-1.0.0"));
        assert!(dtb_str.contains("ns16550a"));
        assert!(dtb_str.contains("virtio,mmio"));
        assert!(dtb_str.contains("memory@80000000"));
        assert!(dtb_str.contains("rv64imac"));
        // Verify initrd properties are absent when not configured.
        assert!(!dtb_str.contains("linux,initrd-start"));
    }

    #[test]
    fn test_dtb_with_initrd() {
        let config = DtbConfig {
            memory_base: 0x8000_0000,
            memory_size: 128 * 1024 * 1024,
            bootargs: "console=ttyS0".to_string(),
            initrd_start: Some(0x8300_0000),
            initrd_end: Some(0x8400_0000),
        };
        let dtb = generate_dtb(&config);
        // Check magic.
        assert_eq!(&dtb[0..4], &[0xD0, 0x0D, 0xFE, 0xED]);
        // Verify totalsize consistency.
        let totalsize =
            u32::from_be_bytes([dtb[4], dtb[5], dtb[6], dtb[7]]) as usize;
        assert_eq!(totalsize, dtb.len());
        // The DTB should now contain initrd property names in the strings block.
        let dtb_str = String::from_utf8_lossy(&dtb);
        assert!(dtb_str.contains("linux,initrd-start"));
        assert!(dtb_str.contains("linux,initrd-end"));
        // The blob should be larger than without initrd (two extra u64 properties).
        let config_no_initrd = DtbConfig {
            memory_base: 0x8000_0000,
            memory_size: 128 * 1024 * 1024,
            bootargs: "console=ttyS0".to_string(),
            initrd_start: None,
            initrd_end: None,
        };
        let dtb_no_initrd = generate_dtb(&config_no_initrd);
        assert!(dtb.len() > dtb_no_initrd.len());
    }
}
