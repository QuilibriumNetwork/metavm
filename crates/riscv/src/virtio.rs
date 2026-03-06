// VirtIO MMIO block device implementation (transport v2, device_id=2).

use crate::memory::Memory;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const VIRTIO_BASE: u64 = 0x1000_1000;
pub const VIRTIO_SIZE: u64 = 0x200;
pub const VIRTIO_MAGIC: u32 = 0x7472_6976; // "virt"
pub const VIRTIO_VERSION: u32 = 2;
pub const VIRTIO_DEVICE_ID_BLOCK: u32 = 2;
pub const VIRTIO_VENDOR_ID: u32 = 0x554D_4551; // "QEMU"

const QUEUE_NUM_MAX: u32 = 256;

// VirtIO block request types
const VIRTIO_BLK_T_IN: u32 = 0; // read
const VIRTIO_BLK_T_OUT: u32 = 1; // write

// Descriptor flags
const VIRTQ_DESC_F_NEXT: u16 = 1;
const _VIRTQ_DESC_F_WRITE: u16 = 2;

// Block request status
const VIRTIO_BLK_S_OK: u8 = 0;
const VIRTIO_BLK_S_IOERR: u8 = 1;

const SECTOR_SIZE: u64 = 512;

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

pub struct VirtioBlockDevice {
    /// Backing storage.
    pub disk: Vec<u8>,

    // MMIO registers
    status: u32,
    device_features_sel: u32,
    driver_features: [u32; 2],
    driver_features_sel: u32,
    queue_sel: u32,
    queue_num: u32,
    queue_ready: u32,
    queue_desc: u64,
    queue_driver: u64, // avail ring base
    queue_device: u64, // used ring base
    interrupt_status: u32,

    // Internal state
    last_avail_idx: u16,
    notify_pending: bool,
}

impl VirtioBlockDevice {
    /// Create a new block device backed by `disk`.
    pub fn new(disk: Vec<u8>) -> Self {
        Self {
            disk,
            status: 0,
            device_features_sel: 0,
            driver_features: [0; 2],
            driver_features_sel: 0,
            queue_sel: 0,
            queue_num: 0,
            queue_ready: 0,
            queue_desc: 0,
            queue_driver: 0,
            queue_device: 0,
            interrupt_status: 0,
            last_avail_idx: 0,
            notify_pending: false,
        }
    }

    // -- MMIO register access ------------------------------------------------

    /// Read a 32-bit MMIO register at `offset` (relative to VIRTIO_BASE).
    pub fn read_u32(&self, offset: u64) -> u32 {
        match offset {
            0x000 => VIRTIO_MAGIC,
            0x004 => VIRTIO_VERSION,
            0x008 => VIRTIO_DEVICE_ID_BLOCK,
            0x00C => VIRTIO_VENDOR_ID,
            0x010 => {
                // DeviceFeatures — selected by device_features_sel.
                // We advertise no feature bits.
                0
            }
            0x034 => QUEUE_NUM_MAX,
            0x044 => self.queue_ready,
            0x060 => self.interrupt_status,
            0x070 => self.status,
            0x0FC => 0, // ConfigGeneration
            // Device-specific config: capacity as u64 LE starting at 0x100.
            0x100 => {
                let capacity = self.disk.len() as u64 / SECTOR_SIZE;
                capacity as u32
            }
            0x104 => {
                let capacity = self.disk.len() as u64 / SECTOR_SIZE;
                (capacity >> 32) as u32
            }
            _ => 0,
        }
    }

    /// Write a 32-bit value to MMIO register at `offset`.
    pub fn write_u32(&mut self, offset: u64, val: u32) {
        match offset {
            0x014 => self.device_features_sel = val,
            0x020 => {
                let sel = self.driver_features_sel as usize;
                if sel < 2 {
                    self.driver_features[sel] = val;
                }
            }
            0x024 => self.driver_features_sel = val,
            0x030 => self.queue_sel = val,
            0x038 => self.queue_num = val,
            0x044 => self.queue_ready = val,
            0x050 => {
                // QueueNotify — flag for processing
                self.notify_pending = true;
            }
            0x064 => {
                // InterruptACK — clear acknowledged bits
                self.interrupt_status &= !val;
            }
            0x070 => self.status = val,
            0x080 => {
                // QueueDescLow
                self.queue_desc = (self.queue_desc & 0xFFFF_FFFF_0000_0000) | val as u64;
            }
            0x084 => {
                // QueueDescHigh
                self.queue_desc = (self.queue_desc & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
            }
            0x090 => {
                // QueueDriverLow (avail ring)
                self.queue_driver = (self.queue_driver & 0xFFFF_FFFF_0000_0000) | val as u64;
            }
            0x094 => {
                // QueueDriverHigh
                self.queue_driver =
                    (self.queue_driver & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
            }
            0x0A0 => {
                // QueueDeviceLow (used ring)
                self.queue_device = (self.queue_device & 0xFFFF_FFFF_0000_0000) | val as u64;
            }
            0x0A4 => {
                // QueueDeviceHigh
                self.queue_device =
                    (self.queue_device & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
            }
            _ => {} // ignore unknown writes
        }
    }

    /// Byte-level read — extracts the appropriate byte from the aligned u32.
    pub fn read_u8(&self, offset: u64) -> u8 {
        let aligned = offset & !0x3;
        let shift = (offset & 0x3) * 8;
        (self.read_u32(aligned) >> shift) as u8
    }

    /// Byte-level write — read-modify-write on the aligned u32.
    pub fn write_u8(&mut self, offset: u64, val: u8) {
        let aligned = offset & !0x3;
        let shift = (offset & 0x3) * 8;
        let old = self.read_u32(aligned);
        let mask = !(0xFFu32 << shift);
        let new = (old & mask) | ((val as u32) << shift);
        self.write_u32(aligned, new);
    }

    // -- Queue processing ----------------------------------------------------

    /// Returns `true` if a QueueNotify write was received and not yet handled.
    pub fn needs_processing(&self) -> bool {
        self.notify_pending
    }

    /// Process all pending entries in the virtqueue.
    ///
    /// Walks the available ring, follows each descriptor chain (header, data,
    /// status), performs the requested block I/O, and posts results in the used
    /// ring.
    pub fn process_queue(&mut self, memory: &mut Memory) {
        self.notify_pending = false;

        if self.queue_ready == 0 || self.queue_num == 0 {
            return;
        }

        // Read current available ring index.
        let avail_idx = memory.load_half(self.queue_driver + 2);

        while self.last_avail_idx != avail_idx {
            let ring_slot =
                (self.last_avail_idx as u32 % self.queue_num) as u64;
            let head_idx =
                memory.load_half(self.queue_driver + 4 + 2 * ring_slot) as u32;

            // --- Descriptor 0: block request header (16 bytes) --------------
            let desc0_addr = self.queue_desc + (head_idx as u64) * 16;
            let header_addr = memory.load_double(desc0_addr);
            let _header_len = memory.load_word(desc0_addr + 8);
            let desc0_flags = memory.load_half(desc0_addr + 12);
            let desc0_next = memory.load_half(desc0_addr + 14);

            let req_type = memory.load_word(header_addr);
            // reserved u32 at header_addr+4 (ignored)
            let sector = memory.load_double(header_addr + 8);

            // --- Descriptor 1: data buffer ----------------------------------
            let desc1_idx = if desc0_flags & VIRTQ_DESC_F_NEXT != 0 {
                desc0_next as u32
            } else {
                head_idx + 1
            };
            let desc1_addr = self.queue_desc + (desc1_idx as u64) * 16;
            let data_addr = memory.load_double(desc1_addr);
            let data_len = memory.load_word(desc1_addr + 8) as u64;
            let desc1_flags = memory.load_half(desc1_addr + 12);
            let desc1_next = memory.load_half(desc1_addr + 14);

            // --- Descriptor 2: status byte ----------------------------------
            let desc2_idx = if desc1_flags & VIRTQ_DESC_F_NEXT != 0 {
                desc1_next as u32
            } else {
                desc1_idx + 1
            };
            let desc2_addr = self.queue_desc + (desc2_idx as u64) * 16;
            let status_addr = memory.load_double(desc2_addr);

            // Perform I/O
            let disk_offset = sector * SECTOR_SIZE;
            let mut status_byte = VIRTIO_BLK_S_OK;
            let mut total_written: u32 = 0;

            match req_type {
                VIRTIO_BLK_T_IN => {
                    // Read from disk into memory.
                    if disk_offset + data_len > self.disk.len() as u64 {
                        status_byte = VIRTIO_BLK_S_IOERR;
                    } else {
                        for i in 0..data_len {
                            let b = self.disk[(disk_offset + i) as usize];
                            memory.store_byte(data_addr + i, b);
                        }
                        total_written = data_len as u32;
                    }
                }
                VIRTIO_BLK_T_OUT => {
                    // Write from memory into disk.
                    if disk_offset + data_len > self.disk.len() as u64 {
                        status_byte = VIRTIO_BLK_S_IOERR;
                    } else {
                        for i in 0..data_len {
                            let b = memory.load_byte(data_addr + i);
                            self.disk[(disk_offset + i) as usize] = b;
                        }
                        total_written = 0; // writes don't report bytes to driver
                    }
                }
                _ => {
                    status_byte = VIRTIO_BLK_S_IOERR;
                }
            }

            // Write status byte.
            memory.store_byte(status_addr, status_byte);

            // Update used ring.
            let used_idx = memory.load_half(self.queue_device + 2);
            let used_slot = (used_idx as u32 % self.queue_num) as u64;
            let used_elem_addr = self.queue_device + 4 + 8 * used_slot;
            memory.store_word(used_elem_addr, head_idx);
            memory.store_word(used_elem_addr + 4, total_written);
            memory.store_half(self.queue_device + 2, used_idx.wrapping_add(1));

            self.last_avail_idx = self.last_avail_idx.wrapping_add(1);
        }

        // Signal used buffer notification.
        if self.last_avail_idx != 0 || avail_idx != 0 {
            self.interrupt_status |= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;

    /// Helper: create a device with `n_sectors` sectors of zeroed storage.
    fn make_device(n_sectors: u64) -> VirtioBlockDevice {
        VirtioBlockDevice::new(vec![0u8; (n_sectors * SECTOR_SIZE) as usize])
    }

    // -- Register read tests -------------------------------------------------

    #[test]
    fn test_magic_version_device_id() {
        let dev = make_device(1);
        assert_eq!(dev.read_u32(0x000), VIRTIO_MAGIC);
        assert_eq!(dev.read_u32(0x004), VIRTIO_VERSION);
        assert_eq!(dev.read_u32(0x008), VIRTIO_DEVICE_ID_BLOCK);
    }

    #[test]
    fn test_vendor_id() {
        let dev = make_device(1);
        assert_eq!(dev.read_u32(0x00C), VIRTIO_VENDOR_ID);
    }

    #[test]
    fn test_status_read_write() {
        let mut dev = make_device(1);
        assert_eq!(dev.read_u32(0x070), 0);
        dev.write_u32(0x070, 0xF);
        assert_eq!(dev.read_u32(0x070), 0xF);
        dev.write_u32(0x070, 0);
        assert_eq!(dev.read_u32(0x070), 0);
    }

    #[test]
    fn test_queue_num_max() {
        let dev = make_device(1);
        assert_eq!(dev.read_u32(0x034), 256);
    }

    #[test]
    fn test_queue_config() {
        let mut dev = make_device(1);

        // Set queue_num
        dev.write_u32(0x038, 128);
        assert_eq!(dev.queue_num, 128);

        // Set queue_ready
        dev.write_u32(0x044, 1);
        assert_eq!(dev.read_u32(0x044), 1);

        // Set descriptor table address (split across low/high)
        dev.write_u32(0x080, 0xDEAD_0000);
        dev.write_u32(0x084, 0x0000_CAFE);
        assert_eq!(dev.queue_desc, 0x0000_CAFE_DEAD_0000);

        // Set avail ring address
        dev.write_u32(0x090, 0x1000_0000);
        dev.write_u32(0x094, 0x0000_0001);
        assert_eq!(dev.queue_driver, 0x0000_0001_1000_0000);

        // Set used ring address
        dev.write_u32(0x0A0, 0x2000_0000);
        dev.write_u32(0x0A4, 0x0000_0002);
        assert_eq!(dev.queue_device, 0x0000_0002_2000_0000);
    }

    #[test]
    fn test_interrupt_status_initial() {
        let dev = make_device(1);
        assert_eq!(dev.read_u32(0x060), 0);
    }

    #[test]
    fn test_interrupt_ack_clears() {
        let mut dev = make_device(1);
        dev.interrupt_status = 0x3;
        assert_eq!(dev.read_u32(0x060), 0x3);
        dev.write_u32(0x064, 0x1);
        assert_eq!(dev.read_u32(0x060), 0x2);
        dev.write_u32(0x064, 0x2);
        assert_eq!(dev.read_u32(0x060), 0x0);
    }

    #[test]
    fn test_config_generation() {
        let dev = make_device(1);
        assert_eq!(dev.read_u32(0x0FC), 0);
    }

    #[test]
    fn test_capacity_config() {
        // 8 sectors = 4096 bytes
        let dev = make_device(8);
        assert_eq!(dev.read_u32(0x100), 8);
        assert_eq!(dev.read_u32(0x104), 0);

        // Large disk: 0x1_0000_0000 sectors
        let dev2 = VirtioBlockDevice::new(vec![0u8; 0]); // 0 capacity
        assert_eq!(dev2.read_u32(0x100), 0);
    }

    #[test]
    fn test_needs_processing_after_notify() {
        let mut dev = make_device(1);
        assert!(!dev.needs_processing());
        dev.write_u32(0x050, 0); // QueueNotify
        assert!(dev.needs_processing());
    }

    // -- Descriptor chain helpers --------------------------------------------

    /// Base addresses used to lay out virtqueue structures in guest memory.
    const DESC_BASE: u64 = 0x8000_0000;
    const AVAIL_BASE: u64 = 0x8000_1000;
    const USED_BASE: u64 = 0x8000_2000;
    const HEADER_ADDR: u64 = 0x8000_3000;
    const DATA_ADDR: u64 = 0x8000_4000;
    const STATUS_ADDR: u64 = 0x8000_5000;

    /// Write a virtqueue descriptor into guest memory.
    fn write_desc(
        mem: &mut Memory,
        base: u64,
        idx: u16,
        addr: u64,
        len: u32,
        flags: u16,
        next: u16,
    ) {
        let off = base + (idx as u64) * 16;
        mem.store_double(off, addr);
        mem.store_word(off + 8, len);
        mem.store_half(off + 12, flags);
        mem.store_half(off + 14, next);
    }

    /// Write a block request header into guest memory.
    fn write_block_header(mem: &mut Memory, addr: u64, req_type: u32, sector: u64) {
        mem.store_word(addr, req_type);
        mem.store_word(addr + 4, 0); // reserved
        mem.store_double(addr + 8, sector);
    }

    /// Set up the device and queue addresses, returning the configured device.
    fn setup_device_and_queue(disk: Vec<u8>) -> VirtioBlockDevice {
        let mut dev = VirtioBlockDevice::new(disk);
        dev.write_u32(0x038, QUEUE_NUM_MAX); // queue_num
        dev.write_u32(0x044, 1); // queue_ready
        dev.write_u32(0x080, DESC_BASE as u32);
        dev.write_u32(0x084, (DESC_BASE >> 32) as u32);
        dev.write_u32(0x090, AVAIL_BASE as u32);
        dev.write_u32(0x094, (AVAIL_BASE >> 32) as u32);
        dev.write_u32(0x0A0, USED_BASE as u32);
        dev.write_u32(0x0A4, (USED_BASE >> 32) as u32);
        dev
    }

    #[test]
    fn test_block_read_via_descriptor_chain() {
        // Prepare a 1-sector disk with known data.
        let mut disk = vec![0u8; 512];
        for i in 0..512 {
            disk[i] = (i & 0xFF) as u8;
        }
        let mut dev = setup_device_and_queue(disk);
        let mut mem = Memory::new();

        // Build descriptor chain: header(0) -> data(1) -> status(2)
        write_desc(
            &mut mem,
            DESC_BASE,
            0,
            HEADER_ADDR,
            16,
            VIRTQ_DESC_F_NEXT,
            1,
        );
        write_desc(
            &mut mem,
            DESC_BASE,
            1,
            DATA_ADDR,
            512,
            VIRTQ_DESC_F_NEXT | _VIRTQ_DESC_F_WRITE,
            2,
        );
        write_desc(
            &mut mem,
            DESC_BASE,
            2,
            STATUS_ADDR,
            1,
            _VIRTQ_DESC_F_WRITE,
            0,
        );

        // Write block request header: type=read(0), sector=0
        write_block_header(&mut mem, HEADER_ADDR, VIRTIO_BLK_T_IN, 0);

        // Populate available ring: flags=0, idx=1, ring[0]=0
        mem.store_half(AVAIL_BASE, 0);
        mem.store_half(AVAIL_BASE + 2, 1);
        mem.store_half(AVAIL_BASE + 4, 0); // ring[0] = descriptor chain head

        // Used ring starts at idx=0
        mem.store_half(USED_BASE + 2, 0);

        // Notify and process
        dev.write_u32(0x050, 0);
        assert!(dev.needs_processing());
        dev.process_queue(&mut mem);
        assert!(!dev.needs_processing());

        // Verify data was read from disk into memory.
        for i in 0u64..512 {
            assert_eq!(
                mem.load_byte(DATA_ADDR + i),
                (i & 0xFF) as u8,
                "mismatch at byte {}",
                i
            );
        }

        // Status byte should be OK (0).
        assert_eq!(mem.load_byte(STATUS_ADDR), VIRTIO_BLK_S_OK);

        // Used ring idx should have advanced.
        assert_eq!(mem.load_half(USED_BASE + 2), 1);

        // Used ring elem: id=0, len=512
        assert_eq!(mem.load_word(USED_BASE + 4), 0);
        assert_eq!(mem.load_word(USED_BASE + 8), 512);

        // Interrupt should be raised.
        assert_eq!(dev.read_u32(0x060) & 1, 1);
    }

    #[test]
    fn test_block_write_via_descriptor_chain() {
        let disk = vec![0u8; 512];
        let mut dev = setup_device_and_queue(disk);
        let mut mem = Memory::new();

        // Build descriptor chain: header(0) -> data(1) -> status(2)
        write_desc(
            &mut mem,
            DESC_BASE,
            0,
            HEADER_ADDR,
            16,
            VIRTQ_DESC_F_NEXT,
            1,
        );
        write_desc(
            &mut mem,
            DESC_BASE,
            1,
            DATA_ADDR,
            512,
            VIRTQ_DESC_F_NEXT,
            2,
        );
        write_desc(
            &mut mem,
            DESC_BASE,
            2,
            STATUS_ADDR,
            1,
            _VIRTQ_DESC_F_WRITE,
            0,
        );

        // Write block request header: type=write(1), sector=0
        write_block_header(&mut mem, HEADER_ADDR, VIRTIO_BLK_T_OUT, 0);

        // Fill data buffer in memory with a pattern.
        for i in 0u64..512 {
            mem.store_byte(DATA_ADDR + i, 0xAB);
        }

        // Populate available ring
        mem.store_half(AVAIL_BASE, 0);
        mem.store_half(AVAIL_BASE + 2, 1);
        mem.store_half(AVAIL_BASE + 4, 0);

        mem.store_half(USED_BASE + 2, 0);

        // Notify and process
        dev.write_u32(0x050, 0);
        dev.process_queue(&mut mem);

        // Verify data was written to disk.
        for i in 0..512 {
            assert_eq!(dev.disk[i], 0xAB, "disk mismatch at byte {}", i);
        }

        // Status byte should be OK.
        assert_eq!(mem.load_byte(STATUS_ADDR), VIRTIO_BLK_S_OK);

        // Interrupt should be raised.
        assert_eq!(dev.read_u32(0x060) & 1, 1);
    }
}
