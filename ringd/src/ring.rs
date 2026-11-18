//! Ring buffer for frame transport
//!
//! Layout:
//! - Header (64 bytes): magic, version, capacity, slot_size, head, tail, flags
//! - Slots: each slot = 40 bytes metadata + payload
//!   Metadata: seq(u64), pts_us(u64), width(u32), height(u32), format(u32), stride(u32), payload_len(u32), flags(u32)

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Magic number for validation
const RING_MAGIC: u64 = 0x4E5346575F4D454D; // "NSFW_MEM"
const RING_VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;

#[repr(C)]
#[derive(Debug)]
pub struct RingHeader {
    pub magic: u64,
    pub version: u32,
    pub capacity: u32,
    pub slot_payload_size: u32,
    pub head: AtomicU64,
    pub tail: AtomicU64,
    pub flags: AtomicU32,
    pub _reserved: [u8; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlotMeta {
    pub seq: u64,
    pub pts_us: u64,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub stride: u32,
    pub payload_len: u32,
    pub flags: u32,
}

const SLOT_META_SIZE: usize = std::mem::size_of::<SlotMeta>();

impl SlotMeta {
    pub fn new() -> Self {
        SlotMeta {
            seq: 0,
            pts_us: 0,
            width: 0,
            height: 0,
            format: 0,
            stride: 0,
            payload_len: 0,
            flags: 0,
        }
    }
}

/// Flags for ring buffer
pub mod flags {
    pub const NONE: u32 = 0;
    pub const WRITER_DONE: u32 = 1 << 0; // Writer has finished
    pub const READER_WAITING: u32 = 1 << 1; // Reader waiting for data
    pub const ERROR: u32 = 1 << 31;
}

/// Frame data structure
#[derive(Debug, Clone)]
pub struct Frame {
    pub seq: u64,
    pub pts_us: u64,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub stride: u32,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn new(
        seq: u64,
        pts_us: u64,
        width: u32,
        height: u32,
        format: u32,
        stride: u32,
        payload: Vec<u8>,
    ) -> Self {
        Frame {
            seq,
            pts_us,
            width,
            height,
            format,
            stride,
            payload,
        }
    }
}

/// Ring buffer wrapper
pub struct RingBuffer {
    shm: crate::platform::SharedMemory,
    capacity: usize,
    slot_payload_size: usize,
    slot_total_size: usize,
}

impl RingBuffer {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn slot_payload_size(&self) -> usize {
        self.slot_payload_size
    }

    /// Create a new ring buffer
    pub fn create(
        name: &str,
        capacity: usize,
        max_payload: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if capacity < 2
            || capacity > u32::MAX as usize
            || max_payload == 0
            || max_payload > u32::MAX as usize
        {
            return Err(RingError::InvalidConfiguration.into());
        }

        let slot_total_size = SLOT_META_SIZE + max_payload;
        let total_size = capacity
            .checked_mul(slot_total_size)
            .and_then(|slots_size| HEADER_SIZE.checked_add(slots_size))
            .ok_or(RingError::InvalidConfiguration)?;

        let mut shm = crate::platform::SharedMemory::create_or_open(name, total_size)?;

        // Initialize header
        let header_ptr = shm.as_mut_slice().as_mut_ptr() as *mut RingHeader;
        unsafe {
            *header_ptr = RingHeader {
                magic: RING_MAGIC,
                version: RING_VERSION,
                capacity: capacity as u32,
                slot_payload_size: max_payload as u32,
                head: AtomicU64::new(0),
                tail: AtomicU64::new(0),
                flags: AtomicU32::new(flags::NONE),
                _reserved: [0; 16],
            };
        }

        // Initialize slots
        for i in 0..capacity {
            let slot_ptr = Self::slot_meta_ptr(&mut shm, i, slot_total_size);
            unsafe {
                *slot_ptr = SlotMeta::new();
            }
        }

        shm.flush()?;

        Ok(RingBuffer {
            shm,
            capacity,
            slot_payload_size: max_payload,
            slot_total_size,
        })
    }

    /// Open existing ring buffer
    pub fn open(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let shm = crate::platform::SharedMemory::open(name, HEADER_SIZE)?;

        // Read header to get capacity and slot size
        let header = Self::header_from(&shm);
        if header.magic != RING_MAGIC || header.version != RING_VERSION {
            return Err(RingError::InvalidHeader.into());
        }
        let capacity = header.capacity as usize;
        let slot_payload_size = header.slot_payload_size as usize;
        if capacity < 2 || slot_payload_size == 0 {
            return Err(RingError::InvalidHeader.into());
        }
        let slot_total_size = SLOT_META_SIZE + slot_payload_size;

        // Re-open with correct size
        let total_size = capacity
            .checked_mul(slot_total_size)
            .and_then(|slots_size| HEADER_SIZE.checked_add(slots_size))
            .ok_or(RingError::InvalidHeader)?;
        drop(shm);
        let shm = crate::platform::SharedMemory::open(name, total_size)?;

        Ok(RingBuffer {
            shm,
            capacity,
            slot_payload_size,
            slot_total_size,
        })
    }

    fn header(&self) -> &RingHeader {
        Self::header_from(&self.shm)
    }

    fn header_from(shm: &crate::platform::SharedMemory) -> &RingHeader {
        let ptr = shm.as_slice().as_ptr() as *const RingHeader;
        unsafe { &*ptr }
    }

    fn header_mut(&mut self) -> &mut RingHeader {
        let ptr = self.shm.as_mut_slice().as_mut_ptr() as *mut RingHeader;
        unsafe { &mut *ptr }
    }

    fn slot_meta_ptr(
        shm: &mut crate::platform::SharedMemory,
        index: usize,
        slot_total_size: usize,
    ) -> *mut SlotMeta {
        let offset = HEADER_SIZE + index * slot_total_size;
        unsafe { shm.as_mut_slice().as_mut_ptr().add(offset) as *mut SlotMeta }
    }

    fn slot_meta(&self, index: usize) -> &SlotMeta {
        let offset = HEADER_SIZE + index * self.slot_total_size;
        let ptr = unsafe { self.shm.as_slice().as_ptr().add(offset) as *const SlotMeta };
        unsafe { &*ptr }
    }

    fn slot_meta_mut(&mut self, index: usize) -> &mut SlotMeta {
        let offset = HEADER_SIZE + index * self.slot_total_size;
        let ptr = unsafe { self.shm.as_mut_slice().as_mut_ptr().add(offset) as *mut SlotMeta };
        unsafe { &mut *ptr }
    }

    fn slot_payload(&self, index: usize) -> &[u8] {
        let offset = HEADER_SIZE + index * self.slot_total_size + SLOT_META_SIZE;
        &self.shm.as_slice()[offset..offset + self.slot_payload_size]
    }

    fn slot_payload_mut(&mut self, index: usize) -> &mut [u8] {
        let offset = HEADER_SIZE + index * self.slot_total_size + SLOT_META_SIZE;
        &mut self.shm.as_mut_slice()[offset..offset + self.slot_payload_size]
    }

    /// Try to push a frame (non-blocking)
    pub fn try_push(&mut self, frame: &Frame) -> Result<(), RingError> {
        let (head, tail) = {
            let header = self.header();
            (
                header.head.load(Ordering::Acquire),
                header.tail.load(Ordering::Acquire),
            )
        };
        let next_tail = (tail + 1) % self.capacity as u64;

        if next_tail == head {
            return Err(RingError::Full);
        }

        let slot_idx = tail as usize;
        let payload = frame.payload.as_slice();
        if payload.len() > self.slot_payload_size {
            return Err(RingError::PayloadTooLarge);
        }

        // Write metadata
        let meta = self.slot_meta_mut(slot_idx);
        meta.seq = frame.seq;
        meta.pts_us = frame.pts_us;
        meta.width = frame.width;
        meta.height = frame.height;
        meta.format = frame.format;
        meta.stride = frame.stride;
        meta.payload_len = payload.len() as u32;
        meta.flags = 0;

        // Write payload
        let slot_payload = self.slot_payload_mut(slot_idx);
        slot_payload[..payload.len()].copy_from_slice(payload);

        // Update tail (release)
        self.header().tail.store(next_tail, Ordering::Release);
        self.shm.flush()?;

        Ok(())
    }

    /// Try to pop a frame (non-blocking)
    pub fn try_pop(&mut self) -> Result<Option<Frame>, RingError> {
        let (head, tail) = {
            let header = self.header();
            (
                header.head.load(Ordering::Acquire),
                header.tail.load(Ordering::Acquire),
            )
        };

        if head == tail {
            return Ok(None); // Empty
        }

        let slot_idx = head as usize;
        let frame = {
            let meta = self.slot_meta(slot_idx);
            let payload_len = meta.payload_len as usize;
            if payload_len == 0 || payload_len > self.slot_payload_size {
                return Err(RingError::Corrupted);
            }

            let mut payload = vec![0u8; payload_len];
            payload.copy_from_slice(&self.slot_payload(slot_idx)[..payload_len]);
            Frame::new(
                meta.seq,
                meta.pts_us,
                meta.width,
                meta.height,
                meta.format,
                meta.stride,
                payload,
            )
        };

        // Clear slot
        let meta_mut = self.slot_meta_mut(slot_idx);
        *meta_mut = SlotMeta::new();

        // Update head (release)
        let next_head = (head + 1) % self.capacity as u64;
        self.header().head.store(next_head, Ordering::Release);
        self.shm.flush()?;

        Ok(Some(frame))
    }

    /// Get available slots for writing
    pub fn available_write(&self) -> usize {
        let header = self.header();
        let head = header.head.load(Ordering::Acquire);
        let tail = header.tail.load(Ordering::Acquire);

        if tail >= head {
            self.capacity - (tail - head) as usize - 1
        } else {
            (head - tail) as usize - 1
        }
    }

    /// Get available frames for reading
    pub fn available_read(&self) -> usize {
        let header = self.header();
        let head = header.head.load(Ordering::Acquire);
        let tail = header.tail.load(Ordering::Acquire);

        if tail >= head {
            (tail - head) as usize
        } else {
            self.capacity - (head - tail) as usize
        }
    }

    /// Mark writer as done
    pub fn mark_writer_done(&mut self) -> Result<(), RingError> {
        let header = self.header_mut();
        header.flags.fetch_or(flags::WRITER_DONE, Ordering::Release);
        self.shm.flush()?;
        Ok(())
    }

    /// Check if writer is done
    pub fn is_writer_done(&self) -> bool {
        let header = self.header();
        header.flags.load(Ordering::Acquire) & flags::WRITER_DONE != 0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RingError {
    #[error("Ring buffer full")]
    Full,
    #[error("Payload too large for slot")]
    PayloadTooLarge,
    #[error("Corrupted slot")]
    Corrupted,
    #[error("Invalid ring-buffer header")]
    InvalidHeader,
    #[error("Invalid ring-buffer capacity or payload size")]
    InvalidConfiguration,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::{Frame, RingBuffer, RingHeader, SlotMeta, HEADER_SIZE, SLOT_META_SIZE};
    use std::mem::size_of;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn shared_ring_transfers_frames_between_open_mappings() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after UNIX epoch")
            .as_nanos();
        let name = format!("nsfw_memd_test_{}_{}", std::process::id(), timestamp);

        let result = (|| {
            let mut writer = RingBuffer::create(&name, 2, 1024)?;
            let mut reader = RingBuffer::open(&name)?;
            let frame = Frame::new(7, 42, 640, 480, 1, 1920, vec![1, 2, 3, 4]);

            assert_eq!(size_of::<RingHeader>(), HEADER_SIZE);
            assert_eq!(size_of::<SlotMeta>(), SLOT_META_SIZE);
            assert_eq!(writer.available_write(), 1);
            writer.try_push(&frame)?;
            assert_eq!(reader.available_read(), 1);
            assert_eq!(reader.slot_meta(0).payload_len, frame.payload.len() as u32);

            let received = reader.try_pop()?.expect("pushed frame should be available");
            assert_eq!(received.seq, frame.seq);
            assert_eq!(received.pts_us, frame.pts_us);
            assert_eq!(received.width, frame.width);
            assert_eq!(received.height, frame.height);
            assert_eq!(received.format, frame.format);
            assert_eq!(received.stride, frame.stride);
            assert_eq!(received.payload, frame.payload);
            assert_eq!(reader.available_read(), 0);

            Ok::<(), Box<dyn std::error::Error>>(())
        })();

        crate::platform::cleanup(&name);
        result.expect("shared ring transfer should succeed");
    }
}
