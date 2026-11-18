//! Cross-platform shared memory abstraction.

use std::io;

#[cfg(unix)]
use libc::{off_t, shm_open, shm_unlink, O_CREAT, O_RDWR};
#[cfg(unix)]
use memmap2::{MmapMut, MmapOptions};
#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};

#[cfg(windows)]
use std::{ffi::OsStr, os::windows::ffi::OsStrExt};
#[cfg(windows)]
use winapi::ctypes::c_void;
#[cfg(windows)]
use winapi::um::{
    handleapi::{CloseHandle, INVALID_HANDLE_VALUE},
    memoryapi::{
        CreateFileMappingW, FlushViewOfFile, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile,
        FILE_MAP_ALL_ACCESS,
    },
    winnt::{HANDLE, PAGE_READWRITE},
};

/// A mapped shared-memory region.
#[derive(Debug)]
pub struct SharedMemory {
    #[cfg(unix)]
    mmap: MmapMut,
    #[cfg(unix)]
    _file: File,
    #[cfg(windows)]
    handle: usize,
    #[cfg(windows)]
    view: usize,
    #[cfg(windows)]
    size: usize,
}

impl SharedMemory {
    /// Create or open a shared-memory region.
    pub fn create_or_open(name: &str, size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        validate_size(size)?;

        #[cfg(unix)]
        {
            let c_name = std::ffi::CString::new(format!("/{name}"))?;
            let fd = unsafe { shm_open(c_name.as_ptr(), O_CREAT | O_RDWR, 0o600) };
            if fd == -1 {
                return Err(io::Error::last_os_error().into());
            }

            let file = unsafe { File::from_raw_fd(fd) };
            if unsafe { libc::ftruncate(file.as_raw_fd(), size as off_t) } != 0 {
                return Err(io::Error::last_os_error().into());
            }

            let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };
            Ok(Self { mmap, _file: file })
        }

        #[cfg(windows)]
        {
            let name_wide = mapping_name(name);
            let handle = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null_mut(),
                    PAGE_READWRITE,
                    (size >> 32) as u32,
                    size as u32,
                    name_wide.as_ptr(),
                )
            };
            if handle.is_null() {
                return Err(io::Error::last_os_error().into());
            }

            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size) };
            if view.is_null() {
                unsafe { CloseHandle(handle) };
                return Err(io::Error::last_os_error().into());
            }

            Ok(Self {
                handle: handle as usize,
                view: view as usize,
                size,
            })
        }
    }

    /// Open an existing shared-memory region.
    pub fn open(name: &str, size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        validate_size(size)?;

        #[cfg(unix)]
        {
            let c_name = std::ffi::CString::new(format!("/{name}"))?;
            let fd = unsafe { shm_open(c_name.as_ptr(), O_RDWR, 0o600) };
            if fd == -1 {
                return Err(io::Error::last_os_error().into());
            }

            let file = unsafe { File::from_raw_fd(fd) };
            let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file)? };
            Ok(Self { mmap, _file: file })
        }

        #[cfg(windows)]
        {
            let name_wide = mapping_name(name);
            let handle = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, name_wide.as_ptr()) };
            if handle.is_null() {
                return Err(io::Error::last_os_error().into());
            }

            let view = unsafe { MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size) };
            if view.is_null() {
                unsafe { CloseHandle(handle) };
                return Err(io::Error::last_os_error().into());
            }

            Ok(Self {
                handle: handle as usize,
                view: view as usize,
                size,
            })
        }
    }

    /// Get mutable access to the shared-memory bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        #[cfg(unix)]
        {
            &mut self.mmap
        }

        #[cfg(windows)]
        unsafe {
            std::slice::from_raw_parts_mut(self.view as *mut u8, self.size)
        }
    }

    /// Get immutable access to the shared-memory bytes.
    pub fn as_slice(&self) -> &[u8] {
        #[cfg(unix)]
        {
            &self.mmap
        }

        #[cfg(windows)]
        unsafe {
            std::slice::from_raw_parts(self.view as *const u8, self.size)
        }
    }

    /// Flush mapped changes.
    pub fn flush(&mut self) -> Result<(), io::Error> {
        #[cfg(unix)]
        {
            self.mmap.flush()
        }

        #[cfg(windows)]
        {
            if unsafe { FlushViewOfFile(self.view as *const c_void, self.size) } == 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        }
    }
}

#[cfg(windows)]
impl Drop for SharedMemory {
    fn drop(&mut self) {
        unsafe {
            UnmapViewOfFile(self.view as *const c_void);
            CloseHandle(self.handle as HANDLE);
        }
    }
}

fn validate_size(size: usize) -> Result<(), io::Error> {
    if size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shared-memory size must be greater than zero",
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn mapping_name(name: &str) -> Vec<u16> {
    OsStr::new(&format!("Local\\{name}"))
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

/// Remove a shared-memory region by name where the platform supports it.
pub fn cleanup(name: &str) {
    #[cfg(unix)]
    {
        if let Ok(c_name) = std::ffi::CString::new(format!("/{name}")) {
            unsafe { shm_unlink(c_name.as_ptr()) };
        }
    }

    #[cfg(windows)]
    {
        let _ = name;
    }
}
