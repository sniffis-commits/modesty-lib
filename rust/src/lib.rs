//! nsfw-lib-rust - Rust mirror of Python nsfw_scanner
//!
//! Provides:
//!   - lib.rs: C ABI exports for DLL/SO usage
//!   - scanner.rs: Image scanning logic (ONNX inference + rating)
//!   - preprocessing.rs: Image preprocessing (mirrors Python exactly)
//!   - config.rs: Configuration (thresholds, model paths)
//!   - model_download.rs: Model download from GitHub placeholder (feature: `download`)
//!   - native.rs: Native Messaging protocol (stdin/stdout) (feature: `download`)
//!
//! Feature `onnx` enables ONNX Runtime model inference (requires C compiler).
//! Feature `download` enables model download + Native Messaging (requires `onnx`).
//! Without `onnx`, all scans return rating: None (not assessed).

pub mod scanner;
pub mod preprocessing;
pub mod config;

#[cfg(feature = "download")]
pub mod model_download;

#[cfg(feature = "download")]
pub mod native;

use std::os::raw::c_char;
use std::ffi::CStr;
use std::ffi::CString;

/// C ABI: Create a new scanner session from model directory.
/// Returns an opaque pointer (must be freed with nsfw_lib_rust_session_free).
#[no_mangle]
pub extern "C" fn nsfw_lib_rust_session_create(model_dir: *const c_char) -> *mut scanner::ScannerSession {
    let model_dir = if model_dir.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(model_dir) }.to_string_lossy().into_owned()
    };

    match scanner::create_session(&model_dir) {
        Ok(session) => Box::into_raw(Box::new(session)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// C ABI: Scan an image file path. Returns JSON string (must be freed with nsfw_lib_rust_string_free).
/// JSON: {"rating": int, "verdict": str, "phrase": str}
#[no_mangle]
pub extern "C" fn nsfw_lib_rust_scan_file(
    session: *mut scanner::ScannerSession,
    image_path: *const c_char,
) -> *mut c_char {
    if session.is_null() || image_path.is_null() {
        return std::ptr::null_mut();
    }

    let image_path = unsafe { CStr::from_ptr(image_path) }.to_string_lossy().into_owned();
    let session = unsafe { &mut *session };

    match scanner::scan_image_file(session, &image_path) {
        Ok(result) => {
            let json = serde_json::to_string(&result).unwrap_or_default();
            match CString::new(json) {
                Ok(c_string) => c_string.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// C ABI: Scan raw image bytes (RGB, width, height). Returns JSON string.
#[no_mangle]
pub extern "C" fn nsfw_lib_rust_scan_bytes(
    session: *mut scanner::ScannerSession,
    width: i32,
    height: i32,
    bytes: *const u8,
    len: usize,
) -> *mut c_char {
    if session.is_null() || bytes.is_null() {
        return std::ptr::null_mut();
    }

    let session = unsafe { &mut *session };
    let bytes = unsafe { std::slice::from_raw_parts(bytes, len) };

    match scanner::scan_image_bytes(session, width, height, bytes) {
        Ok(result) => {
            let json = serde_json::to_string(&result).unwrap_or_default();
            match CString::new(json) {
                Ok(c_string) => c_string.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// C ABI: Free a string allocated by nsfw_lib_rust_scan_*.
#[no_mangle]
pub extern "C" fn nsfw_lib_rust_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// C ABI: Free a session allocated by nsfw_lib_rust_session_create.
#[no_mangle]
pub extern "C" fn nsfw_lib_rust_session_free(session: *mut scanner::ScannerSession) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session)) };
    }
}

/// C ABI: Get the library version string.
#[no_mangle]
pub extern "C" fn nsfw_lib_rust_version() -> *const c_char {
    CString::new("nsfw-lib-rust 0.1.0").unwrap().into_raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_not_null() {
        let version = nsfw_lib_rust_version();
        assert!(!version.is_null());
    }
}
