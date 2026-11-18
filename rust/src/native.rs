#[cfg(feature = "download")]
/// Native Messaging protocol for Chrome/Edge extension communication.
/// Reads messages from stdin (4-byte length prefix + JSON), writes to stdout.

use crate::config::ScannerConfig;
use crate::scanner::{scan_image_file, ScannerSession};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

// Message from extension to daemon
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum IncomingMessage {
    #[serde(rename = "scan-image")]
    ScanImage { image: String },
    #[serde(rename = "get-settings")]
    GetSettings,
    #[serde(rename = "ping")]
    Ping,
}

// Message from daemon to extension
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum OutgoingMessage {
    #[serde(rename = "scan-result")]
    ScanResult {
        rating: Option<i32>,
        verdict: Option<String>,
        phrase: Option<String>,
        error: Option<String>,
    },
    #[serde(rename = "settings")]
    Settings {
        threshold: i32,
        mode: String,
    },
    #[serde(rename = "pong")]
    Pong,
    #[serde(rename = "error")]
    Error { message: String },
}

/// Read a 4-byte little-endian length prefix, then read that many bytes as JSON.
fn read_message() -> Result<Option<Vec<u8>>, io::Error> {
    let mut len_buf = [0u8; 4];
    match io::stdin().read_exact(&mut len_buf) {
        Ok(_) => {}
        Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }

    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    io::stdin().read_exact(&mut buf)?;
    Ok(Some(buf))
}

/// Write a 4-byte length prefix + JSON payload to stdout.
fn write_message(payload: &[u8]) -> io::Result<()> {
    let len = payload.len() as u32;
    io::stdout().write_all(&len.to_le_bytes())?;
    io::stdout().write_all(payload)?;
    io::stdout().flush()
}

/// Main Native Messaging loop.
pub fn run_native_messaging(session: &mut ScannerSession, config: &ScannerConfig) -> io::Result<()> {
    loop {
        match read_message()? {
            Some(bytes) => {
                let incoming: IncomingMessage = match serde_json::from_slice(&bytes) {
                    Ok(m) => m,
                    Err(e) => {
                        let resp = OutgoingMessage::Error {
                            message: format!("Invalid JSON: {}", e),
                        };
                        let payload = serde_json::to_vec(&resp).unwrap();
                        write_message(&payload)?;
                        continue;
                    }
                };

                let response = match incoming {
                    IncomingMessage::ScanImage { image } => {
                        let result = scan_image_file(session, &image);
                        match result {
                            Ok(r) => OutgoingMessage::ScanResult {
                                rating: r.rating,
                                verdict: r.verdict,
                                phrase: r.phrase,
                                error: None,
                            },
                            Err(e) => OutgoingMessage::ScanResult {
                                rating: None,
                                verdict: None,
                                phrase: None,
                                error: Some(e.to_string()),
                            },
                        }
                    }
                    IncomingMessage::GetSettings => OutgoingMessage::Settings {
                        threshold: config.threshold_explicit as i32,
                        mode: "offline".to_string(),
                    },
                    IncomingMessage::Ping => OutgoingMessage::Pong,
                };

                let payload = serde_json::to_vec(&response).unwrap();
                write_message(&payload)?;
            }
            None => return Ok(()), // stdin closed
        }
    }
}