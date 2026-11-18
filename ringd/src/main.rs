//! nsfw-ringd - Modesty shared-memory ring daemon for buffered frame transport
//!
//! This daemon creates a shared-memory ring buffer that allows ffmpeg to write
//! decoded frames while the scanner reads them concurrently, eliminating the
//! pipe bottleneck.

use clap::{Parser, Subcommand};
use log::{debug, error, info};
use std::sync::Arc;
use std::time::Duration;

use nsfw_ringd::{platform, RingBuffer};

#[derive(Parser)]
#[command(
    name = "nsfw-ringd",
    version,
    about = "Modesty shared-memory ring daemon for buffered frame transport"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the daemon
    Start {
        /// Ring buffer name
        #[arg(short, long, default_value = "nsfw_ring")]
        name: String,

        /// Ring capacity (number of frames)
        #[arg(short, long, default_value = "100")]
        capacity: usize,

        /// Maximum payload size per frame (bytes)
        #[arg(short, long, default_value = "2000000")]
        max_payload: usize,

        /// Daemonize (run in background)
        #[arg(short, long)]
        daemon: bool,
    },

    /// Stop the daemon
    Stop {
        /// Ring buffer name
        #[arg(short, long, default_value = "nsfw_ring")]
        name: String,
    },

    /// Show ring buffer status
    Status {
        /// Ring buffer name
        #[arg(short, long, default_value = "nsfw_ring")]
        name: String,
    },

    /// Clean up shared memory
    Cleanup {
        /// Ring buffer name
        #[arg(short, long, default_value = "nsfw_ring")]
        name: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let cli = Cli::parse();

    match cli.command {
        Command::Start {
            name,
            capacity,
            max_payload,
            daemon,
        } => start_daemon(name, capacity, max_payload, daemon).await,
        Command::Stop { name } => stop_daemon(name).await,
        Command::Status { name } => show_status(name).await,
        Command::Cleanup { name } => cleanup_daemon(name).await,
    }
}

async fn start_daemon(
    name: String,
    capacity: usize,
    max_payload: usize,
    _daemon: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "Starting nsfw-memd daemon: name={}, capacity={}, max_payload={}",
        name, capacity, max_payload
    );

    // Create ring buffer
    let ring = RingBuffer::create(&name, capacity, max_payload)?;
    info!(
        "Ring buffer created: {} slots x {} bytes payload",
        capacity, max_payload
    );

    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;

    // Main loop - just keep the ring alive and monitor
    let ring_arc = Arc::new(tokio::sync::Mutex::new(ring));
    let ring_clone = ring_arc.clone();

    // Monitor task
    let monitor = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            let ring = ring_clone.lock().await;
            let avail_read = ring.available_read();
            let avail_write = ring.available_write();
            let writer_done = ring.is_writer_done();
            debug!(
                "Ring status: read={}, write={}, writer_done={}",
                avail_read, avail_write, writer_done
            );
        }
    });

    // Signal handling
    let signal_task = tokio::spawn(async move {
        #[cfg(unix)]
        tokio::select! {
            result = tokio::signal::ctrl_c() => match result {
                Ok(()) => info!("Received SIGINT, shutting down..."),
                Err(error) => {
                    error!("Failed to wait for SIGINT: {error}");
                    return;
                }
            },
            _ = sigterm.recv() => info!("Received SIGTERM, shutting down..."),
        }

        #[cfg(windows)]
        if let Err(error) = tokio::signal::ctrl_c().await {
            error!("Failed to wait for Ctrl-C: {error}");
            return;
        } else {
            info!("Received Ctrl-C, shutting down...");
        }

        let mut ring = ring_arc.lock().await;
        if let Err(error) = ring.mark_writer_done() {
            error!("Failed to mark the ring buffer as complete: {error}");
        }
    });

    // Wait for shutdown signal
    tokio::select! {
        _ = signal_task => {},
        _ = monitor => {},
    }

    info!("Daemon stopped");
    Ok(())
}

async fn stop_daemon(name: String) -> Result<(), Box<dyn std::error::Error>> {
    info!("Stopping daemon: {}", name);

    let mut ring = RingBuffer::open(&name)?;
    ring.mark_writer_done()?;
    info!("Writer done flag set");

    Ok(())
}

async fn show_status(name: String) -> Result<(), Box<dyn std::error::Error>> {
    let ring = RingBuffer::open(&name)?;
    let avail_read = ring.available_read();
    let avail_write = ring.available_write();
    let writer_done = ring.is_writer_done();
    let capacity = ring.capacity();

    println!("Ring buffer: {}", name);
    println!("  Capacity: {} slots", capacity);
    println!("  Available for read: {}", avail_read);
    println!("  Available for write: {}", avail_write);
    println!("  Writer done: {}", writer_done);
    println!("  Slot payload size: {} bytes", ring.slot_payload_size());

    Ok(())
}

async fn cleanup_daemon(name: String) -> Result<(), Box<dyn std::error::Error>> {
    info!("Cleaning up shared memory: {}", name);
    platform::cleanup(&name);
    info!("Cleanup complete");
    Ok(())
}
