//! nsfw-ringd library crate

pub mod platform;
pub mod ring;

pub use ring::{RingBuffer, Frame, RingError, SlotMeta, RingHeader};
pub use platform::{SharedMemory, cleanup};