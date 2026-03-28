//! Pure Rust ISO Base Media File Format (ISOBMFF) parser.
//!
//! Parses HEIF/HEIC and AVIF container formats (ISO 14496-12 + ISO 23008-12).
//! Scoped to still-image use cases — not a full MP4 video parser.

pub mod error;
pub mod types;
mod boxreader;
mod ftyp;

pub use error::IsobmffError;
pub use types::{BoxHeader, Brand, CodecType, FullBoxHeader, Ftyp};

pub use boxreader::{read_box_header, read_full_box_header, BoxIterator};
pub use ftyp::{detect, parse_ftyp};
