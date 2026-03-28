//! Pure Rust ISO Base Media File Format (ISOBMFF) parser.
//!
//! Parses HEIF/HEIC and AVIF container formats (ISO 14496-12 + ISO 23008-12).
//! Scoped to still-image use cases — not a full MP4 video parser.

mod boxreader;
pub mod error;
mod ftyp;
pub mod types;

pub use error::IsobmffError;
pub use types::{BoxHeader, Brand, CodecType, Ftyp, FullBoxHeader};

pub use boxreader::{BoxIterator, read_box_header, read_full_box_header};
pub use ftyp::{detect, parse_ftyp};
