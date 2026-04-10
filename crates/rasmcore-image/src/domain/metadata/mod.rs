//! Metadata types and operations — EXIF, IPTC, XMP, and unified containers.

pub mod exif;
pub mod iptc;
pub mod query;
pub mod set;
pub mod xmp;

// Re-export primary types for convenience
pub use exif::{ExifMetadata, ExifOrientation, has_exif, read_exif, read_metadata, write_exif};
pub use iptc::{IptcMetadata, parse_iptc, serialize_iptc};
pub use query::{
    metadata_dump_json, metadata_dump_json_from_bytes, metadata_read, metadata_read_from_bytes,
};
pub use set::{MetadataChunk, MetadataSet};
pub use xmp::{XmpMetadata, parse_xmp, serialize_xmp};
