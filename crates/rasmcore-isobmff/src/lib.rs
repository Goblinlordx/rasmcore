//! Pure Rust ISO Base Media File Format (ISOBMFF) parser.
//!
//! Parses HEIF/HEIC and AVIF container formats (ISO 14496-12 + ISO 23008-12).
//! Scoped to still-image use cases — not a full MP4 video parser.

mod boxreader;
pub mod error;
mod ftyp;
mod meta;
mod properties;
pub mod types;

pub use error::IsobmffError;
pub use types::{
    Av1Config, BoxHeader, Brand, CodecType, ColorInfo, Extent, Ftyp, FullBoxHeader, HevcConfig,
    ImageSpatialExtents, ItemInfo, ItemLocation, ItemProperties, ItemReference, MetaBox, NalArray,
    PixelInfo, Property, PropertyAssociation, ReferenceType,
};

pub use boxreader::{BoxIterator, read_box_header, read_full_box_header};
pub use ftyp::{detect, parse_ftyp};
pub use meta::parse_meta;
pub use properties::{parse_iprp, resolve_properties};
