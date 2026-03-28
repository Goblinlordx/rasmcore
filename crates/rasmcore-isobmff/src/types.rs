//! Public types for ISOBMFF parsing.

/// Known file brands from the ftyp box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Brand {
    /// HEIC — HEVC-coded still image (Apple default).
    Heic,
    /// HEIX — HEVC-coded still image with extensions.
    Heix,
    /// HEIM — HEVC-coded still image, multi-layer.
    Heim,
    /// HEIS — HEVC-coded still image sequence.
    Heis,
    /// AVIF — AV1-coded still image.
    Avif,
    /// MIF1 — generic HEIF still image (ISO 23008-12).
    Mif1,
    /// MSF1 — HEIF image sequence.
    Msf1,
    /// Unknown brand with raw FourCC.
    Unknown([u8; 4]),
}

impl Brand {
    /// Parse a brand from a 4-byte FourCC.
    pub fn from_fourcc(fourcc: [u8; 4]) -> Self {
        match &fourcc {
            b"heic" => Self::Heic,
            b"heix" => Self::Heix,
            b"heim" => Self::Heim,
            b"heis" => Self::Heis,
            b"avif" => Self::Avif,
            b"mif1" => Self::Mif1,
            b"msf1" => Self::Msf1,
            _ => Self::Unknown(fourcc),
        }
    }

    /// Whether this brand indicates a HEIF still image file.
    pub fn is_heif(&self) -> bool {
        matches!(
            self,
            Self::Heic | Self::Heix | Self::Heim | Self::Heis | Self::Mif1
        )
    }

    /// Whether this brand indicates HEVC-coded content.
    pub fn is_heic(&self) -> bool {
        matches!(self, Self::Heic | Self::Heix | Self::Heim | Self::Heis)
    }

    /// Whether this brand indicates AV1-coded content.
    pub fn is_avif(&self) -> bool {
        matches!(self, Self::Avif)
    }
}

/// Codec type identified by item type FourCC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecType {
    /// HEVC / H.265.
    Hevc,
    /// AV1.
    Av1,
    /// JPEG.
    Jpeg,
    /// Unknown codec with raw FourCC.
    Unknown([u8; 4]),
}

impl CodecType {
    /// Parse codec type from an item type FourCC.
    pub fn from_fourcc(fourcc: [u8; 4]) -> Self {
        match &fourcc {
            b"hvc1" | b"hev1" => Self::Hevc,
            b"av01" => Self::Av1,
            b"jpeg" => Self::Jpeg,
            _ => Self::Unknown(fourcc),
        }
    }
}

/// Parsed box header from the ISOBMFF stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxHeader {
    /// FourCC box type.
    pub box_type: [u8; 4],
    /// Total box size including header. `None` means box extends to end of file.
    pub box_size: Option<u64>,
    /// Offset of the box content (after the header) within the parent data.
    pub content_offset: usize,
    /// Size of the box content (excluding the header).
    pub content_size: Option<u64>,
    /// Size of the header itself (8 for normal, 16 for extended).
    pub header_size: u8,
}

/// Parsed full-box header (version + flags extension).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullBoxHeader {
    /// The base box header.
    pub box_header: BoxHeader,
    /// Version field (1 byte).
    pub version: u8,
    /// Flags field (3 bytes, stored as u32 with high byte zero).
    pub flags: u32,
}

/// Parsed ftyp (file type) box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ftyp {
    /// Major brand.
    pub major_brand: Brand,
    /// Minor version.
    pub minor_version: u32,
    /// List of compatible brands.
    pub compatible_brands: Vec<Brand>,
}
