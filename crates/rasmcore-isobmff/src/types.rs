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

/// Item info entry (infe box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemInfo {
    /// Item ID.
    pub item_id: u32,
    /// Item type as FourCC (e.g., hvc1, av01, grid, Exif).
    pub item_type: [u8; 4],
    /// Item name (UTF-8, may be empty).
    pub item_name: String,
    /// Item protection index (0 = not protected).
    pub protection_index: u16,
}

/// Single extent within an item location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// Offset of this extent relative to base_offset.
    pub offset: u64,
    /// Length of this extent in bytes.
    pub length: u64,
}

/// Item location entry (from iloc box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemLocation {
    /// Item ID.
    pub item_id: u32,
    /// Construction method: 0 = file offset, 1 = idat offset, 2 = item offset.
    pub construction_method: u8,
    /// Data reference index (0 = this file).
    pub data_reference_index: u16,
    /// Base offset added to each extent offset.
    pub base_offset: u64,
    /// List of extents for this item.
    pub extents: Vec<Extent>,
}

/// Reference type FourCC constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceType {
    /// Derived image (grid tiles).
    Dimg,
    /// Thumbnail.
    Thmb,
    /// Auxiliary image (e.g., alpha, depth).
    Auxl,
    /// Content describes (e.g., Exif describes an image).
    Cdsc,
    /// Unknown reference type.
    Unknown([u8; 4]),
}

impl ReferenceType {
    /// Parse from a 4-byte FourCC.
    pub fn from_fourcc(fourcc: [u8; 4]) -> Self {
        match &fourcc {
            b"dimg" => Self::Dimg,
            b"thmb" => Self::Thmb,
            b"auxl" => Self::Auxl,
            b"cdsc" => Self::Cdsc,
            _ => Self::Unknown(fourcc),
        }
    }
}

/// A single item reference (from iref box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemReference {
    /// Reference type.
    pub ref_type: ReferenceType,
    /// Source item ID.
    pub from_item_id: u32,
    /// Target item IDs.
    pub to_item_ids: Vec<u32>,
}

/// Collected results from parsing the meta box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaBox {
    /// Primary item ID (from pitm box).
    pub primary_item_id: u32,
    /// Item info entries (from iinf box).
    pub items: Vec<ItemInfo>,
    /// Item locations (from iloc box).
    pub locations: Vec<ItemLocation>,
    /// Item references (from iref box).
    pub references: Vec<ItemReference>,
}

// ─── Item Properties ────────────────────────────────────────────────────────

/// Association of a property to an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropertyAssociation {
    /// Item ID this property is associated with.
    pub item_id: u32,
    /// 1-based index into the ipco property list.
    pub property_index: u16,
    /// Whether the property is essential for correct rendering.
    pub essential: bool,
}

/// Image spatial extents (ispe property).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpatialExtents {
    pub width: u32,
    pub height: u32,
}

/// Color information (colr property).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorInfo {
    /// ICC profile (type = "prof" or "rICC").
    Icc(Vec<u8>),
    /// NCLX color parameters (type = "nclx").
    Nclx {
        colour_primaries: u16,
        transfer_characteristics: u16,
        matrix_coefficients: u16,
        full_range: bool,
    },
}

/// Pixel information (pixi property).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelInfo {
    /// Bits per channel for each channel.
    pub bits_per_channel: Vec<u8>,
}

/// HEVC decoder configuration record (hvcC property).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcConfig {
    pub configuration_version: u8,
    pub general_profile_space: u8,
    pub general_tier_flag: bool,
    pub general_profile_idc: u8,
    pub general_level_idc: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    /// NAL unit arrays (VPS, SPS, PPS, SEI, etc.).
    pub nal_arrays: Vec<NalArray>,
}

/// A NAL unit array within hvcC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NalArray {
    /// Whether the NAL units are complete (array_completeness flag).
    pub completeness: bool,
    /// NAL unit type (e.g., 32=VPS, 33=SPS, 34=PPS).
    pub nal_type: u8,
    /// Raw NAL unit data (each entry is one NAL unit).
    pub nal_units: Vec<Vec<u8>>,
}

/// AV1 codec configuration record (av1C property).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1Config {
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: bool,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub chroma_sample_position: u8,
    /// Raw config OBUs (if any follow the fixed fields).
    pub config_obus: Vec<u8>,
}

/// A parsed property from ipco.
#[derive(Debug, Clone, PartialEq)]
pub enum Property {
    ImageSpatialExtents(ImageSpatialExtents),
    Color(ColorInfo),
    Pixel(PixelInfo),
    HevcConfig(HevcConfig),
    Av1Config(Av1Config),
    /// Unknown property — stores FourCC and raw data.
    Unknown {
        box_type: [u8; 4],
        data: Vec<u8>,
    },
}

/// Collected item properties from iprp box.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemProperties {
    /// Ordered list of properties from ipco (1-based indexing for ipma).
    pub properties: Vec<Property>,
    /// Associations mapping properties to items.
    pub associations: Vec<PropertyAssociation>,
}
