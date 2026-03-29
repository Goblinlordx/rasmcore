//! Public types for HEVC decoding.

/// HEVC NAL unit types (ITU-T H.265 Table 7-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NalUnitType {
    TrailN = 0,
    TrailR = 1,
    TsaN = 2,
    TsaR = 3,
    StsaN = 4,
    StsaR = 5,
    RadlN = 6,
    RadlR = 7,
    RaslN = 8,
    RaslR = 9,
    BlaWLp = 16,
    BlaWRadl = 17,
    BlaNLp = 18,
    IdrWRadl = 19,
    IdrNLp = 20,
    CraNut = 21,
    VpsNut = 32,
    SpsNut = 33,
    PpsNut = 34,
    AudNut = 35,
    EosNut = 36,
    EobNut = 37,
    FdNut = 38,
    PrefixSeiNut = 39,
    SuffixSeiNut = 40,
    Unknown(u8),
}

impl NalUnitType {
    /// Parse NAL unit type from the 6-bit field.
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::TrailN,
            1 => Self::TrailR,
            2 => Self::TsaN,
            3 => Self::TsaR,
            4 => Self::StsaN,
            5 => Self::StsaR,
            6 => Self::RadlN,
            7 => Self::RadlR,
            8 => Self::RaslN,
            9 => Self::RaslR,
            16 => Self::BlaWLp,
            17 => Self::BlaWRadl,
            18 => Self::BlaNLp,
            19 => Self::IdrWRadl,
            20 => Self::IdrNLp,
            21 => Self::CraNut,
            32 => Self::VpsNut,
            33 => Self::SpsNut,
            34 => Self::PpsNut,
            35 => Self::AudNut,
            36 => Self::EosNut,
            37 => Self::EobNut,
            38 => Self::FdNut,
            39 => Self::PrefixSeiNut,
            40 => Self::SuffixSeiNut,
            v => Self::Unknown(v),
        }
    }

    /// Whether this NAL type is a VCL (Video Coding Layer) NAL unit.
    pub fn is_vcl(&self) -> bool {
        let v = self.as_u8();
        v <= 31
    }

    /// Whether this NAL type is an IDR picture.
    pub fn is_idr(&self) -> bool {
        matches!(self, Self::IdrWRadl | Self::IdrNLp)
    }

    /// Whether this NAL type is an IRAP (Intra Random Access Point).
    pub fn is_irap(&self) -> bool {
        let v = self.as_u8();
        (16..=23).contains(&v)
    }

    /// Get the raw u8 value.
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Unknown(v) => *v,
            other => {
                // Safe because all named variants have explicit discriminants
                // Use match for safety
                match other {
                    Self::TrailN => 0,
                    Self::TrailR => 1,
                    Self::TsaN => 2,
                    Self::TsaR => 3,
                    Self::StsaN => 4,
                    Self::StsaR => 5,
                    Self::RadlN => 6,
                    Self::RadlR => 7,
                    Self::RaslN => 8,
                    Self::RaslR => 9,
                    Self::BlaWLp => 16,
                    Self::BlaWRadl => 17,
                    Self::BlaNLp => 18,
                    Self::IdrWRadl => 19,
                    Self::IdrNLp => 20,
                    Self::CraNut => 21,
                    Self::VpsNut => 32,
                    Self::SpsNut => 33,
                    Self::PpsNut => 34,
                    Self::AudNut => 35,
                    Self::EosNut => 36,
                    Self::EobNut => 37,
                    Self::FdNut => 38,
                    Self::PrefixSeiNut => 39,
                    Self::SuffixSeiNut => 40,
                    Self::Unknown(v) => *v,
                }
            }
        }
    }
}

/// Parsed NAL unit header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NalUnit {
    /// NAL unit type.
    pub nal_type: NalUnitType,
    /// Layer ID (6 bits, usually 0).
    pub nuh_layer_id: u8,
    /// Temporal ID + 1 (3 bits, 1-7).
    pub nuh_temporal_id_plus1: u8,
    /// Raw RBSP data (after emulation prevention byte removal).
    pub rbsp: Vec<u8>,
}

/// Decoded frame output.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Pixel data (RGB8 format).
    pub pixels: Vec<u8>,
    /// Luma (Y) plane — raw reconstructed samples before YCbCr→RGB conversion.
    pub y_plane: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Bit depth (8 or 10).
    pub bit_depth: u8,
}
