//! Image metadata — EXIF, XMP, IPTC, ICC profile, and format-specific chunks.
//!
//! Carries raw bytes per metadata kind. Parsed on the host side (SDK),
//! opaque to the pixel pipeline. The pipeline preserves metadata through
//! the graph and passes it to encoders for write-time embedding.

use crate::color_space::ColorSpace;

/// A single format-specific metadata chunk (e.g., PNG tEXt, GIF comment).
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataChunk {
    pub key: String,
    pub value: Vec<u8>,
}

/// Image metadata container — all metadata kinds from the source image.
///
/// Each field carries raw bytes. The pipeline preserves these opaquely.
/// Color space derivation reads ICC and EXIF to determine the working
/// color space; all other fields pass through unchanged.
#[derive(Debug, Clone, Default)]
pub struct ImageMetadata {
    /// Raw EXIF bytes (APP1 payload for JPEG, eXIf chunk for PNG).
    pub exif: Option<Vec<u8>>,
    /// Raw XMP bytes (XML string as UTF-8).
    pub xmp: Option<Vec<u8>>,
    /// Raw IPTC-IIM bytes (APP13 payload for JPEG).
    pub iptc: Option<Vec<u8>>,
    /// ICC color profile bytes.
    pub icc_profile: Option<Vec<u8>>,
    /// Format-specific metadata chunks.
    pub format_specific: Vec<MetadataChunk>,
}

impl ImageMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.exif.is_none()
            && self.xmp.is_none()
            && self.iptc.is_none()
            && self.icc_profile.is_none()
            && self.format_specific.is_empty()
    }

    /// Create metadata containing only an ICC profile.
    pub fn with_icc(icc_profile: Vec<u8>) -> Self {
        Self {
            icc_profile: Some(icc_profile),
            ..Default::default()
        }
    }
}

// ─── Color Space Derivation ─────────────────────────────────────────────────

/// Derive color space from image metadata.
///
/// Priority:
/// 1. ICC profile description tag → match against known profiles
/// 2. EXIF ColorSpace tag (1=sRGB, 2=Adobe RGB)
/// 3. Fallback to None (caller decides default)
pub fn derive_color_space(metadata: &ImageMetadata) -> Option<ColorSpace> {
    // Try ICC profile first (most authoritative)
    if let Some(ref icc) = metadata.icc_profile {
        if let Some(cs) = icc_to_color_space(icc) {
            return Some(cs);
        }
    }

    // Try EXIF ColorSpace tag
    if let Some(ref exif) = metadata.exif {
        if let Some(cs) = exif_color_space(exif) {
            return Some(cs);
        }
    }

    None
}

/// Match an ICC profile against well-known profiles by reading the
/// profile description tag.
///
/// Matches: sRGB, Display P3, Adobe RGB (1998), Rec. BT.2020, ProPhoto RGB.
fn icc_to_color_space(icc_profile: &[u8]) -> Option<ColorSpace> {
    if icc_profile.len() < 132 {
        return None;
    }

    let tag_count = u32::from_be_bytes(icc_profile[128..132].try_into().ok()?) as usize;
    let desc_sig = u32::from_be_bytes(*b"desc");

    for i in 0..tag_count.min(100) {
        let base = 132 + i * 12;
        if base + 12 > icc_profile.len() {
            break;
        }
        let sig = u32::from_be_bytes(icc_profile[base..base + 4].try_into().ok()?);
        if sig != desc_sig {
            continue;
        }
        let offset = u32::from_be_bytes(icc_profile[base + 4..base + 8].try_into().ok()?) as usize;
        let size = u32::from_be_bytes(icc_profile[base + 8..base + 12].try_into().ok()?) as usize;
        if offset + size > icc_profile.len() || size < 12 {
            break;
        }
        let desc_data = &icc_profile[offset..offset + size];

        // Try 'desc' type (v2 profiles): 4-byte type sig + 4 reserved + u32 ASCII len + ASCII
        if desc_data.len() > 12 {
            let type_sig = &desc_data[0..4];
            if type_sig == b"desc" {
                let ascii_len = u32::from_be_bytes(desc_data[8..12].try_into().ok()?) as usize;
                let end = (12 + ascii_len).min(desc_data.len());
                if let Ok(desc) = std::str::from_utf8(&desc_data[12..end]) {
                    if let Some(cs) = match_icc_description(desc.trim_end_matches('\0').trim()) {
                        return Some(cs);
                    }
                }
            }
        }

        // Try 'mluc' type (v4 profiles): multi-localized Unicode
        if desc_data.len() > 28 && &desc_data[0..4] == b"mluc" {
            let record_count = u32::from_be_bytes(desc_data[8..12].try_into().ok()?) as usize;
            if record_count > 0 {
                let str_len = u32::from_be_bytes(desc_data[20..24].try_into().ok()?) as usize;
                let str_offset = u32::from_be_bytes(desc_data[24..28].try_into().ok()?) as usize;
                if str_offset + str_len <= desc_data.len() {
                    let utf16: Vec<u16> = desc_data[str_offset..str_offset + str_len]
                        .chunks_exact(2)
                        .map(|c| u16::from_be_bytes([c[0], c[1]]))
                        .collect();
                    if let Ok(desc) = String::from_utf16(&utf16) {
                        if let Some(cs) = match_icc_description(desc.trim_end_matches('\0').trim()) {
                            return Some(cs);
                        }
                    }
                }
            }
        }
        break;
    }

    None
}

fn match_icc_description(desc: &str) -> Option<ColorSpace> {
    let lower = desc.to_lowercase();

    // Only match color spaces we have built-in transform support for.
    // Vendor-specific profiles (Adobe RGB, ProPhoto, etc.) remain as ICC
    // data in metadata — transforms are handled via the ICC rendering
    // engine (moxcms) when the pipeline has an ICC transform registered.
    if lower.contains("srgb") || lower.contains("iec61966") {
        Some(ColorSpace::Srgb)
    } else if lower.contains("display p3") {
        Some(ColorSpace::DisplayP3)
    } else if lower.contains("bt.2020") || lower.contains("bt2020") || lower.contains("rec.2020") || lower.contains("rec. 2020") {
        Some(ColorSpace::Rec2020)
    } else if lower.contains("bt.709") || lower.contains("bt709") || lower.contains("rec.709") || lower.contains("rec. 709") {
        Some(ColorSpace::Rec709)
    } else {
        // Unknown ICC profile — could be Adobe RGB, ProPhoto, camera-specific, etc.
        // The ICC profile bytes in metadata are the authoritative color definition.
        // Color transforms for these are provided by registered ICC transform operations.
        None
    }
}

/// Extract color space from EXIF ColorSpace tag (0xA001).
///
/// EXIF ColorSpace: 1 = sRGB, 2 = Adobe RGB, 0xFFFF = Uncalibrated.
fn exif_color_space(exif_data: &[u8]) -> Option<ColorSpace> {
    // Minimal EXIF parsing — find tag 0xA001 in IFD
    if exif_data.len() < 14 {
        return None;
    }

    // Detect byte order
    let big_endian = match &exif_data[0..2] {
        b"MM" => true,
        b"II" => false,
        _ => return None,
    };

    let read_u16 = |data: &[u8], offset: usize| -> Option<u16> {
        if offset + 2 > data.len() { return None; }
        Some(if big_endian {
            u16::from_be_bytes([data[offset], data[offset + 1]])
        } else {
            u16::from_le_bytes([data[offset], data[offset + 1]])
        })
    };

    let read_u32 = |data: &[u8], offset: usize| -> Option<u32> {
        if offset + 4 > data.len() { return None; }
        Some(if big_endian {
            u32::from_be_bytes(data[offset..offset + 4].try_into().ok()?)
        } else {
            u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?)
        })
    };

    // Check TIFF magic
    let magic = read_u16(exif_data, 2)?;
    if magic != 42 {
        return None;
    }

    // IFD0 offset
    let ifd0_offset = read_u32(exif_data, 4)? as usize;
    if ifd0_offset + 2 > exif_data.len() {
        return None;
    }

    // Search IFD0 for ExifIFD pointer (tag 0x8769), then search ExifIFD for ColorSpace (0xA001)
    let search_ifd = |ifd_offset: usize| -> Option<u16> {
        let count = read_u16(exif_data, ifd_offset)? as usize;
        for i in 0..count.min(200) {
            let entry = ifd_offset + 2 + i * 12;
            if entry + 12 > exif_data.len() { break; }
            let tag = read_u16(exif_data, entry)?;
            if tag == 0xA001 {
                // ColorSpace tag — type SHORT (3), count 1
                let value = read_u16(exif_data, entry + 8)?;
                return Some(value);
            }
        }
        None
    };

    // First try IFD0 directly
    if let Some(cs_val) = search_ifd(ifd0_offset) {
        return match cs_val {
            1 => Some(ColorSpace::Srgb),
            2 => Some(ColorSpace::Unknown), // Adobe RGB — ICC profile provides transform
            _ => None,
        };
    }

    // Find ExifIFD pointer in IFD0
    let count = read_u16(exif_data, ifd0_offset)? as usize;
    for i in 0..count.min(200) {
        let entry = ifd0_offset + 2 + i * 12;
        if entry + 12 > exif_data.len() { break; }
        let tag = read_u16(exif_data, entry)?;
        if tag == 0x8769 {
            // ExifIFD pointer
            let exif_ifd_offset = read_u32(exif_data, entry + 8)? as usize;
            if let Some(cs_val) = search_ifd(exif_ifd_offset) {
                return match cs_val {
                    1 => Some(ColorSpace::Srgb),
                    // EXIF ColorSpace=2 means Adobe RGB — no built-in transform,
                    // ICC profile in metadata provides the color definition.
                    // Mark as Unknown so the ICC transform path handles it.
                    2 => Some(ColorSpace::Unknown),
                    _ => None,
                };
            }
        }
    }

    None
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_metadata_is_empty() {
        assert!(ImageMetadata::new().is_empty());
    }

    #[test]
    fn with_icc_is_not_empty() {
        let m = ImageMetadata::with_icc(vec![1, 2, 3]);
        assert!(!m.is_empty());
        assert!(m.icc_profile.is_some());
    }

    #[test]
    fn icc_description_matching() {
        assert_eq!(match_icc_description("sRGB IEC61966-2.1"), Some(ColorSpace::Srgb));
        assert_eq!(match_icc_description("Display P3"), Some(ColorSpace::DisplayP3));
        assert_eq!(match_icc_description("ITU-R BT.2020"), Some(ColorSpace::Rec2020));
        assert_eq!(match_icc_description("Rec. 709"), Some(ColorSpace::Rec709));
        // Vendor profiles return None — handled via ICC transform engine
        assert_eq!(match_icc_description("Adobe RGB (1998)"), None);
        assert_eq!(match_icc_description("ProPhoto RGB"), None);
        assert_eq!(match_icc_description("Unknown Profile"), None);
    }

    #[test]
    fn derive_color_space_no_metadata() {
        let m = ImageMetadata::new();
        assert_eq!(derive_color_space(&m), None);
    }
}
