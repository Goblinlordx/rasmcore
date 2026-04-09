//! Image metadata — generic recursive key-value structure.
//!
//! All metadata kinds (EXIF, XMP, IPTC, ICC, format-specific) are stored
//! as entries in a flat list of key-value pairs. Values can be primitives,
//! raw bytes, nested maps, or arrays — supporting any current or future
//! metadata format without schema changes.
//!
//! The pipeline preserves metadata opaquely through the graph and passes
//! it to encoders for write-time embedding.

use crate::color_space::ColorSpace;

// ─── Generic Metadata Types ─────────────────────────────────────────────────

/// A typed metadata value — recursive, supports any metadata structure.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// UTF-8 string.
    Text(String),
    /// Floating-point number.
    Number(f64),
    /// Integer value.
    Integer(i64),
    /// Boolean flag.
    Flag(bool),
    /// Raw binary data (ICC profiles, raw EXIF bytes, etc.).
    Bytes(Vec<u8>),
    /// Ordered list of values.
    List(Vec<MetadataValue>),
    /// Key-value map (nested structure).
    Map(Vec<MetadataEntry>),
}

/// A single key-value metadata entry.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry {
    pub key: String,
    pub value: MetadataValue,
}

/// Image metadata — generic recursive key-value structure.
///
/// Top-level entries are metadata kinds: "exif", "icc", "xmp", "iptc",
/// format-specific keys like "png:text", etc.
///
/// ```text
/// [
///   { key: "icc", value: Bytes(<raw icc data>) },
///   { key: "exif", value: Map([
///     { key: "orientation", value: Integer(6) },
///     { key: "camera", value: Text("iPhone 15 Pro") },
///     { key: "gps", value: Map([
///       { key: "latitude", value: Number(37.7749) },
///       { key: "longitude", value: Number(-122.4194) },
///     ]) },
///   ]) },
///   { key: "xmp", value: Bytes(<raw xmp xml>) },
/// ]
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageMetadata {
    pub entries: Vec<MetadataEntry>,
}

impl ImageMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get a top-level entry by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.iter().find(|e| e.key == key).map(|e| &e.value)
    }

    /// Get a top-level bytes entry by key.
    pub fn get_bytes(&self, key: &str) -> Option<&[u8]> {
        match self.get(key) {
            Some(MetadataValue::Bytes(b)) => Some(b),
            _ => None,
        }
    }

    /// Set a top-level entry (replaces existing with same key).
    pub fn set(&mut self, key: impl Into<String>, value: MetadataValue) {
        let key = key.into();
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            entry.value = value;
        } else {
            self.entries.push(MetadataEntry { key, value });
        }
    }

    /// Create metadata containing only an ICC profile.
    pub fn with_icc(icc_profile: Vec<u8>) -> Self {
        Self {
            entries: vec![MetadataEntry {
                key: "icc".into(),
                value: MetadataValue::Bytes(icc_profile),
            }],
        }
    }
}

// ─── Color Space Derivation ─────────────────────────────────────────────────

/// Derive color space from image metadata.
///
/// Priority:
/// 1. ICC profile description tag → match against known standard profiles
/// 2. EXIF ColorSpace tag (1=sRGB, 2=Adobe RGB)
/// 3. Fallback to None (caller decides default)
pub fn derive_color_space(metadata: &ImageMetadata) -> Option<ColorSpace> {
    // Try ICC profile first (most authoritative)
    if let Some(icc) = metadata.get_bytes("icc") {
        if let Some(cs) = icc_to_color_space(icc) {
            return Some(cs);
        }
    }

    // Try EXIF ColorSpace tag
    if let Some(exif) = metadata.get_bytes("exif") {
        if let Some(cs) = exif_color_space(exif) {
            return Some(cs);
        }
    }

    None
}

/// Match an ICC profile against well-known profiles by reading the
/// profile description tag.
///
/// Only matches standard/open profiles we have built-in transforms for.
/// Vendor profiles (Adobe RGB, ProPhoto, etc.) return None — their ICC
/// data in metadata is the authoritative color definition.
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

        // Try 'desc' type (v2 profiles)
        if desc_data.len() > 12 && &desc_data[0..4] == b"desc" {
            let ascii_len = u32::from_be_bytes(desc_data[8..12].try_into().ok()?) as usize;
            let end = (12 + ascii_len).min(desc_data.len());
            if let Ok(desc) = std::str::from_utf8(&desc_data[12..end]) {
                if let Some(cs) = match_icc_description(desc.trim_end_matches('\0').trim()) {
                    return Some(cs);
                }
            }
        }

        // Try 'mluc' type (v4 profiles)
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

    if lower.contains("srgb") || lower.contains("iec61966") {
        Some(ColorSpace::Srgb)
    } else if lower.contains("display p3") {
        Some(ColorSpace::DisplayP3)
    } else if lower.contains("bt.2020") || lower.contains("bt2020") || lower.contains("rec.2020") || lower.contains("rec. 2020") {
        Some(ColorSpace::Rec2020)
    } else if lower.contains("bt.709") || lower.contains("bt709") || lower.contains("rec.709") || lower.contains("rec. 709") {
        Some(ColorSpace::Rec709)
    } else {
        None
    }
}

/// Extract color space from EXIF ColorSpace tag (0xA001).
fn exif_color_space(exif_data: &[u8]) -> Option<ColorSpace> {
    if exif_data.len() < 14 {
        return None;
    }

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

    if read_u16(exif_data, 2)? != 42 {
        return None;
    }

    let ifd0_offset = read_u32(exif_data, 4)? as usize;

    let search_ifd = |ifd_offset: usize| -> Option<u16> {
        let count = read_u16(exif_data, ifd_offset)? as usize;
        for i in 0..count.min(200) {
            let entry = ifd_offset + 2 + i * 12;
            if entry + 12 > exif_data.len() { break; }
            if read_u16(exif_data, entry)? == 0xA001 {
                return read_u16(exif_data, entry + 8);
            }
        }
        None
    };

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
        if read_u16(exif_data, entry)? == 0x8769 {
            let exif_ifd_offset = read_u32(exif_data, entry + 8)? as usize;
            if let Some(cs_val) = search_ifd(exif_ifd_offset) {
                return match cs_val {
                    1 => Some(ColorSpace::Srgb),
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
        assert!(m.get_bytes("icc").is_some());
    }

    #[test]
    fn get_and_set() {
        let mut m = ImageMetadata::new();
        m.set("exif", MetadataValue::Bytes(vec![0xFF, 0xE1]));
        m.set("camera", MetadataValue::Text("iPhone".into()));
        assert!(m.get_bytes("exif").is_some());
        assert_eq!(m.get("camera"), Some(&MetadataValue::Text("iPhone".into())));
    }

    #[test]
    fn set_replaces_existing() {
        let mut m = ImageMetadata::new();
        m.set("key", MetadataValue::Text("old".into()));
        m.set("key", MetadataValue::Text("new".into()));
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.get("key"), Some(&MetadataValue::Text("new".into())));
    }

    #[test]
    fn nested_structure() {
        let m = ImageMetadata {
            entries: vec![MetadataEntry {
                key: "exif".into(),
                value: MetadataValue::Map(vec![
                    MetadataEntry { key: "orientation".into(), value: MetadataValue::Integer(6) },
                    MetadataEntry { key: "camera".into(), value: MetadataValue::Text("Canon EOS R5".into()) },
                ]),
            }],
        };
        match m.get("exif") {
            Some(MetadataValue::Map(entries)) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].key, "orientation");
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn icc_description_matching() {
        assert_eq!(match_icc_description("sRGB IEC61966-2.1"), Some(ColorSpace::Srgb));
        assert_eq!(match_icc_description("Display P3"), Some(ColorSpace::DisplayP3));
        assert_eq!(match_icc_description("ITU-R BT.2020"), Some(ColorSpace::Rec2020));
        assert_eq!(match_icc_description("Rec. 709"), Some(ColorSpace::Rec709));
        assert_eq!(match_icc_description("Adobe RGB (1998)"), None);
        assert_eq!(match_icc_description("ProPhoto RGB"), None);
    }

    #[test]
    fn derive_color_space_no_metadata() {
        let m = ImageMetadata::new();
        assert_eq!(derive_color_space(&m), None);
    }
}
