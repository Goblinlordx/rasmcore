/// Unified metadata container for all metadata types across image formats.
///
/// Carries raw bytes per metadata kind — parsed on the host side (SDK),
/// opaque to the pixel pipeline. The pipeline receives a finalized,
/// immutable MetadataSet and embeds it during the write sink phase.
///
/// Design:
/// - Passthrough by default: if provided to write, all present metadata is embedded
/// - Host-side mapping: transform logic (include/exclude/set) runs on host, not WASM
/// - Sequential phase separation: all metadata ops complete before pipeline executes
/// - Same-format raw byte splice: when input/output format match, can copy raw segments

/// A single metadata chunk with a key and raw byte value.
/// Used for format-specific metadata (PNG text chunks, GIF comments, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataChunk {
    pub key: String,
    pub value: Vec<u8>,
}

/// Unified metadata container covering all metadata kinds.
///
/// Each field carries raw bytes for its metadata type. The write sink
/// handles format-specific serialization (e.g., EXIF → JPEG APP1,
/// ICC → PNG iCCP chunk).
#[derive(Debug, Clone, Default)]
pub struct MetadataSet {
    /// Raw EXIF bytes (APP1 payload for JPEG, eXIf chunk for PNG).
    pub exif: Option<Vec<u8>>,

    /// Raw XMP bytes (XML string as UTF-8).
    pub xmp: Option<Vec<u8>>,

    /// Raw IPTC-IIM bytes (APP13 payload for JPEG).
    pub iptc: Option<Vec<u8>>,

    /// ICC color profile bytes.
    pub icc_profile: Option<Vec<u8>>,

    /// Format-specific metadata chunks (PNG tEXt/zTXt/iTXt, GIF comment, etc.).
    pub format_specific: Vec<MetadataChunk>,
}

impl MetadataSet {
    /// Create an empty MetadataSet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if this MetadataSet contains any metadata at all.
    pub fn is_empty(&self) -> bool {
        self.exif.is_none()
            && self.xmp.is_none()
            && self.iptc.is_none()
            && self.icc_profile.is_none()
            && self.format_specific.is_empty()
    }

    /// Create a MetadataSet containing only an ICC profile.
    pub fn with_icc(icc_profile: Vec<u8>) -> Self {
        Self {
            icc_profile: Some(icc_profile),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let ms = MetadataSet::new();
        assert!(ms.is_empty());
    }

    #[test]
    fn with_icc_is_not_empty() {
        let ms = MetadataSet::with_icc(vec![1, 2, 3]);
        assert!(!ms.is_empty());
        assert_eq!(ms.icc_profile, Some(vec![1, 2, 3]));
        assert!(ms.exif.is_none());
    }

    #[test]
    fn with_exif_is_not_empty() {
        let ms = MetadataSet {
            exif: Some(vec![0xFF, 0xE1]),
            ..Default::default()
        };
        assert!(!ms.is_empty());
    }

    #[test]
    fn with_format_specific_is_not_empty() {
        let ms = MetadataSet {
            format_specific: vec![MetadataChunk {
                key: "png:text:Title".to_string(),
                value: b"My Image".to_vec(),
            }],
            ..Default::default()
        };
        assert!(!ms.is_empty());
    }

    #[test]
    fn clone_preserves_all_fields() {
        let ms = MetadataSet {
            exif: Some(vec![1]),
            xmp: Some(vec![2]),
            iptc: Some(vec![3]),
            icc_profile: Some(vec![4]),
            format_specific: vec![MetadataChunk {
                key: "k".to_string(),
                value: vec![5],
            }],
        };
        let cloned = ms.clone();
        assert_eq!(ms.exif, cloned.exif);
        assert_eq!(ms.xmp, cloned.xmp);
        assert_eq!(ms.iptc, cloned.iptc);
        assert_eq!(ms.icc_profile, cloned.icc_profile);
        assert_eq!(ms.format_specific, cloned.format_specific);
    }
}
