use std::io::Cursor;

use super::error::ImageError;

/// EXIF orientation tag values (1-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExifOrientation {
    /// Normal (no transform needed).
    Normal,
    /// Flip horizontal.
    FlipHorizontal,
    /// Rotate 180.
    Rotate180,
    /// Flip vertical.
    FlipVertical,
    /// Transpose (rotate 270 + flip horizontal).
    Transpose,
    /// Rotate 90 CW.
    Rotate90,
    /// Transverse (rotate 90 + flip horizontal).
    Transverse,
    /// Rotate 270 CW.
    Rotate270,
}

impl ExifOrientation {
    /// Parse from EXIF orientation tag value (1-8).
    pub fn from_tag(value: u32) -> Self {
        match value {
            1 => Self::Normal,
            2 => Self::FlipHorizontal,
            3 => Self::Rotate180,
            4 => Self::FlipVertical,
            5 => Self::Transpose,
            6 => Self::Rotate90,
            7 => Self::Transverse,
            8 => Self::Rotate270,
            _ => Self::Normal,
        }
    }

    /// Convert to EXIF tag value (1-8).
    pub fn to_tag(self) -> u32 {
        match self {
            Self::Normal => 1,
            Self::FlipHorizontal => 2,
            Self::Rotate180 => 3,
            Self::FlipVertical => 4,
            Self::Transpose => 5,
            Self::Rotate90 => 6,
            Self::Transverse => 7,
            Self::Rotate270 => 8,
        }
    }
}

/// Structured EXIF metadata read from an image.
#[derive(Debug, Clone, Default)]
pub struct ExifMetadata {
    pub orientation: Option<ExifOrientation>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub date_time: Option<String>,
    pub software: Option<String>,
}

/// Read EXIF metadata from encoded image data (JPEG, TIFF, WebP).
pub fn read_exif(data: &[u8]) -> Result<ExifMetadata, ImageError> {
    let mut cursor = Cursor::new(data);
    let reader = exif::Reader::new();
    let exif_data = reader
        .read_from_container(&mut cursor)
        .map_err(|e| ImageError::ProcessingFailed(format!("EXIF read failed: {e}")))?;

    let mut meta = ExifMetadata::default();

    if let Some(field) = exif_data.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        && let Some(v) = field.value.get_uint(0)
    {
        meta.orientation = Some(ExifOrientation::from_tag(v));
    }

    if let Some(field) = exif_data.get_field(exif::Tag::PixelXDimension, exif::In::PRIMARY)
        && let Some(v) = field.value.get_uint(0)
    {
        meta.width = Some(v);
    }

    if let Some(field) = exif_data.get_field(exif::Tag::PixelYDimension, exif::In::PRIMARY)
        && let Some(v) = field.value.get_uint(0)
    {
        meta.height = Some(v);
    }

    if let Some(field) = exif_data.get_field(exif::Tag::Make, exif::In::PRIMARY) {
        meta.camera_make = Some(
            field
                .display_value()
                .to_string()
                .trim_matches('"')
                .to_string(),
        );
    }

    if let Some(field) = exif_data.get_field(exif::Tag::Model, exif::In::PRIMARY) {
        meta.camera_model = Some(
            field
                .display_value()
                .to_string()
                .trim_matches('"')
                .to_string(),
        );
    }

    if let Some(field) = exif_data.get_field(exif::Tag::DateTime, exif::In::PRIMARY) {
        meta.date_time = Some(
            field
                .display_value()
                .to_string()
                .trim_matches('"')
                .to_string(),
        );
    }

    if let Some(field) = exif_data.get_field(exif::Tag::Software, exif::In::PRIMARY) {
        meta.software = Some(
            field
                .display_value()
                .to_string()
                .trim_matches('"')
                .to_string(),
        );
    }

    Ok(meta)
}

/// Check if the given encoded data contains EXIF metadata.
pub fn has_exif(data: &[u8]) -> bool {
    read_exif(data).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_from_tag_all_values() {
        assert_eq!(ExifOrientation::from_tag(1), ExifOrientation::Normal);
        assert_eq!(
            ExifOrientation::from_tag(2),
            ExifOrientation::FlipHorizontal
        );
        assert_eq!(ExifOrientation::from_tag(3), ExifOrientation::Rotate180);
        assert_eq!(ExifOrientation::from_tag(4), ExifOrientation::FlipVertical);
        assert_eq!(ExifOrientation::from_tag(5), ExifOrientation::Transpose);
        assert_eq!(ExifOrientation::from_tag(6), ExifOrientation::Rotate90);
        assert_eq!(ExifOrientation::from_tag(7), ExifOrientation::Transverse);
        assert_eq!(ExifOrientation::from_tag(8), ExifOrientation::Rotate270);
    }

    #[test]
    fn orientation_from_tag_unknown_defaults_to_normal() {
        assert_eq!(ExifOrientation::from_tag(0), ExifOrientation::Normal);
        assert_eq!(ExifOrientation::from_tag(99), ExifOrientation::Normal);
    }

    #[test]
    fn orientation_roundtrip() {
        for tag in 1..=8 {
            let orient = ExifOrientation::from_tag(tag);
            assert_eq!(orient.to_tag(), tag);
        }
    }

    #[test]
    fn read_exif_from_non_exif_data_returns_error() {
        // PNG doesn't have EXIF — should fail gracefully
        let png_header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let result = read_exif(&png_header);
        assert!(result.is_err());
    }

    #[test]
    fn has_exif_returns_false_for_non_exif() {
        assert!(!has_exif(&[0x89, 0x50, 0x4E, 0x47]));
    }

    #[test]
    fn read_exif_from_truncated_jpeg_returns_error() {
        let truncated_jpeg = [
            0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x08, b'E', b'x', b'i', b'f', 0x00, 0x00,
        ];
        let result = read_exif(&truncated_jpeg);
        // Truncated EXIF should return error, not panic
        assert!(result.is_err());
    }

    #[test]
    fn read_exif_from_valid_jpeg_with_orientation() {
        // Construct a minimal valid JPEG with EXIF containing orientation=6 (Rotate90).
        // JPEG: SOI + APP1(Exif) with TIFF little-endian header + IFD with orientation tag.
        let exif_jpeg = build_exif_jpeg_with_orientation(6);
        let result = read_exif(&exif_jpeg).unwrap();
        assert_eq!(result.orientation, Some(ExifOrientation::Rotate90));
    }

    /// Build a minimal JPEG with EXIF orientation tag for testing.
    fn build_exif_jpeg_with_orientation(orientation: u16) -> Vec<u8> {
        let mut buf = Vec::new();

        // SOI
        buf.extend_from_slice(&[0xFF, 0xD8]);

        // APP1 marker
        buf.extend_from_slice(&[0xFF, 0xE1]);

        // Build EXIF payload first to calculate length
        let mut exif_payload = Vec::new();
        // "Exif\0\0"
        exif_payload.extend_from_slice(b"Exif\x00\x00");

        // TIFF header (little-endian)
        let tiff_start = exif_payload.len();
        exif_payload.extend_from_slice(b"II"); // little-endian
        exif_payload.extend_from_slice(&42u16.to_le_bytes()); // magic
        exif_payload.extend_from_slice(&8u32.to_le_bytes()); // offset to IFD0

        // IFD0 at offset 8 from TIFF start
        let ifd_count: u16 = 1;
        exif_payload.extend_from_slice(&ifd_count.to_le_bytes());

        // Orientation tag: tag=0x0112, type=SHORT(3), count=1, value=orientation
        exif_payload.extend_from_slice(&0x0112u16.to_le_bytes()); // tag
        exif_payload.extend_from_slice(&3u16.to_le_bytes()); // type SHORT
        exif_payload.extend_from_slice(&1u32.to_le_bytes()); // count
        exif_payload.extend_from_slice(&(orientation as u32).to_le_bytes()); // value

        // Next IFD offset = 0 (no more IFDs)
        exif_payload.extend_from_slice(&0u32.to_le_bytes());

        let _ = tiff_start; // suppress warning

        // APP1 length (includes 2 bytes for length field itself)
        let app1_len = (exif_payload.len() + 2) as u16;
        buf.extend_from_slice(&app1_len.to_be_bytes());
        buf.extend_from_slice(&exif_payload);

        // Minimal SOS + EOI to make it a valid JPEG
        buf.extend_from_slice(&[0xFF, 0xD9]);

        buf
    }
}
