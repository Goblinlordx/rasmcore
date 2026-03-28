use std::io::Cursor;

use super::color;
use super::error::ImageError;
use super::metadata_set::MetadataSet;

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

/// Read all metadata from encoded image data without decoding pixels.
///
/// Performs streaming header-only parsing — scans the container structure
/// to extract raw EXIF, XMP, IPTC, and ICC bytes. No pixel data is allocated.
pub fn read_metadata(data: &[u8]) -> Result<MetadataSet, ImageError> {
    if data.len() < 4 {
        return Err(ImageError::InvalidInput("data too short".into()));
    }

    // Detect format from magic bytes and dispatch to format-specific scanner
    if data.starts_with(&[0xFF, 0xD8]) {
        read_metadata_jpeg(data)
    } else if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        read_metadata_png(data)
    } else if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        read_metadata_webp(data)
    } else if data.starts_with(&[0x49, 0x49, 0x2A, 0x00])
        || data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
    {
        read_metadata_tiff(data)
    } else {
        // Unknown format — return empty metadata (not an error)
        Ok(MetadataSet::new())
    }
}

/// Scan JPEG APP markers to extract EXIF (APP1), XMP (APP1), IPTC (APP13), ICC (APP2).
fn read_metadata_jpeg(data: &[u8]) -> Result<MetadataSet, ImageError> {
    let mut metadata = MetadataSet::new();
    let mut pos = 2; // skip SOI (0xFF 0xD8)

    while pos + 4 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let marker = data[pos + 1];

        // Stop at SOS (start of scan) — no more metadata after this
        if marker == 0xDA {
            break;
        }

        // Skip RST markers and standalone markers
        if marker == 0x00 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            pos += 2;
            continue;
        }

        if pos + 4 > data.len() {
            break;
        }

        let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        let seg_start = pos + 4;
        let seg_end = pos + 2 + seg_len;

        if seg_end > data.len() {
            break;
        }

        let payload = &data[seg_start..seg_end];

        match marker {
            // APP1 — EXIF or XMP
            0xE1 => {
                if payload.starts_with(b"Exif\x00\x00") && metadata.exif.is_none() {
                    // Store the full APP1 payload (including "Exif\0\0" header)
                    metadata.exif = Some(payload.to_vec());
                } else if payload.starts_with(b"http://ns.adobe.com/xap/1.0/\x00")
                    && metadata.xmp.is_none()
                {
                    // XMP: skip the namespace URI + null terminator
                    let xmp_start = b"http://ns.adobe.com/xap/1.0/\x00".len();
                    if xmp_start < payload.len() {
                        metadata.xmp = Some(payload[xmp_start..].to_vec());
                    }
                }
            }
            // APP13 — IPTC
            0xED => {
                if payload.starts_with(b"Photoshop 3.0\x00") && metadata.iptc.is_none() {
                    metadata.iptc = Some(payload.to_vec());
                }
            }
            _ => {}
        }

        pos = seg_end;
    }

    // ICC profile extraction via existing robust implementation
    metadata.icc_profile = color::extract_icc_from_jpeg(data);

    Ok(metadata)
}

/// Scan PNG chunks to extract eXIf, iCCP, tEXt, zTXt, iTXt.
fn read_metadata_png(data: &[u8]) -> Result<MetadataSet, ImageError> {
    use super::metadata_set::MetadataChunk;

    let mut metadata = MetadataSet::new();
    let mut pos = 8; // skip PNG signature

    while pos + 12 <= data.len() {
        let chunk_len =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let chunk_type = &data[pos + 4..pos + 8];
        let chunk_data_start = pos + 8;
        let chunk_data_end = chunk_data_start + chunk_len;

        if chunk_data_end + 4 > data.len() {
            break;
        }

        let chunk_data = &data[chunk_data_start..chunk_data_end];

        match chunk_type {
            b"eXIf" => {
                if metadata.exif.is_none() {
                    metadata.exif = Some(chunk_data.to_vec());
                }
            }
            b"tEXt" | b"zTXt" | b"iTXt" => {
                let type_str = std::str::from_utf8(chunk_type).unwrap_or("text");
                metadata.format_specific.push(MetadataChunk {
                    key: format!("png:{type_str}"),
                    value: chunk_data.to_vec(),
                });
            }
            b"IDAT" => {
                // Stop at first IDAT — all metadata chunks must appear before pixel data
                break;
            }
            _ => {}
        }

        // Move past: length(4) + type(4) + data(chunk_len) + CRC(4)
        pos = chunk_data_end + 4;
    }

    // ICC profile extraction via existing robust implementation
    metadata.icc_profile = color::extract_icc_from_png(data);

    Ok(metadata)
}

/// Scan WebP RIFF chunks to extract EXIF, XMP, ICCP.
fn read_metadata_webp(data: &[u8]) -> Result<MetadataSet, ImageError> {
    let mut metadata = MetadataSet::new();

    // WebP RIFF structure: "RIFF" + size(4) + "WEBP" + chunks
    let mut pos = 12; // skip "RIFF" + size + "WEBP"

    while pos + 8 <= data.len() {
        let fourcc = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;
        let chunk_data_start = pos + 8;
        let chunk_data_end = chunk_data_start + chunk_size;

        if chunk_data_end > data.len() {
            break;
        }

        let chunk_data = &data[chunk_data_start..chunk_data_end];

        match fourcc {
            b"EXIF" => {
                if metadata.exif.is_none() {
                    metadata.exif = Some(chunk_data.to_vec());
                }
            }
            b"XMP " => {
                if metadata.xmp.is_none() {
                    metadata.xmp = Some(chunk_data.to_vec());
                }
            }
            b"ICCP" => {
                if metadata.icc_profile.is_none() {
                    metadata.icc_profile = Some(chunk_data.to_vec());
                }
            }
            _ => {}
        }

        // RIFF chunks are padded to even size
        let padded_size = (chunk_size + 1) & !1;
        pos = chunk_data_start + padded_size;
    }

    Ok(metadata)
}

/// Extract metadata from TIFF by reading IFD0 entries.
/// TIFF/EXIF share the same IFD structure — the whole file IS the EXIF data.
fn read_metadata_tiff(data: &[u8]) -> Result<MetadataSet, ImageError> {
    let mut metadata = MetadataSet::new();

    // For TIFF, the entire file structure is EXIF-compatible IFD
    // Store the raw bytes as EXIF data (kamadak-exif can parse TIFF directly)
    metadata.exif = Some(data.to_vec());

    // ICC profile: TIFF tag 34675 (ICC Profile) — would require IFD parsing
    // For now, leave ICC extraction for TIFF as future work

    Ok(metadata)
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

    #[test]
    fn read_metadata_from_jpeg_extracts_exif() {
        let jpeg = build_exif_jpeg_with_orientation(6);
        let ms = read_metadata(&jpeg).unwrap();
        assert!(ms.exif.is_some(), "should extract EXIF from JPEG");
        // EXIF payload starts with "Exif\0\0"
        assert!(ms.exif.as_ref().unwrap().starts_with(b"Exif\x00\x00"));
    }

    #[test]
    fn read_metadata_from_png_returns_ok() {
        // Minimal PNG: signature + IHDR + IDAT + IEND (no metadata chunks)
        let png = build_minimal_png();
        let ms = read_metadata(&png).unwrap();
        assert!(ms.exif.is_none());
        assert!(ms.icc_profile.is_none());
        assert!(ms.format_specific.is_empty());
    }

    #[test]
    fn read_metadata_from_unknown_format_returns_empty() {
        let ms = read_metadata(&[0x00, 0x00, 0x00, 0x00]).unwrap();
        assert!(ms.is_empty());
    }

    #[test]
    fn read_metadata_short_data_returns_error() {
        let result = read_metadata(&[0xFF]);
        assert!(result.is_err());
    }

    fn build_minimal_png() -> Vec<u8> {
        let mut buf = Vec::new();
        // PNG signature
        buf.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR: 1x1 RGB8
        let ihdr_data = [
            0x00, 0x00, 0x00, 0x01, // width=1
            0x00, 0x00, 0x00, 0x01, // height=1
            0x08, // bit depth=8
            0x02, // color type=RGB
            0x00, // compression
            0x00, // filter
            0x00, // interlace
        ];
        append_png_chunk(&mut buf, b"IHDR", &ihdr_data);
        // Minimal IDAT (empty compressed)
        let idat_data = [
            0x08, 0xD7, 0x63, 0x60, 0x60, 0x60, 0x00, 0x00, 0x00, 0x04, 0x00, 0x01,
        ];
        append_png_chunk(&mut buf, b"IDAT", &idat_data);
        // IEND
        append_png_chunk(&mut buf, b"IEND", &[]);
        buf
    }

    fn append_png_chunk(buf: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
        buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
        buf.extend_from_slice(chunk_type);
        buf.extend_from_slice(data);
        // Simple CRC placeholder (not validated in our tests)
        let crc = crc32(chunk_type, data);
        buf.extend_from_slice(&crc.to_be_bytes());
    }

    fn crc32(chunk_type: &[u8], data: &[u8]) -> u32 {
        // Minimal CRC-32 for PNG chunks
        let mut crc: u32 = 0xFFFFFFFF;
        for &byte in chunk_type.iter().chain(data.iter()) {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc
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
