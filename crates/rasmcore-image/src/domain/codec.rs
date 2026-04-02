//! Unified Codec Registry — one registration per format.
//!
//! Every image format and LUT format registers a single `CodecRegistration`
//! entry via `inventory::submit!`. The registry provides:
//! - Format detection (optional detect_fn per format)
//! - Image decode/encode dispatch
//! - LUT decode/encode dispatch
//! - Format metadata (extensions, MIME type)
//!
//! Adding a new format requires ONE `inventory::submit!` block — no dispatch
//! code changes, no hardcoded format lists.

use super::color_lut::ColorLut3D;
use super::error::ImageError;
use super::types::{DecodedImage, ImageInfo};

// ─── CodecRegistration ──────────────────────────────────────────────────────

/// Unified codec registration. Each format registers ONE of these.
///
/// Optional fields (None = not supported for that capability):
/// - `detect_fn` — auto-detect format from header bytes
/// - `decode_fn` — decode image data to pixels
/// - `encode_fn` — encode pixels to format
/// - `decode_lut_fn` — decode LUT file to ColorLut3D
/// - `encode_lut_fn` — encode ColorLut3D to format
pub struct CodecRegistration {
    /// Canonical format name (e.g., "png", "jpeg", "cube")
    pub format: &'static str,
    /// File extensions (e.g., &["jpg", "jpeg"])
    pub extensions: &'static [&'static str],
    /// MIME type (e.g., "image/png")
    pub mime: &'static str,

    /// Auto-detect: returns true if data matches this format's signature.
    /// None = format requires explicit hint (no magic bytes).
    pub detect_fn: Option<fn(&[u8]) -> bool>,
    /// Priority for detection ordering (lower = checked first).
    /// Binary magic: 10-50, structured: 60-100, text: 200+, weak: 300+.
    pub detection_priority: u32,

    /// Decode image data to pixels. None = decode not supported.
    pub decode_fn: Option<fn(&[u8]) -> Result<DecodedImage, ImageError>>,
    /// Encode pixels to format. None = encode not supported.
    pub encode_fn: Option<fn(&[u8], &ImageInfo, Option<u8>) -> Result<Vec<u8>, ImageError>>,

    /// Decode LUT file to ColorLut3D. None = not a LUT format.
    pub decode_lut_fn: Option<fn(&[u8]) -> Result<ColorLut3D, ImageError>>,
    /// Encode ColorLut3D to format. None = not a LUT format or encode not supported.
    pub encode_lut_fn: Option<fn(&ColorLut3D) -> Result<Vec<u8>, ImageError>>,
}

inventory::collect!(&'static CodecRegistration);

// ─── Lookup Functions ───────────────────────────────────────────────────────

/// Find a codec by format name or file extension.
pub fn find_codec(format: &str) -> Option<&'static CodecRegistration> {
    for reg in inventory::iter::<&'static CodecRegistration>.into_iter().copied() {
        if reg.format == format || reg.extensions.contains(&format) {
            return Some(reg);
        }
    }
    None
}

/// Detect format from data using registered detect_fn functions.
/// Returns the format name of the first match, sorted by detection_priority.
pub fn detect_format(data: &[u8]) -> Option<String> {
    let mut entries: Vec<&CodecRegistration> = inventory::iter::<&'static CodecRegistration>
        .into_iter()
        .copied()
        .filter(|r| r.detect_fn.is_some())
        .collect();
    entries.sort_by_key(|e| e.detection_priority);

    for entry in entries {
        if let Some(detect) = entry.detect_fn {
            if detect(data) {
                return Some(entry.format.to_string());
            }
        }
    }
    None
}

// ─── Image Decode ───────────────────────────────────────────────────────────

/// Decode image data with auto-detection.
pub fn decode(data: &[u8]) -> Result<DecodedImage, ImageError> {
    decode_with_hint(data, None)
}

/// Decode image data with an optional format hint.
///
/// - `None` — auto-detect via registered detect_fn functions.
/// - `Some("png")` — use the specified decoder directly. If the codec has
///   a detect_fn and it rejects the data, returns an error.
pub fn decode_with_hint(
    data: &[u8],
    format_hint: Option<&str>,
) -> Result<DecodedImage, ImageError> {
    if let Some(hint) = format_hint {
        if let Some(reg) = find_codec(hint) {
            if let Some(detect) = reg.detect_fn {
                if !detect(data) {
                    return Err(ImageError::InvalidInput(format!(
                        "data does not match format '{}' (detection rejected)",
                        reg.format
                    )));
                }
            }
            if let Some(decode) = reg.decode_fn {
                return decode(data);
            }
            return Err(ImageError::UnsupportedFormat(format!(
                "format '{}' does not support image decoding",
                reg.format
            )));
        }
        return Err(ImageError::UnsupportedFormat(format!(
            "no codec registered for format '{hint}'"
        )));
    }

    // Auto-detect
    let mut entries: Vec<&CodecRegistration> = inventory::iter::<&'static CodecRegistration>
        .into_iter()
        .copied()
        .collect();
    entries.sort_by_key(|e| e.detection_priority);

    for entry in &entries {
        if let Some(detect) = entry.detect_fn {
            if detect(data) {
                if let Some(decode) = entry.decode_fn {
                    return decode(data);
                }
            }
        }
    }

    let detected = detect_format(data).unwrap_or_else(|| "unknown".to_string());
    Err(ImageError::InvalidInput(format!(
        "decode: no image decoder matched (detected as '{detected}')"
    )))
}

// ─── Image Encode ───────────────────────────────────────────────────────────

/// Encode pixels to a format.
pub fn encode(
    pixels: &[u8],
    info: &ImageInfo,
    format: &str,
    quality: Option<u8>,
) -> Result<Vec<u8>, ImageError> {
    if let Some(reg) = find_codec(format) {
        if let Some(encode) = reg.encode_fn {
            return encode(pixels, info, quality);
        }
        return Err(ImageError::UnsupportedFormat(format!(
            "format '{}' does not support image encoding",
            reg.format
        )));
    }
    Err(ImageError::UnsupportedFormat(format!(
        "no codec registered for format '{format}'"
    )))
}

// ─── LUT Decode/Encode ─────────────────────────────────────────────────────

/// Check if a format is a registered LUT format (has encode_lut_fn or decode_lut_fn).
pub fn is_lut_format(format: &str) -> bool {
    find_codec(format).map_or(false, |r| r.encode_lut_fn.is_some() || r.decode_lut_fn.is_some())
}

/// Encode a ColorLut3D to a LUT format.
pub fn encode_lut(
    lut: &ColorLut3D,
    format: &str,
) -> Option<Result<Vec<u8>, ImageError>> {
    let reg = find_codec(format)?;
    let encode = reg.encode_lut_fn?;
    Some(encode(lut))
}

/// Decode a LUT file to ColorLut3D with auto-detection.
pub fn decode_lut(data: &[u8]) -> Result<ColorLut3D, ImageError> {
    decode_lut_with_hint(data, None)
}

/// Decode a LUT file with an optional format hint.
pub fn decode_lut_with_hint(
    data: &[u8],
    format_hint: Option<&str>,
) -> Result<ColorLut3D, ImageError> {
    if let Some(hint) = format_hint {
        if let Some(reg) = find_codec(hint) {
            if let Some(detect) = reg.detect_fn {
                if !detect(data) {
                    return Err(ImageError::InvalidInput(format!(
                        "data does not match LUT format '{}' (detection rejected)",
                        reg.format
                    )));
                }
            }
            if let Some(decode_lut) = reg.decode_lut_fn {
                return decode_lut(data);
            }
            return Err(ImageError::UnsupportedFormat(format!(
                "format '{}' does not support LUT decoding",
                reg.format
            )));
        }
        return Err(ImageError::UnsupportedFormat(format!(
            "no codec registered for LUT format '{hint}'"
        )));
    }

    // Auto-detect
    let mut entries: Vec<&CodecRegistration> = inventory::iter::<&'static CodecRegistration>
        .into_iter()
        .copied()
        .collect();
    entries.sort_by_key(|e| e.detection_priority);

    for entry in &entries {
        if let Some(detect) = entry.detect_fn {
            if detect(data) {
                if let Some(decode_lut) = entry.decode_lut_fn {
                    return decode_lut(data);
                }
            }
        }
    }

    Err(ImageError::InvalidInput(
        "decode_lut: no LUT decoder matched the data".into(),
    ))
}

// ─── Format Queries ─────────────────────────────────────────────────────────

/// List all registered format names.
pub fn all_formats() -> Vec<String> {
    let mut fmts: Vec<String> = inventory::iter::<&'static CodecRegistration>
        .into_iter()
        .copied()
        .map(|r| r.format.to_string())
        .collect();
    fmts.sort();
    fmts.dedup();
    fmts
}

/// List formats that support image decoding.
pub fn supported_decode_formats() -> Vec<String> {
    let mut fmts: Vec<String> = inventory::iter::<&'static CodecRegistration>
        .into_iter()
        .copied()
        .filter(|r| r.decode_fn.is_some())
        .map(|r| r.format.to_string())
        .collect();
    fmts.sort();
    fmts
}

/// List formats that support image encoding.
pub fn supported_encode_formats() -> Vec<String> {
    let mut fmts: Vec<String> = inventory::iter::<&'static CodecRegistration>
        .into_iter()
        .copied()
        .filter(|r| r.encode_fn.is_some())
        .map(|r| r.format.to_string())
        .collect();
    fmts.sort();
    fmts
}

/// List formats that support LUT encoding.
pub fn supported_lut_formats() -> Vec<String> {
    let mut fmts: Vec<String> = inventory::iter::<&'static CodecRegistration>
        .into_iter()
        .copied()
        .filter(|r| r.encode_lut_fn.is_some())
        .map(|r| r.format.to_string())
        .collect();
    fmts.sort();
    fmts
}

/// Check if a format supports decoding.
pub fn can_decode(format: &str) -> bool {
    find_codec(format).map_or(false, |r| r.decode_fn.is_some())
}

/// Check if a format supports encoding.
pub fn can_encode(format: &str) -> bool {
    find_codec(format).map_or(false, |r| r.encode_fn.is_some())
}

/// Get MIME type for a format.
pub fn mime_type(format: &str) -> Option<&'static str> {
    find_codec(format).map(|r| r.mime)
}

/// Get file extensions for a format.
pub fn extensions(format: &str) -> &'static [&'static str] {
    find_codec(format).map_or(&[], |r| r.extensions)
}

// Legacy compatibility wrapper
pub struct CodecRegistry;

impl CodecRegistry {
    pub fn detect_format(data: &[u8]) -> Option<&'static str> {
        detect_format(data).map(|s| Box::leak(s.into_boxed_str()) as &'static str)
    }
    pub fn decode(data: &[u8]) -> Result<DecodedImage, ImageError> {
        super::decoder::decode(data)
    }
    pub fn encode(
        pixels: &[u8],
        info: &ImageInfo,
        format: &str,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, ImageError> {
        super::encoder::encode(pixels, info, format, quality)
    }
    pub fn supported_decode_formats() -> Vec<String> {
        supported_decode_formats()
    }
    pub fn supported_encode_formats() -> Vec<String> {
        supported_encode_formats()
    }
    pub fn can_decode_format(format: &str) -> bool {
        can_decode(format)
    }
    pub fn can_encode_format(format: &str) -> bool {
        can_encode(format)
    }
    pub fn mime_type(format: &str) -> Option<&'static str> {
        mime_type(format)
    }
    pub fn extensions(format: &str) -> &'static [&'static str] {
        extensions(format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_codec_by_name() {
        assert!(find_codec("png").is_some());
        assert!(find_codec("jpeg").is_some());
        assert!(find_codec("nonexistent").is_none());
    }

    #[test]
    fn find_codec_by_extension() {
        let reg = find_codec("jpg").unwrap();
        assert_eq!(reg.format, "jpeg");
    }

    #[test]
    fn detect_png() {
        let data = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(detect_format(&data), Some("png".to_string()));
    }

    #[test]
    fn detect_jpeg() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0];
        assert_eq!(detect_format(&data), Some("jpeg".to_string()));
    }

    #[test]
    fn supported_formats_populated() {
        let decode = supported_decode_formats();
        let encode = supported_encode_formats();
        assert!(decode.len() >= 15, "expected >=15 decode formats, got {}", decode.len());
        assert!(encode.len() >= 10, "expected >=10 encode formats, got {}", encode.len());
    }

    #[test]
    fn lut_formats_populated() {
        let lut = supported_lut_formats();
        assert!(lut.contains(&"cube".to_string()));
    }

    #[test]
    fn can_decode_common() {
        assert!(can_decode("jpeg"));
        assert!(can_decode("png"));
        assert!(!can_decode("nonexistent"));
    }

    #[test]
    fn can_encode_common() {
        assert!(can_encode("jpeg"));
        assert!(can_encode("png"));
    }

    #[test]
    fn mime_types_correct() {
        assert_eq!(mime_type("jpeg"), Some("image/jpeg"));
        assert_eq!(mime_type("png"), Some("image/png"));
    }

    #[test]
    fn is_lut_format_check() {
        assert!(is_lut_format("cube"));
        assert!(!is_lut_format("png"));
    }
}
