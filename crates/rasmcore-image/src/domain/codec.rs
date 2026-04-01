//! Codec trait definitions and registry for extensible format support.
//!
//! Provides `ImageDecoder` and `ImageEncoder` traits that each codec implements,
//! plus a `CodecRegistry` that collects all registered codecs at compile time.
//!
//! Uses a simple const-array approach (not `inventory`) for WASM compatibility.

use super::error::ImageError;
use super::types::{DecodedImage, ImageInfo};

// ─── Decoder Trait ──────────────────────────────────────────────────────────

/// Trait for image format decoders.
///
/// Implementors detect their format via magic bytes and decode pixel data.
pub trait ImageDecoder: Send + Sync {
    /// Check if this decoder can handle the given data (magic byte detection).
    fn can_decode(&self, data: &[u8]) -> bool;

    /// Decode the image data into pixels + metadata.
    fn decode(&self, data: &[u8]) -> Result<DecodedImage, ImageError>;

    /// Short name for this codec (e.g., "jpeg", "png", "webp").
    fn name(&self) -> &'static str;

    /// File extensions this codec handles (e.g., &["jpg", "jpeg"]).
    fn extensions(&self) -> &'static [&'static str];

    /// MIME types this codec handles (e.g., &["image/jpeg"]).
    fn mime_types(&self) -> &'static [&'static str];

    /// Priority for format detection (lower = checked first). Default: 100.
    fn priority(&self) -> u32 {
        100
    }
}

// ─── Encoder Trait ──────────────────────────────────────────────────────────

/// Trait for image format encoders.
pub trait ImageEncoder: Send + Sync {
    /// Encode pixels into the format's byte representation.
    fn encode(
        &self,
        pixels: &[u8],
        info: &ImageInfo,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, ImageError>;

    /// Short name for this codec (e.g., "jpeg", "png").
    fn name(&self) -> &'static str;

    /// File extensions this codec produces (e.g., &["jpg", "jpeg"]).
    fn extensions(&self) -> &'static [&'static str];
}

// ─── Registry ───────────────────────────────────────────────────────────────

/// Codec descriptor for registration.
pub struct DecoderEntry {
    pub name: &'static str,
    pub extensions: &'static [&'static str],
    pub mime_types: &'static [&'static str],
    pub priority: u32,
    pub can_decode: fn(&[u8]) -> bool,
}

pub struct EncoderEntry {
    pub name: &'static str,
    pub extensions: &'static [&'static str],
}

/// Static codec registry.
///
/// Uses existing `decode()` and `encode()` functions. The registry provides
/// format detection and name-based lookup without changing the encode/decode
/// implementation.
pub struct CodecRegistry;

impl CodecRegistry {
    /// Detect format from header bytes.
    pub fn detect_format(data: &[u8]) -> Option<&'static str> {
        super::decoder::detect_format(data).map(|s| {
            // Leak the string for 'static — detect_format returns String
            // In practice this is called rarely and the strings are short
            Box::leak(s.into_boxed_str()) as &'static str
        })
    }

    /// Decode using the standard dispatch chain.
    pub fn decode(data: &[u8]) -> Result<DecodedImage, ImageError> {
        super::decoder::decode(data)
    }

    /// Encode using the standard dispatch chain.
    pub fn encode(
        pixels: &[u8],
        info: &ImageInfo,
        format: &str,
        quality: Option<u8>,
    ) -> Result<Vec<u8>, ImageError> {
        super::encoder::encode(pixels, info, format, quality)
    }

    /// List all supported decode format names.
    pub fn supported_decode_formats() -> &'static [&'static str] {
        &[
            "png", "jpeg", "gif", "webp", "bmp", "tiff", "avif", "qoi", "ico", "tga", "hdr", "pnm",
            "exr", "dds", "jxl", "jp2", "heic", "fits", "svg",
        ]
    }

    /// List all supported encode format names (derived from encoder registry).
    pub fn supported_encode_formats() -> Vec<&'static str> {
        super::encoder::registered_encoders().iter().map(|r| r.format).collect()
    }

    /// Check if a format name is supported for decoding.
    pub fn can_decode_format(format: &str) -> bool {
        Self::supported_decode_formats().contains(&format)
    }

    /// Check if a format name is supported for encoding.
    pub fn can_encode_format(format: &str) -> bool {
        super::encoder::registered_encoders().iter().any(|r| r.format == format || r.extensions.contains(&format))
    }

    /// Get MIME type for a format name.
    /// First checks the encoder registry, then falls back to decode-only formats.
    pub fn mime_type(format: &str) -> Option<&'static str> {
        // Check encoder registry first
        if let Some(reg) = super::encoder::registered_encoders().iter()
            .find(|r| r.format == format || r.extensions.contains(&format))
        {
            return Some(reg.mime);
        }
        // Decode-only formats (no encoder registered)
        match format {
            "svg" => Some("image/svg+xml"),
            "jxl" => Some("image/jxl"),
            _ => None,
        }
    }

    /// Get file extensions for a format name.
    /// First checks the encoder registry, then falls back to decode-only formats.
    pub fn extensions(format: &str) -> &'static [&'static str] {
        // Check encoder registry first
        if let Some(reg) = super::encoder::registered_encoders().iter()
            .find(|r| r.format == format)
        {
            return reg.extensions;
        }
        // Decode-only formats
        match format {
            "svg" => &["svg", "svgz"],
            "jxl" => &["jxl"],
            _ => &[],
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_formats_nonempty() {
        assert!(CodecRegistry::supported_decode_formats().len() >= 15);
        assert!(CodecRegistry::supported_encode_formats().len() >= 10);
    }

    #[test]
    fn can_decode_common_formats() {
        assert!(CodecRegistry::can_decode_format("jpeg"));
        assert!(CodecRegistry::can_decode_format("png"));
        assert!(CodecRegistry::can_decode_format("webp"));
        assert!(!CodecRegistry::can_decode_format("nonexistent"));
    }

    #[test]
    fn can_encode_common_formats() {
        assert!(CodecRegistry::can_encode_format("jpeg"));
        assert!(CodecRegistry::can_encode_format("png"));
        assert!(!CodecRegistry::can_encode_format("svg")); // SVG is decode-only
    }

    #[test]
    fn mime_types() {
        assert_eq!(CodecRegistry::mime_type("jpeg"), Some("image/jpeg"));
        assert_eq!(CodecRegistry::mime_type("png"), Some("image/png"));
        assert_eq!(CodecRegistry::mime_type("svg"), Some("image/svg+xml"));
        assert_eq!(CodecRegistry::mime_type("nonexistent"), None);
    }

    #[test]
    fn extensions() {
        assert_eq!(CodecRegistry::extensions("jpeg"), &["jpg", "jpeg"]);
        assert_eq!(
            CodecRegistry::extensions("pnm"),
            &["pnm", "ppm", "pgm", "pbm"]
        );
    }

    #[test]
    fn detect_png() {
        let png_header = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let format = CodecRegistry::detect_format(&png_header);
        assert_eq!(format, Some("png"));
    }

    #[test]
    fn detect_jpeg() {
        let jpeg_header = [0xFF, 0xD8, 0xFF, 0xE0];
        let format = CodecRegistry::detect_format(&jpeg_header);
        assert_eq!(format, Some("jpeg"));
    }

    #[test]
    fn decode_and_encode_roundtrip() {
        // Create a simple PNG via encode, then decode it
        let pixels = vec![128u8; 4 * 4 * 3]; // 4x4 RGB
        let info = ImageInfo {
            width: 4,
            height: 4,
            format: super::super::types::PixelFormat::Rgb8,
            color_space: super::super::types::ColorSpace::Srgb,
        };

        let png_data = CodecRegistry::encode(&pixels, &info, "png", None).unwrap();
        let decoded = CodecRegistry::decode(&png_data).unwrap();
        assert_eq!(decoded.info.width, 4);
        assert_eq!(decoded.info.height, 4);
    }
}
