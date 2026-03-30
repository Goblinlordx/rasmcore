//! QOI encoder fallback via image crate.
//!
//! This module is only used when native-qoi is NOT enabled.
//! With native-qoi (default), the dispatch in mod.rs routes to
//! native_trivial::encode_qoi() instead.

/// QOI encode configuration. QOI is a fixed lossless format with no parameters.
#[derive(Debug, Clone, Default)]
pub struct QoiEncodeConfig;

/// Encode pixel data to QOI via image crate (fallback when native-qoi is disabled).
#[cfg(not(feature = "native-qoi"))]
pub fn encode(
    img: &image::DynamicImage,
    _info: &crate::domain::types::ImageInfo,
    _config: &QoiEncodeConfig,
) -> Result<Vec<u8>, crate::domain::error::ImageError> {
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::Qoi)
        .map_err(|e| crate::domain::error::ImageError::ProcessingFailed(e.to_string()))?;
    Ok(buf)
}
