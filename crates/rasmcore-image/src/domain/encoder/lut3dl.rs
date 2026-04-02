//! Autodesk .3dl LUT encoder — serialize a ColorLut3D to .3dl format.

use crate::domain::color_lut::ColorLut3D;
use crate::domain::error::ImageError;

/// Encode a ColorLut3D to Autodesk .3dl format (10-bit integer triplets).
pub fn encode(lut: &ColorLut3D) -> Result<Vec<u8>, ImageError> {
    Ok(crate::domain::color_lut::serialize_3dl(lut).into_bytes())
}

inventory::submit! {
    &crate::domain::encoder::StaticLutEncoderRegistration {
        format: "3dl",
        extensions: &["3dl"],
        encode_fn: encode,
    }
}
