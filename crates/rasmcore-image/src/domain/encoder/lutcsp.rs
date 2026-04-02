//! CineSpace .csp LUT encoder — serialize a ColorLut3D to .csp format.

use crate::domain::color_lut::ColorLut3D;
use crate::domain::error::ImageError;

/// Encode a ColorLut3D to CineSpace .csp format.
pub fn encode(lut: &ColorLut3D) -> Result<Vec<u8>, ImageError> {
    Ok(crate::domain::color_lut::serialize_csp(lut).into_bytes())
}

inventory::submit! {
    &crate::domain::encoder::StaticLutEncoderRegistration {
        format: "csp",
        extensions: &["csp"],
        encode_fn: encode,
    }
}
