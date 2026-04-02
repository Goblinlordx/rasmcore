//! Hald CLUT encoder — serialize a ColorLut3D to a Hald CLUT PNG image.

use crate::domain::color_lut::ColorLut3D;
use crate::domain::error::ImageError;

/// Encode a ColorLut3D to a Hald CLUT PNG image.
///
/// Generates a square RGB8 pixel buffer from the CLUT, then encodes as PNG.
pub fn encode(lut: &ColorLut3D) -> Result<Vec<u8>, ImageError> {
    let (pixels, w, h) = crate::domain::color_lut::serialize_hald(lut);
    let info = crate::domain::types::ImageInfo {
        width: w,
        height: h,
        format: crate::domain::types::PixelFormat::Rgb8,
        color_space: crate::domain::types::ColorSpace::Srgb,
    };
    super::encode(&pixels, &info, "png", None)
}

inventory::submit! {
    &crate::domain::encoder::StaticLutEncoderRegistration {
        format: "hald",
        extensions: &["hald"],
        encode_fn: encode,
    }
}
