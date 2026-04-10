use super::super::error::ImageError;
use super::super::types::{ImageInfo, PixelFormat};

/// FITS encode configuration (no parameters).
#[derive(Debug, Clone, Default)]
pub struct FitsEncodeConfig;

/// Encode pixel data to FITS format (with config for uniform API).
pub fn encode(pixels: &[u8], info: &ImageInfo, _config: &FitsEncodeConfig) -> Result<Vec<u8>, ImageError> {
    encode_pixels(pixels, info)
}

/// Encode pixel data to FITS format.
pub fn encode_pixels(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    match info.format {
        PixelFormat::Gray8 => rasmcore_fits::encode_u8(pixels, info.width, info.height)
            .map_err(|e| ImageError::ProcessingFailed(format!("FITS encode: {e}"))),
        PixelFormat::Rgb8 => {
            // FITS is typically grayscale — extract luma from RGB
            let gray: Vec<u8> = pixels
                .chunks_exact(3)
                .map(|rgb| {
                    (0.299 * rgb[0] as f64 + 0.587 * rgb[1] as f64 + 0.114 * rgb[2] as f64).round()
                        as u8
                })
                .collect();
            rasmcore_fits::encode_u8(&gray, info.width, info.height)
                .map_err(|e| ImageError::ProcessingFailed(format!("FITS encode: {e}")))
        }
        other => Err(ImageError::UnsupportedFormat(format!(
            "FITS encode does not support {other:?} pixel format"
        ))),
    }
}

// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "fits",
        format: "fits",
        mime: "image/fits",
        extensions: &["fits", "fit"],
        fn_name: "encode_fits",
        encode_fn: None,
        preferred_output_cs: crate::domain::encoder::EncoderColorSpace::Linear,
    }
}
