//! Native trivial codec encoders (QOI, PNM, BMP, TGA).
//!
//! Each function is feature-gated and only compiled when the corresponding
//! native-* feature is enabled in Cargo.toml.

use super::super::error::ImageError;
use super::super::types::{ImageInfo, PixelFormat};

#[cfg(feature = "native-qoi")]
pub fn encode_qoi(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    let channels = match info.format {
        PixelFormat::Rgba8 => rasmcore_qoi::Channels::Rgba,
        PixelFormat::Rgb8 => rasmcore_qoi::Channels::Rgb,
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "QOI native encode does not support {other:?}"
            )));
        }
    };
    rasmcore_qoi::encode(
        pixels,
        info.width,
        info.height,
        channels,
        rasmcore_qoi::ColorSpace::Srgb,
    )
    .map_err(|e| ImageError::ProcessingFailed(format!("QOI encode: {e}")))
}

#[cfg(feature = "native-pnm")]
pub fn encode_pnm(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    match info.format {
        PixelFormat::Rgb8 => rasmcore_pnm::encode_ppm(pixels, info.width, info.height)
            .map_err(|e| ImageError::ProcessingFailed(format!("PNM encode: {e}"))),
        PixelFormat::Gray8 => rasmcore_pnm::encode_pgm(pixels, info.width, info.height)
            .map_err(|e| ImageError::ProcessingFailed(format!("PNM encode: {e}"))),
        other => Err(ImageError::UnsupportedFormat(format!(
            "PNM native encode does not support {other:?}"
        ))),
    }
}

#[cfg(feature = "native-bmp")]
pub fn encode_bmp(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    match info.format {
        PixelFormat::Rgb8 => rasmcore_bmp::encode_rgb(pixels, info.width, info.height)
            .map_err(|e| ImageError::ProcessingFailed(format!("BMP encode: {e}"))),
        PixelFormat::Gray8 => rasmcore_bmp::encode_gray(pixels, info.width, info.height)
            .map_err(|e| ImageError::ProcessingFailed(format!("BMP encode: {e}"))),
        other => Err(ImageError::UnsupportedFormat(format!(
            "BMP native encode does not support {other:?}"
        ))),
    }
}

// ─── Encoder Registrations for TGA and PNM ─────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "tga",
        format: "tga",
        mime: "image/x-tga",
        extensions: &["tga"],
        fn_name: "encode_tga",
    }
}

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "pnm",
        format: "pnm",
        mime: "image/x-portable-anymap",
        extensions: &["pnm", "ppm", "pgm", "pbm"],
        fn_name: "encode_pnm",
    }
}

#[cfg(feature = "native-tga")]
pub fn encode_tga(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    let w = info.width as u16;
    let h = info.height as u16;
    match info.format {
        PixelFormat::Rgb8 => rasmcore_tga::encode_rgb(pixels, w, h)
            .map_err(|e| ImageError::ProcessingFailed(format!("TGA encode: {e}"))),
        PixelFormat::Rgba8 => rasmcore_tga::encode_rgba(pixels, w, h)
            .map_err(|e| ImageError::ProcessingFailed(format!("TGA encode: {e}"))),
        PixelFormat::Gray8 => rasmcore_tga::encode_gray(pixels, w, h)
            .map_err(|e| ImageError::ProcessingFailed(format!("TGA encode: {e}"))),
        other => Err(ImageError::UnsupportedFormat(format!(
            "TGA native encode does not support {other:?}"
        ))),
    }
}
