//! DDS (DirectDraw Surface) encoder — uncompressed RGBA.
//!
//! The image crate has no DDS encoder, so we write a minimal one.
//! Outputs DDS with DX10 extended header for uncompressed R8G8B8A8 data.

use crate::domain::error::ImageError;
use crate::domain::types::{ImageInfo, PixelFormat};

/// DDS encode configuration (no parameters — uncompressed only).
#[derive(Debug, Clone, Default)]
pub struct DdsEncodeConfig;

/// Encode pixels to DDS format.
pub fn encode_pixels(pixels: &[u8], info: &ImageInfo, _config: &DdsEncodeConfig) -> Result<Vec<u8>, ImageError> {
    encode_dds(pixels, info)
}

// DDS magic number
const DDS_MAGIC: u32 = 0x20534444; // "DDS "

// DDS header flags
const DDSD_CAPS: u32 = 0x1;
const DDSD_HEIGHT: u32 = 0x2;
const DDSD_WIDTH: u32 = 0x4;
const DDSD_PITCH: u32 = 0x8;
const DDSD_PIXELFORMAT: u32 = 0x1000;

// DDS pixel format flags
const DDPF_RGB: u32 = 0x40;
const DDPF_ALPHAPIXELS: u32 = 0x1;

// DDS caps
const DDSCAPS_TEXTURE: u32 = 0x1000;

/// Encode raw pixels as uncompressed DDS (RGBA8 or RGB8).
///
/// Input is converted to RGBA8 for the DDS container. DDS stores pixels
/// in BGRA order, so we swizzle R and B channels.
pub fn encode_dds(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    let (w, h) = (info.width, info.height);

    // Convert to RGBA8 pixel data in BGRA byte order (DDS convention)
    let bgra_pixels = match info.format {
        PixelFormat::Rgba8 => {
            let mut bgra = pixels.to_vec();
            for chunk in bgra.chunks_exact_mut(4) {
                chunk.swap(0, 2); // R <-> B
            }
            bgra
        }
        PixelFormat::Rgb8 => {
            let mut bgra = Vec::with_capacity((w * h * 4) as usize);
            for chunk in pixels.chunks_exact(3) {
                bgra.push(chunk[2]); // B
                bgra.push(chunk[1]); // G
                bgra.push(chunk[0]); // R
                bgra.push(255); // A
            }
            bgra
        }
        PixelFormat::Gray8 => {
            let mut bgra = Vec::with_capacity((w * h * 4) as usize);
            for &v in pixels {
                bgra.push(v); // B
                bgra.push(v); // G
                bgra.push(v); // R
                bgra.push(255); // A
            }
            bgra
        }
        other => {
            return Err(ImageError::UnsupportedFormat(format!(
                "DDS encoding from {other:?} not supported"
            )));
        }
    };

    let pitch = w * 4; // bytes per row for 32-bit RGBA
    let mut out = Vec::with_capacity(128 + bgra_pixels.len());

    // DDS magic
    out.extend_from_slice(&DDS_MAGIC.to_le_bytes());

    // DDS_HEADER (124 bytes)
    out.extend_from_slice(&124u32.to_le_bytes()); // dwSize
    out.extend_from_slice(
        &(DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT).to_le_bytes(),
    ); // dwFlags
    out.extend_from_slice(&h.to_le_bytes()); // dwHeight
    out.extend_from_slice(&w.to_le_bytes()); // dwWidth
    out.extend_from_slice(&pitch.to_le_bytes()); // dwPitchOrLinearSize
    out.extend_from_slice(&0u32.to_le_bytes()); // dwDepth
    out.extend_from_slice(&0u32.to_le_bytes()); // dwMipMapCount
    out.extend_from_slice(&[0u8; 44]); // dwReserved1[11]

    // DDS_PIXELFORMAT (32 bytes)
    out.extend_from_slice(&32u32.to_le_bytes()); // dwSize
    out.extend_from_slice(&(DDPF_RGB | DDPF_ALPHAPIXELS).to_le_bytes()); // dwFlags
    out.extend_from_slice(&0u32.to_le_bytes()); // dwFourCC (not used for uncompressed)
    out.extend_from_slice(&32u32.to_le_bytes()); // dwRGBBitCount
    out.extend_from_slice(&0x00FF0000u32.to_le_bytes()); // dwRBitMask
    out.extend_from_slice(&0x0000FF00u32.to_le_bytes()); // dwGBitMask
    out.extend_from_slice(&0x000000FFu32.to_le_bytes()); // dwBBitMask
    out.extend_from_slice(&0xFF000000u32.to_le_bytes()); // dwABitMask

    // dwCaps
    out.extend_from_slice(&DDSCAPS_TEXTURE.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // dwCaps2
    out.extend_from_slice(&0u32.to_le_bytes()); // dwCaps3
    out.extend_from_slice(&0u32.to_le_bytes()); // dwCaps4
    out.extend_from_slice(&0u32.to_le_bytes()); // dwReserved2

    debug_assert_eq!(out.len(), 128, "DDS header must be exactly 128 bytes");

    // Pixel data
    out.extend_from_slice(&bgra_pixels);

    Ok(out)
}


// ─── Encoder Registration ──────────────────────────────────────────────────

inventory::submit! {
    &crate::domain::encoder::StaticEncoderRegistration {
        name: "dds",
        format: "dds",
        mime: "image/vnd-ms.dds",
        extensions: &["dds"],
        fn_name: "encode_dds",
        encode_fn: None,
        preferred_output_cs: crate::domain::encoder::EncoderColorSpace::Srgb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    #[test]
    fn encode_dds_produces_valid_header() {
        let pixels = vec![128u8; 8 * 8 * 3];
        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_dds(&pixels, &info).unwrap();
        // Check DDS magic
        assert_eq!(&result[..4], b"DDS ");
        // Header size = 124
        assert_eq!(
            u32::from_le_bytes([result[4], result[5], result[6], result[7]]),
            124
        );
        // Total size = 128 header + 8*8*4 pixel data
        assert_eq!(result.len(), 128 + 8 * 8 * 4);
    }

    #[test]
    fn encode_dds_rgba8_swizzles() {
        let pixels = vec![255, 0, 0, 128]; // One RGBA pixel: R=255, G=0, B=0, A=128
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let result = encode_dds(&pixels, &info).unwrap();
        // Pixel data starts at offset 128, should be BGRA: B=0, G=0, R=255, A=128
        assert_eq!(&result[128..132], &[0, 0, 255, 128]);
    }
}
