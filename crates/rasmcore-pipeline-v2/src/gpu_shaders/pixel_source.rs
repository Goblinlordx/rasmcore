//! GPU shaders for host-decoded pixel source conversion.
//!
//! Converts raw pixel bytes (u8 sRGB, u16 linear) to f32 linear RGBA
//! as the first node in the GPU pipeline chain. f32 format needs no
//! conversion shader — data is used directly.

use crate::node::GpuShader;

/// Pixel format for host-decoded data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostPixelFormat {
    /// u8 RGBA sRGB (4 bytes/pixel).
    Rgba8,
    /// u16 RGBA linear (8 bytes/pixel, little-endian).
    Rgba16,
    /// f32 RGBA linear (16 bytes/pixel). No conversion needed.
    RgbaF32,
}

/// WGSL shader for u8 sRGB -> f32 linear conversion.
const RGBA8_SRGB_TO_F32: &str = include_str!("../shaders/rgba8_srgb_to_f32.wgsl");

/// WGSL shader for u16 linear -> f32 linear conversion.
const RGBA16_TO_F32: &str = include_str!("../shaders/rgba16_to_f32.wgsl");

/// Build a GPU conversion shader for the given pixel format.
///
/// Returns `None` for `RgbaF32` — no conversion needed, data uploads directly
/// as the f32 input buffer.
///
/// For u8/u16 formats, the raw pixel bytes are packed into a `u32` storage buffer
/// and passed as `extra_buffers[0]`. The shader unpacks, normalizes, and writes
/// `vec4<f32>` output.
pub fn conversion_shader(
    format: HostPixelFormat,
    raw_bytes: &[u8],
    width: u32,
    height: u32,
) -> Option<GpuShader> {
    match format {
        HostPixelFormat::Rgba8 => {
            let params = build_params(width, height);
            Some(
                GpuShader::new(
                    RGBA8_SRGB_TO_F32.to_string(),
                    "main",
                    [256, 1, 1],
                    params,
                )
                .with_extra_buffers(vec![raw_bytes.to_vec()]),
            )
        }
        HostPixelFormat::Rgba16 => {
            let params = build_params(width, height);
            Some(
                GpuShader::new(
                    RGBA16_TO_F32.to_string(),
                    "main",
                    [256, 1, 1],
                    params,
                )
                .with_extra_buffers(vec![raw_bytes.to_vec()]),
            )
        }
        HostPixelFormat::RgbaF32 => None,
    }
}

/// Build params buffer: [width: u32, height: u32].
fn build_params(width: u32, height: u32) -> Vec<u8> {
    let mut params = Vec::with_capacity(8);
    params.extend_from_slice(&width.to_le_bytes());
    params.extend_from_slice(&height.to_le_bytes());
    params
}
