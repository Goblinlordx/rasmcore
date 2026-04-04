//! Filter: glitch (category: effect)
//!
//! Digital glitch / bad TV effect: horizontal scanline displacement with
//! RGB channel offset and optional noise band corruption.
//! Reference: PicsArt/Pixlr glitch effect.

#[allow(unused_imports)]
use crate::domain::filters::common::*;
use crate::domain::filter_traits::CpuFilter;

/// Deterministic hash noise — returns value in [-1.0, 1.0].
#[inline]
fn hash_noise(x: u32, seed: u32) -> f32 {
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(seed.wrapping_mul(1274126177));
    h = (h ^ (h >> 13)).wrapping_mul(1103515245);
    h = h ^ (h >> 16);
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Parameters for the glitch effect.
#[derive(rasmcore_macros::Filter, Clone)]
#[filter(name = "glitch", category = "effect", reference = "digital glitch / bad TV scanline displacement")]
pub struct GlitchParams {
    /// Maximum horizontal displacement in pixels
    #[param(min = 1.0, max = 100.0, step = 1.0, default = 20.0)]
    pub shift_amount: f32,
    /// RGB channel separation offset in pixels
    #[param(min = 0.0, max = 30.0, step = 1.0, default = 5.0)]
    pub channel_offset: f32,
    /// Fraction of scanlines affected (0.0-1.0)
    #[param(min = 0.0, max = 1.0, step = 0.05, default = 0.3)]
    pub intensity: f32,
    /// Band height range in scanlines
    #[param(min = 1, max = 50, step = 1, default = 8)]
    pub band_height: u32,
    /// Random seed for deterministic output
    #[param(min = 0, max = 999999, step = 1, default = 42, hint = "rc.seed")]
    pub seed: u32,
}

const GLITCH_WGSL: &str = include_str!("../../../shaders/glitch.wgsl");

impl rasmcore_pipeline::GpuCapable for GlitchParams {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::GpuOp>> {
        let mut params = Vec::with_capacity(32);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&self.shift_amount.to_le_bytes());
        params.extend_from_slice(&self.channel_offset.to_le_bytes());
        params.extend_from_slice(&self.intensity.to_le_bytes());
        params.extend_from_slice(&self.band_height.to_le_bytes());
        params.extend_from_slice(&self.seed.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes()); // _pad

        Some(vec![rasmcore_pipeline::GpuOp::Compute {
            shader: rasmcore_gpu_shaders::compose(&[rasmcore_gpu_shaders::PIXEL_OPS], GLITCH_WGSL),
            entry_point: "main",
            workgroup_size: [16, 16, 1],
            params,
            extra_buffers: Vec::new(),
            buffer_format: rasmcore_pipeline::BufferFormat::U32Packed,
        }])
    }
}

impl CpuFilter for GlitchParams {
    fn compute(
        &self,
        request: Rect,
        upstream: &mut (dyn FnMut(Rect) -> Result<Vec<u8>, ImageError> + '_),
        info: &ImageInfo,
    ) -> Result<Vec<u8>, ImageError> {
    let pixels = upstream(request)?;
    let info = &ImageInfo {
        width: request.width,
        height: request.height,
        ..*info
    };
    let pixels = pixels.as_slice();
    validate_format(info.format)?;

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            self.compute(r, &mut u, i8)
        });
    }
    if is_f32(info.format) {
        return process_via_standard(pixels, info, |p8, i8| {
            let r = Rect::new(0, 0, i8.width, i8.height);
            let mut u = |_: Rect| Ok(p8.to_vec());
            self.compute(r, &mut u, i8)
        });
    }

    let w = info.width as usize;
    let h = info.height as usize;
    let ch = channels(info.format);
    if ch < 3 {
        return Err(ImageError::UnsupportedFormat("glitch requires RGB".into()));
    }

    let shift_amount = self.shift_amount;
    let channel_offset = self.channel_offset as isize;
    let intensity = self.intensity;
    let band_h = self.band_height.max(1) as usize;
    let seed = self.seed;

    let mut out = pixels.to_vec();

    // Process in horizontal bands
    let mut y = 0;
    let mut band_idx = 0u32;
    while y < h {
        let band_end = (y + band_h).min(h);

        // Decide if this band is glitched (based on deterministic hash)
        let band_noise = hash_noise(band_idx, seed);
        let is_glitched = band_noise.abs() < intensity;

        if is_glitched {
            // Horizontal shift amount for this band
            let shift = (hash_noise(band_idx, seed.wrapping_add(100)) * shift_amount) as isize;

            for row in y..band_end {
                for x in 0..w {
                    let dst_idx = (row * w + x) * ch;

                    // Shifted source for RGB channels with per-channel offset
                    // Each channel has a distinct source x (chromatic aberration), so
                    // copy_from_slice cannot be used here — suppress the lint.
                    #[allow(clippy::manual_memcpy)]
                    for c in 0..3 {
                        let c_offset = match c {
                            0 => channel_offset,  // Red shifts extra
                            2 => -channel_offset, // Blue shifts opposite
                            _ => 0,               // Green stays with main shift
                        };
                        let sx = (x as isize + shift + c_offset).clamp(0, w as isize - 1) as usize;
                        let src_idx = (row * w + sx) * ch;
                        out[dst_idx + c] = pixels[src_idx + c];
                    }
                    // Alpha unchanged
                    if ch == 4 {
                        out[dst_idx + 3] = pixels[(row * w + x) * ch + 3];
                    }
                }
            }
        }

        y = band_end;
        band_idx += 1;
    }

    Ok(out)
}
}

