use super::{apply_rgb_transform, ImageError, ImageInfo, PixelFormat};

/// Reinhard global tone mapping operator.
#[inline]
pub fn tonemap_reinhard_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (r / (1.0 + r), g / (1.0 + g), b / (1.0 + b))
}

/// Apply Reinhard tone mapping to an image buffer.
pub fn tonemap_reinhard(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, tonemap_reinhard_pixel)
}

/// Drago logarithmic tone mapping operator.
#[derive(Debug, Clone, Copy)]
pub struct DragoParams {
    /// Maximum luminance in the scene. Default: 1.0 (SDR).
    pub l_max: f32,
    /// Bias parameter (0.7–0.9). Higher = more contrast. Default: 0.85.
    pub bias: f32,
}

impl Default for DragoParams {
    fn default() -> Self {
        Self {
            l_max: 1.0,
            bias: 0.85,
        }
    }
}

/// Apply Drago tone mapping to a single pixel.
#[inline]
pub fn tonemap_drago_pixel(r: f32, g: f32, b: f32, params: &DragoParams) -> (f32, f32, f32) {
    let log_max = (1.0 + params.l_max).ln();
    let bias_pow = (params.bias.ln() / 0.5f32.ln()).max(0.01);

    #[inline]
    fn drago_channel(val: f32, log_max: f32, bias_pow: f32) -> f32 {
        if val <= 0.0 {
            return 0.0;
        }
        let mapped = (1.0 + val).ln() / log_max;
        mapped.powf(1.0 / bias_pow).clamp(0.0, 1.0)
    }

    (
        drago_channel(r, log_max, bias_pow),
        drago_channel(g, log_max, bias_pow),
        drago_channel(b, log_max, bias_pow),
    )
}

/// Apply Drago tone mapping to an image buffer.
pub fn tonemap_drago(
    pixels: &[u8],
    info: &ImageInfo,
    params: &DragoParams,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| tonemap_drago_pixel(r, g, b, params))
}

/// Filmic/ACES tone mapping (Narkowicz 2015 approximation).
#[derive(Debug, Clone, Copy)]
pub struct FilmicParams {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
}

impl Default for FilmicParams {
    fn default() -> Self {
        // Narkowicz 2015 ACES fit
        Self {
            a: 2.51,
            b: 0.03,
            c: 2.43,
            d: 0.59,
            e: 0.14,
        }
    }
}

/// Apply filmic/ACES tone mapping to a single pixel.
#[inline]
pub fn tonemap_filmic_pixel(r: f32, g: f32, b: f32, params: &FilmicParams) -> (f32, f32, f32) {
    #[inline]
    fn filmic(x: f32, p: &FilmicParams) -> f32 {
        let num = x * (p.a * x + p.b);
        let den = x * (p.c * x + p.d) + p.e;
        (num / den).clamp(0.0, 1.0)
    }

    (filmic(r, params), filmic(g, params), filmic(b, params))
}

/// Apply filmic/ACES tone mapping to an image buffer.
pub fn tonemap_filmic(
    pixels: &[u8],
    info: &ImageInfo,
    params: &FilmicParams,
) -> Result<Vec<u8>, ImageError> {
    apply_rgb_transform(pixels, info, |r, g, b| {
        tonemap_filmic_pixel(r, g, b, params)
    })
}

// ─── Film Grain Simulation ────────────────────────────────────────────────

/// Film grain simulation parameters.
#[derive(Debug, Clone, Copy)]
pub struct FilmGrainParams {
    /// Grain amount (0.0 = none, 1.0 = heavy). Default: 0.3.
    pub amount: f32,
    /// Grain size in pixels (1.0 = fine, 4.0+ = coarse). Default: 1.5.
    pub size: f32,
    /// Color grain (true) or monochrome grain (false). Default: false.
    pub color: bool,
    /// Random seed for deterministic output. Default: 0.
    pub seed: u32,
}

impl Default for FilmGrainParams {
    fn default() -> Self {
        Self {
            amount: 0.3,
            size: 1.5,
            color: false,
            seed: 0,
        }
    }
}

/// Generate deterministic pseudo-random noise using a hash function.
/// Returns a value in [-1.0, 1.0].
#[inline]
fn hash_noise(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(seed.wrapping_mul(1274126177));
    h = (h ^ (h >> 13)).wrapping_mul(1103515245);
    h = h ^ (h >> 16);
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Apply film grain to an image buffer.
pub fn film_grain(
    pixels: &[u8],
    info: &ImageInfo,
    params: &FilmGrainParams,
) -> Result<Vec<u8>, ImageError> {
    let bpp = match info.format {
        PixelFormat::Rgb8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "film grain requires RGB8 or RGBA8".into(),
            ));
        }
    };
    let w = info.width as usize;
    let h = info.height as usize;
    let expected = w * h * bpp;
    if pixels.len() < expected {
        return Err(ImageError::InvalidParameters("pixel data too small".into()));
    }

    let mut out = pixels.to_vec();
    let inv_size = 1.0 / params.size.max(0.1);

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * bpp;

            let sx = (x as f32 * inv_size) as u32;
            let sy = (y as f32 * inv_size) as u32;

            let r = pixels[idx] as f32 / 255.0;
            let g = pixels[idx + 1] as f32 / 255.0;
            let b = pixels[idx + 2] as f32 / 255.0;
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            let intensity = 4.0 * luma * (1.0 - luma) * params.amount;

            if params.color {
                let nr = hash_noise(sx, sy, params.seed) * intensity;
                let ng = hash_noise(sx, sy, params.seed.wrapping_add(1)) * intensity;
                let nb = hash_noise(sx, sy, params.seed.wrapping_add(2)) * intensity;
                out[idx] = ((r + nr) * 255.0).round().clamp(0.0, 255.0) as u8;
                out[idx + 1] = ((g + ng) * 255.0).round().clamp(0.0, 255.0) as u8;
                out[idx + 2] = ((b + nb) * 255.0).round().clamp(0.0, 255.0) as u8;
            } else {
                let n = hash_noise(sx, sy, params.seed) * intensity;
                out[idx] = ((r + n) * 255.0).round().clamp(0.0, 255.0) as u8;
                out[idx + 1] = ((g + n) * 255.0).round().clamp(0.0, 255.0) as u8;
                out[idx + 2] = ((b + n) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}
