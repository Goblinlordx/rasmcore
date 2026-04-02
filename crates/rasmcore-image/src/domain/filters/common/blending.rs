//! Blending helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Apply per-pixel blend formula in normalized [0, 1] space.
///
/// `a` = foreground channel, `b` = background channel.
/// See [`BlendMode`] variants for individual formulas and validation status.
#[inline]
pub fn blend_channel(a: u8, b: u8, mode: BlendMode) -> u8 {
    let af = a as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let result = match mode {
        BlendMode::Multiply => af * bf,
        BlendMode::Screen => 1.0 - (1.0 - af) * (1.0 - bf),
        BlendMode::Overlay => {
            if bf < 0.5 {
                2.0 * af * bf
            } else {
                1.0 - 2.0 * (1.0 - af) * (1.0 - bf)
            }
        }
        BlendMode::Darken => af.min(bf),
        BlendMode::Lighten => af.max(bf),
        BlendMode::SoftLight => {
            if af < 0.5 {
                bf - (1.0 - 2.0 * af) * bf * (1.0 - bf)
            } else {
                let d = if bf <= 0.25 {
                    ((16.0 * bf - 12.0) * bf + 4.0) * bf
                } else {
                    bf.sqrt()
                };
                bf + (2.0 * af - 1.0) * (d - bf)
            }
        }
        BlendMode::HardLight => {
            if af < 0.5 {
                2.0 * af * bf
            } else {
                1.0 - 2.0 * (1.0 - af) * (1.0 - bf)
            }
        }
        BlendMode::Difference => (af - bf).abs(),
        BlendMode::Exclusion => af + bf - 2.0 * af * bf,
        BlendMode::ColorDodge => {
            if af >= 1.0 {
                1.0
            } else if bf == 0.0 {
                0.0
            } else {
                (bf / (1.0 - af)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if af <= 0.0 {
                0.0
            } else if bf >= 1.0 {
                1.0
            } else {
                1.0 - ((1.0 - bf) / af).min(1.0)
            }
        }
        BlendMode::VividLight => {
            // ColorBurn for a < 0.5, ColorDodge for a >= 0.5
            if af <= 0.5 {
                let a2 = 2.0 * af;
                if a2 <= 0.0 {
                    0.0
                } else if bf >= 1.0 {
                    1.0
                } else {
                    1.0 - ((1.0 - bf) / a2).min(1.0)
                }
            } else {
                let a2 = 2.0 * (af - 0.5);
                if a2 >= 1.0 {
                    1.0
                } else if bf == 0.0 {
                    0.0
                } else {
                    (bf / (1.0 - a2)).min(1.0)
                }
            }
        }
        BlendMode::LinearDodge => af + bf,      // clamped below
        BlendMode::LinearBurn => af + bf - 1.0, // clamped below
        BlendMode::LinearLight => {
            // LinearBurn for a < 0.5, LinearDodge for a >= 0.5
            bf + 2.0 * af - 1.0
        }
        BlendMode::PinLight => {
            if af <= 0.5 {
                bf.min(2.0 * af)
            } else {
                bf.max(2.0 * af - 1.0)
            }
        }
        BlendMode::HardMix => {
            // Threshold the VividLight result at 0.5.
            // Note: the simplified `a + b >= 1` is wrong at fg=0, bg=255
            // where VividLight(0,1) = ColorBurn(0,1) = 0, so threshold → 0.
            let vl = if af <= 0.5 {
                let a2 = 2.0 * af;
                if a2 <= 0.0 {
                    0.0
                } else if bf >= 1.0 {
                    1.0
                } else {
                    1.0 - ((1.0 - bf) / a2).min(1.0)
                }
            } else {
                let a2 = 2.0 * (af - 0.5);
                if a2 >= 1.0 {
                    1.0
                } else if bf == 0.0 {
                    0.0
                } else {
                    (bf / (1.0 - a2)).min(1.0)
                }
            };
            if vl >= 0.5 { 1.0 } else { 0.0 }
        }
        BlendMode::Subtract => bf - af, // clamped below
        BlendMode::Divide => {
            if af <= 0.0 {
                1.0
            } else {
                (bf / af).min(1.0)
            }
        }
        // Pixel-level modes — handled in blend(), not here.
        BlendMode::Dissolve
        | BlendMode::DarkerColor
        | BlendMode::LighterColor
        | BlendMode::Hue
        | BlendMode::Saturation
        | BlendMode::Color
        | BlendMode::Luminosity => unreachable!("pixel-level mode in blend_channel"),
    };
    (result.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Smoothstep function for blend-if feathering.
pub fn blend_if_smoothstep(x: f32, edge0: f32, edge1: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Blend a single pixel using a pixel-level (non-per-channel) mode.
///
/// `fg` and `bg` are RGB slices (3 bytes). `px_idx` is the pixel index
/// used for Dissolve's deterministic hash.
#[inline]
pub fn blend_pixel(fg: &[u8], bg: &[u8], mode: BlendMode, px_idx: u32) -> (u8, u8, u8) {
    match mode {
        BlendMode::Dissolve => {
            // Deterministic hash-based dither: hash the pixel index to get
            // a pseudo-random threshold. If threshold < 128, show fg; else bg.
            // This matches PS behavior for fully-opaque layers (alpha=1.0 → always fg).
            // For the general compositing path (RGBA with partial alpha), the
            // composite node pre-multiplies, so Dissolve on fully-opaque RGB
            // always selects the foreground pixel.
            let hash = px_idx
                .wrapping_mul(2654435761) // Knuth multiplicative hash
                .wrapping_shr(16) as u8;
            if hash < 128 {
                (fg[0], fg[1], fg[2])
            } else {
                (bg[0], bg[1], bg[2])
            }
        }
        BlendMode::DarkerColor => {
            let fg_lum = pixel_luminance(fg[0], fg[1], fg[2]);
            let bg_lum = pixel_luminance(bg[0], bg[1], bg[2]);
            if fg_lum <= bg_lum {
                (fg[0], fg[1], fg[2])
            } else {
                (bg[0], bg[1], bg[2])
            }
        }
        BlendMode::LighterColor => {
            let fg_lum = pixel_luminance(fg[0], fg[1], fg[2]);
            let bg_lum = pixel_luminance(bg[0], bg[1], bg[2]);
            if fg_lum >= bg_lum {
                (fg[0], fg[1], fg[2])
            } else {
                (bg[0], bg[1], bg[2])
            }
        }
        BlendMode::Hue => {
            // W3C: SetLum(SetSat(Cs, Sat(Cb)), Lum(Cb))
            let (sr, sg, sb) = (
                fg[0] as f32 / 255.0,
                fg[1] as f32 / 255.0,
                fg[2] as f32 / 255.0,
            );
            let (br, bg_g, bb) = (
                bg[0] as f32 / 255.0,
                bg[1] as f32 / 255.0,
                bg[2] as f32 / 255.0,
            );
            let (r, g, b) = set_lum_sat(sr, sg, sb, sat(br, bg_g, bb), lum(br, bg_g, bb));
            to_u8_triple(r, g, b)
        }
        BlendMode::Saturation => {
            // W3C: SetLum(SetSat(Cb, Sat(Cs)), Lum(Cb))
            let (sr, sg, sb) = (
                fg[0] as f32 / 255.0,
                fg[1] as f32 / 255.0,
                fg[2] as f32 / 255.0,
            );
            let (br, bg_g, bb) = (
                bg[0] as f32 / 255.0,
                bg[1] as f32 / 255.0,
                bg[2] as f32 / 255.0,
            );
            let (r, g, b) = set_lum_sat(br, bg_g, bb, sat(sr, sg, sb), lum(br, bg_g, bb));
            to_u8_triple(r, g, b)
        }
        BlendMode::Color => {
            // W3C: SetLum(Cs, Lum(Cb))
            let (sr, sg, sb) = (
                fg[0] as f32 / 255.0,
                fg[1] as f32 / 255.0,
                fg[2] as f32 / 255.0,
            );
            let (br, bg_g, bb) = (
                bg[0] as f32 / 255.0,
                bg[1] as f32 / 255.0,
                bg[2] as f32 / 255.0,
            );
            let (r, g, b) = set_lum(sr, sg, sb, lum(br, bg_g, bb));
            to_u8_triple(r, g, b)
        }
        BlendMode::Luminosity => {
            // W3C: SetLum(Cb, Lum(Cs))
            let (sr, sg, sb) = (
                fg[0] as f32 / 255.0,
                fg[1] as f32 / 255.0,
                fg[2] as f32 / 255.0,
            );
            let (br, bg_g, bb) = (
                bg[0] as f32 / 255.0,
                bg[1] as f32 / 255.0,
                bg[2] as f32 / 255.0,
            );
            let (r, g, b) = set_lum(br, bg_g, bb, lum(sr, sg, sb));
            to_u8_triple(r, g, b)
        }
        _ => unreachable!("per-channel mode in blend_pixel"),
    }
}

/// Shared implementation for dodge and burn.
pub fn dodge_burn_impl(
    pixels: &[u8],
    info: &ImageInfo,
    exposure: f32,
    range: u32,
    is_dodge: bool,
) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;

    // Identity fast path
    if exposure.abs() < 1e-6 {
        return Ok(pixels.to_vec());
    }

    // Shared per-pixel math for dodge/burn
    #[inline]
    fn dodge_burn_pixel(r: f32, g: f32, b: f32, exposure: f32, range: u32, is_dodge: bool) -> (f32, f32, f32) {
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let weight = match range {
            0 => { let t = (luma * 2.0).min(1.0); 1.0 - t * t * (3.0 - 2.0 * t) }
            2 => { let t = ((luma - 0.5) * 2.0).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }
            _ => (4.0 * luma * (1.0 - luma)).min(1.0),
        };
        let factor = exposure * weight;
        if is_dodge {
            (r + r * factor, g + g * factor, b + b * factor)
        } else {
            (r * (1.0 - factor), g * (1.0 - factor), b * (1.0 - factor))
        }
    }

    // f32 path: samples are already in [0,1]
    if is_f32(info.format) {
        let ch = channels(info.format);
        if ch < 3 {
            return Err(ImageError::UnsupportedFormat("dodge/burn requires RGB".into()));
        }
        let mut samples: Vec<f32> = pixels
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for chunk in samples.chunks_exact_mut(ch) {
            let (nr, ng, nb) = dodge_burn_pixel(chunk[0], chunk[1], chunk[2], exposure, range, is_dodge);
            chunk[0] = nr.clamp(0.0, 1.0);
            chunk[1] = ng.clamp(0.0, 1.0);
            chunk[2] = nb.clamp(0.0, 1.0);
        }
        return Ok(samples.iter().flat_map(|v| v.to_le_bytes()).collect());
    }

    if is_16bit(info.format) {
        return process_via_8bit(pixels, info, |p8, i8| {
            dodge_burn_impl(p8, i8, exposure, range, is_dodge)
        });
    }

    let ch = channels(info.format);
    if ch < 3 {
        return Err(ImageError::UnsupportedFormat(
            "dodge/burn requires RGB8 or RGBA8".into(),
        ));
    }

    let n = (info.width as usize) * (info.height as usize);
    let mut result = vec![0u8; pixels.len()];

    for i in 0..n {
        let pi = i * ch;
        // Compute luma in [0,1] for range weight, but keep pixel math in [0,255]
        let luma = (0.2126 * pixels[pi] as f32
            + 0.7152 * pixels[pi + 1] as f32
            + 0.0722 * pixels[pi + 2] as f32)
            / 255.0;
        let weight = match range {
            0 => { let t = (luma * 2.0).min(1.0); 1.0 - t * t * (3.0 - 2.0 * t) }
            2 => { let t = ((luma - 0.5) * 2.0).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }
            _ => (4.0 * luma * (1.0 - luma)).min(1.0),
        };
        let factor = exposure * weight;

        for c in 0..3 {
            let v = pixels[pi + c] as f32;
            let adjusted = if is_dodge { v + v * factor } else { v * (1.0 - factor) };
            result[pi + c] = adjusted.round().clamp(0.0, 255.0) as u8;
        }
        if ch == 4 {
            result[pi + 3] = pixels[pi + 3];
        }
    }

    Ok(result)
}

/// Flatten RGBA8 to RGB8 by blending onto a solid background color.
pub fn flatten(
    pixels: &[u8],
    info: &ImageInfo,
    bg: [u8; 3],
) -> Result<(Vec<u8>, ImageInfo), ImageError> {
    if info.format != PixelFormat::Rgba8 {
        return Err(ImageError::UnsupportedFormat(
            "flatten requires RGBA8 input".into(),
        ));
    }
    let npixels = (info.width * info.height) as usize;
    let mut rgb = Vec::with_capacity(npixels * 3);
    for chunk in pixels.chunks_exact(4) {
        let a = chunk[3] as f32 / 255.0;
        let inv_a = 1.0 - a;
        rgb.push((chunk[0] as f32 * a + bg[0] as f32 * inv_a + 0.5) as u8);
        rgb.push((chunk[1] as f32 * a + bg[1] as f32 * inv_a + 0.5) as u8);
        rgb.push((chunk[2] as f32 * a + bg[2] as f32 * inv_a + 0.5) as u8);
    }
    Ok((
        rgb,
        ImageInfo {
            width: info.width,
            height: info.height,
            format: PixelFormat::Rgb8,
            color_space: info.color_space,
        },
    ))
}

