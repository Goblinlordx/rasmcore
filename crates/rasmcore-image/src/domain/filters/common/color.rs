//! Color helpers for filters.

#[allow(unused_imports)]
use super::*;


/// Apply a ColorOp to a pixel buffer via direct per-pixel evaluation.
///
/// No CLUT allocation — evaluates ColorOp::apply() on each pixel's
/// normalized (R,G,B). For pipeline use, ColorOpNode builds a CLUT instead.
pub fn apply_color_op(pixels: &[u8], info: &ImageInfo, op: &ColorOp) -> Result<Vec<u8>, ImageError> {
    validate_format(info.format)?;
    if info.format == PixelFormat::Gray8 || info.format == PixelFormat::Gray16 {
        return Ok(pixels.to_vec());
    }

    // 16-bit color operations: work in f32 [0,1] range
    if is_16bit(info.format) {
        let ch = channels(info.format);
        let samples = bytes_to_u16(pixels);
        let mut result_u16 = samples.clone();
        for chunk in result_u16.chunks_exact_mut(ch) {
            let (r, g, b) = op.apply(
                chunk[0] as f32 / 65535.0,
                chunk[1] as f32 / 65535.0,
                chunk[2] as f32 / 65535.0,
            );
            chunk[0] = (r * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            chunk[1] = (g * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            chunk[2] = (b * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        }
        return Ok(u16_to_bytes(&result_u16));
    }

    let bpp = if info.format == PixelFormat::Rgba8 {
        4
    } else {
        3
    };
    let mut result = pixels.to_vec();
    for chunk in result.chunks_exact_mut(bpp) {
        let (r, g, b) = op.apply(
            chunk[0] as f32 / 255.0,
            chunk[1] as f32 / 255.0,
            chunk[2] as f32 / 255.0,
        );
        chunk[0] = (r * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        chunk[1] = (g * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        chunk[2] = (b * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    Ok(result)
}

/// Clip a color so all channels are in [0,1] while preserving luminance.
#[inline]
pub fn clip_color(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let l = lum(r, g, b);
    let n = r.min(g).min(b);
    let x = r.max(g).max(b);
    let (mut r, mut g, mut b) = (r, g, b);
    if n < 0.0 {
        let ln = l - n;
        r = l + (r - l) * l / ln;
        g = l + (g - l) * l / ln;
        b = l + (b - l) * l / ln;
    }
    if x > 1.0 {
        let xl = x - l;
        let one_l = 1.0 - l;
        r = l + (r - l) * one_l / xl;
        g = l + (g - l) * one_l / xl;
        b = l + (b - l) * one_l / xl;
    }
    (r, g, b)
}

/// Convert to grayscale using weighted channel sum.
///
/// Uses ITU-R BT.709 weights: 0.2126R + 0.7152G + 0.0722B
pub fn grayscale(pixels: &[u8], info: &ImageInfo) -> Result<DecodedImage, ImageError> {
    validate_format(info.format)?;

    let pixel_count = info.width as usize * info.height as usize;

    let gray_pixels = match info.format {
        PixelFormat::Gray8 => pixels.to_vec(),
        PixelFormat::Rgb8 => {
            let mut gray = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(3) {
                let r = chunk[0] as f32;
                let g = chunk[1] as f32;
                let b = chunk[2] as f32;
                gray.push((0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8);
            }
            gray
        }
        PixelFormat::Rgba8 => {
            let mut gray = Vec::with_capacity(pixel_count);
            for chunk in pixels.chunks_exact(4) {
                let r = chunk[0] as f32;
                let g = chunk[1] as f32;
                let b = chunk[2] as f32;
                gray.push((0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8);
            }
            gray
        }
        _ => unreachable!(),
    };

    Ok(DecodedImage {
        pixels: gray_pixels,
        info: ImageInfo {
            width: info.width,
            height: info.height,
            format: PixelFormat::Gray8,
            color_space: info.color_space,
        },
        icc_profile: None,
    })
}

/// Convert hue (degrees) to an RGB tint color at full saturation, 50% lightness.
pub fn hue_to_rgb_tint(hue_deg: f32) -> [f32; 3] {
    let h = (hue_deg % 360.0 + 360.0) % 360.0;
    let c = 1.0f32; // chroma at S=1, L=0.5
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match h_prime as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    // Add lightness offset for L=0.5 (m = L - C/2 = 0)
    [r1, g1, b1]
}

/// Interpolate a color from sorted gradient stops at the given position.
pub fn interpolate_gradient(stops: &[(f32, [u8; 3])], t: f32) -> [u8; 3] {
    if stops.len() == 1 || t <= stops[0].0 {
        return stops[0].1;
    }
    if t >= stops[stops.len() - 1].0 {
        return stops[stops.len() - 1].1;
    }
    // Find the two stops surrounding t
    for i in 0..stops.len() - 1 {
        let (p0, c0) = stops[i];
        let (p1, c1) = stops[i + 1];
        if t >= p0 && t <= p1 {
            let frac = if (p1 - p0).abs() < 1e-9 {
                0.0
            } else {
                (t - p0) / (p1 - p0)
            };
            return [
                (c0[0] as f32 + (c1[0] as f32 - c0[0] as f32) * frac + 0.5) as u8,
                (c0[1] as f32 + (c1[1] as f32 - c0[1] as f32) * frac + 0.5) as u8,
                (c0[2] as f32 + (c1[2] as f32 - c0[2] as f32) * frac + 0.5) as u8,
            ];
        }
    }
    stops[stops.len() - 1].1
}

/// BT.601 luminance in normalized [0,1] space.
#[inline]
pub fn lum(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Parse a JSON string of control points: `[[x,y],[x,y],...]` into `Vec<(f32, f32)>`.
pub fn parse_curve_points(json: &str) -> Result<Vec<(f32, f32)>, ImageError> {
    // Minimal JSON array parser — avoids serde dependency for simple [[f,f],...] arrays
    let s = json.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(ImageError::InvalidParameters(
            "curves points must be a JSON array: [[x,y],...]".into(),
        ));
    }
    // Strip outer brackets
    let inner = &s[1..s.len() - 1];
    let mut points = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, ch) in inner.char_indices() {
        match ch {
            '[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    let pair = &inner[start + 1..i];
                    let parts: Vec<&str> = pair.split(',').collect();
                    if parts.len() != 2 {
                        return Err(ImageError::InvalidParameters(format!(
                            "each curve point must be [x,y], got: [{pair}]"
                        )));
                    }
                    let x: f32 = parts[0].trim().parse().map_err(|_| {
                        ImageError::InvalidParameters(format!(
                            "invalid x in curve point: {}",
                            parts[0].trim()
                        ))
                    })?;
                    let y: f32 = parts[1].trim().parse().map_err(|_| {
                        ImageError::InvalidParameters(format!(
                            "invalid y in curve point: {}",
                            parts[1].trim()
                        ))
                    })?;
                    points.push((x, y));
                }
            }
            _ => {}
        }
    }
    if points.len() < 2 {
        return Err(ImageError::InvalidParameters(
            "curves requires at least 2 control points".into(),
        ));
    }
    Ok(points)
}

/// Parse gradient stops from string format "pos:RRGGBB,pos:RRGGBB,...".
pub fn parse_gradient_stops(stops: &str) -> Result<Vec<(f32, [u8; 3])>, ImageError> {
    let mut result = Vec::new();
    for entry in stops.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let parts: Vec<&str> = entry.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(ImageError::InvalidParameters(format!(
                "gradient stop must be 'pos:RRGGBB', got '{entry}'"
            )));
        }
        let pos: f32 = parts[0].parse().map_err(|_| {
            ImageError::InvalidParameters(format!("invalid position: '{}'", parts[0]))
        })?;
        let hex = parts[1].trim_start_matches('#');
        if hex.len() != 6 {
            return Err(ImageError::InvalidParameters(format!(
                "color must be 6-digit hex, got '{hex}'"
            )));
        }
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex color: '{hex}'")))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex color: '{hex}'")))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex color: '{hex}'")))?;
        result.push((pos, [r, g, b]));
    }
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if result.is_empty() {
        return Err(ImageError::InvalidParameters(
            "gradient must have at least one stop".into(),
        ));
    }
    Ok(result)
}

/// Parse sparse color control points from "x,y:RRGGBB;x,y:RRGGBB;..." format.
pub fn parse_sparse_points(points: &str) -> Result<Vec<(f32, f32, [u8; 3])>, ImageError> {
    let mut result = Vec::new();
    for entry in points.split(';') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let parts: Vec<&str> = entry.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(ImageError::InvalidParameters(format!(
                "sparse point must be 'x,y:RRGGBB', got '{entry}'"
            )));
        }
        let coords: Vec<&str> = parts[0].split(',').collect();
        if coords.len() != 2 {
            return Err(ImageError::InvalidParameters(format!(
                "coordinates must be 'x,y', got '{}'",
                parts[0]
            )));
        }
        let x: f32 = coords[0]
            .trim()
            .parse()
            .map_err(|_| ImageError::InvalidParameters(format!("invalid x: '{}'", coords[0])))?;
        let y: f32 = coords[1]
            .trim()
            .parse()
            .map_err(|_| ImageError::InvalidParameters(format!("invalid y: '{}'", coords[1])))?;
        let hex = parts[1].trim().trim_start_matches('#');
        if hex.len() != 6 {
            return Err(ImageError::InvalidParameters(format!(
                "color must be 6-digit hex, got '{hex}'"
            )));
        }
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex: '{hex}'")))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex: '{hex}'")))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| ImageError::InvalidParameters(format!("invalid hex: '{hex}'")))?;
        result.push((x, y, [r, g, b]));
    }
    if result.is_empty() {
        return Err(ImageError::InvalidParameters(
            "sparse_color requires at least one control point".into(),
        ));
    }
    Ok(result)
}

/// Saturation = max(C) - min(C).
#[inline]
pub fn sat(r: f32, g: f32, b: f32) -> f32 {
    r.max(g).max(b) - r.min(g).min(b)
}

/// Set luminance of a color to `target_l`, then clip.
#[inline]
pub fn set_lum(r: f32, g: f32, b: f32, target_l: f32) -> (f32, f32, f32) {
    let d = target_l - lum(r, g, b);
    clip_color(r + d, g + d, b + d)
}

/// Combined SetSat + SetLum: set saturation then luminance of a color.
#[inline]
pub fn set_lum_sat(r: f32, g: f32, b: f32, target_s: f32, target_l: f32) -> (f32, f32, f32) {
    let (r2, g2, b2) = set_sat(r, g, b, target_s);
    set_lum(r2, g2, b2, target_l)
}

/// Set saturation of color `c` to `target_s`.
///
/// Sorts channels, scales mid proportionally, zeros min, sets max to target_s.
#[inline]
pub fn set_sat(r: f32, g: f32, b: f32, target_s: f32) -> (f32, f32, f32) {
    // Identify min/mid/max channels by index (0=R, 1=G, 2=B)
    let c = [r, g, b];
    let (min_i, mid_i, max_i) = if c[0] <= c[1] {
        if c[1] <= c[2] {
            (0, 1, 2)
        } else if c[0] <= c[2] {
            (0, 2, 1)
        } else {
            (2, 0, 1)
        }
    } else if c[0] <= c[2] {
        (1, 0, 2)
    } else if c[1] <= c[2] {
        (1, 2, 0)
    } else {
        (2, 1, 0)
    };

    let mut out = [0.0f32; 3];
    if c[max_i] > c[min_i] {
        out[mid_i] = (c[mid_i] - c[min_i]) * target_s / (c[max_i] - c[min_i]);
        out[max_i] = target_s;
    }
    // out[min_i] stays 0.0
    (out[0], out[1], out[2])
}

/// Convert multi-channel pixels to single-channel grayscale.
/// Convert multi-channel pixels to single-channel grayscale.
///
/// Uses BT.601 fixed-point: (77*R + 150*G + 29*B + 128) >> 8.
/// Integer-only arithmetic — no floating point in the hot path.
pub fn to_grayscale(pixels: &[u8], channels: usize) -> Vec<u8> {
    if channels == 1 {
        return pixels.to_vec();
    }
    let pixel_count = pixels.len() / channels;

    #[cfg(target_arch = "wasm32")]
    {
        to_grayscale_simd128(pixels, channels, pixel_count)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        to_grayscale_scalar(pixels, channels, pixel_count)
    }
}

#[allow(dead_code)] // scalar fallback, SIMD path used on wasm32
pub fn to_grayscale_scalar(pixels: &[u8], channels: usize, pixel_count: usize) -> Vec<u8> {
    let mut gray = Vec::with_capacity(pixel_count);
    // BT.601 fixed-point: 77/256 ≈ 0.3008, 150/256 ≈ 0.5859, 29/256 ≈ 0.1133
    for i in 0..pixel_count {
        let r = pixels[i * channels] as u32;
        let g = pixels[i * channels + 1] as u32;
        let b = pixels[i * channels + 2] as u32;
        gray.push(((77 * r + 150 * g + 29 * b + 128) >> 8) as u8);
    }
    gray
}

#[cfg(target_arch = "wasm32")]
pub fn to_grayscale_simd128(pixels: &[u8], channels: usize, pixel_count: usize) -> Vec<u8> {
    use std::arch::wasm32::*;

    let mut gray = vec![0u8; pixel_count];

    // Process 4 pixels at a time using i32x4 multiply-accumulate
    let chunks = pixel_count / 4;
    let coeff_r = i32x4_splat(77);
    let coeff_g = i32x4_splat(150);
    let coeff_b = i32x4_splat(29);
    let round = i32x4_splat(128);

    for chunk in 0..chunks {
        let out_base = chunk * 4;

        // Load 4 pixels, extract R/G/B channels
        let mut rv = [0i32; 4];
        let mut gv = [0i32; 4];
        let mut bv = [0i32; 4];
        for p in 0..4 {
            let base = (out_base + p) * channels;
            rv[p] = pixels[base] as i32;
            gv[p] = pixels[base + 1] as i32;
            bv[p] = pixels[base + 2] as i32;
        }

        // SAFETY: rv/gv/bv are [i32; 4] on stack, properly aligned for v128_load
        unsafe {
            let r = v128_load(rv.as_ptr() as *const v128);
            let g = v128_load(gv.as_ptr() as *const v128);
            let b = v128_load(bv.as_ptr() as *const v128);

            // Y = (77*R + 150*G + 29*B + 128) >> 8
            let sum = i32x4_add(
                i32x4_add(i32x4_mul(coeff_r, r), i32x4_mul(coeff_g, g)),
                i32x4_add(i32x4_mul(coeff_b, b), round),
            );
            let shifted = i32x4_shr(sum, 8);

            gray[out_base] = i32x4_extract_lane::<0>(shifted) as u8;
            gray[out_base + 1] = i32x4_extract_lane::<1>(shifted) as u8;
            gray[out_base + 2] = i32x4_extract_lane::<2>(shifted) as u8;
            gray[out_base + 3] = i32x4_extract_lane::<3>(shifted) as u8;
        }
    }

    // Handle remaining pixels
    for i in (chunks * 4)..pixel_count {
        let r = pixels[i * channels] as u32;
        let g = pixels[i * channels + 1] as u32;
        let b = pixels[i * channels + 2] as u32;
        gray[i] = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
    }

    gray
}

