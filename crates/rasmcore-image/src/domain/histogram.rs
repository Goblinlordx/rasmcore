//! Histogram analysis and automatic contrast correction.
//!
//! Provides histogram computation, image statistics, and LUT-based auto-correction
//! operations (equalize, normalize, auto-level, contrast stretch).
//!
//! Supports both 8-bit (256-bin) and 16-bit (65536-bin) pixel formats.
//! Public functions auto-dispatch based on pixel format. All correction operations
//! compute a histogram, derive a LUT, then apply it via the shared `point_ops`
//! infrastructure — fusible with other point ops in the pipeline.

use super::error::ImageError;
use super::point_ops;
use super::types::{ImageInfo, PixelFormat};

/// Per-channel histogram: frequency count for each value 0-255.
pub type Histogram = [u32; 256];

/// Per-channel image statistics.
#[derive(Debug, Clone, Copy)]
pub struct ChannelStats {
    pub min: u8,
    pub max: u8,
    pub mean: f64,
    pub stddev: f64,
}

/// Per-channel 16-bit histogram: frequency count for each value 0-65535.
pub type Histogram16 = Vec<u32>; // 65536 entries, 256 KB per channel

/// 16-bit per-channel image statistics.
#[derive(Debug, Clone, Copy)]
pub struct ChannelStats16 {
    pub min: u16,
    pub max: u16,
    pub mean: f64,
    pub stddev: f64,
}

/// Compute per-channel histograms from pixel data.
///
/// Returns 1 histogram for Gray8, 3 for RGB8, 3 for RGBA8 (alpha ignored).
pub fn histogram(pixels: &[u8], info: &ImageInfo) -> Result<Vec<Histogram>, ImageError> {
    match info.format {
        PixelFormat::Gray8 => {
            let mut h = [0u32; 256];
            for &p in pixels {
                h[p as usize] += 1;
            }
            Ok(vec![h])
        }
        PixelFormat::Rgb8 => {
            let mut hr = [0u32; 256];
            let mut hg = [0u32; 256];
            let mut hb = [0u32; 256];
            for chunk in pixels.chunks_exact(3) {
                hr[chunk[0] as usize] += 1;
                hg[chunk[1] as usize] += 1;
                hb[chunk[2] as usize] += 1;
            }
            Ok(vec![hr, hg, hb])
        }
        PixelFormat::Rgba8 => {
            let mut hr = [0u32; 256];
            let mut hg = [0u32; 256];
            let mut hb = [0u32; 256];
            for chunk in pixels.chunks_exact(4) {
                hr[chunk[0] as usize] += 1;
                hg[chunk[1] as usize] += 1;
                hb[chunk[2] as usize] += 1;
            }
            Ok(vec![hr, hg, hb])
        }
        other => Err(ImageError::UnsupportedFormat(format!(
            "histogram on {other:?} not supported"
        ))),
    }
}

/// Compute per-channel statistics (min, max, mean, stddev) in a single pass.
pub fn statistics(pixels: &[u8], info: &ImageInfo) -> Result<Vec<ChannelStats>, ImageError> {
    let histograms = histogram(pixels, info)?;
    let pixel_count = (info.width as u64) * (info.height as u64);

    Ok(histograms
        .iter()
        .map(|h| stats_from_histogram(h, pixel_count))
        .collect())
}

fn stats_from_histogram(h: &Histogram, pixel_count: u64) -> ChannelStats {
    let n = pixel_count as f64;
    let mut min = 255u8;
    let mut max = 0u8;
    let mut sum = 0u64;
    let mut sum_sq = 0u64;

    for (i, &count) in h.iter().enumerate() {
        if count > 0 {
            if (i as u8) < min {
                min = i as u8;
            }
            if (i as u8) > max {
                max = i as u8;
            }
            sum += i as u64 * count as u64;
            sum_sq += (i as u64) * (i as u64) * count as u64;
        }
    }

    let mean = sum as f64 / n;
    let variance = (sum_sq as f64 / n) - (mean * mean);
    let stddev = variance.max(0.0).sqrt();

    ChannelStats {
        min,
        max,
        mean,
        stddev,
    }
}

// ─── Auto-Correction Operations ─────────────────────────────────────────────
//
// All build a [u8; 256] LUT from the histogram, then apply via point_ops::apply_lut.

/// Histogram equalization — remap pixel values using the cumulative distribution function.
///
/// Produces a uniform histogram (flat distribution) for maximum contrast.
/// Auto-dispatches between 8-bit and 16-bit based on pixel format.
pub fn equalize(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    if is_16bit(info.format) {
        return equalize_16(pixels, info);
    }
    let histograms = histogram(pixels, info)?;
    let pixel_count = (info.width as u64) * (info.height as u64);

    if histograms.len() == 1 {
        let lut = equalize_lut(&histograms[0], pixel_count);
        point_ops::apply_lut(pixels, info, &lut)
    } else {
        apply_per_channel_luts(pixels, info, &histograms, pixel_count, equalize_lut)
    }
}

fn equalize_lut(h: &Histogram, pixel_count: u64) -> [u8; 256] {
    let mut cdf = [0u64; 256];
    cdf[0] = h[0] as u64;
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + h[i] as u64;
    }

    // Find first non-zero CDF value (standard histogram equalization formula)
    let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
    let denom = pixel_count - cdf_min;

    let mut lut = [0u8; 256];
    if denom == 0 {
        // All pixels are the same value — identity
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = i as u8;
        }
    } else {
        for (i, entry) in lut.iter_mut().enumerate() {
            let v = (cdf[i].saturating_sub(cdf_min) as f64 * 255.0 / denom as f64 + 0.5) as u8;
            *entry = v;
        }
    }
    lut
}

/// Normalize — linear stretch matching ImageMagick `-normalize`.
///
/// Clips the darkest 2% and brightest 1% of pixels, then linearly stretches
/// the remaining range to [0, 255].
pub fn normalize(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    contrast_stretch(pixels, info, 2.0, 1.0)
}

/// Auto-level — linear stretch from actual min to actual max (no clipping).
pub fn auto_level(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    contrast_stretch(pixels, info, 0.0, 0.0)
}

/// Contrast stretch — linear remap with configurable black/white point percentiles.
///
/// `black_pct`: percentage of darkest pixels to clip (0.0 = no clip, 0.5 = clip darkest 0.5%)
/// `white_pct`: percentage of brightest pixels to clip
pub fn contrast_stretch(
    pixels: &[u8],
    info: &ImageInfo,
    black_pct: f64,
    white_pct: f64,
) -> Result<Vec<u8>, ImageError> {
    if is_16bit(info.format) {
        return contrast_stretch_16(pixels, info, black_pct, white_pct);
    }
    let histograms = histogram(pixels, info)?;
    let pixel_count = (info.width as u64) * (info.height as u64);

    if histograms.len() == 1 {
        let lut = stretch_lut(&histograms[0], pixel_count, black_pct, white_pct);
        point_ops::apply_lut(pixels, info, &lut)
    } else {
        apply_per_channel_luts(pixels, info, &histograms, pixel_count, |h, n| {
            stretch_lut(h, n, black_pct, white_pct)
        })
    }
}

fn stretch_lut(h: &Histogram, pixel_count: u64, black_pct: f64, white_pct: f64) -> [u8; 256] {
    let black_threshold = (pixel_count as f64 * black_pct / 100.0) as u64;
    let white_threshold = (pixel_count as f64 * white_pct / 100.0) as u64;

    // Find black point (first value exceeding threshold from left)
    let mut cumulative = 0u64;
    let mut black = 0u8;
    for (i, &count) in h.iter().enumerate() {
        cumulative += count as u64;
        if cumulative > black_threshold {
            black = i as u8;
            break;
        }
    }

    // Find white point (first value exceeding threshold from right)
    cumulative = 0;
    let mut white = 255u8;
    for i in (0..256).rev() {
        cumulative += h[i] as u64;
        if cumulative > white_threshold {
            white = i as u8;
            break;
        }
    }

    // Build linear stretch LUT
    let mut lut = [0u8; 256];
    if black >= white {
        // Degenerate — identity
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = i as u8;
        }
    } else {
        let range = (white - black) as f64;
        for (i, entry) in lut.iter_mut().enumerate() {
            let v = ((i as f64 - black as f64) / range * 255.0).clamp(0.0, 255.0);
            *entry = (v + 0.5) as u8;
        }
    }
    lut
}

/// Apply per-channel LUTs to RGB8/RGBA8 pixels.
fn apply_per_channel_luts<F>(
    pixels: &[u8],
    info: &ImageInfo,
    histograms: &[Histogram],
    pixel_count: u64,
    build_lut_fn: F,
) -> Result<Vec<u8>, ImageError>
where
    F: Fn(&Histogram, u64) -> [u8; 256],
{
    let lut_r = build_lut_fn(&histograms[0], pixel_count);
    let lut_g = build_lut_fn(&histograms[1], pixel_count);
    let lut_b = build_lut_fn(&histograms[2], pixel_count);

    let mut result = vec![0u8; pixels.len()];
    match info.format {
        PixelFormat::Rgb8 => {
            for (i, chunk) in pixels.chunks_exact(3).enumerate() {
                let base = i * 3;
                result[base] = lut_r[chunk[0] as usize];
                result[base + 1] = lut_g[chunk[1] as usize];
                result[base + 2] = lut_b[chunk[2] as usize];
            }
        }
        PixelFormat::Rgba8 => {
            for (i, chunk) in pixels.chunks_exact(4).enumerate() {
                let base = i * 4;
                result[base] = lut_r[chunk[0] as usize];
                result[base + 1] = lut_g[chunk[1] as usize];
                result[base + 2] = lut_b[chunk[2] as usize];
                result[base + 3] = chunk[3]; // alpha
            }
        }
        _ => return Err(ImageError::UnsupportedFormat("unexpected format".into())),
    }
    Ok(result)
}

// ─── 16-bit Histogram Infrastructure ────────────────────────────────────────

fn is_16bit(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16
    )
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// Compute per-channel 16-bit histograms (65536 bins per channel).
pub fn histogram_u16(pixels: &[u8], info: &ImageInfo) -> Result<Vec<Histogram16>, ImageError> {
    match info.format {
        PixelFormat::Gray16 => {
            let mut h = vec![0u32; 65536];
            for pair in pixels.chunks_exact(2) {
                h[u16::from_le_bytes([pair[0], pair[1]]) as usize] += 1;
            }
            Ok(vec![h])
        }
        PixelFormat::Rgb16 => {
            let mut hr = vec![0u32; 65536];
            let mut hg = vec![0u32; 65536];
            let mut hb = vec![0u32; 65536];
            for chunk in pixels.chunks_exact(6) {
                hr[read_u16(chunk, 0) as usize] += 1;
                hg[read_u16(chunk, 2) as usize] += 1;
                hb[read_u16(chunk, 4) as usize] += 1;
            }
            Ok(vec![hr, hg, hb])
        }
        PixelFormat::Rgba16 => {
            let mut hr = vec![0u32; 65536];
            let mut hg = vec![0u32; 65536];
            let mut hb = vec![0u32; 65536];
            for chunk in pixels.chunks_exact(8) {
                hr[read_u16(chunk, 0) as usize] += 1;
                hg[read_u16(chunk, 2) as usize] += 1;
                hb[read_u16(chunk, 4) as usize] += 1;
            }
            Ok(vec![hr, hg, hb])
        }
        other => Err(ImageError::UnsupportedFormat(format!(
            "16-bit histogram on {other:?} not supported"
        ))),
    }
}

/// Compute 16-bit per-channel statistics.
pub fn statistics_u16(pixels: &[u8], info: &ImageInfo) -> Result<Vec<ChannelStats16>, ImageError> {
    let histograms = histogram_u16(pixels, info)?;
    let pixel_count = (info.width as u64) * (info.height as u64);
    Ok(histograms
        .iter()
        .map(|h| stats_from_histogram_u16(h, pixel_count))
        .collect())
}

fn stats_from_histogram_u16(h: &Histogram16, pixel_count: u64) -> ChannelStats16 {
    let n = pixel_count as f64;
    let mut min = 65535u16;
    let mut max = 0u16;
    let mut sum = 0u64;
    let mut sum_sq = 0f64; // use f64 to avoid u64 overflow for 65535^2 * count

    for (i, &count) in h.iter().enumerate() {
        if count > 0 {
            let iv = i as u16;
            if iv < min {
                min = iv;
            }
            if iv > max {
                max = iv;
            }
            sum += i as u64 * count as u64;
            sum_sq += (i as f64) * (i as f64) * count as f64;
        }
    }

    let mean = sum as f64 / n;
    let variance = (sum_sq / n) - (mean * mean);
    ChannelStats16 {
        min,
        max,
        mean,
        stddev: variance.max(0.0).sqrt(),
    }
}

fn equalize_16(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
    let histograms = histogram_u16(pixels, info)?;
    let pixel_count = (info.width as u64) * (info.height as u64);

    if histograms.len() == 1 {
        let lut = equalize_lut_u16(&histograms[0], pixel_count);
        point_ops::apply_lut_u16(pixels, info, &lut)
    } else {
        apply_per_channel_luts_u16(pixels, info, &histograms, pixel_count, equalize_lut_u16)
    }
}

fn equalize_lut_u16(h: &Histogram16, pixel_count: u64) -> Vec<u16> {
    let mut cdf = vec![0u64; 65536];
    cdf[0] = h[0] as u64;
    for i in 1..65536 {
        cdf[i] = cdf[i - 1] + h[i] as u64;
    }

    let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
    let denom = pixel_count - cdf_min;

    let mut lut = vec![0u16; 65536];
    if denom == 0 {
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = i as u16;
        }
    } else {
        for (i, entry) in lut.iter_mut().enumerate() {
            let v = (cdf[i].saturating_sub(cdf_min) as f64 * 65535.0 / denom as f64 + 0.5) as u16;
            *entry = v;
        }
    }
    lut
}

fn contrast_stretch_16(
    pixels: &[u8],
    info: &ImageInfo,
    black_pct: f64,
    white_pct: f64,
) -> Result<Vec<u8>, ImageError> {
    let histograms = histogram_u16(pixels, info)?;
    let pixel_count = (info.width as u64) * (info.height as u64);

    if histograms.len() == 1 {
        let lut = stretch_lut_u16(&histograms[0], pixel_count, black_pct, white_pct);
        point_ops::apply_lut_u16(pixels, info, &lut)
    } else {
        apply_per_channel_luts_u16(pixels, info, &histograms, pixel_count, |h, n| {
            stretch_lut_u16(h, n, black_pct, white_pct)
        })
    }
}

fn stretch_lut_u16(h: &Histogram16, pixel_count: u64, black_pct: f64, white_pct: f64) -> Vec<u16> {
    let black_threshold = (pixel_count as f64 * black_pct / 100.0) as u64;
    let white_threshold = (pixel_count as f64 * white_pct / 100.0) as u64;

    let mut cumulative = 0u64;
    let mut black = 0u16;
    for (i, &count) in h.iter().enumerate() {
        cumulative += count as u64;
        if cumulative > black_threshold {
            black = i as u16;
            break;
        }
    }

    cumulative = 0;
    let mut white = 65535u16;
    for i in (0..65536usize).rev() {
        cumulative += h[i] as u64;
        if cumulative > white_threshold {
            white = i as u16;
            break;
        }
    }

    let mut lut = vec![0u16; 65536];
    if black >= white {
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = i as u16;
        }
    } else {
        let range = (white - black) as f64;
        for (i, entry) in lut.iter_mut().enumerate() {
            let v = ((i as f64 - black as f64) / range * 65535.0).clamp(0.0, 65535.0);
            *entry = (v + 0.5) as u16;
        }
    }
    lut
}

/// Apply per-channel 16-bit LUTs to Rgb16/Rgba16 pixels.
fn apply_per_channel_luts_u16<F>(
    pixels: &[u8],
    info: &ImageInfo,
    histograms: &[Histogram16],
    pixel_count: u64,
    build_lut_fn: F,
) -> Result<Vec<u8>, ImageError>
where
    F: Fn(&Histogram16, u64) -> Vec<u16>,
{
    let lut_r = build_lut_fn(&histograms[0], pixel_count);
    let lut_g = build_lut_fn(&histograms[1], pixel_count);
    let lut_b = build_lut_fn(&histograms[2], pixel_count);

    let mut result = vec![0u8; pixels.len()];
    match info.format {
        PixelFormat::Rgb16 => {
            for (chunk_in, chunk_out) in pixels.chunks_exact(6).zip(result.chunks_exact_mut(6)) {
                let r = lut_r[read_u16(chunk_in, 0) as usize];
                let g = lut_g[read_u16(chunk_in, 2) as usize];
                let b = lut_b[read_u16(chunk_in, 4) as usize];
                chunk_out[0..2].copy_from_slice(&r.to_le_bytes());
                chunk_out[2..4].copy_from_slice(&g.to_le_bytes());
                chunk_out[4..6].copy_from_slice(&b.to_le_bytes());
            }
        }
        PixelFormat::Rgba16 => {
            for (chunk_in, chunk_out) in pixels.chunks_exact(8).zip(result.chunks_exact_mut(8)) {
                let r = lut_r[read_u16(chunk_in, 0) as usize];
                let g = lut_g[read_u16(chunk_in, 2) as usize];
                let b = lut_b[read_u16(chunk_in, 4) as usize];
                chunk_out[0..2].copy_from_slice(&r.to_le_bytes());
                chunk_out[2..4].copy_from_slice(&g.to_le_bytes());
                chunk_out[4..6].copy_from_slice(&b.to_le_bytes());
                chunk_out[6] = chunk_in[6]; // alpha lo
                chunk_out[7] = chunk_in[7]; // alpha hi
            }
        }
        _ => {
            return Err(ImageError::UnsupportedFormat(
                "unexpected 16-bit format".into(),
            ));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn test_info(w: u32, h: u32, fmt: PixelFormat) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: fmt,
            color_space: ColorSpace::Srgb,
        }
    }

    // ── Histogram ───────────────────────────────────────────────────────

    #[test]
    fn histogram_gray8_counts() {
        let pixels = vec![0, 0, 0, 128, 128, 255];
        let info = test_info(6, 1, PixelFormat::Gray8);
        let h = histogram(&pixels, &info).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0][0], 3);
        assert_eq!(h[0][128], 2);
        assert_eq!(h[0][255], 1);
        assert_eq!(h[0][1], 0);
    }

    #[test]
    fn histogram_rgb8_per_channel() {
        let pixels = vec![255, 0, 128, 255, 0, 128]; // 2 pixels
        let info = test_info(2, 1, PixelFormat::Rgb8);
        let h = histogram(&pixels, &info).unwrap();
        assert_eq!(h.len(), 3);
        assert_eq!(h[0][255], 2); // R=255 x2
        assert_eq!(h[1][0], 2); // G=0 x2
        assert_eq!(h[2][128], 2); // B=128 x2
    }

    // ── Statistics ──────────────────────────────────────────────────────

    #[test]
    fn statistics_known_values() {
        // 4 pixels: 0, 100, 200, 255
        let pixels = vec![0, 100, 200, 255];
        let info = test_info(4, 1, PixelFormat::Gray8);
        let stats = statistics(&pixels, &info).unwrap();
        assert_eq!(stats[0].min, 0);
        assert_eq!(stats[0].max, 255);
        let expected_mean = (0.0 + 100.0 + 200.0 + 255.0) / 4.0;
        assert!((stats[0].mean - expected_mean).abs() < 0.1);
        assert!(stats[0].stddev > 0.0);
    }

    #[test]
    fn statistics_uniform() {
        let pixels = vec![128u8; 100];
        let info = test_info(100, 1, PixelFormat::Gray8);
        let stats = statistics(&pixels, &info).unwrap();
        assert_eq!(stats[0].min, 128);
        assert_eq!(stats[0].max, 128);
        assert!((stats[0].mean - 128.0).abs() < 0.01);
        assert!(stats[0].stddev < 0.01);
    }

    // ── Equalize ────────────────────────────────────────────────────────

    #[test]
    fn equalize_uniform_input_unchanged() {
        // All same value → equalization has no effect
        let pixels = vec![128u8; 64];
        let info = test_info(64, 1, PixelFormat::Gray8);
        let result = equalize(&pixels, &info).unwrap();
        // All should map to same value
        let first = result[0];
        assert!(result.iter().all(|&v| v == first));
    }

    #[test]
    fn equalize_spreads_histogram() {
        // Clustered values should spread toward full range
        let pixels: Vec<u8> = (100..=110).cycle().take(256).collect();
        let info = test_info(256, 1, PixelFormat::Gray8);
        let result = equalize(&pixels, &info).unwrap();

        let stats_before = statistics(&pixels, &info).unwrap();
        let stats_after = statistics(&result, &info).unwrap();
        // Range should expand
        assert!(
            (stats_after[0].max - stats_after[0].min) > (stats_before[0].max - stats_before[0].min),
            "equalize should expand the value range"
        );
    }

    // ── Normalize ───────────────────────────────────────────────────────

    #[test]
    fn normalize_stretches_to_full_range() {
        // Values 50-200 should stretch to ~0-255
        let pixels: Vec<u8> = (50..=200).collect();
        let info = test_info(pixels.len() as u32, 1, PixelFormat::Gray8);
        let result = normalize(&pixels, &info).unwrap();

        let stats = statistics(&result, &info).unwrap();
        assert!(
            stats[0].min <= 5,
            "min should be near 0, got {}",
            stats[0].min
        );
        assert!(
            stats[0].max >= 250,
            "max should be near 255, got {}",
            stats[0].max
        );
    }

    // ── Auto-level ──────────────────────────────────────────────────────

    #[test]
    fn auto_level_exact_stretch() {
        // Values 100-200 → should map to 0-255
        let pixels: Vec<u8> = (100..=200).collect();
        let info = test_info(pixels.len() as u32, 1, PixelFormat::Gray8);
        let result = auto_level(&pixels, &info).unwrap();

        assert_eq!(result[0], 0, "min should map to 0");
        assert_eq!(*result.last().unwrap(), 255, "max should map to 255");
    }

    #[test]
    fn auto_level_already_full_range() {
        let mut pixels: Vec<u8> = (0..=255).collect();
        pixels.extend(0..=255u8); // 512 pixels
        let info = test_info(512, 1, PixelFormat::Gray8);
        let result = auto_level(&pixels, &info).unwrap();
        // Should be approximately identity
        for (i, (&orig, &out)) in pixels.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i16 - out as i16).abs() <= 1,
                "pixel {i}: {orig} -> {out}"
            );
        }
    }

    // ── Contrast stretch ────────────────────────────────────────────────

    #[test]
    fn contrast_stretch_clips_extremes() {
        // With 5% black/white clip on a ramp
        let pixels: Vec<u8> = (0..=255).collect();
        let info = test_info(256, 1, PixelFormat::Gray8);
        let result = contrast_stretch(&pixels, &info, 5.0, 5.0).unwrap();

        // Bottom ~5% should be clipped to 0, top ~5% to 255
        assert_eq!(result[0], 0);
        assert_eq!(result[255], 255);
        // Middle should be stretched
        assert!(result[128] > 100 && result[128] < 200);
    }

    // ── RGB support ─────────────────────────────────────────────────────

    #[test]
    fn equalize_rgb8() {
        let pixels: Vec<u8> = (0..192).map(|i| (i % 3 * 85) as u8).collect(); // R=0,G=85,B=170 repeating
        let info = test_info(64, 1, PixelFormat::Rgb8);
        let result = equalize(&pixels, &info).unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn equalize_rgba8_preserves_alpha() {
        let pixels = vec![100, 150, 200, 128, 50, 100, 150, 255];
        let info = test_info(2, 1, PixelFormat::Rgba8);
        let result = equalize(&pixels, &info).unwrap();
        assert_eq!(result[3], 128, "alpha should be preserved");
        assert_eq!(result[7], 255, "alpha should be preserved");
    }

    // ── 16-bit histogram tests ─────────────────────────────────────────

    fn make_gray16(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_u16_vec(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    #[test]
    fn histogram_u16_gray16_counts() {
        let pixels = make_gray16(&[0, 0, 1000, 1000, 65535]);
        let info = test_info(5, 1, PixelFormat::Gray16);
        let h = histogram_u16(&pixels, &info).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0][0], 2);
        assert_eq!(h[0][1000], 2);
        assert_eq!(h[0][65535], 1);
        assert_eq!(h[0][1], 0);
    }

    #[test]
    fn statistics_u16_known_values() {
        let pixels = make_gray16(&[0, 32768, 65535]);
        let info = test_info(3, 1, PixelFormat::Gray16);
        let stats = statistics_u16(&pixels, &info).unwrap();
        assert_eq!(stats[0].min, 0);
        assert_eq!(stats[0].max, 65535);
        let expected_mean = (0.0 + 32768.0 + 65535.0) / 3.0;
        assert!(
            (stats[0].mean - expected_mean).abs() < 1.0,
            "mean: {} expected: {}",
            stats[0].mean,
            expected_mean
        );
    }

    #[test]
    fn equalize_gray16_spreads() {
        // Clustered values should spread
        let values: Vec<u16> = (30000..=30100).cycle().take(256).collect();
        let pixels = make_gray16(&values);
        let info = test_info(256, 1, PixelFormat::Gray16);
        let result = equalize(&pixels, &info).unwrap();
        let out = read_u16_vec(&result);
        let min = *out.iter().min().unwrap();
        let max = *out.iter().max().unwrap();
        assert!(
            max - min > 30100 - 30000,
            "equalize should expand range: {min}-{max}"
        );
    }

    #[test]
    fn equalize_gray16_uniform_unchanged() {
        let pixels = make_gray16(&[32768; 64]);
        let info = test_info(64, 1, PixelFormat::Gray16);
        let result = equalize(&pixels, &info).unwrap();
        let out = read_u16_vec(&result);
        let first = out[0];
        assert!(out.iter().all(|&v| v == first));
    }

    #[test]
    fn auto_level_gray16_stretches() {
        // Values 10000-50000 → should stretch to 0-65535
        let values: Vec<u16> = (0..256).map(|i| 10000 + i * 156).collect();
        let pixels = make_gray16(&values);
        let info = test_info(256, 1, PixelFormat::Gray16);
        let result = contrast_stretch(&pixels, &info, 0.0, 0.0).unwrap();
        let out = read_u16_vec(&result);
        assert_eq!(out[0], 0, "min should map to 0");
        assert_eq!(*out.last().unwrap(), 65535, "max should map to 65535");
    }

    #[test]
    fn normalize_gray16() {
        // Verify normalize dispatches to 16-bit path
        let values: Vec<u16> = (0..1024).map(|i| i * 64).collect();
        let pixels = make_gray16(&values);
        let info = test_info(1024, 1, PixelFormat::Gray16);
        let result = normalize(&pixels, &info).unwrap();
        assert_eq!(result.len(), pixels.len());
        let out = read_u16_vec(&result);
        // Should stretch near full range
        assert!(*out.iter().min().unwrap() < 1000);
        assert!(*out.iter().max().unwrap() > 64000);
    }
}
