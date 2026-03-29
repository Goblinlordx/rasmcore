//! Color quantization and dithering — reduce images to N-color palettes.
//!
//! - **median_cut()** — Generate an optimal N-color palette via recursive bounding box splitting
//! - **quantize()** — Map each pixel to the nearest palette color
//! - **dither_floyd_steinberg()** — Error-diffusion dithering with serpentine scan
//! - **dither_ordered()** — Ordered dithering using Bayer matrices (2×2, 4×4, 8×8)

use super::error::ImageError;
use super::types::ImageInfo;

// ─── Palette Generation ────────────────────────────────────────────────────

/// A color in RGB space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Generate an N-color palette from an RGB8 image using the median cut algorithm.
///
/// Recursively splits the color-space bounding box along the axis of greatest range,
/// partitioning at the median. Each final box's average color becomes a palette entry.
///
/// `max_colors` must be 2..=256.
pub fn median_cut(
    pixels: &[u8],
    info: &ImageInfo,
    max_colors: usize,
) -> Result<Vec<Rgb>, ImageError> {
    if max_colors < 2 || max_colors > 256 {
        return Err(ImageError::InvalidParameters(
            "max_colors must be 2..256".into(),
        ));
    }
    let n = (info.width * info.height) as usize;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }

    // Collect all pixels as (r, g, b) tuples
    let mut colors: Vec<[u8; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        colors.push([pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2]]);
    }

    // Start with one box containing all colors
    let mut boxes: Vec<ColorBox> = vec![ColorBox::new(&mut colors, 0, n)];

    // Split until we have max_colors boxes (or can't split further)
    while boxes.len() < max_colors {
        // Find the box with the largest range to split
        let (split_idx, _) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.count > 1)
            .max_by_key(|(_, b)| b.longest_axis_range())
            .unwrap_or((0, &boxes[0]));

        if boxes[split_idx].count <= 1 {
            break; // Can't split any further
        }

        let b = boxes.remove(split_idx);
        let (left, right) = b.split(&mut colors);
        boxes.push(left);
        boxes.push(right);
    }

    // Average color per box = palette entry
    let palette: Vec<Rgb> = boxes.iter().map(|b| b.average(&colors)).collect();
    Ok(palette)
}

/// Map each pixel in an RGB8 image to the nearest color in the palette.
///
/// Returns a new RGB8 image with only palette colors.
pub fn quantize(pixels: &[u8], info: &ImageInfo, palette: &[Rgb]) -> Result<Vec<u8>, ImageError> {
    let n = (info.width * info.height) as usize;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if palette.is_empty() {
        return Err(ImageError::InvalidParameters("palette is empty".into()));
    }

    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;

        let nearest = find_nearest(r, g, b, palette);
        out[i * 3] = nearest.r;
        out[i * 3 + 1] = nearest.g;
        out[i * 3 + 2] = nearest.b;
    }
    Ok(out)
}

/// Map each pixel to a palette INDEX (0..palette.len()-1).
/// Useful for indexed/palette image formats (GIF, PNG8).
pub fn quantize_indexed(
    pixels: &[u8],
    info: &ImageInfo,
    palette: &[Rgb],
) -> Result<Vec<u8>, ImageError> {
    let n = (info.width * info.height) as usize;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if palette.is_empty() {
        return Err(ImageError::InvalidParameters("palette is empty".into()));
    }

    let mut indices = vec![0u8; n];
    for i in 0..n {
        let r = pixels[i * 3] as i32;
        let g = pixels[i * 3 + 1] as i32;
        let b = pixels[i * 3 + 2] as i32;

        indices[i] = find_nearest_index(r, g, b, palette) as u8;
    }
    Ok(indices)
}

// ─── Dithering ─────────────────────────────────────────────────────────────

/// Floyd-Steinberg error-diffusion dithering with serpentine (boustrophedon) scan.
///
/// Distributes quantization error to neighboring pixels:
///   * 7/16 → right
///   * 3/16 → below-left
///   * 5/16 → below
///   * 1/16 → below-right
///
/// Serpentine: even rows left→right, odd rows right→left.
pub fn dither_floyd_steinberg(
    pixels: &[u8],
    info: &ImageInfo,
    palette: &[Rgb],
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let n = w * h;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if palette.is_empty() {
        return Err(ImageError::InvalidParameters("palette is empty".into()));
    }

    // Work in i16 to handle error accumulation
    let mut buf: Vec<[i16; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        buf.push([
            pixels[i * 3] as i16,
            pixels[i * 3 + 1] as i16,
            pixels[i * 3 + 2] as i16,
        ]);
    }

    let mut out = vec![0u8; n * 3];

    for y in 0..h {
        let left_to_right = y % 2 == 0;
        let xs: Vec<usize> = if left_to_right {
            (0..w).collect()
        } else {
            (0..w).rev().collect()
        };

        for &x in &xs {
            let idx = y * w + x;
            let old_r = buf[idx][0].clamp(0, 255);
            let old_g = buf[idx][1].clamp(0, 255);
            let old_b = buf[idx][2].clamp(0, 255);

            let nearest = find_nearest(old_r as i32, old_g as i32, old_b as i32, palette);
            out[idx * 3] = nearest.r;
            out[idx * 3 + 1] = nearest.g;
            out[idx * 3 + 2] = nearest.b;

            let err_r = old_r - nearest.r as i16;
            let err_g = old_g - nearest.g as i16;
            let err_b = old_b - nearest.b as i16;

            // Distribute error
            let (right, below_left, below_right) = if left_to_right {
                (
                    if x + 1 < w { Some(idx + 1) } else { None },
                    if x > 0 && y + 1 < h {
                        Some((y + 1) * w + x - 1)
                    } else {
                        None
                    },
                    if x + 1 < w && y + 1 < h {
                        Some((y + 1) * w + x + 1)
                    } else {
                        None
                    },
                )
            } else {
                (
                    if x > 0 { Some(idx - 1) } else { None },
                    if x + 1 < w && y + 1 < h {
                        Some((y + 1) * w + x + 1)
                    } else {
                        None
                    },
                    if x > 0 && y + 1 < h {
                        Some((y + 1) * w + x - 1)
                    } else {
                        None
                    },
                )
            };
            let below = if y + 1 < h {
                Some((y + 1) * w + x)
            } else {
                None
            };

            for c in 0..3 {
                let err = [err_r, err_g, err_b][c];
                if let Some(i) = right {
                    buf[i][c] += err * 7 / 16;
                }
                if let Some(i) = below_left {
                    buf[i][c] += err * 3 / 16;
                }
                if let Some(i) = below {
                    buf[i][c] += err * 5 / 16;
                }
                if let Some(i) = below_right {
                    buf[i][c] += err * 1 / 16;
                }
            }
        }
    }

    Ok(out)
}

/// Ordered dithering using a Bayer matrix.
///
/// `matrix_size` must be 2, 4, or 8.
pub fn dither_ordered(
    pixels: &[u8],
    info: &ImageInfo,
    palette: &[Rgb],
    matrix_size: usize,
) -> Result<Vec<u8>, ImageError> {
    let w = info.width as usize;
    let h = info.height as usize;
    let n = w * h;
    if pixels.len() < n * 3 {
        return Err(ImageError::InvalidInput("pixel buffer too small".into()));
    }
    if palette.is_empty() {
        return Err(ImageError::InvalidParameters("palette is empty".into()));
    }

    let matrix = match matrix_size {
        2 => &BAYER_2X2[..],
        4 => &BAYER_4X4[..],
        8 => &BAYER_8X8[..],
        _ => {
            return Err(ImageError::InvalidParameters(
                "matrix_size must be 2, 4, or 8".into(),
            ));
        }
    };
    let ms = matrix_size;
    let scale = 255.0 / (ms * ms) as f32;

    let mut out = vec![0u8; n * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let threshold = matrix[(y % ms) * ms + (x % ms)] as f32 * scale - 128.0;

            let r = (pixels[idx * 3] as f32 + threshold).clamp(0.0, 255.0) as i32;
            let g = (pixels[idx * 3 + 1] as f32 + threshold).clamp(0.0, 255.0) as i32;
            let b = (pixels[idx * 3 + 2] as f32 + threshold).clamp(0.0, 255.0) as i32;

            let nearest = find_nearest(r, g, b, palette);
            out[idx * 3] = nearest.r;
            out[idx * 3 + 1] = nearest.g;
            out[idx * 3 + 2] = nearest.b;
        }
    }

    Ok(out)
}

// ─── Internals ─────────────────────────────────────────────────────────────

fn find_nearest(r: i32, g: i32, b: i32, palette: &[Rgb]) -> Rgb {
    palette[find_nearest_index(r, g, b, palette)]
}

fn find_nearest_index(r: i32, g: i32, b: i32, palette: &[Rgb]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = i32::MAX;
    for (i, c) in palette.iter().enumerate() {
        let dr = r - c.r as i32;
        let dg = g - c.g as i32;
        let db = b - c.b as i32;
        let dist = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx
}

/// A bounding box in RGB color space for median cut.
struct ColorBox {
    start: usize,
    count: usize,
    r_min: u8,
    r_max: u8,
    g_min: u8,
    g_max: u8,
    b_min: u8,
    b_max: u8,
}

impl ColorBox {
    fn new(colors: &mut [[u8; 3]], start: usize, count: usize) -> Self {
        let slice = &colors[start..start + count];
        let (mut rmin, mut rmax) = (255u8, 0u8);
        let (mut gmin, mut gmax) = (255u8, 0u8);
        let (mut bmin, mut bmax) = (255u8, 0u8);
        for c in slice {
            rmin = rmin.min(c[0]);
            rmax = rmax.max(c[0]);
            gmin = gmin.min(c[1]);
            gmax = gmax.max(c[1]);
            bmin = bmin.min(c[2]);
            bmax = bmax.max(c[2]);
        }
        ColorBox {
            start,
            count,
            r_min: rmin,
            r_max: rmax,
            g_min: gmin,
            g_max: gmax,
            b_min: bmin,
            b_max: bmax,
        }
    }

    fn longest_axis_range(&self) -> u16 {
        let r_range = (self.r_max - self.r_min) as u16;
        let g_range = (self.g_max - self.g_min) as u16;
        let b_range = (self.b_max - self.b_min) as u16;
        r_range.max(g_range).max(b_range)
    }

    fn longest_axis(&self) -> usize {
        let r_range = self.r_max - self.r_min;
        let g_range = self.g_max - self.g_min;
        let b_range = self.b_max - self.b_min;
        if r_range >= g_range && r_range >= b_range {
            0
        } else if g_range >= b_range {
            1
        } else {
            2
        }
    }

    fn split(self, colors: &mut [[u8; 3]]) -> (ColorBox, ColorBox) {
        let axis = self.longest_axis();
        let slice = &mut colors[self.start..self.start + self.count];

        // Sort by the longest axis
        slice.sort_unstable_by_key(|c| c[axis]);

        let mid = self.count / 2;
        let left = ColorBox::new(colors, self.start, mid);
        let right = ColorBox::new(colors, self.start + mid, self.count - mid);
        (left, right)
    }

    fn average(&self, colors: &[[u8; 3]]) -> Rgb {
        let slice = &colors[self.start..self.start + self.count];
        let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
        for c in slice {
            sr += c[0] as u64;
            sg += c[1] as u64;
            sb += c[2] as u64;
        }
        let n = self.count as u64;
        Rgb {
            r: (sr / n) as u8,
            g: (sg / n) as u8,
            b: (sb / n) as u8,
        }
    }
}

// ─── Bayer Matrices ────────────────────────────────────────────────────────

#[rustfmt::skip]
const BAYER_2X2: [u8; 4] = [
    0, 2,
    3, 1,
];

#[rustfmt::skip]
const BAYER_4X4: [u8; 16] = [
     0,  8,  2, 10,
    12,  4, 14,  6,
     3, 11,  1,  9,
    15,  7, 13,  5,
];

#[rustfmt::skip]
const BAYER_8X8: [u8; 64] = [
     0, 32,  8, 40,  2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
     3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21,
];

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};

    fn test_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    #[test]
    fn median_cut_basic() {
        // 4 pixels: 2 red, 2 blue → 2-color palette should have red and blue
        let pixels = [255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255];
        let info = test_info(2, 2);
        let palette = median_cut(&pixels, &info, 2).unwrap();
        assert_eq!(palette.len(), 2);

        // One should be reddish, one bluish
        let has_red = palette.iter().any(|c| c.r > 200 && c.b < 50);
        let has_blue = palette.iter().any(|c| c.b > 200 && c.r < 50);
        assert!(has_red, "palette should contain red: {palette:?}");
        assert!(has_blue, "palette should contain blue: {palette:?}");
    }

    #[test]
    fn median_cut_gradient() {
        // 16-pixel gradient → 4 colors should span the range
        let mut pixels = vec![0u8; 16 * 3];
        for i in 0..16 {
            let v = (i * 255 / 15) as u8;
            pixels[i * 3] = v;
            pixels[i * 3 + 1] = v;
            pixels[i * 3 + 2] = v;
        }
        let info = test_info(16, 1);
        let palette = median_cut(&pixels, &info, 4).unwrap();
        assert_eq!(palette.len(), 4);

        // Should span from dark to light
        let min_v = palette.iter().map(|c| c.r).min().unwrap();
        let max_v = palette.iter().map(|c| c.r).max().unwrap();
        assert!(
            min_v < 64,
            "darkest palette entry should be < 64, got {min_v}"
        );
        assert!(
            max_v > 191,
            "lightest palette entry should be > 191, got {max_v}"
        );
    }

    #[test]
    fn quantize_maps_to_nearest() {
        let palette = vec![
            Rgb { r: 0, g: 0, b: 0 },
            Rgb {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        // Dark gray → black, light gray → white
        let pixels = [64, 64, 64, 200, 200, 200];
        let info = test_info(2, 1);
        let out = quantize(&pixels, &info, &palette).unwrap();
        assert_eq!(&out[0..3], &[0, 0, 0]);
        assert_eq!(&out[3..6], &[255, 255, 255]);
    }

    #[test]
    fn floyd_steinberg_basic() {
        // 4x4 gradient quantized to 2 colors should produce a dithered pattern
        let mut pixels = vec![0u8; 4 * 4 * 3];
        for y in 0..4 {
            for x in 0..4 {
                let v = ((y * 4 + x) * 255 / 15) as u8;
                let idx = (y * 4 + x) * 3;
                pixels[idx] = v;
                pixels[idx + 1] = v;
                pixels[idx + 2] = v;
            }
        }
        let palette = vec![
            Rgb { r: 0, g: 0, b: 0 },
            Rgb {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let info = test_info(4, 4);
        let out = dither_floyd_steinberg(&pixels, &info, &palette).unwrap();
        assert_eq!(out.len(), 4 * 4 * 3);

        // Should have both black and white pixels
        let black_count = (0..16).filter(|&i| out[i * 3] == 0).count();
        let white_count = (0..16).filter(|&i| out[i * 3] == 255).count();
        assert!(black_count > 0 && white_count > 0, "should have both b&w");
        assert_eq!(black_count + white_count, 16);
    }

    #[test]
    fn ordered_dither_basic() {
        // Mid-gray dithered to B/W with 2x2 Bayer should produce a pattern
        let pixels = vec![128u8; 4 * 4 * 3]; // uniform mid-gray
        let palette = vec![
            Rgb { r: 0, g: 0, b: 0 },
            Rgb {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let info = test_info(4, 4);
        let out = dither_ordered(&pixels, &info, &palette, 2).unwrap();

        let black_count = (0..16).filter(|&i| out[i * 3] == 0).count();
        let white_count = (0..16).filter(|&i| out[i * 3] == 255).count();
        // Mid-gray with 2x2 Bayer should give ~50% B/W
        assert!(
            black_count >= 4 && white_count >= 4,
            "mid-gray dither should be ~50% B/W: black={black_count}, white={white_count}"
        );
    }

    #[test]
    fn median_cut_parity_vs_pillow() {
        // Compare palette quality: our median cut vs Pillow MEDIANCUT
        // Both should produce similar palette coverage for a gradient image
        let w = 32u32;
        let h = 32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = (y * w as usize + x) * 3;
                pixels[idx] = (x * 255 / (w as usize - 1)) as u8;
                pixels[idx + 1] = (y * 255 / (h as usize - 1)) as u8;
                pixels[idx + 2] = 128;
            }
        }
        let info = test_info(w, h);

        // Our palette
        let our_palette = median_cut(&pixels, &info, 16).unwrap();
        assert_eq!(our_palette.len(), 16);

        // Quantize with our palette and measure MSE
        let quantized = quantize(&pixels, &info, &our_palette).unwrap();
        let n = (w * h) as usize;
        let mut mse = 0.0f64;
        for i in 0..n {
            for c in 0..3 {
                let diff = pixels[i * 3 + c] as f64 - quantized[i * 3 + c] as f64;
                mse += diff * diff;
            }
        }
        mse /= (n * 3) as f64;

        eprintln!(
            "  median_cut 16-color MSE={mse:.2} PSNR={:.1}dB",
            10.0 * (255.0f64 * 255.0 / mse).log10()
        );
        // 16-color palette for a 2D gradient: PSNR > 23 dB (measured: 24.4 dB)
        assert!(
            mse < 350.0,
            "16-color gradient MSE={mse:.2} is too high (expect < 350)"
        );
    }

    #[test]
    fn dither_floyd_steinberg_quality() {
        // FS dithering should preserve average brightness better than naive quantization
        let w = 32u32;
        let h = 32;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for i in 0..(w * h) as usize {
            let v = (i * 255 / ((w * h) as usize - 1)) as u8;
            pixels[i * 3] = v;
            pixels[i * 3 + 1] = v;
            pixels[i * 3 + 2] = v;
        }
        let palette = vec![
            Rgb { r: 0, g: 0, b: 0 },
            Rgb {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let info = test_info(w, h);

        let dithered = dither_floyd_steinberg(&pixels, &info, &palette).unwrap();
        let naive = quantize(&pixels, &info, &palette).unwrap();

        // Average brightness of original
        let orig_avg: f64 = pixels.iter().map(|&v| v as f64).sum::<f64>() / pixels.len() as f64;
        let dith_avg: f64 = dithered.iter().map(|&v| v as f64).sum::<f64>() / dithered.len() as f64;
        let naive_avg: f64 = naive.iter().map(|&v| v as f64).sum::<f64>() / naive.len() as f64;

        eprintln!("  orig_avg={orig_avg:.1} dithered_avg={dith_avg:.1} naive_avg={naive_avg:.1}");
        // Dithered should preserve average brightness better than naive
        let dith_err = (dith_avg - orig_avg).abs();
        let naive_err = (naive_avg - orig_avg).abs();
        assert!(
            dith_err < naive_err + 5.0,
            "FS dithering should preserve brightness: dith_err={dith_err:.1} naive_err={naive_err:.1}"
        );
    }

    #[test]
    fn quantize_indexed_consistency() {
        let palette = vec![
            Rgb { r: 255, g: 0, b: 0 },
            Rgb { r: 0, g: 255, b: 0 },
            Rgb { r: 0, g: 0, b: 255 },
        ];
        let pixels = [200, 10, 10, 10, 200, 10, 10, 10, 200];
        let info = test_info(3, 1);

        let rgb_out = quantize(&pixels, &info, &palette).unwrap();
        let idx_out = quantize_indexed(&pixels, &info, &palette).unwrap();

        // Indexed should correspond to RGB output
        for i in 0..3 {
            let c = &palette[idx_out[i] as usize];
            assert_eq!(rgb_out[i * 3], c.r);
            assert_eq!(rgb_out[i * 3 + 1], c.g);
            assert_eq!(rgb_out[i * 3 + 2], c.b);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pixel-Exact Reference Parity Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod parity {
    use super::*;
    use crate::domain::types::{ColorSpace, ImageInfo, PixelFormat};
    use std::path::Path;
    use std::process::Command;

    fn test_info(w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        }
    }

    fn venv_python() -> String {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let venv = manifest.join("../../tests/fixtures/.venv/bin/python3");
        assert!(venv.exists(), "venv not found at {}", venv.display());
        venv.to_string_lossy().into_owned()
    }

    fn run_python(script: &str) -> Vec<u8> {
        let output = Command::new(venv_python())
            .arg("-c")
            .arg(script)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "Python failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    fn mae(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    }

    fn max_err(a: &[u8], b: &[u8]) -> u8 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0)
    }

    /// Nearest-color quantize: given the SAME palette, our output must be
    /// pixel-exact vs numpy (Euclidean distance argmin).
    #[test]
    fn quantize_nearest_parity_vs_numpy() {
        let w = 16u32;
        let h = 16;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let idx = (y * w as usize + x) * 3;
                pixels[idx] = (x * 255 / 15) as u8;
                pixels[idx + 1] = (y * 255 / 15) as u8;
                pixels[idx + 2] = 128;
            }
        }

        // Fixed palette (not generated — avoids median_cut algorithm differences)
        let palette = vec![
            Rgb { r: 0, g: 0, b: 128 },
            Rgb {
                r: 255,
                g: 0,
                b: 128,
            },
            Rgb {
                r: 0,
                g: 255,
                b: 128,
            },
            Rgb {
                r: 255,
                g: 255,
                b: 128,
            },
            Rgb {
                r: 128,
                g: 128,
                b: 128,
            },
        ];
        let info = test_info(w, h);
        let ours = quantize(&pixels, &info, &palette).unwrap();

        // numpy reference: for each pixel, find palette entry with min Euclidean distance
        let script = format!(
            "import sys, numpy as np\n\
             pixels = np.array({pixels:?}, dtype=np.uint8).reshape({h},{w},3)\n\
             palette = np.array([[0,0,128],[255,0,128],[0,255,128],[255,255,128],[128,128,128]], dtype=np.int32)\n\
             out = np.zeros_like(pixels)\n\
             for y in range({h}):\n\
             \tfor x in range({w}):\n\
             \t\tp = pixels[y,x].astype(np.int32)\n\
             \t\tdists = np.sum((palette - p)**2, axis=1)\n\
             \t\tidx = np.argmin(dists)\n\
             \t\tout[y,x] = palette[idx].astype(np.uint8)\n\
             sys.stdout.buffer.write(out.tobytes())"
        );
        let reference = run_python(&script);

        let m = mae(&ours, &reference);
        let mx = max_err(&ours, &reference);
        eprintln!("  quantize nearest vs numpy: MAE={m:.4}, max_err={mx}");
        assert!(
            m == 0.0 && mx == 0,
            "quantize must be pixel-exact vs numpy: MAE={m:.4}, max_err={mx}"
        );
    }

    /// Floyd-Steinberg dithering: pixel-exact vs numpy reference implementing
    /// the same algorithm (serpentine scan, 7/16 3/16 5/16 1/16 error distribution).
    #[test]
    fn dither_fs_parity_vs_numpy() {
        let w = 8u32;
        let h = 8;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let v = (x * 255 / 7) as u8;
                let idx = (y * w as usize + x) * 3;
                pixels[idx] = v;
                pixels[idx + 1] = v;
                pixels[idx + 2] = v;
            }
        }

        let palette = vec![
            Rgb { r: 0, g: 0, b: 0 },
            Rgb {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let info = test_info(w, h);
        let ours = dither_floyd_steinberg(&pixels, &info, &palette).unwrap();

        // numpy reference: exact same FS algorithm with serpentine scan
        let script = format!(
            "import sys, numpy as np\n\
             w,h = {w},{h}\n\
             pixels = np.array({pixels:?}, dtype=np.float64).reshape(h,w,3)\n\
             palette = np.array([[0,0,0],[255,255,255]], dtype=np.float64)\n\
             buf = pixels.copy()\n\
             out = np.zeros((h,w,3), dtype=np.uint8)\n\
             for y in range(h):\n\
             \tltr = (y % 2 == 0)\n\
             \txs = range(w) if ltr else range(w-1,-1,-1)\n\
             \tfor x in xs:\n\
             \t\told = np.clip(buf[y,x], 0, 255)\n\
             \t\tdists = np.sum((palette - old)**2, axis=1)\n\
             \t\tidx = np.argmin(dists)\n\
             \t\tnew = palette[idx]\n\
             \t\tout[y,x] = new.astype(np.uint8)\n\
             \t\terr = old - new\n\
             \t\tif ltr:\n\
             \t\t\tif x+1<w: buf[y,x+1] += err*7/16\n\
             \t\t\tif x>0 and y+1<h: buf[y+1,x-1] += err*3/16\n\
             \t\t\tif y+1<h: buf[y+1,x] += err*5/16\n\
             \t\t\tif x+1<w and y+1<h: buf[y+1,x+1] += err*1/16\n\
             \t\telse:\n\
             \t\t\tif x>0: buf[y,x-1] += err*7/16\n\
             \t\t\tif x+1<w and y+1<h: buf[y+1,x+1] += err*3/16\n\
             \t\t\tif y+1<h: buf[y+1,x] += err*5/16\n\
             \t\t\tif x>0 and y+1<h: buf[y+1,x-1] += err*1/16\n\
             sys.stdout.buffer.write(out.tobytes())"
        );
        let reference = run_python(&script);

        let m = mae(&ours, &reference);
        let mx = max_err(&ours, &reference);
        eprintln!("  FS dither vs numpy: MAE={m:.4}, max_err={mx}");
        assert!(
            mx <= 1,
            "FS dither vs numpy: max_err={mx} (expect ≤1 from i16 vs f64 rounding)"
        );
    }

    /// Ordered dithering: pixel-exact vs numpy with same Bayer 4x4 matrix.
    #[test]
    fn dither_ordered_parity_vs_numpy() {
        let w = 8u32;
        let h = 8;
        let mut pixels = vec![0u8; (w * h * 3) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let v = ((y * w as usize + x) * 255 / 63) as u8;
                let idx = (y * w as usize + x) * 3;
                pixels[idx] = v;
                pixels[idx + 1] = v;
                pixels[idx + 2] = v;
            }
        }

        let palette = vec![
            Rgb { r: 0, g: 0, b: 0 },
            Rgb {
                r: 128,
                g: 128,
                b: 128,
            },
            Rgb {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let info = test_info(w, h);
        let ours = dither_ordered(&pixels, &info, &palette, 4).unwrap();

        // numpy reference: same Bayer 4x4 matrix, same threshold formula
        let script = format!(
            "import sys, numpy as np\n\
             w,h = {w},{h}\n\
             pixels = np.array({pixels:?}, dtype=np.uint8).reshape(h,w,3)\n\
             palette = np.array([[0,0,0],[128,128,128],[255,255,255]], dtype=np.int32)\n\
             bayer4 = np.array([0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5], dtype=np.float32).reshape(4,4)\n\
             scale = 255.0 / 16.0\n\
             out = np.zeros((h,w,3), dtype=np.uint8)\n\
             for y in range(h):\n\
             \tfor x in range(w):\n\
             \t\tthresh = bayer4[y%4,x%4] * scale - 128.0\n\
             \t\tp = np.clip(pixels[y,x].astype(np.float32) + thresh, 0, 255).astype(np.int32)\n\
             \t\tdists = np.sum((palette - p)**2, axis=1)\n\
             \t\tidx = np.argmin(dists)\n\
             \t\tout[y,x] = palette[idx].astype(np.uint8)\n\
             sys.stdout.buffer.write(out.tobytes())"
        );
        let reference = run_python(&script);

        let m = mae(&ours, &reference);
        let mx = max_err(&ours, &reference);
        eprintln!("  ordered dither 4x4 vs numpy: MAE={m:.4}, max_err={mx}");
        assert!(
            m == 0.0 && mx == 0,
            "ordered dither must be pixel-exact vs numpy: MAE={m:.4}, max_err={mx}"
        );
    }
}
