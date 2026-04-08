//! Spatial filters — neighborhood operations on f32 pixel data.
//!
//! All operate on `&[f32]` RGBA (4 channels). Input includes overlap region
//! (expanded by the pipeline via `SpatialFilter::tile_overlap`). Output
//! matches input dimensions. The FilterNode wrapper handles cropping to the
//! requested tile.
//!
//! No format dispatch. No u8/u16 paths. Just f32.

pub mod bilateral;
pub mod bokeh_blur;
pub mod box_blur;
pub mod convolve;
pub mod displacement_map;
pub mod gaussian_blur;
pub mod high_pass;
pub mod lens_blur;
pub mod median;
pub mod motion_blur;
pub mod sharpen;
pub mod smart_sharpen;
pub mod spin_blur;
pub mod tilt_shift;
pub mod zoom_blur;

pub use bilateral::Bilateral;
pub use bokeh_blur::BokehBlur;
pub use box_blur::BoxBlur;
pub use convolve::Convolve;
pub use displacement_map::DisplacementMap;
pub use gaussian_blur::GaussianBlur;
pub use high_pass::HighPass;
pub use lens_blur::LensBlur;
pub use median::Median;
pub use motion_blur::MotionBlur;
pub use sharpen::Sharpen;
pub use smart_sharpen::SmartSharpen;
pub use spin_blur::SpinBlur;
pub use tilt_shift::TiltShift;
pub use zoom_blur::ZoomBlur;

// ─── Shared Helpers ──────────────────────────────────────────────────────────

/// Weighted 4-channel accumulation — structured for auto-vectorization.
///
/// LLVM sees this as a `f32x4` load + fused multiply-add per call.
/// All spatial filter inner loops delegate here for consistent SIMD codegen.
#[inline(always)]
pub(crate) fn accum4(sum: &mut [f32; 4], src: &[f32], weight: f32) {
    sum[0] += weight * src[0];
    sum[1] += weight * src[1];
    sum[2] += weight * src[2];
    sum[3] += weight * src[3];
}

/// Unweighted 4-channel accumulation (weight = 1.0).
#[inline(always)]
pub(crate) fn accum4_unit(sum: &mut [f32; 4], src: &[f32]) {
    sum[0] += src[0];
    sum[1] += src[1];
    sum[2] += src[2];
    sum[3] += src[3];
}

/// Weighted 3-channel accumulation (for filters that skip alpha).
#[inline(always)]
pub(crate) fn accum3(sum: &mut [f32; 3], src: &[f32], weight: f32) {
    sum[0] += weight * src[0];
    sum[1] += weight * src[1];
    sum[2] += weight * src[2];
}

/// Reflect-boundary coordinate clamping.
#[inline]
pub(crate) fn clamp_coord(v: i32, size: usize) -> usize {
    if v < 0 {
        (-v).min(size as i32 - 1) as usize
    } else if v >= size as i32 {
        (2 * size as i32 - v - 2).max(0) as usize
    } else {
        v as usize
    }
}

/// Generate a 1D Gaussian kernel (normalized to sum=1).
/// Default Gaussian truncation multiplier: 5 * sigma per side.
///
/// Energy capture by truncation radius:
/// - 3*sigma: 99.73% (OpenCV default — loses visible tail energy)
/// - 4*sigma: 99.9937% (DaVinci Resolve — good quality/perf balance)
/// - 5*sigma: 99.99994% (our default — effectively perfect for f32 precision)
/// - 6*sigma: 99.9999998% (overkill for f32)
///
/// Reference: Nuke (Foundry) uses ~4.4*sigma for 99.99% energy capture.
/// We use 5*sigma as our reference quality level.
pub(crate) const GAUSSIAN_SIGMA_MULTIPLIER: f32 = 5.0;

pub(crate) fn gaussian_kernel_1d(radius: f32) -> Vec<f32> {
    gaussian_kernel_1d_with_truncation(radius, GAUSSIAN_SIGMA_MULTIPLIER)
}

pub(crate) fn gaussian_kernel_1d_with_truncation(radius: f32, sigma_multiplier: f32) -> Vec<f32> {
    let sigma = radius;
    let ksize = ((sigma * 2.0 * sigma_multiplier + 1.0).round() as usize) | 1; // ensure odd
    let ksize = ksize.max(3);
    let center = ksize / 2;
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - center as f32;
        let w = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(w);
        sum += w;
    }
    let inv = 1.0 / sum;
    for w in &mut kernel {
        *w *= inv;
    }
    kernel
}

/// Generate a flat circular disc kernel.
pub(crate) fn make_disc_kernel(radius: u32) -> Vec<f32> {
    let r = radius as i32;
    let ksize = (r * 2 + 1) as usize;
    let threshold = (r as f32 + 0.5) * (r as f32 + 0.5);
    let mut kernel = vec![0.0f32; ksize * ksize];
    for ky in 0..ksize {
        for kx in 0..ksize {
            let dx = kx as f32 - r as f32;
            let dy = ky as f32 - r as f32;
            if dx * dx + dy * dy <= threshold {
                kernel[ky * ksize + kx] = 1.0;
            }
        }
    }
    kernel
}

/// Generate a regular polygon kernel with N sides and rotation.
pub(crate) fn make_polygon_kernel(radius: u32, sides: u32, rotation_deg: f32) -> Vec<f32> {
    let r = radius as i32;
    let ksize = (r * 2 + 1) as usize;
    let rf = r as f32;
    let rot = rotation_deg.to_radians();
    let sides = sides.max(3) as usize;
    let mut kernel = vec![0.0f32; ksize * ksize];

    // Precompute half-plane normals for each edge of the polygon
    let normals: Vec<(f32, f32)> = (0..sides)
        .map(|i| {
            let angle = rot + std::f32::consts::TAU * i as f32 / sides as f32;
            (angle.cos(), angle.sin())
        })
        .collect();

    for ky in 0..ksize {
        for kx in 0..ksize {
            let dx = kx as f32 - rf;
            let dy = ky as f32 - rf;
            // Point is inside polygon if it's inside ALL half-planes
            let inside = normals.iter().all(|&(nx, ny)| dx * nx + dy * ny <= rf + 0.5);
            if inside {
                kernel[ky * ksize + kx] = 1.0;
            }
        }
    }
    kernel
}

// ─── GPU Helpers ─────────────────────────────────────────────────────────────

/// Helper: generate a 1D Gaussian kernel as f32 bytes for GPU extra_buffer.
pub fn gaussian_kernel_bytes(radius: f32) -> (u32, Vec<u8>) {
    let kernel = gaussian_kernel_1d(radius);
    let center = (kernel.len() / 2) as u32;
    let bytes: Vec<u8> = kernel.iter().flat_map(|w| w.to_le_bytes()).collect();
    (center, bytes)
}

/// Helper: build width/height/radius/pad params.
pub fn blur_params(width: u32, height: u32, kernel_radius: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf.extend_from_slice(&kernel_radius.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Filter, GpuFilter};

    fn solid_rgba(w: u32, h: u32, color: [f32; 4]) -> Vec<f32> {
        let n = w as usize * h as usize;
        let mut px = Vec::with_capacity(n * 4);
        for _ in 0..n {
            px.extend_from_slice(&color);
        }
        px
    }

    fn gradient_rgba(w: u32, h: u32) -> Vec<f32> {
        let mut px = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h {
            for x in 0..w {
                px.push(x as f32 / w as f32);
                px.push(y as f32 / h as f32);
                px.push(0.5);
                px.push(1.0);
            }
        }
        px
    }

    #[test]
    fn gaussian_blur_solid_unchanged() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let blur = GaussianBlur { radius: 3.0 };
        let output = blur.compute(&input, 16, 16).unwrap();
        // Solid color blurred = same color
        assert!((output[0] - 0.5).abs() < 0.01);
        assert!((output[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn gaussian_blur_zero_radius_identity() {
        let input = gradient_rgba(8, 8);
        let blur = GaussianBlur { radius: 0.0 };
        let output = blur.compute(&input, 8, 8).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn gaussian_kernel_5sigma_sums_to_one() {
        // Verify kernel normalization for various sigma values
        for sigma in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            let kernel = gaussian_kernel_1d(sigma);
            let sum: f32 = kernel.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "kernel for sigma={sigma} sums to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn gaussian_kernel_5sigma_larger_than_3sigma() {
        // 5*sigma kernel should have more taps than the old 3*sigma
        let sigma = 5.0;
        let kernel_5s = gaussian_kernel_1d_with_truncation(sigma, 5.0);
        let kernel_3s = gaussian_kernel_1d_with_truncation(sigma, 3.0);
        assert!(
            kernel_5s.len() > kernel_3s.len(),
            "5*sigma kernel ({}) should be larger than 3*sigma kernel ({})",
            kernel_5s.len(), kernel_3s.len()
        );
        // 5*sigma: ksize = round(5*2*5+1) = 51
        // 3*sigma: ksize = round(5*2*3+1) = 31
        assert_eq!(kernel_5s.len(), 51);
        assert_eq!(kernel_3s.len(), 31);
    }

    #[test]
    fn gaussian_kernel_bytes_matches_f32_kernel() {
        let kernel = gaussian_kernel_1d(3.0);
        let (center, bytes) = gaussian_kernel_bytes(3.0);
        assert_eq!(center as usize, kernel.len() / 2);
        assert_eq!(bytes.len(), kernel.len() * 4);
        for (i, &w) in kernel.iter().enumerate() {
            let from_bytes = f32::from_le_bytes(bytes[i*4..i*4+4].try_into().unwrap());
            assert!((from_bytes - w).abs() < 1e-7, "mismatch at index {i}");
        }
    }

    #[test]
    fn box_blur_solid_unchanged() {
        let input = solid_rgba(16, 16, [0.3, 0.6, 0.9, 1.0]);
        let blur = BoxBlur { radius: 2 };
        let output = blur.compute(&input, 16, 16).unwrap();
        assert!((output[0] - 0.3).abs() < 0.01);
        assert!((output[1] - 0.6).abs() < 0.01);
    }

    #[test]
    fn sharpen_preserves_solid() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let sharp = Sharpen { radius: 2.0, amount: 1.0 };
        let output = sharp.compute(&input, 16, 16).unwrap();
        // Sharpening a solid image = no change (no edges)
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn median_removes_outlier() {
        let mut input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        // Set center pixel to an outlier
        let center = (4 * 8 + 4) * 4;
        input[center] = 1.0;
        input[center + 1] = 1.0;
        input[center + 2] = 1.0;

        let med = Median { radius: 1 };
        let output = med.compute(&input, 8, 8).unwrap();
        // Median should replace outlier with neighborhood median (~0.5)
        assert!((output[center] - 0.5).abs() < 0.01);
    }

    #[test]
    fn convolve_identity_kernel() {
        let input = gradient_rgba(8, 8);
        let identity = Convolve {
            kernel: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            kernel_width: 3,
            kernel_height: 3,
            divisor: 1.0,
        };
        let output = identity.compute(&input, 8, 8).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn bilateral_preserves_solid() {
        let input = solid_rgba(8, 8, [0.4, 0.6, 0.8, 1.0]);
        let bilat = Bilateral {
            diameter: 5,
            sigma_color: 0.1,
            sigma_space: 1.0,
        };
        let output = bilat.compute(&input, 8, 8).unwrap();
        assert!((output[0] - 0.4).abs() < 0.01);
        assert!((output[1] - 0.6).abs() < 0.01);
    }

    #[test]
    fn motion_blur_zero_length_identity() {
        let input = gradient_rgba(8, 8);
        let mb = MotionBlur { angle: 0.0, length: 0.0 };
        let output = mb.compute(&input, 8, 8).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn high_pass_solid_produces_midgray() {
        let input = solid_rgba(16, 16, [0.3, 0.3, 0.3, 1.0]);
        let hp = HighPass { radius: 3.0 };
        let output = hp.compute(&input, 16, 16).unwrap();
        // Solid -> blur = same -> high_pass = 0 + 0.5 = 0.5
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn hdr_values_survive_blur() {
        let input = solid_rgba(8, 8, [5.0, -0.5, 100.0, 1.0]);
        let blur = GaussianBlur { radius: 2.0 };
        let output = blur.compute(&input, 8, 8).unwrap();
        // HDR values not clamped
        assert!((output[0] - 5.0).abs() < 0.1);
        assert!((output[2] - 100.0).abs() < 1.0);
    }

    #[test]
    fn displacement_map_identity() {
        let (w, h) = (8u32, 8u32);
        let input = gradient_rgba(w, h);
        let mut map_x = Vec::with_capacity((w * h) as usize);
        let mut map_y = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                map_x.push(x as f32);
                map_y.push(y as f32);
            }
        }
        let dm = DisplacementMap { map_x, map_y };
        let output = dm.compute(&input, w, h).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn output_sizes_correct() {
        let input = gradient_rgba(32, 32);
        let n = 32 * 32 * 4;

        assert_eq!(GaussianBlur { radius: 3.0 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(BoxBlur { radius: 2 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(Sharpen { radius: 2.0, amount: 1.0 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(Median { radius: 1 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(HighPass { radius: 2.0 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(MotionBlur { angle: 45.0, length: 5.0 }.compute(&input, 32, 32).unwrap().len(), n);
    }

    // ── GPU wiring tests ─────────────────────────────────────────────────────

    #[test]
    fn gaussian_blur_gpu_produces_2_passes() {
        let blur = GaussianBlur { radius: 3.0 };
        let passes = blur.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 2, "GaussianBlur should have H+V passes");
        // Kernel buffer should be in extra_buffers
        assert!(!passes[0].extra_buffers.is_empty());
        assert!(!passes[1].extra_buffers.is_empty());
    }

    #[test]
    fn box_blur_gpu_produces_2_passes() {
        let blur = BoxBlur { radius: 3 };
        let passes = blur.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 2);
    }

    #[test]
    fn sharpen_gpu_produces_3_passes() {
        let sharp = Sharpen { radius: 2.0, amount: 1.0 };
        let passes = sharp.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 3, "Sharpen: blur H + blur V + unsharp apply");
    }

    #[test]
    fn high_pass_gpu_produces_3_passes() {
        let hp = HighPass { radius: 2.0 };
        let passes = hp.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 3);
    }

    #[test]
    fn median_gpu_single_pass() {
        let med = Median { radius: 1 };
        let passes = med.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 1);
        assert_eq!(med.workgroup_size(), [16, 16, 1]);
        // Params: width, height, radius, pad
        let params = med.params(64, 64);
        assert_eq!(params.len(), 16);
    }

    #[test]
    fn convolve_gpu_has_kernel_buffer() {
        let conv = Convolve {
            kernel: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            kernel_width: 3,
            kernel_height: 3,
            divisor: 1.0,
        };
        let bufs = conv.extra_buffers();
        assert_eq!(bufs.len(), 1);
        assert_eq!(bufs[0].len(), 9 * 4); // 9 f32 weights
    }

    #[test]
    fn bilateral_gpu_single_pass() {
        let bilat = Bilateral { diameter: 5, sigma_color: 0.1, sigma_space: 3.0 };
        let passes = bilat.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 1);
        // Params should be 32 bytes (8 x u32/f32)
        let params = bilat.params(64, 64);
        assert_eq!(params.len(), 32);
    }

    #[test]
    fn motion_blur_gpu_single_pass() {
        let mb = MotionBlur { angle: 45.0, length: 10.0 };
        let passes = mb.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 1);
        let params = mb.params(64, 64);
        assert_eq!(params.len(), 32);
    }

    #[test]
    fn displacement_map_gpu_has_map_buffer() {
        let n = 8 * 8;
        let dm = DisplacementMap {
            map_x: vec![0.0; n],
            map_y: vec![0.0; n],
        };
        let bufs = dm.extra_buffers();
        assert_eq!(bufs.len(), 1);
        assert_eq!(bufs[0].len(), n * 8); // 2 x f32 per pixel
    }

    #[test]
    fn all_spatial_filters_have_gpu() {
        // Every spatial filter must implement GpuFilter -- verify gpu_shaders() is non-empty
        let w = 32u32;
        let h = 32u32;

        assert!(!GaussianBlur { radius: 3.0 }.gpu_shaders(w, h).is_empty());
        assert!(!BoxBlur { radius: 2 }.gpu_shaders(w, h).is_empty());
        assert!(!Sharpen { radius: 2.0, amount: 1.0 }.gpu_shaders(w, h).is_empty());
        assert!(!Median { radius: 1 }.gpu_shaders(w, h).is_empty());
        assert!(!Convolve {
            kernel: vec![1.0; 9], kernel_width: 3, kernel_height: 3, divisor: 9.0,
        }.gpu_shaders(w, h).is_empty());
        assert!(!Bilateral { diameter: 5, sigma_color: 0.1, sigma_space: 3.0 }.gpu_shaders(w, h).is_empty());
        assert!(!MotionBlur { angle: 0.0, length: 5.0 }.gpu_shaders(w, h).is_empty());
        assert!(!HighPass { radius: 2.0 }.gpu_shaders(w, h).is_empty());
        assert!(!DisplacementMap { map_x: vec![0.0; 1024], map_y: vec![0.0; 1024] }.gpu_shaders(w, h).is_empty());
    }
}
