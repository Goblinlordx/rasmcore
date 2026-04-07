//! Spatial filters — neighborhood operations on f32 pixel data.
//!
//! All operate on `&[f32]` RGBA (4 channels). Input includes overlap region
//! (expanded by the pipeline via `SpatialFilter::tile_overlap`). Output
//! matches input dimensions. The FilterNode wrapper handles cropping to the
//! requested tile.
//!
//! No format dispatch. No u8/u16 paths. Just f32.

use crate::node::{GpuShader, PipelineError};
use crate::ops::{Filter, GpuFilter};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Weighted 4-channel accumulation — structured for auto-vectorization.
///
/// LLVM sees this as a `f32x4` load + fused multiply-add per call.
/// All spatial filter inner loops delegate here for consistent SIMD codegen.
#[inline(always)]
fn accum4(sum: &mut [f32; 4], src: &[f32], weight: f32) {
    sum[0] += weight * src[0];
    sum[1] += weight * src[1];
    sum[2] += weight * src[2];
    sum[3] += weight * src[3];
}

/// Unweighted 4-channel accumulation (weight = 1.0).
#[inline(always)]
fn accum4_unit(sum: &mut [f32; 4], src: &[f32]) {
    sum[0] += src[0];
    sum[1] += src[1];
    sum[2] += src[2];
    sum[3] += src[3];
}

/// Weighted 3-channel accumulation (for filters that skip alpha).
#[inline(always)]
fn accum3(sum: &mut [f32; 3], src: &[f32], weight: f32) {
    sum[0] += weight * src[0];
    sum[1] += weight * src[1];
    sum[2] += weight * src[2];
}

/// Reflect-boundary coordinate clamping.
#[inline]
fn clamp_coord(v: i32, size: usize) -> usize {
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
const GAUSSIAN_SIGMA_MULTIPLIER: f32 = 5.0;

fn gaussian_kernel_1d(radius: f32) -> Vec<f32> {
    gaussian_kernel_1d_with_truncation(radius, GAUSSIAN_SIGMA_MULTIPLIER)
}

fn gaussian_kernel_1d_with_truncation(radius: f32, sigma_multiplier: f32) -> Vec<f32> {
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

// ─── Gaussian Blur ────────────────────────────────────────────────────────────

/// Gaussian blur — separable convolution on f32 data.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gaussian_blur", category = "spatial", cost = "O(n * r) separable", doc = "docs/operations/filters/spatial/gaussian_blur.adoc")]
pub struct GaussianBlur {
    /// Blur radius in pixels. Larger values produce stronger blur.
    #[param(min = 0.0, max = 100.0, default = 1.0)]
    pub radius: f32,
}

impl Filter for GaussianBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let kernel = gaussian_kernel_1d(self.radius);
        let r = kernel.len() / 2;

        // Pass 1: horizontal — fast interior path skips clamp_coord for most pixels
        let mut tmp = vec![0.0f32; w * h * 4];
        for y in 0..h {
            let row_base = y * w * 4;

            // Boundary pixels (left + right margins)
            for x in 0..r.min(w) {
                let mut sum = [0.0f32; 4];
                for (kx, &kw) in kernel.iter().enumerate() {
                    let sx = clamp_coord(x as i32 + kx as i32 - r as i32, w);
                    let idx = row_base + sx * 4;
                    accum4(&mut sum, &input[idx..], kw);
                }
                let out_idx = row_base + x * 4;
                tmp[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
            for x in (w.saturating_sub(r))..w {
                if x < r { continue; } // already handled above
                let mut sum = [0.0f32; 4];
                for (kx, &kw) in kernel.iter().enumerate() {
                    let sx = clamp_coord(x as i32 + kx as i32 - r as i32, w);
                    let idx = row_base + sx * 4;
                    accum4(&mut sum, &input[idx..], kw);
                }
                let out_idx = row_base + x * 4;
                tmp[out_idx..out_idx + 4].copy_from_slice(&sum);
            }

            // Interior pixels — no bounds check needed
            let x_start = r.min(w);
            let x_end = w.saturating_sub(r);
            for x in x_start..x_end {
                let mut sum = [0.0f32; 4];
                // All source positions (x-r..x+r) are in bounds
                let base = row_base + (x - r) * 4;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let idx = base + ki * 4;
                    accum4(&mut sum, &input[idx..], kw);
                }
                let out_idx = row_base + x * 4;
                tmp[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
        }

        // Pass 2: vertical — fast interior path
        let mut out = vec![0.0f32; w * h * 4];
        let stride = w * 4;

        // Boundary rows (top + bottom margins)
        for y in 0..r.min(h) {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for (ky, &kw) in kernel.iter().enumerate() {
                    let sy = clamp_coord(y as i32 + ky as i32 - r as i32, h);
                    let idx = sy * stride + x * 4;
                    accum4(&mut sum, &tmp[idx..], kw);
                }
                let out_idx = y * stride + x * 4;
                out[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
        }
        for y in (h.saturating_sub(r))..h {
            if y < r { continue; }
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for (ky, &kw) in kernel.iter().enumerate() {
                    let sy = clamp_coord(y as i32 + ky as i32 - r as i32, h);
                    let idx = sy * stride + x * 4;
                    accum4(&mut sum, &tmp[idx..], kw);
                }
                let out_idx = y * stride + x * 4;
                out[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
        }

        // Interior rows — no bounds check needed
        let y_start = r.min(h);
        let y_end = h.saturating_sub(r);
        for y in y_start..y_end {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                let px_offset = x * 4;
                let base_row = (y - r) * stride;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let idx = base_row + ki * stride + px_offset;
                    accum4(&mut sum, &tmp[idx..], kw);
                }
                let out_idx = y * stride + x * 4;
                out[out_idx..out_idx + 4].copy_from_slice(&sum);
            }
        }

        Ok(out)
    }
}

impl GaussianBlur {
    pub fn tile_overlap(&self) -> u32 {
        (self.radius * 3.0).ceil() as u32
    }
}

// ─── Box Blur ─────────────────────────────────────────────────────────────────

/// Box blur — running average within radius. O(1) per pixel.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "box_blur", category = "spatial", cost = "O(n * r) separable")]
pub struct BoxBlur {
    #[param(min = 0, max = 100, default = 1)]
    pub radius: u32,
}

impl Filter for BoxBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius == 0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as i32;
        let diameter = (2 * r + 1) as f32;
        let inv_d = 1.0 / diameter;

        // Horizontal pass
        let mut tmp = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for dx in -r..=r {
                    let sx = clamp_coord(x as i32 + dx, w);
                    let idx = (y * w + sx) * 4;
                    accum4_unit(&mut sum, &input[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    tmp[out_idx + c] = sum[c] * inv_d;
                }
            }
        }

        // Vertical pass
        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for dy in -r..=r {
                    let sy = clamp_coord(y as i32 + dy, h);
                    let idx = (sy * w + x) * 4;
                    accum4_unit(&mut sum, &tmp[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    out[out_idx + c] = sum[c] * inv_d;
                }
            }
        }

        Ok(out)
    }

    fn tile_overlap(&self) -> u32 {
        self.radius
    }
}

// ─── Sharpen (Unsharp Mask) ───────────────────────────────────────────────────

/// Unsharp mask sharpening — enhances edges by subtracting blurred from original.
///
/// `output = input + amount * (input - blur(input, radius))`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sharpen", category = "spatial", cost = "O(n * r) via gaussian_blur")]
pub struct Sharpen {
    #[param(min = 0.0, max = 100.0, default = 1.0)]
    pub radius: f32,
    #[param(min = 0.0, max = 10.0, default = 1.0)]
    pub amount: f32,
}

impl Filter for Sharpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.radius };
        let blurred = blur.compute(input, width, height)?;
        let amount = self.amount;
        let mut out = Vec::with_capacity(input.len());
        for (i, &v) in input.iter().enumerate() {
            // Alpha (every 4th channel) preserved unchanged
            if i % 4 == 3 {
                out.push(v);
            } else {
                out.push(v + amount * (v - blurred[i]));
            }
        }
        Ok(out)
    }
}

// ─── Median ───────────────────────────────────────────────────────────────────

/// Median filter — replaces each pixel with median of its neighborhood.
///
/// Effective for salt-and-pepper noise removal while preserving edges.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "median", category = "spatial", cost = "O(n * r^2 * log r)")]
pub struct Median {
    #[param(min = 0, max = 50, default = 1)]
    pub radius: u32,
}

impl Filter for Median {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius == 0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as i32;
        let mut out = vec![0.0f32; w * h * 4];
        let mut neighborhood = Vec::new();

        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    if c == 3 {
                        // Alpha: pass through
                        out[(y * w + x) * 4 + 3] = input[(y * w + x) * 4 + 3];
                        continue;
                    }
                    neighborhood.clear();
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let sx = clamp_coord(x as i32 + dx, w);
                            let sy = clamp_coord(y as i32 + dy, h);
                            neighborhood.push(input[(sy * w + sx) * 4 + c]);
                        }
                    }
                    neighborhood.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    out[(y * w + x) * 4 + c] = neighborhood[neighborhood.len() / 2];
                }
            }
        }

        Ok(out)
    }
}

// ─── General Convolution ──────────────────────────────────────────────────────

/// General NxN convolution with arbitrary kernel.
#[derive(Clone)]
pub struct Convolve {
    pub kernel: Vec<f32>,
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub divisor: f32,
}

impl Filter for Convolve {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let kw = self.kernel_width as usize;
        let kh = self.kernel_height as usize;
        let rw = kw / 2;
        let rh = kh / 2;
        let inv_div = 1.0 / self.divisor;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sx = clamp_coord(x as i32 + kx as i32 - rw as i32, w);
                        let sy = clamp_coord(y as i32 + ky as i32 - rh as i32, h);
                        let k = self.kernel[ky * kw + kx];
                        let idx = (sy * w + sx) * 4;
                        accum4(&mut sum, &input[idx..], k);
                    }
                }
                let out_idx = (y * w + x) * 4;
                out[out_idx] = sum[0] * inv_div;
                out[out_idx + 1] = sum[1] * inv_div;
                out[out_idx + 2] = sum[2] * inv_div;
                out[out_idx + 3] = sum[3] * inv_div;
            }
        }

        Ok(out)
    }
}

// ─── Bilateral Filter ─────────────────────────────────────────────────────────

/// Bilateral filter — edge-preserving smoothing.
///
/// Weights pixels by both spatial distance and color similarity.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "bilateral", category = "spatial", cost = "O(n * d^2)")]
pub struct Bilateral {
    #[param(min = 1, max = 50, default = 5)]
    pub diameter: u32,
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub sigma_color: f32,
    #[param(min = 0.0, max = 100.0, default = 10.0)]
    pub sigma_space: f32,
}

impl Filter for Bilateral {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = (self.diameter / 2) as i32;
        let sc2 = -0.5 / (self.sigma_color * self.sigma_color);
        let ss2 = -0.5 / (self.sigma_space * self.sigma_space);
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let center_idx = (y * w + x) * 4;
                let mut sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = clamp_coord(x as i32 + dx, w);
                        let sy = clamp_coord(y as i32 + dy, h);
                        let idx = (sy * w + sx) * 4;

                        // Spatial weight
                        let dist2 = (dx * dx + dy * dy) as f32;
                        let ws = (dist2 * ss2).exp();

                        // Color weight (Euclidean distance in RGB)
                        let dr = input[idx] - input[center_idx];
                        let dg = input[idx + 1] - input[center_idx + 1];
                        let db = input[idx + 2] - input[center_idx + 2];
                        let color_dist2 = dr * dr + dg * dg + db * db;
                        let wc = (color_dist2 * sc2).exp();

                        let weight = ws * wc;
                        accum3(&mut sum, &input[idx..], weight);
                        weight_sum += weight;
                    }
                }

                let inv_w = if weight_sum > 1e-10 { 1.0 / weight_sum } else { 0.0 };
                out[center_idx] = sum[0] * inv_w;
                out[center_idx + 1] = sum[1] * inv_w;
                out[center_idx + 2] = sum[2] * inv_w;
                out[center_idx + 3] = input[center_idx + 3]; // alpha
            }
        }

        Ok(out)
    }
}

// ─── Motion Blur ──────────────────────────────────────────────────────────────

/// Motion blur — linear directional blur.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "motion_blur", category = "spatial", cost = "O(n * length)")]
pub struct MotionBlur {
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle: f32,  // degrees
    #[param(min = 0.0, max = 200.0, default = 10.0)]
    pub length: f32, // pixels
}

impl Filter for MotionBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.length <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let rad = self.angle.to_radians();
        let dx = rad.cos();
        let dy = rad.sin();
        let steps = self.length.ceil() as usize;
        let inv_steps = 1.0 / (steps as f32 + 1.0);
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for s in 0..=steps {
                    let t = s as f32 - steps as f32 * 0.5;
                    let sx = clamp_coord((x as f32 + t * dx).round() as i32, w);
                    let sy = clamp_coord((y as f32 + t * dy).round() as i32, h);
                    let idx = (sy * w + sx) * 4;
                    accum4_unit(&mut sum, &input[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                out[out_idx] = sum[0] * inv_steps;
                out[out_idx + 1] = sum[1] * inv_steps;
                out[out_idx + 2] = sum[2] * inv_steps;
                out[out_idx + 3] = sum[3] * inv_steps;
            }
        }

        Ok(out)
    }
}

// ─── High Pass ────────────────────────────────────────────────────────────────

/// High pass filter — subtracts blur from original, adding mid-gray offset.
///
/// `output = (input - blur(input)) + 0.5`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "high_pass", category = "spatial", cost = "O(n * r) via gaussian_blur")]
pub struct HighPass {
    #[param(min = 0.0, max = 100.0, default = 3.0)]
    pub radius: f32,
}

impl Filter for HighPass {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.radius };
        let blurred = blur.compute(input, width, height)?;
        let mut out = Vec::with_capacity(input.len());
        for (i, &v) in input.iter().enumerate() {
            if i % 4 == 3 {
                out.push(v); // alpha
            } else {
                out.push((v - blurred[i]) + 0.5);
            }
        }
        Ok(out)
    }
}

// ─── Displacement Map ─────────────────────────────────────────────────────────

/// Displacement map — warp pixels by per-pixel offset fields.
///
/// `map_x` and `map_y` are f32 slices of absolute source coordinates.
#[derive(Clone)]
pub struct DisplacementMap {
    pub map_x: Vec<f32>,
    pub map_y: Vec<f32>,
}

impl Filter for DisplacementMap {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        if self.map_x.len() != n || self.map_y.len() != n {
            return Err(PipelineError::InvalidParams(format!(
                "displacement map size mismatch: expected {n}, got x={} y={}",
                self.map_x.len(),
                self.map_y.len()
            )));
        }

        let mut out = vec![0.0f32; n * 4];
        let wi = w as i32;
        let hi = h as i32;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let sx = self.map_x[idx];
                let sy = self.map_y[idx];

                // Bilinear interpolation
                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let sample = |px: i32, py: i32, c: usize| -> f32 {
                    if px >= 0 && px < wi && py >= 0 && py < hi {
                        input[(py as usize * w + px as usize) * 4 + c]
                    } else {
                        0.0
                    }
                };

                let out_off = idx * 4;
                for c in 0..4 {
                    let v = sample(x0, y0, c) * (1.0 - fx) * (1.0 - fy)
                        + sample(x0 + 1, y0, c) * fx * (1.0 - fy)
                        + sample(x0, y0 + 1, c) * (1.0 - fx) * fy
                        + sample(x0 + 1, y0 + 1, c) * fx * fy;
                    out[out_off + c] = v;
                }
            }
        }

        Ok(out)
    }
}

// ─── Lens Blur ───────────────────────────────────────────────────────────────

/// Lens blur — polygon/disc-shaped bokeh kernel convolution.
///
/// Simulates camera lens defocus with configurable aperture blade count.
/// `blade_count=0` produces a circular disc; `blade_count>=3` produces a polygon.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "lens_blur", category = "spatial", cost = "O(n * r^2)")]
pub struct LensBlur {
    #[param(min = 0, max = 50, default = 5)]
    pub radius: u32,
    /// Aperture blade count. 0 = disc, 5-8 = polygon.
    #[param(min = 0, max = 12, default = 0)]
    pub blade_count: u32,
    /// Blade rotation in degrees.
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub rotation: f32,
}

impl Filter for LensBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius == 0 {
            return Ok(input.to_vec());
        }
        let kernel = if self.blade_count < 3 {
            make_disc_kernel(self.radius)
        } else {
            make_polygon_kernel(self.radius, self.blade_count, self.rotation)
        };
        let ksize = (self.radius * 2 + 1) as usize;
        let divisor: f32 = kernel.iter().sum();
        let conv = Convolve {
            kernel,
            kernel_width: ksize as u32,
            kernel_height: ksize as u32,
            divisor: divisor.max(1e-6),
        };
        conv.compute(input, width, height)
    }

    fn tile_overlap(&self) -> u32 {
        self.radius
    }
}

// ─── Bokeh Blur ──────────────────────────────────────────────────────────────

/// Bokeh blur — disc or hexagon-shaped kernel blur.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "bokeh_blur", category = "spatial", cost = "O(n * r^2)")]
pub struct BokehBlur {
    #[param(min = 1, max = 50, default = 5)]
    pub radius: u32,
    /// Shape: 0 = disc, 1 = hexagon.
    #[param(min = 0, max = 1, default = 0)]
    pub shape: u32,
}

impl Filter for BokehBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let kernel = if self.shape == 1 {
            make_polygon_kernel(self.radius, 6, 0.0) // hexagon
        } else {
            make_disc_kernel(self.radius)
        };
        let ksize = (self.radius * 2 + 1) as usize;
        let divisor: f32 = kernel.iter().sum();
        let conv = Convolve {
            kernel,
            kernel_width: ksize as u32,
            kernel_height: ksize as u32,
            divisor: divisor.max(1e-6),
        };
        conv.compute(input, width, height)
    }

    fn tile_overlap(&self) -> u32 {
        self.radius
    }
}

// ─── Tilt Shift ──────────────────────────────────────────────────────────────

/// Tilt-shift — selective focus with gradient blur falloff.
///
/// Simulates miniature/tilt-shift photography by blurring areas outside
/// a focus band and blending with the original via a smooth gradient mask.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "tilt_shift", category = "spatial", cost = "O(n * radius) via gaussian_blur")]
pub struct TiltShift {
    /// Focus band center position (0.0 = top, 1.0 = bottom).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub focus_position: f32,
    /// Focus band size as fraction of image height.
    #[param(min = 0.0, max = 1.0, default = 0.2)]
    pub band_size: f32,
    /// Maximum blur radius for out-of-focus areas.
    #[param(min = 0.0, max = 100.0, default = 8.0)]
    pub blur_radius: f32,
    /// Band rotation angle in degrees.
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle: f32,
}

impl Filter for TiltShift {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.blur_radius <= 0.0 || self.band_size >= 1.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;

        // Generate fully blurred version
        let blur = GaussianBlur { radius: self.blur_radius };
        let blurred = blur.compute(input, width, height)?;

        // Blend original and blurred based on distance from focus band
        let angle_rad = self.angle.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let focus_y = self.focus_position;
        let half_band = self.band_size * 0.5;
        let transition = half_band.max(0.05); // transition zone

        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let nx = x as f32 / w as f32 - 0.5;
                let ny = y as f32 / h as f32 - focus_y;
                // Rotate to align with band angle
                let dist = (nx * sin_a + ny * cos_a).abs();
                // Compute mask: 0 inside band, 1 far outside
                let t = if dist < half_band {
                    0.0
                } else {
                    let d = (dist - half_band) / transition;
                    let d = d.min(1.0);
                    d * d * (3.0 - 2.0 * d) // smoothstep
                };
                let idx = (y * w + x) * 4;
                for c in 0..4 {
                    out[idx + c] = input[idx + c] + t * (blurred[idx + c] - input[idx + c]);
                }
            }
        }
        Ok(out)
    }
}

// ─── Zoom Blur ───────────────────────────────────────────────────────────────

/// Zoom blur — radial blur from a center point.
///
/// Samples along radial lines from each pixel toward the center,
/// producing a camera zoom effect.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "zoom_blur", category = "spatial", cost = "O(n * samples)")]
pub struct ZoomBlur {
    /// Center X position (0.0-1.0).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub center_x: f32,
    /// Center Y position (0.0-1.0).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub center_y: f32,
    /// Zoom factor. Larger = more blur.
    #[param(min = -1.0, max = 1.0, step = 0.01, default = 0.1)]
    pub factor: f32,
}

impl Filter for ZoomBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.factor.abs() < 1e-6 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let wi = w as i32;
        let hi = h as i32;
        let cx = self.center_x * w as f32;
        let cy = self.center_y * h as f32;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let dx = cx - x as f32;
                let dy = cy - y as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                let samples = ((dist * self.factor.abs()).ceil() as usize).clamp(3, 64);
                let inv_n = 1.0 / samples as f32;

                let mut sum = [0.0f32; 4];
                for s in 0..samples {
                    let t = s as f32 * self.factor / samples as f32;
                    let sx = (x as f32 + dx * t).round() as i32;
                    let sy = (y as f32 + dy * t).round() as i32;
                    let sx = sx.clamp(0, wi - 1) as usize;
                    let sy = sy.clamp(0, hi - 1) as usize;
                    let idx = (sy * w + sx) * 4;
                    accum4_unit(&mut sum, &input[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                out[out_idx] = sum[0] * inv_n;
                out[out_idx + 1] = sum[1] * inv_n;
                out[out_idx + 2] = sum[2] * inv_n;
                out[out_idx + 3] = sum[3] * inv_n;
            }
        }
        Ok(out)
    }
}

// ─── Spin Blur ───────────────────────────────────────────────────────────────

/// Spin blur — rotational blur around a center point.
///
/// Samples along circular arcs around the center, producing a
/// rotational motion effect.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "spin_blur", category = "spatial", cost = "O(n * samples)")]
pub struct SpinBlur {
    /// Center X position (0.0-1.0).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub center_x: f32,
    /// Center Y position (0.0-1.0).
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub center_y: f32,
    /// Maximum rotation angle in degrees.
    #[param(min = 0.0, max = 360.0, default = 10.0)]
    pub angle: f32,
}

impl Filter for SpinBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.angle.abs() < 1e-6 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let wi = w as i32;
        let hi = h as i32;
        let cx = self.center_x * w as f32;
        let cy = self.center_y * h as f32;
        let angle_rad = self.angle.to_radians();

        // Sample count based on angle and max radius
        let half_diag = ((w * w + h * h) as f32).sqrt() * 0.5;
        let n = ((2.0 * angle_rad.abs() * half_diag).ceil() as usize + 2) | 1; // ensure odd
        let n = n.clamp(3, 129);
        let inv_n = 1.0 / n as f32;

        // Precompute rotation table
        let half = n / 2;
        let cos_sin: Vec<(f32, f32)> = (0..n)
            .map(|i| {
                let offset = angle_rad * (i as f32 - half as f32) / n as f32;
                (offset.cos(), offset.sin())
            })
            .collect();

        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let mut sum = [0.0f32; 4];
                for &(cos_t, sin_t) in &cos_sin {
                    let sx = (cx + dx * cos_t - dy * sin_t).round() as i32;
                    let sy = (cy + dx * sin_t + dy * cos_t).round() as i32;
                    let sx = sx.clamp(0, wi - 1) as usize;
                    let sy = sy.clamp(0, hi - 1) as usize;
                    let idx = (sy * w + sx) * 4;
                    accum4_unit(&mut sum, &input[idx..]);
                }
                let out_idx = (y * w + x) * 4;
                out[out_idx] = sum[0] * inv_n;
                out[out_idx + 1] = sum[1] * inv_n;
                out[out_idx + 2] = sum[2] * inv_n;
                out[out_idx + 3] = sum[3] * inv_n;
            }
        }
        Ok(out)
    }
}

// ─── Smart Sharpen ───────────────────────────────────────────────────────────

/// Smart sharpen — edge-preserving sharpening via bilateral unsharp mask.
///
/// Uses a bilateral filter (instead of Gaussian) for the blur step,
/// preserving edges while sharpening. Formula:
/// `output = input + amount * (input - bilateral_blur(input))`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "smart_sharpen", category = "spatial", cost = "O(n * radius^2) via bilateral")]
pub struct SmartSharpen {
    /// Sharpening strength.
    #[param(min = 0.0, max = 5.0, default = 1.0)]
    pub amount: f32,
    /// Bilateral filter radius.
    #[param(min = 1, max = 20, default = 3)]
    pub radius: u32,
    /// Edge preservation threshold (bilateral sigma_color).
    #[param(min = 0.01, max = 1.0, default = 0.1)]
    pub threshold: f32,
}

impl Filter for SmartSharpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.amount.abs() < 1e-6 {
            return Ok(input.to_vec());
        }
        // Edge-preserving blur via bilateral
        let bilateral = Bilateral {
            diameter: self.radius * 2 + 1,
            sigma_color: self.threshold,
            sigma_space: self.radius as f32,
        };
        let blurred = bilateral.compute(input, width, height)?;

        // Unsharp mask: output = input + amount * (input - blurred)
        let amount = self.amount;
        let mut out = input.to_vec();
        for (i, pixel) in out.chunks_exact_mut(4).enumerate() {
            let bi = i * 4;
            pixel[0] += amount * (pixel[0] - blurred[bi]);
            pixel[1] += amount * (pixel[1] - blurred[bi + 1]);
            pixel[2] += amount * (pixel[2] - blurred[bi + 2]);
            // alpha unchanged
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 {
        self.radius + 4 // bilateral radius + safety margin
    }
}

// ─── Kernel Generators ───────────────────────────────────────────────────────

/// Generate a flat circular disc kernel.
fn make_disc_kernel(radius: u32) -> Vec<f32> {
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
fn make_polygon_kernel(radius: u32, sides: u32, rotation_deg: f32) -> Vec<f32> {
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

// ═══════════════════════════════════════════════════════════════════════════════
// GPU filter implementations — declarative via gpu_filter! macros
// ═══════════════════════════════════════════════════════════════════════════════

use crate::gpu_shaders::spatial;

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

// ── GaussianBlur GPU (2-pass separable H+V) ──────────────────────────────────

gpu_filter_passes_only!(GaussianBlur,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let params = blur_params(w, h, kr);
        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], params.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], params)
                .with_extra_buffers(vec![kb]),
        ]
    }
);

// ── BoxBlur GPU (2-pass separable H+V) ───────────────────────────────────────

gpu_filter_passes_only!(BoxBlur,
    passes(self_, w, h) => {
        let params = blur_params(w, h, self_.radius);
        vec![
            GpuShader::new(spatial::BOX_BLUR_H.to_string(), "main", [256, 1, 1], params.clone()),
            GpuShader::new(spatial::BOX_BLUR_V.to_string(), "main", [256, 1, 1], params),
        ]
    }
);

// ── Sharpen GPU (blur H + blur V + unsharp apply) ────────────────────────────

gpu_filter_passes_only!(Sharpen,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let blur_p = blur_params(w, h, kr);
        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&self_.amount.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], blur_p.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], blur_p)
                .with_extra_buffers(vec![kb]),
            // Apply shader reads blurred input; extra_buffer = original (snapshotted before blur)
            GpuShader::new(spatial::SHARPEN_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);

// ── HighPass GPU (blur H + blur V + subtract apply) ─────────────────────────

gpu_filter_passes_only!(HighPass,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let blur_p = blur_params(w, h, kr);
        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], blur_p.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], blur_p)
                .with_extra_buffers(vec![kb]),
            GpuShader::new(spatial::HIGH_PASS_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);

// ── Median GPU (single-pass sorting network) ────────────────────────────────

gpu_filter!(Median,
    shader: spatial::MEDIAN,
    workgroup: [16, 16, 1],
    params(self_, w, h) => [w, h, self_.radius, 0u32]
);

// ── Convolve GPU (single-pass with kernel extra_buffer) ─────────────────────

impl GpuFilter for Convolve {
    fn shader_body(&self) -> &str { spatial::CONVOLVE }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&self.kernel_width.to_le_bytes());
        buf.extend_from_slice(&self.kernel_height.to_le_bytes());
        buf.extend_from_slice(&(1.0f32 / self.divisor).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Pack kernel weights as f32 bytes
        let mut kb = Vec::with_capacity(self.kernel.len() * 4);
        for &w in &self.kernel {
            kb.extend_from_slice(&w.to_le_bytes());
        }
        vec![kb]
    }
}

// ── Bilateral GPU (single-pass neighborhood) ────────────────────────────────

gpu_filter!(Bilateral,
    shader: spatial::BILATERAL,
    workgroup: [16, 16, 1],
    params(self_, w, h) => [
        w, h, self_.diameter / 2, 0u32,
        -0.5f32 / (self_.sigma_color * self_.sigma_color),
        -0.5f32 / (self_.sigma_space * self_.sigma_space),
        0u32, 0u32
    ]
);

// ── MotionBlur GPU (single-pass directional) ────────────────────────────────

gpu_filter!(MotionBlur,
    shader: spatial::MOTION_BLUR,
    workgroup: [256, 1, 1],
    params(self_, w, h) => [
        w, h, self_.length.ceil() as u32, 0u32,
        self_.angle.to_radians().cos(),
        self_.angle.to_radians().sin(),
        1.0f32 / (self_.length.ceil() + 1.0),
        0u32
    ]
);

// ── DisplacementMap GPU (single-pass with map extra_buffer) ─────────────────

impl GpuFilter for DisplacementMap {
    fn shader_body(&self) -> &str { spatial::DISPLACEMENT_MAP }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Interleave map_x and map_y as f32 pairs
        let mut buf = Vec::with_capacity(self.map_x.len() * 8);
        for (&mx, &my) in self.map_x.iter().zip(self.map_y.iter()) {
            buf.extend_from_slice(&mx.to_le_bytes());
            buf.extend_from_slice(&my.to_le_bytes());
        }
        vec![buf]
    }
}

// ── LensBlur / BokehBlur GPU (kernel convolution via extra_buffer) ──────────

impl GpuFilter for LensBlur {
    fn shader_body(&self) -> &str { spatial::LENS_BLUR }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let ksize = self.radius * 2 + 1;
        let kernel = if self.blade_count < 3 {
            make_disc_kernel(self.radius)
        } else {
            make_polygon_kernel(self.radius, self.blade_count, self.rotation)
        };
        let divisor: f32 = kernel.iter().sum::<f32>().max(1e-6);
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&ksize.to_le_bytes());
        buf.extend_from_slice(&ksize.to_le_bytes());
        buf.extend_from_slice(&(1.0f32 / divisor).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        let kernel = if self.blade_count < 3 {
            make_disc_kernel(self.radius)
        } else {
            make_polygon_kernel(self.radius, self.blade_count, self.rotation)
        };
        let mut kb = Vec::with_capacity(kernel.len() * 4);
        for &w in &kernel {
            kb.extend_from_slice(&w.to_le_bytes());
        }
        vec![kb]
    }
}

impl GpuFilter for BokehBlur {
    fn shader_body(&self) -> &str { spatial::LENS_BLUR }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let ksize = self.radius * 2 + 1;
        let kernel = if self.shape == 1 {
            make_polygon_kernel(self.radius, 6, 0.0)
        } else {
            make_disc_kernel(self.radius)
        };
        let divisor: f32 = kernel.iter().sum::<f32>().max(1e-6);
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&ksize.to_le_bytes());
        buf.extend_from_slice(&ksize.to_le_bytes());
        buf.extend_from_slice(&(1.0f32 / divisor).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        let kernel = if self.shape == 1 {
            make_polygon_kernel(self.radius, 6, 0.0)
        } else {
            make_disc_kernel(self.radius)
        };
        let mut kb = Vec::with_capacity(kernel.len() * 4);
        for &w in &kernel {
            kb.extend_from_slice(&w.to_le_bytes());
        }
        vec![kb]
    }
}

// ── ZoomBlur GPU (single-pass radial sampling) ──────────────────────────────

gpu_filter!(ZoomBlur,
    shader: spatial::ZOOM_BLUR,
    workgroup: [256, 1, 1],
    params(self_, w, h) => [
        w, h,
        ((self_.factor.abs() * 64.0).ceil() as u32).clamp(8, 128),
        0u32,
        self_.center_x * w as f32,
        self_.center_y * h as f32,
        self_.factor,
        0u32
    ]
);

// ── SpinBlur GPU (single-pass rotational sampling) ──────────────────────────

gpu_filter!(SpinBlur,
    shader: spatial::SPIN_BLUR,
    workgroup: [256, 1, 1],
    params(self_, w, h) => [
        w, h,
        ((self_.angle.to_radians().abs() * 32.0).ceil() as u32).clamp(8, 128),
        0u32,
        self_.center_x * w as f32,
        self_.center_y * h as f32,
        self_.angle.to_radians(),
        0u32
    ]
);

// ── TiltShift GPU (blur pass + blend pass) ──────────────────────────────────

impl GpuFilter for TiltShift {
    fn shader_body(&self) -> &str { "" } // multi-pass only
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, _w: u32, _h: u32) -> Vec<u8> { Vec::new() }

    fn gpu_shaders(&self, w: u32, h: u32) -> Vec<GpuShader> {
        if self.blur_radius <= 0.0 || self.band_size >= 1.0 {
            return vec![self.gpu_shader(w, h)];
        }
        let (krad, kernel_bytes) = gaussian_kernel_bytes(self.blur_radius);
        let n = w * h;

        // Pass 1: horizontal blur
        let h_params = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&w.to_le_bytes());
            buf.extend_from_slice(&h.to_le_bytes());
            buf.extend_from_slice(&krad.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };
        // Pass 2: vertical blur
        let v_params = h_params.clone();
        // Pass 3: blend with mask
        let angle_rad = self.angle.to_radians();
        let half_band = self.band_size * 0.5;
        let transition = half_band.max(0.05);
        let blend_params = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&w.to_le_bytes());
            buf.extend_from_slice(&h.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&self.focus_position.to_le_bytes());
            buf.extend_from_slice(&half_band.to_le_bytes());
            buf.extend_from_slice(&transition.to_le_bytes());
            buf.extend_from_slice(&angle_rad.cos().to_le_bytes());
            buf.extend_from_slice(&angle_rad.sin().to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };

        vec![
            GpuShader {
                body: spatial::GAUSSIAN_BLUR_H.to_string(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: h_params,
                extra_buffers: vec![kernel_bytes.clone()],
                reduction_buffers: vec![],
                convergence_check: None, loop_dispatch: None,
            },
            GpuShader {
                body: spatial::GAUSSIAN_BLUR_V.to_string(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: v_params,
                extra_buffers: vec![kernel_bytes],
                reduction_buffers: vec![],
                convergence_check: None, loop_dispatch: None,
            },
            GpuShader {
                body: spatial::TILT_SHIFT_BLEND.to_string(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: blend_params,
                extra_buffers: vec![vec![0u8; (n * 16) as usize]], // placeholder for original (snapshot)
                reduction_buffers: vec![],
                convergence_check: None, loop_dispatch: None,
            },
        ]
    }
}

// ── SmartSharpen GPU (bilateral blur + unsharp mask) ────────────────────────
// Uses bilateral shader for blur, then sharpen_apply for unsharp mask.

impl GpuFilter for SmartSharpen {
    fn shader_body(&self) -> &str { "" } // multi-pass only
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, _w: u32, _h: u32) -> Vec<u8> { Vec::new() }

    fn gpu_shaders(&self, w: u32, h: u32) -> Vec<GpuShader> {
        if self.amount.abs() < 1e-6 {
            return vec![self.gpu_shader(w, h)];
        }
        let r = self.radius;
        let sc2 = -0.5f32 / (self.threshold * self.threshold);
        let ss2 = -0.5f32 / (r as f32 * r as f32);
        let n = w * h;

        // Pass 1: bilateral blur
        let bilateral_params = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&w.to_le_bytes());
            buf.extend_from_slice(&h.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&sc2.to_le_bytes());
            buf.extend_from_slice(&ss2.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };
        // Pass 2: unsharp mask
        let sharpen_params = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&w.to_le_bytes());
            buf.extend_from_slice(&h.to_le_bytes());
            buf.extend_from_slice(&self.amount.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf
        };

        vec![
            GpuShader {
                body: spatial::BILATERAL.to_string(),
                entry_point: "main",
                workgroup_size: [16, 16, 1],
                params: bilateral_params,
                extra_buffers: vec![],
                reduction_buffers: vec![],
                convergence_check: None, loop_dispatch: None,
            },
            GpuShader {
                body: spatial::SHARPEN_APPLY.to_string(),
                entry_point: "main",
                workgroup_size: [256, 1, 1],
                params: sharpen_params,
                extra_buffers: vec![vec![0u8; (n * 16) as usize]], // placeholder for original
                reduction_buffers: vec![],
                convergence_check: None, loop_dispatch: None,
            },
        ]
    }
}

// All spatial filters are auto-registered via #[derive(V2Filter)] on their structs.

#[cfg(test)]
mod tests {
    use super::*;

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
        // Solid → blur = same → high_pass = 0 + 0.5 = 0.5
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

    use crate::ops::GpuFilter;

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
        // Params should be 32 bytes (8 × u32/f32)
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
        assert_eq!(bufs[0].len(), n * 8); // 2 × f32 per pixel
    }

    #[test]
    fn all_spatial_filters_have_gpu() {
        // Every spatial filter must implement GpuFilter — verify gpu_shaders() is non-empty
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
