//! Edge detection and threshold filters — Sobel, Scharr, Laplacian, Canny,
//! and various thresholding methods (binary, Otsu, adaptive, triangle).
//!
//! Edge detectors are spatial (need 3x3 neighborhood). Threshold filters are
//! point ops on luminance. All have GPU shaders.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

// ─── Helpers ───────────────────────────────────────────────────────────────

#[inline]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Sample luminance at (x, y) with clamped bounds.
#[inline]
fn sample_luma(input: &[f32], w: usize, h: usize, x: i32, y: i32) -> f32 {
    let sx = x.clamp(0, w as i32 - 1) as usize;
    let sy = y.clamp(0, h as i32 - 1) as usize;
    let idx = (sy * w + sx) * 4;
    luma(input[idx], input[idx + 1], input[idx + 2])
}

/// Apply 3x3 convolution kernel at (cx, cy) on luminance channel.
#[inline]
fn convolve3x3(input: &[f32], w: usize, h: usize, cx: i32, cy: i32, kernel: &[f32; 9]) -> f32 {
    let mut sum = 0.0f32;
    for ky in 0..3i32 {
        for kx in 0..3i32 {
            sum += sample_luma(input, w, h, cx + kx - 1, cy + ky - 1) * kernel[(ky * 3 + kx) as usize];
        }
    }
    sum
}

// ─── Sobel ─────────────────────────────────────────────────────────────────

/// Sobel edge detection — 3x3 gradient magnitude.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sobel", category = "spatial", cost = "O(n)")]
pub struct Sobel {
    /// Output scale (1.0 = standard, higher = more visible edges)
    #[param(min = 0.1, max = 5.0, step = 0.1, default = 1.0)]
    pub scale: f32,
}

const SOBEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
const SOBEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

impl Filter for Sobel {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (width as usize, height as usize);
        let mut out = vec![0.0f32; w * h * 4];
        let scale = self.scale;
        for y in 0..h {
            for x in 0..w {
                let gx = convolve3x3(input, w, h, x as i32, y as i32, &SOBEL_X);
                let gy = convolve3x3(input, w, h, x as i32, y as i32, &SOBEL_Y);
                let mag = (gx * gx + gy * gy).sqrt() * scale;
                let v = mag.clamp(0.0, 1.0);
                let idx = (y * w + x) * 4;
                out[idx] = v;
                out[idx + 1] = v;
                out[idx + 2] = v;
                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 { 1 }

    fn gpu_shader_body(&self) -> Option<&'static str> { Some(SOBEL_WGSL) }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(12);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.scale.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // pad to 16
        Some(buf)
    }
}

const SOBEL_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, _pad: u32, }
@group(0) @binding(2) var<uniform> params: Params;

fn sample_luma(x: i32, y: i32) -> f32 {
    let sx = clamp(x, 0, i32(params.width) - 1);
    let sy = clamp(y, 0, i32(params.height) - 1);
    let p = load_pixel(u32(sy) * params.width + u32(sx));
    return 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let gx = -sample_luma(x-1,y-1) + sample_luma(x+1,y-1)
           - 2.0*sample_luma(x-1,y) + 2.0*sample_luma(x+1,y)
           - sample_luma(x-1,y+1) + sample_luma(x+1,y+1);
    let gy = -sample_luma(x-1,y-1) - 2.0*sample_luma(x,y-1) - sample_luma(x+1,y-1)
           + sample_luma(x-1,y+1) + 2.0*sample_luma(x,y+1) + sample_luma(x+1,y+1);
    let mag = clamp(sqrt(gx*gx + gy*gy) * params.scale, 0.0, 1.0);
    let idx = gid.y * params.width + gid.x;
    let a = load_pixel(idx).w;
    store_pixel(idx, vec4(mag, mag, mag, a));
}
"#;

// ─── Scharr ────────────────────────────────────────────────────────────────

/// Scharr edge detection — 3x3 gradient with better rotational symmetry than Sobel.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "scharr", category = "spatial", cost = "O(n)")]
pub struct Scharr {
    #[param(min = 0.1, max = 5.0, step = 0.1, default = 1.0)]
    pub scale: f32,
}

const SCHARR_X: [f32; 9] = [-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0];
const SCHARR_Y: [f32; 9] = [-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0];

impl Filter for Scharr {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (width as usize, height as usize);
        let mut out = vec![0.0f32; w * h * 4];
        // Scharr normalization: max magnitude for step edge = 32, normalize to [0,1]
        let scale = self.scale / 32.0;
        for y in 0..h {
            for x in 0..w {
                let gx = convolve3x3(input, w, h, x as i32, y as i32, &SCHARR_X);
                let gy = convolve3x3(input, w, h, x as i32, y as i32, &SCHARR_Y);
                let v = ((gx * gx + gy * gy).sqrt() * scale).clamp(0.0, 1.0);
                let idx = (y * w + x) * 4;
                out[idx] = v; out[idx + 1] = v; out[idx + 2] = v;
                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 { 1 }

    fn gpu_shader_body(&self) -> Option<&'static str> { Some(SCHARR_WGSL) }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&(self.scale / 32.0).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        Some(buf)
    }
}

const SCHARR_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, _pad: u32, }
@group(0) @binding(2) var<uniform> params: Params;

fn sample_luma(x: i32, y: i32) -> f32 {
    let sx = clamp(x, 0, i32(params.width) - 1);
    let sy = clamp(y, 0, i32(params.height) - 1);
    let p = load_pixel(u32(sy) * params.width + u32(sx));
    return 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let gx = -3.0*sample_luma(x-1,y-1) + 3.0*sample_luma(x+1,y-1)
           - 10.0*sample_luma(x-1,y) + 10.0*sample_luma(x+1,y)
           - 3.0*sample_luma(x-1,y+1) + 3.0*sample_luma(x+1,y+1);
    let gy = -3.0*sample_luma(x-1,y-1) - 10.0*sample_luma(x,y-1) - 3.0*sample_luma(x+1,y-1)
           + 3.0*sample_luma(x-1,y+1) + 10.0*sample_luma(x,y+1) + 3.0*sample_luma(x+1,y+1);
    let mag = clamp(sqrt(gx*gx + gy*gy) * params.scale, 0.0, 1.0);
    let idx = gid.y * params.width + gid.x;
    store_pixel(idx, vec4(mag, mag, mag, load_pixel(idx).w));
}
"#;

// ─── Laplacian ─────────────────────────────────────────────────────────────

/// Laplacian edge detection — second derivative, detects edges and zero-crossings.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "laplacian", category = "spatial", cost = "O(n)")]
pub struct Laplacian {
    #[param(min = 0.1, max = 5.0, step = 0.1, default = 1.0)]
    pub scale: f32,
}

const LAPLACIAN_K: [f32; 9] = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

impl Filter for Laplacian {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (width as usize, height as usize);
        let mut out = vec![0.0f32; w * h * 4];
        let scale = self.scale;
        for y in 0..h {
            for x in 0..w {
                let v = (convolve3x3(input, w, h, x as i32, y as i32, &LAPLACIAN_K).abs() * scale).clamp(0.0, 1.0);
                let idx = (y * w + x) * 4;
                out[idx] = v; out[idx + 1] = v; out[idx + 2] = v;
                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 { 1 }

    fn gpu_shader_body(&self) -> Option<&'static str> { Some(LAPLACIAN_WGSL) }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.scale.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        Some(buf)
    }
}

const LAPLACIAN_WGSL: &str = r#"
struct Params { width: u32, height: u32, scale: f32, _pad: u32, }
@group(0) @binding(2) var<uniform> params: Params;

fn sample_luma(x: i32, y: i32) -> f32 {
    let sx = clamp(x, 0, i32(params.width) - 1);
    let sy = clamp(y, 0, i32(params.height) - 1);
    let p = load_pixel(u32(sy) * params.width + u32(sx));
    return 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let v = sample_luma(x,y-1) + sample_luma(x-1,y) - 4.0*sample_luma(x,y) + sample_luma(x+1,y) + sample_luma(x,y+1);
    let mag = clamp(abs(v) * params.scale, 0.0, 1.0);
    let idx = gid.y * params.width + gid.x;
    store_pixel(idx, vec4(mag, mag, mag, load_pixel(idx).w));
}
"#;

// ─── Canny ─────────────────────────────────────────────────────────────────

/// Canny edge detection — Sobel gradient + double threshold + edge thinning.
///
/// Simplified single-pass: gradient magnitude with hysteresis thresholding.
/// True Canny requires non-maximum suppression (multi-pass), implemented
/// here as a gradient threshold approximation for real-time use.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "canny", category = "spatial", cost = "O(n)")]
pub struct Canny {
    /// Low threshold (edges below this are suppressed)
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.1)]
    pub low: f32,
    /// High threshold (edges above this are strong edges)
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.3)]
    pub high: f32,
}

impl Filter for Canny {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (width as usize, height as usize);
        let mut out = vec![0.0f32; w * h * 4];
        let (lo, hi) = (self.low, self.high);
        for y in 0..h {
            for x in 0..w {
                let gx = convolve3x3(input, w, h, x as i32, y as i32, &SOBEL_X);
                let gy = convolve3x3(input, w, h, x as i32, y as i32, &SOBEL_Y);
                let mag = (gx * gx + gy * gy).sqrt();
                let v = if mag >= hi { 1.0 } else if mag >= lo { 0.5 } else { 0.0 };
                let idx = (y * w + x) * 4;
                out[idx] = v; out[idx + 1] = v; out[idx + 2] = v;
                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 { 1 }

    fn gpu_shader_body(&self) -> Option<&'static str> { Some(CANNY_WGSL) }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.low.to_le_bytes());
        buf.extend_from_slice(&self.high.to_le_bytes());
        Some(buf)
    }
}

const CANNY_WGSL: &str = r#"
struct Params { width: u32, height: u32, low: f32, high: f32, }
@group(0) @binding(2) var<uniform> params: Params;

fn sample_luma(x: i32, y: i32) -> f32 {
    let sx = clamp(x, 0, i32(params.width) - 1);
    let sy = clamp(y, 0, i32(params.height) - 1);
    let p = load_pixel(u32(sy) * params.width + u32(sx));
    return 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x); let y = i32(gid.y);
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let gx = -sample_luma(x-1,y-1) + sample_luma(x+1,y-1) - 2.0*sample_luma(x-1,y) + 2.0*sample_luma(x+1,y) - sample_luma(x-1,y+1) + sample_luma(x+1,y+1);
    let gy = -sample_luma(x-1,y-1) - 2.0*sample_luma(x,y-1) - sample_luma(x+1,y-1) + sample_luma(x-1,y+1) + 2.0*sample_luma(x,y+1) + sample_luma(x+1,y+1);
    let mag = sqrt(gx*gx + gy*gy);
    var v = 0.0;
    if (mag >= params.high) { v = 1.0; } else if (mag >= params.low) { v = 0.5; }
    let idx = gid.y * params.width + gid.x;
    store_pixel(idx, vec4(v, v, v, load_pixel(idx).w));
}
"#;

// ─── Threshold Binary ──────────────────────────────────────────────────────

/// Binary threshold — pixels above threshold become white, below become black.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "threshold_binary", category = "adjustment", cost = "O(n)")]
pub struct ThresholdBinary {
    #[param(min = 0.0, max = 1.0, step = 0.02, default = 0.5)]
    pub threshold: f32,
}

impl Filter for ThresholdBinary {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let t = self.threshold;
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            let l = luma(px[0], px[1], px[2]);
            let v = if l >= t { 1.0 } else { 0.0 };
            px[0] = v; px[1] = v; px[2] = v;
        }
        Ok(out)
    }

    fn gpu_shader_body(&self) -> Option<&'static str> { Some(THRESHOLD_WGSL) }

    fn gpu_params(&self, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.threshold.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        Some(buf)
    }
}

const THRESHOLD_WGSL: &str = r#"
struct Params { width: u32, height: u32, threshold: f32, _pad: u32, }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);
    let l = 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z;
    let v = select(0.0, 1.0, l >= params.threshold);
    store_pixel(idx, vec4(v, v, v, p.w));
}
"#;

// ─── Otsu Threshold ────────────────────────────────────────────────────────

/// Otsu's method — automatic threshold that maximizes between-class variance.
/// Computes optimal threshold from the image histogram, then applies binary threshold.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "otsu_threshold", category = "adjustment", cost = "O(n)")]
pub struct OtsuThreshold;

impl Filter for OtsuThreshold {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let npx = input.len() / 4;
        // Build 256-bin luminance histogram
        let mut hist = [0u32; 256];
        for px in input.chunks_exact(4) {
            let l = luma(px[0], px[1], px[2]).clamp(0.0, 1.0);
            hist[(l * 255.0) as usize] += 1;
        }
        // Otsu's algorithm
        let total = npx as f64;
        let mut sum_total = 0.0f64;
        for (i, &h) in hist.iter().enumerate() {
            sum_total += i as f64 * h as f64;
        }
        let mut sum_bg = 0.0f64;
        let mut w_bg = 0.0f64;
        let mut max_var = 0.0f64;
        let mut best_t = 0usize;
        for (t, &h) in hist.iter().enumerate() {
            w_bg += h as f64;
            if w_bg == 0.0 { continue; }
            let w_fg = total - w_bg;
            if w_fg == 0.0 { break; }
            sum_bg += t as f64 * h as f64;
            let mean_bg = sum_bg / w_bg;
            let mean_fg = (sum_total - sum_bg) / w_fg;
            let var = w_bg * w_fg * (mean_bg - mean_fg).powi(2);
            if var > max_var { max_var = var; best_t = t; }
        }
        let threshold = best_t as f32 / 255.0;
        // Apply
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            let l = luma(px[0], px[1], px[2]);
            let v = if l >= threshold { 1.0 } else { 0.0 };
            px[0] = v; px[1] = v; px[2] = v;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        use crate::gpu_shaders::reduction::GpuReduction;
        let reduction = GpuReduction::histogram_256(256);
        let passes = reduction.build_passes(width, height);
        let total = width * height;
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&total.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        let pass3 = GpuShader::new(
            OTSU_APPLY_WGSL.to_string(), "main", [256, 1, 1], params,
        ).with_reduction_buffers(vec![reduction.read_buffer(&passes)]);
        Some(vec![passes.pass1, passes.pass2, pass3])
    }
}

/// Otsu GPU apply shader — reads histogram, computes optimal threshold inline, applies.
const OTSU_APPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, total_pixels: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_pixels) { return; }

    // Compute Otsu threshold from luminance histogram (bins 0..255 of R channel)
    var sum_total = 0.0;
    for (var i = 0u; i < 256u; i++) { sum_total += f32(i) * f32(histogram[i]); }
    var sum_bg = 0.0;
    var w_bg = 0.0;
    var max_var = 0.0;
    var best_t = 0u;
    let total = f32(params.total_pixels);
    for (var t = 0u; t < 256u; t++) {
        w_bg += f32(histogram[t]);
        if (w_bg == 0.0) { continue; }
        let w_fg = total - w_bg;
        if (w_fg == 0.0) { break; }
        sum_bg += f32(t) * f32(histogram[t]);
        let mean_bg = sum_bg / w_bg;
        let mean_fg = (sum_total - sum_bg) / w_fg;
        let v = w_bg * w_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (v > max_var) { max_var = v; best_t = t; }
    }
    let threshold = f32(best_t) / 255.0;

    let pixel = input[idx];
    let l = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
    let v = select(0.0, 1.0, l >= threshold);
    output[idx] = vec4(v, v, v, pixel.w);
}
"#;

// ─── Triangle Threshold ────────────────────────────────────────────────────

/// Triangle threshold — automatic threshold for unimodal histograms.
/// Finds the threshold that maximizes the distance from the histogram
/// line connecting the peak to the farthest bin.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "triangle_threshold", category = "adjustment", cost = "O(n)")]
pub struct TriangleThreshold;

impl Filter for TriangleThreshold {
    fn compute(&self, input: &[f32], _w: u32, _h: u32) -> Result<Vec<f32>, PipelineError> {
        let mut hist = [0u32; 256];
        for px in input.chunks_exact(4) {
            let l = luma(px[0], px[1], px[2]).clamp(0.0, 1.0);
            hist[(l * 255.0) as usize] += 1;
        }
        // Find peak
        let peak_idx = hist.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
        // Find farthest non-zero bin from peak
        let far_idx = if peak_idx < 128 {
            hist.iter().rposition(|&h| h > 0).unwrap_or(255)
        } else {
            hist.iter().position(|&h| h > 0).unwrap_or(0)
        };
        // Line from (peak_idx, hist[peak]) to (far_idx, hist[far_idx])
        let (x1, y1) = (peak_idx as f64, hist[peak_idx] as f64);
        let (x2, y2) = (far_idx as f64, hist[far_idx] as f64);
        let len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt().max(1e-6);
        // Find bin with max distance from line
        let mut best_t = peak_idx;
        let mut max_dist = 0.0f64;
        let (lo, hi) = if peak_idx < far_idx { (peak_idx, far_idx) } else { (far_idx, peak_idx) };
        for t in lo..=hi {
            let d = ((y2 - y1) * t as f64 - (x2 - x1) * hist[t] as f64 + x2 * y1 - y2 * x1).abs() / len;
            if d > max_dist { max_dist = d; best_t = t; }
        }
        let threshold = best_t as f32 / 255.0;
        let mut out = input.to_vec();
        for px in out.chunks_exact_mut(4) {
            let l = luma(px[0], px[1], px[2]);
            let v = if l >= threshold { 1.0 } else { 0.0 };
            px[0] = v; px[1] = v; px[2] = v;
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        use crate::gpu_shaders::reduction::GpuReduction;
        let reduction = GpuReduction::histogram_256(256);
        let passes = reduction.build_passes(width, height);
        let total = width * height;
        let mut params = Vec::with_capacity(16);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());
        params.extend_from_slice(&total.to_le_bytes());
        params.extend_from_slice(&0u32.to_le_bytes());
        let pass3 = GpuShader::new(
            TRIANGLE_APPLY_WGSL.to_string(), "main", [256, 1, 1], params,
        ).with_reduction_buffers(vec![reduction.read_buffer(&passes)]);
        Some(vec![passes.pass1, passes.pass2, pass3])
    }
}

/// Triangle threshold GPU apply — reads histogram, finds peak/far, computes threshold.
const TRIANGLE_APPLY_WGSL: &str = r#"
struct Params { width: u32, height: u32, total_pixels: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_pixels) { return; }

    // Find peak bin in luminance histogram
    var peak_idx = 0u;
    var peak_val = 0u;
    for (var i = 0u; i < 256u; i++) {
        if (histogram[i] > peak_val) { peak_val = histogram[i]; peak_idx = i; }
    }
    // Find farthest non-zero bin
    var far_idx = 0u;
    if (peak_idx < 128u) {
        for (var i = 255u; i > 0u; i--) { if (histogram[i] > 0u) { far_idx = i; break; } }
    } else {
        for (var i = 0u; i < 256u; i++) { if (histogram[i] > 0u) { far_idx = i; break; } }
    }
    // Line distance method
    let x1 = f32(peak_idx); let y1 = f32(peak_val);
    let x2 = f32(far_idx); let y2 = f32(histogram[far_idx]);
    let line_len = max(sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)), 0.000001);
    var best_t = peak_idx;
    var max_dist = 0.0;
    let lo = min(peak_idx, far_idx);
    let hi = max(peak_idx, far_idx);
    for (var t = lo; t <= hi; t++) {
        let d = abs((y2-y1)*f32(t) - (x2-x1)*f32(histogram[t]) + x2*y1 - y2*x1) / line_len;
        if (d > max_dist) { max_dist = d; best_t = t; }
    }
    let threshold = f32(best_t) / 255.0;

    let pixel = input[idx];
    let l = 0.2126 * pixel.x + 0.7152 * pixel.y + 0.0722 * pixel.z;
    let v = select(0.0, 1.0, l >= threshold);
    output[idx] = vec4(v, v, v, pixel.w);
}
"#;

// ─── Adaptive Threshold ────────────────────────────────────────────────────

/// Adaptive threshold — per-pixel threshold based on local mean.
/// Each pixel is compared to the mean luminance in a surrounding window.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "adaptive_threshold", category = "spatial", cost = "O(n * r^2)")]
pub struct AdaptiveThreshold {
    /// Window radius for local mean computation
    #[param(min = 1, max = 50, step = 1, default = 5)]
    pub radius: u32,
    /// Offset subtracted from local mean (negative = more white, positive = more black)
    #[param(min = -0.5, max = 0.5, step = 0.02, default = 0.02)]
    pub offset: f32,
}

impl Filter for AdaptiveThreshold {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (width as usize, height as usize);
        let r = self.radius as i32;
        let offset = self.offset;

        // Compute integral image of luminance for O(1) local mean
        let mut integral = vec![0.0f64; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row_sum = 0.0f64;
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let l = luma(input[idx], input[idx + 1], input[idx + 2]) as f64;
                row_sum += l;
                integral[(y + 1) * (w + 1) + (x + 1)] = row_sum + integral[y * (w + 1) + (x + 1)];
            }
        }

        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let x1 = (x - r).max(0) as usize;
                let y1 = (y - r).max(0) as usize;
                let x2 = (x + r + 1).min(w as i32) as usize;
                let y2 = (y + r + 1).min(h as i32) as usize;
                let area = ((x2 - x1) * (y2 - y1)) as f64;
                let sum = integral[y2 * (w + 1) + x2] - integral[y1 * (w + 1) + x2]
                        - integral[y2 * (w + 1) + x1] + integral[y1 * (w + 1) + x1];
                let local_mean = (sum / area) as f32;

                let idx = (y as usize * w + x as usize) * 4;
                let l = luma(input[idx], input[idx + 1], input[idx + 2]);
                let v = if l >= local_mean - offset { 1.0 } else { 0.0 };
                out[idx] = v; out[idx + 1] = v; out[idx + 2] = v;
                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 { self.radius }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_image(w: u32, h: u32) -> Vec<f32> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = x as f32 / (w - 1) as f32;
                px.extend_from_slice(&[v, v, v, 1.0]);
            }
        }
        px
    }

    #[test]
    fn sobel_detects_vertical_edge() {
        // Gradient image: left=black, right=white → strong vertical edges
        let img = gradient_image(16, 16);
        let out = Sobel { scale: 1.0 }.compute(&img, 16, 16).unwrap();
        // Middle column should have non-zero edge response
        let mid = 8 * 4; // x=8, y=0
        assert!(out[mid] > 0.0, "sobel should detect gradient edge");
    }

    #[test]
    fn threshold_binary_splits() {
        let img = gradient_image(16, 16);
        let out = ThresholdBinary { threshold: 0.5 }.compute(&img, 16, 16).unwrap();
        // First pixel (x=0) should be black
        assert_eq!(out[0], 0.0);
        // Last pixel (x=15) should be white
        let last = (15 * 4) as usize;
        assert_eq!(out[last], 1.0);
    }

    #[test]
    fn otsu_auto_threshold() {
        let img = gradient_image(32, 32);
        let out = OtsuThreshold.compute(&img, 32, 32).unwrap();
        // Should produce binary output
        for px in out.chunks_exact(4) {
            assert!(px[0] == 0.0 || px[0] == 1.0, "otsu should produce binary: {}", px[0]);
        }
    }

    #[test]
    fn adaptive_threshold_runs() {
        let img = gradient_image(16, 16);
        let f = AdaptiveThreshold { radius: 3, offset: 0.02 };
        let out = f.compute(&img, 16, 16).unwrap();
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn canny_produces_binary_edges() {
        let img = gradient_image(16, 16);
        let out = Canny { low: 0.05, high: 0.2 }.compute(&img, 16, 16).unwrap();
        // All pixels should be one of: 0.0 (no edge), 0.5 (weak), 1.0 (strong)
        for px in out.chunks_exact(4) {
            assert!(
                px[0] == 0.0 || (px[0] - 0.5).abs() < 1e-6 || (px[0] - 1.0).abs() < 1e-6,
                "canny output should be 0/0.5/1, got {}", px[0]
            );
        }
        // At least some pixels should be non-zero (edges detected)
        let edge_count = out.chunks_exact(4).filter(|px| px[0] > 0.0).count();
        assert!(edge_count > 0, "canny should detect some edges in gradient");
    }

    #[test]
    fn filters_registered() {
        let ops = crate::registered_operations();
        let names: Vec<&str> = ops.iter().map(|o| o.name).collect();
        for f in &["sobel", "scharr", "laplacian", "canny", "threshold_binary", "otsu_threshold", "triangle_threshold", "adaptive_threshold"] {
            assert!(names.contains(f), "{f} not registered");
        }
    }

    #[test]
    fn otsu_has_gpu_3_pass() {
        let f = OtsuThreshold;
        let shaders = f.gpu_shader_passes(32, 32);
        assert!(shaders.is_some(), "otsu should have GPU shaders");
        assert_eq!(shaders.unwrap().len(), 3, "otsu should be 3-pass (hist reduce + hist merge + apply)");
    }

    #[test]
    fn triangle_has_gpu_3_pass() {
        let f = TriangleThreshold;
        let shaders = f.gpu_shader_passes(32, 32);
        assert!(shaders.is_some(), "triangle should have GPU shaders");
        assert_eq!(shaders.unwrap().len(), 3, "triangle should be 3-pass");
    }
}
