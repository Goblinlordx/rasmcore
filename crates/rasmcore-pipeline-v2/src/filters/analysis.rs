//! Analysis and transform filters — feature detection, seam carving,
//! perspective correction, and interpolation.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

fn gpu_params_wh(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf
}
fn gpu_push_f32(buf: &mut Vec<u8>, v: f32) { buf.extend_from_slice(&v.to_le_bytes()); }
fn gpu_push_u32(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_le_bytes()); }

fn sample_bilinear(input: &[f32], width: u32, height: u32, fx: f32, fy: f32) -> [f32; 4] {
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let dx = fx - ix as f32;
    let dy = fy - iy as f32;
    let x0 = ix.max(0).min(width as i32 - 1) as usize;
    let x1 = (ix + 1).max(0).min(width as i32 - 1) as usize;
    let y0 = iy.max(0).min(height as i32 - 1) as usize;
    let y1 = (iy + 1).max(0).min(height as i32 - 1) as usize;
    let w = width as usize;
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let p00 = input[(y0 * w + x0) * 4 + c];
        let p10 = input[(y0 * w + x1) * 4 + c];
        let p01 = input[(y1 * w + x0) * 4 + c];
        let p11 = input[(y1 * w + x1) * 4 + c];
        out[c] = (p00 + dx * (p10 - p00)) + dy * ((p01 + dx * (p11 - p01)) - (p00 + dx * (p10 - p00)));
    }
    out
}

const SAMPLE_BILINEAR_WGSL: &str = r#"
fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx)); let iy = i32(floor(fy));
  let dx = fx - f32(ix); let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.width) - 1);
  let y0 = clamp(iy, 0, i32(params.height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.height) - 1);
  let p00 = input[u32(x0) + u32(y0) * params.width];
  let p10 = input[u32(x1) + u32(y0) * params.width];
  let p01 = input[u32(x0) + u32(y1) * params.width];
  let p11 = input[u32(x1) + u32(y1) * params.width];
  return mix(mix(p00, p10, vec4<f32>(dx)), mix(p01, p11, vec4<f32>(dx)), vec4<f32>(dy));
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// Harris Corners — corner strength map
// ═══════════════════════════════════════════════════════════════════════════

/// Harris corner detection — outputs corner strength as grayscale.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "harris_corners", category = "analysis")]
pub struct HarrisCorners {
    #[param(min = 0.01, max = 0.3, step = 0.01, default = 0.04)]
    pub k: f32,
    #[param(min = 1, max = 5, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

const HARRIS_WGSL: &str = r#"
struct Params { width: u32, height: u32, k: f32, radius: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let r = i32(params.radius);
  var ixx: f32 = 0.0; var iyy: f32 = 0.0; var ixy: f32 = 0.0;
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let sx = clamp(x + dx, 0, w - 1); let sy = clamp(y + dy, 0, h - 1);
      let sxp = clamp(x + dx + 1, 0, w - 1); let syp = clamp(y + dy + 1, 0, h - 1);
      let sxm = clamp(x + dx - 1, 0, w - 1); let sym = clamp(y + dy - 1, 0, h - 1);
      let c = input[u32(sx) + u32(sy) * params.width];
      let cx1 = input[u32(sxp) + u32(sy) * params.width];
      let cx0 = input[u32(sxm) + u32(sy) * params.width];
      let cy1 = input[u32(sx) + u32(syp) * params.width];
      let cy0 = input[u32(sx) + u32(sym) * params.width];
      let luma = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
      let gx = (cx1.r * 0.2126 + cx1.g * 0.7152 + cx1.b * 0.0722) - (cx0.r * 0.2126 + cx0.g * 0.7152 + cx0.b * 0.0722);
      let gy = (cy1.r * 0.2126 + cy1.g * 0.7152 + cy1.b * 0.0722) - (cy0.r * 0.2126 + cy0.g * 0.7152 + cy0.b * 0.0722);
      ixx += gx * gx; iyy += gy * gy; ixy += gx * gy;
    }
  }
  let det = ixx * iyy - ixy * ixy;
  let trace = ixx + iyy;
  let response = det - params.k * trace * trace;
  let v = clamp(response * 1000.0, 0.0, 1.0);
  let orig = input[u32(x) + u32(y) * params.width];
  output[u32(x) + u32(y) * params.width] = vec4<f32>(v, v, v, orig.w);
}
"#;

impl Filter for HarrisCorners {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        let w = width as i32; let h = height as i32; let r = self.radius as i32;
        for y in 0..h {
            for x in 0..w {
                let mut ixx = 0.0f32; let mut iyy = 0.0f32; let mut ixy = 0.0f32;
                for dy in -r..=r { for dx in -r..=r {
                    let sx = (x+dx).max(0).min(w-1) as usize;
                    let sy = (y+dy).max(0).min(h-1) as usize;
                    let sxp = (x+dx+1).max(0).min(w-1) as usize;
                    let sxm = (x+dx-1).max(0).min(w-1) as usize;
                    let syp = (y+dy+1).max(0).min(h-1) as usize;
                    let sym = (y+dy-1).max(0).min(h-1) as usize;
                    let wi = width as usize;
                    let lx1 = input[(sy*wi+sxp)*4]*0.2126 + input[(sy*wi+sxp)*4+1]*0.7152 + input[(sy*wi+sxp)*4+2]*0.0722;
                    let lx0 = input[(sy*wi+sxm)*4]*0.2126 + input[(sy*wi+sxm)*4+1]*0.7152 + input[(sy*wi+sxm)*4+2]*0.0722;
                    let ly1 = input[(syp*wi+sx)*4]*0.2126 + input[(syp*wi+sx)*4+1]*0.7152 + input[(syp*wi+sx)*4+2]*0.0722;
                    let ly0 = input[(sym*wi+sx)*4]*0.2126 + input[(sym*wi+sx)*4+1]*0.7152 + input[(sym*wi+sx)*4+2]*0.0722;
                    let gx = lx1 - lx0; let gy = ly1 - ly0;
                    ixx += gx*gx; iyy += gy*gy; ixy += gx*gy;
                }}
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let response = det - self.k * trace * trace;
                let v = (response * 1000.0).max(0.0).min(1.0);
                let i = ((y * w + x) * 4) as usize;
                out[i] = v; out[i+1] = v; out[i+2] = v;
                out[i+3] = input[i+3];
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.k); gpu_push_u32(&mut p, self.radius);
        Some(vec![GpuShader::new(HARRIS_WGSL.to_string(), "main", [16, 16, 1], p)])
    }

    fn tile_overlap(&self) -> u32 { self.radius + 1 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Perspective Warp
// ═══════════════════════════════════════════════════════════════════════════

/// Perspective warp — apply a 3x3 homography matrix.
/// Parameters are the 8 independent elements (h33 = 1.0).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "perspective_warp", category = "transform")]
pub struct PerspectiveWarp {
    #[param(min = -2.0, max = 2.0, step = 0.001, default = 1.0)] pub h11: f32,
    #[param(min = -2.0, max = 2.0, step = 0.001, default = 0.0)] pub h12: f32,
    #[param(min = -1000.0, max = 1000.0, step = 1.0, default = 0.0)] pub h13: f32,
    #[param(min = -2.0, max = 2.0, step = 0.001, default = 0.0)] pub h21: f32,
    #[param(min = -2.0, max = 2.0, step = 0.001, default = 1.0)] pub h22: f32,
    #[param(min = -1000.0, max = 1000.0, step = 1.0, default = 0.0)] pub h23: f32,
    #[param(min = -0.01, max = 0.01, step = 0.0001, default = 0.0)] pub h31: f32,
    #[param(min = -0.01, max = 0.01, step = 0.0001, default = 0.0)] pub h32: f32,
}

const PERSPECTIVE_WARP_WGSL: &str = r#"
struct Params { width: u32, height: u32, h11: f32, h12: f32, h13: f32, h21: f32, h22: f32, h23: f32, h31: f32, h32: f32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width); let y = f32(idx / params.width);
  let w = params.h31 * x + params.h32 * y + 1.0;
  if (abs(w) < 0.0001) { output[idx] = vec4<f32>(0.0, 0.0, 0.0, 1.0); return; }
  let sx = (params.h11 * x + params.h12 * y + params.h13) / w;
  let sy = (params.h21 * x + params.h22 * y + params.h23) / w;
  output[idx] = sample_bilinear_f32(sx, sy);
}
"#;

impl Filter for PerspectiveWarp {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = vec![0.0f32; input.len()];
        for y in 0..height {
            for x in 0..width {
                let xf = x as f32; let yf = y as f32;
                let w = self.h31 * xf + self.h32 * yf + 1.0;
                let (sx, sy) = if w.abs() > 0.0001 {
                    ((self.h11 * xf + self.h12 * yf + self.h13) / w,
                     (self.h21 * xf + self.h22 * yf + self.h23) / w)
                } else { (0.0, 0.0) };
                let px = sample_bilinear(input, width, height, sx, sy);
                let i = ((y * width + x) * 4) as usize;
                out[i..i+4].copy_from_slice(&px);
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let shader = format!("{SAMPLE_BILINEAR_WGSL}\n{PERSPECTIVE_WARP_WGSL}");
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.h11); gpu_push_f32(&mut p, self.h12); gpu_push_f32(&mut p, self.h13);
        gpu_push_f32(&mut p, self.h21); gpu_push_f32(&mut p, self.h22); gpu_push_f32(&mut p, self.h23);
        gpu_push_f32(&mut p, self.h31); gpu_push_f32(&mut p, self.h32);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(shader, "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Perspective Correct — keystone correction (simplified)
// ═══════════════════════════════════════════════════════════════════════════

/// Keystone correction — correct converging vertical/horizontal lines.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "perspective_correct", category = "transform")]
pub struct PerspectiveCorrect {
    #[param(min = -0.005, max = 0.005, step = 0.0001, default = 0.0)] pub vertical: f32,
    #[param(min = -0.005, max = 0.005, step = 0.0001, default = 0.0)] pub horizontal: f32,
}

impl Filter for PerspectiveCorrect {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        // Delegate to PerspectiveWarp with appropriate homography
        let warp = PerspectiveWarp {
            h11: 1.0, h12: 0.0, h13: 0.0,
            h21: 0.0, h22: 1.0, h23: 0.0,
            h31: self.horizontal, h32: self.vertical,
        };
        warp.compute(input, width, height)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let warp = PerspectiveWarp {
            h11: 1.0, h12: 0.0, h13: 0.0,
            h21: 0.0, h22: 1.0, h23: 0.0,
            h31: self.horizontal, h32: self.vertical,
        };
        warp.gpu_shader_passes(width, height)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sparse Color — Shepard's weighted interpolation
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse color interpolation — blend image toward sparse color points
/// using inverse-distance weighting (Shepard's method).
/// Uses 4 fixed color control points for simplicity.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "sparse_color", category = "effect")]
pub struct SparseColor {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub x1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub y1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub r1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub g1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub b1: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub x2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub y2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub r2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.0)] pub g2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 1.0)] pub b2: f32,
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)] pub strength: f32,
}

const SPARSE_COLOR_WGSL: &str = r#"
struct Params { width: u32, height: u32, x1: f32, y1: f32, r1: f32, g1: f32, b1: f32, x2: f32, y2: f32, r2: f32, g2: f32, b2: f32, strength: f32, _p1: u32, _p2: u32, _p3: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let x = f32(idx % params.width) / f32(params.width);
  let y = f32(idx / params.width) / f32(params.height);
  let d1 = max(length(vec2<f32>(x - params.x1, y - params.y1)), 0.001);
  let d2 = max(length(vec2<f32>(x - params.x2, y - params.y2)), 0.001);
  let w1 = 1.0 / (d1 * d1); let w2 = 1.0 / (d2 * d2);
  let total = w1 + w2;
  let interp = (vec3<f32>(params.r1, params.g1, params.b1) * w1 + vec3<f32>(params.r2, params.g2, params.b2) * w2) / total;
  let px = input[idx];
  output[idx] = vec4<f32>(mix(px.rgb, interp, vec3<f32>(params.strength)), px.w);
}
"#;

impl Filter for SparseColor {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut out = input.to_vec();
        for y in 0..height {
            for x in 0..width {
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                let d1 = ((nx-self.x1).powi(2) + (ny-self.y1).powi(2)).sqrt().max(0.001);
                let d2 = ((nx-self.x2).powi(2) + (ny-self.y2).powi(2)).sqrt().max(0.001);
                let w1 = 1.0/(d1*d1); let w2 = 1.0/(d2*d2);
                let total = w1 + w2;
                let i = ((y * width + x) * 4) as usize;
                for c in 0..3 {
                    let colors = [
                        [self.r1, self.g1, self.b1],
                        [self.r2, self.g2, self.b2],
                    ];
                    let interp = (colors[0][c] * w1 + colors[1][c] * w2) / total;
                    out[i+c] = out[i+c] + self.strength * (interp - out[i+c]);
                }
            }
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, _w: u32, _h: u32) -> Option<Vec<GpuShader>> {
        let mut p = gpu_params_wh(_w, _h);
        gpu_push_f32(&mut p, self.x1); gpu_push_f32(&mut p, self.y1);
        gpu_push_f32(&mut p, self.r1); gpu_push_f32(&mut p, self.g1); gpu_push_f32(&mut p, self.b1);
        gpu_push_f32(&mut p, self.x2); gpu_push_f32(&mut p, self.y2);
        gpu_push_f32(&mut p, self.r2); gpu_push_f32(&mut p, self.g2); gpu_push_f32(&mut p, self.b2);
        gpu_push_f32(&mut p, self.strength);
        gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0); gpu_push_u32(&mut p, 0);
        Some(vec![GpuShader::new(SPARSE_COLOR_WGSL.to_string(), "main", [256, 1, 1], p)])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Seam Carve Width — content-aware width reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Content-aware width reduction via seam carving.
/// Removes low-energy vertical seams.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "seam_carve_width", category = "transform")]
pub struct SeamCarveWidth {
    #[param(min = 1, max = 500, step = 1, default = 50)]
    pub seams: u32,
}

impl Filter for SeamCarveWidth {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let mut buf = input.to_vec();
        let mut w = width as usize;
        let h = height as usize;
        for _ in 0..self.seams.min(w as u32 - 1) {
            // Compute energy (gradient magnitude)
            let mut energy = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let xp = (x + 1).min(w - 1); let xm = x.saturating_sub(1);
                    let yp = (y + 1).min(h - 1); let ym = y.saturating_sub(1);
                    let ix1 = (y * w + xp) * 4; let ix0 = (y * w + xm) * 4;
                    let iy1 = (yp * w + x) * 4; let iy0 = (ym * w + x) * 4;
                    let gx = ((buf[ix1] - buf[ix0]).powi(2) + (buf[ix1+1] - buf[ix0+1]).powi(2) + (buf[ix1+2] - buf[ix0+2]).powi(2)).sqrt();
                    let gy = ((buf[iy1] - buf[iy0]).powi(2) + (buf[iy1+1] - buf[iy0+1]).powi(2) + (buf[iy1+2] - buf[iy0+2]).powi(2)).sqrt();
                    energy[y * w + x] = gx + gy;
                }
            }
            // DP: find minimum energy vertical seam
            let mut dp = energy.clone();
            for y in 1..h {
                for x in 0..w {
                    let up = dp[(y-1)*w + x];
                    let ul = if x > 0 { dp[(y-1)*w + x - 1] } else { f32::MAX };
                    let ur = if x < w-1 { dp[(y-1)*w + x + 1] } else { f32::MAX };
                    dp[y*w + x] += up.min(ul).min(ur);
                }
            }
            // Backtrack
            let mut seam = vec![0usize; h];
            seam[h-1] = (0..w).min_by(|&a, &b| dp[(h-1)*w+a].partial_cmp(&dp[(h-1)*w+b]).unwrap()).unwrap();
            for y in (0..h-1).rev() {
                let x = seam[y+1];
                let mut best = x;
                if x > 0 && dp[y*w+x-1] < dp[y*w+best] { best = x-1; }
                if x < w-1 && dp[y*w+x+1] < dp[y*w+best] { best = x+1; }
                seam[y] = best;
            }
            // Remove seam
            let mut new_buf = Vec::with_capacity((w-1) * h * 4);
            for y in 0..h {
                for x in 0..w {
                    if x != seam[y] {
                        let i = (y * w + x) * 4;
                        new_buf.extend_from_slice(&buf[i..i+4]);
                    }
                }
            }
            buf = new_buf;
            w -= 1;
        }
        // Pad back to original width with black (output must be same size)
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let si = (y * w + x) * 4;
                let di = (y * width as usize + x) * 4;
                out[di..di+4].copy_from_slice(&buf[si..si+4]);
            }
        }
        Ok(out)
    }

    // Seam carving is inherently sequential (DP row dependency).
    // GPU energy map computation is possible but the DP backtrack is serial.
    // The energy computation could be a GPU pre-pass, but for now CPU-only.
    // TODO: GPU energy map + CPU DP backtrack hybrid.
}

/// Content-aware height reduction via seam carving.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "seam_carve_height", category = "transform")]
pub struct SeamCarveHeight {
    #[param(min = 1, max = 500, step = 1, default = 50)]
    pub seams: u32,
}

impl Filter for SeamCarveHeight {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        // Transpose → seam_carve_width → transpose back
        let mut transposed = vec![0.0f32; input.len()];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let si = (y * width as usize + x) * 4;
                let di = (x * height as usize + y) * 4;
                transposed[di..di+4].copy_from_slice(&input[si..si+4]);
            }
        }
        let scw = SeamCarveWidth { seams: self.seams };
        let carved = scw.compute(&transposed, height, width)?;
        // Transpose back
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..width as usize {
            for x in 0..height as usize {
                let si = (y * height as usize + x) * 4;
                let di = (x * width as usize + y) * 4;
                out[di..di+4].copy_from_slice(&carved[si..si+4]);
            }
        }
        Ok(out)
    }
}

/// Smart crop — crop to most salient region based on edge energy.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "smart_crop", category = "transform")]
pub struct SmartCrop {
    #[param(min = 0.1, max = 1.0, step = 0.01, default = 0.75)]
    pub ratio: f32,
}

impl Filter for SmartCrop {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize; let h = height as usize;
        let cw = (w as f32 * self.ratio) as usize;
        let ch = (h as f32 * self.ratio) as usize;
        if cw >= w || ch >= h { return Ok(input.to_vec()); }
        // Find region with maximum gradient energy
        let mut best_x = 0; let mut best_y = 0; let mut best_energy = f32::NEG_INFINITY;
        let step = 4.max(cw / 10);
        for sy in (0..=h-ch).step_by(step) {
            for sx in (0..=w-cw).step_by(step) {
                let mut energy = 0.0f32;
                for y in (sy..sy+ch).step_by(4) {
                    for x in (sx..sx+cw).step_by(4) {
                        let i = (y * w + x) * 4;
                        let xp = (x+1).min(w-1) * 4 + y * w * 4;
                        let gx = (input[i] - input.get(xp).copied().unwrap_or(0.0)).abs();
                        energy += gx;
                    }
                }
                if energy > best_energy { best_energy = energy; best_x = sx; best_y = sy; }
            }
        }
        // Extract crop, pad to original size
        let mut out = vec![0.0f32; (width * height * 4) as usize];
        for y in 0..ch {
            for x in 0..cw {
                let si = ((best_y + y) * w + best_x + x) * 4;
                let di = (y * w + x) * 4;
                out[di..di+4].copy_from_slice(&input[si..si+4]);
            }
        }
        Ok(out)
    }
}

/// Hough line detection — output line visualization.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "hough_lines", category = "analysis")]
pub struct HoughLines {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub threshold: f32,
}

impl Filter for HoughLines {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        // Simplified: output edge-detected image (Hough accumulator is complex)
        // Full Hough would need accumulator array + non-max suppression
        let mut out = vec![0.0f32; input.len()];
        let w = width as i32; let h = height as i32;
        for y in 1..h-1 {
            for x in 1..w-1 {
                let i = (y * w + x) as usize * 4;
                let lx = input[((y*w+x+1)*4) as usize] * 0.2126 + input[((y*w+x+1)*4+1) as usize] * 0.7152 + input[((y*w+x+1)*4+2) as usize] * 0.0722;
                let lm = input[((y*w+x-1)*4) as usize] * 0.2126 + input[((y*w+x-1)*4+1) as usize] * 0.7152 + input[((y*w+x-1)*4+2) as usize] * 0.0722;
                let edge = (lx - lm).abs();
                let v = if edge > self.threshold { 1.0 } else { 0.0 };
                out[i] = v; out[i+1] = v; out[i+2] = v; out[i+3] = input[i+3];
            }
        }
        Ok(out)
    }
}

/// Connected components labeling — assigns unique colors to connected regions.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "connected_components", category = "analysis")]
pub struct ConnectedComponents {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub threshold: f32,
}

impl Filter for ConnectedComponents {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize; let h = height as usize;
        let mut labels = vec![0u32; w * h];
        let mut next_label = 1u32;
        // Simple two-pass labeling
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let luma = input[i] * 0.2126 + input[i+1] * 0.7152 + input[i+2] * 0.0722;
                if luma <= self.threshold { continue; }
                let left = if x > 0 { labels[y * w + x - 1] } else { 0 };
                let above = if y > 0 { labels[(y-1) * w + x] } else { 0 };
                if left > 0 { labels[y * w + x] = left; }
                else if above > 0 { labels[y * w + x] = above; }
                else { labels[y * w + x] = next_label; next_label += 1; }
            }
        }
        // Colorize labels
        let mut out = vec![0.0f32; input.len()];
        for y in 0..h {
            for x in 0..w {
                let label = labels[y * w + x];
                let i = (y * w + x) * 4;
                if label > 0 {
                    let hue = (label as f32 * 137.508) % 360.0; // golden angle spacing
                    let (r, g, b) = hsl_to_rgb(hue, 0.8, 0.5);
                    out[i] = r; out[i+1] = g; out[i+2] = b;
                }
                out[i+3] = input[i+3];
            }
        }
        Ok(out)
    }
}

/// Template match — cross-correlation with a self-derived template.
/// Uses center region as template, outputs correlation strength map.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "template_match", category = "analysis")]
pub struct TemplateMatch {
    #[param(min = 4, max = 64, step = 2, default = 16, hint = "rc.pixels")]
    pub template_size: u32,
}

impl Filter for TemplateMatch {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize; let h = height as usize;
        let ts = self.template_size as usize;
        let tx = w / 2 - ts / 2; let ty = h / 2 - ts / 2;
        // Extract template from center
        let mut tmpl = Vec::with_capacity(ts * ts);
        for dy in 0..ts {
            for dx in 0..ts {
                let i = ((ty + dy) * w + tx + dx) * 4;
                tmpl.push(input[i] * 0.2126 + input[i+1] * 0.7152 + input[i+2] * 0.0722);
            }
        }
        // Normalized cross-correlation
        let mut out = vec![0.0f32; input.len()];
        for y in 0..h.saturating_sub(ts) {
            for x in 0..w.saturating_sub(ts) {
                let mut sum = 0.0f32;
                for dy in 0..ts {
                    for dx in 0..ts {
                        let i = ((y + dy) * w + x + dx) * 4;
                        let luma = input[i] * 0.2126 + input[i+1] * 0.7152 + input[i+2] * 0.0722;
                        sum += luma * tmpl[dy * ts + dx];
                    }
                }
                let v = (sum / (ts * ts) as f32).min(1.0);
                let oi = (y * w + x) * 4;
                out[oi] = v; out[oi+1] = v; out[oi+2] = v; out[oi+3] = 1.0;
            }
        }
        Ok(out)
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x),
        3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_analysis_transform_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["harris_corners", "perspective_warp", "perspective_correct",
                       "sparse_color", "seam_carve_width", "seam_carve_height",
                       "smart_crop", "hough_lines", "connected_components", "template_match"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn perspective_identity() {
        let input = vec![0.5, 0.3, 0.1, 1.0, 0.8, 0.6, 0.4, 1.0];
        let f = PerspectiveWarp {
            h11: 1.0, h12: 0.0, h13: 0.0,
            h21: 0.0, h22: 1.0, h23: 0.0,
            h31: 0.0, h32: 0.0,
        };
        let out = f.compute(&input, 2, 1).unwrap();
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.02, "expected {a}, got {b}");
        }
    }
}
