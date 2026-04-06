//! Morphology filters — min/max kernel operations on f32 pixel data.
//!
//! Dilate (max), erode (min), and compound operations (open, close,
//! gradient, tophat, blackhat, skeletonize). GPU shaders use per-pixel
//! neighborhood min/max scans.

use crate::node::{GpuShader, PipelineError};
use crate::ops::Filter;

// ─── GPU helpers ───────────────────────────────────────────────────────────

fn gpu_params_wh(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf
}

fn gpu_params_push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ─── Shared WGSL ───────────────────────────────────────────────────────────

const DILATE_WGSL: &str = r#"
struct Params { width: u32, height: u32, radius: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let r = i32(params.radius);
  var mx = vec4<f32>(-1e30);
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let sx = clamp(x + dx, 0, w - 1);
      let sy = clamp(y + dy, 0, h - 1);
      mx = max(mx, input[u32(sx) + u32(sy) * params.width]);
    }
  }
  let orig = input[u32(x) + u32(y) * params.width];
  mx.w = orig.w; // preserve alpha
  output[u32(x) + u32(y) * params.width] = mx;
}
"#;

const ERODE_WGSL: &str = r#"
struct Params { width: u32, height: u32, radius: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let r = i32(params.radius);
  var mn = vec4<f32>(1e30);
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let sx = clamp(x + dx, 0, w - 1);
      let sy = clamp(y + dy, 0, h - 1);
      mn = min(mn, input[u32(sx) + u32(sy) * params.width]);
    }
  }
  let orig = input[u32(x) + u32(y) * params.width];
  mn.w = orig.w;
  output[u32(x) + u32(y) * params.width] = mn;
}
"#;

// ─── CPU helpers ───────────────────────────────────────────────────────────

fn dilate_cpu(input: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    let r = radius as i32;
    let w = width as i32;
    let h = height as i32;
    for y in 0..h {
        for x in 0..w {
            let mut mx = [f32::NEG_INFINITY; 3];
            for dy in -r..=r {
                for dx in -r..=r {
                    let sx = (x + dx).max(0).min(w - 1) as usize;
                    let sy = (y + dy).max(0).min(h - 1) as usize;
                    let si = (sy * width as usize + sx) * 4;
                    mx[0] = mx[0].max(input[si]);
                    mx[1] = mx[1].max(input[si + 1]);
                    mx[2] = mx[2].max(input[si + 2]);
                }
            }
            let i = (y as usize * width as usize + x as usize) * 4;
            out[i] = mx[0];
            out[i + 1] = mx[1];
            out[i + 2] = mx[2];
            out[i + 3] = input[i + 3]; // alpha
        }
    }
    out
}

fn erode_cpu(input: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    let r = radius as i32;
    let w = width as i32;
    let h = height as i32;
    for y in 0..h {
        for x in 0..w {
            let mut mn = [f32::INFINITY; 3];
            for dy in -r..=r {
                for dx in -r..=r {
                    let sx = (x + dx).max(0).min(w - 1) as usize;
                    let sy = (y + dy).max(0).min(h - 1) as usize;
                    let si = (sy * width as usize + sx) * 4;
                    mn[0] = mn[0].min(input[si]);
                    mn[1] = mn[1].min(input[si + 1]);
                    mn[2] = mn[2].min(input[si + 2]);
                }
            }
            let i = (y as usize * width as usize + x as usize) * 4;
            out[i] = mn[0];
            out[i + 1] = mn[1];
            out[i + 2] = mn[2];
            out[i + 3] = input[i + 3];
        }
    }
    out
}

/// Copy input to a reduction buffer (snapshot) and pass through to output.
const SNAPSHOT_WGSL: &str = r#"
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> snapshot: array<vec4<f32>>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];
  snapshot[idx] = px;
  output[idx] = px;
}
"#;

/// Subtract: output = clamp(snapshot − input, 0, 1) per RGB channel.
const SUB_SNAP_MINUS_CURRENT_WGSL: &str = r#"
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> snapshot: array<vec4<f32>>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let a = snapshot[idx]; let b = input[idx];
  output[idx] = vec4<f32>(max(a.r - b.r, 0.0), max(a.g - b.g, 0.0), max(a.b - b.b, 0.0), a.w);
}
"#;

/// Subtract: output = clamp(current − snapshot, 0, 1) per RGB channel.
const SUB_CURRENT_MINUS_SNAP_WGSL: &str = r#"
struct Params { width: u32, height: u32, _p1: u32, _p2: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> snapshot: array<vec4<f32>>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let a = input[idx]; let b = snapshot[idx];
  output[idx] = vec4<f32>(max(a.r - b.r, 0.0), max(a.g - b.g, 0.0), max(a.b - b.b, 0.0), a.w);
}
"#;

/// Erode reading from a snapshot reduction buffer (binding 3) instead of input.
const ERODE_FROM_SNAP_WGSL: &str = r#"
struct Params { width: u32, height: u32, radius: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> snapshot: array<vec4<f32>>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  let w = i32(params.width); let h = i32(params.height);
  if (x >= w || y >= h) { return; }
  let r = i32(params.radius);
  var mn = vec4<f32>(1e30);
  for (var dy = -r; dy <= r; dy = dy + 1) {
    for (var dx = -r; dx <= r; dx = dx + 1) {
      let sx = clamp(x + dx, 0, w - 1);
      let sy = clamp(y + dy, 0, h - 1);
      mn = min(mn, snapshot[u32(sx) + u32(sy) * params.width]);
    }
  }
  let orig = snapshot[u32(x) + u32(y) * params.width];
  mn.w = orig.w;
  output[u32(x) + u32(y) * params.width] = mn;
}
"#;

use crate::node::ReductionBuffer;

fn make_snapshot_shader(width: u32, height: u32, buf_id: u32) -> GpuShader {
    let mut params = gpu_params_wh(width, height);
    gpu_params_push_u32(&mut params, 0);
    gpu_params_push_u32(&mut params, 0);
    let buf_size = (width as usize) * (height as usize) * 16; // vec4<f32> per pixel
    GpuShader {
        body: SNAPSHOT_WGSL.to_string(),
        entry_point: "main",
        workgroup_size: [16, 16, 1],
        params,
        extra_buffers: vec![],
        reduction_buffers: vec![ReductionBuffer {
            id: buf_id,
            initial_data: vec![0u8; buf_size],
            read_write: true,
        }],
    }
}

fn make_sub_shader(wgsl: &str, width: u32, height: u32, buf_id: u32) -> GpuShader {
    let mut params = gpu_params_wh(width, height);
    gpu_params_push_u32(&mut params, 0);
    gpu_params_push_u32(&mut params, 0);
    GpuShader {
        body: wgsl.to_string(),
        entry_point: "main",
        workgroup_size: [16, 16, 1],
        params,
        extra_buffers: vec![],
        reduction_buffers: vec![ReductionBuffer {
            id: buf_id,
            initial_data: vec![], // reuse existing allocation
            read_write: false,    // read-only on this pass
        }],
    }
}

/// Erode reading from a snapshot buffer instead of ping-pong input.
fn make_erode_from_snap_shader(width: u32, height: u32, radius: u32, snap_id: u32) -> GpuShader {
    let mut params = gpu_params_wh(width, height);
    gpu_params_push_u32(&mut params, radius);
    gpu_params_push_u32(&mut params, 0);
    GpuShader {
        body: ERODE_FROM_SNAP_WGSL.to_string(),
        entry_point: "main",
        workgroup_size: [16, 16, 1],
        params,
        extra_buffers: vec![],
        reduction_buffers: vec![ReductionBuffer {
            id: snap_id,
            initial_data: vec![],
            read_write: false,
        }],
    }
}

fn make_dilate_shader(width: u32, height: u32, radius: u32) -> GpuShader {
    let mut params = gpu_params_wh(width, height);
    gpu_params_push_u32(&mut params, radius);
    gpu_params_push_u32(&mut params, 0);
    GpuShader::new(DILATE_WGSL.to_string(), "main", [16, 16, 1], params)
}

fn make_erode_shader(width: u32, height: u32, radius: u32) -> GpuShader {
    let mut params = gpu_params_wh(width, height);
    gpu_params_push_u32(&mut params, radius);
    gpu_params_push_u32(&mut params, 0);
    GpuShader::new(ERODE_WGSL.to_string(), "main", [16, 16, 1], params)
}

// ═══════════════════════════════════════════════════════════════════════════
// Dilate
// ═══════════════════════════════════════════════════════════════════════════

/// Dilate — maximum filter (expands bright regions).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "dilate", category = "morphology")]
pub struct Dilate {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for Dilate {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(dilate_cpu(input, width, height, self.radius))
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        Some(vec![make_dilate_shader(width, height, self.radius)])
    }

    fn tile_overlap(&self) -> u32 { self.radius }
}

// ═══════════════════════════════════════════════════════════════════════════
// Erode
// ═══════════════════════════════════════════════════════════════════════════

/// Erode — minimum filter (expands dark regions).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "erode", category = "morphology")]
pub struct Erode {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for Erode {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        Ok(erode_cpu(input, width, height, self.radius))
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        Some(vec![make_erode_shader(width, height, self.radius)])
    }

    fn tile_overlap(&self) -> u32 { self.radius }
}

// ═══════════════════════════════════════════════════════════════════════════
// Morph Open (erode → dilate)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological opening — erode then dilate (removes small bright spots).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_open", category = "morphology")]
pub struct MorphOpen {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphOpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let eroded = erode_cpu(input, width, height, self.radius);
        Ok(dilate_cpu(&eroded, width, height, self.radius))
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        Some(vec![
            make_erode_shader(width, height, self.radius),
            make_dilate_shader(width, height, self.radius),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius * 2 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Morph Close (dilate → erode)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological closing — dilate then erode (fills small dark holes).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_close", category = "morphology")]
pub struct MorphClose {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphClose {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let dilated = dilate_cpu(input, width, height, self.radius);
        Ok(erode_cpu(&dilated, width, height, self.radius))
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        Some(vec![
            make_dilate_shader(width, height, self.radius),
            make_erode_shader(width, height, self.radius),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius * 2 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Morph Gradient (dilate − erode)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological gradient — dilate minus erode (edge detection).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_gradient", category = "morphology")]
pub struct MorphGradient {
    #[param(min = 1, max = 20, step = 1, default = 1, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphGradient {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let dilated = dilate_cpu(input, width, height, self.radius);
        let eroded = erode_cpu(input, width, height, self.radius);
        let mut out = vec![0.0f32; input.len()];
        for i in (0..input.len()).step_by(4) {
            out[i] = (dilated[i] - eroded[i]).abs();
            out[i + 1] = (dilated[i + 1] - eroded[i + 1]).abs();
            out[i + 2] = (dilated[i + 2] - eroded[i + 2]).abs();
            out[i + 3] = input[i + 3];
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        let buf_size = (width as usize) * (height as usize) * 16;
        // Pass 0: snapshot original → buffer 0
        // Pass 1: dilate
        // Pass 2: snapshot dilated → buffer 1
        // Pass 3: erode from buffer 0 (erode the original, not the dilated)
        // Pass 4: diff(buffer 1 [dilate], buffer 0... wait)
        //
        // Actually simpler: after pass 3, erode result is in ping-pong.
        // We need |dilate − erode|. Dilate is in buffer 1. Erode is current input.
        // Use sub shader: |buffer1 − current|
        Some(vec![
            // Pass 0: snapshot original → buf 0
            make_snapshot_shader(width, height, 0),
            // Pass 1: dilate (from original via passthrough)
            make_dilate_shader(width, height, self.radius),
            // Pass 2: snapshot dilate result → buf 1, passthrough
            {
                let mut params = gpu_params_wh(width, height);
                gpu_params_push_u32(&mut params, 0);
                gpu_params_push_u32(&mut params, 0);
                GpuShader {
                    body: SNAPSHOT_WGSL.to_string(),
                    entry_point: "main",
                    workgroup_size: [16, 16, 1],
                    params,
                    extra_buffers: vec![],
                    reduction_buffers: vec![ReductionBuffer {
                        id: 1,
                        initial_data: vec![0u8; buf_size],
                        read_write: true,
                    }],
                }
            },
            // Pass 3: erode from original (buf 0)
            make_erode_from_snap_shader(width, height, self.radius, 0),
            // Pass 4: |dilate (buf 1) − erode (current ping-pong input)|
            // Uses SUB_SNAP_MINUS_CURRENT: |snapshot − input| where snapshot=buf1=dilate
            make_sub_shader(SUB_SNAP_MINUS_CURRENT_WGSL, width, height, 1),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius }
}

// ═══════════════════════════════════════════════════════════════════════════
// Morph Top Hat (input − open)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological top hat — input minus opening (isolates bright details).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_tophat", category = "morphology")]
pub struct MorphTophat {
    #[param(min = 1, max = 20, step = 1, default = 2, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphTophat {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let eroded = erode_cpu(input, width, height, self.radius);
        let opened = dilate_cpu(&eroded, width, height, self.radius);
        let mut out = vec![0.0f32; input.len()];
        for i in (0..input.len()).step_by(4) {
            out[i] = (input[i] - opened[i]).max(0.0);
            out[i + 1] = (input[i + 1] - opened[i + 1]).max(0.0);
            out[i + 2] = (input[i + 2] - opened[i + 2]).max(0.0);
            out[i + 3] = input[i + 3];
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        // Pass 0: snapshot original → reduction buffer, passthrough to output
        // Pass 1: erode
        // Pass 2: dilate (open result)
        // Pass 3: output = snapshot − open (original − opened)
        Some(vec![
            make_snapshot_shader(width, height, 0),
            make_erode_shader(width, height, self.radius),
            make_dilate_shader(width, height, self.radius),
            make_sub_shader(SUB_SNAP_MINUS_CURRENT_WGSL, width, height, 0),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius * 2 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Morph Black Hat (close − input)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological black hat — closing minus input (isolates dark details).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "morph_blackhat", category = "morphology")]
pub struct MorphBlackhat {
    #[param(min = 1, max = 20, step = 1, default = 2, hint = "rc.pixels")]
    pub radius: u32,
}

impl Filter for MorphBlackhat {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let dilated = dilate_cpu(input, width, height, self.radius);
        let closed = erode_cpu(&dilated, width, height, self.radius);
        let mut out = vec![0.0f32; input.len()];
        for i in (0..input.len()).step_by(4) {
            out[i] = (closed[i] - input[i]).max(0.0);
            out[i + 1] = (closed[i + 1] - input[i + 1]).max(0.0);
            out[i + 2] = (closed[i + 2] - input[i + 2]).max(0.0);
            out[i + 3] = input[i + 3];
        }
        Ok(out)
    }

    fn gpu_shader_passes(&self, width: u32, height: u32) -> Option<Vec<GpuShader>> {
        // Pass 0: snapshot original → reduction buffer, passthrough to output
        // Pass 1: dilate
        // Pass 2: erode (close result)
        // Pass 3: output = close − snapshot (closed − original)
        Some(vec![
            make_snapshot_shader(width, height, 0),
            make_dilate_shader(width, height, self.radius),
            make_erode_shader(width, height, self.radius),
            make_sub_shader(SUB_CURRENT_MINUS_SNAP_WGSL, width, height, 0),
        ])
    }

    fn tile_overlap(&self) -> u32 { self.radius * 2 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Skeletonize
// ═══════════════════════════════════════════════════════════════════════════

/// Skeletonize — iterative morphological thinning to 1-pixel skeleton.
/// Operates on luminance: pixel is "on" if luma > threshold.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "skeletonize", category = "morphology")]
pub struct Skeletonize {
    #[param(min = 0.0, max = 1.0, step = 0.01, default = 0.5)]
    pub threshold: f32,
    #[param(min = 1, max = 100, step = 1, default = 50)]
    pub iterations: u32,
}

impl Filter for Skeletonize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;

        // Convert to binary image (luminance thresholding)
        let mut binary = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let luma = input[i] * 0.2126 + input[i + 1] * 0.7152 + input[i + 2] * 0.0722;
                binary[y * w + x] = luma > self.threshold;
            }
        }

        // Zhang-Suen thinning algorithm
        for _ in 0..self.iterations {
            let mut changed = false;

            // Step 1
            let mut to_remove = Vec::new();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    if !binary[y * w + x] { continue; }
                    let (transitions, neighbors) = zhang_suen_check(&binary, w, x, y);
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 {
                        let p2 = binary[(y - 1) * w + x] as u8;
                        let p4 = binary[y * w + x + 1] as u8;
                        let p6 = binary[(y + 1) * w + x] as u8;
                        let p8 = binary[y * w + x - 1] as u8;
                        if p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0 {
                            to_remove.push(y * w + x);
                        }
                    }
                }
            }
            for &idx in &to_remove { binary[idx] = false; changed = true; }

            // Step 2
            to_remove.clear();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    if !binary[y * w + x] { continue; }
                    let (transitions, neighbors) = zhang_suen_check(&binary, w, x, y);
                    if neighbors >= 2 && neighbors <= 6 && transitions == 1 {
                        let p2 = binary[(y - 1) * w + x] as u8;
                        let p4 = binary[y * w + x + 1] as u8;
                        let p6 = binary[(y + 1) * w + x] as u8;
                        let p8 = binary[y * w + x - 1] as u8;
                        if p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0 {
                            to_remove.push(y * w + x);
                        }
                    }
                }
            }
            for &idx in &to_remove { binary[idx] = false; changed = true; }

            if !changed { break; }
        }

        // Convert back to f32 RGBA
        let mut out = vec![0.0f32; input.len()];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let v = if binary[y * w + x] { 1.0f32 } else { 0.0 };
                out[i] = v;
                out[i + 1] = v;
                out[i + 2] = v;
                out[i + 3] = input[i + 3];
            }
        }
        Ok(out)
    }

    fn tile_overlap(&self) -> u32 { 1 }
}

/// Zhang-Suen helper: count transitions and neighbors in 8-connected ring.
fn zhang_suen_check(binary: &[bool], w: usize, x: usize, y: usize) -> (u8, u8) {
    let p = [
        binary[(y - 1) * w + x],     // P2
        binary[(y - 1) * w + x + 1], // P3
        binary[y * w + x + 1],       // P4
        binary[(y + 1) * w + x + 1], // P5
        binary[(y + 1) * w + x],     // P6
        binary[(y + 1) * w + x - 1], // P7
        binary[y * w + x - 1],       // P8
        binary[(y - 1) * w + x - 1], // P9
    ];
    let neighbors = p.iter().filter(|&&v| v).count() as u8;
    let mut transitions = 0u8;
    for i in 0..8 {
        if !p[i] && p[(i + 1) % 8] { transitions += 1; }
    }
    (transitions, neighbors)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_morphology_filters_registered() {
        let factories = crate::registered_filter_factories();
        for name in &["dilate", "erode", "morph_open", "morph_close",
                       "morph_gradient", "morph_tophat", "morph_blackhat", "skeletonize"] {
            assert!(factories.contains(name), "{name} not registered");
        }
    }

    #[test]
    fn dilate_expands_bright() {
        // 3x3 image, center pixel bright, rest dark
        let mut input = vec![0.0f32; 3 * 3 * 4];
        let center = (1 * 3 + 1) * 4;
        input[center] = 1.0;
        input[center + 1] = 1.0;
        input[center + 2] = 1.0;
        input[center + 3] = 1.0;

        let f = Dilate { radius: 1 };
        let out = f.compute(&input, 3, 3).unwrap();

        // All pixels should be bright after dilation
        for y in 0..3 {
            for x in 0..3 {
                let i = (y * 3 + x) * 4;
                assert!(out[i] >= 0.99, "pixel ({x},{y}) R={} should be 1.0", out[i]);
            }
        }
    }

    #[test]
    fn erode_shrinks_bright() {
        // 3x3 all bright except one dark corner
        let mut input = vec![1.0f32; 3 * 3 * 4];
        input[0] = 0.0;
        input[1] = 0.0;
        input[2] = 0.0;

        let f = Erode { radius: 1 };
        let out = f.compute(&input, 3, 3).unwrap();

        // Center should be eroded (dark pixel in neighborhood)
        let center = (1 * 3 + 1) * 4;
        assert!(out[center] < 0.01, "center should be eroded");
    }

    #[test]
    fn open_close_differ() {
        let mut input = vec![0.5f32; 8 * 8 * 4];
        // Add some noise
        input[0] = 1.0;
        input[4] = 0.0;

        let open = MorphOpen { radius: 1 };
        let close = MorphClose { radius: 1 };
        let o = open.compute(&input, 8, 8).unwrap();
        let c = close.compute(&input, 8, 8).unwrap();
        assert_ne!(o, c, "open and close should differ");
    }
}
