//! Morphology filters — min/max kernel operations on f32 pixel data.
//!
//! Dilate (max), erode (min), and compound operations (open, close,
//! gradient, tophat, blackhat, skeletonize). GPU shaders use per-pixel
//! neighborhood min/max scans.

mod dilate;
mod erode;
mod morph_open;
mod morph_close;
mod morph_gradient;
mod morph_tophat;
mod morph_blackhat;
mod skeletonize;

pub use dilate::Dilate;
pub use erode::Erode;
pub use morph_open::MorphOpen;
pub use morph_close::MorphClose;
pub use morph_gradient::MorphGradient;
pub use morph_tophat::MorphTophat;
pub use morph_blackhat::MorphBlackhat;
pub use skeletonize::Skeletonize;

use crate::node::GpuShader;

use super::helpers::gpu_params_wh;

pub(super) fn gpu_params_push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ─── Shared WGSL ───────────────────────────────────────────────────────────

pub(super) const DILATE_WGSL: &str = r#"
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

pub(super) const ERODE_WGSL: &str = r#"
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

pub(super) fn dilate_cpu(input: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
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

pub(super) fn erode_cpu(input: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
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
pub(super) const SNAPSHOT_WGSL: &str = r#"
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
pub(super) const SUB_SNAP_MINUS_CURRENT_WGSL: &str = r#"
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
pub(super) const SUB_CURRENT_MINUS_SNAP_WGSL: &str = r#"
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
pub(super) const ERODE_FROM_SNAP_WGSL: &str = r#"
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

pub(super) fn make_snapshot_shader(width: u32, height: u32, buf_id: u32) -> GpuShader {
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
            convergence_check: None,
            loop_dispatch: None,
    }
}

pub(super) fn make_sub_shader(wgsl: &str, width: u32, height: u32, buf_id: u32) -> GpuShader {
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
            convergence_check: None,
            loop_dispatch: None,
    }
}

/// Erode reading from a snapshot buffer instead of ping-pong input.
pub(super) fn make_erode_from_snap_shader(width: u32, height: u32, radius: u32, snap_id: u32) -> GpuShader {
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
            convergence_check: None,
            loop_dispatch: None,
    }
}

pub(super) fn make_dilate_shader(width: u32, height: u32, radius: u32) -> GpuShader {
    let mut params = gpu_params_wh(width, height);
    gpu_params_push_u32(&mut params, radius);
    gpu_params_push_u32(&mut params, 0);
    GpuShader::new(DILATE_WGSL.to_string(), "main", [16, 16, 1], params)
}

pub(super) fn make_erode_shader(width: u32, height: u32, radius: u32) -> GpuShader {
    let mut params = gpu_params_wh(width, height);
    gpu_params_push_u32(&mut params, radius);
    gpu_params_push_u32(&mut params, 0);
    GpuShader::new(ERODE_WGSL.to_string(), "main", [16, 16, 1], params)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;

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
