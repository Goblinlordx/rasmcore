//! GPU shaders for stamp-based brush rendering.
//!
//! Two-pass pipeline:
//! 1. **Stamp pass** — render dab instances into accumulation buffer.
//!    Per-pixel: iterate over dab instances, evaluate parametric tip,
//!    max-opacity blend. Supports custom tip textures.
//! 2. **Composite pass** — blend accumulation onto source layer.
//!    Standard premultiplied "over" compositing with stroke opacity.
//!
//! # Buffer layout
//!
//! - `input` (binding 0): Source layer pixels (f32 RGBA).
//! - `output` (binding 1): Composited result (f32 RGBA).
//! - `params` (binding 2): BrushGpuParams uniform.
//! - `extra_buffers[0]` (binding 3): DabInstance array (packed f32×7 per dab).
//! - `extra_buffers[1]` (binding 4): Brush tip texture (grayscale f32, size×size).
//!   If empty/absent, shader uses parametric circle falloff.

/// GPU-side brush params uniform (48 bytes, 16-byte aligned).
#[derive(Debug, Clone, Copy)]
pub struct BrushGpuParams {
    pub width: u32,
    pub height: u32,
    pub dab_count: u32,
    pub first_new_dab: u32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub color_a: f32,
    pub stroke_opacity: f32,
    pub tip_size: u32,
}

impl BrushGpuParams {
    /// Serialize to uniform buffer bytes (48 bytes, padded to 16-byte alignment).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(48);
        buf.extend_from_slice(&self.width.to_le_bytes());
        buf.extend_from_slice(&self.height.to_le_bytes());
        buf.extend_from_slice(&self.dab_count.to_le_bytes());
        buf.extend_from_slice(&self.first_new_dab.to_le_bytes());
        buf.extend_from_slice(&self.color_r.to_le_bytes());
        buf.extend_from_slice(&self.color_g.to_le_bytes());
        buf.extend_from_slice(&self.color_b.to_le_bytes());
        buf.extend_from_slice(&self.color_a.to_le_bytes());
        buf.extend_from_slice(&self.stroke_opacity.to_le_bytes());
        buf.extend_from_slice(&self.tip_size.to_le_bytes());
        // Pad to 48 bytes (12 × 4)
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
}

/// Serialize a slice of CPU DabInstances into GPU buffer bytes.
///
/// Layout: 7 contiguous f32s per dab (x, y, size, opacity, angle, roundness, hardness).
pub fn serialize_dab_instances(dabs: &[crate::brush::types::DabInstance]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(dabs.len() * 28);
    for d in dabs {
        buf.extend_from_slice(&d.x.to_le_bytes());
        buf.extend_from_slice(&d.y.to_le_bytes());
        buf.extend_from_slice(&d.size.to_le_bytes());
        buf.extend_from_slice(&d.opacity.to_le_bytes());
        buf.extend_from_slice(&d.angle.to_le_bytes());
        buf.extend_from_slice(&d.roundness.to_le_bytes());
        buf.extend_from_slice(&d.hardness.to_le_bytes());
    }
    buf
}

/// Serialize an f32 tip texture into GPU buffer bytes.
pub fn serialize_tip_texture(tip: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(tip.len() * 4);
    for &v in tip {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Stamp pass — render dab instances into the output buffer.
///
/// Per-pixel shader: for each output pixel, iterate over all dab instances
/// that overlap this pixel. Evaluate parametric circle tip (with hardness,
/// roundness, rotation) or sample custom tip texture. Max-opacity blend.
pub const STAMP_PASS: &str = r#"
struct BrushParams {
  width: u32,
  height: u32,
  dab_count: u32,
  first_new_dab: u32,
  color_r: f32,
  color_g: f32,
  color_b: f32,
  color_a: f32,
  stroke_opacity: f32,
  tip_size: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: BrushParams;
@group(0) @binding(3) var<storage, read> dabs: array<f32>;
@group(0) @binding(4) var<storage, read> tip_tex: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn stamp_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= params.width || py >= params.height) { return; }
  let idx = py * params.width + px;

  var accum = output[idx];

  let pixel_x = f32(px) + 0.5;
  let pixel_y = f32(py) + 0.5;

  for (var di = params.first_new_dab; di < params.dab_count; di++) {
    let base = di * 7u;
    let dab_x = dabs[base];
    let dab_y = dabs[base + 1u];
    let dab_size = dabs[base + 2u];
    let dab_opacity = dabs[base + 3u];
    let dab_angle = dabs[base + 4u];
    let dab_roundness = dabs[base + 5u];
    let dab_hardness = dabs[base + 6u];

    let half = dab_size * 0.5;
    if (pixel_x < dab_x - half || pixel_x > dab_x + half ||
        pixel_y < dab_y - half || pixel_y > dab_y + half) {
      continue;
    }

    let dx = pixel_x - dab_x;
    let dy = pixel_y - dab_y;

    var stamp_alpha: f32;

    if (params.tip_size > 0u) {
      let cos_a = cos(dab_angle);
      let sin_a = sin(dab_angle);
      let rx = dx * cos_a + dy * sin_a;
      let ry = -dx * sin_a + dy * cos_a;
      let scale = f32(params.tip_size) / dab_size;
      let tx = rx * scale + f32(params.tip_size) * 0.5;
      let ty = ry * scale + f32(params.tip_size) * 0.5;
      stamp_alpha = bilinear_sample_tip(tx, ty, params.tip_size);
    } else {
      let cos_a = cos(dab_angle);
      let sin_a = sin(dab_angle);
      let rx = dx * cos_a + dy * sin_a;
      let inv_round = select(1.0 / dab_roundness, 100.0, dab_roundness < 1e-6);
      let ry = (-dx * sin_a + dy * cos_a) * inv_round;
      let dist = sqrt(rx * rx + ry * ry) / half;

      if (dist >= 1.0) {
        continue;
      }
      if (dab_hardness >= 1.0) {
        stamp_alpha = 1.0;
      } else {
        let inner = dab_hardness;
        if (dist <= inner) {
          stamp_alpha = 1.0;
        } else {
          let t = (dist - inner) / (1.0 - inner);
          stamp_alpha = max(1.0 - t * t, 0.0);
        }
      }
    }

    let sa = stamp_alpha * dab_opacity;

    if (sa > accum.w) {
      accum = vec4<f32>(
        params.color_r * sa,
        params.color_g * sa,
        params.color_b * sa,
        sa,
      );
    }
  }

  output[idx] = accum;
}

fn bilinear_sample_tip(tx: f32, ty: f32, size: u32) -> f32 {
  let fsize = f32(size);
  if (tx < 0.0 || tx >= fsize || ty < 0.0 || ty >= fsize) {
    return 0.0;
  }
  let x0 = u32(floor(tx));
  let y0 = u32(floor(ty));
  let x1 = min(x0 + 1u, size - 1u);
  let y1 = min(y0 + 1u, size - 1u);
  let fx = tx - floor(tx);
  let fy = ty - floor(ty);

  let v00 = tip_tex[y0 * size + x0];
  let v10 = tip_tex[y0 * size + x1];
  let v01 = tip_tex[y1 * size + x0];
  let v11 = tip_tex[y1 * size + x1];

  let top = mix(v00, v10, fx);
  let bot = mix(v01, v11, fx);
  return mix(top, bot, fy);
}
"#;

/// Composite pass — blend accumulation buffer onto source layer.
pub const COMPOSITE_PASS: &str = r#"
struct BrushParams {
  width: u32,
  height: u32,
  dab_count: u32,
  first_new_dab: u32,
  color_r: f32,
  color_g: f32,
  color_b: f32,
  color_a: f32,
  stroke_opacity: f32,
  tip_size: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: BrushParams;
@group(0) @binding(3) var<storage, read> layer: array<vec4<f32>>;

@compute @workgroup_size(16, 16, 1)
fn composite_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= params.width || py >= params.height) { return; }
  let idx = py * params.width + px;

  let src = layer[idx];
  let acc = accum[idx];
  let sa = acc.w * params.stroke_opacity;

  if (sa < 1e-6) {
    output[idx] = src;
    return;
  }

  let inv_sa = 1.0 - sa;
  output[idx] = vec4<f32>(
    acc.x * params.stroke_opacity + src.x * inv_sa,
    acc.y * params.stroke_opacity + src.y * inv_sa,
    acc.z * params.stroke_opacity + src.z * inv_sa,
    sa + src.w * inv_sa,
  );
}
"#;

/// Build the GpuShader sequence for brush stamp + composite.
///
/// Returns two shaders: [stamp_pass, composite_pass].
/// The stamp pass renders dabs into the ping-pong buffer (initially zeroed).
/// The composite pass blends the accumulation onto the source layer.
pub fn brush_gpu_shaders(
    gpu_params: &BrushGpuParams,
    dab_bytes: Vec<u8>,
    layer_bytes: Vec<u8>,
    tip_bytes: Vec<u8>,
) -> Vec<crate::node::GpuShader> {
    let params_bytes = gpu_params.to_bytes();

    let stamp = crate::node::GpuShader {
        body: STAMP_PASS.to_string(),
        entry_point: "stamp_main",
        workgroup_size: [16, 16, 1],
        params: params_bytes.clone(),
        extra_buffers: vec![dab_bytes, tip_bytes],
        reduction_buffers: vec![],
        convergence_check: None,
        loop_dispatch: None,
    };

    let composite = crate::node::GpuShader {
        body: COMPOSITE_PASS.to_string(),
        entry_point: "composite_main",
        workgroup_size: [16, 16, 1],
        params: params_bytes,
        extra_buffers: vec![layer_bytes],
        reduction_buffers: vec![],
        convergence_check: None,
        loop_dispatch: None,
    };

    vec![stamp, composite]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_gpu_params_serialization() {
        let params = BrushGpuParams {
            width: 64, height: 64,
            dab_count: 5, first_new_dab: 0,
            color_r: 1.0, color_g: 0.0, color_b: 0.0, color_a: 1.0,
            stroke_opacity: 1.0, tip_size: 0,
        };
        let bytes = params.to_bytes();
        assert_eq!(bytes.len(), 48); // 12 × 4 bytes (16-byte aligned)

        // Verify width
        let width = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(width, 64);
        // Verify color_r at offset 16
        let color_r = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert!((color_r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn serialize_dab_instances_roundtrip() {
        let dabs = vec![
            crate::brush::types::DabInstance {
                x: 10.0, y: 20.0, size: 15.0, opacity: 0.8,
                angle: 0.5, roundness: 1.0, hardness: 0.7,
            },
            crate::brush::types::DabInstance {
                x: 30.0, y: 40.0, size: 25.0, opacity: 1.0,
                angle: 0.0, roundness: 0.5, hardness: 1.0,
            },
        ];
        let bytes = serialize_dab_instances(&dabs);
        assert_eq!(bytes.len(), 2 * 28); // 2 dabs × 7 floats × 4 bytes

        // Verify first dab x
        let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert!((x - 10.0).abs() < 1e-6);
        // Verify second dab size (offset: 28 + 2*4 = 36)
        let size = f32::from_le_bytes([bytes[36], bytes[37], bytes[38], bytes[39]]);
        assert!((size - 25.0).abs() < 1e-6);
    }

    #[test]
    fn brush_gpu_shaders_produces_two_passes() {
        let params = BrushGpuParams {
            width: 64, height: 64,
            dab_count: 5, first_new_dab: 0,
            color_r: 1.0, color_g: 0.0, color_b: 0.0, color_a: 1.0,
            stroke_opacity: 1.0, tip_size: 0,
        };
        let shaders = brush_gpu_shaders(
            &params,
            vec![0u8; 28 * 5],
            vec![0u8; 64 * 64 * 16],
            vec![],
        );
        assert_eq!(shaders.len(), 2);
        assert_eq!(shaders[0].entry_point, "stamp_main");
        assert_eq!(shaders[1].entry_point, "composite_main");
        assert_eq!(shaders[0].extra_buffers.len(), 2);
        assert_eq!(shaders[1].extra_buffers.len(), 1);
    }

    #[test]
    fn incremental_rendering_params() {
        let params = BrushGpuParams {
            width: 32, height: 32,
            dab_count: 10, first_new_dab: 5,
            color_r: 0.0, color_g: 0.0, color_b: 1.0, color_a: 1.0,
            stroke_opacity: 0.8, tip_size: 0,
        };
        assert_eq!(params.first_new_dab, 5);
        assert_eq!(params.dab_count, 10);
    }

    #[test]
    fn tip_texture_serialization() {
        let tip = vec![0.0f32, 0.5, 1.0, 0.5];
        let bytes = serialize_tip_texture(&tip);
        assert_eq!(bytes.len(), 16); // 4 floats × 4 bytes
        let v = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert!((v - 0.5).abs() < 1e-6);
    }
}
