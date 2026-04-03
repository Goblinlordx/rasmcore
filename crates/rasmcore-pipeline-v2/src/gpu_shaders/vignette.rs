//! GPU shader for Gaussian vignette.

/// Gaussian vignette — builds elliptical mask and multiplies.
///
/// All in one pass: compute elliptical distance, apply Gaussian falloff, multiply.
/// This avoids the multi-pass blur-mask approach of the CPU path by using
/// a smooth analytical Gaussian falloff instead.
pub const VIGNETTE_GAUSSIAN: &str = r#"
struct Params {
  width: u32,
  height: u32,
  sigma: f32,
  x_inset: f32,
  y_inset: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.x + gid.y * params.width;

  let cx = f32(params.width) / 2.0;
  let cy = f32(params.height) / 2.0;
  let rx = max(cx - params.x_inset, 1.0);
  let ry = max(cy - params.y_inset, 1.0);

  let dx = (f32(gid.x) - cx) / rx;
  let dy = (f32(gid.y) - cy) / ry;
  let dist2 = dx * dx + dy * dy;

  // Smooth Gaussian falloff outside unit ellipse
  let falloff = select(
    exp(-0.5 * (dist2 - 1.0) / max(params.sigma * params.sigma, 0.001)),
    1.0,
    dist2 <= 1.0
  );

  let pixel = input[idx];
  output[idx] = vec4<f32>(pixel.xyz * falloff, pixel.w);
}
"#;
