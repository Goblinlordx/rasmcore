// Chromatic aberration removal — radial R/B channel realignment
// Uses bilinear sampling for sub-pixel accuracy.

struct Params {
  width: u32,
  height: u32,
  red_shift: f32,
  blue_shift: f32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let cx = f32(params.width) * 0.5;
  let cy = f32(params.height) * 0.5;
  let max_r2 = cx * cx + cy * cy;

  let dx = f32(x) - cx;
  let dy = f32(y) - cy;
  let r2 = dx * dx + dy * dy;

  // Red channel: shift by red_shift * (r²/max_r²)
  let r_scale = 1.0 + params.red_shift * r2 / max_r2;
  let rx = cx + dx * r_scale;
  let ry = cy + dy * r_scale;
  let r_color = sample_bilinear(rx, ry);

  // Green: unchanged
  let g_color = unpack(input[y * params.width + x]);

  // Blue channel: shift by blue_shift * (r²/max_r²)
  let b_scale = 1.0 + params.blue_shift * r2 / max_r2;
  let bx = cx + dx * b_scale;
  let by = cy + dy * b_scale;
  let b_color = sample_bilinear(bx, by);

  output[y * params.width + x] = pack(vec4<f32>(r_color.r, g_color.g, b_color.b, g_color.a));
}
