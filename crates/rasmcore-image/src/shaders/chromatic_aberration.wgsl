// Chromatic aberration — radial RGB fringing from center

struct Params {
  width: u32,
  height: u32,
  strength: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn clamp_coord(v: f32, max_val: u32) -> u32 {
  return u32(clamp(v, 0.0, f32(max_val - 1u)));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let cx = f32(params.width) * 0.5;
  let cy = f32(params.height) * 0.5;
  let max_dist = sqrt(cx * cx + cy * cy);

  let fx = f32(x);
  let fy = f32(y);
  let dx = fx - cx;
  let dy = fy - cy;
  let dist = sqrt(dx * dx + dy * dy);

  let r_scale = params.strength / max(max_dist, 1e-6);
  let b_scale = -params.strength / max(max_dist, 1e-6);
  let inv_dist = 1.0 / max(dist, 1e-6);
  let nx = dx * inv_dist;
  let ny = dy * inv_dist;

  // Red: shift outward
  let r_shift = dist * r_scale;
  let r_sx = clamp_coord(fx + nx * r_shift, params.width);
  let r_sy = clamp_coord(fy + ny * r_shift, params.height);
  let r_pixel = unpack(input[r_sy * params.width + r_sx]);

  // Green: no shift
  let g_pixel = unpack(input[y * params.width + x]);

  // Blue: shift inward
  let b_shift = dist * b_scale;
  let b_sx = clamp_coord(fx + nx * b_shift, params.width);
  let b_sy = clamp_coord(fy + ny * b_shift, params.height);
  let b_pixel = unpack(input[b_sy * params.width + b_sx]);

  output[y * params.width + x] = pack(vec4<f32>(r_pixel.r, g_pixel.g, b_pixel.b, g_pixel.a));
}
