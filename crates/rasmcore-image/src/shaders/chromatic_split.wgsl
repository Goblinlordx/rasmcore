// Chromatic split — per-channel spatial offset (prism/RGB split)

struct Params {
  width: u32,
  height: u32,
  red_dx: f32,
  red_dy: f32,
  green_dx: f32,
  green_dy: f32,
  blue_dx: f32,
  blue_dy: f32,
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

  let fx = f32(x);
  let fy = f32(y);

  // Sample each channel from its offset position
  let r_sx = clamp_coord(fx + params.red_dx, params.width);
  let r_sy = clamp_coord(fy + params.red_dy, params.height);
  let r_pixel = unpack(input[r_sy * params.width + r_sx]);

  let g_sx = clamp_coord(fx + params.green_dx, params.width);
  let g_sy = clamp_coord(fy + params.green_dy, params.height);
  let g_pixel = unpack(input[g_sy * params.width + g_sx]);

  let b_sx = clamp_coord(fx + params.blue_dx, params.width);
  let b_sy = clamp_coord(fy + params.blue_dy, params.height);
  let b_pixel = unpack(input[b_sy * params.width + b_sx]);

  // Original pixel for alpha
  let orig = unpack(input[y * params.width + x]);

  output[y * params.width + x] = pack(vec4<f32>(r_pixel.r, g_pixel.g, b_pixel.b, orig.a));
}
