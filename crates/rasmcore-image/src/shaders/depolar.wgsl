// Depolar — inverse polar coordinate transform (polar to Cartesian)
// Reverses the polar transform: maps (angle, radius) back to (x, y).

struct Params {
  width: u32,
  height: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn unpack(pixel: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(pixel & 0xFFu),
    f32((pixel >> 8u) & 0xFFu),
    f32((pixel >> 16u) & 0xFFu),
    f32((pixel >> 24u) & 0xFFu),
  );
}

fn pack(color: vec4<f32>) -> u32 {
  let r = u32(clamp(color.x, 0.0, 255.0));
  let g = u32(clamp(color.y, 0.0, 255.0));
  let b = u32(clamp(color.z, 0.0, 255.0));
  let a = u32(clamp(color.w, 0.0, 255.0));
  return r | (g << 8u) | (b << 16u) | (a << 24u);
}

fn sample_bilinear(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let dx = fx - f32(ix);
  let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.width) - 1);
  let y0 = clamp(iy, 0, i32(params.height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.height) - 1);
  let p00 = unpack(input[u32(x0) + u32(y0) * params.width]);
  let p10 = unpack(input[u32(x1) + u32(y0) * params.width]);
  let p01 = unpack(input[u32(x0) + u32(y1) * params.width]);
  let p11 = unpack(input[u32(x1) + u32(y1) * params.width]);
  return mix(mix(p00, p10, dx), mix(p01, p11, dx), dy);
}

const PI: f32 = 3.14159265358979;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let cx = f32(params.width) * 0.5;
  let cy = f32(params.height) * 0.5;
  let max_radius = sqrt(cx * cx + cy * cy);

  // Source pixel (x, y) is in Cartesian space
  let dx = f32(x) - cx;
  let dy = f32(y) - cy;
  let radius = sqrt(dx * dx + dy * dy);
  var angle = atan2(dy, dx);
  if (angle < 0.0) {
    angle += 2.0 * PI;
  }

  // Map to polar image coordinates
  let sx = angle / (2.0 * PI) * f32(params.width);
  let sy = radius / max_radius * f32(params.height);

  output[x + y * params.width] = pack(sample_bilinear(sx, sy));
}
