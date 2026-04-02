// Liquify expand — push pixels away from brush center (bloat)
// Format-agnostic: uses load_pixel/store_pixel from composed I/O fragment.

struct Params {
  width: u32,
  height: u32,
  center_x: f32,
  center_y: f32,
  radius: f32,
  strength: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn gaussian_weight(dist: f32, radius: f32) -> f32 {
  let t = dist / radius;
  return exp(-2.0 * t * t);
}

fn sample_bilinear_load(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let dx = fx - f32(ix);
  let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.width) - 1);
  let y0 = clamp(iy, 0, i32(params.height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.height) - 1);
  let p00 = load_pixel(u32(x0) + u32(y0) * params.width);
  let p10 = load_pixel(u32(x1) + u32(y0) * params.width);
  let p01 = load_pixel(u32(x0) + u32(y1) * params.width);
  let p11 = load_pixel(u32(x1) + u32(y1) * params.width);
  return mix(mix(p00, p10, vec4<f32>(dx)), mix(p01, p11, vec4<f32>(dx)), vec4<f32>(dy));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let xf = f32(x);
  let yf = f32(y);
  let ddx = xf - params.center_x;
  let ddy = yf - params.center_y;
  let dist = sqrt(ddx * ddx + ddy * ddy);

  if (dist >= params.radius || dist < 0.001) {
    store_pixel(x + y * params.width, load_pixel(x + y * params.width));
    return;
  }

  let w = gaussian_weight(dist, params.radius) * params.strength;
  // Expand inverse: source = (output + center * w) / (1 + w)
  let inv = 1.0 / (1.0 + w);
  let sx = (xf + params.center_x * w) * inv;
  let sy = (yf + params.center_y * w) * inv;
  store_pixel(x + y * params.width, sample_bilinear_load(sx, sy));
}
