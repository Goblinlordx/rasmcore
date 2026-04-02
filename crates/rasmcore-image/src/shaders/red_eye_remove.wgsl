// Red-eye removal — localized red desaturation within circular region

struct Params {
  width: u32,
  height: u32,
  center_x: u32,
  center_y: u32,
  radius: u32,
  darken: f32,
  threshold: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Simplified RGB to HSL (hue + saturation + luminance)
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> vec3<f32> {
  let mx = max(r, max(g, b));
  let mn = min(r, min(g, b));
  let l = (mx + mn) * 0.5;
  let delta = mx - mn;
  if (delta < 1e-6) { return vec3<f32>(0.0, 0.0, l); }
  let s = select(delta / (2.0 - mx - mn), delta / (mx + mn), l < 0.5);
  var h: f32;
  if (mx == r) { h = 60.0 * ((g - b) / delta % 6.0); }
  else if (mx == g) { h = 60.0 * ((b - r) / delta + 2.0); }
  else { h = 60.0 * ((r - g) / delta + 4.0); }
  if (h < 0.0) { h = h + 360.0; }
  return vec3<f32>(h, s, l);
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
  if (s < 1e-6) { return vec3<f32>(l, l, l); }
  let c = (1.0 - abs(2.0 * l - 1.0)) * s;
  let hp = h / 60.0;
  let x = c * (1.0 - abs(hp % 2.0 - 1.0));
  var r1: f32; var g1: f32; var b1: f32;
  if (hp < 1.0) { r1 = c; g1 = x; b1 = 0.0; }
  else if (hp < 2.0) { r1 = x; g1 = c; b1 = 0.0; }
  else if (hp < 3.0) { r1 = 0.0; g1 = c; b1 = x; }
  else if (hp < 4.0) { r1 = 0.0; g1 = x; b1 = c; }
  else if (hp < 5.0) { r1 = x; g1 = 0.0; b1 = c; }
  else { r1 = c; g1 = 0.0; b1 = x; }
  let m = l - c * 0.5;
  return vec3<f32>(r1 + m, g1 + m, b1 + m);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = y * params.width + x;
  let pixel = unpack(input[idx]);

  let dx = f32(x) - f32(params.center_x);
  let dy = f32(y) - f32(params.center_y);
  let dist = sqrt(dx * dx + dy * dy);
  let r = f32(params.radius);

  if (dist > r) {
    output[idx] = input[idx];
    return;
  }

  let hsl = rgb_to_hsl(pixel.r, pixel.g, pixel.b);
  let hue = hsl.x;
  let sat = hsl.y;
  let lum = hsl.z;

  // Only correct red-hue, high-saturation pixels
  let is_red = hue >= 340.0 || hue <= 20.0;
  if (!is_red || sat < params.threshold) {
    output[idx] = input[idx];
    return;
  }

  let edge_factor = 1.0 - (dist / r) * (dist / r);
  let new_rgb = hsl_to_rgb(hue, 0.0, lum * (1.0 - params.darken * edge_factor));
  output[idx] = pack(vec4<f32>(new_rgb.x, new_rgb.y, new_rgb.z, pixel.a));
}
