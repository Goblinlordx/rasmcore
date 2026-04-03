// Modulate — combined brightness/saturation/hue in HSL, f32 per-pixel
// brightness = multiplier (1.0 = unchanged), saturation = multiplier, hue = degrees offset

struct Params {
  width: u32,
  height: u32,
  brightness: f32,
  saturation: f32,
  hue: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> vec3<f32> {
  let mx = max(max(r, g), b);
  let mn = min(min(r, g), b);
  let l = (mx + mn) * 0.5;
  if (mx == mn) { return vec3<f32>(0.0, 0.0, l); }
  let d = mx - mn;
  let s = select(d / (mx + mn), d / (2.0 - mx - mn), l > 0.5);
  var h: f32;
  if (mx == r) { h = (g - b) / d + select(0.0, 6.0, g < b); }
  else if (mx == g) { h = (b - r) / d + 2.0; }
  else { h = (r - g) / d + 4.0; }
  return vec3<f32>(h * 60.0, s, l);
}

fn hue_channel(p: f32, q: f32, t_in: f32) -> f32 {
  var t = t_in;
  if (t < 0.0) { t += 360.0; }
  if (t > 360.0) { t -= 360.0; }
  if (t < 60.0) { return p + (q - p) * t / 60.0; }
  if (t < 180.0) { return q; }
  if (t < 240.0) { return p + (q - p) * (240.0 - t) / 60.0; }
  return p;
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
  if (s == 0.0) { return vec3<f32>(l); }
  let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
  let p = 2.0 * l - q;
  return vec3<f32>(
    hue_channel(p, q, h + 120.0),
    hue_channel(p, q, h),
    hue_channel(p, q, h - 120.0),
  );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];
  var hsl = rgb_to_hsl(px.r, px.g, px.b);
  hsl.x = hsl.x + params.hue;
  if (hsl.x >= 360.0) { hsl.x -= 360.0; }
  if (hsl.x < 0.0) { hsl.x += 360.0; }
  hsl.y = clamp(hsl.y * params.saturation, 0.0, 1.0);
  hsl.z = clamp(hsl.z * params.brightness, 0.0, 1.0);
  let rgb = hsl_to_rgb(hsl.x, hsl.y, hsl.z);
  output[idx] = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), px.a);
}
