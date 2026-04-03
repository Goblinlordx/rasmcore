// Halftone — CMYK-style dot pattern via rotated sine-wave screening.

struct Params {
  width: u32,
  height: u32,
  freq: f32,
  angle0: f32,
  angle1: f32,
  angle2: f32,
  angle3: f32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn screen(x: f32, y: f32, angle: f32, freq: f32) -> f32 {
  let ca = cos(angle);
  let sa = sin(angle);
  let rx = x * ca + y * sa;
  let ry = -x * sa + y * ca;
  return (sin(rx * freq) * sin(ry * freq) + 1.0) * 0.5;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let xf = f32(gid.x);
  let yf = f32(gid.y);

  // RGB to CMYK
  let k = 1.0 - max(pixel.x, max(pixel.y, pixel.z));
  let inv_k = select(0.0, 1.0 / (1.0 - k), k < 1.0);
  let c = (1.0 - pixel.x - k) * inv_k;
  let m = (1.0 - pixel.y - k) * inv_k;
  let yc = (1.0 - pixel.z - k) * inv_k;

  // Screen each channel at its angle
  let sc = select(0.0, 1.0, c > screen(xf, yf, params.angle0, params.freq));
  let sm = select(0.0, 1.0, m > screen(xf, yf, params.angle1, params.freq));
  let sy = select(0.0, 1.0, yc > screen(xf, yf, params.angle2, params.freq));
  let sk = select(0.0, 1.0, k > screen(xf, yf, params.angle3, params.freq));

  // CMYK back to RGB
  let inv_k2 = 1.0 - sk;
  let r = (1.0 - sc) * inv_k2;
  let g = (1.0 - sm) * inv_k2;
  let b = (1.0 - sy) * inv_k2;

  store_pixel(idx, vec4<f32>(r, g, b, pixel.w));
}
