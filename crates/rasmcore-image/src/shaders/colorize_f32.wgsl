// Colorize — W3C Color blend mode (SetLum/ClipColor), f32 per-pixel
// target_r/g/b = target color [0,1], amount = blend factor

struct Params {
  width: u32,
  height: u32,
  target_r: f32,
  target_g: f32,
  target_b: f32,
  amount: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];

  // BT.601 luma (matches Photoshop Luminosity weights)
  let pixel_luma = 0.299 * px.r + 0.587 * px.g + 0.114 * px.b;
  let target_luma = 0.299 * params.target_r + 0.587 * params.target_g + 0.114 * params.target_b;

  // W3C SetLum: shift target color to match pixel's luma
  let d = pixel_luma - target_luma;
  var cr = params.target_r + d;
  var cg = params.target_g + d;
  var cb = params.target_b + d;

  // W3C ClipColor: clamp to [0,1] preserving luma
  let l = pixel_luma;
  let n = min(min(cr, cg), cb);
  let mx = max(max(cr, cg), cb);
  if (n < 0.0) {
    let ln = l - n;
    cr = l + (cr - l) * l / ln;
    cg = l + (cg - l) * l / ln;
    cb = l + (cb - l) * l / ln;
  }
  if (mx > 1.0) {
    let xl = mx - l;
    let one_l = 1.0 - l;
    cr = l + (cr - l) * one_l / xl;
    cg = l + (cg - l) * one_l / xl;
    cb = l + (cb - l) * one_l / xl;
  }

  // Lerp between original and colorized
  let or = px.r + (cr - px.r) * params.amount;
  let og = px.g + (cg - px.g) * params.amount;
  let ob = px.b + (cb - px.b) * params.amount;

  output[idx] = vec4<f32>(clamp(or, 0.0, 1.0), clamp(og, 0.0, 1.0), clamp(ob, 0.0, 1.0), px.a);
}
