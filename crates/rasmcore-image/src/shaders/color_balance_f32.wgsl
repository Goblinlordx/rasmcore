// Color balance — PS-style shadow/midtone/highlight CMY-RGB, f32 per-pixel

struct Params {
  width: u32,
  height: u32,
  shadow_r: f32,
  shadow_g: f32,
  shadow_b: f32,
  midtone_r: f32,
  midtone_g: f32,
  midtone_b: f32,
  highlight_r: f32,
  highlight_g: f32,
  highlight_b: f32,
  preserve_lum: f32,  // 1.0 = preserve, 0.0 = don't
  _pad: u32,
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

  let luma = 0.2126 * px.r + 0.7152 * px.g + 0.0722 * px.b;

  let shadow_w = min((1.0 - luma) * (1.0 - luma) * 1.5, 1.0);
  let highlight_w = min(luma * luma * 1.5, 1.0);
  let midtone_w = max(1.0 - shadow_w - highlight_w, 0.0);

  let dr = params.shadow_r * shadow_w + params.midtone_r * midtone_w + params.highlight_r * highlight_w;
  let dg = params.shadow_g * shadow_w + params.midtone_g * midtone_w + params.highlight_g * highlight_w;
  let db = params.shadow_b * shadow_w + params.midtone_b * midtone_w + params.highlight_b * highlight_w;

  var out_r = clamp(px.r + dr, 0.0, 1.0);
  var out_g = clamp(px.g + dg, 0.0, 1.0);
  var out_b = clamp(px.b + db, 0.0, 1.0);

  if (params.preserve_lum > 0.5) {
    let new_luma = 0.2126 * out_r + 0.7152 * out_g + 0.0722 * out_b;
    if (new_luma > 1e-6) {
      let scale = luma / new_luma;
      out_r = clamp(out_r * scale, 0.0, 1.0);
      out_g = clamp(out_g * scale, 0.0, 1.0);
      out_b = clamp(out_b * scale, 0.0, 1.0);
    }
  }

  output[idx] = vec4<f32>(out_r, out_g, out_b, px.a);
}
