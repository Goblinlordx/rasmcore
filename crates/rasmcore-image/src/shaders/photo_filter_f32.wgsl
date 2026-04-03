// Photo filter — blend pixel with filter color at density, optionally preserve luminance
// BT.709 luma: 0.2126*R + 0.7152*G + 0.0722*B

struct Params {
  width: u32,
  height: u32,
  color_r: f32,
  color_g: f32,
  color_b: f32,
  density: f32,
  preserve_luminosity: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn bt709_luma(c: vec3<f32>) -> f32 {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];

  let filter_color = vec3<f32>(params.color_r, params.color_g, params.color_b);
  var blended = mix(px.rgb, filter_color, params.density);

  if (params.preserve_luminosity != 0u) {
    let orig_luma = bt709_luma(px.rgb);
    let new_luma = bt709_luma(blended);
    if (new_luma > 0.0) {
      blended = blended * (orig_luma / new_luma);
    }
  }

  output[idx] = vec4<f32>(clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0)), px.a);
}
