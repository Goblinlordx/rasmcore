// Dodge / Burn — luminance-weighted tonal range, f32 per-pixel
// range: 0 = shadows, 1 = midtones, 2 = highlights
// is_dodge: 1 = dodge (lighten), 0 = burn (darken)

struct Params {
  width: u32,
  height: u32,
  exposure: f32,
  range: u32,
  is_dodge: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn range_weight(luma: f32, range: u32) -> f32 {
  if (range == 0u) {
    // Shadows: smoothstep falloff from dark to mid
    return 1.0 - smoothstep(0.0, 0.5, luma);
  } else if (range == 1u) {
    // Midtones: bell curve peaking at 0.5
    return 4.0 * luma * (1.0 - luma);
  } else {
    // Highlights: smoothstep ramp from mid to bright
    return smoothstep(0.5, 1.0, luma);
  }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = unpack_f32(input[idx]);

  // BT.709 luminance
  let luma = 0.2126 * px.r + 0.7152 * px.g + 0.0722 * px.b;
  let w = range_weight(luma, params.range);
  let factor = params.exposure * w;

  var rgb: vec3<f32>;
  if (params.is_dodge == 1u) {
    // Dodge: lighten
    rgb = px.rgb + px.rgb * factor;
  } else {
    // Burn: darken
    rgb = px.rgb * (1.0 - factor);
  }

  let clamped = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  output[idx] = pack_f32(vec4<f32>(clamped, px.a));
}
