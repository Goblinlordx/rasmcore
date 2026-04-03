// White balance — color temperature + tint shift, f32 per-pixel
// Tanner Helland Planckian locus approximation, normalized to D65 (6500K)

struct Params {
  width: u32,
  height: u32,
  temperature: f32,
  tint: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

// Convert Kelvin to linear RGB multiplier (Tanner Helland formula)
fn kelvin_to_rgb(temp_k: f32) -> vec3<f32> {
  let t = temp_k / 100.0;
  var r: f32;
  var g: f32;
  var b: f32;

  // Red
  if (t <= 66.0) {
    r = 255.0;
  } else {
    r = 329.698727446 * pow(t - 60.0, -0.1332047592);
    r = clamp(r, 0.0, 255.0);
  }

  // Green
  if (t <= 66.0) {
    g = 99.4708025861 * log(t) - 161.1195681661;
    g = clamp(g, 0.0, 255.0);
  } else {
    g = 288.1221695283 * pow(t - 60.0, -0.0755148492);
    g = clamp(g, 0.0, 255.0);
  }

  // Blue
  if (t >= 66.0) {
    b = 255.0;
  } else if (t <= 19.0) {
    b = 0.0;
  } else {
    b = 138.5177312231 * log(t - 10.0) - 305.0447927307;
    b = clamp(b, 0.0, 255.0);
  }

  return vec3<f32>(r, g, b) / 255.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }
  let idx = x + y * params.width;
  let px = input[idx];

  // Compute multiplier relative to D65 white point (6500K)
  let target_rgb = kelvin_to_rgb(params.temperature);
  let ref_rgb = kelvin_to_rgb(6500.0);
  var multiplier = ref_rgb / target_rgb;

  // Apply tint as green/magenta shift
  // Positive tint = more green, negative = more magenta (red+blue)
  let tint_factor = 1.0 + params.tint * 0.05;
  multiplier.g = multiplier.g * tint_factor;

  let balanced = clamp(px.rgb * multiplier, vec3<f32>(0.0), vec3<f32>(1.0));
  output[idx] = vec4<f32>(balanced, px.a);
}
