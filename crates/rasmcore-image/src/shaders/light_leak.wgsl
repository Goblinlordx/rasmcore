// Light leak — procedural warm gradient overlay via screen blend

struct Params {
  width: u32,
  height: u32,
  intensity: f32,
  position_x: f32,
  position_y: f32,
  radius: f32,
  leak_r: f32,
  leak_g: f32,
  leak_b: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let idx = y * params.width + x;
  let pixel = unpack(input[idx]);

  let cx = params.position_x * f32(params.width);
  let cy = params.position_y * f32(params.height);
  let diag = sqrt(f32(params.width * params.width + params.height * params.height));
  let leak_radius = params.radius * diag;
  let inv_radius = 1.0 / max(leak_radius, 1.0);

  let dx = f32(x) - cx;
  let dy = f32(y) - cy;
  let dist = sqrt(dx * dx + dy * dy);
  let t = clamp(1.0 - dist * inv_radius, 0.0, 1.0);
  let leak_strength = t * t * params.intensity;

  // Screen blend: 1 - (1-base)(1-overlay)
  let or = 1.0 - (1.0 - pixel.r) * (1.0 - params.leak_r * leak_strength);
  let og = 1.0 - (1.0 - pixel.g) * (1.0 - params.leak_g * leak_strength);
  let ob = 1.0 - (1.0 - pixel.b) * (1.0 - params.leak_b * leak_strength);

  output[idx] = pack(vec4<f32>(or, og, ob, pixel.a));
}
