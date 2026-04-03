// f32 variant
// Spin blur — per-pixel radial sampling along circular arc
// Each pixel samples N taps along an arc centered at (center_x, center_y)

struct Params {
  width: u32,
  height: u32,
  center_x: f32,
  center_y: f32,
  angle: f32,
  samples: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let cx = params.center_x * f32(params.width);
  let cy = params.center_y * f32(params.height);
  let px = f32(x) - cx;
  let py = f32(y) - cy;

  var sum = vec4<f32>(0.0);
  let n = f32(params.samples);

  for (var i = 0u; i < params.samples; i++) {
    let t = (f32(i) / n - 0.5) * params.angle;
    let ct = cos(t);
    let st = sin(t);
    let rx = px * ct - py * st + cx;
    let ry = px * st + py * ct + cy;
    sum += sample_bilinear_f32(rx, ry);
  }

  output[x + y * params.width] = sum / n;
}
