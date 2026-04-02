// Gaussian blur f32 — separable H+V with precomputed kernel weights
// Uses array<vec4<f32>> storage buffers for full-precision inter-pass data.
// Two dispatches: blur_h (horizontal, 256x1) then blur_v (vertical, 1x256)
// Kernel weights are passed as extra storage buffer.

struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  var sum = vec4<f32>(0.0);
  let r = i32(params.radius);
  for (var i = -r; i <= r; i++) {
    let sx = clamp(i32(x) + i, 0, i32(params.width) - 1);
    sum += unpack_f32(input[u32(sx) + y * params.width]) * kernel[u32(i + r)];
  }
  output[x + y * params.width] = pack_f32(sum);
}

@compute @workgroup_size(1, 256, 1)
fn blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  var sum = vec4<f32>(0.0);
  let r = i32(params.radius);
  for (var i = -r; i <= r; i++) {
    let sy = clamp(i32(y) + i, 0, i32(params.height) - 1);
    sum += unpack_f32(input[x + u32(sy) * params.width]) * kernel[u32(i + r)];
  }
  output[x + y * params.width] = pack_f32(sum);
}
