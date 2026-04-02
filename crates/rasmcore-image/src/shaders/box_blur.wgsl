// Box blur — separable H+V averaging filter
// Two dispatches: blur_h then blur_v with simple averaging.

struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256, 1, 1)
fn blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  var sum = vec4<f32>(0.0);
  let r = i32(params.radius);
  let diameter = f32(2u * params.radius + 1u);
  for (var i = -r; i <= r; i++) {
    let sx = clamp(i32(x) + i, 0, i32(params.width) - 1);
    sum += unpack(input[u32(sx) + y * params.width]);
  }
  output[x + y * params.width] = pack(sum / diameter);
}

@compute @workgroup_size(1, 256, 1)
fn blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  var sum = vec4<f32>(0.0);
  let r = i32(params.radius);
  let diameter = f32(2u * params.radius + 1u);
  for (var i = -r; i <= r; i++) {
    let sy = clamp(i32(y) + i, 0, i32(params.height) - 1);
    sum += unpack(input[x + u32(sy) * params.width]);
  }
  output[x + y * params.width] = pack(sum / diameter);
}
