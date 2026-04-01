// Gaussian blur — separable H+V with precomputed kernel weights
// Two dispatches: blur_h (horizontal, 256x1) then blur_v (vertical, 1x256)
// Kernel weights are passed as extra storage buffer.

struct Params {
  width: u32,
  height: u32,
  radius: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

fn unpack(pixel: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(pixel & 0xFFu),
    f32((pixel >> 8u) & 0xFFu),
    f32((pixel >> 16u) & 0xFFu),
    f32((pixel >> 24u) & 0xFFu),
  );
}

fn pack(color: vec4<f32>) -> u32 {
  let r = u32(clamp(color.x, 0.0, 255.0));
  let g = u32(clamp(color.y, 0.0, 255.0));
  let b = u32(clamp(color.z, 0.0, 255.0));
  let a = u32(clamp(color.w, 0.0, 255.0));
  return r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(256, 1, 1)
fn blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  var sum = vec4<f32>(0.0);
  let r = i32(params.radius);
  for (var i = -r; i <= r; i++) {
    let sx = clamp(i32(x) + i, 0, i32(params.width) - 1);
    sum += unpack(input[u32(sx) + y * params.width]) * kernel[u32(i + r)];
  }
  output[x + y * params.width] = pack(sum);
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
    sum += unpack(input[x + u32(sy) * params.width]) * kernel[u32(i + r)];
  }
  output[x + y * params.width] = pack(sum);
}
