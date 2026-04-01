// Zhang-Suen thinning — one sub-iteration per dispatch.
// Params.sub_iter: 0 = first sub-iteration, 1 = second sub-iteration.
// Input/output are packed RGBA8 (only R channel used for binary).

struct Params {
  width: u32,
  height: u32,
  sub_iter: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_binary(x: i32, y: i32) -> u32 {
  if (x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.height)) {
    return 0u;
  }
  let pixel = input[u32(y) * params.width + u32(x)];
  let r = pixel & 0xFFu;
  return select(0u, 1u, r > 0u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);

  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let idx = gid.y * params.width + gid.x;
  let p1 = get_binary(x, y);

  // Pass through background pixels unchanged
  if (p1 == 0u) {
    output[idx] = input[idx];
    return;
  }

  // 8-connected neighbors (clockwise from north)
  let p2 = get_binary(x, y - 1);      // N
  let p3 = get_binary(x + 1, y - 1);  // NE
  let p4 = get_binary(x + 1, y);      // E
  let p5 = get_binary(x + 1, y + 1);  // SE
  let p6 = get_binary(x, y + 1);      // S
  let p7 = get_binary(x - 1, y + 1);  // SW
  let p8 = get_binary(x - 1, y);      // W
  let p9 = get_binary(x - 1, y - 1);  // NW

  // Count 0→1 transitions
  var transitions = 0u;
  transitions += select(0u, 1u, p2 == 0u && p3 == 1u);
  transitions += select(0u, 1u, p3 == 0u && p4 == 1u);
  transitions += select(0u, 1u, p4 == 0u && p5 == 1u);
  transitions += select(0u, 1u, p5 == 0u && p6 == 1u);
  transitions += select(0u, 1u, p6 == 0u && p7 == 1u);
  transitions += select(0u, 1u, p7 == 0u && p8 == 1u);
  transitions += select(0u, 1u, p8 == 0u && p9 == 1u);
  transitions += select(0u, 1u, p9 == 0u && p2 == 1u);

  // Count non-zero neighbors
  let neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

  var should_delete = transitions == 1u && neighbors >= 2u && neighbors <= 6u;

  if (params.sub_iter == 0u) {
    // Sub-iteration 1
    should_delete = should_delete && (p2 * p4 * p6 == 0u) && (p4 * p6 * p8 == 0u);
  } else {
    // Sub-iteration 2
    should_delete = should_delete && (p2 * p4 * p8 == 0u) && (p2 * p6 * p8 == 0u);
  }

  if (should_delete) {
    output[idx] = 0u; // Delete pixel (set to black)
  } else {
    output[idx] = input[idx]; // Keep pixel
  }
}
