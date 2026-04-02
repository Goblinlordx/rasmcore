// Mirror/kaleidoscope — angular segment reflection

struct Params {
  width: u32,
  height: u32,
  segments: u32,
  mode: u32,
  angle_offset: f32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  var src_x: u32;
  var src_y: u32;

  if (params.mode == 0u) {
    // Horizontal mirror
    let mid = params.width / 2u;
    if (x >= mid) {
      src_x = params.width - 1u - x;
    } else {
      src_x = x;
    }
    src_y = y;
  } else if (params.mode == 1u) {
    // Vertical mirror
    let mid = params.height / 2u;
    if (y >= mid) {
      src_y = params.height - 1u - y;
    } else {
      src_y = y;
    }
    src_x = x;
  } else {
    // Kaleidoscope
    let cx = f32(params.width) * 0.5;
    let cy = f32(params.height) * 0.5;
    let dx = f32(x) - cx;
    let dy = f32(y) - cy;

    var theta = atan2(dy, dx) - params.angle_offset;
    let radius = sqrt(dx * dx + dy * dy);

    // Normalize to [0, 2*PI)
    theta = theta - floor(theta / TWO_PI) * TWO_PI;

    let seg_angle = TWO_PI / f32(params.segments);
    let seg_idx = u32(theta / seg_angle);
    let seg_start = f32(seg_idx) * seg_angle;
    let local_angle = theta - seg_start;

    var mapped_angle: f32;
    if (seg_idx % 2u == 0u) {
      mapped_angle = local_angle + params.angle_offset;
    } else {
      mapped_angle = (seg_angle - local_angle) + params.angle_offset;
    }

    let sx = clamp(i32(cx + radius * cos(mapped_angle)), 0, i32(params.width) - 1);
    let sy = clamp(i32(cy + radius * sin(mapped_angle)), 0, i32(params.height) - 1);
    src_x = u32(sx);
    src_y = u32(sy);
  }

  output[y * params.width + x] = input[src_y * params.width + src_x];
}
