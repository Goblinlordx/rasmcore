// Swirl — rotational distortion that decreases with distance from center

struct Params {
  width: u32,
  height: u32,
  angle: f32,
  radius: f32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let cx = f32(params.width) * 0.5;
  let cy = f32(params.height) * 0.5;
  let dx = f32(x) - cx;
  let dy = f32(y) - cy;
  let dist = sqrt(dx * dx + dy * dy);

  var sx: f32;
  var sy: f32;
  if (dist < params.radius && params.radius > 0.0) {
    // Swirl angle decreases linearly with distance from center
    let t = 1.0 - dist / params.radius;
    let swirl_angle = params.angle * t * t;
    let ct = cos(swirl_angle);
    let st = sin(swirl_angle);
    sx = dx * ct - dy * st + cx;
    sy = dx * st + dy * ct + cy;
  } else {
    sx = f32(x);
    sy = f32(y);
  }

  output[x + y * params.width] = pack(sample_bilinear(sx, sy));
}
