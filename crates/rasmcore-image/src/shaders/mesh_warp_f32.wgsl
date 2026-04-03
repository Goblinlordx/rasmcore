// Mesh warp — grid-based distortion with bilinear control point interpolation.
// Uses Newton's method to invert the destination grid mapping.
//
// Extra buffer @binding(3): control point grid as array<vec4<f32>>
//   Each vec4 = (src_x, src_y, dst_x, dst_y) in pixel coordinates.
//   Grid is (grid_cols × grid_rows) points, row-major.

struct Params {
  width: u32,
  height: u32,
  grid_cols: u32,
  grid_rows: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> grid: array<vec4<f32>>;

fn sample_bilinear_f32(fx: f32, fy: f32) -> vec4<f32> {
  let ix = i32(floor(fx));
  let iy = i32(floor(fy));
  let dx = fx - f32(ix);
  let dy = fy - f32(iy);
  let x0 = clamp(ix, 0, i32(params.width) - 1);
  let x1 = clamp(ix + 1, 0, i32(params.width) - 1);
  let y0 = clamp(iy, 0, i32(params.height) - 1);
  let y1 = clamp(iy + 1, 0, i32(params.height) - 1);
  let p00 = input[u32(x0) + u32(y0) * params.width];
  let p10 = input[u32(x1) + u32(y0) * params.width];
  let p01 = input[u32(x0) + u32(y1) * params.width];
  let p11 = input[u32(x1) + u32(y1) * params.width];
  return mix(mix(p00, p10, vec4<f32>(dx)), mix(p01, p11, vec4<f32>(dx)), vec4<f32>(dy));
}

fn cross2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
  return ax * by - ay * bx;
}

// Check if point (px, py) is inside quad (CW or CCW) using cross-product test.
fn point_in_quad(px: f32, py: f32, gi: u32) -> bool {
  let tl = grid[gi];
  let tr = grid[gi + 1u];
  let bl = grid[gi + params.grid_cols];
  let br = grid[gi + params.grid_cols + 1u];

  let d0 = cross2d(tr.z - tl.z, tr.w - tl.w, px - tl.z, py - tl.w);
  let d1 = cross2d(br.z - tr.z, br.w - tr.w, px - tr.z, py - tr.w);
  let d2 = cross2d(bl.z - br.z, bl.w - br.w, px - br.z, py - br.w);
  let d3 = cross2d(tl.z - bl.z, tl.w - bl.w, px - bl.z, py - bl.w);

  let all_pos = d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0;
  let all_neg = d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0;
  return all_pos || all_neg;
}

// Newton's method: invert bilinear mapping to find (u, v) in [0,1]^2
// such that bilinear(u,v) ≈ (px, py) in destination space.
// Then map (u, v) back to source coords via bilinear interpolation on source grid.
fn mesh_inverse(px: f32, py: f32, gi: u32) -> vec2<f32> {
  let tl = grid[gi];
  let tr = grid[gi + 1u];
  let bl = grid[gi + params.grid_cols];
  let br = grid[gi + params.grid_cols + 1u];

  var u: f32 = 0.5;
  var v: f32 = 0.5;

  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    let fx = (1.0 - v) * ((1.0 - u) * tl.z + u * tr.z) + v * ((1.0 - u) * bl.z + u * br.z);
    let fy = (1.0 - v) * ((1.0 - u) * tl.w + u * tr.w) + v * ((1.0 - u) * bl.w + u * br.w);
    let ex = px - fx;
    let ey = py - fy;

    if (ex * ex + ey * ey < 0.001) { break; }

    let dfx_du = (1.0 - v) * (tr.z - tl.z) + v * (br.z - bl.z);
    let dfx_dv = (1.0 - u) * (bl.z - tl.z) + u * (br.z - tr.z);
    let dfy_du = (1.0 - v) * (tr.w - tl.w) + v * (br.w - bl.w);
    let dfy_dv = (1.0 - u) * (bl.w - tl.w) + u * (br.w - tr.w);

    let det = dfx_du * dfy_dv - dfx_dv * dfy_du;
    if (abs(det) < 1e-10) { break; }
    let inv_det = 1.0 / det;

    u = clamp(u + (dfy_dv * ex - dfx_dv * ey) * inv_det, 0.0, 1.0);
    v = clamp(v + (dfx_du * ey - dfy_du * ex) * inv_det, 0.0, 1.0);
  }

  // Map (u, v) to source coordinates
  let src_x = (1.0 - v) * ((1.0 - u) * tl.x + u * tr.x) + v * ((1.0 - u) * bl.x + u * br.x);
  let src_y = (1.0 - v) * ((1.0 - u) * tl.y + u * tr.y) + v * ((1.0 - u) * bl.y + u * br.y);
  return vec2<f32>(src_x, src_y);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let px = f32(x);
  let py = f32(y);

  // Find containing quad in destination grid
  var found: bool = false;
  var src: vec2<f32> = vec2<f32>(px, py);

  for (var r: u32 = 0u; r < params.grid_rows - 1u; r = r + 1u) {
    for (var c: u32 = 0u; c < params.grid_cols - 1u; c = c + 1u) {
      let gi = r * params.grid_cols + c;
      if (point_in_quad(px, py, gi)) {
        src = mesh_inverse(px, py, gi);
        found = true;
        break;
      }
    }
    if (found) { break; }
  }

  output[x + y * params.width] = sample_bilinear_f32(src.x, src.y);
}
