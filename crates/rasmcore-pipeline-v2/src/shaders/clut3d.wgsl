// 3D Color LUT trilinear interpolation — GPU shader for FusedClutNode.
//
// The 3D LUT grid is passed as an extra read-only storage buffer.
// Grid layout: flattened array of grid_size^3 * 3 f32 values (R, G, B per grid point).
// Index formula: (b * grid_size * grid_size + g * grid_size + r) * 3

struct Params {
    width: u32,
    height: u32,
    grid_size: u32,
    _pad: u32,
}

@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> lut: array<f32>;

fn lut_sample(ri: u32, gi: u32, bi: u32, ch: u32) -> f32 {
    let gs = params.grid_size;
    let idx = (bi * gs * gs + gi * gs + ri) * 3u + ch;
    return lut[idx];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let idx = gid.y * params.width + gid.x;
    let p = load_pixel(idx);

    let n = f32(params.grid_size - 1u);
    let ri = clamp(p.x * n, 0.0, n);
    let gi = clamp(p.y * n, 0.0, n);
    let bi = clamp(p.z * n, 0.0, n);

    let r0 = u32(floor(ri));
    let g0 = u32(floor(gi));
    let b0 = u32(floor(bi));
    let r1 = min(r0 + 1u, params.grid_size - 1u);
    let g1 = min(g0 + 1u, params.grid_size - 1u);
    let b1 = min(b0 + 1u, params.grid_size - 1u);

    let fr = ri - f32(r0);
    let fg = gi - f32(g0);
    let fb = bi - f32(b0);

    // Trilinear interpolation per output channel
    var result: vec3<f32>;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let c000 = lut_sample(r0, g0, b0, ch);
        let c100 = lut_sample(r1, g0, b0, ch);
        let c010 = lut_sample(r0, g1, b0, ch);
        let c110 = lut_sample(r1, g1, b0, ch);
        let c001 = lut_sample(r0, g0, b1, ch);
        let c101 = lut_sample(r1, g0, b1, ch);
        let c011 = lut_sample(r0, g1, b1, ch);
        let c111 = lut_sample(r1, g1, b1, ch);

        let c00 = c000 + fr * (c100 - c000);
        let c01 = c001 + fr * (c101 - c001);
        let c10 = c010 + fr * (c110 - c010);
        let c11 = c011 + fr * (c111 - c011);

        let c0 = c00 + fg * (c10 - c00);
        let c1 = c01 + fg * (c11 - c01);

        result[ch] = c0 + fb * (c1 - c0);
    }

    store_pixel(idx, vec4(result, p.w));
}
