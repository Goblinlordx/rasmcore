//! name: invert_gpu
//! category: adjustment
//! strategy: gpu

// GPU strategy invert: provides WGSL shader for GPU execution and
// a compute() CPU fallback. Self-inverse: applying twice returns
// the original image.

fn gpu_shader() {
    `@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = dims.x * dims.y;
    if (idx >= total) { return; }
    let p = input[idx];
    let r = 255u - (p & 0xFFu);
    let g = 255u - ((p >> 8u) & 0xFFu);
    let b = 255u - ((p >> 16u) & 0xFFu);
    let a = (p >> 24u) & 0xFFu;
    output[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
}`
}

fn compute(pixels, info, params) {
    let out = pixels;
    for i in range(0, out.len()) {
        out[i] = 255 - out[i];
    }
    out
}
