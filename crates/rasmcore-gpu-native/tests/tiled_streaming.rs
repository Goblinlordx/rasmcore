//! GPU tiled streaming PoC — validates reduction buffer accumulation across tiles.
//!
//! These tests require a GPU. They skip gracefully if no adapter is found.

use rasmcore_gpu_native::WgpuExecutorV2;
use rasmcore_pipeline_v2::node::{GpuShader, ReductionBuffer};
use rasmcore_pipeline_v2::rect::{tiles, extract_tile, place_tile};

/// Try to create a GPU executor. Skip test if no GPU.
fn try_gpu() -> Option<WgpuExecutorV2> {
    WgpuExecutorV2::try_new().ok()
}

/// Generate a deterministic 256x256 test image with varying pixel values.
/// Each pixel RGBA = (r, g, b, 1.0) where r/g/b depend on position.
fn make_test_image(w: u32, h: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = x as f32 / w as f32;
            let g = y as f32 / h as f32;
            let b = ((x + y) as f32 / (w + h) as f32).min(1.0);
            pixels.extend_from_slice(&[r, g, b, 1.0]);
        }
    }
    pixels
}

/// CPU reference: compute per-channel sum of an f32 RGBA image.
fn cpu_channel_sum(pixels: &[f32]) -> [f64; 4] {
    let mut sum = [0.0f64; 4];
    for px in pixels.chunks(4) {
        sum[0] += px[0] as f64;
        sum[1] += px[1] as f64;
        sum[2] += px[2] as f64;
        sum[3] += px[3] as f64;
    }
    sum
}

/// A simple GPU shader that passes pixels through while atomically accumulating
/// per-channel sums into a reduction buffer.
///
/// Reduction buffer layout: 4 x u32 (as f32 bit-cast).
/// This uses f32 addition without atomics — NOT production quality,
/// but sufficient for PoC verification with small workgroups.
///
/// For the PoC, we use a dead-simple approach: each invocation adds its pixel
/// to the reduction buffer. This works because we use workgroup_size=1 to avoid
/// race conditions within workgroups (serializing all work).
/// Workgroup size for the sum shader — 8x8 = 64 threads per workgroup.
const SUM_WG: [u32; 3] = [8, 8, 1];

fn sum_shader_body() -> String {
    // 2D dispatch shader: each thread handles one pixel.
    // Workgroup-shared memory for local partial sums.
    // Thread 0 of each workgroup writes to the global reduction buffer.
    r#"
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@group(0) @binding(3) var<storage, read_write> accum: array<f32>;

var<workgroup> local_sum: array<f32, 4>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let width = params.x;
    let height = params.y;
    let idx = gid.y * width + gid.x;
    let total = width * height;

    // Initialize shared memory
    if (lid == 0u) {
        local_sum[0] = 0.0;
        local_sum[1] = 0.0;
        local_sum[2] = 0.0;
        local_sum[3] = 0.0;
    }
    workgroupBarrier();

    // Sequential accumulation — each thread contributes one at a time
    for (var i = 0u; i < 64u; i++) {
        if (lid == i && gid.x < width && gid.y < height) {
            let px = input[idx];
            output[idx] = px;
            local_sum[0] += px.x;
            local_sum[1] += px.y;
            local_sum[2] += px.z;
            local_sum[3] += px.w;
        }
        workgroupBarrier();
    }

    // Thread 0 writes workgroup partial sum to global reduction buffer
    // Flat workgroup index: wid.y * nwg.x + wid.x
    if (lid == 0u) {
        let wg_flat = wid.y * nwg.x + wid.x;
        accum[wg_flat * 4u + 0u] = local_sum[0];
        accum[wg_flat * 4u + 1u] = local_sum[1];
        accum[wg_flat * 4u + 2u] = local_sum[2];
        accum[wg_flat * 4u + 3u] = local_sum[3];
    }
}
"#.to_string()
}

/// Calculate number of workgroups for a given image size with the sum shader.
fn sum_num_workgroups(w: u32, h: u32) -> u32 {
    let wg_x = (w + SUM_WG[0] - 1) / SUM_WG[0];
    let wg_y = (h + SUM_WG[1] - 1) / SUM_WG[1];
    wg_x * wg_y
}

#[test]
fn gpu_reduction_buffer_single_dispatch() {
    let gpu = match try_gpu() {
        Some(g) => g,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    let w = 64u32;
    let h = 64u32;
    let pixels = make_test_image(w, h);
    let cpu_sum = cpu_channel_sum(&pixels);

    let num_wg = sum_num_workgroups(w, h);
    let buf_size = num_wg as usize * 4 * 4; // 4 floats per workgroup

    let params = [w, h, 0u32, 0u32];
    let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();

    let ops = vec![GpuShader::new(
        sum_shader_body(),
        "main",
        SUM_WG,
        params_bytes,
    )
    .with_reduction_buffers(vec![ReductionBuffer {
        id: 0,
        initial_data: vec![0u8; buf_size],
        read_write: true,
    }])];

    let (output, reduction_data) = gpu
        .execute_with_reduction_readback(&ops, &pixels, w, h)
        .expect("GPU execute failed");

    // Output should be passthrough
    assert_eq!(output.len(), pixels.len());

    // Sum the partial workgroup sums from the reduction buffer
    assert_eq!(reduction_data.len(), 1);
    let (id, buf_bytes) = &reduction_data[0];
    assert_eq!(*id, 0);
    let reduction_floats: &[f32] = bytemuck::cast_slice(buf_bytes);

    // Debug: show first few workgroup sums
    eprintln!("reduction buffer: {} floats ({} bytes)", reduction_floats.len(), buf_bytes.len());
    for wg in 0..4.min(num_wg as usize) {
        eprintln!("  wg {wg}: r={:.4} g={:.4} b={:.4} a={:.4}",
            reduction_floats[wg * 4], reduction_floats[wg * 4 + 1],
            reduction_floats[wg * 4 + 2], reduction_floats[wg * 4 + 3]);
    }

    let mut gpu_sum = [0.0f64; 4];
    for wg in 0..num_wg as usize {
        gpu_sum[0] += reduction_floats[wg * 4] as f64;
        gpu_sum[1] += reduction_floats[wg * 4 + 1] as f64;
        gpu_sum[2] += reduction_floats[wg * 4 + 2] as f64;
        gpu_sum[3] += reduction_floats[wg * 4 + 3] as f64;
    }

    // Compare with tolerance (f32 accumulation loses precision)
    let tol = 1.0; // generous for f32 accumulation of 4096 values
    for ch in 0..4 {
        assert!(
            (gpu_sum[ch] - cpu_sum[ch]).abs() < tol,
            "channel {ch}: GPU sum {:.4} vs CPU sum {:.4} (diff {:.6})",
            gpu_sum[ch],
            cpu_sum[ch],
            (gpu_sum[ch] - cpu_sum[ch]).abs()
        );
    }
}

#[test]
fn tiled_passthrough_matches_full() {
    // Validate that processing tiles separately produces identical output
    // to processing the full image (for a passthrough shader).
    let gpu = match try_gpu() {
        Some(g) => g,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    let w = 256u32;
    let h = 256u32;
    let pixels = make_test_image(w, h);

    // Full-image passthrough
    let full_output = gpu
        .execute_with_reduction_readback(&[], &pixels, w, h)
        .expect("full execute failed")
        .0;

    // Tiled passthrough (64x64 tiles)
    let tile_rects = tiles(w, h, 64, 64);
    assert_eq!(tile_rects.len(), 16);

    let mut tiled_output = vec![0.0f32; (w * h * 4) as usize];
    for tile in &tile_rects {
        let tile_pixels = extract_tile(&pixels, w, *tile);
        let tile_result = gpu
            .execute_with_reduction_readback(&[], &tile_pixels, tile.width, tile.height)
            .expect("tile execute failed")
            .0;
        place_tile(&mut tiled_output, w, *tile, &tile_result);
    }

    assert_eq!(full_output, tiled_output);
}

#[test]
fn tiled_reduction_accumulates_correctly() {
    // The key PoC test: dispatch a reduction shader on multiple tiles and verify
    // the partial sums can be combined to match the full-image result.
    let gpu = match try_gpu() {
        Some(g) => g,
        None => {
            eprintln!("SKIP: no GPU available");
            return;
        }
    };

    let w = 128u32;
    let h = 128u32;
    let pixels = make_test_image(w, h);
    let cpu_sum = cpu_channel_sum(&pixels);

    let tile_rects = tiles(w, h, 64, 64);
    assert_eq!(tile_rects.len(), 4);

    // Process each tile, collect per-tile reduction partials
    let mut all_partials = Vec::new();
    for tile in &tile_rects {
        let tile_pixels = extract_tile(&pixels, w, *tile);
        let tile_wg = sum_num_workgroups(tile.width, tile.height);
        let buf_size = tile_wg as usize * 4 * 4;

        let params = [tile.width, tile.height, 0u32, 0u32];
        let params_bytes: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();

        let ops = vec![GpuShader::new(
            sum_shader_body(),
            "main",
            SUM_WG,
            params_bytes,
        )
        .with_reduction_buffers(vec![ReductionBuffer {
            id: 0,
            initial_data: vec![0u8; buf_size],
            read_write: true,
        }])];

        let (_, reduction_data) = gpu
            .execute_with_reduction_readback(&ops, &tile_pixels, tile.width, tile.height)
            .expect("tile execute failed");

        assert!(!reduction_data.is_empty());
        let buf_bytes = &reduction_data[0].1;
        let floats: &[f32] = bytemuck::cast_slice(buf_bytes);

        // Sum this tile's workgroup partials
        let mut tile_sum = [0.0f64; 4];
        for wg in 0..(floats.len() / 4) {
            tile_sum[0] += floats[wg * 4] as f64;
            tile_sum[1] += floats[wg * 4 + 1] as f64;
            tile_sum[2] += floats[wg * 4 + 2] as f64;
            tile_sum[3] += floats[wg * 4 + 3] as f64;
        }
        all_partials.push(tile_sum);
    }

    // Combine all tile partial sums
    let mut combined = [0.0f64; 4];
    for partial in &all_partials {
        for ch in 0..4 {
            combined[ch] += partial[ch];
        }
    }

    // Compare with CPU reference
    let tol = 2.0; // generous for f32 accumulation
    for ch in 0..4 {
        assert!(
            (combined[ch] - cpu_sum[ch]).abs() < tol,
            "channel {ch}: combined tiled sum {:.4} vs CPU sum {:.4} (diff {:.6})",
            combined[ch],
            cpu_sum[ch],
            (combined[ch] - cpu_sum[ch]).abs()
        );
    }
}
