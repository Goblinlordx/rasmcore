//! GPU shader correctness tests via native wgpu executor.
//!
//! Tests run WGSL shaders on real GPU hardware and compare against CPU reference.
//! Skip gracefully on machines without GPU (CI without GPU sees "skipping").

use std::rc::Rc;
use rasmcore_pipeline_v2::gpu::GpuExecutor;
use rasmcore_pipeline_v2::node::GpuShader;
use rasmcore_pipeline_v2::ops::Filter;

// ─── GPU availability helper ─────────────────────────────────────────────────

fn try_gpu() -> Option<Rc<dyn GpuExecutor>> {
    match rasmcore_gpu_native::WgpuExecutorV2::try_new() {
        Ok(e) => Some(Rc::new(e)),
        Err(e) => {
            eprintln!("No GPU available — skipping: {e}");
            None
        }
    }
}

macro_rules! require_gpu {
    () => {
        match try_gpu() {
            Some(gpu) => gpu,
            None => return,
        }
    };
}

// ─── Test image helpers ──────────────────────────────────────────────────────

/// Generate a gradient RGBA f32 image (4 channels per pixel).
fn gradient_image(w: u32, h: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = x as f32 / (w - 1).max(1) as f32;
            let g = y as f32 / (h - 1).max(1) as f32;
            let b = ((x + y) as f32 / (w + h - 2).max(1) as f32).min(1.0);
            pixels.extend_from_slice(&[r, g, b, 1.0]);
        }
    }
    pixels
}

/// Assert two pixel arrays are close within tolerance per channel.
fn assert_pixels_close(gpu: &[f32], cpu: &[f32], tolerance: f32, label: &str) {
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "{label}: length mismatch gpu={} cpu={}",
        gpu.len(),
        cpu.len()
    );
    let mut max_err: f32 = 0.0;
    let mut max_idx = 0;
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let err = (g - c).abs();
        if err > max_err {
            max_err = err;
            max_idx = i;
        }
    }
    assert!(
        max_err <= tolerance,
        "{label}: max error {max_err:.6} at index {max_idx} (gpu={:.6}, cpu={:.6}) exceeds tolerance {tolerance}",
        gpu[max_idx],
        cpu[max_idx],
    );
}

// ─── Filter GPU/CPU parity helper ────────────────────────────────────────────

fn test_filter_parity<F: Filter>(
    gpu: &Rc<dyn GpuExecutor>,
    filter: &F,
    w: u32,
    h: u32,
    tolerance: f32,
    name: &str,
) {
    let input = gradient_image(w, h);

    // CPU reference
    let cpu_out = filter.compute(&input, w, h).expect("CPU compute failed");

    // GPU path: get shader, execute via executor
    let shader = filter.gpu_shader_body();
    if shader.is_none() {
        eprintln!("  {name}: no GPU shader — skipping");
        return;
    }

    let params = filter.gpu_params(w, h).unwrap_or_default();
    let extra = filter.gpu_extra_buffers();

    let composed = rasmcore_pipeline_v2::filter_node::compose_shader(shader.unwrap());
    let ops = vec![GpuShader {
        body: composed,
        entry_point: filter.gpu_entry_point(),
        workgroup_size: filter.gpu_workgroup_size(),
        params,
        extra_buffers: extra,
        reduction_buffers: vec![],
        convergence_check: None,
            loop_dispatch: None,
    }];

    let gpu_out = gpu
        .execute(&ops, &input, w, h)
        .expect("GPU execute failed");

    assert_pixels_close(&gpu_out, &cpu_out, tolerance, name);
    eprintln!("  {name}: PASS (max error within {tolerance})");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Filter Parity Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn adjustment_filters_gpu_parity() {
    let gpu = require_gpu!();
    use rasmcore_pipeline_v2::filters::adjustment::*;

    let (w, h) = (32, 32);
    let tol = 1e-4;

    test_filter_parity(&gpu, &Brightness { amount: 0.3 }, w, h, tol, "brightness");
    test_filter_parity(&gpu, &Contrast { amount: 0.5 }, w, h, tol, "contrast");
    test_filter_parity(&gpu, &Gamma { gamma: 2.2 }, w, h, tol, "gamma");
    test_filter_parity(&gpu, &Invert, w, h, tol, "invert");
    test_filter_parity(
        &gpu,
        &Levels { black: 0.1, white: 0.9, gamma: 1.5 },
        w, h, tol, "levels",
    );
}

#[test]
fn color_filters_gpu_parity() {
    let gpu = require_gpu!();
    use rasmcore_pipeline_v2::filters::color::*;

    let (w, h) = (32, 32);
    let tol = 1e-3; // color transforms have slightly more error

    test_filter_parity(&gpu, &HueRotate { degrees: 90.0 }, w, h, tol, "hue_rotate");
    test_filter_parity(&gpu, &Saturate { factor: 1.5 }, w, h, tol, "saturate");
    test_filter_parity(&gpu, &Sepia { intensity: 0.8 }, w, h, tol, "sepia");
    test_filter_parity(&gpu, &Vibrance { amount: 0.5 }, w, h, tol, "vibrance");
    test_filter_parity(
        &gpu,
        &Colorize { target_r: 0.2, target_g: 0.4, target_b: 0.8, amount: 0.5 },
        w, h, tol, "colorize",
    );
}

#[test]
fn effect_filters_gpu_parity() {
    let gpu = require_gpu!();
    use rasmcore_pipeline_v2::filters::effect::*;

    let (w, h) = (32, 32);
    let tol = 1e-3;

    test_filter_parity(&gpu, &Pixelate { block_size: 4 }, w, h, tol, "pixelate");
    test_filter_parity(&gpu, &Emboss, w, h, tol, "emboss");
}

#[test]
fn grading_filters_gpu_parity() {
    let gpu = require_gpu!();
    use rasmcore_pipeline_v2::filters::grading::*;

    let (w, h) = (32, 32);
    let tol = 1e-3;

    test_filter_parity(
        &gpu,
        &TonemapReinhard,
        w, h, tol, "tonemap_reinhard",
    );
    test_filter_parity(
        &gpu,
        &TonemapFilmic { a: 2.51, b: 0.03, c: 2.43, d: 0.59, e: 0.14 },
        w, h, tol, "tonemap_filmic",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Fused Shader Tests (via Graph)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn fused_brightness_contrast_gpu_parity() {
    let gpu = require_gpu!();
    use rasmcore_pipeline_v2::filters::adjustment::*;
    use rasmcore_pipeline_v2::graph::Graph;
    use rasmcore_pipeline_v2::rect::Rect;
    use rasmcore_pipeline_v2::node::{Node, NodeInfo};
    use rasmcore_pipeline_v2::color_space::ColorSpace;

    let (w, h) = (32, 32);
    let input = gradient_image(w, h);

    // CPU reference: brightness then contrast
    let bright = Brightness { amount: 0.1 };
    let contrast = Contrast { amount: 0.5 };
    let step1 = bright.compute(&input, w, h).unwrap();
    let cpu_out = contrast.compute(&step1, w, h).unwrap();

    // GPU path: via graph with executor
    struct TestSource { pixels: Vec<f32>, w: u32, h: u32 }
    impl Node for TestSource {
        fn info(&self) -> NodeInfo {
            NodeInfo { width: self.w, height: self.h, color_space: ColorSpace::Linear }
        }
        fn compute(&self, _req: Rect, _up: &mut dyn rasmcore_pipeline_v2::node::Upstream) -> Result<Vec<f32>, rasmcore_pipeline_v2::node::PipelineError> {
            Ok(self.pixels.clone())
        }
        fn upstream_ids(&self) -> Vec<u32> { vec![] }
    }

    let mut graph = Graph::new(16 * 1024 * 1024);
    graph.set_gpu_executor(gpu);
    let src = graph.add_node(Box::new(TestSource { pixels: input, w, h }));
    let info = graph.node_info(src).unwrap();
    let b_node = rasmcore_pipeline_v2::registry::create_filter_node(
        "brightness", src, info.clone(),
        &{
            let mut m = rasmcore_pipeline_v2::registry::ParamMap::new();
            m.floats.insert("amount".into(), 0.1);
            m
        },
    ).unwrap();
    let b_id = graph.add_node(b_node);
    let c_node = rasmcore_pipeline_v2::registry::create_filter_node(
        "contrast", b_id, info,
        &{
            let mut m = rasmcore_pipeline_v2::registry::ParamMap::new();
            m.floats.insert("amount".into(), 0.5);
            m
        },
    ).unwrap();
    let c_id = graph.add_node(c_node);

    let gpu_out = graph.request_full(c_id).unwrap();

    assert_pixels_close(&gpu_out, &cpu_out, 1e-3, "fused brightness+contrast");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Passthrough (empty shader list) Test
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn passthrough_no_shaders_returns_input() {
    let gpu = require_gpu!();

    let input = gradient_image(8, 8);
    let result = gpu.execute(&[], &input, 8, 8).unwrap();
    assert_eq!(result, input, "empty shader list should return input unchanged");
}
