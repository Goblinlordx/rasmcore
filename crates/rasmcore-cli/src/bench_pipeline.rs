//! bench-pipeline — V2 GPU vs CPU benchmark for N brightness nodes.
//!
//! Builds a V2 pipeline graph with a chain of brightness adjustments,
//! runs it with and without the GPU executor, and compares timings.
//!
//! Usage: bench-pipeline [--nodes N] [--width W] [--height H]
//!
//! Defaults: 7 nodes, 1920x1080

#[cfg(feature = "gpu")]
use rasmcore_gpu_native as gpu_executor_v2;

use rasmcore_pipeline_v2::color_space::ColorSpace;
use rasmcore_pipeline_v2::filter_node::FilterNode;
use rasmcore_pipeline_v2::filters::adjustment::Brightness;
use rasmcore_pipeline_v2::graph::Graph;
use rasmcore_pipeline_v2::node::NodeInfo;
use rasmcore_pipeline_v2::trace::TraceEventKind;
use std::rc::Rc;
use std::time::Instant;

/// Generate a smooth RGBA gradient image (f32, 4 channels per pixel).
fn gradient_image(width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut pixels = Vec::with_capacity(w * h * 4);
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

/// Source node that holds precomputed pixel data.
struct SourceNode {
    pixels: Vec<f32>,
    info: NodeInfo,
}

impl rasmcore_pipeline_v2::node::Node for SourceNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: rasmcore_pipeline_v2::Rect,
        _upstream: &mut dyn rasmcore_pipeline_v2::node::Upstream,
    ) -> Result<Vec<f32>, rasmcore_pipeline_v2::node::PipelineError> {
        // Full-image request — return all pixels
        let w = self.info.width as usize;
        let rw = request.width as usize;
        let rh = request.height as usize;
        let rx = request.x as usize;
        let ry = request.y as usize;

        if rx == 0 && ry == 0 && rw == w && rh == self.info.height as usize {
            return Ok(self.pixels.clone());
        }

        // Sub-region
        let mut out = Vec::with_capacity(rw * rh * 4);
        for row in 0..rh {
            let src = ((ry + row) * w + rx) * 4;
            out.extend_from_slice(&self.pixels[src..src + rw * 4]);
        }
        Ok(out)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![]
    }
}

fn build_graph(width: u32, height: u32, num_nodes: usize) -> (Graph, u32) {
    let mut graph = Graph::new(0); // no cache — force re-compute each time

    let pixels = gradient_image(width, height);
    let info = NodeInfo {
        width,
        height,
        color_space: ColorSpace::Linear,
    };

    let src = graph.add_node(Box::new(SourceNode {
        pixels,
        info: info.clone(),
    }));

    let mut prev = src;
    // Alternate small positive and negative brightness to avoid trivial fusion to identity
    for i in 0..num_nodes {
        let amount = if i % 2 == 0 { 0.02 } else { -0.02 };
        let node = FilterNode::new(prev, info.clone(), Brightness { amount });
        prev = graph.add_node(Box::new(node));
    }

    (graph, prev)
}

fn main() {
    let mut num_nodes = 7usize;
    let mut width = 1920u32;
    let mut height = 1080u32;

    // Simple arg parsing
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nodes" => {
                num_nodes = args[i + 1].parse().expect("invalid --nodes");
                i += 2;
            }
            "--width" => {
                width = args[i + 1].parse().expect("invalid --width");
                i += 2;
            }
            "--height" => {
                height = args[i + 1].parse().expect("invalid --height");
                i += 2;
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    let pixels = (width as u64) * (height as u64);
    println!("bench-pipeline: {}x{} ({} Mpx), {} brightness nodes", width, height, pixels / 1_000_000, num_nodes);
    println!();

    // ── CPU benchmark ──
    {
        let (mut graph, sink) = build_graph(width, height, num_nodes);
        graph.set_tracing(true);

        let t = Instant::now();
        let _result = graph.request_full(sink).unwrap();
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let trace = graph.take_trace();
        let fusion_us: u64 = trace.by_kind(TraceEventKind::Fusion).iter().map(|e| e.duration_us).sum();
        let cpu_us: u64 = trace.by_kind(TraceEventKind::CpuFallback).iter().map(|e| e.duration_us).sum();

        println!("CPU path:");
        println!("  Total:    {cpu_ms:.1} ms");
        println!("  Fusion:   {:.1} ms", fusion_us as f64 / 1000.0);
        println!("  Compute:  {:.1} ms", cpu_us as f64 / 1000.0);
        println!("  Events:   {} fusion, {} cpu_fallback",
            trace.by_kind(TraceEventKind::Fusion).len(),
            trace.by_kind(TraceEventKind::CpuFallback).len());
        println!();
    }

    // ── GPU benchmark ──
    #[cfg(feature = "gpu")]
    {
        match gpu_executor_v2::WgpuExecutorV2::try_new() {
            Ok(executor) => {
                println!("GPU adapter: {}", executor.adapter_name());

                // Cold start (includes shader compile)
                let (mut graph, sink) = build_graph(width, height, num_nodes);
                graph.set_gpu_executor(Rc::new(executor));
                graph.set_tracing(true);

                let t = Instant::now();
                let _result = graph.request_full(sink).unwrap();
                let cold_ms = t.elapsed().as_secs_f64() * 1000.0;

                let trace = graph.take_trace();
                let fusion_us: u64 = trace.by_kind(TraceEventKind::Fusion).iter().map(|e| e.duration_us).sum();
                let gpu_us: u64 = trace.by_kind(TraceEventKind::GpuDispatch).iter().map(|e| e.duration_us).sum();
                let cpu_us: u64 = trace.by_kind(TraceEventKind::CpuFallback).iter().map(|e| e.duration_us).sum();

                println!("GPU path (cold):");
                println!("  Total:    {cold_ms:.1} ms");
                println!("  Fusion:   {:.1} ms", fusion_us as f64 / 1000.0);
                println!("  GPU:      {:.1} ms", gpu_us as f64 / 1000.0);
                if cpu_us > 0 {
                    println!("  CPU fb:   {:.1} ms", cpu_us as f64 / 1000.0);
                }
                println!("  Events:   {} fusion, {} gpu_dispatch, {} cpu_fallback",
                    trace.by_kind(TraceEventKind::Fusion).len(),
                    trace.by_kind(TraceEventKind::GpuDispatch).len(),
                    trace.by_kind(TraceEventKind::CpuFallback).len());
                println!();

                // Warm run (shader cached, device warm)
                {
                    // Rebuild graph to reset cache and optimized flag
                    let (mut graph, sink) = build_graph(width, height, num_nodes);
                    // Reuse same executor (shader cache warm)
                    // Need a new executor since the old one was moved — but Rc lets us share
                    // Actually, the executor was moved into the first graph. Let's create fresh.
                    match gpu_executor_v2::WgpuExecutorV2::try_new() {
                        Ok(exec2) => {
                            graph.set_gpu_executor(Rc::new(exec2));
                            graph.set_tracing(true);

                            // Warmup
                            let _ = graph.request_full(sink).unwrap();
                            graph.clear_cache();
                            let _ = graph.take_trace();

                            // Timed run (graph is already optimized, shader compiled)
                            let t = Instant::now();
                            let _result = graph.request_full(sink).unwrap();
                            let warm_ms = t.elapsed().as_secs_f64() * 1000.0;

                            let trace = graph.take_trace();
                            let gpu_us: u64 = trace.by_kind(TraceEventKind::GpuDispatch).iter().map(|e| e.duration_us).sum();

                            println!("GPU path (warm):");
                            println!("  Total:    {warm_ms:.1} ms");
                            println!("  GPU:      {:.1} ms", gpu_us as f64 / 1000.0);
                            println!("  Events:   {} gpu_dispatch, {} cpu_fallback",
                                trace.by_kind(TraceEventKind::GpuDispatch).len(),
                                trace.by_kind(TraceEventKind::CpuFallback).len());
                        }
                        Err(e) => println!("  Warm run skipped: {e}"),
                    }
                }
            }
            Err(e) => {
                println!("GPU not available: {e}");
                println!("(only CPU results shown)");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled — only CPU results shown");
    }
}
