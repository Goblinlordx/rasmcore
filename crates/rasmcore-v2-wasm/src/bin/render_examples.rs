//! Render before/after example images for documentation.
//!
//! For each registered V2 filter, applies it with showcase params
//! to a reference image and writes the result as a PNG.
//!
//! Usage: cargo run --bin render_examples -p rasmcore-v2-wasm
//! Output: /tmp/docs-examples/{name}-after.png

// Force linker to include filter and codec registrations
#[allow(unused_imports)]
use rasmcore_pipeline_v2::filters as _f;
#[allow(unused_imports)]
use rasmcore_codecs_v2 as _c;

use rasmcore_pipeline_v2 as v2;
use rasmcore_pipeline_v2::graph::Graph;
use rasmcore_pipeline_v2::node::{NodeInfo, PipelineError};
use rasmcore_pipeline_v2::registry::ParamMap;
use std::fs;
use std::path::Path;

/// Source node that holds f32 pixel data (same as WASM adapter).
struct SourceNode {
    pixels: Vec<f32>,
    info: NodeInfo,
}

impl v2::Node for SourceNode {
    fn info(&self) -> NodeInfo { self.info.clone() }
    fn compute(
        &self,
        _request: v2::Rect,
        _upstream: &mut dyn v2::Upstream,
    ) -> Result<Vec<f32>, PipelineError> {
        Ok(self.pixels.clone())
    }
    fn upstream_ids(&self) -> Vec<u32> { vec![] }
}

/// Generate a 400x300 gradient test image (f32 linear RGBA).
fn generate_reference_image() -> (Vec<f32>, u32, u32) {
    let w = 400u32;
    let h = 300u32;
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);

    for y in 0..h {
        let fy = y as f32 / (h - 1) as f32;
        for x in 0..w {
            let fx = x as f32 / (w - 1) as f32;
            // Color gradient: red→green horizontal, blue vertical
            let r = fx * (1.0 - fy * 0.3);
            let g = (1.0 - (fx - 0.5).abs() * 2.0).max(0.0) * (1.0 - fy * 0.5);
            let b = fy * (1.0 - fx * 0.3);
            pixels.extend_from_slice(&[r, g, b, 1.0]);
        }
    }
    (pixels, w, h)
}

/// Encode f32 linear pixels to PNG (sRGB u8).
fn encode_png(pixels: &[f32], width: u32, height: u32) -> Vec<u8> {
    rasmcore_codecs_v2::encode(pixels, width, height, "png", None)
        .expect("PNG encode failed")
}

/// Get showcase param values that produce a visible effect.
fn showcase_params(name: &str, params: &[v2::ParamDescriptor]) -> Option<ParamMap> {
    let mut map = ParamMap::new();

    for p in params {
        let value: f64 = match (name, p.name) {
            ("brightness", "amount") => 0.15,
            ("contrast", "amount") => 0.4,
            ("gamma", "gamma") => 1.5,
            ("exposure", "ev") => 1.0,
            ("saturation", _) => 0.5,
            ("vibrance", _) => 0.5,
            ("posterize", "levels") => 6.0,
            ("sigmoidal_contrast", "contrast") => 5.0,
            ("sigmoidal_contrast", "midpoint") => 0.5,
            ("gaussian_blur", "radius") => 3.0,
            ("box_blur", "radius") => 3.0,
            ("sharpen", "amount") => 1.5,
            ("unsharp_mask", "amount") => 1.5,
            ("unsharp_mask", "radius") => 2.0,
            ("hue_rotate", _) => 90.0,
            ("sepia", _) => 0.8,
            ("vignette", "sigma") => 0.4,
            _ => {
                if let Some(def) = p.default {
                    if def.abs() < 1e-6 {
                        if let (Some(min), Some(max)) = (p.min, p.max) {
                            min + (max - min) * 0.3
                        } else { 0.3 }
                    } else { def }
                } else { 0.5 }
            }
        };

        match p.value_type {
            v2::ParamType::F32 | v2::ParamType::F64 => {
                map.floats.insert(p.name.to_string(), value as f32);
            }
            v2::ParamType::U32 => {
                map.ints.insert(p.name.to_string(), value as i64);
            }
            v2::ParamType::I32 => {
                map.ints.insert(p.name.to_string(), value as i64);
            }
            v2::ParamType::Bool => {
                map.bools.insert(p.name.to_string(), value > 0.5);
            }
            _ => {}
        }
    }

    Some(map)
}

fn main() {
    let out_dir = Path::new("/tmp/docs-examples");
    fs::create_dir_all(out_dir).expect("create output dir");

    let (ref_pixels, w, h) = generate_reference_image();
    let ref_png = encode_png(&ref_pixels, w, h);
    fs::write(out_dir.join("reference.png"), &ref_png).expect("write reference");

    let info = NodeInfo {
        width: w,
        height: h,
        color_space: v2::ColorSpace::Linear,
    };

    let filters = v2::registry::registered_filter_registrations();
    let mut rendered = 0;
    let mut skipped = 0;

    for f in &filters {
        let params = match showcase_params(f.name, f.params) {
            Some(p) => p,
            None => { skipped += 1; continue; }
        };

        let source = Box::new(SourceNode { pixels: ref_pixels.clone(), info: info.clone() });
        let mut graph = Graph::new(16 * 1024 * 1024);
        let src_id = graph.add_node(source);

        let filter_node = (f.factory)(src_id, info.clone(), &params);
        let filter_id = graph.add_node(filter_node);

        match graph.request_full(filter_id) {
            Ok(output) => {
                let png = encode_png(&output, w, h);
                fs::write(out_dir.join(format!("{}-after.png", f.name)), &png).unwrap();
                rendered += 1;
            }
            Err(e) => {
                eprintln!("WARN: {} failed: {}", f.name, e);
                skipped += 1;
            }
        }
    }

    println!("Rendered: {} examples, skipped: {}", rendered, skipped);
    println!("Output: {}", out_dir.display());
}
