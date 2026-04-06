//! Render before/after example images for documentation.
//!
//! For each registered V2 filter, applies it with showcase params
//! to a reference photo and writes the result as a PNG.
//!
//! Usage: cargo run --bin render_examples -p rasmcore-v2-wasm
//! Output: /tmp/docs-examples/{name}-after.png

#[allow(unused_imports)]
use rasmcore_pipeline_v2::filters as _f;
#[allow(unused_imports)]
use rasmcore_codecs_v2 as _c;

use rasmcore_pipeline_v2 as v2;
use rasmcore_pipeline_v2::graph::Graph;
use rasmcore_pipeline_v2::node::{NodeInfo, PipelineError};
use rasmcore_pipeline_v2::registry::ParamMap;
use std::fs;
use std::path::{Path, PathBuf};

struct SourceNode { pixels: Vec<f32>, info: NodeInfo }

impl v2::Node for SourceNode {
    fn info(&self) -> NodeInfo { self.info.clone() }
    fn compute(&self, _r: v2::Rect, _u: &mut dyn v2::Upstream) -> Result<Vec<f32>, PipelineError> {
        Ok(self.pixels.clone())
    }
    fn upstream_ids(&self) -> Vec<u32> { vec![] }
}

fn find_project_root() -> PathBuf {
    let mut dir = std::env::current_dir().expect("cwd");
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("crates").exists() { return dir; }
        if !dir.pop() { return std::env::current_dir().expect("cwd"); }
    }
}

fn load_reference(project_root: &Path) -> (Vec<f32>, u32, u32) {
    for name in ["reference-landscape.jpg", "reference-portrait.jpg", "reference-architecture.jpg"] {
        let path = project_root.join("docs/assets").join(name);
        if path.exists() {
            let data = fs::read(&path).expect("read ref");
            if let Ok(d) = rasmcore_codecs_v2::decode(&data) {
                println!("Reference: {} ({}x{})", name, d.info.width, d.info.height);
                return (d.pixels, d.info.width, d.info.height);
            }
        }
    }
    eprintln!("WARN: no photos found, using gradient");
    let (w, h) = (400u32, 300u32);
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h { for x in 0..w {
        let (fx, fy) = (x as f32 / (w-1) as f32, y as f32 / (h-1) as f32);
        px.extend_from_slice(&[fx, (1.0-(fx-0.5).abs()*2.0).max(0.0), fy, 1.0]);
    }}
    (px, w, h)
}

fn encode_png(pixels: &[f32], w: u32, h: u32) -> Vec<u8> {
    rasmcore_codecs_v2::encode(pixels, w, h, "png", None).expect("PNG encode")
}

fn showcase_params(name: &str, params: &[v2::ParamDescriptor]) -> ParamMap {
    let mut map = ParamMap::new();
    for p in params {
        let v: f64 = match (name, p.name) {
            // Adjustment — visible but not extreme
            ("brightness", "amount") => 0.15,
            ("contrast", "amount") => 0.4,
            ("gamma", "gamma") => 1.5,
            ("exposure", "ev") => 1.0,
            ("posterize", "levels") => 6.0,
            ("sigmoidal_contrast", "contrast") => 5.0,
            ("sigmoidal_contrast", "midpoint") => 0.5,
            ("levels", "black_point") => 0.05,
            ("levels", "white_point") => 0.95,
            ("burn", "amount") => 0.3,
            ("dodge", "amount") => 0.3,
            ("solarize", "threshold") => 0.5,

            // Spatial — moderate blur/sharpen
            ("gaussian_blur", "radius") => 3.0,
            ("box_blur", "radius") => 3.0,
            ("motion_blur", "radius") => 5.0,
            ("motion_blur", "angle") => 45.0,
            ("sharpen", "amount") => 1.5,
            ("bilateral", "sigma_spatial") => 5.0,
            ("bilateral", "sigma_range") => 0.1,
            ("median", "radius") => 2.0,

            // Color — clearly visible shifts
            ("hue_rotate", _) => 90.0,
            ("sepia", _) => 0.8,
            ("saturate", "amount") => 0.5,
            ("vibrance", _) => 0.5,
            ("colorize", "hue") => 200.0,
            ("colorize", "saturation") => 0.6,
            ("colorize", "strength") => 0.5,
            ("white_balance_temperature", "temperature") => 7500.0,
            ("photo_filter", "hue") => 30.0,
            ("photo_filter", "strength") => 0.4,
            ("lab_adjust", "lightness") => 10.0,
            ("modulate", "hue") => 30.0,
            ("selective_color", "target_hue") => 0.0,
            ("selective_color", "hue_shift") => 30.0,
            ("selective_color", "range") => 60.0,

            // Enhancement — dramatic enough to see
            ("vignette", "sigma") => 0.15,
            ("vignette", "x_inset") => 50.0,
            ("vignette", "y_inset") => 50.0,
            ("vignette_powerlaw", "strength") => 0.8,
            ("vignette_powerlaw", "falloff") => 2.5,
            ("clarity", "amount") => 0.6,
            ("clarity", "radius") => 15.0,
            ("shadow_highlight", "shadows") => 0.4,
            ("shadow_highlight", "highlights") => -0.3,
            ("dehaze", "strength") => 0.5,
            ("clahe", "clip_limit") => 3.0,

            // Effect — obvious visual change
            ("film_grain", "amount") => 0.15,
            ("film_grain_grading", "amount") => 0.15,
            ("chromatic_aberration", "amount") => 5.0,
            ("pixelate", "size") => 8.0,
            ("emboss", "strength") => 1.0,
            ("oil_paint", "radius") => 3.0,
            ("halftone", "dot_size") => 4.0,
            ("glitch", "amount") => 0.3,
            ("light_leak", "intensity") => 0.5,

            // Grading
            ("split_toning", "shadow_hue") => 220.0,
            ("split_toning", "highlight_hue") => 40.0,
            ("split_toning", "shadow_strength") => 0.3,
            ("split_toning", "highlight_strength") => 0.3,
            ("asc_cdl", "slope_r") => 1.2,
            ("asc_cdl", "power_r") => 0.9,
            ("lift_gamma_gain", "lift_r") => -0.05,
            ("lift_gamma_gain", "gain_r") => 1.1,
            ("tonemap_reinhard", "exposure") => 1.5,
            ("tonemap_filmic", "exposure") => 1.5,
            ("tonemap_drago", "exposure") => 1.5,

            // Dither/quantize — visible pattern
            ("dither_ordered", "levels") => 4.0,
            ("dither_floyd_steinberg", "levels") => 4.0,
            ("quantize", "levels") => 4.0,
            ("kmeans_quantize", "k") => 8.0,

            // Scope — use smaller size for faster examples
            ("scope_histogram", "scope_size") => 256.0,
            ("scope_waveform", "scope_size") => 256.0,
            ("scope_parade", "scope_size") => 256.0,
            ("scope_vectorscope", "scope_size") => 256.0,

            // Fallback: use default or 30% of range
            _ => p.default.filter(|d| d.abs() > 1e-6).unwrap_or_else(|| {
                match (p.min, p.max) { (Some(lo), Some(hi)) => lo + (hi - lo) * 0.3, _ => 0.3 }
            }),
        };
        match p.value_type {
            v2::ParamType::F32 | v2::ParamType::F64 => { map.floats.insert(p.name.into(), v as f32); }
            v2::ParamType::U32 | v2::ParamType::I32 => { map.ints.insert(p.name.into(), v as i64); }
            v2::ParamType::Bool => { map.bools.insert(p.name.into(), v > 0.5); }
            _ => {}
        }
    }
    map
}

fn main() {
    let root = find_project_root();
    let out = Path::new("/tmp/docs-examples");
    fs::create_dir_all(out).unwrap();

    let (ref_px, w, h) = load_reference(&root);
    fs::write(out.join("reference.png"), encode_png(&ref_px, w, h)).unwrap();

    // Also copy all reference photos as PNGs
    for name in ["landscape", "portrait", "architecture"] {
        let path = root.join(format!("docs/assets/reference-{}.jpg", name));
        if path.exists() {
            if let Ok(d) = rasmcore_codecs_v2::decode(&fs::read(&path).unwrap()) {
                fs::write(out.join(format!("reference-{}.png", name)), encode_png(&d.pixels, d.info.width, d.info.height)).unwrap();
            }
        }
    }

    let info = NodeInfo { width: w, height: h, color_space: v2::ColorSpace::Linear };
    let filters = v2::registry::registered_filter_registrations();
    let (mut ok, mut fail) = (0usize, 0usize);

    for f in &filters {
        let params = showcase_params(f.name, f.params);
        let src = Box::new(SourceNode { pixels: ref_px.clone(), info: info.clone() });
        let mut g = Graph::new(16 * 1024 * 1024);
        let sid = g.add_node(src);
        let fid = g.add_node((f.factory)(sid, info.clone(), &params));
        match g.request_full(fid) {
            Ok(out_px) => {
                // Use output node dimensions (may differ from input for scope/transform filters)
                let out_info = g.node_info(fid).unwrap_or(info.clone());
                fs::write(out.join(format!("{}-after.png", f.name)), encode_png(&out_px, out_info.width, out_info.height)).unwrap();
                ok += 1;
            }
            Err(e) => { eprintln!("WARN: {} failed: {}", f.name, e); fail += 1; }
        }
    }

    println!("Rendered: {ok} examples, skipped: {fail}");
    println!("Output: {}", out.display());
}
