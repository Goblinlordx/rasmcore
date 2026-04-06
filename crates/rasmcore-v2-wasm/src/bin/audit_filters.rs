//! Audit all registered V2 filters — test each with default params.
//!
//! Produces a structured report classifying each filter as PASS, INTERFACE_BUG,
//! or IMPL_BUG with root cause details.

// Force linker to include filter registrations
#[allow(unused_imports)]
use rasmcore_pipeline_v2::filters as _v2_filters;
#[allow(unused_imports)]
use rasmcore_codecs_v2 as _v2_codecs;
use rasmcore_pipeline_v2::filters::scope;

use rasmcore_pipeline_v2 as v2;
use v2::{Graph, NodeInfo, ColorSpace, ParamMap, ParamType};
use v2::node::{Node, PipelineError, Upstream};
use v2::rect::Rect;

struct TestSource {
    pixels: Vec<f32>,
    info: NodeInfo,
}
impl Node for TestSource {
    fn info(&self) -> NodeInfo { self.info.clone() }
    fn compute(&self, _r: Rect, _u: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
        Ok(self.pixels.clone())
    }
    fn upstream_ids(&self) -> Vec<u32> { vec![] }
}

fn default_params(params: &[v2::ParamDescriptor]) -> ParamMap {
    let mut map = ParamMap::new();
    for p in params {
        let v: f64 = p.default.filter(|d| d.abs() > 1e-6).unwrap_or_else(|| {
            match (p.min, p.max) {
                (Some(lo), Some(hi)) => lo + (hi - lo) * 0.3,
                _ => 0.3
            }
        });
        match p.value_type {
            ParamType::F32 | ParamType::F64 => { map.floats.insert(p.name.into(), v as f32); }
            ParamType::U32 | ParamType::I32 => { map.ints.insert(p.name.into(), v as i64); }
            ParamType::Bool => { map.bools.insert(p.name.into(), v > 0.5); }
            _ => {}
        }
    }
    map
}

enum AuditResult {
    Pass,
    InterfaceBug(String),
    ImplBug(String),
}

fn audit_filter(
    f: &&v2::FilterFactoryRegistration,
    pixels: &[f32],
    info: &NodeInfo,
) -> AuditResult {
    let params = default_params(f.params);
    let src = Box::new(TestSource { pixels: pixels.to_vec(), info: info.clone() });
    let mut g = Graph::new(0);
    let sid = g.add_node(src);
    let node = (f.factory)(sid, info.clone(), &params);
    let fid = g.add_node(node);

    match g.request_full(fid) {
        Ok(out) => {
            let out_info = g.node_info(fid).unwrap();
            let expected = out_info.width as usize * out_info.height as usize * 4;
            if out.len() != expected {
                return AuditResult::ImplBug(format!(
                    "size mismatch: got {} floats, expected {} ({}x{}x4)",
                    out.len(), expected, out_info.width, out_info.height
                ));
            }
            let nan_count = out.iter().filter(|v| v.is_nan()).count();
            let inf_count = out.iter().filter(|v| v.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                return AuditResult::ImplBug(format!(
                    "bad values: {} NaN, {} Inf in {} floats",
                    nan_count, inf_count, out.len()
                ));
            }
            AuditResult::Pass
        }
        Err(e) => AuditResult::InterfaceBug(format!("{}", e)),
    }
}

fn main() {
    scope::ensure_linked();

    let w = 32u32;
    let h = 32u32;
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = x as f32 / (w - 1) as f32;
            let g = y as f32 / (h - 1) as f32;
            let b = ((x + y) as f32 / (w + h - 2) as f32).min(1.0);
            pixels.extend_from_slice(&[r, g, b, 1.0]);
        }
    }
    let info = NodeInfo { width: w, height: h, color_space: ColorSpace::Linear };

    let filters = v2::registry::registered_filter_registrations();
    let mut pass = Vec::new();
    let mut iface_bugs = Vec::new();
    let mut impl_bugs = Vec::new();

    for f in &filters {
        match audit_filter(&f, &pixels, &info) {
            AuditResult::Pass => pass.push((f.name, f.category)),
            AuditResult::InterfaceBug(msg) => iface_bugs.push((f.name, f.category, msg)),
            AuditResult::ImplBug(msg) => impl_bugs.push((f.name, f.category, msg)),
        }
    }

    // Print report
    let total = pass.len() + iface_bugs.len() + impl_bugs.len();
    println!("# Filter Audit Report\n");
    println!("Total registered: {}", total);
    println!("PASS: {}", pass.len());
    println!("INTERFACE_BUG: {}", iface_bugs.len());
    println!("IMPL_BUG: {}\n", impl_bugs.len());

    if !iface_bugs.is_empty() {
        println!("## Interface Bugs\n");
        println!("| Filter | Category | Error |");
        println!("|--------|----------|-------|");
        for (name, cat, msg) in &iface_bugs {
            println!("| {} | {} | {} |", name, cat, msg);
        }
        println!();
    }

    if !impl_bugs.is_empty() {
        println!("## Implementation Bugs\n");
        println!("| Filter | Category | Issue |");
        println!("|--------|----------|-------|");
        for (name, cat, msg) in &impl_bugs {
            println!("| {} | {} | {} |", name, cat, msg);
        }
        println!();
    }

    println!("## Passing Filters ({})\n", pass.len());
    println!("| Filter | Category |");
    println!("|--------|----------|");
    for (name, cat) in &pass {
        println!("| {} | {} |", name, cat);
    }

    // Exit with error if any failures
    if !iface_bugs.is_empty() || !impl_bugs.is_empty() {
        std::process::exit(1);
    }
}
