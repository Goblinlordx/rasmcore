//! Audit all registered V2 filters — test the ACTUAL docs playground path.
//!
//! This exercises the binary param serialization → deserialize_params → apply_filter
//! path, which is what the docs playground and web-ui use. NOT the direct ParamMap
//! path that the macro-generated factories guarantee to work.

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

/// Encode params into binary format exactly as the JS playground does:
/// [name_len:u8, name_bytes, type:u8, value_bytes] repeated
/// type 0 = f32 (4 bytes LE), type 1 = u32 (4 bytes LE), type 2 = bool (1 byte)
fn serialize_params_binary(params: &[v2::ParamDescriptor]) -> Vec<u8> {
    let mut buf = Vec::new();
    for p in params {
        let v: f64 = p.default.filter(|d| d.abs() > 1e-6).unwrap_or_else(|| {
            match (p.min, p.max) {
                (Some(lo), Some(hi)) => lo + (hi - lo) * 0.3,
                _ => 0.3,
            }
        });

        let name_bytes = p.name.as_bytes();
        buf.push(name_bytes.len() as u8);
        buf.extend_from_slice(name_bytes);

        match p.value_type {
            ParamType::F32 | ParamType::F64 => {
                buf.push(0);
                buf.extend_from_slice(&(v as f32).to_le_bytes());
            }
            ParamType::U32 | ParamType::I32 => {
                buf.push(1);
                buf.extend_from_slice(&(v as u32).to_le_bytes());
            }
            ParamType::Bool => {
                buf.push(2);
                buf.push(if v > 0.5 { 1 } else { 0 });
            }
            ParamType::String | ParamType::Rect => {}
            ParamType::NodeRef | ParamType::FontRef | ParamType::LutRef => {
                buf.push(3); // ref types use u32 payload
                buf.extend_from_slice(&(v as u32).to_le_bytes());
            }
        }
    }
    buf
}

/// Deserialize binary params — same code path as the WASM adapter
fn deserialize_params(buf: &[u8]) -> ParamMap {
    let mut map = ParamMap::new();
    let mut i = 0;
    while i < buf.len() {
        let name_len = buf[i] as usize;
        i += 1;
        if i + name_len > buf.len() { break; }
        let name = String::from_utf8_lossy(&buf[i..i + name_len]).to_string();
        i += name_len;
        if i >= buf.len() { break; }
        let value_type = buf[i];
        i += 1;
        match value_type {
            0 => {
                if i + 4 > buf.len() { break; }
                let v = f32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
                map.floats.insert(name, v);
                i += 4;
            }
            1 => {
                if i + 4 > buf.len() { break; }
                let v = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]);
                map.ints.insert(name, v as i64);
                i += 4;
            }
            2 => {
                if i + 1 > buf.len() { break; }
                map.bools.insert(name, buf[i] != 0);
                i += 1;
            }
            _ => break,
        }
    }
    map
}

/// Build ParamMap directly (the "known good" path)
fn direct_params(params: &[v2::ParamDescriptor]) -> ParamMap {
    let mut map = ParamMap::new();
    for p in params {
        let v: f64 = p.default.filter(|d| d.abs() > 1e-6).unwrap_or_else(|| {
            match (p.min, p.max) {
                (Some(lo), Some(hi)) => lo + (hi - lo) * 0.3,
                _ => 0.3,
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

fn test_filter(
    name: &str,
    pixels: &[f32],
    info: &NodeInfo,
    params: &ParamMap,
) -> Result<Vec<f32>, String> {
    let src = Box::new(TestSource { pixels: pixels.to_vec(), info: info.clone() });
    let mut g = Graph::new(0);
    let sid = g.add_node(src);
    let node = v2::create_filter_node(name, sid, info.clone(), params)
        .ok_or_else(|| format!("create_filter_node returned None"))?;
    let fid = g.add_node(node);
    let out = g.request_full(fid).map_err(|e| format!("{}", e))?;
    let out_info = g.node_info(fid).map_err(|e| format!("{}", e))?;
    let expected = out_info.width as usize * out_info.height as usize * 4;
    if out.len() != expected {
        return Err(format!("size mismatch: {} vs {}", out.len(), expected));
    }
    let nan = out.iter().filter(|v| v.is_nan()).count();
    let inf = out.iter().filter(|v| v.is_infinite()).count();
    if nan > 0 || inf > 0 {
        return Err(format!("{} NaN, {} Inf in output", nan, inf));
    }
    Ok(out)
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

    println!("# Filter Audit Report");
    println!();
    println!("Tests both code paths:");
    println!("- **Direct**: ParamMap built from defaults (Rust-only, always works)");
    println!("- **Binary**: Params serialized → deserialized (what the playground does)");
    println!();

    let mut results = Vec::new();

    for f in &filters {
        let direct_map = direct_params(f.params);
        let binary_buf = serialize_params_binary(f.params);
        let binary_map = deserialize_params(&binary_buf);

        // Compare param maps
        let mut param_diffs = Vec::new();
        for p in f.params {
            match p.value_type {
                ParamType::F32 | ParamType::F64 => {
                    let d = direct_map.floats.get(p.name).copied().unwrap_or(f32::NAN);
                    let b = binary_map.floats.get(p.name).copied().unwrap_or(f32::NAN);
                    if (d - b).abs() > 1e-6 || d.is_nan() != b.is_nan() {
                        param_diffs.push(format!("{}(f32): direct={:.4} binary={:.4}", p.name, d, b));
                    }
                }
                ParamType::U32 | ParamType::I32 => {
                    let d = direct_map.ints.get(p.name).copied();
                    let b = binary_map.ints.get(p.name).copied();
                    if d != b {
                        param_diffs.push(format!("{}(int): direct={:?} binary={:?}", p.name, d, b));
                    }
                }
                ParamType::Bool => {
                    let d = direct_map.bools.get(p.name).copied();
                    let b = binary_map.bools.get(p.name).copied();
                    if d != b {
                        param_diffs.push(format!("{}(bool): direct={:?} binary={:?}", p.name, d, b));
                    }
                }
                _ => {}
            }
        }

        let direct_result = test_filter(f.name, &pixels, &info, &direct_map);
        let binary_result = test_filter(f.name, &pixels, &info, &binary_map);

        // Compare outputs if both succeeded
        let output_match = match (&direct_result, &binary_result) {
            (Ok(a), Ok(b)) => {
                if a.len() != b.len() {
                    Some(format!("different lengths: {} vs {}", a.len(), b.len()))
                } else {
                    let max_diff = a.iter().zip(b.iter())
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    if max_diff > 1e-4 {
                        Some(format!("max pixel diff: {:.6}", max_diff))
                    } else {
                        None
                    }
                }
            }
            _ => None,
        };

        results.push((
            f.name,
            f.category,
            f.params.len(),
            param_diffs,
            direct_result.is_ok(),
            binary_result.is_ok(),
            direct_result.err(),
            binary_result.err(),
            output_match,
        ));
    }

    // Summary
    let total = results.len();
    let both_pass = results.iter().filter(|r| r.4 && r.5).count();
    let binary_fail = results.iter().filter(|r| r.4 && !r.5).count();
    let both_fail = results.iter().filter(|r| !r.4).count();
    let param_mismatch = results.iter().filter(|r| !r.3.is_empty()).count();
    let output_diff = results.iter().filter(|r| r.8.is_some()).count();

    println!("## Summary");
    println!();
    println!("| Metric | Count |");
    println!("|--------|-------|");
    println!("| Total filters | {} |", total);
    println!("| Both paths pass | {} |", both_pass);
    println!("| Binary-only failure | {} |", binary_fail);
    println!("| Both paths fail | {} |", both_fail);
    println!("| Param serialization mismatch | {} |", param_mismatch);
    println!("| Output differs between paths | {} |", output_diff);
    println!();

    // Failures
    let failures: Vec<_> = results.iter().filter(|r| !r.4 || !r.5).collect();
    if !failures.is_empty() {
        println!("## Failures");
        println!();
        println!("| Filter | Cat | Params | Direct | Binary | Error |");
        println!("|--------|-----|--------|--------|--------|-------|");
        for r in &failures {
            let d = if r.4 { "PASS" } else { "FAIL" };
            let b = if r.5 { "PASS" } else { "FAIL" };
            let err = r.7.as_deref().or(r.6.as_deref()).unwrap_or("");
            println!("| {} | {} | {} | {} | {} | {} |", r.0, r.1, r.2, d, b, err);
        }
        println!();
    }

    // Param mismatches
    let mismatches: Vec<_> = results.iter().filter(|r| !r.3.is_empty()).collect();
    if !mismatches.is_empty() {
        println!("## Param Serialization Mismatches");
        println!();
        for r in &mismatches {
            println!("**{}** ({}): {}", r.0, r.1, r.3.join(", "));
        }
        println!();
    }

    // Output diffs
    let diffs: Vec<_> = results.iter().filter(|r| r.8.is_some()).collect();
    if !diffs.is_empty() {
        println!("## Output Differences (both pass but different results)");
        println!();
        println!("| Filter | Cat | Difference |");
        println!("|--------|-----|------------|");
        for r in &diffs {
            println!("| {} | {} | {} |", r.0, r.1, r.8.as_deref().unwrap());
        }
        println!();
    }

    // Pass list
    println!("## Passing Filters ({}/{})", both_pass, total);
    println!();
    for r in results.iter().filter(|r| r.4 && r.5 && r.3.is_empty() && r.8.is_none()) {
        println!("- {} ({})", r.0, r.1);
    }

    if binary_fail > 0 || both_fail > 0 {
        std::process::exit(1);
    }
}
