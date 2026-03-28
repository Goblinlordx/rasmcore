//! Performance: ImageMagick vs libvips vs rasmcore (stateless + pipeline + WASM)
//!
//! Run: cargo test -p wasm-integration --test wasm_bench -- --nocapture --ignored

use std::process::Command;
use std::time::{Duration, Instant};

use rasmcore_image::domain::pipeline::graph::NodeGraph;
use rasmcore_image::domain::pipeline::nodes::{filters as pf, sink, source, transform as pt};
use rasmcore_image::domain::types::*;
use rasmcore_image::domain::{decoder, filters, transform};
use wasm_integration::exports::rasmcore::image::transform::ResizeFilter;
use wasm_integration::*;

const W: u32 = 2;
const N: u32 = 10;

fn fmt(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 { format!("{us}us") }
    else if us < 1_000_000 { format!("{:.2}ms", us as f64 / 1000.0) }
    else { format!("{:.2}s", us as f64 / 1_000_000.0) }
}

fn bench<F: FnMut()>(mut f: F) -> Duration {
    for _ in 0..W { f(); }
    let s = Instant::now();
    for _ in 0..N { f(); }
    s.elapsed() / N
}

fn has(cmd: &str) -> bool {
    Command::new(cmd).arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().is_ok_and(|s| s.success())
}

#[test]
#[ignore]
fn performance_comparison() {
    let fp = fixtures_dir().join("inputs/gradient_64x64.png");
    let ff = std::fs::canonicalize(&fp).unwrap().to_str().unwrap().to_string();
    let data = load_fixture("gradient_64x64.png");
    let hm = has("magick");
    let hv = has("vips");
    let dec = decoder::decode(&data).unwrap();
    let (mut st, bi) = instantiate_image_component();
    let wd = bi.rasmcore_image_decoder().call_decode(&mut st, &data).unwrap().unwrap();
    let tr = bi.rasmcore_image_transform();
    let fl = bi.rasmcore_image_filters();
    let na = "N/A".to_string();

    // ── Single: resize ──
    let im_r = if hm { bench(|| { Command::new("magick").args([&ff, "-resize", "32x16!", "null:"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let vp_r = if hv { bench(|| { Command::new("vips").args(["thumbnail", &ff, "/dev/null", "32", "--height", "16"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let ns_r = bench(|| { let _ = transform::resize(&dec.pixels, &dec.info, 32, 16, rasmcore_image::domain::types::ResizeFilter::Lanczos3).unwrap(); });
    let np_r = bench(|| {
        let mut g = NodeGraph::new(4 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let r = g.add_node(Box::new(pt::ResizeNode::new(s, si, 32, 16, rasmcore_image::domain::types::ResizeFilter::Lanczos3)));
        let _ = sink::write(&mut g, r, "png", None).unwrap();
    });
    let ws_r = bench(|| { let _ = tr.call_resize(&mut st, &wd.pixels, wd.info, 32, 16, ResizeFilter::Lanczos3).unwrap().unwrap(); });

    // ── Single: blur ──
    let im_b = if hm { bench(|| { Command::new("magick").args([&ff, "-blur", "0x2", "null:"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let vp_b = if hv { bench(|| { Command::new("vips").args(["gaussblur", &ff, "/dev/null", "2"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let ns_b = bench(|| { let _ = filters::blur(&dec.pixels, &dec.info, 2.0).unwrap(); });
    let np_b = bench(|| {
        let mut g = NodeGraph::new(4 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let b = g.add_node(Box::new(pf::BlurNode::new(s, si, 2.0)));
        let _ = sink::write(&mut g, b, "png", None).unwrap();
    });
    let ws_b = bench(|| { let _ = fl.call_blur(&mut st, &wd.pixels, wd.info, 2.0).unwrap().unwrap(); });

    // ── Chain: read → resize 32x32 → blur → sharpen → brightness → write JPEG ──
    let im_c = if hm { bench(|| { Command::new("magick").args([&ff, "-resize", "32x32!", "-blur", "0x2", "-sharpen", "0x1", "-brightness-contrast", "10", "null:"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let vp_c = if hv { bench(|| {
        Command::new("sh").args(["-c", &format!(
            "vips thumbnail '{}' /tmp/_rb.v 32 --height 32 && vips gaussblur /tmp/_rb.v /tmp/_rb2.v 2 && vips sharpen /tmp/_rb2.v /tmp/_rb3.v && vips linear /tmp/_rb3.v /dev/null 1.0 25.5", ff
        )]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap();
    }) } else { Duration::ZERO };

    // Native stateless chain: 5 separate allocations + copies
    let ns_c = bench(|| {
        let d = decoder::decode(&data).unwrap();
        let r = transform::resize(&d.pixels, &d.info, 32, 32, rasmcore_image::domain::types::ResizeFilter::Lanczos3).unwrap();
        let b = filters::blur(&r.pixels, &r.info, 2.0).unwrap();
        let s = filters::sharpen(&b, &r.info, 1.0).unwrap();
        let _ = filters::brightness(&s, &r.info, 0.1).unwrap();
    });

    // Native pipeline chain: single graph, cached regions
    let np_c = bench(|| {
        let mut g = NodeGraph::new(4 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let r = g.add_node(Box::new(pt::ResizeNode::new(s, si, 32, 32, rasmcore_image::domain::types::ResizeFilter::Lanczos3)));
        let ri = g.node_info(r).unwrap();
        let b = g.add_node(Box::new(pf::BlurNode::new(r, ri.clone(), 2.0)));
        let sh = g.add_node(Box::new(pf::SharpenNode::new(b, ri.clone(), 1.0)));
        let br = g.add_node(Box::new(pf::BrightnessNode::new(sh, ri, 0.1)));
        let _ = sink::write(&mut g, br, "jpeg", Some(85)).unwrap();
    });

    // WASM stateless chain: 5 cross-boundary pixel copies
    let ws_c = bench(|| {
        let d = bi.rasmcore_image_decoder().call_decode(&mut st, &data).unwrap().unwrap();
        let (rp, ri) = tr.call_resize(&mut st, &d.pixels, d.info, 32, 32, ResizeFilter::Lanczos3).unwrap().unwrap();
        let bp = fl.call_blur(&mut st, &rp, ri, 2.0).unwrap().unwrap();
        let sp = fl.call_sharpen(&mut st, &bp, ri, 1.0).unwrap().unwrap();
        let _ = fl.call_brightness(&mut st, &sp, ri, 0.1).unwrap().unwrap();
    });

    // ── Report ──
    println!("\n============================================================================================");
    println!("        Performance Comparison (64x64 PNG, {N} iterations)");
    println!("============================================================================================\n");
    println!("  {:<25} {:>9} {:>9} {:>9} {:>9} {:>9}", "", "magick", "vips", "Natv/SL", "Natv/PL", "WASM/SL");
    println!("  {:<25} {:>9} {:>9} {:>9} {:>9} {:>9}", "", "------", "----", "-------", "-------", "-------");
    macro_rules! r { ($n:expr,$a:expr,$b:expr,$c:expr,$d:expr,$e:expr) => {
        println!("  {:<25} {:>9} {:>9} {:>9} {:>9} {:>9}", $n,
            if hm{fmt($a)}else{na.clone()}, if hv{fmt($b)}else{na.clone()},
            fmt($c), fmt($d), fmt($e));
    }}
    println!("  Single operations:");
    r!("  resize 32x16", im_r, vp_r, ns_r, np_r, ws_r);
    r!("  blur r=2", im_b, vp_b, ns_b, np_b, ws_b);
    println!();
    println!("  Chain: read -> resize -> blur -> sharpen -> brightness -> write");
    r!("  chain (5 ops)", im_c, vp_c, ns_c, np_c, ws_c);
    println!("\n  SL=Stateless (separate alloc per op)  PL=Pipeline (NodeGraph + SpatialCache)");
    println!("============================================================================================\n");

    // ═══════════════════════════════════════════════════════════════
    //  1080p (1920x1080) — shows pipeline advantage on real images
    // ═══════════════════════════════════════════════════════════════

    let hd_path = fixtures_dir().join("inputs/gradient_1920x1080.png");
    if !hd_path.exists() {
        println!("  (skipping 1080p bench — gradient_1920x1080.png not found)\n");
        return;
    }
    let hd_file = std::fs::canonicalize(&hd_path).unwrap().to_str().unwrap().to_string();
    let hd_data = std::fs::read(&hd_path).unwrap();
    let hd_dec = decoder::decode(&hd_data).unwrap();
    let hd_wd = bi.rasmcore_image_decoder().call_decode(&mut st, &hd_data).unwrap().unwrap();

    // ── 1080p resize to 960x540 ──
    let hd_im_r = if hm { bench(|| { Command::new("magick").args([&hd_file, "-resize", "960x540!", "null:"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let hd_vp_r = if hv { bench(|| { Command::new("vips").args(["thumbnail", &hd_file, "/dev/null", "960", "--height", "540"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let hd_ns_r = bench(|| { let _ = transform::resize(&hd_dec.pixels, &hd_dec.info, 960, 540, rasmcore_image::domain::types::ResizeFilter::Lanczos3).unwrap(); });
    let hd_np_r = bench(|| {
        let mut g = NodeGraph::new(32 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(hd_data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let r = g.add_node(Box::new(pt::ResizeNode::new(s, si, 960, 540, rasmcore_image::domain::types::ResizeFilter::Lanczos3)));
        let _ = sink::write(&mut g, r, "png", None).unwrap();
    });
    let hd_ws_r = bench(|| { let _ = tr.call_resize(&mut st, &hd_wd.pixels, hd_wd.info, 960, 540, ResizeFilter::Lanczos3).unwrap().unwrap(); });

    // ── 1080p blur ──
    let hd_im_b = if hm { bench(|| { Command::new("magick").args([&hd_file, "-blur", "0x2", "null:"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let hd_vp_b = if hv { bench(|| { Command::new("vips").args(["gaussblur", &hd_file, "/dev/null", "2"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let hd_ns_b = bench(|| { let _ = filters::blur(&hd_dec.pixels, &hd_dec.info, 2.0).unwrap(); });
    let hd_np_b = bench(|| {
        let mut g = NodeGraph::new(32 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(hd_data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let b = g.add_node(Box::new(pf::BlurNode::new(s, si, 2.0)));
        let _ = sink::write(&mut g, b, "png", None).unwrap();
    });
    let hd_ws_b = bench(|| { let _ = fl.call_blur(&mut st, &hd_wd.pixels, hd_wd.info, 2.0).unwrap().unwrap(); });

    // ── 1080p chain: read → resize 960x540 → blur → sharpen → brightness → write JPEG ──
    let hd_im_c = if hm { bench(|| { Command::new("magick").args([&hd_file, "-resize", "960x540!", "-blur", "0x2", "-sharpen", "0x1", "-brightness-contrast", "10", "null:"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap(); }) } else { Duration::ZERO };
    let hd_vp_c = if hv { bench(|| {
        Command::new("sh").args(["-c", &format!(
            "vips thumbnail '{}' /tmp/_rb.v 960 --height 540 && vips gaussblur /tmp/_rb.v /tmp/_rb2.v 2 && vips sharpen /tmp/_rb2.v /tmp/_rb3.v && vips linear /tmp/_rb3.v /dev/null 1.0 25.5", hd_file
        )]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().unwrap();
    }) } else { Duration::ZERO };
    let hd_ns_c = bench(|| {
        let d = decoder::decode(&hd_data).unwrap();
        let r = transform::resize(&d.pixels, &d.info, 960, 540, rasmcore_image::domain::types::ResizeFilter::Lanczos3).unwrap();
        let b = filters::blur(&r.pixels, &r.info, 2.0).unwrap();
        let s = filters::sharpen(&b, &r.info, 1.0).unwrap();
        let _ = filters::brightness(&s, &r.info, 0.1).unwrap();
    });
    let hd_np_c = bench(|| {
        let mut g = NodeGraph::new(32 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(hd_data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let r = g.add_node(Box::new(pt::ResizeNode::new(s, si, 960, 540, rasmcore_image::domain::types::ResizeFilter::Lanczos3)));
        let ri = g.node_info(r).unwrap();
        let b = g.add_node(Box::new(pf::BlurNode::new(r, ri.clone(), 2.0)));
        let sh = g.add_node(Box::new(pf::SharpenNode::new(b, ri.clone(), 1.0)));
        let br = g.add_node(Box::new(pf::BrightnessNode::new(sh, ri, 0.1)));
        let _ = sink::write(&mut g, br, "jpeg", Some(85)).unwrap();
    });
    let hd_ws_c = bench(|| {
        let d = bi.rasmcore_image_decoder().call_decode(&mut st, &hd_data).unwrap().unwrap();
        let (rp, ri) = tr.call_resize(&mut st, &d.pixels, d.info, 960, 540, ResizeFilter::Lanczos3).unwrap().unwrap();
        let bp = fl.call_blur(&mut st, &rp, ri, 2.0).unwrap().unwrap();
        let sp = fl.call_sharpen(&mut st, &bp, ri, 1.0).unwrap().unwrap();
        let _ = fl.call_brightness(&mut st, &sp, ri, 0.1).unwrap().unwrap();
    });

    println!("============================================================================================");
    println!("        Performance Comparison (1920x1080 PNG, {N} iterations)");
    println!("============================================================================================\n");
    println!("  {:<25} {:>9} {:>9} {:>9} {:>9} {:>9}", "", "magick", "vips", "Natv/SL", "Natv/PL", "WASM/SL");
    println!("  {:<25} {:>9} {:>9} {:>9} {:>9} {:>9}", "", "------", "----", "-------", "-------", "-------");
    println!("  Single operations:");
    r!("  resize 960x540", hd_im_r, hd_vp_r, hd_ns_r, hd_np_r, hd_ws_r);
    r!("  blur r=2", hd_im_b, hd_vp_b, hd_ns_b, hd_np_b, hd_ws_b);
    println!();
    println!("  Chain: read -> resize 960x540 -> blur -> sharpen -> brightness -> write JPEG");
    r!("  chain (5 ops)", hd_im_c, hd_vp_c, hd_ns_c, hd_np_c, hd_ws_c);
    println!("\n  SL=Stateless (separate alloc per op)  PL=Pipeline (NodeGraph + SpatialCache)");
    println!("  1080p image: 1920x1080 RGB = 6.2 MB decoded pixels");
    println!("============================================================================================\n");
}
