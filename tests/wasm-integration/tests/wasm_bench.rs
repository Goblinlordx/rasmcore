//! Performance: ImageMagick vs libvips vs rasmcore per-op + pipeline chain
//!
//! Run with RELEASE for real SIMD numbers:
//!   cargo test -p wasm-integration --test wasm_bench --release -- --nocapture --ignored

use std::process::Command;
use std::time::{Duration, Instant};

use rasmcore_image::domain::pipeline::graph::NodeGraph;
use rasmcore_image::domain::pipeline::nodes::{filters as pf, sink, source, transform as pt};
use rasmcore_image::domain::types::*;
use rasmcore_image::domain::{decoder, filters, transform};
use wasm_integration::*;

const W: u32 = 3;
const N: u32 = 10;

fn fmt(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 {
        format!("{us}us")
    } else if us < 1_000_000 {
        format!("{:.2}ms", us as f64 / 1000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

fn bench<F: FnMut()>(mut f: F) -> Duration {
    for _ in 0..W {
        f();
    }
    let s = Instant::now();
    for _ in 0..N {
        f();
    }
    s.elapsed() / N
}

fn has(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn run_magick(args: &[&str]) {
    let s = Command::new("magick")
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .unwrap();
    assert!(s.success());
}

fn run_vips(args: &[&str]) -> bool {
    Command::new("vips")
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
#[ignore]
fn perf_1080p() {
    let hd_path = fixtures_dir().join("inputs/gradient_1920x1080.png");
    if !hd_path.exists() {
        println!("(skipping — gradient_1920x1080.png not found)");
        return;
    }
    let ff = std::fs::canonicalize(&hd_path)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let data = std::fs::read(&hd_path).unwrap();
    let hm = has("magick");
    let hv = has("vips");

    // Pre-decode so we measure OPERATION time, not decode time
    let dec = decoder::decode(&data).unwrap();
    let na = "N/A".to_string();

    // WASM component (built with SIMD128)
    let (mut st, bi) = instantiate_image_component();
    let wd = bi
        .rasmcore_image_decoder()
        .call_decode(&mut st, &data)
        .unwrap()
        .unwrap();
    let wtr = bi.rasmcore_image_transform();
    let wfl = bi.rasmcore_image_filters();

    println!();
    println!(
        "================================================================================================"
    );
    println!("  1920x1080 Performance ({N} iter, {} warmup)", W);
    println!(
        "================================================================================================"
    );
    println!();
    println!(
        "  {:<28} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "", "magick", "vips", "Natv/SL", "Natv/PL", "WASM/SL"
    );
    println!(
        "  {:<28} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "", "------", "----", "-------", "-------", "-------"
    );

    macro_rules! row {
        ($n:expr, $im:expr, $vp:expr, $ns:expr, $np:expr, $ws:expr) => {
            println!(
                "  {:<28} {:>9} {:>9} {:>9} {:>9} {:>9}",
                $n,
                if hm { fmt($im) } else { na.clone() },
                if hv { fmt($vp) } else { na.clone() },
                fmt($ns),
                fmt($np),
                fmt($ws),
            );
        };
    }

    // ── Per-op: Resize ──
    let im = if hm {
        bench(|| run_magick(&[&ff, "-resize", "960x540!", "null:"]))
    } else {
        Duration::ZERO
    };
    let vp = if hv {
        bench(|| {
            run_vips(&["thumbnail", &ff, "/dev/null", "960", "--height", "540"]);
        })
    } else {
        Duration::ZERO
    };
    let ns = bench(|| {
        let _ =
            transform::resize(&dec.pixels, &dec.info, 960, 540, ResizeFilter::Lanczos3).unwrap();
    });
    let np = bench(|| {
        let mut g = NodeGraph::new(32 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let r = g.add_node(Box::new(pt::ResizeNode::new(
            s,
            si,
            960,
            540,
            ResizeFilter::Lanczos3,
        )));
        let _ = sink::write(&mut g, r, "png", None, None).unwrap();
    });
    let ws = bench(|| {
        let _ = wtr
            .call_resize(
                &mut st,
                &wd.pixels,
                wd.info,
                960,
                540,
                wasm_integration::exports::rasmcore::image::transform::ResizeFilter::Lanczos3,
            )
            .unwrap()
            .unwrap();
    });
    row!("resize 960x540 (lanczos3)", im, vp, ns, np, ws);

    // ── Per-op: Blur ──
    let im = if hm {
        bench(|| run_magick(&[&ff, "-blur", "0x2", "null:"]))
    } else {
        Duration::ZERO
    };
    let vp = if hv {
        bench(|| {
            run_vips(&["gaussblur", &ff, "/dev/null", "2"]);
        })
    } else {
        Duration::ZERO
    };
    let ns = bench(|| {
        let _ = filters::blur(&dec.pixels, &dec.info, 2.0).unwrap();
    });
    let np = bench(|| {
        let mut g = NodeGraph::new(32 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let b = g.add_node(Box::new(pf::BlurNode::new(s, si, 2.0)));
        let _ = sink::write(&mut g, b, "png", None, None).unwrap();
    });
    let ws = bench(|| {
        let _ = wfl
            .call_blur(&mut st, &wd.pixels, wd.info, 2.0)
            .unwrap()
            .unwrap();
    });
    row!("blur (gaussian r=2)", im, vp, ns, np, ws);

    // ── Per-op: Sharpen ──
    let im = if hm {
        bench(|| run_magick(&[&ff, "-sharpen", "0x1", "null:"]))
    } else {
        Duration::ZERO
    };
    let ns = bench(|| {
        let _ = filters::sharpen(&dec.pixels, &dec.info, 1.0).unwrap();
    });
    let ws = bench(|| {
        let _ = wfl
            .call_sharpen(&mut st, &wd.pixels, wd.info, 1.0)
            .unwrap()
            .unwrap();
    });
    row!("sharpen", im, Duration::ZERO, ns, Duration::ZERO, ws);

    // ── Per-op: Brightness ──
    let im = if hm {
        bench(|| run_magick(&[&ff, "-brightness-contrast", "20", "null:"]))
    } else {
        Duration::ZERO
    };
    let ns = bench(|| {
        let _ = filters::brightness(&dec.pixels, &dec.info, 0.2).unwrap();
    });
    let ws = bench(|| {
        let _ = wfl
            .call_brightness(&mut st, &wd.pixels, wd.info, 0.2)
            .unwrap()
            .unwrap();
    });
    row!(
        "brightness +0.2",
        im,
        Duration::ZERO,
        ns,
        Duration::ZERO,
        ws
    );

    // ── Per-op: Contrast ──
    let ns = bench(|| {
        let _ = filters::contrast(&dec.pixels, &dec.info, 0.5).unwrap();
    });
    let ws = bench(|| {
        let _ = wfl
            .call_contrast(&mut st, &wd.pixels, wd.info, 0.5)
            .unwrap()
            .unwrap();
    });
    row!(
        "contrast +0.5 (LUT)",
        Duration::ZERO,
        Duration::ZERO,
        ns,
        Duration::ZERO,
        ws
    );

    // ── Per-op: Grayscale ──
    let im = if hm {
        bench(|| run_magick(&[&ff, "-colorspace", "Gray", "null:"]))
    } else {
        Duration::ZERO
    };
    let ns = bench(|| {
        let _ = filters::grayscale(&dec.pixels, &dec.info).unwrap();
    });
    let ws = bench(|| {
        let _ = wfl
            .call_grayscale(&mut st, &wd.pixels, wd.info)
            .unwrap()
            .unwrap();
    });
    row!("grayscale", im, Duration::ZERO, ns, Duration::ZERO, ws);

    println!();
    println!("  Pipeline chains:");
    println!(
        "  {:<28} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "", "magick", "vips", "Natv/SL", "Natv/PL", "WASM/SL"
    );
    println!(
        "  {:<28} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "", "------", "----", "-------", "-------", "-------"
    );

    // ── Chain: resize → blur → sharpen → write JPEG ──
    let im = if hm {
        bench(|| {
            run_magick(&[
                &ff, "-resize", "960x540!", "-blur", "0x2", "-sharpen", "0x1", "null:",
            ])
        })
    } else {
        Duration::ZERO
    };
    let vp = if hv {
        bench(|| {
            Command::new("sh")
                .args([
                    "-c",
                    &format!(
                    "vips thumbnail '{}' /tmp/_rb.v 960 --height 540 && vips gaussblur /tmp/_rb.v /tmp/_rb2.v 2 && vips sharpen /tmp/_rb2.v /dev/null",
                    ff
                ),
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap();
        })
    } else {
        Duration::ZERO
    };
    let ns = bench(|| {
        let r =
            transform::resize(&dec.pixels, &dec.info, 960, 540, ResizeFilter::Lanczos3).unwrap();
        let b = filters::blur(&r.pixels, &r.info, 2.0).unwrap();
        let _ = filters::sharpen(&b, &r.info, 1.0).unwrap();
    });
    let np = bench(|| {
        let mut g = NodeGraph::new(32 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let r = g.add_node(Box::new(pt::ResizeNode::new(
            s,
            si,
            960,
            540,
            ResizeFilter::Lanczos3,
        )));
        let ri = g.node_info(r).unwrap();
        let b = g.add_node(Box::new(pf::BlurNode::new(r, ri.clone(), 2.0)));
        let sh = g.add_node(Box::new(pf::SharpenNode::new(b, ri, 1.0)));
        let _ = sink::write_jpeg(
            &mut g,
            sh,
            &rasmcore_image::domain::encoder::jpeg::JpegEncodeConfig {
                quality: 85,
                progressive: false,
            },
            None,
        )
        .unwrap();
    });
    let ws = bench(|| {
        use wasm_integration::exports::rasmcore::image::transform::ResizeFilter as WRF;
        let d = bi
            .rasmcore_image_decoder()
            .call_decode(&mut st, &data)
            .unwrap()
            .unwrap();
        let (rp, ri) = wtr
            .call_resize(&mut st, &d.pixels, d.info, 960, 540, WRF::Lanczos3)
            .unwrap()
            .unwrap();
        let bp = wfl.call_blur(&mut st, &rp, ri, 2.0).unwrap().unwrap();
        let _ = wfl.call_sharpen(&mut st, &bp, ri, 1.0).unwrap().unwrap();
    });
    row!("resize+blur+sharpen→JPEG", im, vp, ns, np, ws);

    // ── Chain: resize → brightness → contrast → grayscale → write PNG ──
    let im = if hm {
        bench(|| {
            run_magick(&[
                &ff,
                "-resize",
                "960x540!",
                "-brightness-contrast",
                "20",
                "-brightness-contrast",
                "0x30",
                "-colorspace",
                "Gray",
                "null:",
            ])
        })
    } else {
        Duration::ZERO
    };
    let ns = bench(|| {
        let r =
            transform::resize(&dec.pixels, &dec.info, 960, 540, ResizeFilter::Lanczos3).unwrap();
        let b = filters::brightness(&r.pixels, &r.info, 0.2).unwrap();
        let c = filters::contrast(&b, &r.info, 0.3).unwrap();
        let _ = filters::grayscale(&c, &r.info).unwrap();
    });
    let np = bench(|| {
        let mut g = NodeGraph::new(32 << 20);
        let s = g.add_node(Box::new(source::SourceNode::new(data.clone()).unwrap()));
        let si = g.node_info(s).unwrap();
        let r = g.add_node(Box::new(pt::ResizeNode::new(
            s,
            si,
            960,
            540,
            ResizeFilter::Lanczos3,
        )));
        let ri = g.node_info(r).unwrap();
        let br = g.add_node(Box::new(pf::BrightnessNode::new(r, ri.clone(), 0.2)));
        let ct = g.add_node(Box::new(pf::ContrastNode::new(br, ri.clone(), 0.3)));
        let gr = g.add_node(Box::new(pf::GrayscaleNode::new(ct, ri)));
        let _ = sink::write_png(
            &mut g,
            gr,
            &rasmcore_image::domain::encoder::png::PngEncodeConfig {
                compression_level: 6,
                filter_type: rasmcore_image::domain::encoder::png::PngFilterType::Adaptive,
            },
            None,
        )
        .unwrap();
    });
    let ws = bench(|| {
        use wasm_integration::exports::rasmcore::image::transform::ResizeFilter as WRF;
        let d = bi
            .rasmcore_image_decoder()
            .call_decode(&mut st, &data)
            .unwrap()
            .unwrap();
        let (rp, ri) = wtr
            .call_resize(&mut st, &d.pixels, d.info, 960, 540, WRF::Lanczos3)
            .unwrap()
            .unwrap();
        let bp = wfl.call_brightness(&mut st, &rp, ri, 0.2).unwrap().unwrap();
        let cp = wfl.call_contrast(&mut st, &bp, ri, 0.3).unwrap().unwrap();
        let _ = wfl.call_grayscale(&mut st, &cp, ri).unwrap().unwrap();
    });
    row!("resize+brt+ctr+gray→PNG", im, Duration::ZERO, ns, np, ws);

    println!();
    println!("  SL = Stateless (separate alloc per op, pre-decoded input)");
    println!("  PL = Pipeline (NodeGraph + SpatialCache, includes decode from PNG)");
    println!(
        "============================================================================================\n"
    );
}
