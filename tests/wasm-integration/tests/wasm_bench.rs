//! Three-tier performance comparison:
//!   1. ImageMagick CLI (reference tool — local binary preferred, Docker fallback)
//!   2. Native Rust (domain functions, no WASM)
//!   3. WASM-in-wasmtime (full component model stack)
//!
//! Run with: cargo test -p wasm-integration --test wasm_bench -- --nocapture --ignored

use std::process::Command;
use std::time::{Duration, Instant};

use rasmcore_image::domain::{decoder, encoder, filters, transform};
use wasm_integration::exports::rasmcore::image::transform::ResizeFilter;
use wasm_integration::*;

const WARMUP_ITERS: u32 = 2;
const BENCH_ITERS: u32 = 10;
const DOCKER_IMAGE: &str = "dpokidov/imagemagick:7.1.2-12";

fn fmt_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 {
        format!("{us}µs")
    } else if us < 1_000_000 {
        format!("{:.2}ms", us as f64 / 1000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

// ─── ImageMagick backend detection ───

enum MagickBackend {
    Local(String),  // path to magick binary
    Docker(String), // docker image name
    None,
}

impl MagickBackend {
    fn label(&self) -> &str {
        match self {
            MagickBackend::Local(_) => "ImageMagick (local)",
            MagickBackend::Docker(_) => "ImageMagick (Docker)",
            MagickBackend::None => "ImageMagick",
        }
    }
}

fn detect_magick() -> MagickBackend {
    // Prefer local magick binary — no container overhead
    if let Ok(output) = Command::new("magick")
        .arg("--version")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            let first_line = version.lines().next().unwrap_or("unknown");
            eprintln!("  Using local ImageMagick: {first_line}");
            return MagickBackend::Local("magick".into());
        }
    }

    // Fallback to Docker
    if Command::new("docker")
        .args(["image", "inspect", DOCKER_IMAGE])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
    {
        eprintln!("  Using Docker ImageMagick: {DOCKER_IMAGE}");
        return MagickBackend::Docker(DOCKER_IMAGE.into());
    }

    MagickBackend::None
}

fn run_magick_cmd(backend: &MagickBackend, fixture_dir: &str, args: &[&str]) {
    match backend {
        MagickBackend::Local(bin) => {
            let status = Command::new(bin)
                .args(args)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .expect("failed to run magick");
            assert!(status.success(), "magick command failed: {args:?}");
        }
        MagickBackend::Docker(image) => {
            let vol = format!("{fixture_dir}:/work:ro");
            let mut cmd_args = vec!["run", "--rm", "--entrypoint", "magick", "-v", &vol, image];
            cmd_args.extend_from_slice(args);
            let status = Command::new("docker")
                .args(&cmd_args)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .expect("failed to run docker");
            assert!(status.success(), "docker magick command failed: {args:?}");
        }
        MagickBackend::None => unreachable!(),
    }
}

fn bench_magick(backend: &MagickBackend, fixture_dir: &str, args: &[&str]) -> Duration {
    for _ in 0..WARMUP_ITERS {
        run_magick_cmd(backend, fixture_dir, args);
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        run_magick_cmd(backend, fixture_dir, args);
    }
    start.elapsed() / BENCH_ITERS
}

// ─── Generic bench for native Rust ───

fn bench_native<F: FnMut()>(mut f: F) -> Duration {
    for _ in 0..WARMUP_ITERS {
        f();
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        f();
    }
    start.elapsed() / BENCH_ITERS
}

// ─── Main benchmark ───

#[test]
#[ignore] // Run explicitly: cargo test -p wasm-integration --test wasm_bench -- --ignored --nocapture
fn three_tier_performance_comparison() {
    let fixture_path = fixtures_dir().join("inputs/gradient_64x64.png");
    let fixture_abs = std::fs::canonicalize(&fixture_path).unwrap();
    let fixture_dir = fixture_abs.parent().unwrap().to_str().unwrap().to_string();
    let fixture_file = fixture_abs.to_str().unwrap().to_string();
    let data = load_fixture("gradient_64x64.png");

    // Detect ImageMagick backend
    let backend = detect_magick();
    let has_magick = !matches!(backend, MagickBackend::None);

    // For local magick, args use absolute paths; for Docker, /work/ paths
    let input_path = match &backend {
        MagickBackend::Local(_) => fixture_file.as_str(),
        MagickBackend::Docker(_) => "/work/gradient_64x64.png",
        MagickBackend::None => "",
    };

    // Prepare native inputs
    let native_decoded = decoder::decode(&data).unwrap();

    // Prepare WASM runtime
    let (mut store, bindings) = instantiate_image_component();
    let wasm_decoded = bindings
        .rasmcore_image_decoder()
        .call_decode(&mut store, &data)
        .unwrap()
        .unwrap();

    // ── Decode ──
    let magick_decode = if has_magick {
        bench_magick(&backend, &fixture_dir, &[input_path, "-ping", "null:"])
    } else {
        Duration::ZERO
    };

    let native_decode = bench_native(|| {
        let _ = decoder::decode(&data).unwrap();
    });

    let dec = bindings.rasmcore_image_decoder();
    for _ in 0..WARMUP_ITERS {
        let _ = dec.call_decode(&mut store, &data).unwrap().unwrap();
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = dec.call_decode(&mut store, &data).unwrap().unwrap();
    }
    let wasm_decode = start.elapsed() / BENCH_ITERS;

    // ── Encode ──
    let magick_encode = if has_magick {
        bench_magick(&backend, &fixture_dir, &[input_path, "PNG:/dev/null"])
    } else {
        Duration::ZERO
    };

    let native_encode = bench_native(|| {
        let _ = encoder::encode(&native_decoded.pixels, &native_decoded.info, "png", None).unwrap();
    });

    let enc = bindings.rasmcore_image_encoder();
    for _ in 0..WARMUP_ITERS {
        let _ = enc
            .call_encode(
                &mut store,
                &wasm_decoded.pixels,
                wasm_decoded.info,
                "png",
                None,
            )
            .unwrap()
            .unwrap();
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = enc
            .call_encode(
                &mut store,
                &wasm_decoded.pixels,
                wasm_decoded.info,
                "png",
                None,
            )
            .unwrap()
            .unwrap();
    }
    let wasm_encode = start.elapsed() / BENCH_ITERS;

    // ── Resize ──
    let magick_resize = if has_magick {
        bench_magick(
            &backend,
            &fixture_dir,
            &[input_path, "-resize", "32x16!", "null:"],
        )
    } else {
        Duration::ZERO
    };

    let native_resize = bench_native(|| {
        let _ = transform::resize(
            &native_decoded.pixels,
            &native_decoded.info,
            32,
            16,
            rasmcore_image::domain::types::ResizeFilter::Lanczos3,
        )
        .unwrap();
    });

    let tr = bindings.rasmcore_image_transform();
    for _ in 0..WARMUP_ITERS {
        let _ = tr
            .call_resize(
                &mut store,
                &wasm_decoded.pixels,
                wasm_decoded.info,
                32,
                16,
                ResizeFilter::Lanczos3,
            )
            .unwrap()
            .unwrap();
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = tr
            .call_resize(
                &mut store,
                &wasm_decoded.pixels,
                wasm_decoded.info,
                32,
                16,
                ResizeFilter::Lanczos3,
            )
            .unwrap()
            .unwrap();
    }
    let wasm_resize = start.elapsed() / BENCH_ITERS;

    // ── Blur ──
    let magick_blur = if has_magick {
        bench_magick(
            &backend,
            &fixture_dir,
            &[input_path, "-blur", "0x2", "null:"],
        )
    } else {
        Duration::ZERO
    };

    let native_blur = bench_native(|| {
        let _ = filters::blur(&native_decoded.pixels, &native_decoded.info, 2.0).unwrap();
    });

    let fl = bindings.rasmcore_image_filters();
    for _ in 0..WARMUP_ITERS {
        let _ = fl
            .call_blur(&mut store, &wasm_decoded.pixels, wasm_decoded.info, 2.0)
            .unwrap()
            .unwrap();
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = fl
            .call_blur(&mut store, &wasm_decoded.pixels, wasm_decoded.info, 2.0)
            .unwrap()
            .unwrap();
    }
    let wasm_blur = start.elapsed() / BENCH_ITERS;

    // ── Report ──
    let na = "N/A".to_string();
    let magick_label = backend.label();
    println!();
    println!("================================================================================");
    println!("         Three-Tier Performance Comparison (64x64 PNG, {BENCH_ITERS} iterations)");
    println!("================================================================================");
    println!();
    println!(
        "  {:<12} {:>22}  {:>14}  {:>14}",
        "Operation", magick_label, "Native Rust", "WASM/wasmtime"
    );
    println!(
        "  {:<12} {:>22}  {:>14}  {:>14}",
        "---------", "---------------------", "-----------", "-------------"
    );
    println!(
        "  {:<12} {:>22}  {:>14}  {:>14}",
        "decode",
        if has_magick {
            fmt_duration(magick_decode)
        } else {
            na.clone()
        },
        fmt_duration(native_decode),
        fmt_duration(wasm_decode),
    );
    println!(
        "  {:<12} {:>22}  {:>14}  {:>14}",
        "encode",
        if has_magick {
            fmt_duration(magick_encode)
        } else {
            na.clone()
        },
        fmt_duration(native_encode),
        fmt_duration(wasm_encode),
    );
    println!(
        "  {:<12} {:>22}  {:>14}  {:>14}",
        "resize",
        if has_magick {
            fmt_duration(magick_resize)
        } else {
            na.clone()
        },
        fmt_duration(native_resize),
        fmt_duration(wasm_resize),
    );
    println!(
        "  {:<12} {:>22}  {:>14}  {:>14}",
        "blur",
        if has_magick {
            fmt_duration(magick_blur)
        } else {
            na.clone()
        },
        fmt_duration(native_blur),
        fmt_duration(wasm_blur),
    );
    println!();
    if !has_magick {
        println!("  Note: No ImageMagick found. Install via:");
        println!("    brew install imagemagick   (local, recommended)");
        println!("    docker pull {DOCKER_IMAGE}  (Docker fallback)");
    }
    println!("================================================================================");
    println!();
}
