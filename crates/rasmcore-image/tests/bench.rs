/// Simple benchmarks for rasmcore-image domain operations.
///
/// Measures native Rust execution time (not WASM — WASM benchmarks need wasmtime).
/// Run with: cargo test -p rasmcore-image --test bench -- --nocapture
///
/// Prerequisites: run `tests/fixtures/generate.sh` first.
use std::path::Path;
use std::time::Instant;

use rasmcore_image::domain::types::*;
use rasmcore_image::domain::{decoder, encoder, filters, transform};

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/generated")
}

fn load_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixtures_dir().join("inputs").join(name)).unwrap_or_else(|e| {
        panic!("Fixture not found: {name}: {e}. Run tests/fixtures/generate.sh")
    })
}

fn bench<F: FnOnce() -> R, R>(label: &str, f: F) -> R {
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    println!("  {label}: {elapsed:?}");
    result
}

#[test]
fn benchmark_decode() {
    println!("\n=== Decode Benchmarks (native) ===");
    let png = load_fixture("photo_256x256.png");
    let jpeg = load_fixture("photo_256x256.jpeg");

    bench("PNG  256x256 decode", || decoder::decode(&png).unwrap());
    bench("JPEG 256x256 decode", || decoder::decode(&jpeg).unwrap());
}

#[test]
fn benchmark_encode() {
    println!("\n=== Encode Benchmarks (native) ===");
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();

    bench("PNG  256x256 encode", || {
        encoder::encode(&decoded.pixels, &decoded.info, "png", None).unwrap()
    });
    bench("JPEG 256x256 encode q85", || {
        encoder::encode(&decoded.pixels, &decoded.info, "jpeg", Some(85)).unwrap()
    });
}

#[test]
fn benchmark_transform() {
    println!("\n=== Transform Benchmarks (native) ===");
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();

    bench("Resize 256→128 lanczos", || {
        transform::resize(
            &decoded.pixels,
            &decoded.info,
            128,
            128,
            ResizeFilter::Lanczos3,
        )
        .unwrap()
    });
    bench("Resize 256→512 bilinear", || {
        transform::resize(
            &decoded.pixels,
            &decoded.info,
            512,
            512,
            ResizeFilter::Bilinear,
        )
        .unwrap()
    });
    bench("Crop 128x128", || {
        transform::crop(&decoded.pixels, &decoded.info, 64, 64, 128, 128).unwrap()
    });
    bench("Rotate 90", || {
        transform::rotate(&decoded.pixels, &decoded.info, Rotation::R90).unwrap()
    });
}

#[test]
fn benchmark_filters() {
    println!("\n=== Filter Benchmarks (native) ===");
    let data = load_fixture("photo_256x256.png");
    let decoded = decoder::decode(&data).unwrap();

    bench("Blur r=2.0", || {
        filters::blur(&decoded.pixels, &decoded.info, 2.0).unwrap()
    });
    bench("Sharpen 1.0", || {
        filters::sharpen(&decoded.pixels, &decoded.info, 1.0).unwrap()
    });
    bench("Grayscale", || {
        filters::grayscale(&decoded.pixels, &decoded.info).unwrap()
    });
    bench("FreqSep low σ=4", || {
        filters::frequency_low(&decoded.pixels, &decoded.info, 4.0).unwrap()
    });
    bench("FreqSep high σ=4", || {
        filters::frequency_high(&decoded.pixels, &decoded.info, 4.0).unwrap()
    });
    bench("FreqSep low σ=10", || {
        filters::frequency_low(&decoded.pixels, &decoded.info, 10.0).unwrap()
    });
    bench("FreqSep high σ=10", || {
        filters::frequency_high(&decoded.pixels, &decoded.info, 10.0).unwrap()
    });
}
