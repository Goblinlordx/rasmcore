//! Performance benchmark suite — rasmcore vs reference tools.
//!
//! Run the full suite:     cargo bench --bench perf
//! Run decoder only:       cargo bench --bench perf -- decoder
//! Run encoder/jpeg:       cargo bench --bench perf -- encoder/jpeg
//! Run filter/bilateral:   cargo bench --bench perf -- filter/bilateral
//! Run pipeline only:      cargo bench --bench perf -- pipeline
//!
//! Results are written to target/criterion/ in JSON format.
//! Use scripts/bench-report.py to generate a Markdown comparison table.

mod ref_tools;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::path::Path;

use rasmcore_image::domain::types::*;
use rasmcore_image::domain::{decoder, encoder, filters, transform};

// ─── Fixture Helpers ─────────────────────────────────────────────────────

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/generated/inputs")
}

fn load_fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()))
}

/// Ensure a test image exists at the given size. Generates via ImageMagick or Rust fallback.
fn ensure_input(fmt: &str, size: u32) -> std::path::PathBuf {
    let name = format!("bench_{size}x{size}.{fmt}");
    let path = std::env::temp_dir().join(&name);
    if path.exists() {
        return path;
    }
    let src = fixture_dir().join("photo_256x256.png");
    if src.exists() && ref_tools::has_tool("magick") {
        let sz = format!("{size}x{size}!");
        let _ = std::process::Command::new("magick")
            .args([
                "convert",
                src.to_str().unwrap(),
                "-resize",
                &sz,
                path.to_str().unwrap(),
            ])
            .output();
    } else {
        let png_data = load_fixture("photo_256x256.png");
        let dec = decoder::decode(&png_data).unwrap();
        let resized =
            transform::resize(&dec.pixels, &dec.info, size, size, ResizeFilter::Lanczos3).unwrap();
        let encoded = encoder::encode(&resized.pixels, &resized.info, fmt, Some(95)).unwrap();
        std::fs::write(&path, &encoded).unwrap();
    }
    path
}

// ─── Decoder Benchmarks ──────────────────────────────────────────────────

fn decoder_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("decoder");

    for &size in &[256u32, 512, 1024] {
        let pixel_count = (size * size) as u64;
        group.throughput(Throughput::Elements(pixel_count));

        // JPEG decode
        let jpeg_path = ensure_input("jpeg", size);
        let jpeg_data = std::fs::read(&jpeg_path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("jpeg/rasmcore", size),
            &jpeg_data,
            |b, data| {
                b.iter(|| decoder::decode(data).unwrap());
            },
        );

        if ref_tools::has_tool("djpeg") {
            let p = jpeg_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("jpeg/libjpeg-turbo", size), |b| {
                b.iter(|| ref_tools::djpeg_decode(&p));
            });
        }

        if ref_tools::has_tool("magick") {
            let p = jpeg_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("jpeg/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_decode(&p));
            });
        }

        if ref_tools::has_tool("vips") {
            let p = jpeg_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("jpeg/libvips", size), |b| {
                b.iter(|| ref_tools::vips_decode(&p));
            });
        }

        // PNG decode
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("png/rasmcore", size),
            &png_data,
            |b, data| {
                b.iter(|| decoder::decode(data).unwrap());
            },
        );

        if ref_tools::has_tool("magick") {
            let p = png_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("png/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_decode(&p));
            });
        }

        if ref_tools::has_tool("vips") {
            let p = png_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("png/libvips", size), |b| {
                b.iter(|| ref_tools::vips_decode(&p));
            });
        }

        // WebP decode
        let webp_path = ensure_input("webp", size);
        let webp_data = std::fs::read(&webp_path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("webp/rasmcore", size),
            &webp_data,
            |b, data| {
                b.iter(|| decoder::decode(data).unwrap());
            },
        );

        if ref_tools::has_tool("dwebp") {
            let p = webp_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("webp/dwebp", size), |b| {
                b.iter(|| ref_tools::dwebp_decode(&p));
            });
        }

        if ref_tools::has_tool("magick") {
            let p = webp_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("webp/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_decode(&p));
            });
        }
    }

    group.finish();
}

// ─── Encoder Benchmarks ──────────────────────────────────────────────────

fn encoder_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        group.throughput(Throughput::Elements((size * size) as u64));

        for &quality in &[75u8, 85, 95] {
            let label = format!("{size}/q{quality}");

            // JPEG encode
            group.bench_function(BenchmarkId::new("jpeg/rasmcore", &label), |b| {
                b.iter(|| encoder::encode(&dec.pixels, &dec.info, "jpeg", Some(quality)).unwrap());
            });

            if ref_tools::has_tool("magick") {
                let p = png_path_str.clone();
                group.bench_function(BenchmarkId::new("jpeg/imagemagick", &label), |b| {
                    b.iter(|| ref_tools::magick_encode(&p, "jpeg", Some(quality)));
                });
            }

            if ref_tools::has_tool("vips") {
                let p = png_path_str.clone();
                group.bench_function(BenchmarkId::new("jpeg/libvips", &label), |b| {
                    b.iter(|| ref_tools::vips_encode(&p, "jpeg", Some(quality)));
                });
            }

            // WebP encode
            group.bench_function(BenchmarkId::new("webp/rasmcore", &label), |b| {
                b.iter(|| encoder::encode(&dec.pixels, &dec.info, "webp", Some(quality)).unwrap());
            });

            if ref_tools::has_tool("cwebp") {
                let p = png_path_str.clone();
                group.bench_function(BenchmarkId::new("webp/cwebp", &label), |b| {
                    b.iter(|| ref_tools::cwebp_encode(&p, quality));
                });
            }
        }

        // PNG encode (no quality param)
        group.bench_function(BenchmarkId::new("png/rasmcore", size), |b| {
            b.iter(|| encoder::encode(&dec.pixels, &dec.info, "png", None).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("png/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_encode(&p, "png", None));
            });
        }

        if ref_tools::has_tool("vips") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("png/libvips", size), |b| {
                b.iter(|| ref_tools::vips_encode(&p, "png", None));
            });
        }
    }

    group.finish();
}

// ─── Filter/Transform Benchmarks ─────────────────────────────────────────

fn filter_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        group.throughput(Throughput::Elements((size * size) as u64));

        // Resize (Lanczos3, 50% downscale)
        let half = size / 2;
        group.bench_function(BenchmarkId::new("resize/rasmcore", size), |b| {
            b.iter(|| {
                transform::resize(&dec.pixels, &dec.info, half, half, ResizeFilter::Lanczos3)
                    .unwrap()
            });
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            let sz = format!("{half}x{half}!");
            group.bench_function(BenchmarkId::new("resize/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &["-filter", "Lanczos", "-resize", &sz],
                        "png",
                        None,
                    )
                });
            });
        }

        // Blur (Gaussian, radius=3)
        group.bench_function(BenchmarkId::new("blur/rasmcore", size), |b| {
            b.iter(|| filters::blur(&dec.pixels, &dec.info, 3.0).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("blur/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-blur", "0x3"], "png", None));
            });
        }

        // Sharpen
        group.bench_function(BenchmarkId::new("sharpen/rasmcore", size), |b| {
            b.iter(|| filters::sharpen(&dec.pixels, &dec.info, 1.0).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("sharpen/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-sharpen", "0x1"], "png", None));
            });
        }

        // Grayscale
        group.bench_function(BenchmarkId::new("grayscale/rasmcore", size), |b| {
            b.iter(|| filters::grayscale(&dec.pixels, &dec.info).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("grayscale/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-colorspace", "Gray"], "png", None));
            });
        }
    }

    // Bilateral and CLAHE (expensive — smaller sizes only)
    for &size in &[256u32, 512] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();

        // Convert to grayscale for bilateral/CLAHE — handle RGBA by dropping alpha first
        let rgb_pixels = if dec.info.format == PixelFormat::Rgba8 {
            dec.pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<u8>>()
        } else {
            dec.pixels.clone()
        };
        let rgb_info = ImageInfo {
            format: PixelFormat::Rgb8,
            ..dec.info
        };
        let gray_dec = filters::grayscale(&rgb_pixels, &rgb_info).unwrap();

        group.bench_function(BenchmarkId::new("bilateral/rasmcore", size), |b| {
            b.iter(|| filters::bilateral(&gray_dec.pixels, &gray_dec.info, 9, 75.0, 75.0).unwrap());
        });

        group.bench_function(BenchmarkId::new("clahe/rasmcore", size), |b| {
            b.iter(|| filters::clahe(&gray_dec.pixels, &gray_dec.info, 2.0, 8).unwrap());
        });
    }

    group.finish();
}

// ─── Pipeline Benchmarks ─────────────────────────────────────────────────

fn pipeline_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");
    group.sample_size(10); // pipelines are slow

    for &size in &[256u32, 1024] {
        let jpeg_path = ensure_input("jpeg", size);
        let jpeg_data = std::fs::read(&jpeg_path).unwrap();
        let jpeg_path_str = jpeg_path.to_str().unwrap().to_string();
        let half = size / 2;

        // Pipeline A: decode JPEG → resize 50% → sharpen → encode JPEG q85
        group.bench_function(
            BenchmarkId::new("A_decode-resize-sharpen-encode/rasmcore", size),
            |b| {
                b.iter(|| {
                    let dec = decoder::decode(&jpeg_data).unwrap();
                    let resized = transform::resize(
                        &dec.pixels,
                        &dec.info,
                        half,
                        half,
                        ResizeFilter::Lanczos3,
                    )
                    .unwrap();
                    let sharp = filters::sharpen(&resized.pixels, &resized.info, 1.0).unwrap();
                    encoder::encode(&sharp, &resized.info, "jpeg", Some(85)).unwrap()
                });
            },
        );

        if ref_tools::has_tool("magick") {
            let p = jpeg_path_str.clone();
            let sz = format!("{half}x{half}!");
            group.bench_function(
                BenchmarkId::new("A_decode-resize-sharpen-encode/imagemagick", size),
                |b| {
                    b.iter(|| {
                        ref_tools::magick_pipeline(
                            &p,
                            &["-filter", "Lanczos", "-resize", &sz, "-sharpen", "0x1"],
                            "jpeg",
                            Some(85),
                        )
                    });
                },
            );
        }

        if ref_tools::has_tool("vipsthumbnail") {
            let p = jpeg_path_str.clone();
            group.bench_function(
                BenchmarkId::new("A_decode-resize-sharpen-encode/libvips", size),
                |b| {
                    b.iter(|| {
                        let tmp = std::env::temp_dir().join("rasmcore_bench_vips_pipe_a.jpeg");
                        let tmp_q = format!("{}[Q=85]", tmp.to_str().unwrap());
                        let sz = half.to_string();
                        let _ = std::process::Command::new("vipsthumbnail")
                            .args([&p, "-s", &sz, "--sharpen=mild", "-o", &tmp_q])
                            .output();
                        std::fs::read(&tmp).unwrap_or_default()
                    });
                },
            );
        }

        // Pipeline C: decode JPEG → grayscale → resize 25% → encode WebP q80
        let quarter = size / 4;
        group.bench_function(
            BenchmarkId::new("C_decode-gray-resize-webp/rasmcore", size),
            |b| {
                b.iter(|| {
                    let dec = decoder::decode(&jpeg_data).unwrap();
                    let gray = filters::grayscale(&dec.pixels, &dec.info).unwrap();
                    let resized = transform::resize(
                        &gray.pixels,
                        &gray.info,
                        quarter,
                        quarter,
                        ResizeFilter::Lanczos3,
                    )
                    .unwrap();
                    // WebP needs RGB — expand gray to RGB
                    let rgb: Vec<u8> = resized.pixels.iter().flat_map(|&g| [g, g, g]).collect();
                    let rgb_info = ImageInfo {
                        width: quarter,
                        height: quarter,
                        format: PixelFormat::Rgb8,
                        color_space: resized.info.color_space,
                    };
                    encoder::encode(&rgb, &rgb_info, "webp", Some(80)).unwrap()
                });
            },
        );

        if ref_tools::has_tool("magick") {
            let p = jpeg_path_str.clone();
            let sz = format!("{quarter}x{quarter}!");
            group.bench_function(
                BenchmarkId::new("C_decode-gray-resize-webp/imagemagick", size),
                |b| {
                    b.iter(|| {
                        ref_tools::magick_pipeline(
                            &p,
                            &["-colorspace", "Gray", "-resize", &sz],
                            "webp",
                            Some(80),
                        )
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    decoder_benchmarks,
    encoder_benchmarks,
    filter_benchmarks,
    pipeline_benchmarks
);
criterion_main!(benches);
