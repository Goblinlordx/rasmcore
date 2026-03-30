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

use rasmcore_image::domain::filters::{
    AdaptiveMethod, BlendMode, BokehShape, MertensParams, MorphShape, NlmAlgorithm, NlmParams,
};
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

        // TIFF decode
        let tiff_path = ensure_input("tiff", size);
        let tiff_data = std::fs::read(&tiff_path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("tiff/rasmcore", size),
            &tiff_data,
            |b, data| {
                b.iter(|| decoder::decode(data).unwrap());
            },
        );

        if ref_tools::has_tool("magick") {
            let p = tiff_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("tiff/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_decode(&p));
            });
        }

        if ref_tools::has_tool("vips") {
            let p = tiff_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("tiff/libvips", size), |b| {
                b.iter(|| ref_tools::vips_decode(&p));
            });
        }

        // GIF decode
        let gif_path = ensure_input("gif", size);
        let gif_data = std::fs::read(&gif_path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("gif/rasmcore", size),
            &gif_data,
            |b, data| {
                b.iter(|| decoder::decode(data).unwrap());
            },
        );

        if ref_tools::has_tool("magick") {
            let p = gif_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("gif/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_decode(&p));
            });
        }

        // BMP decode
        let bmp_path = ensure_input("bmp", size);
        let bmp_data = std::fs::read(&bmp_path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("bmp/rasmcore", size),
            &bmp_data,
            |b, data| {
                b.iter(|| decoder::decode(data).unwrap());
            },
        );

        if ref_tools::has_tool("magick") {
            let p = bmp_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("bmp/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_decode(&p));
            });
        }

        // QOI decode
        let qoi_path = ensure_input("qoi", size);
        let qoi_data = std::fs::read(&qoi_path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("qoi/rasmcore", size),
            &qoi_data,
            |b, data| {
                b.iter(|| decoder::decode(data).unwrap());
            },
        );

        // AVIF decode (skip if codec not compiled in)
        let avif_path = ensure_input("avif", size);
        let avif_data = std::fs::read(&avif_path).unwrap();

        if decoder::decode(&avif_data).is_ok() {
            group.bench_with_input(
                BenchmarkId::new("avif/rasmcore", size),
                &avif_data,
                |b, data| {
                    b.iter(|| decoder::decode(data).unwrap());
                },
            );

            if ref_tools::has_tool("magick") {
                let p = avif_path.to_str().unwrap().to_string();
                group.bench_function(BenchmarkId::new("avif/imagemagick", size), |b| {
                    b.iter(|| ref_tools::magick_decode(&p));
                });
            }

            if ref_tools::has_tool("vips") {
                let p = avif_path.to_str().unwrap().to_string();
                group.bench_function(BenchmarkId::new("avif/libvips", size), |b| {
                    b.iter(|| ref_tools::vips_decode(&p));
                });
            }
        }
    }

    // HEVC decode (requires nonfree-hevc feature)
    #[cfg(feature = "nonfree-hevc")]
    {
        for &size in &[256u32, 512] {
            let heic_path = ensure_input("heic", size);
            let heic_data = std::fs::read(&heic_path).unwrap();
            let pixel_count = (size * size) as u64;
            group.throughput(Throughput::Elements(pixel_count));

            group.bench_with_input(
                BenchmarkId::new("heic/rasmcore", size),
                &heic_data,
                |b, data| {
                    b.iter(|| decoder::decode(data).unwrap());
                },
            );

            if ref_tools::has_tool("ffmpeg") {
                let p = heic_path.to_str().unwrap().to_string();
                group.bench_function(BenchmarkId::new("heic/ffmpeg", size), |b| {
                    b.iter(|| {
                        ref_tools::run_timed(
                            "ffmpeg",
                            &["-i", &p, "-f", "rawvideo", "-y", "/dev/null"],
                        )
                    });
                });
            }
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
        let raw_dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        // Ensure 8-bit for encoders that don't support 16-bit
        let dec = if raw_dec.info.format == PixelFormat::Rgb16
            || raw_dec.info.format == PixelFormat::Rgba16
        {
            let pixels_8: Vec<u8> = raw_dec
                .pixels
                .chunks_exact(2)
                .map(|c| (u16::from_le_bytes([c[0], c[1]]) >> 8) as u8)
                .collect();
            let info_8 = ImageInfo {
                format: if raw_dec.info.format == PixelFormat::Rgb16 {
                    PixelFormat::Rgb8
                } else {
                    PixelFormat::Rgba8
                },
                ..raw_dec.info
            };
            DecodedImage {
                pixels: pixels_8,
                info: info_8,
                icc_profile: raw_dec.icc_profile,
            }
        } else {
            raw_dec
        };

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

        // TIFF encode (lossless)
        group.bench_function(BenchmarkId::new("tiff/rasmcore", size), |b| {
            b.iter(|| encoder::encode(&dec.pixels, &dec.info, "tiff", None).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("tiff/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_encode(&p, "tiff", None));
            });
        }

        if ref_tools::has_tool("vips") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("tiff/libvips", size), |b| {
                b.iter(|| ref_tools::vips_encode(&p, "tiff", None));
            });
        }

        // AVIF encode (lossy, quality sweep — skip if codec not compiled in)
        if encoder::encode(&dec.pixels, &dec.info, "avif", Some(85)).is_ok() {
            for &quality in &[75u8, 85, 95] {
                let label = format!("{size}/q{quality}");

                group.bench_function(BenchmarkId::new("avif/rasmcore", &label), |b| {
                    b.iter(|| {
                        encoder::encode(&dec.pixels, &dec.info, "avif", Some(quality)).unwrap()
                    });
                });

                if ref_tools::has_tool("magick") {
                    let p = png_path_str.clone();
                    group.bench_function(BenchmarkId::new("avif/imagemagick", &label), |b| {
                        b.iter(|| ref_tools::magick_encode(&p, "avif", Some(quality)));
                    });
                }
            }
        }

        // QOI encode (lossless, no quality param)
        group.bench_function(BenchmarkId::new("qoi/rasmcore", size), |b| {
            b.iter(|| encoder::encode(&dec.pixels, &dec.info, "qoi", None).unwrap());
        });
    }

    // HEVC encode (requires nonfree-hevc feature)
    #[cfg(feature = "nonfree-hevc")]
    {
        for &size in &[256u32, 512] {
            let png_path = ensure_input("png", size);
            let png_data = std::fs::read(&png_path).unwrap();
            let dec = decoder::decode(&png_data).unwrap();

            group.throughput(Throughput::Elements((size * size) as u64));

            for &quality in &[75u8, 85, 95] {
                let label = format!("{size}/q{quality}");

                group.bench_function(BenchmarkId::new("heic/rasmcore", &label), |b| {
                    b.iter(|| {
                        encoder::encode(&dec.pixels, &dec.info, "heic", Some(quality)).unwrap()
                    });
                });
            }
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

    // Bilateral and CLAHE (expensive — includes 1024 for bilateral)
    for &size in &[256u32, 512, 1024] {
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

// ─── Spatial Filter Benchmarks ──────────────────────────────────────────

fn spatial_filter_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial");

    // --- Convolution and blur filters at 256/512 ---
    for &size in &[256u32, 512] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        // Ensure RGB8
        let (pixels, info) = if dec.info.format == PixelFormat::Rgba8 {
            let rgb: Vec<u8> = dec
                .pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            (
                rgb,
                ImageInfo {
                    format: PixelFormat::Rgb8,
                    ..dec.info
                },
            )
        } else {
            (dec.pixels.clone(), dec.info)
        };

        // Grayscale for filters that require Gray8
        let gray_dec = filters::grayscale(&pixels, &info).unwrap();

        group.throughput(Throughput::Elements((size * size) as u64));

        // Median (radius=3)
        let px = pixels.clone();
        let inf = info;
        group.bench_function(BenchmarkId::new("median/rasmcore", size), |b| {
            b.iter(|| filters::median(&px, &inf, 3).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("median/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(&p, &["-statistic", "Median", "7x7"], "png", None)
                });
            });
        }

        // Bokeh blur — disc (radius=5)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("bokeh_disc/rasmcore", size), |b| {
            b.iter(|| filters::bokeh_blur(&px, &inf, 5, BokehShape::Disc).unwrap());
        });

        // Bokeh blur — hexagon (radius=5)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("bokeh_hex/rasmcore", size), |b| {
            b.iter(|| filters::bokeh_blur(&px, &inf, 5, BokehShape::Hexagon).unwrap());
        });

        // Convolve (5x5 sharpen kernel)
        let px = pixels.clone();
        #[rustfmt::skip]
        let kernel: Vec<f32> = vec![
             0.0, -1.0, -1.0, -1.0,  0.0,
            -1.0,  2.0, -4.0,  2.0, -1.0,
            -1.0, -4.0, 13.0, -4.0, -1.0,
            -1.0,  2.0, -4.0,  2.0, -1.0,
             0.0, -1.0, -1.0, -1.0,  0.0,
        ];
        group.bench_function(BenchmarkId::new("convolve_5x5/rasmcore", size), |b| {
            b.iter(|| filters::convolve(&px, &inf, &kernel, 5, 5, 1.0).unwrap());
        });

        // Gaussian blur CV (sigma=2.0)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("gaussian_blur_cv/rasmcore", size), |b| {
            b.iter(|| filters::gaussian_blur_cv(&px, &inf, 2.0).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("gaussian_blur_cv/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-blur", "0x2"], "png", None));
            });
        }

        // Guided filter (Gray8, radius=4, epsilon=0.01)
        group.bench_function(BenchmarkId::new("guided_filter/rasmcore", size), |b| {
            b.iter(|| filters::guided_filter(&gray_dec.pixels, &gray_dec.info, 4, 0.01).unwrap());
        });

        // NLM denoise (Gray8, defaults) — expensive, sample_size(10)
        group.sample_size(10);
        let nlm_params = NlmParams {
            h: 10.0,
            patch_size: 7,
            search_size: 21,
            algorithm: NlmAlgorithm::OpenCv,
        };
        group.bench_function(BenchmarkId::new("nlm_denoise/rasmcore", size), |b| {
            b.iter(|| filters::nlm_denoise(&gray_dec.pixels, &gray_dec.info, &nlm_params).unwrap());
        });
        group.sample_size(100); // reset
    }

    // --- Convolve and gaussian_blur_cv also at 1024 ---
    {
        let size = 1024u32;
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        let (pixels, info) = if dec.info.format == PixelFormat::Rgba8 {
            let rgb: Vec<u8> = dec
                .pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            (
                rgb,
                ImageInfo {
                    format: PixelFormat::Rgb8,
                    ..dec.info
                },
            )
        } else {
            (dec.pixels.clone(), dec.info)
        };

        group.throughput(Throughput::Elements((size * size) as u64));

        // Convolve 5x5 at 1024
        let px = pixels.clone();
        let inf = info;
        #[rustfmt::skip]
        let kernel: Vec<f32> = vec![
             0.0, -1.0, -1.0, -1.0,  0.0,
            -1.0,  2.0, -4.0,  2.0, -1.0,
            -1.0, -4.0, 13.0, -4.0, -1.0,
            -1.0,  2.0, -4.0,  2.0, -1.0,
             0.0, -1.0, -1.0, -1.0,  0.0,
        ];
        group.bench_function(BenchmarkId::new("convolve_5x5/rasmcore", size), |b| {
            b.iter(|| filters::convolve(&px, &inf, &kernel, 5, 5, 1.0).unwrap());
        });

        // Gaussian blur CV at 1024
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("gaussian_blur_cv/rasmcore", size), |b| {
            b.iter(|| filters::gaussian_blur_cv(&px, &inf, 2.0).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("gaussian_blur_cv/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-blur", "0x2"], "png", None));
            });
        }
    }

    group.finish();
}

// ─── Morphological Filter Benchmarks ────────────────────────────────────

fn morphological_filter_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphological");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        let (pixels, info) = if dec.info.format == PixelFormat::Rgba8 {
            let rgb: Vec<u8> = dec
                .pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            (
                rgb,
                ImageInfo {
                    format: PixelFormat::Rgb8,
                    ..dec.info
                },
            )
        } else {
            (dec.pixels.clone(), dec.info)
        };

        group.throughput(Throughput::Elements((size * size) as u64));

        // Erode (rect 3x3)
        let px = pixels.clone();
        let inf = info;
        group.bench_function(BenchmarkId::new("erode/rasmcore", size), |b| {
            b.iter(|| filters::erode(&px, &inf, 3, MorphShape::Rect).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("erode/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &["-morphology", "Erode", "Square:1"],
                        "png",
                        None,
                    )
                });
            });
        }

        // Dilate (rect 3x3)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("dilate/rasmcore", size), |b| {
            b.iter(|| filters::dilate(&px, &inf, 3, MorphShape::Rect).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("dilate/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &["-morphology", "Dilate", "Square:1"],
                        "png",
                        None,
                    )
                });
            });
        }

        // Morph open (rect 3x3)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("morph_open/rasmcore", size), |b| {
            b.iter(|| filters::morph_open(&px, &inf, 3, MorphShape::Rect).unwrap());
        });

        // Morph close (rect 3x3)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("morph_close/rasmcore", size), |b| {
            b.iter(|| filters::morph_close(&px, &inf, 3, MorphShape::Rect).unwrap());
        });
    }

    group.finish();
}

// ─── Enhancement Filter Benchmarks ──────────────────────────────────────

fn enhancement_filter_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("enhancement");
    group.sample_size(10); // these are expensive multi-pass algorithms

    for &size in &[256u32, 512] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();

        // Ensure RGB8 for filters
        let (pixels, info) = if dec.info.format == PixelFormat::Rgba8 {
            let rgb: Vec<u8> = dec
                .pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            (
                rgb,
                ImageInfo {
                    format: PixelFormat::Rgb8,
                    ..dec.info
                },
            )
        } else {
            (dec.pixels.clone(), dec.info)
        };

        group.throughput(Throughput::Elements((size * size) as u64));

        // Retinex SSR (sigma=80)
        let px = pixels.clone();
        let inf = info;
        group.bench_function(BenchmarkId::new("retinex_ssr/rasmcore", size), |b| {
            b.iter(|| filters::retinex_ssr(&px, &inf, 80.0).unwrap());
        });

        // Dehaze (patch_radius=7, omega=0.95, t_min=0.1)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("dehaze/rasmcore", size), |b| {
            b.iter(|| filters::dehaze(&px, &inf, 7, 0.95, 0.1).unwrap());
        });

        // Clarity (amount=0.5, sigma=2.0)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("clarity/rasmcore", size), |b| {
            b.iter(|| filters::clarity(&px, &inf, 0.5, 2.0).unwrap());
        });

        // Frequency separation — low-pass (sigma=4)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("freq_low/rasmcore", size), |b| {
            b.iter(|| filters::frequency_low(&px, &inf, 4.0).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("freq_low/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-blur", "0x4"], "png", None));
            });
        }

        // Frequency separation — high-pass (sigma=4)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("freq_high/rasmcore", size), |b| {
            b.iter(|| filters::frequency_high(&px, &inf, 4.0).unwrap());
        });

        if ref_tools::has_tool("magick") {
            // ImageMagick high-pass: original - blur + 50% gray
            let p = png_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("freq_high/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &[
                            "(",
                            "+clone",
                            "-blur",
                            "0x4",
                            ")",
                            "-compose",
                            "Mathematics",
                            "-define",
                            "compose:args=0,-1,1,0.5",
                            "-composite",
                        ],
                        "png",
                        None,
                    )
                });
            });
        }
    }

    // Mertens fusion at 256 only (very expensive, 3 input images)
    {
        let size = 256u32;
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();

        let (pixels, info) = if dec.info.format == PixelFormat::Rgba8 {
            let rgb: Vec<u8> = dec
                .pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            (
                rgb,
                ImageInfo {
                    format: PixelFormat::Rgb8,
                    ..dec.info
                },
            )
        } else {
            (dec.pixels.clone(), dec.info)
        };

        // Generate 3 "exposures" by adjusting brightness
        let dark: Vec<u8> = pixels.iter().map(|&p| (p as f32 * 0.5) as u8).collect();
        let bright: Vec<u8> = pixels
            .iter()
            .map(|&p| ((p as f32 * 1.5).min(255.0)) as u8)
            .collect();

        let images: Vec<&[u8]> = vec![dark.as_slice(), pixels.as_slice(), bright.as_slice()];
        let params = MertensParams {
            contrast_weight: 1.0,
            saturation_weight: 1.0,
            exposure_weight: 1.0,
        };

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_function(BenchmarkId::new("mertens_fusion/rasmcore", size), |b| {
            b.iter(|| filters::mertens_fusion(&images, &info, &params).unwrap());
        });
    }

    group.finish();
}

// ─── Geometric Warp Benchmarks ──────────────────────────────────────────

fn geometric_warp_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        let (pixels, info) = if dec.info.format == PixelFormat::Rgba8 {
            let rgb: Vec<u8> = dec
                .pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            (
                rgb,
                ImageInfo {
                    format: PixelFormat::Rgb8,
                    ..dec.info
                },
            )
        } else {
            (dec.pixels.clone(), dec.info)
        };

        group.throughput(Throughput::Elements((size * size) as u64));

        // Perspective warp — moderate rotation (~10% corner shift)
        // Homography matrix for a mild perspective transform
        let shift = size as f64 * 0.1;
        let s = size as f64;
        // Simple perspective: map unit square corners with slight shift
        // Using a pre-computed homography for corners shifted by ~10%
        let matrix: [f64; 9] = {
            // Source corners: (0,0), (s,0), (s,s), (0,s)
            // Dest corners:   (shift,0), (s-shift,0), (s,s), (0,s)
            // Approximate with a simple projective matrix
            let sx = (s - 2.0 * shift) / s;
            [sx, 0.0, shift, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        };

        let px = pixels.clone();
        let inf = info;
        group.bench_function(BenchmarkId::new("perspective_warp/rasmcore", size), |b| {
            b.iter(|| filters::perspective_warp(&px, &inf, &matrix, size, size).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            let args_str = format!(
                "0,0,{shift},0  {s},0,{},0  {s},{s},{s},{s}  0,{s},0,{s}",
                s - shift
            );
            group.bench_function(
                BenchmarkId::new("perspective_warp/imagemagick", size),
                |b| {
                    b.iter(|| {
                        ref_tools::magick_pipeline(
                            &p,
                            &["-distort", "Perspective", &args_str],
                            "png",
                            None,
                        )
                    });
                },
            );
        }

        // Displacement map — barrel distortion
        let pixel_count = (size * size) as usize;
        let cx = size as f32 / 2.0;
        let cy = size as f32 / 2.0;
        let k = 0.0001_f32; // mild barrel distortion coefficient
        let mut map_x = vec![0.0_f32; pixel_count];
        let mut map_y = vec![0.0_f32; pixel_count];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r2 = dx * dx + dy * dy;
                let factor = 1.0 + k * r2;
                map_x[(y * size + x) as usize] = cx + dx * factor;
                map_y[(y * size + x) as usize] = cy + dy * factor;
            }
        }

        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("displacement_map/rasmcore", size), |b| {
            b.iter(|| filters::displacement_map(&px, &inf, &map_x, &map_y).unwrap());
        });

        // Affine — 15-degree rotation around center
        let angle_rad = 15.0_f64 * std::f64::consts::PI / 180.0;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let cx64 = size as f64 / 2.0;
        let cy64 = size as f64 / 2.0;
        // Affine: rotate around center = translate(-cx,-cy) * rotate * translate(cx,cy)
        let tx = cx64 - cos_a * cx64 + sin_a * cy64;
        let ty = cy64 - sin_a * cx64 - cos_a * cy64;
        let affine_matrix: [f64; 6] = [cos_a, -sin_a, tx, sin_a, cos_a, ty];

        let px = pixels.clone();
        let bg = [0u8, 0, 0];
        group.bench_function(BenchmarkId::new("affine/rasmcore", size), |b| {
            b.iter(|| transform::affine(&px, &inf, &affine_matrix, size, size, &bg).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("affine/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-distort", "SRT", "15"], "png", None));
            });
        }
    }

    group.finish();
}

// ─── Transform Benchmarks ───────────────────────────────────────────────

fn transform_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        group.throughput(Throughput::Elements((size * size) as u64));

        // Crop — center 50%
        let crop_size = size / 2;
        let crop_offset = size / 4;
        group.bench_function(BenchmarkId::new("crop/rasmcore", size), |b| {
            b.iter(|| {
                transform::crop(
                    &dec.pixels,
                    &dec.info,
                    crop_offset,
                    crop_offset,
                    crop_size,
                    crop_size,
                )
                .unwrap()
            });
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            let geom = format!("{crop_size}x{crop_size}+{crop_offset}+{crop_offset}");
            group.bench_function(BenchmarkId::new("crop/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-crop", &geom], "png", None));
            });
        }

        // Rotate 90
        group.bench_function(BenchmarkId::new("rotate_90/rasmcore", size), |b| {
            b.iter(|| transform::rotate(&dec.pixels, &dec.info, Rotation::R90).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("rotate_90/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-rotate", "90"], "png", None));
            });
        }

        // Flip horizontal
        group.bench_function(BenchmarkId::new("flip_horizontal/rasmcore", size), |b| {
            b.iter(|| transform::flip(&dec.pixels, &dec.info, FlipDirection::Horizontal).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("flip_horizontal/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-flop"], "png", None));
            });
        }

        // Pad — add 10px border
        let fill = [0u8, 0, 0];
        group.bench_function(BenchmarkId::new("pad/rasmcore", size), |b| {
            b.iter(|| transform::pad(&dec.pixels, &dec.info, 10, 10, 10, 10, &fill).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("pad/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &["-bordercolor", "black", "-border", "10"],
                        "png",
                        None,
                    )
                });
            });
        }

        // Trim
        group.bench_function(BenchmarkId::new("trim/rasmcore", size), |b| {
            b.iter(|| transform::trim(&dec.pixels, &dec.info, 10).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("trim/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-trim"], "png", None));
            });
        }
    }

    group.finish();
}

// ─── Color & Tone Benchmarks ────────────────────────────────────────────

fn color_tone_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_tone");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        // Ensure RGB8
        let (pixels, info) = if dec.info.format == PixelFormat::Rgba8 {
            let rgb: Vec<u8> = dec
                .pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect();
            (
                rgb,
                ImageInfo {
                    format: PixelFormat::Rgb8,
                    ..dec.info
                },
            )
        } else {
            (dec.pixels.clone(), dec.info)
        };

        group.throughput(Throughput::Elements((size * size) as u64));

        // Brightness (+25%)
        let px = pixels.clone();
        let inf = info;
        group.bench_function(BenchmarkId::new("brightness/rasmcore", size), |b| {
            b.iter(|| filters::brightness(&px, &inf, 0.25).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("brightness/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(&p, &["-evaluate", "Add", "25%"], "png", None)
                });
            });
        }

        // Contrast (+0.5)
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("contrast/rasmcore", size), |b| {
            b.iter(|| filters::contrast(&px, &inf, 0.5).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("contrast/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(&p, &["-sigmoidal-contrast", "5,50%"], "png", None)
                });
            });
        }

        // Hue rotate
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("hue_rotate/rasmcore", size), |b| {
            b.iter(|| filters::hue_rotate(&px, &inf, 90.0).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            // IM -modulate: brightness,saturation,hue (hue is 0-200 scale, 100=no change)
            // 90 degrees = 25% of 360 = +50 on IM scale = 150
            group.bench_function(BenchmarkId::new("hue_rotate/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(&p, &["-modulate", "100,100,150"], "png", None)
                });
            });
        }

        // Saturate
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("saturate/rasmcore", size), |b| {
            b.iter(|| filters::saturate(&px, &inf, 1.5).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            // IM -modulate: 100=no brightness change, 150=1.5x saturation, 100=no hue change
            group.bench_function(BenchmarkId::new("saturate/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(&p, &["-modulate", "100,150,100"], "png", None)
                });
            });
        }

        // Sepia
        let px = pixels.clone();
        group.bench_function(BenchmarkId::new("sepia/rasmcore", size), |b| {
            b.iter(|| filters::sepia(&px, &inf, 0.8).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("sepia/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-sepia-tone", "80%"], "png", None));
            });
        }
    }

    group.finish();
}

// ─── Edge Detection Benchmarks ──────────────────────────────────────────

fn edge_detection_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();
        let png_path_str = png_path.to_str().unwrap().to_string();

        // Convert to grayscale for edge detectors
        let rgb_info = ImageInfo {
            width: dec.info.width,
            height: dec.info.height,
            format: PixelFormat::Rgb8,
            color_space: dec.info.color_space,
        };
        let rgb_pixels = if dec.info.format == PixelFormat::Rgba8 {
            dec.pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<u8>>()
        } else {
            dec.pixels.clone()
        };
        let gray_dec = filters::grayscale(&rgb_pixels, &rgb_info).unwrap();
        let gray_pixels = gray_dec.pixels;
        let gray_info = gray_dec.info;

        group.throughput(Throughput::Elements((size * size) as u64));

        // Sobel
        group.bench_function(BenchmarkId::new("sobel/rasmcore", size), |b| {
            b.iter(|| filters::sobel(&gray_pixels, &gray_info).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("sobel/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-edge", "1"], "png", None));
            });
        }

        // Scharr
        group.bench_function(BenchmarkId::new("scharr/rasmcore", size), |b| {
            b.iter(|| filters::scharr(&gray_pixels, &gray_info).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("scharr/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &["-colorspace", "Gray", "-morphology", "Convolve", "Scharr"],
                        "png",
                        None,
                    )
                });
            });
        }

        // Laplacian
        group.bench_function(BenchmarkId::new("laplacian/rasmcore", size), |b| {
            b.iter(|| filters::laplacian(&gray_pixels, &gray_info).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("laplacian/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &["-colorspace", "Gray", "-morphology", "Convolve", "Laplacian:0"],
                        "png",
                        None,
                    )
                });
            });
        }

        // Canny
        group.bench_function(BenchmarkId::new("canny/rasmcore", size), |b| {
            b.iter(|| filters::canny(&gray_pixels, &gray_info, 50.0, 150.0).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new("canny/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &["-colorspace", "Gray", "-canny", "0x1+10%+30%"],
                        "png",
                        None,
                    )
                });
            });
        }
    }

    group.finish();
}

// ─── Threshold & Analysis Benchmarks ────────────────────────────────────

fn threshold_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();

        let rgb_info = ImageInfo {
            width: dec.info.width,
            height: dec.info.height,
            format: PixelFormat::Rgb8,
            color_space: dec.info.color_space,
        };
        let rgb_pixels = if dec.info.format == PixelFormat::Rgba8 {
            dec.pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<u8>>()
        } else {
            dec.pixels.clone()
        };
        let gray_dec = filters::grayscale(&rgb_pixels, &rgb_info).unwrap();
        let gray_pixels = gray_dec.pixels;
        let gray_info = gray_dec.info;

        group.throughput(Throughput::Elements((size * size) as u64));

        // Otsu threshold
        group.bench_function(BenchmarkId::new("otsu/rasmcore", size), |b| {
            b.iter(|| filters::otsu_threshold(&gray_pixels, &gray_info).unwrap());
        });

        if ref_tools::has_tool("magick") {
            let p = png_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("otsu/imagemagick", size), |b| {
                b.iter(|| {
                    ref_tools::magick_pipeline(
                        &p,
                        &["-colorspace", "Gray", "-auto-threshold", "OTSU"],
                        "png",
                        None,
                    )
                });
            });
        }

        // Triangle threshold
        group.bench_function(BenchmarkId::new("triangle/rasmcore", size), |b| {
            b.iter(|| filters::triangle_threshold(&gray_pixels, &gray_info).unwrap());
        });

        // Adaptive threshold (Gaussian, block_size=11)
        group.bench_function(BenchmarkId::new("adaptive_gaussian/rasmcore", size), |b| {
            b.iter(|| {
                filters::adaptive_threshold(
                    &gray_pixels,
                    &gray_info,
                    255,
                    AdaptiveMethod::Gaussian,
                    11,
                    2.0,
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

// ─── Alpha, Blend, Vignette, Pyramid Benchmarks ────────────────────────

fn alpha_blend_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("alpha_blend");

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_data = std::fs::read(&png_path).unwrap();
        let dec = decoder::decode(&png_data).unwrap();

        let base_info = ImageInfo {
            width: dec.info.width,
            height: dec.info.height,
            format: dec.info.format,
            color_space: dec.info.color_space,
        };

        // Build RGBA8 input (add alpha channel if needed)
        let rgba_info = ImageInfo {
            format: PixelFormat::Rgba8,
            ..base_info
        };
        let rgba_pixels = if base_info.format == PixelFormat::Rgba8 {
            dec.pixels.clone()
        } else {
            dec.pixels
                .chunks_exact(3)
                .flat_map(|c| [c[0], c[1], c[2], 200])
                .collect::<Vec<u8>>()
        };

        // RGB8 for vignette/pyramid
        let rgb_info = ImageInfo {
            format: PixelFormat::Rgb8,
            ..base_info
        };
        let rgb_pixels = if base_info.format == PixelFormat::Rgba8 {
            dec.pixels
                .chunks_exact(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<u8>>()
        } else {
            dec.pixels.clone()
        };

        group.throughput(Throughput::Elements((size * size) as u64));

        // Premultiply
        group.bench_function(BenchmarkId::new("premultiply/rasmcore", size), |b| {
            b.iter(|| filters::premultiply(&rgba_pixels, &rgba_info).unwrap());
        });

        // Unpremultiply
        group.bench_function(BenchmarkId::new("unpremultiply/rasmcore", size), |b| {
            b.iter(|| filters::unpremultiply(&rgba_pixels, &rgba_info).unwrap());
        });

        // Blend modes (self-blend for consistency)
        group.bench_function(BenchmarkId::new("blend_multiply/rasmcore", size), |b| {
            b.iter(|| {
                filters::blend(
                    &rgba_pixels,
                    &rgba_info,
                    &rgba_pixels,
                    &rgba_info,
                    BlendMode::Multiply,
                )
                .unwrap()
            });
        });

        if ref_tools::has_tool("magick") {
            let p = png_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("blend_multiply/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_composite(&p, &p, "Multiply", "png"));
            });
        }

        group.bench_function(BenchmarkId::new("blend_screen/rasmcore", size), |b| {
            b.iter(|| {
                filters::blend(
                    &rgba_pixels,
                    &rgba_info,
                    &rgba_pixels,
                    &rgba_info,
                    BlendMode::Screen,
                )
                .unwrap()
            });
        });

        if ref_tools::has_tool("magick") {
            let p = png_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("blend_screen/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_composite(&p, &p, "Screen", "png"));
            });
        }

        // Vignette powerlaw
        group.bench_function(BenchmarkId::new("vignette_powerlaw/rasmcore", size), |b| {
            b.iter(|| {
                filters::vignette_powerlaw(&rgb_pixels, &rgb_info, 0.5, 2.0, size, size, 0, 0)
                    .unwrap()
            });
        });

        if ref_tools::has_tool("magick") {
            let p = png_path.to_str().unwrap().to_string();
            group.bench_function(BenchmarkId::new("vignette/imagemagick", size), |b| {
                b.iter(|| ref_tools::magick_pipeline(&p, &["-vignette", "0x10"], "png", None));
            });
        }

        // Pyramid down/up (Gray8 only)
        {
            let gray_dec = filters::grayscale(&rgb_pixels, &rgb_info).unwrap();
            let gp = gray_dec.pixels;
            let gi = gray_dec.info;

            group.bench_function(BenchmarkId::new("pyr_down/rasmcore", size), |b| {
                b.iter(|| filters::pyr_down(&gp, &gi).unwrap());
            });

            group.bench_function(BenchmarkId::new("pyr_up/rasmcore", size), |b| {
                b.iter(|| filters::pyr_up(&gp, &gi).unwrap());
            });
        }
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

// ─── CLI Comparison Benchmarks ──────────────────────────────────────────
//
// Fair apples-to-apples comparison: all tools measured as process spawns
// (including rasmcore). This captures process startup, library init, file I/O,
// and actual codec work — the same conditions a user experiences.

use std::sync::OnceLock;

/// Build the bench_codec example binary (release mode) once per benchmark run.
fn bench_codec_bin() -> &'static str {
    static BIN: OnceLock<String> = OnceLock::new();
    BIN.get_or_init(|| {
        // The binary should already be built by `cargo bench` since it compiles
        // the whole crate. Locate it relative to the current executable.
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let target_dir = manifest_dir.join("../../target/release/examples/bench_codec");
        if target_dir.exists() {
            return target_dir.to_str().unwrap().to_string();
        }
        // Fallback: try to build it
        let status = std::process::Command::new("cargo")
            .args([
                "build",
                "--release",
                "--example",
                "bench_codec",
                "-p",
                "rasmcore-image",
            ])
            .status()
            .expect("failed to build bench_codec");
        assert!(status.success(), "cargo build bench_codec failed");
        target_dir.to_str().unwrap().to_string()
    })
}

fn cli_decoder_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("cli_decoder");
    group.sample_size(10); // process spawns are slow

    let rasmcore_bin = bench_codec_bin();

    for &size in &[256u32, 512, 1024] {
        let pixel_count = (size * size) as u64;
        group.throughput(Throughput::Elements(pixel_count));

        let codecs: &[(&str, &str)] = &[
            ("jpeg", "jpeg"),
            ("png", "png"),
            ("webp", "webp"),
            ("tiff", "tiff"),
            ("gif", "gif"),
            ("bmp", "bmp"),
            ("qoi", "qoi"),
        ];

        for &(codec, ext) in codecs {
            let path = ensure_input(ext, size);
            let path_str = path.to_str().unwrap().to_string();

            // rasmcore CLI
            let bin = rasmcore_bin.to_string();
            let p = path_str.clone();
            group.bench_function(BenchmarkId::new(format!("{codec}/rasmcore"), size), |b| {
                b.iter(|| {
                    let out = std::process::Command::new(&bin)
                        .args(["decode", &p])
                        .output()
                        .unwrap();
                    assert!(out.status.success());
                });
            });

            // ImageMagick
            if ref_tools::has_tool("magick") {
                let p = path_str.clone();
                group.bench_function(
                    BenchmarkId::new(format!("{codec}/imagemagick"), size),
                    |b| {
                        b.iter(|| ref_tools::magick_decode(&p));
                    },
                );
            }

            // Codec-specific reference tools
            match codec {
                "jpeg" if ref_tools::has_tool("djpeg") => {
                    let p = path_str.clone();
                    group.bench_function(BenchmarkId::new("jpeg/libjpeg-turbo", size), |b| {
                        b.iter(|| ref_tools::djpeg_decode(&p));
                    });
                }
                "webp" if ref_tools::has_tool("dwebp") => {
                    let p = path_str.clone();
                    group.bench_function(BenchmarkId::new("webp/dwebp", size), |b| {
                        b.iter(|| ref_tools::dwebp_decode(&p));
                    });
                }
                "png" | "tiff" if ref_tools::has_tool("vips") => {
                    let p = path_str.clone();
                    group.bench_function(BenchmarkId::new(format!("{codec}/libvips"), size), |b| {
                        b.iter(|| ref_tools::vips_decode(&p));
                    });
                }
                _ => {}
            }
        }
    }

    group.finish();
}

fn cli_encoder_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("cli_encoder");
    group.sample_size(10);

    let rasmcore_bin = bench_codec_bin();

    for &size in &[256u32, 512, 1024] {
        let png_path = ensure_input("png", size);
        let png_path_str = png_path.to_str().unwrap().to_string();

        group.throughput(Throughput::Elements((size * size) as u64));

        // Lossy codecs with quality sweep
        // Lossy codecs: (display_name, encode_format, qualities)
        let lossy_codecs: &[(&str, &str, &[u8])] = &[
            ("jpeg-quality", "jpeg", &[75, 85, 95]),
            ("jpeg-turbo", "jpeg-turbo", &[75, 85, 95]),
            ("webp", "webp", &[75, 85, 95]),
        ];

        for &(display, enc_fmt, qualities) in lossy_codecs {
            for &quality in qualities {
                let label = format!("{size}/q{quality}");
                let q_str = quality.to_string();

                // rasmcore CLI
                let bin = rasmcore_bin.to_string();
                let p = png_path_str.clone();
                let fmt = enc_fmt.to_string();
                group.bench_function(
                    BenchmarkId::new(format!("{display}/rasmcore"), &label),
                    |b| {
                        b.iter(|| {
                            let out = std::process::Command::new(&bin)
                                .args(["encode", &p, &fmt, &q_str])
                                .stdout(std::process::Stdio::null())
                                .output()
                                .unwrap();
                            assert!(out.status.success());
                        });
                    },
                );

                // ImageMagick (same reference for both jpeg modes)
                if (display == "jpeg-quality" || display == "webp") && ref_tools::has_tool("magick")
                {
                    let magick_fmt = if display.starts_with("jpeg") {
                        "jpeg"
                    } else {
                        enc_fmt
                    };
                    let p = png_path_str.clone();
                    group.bench_function(
                        BenchmarkId::new(format!("{display}/imagemagick"), &label),
                        |b| {
                            b.iter(|| ref_tools::magick_encode(&p, magick_fmt, Some(quality)));
                        },
                    );
                }

                // cwebp for WebP
                if display == "webp" && ref_tools::has_tool("cwebp") {
                    let p = png_path_str.clone();
                    group.bench_function(BenchmarkId::new("webp/cwebp", &label), |b| {
                        b.iter(|| ref_tools::cwebp_encode(&p, quality));
                    });
                }
            }
        }

        // Lossless codecs
        for &codec in &["png", "tiff", "qoi"] {
            // rasmcore CLI
            let bin = rasmcore_bin.to_string();
            let p = png_path_str.clone();
            group.bench_function(BenchmarkId::new(format!("{codec}/rasmcore"), size), |b| {
                b.iter(|| {
                    let out = std::process::Command::new(&bin)
                        .args(["encode", &p, codec])
                        .stdout(std::process::Stdio::null())
                        .output()
                        .unwrap();
                    assert!(out.status.success());
                });
            });

            // ImageMagick (not for QOI)
            if codec != "qoi" && ref_tools::has_tool("magick") {
                let p = png_path_str.clone();
                group.bench_function(
                    BenchmarkId::new(format!("{codec}/imagemagick"), size),
                    |b| {
                        b.iter(|| ref_tools::magick_encode(&p, codec, None));
                    },
                );
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    decoder_benchmarks,
    encoder_benchmarks,
    filter_benchmarks,
    spatial_filter_benchmarks,
    morphological_filter_benchmarks,
    enhancement_filter_benchmarks,
    geometric_warp_benchmarks,
    transform_benchmarks,
    color_tone_benchmarks,
    edge_detection_benchmarks,
    threshold_benchmarks,
    alpha_blend_benchmarks,
    pipeline_benchmarks,
    cli_decoder_benchmarks,
    cli_encoder_benchmarks
);
criterion_main!(benches);
