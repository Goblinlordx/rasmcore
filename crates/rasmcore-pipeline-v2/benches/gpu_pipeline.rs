//! GPU pipeline benchmarks — correctness + throughput for V2 f32 filters.
//!
//! Three benchmark groups:
//! 1. **Correctness — identity**: filters with neutral params produce output == input
//! 2. **Correctness — difference**: filters with non-neutral params produce output != input
//! 3. **Performance**: single-op and multi-op pipeline throughput at 400px and 4K
//!
//! All benchmarks use CPU Filter::compute (the f32 fallback path).
//! GPU executor benchmarks are gated behind a `gpu` feature flag for when
//! the V2 WgpuExecutor is implemented.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rasmcore_pipeline_v2::filters::adjustment::{Brightness, Contrast, Invert};
use rasmcore_pipeline_v2::filters::color::{HueRotate, Saturate};
use rasmcore_pipeline_v2::filters::spatial::GaussianBlur;
use rasmcore_pipeline_v2::fusion::{lower_to_closure, lower_to_f32_lut};
use rasmcore_pipeline_v2::ops::{Filter, PointOpExpr};

// ─── Synthetic Test Images ─────────────────────────────────────────────────

/// Generate a smooth RGBA gradient image (f32, 4 channels per pixel).
/// R varies horizontally, G varies vertically, B varies diagonally.
fn gradient_image(width: u32, height: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut pixels = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let r = x as f32 / (w - 1).max(1) as f32;
            let g = y as f32 / (h - 1).max(1) as f32;
            let b = ((x + y) as f32 / (w + h - 2).max(1) as f32).min(1.0);
            pixels.extend_from_slice(&[r, g, b, 1.0]);
        }
    }
    pixels
}

const SMALL_W: u32 = 400;
const SMALL_H: u32 = 400;
const LARGE_W: u32 = 3840;
const LARGE_H: u32 = 2160;

// ─── Correctness — Identity Tests ──────────────────────────────────────────

/// Assert all pixels match within f32 tolerance.
fn assert_pixels_equal(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut diff_count = 0usize;
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        if diff > tol {
            diff_count += 1;
            max_diff = max_diff.max(diff);
            if diff_count == 1 {
                let pixel = i / 4;
                let ch = i % 4;
                eprintln!(
                    "{label}: first diff at pixel {pixel} ch {ch}: {va:.6} vs {vb:.6} (diff {diff:.6})"
                );
            }
        }
    }
    assert_eq!(
        diff_count, 0,
        "{label}: {diff_count} values differ (max diff {max_diff:.6}, tol {tol})"
    );
}

/// Assert at least some pixels differ.
fn assert_pixels_differ(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let differ = a
        .iter()
        .zip(b.iter())
        .any(|(&va, &vb)| (va - vb).abs() > 1e-6);
    assert!(differ, "{label}: expected output to differ from input");
}

fn correctness_identity(c: &mut Criterion) {
    let mut group = c.benchmark_group("correctness_identity");
    group.sample_size(10);

    let input = gradient_image(SMALL_W, SMALL_H);

    // Brightness with offset=0 should be identity
    group.bench_function("brightness_0", |b| {
        b.iter(|| {
            let f = Brightness { amount: 0.0 };
            let out = f.compute(black_box(&input), SMALL_W, SMALL_H).unwrap();
            assert_pixels_equal(&input, &out, 1e-6, "brightness(0)");
        })
    });

    // Contrast with amount=0 should be identity (factor = 1 + 0 = 1)
    group.bench_function("contrast_0", |b| {
        b.iter(|| {
            let f = Contrast { amount: 0.0 };
            let out = f.compute(black_box(&input), SMALL_W, SMALL_H).unwrap();
            assert_pixels_equal(&input, &out, 1e-6, "contrast(1)");
        })
    });

    // Double invert should roundtrip to identity
    group.bench_function("double_invert", |b| {
        b.iter(|| {
            let f = Invert;
            let once = f.compute(black_box(&input), SMALL_W, SMALL_H).unwrap();
            let twice = f.compute(&once, SMALL_W, SMALL_H).unwrap();
            assert_pixels_equal(&input, &twice, 1e-6, "double_invert");
        })
    });

    // HueRotate with 0 degrees should be identity
    group.bench_function("hue_rotate_0", |b| {
        b.iter(|| {
            let f = HueRotate { degrees: 0.0 };
            let out = f.compute(black_box(&input), SMALL_W, SMALL_H).unwrap();
            assert_pixels_equal(&input, &out, 0.01, "hue_rotate(0)");
        })
    });

    // Saturate with factor=1 should be identity
    group.bench_function("saturate_1", |b| {
        b.iter(|| {
            let f = Saturate { factor: 1.0 };
            let out = f.compute(black_box(&input), SMALL_W, SMALL_H).unwrap();
            assert_pixels_equal(&input, &out, 0.01, "saturate(1)");
        })
    });

    group.finish();
}

// ─── Correctness — Difference Tests ────────────────────────────────────────

fn correctness_difference(c: &mut Criterion) {
    let mut group = c.benchmark_group("correctness_difference");
    group.sample_size(10);

    let input = gradient_image(SMALL_W, SMALL_H);

    group.bench_function("brightness_0.2", |b| {
        b.iter(|| {
            let f = Brightness { amount: 0.2 };
            let out = f.compute(black_box(&input), SMALL_W, SMALL_H).unwrap();
            assert_pixels_differ(&input, &out, "brightness(0.2)");
        })
    });

    group.bench_function("invert_1x", |b| {
        b.iter(|| {
            let f = Invert;
            let out = f.compute(black_box(&input), SMALL_W, SMALL_H).unwrap();
            assert_pixels_differ(&input, &out, "invert(1x)");
        })
    });

    group.bench_function("blur_3.0", |b| {
        b.iter(|| {
            let f = GaussianBlur { radius: 3.0 };
            let out = f.compute(black_box(&input), SMALL_W, SMALL_H).unwrap();
            assert_pixels_differ(&input, &out, "blur(3.0)");
        })
    });

    group.finish();
}

// ─── Performance — Single-Op ───────────────────────────────────────────────

fn perf_single_op(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_op");

    let images: Vec<(&str, u32, u32, Vec<f32>)> = vec![
        ("400px", SMALL_W, SMALL_H, gradient_image(SMALL_W, SMALL_H)),
        ("4K", LARGE_W, LARGE_H, gradient_image(LARGE_W, LARGE_H)),
    ];

    let filters: Vec<(&str, Box<dyn Filter>)> = vec![
        ("brightness", Box::new(Brightness { amount: 0.1 })),
        ("contrast", Box::new(Contrast { amount: 0.5 })),
        ("invert", Box::new(Invert)),
        ("hue_rotate", Box::new(HueRotate { degrees: 45.0 })),
        ("blur_1", Box::new(GaussianBlur { radius: 1.0 })),
    ];

    for (res_name, w, h, input) in &images {
        let pixels = (*w as u64) * (*h as u64);
        group.throughput(Throughput::Elements(pixels));

        for (filter_name, filter) in &filters {
            group.bench_with_input(
                BenchmarkId::new(*filter_name, *res_name),
                input,
                |b, input| {
                    b.iter(|| filter.compute(black_box(input), *w, *h).unwrap());
                },
            );
        }
    }

    group.finish();
}

// ─── Performance — Multi-Op Pipeline ───────────────────────────────────────

fn perf_multi_op(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_op_pipeline");

    let images: Vec<(&str, u32, u32, Vec<f32>)> = vec![
        ("400px", SMALL_W, SMALL_H, gradient_image(SMALL_W, SMALL_H)),
        ("4K", LARGE_W, LARGE_H, gradient_image(LARGE_W, LARGE_H)),
    ];

    // 6-filter pipeline: brightness → contrast → blur → invert → hue_rotate → saturate
    let pipeline: Vec<Box<dyn Filter>> = vec![
        Box::new(Brightness { amount: 0.05 }),
        Box::new(Contrast { amount: 0.2 }),
        Box::new(GaussianBlur { radius: 1.0 }),
        Box::new(Invert),
        Box::new(HueRotate { degrees: 30.0 }),
        Box::new(Saturate { factor: 1.3 }),
    ];

    for (res_name, w, h, input) in &images {
        let pixels = (*w as u64) * (*h as u64);
        group.throughput(Throughput::Elements(pixels));

        group.bench_with_input(
            BenchmarkId::new("6_filters", *res_name),
            input,
            |b, input| {
                b.iter(|| {
                    let mut data = input.clone();
                    for filter in &pipeline {
                        data = filter.compute(black_box(&data), *w, *h).unwrap();
                    }
                    data
                });
            },
        );
    }

    group.finish();
}

// ─── Performance — Fused Point-Op Chain (LUT vs Closure) ──────────────────

fn perf_fused_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_chain");

    // Build a 7-filter expression tree: brightness(+0.05) composed 7 times
    // This simulates a long fused chain with deep recursive evaluation.
    let mut expr = PointOpExpr::Input;
    for _ in 0..7 {
        expr = PointOpExpr::Add(
            Box::new(expr),
            Box::new(PointOpExpr::Constant(0.01)),
        );
    }

    let images: Vec<(&str, u32, u32, Vec<f32>)> = vec![
        ("400px", SMALL_W, SMALL_H, gradient_image(SMALL_W, SMALL_H)),
        ("4K", LARGE_W, LARGE_H, gradient_image(LARGE_W, LARGE_H)),
    ];

    for (res_name, _w, _h, input) in &images {
        let pixels = input.len() / 4;
        group.throughput(Throughput::Elements(pixels as u64));

        // Benchmark: LUT path (new)
        let lut = lower_to_f32_lut(&expr);
        group.bench_with_input(
            BenchmarkId::new("lut", *res_name),
            input,
            |b, input| {
                b.iter(|| {
                    let mut out = input.clone();
                    for pixel in out.chunks_exact_mut(4) {
                        pixel[0] = lut.apply(black_box(pixel[0]));
                        pixel[1] = lut.apply(black_box(pixel[1]));
                        pixel[2] = lut.apply(black_box(pixel[2]));
                    }
                    out
                });
            },
        );

        // Benchmark: closure path (old)
        let closure = lower_to_closure(&expr);
        group.bench_with_input(
            BenchmarkId::new("closure", *res_name),
            input,
            |b, input| {
                b.iter(|| {
                    let mut out = input.clone();
                    for pixel in out.chunks_exact_mut(4) {
                        pixel[0] = closure(black_box(pixel[0]));
                        pixel[1] = closure(black_box(pixel[1]));
                        pixel[2] = closure(black_box(pixel[2]));
                    }
                    out
                });
            },
        );
    }

    group.finish();
}

// ─── Performance — Complex Fused Chain (Gamma + Contrast) ─────────────────

fn perf_fused_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_complex");

    // brightness(+0.1) → contrast(1.5) → gamma(2.2): includes Pow (transcendental)
    let bright = PointOpExpr::Add(
        Box::new(PointOpExpr::Input),
        Box::new(PointOpExpr::Constant(0.1)),
    );
    let contrast = PointOpExpr::Add(
        Box::new(PointOpExpr::Mul(
            Box::new(PointOpExpr::Sub(
                Box::new(bright),
                Box::new(PointOpExpr::Constant(0.5)),
            )),
            Box::new(PointOpExpr::Constant(1.5)),
        )),
        Box::new(PointOpExpr::Constant(0.5)),
    );
    let expr = PointOpExpr::Pow(
        Box::new(PointOpExpr::Max(
            Box::new(contrast),
            Box::new(PointOpExpr::Constant(0.0)),
        )),
        Box::new(PointOpExpr::Constant(1.0 / 2.2)),
    );

    let input = gradient_image(LARGE_W, LARGE_H);
    let pixels = (LARGE_W as u64) * (LARGE_H as u64);
    group.throughput(Throughput::Elements(pixels));

    let lut = lower_to_f32_lut(&expr);
    group.bench_function("lut_4K", |b| {
        b.iter(|| {
            let mut out = input.clone();
            for pixel in out.chunks_exact_mut(4) {
                pixel[0] = lut.apply(black_box(pixel[0]));
                pixel[1] = lut.apply(black_box(pixel[1]));
                pixel[2] = lut.apply(black_box(pixel[2]));
            }
            out
        });
    });

    let closure = lower_to_closure(&expr);
    group.bench_function("closure_4K", |b| {
        b.iter(|| {
            let mut out = input.clone();
            for pixel in out.chunks_exact_mut(4) {
                pixel[0] = closure(black_box(pixel[0]));
                pixel[1] = closure(black_box(pixel[1]));
                pixel[2] = closure(black_box(pixel[2]));
            }
            out
        });
    });

    group.finish();
}

// ─── Main ──────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    correctness_identity,
    correctness_difference,
    perf_single_op,
    perf_multi_op,
    perf_fused_chain,
    perf_fused_complex,
);
criterion_main!(benches);
