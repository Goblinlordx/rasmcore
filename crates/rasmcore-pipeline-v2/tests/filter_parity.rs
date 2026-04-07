//! Filter parity validation — verifies filter implementations against
//! analytical ground truth (formula-based for point/color ops).

use rasmcore_pipeline_v2::ops::Filter;

fn gradient_image(w: u32, h: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = x as f32 / w as f32;
            let g = y as f32 / h as f32;
            let b = ((x + y) as f32 / (w + h) as f32).min(1.0);
            pixels.extend_from_slice(&[r, g, b, 1.0]);
        }
    }
    pixels
}

fn assert_parity(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let diff = actual.iter().zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff <= tolerance, "FAIL {name}: max diff {diff:.8} > tol {tolerance:.8}");
}

const W: u32 = 16;
const H: u32 = 16;
const POINT_TOL: f32 = 1e-6;
const COLOR_TOL: f32 = 1e-4;
const SPATIAL_TOL: f32 = 1e-2;

// ─── Point Ops ──────────────────────────────────────────────────────────────

#[test]
fn parity_brightness() {
    use rasmcore_pipeline_v2::filters::adjustment::Brightness;
    let input = gradient_image(W, H);
    let f = Brightness { amount: 0.3 };
    let actual = f.compute(&input, W, H).unwrap();
    let expected: Vec<f32> = input.chunks(4)
        .flat_map(|px| [px[0] + 0.3, px[1] + 0.3, px[2] + 0.3, px[3]])
        .collect();
    assert_parity("brightness", &actual, &expected, POINT_TOL);
}

#[test]
fn parity_contrast() {
    use rasmcore_pipeline_v2::filters::adjustment::Contrast;
    let input = gradient_image(W, H);
    let amount = 0.5f32;
    let f = Contrast { amount };
    let actual = f.compute(&input, W, H).unwrap();
    // Formula: factor = 1.0 + amount; out = (in - 0.5) * factor + 0.5
    let factor = 1.0 + amount;
    let expected: Vec<f32> = input.chunks(4)
        .flat_map(|px| {
            [(px[0] - 0.5) * factor + 0.5, (px[1] - 0.5) * factor + 0.5,
             (px[2] - 0.5) * factor + 0.5, px[3]]
        }).collect();
    assert_parity("contrast", &actual, &expected, POINT_TOL);
}

#[test]
fn parity_gamma() {
    use rasmcore_pipeline_v2::filters::adjustment::Gamma;
    let input = gradient_image(W, H);
    let f = Gamma { gamma: 2.2 };
    let actual = f.compute(&input, W, H).unwrap();
    let ig = 1.0 / 2.2f32;
    let expected: Vec<f32> = input.chunks(4)
        .flat_map(|px| {
            let r = if px[0] > 0.0 { px[0].powf(ig) } else { 0.0 };
            let g = if px[1] > 0.0 { px[1].powf(ig) } else { 0.0 };
            let b = if px[2] > 0.0 { px[2].powf(ig) } else { 0.0 };
            [r, g, b, px[3]]
        }).collect();
    assert_parity("gamma", &actual, &expected, POINT_TOL);
}

#[test]
fn parity_invert() {
    use rasmcore_pipeline_v2::filters::adjustment::Invert;
    let input = gradient_image(W, H);
    let actual = Invert.compute(&input, W, H).unwrap();
    let expected: Vec<f32> = input.chunks(4)
        .flat_map(|px| [1.0 - px[0], 1.0 - px[1], 1.0 - px[2], px[3]])
        .collect();
    assert_parity("invert", &actual, &expected, POINT_TOL);
}

#[test]
fn parity_exposure() {
    use rasmcore_pipeline_v2::filters::adjustment::Exposure;
    let input = gradient_image(W, H);
    let f = Exposure { ev: 1.0 };
    let actual = f.compute(&input, W, H).unwrap();
    let mul = 2.0f32;
    let expected: Vec<f32> = input.chunks(4)
        .flat_map(|px| [px[0] * mul, px[1] * mul, px[2] * mul, px[3]])
        .collect();
    assert_parity("exposure", &actual, &expected, POINT_TOL);
}

#[test]
fn parity_posterize() {
    use rasmcore_pipeline_v2::filters::adjustment::Posterize;
    let input = gradient_image(W, H);
    let f = Posterize { levels: 4 };
    let actual = f.compute(&input, W, H).unwrap();
    let expected: Vec<f32> = input.chunks(4)
        .flat_map(|px| {
            let q = |v: f32| (v * 4.0).floor().min(3.0) / 3.0;
            [q(px[0]), q(px[1]), q(px[2]), px[3]]
        }).collect();
    assert_parity("posterize", &actual, &expected, POINT_TOL);
}

#[test]
fn parity_solarize() {
    use rasmcore_pipeline_v2::filters::adjustment::Solarize;
    let input = gradient_image(W, H);
    let f = Solarize { threshold: 0.5 };
    let actual = f.compute(&input, W, H).unwrap();
    let expected: Vec<f32> = input.chunks(4)
        .flat_map(|px| {
            let s = |v: f32| if v > 0.5 { 1.0 - v } else { v };
            [s(px[0]), s(px[1]), s(px[2]), px[3]]
        }).collect();
    assert_parity("solarize", &actual, &expected, POINT_TOL);
}

// ─── Color Ops ──────────────────────────────────────────────────────────────

#[test]
fn parity_sepia() {
    use rasmcore_pipeline_v2::filters::color::Sepia;
    let input = gradient_image(W, H);
    let f = Sepia { intensity: 1.0 };
    let actual = f.compute(&input, W, H).unwrap();
    // Formula: blend between original and sepia matrix (clamped to 1.0)
    let expected: Vec<f32> = input.chunks(4)
        .flat_map(|px| {
            let (r, g, b) = (px[0], px[1], px[2]);
            let sr = (0.393 * r + 0.769 * g + 0.189 * b).min(1.0);
            let sg = (0.349 * r + 0.686 * g + 0.168 * b).min(1.0);
            let sb = (0.272 * r + 0.534 * g + 0.131 * b).min(1.0);
            [sr, sg, sb, px[3]]
        }).collect();
    assert_parity("sepia", &actual, &expected, COLOR_TOL);
}

#[test]
fn parity_saturate_zero_is_grayscale() {
    use rasmcore_pipeline_v2::filters::color::Saturate;
    let input = gradient_image(W, H);
    let f = Saturate { factor: 0.0 };
    let actual = f.compute(&input, W, H).unwrap();
    for (i, px) in actual.chunks(4).enumerate() {
        let spread = (px[0] - px[1]).abs().max((px[1] - px[2]).abs());
        assert!(spread < COLOR_TOL,
            "pixel {i}: saturate(0) should be grayscale, got [{:.4},{:.4},{:.4}]",
            px[0], px[1], px[2]);
    }
}

#[test]
fn parity_channel_mixer_identity() {
    use rasmcore_pipeline_v2::filters::color::ChannelMixer;
    let input = gradient_image(W, H);
    let f = ChannelMixer { matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] };
    let actual = f.compute(&input, W, H).unwrap();
    assert_parity("channel_mixer_identity", &actual, &input, POINT_TOL);
}

// ─── Spatial Ops ────────────────────────────────────────────────────────────

#[test]
fn parity_gaussian_blur_solid() {
    use rasmcore_pipeline_v2::filters::spatial::GaussianBlur;
    let solid: Vec<f32> = (0..W * H).flat_map(|_| [0.5f32, 0.3, 0.7, 1.0]).collect();
    let f = GaussianBlur { radius: 3.0 };
    let actual = f.compute(&solid, W, H).unwrap();
    assert_parity("gaussian_blur_solid", &actual, &solid, SPATIAL_TOL);
}

// ─── Enhancement Ops ────────────────────────────────────────────────────────

#[test]
fn parity_auto_level_range() {
    use rasmcore_pipeline_v2::filters::enhancement::AutoLevel;
    let input: Vec<f32> = (0..W * H)
        .flat_map(|i| {
            let v = 0.2 + (i as f32 / (W * H - 1) as f32) * 0.6;
            [v, v, v, 1.0]
        }).collect();
    let actual = AutoLevel.compute(&input, W, H).unwrap();
    let vals: Vec<f32> = actual.chunks(4).map(|px| px[0]).collect();
    let min = vals.iter().cloned().fold(f32::MAX, f32::min);
    let max = vals.iter().cloned().fold(f32::MIN, f32::max);
    assert!(min < 0.01, "auto_level min={min}");
    assert!(max > 0.99, "auto_level max={max}");
}

#[test]
fn parity_equalize_range() {
    use rasmcore_pipeline_v2::filters::enhancement::Equalize;
    let input = gradient_image(W, H);
    let actual = Equalize.compute(&input, W, H).unwrap();
    let vals: Vec<f32> = actual.chunks(4).map(|px| px[0]).collect();
    let min = vals.iter().cloned().fold(f32::MAX, f32::min);
    let max = vals.iter().cloned().fold(f32::MIN, f32::max);
    assert!(min < 0.1, "equalize min={min}");
    assert!(max > 0.9, "equalize max={max}");
}

// ─── Report ─────────────────────────────────────────────────────────────────

#[test]
fn parity_matrix_report() {
    let factories = rasmcore_pipeline_v2::registered_filter_factories();
    let validated = [
        "brightness", "contrast", "gamma", "invert", "exposure",
        "posterize", "solarize",
        "sepia", "saturate", "channel_mixer",
        "gaussian_blur",
        "auto_level", "equalize",
    ];
    let total = factories.len();
    let validated_count = factories.iter()
        .filter(|n| validated.contains(&n.as_ref()))
        .count();

    eprintln!("\n=== Filter Parity Matrix ===");
    eprintln!("Validated: {validated_count}/{total} ({:.0}%)",
        validated_count as f32 / total as f32 * 100.0);
    eprintln!("Point ops: brightness, contrast, gamma, invert, exposure, posterize, solarize");
    eprintln!("Color ops: sepia, saturate, channel_mixer");
    eprintln!("Spatial: gaussian_blur (solid invariant)");
    eprintln!("Enhancement: auto_level, equalize (range checks)");
    eprintln!("Remaining: {} filters untested (generators, draw, morphology, etc.)", total - validated_count);
}
