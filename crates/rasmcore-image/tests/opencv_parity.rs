//! OpenCV parity tests — validates filters against OpenCV 4.13 reference output.
//!
//! Canonical test image set (7 images × 3 filters = 21 reference comparisons):
//!   gradient_128    — smooth horizontal gradient (low frequency)
//!   checker_128     — 8px checkerboard (high frequency, edge preservation)
//!   noisy_flat_128  — Gaussian noise σ=20 around 128 (denoising)
//!   sharp_edges_128 — step function + rectangle (edge ringing)
//!   photo_128       — sinusoidal + noise (realistic content)
//!   flat_128        — all 128 (degenerate case)
//!   highcontrast_128— sinusoidal extreme darks/brights
//!
//! Reference: OpenCV 4.13.0 via opencv-contrib-python-headless in venv.
//! Fixtures stored as raw grayscale bytes in tests/fixtures/opencv/.

use rasmcore_image::domain::filters;
use rasmcore_image::domain::types::{ColorSpace, ImageInfo, PixelFormat};

fn load_fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/opencv/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {path}: {e}"))
}

fn mae(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn max_error(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn info_128() -> ImageInfo {
    ImageInfo {
        width: 128,
        height: 128,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    }
}

// ─── Canonical Test Image Names ───────────────────────────────────────────

const TEST_IMAGES: &[&str] = &[
    "gradient_128",
    "checker_128",
    "noisy_flat_128",
    "sharp_edges_128",
    "photo_128",
    "flat_128",
    "highcontrast_128",
];

// ─── CLAHE Parity ─────────────────────────────────────────────────────────

#[test]
fn clahe_all_images_match_opencv() {
    let info = info_128();
    let mut total_mae = 0.0;
    let mut total_max = 0u8;
    let mut count = 0;

    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_clahe.raw"));
        let ours = filters::clahe(&input, &info, 2.0, 8).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        total_mae += e;
        total_max = total_max.max(m);
        count += 1;

        eprintln!("CLAHE {name:20}: MAE={e:.4}, max_err={m}");

        assert!(
            e < 0.5,
            "CLAHE {name}: MAE {e:.4} > 0.5 — not pixel-exact with OpenCV"
        );
        assert!(
            m <= 1,
            "CLAHE {name}: max error {m} > 1"
        );
    }

    let avg_mae = total_mae / count as f64;
    eprintln!("\nCLAHE summary: avg_MAE={avg_mae:.4}, worst_max_err={total_max}");
}

// ─── Bilateral Parity ─────────────────────────────────────────────────────

#[test]
fn bilateral_all_images_match_opencv() {
    let info = info_128();
    let mut total_mae = 0.0;
    let mut total_max = 0u8;
    let mut count = 0;

    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_bilateral.raw"));
        let ours = filters::bilateral(&input, &info, 9, 75.0, 75.0).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        total_mae += e;
        total_max = total_max.max(m);
        count += 1;

        eprintln!("Bilateral {name:20}: MAE={e:.4}, max_err={m}");

        assert!(
            e < 0.5,
            "Bilateral {name}: MAE {e:.4} > 0.5 — not pixel-exact with OpenCV"
        );
        assert!(
            m <= 1,
            "Bilateral {name}: max error {m} > 1"
        );
    }

    let avg_mae = total_mae / count as f64;
    eprintln!("\nBilateral summary: avg_MAE={avg_mae:.4}, worst_max_err={total_max}");
}

// ─── Guided Filter Parity ─────────────────────────────────────────────────

#[test]
fn guided_all_images_match_opencv() {
    let info = info_128();
    let mut total_mae = 0.0;
    let mut total_max = 0u8;
    let mut count = 0;

    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_guided.raw"));
        let ours = filters::guided_filter(&input, &info, 4, 0.01).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        total_mae += e;
        total_max = total_max.max(m);
        count += 1;

        eprintln!("Guided {name:20}: MAE={e:.4}, max_err={m}");

        assert!(
            e < 0.5,
            "Guided {name}: MAE {e:.4} > 0.5 — not pixel-exact with OpenCV"
        );
        assert!(
            m <= 2,
            "Guided {name}: max error {m} > 2"
        );
    }

    let avg_mae = total_mae / count as f64;
    eprintln!("\nGuided summary: avg_MAE={avg_mae:.4}, worst_max_err={total_max}");
}

// ─── Color Science Parity ─────────────────────────────────────────────────

use rasmcore_image::domain::color_spaces;

#[test]
fn lab_conversion_matches_opencv() {
    let input = load_fixture("color_gradient_128_rgb.raw");
    let reference_lab = load_fixture("color_gradient_128_lab.raw");

    let info = ImageInfo {
        width: 128, height: 128,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };

    let our_lab = color_spaces::image_rgb_to_lab(&input, &info).unwrap();

    // OpenCV Lab: L in [0,255] (L*255/100), a/b in [0,255] (offset +128)
    // Our Lab: L in [0,100], a/b in [-128,127]
    // Convert ours to OpenCV convention for comparison
    let n = (128 * 128) as usize;
    let mut max_err = 0u8;
    let mut total_err = 0.0f64;
    for i in 0..n {
        let our_l = (our_lab[i * 3] * 255.0 / 100.0).round().clamp(0.0, 255.0) as u8;
        let our_a = (our_lab[i * 3 + 1] + 128.0).round().clamp(0.0, 255.0) as u8;
        let our_b = (our_lab[i * 3 + 2] + 128.0).round().clamp(0.0, 255.0) as u8;

        let ref_l = reference_lab[i * 3];
        let ref_a = reference_lab[i * 3 + 1];
        let ref_b = reference_lab[i * 3 + 2];

        let dl = (our_l as i16 - ref_l as i16).unsigned_abs() as u8;
        let da = (our_a as i16 - ref_a as i16).unsigned_abs() as u8;
        let db = (our_b as i16 - ref_b as i16).unsigned_abs() as u8;
        let err = dl.max(da).max(db);
        max_err = max_err.max(err);
        total_err += dl as f64 + da as f64 + db as f64;
    }
    let mae = total_err / (n * 3) as f64;

    eprintln!("Lab vs OpenCV: MAE={mae:.4}, max_err={max_err}");
    assert!(mae < 2.0, "Lab MAE {mae:.4} > 2.0 vs OpenCV");
    assert!(max_err <= 3, "Lab max error {max_err} > 3 vs OpenCV");
}

#[test]
fn perspective_warp_matches_opencv() {
    let input = load_fixture("gradient_128_gray.raw");
    let reference = load_fixture("gradient_128_perspective.raw");

    let info = ImageInfo {
        width: 128, height: 128,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let src = [(0.0f32, 0.0), (127.0, 0.0), (127.0, 127.0), (0.0, 127.0)];
    let dst = [(10.0f32, 10.0), (117.0, 5.0), (120.0, 122.0), (8.0, 120.0)];

    let ours = color_spaces::perspective_warp(&input, &info, &src, &dst, 128, 128).unwrap();

    // Compare only non-zero pixels (border pixels may differ in how out-of-bounds is handled)
    let mut total_err = 0.0f64;
    let mut count = 0u64;
    let mut max_err = 0u8;
    for i in 0..ours.len() {
        if reference[i] > 0 && ours[i] > 0 {
            let err = (ours[i] as i16 - reference[i] as i16).unsigned_abs() as u8;
            max_err = max_err.max(err);
            total_err += err as f64;
            count += 1;
        }
    }
    let mae = if count > 0 { total_err / count as f64 } else { 0.0 };

    eprintln!("Perspective vs OpenCV: MAE={mae:.4}, max_err={max_err}, compared={count}/{}", ours.len());
    assert!(mae < 2.0, "Perspective MAE {mae:.4} > 2.0 vs OpenCV");
}

#[test]
fn gray_world_wb_matches_reference() {
    let input = load_fixture("blue_tinted_64_rgb.raw");
    let reference = load_fixture("blue_tinted_64_grayworld.raw");

    let info = ImageInfo {
        width: 64, height: 64,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };

    let ours = color_spaces::white_balance_gray_world(&input, &info).unwrap();

    let error = mae(&ours, &reference);
    let max_err = max_error(&ours, &reference);

    eprintln!("Gray-world WB: MAE={error:.4}, max_err={max_err}");
    assert!(error < 1.0, "Gray-world MAE {error:.4} > 1.0");
    assert!(max_err <= 1, "Gray-world max error {max_err} > 1");
}

// ─── OKLab + Bradford Parity (colour-science 0.4.7) ─────────────────────

#[test]
fn oklab_matches_colour_science() {
    // Reference values from colour-science 0.4.7 (colour.XYZ_to_Oklab)
    let test_cases: &[(&str, (f32, f32, f32), (f32, f32, f32))] = &[
        ("red",    (1.0, 0.0, 0.0), (0.627926, 0.224888, 0.125805)),
        ("green",  (0.0, 1.0, 0.0), (0.866452, -0.233921, 0.179422)),
        ("blue",   (0.0, 0.0, 1.0), (0.452033, -0.032352, -0.311621)),
        ("gray",   (0.5, 0.5, 0.5), (0.598182, 0.000001, -0.000068)),
        ("custom", (0.5, 0.3, 0.8), (0.541759, 0.089488, -0.166634)),
        ("white",  (1.0, 1.0, 1.0), (1.000002, 0.000002, -0.000114)),
        ("black",  (0.0, 0.0, 0.0), (0.000000, 0.000000, 0.000000)),
    ];

    let mut max_err: f32 = 0.0;
    for (name, (r, g, b), (ref_l, ref_a, ref_b)) in test_cases {
        let (l, a, bv) = color_spaces::rgb_to_oklab(*r, *g, *b);
        let err = (l - ref_l).abs().max((a - ref_a).abs()).max((bv - ref_b).abs());
        max_err = max_err.max(err);
        eprintln!("OKLab {name:8}: L={l:.6} a={a:.6} b={bv:.6} (ref: {ref_l:.6},{ref_a:.6},{ref_b:.6}) err={err:.6}");
    }

    assert!(max_err < 0.001, "OKLab max error {max_err:.6} > 0.001 vs colour-science");
}

#[test]
fn bradford_matches_colour_science() {
    // Reference: colour-science 0.4.7, Von Kries method with Bradford transform
    // D65->D50: XYZ(0.5, 0.4, 0.3) → (0.518086, 0.405866, 0.226963)
    let (x, y, z) = color_spaces::bradford_adapt(
        0.5, 0.4, 0.3,
        color_spaces::Illuminant::D65,
        color_spaces::Illuminant::D50,
    );
    eprintln!("Bradford D65->D50: X={x:.6} Y={y:.6} Z={z:.6}");
    eprintln!("  Reference:       X=0.518086 Y=0.405866 Z=0.226963");
    let err = (x - 0.518086f32).abs().max((y - 0.405866).abs()).max((z - 0.226963).abs());
    assert!(err < 0.01, "Bradford D65->D50 error {err:.6} > 0.01");

    // D65->A: XYZ(0.5, 0.4, 0.3) → (0.606172, 0.425971, 0.096777)
    let (x, y, z) = color_spaces::bradford_adapt(
        0.5, 0.4, 0.3,
        color_spaces::Illuminant::D65,
        color_spaces::Illuminant::A,
    );
    eprintln!("Bradford D65->A:   X={x:.6} Y={y:.6} Z={z:.6}");
    eprintln!("  Reference:       X=0.606172 Y=0.425971 Z=0.096777");
    let err = (x - 0.606172f32).abs().max((y - 0.425971).abs()).max((z - 0.096777).abs());
    assert!(err < 0.01, "Bradford D65->A error {err:.6} > 0.01");
}
