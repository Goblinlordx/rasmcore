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
use rasmcore_pipeline::Rect;

/// Load an OpenCV reference fixture. Returns the data if available.
/// If the fixture file is missing (not generated yet), panics with a helpful message.
///
/// To generate fixtures: run `python3 scripts/generate-opencv-fixtures.py`
/// from the workspace root (requires opencv-python-headless in a venv).
fn load_fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/opencv/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    match std::fs::read(&path) {
        Ok(data) => data,
        Err(_) => {
            eprintln!(
                "SKIP {name}: fixture not found. Regenerate with:\n  \
                 python3 scripts/generate-opencv-fixtures.py"
            );
            // Return empty to allow test to detect and skip
            Vec::new()
        }
    }
}

/// Check if opencv fixtures are available (generated locally).
fn opencv_fixtures_available() -> bool {
    let path = format!(
        "{}/tests/fixtures/opencv/gradient_128_gray.raw",
        env!("CARGO_MANIFEST_DIR")
    );
    std::path::Path::new(&path).exists()
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

/// Skip guard — returns early from test if opencv fixtures aren't generated locally.
macro_rules! require_opencv_fixtures {
    () => {
        if !opencv_fixtures_available() {
            eprintln!("SKIP: opencv fixtures not generated. Run: python3 scripts/generate-opencv-fixtures.py");
            return;
        }
    };
}

// ─── CLAHE Parity ─────────────────────────────────────────────────────────

#[test]
fn clahe_all_images_match_opencv() {
    require_opencv_fixtures!();
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
        assert!(m <= 1, "CLAHE {name}: max error {m} > 1");
    }

    let avg_mae = total_mae / count as f64;
    eprintln!("\nCLAHE summary: avg_MAE={avg_mae:.4}, worst_max_err={total_max}");
}

// ─── Bilateral Parity ─────────────────────────────────────────────────────

#[test]
fn bilateral_all_images_match_opencv() {
    require_opencv_fixtures!();
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
        assert!(m <= 1, "Bilateral {name}: max error {m} > 1");
    }

    let avg_mae = total_mae / count as f64;
    eprintln!("\nBilateral summary: avg_MAE={avg_mae:.4}, worst_max_err={total_max}");
}

// ─── Guided Filter Parity ─────────────────────────────────────────────────

#[test]
fn guided_all_images_match_opencv() {
    require_opencv_fixtures!();
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
        assert!(m <= 2, "Guided {name}: max error {m} > 2");
    }

    let avg_mae = total_mae / count as f64;
    eprintln!("\nGuided summary: avg_MAE={avg_mae:.4}, worst_max_err={total_max}");
}

// ─── Color Science Parity ─────────────────────────────────────────────────

use rasmcore_image::domain::color_spaces;

#[test]
fn lab_conversion_matches_opencv() {
    require_opencv_fixtures!();
    let input = load_fixture("color_gradient_128_rgb.raw");
    let reference_lab = load_fixture("color_gradient_128_lab.raw");

    let info = ImageInfo {
        width: 128,
        height: 128,
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
    require_opencv_fixtures!();
    let input = load_fixture("gradient_128_gray.raw");
    let reference = load_fixture("gradient_128_perspective.raw");

    let info = ImageInfo {
        width: 128,
        height: 128,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let src = [(0.0f64, 0.0), (127.0, 0.0), (127.0, 127.0), (0.0, 127.0)];
    let dst = [(10.0f64, 10.0), (117.0, 5.0), (120.0, 122.0), (8.0, 120.0)];

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
    let mae = if count > 0 {
        total_err / count as f64
    } else {
        0.0
    };

    eprintln!(
        "Perspective vs OpenCV: MAE={mae:.4}, max_err={max_err}, compared={count}/{}",
        ours.len()
    );
    assert!(mae < 2.0, "Perspective MAE {mae:.4} > 2.0 vs OpenCV");
}

#[test]
fn gray_world_wb_matches_reference() {
    require_opencv_fixtures!();
    let input = load_fixture("blue_tinted_64_rgb.raw");
    let reference = load_fixture("blue_tinted_64_grayworld.raw");

    let info = ImageInfo {
        width: 64,
        height: 64,
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
    require_opencv_fixtures!();
    // Reference values from colour-science 0.4.7 (colour.XYZ_to_Oklab)
    let test_cases: &[(&str, (f64, f64, f64), (f64, f64, f64))] = &[
        ("red", (1.0, 0.0, 0.0), (0.627926, 0.224888, 0.125805)),
        ("green", (0.0, 1.0, 0.0), (0.866452, -0.233921, 0.179422)),
        ("blue", (0.0, 0.0, 1.0), (0.452033, -0.032352, -0.311621)),
        ("gray", (0.5, 0.5, 0.5), (0.598182, 0.000001, -0.000068)),
        ("custom", (0.5, 0.3, 0.8), (0.541759, 0.089488, -0.166634)),
        ("white", (1.0, 1.0, 1.0), (1.000002, 0.000002, -0.000114)),
        ("black", (0.0, 0.0, 0.0), (0.000000, 0.000000, 0.000000)),
    ];

    let mut max_err: f64 = 0.0;
    for (name, (r, g, b), (ref_l, ref_a, ref_b)) in test_cases {
        let (l, a, bv) = color_spaces::rgb_to_oklab(*r, *g, *b);
        let err = (l - ref_l)
            .abs()
            .max((a - ref_a).abs())
            .max((bv - ref_b).abs());
        max_err = max_err.max(err);
        eprintln!(
            "OKLab {name:8}: L={l:.6} a={a:.6} b={bv:.6} (ref: {ref_l:.6},{ref_a:.6},{ref_b:.6}) err={err:.6}"
        );
    }

    assert!(
        max_err < 0.001,
        "OKLab max error {max_err:.6} > 0.001 vs colour-science"
    );
}

#[test]
fn bradford_matches_colour_science() {
    require_opencv_fixtures!();
    // Reference: colour-science 0.4.7, Von Kries method with Bradford transform
    // D65->D50: XYZ(0.5, 0.4, 0.3) → (0.518086, 0.405866, 0.226963)
    let (x, y, z) = color_spaces::bradford_adapt(
        0.5,
        0.4,
        0.3,
        color_spaces::Illuminant::D65,
        color_spaces::Illuminant::D50,
    );
    eprintln!("Bradford D65->D50: X={x:.6} Y={y:.6} Z={z:.6}");
    eprintln!("  Reference:       X=0.518086 Y=0.405866 Z=0.226963");
    let err = (x - 0.518086f64)
        .abs()
        .max((y - 0.405866).abs())
        .max((z - 0.226963).abs());
    assert!(err < 0.01, "Bradford D65->D50 error {err:.6} > 0.01");

    // D65->A: XYZ(0.5, 0.4, 0.3) → (0.606172, 0.425971, 0.096777)
    let (x, y, z) = color_spaces::bradford_adapt(
        0.5,
        0.4,
        0.3,
        color_spaces::Illuminant::D65,
        color_spaces::Illuminant::A,
    );
    eprintln!("Bradford D65->A:   X={x:.6} Y={y:.6} Z={z:.6}");
    eprintln!("  Reference:       X=0.606172 Y=0.425971 Z=0.096777");
    let err = (x - 0.606172f64)
        .abs()
        .max((y - 0.425971).abs())
        .max((z - 0.096777).abs());
    assert!(err < 0.01, "Bradford D65->A error {err:.6} > 0.01");
}

#[test]
fn lab_matches_colour_science() {
    require_opencv_fixtures!();
    // Reference: colour-science 0.4.7 (f64, our true reference for Lab)
    // OpenCV uses 16-bit fixed-point internally — our f64 is more precise.
    let test_cases: &[(&str, (f64, f64, f64), (f64, f64, f64))] = &[
        (
            "red",
            (1.0, 0.0, 0.0),
            (53.2328817858, 80.1111777431, 67.2237036669),
        ),
        (
            "green",
            (0.0, 1.0, 0.0),
            (87.7370334735, -86.1828549966, 83.1878346582),
        ),
        (
            "blue",
            (0.0, 0.0, 1.0),
            (32.3025866672, 79.1980802348, -107.8503556950),
        ),
        (
            "gray",
            (0.5, 0.5, 0.5),
            (53.3889647411, 0.0046229008, 0.0021147334),
        ),
        (
            "white",
            (1.0, 1.0, 1.0),
            (100.0, 0.0077282677, 0.0035352751),
        ),
        ("black", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        (
            "custom",
            (0.8, 0.3, 0.6),
            (52.2589367273, 58.1618923913, -15.7114375190),
        ),
    ];

    let mut max_err: f64 = 0.0;
    for (name, (r, g, b), (ref_l, ref_a, ref_b)) in test_cases {
        let (l, a, bv) = color_spaces::rgb_to_lab(*r, *g, *b);
        let err = (l - ref_l)
            .abs()
            .max((a - ref_a).abs())
            .max((bv - ref_b).abs());
        max_err = max_err.max(err);
        eprintln!("Lab {name:8}: L={l:.10} a={a:.10} b={bv:.10} err={err:.2e}");
    }

    eprintln!("Lab max error vs colour-science: {max_err:.2e}");
    assert!(
        max_err < 1e-6,
        "Lab error {max_err:.2e} > 1e-6 vs colour-science"
    );
}

// ─── Delta E + LCH + Luv Parity (colour-science 0.4.7) ─────────────────

#[test]
fn delta_e_matches_colour_science() {
    require_opencv_fixtures!();
    let pairs: &[(&str, (f64, f64, f64), (f64, f64, f64), f64, f64, f64)] = &[
        (
            "red_green",
            (53.233, 80.109, 67.220),
            (87.737, -86.185, 83.181),
            170.5842137274,
            73.4337884112,
            86.6149551816,
        ),
        (
            "red_blue",
            (53.233, 80.109, 67.220),
            (32.303, 79.197, -107.864),
            176.3329342466,
            70.5760489317,
            52.8786354875,
        ),
        (
            "gray_white",
            (53.389, 0.005, 0.002),
            (100.0, 0.005, 0.002),
            46.6110000000,
            46.6110000000,
            33.4149645009,
        ),
        (
            "similar",
            (50.0, 2.5, 0.0),
            (50.0, 0.0, -2.5),
            3.5355339059,
            3.4077435238,
            4.3064820958,
        ),
        (
            "identical",
            (50.0, 25.0, -10.0),
            (50.0, 25.0, -10.0),
            0.0,
            0.0,
            0.0,
        ),
    ];

    let mut max76 = 0.0f64;
    let mut max94 = 0.0f64;
    let mut max00 = 0.0f64;
    for (name, lab1, lab2, ref76, ref94, ref00) in pairs {
        let de76 = color_spaces::delta_e_76(*lab1, *lab2);
        let de94 = color_spaces::delta_e_94(*lab1, *lab2, false);
        let de00 = color_spaces::delta_e_2000(*lab1, *lab2);
        let e76 = (de76 - ref76).abs();
        max76 = max76.max(e76);
        let e94 = (de94 - ref94).abs();
        max94 = max94.max(e94);
        let e00 = (de00 - ref00).abs();
        max00 = max00.max(e00);
        eprintln!(
            "{name:15}: dE76={de76:.6} (e={e76:.2e}) dE94={de94:.6} (e={e94:.2e}) dE00={de00:.6} (e={e00:.2e})"
        );
    }
    eprintln!("Max errors: dE76={max76:.2e} dE94={max94:.2e} dE00={max00:.2e}");
    assert!(max76 < 1e-6, "Delta E 76 error {max76:.2e}");
    assert!(max94 < 1e-6, "Delta E 94 error {max94:.2e}");
    assert!(max00 < 1e-4, "Delta E 2000 error {max00:.2e}"); // DE2000 has more chained ops
}

#[test]
fn lch_matches_colour_science() {
    require_opencv_fixtures!();
    let cases: &[(&str, (f64, f64, f64), (f64, f64, f64))] = &[
        (
            "red",
            (53.233, 80.109, 67.220),
            (53.233, 104.5752374179, 40.0002382458),
        ),
        (
            "green",
            (87.737, -86.185, 83.181),
            (87.737, 119.7786833539, 136.0161335584),
        ),
        ("gray", (50.0, 0.0, 0.0), (50.0, 0.0, 0.0)),
        (
            "custom",
            (75.0, 20.0, -30.0),
            (75.0, 36.0555127546, 303.6900675260),
        ),
    ];
    let mut max_err = 0.0f64;
    for (name, lab, ref_lch) in cases {
        let (l, c, h) = color_spaces::lab_to_lch(lab.0, lab.1, lab.2);
        let err = (l - ref_lch.0)
            .abs()
            .max((c - ref_lch.1).abs())
            .max((h - ref_lch.2).abs());
        max_err = max_err.max(err);
        eprintln!("LCH {name:8}: L={l:.6} C={c:.6} H={h:.6} err={err:.2e}");
    }
    assert!(max_err < 1e-8, "LCH error {max_err:.2e}");
}

#[test]
fn luv_matches_colour_science() {
    require_opencv_fixtures!();
    let cases: &[(&str, (f64, f64, f64), (f64, f64, f64))] = &[
        (
            "red",
            (1.0, 0.0, 0.0),
            (53.2328817858, 175.0598301857, 37.7617906121),
        ),
        (
            "green",
            (0.0, 1.0, 0.0),
            (87.7370334735, -83.0685534991, 107.4199653357),
        ),
        (
            "blue",
            (0.0, 0.0, 1.0),
            (32.3025866672, -9.3957447040, -130.3515592133),
        ),
        (
            "gray",
            (0.5, 0.5, 0.5),
            (53.3889647411, 0.0072898903, 0.0021849107),
        ),
        (
            "custom",
            (0.8, 0.3, 0.6),
            (52.2589367273, 73.0391180135, -32.3262768981),
        ),
    ];
    let mut max_err = 0.0f64;
    for (name, rgb, ref_luv) in cases {
        let (l, u, v) = color_spaces::rgb_to_luv(rgb.0, rgb.1, rgb.2);
        let err = (l - ref_luv.0)
            .abs()
            .max((u - ref_luv.1).abs())
            .max((v - ref_luv.2).abs());
        max_err = max_err.max(err);
        eprintln!("Luv {name:8}: L={l:.6} u={u:.6} v={v:.6} err={err:.2e}");
    }
    assert!(max_err < 1e-6, "Luv error {max_err:.2e}");
}

// ─── Scharr + Laplacian Parity ──────────────────────────────────────────

#[test]
fn scharr_matches_opencv() {
    require_opencv_fixtures!();
    let test_images = [
        "gradient_128",
        "checker_128",
        "sharp_edges_128",
        "photo_128",
    ];
    let info = info_128();
    let mut worst_mae = 0.0f64;
    let mut worst_max = 0u8;

    for name in &test_images {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_scharr.raw"));
        let ours = filters::scharr(&input, &info).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        worst_mae = worst_mae.max(e);
        worst_max = worst_max.max(m);
        eprintln!("Scharr {name:20}: MAE={e:.4}, max_err={m}");
    }
    eprintln!("Scharr summary: worst_MAE={worst_mae:.4}, worst_max={worst_max}");
    assert!(worst_mae < 1.0, "Scharr MAE {worst_mae:.4} > 1.0");
    assert!(worst_max <= 1, "Scharr max error {worst_max} > 1");
}

#[test]
fn laplacian_matches_opencv() {
    require_opencv_fixtures!();
    let test_images = [
        "gradient_128",
        "checker_128",
        "sharp_edges_128",
        "photo_128",
    ];
    let info = info_128();
    let mut worst_mae = 0.0f64;
    let mut worst_max = 0u8;

    for name in &test_images {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_laplacian.raw"));
        let ours = filters::laplacian(&input, &info).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        worst_mae = worst_mae.max(e);
        worst_max = worst_max.max(m);
        eprintln!("Laplacian {name:20}: MAE={e:.4}, max_err={m}");
    }
    eprintln!("Laplacian summary: worst_MAE={worst_mae:.4}, worst_max={worst_max}");
    assert!(worst_mae < 1.0, "Laplacian MAE {worst_mae:.4} > 1.0");
    assert!(worst_max <= 1, "Laplacian max error {worst_max} > 1");
}

// ─── HDR Merge: Mertens Parity ──────────────────────────────────────────

fn load_fixture_f32(name: &str) -> Vec<f32> {
    let bytes = load_fixture(name);
    assert_eq!(
        bytes.len() % 4,
        0,
        "f32 fixture must be multiple of 4 bytes"
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn mae_f32(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn max_error_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0f64, f64::max)
}

#[test]
fn mertens_fusion_matches_opencv() {
    require_opencv_fixtures!();
    // Load 3 synthetic bracketed exposures (RGB, 64×64)
    let bracket0 = load_fixture("hdr_bracket_0_64x64_rgb.raw");
    let bracket1 = load_fixture("hdr_bracket_1_64x64_rgb.raw");
    let bracket2 = load_fixture("hdr_bracket_2_64x64_rgb.raw");

    let info = ImageInfo {
        width: 64,
        height: 64,
        format: PixelFormat::Rgb8,
        color_space: ColorSpace::Srgb,
    };

    let params = filters::MertensParams {
        contrast_weight: 1.0,
        saturation_weight: 1.0,
        exposure_weight: 1.0,
    };

    // Get f32 result for precision comparison
    let ours_f32 =
        filters::mertens_fusion_f32(&[&bracket0, &bracket1, &bracket2], &info, &params).unwrap();

    // Load OpenCV f32 reference (BGR order from OpenCV)
    let ref_f32 = load_fixture_f32("mertens_64x64_3exp_f32.raw");

    // OpenCV stores BGR, our output is RGB — convert reference to RGB for comparison
    let n_pixels = 64 * 64;
    assert_eq!(ref_f32.len(), n_pixels * 3);
    assert_eq!(ours_f32.len(), n_pixels * 3);

    let mut ref_rgb = vec![0.0f32; n_pixels * 3];
    for i in 0..n_pixels {
        ref_rgb[i * 3] = ref_f32[i * 3 + 2]; // R = B (BGR→RGB)
        ref_rgb[i * 3 + 1] = ref_f32[i * 3 + 1]; // G = G
        ref_rgb[i * 3 + 2] = ref_f32[i * 3]; // B = R
    }

    let e = mae_f32(&ours_f32, &ref_rgb);
    let m = max_error_f32(&ours_f32, &ref_rgb);

    eprintln!("Mertens f32 vs OpenCV: MAE={e:.6}, max_err={m:.6}");
    eprintln!("  (MAE in [0,1] scale; max_err in [0,1] scale)");

    // Also check u8 result
    let ours_u8 =
        filters::mertens_fusion(&[&bracket0, &bracket1, &bracket2], &info, &params).unwrap();

    let ref_u8 = load_fixture("mertens_64x64_3exp_rgb.raw");
    let e_u8 = mae(&ours_u8, &ref_u8);
    let m_u8 = max_error(&ours_u8, &ref_u8);

    eprintln!("Mertens u8 vs OpenCV:  MAE={e_u8:.4}, max_err={m_u8}");

    // Algorithm exactly matches OpenCV MergeMertens (verified against source):
    // - COLOR_RGB2GRAY on BGR data (0.299*B + 0.587*G + 0.114*R)
    // - Saturation: sqrt(sum((ch-mean)²)) — not population std
    // - Well-exposedness: -(ch-0.5)²/0.08 — exact operation order
    // Remaining difference: f32 accumulation order in 6-level pyramid ops.
    // Tier: F32_ROUNDING — max ±4 u8 levels.
    assert!(
        e < 0.01,
        "Mertens f32 MAE {e:.6} > 0.01 — algorithmic divergence from OpenCV"
    );
    assert!(
        m < 0.03,
        "Mertens f32 max_err {m:.6} > 0.03 — likely algorithmic issue"
    );
    assert!(m_u8 <= 4, "Mertens u8 max error {m_u8} > 4");
    assert!(e_u8 < 1.0, "Mertens u8 MAE {e_u8:.4} > 1.0");
}

#[test]
fn mertens_all_canonical_images() {
    require_opencv_fixtures!();
    let params = filters::MertensParams {
        contrast_weight: 1.0,
        saturation_weight: 1.0,
        exposure_weight: 1.0,
    };

    let mut worst_mae_f32 = 0.0f64;
    let mut worst_max_f32 = 0.0f64;
    let mut worst_mae_u8 = 0.0f64;
    let mut worst_max_u8 = 0u8;

    for name in TEST_IMAGES {
        // Load 3 exposure brackets (128×128 RGB, generated from canonical gray images)
        let b0 = load_fixture(&format!("hdr_{name}_bracket_0.raw"));
        let b1 = load_fixture(&format!("hdr_{name}_bracket_1.raw"));
        let b2 = load_fixture(&format!("hdr_{name}_bracket_2.raw"));

        let info = ImageInfo {
            width: 128,
            height: 128,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };

        // f32 comparison
        let ours_f32 = filters::mertens_fusion_f32(&[&b0, &b1, &b2], &info, &params).unwrap();
        let ref_f32 = load_fixture_f32(&format!("hdr_{name}_mertens_f32.raw"));

        // Convert BGR→RGB in reference
        let n_pixels = 128 * 128;
        let mut ref_rgb = vec![0.0f32; n_pixels * 3];
        for i in 0..n_pixels {
            ref_rgb[i * 3] = ref_f32[i * 3 + 2];
            ref_rgb[i * 3 + 1] = ref_f32[i * 3 + 1];
            ref_rgb[i * 3 + 2] = ref_f32[i * 3];
        }

        let e_f32 = mae_f32(&ours_f32, &ref_rgb);
        let m_f32 = max_error_f32(&ours_f32, &ref_rgb);

        // u8 comparison
        let ours_u8 = filters::mertens_fusion(&[&b0, &b1, &b2], &info, &params).unwrap();
        let ref_u8 = load_fixture(&format!("hdr_{name}_mertens_rgb.raw"));
        let e_u8 = mae(&ours_u8, &ref_u8);
        let m_u8 = max_error(&ours_u8, &ref_u8);

        eprintln!(
            "Mertens {name:20}: f32 MAE={e_f32:.6} max={m_f32:.6} | u8 MAE={e_u8:.4} max={m_u8}"
        );

        worst_mae_f32 = worst_mae_f32.max(e_f32);
        worst_max_f32 = worst_max_f32.max(m_f32);
        worst_mae_u8 = worst_mae_u8.max(e_u8);
        worst_max_u8 = worst_max_u8.max(m_u8);

        // Checker pattern (extreme high-frequency) causes pyramid aliasing at every level;
        // even Python-manual-vs-OpenCV has u8 max=17 for checker. Allow higher threshold.
        let limit = if name.contains("checker") { 30 } else { 3 };
        assert!(
            m_u8 <= limit,
            "Mertens {name}: u8 max error {m_u8} > {limit}"
        );
    }

    eprintln!(
        "\nMertens summary: worst f32 MAE={worst_mae_f32:.6} max={worst_max_f32:.6} | \
         worst u8 MAE={worst_mae_u8:.4} max={worst_max_u8}"
    );
}

// ─── Otsu Threshold Parity ──────────────────────────────────────────────

#[test]
fn otsu_threshold_matches_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    // Expected Otsu thresholds from OpenCV 4.13
    let expected: &[(&str, u8)] = &[
        ("gradient_128", 126),
        ("checker_128", 25),
        ("noisy_flat_128", 127),
        ("photo_128", 127),
        ("highcontrast_128", 132),
    ];

    for &(name, cv_thresh) in expected {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let our_thresh = filters::otsu_threshold(&input, &info).unwrap();
        eprintln!("Otsu {name:20}: ours={our_thresh}, cv={cv_thresh}");
        let diff = (our_thresh as i16 - cv_thresh as i16).abs();
        assert!(
            diff <= 1,
            "Otsu {name}: ours={our_thresh}, cv={cv_thresh}, diff={diff} > 1"
        );
    }
}

#[test]
fn triangle_threshold_matches_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    let expected: &[(&str, u8)] = &[
        ("gradient_128", 2),
        ("checker_128", 27),
        ("noisy_flat_128", 164),
        ("photo_128", 139),
        ("highcontrast_128", 129),
    ];

    for &(name, cv_thresh) in expected {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let our_thresh = filters::triangle_threshold(&input, &info).unwrap();
        eprintln!("Triangle {name:20}: ours={our_thresh}, cv={cv_thresh}");
        let diff = (our_thresh as i16 - cv_thresh as i16).abs();
        assert!(
            diff <= 1,
            "Triangle {name}: ours={our_thresh}, cv={cv_thresh}, diff={diff} > 1"
        );
    }
}

#[test]
fn adaptive_mean_matches_opencv() {
    require_opencv_fixtures!();
    let info = info_128();

    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_adaptive_mean.raw"));
        let ours =
            filters::adaptive_threshold(&input, &info, 255, filters::AdaptiveMethod::Mean, 11, 2.0)
                .unwrap();

        let mismatches: usize = ours
            .iter()
            .zip(reference.iter())
            .filter(|(a, b)| a != b)
            .count();
        let mismatch_pct = 100.0 * mismatches as f64 / (128 * 128) as f64;
        eprintln!(
            "Adaptive mean {name:20}: mismatches={mismatches}/{} ({mismatch_pct:.2}%)",
            128 * 128
        );

        // Integer box mean + BORDER_REPLICATE matches OpenCV exactly.
        assert_eq!(
            mismatches, 0,
            "Adaptive mean {name}: {mismatches} pixel mismatches (expected exact match)"
        );
    }
}

#[test]
fn adaptive_gaussian_matches_opencv() {
    require_opencv_fixtures!();
    let info = info_128();

    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_adaptive_gauss.raw"));
        let ours = filters::adaptive_threshold(
            &input,
            &info,
            255,
            filters::AdaptiveMethod::Gaussian,
            11,
            2.0,
        )
        .unwrap();

        let mismatches: usize = ours
            .iter()
            .zip(reference.iter())
            .filter(|(a, b)| a != b)
            .count();
        let mismatch_pct = 100.0 * mismatches as f64 / (128 * 128) as f64;
        eprintln!(
            "Adaptive gauss {name:20}: mismatches={mismatches}/{} ({mismatch_pct:.2}%)",
            128 * 128
        );

        assert!(
            mismatch_pct < 5.0,
            "Adaptive gauss {name}: {mismatch_pct:.2}% mismatches > 5%"
        );
    }
}

// ─── Morphology Parity ──────────────────────────────────────────────────

#[test]
fn erode_dilate_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    let test_images = ["gradient_128", "checker_128", "sharp_edges_128"];
    let shapes: &[(&str, filters::MorphShape)] = &[
        ("rect", filters::MorphShape::Rect),
        ("cross", filters::MorphShape::Cross),
        ("ellipse", filters::MorphShape::Ellipse),
    ];
    let ksizes: &[u32] = &[3, 5, 7];

    let mut worst_mae = 0.0f64;
    let mut worst_max = 0u8;
    let mut total_tests = 0;

    for name in &test_images {
        let input = load_fixture(&format!("{name}_gray.raw"));

        for &(shape_name, shape) in shapes {
            for &ks in ksizes {
                // Erode
                let ref_erode = load_fixture(&format!("{name}_erode_{shape_name}_{ks}.raw"));
                let our_erode = filters::erode(&input, &info, ks, shape).unwrap();
                let e = mae(&our_erode, &ref_erode);
                let m = max_error(&our_erode, &ref_erode);
                worst_mae = worst_mae.max(e);
                worst_max = worst_max.max(m);
                total_tests += 1;

                if m > 0 {
                    eprintln!("Erode {name} {shape_name} {ks}: MAE={e:.4}, max={m}");
                }
                assert!(m <= 1, "Erode {name} {shape_name} {ks}: max_err={m} > 1");

                // Dilate
                let ref_dilate = load_fixture(&format!("{name}_dilate_{shape_name}_{ks}.raw"));
                let our_dilate = filters::dilate(&input, &info, ks, shape).unwrap();
                let e = mae(&our_dilate, &ref_dilate);
                let m = max_error(&our_dilate, &ref_dilate);
                worst_mae = worst_mae.max(e);
                worst_max = worst_max.max(m);
                total_tests += 1;

                if m > 0 {
                    eprintln!("Dilate {name} {shape_name} {ks}: MAE={e:.4}, max={m}");
                }
                assert!(m <= 1, "Dilate {name} {shape_name} {ks}: max_err={m} > 1");
            }
        }
    }

    eprintln!(
        "\nMorphology summary: {total_tests} tests, worst MAE={worst_mae:.4}, worst max={worst_max}"
    );
}

#[test]
fn morph_compound_ops_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    let input = load_fixture("sharp_edges_128_gray.raw");

    let ops: &[(
        &str,
        fn(&[u8], &ImageInfo, u32, filters::MorphShape) -> Result<Vec<u8>, _>,
    )] = &[
        ("open", filters::morph_open as _),
        ("close", filters::morph_close as _),
        ("gradient", filters::morph_gradient as _),
        ("tophat", filters::morph_tophat as _),
        ("blackhat", filters::morph_blackhat as _),
    ];

    for &(op_name, op_fn) in ops {
        let reference = load_fixture(&format!("sharp_edges_128_morph_{op_name}_rect_5.raw"));
        let ours = op_fn(&input, &info, 5, filters::MorphShape::Rect).unwrap();
        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("Morph {op_name:10}: MAE={e:.4}, max_err={m}");
        assert!(m <= 1, "Morph {op_name}: max_err={m} > 1");
    }
}

// ─── Displacement Map Parity ─────────────────────────────────────────────

fn load_f32_fixture(name: &str) -> Vec<f32> {
    let bytes = load_fixture(name);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn displacement_map_barrel_all_images_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    let map_x = load_f32_fixture("displace_barrel_map_x.raw");
    let map_y = load_f32_fixture("displace_barrel_map_y.raw");

    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_displace_barrel.raw"));
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(input.clone());
        let ours = filters::displacement_map(r, &mut u, &info, &map_x, &map_y).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("displace_barrel {name:20}: MAE={e:.4}, max_err={m}");
        assert!(e < 0.5, "displace_barrel {name}: MAE={e:.4} >= 0.5");
        assert!(m <= 1, "displace_barrel {name}: max_err={m} > 1");
    }
}

#[test]
fn displacement_map_wave_all_images_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    let map_x = load_f32_fixture("displace_wave_map_x.raw");
    let map_y = load_f32_fixture("displace_wave_map_y.raw");

    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_displace_wave.raw"));
        let r = Rect::new(0, 0, info.width, info.height);
        let mut u = |_: Rect| Ok(input.clone());
        let ours = filters::displacement_map(r, &mut u, &info, &map_x, &map_y).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("displace_wave {name:20}: MAE={e:.4}, max_err={m}");
        assert!(e < 0.5, "displace_wave {name}: MAE={e:.4} >= 0.5");
        assert!(m <= 1, "displace_wave {name}: max_err={m} > 1");
    }
}

// ─── Bokeh Blur Parity ──────────────────────────────────────────────────

#[test]
fn bokeh_disc_r3_all_images_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_bokeh_disc_3.raw"));
        let r = rasmcore_pipeline::Rect::new(0, 0, info.width, info.height);
        let mut u = |_: rasmcore_pipeline::Rect| Ok(input.clone());
        let ours = filters::bokeh_blur(r, &mut u, &info, &filters::BokehBlurParams { radius: 3, shape: 0 }).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("bokeh_disc_3 {name:20}: MAE={e:.4}, max_err={m}");
        assert!(e < 0.5, "bokeh_disc_3 {name}: MAE={e:.4} >= 0.5");
        assert!(m <= 1, "bokeh_disc_3 {name}: max_err={m} > 1");
    }
}

#[test]
fn bokeh_disc_r7_all_images_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_bokeh_disc_7.raw"));
        let r = rasmcore_pipeline::Rect::new(0, 0, info.width, info.height);
        let mut u = |_: rasmcore_pipeline::Rect| Ok(input.clone());
        let ours = filters::bokeh_blur(r, &mut u, &info, &filters::BokehBlurParams { radius: 7, shape: 0 }).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("bokeh_disc_7 {name:20}: MAE={e:.4}, max_err={m}");
        assert!(e < 0.5, "bokeh_disc_7 {name}: MAE={e:.4} >= 0.5");
        assert!(m <= 1, "bokeh_disc_7 {name}: max_err={m} > 1");
    }
}

#[test]
fn bokeh_hex_r3_all_images_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_bokeh_hex_3.raw"));
        let r = rasmcore_pipeline::Rect::new(0, 0, info.width, info.height);
        let mut u = |_: rasmcore_pipeline::Rect| Ok(input.clone());
        let ours = filters::bokeh_blur(r, &mut u, &info, &filters::BokehBlurParams { radius: 3, shape: 1 }).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("bokeh_hex_3 {name:20}: MAE={e:.4}, max_err={m}");
        assert!(e < 0.5, "bokeh_hex_3 {name}: MAE={e:.4} >= 0.5");
        assert!(m <= 1, "bokeh_hex_3 {name}: max_err={m} > 1");
    }
}

#[test]
fn bokeh_hex_r7_all_images_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_bokeh_hex_7.raw"));
        let r = rasmcore_pipeline::Rect::new(0, 0, info.width, info.height);
        let mut u = |_: rasmcore_pipeline::Rect| Ok(input.clone());
        let ours = filters::bokeh_blur(r, &mut u, &info, &filters::BokehBlurParams { radius: 7, shape: 1 }).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("bokeh_hex_7 {name:20}: MAE={e:.4}, max_err={m}");
        assert!(e < 0.5, "bokeh_hex_7 {name}: MAE={e:.4} >= 0.5");
        assert!(m <= 1, "bokeh_hex_7 {name}: max_err={m} > 1");
    }
}

// ─── Vignette (Gaussian) Parity ─────────────────────────────────────────

/// Validates Gaussian vignette against ImageMagick 7.x reference output.
///
/// Reference: `magick -vignette 0x15+10+10` (sigma=15, x_inset=10, y_inset=10).
/// The residual error comes from IM's anti-aliased ellipse rasterization
/// (sub-pixel coverage) vs our binary ellipse mask. After Gaussian blur
/// the difference is small: MAE < 2.0 at 8-bit.
#[test]
fn vignette_gaussian_all_images_match_imagemagick() {
    require_opencv_fixtures!();
    let info = info_128();
    let sigma = 15.0f32;
    let x_inset = 10u32;
    let y_inset = 10u32;
    let mut total_mae = 0.0;
    let mut total_max = 0u8;

    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_vignette.raw"));
        let ours =
            filters::vignette(&input, &info, sigma, x_inset, y_inset, 128, 128, 0, 0).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        total_mae += e;
        total_max = total_max.max(m);
        eprintln!("vignette {name:20}: MAE={e:.4}, max_err={m}");
        assert!(e < 1.5, "vignette {name}: MAE={e:.4} >= 1.5");
        assert!(m <= 5, "vignette {name}: max_err={m} > 5");
    }

    let avg_mae = total_mae / TEST_IMAGES.len() as f64;
    eprintln!("vignette summary: avg_MAE={avg_mae:.4}, global_max_err={total_max}");
}

// ─── Perspective Warp Parity ────────────────────────────────────────────────
//
// Tests perspective_warp against precomputed OpenCV warpPerspective output.
// Fixture: gradient_128_perspective.raw (generated by gen_perspective_hough.py
// or existing fixture from earlier work).

#[test]
fn perspective_warp_identity_is_exact() {
    require_opencv_fixtures!();
    // Identity matrix warp on a real image should reproduce input exactly.
    // This validates the OpenCV-aligned fixed-point bilinear path at integer coords.
    let input = load_fixture("gradient_128_gray.raw");
    assert_eq!(input.len(), 128 * 128);

    let info = ImageInfo {
        width: 128,
        height: 128,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let result = filters::perspective_warp(&input, &info, &identity, 128, 128).unwrap();

    let e = mae(&result, &input);
    let m = max_error(&result, &input);
    eprintln!("perspective_warp identity: MAE={e:.4}, max_err={m}");

    // At integer coordinates with identity matrix, bilinear weights are [32768, 0, 0, 0].
    // Result should be pixel-exact.
    assert_eq!(e, 0.0, "identity warp should be pixel-exact");
    assert_eq!(m, 0, "identity warp should have zero max error");
}

#[test]
fn perspective_warp_translation_is_exact() {
    require_opencv_fixtures!();
    // Integer translation should also be pixel-exact (no sub-pixel interpolation).
    let input = load_fixture("gradient_128_gray.raw");
    let info = ImageInfo {
        width: 128,
        height: 128,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };

    // Shift right by 5, down by 3: output pixel (x,y) reads input (x+5, y+3)
    let mat = [1.0, 0.0, 5.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0];
    let result = filters::perspective_warp(&input, &info, &mat, 128, 128).unwrap();

    // Verify shifted region matches exactly
    let mut mismatches = 0;
    for y in 0..125 {
        for x in 0..123 {
            let src_idx = (y + 3) * 128 + (x + 5);
            let dst_idx = y * 128 + x;
            if result[dst_idx] != input[src_idx] {
                mismatches += 1;
            }
        }
    }
    eprintln!(
        "perspective_warp translation: {mismatches} mismatches out of {}",
        125 * 123
    );
    assert_eq!(mismatches, 0, "integer translation should be pixel-exact");
}

#[test]
fn perspective_warp_bilinear_weight_table_correct() {
    require_opencv_fixtures!();
    // Verify the OpenCV-aligned weight table: at sub-pixel (0,0), all weight
    // should be on the top-left neighbor. At (16/32, 16/32) = (0.5, 0.5),
    // weight should be equally distributed.
    let tab = filters::build_bilinear_tab_public();

    // Index 0 = sub-pixel (0, 0): full weight on top-left
    assert_eq!(tab[0], [32768, 0, 0, 0], "origin weight wrong");

    // Index 528 = sub-pixel (16, 16) = (0.5, 0.5): equal weights
    // (16 * 32 + 16 = 528)
    let w = tab[528];
    let sum: i32 = w.iter().sum();
    assert_eq!(sum, 32768, "weights must sum to 32768");
    // Each weight should be ~8192 (32768/4)
    for &wt in &w {
        assert!(
            (wt - 8192).abs() <= 1,
            "half-point weight should be ~8192, got {wt}"
        );
    }
}

// ─── Homography Solver Parity ───────────────────────────────────────────────

#[test]
fn homography_solver_identity_exact() {
    require_opencv_fixtures!();
    let pts = [(0.0f32, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
    let h = filters::solve_homography_4pt_public(&pts, &pts).unwrap();
    // OpenCV: M[8] = 1.0 exactly (c22 = 1 assumption)
    assert!((h[8] - 1.0).abs() < 1e-15, "c22 should be exactly 1.0");
    // Identity matrix
    for i in 0..9 {
        let expected = if i == 0 || i == 4 || i == 8 { 1.0 } else { 0.0 };
        assert!(
            (h[i] - expected).abs() < 1e-10,
            "h[{i}] = {}, expected {}",
            h[i],
            expected
        );
    }
}

#[test]
fn homography_solver_perspective_c22_is_one() {
    require_opencv_fixtures!();
    // Non-trivial perspective case: c22 should still be exactly 1.0
    let src = [(0.0f32, 0.0), (127.0, 0.0), (127.0, 127.0), (0.0, 127.0)];
    let dst = [(10.0f32, 5.0), (117.0, 0.0), (122.0, 127.0), (5.0, 122.0)];
    let h = filters::solve_homography_4pt_public(&src, &dst).unwrap();
    assert!(
        (h[8] - 1.0).abs() < 1e-15,
        "c22 = {}, should be exactly 1.0",
        h[8]
    );
}

// ─── Hough Lines PPHT Structural Tests ──────────────────────────────────────

#[test]
fn hough_ppht_deterministic_same_seed() {
    require_opencv_fixtures!();
    let mut pixels = vec![0u8; 64 * 64];
    for x in 5..60 {
        pixels[32 * 64 + x] = 255;
    }
    for y in 10..55 {
        pixels[y * 64 + 20] = 255;
    }
    let info = ImageInfo {
        width: 64,
        height: 64,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };
    let theta = std::f32::consts::PI / 180.0;

    let lines1 = filters::hough_lines_p(&pixels, &info, 1.0, theta, 15, 20, 5, 42).unwrap();
    let lines2 = filters::hough_lines_p(&pixels, &info, 1.0, theta, 15, 20, 5, 42).unwrap();
    assert_eq!(lines1.len(), lines2.len(), "same seed → same line count");
    for (a, b) in lines1.iter().zip(lines2.iter()) {
        assert_eq!(
            (a.x1, a.y1, a.x2, a.y2),
            (b.x1, b.y1, b.x2, b.y2),
            "same seed → same endpoints"
        );
    }
}

#[test]
fn hough_ppht_different_seeds_may_differ() {
    require_opencv_fixtures!();
    let mut pixels = vec![0u8; 64 * 64];
    for x in 5..60 {
        pixels[32 * 64 + x] = 255;
    }
    for y in 10..55 {
        pixels[y * 64 + 20] = 255;
    }
    let info = ImageInfo {
        width: 64,
        height: 64,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };
    let theta = std::f32::consts::PI / 180.0;

    let lines1 = filters::hough_lines_p(&pixels, &info, 1.0, theta, 15, 20, 5, 1).unwrap();
    let lines2 = filters::hough_lines_p(&pixels, &info, 1.0, theta, 15, 20, 5, 2).unwrap();
    // Both should find lines (the image has clear structure)
    assert!(!lines1.is_empty(), "seed 1 should find lines");
    assert!(!lines2.is_empty(), "seed 2 should find lines");
    // But endpoints may differ due to different processing order
    eprintln!(
        "hough seed test: seed1={} lines, seed2={} lines",
        lines1.len(),
        lines2.len()
    );
}

#[test]
fn hough_ppht_vote_decrement_no_duplicates() {
    require_opencv_fixtures!();
    // Single strong line → should produce 1 segment, not many
    let mut pixels = vec![0u8; 100 * 100];
    for x in 10..90 {
        pixels[50 * 100 + x] = 255;
    }
    let info = ImageInfo {
        width: 100,
        height: 100,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    };
    let lines = filters::hough_lines_p(
        &pixels,
        &info,
        1.0,
        std::f32::consts::PI / 180.0,
        30,
        30,
        5,
        0,
    )
    .unwrap();
    assert!(
        lines.len() <= 2,
        "vote decrement should prevent duplicates, got {} lines",
        lines.len()
    );
}

// ─── Canny Edge Detection Parity ─────────────────────────────────────────

#[test]
fn canny_all_images_match_opencv() {
    require_opencv_fixtures!();
    let info = info_128();
    for name in TEST_IMAGES {
        let input = load_fixture(&format!("{name}_gray.raw"));
        let reference = load_fixture(&format!("{name}_canny_50_150.raw"));
        let ours = filters::canny(&input, &info, 50.0, 150.0).unwrap();

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("canny {name:20}: MAE={e:.4}, max_err={m}");
        // Matches OpenCV's tangent-ratio NMS and stack-based hysteresis.
        // Small residual on noisy images from f32 vs int16 precision in TG22.
        assert!(e < 10.0, "canny {name}: MAE={e:.4} >= 10.0");
    }
}

// ─── pyrUp Parity ────────────────────────────────────────────────────────

fn info_64() -> ImageInfo {
    ImageInfo {
        width: 64,
        height: 64,
        format: PixelFormat::Gray8,
        color_space: ColorSpace::Srgb,
    }
}

#[test]
fn pyr_up_all_images_match_opencv() {
    require_opencv_fixtures!();
    let info64 = info_64();
    for name in TEST_IMAGES {
        let input_64 = load_fixture(&format!("{name}_pyrdown_64.raw"));
        let reference = load_fixture(&format!("{name}_pyrup_from64.raw"));
        let (ours, new_info) = filters::pyr_up(&input_64, &info64).unwrap();

        assert_eq!(new_info.width, 128);
        assert_eq!(new_info.height, 128);
        assert_eq!(ours.len(), reference.len());

        let e = mae(&ours, &reference);
        let m = max_error(&ours, &reference);
        eprintln!("pyr_up {name:20}: MAE={e:.4}, max_err={m}");
        assert!(e < 2.0, "pyr_up {name}: MAE={e:.4} >= 2.0");
        assert!(m <= 5, "pyr_up {name}: max_err={m} > 5");
    }
}
