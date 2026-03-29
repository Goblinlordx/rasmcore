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
