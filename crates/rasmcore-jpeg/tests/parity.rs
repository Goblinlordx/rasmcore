//! Three-way codec parity tests for ALL JPEG modes.
//!
//! Validates per codec-validation.md:
//!   A = our_encode(original) → our_decode
//!   B = our_encode(original) → ref_decode (image crate)
//!   C = ref_encode(original) → ref_decode (image crate or ImageMagick)
//!
//! Lossy assertions:
//!   B_quality >= C_quality × 0.9
//!   A ≈ B  (MAE < threshold)
//!
//! Test matrix:
//!   - Baseline 4:2:0, 4:2:2, 4:4:4 at Q25/Q50/Q75/Q95
//!   - Grayscale at Q75/Q95
//!   - Progressive 4:2:0, 4:4:4
//!   - Trellis quantization
//!   - Arithmetic coding (encode/decode roundtrip)
//!   - External fixtures from ImageMagick (baseline + progressive)
//!
//! Reference encoders:
//!   - image crate (baseline sequential)
//!   - ImageMagick (progressive, baseline with sampling control)

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

// ─── Helpers ───────────────────────────────────────────────────────────────

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "pixel buffer length mismatch");
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

fn mae(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn gradient_pixels(w: u32, h: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            p.push((x * 255 / w.max(1)) as u8);
            p.push((y * 255 / h.max(1)) as u8);
            p.push(128);
        }
    }
    p
}

fn checkerboard_pixels(w: u32, h: u32, cell: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = if ((x / cell) + (y / cell)) % 2 == 0 {
                240u8
            } else {
                16u8
            };
            p.push(v);
            p.push(v);
            p.push(v);
        }
    }
    p
}

fn gray_gradient(w: u32, h: u32) -> Vec<u8> {
    (0..w * h).map(|i| (i * 255 / (w * h)) as u8).collect()
}

fn magick_available() -> bool {
    Command::new("magick")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// mozjpeg cjpeg paths to search (brew installs to non-standard location).
const MOZJPEG_PATHS: &[&str] = &[
    "/opt/homebrew/opt/mozjpeg/bin/cjpeg",
    "/usr/local/opt/mozjpeg/bin/cjpeg",
    "/usr/bin/mozjpeg-cjpeg",
];

fn mozjpeg_cjpeg_path() -> Option<&'static str> {
    for path in MOZJPEG_PATHS {
        if std::path::Path::new(path).exists() {
            return Some(path);
        }
    }
    // Check if system cjpeg is mozjpeg
    if let Ok(out) = Command::new("cjpeg").arg("-version").output() {
        let version = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        if version.contains("mozjpeg") || stderr.contains("mozjpeg") {
            return Some("cjpeg");
        }
    }
    None
}

/// Encode via mozjpeg cjpeg CLI. Returns None if mozjpeg not available.
fn mozjpeg_encode(pixels: &[u8], w: u32, h: u32, quality: u32) -> Option<Vec<u8>> {
    let cjpeg = mozjpeg_cjpeg_path()?;
    let ppm_path = write_ppm(pixels, w, h);
    let jpg_path = ppm_path.with_extension("moz.jpg");

    let result = Command::new(cjpeg)
        .args(["-quality", &quality.to_string()])
        .args(["-outfile", jpg_path.to_str().unwrap()])
        .arg(ppm_path.to_str().unwrap())
        .output()
        .ok()?;

    let _ = std::fs::remove_file(&ppm_path);
    if !result.status.success() {
        let _ = std::fs::remove_file(&jpg_path);
        return None;
    }
    let data = std::fs::read(&jpg_path).ok();
    let _ = std::fs::remove_file(&jpg_path);
    data
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_ppm(pixels: &[u8], w: u32, h: u32) -> std::path::PathBuf {
    use std::io::Write;
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("rasmcore_parity_{}_{id}.ppm", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    f.write_all(pixels).unwrap();
    path
}

fn write_pgm(pixels: &[u8], w: u32, h: u32) -> std::path::PathBuf {
    use std::io::Write;
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("rasmcore_parity_{}_{id}.pgm", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    write!(f, "P5\n{w} {h}\n255\n").unwrap();
    f.write_all(pixels).unwrap();
    path
}

fn magick_encode(
    pixels: &[u8],
    w: u32,
    h: u32,
    quality: u32,
    extra_args: &[&str],
) -> Option<Vec<u8>> {
    let ppm_path = write_ppm(pixels, w, h);
    let jpg_path = ppm_path.with_extension("jpg");
    let mut cmd = Command::new("magick");
    cmd.arg(ppm_path.to_str().unwrap())
        .args(["-quality", &quality.to_string()]);
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.arg(jpg_path.to_str().unwrap());
    let status = cmd.output().ok()?;
    let _ = std::fs::remove_file(&ppm_path);
    if !status.status.success() {
        let _ = std::fs::remove_file(&jpg_path);
        return None;
    }
    let data = std::fs::read(&jpg_path).ok();
    let _ = std::fs::remove_file(&jpg_path);
    data
}

fn magick_encode_gray(
    pixels: &[u8],
    w: u32,
    h: u32,
    quality: u32,
    extra_args: &[&str],
) -> Option<Vec<u8>> {
    let pgm_path = write_pgm(pixels, w, h);
    let jpg_path = pgm_path.with_extension("jpg");
    let mut cmd = Command::new("magick");
    cmd.arg(pgm_path.to_str().unwrap())
        .args(["-quality", &quality.to_string()]);
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.arg(jpg_path.to_str().unwrap());
    let status = cmd.output().ok()?;
    let _ = std::fs::remove_file(&pgm_path);
    if !status.status.success() {
        let _ = std::fs::remove_file(&jpg_path);
        return None;
    }
    let data = std::fs::read(&jpg_path).ok();
    let _ = std::fs::remove_file(&jpg_path);
    data
}

/// Three-way test helper. Runs A/B/C paths and asserts quality thresholds.
///
/// - `label`: test label for diagnostics
/// - `our_jpeg`: our encoder output (JPEG bytes)
/// - `ref_jpeg`: reference encoder output (JPEG bytes, for C path)
/// - `original`: original pixel data (for PSNR computation)
/// - `is_gray`: if true, compare as luma; else as RGB
/// - `b_ge_c_ratio`: B_quality must be >= C_quality * this ratio
/// - `ab_mae_limit`: MAE(A, B) must be below this
fn three_way_check(
    label: &str,
    our_jpeg: &[u8],
    ref_jpeg: &[u8],
    original: &[u8],
    is_gray: bool,
    b_ge_c_ratio: f64,
    ab_mae_limit: f64,
) {
    // A = our encode → our decode
    let our_decoded = rasmcore_jpeg::decode(our_jpeg).unwrap();
    let a_pixels = &our_decoded.pixels;

    // B = our encode → ref decode
    let ref_our = image::load_from_memory_with_format(our_jpeg, image::ImageFormat::Jpeg).unwrap();
    let b_pixels = if is_gray {
        ref_our.to_luma8().into_raw()
    } else {
        ref_our.to_rgb8().into_raw()
    };

    // C = ref encode → ref decode
    let ref_ref = image::load_from_memory_with_format(ref_jpeg, image::ImageFormat::Jpeg).unwrap();
    let c_pixels = if is_gray {
        ref_ref.to_luma8().into_raw()
    } else {
        ref_ref.to_rgb8().into_raw()
    };

    let b_quality = psnr(&b_pixels, original);
    let c_quality = psnr(&c_pixels, original);
    let a_vs_b = mae(a_pixels, &b_pixels);

    eprintln!(
        "{label}: B={b_quality:.1}dB, C={c_quality:.1}dB, A≈B MAE={a_vs_b:.2}, sizes: ours={} ref={}",
        our_jpeg.len(),
        ref_jpeg.len()
    );

    assert!(
        b_quality >= c_quality * b_ge_c_ratio,
        "{label}: B ({b_quality:.1}dB) should be >= {:.0}% of C ({c_quality:.1}dB)",
        b_ge_c_ratio * 100.0
    );
    assert!(
        a_vs_b < ab_mae_limit,
        "{label}: A≈B MAE ({a_vs_b:.2}) should be < {ab_mae_limit}"
    );

    // File size check: our output should be within 3x of reference.
    // Magick uses optimized Huffman tables; we use standard tables, so our
    // files are larger. The gap is especially wide for high-frequency patterns
    // (checkerboard) and small images.
    let size_ratio = our_jpeg.len() as f64 / ref_jpeg.len() as f64;
    assert!(
        size_ratio < 3.0,
        "{label}: size ratio {size_ratio:.2} exceeds 3.0x"
    );
}

/// Encode with image crate as reference (baseline sequential only).
fn image_crate_encode(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let img = image::RgbImage::from_raw(w, h, pixels.to_vec()).unwrap();
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
    image::DynamicImage::ImageRgb8(img)
        .write_with_encoder(encoder)
        .unwrap();
    buf
}

fn image_crate_encode_gray(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let img = image::GrayImage::from_raw(w, h, pixels.to_vec()).unwrap();
    let mut buf = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buf);
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
    img.write_with_encoder(encoder).unwrap();
    buf
}

// ─── Baseline 4:2:0 Quality Sweep ─────────────────────────────────────────

#[test]
fn baseline_420_quality_sweep() {
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);

    for quality in [25u8, 50, 75, 95] {
        let config = rasmcore_jpeg::EncodeConfig {
            quality,
            quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
            subsampling: rasmcore_jpeg::ChromaSubsampling::Quarter420,
            ..Default::default()
        };
        let our = rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config)
            .unwrap();
        let ref_jpeg = image_crate_encode(&original, w, h, quality);

        three_way_check(
            &format!("baseline_420_Q{quality}"),
            &our,
            &ref_jpeg,
            &original,
            false,
            0.9,
            1.0,
        );
    }
}

// ─── Baseline 4:2:2 ───────────────────────────────────────────────────────

#[test]
fn baseline_422() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);
    let quality = 85u8;

    let config = rasmcore_jpeg::EncodeConfig {
        quality,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        subsampling: rasmcore_jpeg::ChromaSubsampling::Half422,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    // image crate doesn't support 4:2:2 encoding, use magick
    let ref_jpeg = magick_encode(
        &original,
        w,
        h,
        quality as u32,
        &["-sampling-factor", "2x1"],
    )
    .expect("magick 422 failed");

    three_way_check("baseline_422", &our, &ref_jpeg, &original, false, 0.9, 1.0);
}

// ─── Baseline 4:4:4 ───────────────────────────────────────────────────────

#[test]
fn baseline_444() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);
    let quality = 85u8;

    let config = rasmcore_jpeg::EncodeConfig {
        quality,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        subsampling: rasmcore_jpeg::ChromaSubsampling::None444,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    let ref_jpeg = magick_encode(
        &original,
        w,
        h,
        quality as u32,
        &["-sampling-factor", "1x1"],
    )
    .expect("magick 444 failed");

    three_way_check("baseline_444", &our, &ref_jpeg, &original, false, 0.9, 1.0);
}

// ─── Grayscale ─────────────────────────────────────────────────────────────

#[test]
fn grayscale_quality_sweep() {
    let (w, h) = (32, 32);
    let original = gray_gradient(w, h);

    for quality in [75u8, 95] {
        let config = rasmcore_jpeg::EncodeConfig {
            quality,
            quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
            ..Default::default()
        };
        let our =
            rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Gray8, &config)
                .unwrap();
        let ref_jpeg = image_crate_encode_gray(&original, w, h, quality);

        three_way_check(
            &format!("gray_Q{quality}"),
            &our,
            &ref_jpeg,
            &original,
            true,
            0.9,
            1.0,
        );
    }
}

// ─── Progressive ───────────────────────────────────────────────────────────

#[test]
fn progressive_420() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);
    let quality = 85u8;

    let config = rasmcore_jpeg::EncodeConfig {
        quality,
        progressive: true,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    let ref_jpeg = magick_encode(&original, w, h, quality as u32, &["-interlace", "JPEG"])
        .expect("magick progressive failed");

    three_way_check(
        "progressive_420",
        &our,
        &ref_jpeg,
        &original,
        false,
        0.9,
        1.0,
    );
}

#[test]
fn progressive_444() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);
    let quality = 85u8;

    let config = rasmcore_jpeg::EncodeConfig {
        quality,
        progressive: true,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        subsampling: rasmcore_jpeg::ChromaSubsampling::None444,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    let ref_jpeg = magick_encode(
        &original,
        w,
        h,
        quality as u32,
        &["-interlace", "JPEG", "-sampling-factor", "1x1"],
    )
    .expect("magick progressive 444 failed");

    three_way_check(
        "progressive_444",
        &our,
        &ref_jpeg,
        &original,
        false,
        0.9,
        1.0,
    );
}

#[test]
fn progressive_grayscale() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }
    let (w, h) = (32, 32);
    let original = gray_gradient(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        progressive: true,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Gray8, &config).unwrap();

    let ref_jpeg = magick_encode_gray(&original, w, h, 85, &["-interlace", "JPEG"])
        .expect("magick progressive gray failed");

    three_way_check(
        "progressive_gray",
        &our,
        &ref_jpeg,
        &original,
        true,
        0.9,
        1.0,
    );
}

// ─── Trellis Quantization ──────────────────────────────────────────────────

#[test]
fn trellis_420() {
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        trellis: true,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    // Reference: image crate baseline (no trellis available externally)
    let ref_jpeg = image_crate_encode(&original, w, h, 85);

    // Trellis should produce similar or better quality at similar or smaller size
    three_way_check("trellis_420", &our, &ref_jpeg, &original, false, 0.85, 1.0);
}

// ─── Arithmetic Coding ─────────────────────────────────────────────────────

#[test]
fn arithmetic_coding_roundtrip() {
    // Arithmetic coding: A/B only (no external reference encoder supports it).
    // The image crate can't decode arithmetic JPEGs, so we only verify roundtrip.
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        arithmetic_coding: true,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    // A = our encode → our decode
    let decoded = rasmcore_jpeg::decode(&our).unwrap();
    let a_psnr = psnr(&decoded.pixels, &original);

    eprintln!("arithmetic_420: A_psnr={a_psnr:.1}dB, size={}", our.len());

    assert!(
        a_psnr > 30.0,
        "arithmetic roundtrip PSNR should be > 30dB, got {a_psnr:.1}dB"
    );

    // Arithmetic should produce smaller files than Huffman at same quality
    let huffman_config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let huffman = rasmcore_jpeg::encode(
        &original,
        w,
        h,
        rasmcore_jpeg::PixelFormat::Rgb8,
        &huffman_config,
    )
    .unwrap();

    eprintln!(
        "  arithmetic={} bytes vs huffman={} bytes (ratio={:.2})",
        our.len(),
        huffman.len(),
        our.len() as f64 / huffman.len() as f64
    );
    assert!(
        our.len() <= huffman.len(),
        "arithmetic ({}) should be <= huffman ({})",
        our.len(),
        huffman.len()
    );
}

// ─── Checkerboard High-Frequency Pattern ───────────────────────────────────

#[test]
fn checkerboard_420() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }
    let (w, h) = (32, 32);
    let original = checkerboard_pixels(w, h, 4);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    let ref_jpeg = magick_encode(&original, w, h, 85, &[]).expect("magick checkerboard failed");

    three_way_check(
        "checkerboard_420",
        &our,
        &ref_jpeg,
        &original,
        false,
        0.9,
        1.0,
    );
}

// ─── Odd Dimensions ────────────────────────────────────────────────────────

#[test]
fn odd_dimensions_17x13() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }
    let (w, h) = (17, 13);
    let original = gradient_pixels(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 75,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    let ref_jpeg = magick_encode(&original, w, h, 75, &[]).expect("magick odd dims failed");

    three_way_check(
        "odd_17x13",
        &our,
        &ref_jpeg,
        &original,
        false,
        0.9,
        1.0, // Higher tolerance for odd dims + MCU padding
    );
}

// ─── Cross-Decode: External Fixtures ───────────────────────────────────────

#[test]
fn decode_magick_baseline_fixtures() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }

    let cases: Vec<(&str, u32, u32, &[&str])> = vec![
        ("baseline_420", 32, 32, &[]),
        ("baseline_444", 32, 32, &["-sampling-factor", "1x1"]),
        ("progressive_420", 32, 32, &["-interlace", "JPEG"]),
        (
            "progressive_444",
            32,
            32,
            &["-interlace", "JPEG", "-sampling-factor", "1x1"],
        ),
        ("odd_17x13", 17, 13, &[]),
    ];

    for (label, w, h, extra_args) in cases {
        let original = gradient_pixels(w, h);
        let magick_jpeg = magick_encode(&original, w, h, 85, extra_args)
            .expect(&format!("{label}: magick failed"));

        // Our decoder
        let our = rasmcore_jpeg::decode(&magick_jpeg);
        assert!(our.is_ok(), "{label}: our decode failed: {:?}", our.err());
        let our = our.unwrap();
        assert_eq!(our.width, w, "{label}: width");
        assert_eq!(our.height, h, "{label}: height");

        // Ref decoder
        let ref_img =
            image::load_from_memory_with_format(&magick_jpeg, image::ImageFormat::Jpeg).unwrap();
        let ref_px = ref_img.to_rgb8().into_raw();

        let our_psnr = psnr(&our.pixels, &original);
        let cross_mae = mae(&our.pixels, &ref_px);

        eprintln!("{label}: PSNR={our_psnr:.1}dB, cross-MAE={cross_mae:.2}");

        assert!(our_psnr > 15.0, "{label}: PSNR too low: {our_psnr:.1}dB");
        // SA refinement gap for progressive magick files
        let mae_limit = if label.contains("progressive") || label.contains("odd") {
            4.0
        } else {
            1.0
        };
        assert!(
            cross_mae < mae_limit,
            "{label}: cross-decode MAE {cross_mae:.2} > {mae_limit}"
        );
    }
}

// ─── Progressive Quality Sweep ─────────────────────────────────────────────

#[test]
fn progressive_quality_sweep() {
    if !magick_available() {
        eprintln!("SKIP: magick not available");
        return;
    }
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);

    for quality in [50u8, 85, 95] {
        let config = rasmcore_jpeg::EncodeConfig {
            quality,
            progressive: true,
            quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
            ..Default::default()
        };
        let our = rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config)
            .unwrap();

        let ref_jpeg = magick_encode(&original, w, h, quality as u32, &["-interlace", "JPEG"])
            .expect("magick failed");

        three_way_check(
            &format!("progressive_Q{quality}"),
            &our,
            &ref_jpeg,
            &original,
            false,
            0.9,
            1.0,
        );
    }
}

// ─── Optimized Huffman Tables ─────────────────────────────────────────────

#[test]
fn optimized_huffman_420() {
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        optimize_huffman: true,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    // Reference: image crate baseline
    let ref_jpeg = image_crate_encode(&original, w, h, 85);

    // Optimized Huffman should be decodable and comparable quality
    three_way_check(
        "optimized_huffman_420",
        &our,
        &ref_jpeg,
        &original,
        false,
        0.9,
        1.0,
    );
}

#[test]
fn optimized_huffman_reduces_size() {
    let (w, h) = (64, 64);
    // Use photo-like content for best savings demonstration
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let base_r = ((x as f64 / w as f64 * std::f64::consts::PI).sin() * 127.0 + 128.0) as u8;
            let base_g = ((y as f64 / h as f64 * std::f64::consts::PI).cos() * 100.0 + 128.0) as u8;
            let noise = ((x * 17 + y * 31) % 15) as u8;
            pixels.push(base_r.wrapping_add(noise));
            pixels.push(base_g.wrapping_sub(noise.min(base_g)));
            pixels.push(128u8);
        }
    }

    for quality in [50u8, 75, 85] {
        let standard = rasmcore_jpeg::encode(
            &pixels,
            w,
            h,
            rasmcore_jpeg::PixelFormat::Rgb8,
            &rasmcore_jpeg::EncodeConfig {
                quality,
                quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
                ..Default::default()
            },
        )
        .unwrap();

        let optimized = rasmcore_jpeg::encode(
            &pixels,
            w,
            h,
            rasmcore_jpeg::PixelFormat::Rgb8,
            &rasmcore_jpeg::EncodeConfig {
                quality,
                optimize_huffman: true,
                quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
                ..Default::default()
            },
        )
        .unwrap();

        let savings = (1.0 - optimized.len() as f64 / standard.len() as f64) * 100.0;
        eprintln!(
            "OptHuff Q{quality}: standard={} optimized={} savings={savings:.1}%",
            standard.len(),
            optimized.len()
        );

        // Optimized should be smaller or equal
        assert!(
            optimized.len() <= standard.len(),
            "Q{quality}: optimized ({}) should be <= standard ({})",
            optimized.len(),
            standard.len()
        );

        // Both should be decodable by image crate
        assert!(
            image::load_from_memory_with_format(&optimized, image::ImageFormat::Jpeg).is_ok(),
            "Q{quality}: optimized output not decodable"
        );

        // PSNR should be identical (same quantized coefficients)
        let std_img = image::load_from_memory_with_format(&standard, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb8();
        let opt_img = image::load_from_memory_with_format(&optimized, image::ImageFormat::Jpeg)
            .unwrap()
            .to_rgb8();
        let std_psnr = psnr(&pixels, std_img.as_raw());
        let opt_psnr = psnr(&pixels, opt_img.as_raw());

        eprintln!("  std_psnr={std_psnr:.2}dB opt_psnr={opt_psnr:.2}dB");

        // PSNR should be within 0.1dB (different Huffman tables, same coefficients)
        assert!(
            (std_psnr - opt_psnr).abs() < 0.1,
            "Q{quality}: PSNR divergence {:.2}dB (std={std_psnr:.2} opt={opt_psnr:.2})",
            (std_psnr - opt_psnr).abs()
        );
    }
}

#[test]
fn optimized_huffman_with_trellis() {
    let (w, h) = (32, 32);
    let original = gradient_pixels(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        optimize_huffman: true,
        trellis: true,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap();

    // Must be decodable
    let img = image::load_from_memory_with_format(&our, image::ImageFormat::Jpeg);
    assert!(img.is_ok(), "trellis+optimized huffman not decodable");

    let decoded = img.unwrap().to_rgb8();
    let quality = psnr(&original, decoded.as_raw());
    eprintln!(
        "trellis+optimized_huffman: PSNR={quality:.1}dB size={}",
        our.len()
    );
    assert!(quality > 30.0, "PSNR too low: {quality:.1}dB");
}

#[test]
fn optimized_huffman_grayscale() {
    let (w, h) = (32, 32);
    let original = gray_gradient(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        optimize_huffman: true,
        quant_preset: rasmcore_jpeg::QuantPreset::AnnexK,
        ..Default::default()
    };
    let our =
        rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Gray8, &config).unwrap();

    // Must be decodable
    let img = image::load_from_memory_with_format(&our, image::ImageFormat::Jpeg);
    assert!(img.is_ok(), "grayscale optimized huffman not decodable");
}

// ═══════════════════════════════════════════════════════════════════════════
// JPEG Trellis Benchmark Suite
// ═══════════════════════════════════════════════════════════════════════════
//
// Comprehensive benchmark of all trellis features: AC trellis, DC trellis,
// adaptive lambda, progressive+trellis, optimized Huffman+trellis.

/// Encode with specific config and return (jpeg_bytes, psnr, file_size).
fn encode_and_measure(
    pixels: &[u8],
    w: u32,
    h: u32,
    config: &rasmcore_jpeg::EncodeConfig,
) -> (Vec<u8>, f64, usize) {
    let jpeg =
        rasmcore_jpeg::encode(pixels, w, h, rasmcore_jpeg::PixelFormat::Rgb8, config).unwrap();
    let decoded = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg)
        .expect("encoded JPEG must be decodable")
        .to_rgb8();
    let quality = psnr(pixels, decoded.as_raw());
    let size = jpeg.len();
    (jpeg, quality, size)
}

// ─── Quality Sweep ──────────────────────────────────────────────────────────

#[test]
fn trellis_quality_sweep_256() {
    let (w, h) = (256, 256);
    let original = gradient_pixels(w, h);
    let has_magick = magick_available();

    eprintln!("\n=== JPEG Trellis Quality Sweep (256x256 gradient) ===");
    if has_magick {
        eprintln!(
            "{:<6} {:>10} {:>10} {:>10} {:>8} {:>12} {:>12}",
            "Q", "NoTrellis", "Trellis", "Magick", "Savings", "vs_Magick_sz", "vs_Magick_q"
        );
    } else {
        eprintln!(
            "{:<6} {:>10} {:>10} {:>8} {:>10}",
            "Q", "NoTrellis", "Trellis", "Savings", "PSNR_diff"
        );
        eprintln!("  (ImageMagick not available — skipping reference comparison)");
    }

    for quality in [25, 50, 75, 85, 95] {
        let config_base = rasmcore_jpeg::EncodeConfig {
            quality,
            trellis: false,
            ..Default::default()
        };
        let config_trellis = rasmcore_jpeg::EncodeConfig {
            quality,
            trellis: true,
            ..Default::default()
        };

        let (_, psnr_base, size_base) = encode_and_measure(&original, w, h, &config_base);
        let (_, psnr_trellis, size_trellis) = encode_and_measure(&original, w, h, &config_trellis);

        let savings_pct = (1.0 - size_trellis as f64 / size_base as f64) * 100.0;
        let psnr_diff = psnr_trellis - psnr_base;

        // Reference comparison against ImageMagick
        if has_magick {
            if let Some(magick_jpeg) = magick_encode(&original, w, h, quality as u32, &[]) {
                let magick_decoded =
                    image::load_from_memory_with_format(&magick_jpeg, image::ImageFormat::Jpeg)
                        .unwrap()
                        .to_rgb8();
                let magick_psnr = psnr(&original, magick_decoded.as_raw());
                let magick_size = magick_jpeg.len();
                let size_ratio = size_trellis as f64 / magick_size as f64;
                let quality_ratio = psnr_trellis / magick_psnr;

                eprintln!(
                    "Q{:<5} {:>10} {:>10} {:>10} {:>7.1}% {:>11.2}x {:>11.2}x",
                    quality,
                    size_base,
                    size_trellis,
                    magick_size,
                    savings_pct,
                    size_ratio,
                    quality_ratio
                );
            } else {
                eprintln!(
                    "Q{:<5} {:>10} {:>10} {:>10} {:>7.1}% {:>12} {:>12}",
                    quality, size_base, size_trellis, "N/A", savings_pct, "N/A", "N/A"
                );
            }
        } else {
            eprintln!(
                "Q{:<5} {:>10} {:>10} {:>7.1}% {:>+9.2}dB",
                quality, size_base, size_trellis, savings_pct, psnr_diff
            );
        }

        // Trellis should produce smaller or equal files
        assert!(
            size_trellis <= size_base + 50,
            "Q{quality}: trellis ({size_trellis}) should not be much larger than base ({size_base})"
        );

        // PSNR should not degrade more than 3.5dB
        assert!(
            psnr_diff > -3.5,
            "Q{quality}: trellis PSNR ({psnr_trellis:.1}dB) too far below base ({psnr_base:.1}dB)"
        );
    }
}

/// Compare our trellis output directly against ImageMagick across multiple patterns.
#[test]
fn trellis_vs_magick_reference() {
    if !magick_available() {
        eprintln!("SKIP: ImageMagick not available for reference comparison");
        return;
    }

    let patterns: Vec<(&str, Vec<u8>, u32, u32)> = vec![
        ("gradient_256", gradient_pixels(256, 256), 256, 256),
        ("checker_128", checkerboard_pixels(128, 128, 8), 128, 128),
        ("gradient_64", gradient_pixels(64, 64), 64, 64),
    ];

    eprintln!("\n=== Trellis vs ImageMagick Reference (Q85) ===");
    eprintln!(
        "{:<20} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Pattern", "Ours_sz", "Magick_sz", "Size_ratio", "Ours_dB", "Quality_ratio"
    );

    for (name, original, w, h) in &patterns {
        let config = rasmcore_jpeg::EncodeConfig {
            quality: 85,
            trellis: true,
            optimize_huffman: true,
            ..Default::default()
        };
        let (_, our_psnr, our_size) = encode_and_measure(original, *w, *h, &config);

        if let Some(magick_jpeg) = magick_encode(original, *w, *h, 85, &[]) {
            let magick_decoded =
                image::load_from_memory_with_format(&magick_jpeg, image::ImageFormat::Jpeg)
                    .unwrap()
                    .to_rgb8();
            let magick_psnr = psnr(original, magick_decoded.as_raw());
            let magick_size = magick_jpeg.len();
            let size_ratio = our_size as f64 / magick_size as f64;
            let quality_ratio = if magick_psnr.is_infinite() {
                1.0
            } else {
                our_psnr / magick_psnr
            };

            eprintln!(
                "{:<20} {:>10} {:>10} {:>9.2}x {:>9.1}dB {:>11.2}x",
                name, our_size, magick_size, size_ratio, our_psnr, quality_ratio
            );

            // Our trellis should achieve at least 85% of ImageMagick quality
            assert!(
                quality_ratio > 0.85 || our_psnr > 35.0,
                "{name}: quality ratio {quality_ratio:.2} below 0.85"
            );
        }
    }
}

// ─── Feature Combinations ───────────────────────────────────────────────────

#[test]
fn progressive_trellis_parity() {
    let (w, h) = (256, 256);
    let original = gradient_pixels(w, h);

    for quality in [75, 85] {
        let config = rasmcore_jpeg::EncodeConfig {
            quality,
            progressive: true,
            trellis: true,
            ..Default::default()
        };
        let (jpeg, q, size) = encode_and_measure(&original, w, h, &config);
        eprintln!("Progressive+Trellis Q{quality}: {q:.1}dB, {size}B");

        // Must be decodable
        assert!(
            q > 20.0,
            "progressive+trellis Q{quality} PSNR too low: {q:.1}dB"
        );

        // Compare to progressive without trellis
        let config_no_trellis = rasmcore_jpeg::EncodeConfig {
            quality,
            progressive: true,
            trellis: false,
            ..Default::default()
        };
        let (_, _, size_no_trellis) = encode_and_measure(&original, w, h, &config_no_trellis);
        eprintln!(
            "  vs no-trellis: {size_no_trellis}B (savings: {:.1}%)",
            (1.0 - size as f64 / size_no_trellis as f64) * 100.0
        );
    }
}

#[test]
fn trellis_plus_optimized_huffman() {
    let (w, h) = (256, 256);
    let original = gradient_pixels(w, h);

    let configs: Vec<(&str, rasmcore_jpeg::EncodeConfig)> = vec![
        (
            "baseline",
            rasmcore_jpeg::EncodeConfig {
                quality: 85,
                ..Default::default()
            },
        ),
        (
            "trellis",
            rasmcore_jpeg::EncodeConfig {
                quality: 85,
                trellis: true,
                ..Default::default()
            },
        ),
        (
            "opt_huffman",
            rasmcore_jpeg::EncodeConfig {
                quality: 85,
                optimize_huffman: true,
                ..Default::default()
            },
        ),
        (
            "trellis+opt_huffman",
            rasmcore_jpeg::EncodeConfig {
                quality: 85,
                trellis: true,
                optimize_huffman: true,
                ..Default::default()
            },
        ),
    ];

    eprintln!("\n=== Feature Combination Comparison (256x256, Q85) ===");
    let mut baseline_size = 0;
    for (name, config) in &configs {
        let (_, q, size) = encode_and_measure(&original, w, h, config);
        if *name == "baseline" {
            baseline_size = size;
        }
        let savings = if baseline_size > 0 {
            (1.0 - size as f64 / baseline_size as f64) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:<25} {:>8}B  {:.1}dB  ({:+.1}% vs baseline)",
            name, size, q, savings
        );
    }
}

// ─── Scale Tests ────────────────────────────────────────────────────────────

#[test]
fn trellis_large_512() {
    let (w, h) = (512, 512);
    let original = gradient_pixels(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        trellis: true,
        ..Default::default()
    };
    let (_, q, size) = encode_and_measure(&original, w, h, &config);
    eprintln!("Trellis 512x512 Q85: {q:.1}dB, {size}B");
    assert!(q > 25.0, "512x512 trellis PSNR too low: {q:.1}dB");

    let config_base = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        trellis: false,
        ..Default::default()
    };
    let (_, _, size_base) = encode_and_measure(&original, w, h, &config_base);
    let savings = (1.0 - size as f64 / size_base as f64) * 100.0;
    eprintln!("  vs no-trellis: {size_base}B (savings: {savings:.1}%)");
    // With mozjpeg-aligned lambda (quality-preserving), trellis may not save bytes
    // on simple content — it optimizes rate-distortion, not just rate.
    assert!(
        savings > -5.0,
        "trellis should not increase size by more than 5% at 512x512"
    );
}

#[test]
#[ignore] // Large image — run with --include-ignored
fn trellis_large_1024() {
    let (w, h) = (1024, 1024);
    let original = gradient_pixels(w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        trellis: true,
        optimize_huffman: true,
        ..Default::default()
    };
    let (_, q, size) = encode_and_measure(&original, w, h, &config);

    let config_base = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        ..Default::default()
    };
    let (_, q_base, size_base) = encode_and_measure(&original, w, h, &config_base);

    let savings = (1.0 - size as f64 / size_base as f64) * 100.0;
    eprintln!("\n=== 1024x1024 Trellis+OptHuff Q85 ===");
    eprintln!("  Base:    {size_base}B, {q_base:.1}dB");
    eprintln!("  Trellis: {size}B, {q:.1}dB");
    eprintln!("  Savings: {savings:.1}%");
}

// ─── Checkerboard (stress case) ─────────────────────────────────────────────

#[test]
fn trellis_checkerboard_improvement() {
    let (w, h) = (128, 128);
    let original = checkerboard_pixels(w, h, 8);

    let config_base = rasmcore_jpeg::EncodeConfig {
        quality: 75,
        ..Default::default()
    };
    let config_trellis = rasmcore_jpeg::EncodeConfig {
        quality: 75,
        trellis: true,
        ..Default::default()
    };

    let (_, q_base, size_base) = encode_and_measure(&original, w, h, &config_base);
    let (_, q_trellis, size_trellis) = encode_and_measure(&original, w, h, &config_trellis);

    let savings = (1.0 - size_trellis as f64 / size_base as f64) * 100.0;
    eprintln!("\n=== Checkerboard 128x128 Q75 ===");
    eprintln!("  Base:    {size_base}B, {q_base:.1}dB");
    eprintln!("  Trellis: {size_trellis}B, {q_trellis:.1}dB");
    eprintln!("  Savings: {savings:.1}%");

    // Trellis should help significantly on checkerboard
    assert!(
        size_trellis <= size_base,
        "trellis should reduce checkerboard size: {size_trellis} vs {size_base}"
    );
}

// ─── Encode Speed ───────────────────────────────────────────────────────────

#[test]
fn trellis_encode_speed() {
    let (w, h) = (256, 256);
    let original = gradient_pixels(w, h);

    let config_base = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        ..Default::default()
    };
    let config_trellis = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        trellis: true,
        ..Default::default()
    };

    // Warm up
    let _ = encode_and_measure(&original, w, h, &config_base);
    let _ = encode_and_measure(&original, w, h, &config_trellis);

    let iters = 10;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = rasmcore_jpeg::encode(
            &original,
            w,
            h,
            rasmcore_jpeg::PixelFormat::Rgb8,
            &config_base,
        );
    }
    let base_time = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = rasmcore_jpeg::encode(
            &original,
            w,
            h,
            rasmcore_jpeg::PixelFormat::Rgb8,
            &config_trellis,
        );
    }
    let trellis_time = start.elapsed();

    let overhead = trellis_time.as_secs_f64() / base_time.as_secs_f64();
    eprintln!("\n=== Encode Speed (256x256, Q85, {iters} iters) ===");
    eprintln!(
        "  Base:    {:.1}ms/encode",
        base_time.as_secs_f64() * 1000.0 / iters as f64
    );
    eprintln!(
        "  Trellis: {:.1}ms/encode",
        trellis_time.as_secs_f64() * 1000.0 / iters as f64
    );
    eprintln!("  Overhead: {overhead:.2}x");

    // Trellis should not be more than 10x slower (includes bit-width candidate expansion)
    assert!(
        overhead < 10.0,
        "trellis overhead {overhead:.1}x exceeds 10x limit"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// mozjpeg Reference Comparison — The Gold Standard for JPEG Trellis
// ════════════════════��══════════════════════════════════════════════════════

/// Compare our trellis encoder against mozjpeg (the gold standard) across
/// quality levels. Measures quality (PSNR), size, and documents the gap.
#[test]
fn trellis_vs_mozjpeg_quality_sweep() {
    let cjpeg = match mozjpeg_cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: mozjpeg cjpeg not available");
            eprintln!("  Install via: brew install mozjpeg");
            return;
        }
    };
    eprintln!("Using mozjpeg at: {cjpeg}");

    let (w, h) = (256, 256);
    let original = gradient_pixels(w, h);

    eprintln!("\n=== Our Trellis vs mozjpeg (256x256 gradient) ===");
    eprintln!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Q", "Ours_sz", "Moz_sz", "Sz_ratio", "Ours_dB", "Moz_dB", "Q_ratio"
    );

    for quality in [25u8, 50, 75, 85, 95] {
        let config = rasmcore_jpeg::EncodeConfig {
            quality,
            trellis: true,
            optimize_huffman: true,
            ..Default::default()
        };
        let (_, our_psnr, our_size) = encode_and_measure(&original, w, h, &config);

        if let Some(moz_jpeg) = mozjpeg_encode(&original, w, h, quality as u32) {
            let moz_decoded =
                image::load_from_memory_with_format(&moz_jpeg, image::ImageFormat::Jpeg)
                    .unwrap()
                    .to_rgb8();
            let moz_psnr = psnr(&original, moz_decoded.as_raw());
            let moz_size = moz_jpeg.len();

            let size_ratio = our_size as f64 / moz_size as f64;
            let quality_ratio = if moz_psnr.is_infinite() {
                1.0
            } else {
                our_psnr / moz_psnr
            };

            eprintln!(
                "Q{:<5} {:>10} {:>10} {:>9.2}x {:>9.1}dB {:>9.1}dB {:>9.2}x",
                quality, our_size, moz_size, size_ratio, our_psnr, moz_psnr, quality_ratio
            );

            // Our output must be decodable (already verified)
            // Document the gap — don't assert parity since mozjpeg is the gold standard
            // and we expect to be somewhat behind
        } else {
            eprintln!("Q{quality}: mozjpeg encode failed");
        }
    }
}

/// Direct head-to-head: our trellis+opt_huffman vs mozjpeg on multiple patterns.
/// Reports size ratio and quality ratio for each.
#[test]
fn trellis_vs_mozjpeg_patterns() {
    let cjpeg = match mozjpeg_cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: mozjpeg not available");
            return;
        }
    };
    let _ = cjpeg;

    let patterns: Vec<(&str, Vec<u8>, u32, u32)> = vec![
        ("gradient_256", gradient_pixels(256, 256), 256, 256),
        ("gradient_512", gradient_pixels(512, 512), 512, 512),
        ("checker_128", checkerboard_pixels(128, 128, 8), 128, 128),
        ("gradient_64", gradient_pixels(64, 64), 64, 64),
    ];

    eprintln!("\n=== Trellis+OptHuff vs mozjpeg — Multi-Pattern (Q85) ===");
    eprintln!(
        "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Pattern", "Ours_sz", "Moz_sz", "Sz_ratio", "Ours_dB", "Moz_dB", "Q_ratio"
    );

    for (name, original, w, h) in &patterns {
        let config = rasmcore_jpeg::EncodeConfig {
            quality: 85,
            trellis: true,
            optimize_huffman: true,
            ..Default::default()
        };
        let (_, our_psnr, our_size) = encode_and_measure(original, *w, *h, &config);

        if let Some(moz_jpeg) = mozjpeg_encode(original, *w, *h, 85) {
            let moz_decoded =
                image::load_from_memory_with_format(&moz_jpeg, image::ImageFormat::Jpeg)
                    .unwrap()
                    .to_rgb8();
            let moz_psnr = psnr(original, moz_decoded.as_raw());
            let moz_size = moz_jpeg.len();

            let size_ratio = our_size as f64 / moz_size as f64;
            let quality_ratio = if moz_psnr.is_infinite() {
                1.0
            } else {
                our_psnr / moz_psnr
            };

            eprintln!(
                "{:<20} {:>10} {:>10} {:>9.2}x {:>9.1}dB {:>9.1}dB {:>9.2}x",
                name, our_size, moz_size, size_ratio, our_psnr, moz_psnr, quality_ratio
            );
        }
    }
}

/// Encode speed comparison: our trellis vs mozjpeg on 256x256.
#[test]
fn trellis_vs_mozjpeg_speed() {
    let cjpeg = match mozjpeg_cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: mozjpeg not available for speed comparison");
            return;
        }
    };

    let (w, h) = (256, 256);
    let original = gradient_pixels(w, h);
    let ppm_path = write_ppm(&original, w, h);

    let config = rasmcore_jpeg::EncodeConfig {
        quality: 85,
        trellis: true,
        optimize_huffman: true,
        ..Default::default()
    };

    // Warm up
    let _ = rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config);
    let jpg_path = ppm_path.with_extension("speed.jpg");
    let _ = Command::new(cjpeg)
        .args(["-quality", "85", "-outfile"])
        .arg(jpg_path.to_str().unwrap())
        .arg(ppm_path.to_str().unwrap())
        .output();

    let iters = 10;

    // Our encode
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = rasmcore_jpeg::encode(&original, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config);
    }
    let our_time = start.elapsed();

    // mozjpeg encode
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = Command::new(cjpeg)
            .args(["-quality", "85", "-outfile"])
            .arg(jpg_path.to_str().unwrap())
            .arg(ppm_path.to_str().unwrap())
            .output();
    }
    let moz_time = start.elapsed();

    let _ = std::fs::remove_file(&ppm_path);
    let _ = std::fs::remove_file(&jpg_path);

    let our_ms = our_time.as_secs_f64() * 1000.0 / iters as f64;
    let moz_ms = moz_time.as_secs_f64() * 1000.0 / iters as f64;
    let speed_ratio = our_ms / moz_ms;

    eprintln!("\n=== Encode Speed: Ours vs mozjpeg (256x256, Q85, {iters} iters) ===");
    eprintln!("  Our trellis:  {our_ms:.1}ms/encode");
    eprintln!("  mozjpeg:      {moz_ms:.1}ms/encode");
    eprintln!("  Ratio:        {speed_ratio:.2}x (>1 = we're slower)");
    eprintln!("  Note: mozjpeg time includes process spawn overhead");
}

// ═══════════════════════════════════════════════════════════════════════════
// All-Reference Comparison Chart
// ═══════════════════════════════════════════════════════════════════════════

fn vips_available() -> bool {
    Command::new("vips")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn libjpeg_turbo_cjpeg_path() -> Option<&'static str> {
    // System cjpeg is typically libjpeg-turbo (not mozjpeg)
    if let Ok(out) = Command::new("cjpeg").arg("-version").output() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("libjpeg-turbo") {
            return Some("cjpeg");
        }
    }
    None
}

/// Encode via libjpeg-turbo cjpeg (no trellis, fast baseline).
fn libjpeg_turbo_encode(pixels: &[u8], w: u32, h: u32, quality: u32) -> Option<Vec<u8>> {
    let cjpeg = libjpeg_turbo_cjpeg_path()?;
    let ppm_path = write_ppm(pixels, w, h);
    let jpg_path = ppm_path.with_extension("turbo.jpg");
    let result = Command::new(cjpeg)
        .args(["-quality", &quality.to_string()])
        .args(["-outfile", jpg_path.to_str().unwrap()])
        .arg(ppm_path.to_str().unwrap())
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&ppm_path);
    if !result.status.success() {
        let _ = std::fs::remove_file(&jpg_path);
        return None;
    }
    let data = std::fs::read(&jpg_path).ok();
    let _ = std::fs::remove_file(&jpg_path);
    data
}

/// Encode via libvips (uses libjpeg-turbo internally).
fn vips_encode(pixels: &[u8], w: u32, h: u32, quality: u32) -> Option<Vec<u8>> {
    if !vips_available() {
        return None;
    }
    let ppm_path = write_ppm(pixels, w, h);
    let jpg_path = ppm_path.with_extension("vips.jpg");
    let result = Command::new("vips")
        .args([
            "jpegsave",
            ppm_path.to_str().unwrap(),
            jpg_path.to_str().unwrap(),
        ])
        .args(["--Q", &quality.to_string()])
        .output()
        .ok()?;
    let _ = std::fs::remove_file(&ppm_path);
    if !result.status.success() {
        let _ = std::fs::remove_file(&jpg_path);
        return None;
    }
    let data = std::fs::read(&jpg_path).ok();
    let _ = std::fs::remove_file(&jpg_path);
    data
}

fn measure_reference(jpeg: &[u8], original: &[u8]) -> (f64, usize) {
    let decoded = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)
        .unwrap()
        .to_rgb8();
    (psnr(original, decoded.as_raw()), jpeg.len())
}

/// The big chart: our encoder (baseline, trellis, trellis+opt) vs
/// ImageMagick vs libvips vs libjpeg-turbo vs mozjpeg.
/// Measures quality (PSNR), file size, and quality-per-byte efficiency.
#[test]
fn all_reference_comparison_chart() {
    let (w, h) = (256, 256);
    let original = gradient_pixels(w, h);

    let has_magick = magick_available();
    let has_vips = vips_available();
    let has_turbo = libjpeg_turbo_cjpeg_path().is_some();
    let has_mozjpeg = mozjpeg_cjpeg_path().is_some();

    eprintln!("\n{}", "=".repeat(90));
    eprintln!("  ALL-REFERENCE JPEG ENCODER COMPARISON (256x256 gradient)");
    eprintln!("{}", "=".repeat(90));
    eprintln!(
        "Available: magick={has_magick} vips={has_vips} turbo={has_turbo} mozjpeg={has_mozjpeg}\n"
    );

    for quality in [75u8, 85, 95] {
        eprintln!("--- Quality {quality} ---");
        eprintln!(
            "  {:<30} {:>8} {:>10} {:>12}",
            "Encoder", "Size", "PSNR", "dB/KB"
        );

        // Our baseline (no trellis)
        let config_base = rasmcore_jpeg::EncodeConfig {
            quality,
            ..Default::default()
        };
        let (_, base_psnr, base_size) = encode_and_measure(&original, w, h, &config_base);
        let base_eff = base_psnr / (base_size as f64 / 1024.0);
        eprintln!(
            "  {:<30} {:>7}B {:>9.1}dB {:>10.1} dB/KB",
            "rasmcore (baseline)", base_size, base_psnr, base_eff
        );

        // Our trellis
        let config_trellis = rasmcore_jpeg::EncodeConfig {
            quality,
            trellis: true,
            ..Default::default()
        };
        let (_, t_psnr, t_size) = encode_and_measure(&original, w, h, &config_trellis);
        let t_eff = t_psnr / (t_size as f64 / 1024.0);
        eprintln!(
            "  {:<30} {:>7}B {:>9.1}dB {:>10.1} dB/KB",
            "rasmcore (trellis)", t_size, t_psnr, t_eff
        );

        // Our trellis + optimized Huffman
        let config_full = rasmcore_jpeg::EncodeConfig {
            quality,
            trellis: true,
            optimize_huffman: true,
            ..Default::default()
        };
        let (_, f_psnr, f_size) = encode_and_measure(&original, w, h, &config_full);
        let f_eff = f_psnr / (f_size as f64 / 1024.0);
        eprintln!(
            "  {:<30} {:>7}B {:>9.1}dB {:>10.1} dB/KB",
            "rasmcore (trellis+opt_huff)", f_size, f_psnr, f_eff
        );

        // ImageMagick
        if has_magick {
            if let Some(jpeg) = magick_encode(&original, w, h, quality as u32, &[]) {
                let (q, sz) = measure_reference(&jpeg, &original);
                let eff = q / (sz as f64 / 1024.0);
                eprintln!(
                    "  {:<30} {:>7}B {:>9.1}dB {:>10.1} dB/KB",
                    "ImageMagick 7", sz, q, eff
                );
            }
        }

        // libvips
        if has_vips {
            if let Some(jpeg) = vips_encode(&original, w, h, quality as u32) {
                let (q, sz) = measure_reference(&jpeg, &original);
                let eff = q / (sz as f64 / 1024.0);
                eprintln!(
                    "  {:<30} {:>7}B {:>9.1}dB {:>10.1} dB/KB",
                    "libvips 8", sz, q, eff
                );
            }
        }

        // libjpeg-turbo
        if has_turbo {
            if let Some(jpeg) = libjpeg_turbo_encode(&original, w, h, quality as u32) {
                let (q, sz) = measure_reference(&jpeg, &original);
                let eff = q / (sz as f64 / 1024.0);
                eprintln!(
                    "  {:<30} {:>7}B {:>9.1}dB {:>10.1} dB/KB",
                    "libjpeg-turbo", sz, q, eff
                );
            }
        }

        // mozjpeg (gold standard)
        if has_mozjpeg {
            if let Some(jpeg) = mozjpeg_encode(&original, w, h, quality as u32) {
                let (q, sz) = measure_reference(&jpeg, &original);
                let eff = q / (sz as f64 / 1024.0);
                eprintln!(
                    "  {:<30} {:>7}B {:>9.1}dB {:>10.1} dB/KB  ← gold standard",
                    "mozjpeg 4.1", sz, q, eff
                );
            }
        }

        eprintln!();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pixel-Exact Decode Parity — our decoder vs djpeg (libjpeg-turbo 3.1.3)
// ═══════════════════════════════════════════════════════════════════════════
//
// Encode with cjpeg, decode with both djpeg and our decoder, compare
// byte-for-byte. This validates our decoder produces identical output to
// the reference implementation for the same JPEG input.

fn djpeg_available() -> bool {
    Command::new("djpeg")
        .arg("-version")
        .output()
        .map(|o| {
            let stderr = String::from_utf8_lossy(&o.stderr);
            stderr.contains("libjpeg-turbo")
        })
        .unwrap_or(false)
}

fn cjpeg_available() -> bool {
    Command::new("cjpeg")
        .arg("-version")
        .output()
        .map(|o| {
            let stderr = String::from_utf8_lossy(&o.stderr);
            stderr.contains("libjpeg-turbo")
        })
        .unwrap_or(false)
}

/// Encode RGB pixels with cjpeg (libjpeg-turbo) at given quality.
fn cjpeg_encode(pixels: &[u8], w: u32, h: u32, quality: u32, extra_args: &[&str]) -> Vec<u8> {
    let ppm_path = write_ppm(pixels, w, h);
    let jpg_path = ppm_path.with_extension("cjpeg.jpg");
    let mut cmd = Command::new("cjpeg");
    cmd.args(["-quality", &quality.to_string()]);
    for arg in extra_args {
        cmd.arg(*arg);
    }
    cmd.args(["-outfile", jpg_path.to_str().unwrap()])
        .arg(ppm_path.to_str().unwrap());
    let output = cmd.output().expect("cjpeg failed to launch");
    let _ = std::fs::remove_file(&ppm_path);
    assert!(
        output.status.success(),
        "cjpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data = std::fs::read(&jpg_path).unwrap();
    let _ = std::fs::remove_file(&jpg_path);
    data
}

/// Encode grayscale pixels with cjpeg.
fn cjpeg_encode_gray(pixels: &[u8], w: u32, h: u32, quality: u32) -> Vec<u8> {
    let pgm_path = write_pgm(pixels, w, h);
    let jpg_path = pgm_path.with_extension("cjpeg.jpg");
    let output = Command::new("cjpeg")
        .args(["-quality", &quality.to_string()])
        .args(["-grayscale"])
        .args(["-outfile", jpg_path.to_str().unwrap()])
        .arg(pgm_path.to_str().unwrap())
        .output()
        .expect("cjpeg failed to launch");
    let _ = std::fs::remove_file(&pgm_path);
    assert!(
        output.status.success(),
        "cjpeg gray failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data = std::fs::read(&jpg_path).unwrap();
    let _ = std::fs::remove_file(&jpg_path);
    data
}

/// Decode JPEG with djpeg to raw PPM bytes, return just the pixel data.
fn djpeg_decode_rgb(jpeg: &[u8]) -> Vec<u8> {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let jpg_path =
        std::env::temp_dir().join(format!("rasmcore_djpeg_{}_{id}.jpg", std::process::id()));
    let ppm_path = jpg_path.with_extension("ppm");
    std::fs::write(&jpg_path, jpeg).unwrap();
    let output = Command::new("djpeg")
        .args(["-ppm", "-outfile", ppm_path.to_str().unwrap()])
        .arg(jpg_path.to_str().unwrap())
        .output()
        .expect("djpeg failed");
    let _ = std::fs::remove_file(&jpg_path);
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ppm_data = std::fs::read(&ppm_path).unwrap();
    let _ = std::fs::remove_file(&ppm_path);
    // Parse PPM: skip header (P6\n<w> <h>\n255\n)
    let header_end = {
        let mut pos = 0;
        let mut newlines = 0;
        for (i, &b) in ppm_data.iter().enumerate() {
            if b == b'\n' {
                newlines += 1;
                if newlines == 3 {
                    pos = i + 1;
                    break;
                }
            }
        }
        pos
    };
    ppm_data[header_end..].to_vec()
}

/// Decode JPEG with djpeg to raw PGM bytes (grayscale).
fn djpeg_decode_gray(jpeg: &[u8]) -> Vec<u8> {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let jpg_path =
        std::env::temp_dir().join(format!("rasmcore_djpeg_{}_{id}.jpg", std::process::id()));
    let pgm_path = jpg_path.with_extension("pgm");
    std::fs::write(&jpg_path, jpeg).unwrap();
    let output = Command::new("djpeg")
        .args(["-grayscale", "-outfile", pgm_path.to_str().unwrap()])
        .arg(jpg_path.to_str().unwrap())
        .output()
        .expect("djpeg failed");
    let _ = std::fs::remove_file(&jpg_path);
    assert!(
        output.status.success(),
        "djpeg gray failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let pgm_data = std::fs::read(&pgm_path).unwrap();
    let _ = std::fs::remove_file(&pgm_path);
    // Parse PGM: skip header (P5\n<w> <h>\n255\n)
    let header_end = {
        let mut pos = 0;
        let mut newlines = 0;
        for (i, &b) in pgm_data.iter().enumerate() {
            if b == b'\n' {
                newlines += 1;
                if newlines == 3 {
                    pos = i + 1;
                    break;
                }
            }
        }
        pos
    };
    pgm_data[header_end..].to_vec()
}

fn max_err(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Count pixels that differ
fn diff_count(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|&(&x, &y)| x != y).count()
}

// ─── Pixel-Exact Decode Tests ─────────────────────────────────────────────

#[test]
fn decode_parity_solid_color_baseline_444() {
    assert!(djpeg_available(), "djpeg (libjpeg-turbo) required");
    assert!(cjpeg_available(), "cjpeg (libjpeg-turbo) required");

    // Solid red 8x8 at Q95 4:4:4 — minimal JPEG loss, all blocks identical
    let w = 8u32;
    let h = 8;
    let pixels: Vec<u8> = (0..w * h).flat_map(|_| [200u8, 50, 50]).collect();
    let jpeg = cjpeg_encode(&pixels, w, h, 95, &["-sample", "1x1"]);

    let djpeg_out = djpeg_decode_rgb(&jpeg);
    let our_decoded = rasmcore_jpeg::decode(&jpeg).unwrap();

    let m = mae(&our_decoded.pixels, &djpeg_out);
    let mx = max_err(&our_decoded.pixels, &djpeg_out);
    let diffs = diff_count(&our_decoded.pixels, &djpeg_out);
    eprintln!(
        "  decode solid 444: MAE={m:.4}, max_err={mx}, diffs={diffs}/{}",
        djpeg_out.len()
    );
    assert!(
        mx <= 1,
        "Solid color decode: max_err={mx} (expect ≤1 from IDCT rounding)"
    );
}

#[test]
fn decode_parity_gradient_baseline_444() {
    assert!(djpeg_available(), "djpeg (libjpeg-turbo) required");
    assert!(cjpeg_available(), "cjpeg (libjpeg-turbo) required");

    let (w, h) = (32u32, 32);
    let pixels = gradient_pixels(w, h);
    let jpeg = cjpeg_encode(&pixels, w, h, 95, &["-sample", "1x1"]);

    let djpeg_out = djpeg_decode_rgb(&jpeg);
    let our_decoded = rasmcore_jpeg::decode(&jpeg).unwrap();

    let m = mae(&our_decoded.pixels, &djpeg_out);
    let mx = max_err(&our_decoded.pixels, &djpeg_out);
    let diffs = diff_count(&our_decoded.pixels, &djpeg_out);
    eprintln!(
        "  decode gradient 444 Q95: MAE={m:.4}, max_err={mx}, diffs={diffs}/{}",
        djpeg_out.len()
    );
    assert!(
        mx <= 1,
        "Gradient 444 decode: max_err={mx} (expect ≤1 from IDCT rounding)"
    );
}

#[test]
fn decode_parity_gradient_baseline_420() {
    assert!(djpeg_available(), "djpeg (libjpeg-turbo) required");
    assert!(cjpeg_available(), "cjpeg (libjpeg-turbo) required");

    let (w, h) = (32u32, 32);
    let pixels = gradient_pixels(w, h);
    let jpeg = cjpeg_encode(&pixels, w, h, 95, &["-sample", "2x2"]);

    let djpeg_out = djpeg_decode_rgb(&jpeg);
    let our_decoded = rasmcore_jpeg::decode(&jpeg).unwrap();

    let m = mae(&our_decoded.pixels, &djpeg_out);
    let mx = max_err(&our_decoded.pixels, &djpeg_out);
    let diffs = diff_count(&our_decoded.pixels, &djpeg_out);
    eprintln!(
        "  decode gradient 420 Q95: MAE={m:.4}, max_err={mx}, diffs={diffs}/{}",
        djpeg_out.len()
    );
    // 4:2:0 chroma upsampling may differ by ±1 due to different upsample filters
    assert!(
        mx <= 2,
        "Gradient 420 decode: max_err={mx} (expect ≤2 from chroma upsample)"
    );
}

#[test]
fn decode_parity_checkerboard_baseline() {
    assert!(djpeg_available(), "djpeg (libjpeg-turbo) required");
    assert!(cjpeg_available(), "cjpeg (libjpeg-turbo) required");

    let (w, h) = (32u32, 32);
    let pixels = checkerboard_pixels(w, h, 8);
    let jpeg = cjpeg_encode(&pixels, w, h, 95, &["-sample", "1x1"]);

    let djpeg_out = djpeg_decode_rgb(&jpeg);
    let our_decoded = rasmcore_jpeg::decode(&jpeg).unwrap();

    let m = mae(&our_decoded.pixels, &djpeg_out);
    let mx = max_err(&our_decoded.pixels, &djpeg_out);
    eprintln!("  decode checker 444 Q95: MAE={m:.4}, max_err={mx}");
    assert!(
        mx <= 1,
        "Checker 444 decode: max_err={mx} (expect ≤1 from IDCT rounding)"
    );
}

#[test]
fn decode_parity_grayscale() {
    assert!(djpeg_available(), "djpeg (libjpeg-turbo) required");
    assert!(cjpeg_available(), "cjpeg (libjpeg-turbo) required");

    let (w, h) = (32u32, 32);
    let pixels = gray_gradient(w, h);
    let jpeg = cjpeg_encode_gray(&pixels, w, h, 95);

    let djpeg_out = djpeg_decode_gray(&jpeg);
    let our_decoded = rasmcore_jpeg::decode(&jpeg).unwrap();

    let m = mae(&our_decoded.pixels, &djpeg_out);
    let mx = max_err(&our_decoded.pixels, &djpeg_out);
    eprintln!("  decode gray Q95: MAE={m:.4}, max_err={mx}");
    assert!(
        mx <= 1,
        "Gray decode: max_err={mx} (expect ≤1 from IDCT rounding)"
    );
}

#[test]
fn decode_parity_q75_various_subsampling() {
    assert!(djpeg_available(), "djpeg (libjpeg-turbo) required");
    assert!(cjpeg_available(), "cjpeg (libjpeg-turbo) required");

    let (w, h) = (32u32, 32);
    let pixels = gradient_pixels(w, h);

    for (sample, label, max_allowed) in [("1x1", "444", 1), ("2x1", "422", 2), ("2x2", "420", 2)] {
        let jpeg = cjpeg_encode(&pixels, w, h, 75, &["-sample", sample]);
        let djpeg_out = djpeg_decode_rgb(&jpeg);
        let our_decoded = rasmcore_jpeg::decode(&jpeg).unwrap();

        let m = mae(&our_decoded.pixels, &djpeg_out);
        let mx = max_err(&our_decoded.pixels, &djpeg_out);
        eprintln!("  decode {label} Q75: MAE={m:.4}, max_err={mx}");
        assert!(
            mx <= max_allowed,
            "Decode {label} Q75: max_err={mx} > {max_allowed}"
        );
    }
}
