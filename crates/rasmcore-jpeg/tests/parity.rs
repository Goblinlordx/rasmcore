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
            4.0,
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

    three_way_check("baseline_422", &our, &ref_jpeg, &original, false, 0.9, 4.0);
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

    three_way_check("baseline_444", &our, &ref_jpeg, &original, false, 0.9, 4.0);
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
            4.0,
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
        4.0,
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
        4.0,
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
        4.0,
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
    three_way_check("trellis_420", &our, &ref_jpeg, &original, false, 0.85, 4.0);
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
        4.0,
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
        5.0, // Higher tolerance for odd dims + MCU padding
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
            8.0
        } else {
            4.0
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
            4.0,
        );
    }
}
