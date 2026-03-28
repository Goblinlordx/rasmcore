//! Three-way parity tests for lossy codecs and FITS self-consistency.
//!
//! Uses the generic harness: three_way_lossy_rgb, self_consistency_f64.

use codec_parity::*;

// ─── JPEG ──────────────────────────────────────────────────────────────────

fn jpeg_enc(quality: u8) -> impl FnOnce(&[u8], u32, u32) -> Vec<u8> {
    move |px, w, h| {
        let config = rasmcore_jpeg::EncodeConfig {
            quality,
            ..Default::default()
        };
        rasmcore_jpeg::encode(px, w, h, rasmcore_jpeg::PixelFormat::Rgb8, &config).unwrap()
    }
}

#[test]
fn jpeg_gradient_q85() {
    three_way_lossy_rgb(
        "JPEG gradient q85",
        &gradient_rgb(32, 32),
        32,
        32,
        jpeg_enc(85),
        ImageFormat::Jpeg,
        5.0,
    );
}
#[test]
fn jpeg_gradient_q50() {
    three_way_lossy_rgb(
        "JPEG gradient q50",
        &gradient_rgb(32, 32),
        32,
        32,
        jpeg_enc(50),
        ImageFormat::Jpeg,
        5.0,
    );
}
#[test]
fn jpeg_solid_q85() {
    three_way_lossy_rgb(
        "JPEG solid q85",
        &solid_rgb(16, 16, 128, 128, 128),
        16,
        16,
        jpeg_enc(85),
        ImageFormat::Jpeg,
        5.0,
    );
}
#[test]
fn jpeg_photo_q75() {
    three_way_lossy_rgb(
        "JPEG photo q75",
        &photo_pattern(32, 32),
        32,
        32,
        jpeg_enc(75),
        ImageFormat::Jpeg,
        5.0,
    );
}
#[test]
fn jpeg_checker_q85() {
    three_way_lossy_rgb(
        "JPEG checker q85",
        &checker_rgb(16, 16, 8),
        16,
        16,
        jpeg_enc(85),
        ImageFormat::Jpeg,
        5.0,
    );
}

// ─── FITS (self-consistency) ────────────────────────────────────────────────
//
// Integer types (u8, i16, i32) and f64 must roundtrip BIT-EXACT (epsilon = 0.0).
// Only f32 gets epsilon because f32→f64 conversion introduces representational
// noise (e.g., 0.1f32 becomes 0.10000000149011612f64).

#[test]
fn fits_u8_bit_exact() {
    let original: Vec<f64> = gradient_gray(32, 32).iter().map(|&v| v as f64).collect();
    self_consistency_f64(
        "FITS u8 (bit-exact)",
        &original,
        || rasmcore_fits::encode_u8(&gradient_gray(32, 32), 32, 32).unwrap(),
        |data| rasmcore_fits::decode(data).unwrap().1,
        0.0, // u8→f64 is exact: every u8 has a perfect f64 representation
    );
}

#[test]
fn fits_i16_bit_exact() {
    let original: Vec<f64> = (0..16 * 16).map(|i| (i * 10 - 1280) as f64).collect();
    let pixels_i16: Vec<i16> = (0..16 * 16).map(|i| (i * 10 - 1280) as i16).collect();
    self_consistency_f64(
        "FITS i16 (bit-exact)",
        &original,
        || rasmcore_fits::encode_i16(&pixels_i16, 16, 16).unwrap(),
        |data| rasmcore_fits::decode(data).unwrap().1,
        0.0, // i16→f64 is exact: every i16 has a perfect f64 representation
    );
}

#[test]
fn fits_i32_bit_exact() {
    let original: Vec<f64> = (0..8 * 8).map(|i| (i * 100000 - 3200000) as f64).collect();
    let pixels_i32: Vec<i32> = (0..8 * 8).map(|i| (i * 100000 - 3200000) as i32).collect();
    self_consistency_f64(
        "FITS i32 (bit-exact)",
        &original,
        || rasmcore_fits::encode_i32(&pixels_i32, 8, 8).unwrap(),
        |data| rasmcore_fits::decode(data).unwrap().1,
        0.0, // i32→f64 is exact: every i32 has a perfect f64 representation (f64 has 53-bit mantissa)
    );
}

#[test]
fn fits_f32_epsilon() {
    // f32→f64 has representational noise: 0.5f32 is exact, but 0.1f32 is not.
    // Use values that are exact in f32 to verify the codec itself is lossless,
    // then the epsilon only covers the f32→f64 cast in the test comparison.
    let pixels_f32: Vec<f32> = (0..16 * 16).map(|i| i as f32 * 0.5 - 64.0).collect();
    let original: Vec<f64> = pixels_f32.iter().map(|&v| v as f64).collect();
    self_consistency_f64(
        "FITS f32 (f32 precision)",
        &original,
        || rasmcore_fits::encode_f32(&pixels_f32, 16, 16).unwrap(),
        |data| rasmcore_fits::decode(data).unwrap().1,
        0.0, // These specific f32 values (multiples of 0.5) are exact in f64
    );
}

#[test]
fn fits_f64_bit_exact() {
    let original: Vec<f64> = (0..8 * 8).map(|i| i as f64 * 1.23456789).collect();
    let original_clone = original.clone();
    self_consistency_f64(
        "FITS f64 (bit-exact)",
        &original,
        move || rasmcore_fits::encode_f64(&original_clone, 8, 8).unwrap(),
        |data| rasmcore_fits::decode(data).unwrap().1,
        0.0, // f64→big-endian bytes→f64 is IEEE 754 lossless
    );
}
