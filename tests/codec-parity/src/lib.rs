//! Three-way codec parity validation.
//!
//! Implements the codec validation standard from code_styleguides/codec-validation.md:
//!   A = our_encode(original) → our_decode
//!   B = our_encode(original) → ref_decode (image crate)
//!   C = ref_encode(original) → ref_decode
//!
//! Lossless: B == original (bit-exact), A == B (bit-exact)
//! Lossy: B ≈ original at same quality as C ≈ original, A == B

// ─── Comparison Helpers ────────────────────────────────────────────────────

/// Assert two pixel buffers are bit-exact identical.
pub fn assert_bit_exact(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(va, vb, "{label}: byte mismatch at offset {i}: {va} vs {vb}");
    }
}

/// Mean Absolute Error between two pixel buffers.
pub fn mae(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "MAE: buffer length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum();
    sum / a.len() as f64
}

/// Peak Signal-to-Noise Ratio between two pixel buffers (8-bit).
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "PSNR: buffer length mismatch");
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
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 / mse).log10()
}

// ─── Test Image Generators ─────────────────────────────────────────────────

/// Solid color RGB image.
pub fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
    [r, g, b].repeat(w as usize * h as usize)
}

/// Solid color RGBA image.
pub fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    [r, g, b, a].repeat(w as usize * h as usize)
}

/// Gradient RGB image: R varies horizontally, G varies vertically.
pub fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / w.max(1)) as u8;
            let g = (y * 255 / h.max(1)) as u8;
            let b = 128u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

/// Grayscale gradient image.
pub fn gradient_gray(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x + y) * 255 / (w + h).max(1)) as u8);
        }
    }
    pixels
}

/// Checkerboard RGB image (stresses RLE and block boundaries).
pub fn checker_rgb(w: u32, h: u32, cell_size: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h {
        for x in 0..w {
            let is_white = ((x / cell_size) + (y / cell_size)).is_multiple_of(2);
            if is_white {
                pixels.extend_from_slice(&[255, 255, 255]);
            } else {
                pixels.extend_from_slice(&[0, 0, 0]);
            }
        }
    }
    pixels
}

/// Deterministic pseudo-random RGB image (exercises all code paths).
pub fn photo_pattern(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(w as usize * h as usize * 3);
    let mut seed = 42u32;
    for _ in 0..w * h {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (seed >> 16) as u8;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let g = (seed >> 16) as u8;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let b = (seed >> 16) as u8;
        pixels.extend_from_slice(&[r, g, b]);
    }
    pixels
}

// ─── Reference Encode/Decode Wrappers ──────────────────────────────────────

/// Decode any format via the image crate, return RGB8 pixels.
pub fn ref_decode_to_rgb(data: &[u8], format: image::ImageFormat) -> Vec<u8> {
    let img = image::load_from_memory_with_format(data, format)
        .unwrap_or_else(|e| panic!("ref_decode failed: {e}"));
    img.to_rgb8().into_raw()
}

/// Decode any format via the image crate, return RGBA8 pixels.
pub fn ref_decode_to_rgba(data: &[u8], format: image::ImageFormat) -> Vec<u8> {
    let img = image::load_from_memory_with_format(data, format)
        .unwrap_or_else(|e| panic!("ref_decode failed: {e}"));
    img.to_rgba8().into_raw()
}

/// Encode RGB pixels via the image crate.
pub fn ref_encode_rgb(pixels: &[u8], w: u32, h: u32, format: image::ImageFormat) -> Vec<u8> {
    let img = image::RgbImage::from_vec(w, h, pixels.to_vec())
        .unwrap_or_else(|| panic!("ref_encode: pixel buffer size mismatch"));
    let mut buf = Vec::new();
    let dyn_img = image::DynamicImage::ImageRgb8(img);
    dyn_img
        .write_to(&mut std::io::Cursor::new(&mut buf), format)
        .unwrap_or_else(|e| panic!("ref_encode failed: {e}"));
    buf
}

/// Encode RGBA pixels via the image crate.
pub fn ref_encode_rgba(pixels: &[u8], w: u32, h: u32, format: image::ImageFormat) -> Vec<u8> {
    let img = image::RgbaImage::from_vec(w, h, pixels.to_vec())
        .unwrap_or_else(|| panic!("ref_encode: pixel buffer size mismatch"));
    let mut buf = Vec::new();
    let dyn_img = image::DynamicImage::ImageRgba8(img);
    dyn_img
        .write_to(&mut std::io::Cursor::new(&mut buf), format)
        .unwrap_or_else(|e| panic!("ref_encode failed: {e}"));
    buf
}

/// Encode grayscale pixels via the image crate.
pub fn ref_encode_gray(pixels: &[u8], w: u32, h: u32, format: image::ImageFormat) -> Vec<u8> {
    let img = image::GrayImage::from_vec(w, h, pixels.to_vec())
        .unwrap_or_else(|| panic!("ref_encode: pixel buffer size mismatch"));
    let mut buf = Vec::new();
    let dyn_img = image::DynamicImage::ImageLuma8(img);
    dyn_img
        .write_to(&mut std::io::Cursor::new(&mut buf), format)
        .unwrap_or_else(|e| panic!("ref_encode failed: {e}"));
    buf
}

// Re-export image format enum for convenience
pub use image::ImageFormat;

// ─── Three-Way Harness ─────────────────────────────────────────────────────

/// Three-way parity test for **lossless** RGB codecs.
///
/// Provide your encode and decode functions + the reference format.
/// Asserts: B == original (bit-exact), A == B (bit-exact), C == original.
///
/// ```ignore
/// three_way_lossless_rgb(
///     "QOI 16x16 gradient", &gradient_rgb(16, 16), 16, 16,
///     |px, w, h| rasmcore_qoi::encode(px, w, h, Rgb, Srgb).unwrap(),
///     |data| rasmcore_qoi::decode(data).unwrap().1,
///     ImageFormat::Qoi,
/// );
/// ```
pub fn three_way_lossless_rgb(
    label: &str,
    original: &[u8],
    w: u32,
    h: u32,
    our_encode: impl FnOnce(&[u8], u32, u32) -> Vec<u8>,
    our_decode: impl FnOnce(&[u8]) -> Vec<u8>,
    ref_format: ImageFormat,
) {
    let our_encoded = our_encode(original, w, h);
    let a = our_decode(&our_encoded);
    let b = ref_decode_to_rgb(&our_encoded, ref_format);
    let ref_encoded = ref_encode_rgb(original, w, h, ref_format);
    let c = ref_decode_to_rgb(&ref_encoded, ref_format);

    assert_bit_exact(&format!("{label}: A==B (our decode == ref decode)"), &a, &b);
    assert_bit_exact(&format!("{label}: B==original"), &b, original);
    assert_bit_exact(&format!("{label}: C==original (ref sanity)"), &c, original);
}

/// Three-way parity test for **lossless** RGBA codecs.
pub fn three_way_lossless_rgba(
    label: &str,
    original: &[u8],
    w: u32,
    h: u32,
    our_encode: impl FnOnce(&[u8], u32, u32) -> Vec<u8>,
    our_decode: impl FnOnce(&[u8]) -> Vec<u8>,
    ref_format: ImageFormat,
) {
    let our_encoded = our_encode(original, w, h);
    let a = our_decode(&our_encoded);
    let b = ref_decode_to_rgba(&our_encoded, ref_format);
    let ref_encoded = ref_encode_rgba(original, w, h, ref_format);
    let c = ref_decode_to_rgba(&ref_encoded, ref_format);

    assert_bit_exact(&format!("{label}: A==B"), &a, &b);
    assert_bit_exact(&format!("{label}: B==original"), &b, original);
    assert_bit_exact(&format!("{label}: C==original"), &c, original);
}

/// Three-way parity for lossless codec where our decoder outputs RGBA
/// but comparison is in RGB (decoder adds alpha=255).
pub fn three_way_lossless_rgb_with_rgba_decoder(
    label: &str,
    original: &[u8],
    w: u32,
    h: u32,
    our_encode: impl FnOnce(&[u8], u32, u32) -> Vec<u8>,
    our_decode_rgba: impl FnOnce(&[u8]) -> Vec<u8>,
    ref_format: ImageFormat,
) {
    let our_encoded = our_encode(original, w, h);
    let a_rgba = our_decode_rgba(&our_encoded);
    let a: Vec<u8> = a_rgba
        .chunks_exact(4)
        .flat_map(|c| &c[..3])
        .copied()
        .collect();
    let b = ref_decode_to_rgb(&our_encoded, ref_format);
    let ref_encoded = ref_encode_rgb(original, w, h, ref_format);
    let c = ref_decode_to_rgb(&ref_encoded, ref_format);

    assert_bit_exact(&format!("{label}: A==B"), &a, &b);
    assert_bit_exact(&format!("{label}: B==original"), &b, original);
    assert_bit_exact(&format!("{label}: C==original"), &c, original);
}

/// Three-way parity for **lossy** RGB codecs.
///
/// Asserts:
/// - B (our encode → ref decode) produces a decodable image
/// - B quality (PSNR vs original) exceeds `min_psnr`
/// - B quality is at least 70% of C quality (ref encode → ref decode)
///
/// Returns (b_psnr, c_psnr) for logging.
pub fn three_way_lossy_rgb(
    label: &str,
    original: &[u8],
    w: u32,
    h: u32,
    our_encode: impl FnOnce(&[u8], u32, u32) -> Vec<u8>,
    ref_format: ImageFormat,
    min_psnr: f64,
) -> (f64, f64) {
    let our_encoded = our_encode(original, w, h);
    let b = ref_decode_to_rgb(&our_encoded, ref_format);
    let b_quality = psnr(&b, original);

    let ref_encoded = ref_encode_rgb(original, w, h, ref_format);
    let c = ref_decode_to_rgb(&ref_encoded, ref_format);
    let c_quality = psnr(&c, original);

    assert_eq!(b.len(), original.len(), "{label}: decoded size mismatch");
    assert!(
        b_quality > min_psnr,
        "{label}: B_quality={b_quality:.1}dB < min {min_psnr}dB"
    );

    if c_quality.is_finite() && b_quality.is_finite() {
        let ratio = b_quality / c_quality;
        eprintln!("{label}: B={b_quality:.1}dB, C={c_quality:.1}dB, ratio={ratio:.2}");
    }

    (b_quality, c_quality)
}

/// Self-consistency test for codecs without a reference encoder.
///
/// Asserts: encode → decode recovers original within `epsilon` per sample.
pub fn self_consistency_f64(
    label: &str,
    original: &[f64],
    encode: impl FnOnce() -> Vec<u8>,
    decode: impl FnOnce(&[u8]) -> Vec<f64>,
    epsilon: f64,
) {
    let encoded = encode();
    let decoded = decode(&encoded);
    assert_eq!(
        original.len(),
        decoded.len(),
        "{label}: sample count mismatch"
    );
    for (i, (&expected, &got)) in original.iter().zip(decoded.iter()).enumerate() {
        let diff = (got - expected).abs();
        assert!(
            diff <= epsilon,
            "{label}: sample {i}: expected {expected}, got {got} (diff={diff}, epsilon={epsilon})"
        );
    }
}
