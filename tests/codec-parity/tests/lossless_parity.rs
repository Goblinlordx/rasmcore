//! Three-way parity tests for lossless codecs: QOI, PNM, BMP, TGA.
//!
//! For each: A = our_encode→our_decode, B = our_encode→ref_decode, C = ref_encode→ref_decode
//! Assert: B == original (bit-exact), A == B (bit-exact), C == original (bit-exact)

use codec_parity::*;

// ═══════════════════════════════════════════════════════════════════════════
// QOI
// ═══════════════════════════════════════════════════════════════════════════

fn qoi_three_way_rgb(pixels: &[u8], w: u32, h: u32) {
    // Our encode
    let our_encoded = rasmcore_qoi::encode(
        pixels,
        w,
        h,
        rasmcore_qoi::Channels::Rgb,
        rasmcore_qoi::ColorSpace::Srgb,
    )
    .unwrap();

    // A: our_encode → our_decode
    let (_, a) = rasmcore_qoi::decode(&our_encoded).unwrap();

    // B: our_encode → ref_decode
    let b = ref_decode_to_rgb(&our_encoded, ImageFormat::Qoi);

    // C: ref_encode → ref_decode
    let ref_encoded = ref_encode_rgb(pixels, w, h, ImageFormat::Qoi);
    let c = ref_decode_to_rgb(&ref_encoded, ImageFormat::Qoi);

    assert_bit_exact("QOI A==B (our decode matches ref decode)", &a, &b);
    assert_bit_exact(
        "QOI B==original (ref decodes our output correctly)",
        &b,
        pixels,
    );
    assert_bit_exact("QOI C==original (ref roundtrip sanity)", &c, pixels);
}

#[test]
fn qoi_rgb_solid() {
    qoi_three_way_rgb(&solid_rgb(16, 16, 200, 100, 50), 16, 16);
}

#[test]
fn qoi_rgb_gradient() {
    qoi_three_way_rgb(&gradient_rgb(32, 32), 32, 32);
}

#[test]
fn qoi_rgb_checker() {
    qoi_three_way_rgb(&checker_rgb(16, 16, 4), 16, 16);
}

#[test]
fn qoi_rgba_gradient() {
    let w = 16u32;
    let h = 16;
    let mut pixels = Vec::with_capacity(w as usize * h as usize * 4);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 16) as u8);
            pixels.push((y * 16) as u8);
            pixels.push(128);
            pixels.push(200);
        }
    }
    let our_encoded = rasmcore_qoi::encode(
        &pixels,
        w,
        h,
        rasmcore_qoi::Channels::Rgba,
        rasmcore_qoi::ColorSpace::Srgb,
    )
    .unwrap();
    let (_, a) = rasmcore_qoi::decode(&our_encoded).unwrap();
    let b = ref_decode_to_rgba(&our_encoded, ImageFormat::Qoi);
    assert_bit_exact("QOI RGBA A==B", &a, &b);
    assert_bit_exact("QOI RGBA B==original", &b, &pixels);
}

// ═══════════════════════════════════════════════════════════════════════════
// PNM (PPM binary P6)
// ═══════════════════════════════════════════════════════════════════════════

fn ppm_three_way(pixels: &[u8], w: u32, h: u32) {
    let our_encoded = rasmcore_pnm::encode_ppm(pixels, w, h).unwrap();
    let (_, a) = rasmcore_pnm::decode(&our_encoded).unwrap();
    let b = ref_decode_to_rgb(&our_encoded, ImageFormat::Pnm);
    let ref_encoded = ref_encode_rgb(pixels, w, h, ImageFormat::Pnm);
    let c = ref_decode_to_rgb(&ref_encoded, ImageFormat::Pnm);

    assert_bit_exact("PPM A==B", &a, &b);
    assert_bit_exact("PPM B==original", &b, pixels);
    assert_bit_exact("PPM C==original", &c, pixels);
}

#[test]
fn pnm_ppm_solid() {
    ppm_three_way(&solid_rgb(16, 16, 255, 0, 0), 16, 16);
}

#[test]
fn pnm_ppm_gradient() {
    ppm_three_way(&gradient_rgb(32, 32), 32, 32);
}

#[test]
fn pnm_pgm_three_way() {
    let pixels = gradient_gray(16, 16);
    let w = 16u32;
    let h = 16;
    let our_encoded = rasmcore_pnm::encode_pgm(&pixels, w, h).unwrap();
    let (_, a) = rasmcore_pnm::decode(&our_encoded).unwrap();
    let b = ref_decode_to_rgb(&our_encoded, ImageFormat::Pnm);
    // PGM grayscale → ref decodes to RGB (R==G==B==gray)
    // Compare: our decode (gray) vs ref decode (RGB, take R channel)
    let b_gray: Vec<u8> = b.chunks_exact(3).map(|c| c[0]).collect();
    assert_bit_exact("PGM A==B (gray channels)", &a, &b_gray);
    assert_bit_exact("PGM B==original", &b_gray, &pixels);
}

// ═══════════════════════════════════════════════════════════════════════════
// BMP
// ═══════════════════════════════════════════════════════════════════════════

fn bmp_three_way_rgb(pixels: &[u8], w: u32, h: u32) {
    let our_encoded = rasmcore_bmp::encode_rgb(pixels, w, h).unwrap();

    // A: our decode (returns RGBA)
    let (_, a_rgba) = rasmcore_bmp::decode(&our_encoded).unwrap();
    let a_rgb: Vec<u8> = a_rgba
        .chunks_exact(4)
        .flat_map(|c| &c[..3])
        .copied()
        .collect();

    // B: ref decode
    let b = ref_decode_to_rgb(&our_encoded, ImageFormat::Bmp);

    // C: ref roundtrip
    let ref_encoded = ref_encode_rgb(pixels, w, h, ImageFormat::Bmp);
    let c = ref_decode_to_rgb(&ref_encoded, ImageFormat::Bmp);

    assert_bit_exact("BMP RGB A==B", &a_rgb, &b);
    assert_bit_exact("BMP RGB B==original", &b, pixels);
    assert_bit_exact("BMP RGB C==original", &c, pixels);
}

#[test]
fn bmp_rgb_solid() {
    bmp_three_way_rgb(&solid_rgb(16, 16, 0, 255, 0), 16, 16);
}

#[test]
fn bmp_rgb_gradient() {
    bmp_three_way_rgb(&gradient_rgb(32, 32), 32, 32);
}

#[test]
fn bmp_rgb_checker() {
    bmp_three_way_rgb(&checker_rgb(16, 16, 4), 16, 16);
}

#[test]
fn bmp_rgba_three_way() {
    let pixels = solid_rgba(8, 8, 100, 150, 200, 255);
    let w = 8u32;
    let h = 8;
    let our_encoded = rasmcore_bmp::encode_rgba(&pixels, w, h).unwrap();
    let (_, a) = rasmcore_bmp::decode(&our_encoded).unwrap();
    let b = ref_decode_to_rgba(&our_encoded, ImageFormat::Bmp);
    assert_bit_exact("BMP RGBA A==B", &a, &b);
    assert_bit_exact("BMP RGBA B==original", &b, &pixels);
}

// ═══════════════════════════════════════════════════════════════════════════
// TGA
// ═══════════════════════════════════════════════════════════════════════════

fn tga_three_way_rgb(pixels: &[u8], w: u16, h: u16) {
    let our_encoded = rasmcore_tga::encode_rgb(pixels, w, h).unwrap();

    // A: our decode (returns RGBA)
    let (_, a_rgba) = rasmcore_tga::decode(&our_encoded).unwrap();
    let a_rgb: Vec<u8> = a_rgba
        .chunks_exact(4)
        .flat_map(|c| &c[..3])
        .copied()
        .collect();

    // B: ref decode
    let b = ref_decode_to_rgb(&our_encoded, ImageFormat::Tga);

    // C: ref roundtrip
    let ref_encoded = ref_encode_rgb(pixels, w as u32, h as u32, ImageFormat::Tga);
    let c = ref_decode_to_rgb(&ref_encoded, ImageFormat::Tga);

    assert_bit_exact("TGA RGB A==B", &a_rgb, &b);
    assert_bit_exact("TGA RGB B==original", &b, pixels);
    assert_bit_exact("TGA RGB C==original", &c, pixels);
}

#[test]
fn tga_rgb_solid() {
    tga_three_way_rgb(&solid_rgb(16, 16, 50, 100, 200), 16, 16);
}

#[test]
fn tga_rgb_gradient() {
    tga_three_way_rgb(&gradient_rgb(32, 32), 32, 32);
}

#[test]
fn tga_rgba_three_way() {
    let pixels = solid_rgba(8, 8, 100, 150, 200, 180);
    let our_encoded = rasmcore_tga::encode_rgba(&pixels, 8, 8).unwrap();
    let (_, a) = rasmcore_tga::decode(&our_encoded).unwrap();
    let b = ref_decode_to_rgba(&our_encoded, ImageFormat::Tga);
    assert_bit_exact("TGA RGBA A==B", &a, &b);
    assert_bit_exact("TGA RGBA B==original", &b, &pixels);
}

// ═══════════════════════════════════════════════════════════════════════════
// Odd Dimensions (all lossless codecs)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn odd_dimensions_qoi_1x1() {
    qoi_three_way_rgb(&solid_rgb(1, 1, 42, 84, 126), 1, 1);
}

#[test]
fn odd_dimensions_ppm_3x7() {
    ppm_three_way(&gradient_rgb(3, 7), 3, 7);
}

#[test]
fn odd_dimensions_bmp_17x1() {
    bmp_three_way_rgb(&gradient_rgb(17, 1), 17, 1);
}

#[test]
fn odd_dimensions_tga_3x7() {
    tga_three_way_rgb(&gradient_rgb(3, 7), 3, 7);
}
