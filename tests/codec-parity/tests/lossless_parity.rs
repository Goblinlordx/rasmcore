//! Three-way lossless parity tests using the generic harness.
//!
//! Each test provides encode/decode closures — the harness handles
//! A/B/C comparison, bit-exact assertions, and error messages.

use codec_parity::*;

// ─── QOI ───────────────────────────────────────────────────────────────────

fn qoi_enc(px: &[u8], w: u32, h: u32) -> Vec<u8> {
    rasmcore_qoi::encode(
        px,
        w,
        h,
        rasmcore_qoi::Channels::Rgb,
        rasmcore_qoi::ColorSpace::Srgb,
    )
    .unwrap()
}
fn qoi_dec(data: &[u8]) -> Vec<u8> {
    rasmcore_qoi::decode(data).unwrap().1
}

#[test]
fn qoi_rgb_solid() {
    three_way_lossless_rgb(
        "QOI solid",
        &solid_rgb(16, 16, 200, 100, 50),
        16,
        16,
        qoi_enc,
        qoi_dec,
        ImageFormat::Qoi,
    );
}
#[test]
fn qoi_rgb_gradient() {
    three_way_lossless_rgb(
        "QOI gradient",
        &gradient_rgb(32, 32),
        32,
        32,
        qoi_enc,
        qoi_dec,
        ImageFormat::Qoi,
    );
}
#[test]
fn qoi_rgb_checker() {
    three_way_lossless_rgb(
        "QOI checker",
        &checker_rgb(16, 16, 4),
        16,
        16,
        qoi_enc,
        qoi_dec,
        ImageFormat::Qoi,
    );
}
#[test]
fn qoi_rgb_1x1() {
    three_way_lossless_rgb(
        "QOI 1x1",
        &solid_rgb(1, 1, 42, 84, 126),
        1,
        1,
        qoi_enc,
        qoi_dec,
        ImageFormat::Qoi,
    );
}

#[test]
fn qoi_rgba_gradient() {
    let mut px = Vec::with_capacity(16 * 16 * 4);
    for y in 0..16u8 {
        for x in 0..16u8 {
            px.extend_from_slice(&[x * 16, y * 16, 128, 200]);
        }
    }
    three_way_lossless_rgba(
        "QOI RGBA",
        &px,
        16,
        16,
        |px, w, h| {
            rasmcore_qoi::encode(
                px,
                w,
                h,
                rasmcore_qoi::Channels::Rgba,
                rasmcore_qoi::ColorSpace::Srgb,
            )
            .unwrap()
        },
        |d| rasmcore_qoi::decode(d).unwrap().1,
        ImageFormat::Qoi,
    );
}

// ─── PNM ───────────────────────────────────────────────────────────────────

fn ppm_enc(px: &[u8], w: u32, h: u32) -> Vec<u8> {
    rasmcore_pnm::encode_ppm(px, w, h).unwrap()
}
fn ppm_dec(d: &[u8]) -> Vec<u8> {
    rasmcore_pnm::decode(d).unwrap().1
}

#[test]
fn pnm_ppm_solid() {
    three_way_lossless_rgb(
        "PPM solid",
        &solid_rgb(16, 16, 255, 0, 0),
        16,
        16,
        ppm_enc,
        ppm_dec,
        ImageFormat::Pnm,
    );
}
#[test]
fn pnm_ppm_gradient() {
    three_way_lossless_rgb(
        "PPM gradient",
        &gradient_rgb(32, 32),
        32,
        32,
        ppm_enc,
        ppm_dec,
        ImageFormat::Pnm,
    );
}
#[test]
fn pnm_ppm_3x7() {
    three_way_lossless_rgb(
        "PPM 3x7",
        &gradient_rgb(3, 7),
        3,
        7,
        ppm_enc,
        ppm_dec,
        ImageFormat::Pnm,
    );
}

#[test]
fn pnm_pgm_three_way() {
    let pixels = gradient_gray(16, 16);
    let encoded = rasmcore_pnm::encode_pgm(&pixels, 16, 16).unwrap();
    let (_, a) = rasmcore_pnm::decode(&encoded).unwrap();
    let b_gray: Vec<u8> = ref_decode_to_rgb(&encoded, ImageFormat::Pnm)
        .chunks_exact(3)
        .map(|c| c[0])
        .collect();
    assert_bit_exact("PGM A==B", &a, &b_gray);
    assert_bit_exact("PGM B==original", &b_gray, &pixels);
}

// ─── BMP ───────────────────────────────────────────────────────────────────

fn bmp_enc(px: &[u8], w: u32, h: u32) -> Vec<u8> {
    rasmcore_bmp::encode_rgb(px, w, h).unwrap()
}
fn bmp_dec(d: &[u8]) -> Vec<u8> {
    rasmcore_bmp::decode(d)
        .unwrap()
        .1
        .chunks_exact(4)
        .flat_map(|c| &c[..3])
        .copied()
        .collect()
}

#[test]
fn bmp_rgb_solid() {
    three_way_lossless_rgb(
        "BMP solid",
        &solid_rgb(16, 16, 0, 255, 0),
        16,
        16,
        bmp_enc,
        bmp_dec,
        ImageFormat::Bmp,
    );
}
#[test]
fn bmp_rgb_gradient() {
    three_way_lossless_rgb(
        "BMP gradient",
        &gradient_rgb(32, 32),
        32,
        32,
        bmp_enc,
        bmp_dec,
        ImageFormat::Bmp,
    );
}
#[test]
fn bmp_rgb_checker() {
    three_way_lossless_rgb(
        "BMP checker",
        &checker_rgb(16, 16, 4),
        16,
        16,
        bmp_enc,
        bmp_dec,
        ImageFormat::Bmp,
    );
}
#[test]
fn bmp_rgb_17x1() {
    three_way_lossless_rgb(
        "BMP 17x1",
        &gradient_rgb(17, 1),
        17,
        1,
        bmp_enc,
        bmp_dec,
        ImageFormat::Bmp,
    );
}

#[test]
fn bmp_rgba() {
    three_way_lossless_rgb_with_rgba_decoder(
        "BMP RGBA",
        &solid_rgb(8, 8, 100, 150, 200),
        8,
        8,
        bmp_enc,
        |d| rasmcore_bmp::decode(d).unwrap().1,
        ImageFormat::Bmp,
    );
}

// ─── TGA ───────────────────────────────────────────────────────────────────

fn tga_enc(px: &[u8], w: u32, h: u32) -> Vec<u8> {
    rasmcore_tga::encode_rgb(px, w as u16, h as u16).unwrap()
}
fn tga_dec(d: &[u8]) -> Vec<u8> {
    rasmcore_tga::decode(d)
        .unwrap()
        .1
        .chunks_exact(4)
        .flat_map(|c| &c[..3])
        .copied()
        .collect()
}

#[test]
fn tga_rgb_solid() {
    three_way_lossless_rgb(
        "TGA solid",
        &solid_rgb(16, 16, 50, 100, 200),
        16,
        16,
        tga_enc,
        tga_dec,
        ImageFormat::Tga,
    );
}
#[test]
fn tga_rgb_gradient() {
    three_way_lossless_rgb(
        "TGA gradient",
        &gradient_rgb(32, 32),
        32,
        32,
        tga_enc,
        tga_dec,
        ImageFormat::Tga,
    );
}
#[test]
fn tga_rgb_3x7() {
    three_way_lossless_rgb(
        "TGA 3x7",
        &gradient_rgb(3, 7),
        3,
        7,
        tga_enc,
        tga_dec,
        ImageFormat::Tga,
    );
}

#[test]
fn tga_rgba() {
    three_way_lossless_rgb_with_rgba_decoder(
        "TGA RGBA",
        &solid_rgb(8, 8, 100, 150, 200),
        8,
        8,
        tga_enc,
        |d| rasmcore_tga::decode(d).unwrap().1,
        ImageFormat::Tga,
    );
}
