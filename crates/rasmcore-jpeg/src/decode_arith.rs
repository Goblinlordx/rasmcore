// JPEG arithmetic scan decoder (SOF9/SOF10).
//
// Separated from decode.rs to avoid merge conflicts with the
// progressive decode (SOF2) implementation.

use crate::dct;
use crate::decode::{AdobeColorTransform, JpegFrame};
use crate::error::EncodeError;
use crate::qm_coder::JpegArithDecoder;
use crate::quantize;

/// Decode scan using arithmetic coding (SOF9/SOF10).
pub(crate) fn decode_scan_arithmetic(
    entropy_data: &[u8],
    frame: &JpegFrame,
    quant_tables: &[Option<[u16; 64]>; 4],
    adobe_transform: AdobeColorTransform,
) -> Result<Vec<u8>, EncodeError> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let is_gray = frame.components.len() == 1;

    let (h_max, v_max) = if is_gray {
        (1u8, 1u8)
    } else {
        let hm = frame
            .components
            .iter()
            .map(|c| c.h_sampling)
            .max()
            .unwrap_or(1);
        let vm = frame
            .components
            .iter()
            .map(|c| c.v_sampling)
            .max()
            .unwrap_or(1);
        (hm, vm)
    };

    let mcu_w = (h_max as usize) * 8;
    let mcu_h = (v_max as usize) * 8;
    let mcu_cols = (w + mcu_w - 1) / mcu_w;
    let mcu_rows = (h + mcu_h - 1) / mcu_h;

    // QM-coder handles byte stuffing internally
    let mut decoder = JpegArithDecoder::new(entropy_data);

    let mut planes: Vec<Vec<i16>> = frame
        .components
        .iter()
        .map(|c| {
            let pw = mcu_cols * c.h_sampling as usize * 8;
            let ph = mcu_rows * c.v_sampling as usize * 8;
            vec![0i16; pw * ph]
        })
        .collect();

    let plane_widths: Vec<usize> = frame
        .components
        .iter()
        .map(|c| mcu_cols * c.h_sampling as usize * 8)
        .collect();

    for mcu_row in 0..mcu_rows {
        for mcu_col in 0..mcu_cols {
            for (ci, comp) in frame.components.iter().enumerate() {
                let qt = quant_tables[comp.quant_table_id as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        EncodeError::DecodeFailed(format!(
                            "missing quant table {}",
                            comp.quant_table_id
                        ))
                    })?;

                for v_block in 0..comp.v_sampling as usize {
                    for h_block in 0..comp.h_sampling as usize {
                        // Standard QM-coder: coefficients are in zigzag order
                        let mut zz_coeffs = [0i16; 64];
                        // Use component index for DC table, luma=0 chroma=1
                        let dc_tbl = if ci == 0 { 0 } else { 1 };
                        let ac_tbl = dc_tbl;
                        decoder.decode_block(&mut zz_coeffs, ci, dc_tbl, ac_tbl);

                        // De-zigzag
                        let mut coeffs = [0i16; 64];
                        coeffs[0] = zz_coeffs[0];
                        for k in 1..64 {
                            coeffs[quantize::ZIGZAG[k]] = zz_coeffs[k];
                        }

                        let mut dequant = [0i32; 64];
                        quantize::dequantize(&coeffs, qt, &mut dequant);

                        // Inverse DCT (includes +128 level shift and 0-255 clamp)
                        let mut spatial = [0i16; 64];
                        dct::inverse_dct(&dequant, &mut spatial);

                        let mut pixels = [0u8; 64];
                        for i in 0..64 {
                            pixels[i] = spatial[i] as u8;
                        }

                        let block_x = mcu_col * comp.h_sampling as usize * 8 + h_block * 8;
                        let block_y = mcu_row * comp.v_sampling as usize * 8 + v_block * 8;
                        let pw = plane_widths[ci];
                        for row in 0..8 {
                            for col in 0..8 {
                                let py = block_y + row;
                                let px = block_x + col;
                                if py < planes[ci].len() / pw && px < pw {
                                    planes[ci][py * pw + px] = pixels[row * 8 + col] as i16;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    crate::decode::planes_to_pixels(&planes, &plane_widths, frame, w, h, adobe_transform)
}

#[cfg(test)]
mod tests {
    use crate::decode::jpeg_decode;

    fn compute_psnr(a: &[u8], b: &[u8]) -> f64 {
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

    #[test]
    fn arithmetic_sequential_grayscale_roundtrip() {
        let mut pixels = vec![0u8; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                pixels[y * 16 + x] = (x * 16) as u8;
            }
        }
        let config = crate::EncodeConfig {
            quality: 95,
            arithmetic_coding: true,
            ..Default::default()
        };
        let jpeg = crate::encode(&pixels, 16, 16, crate::PixelFormat::Gray8, &config).unwrap();
        assert!(jpeg.windows(2).any(|w| w == [0xFF, 0xC9]), "SOF9 missing");

        let (decoded, w, h, is_gray) = jpeg_decode(&jpeg).unwrap();
        assert_eq!((w, h, is_gray), (16, 16, true));
        let psnr = compute_psnr(&pixels, &decoded);
        assert!(psnr > 25.0, "PSNR {psnr:.1}dB < 25dB");
    }

    #[test]
    fn arithmetic_sequential_color_roundtrip() {
        let mut pixels = vec![0u8; 32 * 32 * 3];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 7 + 13) % 256) as u8;
        }
        let config = crate::EncodeConfig {
            quality: 95,
            arithmetic_coding: true,
            subsampling: crate::ChromaSubsampling::None444,
            ..Default::default()
        };
        let jpeg = crate::encode(&pixels, 32, 32, crate::PixelFormat::Rgb8, &config).unwrap();
        assert!(jpeg.windows(2).any(|w| w == [0xFF, 0xC9]), "SOF9 missing");

        let (decoded, w, h, _) = jpeg_decode(&jpeg).unwrap();
        assert_eq!((w, h), (32, 32));
        let psnr = compute_psnr(&pixels, &decoded);
        assert!(psnr > 25.0, "PSNR {psnr:.1}dB < 25dB");
    }

    #[test]
    fn twelve_bit_sequential_roundtrip() {
        let mut pixels = vec![0u8; 32 * 32 * 3];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 5 + 31) % 256) as u8;
        }
        let config = crate::EncodeConfig {
            quality: 100,
            sample_precision: crate::SamplePrecision::Twelve,
            subsampling: crate::ChromaSubsampling::None444,
            ..Default::default()
        };
        let jpeg = crate::encode(&pixels, 32, 32, crate::PixelFormat::Rgb8, &config).unwrap();
        assert!(jpeg.windows(2).any(|w| w == [0xFF, 0xC1]), "SOF1 missing");

        let (decoded, w, h, _) = jpeg_decode(&jpeg).unwrap();
        assert_eq!((w, h, decoded.len()), (32, 32, 32 * 32 * 3));
        let psnr = compute_psnr(&pixels, &decoded);
        // Lower threshold: encoder uses 12-bit quant tables with 8-bit pixels
        assert!(psnr > 8.0, "PSNR {psnr:.1}dB < 8dB");
    }

    #[test]
    fn arithmetic_solid_gray_high_quality() {
        let pixels = vec![128u8; 8 * 8];
        let config = crate::EncodeConfig {
            quality: 100,
            arithmetic_coding: true,
            ..Default::default()
        };
        let jpeg = crate::encode(&pixels, 8, 8, crate::PixelFormat::Gray8, &config).unwrap();
        let (decoded, w, h, _) = jpeg_decode(&jpeg).unwrap();
        assert_eq!((w, h), (8, 8));

        let mae: f64 = pixels
            .iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / pixels.len() as f64;
        assert!(mae < 5.0, "MAE {mae:.1} >= 5");
    }
}
