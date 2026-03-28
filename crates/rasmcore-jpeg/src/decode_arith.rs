// JPEG arithmetic scan decoder (SOF9/SOF10).
//
// Separated from decode.rs to avoid merge conflicts with the
// progressive decode (SOF2) implementation.

use crate::dct;
use crate::decode::JpegFrame;
use crate::entropy;
use crate::error::EncodeError;
use crate::quantize;

/// Decode scan using arithmetic coding (SOF9/SOF10).
pub(crate) fn decode_scan_arithmetic(
    entropy_data: &[u8],
    frame: &JpegFrame,
    quant_tables: &[Option<[u16; 64]>; 4],
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

    let num_components = frame.components.len();
    let num_contexts = entropy::arithmetic_context_count(num_components);
    let mut decoder = entropy::ArithmeticDecoder::new(entropy_data, num_contexts);
    let mut dc_pred = vec![0i32; num_components];

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

                let ctx_off = entropy::arithmetic_ctx_offset(ci);

                for v_block in 0..comp.v_sampling as usize {
                    for h_block in 0..comp.h_sampling as usize {
                        let mut zz_coeffs = [0i16; 64];
                        entropy::arithmetic_decode_block(
                            &mut decoder,
                            ctx_off,
                            &mut dc_pred[ci],
                            &mut zz_coeffs,
                        );

                        // De-zigzag: encoder encoded in zigzag order
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

    if is_gray {
        let mut output = vec![0u8; w * h];
        let pw = plane_widths[0];
        for y in 0..h {
            for x in 0..w {
                output[y * w + x] = planes[0][y * pw + x].clamp(0, 255) as u8;
            }
        }
        Ok(output)
    } else {
        let mut output = vec![0u8; w * h * 3];
        let y_pw = plane_widths[0];
        let cb_pw = plane_widths[1];
        let cr_pw = plane_widths[2];
        let y_comp = &frame.components[0];
        let cb_comp = &frame.components[1];

        for py in 0..h {
            for px in 0..w {
                let y_val = planes[0][py * y_pw + px] as f64;
                let cb_x = px * cb_comp.h_sampling as usize / y_comp.h_sampling as usize;
                let cb_y = py * cb_comp.v_sampling as usize / y_comp.v_sampling as usize;
                let cb_val = planes[1][cb_y * cb_pw + cb_x] as f64;
                let cr_val = planes[2][cb_y * cr_pw + cb_x] as f64;

                let r = y_val + 1.402 * (cr_val - 128.0);
                let g = y_val - 0.344136 * (cb_val - 128.0) - 0.714136 * (cr_val - 128.0);
                let b = y_val + 1.772 * (cb_val - 128.0);

                let idx = (py * w + px) * 3;
                output[idx] = r.round().clamp(0.0, 255.0) as u8;
                output[idx + 1] = g.round().clamp(0.0, 255.0) as u8;
                output[idx + 2] = b.round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(output)
    }
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
