//! Tests for grading filters

#[allow(unused_imports)]
use crate::domain::filters::common::*;

#[cfg(test)]
mod cube_lut_tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    #[test]
    fn apply_cube_lut_identity() {
        let n = 4;
        let mut cube = format!("LUT_3D_SIZE {n}\n");
        let scale = 1.0 / (n - 1) as f64;
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    cube.push_str(&format!(
                        "{:.6} {:.6} {:.6}\n",
                        r as f64 * scale,
                        g as f64 * scale,
                        b as f64 * scale
                    ));
                }
            }
        }

        let pixels = vec![128u8, 64, 200, 255, 0, 128]; // 2 RGB pixels
        let info = ImageInfo {
            width: 2,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let result = apply_cube_lut(
            Rect::new(0, 0, info.width, info.height),
            &mut |_| Ok(pixels.to_vec()),
            &info,
            cube,
        )
        .unwrap();
        // Identity LUT: output should be close to input
        for i in 0..pixels.len() {
            assert!(
                (result[i] as i16 - pixels[i] as i16).abs() <= 2,
                "pixel {i}: {} -> {} (expected ~{})",
                pixels[i],
                result[i],
                pixels[i]
            );
        }
    }
}

