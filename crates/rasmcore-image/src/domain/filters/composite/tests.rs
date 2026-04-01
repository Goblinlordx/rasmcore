//! Tests for composite filters

use crate::domain::filters::common::*;

#[cfg(test)]
mod blend_tests {
    use super::*;

    fn bc(a: u8, b: u8, mode: BlendMode) -> u8 {
        blend_channel(a, b, mode)
    }

    #[test]
    fn color_dodge_basics() {
        // Black fg → no change to bg
        assert_eq!(bc(0, 128, BlendMode::ColorDodge), 128);
        // White fg → white output (unless bg is 0)
        assert_eq!(bc(255, 128, BlendMode::ColorDodge), 255);
        // Black bg → stays black
        assert_eq!(bc(128, 0, BlendMode::ColorDodge), 0);
    }

    #[test]
    fn color_burn_basics() {
        // White fg → no change to bg
        assert_eq!(bc(255, 128, BlendMode::ColorBurn), 128);
        // Black fg → black output (unless bg is 255)
        assert_eq!(bc(0, 128, BlendMode::ColorBurn), 0);
        // White bg → stays white
        assert_eq!(bc(128, 255, BlendMode::ColorBurn), 255);
    }

    #[test]
    fn linear_dodge_is_addition() {
        assert_eq!(bc(100, 100, BlendMode::LinearDodge), 200);
        // Clamps at 255
        assert_eq!(bc(200, 200, BlendMode::LinearDodge), 255);
        // Identity: adding 0
        assert_eq!(bc(0, 128, BlendMode::LinearDodge), 128);
    }

    #[test]
    fn linear_burn_basics() {
        // a + b - 1.0 in [0,1], clamps at 0
        assert_eq!(bc(0, 0, BlendMode::LinearBurn), 0);
        assert_eq!(bc(255, 255, BlendMode::LinearBurn), 255);
        // 128/255 + 128/255 - 1.0 ≈ 0.004 → ~1
        assert_eq!(bc(128, 128, BlendMode::LinearBurn), 1);
    }

    #[test]
    fn hard_mix_threshold() {
        // Threshold of VividLight result at 0.5
        assert_eq!(bc(128, 128, BlendMode::HardMix), 255);
        assert_eq!(bc(64, 64, BlendMode::HardMix), 0);
        // fg=0: VividLight(0, b) = ColorBurn(0, b) = 0 → threshold → 0
        assert_eq!(bc(0, 255, BlendMode::HardMix), 0);
        assert_eq!(bc(255, 0, BlendMode::HardMix), 255);
    }

    #[test]
    fn subtract_basics() {
        // bg - fg, clamped at 0
        assert_eq!(bc(0, 128, BlendMode::Subtract), 128);
        assert_eq!(bc(128, 128, BlendMode::Subtract), 0);
        assert_eq!(bc(255, 128, BlendMode::Subtract), 0);
    }

    #[test]
    fn divide_basics() {
        // bg / fg
        assert_eq!(bc(255, 128, BlendMode::Divide), 128);
        // Divide by zero (fg=0) → 255
        assert_eq!(bc(0, 128, BlendMode::Divide), 255);
        assert_eq!(bc(128, 0, BlendMode::Divide), 0);
    }

    #[test]
    fn pin_light_basics() {
        // a <= 0.5: min(b, 2a)
        assert_eq!(bc(0, 128, BlendMode::PinLight), 0);
        // a > 0.5: max(b, 2a-1)
        assert_eq!(bc(255, 0, BlendMode::PinLight), 255);
        // mid: a=128 → 2*128/255 ≈ 1.004 > 0.5 → max(b, 2a-1)
        // 2*128/255 - 1 = 0.004 → max(0.5, 0.004) = 0.5
        assert_eq!(bc(128, 128, BlendMode::PinLight), 128);
    }

    #[test]
    fn vivid_light_basics() {
        // a=0 → color burn with 0 → 0
        assert_eq!(bc(0, 128, BlendMode::VividLight), 0);
        // a=255 → color dodge with 1 → 255 (unless bg=0)
        assert_eq!(bc(255, 128, BlendMode::VividLight), 255);
    }

    #[test]
    fn linear_light_basics() {
        // b + 2a - 1 in [0, 1]
        // 128/255 + 2*128/255 - 1 ≈ 0.506 → 129
        assert_eq!(bc(128, 128, BlendMode::LinearLight), 129);
        // 0 + 0 - 1 = -1 → clamped to 0
        assert_eq!(bc(0, 128, BlendMode::LinearLight), 0);
        // 128/255 + 2*255/255 - 1 = 1.502 → clamped to 255
        assert_eq!(bc(255, 128, BlendMode::LinearLight), 255);
    }

    #[test]
    fn blend_function_extended_modes_rgb8() {
        let fg = vec![200u8, 100, 50];
        let bg = vec![100u8, 200, 150];
        let info = ImageInfo {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        // Just verify all new modes produce a result without panicking
        for mode in [
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::VividLight,
            BlendMode::LinearDodge,
            BlendMode::LinearBurn,
            BlendMode::LinearLight,
            BlendMode::PinLight,
            BlendMode::HardMix,
            BlendMode::Subtract,
            BlendMode::Divide,
        ] {
            let result = blend(&fg, &info, &bg, &info, mode).unwrap();
            assert_eq!(result.len(), 3, "mode {:?} should produce 3 bytes", mode);
        }
    }
}

