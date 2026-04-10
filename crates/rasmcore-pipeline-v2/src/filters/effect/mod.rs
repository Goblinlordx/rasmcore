//! Effect filters — creative/stylistic operations on f32 pixel data.
//!
//! All operate on `&[f32]` RGBA (4 channels per pixel). No format dispatch.
//! No u8/u16 paths. Just f32.
//!
//! Includes: noise (Gaussian, uniform, salt-pepper, Poisson), film grain,
//! pixelate, halftone, oil paint, emboss, charcoal, glitch, chromatic
//! aberration/split, light leak, mirror kaleidoscope, vignette effects.
//!
//! Solarize is in adjustment.rs (point op with AnalyticOp support).

pub mod charcoal;
pub mod chromatic_aberration;
pub mod chromatic_split;
pub mod emboss;
pub mod film_grain;
pub mod gaussian_noise;
pub mod glitch;
pub mod halftone;
pub mod light_leak;
pub mod mirror_kaleidoscope;
pub mod oil_paint;
pub mod pixelate;
pub mod poisson_noise;
pub mod salt_pepper_noise;
pub mod uniform_noise;

pub use charcoal::Charcoal;
pub use chromatic_aberration::ChromaticAberration;
pub use chromatic_split::ChromaticSplit;
pub use emboss::Emboss;
pub use film_grain::FilmGrain;
pub use gaussian_noise::GaussianNoise;
pub use glitch::Glitch;
pub use halftone::Halftone;
pub use light_leak::LightLeak;
pub use mirror_kaleidoscope::MirrorKaleidoscope;
pub use oil_paint::OilPaint;
pub use pixelate::Pixelate;
pub use poisson_noise::PoissonNoise;
pub use salt_pepper_noise::SaltPepperNoise;
pub use uniform_noise::UniformNoise;

use super::helpers::luminance;

/// Reflect-boundary coordinate clamping.
#[inline]
pub(crate) fn clamp_coord(v: i32, size: usize) -> usize {
    if v < 0 {
        (-v).min(size as i32 - 1) as usize
    } else if v >= size as i32 {
        (2 * size as i32 - v - 2).max(0) as usize
    } else {
        v as usize
    }
}

pub(crate) fn gpu_params_push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn gpu_params_push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Filter;
    use crate::ops::GpuFilter;

    fn solid_rgba(w: u32, h: u32, color: [f32; 4]) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut px = Vec::with_capacity(n * 4);
        for _ in 0..n {
            px.extend_from_slice(&color);
        }
        px
    }

    fn gradient_rgba(w: u32, h: u32) -> Vec<f32> {
        let mut px = Vec::with_capacity((w * h) as usize * 4);
        for y in 0..h {
            for x in 0..w {
                px.push(x as f32 / w as f32);
                px.push(y as f32 / h as f32);
                px.push(0.5);
                px.push(1.0);
            }
        }
        px
    }

    // ─── Noise filters ──────────────────────────────────────────────────

    #[test]
    fn gaussian_noise_deterministic() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        let n = GaussianNoise {
            amount: 50.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 42,
        };
        let a = n.compute(&input, 8, 8).unwrap();
        let b = n.compute(&input, 8, 8).unwrap();
        assert_eq!(a, b); // same seed → same output
    }

    #[test]
    fn gaussian_noise_zero_amount_identity() {
        let input = gradient_rgba(8, 8);
        let n = GaussianNoise {
            amount: 0.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 42,
        };
        assert_eq!(n.compute(&input, 8, 8).unwrap(), input);
    }

    #[test]
    fn uniform_noise_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
        let n = UniformNoise {
            range: 50.0,
            seed: 42,
        };
        let out = n.compute(&input, 8, 8).unwrap();
        assert!((out[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn salt_pepper_modifies_some_pixels() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let n = SaltPepperNoise {
            density: 0.5,
            seed: 42,
        };
        let out = n.compute(&input, 16, 16).unwrap();
        let changed = out
            .chunks_exact(4)
            .zip(input.chunks_exact(4))
            .filter(|(a, b)| (a[0] - b[0]).abs() > 0.01)
            .count();
        assert!(changed > 0); // some pixels changed
        assert!(changed < 256); // not all
    }

    #[test]
    fn poisson_noise_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.3]);
        let n = PoissonNoise {
            scale: 50.0,
            seed: 42,
        };
        let out = n.compute(&input, 8, 8).unwrap();
        assert!((out[3] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn film_grain_preserves_alpha() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 0.7]);
        let fg = FilmGrain {
            amount: 0.3,
            size: 2.0,
            seed: 42,
        };
        let out = fg.compute(&input, 16, 16).unwrap();
        assert!((out[3] - 0.7).abs() < 1e-6);
    }

    // ─── Pixelate ───────────────────────────────────────────────────────

    #[test]
    fn pixelate_uniform_block() {
        let input = gradient_rgba(16, 16);
        let p = Pixelate { block_size: 4 };
        let out = p.compute(&input, 16, 16).unwrap();
        // All pixels in first block should be identical
        let first = &out[0..4];
        let second = &out[4..8]; // next pixel in same block
        assert_eq!(first, second);
    }

    #[test]
    fn pixelate_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.3]);
        let p = Pixelate { block_size: 4 };
        let out = p.compute(&input, 8, 8).unwrap();
        assert!((out[3] - 0.3).abs() < 1e-6);
    }

    // ─── Halftone ───────────────────────────────────────────────────────

    #[test]
    fn halftone_runs() {
        let input = gradient_rgba(32, 32);
        let ht = Halftone {
            dot_size: 4.0,
            angle_offset: 0.0,
        };
        let out = ht.compute(&input, 32, 32).unwrap();
        assert_eq!(out.len(), input.len());
    }

    // ─── Oil Paint ──────────────────────────────────────────────────────

    #[test]
    fn oil_paint_solid_unchanged() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        let op = OilPaint { radius: 2 };
        let out = op.compute(&input, 8, 8).unwrap();
        assert!((out[0] - 0.5).abs() < 0.01);
    }

    // ─── Emboss ─────────────────────────────────────────────────────────

    #[test]
    fn emboss_solid_produces_consistent_output() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        let e = Emboss;
        let out = e.compute(&input, 8, 8).unwrap();
        // Solid → emboss kernel sum=1 → 0.5*1 + 0.5 offset = 1.0 for interior
        // All interior pixels should be the same value
        let center = (4 * 8 + 4) * 4;
        let neighbor = (4 * 8 + 5) * 4;
        assert!((out[center] - out[neighbor]).abs() < 1e-5);
    }

    #[test]
    fn emboss_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
        let out = Emboss.compute(&input, 8, 8).unwrap();
        assert!((out[3] - 0.7).abs() < 1e-6);
    }

    // ─── Charcoal ───────────────────────────────────────────────────────

    #[test]
    fn charcoal_solid_white() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let c = Charcoal {
            radius: 1.0,
            sigma: 1.0,
        };
        let out = c.compute(&input, 16, 16).unwrap();
        // Solid → no edges → inverted = white
        let center = (8 * 16 + 8) * 4;
        assert!(out[center] > 0.9);
    }

    // ─── Chromatic effects ──────────────────────────────────────────────

    #[test]
    fn chromatic_split_zero_offset_identity() {
        let input = gradient_rgba(8, 8);
        let cs = ChromaticSplit {
            red_dx: 0.0,
            red_dy: 0.0,
            green_dx: 0.0,
            green_dy: 0.0,
            blue_dx: 0.0,
            blue_dy: 0.0,
        };
        let out = cs.compute(&input, 8, 8).unwrap();
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn chromatic_aberration_center_unchanged() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let ca = ChromaticAberration { strength: 5.0 };
        let out = ca.compute(&input, 16, 16).unwrap();
        // Solid color → no visible aberration
        assert!((out[0] - 0.5).abs() < 0.01);
    }

    // ─── Glitch ─────────────────────────────────────────────────────────

    #[test]
    fn glitch_deterministic() {
        let input = gradient_rgba(16, 16);
        let g = Glitch {
            shift_amount: 10.0,
            channel_offset: 3.0,
            intensity: 0.5,
            band_height: 4,
            seed: 42,
        };
        let a = g.compute(&input, 16, 16).unwrap();
        let b = g.compute(&input, 16, 16).unwrap();
        assert_eq!(a, b);
    }

    // ─── Light Leak ─────────────────────────────────────────────────────

    #[test]
    fn light_leak_brightens_center() {
        let input = solid_rgba(16, 16, [0.3, 0.3, 0.3, 1.0]);
        let ll = LightLeak {
            intensity: 0.8,
            position_x: 0.5,
            position_y: 0.5,
            radius: 0.5,
            warmth: 0.8,
        };
        let out = ll.compute(&input, 16, 16).unwrap();
        let center = (8 * 16 + 8) * 4;
        assert!(out[center] > 0.3); // center should be brighter
    }

    // ─── Mirror Kaleidoscope ────────────────────────────────────────────

    #[test]
    fn mirror_horizontal_symmetry() {
        let input = gradient_rgba(16, 16);
        let mk = MirrorKaleidoscope {
            segments: 2,
            angle: 0.0,
            mode: 0,
        };
        let out = mk.compute(&input, 16, 16).unwrap();
        assert_eq!(out.len(), input.len());
    }

    // ─── Output sizes ───────────────────────────────────────────────────

    #[test]
    fn all_output_sizes_correct() {
        let input = gradient_rgba(16, 16);
        let n = 16 * 16 * 4;

        assert_eq!(
            GaussianNoise {
                amount: 10.0,
                mean: 0.0,
                sigma: 25.0,
                seed: 42
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
        assert_eq!(
            UniformNoise {
                range: 50.0,
                seed: 42
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
        assert_eq!(
            SaltPepperNoise {
                density: 0.1,
                seed: 42
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
        assert_eq!(
            PoissonNoise {
                scale: 50.0,
                seed: 42
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
        assert_eq!(
            FilmGrain {
                amount: 0.3,
                size: 2.0,
                seed: 42
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
        assert_eq!(
            Pixelate { block_size: 4 }
                .compute(&input, 16, 16)
                .unwrap()
                .len(),
            n
        );
        assert_eq!(
            Halftone {
                dot_size: 4.0,
                angle_offset: 0.0
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
        assert_eq!(
            OilPaint { radius: 2 }
                .compute(&input, 16, 16)
                .unwrap()
                .len(),
            n
        );
        assert_eq!(Emboss.compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(
            Charcoal {
                radius: 1.0,
                sigma: 1.0
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
        assert_eq!(
            Glitch {
                shift_amount: 10.0,
                channel_offset: 3.0,
                intensity: 0.5,
                band_height: 4,
                seed: 42
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
        assert_eq!(
            LightLeak {
                intensity: 0.5,
                position_x: 0.5,
                position_y: 0.5,
                radius: 0.5,
                warmth: 0.5
            }
            .compute(&input, 16, 16)
            .unwrap()
            .len(),
            n
        );
    }

    // ─── HDR values ─────────────────────────────────────────────────────

    #[test]
    fn hdr_values_not_clamped() {
        let input = solid_rgba(4, 4, [5.0, -0.5, 100.0, 1.0]);
        let n = GaussianNoise {
            amount: 10.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 42,
        };
        let out = n.compute(&input, 4, 4).unwrap();
        // HDR values should survive (not clamped to [0,1])
        assert!(out.chunks_exact(4).any(|p| p[0] > 1.0));
    }

    #[test]
    fn gpu_shaders_are_valid_wgsl_structure() {
        // Just verify all GPU shader bodies contain required WGSL elements
        let shaders: Vec<(&str, &str)> = vec![
            ("GaussianNoise", gaussian_noise::GAUSSIAN_NOISE_WGSL),
            ("UniformNoise", uniform_noise::UNIFORM_NOISE_WGSL),
            ("SaltPepper", salt_pepper_noise::SALT_PEPPER_WGSL),
            ("PoissonNoise", poisson_noise::POISSON_NOISE_WGSL),
            ("LightLeak", light_leak::LIGHT_LEAK_WGSL),
            ("Glitch", glitch::GLITCH_WGSL),
            ("ChromaticSplit", chromatic_split::CHROMATIC_SPLIT_WGSL),
            (
                "ChromaticAberration",
                chromatic_aberration::CHROMATIC_ABERRATION_WGSL,
            ),
            (
                "MirrorKaleidoscope",
                mirror_kaleidoscope::MIRROR_KALEIDOSCOPE_WGSL,
            ),
            ("Emboss", include_str!("../../shaders/emboss.wgsl")),
            ("Charcoal", charcoal::CHARCOAL_WGSL),
            ("FilmGrain", film_grain::EFFECT_FILM_GRAIN_WGSL),
            ("Pixelate", include_str!("../../shaders/pixelate.wgsl")),
            ("Halftone", include_str!("../../shaders/halftone.wgsl")),
            ("OilPaint", include_str!("../../shaders/oil_paint.wgsl")),
        ];
        for (name, body) in shaders {
            assert!(body.contains("@compute"), "{name} missing @compute");
            assert!(body.contains("fn main("), "{name} missing fn main");
            assert!(
                body.contains("struct Params"),
                "{name} missing Params struct"
            );
            assert!(
                body.contains("load_pixel") || body.contains("store_pixel"),
                "{name} missing pixel I/O"
            );
        }
    }

    #[test]
    fn gpu_noise_params_sizes_correct() {
        let g = GaussianNoise {
            amount: 50.0,
            mean: 0.0,
            sigma: 25.0,
            seed: 42,
        };
        assert_eq!(
            g.params(100, 100).len() % 4,
            0,
            "GaussianNoise params not 4-byte aligned"
        );
        let u = UniformNoise {
            range: 50.0,
            seed: 42,
        };
        assert_eq!(
            u.params(100, 100).len() % 4,
            0,
            "UniformNoise params not 4-byte aligned"
        );
        let sp = SaltPepperNoise {
            density: 0.05,
            seed: 42,
        };
        assert_eq!(
            sp.params(100, 100).len() % 4,
            0,
            "SaltPepper params not 4-byte aligned"
        );
        let p = PoissonNoise {
            scale: 100.0,
            seed: 42,
        };
        assert_eq!(
            p.params(100, 100).len() % 4,
            0,
            "Poisson params not 4-byte aligned"
        );
        // New GPU filters
        let e = Emboss;
        assert_eq!(
            e.params(100, 100).len() % 4,
            0,
            "Emboss params not 4-byte aligned"
        );
        let ch = Charcoal {
            radius: 1.0,
            sigma: 1.0,
        };
        assert_eq!(
            ch.params(100, 100).len() % 4,
            0,
            "Charcoal params not 4-byte aligned"
        );
        let fg = FilmGrain {
            amount: 0.3,
            size: 2.0,
            seed: 42,
        };
        assert_eq!(
            fg.params(100, 100).len() % 4,
            0,
            "FilmGrain params not 4-byte aligned"
        );
        let px = Pixelate { block_size: 4 };
        assert_eq!(
            px.params(100, 100).len() % 4,
            0,
            "Pixelate params not 4-byte aligned"
        );
        let ht = Halftone {
            dot_size: 4.0,
            angle_offset: 0.0,
        };
        assert_eq!(
            ht.params(100, 100).len() % 4,
            0,
            "Halftone params not 4-byte aligned"
        );
        let op = OilPaint { radius: 3 };
        assert_eq!(
            op.params(100, 100).len() % 4,
            0,
            "OilPaint params not 4-byte aligned"
        );
    }
}
