use super::*;
use crate::ops::{Filter, GpuFilter};

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

// ─── Auto Level ──────────────────────────────────────────────────────

#[test]
fn auto_level_expands_range() {
    // Input: all values in [0.2, 0.8]
    let mut input = solid_rgba(4, 4, [0.5, 0.5, 0.5, 1.0]);
    input[0] = 0.2; // min R
    input[4] = 0.8; // max R
    let output = AutoLevel.compute(&input, 4, 4).unwrap();
    // Min should map to ~0, max should map to ~1
    assert!(output[0] < 0.01);
    assert!(output[4] > 0.99);
}

#[test]
fn auto_level_preserves_alpha() {
    let input = solid_rgba(4, 4, [0.3, 0.5, 0.7, 0.5]);
    let output = AutoLevel.compute(&input, 4, 4).unwrap();
    assert_eq!(output[3], 0.5);
}

#[test]
fn auto_level_gpu_returns_3_passes() {
    let f = AutoLevel;
    let shaders = f.gpu_shaders(100, 100);
    assert_eq!(shaders.len(), 3, "AutoLevel should use 3-pass ChannelMinMax reduction");
    // Pass 1+2 have read_write reduction buffers, pass 3 has read-only
    assert!(shaders[0].reduction_buffers[0].read_write);
    assert!(!shaders[2].reduction_buffers[0].read_write);
    // All have same buffer ID
    assert_eq!(
        shaders[0].reduction_buffers[0].id,
        shaders[2].reduction_buffers[0].id,
    );
}

#[test]
fn equalize_gpu_returns_3_passes() {
    let f = Equalize;
    let shaders = f.gpu_shaders(100, 100);
    assert_eq!(shaders.len(), 3, "Equalize should use 3-pass Histogram256 reduction");
    assert!(shaders[0].reduction_buffers[0].read_write);
    assert!(!shaders[2].reduction_buffers[0].read_write);
}

// ─── Equalize ────────────��───────────────────────────────────────────

#[test]
fn equalize_spreads_histogram() {
    let input = gradient_rgba(16, 16);
    let output = Equalize.compute(&input, 16, 16).unwrap();
    // Output should use full range
    let r_vals: Vec<f32> = output.chunks_exact(4).map(|p| p[0]).collect();
    assert!(*r_vals.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() < 0.1);
    assert!(*r_vals.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() > 0.9);
}

#[test]
fn normalize_gpu_returns_3_passes() {
    let f = Normalize::default();
    let shaders = f.gpu_shaders(100, 100);
    assert_eq!(shaders.len(), 3, "Normalize should use 3-pass Histogram256 reduction");
    assert!(shaders[0].reduction_buffers[0].read_write);
    assert!(!shaders[2].reduction_buffers[0].read_write);
    // Pass 3 params should encode clip counts
    let params = &shaders[2].params;
    let black_clip_count = u32::from_le_bytes(params[12..16].try_into().unwrap());
    let white_clip_count = u32::from_le_bytes(params[16..20].try_into().unwrap());
    // Default: 2% black, 1% white of 10000 pixels
    assert_eq!(black_clip_count, 200); // 10000 * 0.02
    assert_eq!(white_clip_count, 100); // 10000 * 0.01
}

// ─── Normalize ──────────��────────────────────────��───────────────────

#[test]
fn normalize_clips_extremes() {
    let input = gradient_rgba(16, 16);
    let norm = Normalize::default();
    let output = norm.compute(&input, 16, 16).unwrap();
    assert_eq!(output.len(), input.len());
}

// ─── Frequency Separation ───────────────────────────────────────────��

#[test]
fn frequency_low_is_blurred() {
    let input = gradient_rgba(16, 16);
    let fl = FrequencyLow { sigma: 3.0 };
    let output = fl.compute(&input, 16, 16).unwrap();
    assert_eq!(output.len(), input.len());
}

#[test]
fn frequency_high_solid_midgray() {
    let input = solid_rgba(16, 16, [0.3, 0.3, 0.3, 1.0]);
    let fh = FrequencyHigh { sigma: 3.0 };
    let output = fh.compute(&input, 16, 16).unwrap();
    // Solid -> high pass = 0 + 0.5 = 0.5
    assert!((output[0] - 0.5).abs() < 0.01);
}

// ─── Clarity ─��───────────────────────────────────────────────────────

#[test]
fn clarity_preserves_solid() {
    let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
    let clar = Clarity { amount: 1.0, radius: 5.0 };
    let output = clar.compute(&input, 16, 16).unwrap();
    // Solid: no detail -> no change
    assert!((output[0] - 0.5).abs() < 0.01);
}

#[test]
fn clarity_preserves_alpha() {
    let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
    let clar = Clarity { amount: 1.0, radius: 5.0 };
    let output = clar.compute(&input, 8, 8).unwrap();
    assert!((output[3] - 0.7).abs() < 1e-6);
}

// ─── Dehaze ───��──────────────────────────────���───────────────────────

#[test]
fn dehaze_runs_without_panic() {
    let input = gradient_rgba(16, 16);
    let dh = Dehaze { patch_radius: 3, omega: 0.95, t_min: 0.1 };
    let output = dh.compute(&input, 16, 16).unwrap();
    assert_eq!(output.len(), input.len());
}

// ─── CLAHE ────────���──────────────────────────────────────────────────

#[test]
fn clahe_runs_without_panic() {
    let input = gradient_rgba(32, 32);
    let clahe = Clahe { tile_grid: 4, clip_limit: 2.0 };
    let output = clahe.compute(&input, 32, 32).unwrap();
    assert_eq!(output.len(), input.len());
}

#[test]
fn clahe_preserves_alpha() {
    let input = solid_rgba(32, 32, [0.5, 0.5, 0.5, 0.3]);
    let clahe = Clahe { tile_grid: 4, clip_limit: 2.0 };
    let output = clahe.compute(&input, 32, 32).unwrap();
    assert!((output[3] - 0.3).abs() < 1e-6);
}

// ─── NLM Denoise ────��────────────────────────────────────────────────

#[test]
fn nlm_solid_unchanged() {
    let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
    let nlm = NlmDenoise { h: 0.1, patch_radius: 1, search_radius: 2 };
    let output = nlm.compute(&input, 8, 8).unwrap();
    assert!((output[0] - 0.5).abs() < 0.01);
}

#[test]
fn nlm_preserves_alpha() {
    let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
    let nlm = NlmDenoise { h: 0.1, patch_radius: 1, search_radius: 2 };
    let output = nlm.compute(&input, 8, 8).unwrap();
    assert!((output[3] - 0.7).abs() < 1e-6);
}

// ─── Pyramid Detail Remap ───────���────────────────────────────────���───

#[test]
fn pyramid_detail_remap_runs() {
    let input = gradient_rgba(32, 32);
    let pdr = PyramidDetailRemap { sigma: 0.5, levels: 0 };
    let output = pdr.compute(&input, 32, 32).unwrap();
    assert_eq!(output.len(), input.len());
}

// ─── Retinex ────���──────────────────────────────���─────────────────────

#[test]
fn retinex_ssr_runs() {
    let input = gradient_rgba(16, 16);
    let ssr = RetinexSsr { sigma: 15.0 };
    let output = ssr.compute(&input, 16, 16).unwrap();
    assert_eq!(output.len(), input.len());
}

#[test]
fn retinex_msr_runs() {
    let input = gradient_rgba(16, 16);
    let msr = RetinexMsr { sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0 };
    let output = msr.compute(&input, 16, 16).unwrap();
    assert_eq!(output.len(), input.len());
}

#[test]
fn retinex_msrcr_runs() {
    let input = gradient_rgba(16, 16);
    let msrcr = RetinexMsrcr {
        sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0,
        alpha: 125.0, beta: 46.0,
    };
    let output = msrcr.compute(&input, 16, 16).unwrap();
    assert_eq!(output.len(), input.len());
}

#[test]
fn retinex_ssr_preserves_alpha() {
    let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
    let ssr = RetinexSsr { sigma: 5.0 };
    let output = ssr.compute(&input, 8, 8).unwrap();
    assert!((output[3] - 0.7).abs() < 1e-6);
}

// ─── Shadow/Highlight ────��───────────────────────────────────────────

#[test]
fn shadow_highlight_neutral_noop() {
    let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
    let sh = ShadowHighlight {
        shadows: 0.0, highlights: 0.0, whitepoint: 0.0,
        radius: 10.0, compress: 50.0,
        shadows_ccorrect: 100.0, highlights_ccorrect: 50.0,
    };
    let output = sh.compute(&input, 16, 16).unwrap();
    assert!((output[0] - 0.5).abs() < 0.02);
}

// ─── Vignette ─────────────────────────────────────────���──────────────

#[test]
fn vignette_center_bright_edges_dark() {
    let input = solid_rgba(32, 32, [1.0, 1.0, 1.0, 1.0]);
    let vig = Vignette { sigma: 5.0, x_inset: 4, y_inset: 4 };
    let output = vig.compute(&input, 32, 32).unwrap();
    // Center pixel should be brighter than corner
    let center = (16 * 32 + 16) * 4;
    let corner = 0;
    assert!(output[center] > output[corner]);
}

#[test]
fn vignette_preserves_alpha() {
    let input = solid_rgba(16, 16, [1.0, 1.0, 1.0, 0.5]);
    let vig = Vignette { sigma: 3.0, x_inset: 2, y_inset: 2 };
    let output = vig.compute(&input, 16, 16).unwrap();
    assert!((output[3] - 0.5).abs() < 1e-6);
}

// ─── Vignette Power-law ────────────────────────────────────��─────────

#[test]
fn vignette_powerlaw_center_unaffected() {
    let input = solid_rgba(16, 16, [1.0, 1.0, 1.0, 1.0]);
    let vig = VignettePowerlaw { strength: 0.5, falloff: 2.0 };
    let output = vig.compute(&input, 16, 16).unwrap();
    // Center pixel should be minimally affected (close to center)
    let center = (8 * 16 + 8) * 4;
    assert!(output[center] > 0.95);
}

#[test]
fn vignette_powerlaw_corners_darkened() {
    let input = solid_rgba(16, 16, [1.0, 1.0, 1.0, 1.0]);
    let vig = VignettePowerlaw { strength: 1.0, falloff: 2.0 };
    let output = vig.compute(&input, 16, 16).unwrap();
    // Corner should be darker than center
    let center = (8 * 16 + 8) * 4;
    let corner = 0;
    assert!(output[corner] < output[center]);
}

// ─── Output sizes ────────��──────────────────────────────────��────────

#[test]
fn all_output_sizes_correct() {
    let input = gradient_rgba(16, 16);
    let n = 16 * 16 * 4;

    assert_eq!(AutoLevel.compute(&input, 16, 16).unwrap().len(), n);
    assert_eq!(Equalize.compute(&input, 16, 16).unwrap().len(), n);
    assert_eq!(Normalize::default().compute(&input, 16, 16).unwrap().len(), n);
    assert_eq!(FrequencyLow { sigma: 3.0 }.compute(&input, 16, 16).unwrap().len(), n);
    assert_eq!(FrequencyHigh { sigma: 3.0 }.compute(&input, 16, 16).unwrap().len(), n);
    assert_eq!(
        Clarity { amount: 1.0, radius: 3.0 }.compute(&input, 16, 16).unwrap().len(), n
    );
    assert_eq!(
        RetinexSsr { sigma: 5.0 }.compute(&input, 16, 16).unwrap().len(), n
    );
    assert_eq!(
        VignettePowerlaw { strength: 0.5, falloff: 2.0 }.compute(&input, 16, 16).unwrap().len(), n
    );
}

// ─── HDR values ───────��───────────────────────────────���──────────────

#[test]
fn hdr_values_not_clamped() {
    let input = solid_rgba(4, 4, [5.0, -0.5, 100.0, 1.0]);
    let vig = VignettePowerlaw { strength: 0.5, falloff: 2.0 };
    let output = vig.compute(&input, 4, 4).unwrap();
    // HDR values should still be present (not clamped to [0,1])
    assert!(output.chunks_exact(4).any(|p| p[0] > 1.0));
}

// ── GPU wiring tests ────���────────────────────────────────────────────────

#[test]
fn nlm_denoise_gpu_single_pass() {
    let nlm = NlmDenoise { h: 0.1, patch_radius: 3, search_radius: 7 };
    let passes = nlm.gpu_shaders(64, 64);
    assert_eq!(passes.len(), 1);
    assert_eq!(nlm.workgroup_size(), [16, 16, 1]);
}

#[test]
fn dehaze_gpu_2_passes() {
    let dh = Dehaze { patch_radius: 7, omega: 0.95, t_min: 0.1 };
    let passes = dh.gpu_shaders(64, 64);
    assert_eq!(passes.len(), 2, "Dehaze: dark channel + apply");
}

#[test]
fn shadow_highlight_gpu_3_passes() {
    let sh = ShadowHighlight {
        shadows: 50.0, highlights: 50.0, whitepoint: 0.0,
        radius: 3.0, compress: 50.0,
        shadows_ccorrect: 50.0, highlights_ccorrect: 50.0,
    };
    let passes = sh.gpu_shaders(64, 64);
    assert_eq!(passes.len(), 3, "ShadowHighlight: blur H + blur V + apply");
}

#[test]
fn frequency_low_gpu_2_passes() {
    let fl = FrequencyLow { sigma: 3.0 };
    let passes = fl.gpu_shaders(64, 64);
    assert_eq!(passes.len(), 2);
}

#[test]
fn frequency_high_gpu_3_passes() {
    let fh = FrequencyHigh { sigma: 3.0 };
    let passes = fh.gpu_shaders(64, 64);
    assert_eq!(passes.len(), 3);
}

#[test]
fn clarity_gpu_3_passes() {
    let cl = Clarity { amount: 0.5, radius: 20.0 };
    let passes = cl.gpu_shaders(64, 64);
    assert_eq!(passes.len(), 3);
}

#[test]
fn retinex_ssr_gpu_3_passes() {
    let ssr = RetinexSsr { sigma: 80.0 };
    let passes = ssr.gpu_shaders(64, 64);
    assert_eq!(passes.len(), 3, "RetinexSSR: blur H + blur V + retinex apply");
}

#[test]
fn retinex_msr_gpu_has_multiple_passes() {
    let msr = RetinexMsr { sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0 };
    let passes = msr.gpu_shaders(64, 64);
    // 3 scales x (blur H + blur V + accumulate) = 9 + 2 reduce + 1 normalize = 12
    assert!(passes.len() >= 10, "RetinexMSR should have many passes, got {}", passes.len());
}

#[test]
fn retinex_msrcr_gpu_has_multiple_passes() {
    let msrcr = RetinexMsrcr {
        sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0,
        alpha: 125.0, beta: 46.0,
    };
    let passes = msrcr.gpu_shaders(64, 64);
    // 3 scales x 3 + 1 color restore = 10
    assert!(passes.len() >= 10, "RetinexMSRCR should have many passes, got {}", passes.len());
}

#[test]
fn clahe_gpu_single_pass_with_luts() {
    let clahe = Clahe { tile_grid: 8, clip_limit: 2.0 };
    let passes = clahe.gpu_shaders(64, 64);
    assert_eq!(passes.len(), 1);
    let params = clahe.params(64, 64);
    assert_eq!(params.len(), 32);
}

#[test]
fn pyramid_detail_remap_gpu_multi_pass() {
    let pdr = PyramidDetailRemap { sigma: 0.5, levels: 4 };
    let passes = pdr.gpu_shaders(64, 64);
    // 4 downsample + 4 x (upsample + remap) = 4 + 8 = 12
    assert!(passes.len() >= 8, "Pyramid: should have downsample + remap passes, got {}", passes.len());
}

#[test]
fn all_enhancement_filters_have_gpu() {
    let w = 32u32;
    let h = 32u32;

    // Already wired in previous tracks
    assert!(!AutoLevel.gpu_shaders(w, h).is_empty());
    assert!(!Equalize.gpu_shaders(w, h).is_empty());
    assert!(!Normalize::default().gpu_shaders(w, h).is_empty());
    assert!(!VignettePowerlaw { strength: 0.5, falloff: 2.0 }.gpu_shaders(w, h).is_empty());

    // Newly wired in this track
    assert!(!NlmDenoise { h: 0.1, patch_radius: 3, search_radius: 7 }.gpu_shaders(w, h).is_empty());
    assert!(!Dehaze { patch_radius: 7, omega: 0.95, t_min: 0.1 }.gpu_shaders(w, h).is_empty());
    assert!(!ShadowHighlight {
        shadows: 50.0, highlights: 50.0, whitepoint: 0.0,
        radius: 3.0, compress: 50.0, shadows_ccorrect: 50.0, highlights_ccorrect: 50.0,
    }.gpu_shaders(w, h).is_empty());
    assert!(!FrequencyLow { sigma: 3.0 }.gpu_shaders(w, h).is_empty());
    assert!(!FrequencyHigh { sigma: 3.0 }.gpu_shaders(w, h).is_empty());
    assert!(!Clarity { amount: 0.5, radius: 20.0 }.gpu_shaders(w, h).is_empty());
    assert!(!RetinexSsr { sigma: 80.0 }.gpu_shaders(w, h).is_empty());
    assert!(!RetinexMsr { sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0 }.gpu_shaders(w, h).is_empty());
    assert!(!RetinexMsrcr {
        sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0,
        alpha: 125.0, beta: 46.0,
    }.gpu_shaders(w, h).is_empty());
    assert!(!Clahe { tile_grid: 8, clip_limit: 2.0 }.gpu_shaders(w, h).is_empty());
    assert!(!PyramidDetailRemap { sigma: 0.5, levels: 4 }.gpu_shaders(w, h).is_empty());
}
