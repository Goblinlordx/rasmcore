//! Enhancement filters — image quality improvement operations on f32 pixel data.
//!
//! All operate on `&[f32]` RGBA (4 channels per pixel). No format dispatch.
//! No u8/u16 paths. Just f32.
//!
//! Includes: auto-level, CLAHE, clarity, dehaze, equalize, frequency separation,
//! NLM denoise, normalize, pyramid detail remap, retinex (SSR/MSR/MSRCR),
//! shadow-highlight, vignette (Gaussian + power-law).
//!
//! Dodge and Burn are in adjustment.rs (point ops with AnalyticOp support).

pub mod auto_level;
pub mod clahe;
pub mod clarity;
pub mod dehaze;
pub mod equalize;
pub mod frequency_high;
pub mod frequency_low;
pub mod nlm_denoise;
pub mod normalize;
pub mod pyramid_detail_remap;
pub mod retinex_msr;
pub mod retinex_msrcr;
pub mod retinex_ssr;
pub mod shadow_highlight;
pub mod vignette;
pub mod vignette_powerlaw;

pub use auto_level::AutoLevel;
pub use clahe::Clahe;
pub use clarity::Clarity;
pub use dehaze::Dehaze;
pub use equalize::Equalize;
pub use frequency_high::FrequencyHigh;
pub use frequency_low::FrequencyLow;
pub use nlm_denoise::NlmDenoise;
pub use normalize::Normalize;
pub use pyramid_detail_remap::PyramidDetailRemap;
pub use retinex_msr::RetinexMsr;
pub use retinex_msrcr::RetinexMsrcr;
pub use retinex_ssr::RetinexSsr;
pub use shadow_highlight::ShadowHighlight;
pub use vignette::Vignette;
pub use vignette_powerlaw::VignettePowerlaw;

// ─── Helpers ────────────────────────────────────────────────────────────────

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

#[cfg(test)]
mod tests;
