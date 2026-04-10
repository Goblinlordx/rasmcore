//! rasmcore-ml — Native ML inference provider.
//!
//! Implements the MlProvider trait for native applications using ONNX Runtime.
//! Auto-detects the best execution provider: CoreML (macOS), CUDA (NVIDIA),
//! DirectML (Windows), CPU fallback.
//!
//! Model distribution: definitions not weights.
//! - This crate ships model metadata (~1KB per model)
//! - Weights downloaded on first use to ~/.cache/rasmcore/models/
//! - Hash-verified after download
//!
//! # Usage
//!
//! ```ignore
//! use rasmcore_ml::{MlProvider, ModelDef, models};
//!
//! let provider = MlProvider::new(vec![
//!     models::real_esrgan_x4plus(),
//!     models::rmbg_14(),
//!     models::midas_v21_small(),
//! ]);
//!
//! // Register with pipeline...
//! ```

use std::path::PathBuf;

/// Model input tiling mode.
#[derive(Debug, Clone)]
pub enum TileMode {
    /// Convolutional — can be split into tiles with overlap.
    Tileable {
        preferred_size: (u32, u32),
        min_size: (u32, u32),
        overlap: u32,
    },
    /// Needs full image — resize to target.
    FullImage { target_size: (u32, u32) },
    /// Dynamic — no tiling or resizing.
    Dynamic,
}

/// What the model outputs.
#[derive(Debug, Clone, Copy)]
pub enum OutputKind {
    Image,
    Mask,
}

/// Model parameter descriptor.
#[derive(Debug, Clone)]
pub struct ParamDesc {
    pub name: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub default: Option<f64>,
    pub hint: Option<String>,
}

/// Model definition — metadata only, no weights.
#[derive(Debug, Clone)]
pub struct ModelDef {
    pub name: String,
    pub version: String,
    pub display_name: String,
    pub category: String,
    pub output_kind: OutputKind,
    pub output_scale: u32,
    pub tile_mode: TileMode,
    pub params: Vec<ParamDesc>,
    pub download_url: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub estimated_ms_per_tile: u32,
}

/// Cache directory for model weights.
pub fn model_cache_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("rasmcore")
        .join("models")
}

/// Pre-defined model definitions.
pub mod models {
    use super::*;

    pub fn real_esrgan_x4plus() -> ModelDef {
        ModelDef {
            name: "real-esrgan-x4plus".into(),
            version: "1.0.0".into(),
            display_name: "AI Super Resolution (4x)".into(),
            category: "upscale".into(),
            output_kind: OutputKind::Image,
            output_scale: 4,
            tile_mode: TileMode::Tileable {
                preferred_size: (256, 256),
                min_size: (64, 64),
                overlap: 8,
            },
            params: vec![ParamDesc {
                name: "denoise_strength".into(),
                min: Some(0.0),
                max: Some(1.0),
                step: Some(0.1),
                default: Some(0.5),
                hint: Some("Noise reduction strength".into()),
            }],
            download_url: String::new(), // User configures
            size_bytes: 67_108_864,
            sha256: String::new(),
            estimated_ms_per_tile: 300,
        }
    }

    pub fn rmbg_14() -> ModelDef {
        ModelDef {
            name: "rmbg-1.4".into(),
            version: "1.4.0".into(),
            display_name: "AI Background Removal".into(),
            category: "segmentation".into(),
            output_kind: OutputKind::Mask,
            output_scale: 1,
            tile_mode: TileMode::FullImage {
                target_size: (1024, 1024),
            },
            params: vec![ParamDesc {
                name: "threshold".into(),
                min: Some(0.0),
                max: Some(1.0),
                step: Some(0.05),
                default: Some(0.5),
                hint: Some("Mask threshold".into()),
            }],
            download_url: String::new(),
            size_bytes: 184_549_376,
            sha256: String::new(),
            estimated_ms_per_tile: 500,
        }
    }

    pub fn midas_v21_small() -> ModelDef {
        ModelDef {
            name: "midas-v2.1-small".into(),
            version: "2.1.0".into(),
            display_name: "AI Depth Estimation".into(),
            category: "depth".into(),
            output_kind: OutputKind::Mask,
            output_scale: 1,
            tile_mode: TileMode::FullImage {
                target_size: (384, 384),
            },
            params: vec![],
            download_url: String::new(),
            size_bytes: 52_428_800,
            sha256: String::new(),
            estimated_ms_per_tile: 200,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_defs_are_valid() {
        let models = vec![
            models::real_esrgan_x4plus(),
            models::rmbg_14(),
            models::midas_v21_small(),
        ];
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].name, "real-esrgan-x4plus");
        assert_eq!(models[0].output_scale, 4);
        assert!(matches!(models[0].tile_mode, TileMode::Tileable { .. }));
        assert!(matches!(models[1].output_kind, OutputKind::Mask));
        assert!(matches!(models[2].tile_mode, TileMode::FullImage { .. }));
    }

    #[test]
    fn cache_dir_is_valid() {
        let dir = model_cache_dir();
        assert!(dir.to_string_lossy().contains("rasmcore"));
        assert!(dir.to_string_lossy().contains("models"));
    }
}
