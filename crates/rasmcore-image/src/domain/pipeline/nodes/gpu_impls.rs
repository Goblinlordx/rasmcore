//! GpuCapable implementations for auto-generated filter pipeline nodes.
//!
//! Each filter node that has a corresponding WGSL shader gets a GpuCapable
//! impl here. Shaders are composed from shared fragments (pixel_ops,
//! sample_bilinear) plus filter-specific body code via rasmcore_gpu_shaders.

#[allow(unused_imports)]
use std::sync::LazyLock;

#[allow(unused_imports)]
use super::filters::*;
#[allow(unused_imports)]
use crate::domain::types::PixelFormat;
#[allow(unused_imports)]
use rasmcore_gpu_shaders as shaders;
#[allow(unused_imports)]
use rasmcore_pipeline::{GpuCapable, GpuOp};

#[allow(dead_code)]
fn is_rgba8(node_info: &crate::domain::types::ImageInfo) -> bool {
    node_info.format == PixelFormat::Rgba8
}

// ─── Composed shader sources (built once, cached) ────────────────────────────

static GAUSSIAN_BLUR: LazyLock<String> =
    LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/gaussian_blur.wgsl")));
static BOX_BLUR: LazyLock<String> =
    LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/box_blur.wgsl")));
static SHARPEN: LazyLock<String> =
    LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/sharpen.wgsl")));
static BILATERAL: LazyLock<String> =
    LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/bilateral.wgsl")));
static GUIDED_FILTER: LazyLock<String> =
    LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/guided_filter.wgsl")));
static MEDIAN: LazyLock<String> =
    LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/median.wgsl")));
static HIGH_PASS: LazyLock<String> =
    LazyLock::new(|| shaders::with_pixel_ops(include_str!("../../../shaders/high_pass.wgsl")));
static SPIN_BLUR: LazyLock<String> =
    LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/spin_blur.wgsl")));
static MOTION_BLUR: LazyLock<String> =
    LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/motion_blur.wgsl")));
static ZOOM_BLUR: LazyLock<String> =
    LazyLock::new(|| shaders::with_sampling(include_str!("../../../shaders/zoom_blur.wgsl")));
// ─── Spatial Filters ─────────────────────────────────────────────────────────

// BlurNode: GPU impl migrated to GpuFilter on BlurParams (derive(Filter) pattern)

// HighPassNode, SharpenNode, BilateralNode, GuidedFilterNode:
// GPU impls migrated to GpuFilter on their respective structs (derive(Filter) pattern)

// MedianNode, SpinBlurNode, MotionBlurNode, ZoomBlurNode:
// GPU impls migrated to GpuFilter on their respective structs (derive(Filter) pattern)

// ─── Distortion Filters ──────────────────────────────────────────────────────

// SpherizeNode, SwirlNode, BarrelNode, RippleNode, WaveNode, PolarNode, DepolarNode:
// GPU impls migrated to GpuFilter on their respective structs (derive(Filter) pattern)
