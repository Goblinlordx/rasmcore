//! Filter trait and registry — extensible plugin architecture.
//!
//! Defines a common `ImageFilter` trait that all filters implement, plus a
//! `FilterRegistry` that collects built-in and custom filters for discovery.
//!
//! # Adding a custom filter (external crate)
//!
//! ```ignore
//! use rasmcore_image::domain::filter_registry::*;
//! use rasmcore_image::domain::types::*;
//! use rasmcore_image::domain::error::ImageError;
//!
//! struct MyCustomBlur;
//!
//! impl ImageFilter for MyCustomBlur {
//!     fn name(&self) -> &str { "my_custom_blur" }
//!     fn category(&self) -> FilterCategory { FilterCategory::Blur }
//!     fn param_descriptors(&self) -> Vec<ParamDescriptor> {
//!         vec![ParamDescriptor::float("sigma", 0.1, 100.0, 1.0)]
//!     }
//!     fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError> {
//!         let sigma = input.params.get_float("sigma").unwrap_or(1.0);
//!         // ... your implementation
//!         Ok(input.pixels.to_vec())
//!     }
//! }
//!
//! // Register at startup:
//! registry.register(Box::new(MyCustomBlur));
//! ```

use super::error::ImageError;
use super::types::ImageInfo;
use std::collections::HashMap;

// ─── Filter Parameters ────────────────────────────────────────────────────

/// A single parameter value for a filter invocation.
#[derive(Debug, Clone)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    UInt(u32),
    Bool(bool),
    Color([u8; 3]),
}

/// Named parameter map for filter invocation.
#[derive(Debug, Clone, Default)]
pub struct FilterParams {
    values: HashMap<String, ParamValue>,
}

impl FilterParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_float(mut self, name: &str, val: f32) -> Self {
        self.values.insert(name.to_string(), ParamValue::Float(val));
        self
    }

    pub fn set_int(mut self, name: &str, val: i32) -> Self {
        self.values.insert(name.to_string(), ParamValue::Int(val));
        self
    }

    pub fn set_uint(mut self, name: &str, val: u32) -> Self {
        self.values.insert(name.to_string(), ParamValue::UInt(val));
        self
    }

    pub fn set_bool(mut self, name: &str, val: bool) -> Self {
        self.values.insert(name.to_string(), ParamValue::Bool(val));
        self
    }

    pub fn set_color(mut self, name: &str, rgb: [u8; 3]) -> Self {
        self.values.insert(name.to_string(), ParamValue::Color(rgb));
        self
    }

    pub fn get_float(&self, name: &str) -> Option<f32> {
        match self.values.get(name) {
            Some(ParamValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_int(&self, name: &str) -> Option<i32> {
        match self.values.get(name) {
            Some(ParamValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_uint(&self, name: &str) -> Option<u32> {
        match self.values.get(name) {
            Some(ParamValue::UInt(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.values.get(name) {
            Some(ParamValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_color(&self, name: &str) -> Option<[u8; 3]> {
        match self.values.get(name) {
            Some(ParamValue::Color(v)) => Some(*v),
            _ => None,
        }
    }
}

/// Describes a parameter that a filter accepts.
#[derive(Debug, Clone)]
pub struct ParamDescriptor {
    pub name: String,
    pub param_type: ParamType,
    pub default: ParamValue,
    pub description: String,
}

/// Parameter type with range constraints.
#[derive(Debug, Clone)]
pub enum ParamType {
    Float { min: f32, max: f32 },
    Int { min: i32, max: i32 },
    UInt { min: u32, max: u32 },
    Bool,
    Color,
}

impl ParamDescriptor {
    pub fn float(name: &str, min: f32, max: f32, default: f32) -> Self {
        Self {
            name: name.to_string(),
            param_type: ParamType::Float { min, max },
            default: ParamValue::Float(default),
            description: String::new(),
        }
    }

    pub fn uint(name: &str, min: u32, max: u32, default: u32) -> Self {
        Self {
            name: name.to_string(),
            param_type: ParamType::UInt { min, max },
            default: ParamValue::UInt(default),
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }
}

// ─── Filter Input ─────────────────────────────────────────────────────────

/// Input to a filter: pixel data + metadata + parameters.
pub struct FilterInput<'a> {
    pub pixels: &'a [u8],
    pub info: &'a ImageInfo,
    pub params: &'a FilterParams,
}

// ─── Filter Trait ─────────────────────────────────────────────────────────

/// Category of image filter operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterCategory {
    Blur,
    Sharpen,
    EdgeDetection,
    Denoise,
    Color,
    Contrast,
    PointOp,
    Morphology,
    Transform,
    Composite,
    Enhancement,
    Other,
}

/// The core filter trait. Implement this for any image filter.
///
/// All built-in filters implement this trait, and external crates can
/// implement it to add custom filters that integrate with the pipeline.
pub trait ImageFilter: Send + Sync {
    /// Unique name for this filter (e.g., "blur", "clahe", "my_custom_filter").
    fn name(&self) -> &str;

    /// Category for grouping/discovery.
    fn category(&self) -> FilterCategory;

    /// Describe the parameters this filter accepts.
    fn param_descriptors(&self) -> Vec<ParamDescriptor>;

    /// Apply the filter to the input pixels and return the result.
    fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError>;

    /// Whether this filter is a no-op for the given input and params.
    ///
    /// Used by the pipeline optimizer to skip filters entirely. Examples:
    /// - brightness(0.0) — no change
    /// - gamma(1.0) — identity
    /// - bit depth conversion when already at target depth
    /// - saturate(1.0) — identity
    ///
    /// Default: false (always runs).
    fn is_noop(&self, _input: &FilterInput) -> bool {
        false
    }

    /// Whether this filter can be collapsed into a 256-entry LUT.
    ///
    /// LUT-collapsible filters are per-pixel point operations where the
    /// output depends ONLY on the input value (not neighboring pixels or
    /// position). When true, consecutive LUT-collapsible filters are fused
    /// into a single pass at pipeline optimization time.
    ///
    /// Default: false (conservative — spatial filters, convolutions, etc.)
    fn is_lut_collapsible(&self) -> bool {
        false
    }

    /// Build the 256-entry LUT for 8-bit input (only called if `is_lut_collapsible()` is true).
    ///
    /// Returns `lut[i]` = output value for input value `i`. The pipeline
    /// uses `compose_luts()` to fuse consecutive LUT filters into a single
    /// lookup table, then applies it in one pass via `apply_lut()`.
    ///
    /// Default: identity LUT (no-op).
    fn build_lut(&self, _params: &FilterParams) -> [u8; 256] {
        let mut lut = [0u8; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
        lut
    }

    /// Build the 65536-entry LUT for 16-bit input.
    /// Only called if `is_lut_collapsible()` is true AND input is 16-bit.
    /// Default: delegates to 8-bit LUT with linear scaling.
    fn build_lut_u16(&self, params: &FilterParams) -> Vec<u16> {
        // Default: upscale from 8-bit LUT
        let lut8 = self.build_lut(params);
        (0..65536u32)
            .map(|i| {
                let i8 = (i >> 8) as usize;
                (lut8[i8.min(255)] as u16) << 8 | (lut8[i8.min(255)] as u16)
            })
            .collect()
    }
}

// ─── JSON Param Descriptor (for ConfigParams derive + manifest) ────────────

/// Parameter descriptor for JSON manifest generation.
/// Generated by `#[derive(ConfigParams)]` from struct fields + `#[param]` attributes.
#[derive(Debug, Clone)]
pub struct ParamDescriptorJson {
    pub name: String,
    pub param_type: String,
    pub min: String,
    pub max: String,
    pub step: String,
    pub default_val: String,
    pub label: String,
    pub hint: String,
}

// ─── Static Registration (for proc macro + inventory) ─────────────────────

/// Static filter registration metadata — generated by `#[register_filter]` proc macro.
///
/// Collected across crate boundaries via `inventory`. The adapter layer iterates
/// all registrations to build dispatch tables automatically.
#[derive(Debug)]
pub struct StaticFilterRegistration {
    /// Filter name as exposed in WIT/SDK (e.g., "blur")
    pub name: &'static str,
    /// Category for grouping (e.g., "spatial", "color")
    pub category: &'static str,
    /// UI group — related variants share a group (e.g., "blur" for blur/bokeh_blur/motion_blur)
    pub group: &'static str,
    /// Variant name within the group (e.g., "bokeh", "motion"). Empty for default/standalone.
    pub variant: &'static str,
    /// Provenance — algorithm, paper, or physical model (e.g., "Reinhard 2002")
    pub reference: &'static str,
    /// Number of parameters (beyond pixels + info)
    pub param_count: usize,
    /// Function name in source code
    pub fn_name: &'static str,
    /// Module path for dispatch
    pub module_path: &'static str,
}

inventory::collect!(&'static StaticFilterRegistration);

/// Iterate all registered filters (from all crates linked into the binary).
pub fn registered_filters() -> Vec<&'static StaticFilterRegistration> {
    inventory::iter::<&'static StaticFilterRegistration>
        .into_iter()
        .copied()
        .collect()
}

// ─── Generator Registration ───────────────────────────────────────────────

/// Static generator registration — procedural image sources (no pixel input).
#[derive(Debug)]
pub struct StaticGeneratorRegistration {
    pub name: &'static str,
    pub category: &'static str,
    pub group: &'static str,
    pub variant: &'static str,
    pub reference: &'static str,
    pub param_count: usize,
    pub fn_name: &'static str,
    pub module_path: &'static str,
}

inventory::collect!(&'static StaticGeneratorRegistration);

pub fn registered_generators() -> Vec<&'static StaticGeneratorRegistration> {
    inventory::iter::<&'static StaticGeneratorRegistration>
        .into_iter()
        .copied()
        .collect()
}

// ─── Compositor Registration ──────────────────────────────────────────────

/// Static compositor registration — multi-input blending/composition.
#[derive(Debug)]
pub struct StaticCompositorRegistration {
    pub name: &'static str,
    pub category: &'static str,
    pub group: &'static str,
    pub variant: &'static str,
    pub reference: &'static str,
    pub param_count: usize,
    pub fn_name: &'static str,
    pub module_path: &'static str,
}

inventory::collect!(&'static StaticCompositorRegistration);

pub fn registered_compositors() -> Vec<&'static StaticCompositorRegistration> {
    inventory::iter::<&'static StaticCompositorRegistration>
        .into_iter()
        .copied()
        .collect()
}

// ─── Mapper Registration ──────────────────────────────────────────────────

/// Static mapper registration — format-changing operations.
#[derive(Debug)]
pub struct StaticMapperRegistration {
    pub name: &'static str,
    pub category: &'static str,
    pub group: &'static str,
    pub variant: &'static str,
    pub reference: &'static str,
    pub param_count: usize,
    pub fn_name: &'static str,
    pub module_path: &'static str,
}

inventory::collect!(&'static StaticMapperRegistration);

pub fn registered_mappers() -> Vec<&'static StaticMapperRegistration> {
    inventory::iter::<&'static StaticMapperRegistration>
        .into_iter()
        .copied()
        .collect()
}

// ─── Filter Registry ──────────────────────────────────────────────────────

/// Registry of all available image filters.
///
/// Built-in filters are registered at creation via `with_builtins()`.
/// Custom filters can be added at any time via `register()`.
pub struct FilterRegistry {
    filters: Vec<Box<dyn ImageFilter>>,
    by_name: HashMap<String, usize>,
}

impl FilterRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            by_name: HashMap::new(),
        }
    }

    /// Create a registry pre-loaded with all built-in filters.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        register_builtin_filters(&mut reg);
        reg
    }

    /// Register a custom filter.
    pub fn register(&mut self, filter: Box<dyn ImageFilter>) {
        let name = filter.name().to_string();
        let idx = self.filters.len();
        self.filters.push(filter);
        self.by_name.insert(name, idx);
    }

    /// Look up a filter by name.
    pub fn get(&self, name: &str) -> Option<&dyn ImageFilter> {
        self.by_name
            .get(name)
            .map(|&idx| self.filters[idx].as_ref())
    }

    /// List all registered filter names.
    pub fn names(&self) -> Vec<&str> {
        self.filters.iter().map(|f| f.name()).collect()
    }

    /// List all filters in a category.
    pub fn by_category(&self, category: FilterCategory) -> Vec<&dyn ImageFilter> {
        self.filters
            .iter()
            .filter(|f| f.category() == category)
            .map(|f| f.as_ref())
            .collect()
    }

    /// Total number of registered filters.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl Default for FilterRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ─── Built-in Filter Implementations ──────────────────────────────────────

macro_rules! builtin_filter {
    ($struct_name:ident, $filter_name:expr, $category:expr,
     params: [$($param:expr),*],
     apply: |$input:ident| $body:expr
    ) => {
        pub struct $struct_name;

        impl ImageFilter for $struct_name {
            fn name(&self) -> &str { $filter_name }
            fn category(&self) -> FilterCategory { $category }
            fn param_descriptors(&self) -> Vec<ParamDescriptor> {
                vec![$($param),*]
            }
            fn apply(&self, $input: &FilterInput) -> Result<Vec<u8>, ImageError> {
                $body
            }
        }
    };
}

use super::filters;
use super::point_ops::{self, PointOp};

// Blur filters
builtin_filter!(BlurFilter, "blur", FilterCategory::Blur,
    params: [ParamDescriptor::float("radius", 0.0, 100.0, 1.0).with_description("Gaussian blur radius")],
    apply: |input| {
        let radius = input.params.get_float("radius").unwrap_or(1.0);
        filters::blur_impl(input.pixels, input.info, &filters::BlurParams { radius })
    }
);

builtin_filter!(SharpenFilter, "sharpen", FilterCategory::Sharpen,
    params: [ParamDescriptor::float("amount", 0.0, 10.0, 1.0).with_description("Unsharp mask amount")],
    apply: |input| {
        let amount = input.params.get_float("amount").unwrap_or(1.0);
        filters::sharpen_impl(input.pixels, input.info, &filters::SharpenParams { amount })
    }
);

builtin_filter!(MedianFilter, "median", FilterCategory::Denoise,
    params: [ParamDescriptor::uint("radius", 1, 50, 3).with_description("Median filter radius")],
    apply: |input| {
        let radius = input.params.get_uint("radius").unwrap_or(3);
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::median(r, &mut u, input.info, &filters::MedianParams { radius })
    }
);

// Edge detection
builtin_filter!(SobelFilter, "sobel", FilterCategory::EdgeDetection,
    params: [],
    apply: |input| filters::sobel(input.pixels, input.info)
);

builtin_filter!(CannyFilter, "canny", FilterCategory::EdgeDetection,
    params: [
        ParamDescriptor::float("low_threshold", 0.0, 255.0, 50.0),
        ParamDescriptor::float("high_threshold", 0.0, 255.0, 150.0)
    ],
    apply: |input| {
        let low = input.params.get_float("low_threshold").unwrap_or(50.0);
        let high = input.params.get_float("high_threshold").unwrap_or(150.0);
        filters::canny(input.pixels, input.info, &filters::CannyParams { low_threshold: low, high_threshold: high })
    }
);

// Point operations
// Point operations — all are LUT-collapsible (per-pixel, no neighbors)
pub struct GammaFilter;
impl ImageFilter for GammaFilter {
    fn name(&self) -> &str {
        "gamma"
    }
    fn category(&self) -> FilterCategory {
        FilterCategory::PointOp
    }
    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        vec![ParamDescriptor::float("gamma", 0.1, 10.0, 1.0)]
    }
    fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError> {
        let gamma = input.params.get_float("gamma").unwrap_or(1.0);
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::gamma_registered(r, &mut u, input.info, &filters::GammaParams { gamma_value: gamma })
    }
    fn is_lut_collapsible(&self) -> bool {
        true
    }
    fn build_lut(&self, params: &FilterParams) -> [u8; 256] {
        let gamma = params.get_float("gamma").unwrap_or(1.0);
        point_ops::build_lut(&PointOp::Gamma(gamma))
    }
}

pub struct BrightnessFilter;
impl ImageFilter for BrightnessFilter {
    fn name(&self) -> &str {
        "brightness"
    }
    fn category(&self) -> FilterCategory {
        FilterCategory::PointOp
    }
    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        vec![ParamDescriptor::float("amount", -1.0, 1.0, 0.0)]
    }
    fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError> {
        let amount = input.params.get_float("amount").unwrap_or(0.0);
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::brightness(r, &mut u, input.info, &filters::BrightnessParams { amount })
    }
    fn is_lut_collapsible(&self) -> bool {
        true
    }
    fn build_lut(&self, params: &FilterParams) -> [u8; 256] {
        let amount = params.get_float("amount").unwrap_or(0.0);
        point_ops::build_lut(&PointOp::Brightness(amount))
    }
}

pub struct ContrastFilter;
impl ImageFilter for ContrastFilter {
    fn name(&self) -> &str {
        "contrast"
    }
    fn category(&self) -> FilterCategory {
        FilterCategory::PointOp
    }
    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        vec![ParamDescriptor::float("amount", -1.0, 1.0, 0.0)]
    }
    fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError> {
        let amount = input.params.get_float("amount").unwrap_or(0.0);
        {
            let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
            let p = input.pixels;
            let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
            filters::contrast(r, &mut u, input.info, &filters::ContrastParams { amount })
        }
    }
    fn is_lut_collapsible(&self) -> bool {
        true
    }
    fn build_lut(&self, params: &FilterParams) -> [u8; 256] {
        let amount = params.get_float("amount").unwrap_or(0.0);
        point_ops::build_lut(&PointOp::Contrast(amount))
    }
}

pub struct InvertFilter;
impl ImageFilter for InvertFilter {
    fn name(&self) -> &str {
        "invert"
    }
    fn category(&self) -> FilterCategory {
        FilterCategory::PointOp
    }
    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        vec![]
    }
    fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError> {
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::invert_registered(r, &mut u, input.info)
    }
    fn is_lut_collapsible(&self) -> bool {
        true
    }
    fn build_lut(&self, _params: &FilterParams) -> [u8; 256] {
        point_ops::build_lut(&PointOp::Invert)
    }
}

pub struct ThresholdFilter;
impl ImageFilter for ThresholdFilter {
    fn name(&self) -> &str {
        "threshold"
    }
    fn category(&self) -> FilterCategory {
        FilterCategory::PointOp
    }
    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        vec![ParamDescriptor::uint("level", 0, 255, 128)]
    }
    fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError> {
        let level = input.params.get_uint("level").unwrap_or(128) as u8;
        point_ops::threshold(input.pixels, input.info, level)
    }
    fn is_lut_collapsible(&self) -> bool {
        true
    }
    fn build_lut(&self, params: &FilterParams) -> [u8; 256] {
        let level = params.get_uint("level").unwrap_or(128) as u8;
        point_ops::build_lut(&PointOp::Threshold(level))
    }
}

pub struct PosterizeFilter;
impl ImageFilter for PosterizeFilter {
    fn name(&self) -> &str {
        "posterize"
    }
    fn category(&self) -> FilterCategory {
        FilterCategory::PointOp
    }
    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        vec![ParamDescriptor::uint("levels", 2, 256, 4)]
    }
    fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError> {
        let levels = input.params.get_uint("levels").unwrap_or(4) as u8;
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::posterize_registered(r, &mut u, input.info, &filters::PosterizeParams { levels })
    }
    fn is_lut_collapsible(&self) -> bool {
        true
    }
    fn build_lut(&self, params: &FilterParams) -> [u8; 256] {
        let levels = params.get_uint("levels").unwrap_or(4) as u8;
        point_ops::build_lut(&PointOp::Posterize(levels))
    }
}

// Color operations
builtin_filter!(HueRotateFilter, "hue_rotate", FilterCategory::Color,
    params: [ParamDescriptor::float("degrees", -360.0, 360.0, 0.0)],
    apply: |input| {
        let degrees = input.params.get_float("degrees").unwrap_or(0.0);
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::hue_rotate(r, &mut u, input.info, &filters::HueRotateParams { degrees })
    }
);

builtin_filter!(SaturateFilter, "saturate", FilterCategory::Color,
    params: [ParamDescriptor::float("factor", 0.0, 3.0, 1.0)],
    apply: |input| {
        let factor = input.params.get_float("factor").unwrap_or(1.0);
        {
            let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
            let pixels = input.pixels.to_vec();
            let mut u = |_: rasmcore_pipeline::Rect| Ok(pixels.clone());
            filters::saturate(r, &mut u, input.info, &filters::SaturateParams { factor })
        }
    }
);

builtin_filter!(SepiaFilter, "sepia", FilterCategory::Color,
    params: [ParamDescriptor::float("intensity", 0.0, 1.0, 1.0)],
    apply: |input| {
        let intensity = input.params.get_float("intensity").unwrap_or(1.0);
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::sepia(r, &mut u, input.info, &filters::SepiaParams { intensity })
    }
);

// Histogram operations
builtin_filter!(EqualizeFilter, "equalize", FilterCategory::Contrast,
    params: [],
    apply: |input| {
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::equalize_registered(r, &mut u, input.info)
    }
);

builtin_filter!(NormalizeFilter, "normalize", FilterCategory::Contrast,
    params: [],
    apply: |input| {
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::normalize_registered(r, &mut u, input.info)
    }
);

builtin_filter!(AutoLevelFilter, "auto_level", FilterCategory::Contrast,
    params: [],
    apply: |input| {
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::auto_level_registered(r, &mut u, input.info)
    }
);

// OpenCV-tier filters
builtin_filter!(ClaheFilter, "clahe", FilterCategory::Contrast,
    params: [
        ParamDescriptor::float("clip_limit", 1.0, 100.0, 2.0).with_description("Contrast clip limit"),
        ParamDescriptor::uint("tile_grid", 1, 32, 8).with_description("Tile grid size")
    ],
    apply: |input| {
        let clip = input.params.get_float("clip_limit").unwrap_or(2.0);
        let grid = input.params.get_uint("tile_grid").unwrap_or(8);
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::clahe(r, &mut u, input.info, &filters::ClaheParams { clip_limit: clip, tile_grid: grid })
    }
);

builtin_filter!(BilateralFilter, "bilateral", FilterCategory::Denoise,
    params: [
        ParamDescriptor::uint("diameter", 0, 25, 9).with_description("Filter diameter (0=auto)"),
        ParamDescriptor::float("sigma_color", 1.0, 200.0, 75.0),
        ParamDescriptor::float("sigma_space", 1.0, 200.0, 75.0)
    ],
    apply: |input| {
        let d = input.params.get_uint("diameter").unwrap_or(9);
        let sc = input.params.get_float("sigma_color").unwrap_or(75.0);
        let ss = input.params.get_float("sigma_space").unwrap_or(75.0);
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(input.pixels.to_vec()) };
        filters::bilateral(r, &mut u, input.info, &filters::BilateralParams { diameter: d, sigma_color: sc, sigma_space: ss })
    }
);

builtin_filter!(GuidedFilterEntry, "guided_filter", FilterCategory::Denoise,
    params: [
        ParamDescriptor::uint("radius", 1, 50, 4),
        ParamDescriptor::float("epsilon", 0.001, 1.0, 0.01)
    ],
    apply: |input| {
        let r = input.params.get_uint("radius").unwrap_or(4);
        let eps = input.params.get_float("epsilon").unwrap_or(0.01);
        filters::guided_filter_impl(input.pixels, input.info, &filters::GuidedFilterParams { radius: r, epsilon: eps })
    }
);

// HDR merge operations (multi-image — single-image apply returns error)
builtin_filter!(MertensFusionFilter, "mertens_fusion", FilterCategory::Composite,
    params: [
        ParamDescriptor::float("contrast_weight", 0.0, 10.0, 1.0).with_description("Contrast metric weight"),
        ParamDescriptor::float("saturation_weight", 0.0, 10.0, 1.0).with_description("Saturation metric weight"),
        ParamDescriptor::float("exposure_weight", 0.0, 10.0, 1.0).with_description("Well-exposedness metric weight")
    ],
    apply: |_input| {
        Err(ImageError::InvalidInput(
            "mertens_fusion requires multiple images — use filters::mertens_fusion() directly".into()
        ))
    }
);

builtin_filter!(DebevecHdrFilter, "debevec_hdr", FilterCategory::Composite,
    params: [
        ParamDescriptor::uint("samples", 10, 500, 70).with_description("Sample pixels for response curve"),
        ParamDescriptor::float("lambda", 0.1, 100.0, 10.0).with_description("Smoothness regularization")
    ],
    apply: |_input| {
        Err(ImageError::InvalidInput(
            "debevec_hdr requires multiple images + exposure times — use filters::debevec_response_curve() and filters::debevec_hdr_merge() directly".into()
        ))
    }
);

// Adaptive thresholding
builtin_filter!(OtsuThresholdFilter, "otsu_threshold", FilterCategory::Contrast,
    params: [],
    apply: |input| {
        let t = filters::otsu_threshold(input.pixels, input.info)?;
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let mut u = |_: rasmcore_pipeline::Rect| Ok(input.pixels.to_vec());
        filters::threshold_binary(r, &mut u, input.info, &filters::ThresholdBinaryParams { thresh: t, max_value: 255 })
    }
);

builtin_filter!(TriangleThresholdFilter, "triangle_threshold", FilterCategory::Contrast,
    params: [],
    apply: |input| {
        let t = filters::triangle_threshold(input.pixels, input.info)?;
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let mut u = |_: rasmcore_pipeline::Rect| Ok(input.pixels.to_vec());
        filters::threshold_binary(r, &mut u, input.info, &filters::ThresholdBinaryParams { thresh: t, max_value: 255 })
    }
);

builtin_filter!(AdaptiveThresholdFilter, "adaptive_threshold", FilterCategory::Contrast,
    params: [
        ParamDescriptor::uint("block_size", 3, 99, 11).with_description("Block size (odd, >= 3)"),
        ParamDescriptor::float("c", -100.0, 100.0, 2.0).with_description("Constant subtracted from mean")
    ],
    apply: |input| {
        let block_size = input.params.get_uint("block_size").unwrap_or(11);
        let c = input.params.get_float("c").unwrap_or(2.0) as f64;
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::adaptive_threshold_registered(r, &mut u, input.info, &filters::AdaptiveThresholdParams { max_value: 255, method: 0, block_size, c: c as f32 })
    }
);

// Color quantization and dithering
builtin_filter!(QuantizeFilter, "quantize", FilterCategory::Color,
    params: [ParamDescriptor::float("colors", 2.0, 256.0, 16.0).with_description("Number of palette colors")],
    apply: |input| {
        let colors = input.params.get_float("colors").unwrap_or(16.0) as usize;
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::quantize_registered(r, &mut u, input.info, &filters::QuantizeParams { max_colors: colors as u32 })
    }
);

builtin_filter!(DitherFSFilter, "dither_floyd_steinberg", FilterCategory::Color,
    params: [ParamDescriptor::float("colors", 2.0, 256.0, 16.0).with_description("Number of palette colors")],
    apply: |input| {
        let colors = input.params.get_float("colors").unwrap_or(16.0) as usize;
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::dither_floyd_steinberg_registered(r, &mut u, input.info, &filters::DitherFloydSteinbergParams { max_colors: colors as u32 })
    }
);

builtin_filter!(DitherOrderedFilter, "dither_ordered", FilterCategory::Color,
    params: [
        ParamDescriptor::float("colors", 2.0, 256.0, 16.0).with_description("Number of palette colors"),
        ParamDescriptor::float("matrix_size", 2.0, 8.0, 4.0).with_description("Bayer matrix size (2, 4, or 8)")
    ],
    apply: |input| {
        let colors = input.params.get_float("colors").unwrap_or(16.0) as usize;
        let matrix_size = input.params.get_float("matrix_size").unwrap_or(4.0) as usize;
        let r = rasmcore_pipeline::Rect::new(0, 0, input.info.width, input.info.height);
        let p = input.pixels;
        let mut u = |_: rasmcore_pipeline::Rect| -> Result<Vec<u8>, ImageError> { Ok(p.to_vec()) };
        filters::dither_ordered_registered(r, &mut u, input.info, &filters::DitherOrderedParams { max_colors: colors as u32, map_size: matrix_size as u32 })
    }
);

/// Register all built-in filters into a registry.
fn register_builtin_filters(reg: &mut FilterRegistry) {
    reg.register(Box::new(BlurFilter));
    reg.register(Box::new(SharpenFilter));
    reg.register(Box::new(MedianFilter));
    reg.register(Box::new(SobelFilter));
    reg.register(Box::new(CannyFilter));
    reg.register(Box::new(GammaFilter));
    reg.register(Box::new(BrightnessFilter));
    reg.register(Box::new(ContrastFilter));
    reg.register(Box::new(InvertFilter));
    reg.register(Box::new(ThresholdFilter));
    reg.register(Box::new(PosterizeFilter));
    reg.register(Box::new(HueRotateFilter));
    reg.register(Box::new(SaturateFilter));
    reg.register(Box::new(SepiaFilter));
    reg.register(Box::new(EqualizeFilter));
    reg.register(Box::new(NormalizeFilter));
    reg.register(Box::new(AutoLevelFilter));
    reg.register(Box::new(ClaheFilter));
    reg.register(Box::new(BilateralFilter));
    reg.register(Box::new(GuidedFilterEntry));
    reg.register(Box::new(MertensFusionFilter));
    reg.register(Box::new(DebevecHdrFilter));
    reg.register(Box::new(OtsuThresholdFilter));
    reg.register(Box::new(TriangleThresholdFilter));
    reg.register(Box::new(AdaptiveThresholdFilter));
    reg.register(Box::new(QuantizeFilter));
    reg.register(Box::new(DitherFSFilter));
    reg.register(Box::new(DitherOrderedFilter));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ColorSpace, PixelFormat};

    #[test]
    fn registry_has_all_builtins() {
        let reg = FilterRegistry::with_builtins();
        assert!(
            reg.len() >= 25,
            "expected 25+ built-in filters, got {}",
            reg.len()
        );

        // Spot-check key filters exist
        for name in [
            "blur",
            "sharpen",
            "clahe",
            "bilateral",
            "guided_filter",
            "sobel",
            "canny",
            "gamma",
            "brightness",
            "contrast",
            "invert",
            "sepia",
            "equalize",
            "mertens_fusion",
            "debevec_hdr",
        ] {
            assert!(reg.get(name).is_some(), "missing built-in filter: {name}");
        }
    }

    #[test]
    fn filter_by_category() {
        let reg = FilterRegistry::with_builtins();
        let blur_filters = reg.by_category(FilterCategory::Blur);
        assert!(!blur_filters.is_empty());
        assert!(blur_filters.iter().any(|f| f.name() == "blur"));
    }

    #[test]
    fn apply_filter_via_registry() {
        let reg = FilterRegistry::with_builtins();
        let blur = reg.get("blur").unwrap();

        let info = ImageInfo {
            width: 8,
            height: 8,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![128u8; 64];
        let params = FilterParams::new().set_float("radius", 1.0);
        let input = FilterInput {
            pixels: &pixels,
            info: &info,
            params: &params,
        };

        let result = blur.apply(&input).unwrap();
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn custom_filter_registration() {
        struct DoubleFilter;
        impl ImageFilter for DoubleFilter {
            fn name(&self) -> &str {
                "double"
            }
            fn category(&self) -> FilterCategory {
                FilterCategory::Other
            }
            fn param_descriptors(&self) -> Vec<ParamDescriptor> {
                vec![]
            }
            fn apply(&self, input: &FilterInput) -> Result<Vec<u8>, ImageError> {
                Ok(input.pixels.iter().map(|&v| v.saturating_mul(2)).collect())
            }
        }

        let mut reg = FilterRegistry::with_builtins();
        let before = reg.len();
        reg.register(Box::new(DoubleFilter));
        assert_eq!(reg.len(), before + 1);
        assert!(reg.get("double").is_some());

        let info = ImageInfo {
            width: 4,
            height: 1,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let pixels = vec![10u8, 20, 30, 40];
        let params = FilterParams::new();
        let input = FilterInput {
            pixels: &pixels,
            info: &info,
            params: &params,
        };

        let result = reg.get("double").unwrap().apply(&input).unwrap();
        assert_eq!(result, vec![20, 40, 60, 80]);
    }

    #[test]
    fn param_descriptors_populated() {
        let reg = FilterRegistry::with_builtins();
        let clahe = reg.get("clahe").unwrap();
        let descs = clahe.param_descriptors();
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].name, "clip_limit");
        assert_eq!(descs[1].name, "tile_grid");
    }

    #[test]
    fn all_filter_names_unique() {
        let reg = FilterRegistry::with_builtins();
        let names = reg.names();
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(seen.insert(name), "duplicate filter name: {name}");
        }
    }

    #[test]
    fn static_registration_via_proc_macro() {
        // The #[register_filter] attribute on blur() in filters.rs should
        // register it via inventory. Verify it's discoverable.
        let regs = registered_filters();
        let blur_reg = regs.iter().find(|r| r.name == "blur");
        assert!(
            blur_reg.is_some(),
            "blur should be registered via #[register_filter]. Found: {:?}",
            regs.iter().map(|r| r.name).collect::<Vec<_>>()
        );
        let blur = blur_reg.unwrap();
        assert_eq!(blur.category, "spatial");
        assert_eq!(blur.fn_name, "blur");
    }

    #[test]
    fn config_params_derive_generates_descriptors() {
        let descriptors = crate::domain::filters::BlurParams::param_descriptors();
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].name, "radius");
        assert_eq!(descriptors[0].param_type, "f32");
        assert_eq!(descriptors[0].min, "0.0");
        assert_eq!(descriptors[0].max, "100.0");
        assert_eq!(descriptors[0].step, "0.5");
        assert_eq!(descriptors[0].default_val, "3.0");
        assert_eq!(descriptors[0].label, "Blur radius in pixels");
    }

    #[test]
    fn config_params_derive_generates_default() {
        let params = crate::domain::filters::BlurParams::default();
        assert!((params.radius - 3.0).abs() < 0.001);
    }
}
