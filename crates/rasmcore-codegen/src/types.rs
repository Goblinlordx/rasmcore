//! Shared data types for code generation.
//!
//! These structs are the structured intermediate representation:
//! parse (syn AST) → types (these structs) → gen (output code/JSON).

use serde::Serialize;

/// A parsed `#[register_filter]` registration with all metadata.
#[derive(Debug, Clone, Serialize)]
pub struct FilterReg {
    pub name: String,
    pub category: String,
    pub group: String,
    pub variant: String,
    pub reference: String,
    /// Overlap strategy: "zero", "uniform(N)", "full", "param(name)"
    pub overlap: String,
    /// Whether this filter is a per-channel LUT point operation (fuseable).
    /// Set via `point_op = "true"` in register_filter attributes.
    pub point_op: bool,
    pub color_op: bool,
    /// True if this filter uses the new rect-request signature:
    /// `fn(request: Rect, upstream: &mut UpstreamFn, info, ...)`
    /// False for legacy signature: `fn(pixels: &[u8], info, ...)`
    pub rect_request: bool,
    pub fn_name: String,
    /// (param_name, rust_type) pairs, excluding pixels/info.
    pub params: Vec<(String, String)>,
    /// If set, the struct name (e.g., "BlurParams") for config-based codegen.
    /// Detected by naming convention: `to_pascal_case(name) + "Params"`.
    /// When Some, generators produce config-struct-based signatures.
    /// When None, generators produce individual-parameter signatures (current behavior).
    pub config_struct: Option<String>,
}

/// A parsed simple registration (generator, compositor).
#[derive(Debug, Clone, Serialize)]
pub struct SimpleReg {
    pub name: String,
    pub category: String,
    pub group: String,
    pub variant: String,
    pub reference: String,
}

/// A parsed `#[register_mapper]` registration with full metadata.
///
/// Like `FilterReg` but for format-changing operations that return
/// `Result<(Vec<u8>, ImageInfo), ImageError>` — the output ImageInfo
/// may differ from the input (e.g., RGB8 → Gray8 for grayscale).
#[derive(Debug, Clone, Serialize)]
pub struct MapperReg {
    pub name: String,
    pub category: String,
    pub group: String,
    pub variant: String,
    pub reference: String,
    pub fn_name: String,
    /// (param_name, rust_type) pairs, excluding pixels/info.
    pub params: Vec<(String, String)>,
    /// If set, the struct name for config-based codegen.
    pub config_struct: Option<String>,
    /// Static output pixel format (e.g., "Gray8", "Rgb8", "Rgba8").
    /// When set, the pipeline node reports this format from `info()`
    /// at construction time. Required for correct downstream node setup.
    pub output_format: Option<String>,
}

/// A parsed field from a `#[derive(ConfigParams)]` struct.
#[derive(Debug, Clone, Serialize)]
pub struct ParamField {
    pub name: String,
    pub param_type: String,
    pub min: String,
    pub max: String,
    pub step: String,
    pub default_val: String,
    pub label: String,
    pub hint: String,
}

/// A parsed `#[derive(ConfigParams)]` struct with its fields and metadata.
#[derive(Debug, Clone)]
pub struct ParsedStruct {
    pub fields: Vec<ParamField>,
    pub config_hint: String,
}

/// Strongly-typed representation of a `#[param(...)]` attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum LitValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    Null,
}

impl LitValue {
    /// Convert to a string suitable for JSON output.
    pub fn to_json_string(&self) -> String {
        match self {
            LitValue::Float(f) => format!("{f}"),
            LitValue::Int(i) => format!("{i}"),
            LitValue::Bool(b) => format!("{b}"),
            LitValue::Str(s) => s.clone(),
            LitValue::Null => "null".to_string(),
        }
    }
}

/// Aggregated output of parsing all source files.
#[derive(Debug)]
pub struct CodegenData {
    pub filters: Vec<FilterReg>,
    pub generators: Vec<SimpleReg>,
    pub compositors: Vec<SimpleReg>,
    pub mappers: Vec<MapperReg>,
    pub param_structs: std::collections::HashMap<String, Vec<ParamField>>,
}
