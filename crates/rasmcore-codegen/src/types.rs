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
    pub fn_name: String,
    /// (param_name, rust_type) pairs, excluding pixels/info.
    pub params: Vec<(String, String)>,
    /// If set, the struct name (e.g., "BlurParams") for config-based codegen.
    /// Detected by naming convention: `to_pascal_case(name) + "Params"`.
    /// When Some, generators produce config-struct-based signatures.
    /// When None, generators produce individual-parameter signatures (current behavior).
    pub config_struct: Option<String>,
}

/// A parsed simple registration (generator, compositor, mapper).
#[derive(Debug, Clone, Serialize)]
pub struct SimpleReg {
    pub name: String,
    pub category: String,
    pub group: String,
    pub variant: String,
    pub reference: String,
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
    pub mappers: Vec<SimpleReg>,
    pub param_structs: std::collections::HashMap<String, Vec<ParamField>>,
}
