//! rasmcore-codegen — Code generation for rasmcore.
//!
//! Parses Rust source files via syn AST to extract registration metadata,
//! then generates manifests, adapter code, pipeline nodes, and WIT declarations.
//!
//! Architecture: parse (syn AST) → types (structured data) → gen (output)
//!
//! # Usage from build.rs
//!
//! ```ignore
//! use rasmcore_codegen::{parse, gen};
//! let data = parse::parse_source_files(filters_path, Some(param_types_path), Some(composite_path));
//! gen::generate_all(&data, &out_dir);
//! ```

pub mod generate;
pub mod parse;
pub mod types;
