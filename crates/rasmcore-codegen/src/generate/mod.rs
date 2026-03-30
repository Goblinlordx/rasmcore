//! Code generation modules.
//!
//! Each module takes structured `CodegenData` and produces a specific output format.

pub mod adapter;
pub mod helpers;
pub mod manifest;
pub mod pipeline;
pub mod wit;

use crate::types::CodegenData;
use std::fs;
use std::path::Path;

/// Generate all output files from codegen data.
pub fn generate_all(data: &CodegenData, out_dir: &Path) {
    // param-manifest.json
    let manifest_json = manifest::generate(data);
    fs::write(out_dir.join("param-manifest.json"), &manifest_json).unwrap();

    // Filter adapter dispatch code
    let adapter_code = adapter::generate(&data.filters);
    fs::write(out_dir.join("generated_filter_adapter.rs"), &adapter_code).unwrap();

    // Pipeline node structs
    let nodes_code = pipeline::generate_nodes(&data.filters);
    fs::write(out_dir.join("generated_pipeline_nodes.rs"), &nodes_code).unwrap();

    // Pipeline adapter macro
    let pipe_adapter = pipeline::generate_adapter_macro(&data.filters);
    fs::write(out_dir.join("generated_pipeline_adapter.rs"), &pipe_adapter).unwrap();

    // WIT declarations (to stderr for review)
    let wit_decls = wit::generate(&data.filters);
    if !wit_decls.is_empty() {
        eprintln!("\n--- Generated WIT (copy to filters.wit if new filters added) ---");
        eprint!("{wit_decls}");
        eprintln!("--- End WIT ---\n");
    }

    eprintln!(
        "rasmcore-codegen: Generated {} filters, {} generators, {} compositors, {} mappers",
        data.filters.len(),
        data.generators.len(),
        data.compositors.len(),
        data.mappers.len()
    );
}
