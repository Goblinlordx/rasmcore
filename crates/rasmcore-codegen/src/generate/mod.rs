//! Code generation modules.
//!
//! Each module takes structured `CodegenData` and produces a specific output format.

pub mod adapter;
pub mod cli_dispatch;
pub mod helpers;
pub mod manifest;
pub mod pipeline;
pub mod pipeline_mapper;
pub mod pipeline_write;
pub mod sdk_rust;
pub mod transform;
pub mod wit;

use crate::types::CodegenData;
use std::fs;
use std::path::Path;

/// FNV-1a 64-bit hash — fast, deterministic, no dependencies.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Generate all output files from codegen data.
pub fn generate_all(data: &CodegenData, out_dir: &Path) {
    // param-manifest.json + content hash for SDK version validation
    let manifest_json = manifest::generate(data);
    fs::write(out_dir.join("param-manifest.json"), &manifest_json).unwrap();
    let hash = fnv1a_64(manifest_json.as_bytes());
    fs::write(out_dir.join("param-manifest.hash"), format!("{hash:016x}")).unwrap();

    // Filter adapter dispatch code
    let adapter_code = adapter::generate(&data.filters, &data.param_structs);
    fs::write(out_dir.join("generated_filter_adapter.rs"), &adapter_code).unwrap();

    // Pipeline node structs (filters + mappers)
    let mut nodes_code = pipeline::generate_nodes(&data.filters);
    nodes_code.push_str(&pipeline_mapper::generate_mapper_nodes(&data.mappers));
    fs::write(out_dir.join("generated_pipeline_nodes.rs"), &nodes_code).unwrap();

    // Pipeline adapter macro (filters)
    let pipe_adapter = pipeline::generate_adapter_macro(&data.filters, &data.param_structs);
    fs::write(out_dir.join("generated_pipeline_adapter.rs"), &pipe_adapter).unwrap();

    // Pipeline mapper adapter macro
    let mapper_adapter = pipeline_mapper::generate_mapper_adapter_macro(&data.mappers);
    fs::write(
        out_dir.join("generated_pipeline_mapper_adapter.rs"),
        &mapper_adapter,
    )
    .unwrap();

    // Rust native SDK
    let sdk_rs = sdk_rust::generate(&data.filters);
    fs::write(out_dir.join("generated_sdk_rust.rs"), &sdk_rs).unwrap();

    // CLI dispatch table (filters + mappers)
    let cli_dispatch = cli_dispatch::generate(&data.filters, &data.mappers);
    fs::write(out_dir.join("generated_cli_dispatch.rs"), &cli_dispatch).unwrap();

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
