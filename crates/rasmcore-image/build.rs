//! Build script — uses rasmcore-codegen to parse #[register_filter] annotations
//! and generate adapter code, pipeline nodes, WIT declarations, and param-manifest.json.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let filters_path = Path::new(&manifest_dir).join("src/domain/filters.rs");
    let param_types_path = Path::new(&manifest_dir).join("src/domain/param_types.rs");
    let composite_path = Path::new(&manifest_dir).join("src/domain/composite.rs");
    let encoder_dir = Path::new(&manifest_dir).join("src/domain/encoder");

    // Tell cargo to rerun if source files or build.rs change
    println!("cargo:rerun-if-changed=src/domain/filters.rs");
    println!("cargo:rerun-if-changed=src/domain/param_types.rs");
    println!("cargo:rerun-if-changed=src/domain/composite.rs");
    println!("cargo:rerun-if-changed=src/domain/encoder");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    if !filters_path.exists() {
        // Write empty generated files to prevent include! errors
        std::fs::write(
            out_dir.join("generated_filter_adapter.rs"),
            "// No filters\n",
        )
        .unwrap();
        std::fs::write(
            out_dir.join("generated_pipeline_nodes.rs"),
            "// No filters\n",
        )
        .unwrap();
        std::fs::write(
            out_dir.join("generated_pipeline_adapter.rs"),
            "macro_rules! generated_pipeline_filter_methods { () => {} }\n",
        )
        .unwrap();
        let empty = r#"{"filters":[],"generators":[],"compositors":[],"mappers":[]}"#;
        std::fs::write(out_dir.join("param-manifest.json"), empty).unwrap();
        std::fs::write(out_dir.join("param-manifest.hash"), "0000000000000000").unwrap();
        return;
    }

    // Parse all source files via syn AST
    let pt = if param_types_path.exists() {
        Some(param_types_path.as_path())
    } else {
        None
    };
    let cp = if composite_path.exists() {
        Some(composite_path.as_path())
    } else {
        None
    };
    let data = rasmcore_codegen::parse::parse_source_files(&filters_path, pt, cp);

    // Duplicate filter name detection — fail at compile time
    {
        let mut seen: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        for f in &data.filters {
            if let Some(existing_fn) = seen.get(f.name.as_str()) {
                panic!(
                    "\n\nDuplicate filter name '{}' registered by:\n  fn {}\n  fn {}\n\
                     Rename one to resolve the conflict.\n",
                    f.name, f.fn_name, existing_fn
                );
            }
            seen.insert(&f.name, &f.fn_name);
        }
    }

    // Print summary
    eprintln!(
        "rasmcore build.rs: {} filter(s), {} generator(s), {} compositor(s), {} mapper(s)",
        data.filters.len(),
        data.generators.len(),
        data.compositors.len(),
        data.mappers.len()
    );

    // Generate all output files
    rasmcore_codegen::generate::generate_all(&data, &out_dir);

    // ── Parse encoder configs for pipeline write method generation ──
    let mut encoder_configs = Vec::new();
    if encoder_dir.exists() {
        for entry in std::fs::read_dir(&encoder_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().map(|e| e == "rs").unwrap_or(false) {
                let source = std::fs::read_to_string(&path).unwrap();
                if let Ok(file) = syn::parse_file(&source) {
                    let mut configs =
                        rasmcore_codegen::parse::encoders::extract_encoder_configs(&file);
                    // Only include encoders that have a corresponding sink function
                    let sink_source = std::fs::read_to_string(
                        Path::new(&manifest_dir).join("src/domain/pipeline/nodes/sink.rs"),
                    )
                    .unwrap_or_default();
                    for config in &mut configs {
                        let sig_pattern = format!("fn write_{}", config.format);
                        if let Some(pos) = sink_source.find(&sig_pattern) {
                            let sig_end = sink_source[pos..].find('{').unwrap_or(200);
                            let sig = &sink_source[pos..pos + sig_end];
                            config.sink_takes_metadata = sig.contains("metadata");
                        }
                    }
                    // Filter: only generate adapter for formats with sink functions
                    configs.retain(|c| {
                        let has_sink = sink_source.contains(&format!("fn write_{}", c.format));
                        if !has_sink {
                            eprintln!("rasmcore build.rs: skipping {} (no sink function)", c.format);
                        }
                        has_sink
                    });
                    encoder_configs.extend(configs);
                }
            }
        }
    }

    // Generate pipeline write adapter code
    if !encoder_configs.is_empty() {
        let write_adapter =
            rasmcore_codegen::generate::pipeline_write::generate_adapter_methods(&encoder_configs);
        std::fs::write(
            out_dir.join("generated_pipeline_write_adapter.rs"),
            &write_adapter,
        )
        .unwrap();

        eprintln!(
            "rasmcore build.rs: Generated {} pipeline write adapter method(s)",
            encoder_configs.len()
        );
    } else {
        std::fs::write(
            out_dir.join("generated_pipeline_write_adapter.rs"),
            "macro_rules! generated_pipeline_write_methods { () => {} }\n",
        )
        .unwrap();
    }
}
