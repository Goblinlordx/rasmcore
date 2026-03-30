//! Build script — uses rasmcore-codegen to parse #[register_filter] annotations
//! and generate adapter code, pipeline nodes, WIT declarations, and param-manifest.json.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let filters_path = Path::new(&manifest_dir).join("src/domain/filters.rs");
    let param_types_path = Path::new(&manifest_dir).join("src/domain/param_types.rs");
    let composite_path = Path::new(&manifest_dir).join("src/domain/composite.rs");

    // Tell cargo to rerun if source files or build.rs change
    println!("cargo:rerun-if-changed=src/domain/filters.rs");
    println!("cargo:rerun-if-changed=src/domain/param_types.rs");
    println!("cargo:rerun-if-changed=src/domain/composite.rs");
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
        std::fs::write(
            out_dir.join("param-manifest.json"),
            r#"{"filters":[],"generators":[],"compositors":[],"mappers":[]}"#,
        )
        .unwrap();
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
}
