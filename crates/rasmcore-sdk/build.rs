use std::{env, fs, path::Path};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let image_crate = Path::new(&manifest_dir).join("../rasmcore-image");
    let filters_path = image_crate.join("src/domain/filters.rs");
    let param_types_path = image_crate.join("src/domain/param_types.rs");
    let composite_path = image_crate.join("src/domain/composite.rs");

    println!("cargo:rerun-if-changed={}", filters_path.display());
    if param_types_path.exists() {
        println!("cargo:rerun-if-changed={}", param_types_path.display());
    }
    if composite_path.exists() {
        println!("cargo:rerun-if-changed={}", composite_path.display());
    }

    if !filters_path.exists() {
        return;
    }

    let data = rasmcore_codegen::parse::parse_source_files(
        &filters_path,
        if param_types_path.exists() {
            Some(param_types_path.as_path())
        } else {
            None
        },
        if composite_path.exists() {
            Some(composite_path.as_path())
        } else {
            None
        },
    );

    let out_dir = env::var("OUT_DIR").unwrap();
    let sdk_code = rasmcore_codegen::generate::sdk_rust::generate(&data.filters);
    fs::write(Path::new(&out_dir).join("generated_sdk_rust.rs"), sdk_code).unwrap();

    eprintln!(
        "rasmcore-sdk: Generated SDK with {} filter methods",
        data.filters.len()
    );
}
