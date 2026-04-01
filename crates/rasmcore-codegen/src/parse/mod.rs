//! Source code parsing via syn AST.
//!
//! Parses Rust source files to extract registration metadata:
//! - `#[register_filter]` → `FilterReg`
//! - `#[register_generator/compositor/mapper]` → `SimpleReg`
//! - `#[derive(ConfigParams)]` → `ParsedStruct` / `ParamField`
//! - `#[param(min, max, ...)]` → `ParamAttr` (fully typed)

pub mod config_params;
pub mod encoders;
pub mod filters;
pub mod mappers;
pub mod param_attr;
pub mod simple_regs;
pub mod transforms;

use crate::types::CodegenData;
use std::path::Path;

/// Parse all source files and return aggregated codegen data.
///
/// Accepts paths to: filters.rs, param_types.rs, composite.rs (all optional except filters).
pub fn parse_source_files(
    filters_path: &Path,
    param_types_path: Option<&Path>,
    composite_path: Option<&Path>,
) -> CodegenData {
    let mut all_filters = Vec::new();
    let mut all_generators = Vec::new();
    let mut all_compositors = Vec::new();
    let mut all_mappers = Vec::new();
    let mut all_params = std::collections::HashMap::new();

    // Parse each file separately (syn requires valid individual files)
    for path in [Some(filters_path), param_types_path, composite_path]
        .into_iter()
        .flatten()
    {
        if !path.exists() {
            continue;
        }
        let source = std::fs::read_to_string(path).unwrap_or_default();
        let file = match syn::parse_file(&source) {
            Ok(f) => f,
            Err(e) => {
                eprintln!(
                    "rasmcore-codegen: syn parse error in {}: {e}",
                    path.display()
                );
                continue;
            }
        };

        all_filters.extend(filters::extract_filters(&file));
        all_generators.extend(simple_regs::extract_by_kind(&file, "register_generator"));
        all_compositors.extend(simple_regs::extract_by_kind(&file, "register_compositor"));
        all_mappers.extend(mappers::extract_mappers(&file));
        all_params.extend(config_params::extract_config_params(&file));
    }

    // Auto-detect config structs by naming convention:
    // If param_structs contains "{PascalCaseName}Params", link it to the filter/mapper.
    for filter in &mut all_filters {
        let expected = format!(
            "{}Params",
            crate::generate::helpers::to_pascal_case(&filter.name)
        );
        if all_params.contains_key(&expected) {
            filter.config_struct = Some(expected);
        }
    }
    for mapper in &mut all_mappers {
        let expected = format!(
            "{}Params",
            crate::generate::helpers::to_pascal_case(&mapper.name)
        );
        if all_params.contains_key(&expected) {
            mapper.config_struct = Some(expected);
        }
    }

    CodegenData {
        filters: all_filters,
        generators: all_generators,
        compositors: all_compositors,
        mappers: all_mappers,
        param_structs: all_params,
    }
}

/// Convert a syn::Type to a clean string representation.
pub fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Reference(r) => {
            let inner = type_to_string(&r.elem);
            format!("&{inner}")
        }
        syn::Type::Slice(s) => {
            let inner = type_to_string(&s.elem);
            format!("[{inner}]")
        }
        syn::Type::Path(p) => p
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default(),
        _ => quote::quote!(#ty).to_string().replace(' ', ""),
    }
}
