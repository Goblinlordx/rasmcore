//! Generate pipeline adapter methods for registered transforms.
//!
//! Produces a macro_rules! that expands inside `impl GuestImagePipeline`.
//! Each transform method: gets src_info, converts WIT params, constructs
//! the node, computes hash, calls add_node_derived.

use crate::parse::transforms::{EnumDef, TransformReg};
use std::collections::HashMap;

/// Convert PascalCase to snake_case.
pub fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_ascii_lowercase());
    }
    result
}

/// Convert PascalCase to kebab-case (for WIT identifiers).
fn to_kebab_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('-');
        }
        result.push(ch.to_ascii_lowercase());
    }
    result
}

/// Convert snake_case to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().chain(c).collect(),
            }
        })
        .collect()
}

/// Generate WIT enum declarations for all enum types used by transforms.
pub fn generate_wit_enums(enums: &HashMap<String, EnumDef>, used_enums: &[String]) -> String {
    let mut wit = String::new();
    for enum_name in used_enums {
        if let Some(def) = enums.get(enum_name) {
            let wit_name = to_kebab_case(enum_name);
            let wit_variants: Vec<String> = def.variants.iter().map(|v| to_kebab_case(v)).collect();
            wit.push_str(&format!(
                "    enum {} {{ {} }}\n\n",
                wit_name,
                wit_variants.join(", ")
            ));
        }
    }
    wit
}

/// Generate WIT config records for transforms.
pub fn generate_wit_configs(
    transforms: &[TransformReg],
    enums: &HashMap<String, EnumDef>,
) -> String {
    let mut wit = String::new();
    for t in transforms {
        if t.params.is_empty() {
            continue;
        }
        let wit_name = t.name.replace('_', "-");
        wit.push_str(&format!("    record {}-config {{\n", wit_name));
        for (pname, ptype) in &t.params {
            let wit_type = match classify_param(ptype, enums) {
                ParamKind::Enum(name) => to_kebab_case(name),
                ParamKind::OptionalEnum(name) => format!("option<{}>", to_kebab_case(name)),
                ParamKind::Blob => "buffer".to_string(),
                ParamKind::Primitive => rust_type_to_wit(ptype).to_string(),
            };
            wit.push_str(&format!(
                "        {}: {},\n",
                pname.replace('_', "-"),
                wit_type
            ));
        }
        wit.push_str("    }\n\n");
    }
    wit
}

/// Generate WIT method declarations for transforms.
pub fn generate_wit_methods(transforms: &[TransformReg]) -> String {
    let mut wit = String::new();
    for t in transforms {
        let wit_name = t.name.replace('_', "-");
        if t.multi_input {
            let config_type = format!("{}-config", wit_name);
            wit.push_str(&format!(
                "        {}: func(fg: node-id, bg: node-id, config: {}) -> result<node-id, rasmcore-error>;\n",
                wit_name, config_type
            ));
        } else if t.params.is_empty() {
            wit.push_str(&format!(
                "        {}: func(source: node-id) -> result<node-id, rasmcore-error>;\n",
                wit_name
            ));
        } else {
            let config_type = format!("{}-config", wit_name);
            wit.push_str(&format!(
                "        {}: func(source: node-id, config: {}) -> result<node-id, rasmcore-error>;\n",
                wit_name, config_type
            ));
        }
    }
    wit
}

/// Generate the pipeline adapter macro for transforms.
pub fn generate_adapter_macro(
    transforms: &[TransformReg],
    enums: &HashMap<String, EnumDef>,
) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline transform adapter methods.\n");
    code.push_str("// Do not edit — regenerate by changing transform nodes and rebuilding.\n\n");

    // Generate enum conversion functions for all enum types used (including inside Option<>)
    let mut generated_converters = std::collections::HashSet::new();
    for t in transforms {
        for (_, ptype) in &t.params {
            let enum_name = match classify_param(ptype, enums) {
                ParamKind::Enum(name) => Some(name),
                ParamKind::OptionalEnum(name) => Some(name),
                _ => None,
            };
            if let Some(name) = enum_name {
                if !generated_converters.contains(name) {
                    code.push_str(&generate_enum_converter(name, enums));
                    generated_converters.insert(name.to_string());
                }
            }
        }
    }

    code.push_str("macro_rules! generated_pipeline_transform_methods {\n    () => {\n");

    for t in transforms {
        if t.multi_input {
            code.push_str(&generate_multi_input_transform_method(t, enums));
        } else {
            code.push_str(&generate_single_transform_method(t, enums));
        }
    }

    code.push_str("    } // end macro\n}\n");
    code
}

fn generate_single_transform_method(t: &TransformReg, enums: &HashMap<String, EnumDef>) -> String {
    let mut code = String::new();
    let method_name = &t.name;
    let node_type = &t.node_type;
    let node_module = &t.node_module;

    if t.params.is_empty() {
        // No-param transform (unlikely but possible)
        code.push_str(&format!(
            "    fn {method_name}(&self, source: NodeId) -> Result<NodeId, RasmcoreError> {{\n"
        ));
    } else {
        let config_pascal = to_pascal_case(&format!("{}_config", t.name));
        code.push_str(&format!(
            "    fn {method_name}(&self, source: NodeId, config: pipeline::{config_pascal}) -> Result<NodeId, RasmcoreError> {{\n"
        ));
    }

    // Get source info
    code.push_str(
        "        let src_info = self.graph.borrow().node_info(source).map_err(to_wit_error)?;\n",
    );

    // Build constructor args — convert WIT types to domain types
    let mut constructor_args = vec!["source".to_string(), "src_info".to_string()];
    let mut hash_parts = Vec::new();

    for (pname, ptype) in &t.params {
        match classify_param(ptype, enums) {
            ParamKind::Enum(enum_name) => {
                let converter_fn = format!("__convert_{}", to_snake_case(enum_name));
                code.push_str(&format!(
                    "        let {pname} = {converter_fn}(config.{pname});\n"
                ));
                constructor_args.push(pname.clone());
                hash_parts.push(pname.clone());
            }
            ParamKind::OptionalEnum(enum_name) => {
                let converter_fn = format!("__convert_{}", to_snake_case(enum_name));
                code.push_str(&format!(
                    "        let {pname} = config.{pname}.map({converter_fn});\n"
                ));
                constructor_args.push(pname.clone());
                hash_parts.push(pname.clone());
            }
            ParamKind::Blob => {
                code.push_str(&format!("        let {pname} = config.{pname};\n"));
                constructor_args.push(pname.clone());
            }
            ParamKind::Primitive => {
                code.push_str(&format!("        let {pname} = config.{pname};\n"));
                constructor_args.push(pname.clone());
                hash_parts.push(pname.clone());
            }
        }
    }

    // Compute hash before node construction (avoids move issues with Vec<u8> params)
    let hash_format = if hash_parts.is_empty() {
        "b\"\"".to_string()
    } else {
        let placeholders: Vec<String> = hash_parts.iter().map(|_| "{:?}".to_string()).collect();
        let args = hash_parts.join(", ");
        format!(
            "format!(\"{}\", {}).as_bytes()",
            placeholders.join(","),
            args
        )
    };
    code.push_str("        let mut graph = self.graph.borrow_mut();\n");
    code.push_str("        let upstream_hash = graph.node_hash(source);\n");
    code.push_str(&format!(
        "        let hash = rasmcore_pipeline::compute_hash(&upstream_hash, \"{method_name}\", {hash_format});\n"
    ));

    // Construct node
    let args_str = constructor_args.join(", ");
    if t.fallible {
        code.push_str(&format!(
            "        let node = {node_module}::{node_type}::new({args_str}).map_err(to_wit_error)?;\n"
        ));
    } else {
        code.push_str(&format!(
            "        let node = {node_module}::{node_type}::new({args_str});\n"
        ));
    }

    // Add node with descriptor tracking (kind + name for graph introspection)
    code.push_str(&format!(
        "        Ok(graph.add_node_described(Box::new(node), hash, source, crate::domain::pipeline::graph::NodeKind::Transform, \"{method_name}\"))\n",
    ));

    code.push_str("    }\n\n");
    code
}

/// Generate a multi-input transform adapter method (e.g., composite with fg + bg).
fn generate_multi_input_transform_method(
    t: &TransformReg,
    enums: &HashMap<String, EnumDef>,
) -> String {
    let mut code = String::new();
    let method_name = &t.name;
    let node_type = &t.node_type;
    let node_module = &t.node_module;

    // Multi-input: two source nodes (fg, bg) + config
    let config_pascal = to_pascal_case(&format!("{}_config", t.name));
    code.push_str(&format!(
        "    fn {method_name}(&self, fg: NodeId, bg: NodeId, config: pipeline::{config_pascal}) -> Result<NodeId, RasmcoreError> {{\n"
    ));

    // Get both source infos
    code.push_str("        let graph = self.graph.borrow();\n");
    code.push_str("        let fg_info = graph.node_info(fg).map_err(to_wit_error)?;\n");
    code.push_str("        let bg_info = graph.node_info(bg).map_err(to_wit_error)?;\n");
    code.push_str("        drop(graph);\n");

    // Convert params
    let mut constructor_args = vec![
        "fg".to_string(),
        "bg".to_string(),
        "fg_info".to_string(),
        "bg_info".to_string(),
    ];
    let mut hash_parts = Vec::new();

    for (pname, ptype) in &t.params {
        match classify_param(ptype, enums) {
            ParamKind::Enum(enum_name) => {
                let converter_fn = format!("__convert_{}", to_snake_case(enum_name));
                code.push_str(&format!(
                    "        let {pname} = {converter_fn}(config.{pname});\n"
                ));
                constructor_args.push(pname.clone());
                hash_parts.push(pname.clone());
            }
            ParamKind::OptionalEnum(enum_name) => {
                let converter_fn = format!("__convert_{}", to_snake_case(enum_name));
                code.push_str(&format!(
                    "        let {pname} = config.{pname}.map({converter_fn});\n"
                ));
                constructor_args.push(pname.clone());
                hash_parts.push(pname.clone());
            }
            ParamKind::Blob => {
                code.push_str(&format!("        let {pname} = config.{pname};\n"));
                constructor_args.push(pname.clone());
            }
            ParamKind::Primitive => {
                code.push_str(&format!("        let {pname} = config.{pname};\n"));
                constructor_args.push(pname.clone());
                hash_parts.push(pname.clone());
            }
        }
    }

    // Hash from both upstreams
    let hash_format = if hash_parts.is_empty() {
        "b\"\"".to_string()
    } else {
        let placeholders: Vec<String> = hash_parts.iter().map(|_| "{:?}".to_string()).collect();
        let args = hash_parts.join(", ");
        format!(
            "format!(\"{}\", {}).as_bytes()",
            placeholders.join(","),
            args
        )
    };
    code.push_str("        let fg_hash = self.graph.borrow().node_hash(fg);\n");
    code.push_str("        let bg_hash = self.graph.borrow().node_hash(bg);\n");
    code.push_str(&format!(
        "        let combined = rasmcore_pipeline::compute_hash(&fg_hash, \"{method_name}_fg\", {hash_format});\n"
    ));
    code.push_str(&format!(
        "        let hash = rasmcore_pipeline::compute_hash(&combined, \"{method_name}_bg\", &bg_hash);\n"
    ));

    // Construct node
    let args_str = constructor_args.join(", ");
    if t.fallible {
        code.push_str(&format!(
            "        let node = {node_module}::{node_type}::new({args_str}).map_err(to_wit_error)?;\n"
        ));
    } else {
        code.push_str(&format!(
            "        let node = {node_module}::{node_type}::new({args_str});\n"
        ));
    }

    // Add node (multi-input doesn't use add_node_derived since there are two upstreams)
    code.push_str("        Ok(self.graph.borrow_mut().add_node_with_hash(Box::new(node), hash))\n");

    code.push_str("    }\n\n");
    code
}

/// Generate a WIT enum → domain enum conversion function.
fn generate_enum_converter(enum_name: &str, enums: &HashMap<String, EnumDef>) -> String {
    let def = match enums.get(enum_name) {
        Some(d) => d,
        None => return String::new(),
    };

    let snake = to_snake_case(enum_name);
    let wit_pascal = to_pascal_case(&snake);
    let domain_mod = &def.domain_module;

    let mut code = String::new();
    code.push_str(&format!(
        "fn __convert_{snake}(v: pipeline::{wit_pascal}) -> domain::{domain_mod}::{enum_name} {{\n"
    ));
    code.push_str("    match v {\n");
    for variant in &def.variants {
        code.push_str(&format!(
            "        pipeline::{wit_pascal}::{variant} => domain::{domain_mod}::{enum_name}::{variant},\n"
        ));
    }
    code.push_str("    }\n}\n\n");
    code
}

/// Check if a type is Option<T> and return the inner type if so.
fn unwrap_option(ty: &str) -> Option<&str> {
    ty.strip_prefix("Option<").and_then(|s| s.strip_suffix('>'))
}

/// Determine how to handle a parameter type for codegen.
enum ParamKind<'a> {
    Enum(&'a str),
    OptionalEnum(&'a str),
    Blob,
    Primitive,
}

fn classify_param<'a>(ptype: &'a str, enums: &HashMap<String, EnumDef>) -> ParamKind<'a> {
    if enums.contains_key(ptype) {
        ParamKind::Enum(ptype)
    } else if let Some(inner) = unwrap_option(ptype) {
        if enums.contains_key(inner) {
            ParamKind::OptionalEnum(inner)
        } else {
            ParamKind::Primitive
        }
    } else if ptype == "Vec<u8>" {
        ParamKind::Blob
    } else {
        ParamKind::Primitive
    }
}

fn rust_type_to_wit(ty: &str) -> &str {
    match ty {
        "u8" => "u8",
        "u16" => "u16",
        "u32" => "u32",
        "u64" => "u64",
        "i8" => "s8",
        "i16" => "s16",
        "i32" => "s32",
        "i64" => "s64",
        "f32" => "f32",
        "f64" => "f64",
        "bool" => "bool",
        "String" | "&str" => "string",
        "Vec<u8>" | "Vec < u8 >" => "buffer",
        _ => "u32", // fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_kebab_case_works() {
        assert_eq!(to_kebab_case("ResizeFilter"), "resize-filter");
        assert_eq!(to_kebab_case("R90"), "r90");
        assert_eq!(to_kebab_case("FlipDirection"), "flip-direction");
    }

    #[test]
    fn generate_enum_converter_works() {
        let mut enums = HashMap::new();
        enums.insert(
            "Rotation".to_string(),
            EnumDef {
                name: "Rotation".to_string(),
                variants: vec!["R90".to_string(), "R180".to_string(), "R270".to_string()],
                domain_module: "types".to_string(),
            },
        );
        let code = generate_enum_converter("Rotation", &enums);
        assert!(code.contains("fn __convert_rotation"));
        assert!(code.contains("pipeline::Rotation::R90 => domain::types::Rotation::R90"));
    }
}
