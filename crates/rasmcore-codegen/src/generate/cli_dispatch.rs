//! Generate CLI dispatch function — maps filter/mapper name + string params to typed node construction.

use crate::types::{FilterReg, MapperReg};

use super::helpers::to_pascal_case;

/// Generate a dispatch function that creates pipeline nodes from name + string params.
///
/// Produces `generated_cli_dispatch.rs` with:
/// - `dispatch_filter(name, upstream, info, params) -> Result<Box<dyn ImageNode>, String>`
/// - `list_filters() -> Vec<FilterMeta>` for --list-filters
pub fn generate(filters: &[FilterReg], mappers: &[MapperReg]) -> String {
    let mut code = String::new();
    code.push_str(
        "// Auto-generated CLI dispatch — maps filter names to typed node constructors.\n",
    );
    code.push_str("// Do not edit — regenerated from #[register_filter] annotations.\n\n");
    code.push_str("use crate::domain::pipeline::graph::ImageNode;\n");
    code.push_str("use crate::domain::pipeline::nodes::filters;\n");
    code.push_str("use crate::domain::filters as domain_filters; // ConfigParams structs\n");
    code.push_str("use crate::domain::types::ImageInfo;\n");
    code.push_str("use std::collections::HashMap;\n\n");

    // Helper to parse typed values from string
    code.push_str("#[allow(dead_code)]\n");
    code.push_str(
        "fn get_f32(params: &HashMap<String, String>, key: &str, default: f32) -> f32 {\n",
    );
    code.push_str("    params.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)\n");
    code.push_str("}\n\n");

    code.push_str("#[allow(dead_code)]\n");
    code.push_str(
        "fn get_u32(params: &HashMap<String, String>, key: &str, default: u32) -> u32 {\n",
    );
    code.push_str("    params.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)\n");
    code.push_str("}\n\n");

    code.push_str("#[allow(dead_code)]\n");
    code.push_str("fn get_u8(params: &HashMap<String, String>, key: &str, default: u8) -> u8 {\n");
    code.push_str("    params.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)\n");
    code.push_str("}\n\n");

    code.push_str("#[allow(dead_code)]\n");
    code.push_str(
        "fn get_i32(params: &HashMap<String, String>, key: &str, default: i32) -> i32 {\n",
    );
    code.push_str("    params.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)\n");
    code.push_str("}\n\n");

    code.push_str("#[allow(dead_code)]\n");
    code.push_str(
        "fn get_bool(params: &HashMap<String, String>, key: &str, default: bool) -> bool {\n",
    );
    code.push_str("    params.get(key).map(|v| v == \"true\" || v == \"1\" || v == \"yes\").unwrap_or(default)\n");
    code.push_str("}\n\n");

    code.push_str("#[allow(dead_code)]\n");
    code.push_str(
        "fn get_string(params: &HashMap<String, String>, key: &str, default: &str) -> String {\n",
    );
    code.push_str("    params.get(key).cloned().unwrap_or_else(|| default.to_string())\n");
    code.push_str("}\n\n");

    // Array parsing: comma-separated values, with optional "WxH:" prefix for 2D matrices
    code.push_str("#[allow(dead_code)]\n");
    code.push_str("fn get_f32_array(params: &HashMap<String, String>, key: &str) -> Vec<f32> {\n");
    code.push_str("    match params.get(key) {\n");
    code.push_str("        Some(v) => {\n");
    code.push_str("            // Strip optional \"WxH:\" dimension prefix\n");
    code.push_str("            let values_str = if let Some(pos) = v.find(':') {\n");
    code.push_str("                &v[pos + 1..]\n");
    code.push_str("            } else {\n");
    code.push_str("                v.as_str()\n");
    code.push_str("            };\n");
    code.push_str(
        "            values_str.split(',').filter_map(|s| s.trim().parse().ok()).collect()\n",
    );
    code.push_str("        }\n");
    code.push_str("        None => Vec::new(),\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");

    code.push_str("fn get_f64_array(params: &HashMap<String, String>, key: &str) -> Vec<f64> {\n");
    code.push_str("    match params.get(key) {\n");
    code.push_str("        Some(v) => {\n");
    code.push_str("            let values_str = if let Some(pos) = v.find(':') {\n");
    code.push_str("                &v[pos + 1..]\n");
    code.push_str("            } else {\n");
    code.push_str("                v.as_str()\n");
    code.push_str("            };\n");
    code.push_str(
        "            values_str.split(',').filter_map(|s| s.trim().parse().ok()).collect()\n",
    );
    code.push_str("        }\n");
    code.push_str("        None => Vec::new(),\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");

    // Main dispatch function
    code.push_str("/// Dispatch a filter by name: parse string params, construct typed node.\n");
    code.push_str("pub fn dispatch_filter(\n");
    code.push_str("    name: &str,\n");
    code.push_str("    upstream: u32,\n");
    code.push_str("    info: ImageInfo,\n");
    code.push_str("    params: &HashMap<String, String>,\n");
    code.push_str(") -> Result<Box<dyn ImageNode>, String> {\n");
    code.push_str("    match name {\n");

    for f in filters {
        let node_name = format!("{}Node", to_pascal_case(&f.name));

        // Build param extraction lines
        let mut param_args = Vec::new();
        for (pname, ptype) in &f.params {
            let clean_name = pname.trim_start_matches('_');
            let getter = if ptype.starts_with('&') && ptype.ends_with("Params") {
                // Config struct reference — construct from CLI params
                // The struct type is e.g., &SpinBlurParams → SpinBlurParams
                let struct_name = &ptype[1..]; // strip leading &
                                               // Use Default::default() — the CLI will override via individual params
                                               // This is a simplification; full CLI config struct support would
                                               // parse each field from the HashMap.
                format!("domain_filters::{struct_name}::default()")
            } else {
                match ptype.as_str() {
                    "f32" => format!("get_f32(params, \"{clean_name}\", 0.0)"),
                    "f64" => format!("get_f32(params, \"{clean_name}\", 0.0) as f64"),
                    "u32" => format!("get_u32(params, \"{clean_name}\", 0)"),
                    "u8" => format!("get_u8(params, \"{clean_name}\", 0)"),
                    "i32" => format!("get_i32(params, \"{clean_name}\", 0)"),
                    "bool" => format!("get_bool(params, \"{clean_name}\", false)"),
                    "String" | "&str" => {
                        format!("get_string(params, \"{clean_name}\", \"\")")
                    }
                    "&[f32]" => format!("get_f32_array(params, \"{clean_name}\")"),
                    "&[f64]" => format!("get_f64_array(params, \"{clean_name}\")"),
                    _ => format!("get_f32(params, \"{clean_name}\", 0.0)"),
                }
            };
            param_args.push(getter);
        }

        let args_joined = if param_args.is_empty() {
            String::new()
        } else {
            format!(", {}", param_args.join(", "))
        };

        code.push_str(&format!(
            "        \"{}\" => Ok(Box::new(filters::{node_name}::new(upstream, info{args_joined}))),\n",
            f.name
        ));
    }

    // Mapper dispatch arms — mappers are also dispatched by name
    for m in mappers {
        let node_name = format!("{}MapperNode", to_pascal_case(&m.name));

        let mut param_args = Vec::new();
        for (pname, ptype) in &m.params {
            let clean_name = pname.trim_start_matches('_');
            let getter = if ptype.starts_with('&') && ptype.ends_with("Params") {
                let struct_name = &ptype[1..];
                format!("domain_filters::{struct_name}::default()")
            } else {
                match ptype.as_str() {
                    "f32" => format!("get_f32(params, \"{clean_name}\", 0.0)"),
                    "f64" => format!("get_f32(params, \"{clean_name}\", 0.0) as f64"),
                    "u32" => format!("get_u32(params, \"{clean_name}\", 0)"),
                    "u8" => format!("get_u8(params, \"{clean_name}\", 0)"),
                    "i32" => format!("get_i32(params, \"{clean_name}\", 0)"),
                    "bool" => format!("get_bool(params, \"{clean_name}\", false)"),
                    "String" | "&str" => format!("get_string(params, \"{clean_name}\", \"\")"),
                    _ => format!("get_f32(params, \"{clean_name}\", 0.0)"),
                }
            };
            param_args.push(getter);
        }

        let args_joined = if param_args.is_empty() {
            String::new()
        } else {
            format!(", {}", param_args.join(", "))
        };

        code.push_str(&format!(
            "        \"{}\" => Ok(Box::new(filters::{node_name}::new(upstream, info{args_joined}))),\n",
            m.name
        ));
    }

    code.push_str("        _ => Err(format!(\"Unknown filter: {name}\")),\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");

    // list_filters function for --list-filters
    code.push_str("/// Metadata for a registered filter.\n");
    code.push_str("pub struct FilterMeta {\n");
    code.push_str("    pub name: &'static str,\n");
    code.push_str("    pub category: &'static str,\n");
    code.push_str("    pub params: &'static [(&'static str, &'static str)],\n");
    code.push_str("}\n\n");

    code.push_str("/// List all available filters with their parameter metadata.\n");
    code.push_str("pub fn list_filters() -> Vec<FilterMeta> {\n");
    code.push_str("    vec![\n");

    for f in filters {
        let params_array: Vec<String> = f
            .params
            .iter()
            .map(|(n, t)| {
                let clean = n.trim_start_matches('_');
                format!("(\"{clean}\", \"{}\")", t)
            })
            .collect();
        code.push_str(&format!(
            "        FilterMeta {{ name: \"{}\", category: \"{}\", params: &[{}] }},\n",
            f.name,
            f.category,
            params_array.join(", ")
        ));
    }

    // Include mappers in the filter list (they're dispatched the same way)
    for m in mappers {
        let params_array: Vec<String> = m
            .params
            .iter()
            .map(|(n, t)| {
                let clean = n.trim_start_matches('_');
                format!("(\"{clean}\", \"{}\")", t)
            })
            .collect();
        code.push_str(&format!(
            "        FilterMeta {{ name: \"{}\", category: \"{}\", params: &[{}] }},\n",
            m.name,
            m.category,
            params_array.join(", ")
        ));
    }

    code.push_str("    ]\n");
    code.push_str("}\n");

    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_dispatch_for_simple_filter() {
        let filters = vec![FilterReg {
            name: "blur".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "uniform(5)".to_string(),
            fn_name: "blur".to_string(),
            params: vec![("radius".to_string(), "f32".to_string())],
            config_struct: None,
        }];

        let code = generate(&filters, &[]);
        assert!(code.contains("\"blur\" => Ok(Box::new("));
        assert!(code.contains("BlurNode::new"));
        assert!(code.contains("get_f32(params, \"radius\""));
        assert!(code.contains("pub fn dispatch_filter"));
        assert!(code.contains("pub fn list_filters"));
    }

    #[test]
    fn generates_dispatch_for_mapper() {
        let mappers = vec![crate::types::MapperReg {
            name: "grayscale".to_string(),
            category: "color".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            fn_name: "grayscale_mapper".to_string(),
            params: vec![],
            config_struct: None,
        }];

        let code = generate(&[], &mappers);
        assert!(code.contains("\"grayscale\" => Ok(Box::new("));
        assert!(code.contains("GrayscaleMapperNode::new"));
    }
}
