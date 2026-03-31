//! Generate pipeline node structs and ImageNode impls.

use std::collections::HashMap;

use crate::types::{FilterReg, ParamField};

use super::helpers::{to_binding_type, to_owned_type, to_pascal_case};

/// Generate pipeline node structs + ImageNode impls + pipeline adapter macro.
pub fn generate_nodes(
    filters: &[FilterReg],
    param_structs: &HashMap<String, Vec<ParamField>>,
) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline filter nodes.\n");
    code.push_str("// Do not edit — regenerate by changing filters.rs and rebuilding.\n\n");
    code.push_str("use crate::domain::error::ImageError;\n");
    code.push_str("use crate::domain::filters;\n");
    code.push_str("use crate::domain::filters::*; // ConfigParams structs\n");
    code.push_str("use crate::domain::pipeline::graph::{AccessPattern, ImageNode, bytes_per_pixel, crop_region};\n");
    code.push_str("use crate::domain::types::*;\n");
    code.push_str("use rasmcore_pipeline::Rect;\n\n");

    for f in filters {
        let node_name = format!("{}Node", to_pascal_case(&f.name));
        let domain_fn = &f.fn_name;

        // Struct
        code.push_str(&format!("pub struct {node_name} {{\n"));
        code.push_str("    upstream: u32,\n");
        code.push_str("    source_info: ImageInfo,\n");
        for (pname, ptype) in &f.params {
            let clean_n = pname.trim_start_matches('_');
            code.push_str(&format!("    {clean_n}: {},\n", to_owned_type(ptype)));
        }
        code.push_str("}\n\n");

        // Constructor
        let ctor_params: Vec<String> = f
            .params
            .iter()
            .map(|(n, t)| {
                let clean_n = n.trim_start_matches('_');
                format!("{clean_n}: {}", to_owned_type(t))
            })
            .collect();
        let ctor_sig = if ctor_params.is_empty() {
            "upstream: u32, source_info: ImageInfo".to_string()
        } else {
            format!(
                "upstream: u32, source_info: ImageInfo, {}",
                ctor_params.join(", ")
            )
        };
        let mut all_fields = vec!["upstream".to_string(), "source_info".to_string()];
        all_fields.extend(
            f.params
                .iter()
                .map(|(n, _)| n.trim_start_matches('_').to_string()),
        );

        code.push_str(&format!("impl {node_name} {{\n"));
        code.push_str("    #[allow(clippy::too_many_arguments)]\n");
        code.push_str(&format!("    pub fn new({ctor_sig}) -> Self {{\n"));
        code.push_str(&format!("        Self {{ {} }}\n", all_fields.join(", ")));
        code.push_str("    }\n");
        code.push_str("}\n\n");

        // ImageNode impl — compute input_rect expression
        // Determine expansion from overlap attribute (legacy) or config struct heuristic
        let input_rect_body = input_rect_body(f, param_structs);

        let call_args: Vec<String> = f
            .params
            .iter()
            .map(|(n, t)| {
                let clean_n = n.trim_start_matches('_');
                if t.starts_with("&[") || t == "&str" {
                    format!("&self.{clean_n}")
                } else if t.starts_with('&') {
                    // Reference to a struct (e.g., &SpinBlurParams) — pass as reference
                    format!("&self.{clean_n}")
                } else if t == "String" {
                    format!("self.{clean_n}.clone()")
                } else {
                    format!("self.{clean_n}")
                }
            })
            .collect();
        let full_domain_call = if call_args.is_empty() {
            format!("filters::{domain_fn}(&src_pixels, &region_info)")
        } else {
            format!(
                "filters::{domain_fn}(&src_pixels, &region_info, {})",
                call_args.join(", ")
            )
        };

        code.push_str(&format!("impl ImageNode for {node_name} {{\n"));
        code.push_str("    fn info(&self) -> ImageInfo { self.source_info.clone() }\n\n");
        code.push_str("    fn compute_region(\n");
        code.push_str("        &self,\n");
        code.push_str("        request: Rect,\n");
        code.push_str(
            "        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,\n",
        );
        code.push_str("    ) -> Result<Vec<u8>, ImageError> {\n");
        code.push_str("        let upstream_rect = self.input_rect(request, self.source_info.width, self.source_info.height);\n");
        code.push_str("        let src_pixels = upstream_fn(self.upstream, upstream_rect)?;\n");
        code.push_str("        let region_info = ImageInfo {\n");
        code.push_str("            width: upstream_rect.width,\n");
        code.push_str("            height: upstream_rect.height,\n");
        code.push_str("            ..self.source_info\n");
        code.push_str("        };\n");
        code.push_str(&format!("        let filtered = {full_domain_call}?;\n"));
        code.push_str("        if upstream_rect == request {\n");
        code.push_str("            Ok(filtered)\n");
        code.push_str("        } else {\n");
        code.push_str("            let bpp = bytes_per_pixel(self.source_info.format);\n");
        code.push_str("            let sub = Rect::new(\n");
        code.push_str("                request.x - upstream_rect.x,\n");
        code.push_str("                request.y - upstream_rect.y,\n");
        code.push_str("                request.width,\n");
        code.push_str("                request.height,\n");
        code.push_str("            );\n");
        code.push_str("            let out_rect = Rect::new(0, 0, upstream_rect.width, upstream_rect.height);\n");
        code.push_str("            Ok(crop_region(&filtered, out_rect, sub, bpp))\n");
        code.push_str("        }\n");
        code.push_str("    }\n\n");
        code.push_str(&input_rect_body);
        code.push_str(
            "    fn access_pattern(&self) -> AccessPattern { AccessPattern::LocalNeighborhood }\n",
        );
        code.push_str("}\n\n");
    }

    code
}

/// Determine the `input_rect()` method body for a filter node.
///
/// Priority:
/// 1. Explicit overlap attribute (legacy, non-"zero") — convert to equivalent input_rect
/// 2. Config struct heuristic — detect radius/ksize/sigma/diameter fields
/// 3. Default — zero expansion (point operation)
fn input_rect_body(f: &FilterReg, param_structs: &HashMap<String, Vec<ParamField>>) -> String {
    // Determine the expansion expression
    let expand_expr = input_rect_expand_expr(f, param_structs);
    let Some(expr) = expand_expr else {
        // No expansion needed — use trait default (point operation)
        return String::new();
    };
    format!(
        "    fn input_rect(&self, output: Rect, bounds_w: u32, bounds_h: u32) -> Rect {{ {} }}\n",
        expr
    )
}

/// Determine the expand expression for input_rect, or None for point operations.
fn input_rect_expand_expr(
    f: &FilterReg,
    param_structs: &HashMap<String, Vec<ParamField>>,
) -> Option<String> {
    // Check explicit overlap attribute first (legacy support)
    match f.overlap.as_str() {
        "full" => return Some("Rect::new(0, 0, bounds_w, bounds_h)".to_string()),
        s if s.starts_with("uniform(") => {
            let n = s.trim_start_matches("uniform(").trim_end_matches(')');
            return Some(format!("output.expand_uniform({n}, bounds_w, bounds_h)"));
        }
        s if s.starts_with("param(") => {
            let param_name = s.trim_start_matches("param(").trim_end_matches(')');
            return Some(format!(
                "output.expand_uniform(self.{param_name} as u32, bounds_w, bounds_h)"
            ));
        }
        _ => {} // "zero" or unset — fall through to heuristic
    }

    // Config struct heuristic: detect expansion-related fields from params
    // Only apply to spatial/edge/enhancement categories
    // Categories where neighborhood-based expansion may be needed.
    // The heuristic only fires if expansion-related fields are found,
    // so including a category here won't add expansion to point operations.
    let is_spatial = matches!(
        f.category.as_str(),
        "spatial"
            | "edge"
            | "enhancement"
            | "distortion"
            | "morphology"
            | "effect"
            | "advanced"
            | "tonemapping"
    );
    if !is_spatial {
        return None;
    }

    // First check direct params (non-config-struct filters)
    if let Some(expr) = detect_expansion_field(&f.params, "self") {
        return Some(expr);
    }

    // Then check config struct fields (config-struct-based filters)
    if let Some(struct_name) = &f.config_struct {
        if let Some(fields) = param_structs.get(struct_name) {
            let field_pairs: Vec<(String, String)> = fields
                .iter()
                .map(|pf| (pf.name.clone(), pf.param_type.clone()))
                .collect();
            if let Some(expr) = detect_expansion_field(&field_pairs, "self.config") {
                return Some(expr);
            }
        }
    }

    None
}

/// Detect expansion-related field names in a param list and generate the expand expression.
/// `prefix` is "self" for direct params or "self.config" for config struct fields.
fn detect_expansion_field(params: &[(String, String)], prefix: &str) -> Option<String> {
    for (pname, ptype) in params {
        let clean = pname.trim_start_matches('_');
        // Helper: convert to u32 only if not already u32
        let as_u32 = |field: &str| -> String {
            if ptype == "u32" {
                format!("{prefix}.{field}")
            } else {
                format!("{prefix}.{field} as u32")
            }
        };
        match clean {
            "radius" | "blur_radius" => {
                let expr = as_u32(clean);
                return Some(format!("output.expand_uniform({expr}, bounds_w, bounds_h)"));
            }
            "ksize" => {
                return Some(format!(
                    "output.expand_uniform({prefix}.ksize / 2, bounds_w, bounds_h)"
                ));
            }
            "sigma" => {
                return Some(format!(
                    "output.expand_uniform((3.0 * {prefix}.sigma).ceil() as u32, bounds_w, bounds_h)"
                ));
            }
            "diameter" => {
                return Some(format!(
                    "output.expand_uniform({prefix}.diameter / 2, bounds_w, bounds_h)"
                ));
            }
            "search_size" => {
                return Some(format!(
                    "output.expand_uniform({prefix}.search_size / 2, bounds_w, bounds_h)"
                ));
            }
            "length" if ptype == "u32" || ptype == "f32" => {
                let expr = as_u32("length");
                return Some(format!("output.expand_uniform({expr}, bounds_w, bounds_h)"));
            }
            _ => {}
        }
    }
    None
}

/// Generate the pipeline adapter macro (filter methods on PipelineResource).
pub fn generate_adapter_macro(filters: &[FilterReg]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline filter adapter methods.\n");
    code.push_str("// Do not edit — regenerate by changing filters.rs and rebuilding.\n\n");
    code.push_str("macro_rules! generated_pipeline_filter_methods {\n");
    code.push_str("    () => {\n");

    for f in filters {
        let trait_method = &f.name;
        let node_name = format!("{}Node", to_pascal_case(&f.name));

        let mut sig_params = Vec::new();
        let mut node_ctor_args = Vec::new();
        for (n, t) in &f.params {
            let clean_n = n.trim_start_matches('_');
            sig_params.push(format!("{clean_n}: {}", to_binding_type(t)));
            node_ctor_args.push(clean_n.to_string());
        }

        let full_sig = if sig_params.is_empty() {
            "&self, source: NodeId".to_string()
        } else {
            format!("&self, source: NodeId, {}", sig_params.join(", "))
        };

        let ctor_call = if node_ctor_args.is_empty() {
            format!("filters::{node_name}::new(source, src_info)")
        } else {
            format!(
                "filters::{node_name}::new(source, src_info, {})",
                node_ctor_args.join(", ")
            )
        };

        code.push_str("    #[allow(clippy::too_many_arguments)]\n");
        code.push_str(&format!(
            "    fn {trait_method}({full_sig}) -> Result<NodeId, RasmcoreError> {{\n"
        ));
        code.push_str(
            "        let src_info = self.graph.borrow().node_info(source).map_err(to_wit_error)?;\n",
        );
        code.push_str(&format!("        let node = {ctor_call};\n"));

        // Compute content hash for layer cache: hash(upstream_hash, op_name, params)
        let hash_param_bytes = if node_ctor_args.is_empty() {
            "b\"\"".to_string()
        } else {
            // We can't easily get byte repr of all types, so use the debug string
            format!(
                "format!(\"{}\").as_bytes()",
                node_ctor_args
                    .iter()
                    .map(|n| format!("{{{n}:?}}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        code.push_str("        let upstream_hash = self.graph.borrow().node_hash(source);\n");
        code.push_str(&format!(
            "        let content_hash = rasmcore_pipeline::compute_hash(&upstream_hash, \"{trait_method}\", {hash_param_bytes});\n"
        ));
        code.push_str("        Ok(self.graph.borrow_mut().add_node_with_hash(Box::new(node), content_hash))\n");
        code.push_str("    }\n\n");
    }

    code.push_str("    };\n");
    code.push_str("}\n");
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ParamField;

    fn empty_structs() -> HashMap<String, Vec<ParamField>> {
        HashMap::new()
    }

    fn make_param_field(name: &str, param_type: &str) -> ParamField {
        ParamField {
            name: name.to_string(),
            param_type: param_type.to_string(),
            min: String::new(),
            max: String::new(),
            step: String::new(),
            default_val: String::new(),
            label: String::new(),
            hint: String::new(),
        }
    }

    #[test]
    fn generate_node_struct() {
        let filters = vec![FilterReg {
            name: "blur".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "param(radius)".to_string(),
            fn_name: "blur".to_string(),
            params: vec![("radius".to_string(), "f32".to_string())],
            config_struct: None,
        }];
        let code = generate_nodes(&filters, &empty_structs());
        assert!(code.contains("pub struct BlurNode {"));
        assert!(code.contains("radius: f32,"));
        // Uses input_rect() with expand_uniform instead of overlap()
        assert!(code.contains("self.input_rect(request,"));
        assert!(code.contains("output.expand_uniform(self.radius as u32,"));
    }

    #[test]
    fn generate_pipeline_macro() {
        let filters = vec![FilterReg {
            name: "blur".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "zero".to_string(),
            fn_name: "blur".to_string(),
            params: vec![("radius".to_string(), "f32".to_string())],
            config_struct: None,
        }];
        let code = generate_adapter_macro(&filters);
        assert!(code.contains("fn blur("));
        assert!(code.contains("filters::BlurNode::new(source, src_info, radius)"));
    }

    #[test]
    fn input_rect_heuristic_radius_direct() {
        // Spatial filter with direct radius param → heuristic detects radius
        let f = FilterReg {
            name: "median".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "zero".to_string(),
            fn_name: "median".to_string(),
            params: vec![("radius".to_string(), "u32".to_string())],
            config_struct: None,
        };
        let code = generate_nodes(&[f], &empty_structs());
        // u32 radius — no cast needed
        assert!(code.contains("output.expand_uniform(self.radius,"));
    }

    #[test]
    fn input_rect_heuristic_radius_config_struct() {
        // Spatial filter with config struct containing radius → heuristic via config
        let mut structs = HashMap::new();
        structs.insert(
            "MedianParams".to_string(),
            vec![make_param_field("radius", "u32")],
        );
        let f = FilterReg {
            name: "median".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "zero".to_string(),
            fn_name: "median".to_string(),
            params: vec![("config".to_string(), "&MedianParams".to_string())],
            config_struct: Some("MedianParams".to_string()),
        };
        let code = generate_nodes(&[f], &structs);
        assert!(
            code.contains("output.expand_uniform(self.config.radius,"),
            "Expected config.radius access (no cast for u32), got:\n{code}"
        );
    }

    #[test]
    fn input_rect_heuristic_ksize() {
        let f = FilterReg {
            name: "erode".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "zero".to_string(),
            fn_name: "erode".to_string(),
            params: vec![
                ("ksize".to_string(), "u32".to_string()),
                ("shape".to_string(), "u32".to_string()),
            ],
            config_struct: None,
        };
        let code = generate_nodes(&[f], &empty_structs());
        assert!(code.contains("output.expand_uniform(self.ksize / 2,"));
    }

    #[test]
    fn input_rect_point_op() {
        // Point operation (color category) → no input_rect override
        let f = FilterReg {
            name: "invert".to_string(),
            category: "color".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            overlap: "zero".to_string(),
            fn_name: "invert".to_string(),
            params: vec![],
            config_struct: None,
        };
        let code = generate_nodes(&[f], &empty_structs());
        // Should NOT contain input_rect override — uses trait default
        assert!(!code.contains("fn input_rect"));
    }
}
