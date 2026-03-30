//! Generate pipeline node structs and ImageNode impls.

use crate::types::FilterReg;

use super::helpers::{to_binding_type, to_owned_type, to_pascal_case};

/// Generate pipeline node structs + ImageNode impls + pipeline adapter macro.
pub fn generate_nodes(filters: &[FilterReg]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline filter nodes.\n");
    code.push_str("// Do not edit — regenerate by changing filters.rs and rebuilding.\n\n");
    code.push_str("use crate::domain::error::ImageError;\n");
    code.push_str("use crate::domain::filters;\n");
    code.push_str("use crate::domain::pipeline::graph::{AccessPattern, ImageNode, bytes_per_pixel, crop_region};\n");
    code.push_str("use crate::domain::types::*;\n");
    code.push_str("use rasmcore_pipeline::{Overlap, Rect};\n\n");

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
        code.push_str(&format!("    pub fn new({ctor_sig}) -> Self {{\n"));
        code.push_str(&format!(
            "        Self {{ {} }}\n",
            all_fields.join(", ")
        ));
        code.push_str("    }\n");
        code.push_str("}\n\n");

        // ImageNode impl
        let overlap_expr = match f.overlap.as_str() {
            "full" => "Overlap::uniform(u32::MAX)".to_string(),
            s if s.starts_with("uniform(") => format!("Overlap::{s}"),
            s if s.starts_with("param(") => {
                let param_name = s.trim_start_matches("param(").trim_end_matches(')');
                format!("Overlap::uniform(self.{param_name} as u32)")
            }
            _ => "Overlap::zero()".to_string(),
        };

        let call_args: Vec<String> = f
            .params
            .iter()
            .map(|(n, t)| {
                let clean_n = n.trim_start_matches('_');
                if t.starts_with("&[") || t == "&str" {
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
        code.push_str("        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,\n");
        code.push_str("    ) -> Result<Vec<u8>, ImageError> {\n");
        code.push_str("        let overlap = self.overlap();\n");
        code.push_str("        let upstream_rect = request.expand(&overlap, self.source_info.width, self.source_info.height);\n");
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
        code.push_str(&format!(
            "    fn overlap(&self) -> Overlap {{ {overlap_expr} }}\n"
        ));
        code.push_str(
            "    fn access_pattern(&self) -> AccessPattern { AccessPattern::LocalNeighborhood }\n",
        );
        code.push_str("}\n\n");
    }

    code
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

        code.push_str(&format!(
            "    fn {trait_method}({full_sig}) -> Result<NodeId, RasmcoreError> {{\n"
        ));
        code.push_str(
            "        let src_info = self.graph.borrow().node_info(source).map_err(to_wit_error)?;\n",
        );
        code.push_str(&format!("        let node = {ctor_call};\n"));
        code.push_str("        Ok(self.graph.borrow_mut().add_node(Box::new(node)))\n");
        code.push_str("    }\n\n");
    }

    code.push_str("    };\n");
    code.push_str("}\n");
    code
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }];
        let code = generate_nodes(&filters);
        assert!(code.contains("pub struct BlurNode {"));
        assert!(code.contains("radius: f32,"));
        assert!(code.contains("Overlap::uniform(self.radius as u32)"));
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
        }];
        let code = generate_adapter_macro(&filters);
        assert!(code.contains("fn blur("));
        assert!(code.contains("filters::BlurNode::new(source, src_info, radius)"));
    }
}
