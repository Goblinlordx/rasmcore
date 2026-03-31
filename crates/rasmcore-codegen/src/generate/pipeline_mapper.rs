//! Generate pipeline node structs for mapper operations.
//!
//! Mapper nodes differ from filter nodes in two key ways:
//! 1. The domain function returns `Result<(Vec<u8>, ImageInfo), ImageError>` —
//!    the output ImageInfo may have a different pixel format than the input.
//! 2. `info()` returns the output format (stored after first compute), not source_info.
//!    Before first compute, we return source_info as a conservative estimate.
//!
//! Mapper nodes always use `Overlap::full()` because format-changing operations
//! cannot be meaningfully tiled (the bpp changes between input and output).

use crate::types::MapperReg;

use super::helpers::{to_binding_type, to_owned_type, to_pascal_case};

/// Generate pipeline node structs + ImageNode impls for mappers.
pub fn generate_mapper_nodes(mappers: &[MapperReg]) -> String {
    let mut code = String::new();
    code.push_str("\n// --- Auto-generated pipeline mapper nodes ---\n");
    code.push_str("// Mapper nodes handle format-changing operations (e.g., RGB8 → Gray8).\n");
    code.push_str("// They always use Overlap::full() because format changes can't be tiled.\n\n");
    code.push_str("use std::cell::RefCell;\n\n");

    for m in mappers {
        let node_name = format!("{}MapperNode", to_pascal_case(&m.name));
        let domain_fn = &m.fn_name;

        // Struct — includes output_info cell for caching the real output format
        code.push_str(&format!("pub struct {node_name} {{\n"));
        code.push_str("    upstream: u32,\n");
        code.push_str("    source_info: ImageInfo,\n");
        code.push_str("    /// Cached output info from the mapper (set after first compute).\n");
        code.push_str("    output_info: RefCell<Option<ImageInfo>>,\n");
        for (pname, ptype) in &m.params {
            let clean_n = pname.trim_start_matches('_');
            code.push_str(&format!("    {clean_n}: {},\n", to_owned_type(ptype)));
        }
        code.push_str("}\n\n");

        // Constructor
        let ctor_params: Vec<String> = m
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

        code.push_str(&format!("impl {node_name} {{\n"));
        code.push_str("    #[allow(clippy::too_many_arguments)]\n");
        code.push_str(&format!("    pub fn new({ctor_sig}) -> Self {{\n"));
        // If output_format is declared, compute output_info at construction time
        if let Some(ref fmt) = m.output_format {
            code.push_str(&format!("        let output_info = ImageInfo {{\n"));
            code.push_str(&format!("            format: PixelFormat::{fmt},\n"));
            code.push_str("            ..source_info\n");
            code.push_str("        };\n");
        }

        code.push_str("        Self {\n");
        code.push_str("            upstream,\n");
        code.push_str("            source_info,\n");
        if m.output_format.is_some() {
            code.push_str("            output_info: RefCell::new(Some(output_info)),\n");
        } else {
            code.push_str("            output_info: RefCell::new(None),\n");
        }
        for (pname, _) in &m.params {
            let clean_n = pname.trim_start_matches('_');
            code.push_str(&format!("            {clean_n},\n"));
        }
        code.push_str("        }\n");
        code.push_str("    }\n");
        code.push_str("}\n\n");

        // Call arguments for the domain function
        let call_args: Vec<String> = m
            .params
            .iter()
            .map(|(n, t)| {
                let clean_n = n.trim_start_matches('_');
                if t.starts_with("&[") || t == "&str" {
                    format!("&self.{clean_n}")
                } else if t.starts_with('&') {
                    format!("&self.{clean_n}")
                } else if t == "String" {
                    format!("self.{clean_n}.clone()")
                } else {
                    format!("self.{clean_n}")
                }
            })
            .collect();
        let full_domain_call = if call_args.is_empty() {
            format!("filters::{domain_fn}(&full_pixels, &full_info)")
        } else {
            format!(
                "filters::{domain_fn}(&full_pixels, &full_info, {})",
                call_args.join(", ")
            )
        };

        // ImageNode impl — mapper nodes use Overlap::full() and handle format changes
        code.push_str(&format!("impl ImageNode for {node_name} {{\n"));

        // info() returns cached output info if available, else source_info
        code.push_str("    fn info(&self) -> ImageInfo {\n");
        code.push_str("        self.output_info.borrow().clone().unwrap_or_else(|| self.source_info.clone())\n");
        code.push_str("    }\n\n");

        // compute_region — requests full upstream, runs mapper, crops result
        code.push_str("    fn compute_region(\n");
        code.push_str("        &self,\n");
        code.push_str("        request: Rect,\n");
        code.push_str(
            "        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,\n",
        );
        code.push_str("    ) -> Result<Vec<u8>, ImageError> {\n");
        code.push_str("        // Request the full upstream image (mappers need all pixels)\n");
        code.push_str("        let full_rect = Rect::new(0, 0, self.source_info.width, self.source_info.height);\n");
        code.push_str("        let full_pixels = upstream_fn(self.upstream, full_rect)?;\n");
        code.push_str("        let full_info = ImageInfo {\n");
        code.push_str("            width: self.source_info.width,\n");
        code.push_str("            height: self.source_info.height,\n");
        code.push_str("            ..self.source_info\n");
        code.push_str("        };\n\n");
        code.push_str(&format!("        let (mapped_pixels, output_info) = {full_domain_call}?;\n\n"));
        code.push_str("        // Cache the output info for info() calls\n");
        code.push_str("        *self.output_info.borrow_mut() = Some(output_info.clone());\n\n");
        code.push_str("        // If requesting the full image, return as-is\n");
        code.push_str("        if request == full_rect {\n");
        code.push_str("            return Ok(mapped_pixels);\n");
        code.push_str("        }\n\n");
        code.push_str("        // Crop to requested region using output bpp (may differ from input)\n");
        code.push_str("        let out_bpp = bytes_per_pixel(output_info.format);\n");
        code.push_str("        let out_rect = Rect::new(0, 0, output_info.width, output_info.height);\n");
        code.push_str("        Ok(crop_region(&mapped_pixels, out_rect, request, out_bpp))\n");
        code.push_str("    }\n\n");

        code.push_str(
            "    fn access_pattern(&self) -> AccessPattern { AccessPattern::GlobalTwoPass }\n",
        );
        code.push_str("}\n\n");
    }

    code
}

/// Generate the pipeline adapter macro methods for mappers.
pub fn generate_mapper_adapter_macro(mappers: &[MapperReg]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline mapper adapter methods.\n");
    code.push_str("// Do not edit — regenerate by changing filters.rs and rebuilding.\n\n");
    code.push_str("macro_rules! generated_pipeline_mapper_methods {\n");
    code.push_str("    () => {\n");

    for m in mappers {
        let trait_method = &m.name;
        let node_name = format!("{}MapperNode", to_pascal_case(&m.name));

        let mut sig_params = Vec::new();
        let mut node_ctor_args = Vec::new();
        for (n, t) in &m.params {
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

        let hash_param_bytes = if node_ctor_args.is_empty() {
            "b\"\"".to_string()
        } else {
            format!(
                "format!(\"{}\").as_bytes()",
                node_ctor_args
                    .iter()
                    .map(|n| format!("{{{n}:?}}"))
                    .collect::<Vec<_>>()
                    .join(",")
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

    #[test]
    fn generate_mapper_node_struct() {
        let mappers = vec![MapperReg {
            name: "grayscale".to_string(),
            category: "color".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            fn_name: "grayscale_mapper".to_string(),
            params: vec![],
            config_struct: None,
            output_format: Some("Gray8".to_string()),
        }];
        let code = generate_mapper_nodes(&mappers);
        assert!(code.contains("pub struct GrayscaleMapperNode {"));
        assert!(code.contains("output_info: RefCell<Option<ImageInfo>>"));
        assert!(code.contains("fn info(&self) -> ImageInfo"));
        assert!(code.contains("*self.output_info.borrow_mut() = Some(output_info.clone())"));
        assert!(code.contains("Overlap::uniform(u32::MAX)"));
        assert!(code.contains("AccessPattern::GlobalTwoPass"));
    }

    #[test]
    fn generate_mapper_node_with_params() {
        let mappers = vec![MapperReg {
            name: "add_alpha".to_string(),
            category: "alpha".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            fn_name: "add_alpha".to_string(),
            params: vec![("alpha".to_string(), "u8".to_string())],
            config_struct: None,
            output_format: Some("Rgba8".to_string()),
        }];
        let code = generate_mapper_nodes(&mappers);
        assert!(code.contains("pub struct AddAlphaMapperNode {"));
        assert!(code.contains("alpha: u8,"));
        assert!(code.contains("filters::add_alpha(&full_pixels, &full_info, self.alpha)"));
    }

    #[test]
    fn generate_mapper_adapter_macro_test() {
        let mappers = vec![MapperReg {
            name: "grayscale".to_string(),
            category: "color".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            fn_name: "grayscale_mapper".to_string(),
            params: vec![],
            config_struct: None,
            output_format: Some("Gray8".to_string()),
        }];
        let code = generate_mapper_adapter_macro(&mappers);
        assert!(code.contains("fn grayscale("));
        assert!(code.contains("GrayscaleMapperNode::new(source, src_info)"));
    }
}
