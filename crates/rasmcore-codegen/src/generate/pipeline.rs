//! Generate pipeline node structs and ImageNode impls.

use std::collections::HashMap;

use crate::types::{FilterReg, ParamField};

use super::helpers::{to_owned_type, to_pascal_case, to_qualified_binding_type};

/// Generate pipeline node structs + ImageNode impls + pipeline adapter macro.
pub fn generate_nodes(filters: &[FilterReg]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline filter nodes.\n");
    code.push_str("// Do not edit — regenerate by changing filters.rs and rebuilding.\n\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::error::ImageError;\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::filters;\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::filters::*;\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::param_types::Point2D;\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::point_ops::LutPointOp;\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::color_lut::ColorLutOp;\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::pipeline::graph::{AccessPattern, ImageNode, crop_region};\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::types::bytes_per_pixel;\n");
    code.push_str("#[allow(unused_imports)] use crate::domain::types::*;\n");
    code.push_str("#[allow(unused_imports)] use rasmcore_pipeline::Rect;\n\n");


    for f in filters {
        let node_name = format!("{}Node", to_pascal_case(&f.name));

        if f.derive_style {
            // ── New-style: derive(Filter) struct with CpuFilter trait ──────
            let config_type = f.config_struct.as_deref().unwrap_or("()");

            // Struct: holds config + upstream + info
            code.push_str(&format!("pub struct {node_name} {{\n"));
            code.push_str("    pub(crate) upstream: u32,\n");
            code.push_str("    pub(crate) source_info: ImageInfo,\n");
            code.push_str(&format!("    pub(crate) config: filters::{config_type},\n"));
            code.push_str("}\n\n");

            // Constructor: takes config struct directly
            code.push_str(&format!("impl {node_name} {{\n"));
            code.push_str(&format!(
                "    pub fn new(upstream: u32, source_info: ImageInfo, config: filters::{config_type}) -> Self {{\n"
            ));
            code.push_str("        Self { upstream, source_info, config }\n");
            code.push_str("    }\n");
            code.push_str("}\n\n");

            // ImageNode impl: delegates to CpuFilter::compute()
            code.push_str("#[allow(clippy::unnecessary_cast, unused_variables)]\n");
            code.push_str(&format!("impl ImageNode for {node_name} {{\n"));
            code.push_str("    fn info(&self) -> ImageInfo { self.source_info.clone() }\n\n");
            code.push_str("    fn compute_region(\n");
            code.push_str("        &self,\n");
            code.push_str("        request: Rect,\n");
            code.push_str(
                "        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,\n",
            );
            code.push_str("    ) -> Result<Vec<u8>, ImageError> {\n");
            // Direct inline dispatch — same pattern as old-style codegen.
            // The closure is consumed immediately by compute(), so borrows are valid.
            code.push_str("        use crate::domain::filter_traits::CpuFilter;\n");
            code.push_str("        let __uid = self.upstream;\n");
            code.push_str("        let __info = self.source_info.clone();\n");
            code.push_str("        let __cfg = &self.config;\n");
            code.push_str("        let mut __up = |rect: Rect| upstream_fn(__uid, rect);\n");
            code.push_str("        __cfg.compute(request, &mut __up, &__info)\n");
            code.push_str("    }\n\n");

            // upstream_id()
            code.push_str("    fn upstream_id(&self) -> Option<u32> { Some(self.upstream) }\n");

            // PointOp: generated only if impl PointOp detected in source
            if f.point_op {
                code.push_str("    fn as_point_op_lut(&self) -> Option<[u8; 256]> {\n");
                code.push_str("        use crate::domain::filter_traits::PointOp;\n");
                code.push_str("        Some(self.config.build_lut())\n");
                code.push_str("    }\n");
            }

            // ColorOp: generated only if impl ColorOp detected in source
            if f.color_op {
                code.push_str("    fn as_color_lut_op(&self) -> Option<crate::domain::color_lut::ColorLut3D> {\n");
                code.push_str("        use crate::domain::filter_traits::ColorOp;\n");
                code.push_str("        Some(self.config.build_clut())\n");
                code.push_str("    }\n");
            }

            code.push_str(
                "    fn access_pattern(&self) -> AccessPattern { AccessPattern::LocalNeighborhood }\n",
            );
            code.push_str("}\n\n");

            // GpuCapable: generated only if impl GpuFilter detected in source
            if f.gpu {
                code.push_str(&format!(
                    "impl rasmcore_pipeline::gpu::GpuCapable for {node_name} {{\n"
                ));
                code.push_str("    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {\n");
                code.push_str("        if self.source_info.format != crate::domain::types::PixelFormat::Rgba8 { return None; }\n");
                code.push_str("        use crate::domain::filter_traits::GpuFilter;\n");
                code.push_str("        self.config.gpu_ops(width, height)\n");
                code.push_str("    }\n");
                code.push_str("    fn gpu_ops_with_format(&self, width: u32, height: u32, buffer_format: rasmcore_pipeline::gpu::BufferFormat) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {\n");
                code.push_str("        if self.source_info.format != crate::domain::types::PixelFormat::Rgba8 { return None; }\n");
                code.push_str("        use crate::domain::filter_traits::GpuFilter;\n");
                code.push_str("        self.config.gpu_ops_with_format(width, height, buffer_format)\n");
                code.push_str("    }\n");
                code.push_str("}\n\n");
            }

        } else {
            // ── Old-style: bare function with #[register_filter] ───────────
            let domain_fn = &f.fn_name;

            // Struct
            code.push_str(&format!("pub struct {node_name} {{\n"));
            code.push_str("    pub(crate) upstream: u32,\n");
            code.push_str("    pub(crate) source_info: ImageInfo,\n");
            for (pname, ptype) in &f.params {
                let clean_n = pname.trim_start_matches('_');
                code.push_str(&format!("    pub(crate) {clean_n}: {},\n", to_owned_type(ptype)));
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

            // Build call args
            let call_args: Vec<String> = f
                .params
                .iter()
                .map(|(n, t)| {
                    let clean_n = n.trim_start_matches('_');
                    if t.starts_with('&') {
                        format!("&self.{clean_n}")
                    } else if t == "String" {
                        format!("self.{clean_n}.clone()")
                    } else {
                        format!("self.{clean_n}")
                    }
                })
                .collect();

            code.push_str("#[allow(clippy::unnecessary_cast, unused_variables)]\n");
            code.push_str(&format!("impl ImageNode for {node_name} {{\n"));
            code.push_str("    fn info(&self) -> ImageInfo { self.source_info.clone() }\n\n");
            code.push_str("    fn compute_region(\n");
            code.push_str("        &self,\n");
            code.push_str("        request: Rect,\n");
            code.push_str(
                "        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,\n",
            );
            code.push_str("    ) -> Result<Vec<u8>, ImageError> {\n");

            let extra_args = if call_args.is_empty() {
                String::new()
            } else {
                format!(", {}", call_args.join(", "))
            };
            code.push_str("        let upstream_id = self.upstream;\n");
            code.push_str("        let mut upstream = |rect: Rect| upstream_fn(upstream_id, rect);\n");
            code.push_str(&format!(
                "        filters::{domain_fn}(request, &mut upstream, &self.source_info{extra_args})\n"
            ));
            code.push_str("    }\n\n");

            code.push_str("    fn upstream_id(&self) -> Option<u32> { Some(self.upstream) }\n");

            if f.point_op {
                if f.config_struct.is_some() {
                    code.push_str("    fn as_point_op_lut(&self) -> Option<[u8; 256]> { Some(self.config.build_point_lut()) }\n");
                } else {
                    if f.name == "invert" {
                        code.push_str("    fn as_point_op_lut(&self) -> Option<[u8; 256]> { Some(crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Invert)) }\n");
                    }
                }
            }

            if f.color_op && f.config_struct.is_some() {
                code.push_str("    fn as_color_lut_op(&self) -> Option<crate::domain::color_lut::ColorLut3D> { Some(self.config.build_clut()) }\n");
            }

            code.push_str(
                "    fn access_pattern(&self) -> AccessPattern { AccessPattern::LocalNeighborhood }\n",
            );
            code.push_str("}\n\n");
        }
    }

    code
}

/// Generate the pipeline adapter macro (filter methods on PipelineResource).
///
/// Pipeline WIT uses the same config records as filters — ConfigParams = WIT record.
/// The macro accepts WIT binding types and converts them to domain types.
pub fn generate_adapter_macro(
    filters: &[FilterReg],
    param_structs: &HashMap<String, Vec<ParamField>>,
) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated pipeline filter adapter methods.\n");
    code.push_str("// Do not edit — regenerate by changing filters.rs and rebuilding.\n\n");
    code.push_str("macro_rules! generated_pipeline_filter_methods {\n");
    code.push_str("    () => {\n");

    for f in filters {
        let trait_method = &f.name;
        let node_name = format!("{}Node", to_pascal_case(&f.name));

        // The WIT pipeline signature mirrors the filter signature:
        // (source, [extra params...], [config: record])
        // The adapter converts WIT binding types → domain types
        let mut sig_params = Vec::new();
        let mut node_ctor_args = Vec::new();
        let mut body_lines = Vec::new();
        let mut hash_args = Vec::new();

        // All params come from a single WIT config record.
        let has_config_param = f.params.iter().any(|(_n, t)| t.starts_with('&') && t.ends_with("Params"));
        // derive(Filter) filters have config_struct but no entries in params vec
        let has_derive_config = f.derive_style && f.config_struct.as_ref()
            .and_then(|name| param_structs.get(name.as_str()))
            .map_or(false, |fields| !fields.is_empty());
        let has_any_params = !f.params.is_empty() || has_derive_config;
        // Config struct field names — extra params with same name are already in the struct
        let config_field_names: std::collections::HashSet<String> = if has_config_param {
            if let Some(config_name) = &f.config_struct {
                param_structs.get(config_name.as_str())
                    .map(|fields| fields.iter().map(|f| f.name.trim_start_matches('_').to_string()).collect())
                    .unwrap_or_default()
            } else {
                std::collections::HashSet::new()
            }
        } else {
            std::collections::HashSet::new()
        };

        if has_any_params {
            let wit_config_type = format!(
                "crate::bindings::exports::rasmcore::image::pipeline::{}Config",
                to_pascal_case(&f.name)
            );
            sig_params.push(format!("wit_config: {wit_config_type}"));
            hash_args.push("wit_config".to_string());
        }

        // derive(Filter) with config: convert WIT config → domain config
        if has_derive_config {
            if let Some(config_name) = &f.config_struct {
                let domain_type = to_qualified_binding_type(config_name);
                if let Some(fields) = param_structs.get(config_name.as_str()) {
                    let field_inits: Vec<String> = fields.iter()
                        .map(|field| {
                            let fname = field.name.trim_start_matches('_');
                            format!("            {fname}: wit_config.{fname}")
                        })
                        .collect();
                    body_lines.push(format!(
                        "        let config = {domain_type} {{\n{}\n        }};",
                        field_inits.join(",\n")
                    ));
                }
                node_ctor_args.push("config".to_string());
            }
        }

        for (n, t) in &f.params {
            let clean_n = n.trim_start_matches('_');

            if t.starts_with('&') && t.ends_with("Params") {
                let struct_name = &t[1..];
                let domain_type = to_qualified_binding_type(struct_name);
                if let Some(fields) = param_structs.get(struct_name) {
                    let field_inits: Vec<String> =
                        fields
                            .iter()
                            .map(|field| {
                                let fname = field.name.trim_start_matches('_');
                                let nested = param_structs.get(&field.param_type);
                                if let Some(nested_fields) = nested {
                                    let nested_type = to_qualified_binding_type(&field.param_type);
                                    let nested_inits: Vec<String> = nested_fields.iter().map(|nf| {
                                let nfname = nf.name.trim_start_matches('_');
                                format!("                {nfname}: wit_config.{fname}.{nfname}")
                            }).collect();
                                    format!(
                                        "            {fname}: {nested_type} {{\n{}\n            }}",
                                        nested_inits.join(",\n")
                                    )
                                } else {
                                    format!("            {fname}: wit_config.{fname}")
                                }
                            })
                            .collect();
                    body_lines.push(format!(
                        "        let config = {domain_type} {{\n{}\n        }};",
                        field_inits.join(",\n")
                    ));
                }
                node_ctor_args.push("config".to_string());
            } else if t == "&[Point2D]" {
                // Extract from config, convert to domain type
                body_lines.push(format!(
                    "        let {clean_n}_domain: Vec<crate::domain::param_types::Point2D> = \
                     wit_config.{clean_n}.iter().map(|p| crate::domain::param_types::Point2D {{ x: p.x, y: p.y }}).collect();"
                ));
                node_ctor_args.push(format!("{clean_n}_domain"));
            } else {
                // Extract extra param from config record
                // Clone if the param is also in the config struct (both read wit_config.field)
                let needs_clone = config_field_names.contains(clean_n);
                if needs_clone {
                    body_lines.push(format!("        let {clean_n} = wit_config.{clean_n}.clone();"));
                } else {
                    body_lines.push(format!("        let {clean_n} = wit_config.{clean_n};"));
                }
                node_ctor_args.push(clean_n.to_string());
            }
        }

        let full_sig = if sig_params.is_empty() {
            "&self, source: NodeId".to_string()
        } else {
            format!("&self, source: NodeId, {}", sig_params.join(", "))
        };

        let ctor_call = if node_ctor_args.is_empty() {
            if f.derive_style {
                // Derive-style filters with no config fields — use Default::default()
                format!("filters::{node_name}::new(source, src_info, Default::default())")
            } else {
                format!("filters::{node_name}::new(source, src_info)")
            }
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

        // Compute hash before conversion (which may move values)
        let hash_param_bytes = if hash_args.is_empty() {
            "b\"\"".to_string()
        } else {
            format!(
                "format!(\"{}\").as_bytes()",
                hash_args
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

        for line in &body_lines {
            code.push_str(line);
            code.push('\n');
        }
        code.push_str(&format!("        let node = {ctor_call};\n"));
        code.push_str(&format!(
            "        Ok(self.graph.borrow_mut().add_node_described(Box::new(node), content_hash, source, crate::domain::pipeline::graph::NodeKind::Filter, \"{trait_method}\"))\n"
        ));
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
            fn_name: "blur".to_string(),
            params: vec![("radius".to_string(), "f32".to_string())],
            config_struct: None,
            point_op: false,
            color_op: false,
            gpu: false,
            derive_style: false,
            rect_request: true,
        }];
        let code = generate_nodes(&filters);
        assert!(code.contains("pub struct BlurNode {"));
        assert!(code.contains("radius: f32,"));
        // Uses rect-request style: delegates to filter function directly
        assert!(code.contains("filters::blur(request, &mut upstream, &self.source_info"));
    }

    #[test]
    fn generate_pipeline_macro() {
        let filters = vec![FilterReg {
            name: "blur".to_string(),
            category: "spatial".to_string(),
            group: String::new(),
            variant: String::new(),
            reference: String::new(),
            fn_name: "blur".to_string(),
            params: vec![("radius".to_string(), "f32".to_string())],
            config_struct: None,
            point_op: false,
            color_op: false,
            gpu: false,
            derive_style: false,
            rect_request: true,
        }];
        let code = generate_adapter_macro(&filters, &HashMap::new());
        assert!(code.contains("fn blur("));
        assert!(code.contains("filters::BlurNode::new(source, src_info, radius)"));
    }
}
