//! Generate compare::Guest adapter methods for registered metrics.
//!
//! All metrics have the uniform signature:
//! `fn(a: Vec<u8>, info_a: ImageInfo, b: Vec<u8>, info_b: ImageInfo) -> Result<f64, Error>`

use crate::parse::metrics::MetricReg;

/// Generate the compare::Guest impl block for all registered metrics.
pub fn generate_compare_adapter(metrics: &[MetricReg]) -> String {
    let mut code = String::new();
    code.push_str("// Auto-generated compare adapter methods.\n");
    code.push_str("// Do not edit — regenerate by changing metrics.rs and rebuilding.\n\n");
    code.push_str("impl compare::Guest for Component {\n");

    for m in metrics {
        let fn_name = &m.fn_name;
        let method_name = &m.name;

        code.push_str(&format!("    fn {method_name}(\n"));
        code.push_str("        a: Vec<u8>,\n");
        code.push_str("        info_a: types::ImageInfo,\n");
        code.push_str("        b: Vec<u8>,\n");
        code.push_str("        info_b: types::ImageInfo,\n");
        code.push_str("    ) -> Result<f64, RasmcoreError> {\n");
        code.push_str("        let ia = to_domain_image_info(&info_a);\n");
        code.push_str("        let ib = to_domain_image_info(&info_b);\n");
        code.push_str(&format!(
            "        domain::metrics::{fn_name}(&a, &ia, &b, &ib).map_err(to_wit_error)\n"
        ));
        code.push_str("    }\n\n");
    }

    code.push_str("}\n");
    code
}
