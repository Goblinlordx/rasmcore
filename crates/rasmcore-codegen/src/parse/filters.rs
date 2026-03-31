//! Parse `#[register_filter(...)]` annotations from syn AST.

use crate::types::FilterReg;

/// Extract all FilterReg entries from a parsed Rust source file.
pub fn extract_filters(file: &syn::File) -> Vec<FilterReg> {
    let mut filters = Vec::new();
    for item in &file.items {
        if let syn::Item::Fn(func) = item {
            if let Some(reg) = extract_filter_reg(func) {
                filters.push(reg);
            }
        }
    }
    filters
}

fn extract_filter_reg(func: &syn::ItemFn) -> Option<FilterReg> {
    for attr in &func.attrs {
        let path = attr.path();
        let is_register = path.is_ident("register_filter")
            || path
                .segments
                .last()
                .map(|s| s.ident == "register_filter")
                .unwrap_or(false);
        if !is_register {
            continue;
        }

        // Parse attribute arguments from token stream
        let tokens = match &attr.meta {
            syn::Meta::List(ml) => ml.tokens.to_string(),
            _ => continue,
        };

        let name = extract_kv(&tokens, "name")?;
        let category = extract_kv(&tokens, "category")?;
        let group = extract_kv(&tokens, "group").unwrap_or_default();
        let variant = extract_kv(&tokens, "variant").unwrap_or_default();
        let reference = extract_kv(&tokens, "reference").unwrap_or_default();
        let point_op = extract_kv(&tokens, "point_op")
            .map(|v| v == "true")
            .unwrap_or(false);
        let color_op = extract_kv(&tokens, "color_op")
            .map(|v| v == "true")
            .unwrap_or(false);

        let fn_name = func.sig.ident.to_string();
        let rect_request = true;
        let params = extract_fn_params(&func.sig);

        return Some(FilterReg {
            name,
            category,
            group,
            variant,
            reference,
            point_op,
            color_op,
            rect_request,
            fn_name,
            params,
            config_struct: None, // populated later by parse_source_files
        });
    }
    None
}

/// Extract named parameters from function signature, skipping
/// request/upstream/info (rect-request style).
fn extract_fn_params(sig: &syn::Signature) -> Vec<(String, String)> {
    let mut params = Vec::new();
    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            let full = quote::quote!(#pat_type).to_string();
            if full.contains("pixels")
                || full.contains("info")
                || full.contains("ImageInfo")
                || full.contains("self")
                || full.contains("request : Rect")
                || full.contains("request: Rect")
                || full.contains("upstream")
                || full.contains("UpstreamFn")
            {
                continue;
            }
            let param_name = match pat_type.pat.as_ref() {
                syn::Pat::Ident(ident) => ident.ident.to_string(),
                _ => continue,
            };
            let param_type = super::type_to_string(&pat_type.ty);
            params.push((param_name, param_type));
        }
    }
    params
}

/// Extract a `key = "value"` pair from a stringified token stream.
pub(crate) fn extract_kv(tokens: &str, key: &str) -> Option<String> {
    let pattern = format!("{key} = \"");
    if let Some(start) = tokens.find(&pattern) {
        let rest = &tokens[start + pattern.len()..];
        if let Some(end) = rest.find('"') {
            return Some(rest[..end].to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_filter() {
        let source = r#"
            #[register_filter(name = "blur", category = "spatial")]
            pub fn blur(pixels: &[u8], info: &ImageInfo, radius: f32) -> Result<Vec<u8>, ImageError> {
                todo!()
            }
        "#;
        // Wrap in a module to make it a valid file
        let file = syn::parse_file(source).unwrap();
        let filters = extract_filters(&file);
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].name, "blur");
        assert_eq!(filters[0].category, "spatial");
        assert_eq!(filters[0].fn_name, "blur");
        assert_eq!(filters[0].params.len(), 1);
        assert_eq!(filters[0].params[0].0, "radius");
        assert_eq!(filters[0].params[0].1, "f32");
    }

    #[test]
    fn parse_filter_with_all_attrs() {
        let source = r#"
            #[register_filter(name = "zoom_blur", category = "spatial", group = "blur", variant = "zoom", reference = "GEGL algorithm")]
            pub fn zoom_blur(pixels: &[u8], info: &ImageInfo, center_x: f32, center_y: f32, factor: f32) -> Result<Vec<u8>, ImageError> {
                todo!()
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let filters = extract_filters(&file);
        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].group, "blur");
        assert_eq!(filters[0].variant, "zoom");
        assert_eq!(filters[0].reference, "GEGL algorithm");
        assert_eq!(filters[0].params.len(), 3);
    }

    #[test]
    fn parse_filter_no_params() {
        let source = r#"
            #[register_filter(name = "grayscale", category = "color")]
            pub fn grayscale(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
                todo!()
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let filters = extract_filters(&file);
        assert_eq!(filters.len(), 1);
        assert!(filters[0].params.is_empty());
    }

    #[test]
    fn parse_filter_with_string_param() {
        let source = r#"
            #[register_filter(name = "gradient_map", category = "color")]
            pub fn gradient_map(pixels: &[u8], info: &ImageInfo, stops: String) -> Result<Vec<u8>, ImageError> {
                todo!()
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let filters = extract_filters(&file);
        assert_eq!(filters[0].params[0].1, "String");
    }

    #[test]
    fn parse_multiple_filters() {
        let source = r#"
            #[register_filter(name = "a", category = "x")]
            pub fn a(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> { todo!() }

            fn helper() {}

            #[register_filter(name = "b", category = "y")]
            pub fn b(pixels: &[u8], info: &ImageInfo, v: u32) -> Result<Vec<u8>, ImageError> { todo!() }
        "#;
        let file = syn::parse_file(source).unwrap();
        let filters = extract_filters(&file);
        assert_eq!(filters.len(), 2);
        assert_eq!(filters[0].name, "a");
        assert_eq!(filters[1].name, "b");
    }

    #[test]
    fn skips_non_filter_functions() {
        let source = r#"
            pub fn not_a_filter(x: u32) -> u32 { x }
        "#;
        let file = syn::parse_file(source).unwrap();
        let filters = extract_filters(&file);
        assert!(filters.is_empty());
    }
}
