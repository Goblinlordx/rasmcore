//! Parse `#[register_mapper(...)]` annotations with full parameter extraction.

use crate::types::MapperReg;

use super::filters::extract_kv;

/// Extract all MapperReg entries from a parsed Rust source file.
pub fn extract_mappers(file: &syn::File) -> Vec<MapperReg> {
    let mut mappers = Vec::new();
    for item in &file.items {
        if let syn::Item::Fn(func) = item {
            if let Some(reg) = extract_mapper_reg(func) {
                mappers.push(reg);
            }
        }
    }
    mappers
}

fn extract_mapper_reg(func: &syn::ItemFn) -> Option<MapperReg> {
    for attr in &func.attrs {
        let path = attr.path();
        let is_mapper = path.is_ident("register_mapper")
            || path
                .segments
                .last()
                .map(|s| s.ident == "register_mapper")
                .unwrap_or(false);
        if !is_mapper {
            continue;
        }

        let tokens = match &attr.meta {
            syn::Meta::List(ml) => ml.tokens.to_string(),
            _ => continue,
        };

        let name = extract_kv(&tokens, "name")?;
        let category = extract_kv(&tokens, "category").unwrap_or_default();
        let group = extract_kv(&tokens, "group").unwrap_or_default();
        let variant = extract_kv(&tokens, "variant").unwrap_or_default();
        let reference = extract_kv(&tokens, "reference").unwrap_or_default();

        let fn_name = func.sig.ident.to_string();
        let params = extract_mapper_params(&func.sig);
        let output_format = extract_kv(&tokens, "output_format");

        return Some(MapperReg {
            name,
            category,
            group,
            variant,
            reference,
            fn_name,
            params,
            config_struct: None,
            output_format,
        });
    }
    None
}

/// Extract named parameters from mapper function signature, skipping pixels/info.
fn extract_mapper_params(sig: &syn::Signature) -> Vec<(String, String)> {
    let mut params = Vec::new();
    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            let full = quote::quote!(#pat_type).to_string();
            if full.contains("pixels")
                || full.contains("info")
                || full.contains("ImageInfo")
                || full.contains("self")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mapper_with_params() {
        let source = r#"
            #[register_mapper(name = "add_alpha", category = "alpha", group = "alpha", variant = "add", output_format = "Rgba8")]
            pub fn add_alpha(pixels: &[u8], info: &ImageInfo, alpha: u8) -> Result<(Vec<u8>, ImageInfo), ImageError> {
                todo!()
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let mappers = extract_mappers(&file);
        assert_eq!(mappers.len(), 1);
        assert_eq!(mappers[0].name, "add_alpha");
        assert_eq!(mappers[0].fn_name, "add_alpha");
        assert_eq!(mappers[0].params.len(), 1);
        assert_eq!(mappers[0].params[0].0, "alpha");
        assert_eq!(mappers[0].params[0].1, "u8");
    }

    #[test]
    fn parse_mapper_no_params() {
        let source = r#"
            #[register_mapper(name = "grayscale", category = "color")]
            pub fn grayscale_mapper(pixels: &[u8], info: &ImageInfo) -> Result<(Vec<u8>, ImageInfo), ImageError> {
                todo!()
            }
        "#;
        let file = syn::parse_file(source).unwrap();
        let mappers = extract_mappers(&file);
        assert_eq!(mappers.len(), 1);
        assert_eq!(mappers[0].name, "grayscale");
        assert_eq!(mappers[0].fn_name, "grayscale_mapper");
        assert!(mappers[0].params.is_empty());
    }

    #[test]
    fn ignores_non_mapper_functions() {
        let source = r#"
            #[register_filter(name = "blur", category = "spatial")]
            pub fn blur(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> { todo!() }
        "#;
        let file = syn::parse_file(source).unwrap();
        let mappers = extract_mappers(&file);
        assert!(mappers.is_empty());
    }
}
