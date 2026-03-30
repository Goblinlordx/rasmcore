//! Parse simple registrations: register_generator, register_compositor, register_mapper.

use crate::types::SimpleReg;

use super::filters::extract_kv;

/// Extract all SimpleReg entries for a given registration kind from a parsed file.
pub fn extract_by_kind(file: &syn::File, kind: &str) -> Vec<SimpleReg> {
    let mut regs = Vec::new();
    for item in &file.items {
        if let syn::Item::Fn(func) = item {
            for attr in &func.attrs {
                let is_kind = attr
                    .path()
                    .segments
                    .last()
                    .map(|s| s.ident == kind)
                    .unwrap_or(false);
                if !is_kind {
                    continue;
                }
                if let syn::Meta::List(meta_list) = &attr.meta {
                    let tokens = meta_list.tokens.to_string();
                    if let Some(name) = extract_kv(&tokens, "name") {
                        regs.push(SimpleReg {
                            name,
                            category: extract_kv(&tokens, "category").unwrap_or_default(),
                            group: extract_kv(&tokens, "group").unwrap_or_default(),
                            variant: extract_kv(&tokens, "variant").unwrap_or_default(),
                            reference: extract_kv(&tokens, "reference").unwrap_or_default(),
                        });
                    }
                }
            }
        }
    }
    regs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_generator() {
        let source = r#"
            #[register_generator(name = "perlin_noise", category = "noise")]
            pub fn perlin_noise(width: u32, height: u32) -> Vec<u8> { vec![] }
        "#;
        let file = syn::parse_file(source).unwrap();
        let regs = extract_by_kind(&file, "register_generator");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].name, "perlin_noise");
        assert_eq!(regs[0].category, "noise");
    }

    #[test]
    fn parse_compositor() {
        let source = r#"
            #[register_compositor(name = "blend", category = "composite", group = "blend", variant = "normal")]
            pub fn blend(a: &[u8], b: &[u8]) -> Vec<u8> { vec![] }
        "#;
        let file = syn::parse_file(source).unwrap();
        let regs = extract_by_kind(&file, "register_compositor");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].name, "blend");
        assert_eq!(regs[0].group, "blend");
    }

    #[test]
    fn parse_mapper() {
        let source = r#"
            #[register_mapper(name = "remap", category = "transform")]
            pub fn remap(pixels: &[u8]) -> Vec<u8> { vec![] }
        "#;
        let file = syn::parse_file(source).unwrap();
        let regs = extract_by_kind(&file, "register_mapper");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].name, "remap");
    }

    #[test]
    fn ignores_other_kinds() {
        let source = r#"
            #[register_filter(name = "blur", category = "spatial")]
            pub fn blur(pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> { todo!() }
        "#;
        let file = syn::parse_file(source).unwrap();
        let regs = extract_by_kind(&file, "register_generator");
        assert!(regs.is_empty());
    }
}
