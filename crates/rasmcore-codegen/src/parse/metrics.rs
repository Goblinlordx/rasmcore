//! Parse `#[register_metric(...)]` annotations from syn AST.

/// A parsed metric registration.
#[derive(Debug, Clone)]
pub struct MetricReg {
    pub name: String,
    pub fn_name: String,
}

/// Extract all metric registrations from a parsed source file.
pub fn extract_metrics(file: &syn::File) -> Vec<MetricReg> {
    let mut metrics = Vec::new();
    for item in &file.items {
        if let syn::Item::Fn(func) = item {
            if let Some(reg) = extract_metric_reg(func) {
                metrics.push(reg);
            }
        }
    }
    metrics
}

fn extract_metric_reg(func: &syn::ItemFn) -> Option<MetricReg> {
    for attr in &func.attrs {
        let path = attr.path();
        let is_register = path.is_ident("register_metric")
            || path
                .segments
                .last()
                .map(|s| s.ident == "register_metric")
                .unwrap_or(false);
        if !is_register {
            continue;
        }

        let tokens = match &attr.meta {
            syn::Meta::List(ml) => ml.tokens.to_string(),
            _ => continue,
        };

        let name = extract_kv(&tokens, "name")?;
        let fn_name = func.sig.ident.to_string();

        return Some(MetricReg { name, fn_name });
    }
    None
}

fn extract_kv(tokens: &str, key: &str) -> Option<String> {
    let pattern = format!("{key} = \"");
    let start = tokens.find(&pattern)?;
    let after = &tokens[start + pattern.len()..];
    let end = after.find('"')?;
    Some(after[..end].to_string())
}
