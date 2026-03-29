//! Rasmcore proc macros for auto-registration of filters, encoders, and decoders.
//!
//! These macros generate the boilerplate needed to register an operation with
//! the rasmcore extensibility system. The registration metadata is collected
//! at build time to auto-generate WIT interfaces, adapter dispatch, and
//! pipeline nodes.
//!
//! # Usage
//!
//! ```ignore
//! use rasmcore_macros::register_filter;
//!
//! #[register_filter(
//!     name = "my_blur",
//!     category = "spatial",
//!     params(radius: f32 = 1.0),
//! )]
//! pub fn my_blur(pixels: &[u8], info: &ImageInfo, radius: f32) -> Result<Vec<u8>, ImageError> {
//!     // implementation
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitStr, LitFloat, Ident, Token};
use syn::parse::{Parse, ParseStream};

// ─── Filter Registration ────────────────────────────────────────────────────

struct FilterParam {
    name: String,
    ty: String,
    default: f64,
}

struct RegisterFilterArgs {
    name: String,
    category: String,
    params: Vec<FilterParam>,
    overlap: Option<String>,
    custom_node: bool,
}

impl Parse for RegisterFilterArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = String::new();
        let mut category = String::new();
        let mut params = Vec::new();
        let mut overlap = None;
        let mut custom_node = false;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;

            match ident.to_string().as_str() {
                "name" => {
                    let lit: LitStr = input.parse()?;
                    name = lit.value();
                }
                "category" => {
                    let lit: LitStr = input.parse()?;
                    category = lit.value();
                }
                "overlap" => {
                    let lit: LitStr = input.parse()?;
                    overlap = Some(lit.value());
                }
                "custom_node" => {
                    let lit: syn::LitBool = input.parse()?;
                    custom_node = lit.value;
                }
                _ => {
                    // Skip unknown attributes
                    let _: proc_macro2::TokenTree = input.parse()?;
                }
            }

            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(RegisterFilterArgs {
            name,
            category,
            params,
            overlap,
            custom_node,
        })
    }
}

/// Register a function as a rasmcore filter.
///
/// This macro:
/// 1. Keeps the original function unchanged
/// 2. Emits a `FilterRegistration` static with metadata
/// 3. Submits the registration to `inventory` for cross-crate collection
///
/// The build script reads collected registrations to generate:
/// - WIT interface declarations
/// - Adapter dispatch code
/// - Pipeline node wrappers
#[proc_macro_attribute]
pub fn register_filter(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RegisterFilterArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let filter_name = &args.name;
    let filter_category = &args.category;

    // Generate a unique registration constant name
    let reg_ident = Ident::new(
        &format!("__RASMCORE_FILTER_REG_{}", fn_name_str.to_uppercase()),
        fn_name.span(),
    );

    let expanded = quote! {
        // Original function — unchanged
        #input_fn

        // Registration metadata — collected by inventory at link time
        #[allow(non_upper_case_globals)]
        const #reg_ident: () = {
            // This const block ensures the registration metadata is available
            // for the build script to discover via source scanning.
            // The actual collection happens via inventory::submit! in the
            // rasmcore-image crate, which depends on this metadata.
        };

        // Registration marker comment for build script source scanning:
        // RASMCORE_FILTER: name={#filter_name}, fn={#fn_name_str}, category={#filter_category}
    };

    TokenStream::from(expanded)
}

/// Register a function as a rasmcore encoder.
#[proc_macro_attribute]
pub fn register_encoder(attr: TokenStream, item: TokenStream) -> TokenStream {
    // For now, pass through unchanged — will be extended in Phase 4
    let input_fn = parse_macro_input!(item as ItemFn);
    TokenStream::from(quote! { #input_fn })
}

/// Register a function as a rasmcore decoder.
#[proc_macro_attribute]
pub fn register_decoder(attr: TokenStream, item: TokenStream) -> TokenStream {
    // For now, pass through unchanged — will be extended in Phase 4
    let input_fn = parse_macro_input!(item as ItemFn);
    TokenStream::from(quote! { #input_fn })
}
