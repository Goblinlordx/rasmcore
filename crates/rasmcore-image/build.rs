//! Build script — uses rasmcore-codegen to parse #[register_filter] annotations
//! and generate adapter code, pipeline nodes, WIT declarations, and param-manifest.json.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    // Collect all filter source files (directory structure or single file)
    let filters_dir = Path::new(&manifest_dir).join("src/domain/filters");
    let filters_flat = Path::new(&manifest_dir).join("src/domain/filters.rs");
    let mut filter_paths: Vec<PathBuf> = Vec::new();
    if filters_dir.is_dir() {
        fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_dir() {
                        collect_rs(&p, out);
                    } else if p.extension().is_some_and(|e| e == "rs")
                        && p.file_name().is_some_and(|n| n != "common.rs")
                    {
                        out.push(p);
                    }
                }
            }
        }
        collect_rs(&filters_dir, &mut filter_paths);
        println!("cargo:rerun-if-changed=src/domain/filters");
    } else if filters_flat.exists() {
        filter_paths.push(filters_flat);
        println!("cargo:rerun-if-changed=src/domain/filters.rs");
    }
    let filters_path = filter_paths.first().cloned().unwrap_or_else(|| {
        Path::new(&manifest_dir).join("src/domain/filters.rs")
    });

    let param_types_path = Path::new(&manifest_dir).join("src/domain/param_types.rs");
    let composite_path = Path::new(&manifest_dir).join("src/domain/composite.rs");
    let encoder_dir = Path::new(&manifest_dir).join("src/domain/encoder");
    println!("cargo:rerun-if-changed=src/domain/param_types.rs");
    println!("cargo:rerun-if-changed=src/domain/composite.rs");
    println!("cargo:rerun-if-changed=src/domain/encoder");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    if !filters_path.exists() {
        // Write empty generated files to prevent include! errors
        std::fs::write(
            out_dir.join("generated_filter_adapter.rs"),
            "// No filters\n",
        )
        .unwrap();
        std::fs::write(
            out_dir.join("generated_pipeline_nodes.rs"),
            "// No filters\n",
        )
        .unwrap();
        std::fs::write(
            out_dir.join("generated_pipeline_adapter.rs"),
            "macro_rules! generated_pipeline_filter_methods { () => {} }\n",
        )
        .unwrap();
        let empty = r#"{"filters":[],"generators":[],"compositors":[],"mappers":[]}"#;
        std::fs::write(out_dir.join("param-manifest.json"), empty).unwrap();
        std::fs::write(out_dir.join("param-manifest.hash"), "0000000000000000").unwrap();
        return;
    }

    // Parse all source files via syn AST
    let pt = if param_types_path.exists() {
        Some(param_types_path.as_path())
    } else {
        None
    };
    let cp = if composite_path.exists() {
        Some(composite_path.as_path())
    } else {
        None
    };
    let mut data = rasmcore_codegen::parse::parse_source_files(&filters_path, pt, cp);
    // Parse additional filter files from subdirectories
    for path in &filter_paths {
        if *path == filters_path {
            continue;
        }
        let extra = rasmcore_codegen::parse::parse_source_files(path, None, None);
        data.filters.extend(extra.filters);
        data.generators.extend(extra.generators);
        data.compositors.extend(extra.compositors);
        data.mappers.extend(extra.mappers);
        data.param_structs.extend(extra.param_structs);
    }
    // Re-run auto-linking after merging all files (ConfigParams may be in mod.rs
    // while registrations are in submodule files)
    for filter in &mut data.filters {
        if filter.config_struct.is_none() {
            let expected = format!(
                "{}Params",
                rasmcore_codegen::generate::helpers::to_pascal_case(&filter.name)
            );
            if data.param_structs.contains_key(&expected) {
                filter.config_struct = Some(expected);
            }
        }
    }
    for mapper in &mut data.mappers {
        if mapper.config_struct.is_none() {
            let expected = format!(
                "{}Params",
                rasmcore_codegen::generate::helpers::to_pascal_case(&mapper.name)
            );
            if data.param_structs.contains_key(&expected) {
                mapper.config_struct = Some(expected);
            }
        }
    }

    // Detect GPU-capable nodes from filter `gpu = "true"` attributes
    for f in &data.filters {
        if f.gpu {
            let pascal = rasmcore_codegen::generate::helpers::to_pascal_case(&f.name);
            data.gpu_capable_nodes.insert(format!("{pascal}Node"));
        }
    }

    // Also scan gpu_impls.rs for hand-written GpuCapable impls (transforms, fused nodes)
    // that aren't registered filters (e.g., ComposedAffineNode, FusedClutNode, SkeletonizeParams)
    let gpu_impls_path = Path::new(&manifest_dir).join("src/domain/pipeline/nodes/gpu_impls.rs");
    println!("cargo:rerun-if-changed=src/domain/pipeline/nodes/gpu_impls.rs");
    if gpu_impls_path.exists() {
        let gpu_src = fs::read_to_string(&gpu_impls_path).unwrap_or_default();
        for line in gpu_src.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("impl GpuCapable for ") {
                let token = rest.split_whitespace().next().unwrap_or("");
                let node_name = token.strip_suffix('{').unwrap_or(token);
                if !node_name.is_empty() {
                    data.gpu_capable_nodes.insert(node_name.to_string());
                }
            }
        }
    }
    eprintln!("rasmcore build.rs: {} GPU-capable node(s) detected", data.gpu_capable_nodes.len());

    // Duplicate filter name detection — fail at compile time
    {
        let mut seen: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        for f in &data.filters {
            if let Some(existing_fn) = seen.get(f.name.as_str()) {
                panic!(
                    "\n\nDuplicate filter name '{}' registered by:\n  fn {}\n  fn {}\n\
                     Rename one to resolve the conflict.\n",
                    f.name, f.fn_name, existing_fn
                );
            }
            seen.insert(&f.name, &f.fn_name);
        }
    }

    // Print summary
    eprintln!(
        "rasmcore build.rs: {} filter(s), {} generator(s), {} compositor(s), {} mapper(s)",
        data.filters.len(),
        data.generators.len(),
        data.compositors.len(),
        data.mappers.len()
    );

    // Generate all output files
    rasmcore_codegen::generate::generate_all(&data, &out_dir);

    // ── Parse encoder configs for pipeline write method generation ──
    let mut all_encoder_configs = Vec::new(); // ALL encoders (for dispatch generation)
    let mut encoder_configs = Vec::new(); // Only those with sink functions (for pipeline adapter)
    if encoder_dir.exists() {
        let sink_source = std::fs::read_to_string(
            Path::new(&manifest_dir).join("src/domain/pipeline/nodes/sink.rs"),
        )
        .unwrap_or_default();

        for entry in std::fs::read_dir(&encoder_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().map(|e| e == "rs").unwrap_or(false) {
                let source = std::fs::read_to_string(&path).unwrap();
                if let Ok(file) = syn::parse_file(&source) {
                    let mut configs =
                        rasmcore_codegen::parse::encoders::extract_encoder_configs(&file);
                    for config in &mut configs {
                        // Check if this format has a sink function
                        let sig_pattern = format!("fn write_{}", config.format);
                        if let Some(pos) = sink_source.find(&sig_pattern) {
                            let sig_end = sink_source[pos..].find('{').unwrap_or(200);
                            let sig = &sink_source[pos..pos + sig_end];
                            config.sink_takes_metadata = sig.contains("metadata");
                        }
                        // Extract encode function name from source
                        for item in &file.items {
                            if let syn::Item::Fn(f) = item {
                                let name = f.sig.ident.to_string();
                                if name.starts_with("encode") && f.vis == syn::Visibility::Public(syn::token::Pub::default()) {
                                    config.encode_fn = name;
                                    break;
                                }
                            }
                        }
                    }
                    // All configs go to dispatch generation
                    all_encoder_configs.extend(configs.clone());
                    // Only sink-having configs go to pipeline adapter
                    configs.retain(|c| sink_source.contains(&format!("fn write_{}", c.format)));
                    encoder_configs.extend(configs);
                }
            }
        }
    }

    // Generate pipeline write adapter code
    if !encoder_configs.is_empty() {
        let write_adapter =
            rasmcore_codegen::generate::pipeline_write::generate_adapter_methods(&encoder_configs);
        std::fs::write(
            out_dir.join("generated_pipeline_write_adapter.rs"),
            &write_adapter,
        )
        .unwrap();

        // Also generate stateless encoder adapter
        let stateless_adapter =
            rasmcore_codegen::generate::pipeline_write::generate_stateless_encoder_adapter(
                &encoder_configs,
            );
        std::fs::write(
            out_dir.join("generated_encoder_adapter.rs"),
            &stateless_adapter,
        )
        .unwrap();

        // Generate encode dispatch macro from ALL encoders (not just sink-having ones)
        let encode_dispatch =
            rasmcore_codegen::generate::pipeline_write::generate_encode_dispatch(&all_encoder_configs);
        std::fs::write(
            out_dir.join("generated_encode_dispatch.rs"),
            &encode_dispatch,
        )
        .unwrap();

        eprintln!(
            "rasmcore build.rs: Generated {} pipeline write + stateless encoder adapter method(s)",
            encoder_configs.len()
        );
    } else {
        std::fs::write(
            out_dir.join("generated_pipeline_write_adapter.rs"),
            "macro_rules! generated_pipeline_write_methods { () => {} }\n",
        )
        .unwrap();
        std::fs::write(
            out_dir.join("generated_encoder_adapter.rs"),
            "macro_rules! generated_encoder_methods { () => {} }\n",
        )
        .unwrap();
        std::fs::write(
            out_dir.join("generated_encode_dispatch.rs"),
            "macro_rules! generated_encode_dispatch { ($p:expr, $i:expr, $f:expr, $q:expr) => { Err(crate::domain::error::ImageError::UnsupportedFormat(\"no encoders\".into())) }; }\n",
        )
        .unwrap();
    }

    // ── Parse transform nodes for pipeline transform adapter generation ──
    let transform_dir = Path::new(&manifest_dir).join("src/domain/pipeline/nodes");
    let types_path = Path::new(&manifest_dir).join("src/domain/types.rs");
    let metadata_path = Path::new(&manifest_dir).join("src/domain/metadata/exif.rs");

    println!("cargo:rerun-if-changed=src/domain/pipeline/nodes");
    println!("cargo:rerun-if-changed=src/domain/types.rs");
    println!("cargo:rerun-if-changed=src/domain/metadata/exif.rs");

    // Collect all enum definitions from types.rs, metadata.rs, filters/*.rs
    let filters_common_path = Path::new(&manifest_dir).join("src/domain/filters/common.rs");
    let mut all_enums = std::collections::HashMap::new();
    for (enum_src_path, domain_mod) in [
        (&types_path, "types"),
        (&metadata_path, "metadata"),
        (&filters_path, "filters"),
        (&filters_common_path, "filters"),
    ] {
        if enum_src_path.exists() {
            let source = std::fs::read_to_string(enum_src_path).unwrap_or_default();
            if let Ok(file) = syn::parse_file(&source) {
                all_enums.extend(
                    rasmcore_codegen::parse::transforms::extract_enums(&file, domain_mod),
                );
            }
        }
    }

    // Parse transform registrations from node source files
    // Each file maps to a node module import (e.g., "transform.rs" → "transform")
    let mut all_transforms = Vec::new();
    let transform_files = [
        ("transform.rs", "transform"),
        ("color.rs", "color"),
        ("composite.rs", "composite"),
    ];
    for (filename, module_name) in &transform_files {
        let path = transform_dir.join(filename);
        // Collect source from single file or all files in directory module
        let source = if path.exists() {
            std::fs::read_to_string(&path).unwrap_or_default()
        } else {
            let dir_path = transform_dir.join(filename.trim_end_matches(".rs"));
            if !dir_path.is_dir() {
                continue;
            }
            // Concatenate all .rs files in the directory (skip mod.rs — it has
            // module declarations that break syn::parse_file when concatenated)
            let mut combined = String::new();
            if let Ok(entries) = std::fs::read_dir(&dir_path) {
                for entry in entries.flatten() {
                    let fname = entry.file_name();
                    if entry.path().extension().is_some_and(|e| e == "rs")
                        && fname != "mod.rs"
                    {
                        combined.push_str(&std::fs::read_to_string(entry.path()).unwrap_or_default());
                        combined.push('\n');
                    }
                }
            }
            combined
        };
        if let Ok(file) = syn::parse_file(&source) {
            let transforms = rasmcore_codegen::parse::transforms::extract_transforms(
                &file,
                &all_enums,
                module_name,
            );
            all_transforms.extend(transforms);
        }
    }

    if !all_transforms.is_empty() {
        eprintln!(
            "rasmcore build.rs: Found {} transform(s):",
            all_transforms.len()
        );
        for t in &all_transforms {
            let params: Vec<String> = t.params.iter().map(|(n, ty)| format!("{n}: {ty}")).collect();
            eprintln!(
                "  {} ({}) params=[{}] fallible={} multi_input={}",
                t.name,
                t.node_type,
                params.join(", "),
                t.fallible,
                t.multi_input,
            );
        }

        let transform_adapter = rasmcore_codegen::generate::transform::generate_adapter_macro(
            &all_transforms,
            &all_enums,
        );
        // Patch: composite blend_mode is u32 in WIT but Option<BlendMode> in Rust.
        // Convert 0 → None, non-zero → Some(BlendMode from index).
        let transform_adapter = transform_adapter.replace(
            "let blend_mode = config.blend_mode;",
            "let blend_mode = crate::domain::filters::BlendMode::from_u32(config.blend_mode);",
        );
        std::fs::write(
            out_dir.join("generated_pipeline_transform_adapter.rs"),
            &transform_adapter,
        )
        .unwrap();

        eprintln!(
            "rasmcore build.rs: Generated {} pipeline transform adapter method(s)",
            all_transforms.iter().filter(|t| !t.multi_input).count()
        );
    } else {
        std::fs::write(
            out_dir.join("generated_pipeline_transform_adapter.rs"),
            "macro_rules! generated_pipeline_transform_methods { () => {} }\n",
        )
        .unwrap();
    }

    // ── Regenerate param-manifest.json with transforms + encoders ──
    // The initial generate_all() call produced the manifest without transforms/encoders
    // because they're parsed after filters. Now that we have all data, regenerate.
    data.transforms = all_transforms.clone();
    data.encoders = all_encoder_configs.clone();
    {
        let manifest_json = rasmcore_codegen::generate::manifest::generate(&data);
        std::fs::write(out_dir.join("param-manifest.json"), &manifest_json).unwrap();
        // Recompute hash
        let hash = {
            let mut h: u64 = 0xcbf29ce484222325;
            for &byte in manifest_json.as_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        std::fs::write(out_dir.join("param-manifest.hash"), format!("{hash:016x}")).unwrap();
        eprintln!(
            "rasmcore build.rs: Regenerated manifest with {} transform(s) + {} encoder(s)",
            data.transforms.len(),
            data.encoders.len(),
        );

        // Regenerate CLI dispatch with transforms included
        let cli_dispatch = rasmcore_codegen::generate::cli_dispatch::generate(
            &data.filters, &data.mappers, &data.transforms, &data.gpu_capable_nodes,
        );
        std::fs::write(out_dir.join("generated_cli_dispatch.rs"), &cli_dispatch).unwrap();
        eprintln!(
            "rasmcore build.rs: Regenerated CLI dispatch with {} transform(s)",
            data.transforms.len(),
        );
    }

    // ── Generate WIT from template ──
    // Replaces deprecated generate-wit.cjs — build.rs is the single source of truth.
    {
        let wit_dir = Path::new(&manifest_dir).join("../../wit/image");
        let tmpl_path = wit_dir.join("pipeline.wit.tmpl");
        println!("cargo:rerun-if-changed=../../wit/image/pipeline.wit.tmpl");

        // Generate filters.wit from template (replaces generate-wit.cjs)
        let filters_tmpl_path = wit_dir.join("filters.wit.tmpl");
        println!("cargo:rerun-if-changed=../../wit/image/filters.wit.tmpl");
        if filters_tmpl_path.exists() {
            let filters_tmpl = fs::read_to_string(&filters_tmpl_path).unwrap();
            // Filters only — mappers have different output semantics and aren't
            // exposed in the stateless filters interface (pipeline-only via adapter macros)
            let filters_wit_content =
                rasmcore_codegen::generate::wit::generate(&data.filters, &data.param_structs);
            let filters_wit = filters_tmpl.replace("{{GENERATED_FILTERS}}", &filters_wit_content);
            fs::write(wit_dir.join("filters.wit"), &filters_wit).unwrap();

            // Extract all record names from generated filters.wit for the pipeline import line
            let filter_record_names: Vec<String> = filters_wit_content
                .lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    if trimmed.starts_with("record ") {
                        let name = trimmed
                            .strip_prefix("record ")?
                            .split_whitespace()
                            .next()?;
                        Some(name.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            let filter_import_line = if filter_record_names.is_empty() {
                String::new()
            } else {
                format!(
                    "    use filters.{{{}}};",
                    filter_record_names.join(", ")
                )
            };

            eprintln!(
                "rasmcore build.rs: Generated filters.wit ({} filters + {} mappers, {} records)",
                data.filters.len(),
                data.mappers.len(),
                filter_record_names.len()
            );

            // Generate pipeline.wit
            if tmpl_path.exists() {
                let tmpl = fs::read_to_string(&tmpl_path).unwrap();

            // Generate transform WIT — split into types (enums + records) and methods
            let mut transform_types = String::new(); // goes before resource (interface level)
            let mut transform_methods = String::new(); // goes inside resource
            if !all_transforms.is_empty() {
                // Collect which enums are used by transforms.
                // Exclude enums already defined in other WIT interfaces (imported).
                let externally_defined = ["ExifOrientation"]; // defined in metadata.wit
                let used_enums: Vec<String> = {
                    let mut used = Vec::new();
                    for t in &all_transforms {
                        for (_, ptype) in &t.params {
                            if all_enums.contains_key(ptype.as_str())
                                && !used.contains(ptype)
                                && !externally_defined.contains(&ptype.as_str())
                            {
                                used.push(ptype.clone());
                            }
                            if let Some(inner) = ptype.strip_prefix("Option<").and_then(|s| s.strip_suffix('>'))
                                && all_enums.contains_key(inner)
                                && !used.contains(&inner.to_string())
                                && !externally_defined.contains(&inner)
                            {
                                used.push(inner.to_string());
                            }
                        }
                    }
                    used
                };

                transform_types.push_str(&rasmcore_codegen::generate::transform::generate_wit_enums(
                    &all_enums,
                    &used_enums,
                ));
                transform_types.push_str(&rasmcore_codegen::generate::transform::generate_wit_configs(
                    &all_transforms,
                    &all_enums,
                ));
                transform_methods.push_str(&rasmcore_codegen::generate::transform::generate_wit_methods(
                    &all_transforms,
                ));
            }

            // Generate pipeline filter methods — all use (source, config) or just (source) for zero-param
            let mut filter_methods = String::new();
            for f in &data.filters {
                let wit_name = f.name.replace('_', "-");
                if f.params.is_empty() {
                    filter_methods.push_str(&format!(
                        "        {wit_name}: func(source: node-id) -> result<node-id, rasmcore-error>;\n"
                    ));
                } else {
                    filter_methods.push_str(&format!(
                        "        {wit_name}: func(source: node-id, config: {wit_name}-config) -> result<node-id, rasmcore-error>;\n"
                    ));
                }
            }

            // Fill template
            let pipeline_wit = tmpl
                .replace("{{GENERATED_FILTER_IMPORTS}}", &filter_import_line)
                .replace("{{GENERATED_TRANSFORM_TYPES}}", &transform_types)
                .replace("{{GENERATED_PIPELINE_TRANSFORMS}}", &transform_methods)
                .replace("{{GENERATED_PIPELINE_FILTERS}}", &filter_methods);

            fs::write(wit_dir.join("pipeline.wit"), &pipeline_wit).unwrap();
            eprintln!("rasmcore build.rs: Generated pipeline.wit from template");
            }
        }
    }

    // ── Parse metrics for compare adapter generation ──
    let metrics_path = Path::new(&manifest_dir).join("src/domain/metrics.rs");
    println!("cargo:rerun-if-changed=src/domain/metrics.rs");

    if metrics_path.exists() {
        let source = std::fs::read_to_string(&metrics_path).unwrap_or_default();
        if let Ok(file) = syn::parse_file(&source) {
            let metrics = rasmcore_codegen::parse::metrics::extract_metrics(&file);
            if !metrics.is_empty() {
                eprintln!(
                    "rasmcore build.rs: Found {} metric(s): {}",
                    metrics.len(),
                    metrics.iter().map(|m| m.name.as_str()).collect::<Vec<_>>().join(", ")
                );
                let compare_adapter =
                    rasmcore_codegen::generate::metrics::generate_compare_adapter(&metrics);
                std::fs::write(
                    out_dir.join("generated_compare_adapter.rs"),
                    &compare_adapter,
                )
                .unwrap();
            } else {
                std::fs::write(
                    out_dir.join("generated_compare_adapter.rs"),
                    "// No registered metrics found\n",
                )
                .unwrap();
            }
        }
    } else {
        std::fs::write(
            out_dir.join("generated_compare_adapter.rs"),
            "// No metrics source file\n",
        )
        .unwrap();
    }

    // ─── WGSL shader validation via naga ─────────────────────────────────────
    // Compose shared fragments from rasmcore-gpu-shaders before validating,
    // since shader body files no longer contain pack/unpack/sample_bilinear.
    let gpu_shaders_dir = Path::new(&manifest_dir).join("../rasmcore-gpu-shaders/src/wgsl");
    let pixel_ops = fs::read_to_string(gpu_shaders_dir.join("pixel_ops.wgsl")).unwrap_or_default();
    let sample_bilinear_frag = fs::read_to_string(gpu_shaders_dir.join("sample_bilinear.wgsl")).unwrap_or_default();

    let shaders_dir = Path::new(&manifest_dir).join("src/shaders");
    if shaders_dir.is_dir() {
        println!("cargo:rerun-if-changed=src/shaders");
        println!("cargo:rerun-if-changed=../rasmcore-gpu-shaders/src/wgsl");
        let mut shader_count = 0u32;
        for entry in fs::read_dir(&shaders_dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "wgsl") {
                let body = fs::read_to_string(&path).unwrap_or_else(|e| {
                    panic!("Failed to read shader {}: {e}", path.display())
                });
                // Compose fragments needed by this shader body
                let needs_sample = body.contains("sample_bilinear(")
                    && !body.contains("fn sample_bilinear");
                let composed = if needs_sample {
                    format!("{pixel_ops}\n{sample_bilinear_frag}\n{body}")
                } else {
                    format!("{pixel_ops}\n{body}")
                };
                if let Err(e) = naga::front::wgsl::parse_str(&composed) {
                    panic!(
                        "WGSL validation failed for {}:\n{e}",
                        path.display()
                    );
                }
                shader_count += 1;
            }
        }
        if shader_count > 0 {
            println!(
                "cargo:warning=Validated {shader_count} WGSL shader(s) in src/shaders/"
            );
        }
    }
}
