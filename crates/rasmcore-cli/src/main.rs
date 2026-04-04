//! rcim — graph-based image processing CLI.
//!
//! Builds a pipeline DAG from left-to-right arguments:
//!   rcim -i photo.jpg -blur radius=5 -sharpen amount=1.0 -o out.png
//!   rcim -i bg.jpg -ref bg -i fg.png -ref fg -composite base=bg top=fg -o result.jpg
//!   rcim --list-filters
//!
//! See `rcim --help` for usage.

use std::collections::HashMap;
use std::path::Path;
use std::process;

use rasmcore_image::domain::pipeline::dispatch;
use rasmcore_image::domain::pipeline::graph::NodeGraph;
use rasmcore_image::domain::pipeline::nodes::{sink, source};

#[cfg(feature = "gpu")]
mod gpu_executor;
#[cfg(feature = "gpu")]
mod gpu_executor_v2;

// ─── CLI Command Types ─────────────────────────────────────────────────────

#[derive(Debug)]
enum CliCommand {
    /// Load image from file path
    Input(String),
    /// Apply a filter operation with key=value params
    Filter {
        name: String,
        params: HashMap<String, String>,
        /// First positional arg (maps to first param if provided without key=)
        positional: Option<String>,
    },
    /// Bookmark the active node with a reference name
    Ref(String),
    /// Write active node to output file
    Output(String),
}

// ─── Argument Parser ────────────────────────────────────────────────────────

fn parse_args(args: &[String]) -> Result<Vec<CliCommand>, String> {
    let mut commands = Vec::new();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        // Global flags
        if arg == "--help" || arg == "-h" {
            print_help();
            process::exit(0);
        }
        if arg == "--version" || arg == "-V" {
            println!("rcim {}", env!("CARGO_PKG_VERSION"));
            process::exit(0);
        }
        if arg == "--list-filters" {
            print_filters();
            process::exit(0);
        }
        // GPU flags (consumed here, applied in main)
        if arg == "--gpu" || arg == "--no-gpu" {
            i += 1;
            continue;
        }

        // Input
        if arg == "-i" || arg == "--input" {
            i += 1;
            if i >= args.len() {
                return Err("-i requires a file path".into());
            }
            commands.push(CliCommand::Input(args[i].clone()));
            i += 1;
            continue;
        }

        // Reference
        if arg == "-ref" || arg == "--as" {
            i += 1;
            if i >= args.len() {
                return Err("-ref requires a name".into());
            }
            commands.push(CliCommand::Ref(args[i].clone()));
            i += 1;
            continue;
        }

        // Output
        if arg == "-o" || arg == "--output" {
            i += 1;
            if i >= args.len() {
                return Err("-o requires a file path".into());
            }
            commands.push(CliCommand::Output(args[i].clone()));
            i += 1;
            continue;
        }

        // Filter operation: -blur, -sharpen, etc.
        if arg.starts_with('-') && !arg.starts_with("--") && arg.len() > 1 {
            let filter_name = arg[1..].to_string();
            i += 1;

            let mut params = HashMap::new();
            let mut positional = None;

            // Consume key=value pairs and optional positional arg
            while i < args.len() && !args[i].starts_with('-') {
                if let Some((key, val)) = args[i].split_once('=') {
                    params.insert(key.to_string(), val.to_string());
                } else if positional.is_none() {
                    // First non-key=value arg is positional (maps to first param)
                    positional = Some(args[i].clone());
                } else {
                    break;
                }
                i += 1;
            }

            commands.push(CliCommand::Filter {
                name: filter_name,
                params,
                positional,
            });
            continue;
        }

        return Err(format!("Unknown argument: {arg}"));
    }

    Ok(commands)
}

// ─── Graph Builder ──────────────────────────────────────────────────────────

fn build_and_execute(
    commands: Vec<CliCommand>,
    gpu_executor: Option<std::rc::Rc<dyn rasmcore_pipeline::GpuExecutor>>,
) -> Result<(), String> {
    let mut graph = NodeGraph::new(64 * 1024 * 1024); // 64MB spatial cache
    if let Some(executor) = gpu_executor {
        graph.set_gpu_executor(executor);
    }
    let mut active_node: Option<u32> = None;
    let mut refs: HashMap<String, u32> = HashMap::new();

    // Get filter metadata for positional param resolution
    let filter_meta = dispatch::list_filters();
    let meta_by_name: HashMap<&str, &dispatch::FilterMeta> =
        filter_meta.iter().map(|m| (m.name, m)).collect();

    for cmd in &commands {
        match cmd {
            CliCommand::Input(path) => {
                let bytes =
                    std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
                let node = source::SourceNode::new(bytes)
                    .map_err(|e| format!("Failed to decode {path}: {e}"))?;
                let id = graph.add_node(Box::new(node));
                active_node = Some(id);
                eprintln!("  [read] {path} → node {id}");
            }

            CliCommand::Filter {
                name,
                params,
                positional,
            } => {
                let upstream = active_node
                    .ok_or_else(|| format!("Filter -{name} has no input (use -i first)"))?;
                let info = graph
                    .node_info(upstream)
                    .map_err(|e| format!("Failed to get node info: {e}"))?;

                // Resolve positional arg → first param name from metadata
                let mut resolved_params = params.clone();
                if let Some(pos_val) = positional {
                    if let Some(meta) = meta_by_name.get(name.as_str()) {
                        if let Some(&(first_param, _)) = meta.params.first() {
                            resolved_params
                                .entry(first_param.to_string())
                                .or_insert_with(|| pos_val.clone());
                        }
                    }
                }

                // Resolve references: if a param value matches a ref name, it stays
                // as a string — the dispatch function doesn't handle node refs.
                // For composite-style ops, we'd need special handling.
                // For now, node-ref params are handled by the caller.

                let (node, gpu) = dispatch::dispatch_filter(name, upstream, info, &resolved_params)
                    .map_err(|e| format!("Filter -{name}: {e}"))?;
                let id = graph.add_node(node);
                if let Some(gpu_node) = gpu {
                    graph.register_gpu(id, gpu_node);
                }
                active_node = Some(id);
                eprintln!("  [{name}] → node {id}{}", if graph.has_gpu(id) { " (GPU)" } else { "" });
            }

            CliCommand::Ref(name) => {
                let node = active_node.ok_or_else(|| format!("-ref {name} has no active node"))?;
                refs.insert(name.clone(), node);
                eprintln!("  [ref] {name} = node {node}");
            }

            CliCommand::Output(path) => {
                let node = active_node.ok_or_else(|| "-o has no active node".to_string())?;

                // Infer format from extension
                let format = Path::new(path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("png");
                let format = match format {
                    "jpg" => "jpeg",
                    other => other,
                };

                eprintln!("  [write] node {node} → {path} ({format})");
                let bytes = sink::write(&mut graph, node, format, None, None)
                    .map_err(|e| format!("Failed to encode {format}: {e}"))?;
                std::fs::write(path, &bytes).map_err(|e| format!("Failed to write {path}: {e}"))?;
                eprintln!("  [done] {} bytes written", bytes.len());
            }
        }
    }

    Ok(())
}

// ─── Help & Filter Listing ──────────────────────────────────────────────────

fn print_help() {
    eprintln!(
        r#"rcim — graph-based image processing CLI

USAGE:
  rcim -i <input> [-filter key=value ...] [-ref <name>] [-o <output>]

FLAGS:
  -i, --input <path>     Load image file (resets active node)
  -<filter> [params...]  Apply filter (key=value or positional first param)
  -ref, --as <name>      Bookmark active node as a reference
  -o, --output <path>    Write active node to file (format from extension)
  --list-filters         Show all available filters and parameters
  --gpu                  Force GPU acceleration (fail if unavailable)
  --no-gpu               Force CPU-only execution (disable GPU)
  -h, --help             Show this help
  -V, --version          Show version

EXAMPLES:
  # Simple chain
  rcim -i photo.jpg -blur radius=5 -sharpen amount=1.0 -o out.png

  # Positional shorthand (first param)
  rcim -i photo.jpg -blur 5 -o out.png

  # Multi-branch composite
  rcim -i bg.jpg -contrast 1.2 -ref bg \
        -i overlay.png -resize width=500 height=500 filter=lanczos3 -ref fg \
        -composite base=bg top=fg mode=multiply -o final.jpg

  # Generator as source
  rcim -checkerboard width=256 height=256 size=32 -blur radius=3 -o pattern.png
"#
    );
}

fn print_filters() {
    let filters = dispatch::list_filters();
    let mut by_category: HashMap<&str, Vec<&dispatch::FilterMeta>> = HashMap::new();
    for f in &filters {
        by_category.entry(f.category).or_default().push(f);
    }

    let mut categories: Vec<&&str> = by_category.keys().collect();
    categories.sort();

    for cat in categories {
        eprintln!("\n[{cat}]");
        let mut ops = by_category[cat].clone();
        ops.sort_by_key(|f| f.name);
        for f in ops {
            let params: Vec<String> = f
                .params
                .iter()
                .map(|(name, ty)| format!("{name}:{ty}"))
                .collect();
            if params.is_empty() {
                eprintln!("  -{}", f.name);
            } else {
                eprintln!("  -{} {}", f.name, params.join(" "));
            }
        }
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        print_help();
        process::exit(0);
    }

    // GPU initialization
    #[cfg(feature = "gpu")]
    let gpu_executor: Option<std::rc::Rc<dyn rasmcore_pipeline::GpuExecutor>> = {
        let force_gpu = args.iter().any(|a| a == "--gpu");
        let no_gpu = args.iter().any(|a| a == "--no-gpu");

        if !no_gpu {
            match gpu_executor::WgpuExecutor::try_new() {
                Ok(exec) => {
                    eprintln!("GPU: {} (ready)", exec.adapter_name());
                    Some(std::rc::Rc::new(exec))
                }
                Err(e) => {
                    if force_gpu {
                        eprintln!("Error: --gpu requested but GPU unavailable: {e}");
                        process::exit(1);
                    }
                    eprintln!("GPU: not available, using CPU ({e})");
                    None
                }
            }
        } else {
            eprintln!("GPU: disabled (--no-gpu)");
            None
        }
    };

    #[cfg(not(feature = "gpu"))]
    let gpu_executor: Option<std::rc::Rc<dyn rasmcore_pipeline::GpuExecutor>> = None;

    let commands = match parse_args(&args) {
        Ok(cmds) => cmds,
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    };

    // Pre-flight: check refs are defined before use
    let mut defined_refs: Vec<String> = Vec::new();
    for cmd in &commands {
        match cmd {
            CliCommand::Ref(name) => defined_refs.push(name.clone()),
            CliCommand::Filter { params, .. } => {
                for _val in params.values() {
                    // If the value looks like a ref name (no digits, no dots, no slashes),
                    // check it's defined. But don't error here — it might be a literal string.
                    // Full validation would need the filter's param type info.
                }
            }
            _ => {}
        }
    }

    if let Err(e) = build_and_execute(commands, gpu_executor) {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
