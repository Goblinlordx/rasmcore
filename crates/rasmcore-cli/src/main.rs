//! rcim — graph-based image processing CLI (V2 pipeline).
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

use rasmcore_pipeline_v2::graph::Graph;
use rasmcore_pipeline_v2::node::NodeInfo;
use rasmcore_pipeline_v2::registry::{
    create_filter_node, decode_via_registry, encode_via_registry,
    registered_filter_registrations, ParamMap,
};

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

// ─── V2 Source Node ────────────────────────────────────────────────────────

/// Source node that holds decoded f32 pixel data for the V2 graph.
struct SourceNode {
    pixels: Vec<f32>,
    info: NodeInfo,
}

impl rasmcore_pipeline_v2::node::Node for SourceNode {
    fn info(&self) -> NodeInfo {
        self.info.clone()
    }

    fn compute(
        &self,
        request: rasmcore_pipeline_v2::Rect,
        _upstream: &mut dyn rasmcore_pipeline_v2::node::Upstream,
    ) -> Result<Vec<f32>, rasmcore_pipeline_v2::node::PipelineError> {
        let w = self.info.width as usize;
        let rw = request.width as usize;
        let rh = request.height as usize;
        let rx = request.x as usize;
        let ry = request.y as usize;

        if rx == 0 && ry == 0 && rw == w && rh == self.info.height as usize {
            return Ok(self.pixels.clone());
        }

        let mut out = Vec::with_capacity(rw * rh * 4);
        for row in 0..rh {
            let src = ((ry + row) * w + rx) * 4;
            out.extend_from_slice(&self.pixels[src..src + rw * 4]);
        }
        Ok(out)
    }

    fn upstream_ids(&self) -> Vec<u32> {
        vec![]
    }
}

// ─── Param Conversion ──────────────────────────────────────────────────────

/// Convert CLI string params (key=value) to V2 ParamMap with typed values.
fn to_param_map(params: &HashMap<String, String>) -> ParamMap {
    let mut pm = ParamMap::new();
    for (key, val) in params {
        // Try f32 first
        if let Ok(f) = val.parse::<f32>() {
            pm.floats.insert(key.clone(), f);
        } else if val == "true" || val == "false" {
            pm.bools.insert(key.clone(), val == "true");
        } else {
            pm.strings.insert(key.clone(), val.clone());
        }
    }
    pm
}

// ─── Graph Builder ──────────────────────────────────────────────────────────

fn build_and_execute(
    commands: Vec<CliCommand>,
    #[cfg(feature = "gpu")] gpu_executor: Option<std::rc::Rc<dyn rasmcore_pipeline_v2::gpu::GpuExecutor>>,
) -> Result<(), String> {
    let mut graph = Graph::new(64 * 1024 * 1024); // 64MB spatial cache

    #[cfg(feature = "gpu")]
    if let Some(executor) = gpu_executor {
        graph.set_gpu_executor(executor);
    }

    let mut active_node: Option<u32> = None;
    let mut refs: HashMap<String, u32> = HashMap::new();

    // Get filter metadata for positional param resolution
    let filter_regs = registered_filter_registrations();
    let meta_by_name: HashMap<&str, _> = filter_regs.iter().map(|r| (r.name, *r)).collect();

    for cmd in &commands {
        match cmd {
            CliCommand::Input(path) => {
                let bytes =
                    std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
                let decoded = decode_via_registry(&bytes)
                    .ok_or_else(|| format!("Unsupported image format: {path}"))?
                    .map_err(|e| format!("Failed to decode {path}: {e}"))?;

                let info = NodeInfo {
                    width: decoded.width,
                    height: decoded.height,
                    color_space: decoded.color_space,
                };
                let id = graph.add_node(Box::new(SourceNode {
                    pixels: decoded.pixels,
                    info,
                }));
                active_node = Some(id);
                eprintln!("  [read] {path} → node {id} ({}x{})", decoded.width, decoded.height);
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
                        if let Some(first_param) = meta.params.first() {
                            resolved_params
                                .entry(first_param.name.to_string())
                                .or_insert_with(|| pos_val.clone());
                        }
                    }
                }

                let pm = to_param_map(&resolved_params);
                let node = create_filter_node(name, upstream, info, &pm)
                    .ok_or_else(|| format!("Unknown filter: {name}"))?;
                let id = graph.add_node(node);
                active_node = Some(id);
                eprintln!("  [{name}] → node {id}");
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

                // Execute the graph to get f32 pixels
                let pixels = graph
                    .request_full(node)
                    .map_err(|e| format!("Pipeline execution failed: {e}"))?;
                let info = graph
                    .node_info(node)
                    .map_err(|e| format!("Failed to get node info: {e}"))?;

                // Encode via V2 codec registry
                let bytes = encode_via_registry(format, &pixels, info.width, info.height, &ParamMap::new())
                    .ok_or_else(|| format!("Unsupported output format: {format}"))?
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
    let regs = registered_filter_registrations();
    let mut by_category: HashMap<&str, Vec<_>> = HashMap::new();
    for r in &regs {
        let cat = if r.category.is_empty() { "uncategorized" } else { r.category };
        by_category.entry(cat).or_default().push(*r);
    }

    let mut categories: Vec<&&str> = by_category.keys().collect();
    categories.sort();

    for cat in categories {
        eprintln!("\n[{cat}]");
        let mut ops = by_category[cat].clone();
        ops.sort_by_key(|r| r.name);
        for r in ops {
            let params: Vec<String> = r
                .params
                .iter()
                .map(|p| format!("{}:{:?}", p.name, p.value_type))
                .collect();
            if params.is_empty() {
                eprintln!("  -{}", r.name);
            } else {
                eprintln!("  -{} {}", r.name, params.join(" "));
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
    let gpu_executor: Option<std::rc::Rc<dyn rasmcore_pipeline_v2::gpu::GpuExecutor>> = {
        let force_gpu = args.iter().any(|a| a == "--gpu");
        let no_gpu = args.iter().any(|a| a == "--no-gpu");

        if !no_gpu {
            match gpu_executor_v2::WgpuExecutorV2::try_new() {
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

    let commands = match parse_args(&args) {
        Ok(cmds) => cmds,
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    };

    #[cfg(feature = "gpu")]
    if let Err(e) = build_and_execute(commands, gpu_executor) {
        eprintln!("Error: {e}");
        process::exit(1);
    }

    #[cfg(not(feature = "gpu"))]
    if let Err(e) = build_and_execute(commands) {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
