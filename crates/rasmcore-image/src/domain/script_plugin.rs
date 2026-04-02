//! Rhai-based script plugin system.
//!
//! Loads `.rhai` script files as pipeline filter nodes. Scripts declare their
//! execution strategy (fusable, GPU, compute, composite) and provide
//! implementations. The engine validates, compiles, and wraps them as
//! standard `ImageNode` pipeline nodes.
//!
//! Scripts are passed as source strings at pipeline construction time
//! (WASM components cannot access the host filesystem).

use std::collections::HashMap;
use std::sync::Arc;

use rhai::{AST, Dynamic, Engine, Map, Scope};

use crate::domain::error::ImageError;
use crate::domain::pipeline::graph::{AccessPattern, ImageNode};
use crate::domain::types::*;
use rasmcore_pipeline::Rect;

// ─── Script Metadata ───────────────────────────────────────────────────────

/// Execution strategy declared by a script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptStrategy {
    /// Per-channel LUT, participates in LUT fusion.
    Fusable,
    /// Provides WGSL shader + CPU fallback.
    Gpu,
    /// Eager per-tile compute with builtin() calls.
    Compute,
    /// Returns a lazy node graph via .apply() chains.
    Composite,
}

/// A single config parameter declared in script metadata.
#[derive(Debug, Clone)]
pub struct ScriptParam {
    pub name: String,
    pub param_type: String, // "f32", "u32", "bool"
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// Parsed metadata from script header comments.
#[derive(Debug, Clone)]
pub struct ScriptMetadata {
    pub name: String,
    pub category: String,
    pub strategy: ScriptStrategy,
    pub params: Vec<ScriptParam>,
}

/// A config value in a composite graph — either a scalar or a node reference.
#[derive(Debug, Clone)]
pub enum ConfigValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    NodeRef(u32), // references another node by ID
}

/// A node reference in a composite graph (returned by .apply() chains).
#[derive(Debug, Clone)]
pub struct NodeRef {
    pub id: u32,
    pub filter_name: String,
    pub config: HashMap<String, ConfigValue>,
    pub upstream_id: u32,
}

/// A compiled and validated script ready for use as a pipeline node.
#[derive(Clone)]
pub struct CompiledScript {
    pub metadata: ScriptMetadata,
    pub ast: AST,
    pub source: String,
}

// ─── Metadata Parsing ──────────────────────────────────────────────────────

/// Parse script metadata from header comments.
///
/// Format:
/// ```text
/// //! name: my_filter
/// //! category: effect
/// //! strategy: compute
/// //! param radius: f32 = 5.0 [0.1, 50.0]
/// ```
pub fn parse_metadata(source: &str) -> Result<ScriptMetadata, String> {
    let mut name = None;
    let mut category = None;
    let mut strategy = None;
    let mut params = Vec::new();

    for line in source.lines() {
        let line = line.trim();
        if !line.starts_with("//!") {
            if !line.is_empty() && !line.starts_with("//") {
                break; // End of header
            }
            continue;
        }
        let content = line.trim_start_matches("//!").trim();

        if let Some(val) = content.strip_prefix("name:") {
            name = Some(val.trim().to_string());
        } else if let Some(val) = content.strip_prefix("category:") {
            category = Some(val.trim().to_string());
        } else if let Some(val) = content.strip_prefix("strategy:") {
            strategy = Some(match val.trim() {
                "fusable" => ScriptStrategy::Fusable,
                "gpu" => ScriptStrategy::Gpu,
                "compute" => ScriptStrategy::Compute,
                "composite" => ScriptStrategy::Composite,
                other => return Err(format!("unknown strategy: {other}")),
            });
        } else if let Some(val) = content.strip_prefix("param ") {
            params.push(parse_param_line(val)?);
        }
    }

    Ok(ScriptMetadata {
        name: name.ok_or("missing //! name:")?,
        category: category.unwrap_or_else(|| "custom".to_string()),
        strategy: strategy.ok_or("missing //! strategy:")?,
        params,
    })
}

fn parse_param_line(line: &str) -> Result<ScriptParam, String> {
    // Format: "radius: f32 = 5.0 [0.1, 50.0]"
    let (name_type, rest) = line.split_once('=').ok_or("param missing '='")?;
    let (name, ptype) = name_type.split_once(':').ok_or("param missing ':'")?;
    let name = name.trim().to_string();
    let ptype = ptype.trim().to_string();

    let rest = rest.trim();
    let (default_str, range) = if let Some(bracket_pos) = rest.find('[') {
        (&rest[..bracket_pos], Some(&rest[bracket_pos..]))
    } else {
        (rest, None)
    };

    let default: f64 = default_str
        .trim()
        .parse()
        .map_err(|_| format!("invalid default for param {name}"))?;

    let (min, max) = if let Some(range) = range {
        let range = range.trim_start_matches('[').trim_end_matches(']');
        let parts: Vec<&str> = range.split(',').collect();
        if parts.len() == 2 {
            (
                parts[0].trim().parse().ok(),
                parts[1].trim().parse().ok(),
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Ok(ScriptParam {
        name,
        param_type: ptype,
        default,
        min,
        max,
    })
}

// ─── Script Registry ───────────────────────────────────────────────────────

/// Registry of loaded script plugins, keyed by filter name.
pub struct ScriptRegistry {
    engine: Engine,
    scripts: HashMap<String, CompiledScript>,
}

impl ScriptRegistry {
    /// Create a new registry and compile/validate all provided script sources.
    pub fn new(sources: &[String]) -> Result<Self, String> {
        let engine = create_engine();
        let mut scripts = HashMap::new();

        for source in sources {
            let metadata = parse_metadata(source)?;
            let ast = engine
                .compile(source)
                .map_err(|e| format!("script '{}' compile error: {e}", metadata.name))?;

            // Validate strategy matches provided functions
            validate_strategy(&engine, &ast, &metadata)?;

            let name = metadata.name.clone();
            scripts.insert(
                name.clone(),
                CompiledScript {
                    metadata,
                    ast,
                    source: source.clone(),
                },
            );
        }

        Ok(Self { engine, scripts })
    }

    /// Look up a compiled script by filter name.
    pub fn get(&self, name: &str) -> Option<&CompiledScript> {
        self.scripts.get(name)
    }

    /// List all registered script filter names.
    pub fn list(&self) -> Vec<&str> {
        self.scripts.keys().map(|s| s.as_str()).collect()
    }

    /// Get the Rhai engine (shared across all scripts).
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Check if a filter name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.scripts.contains_key(name)
    }
}

// ─── Engine Setup ──────────────────────────────────────────────────────────

fn create_engine() -> Engine {
    let mut engine = Engine::new();

    // Sandboxing: set execution limits
    engine.set_max_operations(1_000_000); // prevent infinite loops
    engine.set_max_call_levels(32); // prevent stack overflow

    // Register NodeRef type for composite graphs
    engine
        .register_type_with_name::<NodeRef>("NodeRef")
        .register_fn("apply", node_ref_apply);

    // Register helper functions
    engine.register_fn("clamp_value", |v: f64, min: f64, max: f64| -> f64 {
        v.clamp(min, max)
    });
    engine.register_fn("pack_u32", |v: i64| -> Vec<u8> {
        (v as u32).to_le_bytes().to_vec()
    });
    engine.register_fn("pack_f32", |v: f64| -> Vec<u8> {
        (v as f32).to_le_bytes().to_vec()
    });

    // Pixel buffer helpers for compute/GPU scripts.
    // Rhai's Blob type (Vec<u8>) is used directly — scripts index into it.
    // get_pixel/set_pixel operate on Blob at byte offsets.
    engine.register_fn("blob_len", |b: rhai::Blob| -> i64 { b.len() as i64 });

    engine
}

/// NodeRef.apply("filter_name", #{ param: value }) -> NodeRef
///
/// Config values can be scalars (f64, i64, bool, string) or NodeRef values.
/// When a NodeRef is passed as a config value (e.g., `source: other_node`),
/// the engine wires it as an additional upstream connection in the graph.
fn node_ref_apply(node: &mut NodeRef, name: &str, config: Map) -> NodeRef {
    static NEXT_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let mut cfg = HashMap::new();
    for (k, v) in &config {
        let cv = if let Ok(f) = v.as_float() {
            ConfigValue::Float(f)
        } else if let Ok(i) = v.as_int() {
            ConfigValue::Int(i)
        } else if let Ok(b) = v.as_bool() {
            ConfigValue::Bool(b)
        } else if let Ok(s) = v.clone().into_string() {
            ConfigValue::Str(s.to_string())
        } else if let Some(nr) = v.clone().try_cast::<NodeRef>() {
            ConfigValue::NodeRef(nr.id)
        } else {
            continue;
        };
        cfg.insert(k.to_string(), cv);
    }

    NodeRef {
        id,
        filter_name: name.to_string(),
        config: cfg,
        upstream_id: node.id,
    }
}

// ─── Validation ────────────────────────────────────────────────────────────

fn validate_strategy(engine: &Engine, ast: &AST, metadata: &ScriptMetadata) -> Result<(), String> {
    let has_fn = |name: &str| -> bool {
        ast.iter_functions().any(|f| f.name == name)
    };

    match metadata.strategy {
        ScriptStrategy::Fusable => {
            if !has_fn("lut") {
                return Err(format!(
                    "script '{}': fusable strategy requires fn lut(input, channel, params)",
                    metadata.name
                ));
            }
        }
        ScriptStrategy::Gpu => {
            if !has_fn("gpu_shader") {
                return Err(format!(
                    "script '{}': gpu strategy requires fn gpu_shader()",
                    metadata.name
                ));
            }
            if !has_fn("compute") {
                return Err(format!(
                    "script '{}': gpu strategy requires fn compute() as CPU fallback",
                    metadata.name
                ));
            }
        }
        ScriptStrategy::Compute => {
            if !has_fn("compute") {
                return Err(format!(
                    "script '{}': compute strategy requires fn compute(pixels, info, params)",
                    metadata.name
                ));
            }
        }
        ScriptStrategy::Composite => {
            if !has_fn("graph") {
                return Err(format!(
                    "script '{}': composite strategy requires fn graph(input, params)",
                    metadata.name
                ));
            }
        }
    }

    Ok(())
}

// ─── ScriptNode (ImageNode wrapper) ────────────────────────────────────────

/// A pipeline node backed by a compiled Rhai script.
pub struct ScriptNode {
    script: CompiledScript,
    upstream: u32,
    source_info: ImageInfo,
    config_values: HashMap<String, f64>,
    /// Reference to the shared registry (for builtin() calls in compute strategy)
    dispatch_fn: Option<Arc<dyn Fn(&str, &[u8], &ImageInfo, &HashMap<String, String>) -> Result<Vec<u8>, ImageError> + Send + Sync>>,
}

impl ScriptNode {
    pub fn new(
        script: CompiledScript,
        upstream: u32,
        source_info: ImageInfo,
        config: &HashMap<String, String>,
    ) -> Self {
        let mut config_values = HashMap::new();
        for param in &script.metadata.params {
            let val = config
                .get(&param.name)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(param.default);
            config_values.insert(param.name.clone(), val);
        }

        Self {
            script,
            upstream,
            source_info,
            config_values,
            dispatch_fn: None,
        }
    }

    /// Set the dispatch function for builtin() calls.
    pub fn with_dispatch(
        mut self,
        f: Arc<dyn Fn(&str, &[u8], &ImageInfo, &HashMap<String, String>) -> Result<Vec<u8>, ImageError> + Send + Sync>,
    ) -> Self {
        self.dispatch_fn = Some(f);
        self
    }

    fn make_params_map(&self) -> Map {
        let mut map = Map::new();
        for (k, v) in &self.config_values {
            map.insert(k.clone().into(), Dynamic::from(*v));
        }
        map
    }

    /// For fusable strategy: evaluate lut() for all 256 values per channel.
    fn build_lut(&self, engine: &Engine) -> Option<[u8; 256]> {
        if self.script.metadata.strategy != ScriptStrategy::Fusable {
            return None;
        }

        let params = self.make_params_map();
        let mut lut = [0u8; 256];

        for i in 0..256u32 {
            let mut scope = Scope::new();
            let result = engine
                .call_fn::<Dynamic>(
                    &mut scope,
                    &self.script.ast,
                    "lut",
                    (i as f64, 0i64, params.clone()),
                )
                .ok()?;

            let val = if let Ok(f) = result.as_float() {
                f.clamp(0.0, 255.0) as u8
            } else if let Ok(i) = result.as_int() {
                (i as u8).clamp(0, 255)
            } else {
                return None;
            };
            lut[i as usize] = val;
        }

        Some(lut)
    }

    /// For compute/GPU strategy: call the Rhai compute(pixels, info, params) function.
    ///
    /// Passes pixel data as a Rhai Blob (Vec<u8>) and image metadata as a Map.
    /// The script mutates the blob in-place or returns a new one.
    fn run_compute(&self, pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
        let engine = create_engine();
        let params = self.make_params_map();

        // Build info map: { width, height, channels, len }
        let channels = info.format.bytes_per_pixel() as i64;
        let mut info_map = Map::new();
        info_map.insert("width".into(), Dynamic::from(info.width as i64));
        info_map.insert("height".into(), Dynamic::from(info.height as i64));
        info_map.insert("channels".into(), Dynamic::from(channels));
        info_map.insert("len".into(), Dynamic::from(pixels.len() as i64));

        // Pass pixels as Rhai Blob (Vec<u8>) — supports indexing and mutation
        let blob: rhai::Blob = pixels.to_vec();

        let mut scope = Scope::new();
        let result = engine
            .call_fn::<Dynamic>(
                &mut scope,
                &self.script.ast,
                "compute",
                (blob, info_map, params),
            )
            .map_err(|e| {
                ImageError::ScriptError(format!(
                    "script '{}': compute() failed: {e}",
                    self.script.metadata.name
                ))
            })?;

        // The script must return a Blob (Vec<u8>)
        result.into_blob().map_err(|_| {
            ImageError::ScriptError(format!(
                "script '{}': compute() must return a Blob (byte array)",
                self.script.metadata.name
            ))
        })
    }

    /// For composite strategy: evaluate graph() and return the node DAG.
    pub fn build_graph(&self, engine: &Engine) -> Result<Vec<NodeRef>, String> {
        let params = self.make_params_map();
        let input = NodeRef {
            id: 0,
            filter_name: "__input__".to_string(),
            config: HashMap::new(),
            upstream_id: 0,
        };

        let mut scope = Scope::new();
        let result = engine
            .call_fn::<Dynamic>(&mut scope, &self.script.ast, "graph", (input, params))
            .map_err(|e| format!("graph() failed: {e}"))?;

        // The result is the output NodeRef — walk back to collect all nodes
        if let Some(node_ref) = result.try_cast::<NodeRef>() {
            Ok(vec![node_ref])
        } else {
            Err("graph() must return a NodeRef".to_string())
        }
    }
}

impl ImageNode for ScriptNode {
    fn info(&self) -> ImageInfo {
        self.source_info.clone()
    }

    fn compute_region(
        &self,
        request: Rect,
        upstream_fn: &mut dyn FnMut(u32, Rect) -> Result<Vec<u8>, ImageError>,
    ) -> Result<Vec<u8>, ImageError> {
        match self.script.metadata.strategy {
            ScriptStrategy::Fusable => {
                // Fusable nodes use LUT path, not compute_region.
                // But as fallback for unfused single-node case:
                let engine = create_engine();
                let lut = self.build_lut(&engine).ok_or_else(|| {
                    ImageError::ScriptError(format!(
                        "script '{}': lut() failed to produce valid LUT",
                        self.script.metadata.name
                    ))
                })?;
                let pixels = upstream_fn(self.upstream, request)?;
                let info = ImageInfo {
                    width: request.width,
                    height: request.height,
                    ..self.source_info
                };
                crate::domain::point_ops::apply_lut(&pixels, &info, &lut)
            }
            ScriptStrategy::Compute | ScriptStrategy::Gpu => {
                // Both compute and GPU (CPU fallback) call the Rhai compute() fn.
                // Pixels are passed as a Rhai Blob (Vec<u8>) for direct indexing.
                let pixels = upstream_fn(self.upstream, request)?;
                let info = ImageInfo {
                    width: request.width,
                    height: request.height,
                    ..self.source_info
                };
                self.run_compute(&pixels, &info)
            }
            ScriptStrategy::Composite => {
                // Composite nodes should be expanded into sub-graphs by the
                // pipeline builder, not executed directly
                Err(ImageError::ScriptError(format!(
                    "script '{}': composite nodes must be expanded by the pipeline builder, not executed directly",
                    self.script.metadata.name
                )))
            }
        }
    }

    fn upstream_id(&self) -> Option<u32> {
        Some(self.upstream)
    }

    fn access_pattern(&self) -> AccessPattern {
        match self.script.metadata.strategy {
            ScriptStrategy::Fusable => AccessPattern::Sequential,
            ScriptStrategy::Composite => AccessPattern::Sequential,
            _ => AccessPattern::LocalNeighborhood,
        }
    }

    fn as_point_op_lut(&self) -> Option<[u8; 256]> {
        if self.script.metadata.strategy != ScriptStrategy::Fusable {
            return None;
        }
        let engine = create_engine();
        self.build_lut(&engine)
    }
}

impl rasmcore_pipeline::gpu::GpuCapable for ScriptNode {
    fn gpu_ops(&self, width: u32, height: u32) -> Option<Vec<rasmcore_pipeline::gpu::GpuOp>> {
        if self.script.metadata.strategy != ScriptStrategy::Gpu {
            return None;
        }

        let engine = create_engine();
        let mut scope = Scope::new();
        let shader_src = engine
            .call_fn::<String>(&mut scope, &self.script.ast, "gpu_shader", ())
            .ok()?;

        // Pack width/height as uniform params (8 bytes, little-endian)
        let mut params = Vec::with_capacity(8);
        params.extend_from_slice(&width.to_le_bytes());
        params.extend_from_slice(&height.to_le_bytes());

        Some(vec![rasmcore_pipeline::gpu::GpuOp::Compute {
            shader: shader_src,
            entry_point: "main",
            workgroup_size: [256, 1, 1],
            params,
            extra_buffers: vec![],
        }])
    }
}

// ─── Pipeline Dispatch Integration ─────────────────────────────────────────

/// Dispatch a script filter by name, returning a boxed ImageNode.
///
/// This is the runtime dispatch counterpart to the generated `dispatch_filter`.
/// The pipeline calls generated dispatch first; if not found, tries this.
pub fn dispatch_script_filter(
    registry: &ScriptRegistry,
    name: &str,
    upstream: u32,
    info: ImageInfo,
    params: &HashMap<String, String>,
) -> Result<Box<dyn ImageNode>, String> {
    let script = registry
        .get(name)
        .ok_or_else(|| format!("unknown script filter: {name}"))?
        .clone();

    Ok(Box::new(ScriptNode::new(script, upstream, info, params)))
}

/// Get the list of script filter names and their metadata for the param manifest.
pub fn script_filter_manifest(registry: &ScriptRegistry) -> Vec<ScriptFilterManifestEntry> {
    registry
        .scripts
        .values()
        .map(|s| ScriptFilterManifestEntry {
            name: s.metadata.name.clone(),
            category: s.metadata.category.clone(),
            strategy: format!("{:?}", s.metadata.strategy),
            params: s
                .metadata
                .params
                .iter()
                .map(|p| ScriptParamManifestEntry {
                    name: p.name.clone(),
                    param_type: p.param_type.clone(),
                    default: p.default,
                    min: p.min,
                    max: p.max,
                })
                .collect(),
        })
        .collect()
}

/// Manifest entry for a script filter (for param-manifest.json integration).
#[derive(Debug, Clone)]
pub struct ScriptFilterManifestEntry {
    pub name: String,
    pub category: String,
    pub strategy: String,
    pub params: Vec<ScriptParamManifestEntry>,
}

/// Manifest entry for a script param.
#[derive(Debug, Clone)]
pub struct ScriptParamManifestEntry {
    pub name: String,
    pub param_type: String,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_metadata_basic() {
        let source = r#"
//! name: warm_tint
//! category: color
//! strategy: fusable
//! param warmth: f32 = 0.2 [0.0, 1.0]

fn lut(input, channel, params) {
    input
}
"#;
        let meta = parse_metadata(source).unwrap();
        assert_eq!(meta.name, "warm_tint");
        assert_eq!(meta.category, "color");
        assert_eq!(meta.strategy, ScriptStrategy::Fusable);
        assert_eq!(meta.params.len(), 1);
        assert_eq!(meta.params[0].name, "warmth");
        assert_eq!(meta.params[0].param_type, "f32");
        assert!((meta.params[0].default - 0.2).abs() < 1e-6);
        assert!((meta.params[0].min.unwrap() - 0.0).abs() < 1e-6);
        assert!((meta.params[0].max.unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_metadata_multi_params() {
        let source = r#"
//! name: glow
//! category: effect
//! strategy: compute
//! param radius: f32 = 5.0 [0.1, 50.0]
//! param intensity: f32 = 0.5 [0.0, 1.0]
//! param blend: bool = 1.0

fn compute(pixels, info, params) { pixels }
"#;
        let meta = parse_metadata(source).unwrap();
        assert_eq!(meta.name, "glow");
        assert_eq!(meta.strategy, ScriptStrategy::Compute);
        assert_eq!(meta.params.len(), 3);
    }

    #[test]
    fn registry_compiles_and_validates() {
        let scripts = vec![
            r#"
//! name: test_fusable
//! category: test
//! strategy: fusable
//! param amount: f32 = 1.0 [0.0, 2.0]

fn lut(input, channel, params) {
    clamp_value(input * params.amount, 0.0, 255.0)
}
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        assert!(registry.contains("test_fusable"));
        assert_eq!(registry.list().len(), 1);
    }

    #[test]
    fn registry_rejects_missing_fn() {
        let scripts = vec![
            r#"
//! name: bad_fusable
//! category: test
//! strategy: fusable

fn compute(pixels, info, params) { pixels }
"#
            .to_string(),
        ];

        let err = ScriptRegistry::new(&scripts).err().expect("should fail");
        assert!(err.contains("fusable strategy requires fn lut"));
    }

    #[test]
    fn fusable_lut_evaluation() {
        let scripts = vec![
            r#"
//! name: brighten
//! category: test
//! strategy: fusable
//! param amount: f32 = 10.0 [0.0, 255.0]

fn lut(input, channel, params) {
    clamp_value(input + params.amount, 0.0, 255.0)
}
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        let script = registry.get("brighten").unwrap().clone();

        let config: HashMap<String, String> = HashMap::new();
        let node = ScriptNode::new(
            script,
            0,
            ImageInfo {
                width: 1,
                height: 1,
                format: PixelFormat::Gray8,
                color_space: ColorSpace::Srgb,
            },
            &config,
        );

        let lut = node.as_point_op_lut().unwrap();
        // With default amount=10: lut[0] = 10, lut[245] = 255, lut[255] = 255
        assert_eq!(lut[0], 10);
        assert_eq!(lut[100], 110);
        assert_eq!(lut[245], 255);
        assert_eq!(lut[255], 255);
    }

    #[test]
    fn composite_graph_building() {
        let scripts = vec![
            r#"
//! name: test_composite
//! category: test
//! strategy: composite
//! param strength: f32 = 0.5 [0.0, 1.0]

fn graph(input, params) {
    input
        .apply("blur", #{ radius: params.strength * 10.0 })
        .apply("contrast", #{ amount: params.strength })
}
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        let script = registry.get("test_composite").unwrap().clone();

        let config: HashMap<String, String> = HashMap::new();
        let node = ScriptNode::new(
            script,
            0,
            ImageInfo {
                width: 100,
                height: 100,
                format: PixelFormat::Rgb8,
                color_space: ColorSpace::Srgb,
            },
            &config,
        );

        let engine = create_engine();
        let graph = node.build_graph(&engine).unwrap();
        assert_eq!(graph.len(), 1);
        // The returned node is the last in the chain (contrast)
        assert_eq!(graph[0].filter_name, "contrast");
    }

    #[test]
    fn composite_multi_input_blend() {
        let scripts = vec![
            r#"
//! name: test_glow_composite
//! category: test
//! strategy: composite
//! param radius: f32 = 5.0 [0.1, 50.0]
//! param intensity: f32 = 0.5 [0.0, 1.0]

fn graph(input, params) {
    let blurred = input.apply("blur", #{ radius: params.radius * 3.0 });
    blurred.apply("blend", #{ source: input, mode: "screen", opacity: params.intensity })
}
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        let script = registry.get("test_glow_composite").unwrap().clone();

        let config: HashMap<String, String> = HashMap::new();
        let node = ScriptNode::new(
            script,
            0,
            ImageInfo {
                width: 100,
                height: 100,
                format: PixelFormat::Rgb8,
                color_space: ColorSpace::Srgb,
            },
            &config,
        );

        let engine = create_engine();
        let graph = node.build_graph(&engine).unwrap();
        assert_eq!(graph.len(), 1);
        // The returned node is the blend
        assert_eq!(graph[0].filter_name, "blend");
        // The blend node's config should contain a NodeRef for "source"
        match &graph[0].config["source"] {
            ConfigValue::NodeRef(id) => {
                // Should reference the input node (id=0)
                assert_eq!(*id, 0);
            }
            other => panic!("expected NodeRef, got {:?}", other),
        }
    }

    #[test]
    fn compute_invert_single_application() {
        let scripts = vec![
            r#"
//! name: invert_compute
//! category: test
//! strategy: compute

fn compute(pixels, info, params) {
    let out = pixels;
    for i in range(0, out.len()) {
        out[i] = 255 - out[i];
    }
    out
}
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        let script = registry.get("invert_compute").unwrap().clone();

        let info = ImageInfo {
            width: 4,
            height: 1,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let config: HashMap<String, String> = HashMap::new();
        let node = ScriptNode::new(script, 0, info, &config);

        let input_pixels = vec![0u8, 64, 128, 255];
        let result = node
            .compute_region(
                Rect::new(0, 0, 4, 1),
                &mut |_, _| Ok(input_pixels.clone()),
            )
            .unwrap();

        assert_eq!(result, vec![255, 191, 127, 0]);
    }

    #[test]
    fn gpu_invert_cpu_fallback_single_application() {
        let scripts = vec![
            r#"
//! name: invert_gpu
//! category: test
//! strategy: gpu

fn gpu_shader() {
    `@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = dims.x * dims.y;
    if (idx >= total) { return; }
    let p = input[idx];
    let r = 255u - (p & 0xFFu);
    let g = 255u - ((p >> 8u) & 0xFFu);
    let b = 255u - ((p >> 16u) & 0xFFu);
    let a = (p >> 24u) & 0xFFu;
    output[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
}`
}

fn compute(pixels, info, params) {
    let out = pixels;
    for i in range(0, out.len()) {
        out[i] = 255 - out[i];
    }
    out
}
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        let script = registry.get("invert_gpu").unwrap().clone();

        let info = ImageInfo {
            width: 4,
            height: 1,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let config: HashMap<String, String> = HashMap::new();
        let node = ScriptNode::new(script, 0, info, &config);

        // CPU fallback path (compute_region for GPU strategy)
        let input_pixels = vec![0u8, 64, 128, 255];
        let result = node
            .compute_region(
                Rect::new(0, 0, 4, 1),
                &mut |_, _| Ok(input_pixels.clone()),
            )
            .unwrap();

        assert_eq!(result, vec![255, 191, 127, 0]);
    }

    #[test]
    fn script_node_fusable_compute_region() {
        let scripts = vec![
            r#"
//! name: invert_script
//! category: test
//! strategy: fusable

fn lut(input, channel, params) {
    255.0 - input
}
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        let script = registry.get("invert_script").unwrap().clone();

        let info = ImageInfo {
            width: 4,
            height: 1,
            format: PixelFormat::Gray8,
            color_space: ColorSpace::Srgb,
        };
        let config: HashMap<String, String> = HashMap::new();
        let node = ScriptNode::new(script, 0, info.clone(), &config);

        let input_pixels = vec![0u8, 64, 128, 255];
        let result = node
            .compute_region(
                Rect::new(0, 0, 4, 1),
                &mut |_, _| Ok(input_pixels.clone()),
            )
            .unwrap();

        assert_eq!(result, vec![255, 191, 127, 0]);
    }

    // ─── Self-Inverse (Even/Odd) Tests ────────────────────────────────────

    /// Helper: load an invert script, apply once (odd) and twice (even),
    /// verify inversion and identity respectively.
    fn assert_self_inverse(script_source: &str, name: &str) {
        let scripts = vec![script_source.to_string()];
        let registry = ScriptRegistry::new(&scripts).unwrap();
        let script = registry.get(name).unwrap().clone();

        let info = ImageInfo {
            width: 4,
            height: 1,
            format: PixelFormat::Rgb8,
            color_space: ColorSpace::Srgb,
        };
        let config: HashMap<String, String> = HashMap::new();

        // 12 bytes: 4 pixels * 3 channels (RGB)
        let original = vec![0u8, 64, 128, 255, 100, 200, 10, 20, 30, 240, 250, 5];
        let expected_inverted: Vec<u8> = original.iter().map(|&b| 255 - b).collect();

        // Single application (odd) = inverted
        let node1 = ScriptNode::new(script.clone(), 0, info.clone(), &config);
        let once = node1
            .compute_region(
                Rect::new(0, 0, 4, 1),
                &mut |_, _| Ok(original.clone()),
            )
            .unwrap();
        assert_eq!(once, expected_inverted, "{name}: single apply should invert");

        // Double application (even) = identity
        let node2 = ScriptNode::new(script, 0, info, &config);
        let twice = node2
            .compute_region(
                Rect::new(0, 0, 4, 1),
                &mut |_, _| Ok(once.clone()),
            )
            .unwrap();
        assert_eq!(twice, original, "{name}: double apply should be identity");
    }

    #[test]
    fn fusable_invert_self_inverse() {
        assert_self_inverse(
            r#"
//! name: inv_f
//! category: test
//! strategy: fusable

fn lut(input, channel, params) {
    255.0 - input
}
"#,
            "inv_f",
        );
    }

    #[test]
    fn compute_invert_self_inverse() {
        assert_self_inverse(
            r#"
//! name: inv_c
//! category: test
//! strategy: compute

fn compute(pixels, info, params) {
    let out = pixels;
    for i in range(0, out.len()) {
        out[i] = 255 - out[i];
    }
    out
}
"#,
            "inv_c",
        );
    }

    #[test]
    fn gpu_invert_self_inverse() {
        assert_self_inverse(
            r#"
//! name: inv_g
//! category: test
//! strategy: gpu

fn gpu_shader() {
    `@compute @workgroup_size(256) fn main() {}`
}

fn compute(pixels, info, params) {
    let out = pixels;
    for i in range(0, out.len()) {
        out[i] = 255 - out[i];
    }
    out
}
"#,
            "inv_g",
        );
    }

    #[test]
    fn gpu_invert_produces_valid_gpu_ops() {
        use rasmcore_pipeline::gpu::GpuCapable;

        let scripts = vec![
            r#"
//! name: inv_gpu_ops
//! category: test
//! strategy: gpu

fn gpu_shader() {
    `@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {}`
}

fn compute(pixels, info, params) { pixels }
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        let script = registry.get("inv_gpu_ops").unwrap().clone();

        let info = ImageInfo {
            width: 64,
            height: 64,
            format: PixelFormat::Rgba8,
            color_space: ColorSpace::Srgb,
        };
        let config: HashMap<String, String> = HashMap::new();
        let node = ScriptNode::new(script, 0, info, &config);

        let ops = node.gpu_ops(64, 64).expect("GPU strategy should produce ops");
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            rasmcore_pipeline::gpu::GpuOp::Compute { shader, entry_point, .. } => {
                assert!(shader.contains("@compute"), "shader should contain @compute");
                assert_eq!(*entry_point, "main");
            }
            _ => panic!("expected GpuOp::Compute"),
        }
    }

    #[test]
    fn script_manifest_lists_all_invert_scripts() {
        let scripts = vec![
            r#"
//! name: invert_fusable
//! category: adjustment
//! strategy: fusable

fn lut(input, channel, params) { 255.0 - input }
"#
            .to_string(),
            r#"
//! name: invert_compute
//! category: adjustment
//! strategy: compute

fn compute(pixels, info, params) { pixels }
"#
            .to_string(),
            r#"
//! name: invert_gpu
//! category: adjustment
//! strategy: gpu

fn gpu_shader() { `@compute @workgroup_size(1) fn main() {}` }
fn compute(pixels, info, params) { pixels }
"#
            .to_string(),
        ];

        let registry = ScriptRegistry::new(&scripts).unwrap();
        let manifest = script_filter_manifest(&registry);
        let names: Vec<&str> = manifest.iter().map(|e| e.name.as_str()).collect();

        assert!(names.contains(&"invert_fusable"), "manifest should contain invert_fusable");
        assert!(names.contains(&"invert_compute"), "manifest should contain invert_compute");
        assert!(names.contains(&"invert_gpu"), "manifest should contain invert_gpu");
        assert_eq!(manifest.len(), 3);
    }
}
