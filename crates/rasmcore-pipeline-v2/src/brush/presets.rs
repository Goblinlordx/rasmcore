//! Standard brush presets — ready-to-use brush configurations.

use super::types::{BrushParams, DynamicsCurve, DynamicsCurves};

/// A named brush preset with metadata.
#[derive(Debug, Clone)]
pub struct BrushPreset {
    pub name: &'static str,
    pub display_name: &'static str,
    pub category: &'static str,
    pub description: &'static str,
    pub params: BrushParams,
}

fn linear_curve() -> DynamicsCurve {
    DynamicsCurve {
        points: vec![(0.0, 0.0), (1.0, 1.0)],
    }
}

fn concave_curve() -> DynamicsCurve {
    DynamicsCurve {
        points: vec![(0.0, 0.0), (0.3, 0.6), (0.6, 0.85), (1.0, 1.0)],
    }
}

fn convex_curve() -> DynamicsCurve {
    DynamicsCurve {
        points: vec![(0.0, 0.0), (0.4, 0.15), (0.7, 0.4), (1.0, 1.0)],
    }
}

fn flat_curve() -> DynamicsCurve {
    DynamicsCurve { points: vec![] }
}

pub fn pencil() -> BrushPreset {
    BrushPreset {
        name: "pencil",
        display_name: "Pencil",
        category: "drawing",
        description: "Small hard tip with pressure-to-size. Natural pencil feel.",
        params: BrushParams {
            diameter: 8.0,
            spacing: 0.10,
            hardness: 0.95,
            flow: 0.9,
            opacity: 1.0,
            angle: 0.0,
            roundness: 1.0,
            scatter: 0.0,
            smoothing: 0.3,
            dynamics: DynamicsCurves {
                pressure_size: concave_curve(),
                pressure_opacity: flat_curve(),
                velocity_size: flat_curve(),
                tilt_angle: flat_curve(),
            },
        },
    }
}

pub fn pen() -> BrushPreset {
    BrushPreset {
        name: "pen",
        display_name: "Pen",
        category: "drawing",
        description: "Medium hard tip with pressure-to-opacity. Clean ink strokes.",
        params: BrushParams {
            diameter: 12.0,
            spacing: 0.05,
            hardness: 1.0,
            flow: 1.0,
            opacity: 1.0,
            angle: 0.0,
            roundness: 1.0,
            scatter: 0.0,
            smoothing: 0.5,
            dynamics: DynamicsCurves {
                pressure_size: flat_curve(),
                pressure_opacity: linear_curve(),
                velocity_size: flat_curve(),
                tilt_angle: flat_curve(),
            },
        },
    }
}

pub fn airbrush() -> BrushPreset {
    BrushPreset {
        name: "airbrush",
        display_name: "Airbrush",
        category: "painting",
        description: "Large soft tip with pressure-to-flow. Gradual opacity buildup.",
        params: BrushParams {
            diameter: 80.0,
            spacing: 0.25,
            hardness: 0.15,
            flow: 0.3,
            opacity: 1.0,
            angle: 0.0,
            roundness: 1.0,
            scatter: 0.0,
            smoothing: 0.7,
            dynamics: DynamicsCurves {
                pressure_size: flat_curve(),
                pressure_opacity: concave_curve(),
                velocity_size: flat_curve(),
                tilt_angle: flat_curve(),
            },
        },
    }
}

pub fn watercolor() -> BrushPreset {
    BrushPreset {
        name: "watercolor",
        display_name: "Watercolor",
        category: "painting",
        description: "Soft edge with velocity-to-size and scatter. Organic, fluid feel.",
        params: BrushParams {
            diameter: 40.0,
            spacing: 0.15,
            hardness: 0.3,
            flow: 0.6,
            opacity: 0.8,
            angle: 0.0,
            roundness: 0.85,
            scatter: 0.3,
            smoothing: 0.6,
            dynamics: DynamicsCurves {
                pressure_size: linear_curve(),
                pressure_opacity: concave_curve(),
                velocity_size: convex_curve(),
                tilt_angle: flat_curve(),
            },
        },
    }
}

pub fn chalk() -> BrushPreset {
    BrushPreset {
        name: "chalk",
        display_name: "Chalk",
        category: "drawing",
        description: "Textured tip with tilt-to-size and scatter. Rough, grainy strokes.",
        params: BrushParams {
            diameter: 25.0,
            spacing: 0.12,
            hardness: 0.6,
            flow: 0.75,
            opacity: 0.9,
            angle: 0.0,
            roundness: 0.7,
            scatter: 0.2,
            smoothing: 0.3,
            dynamics: DynamicsCurves {
                pressure_size: linear_curve(),
                pressure_opacity: flat_curve(),
                velocity_size: flat_curve(),
                tilt_angle: linear_curve(),
            },
        },
    }
}

pub fn registered_presets() -> Vec<BrushPreset> {
    vec![pencil(), pen(), airbrush(), watercolor(), chalk()]
}

pub fn find_preset(name: &str) -> Option<BrushPreset> {
    registered_presets().into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_presets_unique_names() {
        let presets = registered_presets();
        let mut names: Vec<&str> = presets.iter().map(|p| p.name).collect();
        let count = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn all_presets_valid_params() {
        for p in registered_presets() {
            assert!(p.params.diameter > 0.0, "{}", p.name);
            assert!(p.params.spacing > 0.0, "{}", p.name);
            assert!(
                p.params.hardness >= 0.0 && p.params.hardness <= 1.0,
                "{}",
                p.name
            );
        }
    }

    #[test]
    fn find_all_by_name() {
        for name in ["pencil", "pen", "airbrush", "watercolor", "chalk"] {
            assert!(find_preset(name).is_some(), "Missing: {name}");
        }
        assert!(find_preset("nonexistent").is_none());
    }

    #[test]
    fn presets_visually_distinct() {
        let pencil = find_preset("pencil").unwrap();
        let airbrush = find_preset("airbrush").unwrap();
        assert!(pencil.params.diameter < airbrush.params.diameter);
        assert!(pencil.params.hardness > airbrush.params.hardness);
    }
}
