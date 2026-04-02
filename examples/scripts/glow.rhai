//! name: glow
//! category: effect
//! strategy: compute
//! param radius: f32 = 5.0 [0.1, 50.0]
//! param intensity: f32 = 0.5 [0.0, 1.0]

// Compute strategy: composes built-in filters eagerly per tile.
// Calls builtin() to invoke blur and blend from the built-in registry.

fn compute(pixels, info, params) {
    let blurred = builtin("blur", pixels, info, #{ radius: params.radius * 3.0 });
    builtin("blend", pixels, blurred, info, #{ mode: "screen", opacity: params.intensity })
}

fn input_rect(output, bounds_w, bounds_h, params) {
    expand_uniform(output, (params.radius * 9.0).ceil(), bounds_w, bounds_h)
}
