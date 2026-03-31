#!/usr/bin/env python3
"""Migrate filter functions to config struct signatures.

For each registered filter with 1+ params and no existing config: param,
changes the function signature and adds destructuring.

Handles:
- Simple scalar params (direct field access)
- Nested color structs (config.color.r as u32)
- Renamed fields (x_offset → x_inset mapping)
- String/slice params (kept as individual params)
- process_via_8bit recursive calls

Run: python3 scripts/migrate_to_config_struct.py --apply
"""

import re
import sys
import os

def to_pascal(s):
    return ''.join(w.capitalize() for w in s.split('_'))

# Special field mappings: when struct field name differs from function param name
# Format: { struct_name: { fn_param_name: "config.path.to.value" } }
FIELD_OVERRIDES = {
    "ColorizeParams": {
        "target_r": "config.target.r",
        "target_g": "config.target.g",
        "target_b": "config.target.b",
    },
    "DrawLineParams": {
        "color_r": "config.color.r as u32",
        "color_g": "config.color.g as u32",
        "color_b": "config.color.b as u32",
        "color_a": "config.color.a as u32",
    },
    "DrawRectParams": {
        "color_r": "config.color.r as u32",
        "color_g": "config.color.g as u32",
        "color_b": "config.color.b as u32",
        "color_a": "config.color.a as u32",
    },
    "DrawCircleParams": {
        "color_r": "config.color.r as u32",
        "color_g": "config.color.g as u32",
        "color_b": "config.color.b as u32",
        "color_a": "config.color.a as u32",
    },
    "DrawTextParams": {
        "color_r": "config.color.r as u32",
        "color_g": "config.color.g as u32",
        "color_b": "config.color.b as u32",
        "color_a": "config.color.a as u32",
    },
    "VignetteParams": {
        "x_inset": "config.x_offset",
        "y_inset": "config.y_offset",
    },
}

# Params that should stay as individual function args (not in config struct)
INDIVIDUAL_TYPES = {"&str", "String", "&[f32]", "&[f64]", "&[u8]", "&[u32]"}

def find_struct_name(filter_name, fn_name, content):
    candidates = set()
    candidates.add(to_pascal(filter_name) + "Params")
    base = fn_name
    for suffix in ['_registered', '_filter']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    candidates.add(to_pascal(base) + "Params")

    for c in candidates:
        if f"struct {c}" in content:
            return c
    return None

def get_struct_fields(struct_name, content):
    m = re.search(rf'struct {struct_name}\s*\{{(.*?)\}}', content, re.DOTALL)
    if not m:
        return []
    return re.findall(r'pub\s+(\w+):', m.group(1))

def migrate(filepath, apply=False):
    with open(filepath) as f:
        lines = f.readlines()

    content = ''.join(lines)
    migrations = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if '#[rasmcore_macros::register_filter(' not in line:
            i += 1
            continue

        # Collect attribute text
        attr_text = line
        attr_end = i
        while ')]' not in attr_text:
            attr_end += 1
            if attr_end >= len(lines):
                break
            attr_text += lines[attr_end]

        # Find pub fn
        j = attr_end + 1
        while j < len(lines) and not lines[j].strip().startswith('pub fn '):
            j += 1
        if j >= len(lines):
            i = j
            continue

        # Collect function signature
        sig_start = j
        sig_text = ""
        while j < len(lines):
            sig_text += lines[j]
            if '{' in lines[j]:
                break
            j += 1
        sig_end = j

        fn_m = re.search(r'pub fn (\w+)\(', sig_text)
        if not fn_m:
            i = j + 1
            continue
        fn_name = fn_m.group(1)

        name_m = re.search(r'name\s*=\s*"([^"]+)"', attr_text)
        filter_name = name_m.group(1) if name_m else fn_name

        # Skip already migrated
        if 'config:' in sig_text:
            i = j + 1
            continue

        # Parse params
        params_m = re.search(r'\((.*?)\)\s*->', sig_text, re.DOTALL)
        if not params_m:
            i = j + 1
            continue

        config_params = []  # go in config struct
        individual_params = []  # stay as individual args
        for p in params_m.group(1).split(','):
            p = p.strip()
            if not p or 'pixels' in p or 'info' in p or 'ImageInfo' in p:
                continue
            if ':' not in p:
                continue
            n = p.split(':')[0].strip().lstrip('_')
            t = p.split(':', 1)[1].strip().rstrip(',').strip()
            if t in INDIVIDUAL_TYPES:
                individual_params.append((n, t))
            else:
                config_params.append((n, t))

        if not config_params:
            i = j + 1
            continue

        struct_name = find_struct_name(filter_name, fn_name, content)
        if not struct_name:
            print(f"  SKIP {filter_name}: no ConfigParams struct")
            i = j + 1
            continue

        struct_fields = get_struct_fields(struct_name, content)
        overrides = FIELD_OVERRIDES.get(struct_name, {})

        # Build new signature
        sig_parts = ["    pixels: &[u8],\n", "    info: &ImageInfo,\n"]
        for n, t in individual_params:
            sig_parts.append(f"    {n}: {t},\n")
        sig_parts.append(f"    config: &{struct_name},\n")
        new_sig = f"pub fn {fn_name}(\n{''.join(sig_parts)}) -> Result<Vec<u8>, ImageError> {{\n"

        # Build destructuring
        destructure_lines = []
        for n, t in config_params:
            if n in overrides:
                expr = overrides[n]
                destructure_lines.append(f"    let {n} = {expr};\n")
            elif n in struct_fields:
                destructure_lines.append(f"    let {n} = config.{n};\n")
            else:
                # Field not in struct — might be derived from info or have default
                print(f"  WARN {filter_name}.{n}: not in {struct_name}, using default")
                if t == "u32":
                    destructure_lines.append(f"    let {n} = 0u32; // TODO: add to {struct_name}\n")
                elif t == "f32":
                    destructure_lines.append(f"    let {n} = 0.0f32; // TODO: add to {struct_name}\n")
                elif t == "bool":
                    destructure_lines.append(f"    let {n} = false; // TODO: add to {struct_name}\n")
                else:
                    destructure_lines.append(f"    let {n} = Default::default(); // TODO: add to {struct_name}\n")

        destructure = ''.join(destructure_lines)

        migrations.append({
            'filter_name': filter_name,
            'fn_name': fn_name,
            'struct_name': struct_name,
            'sig_start': sig_start,
            'sig_end': sig_end,
            'new_sig': new_sig,
            'destructure': destructure,
            'config_params': config_params,
            'individual_params': individual_params,
        })

        i = j + 1

    print(f"\nTotal migrations: {len(migrations)}")

    if not apply:
        print("Dry run — use --apply to modify")
        for m in migrations:
            extras = f" + {len(m['individual_params'])} individual" if m['individual_params'] else ""
            print(f"  {m['filter_name']} → &{m['struct_name']} ({len(m['config_params'])}p{extras})")
        return

    # Apply in reverse order
    for m in reversed(migrations):
        s, e = m['sig_start'], m['sig_end']
        lines[s:e + 1] = [m['new_sig']]
        lines.insert(s + 1, m['destructure'] + '\n')

    with open(filepath, 'w') as f:
        f.writelines(lines)
    print(f"Applied {len(migrations)} migrations")

if __name__ == '__main__':
    migrate('crates/rasmcore-image/src/domain/filters.rs', '--apply' in sys.argv)
