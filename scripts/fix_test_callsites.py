#!/usr/bin/env python3
"""Fix test call sites for config-struct-migrated filters.

Transforms calls like:
  blur(&pixels, &info, 5.0)
To:
  blur(&pixels, &info, &BlurParams { radius: 5.0 })

Uses the actual ConfigParams struct field definitions to build correct constructors.
"""

import re
import sys

def to_pascal(s):
    return ''.join(w.capitalize() for w in s.split('_'))

def parse_struct_fields(content, struct_name):
    """Extract field names and types from a ConfigParams struct."""
    m = re.search(rf'pub struct {struct_name}\s*\{{(.*?)\}}', content, re.DOTALL)
    if not m:
        return []
    fields = re.findall(r'pub\s+(\w+):\s*([\w:]+(?:<[^>]+>)?)', m.group(1))
    return fields

def get_migrated_filters(content):
    """Find all filters that take config: &XxxParams."""
    pattern = r'pub fn (\w+)\(\s*\n?\s*pixels:.*?config:\s*&(\w+Params)'
    return {fn: struct for fn, struct in re.findall(pattern, content, re.DOTALL)}

def fix_call(line, fn_name, struct_name, fields, content):
    """Fix a single function call from individual params to config struct."""
    # Match: fn_name(&expr, &expr, arg1, arg2, ...)
    # or: fn_name(&expr, &expr, arg1) where there might be &str before config

    # Check if this function has individual params before config (like text: &str)
    fn_sig_match = re.search(rf'pub fn {fn_name}\((.*?)\)', content, re.DOTALL)
    if not fn_sig_match:
        return line

    sig = fn_sig_match.group(1)
    # Check for params between info and config
    individual_before = []
    parts = sig.split(',')
    found_info = False
    for p in parts:
        p = p.strip()
        if 'info' in p or 'ImageInfo' in p:
            found_info = True
            continue
        if 'config' in p:
            break
        if found_info and ':' in p:
            pname = p.split(':')[0].strip()
            ptype = p.split(':')[1].strip()
            individual_before.append((pname, ptype))

    # Build regex to match the call
    # Pattern: fn_name(pixels_expr, info_expr, [individual_args...,] config_args...)
    # This is complex because args can be expressions with commas inside parens
    # Simplified: just find the function call and replace args

    return line  # Placeholder — actual replacement done below

def main():
    filepath = 'crates/rasmcore-image/src/domain/filters.rs'
    with open(filepath) as f:
        content = f.read()

    # Get all migrated filters
    migrated = get_migrated_filters(content)
    print(f"Found {len(migrated)} migrated filters")

    # Build struct field info
    struct_fields = {}
    for fn_name, struct_name in migrated.items():
        fields = parse_struct_fields(content, struct_name)
        struct_fields[struct_name] = fields
        # print(f"  {fn_name} → {struct_name}: {[f[0] for f in fields]}")

    # Now process each line looking for test call patterns
    lines = content.split('\n')
    in_test = False
    changes = 0

    for i, line in enumerate(lines):
        # Track if we're in a test module
        if '#[cfg(test)]' in line:
            in_test = True
        if in_test and line.strip() == '}' and not any(c in line for c in ['//', '/*']):
            # Might be end of test module — rough heuristic
            pass

        # Look for filter function calls in test code
        for fn_name, struct_name in migrated.items():
            # Skip if this line doesn't contain the function call
            if f'{fn_name}(' not in line and f'{fn_name} (' not in line:
                continue

            # Skip if it already has the config struct
            if struct_name in line or 'config' in line:
                continue

            # Skip function definitions
            if 'pub fn ' in line or 'fn ' in line.lstrip() and '(' in line and ')' not in line:
                continue

            fields = struct_fields.get(struct_name, [])
            if not fields:
                continue

            # Try to match and fix the call
            # Pattern: fn_name(expr1, expr2, val1, val2, ...)
            # We need to extract the values after the first 2 args (pixels, info)
            call_pattern = rf'({fn_name})\(([^)]*)\)'
            m = re.search(call_pattern, line)
            if not m:
                # Might be multi-line call — skip for now
                continue

            call_args = m.group(2)
            args = split_args(call_args)

            if len(args) < 2:
                continue

            # First 2 args are pixels and info
            pixels_arg = args[0].strip()
            info_arg = args[1].strip()

            # Check if function has individual params before config (like text, kernel)
            fn_sig = re.search(rf'pub fn {fn_name}\((.*?)\)\s*->', content, re.DOTALL)
            if not fn_sig:
                continue

            sig_params = fn_sig.group(1).split(',')
            individual_params = []
            config_start_idx = 2  # after pixels, info
            for sp in sig_params:
                sp = sp.strip()
                if 'pixels' in sp or 'info' in sp or 'ImageInfo' in sp:
                    continue
                if 'config' in sp:
                    break
                if ':' in sp:
                    individual_params.append(sp.split(':')[0].strip())
                    config_start_idx += 1

            # Now args[2:2+len(individual_params)] are individual args
            # and args[2+len(individual_params):] should map to config struct fields
            ind_count = len(individual_params)
            ind_args = args[2:2+ind_count]
            config_args = args[2+ind_count:]

            if len(config_args) != len(fields):
                # Mismatch — skip
                # print(f"  SKIP {fn_name} line {i+1}: {len(config_args)} args vs {len(fields)} fields")
                continue

            # Check for nested types (ColorRgba, ColorRgb)
            field_assignments = []
            arg_idx = 0
            for fname, ftype in fields:
                if 'ColorRgba' in ftype:
                    # Need 4 args: r, g, b, a
                    if arg_idx + 3 < len(config_args):
                        r, g, b, a = [a.strip() for a in config_args[arg_idx:arg_idx+4]]
                        field_assignments.append(f'{fname}: super::param_types::ColorRgba {{ r: {r} as u8, g: {g} as u8, b: {b} as u8, a: {a} as u8 }}')
                        arg_idx += 4
                    else:
                        break
                elif 'ColorRgb' in ftype:
                    # Need 3 args: r, g, b
                    if arg_idx + 2 < len(config_args):
                        r, g, b = [a.strip() for a in config_args[arg_idx:arg_idx+3]]
                        field_assignments.append(f'{fname}: super::param_types::ColorRgb {{ r: {r} as u8, g: {g} as u8, b: {b} as u8 }}')
                        arg_idx += 3
                    else:
                        break
                else:
                    if arg_idx < len(config_args):
                        field_assignments.append(f'{fname}: {config_args[arg_idx].strip()}')
                        arg_idx += 1

            if len(field_assignments) != len(fields):
                continue

            # Build new call
            struct_init = ', '.join(field_assignments)
            new_args = [pixels_arg, info_arg] + ind_args + [f'&{struct_name} {{ {struct_init} }}']
            new_call = f'{fn_name}({", ".join(new_args)})'
            new_line = line[:m.start()] + new_call + line[m.end():]
            lines[i] = new_line
            changes += 1

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Fixed {changes} test call sites")

def split_args(s):
    """Split comma-separated args, respecting nested parens/brackets."""
    args = []
    depth = 0
    current = []
    for c in s:
        if c in '([{':
            depth += 1
            current.append(c)
        elif c in ')]}':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            args.append(''.join(current))
            current = []
        else:
            current.append(c)
    if current:
        args.append(''.join(current))
    return args

if __name__ == '__main__':
    main()
