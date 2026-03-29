#!/usr/bin/env python3
"""
Generate a Markdown performance comparison report from Criterion benchmark results.

Usage:
    python3 scripts/bench-report.py [target/criterion]

Reads the Criterion JSON output and produces:
1. A Markdown table comparing rasmcore vs each reference tool
2. A JSON summary for programmatic consumption

Output: benches/REPORT.md and benches/report.json
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

def find_estimates(criterion_dir: Path):
    """Walk Criterion output and extract benchmark estimates."""
    results = []
    for est_file in criterion_dir.rglob("new/estimates.json"):
        # Path structure: target/criterion/<group>/<bench_id>/new/estimates.json
        bench_path = est_file.parent.parent
        bench_id = bench_path.name
        group_path = bench_path.parent
        group = group_path.name

        try:
            with open(est_file) as f:
                data = json.load(f)

            # Extract median time in nanoseconds
            median_ns = data.get("median", {}).get("point_estimate", 0)

            # Parse the benchmark ID: "jpeg/rasmcore/256" or "jpeg/imagemagick/256"
            parts = bench_id.split("/") if "/" in bench_id else [bench_id]

            results.append({
                "group": group,
                "bench_id": bench_id,
                "parts": parts,
                "median_ns": median_ns,
                "median_ms": median_ns / 1_000_000,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return results


def group_results(results):
    """Group results by operation and size for comparison."""
    # Key: (group, operation, size) → {tool: median_ms}
    grouped = defaultdict(dict)

    for r in results:
        parts = r["bench_id"].replace("\\", "/").split("/")
        if len(parts) >= 2:
            operation = parts[0]
            tool = parts[1]
            size = parts[2] if len(parts) > 2 else "default"
        else:
            operation = r["bench_id"]
            tool = "unknown"
            size = "default"

        key = (r["group"], operation, size)
        grouped[key][tool] = r["median_ms"]

    return grouped


def generate_markdown(grouped, output_path: Path):
    """Generate Markdown comparison table."""
    lines = [
        "# Performance Benchmark Report",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
    ]

    # Group by benchmark group (decoder, encoder, filter, pipeline)
    by_group = defaultdict(list)
    for (group, op, size), tools in sorted(grouped.items()):
        by_group[group].append((op, size, tools))

    for group_name, entries in sorted(by_group.items()):
        lines.append(f"### {group_name.title()}")
        lines.append("")

        # Collect all tool names
        all_tools = set()
        for _, _, tools in entries:
            all_tools.update(tools.keys())
        tool_order = sorted(all_tools)

        # Table header
        header = "| Operation | Size |"
        separator = "|-----------|------|"
        for tool in tool_order:
            header += f" {tool} (ms) |"
            separator += "----------:|"

        lines.append(header)
        lines.append(separator)

        for op, size, tools in sorted(entries):
            row = f"| {op} | {size} |"
            rasmcore_time = tools.get("rasmcore", None)

            for tool in tool_order:
                ms = tools.get(tool)
                if ms is not None:
                    # Add relative comparison to rasmcore
                    if rasmcore_time and tool != "rasmcore" and rasmcore_time > 0:
                        ratio = ms / rasmcore_time
                        if ratio < 1:
                            row += f" {ms:.2f} ({ratio:.1f}x faster) |"
                        else:
                            row += f" {ms:.2f} ({ratio:.1f}x slower) |"
                    else:
                        row += f" {ms:.2f} |"
                else:
                    row += " — |"
            lines.append(row)

        lines.append("")

    # Write
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to: {output_path}")


def generate_json(grouped, output_path: Path):
    """Generate JSON summary."""
    data = {}
    for (group, op, size), tools in sorted(grouped.items()):
        key = f"{group}/{op}/{size}"
        data[key] = {tool: round(ms, 3) for tool, ms in tools.items()}

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"JSON written to: {output_path}")


def main():
    criterion_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("target/criterion")

    if not criterion_dir.exists():
        print(f"Criterion output not found at {criterion_dir}")
        print("Run benchmarks first: cargo bench --bench perf -p rasmcore-image")
        sys.exit(1)

    results = find_estimates(criterion_dir)
    if not results:
        print("No benchmark results found.")
        sys.exit(1)

    print(f"Found {len(results)} benchmark results")

    grouped = group_results(results)

    # Output paths
    report_dir = Path("crates/rasmcore-image/benches")
    report_dir.mkdir(parents=True, exist_ok=True)

    generate_markdown(grouped, report_dir / "REPORT.md")
    generate_json(grouped, report_dir / "report.json")


if __name__ == "__main__":
    main()
