#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load_rows(path: Path):
    if not path.exists():
        return []
    with path.open() as fin:
        return list(csv.DictReader(fin))


def scene_name(value: str):
    if "office" in value:
        return "office"
    if "turtlebot_pyebaekRoom_1_scene_1" in value or "turtlebot_compact_locality_residual" in value:
        return "turtlebot_pyebaekRoom_1_scene_1"
    return value


def format_method(row):
    variant = row["variant"]
    scene = row["scene"]
    if variant == "ply" and "baseline" in scene:
        return "baseline_ply"
    if variant == "ply":
        return "compact_train_ply"
    if variant == "compact_fixed" and "baseline" in scene:
        return "baseline_compact_fixed"
    if variant == "compact_fixed":
        return "compact_fixed"
    if variant in {"compact_export_shdrop", "compact_export_shdrop_aggressive"}:
        return "export_shdrop"
    if variant == "compact_locality_residual":
        return "locality_residual"
    return variant


def format_cell(value):
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
        return f"{number:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return str(value)


def write_markdown(rows, output_path: Path):
    columns = [
        "scene",
        "method",
        "num_points",
        "ply_bytes",
        "compact_bytes",
        "compression_ratio",
        "avg_psnr",
        "avg_dssim",
        "avg_render_ms",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a flat phase-1 paper table from current result csv files.")
    parser.add_argument("--formal-summary", default=str(ROOT / "results_smoke/formal_compare_summary.csv"))
    parser.add_argument("--shdrop-summary", default=str(ROOT / "results_smoke/export_shdrop_aggressive_compare.csv"))
    parser.add_argument("--locality-summary", default=str(ROOT / "results_smoke/export_locality_residual_compare.csv"))
    parser.add_argument("--output", default=str(ROOT / "results_smoke/phase1_main_table.csv"))
    parser.add_argument("--markdown-output", default=str(ROOT / "results_smoke/phase1_main_table.md"))
    args = parser.parse_args()

    rows = []
    for row in load_rows(Path(args.formal_summary)):
        method = format_method(row)
        if method not in {"baseline_ply", "baseline_compact_fixed", "compact_train_ply", "compact_fixed"}:
            continue
        rows.append(
            {
                "scene": scene_name(row["scene"]),
                "method": method,
                "num_points": row["num_points"],
                "ply_bytes": row["ply_bytes"],
                "compact_bytes": row["compact_bytes"],
                "compression_ratio": row["compression_ratio"],
                "avg_psnr": row["avg_psnr"],
                "avg_dssim": row["avg_dssim"],
                "avg_render_ms": row["avg_render_ms"],
            }
        )
    for summary_path, expected_method in [
        (Path(args.shdrop_summary), "export_shdrop"),
        (Path(args.locality_summary), "locality_residual"),
    ]:
        for row in load_rows(summary_path):
            method = format_method(row)
            if method != expected_method:
                continue
            rows.append(
                {
                    "scene": scene_name(row["scene"]),
                    "method": method,
                    "num_points": row["num_points"],
                    "ply_bytes": row["ply_bytes"],
                    "compact_bytes": row["compact_bytes"],
                    "compression_ratio": row["compression_ratio"],
                    "avg_psnr": row["avg_psnr"],
                    "avg_dssim": row["avg_dssim"],
                    "avg_render_ms": row["avg_render_ms"],
                }
            )

    method_order = {
        "baseline_ply": 0,
        "baseline_compact_fixed": 1,
        "compact_train_ply": 2,
        "compact_fixed": 3,
        "export_shdrop": 4,
        "locality_residual": 5,
    }
    rows.sort(key=lambda row: (row["scene"], method_order.get(row["method"], 99)))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "scene",
                "method",
                "num_points",
                "ply_bytes",
                "compact_bytes",
                "compression_ratio",
                "avg_psnr",
                "avg_dssim",
                "avg_render_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown_output = Path(args.markdown_output).expanduser().resolve()
    write_markdown(rows, markdown_output)
    print(f"[phase1-table] wrote {len(rows)} rows to {output_path}")
    print(f"[phase1-table] wrote markdown table to {markdown_output}")


if __name__ == "__main__":
    raise SystemExit(main())
