#!/usr/bin/env python3

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    subprocess.run(cmd, check=True, cwd=ROOT)


def load_rows(path: Path):
    if not path.exists():
        return []
    with path.open() as fin:
        return list(csv.DictReader(fin))


def write_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value):
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
        return f"{number:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return str(value)


def write_markdown(path: Path, rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build phase-2 paper tables from formal phase-1 results plus phase-2 summaries.")
    parser.add_argument(
        "--phase2-manifest",
        default=str(ROOT / "scripts/phase2_smoke_manifest.json"),
        help="Manifest describing phase-2 decoded compact evaluations",
    )
    parser.add_argument(
        "--phase2-summary",
        default=str(ROOT / "results_smoke/phase2_smoke_summary.csv"),
        help="Intermediate CSV summary for phase-2 entries",
    )
    parser.add_argument(
        "--phase2-summary-md",
        default=str(ROOT / "results_smoke/phase2_smoke_summary.md"),
        help="Intermediate markdown summary for phase-2 entries",
    )
    parser.add_argument(
        "--phase1-main",
        default=str(ROOT / "results_smoke/phase1_main_table.csv"),
        help="Existing merged Phase-1 main table used as the base",
    )
    parser.add_argument(
        "--main-output",
        default=str(ROOT / "results_smoke/phase2_main_table.csv"),
        help="Unified phase-1/phase-2 main table CSV",
    )
    parser.add_argument(
        "--main-md-output",
        default=str(ROOT / "results_smoke/phase2_main_table.md"),
        help="Unified phase-1/phase-2 main table markdown",
    )
    parser.add_argument(
        "--paper-output",
        default=str(ROOT / "results_smoke/phase2_paper_main_table.csv"),
        help="Paper-oriented main table CSV",
    )
    parser.add_argument(
        "--paper-md-output",
        default=str(ROOT / "results_smoke/phase2_paper_main_table.md"),
        help="Paper-oriented main table markdown",
    )
    parser.add_argument(
        "--takeaways-output",
        default=str(ROOT / "results_smoke/phase2_paper_takeaways.md"),
        help="Phase-2 takeaways markdown output",
    )
    args = parser.parse_args()

    run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_compact_results.py"),
            "--manifest",
            str(Path(args.phase2_manifest).expanduser().resolve()),
            "--output",
            str(Path(args.phase2_summary).expanduser().resolve()),
            "--markdown-output",
            str(Path(args.phase2_summary_md).expanduser().resolve()),
        ]
    )

    phase1_rows = load_rows(Path(args.phase1_main).expanduser().resolve())
    run(
        [
            sys.executable,
            str(ROOT / "scripts/build_phase1_main_table.py"),
            "--formal-summary",
            str(ROOT / "results_smoke/formal_compare_summary.csv"),
            "--shdrop-summary",
            str(ROOT / "results_smoke/export_shdrop_aggressive_compare.csv"),
            "--locality-summary",
            str(ROOT / "results_smoke/export_locality_residual_compare.csv"),
            "--extra-summary",
            str(Path(args.phase2_summary).expanduser().resolve()),
            "--output",
            str(ROOT / "results_smoke/.phase2_methods_only.csv"),
            "--markdown-output",
            str(ROOT / "results_smoke/.phase2_methods_only.md"),
        ]
    )

    phase2_only_rows = [
        row
        for row in load_rows(ROOT / "results_smoke/.phase2_methods_only.csv")
        if row.get("method") == "phase2_residual_field"
    ]
    merged = list(phase1_rows)
    existing_keys = {(row["scene"], row["method"]) for row in merged}
    for row in phase2_only_rows:
        key = (row["scene"], row["method"])
        if key not in existing_keys:
            merged.append(row)
            existing_keys.add(key)

    merged.sort(key=lambda row: (row["scene"], row["method"]))
    fieldnames = [
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
    write_rows(Path(args.main_output).expanduser().resolve(), merged, fieldnames)
    write_markdown(Path(args.main_md_output).expanduser().resolve(), merged, fieldnames)

    run(
        [
            sys.executable,
            str(ROOT / "scripts/build_phase1_paper_artifacts.py"),
            "--phase1-main",
            str(Path(args.main_output).expanduser().resolve()),
            "--ablation",
            str(ROOT / "results_smoke/ablation_summary.csv"),
            "--breakdown",
            str(ROOT / "results_smoke/formal_compare_compact_breakdown.csv"),
            "--main-output",
            str(Path(args.paper_output).expanduser().resolve()),
            "--main-md-output",
            str(Path(args.paper_md_output).expanduser().resolve()),
            "--ablation-output",
            str(ROOT / "results_smoke/phase2_paper_ablation_table.csv"),
            "--ablation-md-output",
            str(ROOT / "results_smoke/phase2_paper_ablation_table.md"),
            "--takeaways-output",
            str(Path(args.takeaways_output).expanduser().resolve()),
        ]
    )

    print(f"[phase2-paper] wrote phase-2 summary to {Path(args.phase2_summary).expanduser().resolve()}")
    print(f"[phase2-paper] wrote unified main table to {Path(args.main_output).expanduser().resolve()}")
    print(f"[phase2-paper] wrote paper table to {Path(args.paper_output).expanduser().resolve()}")
    print(f"[phase2-paper] wrote takeaways to {Path(args.takeaways_output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
