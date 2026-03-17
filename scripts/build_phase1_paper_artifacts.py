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


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None, "-"):
        return default
    return float(value)


def as_int(row, key, default=0):
    value = row.get(key, "")
    if value in ("", None, "-"):
        return default
    return int(float(value))


def mb_from_bytes(value):
    return value / 1_000_000.0


def pct_reduction(baseline, current):
    if baseline <= 0:
        return 0.0
    return 100.0 * (1.0 - current / baseline)


def choose_model_bytes(row):
    method = row["method"]
    if method in {"baseline_ply", "compact_train_ply"}:
        return as_int(row, "ply_bytes")
    return as_int(row, "compact_bytes")


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                text = f"{value:.4f}".rstrip("0").rstrip(".")
            else:
                text = str(value)
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_main_table(phase1_rows):
    grouped = {}
    for row in phase1_rows:
        grouped.setdefault(row["scene"], {})[row["method"]] = row

    method_specs = [
        ("baseline_ply", "OmniGS baseline", "ply"),
        ("compact_train_ply", "Ours train-time compact", "ply"),
        ("locality_residual", "Ours final compact", "compact"),
        ("phase2_residual_field", "Ours phase-2 residual field", "compact"),
    ]

    output_rows = []
    for scene in sorted(grouped):
        scene_rows = grouped[scene]
        if "baseline_ply" not in scene_rows:
            continue
        baseline = scene_rows["baseline_ply"]
        baseline_points = as_int(baseline, "num_points")
        baseline_bytes = as_int(baseline, "ply_bytes")
        baseline_psnr = as_float(baseline, "avg_psnr")

        for method, display_name, fmt in method_specs:
            if method not in scene_rows:
                continue
            row = scene_rows[method]
            current_points = as_int(row, "num_points")
            current_bytes = choose_model_bytes(row)
            current_psnr = as_float(row, "avg_psnr")
            output_rows.append(
                {
                    "scene": scene,
                    "setting": display_name,
                    "format": fmt,
                    "gaussians": current_points,
                    "gaussian_drop_pct": pct_reduction(baseline_points, current_points),
                    "model_mb": mb_from_bytes(current_bytes),
                    "size_drop_pct": pct_reduction(baseline_bytes, current_bytes),
                    "size_ratio_vs_baseline": baseline_bytes / current_bytes if current_bytes else 0.0,
                    "psnr": current_psnr,
                    "psnr_drop_db": baseline_psnr - current_psnr,
                    "dssim": as_float(row, "avg_dssim"),
                }
            )

    return output_rows


def build_ablation_table(ablation_rows):
    grouped = {}
    for row in ablation_rows:
        grouped.setdefault(row["scene"], {})[row["variant"]] = row

    scene = "office_30k_ablation"
    baseline = grouped["office_baseline_30k_ablation"]["ply"]
    baseline_points = as_int(baseline, "num_points")
    baseline_ply_bytes = as_int(baseline, "ply_bytes")
    baseline_psnr = as_float(baseline, "avg_psnr")

    specs = [
        ("office_baseline_30k_ablation", "Baseline"),
        ("office_prune_only_30k_ablation", "Prune only"),
        ("office_prune_sh_30k_ablation", "Prune + adaptive SH"),
        ("office_full_compact_30k_ablation", "Full compact train"),
    ]

    output_rows = []
    for scene_key, display_name in specs:
        ply_row = grouped[scene_key]["ply"]
        compact_row = grouped[scene_key]["compact_fixed"]
        points = as_int(ply_row, "num_points")
        ply_bytes = as_int(ply_row, "ply_bytes")
        compact_bytes = as_int(compact_row, "compact_bytes")
        psnr = as_float(ply_row, "avg_psnr")
        output_rows.append(
            {
                "scene": scene,
                "variant": display_name,
                "gaussians": points,
                "gaussian_drop_pct": pct_reduction(baseline_points, points),
                "ply_mb": mb_from_bytes(ply_bytes),
                "compact_mb": mb_from_bytes(compact_bytes),
                "compact_ratio_vs_baseline_ply": baseline_ply_bytes / compact_bytes if compact_bytes else 0.0,
                "psnr": psnr,
                "psnr_delta_db": psnr - baseline_psnr,
            }
        )

    return output_rows


def build_takeaways(main_rows, breakdown_rows, ablation_rows):
    def delta_phrase(current, reference, unit, lower_is_better=False):
        delta = current - reference
        magnitude = abs(delta)
        if magnitude < 1e-9:
            return f"keeps {unit} unchanged"
        if lower_is_better:
            verb = "improves" if delta < 0 else "worsens"
        else:
            verb = "improves" if delta > 0 else "drops"
        return f"{verb} {unit} by {magnitude:.3f}"

    def find_main(scene, setting):
        for row in main_rows:
            if row["scene"] == scene and row["setting"] == setting:
                return row
        return None

    def find_breakdown(label_suffix):
        for row in breakdown_rows:
            if row["label"].endswith(label_suffix):
                return row
        return None

    office_baseline = find_main("office", "OmniGS baseline")
    office_train = find_main("office", "Ours train-time compact")
    office_final = find_main("office", "Ours final compact")
    office_phase2 = find_main("office", "Ours phase-2 residual field")
    turtle_baseline = find_main("turtlebot_pyebaekRoom_1_scene_1", "OmniGS baseline")
    turtle_train = find_main("turtlebot_pyebaekRoom_1_scene_1", "Ours train-time compact")
    turtle_final = find_main("turtlebot_pyebaekRoom_1_scene_1", "Ours final compact")
    turtle_phase2 = find_main("turtlebot_pyebaekRoom_1_scene_1", "Ours phase-2 residual field")

    office_fixed = find_breakdown("office_compact_30k_compact_fixed")
    office_locality = find_breakdown("office_compact_30k_compact_locality_residual")
    turtle_fixed = find_breakdown("turtlebot_pyebaekRoom_1_scene_1_compact_32000_compact_fixed")
    turtle_locality = find_breakdown("turtlebot_pyebaekRoom_1_scene_1_compact_32000_compact_locality_residual")

    lines = []
    title = "# Compact GS Paper Takeaways" if (office_phase2 or turtle_phase2) else "# Phase-1 Paper Takeaways"
    lines.append(title)
    lines.append("")
    lines.append("## Main Result")
    lines.append(
        f"- On `office`, train-time compact reduces Gaussians by {office_train['gaussian_drop_pct']:.1f}% "
        f"with only {office_train['psnr_drop_db']:.3f} dB PSNR drop, and the final locality codec reduces stored "
        f"model size from {office_baseline['model_mb']:.2f} MB to {office_final['model_mb']:.2f} MB "
        f"({office_final['size_ratio_vs_baseline']:.2f}x smaller) with {office_final['psnr_drop_db']:.3f} dB PSNR drop."
    )
    lines.append(
        f"- On `turtlebot_pyebaekRoom_1_scene_1`, train-time compact reduces Gaussians by {turtle_train['gaussian_drop_pct']:.1f}% "
        f"with only {turtle_train['psnr_drop_db']:.3f} dB PSNR drop, and the final locality codec reduces stored "
        f"model size from {turtle_baseline['model_mb']:.2f} MB to {turtle_final['model_mb']:.2f} MB "
        f"({turtle_final['size_ratio_vs_baseline']:.2f}x smaller) with {turtle_final['psnr_drop_db']:.3f} dB PSNR drop."
    )
    lines.append("")
    lines.append("## Why It Works")
    lines.append(
        f"- The dominant storage term is `f_rest`. In `office`, locality residual reduces `f_rest` share from "
        f"{100.0 * as_float(office_fixed, 'f_rest_share'):.1f}% to {100.0 * as_float(office_locality, 'f_rest_share'):.1f}%."
    )
    lines.append(
        f"- In `turtlebot_pyebaekRoom_1_scene_1`, locality residual reduces `f_rest` share from "
        f"{100.0 * as_float(turtle_fixed, 'f_rest_share'):.1f}% to {100.0 * as_float(turtle_locality, 'f_rest_share'):.1f}%."
    )
    lines.append(
        f"- Adaptive mixed precision is actually used: the `int4` block ratio is "
        f"{100.0 * as_float(office_locality, 'f_rest_int4_block_ratio'):.1f}% on `office` and "
        f"{100.0 * as_float(turtle_locality, 'f_rest_int4_block_ratio'):.1f}% on `turtlebot_pyebaekRoom_1_scene_1`."
    )
    if office_phase2 or turtle_phase2:
        lines.append("")
        lines.append("## Phase 2 Readout")
        if office_phase2 and office_final:
            lines.append(
                f"- On `office`, the phase-2 residual field changes model size from {office_final['model_mb']:.2f} MB "
                f"to {office_phase2['model_mb']:.2f} MB and {delta_phrase(office_phase2['psnr'], office_final['psnr'], 'PSNR')} "
                f"relative to the phase-1 final compact."
            )
        if turtle_phase2 and turtle_final:
            lines.append(
                f"- On `turtlebot_pyebaekRoom_1_scene_1`, the phase-2 residual field changes model size from "
                f"{turtle_final['model_mb']:.2f} MB to {turtle_phase2['model_mb']:.2f} MB and "
                f"{delta_phrase(turtle_phase2['psnr'], turtle_final['psnr'], 'PSNR')} relative to the phase-1 final compact."
            )
    lines.append("")
    lines.append("## Ablation Readout")
    ablation_map = {row["variant"]: row for row in ablation_rows}
    prune_only = ablation_map["Prune only"]
    prune_sh = ablation_map["Prune + adaptive SH"]
    full_compact = ablation_map["Full compact train"]
    lines.append(
        f"- `Prune only` captures most of the training-side savings: {prune_only['gaussian_drop_pct']:.1f}% fewer Gaussians "
        f"than baseline with a {prune_only['psnr_delta_db']:+.3f} dB PSNR change."
    )
    lines.append(
        f"- Adding adaptive SH changes PSNR by {prune_sh['psnr_delta_db']:+.3f} dB relative to baseline and mostly prepares "
        f"`f_rest` for export-time compression rather than producing a large standalone gain."
    )
    lines.append(
        f"- The final locality codec is the main deployment-stage win: compared with the compact-trained PLY model, "
        f"it shrinks storage by {pct_reduction(office_train['model_mb'], office_final['model_mb']):.1f}% on `office` and "
        f"{pct_reduction(turtle_train['model_mb'], turtle_final['model_mb']):.1f}% on `turtlebot_pyebaekRoom_1_scene_1`."
    )
    lines.append("")
    lines.append("## Recommended Paper Story")
    lines.append("- Use `OmniGS baseline`, `Ours train-time compact`, and `Ours final compact` as the main table rows.")
    lines.append("- Put `export_shdrop` into supplementary material unless you want an explicit intermediate codec ablation.")
    lines.append("- Keep the ablation focused on `pruning`, `adaptive SH`, and `locality residual codec`.")
    if office_phase2 or turtle_phase2:
        lines.append("- Present `Ours phase-2 residual field` as the second-stage extension row, not as the core Phase-1 method.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Build paper-ready phase-1 tables and takeaways.")
    parser.add_argument("--phase1-main", default=str(ROOT / "results_smoke/phase1_main_table.csv"))
    parser.add_argument("--ablation", default=str(ROOT / "results_smoke/ablation_summary.csv"))
    parser.add_argument("--breakdown", default=str(ROOT / "results_smoke/formal_compare_compact_breakdown.csv"))
    parser.add_argument("--main-output", default=str(ROOT / "results_smoke/phase1_paper_main_table.csv"))
    parser.add_argument("--main-md-output", default=str(ROOT / "results_smoke/phase1_paper_main_table.md"))
    parser.add_argument("--ablation-output", default=str(ROOT / "results_smoke/phase1_paper_ablation_table.csv"))
    parser.add_argument("--ablation-md-output", default=str(ROOT / "results_smoke/phase1_paper_ablation_table.md"))
    parser.add_argument("--takeaways-output", default=str(ROOT / "results_smoke/phase1_paper_takeaways.md"))
    args = parser.parse_args()

    phase1_rows = load_rows(Path(args.phase1_main))
    ablation_rows = load_rows(Path(args.ablation))
    breakdown_rows = load_rows(Path(args.breakdown))

    main_rows = build_main_table(phase1_rows)
    ablation_table_rows = build_ablation_table(ablation_rows)
    takeaways = build_takeaways(main_rows, breakdown_rows, ablation_table_rows)

    main_fields = [
        "scene",
        "setting",
        "format",
        "gaussians",
        "gaussian_drop_pct",
        "model_mb",
        "size_drop_pct",
        "size_ratio_vs_baseline",
        "psnr",
        "psnr_drop_db",
        "dssim",
    ]
    ablation_fields = [
        "scene",
        "variant",
        "gaussians",
        "gaussian_drop_pct",
        "ply_mb",
        "compact_mb",
        "compact_ratio_vs_baseline_ply",
        "psnr",
        "psnr_delta_db",
    ]

    main_output = Path(args.main_output)
    ablation_output = Path(args.ablation_output)
    write_csv(main_output, main_rows, main_fields)
    write_markdown(Path(args.main_md_output), markdown_table(main_rows, main_fields))
    write_csv(ablation_output, ablation_table_rows, ablation_fields)
    write_markdown(Path(args.ablation_md_output), markdown_table(ablation_table_rows, ablation_fields))
    write_markdown(Path(args.takeaways_output), takeaways)

    print(f"[paper-phase1] wrote main table to {main_output}")
    print(f"[paper-phase1] wrote ablation table to {ablation_output}")
    print(f"[paper-phase1] wrote takeaways to {Path(args.takeaways_output)}")


if __name__ == "__main__":
    raise SystemExit(main())
