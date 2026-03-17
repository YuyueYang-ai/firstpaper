#!/usr/bin/env python3

import argparse
import csv
import json
import struct
from pathlib import Path


def avg_metric(path: Path):
    if not path.exists():
        return None
    values = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(float(line.split()[-1]))
    if not values:
        return None
    return sum(values) / len(values)


def eval_metrics(eval_dir: Path):
    if not eval_dir.exists():
        return {"avg_psnr": None, "avg_dssim": None, "avg_render_ms": None}
    return {
        "avg_psnr": avg_metric(eval_dir / "psnr.txt"),
        "avg_dssim": avg_metric(eval_dir / "dssim.txt"),
        "avg_render_ms": avg_metric(eval_dir / "render_time.txt"),
    }


def path_size(path: Path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def read_sh_levels(compact_dir: Path, num_points: int):
    path = compact_dir / "sh_levels.bin"
    if not path.exists():
        return []
    raw = path.read_bytes()
    if num_points <= 0:
        return []
    if len(raw) == num_points:
        return list(raw)
    if len(raw) == num_points * 4:
        return [value[0] for value in struct.iter_unpack("<i", raw)]
    return []


def format_cell(value):
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def write_markdown(rows, output_path: Path):
    columns = [
        "label",
        "scene",
        "num_points",
        "active_sh_degree",
        "f_rest_representation",
        "f_rest_quantization_mode",
        "f_rest_block_size_points",
        "f_rest_high_sh_block_size_points",
        "f_rest_low_sh_block_size_points",
        "f_rest_int4_block_ratio",
        "mean_sh_level",
        "dominant_sh_level",
        "sh_level_0_pct",
        "sh_level_1_pct",
        "sh_level_2_pct",
        "sh_level_3_pct",
        "rest_payload_ratio",
        "compact_bytes",
        "f_rest_share",
        "rotation_share",
        "xyz_share",
        "avg_psnr",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def round_or_none(value, digits=6):
    return None if value is None else round(value, digits)


def analyze_entry(entry):
    compact_dir = Path(entry["compact_dir"]).expanduser().resolve()
    eval_dir = Path(entry["eval_dir"]).expanduser().resolve() if entry.get("eval_dir") else None
    meta = json.loads((compact_dir / "metadata.json").read_text())
    meta_format = meta.get("format", "")

    num_points = int(meta["num_points"])
    max_sh_degree = int(meta["max_sh_degree"])
    active_sh_degree = int(meta["active_sh_degree"])
    sh_levels = read_sh_levels(compact_dir, num_points)
    if meta_format == "phase2_residual_field_compact":
        rest_payload_values = sum(max(0, ((level + 1) * (level + 1) - 1) * 3) for level in sh_levels)
    else:
        rest_payload_values = int(meta["rest_payload_values"])
    max_rest_payload_values = num_points * (((max_sh_degree + 1) * (max_sh_degree + 1)) - 1) * 3
    rest_payload_ratio = (rest_payload_values / max_rest_payload_values) if max_rest_payload_values else None
    if meta_format == "phase2_residual_field_compact":
        f_rest_representation = meta.get("representation", "anchors_plus_field_weights")
        f_rest_quantization_mode = "block_base_fp16+field_weights"
        f_rest_block_size_points = None
        f_rest_block_count = meta.get("features_rest_block_bases_shape", [0])[0]
        f_rest_high_sh_block_size_points = meta.get("locality_high_sh_block_size")
        f_rest_low_sh_block_size_points = meta.get("locality_low_sh_block_size")
        f_rest_int4_block_count = None
        f_rest_int8_block_count = None
    else:
        f_rest_meta = meta.get("f_rest", {})
        f_rest_representation = f_rest_meta.get("representation", "direct_payload")
        f_rest_quantization_mode = f_rest_meta.get("quantization_mode", "global_1d")
        f_rest_block_size_points = f_rest_meta.get("block_size_points")
        f_rest_block_count = f_rest_meta.get("block_count")
        f_rest_high_sh_block_size_points = f_rest_meta.get("high_sh_block_size_points")
        f_rest_low_sh_block_size_points = f_rest_meta.get("low_sh_block_size_points")
        f_rest_int4_block_count = f_rest_meta.get("int4_block_count")
        f_rest_int8_block_count = f_rest_meta.get("int8_block_count")
    if f_rest_int4_block_count is not None and f_rest_block_count:
        f_rest_int4_block_ratio = f_rest_int4_block_count / f_rest_block_count
    else:
        f_rest_int4_block_ratio = None

    counts = [0] * (max_sh_degree + 1)
    for level in sh_levels:
        if 0 <= level <= max_sh_degree:
            counts[level] += 1
    total_levels = sum(counts)
    fractions = [(count / total_levels) if total_levels else 0.0 for count in counts]
    mean_sh_level = sum(level * counts[level] for level in range(len(counts))) / total_levels if total_levels else None
    dominant_sh_level = max(range(len(counts)), key=lambda idx: counts[idx]) if total_levels else None

    file_sizes = {}
    for name in [
        "xyz.bin",
        "f_dc.bin",
        "features_dc.bin",
        "f_rest.bin",
        "features_rest_block_bases.bin",
        "f_rest_block_mins.bin",
        "f_rest_block_maxs.bin",
        "f_rest_block_levels.bin",
        "f_rest_block_point_counts.bin",
        "f_rest_block_bits.bin",
        "f_rest_block_base.bin",
        "f_rest_block_scale.bin",
        "opacity.bin",
        "scaling.bin",
        "rotation.bin",
        "sh_levels.bin",
        "xyz_normalized.bin",
        "field_weights.pt",
        "metadata.json",
    ]:
        file_sizes[name] = path_size(compact_dir / name)
    compact_bytes = path_size(compact_dir)

    def share(name):
        return (file_sizes[name] / compact_bytes) if compact_bytes else None

    metrics = eval_metrics(eval_dir / "0_test") if eval_dir and (eval_dir / "0_test").exists() else (
        eval_metrics(eval_dir) if eval_dir else {"avg_psnr": None, "avg_dssim": None, "avg_render_ms": None}
    )

    row = {
        "label": entry["label"],
        "scene": entry["scene"],
        "variant": entry["variant"],
        "compact_dir": str(compact_dir),
        "num_points": num_points,
        "max_sh_degree": max_sh_degree,
        "active_sh_degree": active_sh_degree,
        "mean_sh_level": round_or_none(mean_sh_level, 4),
        "dominant_sh_level": dominant_sh_level,
        "rest_payload_values": rest_payload_values,
        "max_rest_payload_values": max_rest_payload_values,
        "rest_payload_ratio": round_or_none(rest_payload_ratio, 6),
        "f_rest_representation": f_rest_representation,
        "f_rest_quantization_mode": f_rest_quantization_mode,
        "f_rest_block_size_points": f_rest_block_size_points,
        "f_rest_block_count": f_rest_block_count,
        "f_rest_high_sh_block_size_points": f_rest_high_sh_block_size_points,
        "f_rest_low_sh_block_size_points": f_rest_low_sh_block_size_points,
        "f_rest_int4_block_count": f_rest_int4_block_count,
        "f_rest_int8_block_count": f_rest_int8_block_count,
        "f_rest_int4_block_ratio": round_or_none(f_rest_int4_block_ratio),
        "compact_bytes": compact_bytes,
        "xyz_bytes": file_sizes["xyz.bin"],
        "f_dc_bytes": max(file_sizes["f_dc.bin"], file_sizes["features_dc.bin"]),
        "f_rest_bytes": file_sizes["f_rest.bin"],
        "f_rest_block_mins_bytes": file_sizes["f_rest_block_mins.bin"],
        "f_rest_block_maxs_bytes": file_sizes["f_rest_block_maxs.bin"],
        "f_rest_block_levels_bytes": file_sizes["f_rest_block_levels.bin"],
        "f_rest_block_point_counts_bytes": file_sizes["f_rest_block_point_counts.bin"],
        "f_rest_block_bits_bytes": file_sizes["f_rest_block_bits.bin"],
        "f_rest_block_base_bytes": max(file_sizes["f_rest_block_base.bin"], file_sizes["features_rest_block_bases.bin"]),
        "f_rest_block_scale_bytes": file_sizes["f_rest_block_scale.bin"],
        "opacity_bytes": file_sizes["opacity.bin"],
        "scaling_bytes": file_sizes["scaling.bin"],
        "rotation_bytes": file_sizes["rotation.bin"],
        "sh_levels_bytes": file_sizes["sh_levels.bin"],
        "xyz_normalized_bytes": file_sizes["xyz_normalized.bin"],
        "field_weights_bytes": file_sizes["field_weights.pt"],
        "metadata_bytes": file_sizes["metadata.json"],
        "xyz_share": round_or_none(share("xyz.bin")),
        "f_dc_share": round_or_none(max(file_sizes["f_dc.bin"], file_sizes["features_dc.bin"]) / compact_bytes if compact_bytes else None),
        "f_rest_share": round_or_none(share("f_rest.bin")),
        "f_rest_block_mins_share": round_or_none(share("f_rest_block_mins.bin")),
        "f_rest_block_maxs_share": round_or_none(share("f_rest_block_maxs.bin")),
        "f_rest_block_levels_share": round_or_none(share("f_rest_block_levels.bin")),
        "f_rest_block_point_counts_share": round_or_none(share("f_rest_block_point_counts.bin")),
        "f_rest_block_bits_share": round_or_none(share("f_rest_block_bits.bin")),
        "f_rest_block_base_share": round_or_none(max(file_sizes["f_rest_block_base.bin"], file_sizes["features_rest_block_bases.bin"]) / compact_bytes if compact_bytes else None),
        "f_rest_block_scale_share": round_or_none(share("f_rest_block_scale.bin")),
        "opacity_share": round_or_none(share("opacity.bin")),
        "scaling_share": round_or_none(share("scaling.bin")),
        "rotation_share": round_or_none(share("rotation.bin")),
        "sh_levels_share": round_or_none(share("sh_levels.bin")),
        "metadata_share": round_or_none(share("metadata.json")),
        "avg_psnr": round_or_none(metrics["avg_psnr"]),
        "avg_dssim": round_or_none(metrics["avg_dssim"]),
        "avg_render_ms": round_or_none(metrics["avg_render_ms"]),
    }

    for level in range(max_sh_degree + 1):
        row[f"sh_level_{level}_count"] = counts[level]
        row[f"sh_level_{level}_pct"] = round_or_none(fractions[level], 6)
    return row


def main():
    parser = argparse.ArgumentParser(description="Analyze compact package composition and SH-level distribution.")
    parser.add_argument("--manifest", required=True, help="JSON manifest generated by the formal runner")
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--markdown-output", help="Optional markdown output path")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    entries = []
    for entry in manifest:
        compact_dir = entry.get("compact_dir")
        model_path = entry.get("model_path")
        if not compact_dir or not model_path:
            continue
        if Path(model_path).expanduser().resolve().is_dir():
            entries.append(entry)
    rows = [analyze_entry(entry) for entry in entries]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    print(f"[compact-analysis] wrote {len(rows)} rows to {output_path}")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).expanduser().resolve()
        write_markdown(rows, markdown_path)
        print(f"[compact-analysis] wrote markdown table to {markdown_path}")


if __name__ == "__main__":
    raise SystemExit(main())
