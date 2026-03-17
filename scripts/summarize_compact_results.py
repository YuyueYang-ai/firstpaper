#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


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
    if not eval_dir:
        return {"avg_psnr": None, "avg_dssim": None, "avg_render_ms": None}
    return {
        "avg_psnr": avg_metric(eval_dir / "psnr.txt"),
        "avg_dssim": avg_metric(eval_dir / "dssim.txt"),
        "avg_render_ms": avg_metric(eval_dir / "render_time.txt"),
    }


def path_size(path: Path):
    if not path or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def read_num_points_from_ply(ply_path: Path):
    if not ply_path or not ply_path.exists():
        return None
    with ply_path.open("rb") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            text = line.decode("ascii", errors="ignore").strip()
            if text.startswith("element vertex "):
                return int(text.split()[-1])
            if text == "end_header":
                break
    return None


def compact_metadata(compact_dir: Path):
    if not compact_dir or not compact_dir.exists():
        return {}
    meta_path = compact_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def round_or_none(value, digits=6):
    return None if value is None else round(value, digits)


def normalize_entry(entry):
    result = dict(entry)
    for key in ["model_path", "ply_path", "compact_dir", "eval_dir", "train_root"]:
        if key in result and result[key]:
            result[key] = str(Path(result[key]).expanduser().resolve())
        else:
            result[key] = ""
    return result


def scan_result_root(result_root: Path):
    rows = []
    for ply_path in sorted(result_root.glob("*/ply/point_cloud/iteration_*/point_cloud.ply")):
        checkpoint_dir = ply_path.parents[3]
        iteration_name = ply_path.parent.name
        compact_fixed = checkpoint_dir / "ply/compact_fixed" / iteration_name
        compact = checkpoint_dir / "ply/compact" / iteration_name
        eval_dir = result_root / f"{checkpoint_dir.name}_test/0_test"
        rows.append(
            {
                "label": f"{result_root.name}:{checkpoint_dir.name}",
                "scene": result_root.name,
                "variant": "inventory",
                "train_root": str(result_root),
                "model_path": str(ply_path),
                "ply_path": str(ply_path),
                "compact_dir": str(compact_fixed if compact_fixed.exists() else compact if compact.exists() else ""),
                "eval_dir": str(eval_dir if eval_dir.exists() else ""),
            }
        )
    return rows


def load_entries(args):
    entries = []
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        entries.extend(manifest)
    for root in args.scan_root:
        entries.extend(scan_result_root(Path(root).expanduser().resolve()))
    return [normalize_entry(entry) for entry in entries]


def build_row(entry):
    model_path = Path(entry["model_path"]) if entry["model_path"] else None
    ply_path = Path(entry["ply_path"]) if entry["ply_path"] else None
    compact_dir = Path(entry["compact_dir"]) if entry["compact_dir"] else None
    eval_dir = Path(entry["eval_dir"]) if entry["eval_dir"] else None

    meta = compact_metadata(compact_dir)
    metrics = eval_metrics(eval_dir)
    ply_bytes = path_size(ply_path)
    compact_bytes = path_size(compact_dir)
    num_points = meta.get("num_points") or read_num_points_from_ply(ply_path)
    compact_kind = ""
    if compact_dir and compact_dir.exists():
        compact_kind = compact_dir.parents[0].name

    row = {
        "label": entry.get("label", ""),
        "scene": entry.get("scene", ""),
        "variant": entry.get("variant", ""),
        "train_root": entry.get("train_root", ""),
        "model_path": entry.get("model_path", ""),
        "model_format": "compact" if compact_dir and model_path and model_path.is_dir() else "ply",
        "ply_path": entry.get("ply_path", ""),
        "compact_dir": entry.get("compact_dir", ""),
        "compact_kind": compact_kind,
        "eval_dir": entry.get("eval_dir", ""),
        "num_points": num_points,
        "active_sh_degree": meta.get("active_sh_degree"),
        "rest_payload_values": meta.get("rest_payload_values"),
        "ply_bytes": ply_bytes,
        "compact_bytes": compact_bytes,
        "compression_ratio": round_or_none((ply_bytes / compact_bytes) if ply_bytes and compact_bytes else None, 4),
        "avg_psnr": round_or_none(metrics["avg_psnr"]),
        "avg_dssim": round_or_none(metrics["avg_dssim"]),
        "avg_render_ms": round_or_none(metrics["avg_render_ms"]),
    }
    return row


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
        "variant",
        "model_format",
        "num_points",
        "active_sh_degree",
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
    parser = argparse.ArgumentParser(description="Summarize PLY/compact evaluation results into CSV.")
    parser.add_argument("--manifest", help="JSON manifest describing explicit entries")
    parser.add_argument("--scan-root", action="append", default=[], help="Result root to inventory automatically")
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--markdown-output", help="Optional markdown table output path")
    args = parser.parse_args()

    entries = load_entries(args)
    rows = [build_row(entry) for entry in entries]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "scene",
        "variant",
        "train_root",
        "model_path",
        "model_format",
        "ply_path",
        "compact_dir",
        "compact_kind",
        "eval_dir",
        "num_points",
        "active_sh_degree",
        "rest_payload_values",
        "ply_bytes",
        "compact_bytes",
        "compression_ratio",
        "avg_psnr",
        "avg_dssim",
        "avg_render_ms",
    ]
    with output_path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[summary] wrote {len(rows)} rows to {output_path}")
    if args.markdown_output:
        markdown_path = Path(args.markdown_output).expanduser().resolve()
        write_markdown(rows, markdown_path)
        print(f"[summary] wrote markdown table to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
