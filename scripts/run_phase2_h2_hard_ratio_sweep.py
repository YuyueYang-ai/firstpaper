#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "scripts/phase2_h2_hard_ratio_manifest.json"
DEFAULT_PREPARE_CFG = ROOT / "cfg/lonlat/office_compact_phase2_h0_debug.yaml"
DEFAULT_TRAIN_CFG = ROOT / "cfg/lonlat/phase2_selective_hybrid_h2_smoke.yaml"
DEFAULT_EVAL_CFG = ROOT / "cfg/lonlat/office_compact_30k_formal.yaml"
DEFAULT_SOURCE_PLY = ROOT / "results/formal_compare/office_compact_phase2_30k/30081_shutdown/ply/point_cloud/iteration_30081/point_cloud.ply"
DEFAULT_DATASET = ROOT / "data/office"
DEFAULT_PREPARE_BINARY = ROOT / "bin/prepare_phase2"
DEFAULT_TRAIN_BINARY = ROOT / "bin/train_phase2_residual_field"
DEFAULT_EVAL_BINARY = ROOT / "bin/test_openmvg_lonlat"
DEFAULT_RESULTS_ROOT = ROOT / "results_smoke/phase2_h2_hard_ratio_sweep"


def runtime_env():
    env = dict(os.environ)
    ld_parts = [
        str(ROOT / "lib"),
        str(ROOT / ".deps/opencv-install/lib"),
        str(ROOT / ".deps/jsoncpp-install/lib"),
    ]
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        ld_parts.extend(
            [
                str(Path(conda_prefix) / "lib"),
                str(Path(conda_prefix) / "lib/python3.8/site-packages/torch/lib"),
            ]
        )
    else:
        fallback_conda = Path("/home/yunxiangyang/miniconda3/envs/ominiGS")
        ld_parts.extend(
            [
                str(fallback_conda / "lib"),
                str(fallback_conda / "lib/python3.8/site-packages/torch/lib"),
            ]
        )
    ld_parts.append(env.get("LD_LIBRARY_PATH", ""))
    env["LD_LIBRARY_PATH"] = ":".join(part for part in ld_parts if part)
    return env


def format_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, str):
        if value.startswith('"') and value.endswith('"'):
            return value
        return f"\"{value}\""
    raise TypeError(f"Unsupported config value type: {type(value)!r}")


def update_cfg_text(base_text: str, overrides):
    lines = base_text.splitlines()
    key_to_index = {}
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip()
        key_to_index[key] = idx

    for key, value in overrides.items():
        rendered = f"{key}: {format_value(value)}"
        if key in key_to_index:
            lines[key_to_index[key]] = rendered
        else:
            lines.append(rendered)
    return "\n".join(lines) + "\n"


def run_logged(cmd, log_path: Path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fout:
        fout.write("$ " + " ".join(cmd) + "\n")
        fout.flush()
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=fout,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.returncode


def avg_metric(path: Path):
    if not path.exists():
        return None
    values = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                values.append(float(parts[-1]))
            except ValueError:
                pass
    if not values:
        return None
    return sum(values) / len(values)


def path_size(path: Path):
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def round_or_none(value, digits=6):
    return None if value is None else round(value, digits)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value):
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def write_markdown(path: Path, rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(col)) for col in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_row(exp, prep_out: Path, train_out: Path, eval_out: Path, source_ply: Path):
    selector_summary = load_json(prep_out / "phase2/iteration_0/hybrid_selector_summary.json")
    training_summary = load_json(train_out / "training_summary.json")
    compact_dir = train_out / "phase2_compact/iteration_0"
    metrics_dir = eval_out / "0_test"
    compact_bytes = path_size(compact_dir)
    source_ply_bytes = path_size(source_ply)
    return {
        "name": exp["name"],
        "hard_point_ratio": exp["hard_point_ratio"],
        "notes": exp.get("notes", ""),
        "status": "done" if compact_dir.exists() and (metrics_dir / "psnr.txt").exists() else "failed",
        "num_blocks": selector_summary.get("num_blocks"),
        "num_hard_blocks": selector_summary.get("num_hard_blocks"),
        "num_hard_points": selector_summary.get("num_hard_points"),
        "realized_hard_point_ratio": round_or_none(selector_summary.get("realized_hard_point_ratio"), 4),
        "mean_block_mse_hard": round_or_none(selector_summary.get("mean_block_mse_hard")),
        "mean_block_mse_easy": round_or_none(selector_summary.get("mean_block_mse_easy")),
        "mean_block_explicit_bpp_hard": round_or_none(selector_summary.get("mean_block_explicit_bpp_hard")),
        "mean_block_explicit_bpp_easy": round_or_none(selector_summary.get("mean_block_explicit_bpp_easy")),
        "best_eval_mse": round_or_none(training_summary.get("best_eval_mse")),
        "train_point_count": training_summary.get("train_point_count"),
        "compact_bytes": compact_bytes,
        "compression_ratio_vs_source": round_or_none((source_ply_bytes / compact_bytes) if source_ply_bytes and compact_bytes else None, 4),
        "avg_psnr": round_or_none(avg_metric(metrics_dir / "psnr.txt")),
        "avg_dssim": round_or_none(avg_metric(metrics_dir / "dssim.txt")),
        "avg_render_ms": round_or_none(avg_metric(metrics_dir / "render_time.txt")),
        "prep_dir": str(prep_out / "phase2/iteration_0"),
        "train_dir": str(train_out),
        "compact_dir": str(compact_dir),
        "eval_dir": str(metrics_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Run H2 hard-point-ratio sweep on office.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--prepare-cfg", default=str(DEFAULT_PREPARE_CFG))
    parser.add_argument("--train-cfg", default=str(DEFAULT_TRAIN_CFG))
    parser.add_argument("--eval-cfg", default=str(DEFAULT_EVAL_CFG))
    parser.add_argument("--source-ply", default=str(DEFAULT_SOURCE_PLY))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--prepare-binary", default=str(DEFAULT_PREPARE_BINARY))
    parser.add_argument("--train-binary", default=str(DEFAULT_TRAIN_BINARY))
    parser.add_argument("--eval-binary", default=str(DEFAULT_EVAL_BINARY))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    prepare_cfg_path = Path(args.prepare_cfg).resolve()
    train_cfg_path = Path(args.train_cfg).resolve()
    eval_cfg_path = Path(args.eval_cfg).resolve()
    source_ply = Path(args.source_ply).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    prepare_binary = Path(args.prepare_binary).resolve()
    train_binary = Path(args.train_binary).resolve()
    eval_binary = Path(args.eval_binary).resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_root = Path(args.results_root).resolve().parent / f"{Path(args.results_root).name}_{timestamp}"
    configs_dir = results_root / "generated_configs"
    logs_dir = results_root / "logs"
    env = runtime_env()

    base_prepare_text = prepare_cfg_path.read_text()
    base_train_text = train_cfg_path.read_text()
    rows = []
    status_records = []

    for exp in manifest:
        name = exp["name"]
        ratio = exp["hard_point_ratio"]
        run_root = results_root / name
        prep_out = run_root / "prepare"
        train_out = run_root / "train"
        eval_out = run_root / "eval"
        cfg_out = configs_dir / f"{name}.prepare.yaml"
        train_cfg_out = configs_dir / f"{name}.train.yaml"

        cfg_overrides = {
            "Phase2HybridSelector.enable": 1,
            "Phase2HybridSelector.hard_point_ratio": ratio,
        }
        optional_selector_keys = [
            "alpha",
            "beta",
            "gamma",
            "delta",
            "min_hard_blocks",
            "max_hard_blocks",
            "save_debug_tensors",
            "explicit_cost_int4_rel_mse_threshold",
        ]
        for key in optional_selector_keys:
            if key in exp:
                cfg_overrides[f"Phase2HybridSelector.{key}"] = exp[key]
        if "compression_f_rest_locality_int4_rel_mse_threshold" in exp:
            cfg_overrides["Compression.f_rest_locality_int4_rel_mse_threshold"] = exp[
                "compression_f_rest_locality_int4_rel_mse_threshold"
            ]

        cfg_text = update_cfg_text(base_prepare_text, cfg_overrides)
        cfg_out.parent.mkdir(parents=True, exist_ok=True)
        cfg_out.write_text(cfg_text)

        train_cfg_overrides = exp.get("train_cfg_overrides", {})
        train_cfg_text = update_cfg_text(base_train_text, train_cfg_overrides) if train_cfg_overrides else base_train_text
        train_cfg_out.write_text(train_cfg_text)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "config.json").write_text(json.dumps(exp, indent=2, ensure_ascii=False) + "\n")

        if args.force:
            subprocess.run(["rm", "-rf", str(run_root)], check=False)

        run_status = {
            "name": name,
            "hard_point_ratio": ratio,
            "status": "running",
            "run_root": str(run_root),
        }
        status_records.append(run_status)

        prep_cmd = [str(prepare_binary), str(cfg_out), str(source_ply), str(prep_out)]
        prep_log = logs_dir / f"{name}.prepare.log"
        code = run_logged(prep_cmd, prep_log, env)
        if code != 0:
            run_status["status"] = "prepare_failed"
            run_status["prepare_log"] = str(prep_log)
            continue

        train_cmd = [str(train_binary), str(train_cfg_out), str(prep_out / "phase2/iteration_0"), str(train_out)]
        train_log = logs_dir / f"{name}.train.log"
        code = run_logged(train_cmd, train_log, env)
        if code != 0:
            run_status["status"] = "train_failed"
            run_status["train_log"] = str(train_log)
            continue

        eval_cmd = [str(eval_binary), str(eval_cfg_path), str(dataset_path), str(train_out / "phase2_compact/iteration_0"), str(eval_out)]
        eval_log = logs_dir / f"{name}.eval.log"
        code = run_logged(eval_cmd, eval_log, env)
        if code != 0:
            run_status["status"] = "eval_failed"
            run_status["eval_log"] = str(eval_log)
            continue

        run_status["status"] = "done"
        rows.append(build_row(exp, prep_out, train_out, eval_out, source_ply))

    summary_csv = results_root / "summary.csv"
    summary_md = results_root / "summary.md"
    status_json = results_root / "status.json"
    fieldnames = [
        "name",
        "hard_point_ratio",
        "notes",
        "status",
        "num_blocks",
        "num_hard_blocks",
        "num_hard_points",
        "realized_hard_point_ratio",
        "mean_block_mse_hard",
        "mean_block_mse_easy",
        "mean_block_explicit_bpp_hard",
        "mean_block_explicit_bpp_easy",
        "best_eval_mse",
        "train_point_count",
        "compact_bytes",
        "compression_ratio_vs_source",
        "avg_psnr",
        "avg_dssim",
        "avg_render_ms",
        "prep_dir",
        "train_dir",
        "compact_dir",
        "eval_dir",
    ]
    write_csv(summary_csv, rows, fieldnames)
    write_markdown(summary_md, rows, fieldnames[:19])
    status_json.write_text(json.dumps(status_records, indent=2, ensure_ascii=False) + "\n")

    print(f"results_root={results_root}")
    print(f"summary_csv={summary_csv}")
    print(f"summary_md={summary_md}")
    print(f"status_json={status_json}")


if __name__ == "__main__":
    main()
