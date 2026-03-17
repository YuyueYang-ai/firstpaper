#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "scripts/phase2_v3_sweep_manifest.json"
DEFAULT_BASE_CFG = ROOT / "cfg/lonlat/office_compact_phase2_30k_formal.yaml"
DEFAULT_FROZEN_DIR = ROOT / "results_smoke/office_phase2_v3_frozen/phase2/iteration_0"
DEFAULT_DATASET = ROOT / "data/office"
DEFAULT_RESULTS_ROOT = ROOT / "results_smoke/phase2_v3_sweep"
DEFAULT_TRAIN_BINARY = ROOT / "bin/train_phase2_residual_field"
DEFAULT_EVAL_BINARY = ROOT / "bin/test_openmvg_lonlat"
DEFAULT_SOURCE_PLY = ROOT / "results/office_compact_30k/30081_shutdown/ply/point_cloud/iteration_30081/point_cloud.ply"


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


def path_size(path: Path):
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def read_metadata(compact_dir: Path):
    meta_path = compact_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def read_training_summary(path: Path):
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
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_logged(cmd, log_path: Path, env, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.flush()
        if dry_run:
            log_file.write("[dry-run]\n")
            return 0
        process = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return process.returncode


def experiment_done(result_dir: Path, eval_dir: Path):
    summary_path = result_dir / "training_summary.json"
    compact_dir = result_dir / "phase2_compact" / "iteration_0"
    return (
        summary_path.exists()
        and compact_dir.exists()
        and (eval_dir / "0_test/psnr.txt").exists()
        and (eval_dir / "0_test/dssim.txt").exists()
        and (eval_dir / "0_test/render_time.txt").exists()
    )


def build_row(exp, result_dir: Path, eval_dir: Path, source_ply: Path | None):
    summary = read_training_summary(result_dir / "training_summary.json")
    compact_dir = result_dir / "phase2_compact" / "iteration_0"
    meta = read_metadata(compact_dir)
    eval_leaf = eval_dir / "0_test"
    compact_bytes = path_size(compact_dir)
    source_ply_bytes = path_size(source_ply) if source_ply and source_ply.exists() else None
    return {
        "name": exp["name"],
        "scene": exp.get("scene", ""),
        "notes": exp.get("notes", ""),
        "status": "done" if compact_dir.exists() else "missing_output",
        "hidden_dim": exp["overrides"].get("Phase2Field.hidden_dim"),
        "num_hidden_layers": exp["overrides"].get("Phase2Field.num_hidden_layers"),
        "block_embedding_dim": exp["overrides"].get("Phase2Field.block_embedding_dim"),
        "learning_rate": exp["overrides"].get("Phase2Field.learning_rate"),
        "hashgrid_num_levels": exp["overrides"].get("Phase2Field.hashgrid_num_levels"),
        "hashgrid_features_per_level": exp["overrides"].get("Phase2Field.hashgrid_features_per_level"),
        "hashgrid_log2_hashmap_size": exp["overrides"].get("Phase2Field.hashgrid_log2_hashmap_size"),
        "hashgrid_base_resolution": exp["overrides"].get("Phase2Field.hashgrid_base_resolution"),
        "hashgrid_per_level_scale": exp["overrides"].get("Phase2Field.hashgrid_per_level_scale"),
        "max_steps": summary.get("max_steps"),
        "trained_steps": summary.get("trained_steps"),
        "best_eval_mse": round_or_none(summary.get("best_eval_mse")),
        "final_eval_mse": round_or_none(summary.get("final_eval_mse")),
        "num_points": meta.get("num_points") or summary.get("num_points"),
        "compact_bytes": compact_bytes,
        "source_ply_bytes": source_ply_bytes,
        "compression_ratio_vs_source": round_or_none((source_ply_bytes / compact_bytes) if source_ply_bytes and compact_bytes else None, 4),
        "avg_psnr": round_or_none(avg_metric(eval_leaf / "psnr.txt")),
        "avg_dssim": round_or_none(avg_metric(eval_leaf / "dssim.txt")),
        "avg_render_ms": round_or_none(avg_metric(eval_leaf / "render_time.txt")),
        "result_dir": str(result_dir),
        "compact_dir": str(compact_dir),
        "eval_dir": str(eval_leaf),
    }


def load_manifest(path: Path):
    return json.loads(path.read_text())


def write_status_json(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run 8 Phase2-v3 sweep experiments sequentially and summarize results.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="JSON manifest with the 8 experiment definitions")
    parser.add_argument("--base-cfg", default=str(DEFAULT_BASE_CFG), help="Base YAML config used to generate per-run configs")
    parser.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR), help="Phase2 frozen package directory")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET), help="Dataset path for renderer evaluation")
    parser.add_argument("--train-binary", default=str(DEFAULT_TRAIN_BINARY), help="Phase2 field training binary")
    parser.add_argument("--eval-binary", default=str(DEFAULT_EVAL_BINARY), help="Renderer evaluation binary")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT), help="Root directory for sweep outputs")
    parser.add_argument("--source-ply", default=str(DEFAULT_SOURCE_PLY), help="Optional reference PLY for compression ratio reporting")
    parser.add_argument("--experiment", action="append", default=[], help="Run only selected experiment names; can repeat")
    parser.add_argument("--override-max-steps", type=int, help="Override Phase2Field.max_steps for all runs")
    parser.add_argument("--dry-run", action="store_true", help="Write generated configs and commands without executing")
    parser.add_argument("--force", action="store_true", help="Rerun even if outputs already exist")
    parser.add_argument("--summary-output", help="CSV summary path; defaults inside results-root")
    parser.add_argument("--markdown-output", help="Markdown summary path; defaults inside results-root")
    parser.add_argument("--status-output", help="JSON status path; defaults inside results-root")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    base_cfg_path = Path(args.base_cfg).expanduser().resolve()
    frozen_dir = Path(args.frozen_dir).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    train_binary = Path(args.train_binary).expanduser().resolve()
    eval_binary = Path(args.eval_binary).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    source_ply = Path(args.source_ply).expanduser().resolve() if args.source_ply else None

    summary_output = Path(args.summary_output).expanduser().resolve() if args.summary_output else results_root / "summary.csv"
    markdown_output = Path(args.markdown_output).expanduser().resolve() if args.markdown_output else results_root / "summary.md"
    status_output = Path(args.status_output).expanduser().resolve() if args.status_output else results_root / "status.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base cfg not found: {base_cfg_path}")
    if not frozen_dir.exists():
        raise FileNotFoundError(f"frozen package not found: {frozen_dir}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset path not found: {dataset_path}")
    if not train_binary.exists():
        raise FileNotFoundError(f"train binary not found: {train_binary}")
    if not eval_binary.exists():
        raise FileNotFoundError(f"eval binary not found: {eval_binary}")

    experiments = load_manifest(manifest_path)
    if args.experiment:
        wanted = set(args.experiment)
        experiments = [exp for exp in experiments if exp["name"] in wanted]

    if not experiments:
        raise RuntimeError("no experiments selected")

    env = runtime_env()
    base_text = base_cfg_path.read_text()
    results_root.mkdir(parents=True, exist_ok=True)
    config_root = results_root / "configs"
    log_root = results_root / "logs"
    config_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    common_overrides = {
        "Phase2Field.max_steps": args.override_max_steps if args.override_max_steps is not None else 10000,
        "Phase2Field.log_interval": 500,
        "Phase2Field.eval_interval": 1000,
        "Phase2Field.batch_size": 8192,
        "Phase2Field.save_decoded_compact": 0,
        "Phase2Field.save_phase2_compact": 1,
        "Phase2Field.phase2_compact_pack_sh_levels": 1,
        "Phase2Field.phase2_compact_fdc_quant_bits": 8,
        "Phase2Field.phase2_compact_use_geometry_codec": 1,
        "Phase2Field.phase2_compact_geometry_quant_bits": 16,
    }

    rows = []
    status_records = []

    for index, exp in enumerate(experiments, start=1):
        name = exp["name"]
        result_dir = results_root / name
        eval_dir = result_dir / "eval"
        cfg_path = config_root / f"{name}.yaml"
        train_log = log_root / f"{name}.train.log"
        eval_log = log_root / f"{name}.eval.log"

        overrides = dict(common_overrides)
        overrides.update(exp.get("overrides", {}))
        cfg_path.write_text(update_cfg_text(base_text, overrides))

        record = {
            "name": name,
            "scene": exp.get("scene", ""),
            "notes": exp.get("notes", ""),
            "status": "pending",
            "config_path": str(cfg_path),
            "result_dir": str(result_dir),
            "eval_dir": str(eval_dir),
        }
        write_status_json(status_output, status_records + [record])

        if not args.force and experiment_done(result_dir, eval_dir):
            record["status"] = "skipped_existing"
            rows.append(build_row(exp, result_dir, eval_dir, source_ply))
            status_records.append(record)
            write_status_json(status_output, status_records)
            continue

        print(f"[phase2-sweep] ({index}/{len(experiments)}) {name}")
        train_cmd = [str(train_binary), str(cfg_path), str(frozen_dir), str(result_dir)]
        eval_cmd = [str(eval_binary), str(cfg_path), str(dataset_path), str(result_dir / "phase2_compact/iteration_0"), str(eval_dir)]

        record["status"] = "running_train"
        write_status_json(status_output, status_records + [record])
        start = time.time()
        train_rc = run_logged(train_cmd, train_log, env, dry_run=args.dry_run)
        if train_rc != 0:
            record["status"] = "train_failed"
            record["elapsed_sec"] = round(time.time() - start, 3)
            status_records.append(record)
            write_status_json(status_output, status_records)
            raise RuntimeError(f"{name} train failed, see {train_log}")

        record["status"] = "running_eval"
        write_status_json(status_output, status_records + [record])
        eval_rc = run_logged(eval_cmd, eval_log, env, dry_run=args.dry_run)
        if eval_rc != 0:
            record["status"] = "eval_failed"
            record["elapsed_sec"] = round(time.time() - start, 3)
            status_records.append(record)
            write_status_json(status_output, status_records)
            raise RuntimeError(f"{name} eval failed, see {eval_log}")

        record["status"] = "done"
        record["elapsed_sec"] = round(time.time() - start, 3)
        status_records.append(record)
        write_status_json(status_output, status_records)
        rows.append(build_row(exp, result_dir, eval_dir, source_ply))

        fieldnames = list(rows[0].keys()) if rows else []
        write_csv(summary_output, rows, fieldnames)
        write_markdown(
            markdown_output,
            rows,
            [
                "name",
                "hidden_dim",
                "num_hidden_layers",
                "block_embedding_dim",
                "learning_rate",
                "hashgrid_num_levels",
                "hashgrid_features_per_level",
                "best_eval_mse",
                "compact_bytes",
                "compression_ratio_vs_source",
                "avg_psnr",
                "avg_dssim",
                "avg_render_ms",
                "status",
            ],
        )

    if rows:
        fieldnames = list(rows[0].keys())
        write_csv(summary_output, rows, fieldnames)
        write_markdown(
            markdown_output,
            rows,
            [
                "name",
                "hidden_dim",
                "num_hidden_layers",
                "block_embedding_dim",
                "learning_rate",
                "hashgrid_num_levels",
                "hashgrid_features_per_level",
                "best_eval_mse",
                "compact_bytes",
                "compression_ratio_vs_source",
                "avg_psnr",
                "avg_dssim",
                "avg_render_ms",
                "status",
            ],
        )

    print(f"[phase2-sweep] wrote status to {status_output}")
    print(f"[phase2-sweep] wrote summary to {summary_output}")
    print(f"[phase2-sweep] wrote markdown to {markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
