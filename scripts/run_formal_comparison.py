#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_OUTPUT = ROOT / "results_smoke/formal_compare_manifest.json"
DEFAULT_SUMMARY_OUTPUT = ROOT / "results_smoke/formal_compare_summary.csv"
DEFAULT_MARKDOWN_OUTPUT = ROOT / "results_smoke/formal_compare_summary.md"
DEFAULT_COMPACT_ANALYSIS_OUTPUT = ROOT / "results_smoke/formal_compare_compact_breakdown.csv"
DEFAULT_COMPACT_ANALYSIS_MARKDOWN_OUTPUT = ROOT / "results_smoke/formal_compare_compact_breakdown.md"
DEFAULT_EXPORT_BINARY = ROOT / "bin/export_compact"
DEFAULT_PREPARE_PHASE2_BINARY = ROOT / "bin/prepare_phase2"
DEFAULT_TRAIN_PHASE2_BINARY = ROOT / "bin/train_phase2_residual_field"
DEFAULT_FIXED_EXPORT_CFG = ROOT / "cfg/lonlat/compact_export_fixed.yaml"
DEFAULT_SHDROP_EXPORT_CFG = ROOT / "cfg/lonlat/compact_export_shdrop_aggressive.yaml"
DEFAULT_LOCALITY_EXPORT_CFG = ROOT / "cfg/lonlat/compact_export_locality_residual.yaml"


def runtime_env():
    env = dict(os.environ)
    ld_parts = [
        str(ROOT / "lib"),
        str(ROOT / ".deps/opencv-install/lib"),
        str(ROOT / ".deps/jsoncpp-install/lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join([part for part in ld_parts if part])
    return env


def iter_key(path: Path):
    checkpoint_name = path.parents[3].name
    numeric = 0
    for token in checkpoint_name.split("_"):
        if token.isdigit():
            numeric = int(token)
            break
    return (numeric, checkpoint_name.endswith("_shutdown"), checkpoint_name)


def latest_ply(result_dir: Path):
    ply_paths = sorted(
        result_dir.glob("*/ply/point_cloud/iteration_*/point_cloud.ply"),
        key=iter_key,
    )
    if not ply_paths:
        raise RuntimeError(f"no checkpoints found under {result_dir}")
    return ply_paths[-1]


def latest_iteration_dir(root_dir: Path):
    iteration_dirs = sorted(
        [path for path in root_dir.glob("iteration_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_")[-1]) if path.name.split("_")[-1].isdigit() else -1,
    )
    if not iteration_dirs:
        raise RuntimeError(f"no iteration directories found under {root_dir}")
    return iteration_dirs[-1]


def resolve_phase2_frozen_dir(checkpoint_dir: Path, expected_iteration_name: str):
    candidates = [
        checkpoint_dir / "ply/phase2" / expected_iteration_name,
        checkpoint_dir / "phase2" / expected_iteration_name,
        checkpoint_dir / "ply/phase2" / "iteration_0",
        checkpoint_dir / "phase2" / "iteration_0",
    ]
    for candidate in candidates:
        if (candidate / "metadata.json").exists():
            return candidate

    roots = [
        checkpoint_dir / "ply/phase2",
        checkpoint_dir / "phase2",
    ]
    for root in roots:
        if root.exists():
            try:
                candidate = latest_iteration_dir(root)
            except RuntimeError:
                continue
            if (candidate / "metadata.json").exists():
                return candidate

    raise RuntimeError(f"no valid phase2 frozen package found under {checkpoint_dir}")


def reexport_fixed(cfg_path: Path, result_dir: Path, env, force: bool):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/reexport_compact_fixed.py"),
        "--job",
        f"{result_dir.name}|{cfg_path}|{result_dir}",
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def eval_metrics_exist(eval_dir: Path):
    return (
        (eval_dir / "0_test/psnr.txt").exists()
        and (eval_dir / "0_test/dssim.txt").exists()
        and (eval_dir / "0_test/render_time.txt").exists()
    )


def experiment_entries(exp, result_dir: Path):
    try:
        ply_path = latest_ply(result_dir)
    except RuntimeError:
        return []

    checkpoint_dir = ply_path.parents[3]
    checkpoint_name = checkpoint_dir.name
    iteration_name = ply_path.parent.name
    compact_fixed_dir = checkpoint_dir / "ply/compact_fixed" / iteration_name

    rows = [
        {
            "label": f"{exp['name']}_ply",
            "scene": exp["name"],
            "variant": "ply",
            "train_root": str(result_dir),
            "model_path": str(ply_path),
            "ply_path": str(ply_path),
            "compact_dir": str(compact_fixed_dir if compact_fixed_dir.exists() else ""),
            "eval_dir": str(result_dir / f"{checkpoint_name}_eval_ply/0_test"),
        }
    ]
    if compact_fixed_dir.exists():
        rows.append(
            {
                "label": f"{exp['name']}_compact_fixed",
                "scene": exp["name"],
                "variant": "compact_fixed",
                "train_root": str(result_dir),
                "model_path": str(compact_fixed_dir),
                "ply_path": str(ply_path),
                "compact_dir": str(compact_fixed_dir),
                "eval_dir": str(result_dir / f"{checkpoint_name}_eval_compact_fixed/0_test"),
            }
        )
    compact_export_shdrop_dir = checkpoint_dir / "ply/compact_export_shdrop" / iteration_name
    if compact_export_shdrop_dir.exists():
        rows.append(
            {
                "label": f"{exp['name']}_compact_export_shdrop",
                "scene": exp["name"],
                "variant": "compact_export_shdrop",
                "train_root": str(result_dir),
                "model_path": str(compact_export_shdrop_dir),
                "ply_path": str(ply_path),
                "compact_dir": str(compact_export_shdrop_dir),
                "eval_dir": str(result_dir / f"{checkpoint_name}_eval_compact_export_shdrop/0_test"),
            }
        )
    compact_locality_dir = checkpoint_dir / "ply/compact_locality_residual" / iteration_name
    if compact_locality_dir.exists():
        rows.append(
            {
                "label": f"{exp['name']}_compact_locality_residual",
                "scene": exp["name"],
                "variant": "compact_locality_residual",
                "train_root": str(result_dir),
                "model_path": str(compact_locality_dir),
                "ply_path": str(ply_path),
                "compact_dir": str(compact_locality_dir),
                "eval_dir": str(result_dir / f"{checkpoint_name}_eval_compact_locality_residual/0_test"),
            }
        )
    phase2_field_dir = checkpoint_dir / "phase2_field" / iteration_name
    phase2_compact_dir = phase2_field_dir / "phase2_compact" / "iteration_0"
    phase2_decoded_dir = phase2_field_dir / "decoded_compact" / "iteration_0"
    if phase2_compact_dir.exists():
        rows.append(
            {
                "label": f"{exp['name']}_phase2_compact",
                "scene": exp["name"],
                "variant": "phase2_compact",
                "train_root": str(result_dir),
                "model_path": str(phase2_compact_dir),
                "ply_path": str(ply_path),
                "compact_dir": str(phase2_compact_dir),
                "eval_dir": str(result_dir / f"{checkpoint_name}_eval_phase2_compact/0_test"),
            }
        )
    elif phase2_decoded_dir.exists():
        rows.append(
            {
                "label": f"{exp['name']}_phase2_decoded_compact",
                "scene": exp["name"],
                "variant": "phase2_decoded_compact",
                "train_root": str(result_dir),
                "model_path": str(phase2_decoded_dir),
                "ply_path": str(ply_path),
                "compact_dir": str(phase2_decoded_dir),
                "eval_dir": str(result_dir / f"{checkpoint_name}_eval_phase2_decoded_compact/0_test"),
            }
        )
    return rows


def write_summary(
    experiments,
    manifest_output: Path,
    summary_output: Path,
    markdown_output: Path,
    compact_analysis_output: Path,
    compact_analysis_markdown_output: Path,
    env):
    rows = []
    for exp in experiments:
        result_dir = (ROOT / exp["result_dir"]).resolve()
        rows.extend(experiment_entries(exp, result_dir))

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(rows, indent=2) + "\n")

    if not rows:
        return

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_compact_results.py"),
            "--manifest",
            str(manifest_output),
            "--output",
            str(summary_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_compact_packages.py"),
            "--manifest",
            str(manifest_output),
            "--output",
            str(compact_analysis_output),
            "--markdown-output",
            str(compact_analysis_markdown_output),
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )


def run_one(exp, env, force_reexport, skip_existing):
    cfg_path = (ROOT / exp["cfg_path"]).resolve()
    dataset_path = (ROOT / exp["dataset_path"]).resolve()
    result_dir = (ROOT / exp["result_dir"]).resolve()
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    export_binary = DEFAULT_EXPORT_BINARY.resolve()
    prepare_phase2_binary = DEFAULT_PREPARE_PHASE2_BINARY.resolve()
    train_phase2_binary = Path(exp.get("phase2_binary", DEFAULT_TRAIN_PHASE2_BINARY)).resolve()

    if not skip_existing:
        train_cmd = [
            str((ROOT / exp["train_binary"]).resolve()),
            str(cfg_path),
            str(dataset_path),
            str(result_dir),
            "no_viewer",
        ]
        subprocess.run(train_cmd, check=True, cwd=ROOT, env=env)
    else:
        try:
            latest_ply(result_dir)
        except RuntimeError:
            train_cmd = [
                str((ROOT / exp["train_binary"]).resolve()),
                str(cfg_path),
                str(dataset_path),
                str(result_dir),
                "no_viewer",
            ]
            subprocess.run(train_cmd, check=True, cwd=ROOT, env=env)

    ply_path = latest_ply(result_dir)
    checkpoint_dir = ply_path.parents[3]
    checkpoint_name = checkpoint_dir.name
    iteration_name = ply_path.parent.name

    eval_ply_dir = result_dir / f"{checkpoint_name}_eval_ply"
    if force_reexport or not eval_metrics_exist(eval_ply_dir):
        subprocess.run(
            [
                str((ROOT / exp["eval_binary"]).resolve()),
                str(cfg_path),
                str(dataset_path),
                str(ply_path),
                str(eval_ply_dir),
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )

    compact_fixed_dir = checkpoint_dir / "ply/compact_fixed" / iteration_name
    if exp.get("export_compact_variants", False):
        if force_reexport or not compact_fixed_dir.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/reexport_compact_fixed.py"),
                    "--job",
                    f"{result_dir.name}|{DEFAULT_FIXED_EXPORT_CFG.resolve()}|{result_dir}",
                ] + (["--force"] if force_reexport else []),
                check=True,
                cwd=ROOT,
                env=env,
            )
        eval_compact_dir = result_dir / f"{checkpoint_name}_eval_compact_fixed"
        if force_reexport or not eval_metrics_exist(eval_compact_dir):
            subprocess.run(
                [
                    str((ROOT / exp["eval_binary"]).resolve()),
                    str(cfg_path),
                    str(dataset_path),
                    str(compact_fixed_dir),
                    str(eval_compact_dir),
                ],
                check=True,
                cwd=ROOT,
                env=env,
            )

        export_variants = [
            ("compact_export_shdrop", DEFAULT_SHDROP_EXPORT_CFG.resolve(), result_dir / f"{checkpoint_name}_eval_compact_export_shdrop"),
            ("compact_locality_residual", DEFAULT_LOCALITY_EXPORT_CFG.resolve(), result_dir / f"{checkpoint_name}_eval_compact_locality_residual"),
        ]
        for variant_name, export_cfg, eval_dir in export_variants:
            target_dir = checkpoint_dir / "ply" / variant_name / iteration_name
            tmp_root = checkpoint_dir / "ply" / f".{variant_name}_tmp"
            if force_reexport and target_dir.exists():
                subprocess.run(["rm", "-rf", str(target_dir)], check=True, cwd=ROOT, env=env)
            if force_reexport and tmp_root.exists():
                subprocess.run(["rm", "-rf", str(tmp_root)], check=True, cwd=ROOT, env=env)
            if force_reexport or not target_dir.exists():
                subprocess.run(
                    [str(export_binary), str(export_cfg), str(ply_path), str(tmp_root)],
                    check=True,
                    cwd=ROOT,
                    env=env,
                )
                produced_dir = tmp_root / "compact/iteration_0"
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rm", "-rf", str(target_dir)], check=False, cwd=ROOT, env=env)
                subprocess.run(["mv", str(produced_dir), str(target_dir)], check=True, cwd=ROOT, env=env)
                subprocess.run(["rm", "-rf", str(tmp_root)], check=True, cwd=ROOT, env=env)
            if force_reexport or not eval_metrics_exist(eval_dir):
                subprocess.run(
                    [
                        str((ROOT / exp["eval_binary"]).resolve()),
                        str(cfg_path),
                        str(dataset_path),
                        str(target_dir),
                        str(eval_dir),
                    ],
                    check=True,
                    cwd=ROOT,
                    env=env,
                )

    if exp.get("run_phase2_field", False):
        frozen_root = checkpoint_dir / "ply/phase2"
        if force_reexport and frozen_root.exists():
            shutil.rmtree(frozen_root, ignore_errors=True)
        try:
            frozen_dir = resolve_phase2_frozen_dir(checkpoint_dir, iteration_name)
        except RuntimeError:
            frozen_dir = None
        if force_reexport or frozen_dir is None:
            subprocess.run(
                [str(prepare_phase2_binary), str(cfg_path), str(ply_path), str(checkpoint_dir / "ply")],
                check=True,
                cwd=ROOT,
                env=env,
            )
            frozen_dir = resolve_phase2_frozen_dir(checkpoint_dir, iteration_name)

        phase2_output_dir = checkpoint_dir / "phase2_field" / iteration_name
        if force_reexport and phase2_output_dir.exists():
            shutil.rmtree(phase2_output_dir, ignore_errors=True)
        if force_reexport or not (phase2_output_dir / "training_summary.json").exists():
            phase2_output_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [str(train_phase2_binary), str(cfg_path), str(frozen_dir), str(phase2_output_dir)],
                check=True,
                cwd=ROOT,
                env=env,
            )

        phase2_compact_dir = phase2_output_dir / "phase2_compact" / "iteration_0"
        phase2_decoded_dir = phase2_output_dir / "decoded_compact" / "iteration_0"
        model_eval_path = phase2_compact_dir if phase2_compact_dir.exists() else phase2_decoded_dir
        eval_phase2_dir = result_dir / (
            f"{checkpoint_name}_eval_phase2_compact"
            if phase2_compact_dir.exists()
            else f"{checkpoint_name}_eval_phase2_decoded_compact"
        )
        if force_reexport or not eval_metrics_exist(eval_phase2_dir):
            subprocess.run(
                [
                    str((ROOT / exp["eval_binary"]).resolve()),
                    str(cfg_path),
                    str(dataset_path),
                    str(model_eval_path),
                    str(eval_phase2_dir),
                ],
                check=True,
                cwd=ROOT,
                env=env,
            )


def main():
    parser = argparse.ArgumentParser(description="Launch formal baseline/compact comparison experiments.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "scripts/formal_experiment_manifest.json"),
        help="Path to experiment manifest json",
    )
    parser.add_argument("--experiment", action="append", help="Only run selected experiment names")
    parser.add_argument("--force-reexport", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force retraining and reevaluation even if outputs exist")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--summary-manifest-output", default=str(DEFAULT_MANIFEST_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--summary-markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    parser.add_argument("--compact-analysis-output", default=str(DEFAULT_COMPACT_ANALYSIS_OUTPUT))
    parser.add_argument("--compact-analysis-markdown-output", default=str(DEFAULT_COMPACT_ANALYSIS_MARKDOWN_OUTPUT))
    args = parser.parse_args()

    experiments = json.loads(Path(args.manifest).read_text())
    selected = set(args.experiment or [])
    if selected:
        experiments = [exp for exp in experiments if exp["name"] in selected]

    env = runtime_env()
    manifest_output = Path(args.summary_manifest_output).expanduser().resolve()
    summary_output = Path(args.summary_output).expanduser().resolve()
    markdown_output = Path(args.summary_markdown_output).expanduser().resolve()
    compact_analysis_output = Path(args.compact_analysis_output).expanduser().resolve()
    compact_analysis_markdown_output = Path(args.compact_analysis_markdown_output).expanduser().resolve()

    for exp in experiments:
        print(f"[formal] start {exp['name']}")
        try:
            run_one(exp, env, args.force_reexport, skip_existing=not args.force)
            print(f"[formal] done {exp['name']}")
        except Exception as exc:
            print(f"[formal] failed {exp['name']}: {exc}", file=sys.stderr)
            traceback.print_exc()
            write_summary(
                experiments,
                manifest_output,
                summary_output,
                markdown_output,
                compact_analysis_output,
                compact_analysis_markdown_output,
                env)
            if not args.continue_on_error:
                raise
        write_summary(
            experiments,
            manifest_output,
            summary_output,
            markdown_output,
            compact_analysis_output,
            compact_analysis_markdown_output,
            env)

    write_summary(
        experiments,
        manifest_output,
        summary_output,
        markdown_output,
        compact_analysis_output,
        compact_analysis_markdown_output,
        env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
