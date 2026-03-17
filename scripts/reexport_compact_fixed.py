#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JOBS = [
    {
        "name": "office_compact_30k",
        "cfg": ROOT / "cfg/lonlat/office_compact_30k.yaml",
        "result_root": ROOT / "results/office_compact_30k",
    },
    {
        "name": "turtlebot_pyebaekRoom_1_scene_1_compact_32000",
        "cfg": ROOT / "cfg/lonlat/omniscenes_turtlebot_pyebaekRoom_32000.yaml",
        "result_root": ROOT / "results/turtlebot_pyebaekRoom_1_scene_1_compact_32000",
    },
]


def parse_job(text: str):
    parts = text.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("job must be name|cfg_path|result_root")
    return {
        "name": parts[0],
        "cfg": Path(parts[1]).expanduser().resolve(),
        "result_root": Path(parts[2]).expanduser().resolve(),
    }


def runtime_env():
    env = dict(**__import__("os").environ)
    lib_parts = [
        str(ROOT / "lib"),
        str(ROOT / ".deps/opencv-install/lib"),
        str(ROOT / ".deps/jsoncpp-install/lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join([part for part in lib_parts if part])
    return env


def iter_key(path: Path):
    checkpoint_name = path.parents[3].name
    numeric = 0
    for token in checkpoint_name.split("_"):
        if token.isdigit():
            numeric = int(token)
            break
    return (numeric, checkpoint_name.endswith("_shutdown"), checkpoint_name)


def find_ply_paths(result_root: Path):
    return sorted(
        result_root.glob("*/ply/point_cloud/iteration_*/point_cloud.ply"),
        key=iter_key,
    )


def reexport_checkpoint(binary: Path, cfg_path: Path, ply_path: Path, env, force: bool):
    checkpoint_dir = ply_path.parents[3]
    iteration_name = ply_path.parent.name
    tmp_root = checkpoint_dir / "ply/.compact_fixed_tmp"
    target_root = checkpoint_dir / "ply/compact_fixed"
    target_dir = target_root / iteration_name

    if target_dir.exists():
        if not force:
            return "skip", target_dir
        shutil.rmtree(target_dir)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    subprocess.run(
        [str(binary), str(cfg_path), str(ply_path), str(tmp_root)],
        check=True,
        cwd=ROOT,
        env=env,
    )

    produced_dir = tmp_root / "compact/iteration_0"
    if not produced_dir.exists():
        raise RuntimeError(f"re-export output missing: {produced_dir}")

    target_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced_dir), str(target_dir))
    shutil.rmtree(tmp_root)
    return "ok", target_dir


def main():
    parser = argparse.ArgumentParser(description="Re-export point_cloud.ply checkpoints into compact_fixed packages.")
    parser.add_argument("--job", action="append", type=parse_job, help="name|cfg_path|result_root")
    parser.add_argument("--binary", default=str(ROOT / "bin/export_compact"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jobs = args.job if args.job else DEFAULT_JOBS
    binary = Path(args.binary).expanduser().resolve()
    env = runtime_env()

    total_done = 0
    total_skipped = 0
    for job in jobs:
        ply_paths = find_ply_paths(job["result_root"])
        print(f"[reexport] {job['name']}: found {len(ply_paths)} checkpoints")
        for ply_path in ply_paths:
            status, out_dir = reexport_checkpoint(binary, job["cfg"], ply_path, env, args.force)
            print(f"[reexport] {status} {ply_path} -> {out_dir}")
            if status == "ok":
                total_done += 1
            else:
                total_skipped += 1

    print(f"[reexport] done={total_done} skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
