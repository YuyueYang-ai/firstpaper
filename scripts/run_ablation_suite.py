#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    env = dict(os.environ)
    ld_parts = [
        str(ROOT / "lib"),
        str(ROOT / ".deps/opencv-install/lib"),
        str(ROOT / ".deps/jsoncpp-install/lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join([part for part in ld_parts if part])

    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_formal_comparison.py"),
        "--manifest",
        str(ROOT / "scripts/ablation_experiment_manifest.json"),
        "--summary-manifest-output",
        str(ROOT / "results_smoke/ablation_manifest.json"),
        "--summary-output",
        str(ROOT / "results_smoke/ablation_summary.csv"),
        "--summary-markdown-output",
        str(ROOT / "results_smoke/ablation_summary.md"),
        "--compact-analysis-output",
        str(ROOT / "results_smoke/ablation_compact_breakdown.csv"),
        "--compact-analysis-markdown-output",
        str(ROOT / "results_smoke/ablation_compact_breakdown.md"),
    ]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
