"""Run one isolated validation stage and write a machine-readable report.

Heavy PyTorch stages are intentionally isolated. Some CPU/OpenMP runtimes do
not reliably support launching several independent training workloads from the
same long-lived supervisor process. CI therefore invokes this script once per
stage, which also makes failures attributable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(name: str, command: list[str], timeout: int, log_dir: Path) -> dict:
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    log = log_dir / f"{name}.log"
    try:
        with log.open("w", encoding="utf-8", errors="replace") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\nTIMEOUT\n")
        returncode = 124
        timed_out = True
    duration = time.perf_counter() - started
    output = log.read_text(encoding="utf-8", errors="replace")
    return {
        "name": name,
        "command": command,
        "returncode": returncode,
        "ok": returncode == 0,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "log": str(log.relative_to(ROOT)),
        "tail": output.splitlines()[-20:],
    }


def _commands(stage: str, timeout: int) -> list[tuple[str, list[str], int]]:
    if stage == "quick":
        return [
            ("compileall", [sys.executable, "-m", "compileall", "-q", "ultralytics", "etfnet_cli.py", "tools", "tests", "VALIDATION"], 120),
            ("pytest", [sys.executable, "-m", "pytest", "-q"], timeout),
            ("smoke", [sys.executable, "tests/smoke_pipeline.py"], timeout),
            ("model_configs", [sys.executable, "tools/validate_model_configs.py", "--output", "VALIDATION/model_configs_v3.json"], timeout),
        ]
    if stage == "exact-resume":
        return [("exact_resume", [sys.executable, "tests/exact_resume_check.py", "--output", "VALIDATION/exact_resume_v3.json"], timeout)]
    if stage == "proxy":
        return [(
            "realistic_proxy",
            [sys.executable, "VALIDATION/benchmark_realistic_proxy.py", "--seeds", "0", "1", "2", "3", "4", "--epochs", "8", "--threads", "1", "--output", "VALIDATION/realistic_proxy_5seed_v3.json"],
            timeout,
        )]
    if stage == "pipeline":
        return [("full_pipeline", ["bash", "tests/full_pipeline_smoke.sh"], timeout)]
    raise KeyError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("quick", "exact-resume", "proxy", "pipeline"), default="quick")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    output_path = args.output or ROOT / f"VALIDATION/extended_validation_{args.stage.replace('-', '_')}_v3.json"
    log_dir = ROOT / "VALIDATION/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = [_run(name, command, step_timeout, log_dir)
               for name, command, step_timeout in _commands(args.stage, args.timeout)]
    report = {
        "schema_version": 2,
        "ok": all(item["ok"] for item in results),
        "stage": args.stage,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "results": results,
        "source_hashes": {
            "block.py": _sha256(ROOT / "ultralytics/nn/modules/block.py"),
            "rgb_ir.py": _sha256(ROOT / "ultralytics/data/rgb_ir.py"),
            "rgb_ir_check.py": _sha256(ROOT / "ultralytics/data/rgb_ir_check.py"),
            "model_yaml": _sha256(ROOT / "ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
